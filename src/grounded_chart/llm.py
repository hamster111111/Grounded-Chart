from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse


class LLMClient(Protocol):
    def complete_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """Return raw model text."""

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Return a parsed JSON object from model output."""


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    model: str
    api_key: str
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    no_proxy: bool = False

    @classmethod
    def from_env(cls, prefix: str = "GCHART") -> "OpenAICompatibleConfig":
        model = os.environ[f"{prefix}_MODEL"]
        api_key = os.environ[f"{prefix}_API_KEY"]
        base_url = os.environ.get(f"{prefix}_BASE_URL")
        temperature = float(os.environ.get(f"{prefix}_TEMPERATURE", "0"))
        max_tokens_raw = os.environ.get(f"{prefix}_MAX_TOKENS")
        max_tokens = int(max_tokens_raw) if max_tokens_raw else None
        no_proxy = os.environ.get(f"{prefix}_NO_PROXY", "").strip().lower() in {"1", "true", "yes", "on"}
        return cls(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            no_proxy=no_proxy,
        )


class OpenAICompatibleLLMClient:
    """Thin wrapper over an OpenAI-compatible chat completion API."""

    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self.config = config
        self._client = None

    def complete_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        client = self._ensure_client()
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature if temperature is not None else self.config.temperature,
        }
        resolved_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        if resolved_max_tokens is not None:
            kwargs["max_tokens"] = resolved_max_tokens
        response = client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        content = getattr(message, "content", "")
        if isinstance(content, list):
            return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
        return str(content or "")

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        text = self.complete_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return extract_json_object(text)

    def _ensure_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "OpenAI-compatible LLM support requires the optional dependency `openai`."
                ) from exc
            if self.config.no_proxy and self.config.base_url:
                _apply_no_proxy_for_base_url(self.config.base_url)
            kwargs = {"api_key": self.config.api_key}
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._client = OpenAI(**kwargs)
        return self._client


def extract_json_object(text: str) -> dict[str, Any]:
    normalized = str(text or "").strip()
    if normalized.startswith("```"):
        normalized = _strip_code_fence(normalized)
    try:
        loaded = json.loads(normalized)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass

    start = normalized.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(normalized)):
            char = normalized[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = normalized[start : index + 1]
                    try:
                        loaded = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(loaded, dict):
                        return loaded
        start = normalized.find("{", start + 1)
    raise ValueError(f"Model output did not contain a valid JSON object: {text!r}")


def _strip_code_fence(text: str) -> str:
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return text


def _apply_no_proxy_for_base_url(base_url: str) -> None:
    parsed = urlparse(base_url)
    host = parsed.hostname
    if not host:
        return
    merged = _merge_no_proxy_values(os.environ.get("NO_PROXY"), os.environ.get("no_proxy"), host)
    os.environ["NO_PROXY"] = merged
    os.environ["no_proxy"] = merged


def _merge_no_proxy_value(existing: str | None, host: str) -> str:
    normalized_host = host.strip()
    if not normalized_host:
        return existing or ""
    entries = [entry.strip() for entry in str(existing or "").split(",") if entry.strip()]
    if normalized_host not in entries:
        entries.append(normalized_host)
    return ",".join(entries)


def _merge_no_proxy_values(upper: str | None, lower: str | None, host: str) -> str:
    merged = str(upper or "")
    for entry in [item.strip() for item in str(lower or "").split(",") if item.strip()]:
        merged = _merge_no_proxy_value(merged, entry)
    return _merge_no_proxy_value(merged, host)
