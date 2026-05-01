import os
import tempfile
import unittest
from pathlib import Path

from grounded_chart.llm import OpenAICompatibleConfig, OpenAICompatibleLLMClient, _apply_no_proxy_for_base_url, _merge_no_proxy_value, _merge_no_proxy_values


class NoProxyHelpersTest(unittest.TestCase):
    def test_merge_no_proxy_value_appends_host_once(self):
        merged = _merge_no_proxy_value("localhost,127.0.0.1", "api.deepseek.com")
        self.assertEqual("localhost,127.0.0.1,api.deepseek.com", merged)

        merged_again = _merge_no_proxy_value(merged, "api.deepseek.com")
        self.assertEqual("localhost,127.0.0.1,api.deepseek.com", merged_again)

    def test_apply_no_proxy_for_base_url_updates_both_env_keys(self):
        old_upper = os.environ.get("NO_PROXY")
        old_lower = os.environ.get("no_proxy")
        os.environ["NO_PROXY"] = "localhost,127.0.0.1"
        try:
            _apply_no_proxy_for_base_url("https://api.deepseek.com/v1")
            self.assertEqual("localhost,127.0.0.1,api.deepseek.com", os.environ["NO_PROXY"])
            self.assertEqual("localhost,127.0.0.1,api.deepseek.com", os.environ["no_proxy"])
        finally:
            if old_upper is None:
                os.environ.pop("NO_PROXY", None)
            else:
                os.environ["NO_PROXY"] = old_upper
            if old_lower is None:
                os.environ.pop("no_proxy", None)
            else:
                os.environ["no_proxy"] = old_lower

    def test_merge_no_proxy_values_combines_upper_and_lower(self):
        merged = _merge_no_proxy_values("localhost", "127.0.0.1", "api.deepseek.com")
        self.assertEqual("localhost,127.0.0.1,api.deepseek.com", merged)


class ImageCompletionTest(unittest.TestCase):
    def test_openai_compatible_client_sends_image_url_content(self):
        class FakeCompletions:
            def __init__(self):
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs

                class Message:
                    content = '{"ok": true}'

                class Choice:
                    message = Message()

                class Response:
                    choices = [Choice()]
                    model = "fake-vlm"
                    usage = None

                return Response()

        class FakeChat:
            def __init__(self):
                self.completions = FakeCompletions()

        class FakeClient:
            def __init__(self):
                self.chat = FakeChat()

        client = OpenAICompatibleLLMClient(OpenAICompatibleConfig(model="fake-vlm", api_key="sk-test"))
        fake_client = FakeClient()
        client._client = fake_client

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "figure.png"
            image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
            result = client.complete_json_with_image_trace(
                system_prompt="system",
                user_prompt="user",
                image_path=image_path,
            )

        messages = fake_client.chat.completions.kwargs["messages"]
        content = messages[1]["content"]
        self.assertTrue(result.payload["ok"])
        self.assertEqual("text", content[0]["type"])
        self.assertEqual("image_url", content[1]["type"])
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
