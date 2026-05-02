from __future__ import annotations

import contextlib
import io
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

_VISIBLE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".html"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}


@dataclass(frozen=True)
class ChartRenderResult:
    """Result of executing plotting code for user-visible artifacts."""

    ok: bool
    image_path: Path | None = None
    artifact_paths: tuple[Path, ...] = ()
    backend: str = "unknown"
    stdout: str = ""
    stderr: str = ""
    exception_type: str | None = None
    exception_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ChartImageRenderer:
    """Execute generated chart code and export a visible figure artifact.

    This renderer is not a sandbox. It is intended for controlled benchmark and
    local research runs, mirroring the existing trace runner assumptions.
    """

    def render(
        self,
        code: str,
        *,
        rows: tuple[dict[str, Any], ...],
        output_dir: str | Path,
        output_filename: str = "figure.png",
        file_path: str | Path | None = None,
        globals_dict: dict[str, Any] | None = None,
    ) -> ChartRenderResult:
        output_root = Path(output_dir).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        output_path = (output_root / output_filename).resolve()
        _clear_stale_targets(output_path)
        before = _existing_artifacts(output_root)
        stdout = io.StringIO()
        stderr = io.StringIO()
        exec_globals: dict[str, Any] = {
            "__builtins__": __builtins__,
            "__name__": "__main__",
            "__file__": str(file_path) if file_path is not None else "<grounded_chart_render>",
            "rows": [dict(row) for row in rows],
            "OUTPUT_PATH": str(output_path),
        }
        if globals_dict:
            exec_globals.update(dict(globals_dict))

        try:
            with _pushd(output_root), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                _prepare_matplotlib()
                exec(code, exec_globals)
                backend = _infer_backend(exec_globals)
                image_path = _ensure_visible_artifact(
                    exec_globals,
                    output_root=output_root,
                    output_path=output_path,
                    before=before,
                )
                artifacts = _new_artifacts(output_root, before)
                if image_path is not None and image_path not in artifacts:
                    artifacts = (*artifacts, image_path)
                return ChartRenderResult(
                    ok=image_path is not None and image_path.exists(),
                    image_path=image_path,
                    artifact_paths=tuple(dict.fromkeys(artifacts)),
                    backend=backend,
                    stdout=stdout.getvalue(),
                    stderr=stderr.getvalue(),
                    metadata={"output_path": str(output_path)},
                )
        except Exception as exc:
            artifacts = _new_artifacts(output_root, before)
            image_path = output_path if output_path.exists() else None
            return ChartRenderResult(
                ok=False,
                image_path=image_path,
                artifact_paths=artifacts,
                backend=_infer_backend(exec_globals),
                stdout=stdout.getvalue(),
                stderr=stderr.getvalue(),
                exception_type=type(exc).__name__,
                exception_message=str(exc),
                metadata={"output_path": str(output_path)},
            )


def _prepare_matplotlib() -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        return


def _ensure_visible_artifact(
    exec_globals: dict[str, Any],
    *,
    output_root: Path,
    output_path: Path,
    before: set[Path],
) -> Path | None:
    if output_path.exists():
        return output_path

    discovered_image = _first_new_artifact(output_root, before, _IMAGE_SUFFIXES)
    if discovered_image is not None:
        if discovered_image.resolve() != output_path.resolve():
            shutil.copyfile(discovered_image, output_path)
        return output_path

    if output_path.suffix.lower() in _IMAGE_SUFFIXES and _save_matplotlib_current_figure(output_path):
        return output_path

    if output_path.suffix.lower() in _IMAGE_SUFFIXES and _save_plotly_figure(exec_globals, output_path):
        return output_path

    discovered_html = _first_new_artifact(output_root, before, {".html"})
    if discovered_html is not None:
        normalized_html = output_path.with_suffix(".html")
        if discovered_html.resolve() != normalized_html.resolve():
            shutil.copyfile(discovered_html, normalized_html)
        return normalized_html

    html_path = output_path if output_path.suffix.lower() == ".html" else output_path.with_suffix(".html")
    if _save_plotly_html(exec_globals, html_path):
        return html_path
    return None


def _save_matplotlib_current_figure(output_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt

        figure_numbers = plt.get_fignums()
        if not figure_numbers:
            return False
        figure = plt.figure(figure_numbers[-1])
        figure.savefig(output_path, bbox_inches="tight")
        return output_path.exists()
    except Exception:
        return False


def _save_plotly_figure(exec_globals: dict[str, Any], output_path: Path) -> bool:
    fig = _find_plotly_figure(exec_globals)
    if fig is None or not hasattr(fig, "write_image"):
        return False
    try:
        fig.write_image(str(output_path))
        return output_path.exists()
    except Exception:
        return False


def _save_plotly_html(exec_globals: dict[str, Any], output_path: Path) -> bool:
    fig = _find_plotly_figure(exec_globals)
    if fig is None or not hasattr(fig, "write_html"):
        return False
    try:
        fig.write_html(str(output_path))
        return output_path.exists()
    except Exception:
        return False


def _find_plotly_figure(exec_globals: dict[str, Any]) -> Any | None:
    fig = exec_globals.get("fig")
    if fig is not None and _looks_like_plotly_figure(fig):
        return fig
    for value in exec_globals.values():
        if _looks_like_plotly_figure(value):
            return value
    return None


def _looks_like_plotly_figure(value: Any) -> bool:
    module = value.__class__.__module__ if value is not None else ""
    return module.startswith("plotly") and (hasattr(value, "write_html") or hasattr(value, "write_image"))


def _infer_backend(exec_globals: dict[str, Any]) -> str:
    if _find_plotly_figure(exec_globals) is not None:
        return "plotly"
    for value in exec_globals.values():
        module = getattr(value, "__name__", None) or getattr(value.__class__, "__module__", "")
        if isinstance(module, str) and module.startswith("matplotlib"):
            return "matplotlib"
    try:
        import matplotlib.pyplot as plt

        if plt.get_fignums():
            return "matplotlib"
    except Exception:
        pass
    return "unknown"


def _existing_artifacts(output_root: Path) -> set[Path]:
    return {path.resolve() for path in output_root.glob("*") if path.is_file() and path.suffix.lower() in _VISIBLE_SUFFIXES}


def _new_artifacts(output_root: Path, before: set[Path]) -> tuple[Path, ...]:
    artifacts = [
        path
        for path in output_root.glob("*")
        if path.is_file() and path.suffix.lower() in _VISIBLE_SUFFIXES and path.resolve() not in before
    ]
    return tuple(sorted(artifacts, key=lambda item: str(item)))


def _first_new_artifact(output_root: Path, before: set[Path], suffixes: set[str]) -> Path | None:
    candidates = [
        path
        for path in output_root.glob("*")
        if path.is_file() and path.suffix.lower() in suffixes and path.resolve() not in before
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item.suffix.lower() != ".png", str(item)))
    return candidates[0]


def _clear_stale_targets(output_path: Path) -> None:
    for path in (output_path, output_path.with_suffix(".html")):
        try:
            if path.exists() and path.is_file():
                path.unlink()
        except OSError:
            pass


@contextlib.contextmanager
def _pushd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)



