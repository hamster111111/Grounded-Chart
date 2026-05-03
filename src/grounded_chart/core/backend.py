from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from grounded_chart.core.schema import FigureTrace, PlotTrace

SupportTier = Literal["native", "spec_accessible", "render_only", "unsupported"]
VerificationMode = Literal["hard", "partial", "soft", "none"]


@dataclass(frozen=True)
class BackendProfile:
    backend_name: str
    support_tier: SupportTier
    verification_mode: VerificationMode
    supported_chart_types: tuple[str, ...] = ()
    trace_capabilities: tuple[str, ...] = ()
    verification_capabilities: tuple[str, ...] = ()
    repair_capabilities: tuple[str, ...] = ()
    notes: str = ""

    @property
    def supports_hard_verification(self) -> bool:
        return self.verification_mode == "hard"


MATPLOTLIB_2D_PROFILE = BackendProfile(
    backend_name="matplotlib_2d",
    support_tier="native",
    verification_mode="hard",
    supported_chart_types=("bar", "line", "scatter", "pie"),
    trace_capabilities=("chart_type", "x_values", "y_values", "labels", "order"),
    verification_capabilities=("data_operation", "encoding", "order", "basic_annotation"),
    repair_capabilities=("local_patch", "data_transformation_patch", "structural_regeneration"),
    notes="Primary MVP backend.",
)

MATPLOTLIB_3D_PROFILE = BackendProfile(
    backend_name="matplotlib_3d",
    support_tier="native",
    verification_mode="partial",
    supported_chart_types=("scatter3d", "line3d", "bar3d"),
    trace_capabilities=("chart_type", "x_values", "y_values", "z_values", "labels"),
    verification_capabilities=("data_operation", "basic_3d_encoding"),
    repair_capabilities=("local_patch", "data_transformation_patch"),
    notes="3D charts are treated as partial verification until viewpoint and surface semantics are supported.",
)

UNKNOWN_BACKEND_PROFILE = BackendProfile(
    backend_name="unknown",
    support_tier="unsupported",
    verification_mode="none",
    notes="Unsupported backend should trigger abstention for hard fidelity claims.",
)

PLOTLY_PROFILE = BackendProfile(
    backend_name="plotly",
    support_tier="spec_accessible",
    verification_mode="soft",
    supported_chart_types=("sunburst", "treemap", "bar", "line", "scatter", "pie"),
    trace_capabilities=("figure_title", "trace_types", "trace_count", "layout_metadata"),
    verification_capabilities=("backend_detection", "minimal_structure", "title"),
    repair_capabilities=("structural_regeneration",),
    notes="Minimal backend support: detect Plotly execution and expose coarse structural metadata without hard data-fidelity claims.",
)


def infer_backend_profile(
    actual_trace: "PlotTrace | None" = None,
    actual_figure: "FigureTrace | None" = None,
    generated_code: str = "",
) -> BackendProfile:
    backend_name = infer_backend_name(actual_trace=actual_trace, actual_figure=actual_figure, generated_code=generated_code)
    if backend_name == PLOTLY_PROFILE.backend_name:
        return PLOTLY_PROFILE
    if backend_name == MATPLOTLIB_3D_PROFILE.backend_name:
        return MATPLOTLIB_3D_PROFILE
    if backend_name == MATPLOTLIB_2D_PROFILE.backend_name:
        return MATPLOTLIB_2D_PROFILE
    return UNKNOWN_BACKEND_PROFILE


def infer_backend_name(
    actual_trace: "PlotTrace | None" = None,
    actual_figure: "FigureTrace | None" = None,
    generated_code: str = "",
) -> str:
    trace_backend = _raw_backend(actual_trace.raw) if actual_trace is not None else None
    figure_backend = _raw_backend(actual_figure.raw) if actual_figure is not None else None
    if trace_backend == "plotly" or figure_backend == "plotly":
        return PLOTLY_PROFILE.backend_name
    if (actual_trace is not None and actual_trace.source == "plotly_figure") or (actual_figure is not None and actual_figure.source == "plotly_figure"):
        return PLOTLY_PROFILE.backend_name
    if actual_figure is not None and any(str(axis.projection).lower() == "plotly" for axis in actual_figure.axes):
        return PLOTLY_PROFILE.backend_name

    if actual_figure is not None and any(str(axis.projection).lower() == "3d" for axis in actual_figure.axes):
        return MATPLOTLIB_3D_PROFILE.backend_name
    if (actual_trace is not None and str(actual_trace.source).startswith("matplotlib")) or (
        actual_figure is not None and actual_figure.source == "matplotlib_figure"
    ):
        return MATPLOTLIB_2D_PROFILE.backend_name

    lowered = generated_code.lower()
    if any(token in lowered for token in ("plotly.express", "plotly.graph_objects", "import plotly", "pio.write_image", "px.", "go.")):
        return PLOTLY_PROFILE.backend_name
    if any(
        token in lowered
        for token in (
            "mpl_toolkits.mplot3d",
            "projection='3d'",
            'projection="3d"',
            ".bar3d(",
            ".voxels(",
            ".plot_surface(",
            ".tricontourf(",
        )
    ):
        return MATPLOTLIB_3D_PROFILE.backend_name
    if "matplotlib" in lowered or "plt." in lowered:
        return MATPLOTLIB_2D_PROFILE.backend_name
    return UNKNOWN_BACKEND_PROFILE.backend_name


def _raw_backend(raw: object) -> str | None:
    if isinstance(raw, dict):
        backend = raw.get("backend")
        if backend is not None:
            return str(backend)
    return None
