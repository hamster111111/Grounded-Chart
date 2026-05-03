import unittest

from grounded_chart.api import (
    MATPLOTLIB_2D_PROFILE,
    MATPLOTLIB_3D_PROFILE,
    PLOTLY_PROFILE,
    AxisTrace,
    FigureTrace,
    PlotTrace,
    UNKNOWN_BACKEND_PROFILE,
    infer_backend_name,
    infer_backend_profile,
)


class BackendProfileTest(unittest.TestCase):
    def test_backend_support_tiers(self):
        self.assertTrue(MATPLOTLIB_2D_PROFILE.supports_hard_verification)
        self.assertFalse(MATPLOTLIB_3D_PROFILE.supports_hard_verification)
        self.assertEqual(PLOTLY_PROFILE.support_tier, "spec_accessible")
        self.assertEqual(PLOTLY_PROFILE.verification_mode, "soft")
        self.assertEqual(UNKNOWN_BACKEND_PROFILE.verification_mode, "none")

    def test_infers_plotly_backend_from_trace(self):
        plot_trace = PlotTrace(chart_type="pie", points=(), source="plotly_figure", raw={"backend": "plotly"})
        figure_trace = FigureTrace(
            title="Plotly Figure",
            axes=(AxisTrace(index=0, projection="plotly"),),
            source="plotly_figure",
            raw={"backend": "plotly"},
        )

        profile = infer_backend_profile(actual_trace=plot_trace, actual_figure=figure_trace)

        self.assertEqual(profile.backend_name, "plotly")
        self.assertEqual(infer_backend_name(actual_trace=plot_trace, actual_figure=figure_trace), "plotly")

    def test_infers_matplotlib_3d_backend_from_figure(self):
        figure_trace = FigureTrace(
            axes=(AxisTrace(index=0, projection="3d"),),
            source="matplotlib_figure",
        )

        profile = infer_backend_profile(actual_figure=figure_trace)

        self.assertEqual(profile.backend_name, "matplotlib_3d")

    def test_infers_backend_from_code_when_trace_missing(self):
        profile = infer_backend_profile(generated_code="import plotly.express as px\nfig = px.sunburst(...)")
        self.assertEqual(profile.backend_name, "plotly")


if __name__ == "__main__":
    unittest.main()
