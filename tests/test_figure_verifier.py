import unittest

from grounded_chart import ArtistTrace, AxisRequirementSpec, DataPoint, FigureRequirementSpec, FigureTrace, MatplotlibTraceRunner, PlotTrace
from grounded_chart.schema import AxisTrace
from grounded_chart.verifier import OperatorLevelVerifier


class FigureVerifierTest(unittest.TestCase):
    def test_passes_matching_figure_requirements(self):
        code = """
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(4, 3))
fig.suptitle("Revenue Dashboard")
ax.bar(["A", "B"], [1, 2], label="Sales")
ax.set_title("Sales by Category")
ax.set_xlabel("Category")
ax.set_ylabel("Sales")
ax.text(0, 1, "A")
ax.legend()
"""
        run_trace = MatplotlibTraceRunner().run_code_with_figure(code)
        expected_trace = PlotTrace("bar", (DataPoint("A", 1), DataPoint("B", 2)), source="expected")
        requirements = FigureRequirementSpec(
            axes_count=1,
            figure_title="Revenue Dashboard",
            size_inches=(4, 3),
            axes=(
                AxisRequirementSpec(
                    title="Sales by Category",
                    xlabel="Category",
                    ylabel="Sales",
                    legend_labels=("Sales",),
                    artist_types=("patch",),
                    text_contains=("A",),
                ),
            ),
        )

        report = OperatorLevelVerifier().verify(
            expected_trace,
            run_trace.plot_trace,
            expected_figure=requirements,
            actual_figure=run_trace.figure_trace,
        )

        self.assertTrue(report.ok)

    def test_detects_figure_requirement_errors(self):
        code = """
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.bar(["A", "B"], [1, 2])
ax.set_title("Wrong")
ax.set_xlabel("Category")
"""
        run_trace = MatplotlibTraceRunner().run_code_with_figure(code)
        expected_trace = PlotTrace("bar", (DataPoint("A", 1), DataPoint("B", 2)), source="expected")
        requirements = FigureRequirementSpec(
            axes_count=2,
            figure_title="Expected Figure",
            axes=(
                AxisRequirementSpec(
                    title="Sales by Category",
                    xlabel="Category",
                    ylabel="Sales",
                    projection="polar",
                    legend_labels=("Sales",),
                    text_contains=("peak",),
                ),
            ),
        )

        report = OperatorLevelVerifier().verify(
            expected_trace,
            run_trace.plot_trace,
            expected_figure=requirements,
            actual_figure=run_trace.figure_trace,
        )

        self.assertFalse(report.ok)
        self.assertIn("wrong_axes_count", report.error_codes)
        self.assertIn("wrong_figure_title", report.error_codes)
        self.assertIn("wrong_axis_title", report.error_codes)
        self.assertIn("wrong_y_label", report.error_codes)
        self.assertIn("wrong_projection", report.error_codes)
        self.assertIn("missing_legend_label", report.error_codes)
        self.assertIn("missing_annotation_text", report.error_codes)

    def test_figure_only_mode_ignores_data_mismatch(self):
        code = """
import matplotlib.pyplot as plt
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.bar(['A', 'B'], [1, 2], label='Sales')
ax.set_title('Sales by Category')
ax.legend()
"""
        run_trace = MatplotlibTraceRunner().run_code_with_figure(code)
        expected_trace = PlotTrace("bar", (DataPoint("X", 99),), source="expected")
        requirements = FigureRequirementSpec(
            axes_count=1,
            axes=(
                AxisRequirementSpec(
                    title="Sales by Category",
                    projection="polar",
                    legend_labels=("Sales",),
                ),
            ),
        )

        report = OperatorLevelVerifier().verify(
            expected_trace,
            run_trace.plot_trace,
            expected_figure=requirements,
            actual_figure=run_trace.figure_trace,
            verify_data=False,
        )

        self.assertTrue(report.ok)

    def test_detects_insufficient_artist_counts(self):
        code = """
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.bar(["A", "B"], [1, 2])
"""
        run_trace = MatplotlibTraceRunner().run_code_with_figure(code)
        expected_trace = PlotTrace("bar", (DataPoint("A", 1), DataPoint("B", 2)), source="expected")
        requirements = FigureRequirementSpec(
            axes_count=1,
            axes=(
                AxisRequirementSpec(
                    artist_types=("patch",),
                    min_artist_counts={"patch": 3},
                ),
            ),
        )

        report = OperatorLevelVerifier().verify(
            expected_trace,
            run_trace.plot_trace,
            expected_figure=requirements,
            actual_figure=run_trace.figure_trace,
        )

        self.assertFalse(report.ok)
        self.assertIn("insufficient_artist_count", report.error_codes)

    def test_detects_wrong_artist_counts(self):
        code = """
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.bar(["A", "B"], [1, 2])
"""
        run_trace = MatplotlibTraceRunner().run_code_with_figure(code)
        expected_trace = PlotTrace("bar", (DataPoint("A", 1), DataPoint("B", 2)), source="expected")
        requirements = FigureRequirementSpec(
            axes_count=1,
            axes=(
                AxisRequirementSpec(
                    artist_counts={"patch": 1},
                ),
            ),
        )

        report = OperatorLevelVerifier().verify(
            expected_trace,
            run_trace.plot_trace,
            expected_figure=requirements,
            actual_figure=run_trace.figure_trace,
        )

        self.assertFalse(report.ok)
        self.assertIn("wrong_artist_count", report.error_codes)

    def test_hist2d_requirement_accepts_image_artist(self):
        expected_trace = PlotTrace("unknown", (), source="expected")
        requirements = FigureRequirementSpec(
            axes_count=3,
            axes=(
                AxisRequirementSpec(axis_index=1, artist_types=("hist2d",)),
                AxisRequirementSpec(axis_index=2, artist_types=("hist2d",)),
            ),
        )
        actual_figure = FigureTrace(
            axes=(
                AxisTrace(index=0, artists=(ArtistTrace(artist_type="line", count=1),)),
                AxisTrace(index=1, artists=(ArtistTrace(artist_type="image", count=1),)),
                AxisTrace(index=2, artists=(ArtistTrace(artist_type="image", count=1),)),
            ),
            source="matplotlib_figure",
        )

        report = OperatorLevelVerifier().verify(
            expected_trace,
            PlotTrace("unknown", (), source="matplotlib_figure"),
            expected_figure=requirements,
            actual_figure=actual_figure,
            verify_data=False,
        )

        self.assertTrue(report.ok)

    def test_plotly_annotation_text_can_satisfy_figure_requirement(self):
        expected_trace = PlotTrace("pie", (), source="expected")
        requirements = FigureRequirementSpec(
            axes_count=1,
            axes=(
                AxisRequirementSpec(
                    title="Global Food Security Index, 2020",
                    projection="plotly",
                    text_contains=("Overall score 0-100, 100 = best environment",),
                ),
            ),
        )
        actual_figure = FigureTrace(
            title="Global Food Security Index, 2020",
            axes=(
                AxisTrace(
                    index=0,
                    title="Global Food Security Index, 2020",
                    projection="plotly",
                    texts=("Overall score 0-100, 100 = best environment",),
                ),
            ),
            source="plotly_figure",
        )

        report = OperatorLevelVerifier().verify(
            expected_trace,
            PlotTrace("pie", (), source="plotly_figure"),
            expected_figure=requirements,
            actual_figure=actual_figure,
            verify_data=False,
        )

        self.assertTrue(report.ok)

    def test_plotly_axis_and_legend_can_satisfy_figure_requirement(self):
        expected_trace = PlotTrace("bar", (), source="expected")
        requirements = FigureRequirementSpec(
            axes_count=1,
            axes=(
                AxisRequirementSpec(
                    title="Plotly Bar",
                    xlabel="Category",
                    ylabel="Sales",
                    projection="plotly",
                    legend_labels=("North", "South"),
                ),
            ),
        )
        actual_figure = FigureTrace(
            title="Plotly Bar",
            axes=(
                AxisTrace(
                    index=0,
                    title="Plotly Bar",
                    xlabel="Category",
                    ylabel="Sales",
                    projection="plotly",
                    legend_labels=("North", "South"),
                ),
            ),
            source="plotly_figure",
        )

        report = OperatorLevelVerifier().verify(
            expected_trace,
            PlotTrace("bar", (), source="plotly_figure"),
            expected_figure=requirements,
            actual_figure=actual_figure,
            verify_data=False,
        )

        self.assertTrue(report.ok)

    def test_figure_title_requirement_can_fallback_to_single_axis_title(self):
        expected_trace = PlotTrace("scatter", (), source="expected")
        requirements = FigureRequirementSpec(
            axes_count=1,
            figure_title="A colored bubble plot",
        )
        actual_figure = FigureTrace(
            title="",
            axes=(
                AxisTrace(
                    index=0,
                    title="A colored bubble plot",
                    xlabel="X Variable",
                    ylabel="Y Variable",
                ),
            ),
            source="matplotlib_figure",
        )

        report = OperatorLevelVerifier().verify(
            expected_trace,
            PlotTrace("scatter", (), source="matplotlib_figure"),
            expected_figure=requirements,
            actual_figure=actual_figure,
            verify_data=False,
        )

        self.assertTrue(report.ok)

    def test_figure_title_requirement_can_fallback_when_only_one_axis_has_title(self):
        expected_trace = PlotTrace("scatter", (), source="expected")
        requirements = FigureRequirementSpec(
            figure_title="A colored bubble plot",
        )
        actual_figure = FigureTrace(
            title="",
            axes=(
                AxisTrace(
                    index=0,
                    title="A colored bubble plot",
                    xlabel="X Variable",
                    ylabel="Y Variable",
                ),
                AxisTrace(
                    index=1,
                    title="",
                    xlabel="",
                    ylabel="Color",
                ),
            ),
            source="matplotlib_figure",
        )

        report = OperatorLevelVerifier().verify(
            expected_trace,
            PlotTrace("scatter", (), source="matplotlib_figure"),
            expected_figure=requirements,
            actual_figure=actual_figure,
            verify_data=False,
        )

        self.assertTrue(report.ok)

    def test_runs_main_guarded_matplotlib_scripts(self):
        code = """
import matplotlib.pyplot as plt

def main():
    fig, ax = plt.subplots()
    ax.set_title("Main Guard Ran")
    ax.plot([1, 2], [3, 4])

if __name__ == "__main__":
    main()
"""
        run_trace = MatplotlibTraceRunner().run_code_with_figure(code)

        self.assertEqual(1, run_trace.figure_trace.axes_count)
        self.assertEqual("Main Guard Ran", run_trace.figure_trace.axes[0].title)


if __name__ == "__main__":
    unittest.main()
