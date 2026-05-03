import unittest

from grounded_chart.api import AxisTrace, DataPoint, FigureTrace, LLMExpectedArtifactExtractor, OperatorLevelVerifier, PlotTrace, extract_expected_trace_from_text
from grounded_chart.data.expected_artifacts import extract_expected_trace_from_texts
from grounded_chart.runtime.llm import LLMCompletionTrace, LLMJsonResult




class _FakeJsonClient:
    def __init__(self, payload):
        self.payload = payload
        self.last_system_prompt = ""
        self.last_user_prompt = ""

    def complete_json_with_trace(self, **kwargs):
        self.last_system_prompt = kwargs.get("system_prompt", "")
        self.last_user_prompt = kwargs.get("user_prompt", "")
        return LLMJsonResult(
            payload=dict(self.payload),
            trace=LLMCompletionTrace(raw_text="{}", parsed_json=dict(self.payload)),
        )

class ExpectedArtifactExtractionTest(unittest.TestCase):
    def test_extracts_matplotbench_explicit_bar_points(self):
        text = (
            "We'll call the first function with the figure and a rectangle, and create a bar plot "
            "on the auxiliary axes. Let's use the numbers 1, 2, 3, and 4 for the x-values, "
            "and the numbers 4, 3, 2, and 4 for the y-values."
        )

        trace = extract_expected_trace_from_text(text, source="expert_instruction")

        self.assertIsNotNone(trace)
        self.assertEqual("bar", trace.chart_type)
        self.assertEqual(
            (DataPoint(1, 4), DataPoint(2, 3), DataPoint(3, 2), DataPoint(4, 4)),
            trace.points,
        )
        self.assertEqual("expert_instruction:explicit_point_sequence_v1", trace.source)
        self.assertIn("matched_text", trace.raw)

    def test_extracts_label_first_x_y_points(self):
        text = "The instruction specifies bar x-values 1,2,3,4 with y-values 4,3,2,4."

        trace = extract_expected_trace_from_text(text)

        self.assertIsNotNone(trace)
        self.assertEqual("bar", trace.chart_type)
        self.assertEqual((DataPoint(1, 4), DataPoint(2, 3), DataPoint(3, 2), DataPoint(4, 4)), trace.points)

    def test_rejects_ambiguous_random_values_without_explicit_xy_pair(self):
        text = "Use 20 random theta values between 0 and half pi and 20 random radius values between 1 and 2."

        self.assertIsNone(extract_expected_trace_from_text(text))

    def test_uses_ordered_source_texts(self):
        trace = extract_expected_trace_from_texts(
            (
                ("simple_instruction", "Create a bar plot with specific x and y values."),
                ("expert_instruction", "Create a bar plot with x-values 1, 2 and y-values 3, 4."),
            )
        )

        self.assertIsNotNone(trace)
        self.assertEqual((DataPoint(1, 3), DataPoint(2, 4)), trace.points)
        self.assertTrue(trace.source.startswith("expert_instruction:"))


    def test_llm_extractor_accepts_source_grounded_plot_points(self):
        text = "Create a bar plot using x-values 1, 2 and y-values 4, 3."
        client = _FakeJsonClient(
            {
                "artifacts": [
                    {
                        "artifact_type": "plot_points",
                        "panel_id": "panel_0",
                        "chart_type": "bar",
                        "value": {"role": "main_data", "points": [{"x": 1, "y": 4}, {"x": 2, "y": 3}]},
                        "source_span": "x-values 1, 2 and y-values 4, 3",
                        "confidence": 0.91,
                    }
                ]
            }
        )

        result = LLMExpectedArtifactExtractor(client).extract(text, source="expert_instruction", case_id="case-1")

        self.assertEqual(1, len(result.artifacts))
        self.assertEqual(1, len(result.plot_traces))
        self.assertEqual("bar", result.primary_trace.chart_type)
        self.assertEqual((DataPoint(1, 4, {"source_span": "x-values 1, 2 and y-values 4, 3", "confidence": 0.91, "role": "main_data", "panel_id": "panel_0"}), DataPoint(2, 3, {"source_span": "x-values 1, 2 and y-values 4, 3", "confidence": 0.91, "role": "main_data", "panel_id": "panel_0"})), result.primary_trace.points)
        self.assertIn("source_span", result.primary_trace.raw)
        self.assertIn("Return JSON only", client.last_user_prompt)

    def test_llm_annotation_marker_points_do_not_become_primary_trace(self):
        text = "Plot a scatter point on the center with color 'red' and size 3."
        client = _FakeJsonClient(
            {
                "artifacts": [
                    {
                        "artifact_type": "plot_points",
                        "panel_id": "panel_0",
                        "chart_type": "scatter",
                        "value": {"role": "annotation_marker", "points": [{"x": 1, "y": 1}]},
                        "source_span": "Plot a scatter point on the center with color 'red' and size 3",
                        "confidence": 0.9,
                    }
                ]
            }
        )

        result = LLMExpectedArtifactExtractor(client).extract(text)

        self.assertEqual(1, len(result.artifacts))
        self.assertEqual("annotation_marker", result.artifacts[0].role)
        self.assertEqual("annotation_marker", result.artifacts[0].value["role"])
        self.assertEqual((), result.plot_traces)

    def test_llm_extractor_rejects_plot_points_without_concrete_coordinates(self):
        text = "Plot a scatter point on the center with color 'red' and size 3."
        client = _FakeJsonClient(
            {
                "artifacts": [
                    {
                        "artifact_type": "plot_points",
                        "chart_type": "scatter",
                        "value": {"role": "annotation_marker", "points": [{"x": None, "y": None}]},
                        "source_span": "Plot a scatter point on the center with color 'red' and size 3",
                        "confidence": 0.9,
                    }
                ]
            }
        )

        result = LLMExpectedArtifactExtractor(client).extract(text)

        self.assertEqual((), result.artifacts)
        self.assertEqual((), result.plot_traces)

    def test_llm_plot_points_without_role_are_not_primary_trace(self):
        text = "Create a bar plot using x-values 1, 2 and y-values 4, 3."
        client = _FakeJsonClient(
            {
                "artifacts": [
                    {
                        "artifact_type": "plot_points",
                        "chart_type": "bar",
                        "value": {"points": [{"x": 1, "y": 4}, {"x": 2, "y": 3}]},
                        "source_span": "x-values 1, 2 and y-values 4, 3",
                        "confidence": 0.91,
                    }
                ]
            }
        )

        result = LLMExpectedArtifactExtractor(client).extract(text)

        self.assertEqual(1, len(result.artifacts))
        self.assertEqual("unknown", result.artifacts[0].role)
        self.assertEqual((), result.plot_traces)

    def test_llm_extractor_rejects_unverifiable_source_span(self):
        text = "Create a bar plot."
        client = _FakeJsonClient(
            {
                "artifacts": [
                    {
                        "artifact_type": "plot_points",
                        "chart_type": "bar",
                        "value": {"points": [{"x": 1, "y": 4}]},
                        "source_span": "x-values 1 and y-values 4",
                        "confidence": 0.99,
                    }
                ]
            }
        )

        result = LLMExpectedArtifactExtractor(client).extract(text)

        self.assertEqual((), result.artifacts)
        self.assertEqual((), result.plot_traces)

    def test_llm_extractor_keeps_non_trace_artifacts_without_plot_trace(self):
        text = "The plot should not display any radial gridlines and should have a title Electron Transitions."
        client = _FakeJsonClient(
            {
                "artifacts": [
                    {
                        "artifact_type": "title_requirement",
                        "value": {"title": "Electron Transitions"},
                        "source_span": "title Electron Transitions",
                        "confidence": 0.8,
                    },
                    {
                        "artifact_type": "axis_requirement",
                        "value": {"radial_gridlines": False},
                        "source_span": "not display any radial gridlines",
                        "confidence": 0.8,
                    },
                ]
            }
        )

        result = LLMExpectedArtifactExtractor(client).extract(text)

        self.assertEqual(2, len(result.artifacts))
        self.assertEqual((), result.plot_traces)
        self.assertEqual("title_requirement", result.artifacts[0].artifact_type)
        self.assertEqual("axis_requirement", result.artifacts[1].artifact_type)

    def test_llm_extractor_compiles_non_trace_artifacts_to_figure_requirements(self):
        title = "Uniform transparency value for all bars and edges"
        text = (
            "Create a visual space with two sections arranged side by side and set the size of this visual space to 8x4. "
            f"Set the title of this section to {title}. "
            "Use legend labels North and South. "
            "Label the x-axis as Year and use a log y scale. "
            "Add annotation text Overall score 0-100."
        )
        client = _FakeJsonClient(
            {
                "artifacts": [
                    {
                        "artifact_type": "subplot_layout",
                        "value": {"rows": 1, "columns": 2, "figure_size": [8, 4]},
                        "source_span": "two sections arranged side by side and set the size of this visual space to 8x4",
                        "confidence": 0.9,
                    },
                    {
                        "artifact_type": "title_requirement",
                        "value": {"text": title},
                        "source_span": f"Set the title of this section to {title}",
                        "confidence": 0.9,
                    },
                    {
                        "artifact_type": "legend_requirement",
                        "value": {"labels": ["North", "South"]},
                        "source_span": "Use legend labels North and South",
                        "confidence": 0.9,
                    },
                    {
                        "artifact_type": "axis_requirement",
                        "value": {"xlabel": "Year", "yscale": "log"},
                        "source_span": "Label the x-axis as Year and use a log y scale",
                        "confidence": 0.9,
                    },
                    {
                        "artifact_type": "annotation_requirement",
                        "value": {"text": "Overall score 0-100"},
                        "source_span": "Add annotation text Overall score 0-100",
                        "confidence": 0.9,
                    },
                ]
            }
        )

        result = LLMExpectedArtifactExtractor(client).extract(text)

        figure = result.figure_requirements
        self.assertIsNotNone(figure)
        self.assertEqual(2, figure.axes_count)
        self.assertEqual((8.0, 4.0), figure.size_inches)
        self.assertEqual("two sections arranged side by side and set the size of this visual space to 8x4", figure.source_spans["axes_count"])
        self.assertEqual(1, len(figure.axes))
        axis = figure.axes[0]
        self.assertEqual(title, axis.title)
        self.assertEqual("Year", axis.xlabel)
        self.assertEqual("log", axis.yscale)
        self.assertEqual(("North", "South"), axis.legend_labels)
        self.assertEqual(("Overall score 0-100",), axis.text_contains)
        self.assertEqual(("panel_0.axis_0.legend_labels",), axis.provenance["legend_label:South"])
        self.assertEqual("Use legend labels North and South", axis.source_spans["legend_labels"])

    def test_compiled_non_trace_artifacts_drive_figure_verifier(self):
        title = "Uniform transparency value for all bars and edges"
        text = (
            "Create a visual space with two sections arranged side by side and set the size of this visual space to 8x4. "
            f"Set the title of this section to {title}. "
            "Use legend labels North and South. "
            "Label the x-axis as Year and use a log y scale. "
            "Add annotation text Overall score 0-100."
        )
        client = _FakeJsonClient(
            {
                "artifacts": [
                    {
                        "artifact_type": "subplot_layout",
                        "value": {"rows": 1, "columns": 2, "figure_size": [8, 4]},
                        "source_span": "two sections arranged side by side and set the size of this visual space to 8x4",
                        "confidence": 0.9,
                    },
                    {
                        "artifact_type": "title_requirement",
                        "value": {"text": title},
                        "source_span": f"Set the title of this section to {title}",
                        "confidence": 0.9,
                    },
                    {
                        "artifact_type": "legend_requirement",
                        "value": {"labels": ["North", "South"]},
                        "source_span": "Use legend labels North and South",
                        "confidence": 0.9,
                    },
                    {
                        "artifact_type": "axis_requirement",
                        "value": {"xlabel": "Year", "yscale": "log"},
                        "source_span": "Label the x-axis as Year and use a log y scale",
                        "confidence": 0.9,
                    },
                    {
                        "artifact_type": "annotation_requirement",
                        "value": {"text": "Overall score 0-100"},
                        "source_span": "Add annotation text Overall score 0-100",
                        "confidence": 0.9,
                    },
                ]
            }
        )
        figure = LLMExpectedArtifactExtractor(client).extract(text).figure_requirements
        actual_figure = FigureTrace(
            size_inches=(8.0, 4.0),
            axes=(
                AxisTrace(
                    index=0,
                    title=title,
                    xlabel="Year",
                    yscale="linear",
                    legend_labels=("North",),
                    texts=(),
                ),
                AxisTrace(index=1),
            ),
        )

        report = OperatorLevelVerifier().verify(
            PlotTrace("bar", (), source="expected"),
            PlotTrace("bar", (), source="actual"),
            expected_figure=figure,
            actual_figure=actual_figure,
            verify_data=False,
        )

        error_codes = {error.code for error in report.errors}
        self.assertIn("wrong_y_scale", error_codes)
        self.assertIn("missing_legend_label", error_codes)
        self.assertIn("missing_annotation_text", error_codes)
        errors_by_code = {error.code: error for error in report.errors}
        self.assertEqual("panel_0.axis_0.yscale", errors_by_code["wrong_y_scale"].requirement_id)
        self.assertEqual("panel_0.axis_0.legend_labels", errors_by_code["missing_legend_label"].requirement_id)
        self.assertEqual("panel_0.axis_0.text_contains", errors_by_code["missing_annotation_text"].requirement_id)

    def test_llm_non_verifiable_artifacts_do_not_compile_to_figure_requirements(self):
        text = "Use a CSV file named data.csv and compute values with a custom formula."
        client = _FakeJsonClient(
            {
                "artifacts": [
                    {
                        "artifact_type": "data_file_constraint",
                        "value": {"file": "data.csv"},
                        "source_span": "CSV file named data.csv",
                        "confidence": 0.9,
                    },
                    {
                        "artifact_type": "data_generation_constraint",
                        "value": {"formula": "custom formula"},
                        "source_span": "compute values with a custom formula",
                        "confidence": 0.9,
                    },
                ]
            }
        )

        result = LLMExpectedArtifactExtractor(client).extract(text)

        self.assertEqual(2, len(result.artifacts))
        self.assertIsNone(result.figure_requirements)

    def test_llm_extractor_rejects_low_confidence_artifact(self):
        text = "Use a CSV file named data.csv."
        client = _FakeJsonClient(
            {
                "artifacts": [
                    {
                        "artifact_type": "data_file_constraint",
                        "value": {"file": "data.csv"},
                        "source_span": "CSV file named data.csv",
                        "confidence": 0.3,
                    }
                ]
            }
        )

        result = LLMExpectedArtifactExtractor(client).extract(text)

        self.assertEqual((), result.artifacts)

    def test_compiler_distinguishes_figure_size_from_subplot_grid(self):
        text = "Create a 3D plot with a figure size of 6x5 using add a subplot."
        client = _FakeJsonClient(
            {
                "artifacts": [
                    {
                        "artifact_type": "subplot_layout",
                        "value": {"rows": 6, "columns": 5, "figure_size": [6, 5]},
                        "source_span": "Create a 3D plot with a figure size of 6x5 using add a subplot",
                        "confidence": 0.9,
                    }
                ]
            }
        )

        figure = LLMExpectedArtifactExtractor(client).extract(text).figure_requirements

        self.assertIsNotNone(figure)
        self.assertEqual((6.0, 5.0), figure.size_inches)
        self.assertIsNone(figure.axes_count)

    def test_compiler_distinguishes_subplot_grid_from_figure_size(self):
        text = "Create a figure with 4 subplots arranged in a 2x2 grid."
        client = _FakeJsonClient(
            {
                "artifacts": [
                    {
                        "artifact_type": "subplot_layout",
                        "value": {"rows": 2, "columns": 2, "figure_size": [2, 2]},
                        "source_span": "Create a figure with 4 subplots arranged in a 2x2 grid",
                        "confidence": 0.9,
                    }
                ]
            }
        )

        figure = LLMExpectedArtifactExtractor(client).extract(text).figure_requirements

        self.assertIsNotNone(figure)
        self.assertEqual(4, figure.axes_count)
        self.assertIsNone(figure.size_inches)

    def test_compiler_counts_grid_plus_spanning_bottom_plot(self):
        text = "Create a subplot mosaic with a layout of 2x2 bar plots at the top two rows and a larger plot spanning the entire bottom row for a cosine curve."
        client = _FakeJsonClient(
            {
                "artifacts": [
                    {
                        "artifact_type": "subplot_layout",
                        "value": {},
                        "source_span": text,
                        "confidence": 0.9,
                    }
                ]
            }
        )

        figure = LLMExpectedArtifactExtractor(client).extract(text).figure_requirements

        self.assertIsNotNone(figure)
        self.assertEqual(5, figure.axes_count)

    def test_compiler_rejects_title_text_not_in_source_span(self):
        text = "Set the title of the chart."
        client = _FakeJsonClient(
            {
                "artifacts": [
                    {
                        "artifact_type": "title_requirement",
                        "value": {"text": "Distribution of fruits among age groups"},
                        "source_span": "Set the title of the chart",
                        "confidence": 0.9,
                    }
                ]
            }
        )

        result = LLMExpectedArtifactExtractor(client).extract(text)

        self.assertIsNone(result.figure_requirements)

    def test_compiler_rejects_title_without_title_cue(self):
        text = "Create the 17 segment model using the bullseye plot function."
        client = _FakeJsonClient(
            {
                "artifacts": [
                    {
                        "artifact_type": "title_requirement",
                        "value": {"text": "17 segment model"},
                        "source_span": "Create the 17 segment model using the bullseye plot function",
                        "confidence": 0.9,
                    }
                ]
            }
        )

        result = LLMExpectedArtifactExtractor(client).extract(text)

        self.assertIsNone(result.figure_requirements)

    def test_compiler_normalizes_natural_language_axis_index(self):
        text = "Set the title of the fourth subplot to Revenue."
        client = _FakeJsonClient(
            {
                "artifacts": [
                    {
                        "artifact_type": "title_requirement",
                        "value": {"text": "Revenue", "axis_index": 4},
                        "source_span": "Set the title of the fourth subplot to Revenue",
                        "confidence": 0.9,
                    }
                ]
            }
        )

        figure = LLMExpectedArtifactExtractor(client).extract(text).figure_requirements

        self.assertIsNotNone(figure)
        self.assertEqual(3, figure.axes[0].axis_index)
        self.assertEqual("Revenue", figure.axes[0].title)
    def test_compiler_shifts_one_based_panel_ids_when_axes_count_is_known(self):
        text = (
            "Create a figure with 4 subplots arranged in a 2x2 grid. "
            "Title the first subplot 'A'. Title the fourth subplot 'D'."
        )
        client = _FakeJsonClient(
            {
                "artifacts": [
                    {
                        "artifact_type": "subplot_layout",
                        "value": {"rows": 2, "columns": 2},
                        "source_span": "Create a figure with 4 subplots arranged in a 2x2 grid",
                        "confidence": 0.9,
                    },
                    {
                        "artifact_type": "title_requirement",
                        "panel_id": "panel_1",
                        "value": {"text": "A"},
                        "source_span": "Title the first subplot 'A'",
                        "confidence": 0.9,
                    },
                    {
                        "artifact_type": "title_requirement",
                        "panel_id": "panel_4",
                        "value": {"text": "D"},
                        "source_span": "Title the fourth subplot 'D'",
                        "confidence": 0.9,
                    },
                ]
            }
        )

        figure = LLMExpectedArtifactExtractor(client).extract(text).figure_requirements

        self.assertIsNotNone(figure)
        self.assertEqual((0, 3), tuple(axis.axis_index for axis in figure.axes))
        self.assertEqual(("A", "D"), tuple(axis.title for axis in figure.axes))
        self.assertEqual("panel_0.axis_3.title", figure.axes[1].provenance["title"][0])

    def test_compiler_rejects_axis_label_value_not_in_source_span(self):
        text = "Add axis labels for each measurement dimension."
        client = _FakeJsonClient(
            {
                "artifacts": [
                    {
                        "artifact_type": "axis_requirement",
                        "value": {"xlabel": "Petal Length", "ylabel": "Petal Width", "zlabel": "Sepal Length"},
                        "source_span": "Add axis labels for each measurement dimension",
                        "confidence": 0.9,
                    }
                ]
            }
        )

        result = LLMExpectedArtifactExtractor(client).extract(text)

        self.assertIsNone(result.figure_requirements)

    def test_compiler_rejects_boolean_tick_labels(self):
        text = "Ensure that the axis has no x or y ticks and assign a title like \"Flow Diagram of a Gadget\"."
        client = _FakeJsonClient(
            {
                "artifacts": [
                    {
                        "artifact_type": "title_requirement",
                        "value": {"text": "Flow Diagram of a Gadget"},
                        "source_span": "assign a title like \"Flow Diagram of a Gadget\"",
                        "confidence": 0.9,
                    },
                    {
                        "artifact_type": "axis_requirement",
                        "value": {"xtick_labels": False, "ytick_labels": False},
                        "source_span": "Ensure that the axis has no x or y ticks",
                        "confidence": 0.9,
                    },
                ]
            }
        )

        figure = LLMExpectedArtifactExtractor(client).extract(text).figure_requirements

        self.assertIsNotNone(figure)
        self.assertEqual("Flow Diagram of a Gadget", figure.axes[0].title)
        self.assertEqual((), figure.axes[0].xtick_labels)
        self.assertEqual((), figure.axes[0].ytick_labels)
if __name__ == "__main__":
    unittest.main()


