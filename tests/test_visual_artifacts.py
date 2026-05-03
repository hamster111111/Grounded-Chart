import unittest

from grounded_chart.api import (
    ArtistTrace,
    AxisTrace,
    ExpectedArtifactNode,
    FigureRequirementSpec,
    FigureTrace,
    GroundedChartPipeline,
    HeuristicIntentParser,
    MatplotlibTraceRunner,
    OperatorLevelVerifier,
    PlotTrace,
    TableSchema,
    compile_expected_artifacts_to_figure,
    extract_actual_visual_artifacts,
    extract_code_structure_artifacts,
)


class VisualArtifactVerifierTest(unittest.TestCase):
    def test_compiles_panel_chart_type_and_connector_contracts(self):
        figure = compile_expected_artifacts_to_figure(
            (
                ExpectedArtifactNode(
                    artifact_type="panel_chart_type",
                    value={"chart_type": "pie"},
                    source_span="For the pie chart, create the pie chart on the first subplot",
                    confidence=0.9,
                    panel_id="panel_0",
                    chart_type="pie",
                    extractor="test",
                ),
                ExpectedArtifactNode(
                    artifact_type="panel_chart_type",
                    value={"chart_type": "stacked_bar"},
                    source_span="For the stacked bar chart, create a stacked bar chart on the second subplot",
                    confidence=0.9,
                    panel_id="panel_1",
                    chart_type="stacked_bar",
                    extractor="test",
                ),
                ExpectedArtifactNode(
                    artifact_type="visual_relation",
                    value={"relation_type": "connector", "count": 2, "color": "black", "linewidth": 1},
                    source_span="draw connecting lines between the separated slice and the stacked bar chart",
                    confidence=0.9,
                    extractor="test",
                ),
            )
        )

        self.assertIsNotNone(figure)
        contracts = {contract["artifact_id"]: contract for contract in figure.artifact_contracts}
        self.assertEqual(3, len(contracts))
        self.assertEqual("pie", contracts["expected.visual.panel_chart_type.0"]["expected"]["chart_type"])
        self.assertEqual("stacked_bar", contracts["expected.visual.panel_chart_type.1"]["expected"]["chart_type"])
        self.assertEqual(2, contracts["expected.visual.connector.2"]["expected"]["count"])

    def test_visual_artifact_verifier_fails_missing_connector(self):
        expected = FigureRequirementSpec(
            axes_count=2,
            artifact_contracts=(
                {
                    "artifact_id": "expected.visual.connector.0",
                    "artifact_type": "connector",
                    "expected": {"count": 2, "color": "black"},
                    "source_requirement_id": "figure.visual_relation.connector",
                    "match_policy": "count_at_least",
                },
            ),
        )
        actual = FigureTrace(
            axes=(
                AxisTrace(index=0, artists=(ArtistTrace("pie", count=3),)),
                AxisTrace(index=1, artists=(ArtistTrace("bar", count=4),)),
            ),
            raw={"figure_line_count": 0, "figure_connection_patch_count": 0},
        )

        report = OperatorLevelVerifier().verify(
            PlotTrace("unknown", ()),
            PlotTrace("unknown", ()),
            expected_figure=expected,
            actual_figure=actual,
            verify_data=False,
        )

        self.assertFalse(report.ok)
        self.assertIn("missing_visual_relation", report.error_codes)

    def test_visual_artifact_verifier_accepts_connector_count(self):
        expected = FigureRequirementSpec(
            axes_count=2,
            artifact_contracts=(
                {
                    "artifact_id": "expected.visual.connector.0",
                    "artifact_type": "connector",
                    "expected": {"count": 2, "color": "black"},
                    "source_requirement_id": "figure.visual_relation.connector",
                    "match_policy": "count_at_least",
                },
            ),
        )
        actual = FigureTrace(
            axes=(AxisTrace(index=0), AxisTrace(index=1)),
            raw={"figure_line_count": 2, "figure_connection_patch_count": 0},
        )

        report = OperatorLevelVerifier().verify(
            PlotTrace("unknown", ()),
            PlotTrace("unknown", ()),
            expected_figure=expected,
            actual_figure=actual,
            verify_data=False,
        )

        self.assertTrue(report.ok)

    def test_stacked_bar_contract_uses_coverage_gate(self):
        expected = FigureRequirementSpec(
            artifact_contracts=(
                {
                    "artifact_id": "expected.visual.panel_chart_type.0",
                    "artifact_type": "panel_chart_type",
                    "expected": {"chart_type": "stacked_bar"},
                    "locator": {"axis_index": 0},
                    "source_requirement_id": "panel_0.chart_type",
                    "match_policy": "chart_type_family",
                },
            ),
        )
        actual = FigureTrace(
            axes=(AxisTrace(index=0, artists=(ArtistTrace("bar", count=4),)),),
        )

        report = OperatorLevelVerifier().verify(
            PlotTrace("unknown", ()),
            PlotTrace("unknown", ()),
            expected_figure=expected,
            actual_figure=actual,
            verify_data=False,
        )

        self.assertFalse(report.ok)
        self.assertIn("under_verified_visual_artifact", report.error_codes)

    def test_code_structure_extractor_detects_bar_subtypes_and_connector(self):
        code = """
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
fig, (ax1, ax2) = plt.subplots(1, 2)
labels = ["A", "B"]
men = [1, 2]
women = [3, 4]
width = 0.35
x = [0, 1]
ax1.pie([1, 2], explode=(0, 0.1))
ax2.bar(x, men, width)
ax2.bar([item + width for item in x], women, width)
ax2.bar(labels, women, bottom=men)
fig.patches.append(ConnectionPatch(xyA=(0, 0), xyB=(1, 1), coordsA=ax1.transData, coordsB=ax2.transData))
"""
        actuals = extract_code_structure_artifacts(code)
        by_structure = {artifact["value"]["structure"]: artifact for artifact in actuals}

        self.assertEqual(0, by_structure["exploded_pie"]["locator"]["axis_index"])
        self.assertEqual(1, by_structure["stacked_bar"]["locator"]["axis_index"])
        self.assertEqual(1, by_structure["grouped_bar"]["locator"]["axis_index"])
        self.assertIn("connector", by_structure)
        self.assertIn("subplot_layout", by_structure)

    def test_runner_attaches_code_structure_artifacts_to_figure_trace(self):
        code = """
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
labels = ["A", "B"]
base = [1, 2]
top = [3, 4]
ax.bar(labels, base)
ax.bar(labels, top, bottom=base)
"""
        result = MatplotlibTraceRunner().run_code_with_figure(code)
        structures = {
            artifact["value"]["structure"]
            for artifact in result.figure_trace.raw.get("code_structure_artifacts", ())
        }

        self.assertIn("stacked_bar", structures)

    def test_stacked_bar_contract_passes_with_static_code_structure_evidence(self):
        expected = FigureRequirementSpec(
            artifact_contracts=(
                {
                    "artifact_id": "expected.visual.panel_chart_type.0",
                    "artifact_type": "panel_chart_type",
                    "expected": {"chart_type": "stacked_bar"},
                    "locator": {"axis_index": 0},
                    "source_requirement_id": "panel_0.chart_type",
                    "match_policy": "chart_type_family",
                },
            ),
        )
        actual = FigureTrace(
            axes=(AxisTrace(index=0, artists=(ArtistTrace("bar", count=4),)),),
            raw={
                "code_structure_artifacts": (
                    {
                        "artifact_id": "actual.code_structure.stacked_bar.0",
                        "artifact_type": "code_structure",
                        "value": {"structure": "stacked_bar", "evidence": [{"line": 8, "span": "ax.bar(..., bottom=base)"}]},
                        "locator": {"axis_index": 0, "panel_id": "panel_0"},
                        "source": "code_structure_ast_v1",
                    },
                )
            },
        )

        report = OperatorLevelVerifier().verify(
            PlotTrace("unknown", ()),
            PlotTrace("unknown", ()),
            expected_figure=expected,
            actual_figure=actual,
            verify_data=False,
        )

        self.assertTrue(report.ok)

    def test_stacked_bar_contract_keeps_locator_specific_coverage_gate(self):
        expected = FigureRequirementSpec(
            axes_count=2,
            artifact_contracts=(
                {
                    "artifact_id": "expected.visual.panel_chart_type.1",
                    "artifact_type": "panel_chart_type",
                    "expected": {"chart_type": "stacked_bar"},
                    "locator": {"axis_index": 1},
                    "source_requirement_id": "panel_1.chart_type",
                    "match_policy": "chart_type_family",
                },
            ),
        )
        actual = FigureTrace(
            axes=(
                AxisTrace(index=0, artists=(ArtistTrace("bar", count=4),)),
                AxisTrace(index=1, artists=(ArtistTrace("bar", count=4),)),
            ),
            raw={
                "code_structure_artifacts": (
                    {
                        "artifact_id": "actual.code_structure.stacked_bar.0",
                        "artifact_type": "code_structure",
                        "value": {"structure": "stacked_bar", "evidence": [{"line": 8, "span": "ax0.bar(..., bottom=base)"}]},
                        "locator": {"axis_index": 0, "panel_id": "panel_0"},
                        "source": "code_structure_ast_v1",
                    },
                )
            },
        )

        report = OperatorLevelVerifier().verify(
            PlotTrace("unknown", ()),
            PlotTrace("unknown", ()),
            expected_figure=expected,
            actual_figure=actual,
            verify_data=False,
        )

        self.assertFalse(report.ok)
        self.assertIn("under_verified_visual_artifact", report.error_codes)

    def test_pipeline_evidence_graph_includes_visual_artifacts(self):
        expected = FigureRequirementSpec(
            artifact_contracts=(
                {
                    "artifact_id": "expected.visual.connector.0",
                    "artifact_type": "connector",
                    "expected": {"count": 2},
                    "source_requirement_id": "figure.visual_relation.connector",
                },
            ),
        )
        actual_figure = FigureTrace(
            axes=(AxisTrace(index=0), AxisTrace(index=1)),
            raw={"figure_line_count": 2, "figure_connection_patch_count": 0},
        )
        result = GroundedChartPipeline(parser=HeuristicIntentParser()).run(
            query="Create a side-by-side plot with two connecting lines.",
            schema=TableSchema(columns={}),
            rows=(),
            actual_trace=PlotTrace("unknown", ()),
            expected_figure=expected,
            actual_figure=actual_figure,
            verify_data=False,
        )

        expected_ids = {artifact.artifact_id for artifact in result.evidence_graph.expected_artifacts}
        actual_ids = {artifact.artifact_id for artifact in result.evidence_graph.actual_artifacts}
        requirements = {requirement.requirement_id: requirement for requirement in result.evidence_graph.requirements}
        actual_artifacts = {artifact.artifact_id: artifact for artifact in result.evidence_graph.actual_artifacts}
        links = {link.requirement_id: link for link in result.evidence_graph.links}

        self.assertIn("expected.visual.connector.0", expected_ids)
        self.assertIn("actual.figure.connector", actual_ids)
        self.assertIn("figure.visual_relation.connector", requirements)
        self.assertIn("figure.visual_relation.connector", actual_artifacts["actual.figure.connector"].requirement_ids)
        self.assertEqual("expected.visual.connector.0", links["figure.visual_relation.connector"].expected_artifact_id)
        self.assertEqual("actual.figure.connector", links["figure.visual_relation.connector"].actual_artifact_id)
        self.assertTrue(result.report.ok)

    def test_pipeline_evidence_graph_links_code_structure_to_chart_requirement(self):
        expected = FigureRequirementSpec(
            artifact_contracts=(
                {
                    "artifact_id": "expected.visual.panel_chart_type.0",
                    "artifact_type": "panel_chart_type",
                    "expected": {"chart_type": "stacked_bar"},
                    "locator": {"axis_index": 0, "panel_id": "panel_0"},
                    "source_requirement_id": "panel_0.chart_type",
                    "match_policy": "chart_type_family",
                },
            ),
        )
        actual_figure = FigureTrace(
            axes=(AxisTrace(index=0, artists=(ArtistTrace("bar", count=4),)),),
            raw={
                "code_structure_artifacts": (
                    {
                        "artifact_id": "actual.code_structure.stacked_bar.0",
                        "artifact_type": "code_structure",
                        "value": {"structure": "stacked_bar", "evidence": [{"line": 8, "span": "ax.bar(..., bottom=base)"}]},
                        "locator": {"axis_index": 0, "panel_id": "panel_0"},
                        "source": "code_structure_ast_v1",
                    },
                )
            },
        )
        result = GroundedChartPipeline(parser=HeuristicIntentParser()).run(
            query="Create a stacked bar chart.",
            schema=TableSchema(columns={}),
            rows=(),
            actual_trace=PlotTrace("unknown", ()),
            expected_figure=expected,
            actual_figure=actual_figure,
            verify_data=False,
        )

        actual_artifacts = {artifact.artifact_id: artifact for artifact in result.evidence_graph.actual_artifacts}
        links = {link.requirement_id: link for link in result.evidence_graph.links}

        self.assertTrue(result.report.ok)
        self.assertIn("panel_0.chart_type", actual_artifacts["actual.code_structure.stacked_bar.0"].requirement_ids)
        self.assertEqual("expected.visual.panel_chart_type.0", links["panel_0.chart_type"].expected_artifact_id)
        self.assertEqual("actual.code_structure.stacked_bar.0", links["panel_0.chart_type"].actual_artifact_id)

    def test_actual_extractor_exposes_layout_panel_types_and_connectors(self):
        figure = FigureTrace(
            axes=(
                AxisTrace(index=0, bounds=(0.1, 0.1, 0.35, 0.8), artists=(ArtistTrace("pie", count=3),)),
                AxisTrace(index=1, bounds=(0.55, 0.1, 0.35, 0.8), artists=(ArtistTrace("bar", count=4),)),
            ),
            raw={"figure_line_count": 2, "figure_connection_patch_count": 1},
        )

        actuals = {artifact.artifact_id: artifact for artifact in extract_actual_visual_artifacts(figure)}

        self.assertEqual("side_by_side", actuals["actual.figure.layout"].value["orientation"])
        self.assertEqual(["pie"], actuals["actual.panel_0.chart_type"].value["chart_types"])
        self.assertEqual(3, actuals["actual.figure.connector"].value["count"])


if __name__ == "__main__":
    unittest.main()







