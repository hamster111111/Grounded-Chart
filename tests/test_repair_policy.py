import unittest

from grounded_chart.api import (
    AxisRequirementSpec,
    AxisTrace,
    Artifact,
    DataPoint,
    EvidenceGraph,
    EvidenceLink,
    FigureRequirementSpec,
    FigureTrace,
    PlotTrace,
    RuleBasedRepairPlanner,
    RuleBasedRepairer,
    RequirementNode,
    VerificationError,
    VerificationReport,
    apply_auto_repair_gate,
    repair_action_class_for_scope,
)
from grounded_chart.orchestration.pipeline import _diagnostic_repairability_from_evidence
from grounded_chart.verification.verifier import OperatorLevelVerifier


class RepairPolicyTest(unittest.TestCase):
    def test_data_mismatch_maps_to_data_transformation_patch(self):
        expected = PlotTrace("bar", (DataPoint("A", 25), DataPoint("B", 7)), source="expected")
        actual = PlotTrace("bar", (DataPoint("A", 10), DataPoint("A", 15), DataPoint("B", 7)), source="actual")
        report = OperatorLevelVerifier().verify(expected, actual)
        plan = RuleBasedRepairPlanner().plan(report)
        self.assertEqual(plan.repair_level, 2)
        self.assertEqual(plan.scope, "data_transformation")
        self.assertTrue(plan.should_repair)

    def test_clean_report_maps_to_no_repair(self):
        trace = PlotTrace("bar", (DataPoint("A", 25),), source="same")
        report = OperatorLevelVerifier().verify(trace, trace)
        plan = RuleBasedRepairPlanner().plan(report)
        self.assertEqual(plan.repair_level, 0)
        self.assertFalse(plan.should_repair)

    def test_execution_error_maps_to_local_patch(self):
        expected = PlotTrace("bar", (), source="expected")
        actual = PlotTrace("unknown", (), source="execution_error")
        report = VerificationReport(
            ok=False,
            errors=(
                VerificationError(
                    code="execution_error",
                    message="PathPatch.set() got an unexpected keyword argument 'trunkcolor'",
                    expected=None,
                    actual="AttributeError",
                    operator="execution",
                    requirement_id="panel_0.chart_type",
                    severity="error",
                ),
            ),
            expected_trace=expected,
            actual_trace=actual,
            expected_figure=None,
            actual_figure=None,
        )

        plan = RuleBasedRepairPlanner().plan(report)

        self.assertEqual(plan.repair_level, 1)
        self.assertEqual(plan.scope, "local_patch")

    def test_plotly_soft_backend_maps_to_backend_specific_regeneration(self):
        expected = PlotTrace("bar", (), source="expected")
        actual = PlotTrace("pie", (), source="plotly_figure", raw={"backend": "plotly", "trace_types": ["sunburst"]})
        report = OperatorLevelVerifier().verify(
            expected,
            actual,
            expected_figure=FigureRequirementSpec(
                axes=(AxisRequirementSpec(axis_index=0, title="Expected Title"),),
            ),
            actual_figure=FigureTrace(
                title="Wrong Title",
                axes=(),
                source="plotly_figure",
                raw={"backend": "plotly"},
            ),
            verify_data=False,
        )

        plan = RuleBasedRepairPlanner().plan(report)

        self.assertEqual(plan.repair_level, 3)
        self.assertEqual(plan.scope, "backend_specific_regeneration")
        self.assertTrue(plan.should_repair)
        self.assertIn("plotly", plan.reason)

    def test_unknown_backend_code_maps_to_backend_specific_regeneration(self):
        expected = PlotTrace("bar", (), source="expected")
        actual = PlotTrace("unknown", (), source="unknown")
        report = OperatorLevelVerifier().verify(expected, actual)

        plan = RuleBasedRepairPlanner().plan(report, generated_code="import holoviews as hv\nhv.Chord([])")

        self.assertEqual(plan.repair_level, 3)
        self.assertEqual(plan.scope, "backend_specific_regeneration")

    def test_repairer_uses_backend_specific_instruction_for_plotly(self):
        expected = PlotTrace("bar", (), source="expected")
        actual = PlotTrace("pie", (), source="plotly_figure", raw={"backend": "plotly", "trace_types": ["sunburst"]})
        report = OperatorLevelVerifier().verify(
            expected,
            actual,
            expected_figure=FigureRequirementSpec(
                axes=(AxisRequirementSpec(axis_index=0, title="Expected Title"),),
            ),
            actual_figure=FigureTrace(
                title="Wrong Title",
                axes=(),
                source="plotly_figure",
                raw={"backend": "plotly"},
            ),
            verify_data=False,
        )

        patch = RuleBasedRepairer().propose(
            code="import plotly.express as px\nfig = px.sunburst(...)",
            plan=type("Plan", (), {"chart_type": "bar"})(),
            report=report,
        )

        self.assertEqual(patch.repair_plan.scope, "backend_specific_regeneration")
        self.assertIn("backend-specific", patch.instruction.lower())

    def test_repairer_emits_structured_patch_ops_for_local_title_fix(self):
        expected = PlotTrace("bar", (DataPoint("A", 1),), source="expected")
        actual = PlotTrace("bar", (DataPoint("A", 1),), source="actual")
        report = OperatorLevelVerifier().verify(
            expected,
            actual,
            expected_figure=FigureRequirementSpec(
                axes=(AxisRequirementSpec(axis_index=0, title="Expected Title"),),
            ),
            actual_figure=FigureTrace(
                axes=(AxisTrace(index=0, title="Wrong Title"),),
                source="matplotlib_figure",
            ),
            verify_data=False,
        )

        patch = RuleBasedRepairer().propose(
            code="import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nax.set_title('Wrong Title')",
            plan=type("Plan", (), {"chart_type": "bar"})(),
            report=report,
        )

        self.assertEqual("local_patch", patch.repair_plan.scope)
        self.assertEqual(1, len(patch.patch_ops))
        self.assertEqual("replace_call_arg", patch.patch_ops[0].op)
        self.assertEqual("set_title", patch.patch_ops[0].anchor.name)

    def test_repairer_emits_structured_patch_ops_for_execution_error_keyword_fix(self):
        expected = PlotTrace("bar", (), source="expected")
        actual = PlotTrace("unknown", (), source="execution_error")
        report = VerificationReport(
            ok=False,
            errors=(
                VerificationError(
                    code="execution_error",
                    message="PathPatch.set() got an unexpected keyword argument 'trunkcolor'",
                    expected=None,
                    actual="AttributeError",
                    operator="execution",
                    requirement_id="panel_0.chart_type",
                    severity="error",
                ),
            ),
            expected_trace=expected,
            actual_trace=actual,
            expected_figure=None,
            actual_figure=None,
        )

        patch = RuleBasedRepairer().propose(
            code="from matplotlib.sankey import Sankey\nsankey.add(flows=[1, -1], trunkcolor='red', patchlabel='x')",
            plan=type("Plan", (), {"chart_type": "bar"})(),
            report=report,
        )

        self.assertEqual("local_patch", patch.repair_plan.scope)
        self.assertEqual(1, len(patch.patch_ops))
        self.assertEqual("remove_keyword_arg", patch.patch_ops[0].op)
        self.assertEqual("trunkcolor", patch.patch_ops[0].keyword)
        self.assertEqual("add", patch.patch_ops[0].anchor.name)

    def test_repairer_emits_replace_text_patch_for_simple_data_transformation(self):
        expected = PlotTrace("bar", (DataPoint("A", 25), DataPoint("B", 7)), source="expected")
        actual = PlotTrace("bar", (DataPoint("A", 10), DataPoint("A", 15), DataPoint("B", 7)), source="actual")
        report = OperatorLevelVerifier().verify(expected, actual)

        patch = RuleBasedRepairer().propose(
            code=(
                "import matplotlib.pyplot as plt\n"
                "categories = [row['category'] for row in rows]\n"
                "sales = [row['sales'] for row in rows]\n"
                "plt.bar(categories, sales)\n"
            ),
            plan=type(
                "Plan",
                (),
                {
                    "chart_type": "bar",
                    "dimensions": ("category",),
                    "measure": type("M", (), {"column": "sales", "agg": "sum"})(),
                    "sort": type("S", (), {"by": "measure", "direction": "asc"})(),
                    "limit": None,
                    "raw_query": "Show total sales by category in a bar chart, ascending.",
                },
            )(),
            report=report,
        )

        self.assertEqual("data_transformation", patch.repair_plan.scope)
        self.assertEqual(1, len(patch.patch_ops))
        self.assertEqual("replace_text", patch.patch_ops[0].op)
        self.assertIn("categories = [row['category'] for row in rows]", patch.patch_ops[0].anchor.text)
        self.assertIn("totals = {}", patch.patch_ops[0].new_value)

    def test_strict_policy_blocks_diagnose_only_auto_repair(self):
        expected = PlotTrace("bar", (DataPoint("A", 25), DataPoint("B", 7)), source="expected")
        actual = PlotTrace("bar", (DataPoint("A", 10), DataPoint("A", 15), DataPoint("B", 7)), source="actual")
        report = OperatorLevelVerifier().verify(expected, actual)
        plan = RuleBasedRepairPlanner().plan(report)

        decision = apply_auto_repair_gate(
            plan,
            case_metadata={"repairability": "diagnose_only"},
            mode="strict",
        )

        self.assertTrue(decision.blocked_by_policy)
        self.assertFalse(decision.should_repair)
        self.assertEqual(4, decision.effective_plan.repair_level)
        self.assertEqual("policy_blocked", decision.effective_plan.scope)
        self.assertIn("Strict repair policy blocked automatic repair", decision.effective_plan.reason)

    def test_exploratory_policy_keeps_diagnose_only_auto_repair_enabled(self):
        expected = PlotTrace("bar", (DataPoint("A", 25), DataPoint("B", 7)), source="expected")
        actual = PlotTrace("bar", (DataPoint("A", 10), DataPoint("A", 15), DataPoint("B", 7)), source="actual")
        report = OperatorLevelVerifier().verify(expected, actual)
        plan = RuleBasedRepairPlanner().plan(report)

        decision = apply_auto_repair_gate(
            plan,
            case_metadata={"repairability": "diagnose_only"},
            mode="exploratory",
        )

        self.assertFalse(decision.blocked_by_policy)
        self.assertTrue(decision.should_repair)
        self.assertEqual("data_transformation", decision.effective_plan.scope)


    def test_axis_layout_maps_to_structural_regeneration(self):
        report = VerificationReport(
            ok=False,
            errors=(
                VerificationError(
                    code="wrong_axis_layout",
                    message="Axis bounds mismatch.",
                    expected=[0.1, 0.5, 0.3, 0.3],
                    actual=[0.1, 0.1, 0.8, 0.2],
                    operator="figure_layout",
                ),
            ),
            expected_trace=PlotTrace("line", (), source="expected"),
            actual_trace=PlotTrace("line", (), source="actual"),
        )

        plan = RuleBasedRepairPlanner().plan(report)

        self.assertEqual(3, plan.repair_level)
        self.assertEqual("structural_regeneration", plan.scope)
        self.assertIn("axis-layout", plan.reason)

    def test_artist_count_without_expected_points_maps_to_diagnose_only(self):
        report = VerificationReport(
            ok=False,
            errors=(
                VerificationError(
                    code="wrong_artist_count",
                    message="Wrong number of bar patches.",
                    expected={"patch": 4},
                    actual={"patch": 5},
                    operator="artist",
                ),
            ),
            expected_trace=PlotTrace("bar", (), source="expected"),
            actual_trace=PlotTrace("bar", (), source="actual"),
        )

        plan = RuleBasedRepairPlanner().plan(report)

        self.assertEqual(4, plan.repair_level)
        self.assertEqual("diagnose_only", plan.scope)
        self.assertFalse(plan.should_repair)
        self.assertIn("expected plot-point artifacts", plan.reason)

    def test_artist_count_with_expected_points_maps_to_data_transformation(self):
        report = VerificationReport(
            ok=False,
            errors=(
                VerificationError(
                    code="wrong_artist_count",
                    message="Wrong number of bar patches.",
                    expected={"patch": 4},
                    actual={"patch": 5},
                    operator="artist",
                ),
            ),
            expected_trace=PlotTrace(
                "bar",
                (DataPoint(1, 4), DataPoint(2, 3), DataPoint(3, 2), DataPoint(4, 4)),
                source="expected",
            ),
            actual_trace=PlotTrace("bar", (), source="actual"),
        )

        plan = RuleBasedRepairPlanner().plan(report)

        self.assertEqual(2, plan.repair_level)
        self.assertEqual("data_transformation", plan.scope)
        self.assertTrue(plan.should_repair)
        self.assertIn("explicit expected plot points", plan.reason)

    def test_diagnostic_failure_atom_can_mark_case_diagnose_only(self):
        requirement = RequirementNode(
            requirement_id="panel_0.axis_0.artist_counts",
            scope="panel",
            type="encoding",
            name="artist_counts",
            value={"patch": 4},
            panel_id="panel_0",
        )
        graph = EvidenceGraph(
            requirements=(requirement,),
            expected_artifacts=(
                Artifact(
                    artifact_id="expected.figure_requirements",
                    kind="expected",
                    requirement_ids=(requirement.requirement_id,),
                    payload={"artist_counts": {"patch": 4}},
                    source="test",
                ),
            ),
            actual_artifacts=(
                Artifact(
                    artifact_id="actual.figure_trace",
                    kind="actual",
                    requirement_ids=(requirement.requirement_id,),
                    payload={"artist_counts": {"patch": 5}},
                    source="test",
                ),
            ),
            links=(
                EvidenceLink(
                    requirement_id=requirement.requirement_id,
                    expected_artifact_id="expected.figure_requirements",
                    actual_artifact_id="actual.figure_trace",
                    verdict="fail",
                    error_codes=("wrong_artist_count",),
                    message="Wrong number of visual artists.",
                ),
            ),
        )

        self.assertEqual("diagnose_only", _diagnostic_repairability_from_evidence(graph))


    def test_mixed_repairable_and_diagnostic_failure_atoms_do_not_globally_block_repair(self):
        data_requirement = RequirementNode(
            requirement_id="panel_0.aggregation",
            scope="panel",
            type="data_operation",
            name="aggregation",
            value={"points": 4},
            panel_id="panel_0",
        )
        artist_requirement = RequirementNode(
            requirement_id="panel_0.axis_0.artist_counts",
            scope="panel",
            type="encoding",
            name="artist_counts",
            value={"patch": 4},
            panel_id="panel_0",
        )
        graph = EvidenceGraph(
            requirements=(data_requirement, artist_requirement),
            expected_artifacts=(
                Artifact(
                    artifact_id="expected.plot_trace",
                    kind="expected",
                    requirement_ids=(data_requirement.requirement_id,),
                    payload=[{"x": 1, "y": 4}],
                    source="test",
                ),
                Artifact(
                    artifact_id="expected.figure_requirements",
                    kind="expected",
                    requirement_ids=(artist_requirement.requirement_id,),
                    payload={"artist_counts": {"patch": 4}},
                    source="test",
                ),
            ),
            actual_artifacts=(
                Artifact(
                    artifact_id="actual.plot_points",
                    kind="actual",
                    requirement_ids=(data_requirement.requirement_id,),
                    payload=[{"x": 1, "y": 10}],
                    source="test",
                ),
                Artifact(
                    artifact_id="actual.figure_trace",
                    kind="actual",
                    requirement_ids=(artist_requirement.requirement_id,),
                    payload={"artist_counts": {"patch": 5}},
                    source="test",
                ),
            ),
            links=(
                EvidenceLink(
                    requirement_id=data_requirement.requirement_id,
                    expected_artifact_id="expected.plot_trace",
                    actual_artifact_id="actual.plot_points",
                    verdict="fail",
                    error_codes=("wrong_aggregation_value",),
                    message="Wrong plotted value.",
                ),
                EvidenceLink(
                    requirement_id=artist_requirement.requirement_id,
                    expected_artifact_id="expected.figure_requirements",
                    actual_artifact_id="actual.figure_trace",
                    verdict="fail",
                    error_codes=("wrong_artist_count",),
                    message="Wrong number of visual artists.",
                ),
            ),
        )

        self.assertIsNone(_diagnostic_repairability_from_evidence(graph))

    def test_repair_action_class_maps_scopes_to_stable_taxonomy(self):
        self.assertEqual("none", repair_action_class_for_scope("none"))
        self.assertEqual("local_patch", repair_action_class_for_scope("local_patch"))
        self.assertEqual("data_block_regeneration", repair_action_class_for_scope("data_transformation"))
        self.assertEqual("structural_regeneration", repair_action_class_for_scope("structural_regeneration"))
        self.assertEqual("structural_regeneration", repair_action_class_for_scope("backend_specific_regeneration"))
        self.assertEqual("abstain", repair_action_class_for_scope("policy_blocked"))
        self.assertEqual("abstain", repair_action_class_for_scope("diagnose_only"))


if __name__ == "__main__":
    unittest.main()
