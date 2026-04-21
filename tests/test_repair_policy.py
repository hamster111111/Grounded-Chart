import unittest

from grounded_chart import AxisRequirementSpec, DataPoint, FigureRequirementSpec, FigureTrace, PlotTrace, RuleBasedRepairPlanner, RuleBasedRepairer
from grounded_chart.verifier import OperatorLevelVerifier


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


if __name__ == "__main__":
    unittest.main()
