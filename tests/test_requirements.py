import unittest

from grounded_chart.api import ChartRequirementPlan, PanelRequirementPlan, RequirementNode


class RequirementDslTest(unittest.TestCase):
    def test_requirement_status_controls_verifiability(self):
        explicit = RequirementNode(
            requirement_id="r_chart_type",
            scope="panel",
            panel_id="p1",
            type="encoding",
            name="chart_type",
            value="bar",
            source_span="bar chart",
        )
        ambiguous = RequirementNode(
            requirement_id="r_metric",
            scope="panel",
            panel_id="p1",
            type="data_operation",
            name="measure",
            value=None,
            source_span="performance",
            status="ambiguous",
        )
        plan = ChartRequirementPlan(
            requirements=(explicit, ambiguous),
            panels=(PanelRequirementPlan(panel_id="p1", chart_type="bar"),),
        )
        self.assertEqual(plan.verifiable_requirements, (explicit,))
        self.assertEqual(plan.ambiguous_requirements, (ambiguous,))


if __name__ == "__main__":
    unittest.main()
