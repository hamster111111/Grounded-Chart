import unittest

from grounded_chart.core.construction_plan import ChartConstructionPlan, VisualPanelPlan
from grounded_chart.agents.layout import LayoutCritique
from grounded_chart.agents.plan_revision import LayoutOnlyPlanRevisionAgent


class LayoutOnlyPlanRevisionAgentTest(unittest.TestCase):
    def test_applies_panel_bounds_update_and_rejects_semantic_update(self):
        plan = ChartConstructionPlan(
            plan_type="chart_construction_plan_v2",
            layout_strategy="single_panel",
            panels=(
                VisualPanelPlan(
                    panel_id="panel.main",
                    role="primary_chart",
                    bounds=(0.1, 0.1, 0.8, 0.8),
                ),
            ),
        )
        critique = LayoutCritique(
            ok=False,
            failed_contracts=("layout.panel_bounds",),
            diagnosis="Main panel crowds insets.",
            recommended_plan_updates=(
                {
                    "target": "panel.main",
                    "field": "bounds",
                    "operation": "set",
                    "value": [0.08, 0.12, 0.7, 0.62],
                },
                {
                    "target": "panel.main",
                    "field": "chart_type",
                    "operation": "set",
                    "value": "line",
                },
            ),
        )

        result = LayoutOnlyPlanRevisionAgent().revise(
            construction_plan=plan,
            critique=critique,
            query="plot chart",
        )

        self.assertTrue(result.applied)
        self.assertEqual((0.08, 0.12, 0.7, 0.62), result.revised_plan.panels[0].bounds)
        self.assertEqual(1, len(result.applied_updates))
        self.assertEqual(1, len(result.rejected_updates))
        self.assertIn("semantic field", result.rejected_updates[0]["decision_reason"])
        self.assertEqual("layout", result.revised_plan.decisions[-1].category)

    def test_generic_inset_fallback_when_critic_reports_occlusion_without_bounds(self):
        plan = ChartConstructionPlan(
            plan_type="chart_construction_plan_v2",
            layout_strategy="main_axes_with_top_insets",
            panels=(
                VisualPanelPlan(panel_id="panel.main", role="primary_composite_chart", bounds=(0.08, 0.12, 0.74, 0.74)),
                VisualPanelPlan(panel_id="panel.pie_2002", role="inset_pie_chart", bounds=(0.18, 0.76, 0.12, 0.12)),
                VisualPanelPlan(panel_id="panel.pie_2008", role="inset_pie_chart", bounds=(0.36, 0.76, 0.12, 0.12)),
            ),
        )
        critique = LayoutCritique(
            ok=False,
            failed_contracts=("layout.no_occlusion", "layout.inset_anchor_alignment"),
            diagnosis="Inset pies overlap the main chart and need generic top-band redistribution.",
            recommended_plan_updates=(
                {
                    "target": "panel.pie_2002",
                    "field": "chart_type",
                    "operation": "set",
                    "value": "bar",
                },
            ),
        )

        result = LayoutOnlyPlanRevisionAgent().revise(
            construction_plan=plan,
            critique=critique,
            query="plot chart",
        )

        self.assertTrue(result.applied)
        self.assertEqual(2, len(result.applied_updates))
        self.assertGreater(result.revised_plan.panels[1].bounds[1], 0.8)
        self.assertGreater(result.revised_plan.panels[2].bounds[1], 0.8)
        self.assertEqual((0.08, 0.12, 0.74, 0.74), result.revised_plan.panels[0].bounds)

    def test_applies_global_elements_shorthand_update(self):
        plan = ChartConstructionPlan(
            plan_type="chart_construction_plan_v2",
            layout_strategy="single_panel",
            global_elements=(
                {"type": "legend", "placement": "bottom_or_inside_free_space"},
                {"type": "title", "placement": "main_axis"},
            ),
        )
        critique = LayoutCritique(
            ok=False,
            failed_contracts=("layout.legend_collision", "layout.title_collision"),
            diagnosis="Legend and title collide with plot area.",
            recommended_plan_updates=(
                {
                    "target": "global_elements",
                    "field": "legend.placement",
                    "operation": "update",
                    "value": "outside_right",
                },
                {
                    "target": "global_elements",
                    "field": "title.placement",
                    "operation": "update",
                    "value": "global_figure_title",
                },
            ),
        )

        result = LayoutOnlyPlanRevisionAgent().revise(
            construction_plan=plan,
            critique=critique,
            query="plot chart",
        )

        self.assertTrue(result.applied)
        self.assertEqual(2, len(result.applied_updates))
        self.assertEqual("outside_right", result.revised_plan.global_elements[0]["placement"])
        self.assertEqual("global_figure_title", result.revised_plan.global_elements[1]["placement"])

    def test_accepts_vlm_layout_agent_update_shapes(self):
        plan = ChartConstructionPlan(
            plan_type="chart_construction_plan_v2",
            layout_strategy="main_axes_with_top_insets",
            figure_size=(16.0, 10.0),
            panels=(
                VisualPanelPlan(panel_id="panel.main", role="primary_composite_chart", bounds=(0.08, 0.12, 0.74, 0.74)),
                VisualPanelPlan(panel_id="panel.pie_2016", role="inset_pie_chart", bounds=(0.64, 0.76, 0.12, 0.12)),
            ),
            global_elements=({"type": "legend", "placement": "bottom_or_inside_free_space"},),
        )
        critique = LayoutCritique(
            ok=False,
            failed_contracts=("layout.visual_crowding", "layout.inset_anchor_alignment", "layout.legend_collision"),
            diagnosis="Qwen VLM style layout critique.",
            recommended_plan_updates=(
                {
                    "target": "figure_size",
                    "field": "value",
                    "operation": "update",
                    "value": [18.0, 10.0],
                },
                {
                    "target": "panel.pie_2016",
                    "field": "layout_notes",
                    "operation": "append",
                    "value": "Verify x-anchor mapping strictly aligns pie center with 2016 tick position.",
                },
                {
                    "target": "global_elements.legend",
                    "field": "placement",
                    "operation": "update",
                    "value": "right_margin",
                },
            ),
        )

        result = LayoutOnlyPlanRevisionAgent().revise(
            construction_plan=plan,
            critique=critique,
            query="plot chart",
        )

        self.assertTrue(result.applied)
        self.assertEqual((18.0, 10.0), result.revised_plan.figure_size)
        self.assertIn("2016 tick", result.revised_plan.panels[1].layout_notes[0])
        self.assertEqual("right_margin", result.revised_plan.global_elements[0]["placement"])
        self.assertEqual(0, len(result.rejected_updates))


if __name__ == "__main__":
    unittest.main()
