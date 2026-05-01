from __future__ import annotations

import unittest

from grounded_chart.figure_reader import FigureAudit
from grounded_chart.layout_critic import LayoutCritique
from grounded_chart.plan_feedback import plan_updates_from_feedback
from grounded_chart.plan_revision import LayoutOnlyPlanRevisionAgent
from grounded_chart.construction_plan import ChartConstructionPlan, VisualPanelPlan


class PlanFeedbackTest(unittest.TestCase):
    def test_layout_update_normalizes_to_plan_agent_feedback(self) -> None:
        critique = LayoutCritique(
            ok=False,
            failed_contracts=("layout.inset_anchor_alignment",),
            diagnosis="Inset is not aligned with its year.",
            recommended_plan_updates=(
                {
                    "target": "panel.pie_2016",
                    "field": "bounds",
                    "operation": "replace",
                    "value": [0.78, 0.76, 0.04, 0.12],
                    "reason": "Move inset away from legend crowding.",
                },
            ),
            confidence=0.8,
            metadata={"critic_name": "vlm_layout_agent_v1"},
        )

        feedback = critique.normalized_plan_feedback()
        self.assertEqual(1, len(feedback))
        self.assertEqual("vlm_layout_agent_v1", feedback[0]["source_agent"])
        self.assertEqual("PlanAgent", feedback[0]["suggested_plan_action"]["target_agent"])
        self.assertEqual("panel.pie_2016", feedback[0]["suggested_plan_action"]["target_ref"])
        updates = plan_updates_from_feedback(feedback)
        self.assertEqual("panel.pie_2016", updates[0]["target"])
        self.assertEqual("layout_notes", updates[0]["field"])
        self.assertEqual("bounds", updates[0]["legacy_field_suppressed"])
        self.assertIn("ExecutorAgent during figure execution", updates[0]["value"][0])
        self.assertNotIn("deterministic layout planning", updates[0]["value"][0])

    def test_figure_reader_executor_note_is_rerouted_to_plan_agent(self) -> None:
        audit = FigureAudit(
            ok=False,
            summary="Secondary y-axis range is wrong.",
            data_semantic_warnings=(
                {
                    "issue_type": "scale_deviation",
                    "severity": "warning",
                    "evidence": "Right y-axis is 40-90 instead of 35-105.",
                    "affected_region": "right y-axis",
                    "related_plan_ref": "layer.consumption_area.semantic_modifiers.scale_policy",
                    "recommendation": "Represent the 35-105 range as a binding plan constraint.",
                },
            ),
            recommended_plan_notes=(
                {
                    "note": "Strictly apply the requested secondary y-axis range (35-105).",
                    "target_agent": "ExecutorAgent",
                },
            ),
            confidence=0.9,
            metadata={"agent_name": "vlm_figure_reader_v1"},
        )

        feedback = audit.normalized_plan_feedback()
        self.assertEqual(2, len(feedback))
        self.assertTrue(all(item["suggested_plan_action"]["target_agent"] == "PlanAgent" for item in feedback))
        note_action = feedback[1]["suggested_plan_action"]
        self.assertEqual("ExecutorAgent", note_action["original_target_agent"])
        updates = plan_updates_from_feedback(feedback)
        self.assertTrue(all(update["suggested_plan_action"]["target_agent"] == "PlanAgent" for update in updates))
        self.assertIn("35-105", updates[1]["value"][0])

    def test_plan_revision_can_consume_plan_feedback_without_legacy_updates(self) -> None:
        plan = ChartConstructionPlan(
            plan_type="chart_construction_plan_v2",
            layout_strategy="single_panel",
            panels=(VisualPanelPlan(panel_id="panel.main", role="main"),),
        )
        critique = LayoutCritique(
            ok=False,
            plan_feedback=(
                {
                    "issue_id": "figure_reader.scale_deviation.1",
                    "source_agent": "FigureReaderAgent",
                    "issue_type": "scale_deviation",
                    "severity": "warning",
                    "evidence": "Right y-axis is 40-90 instead of 35-105.",
                    "affected_region": "right y-axis",
                    "related_plan_ref": "panel.main",
                    "suggested_plan_action": {
                        "target_agent": "PlanAgent",
                        "action_type": "enforce_existing_requirement",
                        "target_ref": "panel.main",
                        "proposal": "Represent the 35-105 secondary y-axis range as a binding plan note.",
                    },
                    "confidence": 0.8,
                },
            ),
        )

        result = LayoutOnlyPlanRevisionAgent().revise(
            construction_plan=plan,
            critique=critique,
            query="Plot consumption with y-axis range 35-105.",
        )

        self.assertTrue(result.applied)
        self.assertIn("35-105", result.revised_plan.panels[0].layout_notes[0])

    def test_plan_revision_merges_legacy_updates_and_plan_feedback_updates(self) -> None:
        plan = ChartConstructionPlan(
            plan_type="chart_construction_plan_v2",
            layout_strategy="single_panel",
            panels=(
                VisualPanelPlan(panel_id="panel.main", role="main"),
                VisualPanelPlan(panel_id="panel.pie_2016", role="inset_pie_chart"),
            ),
        )
        critique = LayoutCritique(
            ok=False,
            recommended_plan_updates=(
                {
                    "target": "panel.main",
                    "field": "layout_notes",
                    "operation": "append",
                    "value": ["Waterfall series grouping is unclear."],
                    "plan_feedback_id": "fb_waterfall_series",
                },
            ),
            plan_feedback=(
                {
                    "issue_id": "layout_inset_bounds_03",
                    "source_agent": "LayoutAgent",
                    "issue_type": "layout.panel_bounds",
                    "severity": "warning",
                    "evidence": "2016 pie chart exceeds the right edge.",
                    "affected_region": "panel.pie_2016",
                    "related_plan_ref": "panel.pie_2016",
                    "suggested_plan_action": {
                        "target_agent": "PlanAgent",
                        "action_type": "revise_plan_contract",
                        "target_ref": "panel.pie_2016",
                        "proposal": "Move the 2016 inset away from the right legend area.",
                    },
                    "confidence": 0.8,
                },
            ),
        )

        result = LayoutOnlyPlanRevisionAgent().revise(
            construction_plan=plan,
            critique=critique,
            query="Plot chart with inset pies.",
        )

        self.assertTrue(result.applied)
        self.assertEqual(2, len(result.applied_updates))
        self.assertIn("Waterfall series", result.revised_plan.panels[0].layout_notes[0])
        self.assertIn("2016 inset", result.revised_plan.panels[1].layout_notes[0])


if __name__ == "__main__":
    unittest.main()
