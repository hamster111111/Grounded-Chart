from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from grounded_chart.figure_reader import FigureAudit, VLMFigureReaderAgent, figure_audit_plan_feedback, normalized_figure_audit_notes
from grounded_chart.llm import LLMCompletionTrace, LLMJsonResult
from grounded_chart.rendering import ChartRenderResult


class FakeVisionClient:
    def __init__(self) -> None:
        self.calls = []

    def complete_json_with_image_trace(self, **kwargs):
        self.calls.append(kwargs)
        return LLMJsonResult(
            payload={
                "ok": False,
                "summary": "The chart is hard to read because series identity and change role are conflated.",
                "readability_issues": [
                    {
                        "issue_type": "legend_ambiguity",
                        "severity": "warning",
                        "evidence": "The legend does not clearly separate series and change roles.",
                        "affected_region": "legend",
                        "related_plan_ref": "panel.main",
                        "recommendation": "Clarify legend semantics and visual channels.",
                    }
                ],
                "encoding_confusions": [
                    {
                        "issue_type": "color_role_series_confusion",
                        "severity": "warning",
                        "evidence": "Bars use color in a way that may not distinguish both semantics.",
                        "affected_region": "main bars",
                        "related_plan_ref": "panel.main.layers",
                        "recommendation": "Use separate cues for change role and series identity.",
                    }
                ],
                "data_semantic_warnings": [],
                "suspicious_artifacts": [],
                "unclear_regions": [],
                "recommended_plan_notes": [
                    {
                        "issue_type": "color_role_series_confusion",
                        "severity": "warning",
                        "evidence": "The rendered chart does not visually separate two semantic dimensions.",
                        "affected_region": "main bars",
                        "related_plan_ref": "panel.main",
                        "recommendation": "Assign color to change_role and a secondary cue such as hatch or edge style to series identity.",
                    }
                ],
                "confidence": 0.76,
            },
            trace=LLMCompletionTrace(model="fake-vlm", raw_text="{}"),
        )


class VLMFigureReaderAgentTest(unittest.TestCase):
    def test_vlm_figure_reader_uses_rendered_image_and_returns_audit(self) -> None:
        client = FakeVisionClient()
        agent = VLMFigureReaderAgent(client)
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "round1.png"
            image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
            audit = agent.audit(
                query="Create a waterfall chart.",
                construction_plan={"panels": [{"panel_id": "panel.main", "layers": []}]},
                generated_code="ax.bar(x, y, color='steelblue')",
                render_result=ChartRenderResult(ok=True, image_path=image_path),
                actual_figure=None,
                artifact_workspace_report={"artifacts": []},
                generation_context={"native_id": 98},
            )

        self.assertFalse(audit.ok)
        self.assertEqual("vlm", audit.metadata["mode"])
        self.assertEqual(image_path, client.calls[0]["image_path"])
        self.assertIn("FigureReaderAgent", client.calls[0]["system_prompt"])
        self.assertIn("construction_plan_visual_semantics", client.calls[0]["user_prompt"])
        self.assertEqual("color_role_series_confusion", audit.encoding_confusions[0]["issue_type"])

    def test_figure_audit_plan_feedback_is_bounded(self) -> None:
        client = FakeVisionClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "round1.png"
            image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
            audit = VLMFigureReaderAgent(client).audit(
                query="Create a chart.",
                construction_plan={"panels": []},
                generated_code="",
                render_result=ChartRenderResult(ok=True, image_path=image_path),
            )

        feedback = figure_audit_plan_feedback(audit)
        self.assertIsNotNone(feedback)
        self.assertFalse(feedback["ok"])
        self.assertEqual(2, feedback["issue_count"])
        self.assertIn("Do not change source data", " ".join(feedback["scope_rules"]))

    def test_normalizes_qwen_note_target_agent_shape(self) -> None:
        audit = FigureAudit(
            ok=False,
            recommended_plan_notes=(
                {
                    "note": "Strictly apply the requested secondary y-axis range (35-105) for the consumption area chart.",
                    "target_agent": "ExecutorAgent",
                },
                {
                    "note": "Add explicit category labels to pie chart wedges.",
                    "target_agent": "PlanAgent",
                },
            ),
        )

        notes = normalized_figure_audit_notes(audit)
        self.assertEqual(2, len(notes))
        self.assertEqual(
            "Strictly apply the requested secondary y-axis range (35-105) for the consumption area chart.",
            notes[0]["recommendation"],
        )
        self.assertEqual("ExecutorAgent", notes[0]["target_agent"])
        self.assertEqual("figure_audit_note", notes[0]["issue_type"])
        feedback = figure_audit_plan_feedback(audit)
        self.assertIn("35-105", feedback["recommended_plan_notes"][0]["recommendation"])
        plan_feedback = audit.normalized_plan_feedback()
        self.assertEqual("PlanAgent", plan_feedback[0]["suggested_plan_action"]["target_agent"])
        self.assertEqual("ExecutorAgent", plan_feedback[0]["suggested_plan_action"]["original_target_agent"])


if __name__ == "__main__":
    unittest.main()
