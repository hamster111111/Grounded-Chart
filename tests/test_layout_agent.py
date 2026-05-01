from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from grounded_chart.layout_critic import VLMLayoutAgent
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
                "failed_contracts": ["layout.no_occlusion"],
                "diagnosis": "Inset overlaps the main plotting region.",
                "recommended_plan_updates": [
                    {
                        "target": "panel.pie_2002",
                        "field": "bounds",
                        "operation": "set",
                        "value": [0.12, 0.82, 0.1, 0.1],
                        "reason": "Move inset into a readable top band.",
                    }
                ],
                "confidence": 0.82,
            },
            trace=LLMCompletionTrace(model="fake-vlm", raw_text="{}"),
        )


class VLMLayoutAgentTest(unittest.TestCase):
    def test_vlm_layout_agent_uses_rendered_image_and_returns_layout_critique(self) -> None:
        client = FakeVisionClient()
        agent = VLMLayoutAgent(client)
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "round1.png"
            image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
            critique = agent.critique(
                query="Plot a chart with inset pies.",
                construction_plan={"layout_strategy": "main_axes_with_top_insets", "panels": []},
                generated_code="fig.add_axes([0.2, 0.7, 0.1, 0.1])",
                render_result=ChartRenderResult(ok=True, image_path=image_path),
                actual_figure=None,
                generation_context={"native_id": 98},
            )

        self.assertFalse(critique.ok)
        self.assertEqual(("layout.no_occlusion",), critique.failed_contracts)
        self.assertEqual("vlm", critique.metadata["mode"])
        self.assertEqual(image_path, client.calls[0]["image_path"])
        self.assertIn("rendered chart", client.calls[0]["system_prompt"].lower())
        self.assertIn("construction_plan_layout", client.calls[0]["user_prompt"])
        feedback = critique.normalized_plan_feedback()
        self.assertEqual("PlanAgent", feedback[0]["suggested_plan_action"]["target_agent"])
        self.assertEqual("panel.pie_2002", feedback[0]["suggested_plan_action"]["target_ref"])
        self.assertEqual("layout_notes", feedback[0]["suggested_plan_action"]["legacy_plan_update"]["field"])
        self.assertEqual("bounds", feedback[0]["suggested_plan_action"]["legacy_plan_update"]["legacy_field_suppressed"])
        self.assertIn(
            "ExecutorAgent during figure execution",
            feedback[0]["suggested_plan_action"]["legacy_plan_update"]["value"][0],
        )


if __name__ == "__main__":
    unittest.main()
