from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from grounded_chart.agents.layout import VLMLayoutAgent
from grounded_chart.runtime.llm import LLMCompletionTrace, LLMJsonResult
from grounded_chart.runtime.rendering import ChartRenderResult


class FakeVisionClient:
    def __init__(self) -> None:
        self.calls = []

    def complete_json_with_image_trace(self, **kwargs):
        self.calls.append(kwargs)
        return LLMJsonResult(
            payload={
                "ok": False,
                "summary": "Inset overlaps the main plotting region.",
                "issues": [
                    {
                        "issue_type": "inset_overlap",
                        "severity": "high",
                        "evidence": "Inset overlaps the main plotting region.",
                        "affected_region": "top band",
                        "related_plan_ref": "panel.pie_2002",
                        "recommendation": "Move inset into a readable non-overlapping top band.",
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
        self.assertEqual(("layout.inset_overlap",), critique.failed_contracts)
        self.assertEqual("vlm", critique.metadata["mode"])
        self.assertEqual("image_and_original_task_only", critique.metadata["input_policy"])
        self.assertEqual(image_path, client.calls[0]["image_path"])
        self.assertIn("rendered chart", client.calls[0]["system_prompt"].lower())
        user_prompt = client.calls[0]["user_prompt"]
        self.assertIn("original_task", user_prompt)
        self.assertNotIn("construction_plan_layout", user_prompt)
        self.assertNotIn("generated_code_layout_excerpt", user_prompt)
        self.assertNotIn("actual_figure_layout_trace", user_prompt)
        feedback = critique.normalized_plan_feedback()
        self.assertEqual("PlanAgent", feedback[0]["suggested_plan_action"]["target_agent"])
        self.assertEqual("error", feedback[0]["severity"])
        self.assertEqual("panel.pie_2002", feedback[0]["suggested_plan_action"]["target_ref"])
        self.assertEqual("revise_layout_contract", feedback[0]["suggested_plan_action"]["action_type"])


if __name__ == "__main__":
    unittest.main()
