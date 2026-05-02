import unittest
import json
import tempfile
from pathlib import Path

from grounded_chart.plan_agent import LLMPlanAgent, PlanAgentRequest
from grounded_chart.llm import LLMCompletionTrace, LLMJsonResult
from grounded_chart.construction_plan import ChartConstructionPlan, VisualLayerPlan, VisualPanelPlan


class LLMPlanAgentTest(unittest.TestCase):
    def test_llm_plan_agent_parses_complete_construction_plan(self):
        class FakeClient:
            def complete_json_with_trace(self, **kwargs):
                return LLMJsonResult(
                    payload={
                        "agent_name": "fake_plan_agent",
                        "construction_plan": {
                            "plan_type": "chart_construction_plan_v2",
                            "layout_strategy": "single_panel",
                            "figure_size": [8, 4],
                            "panels": [
                                {
                                    "panel_id": "panel.main",
                                    "role": "main_chart",
                                    "layers": [
                                        {
                                            "layer_id": "layer.bar",
                                            "chart_type": "bar",
                                            "role": "main_bar_layer",
                                            "x": "product",
                                            "y": ["sales"],
                                        }
                                    ],
                                }
                            ],
                            "global_elements": [{"type": "title", "text": "Sales"}],
                            "decisions": [],
                            "assumptions": [],
                            "constraints": ["Use source data only."],
                            "data_transform_plan": [],
                            "execution_steps": [],
                            "style_policy": {},
                        },
                        "feedback_resolution": [
                            {"issue_id": "fb1", "status": "addressed", "plan_change": "single panel"}
                        ],
                        "rationale": "fake",
                    },
                    trace=LLMCompletionTrace(model="fake-model", raw_text="{}"),
                )

        result = LLMPlanAgent(FakeClient()).build_plan(PlanAgentRequest(query="plot sales"))

        self.assertEqual("fake_plan_agent", result.agent_name)
        self.assertEqual("chart_construction_plan_v2", result.plan.plan_type)
        self.assertEqual("panel.main", result.plan.panels[0].panel_id)
        self.assertEqual("bar", result.plan.panels[0].layers[0].chart_type)
        self.assertEqual("addressed", result.feedback_resolution[0]["status"])
        self.assertEqual("fake-model", result.llm_trace.model)

    def test_llm_plan_agent_autofills_missing_feedback_resolution(self):
        class FakeClient:
            def complete_json_with_trace(self, **kwargs):
                return LLMJsonResult(
                    payload={
                        "agent_name": "fake_plan_agent",
                        "construction_plan": {
                            "plan_type": "chart_construction_plan_v2",
                            "layout_strategy": "single_panel",
                            "figure_size": [8, 4],
                            "panels": [
                                {
                                    "panel_id": "panel.main",
                                    "role": "main_chart",
                                    "layers": [
                                        {
                                            "layer_id": "layer.bar",
                                            "chart_type": "bar",
                                            "role": "main_bar_layer",
                                            "x": "product",
                                            "y": ["sales"],
                                            "visual_channel_plan": {"color": "series"},
                                        }
                                    ],
                                }
                            ],
                            "global_elements": [{"type": "legend", "placement": "bottom"}],
                            "decisions": [],
                            "assumptions": [],
                            "constraints": [],
                            "data_transform_plan": [],
                            "execution_steps": [],
                            "style_policy": {},
                        },
                        "feedback_resolution": [],
                        "rationale": "fake",
                    },
                    trace=LLMCompletionTrace(model="fake-model", raw_text="{}"),
                )

        request = PlanAgentRequest(
            query="plot sales",
            feedback_bundle={
                "feedback_items": [
                    {
                        "issue_id": "fb_missing",
                        "issue_type": "legend_missing",
                        "evidence": "Legend does not explain colors.",
                    }
                ]
            },
        )
        result = LLMPlanAgent(FakeClient()).build_plan(request)

        self.assertEqual(1, len(result.feedback_resolution))
        self.assertEqual("fb_missing", result.feedback_resolution[0]["issue_id"])
        self.assertEqual("missing_from_model", result.feedback_resolution[0]["status"])
        self.assertEqual("framework_autofill", result.feedback_resolution[0]["source"])
        self.assertEqual("framework_autofill", result.metadata["feedback_resolution_source"])

    def test_plan_agent_prompt_delegates_numeric_bounds_to_executor(self):
        class CapturingClient:
            def __init__(self):
                self.system_prompt = ""
                self.user_prompt = ""

            def complete_json_with_trace(self, **kwargs):
                self.system_prompt = kwargs["system_prompt"]
                self.user_prompt = kwargs["user_prompt"]
                return LLMJsonResult(
                    payload={
                        "agent_name": "fake_plan_agent",
                        "construction_plan": {
                            "plan_type": "chart_construction_plan_v2",
                            "layout_strategy": "semantic_layout",
                            "figure_size": [8, 4],
                            "panels": [
                                {
                                    "panel_id": "panel.main",
                                    "role": "main_chart",
                                    "bounds": None,
                                    "placement_policy": {"region": "main"},
                                    "layers": [
                                        {
                                            "layer_id": "layer.bar",
                                            "chart_type": "bar",
                                            "role": "main_bar_layer",
                                        }
                                    ],
                                }
                            ],
                            "global_elements": [],
                            "decisions": [],
                            "assumptions": [],
                            "constraints": [],
                            "data_transform_plan": [],
                            "execution_steps": [],
                            "style_policy": {},
                        },
                        "feedback_resolution": [],
                        "rationale": "fake",
                    },
                    trace=LLMCompletionTrace(model="fake-model", raw_text="{}"),
                )

        client = CapturingClient()
        LLMPlanAgent(client).build_plan(PlanAgentRequest(query="plot sales"))

        self.assertIn("Do not invent hard numeric bounds", client.system_prompt)
        self.assertIn("ExecutorAgent is responsible for computing concrete", client.system_prompt)
        self.assertIn("Do not output numeric panel bounds", client.user_prompt)

    def test_llm_plan_agent_writes_file_backed_workspace(self):
        class FakeClient:
            def complete_json_with_trace(self, **kwargs):
                return LLMJsonResult(
                    payload={
                        "agent_name": "workspace_plan_agent",
                        "construction_plan": {
                            "plan_type": "chart_construction_plan_v2",
                            "layout_strategy": "single_panel",
                            "figure_size": [8, 4],
                            "panels": [
                                {
                                    "panel_id": "panel.main",
                                    "role": "main_chart",
                                    "layers": [
                                        {
                                            "layer_id": "layer.bar",
                                            "chart_type": "bar",
                                            "role": "main_bar_layer",
                                            "x": "product",
                                            "y": ["sales"],
                                        }
                                    ],
                                }
                            ],
                            "global_elements": [],
                            "decisions": [],
                            "assumptions": [],
                            "constraints": [],
                            "data_transform_plan": [],
                            "execution_steps": [],
                            "style_policy": {},
                        },
                        "feedback_resolution": [],
                    },
                    trace=LLMCompletionTrace(model="fake-model", raw_text="{}"),
                )

        with tempfile.TemporaryDirectory() as output_tmp:
            result = LLMPlanAgent(FakeClient()).build_plan(
                PlanAgentRequest(query="plot sales", case_id="case_a", output_root=Path(output_tmp))
            )
            workspace = Path(output_tmp) / "PlanAgent" / "round_1"

            self.assertTrue((workspace / "input_manifest.json").exists())
            self.assertTrue((workspace / "prompt_payload.json").exists())
            self.assertTrue((workspace / "plan.json").exists())
            self.assertTrue((workspace / "task_memory.json").exists())
            self.assertTrue((workspace / "self_check.json").exists())
            self.assertEqual(str(workspace), result.metadata["workspace_dir"])
            self.assertEqual("file_backed", result.metadata["state_mode"])
            self.assertTrue(result.metadata["self_check_ok"])

    def test_round_two_loads_previous_memory_and_compacts_previous_plan(self):
        class CapturingClient:
            def __init__(self):
                self.user_prompt = ""

            def complete_json_with_trace(self, **kwargs):
                self.user_prompt = kwargs["user_prompt"]
                return LLMJsonResult(
                    payload={
                        "agent_name": "workspace_plan_agent",
                        "construction_plan": {
                            "plan_type": "chart_construction_plan_v2",
                            "layout_strategy": "replanned",
                            "figure_size": [8, 4],
                            "panels": [
                                {
                                    "panel_id": "panel.main",
                                    "role": "main_chart",
                                    "layers": [
                                        {
                                            "layer_id": "layer.line",
                                            "chart_type": "line",
                                            "role": "main_line_layer",
                                            "x": "year",
                                            "y": ["value"],
                                        }
                                    ],
                                }
                            ],
                            "global_elements": [],
                            "decisions": [],
                            "assumptions": [],
                            "constraints": [],
                            "data_transform_plan": [],
                            "execution_steps": [],
                            "style_policy": {},
                        },
                        "feedback_resolution": [
                            {
                                "issue_id": "fb_layout",
                                "status": "addressed",
                                "plan_change": "Use replanned layout.",
                                "affected_plan_refs": ["panels.panel.main"],
                            }
                        ],
                    },
                    trace=LLMCompletionTrace(model="fake-model", raw_text="{}"),
                )

        previous_plan = ChartConstructionPlan(
            plan_type="chart_construction_plan_v2",
            layout_strategy="initial",
            panels=(
                VisualPanelPlan(
                    panel_id="panel.main",
                    role="main_chart",
                    layers=(
                        VisualLayerPlan(
                            layer_id="layer.line",
                            chart_type="line",
                            role="main_line_layer",
                            x="year",
                            y=("value",),
                        ),
                    ),
                ),
            ),
        )

        with tempfile.TemporaryDirectory() as output_tmp:
            first_client = CapturingClient()
            LLMPlanAgent(first_client).build_plan(
                PlanAgentRequest(query="plot trend", case_id="case_b", output_root=Path(output_tmp), round_index=1)
            )
            second_client = CapturingClient()
            LLMPlanAgent(second_client).build_plan(
                PlanAgentRequest(
                    query="plot trend",
                    case_id="case_b",
                    output_root=Path(output_tmp),
                    round_index=2,
                    context={"plan_replanning": {"previous_construction_plan": previous_plan.to_dict(), "feedback_bundle": {"feedback_items": [{"issue_id": "nested"}]}}},
                    previous_plan=previous_plan,
                    feedback_bundle={
                        "feedback_items": [
                            {"issue_id": "fb_layout", "evidence": "Legend overlaps chart."}
                        ]
                    },
                )
            )
            payload = json.loads(second_client.user_prompt.split("\n", 1)[1])

            self.assertIn("loaded_memory", payload)
            self.assertEqual("replanned", json.loads((Path(output_tmp) / "PlanAgent" / "round_2" / "plan.json").read_text(encoding="utf-8"))["layout_strategy"])
            self.assertEqual("initial", payload["previous_plan_summary"]["layout_strategy"])
            self.assertNotIn("previous_plan", payload)
            self.assertNotIn("previous_construction_plan", payload["context"]["plan_replanning"])
            self.assertNotIn("feedback_bundle", payload["context"]["plan_replanning"])
            self.assertIn("previous_plan_summary", payload["context"]["plan_replanning"])
            self.assertIn("feedback_summary", payload["context"]["plan_replanning"])


if __name__ == "__main__":
    unittest.main()
