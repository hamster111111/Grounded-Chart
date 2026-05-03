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
                        "feedback_handling": [
                            {"issue_id": "fb1", "status": "planned", "plan_change": "single panel"}
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
        self.assertEqual("planned", result.feedback_handling[0]["status"])
        self.assertEqual("fake-model", result.llm_trace.model)

    def test_llm_plan_agent_accepts_revised_plan_payload_key(self):
        class FakeClient:
            def complete_json_with_trace(self, **kwargs):
                return LLMJsonResult(
                    payload={
                        "agent_name": "fake_replan_agent",
                        "revised_plan": {
                            "plan_type": "chart_construction_plan_v2",
                            "layout_strategy": "feedback_replanned",
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
                        "feedback_handling": [],
                    },
                    trace=LLMCompletionTrace(model="fake-model", raw_text='{"revised_plan": {}}'),
                )

        with tempfile.TemporaryDirectory() as output_tmp:
            result = LLMPlanAgent(FakeClient()).build_plan(
                PlanAgentRequest(query="revise plot", case_id="case_replan", output_root=Path(output_tmp), round_index=2)
            )
            workspace = Path(output_tmp) / "PlanAgent" / "round_2"
            response = json.loads((workspace / "llm_response.json").read_text(encoding="utf-8"))

            self.assertEqual("feedback_replanned", result.plan.layout_strategy)
            self.assertEqual("revised_plan", result.metadata["plan_payload_key"])
            self.assertEqual("fake_replan_agent", result.agent_name)
            self.assertIn("revised_plan", response["payload"])

    def test_llm_plan_agent_writes_parse_error_for_missing_plan_payload(self):
        class FakeClient:
            def complete_json_with_trace(self, **kwargs):
                return LLMJsonResult(
                    payload={"agent_name": "bad_plan_agent", "rationale": "missing plan"},
                    trace=LLMCompletionTrace(model="fake-model", raw_text='{"agent_name": "bad_plan_agent"}'),
                )

        with tempfile.TemporaryDirectory() as output_tmp:
            with self.assertRaisesRegex(ValueError, "missing a typed plan"):
                LLMPlanAgent(FakeClient()).build_plan(
                    PlanAgentRequest(query="plot sales", case_id="case_bad", output_root=Path(output_tmp))
                )
            workspace = Path(output_tmp) / "PlanAgent" / "round_1"
            parse_error = json.loads((workspace / "parse_error.json").read_text(encoding="utf-8"))
            response = json.loads((workspace / "llm_response.json").read_text(encoding="utf-8"))

            self.assertEqual("missing_plan_object", parse_error["error_type"])
            self.assertEqual(["agent_name", "rationale"], parse_error["payload_keys"])
            self.assertIn("revised_plan", parse_error["accepted_plan_keys"])
            self.assertEqual("bad_plan_agent", response["payload"]["agent_name"])

    def test_llm_plan_agent_accepts_feedback_handling_without_autofill(self):
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
                        "feedback_handling": [
                            {
                                "issue_id": "fb_missing",
                                "status": "planned",
                                "plan_change": "Use a clearer legend layout.",
                            }
                        ],
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

        self.assertEqual(1, len(result.feedback_handling))
        self.assertEqual("fb_missing", result.feedback_handling[0]["issue_id"])
        self.assertEqual("planned", result.feedback_handling[0]["status"])
        self.assertEqual("Use a clearer legend layout.", result.feedback_handling[0]["plan_change"])
        self.assertEqual(1, result.metadata["feedback_handling_count"])

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
                        "feedback_handling": [],
                        "rationale": "fake",
                    },
                    trace=LLMCompletionTrace(model="fake-model", raw_text="{}"),
                )

        client = CapturingClient()
        LLMPlanAgent(client).build_plan(PlanAgentRequest(query="plot sales"))

        self.assertIn("compact freeform execution plan brief", client.system_prompt)
        self.assertIn("ExecutorAgent compute concrete data transforms and layout", client.system_prompt)
        self.assertIn("Build or revise the chart execution plan brief", client.user_prompt)
        self.assertNotIn("self-certify", client.system_prompt)

    def test_llm_plan_agent_accepts_freeform_plan_brief_with_scaffold_bridge(self):
        class FakeClient:
            def complete_json_with_trace(self, **kwargs):
                return LLMJsonResult(
                    payload={
                        "agent_name": "freeform_plan_agent",
                        "plan_brief": {
                            "feedback_handling": [
                                {
                                    "issue_id": "fb_layout",
                                    "decision": "accept",
                                    "execution_step_ids": ["step_2"],
                                    "instruction": "Move the legend outside dense marks.",
                                }
                            ],
                            "execution_plan": [
                                {
                                    "step_id": "step_1",
                                    "goal": "Load prepared source-grounded artifacts.",
                                },
                                {
                                    "step_id": "step_2",
                                    "goal": "Draw the chart with the legend outside dense marks.",
                                },
                            ],
                            "hard_constraints": ["Do not change plotted values."],
                        },
                        "rationale": "Use a brief instead of a rigid schema.",
                    },
                    trace=LLMCompletionTrace(model="fake-model", raw_text='{"plan_brief": {}}'),
                )

        scaffold = ChartConstructionPlan(
            plan_type="chart_construction_plan_v2",
            layout_strategy="single_panel",
            panels=(
                VisualPanelPlan(
                    panel_id="panel.main",
                    role="main_chart",
                    layers=(
                        VisualLayerPlan(
                            layer_id="layer.bar",
                            chart_type="bar",
                            role="main_bar_layer",
                            x="product",
                            y=("sales",),
                        ),
                    ),
                ),
            ),
        )
        request = PlanAgentRequest(
            query="plot sales",
            case_id="case_freeform",
            output_root=None,
            scaffold_plan=scaffold,
            feedback_bundle={"feedback_items": [{"issue_id": "fb_layout", "evidence": "Legend overlaps marks."}]},
        )

        result = LLMPlanAgent(FakeClient()).build_plan(request)

        self.assertEqual("freeform_plan_agent", result.agent_name)
        self.assertEqual("freeform_plan_brief_bridge", result.metadata["plan_mode"])
        self.assertEqual("plan_brief", result.metadata["plan_payload_key"])
        self.assertEqual("bar", result.plan.panels[0].layers[0].chart_type)
        self.assertEqual("step_1", result.plan.execution_steps[0]["step_id"])
        self.assertEqual("planned", result.feedback_handling[0]["status"])
        self.assertEqual("model_plan_brief", result.feedback_handling[0]["source"])
        self.assertEqual(["step_2"], result.feedback_handling[0]["affected_plan_refs"])
        self.assertIn("hard_constraints", result.plan_brief)

    def test_llm_plan_agent_writes_freeform_plan_brief_artifact(self):
        class FakeClient:
            def complete_json_with_trace(self, **kwargs):
                return LLMJsonResult(
                    payload={
                        "agent_name": "freeform_plan_agent",
                        "execution_plan": [
                            {"step_id": "step_1", "goal": "Draw a line chart."}
                        ],
                        "hard_constraints": ["Use source rows."],
                    },
                    trace=LLMCompletionTrace(model="fake-model", raw_text='{"execution_plan": []}'),
                )

        scaffold = ChartConstructionPlan(
            plan_type="chart_construction_plan_v2",
            layout_strategy="single_panel",
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
            result = LLMPlanAgent(FakeClient()).build_plan(
                PlanAgentRequest(
                    query="plot trend",
                    case_id="case_freeform_file",
                    output_root=Path(output_tmp),
                    scaffold_plan=scaffold,
                )
            )
            workspace = Path(output_tmp) / "PlanAgent" / "round_1"
            brief = json.loads((workspace / "plan_brief.json").read_text(encoding="utf-8"))
            memory = json.loads((workspace / "task_memory.json").read_text(encoding="utf-8"))

            self.assertEqual("execution_plan", result.metadata["plan_payload_key"])
            self.assertTrue((workspace / "plan.json").exists())
            self.assertEqual("step_1", brief["execution_plan"][0]["step_id"])
            self.assertEqual("single_panel", memory["layout_strategy"])

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
                        "feedback_handling": [],
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
            self.assertTrue((workspace / "feedback_handling.json").exists())
            self.assertEqual(str(workspace), result.metadata["workspace_dir"])
            self.assertEqual("file_backed", result.metadata["state_mode"])

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
                        "feedback_handling": [
                            {
                                "issue_id": "fb_layout",
                                "status": "planned",
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
