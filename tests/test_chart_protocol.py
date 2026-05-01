from __future__ import annotations

import unittest

from grounded_chart.chart_protocol import ChartProtocolAgent, validate_protocol
from grounded_chart.llm import LLMCompletionTrace, LLMJsonResult


class ChartProtocolTest(unittest.TestCase):
    def test_fallback_protocol_is_case_derived_not_fixed_waterfall_recipe(self) -> None:
        protocol = ChartProtocolAgent().build_protocol(
            chart_type="Waterfall",
            context={
                "construction_plan": {
                    "panels": [
                        {
                            "layers": [
                                {
                                    "layer_id": "layer.import_waterfall",
                                    "chart_type": "waterfall",
                                    "x": "Year",
                                    "y": ["Urban", "Rural"],
                                    "visual_channel_plan": {
                                        "channel_allocation": {"fill_color": {"field": "series"}},
                                    },
                                }
                            ]
                        }
                    ]
                },
                "prepared_artifacts": [
                    {"name": "step_02_imports_waterfall_render_table.csv", "relative_path": "execution/round_1/step_02_imports_waterfall_render_table.csv"}
                ],
            },
        )
        report = validate_protocol(protocol)

        self.assertTrue(report.ok, [issue.to_dict() for issue in report.issues])
        required = set(protocol.required_artifact_columns)
        self.assertIn("bar_bottom", required)
        self.assertIn("bar_height", required)
        self.assertIn("bar_top", required)
        self.assertIn("series", required)
        self.assertIn("fill_color_role", required)
        protocol_text = str(protocol.to_dict()).lower()
        self.assertIn("series", protocol_text)
        self.assertNotIn("increase -> positive/up color", protocol_text)
        hard_contracts = [item for item in protocol.visual_channel_contracts if item.strength == "hard"]
        self.assertTrue(any(item.channel == "fill_color" for item in hard_contracts))
        fill_contract = next(item for item in hard_contracts if item.channel == "fill_color")
        self.assertEqual("series_identity", fill_contract.semantic_dimension)
        self.assertEqual("series", fill_contract.field)
        self.assertEqual("hard_fidelity", fill_contract.contract_tier)
        self.assertEqual("choose_palette", fill_contract.executor_freedom)
        self.assertIn("free_design", protocol.contract_tiers)

    def test_agent_uses_llm_for_known_chart_types_when_client_is_available(self) -> None:
        class FakeClient:
            def complete_json_with_trace(self, **kwargs):
                return LLMJsonResult(
                    payload={
                        "chart_type": "waterfall",
                        "protocol_id": "llm_case_waterfall",
                        "semantic_interpretation": {"primary_color_semantic": "series identity"},
                        "data_artifacts": [
                            {
                                "name": "step_02_imports_waterfall_render_table.csv",
                                "columns": ["x_position", "bar_height", "bar_bottom", "fill_color_role"],
                            }
                        ],
                        "geometry_rules": ["Use bar_bottom and bar_height from the render table."],
                        "visual_channel_policy": {"fill_color": {"field": "fill_color_role"}},
                        "visual_channel_contracts": [
                            {
                                "layer_id": "layer.import_waterfall",
                                "channel": "fill_color",
                                "semantic_dimension": "series_identity",
                                "field": "fill_color_role",
                                "strength": "hard",
                                "domain_source": "artifact_schema",
                                "executor_freedom": "choose_palette",
                                "constraints": ["same value same color"],
                            }
                        ],
                        "rendering_rules": ["Use prepared artifacts."],
                        "required_artifact_columns": ["x_position", "bar_height", "bar_bottom", "fill_color_role"],
                        "plotting_primitives": [{"primitive": "bar"}],
                        "forbidden_shortcuts": ["Do not draw zero-based ordinary bars."],
                        "uncertainties": [],
                        "assumptions": [],
                    },
                    trace=LLMCompletionTrace(model="fake-protocol-model", raw_text="{}"),
                )

        protocol = ChartProtocolAgent(FakeClient()).build_protocol(chart_type="Waterfall", context={"query": "plot"})

        self.assertEqual("waterfall", protocol.chart_type)
        self.assertEqual("llm", protocol.source)
        self.assertEqual("llm_case_waterfall", protocol.protocol_id)
        self.assertEqual("fake-protocol-model", protocol.llm_trace.model)
        self.assertEqual("fill_color_role", protocol.visual_channel_policy["fill_color"]["field"])
        self.assertEqual("fill_color_role", protocol.visual_channel_contracts[0].field)
        self.assertEqual("hard_fidelity", protocol.visual_channel_contracts[0].contract_tier)
        self.assertEqual("choose_palette", protocol.visual_channel_contracts[0].executor_freedom)

    def test_soft_and_free_visual_channel_contracts_are_non_blocking_protocol_tiers(self) -> None:
        class FakeClient:
            def complete_json_with_trace(self, **kwargs):
                return LLMJsonResult(
                    payload={
                        "chart_type": "line",
                        "protocol_id": "llm_case_line",
                        "contract_tiers": {
                            "hard_fidelity": "Data and semantic bindings.",
                            "soft_guidance": "Readability preferences.",
                            "free_design": "Palette and exact style.",
                        },
                        "semantic_interpretation": {"summary": "trend line"},
                        "data_artifacts": [{"name": "trend.csv", "columns": ["x", "y"]}],
                        "geometry_rules": ["Use x and y values."],
                        "visual_channel_policy": {"line_color": "delegated"},
                        "visual_channel_contracts": [
                            {
                                "layer_id": "layer.trend",
                                "channel": "line_width",
                                "semantic_dimension": "emphasis",
                                "strength": "soft",
                                "contract_tier": "soft_guidance",
                            },
                            {
                                "layer_id": "layer.trend",
                                "channel": "font_family",
                                "semantic_dimension": "aesthetic_choice",
                                "strength": "free",
                                "contract_tier": "free_design",
                            },
                        ],
                        "rendering_rules": ["Make the trend readable."],
                        "required_artifact_columns": ["x", "y"],
                        "plotting_primitives": [{"primitive": "line"}],
                        "forbidden_shortcuts": [],
                        "uncertainties": [],
                        "assumptions": [],
                    },
                    trace=LLMCompletionTrace(model="fake-protocol-model", raw_text="{}"),
                )

        protocol = ChartProtocolAgent(FakeClient()).build_protocol(chart_type="line", context={"query": "plot"})
        report = validate_protocol(protocol)

        self.assertTrue(report.ok, [issue.to_dict() for issue in report.issues])
        self.assertEqual("soft_guidance", protocol.visual_channel_contracts[0].contract_tier)
        self.assertEqual("free_design", protocol.visual_channel_contracts[1].contract_tier)


if __name__ == "__main__":
    unittest.main()
