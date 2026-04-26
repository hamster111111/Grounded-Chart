import json
import tempfile
import unittest
from pathlib import Path

from grounded_chart import (
    AxisRequirementSpec,
    FigureRequirementSpec,
    GroundedChartPipeline,
    HeuristicIntentParser,
    LLMCompletionTrace,
    LLMJsonResult,
    LLMRepairer,
    LLMUsage,
    RuleBasedRepairer,
    TableSchema,
)
from grounded_chart_adapters import BatchRunner, InMemoryCaseAdapter
from grounded_chart_adapters.base import ChartCase


class StubTraceLLMClient:
    def __init__(self, payload, trace):
        self.payload = payload
        self.trace = trace

    def complete_text(self, **kwargs):
        raise AssertionError("complete_text should not be called in this test")

    def complete_json(self, **kwargs):
        return dict(self.payload)

    def complete_json_with_trace(self, **kwargs):
        return LLMJsonResult(payload=dict(self.payload), trace=self.trace)


class AdapterReportingTest(unittest.TestCase):
    def test_batch_report_summarizes_pass_fail_and_errors(self):
        adapter = InMemoryCaseAdapter([self._passing_case(), self._missing_groupby_case()])
        pipeline = GroundedChartPipeline(parser=HeuristicIntentParser(), repairer=RuleBasedRepairer())

        batch = BatchRunner(pipeline).run(adapter)
        summary = batch.report.summary

        self.assertEqual(summary.total_cases, 2)
        self.assertEqual(summary.completed_cases, 2)
        self.assertEqual(summary.passed_cases, 1)
        self.assertEqual(summary.failed_cases, 1)
        self.assertEqual(summary.errored_cases, 0)
        self.assertEqual(summary.pass_rate, 0.5)
        self.assertEqual(summary.parse_source_counts["predicted"], 2)
        self.assertEqual(summary.total_requirements, 10)
        self.assertEqual(summary.verifiable_requirements, 10)
        self.assertEqual(summary.passed_requirements, 8)
        self.assertEqual(summary.failed_requirements, 2)
        self.assertEqual(summary.abstained_requirements, 0)
        self.assertEqual(summary.unsupported_requirements, 0)
        self.assertEqual(summary.ambiguous_requirements, 0)
        self.assertEqual(summary.requirement_coverage, 1.0)
        self.assertEqual(summary.requirement_satisfaction, 0.8)
        self.assertIn("length_mismatch_extra_points", summary.error_counts)
        self.assertEqual(summary.failed_requirement_type_counts["data_operation"], 1)
        self.assertEqual(summary.failed_requirement_type_counts["presentation_constraint"], 1)
        self.assertEqual(summary.failure_stage_counts["requirement_violation"], 1)
        self.assertEqual(summary.failure_stage_counts["none"], 1)
        self.assertEqual(summary.failure_family_counts["mixed"], 1)
        self.assertEqual(summary.failure_family_counts["none"], 1)
        self.assertEqual(summary.repair_level_counts["2"], 1)
        self.assertEqual(summary.repair_scope_counts["data_transformation"], 1)
        self.assertEqual(summary.repair_scope_counts["none"], 1)
        self.assertEqual(summary.repair_action_class_counts["data_block_regeneration"], 1)
        self.assertEqual(summary.repair_action_class_counts["none"], 1)
        self.assertEqual(summary.repair_action_outcomes["data_block_regeneration"]["failed"], 1)
        self.assertEqual(summary.backend_name_counts["matplotlib_2d"], 2)
        self.assertEqual(summary.backend_support_tier_counts["native"], 2)
        self.assertEqual(summary.backend_verification_mode_counts["hard"], 2)

        failed_report = next(case for case in batch.report.cases if case.case_id == "fail-missing-groupby")
        self.assertEqual(failed_report.status, "failed")
        self.assertEqual(failed_report.parse_source, "predicted")
        self.assertEqual(failed_report.repair_level, 2)
        self.assertEqual(failed_report.repair_scope, "data_transformation")
        self.assertEqual(failed_report.repair_action_class, "data_block_regeneration")
        self.assertIn("data preparation", failed_report.repair_routing_reason.lower())
        self.assertEqual(failed_report.requirement_metrics["failed_requirements"], 2)
        self.assertEqual(failed_report.requirement_metrics["passed_requirements"], 3)
        self.assertEqual(failed_report.failure_taxonomy["primary_stage"], "requirement_violation")
        self.assertEqual(failed_report.failure_taxonomy["primary_family"], "mixed")
        self.assertEqual(
            failed_report.failure_taxonomy["failed_requirement_ids"],
            ["panel_0.aggregation", "panel_0.sort"],
        )
        self.assertEqual(failed_report.expected_points[0]["x"], "B")
        self.assertEqual(failed_report.actual_points[0]["x"], "A")
        self.assertEqual(failed_report.backend_profile["backend_name"], "matplotlib_2d")
        self.assertIsNotNone(failed_report.requirement_plan)
        self.assertIsNotNone(failed_report.evidence_graph)
        self.assertIn("panel_0.chart_type", {req["requirement_id"] for req in failed_report.requirement_plan["requirements"]})
        chart_type_requirement = next(
            req for req in failed_report.requirement_plan["requirements"] if req["requirement_id"] == "panel_0.chart_type"
        )
        self.assertEqual("error", chart_type_requirement["severity"])
        self.assertEqual("exact", chart_type_requirement["match_policy"])
        expected_artifacts = {artifact["artifact_id"]: artifact for artifact in failed_report.evidence_graph["expected_artifacts"]}
        self.assertIn("expected.aggregated_table", expected_artifacts)
        self.assertIn("payload_preview", expected_artifacts["expected.aggregated_table"])
        self.assertEqual(
            [{"x": "A", "y": 25.0}, {"x": "B", "y": 7.0}],
            expected_artifacts["expected.aggregated_table"]["payload_preview"],
        )
        actual_artifacts = {artifact["artifact_id"]: artifact for artifact in failed_report.evidence_graph["actual_artifacts"]}
        self.assertIn("actual.data_variables", actual_artifacts)
        self.assertIn("actual.aggregated_table", actual_artifacts)
        self.assertIn("actual.sorted_table", actual_artifacts)
        variables = {item["name"]: item["preview"] for item in actual_artifacts["actual.data_variables"]["payload_preview"]}
        self.assertEqual(["A", "A", "B"], variables["categories"])
        self.assertEqual([10, 15, 7], variables["sales"])
        aggregation_chain = next(
            item
            for item in failed_report.artifact_chain_summary
            if item["requirement_id"] == "panel_0.aggregation"
        )
        self.assertEqual("expected.aggregated_table", aggregation_chain["expected_artifact_id"])
        self.assertEqual("actual.aggregated_table", aggregation_chain["actual_artifact_id"])
        self.assertIn("compare expected.aggregated_table/aggregation with actual.aggregated_table/observed_aggregated_table", aggregation_chain["diagnosis"])
        self.assertEqual([{"x": "A", "y": 25.0}, {"x": "B", "y": 7.0}], aggregation_chain["expected_preview"])
        self.assertEqual([{"x": "A", "y": 10}, {"x": "A", "y": 15}, {"x": "B", "y": 7}], aggregation_chain["actual_preview"])
        aggregation_atom = next(
            item
            for item in failed_report.failure_atoms
            if item["requirement_id"] == "panel_0.aggregation"
        )
        self.assertEqual("missing_groupby_or_aggregation", aggregation_atom["mismatch_type"])
        self.assertEqual("data_transformation", aggregation_atom["suggested_action_scope"])
        self.assertEqual("expected.aggregated_table", aggregation_atom["expected_artifact_id"])
        self.assertEqual("actual.aggregated_table", aggregation_atom["actual_artifact_id"])
        self.assertIn("panel_0.aggregation: missing_groupby_or_aggregation", aggregation_atom["evidence_summary"])
        self.assertEqual((), failed_report.repair_attempts)

    def test_batch_report_separates_warning_only_failures(self):
        adapter = InMemoryCaseAdapter([self._warning_only_figure_size_case()])
        pipeline = GroundedChartPipeline(parser=HeuristicIntentParser())

        batch = BatchRunner(pipeline).run(adapter)
        summary = batch.report.summary
        case_report = batch.report.cases[0]

        self.assertEqual(case_report.status, "failed")
        self.assertEqual(case_report.case_verdict, "warning_only_failed")
        self.assertEqual(case_report.error_codes, ("wrong_figure_size",))
        self.assertEqual(case_report.errors[0]["severity"], "warning")
        self.assertEqual(case_report.errors[0]["match_policy"], "numeric_close")
        self.assertEqual(case_report.requirement_metrics["hard_failed_requirements"], 0)
        self.assertEqual(case_report.requirement_metrics["warning_failed_requirements"], 1)
        self.assertEqual(case_report.requirement_metrics["failed_requirement_severity_counts"], {"warning": 1})
        self.assertEqual(summary.passed_cases, 0)
        self.assertEqual(summary.failed_cases, 1)
        self.assertEqual(summary.hard_failed_cases, 0)
        self.assertEqual(summary.warning_only_failed_cases, 1)
        self.assertEqual(summary.hard_failed_requirements, 0)
        self.assertEqual(summary.warning_failed_requirements, 1)
        self.assertEqual(summary.failed_requirement_severity_counts["warning"], 1)
        self.assertEqual(summary.soft_passed_cases, 1)
        self.assertEqual(summary.pass_rate, 0.0)
        self.assertEqual(summary.hard_pass_rate, 1.0)
        self.assertEqual(summary.case_verdict_counts["warning_only_failed"], 1)
        self.assertEqual(batch.report.to_dict()["cases"][0]["case_verdict"], "warning_only_failed")
    def test_batch_report_records_execution_errors_when_enabled(self):
        adapter = InMemoryCaseAdapter([self._passing_case(), self._execution_error_case()])
        pipeline = GroundedChartPipeline(parser=HeuristicIntentParser(), repairer=RuleBasedRepairer())

        batch = BatchRunner(pipeline, continue_on_error=True).run(adapter)
        summary = batch.report.summary

        self.assertEqual(summary.total_cases, 2)
        self.assertEqual(summary.completed_cases, 1)
        self.assertEqual(summary.passed_cases, 1)
        self.assertEqual(summary.errored_cases, 1)
        self.assertEqual(summary.pass_rate, 1.0)
        self.assertEqual(summary.overall_pass_rate, 0.5)
        self.assertEqual(summary.parse_source_counts["predicted"], 2)
        self.assertEqual(summary.exception_counts["NameError"], 1)
        self.assertEqual(summary.error_counts["execution_error"], 1)
        self.assertEqual(summary.failure_stage_counts["code_execution_failure"], 1)
        self.assertEqual(summary.failure_family_counts["runtime_compatibility"], 1)
        self.assertEqual(summary.repair_action_class_counts["abstain"], 1)
        self.assertEqual(summary.backend_name_counts["matplotlib_2d"], 2)

        error_report = next(case for case in batch.report.cases if case.case_id == "error-code")
        self.assertEqual(error_report.status, "error")
        self.assertEqual(error_report.parse_source, "predicted")
        self.assertEqual(error_report.exception_type, "NameError")
        self.assertEqual(error_report.errors[0]["operator"], "execution")
        self.assertEqual(error_report.backend_profile["backend_name"], "matplotlib_2d")
        self.assertEqual(error_report.repair_action_class, "abstain")
        self.assertEqual(error_report.failure_taxonomy["primary_stage"], "code_execution_failure")
        self.assertEqual(error_report.failure_taxonomy["primary_family"], "runtime_compatibility")
        self.assertIsNone(error_report.requirement_metrics)

    def test_batch_report_exports_json_and_jsonl(self):
        adapter = InMemoryCaseAdapter([self._passing_case()])
        pipeline = GroundedChartPipeline(parser=HeuristicIntentParser(), repairer=RuleBasedRepairer())
        report = BatchRunner(pipeline).run(adapter).report

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "report.json"
            jsonl_path = Path(tmpdir) / "cases.jsonl"
            report.write_json(json_path)
            report.write_jsonl(jsonl_path)

            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(loaded["summary"]["total_cases"], 1)
        self.assertEqual(loaded["summary"]["total_requirements"], 5)
        self.assertEqual(len(loaded["cases"]), 1)
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["case_id"], "pass-aggregate")
        self.assertEqual(loaded["cases"][0]["artifact_chain_summary"], [])
        self.assertEqual(loaded["cases"][0]["failure_atoms"], [])
        self.assertEqual(loaded["summary"]["backend_name_counts"]["matplotlib_2d"], 1)
        self.assertEqual(loaded["cases"][0]["backend_profile"]["backend_name"], "matplotlib_2d")
        self.assertEqual(loaded["cases"][0]["failure_taxonomy"]["primary_stage"], "none")

    def test_in_memory_adapter_has_batch_convenience_method(self):
        adapter = InMemoryCaseAdapter([self._passing_case()])
        pipeline = GroundedChartPipeline(parser=HeuristicIntentParser(), repairer=RuleBasedRepairer())

        batch = adapter.run_batch(pipeline)

        self.assertEqual(batch.report.summary.passed_cases, 1)
        self.assertEqual(batch.run_results[0].case.case_id, "pass-aggregate")

    def test_batch_report_marks_plotly_backend_when_detected(self):
        adapter = InMemoryCaseAdapter([self._plotly_case()])
        pipeline = GroundedChartPipeline(parser=HeuristicIntentParser(), repairer=RuleBasedRepairer())

        batch = BatchRunner(pipeline).run(adapter)
        case_report = batch.report.cases[0]

        self.assertEqual(batch.report.summary.backend_name_counts["plotly"], 1)
        self.assertEqual(batch.report.summary.backend_support_tier_counts["spec_accessible"], 1)
        self.assertEqual(batch.report.summary.backend_verification_mode_counts["soft"], 1)
        self.assertEqual(case_report.backend_profile["backend_name"], "plotly")
        self.assertEqual(case_report.backend_profile["support_tier"], "spec_accessible")
        self.assertEqual(case_report.backend_profile["verification_mode"], "soft")

    def test_batch_report_includes_repair_attempts_when_loop_enabled(self):
        adapter = InMemoryCaseAdapter([self._repairable_title_case()])
        pipeline = GroundedChartPipeline(
            parser=HeuristicIntentParser(),
            repairer=RuleBasedRepairer(),
            enable_bounded_repair_loop=True,
            max_repair_rounds=2,
        )

        batch = BatchRunner(pipeline).run(adapter)
        case_report = batch.report.cases[0]

        self.assertEqual(case_report.status, "passed")
        self.assertEqual(len(case_report.repair_attempts), 1)
        self.assertTrue(case_report.repair_attempts[0]["applied"])
        self.assertIn("panel_0.axis_0.title", case_report.repair_attempts[0]["resolved_requirement_ids"])
        self.assertIn("Sales by Category", case_report.repaired_code)

    def test_batch_report_can_recover_execution_error_via_repair_loop(self):
        adapter = InMemoryCaseAdapter([self._unsupported_kwarg_case()])
        pipeline = GroundedChartPipeline(
            parser=HeuristicIntentParser(),
            repairer=RuleBasedRepairer(),
            enable_bounded_repair_loop=True,
            max_repair_rounds=2,
        )

        batch = BatchRunner(pipeline, continue_on_error=True).run(adapter)
        summary = batch.report.summary
        case_report = batch.report.cases[0]

        self.assertEqual(summary.total_cases, 1)
        self.assertEqual(summary.completed_cases, 1)
        self.assertEqual(summary.passed_cases, 1)
        self.assertEqual(summary.errored_cases, 0)
        self.assertEqual(case_report.status, "passed")
        self.assertEqual(case_report.exception_type, "AttributeError")
        self.assertTrue(case_report.repair_attempts)
        self.assertNotIn("trunkcolor=", case_report.repaired_code)

    def test_batch_report_includes_llm_trace_for_repair(self):
        adapter = InMemoryCaseAdapter([self._repairable_title_case()])
        pipeline = GroundedChartPipeline(
            parser=HeuristicIntentParser(),
            repairer=LLMRepairer(
                StubTraceLLMClient(
                    payload={
                        "instruction": "Fix the title only.",
                        "patch_ops": [
                            {
                                "op": "replace_call_arg",
                                "anchor": {"kind": "method_call", "name": "set_title", "occurrence": 1},
                                "arg_index": 0,
                                "new_value": "Sales by Category",
                            }
                        ],
                        "loop_signal": "continue",
                        "loop_reason": "Localized title fix should be enough.",
                    },
                    trace=LLMCompletionTrace(
                        provider="openrouter.ai",
                        model="qwen/qwen2.5-coder-32b-instruct",
                        base_url="https://openrouter.ai/api/v1",
                        raw_text='{"instruction":"Fix the title only.","loop_signal":"continue"}',
                        parsed_json={"instruction": "Fix the title only.", "loop_signal": "continue"},
                        usage=LLMUsage(prompt_tokens=210, completion_tokens=55, total_tokens=265, raw={"total_tokens": 265}),
                        raw_response={"id": "repair-trace-test", "usage": {"total_tokens": 265}},
                    ),
                )
            ),
            enable_bounded_repair_loop=True,
            max_repair_rounds=2,
        )

        batch = BatchRunner(pipeline).run(adapter)
        case_report = batch.report.cases[0]

        self.assertEqual(case_report.status, "passed")
        self.assertIsNotNone(case_report.repair_trace)
        self.assertEqual("openrouter.ai", case_report.repair_trace["provider"])
        self.assertEqual(265, case_report.repair_trace["usage"]["total_tokens"])
        self.assertEqual("repair-trace-test", case_report.repair_trace["raw_response"]["id"])
        self.assertEqual(1, len(case_report.repair_patch_ops))
        self.assertEqual("replace_call_arg", case_report.repair_patch_ops[0]["op"])
        self.assertEqual(1, len(case_report.repair_attempts))
        self.assertIsNotNone(case_report.repair_attempts[0]["llm_trace"])
        self.assertEqual(1, len(case_report.repair_attempts[0]["patch_ops"]))
        self.assertEqual(
            "qwen/qwen2.5-coder-32b-instruct",
            case_report.repair_attempts[0]["llm_trace"]["model"],
        )

    def _passing_case(self):
        return ChartCase(
            case_id="pass-aggregate",
            query="Show total sales by category in a bar chart, ascending.",
            schema=TableSchema(columns={"category": "str", "sales": "number"}),
            rows=(
                {"category": "A", "sales": 10},
                {"category": "A", "sales": 15},
                {"category": "B", "sales": 7},
            ),
            generated_code="""
import matplotlib.pyplot as plt
totals = {}
for row in rows:
    totals[row["category"]] = totals.get(row["category"], 0) + row["sales"]
items = sorted(totals.items(), key=lambda item: item[1])
plt.bar([key for key, value in items], [value for key, value in items])
""",
        )

    def _missing_groupby_case(self):
        return ChartCase(
            case_id="fail-missing-groupby",
            query="Show total sales by category in a bar chart, ascending.",
            schema=TableSchema(columns={"category": "str", "sales": "number"}),
            rows=(
                {"category": "A", "sales": 10},
                {"category": "A", "sales": 15},
                {"category": "B", "sales": 7},
            ),
            generated_code="""
import matplotlib.pyplot as plt
categories = [row["category"] for row in rows]
sales = [row["sales"] for row in rows]
plt.bar(categories, sales)
""",
        )

    def _warning_only_figure_size_case(self):
        return ChartCase(
            case_id="warning-figure-size",
            query="Create a chart with figure size 6x8.",
            schema=TableSchema(columns={}),
            rows=(),
            generated_code="""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 12))
ax.plot([1, 2], [3, 4])
""",
            figure_requirements=FigureRequirementSpec(size_inches=(6.0, 8.0)),
            verification_mode="figure_only",
        )
    def _execution_error_case(self):
        return ChartCase(
            case_id="error-code",
            query="Show total sales by category in a bar chart, ascending.",
            schema=TableSchema(columns={"category": "str", "sales": "number"}),
            rows=(
                {"category": "A", "sales": 10},
                {"category": "B", "sales": 7},
            ),
            generated_code="""
import matplotlib.pyplot as plt
plt.bar(missing_categories, [1, 2])
""",
        )

    def _plotly_case(self):
        return ChartCase(
            case_id="plotly-case",
            query="Show a global food security sunburst chart.",
            schema=TableSchema(columns={}),
            rows=(),
            generated_code="""
import pandas as pd
import plotly.express as px
import plotly.io as pio
df = pd.DataFrame([
    {'Major Area': 'Asia', 'Regions': 'East', 'Country': 'China', 'Overall score': 70},
    {'Major Area': 'Europe', 'Regions': 'West', 'Country': 'France', 'Overall score': 80},
])
fig = px.sunburst(df, path=['Major Area', 'Regions', 'Country'], values='Overall score', title='Plotly Smoke')
pio.write_image(fig, 'novice_final.png', width=1000, height=1000)
""",
            verification_mode="figure_only",
        )

    def _repairable_title_case(self):
        return ChartCase(
            case_id="repairable-title",
            query="Show total sales by category in a bar chart.",
            schema=TableSchema(columns={"category": "str", "sales": "number"}),
            rows=(
                {"category": "A", "sales": 10},
                {"category": "B", "sales": 7},
            ),
            generated_code="""
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.bar(["A", "B"], [10, 7])
ax.set_title("Wrong Title")
ax.set_xlabel("Category")
ax.set_ylabel("Sales")
""",
            figure_requirements=FigureRequirementSpec(
                axes_count=1,
                axes=(
                    AxisRequirementSpec(
                        axis_index=0,
                        title="Sales by Category",
                        xlabel="Category",
                        ylabel="Sales",
                    ),
                ),
            ),
        )

    def _unsupported_kwarg_case(self):
        return ChartCase(
            case_id="repair-runtime-kwarg",
            query="Show a Sankey diagram from source to target.",
            schema=TableSchema(columns={"source": "str", "target": "str"}),
            rows=(
                {"source": "source", "target": "target"},
                {"source": "source", "target": "target"},
            ),
            generated_code="""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey
df = pd.DataFrame(rows)
link_data = df.groupby(['source', 'target']).size().reset_index(name='weight')
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
sankey = Sankey(ax=ax, scale=0.5, offset=0.0)
for _, row in link_data.iterrows():
    sankey.add(
        flows=[row['weight'], -row['weight']],
        labels=[row['source'], row['target']],
        orientations=[0, 0],
        trunklength=1,
        trunkcolor='red',
        patchlabel=row['source'],
        pathlengths=[0.5, 0.5],
        color='blue'
    )
    break
sankey.finish()
plt.title("Sankey Diagram: Flow from source to target")
""",
            figure_requirements=FigureRequirementSpec(
                axes_count=1,
                axes=(
                    AxisRequirementSpec(
                        axis_index=0,
                        title="Sankey Diagram: Flow from source to target",
                        text_contains=("source", "target"),
                        artist_types=("patch",),
                    ),
                ),
            ),
            verification_mode="figure_only",
        )


if __name__ == "__main__":
    unittest.main()
