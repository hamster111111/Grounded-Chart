import json
import tempfile
import unittest
from pathlib import Path

from grounded_chart import AxisRequirementSpec, FigureRequirementSpec, GroundedChartPipeline, HeuristicIntentParser, RuleBasedRepairer, TableSchema
from grounded_chart_adapters import BatchRunner, InMemoryCaseAdapter
from grounded_chart_adapters.base import ChartCase


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
        self.assertIn("length_mismatch_extra_points", summary.error_counts)
        self.assertEqual(summary.repair_level_counts["2"], 1)
        self.assertEqual(summary.repair_scope_counts["data_transformation"], 1)
        self.assertEqual(summary.repair_scope_counts["none"], 1)
        self.assertEqual(summary.backend_name_counts["matplotlib_2d"], 2)
        self.assertEqual(summary.backend_support_tier_counts["native"], 2)
        self.assertEqual(summary.backend_verification_mode_counts["hard"], 2)

        failed_report = next(case for case in batch.report.cases if case.case_id == "fail-missing-groupby")
        self.assertEqual(failed_report.status, "failed")
        self.assertEqual(failed_report.repair_level, 2)
        self.assertEqual(failed_report.repair_scope, "data_transformation")
        self.assertEqual(failed_report.expected_points[0]["x"], "B")
        self.assertEqual(failed_report.actual_points[0]["x"], "A")
        self.assertEqual(failed_report.backend_profile["backend_name"], "matplotlib_2d")
        self.assertIsNotNone(failed_report.requirement_plan)
        self.assertIsNotNone(failed_report.evidence_graph)
        self.assertIn("panel_0.chart_type", {req["requirement_id"] for req in failed_report.requirement_plan["requirements"]})
        self.assertEqual((), failed_report.repair_attempts)

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
        self.assertEqual(summary.exception_counts["NameError"], 1)
        self.assertEqual(summary.error_counts["execution_error"], 1)
        self.assertEqual(summary.backend_name_counts["matplotlib_2d"], 2)

        error_report = next(case for case in batch.report.cases if case.case_id == "error-code")
        self.assertEqual(error_report.status, "error")
        self.assertEqual(error_report.exception_type, "NameError")
        self.assertEqual(error_report.errors[0]["operator"], "execution")
        self.assertEqual(error_report.backend_profile["backend_name"], "matplotlib_2d")

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
        self.assertEqual(len(loaded["cases"]), 1)
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["case_id"], "pass-aggregate")
        self.assertEqual(loaded["summary"]["backend_name_counts"]["matplotlib_2d"], 1)
        self.assertEqual(loaded["cases"][0]["backend_profile"]["backend_name"], "matplotlib_2d")

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
