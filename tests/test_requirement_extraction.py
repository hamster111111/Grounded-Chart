import tempfile
import unittest
from pathlib import Path

from grounded_chart.api import ChartIntentPlan, MeasureSpec, ParsedRequirementBundle, TableSchema
from grounded_chart.core.requirements import ChartRequirementPlan, PanelRequirementPlan, RequirementNode
from grounded_chart_adapters import RequirementExtractionRunner
from grounded_chart_adapters.matplotbench import MatplotBenchRecord


class StubRequirementParser:
    def parse(self, query: str, schema: TableSchema) -> ChartIntentPlan:
        raise AssertionError("parse should not be called when parse_requirements is available")

    def parse_requirements(self, query: str, schema: TableSchema) -> ParsedRequirementBundle:
        plan = ChartIntentPlan(
            chart_type="line",
            dimensions=(),
            measure=MeasureSpec(column=None, agg="none"),
            raw_query=query,
            confidence=0.8,
        )
        requirement = RequirementNode(
            requirement_id="panel_0.artist_type_0",
            scope="panel",
            type="encoding",
            name="artist_type",
            value="line",
            source_span="line plot",
            status="explicit",
            confidence=0.95,
            depends_on=(),
            priority="core",
            panel_id="panel_0",
        )
        requirement_plan = ChartRequirementPlan(
            requirements=(requirement,),
            panels=(
                PanelRequirementPlan(
                    panel_id="panel_0",
                    chart_type="line",
                    requirement_ids=("panel_0.artist_type_0",),
                    data_ops={},
                    encodings={"artist_type": "line"},
                    annotations={},
                    presentation_constraints={},
                ),
            ),
            raw_query=query,
        )
        return ParsedRequirementBundle(plan=plan, requirement_plan=requirement_plan, raw_response={"chart_type": "line"})


class ErrorParser:
    def parse(self, query: str, schema: TableSchema) -> ChartIntentPlan:
        raise AssertionError("parse should not be called")

    def parse_requirements(self, query: str, schema: TableSchema) -> ParsedRequirementBundle:
        raise RuntimeError("boom")


class RequirementExtractionRunnerTest(unittest.TestCase):
    def test_runner_processes_native_matplotbench_records(self):
        parser = StubRequirementParser()
        runner = RequirementExtractionRunner(parser)
        record = MatplotBenchRecord(
            case_id="12",
            simple_instruction="simple",
            expert_instruction="Create a line plot of exponential decay.",
            metadata={"native_id": 12, "score": 17},
        )

        report = runner.run([record])

        self.assertEqual(report.summary.total_cases, 1)
        self.assertEqual(report.summary.ok_cases, 1)
        self.assertEqual(report.summary.total_requirements, 1)
        self.assertEqual(report.summary.chart_type_counts["line"], 1)
        self.assertEqual(report.summary.top_requirement_names["artist_type"], 1)
        case = report.cases[0]
        self.assertEqual(case.status, "ok")
        self.assertEqual(case.query, "Create a line plot of exponential decay.")
        self.assertEqual(case.metadata["native_id"], 12)
        self.assertEqual(case.requirement_plan["requirements"][0]["name"], "artist_type")

    def test_runner_respects_explicit_query_over_expert_instruction(self):
        parser = StubRequirementParser()
        runner = RequirementExtractionRunner(parser)
        record = {
            "case_id": "explicit-query-case",
            "query": "Use the benchmark-facing simple query.",
            "simple_instruction": "Use the simple query.",
            "expert_instruction": "Use an expert-only instruction.",
        }

        report = runner.run([record])

        self.assertEqual(report.cases[0].query, "Use the benchmark-facing simple query.")
    def test_runner_records_parser_errors_when_continue_on_error(self):
        runner = RequirementExtractionRunner(ErrorParser(), continue_on_error=True)

        report = runner.run(
            [
                {
                    "case_id": "error-case",
                    "expert_instruction": "Make a scatter plot.",
                    "native_id": 9,
                    "score": 0,
                }
            ]
        )

        self.assertEqual(report.summary.total_cases, 1)
        self.assertEqual(report.summary.errored_cases, 1)
        self.assertEqual(report.summary.exception_counts["RuntimeError"], 1)
        self.assertEqual(report.cases[0].status, "error")
        self.assertEqual(report.cases[0].exception_type, "RuntimeError")

    def test_report_can_write_json_and_jsonl(self):
        parser = StubRequirementParser()
        runner = RequirementExtractionRunner(parser)
        record = {
            "case_id": "json-case",
            "expert_instruction": "Create a line plot.",
            "native_id": 3,
            "score": 25,
        }
        report = runner.run([record])

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            json_path = root / "report.json"
            jsonl_path = root / "cases.jsonl"
            report.write_json(json_path)
            report.write_jsonl(jsonl_path)

            self.assertTrue(json_path.exists())
            self.assertTrue(jsonl_path.exists())
            self.assertIn('"total_cases": 1', json_path.read_text(encoding="utf-8"))
            self.assertIn('"case_id": "json-case"', jsonl_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
