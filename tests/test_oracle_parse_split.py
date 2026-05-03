import unittest

from grounded_chart.api import ChartIntentPlan, GroundedChartPipeline, MeasureSpec, ParsedRequirementBundle, TableSchema
from grounded_chart.core.requirements import ChartRequirementPlan, PanelRequirementPlan, RequirementNode
from grounded_chart_adapters import BatchRunner, InMemoryCaseAdapter
from grounded_chart_adapters.base import ChartCase


class FailingParser:
    def parse(self, query: str, schema: TableSchema):
        raise AssertionError("parse should not be called when oracle parsed_requirements are provided")

    def parse_requirements(self, query: str, schema: TableSchema):
        raise AssertionError("parse_requirements should not be called when oracle parsed_requirements are provided")


class OracleParseSplitTest(unittest.TestCase):
    def test_batch_runner_uses_oracle_parsed_requirements_without_calling_parser(self):
        bundle = ParsedRequirementBundle(
            plan=ChartIntentPlan(
                chart_type="bar",
                dimensions=("category",),
                measure=MeasureSpec(column="sales", agg="sum"),
                raw_query="Show total sales by category in a bar chart.",
                confidence=1.0,
            ),
            requirement_plan=ChartRequirementPlan(
                requirements=(
                    RequirementNode(
                        requirement_id="panel_0.chart_type",
                        scope="panel",
                        type="encoding",
                        name="chart_type",
                        value="bar",
                        source_span="bar chart",
                        status="explicit",
                        confidence=1.0,
                        panel_id="panel_0",
                    ),
                    RequirementNode(
                        requirement_id="panel_0.aggregation",
                        scope="panel",
                        type="data_operation",
                        name="aggregation",
                        value="sum",
                        source_span="total",
                        status="explicit",
                        confidence=1.0,
                        panel_id="panel_0",
                    ),
                    RequirementNode(
                        requirement_id="panel_0.measure_column",
                        scope="panel",
                        type="data_operation",
                        name="measure_column",
                        value="sales",
                        source_span="sales",
                        status="explicit",
                        confidence=1.0,
                        panel_id="panel_0",
                    ),
                    RequirementNode(
                        requirement_id="panel_0.dimensions",
                        scope="panel",
                        type="data_operation",
                        name="dimensions",
                        value=("category",),
                        source_span="category",
                        status="explicit",
                        confidence=1.0,
                        panel_id="panel_0",
                    ),
                ),
                panels=(
                    PanelRequirementPlan(
                        panel_id="panel_0",
                        chart_type="bar",
                        requirement_ids=(
                            "panel_0.chart_type",
                            "panel_0.aggregation",
                            "panel_0.measure_column",
                            "panel_0.dimensions",
                        ),
                        data_ops={
                            "dimensions": ("category",),
                            "measure_column": "sales",
                            "aggregation": "sum",
                            "filters": (),
                        },
                        encodings={"chart_type": "bar"},
                        annotations={},
                        presentation_constraints={},
                    ),
                ),
                raw_query="Show total sales by category in a bar chart.",
            ),
            raw_response={"source": "oracle"},
        )
        case = ChartCase(
            case_id="oracle-pass",
            query="Show total sales by category in a bar chart.",
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
plt.bar(list(totals.keys()), list(totals.values()))
""",
            parsed_requirements=bundle,
            parse_source="oracle",
        )

        batch = BatchRunner(
            GroundedChartPipeline(
                parser=FailingParser(),
                repairer=None,
                enable_bounded_repair_loop=False,
            )
        ).run(InMemoryCaseAdapter([case]))

        self.assertEqual(1, batch.report.summary.parse_source_counts["oracle"])
        self.assertEqual("oracle", batch.report.cases[0].parse_source)
        self.assertEqual("oracle", batch.run_results[0].pipeline_result.parse_source)
        self.assertEqual({"source": "oracle"}, batch.run_results[0].pipeline_result.parser_raw_response)


if __name__ == "__main__":
    unittest.main()
