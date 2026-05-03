from grounded_chart.api import GroundedChartPipeline, HeuristicIntentParser, RuleBasedRepairer, TableSchema
from grounded_chart_adapters import BatchRunner, ChartCase, InMemoryCaseAdapter


def main() -> None:
    schema = TableSchema(columns={"category": "str", "sales": "number"})
    rows = (
        {"category": "A", "sales": 10},
        {"category": "A", "sales": 15},
        {"category": "B", "sales": 7},
    )
    cases = [
        ChartCase(
            case_id="good-aggregate",
            query="Show total sales by category in a bar chart, ascending.",
            schema=schema,
            rows=rows,
            generated_code="""
import matplotlib.pyplot as plt
totals = {}
for row in rows:
    totals[row["category"]] = totals.get(row["category"], 0) + row["sales"]
items = sorted(totals.items(), key=lambda item: item[1])
plt.bar([key for key, value in items], [value for key, value in items])
""",
        ),
        ChartCase(
            case_id="bad-missing-groupby",
            query="Show total sales by category in a bar chart, ascending.",
            schema=schema,
            rows=rows,
            generated_code="""
import matplotlib.pyplot as plt
categories = [row["category"] for row in rows]
sales = [row["sales"] for row in rows]
plt.bar(categories, sales)
""",
        ),
    ]

    pipeline = GroundedChartPipeline(parser=HeuristicIntentParser(), repairer=RuleBasedRepairer())
    batch = BatchRunner(pipeline).run(InMemoryCaseAdapter(cases))

    print("Summary:", batch.report.summary.to_dict())
    for case_report in batch.report.cases:
        print(
            case_report.case_id,
            case_report.status,
            "errors=",
            list(case_report.error_codes),
            "repair=",
            case_report.repair_scope,
        )


if __name__ == "__main__":
    main()
