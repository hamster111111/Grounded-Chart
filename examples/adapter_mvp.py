from grounded_chart.api import GroundedChartPipeline, HeuristicIntentParser, RuleBasedRepairer, TableSchema
from grounded_chart_adapters import ChartCase, InMemoryCaseAdapter

case = ChartCase(
    case_id="adapter-toy-1",
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

pipeline = GroundedChartPipeline(parser=HeuristicIntentParser(), repairer=RuleBasedRepairer())
result = InMemoryCaseAdapter([case]).run(pipeline)[0]

print("Case:", result.case.case_id)
print("OK:", result.pipeline_result.report.ok)
print("Errors:", result.pipeline_result.report.error_codes)
print("Repair scope:", result.pipeline_result.repair_plan.scope if result.pipeline_result.repair_plan else None)
