from grounded_chart import GroundedChartPipeline, HeuristicIntentParser, RuleBasedRepairer, TableSchema
from grounded_chart.trace_runner import MatplotlibTraceRunner

rows = [
    {"category": "A", "sales": 10},
    {"category": "A", "sales": 15},
    {"category": "B", "sales": 7},
]

schema = TableSchema(columns={"category": "str", "sales": "number"})
query = "Show total sales by category in a bar chart, ascending."

generated_code = """
import matplotlib.pyplot as plt
categories = [row["category"] for row in rows]
sales = [row["sales"] for row in rows]
plt.bar(categories, sales)
"""

actual_trace = MatplotlibTraceRunner().run_code(generated_code, globals_dict={"rows": rows})
pipeline = GroundedChartPipeline(parser=HeuristicIntentParser(), repairer=RuleBasedRepairer())
result = pipeline.run(
    query=query,
    schema=schema,
    rows=rows,
    actual_trace=actual_trace,
    generated_code=generated_code,
)

print("Actual trace:", [(point.x, point.y) for point in actual_trace.points])
print("Expected trace:", [(point.x, point.y) for point in result.expected_trace.points])
print("OK:", result.report.ok)
print("Errors:", result.report.error_codes)
print("Repair level:", result.repair_plan.repair_level if result.repair_plan else None)
print("Repair scope:", result.repair_plan.scope if result.repair_plan else None)
print("Repair instruction:", result.repair.instruction if result.repair else None)
