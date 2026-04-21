from grounded_chart import (
    GroundedChartPipeline,
    HeuristicIntentParser,
    RuleBasedRepairer,
    TableSchema,
)
from grounded_chart.trace_runner import ManualTraceRunner

rows = [
    {"category": "A", "sales": 10},
    {"category": "A", "sales": 15},
    {"category": "B", "sales": 7},
]

schema = TableSchema(columns={"category": "str", "sales": "number"})
query = "Show total sales by category in a bar chart, ascending."

# Simulate a bad generated chart: it plotted raw rows instead of grouped totals.
actual_trace = ManualTraceRunner().from_points(
    "bar",
    [("A", 10), ("A", 15), ("B", 7)],
    source="simulated_bad_code",
)

pipeline = GroundedChartPipeline(
    parser=HeuristicIntentParser(),
    repairer=RuleBasedRepairer(),
)
result = pipeline.run(query=query, schema=schema, rows=rows, actual_trace=actual_trace)

print("Plan:", result.plan)
print("Expected:", result.expected_trace.points)
print("Actual:", result.actual_trace.points)
print("OK:", result.report.ok)
print("Errors:", result.report.error_codes)
print("Repair:", result.repair.instruction if result.repair else None)
