import unittest

from grounded_chart.api import CanonicalExecutor, ChartIntentPlan, MeasureSpec, SortSpec


class CanonicalExecutorTest(unittest.TestCase):
    def test_groupby_sum_and_sort(self):
        rows = [
            {"category": "A", "sales": 10},
            {"category": "A", "sales": 15},
            {"category": "B", "sales": 7},
        ]
        plan = ChartIntentPlan(
            chart_type="bar",
            dimensions=("category",),
            measure=MeasureSpec(column="sales", agg="sum"),
            sort=SortSpec(by="measure", direction="asc"),
        )
        trace = CanonicalExecutor().execute(plan, rows)
        self.assertEqual([(point.x, point.y) for point in trace.points], [("B", 7.0), ("A", 25.0)])


if __name__ == "__main__":
    unittest.main()
