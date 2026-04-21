import unittest

from grounded_chart import AxisRequirementSpec, FigureRequirementSpec, GroundedChartPipeline, HeuristicIntentParser, RepairPatch, RuleBasedRepairer, TableSchema
from grounded_chart.repair_policy import RepairPlan
from grounded_chart_adapters import ChartCase, InMemoryCaseAdapter


class RepairLoopTest(unittest.TestCase):
    def test_bounded_repair_loop_fixes_axis_title_in_one_round(self):
        case = ChartCase(
            case_id="repair-title",
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
        pipeline = GroundedChartPipeline(
            parser=HeuristicIntentParser(),
            repairer=RuleBasedRepairer(),
            enable_bounded_repair_loop=True,
            max_repair_rounds=2,
        )

        result = InMemoryCaseAdapter([case]).run(pipeline)[0].pipeline_result

        self.assertTrue(result.report.ok)
        self.assertTrue(result.repair_attempts)
        self.assertEqual(1, result.repair_attempts[0].round_index)
        self.assertTrue(result.repair_attempts[0].applied)
        self.assertIn("panel_0.axis_0.title", result.repair_attempts[0].resolved_requirement_ids)
        self.assertIsNotNone(result.repaired_code)
        self.assertIn("Sales by Category", result.repaired_code)

    def test_bounded_repair_loop_accepts_repaired_code_from_repairer(self):
        class StubRepairer:
            def propose(self, code, plan, report):
                return RepairPatch(
                    strategy="stub_llm",
                    instruction="Replace title only.",
                    target_error_codes=report.error_codes,
                    repair_plan=RepairPlan(
                        repair_level=1,
                        scope="local_patch",
                        target_error_codes=report.error_codes,
                        allowed_edits=("title call",),
                        forbidden_edits=("data logic",),
                        reason="stub",
                    ),
                    repaired_code="""
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.bar(["A", "B"], [10, 7])
ax.set_title("Sales by Category")
ax.set_xlabel("Category")
ax.set_ylabel("Sales")
""",
                )

        case = ChartCase(
            case_id="repair-title-llm",
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
                axes=(AxisRequirementSpec(axis_index=0, title="Sales by Category", xlabel="Category", ylabel="Sales"),),
            ),
        )
        pipeline = GroundedChartPipeline(
            parser=HeuristicIntentParser(),
            repairer=StubRepairer(),
            enable_bounded_repair_loop=True,
            max_repair_rounds=1,
        )

        result = InMemoryCaseAdapter([case]).run(pipeline)[0].pipeline_result

        self.assertTrue(result.report.ok)
        self.assertEqual("stub_llm", result.repair_attempts[0].strategy)
        self.assertIn("Sales by Category", result.repaired_code)


if __name__ == "__main__":
    unittest.main()
