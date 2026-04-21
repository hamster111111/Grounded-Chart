import unittest

from grounded_chart import AxisRequirementSpec, AxisTrace, DataPoint, FigureRequirementSpec, FigureTrace, LLMRepairer, PlotTrace
from grounded_chart.verifier import OperatorLevelVerifier


class StubLLMClient:
    def __init__(self, payload):
        self.payload = payload

    def complete_text(self, **kwargs):
        raise AssertionError("complete_text should not be called in this test")

    def complete_json(self, **kwargs):
        return dict(self.payload)


class LLMRepairerTest(unittest.TestCase):
    def test_llm_repairer_returns_scoped_repaired_code(self):
        expected = PlotTrace("bar", (DataPoint("A", 10), DataPoint("B", 7)), source="expected")
        actual = PlotTrace("bar", (DataPoint("A", 10), DataPoint("B", 7)), source="actual")
        report = OperatorLevelVerifier().verify(
            expected,
            actual,
            expected_figure=FigureRequirementSpec(
                axes=(AxisRequirementSpec(axis_index=0, title="Sales by Category"),),
            ),
            actual_figure=FigureTrace(
                axes=(AxisTrace(index=0, title="Wrong Title"),),
                title="",
                source="matplotlib_figure",
            ),
            verify_data=False,
        )

        repairer = LLMRepairer(
            StubLLMClient(
                {
                    "instruction": "Fix the figure title only.",
                    "repaired_code": "import matplotlib.pyplot as plt\nplt.gcf().suptitle('Sales by Category')\n",
                }
            )
        )
        patch = repairer.propose(
            code="import matplotlib.pyplot as plt\nplt.gcf().suptitle('Wrong Title')\n",
            plan=type("Plan", (), {"chart_type": "bar", "dimensions": (), "measure": type("M", (), {"column": "sales", "agg": "sum"})(), "sort": None, "limit": None, "raw_query": "x"})(),
            report=report,
        )

        self.assertEqual("llm_scoped_repair", patch.strategy)
        self.assertEqual("local_patch", patch.repair_plan.scope)
        self.assertIn("Sales by Category", patch.repaired_code)


if __name__ == "__main__":
    unittest.main()
