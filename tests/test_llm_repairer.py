import unittest

from grounded_chart import (
    AxisRequirementSpec,
    AxisTrace,
    CanonicalExecutor,
    ChartIntentPlan,
    DataPoint,
    FigureRequirementSpec,
    FigureTrace,
    LLMCompletionTrace,
    LLMJsonResult,
    LLMRepairer,
    LLMUsage,
    MeasureSpec,
    PlotTrace,
    SortSpec,
    VerificationError,
    VerificationReport,
    build_evidence_graph,
    build_requirement_plan,
)
from grounded_chart.verifier import OperatorLevelVerifier


class StubLLMClient:
    def __init__(self, payload, trace=None):
        self.payload = payload
        self.trace = trace
        self.calls = []

    def complete_text(self, **kwargs):
        raise AssertionError("complete_text should not be called in this test")

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        return dict(self.payload)

    def complete_json_with_trace(self, **kwargs):
        self.calls.append(kwargs)
        return LLMJsonResult(payload=dict(self.payload), trace=self.trace)


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
                    "patch_ops": [
                        {
                            "op": "replace_call_arg",
                            "anchor": {"kind": "method_call", "name": "suptitle", "occurrence": 1},
                            "arg_index": 0,
                            "new_value": "Sales by Category",
                        }
                    ],
                },
                trace=LLMCompletionTrace(
                    provider="openrouter.ai",
                    model="qwen/qwen2.5-coder-32b-instruct",
                    base_url="https://openrouter.ai/api/v1",
                    raw_text='{"instruction":"Fix the figure title only.","repaired_code":"..."}',
                    parsed_json={
                        "instruction": "Fix the figure title only.",
                        "repaired_code": "import matplotlib.pyplot as plt\nplt.gcf().suptitle('Sales by Category')\n",
                    },
                    usage=LLMUsage(prompt_tokens=120, completion_tokens=40, total_tokens=160, raw={"total_tokens": 160}),
                    raw_response={"id": "chatcmpl-test", "usage": {"total_tokens": 160}},
                ),
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
        self.assertIsNotNone(patch.llm_trace)
        self.assertEqual("openrouter.ai", patch.llm_trace.provider)
        self.assertEqual(160, patch.llm_trace.usage.total_tokens)
        self.assertEqual("chatcmpl-test", patch.llm_trace.raw_response["id"])
        self.assertEqual(1, len(patch.patch_ops))
        self.assertEqual("suptitle", patch.patch_ops[0].anchor.name)


    def test_llm_repairer_normalizes_plotly_update_layout_dict_patch(self):
        expected = PlotTrace("pie", (), source="expected")
        actual = PlotTrace("pie", (), source="plotly_figure", raw={"backend": "plotly"})
        report = OperatorLevelVerifier().verify(
            expected,
            actual,
            expected_figure=FigureRequirementSpec(
                axes=(
                    AxisRequirementSpec(
                        axis_index=0,
                        title="Global Food Security Index, 2020",
                        text_contains=("Overall score 0-100, 100 = best environment",),
                    ),
                ),
            ),
            actual_figure=FigureTrace(
                title="Wrong Title",
                axes=(AxisTrace(index=0, title="Wrong Title", texts=("Wrong Subtitle",), projection="plotly"),),
                source="plotly_figure",
                raw={"backend": "plotly"},
            ),
            verify_data=False,
        )
        repairer = LLMRepairer(
            StubLLMClient(
                {
                    "instruction": "Patch Plotly layout text only.",
                    "repaired_code": None,
                    "patch_ops": [
                        {
                            "op": "replace_call_arg",
                            "anchor": {"kind": "method_call", "name": "update_layout", "occurrence": 1},
                            "arg_index": 0,
                            "new_value": {
                                "title": {"text": "Global Food Security Index, 2020"},
                                "annotations": [{"text": "Overall score 0-100, 100 = best environment"}],
                                "width": 1000,
                            },
                        }
                    ],
                }
            )
        )

        patch = repairer.propose(
            code="fig.update_layout(title={'text': 'Wrong Title'}, annotations=[{'text': 'Wrong Subtitle'}])",
            plan=type("Plan", (), {"chart_type": "bar", "dimensions": (), "measure": type("M", (), {"column": None, "agg": "none"})(), "sort": None, "limit": None, "raw_query": "x"})(),
            report=report,
        )

        self.assertEqual("local_patch", patch.repair_plan.scope)
        self.assertEqual(["title", "annotations"], [operation.keyword for operation in patch.patch_ops])
        self.assertNotIn("width", [operation.keyword for operation in patch.patch_ops])

    def test_llm_repairer_can_disable_evidence_failure_atoms(self):
        plan = ChartIntentPlan(
            chart_type="bar",
            dimensions=("category",),
            measure=MeasureSpec(column="sales", agg="sum"),
            sort=SortSpec(by="measure", direction="asc"),
            raw_query="Show total sales by category in a bar chart, ascending.",
        )
        rows = (
            {"category": "A", "sales": 10},
            {"category": "A", "sales": 15},
            {"category": "B", "sales": 7},
        )
        expected = CanonicalExecutor().execute(plan, rows)
        actual = PlotTrace(
            "bar",
            (DataPoint("A", 10), DataPoint("A", 15), DataPoint("B", 7)),
            source="actual",
        )
        report = OperatorLevelVerifier().verify(expected, actual, enforce_order=True)
        evidence_graph = build_evidence_graph(
            build_requirement_plan(plan),
            report,
            expected,
            actual,
            None,
        )
        client = StubLLMClient({"instruction": "Repair grouped totals only.", "patch_ops": []})
        repairer = LLMRepairer(client, include_failure_atoms=False)

        repairer.propose(
            code="categories = [row['category'] for row in rows]\nsales = [row['sales'] for row in rows]\nplt.bar(categories, sales)\n",
            plan=plan,
            report=report,
            evidence_graph=evidence_graph,
        )

        prompt = client.calls[0]["user_prompt"]
        self.assertNotIn("Evidence-grounded failure atoms", prompt)
        self.assertNotIn("missing_groupby_or_aggregation", prompt)
        self.assertIn("Verification errors", prompt)
    def test_llm_repairer_includes_evidence_failure_atoms(self):
        plan = ChartIntentPlan(
            chart_type="bar",
            dimensions=("category",),
            measure=MeasureSpec(column="sales", agg="sum"),
            sort=SortSpec(by="measure", direction="asc"),
            raw_query="Show total sales by category in a bar chart, ascending.",
        )
        rows = (
            {"category": "A", "sales": 10},
            {"category": "A", "sales": 15},
            {"category": "B", "sales": 7},
        )
        expected = CanonicalExecutor().execute(plan, rows)
        actual = PlotTrace(
            "bar",
            (DataPoint("A", 10), DataPoint("A", 15), DataPoint("B", 7)),
            source="actual",
        )
        report = OperatorLevelVerifier().verify(expected, actual, enforce_order=True)
        evidence_graph = build_evidence_graph(
            build_requirement_plan(plan),
            report,
            expected,
            actual,
            None,
        )
        client = StubLLMClient({"instruction": "Repair grouped totals only.", "patch_ops": []})
        repairer = LLMRepairer(client)

        repairer.propose(
            code="categories = [row['category'] for row in rows]\nsales = [row['sales'] for row in rows]\nplt.bar(categories, sales)\n",
            plan=plan,
            report=report,
            evidence_graph=evidence_graph,
        )

        prompt = client.calls[0]["user_prompt"]
        self.assertIn("Evidence-grounded failure atoms", prompt)
        self.assertIn("missing_groupby_or_aggregation", prompt)
        self.assertIn("expected.aggregated_table", prompt)
        self.assertIn("actual.aggregated_table", prompt)
        self.assertIn("actual.sorted_table", prompt)
        self.assertIn('"suggested_action_scope": "data_transformation"', prompt)

    def test_llm_repairer_normalizes_setter_keyword_patch_ops(self):
        report = VerificationReport(
            ok=False,
            errors=(
                VerificationError(
                    code="wrong_axis_title",
                    message="Axis title mismatch.",
                    expected="Expected Title",
                    actual="Wrong Title",
                    requirement_id="panel_0.axis_0.title",
                ),
            ),
            expected_trace=PlotTrace("line", (), source="expected"),
            actual_trace=PlotTrace("line", (), source="actual"),
        )
        client = StubLLMClient(
            {
                "instruction": "Fix the axis title only.",
                "patch_ops": [
                    {
                        "op": "replace_keyword_arg",
                        "anchor": {"kind": "method_call", "name": "set_title", "occurrence": 1},
                        "keyword": "title",
                        "new_value": "\"Expected Title\"",
                    }
                ],
            }
        )
        repairer = LLMRepairer(client)

        patch = repairer.propose(
            code="ax.set_title('Wrong Title')\n",
            plan=ChartIntentPlan(
                chart_type="line",
                dimensions=(),
                measure=MeasureSpec(column="value", agg="none"),
                raw_query="Set the title to Expected Title.",
            ),
            report=report,
        )

        self.assertEqual(1, len(patch.patch_ops))
        self.assertEqual("replace_call_arg", patch.patch_ops[0].op)
        self.assertEqual(0, patch.patch_ops[0].arg_index)
        self.assertIsNone(patch.patch_ops[0].keyword)
        self.assertEqual("Expected Title", patch.patch_ops[0].new_value)

    def test_llm_repairer_normalizes_null_unsupported_keyword_patch_ops(self):
        report = VerificationReport(
            ok=False,
            errors=(
                VerificationError(
                    code="execution_error",
                    message="Sankey.add() got an unexpected keyword argument 'trunkcolor'",
                ),
            ),
            expected_trace=PlotTrace("sankey", (), source="expected"),
            actual_trace=PlotTrace("unknown", (), source="execution_error"),
        )
        client = StubLLMClient(
            {
                "instruction": "Remove the unsupported Sankey keyword.",
                "patch_ops": [
                    {
                        "op": "replace_keyword_arg",
                        "anchor": {"kind": "method_call", "name": "add", "occurrence": 1},
                        "keyword": "trunkcolor",
                        "new_value": None,
                    }
                ],
            }
        )
        repairer = LLMRepairer(client)

        patch = repairer.propose(
            code="sankey.add(flows=[1, -1], trunkcolor='red')\n",
            plan=ChartIntentPlan(
                chart_type="sankey",
                dimensions=(),
                measure=MeasureSpec(column="weight", agg="sum"),
                raw_query="Draw a sankey chart.",
            ),
            report=report,
        )

        self.assertEqual(1, len(patch.patch_ops))
        self.assertEqual("remove_keyword_arg", patch.patch_ops[0].op)
        self.assertEqual("trunkcolor", patch.patch_ops[0].keyword)
        self.assertIsNone(patch.patch_ops[0].new_value)

    def test_llm_repairer_normalizes_literal_list_patch_values(self):
        report = VerificationReport(
            ok=False,
            errors=(
                VerificationError(
                    code="wrong_y_tick_labels",
                    message="Tick labels mismatch.",
                    expected=("4", "3", "2", "1"),
                    actual=("1", "2", "3", "4"),
                    requirement_id="panel_0.axis_0.ytick_labels",
                ),
            ),
            expected_trace=PlotTrace("bar", (), source="expected"),
            actual_trace=PlotTrace("bar", (), source="actual"),
        )
        client = StubLLMClient(
            {
                "instruction": "Fix y tick labels only.",
                "patch_ops": [
                    {
                        "op": "replace_call_arg",
                        "anchor": {"kind": "method_call", "name": "set_yticklabels", "occurrence": 1},
                        "arg_index": 0,
                        "new_value": "['4', '3', '2', '1']",
                    }
                ],
            }
        )
        repairer = LLMRepairer(client)

        patch = repairer.propose(
            code="ax.set_yticklabels(['1', '2', '3', '4'])\n",
            plan=ChartIntentPlan(
                chart_type="bar",
                dimensions=(),
                measure=MeasureSpec(column="value", agg="none"),
                raw_query="Set y tick labels.",
            ),
            report=report,
        )

        self.assertEqual(["4", "3", "2", "1"], patch.patch_ops[0].new_value)

    def test_llm_repairer_prompt_warns_to_preserve_nonfailed_structure(self):
        report = VerificationReport(
            ok=False,
            errors=(
                VerificationError(
                    code="wrong_axis_title",
                    message="Axis title mismatch.",
                    expected="Global Food Security Index, 2020",
                    actual="",
                    requirement_id="panel_0.axis_0.title",
                ),
            ),
            expected_trace=PlotTrace("bar", (), source="expected"),
            actual_trace=PlotTrace("sunburst", (), source="actual"),
        )
        client = StubLLMClient({"instruction": "Fix title only.", "patch_ops": []})
        repairer = LLMRepairer(client)

        repairer.propose(
            code="fig = px.sunburst(df, path=['Major Area', 'Regions', 'Country'])\n",
            plan=ChartIntentPlan(
                chart_type="bar",
                dimensions=(),
                measure=MeasureSpec(column="Overall score", agg="mean"),
                raw_query="Fix the title.",
            ),
            report=report,
        )

        prompt = client.calls[0]["user_prompt"]
        self.assertIn("Preservation policy", prompt)
        self.assertIn("predicted requirement plan may be incomplete or wrong", prompt)
        self.assertIn("keep a sunburst as sunburst", prompt)
        self.assertIn("Do not switch plotting backend", prompt)

if __name__ == "__main__":
    unittest.main()
