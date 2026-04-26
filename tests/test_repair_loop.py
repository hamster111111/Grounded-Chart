import unittest

from grounded_chart import (
    AxisRequirementSpec,
    FigureRequirementSpec,
    GroundedChartPipeline,
    HeuristicIntentParser,
    PatchAnchor,
    PatchOperation,
    RepairPatch,
    RuleBasedRepairer,
    TableSchema,
)
from grounded_chart.repair_loop import _patch_op_budget, _repaired_code_preservation_rejection, _validate_strict_local_patch_ops
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
            max_repair_rounds=3,
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

    def test_strict_local_patch_disallows_freeform_repaired_code_fallback(self):
        class FreeformRepairer:
            def propose(self, code, plan, report):
                return RepairPatch(
                    strategy="freeform_stub",
                    instruction="Replace title only.",
                    target_error_codes=report.error_codes,
                    repair_plan=RepairPlan(
                        repair_level=1,
                        scope="local_patch",
                        target_error_codes=report.error_codes,
                        allowed_edits=("title call",),
                        forbidden_edits=("data logic",),
                        reason="Local patch only.",
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
            case_id="repair-title-strict-freeform-blocked",
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
            repairer=FreeformRepairer(),
            enable_bounded_repair_loop=True,
            max_repair_rounds=1,
            repair_policy_mode="strict",
        )

        result = InMemoryCaseAdapter([case]).run(pipeline)[0].pipeline_result

        self.assertFalse(result.report.ok)
        self.assertEqual("stop", result.repair_loop_status)
        self.assertIn("structured patch_ops", result.repair_loop_reason)
        self.assertEqual(1, len(result.repair_attempts))
        self.assertFalse(result.repair_attempts[0].applied)
        self.assertEqual("freeform_stub", result.repair_attempts[0].strategy)

    def test_bounded_repair_loop_applies_structured_patch_ops(self):
        class StructuredPatchRepairer:
            def propose(self, code, plan, report):
                return RepairPatch(
                    strategy="structured_patch_stub",
                    instruction="Fix title via constrained patch op.",
                    target_error_codes=report.error_codes,
                    repair_plan=RepairPlan(
                        repair_level=1,
                        scope="local_patch",
                        target_error_codes=report.error_codes,
                        allowed_edits=("title call",),
                        forbidden_edits=("data logic",),
                        reason="Use a localized structured patch.",
                    ),
                    patch_ops=(
                        PatchOperation(
                            op="replace_call_arg",
                            anchor=PatchAnchor(kind="method_call", name="set_title", occurrence=1),
                            arg_index=0,
                            new_value="Sales by Category",
                            description="Update the title only.",
                        ),
                    ),
                )

        case = ChartCase(
            case_id="repair-title-structured-patch",
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
            repairer=StructuredPatchRepairer(),
            enable_bounded_repair_loop=True,
            max_repair_rounds=1,
            repair_policy_mode="strict",
        )

        result = InMemoryCaseAdapter([case]).run(pipeline)[0].pipeline_result

        self.assertTrue(result.report.ok)
        self.assertEqual("structured_patch_stub", result.repair_attempts[0].strategy)
        self.assertEqual(1, len(result.repair_attempts[0].patch_ops))
        self.assertIn("Sales by Category", result.repaired_code)

    def test_bounded_repair_loop_applies_data_transformation_patch_ops(self):
        case = ChartCase(
            case_id="repair-data-transformation-aggregate",
            query="Show total sales by category in a bar chart, ascending.",
            schema=TableSchema(columns={"category": "str", "sales": "number"}),
            rows=(
                {"category": "A", "sales": 10},
                {"category": "A", "sales": 15},
                {"category": "B", "sales": 7},
            ),
            generated_code="""
import matplotlib.pyplot as plt
categories = [row['category'] for row in rows]
sales = [row['sales'] for row in rows]
plt.bar(categories, sales)
""",
        )
        pipeline = GroundedChartPipeline(
            parser=HeuristicIntentParser(),
            repairer=RuleBasedRepairer(),
            enable_bounded_repair_loop=True,
            max_repair_rounds=2,
            repair_policy_mode="exploratory",
        )

        result = InMemoryCaseAdapter([case]).run(pipeline)[0].pipeline_result

        self.assertTrue(result.report.ok)
        self.assertEqual("data_transformation", result.repair_plan.scope)
        self.assertTrue(result.repair_attempts)
        self.assertEqual("replace_text", result.repair_attempts[0].patch_ops[0].op)
        self.assertIn("totals = {}", result.repaired_code)
        self.assertIn("items = sorted(items", result.repaired_code)

    def test_strict_local_patch_rejects_non_whitelisted_patch_ops(self):
        class BadStructuredPatchRepairer:
            def propose(self, code, plan, report):
                return RepairPatch(
                    strategy="bad_structured_patch_stub",
                    instruction="Try a non-whitelisted title fix.",
                    target_error_codes=report.error_codes,
                    repair_plan=RepairPlan(
                        repair_level=1,
                        scope="local_patch",
                        target_error_codes=report.error_codes,
                        allowed_edits=("title call",),
                        forbidden_edits=("data logic",),
                        reason="Use a localized structured patch.",
                    ),
                    patch_ops=(
                        PatchOperation(
                            op="replace_call_arg",
                            anchor=PatchAnchor(kind="method_call", name="set_xlabel", occurrence=1),
                            arg_index=0,
                            new_value="Sales by Category",
                            description="This should be rejected because the failed family is title, not xlabel.",
                        ),
                    ),
                )

        case = ChartCase(
            case_id="repair-title-structured-patch-rejected",
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
            repairer=BadStructuredPatchRepairer(),
            enable_bounded_repair_loop=True,
            max_repair_rounds=1,
            repair_policy_mode="strict",
        )

        result = InMemoryCaseAdapter([case]).run(pipeline)[0].pipeline_result

        self.assertFalse(result.report.ok)
        self.assertEqual("stop", result.repair_loop_status)
        self.assertIn("whitelist rejected", result.repair_loop_reason)
        self.assertEqual(1, len(result.repair_attempts))
        self.assertFalse(result.repair_attempts[0].applied)
        self.assertEqual("bad_structured_patch_stub", result.repair_attempts[0].strategy)

    def test_strict_execution_error_accepts_exact_keyword_structured_patch(self):
        case = ChartCase(
            case_id="repair-runtime-kwarg-strict",
            query="Show a Sankey diagram from source to target.",
            schema=TableSchema(columns={"source": "str", "target": "str"}),
            rows=(
                {"source": "source", "target": "target"},
                {"source": "source", "target": "target"},
            ),
            generated_code="""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey

df = pd.DataFrame(rows)
link_data = df.groupby(['source', 'target']).size().reset_index(name='weight')
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
sankey = Sankey(ax=ax, scale=0.5, offset=0.0)
first_row = True
for _, row in link_data.iterrows():
    if first_row:
        sankey.add(
            flows=[row['weight'], -row['weight']],
            labels=[row['source'], row['target']],
            orientations=[0, 0],
            trunklength=1,
            trunkcolor='red',
            patchlabel=row['source'],
            pathlengths=[0.5, 0.5],
            color='blue'
        )
        first_row = False
sankey.finish()
plt.title("Sankey Diagram: Flow from source to target")
""",
            figure_requirements=FigureRequirementSpec(
                axes_count=1,
                axes=(
                    AxisRequirementSpec(
                        axis_index=0,
                        title="Sankey Diagram: Flow from source to target",
                        text_contains=("source", "target"),
                        artist_types=("patch",),
                    ),
                ),
            ),
            verification_mode="figure_only",
        )
        pipeline = GroundedChartPipeline(
            parser=HeuristicIntentParser(),
            repairer=RuleBasedRepairer(),
            enable_bounded_repair_loop=True,
            max_repair_rounds=2,
            repair_policy_mode="strict",
        )

        batch = InMemoryCaseAdapter([case]).run_batch(pipeline, continue_on_error=True)
        report = batch.report.cases[0]

        self.assertEqual("passed", report.status)
        self.assertTrue(report.repair_attempts)
        self.assertEqual("remove_keyword_arg", report.repair_attempts[0]["patch_ops"][0]["op"])
        self.assertEqual("trunkcolor", report.repair_attempts[0]["patch_ops"][0]["keyword"])

    def test_strict_execution_error_rejects_wrong_keyword_patch(self):
        class WrongKeywordRepairer:
            def propose(self, code, plan, report):
                return RepairPatch(
                    strategy="wrong_keyword_stub",
                    instruction="Remove the wrong keyword.",
                    target_error_codes=report.error_codes,
                    repair_plan=RepairPlan(
                        repair_level=1,
                        scope="local_patch",
                        target_error_codes=report.error_codes,
                        allowed_edits=("runtime kwarg",),
                        forbidden_edits=("data logic",),
                        reason="Runtime keyword patch only.",
                    ),
                    patch_ops=(
                        PatchOperation(
                            op="remove_keyword_arg",
                            anchor=PatchAnchor(kind="method_call", name="add", occurrence=1),
                            keyword="color",
                            description="This should be rejected because the exception named trunkcolor.",
                        ),
                    ),
                )

        case = ChartCase(
            case_id="repair-runtime-kwarg-strict-wrong-keyword",
            query="Show a Sankey diagram from source to target.",
            schema=TableSchema(columns={"source": "str", "target": "str"}),
            rows=(
                {"source": "source", "target": "target"},
                {"source": "source", "target": "target"},
            ),
            generated_code="""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey

df = pd.DataFrame(rows)
link_data = df.groupby(['source', 'target']).size().reset_index(name='weight')
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
sankey = Sankey(ax=ax, scale=0.5, offset=0.0)
first_row = True
for _, row in link_data.iterrows():
    if first_row:
        sankey.add(
            flows=[row['weight'], -row['weight']],
            labels=[row['source'], row['target']],
            orientations=[0, 0],
            trunklength=1,
            trunkcolor='red',
            patchlabel=row['source'],
            pathlengths=[0.5, 0.5],
            color='blue'
        )
        first_row = False
sankey.finish()
plt.title("Sankey Diagram: Flow from source to target")
""",
            figure_requirements=FigureRequirementSpec(
                axes_count=1,
                axes=(
                    AxisRequirementSpec(
                        axis_index=0,
                        title="Sankey Diagram: Flow from source to target",
                        text_contains=("source", "target"),
                        artist_types=("patch",),
                    ),
                ),
            ),
            verification_mode="figure_only",
        )
        pipeline = GroundedChartPipeline(
            parser=HeuristicIntentParser(),
            repairer=WrongKeywordRepairer(),
            enable_bounded_repair_loop=True,
            max_repair_rounds=1,
            repair_policy_mode="strict",
        )

        result = InMemoryCaseAdapter([case]).run_batch(pipeline, continue_on_error=True).report.cases[0]

        self.assertEqual("failed", result.status)
        self.assertEqual("stop", result.repair_loop_status)
        self.assertIn("whitelist rejected", result.repair_loop_reason)
        self.assertFalse(result.repair_attempts[0]["applied"])

    def test_bounded_repair_loop_reuses_initial_patch_without_duplicate_propose(self):
        class CountingRepairer:
            def __init__(self) -> None:
                self.calls = 0

            def propose(self, code, plan, report, escalation_scope=None):
                self.calls += 1
                return RepairPatch(
                    strategy="counting_stub",
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

        repairer = CountingRepairer()
        case = ChartCase(
            case_id="repair-title-reuse",
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
            repairer=repairer,
            enable_bounded_repair_loop=True,
            max_repair_rounds=2,
        )

        result = InMemoryCaseAdapter([case]).run(pipeline)[0].pipeline_result

        self.assertTrue(result.report.ok)
        self.assertEqual(1, repairer.calls)
        self.assertEqual("stop", result.repair_attempts[0].decision_action)
        self.assertEqual("Verifier passed after repair.", result.repair_attempts[0].decision_reason)

    def test_bounded_repair_loop_escalates_after_no_progress(self):
        class EscalatingRepairer:
            def __init__(self) -> None:
                self.calls = 0
                self.scopes = []

            def propose(self, code, plan, report, escalation_scope=None):
                self.calls += 1
                self.scopes.append(escalation_scope)
                if escalation_scope == "structural_regeneration":
                    return RepairPatch(
                        strategy="escalating_stub",
                        instruction="Regenerate the plotting code.",
                        target_error_codes=report.error_codes,
                        repair_plan=RepairPlan(
                            repair_level=3,
                            scope="structural_regeneration",
                            target_error_codes=report.error_codes,
                            allowed_edits=("full plotting code",),
                            forbidden_edits=("inventing requirements",),
                            reason="Escalated regeneration.",
                        ),
                        repaired_code="""
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.bar(["A", "B"], [10, 7])
ax.set_title("Sales by Category")
ax.set_xlabel("Category")
ax.set_ylabel("Sales")
""",
                        loop_signal="continue",
                        loop_reason="After escalation, this broader rewrite should be re-verified once.",
                    )
                return RepairPatch(
                    strategy="escalating_stub",
                    instruction="Try another title patch first.",
                    target_error_codes=report.error_codes,
                    repair_plan=RepairPlan(
                        repair_level=1,
                        scope="local_patch",
                        target_error_codes=report.error_codes,
                        allowed_edits=("title call",),
                        forbidden_edits=("data logic",),
                        reason="Local patch first.",
                    ),
                    repaired_code="""
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.bar(["A", "B"], [10, 7])
ax.set_title("Still Wrong")
ax.set_xlabel("Category")
ax.set_ylabel("Sales")
""",
                    loop_signal="continue",
                    loop_reason="Try one scoped fix first; escalate if this still does not improve anything.",
                )

        repairer = EscalatingRepairer()
        case = ChartCase(
            case_id="repair-title-escalate",
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
            repairer=repairer,
            enable_bounded_repair_loop=True,
            max_repair_rounds=3,
        )

        result = InMemoryCaseAdapter([case]).run(pipeline)[0].pipeline_result

        self.assertTrue(result.report.ok)
        self.assertEqual(2, repairer.calls)
        self.assertEqual([None, "structural_regeneration"], repairer.scopes)
        self.assertEqual("escalate", result.repair_attempts[0].decision_action)
        self.assertEqual("structural_regeneration", result.repair_attempts[0].decision_next_scope)
        self.assertEqual("stop", result.repair_attempts[1].decision_action)

    def test_execution_error_after_local_patch_gets_one_more_local_round_before_escalation(self):
        class RuntimeRetryRepairer:
            def __init__(self) -> None:
                self.calls = 0
                self.scopes = []

            def propose(self, code, plan, report, escalation_scope=None):
                self.calls += 1
                self.scopes.append(escalation_scope)
                if self.calls == 1:
                    return RepairPatch(
                        strategy="runtime_retry_stub",
                        instruction="First localized fix.",
                        target_error_codes=report.error_codes,
                        repair_plan=RepairPlan(
                            repair_level=1,
                            scope="local_patch",
                            target_error_codes=report.error_codes,
                            allowed_edits=("title call",),
                            forbidden_edits=("data logic",),
                            reason="local retry",
                        ),
                        repaired_code="""
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.bar(["A", "B"], [10, 7])
ax.set_title("Sales by Category")
ax.set_xlabel("Category")
ax.set_ylabel("Sales")
raise RuntimeError("boom")
""",
                    )
                return RepairPatch(
                    strategy="runtime_retry_stub",
                    instruction="Second localized fix.",
                    target_error_codes=report.error_codes,
                    repair_plan=RepairPlan(
                        repair_level=1,
                        scope="local_patch",
                        target_error_codes=report.error_codes,
                        allowed_edits=("title call",),
                        forbidden_edits=("data logic",),
                        reason="local retry",
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

        repairer = RuntimeRetryRepairer()
        case = ChartCase(
            case_id="repair-runtime-retry-local",
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
            repairer=repairer,
            enable_bounded_repair_loop=True,
            max_repair_rounds=3,
        )

        result = InMemoryCaseAdapter([case]).run(pipeline)[0].pipeline_result

        self.assertTrue(result.report.ok)
        self.assertEqual(2, repairer.calls)
        self.assertEqual([None, None], repairer.scopes)
        self.assertEqual("continue", result.repair_attempts[0].decision_action)
        self.assertEqual("stop", result.repair_attempts[1].decision_action)

    def test_strict_policy_blocks_diagnose_only_case_before_repair_loop(self):
        case = ChartCase(
            case_id="repair-title-blocked",
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
            metadata={
                "repairability": "diagnose_only",
                "expected_improvement": "no_auto_fix",
            },
        )
        pipeline = GroundedChartPipeline(
            parser=HeuristicIntentParser(),
            repairer=RuleBasedRepairer(),
            repair_policy_mode="strict",
            enable_bounded_repair_loop=True,
            max_repair_rounds=2,
        )

        result = InMemoryCaseAdapter([case]).run(pipeline)[0].pipeline_result

        self.assertFalse(result.report.ok)
        self.assertEqual("policy_gate_abstain", result.repair.strategy)
        self.assertEqual("policy_blocked", result.repair_plan.scope)
        self.assertEqual(4, result.repair_plan.repair_level)
        self.assertFalse(result.repair_attempts)

    def test_exploratory_policy_allows_same_case_to_repair(self):
        case = ChartCase(
            case_id="repair-title-exploratory",
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
            metadata={
                "repairability": "diagnose_only",
                "expected_improvement": "no_auto_fix",
            },
        )
        pipeline = GroundedChartPipeline(
            parser=HeuristicIntentParser(),
            repairer=RuleBasedRepairer(),
            repair_policy_mode="exploratory",
            enable_bounded_repair_loop=True,
            max_repair_rounds=2,
        )

        result = InMemoryCaseAdapter([case]).run(pipeline)[0].pipeline_result

        self.assertTrue(result.report.ok)
        self.assertEqual("local_patch", result.repair_plan.scope)
        self.assertTrue(result.repair_attempts)

    def test_bounded_repair_loop_fixes_tick_labels(self):
        case = ChartCase(
            case_id="repair-ticks",
            query="Show a 3D bar chart.",
            schema=TableSchema(columns={}),
            rows=(),
            generated_code="""
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot([0, 1], [0, 1], [2, 3])
ax.set_yticks([0, 1])
ax.set_yticklabels(['k=0', 'k=1'])
""",
            figure_requirements=FigureRequirementSpec(
                axes_count=1,
                axes=(
                    AxisRequirementSpec(
                        axis_index=0,
                        ytick_labels=("4", "3"),
                    ),
                ),
            ),
            verification_mode="figure_only",
        )
        pipeline = GroundedChartPipeline(
            parser=HeuristicIntentParser(),
            repairer=RuleBasedRepairer(),
            enable_bounded_repair_loop=True,
            max_repair_rounds=2,
        )

        result = InMemoryCaseAdapter([case]).run(pipeline)[0].pipeline_result

        self.assertTrue(result.report.ok)
        self.assertIn("set_yticklabels(['4', '3'])", result.repaired_code)

    def test_bounded_repair_loop_fixes_plotly_title_and_annotation(self):
        case = ChartCase(
            case_id="repair-plotly-title",
            query="Create a Plotly sunburst chart with a subtitle.",
            schema=TableSchema(columns={}),
            rows=(),
            generated_code="""
import pandas as pd
import plotly.express as px
import plotly.io as pio
df = pd.DataFrame([
    {'Major Area': 'Asia', 'Regions': 'East', 'Country': 'China', 'Overall score': 70},
    {'Major Area': 'Europe', 'Regions': 'West', 'Country': 'France', 'Overall score': 80}
])
fig = px.sunburst(df, path=['Major Area', 'Regions', 'Country'], values='Overall score')
fig.update_layout(
    title={
        'text': 'Wrong Title',
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    annotations=[
        dict(
            text='Wrong Subtitle',
            x=0.5,
            y=0.02,
            xref='paper',
            yref='paper',
            showarrow=False
        )
    ]
)
pio.write_image(fig, 'novice_final.png', width=1000, height=1000)
""",
            figure_requirements=FigureRequirementSpec(
                axes_count=1,
                axes=(
                    AxisRequirementSpec(
                        axis_index=0,
                        title="Global Food Security Index, 2020",
                        projection="plotly",
                        text_contains=("Overall score 0-100, 100 = best environment",),
                    ),
                ),
            ),
            verification_mode="figure_only",
        )
        pipeline = GroundedChartPipeline(
            parser=HeuristicIntentParser(),
            repairer=RuleBasedRepairer(),
            enable_bounded_repair_loop=True,
            max_repair_rounds=2,
        )

        result = InMemoryCaseAdapter([case]).run(pipeline)[0].pipeline_result

        self.assertTrue(result.report.ok)
        self.assertIn("Global Food Security Index, 2020", result.repaired_code)
        self.assertIn("Overall score 0-100, 100 = best environment", result.repaired_code)

    def test_bounded_repair_loop_recovers_from_unsupported_keyword_argument(self):
        case = ChartCase(
            case_id="repair-runtime-kwarg",
            query="Show a Sankey diagram from source to target.",
            schema=TableSchema(columns={"source": "str", "target": "str"}),
            rows=(
                {"source": "source", "target": "target"},
                {"source": "source", "target": "target"},
            ),
            generated_code="""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey

df = pd.DataFrame(rows)
link_data = df.groupby(['source', 'target']).size().reset_index(name='weight')
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
sankey = Sankey(ax=ax, scale=0.5, offset=0.0)
first_row = True
for _, row in link_data.iterrows():
    if first_row:
        sankey.add(
            flows=[row['weight'], -row['weight']],
            labels=[row['source'], row['target']],
            orientations=[0, 0],
            trunklength=1,
            trunkcolor='red',
            patchlabel=row['source'],
            pathlengths=[0.5, 0.5],
            color='blue'
        )
        first_row = False
sankey.finish()
plt.title("Sankey Diagram: Flow from source to target")
""",
            figure_requirements=FigureRequirementSpec(
                axes_count=1,
                axes=(
                    AxisRequirementSpec(
                        axis_index=0,
                        title="Sankey Diagram: Flow from source to target",
                        text_contains=("source", "target"),
                        artist_types=("patch",),
                    ),
                ),
            ),
            verification_mode="figure_only",
        )
        pipeline = GroundedChartPipeline(
            parser=HeuristicIntentParser(),
            repairer=RuleBasedRepairer(),
            enable_bounded_repair_loop=True,
            max_repair_rounds=2,
        )

        batch = InMemoryCaseAdapter([case]).run_batch(pipeline)
        report = batch.report.cases[0]

        self.assertEqual(report.status, "passed")
        self.assertEqual(report.exception_type, "AttributeError")
        self.assertIn("trunkcolor", report.exception_message)
        self.assertTrue(report.repair_attempts)
        self.assertEqual(1, len(report.repair_attempts))
        self.assertIn("patchlabel=row['source']", report.repaired_code)
        self.assertNotIn("trunkcolor=", report.repaired_code)

    def test_bounded_repair_loop_appends_missing_text_even_if_token_exists_in_code(self):
        case = ChartCase(
            case_id="repair-missing-text-token-collision",
            query="Show a Sankey diagram from source to target.",
            schema=TableSchema(columns={"source": "str", "target": "str"}),
            rows=(
                {"source": "A", "target": "B"},
                {"source": "A", "target": "C"},
            ),
            generated_code="""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey

df = pd.DataFrame(rows)
link_data = df.groupby(['source', 'target']).size().reset_index(name='weight')
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
sankey = Sankey(ax=ax, scale=0.5, offset=0.0)
for _, row in link_data.iterrows():
    sankey.add(
        flows=[row['weight'], -row['weight']],
        labels=[row['source'], row['target']],
        orientations=[0, 0],
        trunklength=1,
        patchlabel=row['source'],
        pathlengths=[0.5, 0.5],
        color='blue'
    )
sankey.finish()
plt.title("Sankey Diagram: Flow from source to target")
""",
            figure_requirements=FigureRequirementSpec(
                axes_count=1,
                axes=(
                    AxisRequirementSpec(
                        axis_index=0,
                        title="Sankey Diagram: Flow from source to target",
                        text_contains=("source", "target"),
                        artist_types=("patch",),
                    ),
                ),
            ),
            verification_mode="figure_only",
        )
        pipeline = GroundedChartPipeline(
            parser=HeuristicIntentParser(),
            repairer=RuleBasedRepairer(),
            enable_bounded_repair_loop=True,
            max_repair_rounds=2,
        )

        result = InMemoryCaseAdapter([case]).run(pipeline)[0].pipeline_result

        self.assertTrue(result.report.ok)
        self.assertIn("plt.gca().text(0.5, 0.5, 'source'", result.repaired_code)
        self.assertIn("plt.gca().text(0.5, 0.5, 'target'", result.repaired_code)

    def test_strict_local_patch_accepts_sort_keyword_patch(self):
        class SortPatchRepairer:
            def propose(self, code, plan, report, escalation_scope=None, evidence_graph=None):
                return RepairPatch(
                    strategy="sort_patch_stub",
                    instruction="Change sorted reverse flag only.",
                    target_error_codes=report.error_codes,
                    repair_plan=RepairPlan(
                        repair_level=1,
                        scope="local_patch",
                        target_error_codes=report.error_codes,
                        allowed_edits=("sort call",),
                        forbidden_edits=("data loading",),
                        reason="Only sort order failed.",
                    ),
                    patch_ops=(
                        PatchOperation(
                            op="replace_keyword_arg",
                            anchor=PatchAnchor(kind="function_call", name="sorted", occurrence=1),
                            keyword="reverse",
                            new_value=False,
                        ),
                    ),
                )

        case = ChartCase(
            case_id="repair-sort-strict",
            query="Show total sales by category in a bar chart, ascending.",
            schema=TableSchema(columns={"category": "str", "sales": "number"}),
            rows=(
                {"category": "A", "sales": 10},
                {"category": "B", "sales": 7},
            ),
            generated_code="""
import matplotlib.pyplot as plt
totals = {}
for row in rows:
    totals[row["category"]] = totals.get(row["category"], 0) + row["sales"]
items = sorted(totals.items(), key=lambda item: item[1], reverse=True)
plt.bar([key for key, value in items], [value for key, value in items])
""",
        )
        pipeline = GroundedChartPipeline(
            parser=HeuristicIntentParser(),
            repairer=SortPatchRepairer(),
            enable_bounded_repair_loop=True,
            max_repair_rounds=1,
            repair_policy_mode="strict",
        )

        result = InMemoryCaseAdapter([case]).run(pipeline)[0].pipeline_result

        self.assertTrue(result.report.ok)
        self.assertEqual("stop", result.repair_loop_status)
        self.assertEqual("Verifier passed after repair.", result.repair_loop_reason)
        self.assertEqual(1, len(result.repair_attempts))
        self.assertTrue(result.repair_attempts[0].applied)
    def test_repaired_code_preservation_gate_rejects_chart_constructor_change(self):
        reason = _repaired_code_preservation_rejection(
            current_code="fig = px.sunburst(df, path=['Major Area', 'Regions'])\n",
            repaired_code="fig = px.bar(df, x='Major Area', y='Overall score')\n",
            target_error_codes=("wrong_axis_title", "missing_annotation_text"),
        )

        self.assertIsNotNone(reason)
        self.assertIn("changed chart constructors", reason)

    def test_repaired_code_preservation_gate_allows_structural_chart_change(self):
        reason = _repaired_code_preservation_rejection(
            current_code="fig = px.sunburst(df, path=['Major Area', 'Regions'])\n",
            repaired_code="fig = px.bar(df, x='Major Area', y='Overall score')\n",
            target_error_codes=("missing_artist_type",),
        )

        self.assertIsNone(reason)


    def test_strict_local_patch_accepts_plotly_layout_title_annotation_ops(self):
        reason = _validate_strict_local_patch_ops(
            report_errors=(),
            target_error_codes=("wrong_axis_title", "missing_annotation_text"),
            patch_ops=(
                PatchOperation(
                    op="replace_keyword_arg",
                    anchor=PatchAnchor(kind="method_call", name="update_layout", occurrence=1),
                    keyword="title",
                    new_value={"text": "Global Food Security Index, 2020"},
                ),
                PatchOperation(
                    op="replace_keyword_arg",
                    anchor=PatchAnchor(kind="method_call", name="update_layout", occurrence=1),
                    keyword="annotations",
                    new_value=[{"text": "Overall score 0-100, 100 = best environment"}],
                ),
            ),
        )

        self.assertIsNone(reason)

    def test_local_patch_budget_scales_with_error_count(self):
        self.assertEqual(2, _patch_op_budget("local_patch"))
        self.assertEqual(2, _patch_op_budget("local_patch", error_count=1))
        self.assertEqual(4, _patch_op_budget("local_patch", error_count=4))
        self.assertEqual(8, _patch_op_budget("local_patch", error_count=20))
        self.assertEqual(3, _patch_op_budget("data_transformation", error_count=10))

    def test_strict_local_patch_accepts_legend_annotation_projection_ops(self):
        reason = _validate_strict_local_patch_ops(
            report_errors=(),
            target_error_codes=("missing_legend_label", "missing_annotation_text", "wrong_projection"),
            patch_ops=(
                PatchOperation(
                    op="replace_keyword_arg",
                    anchor=PatchAnchor(kind="method_call", name="bar", occurrence=1),
                    keyword="label",
                    new_value="Sales",
                ),
                PatchOperation(
                    op="insert_after_anchor",
                    anchor=PatchAnchor(kind="method_call", name="bar", occurrence=1),
                    new_value="ax.legend()\nax.annotate('lowest', xy=('B', 7))",
                ),
                PatchOperation(
                    op="replace_keyword_arg",
                    anchor=PatchAnchor(kind="function_call", name="subplots", occurrence=1),
                    keyword="subplot_kw",
                    new_value={"projection": "rectilinear"},
                ),
            ),
        )

        self.assertIsNone(reason)

if __name__ == "__main__":
    unittest.main()
