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


if __name__ == "__main__":
    unittest.main()
