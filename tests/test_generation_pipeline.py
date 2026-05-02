import json
import csv
import tempfile
import unittest
from pathlib import Path

from grounded_chart import (
    AxisRequirementSpec,
    ChartCodeGeneration,
    ChartConstructionPlan,
    ChartGenerationPipeline,
    PatchAnchor,
    PatchOperation,
    ChartRenderResult,
    FigureRequirementSpec,
    FigureAudit,
    GroundedChartPipeline,
    HeuristicIntentParser,
    LLMChartCodeGenerator,
    LayoutCritique,
    PlanAgentResult,
    RepairPatch,
    StaticChartCodeGenerator,
    TableSchema,
    VisualLayerPlan,
    VisualPanelPlan,
)
from grounded_chart.codegen import ChartCodeGenerationRequest
from grounded_chart.llm import LLMCompletionTrace, LLMJsonResult


STATIC_BAR_CODE = """
import matplotlib.pyplot as plt

labels = [row["product"] for row in rows]
values = [row["sales"] for row in rows]
fig, ax = plt.subplots(figsize=(4, 3))
ax.bar(labels, values)
ax.set_xlabel("product")
ax.set_ylabel("sales")
ax.set_title("Sales by product")
fig.savefig(OUTPUT_PATH, bbox_inches="tight")
""".strip()


WRONG_TITLE_CODE = STATIC_BAR_CODE.replace("Sales by product", "Wrong title")
RENDER_FAILURE_CODE = f"{STATIC_BAR_CODE}\n# bad_bbox_marker"

class ChartGenerationPipelineTest(unittest.TestCase):
    def test_static_generation_pipeline_writes_code_report_and_image(self):
        pipeline = ChartGenerationPipeline(
            code_generator=StaticChartCodeGenerator(STATIC_BAR_CODE),
            verifier_pipeline=GroundedChartPipeline(parser=HeuristicIntentParser()),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = pipeline.run(
                query="Create a bar chart of sales by product titled Sales by product.",
                schema=TableSchema(columns={"product": "string", "sales": "number"}),
                rows=(
                    {"product": "A", "sales": 3},
                    {"product": "B", "sales": 5},
                ),
                output_dir=Path(tmpdir),
                case_id="generation_smoke",
            )

            self.assertTrue(result.pipeline_result.report.ok)
            self.assertTrue(result.render_result.ok)
            self.assertIsNotNone(result.image_path)
            self.assertTrue(result.image_path.exists())
            self.assertTrue(result.initial_code_path.exists())
            self.assertTrue(result.final_code_path.exists())
            self.assertTrue(result.report_path.exists())
            self.assertTrue(result.manifest_path.exists())

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual("generation_smoke", manifest["case_id"])
            self.assertTrue(manifest["render"]["ok"])
            self.assertTrue(manifest["verification"]["ok"])
            self.assertEqual(str(result.image_path), manifest["image_path"])
            self.assertEqual("initial_code", manifest["verification"]["final_code_source"])
            self.assertTrue(manifest["verification"]["final_code_verified"])

    def test_renderer_receives_artifact_workspace_globals(self):
        class CapturingRenderer:
            def __init__(self):
                self.globals_dict = None

            def render(self, code, *, rows, output_dir, output_filename="figure.png", file_path=None, globals_dict=None):
                self.globals_dict = globals_dict
                output_path = Path(output_dir) / output_filename
                output_path.write_bytes(b"fake-png")
                return ChartRenderResult(
                    ok=True,
                    image_path=output_path,
                    artifact_paths=(output_path,),
                    backend="matplotlib",
                    metadata={"output_path": str(output_path)},
                )

        renderer = CapturingRenderer()
        pipeline = ChartGenerationPipeline(
            code_generator=StaticChartCodeGenerator(STATIC_BAR_CODE),
            verifier_pipeline=GroundedChartPipeline(parser=HeuristicIntentParser()),
            renderer=renderer,
        )

        with tempfile.TemporaryDirectory() as output_tmp:
            result = pipeline.run(
                query="Create a bar chart of sales by product titled Sales by product.",
                schema=TableSchema(columns={"product": "string", "sales": "number"}),
                rows=(
                    {"product": "A", "sales": 3},
                    {"product": "B", "sales": 5},
                ),
                output_dir=Path(output_tmp),
                case_id="renderer_globals",
            )

            self.assertTrue(result.render_result.ok)
            self.assertIsNotNone(renderer.globals_dict)
            self.assertIn("artifact_workspace", renderer.globals_dict)
            self.assertEqual(str(Path(output_tmp).resolve()), renderer.globals_dict["artifact_workspace"]["root"])

    def test_one_shot_repaired_code_is_reverified_before_final_render(self):
        class OneShotRepairer:
            def __init__(self):
                self.calls = 0

            def propose(self, code, plan, report, **kwargs):
                self.calls += 1
                return RepairPatch(
                    strategy="fake_one_shot",
                    instruction="Fix the title.",
                    target_error_codes=report.error_codes,
                    repaired_code=STATIC_BAR_CODE,
                )

        repairer = OneShotRepairer()
        pipeline = ChartGenerationPipeline(
            code_generator=StaticChartCodeGenerator(WRONG_TITLE_CODE),
            verifier_pipeline=GroundedChartPipeline(
                parser=HeuristicIntentParser(),
                repairer=repairer,
                enable_bounded_repair_loop=False,
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = pipeline.run(
                query="Create a bar chart of sales by product titled Sales by product.",
                schema=TableSchema(columns={"product": "string", "sales": "number"}),
                rows=(
                    {"product": "A", "sales": 3},
                    {"product": "B", "sales": 5},
                ),
                output_dir=Path(tmpdir),
                case_id="one_shot_repair",
                expected_figure=FigureRequirementSpec(
                    axes=(AxisRequirementSpec(axis_index=0, title="Sales by product"),)
                ),
            )

            self.assertEqual(1, repairer.calls)
            self.assertTrue(result.pipeline_result.report.ok)
            self.assertTrue(result.render_result.ok)
            self.assertEqual(STATIC_BAR_CODE, result.final_code)
            self.assertEqual(STATIC_BAR_CODE, result.pipeline_result.repaired_code)
            self.assertTrue((Path(tmpdir) / "generated_repair_candidate.py").exists())

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual("one_shot_repair_candidate", manifest["verification"]["final_code_source"])
            self.assertTrue(manifest["metadata"]["one_shot_repair_accepted"])
            self.assertTrue(manifest["verification"]["final_code_verified"])

    def test_render_failure_is_repaired_and_reverified_before_acceptance(self):
        class FakeRenderFailureRenderer:
            def __init__(self):
                self.calls = 0

            def render(self, code, *, rows, output_dir, output_filename="figure.png", file_path=None, globals_dict=None):
                self.calls += 1
                output_path = Path(output_dir) / output_filename
                if "bad_bbox_marker" in code:
                    return ChartRenderResult(
                        ok=False,
                        backend="matplotlib",
                        exception_type="MemoryError",
                        exception_message="bad allocation",
                        metadata={"output_path": str(output_path)},
                    )
                output_path.write_bytes(b"fake-png")
                return ChartRenderResult(
                    ok=True,
                    image_path=output_path,
                    artifact_paths=(output_path,),
                    backend="matplotlib",
                    metadata={"output_path": str(output_path)},
                )

        class RenderRepairer:
            def __init__(self):
                self.calls = 0
                self.last_error_codes = None

            def propose(self, code, plan, report, **kwargs):
                self.calls += 1
                self.last_error_codes = report.error_codes
                return RepairPatch(
                    strategy="fake_render_repair",
                    instruction="Remove render-unsafe export behavior.",
                    target_error_codes=report.error_codes,
                    repaired_code=STATIC_BAR_CODE,
                )

        renderer = FakeRenderFailureRenderer()
        repairer = RenderRepairer()
        pipeline = ChartGenerationPipeline(
            code_generator=StaticChartCodeGenerator(RENDER_FAILURE_CODE),
            verifier_pipeline=GroundedChartPipeline(
                parser=HeuristicIntentParser(),
                repairer=repairer,
                enable_bounded_repair_loop=False,
            ),
            renderer=renderer,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = pipeline.run(
                query="Create a bar chart of sales by product titled Sales by product.",
                schema=TableSchema(columns={"product": "string", "sales": "number"}),
                rows=(
                    {"product": "A", "sales": 3},
                    {"product": "B", "sales": 5},
                ),
                output_dir=Path(tmpdir),
                case_id="render_repair",
            )

            self.assertEqual(1, repairer.calls)
            self.assertEqual(("execution_error",), repairer.last_error_codes)
            self.assertEqual(2, renderer.calls)
            self.assertTrue(result.pipeline_result.report.ok)
            self.assertTrue(result.render_result.ok)
            self.assertEqual(STATIC_BAR_CODE, result.final_code)
            self.assertTrue((Path(tmpdir) / "generated_render_repair_candidate.py").exists())

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual("render_repair_candidate", manifest["verification"]["final_code_source"])
            self.assertTrue(manifest["metadata"]["render_repair_attempted"])
            self.assertTrue(manifest["metadata"]["render_repair_accepted"])
            self.assertEqual("accepted", manifest["metadata"]["render_repair_reason"])
            self.assertEqual("MemoryError", manifest["metadata"]["render_repair_exception_type"])

    def test_render_failure_can_use_structured_patch_ops_as_candidate(self):
        class FakeRenderFailureRenderer:
            def __init__(self):
                self.calls = 0

            def render(self, code, *, rows, output_dir, output_filename="figure.png", file_path=None, globals_dict=None):
                self.calls += 1
                output_path = Path(output_dir) / output_filename
                if "bbox_inches=\"tight\"" in code:
                    return ChartRenderResult(ok=False, exception_type="MemoryError", exception_message="bad allocation")
                output_path.write_bytes(b"fake-png")
                return ChartRenderResult(ok=True, image_path=output_path, artifact_paths=(output_path,), backend="matplotlib")

        class PatchOpRepairer:
            def __init__(self):
                self.calls = 0

            def propose(self, code, plan, report, **kwargs):
                self.calls += 1
                return RepairPatch(
                    strategy="fake_patch_ops_render_repair",
                    instruction="Disable tight bbox export.",
                    target_error_codes=report.error_codes,
                    patch_ops=(
                        PatchOperation(
                            op="replace_keyword_arg",
                            anchor=PatchAnchor(kind="method_call", name="savefig", occurrence=1),
                            keyword="bbox_inches",
                            new_value=None,
                        ),
                    ),
                )

        renderer = FakeRenderFailureRenderer()
        repairer = PatchOpRepairer()
        pipeline = ChartGenerationPipeline(
            code_generator=StaticChartCodeGenerator(STATIC_BAR_CODE),
            verifier_pipeline=GroundedChartPipeline(
                parser=HeuristicIntentParser(),
                repairer=repairer,
                enable_bounded_repair_loop=False,
            ),
            renderer=renderer,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = pipeline.run(
                query="Create a bar chart of sales by product titled Sales by product.",
                schema=TableSchema(columns={"product": "string", "sales": "number"}),
                rows=(
                    {"product": "A", "sales": 3},
                    {"product": "B", "sales": 5},
                ),
                output_dir=Path(tmpdir),
                case_id="render_patch_ops_repair",
            )

            self.assertEqual(1, repairer.calls)
            self.assertEqual(2, renderer.calls)
            self.assertTrue(result.render_result.ok)
            self.assertTrue(result.pipeline_result.report.ok)
            self.assertIn("bbox_inches=None", result.final_code)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual("patch_ops", manifest["metadata"]["render_repair_candidate_source"])
        self.assertTrue(manifest["metadata"]["render_repair_patch_applied"])
        self.assertTrue(manifest["metadata"]["render_repair_accepted"])

    def test_generation_pipeline_builds_source_file_plan_before_codegen(self):
        class CapturingGenerator:
            def __init__(self):
                self.request = None

            def generate(self, request):
                self.request = request
                code = """
import pandas as pd
import matplotlib.pyplot as plt

prepared = pd.read_csv('ExecutorAgent/round_1/step_02_imports_waterfall_render_table.csv')
fig, ax = plt.subplots(figsize=(4, 3))
ax.bar(prepared['x_position'], prepared['bar_height'], bottom=prepared['bar_bottom'], width=prepared['bar_width'])
ax.set_title('Imports')
fig.savefig(OUTPUT_PATH, bbox_inches='tight')
""".strip()
                return ChartCodeGeneration(code=code, generator_name="capture")

        generator = CapturingGenerator()
        pipeline = ChartGenerationPipeline(
            code_generator=generator,
            verifier_pipeline=GroundedChartPipeline(parser=HeuristicIntentParser()),
        )

        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory() as output_tmp:
            source_root = Path(source_tmp)
            (source_root / "Imports.csv").write_text("Year,Urban,Rural\n2000,1,2\n2001,3,4\n", encoding="utf-8")
            result = pipeline.run(
                query="Use Imports.csv to create a waterfall chart of Urban and Rural imports over Year.",
                schema=TableSchema(columns={}),
                rows=(),
                output_dir=Path(output_tmp),
                case_id="source_file_generation",
                source_workspace=source_root,
                verification_mode="figure_only",
            )

            self.assertTrue((Path(output_tmp) / "Imports.csv").exists())
            self.assertIsNotNone(generator.request)
            self.assertEqual("source_file_grounded", generator.request.generation_mode)
            context = generator.request.context
            self.assertEqual(["Imports.csv"], [item["name"] for item in context["source_data_plan"]["files"]])
            self.assertEqual(["Year", "Urban", "Rural"], context["source_data_execution"]["loaded_tables"][0]["columns"])
            self.assertIn("construction_plan", context)
            self.assertEqual("chart_construction_plan_v2", context["construction_plan"]["plan_type"])
            self.assertTrue(result.render_result.ok)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(["Imports.csv"], manifest["metadata"]["source_data_files"])
            self.assertEqual("chart_construction_plan_v2", manifest["metadata"]["construction_plan_type"])
            self.assertTrue(manifest["metadata"]["plan_valid_ok"])
            self.assertTrue((Path(output_tmp) / "construction_plan_validation_report.json").exists())
            self.assertTrue(manifest["metadata"]["artifact_workspace_ok"])
            self.assertTrue((Path(output_tmp) / "artifact_workspace_manifest.json").exists())
            self.assertTrue((Path(output_tmp) / "PlanAgent" / "round_1" / "plan.json").exists())
            self.assertTrue((Path(output_tmp) / "PlanAgent" / "round_1" / "plan.md").exists())
            self.assertTrue((Path(output_tmp) / "ExecutorAgent" / "round_1" / "steps.md").exists())
            self.assertTrue((Path(output_tmp) / "ExecutorAgent" / "round_1" / "step_01_sources_summary.json").exists())
            self.assertTrue((Path(output_tmp) / "ExecutorAgent" / "round_1" / "chart_protocols" / "waterfall_protocol.json").exists())
            self.assertTrue((Path(output_tmp) / "ExecutorAgent" / "round_1" / "step_02_imports_waterfall_render_table.csv").exists())
            artifact_workspace = json.loads((Path(output_tmp) / "artifact_workspace_manifest.json").read_text(encoding="utf-8"))
            render_artifact = next(
                item
                for item in artifact_workspace["artifacts"]
                if item["name"] == "step_02_imports_waterfall_render_table.csv"
            )
            self.assertIn("schema", render_artifact)
            self.assertIn("fill_color_role", render_artifact["schema"]["columns"])
            self.assertTrue(manifest["metadata"]["executor_fidelity_ok"])
            self.assertTrue((Path(output_tmp) / "executor_fidelity_report.json").exists())

    def test_generation_pipeline_writes_prepared_import_artifact_without_diffing_source_values(self):
        class CapturingGenerator:
            def generate(self, request):
                code = """
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('ExecutorAgent/round_1/step_02_imports_waterfall_render_table.csv')
fig, ax = plt.subplots(figsize=(4, 3))
ax.bar(df['x_position'], df['bar_height'], bottom=df['bar_bottom'], width=df['bar_width'])
ax.set_title('Imports')
fig.savefig(OUTPUT_PATH, bbox_inches='tight')
""".strip()
                return ChartCodeGeneration(code=code, generator_name="capture")

        pipeline = ChartGenerationPipeline(
            code_generator=CapturingGenerator(),
            verifier_pipeline=GroundedChartPipeline(parser=HeuristicIntentParser()),
        )

        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory() as output_tmp:
            source_root = Path(source_tmp)
            (source_root / "Imports.csv").write_text(
                "Year,Urban,Rural\n2000,10,20\n2001,1,-2\n2002,30,18\n",
                encoding="utf-8",
            )
            result = pipeline.run(
                query="Create a waterfall chart from Imports.csv with Urban and Rural yearly changes.",
                schema=TableSchema(columns={}),
                rows=(),
                output_dir=Path(output_tmp),
                case_id="source_file_waterfall_generation",
                source_workspace=source_root,
                verification_mode="figure_only",
            )

            artifact_path = Path(output_tmp) / "ExecutorAgent" / "round_1" / "step_02_imports_waterfall_values.csv"
            self.assertTrue(result.render_result.ok)
            with artifact_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            row_2001 = next(row for row in rows if row["Year"] == "2001")
            self.assertEqual("direct_use_source_values", row_2001["transform"])
            self.assertEqual("1", row_2001["Urban"])
            self.assertEqual("1", row_2001["Urban_plot"])
            self.assertEqual("-2", row_2001["Rural"])
            self.assertEqual("-2", row_2001["Rural_plot"])

            render_artifact_path = Path(output_tmp) / "ExecutorAgent" / "round_1" / "step_02_imports_waterfall_render_table.csv"
            with render_artifact_path.open("r", encoding="utf-8", newline="") as handle:
                render_rows = list(csv.DictReader(handle))
            urban_2001 = next(row for row in render_rows if row["Year"] == "2001" and row["series"] == "Urban")
            rural_2001 = next(row for row in render_rows if row["Year"] == "2001" and row["series"] == "Rural")
            self.assertEqual("delta", urban_2001["role"])
            self.assertEqual("10.0", urban_2001["bar_bottom"])
            self.assertEqual("1.0", urban_2001["bar_height"])
            self.assertEqual("11.0", urban_2001["bar_top"])
            self.assertEqual("20.0", rural_2001["bar_bottom"])
            self.assertEqual("-2.0", rural_2001["bar_height"])
            self.assertEqual("18.0", rural_2001["bar_top"])
            self.assertEqual("increase", urban_2001["change_role"])
            self.assertEqual("Urban", urban_2001["fill_color_role"])
            self.assertEqual("Rural", rural_2001["fill_color_role"])

    def test_artifact_compiler_uses_layer_schema_not_fixed_import_columns(self):
        class CapturingGenerator:
            def __init__(self):
                self.request = None

            def generate(self, request):
                self.request = request
                artifacts = request.context["artifact_workspace"]["artifacts"]
                geometry = next(
                    item
                    for item in artifacts
                    if item.get("artifact_role") == "waterfall_geometry"
                    and item.get("chart_type") == "waterfall"
                    and not item.get("legacy_alias")
                )
                code = f"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv({geometry["relative_path"]!r})
fig, ax = plt.subplots(figsize=(4, 3))
ax.bar(df['x_position'], df['bar_height'], bottom=df['bar_bottom'], width=df['bar_width'])
ax.set_title('Trade')
fig.savefig(OUTPUT_PATH, bbox_inches='tight')
""".strip()
                return ChartCodeGeneration(code=code, generator_name="capture")

        generator = CapturingGenerator()
        pipeline = ChartGenerationPipeline(
            code_generator=generator,
            verifier_pipeline=GroundedChartPipeline(parser=HeuristicIntentParser()),
        )

        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory() as output_tmp:
            source_root = Path(source_tmp)
            (source_root / "TradeFlow.csv").write_text(
                "Year,City,Suburb\n2020,5,7\n2021,-2,3\n2022,9,11\n",
                encoding="utf-8",
            )
            result = pipeline.run(
                query="Use TradeFlow.csv to create a waterfall chart of City and Suburb values over Year.",
                schema=TableSchema(columns={}),
                rows=(),
                output_dir=Path(output_tmp),
                case_id="generic_waterfall_schema",
                source_workspace=source_root,
                verification_mode="figure_only",
            )

            self.assertTrue(result.render_result.ok)
            workspace = json.loads((Path(output_tmp) / "artifact_workspace_manifest.json").read_text(encoding="utf-8"))
            geometry = next(
                item
                for item in workspace["artifacts"]
                if item.get("artifact_role") == "waterfall_geometry" and not item.get("legacy_alias")
            )
            self.assertEqual("TradeFlow.csv", geometry["source_table"])
            self.assertEqual(["City", "Suburb"], geometry["series_columns"])
            with (Path(output_tmp) / geometry["relative_path"]).open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            city_2021 = next(row for row in rows if row["Year"] == "2021" and row["series"] == "City")
            suburb_2021 = next(row for row in rows if row["Year"] == "2021" and row["series"] == "Suburb")
            self.assertEqual("-2.0", city_2021["bar_height"])
            self.assertEqual("City", city_2021["fill_color_role"])
            self.assertEqual("Suburb", suburb_2021["fill_color_role"])
            self.assertNotIn("Urban", geometry["schema"]["columns"])
            self.assertTrue(json.loads((Path(output_tmp) / "executor_fidelity_report.json").read_text(encoding="utf-8"))["ok"])

    def test_generation_pipeline_preserves_area_overlap_modifier_from_instruction_evidence(self):
        class CapturingGenerator:
            def __init__(self):
                self.request = None

            def generate(self, request):
                self.request = request
                code = """
import pandas as pd
import matplotlib.pyplot as plt

area = pd.read_csv('ExecutorAgent/round_1/step_03_consumption_area_values.csv')
fig, ax = plt.subplots(figsize=(4, 3))
ax.fill_between(area['x_index'], area['Urban_fill_bottom'], area['Urban_fill_top'])
ax.fill_between(area['x_index'], area['Rural_fill_bottom'], area['Rural_fill_top'])
ax.set_title('Consumption')
fig.savefig(OUTPUT_PATH, bbox_inches='tight')
""".strip()
                return ChartCodeGeneration(code=code, generator_name="capture")

        generator = CapturingGenerator()
        pipeline = ChartGenerationPipeline(
            code_generator=generator,
            verifier_pipeline=GroundedChartPipeline(parser=HeuristicIntentParser()),
        )
        simple = "Plot a stacked area chart for consumption data on a secondary y-axis."
        expert = (
            "Plot urban and rural consumption data on a secondary y-axis. "
            "Use translucent colors for urban and rural areas to indicate overlapping consumption. "
            "Set y-axis scale from 35 to 105 kg."
        )

        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory() as output_tmp:
            source_root = Path(source_tmp)
            (source_root / "Consumption.csv").write_text(
                "Year,Urban,Rural\n2000,35,100\n2001,40,90\n",
                encoding="utf-8",
            )
            result = pipeline.run(
                query=simple,
                schema=TableSchema(columns={}),
                rows=(),
                output_dir=Path(output_tmp),
                case_id="area_modifier_generation",
                source_workspace=source_root,
                verification_mode="figure_only",
                generation_context={"expert_instruction": expert},
            )

            area_layer = next(
                layer
                for panel in generator.request.context["construction_plan"]["panels"]
                for layer in panel["layers"]
                if layer["chart_type"] == "area"
            )
            self.assertEqual("overlap", area_layer["semantic_modifiers"]["composition"])
            self.assertEqual({"type": "explicit_range", "min": 35.0, "max": 105.0}, area_layer["semantic_modifiers"]["scale_policy"])
            artifact_path = Path(output_tmp) / "ExecutorAgent" / "round_1" / "step_03_consumption_area_values.csv"
            with artifact_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual("overlap", rows[0]["composition_policy"])
            self.assertEqual("35.0", rows[0]["Urban_fill_bottom"])
            self.assertEqual("35.0", rows[0]["Rural_fill_bottom"])
            self.assertEqual("100.0", rows[0]["Rural_fill_top"])
            self.assertTrue(result.render_result.ok)

    def test_layout_replanning_sends_feedback_to_plan_agent_and_reruns_codegen_without_repair_attempt(self):
        class SequenceGenerator:
            def __init__(self):
                self.requests = []

            def generate(self, request):
                self.requests.append(request)
                title = "Initial layout" if len(self.requests) == 1 else "Replanned layout"
                code = f"""
import matplotlib.pyplot as plt

labels = [row["product"] for row in rows]
values = [row["sales"] for row in rows]
fig, ax = plt.subplots(figsize=(4, 3))
ax.bar(labels, values)
ax.set_title({title!r})
fig.savefig(OUTPUT_PATH, bbox_inches='tight')
""".strip()
                return ChartCodeGeneration(code=code, generator_name="sequence")

        class FakeLayoutCritic:
            def __init__(self):
                self.calls = 0

            def critique(self, **kwargs):
                self.calls += 1
                return LayoutCritique(
                    ok=False,
                    failed_contracts=("layout.no_occlusion", "layout.panel_bounds"),
                    diagnosis="Inset panel overlaps the main chart region.",
                    recommended_plan_updates=(
                        {
                            "target": "panel.main",
                            "field": "bounds",
                            "operation": "set",
                            "value": [0.08, 0.1, 0.7, 0.62],
                            "reason": "Reserve more top band space for layout readability.",
                        },
                    ),
                    confidence=0.8,
                )

        generator = SequenceGenerator()
        critic = FakeLayoutCritic()
        pipeline = ChartGenerationPipeline(
            code_generator=generator,
            verifier_pipeline=GroundedChartPipeline(parser=HeuristicIntentParser()),
            layout_critic=critic,
            enable_layout_replanning=True,
            layout_replan_rounds=1,
            layout_replan_acceptance="internal_verifier",
        )

        with tempfile.TemporaryDirectory() as output_tmp:
            result = pipeline.run(
                query="Create a bar chart of sales by product.",
                schema=TableSchema(columns={"product": "string", "sales": "number"}),
                rows=(
                    {"product": "A", "sales": 3},
                    {"product": "B", "sales": 5},
                ),
                output_dir=Path(output_tmp),
                case_id="layout_replanning",
                verification_mode="figure_only",
            )

            self.assertEqual(2, len(generator.requests))
            self.assertEqual(1, critic.calls)
            self.assertIn("Replanned layout", result.final_code)
            self.assertEqual(0, len(result.pipeline_result.repair_attempts))
            self.assertTrue(result.metadata["layout_replanning_attempted"])
            self.assertTrue(result.metadata["layout_replanning_accepted"])
            self.assertEqual(1, result.metadata["layout_replanning_rounds"])
            self.assertEqual("layout_replan_round_1", result.metadata["final_code_source"])
            self.assertTrue((Path(output_tmp) / "LayoutAgent" / "round_1" / "layout_critique.json").exists())
            self.assertTrue((Path(output_tmp) / "LayoutAgent" / "round_1" / "feedback_bundle.json").exists())
            self.assertTrue((Path(output_tmp) / "LayoutAgent" / "round_1" / "replan_trace.json").exists())
            self.assertTrue((Path(output_tmp) / "PlanAgent" / "round_2" / "plan.json").exists())
            self.assertTrue((Path(output_tmp) / "ExecutorAgent" / "round_2" / "steps.md").exists())
            self.assertEqual(Path(output_tmp).resolve() / "round2.png", result.image_path)
            self.assertTrue((Path(output_tmp) / "round1.png").exists())
            self.assertTrue((Path(output_tmp) / "round2.png").exists())
            revised_context = generator.requests[1].context
            self.assertEqual("feedback_to_plan_agent", revised_context["layout_replanning"]["replan_mode"])
            self.assertIn("feedback_bundle", revised_context["layout_replanning"])
            self.assertIn(
                "Reserve more top band space",
                json.dumps(revised_context["layout_replanning"]["feedback_bundle"], ensure_ascii=False),
            )
            self.assertIn(
                "Replan Feedback",
                (Path(output_tmp) / "PlanAgent" / "round_2" / "plan.md").read_text(encoding="utf-8"),
            )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(manifest["metadata"]["layout_replanning_accepted"])
            self.assertEqual("layout_replan_round_1", manifest["verification"]["final_code_source"])
            self.assertEqual("round2.png", Path(manifest["image_path"]).name)
            self.assertEqual("round1.png", Path(manifest["metadata"]["layout_replanning_round_images"][0]["image_path"]).name)
            self.assertEqual("round2.png", Path(manifest["metadata"]["layout_replanning_round_images"][1]["image_path"]).name)

    def test_layout_replanning_uses_plan_agent_for_revised_plan(self):
        class SequenceGenerator:
            def __init__(self):
                self.requests = []

            def generate(self, request):
                self.requests.append(request)
                title = "Initial" if len(self.requests) == 1 else "Replanned"
                code = f"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(4, 3))
ax.bar([row["product"] for row in rows], [row["sales"] for row in rows])
ax.set_title({title!r})
fig.savefig(OUTPUT_PATH, bbox_inches='tight')
""".strip()
                return ChartCodeGeneration(code=code, generator_name="sequence")

        class FakePlanAgent:
            def __init__(self):
                self.requests = []

            def build_plan(self, request):
                self.requests.append(request)
                tag = "initial" if len(self.requests) == 1 else "replan"
                return PlanAgentResult(
                    plan=ChartConstructionPlan(
                        plan_type="chart_construction_plan_v2",
                        layout_strategy=f"{tag}_strategy",
                        panels=(
                            VisualPanelPlan(
                                panel_id="panel.main",
                                role="main",
                                layers=(
                                    VisualLayerPlan(
                                        layer_id="layer.bar",
                                        chart_type="bar",
                                        role="main_bar_layer",
                                        x="product",
                                        y=("sales",),
                                    ),
                                ),
                            ),
                        ),
                    ),
                    agent_name=f"fake_{tag}_plan_agent",
                )

        class FakeLayoutCritic:
            def critique(self, **kwargs):
                return LayoutCritique(
                    ok=False,
                    failed_contracts=("layout.no_occlusion",),
                    diagnosis="Legend overlaps bars.",
                )

        generator = SequenceGenerator()
        plan_agent = FakePlanAgent()
        pipeline = ChartGenerationPipeline(
            code_generator=generator,
            verifier_pipeline=GroundedChartPipeline(parser=HeuristicIntentParser()),
            layout_critic=FakeLayoutCritic(),
            plan_agent=plan_agent,
            enable_layout_replanning=True,
            layout_replan_rounds=1,
            layout_replan_acceptance="internal_verifier",
        )

        with tempfile.TemporaryDirectory() as output_tmp:
            result = pipeline.run(
                query="Create a bar chart of sales by product.",
                schema=TableSchema(columns={"product": "string", "sales": "number"}),
                rows=({"product": "A", "sales": 3},),
                output_dir=Path(output_tmp),
                case_id="llm_plan_agent_replanning",
                verification_mode="figure_only",
            )

            self.assertEqual(2, len(plan_agent.requests))
            self.assertIsNotNone(plan_agent.requests[1].feedback_bundle)
            self.assertEqual("fake_replan_plan_agent", result.metadata["plan_agent"]["agent_name"])
            self.assertEqual("replan_strategy", result.metadata["layout_strategy"])
            self.assertEqual("replan_strategy", generator.requests[1].context["construction_plan"]["layout_strategy"])

    def test_layout_replanning_defaults_to_candidate_only_without_promoting_final_code(self):
        class SequenceGenerator:
            def __init__(self):
                self.requests = []

            def generate(self, request):
                self.requests.append(request)
                title = "Initial layout" if len(self.requests) == 1 else "Candidate layout"
                layout_round = "round_1" if len(self.requests) == 1 else "round_2"
                code = f"""
import json
import matplotlib.pyplot as plt
from pathlib import Path

labels = [row["product"] for row in rows]
values = [row["sales"] for row in rows]
fig, ax = plt.subplots(figsize=(4, 3))
ax.bar(labels, values)
ax.set_title({title!r})
Path(OUTPUT_PATH).with_name('computed_layout.json').write_text(json.dumps({{'layout_round': {layout_round!r}}}))
Path(OUTPUT_PATH).with_name('layout_decisions.md').write_text({('# ' + layout_round)!r})
fig.savefig(OUTPUT_PATH, bbox_inches='tight')
""".strip()
                return ChartCodeGeneration(code=code, generator_name="sequence")

        class FakeLayoutCritic:
            def critique(self, **kwargs):
                return LayoutCritique(
                    ok=False,
                    failed_contracts=("layout.panel_bounds",),
                    diagnosis="Need more top-band room.",
                    recommended_plan_updates=(
                        {
                            "target": "panel.main",
                            "field": "bounds",
                            "operation": "set",
                            "value": [0.08, 0.1, 0.7, 0.62],
                        },
                    ),
                )

        generator = SequenceGenerator()
        pipeline = ChartGenerationPipeline(
            code_generator=generator,
            verifier_pipeline=GroundedChartPipeline(parser=HeuristicIntentParser()),
            layout_critic=FakeLayoutCritic(),
            enable_layout_replanning=True,
            layout_replan_rounds=1,
        )

        with tempfile.TemporaryDirectory() as output_tmp:
            result = pipeline.run(
                query="Create a bar chart of sales by product.",
                schema=TableSchema(columns={"product": "string", "sales": "number"}),
                rows=(
                    {"product": "A", "sales": 3},
                    {"product": "B", "sales": 5},
                ),
                output_dir=Path(output_tmp),
                case_id="layout_candidate_only",
                verification_mode="figure_only",
            )

            self.assertEqual(2, len(generator.requests))
            self.assertIn("Initial layout", result.final_code)
            self.assertNotIn("Candidate layout", result.final_code)
            self.assertTrue(result.metadata["layout_replanning_attempted"])
            self.assertFalse(result.metadata["layout_replanning_accepted"])
            self.assertEqual("candidate_only_not_promoted", result.metadata["layout_replanning_reason"])
            self.assertEqual("candidate_only", result.metadata["layout_replanning_acceptance_policy"])
            self.assertTrue((Path(output_tmp) / "generated_layout_replan_round_1.py").exists())
            self.assertEqual(Path(output_tmp).resolve() / "round1.png", result.image_path)
            self.assertTrue((Path(output_tmp) / "round1.png").exists())
            self.assertTrue((Path(output_tmp) / "round2.png").exists())
            self.assertIn("previous_code", generator.requests[1].context["layout_replanning"])
            self.assertEqual("feedback_to_plan_agent", generator.requests[1].context["layout_replanning"]["replan_mode"])
            self.assertIn("Need more top-band room", json.dumps(generator.requests[1].context["layout_replanning"]["feedback_bundle"], ensure_ascii=False))
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual("round1.png", Path(manifest["image_path"]).name)
            self.assertEqual("round1.png", Path(manifest["metadata"]["layout_replanning_round_images"][0]["image_path"]).name)
            self.assertEqual("round2.png", Path(manifest["metadata"]["layout_replanning_round_images"][1]["image_path"]).name)
            self.assertFalse(manifest["metadata"]["layout_replanning_round_images"][1]["promoted_to_final"])
            records_by_round = manifest["metadata"]["executor_layout_records_by_round"]
            round1_layout_path = Path(records_by_round["round_1"]["computed_layout_json"])
            round2_layout_path = Path(records_by_round["round_2"]["computed_layout_json"])
            self.assertEqual(Path(output_tmp) / "ExecutorAgent" / "round_1" / "computed_layout.json", round1_layout_path)
            self.assertEqual(Path(output_tmp) / "ExecutorAgent" / "round_2" / "computed_layout.json", round2_layout_path)
            self.assertEqual({"layout_round": "round_1"}, json.loads(round1_layout_path.read_text(encoding="utf-8")))
            self.assertEqual({"layout_round": "round_2"}, json.loads(round2_layout_path.read_text(encoding="utf-8")))
            self.assertEqual(
                str(round1_layout_path),
                manifest["metadata"]["executor_layout_records_final"]["computed_layout_json"],
            )

    def test_figure_reader_feedback_is_recorded_and_passed_to_replanning_codegen(self):
        class SequenceGenerator:
            def __init__(self):
                self.requests = []

            def generate(self, request):
                self.requests.append(request)
                title = "Initial visual" if len(self.requests) == 1 else "Visual audit candidate"
                code = f"""
import matplotlib.pyplot as plt

labels = [row["product"] for row in rows]
values = [row["sales"] for row in rows]
fig, ax = plt.subplots(figsize=(4, 3))
ax.bar(labels, values, color='steelblue')
ax.set_title({title!r})
fig.savefig(OUTPUT_PATH, bbox_inches='tight')
""".strip()
                return ChartCodeGeneration(code=code, generator_name="sequence")

        class OkLayoutCritic:
            def critique(self, **kwargs):
                return LayoutCritique(ok=True, confidence=0.9)

        class FakeFigureReader:
            def __init__(self):
                self.calls = 0

            def audit(self, **kwargs):
                self.calls += 1
                return FigureAudit(
                    ok=False,
                    summary="Series and color role are visually ambiguous.",
                    encoding_confusions=(
                        {
                            "issue_type": "color_role_series_confusion",
                            "severity": "warning",
                            "evidence": "The rendered bars do not separate semantic dimensions.",
                            "related_plan_ref": "panel.main",
                            "recommendation": "Use separate visual cues for semantic dimensions.",
                        },
                    ),
                    recommended_plan_notes=(
                        {
                            "note": "Use color for role and hatch or edge style for series identity.",
                            "target_agent": "ExecutorAgent",
                        },
                    ),
                    confidence=0.7,
                )

        generator = SequenceGenerator()
        reader = FakeFigureReader()
        pipeline = ChartGenerationPipeline(
            code_generator=generator,
            verifier_pipeline=GroundedChartPipeline(parser=HeuristicIntentParser()),
            layout_critic=OkLayoutCritic(),
            figure_reader=reader,
            enable_layout_replanning=True,
            layout_replan_rounds=1,
        )

        with tempfile.TemporaryDirectory() as output_tmp:
            result = pipeline.run(
                query="Create a bar chart of sales by product.",
                schema=TableSchema(columns={"product": "string", "sales": "number"}),
                rows=(
                    {"product": "A", "sales": 3},
                    {"product": "B", "sales": 5},
                ),
                output_dir=Path(output_tmp),
                case_id="figure_reader_feedback",
                verification_mode="figure_only",
            )

            self.assertEqual(2, len(generator.requests))
            self.assertEqual(1, reader.calls)
            self.assertTrue(result.metadata["figure_reader_enabled"])
            self.assertFalse(result.metadata["layout_replanning_accepted"])
            audit_path = Path(output_tmp) / "FigureReaderAgent" / "round_1" / "figure_audit.json"
            self.assertTrue(audit_path.exists())
            audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual("Series and color role are visually ambiguous.", audit_payload["summary"])
            layout_critique_path = Path(output_tmp) / "LayoutAgent" / "round_1" / "layout_critique.json"
            layout_critique_payload = json.loads(layout_critique_path.read_text(encoding="utf-8"))
            self.assertTrue(layout_critique_payload["ok"])
            combined_path = Path(output_tmp) / "LayoutAgent" / "round_1" / "combined_replanning_critique.json"
            self.assertTrue(combined_path.exists())
            context = generator.requests[1].context["layout_replanning"]
            self.assertEqual("bounded_visual_presentation_and_encoding", context["allowed_edit_scope"])
            self.assertEqual("feedback_to_plan_agent", context["replan_mode"])
            self.assertIn("feedback_bundle", context)
            self.assertFalse(context["figure_audit_feedback"]["ok"])
            self.assertIn("hatch", json.dumps(context["figure_audit_feedback"], ensure_ascii=False))
            self.assertIn("ExecutorAgent", json.dumps(context["figure_audit_feedback"], ensure_ascii=False))
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            trace = manifest["metadata"]["layout_replanning_trace"][0]
            self.assertIn("visual.encoding_confusion", trace["combined_critique"]["failed_contracts"])
            self.assertIn("hatch", json.dumps(trace["combined_critique"]["recommended_plan_updates"], ensure_ascii=False))
            self.assertEqual(str(audit_path), trace["artifacts"]["figure_audit_path"])
            self.assertEqual(str(combined_path), trace["artifacts"]["combined_critique_path"])

    def test_llm_code_generator_strips_code_fence_and_records_trace(self):
        class FakeClient:
            def complete_json_with_trace(self, **kwargs):
                return LLMJsonResult(
                    payload={
                        "code": "```python\nprint(rows[0])\n```",
                        "backend": "matplotlib",
                        "instruction": "use rows",
                        "notes": ["fake"],
                        "assumptions": [],
                    },
                    trace=LLMCompletionTrace(model="fake-model", raw_text="{}"),
                )

        generator = LLMChartCodeGenerator(FakeClient())
        result = generator.generate(
            ChartCodeGenerationRequest(
                query="plot sales",
                schema=TableSchema(columns={"product": "string", "sales": "number"}),
                rows=({"product": "A", "sales": 1},),
            )
        )

        self.assertEqual("print(rows[0])", result.code)
        self.assertEqual("matplotlib", result.backend_hint)
        self.assertEqual("fake-model", result.llm_trace.model)


if __name__ == "__main__":
    unittest.main()






