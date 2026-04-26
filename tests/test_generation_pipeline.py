import json
import tempfile
import unittest
from pathlib import Path

from grounded_chart import (
    AxisRequirementSpec,
    ChartGenerationPipeline,
    PatchAnchor,
    PatchOperation,
    ChartRenderResult,
    FigureRequirementSpec,
    GroundedChartPipeline,
    HeuristicIntentParser,
    LLMChartCodeGenerator,
    RepairPatch,
    StaticChartCodeGenerator,
    TableSchema,
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

            def render(self, code, *, rows, output_dir, output_filename="figure.png", file_path=None):
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

            def render(self, code, *, rows, output_dir, output_filename="figure.png", file_path=None):
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






