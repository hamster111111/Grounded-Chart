import tempfile
import unittest
from pathlib import Path

from grounded_chart_adapters.html_report import write_batch_report_html
from grounded_chart_adapters.reporting import BatchReport, CaseReport


class HtmlReportTest(unittest.TestCase):
    def test_write_batch_report_html_renders_case_and_summary(self):
        report = BatchReport.from_case_reports(
            [
                CaseReport(
                    case_id="plotly-soft-case",
                    status="failed",
                    ok=False,
                    query="Create a Plotly chart with a missing annotation.",
                    expected_chart_type="bar",
                    actual_chart_type="bar",
                    backend_profile={
                        "backend_name": "plotly",
                        "support_tier": "spec_accessible",
                        "verification_mode": "soft",
                        "notes": "Plotly smoke case",
                    },
                    figure_requirements={
                        "axes_count": 1,
                        "figure_title": "Expected title",
                        "axes": [
                            {
                                "axis_index": 0,
                                "title": "Expected axis",
                                "xlabel": "category",
                                "ylabel": "value",
                                "projection": "plotly",
                                "artist_types": ["bar"],
                            }
                        ],
                    },
                    actual_figure={
                        "title": "Actual title",
                        "axes_count": 1,
                        "size_inches": [10, 6],
                        "axes": [
                            {
                                "index": 0,
                                "title": "Actual axis",
                                "xlabel": "category",
                                "ylabel": "value",
                                "projection": "plotly",
                                "artists": [{"artist_type": "bar"}],
                            }
                        ],
                    },
                    error_codes=("wrong_axis_title", "missing_annotation_text"),
                    errors=(
                        {
                            "code": "wrong_axis_title",
                            "operator": "equals",
                            "expected": "Expected axis",
                            "actual": "Actual axis",
                            "severity": "error",
                        },
                    ),
                    repair_level=3,
                    repair_scope="backend_specific_regeneration",
                    repair_strategy="regenerate_with_backend_constraints",
                    repair_instruction="Regenerate for Plotly with explicit annotation and title constraints.",
                    case_metadata={
                        "reason": "Soft verification example",
                        "source_code": r"D:\tmp\example.py",
                        "native_id": 90,
                        "gpt4v_score": 1,
                        "expected_failure_family": "plotly_soft",
                    },
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            write_batch_report_html(report, output_path, title="Smoke HTML")
            html = output_path.read_text(encoding="utf-8")

        self.assertIn("Smoke HTML", html)
        self.assertIn("GroundedChart Smoke Browser", html)
        self.assertIn("plotly-soft-case", html)
        self.assertIn("backend_specific_regeneration", html)
        self.assertIn('"backend_name": "plotly"', html)
        self.assertIn('"total_cases": 1', html)


if __name__ == "__main__":
    unittest.main()
