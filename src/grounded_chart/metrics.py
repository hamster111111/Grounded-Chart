from __future__ import annotations

from collections import Counter

from grounded_chart.schema import VerificationReport


def report_metrics(reports: list[VerificationReport]) -> dict[str, object]:
    total = len(reports)
    passed = sum(1 for report in reports if report.ok)
    errors = Counter(error.code for report in reports for error in report.errors)
    operators = Counter(error.operator or "unknown" for report in reports for error in report.errors)
    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "error_counts": dict(errors),
        "operator_counts": dict(operators),
    }
