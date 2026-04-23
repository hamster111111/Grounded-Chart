"""Adapters for benchmark-specific case loading and evaluation glue.

This package is intentionally separate from `grounded_chart`, which contains
the core framework objects and verification logic.
"""

from grounded_chart_adapters.base import AdapterRunResult, BenchmarkAdapter, ChartCase
from grounded_chart_adapters.batch import BatchRunner, BatchRunResult
from grounded_chart_adapters.html_report import write_batch_report_html
from grounded_chart_adapters.json_case import JsonCaseAdapter
from grounded_chart_adapters.matplotbench import (
    MatplotBenchInstructionAdapter,
    MatplotBenchRecord,
    MatplotBenchWorkspaceAdapter,
    MatplotBenchWorkspaceRecord,
)
from grounded_chart_adapters.memory import InMemoryCaseAdapter
from grounded_chart_adapters.requirement_extraction import (
    RequirementExtractionCaseReport,
    RequirementExtractionReport,
    RequirementExtractionRunner,
    RequirementExtractionSummary,
)
from grounded_chart_adapters.reporting import BatchReport, BatchSummary, CaseReport

__all__ = [
    "AdapterRunResult",
    "BenchmarkAdapter",
    "BatchReport",
    "BatchRunner",
    "BatchRunResult",
    "BatchSummary",
    "CaseReport",
    "ChartCase",
    "InMemoryCaseAdapter",
    "JsonCaseAdapter",
    "MatplotBenchInstructionAdapter",
    "MatplotBenchRecord",
    "MatplotBenchWorkspaceAdapter",
    "MatplotBenchWorkspaceRecord",
    "RequirementExtractionCaseReport",
    "RequirementExtractionReport",
    "RequirementExtractionRunner",
    "RequirementExtractionSummary",
    "write_batch_report_html",
]
