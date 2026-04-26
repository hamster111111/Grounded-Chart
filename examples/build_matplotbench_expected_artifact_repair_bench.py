from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    compare_report_path = project_root / args.evidence_report
    output_path = project_root / args.output
    report = _load_json_dict(compare_report_path)

    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for case in report.get("cases", []):
        case_id = str(case.get("case_id") or "")
        metadata = dict(case.get("case_metadata") or {})
        code_path = Path(str(metadata.get("source_code") or ""))
        figure_requirements = case.get("figure_requirements")
        if not case_id:
            skipped.append({"case_id": case_id, "reason": "missing_case_id"})
            continue
        if not code_path.exists():
            skipped.append({"case_id": case_id, "reason": "missing_source_code", "path": str(code_path)})
            continue
        if not isinstance(figure_requirements, dict):
            skipped.append({"case_id": case_id, "reason": "missing_figure_requirements"})
            continue

        artifact_contracts = _visual_artifact_contracts(case)
        figure_requirements, reconciliation = _reconcile_figure_requirements(figure_requirements, artifact_contracts)

        expected_points = case.get("expected_points") or []
        expected_trace = None
        if expected_points:
            expected_trace = {
                "chart_type": case.get("expected_chart_type") or "unknown",
                "source": "expected_artifact_compare_report",
                "points": expected_points,
            }

        metadata.update(
            {
                "expected_artifact_repair_source": str(compare_report_path),
                "expected_artifact_error_codes": list(case.get("error_codes", [])),
                "expected_artifact_case_verdict": case.get("case_verdict"),
                "expected_artifact_status": case.get("status"),
                "artifact_contract_count": len(artifact_contracts),
            }
        )
        if reconciliation:
            metadata["figure_requirement_reconciliation"] = reconciliation
        records.append(
            {
                "case_id": case_id,
                "query": case.get("query") or case_id,
                "schema": {"columns": {}},
                "rows": [],
                "generated_code_path": str(code_path),
                "verification_mode": "figure_and_data" if expected_trace is not None else "figure_only",
                "expected_trace": expected_trace,
                "figure_requirements": figure_requirements,
                "metadata": metadata,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "evidence_report": str(compare_report_path),
        "output": str(output_path),
        "records": len(records),
        "skipped": skipped,
        "cases_with_artifact_contracts": sum(1 for record in records if record["figure_requirements"].get("artifact_contracts")),
        "artifact_contracts": sum(len(record["figure_requirements"].get("artifact_contracts", [])) for record in records),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a repair-comparison bench from an expected-artifact verifier report."
    )
    parser.add_argument(
        "--evidence-report",
        default="outputs/expected_artifact_matplotbench_native_compare_30_latest/evidence_artifact_verifier/report.json",
    )
    parser.add_argument("--output", default="benchmarks/matplotbench_expected_artifact_repair_30.json")
    return parser.parse_args()


def _visual_artifact_contracts(case: dict[str, Any]) -> list[dict[str, Any]]:
    graph = case.get("evidence_graph") if isinstance(case.get("evidence_graph"), dict) else {}
    contracts: list[dict[str, Any]] = []
    for artifact in graph.get("expected_artifacts", []):
        if not isinstance(artifact, dict):
            continue
        artifact_id = str(artifact.get("artifact_id") or "")
        if not artifact_id.startswith("expected.visual."):
            continue
        payload = artifact.get("payload_preview")
        if not isinstance(payload, dict):
            continue
        contract = {
            "artifact_type": payload.get("artifact_type"),
            "expected": payload.get("expected", {}),
            "locator": payload.get("locator", {}),
            "source_requirement_id": payload.get("source_requirement_id"),
            "criticality": payload.get("criticality", "hard"),
            "match_policy": payload.get("match_policy", "exact"),
            "supported_by_verifier": payload.get("supported_by_verifier", True),
            "source_span": payload.get("source_span", ""),
        }
        if contract["artifact_type"]:
            contracts.append(contract)
    return contracts


def _reconcile_figure_requirements(
    raw_figure_requirements: dict[str, Any], artifact_contracts: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    figure_requirements = dict(raw_figure_requirements)
    figure_requirements["artifact_contracts"] = artifact_contracts
    implied_axes_count = _implied_axes_count_from_panel_contracts(artifact_contracts)
    current_axes_count = _positive_int(figure_requirements.get("axes_count"))
    if current_axes_count is None or implied_axes_count is None or current_axes_count >= implied_axes_count:
        return figure_requirements, None

    figure_requirements["axes_count"] = implied_axes_count
    source_spans = dict(figure_requirements.get("source_spans") or {})
    source_spans.setdefault("axes_count", "Inferred from expected visual panel contracts.")
    figure_requirements["source_spans"] = source_spans
    return figure_requirements, {
        "field": "axes_count",
        "reason": "panel_contract_count_exceeds_axes_count",
        "original": current_axes_count,
        "reconciled": implied_axes_count,
    }


def _implied_axes_count_from_panel_contracts(artifact_contracts: list[dict[str, Any]]) -> int | None:
    panel_ids: set[str] = set()
    for contract in artifact_contracts:
        if not isinstance(contract, dict):
            continue
        artifact_type = str(contract.get("artifact_type") or "")
        if not artifact_type.startswith("panel_"):
            continue
        locator = contract.get("locator") if isinstance(contract.get("locator"), dict) else {}
        panel_id = str(locator.get("panel_id") or "").strip()
        if panel_id:
            panel_ids.add(panel_id)
    return len(panel_ids) if panel_ids else None


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _load_json_dict(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


if __name__ == "__main__":
    main()
