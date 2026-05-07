from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_research.validation.gate import (
    ValidationGateReport,
    write_validity_gate_artifacts,
)


def _report_with_report_only_metric() -> ValidationGateReport:
    top_decile = {
        "metric": "top_decile_20d_excess_return",
        "target_column": "forward_return_20",
        "sample_scope": "oos_labeled_predictions",
        "status": "report_only",
        "reason": (
            "report-only diagnostic; not used for scoring, action, ranking, "
            "thresholding, or gating"
        ),
        "report_only": True,
        "decision_use": "none",
        "top_decile_20d_excess_return": 0.021,
    }
    return ValidationGateReport(
        system_validity_status="pass",
        strategy_candidate_status="warning",
        hard_fail=False,
        warning=True,
        strategy_pass=False,
        system_validity_pass=True,
        warnings=[],
        hard_fail_reasons=[],
        metrics={"top_decile_20d_excess_return": 0.021},
        evidence={"top_decile_20d_excess_return": top_decile},
        horizons=["20d"],
        required_validation_horizon="20d",
        embargo_periods={"20d": 20},
        benchmark_results=[],
        ablation_results=[],
        gate_results={
            "top_decile_20d_excess_return": {
                "status": "diagnostic",
                "affects_strategy": False,
                "affects_pass_fail": False,
                "decision_use": "none",
            }
        },
        official_message="report-only metric is isolated from strategy decisions",
    )


def _flatten_keys_and_values(value: object) -> list[str]:
    if isinstance(value, dict):
        flattened: list[str] = []
        for key, item in value.items():
            flattened.append(str(key))
            flattened.extend(_flatten_keys_and_values(item))
        return flattened
    if isinstance(value, list):
        flattened = []
        for item in value:
            flattened.extend(_flatten_keys_and_values(item))
        return flattened
    return [str(value)]


def test_report_only_artifact_writer_creates_only_static_report_files(tmp_path: Path) -> None:
    report = _report_with_report_only_metric()

    json_path, markdown_path = write_validity_gate_artifacts(report, tmp_path)

    assert json_path.name == "validity_gate.json"
    assert markdown_path.name == "validity_report.md"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "validity_gate.json",
        "validity_report.md",
    ]
    assert not any(
        forbidden in path.name.lower()
        for path in tmp_path.iterdir()
        for forbidden in ("order", "broker", "execution", "live_trade")
    )


@pytest.mark.parametrize("rendered", ["json", "markdown", "html"])
def test_report_only_outputs_contain_no_live_order_or_execution_payloads(
    rendered: str,
) -> None:
    report = _report_with_report_only_metric()
    if rendered == "json":
        payload = json.loads(report.to_json())
        searchable = _flatten_keys_and_values(payload)
    elif rendered == "markdown":
        searchable = [report.to_markdown()]
    else:
        searchable = [report.to_html()]

    joined = "\n".join(searchable).lower()
    assert "top_decile_20d_excess_return" in joined
    assert "report_only" in joined or "report only" in joined
    assert "decision_use" in joined or "decision use" in joined
    assert "place_order" not in joined
    assert "submit_order" not in joined
    assert "create_order" not in joined
    assert "broker" not in joined
    assert "live execution" not in joined
    assert "execution_mode" not in joined
