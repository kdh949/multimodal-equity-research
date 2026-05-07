from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research.validation import (
    GATE_RULE_STATUSES,
    SYSTEM_VALIDITY_GATE_OUTPUT_SCHEMA_ID,
    SYSTEM_VALIDITY_GATE_OUTPUT_SCHEMA_VERSION,
    SYSTEM_VALIDITY_GATE_REPORT_SCHEMA_ID,
    SYSTEM_VALIDITY_GATE_REPORT_SCHEMA_VERSION,
    SYSTEM_VALIDITY_GATE_REQUIRED_OUTPUT_SECTIONS,
    GateFailureReason,
    GateRuleStatus,
    SerializableValidityGateReport,
    SystemValidityGateOutputSchema,
    SystemValidityGateReportSchema,
    build_system_validity_gate_output_schema,
    build_system_validity_gate_report_schema,
    build_validity_gate_report,
    default_system_validity_gate_output_schema,
    default_system_validity_gate_report_schema,
)


def test_default_system_validity_gate_output_schema_covers_stage1_contract() -> None:
    schema = default_system_validity_gate_output_schema()
    payload = schema.to_dict()

    assert payload["schema_id"] == SYSTEM_VALIDITY_GATE_OUTPUT_SCHEMA_ID
    assert payload["schema_version"] == SYSTEM_VALIDITY_GATE_OUTPUT_SCHEMA_VERSION
    assert payload["required_sections"] == list(SYSTEM_VALIDITY_GATE_REQUIRED_OUTPUT_SECTIONS)
    assert payload["statuses"]["required_fields"] == [
        "system_validity_status",
        "strategy_candidate_status",
        "system_validity_pass",
        "strategy_pass",
        "hard_fail",
        "warning",
        "official_message",
    ]
    assert "oos_fold_count" in payload["metrics"]["required_fields"]
    assert "mean_rank_ic" in payload["metrics"]["required_fields"]
    assert "oos_rank_ic" in payload["metrics"]["required_fields"]
    assert "positive_fold_ratio" in payload["metrics"]["required_fields"]
    assert "strategy_excess_return_vs_spy" in payload["metrics"]["required_fields"]
    assert "strategy_excess_return_vs_equal_weight" in payload["metrics"]["required_fields"]
    assert "baseline_sample_alignment" in payload["metrics"]["required_fields"]
    assert sorted(payload["validation_result_schemas"]) == [
        "backtest",
        "out_of_sample",
        "risk_rules",
        "walk_forward",
    ]
    assert "leakage" in payload["gate_results"]["required_fields"]
    assert "walk_forward_oos" in payload["gate_results"]["required_fields"]
    assert "deterministic_strategy_validity" in payload["gate_results"]["required_fields"]
    assert "purge_embargo_application" in payload["evidence"]["required_fields"]
    assert "ablation_required_scenarios" in payload["evidence"]["required_fields"]
    assert payload["artifact_contract"]["schema_embedded_in_top_level_payload"] is True
    assert payload["artifact_contract"]["artifact_manifest_required"] is True
    assert payload["scope_bounds"]["real_trading_orders"] == "excluded"
    assert payload["scope_bounds"]["llm_trade_decisions"] == "excluded"
    assert payload["scope_bounds"]["target_horizon"] == "forward_return_20"
    assert payload["scope_bounds"]["top_decile_20d_excess_return"] == "report_only"


def test_gate_rule_status_enum_covers_validation_result_statuses() -> None:
    assert GateRuleStatus.PASS == "pass"
    assert GATE_RULE_STATUSES == (
        "pass",
        "warning",
        "fail",
        "hard_fail",
        "insufficient_data",
        "not_evaluable",
        "skipped",
    )

    payload = build_system_validity_gate_output_schema()
    for result_schema in payload["validation_result_schemas"].values():
        assert result_schema["status_enum"] == list(GATE_RULE_STATUSES)
        assert "status" in result_schema["required_fields"]
        assert "passed" in result_schema["required_fields"]


def test_output_schema_defines_backtest_walk_forward_oos_and_risk_results() -> None:
    payload = build_system_validity_gate_output_schema()
    schemas = payload["validation_result_schemas"]

    assert schemas["backtest"]["required_fields"] == [
        "status",
        "passed",
        "cost_adjusted_cumulative_return",
        "excess_return_vs_spy",
        "excess_return_vs_equal_weight",
        "average_daily_turnover",
        "max_drawdown",
        "sample_alignment",
    ]
    assert schemas["walk_forward"]["required_fields"] == [
        "status",
        "passed",
        "fold_count",
        "oos_fold_count",
        "target_column",
        "target_horizon",
        "purge_periods",
        "embargo_periods",
    ]
    assert schemas["out_of_sample"]["required_fields"] == [
        "status",
        "passed",
        "oos_rank_ic",
        "mean_rank_ic",
        "positive_fold_ratio",
        "positive_fold_ratio_threshold",
        "oos_fold_count",
    ]
    assert schemas["risk_rules"]["required_fields"] == [
        "status",
        "passed",
        "max_holdings_passed",
        "max_symbol_weight_passed",
        "max_sector_weight_passed",
        "drawdown_passed",
        "turnover_passed",
        "violations",
    ]


def test_system_validity_gate_output_schema_rejects_invalid_version() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        SystemValidityGateOutputSchema(schema_version="system_validity_gate_output.v0")


def test_default_system_validity_gate_report_schema_defines_structured_reasons() -> None:
    schema = default_system_validity_gate_report_schema()
    payload = schema.to_dict()

    assert payload["schema_id"] == SYSTEM_VALIDITY_GATE_REPORT_SCHEMA_ID
    assert payload["schema_version"] == SYSTEM_VALIDITY_GATE_REPORT_SCHEMA_VERSION
    assert payload["status_enum"] == list(GATE_RULE_STATUSES)
    assert payload["top_level_statuses"]["required_fields"] == [
        "system_validity_status",
        "strategy_candidate_status",
        "system_validity_pass",
        "strategy_pass",
        "hard_fail",
        "warning",
        "official_message",
    ]
    assert payload["gate_result"]["required_fields"] == [
        "gate",
        "status",
        "passed",
        "severity",
        "reason_code",
        "reason",
        "metric",
        "value",
        "threshold",
        "operator",
        "affects_system",
        "affects_strategy",
    ]

    reasons = payload["structured_reasons"]
    assert "gate_failure_reason" in reasons
    assert reasons["pass_fail_reason"]["required_fields"] == [
        "category",
        "entity_id",
        "rule",
        "metric",
        "status",
        "passed",
        "reason_code",
        "reason",
    ]
    assert reasons["warning_reason"]["required_fields"] == [
        "code",
        "severity",
        "gate",
        "metric",
        "message",
    ]
    assert payload["artifact_contract"]["pass_fail_reasons_field"] == (
        "structured_pass_fail_reasons"
    )
    assert payload["artifact_contract"]["gate_failure_reasons_field"] == (
        "gate_failure_reasons"
    )
    assert payload["artifact_contract"]["structured_gate_failure_report_field"] == (
        "structured_gate_failure_report"
    )
    assert payload["artifact_contract"]["warnings_field"] == "structured_warnings"


def test_system_validity_gate_report_schema_rejects_invalid_version() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        SystemValidityGateReportSchema(schema_version="system_validity_gate_report.v0")


def test_gate_failure_reason_model_is_json_serializable() -> None:
    reason = GateFailureReason(
        gate="rank_ic",
        status="fail",
        reason_code="rank_ic_non_positive",
        reason="mean rank IC must be positive",
        metric="mean_rank_ic",
        value=-0.01,
        threshold=0.0,
        operator=">",
        affects_system=False,
        affects_strategy=True,
    )

    payload = reason.to_dict()
    assert payload["schema_version"] == "gate_failure_reason.v1"
    assert payload["gate"] == "rank_ic"
    assert payload["status"] == "fail"


def test_serializable_validity_gate_report_model_exports_failure_reasons() -> None:
    report = SerializableValidityGateReport(
        system_validity_status="pass",
        strategy_candidate_status="fail",
        system_validity_pass=True,
        strategy_pass=False,
        hard_fail=False,
        warning=False,
        official_message="시스템은 유효하지만 현재 전략 후보는 배포/사용 부적합",
        gate_failure_reasons=(
            GateFailureReason(
                gate="cost_adjusted_performance",
                status="fail",
                reason_code="required_outperformance_rule_not_passed",
                reason="strategy must outperform SPY and equal-weight baselines",
            ),
        ),
        gate_results={"cost_adjusted_performance": {"status": "fail"}},
        metrics={"oos_fold_count": 2},
        evidence={"baseline_sample_alignment": {"status": "pass"}},
        artifact_manifest={"json_artifact": "validity_gate_report.json"},
        report_path="reports/validity_report.md",
    )

    payload = report.to_dict()
    assert payload["schema_version"] == "serializable_validity_gate_report.v1"
    assert payload["gate_failure_reasons"][0]["gate"] == "cost_adjusted_performance"
    assert '"gate_failure_reasons"' in report.to_json()


def test_validity_gate_report_serializes_output_schema() -> None:
    report = build_validity_gate_report(
        _predictions(),
        _validation_summary(),
        _equity_curve(),
        SimpleNamespace(cagr=0.25, sharpe=1.0, max_drawdown=-0.05, turnover=0.10),
        ablation_summary=_ablation_summary(),
        config=SimpleNamespace(gap_periods=20, embargo_periods=20),
    )

    payload = report.to_dict()
    schema = payload["system_validity_gate_output_schema"]
    report_schema = payload["system_validity_gate_report_schema"]
    assert schema == build_system_validity_gate_output_schema()
    assert report_schema == build_system_validity_gate_report_schema()
    assert schema["schema_version"] == SYSTEM_VALIDITY_GATE_OUTPUT_SCHEMA_VERSION
    assert report_schema["schema_version"] == SYSTEM_VALIDITY_GATE_REPORT_SCHEMA_VERSION
    assert payload["metrics"]["system_validity_gate_output_schema"] == schema
    assert payload["evidence"]["system_validity_gate_output_schema"] == schema
    assert payload["metrics"]["system_validity_gate_report_schema"] == report_schema
    assert payload["evidence"]["system_validity_gate_report_schema"] == report_schema
    assert "gate_failure_reasons" in payload
    assert "serializable_gate_report" in payload
    assert payload["serializable_gate_report"]["schema_version"] == (
        "serializable_validity_gate_report.v1"
    )
    assert payload["serializable_gate_report"]["gate_failure_reasons"] == payload[
        "gate_failure_reasons"
    ]

    for field in schema["statuses"]["required_fields"]:
        assert field in payload
    for field in schema["metrics"]["required_fields"]:
        assert field in payload["metrics"]
    for field in schema["gate_results"]["required_fields"]:
        assert field in payload["gate_results"]
    for field in schema["evidence"]["required_fields"]:
        assert field in payload["evidence"]


def _predictions() -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=21, freq="B")
    rows = []
    for fold, date in enumerate(dates):
        for ticker, value in zip(("AAPL", "MSFT", "SPY"), (0.03, 0.02, 0.01), strict=True):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "fold": fold,
                    "is_oos": fold >= 19,
                    "expected_return": value,
                    "forward_return_1": value,
                    "forward_return_5": value,
                    "forward_return_20": value,
                }
            )
    return pd.DataFrame(rows)


def _validation_summary() -> pd.DataFrame:
    test_starts = pd.date_range("2026-02-02", periods=21, freq="B")
    return pd.DataFrame(
        {
            "fold": range(21),
            "train_end": test_starts - pd.Timedelta(days=30),
            "test_start": test_starts,
            "is_oos": [False] * 19 + [True, True],
            "labeled_test_observations": [3] * 21,
            "train_observations": [252] * 21,
            "target_column": ["forward_return_20"] * 21,
            "prediction_horizon_periods": [20] * 21,
            "gap_periods": [20] * 21,
            "purge_periods": [20] * 21,
            "purged_date_count": [20] * 21,
            "purge_applied": [True] * 21,
            "embargo_periods": [20] * 21,
            "embargoed_date_count": [20] * 21,
            "embargo_applied": [True] * 21,
        }
    )


def _equity_curve() -> pd.DataFrame:
    dates = pd.date_range("2026-03-02", periods=21, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": [0.001] * len(dates),
            "cost_adjusted_return": [0.001] * len(dates),
            "benchmark_return": [0.0] * len(dates),
            "turnover": [0.10] * len(dates),
        }
    )


def _ablation_summary() -> list[dict[str, object]]:
    return [
        {"scenario": "all_features", "sharpe": 1.0, "excess_return": 0.10},
        {"scenario": "price_only", "sharpe": 0.4, "excess_return": 0.04},
        {"scenario": "text_only", "sharpe": 0.3, "excess_return": 0.03},
        {"scenario": "sec_only", "sharpe": 0.2, "excess_return": 0.02},
        {"scenario": "no_model_proxy", "sharpe": 0.5, "excess_return": 0.05},
        {"scenario": "no_costs", "sharpe": 0.6, "excess_return": 0.06},
    ]
