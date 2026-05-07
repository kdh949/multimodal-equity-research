from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research.validation import (
    STRATEGY_CANDIDATE_STATUSES,
    SYSTEM_VALIDITY_GATE_POLICY_ID,
    SYSTEM_VALIDITY_GATE_POLICY_SCHEMA_VERSION,
    SYSTEM_VALIDITY_STATUSES,
    ValidationGateReport,
    build_system_validity_gate_criteria,
    build_validity_gate_report,
    write_validity_gate_artifacts,
)


def _passing_predictions() -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=21, freq="B")
    rows = []
    for fold, date in enumerate(dates):
        for ticker, value in zip(("AAPL", "MSFT", "SPY"), (0.03, 0.02, 0.01), strict=True):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "fold": fold,
                    "is_oos": fold >= len(dates) - 2,
                    "expected_return": value,
                    "forward_return_1": 0.0,
                    "forward_return_5": value,
                    "forward_return_20": value,
                }
            )
    return pd.DataFrame(rows)


def _passing_validation_summary() -> pd.DataFrame:
    test_starts = pd.date_range("2026-02-02", periods=21, freq="B")
    return pd.DataFrame(
        {
            "fold": range(21),
            "train_end": test_starts - pd.Timedelta(days=7),
            "test_start": test_starts,
            "is_oos": [False] * 19 + [True, True],
            "labeled_test_observations": [3] * 21,
            "train_observations": [60] * 21,
        }
    )


def _passing_equity_curve() -> pd.DataFrame:
    dates = pd.date_range("2026-03-02", periods=21, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": [0.001] * len(dates),
            "gross_return": [0.001] * len(dates),
            "cost_adjusted_return": [0.001] * len(dates),
            "benchmark_return": [0.0] * len(dates),
            "turnover": [0.10] * len(dates),
        }
    )


def _passing_ablation_summary() -> list[dict[str, object]]:
    return [
        {"scenario": "all_features", "sharpe": 1.0, "excess_return": 0.10},
        {"scenario": "price_only", "sharpe": 0.4, "excess_return": 0.04},
        {"scenario": "text_only", "sharpe": 0.3, "excess_return": 0.03},
        {"scenario": "sec_only", "sharpe": 0.2, "excess_return": 0.02},
        {"scenario": "no_model_proxy", "sharpe": 0.5, "excess_return": 0.05},
        {"scenario": "no_costs", "sharpe": 0.6, "excess_return": 0.06},
    ]


def _passing_metrics() -> SimpleNamespace:
    return SimpleNamespace(cagr=0.25, sharpe=1.0, max_drawdown=-0.05, turnover=0.10)


def test_system_validity_status_contract_is_exactly_three_values() -> None:
    assert SYSTEM_VALIDITY_STATUSES == ("pass", "hard_fail", "not_evaluable")


def test_system_validity_gate_criteria_define_pass_fail_contract() -> None:
    criteria = build_system_validity_gate_criteria()

    assert criteria["policy_id"] == SYSTEM_VALIDITY_GATE_POLICY_ID
    assert criteria["schema_version"] == SYSTEM_VALIDITY_GATE_POLICY_SCHEMA_VERSION
    assert criteria["status_precedence"] == ["hard_fail", "not_evaluable", "pass"]
    assert criteria["status_contract"] == {
        "pass": "all system criteria pass and no required data criterion is insufficient",
        "hard_fail": "one or more structural validity criteria fail",
        "not_evaluable": "no structural failure exists, but required data/evidence is insufficient",
    }
    assert [row["criterion_id"] for row in criteria["criteria"]] == [
        "target_horizon_forward_return_20",
        "feature_availability_cutoff",
        "purge_embargo_horizon_consistency",
        "walk_forward_oos",
        "benchmark_equal_weight_sample_alignment",
        "artifact_reproducibility_contract",
    ]
    assert criteria["criteria"][3]["gate_result"] == "walk_forward_oos"
    assert "oos_fold_count >= 2" in criteria["criteria"][3]["pass_condition"]
    assert "embargo_periods=0" in criteria["criteria"][2]["hard_fail_condition"]
    assert "system_validity_gate_output_schema" in criteria["criteria"][5]["required_evidence"]


def test_strategy_candidate_status_contract_is_exactly_five_values() -> None:
    assert STRATEGY_CANDIDATE_STATUSES == (
        "pass",
        "warning",
        "fail",
        "insufficient_data",
        "not_evaluable",
    )


def test_system_validity_status_serializes_pass() -> None:
    report = build_validity_gate_report(
        _passing_predictions(),
        _passing_validation_summary(),
        _passing_equity_curve(),
        _passing_metrics(),
        ablation_summary=_passing_ablation_summary(),
        config=SimpleNamespace(gap_periods=60, embargo_periods=60),
    )

    assert report.system_validity_status == "pass"
    assert report.to_dict()["system_validity_status"] == "pass"
    assert report.gate_results["system_validity_artifact_contract"]["status"] == "pass"
    assert (
        report.metrics["system_validity_gate_criteria"]["policy_id"]
        == SYSTEM_VALIDITY_GATE_POLICY_ID
    )
    assert (
        report.evidence["system_validity_gate_criteria"]
        == report.metrics["system_validity_gate_criteria"]
    )
    assert "- System validity: `pass`" in report.to_markdown()


def test_system_validity_gate_detects_overlapping_train_test_windows() -> None:
    validation_summary = _passing_validation_summary()
    validation_summary.loc[3, "train_end"] = validation_summary.loc[3, "test_start"]

    with pytest.raises(ValueError, match="training window must end before the test window starts"):
        build_validity_gate_report(
            _passing_predictions(),
            validation_summary,
            _passing_equity_curve(),
            _passing_metrics(),
            ablation_summary=_passing_ablation_summary(),
            config=SimpleNamespace(gap_periods=60, embargo_periods=60),
        )


def test_validity_gate_preserves_purge_embargo_application_evidence() -> None:
    validation_summary = _passing_validation_summary().assign(
        train_start=pd.date_range("2025-12-01", periods=21, freq="B"),
        validation_start=lambda frame: frame["test_start"].where(~frame["is_oos"], pd.NaT),
        validation_end=lambda frame: (
            frame["test_start"] + pd.Timedelta(days=4)
        ).where(~frame["is_oos"], pd.NaT),
        test_end=lambda frame: frame["test_start"] + pd.Timedelta(days=4),
        oos_test_start=lambda frame: frame["test_start"].where(frame["is_oos"], pd.NaT),
        oos_test_end=lambda frame: frame["test_end"].where(frame["is_oos"], pd.NaT),
        train_periods=252,
        validation_periods=60,
        test_periods=60,
        target_column="forward_return_20",
        prediction_horizon_periods=20,
        gap_periods=20,
        purge_periods=20,
        purge_gap_periods=20,
        purged_date_count=20,
        purge_start=pd.date_range("2026-01-02", periods=21, freq="B"),
        purge_end=pd.date_range("2026-01-30", periods=21, freq="B"),
        purge_applied=True,
        embargo_periods=20,
        embargoed_date_count=20,
        embargo_start=pd.date_range("2026-03-02", periods=21, freq="B"),
        embargo_end=pd.date_range("2026-03-30", periods=21, freq="B"),
        embargo_applied=True,
    )

    report = build_validity_gate_report(
        _passing_predictions(),
        validation_summary,
        _passing_equity_curve(),
        _passing_metrics(),
        ablation_summary=_passing_ablation_summary(),
        config=SimpleNamespace(gap_periods=20, embargo_periods=20),
    )

    application = report.metrics["purge_embargo_application"]
    assert application["status"] == "pass"
    assert application["target_horizon"] == 20
    assert application["min_purged_date_count"] == 20
    assert application["min_embargo_periods"] == 20
    assert application["all_folds_purge_applied"] is True
    assert application["all_folds_embargo_applied"] is True
    assert report.evidence["leakage"]["purge_embargo_application"] == application
    assert report.to_dict()["metrics"]["purge_embargo_application"]["folds"][0][
        "purged_date_count"
    ] == 20
    fold_evidence = report.to_dict()["metrics"]["purge_embargo_application"]["folds"]
    assert fold_evidence[0]["train_start"].startswith("2025-12-01")
    assert fold_evidence[0]["validation_start"].startswith("2026-02-02")
    assert fold_evidence[0]["test_start"].startswith("2026-02-02")
    assert fold_evidence[0]["train_periods"] == 252
    assert fold_evidence[0]["validation_periods"] == 60
    assert fold_evidence[0]["test_periods"] == 60
    assert fold_evidence[0]["purge_periods"] == 20
    assert fold_evidence[-1]["validation_start"] is None
    assert fold_evidence[-1]["oos_test_start"] is not None


def test_system_validity_gate_hard_fails_when_oos_fold_count_is_below_two() -> None:
    validation_summary = _passing_validation_summary()
    validation_summary["is_oos"] = [False] * 20 + [True]

    report = build_validity_gate_report(
        _passing_predictions(),
        validation_summary,
        _passing_equity_curve(),
        _passing_metrics(),
        ablation_summary=_passing_ablation_summary(),
        config=SimpleNamespace(gap_periods=60, embargo_periods=60),
    )

    walk_forward_gate = report.gate_results["walk_forward_oos"]
    assert report.system_validity_status == "hard_fail"
    assert report.system_validity_pass is False
    assert walk_forward_gate["status"] == "hard_fail"
    assert walk_forward_gate["oos_fold_count"] == 1
    assert walk_forward_gate["required_min_oos_folds"] == 2
    assert "oos_fold_count=1 is below required=2" in walk_forward_gate["reasons"]


def test_strategy_candidate_fails_when_cost_adjusted_return_is_below_configured_threshold(
    tmp_path,
) -> None:
    equity_curve = _passing_equity_curve()
    equity_curve["portfolio_return"] = [0.040] * len(equity_curve)
    equity_curve["gross_return"] = [0.041] * len(equity_curve)
    equity_curve["cost_adjusted_return"] = [0.040] * len(equity_curve)
    equity_curve["benchmark_return"] = [0.005] * len(equity_curve)
    configured_threshold = 2.0
    expected_cost_adjusted_return = (1.040 ** len(equity_curve)) - 1.0

    report = build_validity_gate_report(
        _passing_predictions(),
        _passing_validation_summary(),
        equity_curve,
        _passing_metrics(),
        ablation_summary=_passing_ablation_summary(),
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_20",
            required_validation_horizon=20,
            benchmark_ticker="SPY",
            gap_periods=60,
            embargo_periods=60,
            cost_bps=0.0,
            slippage_bps=0.0,
            cost_adjusted_collapse_threshold=configured_threshold,
        ),
    )

    cost_gate = report.gate_results["cost_adjusted_performance"]
    assert report.system_validity_status == "pass"
    assert report.strategy_candidate_status == "fail"
    assert report.strategy_pass is False
    assert report.warning is False
    assert cost_gate["status"] == "fail"
    assert cost_gate["failed_baselines"] == []
    assert cost_gate["passed_baselines"] == ["SPY", "equal_weight"]
    assert cost_gate["baseline_excess_return_statuses"] == {
        "SPY": "pass",
        "equal_weight": "pass",
    }
    assert cost_gate["cost_adjusted_cumulative_return"] == pytest.approx(
        expected_cost_adjusted_return
    )
    assert cost_gate["collapse_threshold"] == configured_threshold
    assert cost_gate["reason_metadata"] == {
        "code": "cost_adjusted_cumulative_return_at_or_below_collapse_threshold",
        "metric": "cost_adjusted_cumulative_return",
        "value": pytest.approx(expected_cost_adjusted_return),
        "threshold": configured_threshold,
        "operator": ">",
    }

    json_path, markdown_path = write_validity_gate_artifacts(report, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert payload["strategy_candidate_status"] == "fail"
    assert payload["metrics"]["cost_adjusted_collapse_threshold"] == configured_threshold
    assert payload["evidence"]["cost_adjusted_collapse_check"]["status"] == "fail"
    assert payload["gate_results"]["cost_adjusted_performance"]["collapse_threshold"] == (
        configured_threshold
    )
    assert payload["strategy_failure_summary"][0]["gate"] == "cost_adjusted_performance"
    assert payload["strategy_failure_summary"][0]["threshold"] == configured_threshold
    assert "- Strategy candidate: `fail`" in markdown
    assert "| cost_adjusted_performance | fail | cost-adjusted cumulative return" in markdown


def test_validity_gate_artifact_records_stage1_contract_fields(tmp_path) -> None:
    report = build_validity_gate_report(
        _passing_predictions(),
        _passing_validation_summary(),
        _passing_equity_curve(),
        _passing_metrics(),
        ablation_summary=_passing_ablation_summary(),
        config=SimpleNamespace(gap_periods=60, embargo_periods=60),
    )

    payload = report.to_dict()
    required_fields = {
        "hard_fail",
        "warning",
        "strategy_pass",
        "system_validity_pass",
        "metrics",
        "evidence",
    }
    assert required_fields <= payload.keys()
    assert payload["hard_fail"] is False
    assert payload["warning"] is False
    assert payload["strategy_pass"] is True
    assert payload["system_validity_pass"] is True
    assert isinstance(payload["metrics"], dict)
    assert isinstance(payload["evidence"], dict)

    json_path, markdown_path = write_validity_gate_artifacts(report, tmp_path)
    artifact_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert {field: artifact_payload[field] for field in required_fields} == {
        field: payload[field] for field in required_fields
    }

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "- Hard fail: `False`" in markdown
    assert "- Warning: `False`" in markdown
    assert "- Strategy pass: `True`" in markdown
    assert "- System validity pass: `True`" in markdown
    assert "## Metrics" in markdown
    assert "## Evidence" in markdown


def test_system_validity_status_serializes_hard_fail() -> None:
    report = build_validity_gate_report(
        _passing_predictions(),
        _passing_validation_summary(),
        _passing_equity_curve(),
        _passing_metrics(),
        ablation_summary=_passing_ablation_summary(),
        config=SimpleNamespace(gap_periods=0, embargo_periods=60),
    )

    assert report.system_validity_status == "hard_fail"
    assert report.to_dict()["system_validity_status"] == "hard_fail"
    assert "- System validity: `hard_fail`" in report.to_markdown()


def test_system_validity_status_serializes_not_evaluable() -> None:
    report = build_validity_gate_report(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        _passing_metrics(),
        config=SimpleNamespace(gap_periods=60, embargo_periods=60),
    )

    assert report.system_validity_status == "not_evaluable"
    assert report.to_dict()["system_validity_status"] == "not_evaluable"
    assert "- System validity: `not_evaluable`" in report.to_markdown()


def test_undersized_dataset_propagates_insufficient_data_without_hard_fail(tmp_path) -> None:
    validation_summary = pd.DataFrame(
        [
            {
                "fold": pd.NA,
                "fold_type": "skipped",
                "is_oos": False,
                "validation_status": "insufficient_data",
                "skip_status": "skipped",
                "skip_code": "insufficient_labeled_dates",
                "reason": "not enough labeled dates to create a walk-forward fold",
                "fold_count": 0,
                "candidate_fold_count": 0,
                "candidate_date_count": 8,
                "labeled_date_count": 8,
                "required_min_date_count": 26,
                "min_train_observations": 80,
            }
        ]
    )
    predictions = pd.DataFrame(
        columns=[
            "date",
            "ticker",
            "expected_return",
            "forward_return_5",
            "forward_return_20",
            "fold",
            "is_oos",
        ]
    )
    report = build_validity_gate_report(
        predictions,
        validation_summary,
        pd.DataFrame(columns=["date", "portfolio_return", "turnover"]),
        _passing_metrics(),
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_20",
            required_validation_horizon=20,
            gap_periods=60,
            embargo_periods=60,
            benchmark_ticker="SPY",
            cost_bps=5.0,
            slippage_bps=2.0,
        ),
    )

    assert report.system_validity_status == "not_evaluable"
    assert report.strategy_candidate_status == "insufficient_data"
    assert report.hard_fail is False
    assert report.system_validity_pass is False
    assert report.gate_results["walk_forward_oos"]["status"] == "insufficient_data"
    assert report.metrics["fold_count"] == 0
    assert report.metrics["insufficient_data"] is True
    assert "walk_forward_oos" in report.metrics["insufficient_data_reasons"][0]

    payload = report.to_dict()
    assert payload["hard_fail"] is False
    assert payload["gate_results"]["walk_forward_oos"]["skip_code"] == "insufficient_labeled_dates"
    assert payload["strategy_candidate_status"] == "insufficient_data"

    json_path, markdown_path = write_validity_gate_artifacts(report, tmp_path)
    artifact_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert artifact_payload["hard_fail"] is False
    assert artifact_payload["strategy_candidate_status"] == "insufficient_data"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "- Strategy candidate: `insufficient_data`" in markdown
    assert "| walk_forward_oos | insufficient_data |" in markdown


def test_validation_gate_report_rejects_unknown_system_validity_status() -> None:
    with pytest.raises(ValueError, match="system_validity_status must be one of"):
        ValidationGateReport(
            system_validity_status="warning",
            strategy_candidate_status="pass",
            hard_fail=False,
            warning=False,
            strategy_pass=True,
            system_validity_pass=True,
            warnings=[],
            hard_fail_reasons=[],
            metrics={},
            evidence={},
            horizons=[],
            required_validation_horizon="5d",
            embargo_periods={},
            benchmark_results=[],
            ablation_results=[],
            gate_results={},
            official_message="",
        )


@pytest.mark.parametrize("strategy_status", STRATEGY_CANDIDATE_STATUSES)
def test_validation_gate_report_accepts_strategy_candidate_status_contract(
    strategy_status: str,
) -> None:
    report = ValidationGateReport(
        system_validity_status="pass",
        strategy_candidate_status=strategy_status,
        hard_fail=False,
        warning=strategy_status == "warning",
        strategy_pass=strategy_status == "pass",
        system_validity_pass=True,
        warnings=[],
        hard_fail_reasons=[],
        metrics={},
        evidence={},
        horizons=[],
        required_validation_horizon="5d",
        embargo_periods={},
        benchmark_results=[],
        ablation_results=[],
        gate_results={},
        official_message="",
    )

    assert report.to_dict()["strategy_candidate_status"] == strategy_status
    assert f"- Strategy candidate: `{strategy_status}`" in report.to_markdown()


def test_validation_gate_report_rejects_unknown_strategy_candidate_status() -> None:
    with pytest.raises(ValueError, match="strategy_candidate_status must be one of"):
        ValidationGateReport(
            system_validity_status="pass",
            strategy_candidate_status="hard_fail",
            hard_fail=False,
            warning=False,
            strategy_pass=False,
            system_validity_pass=True,
            warnings=[],
            hard_fail_reasons=[],
            metrics={},
            evidence={},
            horizons=[],
            required_validation_horizon="5d",
            embargo_periods={},
            benchmark_results=[],
            ablation_results=[],
            gate_results={},
            official_message="",
        )
