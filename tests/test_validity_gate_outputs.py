from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research.validation import (
    OFFICIAL_STRATEGY_FAIL_MESSAGE,
    ValidationGateThresholds,
    build_validity_gate_report,
    write_validity_gate_artifacts,
)


def _load_streamlit_app_module():
    app_path = Path(__file__).parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("quant_research_streamlit_app", app_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Streamlit app module from {app_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


streamlit_app = _load_streamlit_app_module()


def _sufficient_predictions() -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=21, freq="B")
    rows: list[dict[str, object]] = []
    for fold, date in enumerate(dates):
        for ticker, expected, one_day, five_day, twenty_day in zip(
            ("AAPL", "MSFT", "SPY"),
            (0.030, 0.020, 0.010),
            (0.012, 0.008, 0.004),
            (0.030, 0.020, 0.010),
            (0.080, 0.050, 0.020),
            strict=True,
        ):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "fold": fold,
                    "is_oos": fold >= len(dates) - 2,
                    "expected_return": expected,
                    "forward_return_1": one_day,
                    "forward_return_5": five_day,
                    "forward_return_20": twenty_day,
                }
            )
    return pd.DataFrame(rows)


def _sufficient_validation_summary() -> pd.DataFrame:
    test_starts = pd.date_range("2026-02-02", periods=21, freq="B")
    return pd.DataFrame(
        {
            "fold": range(len(test_starts)),
            "train_end": test_starts - pd.Timedelta(days=10),
            "test_start": test_starts,
            "is_oos": [False] * (len(test_starts) - 2) + [True, True],
            "labeled_test_observations": [3] * len(test_starts),
            "train_observations": [120] * len(test_starts),
        }
    )


def _sufficient_equity_curve() -> pd.DataFrame:
    dates = pd.date_range("2026-03-02", periods=21, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": [0.040] * len(dates),
            "gross_return": [0.041] * len(dates),
            "cost_adjusted_return": [0.040] * len(dates),
            "benchmark_return": [0.005] * len(dates),
            "turnover": [0.10] * len(dates),
            "transaction_cost_return": [0.0007] * len(dates),
            "slippage_cost_return": [0.0003] * len(dates),
            "total_cost_return": [0.0010] * len(dates),
            "portfolio_volatility_estimate": [0.018] * len(dates),
            "position_sizing_validation_status": ["pass"] * len(dates),
            "position_sizing_validation_rule": [
                "post_cost_position_sizing_constraints_v1"
            ]
            * len(dates),
            "position_sizing_validation_reason": [
                "risk_concentration_and_leverage_limits_passed_after_costs"
            ]
            * len(dates),
            "position_count": [2] * len(dates),
            "max_position_weight": [0.10] * len(dates),
            "max_sector_exposure": [0.20] * len(dates),
            "max_position_risk_contribution": [0.55] * len(dates),
        }
    )


def _stage1_ablation_summary() -> list[dict[str, object]]:
    return [
        {"scenario": "all_features", "kind": "signal", "sharpe": 1.20, "excess_return": 0.12},
        {"scenario": "price_only", "kind": "data_channel", "sharpe": 0.40, "excess_return": 0.04},
        {"scenario": "text_only", "kind": "data_channel", "sharpe": 0.30, "excess_return": 0.03},
        {"scenario": "sec_only", "kind": "data_channel", "sharpe": 0.20, "excess_return": 0.02},
        {
            "scenario": "no_model_proxy",
            "kind": "pipeline_control",
            "sharpe": 0.50,
            "excess_return": 0.05,
            "pipeline_controls": {
                "model_proxy": False,
                "cost": True,
                "slippage": True,
                "turnover": True,
            },
            "validation_status": "pass",
            "validation_fold_count": 21,
            "validation_oos_fold_count": 2,
            "deterministic_signal_evaluation_metrics": {
                "return_basis": "cost_adjusted_return",
                "action_counts": {"BUY": 4, "SELL": 2, "HOLD": 15},
                "cost_adjusted_cumulative_return": 0.10,
                "average_daily_turnover": 0.10,
                "total_cost_return": 0.01,
            },
        },
        {
            "scenario": "no_costs",
            "kind": "cost",
            "sharpe": 1.10,
            "excess_return": 0.14,
            "pipeline_controls": {
                "model_proxy": True,
                "cost": False,
                "slippage": False,
                "turnover": False,
            },
        },
    ]


def _stage1_config() -> SimpleNamespace:
    return SimpleNamespace(
        tickers=["AAPL", "MSFT"],
        prediction_target_column="forward_return_20",
        required_validation_horizon=20,
        benchmark_ticker="SPY",
        gap_periods=60,
        embargo_periods=60,
        cost_bps=5.0,
        slippage_bps=2.0,
        top_n=20,
        max_symbol_weight=0.10,
        max_sector_weight=0.30,
        portfolio_covariance_lookback=20,
        covariance_aware_risk_enabled=True,
        covariance_return_column="return_1",
        covariance_min_periods=20,
        portfolio_volatility_limit=0.04,
        max_position_risk_contribution=1.0,
    )


def _strategy_metrics() -> SimpleNamespace:
    return SimpleNamespace(
        cagr=1.50,
        sharpe=1.25,
        max_drawdown=-0.04,
        turnover=0.10,
        average_portfolio_volatility_estimate=0.018,
        max_portfolio_volatility_estimate=0.018,
        max_position_risk_contribution=0.55,
    )


def _sufficient_report():
    return build_validity_gate_report(
        _sufficient_predictions(),
        _sufficient_validation_summary(),
        _sufficient_equity_curve(),
        _strategy_metrics(),
        ablation_summary=_stage1_ablation_summary(),
        config=_stage1_config(),
    )


def _sufficient_report_with_failing_one_day_diagnostic():
    predictions = _sufficient_predictions()
    reversed_one_day_returns = {"AAPL": 0.001, "MSFT": 0.002, "SPY": 0.003}
    predictions["forward_return_1"] = predictions["ticker"].map(reversed_one_day_returns)
    return build_validity_gate_report(
        predictions,
        _sufficient_validation_summary(),
        _sufficient_equity_curve(),
        _strategy_metrics(),
        ablation_summary=_stage1_ablation_summary(),
        config=_stage1_config(),
    )


def _sufficient_report_with_failing_twenty_day_robustness():
    predictions = _sufficient_predictions()
    reversed_twenty_day_returns = {"AAPL": 0.001, "MSFT": 0.002, "SPY": 0.003}
    predictions["forward_return_20"] = predictions["ticker"].map(reversed_twenty_day_returns)
    return build_validity_gate_report(
        predictions,
        _sufficient_validation_summary(),
        _sufficient_equity_curve(),
        _strategy_metrics(),
        ablation_summary=_stage1_ablation_summary(),
        config=_stage1_config(),
    )


def _sufficient_report_with_positive_fold_ratio(negative_date_count: int):
    predictions = _sufficient_predictions()
    negative_partial_rank_returns = {"AAPL": 0.001, "MSFT": 0.003, "SPY": 0.002}
    positive_rank_returns = {"AAPL": 0.080, "MSFT": 0.050, "SPY": 0.020}
    dates = sorted(predictions["date"].unique())
    negative_dates = set(dates[:negative_date_count])
    for ticker, value in negative_partial_rank_returns.items():
        mask = predictions["date"].isin(negative_dates) & predictions["ticker"].eq(ticker)
        predictions.loc[mask, "forward_return_20"] = value
    for ticker, value in positive_rank_returns.items():
        mask = ~predictions["date"].isin(negative_dates) & predictions["ticker"].eq(ticker)
        predictions.loc[mask, "forward_return_20"] = value
    return build_validity_gate_report(
        predictions,
        _sufficient_validation_summary(),
        _sufficient_equity_curve(),
        _strategy_metrics(),
        ablation_summary=_stage1_ablation_summary(),
        config=_stage1_config(),
    )


def _sufficient_report_with_low_positive_fold_ratio():
    return _sufficient_report_with_positive_fold_ratio(negative_date_count=13)


def _sufficient_report_with_passing_positive_fold_ratio():
    return _sufficient_report_with_positive_fold_ratio(negative_date_count=7)


def _report_with_insufficient_twenty_day_robustness():
    predictions = _sufficient_predictions()
    short_dates = sorted(predictions["date"].unique())[:5]
    predictions = predictions[predictions["date"].isin(short_dates)].copy()
    predictions["is_oos"] = predictions["date"] == short_dates[-1]

    validation_summary = _sufficient_validation_summary().head(len(short_dates)).copy()
    validation_summary["is_oos"] = [False] * (len(short_dates) - 1) + [True]

    return build_validity_gate_report(
        predictions,
        validation_summary,
        _sufficient_equity_curve().head(len(short_dates)),
        _strategy_metrics(),
        ablation_summary=_stage1_ablation_summary(),
        config=_stage1_config(),
    )


def _insufficient_validation_summary() -> pd.DataFrame:
    return pd.DataFrame(
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


def _empty_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "ticker",
            "expected_return",
            "forward_return_1",
            "forward_return_5",
            "forward_return_20",
            "fold",
            "is_oos",
        ]
    )


def _empty_equity_curve() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "portfolio_return",
            "gross_return",
            "cost_adjusted_return",
            "benchmark_return",
            "turnover",
        ]
    )


def _insufficient_report():
    return build_validity_gate_report(
        _empty_predictions(),
        _insufficient_validation_summary(),
        _empty_equity_curve(),
        _strategy_metrics(),
        config=_stage1_config(),
    )


def _horizon_json_snapshot(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    metrics = payload["metrics"]
    assert isinstance(metrics, dict)
    horizon_metrics = metrics["horizon_metrics"]
    assert isinstance(horizon_metrics, dict)
    return {
        horizon: {
            "label": row["label"],
            "role": row["role"],
            "target_column": row["target_column"],
            "status": row["status"],
            "affects_pass_fail": row["affects_pass_fail"],
            "insufficient_data": row["insufficient_data"],
            "insufficient_data_code": row["insufficient_data_code"],
        }
        for horizon, row in horizon_metrics.items()
    }


def _gate_status_snapshot(payload: dict[str, object]) -> dict[str, str]:
    gate_results = payload["gate_results"]
    assert isinstance(gate_results, dict)
    return {
        gate: gate_results[gate]["status"]
        for gate in (
            "leakage",
            "walk_forward_oos",
            "rank_ic",
            "cost_adjusted_performance",
            "benchmark_comparison",
            "turnover",
            "drawdown",
            "ablation",
        )
    }


def test_validity_gate_sufficient_data_json_markdown_html_snapshots(tmp_path) -> None:
    report = _sufficient_report()

    payload = json.loads(report.to_json())
    assert {
        "system_validity_status": payload["system_validity_status"],
        "strategy_candidate_status": payload["strategy_candidate_status"],
        "hard_fail": payload["hard_fail"],
        "warning": payload["warning"],
        "strategy_pass": payload["strategy_pass"],
        "system_validity_pass": payload["system_validity_pass"],
        "required_validation_horizon": payload["required_validation_horizon"],
        "metrics": {
            "fold_count": payload["metrics"]["fold_count"],
            "oos_fold_count": payload["metrics"]["oos_fold_count"],
            "insufficient_data": payload["metrics"]["insufficient_data"],
            "target_column": payload["metrics"]["target_column"],
        },
        "gate_statuses": _gate_status_snapshot(payload),
        "horizons": _horizon_json_snapshot(payload),
    } == {
        "system_validity_status": "pass",
        "strategy_candidate_status": "pass",
        "hard_fail": False,
        "warning": False,
        "strategy_pass": True,
        "system_validity_pass": True,
        "required_validation_horizon": "20d",
        "metrics": {
            "fold_count": 21,
                "oos_fold_count": 2,
            "insufficient_data": False,
            "target_column": "forward_return_20",
        },
        "gate_statuses": {
            "leakage": "pass",
            "walk_forward_oos": "pass",
            "rank_ic": "pass",
            "cost_adjusted_performance": "pass",
            "benchmark_comparison": "pass",
            "turnover": "pass",
            "drawdown": "pass",
            "ablation": "pass",
        },
        "horizons": {
            "1d": {
                "label": "diagnostic",
                "role": "diagnostic",
                "target_column": "forward_return_1",
                "status": "pass",
                "affects_pass_fail": False,
                "insufficient_data": False,
                "insufficient_data_code": None,
            },
            "5d": {
                "label": "diagnostic",
                "role": "diagnostic",
                "target_column": "forward_return_5",
                "status": "pass",
                "affects_pass_fail": False,
                "insufficient_data": False,
                "insufficient_data_code": None,
            },
            "20d": {
                "label": "required",
                "role": "decision",
                "target_column": "forward_return_20",
                "status": "pass",
                "affects_pass_fail": True,
                "insufficient_data": False,
                "insufficient_data_code": None,
            },
        },
    }

    json_path, markdown_path = write_validity_gate_artifacts(report, tmp_path)
    assert json.loads(json_path.read_text(encoding="utf-8")) == payload

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "- System validity: `pass`" in markdown
    assert "- Strategy candidate: `pass`" in markdown
    assert "| 5d | diagnostic | diagnostic | forward_return_5 | False | pass | False |" in markdown
    assert "| 20d | required | decision | forward_return_20 | True | pass | False |" in markdown
    assert "## Cost-Adjusted Strategy Comparison" in markdown
    assert "## Baseline Comparison Inputs" in markdown
    assert "## No-Model-Proxy Ablation" in markdown

    html = report.to_html()
    assert "<h1>Validity Gate Report</h1>" in html
    assert "<h2>Status</h2>" in html
    assert "<td>System validity</td>" in html
    assert "<td>Strategy candidate</td>" in html
    assert "<h2>Horizon Diagnostics</h2>" in html
    assert "<td>20d</td>" in html
    assert "<td>required</td>" in html
    assert "<td>forward_return_20</td>" in html
    assert "<h2>Cost-Adjusted Strategy Comparison</h2>" in html
    assert "<h2>Baseline Comparison Inputs</h2>" in html
    assert "<h2>No-Model-Proxy Ablation</h2>" in html
    assert "<td>Model proxy enabled</td>" in html


def test_validity_gate_report_records_covariance_aware_risk_application(tmp_path) -> None:
    report = _sufficient_report()

    payload = json.loads(report.to_json())
    covariance_risk = payload["metrics"]["covariance_aware_risk"]
    assert covariance_risk["configured_enabled"] is True
    assert covariance_risk["applied"] is True
    assert covariance_risk["status"] == "applied"
    assert covariance_risk["parameters"] == {
        "schema_version": "portfolio_risk_constraints.v1",
        "max_holdings": 20,
        "max_symbol_weight": 0.10,
        "max_sector_weight": 0.30,
        "max_position_risk_contribution": 1.0,
        "portfolio_volatility_limit": 0.04,
        "lookback_periods": 20,
        "return_column": "return_1",
        "min_periods": 20,
        "fallback": "diagonal_predicted_volatility",
        "long_only": True,
        "v1_exclusions": ["correlation_cluster_weight"],
    }
    assert covariance_risk["realized_metrics"][
        "average_portfolio_volatility_estimate"
    ] == pytest.approx(0.018)
    assert covariance_risk["realized_metrics"][
        "latest_position_sizing_validation_status"
    ] == "pass"
    assert payload["evidence"]["covariance_aware_risk"] == covariance_risk

    markdown = report.to_markdown()
    assert "## Covariance-Aware Risk" in markdown
    assert "| Applied | True |" in markdown
    assert "| Return column | return_1 |" in markdown
    assert "| Lookback periods | 20 |" in markdown
    assert "| Max symbol weight | 0.1000 |" in markdown

    html = report.to_html()
    assert "<h2>Covariance-Aware Risk</h2>" in html
    assert "<td>Applied</td>" in html

    _, markdown_path = write_validity_gate_artifacts(report, tmp_path)
    assert "## Covariance-Aware Risk" in markdown_path.read_text(encoding="utf-8")


def test_validity_gate_outputs_explain_rule_results_and_final_strategy_status(
    tmp_path,
) -> None:
    config = _stage1_config()
    config.cost_adjusted_collapse_threshold = 2.0
    report = build_validity_gate_report(
        _sufficient_predictions(),
        _sufficient_validation_summary(),
        _sufficient_equity_curve(),
        _strategy_metrics(),
        ablation_summary=_stage1_ablation_summary(),
        config=config,
    )

    payload = report.to_dict()
    explanations = {
        row["rule"]: row
        for row in payload["rule_result_explanations"]
    }
    final_status = payload["final_strategy_status_explanation"]

    assert payload["final_strategy_status"] == "fail"
    assert payload["strategy_candidate_status"] == "fail"
    assert final_status["final_strategy_status"] == "fail"
    assert final_status["system_validity_status"] == "pass"
    assert final_status["strategy_pass"] is False
    assert final_status["blocking_rules"] == [
        "cost_adjusted_performance",
        "deterministic_strategy_validity",
    ]
    assert "cost_adjusted_performance" in final_status["reason"]

    assert explanations["leakage"]["status"] == "pass"
    assert explanations["leakage"]["passed"] is True
    assert explanations["cost_adjusted_performance"]["status"] == "fail"
    assert explanations["cost_adjusted_performance"]["passed"] is False
    assert explanations["cost_adjusted_performance"]["reason_code"] == (
        "cost_adjusted_cumulative_return_at_or_below_collapse_threshold"
    )
    assert explanations["cost_adjusted_performance"]["metric"] == (
        "cost_adjusted_cumulative_return"
    )
    assert explanations["cost_adjusted_performance"]["threshold"] == 2.0
    assert explanations["rank_ic_1d_diagnostic"]["passed"] is None
    assert explanations["rank_ic_1d_diagnostic"]["affects_strategy"] is False

    json_path, markdown_path = write_validity_gate_artifacts(report, tmp_path)
    artifact_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert artifact_payload["rule_result_explanations"] == payload[
        "rule_result_explanations"
    ]
    assert artifact_payload["final_strategy_status_explanation"] == final_status

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## Final Strategy Status Explanation" in markdown
    assert "## Rule Result Explanations" in markdown
    assert "| Final strategy status | `fail` |" in markdown
    assert (
        "| cost_adjusted_performance | fail | False | True | True | "
        "cost-adjusted cumulative return"
    ) in markdown

    html = report.to_html()
    assert "<h2>Final Strategy Status Explanation</h2>" in html
    assert "<h2>Rule Result Explanations</h2>" in html
    assert "<td>cost_adjusted_performance</td>" in html
    assert "<td>fail</td>" in html


def test_validity_gate_result_summary_serializes_failures_warnings_and_key_metrics(
    tmp_path,
) -> None:
    config = _stage1_config()
    config.cost_adjusted_collapse_threshold = 2.0
    config.monthly_turnover_budget = 0.50
    equity_curve = _sufficient_equity_curve()
    equity_curve["turnover"] = [0.10] * 10 + [0.20] * 11

    report = build_validity_gate_report(
        _sufficient_predictions(),
        _sufficient_validation_summary(),
        equity_curve,
        _strategy_metrics(),
        ablation_summary=_stage1_ablation_summary(),
        config=config,
    )

    payload = report.to_dict()
    summary = payload["validity_gate_result_summary"]

    assert summary["schema_version"] == "validity_gate_result_summary.v1"
    assert summary["system_validity_status"] == "pass"
    assert summary["strategy_candidate_status"] == "fail"
    assert summary["final_gate_decision"] == "FAIL"
    assert summary["failure_reason_count"] >= 1
    assert summary["warning_count"] >= 1
    assert summary["key_metrics"]["oos_fold_count"] == 2
    assert summary["key_metrics"]["target_horizon"] == 20
    assert summary["key_metrics"]["positive_fold_ratio_threshold"] == pytest.approx(0.65)
    assert summary["key_metrics"]["strategy_excess_return_vs_spy"] > 0
    assert any(
        row["gate"] == "cost_adjusted_performance"
        and row["reason_code"]
        == "cost_adjusted_cumulative_return_at_or_below_collapse_threshold"
        for row in summary["failure_reasons"]
    )
    assert any(
        row["gate"] == "monthly_turnover_budget"
        and row["code"] == "monthly_turnover_budget_exceeded"
        for row in summary["warnings"]
    )
    assert payload["metrics"]["validity_gate_result_summary"] == summary
    assert payload["evidence"]["validity_gate_result_summary"] == summary

    json_path, markdown_path = write_validity_gate_artifacts(report, tmp_path)
    artifact_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert artifact_payload["validity_gate_result_summary"] == summary

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## Validity Gate Result Summary" in markdown
    assert "### Summary Failure Reasons" in markdown
    assert "### Summary Warnings" in markdown
    assert "### Summary Key Metrics" in markdown
    assert "- cost_adjusted_performance: cost-adjusted cumulative return" in markdown
    assert "- monthly_turnover_budget: monthly_turnover_budget:" in markdown

    html = report.to_html()
    assert "<h2>Validity Gate Result Summary</h2>" in html
    assert "<h2>Summary Failure Reasons</h2>" in html
    assert "<h2>Summary Warnings</h2>" in html
    assert "<h2>Summary Key Metrics</h2>" in html
    assert "<td>cost_adjusted_performance</td>" in html
    assert "<td>monthly_turnover_budget</td>" in html


def test_validity_gate_json_contract_includes_model_baseline_ablation_metrics_and_reasons(
    tmp_path,
) -> None:
    report = _sufficient_report()
    payload = report.to_dict()

    full_model_metrics = payload["full_model_metrics"]
    assert full_model_metrics["entity_id"] == "all_features"
    assert full_model_metrics["role"] == "full_model"
    assert full_model_metrics["status"] == "pass"
    assert full_model_metrics["metrics"]["sharpe"] == pytest.approx(1.25)
    assert full_model_metrics["metrics"]["cost_adjusted_cumulative_return"] > 0
    assert full_model_metrics["validation_metrics"]["mean_rank_ic"] > 0
    assert payload["metrics"]["full_model_metrics"] == full_model_metrics
    assert payload["evidence"]["full_model_metrics"] == full_model_metrics

    baseline_metrics = {
        row["entity_id"]: row
        for row in payload["baseline_metrics"]
    }
    assert {"no_model_proxy", "return_baseline_spy", "return_baseline_equal_weight"}.issubset(
        baseline_metrics
    )
    assert baseline_metrics["no_model_proxy"]["role"] == "model_baseline"
    assert baseline_metrics["no_model_proxy"]["metrics"]["sharpe"] == pytest.approx(0.50)
    assert baseline_metrics["return_baseline_spy"]["role"] == "return_baseline"
    assert "cost_adjusted_cumulative_return" in baseline_metrics[
        "return_baseline_equal_weight"
    ]["metrics"]

    ablation_metrics = {
        row["entity_id"]: row
        for row in payload["ablation_metrics"]
    }
    assert {"price_only", "text_only", "sec_only", "no_costs"}.issubset(ablation_metrics)
    assert ablation_metrics["price_only"]["metrics"]["sharpe"] == pytest.approx(0.40)
    assert ablation_metrics["no_costs"]["kind"] == "cost"
    assert payload["metrics"]["ablation_metrics"] == payload["ablation_metrics"]

    reasons = payload["structured_pass_fail_reasons"]
    gate_reasons = [row for row in reasons if row["category"] == "gate"]
    comparison_reasons = [
        row for row in reasons if row["category"] == "model_comparison"
    ]
    assert any(
        row["rule"] == "rank_ic" and row["status"] == "pass" and row["passed"] is True
        for row in gate_reasons
    )
    assert any(
        row["rule"] == "model_outperformance"
        and row["metric"] == "sharpe"
        and row["baseline"] == "no_model_proxy"
        and row["status"] in {"pass", "fail", "not_evaluable"}
        for row in comparison_reasons
    )

    json_path, markdown_path = write_validity_gate_artifacts(report, tmp_path)
    artifact_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert artifact_payload["structured_pass_fail_reasons"] == reasons

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## Full Model Metrics" in markdown
    assert "## Baseline Metrics" in markdown
    assert "## Ablation Metrics" in markdown
    assert "## Structured Pass/Fail Reasons" in markdown

    html = report.to_html()
    assert "<h2>Full Model Metrics</h2>" in html
    assert "<h2>Baseline Metrics</h2>" in html
    assert "<h2>Ablation Metrics</h2>" in html
    assert "<h2>Structured Pass/Fail Reasons</h2>" in html
    assert "<td>all_features</td>" in html
    assert "<td>no_model_proxy</td>" in html
    assert "<td>model_outperformance</td>" in html


def test_report_schema_collects_all_gate_failure_and_warning_reasons() -> None:
    failing_report = _sufficient_report_with_low_positive_fold_ratio()
    failing_payload = failing_report.to_dict()
    failing_reasons = {
        (row["category"], row["rule"], row["status"]): row
        for row in failing_payload["structured_pass_fail_reasons"]
        if row["category"] == "gate"
    }

    rank_failure = failing_reasons[("gate", "rank_ic", "fail")]
    assert rank_failure["reason_code"] == "positive_fold_ratio_below_minimum"
    assert rank_failure["reason"] == failing_report.gate_results["rank_ic"]["reason"]
    assert rank_failure["affects_strategy"] is True

    warning_report = _insufficient_report()
    warning_payload = warning_report.to_dict()
    warning_gates = {
        gate
        for gate, result in warning_report.gate_results.items()
        if isinstance(result, dict) and result.get("status") == "warning"
    }
    structured_warning_gates = {
        row["gate"] for row in warning_payload["structured_warnings"]
    }

    assert warning_gates
    assert warning_gates.issubset(structured_warning_gates)
    assert all(
        {"code", "severity", "gate", "metric", "message"}.issubset(row)
        for row in warning_payload["structured_warnings"]
    )
    assert warning_payload["warnings"] == [
        warning_report.gate_results[gate]["reason"]
        for gate in warning_report.gate_results
        if isinstance(warning_report.gate_results[gate], dict)
        and warning_report.gate_results[gate].get("status") == "warning"
    ]
    assert warning_payload["metrics"]["structured_pass_fail_reasons"] == (
        warning_payload["structured_pass_fail_reasons"]
    )
    assert warning_payload["evidence"]["structured_pass_fail_reasons"] == (
        warning_payload["structured_pass_fail_reasons"]
    )


def test_gate_failure_report_groups_failed_gates_with_severity_and_metrics(tmp_path) -> None:
    config = _stage1_config()
    config.cost_adjusted_collapse_threshold = 2.0
    report = build_validity_gate_report(
        _sufficient_predictions(),
        _sufficient_validation_summary(),
        _sufficient_equity_curve(),
        _strategy_metrics(),
        ablation_summary=_stage1_ablation_summary(),
        config=config,
    )

    payload = report.to_dict()
    failure_report = payload["structured_gate_failure_report"]
    gates = {row["gate"]: row for row in failure_report["gates"]}

    assert failure_report["schema_version"] == "structured_gate_failure_report.v1"
    assert failure_report["strategy_candidate_status"] == "fail"
    assert failure_report["failed_gate_count"] == len(gates)
    assert failure_report["reason_count"] == len(payload["gate_failure_reasons"])
    assert "cost_adjusted_performance" in gates
    assert gates["cost_adjusted_performance"]["severity"] == "fail"
    assert gates["cost_adjusted_performance"]["top_reason_code"] == (
        "cost_adjusted_cumulative_return_at_or_below_collapse_threshold"
    )
    related_metrics = {
        row["metric"]: row
        for row in gates["cost_adjusted_performance"]["related_metrics"]
    }
    assert related_metrics["cost_adjusted_cumulative_return"]["threshold"] == 2.0
    assert related_metrics["cost_adjusted_cumulative_return"]["operator"] == ">"
    assert related_metrics["cost_adjusted_cumulative_return"]["value"] > 0
    assert payload["metrics"]["structured_gate_failure_report"] == failure_report
    assert payload["evidence"]["structured_gate_failure_report"] == failure_report
    assert payload["serializable_gate_report"]["structured_gate_failure_report"] == (
        failure_report
    )

    assert failure_report["severity_counts"]["fail"] >= 1

    json_path, markdown_path = write_validity_gate_artifacts(report, tmp_path)
    artifact_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert artifact_payload["structured_gate_failure_report"] == failure_report

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## Structured Gate Failure Report" in markdown
    assert "| cost_adjusted_performance | fail | fail |" in markdown

    html = report.to_html()
    assert "<h2>Structured Gate Failure Report</h2>" in html
    assert "<td>cost_adjusted_performance</td>" in html


def test_system_validity_hard_fail_returns_expected_structured_failure_reasons(
    tmp_path,
) -> None:
    config = _stage1_config()
    config.embargo_periods = 0

    report = build_validity_gate_report(
        _sufficient_predictions(),
        _sufficient_validation_summary(),
        _sufficient_equity_curve(),
        _strategy_metrics(),
        ablation_summary=_stage1_ablation_summary(),
        config=config,
    )

    payload = report.to_dict()
    expected_reasons = {
        "embargo_periods=0 is below target horizon=20",
        "configured embargo=0 is below target horizon=20",
    }
    system_failures = [
        row
        for row in payload["gate_failure_reasons"]
        if row["gate"] == "system_validity"
        and row["reason_code"] == "system_validity_hard_fail"
    ]
    system_failure_reasons = {row["reason"] for row in system_failures}

    assert report.system_validity_status == "hard_fail"
    assert report.strategy_candidate_status == "not_evaluable"
    assert report.hard_fail is True
    assert expected_reasons.issubset(system_failure_reasons)
    assert all(row["status"] == "hard_fail" for row in system_failures)
    assert all(row["severity"] == "hard_fail" for row in system_failures)
    assert all(row["affects_system"] is True for row in system_failures)
    assert all(row["affects_strategy"] is False for row in system_failures)

    leakage_reason = next(
        row
        for row in payload["gate_failure_reasons"]
        if row["gate"] == "leakage"
        and row["reason_code"] == "leakage_not_passed"
    )
    assert leakage_reason["status"] == "hard_fail"
    assert leakage_reason["severity"] == "hard_fail"
    assert leakage_reason["reason"] == (
        "embargo_periods=0 is below target horizon=20; "
        "configured embargo=0 is below target horizon=20"
    )
    assert leakage_reason["affects_system"] is True

    failure_report = payload["structured_gate_failure_report"]
    grouped_failures = {row["gate"]: row for row in failure_report["gates"]}
    assert grouped_failures["system_validity"]["top_reason_code"] == (
        "system_validity_hard_fail"
    )
    assert grouped_failures["system_validity"]["severity"] == "hard_fail"
    assert expected_reasons.issubset(
        {row["reason"] for row in grouped_failures["system_validity"]["reasons"]}
    )
    assert grouped_failures["leakage"]["top_reason_code"] == "leakage_not_passed"

    summary_reasons = payload["validity_gate_result_summary"]["failure_reasons"]
    assert expected_reasons.issubset(
        {
            row["reason"]
            for row in summary_reasons
            if row["gate"] == "system_validity"
            and row["reason_code"] == "system_validity_hard_fail"
        }
    )

    json_path, markdown_path = write_validity_gate_artifacts(report, tmp_path)
    artifact_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert artifact_payload["gate_failure_reasons"] == payload["gate_failure_reasons"]
    assert artifact_payload["structured_gate_failure_report"] == failure_report
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "- System validity: `hard_fail`" in markdown
    assert "embargo_periods=0 is below target horizon=20" in markdown


def test_one_day_diagnostic_is_reported_but_does_not_gate_sufficient_validity() -> None:
    report = _sufficient_report_with_failing_one_day_diagnostic()

    assert report.system_validity_status == "pass"
    assert report.system_validity_pass is True
    assert report.hard_fail is False
    assert report.metrics["insufficient_data"] is False
    assert report.gate_results["rank_ic"]["status"] == "pass"

    one_day = report.metrics["horizon_metrics"]["1d"]
    assert one_day["label"] == "diagnostic"
    assert one_day["role"] == "diagnostic"
    assert one_day["target_column"] == "forward_return_1"
    assert one_day["status"] == "fail"
    assert one_day["rank_ic_status"] == "fail"
    assert one_day["affects_pass_fail"] is False
    assert one_day["insufficient_data"] is False
    assert one_day["mean_rank_ic"] < 0
    assert report.metrics["diagnostic_horizon_metrics"]["1d"] == one_day

    diagnostic_gate = report.gate_results["rank_ic_1d_diagnostic"]
    assert diagnostic_gate["status"] == "diagnostic"
    assert diagnostic_gate["diagnostic_status"] == "fail"
    assert diagnostic_gate["affects_pass_fail"] is False
    assert diagnostic_gate["affects_system"] is False
    assert diagnostic_gate["affects_strategy"] is False

    payload = json.loads(report.to_json())
    assert payload["metrics"]["horizon_metrics"]["1d"]["status"] == "fail"
    assert payload["gate_results"]["rank_ic_1d_diagnostic"]["diagnostic_status"] == "fail"
    markdown = report.to_markdown()
    assert "| rank_ic_1d_diagnostic | diagnostic |" in markdown
    assert "| 1d | diagnostic | diagnostic | forward_return_1 | False | fail | False |" in markdown
    html = report.to_html()
    assert "<h2>Horizon Diagnostics</h2>" in html
    assert "<td>1d</td>" in html
    assert "<td>fail</td>" in html


def test_twenty_day_robustness_fails_not_skips_when_window_data_is_sufficient() -> None:
    report = _sufficient_report_with_failing_twenty_day_robustness()

    assert report.system_validity_status == "pass"
    assert report.system_validity_pass is True
    assert report.strategy_candidate_status == "fail"
    assert report.strategy_pass is False
    assert report.hard_fail is False
    assert report.metrics["insufficient_data"] is False
    assert report.gate_results["rank_ic"]["status"] == "fail"

    twenty_day = report.metrics["horizon_metrics"]["20d"]
    assert twenty_day["label"] == "required"
    assert twenty_day["role"] == "decision"
    assert twenty_day["target_column"] == "forward_return_20"
    assert twenty_day["status"] == "fail"
    assert twenty_day["rank_ic_status"] == "fail"
    assert twenty_day["affects_pass_fail"] is True
    assert twenty_day["minimum_observation_guard"] is True
    assert twenty_day["required_min_observations"] == 21
    assert twenty_day["max_observations_per_ticker"] == 21
    assert twenty_day["supported"] is True
    assert twenty_day["insufficient_data"] is False
    assert twenty_day["insufficient_data_status"] is None
    assert twenty_day["insufficient_data_code"] is None
    assert twenty_day["mean_rank_ic"] < 0
    assert twenty_day["rank_ic_count"] == 21

    payload = json.loads(report.to_json())
    assert payload["metrics"]["horizon_metrics"]["20d"]["status"] == "fail"
    assert payload["metrics"]["horizon_metrics"]["20d"]["insufficient_data"] is False
    markdown = report.to_markdown()
    assert "| 20d | required | decision | forward_return_20 | True | fail | False |" in markdown
    html = report.to_html()
    assert "<td>20d</td>" in html
    assert "<td>required</td>" in html
    assert "<td>fail</td>" in html


def test_positive_fold_ratio_is_core_strategy_candidate_pass_criterion() -> None:
    report = _sufficient_report_with_low_positive_fold_ratio()

    assert report.system_validity_status == "pass"
    assert report.system_validity_pass is True
    assert report.strategy_candidate_status == "fail"
    assert report.strategy_pass is False

    rank_gate = report.gate_results["rank_ic"]
    assert rank_gate["status"] == "fail"
    assert rank_gate["reason_metadata"] == {
        "code": "positive_fold_ratio_below_minimum",
        "metric": "positive_fold_ratio",
        "value": pytest.approx(8 / 21),
        "threshold": 0.65,
        "operator": ">=",
    }
    assert "positive_fold_ratio=0.3810 is below required=0.6500" == rank_gate["reason"]

    twenty_day = report.metrics["horizon_metrics"]["20d"]
    assert twenty_day["status"] == "fail"
    assert twenty_day["mean_rank_ic"] > 0
    assert twenty_day["oos_rank_ic"] > 0
    assert twenty_day["positive_fold_ratio"] == pytest.approx(8 / 21)
    assert twenty_day["positive_fold_ratio_threshold"] == 0.65
    assert twenty_day["positive_fold_ratio_passed"] is False
    assert report.metrics["positive_fold_ratio"] == pytest.approx(8 / 21)
    assert report.metrics["positive_fold_ratio_threshold"] == 0.65
    assert report.metrics["positive_fold_ratio_passed"] is False

    failure_rows = report.strategy_failure_summary
    assert any(
        row["gate"] == "rank_ic"
        and row["metric"] == "positive_fold_ratio"
        and row["reason_code"] == "positive_fold_ratio_below_minimum"
        for row in failure_rows
    )

    payload = json.loads(report.to_json())
    assert payload["metrics"]["positive_fold_ratio_threshold"] == 0.65
    assert payload["metrics"]["positive_fold_ratio_passed"] is False
    assert payload["metrics"]["horizon_metrics"]["20d"]["positive_fold_ratio_passed"] is False
    assert payload["full_model_metrics"]["validation_metrics"]["positive_fold_ratio_threshold"] == 0.65
    assert payload["full_model_metrics"]["validation_metrics"]["positive_fold_ratio_passed"] is False

    markdown = report.to_markdown()
    assert "- Positive fold ratio threshold: 0.6500" in markdown
    assert "- Positive fold ratio threshold passed: `False`" in markdown
    assert "| Positive Fold Ratio Threshold | 0.6500 |" in markdown
    assert "| Positive Fold Ratio Passed | False |" in markdown

    html = report.to_html()
    assert "<td>Positive fold ratio threshold</td>" in html
    assert "<td>0.6500</td>" in html
    assert "<td>Positive fold ratio threshold passed</td>" in html


def test_positive_fold_ratio_passes_strategy_candidate_when_threshold_is_met() -> None:
    report = _sufficient_report_with_passing_positive_fold_ratio()

    assert report.system_validity_status == "pass"
    assert report.system_validity_pass is True
    assert report.strategy_candidate_status == "pass"
    assert report.strategy_pass is True

    rank_gate = report.gate_results["rank_ic"]
    assert rank_gate["status"] == "pass"
    assert rank_gate["reason"] == "rank IC thresholds passed"

    twenty_day = report.metrics["horizon_metrics"]["20d"]
    assert twenty_day["status"] == "pass"
    assert twenty_day["mean_rank_ic"] > 0
    assert twenty_day["oos_rank_ic"] > 0
    assert twenty_day["positive_fold_ratio"] == pytest.approx(14 / 21)
    assert twenty_day["positive_fold_ratio_threshold"] == 0.65
    assert twenty_day["positive_fold_ratio_passed"] is True
    assert report.metrics["positive_fold_ratio"] == pytest.approx(14 / 21)
    assert report.metrics["positive_fold_ratio_threshold"] == 0.65
    assert report.metrics["positive_fold_ratio_passed"] is True

    payload = json.loads(report.to_json())
    assert payload["metrics"]["positive_fold_ratio_passed"] is True
    assert payload["metrics"]["horizon_metrics"]["20d"]["positive_fold_ratio_passed"] is True
    assert payload["full_model_metrics"]["validation_metrics"]["positive_fold_ratio_passed"] is True

    markdown = report.to_markdown()
    assert "- Positive fold ratio threshold: 0.6500" in markdown
    assert "- Positive fold ratio threshold passed: `True`" in markdown


def test_twenty_day_robustness_reports_insufficient_data_without_gating_required_validity() -> None:
    report = _report_with_insufficient_twenty_day_robustness()

    assert report.system_validity_status == "not_evaluable"
    assert report.system_validity_pass is False
    assert report.metrics["insufficient_data"] is True
    assert report.gate_results["rank_ic"]["status"] == "insufficient_data"

    diagnostic_five_day = report.metrics["horizon_metrics"]["5d"]
    assert diagnostic_five_day["status"] == "pass"
    assert diagnostic_five_day["affects_pass_fail"] is False
    assert diagnostic_five_day["insufficient_data"] is False

    twenty_day = report.metrics["horizon_metrics"]["20d"]
    assert twenty_day["label"] == "required"
    assert twenty_day["role"] == "decision"
    assert twenty_day["target_column"] == "forward_return_20"
    assert twenty_day["affects_pass_fail"] is True
    assert twenty_day["minimum_observation_guard"] is True
    assert twenty_day["required_min_observations"] == 21
    assert twenty_day["max_observations_per_ticker"] == 5
    assert twenty_day["supported"] is False
    assert twenty_day["status"] == "insufficient_data"
    assert twenty_day["rank_ic_status"] == "insufficient_data"
    assert twenty_day["insufficient_data"] is True
    assert twenty_day["insufficient_data_status"] == "insufficient_data"
    assert twenty_day["insufficient_data_code"] == "insufficient_window_observations"
    assert "requires 21 observations for the 20d window" in twenty_day["rank_ic_reason"]
    assert twenty_day["rank_ic_count"] == 0

    payload = json.loads(report.to_json())
    payload_twenty_day = payload["metrics"]["horizon_metrics"]["20d"]
    assert payload_twenty_day["status"] == "insufficient_data"
    assert payload_twenty_day["insufficient_data_code"] == "insufficient_window_observations"

    markdown = report.to_markdown()
    assert (
        "| 20d | required | decision | forward_return_20 | True | insufficient_data | True |"
        in markdown
    )
    assert "insufficient_window_observations" in markdown

    html = report.to_html()
    assert "<td>20d</td>" in html
    assert "<td>insufficient_window_observations</td>" in html


def test_one_day_diagnostic_remains_non_gating_when_insufficient_data() -> None:
    report = _insufficient_report()

    assert report.system_validity_status == "not_evaluable"
    assert report.strategy_candidate_status == "insufficient_data"
    assert report.system_validity_pass is False
    assert report.hard_fail is False
    assert report.gate_results["rank_ic"]["status"] == "insufficient_data"

    one_day = report.metrics["horizon_metrics"]["1d"]
    assert one_day["label"] == "diagnostic"
    assert one_day["role"] == "diagnostic"
    assert one_day["target_column"] == "forward_return_1"
    assert one_day["status"] == "insufficient_data"
    assert one_day["rank_ic_status"] == "insufficient_data"
    assert one_day["affects_pass_fail"] is False
    assert one_day["insufficient_data"] is True
    assert one_day["insufficient_data_code"] == "empty_predictions"
    assert report.metrics["diagnostic_horizon_metrics"]["1d"] == one_day

    diagnostic_gate = report.gate_results["rank_ic_1d_diagnostic"]
    assert diagnostic_gate["status"] == "diagnostic"
    assert diagnostic_gate["diagnostic_status"] == "insufficient_data"
    assert diagnostic_gate["rank_ic_status"] == "insufficient_data"
    assert diagnostic_gate["affects_pass_fail"] is False
    assert diagnostic_gate["affects_system"] is False
    assert diagnostic_gate["affects_strategy"] is False

    payload = report.to_dict()
    assert payload["gate_results"]["rank_ic_1d_diagnostic"]["status"] == "diagnostic"
    assert payload["gate_results"]["rank_ic_1d_diagnostic"]["affects_system"] is False
    assert payload["gate_results"]["rank_ic_1d_diagnostic"]["affects_strategy"] is False
    assert not any(
        reason.startswith("rank_ic_1d_diagnostic:")
        for reason in payload["metrics"]["insufficient_data_reasons"]
    )


def test_validity_gate_insufficient_data_json_markdown_html_snapshots(tmp_path) -> None:
    report = _insufficient_report()

    payload = json.loads(report.to_json())
    assert {
        "system_validity_status": payload["system_validity_status"],
        "strategy_candidate_status": payload["strategy_candidate_status"],
        "hard_fail": payload["hard_fail"],
        "warning": payload["warning"],
        "strategy_pass": payload["strategy_pass"],
        "system_validity_pass": payload["system_validity_pass"],
        "metrics": {
            "fold_count": payload["metrics"]["fold_count"],
            "oos_fold_count": payload["metrics"]["oos_fold_count"],
            "insufficient_data": payload["metrics"]["insufficient_data"],
            "insufficient_data_reasons": payload["metrics"]["insufficient_data_reasons"],
            "target_column": payload["metrics"]["target_column"],
        },
        "gate_statuses": _gate_status_snapshot(payload),
        "horizons": _horizon_json_snapshot(payload),
    } == {
        "system_validity_status": "not_evaluable",
        "strategy_candidate_status": "insufficient_data",
        "hard_fail": False,
        "warning": True,
        "strategy_pass": False,
        "system_validity_pass": False,
        "metrics": {
            "fold_count": 0,
            "oos_fold_count": 0,
            "insufficient_data": True,
            "insufficient_data_reasons": [
                "walk_forward_oos: not enough labeled dates to create a walk-forward fold",
                "rank_ic: rank IC is not evaluable because the dataset has max_observations_per_ticker=0 but requires 21 observations for the 20d window",
                "cost_adjusted_performance: cost-adjusted baselines are unavailable",
                "benchmark_comparison: benchmark comparison is unavailable",
                "monthly_turnover_budget: monthly turnover is unavailable",
            ],
            "target_column": "forward_return_20",
        },
        "gate_statuses": {
            "leakage": "pass",
            "walk_forward_oos": "insufficient_data",
            "rank_ic": "insufficient_data",
            "cost_adjusted_performance": "not_evaluable",
            "benchmark_comparison": "not_evaluable",
            "turnover": "pass",
            "drawdown": "pass",
            "ablation": "warning",
        },
        "horizons": {
            "1d": {
                "label": "diagnostic",
                "role": "diagnostic",
                "target_column": "forward_return_1",
                "status": "insufficient_data",
                "affects_pass_fail": False,
                "insufficient_data": True,
                "insufficient_data_code": "empty_predictions",
            },
            "5d": {
                "label": "diagnostic",
                "role": "diagnostic",
                "target_column": "forward_return_5",
                "status": "insufficient_data",
                "affects_pass_fail": False,
                "insufficient_data": True,
                "insufficient_data_code": "empty_predictions",
            },
            "20d": {
                "label": "required",
                "role": "decision",
                "target_column": "forward_return_20",
                "status": "insufficient_data",
                "affects_pass_fail": True,
                "insufficient_data": True,
                "insufficient_data_code": "insufficient_window_observations",
            },
        },
    }

    json_path, markdown_path = write_validity_gate_artifacts(report, tmp_path)
    assert json.loads(json_path.read_text(encoding="utf-8")) == payload

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "- System validity: `not_evaluable`" in markdown
    assert "- Strategy candidate: `insufficient_data`" in markdown
    assert "| walk_forward_oos | insufficient_data | not enough labeled dates" in markdown
    assert "| 20d | required | decision | forward_return_20 | True | insufficient_data | True |" in markdown
    assert "empty_predictions" in markdown
    assert "insufficient_window_observations" in markdown

    html = report.to_html()
    assert "<h2>Status</h2>" in html
    assert "<td>not_evaluable</td>" in html
    assert "<td>insufficient_data</td>" in html
    assert "<td>empty_predictions</td>" in html
    assert "<td>insufficient_window_observations</td>" in html
    assert "<h2>Metrics</h2>" in html


def test_streamlit_horizon_rows_format_sufficient_and_insufficient_reports() -> None:
    sufficient_rows = {
        row["horizon"]: row for row in streamlit_app._validity_horizon_rows(_sufficient_report())
    }
    insufficient_rows = {
        row["horizon"]: row for row in streamlit_app._validity_horizon_rows(_insufficient_report())
    }

    assert list(sufficient_rows) == ["1d", "5d", "20d"]
    assert list(insufficient_rows) == ["1d", "5d", "20d"]
    assert {
        horizon: {
            "horizon_periods": row["horizon_periods"],
            "label": row["label"],
            "role": row["role"],
            "status": row["status"],
            "insufficient_data_status": row["insufficient_data_status"],
            "insufficient_data_code": row["insufficient_data_code"],
        }
        for horizon, row in sufficient_rows.items()
    } == {
        "1d": {
            "horizon_periods": 1,
            "label": "diagnostic",
            "role": "diagnostic",
            "status": "pass",
            "insufficient_data_status": None,
            "insufficient_data_code": None,
        },
        "5d": {
            "horizon_periods": 5,
            "label": "diagnostic",
            "role": "diagnostic",
            "status": "pass",
            "insufficient_data_status": None,
            "insufficient_data_code": None,
        },
        "20d": {
            "horizon_periods": 20,
            "label": "required",
            "role": "decision",
            "status": "pass",
            "insufficient_data_status": None,
            "insufficient_data_code": None,
        },
    }
    assert insufficient_rows["5d"]["status"] == "insufficient_data"
    assert insufficient_rows["5d"]["insufficient_data_status"] == "insufficient_data"
    assert insufficient_rows["5d"]["insufficient_data_code"] == "empty_predictions"
    assert insufficient_rows["20d"]["insufficient_data_code"] == "insufficient_window_observations"


def test_streamlit_structured_warning_rows_include_turnover_budget_warning() -> None:
    report = build_validity_gate_report(
        _sufficient_predictions(),
        _sufficient_validation_summary(),
        _sufficient_equity_curve(),
        _strategy_metrics(),
        ablation_summary=_stage1_ablation_summary(),
        config=_stage1_config(),
        thresholds=ValidationGateThresholds(monthly_turnover_budget=0.50),
    )

    rows = streamlit_app._validity_structured_warning_rows(report)

    assert report.strategy_candidate_status == "pass"
    assert rows
    assert rows[0]["code"] == "monthly_turnover_budget_exceeded"
    assert rows[0]["gate"] == "monthly_turnover_budget"
    assert rows[0]["value"] == pytest.approx(2.10)
    assert rows[0]["threshold"] == 0.50
    assert rows[0]["realized_turnover"] == pytest.approx(2.10)
    assert rows[0]["budget"] == 0.50
    assert "realized max monthly turnover" in str(rows[0]["message"])

    turnover_rows = streamlit_app._validity_turnover_budget_warning_rows(report)
    assert turnover_rows == rows


def test_streamlit_gate_result_rows_include_cost_collapse_reason_metadata() -> None:
    report = build_validity_gate_report(
        _sufficient_predictions(),
        _sufficient_validation_summary(),
        _sufficient_equity_curve(),
        _strategy_metrics(),
        ablation_summary=_stage1_ablation_summary(),
        config=_stage1_config(),
        thresholds=ValidationGateThresholds(cost_adjusted_collapse_threshold=2.0),
    )

    rows = {
        row["gate"]: row
        for row in streamlit_app._validity_gate_result_rows(report)
    }

    assert report.strategy_candidate_status == "fail"
    assert rows["cost_adjusted_performance"]["status"] == "fail"
    assert rows["cost_adjusted_performance"]["reason_code"] == (
        "cost_adjusted_cumulative_return_at_or_below_collapse_threshold"
    )
    assert rows["cost_adjusted_performance"]["metric"] == "cost_adjusted_cumulative_return"
    assert rows["cost_adjusted_performance"]["threshold"] == 2.0
    assert rows["cost_adjusted_performance"]["operator"] == ">"


def test_streamlit_status_metric_and_failure_rows_cover_pass_and_fail_reports() -> None:
    pass_report = _sufficient_report()
    fail_report = build_validity_gate_report(
        _sufficient_predictions(),
        _sufficient_validation_summary(),
        _sufficient_equity_curve(),
        _strategy_metrics(),
        ablation_summary=_stage1_ablation_summary(),
        config=_stage1_config(),
        thresholds=ValidationGateThresholds(cost_adjusted_collapse_threshold=2.0),
    )

    pass_status = streamlit_app._validity_final_strategy_status_rows(pass_report)
    fail_status = streamlit_app._validity_final_strategy_status_rows(fail_report)
    assert pass_status[0]["final_strategy_status"] == "pass"
    assert pass_status[0]["system_validity_status"] == "pass"
    assert pass_status[0]["strategy_pass"] is True
    assert pass_status[0]["blocking_rules"] == []
    assert pass_status[0]["warning_rules"] == []
    assert pass_status[0]["insufficient_data_rules"] == []
    assert "passed" in pass_status[0]["reason"]
    assert fail_status[0]["final_strategy_status"] == "fail"
    assert fail_status[0]["system_validity_status"] == "pass"
    assert fail_status[0]["strategy_pass"] is False
    assert fail_status[0]["blocking_rules"] == [
        "cost_adjusted_performance",
        "deterministic_strategy_validity",
    ]
    assert fail_status[0]["official_message"] == OFFICIAL_STRATEGY_FAIL_MESSAGE

    pass_full_model_rows = streamlit_app._validity_full_model_metric_rows(pass_report)
    fail_full_model_rows = streamlit_app._validity_full_model_metric_rows(fail_report)
    assert pass_full_model_rows[0]["entity_id"] == "all_features"
    assert pass_full_model_rows[0]["role"] == "full_model"
    assert pass_full_model_rows[0]["status"] == "pass"
    assert pass_full_model_rows[0]["sharpe"] == pytest.approx(1.25)
    assert pass_full_model_rows[0]["mean_rank_ic"] > 0
    assert pass_full_model_rows[0]["cost_adjusted_cumulative_return"] > 0
    assert fail_full_model_rows[0]["entity_id"] == "all_features"
    assert fail_full_model_rows[0]["status"] == "fail"

    baseline_rows = streamlit_app._validity_metric_contract_rows(
        pass_report.baseline_metrics
    )
    ablation_rows = streamlit_app._validity_metric_contract_rows(
        pass_report.ablation_metrics
    )
    assert {"entity_id", "role", "status", "sharpe"}.issubset(baseline_rows[0])
    assert {"no_model_proxy", "return_baseline_spy", "return_baseline_equal_weight"}.issubset(
        {row["entity_id"] for row in baseline_rows}
    )
    assert {"entity_id", "role", "kind", "status", "sharpe"}.issubset(
        ablation_rows[0]
    )
    assert {"price_only", "text_only", "sec_only", "no_costs"}.issubset(
        {row["entity_id"] for row in ablation_rows}
    )

    assert streamlit_app._validity_strategy_failure_rows(pass_report) == []
    failure_rows = streamlit_app._validity_strategy_failure_rows(fail_report)
    assert failure_rows[0]["gate"] == "cost_adjusted_performance"
    assert failure_rows[0]["collapse_status"] == "fail"
    assert failure_rows[0]["reason_code"] == (
        "cost_adjusted_cumulative_return_at_or_below_collapse_threshold"
    )


def test_report_outputs_surface_strategy_fail_status_and_collapse_reason(tmp_path) -> None:
    report = build_validity_gate_report(
        _sufficient_predictions(),
        _sufficient_validation_summary(),
        _sufficient_equity_curve(),
        _strategy_metrics(),
        ablation_summary=_stage1_ablation_summary(),
        config=_stage1_config(),
        thresholds=ValidationGateThresholds(cost_adjusted_collapse_threshold=2.0),
    )

    json_path, markdown_path = write_validity_gate_artifacts(report, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert payload["strategy_candidate_status"] == "fail"
    assert payload["official_message"] == OFFICIAL_STRATEGY_FAIL_MESSAGE
    assert payload["collapse_status"] == "fail"
    assert payload["collapse_reason_code"] == (
        "cost_adjusted_cumulative_return_at_or_below_collapse_threshold"
    )
    assert "cost-adjusted cumulative return" in payload["collapse_reason"]
    assert payload["strategy_failure_summary"][0]["gate"] == "cost_adjusted_performance"
    assert payload["strategy_failure_summary"][0]["status"] == "fail"
    assert payload["strategy_failure_summary"][0]["collapse_status"] == "fail"
    assert payload["strategy_failure_summary"][0]["reason_code"] == (
        "cost_adjusted_cumulative_return_at_or_below_collapse_threshold"
    )

    assert "- Strategy candidate: `fail`" in markdown
    assert f"- Official message: {OFFICIAL_STRATEGY_FAIL_MESSAGE}" in markdown
    assert "- Collapse status: `fail`" in markdown
    assert "- Collapse reason: cost-adjusted cumulative return" in markdown
    assert "## Strategy Failure Summary" in markdown
    assert "| cost_adjusted_performance | fail | cost-adjusted cumulative return" in markdown
    assert "cost_adjusted_cumulative_return_at_or_below_collapse_threshold" in markdown

    html = report.to_html()
    assert "<td>Strategy candidate</td>" in html
    assert "<td>fail</td>" in html
    assert f"<td>{OFFICIAL_STRATEGY_FAIL_MESSAGE}</td>" in html
    assert "<td>Collapse status</td>" in html
    assert "<td>Collapse reason</td>" in html
    assert "<h2>Strategy Failure Summary</h2>" in html
    assert "<td>cost_adjusted_performance</td>" in html
    assert "<td>cost_adjusted_cumulative_return_at_or_below_collapse_threshold</td>" in html


def test_markdown_html_surface_turnover_budget_warning_details_without_regating() -> None:
    report = build_validity_gate_report(
        _sufficient_predictions(),
        _sufficient_validation_summary(),
        _sufficient_equity_curve(),
        _strategy_metrics(),
        ablation_summary=_stage1_ablation_summary(),
        config=_stage1_config(),
        thresholds=ValidationGateThresholds(monthly_turnover_budget=0.50),
    )

    assert report.gate_results["turnover"]["status"] == "pass"
    assert report.gate_results["monthly_turnover_budget"]["affects_strategy"] is False
    assert report.strategy_candidate_status == "pass"

    markdown = report.to_markdown()
    assert (
        "| Code | Gate | Combined Gate | Severity | Metric | Realized Turnover | Budget |"
        in markdown
    )
    assert (
        "| monthly_turnover_budget_exceeded | monthly_turnover_budget | turnover | "
        "warning | max_monthly_turnover | 2.1000 | 0.5000 | 2.1000 | 0.5000 |"
        in markdown
    )
    assert (
        "monthly_turnover_budget: realized max monthly turnover 2.1000 "
        "exceeds configured budget 0.5000"
    ) in markdown

    html = report.to_html()
    assert "<th>Combined Gate</th>" in html
    assert "<th>Realized Turnover</th>" in html
    assert "<th>Budget</th>" in html
    assert "<th>Message</th>" in html
    assert "<td>monthly_turnover_budget_exceeded</td>" in html
    assert "<td>monthly_turnover_budget</td>" in html
    assert "<td>turnover</td>" in html
    assert "<td>2.1000</td>" in html
    assert (
        "<td>monthly_turnover_budget: realized max monthly turnover 2.1000 "
        "exceeds configured budget 0.5000</td>"
    ) in html


class _FakeMetricColumn:
    def __init__(self) -> None:
        self.metrics: list[tuple[str, object]] = []
        self.captions: list[str] = []

    def metric(self, label: str, value: object) -> None:
        self.metrics.append((label, value))

    def caption(self, value: object) -> None:
        self.captions.append(str(value))


class _FakeStreamlit:
    def __init__(self) -> None:
        self.column_groups: list[list[_FakeMetricColumn]] = []
        self.dataframes: list[pd.DataFrame] = []
        self.subheaders: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def columns(self, count: int) -> list[_FakeMetricColumn]:
        columns = [_FakeMetricColumn() for _ in range(count)]
        self.column_groups.append(columns)
        return columns

    def dataframe(self, value: pd.DataFrame, **_: object) -> None:
        self.dataframes.append(value)

    def subheader(self, value: str) -> None:
        self.subheaders.append(value)

    def warning(self, value: str) -> None:
        self.warnings.append(str(value))

    def error(self, value: str) -> None:
        self.errors.append(str(value))


def test_streamlit_failure_summary_renderer_surfaces_collapse_reason(monkeypatch) -> None:
    report = build_validity_gate_report(
        _sufficient_predictions(),
        _sufficient_validation_summary(),
        _sufficient_equity_curve(),
        _strategy_metrics(),
        ablation_summary=_stage1_ablation_summary(),
        config=_stage1_config(),
        thresholds=ValidationGateThresholds(cost_adjusted_collapse_threshold=2.0),
    )
    fake_streamlit = _FakeStreamlit()
    monkeypatch.setattr(streamlit_app, "st", fake_streamlit)

    streamlit_app._render_validity_failure_summary(report)

    assert fake_streamlit.errors
    assert fake_streamlit.errors[0] == OFFICIAL_STRATEGY_FAIL_MESSAGE
    assert fake_streamlit.subheaders == ["Validity Gate Failure Summary"]
    assert len(fake_streamlit.dataframes) == 1
    failure_table = fake_streamlit.dataframes[0]
    assert failure_table.loc[0, "gate"] == "cost_adjusted_performance"
    assert failure_table.loc[0, "status"] == "fail"
    assert failure_table.loc[0, "collapse_status"] == "fail"
    assert failure_table.loc[0, "reason_code"] == (
        "cost_adjusted_cumulative_return_at_or_below_collapse_threshold"
    )


def test_streamlit_turnover_budget_warning_renderer_surfaces_message_and_budget(
    monkeypatch,
) -> None:
    report = build_validity_gate_report(
        _sufficient_predictions(),
        _sufficient_validation_summary(),
        _sufficient_equity_curve(),
        _strategy_metrics(),
        ablation_summary=_stage1_ablation_summary(),
        config=_stage1_config(),
        thresholds=ValidationGateThresholds(monthly_turnover_budget=0.50),
    )
    fake_streamlit = _FakeStreamlit()
    monkeypatch.setattr(streamlit_app, "st", fake_streamlit)

    streamlit_app._render_turnover_budget_warnings(report)

    assert fake_streamlit.subheaders == ["Validity Gate Turnover Budget Warnings"]
    assert len(fake_streamlit.warnings) == 1
    assert "monthly_turnover_budget: realized max monthly turnover" in fake_streamlit.warnings[0]
    assert len(fake_streamlit.dataframes) == 1
    warning_table = fake_streamlit.dataframes[0]
    assert {
        "code",
        "combined_gate",
        "realized_turnover",
        "budget",
        "message",
        "monthly_turnover",
    }.issubset(warning_table.columns)
    assert warning_table.loc[0, "code"] == "monthly_turnover_budget_exceeded"
    assert warning_table.loc[0, "realized_turnover"] == pytest.approx(2.10)
    assert warning_table.loc[0, "budget"] == 0.50


@pytest.mark.parametrize(
    ("report_factory", "expected_metrics", "expected_caption_fragments"),
    [
        (
            _sufficient_report,
            [("1d Diagnostic", "pass"), ("20d Required", "pass")],
            [],
        ),
        (
            _insufficient_report,
            [
                ("1d Diagnostic", "insufficient_data"),
                ("20d Required", "insufficient_data"),
            ],
            [
                "Insufficient data: insufficient_data",
                "requires 21 observations for the 20d window",
                "rank IC is not evaluable because predictions are empty",
            ],
        ),
    ],
)
def test_streamlit_horizon_status_summary_formatting(
    monkeypatch,
    report_factory,
    expected_metrics,
    expected_caption_fragments,
) -> None:
    fake_streamlit = _FakeStreamlit()
    monkeypatch.setattr(streamlit_app, "st", fake_streamlit)

    streamlit_app._render_horizon_status_summary(
        streamlit_app._validity_horizon_rows(report_factory())
    )

    assert len(fake_streamlit.column_groups) == 1
    columns = fake_streamlit.column_groups[0]
    rendered_metrics = [column.metrics[0] for column in columns]
    rendered_captions = [caption for column in columns for caption in column.captions]

    assert rendered_metrics == expected_metrics
    for fragment in expected_caption_fragments:
        assert any(fragment in caption for caption in rendered_captions)
