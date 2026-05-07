from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research.validation import ValidationGateThresholds, build_validity_gate_report


def _predictions() -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=21, freq="B")
    rows: list[dict[str, object]] = []
    for fold, date in enumerate(dates):
        for ticker, expected, forward_20 in (
            ("AAPL", 0.030, 0.080),
            ("MSFT", 0.020, 0.050),
            ("SPY", 0.010, 0.020),
        ):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "fold": fold,
                    "is_oos": fold >= 19,
                    "expected_return": expected,
                    "forward_return_1": forward_20 / 20.0,
                    "forward_return_5": forward_20 / 4.0,
                    "forward_return_20": forward_20,
                }
            )
    return pd.DataFrame(rows)


def _validation_summary() -> pd.DataFrame:
    test_starts = pd.date_range("2026-02-02", periods=21, freq="B")
    return pd.DataFrame(
        {
            "fold": range(len(test_starts)),
            "train_start": pd.date_range("2025-01-02", periods=21, freq="B"),
            "train_end": test_starts - pd.Timedelta(days=30),
            "test_start": test_starts,
            "test_end": test_starts + pd.Timedelta(days=27),
            "is_oos": [False] * 19 + [True, True],
            "labeled_test_observations": [3] * len(test_starts),
            "train_observations": [252] * len(test_starts),
            "train_periods": [252] * len(test_starts),
            "test_periods": [60] * len(test_starts),
            "target_column": ["forward_return_20"] * len(test_starts),
            "prediction_horizon_periods": [20] * len(test_starts),
            "gap_periods": [20] * len(test_starts),
            "purge_periods": [20] * len(test_starts),
            "purge_gap_periods": [20] * len(test_starts),
            "purged_date_count": [20] * len(test_starts),
            "purge_applied": [True] * len(test_starts),
            "embargo_periods": [20] * len(test_starts),
            "embargoed_date_count": [20] * len(test_starts),
            "embargo_applied": [True] * len(test_starts),
        }
    )


def _equity_curve(*, turnover: float = 0.10) -> pd.DataFrame:
    dates = pd.date_range("2026-03-02", periods=21, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": [0.040] * len(dates),
            "gross_return": [0.041] * len(dates),
            "cost_adjusted_return": [0.040] * len(dates),
            "benchmark_return": [0.005] * len(dates),
            "turnover": [turnover] * len(dates),
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


def _ablation_summary() -> list[dict[str, object]]:
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
            "deterministic_signal_evaluation_metrics": {
                "return_basis": "cost_adjusted_return",
                "cost_adjusted_cumulative_return": 0.10,
                "average_daily_turnover": 0.10,
                "total_cost_return": 0.01,
            },
        },
        {"scenario": "no_costs", "kind": "cost", "sharpe": 1.10, "excess_return": 0.14},
    ]


def _config(**overrides: object) -> SimpleNamespace:
    values = {
        "tickers": ["AAPL", "MSFT"],
        "prediction_target_column": "forward_return_20",
        "required_validation_horizon": 20,
        "benchmark_ticker": "SPY",
        "gap_periods": 20,
        "embargo_periods": 20,
        "cost_bps": 5.0,
        "slippage_bps": 2.0,
        "top_n": 20,
        "max_symbol_weight": 0.10,
        "max_sector_weight": 0.30,
        "portfolio_covariance_lookback": 20,
        "covariance_aware_risk_enabled": True,
        "covariance_return_column": "return_1",
        "covariance_min_periods": 20,
        "portfolio_volatility_limit": 0.04,
        "max_position_risk_contribution": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _metrics(**overrides: object) -> SimpleNamespace:
    values = {
        "cagr": 1.50,
        "sharpe": 1.25,
        "max_drawdown": -0.04,
        "turnover": 0.10,
        "average_portfolio_volatility_estimate": 0.018,
        "max_portfolio_volatility_estimate": 0.018,
        "max_position_risk_contribution": 0.55,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _thresholds() -> ValidationGateThresholds:
    return ValidationGateThresholds(
        min_folds=5,
        min_oos_folds=2,
        min_rank_ic=0.02,
        min_positive_fold_ratio=0.65,
        max_daily_turnover=0.25,
        max_monthly_turnover=6.0,
        drawdown_pass=-0.20,
        drawdown_warning=-0.35,
    )


def _report(
    *,
    predictions: pd.DataFrame | None = None,
    validation_summary: pd.DataFrame | None = None,
    equity_curve: pd.DataFrame | None = None,
    strategy_metrics: SimpleNamespace | None = None,
    config: SimpleNamespace | None = None,
    thresholds: ValidationGateThresholds | None = None,
):
    return build_validity_gate_report(
        _predictions() if predictions is None else predictions,
        _validation_summary() if validation_summary is None else validation_summary,
        _equity_curve() if equity_curve is None else equity_curve,
        _metrics() if strategy_metrics is None else strategy_metrics,
        ablation_summary=_ablation_summary(),
        config=_config() if config is None else config,
        thresholds=_thresholds() if thresholds is None else thresholds,
    )


def test_validity_gate_passes_canonical_backtest_walk_forward_oos_and_risk_combo() -> None:
    report = _report()

    assert report.system_validity_status == "pass"
    assert report.strategy_candidate_status == "pass"
    assert report.metrics["target_column"] == "forward_return_20"
    assert report.metrics["oos_fold_count"] == 2
    assert report.gate_results["leakage"]["status"] == "pass"
    assert report.gate_results["walk_forward_oos"]["status"] == "pass"
    assert report.gate_results["turnover"]["status"] == "pass"
    assert report.gate_results["drawdown"]["status"] == "pass"
    assert report.gate_results["deterministic_gate_aggregation"]["final_decision"] == "PASS"

    covariance_risk = report.metrics["covariance_aware_risk"]
    assert covariance_risk["status"] == "applied"
    assert covariance_risk["application_evidence"] == {
        "has_portfolio_volatility_estimate": True,
        "has_position_sizing_validation": True,
        "has_risk_contribution_metric": True,
    }
    assert (
        covariance_risk["realized_metrics"]["latest_position_sizing_validation_status"]
        == "pass"
    )


@pytest.mark.parametrize(
    ("case", "mutate", "failed_gate", "reason_fragment"),
    [
        (
            "forward_return_20_zero_embargo",
            lambda predictions, validation, equity, metrics, config: setattr(
                config,
                "embargo_periods",
                0,
            ),
            "leakage",
            "embargo_periods=0 is below target horizon=20",
        ),
        (
            "single_oos_fold",
            lambda predictions, validation, equity, metrics, config: validation.__setitem__(
                "is_oos",
                [False] * 20 + [True],
            ),
            "walk_forward_oos",
            "oos_fold_count=1 is below required=2",
        ),
    ],
)
def test_validity_gate_hard_fails_structural_walk_forward_oos_combinations(
    case: str,
    mutate: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, SimpleNamespace, SimpleNamespace], None],
    failed_gate: str,
    reason_fragment: str,
) -> None:
    predictions = _predictions()
    validation = _validation_summary()
    equity = _equity_curve()
    metrics = _metrics()
    config = _config()
    mutate(predictions, validation, equity, metrics, config)

    report = _report(
        predictions=predictions,
        validation_summary=validation,
        equity_curve=equity,
        strategy_metrics=metrics,
        config=config,
    )

    assert case
    assert report.system_validity_status == "hard_fail"
    assert report.strategy_candidate_status == "not_evaluable"
    assert report.gate_results[failed_gate]["status"] == "hard_fail"
    assert reason_fragment in report.gate_results[failed_gate]["reason"]
    assert report.gate_results["deterministic_gate_aggregation"]["final_decision"] == "FAIL"


def test_validity_gate_fails_strategy_when_backtest_turnover_and_risk_rules_breach() -> None:
    equity = _equity_curve(turnover=0.40)
    equity["position_sizing_validation_status"] = "fail"
    equity["position_sizing_validation_reason"] = "max_symbol_or_sector_weight_breached"
    equity["max_position_weight"] = 0.12
    equity["max_sector_exposure"] = 0.35
    metrics = _metrics(
        turnover=0.40,
        max_position_risk_contribution=1.10,
        max_portfolio_volatility_estimate=0.050,
    )
    report = _report(equity_curve=equity, strategy_metrics=metrics)

    assert report.system_validity_status == "pass"
    assert report.strategy_candidate_status == "fail"
    assert report.gate_results["walk_forward_oos"]["status"] == "pass"
    assert report.gate_results["turnover"]["status"] == "fail"
    assert report.gate_results["turnover"]["daily_status"] == "warning"
    assert report.gate_results["turnover"]["monthly_status"] == "warning"
    assert "turnover" in report.gate_results["deterministic_strategy_validity"]["failed_rules"]

    covariance_risk = report.metrics["covariance_aware_risk"]
    assert covariance_risk["status"] == "applied"
    assert covariance_risk["realized_metrics"]["latest_position_sizing_validation_status"] == "fail"
    assert covariance_risk["realized_metrics"]["latest_max_position_weight"] == pytest.approx(0.12)
    assert covariance_risk["realized_metrics"]["latest_max_sector_exposure"] == pytest.approx(0.35)


def test_validity_gate_fails_strategy_when_mean_and_oos_rank_ic_are_nonpositive() -> None:
    predictions = _predictions()
    reversed_returns = {"AAPL": 0.010, "MSFT": 0.020, "SPY": 0.030}
    predictions["forward_return_20"] = predictions["ticker"].map(reversed_returns)

    report = _report(predictions=predictions)

    assert report.system_validity_status == "pass"
    assert report.gate_results["walk_forward_oos"]["status"] == "pass"
    assert report.metrics["oos_fold_count"] == 2
    assert report.metrics["mean_rank_ic"] <= 0.0
    assert report.metrics["oos_rank_ic"] <= 0.0
    assert report.gate_results["rank_ic"]["status"] == "fail"
    assert report.strategy_candidate_status == "fail"
    assert report.gate_results["deterministic_gate_aggregation"]["final_decision"] == "FAIL"
