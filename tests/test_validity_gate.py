from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research.backtest.metrics import PerformanceMetrics
from quant_research.validation.gate import (
    OFFICIAL_STRATEGY_FAIL_MESSAGE,
    build_validity_gate_report,
    write_validity_gate_artifacts,
)
from quant_research.validation.walk_forward import WalkForwardConfig


def test_validity_gate_passes_structural_contract_for_clean_inputs() -> None:
    report = build_validity_gate_report(
        _predictions(),
        _validation_summary(),
        _equity_curve(),
        _metrics(),
        _ablation_summary(),
        config=_config(),
        walk_forward_config=WalkForwardConfig(gap_periods=5, embargo_periods=5, min_train_observations=50),
    )

    assert report.system_validity_status == "pass"
    assert report.strategy_candidate_status == "pass"
    assert report.system_validity_pass is True
    assert report.strategy_pass is True
    assert report.hard_fail is False
    assert report.warning is False
    assert report.metrics["target_column"] == "forward_return_5"
    assert report.metrics["realized_return_column"] == "forward_return_5"
    assert report.metrics["label_columns"] == ["forward_return_1", "forward_return_5"]
    assert report.metrics["label_coverage"]["forward_return_5"] == 1.0
    assert report.target_horizon == 5
    assert report.requested_gap_periods == 5
    assert report.effective_gap_periods == 5
    assert report.required_validation_horizon == "5d"
    assert {"hard_fail", "warning", "strategy_pass", "system_validity_pass"}.issubset(report.to_dict())


def test_validity_gate_hard_fails_horizon_embargo_violation() -> None:
    report = build_validity_gate_report(
        _predictions(),
        _validation_summary(),
        _equity_curve(),
        _metrics(),
        _ablation_summary(),
        config=_config(),
        walk_forward_config=WalkForwardConfig(gap_periods=1, embargo_periods=0, min_train_observations=50),
    )

    assert report.system_validity_status == "hard_fail"
    assert report.hard_fail is True
    assert any("effective_gap_periods" in reason for reason in report.hard_fail_reasons)
    assert any("effective_embargo_periods" in reason for reason in report.hard_fail_reasons)


def test_validity_gate_warns_when_requested_gap_is_raised_to_horizon() -> None:
    report = build_validity_gate_report(
        _predictions(),
        _validation_summary(),
        _equity_curve(),
        _metrics(),
        _ablation_summary(),
        config=_config(),
        walk_forward_config=WalkForwardConfig(
            gap_periods=5,
            embargo_periods=5,
            requested_gap_periods=1,
            requested_embargo_periods=0,
            target_horizon=5,
            min_train_observations=50,
        ),
    )

    assert report.system_validity_status == "pass"
    assert report.hard_fail is False
    assert report.warning is True
    assert report.requested_gap_periods == 1
    assert report.effective_gap_periods == 5
    assert any("requested_gap_periods=1" in warning for warning in report.warnings)
    assert any("requested_embargo_periods=0" in warning for warning in report.warnings)


def test_validity_gate_equal_weight_baseline_uses_selected_horizon() -> None:
    report = build_validity_gate_report(
        _predictions(),
        _validation_summary(),
        _equity_curve(),
        _metrics(),
        _ablation_summary(),
        config=_config(),
        walk_forward_config=WalkForwardConfig(gap_periods=5, embargo_periods=5, min_train_observations=50),
    )

    equal_weight = next(result for result in report.benchmark_results if result["name"] == "equal_weight")
    predictions = _predictions()
    selected_horizon_returns = predictions.groupby("date")["forward_return_5"].mean()
    one_day_returns = predictions.groupby("date")["forward_return_1"].mean()

    assert equal_weight["cagr"] == pytest.approx(_manual_cagr(selected_horizon_returns))
    assert equal_weight["cagr"] != pytest.approx(_manual_cagr(one_day_returns))


def test_validity_gate_reports_strategy_fail_separately_from_system_pass() -> None:
    weak_metrics = PerformanceMetrics(
        cagr=-0.05,
        annualized_volatility=0.10,
        sharpe=0.10,
        max_drawdown=-0.12,
        hit_rate=0.48,
        turnover=0.05,
        exposure=0.30,
        benchmark_cagr=0.10,
        excess_return=-0.15,
    )

    report = build_validity_gate_report(
        _predictions(),
        _validation_summary(),
        _equity_curve(),
        weak_metrics,
        _ablation_summary(),
        config=_config(),
        walk_forward_config=WalkForwardConfig(gap_periods=5, embargo_periods=5, min_train_observations=50),
    )

    assert report.system_validity_status == "pass"
    assert report.strategy_candidate_status == "fail"
    assert report.official_message == OFFICIAL_STRATEGY_FAIL_MESSAGE


def test_validity_gate_artifacts_write_json_and_markdown(tmp_path) -> None:
    report = build_validity_gate_report(
        _predictions(),
        _validation_summary(),
        _equity_curve(),
        _metrics(),
        _ablation_summary(),
        config=_config(),
        walk_forward_config=WalkForwardConfig(gap_periods=5, embargo_periods=5, min_train_observations=50),
    )

    json_path, markdown_path = write_validity_gate_artifacts(report, tmp_path)

    assert json_path.read_text(encoding="utf-8").startswith("{")
    assert "Validity Gate Report" in markdown_path.read_text(encoding="utf-8")


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        prediction_target_column="forward_return_5",
        required_validation_horizon=5,
        benchmark_ticker="SPY",
    )


def _metrics() -> PerformanceMetrics:
    return PerformanceMetrics(
        cagr=0.30,
        annualized_volatility=0.12,
        sharpe=1.10,
        max_drawdown=-0.02,
        hit_rate=0.58,
        turnover=0.10,
        exposure=0.60,
        benchmark_cagr=0.10,
        excess_return=0.20,
    )


def _validation_summary() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=60)
    return pd.DataFrame(
        {
            "fold": list(range(5)),
            "train_start": dates[[0, 5, 10, 15, 20]],
            "train_end": dates[[10, 15, 20, 25, 30]],
            "test_start": dates[[16, 21, 26, 31, 36]],
            "test_end": dates[[20, 25, 30, 35, 40]],
            "train_observations": [60, 64, 68, 72, 76],
            "labeled_test_observations": [12, 12, 12, 12, 12],
            "is_oos": [False, False, False, False, True],
        }
    )


def _predictions() -> pd.DataFrame:
    rows = []
    for fold, date_value in enumerate(pd.bdate_range("2026-03-02", periods=5)):
        is_oos = fold == 4
        for rank, ticker in enumerate(["SPY", "AAPL", "MSFT"]):
            expected = 0.003 - rank * 0.001
            rows.append(
                {
                    "date": date_value,
                    "ticker": ticker,
                    "expected_return": expected,
                    "forward_return_5": expected * 0.2,
                    "forward_return_1": 0.0005 + rank * 0.0001,
                    "fold": fold,
                    "is_oos": is_oos,
                }
            )
    return pd.DataFrame(rows)


def _equity_curve() -> pd.DataFrame:
    returns = [0.004, 0.003, -0.001, 0.005, 0.002]
    benchmark = [0.001, 0.001, -0.001, 0.001, 0.001]
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2026-03-02", periods=5),
            "portfolio_return": returns,
            "benchmark_return": benchmark,
            "equity": pd.Series([1 + value for value in returns]).cumprod(),
            "benchmark_equity": pd.Series([1 + value for value in benchmark]).cumprod(),
            "turnover": [0.10] * 5,
            "exposure": [0.60] * 5,
        }
    )


def _ablation_summary() -> list[dict[str, object]]:
    return [
        {"scenario": "all_features", "sharpe": 1.10, "excess_return": 0.20},
        {"scenario": "full_model_features", "sharpe": 1.10, "excess_return": 0.20},
        {"scenario": "price_only", "sharpe": 0.70, "excess_return": 0.05},
        {"scenario": "text_only", "sharpe": 0.20, "excess_return": -0.01},
        {"scenario": "sec_only", "sharpe": 0.15, "excess_return": -0.02},
        {"scenario": "no_model_proxy", "sharpe": 0.80, "excess_return": 0.08},
        {"scenario": "no_costs", "sharpe": 1.20, "excess_return": 0.25},
    ]


def _manual_cagr(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    years = max(len(values) / 252, 1 / 252)
    equity = (1 + values).cumprod()
    return float(equity.iloc[-1] ** (1 / years) - 1)
