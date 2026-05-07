from __future__ import annotations

import pandas as pd
import pytest

from quant_research.backtest.engine import BacktestConfig, run_long_only_backtest
from quant_research.backtest.metrics import (
    analyze_transaction_cost_scenarios,
    calculate_average_daily_turnover,
    calculate_cost_adjusted_returns,
    calculate_daily_position_turnover,
    calculate_metrics,
    calculate_portfolio_turnover,
    reprice_equity_curve_for_transaction_costs,
)
from quant_research.validation.config import (
    TransactionCostSensitivityConfig,
    TransactionCostSensitivityScenario,
)


def test_backtest_config_exposes_canonical_cost_slippage_and_turnover_defaults() -> None:
    config = BacktestConfig()

    assert config.cost_bps == 5.0
    assert config.slippage_bps == 2.0
    assert config.average_daily_turnover_budget == 0.25
    assert config.max_daily_turnover is None
    assert config.realized_return_column == "forward_return_20"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cost_bps": -0.01}, "cost_bps must be non-negative"),
        ({"slippage_bps": -0.01}, "slippage_bps must be non-negative"),
        ({"average_daily_turnover_budget": 0.0}, "average_daily_turnover_budget"),
        ({"average_daily_turnover_budget": 1.01}, "average_daily_turnover_budget"),
        ({"max_daily_turnover": -0.01}, "max_daily_turnover"),
        ({"max_daily_turnover": 2.01}, "max_daily_turnover"),
        ({"benchmark_ticker": " "}, "benchmark_ticker must not be blank"),
        ({"realized_return_column": "return_20"}, "forward_return_<horizon>"),
    ],
)
def test_backtest_config_validates_cost_slippage_turnover_and_return_contract(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        BacktestConfig(**kwargs)


def test_pipeline_backtest_config_carries_turnover_budget_with_effective_costs() -> None:
    from quant_research import pipeline
    from quant_research.pipeline import PipelineConfig

    config = PipelineConfig(
        cost_bps=8.0,
        slippage_bps=3.0,
        average_daily_turnover_budget=0.20,
    )

    backtest_config = pipeline._backtest_config(config)

    assert backtest_config.cost_bps == 8.0
    assert backtest_config.slippage_bps == 3.0
    assert backtest_config.average_daily_turnover_budget == 0.20


def test_portfolio_turnover_sums_absolute_weight_changes() -> None:
    turnover = calculate_portfolio_turnover(
        {"AAPL": 1.0, "MSFT": 0.25},
        {"MSFT": 0.50, "NVDA": 0.25},
    )

    assert turnover == pytest.approx(1.50)


def test_cost_adjusted_returns_apply_cost_slippage_and_turnover_drag() -> None:
    adjusted = calculate_cost_adjusted_returns(
        [0.01, -0.02],
        [0.50, 2.00],
        cost_bps=8.0,
        slippage_bps=3.0,
    )

    assert adjusted["transaction_cost_return"].tolist() == pytest.approx([0.0004, 0.0016])
    assert adjusted["slippage_cost_return"].tolist() == pytest.approx([0.00015, 0.0006])
    assert adjusted["total_cost_return"].tolist() == pytest.approx([0.00055, 0.0022])
    assert adjusted["turnover_cost_return"].tolist() == pytest.approx(
        adjusted["total_cost_return"].tolist()
    )
    assert adjusted["cost_adjusted_return"].tolist() == pytest.approx([0.00945, -0.0222])
    assert adjusted["net_return"].tolist() == pytest.approx(
        adjusted["cost_adjusted_return"].tolist()
    )

    scalar_adjusted = calculate_cost_adjusted_returns(
        0.01, [0.50, 1.00], cost_bps=10.0, slippage_bps=0.0
    )
    assert scalar_adjusted["cost_adjusted_return"].tolist() == pytest.approx([0.0095, 0.0090])


def test_cost_adjusted_returns_reject_negative_turnover_cost_and_slippage_inputs() -> None:
    with pytest.raises(ValueError, match="turnover must be non-negative"):
        calculate_cost_adjusted_returns([0.01], [-0.01], cost_bps=5.0, slippage_bps=2.0)

    with pytest.raises(ValueError, match="cost_bps must be non-negative"):
        calculate_cost_adjusted_returns([0.01], [1.0], cost_bps=-0.01, slippage_bps=2.0)

    with pytest.raises(ValueError, match="slippage_bps must be non-negative"):
        calculate_cost_adjusted_returns([0.01], [1.0], cost_bps=5.0, slippage_bps=-0.01)


def test_average_daily_turnover_comes_from_daily_position_changes() -> None:
    positions = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-03",
                ]
            ),
            "ticker": ["AAPL", "AAPL", "MSFT", "MSFT"],
            "weight": [0.50, 0.50, 0.25, 0.25],
        }
    )

    daily = calculate_daily_position_turnover(positions)
    average = calculate_average_daily_turnover(positions)

    assert daily["turnover"].tolist() == pytest.approx([0.50, 0.25, 0.50])
    assert average == pytest.approx((0.50 + 0.25 + 0.50) / 3)


def test_backtest_turnover_matches_deterministic_target_position_changes() -> None:
    frame = _prediction_frame(
        dates=[
            "2026-01-01",
            "2026-01-01",
            "2026-01-02",
            "2026-01-02",
            "2026-01-05",
            "2026-01-05",
        ],
        tickers=["AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT"],
        expected_returns=[0.05, 0.01, 0.01, 0.05, 0.05, 0.05],
        text_risk_scores=[0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    )

    result = run_long_only_backtest(
        frame,
        BacktestConfig(top_n=1, max_symbol_weight=1.0, portfolio_volatility_limit=1.0),
    )
    position_turnover = calculate_average_daily_turnover(
        result.weights,
        date_index=result.equity_curve["holding_start_date"],
    )

    assert result.equity_curve["turnover"].tolist() == pytest.approx([1.0, 2.0, 1.0])
    assert result.metrics.turnover == pytest.approx((1.0 + 2.0 + 1.0) / 3)
    assert result.metrics.turnover == pytest.approx(position_turnover)


def test_backtest_records_period_and_cumulative_turnover() -> None:
    frame = _prediction_frame(
        dates=[
            "2026-01-01",
            "2026-01-01",
            "2026-01-02",
            "2026-01-02",
            "2026-01-05",
            "2026-01-05",
        ],
        tickers=["AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT"],
        expected_returns=[0.05, 0.01, 0.01, 0.05, 0.05, 0.05],
        text_risk_scores=[0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    )

    result = run_long_only_backtest(
        frame,
        BacktestConfig(top_n=1, max_symbol_weight=1.0, portfolio_volatility_limit=1.0),
    )

    assert result.equity_curve["period_turnover"].tolist() == pytest.approx([1.0, 2.0, 1.0])
    assert result.equity_curve["period_turnover"].tolist() == pytest.approx(
        result.equity_curve["turnover"].tolist()
    )
    assert result.equity_curve["cumulative_turnover"].tolist() == pytest.approx(
        [1.0, 3.0, 4.0]
    )


def test_backtest_enforces_configured_daily_turnover_budget() -> None:
    frame = _prediction_frame(
        dates=["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"],
        tickers=["AAPL", "MSFT", "AAPL", "MSFT"],
        expected_returns=[0.05, 0.01, 0.01, 0.05],
        text_risk_scores=[0.0, 0.0, 0.0, 0.0],
    )

    result = run_long_only_backtest(
        frame,
        BacktestConfig(
            top_n=1,
            max_symbol_weight=1.0,
            portfolio_volatility_limit=1.0,
            max_daily_turnover=0.25,
        ),
    )

    assert result.equity_curve["turnover"].max() <= 0.2500001


def test_backtest_records_explicit_cost_adjusted_return_breakdown() -> None:
    frame = _prediction_frame(
        dates=["2026-01-01", "2026-01-02"],
        tickers=["AAPL", "AAPL"],
        expected_returns=[0.05, 0.05],
        text_risk_scores=[0.0, 0.0],
    )

    result = run_long_only_backtest(
        frame,
        BacktestConfig(
            top_n=1,
            cost_bps=8.0,
            slippage_bps=3.0,
            max_symbol_weight=1.0,
            portfolio_volatility_limit=1.0,
        ),
    )

    row = result.equity_curve.iloc[0]
    assert row["gross_return"] == pytest.approx(0.0)
    assert row["deterministic_strategy_return"] == pytest.approx(row["gross_return"])
    assert row["turnover"] == pytest.approx(1.0)
    assert row["transaction_cost_return"] == pytest.approx(0.0008)
    assert row["slippage_cost_return"] == pytest.approx(0.0003)
    assert row["total_cost_return"] == pytest.approx(0.0011)
    assert row["portfolio_return"] == pytest.approx(row["gross_return"] - row["total_cost_return"])
    assert row["cost_adjusted_return"] == pytest.approx(row["portfolio_return"])


def test_backtest_charges_costs_only_on_position_weight_changes() -> None:
    frame = _prediction_frame(
        dates=[
            "2026-01-01",
            "2026-01-01",
            "2026-01-02",
            "2026-01-02",
            "2026-01-05",
            "2026-01-05",
        ],
        tickers=["AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT"],
        expected_returns=[0.05, 0.01, 0.05, 0.01, 0.01, 0.05],
        text_risk_scores=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    frame["forward_return_20"] = [0.01, 0.00, 0.02, 0.00, 0.00, 0.03]

    result = run_long_only_backtest(
        frame,
        BacktestConfig(
            top_n=1,
            cost_bps=10.0,
            slippage_bps=5.0,
            max_symbol_weight=1.0,
            portfolio_volatility_limit=1.0,
        ),
    )

    curve = result.equity_curve
    assert curve["turnover"].tolist() == pytest.approx([1.0, 0.0, 2.0])
    assert curve["gross_return"].tolist() == pytest.approx([0.01, 0.02, 0.03])
    assert curve["transaction_cost_return"].tolist() == pytest.approx([0.001, 0.0, 0.002])
    assert curve["slippage_cost_return"].tolist() == pytest.approx([0.0005, 0.0, 0.001])
    assert curve["portfolio_return"].tolist() == pytest.approx([0.0085, 0.02, 0.027])
    assert curve["cost_adjusted_return"].tolist() == pytest.approx(
        curve["portfolio_return"].tolist()
    )


def test_backtest_records_position_level_net_returns_after_costs() -> None:
    frame = _prediction_frame(
        dates=[
            "2026-01-01",
            "2026-01-01",
            "2026-01-02",
            "2026-01-02",
        ],
        tickers=["AAPL", "MSFT", "AAPL", "MSFT"],
        expected_returns=[0.05, 0.04, 0.05, 0.04],
        text_risk_scores=[0.0, 0.0, 0.0, 0.0],
    )
    frame["forward_return_20"] = [0.02, 0.01, 0.03, 0.01]

    result = run_long_only_backtest(
        frame,
        BacktestConfig(
            top_n=2,
            cost_bps=10.0,
            slippage_bps=5.0,
            max_symbol_weight=0.50,
            max_sector_weight=1.0,
            portfolio_volatility_limit=1.0,
        ),
    )

    positions = result.weights
    first_day = positions[positions["signal_date"].eq(pd.Timestamp("2026-01-01"))]
    expected_columns = {
        "previous_weight",
        "realized_return",
        "gross_return_contribution",
        "position_turnover",
        "transaction_cost_return",
        "slippage_cost_return",
        "total_cost_return",
        "net_return_contribution",
        "position_net_return",
    }

    assert expected_columns.issubset(positions.columns)
    assert first_day["gross_return_contribution"].sum() == pytest.approx(
        result.equity_curve.iloc[0]["gross_return"]
    )
    assert first_day["total_cost_return"].sum() == pytest.approx(
        result.equity_curve.iloc[0]["total_cost_return"]
    )
    assert first_day["net_return_contribution"].sum() == pytest.approx(
        result.equity_curve.iloc[0]["cost_adjusted_return"]
    )
    assert first_day["position_net_return"].tolist() == pytest.approx(
        first_day["net_return_contribution"].tolist()
    )


def test_backtest_records_positions_on_future_holding_start_date() -> None:
    frame = _prediction_frame(
        dates=["2026-01-01", "2026-01-02"],
        tickers=["AAPL", "AAPL"],
        expected_returns=[0.05, 0.05],
        text_risk_scores=[0.0, 0.0],
    )

    result = run_long_only_backtest(
        frame,
        BacktestConfig(top_n=1, max_symbol_weight=1.0, portfolio_volatility_limit=1.0),
    )

    assert result.weights.iloc[0]["signal_date"] == pd.Timestamp("2026-01-01")
    assert result.weights.iloc[0]["date"] == pd.Timestamp("2026-01-02")
    assert result.weights.iloc[0]["holding_start_date"] > result.weights.iloc[0]["signal_date"]
    assert result.equity_curve.iloc[0]["holding_start_date"] > result.equity_curve.iloc[0]["date"]


def test_backtest_rejects_non_forward_realized_return_column() -> None:
    frame = _prediction_frame(
        dates=["2026-01-01", "2026-01-02"],
        tickers=["AAPL", "AAPL"],
        expected_returns=[0.05, 0.05],
        text_risk_scores=[0.0, 0.0],
    )

    with pytest.raises(ValueError, match="forward_return_<horizon>"):
        run_long_only_backtest(frame, BacktestConfig(realized_return_column="return_1"))


def test_backtest_rejects_non_future_holding_window_metadata() -> None:
    frame = _prediction_frame(
        dates=["2026-01-01", "2026-01-02"],
        tickers=["AAPL", "AAPL"],
        expected_returns=[0.05, 0.05],
        text_risk_scores=[0.0, 0.0],
    ).assign(
        signal_date=pd.to_datetime(["2026-01-01", "2026-01-02"]),
        holding_start_date=pd.to_datetime(["2026-01-01", "2026-01-03"]),
        return_label_date=pd.to_datetime(["2026-01-02", "2026-01-03"]),
        realized_return_column="forward_return_20",
        return_horizon=20,
    )

    with pytest.raises(ValueError, match="holding_start_date must be after signal_date"):
        run_long_only_backtest(frame)


def test_reported_performance_metrics_use_cost_adjusted_returns_before_gross_returns() -> None:
    net_returns = pd.Series([0.010, -0.005, 0.020])
    gross_returns = pd.Series([0.015, 0.000, 0.025])
    total_costs = gross_returns - net_returns
    equity_curve = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-02", periods=3, freq="B"),
            "equity": (1 + net_returns).cumprod(),
            "benchmark_equity": [1.0, 1.0, 1.0],
            "portfolio_return": gross_returns,
            "gross_return": gross_returns,
            "cost_adjusted_return": net_returns,
            "transaction_cost_return": total_costs * 0.60,
            "slippage_cost_return": total_costs * 0.40,
            "total_cost_return": total_costs,
            "turnover": [1.0, 1.0, 1.0],
            "exposure": [1.0, 1.0, 1.0],
        }
    )

    metrics = calculate_metrics(equity_curve)

    expected_net_sharpe = float(net_returns.mean() / net_returns.std(ddof=0) * (252**0.5))
    gross_sharpe = float(gross_returns.mean() / gross_returns.std(ddof=0) * (252**0.5))
    expected_net_cagr = float(((1 + net_returns).prod()) ** (252 / len(net_returns)) - 1)
    expected_gross_cagr = float(((1 + gross_returns).prod()) ** (252 / len(gross_returns)) - 1)
    assert metrics.cagr == pytest.approx(expected_net_cagr)
    assert metrics.net_cagr == pytest.approx(expected_net_cagr)
    assert metrics.gross_cagr == pytest.approx(expected_gross_cagr)
    assert metrics.return_basis == "cost_adjusted_return"
    assert metrics.sharpe == pytest.approx(expected_net_sharpe)
    assert metrics.sharpe != pytest.approx(gross_sharpe)
    assert metrics.net_cumulative_return == pytest.approx((1 + net_returns).prod() - 1)
    assert metrics.cost_adjusted_cumulative_return == pytest.approx(
        metrics.net_cumulative_return
    )
    assert metrics.gross_cumulative_return == pytest.approx((1 + gross_returns).prod() - 1)
    assert metrics.transaction_cost_return == pytest.approx((total_costs * 0.60).sum())
    assert metrics.slippage_cost_return == pytest.approx((total_costs * 0.40).sum())
    assert metrics.total_cost_return == pytest.approx(total_costs.sum())


def test_performance_metrics_excess_return_uses_cost_adjusted_strategy_and_benchmark_returns() -> None:
    net_returns = pd.Series([0.010, 0.020, -0.005])
    gross_returns = pd.Series([0.020, 0.030, 0.005])
    benchmark_net_returns = pd.Series([0.006, 0.004, 0.003])
    benchmark_gross_returns = pd.Series([0.010, 0.008, 0.007])
    equity_curve = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-02", periods=3, freq="B"),
            "portfolio_return": gross_returns,
            "gross_return": gross_returns,
            "cost_adjusted_return": net_returns,
            "benchmark_return": benchmark_gross_returns,
            "cost_adjusted_benchmark_return": benchmark_net_returns,
            "transaction_cost_return": gross_returns - net_returns,
            "slippage_cost_return": [0.0, 0.0, 0.0],
            "total_cost_return": gross_returns - net_returns,
            "benchmark_transaction_cost_return": benchmark_gross_returns
            - benchmark_net_returns,
            "benchmark_slippage_cost_return": [0.0, 0.0, 0.0],
            "benchmark_total_cost_return": benchmark_gross_returns - benchmark_net_returns,
            "turnover": [1.0, 0.5, 0.25],
            "exposure": [1.0, 1.0, 1.0],
        }
    )

    metrics = calculate_metrics(equity_curve)

    expected_strategy_cagr = float(((1 + net_returns).prod()) ** (252 / len(net_returns)) - 1)
    expected_benchmark_cagr = float(
        ((1 + benchmark_net_returns).prod()) ** (252 / len(benchmark_net_returns)) - 1
    )
    gross_strategy_cagr = float(((1 + gross_returns).prod()) ** (252 / len(gross_returns)) - 1)
    gross_benchmark_cagr = float(
        ((1 + benchmark_gross_returns).prod()) ** (252 / len(benchmark_gross_returns)) - 1
    )

    assert metrics.cagr == pytest.approx(expected_strategy_cagr)
    assert metrics.gross_cagr == pytest.approx(gross_strategy_cagr)
    assert metrics.benchmark_cagr == pytest.approx(expected_benchmark_cagr)
    assert metrics.benchmark_cagr != pytest.approx(gross_benchmark_cagr)
    assert metrics.excess_return == pytest.approx(expected_strategy_cagr - expected_benchmark_cagr)


def test_performance_metrics_accept_period_turnover_column() -> None:
    equity_curve = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-02", periods=3, freq="B"),
            "cost_adjusted_return": [0.01, 0.0, -0.005],
            "benchmark_return": [0.0, 0.0, 0.0],
            "period_turnover": [1.0, 0.5, 0.0],
            "exposure": [1.0, 1.0, 1.0],
        }
    )

    metrics = calculate_metrics(equity_curve)

    assert metrics.turnover == pytest.approx(0.5)


def test_performance_metrics_fall_back_to_equity_curves_when_return_columns_are_absent() -> None:
    equity_curve = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-02", periods=3, freq="B"),
            "equity": [1.0100, 1.0302, 1.019898],
            "benchmark_equity": [1.0050, 1.010025, 1.02012525],
            "turnover": [1.0, 0.0, 0.5],
            "exposure": [1.0, 0.8, 0.9],
        }
    )

    metrics = calculate_metrics(equity_curve)

    expected_returns = pd.Series([0.010, 0.020, -0.010])
    expected_benchmark_returns = pd.Series([0.005, 0.005, 0.010])
    expected_cagr = float(((1 + expected_returns).prod()) ** (252 / 3) - 1)
    expected_benchmark_cagr = float(
        ((1 + expected_benchmark_returns).prod()) ** (252 / 3) - 1
    )
    assert metrics.return_basis == "cost_adjusted_return"
    assert metrics.cagr == pytest.approx(expected_cagr)
    assert metrics.net_cumulative_return == pytest.approx((1 + expected_returns).prod() - 1)
    assert metrics.benchmark_cagr == pytest.approx(expected_benchmark_cagr)
    assert metrics.excess_return == pytest.approx(expected_cagr - expected_benchmark_cagr)
    assert metrics.turnover == pytest.approx(0.5)
    assert metrics.exposure == pytest.approx(0.9)


def test_performance_metrics_reconstruct_gross_returns_from_net_returns_and_costs() -> None:
    net_returns = pd.Series([0.010, -0.005, 0.020])
    total_costs = pd.Series([0.002, 0.001, 0.003])
    equity_curve = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-02", periods=3, freq="B"),
            "cost_adjusted_return": net_returns,
            "benchmark_return": [0.0, 0.0, 0.0],
            "transaction_cost_return": total_costs * 0.75,
            "slippage_cost_return": total_costs * 0.25,
            "total_cost_return": total_costs,
            "turnover": [1.0, 0.5, 0.25],
            "exposure": [1.0, 1.0, 1.0],
        }
    )

    metrics = calculate_metrics(equity_curve)

    expected_gross_returns = net_returns + total_costs
    assert metrics.net_cumulative_return == pytest.approx((1 + net_returns).prod() - 1)
    assert metrics.gross_cumulative_return == pytest.approx(
        (1 + expected_gross_returns).prod() - 1
    )
    assert metrics.transaction_cost_return == pytest.approx((total_costs * 0.75).sum())
    assert metrics.slippage_cost_return == pytest.approx((total_costs * 0.25).sum())
    assert metrics.total_cost_return == pytest.approx(total_costs.sum())


def test_reprice_equity_curve_applies_scenario_costs_to_strategy_and_benchmark() -> None:
    equity_curve = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-02", periods=3, freq="B"),
            "gross_return": [0.010, 0.020, -0.005],
            "cost_adjusted_return": [0.010, 0.020, -0.005],
            "benchmark_gross_return": [0.005, 0.004, 0.003],
            "benchmark_turnover": [1.0, 0.0, 0.0],
            "turnover": [1.0, 0.5, 0.0],
            "exposure": [1.0, 1.0, 1.0],
        }
    )

    repriced = reprice_equity_curve_for_transaction_costs(
        equity_curve,
        cost_bps=10.0,
        slippage_bps=5.0,
        average_daily_turnover_budget=0.25,
        max_daily_turnover=0.75,
    )

    assert repriced["transaction_cost_return"].tolist() == pytest.approx(
        [0.0010, 0.0005, 0.0]
    )
    assert repriced["slippage_cost_return"].tolist() == pytest.approx(
        [0.0005, 0.00025, 0.0]
    )
    assert repriced["cost_adjusted_return"].tolist() == pytest.approx(
        [0.0085, 0.01925, -0.005]
    )
    assert repriced["benchmark_transaction_cost_return"].tolist() == pytest.approx(
        [0.0010, 0.0, 0.0]
    )
    assert repriced["cost_adjusted_benchmark_return"].tolist() == pytest.approx(
        [0.0035, 0.004, 0.003]
    )
    assert repriced["cost_bps"].eq(10.0).all()
    assert repriced["slippage_bps"].eq(5.0).all()
    assert repriced["average_daily_turnover_budget"].eq(0.25).all()
    assert repriced["max_daily_turnover"].eq(0.75).all()


def test_transaction_cost_scenario_analysis_reports_scenario_metrics_and_turnover_breaches() -> None:
    equity_curve = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-02", periods=3, freq="B"),
            "gross_return": [0.010, 0.020, -0.005],
            "benchmark_gross_return": [0.004, 0.004, 0.004],
            "benchmark_turnover": [1.0, 0.0, 0.0],
            "turnover": [1.0, 0.5, 0.0],
            "exposure": [1.0, 1.0, 1.0],
        }
    )
    sensitivity_config = TransactionCostSensitivityConfig(
        baseline_scenario_id="canonical_costs",
        scenarios=(
            TransactionCostSensitivityScenario(
                scenario_id="canonical_costs",
                label="Canonical costs",
                cost_bps=5.0,
                slippage_bps=2.0,
                average_daily_turnover_budget=0.25,
                max_daily_turnover=0.75,
            ),
            TransactionCostSensitivityScenario(
                scenario_id="high_costs",
                label="High costs",
                cost_bps=10.0,
                slippage_bps=5.0,
                average_daily_turnover_budget=0.75,
                max_daily_turnover=1.25,
            ),
        ),
    )

    analysis = analyze_transaction_cost_scenarios(
        equity_curve,
        sensitivity_config=sensitivity_config,
    )
    backtest_like = type("BacktestLike", (), {"equity_curve": equity_curve})()
    analysis_from_result = analyze_transaction_cost_scenarios(
        backtest_like,
        sensitivity_config=sensitivity_config,
    )
    rows = analysis.summary.set_index("scenario_id")

    assert set(analysis.equity_curves) == {"canonical_costs", "high_costs"}
    assert analysis_from_result.summary["scenario_id"].tolist() == analysis.summary[
        "scenario_id"
    ].tolist()
    assert rows.loc["canonical_costs", "is_baseline"]
    assert rows.loc["canonical_costs", "total_cost_bps"] == pytest.approx(7.0)
    assert rows.loc["high_costs", "total_cost_bps"] == pytest.approx(15.0)
    assert rows.loc["high_costs", "total_cost_return"] > rows.loc[
        "canonical_costs", "total_cost_return"
    ]
    assert rows.loc["high_costs", "cost_adjusted_cumulative_return"] < rows.loc[
        "canonical_costs", "cost_adjusted_cumulative_return"
    ]
    assert rows.loc[
        "canonical_costs", "baseline_cost_adjusted_cumulative_return_delta"
    ] == pytest.approx(0.0)
    assert rows.loc["high_costs", "baseline_total_cost_return_delta"] > 0
    assert not bool(rows.loc["canonical_costs", "turnover_budget_pass"])
    assert not bool(rows.loc["canonical_costs", "max_daily_turnover_pass"])
    assert bool(rows.loc["high_costs", "turnover_budget_pass"])
    assert bool(rows.loc["high_costs", "max_daily_turnover_pass"])


def _prediction_frame(
    *,
    dates: list[str],
    tickers: list[str],
    expected_returns: list[float],
    text_risk_scores: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "ticker": tickers,
            "expected_return": expected_returns,
            "predicted_volatility": [0.01] * len(dates),
            "downside_quantile": [0.0] * len(dates),
            "model_confidence": [1.0] * len(dates),
            "text_risk_score": text_risk_scores,
            "sec_risk_flag": [0.0] * len(dates),
            "sec_risk_flag_20d": [0.0] * len(dates),
            "news_negative_ratio": [0.0] * len(dates),
            "liquidity_score": [20.0] * len(dates),
            "forward_return_1": [0.0] * len(dates),
            "forward_return_20": [0.0] * len(dates),
        }
    )
