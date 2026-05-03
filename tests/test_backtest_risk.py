from __future__ import annotations

import pandas as pd

from quant_research.backtest.engine import BacktestConfig, run_long_only_backtest


def test_backtest_caps_symbol_weight() -> None:
    frame = _prediction_frame(
        returns=[0.01, 0.01],
        tickers=["AAPL", "MSFT"],
        dates=["2026-01-01", "2026-01-01"],
    )

    result = run_long_only_backtest(
        frame,
        BacktestConfig(top_n=2, max_symbol_weight=0.25, portfolio_volatility_limit=1.0),
    )

    assert not result.weights.empty
    assert result.weights["weight"].max() <= 0.25
    assert result.equity_curve["exposure"].iloc[0] == 0.5


def test_backtest_scales_portfolio_volatility() -> None:
    frame = _prediction_frame(
        returns=[0.01, 0.01],
        tickers=["AAPL", "MSFT"],
        dates=["2026-01-01", "2026-01-01"],
        volatility=0.10,
    )

    result = run_long_only_backtest(
        frame,
        BacktestConfig(
            top_n=2,
            max_symbol_weight=1.0,
            portfolio_volatility_limit=0.02,
        ),
    )

    assert result.equity_curve["portfolio_volatility_estimate"].iloc[0] <= 0.0200001


def test_backtest_drawdown_stop_forces_cash_after_breach() -> None:
    frame = _prediction_frame(
        returns=[-0.50, 0.10, 0.10],
        tickers=["AAPL", "AAPL", "AAPL"],
        dates=["2026-01-01", "2026-01-02", "2026-01-05"],
    )

    result = run_long_only_backtest(
        frame,
        BacktestConfig(top_n=1, max_symbol_weight=1.0, max_drawdown_stop=0.20),
    )

    assert result.equity_curve["risk_stop_active"].iloc[1]
    assert result.equity_curve["exposure"].iloc[1] == 0.0


def test_backtest_records_next_period_weight_timing() -> None:
    frame = _prediction_frame(
        returns=[0.01, 0.01],
        tickers=["AAPL", "AAPL"],
        dates=["2026-01-01", "2026-01-02"],
    )

    result = run_long_only_backtest(frame, BacktestConfig(top_n=1, max_symbol_weight=1.0))

    assert {"signal_date", "effective_date", "ticker", "weight"}.issubset(result.weights.columns)
    assert result.weights["effective_date"].iloc[0] > result.weights["signal_date"].iloc[0]


def _prediction_frame(
    returns: list[float],
    tickers: list[str],
    dates: list[str],
    volatility: float = 0.01,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "ticker": tickers,
            "expected_return": [0.05] * len(dates),
            "predicted_volatility": [volatility] * len(dates),
            "downside_quantile": [0.0] * len(dates),
            "model_confidence": [1.0] * len(dates),
            "text_risk_score": [0.0] * len(dates),
            "sec_risk_flag": [0.0] * len(dates),
            "sec_risk_flag_20d": [0.0] * len(dates),
            "news_negative_ratio": [0.0] * len(dates),
            "liquidity_score": [20.0] * len(dates),
            "forward_return_1": returns,
        }
    )
