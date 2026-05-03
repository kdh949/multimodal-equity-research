from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_research.backtest.metrics import PerformanceMetrics, calculate_metrics
from quant_research.signals.engine import DeterministicSignalEngine, SignalEngineConfig


@dataclass(frozen=True)
class BacktestConfig:
    top_n: int = 3
    cost_bps: float = 5.0
    slippage_bps: float = 2.0
    benchmark_ticker: str = "SPY"


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.DataFrame
    weights: pd.DataFrame
    signals: pd.DataFrame
    metrics: PerformanceMetrics


def run_long_only_backtest(frame: pd.DataFrame, config: BacktestConfig | None = None) -> BacktestResult:
    config = config or BacktestConfig()
    signal_config = SignalEngineConfig(cost_bps=config.cost_bps, slippage_bps=config.slippage_bps)
    signal_engine = DeterministicSignalEngine(signal_config)
    signals = signal_engine.generate(frame)

    dates = sorted(signals["date"].dropna().unique())
    previous_weights: dict[str, float] = {}
    equity = 1.0
    benchmark_equity = 1.0
    rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    cost_rate = (config.cost_bps + config.slippage_bps) / 10_000

    for current_date in dates:
        day = signals[signals["date"] == current_date].copy()
        buy = day[day["action"] == "BUY"].sort_values("signal_score", ascending=False).head(config.top_n)
        if buy.empty:
            target_weights: dict[str, float] = {}
        else:
            weight = 1.0 / len(buy)
            target_weights = {ticker: weight for ticker in buy["ticker"]}

        tickers = set(previous_weights).union(target_weights)
        turnover = sum(abs(target_weights.get(ticker, 0.0) - previous_weights.get(ticker, 0.0)) for ticker in tickers)
        gross_return = 0.0
        for ticker, weight in target_weights.items():
            realized = day.loc[day["ticker"] == ticker, "forward_return_1"]
            if not realized.empty and pd.notna(realized.iloc[0]):
                gross_return += weight * float(realized.iloc[0])
            weight_rows.append({"date": current_date, "ticker": ticker, "weight": weight})

        net_return = gross_return - turnover * cost_rate
        benchmark_return = _benchmark_return(day, config.benchmark_ticker)
        equity *= 1 + net_return
        benchmark_equity *= 1 + benchmark_return
        rows.append(
            {
                "date": current_date,
                "portfolio_return": net_return,
                "gross_return": gross_return,
                "benchmark_return": benchmark_return,
                "equity": equity,
                "benchmark_equity": benchmark_equity,
                "turnover": turnover,
                "exposure": sum(target_weights.values()),
            }
        )
        previous_weights = target_weights

    equity_curve = pd.DataFrame(rows)
    weights = pd.DataFrame(weight_rows, columns=["date", "ticker", "weight"])
    metrics = calculate_metrics(equity_curve)
    return BacktestResult(equity_curve=equity_curve, weights=weights, signals=signals, metrics=metrics)


def _benchmark_return(day: pd.DataFrame, benchmark_ticker: str) -> float:
    benchmark = day.loc[day["ticker"] == benchmark_ticker, "forward_return_1"]
    if benchmark.empty or pd.isna(benchmark.iloc[0]):
        return 0.0
    return float(benchmark.iloc[0])
