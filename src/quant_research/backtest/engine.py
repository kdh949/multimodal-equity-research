from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_research.backtest.metrics import PerformanceMetrics, calculate_metrics
from quant_research.signals.engine import DeterministicSignalEngine, SignalEngineConfig


@dataclass(frozen=True)
class BacktestConfig:
    top_n: int = 3
    cost_bps: float = 5.0
    slippage_bps: float = 2.0
    benchmark_ticker: str = "SPY"
    max_symbol_weight: float = 0.35
    portfolio_volatility_limit: float = 0.04
    max_drawdown_stop: float = 0.20


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
    peak_equity = 1.0
    benchmark_equity = 1.0
    risk_stop_active = False
    rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    cost_rate = (config.cost_bps + config.slippage_bps) / 10_000

    for idx, current_date in enumerate(dates):
        day = signals[signals["date"] == current_date].copy()
        effective_date = dates[idx + 1] if idx + 1 < len(dates) else pd.NaT

        current_drawdown = equity / peak_equity - 1
        if current_drawdown <= -abs(config.max_drawdown_stop):
            risk_stop_active = True

        if risk_stop_active:
            target_weights: dict[str, float] = {}
        else:
            target_weights = _select_stateful_targets(day, previous_weights, config)
            target_weights = _apply_portfolio_volatility_limit(day, target_weights, config)

        tickers = set(previous_weights).union(target_weights)
        turnover = sum(abs(target_weights.get(ticker, 0.0) - previous_weights.get(ticker, 0.0)) for ticker in tickers)
        gross_return = 0.0
        for ticker, weight in target_weights.items():
            realized = day.loc[day["ticker"] == ticker, "forward_return_1"]
            if not realized.empty and pd.notna(realized.iloc[0]):
                gross_return += weight * float(realized.iloc[0])
            weight_rows.append(
                {
                    "signal_date": current_date,
                    "effective_date": effective_date,
                    "ticker": ticker,
                    "weight": weight,
                }
            )

        net_return = gross_return - turnover * cost_rate
        benchmark_return = _benchmark_return(day, config.benchmark_ticker)
        equity *= 1 + net_return
        peak_equity = max(peak_equity, equity)
        benchmark_equity *= 1 + benchmark_return
        rows.append(
            {
                "date": current_date,
                "return_date": effective_date,
                "portfolio_return": net_return,
                "gross_return": gross_return,
                "benchmark_return": benchmark_return,
                "equity": equity,
                "benchmark_equity": benchmark_equity,
                "turnover": turnover,
                "exposure": sum(target_weights.values()),
                "portfolio_volatility_estimate": _portfolio_volatility_estimate(day, target_weights),
                "risk_stop_active": risk_stop_active,
            }
        )
        previous_weights = target_weights

    equity_curve = pd.DataFrame(rows)
    weights = pd.DataFrame(weight_rows, columns=["signal_date", "effective_date", "ticker", "weight"])
    metrics = calculate_metrics(equity_curve)
    return BacktestResult(equity_curve=equity_curve, weights=weights, signals=signals, metrics=metrics)


def _benchmark_return(day: pd.DataFrame, benchmark_ticker: str) -> float:
    benchmark = day.loc[day["ticker"] == benchmark_ticker, "forward_return_1"]
    if benchmark.empty or pd.isna(benchmark.iloc[0]):
        return 0.0
    return float(benchmark.iloc[0])


def _select_stateful_targets(
    day: pd.DataFrame,
    previous_weights: dict[str, float],
    config: BacktestConfig,
) -> dict[str, float]:
    sell_tickers = set(day.loc[day["action"] == "SELL", "ticker"])
    candidates = [ticker for ticker in previous_weights if ticker not in sell_tickers]

    buy = day[day["action"] == "BUY"].sort_values("signal_score", ascending=False)
    for ticker in buy["ticker"]:
        if ticker not in candidates:
            candidates.append(ticker)

    if not candidates:
        return {}

    score_by_ticker = day.set_index("ticker")["signal_score"].to_dict()
    selected = sorted(candidates, key=lambda ticker: score_by_ticker.get(ticker, -np.inf), reverse=True)[
        : config.top_n
    ]
    weight = min(1.0 / len(selected), config.max_symbol_weight)
    return {ticker: weight for ticker in selected}


def _apply_portfolio_volatility_limit(
    day: pd.DataFrame,
    target_weights: dict[str, float],
    config: BacktestConfig,
) -> dict[str, float]:
    estimate = _portfolio_volatility_estimate(day, target_weights)
    if estimate <= config.portfolio_volatility_limit or estimate == 0:
        return target_weights
    scale = config.portfolio_volatility_limit / estimate
    return {ticker: weight * scale for ticker, weight in target_weights.items()}


def _portfolio_volatility_estimate(day: pd.DataFrame, weights: dict[str, float]) -> float:
    if not weights:
        return 0.0
    volatility_by_ticker = day.set_index("ticker")["predicted_volatility"].fillna(0.0).to_dict()
    return float(
        np.sqrt(
            sum((weight * float(volatility_by_ticker.get(ticker, 0.0))) ** 2 for ticker, weight in weights.items())
        )
    )
