from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceMetrics:
    cagr: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    hit_rate: float
    turnover: float
    exposure: float
    benchmark_cagr: float
    excess_return: float


def calculate_metrics(equity_curve: pd.DataFrame) -> PerformanceMetrics:
    if equity_curve.empty:
        return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)

    returns = equity_curve["portfolio_return"].fillna(0.0)
    years = max(len(equity_curve) / 252, 1 / 252)
    ending_equity = float(equity_curve["equity"].iloc[-1])
    benchmark_equity = float(equity_curve["benchmark_equity"].iloc[-1])
    cagr = ending_equity ** (1 / years) - 1
    benchmark_cagr = benchmark_equity ** (1 / years) - 1
    vol = returns.std(ddof=0) * np.sqrt(252)
    sharpe = returns.mean() / returns.std(ddof=0) * np.sqrt(252) if returns.std(ddof=0) > 0 else 0.0
    drawdown = equity_curve["equity"] / equity_curve["equity"].cummax() - 1
    hit_rate = (returns > 0).mean()
    turnover = equity_curve["turnover"].mean() if "turnover" in equity_curve else 0.0
    exposure = equity_curve["exposure"].mean() if "exposure" in equity_curve else 0.0
    excess_return = cagr - benchmark_cagr
    return PerformanceMetrics(
        cagr=float(cagr),
        annualized_volatility=float(vol),
        sharpe=float(sharpe),
        max_drawdown=float(drawdown.min()),
        hit_rate=float(hit_rate),
        turnover=float(turnover),
        exposure=float(exposure),
        benchmark_cagr=float(benchmark_cagr),
        excess_return=float(excess_return),
    )
