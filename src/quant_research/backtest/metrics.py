from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quant_research.backtest.covariance import estimate_portfolio_covariance_matrix
from quant_research.performance import calculate_return_series_metrics


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
    return_basis: str = "cost_adjusted_return"
    gross_cagr: float = 0.0
    gross_cumulative_return: float = 0.0
    net_cagr: float = 0.0
    net_cumulative_return: float = 0.0
    cost_adjusted_cumulative_return: float = 0.0
    benchmark_cost_adjusted_cagr: float = 0.0
    benchmark_cost_adjusted_cumulative_return: float = 0.0
    transaction_cost_return: float = 0.0
    slippage_cost_return: float = 0.0
    total_cost_return: float = 0.0
    average_portfolio_volatility_estimate: float = 0.0
    max_portfolio_volatility_estimate: float = 0.0
    max_position_weight: float = 0.0
    max_sector_exposure: float = 0.0
    max_position_risk_contribution: float = 0.0
    position_sizing_validation_pass_rate: float = 0.0
    position_sizing_validation_status: str = "not_evaluated"
    position_sizing_validation_rule: str = ""

    def __post_init__(self) -> None:
        if self.net_cagr == 0.0 and self.cagr != 0.0:
            object.__setattr__(self, "net_cagr", self.cagr)
        if self.benchmark_cost_adjusted_cagr == 0.0 and self.benchmark_cagr != 0.0:
            object.__setattr__(
                self,
                "benchmark_cost_adjusted_cagr",
                self.benchmark_cagr,
            )


@dataclass(frozen=True)
class PortfolioRiskMetrics:
    portfolio_volatility: float
    gross_exposure: float
    net_exposure: float
    cash_weight: float
    max_symbol_weight: float
    max_sector_weight: float
    herfindahl_index: float
    effective_holdings: float
    top_5_weight: float
    risk_contributions: pd.DataFrame
    sector_exposures: pd.DataFrame


@dataclass(frozen=True)
class TransactionCostScenarioAnalysis:
    summary: pd.DataFrame
    equity_curves: dict[str, pd.DataFrame]


def calculate_portfolio_turnover(
    previous_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
) -> float:
    tickers = set(previous_weights).union(target_weights)
    return float(
        sum(
            abs(
                _finite_weight(target_weights.get(ticker, 0.0))
                - _finite_weight(previous_weights.get(ticker, 0.0))
            )
            for ticker in tickers
        )
    )


def calculate_portfolio_risk_metrics(
    positions: pd.DataFrame | Mapping[str, float],
    *,
    covariance: pd.DataFrame | None = None,
    ticker_column: str = "ticker",
    weight_column: str = "weight",
    sector_column: str = "sector",
    volatility_column: str | None = None,
    long_only: bool = True,
) -> PortfolioRiskMetrics:
    """Calculate position-level risk contribution, volatility, and concentration metrics."""

    frame = _normalize_risk_positions(
        positions,
        ticker_column=ticker_column,
        weight_column=weight_column,
        sector_column=sector_column,
        volatility_column=volatility_column,
        long_only=long_only,
    )
    if frame.empty:
        return PortfolioRiskMetrics(
            portfolio_volatility=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
            cash_weight=1.0,
            max_symbol_weight=0.0,
            max_sector_weight=0.0,
            herfindahl_index=0.0,
            effective_holdings=0.0,
            top_5_weight=0.0,
            risk_contributions=_empty_risk_contributions(),
            sector_exposures=_empty_sector_exposures(),
        )

    tickers = frame["ticker"].tolist()
    weights = frame["weight"].to_numpy(dtype=float)
    covariance_matrix = _risk_covariance_matrix(
        frame,
        tickers,
        covariance=covariance,
    )
    portfolio_variance = float(weights @ covariance_matrix @ weights.T)
    if not np.isfinite(portfolio_variance) or portfolio_variance < -1e-12:
        raise ValueError("portfolio variance must be finite and non-negative")
    portfolio_volatility = float(np.sqrt(max(portfolio_variance, 0.0)))
    marginal = covariance_matrix @ weights
    if portfolio_volatility > 0:
        risk_contribution = weights * marginal / portfolio_volatility
        risk_contribution_pct = risk_contribution / portfolio_volatility
    else:
        risk_contribution = np.zeros(len(weights), dtype=float)
        risk_contribution_pct = np.zeros(len(weights), dtype=float)

    gross_exposure = float(np.abs(weights).sum())
    net_exposure = float(weights.sum())
    normalized_abs_weight = (
        np.abs(weights) / gross_exposure if gross_exposure > 0 else np.zeros(len(weights))
    )
    herfindahl = float(np.square(normalized_abs_weight).sum())
    sector_exposures = _sector_exposure_frame(frame)
    max_sector_weight = (
        float(sector_exposures["weight"].abs().max()) if not sector_exposures.empty else 0.0
    )
    return PortfolioRiskMetrics(
        portfolio_volatility=portfolio_volatility,
        gross_exposure=gross_exposure,
        net_exposure=net_exposure,
        cash_weight=float(max(0.0, 1.0 - gross_exposure)),
        max_symbol_weight=float(np.abs(weights).max()) if len(weights) else 0.0,
        max_sector_weight=max_sector_weight,
        herfindahl_index=herfindahl,
        effective_holdings=float(1.0 / herfindahl) if herfindahl > 0 else 0.0,
        top_5_weight=float(np.sort(np.abs(weights))[-5:].sum()) if len(weights) else 0.0,
        risk_contributions=pd.DataFrame(
            {
                "ticker": tickers,
                "weight": weights,
                "risk_contribution": risk_contribution,
                "risk_contribution_pct": risk_contribution_pct,
                "marginal_risk": marginal,
            }
        ),
        sector_exposures=sector_exposures,
    )


def calculate_covariance_aware_portfolio_risk_metrics(
    positions: pd.DataFrame,
    return_history: pd.DataFrame,
    *,
    ticker_column: str = "ticker",
    weight_column: str = "weight",
    sector_column: str = "sector",
    date_column: str = "date",
    return_ticker_column: str = "ticker",
    return_column: str = "return_1",
    min_periods: int = 2,
    annualization_factor: float | None = None,
    long_only: bool = True,
) -> PortfolioRiskMetrics:
    """Calculate portfolio risk from an aligned covariance matrix of held returns."""

    covariance = estimate_portfolio_covariance_matrix(
        positions,
        return_history,
        ticker_column=ticker_column,
        weight_column=weight_column,
        date_column=date_column,
        return_ticker_column=return_ticker_column,
        return_column=return_column,
        min_periods=min_periods,
        annualization_factor=annualization_factor,
    )
    return calculate_portfolio_risk_metrics(
        positions,
        covariance=covariance if not covariance.empty else None,
        ticker_column=ticker_column,
        weight_column=weight_column,
        sector_column=sector_column,
        long_only=long_only,
    )


def calculate_cost_adjusted_returns(
    gross_returns: object,
    turnover: object,
    *,
    cost_bps: float,
    slippage_bps: float,
) -> pd.DataFrame:
    """Apply transaction-cost and slippage drag to deterministic strategy returns."""

    gross = _coerce_numeric_series(gross_returns, name="gross_return")
    traded = _coerce_numeric_series(turnover, name="turnover")
    gross, traded = _align_cost_adjustment_inputs(gross, traded)
    transaction_cost_rate = _basis_points_to_rate(cost_bps, "cost_bps")
    slippage_rate = _basis_points_to_rate(slippage_bps, "slippage_bps")

    transaction_cost = traded * transaction_cost_rate
    slippage_cost = traded * slippage_rate
    total_cost = transaction_cost + slippage_cost
    net_return = gross - total_cost
    return pd.DataFrame(
        {
            "gross_return": gross.to_numpy(dtype=float),
            "turnover": traded.to_numpy(dtype=float),
            "transaction_cost_return": transaction_cost.to_numpy(dtype=float),
            "slippage_cost_return": slippage_cost.to_numpy(dtype=float),
            "total_cost_return": total_cost.to_numpy(dtype=float),
            "turnover_cost_return": total_cost.to_numpy(dtype=float),
            "cost_adjusted_return": net_return.to_numpy(dtype=float),
            "net_return": net_return.to_numpy(dtype=float),
        },
        index=gross.index,
    )


def calculate_cost_adjusted_strategy_returns(
    gross_returns: object,
    turnover: object,
    *,
    cost_bps: float,
    slippage_bps: float,
) -> pd.DataFrame:
    return calculate_cost_adjusted_returns(
        gross_returns,
        turnover,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
    )


def analyze_transaction_cost_scenarios(
    equity_curve: object,
    *,
    sensitivity_config: object | None = None,
) -> TransactionCostScenarioAnalysis:
    """Reprice one backtest equity curve under configured cost/slippage scenarios."""

    if sensitivity_config is None:
        from quant_research.validation.config import default_transaction_cost_sensitivity_config

        sensitivity_config = default_transaction_cost_sensitivity_config()
    equity_curve = _coerce_equity_curve_frame(equity_curve)
    if equity_curve.empty:
        return TransactionCostScenarioAnalysis(
            summary=_empty_transaction_cost_scenario_summary(),
            equity_curves={},
        )

    scenarios = tuple(sensitivity_config.scenarios)
    baseline_scenario_id = str(sensitivity_config.baseline_scenario_id)
    scenario_curves: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    baseline_metrics: PerformanceMetrics | None = None

    for scenario in scenarios:
        scenario_id = str(scenario.scenario_id)
        repriced_curve = reprice_equity_curve_for_transaction_costs(
            equity_curve,
            cost_bps=float(scenario.cost_bps),
            slippage_bps=float(scenario.slippage_bps),
            average_daily_turnover_budget=float(
                scenario.average_daily_turnover_budget
            ),
            max_daily_turnover=scenario.max_daily_turnover,
        )
        metrics = calculate_metrics(repriced_curve)
        if scenario_id == baseline_scenario_id:
            baseline_metrics = metrics
        scenario_curves[scenario_id] = repriced_curve
        rows.append(
            _transaction_cost_scenario_row(
                scenario,
                metrics,
                repriced_curve,
                baseline_scenario_id=baseline_scenario_id,
            )
        )

    summary = pd.DataFrame(rows)
    if baseline_metrics is not None and not summary.empty:
        summary["baseline_cost_adjusted_cumulative_return_delta"] = (
            summary["cost_adjusted_cumulative_return"]
            - baseline_metrics.cost_adjusted_cumulative_return
        )
        summary["baseline_excess_return_delta"] = (
            summary["excess_return"] - baseline_metrics.excess_return
        )
        summary["baseline_total_cost_return_delta"] = (
            summary["total_cost_return"] - baseline_metrics.total_cost_return
        )
    return TransactionCostScenarioAnalysis(summary=summary, equity_curves=scenario_curves)


def reprice_equity_curve_for_transaction_costs(
    equity_curve: object,
    *,
    cost_bps: float,
    slippage_bps: float,
    average_daily_turnover_budget: float | None = None,
    max_daily_turnover: float | None = None,
) -> pd.DataFrame:
    """Apply cost/slippage assumptions to gross strategy and benchmark returns."""

    equity_curve = _coerce_equity_curve_frame(equity_curve)
    if equity_curve.empty:
        return equity_curve.copy()

    output = equity_curve.copy()
    gross_returns = _gross_return_series(output, _portfolio_return_series(output))
    turnover = _turnover_series(output, default=0.0)
    adjusted = calculate_cost_adjusted_returns(
        gross_returns,
        turnover,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
    )
    output["gross_return"] = adjusted["gross_return"].to_numpy(dtype=float)
    output["deterministic_strategy_return"] = adjusted["gross_return"].to_numpy(dtype=float)
    output["portfolio_return"] = adjusted["cost_adjusted_return"].to_numpy(dtype=float)
    output["cost_adjusted_return"] = adjusted["cost_adjusted_return"].to_numpy(dtype=float)
    output["net_return"] = adjusted["net_return"].to_numpy(dtype=float)
    output["transaction_cost_return"] = adjusted["transaction_cost_return"].to_numpy(dtype=float)
    output["slippage_cost_return"] = adjusted["slippage_cost_return"].to_numpy(dtype=float)
    output["total_cost_return"] = adjusted["total_cost_return"].to_numpy(dtype=float)
    output["turnover_cost_return"] = adjusted["turnover_cost_return"].to_numpy(dtype=float)
    output["turnover"] = adjusted["turnover"].to_numpy(dtype=float)
    output["period_turnover"] = adjusted["turnover"].to_numpy(dtype=float)
    output["cost_bps"] = float(cost_bps)
    output["slippage_bps"] = float(slippage_bps)
    if average_daily_turnover_budget is not None:
        output["average_daily_turnover_budget"] = float(average_daily_turnover_budget)
    if max_daily_turnover is not None:
        output["max_daily_turnover"] = float(max_daily_turnover)
    output["equity"] = (1.0 + output["cost_adjusted_return"]).cumprod()

    benchmark_gross = _benchmark_gross_return_series(output)
    benchmark_turnover = _benchmark_turnover_series(output)
    benchmark_adjusted = calculate_cost_adjusted_returns(
        benchmark_gross,
        benchmark_turnover,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
    )
    output["benchmark_return"] = benchmark_adjusted["gross_return"].to_numpy(dtype=float)
    output["benchmark_gross_return"] = benchmark_adjusted["gross_return"].to_numpy(dtype=float)
    output["cost_adjusted_benchmark_return"] = benchmark_adjusted[
        "cost_adjusted_return"
    ].to_numpy(dtype=float)
    output["benchmark_transaction_cost_return"] = benchmark_adjusted[
        "transaction_cost_return"
    ].to_numpy(dtype=float)
    output["benchmark_slippage_cost_return"] = benchmark_adjusted[
        "slippage_cost_return"
    ].to_numpy(dtype=float)
    output["benchmark_total_cost_return"] = benchmark_adjusted[
        "total_cost_return"
    ].to_numpy(dtype=float)
    output["benchmark_turnover"] = benchmark_adjusted["turnover"].to_numpy(dtype=float)
    output["benchmark_equity"] = (1.0 + output["benchmark_return"]).cumprod()
    output["cost_adjusted_benchmark_equity"] = (
        1.0 + output["cost_adjusted_benchmark_return"]
    ).cumprod()
    return output


def calculate_daily_position_turnover(
    positions: pd.DataFrame,
    *,
    date_column: str | None = None,
    ticker_column: str = "ticker",
    weight_column: str = "weight",
    date_index: Iterable[object] | None = None,
) -> pd.DataFrame:
    resolved_date_column = _resolve_position_date_column(positions, date_column)
    if resolved_date_column is None:
        if date_index is None:
            return pd.DataFrame(columns=["date", "turnover"])
        dates = _coerce_sorted_dates(date_index)
        return pd.DataFrame({"date": dates, "turnover": [0.0] * len(dates)})
    _require_columns(positions, [resolved_date_column, ticker_column, weight_column])

    frame = positions[[resolved_date_column, ticker_column, weight_column]].copy()
    frame["date"] = pd.to_datetime(frame[resolved_date_column], errors="coerce")
    frame = frame.dropna(subset=["date", ticker_column])
    frame[ticker_column] = frame[ticker_column].astype(str)
    frame[weight_column] = pd.to_numeric(frame[weight_column], errors="coerce").fillna(0.0)
    frame = frame[frame[ticker_column].str.len() > 0]

    dates = _position_turnover_dates(frame["date"], date_index)
    if len(dates) == 0:
        return pd.DataFrame(columns=["date", "turnover"])

    if frame.empty:
        return pd.DataFrame({"date": dates, "turnover": [0.0] * len(dates)})

    weights = (
        frame.pivot_table(
            index="date",
            columns=ticker_column,
            values=weight_column,
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(dates, fill_value=0.0)
        .sort_index()
    )
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    return pd.DataFrame({"date": weights.index, "turnover": turnover.to_numpy(dtype=float)})


def calculate_average_daily_turnover(
    positions: pd.DataFrame,
    *,
    date_column: str | None = None,
    ticker_column: str = "ticker",
    weight_column: str = "weight",
    date_index: Iterable[object] | None = None,
) -> float:
    daily = calculate_daily_position_turnover(
        positions,
        date_column=date_column,
        ticker_column=ticker_column,
        weight_column=weight_column,
        date_index=date_index,
    )
    if daily.empty:
        return 0.0
    return float(daily["turnover"].mean())


def calculate_metrics(equity_curve: pd.DataFrame) -> PerformanceMetrics:
    if equity_curve.empty:
        return PerformanceMetrics(
            cagr=0.0,
            annualized_volatility=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            hit_rate=0.0,
            turnover=0.0,
            exposure=0.0,
            benchmark_cagr=0.0,
            excess_return=0.0,
        )

    returns = _portfolio_return_series(equity_curve)
    metrics = calculate_return_series_metrics(returns, _date_series(equity_curve))
    gross_returns = _gross_return_series(equity_curve, returns)
    gross_metrics = calculate_return_series_metrics(
        gross_returns,
        _date_series(equity_curve),
    )
    benchmark_metrics = calculate_return_series_metrics(
        _benchmark_return_series(equity_curve),
        _date_series(equity_curve),
    )
    vol = returns.std(ddof=0) * np.sqrt(252)
    hit_rate = (returns > 0).mean()
    turnover_column = "turnover" if "turnover" in equity_curve else "period_turnover"
    turnover = equity_curve[turnover_column].mean() if turnover_column in equity_curve else 0.0
    exposure = equity_curve["exposure"].mean() if "exposure" in equity_curve else 0.0
    sizing_status, sizing_pass_rate = _position_sizing_validation_summary(equity_curve)
    excess_return = metrics.cagr - benchmark_metrics.cagr
    return PerformanceMetrics(
        return_basis=_return_basis(equity_curve),
        cagr=float(metrics.cagr),
        annualized_volatility=float(vol),
        sharpe=float(metrics.sharpe),
        max_drawdown=float(metrics.max_drawdown),
        hit_rate=float(hit_rate),
        turnover=float(turnover),
        exposure=float(exposure),
        benchmark_cagr=float(benchmark_metrics.cagr),
        excess_return=float(excess_return),
        gross_cagr=float(gross_metrics.cagr),
        gross_cumulative_return=float(gross_metrics.cumulative_return),
        net_cagr=float(metrics.cagr),
        net_cumulative_return=float(metrics.cumulative_return),
        cost_adjusted_cumulative_return=float(metrics.cumulative_return),
        benchmark_cost_adjusted_cagr=float(benchmark_metrics.cagr),
        benchmark_cost_adjusted_cumulative_return=float(benchmark_metrics.cumulative_return),
        transaction_cost_return=_sum_return_cost(equity_curve, "transaction_cost_return"),
        slippage_cost_return=_sum_return_cost(equity_curve, "slippage_cost_return"),
        total_cost_return=_sum_return_cost(
            equity_curve,
            "total_cost_return",
            fallback_column="turnover_cost_return",
        ),
        average_portfolio_volatility_estimate=_finite_column_mean(
            equity_curve,
            "portfolio_volatility_estimate",
        ),
        max_portfolio_volatility_estimate=_finite_column_max(
            equity_curve,
            "portfolio_volatility_estimate",
        ),
        max_position_weight=_finite_column_max(equity_curve, "max_position_weight"),
        max_sector_exposure=_finite_column_max(equity_curve, "max_sector_exposure"),
        max_position_risk_contribution=_finite_column_max(
            equity_curve,
            "max_position_risk_contribution",
        ),
        position_sizing_validation_pass_rate=sizing_pass_rate,
        position_sizing_validation_status=sizing_status,
        position_sizing_validation_rule=_latest_string_value(
            equity_curve,
            "position_sizing_validation_rule",
        ),
    )


def _finite_column_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce")
    values = values[np.isfinite(values)]
    if values.empty:
        return 0.0
    return float(values.mean())


def _finite_column_max(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce")
    values = values[np.isfinite(values)]
    if values.empty:
        return 0.0
    return float(values.max())


def _latest_string_value(frame: pd.DataFrame, column: str) -> str:
    if column not in frame:
        return ""
    values = frame[column].dropna().astype(str)
    values = values[values.str.len() > 0]
    if values.empty:
        return ""
    return str(values.iloc[-1])


def _position_sizing_validation_summary(equity_curve: pd.DataFrame) -> tuple[str, float]:
    if "position_sizing_validation_status" not in equity_curve:
        return "not_evaluated", 0.0
    statuses = equity_curve["position_sizing_validation_status"].dropna().astype(str)
    statuses = statuses[statuses.str.len() > 0]
    if statuses.empty:
        return "not_evaluated", 0.0
    pass_rate = float(statuses.str.lower().eq("pass").mean())
    if pass_rate >= 1.0:
        return "pass", pass_rate
    if pass_rate > 0.0:
        return "partial", pass_rate
    return "fail", pass_rate


def _finite_weight(value: object) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return 0.0
    return float(numeric)


def _normalize_risk_positions(
    positions: pd.DataFrame | Mapping[str, float],
    *,
    ticker_column: str,
    weight_column: str,
    sector_column: str,
    volatility_column: str | None,
    long_only: bool,
) -> pd.DataFrame:
    if isinstance(positions, pd.DataFrame):
        _require_columns(positions, [ticker_column, weight_column])
        selected = [ticker_column, weight_column]
        if sector_column in positions:
            selected.append(sector_column)
        resolved_volatility = _resolve_volatility_column(positions, volatility_column)
        if resolved_volatility is not None and resolved_volatility not in selected:
            selected.append(resolved_volatility)
        frame = positions[selected].copy()
        frame = frame.rename(columns={ticker_column: "ticker", weight_column: "weight"})
        if sector_column in frame:
            frame = frame.rename(columns={sector_column: "sector"})
        else:
            frame["sector"] = "unknown"
        if resolved_volatility is not None:
            frame = frame.rename(columns={resolved_volatility: "volatility"})
        else:
            frame["volatility"] = 0.0
    else:
        frame = pd.DataFrame(
            [
                {
                    "ticker": ticker,
                    "weight": weight,
                    "sector": "unknown",
                    "volatility": 0.0,
                }
                for ticker, weight in positions.items()
            ],
            columns=["ticker", "weight", "sector", "volatility"],
        )

    frame["ticker"] = frame["ticker"].fillna("").astype(str).str.strip()
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
    frame["sector"] = (
        frame["sector"].fillna("unknown").astype(str).str.strip().replace("", "unknown")
    )
    frame["volatility"] = pd.to_numeric(frame["volatility"], errors="coerce").fillna(0.0)
    invalid = frame["ticker"].eq("") | frame["weight"].isna()
    if invalid.any():
        raise ValueError("positions frame contains invalid ticker or weight values")
    if not np.isfinite(frame["weight"].to_numpy(dtype=float)).all():
        raise ValueError("positions frame contains non-finite weights")
    if long_only and (frame["weight"] < -1e-12).any():
        raise ValueError("long-only portfolio risk metrics require non-negative weights")
    if (frame["volatility"] < 0).any():
        raise ValueError("position volatility values must be non-negative")

    grouped = (
        frame.groupby("ticker", as_index=False)
        .agg(
            weight=("weight", "sum"),
            sector=("sector", "first"),
            volatility=("volatility", "mean"),
        )
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    return grouped[grouped["weight"].abs() > 1e-12].reset_index(drop=True)


def _risk_covariance_matrix(
    frame: pd.DataFrame,
    tickers: list[str],
    *,
    covariance: pd.DataFrame | None,
) -> np.ndarray:
    if covariance is None:
        volatility = frame["volatility"].to_numpy(dtype=float)
        return np.diag(np.square(volatility))

    matrix = _validate_risk_covariance(covariance)
    missing = [ticker for ticker in tickers if ticker not in matrix.index]
    if missing:
        raise ValueError(f"covariance matrix missing held tickers: {missing}")
    return matrix.loc[tickers, tickers].to_numpy(dtype=float)


def _validate_risk_covariance(covariance: pd.DataFrame) -> pd.DataFrame:
    if covariance.empty:
        raise ValueError("covariance matrix must not be empty for non-empty positions")
    if list(covariance.index) != list(covariance.columns):
        raise ValueError("covariance matrix index and columns must match")
    matrix = covariance.astype(float)
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("covariance matrix contains non-finite values")
    diagonal = np.diag(values)
    if (diagonal < -1e-12).any():
        raise ValueError("covariance matrix diagonal must be non-negative")
    symmetric = (values + values.T) / 2.0
    return pd.DataFrame(
        symmetric,
        index=matrix.index.astype(str),
        columns=matrix.columns.astype(str),
    )


def _resolve_volatility_column(
    frame: pd.DataFrame,
    volatility_column: str | None,
) -> str | None:
    if volatility_column is not None:
        _require_columns(frame, [volatility_column])
        return volatility_column
    for candidate in ("predicted_volatility", "volatility", "realized_volatility"):
        if candidate in frame:
            return candidate
    return None


def _sector_exposure_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_sector_exposures()
    return (
        frame.groupby("sector", as_index=False)["weight"]
        .sum()
        .assign(abs_weight=lambda data: data["weight"].abs())
        .sort_values(["abs_weight", "sector"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _empty_risk_contributions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ticker",
            "weight",
            "risk_contribution",
            "risk_contribution_pct",
            "marginal_risk",
        ]
    )


def _empty_sector_exposures() -> pd.DataFrame:
    return pd.DataFrame(columns=["sector", "weight", "abs_weight"])


def _coerce_numeric_series(
    values: object,
    *,
    name: str,
) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.copy()
    elif np.isscalar(values):
        series = pd.Series([values])
    else:
        series = pd.Series(values)
    series = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if (series < 0).any() and name == "turnover":
        raise ValueError("turnover must be non-negative")
    series.name = name
    return series.astype(float)


def _align_cost_adjustment_inputs(
    gross: pd.Series,
    turnover: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    if len(gross) == len(turnover):
        index = _preferred_index(gross.index, turnover.index)
        aligned_gross = gross.copy()
        aligned_turnover = turnover.copy()
        aligned_gross.index = index
        aligned_turnover.index = index
        return aligned_gross, aligned_turnover
    if len(gross) == 1:
        return pd.Series([gross.iloc[0]] * len(turnover), index=turnover.index, name=gross.name), turnover
    if len(turnover) == 1:
        return gross, pd.Series([turnover.iloc[0]] * len(gross), index=gross.index, name=turnover.name)
    raise ValueError("gross_returns and turnover must have the same length or one scalar input")


def _preferred_index(left: pd.Index, right: pd.Index) -> pd.Index:
    if isinstance(left, pd.RangeIndex) and left.start == 0 and left.step == 1:
        return right
    return left


def _basis_points_to_rate(value: float, name: str) -> float:
    numeric = _finite_weight(value)
    if numeric < 0:
        raise ValueError(f"{name} must be non-negative")
    return numeric / 10_000


def _portfolio_return_series(equity_curve: pd.DataFrame) -> pd.Series:
    for column in ("cost_adjusted_return", "net_return", "portfolio_return"):
        if column in equity_curve:
            return pd.to_numeric(equity_curve[column], errors="coerce").fillna(0.0)
    if "equity" in equity_curve:
        return _returns_from_equity(equity_curve["equity"])
    return pd.Series(0.0, index=equity_curve.index)


def _gross_return_series(equity_curve: pd.DataFrame, net_returns: pd.Series) -> pd.Series:
    for column in ("gross_return", "deterministic_strategy_return"):
        if column in equity_curve:
            return pd.to_numeric(equity_curve[column], errors="coerce").fillna(0.0)
    total_cost = _return_cost_series(equity_curve, "total_cost_return", "turnover_cost_return")
    if total_cost.empty:
        total_cost = _return_cost_series(equity_curve, "transaction_cost_return").add(
            _return_cost_series(equity_curve, "slippage_cost_return"),
            fill_value=0.0,
        )
    if total_cost.empty:
        return net_returns.copy()
    return net_returns.add(total_cost.reindex(net_returns.index, fill_value=0.0), fill_value=0.0)


def _benchmark_return_series(equity_curve: pd.DataFrame) -> pd.Series:
    for column in ("cost_adjusted_benchmark_return", "benchmark_return"):
        if column in equity_curve:
            return pd.to_numeric(equity_curve[column], errors="coerce").fillna(0.0)
    for column in ("cost_adjusted_benchmark_equity", "benchmark_equity"):
        if column in equity_curve:
            return _returns_from_equity(equity_curve[column])
    return pd.Series(0.0, index=equity_curve.index)


def _benchmark_gross_return_series(equity_curve: pd.DataFrame) -> pd.Series:
    for column in ("benchmark_gross_return", "benchmark_return"):
        if column in equity_curve:
            return pd.to_numeric(equity_curve[column], errors="coerce").fillna(0.0)
    for column in ("benchmark_equity", "cost_adjusted_benchmark_equity"):
        if column in equity_curve:
            return _returns_from_equity(equity_curve[column])
    return pd.Series(0.0, index=equity_curve.index)


def _turnover_series(equity_curve: pd.DataFrame, *, default: float) -> pd.Series:
    for column in ("turnover", "period_turnover"):
        if column in equity_curve:
            return pd.to_numeric(equity_curve[column], errors="coerce").fillna(default)
    return pd.Series(default, index=equity_curve.index, dtype=float)


def _benchmark_turnover_series(equity_curve: pd.DataFrame) -> pd.Series:
    if "benchmark_turnover" in equity_curve:
        return pd.to_numeric(equity_curve["benchmark_turnover"], errors="coerce").fillna(0.0)
    turnover = pd.Series(0.0, index=equity_curve.index, dtype=float)
    if not turnover.empty:
        turnover.iloc[0] = 1.0
    return turnover


def _date_series(equity_curve: pd.DataFrame) -> pd.Series | None:
    if "date" not in equity_curve:
        return None
    return equity_curve["date"]


def _return_basis(equity_curve: pd.DataFrame) -> str:
    for column in ("cost_adjusted_return", "net_return", "portfolio_return", "equity"):
        if column in equity_curve:
            return "cost_adjusted_return" if column == "equity" else column
    return "cost_adjusted_return"


def _sum_return_cost(
    equity_curve: pd.DataFrame,
    column: str,
    *,
    fallback_column: str | None = None,
) -> float:
    series = _return_cost_series(equity_curve, column, fallback_column)
    if series.empty:
        return 0.0
    return float(series.sum())


def _return_cost_series(
    equity_curve: pd.DataFrame,
    column: str,
    fallback_column: str | None = None,
) -> pd.Series:
    for candidate in (column, fallback_column):
        if candidate and candidate in equity_curve:
            return pd.to_numeric(equity_curve[candidate], errors="coerce").fillna(0.0)
    return pd.Series(dtype=float)


def _returns_from_equity(equity: pd.Series) -> pd.Series:
    values = pd.to_numeric(equity, errors="coerce").fillna(1.0)
    previous = values.shift(1, fill_value=1.0)
    returns = values / previous.replace(0.0, np.nan) - 1.0
    return returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _resolve_position_date_column(positions: pd.DataFrame, date_column: str | None) -> str | None:
    if date_column is not None:
        return date_column
    for candidate in ("date", "holding_start_date", "effective_date", "signal_date"):
        if candidate in positions:
            return candidate
    return None


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"positions frame is missing required column(s): {joined}")


def _position_turnover_dates(frame_dates: pd.Series, date_index: Iterable[object] | None) -> pd.DatetimeIndex:
    if date_index is None:
        return _coerce_sorted_dates(frame_dates)
    return _coerce_sorted_dates(date_index)


def _coerce_sorted_dates(values: Iterable[object]) -> pd.DatetimeIndex:
    dates = pd.to_datetime(pd.Index(values), errors="coerce")
    dates = dates[~pd.isna(dates)]
    return pd.DatetimeIndex(dates.unique()).sort_values()


def _transaction_cost_scenario_row(
    scenario: Any,
    metrics: PerformanceMetrics,
    equity_curve: pd.DataFrame,
    *,
    baseline_scenario_id: str,
) -> dict[str, object]:
    average_daily_turnover_budget = float(scenario.average_daily_turnover_budget)
    max_daily_turnover = scenario.max_daily_turnover
    max_turnover_observed = (
        float(pd.to_numeric(equity_curve["turnover"], errors="coerce").fillna(0.0).max())
        if "turnover" in equity_curve and not equity_curve.empty
        else 0.0
    )
    scenario_id = str(scenario.scenario_id)
    cost_bps = float(scenario.cost_bps)
    slippage_bps = float(scenario.slippage_bps)
    return {
        "scenario_id": scenario_id,
        "scenario": scenario_id,
        "label": str(scenario.label),
        "is_baseline": scenario_id == baseline_scenario_id,
        "cost_bps": cost_bps,
        "slippage_bps": slippage_bps,
        "total_cost_bps": cost_bps + slippage_bps,
        "average_daily_turnover_budget": average_daily_turnover_budget,
        "max_daily_turnover": None if max_daily_turnover is None else float(max_daily_turnover),
        "observations": int(len(equity_curve)),
        "return_basis": metrics.return_basis,
        "cagr": metrics.cagr,
        "sharpe": metrics.sharpe,
        "max_drawdown": metrics.max_drawdown,
        "hit_rate": metrics.hit_rate,
        "turnover": metrics.turnover,
        "max_turnover": max_turnover_observed,
        "turnover_budget_pass": metrics.turnover <= average_daily_turnover_budget + 1e-12,
        "max_daily_turnover_pass": (
            True
            if max_daily_turnover is None
            else max_turnover_observed <= float(max_daily_turnover) + 1e-12
        ),
        "gross_cagr": metrics.gross_cagr,
        "gross_cumulative_return": metrics.gross_cumulative_return,
        "net_cagr": metrics.net_cagr,
        "cost_adjusted_cumulative_return": metrics.cost_adjusted_cumulative_return,
        "benchmark_cagr": metrics.benchmark_cagr,
        "benchmark_cost_adjusted_cagr": metrics.benchmark_cost_adjusted_cagr,
        "benchmark_cost_adjusted_cumulative_return": (
            metrics.benchmark_cost_adjusted_cumulative_return
        ),
        "excess_return": metrics.excess_return,
        "transaction_cost_return": metrics.transaction_cost_return,
        "slippage_cost_return": metrics.slippage_cost_return,
        "total_cost_return": metrics.total_cost_return,
    }


def _empty_transaction_cost_scenario_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "scenario_id",
            "scenario",
            "label",
            "is_baseline",
            "cost_bps",
            "slippage_bps",
            "total_cost_bps",
            "average_daily_turnover_budget",
            "max_daily_turnover",
            "observations",
            "return_basis",
            "cagr",
            "sharpe",
            "max_drawdown",
            "hit_rate",
            "turnover",
            "max_turnover",
            "turnover_budget_pass",
            "max_daily_turnover_pass",
            "gross_cagr",
            "gross_cumulative_return",
            "net_cagr",
            "cost_adjusted_cumulative_return",
            "benchmark_cagr",
            "benchmark_cost_adjusted_cagr",
            "benchmark_cost_adjusted_cumulative_return",
            "excess_return",
            "transaction_cost_return",
            "slippage_cost_return",
            "total_cost_return",
            "baseline_cost_adjusted_cumulative_return_delta",
            "baseline_excess_return_delta",
            "baseline_total_cost_return_delta",
        ]
    )


def _coerce_equity_curve_frame(value: object) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    if hasattr(value, "equity_curve"):
        equity_curve = value.equity_curve
        if isinstance(equity_curve, pd.DataFrame):
            return equity_curve
    raise TypeError("transaction cost scenario analysis requires an equity curve DataFrame")
