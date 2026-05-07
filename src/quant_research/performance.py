from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class ReturnSeriesMetrics:
    cagr: float
    sharpe: float
    max_drawdown: float
    cumulative_return: float
    observations: int
    evaluation_start: str | None = None
    evaluation_end: str | None = None

    def to_dict(self) -> dict[str, float | int | str | None]:
        return asdict(self)


def align_return_series(
    returns: object,
    dates: object | None = None,
    *,
    name: str = "return",
) -> pd.Series:
    """Return a numeric series aligned to optional dates for metric calculations."""

    values = _coerce_return_series(returns, name=name)
    if dates is None:
        return values

    date_values = _coerce_date_series(dates)
    if len(values) != len(date_values):
        raise ValueError("returns and dates must have the same length")

    frame = pd.DataFrame(
        {
            "date": date_values.to_numpy(),
            name: values.to_numpy(dtype=float),
        }
    )
    frame = frame.dropna(subset=["date"]).sort_values("date")
    return pd.Series(
        frame[name].to_numpy(dtype=float),
        index=pd.DatetimeIndex(frame["date"], name="date"),
        name=name,
    )


def calculate_cagr(
    returns: object,
    dates: object | None = None,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    values = align_return_series(returns, dates)
    if values.empty:
        return 0.0

    equity = _equity_curve(values)
    ending_equity = float(equity.iloc[-1])
    if ending_equity <= 0:
        return -1.0

    years = max(len(values) / float(periods_per_year), 1.0 / float(periods_per_year))
    return float(ending_equity ** (1.0 / years) - 1.0)


def calculate_sharpe(
    returns: object,
    dates: object | None = None,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    values = align_return_series(returns, dates)
    if values.empty:
        return 0.0

    std = float(values.std(ddof=0))
    if std <= 0:
        return 0.0
    return float(values.mean() / std * np.sqrt(periods_per_year))


def calculate_max_drawdown(returns: object, dates: object | None = None) -> float:
    values = align_return_series(returns, dates)
    if values.empty:
        return 0.0

    equity = _equity_curve(values)
    equity_with_initial_capital = pd.concat(
        [pd.Series([1.0], index=[-1], dtype=float), equity.reset_index(drop=True)]
    ).reset_index(drop=True)
    drawdown = equity_with_initial_capital / equity_with_initial_capital.cummax() - 1.0
    return float(drawdown.min())


def calculate_cumulative_return(returns: object, dates: object | None = None) -> float:
    values = align_return_series(returns, dates)
    if values.empty:
        return 0.0
    return float(_equity_curve(values).iloc[-1] - 1.0)


def calculate_return_series_metrics(
    returns: object,
    dates: object | None = None,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> ReturnSeriesMetrics:
    values = align_return_series(returns, dates)
    if values.empty:
        return ReturnSeriesMetrics(
            cagr=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            cumulative_return=0.0,
            observations=0,
            evaluation_start=None,
            evaluation_end=None,
        )

    return ReturnSeriesMetrics(
        cagr=calculate_cagr(values, periods_per_year=periods_per_year),
        sharpe=calculate_sharpe(values, periods_per_year=periods_per_year),
        max_drawdown=calculate_max_drawdown(values),
        cumulative_return=calculate_cumulative_return(values),
        observations=int(len(values)),
        evaluation_start=_date_bound(values.index, "min"),
        evaluation_end=_date_bound(values.index, "max"),
    )


def _coerce_return_series(values: object, *, name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.copy()
    elif np.isscalar(values):
        series = pd.Series([values])
    else:
        series = pd.Series(values)
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    numeric.name = name
    return numeric


def _coerce_date_series(values: object) -> pd.Series:
    dates = pd.Series(pd.to_datetime(values, errors="coerce"))
    if dates.empty:
        return dates
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    return dates.dt.normalize()


def _equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.astype(float)).cumprod()


def _date_bound(index: pd.Index, method: str) -> str | None:
    if not isinstance(index, pd.DatetimeIndex):
        return None
    values = pd.Series(index).dropna()
    if values.empty:
        return None
    bound: Any = values.min() if method == "min" else values.max()
    return pd.Timestamp(bound).date().isoformat()
