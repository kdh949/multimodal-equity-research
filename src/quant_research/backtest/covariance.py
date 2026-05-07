from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def estimate_portfolio_covariance_matrix(
    positions: pd.DataFrame,
    return_history: pd.DataFrame,
    *,
    ticker_column: str = "ticker",
    weight_column: str = "weight",
    date_column: str = "date",
    return_ticker_column: str = "ticker",
    return_column: str = "return_1",
    min_periods: int = 2,
    annualization_factor: float | None = None,
) -> pd.DataFrame:
    """Estimate covariance for the non-zero holdings in a portfolio.

    This keeps the covariance sample aligned to the portfolio's held tickers so
    downstream risk metrics cannot silently mix in assets that are outside the
    evaluated portfolio.
    """

    _require_columns(positions, [ticker_column, weight_column])
    tickers = _held_position_tickers(
        positions,
        ticker_column=ticker_column,
        weight_column=weight_column,
    )
    if not tickers:
        return pd.DataFrame()
    return estimate_covariance_matrix(
        return_history,
        tickers=tickers,
        date_column=date_column,
        ticker_column=return_ticker_column,
        return_column=return_column,
        min_periods=min_periods,
        annualization_factor=annualization_factor,
    )


def prepare_covariance_return_matrix(
    return_history: pd.DataFrame,
    *,
    tickers: Iterable[object] | None = None,
    date_column: str = "date",
    ticker_column: str = "ticker",
    return_column: str = "return_1",
    min_periods: int = 2,
) -> pd.DataFrame:
    """Normalize and validate asset return history before covariance estimation."""

    _require_columns(return_history, [date_column, ticker_column, return_column])
    if min_periods < 1:
        raise ValueError("min_periods must be at least 1")

    requested_tickers = _normalize_requested_tickers(tickers)
    frame = return_history[[date_column, ticker_column, return_column]].copy()
    frame["date"] = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()
    frame["ticker"] = frame[ticker_column].fillna("").astype(str).str.strip()
    frame["return"] = pd.to_numeric(frame[return_column], errors="coerce")

    invalid = frame["date"].isna() | frame["ticker"].eq("") | frame["return"].isna()
    if invalid.any():
        raise ValueError("covariance return history contains invalid date, ticker, or return values")

    if requested_tickers is not None:
        frame = frame[frame["ticker"].isin(requested_tickers)]

    duplicate = frame.duplicated(["date", "ticker"], keep=False)
    if duplicate.any():
        raise ValueError("covariance return history contains duplicate date/ticker observations")

    if frame.empty:
        return pd.DataFrame()

    returns = (
        frame.pivot(index="date", columns="ticker", values="return")
        .sort_index()
        .sort_index(axis=1)
    )
    if requested_tickers is not None:
        missing = [ticker for ticker in requested_tickers if ticker not in returns.columns]
        if missing:
            raise ValueError(f"covariance return history is missing requested tickers: {missing}")
        returns = returns.loc[:, requested_tickers]

    incomplete_rows = returns.isna().any(axis=1)
    if incomplete_rows.any():
        raise ValueError("covariance return history is not aligned across tickers by date")
    if len(returns) < min_periods:
        raise ValueError(
            f"covariance return history requires at least {min_periods} aligned periods"
        )
    return returns


def estimate_covariance_matrix(
    return_history: pd.DataFrame,
    *,
    tickers: Iterable[object] | None = None,
    date_column: str = "date",
    ticker_column: str = "ticker",
    return_column: str = "return_1",
    min_periods: int = 2,
    annualization_factor: float | None = None,
) -> pd.DataFrame:
    """Estimate an asset return covariance matrix from aligned historical returns."""

    returns = prepare_covariance_return_matrix(
        return_history,
        tickers=tickers,
        date_column=date_column,
        ticker_column=ticker_column,
        return_column=return_column,
        min_periods=min_periods,
    )
    covariance = returns.cov()
    covariance = _finalize_square_matrix(covariance, "covariance")
    if annualization_factor is not None:
        factor = _positive_float(annualization_factor, "annualization_factor")
        covariance = covariance * factor
    return covariance


def estimate_correlation_matrix(
    return_history: pd.DataFrame,
    *,
    tickers: Iterable[object] | None = None,
    date_column: str = "date",
    ticker_column: str = "ticker",
    return_column: str = "return_1",
    min_periods: int = 2,
) -> pd.DataFrame:
    """Estimate an asset return correlation matrix from aligned historical returns."""

    covariance = estimate_covariance_matrix(
        return_history,
        tickers=tickers,
        date_column=date_column,
        ticker_column=ticker_column,
        return_column=return_column,
        min_periods=min_periods,
    )
    return covariance_to_correlation_matrix(covariance)


def covariance_to_correlation_matrix(covariance: pd.DataFrame) -> pd.DataFrame:
    """Convert a covariance matrix to a finite correlation matrix."""

    covariance = _finalize_square_matrix(covariance, "covariance")
    diagonal = np.diag(covariance.to_numpy(dtype=float))
    if (diagonal < -1e-12).any():
        raise ValueError("covariance matrix diagonal must be non-negative")
    stddev = np.sqrt(np.maximum(diagonal, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        values = covariance.to_numpy(dtype=float) / np.outer(stddev, stddev)
    values = np.where(np.isfinite(values), values, 0.0)
    values = np.clip(values, -1.0, 1.0)
    np.fill_diagonal(values, 1.0)
    correlation = pd.DataFrame(values, index=covariance.index, columns=covariance.columns)
    return _finalize_square_matrix(correlation, "correlation")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"covariance return history missing required columns: {sorted(missing)}")


def _normalize_requested_tickers(tickers: Iterable[object] | None) -> list[str] | None:
    if tickers is None:
        return None
    normalized: list[str] = []
    for ticker in tickers:
        value = "" if pd.isna(ticker) else str(ticker).strip()
        if not value:
            raise ValueError("covariance requested tickers must be non-empty")
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("covariance requested tickers must be non-empty")
    return normalized


def _held_position_tickers(
    positions: pd.DataFrame,
    *,
    ticker_column: str,
    weight_column: str,
) -> list[str]:
    frame = positions[[ticker_column, weight_column]].copy()
    frame["ticker"] = frame[ticker_column].fillna("").astype(str).str.strip()
    frame["weight"] = pd.to_numeric(frame[weight_column], errors="coerce")
    invalid = frame["ticker"].eq("") | frame["weight"].isna()
    if invalid.any():
        raise ValueError("portfolio positions contain invalid ticker or weight values")
    if not np.isfinite(frame["weight"].to_numpy(dtype=float)).all():
        raise ValueError("portfolio positions contain non-finite weights")
    tickers: list[str] = []
    for ticker in frame.loc[frame["weight"].abs() > 1e-12, "ticker"]:
        if ticker not in tickers:
            tickers.append(ticker)
    return tickers


def _finalize_square_matrix(matrix: pd.DataFrame, name: str) -> pd.DataFrame:
    if matrix.empty:
        return matrix
    if list(matrix.index) != list(matrix.columns):
        raise ValueError(f"{name} matrix index and columns must match")
    numeric = matrix.astype(float)
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} matrix contains non-finite values")
    symmetric = (values + values.T) / 2.0
    return pd.DataFrame(symmetric, index=numeric.index, columns=numeric.columns)


def _positive_float(value: float, name: str) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be positive")
    return numeric
