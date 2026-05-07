from __future__ import annotations

import pandas as pd
import pytest

from quant_research.validation import (
    align_return_series,
    calculate_cagr,
    calculate_max_drawdown,
    calculate_return_series_metrics,
    calculate_sharpe,
)


def test_return_series_metrics_calculate_cagr_sharpe_and_max_drawdown() -> None:
    returns = pd.Series([-0.10, 0.02, 0.03])

    metrics = calculate_return_series_metrics(returns)

    expected_cagr = float(((1 + returns).prod()) ** (252 / len(returns)) - 1)
    expected_sharpe = float(returns.mean() / returns.std(ddof=0) * (252**0.5))
    assert metrics.cagr == pytest.approx(expected_cagr)
    assert metrics.sharpe == pytest.approx(expected_sharpe)
    assert metrics.max_drawdown == pytest.approx(-0.10)
    assert calculate_cagr(returns) == pytest.approx(expected_cagr)
    assert calculate_sharpe(returns) == pytest.approx(expected_sharpe)
    assert calculate_max_drawdown(returns) == pytest.approx(-0.10)


def test_return_series_metrics_align_returns_to_dates_before_reporting_bounds() -> None:
    aligned = align_return_series(
        [0.02, 0.01, None],
        ["2026-01-05", "2026-01-02", "not-a-date"],
    )

    metrics = calculate_return_series_metrics(
        [0.02, 0.01, None],
        ["2026-01-05", "2026-01-02", "not-a-date"],
    )

    assert aligned.tolist() == pytest.approx([0.01, 0.02])
    assert [value.date().isoformat() for value in aligned.index] == [
        "2026-01-02",
        "2026-01-05",
    ]
    assert metrics.observations == 2
    assert metrics.evaluation_start == "2026-01-02"
    assert metrics.evaluation_end == "2026-01-05"
    assert metrics.cumulative_return == pytest.approx((1 + 0.01) * (1 + 0.02) - 1)
    assert calculate_cagr(
        [0.02, 0.01, None],
        ["2026-01-05", "2026-01-02", "not-a-date"],
    ) == pytest.approx(metrics.cagr)


def test_return_series_metrics_require_aligned_return_and_date_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        calculate_return_series_metrics([0.01], ["2026-01-02", "2026-01-05"])


def test_return_series_metrics_coerce_missing_and_infinite_returns_to_zero() -> None:
    returns = pd.Series([0.10, None, float("inf"), -0.50])

    metrics = calculate_return_series_metrics(returns)

    expected_clean_returns = pd.Series([0.10, 0.0, 0.0, -0.50])
    assert metrics.observations == 4
    assert metrics.cumulative_return == pytest.approx(
        (1 + expected_clean_returns).prod() - 1
    )
    assert metrics.max_drawdown == pytest.approx(-0.50)


def test_return_series_metrics_return_zero_sharpe_for_zero_volatility_series() -> None:
    returns = pd.Series([0.01, 0.01, 0.01])

    metrics = calculate_return_series_metrics(returns)

    assert metrics.sharpe == 0.0
    assert calculate_sharpe(returns) == 0.0


def test_return_series_metrics_respect_custom_periods_per_year_for_annualization() -> None:
    returns = pd.Series([0.10, 0.10])

    metrics = calculate_return_series_metrics(returns, periods_per_year=2)

    assert metrics.cagr == pytest.approx((1.10 * 1.10) - 1.0)
