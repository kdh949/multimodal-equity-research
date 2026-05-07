from __future__ import annotations

import pandas as pd
import pytest

from quant_research.backtest import (
    covariance_to_correlation_matrix,
    estimate_correlation_matrix,
    estimate_covariance_matrix,
    estimate_portfolio_covariance_matrix,
    prepare_covariance_return_matrix,
)


def test_prepare_covariance_return_matrix_sorts_and_aligns_asset_returns() -> None:
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-06", "2026-01-05", "2026-01-06", "2026-01-05"]
            ),
            "ticker": [" MSFT ", "AAPL", "AAPL", "MSFT"],
            "return_1": [0.02, 0.01, 0.03, -0.01],
        }
    )

    matrix = prepare_covariance_return_matrix(history)

    assert matrix.index.tolist() == [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")]
    assert matrix.columns.tolist() == ["AAPL", "MSFT"]
    assert matrix.loc[pd.Timestamp("2026-01-05"), "AAPL"] == pytest.approx(0.01)
    assert matrix.loc[pd.Timestamp("2026-01-06"), "MSFT"] == pytest.approx(0.02)


def test_prepare_covariance_return_matrix_preserves_requested_ticker_alignment() -> None:
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-05", "2026-01-05", "2026-01-06", "2026-01-06"]),
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "return_1": [0.01, -0.01, 0.03, 0.02],
        }
    )

    matrix = prepare_covariance_return_matrix(history, tickers=["MSFT", "AAPL"])

    assert matrix.columns.tolist() == ["MSFT", "AAPL"]
    assert matrix.loc[pd.Timestamp("2026-01-05"), "MSFT"] == pytest.approx(-0.01)
    assert matrix.loc[pd.Timestamp("2026-01-05"), "AAPL"] == pytest.approx(0.01)
    assert matrix.loc[pd.Timestamp("2026-01-06"), "MSFT"] == pytest.approx(0.02)
    assert matrix.loc[pd.Timestamp("2026-01-06"), "AAPL"] == pytest.approx(0.03)


def test_prepare_covariance_return_matrix_rejects_duplicate_asset_date_observations() -> None:
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-05", "2026-01-05", "2026-01-06"]),
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "return_1": [0.01, 0.02, 0.03],
        }
    )

    with pytest.raises(ValueError, match="duplicate date/ticker"):
        prepare_covariance_return_matrix(history)


def test_prepare_covariance_return_matrix_rejects_misaligned_return_history() -> None:
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-05", "2026-01-05", "2026-01-06"]),
            "ticker": ["AAPL", "MSFT", "AAPL"],
            "return_1": [0.01, -0.01, 0.03],
        }
    )

    with pytest.raises(ValueError, match="not aligned"):
        prepare_covariance_return_matrix(history, tickers=["AAPL", "MSFT"])


def test_prepare_covariance_return_matrix_rejects_missing_requested_ticker() -> None:
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
            "ticker": ["AAPL", "AAPL"],
            "return_1": [0.01, 0.03],
        }
    )

    with pytest.raises(ValueError, match="missing requested tickers"):
        prepare_covariance_return_matrix(history, tickers=["AAPL", "MSFT"])


def test_estimate_covariance_matrix_uses_aligned_sample_returns() -> None:
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-05",
                    "2026-01-05",
                    "2026-01-06",
                    "2026-01-06",
                    "2026-01-07",
                    "2026-01-07",
                ]
            ),
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT"],
            "return_1": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        }
    )

    covariance = estimate_covariance_matrix(history, tickers=["MSFT", "AAPL"])

    expected = prepare_covariance_return_matrix(history, tickers=["MSFT", "AAPL"]).cov()
    pd.testing.assert_frame_equal(covariance, expected)
    assert covariance.index.tolist() == ["MSFT", "AAPL"]
    assert covariance.columns.tolist() == ["MSFT", "AAPL"]
    assert covariance.loc["AAPL", "MSFT"] == pytest.approx(covariance.loc["MSFT", "AAPL"])


def test_estimate_covariance_matrix_applies_optional_annualization_factor() -> None:
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-05", "2026-01-05", "2026-01-06", "2026-01-06"]
            ),
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "return_1": [0.01, 0.02, 0.03, 0.04],
        }
    )

    daily = estimate_covariance_matrix(history)
    annualized = estimate_covariance_matrix(history, annualization_factor=252)

    pd.testing.assert_frame_equal(annualized, daily * 252)


def test_estimate_portfolio_covariance_matrix_uses_held_tickers_only() -> None:
    positions = pd.DataFrame(
        {
            "ticker": ["MSFT", "CASH", "AAPL"],
            "weight": [0.40, 0.0, 0.60],
        }
    )
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-05",
                    "2026-01-05",
                    "2026-01-05",
                    "2026-01-06",
                    "2026-01-06",
                    "2026-01-06",
                    "2026-01-07",
                    "2026-01-07",
                    "2026-01-07",
                ]
            ),
            "ticker": ["AAPL", "MSFT", "NVDA"] * 3,
            "return_1": [0.01, 0.02, -0.01, 0.03, 0.04, -0.02, 0.05, 0.06, -0.03],
        }
    )

    covariance = estimate_portfolio_covariance_matrix(positions, history)

    expected = prepare_covariance_return_matrix(history, tickers=["MSFT", "AAPL"]).cov()
    pd.testing.assert_frame_equal(covariance, expected)
    assert covariance.index.tolist() == ["MSFT", "AAPL"]
    assert "NVDA" not in covariance.columns
    assert "CASH" not in covariance.columns


def test_estimate_correlation_matrix_returns_finite_symmetric_matrix() -> None:
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-05",
                    "2026-01-05",
                    "2026-01-06",
                    "2026-01-06",
                    "2026-01-07",
                    "2026-01-07",
                ]
            ),
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT"],
            "return_1": [0.01, 0.01, 0.02, 0.01, 0.03, 0.01],
        }
    )

    correlation = estimate_correlation_matrix(history)

    assert correlation.index.tolist() == ["AAPL", "MSFT"]
    assert correlation.columns.tolist() == ["AAPL", "MSFT"]
    assert correlation.loc["AAPL", "AAPL"] == pytest.approx(1.0)
    assert correlation.loc["MSFT", "MSFT"] == pytest.approx(1.0)
    assert correlation.loc["AAPL", "MSFT"] == pytest.approx(0.0)
    assert correlation.loc["MSFT", "AAPL"] == pytest.approx(0.0)


def test_covariance_to_correlation_matrix_rejects_invalid_diagonal() -> None:
    covariance = pd.DataFrame(
        [[0.01, 0.0], [0.0, -0.01]],
        index=["AAPL", "MSFT"],
        columns=["AAPL", "MSFT"],
    )

    with pytest.raises(ValueError, match="diagonal must be non-negative"):
        covariance_to_correlation_matrix(covariance)
