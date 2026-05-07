from __future__ import annotations

import pandas as pd
import pytest

from quant_research.backtest.metrics import (
    calculate_covariance_aware_portfolio_risk_metrics,
    calculate_portfolio_risk_metrics,
)


def test_portfolio_risk_metrics_calculates_volatility_and_risk_contribution() -> None:
    positions = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "weight": [0.60, 0.30],
            "sector": ["Information Technology", "Information Technology"],
        }
    )
    covariance = pd.DataFrame(
        [[0.0004, 0.0001], [0.0001, 0.0009]],
        index=["AAPL", "MSFT"],
        columns=["AAPL", "MSFT"],
    )

    metrics = calculate_portfolio_risk_metrics(positions, covariance=covariance)

    expected_variance = (0.60**2 * 0.0004) + (2 * 0.60 * 0.30 * 0.0001) + (0.30**2 * 0.0009)
    expected_volatility = expected_variance**0.5
    assert metrics.portfolio_volatility == pytest.approx(expected_volatility)
    assert metrics.gross_exposure == pytest.approx(0.90)
    assert metrics.net_exposure == pytest.approx(0.90)
    assert metrics.max_symbol_weight == pytest.approx(0.60)
    assert metrics.max_sector_weight == pytest.approx(0.90)
    assert metrics.effective_holdings == pytest.approx(1 / ((2 / 3) ** 2 + (1 / 3) ** 2))

    contribution = metrics.risk_contributions.set_index("ticker")
    assert contribution["risk_contribution"].sum() == pytest.approx(expected_volatility)
    assert contribution["risk_contribution_pct"].sum() == pytest.approx(1.0)
    assert (
        contribution.loc["AAPL", "risk_contribution"]
        > contribution.loc["MSFT", "risk_contribution"]
    )


def test_portfolio_risk_metrics_uses_position_volatility_fallback() -> None:
    positions = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "JPM"],
            "weight": [0.40, 0.30, 0.20],
            "predicted_volatility": [0.20, 0.10, 0.15],
            "sector": ["Technology", "Technology", "Financials"],
        }
    )

    metrics = calculate_portfolio_risk_metrics(positions)

    expected_volatility = ((0.40 * 0.20) ** 2 + (0.30 * 0.10) ** 2 + (0.20 * 0.15) ** 2) ** 0.5
    assert metrics.portfolio_volatility == pytest.approx(expected_volatility)
    assert metrics.max_sector_weight == pytest.approx(0.70)
    assert metrics.sector_exposures.set_index("sector").loc[
        "Technology", "weight"
    ] == pytest.approx(0.70)


def test_covariance_aware_portfolio_risk_metrics_estimates_held_return_covariance() -> None:
    positions = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "weight": [0.60, 0.40],
            "sector": ["Technology", "Technology"],
            "predicted_volatility": [0.01, 0.01],
        }
    )
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
            "return_1": [0.02, 0.01, -0.02, -0.01, 0.02, 0.01],
        }
    )

    metrics = calculate_covariance_aware_portfolio_risk_metrics(positions, history)

    covariance = history.pivot(index="date", columns="ticker", values="return_1").cov()
    weights = pd.Series({"AAPL": 0.60, "MSFT": 0.40})
    expected_volatility = float((weights @ covariance.loc[weights.index, weights.index] @ weights) ** 0.5)
    assert metrics.portfolio_volatility == pytest.approx(expected_volatility)
    assert metrics.risk_contributions["risk_contribution_pct"].sum() == pytest.approx(1.0)
    assert metrics.max_sector_weight == pytest.approx(1.0)


def test_portfolio_risk_metrics_rejects_negative_long_only_weight() -> None:
    positions = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "weight": [0.5, -0.1]})

    with pytest.raises(ValueError, match="long-only"):
        calculate_portfolio_risk_metrics(positions)


def test_portfolio_risk_metrics_rejects_covariance_without_all_tickers() -> None:
    positions = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "weight": [0.5, 0.5]})
    covariance = pd.DataFrame([[0.01]], index=["AAPL"], columns=["AAPL"])

    with pytest.raises(ValueError, match="missing held tickers"):
        calculate_portfolio_risk_metrics(positions, covariance=covariance)
