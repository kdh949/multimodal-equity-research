from __future__ import annotations

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from quant_research.data.market import SyntheticMarketDataProvider
from quant_research.data.sec import SyntheticSecProvider
from quant_research.features.price import build_price_features
from quant_research.features.sec import build_sec_features


def test_sec_features_normalize_filings_and_facts() -> None:
    market = build_price_features(SyntheticMarketDataProvider(periods=140).get_history(["AAPL"]))
    provider = SyntheticSecProvider()
    filings = {"AAPL": provider.recent_filings("320193")}
    facts = {"AAPL": provider.companyfacts_frame("320193")}

    sec_features = build_sec_features(filings, facts, market)

    assert {"sec_8k_count", "sec_10q_count", "sec_form4_count", "revenue_growth"}.issubset(
        sec_features.columns
    )
    assert sec_features["sec_risk_flag_20d"].max() >= 0
    assert is_datetime64_any_dtype(sec_features["date"])


def test_sec_features_handle_empty_inputs() -> None:
    calendar = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=5), "ticker": "AAPL"})
    sec_features = build_sec_features({}, {}, calendar)

    assert len(sec_features) == 5
    assert sec_features["revenue_growth"].eq(0).all()
