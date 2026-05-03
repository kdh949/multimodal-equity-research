from __future__ import annotations

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from quant_research.data.market import SyntheticMarketDataProvider
from quant_research.data.sec import (
    SyntheticSecProvider,
    extract_companyconcept_frame,
    extract_frame_values,
    merge_fact_frames,
)
from quant_research.features.price import build_price_features
from quant_research.features.sec import build_sec_features


def test_sec_features_normalize_filings_and_facts() -> None:
    market = build_price_features(SyntheticMarketDataProvider(periods=140).get_history(["AAPL"]))
    provider = SyntheticSecProvider()
    filings = {"AAPL": provider.recent_filings("320193")}
    facts = {"AAPL": provider.companyfacts_frame("320193").assign(sec_frame_assets=123.0)}

    sec_features = build_sec_features(filings, facts, market)

    assert {
        "sec_8k_count",
        "sec_10q_count",
        "sec_form4_count",
        "revenue_growth",
        "sec_event_tag",
        "sec_event_confidence",
        "sec_summary_ref",
        "sec_frame_assets",
    }.issubset(sec_features.columns)
    assert sec_features["sec_risk_flag_20d"].max() >= 0
    assert is_datetime64_any_dtype(sec_features["date"])


def test_sec_features_handle_empty_inputs() -> None:
    calendar = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=5), "ticker": "AAPL"})
    sec_features = build_sec_features({}, {}, calendar)

    assert len(sec_features) == 5
    assert sec_features["revenue_growth"].eq(0).all()


def test_sec_companyconcept_and_frame_extractors_normalize_payloads() -> None:
    concept = {
        "units": {
            "USD": [
                {"end": "2025-12-31", "val": "100"},
                {"end": "2026-03-31", "val": "120"},
            ]
        }
    }
    frame_payload = {"data": [{"cik": 320193, "val": 5000}, {"cik": "0000789019", "val": 7000}]}

    concept_frame = extract_companyconcept_frame(concept, "net_income")
    frame_values = extract_frame_values(frame_payload, "sec_frame_assets")
    merged = merge_fact_frames(pd.DataFrame({"period_end": pd.to_datetime(["2025-12-31"])}), concept_frame)

    assert concept_frame["net_income"].tolist() == [100, 120]
    assert frame_values["cik"].tolist() == ["0000320193", "0000789019"]
    assert "net_income" in merged.columns
