from __future__ import annotations

from pathlib import Path

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
from quant_research.features.sec import _daily_filing_features, build_sec_features


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


class _FakeFilingEventExtractor:
    def __init__(self, payload: dict[str, object] | None = None, model_id: str | None = None) -> None:
        self.payload = payload or {
            "event_tag": "material_event",
            "risk_flag": True,
            "confidence": 0.89,
            "summary_ref": "fake summary",
        }
        if model_id is not None:
            self.model_id = model_id
        self.calls = 0

    def extract(self, text: str) -> dict[str, object]:
        del text
        self.calls += 1
        return dict(self.payload)


def test_sec_daily_filing_features_reuses_cache_within_batch(tmp_path: Path) -> None:
    filings = pd.DataFrame(
        {
            "filing_date": ["2026-01-02", "2026-01-02"],
            "form": ["8-K", "8-K"],
            "primary_document": ["8k_q1.htm", "8k_q1.htm"],
            "accession_number": [None, None],
        }
    )
    fake_extractor = _FakeFilingEventExtractor()
    cache_path = tmp_path / "filing_cache.jsonl"

    result = _daily_filing_features(
        filings,
        "AAPL",
        filing_extractor=fake_extractor,
        filing_cache_path=cache_path,
    )

    assert fake_extractor.calls == 1
    assert "sec_event_tag" in result.columns
    assert result["sec_event_tag"].iloc[0] == "material_event"


def test_sec_daily_filing_features_reuses_cache_across_runs(tmp_path: Path) -> None:
    filings = pd.DataFrame(
        {
            "filing_date": ["2026-01-03", "2026-01-03"],
            "form": ["4", "4"],
            "primary_document": ["form4_20260103.xml", "form4_20260103.xml"],
            "accession_number": [None, None],
        }
    )
    cache_path = tmp_path / "filing_cache.jsonl"
    first_extractor = _FakeFilingEventExtractor(
        {"event_tag": "insider_activity", "risk_flag": False, "confidence": 0.54, "summary_ref": "cached"}
    )
    second_extractor = _FakeFilingEventExtractor(
        {"event_tag": "ignored", "risk_flag": False, "confidence": 0.1, "summary_ref": "cache-should-ignore"}
    )

    first = _daily_filing_features(
        filings,
        "AAPL",
        filing_extractor=first_extractor,
        filing_cache_path=cache_path,
    )
    second = _daily_filing_features(
        filings,
        "AAPL",
        filing_extractor=second_extractor,
        filing_cache_path=cache_path,
    )

    assert first_extractor.calls == 1
    assert second_extractor.calls == 0
    assert second["sec_event_tag"].iloc[0] == "insider_activity"
    assert first["sec_event_confidence"].iloc[0] == second["sec_event_confidence"].iloc[0]


def test_sec_daily_filing_features_regenerates_when_cache_broken(tmp_path: Path) -> None:
    filings = pd.DataFrame(
        {
            "filing_date": ["2026-01-04"],
            "form": ["10-Q"],
            "primary_document": ["10q.htm"],
            "accession_number": [None],
        }
    )
    cache_path = tmp_path / "filing_cache.jsonl"
    cache_path.write_text("{broken-json")
    extractor = _FakeFilingEventExtractor(
        {
            "event_tag": "quarterly_report",
            "risk_flag": False,
            "confidence": 0.72,
            "summary_ref": "fixed",
        }
    )

    result = _daily_filing_features(
        filings,
        "MSFT",
        filing_extractor=extractor,
        filing_cache_path=cache_path,
    )

    assert extractor.calls == 1
    assert result["sec_event_tag"].iloc[0] == "quarterly_report"


def test_sec_daily_filing_features_separates_cache_by_extractor_namespace(tmp_path: Path) -> None:
    filings = pd.DataFrame(
        {
            "filing_date": ["2026-01-05"],
            "form": ["8-K"],
            "primary_document": ["8k.htm"],
            "accession_number": ["0000000000-26-000001"],
        }
    )
    cache_path = tmp_path / "filing_cache.jsonl"
    rules_extractor = _FakeFilingEventExtractor(
        {"event_tag": "current_report", "risk_flag": False, "confidence": 0.4, "summary_ref": "rules"},
        model_id="rules",
    )
    llm_extractor = _FakeFilingEventExtractor(
        {"event_tag": "guidance", "risk_flag": True, "confidence": 0.91, "summary_ref": "llm"},
        model_id="fingpt",
    )

    first = _daily_filing_features(
        filings,
        "AAPL",
        filing_extractor=rules_extractor,
        filing_cache_path=cache_path,
    )
    second = _daily_filing_features(
        filings,
        "AAPL",
        filing_extractor=llm_extractor,
        filing_cache_path=cache_path,
    )

    assert rules_extractor.calls == 1
    assert llm_extractor.calls == 1
    assert first["sec_event_tag"].iloc[0] == "current_report"
    assert second["sec_event_tag"].iloc[0] == "guidance"
