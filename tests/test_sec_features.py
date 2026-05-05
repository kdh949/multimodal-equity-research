from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from quant_research.data import sec as sec_module
from quant_research.data.market import SyntheticMarketDataProvider
from quant_research.data.sec import (
    SyntheticSecProvider,
    extract_companyconcept_frame,
    extract_frame_values,
    merge_fact_frames,
)
from quant_research.features.price import build_price_features
from quant_research.features.sec import (
    _daily_filing_features,
    _extract_filing_sections,
    build_sec_features,
)


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
        self.received_texts: list[str] = []

    def extract(self, text: str) -> dict[str, object]:
        self.received_texts.append(text)
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


def test_sec_daily_filing_features_prefers_document_text_and_emits_body_stats() -> None:
    filings = pd.DataFrame(
        {
            "filing_date": ["2026-01-06"],
            "form": ["8-K"],
            "primary_document": ["8k.htm"],
            "accession_number": ["0000000000-26-000001"],
            "document_text": ["Company announced risks. The filing discusses lawsuit and investigation risk."],
        }
    )
    extractor = _FakeFilingEventExtractor(
        {"event_tag": "legal", "risk_flag": False, "confidence": 0.81, "summary_ref": "risk heavy"}
    )

    result = _daily_filing_features(
        filings,
        "AAPL",
        filing_extractor=extractor,
    )

    assert extractor.calls == 1
    assert extractor.received_texts[0] == filings["document_text"].iloc[0]
    assert result["sec_filing_text_available"].iloc[0] == 1.0
    assert result["sec_filing_text_length"].iloc[0] == float(len(filings["document_text"].iloc[0]))
    assert result["sec_filing_risk_keyword_count"].iloc[0] == float(4)


def test_sec_daily_filing_features_uses_fallback_text_when_body_is_missing() -> None:
    filings = pd.DataFrame(
        {
            "filing_date": ["2026-01-08"],
            "form": ["8-K"],
            "primary_document": ["8k.htm"],
            "accession_number": ["0000000000-26-000008"],
            "document_text": [float("nan")],
        }
    )
    extractor = _FakeFilingEventExtractor(
        {"event_tag": "fallback", "risk_flag": False, "confidence": 0.22, "summary_ref": "missing body"}
    )

    result = _daily_filing_features(
        filings,
        "AAPL",
        filing_extractor=extractor,
    )

    assert extractor.calls == 1
    assert extractor.received_texts[0] == "Form 8-K 8k.htm"
    assert result["sec_filing_text_available"].iloc[0] == 0.0
    assert result["sec_filing_text_length"].iloc[0] == 0.0


def test_sec_daily_filing_features_extracts_filing_sections_and_risk_section_count() -> None:
    filings = pd.DataFrame(
        {
            "filing_date": ["2026-01-07"],
            "form": ["8-K"],
            "primary_document": ["8k.htm"],
            "accession_number": ["0000000000-26-000007"],
            "document_text": [
                "Item 1.01 Business Overview. "
                "Item 1A Risk Factors. The company faces litigation and investigation risks. "
                "Item 2 Properties and operations."
            ],
        }
    )
    extractor = _FakeFilingEventExtractor(
        {
            "event_tag": "material_event",
            "risk_flag": False,
            "confidence": 0.72,
            "summary_ref": "sections parsed",
        }
    )

    result = _daily_filing_features(
        filings,
        "AAPL",
        filing_extractor=extractor,
    )

    assert result["sec_filing_sections"].iloc[0] == "item_1_01|item_1a|item_2"
    assert result["sec_filing_section_count"].iloc[0] == 3.0
    assert result["sec_filing_risk_section_count"].iloc[0] == 1.0


def test_sec_extract_sections_helper_recognizes_item_headings() -> None:
    text = (
        "Item 1.01 Overview Item 1A Risk Factors. "
        "Item 2.01 Properties. Item 3 results."
    )

    sections = _extract_filing_sections(text)

    assert sections == ["item_1_01", "item_1a", "item_2_01", "item_3"]


def test_sec_fetch_filing_document_text_parses_html_and_reuses_cache(tmp_path, monkeypatch) -> None:
    class _Response:
        def __init__(self, content: bytes, content_type: str) -> None:
            self.content = content
            self.headers = {"content-type": content_type}

        def raise_for_status(self) -> None:
            return None

    calls: list[str] = []
    payload = b"<html><body><h1>8-K filing</h1><p>Quarterly risk of litigation appears.</p><script>ignore()</script></body></html>"

    def fake_get(url: str, timeout: float | None = None) -> _Response:
        del timeout
        calls.append(url)
        return _Response(payload, "text/html")

    client = sec_module.SecEdgarClient(cache_dir=tmp_path / "sec")
    monkeypatch.setattr(client.session, "get", fake_get)

    text = client.fetch_filing_document("320193", "0000320193-26-000001", "primary.html")
    assert "ignore" not in text
    assert "risk" in text.lower()
    assert calls == [
        "https://www.sec.gov/Archives/edgar/data/320193/000032019326000001/primary.html"
    ]

    calls.clear()
    text_again = client.fetch_filing_document("320193", "0000320193-26-000001", "primary.html")
    assert text == text_again
    assert not calls


def test_sec_get_json_regenerates_when_cache_is_corrupt(tmp_path, monkeypatch) -> None:
    class _Response:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        @property
        def content(self) -> bytes:
            return json.dumps(self.payload).encode("utf-8")

        @property
        def headers(self) -> dict[str, str]:
            return {"content-type": "application/json"}

        def json(self) -> dict[str, object]:
            return self.payload

    calls: list[str] = []
    payload = {"facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [{"end": "2026-01-01", "val": 1}]}}}}}

    def fake_get(url: str, timeout: float | None = None) -> _Response:
        del timeout
        calls.append(url)
        return _Response(payload)

    cache_dir = tmp_path / "sec"
    client = sec_module.SecEdgarClient(cache_dir=cache_dir)
    cache_path = cache_dir / "companyfacts_0000320193.json"
    cache_path.write_text("{broken-json")

    monkeypatch.setattr(client.session, "get", fake_get)
    result = client.get_companyfacts("320193")

    assert calls == ["https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"]
    assert result == payload


def test_sec_fetch_filing_document_text_parses_xml_content(tmp_path, monkeypatch) -> None:
    class _Response:
        def __init__(self, content: bytes) -> None:
            self.content = content
            self.headers = {"content-type": "application/xml"}

        def raise_for_status(self) -> None:
            return None

    xml = b"<?xml version='1.0'?><filing><body>Litigation disclosure and risk report.</body></filing>"

    def fake_get(url: str, timeout: float | None = None) -> _Response:
        del url
        del timeout
        return _Response(xml)

    client = sec_module.SecEdgarClient(cache_dir=tmp_path / "sec_xml")
    monkeypatch.setattr(client.session, "get", fake_get)

    text = client.fetch_filing_document("320193", "0000320193-26-000002", "primary.xml")
    assert "litigation" in text.lower()


def test_sec_recent_filings_keeps_body_metadata_when_enabled(tmp_path, monkeypatch) -> None:
    client = sec_module.SecEdgarClient(cache_dir=tmp_path / "sec_unit")
    submissions = {
        "filings": {
            "recent": {
                "accessionNumber": ["0000000000-26-000001"],
                "filingDate": ["2026-01-02"],
                "reportDate": ["2026-01-01"],
                "form": ["8-K"],
                "primaryDocument": ["doc.htm"],
            }
        }
    }

    monkeypatch.setattr(client, "_get_json", lambda *args, **kwargs: submissions)
    monkeypatch.setattr(
        client,
        "fetch_filing_document",
        lambda cik, accession_number, primary_document: f"document({cik}|{accession_number}|{primary_document})",
    )

    filings = client.recent_filings("320193", {"8-K"}, include_document_text=True)

    assert filings["cik"].iloc[0] == "0000320193"
    assert filings["accession_number_no_dash"].iloc[0] == "000000000026000001"
    assert filings["document_text"].iloc[0] == "document(0000320193|0000000000-26-000001|doc.htm)"
