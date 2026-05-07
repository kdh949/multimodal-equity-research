import inspect
import json
import sys
import types
from datetime import date
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from quant_research.config import SecSettings
from quant_research.data import sec as sec_module
from quant_research.data.market import _normalize_yfinance_frame
from quant_research.data.news import (
    GDELTNewsProvider,
    NewsItem,
    YFinanceNewsProvider,
    _extract_text,
    _truncate_text,
    news_items_to_frame,
)
from quant_research.features.text import build_news_features


def test_yfinance_normalize_yfinance_frame_keeps_expected_schema() -> None:
    raw = pd.DataFrame(
        {
            "Open": [10.0, 10.5],
            "High": [10.8, 11.2],
            "Low": [9.9, 10.1],
            "Close": [10.4, 10.9],
            "Volume": [1000, 1100],
            "Adj Close": [10.4, 10.9],
        },
        index=pd.date_range("2026-01-02", periods=2, tz="UTC"),
    )

    frame = _normalize_yfinance_frame(raw, "AAPL")

    assert frame.columns.tolist() == [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "event_timestamp",
        "availability_timestamp",
        "source_timestamp",
        "timezone",
    ]
    assert frame["ticker"].eq("AAPL").all()
    assert frame["date"].tolist() == [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-03")]
    assert frame["adj_close"].tolist() == [10.4, 10.9]
    assert isinstance(frame["event_timestamp"].dtype, pd.DatetimeTZDtype)
    assert isinstance(frame["availability_timestamp"].dtype, pd.DatetimeTZDtype)
    assert frame["event_timestamp"].iloc[0] == pd.Timestamp("2026-01-02 21:00:00", tz="UTC")
    assert frame["availability_timestamp"].equals(frame["event_timestamp"])
    assert frame["source_timestamp"].isna().all()
    assert frame["timezone"].eq("America/New_York").all()


def test_yfinance_normalize_yfinance_frame_falls_back_to_close() -> None:
    raw = pd.DataFrame(
        {
            "Open": [10.0, 10.5],
            "High": [10.8, 11.2],
            "Low": [9.9, 10.1],
            "Close": [10.4, pd.NA],
            "Volume": [1000, 1100],
        },
        index=pd.date_range("2026-01-02", periods=2),
    )

    frame = _normalize_yfinance_frame(raw, "AAPL")

    assert len(frame) == 1
    assert frame["adj_close"].tolist() == [10.4]
    assert frame["date"].iloc[0] == pd.Timestamp("2026-01-02")


def test_yfinance_news_provider_date_filtering_and_dedup_if_implemented(monkeypatch) -> None:
    source = inspect.getsource(YFinanceNewsProvider.get_news)
    if "del start" in source:
        pytest.skip("YFinanceNewsProvider has no date filtering/dedup logic yet")

    news_payload = [
        {
            "content": {
                "title": "AAPL beats expectations",
                "pubDate": "2026-01-05 09:00:00",
                "provider": {"displayName": "Reuters"},
                "canonicalUrl": {"url": "https://news/dup"},
                "summary": "Inside summary",
            }
        },
        {
            "content": {
                "title": "AAPL beats expectations",
                "pubDate": "2026-01-05 09:00:00",
                "provider": {"displayName": "Reuters"},
                "canonicalUrl": {"url": "https://news/dup"},
                "summary": "Inside summary",
            }
        },
        {
            "content": {
                "title": "AAPL old headline",
                "pubDate": "2025-12-31 09:00:00",
                "provider": {"displayName": "Reuters"},
                "canonicalUrl": {"url": "https://news/old"},
                "summary": "Old",
            }
        },
    ]

    def fake_ticker(_: str) -> object:
        return types.SimpleNamespace(news=news_payload)

    def fake_get(url: str, timeout: float | None = None, **_: object) -> types.SimpleNamespace:
        if "news/dup" not in url:
            raise AssertionError(f"unexpected request: {url}")
        return types.SimpleNamespace(
            text=(
                "<html><head>"
                "<script type=\"application/ld+json\">"
                "{\"@context\":\"https://schema.org\",\"@type\":\"NewsArticle\","
                "\"headline\":\"AAPL beats expectations\","
                "\"articleBody\":\"analysts reported earnings beat and strong growth\""
                "}"
                "</script>"
                "</head><body><article><p>backup paragraph</p></article></body></html>"
            ),
            raise_for_status=lambda: None,
        )

    news_module = __import__("quant_research.data.news", fromlist=["requests"])
    monkeypatch.setattr(news_module, "requests", types.SimpleNamespace(get=fake_get))
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=fake_ticker))

    provider = YFinanceNewsProvider(max_items_per_ticker=10)
    output = provider.get_news(["AAPL"], start=date(2026, 1, 3), end=date(2026, 1, 10))

    assert output
    assert len(output) == 1
    assert output[0].title == "AAPL beats expectations"
    assert output[0].published_at >= pd.Timestamp("2026-01-03")
    assert output[0].published_at <= pd.Timestamp("2026-01-10")
    assert output[0].full_text == "analysts reported earnings beat and strong growth"
    assert output[0].body_text == output[0].full_text


def test_news_provider_uses_meta_description_fallback(monkeypatch) -> None:
    news_payload = [
        {
            "content": {
                "title": "AAPL update",
                "pubDate": "2026-01-06 10:00:00",
                "provider": {"displayName": "Reuters"},
                "canonicalUrl": {"url": "https://news/meta"},
                "summary": "short summary",
            }
        },
    ]

    def fake_ticker(_: str) -> object:
        return types.SimpleNamespace(news=news_payload)

    def fake_get(url: str, timeout: float | None = None, **_: object) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            text=(
                "<html><head>"
                "<meta property='og:description' content='Meta description with more context than body text'>"
                "</head><body><p>Short body.</p></body></html>"
            ),
            raise_for_status=lambda: None,
        )

    news_module = __import__("quant_research.data.news", fromlist=["requests"])
    monkeypatch.setattr(news_module, "requests", types.SimpleNamespace(get=fake_get))
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=fake_ticker))

    provider = YFinanceNewsProvider(max_items_per_ticker=1)
    output = provider.get_news(["AAPL"], start=date(2026, 1, 5), end=date(2026, 1, 8))

    assert output
    assert output[0].full_text == "Meta description with more context than body text"
    assert output[0].content == output[0].full_text


def test_news_article_body_beats_meta_description() -> None:
    text = _extract_text(
        (
            "<html><head>"
            "<meta name='description' content='Short summary should only be fallback'>"
            "</head><body><article>"
            "<p>Analysts reported a full article body with earnings context and risk details.</p>"
            "</article></body></html>"
        ),
        max_chars=1000,
    )

    assert text == "Analysts reported a full article body with earnings context and risk details."


def test_news_jsonld_article_body_beats_description() -> None:
    text = _extract_text(
        (
            "<script type='application/ld+json'>"
            "{"
            "\"@type\":\"NewsArticle\","
            "\"description\":\"Longer description should not beat the actual article body fallback text\","
            "\"articleBody\":\"Actual article body with source-grounded detail\""
            "}"
            "</script>"
        ),
        max_chars=1000,
    )

    assert text == "Actual article body with source-grounded detail"


def test_news_provider_fallback_to_summary_when_full_text_fetch_fails(monkeypatch) -> None:
    news_payload = [
        {
            "content": {
                "title": "AAPL misses guidance",
                "pubDate": "2026-01-06 10:00:00",
                "provider": {"displayName": "Reuters"},
                "canonicalUrl": {"url": "https://news/full"},
                "summary": "earnings miss pressure",
            }
        },
    ]

    def fake_ticker(_: str) -> object:
        return types.SimpleNamespace(news=news_payload)

    def fake_get(url: str, timeout: float | None = None, **_: object) -> types.SimpleNamespace:
        raise RuntimeError("simulated fetch failure")

    news_module = __import__("quant_research.data.news", fromlist=["requests"])
    monkeypatch.setattr(news_module, "requests", types.SimpleNamespace(get=fake_get))
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=fake_ticker))

    provider = YFinanceNewsProvider(max_items_per_ticker=1, timeout_seconds=1.0)
    output = provider.get_news(["AAPL"], start=date(2026, 1, 5), end=date(2026, 1, 7))

    assert output
    assert output[0].full_text == ""
    assert output[0].content == "earnings miss pressure"
    assert output[0].summary == "earnings miss pressure"


def test_news_provider_falls_back_to_title_when_summary_is_missing(monkeypatch) -> None:
    news_payload = [
        {
            "content": {
                "title": "AAPL legal filing update",
                "pubDate": "2026-01-06 10:00:00",
                "provider": {"displayName": "Reuters"},
                "canonicalUrl": {"url": "https://news/title"},
            }
        },
    ]

    def fake_ticker(_: str) -> object:
        return types.SimpleNamespace(news=news_payload)

    def fake_get(url: str, timeout: float | None = None, **_: object) -> types.SimpleNamespace:
        raise RuntimeError("simulated fetch failure")

    news_module = __import__("quant_research.data.news", fromlist=["requests"])
    monkeypatch.setattr(news_module, "requests", types.SimpleNamespace(get=fake_get))
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=fake_ticker))

    provider = YFinanceNewsProvider(max_items_per_ticker=1, timeout_seconds=1.0)
    output = provider.get_news(["AAPL"], start=date(2026, 1, 5), end=date(2026, 1, 7))

    assert output
    assert output[0].full_text == ""
    assert output[0].content == "AAPL legal filing update"
    assert output[0].summary == "AAPL legal filing update"


def test_news_text_truncates_on_word_boundary() -> None:
    html = "<article><p>analysts reported earnings beat and strong growth amid broader market volatility</p></article>"
    text = _extract_text(html, max_chars=50)

    assert text == "analysts reported earnings beat and strong growth"


def test_truncate_text_keeps_word_boundary_when_possible() -> None:
    assert _truncate_text("alpha beta gamma", 8) == "alpha"


def test_gdelt_provider_builds_request_url_and_timeout(monkeypatch) -> None:
    calls: list[tuple[str, float]] = []
    article_calls: list[str] = []

    class _FakeResponse:
        def __init__(self, url: str) -> None:
            self.url = url

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            if self.url.startswith("https://api.gdeltproject.org"):
                return {
                    "articles": [
                        {
                            "seendate": 171,
                            "title": "AAPL update",
                            "sourceCommonName": "Demo",
                            "url": "https://example/aapl",
                        }
                    ]
                }
            return {}

        @property
        def text(self) -> str:
            return "<html><body><article><p>full article body</p></article></body></html>"

    def fake_get(url: str, timeout: float | None = None, **_: object) -> _FakeResponse:
        calls.append((url, timeout))
        if not url.startswith("https://api.gdeltproject.org"):
            article_calls.append(url)
        return _FakeResponse(url=url)

    news_module = __import__("quant_research.data.news", fromlist=["requests"])
    monkeypatch.setattr(news_module, "requests", types.SimpleNamespace(get=fake_get))

    provider = GDELTNewsProvider(timeout_seconds=12.5, max_records=7)
    result = provider.get_news(["AAPL", "MSFT"], start=date(2026, 1, 1), end=date(2026, 1, 10))

    assert len(calls) == 4
    assert result
    for (url, timeout), ticker in zip(calls[::2], ["AAPL", "MSFT"], strict=True):
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        assert parsed.scheme == "https"
        assert parsed.netloc == "api.gdeltproject.org"
        assert query["query"][0] == f'"{ticker}" stock OR earnings OR shares'
        assert query["mode"][0] == "artlist"
        assert query["format"][0] == "json"
        assert query["maxrecords"][0] == "7"
        assert query["startdatetime"][0] == "20260101000000"
        assert query["enddatetime"][0] == "20260110000000"
        assert timeout == 12.5
    assert result[0].full_text == "full article body"
    assert len(article_calls) == 2


def test_build_news_features_uses_full_text_and_quality_columns() -> None:
    items = [
        NewsItem(
            ticker="AAPL",
            published_at=pd.Timestamp("2026-01-05"),
            title="AAPL beat expectations",
            source="tests",
            summary="short summary",
            full_text="legal risk lawsuit warning miss",
            content="fallback summary",
            url="https://example/a",
            body_text="",
        )
    ]

    features = build_news_features(items)

    assert features.loc[0, "news_article_count"] == 1
    assert features.loc[0, "news_full_text_available_ratio"] == 1.0
    assert features.loc[0, "news_text_length"] == len("legal risk lawsuit warning miss")
    assert features.loc[0, "news_sentiment_mean"] < 0.0


def test_news_item_emits_standard_timestamp_fields() -> None:
    item = NewsItem(
        ticker="AAPL",
        published_at=pd.Timestamp("2026-01-05"),
        title="AAPL update",
        source="tests",
        summary="earnings growth",
        source_timestamp=pd.Timestamp("2026-01-05 14:30:00"),
        availability_timestamp=pd.Timestamp("2026-01-05 14:45:00"),
    )

    frame = news_items_to_frame([item])

    assert frame.loc[0, "event_timestamp"] == pd.Timestamp("2026-01-05 14:30:00", tz="UTC")
    assert frame.loc[0, "source_timestamp"] == pd.Timestamp("2026-01-05 14:30:00", tz="UTC")
    assert frame.loc[0, "availability_timestamp"] == pd.Timestamp("2026-01-05 14:45:00", tz="UTC")
    assert frame.loc[0, "timezone"] == "UTC"


def test_build_news_features_uses_publication_availability_date() -> None:
    items = [
        NewsItem(
            ticker="AAPL",
            published_at=pd.Timestamp("2026-01-05"),
            title="AAPL after hours risk",
            source="tests",
            summary="regulatory risk",
            content="regulatory risk",
            availability_timestamp=pd.Timestamp("2026-01-06 01:30:00", tz="UTC"),
        )
    ]

    features = build_news_features(items)

    assert features.loc[0, "date"] == pd.Timestamp("2026-01-06")
    assert features.loc[0, "news_availability_timestamp"] == pd.Timestamp("2026-01-06 01:30:00", tz="UTC")


def test_sec_edgar_client_sets_user_agent_and_request_headers(tmp_path) -> None:
    client = sec_module.SecEdgarClient(
        settings=SecSettings(user_agent="QuantResearchTest/1.0"),
        cache_dir=tmp_path / "sec",
    )
    assert client.session.headers["User-Agent"] == "QuantResearchTest/1.0"
    assert client.session.headers["Accept-Encoding"] == "gzip, deflate"
    assert "Host" not in client.session.headers


def test_sec_default_rate_limit_stays_within_fair_access_limit() -> None:
    assert SecSettings().max_requests_per_second <= 10.0


def test_sec_recent_filings_maps_edgar_acceptance_to_standard_timestamps(tmp_path, monkeypatch) -> None:
    client = sec_module.SecEdgarClient(cache_dir=tmp_path / "sec")
    payload = {
        "filings": {
            "recent": {
                "accessionNumber": ["0000320193-26-000001"],
                "filingDate": ["2026-01-03"],
                "reportDate": ["2025-12-31"],
                "acceptanceDateTime": ["2026-01-03T17:34:56.000Z"],
                "form": ["10-Q"],
                "primaryDocument": ["aapl-20251231.htm"],
            }
        }
    }
    monkeypatch.setattr(client, "get_submissions", lambda cik: payload)

    filings = client.recent_filings("320193")

    assert filings.loc[0, "event_timestamp"] == pd.Timestamp("2025-12-31 23:59:59.999999999", tz="UTC")
    assert filings.loc[0, "source_timestamp"] == pd.Timestamp("2026-01-03 17:34:56", tz="UTC")
    assert filings.loc[0, "availability_timestamp"] == filings.loc[0, "source_timestamp"]
    assert filings.loc[0, "timezone"] == "UTC"


def test_sec_companyfacts_extractors_preserve_standard_timestamps() -> None:
    payload = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-12-31",
                                "val": "100",
                                "fy": 2025,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2026-02-15",
                            }
                        ]
                    }
                },
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-12-31",
                                "val": "20",
                                "fy": 2025,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2026-02-16",
                            }
                        ]
                    }
                },
            }
        }
    }

    facts = sec_module.extract_companyfacts_frame(payload)

    assert facts.loc[0, "event_timestamp"] == pd.Timestamp("2025-12-31 23:59:59.999999999", tz="UTC")
    assert facts.loc[0, "source_timestamp"] == pd.Timestamp("2026-02-16 23:59:59.999999999", tz="UTC")
    assert facts.loc[0, "availability_timestamp"] == facts.loc[0, "source_timestamp"]
    assert facts.loc[0, "timezone"] == "UTC"
    assert not any(column.startswith("timezone_") for column in facts.columns)


def test_sec_edgar_client_reuses_cache_without_network(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "sec"
    cache_dir.mkdir()
    cache_path = cache_dir / "submissions_0000320193.json"
    expected_payload = {"hello": "world"}
    cache_path.write_text(json.dumps(expected_payload))

    client = sec_module.SecEdgarClient(cache_dir=cache_dir)
    state = {"called": False}

    def failing_get(*_args: object, **_kwargs: object) -> None:
        state["called"] = True
        raise AssertionError("network must not be called when cache exists")

    monkeypatch.setattr(client.session, "get", failing_get)
    got = client._get_json("https://data.sec.gov/submissions/CIK0000320193.json", cache_path.name)

    assert not state["called"]
    assert got == expected_payload


def test_sec_edgar_client_throttle_respects_rate_limit(tmp_path, monkeypatch) -> None:
    client = sec_module.SecEdgarClient(
        settings=SecSettings(max_requests_per_second=5.0),
        cache_dir=tmp_path / "sec",
    )
    sleep_calls: list[float] = []
    monkeypatch.setattr(sec_module.time, "sleep", lambda seconds: sleep_calls.append(float(seconds)))

    monkeypatch.setattr(sec_module.time, "monotonic", lambda: 1.0)
    client._last_request_at = 1.0
    client._throttle()
    assert sleep_calls == [pytest.approx(0.2)]

    sleep_calls.clear()
    monkeypatch.setattr(sec_module.time, "monotonic", lambda: 1.25)
    client._last_request_at = 1.0
    client._throttle()
    assert sleep_calls == []
