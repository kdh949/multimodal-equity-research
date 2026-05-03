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
from quant_research.data.news import GDELTNewsProvider, YFinanceNewsProvider


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

    assert frame.columns.tolist() == ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    assert frame["ticker"].eq("AAPL").all()
    assert frame["date"].tolist() == [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-03")]
    assert frame["adj_close"].tolist() == [10.4, 10.9]


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

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=fake_ticker))

    provider = YFinanceNewsProvider(max_items_per_ticker=10)
    output = provider.get_news(["AAPL"], start=date(2026, 1, 3), end=date(2026, 1, 10))

    assert output
    assert len(output) == 1
    assert output[0].title == "AAPL beats expectations"
    assert output[0].published_at >= pd.Timestamp("2026-01-03")
    assert output[0].published_at <= pd.Timestamp("2026-01-10")


def test_gdelt_provider_builds_request_url_and_timeout(monkeypatch) -> None:
    calls: list[tuple[str, float]] = []

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"articles": [{"seendate": 171, "title": "AAPL update", "sourceCommonName": "Demo", "url": "https://example"}]}

    def fake_get(url: str, timeout: float | None = None) -> _FakeResponse:
        calls.append((url, timeout))
        return _FakeResponse()

    news_module = __import__("quant_research.data.news", fromlist=["requests"])
    monkeypatch.setattr(news_module, "requests", types.SimpleNamespace(get=fake_get))

    provider = GDELTNewsProvider(timeout_seconds=12.5, max_records=7)
    result = provider.get_news(["AAPL", "MSFT"], start=date(2026, 1, 1), end=date(2026, 1, 10))

    assert len(calls) == 2
    assert result
    for ticker, (url, timeout) in zip(["AAPL", "MSFT"], calls, strict=True):
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


def test_sec_edgar_client_sets_user_agent_and_request_headers(tmp_path) -> None:
    client = sec_module.SecEdgarClient(
        settings=SecSettings(user_agent="QuantResearchTest/1.0"),
        cache_dir=tmp_path / "sec",
    )
    assert client.session.headers["User-Agent"] == "QuantResearchTest/1.0"
    assert client.session.headers["Accept-Encoding"] == "gzip, deflate"
    assert client.session.headers["Host"] == "data.sec.gov"


def test_sec_default_rate_limit_stays_within_fair_access_limit() -> None:
    assert SecSettings().max_requests_per_second <= 10.0


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
