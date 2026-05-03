from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol
from urllib.parse import quote_plus

import pandas as pd
import requests


@dataclass(frozen=True)
class NewsItem:
    ticker: str
    published_at: pd.Timestamp
    title: str
    source: str
    url: str = ""
    summary: str = ""


class NewsProvider(Protocol):
    def get_news(self, tickers: list[str], start: str | date, end: str | date) -> list[NewsItem]:
        """Return news items keyed by ticker."""


@dataclass
class SyntheticNewsProvider:
    seed_texts: tuple[str, ...] = (
        "beats earnings expectations as revenue growth accelerates",
        "faces margin pressure and regulatory risk after filing update",
        "announces product launch with strong demand outlook",
        "warns about macro uncertainty and softer guidance",
        "analysts highlight resilient cash flow and buyback capacity",
    )

    def get_news(self, tickers: list[str], start: str | date, end: str | date) -> list[NewsItem]:
        dates = pd.bdate_range(start=pd.Timestamp(start), end=pd.Timestamp(end))
        if len(dates) == 0:
            return []
        items: list[NewsItem] = []
        for ticker_idx, ticker in enumerate(tickers):
            for offset, day in enumerate(dates[::10]):
                text = self.seed_texts[(ticker_idx + offset) % len(self.seed_texts)]
                items.append(
                    NewsItem(
                        ticker=ticker,
                        published_at=pd.Timestamp(day),
                        title=f"{ticker} {text}",
                        source="synthetic",
                        url=f"synthetic://{ticker}/{day.date()}",
                        summary=text,
                    )
                )
        return items


@dataclass
class YFinanceNewsProvider:
    max_items_per_ticker: int = 50

    def get_news(self, tickers: list[str], start: str | date, end: str | date) -> list[NewsItem]:
        del start
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise RuntimeError("yfinance is required for live ticker news") from exc

        end_ts = pd.Timestamp(end)
        items: list[NewsItem] = []
        for ticker in tickers:
            for raw in (yf.Ticker(ticker).news or [])[: self.max_items_per_ticker]:
                content = raw.get("content", raw)
                title = content.get("title") or raw.get("title") or ""
                provider = content.get("provider", {}) if isinstance(content.get("provider"), dict) else {}
                source = provider.get("displayName") or raw.get("publisher") or "yfinance"
                published_raw = content.get("pubDate") or raw.get("providerPublishTime") or end_ts
                published = _parse_news_timestamp(published_raw)
                items.append(
                    NewsItem(
                        ticker=ticker,
                        published_at=published,
                        title=title,
                        source=source,
                        url=content.get("canonicalUrl", {}).get("url", raw.get("link", "")),
                        summary=content.get("summary", ""),
                    )
                )
        return items


@dataclass
class GDELTNewsProvider:
    timeout_seconds: float = 10.0
    max_records: int = 50

    def get_news(self, tickers: list[str], start: str | date, end: str | date) -> list[NewsItem]:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        items: list[NewsItem] = []
        for ticker in tickers:
            query = quote_plus(f'"{ticker}" stock OR earnings OR shares')
            url = (
                "https://api.gdeltproject.org/api/v2/doc/doc"
                f"?query={query}&mode=artlist&format=json&maxrecords={self.max_records}"
                f"&startdatetime={start_ts:%Y%m%d%H%M%S}&enddatetime={end_ts:%Y%m%d%H%M%S}"
            )
            response = requests.get(url, timeout=self.timeout_seconds)
            response.raise_for_status()
            for article in response.json().get("articles", []):
                items.append(
                    NewsItem(
                        ticker=ticker,
                        published_at=_parse_news_timestamp(article.get("seendate", end_ts)),
                        title=article.get("title", ""),
                        source=article.get("sourceCommonName", "gdelt"),
                        url=article.get("url", ""),
                        summary=article.get("title", ""),
                    )
                )
        return items


def _parse_news_timestamp(value: object) -> pd.Timestamp:
    if isinstance(value, int | float):
        return pd.to_datetime(value, unit="s").tz_localize(None).normalize()
    return pd.to_datetime(value, errors="coerce").tz_localize(None).normalize()
