from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
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
    content: str = ""
    full_text: str = ""
    body_text: str = ""


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
                        content=text,
                        full_text="",
                        body_text="",
                    )
                )
        return items


@dataclass
class YFinanceNewsProvider:
    max_items_per_ticker: int = 50
    timeout_seconds: float = 6.0
    max_full_text_chars: int = 12000

    def get_news(self, tickers: list[str], start: str | date, end: str | date) -> list[NewsItem]:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise RuntimeError("yfinance is required for live ticker news") from exc

        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end)
        items: list[NewsItem] = []
        seen: set[tuple[str, str, str]] = set()
        for ticker in tickers:
            for raw in (yf.Ticker(ticker).news or [])[: self.max_items_per_ticker]:
                content = raw.get("content", raw)
                title = content.get("title") or raw.get("title") or ""
                provider = content.get("provider", {}) if isinstance(content.get("provider"), dict) else {}
                source = provider.get("displayName") or raw.get("publisher") or "yfinance"
                published_raw = content.get("pubDate") or raw.get("providerPublishTime") or end_ts
                published = _parse_news_timestamp(published_raw)
                if pd.isna(published) or published < start_ts or published > end_ts.normalize():
                    continue
                url = content.get("canonicalUrl", {}).get("url", raw.get("link", ""))
                key = (ticker, title, url)
                if key in seen:
                    continue
                seen.add(key)
                full_text = _fetch_full_text(url, self.timeout_seconds, self.max_full_text_chars)
                content_text = str(content.get("summary", "")).strip()
                if not content_text:
                    content_text = str(title).strip()
                items.append(
                    NewsItem(
                        ticker=ticker,
                        published_at=published,
                        title=title,
                        source=source,
                        url=url,
                        summary=content_text,
                        content=content_text if not full_text else full_text,
                        full_text=full_text,
                        body_text=full_text,
                    )
                )
        return items


@dataclass
class GDELTNewsProvider:
    timeout_seconds: float = 10.0
    max_records: int = 50
    max_full_text_chars: int = 12000

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
                article_title = article.get("title", "")
                article_url = article.get("url", "")
                full_text = _fetch_full_text(article_url, self.timeout_seconds, self.max_full_text_chars)
                summary = str(article_title)
                items.append(
                    NewsItem(
                        ticker=ticker,
                        published_at=_parse_news_timestamp(article.get("seendate", end_ts)),
                        title=article_title,
                        source=article.get("sourceCommonName", "gdelt"),
                        url=article_url,
                        summary=summary,
                        content=summary if not full_text else full_text,
                        full_text=full_text,
                        body_text=full_text,
                    )
                )
        return items


def _parse_news_timestamp(value: object) -> pd.Timestamp:
    if isinstance(value, int | float):
        return pd.to_datetime(value, unit="s").tz_localize(None).normalize()
    return pd.to_datetime(value, errors="coerce").tz_localize(None).normalize()


def _fetch_full_text(url: str, timeout_seconds: float, max_chars: int) -> str:
    if not url or max_chars <= 0:
        return ""
    try:
        response = requests.get(
            url,
            timeout=timeout_seconds,
            headers={"User-Agent": "QuantResearchNews/1.0"},
        )
        response.raise_for_status()
        raw = getattr(response, "text", "")
        if not raw:
            return ""
        return _extract_text(raw, max_chars)
    except Exception:
        return ""


def _extract_text(html_text: str, max_chars: int) -> str:
    try:
        parser = _NewsArticleExtractor()
        parser.feed(html_text)
        parser.close()

        candidates = [
            _extract_from_jsonld_payloads(parser.jsonld_payloads),
            parser.article_text,
            parser.main_text,
            parser.body_text,
            parser.full_text,
            parser.meta_description,
        ]
        text = ""
        for candidate in candidates:
            if candidate and _is_candidate_text(candidate, min_len=24):
                text = candidate
                break
    except Exception:
        text = ""
    if not text:
        text = _extract_text_via_regex(html_text)
    if not text:
        return ""
    return _truncate_text(_normalize_space(text), max_chars)


def _extract_from_jsonld_payloads(raw_payloads: list[str]) -> str:
    parsed_payloads = [
        parsed
        for payload in raw_payloads
        if (parsed := _parse_jsonld_payload(payload)) is not None
    ]
    for key in ("articlebody", "articleBody", "description", "headline"):
        text_candidates: list[str] = []
        for parsed in parsed_payloads:
            text_candidates.extend(_walk_jsonld_text_for_key(parsed, key))
        selected = _select_best_candidate(text_candidates)
        if selected:
            return selected
    return ""


def _parse_jsonld_payload(payload: str) -> object | None:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _walk_jsonld_text_for_key(node: object, target_key: str) -> list[str]:
    if isinstance(node, dict):
        text_candidates: list[str] = []
        for key, value in node.items():
            if str(key).lower() == target_key.lower() and isinstance(value, str):
                text_candidates.append(value.strip())
                continue
            if isinstance(value, dict | list):
                text_candidates.extend(_walk_jsonld_text_for_key(value, target_key))
        return text_candidates
    if isinstance(node, list):
        text_candidates = []
        for item in node:
            text_candidates.extend(_walk_jsonld_text_for_key(item, target_key))
        return text_candidates
    return []


def _select_best_candidate(candidates: list[str]) -> str:
    cleaned = [candidate.strip() for candidate in candidates if isinstance(candidate, str)]
    cleaned = [candidate for candidate in cleaned if _is_candidate_text(candidate, min_len=24)]
    if not cleaned:
        return ""
    return max(cleaned, key=len)


def _is_candidate_text(value: str, min_len: int) -> bool:
    normalized = _normalize_space(value)
    if len(normalized) < min_len:
        return False
    if len(set(normalized)) <= 1:
        return False
    return True


def _truncate_text(value: str, max_chars: int) -> str:
    if max_chars <= 0:
        return value
    if len(value) <= max_chars:
        return value
    truncated = value[:max_chars]
    boundary = truncated.rfind(" ")
    if boundary > 0:
        truncated = truncated[:boundary]
    return truncated.rstrip()


class _NewsArticleExtractor(HTMLParser):
    _SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas"}
    _NOISE_TAGS = {"header", "footer", "nav", "aside"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_stack: list[str] = []
        self._noise_depth = 0
        self._article_depth = 0
        self._main_depth = 0
        self._body_hint_depth = 0
        self._jsonld_depth = 0
        self._jsonld_chunks: list[str] = []
        self.jsonld_payloads: list[str] = []
        self._meta_values: dict[str, str] = {}
        self._all_text_chunks: list[str] = []
        self._article_text_chunks: list[str] = []
        self._main_text_chunks: list[str] = []
        self._body_text_chunks: list[str] = []

    @property
    def full_text(self) -> str:
        return " ".join(self._all_text_chunks)

    @property
    def article_text(self) -> str:
        return " ".join(self._article_text_chunks)

    @property
    def main_text(self) -> str:
        return " ".join(self._main_text_chunks)

    @property
    def body_text(self) -> str:
        return " ".join(self._body_text_chunks)

    @property
    def meta_description(self) -> str:
        for key in ("og:description", "twitter:description", "description", "dc.description", "itemprop:description"):
            value = self._meta_values.get(key)
            if value:
                return value
        return ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        attr_map = {k.lower(): (v or "").strip() for k, v in attrs}

        if tag_lower == "meta":
            self._capture_meta(attr_map)

        if tag_lower == "script" and self._is_jsonld_script(attr_map):
            self._jsonld_depth += 1
            self._jsonld_chunks.append("")
            return

        if tag_lower in self._SKIP_TAGS:
            self._skip_stack.append(tag_lower)
            return

        if tag_lower in self._NOISE_TAGS:
            self._noise_depth += 1

        if tag_lower == "article":
            self._article_depth += 1
        if tag_lower == "main":
            self._main_depth += 1
        if tag_lower in {"article", "main"} or self._is_body_container(attr_map):
            self._body_hint_depth += 1

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()

        if self._jsonld_depth and tag_lower == "script":
            payload = self._jsonld_chunks.pop()
            if payload.strip():
                self.jsonld_payloads.append(payload.strip())
            self._jsonld_depth -= 1
            return

        if self._skip_stack and self._skip_stack[-1] == tag_lower:
            self._skip_stack.pop()
            return

        if tag_lower == "header" or tag_lower == "footer" or tag_lower == "nav" or tag_lower == "aside":
            self._noise_depth = max(0, self._noise_depth - 1)
        if tag_lower == "article":
            self._article_depth = max(0, self._article_depth - 1)
        if tag_lower == "main":
            self._main_depth = max(0, self._main_depth - 1)
        if self._body_hint_depth and (tag_lower in {"article", "main", "div", "section", "p"}):
            self._body_hint_depth = max(0, self._body_hint_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._jsonld_depth:
            self._jsonld_chunks[-1] += data
            return
        if self._skip_stack or self._noise_depth > 0:
            return
        clean = data.strip()
        if not clean:
            return

        self._all_text_chunks.append(clean)
        if self._article_depth > 0:
            self._article_text_chunks.append(clean)
        if self._main_depth > 0:
            self._main_text_chunks.append(clean)
        if self._body_hint_depth > 0:
            self._body_text_chunks.append(clean)

    def _capture_meta(self, attrs: dict[str, str]) -> None:
        meta_name = attrs.get("name", "").strip().lower()
        meta_property = attrs.get("property", "").strip().lower()
        meta_itemprop = attrs.get("itemprop", "").strip().lower()
        key = meta_property or meta_name or meta_itemprop
        if not key:
            return
        content = attrs.get("content", "").strip()
        if not content:
            return
        if key in {"og:description", "twitter:description", "description", "dc.description", "itemprop:description"}:
            self._meta_values.setdefault(key, content)

    def _is_jsonld_script(self, attrs: dict[str, str]) -> bool:
        return attrs.get("type", "").lower() == "application/ld+json"

    def _is_body_container(self, attrs: dict[str, str]) -> bool:
        combined = " ".join(attrs.get(key, "").lower() for key in ("class", "id")).strip()
        if not combined:
            return False
        deny = ("menu", "header", "footer", "sidebar", "nav", "widget", "cookie", "related", "social", "share", "recommend")
        if any(token in combined for token in deny):
            return False
        include = (
            "article",
            "story",
            "post",
            "entry",
            "content",
            "text",
            "body",
            "news",
            "headline",
            "story-body",
            "article-body",
            "post-content",
            "main-content",
            "article_content",
        )
        return any(token in combined for token in include)


def _extract_text_via_regex(html_text: str) -> str:
    text = re.sub(
        r"(?is)<(script|style|noscript|svg|canvas).*?>.*?</\1>",
        " ",
        html_text,
    )
    text = re.sub(r"(?s)<!--.*?-->", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def _normalize_space(value: str) -> str:
    return " ".join(value.split())
