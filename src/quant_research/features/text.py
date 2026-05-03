from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from quant_research.data.news import NewsItem

POSITIVE_WORDS = {
    "beat",
    "beats",
    "growth",
    "accelerates",
    "strong",
    "resilient",
    "launch",
    "demand",
    "buyback",
    "profit",
    "upgrade",
}
NEGATIVE_WORDS = {
    "risk",
    "pressure",
    "regulatory",
    "warns",
    "uncertainty",
    "softer",
    "miss",
    "downgrade",
    "lawsuit",
    "investigation",
}
EVENT_KEYWORDS = {
    "earnings": "earnings",
    "guidance": "guidance",
    "regulatory": "regulatory",
    "lawsuit": "legal",
    "investigation": "legal",
    "merger": "mna",
    "acquisition": "mna",
    "buyback": "capital_return",
}


@dataclass
class KeywordSentimentAnalyzer:
    def score(self, text: str) -> dict[str, float | str | bool]:
        tokens = set(re.findall(r"[a-zA-Z]+", text.lower()))
        pos = len(tokens.intersection(POSITIVE_WORDS))
        neg = len(tokens.intersection(NEGATIVE_WORDS))
        raw = pos - neg
        sentiment = math.tanh(raw / 3)
        label = "positive" if sentiment > 0.15 else "negative" if sentiment < -0.15 else "neutral"
        confidence = min(1.0, 0.45 + abs(sentiment))
        event_tags = sorted({tag for keyword, tag in EVENT_KEYWORDS.items() if keyword in tokens})
        return {
            "sentiment_score": sentiment,
            "negative_flag": sentiment < -0.15,
            "label": label,
            "confidence": confidence,
            "event_tag": ",".join(event_tags),
            "risk_flag": bool(tokens.intersection(NEGATIVE_WORDS)),
        }


def build_news_features(news_items: list[NewsItem], analyzer: KeywordSentimentAnalyzer | None = None) -> pd.DataFrame:
    analyzer = analyzer or KeywordSentimentAnalyzer()
    rows: list[dict[str, object]] = []
    for item in news_items:
        text = f"{item.title}. {item.summary}".strip()
        scored = analyzer.score(text)
        rows.append(
            {
                "date": pd.Timestamp(item.published_at).normalize(),
                "ticker": item.ticker,
                "source": item.source,
                "sentiment_score": float(scored["sentiment_score"]),
                "negative_flag": bool(scored["negative_flag"]),
                "event_tag": str(scored["event_tag"]),
                "risk_flag": bool(scored["risk_flag"]),
                "confidence": float(scored["confidence"]),
            }
        )
    if not rows:
        return _empty_news_features()

    frame = pd.DataFrame(rows)
    aggregations = []
    for (date, ticker), group in frame.groupby(["date", "ticker"]):
        event_counter = Counter(tag for tags in group["event_tag"] for tag in str(tags).split(",") if tag)
        aggregations.append(
            {
                "date": date,
                "ticker": ticker,
                "news_article_count": len(group),
                "news_sentiment_mean": group["sentiment_score"].mean(),
                "news_negative_ratio": group["negative_flag"].mean(),
                "news_source_count": group["source"].nunique(),
                "news_event_count": sum(event_counter.values()),
                "news_top_event": event_counter.most_common(1)[0][0] if event_counter else "none",
                "text_risk_score": group["risk_flag"].mean(),
                "news_confidence_mean": group["confidence"].mean(),
            }
        )
    return pd.DataFrame(aggregations).sort_values(["date", "ticker"]).reset_index(drop=True)


def expand_news_features_to_calendar(news_features: pd.DataFrame, calendar: pd.DataFrame, decay: float = 0.85) -> pd.DataFrame:
    if news_features.empty:
        expanded = calendar[["date", "ticker"]].drop_duplicates().copy()
        for column in _empty_news_features().columns:
            if column not in {"date", "ticker"}:
                expanded[column] = 0.0 if column != "news_top_event" else "none"
        return expanded

    base = calendar[["date", "ticker"]].drop_duplicates().sort_values(["ticker", "date"])
    merged = base.merge(news_features, on=["date", "ticker"], how="left")
    numeric_cols = [
        "news_article_count",
        "news_sentiment_mean",
        "news_negative_ratio",
        "news_source_count",
        "news_event_count",
        "text_risk_score",
        "news_confidence_mean",
    ]
    merged[numeric_cols] = merged[numeric_cols].fillna(0.0)
    merged["news_top_event"] = merged["news_top_event"].fillna("none")
    merged["news_recency_decay"] = merged.groupby("ticker")["news_article_count"].transform(
        lambda series: series.ewm(alpha=1 - decay, adjust=False).mean()
    )
    return merged


def _empty_news_features() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "ticker",
            "news_article_count",
            "news_sentiment_mean",
            "news_negative_ratio",
            "news_source_count",
            "news_event_count",
            "news_top_event",
            "text_risk_score",
            "news_confidence_mean",
        ]
    )
