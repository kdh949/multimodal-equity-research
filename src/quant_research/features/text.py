from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from quant_research.data.news import NewsItem
from quant_research.data.timestamps import PRICE_TIMEZONE, UTC_TIMEZONE, timestamp_utc

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

RECENCY_ALPHA_DEFAULT = 0.15
RECENCY_WINDOWS = (5, 20)
NEWS_LAG_PERIODS_DEFAULT = 1
_US_MARKET_CLOSE_OFFSET = pd.Timedelta(hours=16)


@dataclass
class KeywordSentimentAnalyzer:
    def score(self, text: str) -> dict[str, float | str | bool]:
        tokens = re.findall(r"[a-zA-Z]+", text.lower())
        if not tokens:
            return {
                "sentiment_score": 0.0,
                "negative_flag": False,
                "label": "neutral",
                "confidence": 0.45,
                "event_tag": "",
                "risk_flag": False,
                "token_count": 0.0,
            }

        unique_tokens = set(tokens)
        pos = len(unique_tokens.intersection(POSITIVE_WORDS))
        neg = len(unique_tokens.intersection(NEGATIVE_WORDS))
        raw = pos - neg
        token_count = float(len(tokens))
        sentiment = math.tanh((raw / token_count) * 3.5)
        confidence = min(1.0, 0.45 + abs(sentiment) + min(0.4, token_count / 30.0))
        label = "positive" if sentiment > 0.12 else "negative" if sentiment < -0.12 else "neutral"
        event_tags = sorted({tag for keyword, tag in EVENT_KEYWORDS.items() if keyword in unique_tokens})
        return {
            "sentiment_score": sentiment,
            "negative_flag": sentiment < -0.15,
            "label": label,
            "confidence": confidence,
            "event_tag": ",".join(event_tags),
            "risk_flag": bool(unique_tokens.intersection(NEGATIVE_WORDS)),
            "token_count": token_count,
        }


def build_news_features(
    news_items: list[NewsItem],
    analyzer: KeywordSentimentAnalyzer | None = None,
) -> pd.DataFrame:
    analyzer = analyzer or KeywordSentimentAnalyzer()
    rows: list[dict[str, object]] = []
    for item in news_items:
        full_text = _full_text(item)
        scoring_text = full_text or f"{item.title}. {item.summary}".strip()
        scored = analyzer.score(scoring_text)
        rows.append(
            {
                "date": _feature_date(item.availability_timestamp),
                "ticker": item.ticker,
                "source": item.source,
                "event_timestamp": item.event_timestamp,
                "availability_timestamp": item.availability_timestamp,
                "source_timestamp": item.source_timestamp,
                "timezone": item.timezone or UTC_TIMEZONE,
                "sentiment_score": float(scored["sentiment_score"]),
                "negative_flag": bool(scored["negative_flag"]),
                "event_tag": str(scored["event_tag"]),
                "risk_flag": bool(scored["risk_flag"]),
                "confidence": float(scored["confidence"]),
                "news_token_count": _score_token_count(scored, scoring_text),
                "news_text_length": float(len(scoring_text)),
                "news_full_text_available": 1.0 if bool(full_text) else 0.0,
            }
        )

    if not rows:
        return _empty_news_features()

    frame = pd.DataFrame(rows)
    aggregations = []
    for (date, ticker), group in frame.groupby(["date", "ticker"]):
        event_counter = Counter(tag for tags in group["event_tag"] for tag in str(tags).split(",") if tag)
        confidence_sum = float(group["confidence"].sum())
        if confidence_sum > 0:
            weighted_sentiment = float((group["sentiment_score"] * group["confidence"]).sum() / confidence_sum)
        else:
            weighted_sentiment = float(group["sentiment_score"].mean())
        article_count = len(group)
        aggregations.append(
            {
                "date": date,
                "ticker": ticker,
                "news_event_timestamp": _timestamp_min(group["event_timestamp"]),
                "news_availability_timestamp": _timestamp_max(group["availability_timestamp"]),
                "news_source_timestamp": _timestamp_max(group["source_timestamp"]),
                "news_timezone": UTC_TIMEZONE,
                "news_article_count": article_count,
                "news_sentiment_mean": weighted_sentiment,
                "news_negative_ratio": group["negative_flag"].mean(),
                "news_source_count": group["source"].nunique(),
                "news_source_diversity": group["source"].nunique() / float(article_count),
                "news_event_count": sum(event_counter.values()),
                "news_top_event": event_counter.most_common(1)[0][0] if event_counter else "none",
                "text_risk_score": group["risk_flag"].mean(),
                "news_confidence_mean": group["confidence"].mean(),
                "news_token_count_mean": group["news_token_count"].mean(),
                "news_text_length": group["news_text_length"].mean(),
                "news_full_text_available_ratio": group["news_full_text_available"].mean(),
            }
        )
    return pd.DataFrame(aggregations).sort_values(["date", "ticker"]).reset_index(drop=True)


def expand_news_features_to_calendar(
    news_features: pd.DataFrame,
    calendar: pd.DataFrame,
    decay: float = 0.85,
    coverage_windows: tuple[int, ...] = RECENCY_WINDOWS,
    news_lag_periods: int = NEWS_LAG_PERIODS_DEFAULT,
) -> pd.DataFrame:
    if news_features.empty:
        expanded = calendar[["date", "ticker"]].drop_duplicates().copy()
        for column in _empty_news_features().columns:
            if column not in {"date", "ticker"}:
                if column == "news_top_event":
                    expanded[column] = "none"
                elif column == "news_timezone":
                    expanded[column] = UTC_TIMEZONE
                elif column in {"news_event_timestamp", "news_availability_timestamp", "news_source_timestamp"}:
                    expanded[column] = pd.NaT
                else:
                    expanded[column] = 0.0
        return expanded

    _validate_news_feature_merge_cutoffs(news_features)
    base = calendar[["date", "ticker"]].drop_duplicates().sort_values(["ticker", "date"])
    merged = base.merge(news_features, on=["date", "ticker"], how="left")
    numeric_cols = [
        "news_article_count",
        "news_sentiment_mean",
        "news_negative_ratio",
        "news_source_count",
        "news_source_diversity",
        "news_event_count",
        "text_risk_score",
        "news_confidence_mean",
        "news_token_count_mean",
        "news_text_length",
        "news_full_text_available_ratio",
    ]
    merged["date"] = pd.to_datetime(merged["date"]).dt.normalize()
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
    merged[numeric_cols] = merged[numeric_cols].fillna(0.0)
    merged["news_top_event"] = merged["news_top_event"].fillna("none")
    merged = _add_news_recency_features(
        merged,
        decay=decay,
        coverage_windows=coverage_windows,
    )
    merged = _apply_news_lag(merged, news_lag_periods=news_lag_periods)
    numeric_fill = [
        column
        for column in merged.columns
        if (column.startswith("news_") or column.startswith("text_"))
        and pd.api.types.is_numeric_dtype(merged[column])
    ]
    merged[numeric_fill] = merged[numeric_fill].fillna(0.0)
    merged["news_top_event"] = merged["news_top_event"].fillna("none")
    if "news_timezone" in merged:
        merged["news_timezone"] = merged["news_timezone"].fillna(UTC_TIMEZONE)
    return merged


def _empty_news_features() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "ticker",
            "news_event_timestamp",
            "news_availability_timestamp",
            "news_source_timestamp",
            "news_timezone",
            "news_article_count",
            "news_sentiment_mean",
            "news_negative_ratio",
            "news_source_count",
            "news_source_diversity",
            "news_event_count",
            "news_top_event",
            "text_risk_score",
            "news_confidence_mean",
            "news_token_count_mean",
            "news_text_length",
            "news_full_text_available_ratio",
            "news_recency_decay",
            "news_staleness_days",
            "news_coverage_5d",
            "news_coverage_20d",
        ]
    )


def _full_text(item: NewsItem) -> str:
    if item.full_text:
        return item.full_text
    if item.body_text:
        return item.body_text
    return item.content


def _add_news_recency_features(
    frame: pd.DataFrame,
    decay: float = 0.85,
    coverage_windows: tuple[int, ...] = RECENCY_WINDOWS,
) -> pd.DataFrame:
    frame = frame.sort_values(["ticker", "date"]).copy()
    decay = max(0.0, min(float(decay), 0.99))
    alpha = max(1.0 - decay, RECENCY_ALPHA_DEFAULT)
    grouped = frame.groupby("ticker")
    frame["news_recency_decay"] = grouped["news_article_count"].transform(
        lambda series: series.ewm(alpha=alpha, adjust=False).mean()
    )
    staleness = pd.Series(index=frame.index, dtype=float)
    for _, group in grouped:
        staleness.loc[group.index] = _staleness_days(group)
    frame["news_staleness_days"] = staleness
    for window in coverage_windows:
        frame[f"news_coverage_{window}d"] = grouped["news_article_count"].transform(
            lambda series, lookback=window: series.rolling(lookback, min_periods=1).sum()
        )
    return frame


def _staleness_days(group: pd.DataFrame) -> pd.Series:
    ordered = group.sort_values("date").copy()
    ordered["date"] = pd.to_datetime(ordered["date"])
    stale = pd.Series(index=ordered.index, dtype=float)
    last_seen: pd.Timestamp | float | None = None
    for index, row in ordered.iterrows():
        if row["news_article_count"] > 0:
            last_seen = row["date"]
        if last_seen is None:
            stale.at[index] = 999.0
        else:
            stale.at[index] = (row["date"] - last_seen).days
    return stale.reindex(group.index)


def _apply_news_lag(frame: pd.DataFrame, news_lag_periods: int) -> pd.DataFrame:
    if news_lag_periods <= 0:
        return frame
    frame = frame.sort_values(["ticker", "date"]).copy()
    grouped = frame.groupby("ticker")
    lag_columns_numeric = [
        column
        for column in frame.columns
        if (column.startswith("news_") or column.startswith("text_"))
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    for column in lag_columns_numeric:
        frame[column] = grouped[column].shift(news_lag_periods)
    for column in [
        "news_top_event",
        "news_event_timestamp",
        "news_availability_timestamp",
        "news_source_timestamp",
        "news_timezone",
    ]:
        if column in frame:
            frame[column] = grouped[column].shift(news_lag_periods)
    return frame


def _validate_news_feature_merge_cutoffs(news_features: pd.DataFrame) -> None:
    required = {"date", "ticker", "news_event_timestamp", "news_availability_timestamp"}
    missing = sorted(required.difference(news_features.columns))
    if missing:
        raise ValueError(f"news_features must include point-in-time text metadata columns: {missing}")

    normalized = news_features.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.normalize()
    availability = timestamp_utc(normalized["news_availability_timestamp"], UTC_TIMEZONE)
    event_timestamp = timestamp_utc(normalized["news_event_timestamp"], UTC_TIMEZONE)

    missing_availability = availability.isna()
    if missing_availability.any():
        first_index = normalized.index[missing_availability][0]
        raise ValueError(
            f"news_features row {first_index} has sentiment/event/risk features without "
            "news_availability_timestamp"
        )

    impossible_order = event_timestamp.notna() & (availability < event_timestamp)
    if impossible_order.any():
        first_index = normalized.index[impossible_order][0]
        raise ValueError(
            f"news_features row {first_index} has news_availability_timestamp before "
            "news_event_timestamp"
        )

    earliest_feature_date = availability.map(_feature_date)
    early_merge = normalized["date"].notna() & earliest_feature_date.notna() & (
        normalized["date"] < earliest_feature_date
    )
    if early_merge.any():
        first_index = normalized.index[early_merge][0]
        ticker = normalized.loc[first_index, "ticker"]
        feature_date = normalized.loc[first_index, "date"]
        available_date = earliest_feature_date.loc[first_index]
        raise ValueError(
            "news_features row "
            f"{first_index} for {ticker} would merge sentiment/event/risk features on "
            f"{feature_date.date()} before public availability feature date {available_date.date()}"
        )


def _feature_date(value: object) -> pd.Timestamp:
    timestamp = _as_utc_timestamp(value)
    if pd.isna(timestamp):
        return pd.NaT
    local_timestamp = timestamp.tz_convert(PRICE_TIMEZONE)
    local_midnight = local_timestamp.normalize()
    local_date = local_midnight.tz_localize(None)
    if (local_timestamp - local_midnight) > _US_MARKET_CLOSE_OFFSET:
        local_date = local_date + pd.offsets.BDay(1)
    return pd.Timestamp(local_date).normalize()


def _as_utc_timestamp(value: object) -> pd.Timestamp:
    if value is None or pd.isna(value):
        return pd.NaT
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    if isinstance(parsed, pd.Timestamp) and parsed.tzinfo is not None:
        return parsed.tz_convert(UTC_TIMEZONE)
    return pd.Timestamp(parsed).tz_localize(UTC_TIMEZONE)


def _timestamp_max(values: pd.Series) -> pd.Timestamp:
    normalized = timestamp_utc(values.dropna(), UTC_TIMEZONE)
    if normalized.empty:
        return pd.NaT
    return normalized.max()


def _timestamp_min(values: pd.Series) -> pd.Timestamp:
    normalized = timestamp_utc(values.dropna(), UTC_TIMEZONE)
    if normalized.empty:
        return pd.NaT
    return normalized.min()


def _score_token_count(scored: dict[str, float | str | bool], text: str) -> float:
    raw = scored.get("token_count")
    if isinstance(raw, bool):
        return 0.0
    try:
        if raw is not None:
            return float(raw)
    except (TypeError, ValueError):
        pass
    return float(len(re.findall(r"[a-zA-Z]+", text.lower())))
