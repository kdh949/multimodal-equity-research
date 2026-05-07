from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from quant_research.data.timestamps import (
    PRICE_TIMEZONE,
    UTC_TIMEZONE,
    coalesce_timestamps,
    date_end_utc,
    timestamp_utc,
)

DEFAULT_FILING_EVENT_CACHE_PATH = Path("data/processed") / "sec" / "filing_extraction_cache.jsonl"
_CACHE_RECORD_VERSION = 1
_SECTION_LIST_SEPARATOR = "|"
_FILING_SECTION_RE = re.compile(
    r"(?i)\bitem\s+([0-9]{1,2}(?:\.[0-9]{1,2})?[a-z]?)\b",
)
_FILING_SECTION_RISK_HINTS = ("risk", "litigation", "investigation", "bankruptcy", "fraud")
_US_MARKET_CLOSE_OFFSET = pd.Timedelta(hours=16)


def build_sec_features(
    filings_by_ticker: dict[str, pd.DataFrame],
    facts_by_ticker: dict[str, pd.DataFrame],
    calendar: pd.DataFrame,
    filing_extractor: object | None = None,
    filing_cache_path: str | Path | None = None,
) -> pd.DataFrame:
    base = calendar[["date", "ticker"]].drop_duplicates().copy()
    base["date"] = pd.to_datetime(base["date"]).dt.normalize()
    rows: list[pd.DataFrame] = []
    for ticker, group in base.groupby("ticker"):
        features = group.sort_values("date").copy()
        filings = filings_by_ticker.get(ticker, pd.DataFrame())
        filing_daily = _daily_filing_features(
            filings,
            ticker,
            filing_extractor,
            filing_cache_path,
        )
        filing_daily = _align_filing_features_to_calendar(filing_daily, features["date"])
        features = features.merge(filing_daily, on=["date", "ticker"], how="left")
        for column in [
            "sec_8k_count",
            "sec_10q_count",
            "sec_10k_count",
            "sec_form4_count",
            "sec_risk_flag",
            "sec_filing_section_count",
            "sec_filing_risk_section_count",
        ]:
            features[column] = features[column].fillna(0.0)
            features[f"{column}_20d"] = features[column].rolling(20, min_periods=1).sum()
        for column in [
            "sec_filing_text_available",
            "sec_filing_text_length",
            "sec_filing_risk_keyword_count",
        ]:
            features[column] = features[column].fillna(0.0)
            features[f"{column}_20d"] = features[column].rolling(20, min_periods=1).sum()
        features["sec_event_tag"] = features["sec_event_tag"].fillna("none")
        features["sec_event_confidence"] = features["sec_event_confidence"].fillna(0.0)
        features["sec_summary_ref"] = features["sec_summary_ref"].fillna("")
        for column in ["sec_event_timestamp", "sec_availability_timestamp", "sec_source_timestamp"]:
            if column in features:
                features[column] = features[column].ffill()
        if "sec_timezone" in features:
            features["sec_timezone"] = features["sec_timezone"].fillna(UTC_TIMEZONE)
        features["sec_filing_sections"] = features["sec_filing_sections"].fillna("")

        facts = facts_by_ticker.get(ticker, pd.DataFrame())
        features = _merge_fact_features(features, facts)
        rows.append(features)
    if not rows:
        return base
    return pd.concat(rows, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


def _daily_filing_features(
    filings: pd.DataFrame,
    ticker: str,
    filing_extractor: object | None = None,
    filing_cache_path: str | Path | None = None,
) -> pd.DataFrame:
    from quant_research.models.text import FilingEventExtractor

    if filings.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "sec_8k_count",
                "sec_10q_count",
                "sec_10k_count",
                "sec_form4_count",
                "sec_risk_flag",
                "sec_event_tag",
                "sec_event_confidence",
                "sec_summary_ref",
                "sec_event_timestamp",
                "sec_availability_timestamp",
                "sec_source_timestamp",
                "sec_timezone",
                "sec_filing_text_available",
                "sec_filing_text_length",
                "sec_filing_risk_keyword_count",
                "sec_filing_section_count",
                "sec_filing_risk_section_count",
                "sec_filing_sections",
            ]
        )
    frame = filings.copy()
    frame = _ensure_filing_timestamp_columns(frame)
    frame = _drop_filings_with_future_report_events(frame)
    if frame.empty:
        return _daily_filing_features(
            pd.DataFrame(),
            ticker,
            filing_extractor=filing_extractor,
            filing_cache_path=filing_cache_path,
        )
    frame["date"] = _filing_feature_date(frame, fallback_column="filing_date")
    frame["ticker"] = ticker
    frame["sec_8k_count"] = (frame["form"] == "8-K").astype(float)
    frame["sec_10q_count"] = (frame["form"] == "10-Q").astype(float)
    frame["sec_10k_count"] = (frame["form"] == "10-K").astype(float)
    frame["sec_form4_count"] = (frame["form"] == "4").astype(float)
    frame["sec_risk_flag"] = frame["form"].isin({"8-K", "4"}).astype(float)
    extractor = filing_extractor or FilingEventExtractor()

    cache_path = Path(filing_cache_path) if filing_cache_path is not None else None
    cache = _load_filing_extraction_cache(cache_path) if cache_path is not None else {}
    cache_updated = False
    event_tags: list[str] = []
    confidences: list[float] = []
    summary_refs: list[str] = []
    body_available: list[float] = []
    body_lengths: list[float] = []
    risk_keyword_counts: list[float] = []
    filing_section_strings: list[str] = []
    filing_section_counts: list[float] = []
    filing_risk_section_counts: list[float] = []

    for row in frame.itertuples(index=False):
        body_text = _as_document_text(row)
        text = _build_filing_text(row, body_text)
        cache_key = _build_filing_cache_key(row, text, extractor)
        cached = cache.get(cache_key)

        if cached:
            event_tag = str(cached.get("event_tag", "none"))
            confidence = _safe_float(cached.get("confidence", 0.0))
            summary_ref = str(cached.get("summary_ref", ""))
            risk_keyword_count = _safe_float(cached.get("risk_keyword_count", 0.0))
            body_available_value = _safe_float(cached.get("body_available", 0.0))
            body_length = _safe_float(cached.get("body_length", 0.0))
            sections = _split_filing_sections(str(cached.get("sections", "")))
            if not sections:
                sections = _extract_filing_sections(body_text)
            section_count = _safe_float(cached.get("section_count", _safe_float(len(sections))))
            risk_section_count = _safe_float(cached.get("risk_section_count", _safe_float(_count_filing_risk_sections(body_text, sections))))
        else:
            extracted = extractor.extract(text)
            event_tag = str(extracted.get("event_tag", "none"))
            confidence = _safe_float(extracted.get("confidence", 0.0))
            summary_ref = str(extracted.get("summary_ref", ""))
            risk_keyword_count = _count_filing_risk_keywords(body_text)
            body_available_value = 1.0 if body_text else 0.0
            body_length = float(len(body_text))
            sections = _extract_filing_sections(body_text)
            section_count = float(len(sections))
            risk_section_count = _count_filing_risk_sections(body_text, sections)
            if cache_path is not None and _should_cache_extraction(extractor):
                cache[cache_key] = {
                    "version": _CACHE_RECORD_VERSION,
                    "event_tag": event_tag,
                    "confidence": confidence,
                    "summary_ref": summary_ref,
                    "risk_flag": bool(extracted.get("risk_flag", False)),
                    "risk_keyword_count": risk_keyword_count,
                    "body_available": body_available_value,
                    "body_length": body_length,
                    "section_count": section_count,
                    "risk_section_count": risk_section_count,
                    "sections": _SECTION_LIST_SEPARATOR.join(sections),
                }
                cache_updated = True

        event_tags.append(event_tag)
        confidences.append(confidence)
        summary_refs.append(summary_ref)
        body_available.append(body_available_value)
        body_lengths.append(body_length)
        risk_keyword_counts.append(risk_keyword_count)
        filing_section_strings.append(_SECTION_LIST_SEPARATOR.join(sections))
        filing_section_counts.append(section_count)
        filing_risk_section_counts.append(risk_section_count)

    frame["sec_filing_text_available"] = body_available
    frame["sec_filing_text_length"] = body_lengths
    frame["sec_filing_risk_keyword_count"] = risk_keyword_counts
    frame["sec_filing_sections"] = filing_section_strings
    frame["sec_filing_section_count"] = filing_section_counts
    frame["sec_filing_risk_section_count"] = filing_risk_section_counts
    frame["sec_event_tag"] = event_tags
    frame["sec_event_confidence"] = confidences
    frame["sec_summary_ref"] = summary_refs

    if cache_updated and cache_path is not None:
        _save_filing_extraction_cache(cache, cache_path)

    rows: list[dict[str, object]] = []
    numeric_columns = [
        "sec_8k_count",
        "sec_10q_count",
        "sec_10k_count",
        "sec_form4_count",
        "sec_risk_flag",
        "sec_filing_text_available",
        "sec_filing_text_length",
        "sec_filing_risk_keyword_count",
        "sec_filing_section_count",
        "sec_filing_risk_section_count",
    ]
    for (date, grouped_ticker), group in frame.groupby(["date", "ticker"]):
        event_counter = Counter(
            tag for tags in group["sec_event_tag"] for tag in str(tags).split(",") if tag and tag != "none"
        )
        section_list = _merge_filing_sections(group["sec_filing_sections"])
        row = {
            "date": date,
            "ticker": grouped_ticker,
            **{column: group[column].sum() for column in numeric_columns},
            "sec_event_tag": event_counter.most_common(1)[0][0] if event_counter else "none",
            "sec_event_confidence": group["sec_event_confidence"].mean(),
            "sec_summary_ref": group["sec_summary_ref"].iloc[0],
            "sec_event_timestamp": _timestamp_min(group["event_timestamp"]),
            "sec_availability_timestamp": _timestamp_max(group["availability_timestamp"]),
            "sec_source_timestamp": _timestamp_max(group["source_timestamp"]),
            "sec_timezone": UTC_TIMEZONE,
            "sec_filing_sections": _SECTION_LIST_SEPARATOR.join(section_list),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["date", "ticker"])


def _align_filing_features_to_calendar(filing_daily: pd.DataFrame, calendar_dates: pd.Series) -> pd.DataFrame:
    if filing_daily.empty:
        return filing_daily

    trading_dates = pd.to_datetime(calendar_dates).dropna().drop_duplicates().sort_values().to_numpy()
    if len(trading_dates) == 0:
        return filing_daily.iloc[0:0].copy()

    aligned = filing_daily.copy()
    filing_dates = pd.to_datetime(aligned["date"]).dt.normalize().to_numpy()
    positions = trading_dates.searchsorted(filing_dates, side="left")
    valid = positions < len(trading_dates)
    aligned = aligned.loc[valid].copy()
    if aligned.empty:
        return aligned
    aligned["date"] = pd.to_datetime(trading_dates[positions[valid]]).normalize()

    numeric_columns = [
        column
        for column in [
            "sec_8k_count",
            "sec_10q_count",
            "sec_10k_count",
            "sec_form4_count",
            "sec_risk_flag",
            "sec_filing_text_available",
            "sec_filing_text_length",
            "sec_filing_risk_keyword_count",
            "sec_filing_section_count",
            "sec_filing_risk_section_count",
        ]
        if column in aligned.columns
    ]
    rows: list[dict[str, object]] = []
    for (date, ticker), group in aligned.groupby(["date", "ticker"], sort=True):
        event_counter = Counter(
            tag for tags in group["sec_event_tag"] for tag in str(tags).split(",") if tag and tag != "none"
        )
        section_list = _merge_filing_sections(group["sec_filing_sections"])
        rows.append(
            {
                "date": date,
                "ticker": ticker,
                **{column: group[column].sum() for column in numeric_columns},
                "sec_event_tag": event_counter.most_common(1)[0][0] if event_counter else "none",
                "sec_event_confidence": group["sec_event_confidence"].mean(),
                "sec_summary_ref": group["sec_summary_ref"].iloc[0],
                "sec_event_timestamp": _timestamp_min(group["sec_event_timestamp"]),
                "sec_availability_timestamp": _timestamp_max(group["sec_availability_timestamp"]),
                "sec_source_timestamp": _timestamp_max(group["sec_source_timestamp"]),
                "sec_timezone": UTC_TIMEZONE,
                "sec_filing_sections": _SECTION_LIST_SEPARATOR.join(section_list),
            }
        )
    return pd.DataFrame(rows).sort_values(["date", "ticker"])


def _merge_fact_features(features: pd.DataFrame, facts: pd.DataFrame) -> pd.DataFrame:
    if facts.empty:
        for column in ["revenue_growth", "net_income_growth", "assets_growth"]:
            features[column] = 0.0
        return features

    fact_frame = facts.copy()
    fact_frame["period_end"] = pd.to_datetime(fact_frame["period_end"], errors="coerce").dt.normalize()
    fact_frame = _ensure_fact_timestamp_columns(fact_frame)
    fact_frame = _drop_rows_available_before_event(fact_frame)
    if fact_frame.empty:
        for column in ["revenue_growth", "net_income_growth", "assets_growth"]:
            features[column] = 0.0
        return features
    fact_frame = fact_frame.sort_values("period_end")
    for raw, column in [
        ("revenue", "revenue_growth"),
        ("net_income", "net_income_growth"),
        ("assets", "assets_growth"),
    ]:
        if raw in fact_frame:
            fact_frame[column] = fact_frame[raw].pct_change(4).fillna(fact_frame[raw].pct_change()).fillna(0.0)
        else:
            fact_frame[column] = 0.0
    fact_frame["sec_fact_event_timestamp"] = fact_frame["event_timestamp"]
    fact_frame["sec_fact_availability_timestamp"] = fact_frame["availability_timestamp"]
    fact_frame["sec_fact_source_timestamp"] = fact_frame["source_timestamp"]
    fact_frame["sec_fact_timezone"] = fact_frame["timezone"].fillna(UTC_TIMEZONE)
    fact_frame["available_date"] = _availability_date(fact_frame, fallback_column="period_end")
    extra_columns = [column for column in fact_frame.columns if column.startswith("sec_frame_")]
    fact_frame = fact_frame[
        [
            "available_date",
            "revenue_growth",
            "net_income_growth",
            "assets_growth",
            "sec_fact_event_timestamp",
            "sec_fact_availability_timestamp",
            "sec_fact_source_timestamp",
            "sec_fact_timezone",
            *extra_columns,
        ]
    ]
    fact_frame = fact_frame.sort_values(["available_date"]).dropna(subset=["available_date"])

    merged = pd.merge_asof(
        features.sort_values("date"),
        fact_frame.rename(columns={"available_date": "date"}).sort_values("date"),
        on="date",
        direction="backward",
    )
    for column in ["revenue_growth", "net_income_growth", "assets_growth"]:
        merged[column] = merged[column].fillna(0.0)
    for column in [column for column in fact_frame.columns if column.startswith("sec_frame_")]:
        merged[column] = merged[column].ffill().fillna(0.0)
    for column in ["sec_fact_event_timestamp", "sec_fact_availability_timestamp", "sec_fact_source_timestamp"]:
        if column in merged:
            merged[column] = merged[column].ffill()
    if "sec_fact_timezone" in merged:
        merged["sec_fact_timezone"] = merged["sec_fact_timezone"].ffill().fillna(UTC_TIMEZONE)
    return merged


def _ensure_filing_timestamp_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "filing_date" in normalized:
        filing_timestamp = date_end_utc(normalized["filing_date"], UTC_TIMEZONE)
    else:
        filing_timestamp = pd.Series(pd.NaT, index=normalized.index, dtype="datetime64[ns, UTC]")
    if "source_timestamp" not in normalized:
        normalized["source_timestamp"] = coalesce_timestamps(
            _coalesced_timestamp_columns(
                normalized,
                (
                    "public_timestamp",
                    "publication_timestamp",
                    "published_at",
                    "public_date",
                    "acceptance_datetime",
                ),
            ),
            filing_timestamp,
        )
    if "event_timestamp" not in normalized:
        if "report_date" in normalized:
            report_timestamp = date_end_utc(normalized["report_date"], UTC_TIMEZONE)
        else:
            report_timestamp = pd.Series(pd.NaT, index=normalized.index, dtype="datetime64[ns, UTC]")
        normalized["event_timestamp"] = coalesce_timestamps(
            report_timestamp,
            normalized["source_timestamp"],
            filing_timestamp,
        )
    if "availability_timestamp" not in normalized:
        normalized["availability_timestamp"] = coalesce_timestamps(
            _coalesced_timestamp_columns(
                normalized,
                (
                    "collected_at",
                    "collection_timestamp",
                    "retrieved_at",
                    "downloaded_at",
                    "available_at",
                    "available_timestamp",
                ),
            ),
            normalized["source_timestamp"],
            filing_timestamp,
        )
    if "timezone" not in normalized:
        normalized["timezone"] = UTC_TIMEZONE
    return normalized


def _coalesced_timestamp_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    values = [
        timestamp_utc(frame[column], UTC_TIMEZONE)
        for column in columns
        if column in frame
    ]
    if not values:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    return coalesce_timestamps(*values)


def _drop_filings_with_future_report_events(frame: pd.DataFrame) -> pd.DataFrame:
    if "report_date" not in frame:
        return frame
    report_event_timestamp = date_end_utc(frame["report_date"], UTC_TIMEZONE)
    availability_timestamp = timestamp_utc(frame["availability_timestamp"], UTC_TIMEZONE)
    keep = (
        report_event_timestamp.isna()
        | availability_timestamp.isna()
        | (availability_timestamp >= report_event_timestamp)
    )
    return frame.loc[keep].copy()


def _ensure_fact_timestamp_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "event_timestamp" not in normalized:
        normalized["event_timestamp"] = date_end_utc(normalized["period_end"], UTC_TIMEZONE)
    if "source_timestamp" not in normalized:
        if "filed" in normalized:
            normalized["source_timestamp"] = date_end_utc(normalized["filed"], UTC_TIMEZONE)
        else:
            normalized["source_timestamp"] = pd.Series(pd.NaT, index=normalized.index, dtype="datetime64[ns, UTC]")
    if "availability_timestamp" not in normalized:
        normalized["availability_timestamp"] = coalesce_timestamps(
            normalized["source_timestamp"],
            normalized["event_timestamp"],
        )
    if "timezone" not in normalized:
        normalized["timezone"] = UTC_TIMEZONE
    return normalized


def _drop_rows_available_before_event(frame: pd.DataFrame) -> pd.DataFrame:
    event_timestamp = timestamp_utc(frame["event_timestamp"], UTC_TIMEZONE)
    availability_timestamp = timestamp_utc(frame["availability_timestamp"], UTC_TIMEZONE)
    keep = event_timestamp.isna() | availability_timestamp.isna() | (availability_timestamp >= event_timestamp)
    return frame.loc[keep].copy()


def _availability_date(frame: pd.DataFrame, fallback_column: str) -> pd.Series:
    if "availability_timestamp" in frame:
        availability = timestamp_utc(frame["availability_timestamp"], UTC_TIMEZONE)
    else:
        availability = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    if fallback_column in frame:
        fallback = date_end_utc(frame[fallback_column], UTC_TIMEZONE)
        availability = coalesce_timestamps(availability, fallback)
    return availability.dt.tz_convert(UTC_TIMEZONE).dt.tz_localize(None).dt.normalize()


def _filing_feature_date(frame: pd.DataFrame, fallback_column: str) -> pd.Series:
    if "availability_timestamp" in frame:
        availability = timestamp_utc(frame["availability_timestamp"], UTC_TIMEZONE)
    else:
        availability = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    if fallback_column in frame:
        fallback = date_end_utc(frame[fallback_column], UTC_TIMEZONE)
        availability = coalesce_timestamps(availability, fallback)

    local_availability = availability.dt.tz_convert(PRICE_TIMEZONE)
    local_midnight = local_availability.dt.normalize()
    local_dates = local_midnight.dt.tz_localize(None)
    after_close = (local_availability - local_midnight) > _US_MARKET_CLOSE_OFFSET
    return local_dates.mask(after_close.fillna(False), local_dates + pd.offsets.BDay(1)).dt.normalize()


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


def _load_filing_extraction_cache(cache_path: Path) -> dict[str, dict[str, object]]:
    if not cache_path.exists():
        return {}
    try:
        payload_text = cache_path.read_text(encoding="utf-8")
    except OSError:
        return {}

    payload_text = payload_text.strip()
    if not payload_text:
        return {}

    data: dict[str, dict[str, object]] = {}

    for entry in _iter_cache_records(payload_text, is_jsonl=cache_path.suffix == ".jsonl"):
        key = entry.get("key")
        if not isinstance(key, str):
            continue
        record_version = entry.get("version")
        if record_version is not None and str(record_version) != str(_CACHE_RECORD_VERSION):
            continue
        event_tag = entry.get("event_tag")
        if not isinstance(event_tag, str):
            event_tag = "none"
        risk_flag = entry.get("risk_flag")
        confidence = _safe_float(entry.get("confidence", 0.0))
        summary_ref = entry.get("summary_ref")
        data[key] = {
            "event_tag": event_tag,
            "risk_flag": bool(risk_flag),
            "confidence": confidence,
            "summary_ref": str(summary_ref) if summary_ref is not None else "",
            "risk_keyword_count": _safe_float(entry.get("risk_keyword_count", 0.0)),
            "body_available": _safe_float(entry.get("body_available", 0.0)),
            "body_length": _safe_float(entry.get("body_length", 0.0)),
            "section_count": _safe_float(entry.get("section_count", 0.0)),
            "risk_section_count": _safe_float(entry.get("risk_section_count", 0.0)),
            "sections": str(entry.get("sections", "")),
        }
    return data


def _iter_cache_records(payload_text: str, is_jsonl: bool = False) -> Iterable[dict[str, object]]:
    if not is_jsonl:
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            payload = None

        if isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(key, str) and isinstance(value, dict):
                    value = dict(value)
                    value["key"] = key
                    yield value
            return

        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    yield item
            return

    for line in payload_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            yield entry


def _save_filing_extraction_cache(cache: dict[str, dict[str, object]], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            for key, entry in cache.items():
                record = dict(entry)
                record["key"] = key
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
        tmp_path.replace(cache_path)
    except OSError:
        return


def _build_filing_text(row: pd.Series | object, body_text: str = "") -> str:
    if _has_value(body_text):
        return str(body_text)
    return f"Form {getattr(row, 'form', '')} {getattr(row, 'primary_document', '')}"


def _as_document_text(row: pd.Series | object) -> str:
    raw = getattr(row, "document_text", "")
    if raw is None:
        return ""
    if isinstance(raw, float) and pd.isna(raw):
        return ""
    return str(raw)


def _count_filing_risk_keywords(value: str) -> float:
    return float(len(_FILING_RISK_KEYWORD_RE.findall(str(value))))


def _extract_filing_sections(text: str) -> list[str]:
    clean = re.sub(r"\s+", " ", str(text)).strip()
    if not clean:
        return []
    sections: list[str] = []
    for match in _FILING_SECTION_RE.finditer(clean):
        token = _normalize_filing_section_token(match.group(1))
        if token and token not in sections:
            sections.append(token)
    return sections


def _split_filing_sections(raw_sections: str) -> list[str]:
    values = []
    for item in str(raw_sections).split(_SECTION_LIST_SEPARATOR):
        section = item.strip().lower()
        if section and section not in values:
            values.append(section)
    return values


def _count_filing_risk_sections(text: str, sections: list[str]) -> float:
    if sections:
        return float(
            sum(
                1
                for section in sections
                if section.startswith("item_1a") or any(risk in section for risk in _FILING_SECTION_RISK_HINTS)
            )
        )
    return float(1 if re.search(r"\bitem\s+1a\b", str(text), flags=re.IGNORECASE) else 0)


def _merge_filing_sections(section_values: pd.Series) -> list[str]:
    ordered_sections: list[str] = []
    for entry in section_values.dropna():
        for section in _split_filing_sections(str(entry)):
            if section and section not in ordered_sections:
                ordered_sections.append(section)
    return ordered_sections


_FILING_RISK_KEYWORD_RE = re.compile(
    r"\b(?:risk|risks|restatement|lawsuit|investigation|impairment|litigation|bankruptcy|fraud|probe)\b",
    re.IGNORECASE,
)


def _normalize_filing_section_token(value: str) -> str:
    normalized = re.sub(r"\s+", "", value.lower())
    if not normalized:
        return ""
    normalized = normalized.replace(".", "_").replace("-", "_")
    return f"item_{normalized}"


def _build_filing_cache_key(row: pd.Series | object, text: str, extractor: object) -> str:
    namespace = _extractor_cache_namespace(extractor)
    accession_number = getattr(row, "accession_number", "")
    if _has_value(accession_number):
        identity = f"accession:{str(accession_number).strip()}"
        return _stable_hash(f"{namespace}|{identity}")
    form = getattr(row, "form", "")
    primary_document = getattr(row, "primary_document", "")
    filing_date = getattr(row, "filing_date", None)
    if filing_date is not None:
        filing_date = pd.to_datetime(filing_date, errors="coerce")
        filing_date = filing_date.strftime("%Y-%m-%d") if pd.notna(filing_date) else "na"
    else:
        filing_date = "na"
    key_basis = f"fallback:{form}|{primary_document}|{filing_date}|{_stable_hash(text)}"
    return _stable_hash(f"{namespace}|{key_basis}")


def _extractor_cache_namespace(extractor: object) -> str:
    parts = [f"{extractor.__class__.__module__}.{extractor.__class__.__name__}"]
    for attr in (
        "model_id",
        "base_model_id",
        "runtime",
        "runtime_model_path",
        "ollama_model",
        "use_local_model",
    ):
        if hasattr(extractor, attr):
            value = getattr(extractor, attr)
            if value not in (None, ""):
                parts.append(f"{attr}={value}")
    return "|".join(parts)


def _should_cache_extraction(extractor: object) -> bool:
    if bool(getattr(extractor, "use_local_model", False)):
        return getattr(extractor, "last_source", None) == "local"
    return True


def _safe_float(value: object) -> float:
    try:
        if isinstance(value, bool):
            return float(value)
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _has_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip() != ""
    try:
        return not pd.isna(value)
    except (TypeError, ValueError):
        return True


def _stable_hash(value: str) -> str:
    normalized = " ".join(str(value).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
