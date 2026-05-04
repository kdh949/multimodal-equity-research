from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

DEFAULT_FILING_EVENT_CACHE_PATH = Path("data/processed") / "sec" / "filing_extraction_cache.jsonl"
_CACHE_RECORD_VERSION = 1


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
        features = features.merge(filing_daily, on=["date", "ticker"], how="left")
        for column in ["sec_8k_count", "sec_10q_count", "sec_10k_count", "sec_form4_count", "sec_risk_flag"]:
            features[column] = features[column].fillna(0.0)
            features[f"{column}_20d"] = features[column].rolling(20, min_periods=1).sum()
        features["sec_event_tag"] = features["sec_event_tag"].fillna("none")
        features["sec_event_confidence"] = features["sec_event_confidence"].fillna(0.0)
        features["sec_summary_ref"] = features["sec_summary_ref"].fillna("")

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
            ]
        )
    frame = filings.copy()
    frame["date"] = pd.to_datetime(frame["filing_date"], errors="coerce").dt.normalize()
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

    for row in frame.itertuples(index=False):
        text = _build_filing_text(row)
        cache_key = _build_filing_cache_key(row, text, extractor)
        cached = cache.get(cache_key)

        if cached:
            event_tag = str(cached.get("event_tag", "none"))
            confidence = _safe_float(cached.get("confidence", 0.0))
            summary_ref = str(cached.get("summary_ref", ""))
        else:
            extracted = extractor.extract(text)
            event_tag = str(extracted.get("event_tag", "none"))
            confidence = _safe_float(extracted.get("confidence", 0.0))
            summary_ref = str(extracted.get("summary_ref", ""))
            if cache_path is not None and _should_cache_extraction(extractor):
                cache[cache_key] = {
                    "version": _CACHE_RECORD_VERSION,
                    "event_tag": event_tag,
                    "confidence": confidence,
                    "summary_ref": summary_ref,
                    "risk_flag": bool(extracted.get("risk_flag", False)),
                }
                cache_updated = True

        event_tags.append(event_tag)
        confidences.append(confidence)
        summary_refs.append(summary_ref)

    frame["sec_event_tag"] = event_tags
    frame["sec_event_confidence"] = confidences
    frame["sec_summary_ref"] = summary_refs

    if cache_updated and cache_path is not None:
        _save_filing_extraction_cache(cache, cache_path)

    rows: list[dict[str, object]] = []
    numeric_columns = ["sec_8k_count", "sec_10q_count", "sec_10k_count", "sec_form4_count", "sec_risk_flag"]
    for (date, grouped_ticker), group in frame.groupby(["date", "ticker"]):
        event_counter = Counter(
            tag for tags in group["sec_event_tag"] for tag in str(tags).split(",") if tag and tag != "none"
        )
        row = {
            "date": date,
            "ticker": grouped_ticker,
            **{column: group[column].sum() for column in numeric_columns},
            "sec_event_tag": event_counter.most_common(1)[0][0] if event_counter else "none",
            "sec_event_confidence": group["sec_event_confidence"].mean(),
            "sec_summary_ref": group["sec_summary_ref"].iloc[0],
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["date", "ticker"])


def _merge_fact_features(features: pd.DataFrame, facts: pd.DataFrame) -> pd.DataFrame:
    if facts.empty:
        for column in ["revenue_growth", "net_income_growth", "assets_growth"]:
            features[column] = 0.0
        return features

    fact_frame = facts.copy()
    fact_frame["period_end"] = pd.to_datetime(fact_frame["period_end"], errors="coerce").dt.normalize()
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
    extra_columns = [column for column in fact_frame.columns if column.startswith("sec_frame_")]
    fact_frame = fact_frame[
        ["period_end", "revenue_growth", "net_income_growth", "assets_growth", *extra_columns]
    ]

    merged = pd.merge_asof(
        features.sort_values("date"),
        fact_frame.rename(columns={"period_end": "date"}).sort_values("date"),
        on="date",
        direction="backward",
    )
    for column in ["revenue_growth", "net_income_growth", "assets_growth"]:
        merged[column] = merged[column].fillna(0.0)
    for column in [column for column in fact_frame.columns if column.startswith("sec_frame_")]:
        merged[column] = merged[column].ffill().fillna(0.0)
    return merged


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


def _build_filing_text(row: pd.Series | object) -> str:
    return f"Form {getattr(row, 'form', '')} {getattr(row, 'primary_document', '')}"


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
