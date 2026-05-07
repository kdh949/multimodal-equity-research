from __future__ import annotations

import pandas as pd

from quant_research.data.timestamps import validate_generated_feature_cutoffs
from quant_research.features.text import expand_news_features_to_calendar


def fuse_features(
    price_features: pd.DataFrame,
    news_features: pd.DataFrame,
    sec_features: pd.DataFrame,
) -> pd.DataFrame:
    frame = _normalize_features(price_features, required_columns={"date", "ticker"}, label="price_features")
    if frame.empty:
        return pd.DataFrame()

    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame = frame.sort_values(["date", "ticker"]).drop_duplicates(["date", "ticker"]).reset_index(drop=True)

    expanded_news = expand_news_features_to_calendar(news_features, frame)
    expanded_news["date"] = pd.to_datetime(expanded_news["date"]).dt.normalize()
    expanded_news["ticker"] = expanded_news["ticker"].astype(str).str.upper()
    expanded_news = expanded_news.sort_values(["date", "ticker"]).drop_duplicates(["date", "ticker"]).reset_index(drop=True)

    sec_aligned = _normalize_features(sec_features, required_columns={"date", "ticker"}, label="sec_features")
    if not sec_aligned.empty:
        sec_aligned["ticker"] = sec_aligned["ticker"].astype(str).str.upper()
        sec_aligned["date"] = pd.to_datetime(sec_aligned["date"]).dt.normalize()
        sec_aligned = sec_aligned.sort_values(["date", "ticker"]).drop_duplicates(["date", "ticker"]).reset_index(drop=True)

    fused = frame.merge(expanded_news, on=["date", "ticker"], how="left")
    fused = fused.merge(sec_aligned, on=["date", "ticker"], how="left")
    fused = fused.sort_values(["date", "ticker"]).reset_index(drop=True)
    fused["date"] = pd.to_datetime(fused["date"]).dt.normalize()
    if fused.duplicated(subset=["date", "ticker"]).any():
        fused = fused.drop_duplicates(subset=["date", "ticker"], keep="last").reset_index(drop=True)

    fill_zero_prefixes = (
        "news_",
        "text_",
        "sec_",
        "revenue_",
        "net_income_",
        "assets_",
    )
    for column in fused.columns:
        if (
            column.startswith(fill_zero_prefixes)
            and column != "news_top_event"
            and pd.api.types.is_numeric_dtype(fused[column])
        ):
            fused[column] = fused[column].fillna(0.0)

    for column in ["news_top_event", "sec_event_tag"]:
        if column in fused:
            fused[column] = fused[column].fillna("none").astype(str)
            fused[column] = fused[column].where(~fused[column].str.lower().isin(["", "none", "none ", "nan", "null"]), "none")
    if "sec_summary_ref" in fused:
        fused["sec_summary_ref"] = fused["sec_summary_ref"].fillna("").astype(str)

    validate_generated_feature_cutoffs(fused, label="fused feature pipeline")
    return fused


def _normalize_features(frame: pd.DataFrame, required_columns: set[str], label: str) -> pd.DataFrame:
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} must include {sorted(missing)}")
    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.normalize()
    return normalized
