from __future__ import annotations

import pandas as pd

from quant_research.features.text import expand_news_features_to_calendar


def fuse_features(
    price_features: pd.DataFrame,
    news_features: pd.DataFrame,
    sec_features: pd.DataFrame,
) -> pd.DataFrame:
    frame = price_features.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    expanded_news = expand_news_features_to_calendar(news_features, frame)
    fused = frame.merge(expanded_news, on=["date", "ticker"], how="left")
    fused = fused.merge(sec_features, on=["date", "ticker"], how="left")
    fused = fused.sort_values(["date", "ticker"]).reset_index(drop=True)

    fill_zero_prefixes = (
        "news_",
        "text_",
        "sec_",
        "revenue_",
        "net_income_",
        "assets_",
    )
    for column in fused.columns:
        if column.startswith(fill_zero_prefixes) and column != "news_top_event":
            fused[column] = fused[column].fillna(0.0)
    if "news_top_event" in fused:
        fused["news_top_event"] = fused["news_top_event"].fillna("none")
    return fused
