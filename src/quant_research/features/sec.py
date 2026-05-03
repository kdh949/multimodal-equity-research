from __future__ import annotations

from collections import Counter

import pandas as pd


def build_sec_features(
    filings_by_ticker: dict[str, pd.DataFrame],
    facts_by_ticker: dict[str, pd.DataFrame],
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    base = calendar[["date", "ticker"]].drop_duplicates().copy()
    base["date"] = pd.to_datetime(base["date"]).dt.normalize()
    rows: list[pd.DataFrame] = []
    for ticker, group in base.groupby("ticker"):
        features = group.sort_values("date").copy()
        filings = filings_by_ticker.get(ticker, pd.DataFrame())
        filing_daily = _daily_filing_features(filings, ticker)
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


def _daily_filing_features(filings: pd.DataFrame, ticker: str) -> pd.DataFrame:
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
    extractor = FilingEventExtractor()
    extracted = frame.apply(
        lambda row: extractor.extract(f"Form {row['form']} {row.get('primary_document', '')}"),
        axis=1,
        result_type="expand",
    )
    frame["sec_event_tag"] = extracted["event_tag"]
    frame["sec_event_confidence"] = extracted["confidence"].astype(float)
    frame["sec_summary_ref"] = extracted["summary_ref"]

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
