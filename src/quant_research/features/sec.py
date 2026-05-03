from __future__ import annotations

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

        facts = facts_by_ticker.get(ticker, pd.DataFrame())
        features = _merge_fact_features(features, facts)
        rows.append(features)
    if not rows:
        return base
    return pd.concat(rows, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


def _daily_filing_features(filings: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if filings.empty:
        return pd.DataFrame(columns=["date", "ticker", "sec_8k_count", "sec_10q_count", "sec_10k_count", "sec_form4_count", "sec_risk_flag"])
    frame = filings.copy()
    frame["date"] = pd.to_datetime(frame["filing_date"], errors="coerce").dt.normalize()
    frame["ticker"] = ticker
    frame["sec_8k_count"] = (frame["form"] == "8-K").astype(float)
    frame["sec_10q_count"] = (frame["form"] == "10-Q").astype(float)
    frame["sec_10k_count"] = (frame["form"] == "10-K").astype(float)
    frame["sec_form4_count"] = (frame["form"] == "4").astype(float)
    frame["sec_risk_flag"] = frame["form"].isin({"8-K", "4"}).astype(float)
    return (
        frame.groupby(["date", "ticker"], as_index=False)[
            ["sec_8k_count", "sec_10q_count", "sec_10k_count", "sec_form4_count", "sec_risk_flag"]
        ]
        .sum()
        .sort_values(["date", "ticker"])
    )


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
    fact_frame = fact_frame[["period_end", "revenue_growth", "net_income_growth", "assets_growth"]]

    merged = pd.merge_asof(
        features.sort_values("date"),
        fact_frame.rename(columns={"period_end": "date"}).sort_values("date"),
        on="date",
        direction="backward",
    )
    for column in ["revenue_growth", "net_income_growth", "assets_growth"]:
        merged[column] = merged[column].fillna(0.0)
    return merged
