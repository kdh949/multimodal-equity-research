from __future__ import annotations

import json
from datetime import date

import pandas as pd

from quant_research.data.market import SyntheticMarketDataProvider
from quant_research.data.news import LocalNewsProvider, NewsItem, news_items_to_frame
from quant_research.data.sec import extract_companyfacts_frame
from quant_research.data.timestamps import (
    FEATURE_AVAILABILITY_SCHEMA_VERSION,
    FeatureAvailabilitySchema,
    filter_available_as_of,
    standardize_feature_availability_metadata,
    validate_feature_availability,
)
from quant_research.features.price import build_price_features
from quant_research.features.sec import build_sec_features
from quant_research.features.text import build_news_features, expand_news_features_to_calendar


def test_market_provider_normalizes_daily_bars_to_new_york_close_utc() -> None:
    provider = SyntheticMarketDataProvider(periods=1, seed=7)

    frame = provider.get_history(["AAPL"], end=date(2026, 1, 2))

    _assert_utc_timestamp_column(frame, "event_timestamp")
    _assert_utc_timestamp_column(frame, "availability_timestamp")
    assert frame.loc[0, "date"] == pd.Timestamp("2026-01-02")
    assert frame.loc[0, "event_timestamp"] == pd.Timestamp("2026-01-02 21:00:00", tz="UTC")
    assert frame.loc[0, "availability_timestamp"] == frame.loc[0, "event_timestamp"]
    assert pd.isna(frame.loc[0, "source_timestamp"])
    assert frame.loc[0, "timezone"] == "America/New_York"


def test_news_provider_normalizes_explicit_availability_and_blocks_future_available_item(
    tmp_path,
) -> None:
    path = tmp_path / "news_items.jsonl"
    rows = [
        {
            "ticker": "AAPL",
            "published_at": "2026-01-05T14:00:00Z",
            "availability_timestamp": "2026-01-05T14:00:00Z",
            "source_timestamp": "2026-01-05T14:00:00Z",
            "title": "AAPL earnings beat",
            "source": "unit",
            "summary": "earnings beat growth",
            "content": "earnings beat growth",
        },
        {
            "ticker": "AAPL",
            "published_at": "2026-01-05T14:30:00Z",
            "availability_timestamp": "2026-01-07T01:30:00Z",
            "source_timestamp": "2026-01-05T14:30:00Z",
            "title": "AAPL regulatory risk",
            "source": "unit",
            "summary": "regulatory risk investigation",
            "content": "regulatory risk investigation",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    provider = LocalNewsProvider(data_path=str(path))

    items = provider.get_news(["AAPL"], start=date(2026, 1, 5), end=date(2026, 1, 7))
    frame = news_items_to_frame(items)
    features = build_news_features(items)
    calendar = pd.DataFrame({"date": pd.bdate_range("2026-01-05", periods=3), "ticker": "AAPL"})
    expanded = expand_news_features_to_calendar(features, calendar, news_lag_periods=0)

    _assert_utc_timestamp_column(frame, "event_timestamp")
    _assert_utc_timestamp_column(frame, "availability_timestamp")
    assert frame["availability_timestamp"].tolist() == [
        pd.Timestamp("2026-01-05 14:00:00", tz="UTC"),
        pd.Timestamp("2026-01-07 01:30:00", tz="UTC"),
    ]
    before_available = expanded.loc[expanded["date"].eq(pd.Timestamp("2026-01-06"))].iloc[0]
    after_available = expanded.loc[expanded["date"].eq(pd.Timestamp("2026-01-07"))].iloc[0]
    assert before_available["news_article_count"] == 0.0
    assert after_available["news_article_count"] == 1.0
    assert after_available["news_availability_timestamp"] == pd.Timestamp("2026-01-07 01:30:00", tz="UTC")


def test_news_provider_sorts_by_published_and_collected_and_features_wait_for_collection(
    tmp_path,
) -> None:
    path = tmp_path / "news_items.jsonl"
    rows = [
        {
            "ticker": "AAPL",
            "published_at": "2026-01-05T13:00:00Z",
            "collected_at": "2026-01-07T13:30:00Z",
            "title": "AAPL late collected regulatory risk",
            "source": "unit",
            "summary": "regulatory risk investigation",
            "content": "regulatory risk investigation",
        },
        {
            "ticker": "AAPL",
            "published_at": "2026-01-05T12:00:00Z",
            "collected_at": "2026-01-05T13:30:00Z",
            "title": "AAPL earlier collected earnings beat",
            "source": "unit",
            "summary": "earnings beat growth",
            "content": "earnings beat growth",
        },
        {
            "ticker": "AAPL",
            "published_at": "2026-01-06T12:00:00Z",
            "collected_at": "2026-01-06T22:30:00Z",
            "title": "AAPL after-close guidance risk",
            "source": "unit",
            "summary": "guidance risk pressure",
            "content": "guidance risk pressure",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    provider = LocalNewsProvider(data_path=str(path))

    items = provider.get_news(["AAPL"], start=date(2026, 1, 5), end=date(2026, 1, 7))
    frame = news_items_to_frame(items)
    features = build_news_features(items)
    calendar = pd.DataFrame({"date": pd.bdate_range("2026-01-05", periods=3), "ticker": "AAPL"})
    expanded = expand_news_features_to_calendar(features, calendar, news_lag_periods=0)

    assert frame["title"].tolist() == [
        "AAPL earlier collected earnings beat",
        "AAPL late collected regulatory risk",
        "AAPL after-close guidance risk",
    ]
    assert frame["collected_at"].tolist() == [
        pd.Timestamp("2026-01-05 13:30:00", tz="UTC"),
        pd.Timestamp("2026-01-07 13:30:00", tz="UTC"),
        pd.Timestamp("2026-01-06 22:30:00", tz="UTC"),
    ]
    assert frame["availability_timestamp"].equals(frame["collected_at"])

    jan5 = expanded.loc[expanded["date"].eq(pd.Timestamp("2026-01-05"))].iloc[0]
    jan6 = expanded.loc[expanded["date"].eq(pd.Timestamp("2026-01-06"))].iloc[0]
    jan7 = expanded.loc[expanded["date"].eq(pd.Timestamp("2026-01-07"))].iloc[0]
    assert jan5["news_article_count"] == 1.0
    assert jan5["news_top_event"] == "earnings"
    assert jan6["news_article_count"] == 0.0
    assert pd.isna(jan6["news_availability_timestamp"])
    assert jan7["news_article_count"] == 2.0
    assert jan7["news_availability_timestamp"] == pd.Timestamp("2026-01-07 13:30:00", tz="UTC")


def test_news_item_normalizes_naive_provider_timestamps_to_utc() -> None:
    item = NewsItem(
        ticker="AAPL",
        published_at=pd.Timestamp("2026-01-05"),
        title="AAPL update",
        source="unit",
        source_timestamp=pd.Timestamp("2026-01-05 09:30:00"),
        availability_timestamp=pd.Timestamp("2026-01-05 09:45:00"),
    )

    frame = news_items_to_frame([item])

    assert frame.loc[0, "collected_at"] == pd.Timestamp("2026-01-05 09:45:00", tz="UTC")
    assert frame.loc[0, "event_timestamp"] == pd.Timestamp("2026-01-05 09:30:00", tz="UTC")
    assert frame.loc[0, "source_timestamp"] == pd.Timestamp("2026-01-05 09:30:00", tz="UTC")
    assert frame.loc[0, "availability_timestamp"] == pd.Timestamp("2026-01-05 09:45:00", tz="UTC")
    assert frame.loc[0, "timezone"] == "UTC"


def test_news_feature_trading_date_mapping_handles_utc_eastern_and_local_boundaries() -> None:
    items = [
        NewsItem(
            ticker="AAPL",
            published_at=pd.Timestamp("2026-01-06"),
            title="AAPL pre-close earnings beat",
            source="unit",
            content="earnings beat growth",
            source_timestamp=pd.Timestamp("2026-01-06 15:59:00", tz="America/New_York"),
            availability_timestamp=pd.Timestamp("2026-01-06 15:59:00", tz="America/New_York"),
            timezone="America/New_York",
        ),
        NewsItem(
            ticker="AAPL",
            published_at=pd.Timestamp("2026-01-06"),
            title="AAPL after-close regulatory risk",
            source="unit",
            content="regulatory risk investigation",
            source_timestamp=pd.Timestamp("2026-01-06 21:01:00", tz="UTC"),
            availability_timestamp=pd.Timestamp("2026-01-06 21:01:00", tz="UTC"),
            timezone="UTC",
        ),
        NewsItem(
            ticker="AAPL",
            published_at=pd.Timestamp("2026-01-07"),
            title="AAPL local-time guidance risk",
            source="unit",
            content="guidance risk pressure",
            source_timestamp=pd.Timestamp("2026-01-07 06:02:00", tz="Asia/Seoul"),
            availability_timestamp=pd.Timestamp("2026-01-07 06:02:00", tz="Asia/Seoul"),
            timezone="Asia/Seoul",
        ),
    ]
    calendar = pd.DataFrame({"date": pd.bdate_range("2026-01-06", periods=2), "ticker": "AAPL"})

    features = build_news_features(items)
    expanded = expand_news_features_to_calendar(features, calendar, news_lag_periods=0)

    same_day = expanded.loc[expanded["date"].eq(pd.Timestamp("2026-01-06"))].iloc[0]
    next_day = expanded.loc[expanded["date"].eq(pd.Timestamp("2026-01-07"))].iloc[0]
    assert same_day["news_article_count"] == 1
    assert same_day["news_availability_timestamp"] == pd.Timestamp("2026-01-06 20:59:00", tz="UTC")
    assert next_day["news_article_count"] == 2
    assert next_day["news_availability_timestamp"] == pd.Timestamp("2026-01-06 21:02:00", tz="UTC")
    assert set(features["date"]) == {pd.Timestamp("2026-01-06"), pd.Timestamp("2026-01-07")}


def test_sec_filing_uses_acceptance_time_for_availability_not_report_event_date() -> None:
    calendar = pd.DataFrame({"date": pd.bdate_range("2026-01-05", periods=4), "ticker": "AAPL"})
    filings = pd.DataFrame(
        {
            "filing_date": [pd.Timestamp("2026-01-06")],
            "report_date": [pd.Timestamp("2025-12-31")],
            "acceptance_datetime": [pd.Timestamp("2026-01-07 01:30:00", tz="UTC")],
            "form": ["10-Q"],
            "primary_document": ["q1.htm"],
            "accession_number": ["0000320193-26-000001"],
            "document_text": ["quarterly filing"],
        }
    )

    features = build_sec_features({"AAPL": filings}, {"AAPL": pd.DataFrame()}, calendar)

    before_available = features.loc[features["date"].eq(pd.Timestamp("2026-01-06"))].iloc[0]
    after_available = features.loc[features["date"].eq(pd.Timestamp("2026-01-07"))].iloc[0]
    assert before_available["sec_10q_count"] == 0.0
    assert after_available["sec_10q_count"] == 1.0
    assert after_available["sec_event_timestamp"] == pd.Timestamp("2025-12-31 23:59:59.999999999", tz="UTC")
    assert after_available["sec_availability_timestamp"] == pd.Timestamp("2026-01-07 01:30:00", tz="UTC")
    assert after_available["sec_source_timestamp"] == after_available["sec_availability_timestamp"]


def test_sec_filing_trading_date_mapping_handles_utc_eastern_and_local_boundaries() -> None:
    calendar = pd.DataFrame({"date": pd.bdate_range("2026-01-06", periods=2), "ticker": "AAPL"})
    filings = pd.DataFrame(
        {
            "filing_date": [
                pd.Timestamp("2026-01-06"),
                pd.Timestamp("2026-01-06"),
                pd.Timestamp("2026-01-07"),
            ],
            "report_date": [
                pd.Timestamp("2025-12-31"),
                pd.Timestamp("2025-12-31"),
                pd.Timestamp("2025-12-31"),
            ],
            "acceptance_datetime": [
                pd.Timestamp("2026-01-06 15:59:00", tz="America/New_York"),
                pd.Timestamp("2026-01-06 21:01:00", tz="UTC"),
                pd.Timestamp("2026-01-07 06:02:00", tz="Asia/Seoul"),
            ],
            "form": ["8-K", "10-Q", "4"],
            "primary_document": ["premarket.htm", "afterclose-q.htm", "afterclose-local.xml"],
            "accession_number": [
                "0000320193-26-000101",
                "0000320193-26-000102",
                "0000320193-26-000103",
            ],
            "document_text": [
                "pre-close current report",
                "after-close quarterly report",
                "local-time form 4",
            ],
        }
    )

    features = build_sec_features({"AAPL": filings}, {"AAPL": pd.DataFrame()}, calendar)

    same_day = features.loc[features["date"].eq(pd.Timestamp("2026-01-06"))].iloc[0]
    next_day = features.loc[features["date"].eq(pd.Timestamp("2026-01-07"))].iloc[0]
    assert same_day["sec_8k_count"] == 1.0
    assert same_day["sec_10q_count"] == 0.0
    assert same_day["sec_form4_count"] == 0.0
    assert same_day["sec_availability_timestamp"] == pd.Timestamp("2026-01-06 20:59:00", tz="UTC")
    assert next_day["sec_8k_count"] == 0.0
    assert next_day["sec_10q_count"] == 1.0
    assert next_day["sec_form4_count"] == 1.0
    assert next_day["sec_availability_timestamp"] == pd.Timestamp("2026-01-06 21:02:00", tz="UTC")


def test_sec_xbrl_facts_use_filing_date_availability_and_utc_date_end() -> None:
    payload = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-12-31",
                                "val": "100",
                                "fy": 2025,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2026-02-16",
                            }
                        ]
                    }
                }
            }
        }
    }

    facts = extract_companyfacts_frame(payload)

    _assert_utc_timestamp_column(facts, "event_timestamp")
    _assert_utc_timestamp_column(facts, "availability_timestamp")
    assert facts.loc[0, "event_timestamp"] == pd.Timestamp("2025-12-31 23:59:59.999999999", tz="UTC")
    assert facts.loc[0, "source_timestamp"] == pd.Timestamp("2026-02-16 23:59:59.999999999", tz="UTC")
    assert facts.loc[0, "availability_timestamp"] == facts.loc[0, "source_timestamp"]
    assert facts.loc[0, "timezone"] == "UTC"


def test_feature_availability_schema_standardizes_as_of_publication_and_available_times() -> None:
    raw = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "as_of": ["2026-01-05 16:00:00", "2026-01-05 16:00:00"],
            "published_at": [
                "2026-01-05T21:05:00Z",
                "2026-01-05T21:10:00Z",
            ],
            "sample_timestamp": [
                "2026-01-06T00:00:00Z",
                "2026-01-06T00:00:00Z",
            ],
            "timezone": ["America/New_York", "America/New_York"],
            "sentiment_score": [0.2, -0.1],
        }
    )

    standardized = standardize_feature_availability_metadata(
        raw,
        as_of_column="as_of",
        publication_column="published_at",
    )
    schema = FeatureAvailabilitySchema(source_family="text")
    result = validate_feature_availability(
        standardized,
        schema,
        feature_names=["sentiment_score"],
    )

    assert standardized.loc[0, "as_of_timestamp"] == pd.Timestamp("2026-01-05 21:00:00", tz="UTC")
    assert standardized.loc[0, "publication_timestamp"] == pd.Timestamp("2026-01-05 21:05:00", tz="UTC")
    assert standardized.loc[0, "availability_timestamp"] == standardized.loc[0, "publication_timestamp"]
    assert result.passed
    assert result.to_dict()["schema"]["schema_version"] == FEATURE_AVAILABILITY_SCHEMA_VERSION
    assert result.to_dict()["schema"]["cutoff_rule"] == "availability_timestamp <= sample_timestamp"


def test_feature_availability_validation_fails_future_available_rows() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "as_of_timestamp": [
                pd.Timestamp("2026-01-05 21:00:00", tz="UTC"),
                pd.Timestamp("2026-01-06 21:00:00", tz="UTC"),
            ],
            "publication_timestamp": [
                pd.Timestamp("2026-01-05 21:05:00", tz="UTC"),
                pd.Timestamp("2026-01-08 14:00:00", tz="UTC"),
            ],
            "availability_timestamp": [
                pd.Timestamp("2026-01-05 21:05:00", tz="UTC"),
                pd.Timestamp("2026-01-08 14:00:00", tz="UTC"),
            ],
            "sample_timestamp": [
                pd.Timestamp("2026-01-06 00:00:00", tz="UTC"),
                pd.Timestamp("2026-01-07 00:00:00", tz="UTC"),
            ],
            "timezone": ["UTC", "UTC"],
            "text_risk_score": [0.0, 1.0],
        }
    )

    result = validate_feature_availability(
        frame,
        FeatureAvailabilitySchema(source_family="text"),
        feature_names=["text_risk_score"],
    )

    assert not result.passed
    assert result.cutoff_violation_count == 1
    assert [issue.code for issue in result.issues] == ["cutoff_violation"]


def test_feature_availability_validation_flags_only_leaky_asset_date_rows() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "AAPL"],
            "feature_date": [
                pd.Timestamp("2026-01-06"),
                pd.Timestamp("2026-01-06"),
                pd.Timestamp("2026-01-07"),
            ],
            "as_of_timestamp": [
                pd.Timestamp("2026-01-06 21:00:00", tz="UTC"),
                pd.Timestamp("2026-01-06 21:00:00", tz="UTC"),
                pd.Timestamp("2026-01-07 21:00:00", tz="UTC"),
            ],
            "publication_timestamp": [
                pd.Timestamp("2026-01-08 01:00:00", tz="UTC"),
                pd.Timestamp("2026-01-06 21:05:00", tz="UTC"),
                pd.Timestamp("2026-01-07 21:05:00", tz="UTC"),
            ],
            "availability_timestamp": [
                pd.Timestamp("2026-01-08 01:00:00", tz="UTC"),
                pd.Timestamp("2026-01-06 21:05:00", tz="UTC"),
                pd.Timestamp("2026-01-07 21:05:00", tz="UTC"),
            ],
            "timezone": ["UTC", "UTC", "UTC"],
            "text_risk_score": [1.0, 0.0, 0.2],
        },
        index=["AAPL-2026-01-06", "MSFT-2026-01-06", "AAPL-2026-01-07"],
    )
    schema = FeatureAvailabilitySchema(
        source_family="text",
        sample_timestamp_column="feature_date",
    )

    result = validate_feature_availability(
        frame,
        schema,
        feature_names=["text_risk_score"],
        sample_timestamp_mode="date_end",
    )

    assert not result.passed
    assert result.cutoff_violation_count == 1
    assert [(issue.row_index, issue.code) for issue in result.issues] == [
        ("AAPL-2026-01-06", "cutoff_violation")
    ]


def test_feature_availability_validation_accepts_asset_date_rows_at_or_before_cutoff() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "feature_date": [
                pd.Timestamp("2026-01-06"),
                pd.Timestamp("2026-01-06"),
                pd.Timestamp("2026-01-07"),
                pd.Timestamp("2026-01-07"),
            ],
            "as_of_timestamp": [
                pd.Timestamp("2026-01-06 21:00:00", tz="UTC"),
                pd.Timestamp("2026-01-06 21:00:00", tz="UTC"),
                pd.Timestamp("2026-01-07 21:00:00", tz="UTC"),
                pd.Timestamp("2026-01-07 21:00:00", tz="UTC"),
            ],
            "publication_timestamp": [
                pd.Timestamp("2026-01-06 21:00:00", tz="UTC"),
                pd.Timestamp("2026-01-06 23:59:59", tz="UTC"),
                pd.Timestamp("2026-01-07 21:00:00", tz="UTC"),
                pd.Timestamp("2026-01-07 23:59:59", tz="UTC"),
            ],
            "availability_timestamp": [
                pd.Timestamp("2026-01-06 21:00:00", tz="UTC"),
                pd.Timestamp("2026-01-06 23:59:59", tz="UTC"),
                pd.Timestamp("2026-01-07 21:00:00", tz="UTC"),
                pd.Timestamp("2026-01-07 23:59:59", tz="UTC"),
            ],
            "timezone": ["UTC", "UTC", "UTC", "UTC"],
            "text_risk_score": [0.1, -0.1, 0.2, -0.2],
        },
        index=[
            "AAPL-2026-01-06",
            "MSFT-2026-01-06",
            "AAPL-2026-01-07",
            "MSFT-2026-01-07",
        ],
    )
    schema = FeatureAvailabilitySchema(
        source_family="text",
        sample_timestamp_column="feature_date",
    )

    result = validate_feature_availability(
        frame,
        schema,
        feature_names=["text_risk_score"],
        sample_timestamp_mode="date_end",
    )

    assert result.passed
    assert result.cutoff_violation_count == 0
    assert result.issues == ()


def test_filter_available_as_of_removes_provider_rows_after_sample_cutoff() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "sample_date": [
                pd.Timestamp("2026-01-05"),
                pd.Timestamp("2026-01-05"),
                pd.Timestamp("2026-01-06"),
            ],
            "availability_timestamp": [
                pd.Timestamp("2026-01-05 21:00:00", tz="UTC"),
                pd.Timestamp("2026-01-06 00:30:00", tz="UTC"),
                pd.Timestamp("2026-01-06 14:00:00", tz="UTC"),
            ],
            "value": [1.0, 2.0, 3.0],
        }
    )

    filtered = filter_available_as_of(
        frame,
        frame["sample_date"],
        sample_timestamp_mode="date_end",
    )

    assert filtered["value"].tolist() == [1.0, 3.0]


def test_price_feature_builder_excludes_late_provider_rows_before_deriving_features() -> None:
    dates = pd.bdate_range("2026-01-02", periods=30)
    price_data = pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAPL",
            "open": [100.0 + idx for idx in range(len(dates))],
            "high": [101.0 + idx for idx in range(len(dates))],
            "low": [99.0 + idx for idx in range(len(dates))],
            "close": [100.5 + idx for idx in range(len(dates))],
            "adj_close": [100.5 + idx for idx in range(len(dates))],
            "volume": [1_000_000 + idx * 1_000 for idx in range(len(dates))],
            "availability_timestamp": [
                pd.Timestamp(date_value).tz_localize("UTC") + pd.Timedelta(hours=20)
                for date_value in dates
            ],
        }
    )
    late_date = dates[10]
    price_data.loc[price_data["date"].eq(late_date), "availability_timestamp"] = pd.Timestamp(
        "2026-03-01 00:00:00",
        tz="UTC",
    )

    features = build_price_features(price_data)

    assert late_date not in set(features["date"])
    assert len(features) == len(dates) - 1


def test_feature_availability_validation_supports_date_end_sample_cutoffs() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "as_of_timestamp": [pd.Timestamp("2026-01-05 21:00:00", tz="UTC")],
            "publication_timestamp": [pd.Timestamp("2026-01-05 21:00:00", tz="UTC")],
            "availability_timestamp": [pd.Timestamp("2026-01-05 21:00:00", tz="UTC")],
            "sample_date": [pd.Timestamp("2026-01-05")],
            "timezone": ["UTC"],
            "return_20": [0.01],
        }
    )
    schema = FeatureAvailabilitySchema(
        source_family="price",
        sample_timestamp_column="sample_date",
    )

    result = validate_feature_availability(
        frame,
        schema,
        feature_names=["return_20"],
        sample_timestamp_mode="date_end",
    )

    assert result.passed


def test_feature_availability_validation_rejects_missing_required_metadata() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "as_of_timestamp": [pd.Timestamp("2026-01-05 21:00:00", tz="UTC")],
            "availability_timestamp": [pd.NaT],
            "sample_timestamp": [pd.Timestamp("2026-01-06 00:00:00", tz="UTC")],
            "timezone": ["UTC"],
            "revenue_growth": [0.03],
        }
    )

    result = validate_feature_availability(
        frame,
        FeatureAvailabilitySchema(source_family="sec"),
        feature_names=["revenue_growth"],
    )

    assert not result.passed
    assert result.null_counts["availability_timestamp"] == 1
    assert result.issues[0].code == "null_required_metadata"


def _assert_utc_timestamp_column(frame: pd.DataFrame, column: str) -> None:
    assert isinstance(frame[column].dtype, pd.DatetimeTZDtype)
    assert str(frame[column].dt.tz) == "UTC"
    assert frame[column].notna().all()
