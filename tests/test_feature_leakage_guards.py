from __future__ import annotations

import pandas as pd
import pytest

from quant_research.backtest import align_backtest_horizon_inputs
from quant_research.backtest.engine import BacktestConfig, run_long_only_backtest
from quant_research.data.news import NewsItem
from quant_research.data.timestamps import validate_generated_feature_cutoffs
from quant_research.features.fusion import fuse_features
from quant_research.features.price import build_price_features
from quant_research.features.sec import build_sec_features
from quant_research.features.text import build_news_features, expand_news_features_to_calendar
from quant_research.signals.engine import DeterministicSignalEngine
from quant_research.validation.walk_forward import WalkForwardConfig, walk_forward_predict


def test_price_feature_rows_do_not_change_when_future_prices_are_mutated() -> None:
    dates = pd.bdate_range("2026-01-02", periods=80)
    price_data = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["AAPL"] * len(dates) + ["MSFT"] * len(dates),
            "open": [100.0 + idx for idx in range(len(dates))] + [200.0 + idx for idx in range(len(dates))],
            "high": [101.0 + idx for idx in range(len(dates))] + [202.0 + idx for idx in range(len(dates))],
            "low": [99.0 + idx for idx in range(len(dates))] + [198.0 + idx for idx in range(len(dates))],
            "close": [100.5 + idx for idx in range(len(dates))] + [201.0 + idx for idx in range(len(dates))],
            "adj_close": [100.5 + idx for idx in range(len(dates))] + [201.0 + idx for idx in range(len(dates))],
            "volume": [1_000_000 + idx * 1_000 for idx in range(len(dates))]
            + [2_000_000 + idx * 1_000 for idx in range(len(dates))],
        }
    )
    cutoff = dates[45]
    mutated = price_data.copy()
    future_mask = pd.to_datetime(mutated["date"]) > cutoff
    mutated.loc[future_mask, ["open", "high", "low", "close", "adj_close"]] *= 100.0
    mutated.loc[future_mask, "volume"] *= 50

    baseline = build_price_features(price_data)
    changed = build_price_features(mutated)

    feature_columns = [
        column
        for column in baseline.columns
        if not column.startswith("forward_return_")
    ]
    _assert_prefix_equal(baseline, changed, cutoff, feature_columns)


def test_forward_return_20_is_future_label_but_not_price_feature_input() -> None:
    dates = pd.bdate_range("2026-01-02", periods=80)
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
        }
    )
    decision_date = dates[30]
    mutated = price_data.copy()
    future_mask = pd.to_datetime(mutated["date"]) == dates[50]
    mutated.loc[future_mask, ["open", "high", "low", "close", "adj_close"]] *= 3.0

    baseline = build_price_features(price_data)
    changed = build_price_features(mutated)

    feature_columns = [
        column
        for column in baseline.columns
        if not column.startswith("forward_return_")
    ]
    _assert_prefix_equal(baseline, changed, decision_date, feature_columns)

    baseline_label = baseline.loc[
        baseline["date"].eq(decision_date),
        "forward_return_20",
    ].iloc[0]
    changed_label = changed.loc[
        changed["date"].eq(decision_date),
        "forward_return_20",
    ].iloc[0]
    assert changed_label != baseline_label


def test_price_feature_builder_drops_rows_not_available_by_sample_date() -> None:
    dates = pd.bdate_range("2026-01-02", periods=30)
    baseline = pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAPL",
            "open": [100.0 + idx for idx in range(len(dates))],
            "high": [101.0 + idx for idx in range(len(dates))],
            "low": [99.0 + idx for idx in range(len(dates))],
            "close": [100.5 + idx for idx in range(len(dates))],
            "adj_close": [100.5 + idx for idx in range(len(dates))],
            "volume": [1_000_000 + idx * 1_000 for idx in range(len(dates))],
        }
    )
    available = baseline.assign(
        availability_timestamp=[
            pd.Timestamp(date).tz_localize("UTC") + pd.Timedelta(hours=20)
            for date in dates
        ]
    )
    late_revision = available.copy()
    late_revision.loc[10, "adj_close"] = 10_000.0
    late_revision.loc[10, "close"] = 10_000.0
    late_revision.loc[10, "availability_timestamp"] = pd.Timestamp(
        "2026-02-20 12:00:00",
        tz="UTC",
    )

    without_late_revision = build_price_features(available.drop(index=10))
    with_late_revision = build_price_features(late_revision)

    feature_columns = [
        column
        for column in without_late_revision.columns
        if not column.startswith("forward_return_")
    ]
    _assert_prefix_equal(
        without_late_revision,
        with_late_revision,
        dates[-1],
        feature_columns,
    )


def test_price_feature_builder_uses_only_revision_available_by_sample_date() -> None:
    dates = pd.bdate_range("2026-01-02", periods=35)
    baseline = pd.DataFrame(
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
                pd.Timestamp(date).tz_localize("UTC") + pd.Timedelta(hours=20)
                for date in dates
            ],
        }
    )
    revision_date = dates[12]
    future_revision = baseline.loc[baseline["date"].eq(revision_date)].copy()
    future_revision[["open", "high", "low", "close", "adj_close"]] = 10_000.0
    future_revision["volume"] = 99_000_000
    future_revision["availability_timestamp"] = pd.Timestamp("2026-03-01 12:00:00", tz="UTC")

    unrevised = build_price_features(baseline)
    with_future_revision = build_price_features(
        pd.concat([baseline, future_revision], ignore_index=True)
    )

    feature_columns = [
        column
        for column in unrevised.columns
        if not column.startswith("forward_return_")
    ]
    _assert_prefix_equal(unrevised, with_future_revision, dates[-1], feature_columns)


def test_text_feature_rows_do_not_change_when_future_news_is_added() -> None:
    calendar = pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-02", periods=25),
            "ticker": "AAPL",
        }
    )
    cutoff = pd.Timestamp("2026-01-20")
    observed_items = [
        NewsItem(
            ticker="AAPL",
            published_at=pd.Timestamp("2026-01-05"),
            title="AAPL beats earnings expectations",
            source="unit",
            summary="beats earnings growth",
            content="beats earnings growth",
        ),
        NewsItem(
            ticker="AAPL",
            published_at=pd.Timestamp("2026-01-15"),
            title="AAPL faces regulatory risk",
            source="unit",
            summary="regulatory risk investigation",
            content="regulatory risk investigation",
        ),
    ]
    future_items = [
        NewsItem(
            ticker="AAPL",
            published_at=pd.Timestamp("2026-01-23"),
            title="AAPL announces strong buyback",
            source="unit",
            summary="strong buyback profit growth",
            content="strong buyback profit growth",
        )
    ]

    baseline = expand_news_features_to_calendar(build_news_features(observed_items), calendar)
    with_future_news = expand_news_features_to_calendar(
        build_news_features([*observed_items, *future_items]),
        calendar,
    )

    _assert_prefix_equal(baseline, with_future_news, cutoff, list(baseline.columns))


def test_text_event_after_market_close_is_not_in_same_day_feature() -> None:
    calendar = pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-06", periods=2),
            "ticker": "AAPL",
        }
    )
    items = [
        NewsItem(
            ticker="AAPL",
            published_at=pd.Timestamp("2026-01-06"),
            title="AAPL beats earnings before close",
            source="unit",
            summary="beats earnings growth",
            content="beats earnings growth",
            availability_timestamp=pd.Timestamp("2026-01-06 20:59:00", tz="UTC"),
            source_timestamp=pd.Timestamp("2026-01-06 20:59:00", tz="UTC"),
        ),
        NewsItem(
            ticker="AAPL",
            published_at=pd.Timestamp("2026-01-06"),
            title="AAPL faces regulatory risk after close",
            source="unit",
            summary="regulatory risk investigation",
            content="regulatory risk investigation",
            availability_timestamp=pd.Timestamp("2026-01-06 21:30:00", tz="UTC"),
            source_timestamp=pd.Timestamp("2026-01-06 21:30:00", tz="UTC"),
        ),
    ]

    features = build_news_features(items)
    expanded = expand_news_features_to_calendar(features, calendar, news_lag_periods=0)

    same_day = expanded.loc[expanded["date"].eq(pd.Timestamp("2026-01-06"))].iloc[0]
    next_day = expanded.loc[expanded["date"].eq(pd.Timestamp("2026-01-07"))].iloc[0]
    assert same_day["news_article_count"] == 1.0
    assert same_day["news_availability_timestamp"] == pd.Timestamp("2026-01-06 20:59:00", tz="UTC")
    assert same_day["news_top_event"] == "earnings"
    assert next_day["news_article_count"] == 1.0
    assert next_day["news_availability_timestamp"] == pd.Timestamp("2026-01-06 21:30:00", tz="UTC")
    assert next_day["news_top_event"] == "legal"


def test_text_features_ignore_news_text_until_availability_timestamp() -> None:
    calendar = pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-05", periods=20),
            "ticker": "AAPL",
        }
    )
    cutoff = pd.Timestamp("2026-01-21")
    observed_item = NewsItem(
        ticker="AAPL",
        published_at=pd.Timestamp("2026-01-06"),
        title="AAPL beats earnings expectations",
        source="unit",
        summary="beats earnings growth",
        content="beats earnings growth",
        availability_timestamp=pd.Timestamp("2026-01-06 15:00:00", tz="UTC"),
        source_timestamp=pd.Timestamp("2026-01-06 15:00:00", tz="UTC"),
    )
    late_negative_text = NewsItem(
        ticker="AAPL",
        published_at=pd.Timestamp("2026-01-07"),
        title="AAPL article with delayed body text",
        source="unit",
        summary="placeholder",
        content="regulatory risk investigation downgrade",
        availability_timestamp=pd.Timestamp("2026-01-22 15:00:00", tz="UTC"),
        source_timestamp=pd.Timestamp("2026-01-07 15:00:00", tz="UTC"),
    )
    late_positive_text = NewsItem(
        ticker="AAPL",
        published_at=pd.Timestamp("2026-01-07"),
        title="AAPL article with delayed body text",
        source="unit",
        summary="placeholder",
        content="strong buyback profit growth",
        availability_timestamp=pd.Timestamp("2026-01-22 15:00:00", tz="UTC"),
        source_timestamp=pd.Timestamp("2026-01-07 15:00:00", tz="UTC"),
    )

    negative_variant = expand_news_features_to_calendar(
        build_news_features([observed_item, late_negative_text]),
        calendar,
        news_lag_periods=0,
    )
    positive_variant = expand_news_features_to_calendar(
        build_news_features([observed_item, late_positive_text]),
        calendar,
        news_lag_periods=0,
    )

    _assert_prefix_equal(negative_variant, positive_variant, cutoff, list(negative_variant.columns))
    leak_date = pd.Timestamp("2026-01-22")
    negative_row = negative_variant.loc[negative_variant["date"].eq(leak_date)].iloc[0]
    positive_row = positive_variant.loc[positive_variant["date"].eq(leak_date)].iloc[0]
    assert negative_row["news_sentiment_mean"] < positive_row["news_sentiment_mean"]
    assert negative_row["text_risk_score"] > positive_row["text_risk_score"]


def test_sec_feature_rows_do_not_change_when_future_filings_and_facts_are_added() -> None:
    calendar = pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-02", periods=80),
            "ticker": "AAPL",
        }
    )
    cutoff = pd.Timestamp("2026-02-13")
    observed_filings = pd.DataFrame(
        {
            "filing_date": ["2026-01-08", "2026-02-10"],
            "form": ["10-Q", "8-K"],
            "primary_document": ["q1.htm", "event.htm"],
            "accession_number": ["0000320193-26-000001", "0000320193-26-000002"],
            "document_text": [
                "Item 1 quarterly results",
                "Item 1.01 current report with litigation risk",
            ],
        }
    )
    future_filings = pd.DataFrame(
        {
            "filing_date": ["2026-02-20"],
            "form": ["10-K"],
            "primary_document": ["annual.htm"],
            "accession_number": ["0000320193-26-000003"],
            "document_text": ["Item 1A future annual risk factors and investigation"],
        }
    )
    observed_facts = pd.DataFrame(
        {
            "period_end": pd.to_datetime(["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]),
            "revenue": [100.0, 102.0, 106.0, 110.0],
            "net_income": [10.0, 11.0, 11.5, 12.0],
            "assets": [500.0, 510.0, 520.0, 535.0],
        }
    )
    future_facts = pd.DataFrame(
        {
            "period_end": pd.to_datetime(["2026-03-31"]),
            "revenue": [999.0],
            "net_income": [-50.0],
            "assets": [10_000.0],
        }
    )

    baseline = build_sec_features(
        {"AAPL": observed_filings},
        {"AAPL": observed_facts},
        calendar,
    )
    with_future_sec = build_sec_features(
        {"AAPL": pd.concat([observed_filings, future_filings], ignore_index=True)},
        {"AAPL": pd.concat([observed_facts, future_facts], ignore_index=True)},
        calendar,
    )

    _assert_prefix_equal(baseline, with_future_sec, cutoff, list(baseline.columns))


def test_sec_features_drop_filings_with_future_report_date_available_too_early() -> None:
    calendar = pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-05", periods=6),
            "ticker": "AAPL",
        }
    )
    filings = pd.DataFrame(
        {
            "filing_date": ["2026-01-06", "2026-01-07"],
            "report_date": ["2025-12-31", "2026-03-31"],
            "acceptance_datetime": [
                pd.Timestamp("2026-01-06 21:30:00", tz="UTC"),
                pd.Timestamp("2026-01-07 21:30:00", tz="UTC"),
            ],
            "form": ["10-Q", "10-K"],
            "primary_document": ["q1.htm", "future.htm"],
            "accession_number": ["0000320193-26-000001", "0000320193-26-000002"],
            "document_text": ["quarterly filing", "future annual report should be ignored"],
        }
    )

    features = build_sec_features({"AAPL": filings}, {"AAPL": pd.DataFrame()}, calendar)

    assert features["sec_10q_count"].sum() == 1.0
    assert features["sec_10k_count"].sum() == 0.0
    assert not features["sec_event_timestamp"].dropna().gt(
        features["sec_availability_timestamp"].dropna().max()
    ).any()


def test_sec_features_drop_facts_available_before_period_end_boundary() -> None:
    calendar = pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-05", periods=8),
            "ticker": "AAPL",
        }
    )
    facts = pd.DataFrame(
        {
            "period_end": pd.to_datetime(["2025-09-30", "2026-03-31"]),
            "filed": ["2026-01-05", "2026-01-06"],
            "revenue": [100.0, 999.0],
            "net_income": [10.0, -100.0],
            "assets": [500.0, 10_000.0],
        }
    )

    features = build_sec_features({"AAPL": pd.DataFrame()}, {"AAPL": facts}, calendar)

    assert features["revenue_growth"].eq(0.0).all()
    assert features["net_income_growth"].eq(0.0).all()
    assert features["assets_growth"].eq(0.0).all()
    fact_rows = features.dropna(subset=["sec_fact_event_timestamp", "sec_fact_availability_timestamp"])
    assert not fact_rows.empty
    assert fact_rows["sec_fact_availability_timestamp"].ge(fact_rows["sec_fact_event_timestamp"]).all()


def test_fused_feature_pipeline_rejects_future_available_feature_rows() -> None:
    price_features = pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-05", periods=3),
            "ticker": "AAPL",
            "return_1": [0.0, 0.01, 0.02],
            "forward_return_20": [0.03, 0.02, 0.01],
        }
    )
    news_features = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-05")],
            "ticker": ["AAPL"],
            "news_event_timestamp": [pd.Timestamp("2026-01-05 14:00:00", tz="UTC")],
            "news_availability_timestamp": [pd.Timestamp("2026-01-07 14:00:00", tz="UTC")],
            "news_source_timestamp": [pd.Timestamp("2026-01-05 14:00:00", tz="UTC")],
            "news_timezone": ["UTC"],
            "news_article_count": [1.0],
            "news_sentiment_mean": [0.3],
            "news_negative_ratio": [0.0],
            "news_source_count": [1.0],
            "news_source_diversity": [1.0],
            "news_event_count": [1.0],
            "news_top_event": ["earnings"],
            "text_risk_score": [0.0],
            "news_confidence_mean": [0.8],
            "news_token_count_mean": [4.0],
            "news_text_length": [20.0],
            "news_full_text_available_ratio": [1.0],
        }
    )
    empty_sec_features = pd.DataFrame(columns=["date", "ticker"])

    with pytest.raises(ValueError, match="before public availability feature date 2026-01-07"):
        fuse_features(price_features, news_features, empty_sec_features)


def test_fused_feature_pipeline_rejects_text_features_before_market_public_availability() -> None:
    price_features = pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-06", periods=3),
            "ticker": "AAPL",
            "return_1": [0.0, 0.01, 0.02],
            "forward_return_20": [0.03, 0.02, 0.01],
        }
    )
    news_features = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-06")],
            "ticker": ["AAPL"],
            "news_event_timestamp": [pd.Timestamp("2026-01-06 20:30:00", tz="UTC")],
            "news_availability_timestamp": [pd.Timestamp("2026-01-06 21:30:00", tz="UTC")],
            "news_source_timestamp": [pd.Timestamp("2026-01-06 20:30:00", tz="UTC")],
            "news_timezone": ["UTC"],
            "news_article_count": [1.0],
            "news_sentiment_mean": [-0.4],
            "news_negative_ratio": [1.0],
            "news_source_count": [1.0],
            "news_source_diversity": [1.0],
            "news_event_count": [1.0],
            "news_top_event": ["legal"],
            "text_risk_score": [1.0],
            "news_confidence_mean": [0.8],
            "news_token_count_mean": [4.0],
            "news_text_length": [32.0],
            "news_full_text_available_ratio": [1.0],
        }
    )
    empty_sec_features = pd.DataFrame(columns=["date", "ticker"])

    with pytest.raises(ValueError, match="before public availability feature date 2026-01-07"):
        fuse_features(price_features, news_features, empty_sec_features)


def test_fused_feature_pipeline_merges_text_features_on_public_availability_date() -> None:
    price_features = pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-06", periods=3),
            "ticker": "AAPL",
            "return_1": [0.0, 0.01, 0.02],
            "forward_return_20": [0.03, 0.02, 0.01],
        }
    )
    news_features = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-07")],
            "ticker": ["AAPL"],
            "news_event_timestamp": [pd.Timestamp("2026-01-06 20:30:00", tz="UTC")],
            "news_availability_timestamp": [pd.Timestamp("2026-01-06 21:30:00", tz="UTC")],
            "news_source_timestamp": [pd.Timestamp("2026-01-06 20:30:00", tz="UTC")],
            "news_timezone": ["UTC"],
            "news_article_count": [1.0],
            "news_sentiment_mean": [-0.4],
            "news_negative_ratio": [1.0],
            "news_source_count": [1.0],
            "news_source_diversity": [1.0],
            "news_event_count": [1.0],
            "news_top_event": ["legal"],
            "text_risk_score": [1.0],
            "news_confidence_mean": [0.8],
            "news_token_count_mean": [4.0],
            "news_text_length": [32.0],
            "news_full_text_available_ratio": [1.0],
        }
    )
    empty_sec_features = pd.DataFrame(columns=["date", "ticker"])

    fused = fuse_features(price_features, news_features, empty_sec_features)

    before_available = fused.loc[fused["date"].eq(pd.Timestamp("2026-01-06"))].iloc[0]
    public_availability_day = fused.loc[fused["date"].eq(pd.Timestamp("2026-01-07"))].iloc[0]
    lagged_signal_day = fused.loc[fused["date"].eq(pd.Timestamp("2026-01-08"))].iloc[0]
    assert before_available["news_sentiment_mean"] == 0.0
    assert before_available["news_top_event"] == "none"
    assert before_available["text_risk_score"] == 0.0
    assert public_availability_day["news_sentiment_mean"] == 0.0
    assert public_availability_day["news_top_event"] == "none"
    assert public_availability_day["text_risk_score"] == 0.0
    assert lagged_signal_day["news_sentiment_mean"] == -0.4
    assert lagged_signal_day["news_top_event"] == "legal"
    assert lagged_signal_day["text_risk_score"] == 1.0


def test_generated_feature_cutoff_accepts_exact_date_end_and_rejects_one_nanosecond_after() -> None:
    valid = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-05")],
            "ticker": ["AAPL"],
            "news_availability_timestamp": [
                pd.Timestamp("2026-01-05 23:59:59.999999999", tz="UTC")
            ],
            "news_sentiment_mean": [0.1],
        }
    )
    leaky = valid.copy()
    leaky["news_availability_timestamp"] = leaky["news_availability_timestamp"] + pd.Timedelta(
        nanoseconds=1
    )

    validate_generated_feature_cutoffs(valid, label="unit feature frame")
    with pytest.raises(ValueError, match="unavailable at feature date"):
        validate_generated_feature_cutoffs(leaky, label="unit feature frame")


def test_generated_feature_cutoff_checks_all_prefixed_availability_columns() -> None:
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")],
            "ticker": ["AAPL", "AAPL"],
            "model_feature_availability_timestamp": [
                pd.Timestamp("2026-01-05 18:00:00", tz="UTC"),
                pd.Timestamp("2026-01-07 00:00:00", tz="UTC"),
            ],
            "model_score": [0.1, 0.2],
        }
    )

    with pytest.raises(
        ValueError,
        match="model_feature_availability_timestamp.*unavailable at feature date 2026-01-06",
    ):
        validate_generated_feature_cutoffs(frame, label="unit model features")


def test_walk_forward_rejects_features_available_after_sample_date() -> None:
    dates = pd.bdate_range("2026-01-05", periods=8)
    frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAPL",
            "momentum_20": [0.01 * idx for idx in range(len(dates))],
            "text_risk_score": [0.0] * len(dates),
            "text_availability_timestamp": [
                pd.Timestamp(date).tz_localize("UTC") + pd.Timedelta(hours=20)
                for date in dates
            ],
            "forward_return_20": [0.02] * len(dates),
        }
    )
    frame.loc[3, "text_availability_timestamp"] = pd.Timestamp("2026-01-12 00:00:00", tz="UTC")

    with pytest.raises(ValueError, match="text_availability_timestamp.*unavailable at feature date"):
        walk_forward_predict(
            frame,
            WalkForwardConfig(train_periods=3, test_periods=2, embargo_periods=20),
            target="forward_return_20",
        )


def test_walk_forward_rejects_text_or_sec_event_timestamp_after_availability() -> None:
    dates = pd.bdate_range("2026-01-06", periods=8)
    frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAPL",
            "momentum_20": [0.01 * idx for idx in range(len(dates))],
            "sec_risk_flag": [0.0] * len(dates),
            "sec_event_timestamp": [
                pd.Timestamp("2026-01-06 12:00:00", tz="UTC")
            ]
            * len(dates),
            "sec_availability_timestamp": [
                pd.Timestamp("2026-01-06 10:00:00", tz="UTC")
            ]
            * len(dates),
            "forward_return_20": [0.02] * len(dates),
        }
    )

    with pytest.raises(ValueError, match="sec_event_timestamp.*after sec_availability_timestamp"):
        walk_forward_predict(
            frame,
            WalkForwardConfig(train_periods=3, test_periods=2, embargo_periods=20),
            target="forward_return_20",
        )


def test_signal_generation_rows_do_not_change_when_future_inputs_are_mutated() -> None:
    dates = pd.bdate_range("2026-01-05", periods=8)
    frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAPL",
            "expected_return": [0.006, 0.007, 0.008, 0.006, 0.007, 0.015, -0.020, 0.030],
            "predicted_volatility": [0.020] * len(dates),
            "downside_quantile": [-0.020] * len(dates),
            "text_risk_score": [0.0] * len(dates),
            "sec_risk_flag": [0.0] * len(dates),
            "sec_risk_flag_20d": [0.0] * len(dates),
            "news_negative_ratio": [0.0] * len(dates),
            "model_confidence": [0.8] * len(dates),
            "liquidity_score": [20.0] * len(dates),
            "model_prediction_timestamp": [
                pd.Timestamp(date).tz_localize("UTC") + pd.Timedelta(hours=20)
                for date in dates
            ],
        }
    )
    cutoff = dates[4]
    mutated = frame.copy()
    future_mask = pd.to_datetime(mutated["date"]) > cutoff
    mutated.loc[future_mask, "expected_return"] = [-0.50, 0.80, -0.40]
    mutated.loc[future_mask, "predicted_volatility"] = [0.50, 0.01, 0.60]
    mutated.loc[future_mask, "text_risk_score"] = [1.0, 0.0, 1.0]
    mutated.loc[future_mask, "sec_risk_flag"] = [3.0, 0.0, 3.0]

    engine = DeterministicSignalEngine()
    baseline = engine.generate(frame)
    changed = engine.generate(mutated)

    _assert_prefix_equal(
        baseline,
        changed,
        cutoff,
        [
            "date",
            "ticker",
            "signal_score",
            "risk_metric_penalty",
            "action",
            "model_prediction_timestamp",
        ],
    )


def test_signal_generation_rejects_predictions_created_after_signal_date() -> None:
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")],
            "ticker": ["AAPL", "AAPL"],
            "expected_return": [0.010, 0.011],
            "predicted_volatility": [0.020, 0.020],
            "downside_quantile": [-0.020, -0.020],
            "text_risk_score": [0.0, 0.0],
            "sec_risk_flag": [0.0, 0.0],
            "sec_risk_flag_20d": [0.0, 0.0],
            "news_negative_ratio": [0.0, 0.0],
            "model_confidence": [0.8, 0.8],
            "liquidity_score": [20.0, 20.0],
            "model_prediction_timestamp": [
                pd.Timestamp("2026-01-05 20:00:00", tz="UTC"),
                pd.Timestamp("2026-01-07 00:00:00", tz="UTC"),
            ],
        }
    )

    with pytest.raises(ValueError, match="model_prediction_timestamp.*later than signal date"):
        DeterministicSignalEngine().generate(frame)


def test_integrated_multimodal_walk_forward_and_backtest_timing_is_point_in_time() -> None:
    dates = pd.bdate_range("2026-01-05", periods=45)
    price_data = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "open": base + idx,
                "high": base + idx + 1.0,
                "low": base + idx - 1.0,
                "close": base + idx + 0.5,
                "adj_close": base + idx + 0.5,
                "volume": 1_000_000 + idx * 1_000,
            }
            for ticker, base in (("AAPL", 100.0), ("MSFT", 200.0))
            for idx, date in enumerate(dates)
        ]
    )
    calendar = price_data[["date", "ticker"]]
    news_items = [
        NewsItem(
            ticker="AAPL",
            published_at=pd.Timestamp("2026-01-06"),
            title="AAPL legal risk emerges after close",
            source="unit",
            summary="lawsuit regulatory investigation risk",
            content="lawsuit regulatory investigation risk",
            availability_timestamp=pd.Timestamp("2026-01-06 21:30:00", tz="UTC"),
            source_timestamp=pd.Timestamp("2026-01-06 21:30:00", tz="UTC"),
        )
    ]
    sec_filings = pd.DataFrame(
        {
            "filing_date": ["2026-01-06"],
            "acceptance_datetime": [pd.Timestamp("2026-01-06 21:30:00", tz="UTC")],
            "form": ["8-K"],
            "primary_document": ["event.htm"],
            "accession_number": ["0000320193-26-000001"],
            "document_text": ["Item 1.01 litigation risk investigation"],
        }
    )

    price_features = build_price_features(price_data)
    news_features = build_news_features(news_items)
    sec_features = build_sec_features({"AAPL": sec_filings}, {}, calendar)
    fused = fuse_features(price_features, news_features, sec_features)

    jan6 = fused.loc[
        fused["date"].eq(pd.Timestamp("2026-01-06")) & fused["ticker"].eq("AAPL")
    ].iloc[0]
    jan7 = fused.loc[
        fused["date"].eq(pd.Timestamp("2026-01-07")) & fused["ticker"].eq("AAPL")
    ].iloc[0]
    jan8 = fused.loc[
        fused["date"].eq(pd.Timestamp("2026-01-08")) & fused["ticker"].eq("AAPL")
    ].iloc[0]
    assert jan6["sec_risk_flag"] == 0.0
    assert jan6["text_risk_score"] == 0.0
    assert jan7["sec_risk_flag"] == 1.0
    assert jan7["text_risk_score"] == 0.0
    assert jan8["text_risk_score"] == 1.0
    assert jan7["sec_event_timestamp"] <= jan7["sec_availability_timestamp"]
    assert jan8["news_event_timestamp"] <= jan8["news_availability_timestamp"]

    predictions, summary = walk_forward_predict(
        fused,
        WalkForwardConfig(
            train_periods=3,
            test_periods=2,
            gap_periods=20,
            embargo_periods=20,
            prediction_horizon_periods=20,
            min_train_observations=4,
        ),
        target="forward_return_20",
    )
    assert not predictions.empty
    assert not summary.empty
    assert summary["purge_applied"].all()
    assert summary["embargo_applied"].all()

    backtest_input = align_backtest_horizon_inputs(
        fused.assign(expected_return=fused["forward_return_20"].fillna(0.0)),
        return_column="forward_return_20",
    )
    assert not backtest_input.empty
    assert backtest_input["holding_start_date"].gt(backtest_input["signal_date"]).all()
    assert backtest_input["return_label_date"].gt(backtest_input["signal_date"]).all()
    assert backtest_input["realized_return_column"].eq("forward_return_20").all()

    result = run_long_only_backtest(
        backtest_input,
        BacktestConfig(top_n=2, realized_return_column="forward_return_20"),
    )

    assert not result.equity_curve.empty
    assert result.equity_curve["holding_start_date"].gt(result.equity_curve["date"]).all()
    assert result.equity_curve["return_date"].gt(result.equity_curve["date"]).all()


def _assert_prefix_equal(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    cutoff: pd.Timestamp,
    columns: list[str],
) -> None:
    ordered_columns = ["date", "ticker", *[column for column in columns if column not in {"date", "ticker"}]]
    left = _prefix(baseline, cutoff, ordered_columns)
    right = _prefix(candidate, cutoff, ordered_columns)
    pd.testing.assert_frame_equal(left, right, check_dtype=False)


def _prefix(frame: pd.DataFrame, cutoff: pd.Timestamp, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    return (
        result.loc[result["date"] <= cutoff, columns]
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
