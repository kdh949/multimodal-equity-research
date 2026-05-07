from __future__ import annotations

import pandas as pd

from quant_research.data.timestamps import add_price_timestamps
from quant_research.data.market import SyntheticMarketDataProvider
from quant_research.features.price import build_price_features
from quant_research.validation.horizons import DEFAULT_VALIDATION_HORIZONS, forward_return_column


def test_forward_return_uses_next_period_only() -> None:
    data = SyntheticMarketDataProvider(periods=40, seed=1).get_history(["AAPL"])
    features = build_price_features(data)
    features = features.sort_values("date").reset_index(drop=True)

    expected = features["adj_close"].pct_change().shift(-1)
    pd.testing.assert_series_equal(
        features["forward_return_1"],
        expected,
        check_names=False,
    )
    assert pd.isna(features["forward_return_1"].iloc[-1])
    expected_5 = features["adj_close"].pct_change(5).shift(-5)
    pd.testing.assert_series_equal(
        features["forward_return_5"],
        expected_5,
        check_names=False,
    )
    assert features["forward_return_5"].tail(5).isna().all()


def test_default_forward_return_horizons_are_generated_without_lookahead() -> None:
    data = SyntheticMarketDataProvider(periods=45, seed=1).get_history(["AAPL"])
    features = build_price_features(data).sort_values("date").reset_index(drop=True)

    assert DEFAULT_VALIDATION_HORIZONS == (1, 5, 20)
    for horizon in DEFAULT_VALIDATION_HORIZONS:
        column = forward_return_column(horizon)
        expected = features["adj_close"].shift(-horizon) / features["adj_close"] - 1
        pd.testing.assert_series_equal(features[column], expected, check_names=False)
        assert features[column].tail(horizon).isna().all()


def test_price_features_include_liquidity_and_volatility() -> None:
    data = SyntheticMarketDataProvider(periods=80, seed=2).get_history(["AAPL", "MSFT"])
    features = build_price_features(data)

    assert {
        "volatility_20",
        "liquidity_score",
        "rsi_14",
        "ma_ratio_20",
        "forward_return_5",
        "forward_return_20",
    }.issubset(features.columns)
    assert features.groupby("ticker")["forward_return_1"].apply(lambda s: s.notna().sum()).min() > 0


def test_price_and_return_features_do_not_reference_prices_after_feature_date() -> None:
    data = _deterministic_price_data(periods=90, tickers=("AAPL", "MSFT"))
    cutoff = pd.Timestamp("2026-02-17")
    mutated = data.copy()
    future_mask = pd.to_datetime(mutated["date"]) > cutoff
    mutated.loc[future_mask, ["open", "high", "low", "close", "adj_close"]] *= 100.0
    mutated.loc[future_mask, "volume"] *= 25

    baseline = build_price_features(data)
    changed = build_price_features(mutated)

    price_feature_columns = [
        column
        for column in baseline.columns
        if not column.startswith("forward_return_")
        and column not in {"event_timestamp", "availability_timestamp", "source_timestamp", "timezone"}
    ]
    pd.testing.assert_frame_equal(
        _feature_prefix(baseline, cutoff, price_feature_columns),
        _feature_prefix(changed, cutoff, price_feature_columns),
        check_dtype=False,
    )


def test_forward_returns_are_labels_not_price_feature_inputs() -> None:
    data = _deterministic_price_data(periods=70, tickers=("AAPL",))
    decision_date = pd.Timestamp("2026-02-13")
    mutated = data.copy()
    label_window_date = pd.Timestamp("2026-03-13")
    mutated.loc[
        pd.to_datetime(mutated["date"]).eq(label_window_date),
        ["open", "high", "low", "close", "adj_close"],
    ] *= 4.0

    baseline = build_price_features(data)
    changed = build_price_features(mutated)

    price_feature_columns = [
        "return_1",
        "return_5",
        "return_20",
        "high_low_range",
        "dollar_volume",
        "volatility_20",
        "volatility_60",
        "volume_z_20",
        "ma_ratio_20",
        "ma_ratio_60",
        "rsi_14",
        "realized_volatility",
        "liquidity_score",
    ]
    pd.testing.assert_frame_equal(
        _feature_prefix(baseline, decision_date, ["date", "ticker", *price_feature_columns]),
        _feature_prefix(changed, decision_date, ["date", "ticker", *price_feature_columns]),
        check_dtype=False,
    )

    baseline_label = baseline.loc[baseline["date"].eq(decision_date), "forward_return_20"].iloc[0]
    changed_label = changed.loc[changed["date"].eq(decision_date), "forward_return_20"].iloc[0]
    assert changed_label != baseline_label


def test_late_available_price_rows_are_dropped_before_return_features_are_derived() -> None:
    data = _deterministic_price_data(periods=35, tickers=("AAPL",))
    late_date = pd.Timestamp("2026-01-16")
    data.loc[data["date"].eq(late_date), "availability_timestamp"] = pd.Timestamp(
        "2026-03-01 00:00:00",
        tz="UTC",
    )

    features = build_price_features(data)

    assert late_date not in set(features["date"])
    assert len(features) == len(data) - 1


def _deterministic_price_data(periods: int, tickers: tuple[str, ...]) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-02", periods=periods)
    frames: list[pd.DataFrame] = []
    for ticker_index, ticker in enumerate(tickers):
        base = 100.0 + ticker_index * 50.0
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "open": [base + idx for idx in range(periods)],
                    "high": [base + 1.5 + idx for idx in range(periods)],
                    "low": [base - 1.0 + idx for idx in range(periods)],
                    "close": [base + 0.5 + idx for idx in range(periods)],
                    "adj_close": [base + 0.5 + idx for idx in range(periods)],
                    "volume": [1_000_000 + ticker_index * 100_000 + idx * 1_000 for idx in range(periods)],
                }
            )
        )
    return add_price_timestamps(pd.concat(frames, ignore_index=True))


def _feature_prefix(frame: pd.DataFrame, cutoff: pd.Timestamp, columns: list[str]) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.normalize()
    return (
        normalized.loc[normalized["date"] <= cutoff, columns]
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
