from __future__ import annotations

import pandas as pd

from quant_research.data.market import SyntheticMarketDataProvider
from quant_research.features.price import build_price_features
from quant_research.validation.walk_forward import (
    WalkForwardConfig,
    walk_forward_predict,
    walk_forward_splits,
)


def test_walk_forward_splits_never_train_on_future_dates() -> None:
    data = SyntheticMarketDataProvider(periods=120).get_history(["AAPL", "MSFT"])
    features = build_price_features(data).dropna(subset=["forward_return_1"])
    folds = walk_forward_splits(features, WalkForwardConfig(train_periods=40, test_periods=10, gap_periods=1))

    assert folds
    for fold in folds:
        assert fold.train_end < fold.test_start
        assert max(fold.train_dates) < min(fold.test_dates)


def test_walk_forward_marks_final_fold_as_oos() -> None:
    data = SyntheticMarketDataProvider(periods=140).get_history(["AAPL", "MSFT"])
    features = build_price_features(data).dropna(subset=["forward_return_1"])

    predictions, summary = walk_forward_predict(
        features,
        WalkForwardConfig(train_periods=50, test_periods=15, gap_periods=1, min_train_observations=50),
    )

    assert not predictions.empty
    assert not summary.empty
    assert "prediction_count" in summary.columns
    assert "information_coefficient" in summary.columns
    assert summary["is_oos"].sum() == 1
    assert predictions["is_oos"].any()


def test_walk_forward_splits_respects_custom_dates_and_embargo() -> None:
    dates = pd.date_range(start="2026-01-01", periods=15, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "feature": 1.0,
            "ticker": ["AAPL"] * len(dates),
            "forward_return_1": [0.01] * len(dates),
        }
    )
    candidate_dates = pd.DatetimeIndex(dates[[0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13]])
    config = WalkForwardConfig(train_periods=3, test_periods=2, gap_periods=1, embargo_periods=1)
    folds = walk_forward_splits(frame, config, candidate_dates=candidate_dates)

    assert len(folds) >= 2
    for fold in folds:
        assert set(fold.train_dates).issubset(set(candidate_dates))
        assert set(fold.test_dates).issubset(set(candidate_dates))
        assert max(fold.train_dates) < min(fold.test_dates)
    date_position = {date: position for position, date in enumerate(candidate_dates)}
    for index, fold in enumerate(folds[:-1]):
        assert date_position[folds[index + 1].train_start] == date_position[fold.test_end] + config.embargo_periods + 1


def test_walk_forward_predict_skips_targetless_dates() -> None:
    dates = pd.bdate_range(start="2026-01-01", periods=9)
    target = [0.01, 0.02, 0.005, 0.01, 0.0, -0.01, None, 0.02, 0.01]
    frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAPL"] * len(dates),
            "feature_a": list(range(len(dates))),
            "forward_return_1": target,
        }
    )
    predictions, summary = walk_forward_predict(
        frame,
        WalkForwardConfig(train_periods=3, test_periods=2, gap_periods=1, min_train_observations=3),
    )

    assert not predictions.empty
    assert not summary.empty
    assert pd.Timestamp(dates[6]) not in set(predictions["date"].dt.normalize())
    assert summary["labeled_test_observations"].sum() > 0
