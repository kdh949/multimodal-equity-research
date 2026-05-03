from __future__ import annotations

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
    assert summary["is_oos"].sum() == 1
    assert predictions["is_oos"].any()
