from __future__ import annotations

import pandas as pd

from quant_research.data.market import SyntheticMarketDataProvider
from quant_research.features.price import build_price_features


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
