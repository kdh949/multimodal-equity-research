from __future__ import annotations

import numpy as np
import pandas as pd


def build_price_features(price_data: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker", "open", "high", "low", "close", "adj_close", "volume"}
    missing = required.difference(price_data.columns)
    if missing:
        raise ValueError(f"Missing price columns: {sorted(missing)}")

    frame = price_data.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    groups = frame.groupby("ticker", group_keys=False)

    frame["return_1"] = groups["adj_close"].pct_change()
    frame["return_5"] = groups["adj_close"].pct_change(5)
    frame["return_20"] = groups["adj_close"].pct_change(20)
    for horizon in (1, 5, 20):
        frame[f"forward_return_{horizon}"] = (
            groups["adj_close"].pct_change(horizon).groupby(frame["ticker"]).shift(-horizon)
        )
    frame["high_low_range"] = (frame["high"] - frame["low"]) / frame["close"].replace(0, np.nan)
    frame["dollar_volume"] = frame["adj_close"] * frame["volume"]

    frame["volatility_20"] = groups["return_1"].transform(lambda series: series.rolling(20, min_periods=5).std())
    frame["volatility_60"] = groups["return_1"].transform(lambda series: series.rolling(60, min_periods=20).std())
    frame["volume_mean_20"] = groups["volume"].transform(lambda series: series.rolling(20, min_periods=5).mean())
    frame["volume_std_20"] = groups["volume"].transform(lambda series: series.rolling(20, min_periods=5).std())
    frame["volume_z_20"] = (frame["volume"] - frame["volume_mean_20"]) / frame["volume_std_20"].replace(0, np.nan)

    ma_20 = groups["adj_close"].transform(lambda series: series.rolling(20, min_periods=5).mean())
    ma_60 = groups["adj_close"].transform(lambda series: series.rolling(60, min_periods=20).mean())
    frame["ma_ratio_20"] = frame["adj_close"] / ma_20 - 1
    frame["ma_ratio_60"] = frame["adj_close"] / ma_60 - 1
    frame["rsi_14"] = groups["adj_close"].transform(_rsi)
    frame["realized_volatility"] = frame["volatility_20"]
    frame["liquidity_score"] = np.log1p(frame["dollar_volume"]).replace([np.inf, -np.inf], np.nan)

    return frame.replace([np.inf, -np.inf], np.nan)


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window, min_periods=window // 2).mean()
    avg_loss = loss.rolling(window, min_periods=window // 2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))
