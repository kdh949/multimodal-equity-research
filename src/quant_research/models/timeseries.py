from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Chronos2Adapter:
    model_id: str = "autogluon/chronos-2"

    def available(self) -> bool:
        try:
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    def add_proxy_forecasts(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add Chronos-like quantile forecast features without requiring the heavy model at test time."""
        frame = features.copy()
        momentum = frame.groupby("ticker")["return_20"].transform(lambda series: series.shift(1).fillna(0) / 20)
        volatility = frame["volatility_20"].fillna(frame["volatility_20"].median()).fillna(0.02)
        frame["chronos_expected_return"] = momentum
        frame["chronos_downside_quantile"] = momentum - 1.28 * volatility
        frame["chronos_upside_quantile"] = momentum + 1.28 * volatility
        frame["chronos_quantile_width"] = frame["chronos_upside_quantile"] - frame["chronos_downside_quantile"]
        return frame


@dataclass
class GraniteTTMAdapter:
    model_id: str = "ibm-granite/granite-timeseries-ttm-r2"

    def available(self) -> bool:
        try:
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    def add_proxy_forecasts(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add lightweight TTM-style local forecast proxies for fast experiments."""
        frame = features.copy()
        short_signal = frame.groupby("ticker")["return_5"].transform(lambda series: series.shift(1).fillna(0) / 5)
        range_penalty = frame["high_low_range"].fillna(0)
        frame["granite_ttm_expected_return"] = short_signal - 0.1 * range_penalty
        frame["granite_ttm_confidence"] = (1 / (1 + np.exp(-np.abs(short_signal) * 100))).clip(0, 1)
        return frame
