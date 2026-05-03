from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Chronos2Adapter:
    model_id: str = "amazon/chronos-2"
    device_map: str = "auto"
    quantile_levels: tuple[float, float, float] = (0.1, 0.5, 0.9)
    local_files_only: bool = False
    _pipeline: Any = field(default=None, init=False, repr=False)

    def available(self) -> bool:
        try:
            import chronos  # noqa: F401

            return True
        except ImportError:
            return False

    def add_forecasts(
        self,
        features: pd.DataFrame,
        mode: str = "proxy",
        target_column: str = "return_1",
        min_context: int = 64,
        max_inference_windows: int | None = None,
    ) -> pd.DataFrame:
        if mode == "local":
            return self.add_local_forecasts(
                features,
                target_column=target_column,
                min_context=min_context,
                max_inference_windows=max_inference_windows,
            )
        return self.add_proxy_forecasts(features)

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

    def add_local_forecasts(
        self,
        features: pd.DataFrame,
        target_column: str = "return_1",
        min_context: int = 64,
        max_inference_windows: int | None = None,
    ) -> pd.DataFrame:
        frame = self.add_proxy_forecasts(features)
        try:
            pipeline = self._load_pipeline()
        except Exception:
            return frame

        predictions = _rolling_forecast_dates(frame, min_context, max_inference_windows)
        for forecast_date in predictions:
            context = _context_frame(frame, forecast_date, target_column)
            if context.empty:
                continue
            try:
                raw = pipeline.predict_df(
                    context,
                    prediction_length=1,
                    quantile_levels=list(self.quantile_levels),
                    id_column="id",
                    timestamp_column="timestamp",
                    target="target",
                )
            except Exception:
                continue
            normalized = _normalise_chronos_prediction(raw)
            if normalized.empty:
                continue
            _write_forecast_columns(
                frame,
                forecast_date,
                normalized,
                prefix="chronos",
                expected_column="expected_return",
                low_column="downside_quantile",
                high_column="upside_quantile",
            )
        frame["chronos_quantile_width"] = frame["chronos_upside_quantile"] - frame["chronos_downside_quantile"]
        return frame

    def _load_pipeline(self) -> Any:
        if self._pipeline is None:
            from chronos import Chronos2Pipeline

            kwargs: dict[str, object] = {"device_map": self.device_map}
            if self.local_files_only:
                kwargs["local_files_only"] = True
            self._pipeline = Chronos2Pipeline.from_pretrained(self.model_id, **kwargs)
        return self._pipeline


@dataclass
class GraniteTTMAdapter:
    model_id: str = "ibm-granite/granite-timeseries-ttm-r2"
    revision: str | None = None
    fit_strategy: str | None = "zero-shot"
    _forecaster: Any = field(default=None, init=False, repr=False)

    def available(self) -> bool:
        try:
            import sktime.forecasting.ttm  # noqa: F401

            return True
        except ImportError:
            return False

    def add_forecasts(
        self,
        features: pd.DataFrame,
        mode: str = "proxy",
        target_column: str = "return_1",
        min_context: int = 64,
        max_inference_windows: int | None = None,
    ) -> pd.DataFrame:
        if mode == "local":
            return self.add_local_forecasts(
                features,
                target_column=target_column,
                min_context=min_context,
                max_inference_windows=max_inference_windows,
            )
        return self.add_proxy_forecasts(features)

    def add_proxy_forecasts(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add lightweight TTM-style local forecast proxies for fast experiments."""
        frame = features.copy()
        short_signal = frame.groupby("ticker")["return_5"].transform(lambda series: series.shift(1).fillna(0) / 5)
        range_penalty = frame["high_low_range"].fillna(0)
        frame["granite_ttm_expected_return"] = short_signal - 0.1 * range_penalty
        frame["granite_ttm_confidence"] = (1 / (1 + np.exp(-np.abs(short_signal) * 100))).clip(0, 1)
        return frame

    def add_local_forecasts(
        self,
        features: pd.DataFrame,
        target_column: str = "return_1",
        min_context: int = 64,
        max_inference_windows: int | None = None,
    ) -> pd.DataFrame:
        frame = self.add_proxy_forecasts(features)
        predictions = _rolling_forecast_dates(frame, min_context, max_inference_windows)
        for forecast_date in predictions:
            day_context = frame[frame["date"] < forecast_date].copy()
            if day_context.empty:
                continue
            for ticker, ticker_context in day_context.groupby("ticker"):
                series = ticker_context.sort_values("date").set_index("date")[target_column].dropna()
                if len(series) < min_context:
                    continue
                try:
                    forecaster = self._load_forecaster()
                    forecaster.fit(series.astype(float))
                    prediction = forecaster.predict(fh=[1])
                except Exception:
                    continue
                expected = _first_numeric_value(prediction)
                if expected is None:
                    continue
                mask = (frame["date"] == forecast_date) & (frame["ticker"] == ticker)
                frame.loc[mask, "granite_ttm_expected_return"] = expected
                frame.loc[mask, "granite_ttm_confidence"] = min(1.0, abs(expected) * 100)
        return frame

    def _load_forecaster(self) -> Any:
        if self._forecaster is None:
            from sktime.forecasting.ttm import TinyTimeMixerForecaster

            kwargs: dict[str, object] = {"model_path": self.model_id}
            if self.revision:
                kwargs["revision"] = self.revision
            if self.fit_strategy:
                kwargs["fit_strategy"] = self.fit_strategy
            try:
                self._forecaster = TinyTimeMixerForecaster(**kwargs)
            except TypeError:
                kwargs.pop("fit_strategy", None)
                self._forecaster = TinyTimeMixerForecaster(**kwargs)
        return self._forecaster


def _rolling_forecast_dates(
    frame: pd.DataFrame,
    min_context: int,
    max_inference_windows: int | None,
) -> list[pd.Timestamp]:
    dates = sorted(pd.to_datetime(frame["date"]).dt.normalize().unique())
    eligible = [
        pd.Timestamp(forecast_date)
        for forecast_date in dates
        if frame[frame["date"] < forecast_date].groupby("ticker").size().max() >= min_context
    ]
    if max_inference_windows is not None and max_inference_windows > 0:
        return eligible[-max_inference_windows:]
    return eligible


def _context_frame(frame: pd.DataFrame, forecast_date: pd.Timestamp, target_column: str) -> pd.DataFrame:
    context = frame[frame["date"] < forecast_date][["ticker", "date", target_column]].copy()
    context = context.dropna(subset=[target_column])
    if context.empty:
        return pd.DataFrame(columns=["id", "timestamp", "target"])
    return context.rename(columns={"ticker": "id", "date": "timestamp", target_column: "target"})


def _normalise_chronos_prediction(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.reset_index() if isinstance(raw.index, pd.MultiIndex) else raw.copy()
    id_column = next((column for column in ["id", "item_id", "ticker"] if column in frame.columns), None)
    if id_column is None:
        return pd.DataFrame(columns=["ticker", "expected_return", "downside_quantile", "upside_quantile"])
    low = _find_column(frame, ("0.1", 0.1, "q0.1", "p10"))
    median = _find_column(frame, ("0.5", 0.5, "mean", "q0.5", "p50"))
    high = _find_column(frame, ("0.9", 0.9, "q0.9", "p90"))
    if median is None:
        return pd.DataFrame(columns=["ticker", "expected_return", "downside_quantile", "upside_quantile"])
    output = pd.DataFrame(
        {
            "ticker": frame[id_column].astype(str),
            "expected_return": pd.to_numeric(frame[median], errors="coerce"),
            "downside_quantile": pd.to_numeric(frame[low], errors="coerce") if low else pd.to_numeric(frame[median], errors="coerce"),
            "upside_quantile": pd.to_numeric(frame[high], errors="coerce") if high else pd.to_numeric(frame[median], errors="coerce"),
        }
    )
    return output.dropna(subset=["expected_return"]).drop_duplicates("ticker", keep="last")


def _find_column(frame: pd.DataFrame, candidates: tuple[object, ...]) -> object | None:
    columns_by_text = {str(column): column for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        if str(candidate) in columns_by_text:
            return columns_by_text[str(candidate)]
    return None


def _write_forecast_columns(
    frame: pd.DataFrame,
    forecast_date: pd.Timestamp,
    forecasts: pd.DataFrame,
    prefix: str,
    expected_column: str,
    low_column: str,
    high_column: str,
) -> None:
    forecast_by_ticker = forecasts.set_index("ticker")
    for ticker, row in forecast_by_ticker.iterrows():
        mask = (frame["date"] == forecast_date) & (frame["ticker"] == ticker)
        frame.loc[mask, f"{prefix}_{expected_column}"] = row["expected_return"]
        frame.loc[mask, f"{prefix}_{low_column}"] = row["downside_quantile"]
        frame.loc[mask, f"{prefix}_{high_column}"] = row["upside_quantile"]


def _first_numeric_value(prediction: object) -> float | None:
    if isinstance(prediction, (pd.DataFrame, pd.Series)):
        values = pd.to_numeric(pd.Series(np.ravel(prediction.to_numpy())), errors="coerce").dropna()
        return float(values.iloc[0]) if not values.empty else None
    try:
        return float(prediction)
    except (TypeError, ValueError):
        return None
