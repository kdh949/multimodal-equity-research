from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

IDENTIFIER_COLUMNS = {
    "date",
    "ticker",
    "forward_return_1",
    "news_top_event",
}


@dataclass
class TabularReturnModel:
    model_name: str = "lightgbm"
    random_state: int = 42
    feature_columns: list[str] = field(default_factory=list)
    fitted_model: BaseEstimator | None = None
    actual_model_name: str = ""
    winsorize_features: bool = True
    winsorization_lower_quantile: float = 0.01
    winsorization_upper_quantile: float = 0.99
    use_recent_weighting: bool = True
    recent_weight_power: float = 1.0
    training_metadata: dict[str, object] = field(default_factory=dict, init=False)
    _winsorizer: BaseEstimator | None = field(default=None, init=False, repr=False)

    def fit(self, train: pd.DataFrame, target: str = "forward_return_1") -> TabularReturnModel:
        train = train.dropna(subset=[target]).copy()
        self.feature_columns = self.feature_columns or infer_feature_columns(train, target)
        self.training_metadata = {
            "target": target,
            "rows_after_target_filter": len(train),
            "requested_model_name": self.model_name,
        }

        if len(train) < 30 or not self.feature_columns:
            estimator: Any = DummyRegressor(strategy="mean")
            self.actual_model_name = "dummy"
            self.training_metadata["fit_reason"] = "insufficient_data"
        else:
            estimator = _make_estimator(self.model_name, self.random_state)
            self.actual_model_name = getattr(estimator, "__class__", type("Model", (), {})).__name__
            self.training_metadata["fit_reason"] = "model_fitted"

        feature_matrix = _to_numeric_frame(train[self.feature_columns])
        winsorizer = _Winsorizer(
            feature_columns=self.feature_columns,
            lower_quantile=self.winsorization_lower_quantile,
            upper_quantile=self.winsorization_upper_quantile,
            enabled=self.winsorize_features,
        )
        steps = [("winsorize", winsorizer), ("imputer", SimpleImputer(strategy="median")), ("model", estimator)]
        if not self.winsorize_features:
            steps = [("imputer", SimpleImputer(strategy="median")), ("model", estimator)]
        self._winsorizer = winsorizer
        self.fitted_model = Pipeline(steps=steps)
        sample_weight = (
            _recent_weights(
                train["date"],
                enabled=self.use_recent_weighting,
                power=self.recent_weight_power,
            )
            if "date" in train.columns
            else None
        )
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["model__sample_weight"] = sample_weight
        try:
            self.fitted_model.fit(feature_matrix, train[target], **fit_kwargs)
        except TypeError:
            # Some estimator stacks may not accept sample_weight under older versions.
            self.fitted_model.fit(feature_matrix, train[target])

        in_sample_pred = self.fitted_model.predict(feature_matrix)
        scale, bias = _calibration(from_predictions=in_sample_pred, target=train[target])
        self.training_metadata.update(
            {
                "actual_model_name": self.actual_model_name,
                "train_rows": len(train),
                "feature_columns": list(self.feature_columns),
                "feature_count": len(self.feature_columns),
                "winsorized_feature_count": len(getattr(self._winsorizer, "bounds", {})),
                "winsorization_enabled": bool(self.winsorize_features),
                "winsorization_lower_quantile": self.winsorization_lower_quantile,
                "winsorization_upper_quantile": self.winsorization_upper_quantile,
                "recent_weighting_enabled": bool(self.use_recent_weighting),
                "calibration_scale": float(scale),
                "calibration_bias": float(bias),
                "in_sample_mae": float(np.abs(train[target] - in_sample_pred).mean()),
            }
        )
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.fitted_model is None:
            raise RuntimeError("Model must be fitted before predict")
        output = frame[["date", "ticker"]].copy()
        feature_matrix = _to_numeric_frame(frame[self.feature_columns])
        raw = self.fitted_model.predict(feature_matrix)
        calibrated = _apply_calibration(
            raw,
            self.training_metadata.get("calibration_scale", 1.0),
            self.training_metadata.get("calibration_bias", 0.0),
        )
        volatility = _safe_volatility(frame)
        output["raw_expected_return"] = raw
        output["expected_return"] = calibrated
        output["predicted_volatility"] = volatility
        output["downside_quantile"] = output["expected_return"] - 1.65 * output["predicted_volatility"]
        output["upside_quantile"] = output["expected_return"] + 1.65 * output["predicted_volatility"]
        output["quantile_width"] = output["upside_quantile"] - output["downside_quantile"]
        output["model_confidence"] = (
            np.abs(output["expected_return"]) / output["predicted_volatility"].replace(0, np.nan)
        ).clip(0, 3).fillna(0) / 3
        output["model_name"] = self.actual_model_name
        output["model_calibration_scale"] = self.training_metadata.get("calibration_scale", 1.0)
        output["model_calibration_bias"] = self.training_metadata.get("calibration_bias", 0.0)
        return output


class _Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        feature_columns: list[str],
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        enabled: bool = True,
    ) -> None:
        self.feature_columns = feature_columns
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.enabled = enabled
        self.bounds: dict[str, tuple[float, float]] = {}

    def fit(self, frame: pd.DataFrame, y: pd.Series | None = None) -> _Winsorizer:
        if not self.enabled:
            self.bounds = {}
            return self
        numeric_frame = _to_numeric_frame(frame)
        if numeric_frame.empty:
            self.bounds = {}
            return self
        lowered = numeric_frame.quantile(self.lower_quantile)
        raised = numeric_frame.quantile(self.upper_quantile)
        bounds: dict[str, tuple[float, float]] = {}
        for column in numeric_frame.columns:
            low = float(lowered.get(column, np.nan))
            high = float(raised.get(column, np.nan))
            if pd.notna(low) and pd.notna(high) and low <= high:
                bounds[column] = (low, high)
        self.bounds = bounds
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled or not self.bounds:
            return frame
        clipped = _to_numeric_frame(frame)
        for column, (lower, upper) in self.bounds.items():
            if column in clipped:
                clipped[column] = clipped[column].clip(lower, upper)
        return clipped


class _BoostingEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimators: list[BaseEstimator]):
        self.estimators = estimators
        self._fitted_estimators: list[BaseEstimator] = []

    def fit(self, X: Any, y: pd.Series, sample_weight: np.ndarray | list[float] | None = None) -> _BoostingEnsembleRegressor:
        self._fitted_estimators = []
        for estimator in self.estimators:
            model = estimator
            if sample_weight is not None:
                try:
                    model.fit(X, y, sample_weight=sample_weight)
                except TypeError:
                    model.fit(X, y)
            else:
                model.fit(X, y)
            self._fitted_estimators.append(model)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not self._fitted_estimators:
            raise RuntimeError("Model must be fitted before predict")
        predictions = np.column_stack(
            [estimator.predict(X) for estimator in self._fitted_estimators]
        )
        return predictions.mean(axis=1)


def infer_feature_columns(frame: pd.DataFrame, target: str = "forward_return_1") -> list[str]:
    excluded = set(IDENTIFIER_COLUMNS)
    excluded.add(target)
    numeric_columns = frame.select_dtypes(include=[np.number]).columns
    return [column for column in numeric_columns if column not in excluded]


def _make_estimator(model_name: str, random_state: int) -> BaseEstimator:
    normalized = model_name.lower()
    if normalized in {"auto", "ensemble", "boosting_ensemble"}:
        estimators = _build_boosting_estimators(random_state)
        if len(estimators) == 1:
            return estimators[0]
        if estimators:
            return _BoostingEnsembleRegressor(estimators)
    if normalized == "lightgbm":
        candidate = _lightgbm_estimator(random_state)
        if candidate is not None:
            return candidate
        return _hist_gradient_estimator(random_state)
    if normalized == "xgboost":
        candidate = _xgboost_estimator(random_state)
        if candidate is not None:
            return candidate
    if normalized == "catboost":
        candidate = _catboost_estimator(random_state)
        if candidate is not None:
            return candidate
    if normalized in {"auto", "lightgbm", "xgboost", "catboost", "boosting", "ensemble", "boosting_ensemble"}:
        estimators = _build_boosting_estimators(random_state)
        if estimators:
            if len(estimators) == 1:
                return estimators[0]
            return _BoostingEnsembleRegressor(estimators)
    return HistGradientBoostingRegressor(max_iter=160, learning_rate=0.04, random_state=random_state)


def _build_boosting_estimators(random_state: int) -> list[BaseEstimator]:
    candidates: list[BaseEstimator] = []
    lightgbm_estimator = _lightgbm_estimator(random_state)
    if lightgbm_estimator is not None:
        candidates.append(lightgbm_estimator)
    xgboost_estimator = _xgboost_estimator(random_state)
    if xgboost_estimator is not None:
        candidates.append(xgboost_estimator)
    catboost_estimator = _catboost_estimator(random_state)
    if catboost_estimator is not None:
        candidates.append(catboost_estimator)
    return candidates


def _lightgbm_estimator(random_state: int) -> BaseEstimator | None:
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=160,
            learning_rate=0.04,
            max_depth=-1,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            verbose=-1,
            num_threads=1,
        )
    except Exception:
        return None


def _hist_gradient_estimator(random_state: int) -> BaseEstimator:
    return HistGradientBoostingRegressor(
        max_iter=160,
        learning_rate=0.04,
        random_state=random_state,
    )


def _xgboost_estimator(random_state: int) -> BaseEstimator | None:
    if os.getenv("QT_ENABLE_XGBOOST", "").strip().lower() not in {"1", "true", "yes"}:
        return None
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=160,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            objective="reg:squarederror",
        )
    except Exception:
        return None


def _catboost_estimator(random_state: int) -> BaseEstimator | None:
    try:
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=160,
            learning_rate=0.04,
            depth=4,
            random_seed=random_state,
            verbose=False,
        )
    except Exception:
        return None


def _to_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.copy()
    for column in numeric.columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    return numeric


def _recent_weights(
    dates: pd.Series,
    enabled: bool = True,
    power: float = 1.0,
) -> np.ndarray | None:
    if not enabled:
        return None
    timestamp = pd.to_datetime(dates).astype("datetime64[ns]").astype("int64")
    if len(timestamp) < 2:
        return None
    rank = pd.Series(timestamp).rank(method="first")
    normalized = (rank - rank.min()) / (rank.max() - rank.min() + 1e-9)
    power = max(0.0, float(power))
    return np.power(0.4 + 0.6 * normalized, power)


def _apply_calibration(
    prediction: np.ndarray,
    scale: float,
    bias: float,
) -> np.ndarray:
    return np.asarray(prediction) * float(scale) + float(bias)


def _calibration(from_predictions: np.ndarray, target: pd.Series) -> tuple[float, float]:
    raw = pd.Series(np.asarray(from_predictions), index=target.index)
    target_series = pd.Series(target, index=target.index)
    denom = raw.std()
    if not np.isfinite(denom) or denom <= 0:
        return 1.0, 0.0
    target_std = target_series.std()
    if not np.isfinite(target_std) or target_std <= 0:
        return 1.0, 0.0
    scale = np.clip(target_std / denom, 0.35, 2.5)
    bias = float(target_series.mean() - scale * raw.mean())
    return float(scale), float(bias)


def _safe_volatility(frame: pd.DataFrame) -> pd.Series:
    if "volatility_20" in frame:
        volatility = frame["volatility_20"].fillna(frame["volatility_20"].median())
    elif "realized_volatility" in frame:
        volatility = frame["realized_volatility"].fillna(frame["realized_volatility"].median())
    else:
        volatility = pd.Series(0.02, index=frame.index)
    fallback = volatility.replace(0, np.nan).median()
    if pd.isna(fallback):
        fallback = 0.02
    return volatility.fillna(fallback).clip(lower=0.001, upper=0.25)
