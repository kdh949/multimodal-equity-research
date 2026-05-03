from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
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
    fitted_model: object | None = None
    actual_model_name: str = ""

    def fit(self, train: pd.DataFrame, target: str = "forward_return_1") -> TabularReturnModel:
        train = train.dropna(subset=[target]).copy()
        self.feature_columns = self.feature_columns or infer_feature_columns(train, target)
        if len(train) < 30 or not self.feature_columns:
            estimator = DummyRegressor(strategy="mean")
            self.actual_model_name = "dummy"
        else:
            estimator = _make_estimator(self.model_name, self.random_state)
            self.actual_model_name = estimator.__class__.__name__
        self.fitted_model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", estimator),
            ]
        )
        self.fitted_model.fit(train[self.feature_columns], train[target])
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.fitted_model is None:
            raise RuntimeError("Model must be fitted before predict")
        output = frame[["date", "ticker"]].copy()
        expected = self.fitted_model.predict(frame[self.feature_columns])
        volatility = _safe_volatility(frame)
        output["expected_return"] = expected
        output["predicted_volatility"] = volatility
        output["downside_quantile"] = output["expected_return"] - 1.65 * output["predicted_volatility"]
        output["upside_quantile"] = output["expected_return"] + 1.65 * output["predicted_volatility"]
        output["quantile_width"] = output["upside_quantile"] - output["downside_quantile"]
        output["model_confidence"] = (
            np.abs(output["expected_return"]) / output["predicted_volatility"].replace(0, np.nan)
        ).clip(0, 3).fillna(0) / 3
        output["model_name"] = self.actual_model_name
        return output


def infer_feature_columns(frame: pd.DataFrame, target: str = "forward_return_1") -> list[str]:
    excluded = set(IDENTIFIER_COLUMNS)
    excluded.add(target)
    numeric_columns = frame.select_dtypes(include=[np.number]).columns
    return [column for column in numeric_columns if column not in excluded]


def _make_estimator(model_name: str, random_state: int) -> object:
    normalized = model_name.lower()
    if normalized == "lightgbm":
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
            )
        except ImportError:
            pass
    if normalized == "xgboost":
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
        except ImportError:
            pass
    if normalized == "catboost":
        try:
            from catboost import CatBoostRegressor

            return CatBoostRegressor(
                iterations=160,
                learning_rate=0.04,
                depth=4,
                random_seed=random_state,
                verbose=False,
            )
        except ImportError:
            pass
    return HistGradientBoostingRegressor(max_iter=160, learning_rate=0.04, random_state=random_state)


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
