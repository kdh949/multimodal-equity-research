from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_research.models.tabular import TabularReturnModel, infer_feature_columns


@dataclass(frozen=True)
class WalkForwardConfig:
    train_periods: int = 90
    test_periods: int = 20
    gap_periods: int = 1
    min_train_observations: int = 80
    model_name: str = "lightgbm"


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex


def walk_forward_splits(frame: pd.DataFrame, config: WalkForwardConfig) -> list[WalkForwardFold]:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(frame["date"]).dt.normalize().unique()))
    folds: list[WalkForwardFold] = []
    start = 0
    fold_id = 0
    while True:
        train_start_idx = start
        train_end_idx = train_start_idx + config.train_periods
        test_start_idx = train_end_idx + config.gap_periods
        test_end_idx = test_start_idx + config.test_periods
        if test_start_idx >= len(dates):
            break
        test_end_idx = min(test_end_idx, len(dates))
        train_dates = dates[train_start_idx:train_end_idx]
        test_dates = dates[test_start_idx:test_end_idx]
        if len(train_dates) == 0 or len(test_dates) == 0:
            break
        folds.append(
            WalkForwardFold(
                fold=fold_id,
                train_start=train_dates.min(),
                train_end=train_dates.max(),
                test_start=test_dates.min(),
                test_end=test_dates.max(),
                train_dates=train_dates,
                test_dates=test_dates,
            )
        )
        fold_id += 1
        start += config.test_periods
        if test_end_idx >= len(dates):
            break
    return folds


def walk_forward_predict(
    frame: pd.DataFrame,
    config: WalkForwardConfig,
    target: str = "forward_return_1",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = infer_feature_columns(frame, target)
    predictions: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for fold in walk_forward_splits(frame, config):
        train = frame[frame["date"].isin(fold.train_dates)].dropna(subset=[target])
        test = frame[frame["date"].isin(fold.test_dates)].copy()
        if len(train) < config.min_train_observations or test.empty:
            continue
        model = TabularReturnModel(model_name=config.model_name, feature_columns=feature_columns)
        model.fit(train, target=target)
        pred = model.predict(test)
        pred = pred.merge(test[["date", "ticker", target]], on=["date", "ticker"], how="left")
        pred["fold"] = fold.fold
        predictions.append(pred)
        fold_mae = (pred["expected_return"] - pred[target]).abs().mean()
        direction = (pred["expected_return"] * pred[target] > 0).mean()
        summaries.append(
            {
                "fold": fold.fold,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "train_observations": len(train),
                "test_observations": len(test),
                "model_name": model.actual_model_name,
                "mae": fold_mae,
                "directional_accuracy": direction,
            }
        )
    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    summary_frame = pd.DataFrame(summaries)
    return prediction_frame, summary_frame
