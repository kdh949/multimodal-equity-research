from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_research.models.tabular import TabularReturnModel, infer_feature_columns


@dataclass(frozen=True)
class WalkForwardConfig:
    train_periods: int = 90
    test_periods: int = 20
    gap_periods: int = 1
    min_train_observations: int = 80
    model_name: str = "lightgbm"
    winsorize_features: bool = True
    winsorization_lower_quantile: float = 0.01
    winsorization_upper_quantile: float = 0.99
    recent_sample_weighting: bool = True
    native_tabular_isolation: bool = True
    native_model_timeout_seconds: int = 180
    tabular_num_threads: int = 1
    embargo_periods: int = 0
    target_horizon: int = 1
    requested_gap_periods: int | None = None
    requested_embargo_periods: int | None = None


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    train_dates_before_purge: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    target_horizon: int = 1
    requested_gap_periods: int = 1
    requested_embargo_periods: int = 0
    effective_gap_periods: int = 1
    effective_embargo_periods: int = 0
    purged_train_date_count: int = 0
    label_overlap_violations: int = 0


def walk_forward_splits(
    frame: pd.DataFrame,
    config: WalkForwardConfig,
    candidate_dates: pd.DatetimeIndex | None = None,
    target_horizon: int | None = None,
) -> list[WalkForwardFold]:
    if candidate_dates is None:
        if "date" not in frame.columns:
            return []
        dates = pd.DatetimeIndex(sorted(pd.to_datetime(frame["date"]).dt.normalize().dropna().unique()))
    else:
        dates = pd.DatetimeIndex(sorted(pd.to_datetime(candidate_dates).dropna().unique()))
    if len(dates) == 0:
        return []

    horizon = max(1, int(target_horizon or config.target_horizon or 1))
    requested_gap = int(config.requested_gap_periods if config.requested_gap_periods is not None else config.gap_periods)
    requested_embargo = int(
        config.requested_embargo_periods
        if config.requested_embargo_periods is not None
        else config.embargo_periods
    )
    effective_gap = max(0, requested_gap, horizon)
    effective_embargo = max(0, requested_embargo, horizon)
    date_position = {date: position for position, date in enumerate(dates)}
    folds: list[WalkForwardFold] = []
    start = 0
    fold_id = 0
    while True:
        train_start_idx = start
        train_end_idx = train_start_idx + config.train_periods
        test_start_idx = train_end_idx + effective_gap
        if test_start_idx >= len(dates):
            break
        test_end_idx = min(test_start_idx + config.test_periods, len(dates))
        train_dates_before_purge = dates[train_start_idx:train_end_idx]
        test_dates = dates[test_start_idx:test_end_idx]
        train_dates = _purge_overlapping_train_dates(
            train_dates_before_purge,
            test_dates,
            date_position,
            horizon,
        )
        overlap_violations = _count_label_interval_overlaps(
            train_dates,
            test_dates,
            date_position,
            horizon,
        )
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
                train_dates_before_purge=train_dates_before_purge,
                target_horizon=horizon,
                requested_gap_periods=requested_gap,
                requested_embargo_periods=requested_embargo,
                effective_gap_periods=effective_gap,
                effective_embargo_periods=effective_embargo,
                purged_train_date_count=len(train_dates_before_purge) - len(train_dates),
                label_overlap_violations=overlap_violations,
            )
        )
        fold_id += 1
        start = test_end_idx + effective_embargo
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

    labeled_dates = pd.DatetimeIndex(
        pd.to_datetime(frame.loc[frame[target].notna(), "date"]).dropna().dt.normalize().unique()
    )
    target_horizon = _horizon_from_target(target) or max(1, int(config.target_horizon or 1))
    folds = walk_forward_splits(
        frame,
        config,
        candidate_dates=labeled_dates,
        target_horizon=target_horizon,
    )
    if not folds:
        return pd.DataFrame(), pd.DataFrame()
    final_fold_id = folds[-1].fold

    for fold in folds:
        train = frame[frame["date"].isin(fold.train_dates)]
        test = frame[frame["date"].isin(fold.test_dates)]
        train = train.dropna(subset=[target])
        if len(train) < config.min_train_observations or test.empty:
            continue

        model = TabularReturnModel(
            model_name=config.model_name,
            feature_columns=feature_columns,
            winsorize_features=config.winsorize_features,
            winsorization_lower_quantile=config.winsorization_lower_quantile,
            winsorization_upper_quantile=config.winsorization_upper_quantile,
            use_recent_weighting=config.recent_sample_weighting,
            native_tabular_isolation=config.native_tabular_isolation,
            native_model_timeout_seconds=config.native_model_timeout_seconds,
            tabular_num_threads=config.tabular_num_threads,
        )
        model.fit(train, target=target)
        pred = model.predict(test)
        pred = pred.merge(test[["date", "ticker", target]], on=["date", "ticker"], how="left")
        pred["fold"] = fold.fold
        pred["is_oos"] = fold.fold == final_fold_id
        predictions.append(pred)

        labeled = pred.dropna(subset=[target])
        if labeled.empty:
            fold_mae = None
            direction = None
            ic = None
            sign_ic = None
        else:
            errors = (labeled["expected_return"] - labeled[target]).abs()
            fold_mae = float(errors.mean())
            direction = float((labeled["expected_return"] * labeled[target] > 0).mean())
            ic = _safe_correlation(labeled["expected_return"], labeled[target])
            sign_ic = _safe_correlation(np.sign(labeled["expected_return"]), np.sign(labeled[target]))

        summaries.append(
            {
                "fold": fold.fold,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "train_observations": len(train),
                "train_date_count_before_purge": len(fold.train_dates_before_purge),
                "train_date_count_after_purge": len(fold.train_dates),
                "purged_train_date_count": fold.purged_train_date_count,
                "test_observations": len(test),
                "labeled_test_observations": len(labeled),
                "prediction_count": len(pred),
                "target_column": target,
                "target_horizon": fold.target_horizon,
                "requested_gap_periods": fold.requested_gap_periods,
                "requested_embargo_periods": fold.requested_embargo_periods,
                "effective_gap_periods": fold.effective_gap_periods,
                "effective_embargo_periods": fold.effective_embargo_periods,
                "label_overlap_violations": fold.label_overlap_violations,
                "fold_type": "oos" if fold.fold == final_fold_id else "validation",
                "model_name": model.actual_model_name,
                "is_oos": fold.fold == final_fold_id,
                "tabular_fallback_reason": model.training_metadata.get("tabular_fallback_reason"),
                "mae": fold_mae,
                "directional_accuracy": direction,
                "information_coefficient": ic,
                "sign_information_coefficient": sign_ic,
                "model_calibration_scale": model.training_metadata.get("calibration_scale"),
                "model_calibration_bias": model.training_metadata.get("calibration_bias"),
                "winsorized_feature_count": model.training_metadata.get("winsorized_feature_count", 0),
            }
        )
    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    summary_frame = pd.DataFrame(summaries)
    return prediction_frame, summary_frame


def _purge_overlapping_train_dates(
    train_dates: pd.DatetimeIndex,
    test_dates: pd.DatetimeIndex,
    date_position: dict[pd.Timestamp, int],
    horizon: int,
) -> pd.DatetimeIndex:
    if len(train_dates) == 0 or len(test_dates) == 0:
        return train_dates
    kept = [
        train_date
        for train_date in train_dates
        if not _label_interval_overlaps_any_test(train_date, test_dates, date_position, horizon)
    ]
    return pd.DatetimeIndex(kept)


def _count_label_interval_overlaps(
    train_dates: pd.DatetimeIndex,
    test_dates: pd.DatetimeIndex,
    date_position: dict[pd.Timestamp, int],
    horizon: int,
) -> int:
    return sum(
        1
        for train_date in train_dates
        if _label_interval_overlaps_any_test(train_date, test_dates, date_position, horizon)
    )


def _label_interval_overlaps_any_test(
    train_date: pd.Timestamp,
    test_dates: pd.DatetimeIndex,
    date_position: dict[pd.Timestamp, int],
    horizon: int,
) -> bool:
    train_start = date_position[pd.Timestamp(train_date)]
    train_end = train_start + horizon
    for test_date in test_dates:
        test_start = date_position[pd.Timestamp(test_date)]
        test_end = test_start + horizon
        if train_start <= test_end and test_start <= train_end:
            return True
    return False


def _horizon_from_target(target: str) -> int | None:
    prefix = "forward_return_"
    if not target.startswith(prefix):
        return None
    try:
        return int(target.removeprefix(prefix))
    except ValueError:
        return None


def _safe_correlation(left: pd.Series, right: pd.Series) -> float | None:
    left_series = pd.to_numeric(left, errors="coerce")
    right_series = pd.to_numeric(right, errors="coerce")
    common = left_series.notna() & right_series.notna()
    if common.sum() < 2:
        return None
    left_values = left_series.loc[common]
    right_values = right_series.loc[common]
    if left_values.nunique(dropna=True) <= 1 or right_values.nunique(dropna=True) <= 1:
        return None
    return float(np.corrcoef(left_values, right_values)[0, 1])
