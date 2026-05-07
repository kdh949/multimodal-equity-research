from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Protocol

import numpy as np
import pandas as pd

from quant_research.data.timestamps import (
    date_end_utc,
    timestamp_utc,
    validate_event_availability_order,
)
from quant_research.models.tabular import TabularReturnModel, infer_feature_columns
from quant_research.validation.horizons import DEFAULT_PURGE_EMBARGO_DAYS

WALK_FORWARD_INSUFFICIENT_DATA_STATUS = "insufficient_data"
WALK_FORWARD_SKIPPED_STATUS = "skipped"
DETERMINISTIC_FALLBACK_MODEL_NAME = "deterministic_neutral_fallback"
DETERMINISTIC_FALLBACK_POLICY = "neutral_hold"
CANONICAL_WALK_FORWARD_TRAIN_PERIODS = 252
CANONICAL_WALK_FORWARD_TEST_PERIODS = 60
CANONICAL_WALK_FORWARD_TARGET_COLUMN = "forward_return_20"
CANONICAL_WALK_FORWARD_TARGET_HORIZON_PERIODS = 20
CANONICAL_WALK_FORWARD_PURGE_PERIODS = 20
CANONICAL_WALK_FORWARD_EMBARGO_PERIODS = 20
CANONICAL_MIN_OOS_FOLDS = 2
VALID_WALK_FORWARD_EVALUATION_MODES = frozenset(
    {"non_overlapping", "horizon_consistent"}
)
WalkForwardWindowMode = Literal["rolling", "expanding"]
WalkForwardEvaluationMode = Literal["non_overlapping", "horizon_consistent"]


class WalkForwardSplitter(Protocol):
    """Interface for purge/embargo walk-forward splitters."""

    @property
    def config(self) -> PurgeEmbargoWalkForwardConfig:
        ...

    def boundaries(self, date_count: int) -> list[WalkForwardBoundary]:
        ...

    def split(
        self,
        frame: pd.DataFrame,
        candidate_dates: pd.DatetimeIndex | None = None,
    ) -> list[WalkForwardFold]:
        ...


@dataclass(frozen=True, slots=True)
class PurgeEmbargoWalkForwardConfig:
    train_periods: int = CANONICAL_WALK_FORWARD_TRAIN_PERIODS
    test_periods: int = CANONICAL_WALK_FORWARD_TEST_PERIODS
    purge_periods: int = CANONICAL_WALK_FORWARD_PURGE_PERIODS
    embargo_periods: int = CANONICAL_WALK_FORWARD_EMBARGO_PERIODS
    target_column: str = CANONICAL_WALK_FORWARD_TARGET_COLUMN
    window_mode: WalkForwardWindowMode = "rolling"
    evaluation_mode: WalkForwardEvaluationMode = "horizon_consistent"

    def __post_init__(self) -> None:
        train_periods = int(self.train_periods)
        test_periods = int(self.test_periods)
        purge_periods = int(self.purge_periods)
        embargo_periods = int(self.embargo_periods)
        if train_periods < 1 or test_periods < 1:
            raise ValueError("train_periods and test_periods must be positive")
        if purge_periods < 0 or embargo_periods < 0:
            raise ValueError("purge_periods and embargo_periods must be non-negative")
        target_column = str(self.target_column).strip()
        if not target_column:
            raise ValueError("target_column must not be blank")
        window_mode = str(self.window_mode).strip().lower()
        if window_mode not in {"rolling", "expanding"}:
            raise ValueError("window_mode must be either 'rolling' or 'expanding'")
        evaluation_mode = str(self.evaluation_mode).strip().lower()
        if evaluation_mode not in VALID_WALK_FORWARD_EVALUATION_MODES:
            raise ValueError(
                "evaluation_mode must be either 'non_overlapping' or 'horizon_consistent'"
            )

        object.__setattr__(self, "train_periods", train_periods)
        object.__setattr__(self, "test_periods", test_periods)
        object.__setattr__(self, "purge_periods", purge_periods)
        object.__setattr__(self, "embargo_periods", embargo_periods)
        object.__setattr__(self, "target_column", target_column)
        object.__setattr__(self, "window_mode", window_mode)
        object.__setattr__(self, "evaluation_mode", evaluation_mode)

    @property
    def gap_periods(self) -> int:
        return self.purge_periods

    @property
    def target_horizon_periods(self) -> int:
        return _target_horizon(self.target_column) or 1

    @property
    def is_system_valid(self) -> bool:
        return not self.system_validity_issues()

    def system_validity_issues(self) -> tuple[str, ...]:
        issues: list[str] = []
        horizon = self.target_horizon_periods
        if self.target_column != CANONICAL_WALK_FORWARD_TARGET_COLUMN:
            issues.append("target_column must be forward_return_20")
        if self.purge_periods < horizon:
            issues.append("purge_periods must be at least the target horizon")
        if self.embargo_periods < horizon:
            issues.append("embargo_periods must be at least the target horizon")
        if self.target_column == CANONICAL_WALK_FORWARD_TARGET_COLUMN and self.embargo_periods == 0:
            issues.append("forward_return_20 requires non-zero embargo_periods")
        return tuple(issues)

    def to_walk_forward_config(self, **overrides: object) -> WalkForwardConfig:
        payload: dict[str, object] = {
            "train_periods": self.train_periods,
            "test_periods": self.test_periods,
            "window_mode": self.window_mode,
            "gap_periods": self.purge_periods,
            "embargo_periods": self.embargo_periods,
            "prediction_horizon_periods": self.target_horizon_periods,
        }
        payload.update(overrides)
        return WalkForwardConfig(**payload)

    def to_dict(self) -> dict[str, object]:
        return {
            "train_periods": self.train_periods,
            "test_periods": self.test_periods,
            "purge_periods": self.purge_periods,
            "gap_periods": self.gap_periods,
            "embargo_periods": self.embargo_periods,
            "target_column": self.target_column,
            "target_horizon_periods": self.target_horizon_periods,
            "window_mode": self.window_mode,
            "evaluation_mode": self.evaluation_mode,
            "system_validity_issues": list(self.system_validity_issues()),
        }


@dataclass(frozen=True, slots=True)
class PurgeEmbargoWalkForwardSplitter:
    config: PurgeEmbargoWalkForwardConfig = PurgeEmbargoWalkForwardConfig()

    def boundaries(self, date_count: int) -> list[WalkForwardBoundary]:
        return walk_forward_boundaries(date_count, self.config.to_walk_forward_config())

    def split(
        self,
        frame: pd.DataFrame,
        candidate_dates: pd.DatetimeIndex | None = None,
    ) -> list[WalkForwardFold]:
        return walk_forward_splits(
            frame,
            self.config.to_walk_forward_config(),
            candidate_dates=candidate_dates,
        )


DEFAULT_PURGE_EMBARGO_WALK_FORWARD_CONFIG = PurgeEmbargoWalkForwardConfig()


@dataclass(frozen=True)
class WalkForwardConfig:
    train_periods: int = 90
    test_periods: int = 20
    window_mode: str = "rolling"
    gap_periods: int = DEFAULT_PURGE_EMBARGO_DAYS
    min_train_observations: int = 80
    model_name: str = "lightgbm"
    winsorize_features: bool = True
    winsorization_lower_quantile: float = 0.01
    winsorization_upper_quantile: float = 0.99
    recent_sample_weighting: bool = True
    native_tabular_isolation: bool = True
    native_model_timeout_seconds: int = 180
    tabular_num_threads: int = 1
    embargo_periods: int = DEFAULT_PURGE_EMBARGO_DAYS
    prediction_horizon_periods: int = 1
    target_horizon: int | None = None
    requested_gap_periods: int | None = None
    requested_embargo_periods: int | None = None

    def __post_init__(self) -> None:
        train_periods = int(self.train_periods)
        test_periods = int(self.test_periods)
        if train_periods < 1 or test_periods < 1:
            raise ValueError("train_periods and test_periods must be positive")
        window_mode = str(self.window_mode).strip().lower()
        if window_mode not in {"rolling", "expanding"}:
            raise ValueError("window_mode must be either 'rolling' or 'expanding'")
        prediction_horizon_periods = max(
            1,
            int(
                self.target_horizon
                if self.target_horizon is not None
                else self.prediction_horizon_periods
            ),
        )
        requested_gap_periods = (
            int(self.requested_gap_periods)
            if self.requested_gap_periods is not None
            else int(self.gap_periods)
        )
        requested_embargo_periods = (
            int(self.requested_embargo_periods)
            if self.requested_embargo_periods is not None
            else int(self.embargo_periods)
        )
        object.__setattr__(self, "train_periods", train_periods)
        object.__setattr__(self, "test_periods", test_periods)
        object.__setattr__(self, "window_mode", window_mode)
        object.__setattr__(self, "prediction_horizon_periods", prediction_horizon_periods)
        object.__setattr__(self, "target_horizon", prediction_horizon_periods)
        object.__setattr__(self, "requested_gap_periods", requested_gap_periods)
        object.__setattr__(self, "requested_embargo_periods", requested_embargo_periods)
        object.__setattr__(self, "gap_periods", max(int(self.gap_periods), prediction_horizon_periods))
        object.__setattr__(self, "embargo_periods", max(int(self.embargo_periods), prediction_horizon_periods))

    @property
    def effective_gap_periods(self) -> int:
        return self.gap_periods

    @property
    def effective_embargo_periods(self) -> int:
        return self.embargo_periods


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_dates: pd.DatetimeIndex
    purge_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    embargo_dates: pd.DatetimeIndex


@dataclass(frozen=True)
class WalkForwardHyperparameterTuningScope:
    fold: int
    tuning_train_dates: pd.DatetimeIndex
    tuning_validation_dates: pd.DatetimeIndex
    final_test_dates: pd.DatetimeIndex

    @property
    def tuning_train_start(self) -> pd.Timestamp:
        return self.tuning_train_dates.min() if len(self.tuning_train_dates) else pd.NaT

    @property
    def tuning_train_end(self) -> pd.Timestamp:
        return self.tuning_train_dates.max() if len(self.tuning_train_dates) else pd.NaT

    @property
    def tuning_validation_start(self) -> pd.Timestamp:
        return (
            self.tuning_validation_dates.min()
            if len(self.tuning_validation_dates)
            else pd.NaT
        )

    @property
    def tuning_validation_end(self) -> pd.Timestamp:
        return (
            self.tuning_validation_dates.max()
            if len(self.tuning_validation_dates)
            else pd.NaT
        )

    @property
    def final_test_start(self) -> pd.Timestamp:
        return self.final_test_dates.min() if len(self.final_test_dates) else pd.NaT

    @property
    def final_test_end(self) -> pd.Timestamp:
        return self.final_test_dates.max() if len(self.final_test_dates) else pd.NaT


@dataclass(frozen=True)
class WalkForwardBoundary:
    fold: int
    train_start_idx: int
    train_end_idx: int
    purge_start_idx: int
    purge_end_idx: int
    test_start_idx: int
    test_end_idx: int
    train_excluded_idx: tuple[int, ...] = ()


def walk_forward_boundaries(date_count: int, config: WalkForwardConfig) -> list[WalkForwardBoundary]:
    if date_count <= 0:
        return []

    boundaries: list[WalkForwardBoundary] = []
    start = 0
    fold_id = 0
    embargoed_train_indices: set[int] = set()
    while True:
        train_start_idx = 0 if config.window_mode == "expanding" else start
        train_end_idx = start + config.train_periods
        purge_start_idx = train_end_idx
        purge_end_idx = train_end_idx + config.gap_periods
        test_start_idx = purge_end_idx
        if test_start_idx >= date_count:
            break
        test_end_idx = min(test_start_idx + config.test_periods, date_count)
        if train_end_idx <= train_start_idx or test_end_idx <= test_start_idx:
            break
        boundaries.append(
            WalkForwardBoundary(
                fold=fold_id,
                train_start_idx=train_start_idx,
                train_end_idx=train_end_idx,
                purge_start_idx=purge_start_idx,
                purge_end_idx=purge_end_idx,
                test_start_idx=test_start_idx,
                test_end_idx=test_end_idx,
                train_excluded_idx=tuple(
                    sorted(
                        index
                        for index in embargoed_train_indices
                        if train_start_idx <= index < train_end_idx
                    )
                ),
            )
        )
        fold_id += 1
        embargo_periods = max(0, int(config.embargo_periods))
        embargoed_train_indices.update(range(test_end_idx, min(test_end_idx + embargo_periods, date_count)))
        start = test_end_idx + embargo_periods
        if test_end_idx >= date_count:
            break
    return boundaries


def walk_forward_splits(
    frame: pd.DataFrame,
    config: WalkForwardConfig,
    candidate_dates: pd.DatetimeIndex | None = None,
) -> list[WalkForwardFold]:
    if candidate_dates is None:
        if "date" not in frame.columns:
            return []
        dates = pd.DatetimeIndex(sorted(pd.to_datetime(frame["date"]).dt.normalize().dropna().unique()))
    else:
        dates = pd.DatetimeIndex(sorted(pd.to_datetime(candidate_dates).dropna().unique()))
    if len(dates) == 0:
        return []

    folds: list[WalkForwardFold] = []
    for boundary in walk_forward_boundaries(len(dates), config):
        train_dates = _boundary_train_dates(dates, boundary)
        purge_dates = dates[boundary.purge_start_idx:boundary.purge_end_idx]
        test_dates = dates[boundary.test_start_idx:boundary.test_end_idx]
        embargo_dates = dates[
            boundary.test_end_idx:min(boundary.test_end_idx + config.embargo_periods, len(dates))
        ]
        if len(train_dates) == 0 or len(test_dates) == 0:
            break
        folds.append(
            WalkForwardFold(
                fold=boundary.fold,
                train_start=train_dates.min(),
                train_end=train_dates.max(),
                test_start=test_dates.min(),
                test_end=test_dates.max(),
                train_dates=train_dates,
                purge_dates=purge_dates,
                test_dates=test_dates,
                embargo_dates=embargo_dates,
            )
        )
    _raise_for_walk_forward_temporal_integrity_issues(folds, config)
    return folds


def validate_walk_forward_temporal_integrity(
    folds: list[WalkForwardFold],
    config: WalkForwardConfig | None = None,
) -> tuple[str, ...]:
    """Validate that each walk-forward fold trains only on dates before its holdout window."""

    issues: list[str] = []
    for fold in folds:
        fold_label = f"fold {fold.fold}"
        segments = {
            "train": pd.DatetimeIndex(fold.train_dates),
            "purge": pd.DatetimeIndex(fold.purge_dates),
            "validation_test": pd.DatetimeIndex(fold.test_dates),
            "embargo": pd.DatetimeIndex(fold.embargo_dates),
        }
        for segment_name, dates in segments.items():
            if not dates.is_unique:
                issues.append(f"{fold_label} {segment_name} dates contain duplicates")
            if len(dates) > 1 and not dates.is_monotonic_increasing:
                issues.append(f"{fold_label} {segment_name} dates are not monotonic")

        train_dates = segments["train"]
        purge_dates = segments["purge"]
        test_dates = segments["validation_test"]
        embargo_dates = segments["embargo"]
        if len(train_dates) == 0:
            issues.append(f"{fold_label} train window is empty")
        if len(test_dates) == 0:
            issues.append(f"{fold_label} validation/test window is empty")

        segment_items = list(segments.items())
        for left_index, (left_name, left_dates) in enumerate(segment_items):
            for right_name, right_dates in segment_items[left_index + 1:]:
                overlap = set(left_dates).intersection(set(right_dates))
                if overlap:
                    issues.append(
                        f"{fold_label} {left_name} dates overlap {right_name} dates"
                    )

        if len(train_dates) and len(test_dates) and train_dates.max() >= test_dates.min():
            issues.append(
                f"{fold_label} train window includes validation/test future dates"
            )
        if len(train_dates) and len(purge_dates) and train_dates.max() >= purge_dates.min():
            issues.append(f"{fold_label} train window overlaps or follows purge window")
        if len(purge_dates) and len(test_dates) and purge_dates.max() >= test_dates.min():
            issues.append(f"{fold_label} purge window overlaps or follows validation/test window")
        if len(test_dates) and len(embargo_dates) and test_dates.max() >= embargo_dates.min():
            issues.append(f"{fold_label} validation/test window overlaps or follows embargo window")

        if config is not None:
            horizon = int(config.prediction_horizon_periods)
            if len(purge_dates) < horizon:
                issues.append(
                    f"{fold_label} purge window is shorter than prediction horizon "
                    f"({len(purge_dates)} < {horizon})"
                )
            if int(config.embargo_periods) < horizon:
                issues.append(
                    f"{fold_label} embargo_periods is shorter than prediction horizon "
                    f"({config.embargo_periods} < {horizon})"
                )
    return tuple(issues)


def build_walk_forward_hyperparameter_tuning_scopes(
    folds: list[WalkForwardFold],
    *,
    validation_periods: int | None = None,
) -> list[WalkForwardHyperparameterTuningScope]:
    """Reserve leakage-safe inner validation windows for fold-local tuning.

    The returned windows are drawn only from each fold's train dates. The outer
    walk-forward test dates remain final holdout dates and are not available to
    hyperparameter selection.
    """

    scopes: list[WalkForwardHyperparameterTuningScope] = []
    for fold in folds:
        train_dates = pd.DatetimeIndex(fold.train_dates)
        if len(train_dates) < 2:
            tuning_validation_dates = pd.DatetimeIndex([])
            tuning_train_dates = train_dates
        else:
            requested_validation_periods = (
                max(1, int(validation_periods))
                if validation_periods is not None
                else max(1, min(len(train_dates) // 5, len(fold.test_dates) or 1))
            )
            tuning_validation_count = min(requested_validation_periods, len(train_dates) - 1)
            tuning_train_dates = train_dates[:-tuning_validation_count]
            tuning_validation_dates = train_dates[-tuning_validation_count:]
        scopes.append(
            WalkForwardHyperparameterTuningScope(
                fold=fold.fold,
                tuning_train_dates=pd.DatetimeIndex(tuning_train_dates),
                tuning_validation_dates=pd.DatetimeIndex(tuning_validation_dates),
                final_test_dates=pd.DatetimeIndex(fold.test_dates),
            )
        )
    return scopes


def validate_walk_forward_hyperparameter_tuning_scopes(
    scopes: list[WalkForwardHyperparameterTuningScope],
) -> tuple[str, ...]:
    """Validate that hyperparameter tuning never observes outer test dates."""

    issues: list[str] = []
    for scope in scopes:
        fold_label = f"fold {scope.fold}"
        tuning_dates = pd.DatetimeIndex(scope.tuning_train_dates).append(
            pd.DatetimeIndex(scope.tuning_validation_dates)
        )
        final_test_dates = pd.DatetimeIndex(scope.final_test_dates)
        for segment_name, dates in {
            "tuning_train": pd.DatetimeIndex(scope.tuning_train_dates),
            "tuning_validation": pd.DatetimeIndex(scope.tuning_validation_dates),
            "final_test": final_test_dates,
        }.items():
            if not dates.is_unique:
                issues.append(f"{fold_label} {segment_name} dates contain duplicates")
            if len(dates) > 1 and not dates.is_monotonic_increasing:
                issues.append(f"{fold_label} {segment_name} dates are not monotonic")

        if len(scope.tuning_train_dates) == 0:
            issues.append(f"{fold_label} tuning train window is empty")
        if len(scope.tuning_validation_dates) == 0:
            issues.append(f"{fold_label} tuning validation window is empty")
        if len(final_test_dates) == 0:
            issues.append(f"{fold_label} final test window is empty")

        overlap = set(tuning_dates).intersection(set(final_test_dates))
        if overlap:
            issues.append(f"{fold_label} hyperparameter tuning dates overlap final test dates")
        if len(tuning_dates) and len(final_test_dates) and tuning_dates.max() >= final_test_dates.min():
            issues.append(f"{fold_label} hyperparameter tuning dates include final test future dates")
        if (
            len(scope.tuning_train_dates)
            and len(scope.tuning_validation_dates)
            and scope.tuning_train_dates.max() >= scope.tuning_validation_dates.min()
        ):
            issues.append(f"{fold_label} tuning train window overlaps or follows tuning validation window")
    return tuple(issues)


def _raise_for_walk_forward_hyperparameter_tuning_scope_issues(
    scopes: list[WalkForwardHyperparameterTuningScope],
) -> None:
    issues = validate_walk_forward_hyperparameter_tuning_scopes(scopes)
    if issues:
        raise ValueError(
            "walk-forward hyperparameter tuning scope failed: " + "; ".join(issues)
        )


def _hyperparameter_tuning_summary_fields(
    scope: WalkForwardHyperparameterTuningScope,
) -> dict[str, object]:
    tuning_dates = pd.DatetimeIndex(scope.tuning_train_dates).append(
        pd.DatetimeIndex(scope.tuning_validation_dates)
    )
    return {
        "hyperparameter_tuning_status": "pass",
        "hyperparameter_tuning_policy": "fold_inner_train_validation_only",
        "hyperparameter_tuning_scope": "fold_train_only",
        "hyperparameter_tuning_uses_final_test": False,
        "hyperparameter_tuning_train_start": scope.tuning_train_start,
        "hyperparameter_tuning_train_end": scope.tuning_train_end,
        "hyperparameter_tuning_validation_start": scope.tuning_validation_start,
        "hyperparameter_tuning_validation_end": scope.tuning_validation_end,
        "hyperparameter_tuning_train_date_count": int(len(scope.tuning_train_dates)),
        "hyperparameter_tuning_validation_date_count": int(
            len(scope.tuning_validation_dates)
        ),
        "hyperparameter_tuning_final_test_start": scope.final_test_start,
        "hyperparameter_tuning_final_test_end": scope.final_test_end,
        "hyperparameter_tuning_latest_allowed_date": (
            tuning_dates.max() if len(tuning_dates) else pd.NaT
        ),
    }


def _raise_for_walk_forward_temporal_integrity_issues(
    folds: list[WalkForwardFold],
    config: WalkForwardConfig | None = None,
) -> None:
    issues = validate_walk_forward_temporal_integrity(folds, config)
    if issues:
        raise ValueError(
            "walk-forward split temporal integrity failed: " + "; ".join(issues)
        )


def _oos_fold_ids(folds: list[WalkForwardFold]) -> set[int]:
    """Return the canonical OOS holdout fold ids for Stage 1 validation."""

    return {fold.fold for fold in folds[-CANONICAL_MIN_OOS_FOLDS:]}


def _boundary_train_dates(
    dates: pd.DatetimeIndex,
    boundary: WalkForwardBoundary,
) -> pd.DatetimeIndex:
    train_dates = dates[boundary.train_start_idx:boundary.train_end_idx]
    if not boundary.train_excluded_idx:
        return train_dates
    excluded_positions = [
        index - boundary.train_start_idx
        for index in boundary.train_excluded_idx
        if boundary.train_start_idx <= index < boundary.train_end_idx
    ]
    if not excluded_positions:
        return train_dates
    return train_dates.delete(excluded_positions)


def walk_forward_predict(
    frame: pd.DataFrame,
    config: WalkForwardConfig,
    target: str = "forward_return_1",
    *,
    splitter: WalkForwardSplitter | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = _config_for_target(config, target)
    missing = {"date", "ticker", target}.difference(frame.columns)
    if missing:
        reason = f"walk-forward input is missing required columns: {sorted(missing)}"
        return _empty_prediction_frame(target), _insufficient_data_summary(
            frame,
            config,
            target,
            skip_code="missing_required_columns",
            reason=reason,
            missing_columns=sorted(missing),
        )

    _validate_walk_forward_feature_cutoffs(frame)
    feature_columns = infer_feature_columns(frame, target)
    predictions: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []

    labeled_dates = pd.DatetimeIndex(
        pd.to_datetime(frame.loc[frame[target].notna(), "date"]).dropna().dt.normalize().unique()
    )
    folds = (
        splitter.split(frame, candidate_dates=labeled_dates)
        if splitter is not None
        else walk_forward_splits(frame, config, candidate_dates=labeled_dates)
    )
    _raise_for_walk_forward_temporal_integrity_issues(folds, config)
    if not folds:
        reason = (
            "not enough labeled dates to create a walk-forward fold "
            f"(labeled_date_count={len(labeled_dates)}, "
            f"required_min_date_count={_minimum_labeled_dates_for_split(config)})"
        )
        predictions = _deterministic_fallback_predictions(
            frame,
            target,
            skip_code="insufficient_labeled_dates",
            reason=reason,
        )
        return predictions, _insufficient_data_summary(
            frame,
            config,
            target,
            labeled_dates=labeled_dates,
            skip_code="insufficient_labeled_dates",
            reason=reason,
            prediction_count=len(predictions),
            deterministic_fallback_applied=True,
            deterministic_fallback_policy=DETERMINISTIC_FALLBACK_POLICY,
            deterministic_fallback_model=DETERMINISTIC_FALLBACK_MODEL_NAME,
        )
    tuning_scopes = build_walk_forward_hyperparameter_tuning_scopes(folds)
    _raise_for_walk_forward_hyperparameter_tuning_scope_issues(tuning_scopes)
    tuning_scopes_by_fold = {scope.fold: scope for scope in tuning_scopes}
    oos_fold_ids = _oos_fold_ids(folds)
    skipped_folds: list[dict[str, object]] = []

    for fold in folds:
        train = frame[frame["date"].isin(fold.train_dates)]
        test = frame[frame["date"].isin(fold.test_dates)]
        train = train.dropna(subset=[target])
        if len(train) < config.min_train_observations or test.empty:
            skipped_folds.append(
                {
                    "fold": fold.fold,
                    "train_observations": int(len(train)),
                    "test_observations": int(len(test)),
                    "min_train_observations": int(config.min_train_observations),
                    "reason": (
                        "training observations are below the configured minimum"
                        if len(train) < config.min_train_observations
                        else "test window is empty"
                    ),
                }
            )
            continue

        fallback_code, fallback_reason = _fold_deterministic_fallback_reason(
            train,
            test,
            feature_columns,
        )
        if fallback_code is not None:
            pred = _deterministic_fallback_predictions(
                test,
                target,
                fold=fold.fold,
                is_oos=fold.fold in oos_fold_ids,
                skip_code=fallback_code,
                reason=str(fallback_reason),
            )
            predictions.append(pred)
            summaries.append(
                _fold_fallback_summary_row(
                    fold,
                    config,
                    target,
                    train=train,
                    test=test,
                    pred=pred,
                    oos_fold_ids=oos_fold_ids,
                    skip_code=fallback_code,
                    reason=str(fallback_reason),
                    tuning_scope=tuning_scopes_by_fold[fold.fold],
                )
            )
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
        pred["is_oos"] = fold.fold in oos_fold_ids
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
                "validation_start": pd.NaT if fold.fold in oos_fold_ids else fold.test_start,
                "validation_end": pd.NaT if fold.fold in oos_fold_ids else fold.test_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "oos_test_start": fold.test_start if fold.fold in oos_fold_ids else pd.NaT,
                "oos_test_end": fold.test_end if fold.fold in oos_fold_ids else pd.NaT,
                "train_periods": int(config.train_periods),
                "validation_periods": int(config.test_periods),
                "test_periods": int(config.test_periods),
                "window_mode": config.window_mode,
                "target_column": target,
                "target_horizon": int(config.prediction_horizon_periods),
                "target_horizon_periods": int(config.prediction_horizon_periods),
                "prediction_horizon_periods": int(config.prediction_horizon_periods),
                "requested_gap_periods": int(config.requested_gap_periods),
                "requested_embargo_periods": int(config.requested_embargo_periods),
                "effective_gap_periods": int(config.gap_periods),
                "effective_embargo_periods": int(config.embargo_periods),
                "gap_periods": int(config.gap_periods),
                "purge_periods": int(config.gap_periods),
                "purge_gap_periods": int(config.gap_periods),
                "purged_date_count": int(len(fold.purge_dates)),
                "purge_start": fold.purge_dates.min() if len(fold.purge_dates) else pd.NaT,
                "purge_end": fold.purge_dates.max() if len(fold.purge_dates) else pd.NaT,
                "purge_applied": bool(len(fold.purge_dates) >= config.prediction_horizon_periods),
                "embargo_periods": int(config.embargo_periods),
                "embargoed_date_count": int(len(fold.embargo_dates)),
                "embargo_start": fold.embargo_dates.min() if len(fold.embargo_dates) else pd.NaT,
                "embargo_end": fold.embargo_dates.max() if len(fold.embargo_dates) else pd.NaT,
                "embargo_applied": bool(config.embargo_periods >= config.prediction_horizon_periods),
                "train_observations": len(train),
                "test_observations": len(test),
                "labeled_test_observations": len(labeled),
                "prediction_count": len(pred),
                "fold_type": "oos" if fold.fold in oos_fold_ids else "validation",
                "temporal_integrity_status": "pass",
                "train_validation_test_order_valid": True,
                "future_data_in_train": False,
                "label_overlap_violations": 0,
                **_hyperparameter_tuning_summary_fields(tuning_scopes_by_fold[fold.fold]),
                "model_name": model.actual_model_name,
                "is_oos": fold.fold in oos_fold_ids,
                "tabular_fallback_reason": model.training_metadata.get("tabular_fallback_reason"),
                "preprocessing_fit_scope": model.training_metadata.get("preprocessing_fit_scope"),
                "preprocessing_transform_scope": model.training_metadata.get("preprocessing_transform_scope"),
                "preprocessing_pipeline_steps": model.training_metadata.get("preprocessing_pipeline_steps"),
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
    if prediction_frame.empty and summary_frame.empty:
        max_train_observations = max(
            (int(row["train_observations"]) for row in skipped_folds),
            default=0,
        )
        max_test_observations = max(
            (int(row["test_observations"]) for row in skipped_folds),
            default=0,
        )
        reason = (
            "walk-forward folds were created but all were skipped because the "
            "dataset is below configured training/test observation requirements"
        )
        predictions = _deterministic_fallback_predictions(
            frame,
            target,
            skip_code="insufficient_fold_observations",
            reason=reason,
        )
        return predictions, _insufficient_data_summary(
            frame,
            config,
            target,
            labeled_dates=labeled_dates,
            folds=folds,
            skip_code="insufficient_fold_observations",
            reason=reason,
            skipped_fold_count=len(skipped_folds),
            max_train_observations=max_train_observations,
            max_test_observations=max_test_observations,
            prediction_count=len(predictions),
            deterministic_fallback_applied=True,
            deterministic_fallback_policy=DETERMINISTIC_FALLBACK_POLICY,
            deterministic_fallback_model=DETERMINISTIC_FALLBACK_MODEL_NAME,
        )
    if prediction_frame.empty:
        prediction_frame = _empty_prediction_frame(target)
    if not summary_frame.empty:
        summary_frame = append_walk_forward_oos_metrics(
            summary_frame,
            prediction_frame,
            target=target,
        )
    return prediction_frame, summary_frame


def _validate_walk_forward_feature_cutoffs(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    sample_timestamp = date_end_utc(frame["date"])
    for column in _walk_forward_availability_cutoff_columns(frame):
        availability = timestamp_utc(frame[column])
        violation = availability.notna() & (availability > sample_timestamp)
        if violation.any():
            first_index = int(np.flatnonzero(violation.to_numpy())[0])
            raise ValueError(
                f"walk-forward input column {column} contains data unavailable at feature date "
                f"{frame.loc[first_index, 'date']}"
            )

    for column in _walk_forward_prediction_cutoff_columns(frame):
        prediction_time = timestamp_utc(frame[column])
        violation = prediction_time.notna() & (prediction_time > sample_timestamp)
        if violation.any():
            first_index = int(np.flatnonzero(violation.to_numpy())[0])
            raise ValueError(
                f"walk-forward prediction column {column} is later than feature date "
                f"{frame.loc[first_index, 'date']}"
            )
    validate_event_availability_order(frame, label="walk-forward input")


def _walk_forward_availability_cutoff_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column == "availability_timestamp" or str(column).endswith("_availability_timestamp")
    ]


def _walk_forward_prediction_cutoff_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in ("prediction_date", "prediction_timestamp", "model_prediction_timestamp")
        if column in frame
    ]


def calculate_walk_forward_fold_metrics(
    predictions: pd.DataFrame,
    *,
    target: str = CANONICAL_WALK_FORWARD_TARGET_COLUMN,
) -> pd.DataFrame:
    """Calculate fold-level validation metrics from walk-forward predictions."""

    required = {"date", "fold", "expected_return", target}
    if predictions.empty or not required.issubset(predictions.columns):
        return _empty_fold_metric_frame()

    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["expected_return"] = pd.to_numeric(frame["expected_return"], errors="coerce")
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    frame = frame.dropna(subset=["date", "fold", "expected_return", target])
    if frame.empty:
        return _empty_fold_metric_frame()

    if "is_oos" not in frame:
        frame["is_oos"] = False
    frame["is_oos"] = frame["is_oos"].fillna(False).astype(bool)

    daily_rank_ic = _daily_rank_ic_frame(frame, target)
    rows: list[dict[str, object]] = []
    for fold, fold_frame in frame.groupby("fold", sort=True):
        fold_daily = (
            daily_rank_ic[daily_rank_ic["fold"].eq(fold)]
            if not daily_rank_ic.empty
            else pd.DataFrame()
        )
        errors = (fold_frame["expected_return"] - fold_frame[target]).abs()
        fold_is_oos = bool(fold_frame["is_oos"].any())
        rank_values = (
            pd.to_numeric(fold_daily["rank_ic"], errors="coerce").dropna()
            if not fold_daily.empty
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "fold": fold,
                "is_oos": fold_is_oos,
                "fold_start": fold_frame["date"].min(),
                "fold_end": fold_frame["date"].max(),
                "fold_prediction_count": int(len(fold_frame)),
                "fold_labeled_prediction_count": int(
                    fold_frame[["expected_return", target]].dropna().shape[0]
                ),
                "fold_mae": float(errors.mean()) if not errors.empty else None,
                "fold_directional_accuracy": float(
                    (fold_frame["expected_return"] * fold_frame[target] > 0).mean()
                )
                if not fold_frame.empty
                else None,
                "fold_information_coefficient": _safe_correlation(
                    fold_frame["expected_return"],
                    fold_frame[target],
                ),
                "fold_rank_ic": float(rank_values.mean()) if not rank_values.empty else None,
                "fold_rank_ic_count": int(len(rank_values)),
                "fold_positive_rank_ic_day_ratio": (
                    float((rank_values > 0).mean()) if not rank_values.empty else None
                ),
            }
        )
    return pd.DataFrame(rows)


def calculate_walk_forward_oos_summary(
    fold_metrics: pd.DataFrame,
) -> dict[str, object]:
    """Summarize OOS performance across walk-forward folds."""

    if fold_metrics.empty:
        return {
            "oos_fold_count": 0,
            "oos_rank_ic": None,
            "oos_mean_rank_ic": None,
            "oos_rank_ic_positive_fold_ratio": None,
            "oos_rank_ic_count": 0,
            "oos_prediction_count": 0,
            "oos_labeled_prediction_count": 0,
            "oos_mean_mae": None,
            "oos_mean_directional_accuracy": None,
            "oos_mean_information_coefficient": None,
            "oos_start": pd.NaT,
            "oos_end": pd.NaT,
        }

    oos_mask = (
        fold_metrics["is_oos"].fillna(False).astype(bool)
        if "is_oos" in fold_metrics
        else pd.Series(False, index=fold_metrics.index)
    )
    oos = fold_metrics.loc[oos_mask].copy()
    if oos.empty:
        return calculate_walk_forward_oos_summary(pd.DataFrame())

    rank_ic = pd.to_numeric(oos.get("fold_rank_ic"), errors="coerce").dropna()
    return {
        "oos_fold_count": int(len(oos)),
        "oos_rank_ic": float(rank_ic.mean()) if not rank_ic.empty else None,
        "oos_mean_rank_ic": float(rank_ic.mean()) if not rank_ic.empty else None,
        "oos_rank_ic_positive_fold_ratio": (
            float((rank_ic > 0).mean()) if not rank_ic.empty else None
        ),
        "oos_rank_ic_count": int(
            _numeric_series_or_empty(oos.get("fold_rank_ic_count")).fillna(0).sum()
        ),
        "oos_prediction_count": int(
            _numeric_series_or_empty(oos.get("fold_prediction_count")).fillna(0).sum()
        ),
        "oos_labeled_prediction_count": int(
            _numeric_series_or_empty(oos.get("fold_labeled_prediction_count")).fillna(0).sum()
        ),
        "oos_mean_mae": _mean_or_none(oos.get("fold_mae")),
        "oos_mean_directional_accuracy": _mean_or_none(
            oos.get("fold_directional_accuracy")
        ),
        "oos_mean_information_coefficient": _mean_or_none(
            oos.get("fold_information_coefficient")
        ),
        "oos_start": oos["fold_start"].min() if "fold_start" in oos else pd.NaT,
        "oos_end": oos["fold_end"].max() if "fold_end" in oos else pd.NaT,
    }


def append_walk_forward_oos_metrics(
    summary: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    target: str = CANONICAL_WALK_FORWARD_TARGET_COLUMN,
) -> pd.DataFrame:
    """Attach fold-level metrics and OOS aggregate metrics to validation summary."""

    if summary.empty:
        return summary
    enriched = summary.copy()
    fold_metrics = calculate_walk_forward_fold_metrics(predictions, target=target)
    if not fold_metrics.empty and "fold" in enriched:
        metric_columns = [column for column in fold_metrics.columns if column != "is_oos"]
        enriched = enriched.merge(
            fold_metrics[metric_columns],
            on="fold",
            how="left",
            suffixes=("", "_computed"),
        )
    oos_summary = calculate_walk_forward_oos_summary(fold_metrics)
    all_rank_ic = (
        pd.to_numeric(fold_metrics.get("fold_rank_ic"), errors="coerce").dropna()
        if not fold_metrics.empty
        else pd.Series(dtype=float)
    )
    aggregate_metrics: dict[str, object] = {
        "walk_forward_fold_count": int(len(fold_metrics)),
        "walk_forward_mean_rank_ic": (
            float(all_rank_ic.mean()) if not all_rank_ic.empty else None
        ),
        "walk_forward_positive_rank_ic_fold_ratio": (
            float((all_rank_ic > 0).mean()) if not all_rank_ic.empty else None
        ),
        **oos_summary,
    }
    for column, value in aggregate_metrics.items():
        enriched[column] = value
    return enriched


def _config_for_target(config: WalkForwardConfig, target: str) -> WalkForwardConfig:
    target_horizon = _target_horizon(target) or config.prediction_horizon_periods
    prediction_horizon_periods = max(config.prediction_horizon_periods, target_horizon)
    if (
        config.prediction_horizon_periods == prediction_horizon_periods
        and config.gap_periods >= prediction_horizon_periods
        and config.embargo_periods >= prediction_horizon_periods
    ):
        return config
    return replace(
        config,
        prediction_horizon_periods=prediction_horizon_periods,
        target_horizon=prediction_horizon_periods,
    )


def _target_horizon(target: str) -> int | None:
    prefix = "forward_return_"
    if not str(target).startswith(prefix):
        return None
    try:
        horizon = int(str(target).removeprefix(prefix))
    except ValueError:
        return None
    return horizon if horizon >= 1 else None


def _minimum_labeled_dates_for_split(config: WalkForwardConfig) -> int:
    return int(config.train_periods) + int(config.gap_periods) + 1


def _empty_prediction_frame(target: str) -> pd.DataFrame:
    columns = [
        "date",
        "ticker",
        "raw_expected_return",
        "expected_return",
        "predicted_volatility",
        "downside_quantile",
        "upside_quantile",
        "quantile_width",
        "model_confidence",
        "model_name",
        "model_calibration_scale",
        "model_calibration_bias",
        target,
        "fold",
        "is_oos",
        "deterministic_fallback_applied",
        "deterministic_fallback_policy",
        "deterministic_fallback_code",
        "deterministic_fallback_reason",
    ]
    return pd.DataFrame(columns=list(dict.fromkeys(columns)))


def _deterministic_fallback_predictions(
    frame: pd.DataFrame,
    target: str,
    *,
    fold: object = pd.NA,
    is_oos: bool = False,
    skip_code: str,
    reason: str,
) -> pd.DataFrame:
    """Build deterministic neutral predictions for non-evaluable data slices."""

    if not {"date", "ticker"}.issubset(frame.columns):
        return _empty_prediction_frame(target)
    output = frame[["date", "ticker"]].copy()
    if target in frame:
        output[target] = pd.to_numeric(frame[target], errors="coerce").to_numpy()
    else:
        output[target] = np.nan
    output["raw_expected_return"] = 0.0
    output["expected_return"] = 0.0
    output["predicted_volatility"] = _fallback_volatility(frame).to_numpy()
    output["downside_quantile"] = -output["predicted_volatility"]
    output["upside_quantile"] = output["predicted_volatility"]
    output["quantile_width"] = output["upside_quantile"] - output["downside_quantile"]
    output["model_confidence"] = 0.0
    output["model_name"] = DETERMINISTIC_FALLBACK_MODEL_NAME
    output["model_calibration_scale"] = 1.0
    output["model_calibration_bias"] = 0.0
    output["fold"] = fold
    output["is_oos"] = bool(is_oos)
    output["deterministic_fallback_applied"] = True
    output["deterministic_fallback_policy"] = DETERMINISTIC_FALLBACK_POLICY
    output["deterministic_fallback_code"] = skip_code
    output["deterministic_fallback_reason"] = reason
    return output[_empty_prediction_frame(target).columns]


def _fallback_volatility(frame: pd.DataFrame) -> pd.Series:
    for column in ("volatility_20", "realized_volatility", "volatility"):
        if column not in frame:
            continue
        volatility = pd.to_numeric(frame[column], errors="coerce")
        fallback = volatility.replace(0.0, np.nan).median()
        if pd.isna(fallback):
            fallback = 0.02
        return volatility.fillna(fallback).clip(lower=0.001, upper=0.25)
    return pd.Series(0.02, index=frame.index)


def _fold_deterministic_fallback_reason(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[str | None, str | None]:
    train_tickers = _ticker_count(train)
    test_tickers = _ticker_count(test)
    if min(train_tickers, test_tickers) < 2:
        return (
            "single_asset_universe",
            "walk-forward fold has fewer than two tickers, so cross-sectional validation is not evaluable",
        )
    if not feature_columns:
        return (
            "missing_model_features",
            "walk-forward fold has no numeric model feature columns after leakage-safe exclusions",
        )
    train_features = _numeric_feature_frame(train, feature_columns)
    test_features = _numeric_feature_frame(test, feature_columns)
    if train_features.empty or test_features.empty:
        return (
            "missing_model_features",
            "walk-forward fold has no numeric model feature values",
        )
    if int(np.isfinite(train_features.to_numpy(dtype=float)).sum()) == 0:
        return (
            "all_missing_train_features",
            "all training feature values are missing or non-finite",
        )
    if int(np.isfinite(test_features.to_numpy(dtype=float)).sum()) == 0:
        return (
            "all_missing_test_features",
            "all test feature values are missing or non-finite",
        )
    return None, None


def _ticker_count(frame: pd.DataFrame) -> int:
    if "ticker" not in frame:
        return 0
    return int(frame["ticker"].dropna().astype(str).str.strip().replace("", np.nan).dropna().nunique())


def _numeric_feature_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    if not feature_columns:
        return pd.DataFrame(index=frame.index)
    return frame.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce")


def _fold_fallback_summary_row(
    fold: WalkForwardFold,
    config: WalkForwardConfig,
    target: str,
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    pred: pd.DataFrame,
    oos_fold_ids: set[int],
    skip_code: str,
    reason: str,
    tuning_scope: WalkForwardHyperparameterTuningScope,
) -> dict[str, object]:
    labeled = pred.dropna(subset=[target])
    is_oos = fold.fold in oos_fold_ids
    return {
        "fold": fold.fold,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "validation_start": pd.NaT if is_oos else fold.test_start,
        "validation_end": pd.NaT if is_oos else fold.test_end,
        "test_start": fold.test_start,
        "test_end": fold.test_end,
        "oos_test_start": fold.test_start if is_oos else pd.NaT,
        "oos_test_end": fold.test_end if is_oos else pd.NaT,
        "train_periods": int(config.train_periods),
        "validation_periods": int(config.test_periods),
        "test_periods": int(config.test_periods),
        "window_mode": config.window_mode,
        "target_column": target,
        "target_horizon": int(config.prediction_horizon_periods),
        "target_horizon_periods": int(config.prediction_horizon_periods),
        "prediction_horizon_periods": int(config.prediction_horizon_periods),
        "requested_gap_periods": int(config.requested_gap_periods),
        "requested_embargo_periods": int(config.requested_embargo_periods),
        "effective_gap_periods": int(config.gap_periods),
        "effective_embargo_periods": int(config.embargo_periods),
        "gap_periods": int(config.gap_periods),
        "purge_periods": int(config.gap_periods),
        "purge_gap_periods": int(config.gap_periods),
        "purged_date_count": int(len(fold.purge_dates)),
        "purge_start": fold.purge_dates.min() if len(fold.purge_dates) else pd.NaT,
        "purge_end": fold.purge_dates.max() if len(fold.purge_dates) else pd.NaT,
        "purge_applied": bool(len(fold.purge_dates) >= config.prediction_horizon_periods),
        "embargo_periods": int(config.embargo_periods),
        "embargoed_date_count": int(len(fold.embargo_dates)),
        "embargo_start": fold.embargo_dates.min() if len(fold.embargo_dates) else pd.NaT,
        "embargo_end": fold.embargo_dates.max() if len(fold.embargo_dates) else pd.NaT,
        "embargo_applied": bool(config.embargo_periods >= config.prediction_horizon_periods),
        "train_observations": len(train),
        "test_observations": len(test),
        "labeled_test_observations": len(labeled),
        "prediction_count": len(pred),
        "fold_type": "oos" if is_oos else "validation",
        "temporal_integrity_status": "pass",
        "train_validation_test_order_valid": True,
        "future_data_in_train": False,
        "label_overlap_violations": 0,
        **_hyperparameter_tuning_summary_fields(tuning_scope),
        "model_name": DETERMINISTIC_FALLBACK_MODEL_NAME,
        "is_oos": is_oos,
        "validation_status": WALK_FORWARD_INSUFFICIENT_DATA_STATUS,
        "skip_status": WALK_FORWARD_SKIPPED_STATUS,
        "skip_code": skip_code,
        "reason": reason,
        "deterministic_fallback_applied": True,
        "deterministic_fallback_policy": DETERMINISTIC_FALLBACK_POLICY,
        "deterministic_fallback_model": DETERMINISTIC_FALLBACK_MODEL_NAME,
        "mae": float(labeled[target].abs().mean()) if not labeled.empty else None,
        "directional_accuracy": float((labeled[target] == 0.0).mean()) if not labeled.empty else None,
        "information_coefficient": None,
        "sign_information_coefficient": None,
        "tabular_fallback_reason": reason,
        "model_calibration_scale": 1.0,
        "model_calibration_bias": 0.0,
        "winsorized_feature_count": 0,
    }


def _empty_fold_metric_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "fold",
            "is_oos",
            "fold_start",
            "fold_end",
            "fold_prediction_count",
            "fold_labeled_prediction_count",
            "fold_mae",
            "fold_directional_accuracy",
            "fold_information_coefficient",
            "fold_rank_ic",
            "fold_rank_ic_count",
            "fold_positive_rank_ic_day_ratio",
        ]
    )


def _insufficient_data_summary(
    frame: pd.DataFrame,
    config: WalkForwardConfig,
    target: str,
    *,
    skip_code: str,
    reason: str,
    labeled_dates: pd.DatetimeIndex | None = None,
    folds: list[WalkForwardFold] | None = None,
    **extra: object,
) -> pd.DataFrame:
    if labeled_dates is None:
        if {"date", target}.issubset(frame.columns):
            labeled_dates = pd.DatetimeIndex(
                pd.to_datetime(frame.loc[frame[target].notna(), "date"])
                .dropna()
                .dt.normalize()
                .unique()
            )
        else:
            labeled_dates = pd.DatetimeIndex([])
    all_dates = (
        pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="coerce").dropna().dt.normalize().unique())
        if "date" in frame
        else pd.DatetimeIndex([])
    )
    labeled_prediction_count = (
        int(frame[target].notna().sum()) if target in frame else 0
    )
    row: dict[str, object] = {
        "fold": pd.NA,
        "fold_type": WALK_FORWARD_SKIPPED_STATUS,
        "temporal_integrity_status": WALK_FORWARD_SKIPPED_STATUS,
        "train_validation_test_order_valid": pd.NA,
        "future_data_in_train": pd.NA,
        "hyperparameter_tuning_status": WALK_FORWARD_SKIPPED_STATUS,
        "hyperparameter_tuning_policy": "fold_inner_train_validation_only",
        "hyperparameter_tuning_scope": "fold_train_only",
        "hyperparameter_tuning_uses_final_test": False,
        "hyperparameter_tuning_train_start": pd.NaT,
        "hyperparameter_tuning_train_end": pd.NaT,
        "hyperparameter_tuning_validation_start": pd.NaT,
        "hyperparameter_tuning_validation_end": pd.NaT,
        "hyperparameter_tuning_train_date_count": 0,
        "hyperparameter_tuning_validation_date_count": 0,
        "hyperparameter_tuning_final_test_start": pd.NaT,
        "hyperparameter_tuning_final_test_end": pd.NaT,
        "hyperparameter_tuning_latest_allowed_date": pd.NaT,
        "is_oos": False,
        "validation_status": WALK_FORWARD_INSUFFICIENT_DATA_STATUS,
        "skip_status": WALK_FORWARD_SKIPPED_STATUS,
        "skip_code": skip_code,
        "reason": reason,
        "fold_count": 0,
        "candidate_fold_count": int(len(folds or [])),
        "candidate_date_count": int(len(all_dates)),
        "labeled_date_count": int(len(labeled_dates)),
        "labeled_observation_count": labeled_prediction_count,
        "required_min_date_count": _minimum_labeled_dates_for_split(config),
        "train_start": pd.NaT,
        "train_end": pd.NaT,
        "validation_start": pd.NaT,
        "validation_end": pd.NaT,
        "test_start": pd.NaT,
        "test_end": pd.NaT,
        "oos_test_start": pd.NaT,
        "oos_test_end": pd.NaT,
        "train_periods": int(config.train_periods),
        "validation_periods": int(config.test_periods),
        "test_periods": int(config.test_periods),
        "window_mode": config.window_mode,
        "gap_periods": int(config.gap_periods),
        "purge_periods": int(config.gap_periods),
        "purge_gap_periods": int(config.gap_periods),
        "purged_date_count": 0,
        "purge_start": pd.NaT,
        "purge_end": pd.NaT,
        "purge_applied": False,
        "embargo_periods": int(config.embargo_periods),
        "embargoed_date_count": 0,
        "embargo_start": pd.NaT,
        "embargo_end": pd.NaT,
        "embargo_applied": bool(config.embargo_periods >= config.prediction_horizon_periods),
        "prediction_horizon_periods": int(config.prediction_horizon_periods),
        "min_train_observations": int(config.min_train_observations),
        "target_column": target,
        "train_observations": 0,
        "test_observations": 0,
        "labeled_test_observations": 0,
        "prediction_count": 0,
        "mae": None,
        "directional_accuracy": None,
        "information_coefficient": None,
        "sign_information_coefficient": None,
    }
    row.update(extra)
    return pd.DataFrame([row])


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


def _safe_spearman(left: pd.Series, right: pd.Series) -> float | None:
    left_series = pd.to_numeric(left, errors="coerce")
    right_series = pd.to_numeric(right, errors="coerce")
    common = left_series.notna() & right_series.notna()
    if common.sum() < 2:
        return None
    return _safe_correlation(left_series.loc[common].rank(), right_series.loc[common].rank())


def _daily_rank_ic_frame(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (fold, date), day in frame.groupby(["fold", "date"], sort=True):
        rank_ic = _safe_spearman(day["expected_return"], day[target])
        if rank_ic is None:
            continue
        rows.append(
            {
                "fold": fold,
                "date": date,
                "rank_ic": rank_ic,
                "observation_count": int(len(day)),
            }
        )
    return pd.DataFrame(rows, columns=["fold", "date", "rank_ic", "observation_count"])


def _mean_or_none(values: object) -> float | None:
    series = _numeric_series_or_empty(values).dropna()
    if series.empty:
        return None
    return float(series.mean())


def _numeric_series_or_empty(values: object) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    if np.isscalar(values):
        return pd.Series([pd.to_numeric(values, errors="coerce")], dtype=float)
    return pd.to_numeric(pd.Series(values), errors="coerce")
