from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

import quant_research.models.tabular as tabular
import quant_research.validation.walk_forward as walk_forward
from quant_research.data.market import SyntheticMarketDataProvider
from quant_research.features.price import build_price_features
from quant_research.validation.walk_forward import (
    DEFAULT_PURGE_EMBARGO_WALK_FORWARD_CONFIG,
    PurgeEmbargoWalkForwardConfig,
    PurgeEmbargoWalkForwardSplitter,
    WalkForwardConfig,
    WalkForwardHyperparameterTuningScope,
    build_walk_forward_hyperparameter_tuning_scopes,
    walk_forward_boundaries,
    walk_forward_predict,
    walk_forward_splits,
)


def test_canonical_purge_embargo_splitter_config_defines_stage1_contract() -> None:
    config = PurgeEmbargoWalkForwardConfig()

    assert config == DEFAULT_PURGE_EMBARGO_WALK_FORWARD_CONFIG
    assert config.train_periods == 252
    assert config.test_periods == 60
    assert config.purge_periods == 20
    assert config.gap_periods == 20
    assert config.embargo_periods == 20
    assert config.target_column == "forward_return_20"
    assert config.target_horizon_periods == 20
    assert config.evaluation_mode == "horizon_consistent"
    assert config.is_system_valid is True

    payload = config.to_dict()
    assert payload["target_column"] == "forward_return_20"
    assert payload["target_horizon_periods"] == 20
    assert payload["system_validity_issues"] == []


def test_purge_embargo_splitter_config_flags_forward_return_20_zero_embargo() -> None:
    config = PurgeEmbargoWalkForwardConfig(embargo_periods=0)

    assert config.is_system_valid is False
    assert "embargo_periods must be at least the target horizon" in config.system_validity_issues()
    assert "forward_return_20 requires non-zero embargo_periods" in config.system_validity_issues()


def test_purge_embargo_splitter_delegates_to_existing_walk_forward_splits() -> None:
    dates = pd.bdate_range(start="2026-01-01", periods=24)
    frame = pd.DataFrame(
        {
            "date": dates,
            "feature": 1.0,
            "ticker": ["AAPL"] * len(dates),
            "forward_return_1": [0.01] * len(dates),
        }
    )
    config = PurgeEmbargoWalkForwardConfig(
        train_periods=5,
        test_periods=2,
        purge_periods=3,
        embargo_periods=3,
        target_column="forward_return_1",
    )
    splitter = PurgeEmbargoWalkForwardSplitter(config)

    assert isinstance(splitter.config.to_walk_forward_config(), WalkForwardConfig)
    assert splitter.boundaries(len(dates)) == walk_forward_boundaries(
        len(dates),
        config.to_walk_forward_config(),
    )
    assert splitter.boundaries(len(dates))
    assert _fold_signature(splitter.split(frame)) == _fold_signature(
        walk_forward_splits(frame, config.to_walk_forward_config())
    )


def test_walk_forward_predict_uses_injected_splitter_to_build_folds() -> None:
    dates = pd.bdate_range(start="2026-01-01", periods=36)
    frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAPL"] * len(dates),
            "return_1": np.linspace(-0.01, 0.01, len(dates)),
            "liquidity_score": 1.0,
            "forward_return_1": np.linspace(-0.02, 0.02, len(dates)),
        }
    )
    splitter = PurgeEmbargoWalkForwardSplitter(
        PurgeEmbargoWalkForwardConfig(
            train_periods=10,
            test_periods=3,
            purge_periods=2,
            embargo_periods=2,
            target_column="forward_return_1",
        )
    )

    predictions, summary = walk_forward_predict(
        frame,
        WalkForwardConfig(
            train_periods=99,
            test_periods=99,
            gap_periods=99,
            min_train_observations=5,
        ),
        target="forward_return_1",
        splitter=splitter,
    )

    expected_folds = splitter.split(
        frame,
        candidate_dates=pd.DatetimeIndex(pd.to_datetime(frame["date"]).dt.normalize().unique()),
    )
    assert not predictions.empty
    assert summary["fold"].tolist() == [fold.fold for fold in expected_folds]
    assert summary["train_start"].tolist() == [fold.train_start for fold in expected_folds]
    assert summary["test_start"].tolist() == [fold.test_start for fold in expected_folds]


def test_walk_forward_predict_rejects_features_unavailable_after_feature_date() -> None:
    dates = pd.bdate_range(start="2026-01-01", periods=12)
    frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAPL"] * len(dates),
            "return_1": np.linspace(-0.01, 0.01, len(dates)),
            "news_sentiment_mean": 0.1,
            "news_availability_timestamp": [pd.Timestamp(date, tz="UTC") for date in dates],
            "forward_return_1": np.linspace(-0.02, 0.02, len(dates)),
        }
    )
    frame.loc[3, "news_availability_timestamp"] = pd.Timestamp("2026-01-07 00:00:00", tz="UTC")

    with pytest.raises(ValueError, match="unavailable at feature date"):
        walk_forward_predict(
            frame,
            WalkForwardConfig(
                train_periods=4,
                test_periods=2,
                gap_periods=1,
                min_train_observations=4,
            ),
            target="forward_return_1",
        )


def test_walk_forward_predict_rejects_prediction_timestamp_after_feature_date() -> None:
    dates = pd.bdate_range(start="2026-01-01", periods=12)
    frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAPL"] * len(dates),
            "return_1": np.linspace(-0.01, 0.01, len(dates)),
            "proxy_score": np.linspace(-0.02, 0.02, len(dates)),
            "model_prediction_timestamp": [pd.Timestamp(date, tz="UTC") for date in dates],
            "forward_return_1": np.linspace(-0.02, 0.02, len(dates)),
        }
    )
    frame.loc[4, "model_prediction_timestamp"] = pd.Timestamp("2026-01-08 00:00:00", tz="UTC")

    with pytest.raises(ValueError, match="later than feature date"):
        walk_forward_predict(
            frame,
            WalkForwardConfig(
                train_periods=4,
                test_periods=2,
                gap_periods=1,
                min_train_observations=4,
            ),
            target="forward_return_1",
        )


def test_walk_forward_summary_records_train_validation_test_and_purge_embargo_config(
    monkeypatch,
) -> None:
    class RecordingModel:
        actual_model_name = "recording"

        def __init__(self, **kwargs) -> None:
            self.training_metadata = {
                "tabular_fallback_reason": None,
                "calibration_scale": 1.0,
                "calibration_bias": 0.0,
                "winsorized_feature_count": 0,
            }

        def fit(self, train: pd.DataFrame, target: str = "forward_return_1") -> RecordingModel:
            return self

        def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
            return frame[["date", "ticker"]].assign(
                raw_expected_return=0.0,
                expected_return=0.0,
                predicted_volatility=0.01,
                downside_quantile=-0.01,
                upside_quantile=0.01,
                quantile_width=0.02,
                model_confidence=0.0,
                model_name=self.actual_model_name,
                model_calibration_scale=1.0,
                model_calibration_bias=0.0,
            )

    monkeypatch.setattr(walk_forward, "TabularReturnModel", RecordingModel)

    dates = pd.bdate_range(start="2026-01-01", periods=55)
    frame = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "feature_a": float(date_index),
                "forward_return_5": 0.01,
            }
            for date_index, date in enumerate(dates)
            for ticker in ("AAPL", "MSFT")
        ]
    )

    _, summary = walk_forward_predict(
        frame,
        WalkForwardConfig(
            train_periods=6,
            test_periods=3,
            gap_periods=1,
            embargo_periods=1,
            min_train_observations=6,
        ),
        target="forward_return_5",
    )

    assert len(summary) >= 2
    required_columns = {
        "train_start",
        "train_end",
        "validation_start",
        "validation_end",
        "test_start",
        "test_end",
        "oos_test_start",
        "oos_test_end",
        "train_periods",
        "validation_periods",
        "test_periods",
        "purge_periods",
        "purge_gap_periods",
        "embargo_periods",
    }
    assert required_columns.issubset(summary.columns)

    validation_rows = summary[~summary["is_oos"]]
    oos_rows = summary[summary["is_oos"]]
    assert not validation_rows.empty
    assert len(oos_rows) == 2
    assert validation_rows["validation_start"].notna().all()
    assert validation_rows["validation_end"].notna().all()
    assert validation_rows["oos_test_start"].isna().all()
    assert oos_rows["validation_start"].isna().all()
    assert oos_rows["validation_end"].isna().all()
    assert oos_rows["oos_test_start"].notna().all()
    assert oos_rows["oos_test_end"].notna().all()

    for _, row in summary.iterrows():
        assert int(row["train_periods"]) == 6
        assert int(row["validation_periods"]) == 3
        assert int(row["test_periods"]) == 3
        assert int(row["purge_periods"]) >= 5
        assert int(row["purge_gap_periods"]) >= 5
        assert int(row["embargo_periods"]) >= 5
        assert pd.Timestamp(row["train_start"]) <= pd.Timestamp(row["train_end"])
        assert pd.Timestamp(row["test_start"]) <= pd.Timestamp(row["test_end"])


def _fold_signature(folds):
    return [
        (
            fold.fold,
            tuple(fold.train_dates),
            tuple(fold.purge_dates),
            tuple(fold.test_dates),
        )
        for fold in folds
    ]


def test_walk_forward_splits_never_train_on_future_dates() -> None:
    data = SyntheticMarketDataProvider(periods=120).get_history(["AAPL", "MSFT"])
    features = build_price_features(data).dropna(subset=["forward_return_1"])
    folds = walk_forward_splits(features, WalkForwardConfig(train_periods=40, test_periods=10, gap_periods=1))

    assert folds
    for fold in folds:
        assert fold.train_end < fold.test_start
        assert max(fold.train_dates) < min(fold.test_dates)


def test_walk_forward_split_indices_preserve_time_order_and_exclude_test_rows_from_train() -> None:
    dates = pd.bdate_range(start="2026-01-01", periods=18)
    ordered_frame = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "feature": float(date_position),
                "forward_return_1": 0.01,
            }
            for date_position, date in enumerate(dates)
            for ticker in ("AAPL", "MSFT")
        ]
    )
    shuffled_frame = ordered_frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    shuffled_frame.index = pd.Index(
        [f"row_{row_id:03d}" for row_id in range(len(shuffled_frame))],
        name="source_row",
    )

    folds = walk_forward_splits(
        shuffled_frame,
        WalkForwardConfig(train_periods=5, test_periods=3, gap_periods=2, embargo_periods=1),
    )

    assert folds
    for fold in folds:
        assert fold.train_dates.is_monotonic_increasing
        assert fold.test_dates.is_monotonic_increasing
        assert fold.train_end < fold.test_start
        assert set(fold.train_dates).isdisjoint(set(fold.test_dates))

        train_index = shuffled_frame.index[shuffled_frame["date"].isin(fold.train_dates)]
        test_index = shuffled_frame.index[shuffled_frame["date"].isin(fold.test_dates)]
        assert set(train_index).isdisjoint(set(test_index))
        assert shuffled_frame.loc[train_index, "date"].max() < shuffled_frame.loc[test_index, "date"].min()


def test_walk_forward_split_temporal_integrity_validates_train_validation_test_windows() -> None:
    dates = pd.bdate_range(start="2026-01-01", periods=18)
    frame = pd.DataFrame(
        {
            "date": dates,
            "feature": 1.0,
            "ticker": ["AAPL"] * len(dates),
            "forward_return_1": [0.01] * len(dates),
        }
    )
    config = WalkForwardConfig(train_periods=5, test_periods=3, gap_periods=2, embargo_periods=1)

    folds = walk_forward_splits(frame, config)

    assert folds
    assert walk_forward.validate_walk_forward_temporal_integrity(folds, config) == ()

    invalid_fold = walk_forward.WalkForwardFold(
        fold=99,
        train_start=dates[0],
        train_end=dates[8],
        test_start=dates[7],
        test_end=dates[9],
        train_dates=pd.DatetimeIndex(dates[:9]),
        purge_dates=pd.DatetimeIndex(dates[5:7]),
        test_dates=pd.DatetimeIndex(dates[7:10]),
        embargo_dates=pd.DatetimeIndex(dates[10:11]),
    )

    issues = walk_forward.validate_walk_forward_temporal_integrity([invalid_fold], config)

    assert any("train window includes validation/test future dates" in issue for issue in issues)
    assert any("train dates overlap purge dates" in issue for issue in issues)
    assert any("train dates overlap validation_test dates" in issue for issue in issues)


def test_hyperparameter_tuning_scopes_use_only_fold_internal_past_dates() -> None:
    dates = pd.bdate_range(start="2026-01-01", periods=90)
    frame = pd.DataFrame(
        {
            "date": dates,
            "feature": 1.0,
            "ticker": ["AAPL"] * len(dates),
            "forward_return_20": [0.01] * len(dates),
        }
    )
    config = WalkForwardConfig(
        train_periods=12,
        test_periods=3,
        gap_periods=20,
        embargo_periods=20,
        prediction_horizon_periods=20,
    )

    folds = walk_forward_splits(frame, config)
    scopes = build_walk_forward_hyperparameter_tuning_scopes(folds, validation_periods=3)

    assert scopes
    assert walk_forward.validate_walk_forward_hyperparameter_tuning_scopes(scopes) == ()
    for fold, scope in zip(folds, scopes, strict=True):
        fold_train_dates = set(fold.train_dates)
        assert set(scope.tuning_train_dates).issubset(fold_train_dates)
        assert set(scope.tuning_validation_dates).issubset(fold_train_dates)
        assert set(scope.tuning_train_dates).isdisjoint(scope.tuning_validation_dates)
        assert set(scope.tuning_train_dates).isdisjoint(fold.test_dates)
        assert set(scope.tuning_validation_dates).isdisjoint(fold.test_dates)
        assert scope.tuning_validation_end < fold.test_start
        assert scope.final_test_start == fold.test_start
        assert scope.final_test_end == fold.test_end


def test_hyperparameter_tuning_scope_validation_rejects_final_test_leakage() -> None:
    dates = pd.bdate_range(start="2026-01-01", periods=12)
    leaking_scope = WalkForwardHyperparameterTuningScope(
        fold=0,
        tuning_train_dates=pd.DatetimeIndex(dates[:5]),
        tuning_validation_dates=pd.DatetimeIndex([dates[5], dates[8]]),
        final_test_dates=pd.DatetimeIndex(dates[8:10]),
    )

    issues = walk_forward.validate_walk_forward_hyperparameter_tuning_scopes(
        [leaking_scope]
    )

    assert any("hyperparameter tuning dates overlap final test dates" in issue for issue in issues)
    assert any("hyperparameter tuning dates include final test future dates" in issue for issue in issues)


def test_walk_forward_predict_rejects_injected_splitter_with_future_dates_in_train() -> None:
    dates = pd.bdate_range(start="2026-01-01", periods=20)
    frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAPL"] * len(dates),
            "feature_a": np.linspace(0.0, 1.0, len(dates)),
            "forward_return_1": np.linspace(-0.01, 0.01, len(dates)),
        }
    )

    class FutureLeakingSplitter:
        config = PurgeEmbargoWalkForwardConfig(
            train_periods=5,
            test_periods=3,
            purge_periods=1,
            embargo_periods=1,
            target_column="forward_return_1",
        )

        def boundaries(self, date_count: int) -> list[walk_forward.WalkForwardBoundary]:
            del date_count
            return []

        def split(
            self,
            frame: pd.DataFrame,
            candidate_dates: pd.DatetimeIndex | None = None,
        ) -> list[walk_forward.WalkForwardFold]:
            del frame, candidate_dates
            return [
                walk_forward.WalkForwardFold(
                    fold=0,
                    train_start=dates[0],
                    train_end=dates[8],
                    test_start=dates[7],
                    test_end=dates[9],
                    train_dates=pd.DatetimeIndex(dates[:9]),
                    purge_dates=pd.DatetimeIndex(dates[5:7]),
                    test_dates=pd.DatetimeIndex(dates[7:10]),
                    embargo_dates=pd.DatetimeIndex(dates[10:11]),
                )
            ]

    with pytest.raises(ValueError, match="temporal integrity failed"):
        walk_forward_predict(
            frame,
            WalkForwardConfig(
                train_periods=5,
                test_periods=3,
                gap_periods=1,
                embargo_periods=1,
                min_train_observations=5,
            ),
            splitter=FutureLeakingSplitter(),
        )


def test_walk_forward_boundaries_calculate_rolling_windows_from_configured_periods() -> None:
    config = WalkForwardConfig(train_periods=3, test_periods=2, gap_periods=1, embargo_periods=1)

    boundaries = walk_forward_boundaries(12, config)

    assert boundaries
    assert [
        (
            boundary.train_start_idx,
            boundary.train_end_idx,
            boundary.purge_start_idx,
            boundary.purge_end_idx,
            boundary.test_start_idx,
            boundary.test_end_idx,
        )
        for boundary in boundaries[:2]
    ] == [
        (0, 3, 3, 4, 4, 6),
        (7, 10, 10, 11, 11, 12),
    ]


def test_walk_forward_boundaries_promote_gap_and_embargo_to_forward_return_20_horizon() -> None:
    config = WalkForwardConfig(
        train_periods=5,
        test_periods=2,
        gap_periods=0,
        embargo_periods=0,
        prediction_horizon_periods=20,
    )

    boundaries = walk_forward_boundaries(80, config)

    assert config.gap_periods == 20
    assert config.embargo_periods == 20
    assert [
        (
            boundary.train_start_idx,
            boundary.train_end_idx,
            boundary.purge_start_idx,
            boundary.purge_end_idx,
            boundary.test_start_idx,
            boundary.test_end_idx,
        )
        for boundary in boundaries[:2]
    ] == [
        (0, 5, 5, 25, 25, 27),
        (47, 52, 52, 72, 72, 74),
    ]
    for previous, current in zip(boundaries, boundaries[1:], strict=False):
        assert current.train_start_idx - previous.test_end_idx == config.embargo_periods


def test_walk_forward_boundaries_expand_train_window_when_configured() -> None:
    config = WalkForwardConfig(
        train_periods=3,
        test_periods=2,
        window_mode="expanding",
        gap_periods=1,
        embargo_periods=1,
    )

    boundaries = walk_forward_boundaries(12, config)

    assert len(boundaries) == 2
    assert [
        (
            boundary.train_start_idx,
            boundary.train_end_idx,
            boundary.purge_start_idx,
            boundary.purge_end_idx,
            boundary.test_start_idx,
            boundary.test_end_idx,
        )
        for boundary in boundaries
    ] == [
        (0, 3, 3, 4, 4, 6),
        (0, 10, 10, 11, 11, 12),
    ]


def test_walk_forward_splits_support_expanding_train_window() -> None:
    dates = pd.date_range(start="2026-01-01", periods=12, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "feature": 1.0,
            "ticker": ["AAPL"] * len(dates),
            "forward_return_1": [0.01] * len(dates),
        }
    )

    folds = walk_forward_splits(
        frame,
        WalkForwardConfig(
            train_periods=3,
            test_periods=2,
            window_mode="expanding",
            gap_periods=1,
            embargo_periods=1,
        ),
    )

    assert len(folds) == 2
    assert folds[0].train_start == dates[0]
    assert folds[1].train_start == dates[0]
    assert folds[1].train_end > folds[0].train_end
    assert folds[1].train_end < folds[1].test_start


def test_walk_forward_splits_exclude_post_test_embargo_from_expanding_train_window() -> None:
    dates = pd.date_range(start="2026-01-01", periods=16, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "feature": 1.0,
            "ticker": ["AAPL"] * len(dates),
            "forward_return_1": [0.01] * len(dates),
        }
    )

    folds = walk_forward_splits(
        frame,
        WalkForwardConfig(
            train_periods=3,
            test_periods=2,
            window_mode="expanding",
            gap_periods=1,
            embargo_periods=2,
        ),
    )

    assert len(folds) >= 2
    first_fold_test_end_position = dates.get_loc(folds[0].test_end)
    post_test_embargo_dates = dates[
        first_fold_test_end_position + 1 : first_fold_test_end_position + 1 + 2
    ]
    assert set(post_test_embargo_dates).isdisjoint(set(folds[1].train_dates))
    assert set(folds[0].test_dates).issubset(set(folds[1].train_dates))


def test_walk_forward_splits_apply_purge_gap_between_train_and_test() -> None:
    dates = pd.bdate_range(start="2026-01-01", periods=18)
    frame = pd.DataFrame(
        {
            "date": dates,
            "feature": 1.0,
            "ticker": ["AAPL"] * len(dates),
            "forward_return_1": [0.01] * len(dates),
        }
    )

    folds = walk_forward_splits(
        frame,
        WalkForwardConfig(train_periods=5, test_periods=3, gap_periods=2, embargo_periods=1),
    )

    assert folds
    first_fold = folds[0]
    assert list(first_fold.train_dates) == list(dates[:5])
    assert list(first_fold.purge_dates) == list(dates[5:7])
    assert list(first_fold.test_dates) == list(dates[7:10])
    assert set(first_fold.purge_dates).isdisjoint(set(first_fold.train_dates))
    assert set(first_fold.purge_dates).isdisjoint(set(first_fold.test_dates))
    assert len(first_fold.purge_dates) == 2


def test_forward_return_20_splits_keep_train_purge_test_and_embargo_boundaries_disjoint() -> None:
    dates = pd.bdate_range(start="2026-01-01", periods=80)
    frame = pd.DataFrame(
        {
            "date": dates,
            "feature": 1.0,
            "ticker": ["AAPL"] * len(dates),
            "forward_return_20": [0.01] * len(dates),
        }
    )
    config = WalkForwardConfig(
        train_periods=5,
        test_periods=2,
        window_mode="expanding",
        gap_periods=0,
        embargo_periods=0,
        prediction_horizon_periods=20,
    )

    folds = walk_forward_splits(frame, config)

    assert len(folds) >= 2
    first_fold = folds[0]
    assert list(first_fold.train_dates) == list(dates[:5])
    assert list(first_fold.purge_dates) == list(dates[5:25])
    assert list(first_fold.test_dates) == list(dates[25:27])
    for fold in folds:
        train_dates = set(fold.train_dates)
        purge_dates = set(fold.purge_dates)
        test_dates = set(fold.test_dates)
        assert train_dates.isdisjoint(purge_dates)
        assert train_dates.isdisjoint(test_dates)
        assert purge_dates.isdisjoint(test_dates)
        assert len(fold.purge_dates) == 20
        assert fold.train_end < fold.test_start
        assert dates.get_loc(fold.test_start) - dates.get_loc(fold.train_end) - 1 >= 20

    first_test_end_idx = dates.get_loc(first_fold.test_end)
    first_post_test_embargo_dates = set(dates[first_test_end_idx + 1 : first_test_end_idx + 1 + 20])
    assert first_post_test_embargo_dates.isdisjoint(set(folds[1].train_dates))


def test_forward_return_20_boundary_dates_are_exactly_purged_and_embargoed() -> None:
    dates = pd.bdate_range(start="2026-01-01", periods=90)
    frame = pd.DataFrame(
        {
            "date": dates,
            "feature": 1.0,
            "ticker": ["AAPL"] * len(dates),
            "forward_return_20": [0.01] * len(dates),
        }
    )
    config = WalkForwardConfig(
        train_periods=5,
        test_periods=3,
        gap_periods=20,
        embargo_periods=20,
        prediction_horizon_periods=20,
    )

    folds = walk_forward_splits(frame, config)

    assert len(folds) >= 2
    first, second = folds[:2]
    assert first.train_dates[-1] == dates[4]
    assert first.purge_dates[0] == dates[5]
    assert first.purge_dates[-1] == dates[24]
    assert first.test_dates[0] == dates[25]
    assert first.test_dates[-1] == dates[27]
    assert first.embargo_dates[0] == dates[28]
    assert first.embargo_dates[-1] == dates[47]
    assert second.train_dates[0] == dates[48]
    assert set(first.purge_dates).isdisjoint(second.train_dates)
    assert set(first.embargo_dates).isdisjoint(second.train_dates)


def test_walk_forward_predict_does_not_pass_forward_return_labels_as_features(monkeypatch) -> None:
    feature_columns_seen: list[list[str]] = []

    class RecordingModel:
        actual_model_name = "recording"

        def __init__(self, **kwargs) -> None:
            feature_columns_seen.append(list(kwargs["feature_columns"]))
            self.training_metadata = {
                "tabular_fallback_reason": None,
                "calibration_scale": 1.0,
                "calibration_bias": 0.0,
                "winsorized_feature_count": 0,
            }

        def fit(self, train: pd.DataFrame, target: str = "forward_return_1") -> RecordingModel:
            assert target == "forward_return_20"
            assert "forward_return_20" in train.columns
            return self

        def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
            return frame[["date", "ticker"]].assign(
                raw_expected_return=0.0,
                expected_return=0.0,
                predicted_volatility=0.01,
                downside_quantile=-0.01,
                upside_quantile=0.01,
                quantile_width=0.02,
                model_confidence=0.0,
                model_name=self.actual_model_name,
                model_calibration_scale=1.0,
                model_calibration_bias=0.0,
            )

    monkeypatch.setattr(walk_forward, "TabularReturnModel", RecordingModel)

    dates = pd.bdate_range(start="2026-01-01", periods=70)
    frame = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "feature_a": float(date_index),
                "forward_return_1": 0.001,
                "forward_return_5": 0.005,
                "forward_return_20": 0.020,
            }
            for date_index, date in enumerate(dates)
            for ticker in ("AAPL", "MSFT")
        ]
    )

    predictions, summary = walk_forward_predict(
        frame,
        WalkForwardConfig(
            train_periods=10,
            test_periods=3,
            gap_periods=20,
            embargo_periods=20,
            min_train_observations=10,
        ),
        target="forward_return_20",
    )

    assert not predictions.empty
    assert not summary.empty
    assert feature_columns_seen
    for feature_columns in feature_columns_seen:
        assert feature_columns == ["feature_a"]
        assert not any(column.startswith("forward_return_") for column in feature_columns)


def test_walk_forward_config_rejects_unknown_window_mode() -> None:
    with pytest.raises(ValueError, match="window_mode"):
        WalkForwardConfig(window_mode="anchored")


def test_walk_forward_marks_final_fold_as_oos() -> None:
    data = SyntheticMarketDataProvider(periods=220).get_history(["AAPL", "MSFT"])
    features = build_price_features(data).dropna(subset=["forward_return_1"])

    predictions, summary = walk_forward_predict(
        features,
        WalkForwardConfig(train_periods=50, test_periods=15, gap_periods=1, min_train_observations=50),
    )

    assert not predictions.empty
    assert not summary.empty
    assert "prediction_count" in summary.columns
    assert "information_coefficient" in summary.columns
    assert summary["is_oos"].sum() == 2
    assert predictions["is_oos"].any()


def test_walk_forward_splits_respects_custom_dates_and_embargo() -> None:
    dates = pd.date_range(start="2026-01-01", periods=15, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "feature": 1.0,
            "ticker": ["AAPL"] * len(dates),
            "forward_return_1": [0.01] * len(dates),
        }
    )
    candidate_dates = pd.DatetimeIndex(dates[[0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13]])
    config = WalkForwardConfig(train_periods=3, test_periods=2, gap_periods=1, embargo_periods=1)
    folds = walk_forward_splits(frame, config, candidate_dates=candidate_dates)

    assert len(folds) >= 2
    for fold in folds:
        assert set(fold.train_dates).issubset(set(candidate_dates))
        assert set(fold.test_dates).issubset(set(candidate_dates))
        assert max(fold.train_dates) < min(fold.test_dates)
    date_position = {date: position for position, date in enumerate(candidate_dates)}
    for index, fold in enumerate(folds[:-1]):
        assert date_position[folds[index + 1].train_start] == date_position[fold.test_end] + config.embargo_periods + 1


def test_walk_forward_predict_skips_targetless_dates() -> None:
    dates = pd.bdate_range(start="2026-01-01", periods=9)
    target = [0.01, 0.02, 0.005, 0.01, 0.0, -0.01, None, 0.02, 0.01]
    frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAPL"] * len(dates),
            "feature_a": list(range(len(dates))),
            "forward_return_1": target,
        }
    )
    predictions, summary = walk_forward_predict(
        frame,
        WalkForwardConfig(train_periods=3, test_periods=2, gap_periods=1, min_train_observations=3),
    )

    assert not predictions.empty
    assert not summary.empty
    assert pd.Timestamp(dates[6]) not in set(predictions["date"].dt.normalize())
    assert summary["labeled_test_observations"].sum() > 0


def test_walk_forward_predict_returns_structured_insufficient_data_for_undersized_dataset() -> None:
    dates = pd.bdate_range(start="2026-01-01", periods=8)
    frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAPL"] * len(dates),
            "feature_a": list(range(len(dates))),
            "forward_return_5": [0.01] * len(dates),
        }
    )

    predictions, summary = walk_forward_predict(
        frame,
        WalkForwardConfig(train_periods=20, test_periods=5, gap_periods=5),
        target="forward_return_5",
    )

    assert not predictions.empty
    assert {"date", "ticker", "expected_return", "forward_return_5", "fold", "is_oos"}.issubset(
        predictions.columns
    )
    assert predictions["model_name"].eq("deterministic_neutral_fallback").all()
    assert predictions["expected_return"].eq(0.0).all()
    assert predictions["deterministic_fallback_code"].eq("insufficient_labeled_dates").all()
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["validation_status"] == "insufficient_data"
    assert row["skip_status"] == "skipped"
    assert row["skip_code"] == "insufficient_labeled_dates"
    assert row["fold_count"] == 0
    assert row["prediction_count"] == len(predictions)
    assert bool(row["deterministic_fallback_applied"]) is True
    assert row["deterministic_fallback_policy"] == "neutral_hold"
    assert row["labeled_date_count"] == len(dates)
    assert row["required_min_date_count"] > len(dates)
    assert row["train_periods"] == 20
    assert row["validation_periods"] == 5
    assert row["test_periods"] == 5
    assert row["purge_periods"] == 5
    assert row["embargo_periods"] >= 5
    assert pd.isna(row["validation_start"])
    assert pd.isna(row["oos_test_start"])


def test_walk_forward_predict_enforces_target_horizon_gap_and_embargo() -> None:
    dates = pd.bdate_range(start="2026-01-01", periods=48)
    frame = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "feature_a": float(date_index),
                "feature_b": 1.0 if ticker == "AAPL" else -1.0,
                "forward_return_5": 0.01 * ((date_index % 5) - 2),
            }
            for date_index, date in enumerate(dates)
            for ticker in ("AAPL", "MSFT")
        ]
    )

    predictions, summary = walk_forward_predict(
        frame,
        WalkForwardConfig(
            train_periods=8,
            test_periods=4,
            gap_periods=1,
            embargo_periods=1,
            min_train_observations=8,
        ),
        target="forward_return_5",
    )

    assert not predictions.empty
    assert not summary.empty
    date_position = {date: position for position, date in enumerate(dates)}
    fold_rows = summary.sort_values("fold").reset_index(drop=True)
    for _, row in fold_rows.iterrows():
        train_end_position = date_position[pd.Timestamp(row["train_end"])]
        test_start_position = date_position[pd.Timestamp(row["test_start"])]
        assert test_start_position - train_end_position - 1 >= 5
        assert row["target_column"] == "forward_return_5"
        assert row["prediction_horizon_periods"] >= 5
        assert row["gap_periods"] >= 5
        assert row["purge_gap_periods"] >= 5
        assert row["purged_date_count"] >= 5
        assert bool(row["purge_applied"]) is True
        assert row["embargo_periods"] >= 5
        assert bool(row["embargo_applied"]) is True
        assert pd.Timestamp(row["train_end"]) < pd.Timestamp(row["test_start"])

    for index in range(len(fold_rows) - 1):
        test_end_position = date_position[pd.Timestamp(fold_rows.loc[index, "test_end"])]
        next_train_start_position = date_position[
            pd.Timestamp(fold_rows.loc[index + 1, "train_start"])
        ]
        assert next_train_start_position - test_end_position - 1 >= 5


def test_walk_forward_predict_never_fits_on_validation_or_oos_future_dates(monkeypatch) -> None:
    fit_windows: list[pd.DatetimeIndex] = []
    predict_windows: list[pd.DatetimeIndex] = []

    class RecordingModel:
        actual_model_name = "recording"

        def __init__(self, **kwargs) -> None:
            del kwargs
            self.training_metadata = {
                "tabular_fallback_reason": None,
                "calibration_scale": 1.0,
                "calibration_bias": 0.0,
                "winsorized_feature_count": 0,
            }

        def fit(self, train: pd.DataFrame, target: str = "forward_return_1") -> RecordingModel:
            assert target == "forward_return_20"
            fit_windows.append(pd.DatetimeIndex(pd.to_datetime(train["date"]).dt.normalize().unique()))
            return self

        def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
            predict_windows.append(pd.DatetimeIndex(pd.to_datetime(frame["date"]).dt.normalize().unique()))
            return frame[["date", "ticker"]].assign(
                raw_expected_return=0.0,
                expected_return=0.0,
                predicted_volatility=0.01,
                downside_quantile=-0.01,
                upside_quantile=0.01,
                quantile_width=0.02,
                model_confidence=0.0,
                model_name=self.actual_model_name,
                model_calibration_scale=1.0,
                model_calibration_bias=0.0,
            )

    monkeypatch.setattr(walk_forward, "TabularReturnModel", RecordingModel)

    dates = pd.bdate_range(start="2026-01-01", periods=90)
    frame = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "feature_a": float(date_index),
                "feature_b": 1.0 if ticker == "AAPL" else -1.0,
                "forward_return_20": 0.001 * (date_index % 7),
            }
            for date_index, date in enumerate(dates)
            for ticker in ("AAPL", "MSFT")
        ]
    )

    predictions, summary = walk_forward_predict(
        frame,
        WalkForwardConfig(
            train_periods=10,
            test_periods=3,
            gap_periods=0,
            embargo_periods=0,
            min_train_observations=10,
        ),
        target="forward_return_20",
    )

    assert not predictions.empty
    assert not summary.empty
    assert len(fit_windows) == len(predict_windows) == len(summary)
    assert summary["is_oos"].sum() == 2
    assert summary["hyperparameter_tuning_status"].eq("pass").all()
    assert summary["hyperparameter_tuning_scope"].eq("fold_train_only").all()
    assert summary["hyperparameter_tuning_uses_final_test"].eq(False).all()

    for train_dates, test_dates in zip(fit_windows, predict_windows, strict=True):
        assert train_dates.max() < test_dates.min()
        assert set(train_dates).isdisjoint(set(test_dates))

    oos_test_dates = predict_windows[-1]
    for train_dates in fit_windows:
        assert set(train_dates).isdisjoint(set(oos_test_dates))

    for _, row in summary.iterrows():
        assert pd.Timestamp(row["hyperparameter_tuning_latest_allowed_date"]) < pd.Timestamp(
            row["test_start"]
        )
        assert pd.Timestamp(row["hyperparameter_tuning_final_test_start"]) == pd.Timestamp(
            row["test_start"]
        )


def test_walk_forward_preprocessing_is_fit_on_fold_train_only(monkeypatch) -> None:
    fit_matrices: list[np.ndarray] = []
    predict_matrices: list[np.ndarray] = []

    class RecordingRegressor(BaseEstimator, RegressorMixin):
        def fit(self, X, y, sample_weight=None) -> RecordingRegressor:
            del y, sample_weight
            fit_matrices.append(np.asarray(X, dtype=float).copy())
            return self

        def predict(self, X) -> np.ndarray:
            predict_matrices.append(np.asarray(X, dtype=float).copy())
            return np.zeros(len(X), dtype=float)

    def fake_make_estimator(model_name: str, random_state: int, num_threads: int = 1):
        del model_name, random_state, num_threads
        return RecordingRegressor()

    monkeypatch.setattr(tabular, "_make_estimator", fake_make_estimator)

    dates = pd.bdate_range(start="2026-01-01", periods=95)
    frame = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "feature_a": 10000.0 if date_index >= 30 else float(date_index),
                "feature_missing": np.nan if date_index >= 30 else 5.0,
                "forward_return_20": 0.001 * ((date_index % 9) - ticker_index),
            }
            for date_index, date in enumerate(dates)
            for ticker_index, ticker in enumerate(("AAPL", "MSFT"))
        ]
    )

    predictions, summary = walk_forward_predict(
        frame,
        WalkForwardConfig(
            train_periods=16,
            test_periods=3,
            gap_periods=20,
            embargo_periods=20,
            min_train_observations=16,
            model_name="recording",
            native_tabular_isolation=False,
        ),
        target="forward_return_20",
    )

    assert not predictions.empty
    assert not summary.empty
    test_predict_matrices = [
        matrix
        for matrix in predict_matrices
        if len(matrix) == int(summary["test_observations"].iloc[0])
    ]
    assert len(fit_matrices) == len(test_predict_matrices) == len(summary)
    assert summary["preprocessing_fit_scope"].eq("fold_train_only").all()
    assert summary["preprocessing_transform_scope"].eq("validation_test_only").all()
    assert summary["preprocessing_pipeline_steps"].map(lambda steps: steps == ["winsorize", "imputer"]).all()

    for train_matrix, test_matrix in zip(fit_matrices, test_predict_matrices, strict=True):
        assert train_matrix[:, 0].max() < 10000.0
        assert test_matrix[:, 0].max() < 10000.0
        assert np.isclose(test_matrix[:, 1], 5.0).all()


def test_walk_forward_predict_marks_all_skipped_folds_as_insufficient_data() -> None:
    dates = pd.bdate_range(start="2026-01-01", periods=20)
    frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAPL"] * len(dates),
            "feature_a": list(range(len(dates))),
            "forward_return_1": [0.01] * len(dates),
        }
    )

    predictions, summary = walk_forward_predict(
        frame,
        WalkForwardConfig(
            train_periods=5,
            test_periods=2,
            gap_periods=1,
            min_train_observations=100,
        ),
    )

    assert not predictions.empty
    assert predictions["model_name"].eq("deterministic_neutral_fallback").all()
    assert predictions["deterministic_fallback_code"].eq("insufficient_fold_observations").all()
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["validation_status"] == "insufficient_data"
    assert row["skip_code"] == "insufficient_fold_observations"
    assert row["candidate_fold_count"] > 0
    assert row["max_train_observations"] < row["min_train_observations"]
    assert row["prediction_count"] == len(predictions)


def test_walk_forward_predict_uses_deterministic_fallback_for_all_missing_features() -> None:
    dates = pd.bdate_range(start="2026-01-01", periods=18)
    frame = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "feature_a": np.nan,
                "forward_return_1": 0.01 if ticker == "AAPL" else -0.01,
            }
            for date in dates
            for ticker in ("AAPL", "MSFT")
        ]
    )

    predictions, summary = walk_forward_predict(
        frame,
        WalkForwardConfig(
            train_periods=5,
            test_periods=2,
            gap_periods=1,
            min_train_observations=5,
        ),
    )

    assert not predictions.empty
    assert predictions["expected_return"].eq(0.0).all()
    assert predictions["predicted_volatility"].eq(0.02).all()
    assert predictions["deterministic_fallback_code"].eq("all_missing_train_features").all()
    assert summary["validation_status"].eq("insufficient_data").all()
    assert summary["deterministic_fallback_applied"].eq(True).all()


def test_walk_forward_predict_uses_deterministic_fallback_for_single_asset() -> None:
    dates = pd.bdate_range(start="2026-01-01", periods=18)
    frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAPL"] * len(dates),
            "feature_a": np.linspace(0.0, 1.0, len(dates)),
            "forward_return_1": np.linspace(-0.01, 0.01, len(dates)),
        }
    )

    predictions, summary = walk_forward_predict(
        frame,
        WalkForwardConfig(
            train_periods=5,
            test_periods=2,
            gap_periods=1,
            min_train_observations=5,
        ),
    )

    assert not predictions.empty
    assert predictions["model_name"].eq("deterministic_neutral_fallback").all()
    assert predictions["deterministic_fallback_code"].eq("single_asset_universe").all()
    assert summary["skip_code"].eq("single_asset_universe").all()


def test_walk_forward_fold_metrics_and_oos_summary_calculate_rank_ic() -> None:
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-03",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-04",
                    "2026-01-04",
                ]
            ),
            "ticker": ["AAPL", "MSFT", "NVDA"] * 4,
            "expected_return": [
                0.03,
                0.02,
                0.01,
                0.03,
                0.02,
                0.01,
                0.03,
                0.02,
                0.01,
                0.03,
                0.02,
                0.01,
            ],
            "forward_return_20": [
                0.04,
                0.01,
                -0.01,
                -0.02,
                0.01,
                0.04,
                0.05,
                0.02,
                -0.01,
                0.06,
                0.03,
                0.00,
            ],
            "fold": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            "is_oos": [False] * 6 + [True] * 6,
        }
    )

    fold_metrics = walk_forward.calculate_walk_forward_fold_metrics(
        predictions,
        target="forward_return_20",
    )
    oos_summary = walk_forward.calculate_walk_forward_oos_summary(fold_metrics)

    by_fold = {int(row["fold"]): row for row in fold_metrics.to_dict("records")}
    assert by_fold[0]["fold_rank_ic"] == pytest.approx(0.0)
    assert by_fold[0]["fold_rank_ic_count"] == 2
    assert by_fold[0]["fold_positive_rank_ic_day_ratio"] == pytest.approx(0.5)
    assert by_fold[1]["fold_rank_ic"] == pytest.approx(1.0)
    assert by_fold[1]["fold_rank_ic_count"] == 2
    assert oos_summary["oos_fold_count"] == 1
    assert oos_summary["oos_rank_ic"] == pytest.approx(1.0)
    assert oos_summary["oos_rank_ic_positive_fold_ratio"] == pytest.approx(1.0)
    assert oos_summary["oos_rank_ic_count"] == 2


def test_walk_forward_predict_appends_fold_metrics_and_oos_summary(monkeypatch) -> None:
    class RankingModel:
        actual_model_name = "ranking"

        def __init__(self, **kwargs) -> None:
            del kwargs
            self.training_metadata = {
                "tabular_fallback_reason": None,
                "calibration_scale": 1.0,
                "calibration_bias": 0.0,
                "winsorized_feature_count": 0,
            }

        def fit(self, train: pd.DataFrame, target: str = "forward_return_20") -> RankingModel:
            del train, target
            return self

        def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
            ticker_rank = frame["ticker"].map({"AAPL": 3.0, "MSFT": 2.0, "NVDA": 1.0})
            return frame[["date", "ticker"]].assign(
                raw_expected_return=ticker_rank,
                expected_return=ticker_rank,
                predicted_volatility=0.01,
                downside_quantile=-0.01,
                upside_quantile=0.01,
                quantile_width=0.02,
                model_confidence=0.8,
                model_name=self.actual_model_name,
                model_calibration_scale=1.0,
                model_calibration_bias=0.0,
            )

    monkeypatch.setattr(walk_forward, "TabularReturnModel", RankingModel)
    dates = pd.bdate_range(start="2026-01-01", periods=90)
    frame = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "feature_a": float(date_index),
                "feature_b": float(ticker_index),
                "forward_return_20": float(3 - ticker_index) / 100.0,
            }
            for date_index, date in enumerate(dates)
            for ticker_index, ticker in enumerate(("AAPL", "MSFT", "NVDA"))
        ]
    )

    _, summary = walk_forward_predict(
        frame,
        WalkForwardConfig(
            train_periods=8,
            test_periods=3,
            gap_periods=2,
            embargo_periods=2,
            min_train_observations=8,
        ),
        target="forward_return_20",
    )

    assert not summary.empty
    assert {
        "fold_rank_ic",
        "fold_rank_ic_count",
        "walk_forward_mean_rank_ic",
        "walk_forward_positive_rank_ic_fold_ratio",
        "oos_fold_count",
        "oos_rank_ic",
        "oos_rank_ic_count",
        "oos_prediction_count",
    }.issubset(summary.columns)
    assert summary["fold_rank_ic"].dropna().eq(1.0).all()
    assert summary["walk_forward_mean_rank_ic"].iloc[0] == pytest.approx(1.0)
    assert summary["oos_rank_ic"].iloc[0] == pytest.approx(1.0)
    assert summary["oos_fold_count"].iloc[0] == 2


def test_walk_forward_summary_includes_lightgbm_fallback_reason(monkeypatch) -> None:
    FakeLGBM = type(
        "LGBMRegressor",
        (),
        {
            "fit": lambda self, X, y, sample_weight=None: self,
            "predict": lambda self, X: np.zeros(len(X)),
        },
    )

    def fake_make_estimator(model_name: str, random_state: int, num_threads: int = 1):
        del model_name, random_state, num_threads
        return FakeLGBM()

    def fake_lightgbm_subprocess(**kwargs):
        del kwargs
        return {"success": False, "reason": "timed out"}

    monkeypatch.setattr(tabular, "_make_estimator", fake_make_estimator)
    monkeypatch.setattr(tabular, "_run_lightgbm_subprocess", fake_lightgbm_subprocess)

    dates = pd.date_range("2026-01-01", periods=80, freq="D")
    frame = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "volatility_20": 0.02 + (i * 0.0001),
                "return_5": 0.001 * (i % 4 - 2),
                "forward_return_1": 0.001 * (i % 7 - 3),
            }
            for i, date in enumerate(dates)
            for ticker in ("AAPL", "MSFT")
        ]
    )

    _, summary = walk_forward_predict(
        frame,
        WalkForwardConfig(
            train_periods=35,
            test_periods=5,
            gap_periods=1,
            min_train_observations=35,
            model_name="lightgbm",
            native_tabular_isolation=True,
        ),
    )

    assert not summary.empty
    assert "tabular_fallback_reason" in summary.columns
    assert summary["model_name"].eq("HistGradientBoostingRegressor").all()
    assert summary["tabular_fallback_reason"].eq("timed out").all()
