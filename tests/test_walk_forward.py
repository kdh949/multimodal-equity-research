from __future__ import annotations

import numpy as np
import pandas as pd

import quant_research.models.tabular as tabular
from quant_research.data.market import SyntheticMarketDataProvider
from quant_research.features.price import build_price_features
from quant_research.models.tabular import infer_feature_columns
from quant_research.validation.walk_forward import (
    WalkForwardConfig,
    walk_forward_predict,
    walk_forward_splits,
)


def test_walk_forward_splits_never_train_on_future_dates() -> None:
    data = SyntheticMarketDataProvider(periods=120).get_history(["AAPL", "MSFT"])
    features = build_price_features(data).dropna(subset=["forward_return_1"])
    folds = walk_forward_splits(features, WalkForwardConfig(train_periods=40, test_periods=10, gap_periods=1))

    assert folds
    for fold in folds:
        assert fold.train_end < fold.test_start
        assert max(fold.train_dates) < min(fold.test_dates)


def test_walk_forward_marks_final_fold_as_oos() -> None:
    data = SyntheticMarketDataProvider(periods=140).get_history(["AAPL", "MSFT"])
    features = build_price_features(data).dropna(subset=["forward_return_1"])

    predictions, summary = walk_forward_predict(
        features,
        WalkForwardConfig(train_periods=50, test_periods=15, gap_periods=1, min_train_observations=50),
    )

    assert not predictions.empty
    assert not summary.empty
    assert "prediction_count" in summary.columns
    assert "information_coefficient" in summary.columns
    assert summary["is_oos"].sum() == 1
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
    effective_embargo = max(config.embargo_periods, config.target_horizon)
    for index, fold in enumerate(folds[:-1]):
        assert date_position[folds[index + 1].train_start] == date_position[fold.test_end] + effective_embargo + 1


def test_walk_forward_predict_records_effective_horizon_guards() -> None:
    dates = pd.bdate_range(start="2026-01-01", periods=120)
    rows = []
    for index, date_value in enumerate(dates):
        for ticker_offset, ticker in enumerate(["AAPL", "MSFT"]):
            rows.append(
                {
                    "date": date_value,
                    "ticker": ticker,
                    "feature_a": float(index + ticker_offset),
                    "feature_b": float((index % 7) - ticker_offset),
                    "forward_return_5": None if index >= len(dates) - 5 else 0.001 * ((index + ticker_offset) % 9 - 4),
                }
            )
    frame = pd.DataFrame(rows)

    predictions, summary = walk_forward_predict(
        frame,
        WalkForwardConfig(
            train_periods=35,
            test_periods=8,
            gap_periods=1,
            embargo_periods=0,
            min_train_observations=35,
            model_name="hist_gradient",
            native_tabular_isolation=False,
        ),
        target="forward_return_5",
    )

    assert not predictions.empty
    assert not summary.empty
    assert summary["target_column"].eq("forward_return_5").all()
    assert summary["target_horizon"].eq(5).all()
    assert summary["requested_gap_periods"].eq(1).all()
    assert summary["requested_embargo_periods"].eq(0).all()
    assert summary["effective_gap_periods"].eq(5).all()
    assert summary["effective_embargo_periods"].eq(5).all()
    assert summary["label_overlap_violations"].eq(0).all()


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
        {
            "date": dates,
            "ticker": ["AAPL"] * len(dates),
            "volatility_20": [0.02 + (i * 0.0001) for i in range(len(dates))],
            "return_5": [0.001 * (i % 4 - 2) for i in range(len(dates))],
            "forward_return_1": [0.001 * (i % 7 - 3) for i in range(len(dates))],
        }
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


def test_infer_feature_columns_excludes_all_forward_return_targets() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3),
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "return_1": [0.01, 0.02, 0.03],
            "forward_return_1": [0.01, 0.01, 0.01],
            "forward_return_5": [0.02, 0.02, 0.02],
            "forward_return_20": [0.03, 0.03, 0.03],
        }
    )

    assert infer_feature_columns(frame, target="forward_return_5") == ["return_1"]
