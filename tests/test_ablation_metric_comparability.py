from __future__ import annotations

import math
from typing import Any

import pandas as pd
import pytest

import quant_research.pipeline as pipeline
from quant_research.pipeline import PipelineConfig
from quant_research.validation import DEFAULT_VALIDITY_GATE_ABLATION_MODE_IDS


_REQUIRED_ROW_METRIC_FIELDS = {
    "cagr",
    "sharpe",
    "max_drawdown",
    "turnover",
    "excess_return",
    "effective_cost_bps",
    "effective_slippage_bps",
    "validation_status",
    "validation_fold_count",
    "validation_oos_fold_count",
    "validation_prediction_count",
    "validation_labeled_prediction_count",
    "validation_mean_mae",
    "validation_mean_directional_accuracy",
    "validation_mean_information_coefficient",
    "validation_positive_ic_fold_ratio",
    "validation_oos_information_coefficient",
    "signal_evaluation_metrics",
    "deterministic_signal_evaluation_metrics",
    "signal_engine",
    "signal_return_basis",
    "signal_realized_return_column",
    "signal_observation_count",
    "signal_evaluation_observation_count",
    "signal_buy_count",
    "signal_sell_count",
    "signal_hold_count",
    "signal_buy_ratio",
    "signal_sell_ratio",
    "signal_hold_ratio",
    "signal_hit_rate",
    "signal_exposure",
    "signal_average_daily_turnover",
    "signal_max_daily_turnover",
    "signal_gross_cumulative_return",
    "signal_cost_adjusted_cumulative_return",
    "signal_transaction_cost_return",
    "signal_slippage_cost_return",
    "signal_total_cost_return",
    "signal_risk_stop_observation_count",
    "signal_risk_stop_observation_ratio",
    "signal_max_portfolio_volatility_estimate",
}

_REQUIRED_ROW_NUMERIC_FIELDS = {
    "cagr",
    "sharpe",
    "max_drawdown",
    "turnover",
    "excess_return",
    "effective_cost_bps",
    "effective_slippage_bps",
    "validation_fold_count",
    "validation_oos_fold_count",
    "validation_prediction_count",
    "validation_labeled_prediction_count",
    "signal_observation_count",
    "signal_evaluation_observation_count",
    "signal_buy_count",
    "signal_sell_count",
    "signal_hold_count",
    "signal_buy_ratio",
    "signal_sell_ratio",
    "signal_hold_ratio",
    "signal_hit_rate",
    "signal_exposure",
    "signal_average_daily_turnover",
    "signal_max_daily_turnover",
    "signal_gross_cumulative_return",
    "signal_cost_adjusted_cumulative_return",
    "signal_transaction_cost_return",
    "signal_slippage_cost_return",
    "signal_total_cost_return",
    "signal_risk_stop_observation_count",
    "signal_risk_stop_observation_ratio",
    "signal_max_portfolio_volatility_estimate",
}

_OPTIONAL_VALIDATION_NUMERIC_FIELDS = {
    "validation_mean_mae",
    "validation_mean_directional_accuracy",
    "validation_mean_information_coefficient",
    "validation_positive_ic_fold_ratio",
    "validation_oos_information_coefficient",
}

_REQUIRED_SIGNAL_METRIC_FIELDS = {
    "engine",
    "return_basis",
    "realized_return_column",
    "effective_cost_bps",
    "effective_slippage_bps",
    "signal_observations",
    "evaluation_observations",
    "evaluation_start",
    "evaluation_end",
    "action_counts",
    "action_ratios",
    "cagr",
    "annualized_volatility",
    "sharpe",
    "max_drawdown",
    "hit_rate",
    "average_daily_turnover",
    "max_daily_turnover",
    "exposure",
    "average_exposure",
    "benchmark_cagr",
    "excess_return",
    "gross_cumulative_return",
    "cost_adjusted_cumulative_return",
    "transaction_cost_return",
    "slippage_cost_return",
    "total_cost_return",
    "risk_stop_observation_count",
    "risk_stop_observation_ratio",
    "max_portfolio_volatility_estimate",
}

_REQUIRED_SIGNAL_NUMERIC_FIELDS = {
    "effective_cost_bps",
    "effective_slippage_bps",
    "signal_observations",
    "evaluation_observations",
    "cagr",
    "annualized_volatility",
    "sharpe",
    "max_drawdown",
    "hit_rate",
    "average_daily_turnover",
    "max_daily_turnover",
    "exposure",
    "average_exposure",
    "benchmark_cagr",
    "excess_return",
    "gross_cumulative_return",
    "cost_adjusted_cumulative_return",
    "transaction_cost_return",
    "slippage_cost_return",
    "total_cost_return",
    "risk_stop_observation_count",
    "risk_stop_observation_ratio",
    "max_portfolio_volatility_estimate",
}


def test_stage1_ablation_metrics_are_comparable_across_scenarios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = _stage1_ablation_feature_frame()
    predictions = _stage1_prediction_frame(features)

    def fake_walk_forward_predict(
        variant: pd.DataFrame,
        _walk_config: object,
        *,
        target: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        row_count = len(variant)
        return (
            variant[["date", "ticker", target]].assign(
                raw_expected_return=[0.04, -0.02, 0.03, -0.01][:row_count],
                expected_return=[0.04, -0.02, 0.03, -0.01][:row_count],
                predicted_volatility=0.01,
                downside_quantile=-0.01,
                upside_quantile=0.06,
                quantile_width=0.07,
                model_confidence=0.8,
                model_name="metric_comparability_fixture",
                fold=0,
                is_oos=True,
            ),
            pd.DataFrame(
                {
                    "fold": [0],
                    "is_oos": [True],
                    "mae": [0.01],
                    "directional_accuracy": [1.0],
                    "information_coefficient": [0.5],
                    "labeled_test_observations": [row_count],
                    "prediction_count": [row_count],
                }
            ),
        )

    monkeypatch.setattr(pipeline, "walk_forward_predict", fake_walk_forward_predict)

    summary = pipeline._run_ablation_summary(
        predictions,
        features,
        PipelineConfig(
            cost_bps=8.0,
            slippage_bps=3.0,
            native_tabular_isolation=False,
            validity_gate_ablation_modes=DEFAULT_VALIDITY_GATE_ABLATION_MODE_IDS,
        ),
    )

    rows = {str(row["scenario"]): row for row in summary}
    assert set(rows) == set(DEFAULT_VALIDITY_GATE_ABLATION_MODE_IDS)

    metric_field_sets = {
        scenario_id: _REQUIRED_ROW_METRIC_FIELDS.intersection(row)
        for scenario_id, row in rows.items()
    }
    assert all(
        fields == _REQUIRED_ROW_METRIC_FIELDS
        for fields in metric_field_sets.values()
    )

    for scenario_id, row in rows.items():
        assert isinstance(row["validation_status"], str)
        for field in _REQUIRED_ROW_NUMERIC_FIELDS:
            _assert_json_number(row[field], scenario_id, field)
        for field in _OPTIONAL_VALIDATION_NUMERIC_FIELDS:
            _assert_optional_json_number(row[field], scenario_id, field, row)

        signal_metrics = row["deterministic_signal_evaluation_metrics"]
        assert isinstance(signal_metrics, dict)
        assert row["signal_evaluation_metrics"] == signal_metrics
        assert set(signal_metrics) == _REQUIRED_SIGNAL_METRIC_FIELDS
        for field in _REQUIRED_SIGNAL_NUMERIC_FIELDS:
            _assert_json_number(signal_metrics[field], scenario_id, field)
        _assert_signal_action_metrics(signal_metrics, scenario_id)


def _stage1_ablation_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"]
            ),
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "return_1": [0.01, -0.01, 0.02, -0.02],
            "liquidity_score": [20.0, 20.0, 20.0, 20.0],
            "chronos_expected_return": [0.02, -0.01, 0.03, -0.02],
            "granite_ttm_expected_return": [0.03, -0.02, 0.04, -0.03],
            "news_sentiment_mean": [0.1, -0.1, 0.2, -0.2],
            "news_negative_ratio": [0.0, 0.0, 0.0, 0.0],
            "text_risk_score": [0.0, 0.0, 0.0, 0.0],
            "sec_risk_flag": [0.0, 0.0, 0.0, 0.0],
            "sec_risk_flag_20d": [0.0, 0.0, 0.0, 0.0],
            "revenue_growth": [0.2, 0.3, 0.25, 0.35],
            "forward_return_5": [0.04, -0.03, 0.05, -0.04],
        }
    )


def _stage1_prediction_frame(features: pd.DataFrame) -> pd.DataFrame:
    return features[["date", "ticker", "forward_return_5"]].assign(
        raw_expected_return=[0.04, -0.02, 0.03, -0.01],
        expected_return=[0.04, -0.02, 0.03, -0.01],
        predicted_volatility=0.01,
        downside_quantile=-0.01,
        upside_quantile=0.06,
        quantile_width=0.07,
        model_confidence=0.8,
        text_risk_score=features["text_risk_score"],
        sec_risk_flag=features["sec_risk_flag"],
        sec_risk_flag_20d=features["sec_risk_flag_20d"],
        news_negative_ratio=features["news_negative_ratio"],
        liquidity_score=features["liquidity_score"],
    )


def _assert_json_number(value: object, scenario_id: str, field: str) -> None:
    assert type(value) in (int, float), f"{scenario_id}.{field} is {type(value).__name__}"
    assert math.isfinite(float(value)), f"{scenario_id}.{field} must be finite"


def _assert_optional_json_number(
    value: object,
    scenario_id: str,
    field: str,
    row: dict[str, Any],
) -> None:
    if value is None:
        assert row["validation_status"] == "not_evaluable", f"{scenario_id}.{field} is unexpectedly null"
        return
    _assert_json_number(value, scenario_id, field)


def _assert_signal_action_metrics(
    signal_metrics: dict[str, object],
    scenario_id: str,
) -> None:
    action_counts = signal_metrics["action_counts"]
    action_ratios = signal_metrics["action_ratios"]
    assert isinstance(action_counts, dict)
    assert isinstance(action_ratios, dict)
    assert set(action_counts) == {"BUY", "SELL", "HOLD"}
    assert set(action_ratios) == {"BUY", "SELL", "HOLD"}
    for action in ("BUY", "SELL", "HOLD"):
        _assert_json_number(action_counts[action], scenario_id, f"action_counts.{action}")
        _assert_json_number(action_ratios[action], scenario_id, f"action_ratios.{action}")
