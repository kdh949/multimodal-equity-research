from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import pytest

import quant_research.pipeline as pipeline
from quant_research.backtest.engine import BacktestResult
from quant_research.backtest.metrics import PerformanceMetrics
from quant_research.models.tabular import infer_feature_columns
from quant_research.pipeline import PipelineConfig


def test_no_model_proxy_ablation_removes_proxy_inputs_and_keeps_signal_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02"]),
            "ticker": ["AAPL", "MSFT", "AAPL"],
            "return_1": [0.01, -0.01, 0.0],
            "liquidity_score": [12.0, 12.0, 12.0],
            "chronos_expected_return": [0.9, 0.9, 0.9],
            "granite_ttm_expected_return": [0.8, 0.8, 0.8],
            "proxy_expected_return": [0.7, 0.7, 0.7],
            "proxy_predicted_volatility": [0.6, 0.6, 0.6],
            "news_sentiment_mean": [0.2, -0.2, 0.0],
            "news_negative_ratio": [0.0, 0.0, 0.0],
            "text_risk_score": [0.0, 0.0, 0.0],
            "sec_risk_flag": [0.0, 1.0, 0.0],
            "sec_risk_flag_20d": [0.0, 0.0, 0.0],
            "revenue_growth": [0.2, 0.1, 0.0],
            "forward_return_5": [0.05, -0.03, 0.0],
        }
    )
    predictions = features[["date", "ticker", "forward_return_5"]].assign(
        expected_return=0.01,
        predicted_volatility=0.02,
        downside_quantile=-0.02,
        model_confidence=0.5,
    )
    captured_feature_columns: list[tuple[str, ...]] = []
    captured_raw_columns: list[tuple[str, ...]] = []

    def fake_walk_forward_predict(
        variant: pd.DataFrame,
        _walk_config: object,
        *,
        target: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        captured_raw_columns.append(tuple(variant.columns))
        captured_feature_columns.append(tuple(infer_feature_columns(variant, target)))
        return (
            variant[["date", "ticker", target]].assign(
                raw_expected_return=[0.05, -0.01, 0.0],
                expected_return=[0.05, -0.01, 0.0],
                predicted_volatility=[0.01, 0.01, 0.01],
                downside_quantile=[-0.01, -0.01, -0.01],
                upside_quantile=[0.08, 0.01, 0.0],
                quantile_width=[0.09, 0.02, 0.01],
                model_confidence=[0.8, 0.2, 0.1],
                model_name="test",
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
                    "labeled_test_observations": [3],
                    "prediction_count": [3],
                }
            ),
        )

    monkeypatch.setattr(pipeline, "walk_forward_predict", fake_walk_forward_predict)

    summary = pipeline._run_ablation_summary(
        predictions,
        features,
        PipelineConfig(
            cost_bps=5.0,
            slippage_bps=2.0,
            native_tabular_isolation=False,
            validity_gate_ablation_modes=("no-model-proxy",),
        ),
    )

    assert [row["scenario"] for row in summary] == ["no_model_proxy"]
    disallowed_prefixes = ("chronos_", "granite_ttm_", "proxy_")
    assert all(
        not column.startswith(disallowed_prefixes)
        for column in captured_raw_columns[0]
    )
    assert captured_feature_columns == [
        (
            "return_1",
            "liquidity_score",
            "news_sentiment_mean",
            "news_negative_ratio",
            "text_risk_score",
            "sec_risk_flag",
            "sec_risk_flag_20d",
            "revenue_growth",
        )
    ]

    row = summary[0]
    assert row["pipeline_controls"]["model_proxy"] is False
    assert row["pipeline_controls"]["proxy_features"] is False
    assert row["pipeline_controls"]["proxy_model_inputs"] is False
    assert row["toggles"]["include_model_proxy_features"] is False
    assert row["toggles"]["include_proxy_features"] is False
    assert row["toggles"]["include_proxy_model_inputs"] is False
    assert row["proxy_removal_options"] == {
        "remove_proxy_features": True,
        "remove_proxy_model_inputs": True,
    }
    assert row["input_feature_families"] == ["price", "text", "sec"]
    assert row["input_feature_columns"] == list(captured_feature_columns[0])

    signal_metrics = row["deterministic_signal_evaluation_metrics"]
    assert signal_metrics["engine"] == "deterministic_signal_engine"
    assert signal_metrics["return_basis"] == "cost_adjusted_return"
    assert signal_metrics["realized_return_column"] == "forward_return_5"
    assert signal_metrics["signal_observations"] == 3
    assert signal_metrics["evaluation_observations"] == 2
    assert signal_metrics["action_counts"] == {"BUY": 1, "SELL": 0, "HOLD": 2}
    assert signal_metrics["effective_cost_bps"] == 5.0
    assert signal_metrics["effective_slippage_bps"] == 2.0
    assert signal_metrics["total_cost_return"] > 0

    assert row["signal_engine"] == "deterministic_signal_engine"
    assert row["signal_buy_count"] == 1
    assert row["signal_sell_count"] == 0
    assert row["signal_hold_count"] == 2


def test_original_and_no_model_proxy_ablation_share_execution_settings_except_proxy_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"]
            ),
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "return_1": [0.01, -0.01, 0.02, -0.02],
            "liquidity_score": [12.0, 13.0, 12.5, 13.5],
            "chronos_expected_return": [0.02, -0.01, 0.03, -0.02],
            "granite_ttm_expected_return": [0.03, -0.02, 0.04, -0.03],
            "proxy_expected_return": [0.04, -0.03, 0.05, -0.04],
            "proxy_predicted_volatility": [0.10, 0.11, 0.12, 0.13],
            "news_sentiment_mean": [0.1, -0.1, 0.2, -0.2],
            "news_negative_ratio": [0.0, 0.0, 0.0, 0.0],
            "text_risk_score": [0.0, 0.1, 0.0, 0.1],
            "sec_risk_flag": [0.0, 0.0, 0.0, 0.0],
            "sec_risk_flag_20d": [0.0, 0.0, 0.0, 0.0],
            "revenue_growth": [0.2, 0.3, 0.25, 0.35],
            "forward_return_20": [0.04, -0.03, 0.05, -0.04],
        }
    )
    config = PipelineConfig(
        train_periods=252,
        test_periods=60,
        gap_periods=20,
        embargo_periods=20,
        prediction_target_column="forward_return_20",
        required_validation_horizon=20,
        top_n=20,
        cost_bps=5.0,
        slippage_bps=2.0,
        average_daily_turnover_budget=0.25,
        benchmark_ticker="SPY",
        max_symbol_weight=0.10,
        max_daily_turnover=0.25,
        native_tabular_isolation=False,
        validity_gate_ablation_modes=("no-model-proxy",),
    )
    target_column = "forward_return_20"
    walk_forward_calls: list[dict[str, object]] = []
    backtest_calls: list[dict[str, object]] = []

    def fake_walk_forward_predict(
        variant: pd.DataFrame,
        walk_config: object,
        *,
        target: str,
        splitter: object,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        walk_forward_calls.append(
            {
                "columns": tuple(variant.columns),
                "feature_columns": tuple(infer_feature_columns(variant, target)),
                "walk_config": asdict(walk_config),
                "splitter_config": splitter.config.to_dict(),
                "target": target,
            }
        )
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
                model_name="execution_settings_fixture",
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

    def fake_run_long_only_backtest(
        frame: pd.DataFrame,
        backtest_config: object | None = None,
        *,
        benchmark_returns: object | None = None,
    ) -> BacktestResult:
        assert backtest_config is not None
        backtest_calls.append(
            {
                "columns": tuple(frame.columns),
                "controls": pipeline._backtest_control_payload(backtest_config),
                "benchmark_returns": benchmark_returns,
            }
        )
        dates = pd.to_datetime(frame["date"]).drop_duplicates().reset_index(drop=True)
        equity_curve = pd.DataFrame(
            {
                "date": dates,
                "cost_adjusted_return": [0.0] * len(dates),
                "gross_return": [0.0] * len(dates),
                "transaction_cost_return": [0.0] * len(dates),
                "slippage_cost_return": [0.0] * len(dates),
                "total_cost_return": [0.0] * len(dates),
                "turnover": [0.0] * len(dates),
                "exposure": [0.0] * len(dates),
                "portfolio_volatility_estimate": [0.0] * len(dates),
                "risk_stop_active": [False] * len(dates),
            }
        )
        signals = frame.copy()
        signals["action"] = "HOLD"
        return BacktestResult(
            equity_curve=equity_curve,
            weights=pd.DataFrame(),
            signals=signals,
            metrics=PerformanceMetrics(
                cagr=0.0,
                annualized_volatility=0.0,
                sharpe=0.0,
                max_drawdown=0.0,
                hit_rate=0.0,
                turnover=0.0,
                exposure=0.0,
                benchmark_cagr=0.0,
                excess_return=0.0,
            ),
        )

    monkeypatch.setattr(pipeline, "walk_forward_predict", fake_walk_forward_predict)
    monkeypatch.setattr(pipeline, "run_long_only_backtest", fake_run_long_only_backtest)

    original_predictions, _ = pipeline._predict_walk_forward_with_splitter(
        features,
        pipeline._walk_forward_config(config),
        target=target_column,
        splitter=pipeline._walk_forward_splitter(config, target_column=target_column),
    )
    original_predictions = pipeline._attach_signal_features(original_predictions, features)
    original_predictions = pipeline._ensure_backtest_prediction_columns(
        original_predictions,
        target_column,
    )
    pipeline.run_long_only_backtest(
        original_predictions,
        pipeline._backtest_config(config, realized_return_column=target_column),
        benchmark_returns=None,
    )

    summary = pipeline._run_ablation_summary(
        original_predictions,
        features,
        config,
        benchmark_return_series=None,
    )

    assert [row["scenario"] for row in summary] == ["no_model_proxy"]
    assert len(walk_forward_calls) == 2
    assert len(backtest_calls) == 2

    original_walk, ablation_walk = walk_forward_calls
    assert original_walk["walk_config"] == ablation_walk["walk_config"]
    assert original_walk["splitter_config"] == ablation_walk["splitter_config"]
    assert original_walk["target"] == ablation_walk["target"] == target_column
    assert original_walk["walk_config"]["train_periods"] == 252
    assert original_walk["walk_config"]["test_periods"] == 60
    assert original_walk["walk_config"]["gap_periods"] == 20
    assert original_walk["walk_config"]["embargo_periods"] == 20
    assert original_walk["walk_config"]["prediction_horizon_periods"] == 20
    assert original_walk["splitter_config"]["target_column"] == "forward_return_20"
    assert original_walk["splitter_config"]["purge_periods"] == 20
    assert original_walk["splitter_config"]["embargo_periods"] == 20

    proxy_columns = {
        "chronos_expected_return",
        "granite_ttm_expected_return",
        "proxy_expected_return",
        "proxy_predicted_volatility",
    }
    learned_proxy_columns = {"chronos_expected_return", "granite_ttm_expected_return"}
    assert proxy_columns.issubset(set(original_walk["columns"]))
    assert learned_proxy_columns.issubset(set(original_walk["feature_columns"]))
    assert proxy_columns.isdisjoint(set(ablation_walk["columns"]))
    assert (
        set(original_walk["feature_columns"]) - set(ablation_walk["feature_columns"])
        == learned_proxy_columns
    )
    assert set(ablation_walk["feature_columns"]).issubset(set(original_walk["feature_columns"]))

    original_backtest, ablation_backtest = backtest_calls
    assert original_backtest["controls"] == ablation_backtest["controls"]
    assert original_backtest["controls"]["top_n"] == 20
    assert original_backtest["controls"]["cost_bps"] == 5.0
    assert original_backtest["controls"]["slippage_bps"] == 2.0
    assert original_backtest["controls"]["average_daily_turnover_budget"] == 0.25
    assert original_backtest["controls"]["max_symbol_weight"] == 0.10
    assert original_backtest["controls"]["max_daily_turnover"] == 0.25
    assert original_backtest["controls"]["realized_return_column"] == target_column
