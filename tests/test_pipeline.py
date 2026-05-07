from __future__ import annotations

import pandas as pd
import pytest

import quant_research.pipeline as pipeline
from quant_research.models.tabular import infer_feature_columns
from quant_research.pipeline import (
    PipelineConfig,
    _apply_model_feature_ablation_toggles,
    _attach_signal_features,
    _evaluated_ticker_universe,
    run_research_pipeline,
)
from quant_research.validation import (
    DEFAULT_VALIDITY_GATE_ABLATION_MODE_IDS,
    NO_COST_ABLATION_SCENARIO,
    VALID_FEATURE_FAMILIES,
    AblationScenarioConfig,
    AblationToggles,
    default_ablation_registry,
    feature_family_for_column,
)


def test_synthetic_pipeline_runs_end_to_end() -> None:
    result = run_research_pipeline(
        PipelineConfig(
            tickers=["SPY", "AAPL", "MSFT"],
            data_mode="synthetic",
            train_periods=60,
            test_periods=15,
            top_n=2,
            enable_feature_model_ablation=True,
        )
    )

    assert not result.features.empty
    assert not result.predictions.empty
    assert not result.signals.empty
    assert "sec_event_tag" in result.signals
    assert result.signals["sec_event_tag"].map(type).eq(str).all()
    assert "sec_summary_ref" in result.signals
    assert not result.backtest.equity_curve.empty
    assert "forward_return_20" in result.predictions
    assert "forward_return_1" in result.backtest.signals
    assert result.backtest.equity_curve["realized_return_column"].eq("forward_return_20").all()
    assert result.benchmark_return_series is not None
    assert result.benchmark_return_series["return_column"].eq("forward_return_20").all()
    assert result.benchmark_return_series["return_horizon"].eq(20).all()
    assert "is_oos" in result.validation_summary
    assert result.validation_summary["is_oos"].any()
    assert {
        "all_features",
        "no_text_risk",
        "no_sec_risk",
        "no_model_proxy",
        "no_costs",
        "price_only",
        "text_only",
        "sec_only",
        "full_model_features",
        "no_chronos_features",
        "no_granite_features",
        "tabular_without_ts_proxies",
    } == {
        row["scenario"] for row in result.ablation_summary
    }
    allowlists = {
        row["scenario"]: row["feature_family_allowlist"]
        for row in result.ablation_summary
        if row["scenario"] in {"price_only", "text_only", "sec_only"}
    }
    assert allowlists == {
        "price_only": ["price", "chronos", "granite_ttm"],
        "text_only": ["text"],
        "sec_only": ["sec"],
    }


def test_evaluated_ticker_universe_keeps_configured_tickers_that_reach_predictions() -> None:
    predictions = pd.DataFrame({"ticker": ["msft", "UNUSED", "AAPL", "AAPL"]})

    universe = _evaluated_ticker_universe(predictions, ["AAPL", "MSFT", "TSLA"])

    assert universe == ("AAPL", "MSFT")


def test_synthetic_pipeline_features_include_recency_coverage_news_columns() -> None:
    result = run_research_pipeline(
        PipelineConfig(
            tickers=["AAPL", "MSFT"],
            data_mode="synthetic",
            train_periods=60,
            test_periods=15,
            top_n=2,
        )
    )

    for column in [
        "news_recency_decay",
        "news_staleness_days",
        "news_coverage_5d",
        "news_coverage_20d",
    ]:
        assert column in result.features.columns
        assert pd.api.types.is_numeric_dtype(result.features[column])
    assert result.features["news_coverage_20d"].ge(0).all()
    assert result.features["news_coverage_5d"].ge(0).all()
    assert result.features["news_staleness_days"].ge(0).all()


def test_data_channel_ablation_toggles_filter_model_feature_sources() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"]),
            "ticker": ["AAPL"],
            "return_1": [0.01],
            "liquidity_score": [12.0],
            "chronos_expected_return": [0.02],
            "granite_ttm_expected_return": [0.03],
            "news_sentiment_mean": [0.1],
            "news_top_event": ["earnings"],
            "text_risk_score": [0.0],
            "sec_risk_flag": [1.0],
            "sec_event_tag": ["legal"],
            "sec_summary_ref": ["Form 8-K"],
            "revenue_growth": [0.2],
            "net_income_growth": [0.1],
            "assets_growth": [0.05],
            "forward_return_1": [0.01],
            "forward_return_5": [0.04],
        }
    )
    registry = default_ablation_registry()

    price_only = _apply_model_feature_ablation_toggles(frame, registry.get("price_only").toggles)
    assert {"return_1", "liquidity_score", "chronos_expected_return", "granite_ttm_expected_return"}.issubset(
        price_only.columns
    )
    assert {"news_sentiment_mean", "text_risk_score", "sec_risk_flag", "revenue_growth"}.isdisjoint(
        price_only.columns
    )
    assert {"date", "ticker", "forward_return_1", "forward_return_5"}.issubset(price_only.columns)

    text_only = _apply_model_feature_ablation_toggles(frame, registry.get("text_only").toggles)
    assert {"news_sentiment_mean", "text_risk_score"}.issubset(text_only.columns)
    assert {
        "return_1",
        "liquidity_score",
        "chronos_expected_return",
        "granite_ttm_expected_return",
        "sec_risk_flag",
        "revenue_growth",
    }.isdisjoint(text_only.columns)
    assert {"date", "ticker", "forward_return_1", "forward_return_5"}.issubset(text_only.columns)

    sec_only = _apply_model_feature_ablation_toggles(frame, registry.get("sec_only").toggles)
    assert {"sec_risk_flag", "sec_event_tag", "sec_summary_ref", "revenue_growth"}.issubset(sec_only.columns)
    assert {
        "return_1",
        "liquidity_score",
        "chronos_expected_return",
        "granite_ttm_expected_return",
        "news_sentiment_mean",
        "text_risk_score",
    }.isdisjoint(sec_only.columns)
    assert {"date", "ticker", "forward_return_1", "forward_return_5"}.issubset(sec_only.columns)


def test_ablation_feature_selection_excludes_predictions_but_keeps_signal_inputs() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"]),
            "ticker": ["AAPL"],
            "return_1": [0.01],
            "liquidity_score": [12.0],
            "raw_expected_return": [0.01],
            "expected_return": [0.012],
            "predicted_volatility": [0.02],
            "downside_quantile": [-0.03],
            "upside_quantile": [0.04],
            "quantile_width": [0.07],
            "model_confidence": [0.8],
            "model_calibration_scale": [1.1],
            "model_calibration_bias": [0.001],
            "proxy_expected_return": [0.015],
            "proxy_predicted_volatility": [0.025],
            "news_negative_ratio": [0.1],
            "text_risk_score": [0.2],
            "sec_risk_flag": [0.0],
            "sec_risk_flag_20d": [0.0],
            "forward_return_5": [0.04],
        }
    )

    variant = _apply_model_feature_ablation_toggles(
        frame,
        default_ablation_registry().get("all_features").toggles,
    )
    feature_columns = set(infer_feature_columns(variant, "forward_return_5"))

    assert {
        "expected_return",
        "predicted_volatility",
        "downside_quantile",
        "model_confidence",
        "proxy_expected_return",
        "proxy_predicted_volatility",
    }.issubset(variant.columns)
    assert {
        "raw_expected_return",
        "expected_return",
        "predicted_volatility",
        "downside_quantile",
        "upside_quantile",
        "quantile_width",
        "model_confidence",
        "model_calibration_scale",
        "model_calibration_bias",
        "proxy_expected_return",
        "proxy_predicted_volatility",
    }.isdisjoint(feature_columns)
    assert {
        "return_1",
        "liquidity_score",
        "news_negative_ratio",
        "text_risk_score",
        "sec_risk_flag",
        "sec_risk_flag_20d",
    }.issubset(feature_columns)


def test_proxy_model_input_ablation_keeps_features_but_removes_walk_forward_inputs() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"]),
            "ticker": ["AAPL"],
            "return_1": [0.01],
            "liquidity_score": [12.0],
            "chronos_expected_return": [0.02],
            "granite_ttm_expected_return": [0.03],
            "proxy_expected_return": [0.015],
            "proxy_predicted_volatility": [0.025],
            "news_negative_ratio": [0.1],
            "forward_return_5": [0.04],
        }
    )
    scenario = AblationScenarioConfig(
        scenario_id="proxy_inputs_off",
        kind="model_feature",
        label="Proxy inputs off",
        toggles=AblationToggles(include_proxy_model_inputs=False),
    )

    feature_variant = pipeline._apply_model_feature_ablation_scenario(frame, scenario)
    model_input_variant = pipeline._model_input_features_for_ablation(feature_variant, scenario)

    assert {
        "chronos_expected_return",
        "granite_ttm_expected_return",
        "proxy_expected_return",
        "proxy_predicted_volatility",
    }.issubset(feature_variant.columns)
    assert {
        "chronos_expected_return",
        "granite_ttm_expected_return",
        "proxy_expected_return",
        "proxy_predicted_volatility",
    }.isdisjoint(model_input_variant.columns)
    assert tuple(infer_feature_columns(model_input_variant, "forward_return_5")) == (
        "return_1",
        "liquidity_score",
        "news_negative_ratio",
    )


def test_feature_model_ablation_passes_only_permitted_families_to_walk_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "ticker": ["AAPL", "AAPL"],
            "return_1": [0.01, 0.02],
            "liquidity_score": [12.0, 13.0],
            "chronos_expected_return": [0.02, 0.03],
            "granite_ttm_expected_return": [0.03, 0.04],
            "news_sentiment_mean": [0.1, 0.2],
            "text_risk_score": [0.0, 0.1],
            "sec_risk_flag": [1.0, 0.0],
            "revenue_growth": [0.2, 0.3],
            "forward_return_1": [0.01, 0.02],
            "forward_return_5": [0.04, 0.05],
        }
    )
    captured_features: dict[str, tuple[str, ...]] = {}
    captured_families: dict[str, set[str]] = {}
    registry = default_ablation_registry()
    scenario_order = iter(registry.by_kind("data_channel") + registry.by_kind("model_feature"))

    def fake_walk_forward_predict(
        variant: pd.DataFrame,
        _walk_config: object,
        *,
        target: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        scenario = next(scenario_order)
        feature_columns = tuple(infer_feature_columns(variant, target))
        captured_features[scenario.scenario_id] = feature_columns
        captured_families[scenario.scenario_id] = {
            family
            for column in feature_columns
            if (family := feature_family_for_column(column)) is not None
        }
        return (
            variant[["date", "ticker", target]].assign(
                raw_expected_return=0.01,
                expected_return=0.01,
                predicted_volatility=0.02,
                downside_quantile=-0.02,
                upside_quantile=0.04,
                quantile_width=0.06,
                model_confidence=0.5,
                model_name="test",
                fold=0,
                is_oos=True,
            ),
            pd.DataFrame(),
        )

    monkeypatch.setattr(pipeline, "walk_forward_predict", fake_walk_forward_predict)

    pipeline._run_feature_model_ablation_summary(
        frame,
        PipelineConfig(native_tabular_isolation=False),
    )

    assert captured_features["price_only"] == (
        "return_1",
        "liquidity_score",
        "chronos_expected_return",
        "granite_ttm_expected_return",
    )
    assert captured_features["text_only"] == ("news_sentiment_mean", "text_risk_score")
    assert captured_features["sec_only"] == ("sec_risk_flag", "revenue_growth")
    assert captured_features["full_model_features"] == (
        "return_1",
        "liquidity_score",
        "chronos_expected_return",
        "granite_ttm_expected_return",
        "news_sentiment_mean",
        "text_risk_score",
        "sec_risk_flag",
        "revenue_growth",
    )
    assert "chronos_expected_return" not in captured_features["no_chronos_features"]
    assert "granite_ttm_expected_return" not in captured_features["no_granite_features"]
    assert {
        "chronos_expected_return",
        "granite_ttm_expected_return",
    }.isdisjoint(captured_features["tabular_without_ts_proxies"])

    for scenario_id in ("price_only", "text_only", "sec_only"):
        permitted = set(registry.get(scenario_id).permitted_feature_families)
        assert captured_families[scenario_id].issubset(permitted)
        assert captured_families[scenario_id].isdisjoint(VALID_FEATURE_FAMILIES - permitted)


def test_stage1_ablation_summary_runs_data_channel_refits_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-02",
                ]
            ),
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "return_1": [0.01, -0.01, 0.02, -0.02],
            "liquidity_score": [12.0, 13.0, 12.5, 13.5],
            "chronos_expected_return": [0.02, -0.01, 0.03, -0.02],
            "granite_ttm_expected_return": [0.03, -0.02, 0.04, -0.03],
            "news_sentiment_mean": [0.1, -0.1, 0.2, -0.2],
            "text_risk_score": [0.0, 0.1, 0.0, 0.1],
            "sec_risk_flag": [0.0, 0.0, 0.0, 0.0],
            "revenue_growth": [0.2, 0.3, 0.25, 0.35],
            "forward_return_5": [0.04, -0.03, 0.05, -0.04],
        }
    )
    predictions = frame[["date", "ticker", "forward_return_5"]].assign(
        raw_expected_return=[0.02, -0.01, 0.03, -0.02],
        expected_return=[0.02, -0.01, 0.03, -0.02],
        predicted_volatility=0.02,
        downside_quantile=-0.02,
        upside_quantile=0.04,
        quantile_width=0.06,
        model_confidence=0.5,
        text_risk_score=0.0,
        sec_risk_flag=0.0,
        sec_risk_flag_20d=0.0,
        news_negative_ratio=0.0,
        liquidity_score=12.0,
    )
    captured_features: list[tuple[str, ...]] = []

    def fake_walk_forward_predict(
        variant: pd.DataFrame,
        _walk_config: object,
        *,
        target: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        captured_features.append(tuple(infer_feature_columns(variant, target)))
        return (
            variant[["date", "ticker", target]].assign(
                raw_expected_return=0.01,
                expected_return=0.01,
                predicted_volatility=0.02,
                downside_quantile=-0.02,
                upside_quantile=0.04,
                quantile_width=0.06,
                model_confidence=0.5,
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
                    "labeled_test_observations": [len(variant)],
                    "prediction_count": [len(variant)],
                }
            ),
        )

    monkeypatch.setattr(pipeline, "walk_forward_predict", fake_walk_forward_predict)

    summary = pipeline._run_ablation_summary(
        predictions,
        frame,
        PipelineConfig(native_tabular_isolation=False),
    )

    rows = {str(row["scenario"]): row for row in summary}
    assert {"price_only", "text_only", "sec_only"}.issubset(rows)
    assert captured_features[-3:] == [
        (
            "return_1",
            "liquidity_score",
            "chronos_expected_return",
            "granite_ttm_expected_return",
        ),
        ("news_sentiment_mean", "text_risk_score"),
        ("sec_risk_flag", "revenue_growth"),
    ]
    expected_input_families = {
        "price_only": ["price", "chronos", "granite_ttm"],
        "text_only": ["text"],
        "sec_only": ["sec"],
    }
    required_metric_fields = {
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
        "validation_mean_mae",
        "validation_mean_directional_accuracy",
        "validation_mean_information_coefficient",
        "validation_positive_ic_fold_ratio",
        "validation_oos_information_coefficient",
    }
    for scenario_id in ("price_only", "text_only", "sec_only"):
        assert rows[scenario_id]["kind"] == "data_channel"
        assert rows[scenario_id]["input_feature_families"] == expected_input_families[scenario_id]
        assert set(required_metric_fields).issubset(rows[scenario_id])
        assert rows[scenario_id]["validation_status"] == "pass"
        assert rows[scenario_id]["validation_fold_count"] == 1
        assert rows[scenario_id]["validation_oos_fold_count"] == 1
        assert rows[scenario_id]["validation_mean_information_coefficient"] == 0.5


def test_ablation_validation_metrics_preserve_insufficient_data_skip_result() -> None:
    validation_summary = pd.DataFrame(
        [
            {
                "validation_status": "insufficient_data",
                "skip_status": "skipped",
                "skip_code": "insufficient_labeled_dates",
                "reason": "not enough labeled dates to create a walk-forward fold",
                "fold_count": 0,
                "candidate_fold_count": 0,
                "labeled_date_count": 8,
                "required_min_date_count": 26,
                "is_oos": False,
            }
        ]
    )
    predictions = pd.DataFrame(columns=["date", "ticker", "expected_return", "forward_return_5"])

    metrics = pipeline._ablation_validation_metrics(
        validation_summary,
        predictions,
        target_column="forward_return_5",
    )

    assert metrics["validation_status"] == "insufficient_data"
    assert metrics["validation_skip_status"] == "skipped"
    assert metrics["validation_skip_code"] == "insufficient_labeled_dates"
    assert metrics["validation_fold_count"] == 0
    assert metrics["validation_labeled_date_count"] == 8


def test_ablation_summary_computation_produces_exact_five_expected_scenarios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-02",
                ]
            ),
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "return_1": [0.01, -0.01, 0.02, -0.02],
            "liquidity_score": [20.0, 20.0, 20.0, 20.0],
            "chronos_expected_return": [0.02, -0.01, 0.03, -0.02],
            "granite_ttm_expected_return": [0.03, -0.02, 0.04, -0.03],
            "news_sentiment_mean": [0.1, -0.1, 0.2, -0.2],
            "news_negative_ratio": [0.0, 0.0, 0.0, 0.0],
            "text_risk_score": [0.0, 0.2, 0.0, 0.2],
            "sec_risk_flag": [0.0, 0.0, 0.0, 0.0],
            "sec_risk_flag_20d": [0.0, 0.0, 0.0, 0.0],
            "revenue_growth": [0.2, 0.3, 0.25, 0.35],
            "forward_return_5": [0.04, -0.03, 0.05, -0.04],
        }
    )
    predictions = features[["date", "ticker", "forward_return_5"]].assign(
        raw_expected_return=[0.04, -0.02, 0.05, -0.03],
        expected_return=[0.04, -0.02, 0.05, -0.03],
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
    requested_scenarios = (
        "all_features",
        "no_text_risk",
        "no_sec_risk",
        "no_model_proxy",
        NO_COST_ABLATION_SCENARIO,
    )
    expected_scenarios = (
        "all_features",
        "no_text_risk",
        "no_sec_risk",
        NO_COST_ABLATION_SCENARIO,
        "no_model_proxy",
    )
    captured_refit_features: list[tuple[str, ...]] = []

    def fake_walk_forward_predict(
        variant: pd.DataFrame,
        _walk_config: object,
        *,
        target: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        captured_refit_features.append(tuple(infer_feature_columns(variant, target)))
        row_count = len(variant)
        return (
            variant[["date", "ticker", target]].assign(
                raw_expected_return=[0.04, -0.02, 0.05, -0.03][:row_count],
                expected_return=[0.04, -0.02, 0.05, -0.03][:row_count],
                predicted_volatility=0.01,
                downside_quantile=-0.01,
                upside_quantile=0.06,
                quantile_width=0.07,
                model_confidence=0.8,
                model_name="deterministic_fixture",
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
            validity_gate_ablation_modes=requested_scenarios,
        ),
    )

    assert tuple(row["scenario"] for row in summary) == expected_scenarios
    assert len(summary) == 5
    assert len({row["scenario"] for row in summary}) == 5
    assert tuple(row["kind"] for row in summary) == (
        "signal",
        "signal",
        "signal",
        "cost",
        "pipeline_control",
    )
    assert captured_refit_features == [
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

    rows = {str(row["scenario"]): row for row in summary}
    assert rows[NO_COST_ABLATION_SCENARIO]["effective_cost_bps"] == 0.0
    assert rows[NO_COST_ABLATION_SCENARIO]["effective_slippage_bps"] == 0.0
    assert rows["no_model_proxy"]["validation_status"] == "pass"
    for scenario_id in tuple(
        scenario_id for scenario_id in expected_scenarios if scenario_id != NO_COST_ABLATION_SCENARIO
    ):
        assert rows[scenario_id]["effective_cost_bps"] == 8.0
        assert rows[scenario_id]["effective_slippage_bps"] == 3.0


def test_pipeline_config_defaults_to_stage1_validity_gate_ablation_modes() -> None:
    assert PipelineConfig().validity_gate_ablation_modes == DEFAULT_VALIDITY_GATE_ABLATION_MODE_IDS
    assert PipelineConfig(validity_gate_ablation_modes=("no-model-proxy",)).validity_gate_ablation_modes == (
        "no_model_proxy",
    )
    assert PipelineConfig(validity_gate_ablation_modes=("no-cost",)).validity_gate_ablation_modes == (
        NO_COST_ABLATION_SCENARIO,
    )

    with pytest.raises(ValueError, match="unsupported validity gate ablation mode"):
        PipelineConfig(validity_gate_ablation_modes=("unknown_mode",))


def test_no_model_proxy_validity_gate_mode_runs_only_that_pipeline_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "ticker": ["AAPL", "AAPL"],
            "return_1": [0.01, 0.02],
            "liquidity_score": [12.0, 13.0],
            "chronos_expected_return": [0.02, 0.03],
            "granite_ttm_expected_return": [0.03, 0.04],
            "news_sentiment_mean": [0.1, 0.2],
            "text_risk_score": [0.0, 0.1],
            "sec_risk_flag": [1.0, 0.0],
            "revenue_growth": [0.2, 0.3],
            "forward_return_5": [0.04, 0.05],
        }
    )
    predictions = frame[["date", "ticker", "forward_return_5"]].assign(
        raw_expected_return=0.01,
        expected_return=0.01,
        predicted_volatility=0.02,
        downside_quantile=-0.02,
        upside_quantile=0.04,
        quantile_width=0.06,
        model_confidence=0.5,
        text_risk_score=0.0,
        sec_risk_flag=0.0,
        sec_risk_flag_20d=0.0,
        news_negative_ratio=0.0,
        liquidity_score=12.0,
    )
    captured_features: list[tuple[str, ...]] = []

    def fake_walk_forward_predict(
        variant: pd.DataFrame,
        _walk_config: object,
        *,
        target: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        captured_features.append(tuple(infer_feature_columns(variant, target)))
        return (
            variant[["date", "ticker", target]].assign(
                raw_expected_return=0.01,
                expected_return=0.01,
                predicted_volatility=0.02,
                downside_quantile=-0.02,
                upside_quantile=0.04,
                quantile_width=0.06,
                model_confidence=0.5,
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
                }
            ),
        )

    monkeypatch.setattr(pipeline, "walk_forward_predict", fake_walk_forward_predict)

    summary = pipeline._run_ablation_summary(
        predictions,
        frame,
        PipelineConfig(
            native_tabular_isolation=False,
            validity_gate_ablation_modes=("no-model-proxy",),
        ),
    )

    assert [row["scenario"] for row in summary] == ["no_model_proxy"]
    assert captured_features == [
        (
            "return_1",
            "liquidity_score",
            "news_sentiment_mean",
            "text_risk_score",
            "sec_risk_flag",
            "revenue_growth",
        )
    ]
    assert summary[0]["pipeline_controls"]["model_proxy"] is False
    assert summary[0]["validation_mean_information_coefficient"] == 0.5


def test_no_model_proxy_returns_deterministic_signal_evaluation_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "ticker": ["AAPL", "AAPL"],
            "return_1": [0.01, 0.02],
            "liquidity_score": [12.0, 13.0],
            "chronos_expected_return": [0.02, 0.03],
            "granite_ttm_expected_return": [0.03, 0.04],
            "news_sentiment_mean": [0.1, 0.2],
            "text_risk_score": [0.0, 0.0],
            "sec_risk_flag": [1.0, 0.0],
            "sec_risk_flag_20d": [0.0, 0.0],
            "news_negative_ratio": [0.0, 0.0],
            "revenue_growth": [0.2, 0.3],
            "forward_return_5": [0.04, 0.05],
        }
    )
    predictions = frame[["date", "ticker", "forward_return_5"]].assign(
        raw_expected_return=0.01,
        expected_return=0.01,
        predicted_volatility=0.02,
        downside_quantile=-0.02,
        upside_quantile=0.04,
        quantile_width=0.06,
        model_confidence=0.5,
        text_risk_score=0.0,
        sec_risk_flag=0.0,
        sec_risk_flag_20d=0.0,
        news_negative_ratio=0.0,
        liquidity_score=12.0,
    )

    def fake_walk_forward_predict(
        variant: pd.DataFrame,
        _walk_config: object,
        *,
        target: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return (
            variant[["date", "ticker", target]].assign(
                raw_expected_return=0.01,
                expected_return=0.01,
                predicted_volatility=0.02,
                downside_quantile=-0.02,
                upside_quantile=0.04,
                quantile_width=0.06,
                model_confidence=0.5,
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
                }
            ),
        )

    monkeypatch.setattr(pipeline, "walk_forward_predict", fake_walk_forward_predict)

    summary = pipeline._run_ablation_summary(
        predictions,
        frame,
        PipelineConfig(
            cost_bps=5.0,
            slippage_bps=2.0,
            native_tabular_isolation=False,
            validity_gate_ablation_modes=("no-model-proxy",),
        ),
    )

    row = summary[0]
    metrics = row["signal_evaluation_metrics"]

    assert metrics["engine"] == "deterministic_signal_engine"
    assert metrics["return_basis"] == "cost_adjusted_return"
    assert metrics["realized_return_column"] == "forward_return_5"
    assert metrics["effective_cost_bps"] == 5.0
    assert metrics["effective_slippage_bps"] == 2.0
    assert metrics["signal_observations"] == 2
    assert metrics["evaluation_observations"] == 2
    assert metrics["action_counts"] == {"BUY": 1, "SELL": 0, "HOLD": 1}
    assert metrics["action_ratios"] == {"BUY": 0.5, "SELL": 0.0, "HOLD": 0.5}
    assert metrics["risk_stop_observation_count"] == 0
    assert metrics["gross_cumulative_return"] > metrics["cost_adjusted_cumulative_return"]
    assert metrics["transaction_cost_return"] > 0
    assert metrics["slippage_cost_return"] > 0
    assert metrics["total_cost_return"] == pytest.approx(
        metrics["transaction_cost_return"] + metrics["slippage_cost_return"]
    )
    position_metrics = row["position_level_metrics"]
    assert position_metrics["position_count"] > 0
    assert position_metrics["position_return_basis"] == "position_net_return"
    assert position_metrics["position_total_cost_return"] == pytest.approx(
        metrics["total_cost_return"]
    )
    assert position_metrics["position_net_return_contribution"] == pytest.approx(
        position_metrics["position_gross_return_contribution"]
        - position_metrics["position_total_cost_return"]
    )

    assert row["signal_engine"] == "deterministic_signal_engine"
    assert row["signal_buy_count"] == 1
    assert row["signal_sell_count"] == 0
    assert row["signal_hold_count"] == 1
    assert row["signal_total_cost_return"] == pytest.approx(metrics["total_cost_return"])
    assert row["signal_cost_adjusted_cumulative_return"] == pytest.approx(
        metrics["cost_adjusted_cumulative_return"]
    )


def test_no_model_proxy_pipeline_control_removes_optional_proxy_features_only() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"]),
            "ticker": ["AAPL"],
            "return_1": [0.01],
            "liquidity_score": [12.0],
            "chronos_expected_return": [0.02],
            "granite_ttm_expected_return": [0.03],
            "news_sentiment_mean": [0.1],
            "text_risk_score": [0.0],
            "sec_risk_flag": [1.0],
            "revenue_growth": [0.2],
            "proxy_expected_return": [0.02],
            "proxy_predicted_volatility": [0.02],
            "forward_return_1": [0.01],
            "forward_return_5": [0.04],
        }
    )
    toggles = default_ablation_registry().get("no_model_proxy").toggles

    variant = _apply_model_feature_ablation_toggles(frame, toggles)

    assert pipeline._requires_model_refit(toggles) is True
    assert {
        "chronos_expected_return",
        "granite_ttm_expected_return",
        "proxy_expected_return",
        "proxy_predicted_volatility",
    }.isdisjoint(variant.columns)
    assert {"return_1", "liquidity_score", "news_sentiment_mean", "sec_risk_flag"}.issubset(
        variant.columns
    )


def test_no_cost_ablation_disables_effective_cost_slippage_and_turnover_costs() -> None:
    config = PipelineConfig(cost_bps=8.0, slippage_bps=3.0)
    scenario = default_ablation_registry().get(NO_COST_ABLATION_SCENARIO)
    toggles = scenario.toggles

    backtest_config = pipeline._backtest_config_for_ablation(config, scenario)

    assert scenario.kind == "cost"
    assert pipeline._requires_model_refit(toggles) is False
    assert toggles.pipeline_control_toggles() == {
        "model_proxy": True,
        "proxy_features": True,
        "proxy_model_inputs": True,
        "cost": False,
        "slippage": False,
        "turnover": False,
    }
    assert backtest_config.cost_bps == 0.0
    assert backtest_config.slippage_bps == 0.0
    assert pipeline._backtest_config(config).cost_bps == 8.0
    assert pipeline._backtest_config(config).slippage_bps == 3.0


def test_disabled_cost_toggles_do_not_zero_costs_outside_no_cost_scenario() -> None:
    config = PipelineConfig(cost_bps=8.0, slippage_bps=3.0)
    scenario = AblationScenarioConfig(
        scenario_id="cost_controls_off_but_not_active",
        kind="signal",
        label="Cost controls off but not active",
        toggles=AblationToggles(
            include_transaction_costs=False,
            include_slippage=False,
            include_turnover_costs=False,
        ),
    )

    backtest_config = pipeline._backtest_config_for_ablation(config, scenario)
    toggle_only_backtest_config = pipeline._backtest_config_for_ablation(config, scenario.toggles)

    assert scenario.scenario_id != NO_COST_ABLATION_SCENARIO
    assert backtest_config.cost_bps == 8.0
    assert backtest_config.slippage_bps == 3.0
    assert toggle_only_backtest_config.cost_bps == 8.0
    assert toggle_only_backtest_config.slippage_bps == 3.0


def test_no_cost_ablation_mode_runs_only_cost_scenario_without_mutating_config() -> None:
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "expected_return": [0.03, 0.03, 0.03],
            "predicted_volatility": [0.01, 0.01, 0.01],
            "downside_quantile": [-0.01, -0.01, -0.01],
            "upside_quantile": [0.04, 0.04, 0.04],
            "quantile_width": [0.05, 0.05, 0.05],
            "model_confidence": [0.8, 0.8, 0.8],
            "text_risk_score": [0.0, 0.0, 0.0],
            "sec_risk_flag": [0.0, 0.0, 0.0],
            "sec_risk_flag_20d": [0.0, 0.0, 0.0],
            "news_negative_ratio": [0.0, 0.0, 0.0],
            "liquidity_score": [20.0, 20.0, 20.0],
            "forward_return_5": [0.01, 0.01, 0.01],
        }
    )
    config = PipelineConfig(
        cost_bps=8.0,
        slippage_bps=3.0,
        validity_gate_ablation_modes=("no-cost",),
    )

    summary = pipeline._run_ablation_summary(predictions, predictions, config)

    assert [row["scenario"] for row in summary] == [NO_COST_ABLATION_SCENARIO]
    row = summary[0]
    metrics = row["deterministic_signal_evaluation_metrics"]
    assert row["kind"] == "cost"
    assert row["effective_cost_bps"] == 0.0
    assert row["effective_slippage_bps"] == 0.0
    assert metrics["effective_cost_bps"] == 0.0
    assert metrics["effective_slippage_bps"] == 0.0
    assert metrics["total_cost_return"] == 0.0
    assert config.cost_bps == 8.0
    assert config.slippage_bps == 3.0


def test_no_cost_ablation_is_only_stage1_scenario_without_cost_penalties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-02",
                ]
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
            "proxy_expected_return": [0.02, -0.01, 0.03, -0.02],
            "proxy_predicted_volatility": [0.02, 0.02, 0.02, 0.02],
            "forward_return_5": [0.04, -0.03, 0.05, -0.04],
        }
    )
    predictions = features[["date", "ticker", "forward_return_5"]].assign(
        raw_expected_return=[0.04, -0.02, 0.05, -0.03],
        expected_return=[0.04, -0.02, 0.05, -0.03],
        predicted_volatility=0.01,
        downside_quantile=-0.01,
        upside_quantile=0.06,
        quantile_width=0.07,
        model_confidence=0.8,
        text_risk_score=0.0,
        sec_risk_flag=0.0,
        sec_risk_flag_20d=0.0,
        news_negative_ratio=0.0,
        liquidity_score=20.0,
    )

    def fake_walk_forward_predict(
        variant: pd.DataFrame,
        _walk_config: object,
        *,
        target: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        row_count = len(variant)
        return (
            variant[["date", "ticker", target]].assign(
                raw_expected_return=[0.04, -0.02, 0.05, -0.03][:row_count],
                expected_return=[0.04, -0.02, 0.05, -0.03][:row_count],
                predicted_volatility=0.01,
                downside_quantile=-0.01,
                upside_quantile=0.06,
                quantile_width=0.07,
                model_confidence=0.8,
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
                    "labeled_test_observations": [row_count],
                    "prediction_count": [row_count],
                }
            ),
        )

    monkeypatch.setattr(pipeline, "walk_forward_predict", fake_walk_forward_predict)

    config = PipelineConfig(
        cost_bps=8.0,
        slippage_bps=3.0,
        native_tabular_isolation=False,
        validity_gate_ablation_modes=(
            "all_features",
            "no_text_risk",
            "no-cost",
            "no-model-proxy",
            "price_only",
        ),
    )
    summary = pipeline._run_ablation_summary(predictions, features, config)
    rows = {str(row["scenario"]): row for row in summary}

    assert set(rows) == {
        "all_features",
        "no_text_risk",
        NO_COST_ABLATION_SCENARIO,
        "no_model_proxy",
        "price_only",
    }

    no_cost_metrics = rows[NO_COST_ABLATION_SCENARIO][
        "deterministic_signal_evaluation_metrics"
    ]
    assert rows[NO_COST_ABLATION_SCENARIO]["effective_cost_bps"] == 0.0
    assert rows[NO_COST_ABLATION_SCENARIO]["effective_slippage_bps"] == 0.0
    assert no_cost_metrics["effective_cost_bps"] == 0.0
    assert no_cost_metrics["effective_slippage_bps"] == 0.0
    assert no_cost_metrics["transaction_cost_return"] == 0.0
    assert no_cost_metrics["slippage_cost_return"] == 0.0
    assert no_cost_metrics["total_cost_return"] == 0.0
    assert no_cost_metrics["gross_cumulative_return"] == pytest.approx(
        no_cost_metrics["cost_adjusted_cumulative_return"]
    )

    charged_scenarios = ("all_features", "no_text_risk", "no_model_proxy", "price_only")
    for scenario_id in charged_scenarios:
        row = rows[scenario_id]
        metrics = row["deterministic_signal_evaluation_metrics"]

        assert row["effective_cost_bps"] == 8.0
        assert row["effective_slippage_bps"] == 3.0
        assert row["pipeline_controls"]["cost"] is True
        assert row["pipeline_controls"]["slippage"] is True
        assert row["pipeline_controls"]["turnover"] is True
        assert metrics["effective_cost_bps"] == 8.0
        assert metrics["effective_slippage_bps"] == 3.0
        assert metrics["transaction_cost_return"] > 0
        assert metrics["slippage_cost_return"] > 0
        assert metrics["total_cost_return"] == pytest.approx(
            metrics["transaction_cost_return"] + metrics["slippage_cost_return"]
        )
        assert metrics["gross_cumulative_return"] > metrics["cost_adjusted_cumulative_return"]


def test_ablation_runs_inherit_original_period_costs_slippage_and_risk_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dates = pd.to_datetime(
        [
            "2026-01-01",
            "2026-01-01",
            "2026-01-02",
            "2026-01-02",
            "2026-01-03",
            "2026-01-03",
            "2026-01-04",
            "2026-01-04",
        ]
    )
    tickers = ["AAPL", "MSFT"] * 4
    features = pd.DataFrame(
        {
            "date": feature_dates,
            "ticker": tickers,
            "return_1": [0.01, -0.01, 0.02, -0.02, 0.01, -0.01, 0.02, -0.02],
            "liquidity_score": [20.0] * 8,
            "chronos_expected_return": [
                0.02,
                -0.01,
                0.03,
                -0.02,
                0.02,
                -0.01,
                0.03,
                -0.02,
            ],
            "granite_ttm_expected_return": [
                0.03,
                -0.02,
                0.04,
                -0.03,
                0.03,
                -0.02,
                0.04,
                -0.03,
            ],
            "news_sentiment_mean": [0.1, -0.1, 0.2, -0.2, 0.1, -0.1, 0.2, -0.2],
            "news_negative_ratio": [0.0] * 8,
            "text_risk_score": [0.0] * 8,
            "sec_risk_flag": [0.0] * 8,
            "sec_risk_flag_20d": [0.0] * 8,
            "revenue_growth": [0.2, 0.3, 0.25, 0.35, 0.2, 0.3, 0.25, 0.35],
            "forward_return_5": [0.04, -0.03, 0.05, -0.04, 0.04, -0.03, 0.05, -0.04],
        }
    )
    predictions = features.loc[
        features["date"].isin(pd.to_datetime(["2026-01-02", "2026-01-03"]))
    ].copy()
    predictions = predictions[["date", "ticker", "forward_return_5"]].assign(
        raw_expected_return=[0.04, -0.02, 0.05, -0.03],
        expected_return=[0.04, -0.02, 0.05, -0.03],
        predicted_volatility=0.01,
        downside_quantile=-0.01,
        upside_quantile=0.06,
        quantile_width=0.07,
        model_confidence=0.8,
        text_risk_score=0.0,
        sec_risk_flag=0.0,
        sec_risk_flag_20d=0.0,
        news_negative_ratio=0.0,
        liquidity_score=20.0,
    )

    def fake_walk_forward_predict(
        variant: pd.DataFrame,
        _walk_config: object,
        *,
        target: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        row_count = len(variant)
        return (
            variant[["date", "ticker", target]].assign(
                raw_expected_return=[0.04, -0.02, 0.05, -0.03] * (row_count // 4),
                expected_return=[0.04, -0.02, 0.05, -0.03] * (row_count // 4),
                predicted_volatility=0.01,
                downside_quantile=-0.01,
                upside_quantile=0.06,
                quantile_width=0.07,
                model_confidence=0.8,
                model_name="period_inheritance_fixture",
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

    config = PipelineConfig(
        cost_bps=9.0,
        slippage_bps=4.0,
        top_n=2,
        max_symbol_weight=0.4,
        max_correlation_cluster_weight=0.6,
        correlation_cluster_threshold=0.7,
        portfolio_covariance_lookback=5,
        max_daily_turnover=0.5,
        average_daily_turnover_budget=0.2,
        portfolio_volatility_limit=0.5,
        max_drawdown_stop=0.15,
        native_tabular_isolation=False,
        validity_gate_ablation_modes=(
            "all_features",
            "no-cost",
            "no-model-proxy",
            "price_only",
        ),
    )

    summary = pipeline._run_ablation_summary(predictions, features, config)

    rows = {str(row["scenario"]): row for row in summary}
    assert set(rows) == {
        "all_features",
        NO_COST_ABLATION_SCENARIO,
        "no_model_proxy",
        "price_only",
    }
    for scenario_id, row in rows.items():
        assert row["evaluation_start"] == "2026-01-02"
        assert row["evaluation_end"] == "2026-01-03"
        assert row["evaluation_observation_count"] == 2
        assert row["signal_evaluation_start"] == "2026-01-02"
        assert row["signal_evaluation_end"] == "2026-01-03"
        controls = row["inherited_backtest_controls"]
        assert controls["top_n"] == 2
        assert controls["average_daily_turnover_budget"] == 0.2
        assert controls["max_symbol_weight"] == 0.4
        assert controls["max_correlation_cluster_weight"] == 0.6
        assert controls["correlation_cluster_threshold"] == 0.7
        assert controls["portfolio_covariance_lookback"] == 5
        assert controls["max_daily_turnover"] == 0.5
        assert controls["portfolio_volatility_limit"] == 0.5
        assert controls["max_drawdown_stop"] == 0.15
        assert controls["portfolio_risk_constraints"]["max_holdings"] == 2
        assert controls["portfolio_risk_constraints"]["max_symbol_weight"] == 0.4
        assert controls["portfolio_risk_constraints"]["max_position_risk_contribution"] == 1.0
        assert controls["portfolio_risk_constraints"]["adjustment"] == {
            "volatility_scale_strength": 1.0,
            "concentration_scale_strength": 1.0,
            "risk_contribution_scale_strength": 1.0,
        }
        assert controls["realized_return_column"] == "forward_return_5"
        expected_cost_bps = 0.0 if scenario_id == NO_COST_ABLATION_SCENARIO else 9.0
        expected_slippage_bps = 0.0 if scenario_id == NO_COST_ABLATION_SCENARIO else 4.0
        assert controls["cost_bps"] == expected_cost_bps
        assert controls["slippage_bps"] == expected_slippage_bps
        assert row["effective_cost_bps"] == expected_cost_bps
        assert row["effective_slippage_bps"] == expected_slippage_bps


def test_attach_signal_features_preserves_sec_strings() -> None:
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"]),
            "ticker": ["AAPL"],
            "forward_return_1": [0.01],
            "expected_return": [0.005],
        }
    )
    features = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"]),
            "ticker": ["AAPL"],
            "sec_event_tag": ["earnings"],
            "sec_summary_ref": ["Form 8-K: quarterly growth guidance"],
            "news_top_event": ["guidance"],
            "text_risk_score": [0.3],
            "sec_event_confidence": [0.9],
        }
    )

    enriched = _attach_signal_features(predictions, features)

    assert enriched.loc[0, "sec_event_tag"] == "earnings"
    assert enriched.loc[0, "sec_summary_ref"] == "Form 8-K: quarterly growth guidance"
    assert enriched.loc[0, "news_top_event"] == "guidance"
    assert enriched.loc[0, "text_risk_score"] == 0.3
    assert enriched.loc[0, "sec_event_confidence"] == 0.9
    assert enriched.loc[0, "sec_risk_flag"] == 0.0
    assert enriched.loc[0, "forward_return_1"] == 0.01


def test_fingpt_runtime_settings_are_passed_only_for_fingpt_extractor(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeFinGPTExtractor:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def extract(self, text: str) -> dict[str, object]:
            return {
                "event_tag": "none",
                "risk_flag": False,
                "confidence": 0.5,
                "summary_ref": text,
            }

    monkeypatch.setattr(pipeline, "FinGPTEventExtractor", FakeFinGPTExtractor)

    config = PipelineConfig(
        filing_extractor_model="fingpt",
        enable_local_filing_llm=True,
        finma_model_id="ignored",
        fingpt_model_id="finma/fake-fingpt",
        fingpt_base_model_id="meta-llama/Meta-Llama-3-8B",
        fingpt_runtime="llama-cpp",
        fingpt_quantized_model_path="artifacts/model_cache/fingpt-test.gguf",
        fingpt_allow_unquantized_transformers=True,
        fingpt_single_load_lock_path="artifacts/model_locks/fingpt.lock",
    )
    extractor = pipeline._filing_extractor(config)

    assert isinstance(extractor, FakeFinGPTExtractor)
    runtime = captured.get("runtime")
    assert runtime is None or runtime == "llama-cpp"
    runtime_model_path = captured.get("runtime_model_path", captured.get("quantized_model_path"))
    assert runtime_model_path == config.fingpt_quantized_model_path
    single_lock_path = captured.get("single_load_lock_path", captured.get("single_model_load_lock_path"))
    assert single_lock_path == "artifacts/model_locks/fingpt.lock"
    unquantized_transformers = captured.get("allow_unquantized_transformers", False)
    unquantized_fingpt = captured.get("allow_unquantized_fingpt", False)
    assert (
        unquantized_transformers == config.fingpt_allow_unquantized_transformers
        or unquantized_fingpt == config.fingpt_allow_unquantized_transformers
    )


def test_walk_forward_config_carries_native_runtime_guards() -> None:
    config = PipelineConfig(
        train_periods=45,
        test_periods=9,
        gap_periods=2,
        embargo_periods=3,
        model_name="lightgbm",
        native_tabular_isolation=False,
        native_model_timeout_seconds=11,
        tabular_num_threads=3,
    )

    walk_config = pipeline._walk_forward_config(config)

    assert walk_config.train_periods == 45
    assert walk_config.test_periods == 9
    assert walk_config.gap_periods == 20
    assert walk_config.embargo_periods == 20
    assert walk_config.model_name == "lightgbm"
    assert walk_config.native_tabular_isolation is False
    assert walk_config.native_model_timeout_seconds == 11
    assert walk_config.tabular_num_threads == 3


def test_pipeline_walk_forward_wrapper_passes_splitter_to_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_walk_forward_predict(
        variant: pd.DataFrame,
        walk_config: object,
        *,
        target: str,
        splitter: object,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        captured["walk_config"] = walk_config
        captured["target"] = target
        captured["splitter"] = splitter
        return pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(pipeline, "walk_forward_predict", fake_walk_forward_predict)
    config = PipelineConfig(
        train_periods=45,
        test_periods=9,
        gap_periods=3,
        embargo_periods=3,
        prediction_target_column="forward_return_20",
        native_tabular_isolation=False,
    )
    splitter = pipeline._walk_forward_splitter(config, target_column="forward_return_20")

    pipeline._predict_walk_forward_with_splitter(
        pd.DataFrame({"date": []}),
        pipeline._walk_forward_config(config),
        target="forward_return_20",
        splitter=splitter,
    )

    assert captured["target"] == "forward_return_20"
    assert captured["splitter"] is splitter
    assert splitter.config.train_periods == 45
    assert splitter.config.test_periods == 9
    assert splitter.config.purge_periods == 20
    assert splitter.config.embargo_periods == 20
    assert splitter.config.target_column == "forward_return_20"
