from __future__ import annotations

import pandas as pd
import pytest

import quant_research.pipeline as pipeline
from quant_research.pipeline import PipelineConfig, _attach_signal_features, run_research_pipeline


def test_synthetic_pipeline_runs_end_to_end() -> None:
    result = run_research_pipeline(
        PipelineConfig(
            tickers=["SPY", "AAPL", "MSFT"],
            data_mode="synthetic",
            train_periods=120,
            test_periods=40,
            top_n=2,
            model_name="hist_gradient",
            native_tabular_isolation=False,
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
    assert "is_oos" in result.validation_summary
    assert result.validation_summary["is_oos"].any()
    assert {
        "all_features",
        "no_text_risk",
        "no_sec_risk",
        "no_costs",
        "full_model_features",
        "price_only",
        "text_only",
        "sec_only",
        "no_chronos_features",
        "no_granite_features",
        "no_model_proxy",
        "tabular_without_ts_proxies",
    } == {
        row["scenario"] for row in result.ablation_summary
    }
    assert result.validity_report is not None
    assert result.validity_report.required_validation_horizon == "5d"
    assert result.validity_report.system_validity_status in {"pass", "hard_fail", "not_evaluable"}


def test_synthetic_pipeline_features_include_recency_coverage_news_columns() -> None:
    result = run_research_pipeline(
        PipelineConfig(
            tickers=["AAPL", "MSFT"],
            data_mode="synthetic",
            train_periods=60,
            test_periods=15,
            top_n=2,
            model_name="hist_gradient",
            native_tabular_isolation=False,
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
        prediction_target_column="forward_return_1",
        model_name="lightgbm",
        native_tabular_isolation=False,
        native_model_timeout_seconds=11,
        tabular_num_threads=3,
    )

    walk_config = pipeline._walk_forward_config(config)

    assert walk_config.train_periods == 45
    assert walk_config.test_periods == 9
    assert walk_config.gap_periods == 2
    assert walk_config.embargo_periods == 3
    assert walk_config.model_name == "lightgbm"
    assert walk_config.native_tabular_isolation is False
    assert walk_config.native_model_timeout_seconds == 11
    assert walk_config.tabular_num_threads == 3
