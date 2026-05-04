from __future__ import annotations

import pandas as pd

from quant_research.pipeline import PipelineConfig, _attach_signal_features, run_research_pipeline


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
    assert "is_oos" in result.validation_summary
    assert result.validation_summary["is_oos"].any()
    assert {
        "all_features",
        "no_text_risk",
        "no_sec_risk",
        "no_costs",
        "full_model_features",
        "no_chronos_features",
        "no_granite_features",
        "tabular_without_ts_proxies",
    } == {
        row["scenario"] for row in result.ablation_summary
    }


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
