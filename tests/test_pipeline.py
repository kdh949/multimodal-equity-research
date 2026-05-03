from __future__ import annotations

from quant_research.pipeline import PipelineConfig, run_research_pipeline


def test_synthetic_pipeline_runs_end_to_end() -> None:
    result = run_research_pipeline(
        PipelineConfig(
            tickers=["SPY", "AAPL", "MSFT"],
            data_mode="synthetic",
            train_periods=60,
            test_periods=15,
            top_n=2,
        )
    )

    assert not result.features.empty
    assert not result.predictions.empty
    assert not result.signals.empty
    assert not result.backtest.equity_curve.empty
    assert {"all_features", "no_text_risk", "no_sec_risk", "no_costs"} == {
        row["scenario"] for row in result.ablation_summary
    }
