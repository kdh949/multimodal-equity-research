from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research.validation import (
    DEFAULT_MODEL_COMPARISON_METRICS,
    DETERMINISTIC_SIGNAL_ENGINE_ID,
    EQUAL_WEIGHT_BASELINE_TYPE,
    MARKET_BENCHMARK_BASELINE_TYPE,
    REPORT_COMPARISON_TABLE_COLUMNS,
    REPORT_COMPARISON_TABLE_SCHEMA_VERSION,
    STAGE1_COMPARISON_SCHEMA_VERSION,
    ComparisonValidationWindow,
    ValidationGateThresholds,
    build_report_comparison_table,
    build_validity_gate_report,
    default_stage1_comparison_input_schema,
    write_validity_gate_artifacts,
)


def test_default_stage1_comparison_input_schema_covers_required_contract() -> None:
    schema = default_stage1_comparison_input_schema()
    payload = schema.to_dict()

    assert payload["schema_version"] == STAGE1_COMPARISON_SCHEMA_VERSION
    assert payload["full_model"]["entity_id"] == "all_features"
    assert payload["full_model"]["role"] == "full_model"
    assert payload["full_model"]["signal_engine"] == DETERMINISTIC_SIGNAL_ENGINE_ID
    assert payload["full_model"]["model_predictions_are_order_signals"] is False
    assert payload["full_model"]["llm_makes_trading_decisions"] is False
    assert payload["full_model"]["requires_heavy_model"] is False
    assert {"chronos", "granite_ttm", "finbert", "finma", "fingpt", "ollama"}.issubset(
        set(payload["full_model"]["optional_adapters"])
    )

    baselines = {row["role"]: [] for row in payload["baselines"]}
    for row in payload["baselines"]:
        baselines[row["role"]].append(row)
    assert baselines["model_baseline"][0]["entity_id"] == "no_model_proxy"
    assert {
        row["baseline_type"]
        for row in baselines["return_baseline"]
    } == {MARKET_BENCHMARK_BASELINE_TYPE, EQUAL_WEIGHT_BASELINE_TYPE}

    ablation_ids = {row["entity_id"] for row in payload["ablations"]}
    assert {"price_only", "text_only", "sec_only", "no_costs"}.issubset(ablation_ids)
    assert {"chronos_model", "granite_ttm_model", "finbert_model", "ollama_model"}.issubset(
        ablation_ids
    )

    metric_rows = {row["metric_id"]: row for row in payload["metrics"]}
    assert tuple(metric_rows) == DEFAULT_MODEL_COMPARISON_METRICS
    assert metric_rows["turnover"]["direction"] == "lower_is_better"
    assert metric_rows["sharpe"]["direction"] == "higher_is_better"

    windows = {row["window_id"]: row for row in payload["validation_windows"]}
    assert set(windows) == {
        "configured_walk_forward",
        "configured_oos_holdout",
        "strategy_evaluation",
    }
    assert windows["configured_walk_forward"]["target_column"] == "forward_return_20"
    assert windows["configured_walk_forward"]["gap_periods"] == 60
    assert windows["configured_walk_forward"]["embargo_periods"] == 60
    assert windows["configured_oos_holdout"]["is_oos"] is True
    assert "t_plus_1" in windows["strategy_evaluation"]["return_timing"]


def test_comparison_validation_window_rejects_leaky_or_unsafe_timing() -> None:
    with pytest.raises(ValueError, match="training window must end before"):
        ComparisonValidationWindow(
            window_id="bad_fold",
            label="Bad fold",
            role="walk_forward_fold",
            target_column="forward_return_5",
            target_horizon=5,
            gap_periods=5,
            embargo_periods=5,
            train_end="2026-01-10",
            test_start="2026-01-10",
        )

    with pytest.raises(ValueError, match="gap_periods"):
        ComparisonValidationWindow(
            window_id="short_gap",
            label="Short gap",
            role="walk_forward_fold",
            target_column="forward_return_5",
            target_horizon=5,
            gap_periods=1,
            embargo_periods=5,
        )

    with pytest.raises(ValueError, match="t\\+1"):
        ComparisonValidationWindow(
            window_id="unsafe_timing",
            label="Unsafe timing",
            role="strategy_evaluation",
            target_column="forward_return_5",
            target_horizon=5,
            gap_periods=5,
            embargo_periods=5,
            return_timing="same_period_return_application",
        )


def test_validity_gate_artifacts_serialize_comparison_input_and_result_schemas(
    tmp_path,
) -> None:
    report = build_validity_gate_report(
        _schema_predictions(),
        _schema_validation_summary(),
        _schema_equity_curve(),
        SimpleNamespace(cagr=1.0, sharpe=1.1, max_drawdown=-0.03, turnover=0.10),
        ablation_summary=_schema_ablation_summary(),
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_5",
            required_validation_horizon=5,
            benchmark_ticker="SPY",
            gap_periods=5,
            embargo_periods=5,
            cost_bps=5.0,
            slippage_bps=2.0,
        ),
        thresholds=ValidationGateThresholds(min_folds=2),
    )

    payload = report.to_dict()
    input_schema = payload["comparison_input_schema"]
    result_schema = payload["comparison_result_schema"]

    assert payload["stage1_comparison_input_schema"] == input_schema
    assert payload["stage1_comparison_result_schema"] == result_schema
    assert payload["metrics"]["comparison_input_schema"] == input_schema
    assert payload["evidence"]["comparison_result_schema"] == result_schema

    assert input_schema["full_model"]["entity_id"] == "all_features"
    assert {row["role"] for row in input_schema["baselines"]} == {
        "model_baseline",
        "return_baseline",
    }
    assert {row["metric_id"] for row in input_schema["metrics"]} == set(
        DEFAULT_MODEL_COMPARISON_METRICS
    )

    assert result_schema["full_model_result"]["entity_id"] == "all_features"
    assert {
        row["entity_id"]
        for row in result_schema["baseline_results"]
    } >= {"no_model_proxy", "return_baseline_spy", "return_baseline_equal_weight"}
    assert {row["metric"] for row in result_schema["metric_results"]} == set(
        DEFAULT_MODEL_COMPARISON_METRICS
    )
    windows = {row["window_id"]: row for row in result_schema["validation_windows"]}
    assert windows["fold_0"]["role"] == "walk_forward_fold"
    assert windows["fold_1"]["role"] == "oos_holdout"
    assert windows["fold_0"]["train_end"] < windows["fold_0"]["test_start"]

    json_path, markdown_path = write_validity_gate_artifacts(report, tmp_path)
    artifact_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert artifact_payload["comparison_input_schema"] == input_schema
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## Stage 1 Comparison Input Schema" in markdown
    assert "## Stage 1 Comparison Result Schema" in markdown
    assert "<h2>Stage 1 Comparison Input Schema</h2>" in report.to_html()
    assert "<h2>Stage 1 Comparison Result Schema</h2>" in report.to_html()


def test_report_comparison_table_normalizes_strategy_equal_weight_and_proxy() -> None:
    table = build_report_comparison_table(
        {
            "entity_id": "all_features",
            "label": "Original strategy",
            "validation_status": "pass",
            "cost_adjusted_cumulative_return": 0.18,
            "excess_return": 0.07,
            "sharpe": 1.2,
            "max_drawdown": -0.08,
            "average_daily_turnover": 0.12,
            "total_cost_return": 0.01,
            "mean_rank_ic": 0.04,
            "positive_fold_ratio": 0.75,
            "oos_rank_ic": 0.03,
            "return_basis": "cost_adjusted_return",
            "return_column": "forward_return_20",
            "return_horizon": 20,
            "evaluation_start": "2025-01-02",
            "evaluation_end": "2025-12-31",
            "evaluation_observations": 240,
            "cost_bps": 5,
            "slippage_bps": 2,
        },
        {
            "name": "equal_weight",
            "label": "Equal-weight universe baseline",
            "sample_alignment_status": "pass",
            "baseline_type": "equal_weight_universe",
            "cost_adjusted_cumulative_return": 0.11,
            "excess_return": 0.0,
            "sharpe": 0.8,
            "max_drawdown": -0.10,
            "average_daily_turnover": 0.02,
            "total_cost_return": 0.002,
            "return_basis": "cost_adjusted_return",
            "evaluation_start": pd.Timestamp("2025-01-02"),
            "evaluation_end": pd.Timestamp("2025-12-31"),
            "evaluation_observations": 240,
        },
        {
            "scenario": "no_model_proxy",
            "label": "No-model proxy",
            "validation_status": "warning",
            "validation_mean_information_coefficient": 0.025,
            "validation_positive_ic_fold_ratio": 0.6,
            "validation_oos_information_coefficient": 0.02,
            "signal_cost_adjusted_cumulative_return": 0.14,
            "signal_average_daily_turnover": 0.09,
            "deterministic_signal_evaluation_metrics": {
                "return_basis": "cost_adjusted_return",
                "excess_return": 0.03,
                "total_cost_return": 0.008,
            },
        },
    )

    assert tuple(table.columns) == REPORT_COMPARISON_TABLE_COLUMNS
    assert set(table["schema_version"]) == {REPORT_COMPARISON_TABLE_SCHEMA_VERSION}
    assert len(table) == 3 * 9

    entity_roles = table.drop_duplicates("entity_id").set_index("entity_id")["role"].to_dict()
    assert entity_roles == {
        "all_features": "full_model",
        "return_baseline_equal_weight": "return_baseline",
        "no_model_proxy": "model_baseline",
    }

    values = table.set_index(["entity_id", "metric"])["value"].to_dict()
    assert values[("all_features", "cost_adjusted_cumulative_return")] == 0.18
    assert values[("return_baseline_equal_weight", "average_daily_turnover")] == 0.02
    assert values[("no_model_proxy", "mean_rank_ic")] == 0.025
    assert values[("no_model_proxy", "excess_return")] == 0.03

    proxy_rows = table[table["entity_id"] == "no_model_proxy"]
    assert set(proxy_rows["ablation_scenario_id"]) == {"no_model_proxy"}
    assert set(proxy_rows["status"]) == {"warning"}
    assert set(proxy_rows["target_column"]) == {"forward_return_20"}
    assert set(proxy_rows["target_horizon"]) == {20}


def test_report_comparison_table_rejects_invalid_metric_ids() -> None:
    with pytest.raises(ValueError, match="stable snake_case"):
        build_report_comparison_table(
            {"entity_id": "all_features"},
            {"name": "equal_weight"},
            {"scenario": "no_model_proxy"},
            metrics=("bad metric",),
        )


def _schema_predictions() -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=6, freq="B")
    rows: list[dict[str, object]] = []
    for fold, date in enumerate(dates):
        for ticker, expected, realized in (
            ("AAPL", 0.03, 0.05),
            ("MSFT", 0.02, 0.03),
            ("SPY", 0.01, 0.01),
        ):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "expected_return": expected,
                    "forward_return_5": realized,
                    "fold": min(fold // 3, 1),
                    "is_oos": fold >= 3,
                }
            )
    return pd.DataFrame(rows)


def _schema_validation_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": [0, 1],
            "train_start": pd.to_datetime(["2025-09-01", "2025-10-01"]),
            "train_end": pd.to_datetime(["2025-12-19", "2026-01-02"]),
            "test_start": pd.to_datetime(["2025-12-29", "2026-01-12"]),
            "test_end": pd.to_datetime(["2026-01-02", "2026-01-16"]),
            "train_observations": [120, 120],
            "test_observations": [9, 9],
            "labeled_test_observations": [9, 9],
            "prediction_count": [9, 9],
            "is_oos": [False, True],
            "validation_status": ["pass", "pass"],
            "fold_type": ["validation", "oos"],
            "model_name": ["hist_gradient_boosting", "hist_gradient_boosting"],
        }
    )


def _schema_equity_curve() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-02", periods=6, freq="B"),
            "portfolio_return": [0.015] * 6,
            "gross_return": [0.016] * 6,
            "cost_adjusted_return": [0.015] * 6,
            "benchmark_return": [0.004] * 6,
            "turnover": [0.10] * 6,
            "transaction_cost_return": [0.0007] * 6,
            "slippage_cost_return": [0.0003] * 6,
            "total_cost_return": [0.0010] * 6,
        }
    )


def _schema_ablation_summary() -> list[dict[str, object]]:
    return [
        _schema_ablation_row("all_features", "signal", 1.20, 0.20, 0.12, 0.10),
        _schema_ablation_row("no_model_proxy", "pipeline_control", 0.80, 0.12, 0.08, 0.08),
        _schema_ablation_row("price_only", "data_channel", 0.70, 0.10, 0.06, 0.07),
        _schema_ablation_row("text_only", "data_channel", 0.40, 0.06, 0.04, 0.05),
        _schema_ablation_row("sec_only", "data_channel", 0.30, 0.05, 0.03, 0.04),
        _schema_ablation_row("no_costs", "cost", 1.30, 0.22, 0.14, 0.10),
    ]


def _schema_ablation_row(
    scenario: str,
    kind: str,
    sharpe: float,
    excess_return: float,
    cost_adjusted_return: float,
    turnover: float,
) -> dict[str, object]:
    return {
        "scenario": scenario,
        "kind": kind,
        "label": scenario.replace("_", " ").title(),
        "sharpe": sharpe,
        "max_drawdown": -0.03,
        "excess_return": excess_return,
        "turnover": turnover,
        "validation_status": "pass",
        "validation_fold_count": 2,
        "validation_oos_fold_count": 1,
        "validation_mean_information_coefficient": 0.05,
        "validation_positive_ic_fold_ratio": 1.0,
        "validation_oos_information_coefficient": 0.04,
        "signal_average_daily_turnover": turnover,
        "signal_cost_adjusted_cumulative_return": cost_adjusted_return,
        "deterministic_signal_evaluation_metrics": {
            "return_basis": "cost_adjusted_return",
            "cost_adjusted_cumulative_return": cost_adjusted_return,
            "average_daily_turnover": turnover,
            "total_cost_return": 0.01,
            "excess_return": excess_return,
            "action_counts": {"BUY": 2, "SELL": 0, "HOLD": 4},
        },
        "pipeline_controls": {
            "model_proxy": scenario != "no_model_proxy",
            "cost": scenario != "no_costs",
            "slippage": scenario != "no_costs",
            "turnover": scenario != "no_costs",
        },
    }
