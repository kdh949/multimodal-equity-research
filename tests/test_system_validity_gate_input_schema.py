from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research.validation import (
    SYSTEM_VALIDITY_GATE_INPUT_SCHEMA_VERSION,
    SYSTEM_VALIDITY_GATE_REQUIRED_INPUT_SECTIONS,
    SYSTEM_VALIDITY_GATE_SCHEMA_ID,
    SystemValidityGateInputSchema,
    build_system_validity_gate_input_schema,
    build_validity_gate_report,
    default_system_validity_gate_input_schema,
)


def test_default_system_validity_gate_input_schema_covers_stage1_contract() -> None:
    schema = default_system_validity_gate_input_schema()
    payload = schema.to_dict()

    assert payload["schema_id"] == SYSTEM_VALIDITY_GATE_SCHEMA_ID
    assert payload["schema_version"] == SYSTEM_VALIDITY_GATE_INPUT_SCHEMA_VERSION
    assert payload["required_sections"] == list(SYSTEM_VALIDITY_GATE_REQUIRED_INPUT_SECTIONS)
    assert payload["experiment"]["target_horizon"] == "forward_return_20"
    assert payload["experiment"]["diagnostic_horizons"] == [
        "forward_return_1",
        "forward_return_5",
    ]
    assert payload["experiment"]["model_predictions_are_order_signals"] is False
    assert payload["experiment"]["llm_makes_trading_decisions"] is False
    assert payload["universe_snapshot"]["selection_count"] == 150
    assert payload["universe_snapshot"]["fixed_at_experiment_start"] is True
    assert payload["universe_snapshot"]["survivorship_bias_allowed_v1"] is True
    assert payload["walk_forward_config"]["target_column"] == "forward_return_20"
    assert payload["walk_forward_config"]["train_periods"] == 252
    assert payload["walk_forward_config"]["test_periods"] == 60
    assert payload["walk_forward_config"]["embargo_periods"] == 20
    assert payload["walk_forward_config"]["embargo_zero_for_forward_return_20_is_hard_fail"]
    assert payload["portfolio_constraints"]["long_only"] is True
    assert payload["portfolio_constraints"]["max_holdings"] == 20
    assert payload["portfolio_constraints"]["max_symbol_weight"] == 0.10
    assert payload["portfolio_constraints"]["max_sector_weight"] == 0.30
    assert payload["portfolio_constraints"]["correlation_cluster_weight_scope"] == "v1_excluded"
    assert payload["transaction_costs"]["cost_bps"] == 5.0
    assert payload["transaction_costs"]["slippage_bps"] == 2.0
    assert payload["transaction_costs"]["average_daily_turnover_budget"] == 0.25
    assert payload["benchmark_config"]["benchmark_ticker"] == "SPY"
    assert payload["benchmark_config"]["required_baselines"] == [
        "SPY",
        "equal_weight_universe",
    ]
    assert payload["comparison_inputs"]["proxy_ic_improvement_min"] == 0.01
    assert payload["comparison_inputs"]["positive_fold_ratio_min"] == 0.65
    assert payload["scope_bounds"]["real_trading_orders"] == "excluded"
    assert payload["scope_bounds"]["point_in_time_universe"] == "v2"
    assert payload["scope_bounds"]["top_decile_20d_excess_return"] == "report_only"


def test_system_validity_gate_input_schema_defines_required_frames() -> None:
    payload = build_system_validity_gate_input_schema()

    assert payload["predictions"]["sample_alignment_key"] == ["date", "ticker"]
    assert payload["predictions"]["required_columns"] == [
        "date",
        "ticker",
        "fold",
        "is_oos",
        "expected_return",
        "forward_return_20",
    ]
    assert "forward_return_1" in payload["predictions"]["optional_columns"]
    assert "forward_return_5" in payload["predictions"]["optional_columns"]
    assert payload["validation_summary"]["required_columns"] == [
        "fold",
        "train_end",
        "test_start",
        "is_oos",
        "labeled_test_observations",
        "train_observations",
    ]
    assert "embargo_periods" in payload["validation_summary"]["optional_columns"]
    assert payload["equity_curve"]["required_columns"] == [
        "date",
        "portfolio_return",
        "cost_adjusted_return",
        "benchmark_return",
        "turnover",
    ]
    assert payload["backtest_results"]["required_columns"] == [
        "date",
        "portfolio_return",
        "cost_adjusted_return",
        "benchmark_return",
        "equal_weight_return",
        "turnover",
        "holdings_count",
        "max_symbol_weight",
        "max_sector_weight",
    ]
    assert payload["walk_forward_results"]["required_columns"] == [
        "fold",
        "train_end",
        "test_start",
        "is_oos",
        "target_column",
        "prediction_horizon_periods",
        "purge_periods",
        "embargo_periods",
        "labeled_test_observations",
        "train_observations",
    ]
    assert payload["out_of_sample_results"]["required_columns"] == [
        "fold",
        "is_oos",
        "fold_rank_ic",
        "positive_rank_ic",
        "cost_adjusted_excess_return_vs_spy",
        "cost_adjusted_excess_return_vs_equal_weight",
        "max_drawdown",
        "average_daily_turnover",
    ]
    assert payload["risk_rule_results"]["required_columns"] == [
        "rule_id",
        "rule_group",
        "status",
        "passed",
        "metric",
        "value",
        "threshold",
        "operator",
    ]
    assert payload["strategy_metrics"]["required_attributes"] == [
        "cagr",
        "sharpe",
        "max_drawdown",
        "turnover",
    ]
    assert payload["feature_availability_cutoff"]["fail_on_future_availability"] is True


def test_system_validity_gate_input_schema_rejects_invalid_canonical_values() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        SystemValidityGateInputSchema(schema_version="system_validity_gate_input.v0")

    with pytest.raises(ValueError, match="target_column"):
        SystemValidityGateInputSchema(walk_forward_config={"target_column": "forward_return_5"})

    with pytest.raises(ValueError, match="embargo_periods"):
        SystemValidityGateInputSchema(
            walk_forward_config={
                "target_column": "forward_return_20",
                "embargo_periods": 0,
            }
        )


def test_validity_gate_report_serializes_input_schema() -> None:
    report = build_validity_gate_report(
        _predictions(),
        _validation_summary(),
        _equity_curve(),
        SimpleNamespace(cagr=0.25, sharpe=1.0, max_drawdown=-0.05, turnover=0.10),
        ablation_summary=_ablation_summary(),
        config=SimpleNamespace(gap_periods=20, embargo_periods=20),
    )

    payload = report.to_dict()
    schema = payload["system_validity_gate_input_schema"]
    assert schema["schema_version"] == SYSTEM_VALIDITY_GATE_INPUT_SCHEMA_VERSION
    assert payload["metrics"]["system_validity_gate_input_schema"] == schema
    assert payload["evidence"]["system_validity_gate_input_schema"] == schema


def _predictions() -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=21, freq="B")
    rows = []
    for fold, date in enumerate(dates):
        for ticker, value in zip(("AAPL", "MSFT", "SPY"), (0.03, 0.02, 0.01), strict=True):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "fold": fold,
                    "is_oos": fold >= 19,
                    "expected_return": value,
                    "forward_return_1": value,
                    "forward_return_5": value,
                    "forward_return_20": value,
                }
            )
    return pd.DataFrame(rows)


def _validation_summary() -> pd.DataFrame:
    test_starts = pd.date_range("2026-02-02", periods=21, freq="B")
    return pd.DataFrame(
        {
            "fold": range(21),
            "train_end": test_starts - pd.Timedelta(days=30),
            "test_start": test_starts,
            "is_oos": [False] * 19 + [True, True],
            "labeled_test_observations": [3] * 21,
            "train_observations": [252] * 21,
            "target_column": ["forward_return_20"] * 21,
            "prediction_horizon_periods": [20] * 21,
            "gap_periods": [20] * 21,
            "purge_periods": [20] * 21,
            "purged_date_count": [20] * 21,
            "purge_applied": [True] * 21,
            "embargo_periods": [20] * 21,
            "embargoed_date_count": [20] * 21,
            "embargo_applied": [True] * 21,
        }
    )


def _equity_curve() -> pd.DataFrame:
    dates = pd.date_range("2026-03-02", periods=21, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": [0.001] * len(dates),
            "cost_adjusted_return": [0.001] * len(dates),
            "benchmark_return": [0.0] * len(dates),
            "turnover": [0.10] * len(dates),
        }
    )


def _ablation_summary() -> list[dict[str, object]]:
    return [
        {"scenario": "all_features", "sharpe": 1.0, "excess_return": 0.10},
        {"scenario": "price_only", "sharpe": 0.4, "excess_return": 0.04},
        {"scenario": "text_only", "sharpe": 0.3, "excess_return": 0.03},
        {"scenario": "sec_only", "sharpe": 0.2, "excess_return": 0.02},
        {"scenario": "no_model_proxy", "sharpe": 0.5, "excess_return": 0.05},
        {"scenario": "no_costs", "sharpe": 0.6, "excess_return": 0.06},
    ]
