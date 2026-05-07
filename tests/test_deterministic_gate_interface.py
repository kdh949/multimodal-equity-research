from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research.validation import (
    DETERMINISTIC_GATE_INTERFACE_ID,
    DETERMINISTIC_GATE_INTERFACE_SCHEMA_VERSION,
    BuildValidityGateReportProviderFree,
    DeterministicGateBacktestInputs,
    DeterministicGateCostInputs,
    DeterministicGateInputs,
    DeterministicGateRiskRuleInputs,
    DeterministicGateWalkForwardInputs,
    ProviderFreeDeterministicGate,
    aggregate_deterministic_gate_results,
    aggregate_deterministic_validity_gate,
    evaluate_provider_free_deterministic_gate,
)


def test_deterministic_gate_input_model_documents_provider_free_contract() -> None:
    inputs = _gate_inputs()
    payload = inputs.to_dict()

    assert payload["schema_id"] == DETERMINISTIC_GATE_INTERFACE_ID
    assert payload["schema_version"] == DETERMINISTIC_GATE_INTERFACE_SCHEMA_VERSION
    assert payload["provider_free"] is True
    assert payload["decision_engine"] == "deterministic"
    assert payload["llm_makes_trading_decisions"] is False
    assert payload["model_predictions_are_order_signals"] is False
    assert payload["target_horizon"] == "forward_return_20"
    assert {
        "backtest",
        "transaction_costs",
        "slippage",
        "turnover",
        "walk_forward",
        "out_of_sample",
        "risk_rules",
    }.issubset(set(payload["input_sections"]))
    assert payload["costs"]["cost_bps"] == 5.0
    assert payload["costs"]["slippage_bps"] == 2.0
    assert payload["costs"]["average_daily_turnover_budget"] == 0.25
    assert payload["walk_forward"]["embargo_periods"] == 20
    assert payload["walk_forward"]["oos_fold_count"] == 2
    assert payload["risk"]["portfolio_constraints"]["max_holdings"] == 20
    assert payload["risk"]["portfolio_constraints"]["max_symbol_weight"] == 0.10
    assert payload["risk"]["portfolio_constraints"]["max_sector_weight"] == 0.30

    kwargs = DeterministicGateInputs(
        predictions=_predictions(),
        backtest=inputs.backtest,
        walk_forward=inputs.walk_forward,
        costs=inputs.costs,
        ablation_summary=inputs.ablation_summary,
    ).to_report_kwargs()
    assert kwargs["config"].cost_bps == 5.0
    assert kwargs["config"].slippage_bps == 2.0
    assert kwargs["config"].embargo_periods == 20


def test_provider_free_interface_delegates_to_existing_deterministic_gate() -> None:
    evaluator = BuildValidityGateReportProviderFree()
    assert isinstance(evaluator, ProviderFreeDeterministicGate)

    report = evaluate_provider_free_deterministic_gate(_gate_inputs(), evaluator=evaluator)

    assert report.required_validation_horizon == "20d"
    assert report.metrics["target_column"] == "forward_return_20"
    assert report.metrics["strategy_turnover"] == pytest.approx(0.10)
    assert report.metrics["strategy_slippage_cost_return"] == pytest.approx(0.0)
    assert "deterministic_strategy_validity" in report.gate_results


def test_deterministic_gate_aggregation_returns_pass_warning_or_fail() -> None:
    passing = aggregate_deterministic_gate_results(
        {
            "leakage": {"status": "pass"},
            "cost_adjusted_performance": {"status": "pass"},
        }
    )
    assert passing["final_decision"] == "PASS"
    assert passing["final_status"] == "pass"

    warning = aggregate_deterministic_gate_results(
        {
            "leakage": {"status": "pass"},
            "model_value": {"status": "warning", "reason": "proxy-like model value"},
        }
    )
    assert warning["final_decision"] == "WARN"
    assert warning["final_status"] == "warning"
    assert warning["warning_items"] == ["model_value"]

    failed = aggregate_deterministic_validity_gate(
        {
            "leakage": {
                "status": "hard_fail",
                "reason": "embargo_periods=0",
                "affects_system": True,
            },
            "rank_ic": {"status": "pass"},
        }
    )
    payload = failed.to_dict()
    assert payload["system_validity_status"] == "hard_fail"
    assert payload["final_strategy_status"] == "not_evaluable"
    assert payload["hard_fail_rules"] == ["leakage"]


def test_deterministic_gate_input_model_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="cost_bps"):
        DeterministicGateCostInputs(cost_bps=-1.0)

    with pytest.raises(ValueError, match="embargo_periods"):
        DeterministicGateWalkForwardInputs(
            validation_summary=_validation_summary(),
            embargo_periods=0,
        )

    with pytest.raises(ValueError, match="max_holdings"):
        DeterministicGateRiskRuleInputs(
            portfolio_constraints={
                "long_only": True,
                "max_holdings": 21,
                "max_symbol_weight": 0.10,
                "max_sector_weight": 0.30,
            }
        )

    with pytest.raises(ValueError, match="predictions"):
        DeterministicGateInputs(
            predictions=_predictions().drop(columns=["expected_return"]),
            backtest=DeterministicGateBacktestInputs(
                equity_curve=_equity_curve(),
                strategy_metrics=_strategy_metrics(),
            ),
            walk_forward=DeterministicGateWalkForwardInputs(
                validation_summary=_validation_summary(),
            ),
        )


def _gate_inputs() -> DeterministicGateInputs:
    return DeterministicGateInputs(
        predictions=_predictions(),
        backtest=DeterministicGateBacktestInputs(
            equity_curve=_equity_curve(),
            strategy_metrics=_strategy_metrics(),
        ),
        walk_forward=DeterministicGateWalkForwardInputs(
            validation_summary=_validation_summary(),
            walk_forward_config=SimpleNamespace(gap_periods=20, embargo_periods=20),
        ),
        costs=DeterministicGateCostInputs(cost_bps=5.0, slippage_bps=2.0),
        ablation_summary=_ablation_summary(),
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_20",
            benchmark_ticker="SPY",
            gap_periods=20,
            embargo_periods=20,
            cost_bps=5.0,
            slippage_bps=2.0,
        ),
    )


def _strategy_metrics() -> SimpleNamespace:
    return SimpleNamespace(cagr=0.25, sharpe=1.0, max_drawdown=-0.05, turnover=0.10)


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
            "gross_return": [0.001] * len(dates),
            "cost_adjusted_return": [0.001] * len(dates),
            "benchmark_return": [0.0] * len(dates),
            "turnover": [0.10] * len(dates),
            "transaction_cost_return": [0.0] * len(dates),
            "slippage_cost_return": [0.0] * len(dates),
            "total_cost_return": [0.0] * len(dates),
        }
    )


def _ablation_summary() -> list[dict[str, object]]:
    return [
        {"scenario": "all_features", "sharpe": 1.0, "excess_return": 0.10, "rank_ic": 0.04},
        {"scenario": "price_only", "sharpe": 0.4, "excess_return": 0.04},
        {"scenario": "text_only", "sharpe": 0.3, "excess_return": 0.03},
        {"scenario": "sec_only", "sharpe": 0.2, "excess_return": 0.02},
        {
            "scenario": "no_model_proxy",
            "sharpe": 0.5,
            "excess_return": 0.05,
            "rank_ic": 0.02,
        },
        {"scenario": "no_costs", "sharpe": 0.6, "excess_return": 0.06},
    ]
