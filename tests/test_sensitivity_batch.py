from __future__ import annotations

import pandas as pd
import pytest

from quant_research.validation import (
    TRANSACTION_COST_SENSITIVITY_BATCH_COLUMNS,
    TRANSACTION_COST_SENSITIVITY_BATCH_SCHEMA_VERSION,
    TRANSACTION_COST_SENSITIVITY_SUMMARY_SCHEMA_VERSION,
    TransactionCostSensitivityConfig,
    TransactionCostSensitivityScenario,
    build_transaction_cost_sensitivity_batch_table,
    build_transaction_cost_sensitivity_summary_metrics,
    default_transaction_cost_sensitivity_config,
    run_transaction_cost_sensitivity_batch,
)


def test_transaction_cost_sensitivity_batch_reprices_all_configured_scenarios() -> None:
    config = _two_scenario_config()
    result = run_transaction_cost_sensitivity_batch(
        _equity_curve(turnover=[0.3, 0.3, 0.3]),
        sensitivity_config=config,
        batch_id="unit_batch",
    )

    rows = result.summary.set_index("scenario_id")

    assert result.execution_mode == "reprice"
    assert result.batch_id == "unit_batch"
    assert tuple(result.summary.columns) == TRANSACTION_COST_SENSITIVITY_BATCH_COLUMNS
    assert result.summary["scenario_id"].tolist() == ["canonical_costs", "high_costs"]
    assert set(result.equity_curves) == {"canonical_costs", "high_costs"}
    assert rows.loc["canonical_costs", "schema_version"] == (
        TRANSACTION_COST_SENSITIVITY_BATCH_SCHEMA_VERSION
    )
    assert rows.loc["canonical_costs", "status"] == "warning"
    assert rows.loc["high_costs", "status"] == "pass"
    assert rows.loc["high_costs", "total_cost_bps"] == pytest.approx(15.0)
    assert rows.loc["high_costs", "total_cost_return"] > rows.loc[
        "canonical_costs", "total_cost_return"
    ]
    assert result.summary_metrics["schema_version"] == (
        TRANSACTION_COST_SENSITIVITY_SUMMARY_SCHEMA_VERSION
    )
    assert result.summary_metrics["scenario_order"] == ["canonical_costs", "high_costs"]
    assert result.summary_metrics["pass_count"] == 1
    assert result.summary_metrics["warning_count"] == 1
    assert result.summary_metrics["baseline_status"] == "warning"
    assert result.summary_metrics["largest_total_cost_scenario_id"] == "high_costs"
    assert result.summary_metrics["max_total_cost_increase_vs_baseline"] > 0.0

    payload = result.to_dict()
    assert payload["schema_version"] == TRANSACTION_COST_SENSITIVITY_BATCH_SCHEMA_VERSION
    assert payload["scenario_count"] == 2
    assert payload["config"] == config.to_dict()
    assert payload["summary_metrics"] == result.summary_metrics


def test_transaction_cost_sensitivity_results_move_with_cost_and_turnover_parameters() -> None:
    turnover = [0.10, 0.20, 0.40]
    result = run_transaction_cost_sensitivity_batch(
        _equity_curve(returns=[0.01, 0.01, 0.01], turnover=turnover),
        sensitivity_config=default_transaction_cost_sensitivity_config(),
        batch_id="directional_sensitivity",
    )

    rows = result.summary.set_index("scenario_id")
    cost_order = ["no_costs", "low_costs", "canonical_costs", "high_costs"]
    total_cost_returns = rows.loc[cost_order, "total_cost_return"].astype(float)
    cost_adjusted_returns = rows.loc[
        cost_order, "cost_adjusted_cumulative_return"
    ].astype(float)
    expected_cost_returns = {
        "no_costs": 0.0,
        "low_costs": sum(turnover) * 3.0 / 10_000,
        "canonical_costs": sum(turnover) * 7.0 / 10_000,
        "high_costs": sum(turnover) * 15.0 / 10_000,
    }

    assert total_cost_returns.tolist() == sorted(total_cost_returns.tolist())
    assert cost_adjusted_returns.tolist() == sorted(
        cost_adjusted_returns.tolist(),
        reverse=True,
    )
    for scenario_id, expected_total_cost in expected_cost_returns.items():
        assert rows.loc[scenario_id, "total_cost_return"] == pytest.approx(
            expected_total_cost
        )
        assert 0.0 <= rows.loc[scenario_id, "total_cost_return"] <= (
            sum(turnover) * 15.0 / 10_000
        )
    assert rows.loc["no_costs", "baseline_cost_adjusted_cumulative_return_delta"] > 0.0
    assert rows.loc["high_costs", "baseline_cost_adjusted_cumulative_return_delta"] < 0.0

    assert rows.loc["tight_turnover_budget", "turnover"] == pytest.approx(
        sum(turnover) / len(turnover)
    )
    assert bool(rows.loc["tight_turnover_budget", "turnover_budget_pass"]) is False
    assert bool(rows.loc["tight_turnover_budget", "max_daily_turnover_pass"]) is True
    assert rows.loc["tight_turnover_budget", "status"] == "warning"
    assert bool(rows.loc["loose_turnover_budget", "turnover_budget_pass"]) is True
    assert bool(rows.loc["loose_turnover_budget", "max_daily_turnover_pass"]) is True
    assert rows.loc[
        "tight_turnover_budget", "cost_adjusted_cumulative_return"
    ] == pytest.approx(rows.loc["canonical_costs", "cost_adjusted_cumulative_return"])
    assert rows.loc["tight_turnover_budget", "total_cost_return"] == pytest.approx(
        rows.loc["canonical_costs", "total_cost_return"]
    )


def test_transaction_cost_sensitivity_batch_can_execute_scenarios_with_runner() -> None:
    config = _two_scenario_config()
    executed: list[str] = []

    def runner(scenario: TransactionCostSensitivityScenario) -> pd.DataFrame:
        executed.append(scenario.scenario_id)
        turnover = 0.3
        return _equity_curve(
            returns=[0.01, 0.01, -0.005],
            turnover=[turnover, turnover, turnover],
            cost_bps=scenario.cost_bps,
            slippage_bps=scenario.slippage_bps,
        )

    result = run_transaction_cost_sensitivity_batch(
        sensitivity_config=config,
        scenario_runner=runner,
    )
    rows = result.summary.set_index("scenario_id")

    assert executed == ["canonical_costs", "high_costs"]
    assert result.execution_mode == "runner"
    assert rows.loc["canonical_costs", "execution_mode"] == "runner"
    assert rows.loc["canonical_costs", "status"] == "warning"
    assert rows.loc["high_costs", "status"] == "pass"
    assert rows.loc["canonical_costs", "baseline_cost_adjusted_cumulative_return_delta"] == (
        pytest.approx(0.0)
    )
    assert rows.loc["high_costs", "baseline_total_cost_return_delta"] > 0.0
    assert result.summary_metrics["execution_mode"] == "runner"
    assert result.summary_metrics["worst_cost_adjusted_scenario_id"] == "high_costs"


def test_transaction_cost_sensitivity_batch_records_runner_failures() -> None:
    config = _two_scenario_config()

    def runner(scenario: TransactionCostSensitivityScenario) -> pd.DataFrame:
        if scenario.scenario_id == "high_costs":
            raise RuntimeError("repricing failed")
        return _equity_curve(cost_bps=scenario.cost_bps, slippage_bps=scenario.slippage_bps)

    result = run_transaction_cost_sensitivity_batch(
        sensitivity_config=config,
        scenario_runner=runner,
        batch_id="runner_error",
    )
    rows = result.summary.set_index("scenario_id")

    assert rows.loc["canonical_costs", "status"] == "pass"
    assert rows.loc["high_costs", "status"] == "error"
    assert rows.loc["high_costs", "observations"] == 0
    assert rows.loc["high_costs", "error_code"] == "RuntimeError"
    assert rows.loc["high_costs", "error_message"] == "repricing failed"
    assert result.summary_metrics["error_count"] == 1
    assert result.summary_metrics["all_scenarios_evaluable"] is False
    assert result.summary_metrics["error_messages"] == ["high_costs: repricing failed"]


def test_transaction_cost_sensitivity_batch_table_rejects_unknown_scenarios() -> None:
    with pytest.raises(ValueError, match="unknown scenarios"):
        build_transaction_cost_sensitivity_batch_table(
            pd.DataFrame([{"scenario_id": "not_configured"}]),
            sensitivity_config=_two_scenario_config(),
        )


def test_transaction_cost_sensitivity_batch_table_sorts_and_fills_baseline_deltas() -> None:
    raw_summary = pd.DataFrame(
        [
            {
                "scenario_id": "high_costs",
                "label": "High costs",
                "is_baseline": False,
                "cost_bps": 10.0,
                "slippage_bps": 5.0,
                "total_cost_bps": 15.0,
                "average_daily_turnover_budget": 0.75,
                "max_daily_turnover": 1.0,
                "observations": 3,
                "cost_adjusted_cumulative_return": 0.03,
                "excess_return": 0.01,
                "total_cost_return": 0.004,
                "turnover_budget_pass": True,
                "max_daily_turnover_pass": True,
            },
            {
                "scenario_id": "canonical_costs",
                "label": "Canonical costs",
                "is_baseline": True,
                "cost_bps": 5.0,
                "slippage_bps": 2.0,
                "total_cost_bps": 7.0,
                "average_daily_turnover_budget": 0.25,
                "max_daily_turnover": 0.25,
                "observations": 3,
                "cost_adjusted_cumulative_return": 0.05,
                "excess_return": 0.02,
                "total_cost_return": 0.002,
                "turnover_budget_pass": False,
                "max_daily_turnover_pass": False,
            },
        ]
    )

    table = build_transaction_cost_sensitivity_batch_table(
        raw_summary,
        sensitivity_config=_two_scenario_config(),
        batch_id="ordered",
    )
    rows = table.set_index("scenario_id")

    assert table["scenario_id"].tolist() == ["canonical_costs", "high_costs"]
    assert rows.loc[
        "canonical_costs", "baseline_cost_adjusted_cumulative_return_delta"
    ] == pytest.approx(0.0)
    assert rows.loc[
        "high_costs", "baseline_cost_adjusted_cumulative_return_delta"
    ] == pytest.approx(-0.02)
    assert rows.loc["high_costs", "baseline_excess_return_delta"] == pytest.approx(
        -0.01
    )
    assert rows.loc["high_costs", "baseline_total_cost_return_delta"] == pytest.approx(
        0.002
    )


def test_transaction_cost_sensitivity_summary_metrics_compare_scenarios() -> None:
    result = run_transaction_cost_sensitivity_batch(
        _equity_curve(turnover=[0.3, 0.3, 0.3]),
        sensitivity_config=_two_scenario_config(),
        batch_id="comparison",
    )

    metrics = build_transaction_cost_sensitivity_summary_metrics(
        result.summary.sample(frac=1.0, random_state=1),
        sensitivity_config=_two_scenario_config(),
        batch_id="comparison",
        execution_mode="regression",
    )

    assert metrics["schema_version"] == TRANSACTION_COST_SENSITIVITY_SUMMARY_SCHEMA_VERSION
    assert metrics["scenario_order"] == ["canonical_costs", "high_costs"]
    assert metrics["scenario_count"] == 2
    assert metrics["status_counts"] == {
        "pass": 1,
        "warning": 1,
        "insufficient_data": 0,
        "error": 0,
    }
    assert metrics["all_scenarios_evaluable"] is True
    assert metrics["all_turnover_budgets_pass"] is False
    assert metrics["turnover_budget_breach_count"] == 1
    assert metrics["max_daily_turnover_breach_count"] == 1
    assert metrics["best_cost_adjusted_scenario_id"] == "canonical_costs"
    assert metrics["worst_cost_adjusted_scenario_id"] == "high_costs"
    assert metrics["largest_total_cost_scenario_id"] == "high_costs"
    assert metrics["max_cost_adjusted_return_loss_vs_baseline"] > 0.0
    assert metrics["max_total_cost_increase_vs_baseline"] > 0.0


def test_transaction_cost_sensitivity_batch_requires_baseline_for_comparison() -> None:
    config = _two_scenario_config()
    with pytest.raises(ValueError, match="configured baseline scenario"):
        build_transaction_cost_sensitivity_batch_table(
            pd.DataFrame([{"scenario_id": "high_costs"}]),
            sensitivity_config=config,
        )


def test_transaction_cost_sensitivity_batch_requires_input_without_runner() -> None:
    with pytest.raises(ValueError, match="equity_curve_or_result is required"):
        run_transaction_cost_sensitivity_batch(sensitivity_config=_two_scenario_config())


def _two_scenario_config() -> TransactionCostSensitivityConfig:
    return TransactionCostSensitivityConfig(
        baseline_scenario_id="canonical_costs",
        scenarios=(
            TransactionCostSensitivityScenario(
                scenario_id="canonical_costs",
                label="Canonical costs",
                cost_bps=5.0,
                slippage_bps=2.0,
                average_daily_turnover_budget=0.25,
                max_daily_turnover=0.25,
            ),
            TransactionCostSensitivityScenario(
                scenario_id="high_costs",
                label="High costs",
                cost_bps=10.0,
                slippage_bps=5.0,
                average_daily_turnover_budget=0.75,
                max_daily_turnover=1.00,
            ),
        ),
    )


def _equity_curve(
    *,
    returns: list[float] | None = None,
    turnover: list[float] | None = None,
    cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> pd.DataFrame:
    returns = returns or [0.02, -0.01, 0.015]
    turnover = turnover or [0.2, 0.2, 0.2]
    transaction_cost = [value * cost_bps / 10_000 for value in turnover]
    slippage_cost = [value * slippage_bps / 10_000 for value in turnover]
    total_cost = [
        transaction + slippage
        for transaction, slippage in zip(transaction_cost, slippage_cost, strict=True)
    ]
    net_returns = [
        gross - cost for gross, cost in zip(returns, total_cost, strict=True)
    ]
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "gross_return": returns,
            "portfolio_return": net_returns,
            "cost_adjusted_return": net_returns,
            "net_return": net_returns,
            "benchmark_return": [0.005, 0.004, -0.002],
            "cost_adjusted_benchmark_return": [0.005, 0.004, -0.002],
            "turnover": turnover,
            "period_turnover": turnover,
            "exposure": [1.0, 1.0, 1.0],
            "transaction_cost_return": transaction_cost,
            "slippage_cost_return": slippage_cost,
            "total_cost_return": total_cost,
            "cost_bps": [cost_bps] * 3,
            "slippage_bps": [slippage_bps] * 3,
        }
    )
    frame["equity"] = (1.0 + frame["cost_adjusted_return"]).cumprod()
    frame["benchmark_equity"] = (1.0 + frame["benchmark_return"]).cumprod()
    frame["cost_adjusted_benchmark_equity"] = (
        1.0 + frame["cost_adjusted_benchmark_return"]
    ).cumprod()
    return frame
