from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research.validation.gate import (
    MAX_DAILY_TURNOVER,
    MAX_MONTHLY_TURNOVER,
    ValidationGateThresholds,
    build_validity_gate_report,
    evaluate_average_daily_turnover_gate,
    evaluate_monthly_turnover_budget_gate,
    evaluate_monthly_turnover_gate,
    evaluate_turnover_gate,
    evaluate_turnover_validity_gate,
)


def _passing_predictions() -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=5, freq="B")
    rows = []
    for fold, date in enumerate(dates):
        for ticker, value in zip(("AAPL", "MSFT", "SPY"), (0.03, 0.02, 0.01), strict=True):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "fold": fold,
                    "is_oos": fold == len(dates) - 1,
                    "expected_return": value,
                    "forward_return_1": value / 100,
                    "forward_return_5": value,
                }
            )
    return pd.DataFrame(rows)


def _passing_validation_summary() -> pd.DataFrame:
    test_starts = pd.date_range("2026-02-02", periods=5, freq="B")
    return pd.DataFrame(
        {
            "fold": range(5),
            "train_end": test_starts - pd.Timedelta(days=7),
            "test_start": test_starts,
            "is_oos": [False, False, False, False, True],
            "labeled_test_observations": [3, 3, 3, 3, 3],
            "train_observations": [60, 60, 60, 60, 60],
        }
    )


def _passing_ablation_summary() -> list[dict[str, object]]:
    return [
        {"scenario": "all_features", "sharpe": 1.0, "excess_return": 0.10},
        {"scenario": "price_only", "sharpe": 0.4, "excess_return": 0.04},
        {"scenario": "text_only", "sharpe": 0.3, "excess_return": 0.03},
        {"scenario": "sec_only", "sharpe": 0.2, "excess_return": 0.02},
        {"scenario": "no_model_proxy", "sharpe": 0.5, "excess_return": 0.05},
        {"scenario": "no_costs", "sharpe": 0.6, "excess_return": 0.06},
    ]


def _passing_equity_curve(turnover: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-03-02", periods=len(turnover), freq="B"),
            "portfolio_return": [0.001] * len(turnover),
            "gross_return": [0.001] * len(turnover),
            "cost_adjusted_return": [0.001] * len(turnover),
            "benchmark_return": [0.0] * len(turnover),
            "turnover": turnover,
        }
    )


def test_average_daily_turnover_gate_passes_at_or_below_35_percent() -> None:
    thresholds = ValidationGateThresholds(max_daily_turnover=MAX_DAILY_TURNOVER)

    below = evaluate_average_daily_turnover_gate(0.3499, thresholds=thresholds)
    boundary = evaluate_average_daily_turnover_gate(0.35, thresholds=thresholds)

    assert below["status"] == "pass"
    assert boundary["status"] == "pass"
    assert boundary["value"] == 0.35
    assert boundary["threshold"] == 0.35
    assert boundary["operator"] == "<="


def test_average_daily_turnover_gate_does_not_pass_above_35_percent() -> None:
    thresholds = ValidationGateThresholds(max_daily_turnover=MAX_DAILY_TURNOVER)

    result = evaluate_average_daily_turnover_gate(0.3501, thresholds=thresholds)

    assert result["status"] != "pass"
    assert result["value"] == 0.3501
    warning = result["structured_warning"]
    assert warning["code"] == "average_daily_turnover_budget_exceeded"
    assert warning["gate"] == "average_daily_turnover"
    assert warning["metric"] == "average_daily_turnover"
    assert warning["realized_turnover"] == 0.3501
    assert warning["threshold"] == MAX_DAILY_TURNOVER
    assert "realized average daily turnover" in warning["message"]


def test_average_daily_turnover_gate_accepts_metrics_mapping_or_object() -> None:
    thresholds = ValidationGateThresholds(max_daily_turnover=MAX_DAILY_TURNOVER)

    mapping_result = evaluate_average_daily_turnover_gate({"turnover": 0.35}, thresholds=thresholds)
    object_result = evaluate_average_daily_turnover_gate(
        SimpleNamespace(turnover=0.35),
        thresholds=thresholds,
    )

    assert mapping_result["status"] == "pass"
    assert object_result["status"] == "pass"


def test_turnover_gate_alias_uses_same_average_daily_threshold() -> None:
    result = evaluate_turnover_gate(MAX_DAILY_TURNOVER)

    assert result["status"] == "pass"
    assert result["threshold"] == MAX_DAILY_TURNOVER


def test_monthly_turnover_budget_gate_passes_when_within_configured_budget() -> None:
    thresholds = ValidationGateThresholds(max_monthly_turnover=1.25)

    result = evaluate_monthly_turnover_budget_gate(1.10, thresholds=thresholds)
    boundary = evaluate_monthly_turnover_gate(1.25, thresholds=thresholds)

    assert result["status"] == "pass"
    assert result["value"] == 1.10
    assert result["threshold"] == 1.25
    assert result["operator"] == "<="
    assert boundary["status"] == "pass"


def test_monthly_turnover_budget_gate_sums_daily_turnover_by_month() -> None:
    equity_curve = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-02-02"]),
            "turnover": [0.30, 0.20, 0.40],
        }
    )
    thresholds = ValidationGateThresholds(monthly_turnover_budget=0.50)

    result = evaluate_monthly_turnover_budget_gate(equity_curve, thresholds=thresholds)

    assert result["status"] == "pass"
    assert result["value"] == 0.50
    assert result["threshold"] == 0.50
    assert result["monthly_turnover"] == {"2026-01": 0.50, "2026-02": 0.40}


def test_monthly_turnover_budget_gate_emits_structured_warning_when_exceeded() -> None:
    equity_curve = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-02-02"]),
            "turnover": [0.35, 0.20, 0.10],
        }
    )
    thresholds = ValidationGateThresholds(monthly_turnover_budget=0.50)

    result = evaluate_monthly_turnover_budget_gate(equity_curve, thresholds=thresholds)

    warning = result["structured_warning"]
    assert result["status"] == "warning"
    assert result["value"] == pytest.approx(0.55)
    assert result["threshold"] == 0.50
    assert warning["code"] == "monthly_turnover_budget_exceeded"
    assert warning["severity"] == "warning"
    assert warning["gate"] == "monthly_turnover_budget"
    assert warning["combined_gate"] == "turnover"
    assert warning["metric"] == "max_monthly_turnover"
    assert warning["realized_turnover"] == pytest.approx(0.55)
    assert warning["budget"] == 0.50
    assert warning["operator"] == "<="
    assert warning["monthly_turnover"]["2026-01"] == pytest.approx(0.55)
    assert warning["monthly_turnover"]["2026-02"] == pytest.approx(0.10)
    assert "realized max monthly turnover" in warning["message"]


def test_default_monthly_turnover_budget_derives_from_daily_budget() -> None:
    result = evaluate_monthly_turnover_budget_gate(MAX_MONTHLY_TURNOVER)

    assert result["status"] == "pass"
    assert result["threshold"] == MAX_MONTHLY_TURNOVER


def test_combined_turnover_validity_passes_when_daily_limit_passes() -> None:
    thresholds = ValidationGateThresholds(max_daily_turnover=0.35, monthly_turnover_budget=0.50)

    result = evaluate_turnover_validity_gate(0.20, 0.75, thresholds=thresholds)

    assert result["status"] == "pass"
    assert result["daily_status"] == "pass"
    assert result["monthly_status"] == "warning"
    assert result["passed_by"] == ["average_daily_turnover"]


def test_combined_turnover_validity_passes_when_monthly_budget_passes() -> None:
    thresholds = ValidationGateThresholds(max_daily_turnover=0.35, monthly_turnover_budget=0.50)

    result = evaluate_turnover_validity_gate(0.40, 0.50, thresholds=thresholds)

    assert result["status"] == "pass"
    assert result["daily_status"] == "warning"
    assert result["monthly_status"] == "pass"
    assert result["passed_by"] == ["monthly_turnover_budget"]


def test_combined_turnover_validity_fails_when_daily_limit_and_monthly_budget_miss() -> None:
    thresholds = ValidationGateThresholds(max_daily_turnover=0.35, monthly_turnover_budget=0.50)

    result = evaluate_turnover_validity_gate(0.40, 0.75, thresholds=thresholds)

    assert result["status"] == "fail"
    assert result["daily_status"] == "warning"
    assert result["monthly_status"] == "warning"
    assert result["passed_by"] == []


def test_validity_report_emits_turnover_warning_when_daily_budget_is_exceeded() -> None:
    thresholds = ValidationGateThresholds(max_daily_turnover=0.35, monthly_turnover_budget=1.00)

    report = build_validity_gate_report(
        _passing_predictions(),
        _passing_validation_summary(),
        _passing_equity_curve([0.10, 0.10, 0.10, 0.10, 0.10]),
        SimpleNamespace(cagr=1.0, sharpe=1.0, max_drawdown=0.0, turnover=0.36),
        ablation_summary=_passing_ablation_summary(),
        config=SimpleNamespace(
            prediction_target_column="forward_return_1",
            gap_periods=5,
            embargo_periods=5,
        ),
        thresholds=thresholds,
    )

    gate = report.gate_results["turnover"]
    assert gate["status"] == "pass"
    assert gate["daily_status"] == "warning"
    assert gate["monthly_status"] == "pass"
    assert report.warning is True
    assert [warning["code"] for warning in report.structured_warnings] == [
        "average_daily_turnover_budget_exceeded"
    ]
    assert any(
        "average_daily_turnover: realized average daily turnover" in warning
        for warning in report.warnings
    )


def test_validity_report_has_no_turnover_warning_when_turnover_is_within_budget() -> None:
    thresholds = ValidationGateThresholds(max_daily_turnover=0.35, monthly_turnover_budget=0.50)

    report = build_validity_gate_report(
        _passing_predictions(),
        _passing_validation_summary(),
        _passing_equity_curve([0.10, 0.10, 0.10, 0.10, 0.10]),
        SimpleNamespace(cagr=1.0, sharpe=1.0, max_drawdown=0.0, turnover=0.35),
        ablation_summary=_passing_ablation_summary(),
        config=SimpleNamespace(
            prediction_target_column="forward_return_1",
            gap_periods=5,
            embargo_periods=5,
        ),
        thresholds=thresholds,
    )

    gate = report.gate_results["turnover"]
    assert gate["status"] == "pass"
    assert gate["daily_status"] == "pass"
    assert gate["monthly_status"] == "pass"
    assert gate["passed_by"] == ["average_daily_turnover", "monthly_turnover_budget"]
    assert report.warning is False
    assert report.warnings == []
    assert report.structured_warnings == []
    assert "## Structured Warnings" not in report.to_markdown()


def test_validity_report_uses_combined_turnover_when_monthly_budget_passes() -> None:
    thresholds = ValidationGateThresholds(max_daily_turnover=0.35, monthly_turnover_budget=0.50)

    report = build_validity_gate_report(
        _passing_predictions(),
        _passing_validation_summary(),
        _passing_equity_curve([0.10, 0.10, 0.10, 0.10, 0.10]),
        SimpleNamespace(cagr=1.0, sharpe=1.0, max_drawdown=0.0, turnover=0.40),
        ablation_summary=_passing_ablation_summary(),
        config=SimpleNamespace(
            prediction_target_column="forward_return_1",
            gap_periods=5,
            embargo_periods=5,
        ),
        thresholds=thresholds,
    )

    gate = report.gate_results["turnover"]
    assert gate["status"] == "pass"
    assert gate["daily_status"] == "warning"
    assert gate["monthly_status"] == "pass"
    assert gate["passed_by"] == ["monthly_turnover_budget"]
    assert report.gate_results["monthly_turnover_budget"]["affects_strategy"] is False
    assert report.warning is True
    assert report.structured_warnings[0]["code"] == "average_daily_turnover_budget_exceeded"
    assert report.strategy_candidate_status == "pass"


def test_validity_report_uses_combined_turnover_when_daily_limit_passes() -> None:
    thresholds = ValidationGateThresholds(max_daily_turnover=0.35, monthly_turnover_budget=0.50)

    report = build_validity_gate_report(
        _passing_predictions(),
        _passing_validation_summary(),
        _passing_equity_curve([0.30, 0.30, 0.0, 0.0, 0.0]),
        SimpleNamespace(cagr=1.0, sharpe=1.0, max_drawdown=0.0, turnover=0.20),
        ablation_summary=_passing_ablation_summary(),
        config=SimpleNamespace(
            prediction_target_column="forward_return_1",
            gap_periods=5,
            embargo_periods=5,
        ),
        thresholds=thresholds,
    )

    gate = report.gate_results["turnover"]
    assert gate["status"] == "pass"
    assert gate["daily_status"] == "pass"
    assert gate["monthly_status"] == "warning"
    assert gate["passed_by"] == ["average_daily_turnover"]
    assert report.gate_results["monthly_turnover_budget"]["status"] == "warning"
    assert report.warning is True
    assert report.strategy_candidate_status == "pass"
    assert report.structured_warnings[0]["code"] == "monthly_turnover_budget_exceeded"
    assert report.structured_warnings[0]["value"] == pytest.approx(0.60)
    assert report.structured_warnings[0]["threshold"] == 0.50
    assert report.to_dict()["structured_warnings"][0]["gate"] == "monthly_turnover_budget"
    assert any(
        "monthly_turnover_budget: realized max monthly turnover" in warning
        for warning in report.warnings
    )
    assert "## Structured Warnings" in report.to_markdown()
    assert "monthly_turnover_budget_exceeded" in report.to_html()
    assert report.strategy_candidate_status == "pass"


def test_validity_report_includes_passing_monthly_turnover_budget_gate() -> None:
    equity_curve = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-02-02"]),
            "portfolio_return": [0.0, 0.0, 0.0],
            "gross_return": [0.001, 0.001, 0.001],
            "cost_adjusted_return": [0.0, 0.0, 0.0],
            "transaction_cost_return": [0.0003, 0.0002, 0.0001],
            "slippage_cost_return": [0.0001, 0.0001, 0.0002],
            "total_cost_return": [0.0004, 0.0003, 0.0003],
            "benchmark_return": [0.0, 0.0, 0.0],
            "turnover": [0.10, 0.20, 0.25],
        }
    )
    metrics = SimpleNamespace(cagr=0.0, sharpe=0.9, max_drawdown=0.0, turnover=0.10)
    thresholds = ValidationGateThresholds(monthly_turnover_budget=0.30)

    report = build_validity_gate_report(
        pd.DataFrame(),
        pd.DataFrame(),
        equity_curve,
        metrics,
        thresholds=thresholds,
    )

    gate = report.gate_results["monthly_turnover_budget"]
    assert gate["status"] == "pass"
    assert gate["value"] == pytest.approx(0.30)
    assert report.metrics["strategy_max_monthly_turnover"] == pytest.approx(0.30)
    assert report.metrics["monthly_turnover_budget"] == 0.30
    assert report.metrics["strategy_transaction_cost_return"] == pytest.approx(0.0006)
    assert report.metrics["strategy_slippage_cost_return"] == pytest.approx(0.0004)
    assert report.metrics["strategy_total_cost_return"] == pytest.approx(0.0010)
    assert report.evidence["cost_adjustment"]["has_explicit_cost_breakdown"] is True


def test_validity_report_spy_baseline_uses_configured_target_horizon() -> None:
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-05",
                    "2026-01-05",
                ]
            ),
            "ticker": ["SPY", "AAPL", "SPY", "AAPL"],
            "expected_return": [0.01, 0.02, 0.02, 0.03],
            "forward_return_1": [0.50, 0.60, 0.50, 0.60],
            "forward_return_5": [0.01, 0.04, 0.02, 0.05],
        }
    )

    report = build_validity_gate_report(
        predictions,
        pd.DataFrame(),
        pd.DataFrame(),
        SimpleNamespace(cagr=0.0, sharpe=0.0, max_drawdown=0.0, turnover=0.0),
        config=SimpleNamespace(
            prediction_target_column="forward_return_5",
            gap_periods=5,
            embargo_periods=5,
            benchmark_ticker="SPY",
        ),
    )

    spy = report.benchmark_results[0]
    expected_cagr = ((1.0 + 0.01) * (1.0 + 0.02)) ** (252 / 2) - 1.0
    assert spy["name"] == "SPY"
    assert spy["return_column"] == "forward_return_5"
    assert spy["return_horizon"] == 5
    assert spy["evaluation_observations"] == 2
    assert spy["evaluation_start"] == "2026-01-02"
    assert spy["evaluation_end"] == "2026-01-05"
    assert spy["cagr"] == pytest.approx(expected_cagr)


def test_validity_report_equal_weight_baseline_uses_tradable_universe_and_rebalance_costs() -> None:
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-05",
                    "2026-01-05",
                    "2026-01-05",
                ]
            ),
            "ticker": ["AAPL", "MSFT", "UNUSED", "AAPL", "MSFT", "UNUSED"],
            "expected_return": [0.02, 0.01, 0.99, 0.02, 0.01, 0.99],
            "forward_return_1": [0.10, 0.00, 9.00, 0.00, 0.10, 9.00],
        }
    )
    equity_curve = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            "portfolio_return": [0.02, 0.02],
            "benchmark_return": [0.0, 0.0],
            "turnover": [1.0, 0.0],
            "cost_bps": [10.0, 10.0],
            "slippage_bps": [0.0, 0.0],
        }
    )

    report = build_validity_gate_report(
        predictions,
        pd.DataFrame(),
        equity_curve,
        SimpleNamespace(cagr=10.0, sharpe=2.0, max_drawdown=0.0, turnover=0.50),
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_1",
            gap_periods=1,
            embargo_periods=1,
            cost_bps=10.0,
            slippage_bps=0.0,
        ),
    )

    equal_weight = next(row for row in report.benchmark_results if row["name"] == "equal_weight")
    expected_net_return = [(0.10 + 0.00) / 2 - 0.001, (0.00 + 0.10) / 2]
    expected_cagr = ((1 + expected_net_return[0]) * (1 + expected_net_return[1])) ** (252 / 2) - 1.0

    assert equal_weight["universe_tickers"] == ["AAPL", "MSFT"]
    assert equal_weight["expected_constituent_count"] == 2
    assert equal_weight["rebalance_count"] == 2
    assert equal_weight["average_daily_turnover"] == pytest.approx(0.5)
    assert equal_weight["transaction_cost_return"] == pytest.approx(0.001)
    assert equal_weight["cost_adjusted_cumulative_return"] == pytest.approx(
        (1 + expected_net_return[0]) * (1 + expected_net_return[1]) - 1.0
    )
    assert equal_weight["cagr"] == pytest.approx(expected_cagr)
    assert equal_weight["gross_cumulative_return"] == pytest.approx((1.05 * 1.05) - 1.0)
    assert "UNUSED" not in equal_weight["universe_tickers"]
    assert "Benchmark Results" in report.to_markdown()


def test_validity_report_exports_pipeline_and_cost_control_scenario_toggles() -> None:
    ablation_summary = [
        {
            "scenario": "no_model_proxy",
            "kind": "pipeline_control",
            "pipeline_controls": {
                "model_proxy": False,
                "cost": True,
                "slippage": True,
                "turnover": True,
            },
            "effective_cost_bps": 5.0,
            "effective_slippage_bps": 2.0,
            "sharpe": 0.1,
        },
        {
            "scenario": "no_costs",
            "kind": "cost",
            "pipeline_controls": {
                "model_proxy": True,
                "cost": False,
                "slippage": False,
                "turnover": False,
            },
            "effective_cost_bps": 0.0,
            "effective_slippage_bps": 0.0,
            "excess_return": 0.01,
            "sharpe": 0.2,
        },
    ]

    report = build_validity_gate_report(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        SimpleNamespace(cagr=0.0, sharpe=0.9, max_drawdown=0.0, turnover=0.10),
        ablation_summary=ablation_summary,
    )

    assert report.evidence["pipeline_control_required_scenarios"] == ["no_model_proxy"]
    assert report.evidence["cost_required_scenarios"] == ["no_costs"]
    assert report.evidence["pipeline_control_toggles"]["no_model_proxy"]["model_proxy"] is False
    assert report.evidence["cost_ablation_toggles"]["no_costs"] == {
        "model_proxy": True,
        "cost": False,
        "slippage": False,
        "turnover": False,
    }
    markdown = report.to_markdown()
    assert "## Pipeline Controls" in markdown
    assert "## Cost Ablations" in markdown
    assert "no_model_proxy" in markdown
    assert "no_costs" in markdown

    html = report.to_html()
    assert "<h2>Pipeline Controls</h2>" in html
    assert "<h2>Cost Ablations</h2>" in html
    assert "<td>no_model_proxy</td>" in html
    assert "<td>no_costs</td>" in html
