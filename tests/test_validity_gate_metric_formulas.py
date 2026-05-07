from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import quant_research.validation.gate as gate_module
from quant_research.validation import build_validity_gate_report
from quant_research.validation.gate import (
    _cost_adjustment_metrics,
    _rank_ic_metrics_for_horizon,
    calculate_top_decile_20d_excess_return,
)


def test_rank_ic_formula_uses_daily_spearman_fold_ratio_and_oos_mean() -> None:
    predictions = pd.DataFrame(
        {
            "date": [
                "2026-01-02",
                "2026-01-02",
                "2026-01-02",
                "2026-01-05",
                "2026-01-05",
                "2026-01-05",
                "2026-01-06",
                "2026-01-06",
                "2026-01-06",
            ],
            "ticker": ["AAPL", "MSFT", "NVDA"] * 3,
            "fold": [0, 0, 0, 0, 0, 0, 1, 1, 1],
            "is_oos": [False, False, False, False, False, False, True, True, True],
            "expected_return": [0.30, 0.20, 0.10] * 3,
            "forward_return_5": [
                0.03,
                0.02,
                0.01,
                0.01,
                0.02,
                0.03,
                0.06,
                0.04,
                0.02,
            ],
        }
    )

    metrics = _rank_ic_metrics_for_horizon(predictions, "forward_return_5", 5)

    assert metrics["rank_ic_count"] == 3
    assert metrics["mean_rank_ic"] == pytest.approx((1.0 - 1.0 + 1.0) / 3.0)
    assert metrics["positive_fold_ratio"] == pytest.approx(0.5)
    assert metrics["oos_rank_ic"] == pytest.approx(1.0)
    assert metrics["insufficient_data"] is False


def test_strategy_cost_adjustment_formula_compounds_returns_and_sums_costs() -> None:
    equity_curve = pd.DataFrame(
        {
            "gross_return": [0.040, -0.020, 0.010],
            "cost_adjusted_return": [0.035, -0.023, 0.008],
            "transaction_cost_return": [0.003, 0.001, 0.001],
            "slippage_cost_return": [0.002, 0.002, 0.001],
            "total_cost_return": [0.005, 0.003, 0.002],
            "turnover": [0.50, 0.30, 0.20],
        }
    )

    metrics = _cost_adjustment_metrics(equity_curve)

    assert metrics["gross_cumulative_return"] == pytest.approx(
        (1.040 * 0.980 * 1.010) - 1.0
    )
    assert metrics["cost_adjusted_cumulative_return"] == pytest.approx(
        (1.035 * 0.977 * 1.008) - 1.0
    )
    assert metrics["transaction_cost_return"] == pytest.approx(0.005)
    assert metrics["slippage_cost_return"] == pytest.approx(0.005)
    assert metrics["total_cost_return"] == pytest.approx(0.010)
    assert metrics["mean_turnover"] == pytest.approx(1.0 / 3.0)
    assert metrics["has_explicit_cost_breakdown"] is True


def test_top_decile_20d_excess_return_uses_oos_predictions_without_future_selection() -> None:
    predictions = pd.DataFrame(
        {
            "date": ["2026-01-02"] * 10 + ["2026-01-05"] * 10,
            "ticker": [f"T{i:02d}" for i in range(10)] * 2,
            "is_oos": [False] * 10 + [True] * 10,
            "expected_return": list(range(10)) + list(range(10)),
            "forward_return_20": [
                0.20,
                0.18,
                0.16,
                0.14,
                0.12,
                0.10,
                0.08,
                0.06,
                0.04,
                0.02,
                -0.08,
                -0.06,
                -0.04,
                -0.02,
                0.00,
                0.02,
                0.04,
                0.06,
                0.08,
                0.30,
            ],
            "prediction_timestamp": ["2026-01-02 16:00:00"] * 10
            + ["2026-01-05 16:00:00"] * 10,
            "return_label_date": ["2026-02-02"] * 10 + ["2026-02-03"] * 10,
        }
    )

    metrics = calculate_top_decile_20d_excess_return(
        predictions,
        universe=[f"T{i:02d}" for i in range(10)],
    )

    assert metrics["sample_scope"] == "oos_labeled_predictions"
    assert metrics["date_count"] == 1
    assert metrics["top_decile_observation_count"] == 1
    assert metrics["mean_top_decile_20d_return"] == pytest.approx(0.30)
    assert metrics["mean_universe_20d_return"] == pytest.approx(0.03)
    assert metrics["top_decile_20d_excess_return"] == pytest.approx(0.27)
    assert metrics["status"] == "report_only"
    assert metrics["report_only"] is True
    assert metrics["decision_use"] == "none"


def test_top_decile_20d_excess_return_has_no_threshold_or_gate_status() -> None:
    predictions = pd.DataFrame(
        {
            "date": ["2026-01-02"] * 10,
            "ticker": [f"T{i:02d}" for i in range(10)],
            "expected_return": list(range(10)),
            "forward_return_20": [0.00] * 9 + [0.01],
            "prediction_timestamp": ["2026-01-02 16:00:00"] * 10,
            "return_label_date": ["2026-02-02"] * 10,
        }
    )

    metrics = calculate_top_decile_20d_excess_return(predictions)

    assert metrics["top_decile_20d_excess_return"] == pytest.approx(0.009)
    assert "report_only_threshold" not in metrics
    assert metrics["status"] == "report_only"
    assert metrics["report_only"] is True
    assert metrics["decision_use"] == "none"


def test_top_decile_20d_excess_return_rejects_future_prediction_timestamps() -> None:
    predictions = pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-02"],
            "ticker": ["AAPL", "MSFT"],
            "expected_return": [0.02, 0.01],
            "forward_return_20": [0.05, 0.01],
            "prediction_timestamp": ["2026-01-03 09:00:00", "2026-01-02 15:00:00"],
        }
    )

    with pytest.raises(ValueError, match="unavailable data"):
        calculate_top_decile_20d_excess_return(predictions)


def test_stage1_baseline_metric_formulas_use_cost_adjusted_excess_returns() -> None:
    dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
    predictions = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[0], dates[1], dates[1], dates[1]],
            "ticker": ["SPY", "AAPL", "MSFT", "SPY", "AAPL", "MSFT"],
            "expected_return": [0.01, 0.03, 0.02, 0.01, 0.03, 0.02],
            "forward_return_5": [0.01, 0.06, 0.04, -0.01, 0.04, 0.06],
            "fold": [0, 0, 0, 1, 1, 1],
            "is_oos": [False, False, False, True, True, True],
        }
    )
    equity_curve = pd.DataFrame(
        {
            "date": dates,
            "gross_return": [0.0215, 0.0300],
            "cost_adjusted_return": [0.0200, 0.0300],
            "portfolio_return": [0.0200, 0.0300],
            "benchmark_return": [0.0100, -0.0100],
            "transaction_cost_return": [0.0010, 0.0000],
            "slippage_cost_return": [0.0005, 0.0000],
            "total_cost_return": [0.0015, 0.0000],
            "turnover": [1.0, 0.0],
        }
    )
    strategy_cost_adjusted_return = (1.0200 * 1.0300) - 1.0
    spy_cost_adjusted_return = ((1.0100 - 0.0015) * (1.0 - 0.0100)) - 1.0
    equal_weight_cost_adjusted_return = ((1.0500 - 0.0015) * 1.0500) - 1.0

    report = build_validity_gate_report(
        predictions,
        pd.DataFrame(),
        equity_curve,
        SimpleNamespace(cagr=2.0, sharpe=1.4, max_drawdown=-0.01, turnover=0.50),
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_5",
            benchmark_ticker="SPY",
            cost_bps=10.0,
            slippage_bps=5.0,
            gap_periods=5,
            embargo_periods=5,
        ),
    )

    rows = {row["name"]: row for row in report.cost_adjusted_metric_comparison}

    assert rows["strategy"]["cost_adjusted_cumulative_return"] == pytest.approx(
        strategy_cost_adjusted_return
    )
    assert rows["strategy"]["gross_cumulative_return"] == pytest.approx(
        (1.0215 * 1.0300) - 1.0
    )
    assert rows["strategy"]["total_cost_return"] == pytest.approx(0.0015)

    assert rows["SPY"]["gross_cumulative_return"] == pytest.approx(
        (1.0100 * 0.9900) - 1.0
    )
    assert rows["SPY"]["cost_adjusted_cumulative_return"] == pytest.approx(
        spy_cost_adjusted_return
    )
    assert rows["SPY"]["average_daily_turnover"] == pytest.approx(0.5)
    assert rows["SPY"]["total_cost_return"] == pytest.approx(0.0015)
    assert rows["SPY"]["excess_return"] == pytest.approx(
        strategy_cost_adjusted_return - spy_cost_adjusted_return
    )
    assert rows["SPY"]["excess_return_status"] == "pass"

    assert rows["equal_weight"]["gross_cumulative_return"] == pytest.approx(
        (1.0500 * 1.0500) - 1.0
    )
    assert rows["equal_weight"]["cost_adjusted_cumulative_return"] == pytest.approx(
        equal_weight_cost_adjusted_return
    )
    assert rows["equal_weight"]["average_daily_turnover"] == pytest.approx(0.5)
    assert rows["equal_weight"]["total_cost_return"] == pytest.approx(0.0015)
    assert rows["equal_weight"]["excess_return"] == pytest.approx(
        strategy_cost_adjusted_return - equal_weight_cost_adjusted_return
    )
    assert rows["equal_weight"]["excess_return_status"] == "fail"


def test_validity_gate_report_exposes_report_only_top_decile_20d_excess_return() -> None:
    dates = pd.date_range("2026-01-02", periods=21, freq="B")
    tickers = [f"T{i:02d}" for i in range(10)]
    rows: list[dict[str, object]] = []
    for fold, date in enumerate(dates):
        for idx, ticker in enumerate(tickers):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "fold": fold,
                    "is_oos": fold >= len(dates) - 2,
                    "expected_return": float(idx),
                    "forward_return_20": 0.30 if idx == 9 else 0.00,
                }
            )
    predictions = pd.DataFrame(rows)
    validation_summary = pd.DataFrame(
        {
            "fold": range(len(dates)),
            "train_end": dates - pd.Timedelta(days=21),
            "test_start": dates,
            "is_oos": [False] * (len(dates) - 2) + [True, True],
            "labeled_test_observations": [len(tickers)] * len(dates),
            "train_observations": [252] * len(dates),
        }
    )
    equity_curve = pd.DataFrame(
        {
            "date": dates,
            "gross_return": [0.01] * len(dates),
            "cost_adjusted_return": [0.01] * len(dates),
            "portfolio_return": [0.01] * len(dates),
            "benchmark_return": [0.00] * len(dates),
            "turnover": [0.10] * len(dates),
            "transaction_cost_return": [0.0] * len(dates),
            "slippage_cost_return": [0.0] * len(dates),
            "total_cost_return": [0.0] * len(dates),
        }
    )

    report = build_validity_gate_report(
        predictions,
        validation_summary,
        equity_curve,
        SimpleNamespace(cagr=1.0, sharpe=1.0, max_drawdown=-0.01, turnover=0.10),
        config=SimpleNamespace(
            tickers=tickers,
            prediction_target_column="forward_return_20",
            benchmark_ticker="SPY",
            gap_periods=20,
            embargo_periods=20,
        ),
    )

    assert report.metrics["top_decile_20d_excess_return"] == pytest.approx(0.27)
    assert report.metrics["top_decile_20d_excess_return_status"] == "report_only"
    assert "top_decile_20d_excess_return_threshold" not in report.metrics
    assert report.metrics["top_decile_20d_excess_return_scope"] == (
        "oos_labeled_predictions"
    )
    assert report.metrics["top_decile_20d_excess_return_decision_use"] == "none"
    evidence = report.evidence["top_decile_20d_excess_return"]
    assert evidence["report_only"] is True
    assert "report_only_threshold" not in evidence
    assert evidence["decision_use"] == "none"
    assert evidence["sample_scope"] == "oos_labeled_predictions"
    assert "Report-Only Research Metrics" in report.to_markdown()
    assert "| top_decile_20d_excess_return | report_only | 0.2700 | forward_return_20 | oos_labeled_predictions | True | none |" in report.to_markdown()
    assert "<h2>Report-Only Research Metrics</h2>" in report.to_html()


def test_top_decile_20d_excess_return_does_not_change_gate_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.date_range("2026-01-02", periods=21, freq="B")
    tickers = [f"T{i:02d}" for i in range(10)]
    rows: list[dict[str, object]] = []
    for fold, date in enumerate(dates):
        for idx, ticker in enumerate(tickers):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "fold": fold,
                    "is_oos": fold >= len(dates) - 2,
                    "expected_return": float(idx),
                    "forward_return_20": 0.30 if idx == 9 else 0.00,
                }
            )
    predictions = pd.DataFrame(rows)
    validation_summary = pd.DataFrame(
        {
            "fold": range(len(dates)),
            "train_end": dates - pd.Timedelta(days=21),
            "test_start": dates,
            "is_oos": [False] * (len(dates) - 2) + [True, True],
            "labeled_test_observations": [len(tickers)] * len(dates),
            "train_observations": [252] * len(dates),
        }
    )
    equity_curve = pd.DataFrame(
        {
            "date": dates,
            "gross_return": [0.01] * len(dates),
            "cost_adjusted_return": [0.01] * len(dates),
            "portfolio_return": [0.01] * len(dates),
            "benchmark_return": [0.00] * len(dates),
            "turnover": [0.10] * len(dates),
            "transaction_cost_return": [0.0] * len(dates),
            "slippage_cost_return": [0.0] * len(dates),
            "total_cost_return": [0.0] * len(dates),
        }
    )
    config = SimpleNamespace(
        tickers=tickers,
        prediction_target_column="forward_return_20",
        benchmark_ticker="SPY",
        gap_periods=20,
        embargo_periods=20,
    )

    def report_only_metric(value: float) -> dict[str, object]:
        return {
            "metric": "top_decile_20d_excess_return",
            "target_column": "forward_return_20",
            "sample_scope": "oos_labeled_predictions",
            "status": "report_only",
            "reason": "report-only diagnostic; not used for scoring, action, ranking, thresholding, or gating",
            "report_only": True,
            "decision_use": "none",
            "top_decile_20d_excess_return": value,
        }

    def build_report_with_metric(value: float):
        monkeypatch.setattr(
            gate_module,
            "calculate_top_decile_20d_excess_return",
            lambda *args, **kwargs: report_only_metric(value),
        )
        return build_validity_gate_report(
            predictions,
            validation_summary,
            equity_curve,
            SimpleNamespace(cagr=1.0, sharpe=1.0, max_drawdown=-0.01, turnover=0.10),
            config=config,
        )

    high_report = build_report_with_metric(99.0)
    low_report = build_report_with_metric(-99.0)

    assert high_report.metrics["top_decile_20d_excess_return"] == 99.0
    assert low_report.metrics["top_decile_20d_excess_return"] == -99.0
    assert high_report.metrics["top_decile_20d_excess_return_status"] == "report_only"
    assert low_report.metrics["top_decile_20d_excess_return_status"] == "report_only"

    decision_fields = (
        "final_gate_decision",
        "final_gate_status",
        "deterministic_gate_aggregation",
        "model_value",
        "deterministic_strategy_validity",
    )
    for field in decision_fields:
        assert high_report.metrics[field] == low_report.metrics[field]
    assert high_report.system_validity_status == low_report.system_validity_status
    assert high_report.strategy_candidate_status == low_report.strategy_candidate_status
    assert high_report.system_validity_pass is low_report.system_validity_pass
    assert high_report.strategy_pass is low_report.strategy_pass
