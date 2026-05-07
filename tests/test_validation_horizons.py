from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from quant_research.models.tabular import infer_feature_columns
from quant_research.pipeline import PipelineConfig
from quant_research.validation import (
    DEFAULT_VALIDATION_HORIZONS,
    REQUIRED_VALIDATION_HORIZON_DAYS,
    build_validity_gate_report,
    default_horizon_labels,
    forward_return_column,
    horizon_label,
    required_validation_horizon_label,
)
from quant_research.validation.walk_forward import WalkForwardConfig


def test_default_validity_gate_horizons_remain_one_five_twenty_days() -> None:
    assert DEFAULT_VALIDATION_HORIZONS == (1, 5, 20)
    assert default_horizon_labels() == ("1d", "5d", "20d")
    assert PipelineConfig().validation_horizons == DEFAULT_VALIDATION_HORIZONS


def test_required_validity_gate_horizon_is_twenty_days() -> None:
    config = PipelineConfig()

    assert REQUIRED_VALIDATION_HORIZON_DAYS == 20
    assert required_validation_horizon_label() == "20d"
    assert config.required_validation_horizon == 20
    assert config.prediction_target_column == "forward_return_20"
    assert config.gap_periods >= 60
    assert config.embargo_periods >= 60


def test_pipeline_gap_and_embargo_defaults_cover_longest_price_lookback() -> None:
    config = PipelineConfig()

    assert config.gap_periods >= 60
    assert config.embargo_periods >= 60


def test_walk_forward_gap_and_embargo_defaults_cover_longest_price_lookback() -> None:
    config = WalkForwardConfig(prediction_horizon_periods=20)

    assert config.gap_periods >= 60
    assert config.embargo_periods >= 60


def test_horizon_helpers_produce_label_and_target_columns() -> None:
    assert horizon_label(5) == "5d"
    assert forward_return_column(20) == "forward_return_20"


def test_all_forward_return_labels_are_excluded_from_tabular_features() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3),
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "feature_a": [1.0, 2.0, 3.0],
            "forward_return_1": [0.01, 0.02, 0.03],
            "forward_return_5": [0.05, 0.06, 0.07],
            "forward_return_20": [0.20, 0.21, 0.22],
        }
    )

    assert infer_feature_columns(frame, target="forward_return_5") == ["feature_a"]


def test_one_and_five_day_results_are_diagnostic_only_for_validity_gate_pass_fail() -> None:
    dates = pd.date_range("2026-01-02", periods=21, freq="B")
    rows = []
    for fold, date in enumerate(dates):
        for ticker, expected, one_day, five_day, twenty_day in zip(
            ("AAPL", "MSFT", "SPY"),
            (0.003, 0.002, 0.001),
            (0.001, 0.002, 0.003),
            (0.001, 0.002, 0.003),
            (0.003, 0.002, 0.001),
            strict=True,
        ):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "fold": fold,
                "is_oos": fold >= len(dates) - 2,
                    "expected_return": expected,
                    "forward_return_1": one_day,
                    "forward_return_5": five_day,
                    "forward_return_20": twenty_day,
                }
            )
    predictions = pd.DataFrame(rows)
    validation_summary = pd.DataFrame(
        {
            "fold": range(len(dates)),
            "train_end": dates - pd.Timedelta(days=7),
            "test_start": dates,
            "is_oos": [False] * (len(dates) - 2) + [True, True],
            "labeled_test_observations": [3] * len(dates),
            "train_observations": [60] * len(dates),
        }
    )
    equity_curve = pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": [0.003] * len(dates),
            "cost_adjusted_return": [0.003] * len(dates),
            "gross_return": [0.003] * len(dates),
            "benchmark_return": [0.001] * len(dates),
            "turnover": [0.10] * len(dates),
        }
    )
    ablation_summary = [
        {"scenario": "all_features", "sharpe": 1.0, "excess_return": 0.10},
        {"scenario": "price_only", "sharpe": 0.4, "excess_return": 0.04},
        {"scenario": "text_only", "sharpe": 0.3, "excess_return": 0.03},
        {"scenario": "sec_only", "sharpe": 0.2, "excess_return": 0.02},
        {"scenario": "no_model_proxy", "sharpe": 0.5, "excess_return": 0.05},
        {"scenario": "no_costs", "sharpe": 0.6, "excess_return": 0.06},
    ]

    report = build_validity_gate_report(
        predictions,
        validation_summary,
        equity_curve,
        SimpleNamespace(cagr=2.0, sharpe=1.2, max_drawdown=-0.01, turnover=0.10),
        ablation_summary=ablation_summary,
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_1",
            benchmark_ticker="SPY",
            gap_periods=60,
            embargo_periods=60,
            cost_bps=0.0,
            slippage_bps=0.0,
        ),
    )

    assert report.system_validity_status == "pass"
    assert report.strategy_candidate_status in {"pass", "warning"}
    assert report.metrics["configured_target_column"] == "forward_return_1"
    assert report.metrics["target_column"] == "forward_return_20"
    assert report.gate_results["rank_ic"]["status"] == "pass"

    one_day = report.metrics["horizon_metrics"]["1d"]
    assert one_day["label"] == "diagnostic"
    assert one_day["role"] == "diagnostic"
    assert one_day["status"] == "fail"
    assert one_day["affects_pass_fail"] is False
    assert one_day["rank_ic_status"] == "fail"
    assert one_day["mean_rank_ic"] < 0
    assert report.metrics["diagnostic_horizon_metrics"]["1d"] == one_day
    assert report.gate_results["rank_ic_1d_diagnostic"]["status"] == "diagnostic"
    assert report.gate_results["rank_ic_1d_diagnostic"]["affects_strategy"] is False
    assert report.gate_results["rank_ic_1d_diagnostic"]["affects_pass_fail"] is False
    five_day = report.metrics["horizon_metrics"]["5d"]
    assert five_day["label"] == "diagnostic"
    assert five_day["role"] == "diagnostic"
    assert five_day["status"] == "fail"
    assert five_day["affects_pass_fail"] is False
    assert report.metrics["diagnostic_horizon_metrics"]["5d"] == five_day
    assert report.gate_results["rank_ic_5d_diagnostic"]["status"] == "diagnostic"
    twenty_day = report.metrics["horizon_metrics"]["20d"]
    assert twenty_day["label"] == "required"
    assert twenty_day["role"] == "decision"
    assert twenty_day["affects_pass_fail"] is True
    payload = json.loads(report.to_json())
    assert payload["metrics"]["horizon_metrics"]["1d"]["label"] == "diagnostic"
    markdown = report.to_markdown()
    assert "## Horizon Diagnostics" in markdown
    assert "| 1d | diagnostic | diagnostic | forward_return_1 | False | fail | False |" in markdown
    assert "| 5d | diagnostic | diagnostic | forward_return_5 | False | fail | False |" in markdown
    html = report.to_html()
    assert "<h2>Horizon Diagnostics</h2>" in html
    assert "<td>1d</td>" in html
    assert "<td>diagnostic</td>" in html
    assert "<td>forward_return_1</td>" in html


def test_twenty_day_robustness_runs_when_window_observations_are_sufficient() -> None:
    dates = pd.date_range("2026-01-02", periods=21, freq="B")
    rows = []
    for fold, date in enumerate(dates):
        for ticker, expected, five_day, twenty_day in zip(
            ("AAPL", "MSFT", "SPY"),
            (0.003, 0.002, 0.001),
            (0.003, 0.002, 0.001),
            (0.003, 0.002, 0.001),
            strict=True,
        ):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "fold": fold,
                "is_oos": fold >= len(dates) - 2,
                    "expected_return": expected,
                    "forward_return_5": five_day,
                    "forward_return_20": twenty_day,
                }
            )
    predictions = pd.DataFrame(rows)
    validation_summary = pd.DataFrame(
        {
            "fold": range(len(dates)),
            "train_end": dates - pd.Timedelta(days=7),
            "test_start": dates,
            "is_oos": [False] * (len(dates) - 2) + [True, True],
            "labeled_test_observations": [3] * len(dates),
            "train_observations": [60] * len(dates),
        }
    )
    equity_curve = pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": [0.003] * len(dates),
            "cost_adjusted_return": [0.003] * len(dates),
            "gross_return": [0.003] * len(dates),
            "benchmark_return": [0.001] * len(dates),
            "turnover": [0.10] * len(dates),
        }
    )
    ablation_summary = [
        {"scenario": "all_features", "sharpe": 1.0, "excess_return": 0.10},
        {"scenario": "price_only", "sharpe": 0.4, "excess_return": 0.04},
        {"scenario": "text_only", "sharpe": 0.3, "excess_return": 0.03},
        {"scenario": "sec_only", "sharpe": 0.2, "excess_return": 0.02},
        {"scenario": "no_model_proxy", "sharpe": 0.5, "excess_return": 0.05},
        {"scenario": "no_costs", "sharpe": 0.6, "excess_return": 0.06},
    ]

    report = build_validity_gate_report(
        predictions,
        validation_summary,
        equity_curve,
        SimpleNamespace(cagr=2.0, sharpe=1.2, max_drawdown=-0.01, turnover=0.10),
        ablation_summary=ablation_summary,
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_20",
            benchmark_ticker="SPY",
            gap_periods=60,
            embargo_periods=60,
            cost_bps=0.0,
            slippage_bps=0.0,
        ),
    )

    assert report.system_validity_status == "pass"
    assert report.strategy_candidate_status in {"pass", "warning"}
    assert report.gate_results["rank_ic"]["status"] == "pass"

    twenty_day = report.metrics["horizon_metrics"]["20d"]
    assert twenty_day["label"] == "required"
    assert twenty_day["role"] == "decision"
    assert twenty_day["status"] == "pass"
    assert twenty_day["affects_pass_fail"] is True
    assert twenty_day["minimum_observation_guard"] is True
    assert twenty_day["required_min_observations"] == 21
    assert twenty_day["max_observations_per_ticker"] == 21
    assert twenty_day["supported"] is True
    assert twenty_day["rank_ic_status"] == "pass"
    assert twenty_day["mean_rank_ic"] > 0
    assert twenty_day["positive_fold_ratio"] == 1.0
    assert twenty_day["oos_rank_ic"] > 0
    assert twenty_day["rank_ic_count"] == 21
    assert report.to_dict()["metrics"]["horizon_metrics"]["20d"] == twenty_day
    payload = json.loads(report.to_json())
    assert payload["metrics"]["horizon_metrics"]["20d"]["label"] == "required"
    assert "| 20d | required | decision | forward_return_20 | True | pass |" in report.to_markdown()
    html = report.to_html()
    assert "<h2>Horizon Diagnostics</h2>" in html
    assert "<td>20d</td>" in html
    assert "<td>required</td>" in html
    assert "<td>forward_return_20</td>" in html


def test_twenty_day_required_horizon_reports_insufficient_window_observations() -> None:
    dates = pd.date_range("2026-01-02", periods=5, freq="B")
    rows = []
    for fold, date in enumerate(dates):
        for ticker, expected, five_day, twenty_day in zip(
            ("AAPL", "MSFT", "SPY"),
            (0.003, 0.002, 0.001),
            (0.003, 0.002, 0.001),
            (0.001, 0.002, 0.003),
            strict=True,
        ):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "fold": fold,
                    "is_oos": fold == len(dates) - 1,
                    "expected_return": expected,
                    "forward_return_5": five_day,
                    "forward_return_20": twenty_day,
                }
            )
    predictions = pd.DataFrame(rows)
    validation_summary = pd.DataFrame(
        {
            "fold": range(5),
            "train_end": dates - pd.Timedelta(days=7),
            "test_start": dates,
            "is_oos": [False, False, False, False, True],
            "labeled_test_observations": [3, 3, 3, 3, 3],
            "train_observations": [60, 60, 60, 60, 60],
        }
    )
    equity_curve = pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": [0.003] * len(dates),
            "cost_adjusted_return": [0.003] * len(dates),
            "gross_return": [0.003] * len(dates),
            "benchmark_return": [0.001] * len(dates),
            "turnover": [0.10] * len(dates),
        }
    )
    ablation_summary = [
        {"scenario": "all_features", "sharpe": 1.0, "excess_return": 0.10},
        {"scenario": "price_only", "sharpe": 0.4, "excess_return": 0.04},
        {"scenario": "text_only", "sharpe": 0.3, "excess_return": 0.03},
        {"scenario": "sec_only", "sharpe": 0.2, "excess_return": 0.02},
        {"scenario": "no_model_proxy", "sharpe": 0.5, "excess_return": 0.05},
        {"scenario": "no_costs", "sharpe": 0.6, "excess_return": 0.06},
    ]

    report = build_validity_gate_report(
        predictions,
        validation_summary,
        equity_curve,
        SimpleNamespace(cagr=2.0, sharpe=1.2, max_drawdown=-0.01, turnover=0.10),
        ablation_summary=ablation_summary,
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_20",
            benchmark_ticker="SPY",
            gap_periods=60,
            embargo_periods=60,
            cost_bps=0.0,
            slippage_bps=0.0,
        ),
    )

    assert report.system_validity_status == "not_evaluable"
    assert report.strategy_candidate_status == "insufficient_data"
    assert report.gate_results["rank_ic"]["status"] == "insufficient_data"

    twenty_day = report.metrics["horizon_metrics"]["20d"]
    assert twenty_day["label"] == "required"
    assert twenty_day["role"] == "decision"
    assert twenty_day["affects_pass_fail"] is True
    assert twenty_day["minimum_observation_guard"] is True
    assert twenty_day["required_min_observations"] == 21
    assert twenty_day["max_observations_per_ticker"] == 5
    assert twenty_day["supported"] is False
    assert twenty_day["status"] == "insufficient_data"
    assert twenty_day["rank_ic_status"] == "insufficient_data"
    assert twenty_day["insufficient_data"] is True
    assert twenty_day["insufficient_data_status"] == "insufficient_data"
    assert twenty_day["insufficient_data_code"] == "insufficient_window_observations"
    assert "requires 21 observations for the 20d window" in twenty_day["rank_ic_reason"]
    assert twenty_day["mean_rank_ic"] is None
    assert twenty_day["rank_ic_count"] == 0
    assert report.to_dict()["metrics"]["horizon_metrics"]["20d"] == twenty_day
    payload = json.loads(report.to_json())
    assert payload["metrics"]["horizon_metrics"]["20d"]["status"] == "insufficient_data"
    markdown = report.to_markdown()
    assert "Insufficient Data Status" in markdown
    assert "Insufficient Data Code" in markdown
    assert "insufficient_window_observations" in markdown
    assert (
        "| 20d | required | decision | forward_return_20 | True | insufficient_data | True |"
        in markdown
    )
    html = report.to_html()
    assert "<h2>Horizon Diagnostics</h2>" in html
    assert "<td>20d</td>" in html
    assert "<td>required</td>" in html
    assert "<td>insufficient_data</td>" in html
    assert "<td>insufficient_window_observations</td>" in html
