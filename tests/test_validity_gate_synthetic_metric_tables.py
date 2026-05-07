from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research.validation.gate import (
    _evaluate_model_value_warning,
    _model_comparison_results,
)


def test_model_comparison_synthetic_metric_tables_cover_windows_baselines_ablations_ties_and_missing_data() -> None:
    ablation_summary = [
        {
            "scenario": "all_features",
            "label": "All features",
            "window_metrics": pd.DataFrame(
                [
                    {
                        "window_id": "fold_0",
                        "window_label": "Fold 0",
                        "window_role": "walk_forward_fold",
                        "rank_ic": 0.030,
                        "positive_fold_ratio": 0.65,
                        "oos_rank_ic": 0.020,
                        "sharpe": 0.70,
                        "turnover": 0.20,
                    },
                    {
                        "window_id": "fold_1",
                        "window_label": "Fold 1 OOS",
                        "window_role": "oos_holdout",
                        "rank_ic": 0.015,
                        "positive_fold_ratio": 0.55,
                        "oos_rank_ic": 0.010,
                        "sharpe": 0.45,
                        "turnover": 0.30,
                    },
                    {
                        "window_id": "fold_2",
                        "window_label": "Fold 2",
                        "window_role": "walk_forward_fold",
                        "sharpe": 0.60,
                        "turnover": 0.15,
                    },
                    {
                        "window_id": "fold_3",
                        "window_label": "Fold 3",
                        "window_role": "walk_forward_fold",
                        "sharpe": 0.50,
                        "turnover": 0.10,
                    },
                ]
            ),
        },
        {
            "scenario": "no_model_proxy",
            "label": "No model proxy",
            "window_metrics": {
                "fold_0": {
                    "rank_ic": 0.030,
                    "positive_fold_ratio": 0.60,
                    "oos_rank_ic": 0.015,
                    "sharpe": 0.70,
                    "turnover": 0.20,
                },
                "fold_1": {
                    "rank_ic": 0.012,
                    "positive_fold_ratio": 0.60,
                    "oos_rank_ic": 0.010,
                    "sharpe": 0.40,
                    "turnover": 0.20,
                },
                "fold_2": {
                    "rank_ic": 0.010,
                    "positive_fold_ratio": 0.50,
                    "oos_rank_ic": 0.005,
                    "sharpe": 0.55,
                    "turnover": 0.15,
                },
            },
        },
        {
            "scenario": "price_only",
            "label": "Price only",
            "window_metrics": [
                {
                    "window_id": "fold_0",
                    "rank_ic": 0.020,
                    "positive_fold_ratio": 0.55,
                    "oos_rank_ic": 0.010,
                    "sharpe": 0.65,
                    "turnover": 0.25,
                },
                {
                    "window_id": "fold_1",
                    "rank_ic": 0.020,
                    "positive_fold_ratio": 0.55,
                    "oos_rank_ic": 0.010,
                    "sharpe": 0.50,
                    "turnover": 0.25,
                },
            ],
        },
        {
            "scenario": "text_only",
            "label": "Text only",
            "sharpe": 0.40,
            "turnover": 0.10,
        },
        {
            "scenario": "sec_only",
            "label": "SEC only",
            "window_metrics": [{"window_id": "fold_1", "sharpe": 0.30, "turnover": 0.30}],
        },
        {
            "scenario": "no_costs",
            "label": "No costs",
            "kind": "cost",
            "window_metrics": [{"window_id": "fold_0", "sharpe": 0.90, "turnover": 0.40}],
        },
    ]
    baseline_comparisons = {
        "SPY": {
            "name": "SPY",
            "sharpe": 0.50,
            "cost_adjusted_cumulative_return": 0.04,
            "excess_return": 0.02,
            "turnover": 0.0,
        },
        "equal_weight": {
            "name": "equal_weight",
            "sharpe": 0.55,
            "cost_adjusted_cumulative_return": 0.05,
            "excess_return": 0.01,
            "turnover": 0.10,
        },
    }
    strategy_cost_adjusted_metrics = [
        {
            "name": "strategy",
            "sharpe": 0.75,
            "cost_adjusted_cumulative_return": 0.08,
            "excess_return": 0.03,
            "average_daily_turnover": 0.12,
        }
    ]

    rows = _model_comparison_results(
        ablation_summary,
        _stage1_config(),
        baseline_comparisons=baseline_comparisons,
        cost_adjusted_metric_comparison=strategy_cost_adjusted_metrics,
    )
    by_key = {
        (row["baseline"], row["window_id"], row["metric"]): row
        for row in rows
    }

    assert {
        (row["baseline"], row["baseline_role"])
        for row in rows
    } == {
        ("no_model_proxy", "model_baseline"),
        ("return_baseline_spy", "return_baseline"),
        ("return_baseline_equal_weight", "return_baseline"),
        ("price_only", "ablation"),
        ("text_only", "ablation"),
        ("sec_only", "ablation"),
        ("no_costs", "diagnostic"),
    }
    assert {row["window_id"] for row in rows if row["baseline"] == "no_model_proxy"} == {
        "fold_0",
        "fold_1",
        "fold_2",
        "fold_3",
    }

    tied_sharpe = by_key[("no_model_proxy", "fold_0", "sharpe")]
    assert tied_sharpe["candidate_value"] == pytest.approx(0.70)
    assert tied_sharpe["baseline_value"] == pytest.approx(0.70)
    assert tied_sharpe["absolute_delta"] == pytest.approx(0.0)
    assert tied_sharpe["relative_delta"] == pytest.approx(0.0)
    assert tied_sharpe["improvement"] == pytest.approx(0.0)
    assert tied_sharpe["outperformance_threshold"] == pytest.approx(0.05)
    assert tied_sharpe["operator"] == "candidate - baseline > 0.05"
    assert tied_sharpe["status"] == "fail"
    assert tied_sharpe["passed"] is False

    tied_turnover = by_key[("no_model_proxy", "fold_0", "turnover")]
    assert tied_turnover["operator"] == "baseline - candidate > 0"
    assert tied_turnover["status"] == "fail"
    assert tied_turnover["passed"] is False

    turnover_regression = by_key[("no_model_proxy", "fold_1", "turnover")]
    assert turnover_regression["candidate_value"] == pytest.approx(0.30)
    assert turnover_regression["baseline_value"] == pytest.approx(0.20)
    assert turnover_regression["absolute_delta"] == pytest.approx(0.10)
    assert turnover_regression["relative_delta"] == pytest.approx(0.50)
    assert turnover_regression["improvement"] == pytest.approx(-0.10)
    assert turnover_regression["status"] == "fail"
    assert turnover_regression["passed"] is False

    sharpe_regression = by_key[("price_only", "fold_1", "sharpe")]
    assert sharpe_regression["candidate_value"] == pytest.approx(0.45)
    assert sharpe_regression["baseline_value"] == pytest.approx(0.50)
    assert sharpe_regression["absolute_delta"] == pytest.approx(-0.05)
    assert sharpe_regression["status"] == "fail"

    missing_baseline_window = by_key[("no_model_proxy", "fold_3", "sharpe")]
    assert missing_baseline_window["status"] == "not_evaluable"
    assert missing_baseline_window["coverage_status"] == "not_evaluable"
    assert missing_baseline_window["reason_code"] == "missing_baseline_window"
    assert missing_baseline_window["candidate_window_available"] is True
    assert missing_baseline_window["baseline_window_available"] is False

    missing_candidate_metric = by_key[("no_model_proxy", "fold_2", "rank_ic")]
    assert missing_candidate_metric["status"] == "not_evaluable"
    assert missing_candidate_metric["reason_code"] == "missing_candidate_metric"

    missing_baseline_metric = by_key[("sec_only", "fold_1", "rank_ic")]
    assert missing_baseline_metric["status"] == "not_evaluable"
    assert missing_baseline_metric["reason_code"] == "missing_baseline_metric"

    return_baseline_rows = [
        row for row in rows if row["baseline_role"] == "return_baseline"
    ]
    assert {row["window_id"] for row in return_baseline_rows} == {"strategy_evaluation"}
    spy_sharpe = by_key[("return_baseline_spy", "strategy_evaluation", "sharpe")]
    assert spy_sharpe["candidate_value"] == pytest.approx(0.75)
    assert spy_sharpe["baseline_value"] == pytest.approx(0.50)
    assert spy_sharpe["improvement"] == pytest.approx(0.25)
    assert spy_sharpe["outperformance_threshold"] == pytest.approx(0.05)
    assert spy_sharpe["status"] == "pass"


def test_model_value_warning_synthetic_tables_distinguish_material_lift_from_ties_and_regressions() -> None:
    warning_gate = _evaluate_model_value_warning(
        [
            _model_value_row(
                "all_features",
                rank_ic=0.030,
                positive_fold_ratio=0.60,
                sharpe=0.70,
                max_drawdown=-0.20,
                cost_adjusted_return=0.100,
                turnover=0.20,
            ),
            _model_value_row(
                "no_model_proxy",
                rank_ic=0.030,
                positive_fold_ratio=0.60,
                sharpe=0.70,
                max_drawdown=-0.20,
                cost_adjusted_return=0.100,
                turnover=0.20,
            ),
            _model_value_row(
                "price_only",
                rank_ic=0.034,
                positive_fold_ratio=0.62,
                sharpe=0.76,
                max_drawdown=-0.18,
                cost_adjusted_return=0.110,
                turnover=0.18,
            ),
        ],
        _stage1_config(),
    )

    assert warning_gate["status"] == "warning"
    assert warning_gate["reason_code"] == "model_value_too_similar_to_proxy_or_price"
    assert warning_gate["warning_baselines"] == ["no_model_proxy", "price_only"]
    assert all(
        metric["materially_better"] is False
        for comparison in warning_gate["comparisons"]
        for metric in comparison["metrics"]
    )

    passing_gate = _evaluate_model_value_warning(
        [
            _model_value_row(
                "all_features",
                rank_ic=0.050,
                positive_fold_ratio=0.72,
                sharpe=0.90,
                max_drawdown=-0.12,
                cost_adjusted_return=0.140,
                turnover=0.12,
            ),
            _model_value_row(
                "no_model_proxy",
                rank_ic=0.030,
                positive_fold_ratio=0.60,
                sharpe=0.70,
                max_drawdown=-0.20,
                cost_adjusted_return=0.100,
                turnover=0.20,
            ),
            _model_value_row(
                "price_only",
                rank_ic=0.030,
                positive_fold_ratio=0.60,
                sharpe=0.70,
                max_drawdown=-0.20,
                cost_adjusted_return=0.100,
                turnover=0.20,
            ),
        ],
        _stage1_config(),
    )

    assert passing_gate["status"] == "pass"
    assert passing_gate["warning_baselines"] == []
    assert all(
        any(metric["materially_better"] is True for metric in comparison["metrics"])
        for comparison in passing_gate["comparisons"]
    )


def test_full_model_outperformance_passes_when_all_required_windows_clear_thresholds() -> None:
    rows = _model_comparison_results(
        [
            {
                "scenario": "all_features",
                "label": "All features",
                "window_metrics": [
                    _outperformance_window(
                        "fold_0",
                        rank_ic=0.030,
                        positive_fold_ratio=0.62,
                        oos_rank_ic=0.020,
                        sharpe=0.62,
                        max_drawdown=-0.15,
                        cost_adjusted_cumulative_return=0.070,
                        excess_return=0.030,
                        turnover=0.15,
                    ),
                    _outperformance_window(
                        "fold_1",
                        rank_ic=0.040,
                        positive_fold_ratio=0.70,
                        oos_rank_ic=0.025,
                        sharpe=0.71,
                        max_drawdown=-0.14,
                        cost_adjusted_cumulative_return=0.085,
                        excess_return=0.042,
                        turnover=0.12,
                    ),
                ],
            },
            {
                "scenario": "no_model_proxy",
                "label": "No model proxy",
                "window_metrics": [
                    _outperformance_window(
                        "fold_0",
                        rank_ic=0.020,
                        positive_fold_ratio=0.55,
                        oos_rank_ic=0.010,
                        sharpe=0.50,
                        max_drawdown=-0.20,
                        cost_adjusted_cumulative_return=0.050,
                        excess_return=0.020,
                        turnover=0.20,
                    ),
                    _outperformance_window(
                        "fold_1",
                        rank_ic=0.030,
                        positive_fold_ratio=0.62,
                        oos_rank_ic=0.015,
                        sharpe=0.60,
                        max_drawdown=-0.18,
                        cost_adjusted_cumulative_return=0.070,
                        excess_return=0.030,
                        turnover=0.18,
                    ),
                ],
            },
        ],
        _stage1_config(),
    )

    no_model_proxy_rows = [
        row for row in rows if row["baseline"] == "no_model_proxy"
    ]

    assert len(no_model_proxy_rows) == 16
    assert {row["window_id"] for row in no_model_proxy_rows} == {"fold_0", "fold_1"}
    assert {row["coverage_status"] for row in no_model_proxy_rows} == {"pass"}
    assert {row["status"] for row in no_model_proxy_rows} == {"pass"}
    assert all(row["passed"] is True for row in no_model_proxy_rows)


def test_full_model_outperformance_fails_when_lift_is_inconsistent_across_windows() -> None:
    rows = _model_comparison_results(
        [
            {
                "scenario": "all_features",
                "label": "All features",
                "window_metrics": [
                    _outperformance_window(
                        "fold_0",
                        rank_ic=0.040,
                        positive_fold_ratio=0.70,
                        oos_rank_ic=0.025,
                        sharpe=0.75,
                        max_drawdown=-0.12,
                        cost_adjusted_cumulative_return=0.100,
                        excess_return=0.050,
                        turnover=0.12,
                    ),
                    _outperformance_window(
                        "fold_1",
                        rank_ic=0.030,
                        positive_fold_ratio=0.61,
                        oos_rank_ic=0.014,
                        sharpe=0.64,
                        max_drawdown=-0.19,
                        cost_adjusted_cumulative_return=0.074,
                        excess_return=0.034,
                        turnover=0.22,
                    ),
                ],
            },
            {
                "scenario": "no_model_proxy",
                "label": "No model proxy",
                "window_metrics": [
                    _outperformance_window(
                        "fold_0",
                        rank_ic=0.020,
                        positive_fold_ratio=0.55,
                        oos_rank_ic=0.010,
                        sharpe=0.60,
                        max_drawdown=-0.20,
                        cost_adjusted_cumulative_return=0.070,
                        excess_return=0.030,
                        turnover=0.20,
                    ),
                    _outperformance_window(
                        "fold_1",
                        rank_ic=0.030,
                        positive_fold_ratio=0.60,
                        oos_rank_ic=0.015,
                        sharpe=0.60,
                        max_drawdown=-0.20,
                        cost_adjusted_cumulative_return=0.070,
                        excess_return=0.030,
                        turnover=0.20,
                    ),
                ],
            },
        ],
        _stage1_config(),
    )
    by_key = {
        (row["window_id"], row["metric"]): row
        for row in rows
        if row["baseline"] == "no_model_proxy"
    }

    assert by_key[("fold_0", "sharpe")]["status"] == "pass"
    assert by_key[("fold_0", "turnover")]["status"] == "pass"

    tied_rank_ic = by_key[("fold_1", "rank_ic")]
    assert tied_rank_ic["coverage_status"] == "pass"
    assert tied_rank_ic["improvement"] == pytest.approx(0.0)
    assert tied_rank_ic["outperformance_threshold"] == pytest.approx(0.005)
    assert tied_rank_ic["status"] == "fail"
    assert tied_rank_ic["passed"] is False

    insufficient_sharpe_lift = by_key[("fold_1", "sharpe")]
    assert insufficient_sharpe_lift["improvement"] == pytest.approx(0.04)
    assert insufficient_sharpe_lift["outperformance_threshold"] == pytest.approx(0.05)
    assert insufficient_sharpe_lift["status"] == "fail"
    assert insufficient_sharpe_lift["passed"] is False

    turnover_regression = by_key[("fold_1", "turnover")]
    assert turnover_regression["improvement"] == pytest.approx(-0.02)
    assert turnover_regression["operator"] == "baseline - candidate > 0"
    assert turnover_regression["status"] == "fail"
    assert turnover_regression["passed"] is False


def test_full_model_outperformance_is_not_evaluable_when_windows_or_metrics_are_insufficient() -> None:
    rows = _model_comparison_results(
        [
            {
                "scenario": "all_features",
                "label": "All features",
                "window_metrics": [
                    {
                        "window_id": "fold_0",
                        "window_label": "Fold 0",
                        "sharpe": 0.80,
                        "turnover": 0.10,
                    }
                ],
            },
            {
                "scenario": "no_model_proxy",
                "label": "No model proxy",
                "window_metrics": [
                    {
                        "window_id": "fold_0",
                        "window_label": "Fold 0",
                        "sharpe": 0.60,
                    },
                    {
                        "window_id": "fold_1",
                        "window_label": "Fold 1 OOS",
                        "rank_ic": 0.020,
                        "sharpe": 0.55,
                        "turnover": 0.20,
                    },
                ],
            },
        ],
        _stage1_config(),
    )
    by_key = {
        (row["window_id"], row["metric"]): row
        for row in rows
        if row["baseline"] == "no_model_proxy"
    }

    missing_candidate_metric = by_key[("fold_0", "rank_ic")]
    assert missing_candidate_metric["status"] == "not_evaluable"
    assert missing_candidate_metric["coverage_status"] == "not_evaluable"
    assert missing_candidate_metric["reason_code"] == "missing_candidate_metric"
    assert missing_candidate_metric["passed"] is None

    missing_baseline_metric = by_key[("fold_0", "turnover")]
    assert missing_baseline_metric["status"] == "not_evaluable"
    assert missing_baseline_metric["coverage_status"] == "not_evaluable"
    assert missing_baseline_metric["reason_code"] == "missing_baseline_metric"
    assert missing_baseline_metric["passed"] is None

    missing_candidate_window = by_key[("fold_1", "sharpe")]
    assert missing_candidate_window["status"] == "not_evaluable"
    assert missing_candidate_window["coverage_status"] == "not_evaluable"
    assert missing_candidate_window["reason_code"] == "missing_candidate_window"
    assert missing_candidate_window["candidate_window_available"] is False
    assert missing_candidate_window["baseline_window_available"] is True
    assert missing_candidate_window["passed"] is None


def _stage1_config() -> SimpleNamespace:
    return SimpleNamespace(
        tickers=["AAPL", "MSFT"],
        prediction_target_column="forward_return_5",
        benchmark_ticker="SPY",
        cost_bps=5.0,
        slippage_bps=2.0,
    )


def _model_value_row(
    scenario: str,
    *,
    rank_ic: float,
    positive_fold_ratio: float,
    sharpe: float,
    max_drawdown: float,
    cost_adjusted_return: float,
    turnover: float,
) -> dict[str, object]:
    return {
        "scenario": scenario,
        "validation_mean_information_coefficient": rank_ic,
        "validation_positive_ic_fold_ratio": positive_fold_ratio,
        "validation_oos_information_coefficient": rank_ic,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "deterministic_signal_evaluation_metrics": {
            "cost_adjusted_cumulative_return": cost_adjusted_return,
            "average_daily_turnover": turnover,
        },
    }


def _outperformance_window(
    window_id: str,
    *,
    rank_ic: float,
    positive_fold_ratio: float,
    oos_rank_ic: float,
    sharpe: float,
    max_drawdown: float,
    cost_adjusted_cumulative_return: float,
    excess_return: float,
    turnover: float,
) -> dict[str, object]:
    return {
        "window_id": window_id,
        "window_label": window_id.replace("_", " ").title(),
        "window_role": "walk_forward_fold",
        "rank_ic": rank_ic,
        "positive_fold_ratio": positive_fold_ratio,
        "oos_rank_ic": oos_rank_ic,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "cost_adjusted_cumulative_return": cost_adjusted_cumulative_return,
        "excess_return": excess_return,
        "turnover": turnover,
    }
