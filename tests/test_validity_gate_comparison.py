from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research.backtest.metrics import calculate_cost_adjusted_returns, calculate_metrics
from quant_research.validation import (
    EQUAL_WEIGHT_BASELINE_TYPE,
    MARKET_BENCHMARK_BASELINE_TYPE,
    BaselineComparisonInput,
    ValidationGateThresholds,
    build_validity_gate_report,
    write_validity_gate_artifacts,
)
from quant_research.validation.gate import (
    _evaluate_cost_adjusted,
    _evaluate_deterministic_strategy_validity,
)


def _comparison_predictions() -> pd.DataFrame:
    dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
    return pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[0], dates[1], dates[1], dates[1]],
            "ticker": ["SPY", "AAPL", "MSFT", "SPY", "AAPL", "MSFT"],
            "expected_return": [0.01, 0.03, 0.02, 0.01, 0.03, 0.02],
            "forward_return_1": [0.01, 0.06, 0.04, -0.01, 0.04, 0.06],
            "fold": [0, 0, 0, 1, 1, 1],
            "is_oos": [False, False, False, True, True, True],
        }
    )


def _comparison_equity_curve() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            "portfolio_return": [0.02, 0.03],
            "cost_adjusted_return": [0.02, 0.03],
            "gross_return": [0.0215, 0.03],
            "transaction_cost_return": [0.0010, 0.0],
            "slippage_cost_return": [0.0005, 0.0],
            "total_cost_return": [0.0015, 0.0],
            "benchmark_return": [0.01, -0.01],
            "turnover": [1.0, 0.0],
            "cost_bps": [10.0, 10.0],
            "slippage_bps": [5.0, 5.0],
        }
    )


def _comparison_metrics() -> SimpleNamespace:
    return SimpleNamespace(cagr=2.0, sharpe=1.4, max_drawdown=-0.01, turnover=0.50)


def _assert_stage1_artifact_contains_both_baseline_comparisons(
    payload: dict[str, object],
) -> None:
    baseline_comparisons = payload["baseline_comparisons"]
    assert isinstance(baseline_comparisons, dict)
    assert set(baseline_comparisons) == {"SPY", "equal_weight"}

    spy = baseline_comparisons["SPY"]
    equal_weight = baseline_comparisons["equal_weight"]
    assert spy["baseline_type"] == "market_benchmark"
    assert spy["return_basis"] == "cost_adjusted_benchmark_return"
    assert equal_weight["baseline_type"] == "equal_weight_universe"
    assert equal_weight["return_basis"] == "cost_adjusted_equal_weight_return"
    assert spy["excess_return_status"] == (
        "pass" if spy["excess_return"] > 0 else "fail"
    )
    assert equal_weight["excess_return_status"] == (
        "pass" if equal_weight["excess_return"] > 0 else "fail"
    )

    entries = payload["baseline_comparison_entries"]
    assert isinstance(entries, list)
    entries_by_name = {row["name"]: row for row in entries}
    assert set(entries_by_name) == {"SPY", "equal_weight"}
    assert entries_by_name["SPY"] == spy
    assert entries_by_name["equal_weight"] == equal_weight

    comparison = payload["cost_adjusted_metric_comparison"]
    assert isinstance(comparison, list)
    baseline_rows = {row["name"]: row for row in comparison if row.get("role") == "baseline"}
    assert set(baseline_rows) == {"SPY", "equal_weight"}
    assert baseline_rows["SPY"]["excess_return_status"] == spy["excess_return_status"]
    assert baseline_rows["equal_weight"]["excess_return_status"] == equal_weight[
        "excess_return_status"
    ]

    benchmark_rows = {row["name"]: row for row in payload["benchmark_results"]}
    assert benchmark_rows["SPY"]["excess_return_status"] == spy["excess_return_status"]
    assert benchmark_rows["equal_weight"]["excess_return_status"] == equal_weight[
        "excess_return_status"
    ]

    cost_gate = payload["gate_results"]["cost_adjusted_performance"]
    assert cost_gate["baseline_excess_returns"]["SPY"] == spy["excess_return"]
    assert cost_gate["baseline_excess_returns"]["equal_weight"] == equal_weight["excess_return"]
    assert cost_gate["baseline_excess_return_statuses"] == {
        "SPY": spy["excess_return_status"],
        "equal_weight": equal_weight["excess_return_status"],
    }


def test_cost_adjusted_gate_requires_strictly_positive_excess_return_boundary() -> None:
    result = _evaluate_cost_adjusted(
        [
            {"name": "SPY", "excess_return": 0.001, "evaluation_observations": 5},
            {"name": "equal_weight", "excess_return": 0.0, "evaluation_observations": 5},
        ]
    )

    assert result["status"] == "fail"
    assert result["operator"] == "excess_return > 0 for every required baseline"
    assert result["baseline_excess_returns"] == {"SPY": 0.001, "equal_weight": 0.0}
    assert result["baseline_excess_return_statuses"] == {
        "SPY": "pass",
        "equal_weight": "fail",
    }
    assert result["passed_baselines"] == ["SPY"]
    assert result["failed_baselines"] == ["equal_weight"]
    assert result["reason"] == (
        "net excess return is not positive versus required baseline(s): equal_weight"
    )


def test_deterministic_strategy_validity_passes_only_when_all_outperformance_rules_pass() -> None:
    passing_gate_results = {
        "cost_adjusted_performance": {"status": "pass", "reason": "cost pass"},
        "benchmark_comparison": {"status": "pass", "reason": "benchmark pass"},
        "turnover": {"status": "pass", "reason": "turnover pass"},
        "drawdown": {"status": "pass", "reason": "drawdown pass"},
    }

    passing = _evaluate_deterministic_strategy_validity(passing_gate_results)
    assert passing["status"] == "pass"
    assert passing["all_required_outperformance_rules_passed"] is True
    assert passing["failed_rules"] == []
    assert passing["passed_rules"] == [
        "cost_adjusted_performance",
        "benchmark_comparison",
        "turnover",
        "drawdown",
    ]

    failing_gate_results = {
        **passing_gate_results,
        "benchmark_comparison": {"status": "warning", "reason": "weak Sharpe"},
        "turnover": {"status": "fail", "reason": "turnover above budget"},
    }
    failing = _evaluate_deterministic_strategy_validity(failing_gate_results)

    assert failing["status"] == "fail"
    assert failing["all_required_outperformance_rules_passed"] is False
    assert failing["failed_rules"] == ["benchmark_comparison", "turnover"]
    assert failing["rule_statuses"] == {
        "cost_adjusted_performance": "pass",
        "benchmark_comparison": "warning",
        "turnover": "fail",
        "drawdown": "pass",
    }
    assert failing["operator"] == (
        "every required deterministic outperformance rule status == pass"
    )
    assert failing["reason_code"] == "required_outperformance_rule_not_passed"
    assert failing["reason_metadata"] == {
        "code": "required_outperformance_rule_not_passed",
        "metric": "benchmark_comparison",
        "value": "warning",
        "threshold": "pass",
        "operator": "==",
    }


def test_cost_adjusted_gate_fails_at_configured_collapse_threshold_boundary() -> None:
    thresholds = ValidationGateThresholds(cost_adjusted_collapse_threshold=0.05)
    baseline_results = [
        {"name": "SPY", "excess_return": 0.010, "evaluation_observations": 5},
        {"name": "equal_weight", "excess_return": 0.020, "evaluation_observations": 5},
    ]

    boundary = _evaluate_cost_adjusted(
        baseline_results,
        cost_adjustment={"cost_adjusted_cumulative_return": 0.05},
        thresholds=thresholds,
    )
    above = _evaluate_cost_adjusted(
        baseline_results,
        cost_adjustment={"cost_adjusted_cumulative_return": 0.0501},
        thresholds=thresholds,
    )

    assert boundary["status"] == "fail"
    assert boundary["failed_baselines"] == []
    assert boundary["collapse_status"] == "fail"
    assert boundary["collapse_threshold"] == 0.05
    assert boundary["collapse_operator"] == ">"
    assert boundary["collapse_reason_code"] == (
        "cost_adjusted_cumulative_return_at_or_below_collapse_threshold"
    )
    assert boundary["reason_metadata"] == {
        "code": "cost_adjusted_cumulative_return_at_or_below_collapse_threshold",
        "metric": "cost_adjusted_cumulative_return",
        "value": 0.05,
        "threshold": 0.05,
        "operator": ">",
    }
    assert above["status"] == "pass"
    assert above["collapse_status"] == "pass"


def test_cost_adjusted_gate_fails_below_configured_collapse_threshold() -> None:
    thresholds = ValidationGateThresholds(cost_adjusted_collapse_threshold=0.05)
    baseline_results = [
        {"name": "SPY", "excess_return": 0.010, "evaluation_observations": 5},
        {"name": "equal_weight", "excess_return": 0.020, "evaluation_observations": 5},
    ]

    result = _evaluate_cost_adjusted(
        baseline_results,
        cost_adjustment={"cost_adjusted_cumulative_return": 0.049},
        thresholds=thresholds,
    )

    assert result["status"] == "fail"
    assert result["failed_baselines"] == []
    assert result["passed_baselines"] == ["SPY", "equal_weight"]
    assert result["baseline_excess_return_statuses"] == {
        "SPY": "pass",
        "equal_weight": "pass",
    }
    assert result["collapse_status"] == "fail"
    assert result["collapse_threshold"] == 0.05
    assert result["collapse_operator"] == ">"
    assert result["cost_adjusted_cumulative_return"] == pytest.approx(0.049)
    assert result["reason"] == (
        "cost-adjusted cumulative return 0.0490 is at or below "
        "configured collapse threshold 0.0500"
    )
    assert result["reason_metadata"] == {
        "code": "cost_adjusted_cumulative_return_at_or_below_collapse_threshold",
        "metric": "cost_adjusted_cumulative_return",
        "value": 0.049,
        "threshold": 0.05,
        "operator": ">",
    }


def test_validity_gate_report_fails_when_cost_adjusted_return_crosses_collapse_threshold(
    tmp_path,
) -> None:
    dates = pd.date_range("2026-01-02", periods=5, freq="B")
    predictions = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "fold": fold,
                "is_oos": fold == len(dates) - 1,
                "expected_return": expected_return,
                "forward_return_5": realized_return,
            }
            for fold, date in enumerate(dates)
            for ticker, expected_return, realized_return in (
                ("AAPL", 0.03, 0.0030),
                ("MSFT", 0.02, 0.0020),
                ("SPY", 0.01, 0.0010),
            )
        ]
    )
    validation_summary = pd.DataFrame(
        {
            "fold": range(len(dates)),
            "train_end": dates - pd.Timedelta(days=7),
            "test_start": dates,
            "is_oos": [False, False, False, False, True],
            "labeled_test_observations": [3] * len(dates),
            "train_observations": [60] * len(dates),
        }
    )
    equity_curve = pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": [0.012] * len(dates),
            "gross_return": [0.012] * len(dates),
            "cost_adjusted_return": [0.012] * len(dates),
            "benchmark_return": [0.001] * len(dates),
            "turnover": [0.10] * len(dates),
        }
    )
    thresholds = ValidationGateThresholds(cost_adjusted_collapse_threshold=0.07)
    report = build_validity_gate_report(
        predictions,
        validation_summary,
        equity_curve,
        SimpleNamespace(cagr=2.0, sharpe=1.2, max_drawdown=-0.05, turnover=0.10),
        ablation_summary=[
            {"scenario": "all_features", "sharpe": 1.0, "excess_return": 0.10},
            {"scenario": "price_only", "sharpe": 0.4, "excess_return": 0.04},
            {"scenario": "text_only", "sharpe": 0.3, "excess_return": 0.03},
            {"scenario": "sec_only", "sharpe": 0.2, "excess_return": 0.02},
            {"scenario": "no_model_proxy", "sharpe": 0.5, "excess_return": 0.05},
            {"scenario": "no_costs", "sharpe": 0.6, "excess_return": 0.06},
        ],
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_5",
            benchmark_ticker="SPY",
            gap_periods=5,
            embargo_periods=5,
            cost_bps=0.0,
            slippage_bps=0.0,
        ),
        thresholds=thresholds,
    )

    strategy_cost_adjusted_return = (1.012**5) - 1.0
    cost_gate = report.gate_results["cost_adjusted_performance"]
    assert report.system_validity_status == "pass"
    assert report.strategy_candidate_status == "fail"
    assert report.strategy_pass is False
    assert cost_gate["status"] == "fail"
    assert cost_gate["failed_baselines"] == []
    assert cost_gate["baseline_excess_return_statuses"] == {
        "SPY": "pass",
        "equal_weight": "pass",
    }
    assert cost_gate["cost_adjusted_cumulative_return"] == pytest.approx(
        strategy_cost_adjusted_return
    )
    assert cost_gate["collapse_threshold"] == 0.07
    assert cost_gate["collapse_check"]["value"] == pytest.approx(
        strategy_cost_adjusted_return
    )
    assert cost_gate["collapse_check"]["threshold"] == 0.07
    assert cost_gate["reason_metadata"]["code"] == (
        "cost_adjusted_cumulative_return_at_or_below_collapse_threshold"
    )
    assert cost_gate["reason_metadata"]["metric"] == "cost_adjusted_cumulative_return"
    assert cost_gate["reason_metadata"]["operator"] == ">"

    json_path, markdown_path = write_validity_gate_artifacts(report, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    payload_cost_gate = payload["gate_results"]["cost_adjusted_performance"]
    assert payload_cost_gate["status"] == "fail"
    assert payload_cost_gate["reason_metadata"]["threshold"] == 0.07
    assert payload["metrics"]["cost_adjusted_collapse_threshold"] == 0.07
    assert payload["evidence"]["cost_adjusted_collapse_check"]["status"] == "fail"
    assert "- Strategy candidate: `fail`" in markdown
    assert (
        "| cost_adjusted_performance | fail | cost-adjusted cumulative return "
        in markdown
    )


def test_stage1_output_comparison_results_include_spy_and_equal_weight_benchmarks(tmp_path) -> None:
    report = build_validity_gate_report(
        _comparison_predictions(),
        pd.DataFrame(),
        _comparison_equity_curve(),
        _comparison_metrics(),
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_1",
            benchmark_ticker="SPY",
            cost_bps=10.0,
            slippage_bps=5.0,
        ),
    )

    comparison_rows = {
        row["name"]: row
        for row in report.cost_adjusted_metric_comparison
        if row.get("role") == "baseline"
    }
    assert set(comparison_rows) == {"SPY", "equal_weight"}
    assert comparison_rows["SPY"]["return_basis"] == "cost_adjusted_benchmark_return"
    assert comparison_rows["equal_weight"]["return_basis"] == "cost_adjusted_equal_weight_return"

    json_path, _ = write_validity_gate_artifacts(report, tmp_path)
    output_payload = json.loads(json_path.read_text(encoding="utf-8"))
    output_comparison_rows = {
        row["name"]: row
        for row in output_payload["cost_adjusted_metric_comparison"]
        if row.get("role") == "baseline"
    }
    assert set(output_comparison_rows) == {"SPY", "equal_weight"}
    assert output_comparison_rows["SPY"]["return_basis"] == "cost_adjusted_benchmark_return"
    assert output_comparison_rows["equal_weight"]["return_basis"] == "cost_adjusted_equal_weight_return"


def test_validity_gate_report_contains_cost_adjusted_side_by_side_comparison() -> None:
    report = build_validity_gate_report(
        _comparison_predictions(),
        pd.DataFrame(),
        _comparison_equity_curve(),
        _comparison_metrics(),
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_1",
            benchmark_ticker="SPY",
            cost_bps=10.0,
            slippage_bps=5.0,
        ),
    )

    comparison = report.cost_adjusted_metric_comparison
    rows = {row["name"]: row for row in comparison}

    assert list(rows) == ["strategy", "SPY", "equal_weight"]
    assert rows["strategy"]["return_basis"] == "cost_adjusted_return"
    assert rows["strategy"]["cost_adjusted_cumulative_return"] == pytest.approx(1.02 * 1.03 - 1)
    assert rows["strategy"]["total_cost_return"] == pytest.approx(0.0015)
    strategy_cost_adjusted_return = rows["strategy"]["cost_adjusted_cumulative_return"]

    assert rows["SPY"]["role"] == "baseline"
    assert rows["SPY"]["excess_return"] == pytest.approx(
        strategy_cost_adjusted_return - rows["SPY"]["cost_adjusted_cumulative_return"]
    )
    assert rows["SPY"]["strategy_excess_return"] == pytest.approx(rows["SPY"]["excess_return"])
    assert rows["SPY"]["excess_return_status"] == "pass"
    assert report.benchmark_results[0]["excess_return"] == pytest.approx(
        rows["SPY"]["excess_return"]
    )
    assert report.benchmark_results[0]["excess_return_status"] == "pass"
    assert rows["SPY"]["gross_cumulative_return"] > rows["SPY"]["cost_adjusted_cumulative_return"]
    assert rows["SPY"]["total_cost_return"] == pytest.approx(0.0015)
    assert rows["SPY"]["average_daily_turnover"] == pytest.approx(0.5)
    assert rows["SPY"]["cost_adjusted_cumulative_return"] == pytest.approx(
        (1 + 0.01 - 0.0015) * (1 - 0.01) - 1
    )

    expected_equal_weight_net = [0.05 - 0.0015, 0.05]
    assert rows["equal_weight"]["return_basis"] == "cost_adjusted_equal_weight_return"
    assert rows["equal_weight"]["average_daily_turnover"] == pytest.approx(0.5)
    assert rows["equal_weight"]["total_cost_return"] == pytest.approx(0.0015)
    assert rows["equal_weight"]["cost_adjusted_cumulative_return"] == pytest.approx(
        (1 + expected_equal_weight_net[0]) * (1 + expected_equal_weight_net[1]) - 1
    )
    assert rows["equal_weight"]["excess_return"] == pytest.approx(
        strategy_cost_adjusted_return - rows["equal_weight"]["cost_adjusted_cumulative_return"]
    )
    assert rows["equal_weight"]["strategy_excess_return"] == pytest.approx(
        rows["equal_weight"]["excess_return"]
    )
    assert rows["equal_weight"]["excess_return_status"] == "fail"
    assert rows["SPY"]["excess_return"] != pytest.approx(rows["equal_weight"]["excess_return"])

    assert list(report.baseline_comparisons) == ["SPY", "equal_weight"]
    assert [row["name"] for row in report.baseline_comparison_entries] == ["SPY", "equal_weight"]
    assert report.baseline_comparisons["SPY"]["baseline_type"] == "market_benchmark"
    assert report.baseline_comparisons["SPY"]["return_basis"] == "cost_adjusted_benchmark_return"
    assert report.baseline_comparisons["SPY"]["cagr"] == pytest.approx(rows["SPY"]["cagr"])
    assert report.baseline_comparisons["SPY"]["excess_return"] == pytest.approx(
        rows["SPY"]["excess_return"]
    )
    assert report.baseline_comparisons["SPY"]["strategy_excess_return"] == pytest.approx(
        rows["SPY"]["strategy_excess_return"]
    )
    assert report.baseline_comparisons["SPY"]["excess_return_status"] == "pass"
    assert report.baseline_comparisons["equal_weight"]["baseline_type"] == "equal_weight_universe"
    assert report.baseline_comparisons["equal_weight"][
        "cost_adjusted_cumulative_return"
    ] == pytest.approx(rows["equal_weight"]["cost_adjusted_cumulative_return"])
    assert report.baseline_comparisons["equal_weight"]["average_daily_turnover"] == pytest.approx(
        rows["equal_weight"]["average_daily_turnover"]
    )
    assert report.baseline_comparisons["equal_weight"]["excess_return"] == pytest.approx(
        rows["equal_weight"]["excess_return"]
    )
    assert report.baseline_comparisons["equal_weight"]["excess_return_status"] == "fail"
    assert report.benchmark_results[1]["excess_return_status"] == "fail"

    payload = report.to_dict()
    assert payload["cost_adjusted_metric_comparison"] == comparison
    assert payload["metrics"]["cost_adjusted_metric_comparison"] == comparison
    side_by_side = report.side_by_side_metric_comparison
    side_by_side_by_metric = {row["metric"]: row for row in side_by_side}
    assert list(side_by_side_by_metric["cagr"]) == [
        "metric",
        "metric_label",
        "strategy",
        "SPY",
        "equal_weight",
    ]
    assert side_by_side_by_metric["cagr"]["strategy"] == pytest.approx(rows["strategy"]["cagr"])
    assert side_by_side_by_metric["cagr"]["SPY"] == pytest.approx(rows["SPY"]["cagr"])
    assert side_by_side_by_metric["cagr"]["equal_weight"] == pytest.approx(
        rows["equal_weight"]["cagr"]
    )
    assert side_by_side_by_metric["cost_adjusted_cumulative_return"][
        "strategy"
    ] == pytest.approx(rows["strategy"]["cost_adjusted_cumulative_return"])
    assert side_by_side_by_metric["cost_adjusted_cumulative_return"]["SPY"] == pytest.approx(
        rows["SPY"]["cost_adjusted_cumulative_return"]
    )
    assert side_by_side_by_metric["cost_adjusted_cumulative_return"][
        "equal_weight"
    ] == pytest.approx(rows["equal_weight"]["cost_adjusted_cumulative_return"])
    assert payload["side_by_side_metric_comparison"] == side_by_side
    assert payload["metrics"]["side_by_side_metric_comparison"] == side_by_side
    assert payload["cost_adjusted_metric_comparison_side_by_side"] == side_by_side
    assert payload["side_by_side_metric_columns"]["strategy"]["cagr"] == pytest.approx(
        rows["strategy"]["cagr"]
    )
    assert payload["side_by_side_metric_columns"]["SPY"]["cagr"] == pytest.approx(
        rows["SPY"]["cagr"]
    )
    assert payload["side_by_side_metric_columns"]["equal_weight"]["cagr"] == pytest.approx(
        rows["equal_weight"]["cagr"]
    )
    assert list(payload["baseline_comparisons"]) == ["SPY", "equal_weight"]
    assert payload["baseline_comparison_entries"] == report.baseline_comparison_entries
    assert payload["metrics"]["baseline_comparisons"] == payload["baseline_comparisons"]
    assert payload["evidence"]["baseline_comparisons"] == payload["baseline_comparisons"]
    assert payload["metrics"]["model_comparison_config"]["baseline_candidate_id"] == "no_model_proxy"
    assert payload["metrics"]["model_comparison_config"]["full_model_candidate_id"] == "all_features"
    assert payload["evidence"]["model_comparison_config"] == payload["metrics"]["model_comparison_config"]

    markdown = report.to_markdown()
    assert "## Cost-Adjusted Strategy Comparison" in markdown
    assert "## Cost-Adjusted Side-by-Side Metrics" in markdown
    assert "## Baseline Comparisons" in markdown
    assert "| strategy | strategy | cost_adjusted_return |" in markdown
    assert "| SPY | baseline | cost_adjusted_benchmark_return |" in markdown
    assert "| equal_weight | baseline | cost_adjusted_equal_weight_return |" in markdown
    assert "| Metric | Strategy | SPY | Equal Weight |" in markdown
    assert "| CAGR |" in markdown
    assert "| Cost-Adjusted Cumulative Return |" in markdown
    assert "| SPY | market_benchmark | cost_adjusted_benchmark_return |" in markdown
    assert "| equal_weight | equal_weight_universe | cost_adjusted_equal_weight_return |" in markdown
    assert "Excess Return Status" in markdown
    html = report.to_html()
    assert "<h2>Cost-Adjusted Side-by-Side Metrics</h2>" in html
    assert "<th>Strategy</th>" in html
    assert "<th>SPY</th>" in html
    assert "<th>Equal Weight</th>" in html
    assert "<th>Excess Return Status</th>" in html


def test_stage1_side_by_side_metrics_align_to_source_rows_and_render_outputs(
    tmp_path,
) -> None:
    report = build_validity_gate_report(
        _comparison_predictions(),
        pd.DataFrame(),
        _comparison_equity_curve(),
        _comparison_metrics(),
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_1",
            benchmark_ticker="SPY",
            cost_bps=10.0,
            slippage_bps=5.0,
        ),
    )
    json_path, markdown_path = write_validity_gate_artifacts(report, tmp_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    comparison = payload["cost_adjusted_metric_comparison"]
    side_by_side = payload["side_by_side_metric_comparison"]
    comparison_rows = {row["name"]: row for row in comparison}
    metric_rows = {row["metric"]: row for row in side_by_side}
    expected_entities = ["strategy", "SPY", "equal_weight"]
    expected_metrics = [
        ("return_basis", "Return Basis"),
        ("cagr", "CAGR"),
        ("sharpe", "Sharpe"),
        ("max_drawdown", "Max Drawdown"),
        ("gross_cumulative_return", "Gross Cumulative Return"),
        ("cost_adjusted_cumulative_return", "Cost-Adjusted Cumulative Return"),
        ("average_daily_turnover", "Avg Daily Turnover"),
        ("transaction_cost_return", "Transaction Cost Return"),
        ("slippage_cost_return", "Slippage Cost Return"),
        ("total_cost_return", "Total Cost Return"),
        ("excess_return", "Excess Return"),
        ("excess_return_status", "Excess Return Status"),
        ("evaluation_observations", "Evaluation Observations"),
        ("evaluation_start", "Evaluation Start"),
        ("evaluation_end", "Evaluation End"),
        ("return_column", "Return Column"),
        ("return_horizon", "Return Horizon"),
    ]

    assert [row["name"] for row in comparison] == expected_entities
    assert [row["name"] for row in comparison if row["role"] == "baseline"] == [
        "SPY",
        "equal_weight",
    ]
    assert list(payload["baseline_comparisons"]) == ["SPY", "equal_weight"]
    assert [row["name"] for row in payload["baseline_comparison_entries"]] == [
        "SPY",
        "equal_weight",
    ]
    assert [row["metric"] for row in side_by_side] == [
        metric for metric, _ in expected_metrics
    ]

    for row in side_by_side:
        assert list(row) == ["metric", "metric_label", *expected_entities]

    for metric, label in expected_metrics:
        row = metric_rows[metric]
        assert row["metric_label"] == label
        for entity in expected_entities:
            assert row[entity] == comparison_rows[entity].get(metric)
            assert payload["side_by_side_metric_columns"][entity][metric] == comparison_rows[
                entity
            ].get(metric)

    for baseline_name in ("SPY", "equal_weight"):
        baseline_comparison = payload["baseline_comparisons"][baseline_name]
        assert baseline_comparison["cost_adjusted_cumulative_return"] == comparison_rows[
            baseline_name
        ]["cost_adjusted_cumulative_return"]
        assert baseline_comparison["excess_return"] == comparison_rows[baseline_name][
            "excess_return"
        ]
        assert baseline_comparison["excess_return_status"] == comparison_rows[baseline_name][
            "excess_return_status"
        ]

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "| Metric | Strategy | SPY | Equal Weight |" in markdown
    assert (
        "| Return Basis | cost_adjusted_return | cost_adjusted_benchmark_return | "
        "cost_adjusted_equal_weight_return |"
    ) in markdown
    assert (
        "| CAGR | "
        f"{comparison_rows['strategy']['cagr']:.4f} | "
        f"{comparison_rows['SPY']['cagr']:.4f} | "
        f"{comparison_rows['equal_weight']['cagr']:.4f} |"
    ) in markdown
    assert "| Evaluation Observations | 2 | 2 | 2 |" in markdown
    assert "| Return Column | forward_return_1 | forward_return_1 | forward_return_1 |" in markdown
    assert "| Return Horizon | 1 | 1 | 1 |" in markdown
    assert (
        "| SPY | market_benchmark | cost_adjusted_benchmark_return |"
        in markdown
    )
    assert (
        "| equal_weight | equal_weight_universe | cost_adjusted_equal_weight_return |"
        in markdown
    )

    html = report.to_html()
    assert "<h2>Cost-Adjusted Side-by-Side Metrics</h2>" in html
    assert "<th>Strategy</th>" in html
    assert "<th>SPY</th>" in html
    assert "<th>Equal Weight</th>" in html
    assert "<td>Return Basis</td>" in html
    assert "<td>cost_adjusted_equal_weight_return</td>" in html
    assert f"<td>{comparison_rows['strategy']['cagr']:.4f}</td>" in html
    assert "<td>Evaluation Observations</td>" in html
    assert "<td>Return Horizon</td>" in html


def test_baseline_excess_returns_use_cost_and_slippage_adjusted_returns_per_baseline() -> None:
    dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
    predictions = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[0], dates[1], dates[1], dates[1]],
            "ticker": ["SPY", "AAPL", "MSFT", "SPY", "AAPL", "MSFT"],
            "expected_return": [0.01, 0.03, 0.02, 0.01, 0.03, 0.02],
            "forward_return_1": [0.03, 0.02, 0.02, 0.0, 0.0, 0.0],
            "fold": [0, 0, 0, 1, 1, 1],
            "is_oos": [False, False, False, True, True, True],
        }
    )
    equity_curve = pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": [0.02, 0.02],
            "cost_adjusted_return": [0.02, 0.02],
            "gross_return": [0.02, 0.02],
            "transaction_cost_return": [0.0, 0.0],
            "slippage_cost_return": [0.0, 0.0],
            "total_cost_return": [0.0, 0.0],
            "benchmark_return": [0.03, 0.0],
            "turnover": [0.0, 0.0],
            "cost_bps": [20.0, 20.0],
            "slippage_bps": [30.0, 30.0],
        }
    )

    report = build_validity_gate_report(
        predictions,
        pd.DataFrame(),
        equity_curve,
        SimpleNamespace(cagr=1.0, sharpe=1.0, max_drawdown=0.0, turnover=0.0),
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_1",
            benchmark_ticker="SPY",
            cost_bps=20.0,
            slippage_bps=30.0,
        ),
    )

    strategy_return = (1.02 * 1.02) - 1.0
    baseline_rows = {
        row["name"]: row
        for row in report.cost_adjusted_metric_comparison
        if row.get("role") == "baseline"
    }
    expected = {
        "SPY": {"gross": 0.03, "cost_adjusted": 0.025},
        "equal_weight": {"gross": 0.02, "cost_adjusted": 0.015},
    }

    assert set(baseline_rows) == {"SPY", "equal_weight"}
    for name, expected_returns in expected.items():
        row = baseline_rows[name]
        gross_excess_return = strategy_return - expected_returns["gross"]
        net_excess_return = strategy_return - expected_returns["cost_adjusted"]

        assert row["gross_cumulative_return"] == pytest.approx(expected_returns["gross"])
        assert row["transaction_cost_return"] == pytest.approx(0.002)
        assert row["slippage_cost_return"] == pytest.approx(0.003)
        assert row["total_cost_return"] == pytest.approx(0.005)
        assert row["cost_adjusted_cumulative_return"] == pytest.approx(
            expected_returns["cost_adjusted"]
        )
        assert row["excess_return"] == pytest.approx(net_excess_return)
        assert row["strategy_excess_return"] == pytest.approx(net_excess_return)
        assert row["excess_return"] != pytest.approx(gross_excess_return)

        baseline_entry = report.baseline_comparisons[name]
        assert baseline_entry["cost_adjusted_cumulative_return"] == pytest.approx(
            expected_returns["cost_adjusted"]
        )
        assert baseline_entry["excess_return"] == pytest.approx(net_excess_return)


def test_validity_gate_report_uses_explicit_spy_baseline_return_series() -> None:
    dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
    predictions = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "expected_return": [0.03, 0.02, 0.03, 0.02],
            "forward_return_5": [0.01, 0.02, 0.02, 0.03],
            "fold": [0, 0, 1, 1],
            "is_oos": [False, False, True, True],
        }
    )
    equity_curve = pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": [0.01, 0.01],
            "gross_return": [0.01, 0.01],
            "cost_adjusted_return": [0.01, 0.01],
            "benchmark_return": [-0.50, -0.50],
            "turnover": [0.10, 0.10],
        }
    )
    benchmark_return_series = pd.DataFrame(
        {
            "date": dates,
            "benchmark_ticker": ["SPY", "SPY"],
            "return_column": ["forward_return_5", "forward_return_5"],
            "return_horizon": [5, 5],
            "benchmark_return": [0.02, 0.03],
        }
    )

    report = build_validity_gate_report(
        predictions,
        pd.DataFrame(),
        equity_curve,
        SimpleNamespace(cagr=0.25, sharpe=0.9, max_drawdown=-0.01, turnover=0.10),
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_5",
            benchmark_ticker="SPY",
            gap_periods=5,
            embargo_periods=5,
            cost_bps=0.0,
            slippage_bps=0.0,
        ),
        benchmark_return_series=benchmark_return_series,
    )

    spy = report.benchmark_results[0]
    expected_cagr = ((1.0 + 0.02) * (1.0 + 0.03)) ** (252 / 2) - 1.0
    assert spy["name"] == "SPY"
    assert spy["return_column"] == "forward_return_5"
    assert spy["return_horizon"] == 5
    assert spy["evaluation_observations"] == 2
    assert spy["evaluation_start"] == "2026-01-02"
    assert spy["evaluation_end"] == "2026-01-05"
    assert spy["cagr"] == pytest.approx(expected_cagr)
    assert report.baseline_comparisons["SPY"]["cagr"] == pytest.approx(expected_cagr)


def test_spy_baseline_metrics_use_strategy_evaluation_window_from_predictions() -> None:
    strategy_dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
    extra_spy_date = pd.Timestamp("2025-12-31")
    predictions = pd.DataFrame(
        {
            "date": [
                extra_spy_date,
                strategy_dates[0],
                strategy_dates[0],
                strategy_dates[0],
                strategy_dates[1],
                strategy_dates[1],
                strategy_dates[1],
            ],
            "ticker": ["SPY", "SPY", "AAPL", "MSFT", "SPY", "AAPL", "MSFT"],
            "expected_return": [0.01, 0.01, 0.03, 0.02, 0.01, 0.03, 0.02],
            "forward_return_5": [0.99, 0.02, 0.01, 0.02, 0.03, 0.02, 0.03],
        }
    )
    equity_curve = pd.DataFrame(
        {
            "date": strategy_dates,
            "portfolio_return": [0.01, 0.01],
            "cost_adjusted_return": [0.01, 0.01],
            "turnover": [0.0, 0.0],
            "realized_return_column": ["forward_return_5", "forward_return_5"],
        }
    )

    report = build_validity_gate_report(
        predictions,
        pd.DataFrame(),
        equity_curve,
        SimpleNamespace(cagr=0.25, sharpe=0.9, max_drawdown=-0.01, turnover=0.0),
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_5",
            benchmark_ticker="SPY",
            gap_periods=5,
            embargo_periods=5,
            cost_bps=0.0,
            slippage_bps=0.0,
        ),
    )

    spy = report.benchmark_results[0]
    expected_cumulative_return = (1.0 + 0.02) * (1.0 + 0.03) - 1.0
    expected_cagr = (1.0 + expected_cumulative_return) ** (252 / 2) - 1.0

    assert spy["evaluation_observations"] == 2
    assert spy["evaluation_start"] == "2026-01-02"
    assert spy["evaluation_end"] == "2026-01-05"
    assert spy["return_column"] == "forward_return_5"
    assert spy["return_horizon"] == 5
    assert spy["cost_adjusted_cumulative_return"] == pytest.approx(
        expected_cumulative_return
    )
    assert spy["cagr"] == pytest.approx(expected_cagr)
    assert report.metrics["spy_baseline_name"] == "SPY"
    assert report.metrics["spy_baseline_cagr"] == pytest.approx(expected_cagr)
    assert report.metrics["spy_baseline_cost_adjusted_cumulative_return"] == pytest.approx(
        expected_cumulative_return
    )
    assert report.metrics["spy_baseline_evaluation_observations"] == 2
    assert report.metrics["spy_baseline_evaluation_start"] == "2026-01-02"
    assert report.metrics["spy_baseline_evaluation_end"] == "2026-01-05"
    assert report.metrics["spy_baseline_return_column"] == "forward_return_5"
    assert report.metrics["spy_baseline_return_horizon"] == 5
    assert report.metrics["market_benchmark_baseline_cagr"] == pytest.approx(expected_cagr)


def test_validity_gate_report_uses_explicit_equal_weight_baseline_return_series() -> None:
    dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
    predictions = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "expected_return": [0.03, 0.02, 0.03, 0.02],
            "forward_return_5": [0.50, 0.50, 0.50, 0.50],
            "fold": [0, 0, 1, 1],
            "is_oos": [False, False, True, True],
        }
    )
    equity_curve = pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": [0.01, 0.01],
            "gross_return": [0.01, 0.01],
            "cost_adjusted_return": [0.01, 0.01],
            "benchmark_return": [0.00, 0.00],
            "turnover": [0.10, 0.10],
        }
    )
    equal_weight_baseline_return_series = pd.DataFrame(
        {
            "date": dates,
            "return_column": ["forward_return_5", "forward_return_5"],
            "return_horizon": [5, 5],
            "equal_weight_return": [0.02, 0.03],
            "constituent_count": [2, 2],
            "expected_constituent_count": [2, 2],
        }
    )

    report = build_validity_gate_report(
        predictions,
        pd.DataFrame(),
        equity_curve,
        SimpleNamespace(cagr=0.25, sharpe=0.9, max_drawdown=-0.01, turnover=0.10),
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_5",
            benchmark_ticker="SPY",
            gap_periods=5,
            embargo_periods=5,
            cost_bps=0.0,
            slippage_bps=0.0,
        ),
        equal_weight_baseline_return_series=equal_weight_baseline_return_series,
    )

    equal_weight = report.baseline_comparisons["equal_weight"]
    expected_cumulative_return = (1.0 + 0.02) * (1.0 + 0.03) - 1.0
    expected_cagr = (1.0 + expected_cumulative_return) ** (252 / 2) - 1.0

    assert equal_weight["return_column"] == "forward_return_5"
    assert equal_weight["return_horizon"] == 5
    assert equal_weight["evaluation_observations"] == 2
    assert equal_weight["evaluation_start"] == "2026-01-02"
    assert equal_weight["evaluation_end"] == "2026-01-05"
    assert equal_weight["cost_adjusted_cumulative_return"] == pytest.approx(
        expected_cumulative_return
    )
    assert equal_weight["cagr"] == pytest.approx(expected_cagr)


def test_equal_weight_baseline_metrics_filter_explicit_series_to_strategy_horizon() -> None:
    dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
    predictions = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "expected_return": [0.03, 0.02, 0.03, 0.02],
            "forward_return_5": [0.02, 0.04, 0.01, 0.05],
            "fold": [0, 0, 1, 1],
            "is_oos": [False, False, True, True],
        }
    )
    equity_curve = pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": [0.01, 0.01],
            "gross_return": [0.01, 0.01],
            "cost_adjusted_return": [0.01, 0.01],
            "benchmark_return": [0.00, 0.00],
            "realized_return_column": ["forward_return_5", "forward_return_5"],
            "turnover": [0.10, 0.10],
        }
    )
    equal_weight_baseline_return_series = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "return_column": [
                "forward_return_5",
                "forward_return_1",
                "forward_return_5",
                "forward_return_1",
            ],
            "return_horizon": [5, 1, 5, 1],
            "equal_weight_return": [0.02, 0.90, 0.03, 0.80],
            "constituent_count": [2, 2, 2, 2],
            "expected_constituent_count": [2, 2, 2, 2],
        }
    )

    report = build_validity_gate_report(
        predictions,
        pd.DataFrame(),
        equity_curve,
        SimpleNamespace(cagr=0.25, sharpe=0.9, max_drawdown=-0.01, turnover=0.10),
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_5",
            benchmark_ticker="SPY",
            gap_periods=5,
            embargo_periods=5,
            cost_bps=0.0,
            slippage_bps=0.0,
        ),
        equal_weight_baseline_return_series=equal_weight_baseline_return_series,
    )

    equal_weight = report.baseline_comparisons["equal_weight"]
    expected_cumulative_return = (1.0 + 0.02) * (1.0 + 0.03) - 1.0
    expected_cagr = (1.0 + expected_cumulative_return) ** (252 / 2) - 1.0

    assert equal_weight["return_column"] == "forward_return_5"
    assert equal_weight["return_horizon"] == 5
    assert equal_weight["evaluation_observations"] == 2
    assert equal_weight["cost_adjusted_cumulative_return"] == pytest.approx(
        expected_cumulative_return
    )
    assert equal_weight["cagr"] == pytest.approx(expected_cagr)
    assert report.metrics["equal_weight_baseline_cagr"] == pytest.approx(expected_cagr)


def test_explicit_baseline_series_are_aligned_to_strategy_evaluation_dates() -> None:
    strategy_dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
    extra_dates = pd.to_datetime(["2026-01-01", "2026-01-06"])
    all_baseline_dates = pd.DatetimeIndex(
        [extra_dates[0], strategy_dates[0], strategy_dates[1], extra_dates[1]]
    )
    predictions = pd.DataFrame(
        {
            "date": [strategy_dates[0], strategy_dates[0], strategy_dates[1], strategy_dates[1]],
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "expected_return": [0.03, 0.02, 0.03, 0.02],
            "forward_return_5": [0.04, 0.01, 0.02, 0.03],
            "fold": [0, 0, 1, 1],
            "is_oos": [False, False, True, True],
        }
    )
    equity_curve = pd.DataFrame(
        {
            "date": strategy_dates,
            "portfolio_return": [0.03, 0.03],
            "gross_return": [0.03, 0.03],
            "cost_adjusted_return": [0.03, 0.03],
            "benchmark_return": [-0.50, -0.50],
            "turnover": [0.10, 0.10],
            "realized_return_column": ["forward_return_5", "forward_return_5"],
        }
    )
    benchmark_return_series = pd.DataFrame(
        {
            "date": all_baseline_dates,
            "benchmark_ticker": ["SPY"] * len(all_baseline_dates),
            "return_column": ["forward_return_5"] * len(all_baseline_dates),
            "return_horizon": [5] * len(all_baseline_dates),
            "benchmark_return": [0.99, 0.02, 0.03, 0.99],
        }
    )
    equal_weight_baseline_return_series = pd.DataFrame(
        {
            "date": all_baseline_dates,
            "return_column": ["forward_return_5"] * len(all_baseline_dates),
            "return_horizon": [5] * len(all_baseline_dates),
            "equal_weight_return": [0.99, 0.04, 0.01, 0.99],
            "constituent_count": [2, 2, 2, 2],
            "expected_constituent_count": [2, 2, 2, 2],
        }
    )

    report = build_validity_gate_report(
        predictions,
        pd.DataFrame(),
        equity_curve,
        SimpleNamespace(cagr=1.0, sharpe=1.0, max_drawdown=-0.01, turnover=0.10),
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_5",
            benchmark_ticker="SPY",
            gap_periods=5,
            embargo_periods=5,
            cost_bps=0.0,
            slippage_bps=0.0,
        ),
        benchmark_return_series=benchmark_return_series,
        equal_weight_baseline_return_series=equal_weight_baseline_return_series,
    )

    comparison_rows = {row["name"]: row for row in report.cost_adjusted_metric_comparison}
    for baseline_name in ("SPY", "equal_weight"):
        row = comparison_rows[baseline_name]
        assert row["evaluation_observations"] == len(strategy_dates)
        assert row["evaluation_start"] == "2026-01-02"
        assert row["evaluation_end"] == "2026-01-05"
        assert row["return_column"] == "forward_return_5"
        assert row["return_horizon"] == 5

    assert comparison_rows["SPY"]["cost_adjusted_cumulative_return"] == pytest.approx(
        (1.0 + 0.02) * (1.0 + 0.03) - 1.0
    )
    assert comparison_rows["equal_weight"]["cost_adjusted_cumulative_return"] == pytest.approx(
        (1.0 + 0.04) * (1.0 + 0.01) - 1.0
    )


def test_strategy_candidate_passes_when_all_required_baseline_excess_returns_pass(
    tmp_path,
) -> None:
    dates = pd.date_range("2026-01-02", periods=5, freq="B")
    predictions = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "fold": fold,
                "is_oos": fold == len(dates) - 1,
                "expected_return": expected_return,
                "forward_return_5": realized_return,
            }
            for fold, date in enumerate(dates)
            for ticker, expected_return, realized_return in (
                ("AAPL", 0.03, 0.010),
                ("MSFT", 0.02, 0.008),
                ("SPY", 0.01, 0.001),
            )
        ]
    )
    validation_summary = pd.DataFrame(
        {
            "fold": range(len(dates)),
            "train_end": dates - pd.Timedelta(days=7),
            "test_start": dates,
            "is_oos": [False, False, False, False, True],
            "labeled_test_observations": [3] * len(dates),
            "train_observations": [60] * len(dates),
        }
    )
    equity_curve = pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": [0.012] * len(dates),
            "gross_return": [0.012] * len(dates),
            "cost_adjusted_return": [0.012] * len(dates),
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
        SimpleNamespace(cagr=2.0, sharpe=1.2, max_drawdown=-0.05, turnover=0.10),
        ablation_summary=ablation_summary,
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_5",
            benchmark_ticker="SPY",
            gap_periods=5,
            embargo_periods=5,
            cost_bps=0.0,
            slippage_bps=0.0,
        ),
    )

    cost_gate = report.gate_results["cost_adjusted_performance"]
    baseline_excess_returns = {
        name: row["strategy_excess_return"]
        for name, row in report.baseline_comparisons.items()
    }
    assert baseline_excess_returns["SPY"] > 0
    assert baseline_excess_returns["equal_weight"] > 0
    assert cost_gate["status"] == "pass"
    assert cost_gate["reason"] == "net excess return is positive versus all required baselines"
    assert cost_gate["operator"] == "excess_return > 0 for every required baseline"
    assert cost_gate["required_baselines"] == ["SPY", "equal_weight"]
    assert cost_gate["passed_baselines"] == ["SPY", "equal_weight"]
    assert cost_gate["failed_baselines"] == []
    assert cost_gate["baseline_excess_returns"]["SPY"] == pytest.approx(
        baseline_excess_returns["SPY"]
    )
    assert cost_gate["baseline_excess_returns"]["equal_weight"] == pytest.approx(
        baseline_excess_returns["equal_weight"]
    )
    assert cost_gate["baseline_excess_return_statuses"] == {
        "SPY": "pass",
        "equal_weight": "pass",
    }
    validity_gate = report.gate_results["deterministic_strategy_validity"]
    assert validity_gate["status"] == "pass"
    assert validity_gate["all_required_outperformance_rules_passed"] is True
    assert validity_gate["failed_rules"] == []
    assert validity_gate["rule_statuses"]["cost_adjusted_performance"] == "pass"
    assert report.baseline_comparisons["SPY"]["excess_return_status"] == "pass"
    assert report.baseline_comparisons["equal_weight"]["excess_return_status"] == "pass"
    assert report.system_validity_status == "pass"
    assert report.strategy_candidate_status == "pass"
    assert report.warning is False
    assert report.strategy_pass is True

    json_path, markdown_path = write_validity_gate_artifacts(report, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert payload["strategy_candidate_status"] == "pass"
    assert payload["gate_results"]["cost_adjusted_performance"]["status"] == "pass"
    assert payload["gate_results"]["cost_adjusted_performance"]["passed_baselines"] == [
        "SPY",
        "equal_weight",
    ]
    assert payload["gate_results"]["cost_adjusted_performance"]["failed_baselines"] == []
    assert payload["gate_results"]["cost_adjusted_performance"][
        "baseline_excess_return_statuses"
    ] == {"SPY": "pass", "equal_weight": "pass"}
    assert payload["gate_results"]["deterministic_strategy_validity"]["status"] == "pass"
    assert payload["metrics"]["deterministic_strategy_validity"] == payload["gate_results"][
        "deterministic_strategy_validity"
    ]
    assert payload["evidence"]["deterministic_strategy_validity"] == payload[
        "gate_results"
    ]["deterministic_strategy_validity"]
    assert payload["baseline_comparisons"]["SPY"]["excess_return_status"] == "pass"
    assert payload["baseline_comparisons"]["equal_weight"]["excess_return_status"] == "pass"
    assert payload["warning"] is False
    assert payload["strategy_pass"] is True
    assert "- Strategy candidate: `pass`" in markdown
    assert (
        "| cost_adjusted_performance | pass | "
        "net excess return is positive versus all required baselines |"
    ) in markdown
    assert (
        "| deterministic_strategy_validity | pass | "
        "all required deterministic outperformance rules passed |"
    ) in markdown


def test_strategy_candidate_fails_and_records_failed_baseline_when_one_baseline_is_missed(
    tmp_path,
) -> None:
    dates = pd.date_range("2026-01-02", periods=5, freq="B")
    predictions = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "fold": fold,
                "is_oos": fold == len(dates) - 1,
                "expected_return": expected_return,
                "forward_return_5": realized_return,
            }
            for fold, date in enumerate(dates)
            for ticker, expected_return, realized_return in (
                ("AAPL", 0.03, 0.010),
                ("MSFT", 0.02, 0.010),
                ("SPY", 0.01, -0.001),
            )
        ]
    )
    validation_summary = pd.DataFrame(
        {
            "fold": range(len(dates)),
            "train_end": dates - pd.Timedelta(days=7),
            "test_start": dates,
            "is_oos": [False, False, False, False, True],
            "labeled_test_observations": [3] * len(dates),
            "train_observations": [60] * len(dates),
        }
    )
    equity_curve = pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": [0.002] * len(dates),
            "gross_return": [0.002] * len(dates),
            "cost_adjusted_return": [0.002] * len(dates),
            "benchmark_return": [-0.001] * len(dates),
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
        SimpleNamespace(cagr=2.0, sharpe=1.2, max_drawdown=-0.05, turnover=0.10),
        ablation_summary=ablation_summary,
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_5",
            benchmark_ticker="SPY",
            gap_periods=5,
            embargo_periods=5,
            cost_bps=0.0,
            slippage_bps=0.0,
        ),
    )

    cost_gate = report.gate_results["cost_adjusted_performance"]
    baseline_excess_returns = {
        name: row["strategy_excess_return"]
        for name, row in report.baseline_comparisons.items()
    }
    assert baseline_excess_returns["SPY"] > 0
    assert baseline_excess_returns["equal_weight"] < 0
    assert cost_gate["status"] == "fail"
    assert cost_gate["reason"] == (
        "net excess return is not positive versus required baseline(s): equal_weight"
    )
    assert cost_gate["operator"] == "excess_return > 0 for every required baseline"
    assert cost_gate["required_baselines"] == ["SPY", "equal_weight"]
    assert cost_gate["passed_baselines"] == ["SPY"]
    assert cost_gate["failed_baselines"] == ["equal_weight"]
    assert cost_gate["baseline_excess_returns"]["SPY"] == pytest.approx(
        baseline_excess_returns["SPY"]
    )
    assert cost_gate["baseline_excess_returns"]["equal_weight"] == pytest.approx(
        baseline_excess_returns["equal_weight"]
    )
    assert cost_gate["baseline_excess_return_statuses"] == {
        "SPY": "pass",
        "equal_weight": "fail",
    }
    validity_gate = report.gate_results["deterministic_strategy_validity"]
    assert validity_gate["status"] == "fail"
    assert validity_gate["all_required_outperformance_rules_passed"] is False
    assert validity_gate["failed_rules"] == ["cost_adjusted_performance"]
    assert validity_gate["rule_statuses"]["cost_adjusted_performance"] == "fail"
    assert validity_gate["reason_code"] == "required_outperformance_rule_not_passed"
    assert report.baseline_comparisons["SPY"]["excess_return_status"] == "pass"
    assert report.baseline_comparisons["equal_weight"]["excess_return_status"] == "fail"
    assert report.system_validity_status == "pass"
    assert report.strategy_candidate_status == "fail"
    assert report.warning is False
    assert report.strategy_pass is False

    json_path, markdown_path = write_validity_gate_artifacts(report, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert payload["strategy_candidate_status"] == "fail"
    assert payload["gate_results"]["cost_adjusted_performance"]["status"] == "fail"
    assert payload["gate_results"]["cost_adjusted_performance"]["failed_baselines"] == [
        "equal_weight",
    ]
    assert payload["gate_results"]["cost_adjusted_performance"][
        "baseline_excess_return_statuses"
    ] == {"SPY": "pass", "equal_weight": "fail"}
    assert payload["gate_results"]["deterministic_strategy_validity"]["status"] == "fail"
    assert payload["gate_results"]["deterministic_strategy_validity"]["failed_rules"] == [
        "cost_adjusted_performance",
    ]
    assert payload["baseline_comparisons"]["SPY"]["excess_return_status"] == "pass"
    assert payload["baseline_comparisons"]["equal_weight"]["excess_return_status"] == "fail"
    assert payload["warning"] is False
    assert payload["strategy_pass"] is False
    assert "- Strategy candidate: `fail`" in markdown
    assert (
        "| cost_adjusted_performance | fail | "
        "net excess return is not positive versus required baseline(s): equal_weight |"
    ) in markdown
    assert (
        "| deterministic_strategy_validity | fail | "
        "required deterministic outperformance rule(s) did not pass: "
        "cost_adjusted_performance |"
    ) in markdown


def test_strategy_candidate_fails_exactly_spy_baseline_while_equal_weight_passes(
    tmp_path,
) -> None:
    dates = pd.date_range("2026-01-02", periods=5, freq="B")
    predictions = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "fold": fold,
                "is_oos": fold == len(dates) - 1,
                "expected_return": expected_return,
                "forward_return_5": realized_return,
            }
            for fold, date in enumerate(dates)
            for ticker, expected_return, realized_return in (
                ("AAPL", 0.03, 0.003),
                ("MSFT", 0.02, 0.001),
                ("SPY", 0.01, 0.000),
            )
        ]
    )
    validation_summary = pd.DataFrame(
        {
            "fold": range(len(dates)),
            "train_end": dates - pd.Timedelta(days=7),
            "test_start": dates,
            "is_oos": [False, False, False, False, True],
            "labeled_test_observations": [3] * len(dates),
            "train_observations": [60] * len(dates),
        }
    )
    equity_curve = pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": [0.004] * len(dates),
            "gross_return": [0.004] * len(dates),
            "cost_adjusted_return": [0.004] * len(dates),
            "benchmark_return": [0.006] * len(dates),
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
        SimpleNamespace(cagr=2.0, sharpe=1.2, max_drawdown=-0.05, turnover=0.10),
        ablation_summary=ablation_summary,
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_5",
            benchmark_ticker="SPY",
            gap_periods=5,
            embargo_periods=5,
            cost_bps=0.0,
            slippage_bps=0.0,
        ),
    )

    cost_gate = report.gate_results["cost_adjusted_performance"]
    baseline_excess_returns = {
        name: row["strategy_excess_return"]
        for name, row in report.baseline_comparisons.items()
    }
    assert baseline_excess_returns["SPY"] < 0
    assert baseline_excess_returns["equal_weight"] > 0
    assert cost_gate["status"] == "fail"
    assert cost_gate["reason"] == (
        "net excess return is not positive versus required baseline(s): SPY"
    )
    assert cost_gate["operator"] == "excess_return > 0 for every required baseline"
    assert cost_gate["required_baselines"] == ["SPY", "equal_weight"]
    assert cost_gate["passed_baselines"] == ["equal_weight"]
    assert cost_gate["failed_baselines"] == ["SPY"]
    assert len(cost_gate["passed_baselines"]) == 1
    assert len(cost_gate["failed_baselines"]) == 1
    assert cost_gate["baseline_excess_returns"]["SPY"] == pytest.approx(
        baseline_excess_returns["SPY"]
    )
    assert cost_gate["baseline_excess_returns"]["equal_weight"] == pytest.approx(
        baseline_excess_returns["equal_weight"]
    )
    assert cost_gate["baseline_excess_return_statuses"] == {
        "SPY": "fail",
        "equal_weight": "pass",
    }
    assert report.baseline_comparisons["SPY"]["excess_return_status"] == "fail"
    assert report.baseline_comparisons["equal_weight"]["excess_return_status"] == "pass"
    assert report.system_validity_status == "pass"
    assert report.strategy_candidate_status == "fail"
    assert report.warning is False
    assert report.strategy_pass is False

    json_path, markdown_path = write_validity_gate_artifacts(report, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert payload["strategy_candidate_status"] == "fail"
    assert payload["gate_results"]["cost_adjusted_performance"]["status"] == "fail"
    assert payload["gate_results"]["cost_adjusted_performance"]["passed_baselines"] == [
        "equal_weight",
    ]
    assert payload["gate_results"]["cost_adjusted_performance"]["failed_baselines"] == [
        "SPY",
    ]
    assert payload["gate_results"]["cost_adjusted_performance"][
        "baseline_excess_return_statuses"
    ] == {"SPY": "fail", "equal_weight": "pass"}
    assert payload["baseline_comparisons"]["SPY"]["excess_return_status"] == "fail"
    assert payload["baseline_comparisons"]["equal_weight"]["excess_return_status"] == "pass"
    assert payload["warning"] is False
    assert payload["strategy_pass"] is False
    assert "- Strategy candidate: `fail`" in markdown
    assert (
        "| cost_adjusted_performance | fail | "
        "net excess return is not positive versus required baseline(s): SPY |"
    ) in markdown


def test_strategy_candidate_fails_when_baseline_excess_return_is_zero() -> None:
    dates = pd.date_range("2026-01-02", periods=5, freq="B")
    predictions = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "fold": fold,
                "is_oos": fold == len(dates) - 1,
                "expected_return": expected_return,
                "forward_return_5": realized_return,
            }
            for fold, date in enumerate(dates)
            for ticker, expected_return, realized_return in (
                ("AAPL", 0.03, 0.002),
                ("MSFT", 0.02, 0.002),
                ("SPY", 0.01, -0.001),
            )
        ]
    )
    validation_summary = pd.DataFrame(
        {
            "fold": range(len(dates)),
            "train_end": dates - pd.Timedelta(days=7),
            "test_start": dates,
            "is_oos": [False, False, False, False, True],
            "labeled_test_observations": [3] * len(dates),
            "train_observations": [60] * len(dates),
        }
    )
    equity_curve = pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": [0.002] * len(dates),
            "gross_return": [0.002] * len(dates),
            "cost_adjusted_return": [0.002] * len(dates),
            "benchmark_return": [-0.001] * len(dates),
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
        SimpleNamespace(cagr=2.0, sharpe=1.2, max_drawdown=-0.05, turnover=0.10),
        ablation_summary=ablation_summary,
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_5",
            benchmark_ticker="SPY",
            gap_periods=5,
            embargo_periods=5,
            cost_bps=0.0,
            slippage_bps=0.0,
        ),
    )

    cost_gate = report.gate_results["cost_adjusted_performance"]
    baseline_excess_returns = {
        name: row["strategy_excess_return"]
        for name, row in report.baseline_comparisons.items()
    }
    assert baseline_excess_returns["SPY"] > 0
    assert baseline_excess_returns["equal_weight"] == pytest.approx(0.0)
    assert cost_gate["status"] == "fail"
    assert cost_gate["passed_baselines"] == ["SPY"]
    assert cost_gate["failed_baselines"] == ["equal_weight"]
    assert cost_gate["baseline_excess_return_statuses"] == {
        "SPY": "pass",
        "equal_weight": "fail",
    }
    assert report.strategy_candidate_status == "fail"


def test_strategy_candidate_fails_when_both_baselines_are_missed(tmp_path) -> None:
    dates = pd.date_range("2026-01-02", periods=5, freq="B")
    predictions = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "fold": fold,
                "is_oos": fold == len(dates) - 1,
                "expected_return": expected_return,
                "forward_return_5": realized_return,
            }
            for fold, date in enumerate(dates)
            for ticker, expected_return, realized_return in (
                ("AAPL", 0.03, 0.006),
                ("MSFT", 0.02, 0.004),
                ("SPY", 0.01, 0.002),
            )
        ]
    )
    validation_summary = pd.DataFrame(
        {
            "fold": range(len(dates)),
            "train_end": dates - pd.Timedelta(days=7),
            "test_start": dates,
            "is_oos": [False, False, False, False, True],
            "labeled_test_observations": [3] * len(dates),
            "train_observations": [60] * len(dates),
        }
    )
    equity_curve = pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": [0.0001] * len(dates),
            "gross_return": [0.0001] * len(dates),
            "cost_adjusted_return": [0.0001] * len(dates),
            "benchmark_return": [0.002] * len(dates),
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
        SimpleNamespace(cagr=0.10, sharpe=1.2, max_drawdown=-0.05, turnover=0.10),
        ablation_summary=ablation_summary,
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_5",
            benchmark_ticker="SPY",
            gap_periods=5,
            embargo_periods=5,
            cost_bps=0.0,
            slippage_bps=0.0,
        ),
    )

    cost_gate = report.gate_results["cost_adjusted_performance"]
    baseline_excess_returns = {
        name: row["strategy_excess_return"]
        for name, row in report.baseline_comparisons.items()
    }
    assert baseline_excess_returns["SPY"] < 0
    assert baseline_excess_returns["equal_weight"] < 0
    assert cost_gate["status"] == "fail"
    assert cost_gate["reason"] == (
        "net excess return is not positive versus required baseline(s): SPY, equal_weight"
    )
    assert cost_gate["operator"] == "excess_return > 0 for every required baseline"
    assert cost_gate["required_baselines"] == ["SPY", "equal_weight"]
    assert cost_gate["passed_baselines"] == []
    assert cost_gate["failed_baselines"] == ["SPY", "equal_weight"]
    assert set(cost_gate["baseline_excess_returns"]) == {"SPY", "equal_weight"}
    assert cost_gate["baseline_excess_returns"]["SPY"] == pytest.approx(
        baseline_excess_returns["SPY"]
    )
    assert cost_gate["baseline_excess_returns"]["equal_weight"] == pytest.approx(
        baseline_excess_returns["equal_weight"]
    )
    assert cost_gate["baseline_excess_return_statuses"] == {
        "SPY": "fail",
        "equal_weight": "fail",
    }
    assert report.system_validity_status == "pass"
    assert report.strategy_candidate_status == "fail"
    assert report.warning is False
    assert report.strategy_pass is False

    json_path, markdown_path = write_validity_gate_artifacts(report, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert payload["strategy_candidate_status"] == "fail"
    assert payload["gate_results"]["cost_adjusted_performance"]["status"] == "fail"
    assert payload["gate_results"]["cost_adjusted_performance"]["operator"] == (
        "excess_return > 0 for every required baseline"
    )
    assert payload["gate_results"]["cost_adjusted_performance"]["required_baselines"] == [
        "SPY",
        "equal_weight",
    ]
    assert payload["gate_results"]["cost_adjusted_performance"]["passed_baselines"] == []
    assert payload["gate_results"]["cost_adjusted_performance"]["failed_baselines"] == [
        "SPY",
        "equal_weight",
    ]
    assert payload["gate_results"]["cost_adjusted_performance"][
        "baseline_excess_return_statuses"
    ] == {"SPY": "fail", "equal_weight": "fail"}
    assert payload["baseline_comparisons"]["SPY"]["excess_return_status"] == "fail"
    assert payload["baseline_comparisons"]["equal_weight"]["excess_return_status"] == "fail"
    assert payload["warning"] is False
    assert payload["strategy_pass"] is False
    assert "- Strategy candidate: `fail`" in markdown
    assert (
        "| cost_adjusted_performance | fail | "
        "net excess return is not positive versus required baseline(s): SPY, equal_weight |"
    ) in markdown


def test_validity_gate_report_preserves_stage1_baseline_comparison_inputs() -> None:
    baseline_inputs = (
        BaselineComparisonInput(
            name="SPY",
            baseline_type=MARKET_BENCHMARK_BASELINE_TYPE,
            return_basis="cost_adjusted_benchmark_return",
            return_column="forward_return_1",
            return_horizon=1,
            data_source="benchmark_return_series",
            benchmark_ticker="SPY",
            cost_bps=10.0,
            slippage_bps=5.0,
        ),
        BaselineComparisonInput(
            name="equal_weight",
            baseline_type=EQUAL_WEIGHT_BASELINE_TYPE,
            return_basis="cost_adjusted_equal_weight_return",
            return_column="forward_return_1",
            return_horizon=1,
            data_source="equal_weight_baseline_return_series",
            universe_tickers=("AAPL", "MSFT"),
            cost_bps=10.0,
            slippage_bps=5.0,
        ),
    )

    report = build_validity_gate_report(
        _comparison_predictions(),
        pd.DataFrame(),
        _comparison_equity_curve(),
        _comparison_metrics(),
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_1",
            benchmark_ticker="SPY",
            cost_bps=10.0,
            slippage_bps=5.0,
        ),
        baseline_comparison_inputs=baseline_inputs,
    )

    payload = report.to_dict()
    assert [row["name"] for row in payload["baseline_comparison_inputs"]] == [
        "SPY",
        "equal_weight",
    ]
    assert [row["baseline_type"] for row in payload["baseline_comparison_inputs"]] == [
        MARKET_BENCHMARK_BASELINE_TYPE,
        EQUAL_WEIGHT_BASELINE_TYPE,
    ]
    assert payload["metrics"]["baseline_comparison_inputs"] == payload["baseline_comparison_inputs"]
    assert payload["evidence"]["baseline_comparison_inputs"] == payload["baseline_comparison_inputs"]
    assert payload["gate_results"]["baseline_inputs"]["status"] == "pass"

    markdown = report.to_markdown()
    assert "## Baseline Comparison Inputs" in markdown
    assert "| SPY | market_benchmark | cost_adjusted_benchmark_return | benchmark_return_series |" in markdown
    assert (
        "| equal_weight | equal_weight_universe | cost_adjusted_equal_weight_return | "
        "equal_weight_baseline_return_series |"
    ) in markdown


def test_validity_gate_report_exports_no_model_proxy_ablation_output_fields(tmp_path) -> None:
    no_model_proxy_row = {
        "scenario": "no_model_proxy",
        "kind": "pipeline_control",
        "label": "No model proxy",
        "description": "Walk-forward model refit without optional proxy features.",
        "pipeline_controls": {
            "model_proxy": False,
            "cost": True,
            "slippage": True,
            "turnover": True,
        },
        "toggles": {
            "include_model_proxy_features": False,
            "include_chronos_features": False,
            "include_granite_ttm_features": False,
        },
        "feature_sources": {"price": True, "text": True, "sec": True},
        "permitted_feature_families": ["price", "text", "sec"],
        "input_feature_families": ["price", "text", "sec"],
        "input_feature_columns": ["return_1", "news_sentiment_mean", "sec_risk_flag"],
        "effective_cost_bps": 5.0,
        "effective_slippage_bps": 2.0,
        "cagr": 0.12,
        "sharpe": 0.7,
        "max_drawdown": -0.08,
        "turnover": 0.18,
        "excess_return": 0.04,
        "validation_status": "pass",
        "validation_fold_count": 5,
        "validation_oos_fold_count": 1,
        "validation_prediction_count": 120,
        "validation_labeled_prediction_count": 115,
        "validation_mean_mae": 0.02,
        "validation_mean_directional_accuracy": 0.56,
        "validation_mean_information_coefficient": 0.04,
        "validation_positive_ic_fold_ratio": 0.8,
        "validation_oos_information_coefficient": 0.03,
        "deterministic_signal_evaluation_metrics": {
            "engine": "deterministic_signal_engine",
            "return_basis": "cost_adjusted_return",
            "action_counts": {"BUY": 2, "SELL": 1, "HOLD": 3},
            "cost_adjusted_cumulative_return": 0.05,
            "average_daily_turnover": 0.18,
            "total_cost_return": 0.003,
        },
    }
    ablation_summary = [
        {"scenario": "all_features", "sharpe": 0.9, "excess_return": 0.06},
        {"scenario": "price_only", "sharpe": 0.3, "excess_return": 0.02},
        {"scenario": "text_only", "sharpe": 0.2, "excess_return": 0.01},
        {"scenario": "sec_only", "sharpe": 0.1, "excess_return": 0.01},
        no_model_proxy_row,
        {"scenario": "no_costs", "sharpe": 0.8, "excess_return": 0.07},
    ]

    report = build_validity_gate_report(
        _comparison_predictions(),
        pd.DataFrame(),
        _comparison_equity_curve(),
        _comparison_metrics(),
        ablation_summary=ablation_summary,
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_1",
            benchmark_ticker="SPY",
            cost_bps=10.0,
            slippage_bps=5.0,
        ),
    )

    no_model_proxy = report.no_model_proxy_ablation
    assert no_model_proxy["available"] is True
    assert no_model_proxy["model_proxy_enabled"] is False
    assert no_model_proxy["pipeline_controls"]["model_proxy"] is False
    assert no_model_proxy["performance_metrics"]["sharpe"] == pytest.approx(0.7)
    assert no_model_proxy["validation_metrics"]["validation_status"] == "pass"
    assert no_model_proxy["validation_metrics"]["validation_oos_fold_count"] == 1
    assert no_model_proxy["deterministic_signal_evaluation_metrics"]["action_counts"] == {
        "BUY": 2,
        "SELL": 1,
        "HOLD": 3,
    }

    payload = report.to_dict()
    assert payload["no_model_proxy_ablation"] == no_model_proxy
    assert payload["metrics"]["no_model_proxy_ablation"] == no_model_proxy
    assert payload["evidence"]["no_model_proxy_ablation"] == no_model_proxy

    json_path, markdown_path = write_validity_gate_artifacts(report, tmp_path)
    artifact = json.loads(json_path.read_text(encoding="utf-8"))
    assert artifact["no_model_proxy_ablation"]["performance_metrics"]["excess_return"] == pytest.approx(0.04)
    assert artifact["no_model_proxy_ablation"]["validation_metrics"]["validation_fold_count"] == 5

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## No-Model-Proxy Ablation" in markdown
    assert "| Model proxy enabled | False |" in markdown
    assert "| Buy / Sell / Hold | 2 / 1 / 3 |" in markdown


def test_stage1_output_payload_reports_model_comparison_candidate_baseline_delta_and_pass_fail(
    tmp_path,
) -> None:
    full_model_row = _stage1_required_ablation_row(
        "all_features",
        "signal",
        "All features",
        sharpe=0.75,
        excess_return=0.080,
        cost_adjusted_return=0.120,
    )
    baseline_row = _stage1_required_ablation_row(
        "no_model_proxy",
        "pipeline_control",
        "No model proxy",
        sharpe=0.50,
        excess_return=0.045,
        cost_adjusted_return=0.090,
        feature_families=["price", "text", "sec"],
        model_proxy_enabled=False,
    )
    full_model_row.update(
        {
            "turnover": 0.30,
            "validation_mean_information_coefficient": 0.060,
            "validation_positive_ic_fold_ratio": 0.80,
            "validation_oos_information_coefficient": 0.050,
            "signal_average_daily_turnover": 0.30,
        }
    )
    full_model_row["deterministic_signal_evaluation_metrics"][
        "average_daily_turnover"
    ] = 0.30
    baseline_row.update(
        {
            "turnover": 0.10,
            "validation_mean_information_coefficient": 0.035,
            "validation_positive_ic_fold_ratio": 0.60,
            "validation_oos_information_coefficient": 0.020,
            "signal_average_daily_turnover": 0.10,
        }
    )
    baseline_row["deterministic_signal_evaluation_metrics"][
        "average_daily_turnover"
    ] = 0.10
    ablation_summary = [
        full_model_row,
        _stage1_required_ablation_row(
            "price_only",
            "data_channel",
            "Price only",
            sharpe=0.40,
            excess_return=0.020,
            cost_adjusted_return=0.040,
        ),
        _stage1_required_ablation_row(
            "text_only",
            "data_channel",
            "Text only",
            sharpe=0.30,
            excess_return=0.010,
            cost_adjusted_return=0.030,
            feature_families=["text"],
        ),
        _stage1_required_ablation_row(
            "sec_only",
            "data_channel",
            "SEC only",
            sharpe=0.20,
            excess_return=0.005,
            cost_adjusted_return=0.020,
            feature_families=["sec"],
        ),
        baseline_row,
        _stage1_required_ablation_row(
            "no_costs",
            "cost",
            "No costs",
            sharpe=0.90,
            excess_return=0.100,
            cost_adjusted_return=0.150,
            cost_enabled=False,
            slippage_enabled=False,
            turnover_enabled=False,
            effective_cost_bps=0.0,
            effective_slippage_bps=0.0,
        ),
    ]

    report = build_validity_gate_report(
        _comparison_predictions(),
        pd.DataFrame(),
        _comparison_equity_curve(),
        _comparison_metrics(),
        ablation_summary=ablation_summary,
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_1",
            benchmark_ticker="SPY",
            cost_bps=10.0,
            slippage_bps=5.0,
        ),
    )

    json_path, _ = write_validity_gate_artifacts(report, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    comparison_results = payload["model_comparison_results"]
    assert payload["metrics"]["model_comparison_results"] == comparison_results
    assert payload["evidence"]["model_comparison_results"] == comparison_results

    expected_metrics = payload["metrics"]["model_comparison_config"]["metrics"]
    assert {
        (row["baseline"], row["baseline_role"])
        for row in comparison_results
    } == {
        ("no_model_proxy", "model_baseline"),
        ("return_baseline_spy", "return_baseline"),
        ("return_baseline_equal_weight", "return_baseline"),
        ("price_only", "ablation"),
        ("text_only", "ablation"),
        ("sec_only", "ablation"),
        ("no_costs", "diagnostic"),
    }
    rows = {
        row["metric"]: row
        for row in comparison_results
        if row["baseline"] == "no_model_proxy"
    }
    assert list(rows) == expected_metrics
    for row in comparison_results:
        assert {
            "window_id",
            "window_label",
            "window_role",
            "candidate",
            "candidate_id",
            "candidate_value",
            "baseline",
            "baseline_id",
            "baseline_label",
            "baseline_role",
            "baseline_value",
            "delta",
            "absolute_delta",
            "relative_delta",
            "improvement",
            "outperformance_threshold",
            "operator",
            "pass_fail",
            "status",
            "passed",
        }.issubset(row)
        assert row["candidate"] == "all_features"
        assert row["candidate_id"] == "all_features"
        assert row["window_id"] == "strategy_evaluation"
        assert row["baseline_id"] == row["baseline"]
        if row["candidate_value"] is not None and row["baseline_value"] is not None:
            assert row["delta"] == pytest.approx(
                row["candidate_value"] - row["baseline_value"]
            )
            assert row["absolute_delta"] == pytest.approx(row["delta"])
            if row["baseline_value"] != 0:
                assert row["relative_delta"] == pytest.approx(
                    row["absolute_delta"] / abs(row["baseline_value"])
                )
            assert row["pass_fail"] in {"pass", "fail"}
            assert isinstance(row["passed"], bool)
        else:
            assert row["delta"] is None
            assert row["absolute_delta"] is None
            assert row["relative_delta"] is None
            assert row["pass_fail"] == "not_evaluable"
            assert row["passed"] is None
        assert row["status"] == row["pass_fail"]

    assert rows["sharpe"]["candidate_value"] == pytest.approx(0.75)
    assert rows["sharpe"]["baseline_value"] == pytest.approx(0.50)
    assert rows["sharpe"]["delta"] == pytest.approx(0.25)
    assert rows["sharpe"]["absolute_delta"] == pytest.approx(0.25)
    assert rows["sharpe"]["relative_delta"] == pytest.approx(0.50)
    assert rows["sharpe"]["improvement"] == pytest.approx(0.25)
    assert rows["sharpe"]["outperformance_threshold"] == pytest.approx(0.05)
    assert rows["sharpe"]["operator"] == "candidate - baseline > 0.05"
    assert rows["sharpe"]["pass_fail"] == "pass"

    assert rows["turnover"]["candidate_value"] == pytest.approx(0.30)
    assert rows["turnover"]["baseline_value"] == pytest.approx(0.10)
    assert rows["turnover"]["delta"] == pytest.approx(0.20)
    assert rows["turnover"]["absolute_delta"] == pytest.approx(0.20)
    assert rows["turnover"]["relative_delta"] == pytest.approx(2.0)
    assert rows["turnover"]["improvement"] == pytest.approx(-0.20)
    assert rows["turnover"]["outperformance_threshold"] == pytest.approx(0.0)
    assert rows["turnover"]["operator"] == "baseline - candidate > 0"
    assert rows["turnover"]["pass_fail"] == "fail"

    spy_sharpe = next(
        row
        for row in comparison_results
        if row["baseline"] == "return_baseline_spy" and row["metric"] == "sharpe"
    )
    assert spy_sharpe["candidate_value"] == pytest.approx(1.4)
    assert spy_sharpe["baseline_role"] == "return_baseline"
    assert spy_sharpe["absolute_delta"] == pytest.approx(
        spy_sharpe["candidate_value"] - spy_sharpe["baseline_value"]
    )


def test_stage1_model_comparison_results_align_metrics_per_validation_window(
    tmp_path,
) -> None:
    full_model_row = _stage1_required_ablation_row(
        "all_features",
        "signal",
        "All features",
        sharpe=0.70,
        excess_return=0.080,
        cost_adjusted_return=0.120,
    )
    full_model_row["window_metrics"] = [
        {
            "window_id": "fold_0",
            "window_label": "Fold 0",
            "window_role": "walk_forward_fold",
            "sharpe": 0.60,
            "turnover": 0.10,
        },
        {
            "window_id": "fold_1",
            "window_label": "Fold 1 OOS",
            "window_role": "oos_holdout",
            "sharpe": 0.40,
            "turnover": 0.30,
        },
    ]
    no_model_proxy_row = _stage1_required_ablation_row(
        "no_model_proxy",
        "pipeline_control",
        "No model proxy",
        sharpe=0.50,
        excess_return=0.045,
        cost_adjusted_return=0.090,
        feature_families=["price", "text", "sec"],
        model_proxy_enabled=False,
    )
    no_model_proxy_row["window_metrics"] = [
        {"window_id": "fold_0", "sharpe": 0.50, "turnover": 0.20},
        {"window_id": "fold_1", "sharpe": 0.35, "turnover": 0.20},
    ]
    price_only_row = _stage1_required_ablation_row(
        "price_only",
        "data_channel",
        "Price only",
        sharpe=0.40,
        excess_return=0.020,
        cost_adjusted_return=0.040,
        feature_families=["price", "chronos", "granite_ttm"],
    )
    price_only_row["window_metrics"] = {
        "fold_0": {"sharpe": 0.55, "turnover": 0.20},
        "fold_1": {"sharpe": 0.45, "turnover": 0.20},
    }
    ablation_summary = [
        full_model_row,
        price_only_row,
        _stage1_required_ablation_row(
            "text_only",
            "data_channel",
            "Text only",
            sharpe=0.30,
            excess_return=0.010,
            cost_adjusted_return=0.030,
            feature_families=["text"],
        ),
        _stage1_required_ablation_row(
            "sec_only",
            "data_channel",
            "SEC only",
            sharpe=0.20,
            excess_return=0.005,
            cost_adjusted_return=0.020,
            feature_families=["sec"],
        ),
        no_model_proxy_row,
        _stage1_required_ablation_row(
            "no_costs",
            "cost",
            "No costs",
            sharpe=0.90,
            excess_return=0.100,
            cost_adjusted_return=0.150,
            cost_enabled=False,
            slippage_enabled=False,
            turnover_enabled=False,
            effective_cost_bps=0.0,
            effective_slippage_bps=0.0,
        ),
    ]

    report = build_validity_gate_report(
        _comparison_predictions(),
        pd.DataFrame(),
        _comparison_equity_curve(),
        _comparison_metrics(),
        ablation_summary=ablation_summary,
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_1",
            benchmark_ticker="SPY",
            cost_bps=10.0,
            slippage_bps=5.0,
        ),
    )

    json_path, _ = write_validity_gate_artifacts(report, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    result_rows = payload["model_comparison_results"]
    rows = {
        (row["baseline"], row["window_id"], row["metric"]): row
        for row in result_rows
        if row["baseline"] in {"no_model_proxy", "price_only"}
    }

    assert rows[("no_model_proxy", "fold_0", "sharpe")]["absolute_delta"] == pytest.approx(0.10)
    assert rows[("no_model_proxy", "fold_0", "sharpe")]["relative_delta"] == pytest.approx(0.20)
    assert rows[("no_model_proxy", "fold_0", "sharpe")]["pass_fail"] == "pass"
    assert rows[("no_model_proxy", "fold_1", "turnover")]["absolute_delta"] == pytest.approx(0.10)
    assert rows[("no_model_proxy", "fold_1", "turnover")]["relative_delta"] == pytest.approx(0.50)
    assert rows[("no_model_proxy", "fold_1", "turnover")]["pass_fail"] == "fail"

    assert rows[("price_only", "fold_0", "sharpe")]["absolute_delta"] == pytest.approx(0.05)
    assert rows[("price_only", "fold_0", "sharpe")]["outperformance_threshold"] == pytest.approx(0.05)
    assert rows[("price_only", "fold_0", "sharpe")]["pass_fail"] == "fail"
    assert rows[("price_only", "fold_1", "sharpe")]["absolute_delta"] == pytest.approx(-0.05)
    assert rows[("price_only", "fold_1", "sharpe")]["relative_delta"] == pytest.approx(-0.05 / 0.45)
    assert rows[("price_only", "fold_1", "sharpe")]["pass_fail"] == "fail"
    assert rows[("price_only", "fold_1", "sharpe")]["window_label"] == "Fold 1 OOS"


def test_stage1_model_comparison_reports_missing_required_baseline_model() -> None:
    ablation_summary = [
        _stage1_required_ablation_row(
            "all_features",
            "signal",
            "All features",
            sharpe=0.70,
            excess_return=0.080,
            cost_adjusted_return=0.120,
        ),
        _stage1_required_ablation_row(
            "price_only",
            "data_channel",
            "Price only",
            sharpe=0.40,
            excess_return=0.020,
            cost_adjusted_return=0.040,
        ),
    ]

    report = build_validity_gate_report(
        _comparison_predictions(),
        pd.DataFrame(),
        _comparison_equity_curve(),
        _comparison_metrics(),
        ablation_summary=ablation_summary,
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_1",
            benchmark_ticker="SPY",
            cost_bps=10.0,
            slippage_bps=5.0,
        ),
    )

    rows = [
        row
        for row in report.model_comparison_results
        if row["baseline"] == "no_model_proxy"
    ]
    assert rows
    assert {row["status"] for row in rows} == {"not_evaluable"}
    assert {row["coverage_status"] for row in rows} == {"not_evaluable"}
    assert {row["reason_code"] for row in rows} == {"missing_baseline_model"}
    assert all(row["baseline_model_available"] is False for row in rows)
    assert all("baseline model result is missing: no_model_proxy" in row["reason"] for row in rows)


def test_stage1_model_comparison_reports_missing_window_and_metric_statuses() -> None:
    full_model_row = _stage1_required_ablation_row(
        "all_features",
        "signal",
        "All features",
        sharpe=0.70,
        excess_return=0.080,
        cost_adjusted_return=0.120,
    )
    full_model_row["window_metrics"] = [
        {"window_id": "fold_0", "window_label": "Fold 0", "sharpe": 0.60},
        {"window_id": "fold_1", "window_label": "Fold 1", "sharpe": 0.50},
    ]
    no_model_proxy_row = _stage1_required_ablation_row(
        "no_model_proxy",
        "pipeline_control",
        "No model proxy",
        sharpe=0.50,
        excess_return=0.045,
        cost_adjusted_return=0.090,
        feature_families=["price", "text", "sec"],
        model_proxy_enabled=False,
    )
    no_model_proxy_row["window_metrics"] = [
        {"window_id": "fold_0", "sharpe": 0.40}
    ]

    report = build_validity_gate_report(
        _comparison_predictions(),
        pd.DataFrame(),
        _comparison_equity_curve(),
        _comparison_metrics(),
        ablation_summary=[full_model_row, no_model_proxy_row],
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_1",
            benchmark_ticker="SPY",
            cost_bps=10.0,
            slippage_bps=5.0,
        ),
    )

    rows = {
        (row["window_id"], row["metric"]): row
        for row in report.model_comparison_results
        if row["baseline"] == "no_model_proxy"
    }
    assert rows[("fold_0", "sharpe")]["coverage_status"] == "pass"
    assert rows[("fold_0", "rank_ic")]["status"] == "not_evaluable"
    assert rows[("fold_0", "rank_ic")]["reason_code"] == "missing_candidate_metric"
    assert rows[("fold_1", "sharpe")]["status"] == "not_evaluable"
    assert rows[("fold_1", "sharpe")]["reason_code"] == "missing_baseline_window"
    assert rows[("fold_1", "sharpe")]["candidate_window_available"] is True
    assert rows[("fold_1", "sharpe")]["baseline_window_available"] is False


def test_full_model_similar_to_proxy_or_price_features_emits_model_value_warning(
    tmp_path,
) -> None:
    dates = pd.date_range("2026-01-02", periods=21, freq="B")
    predictions = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "fold": fold,
                "is_oos": fold == len(dates) - 1,
                "expected_return": expected_return,
                "forward_return_5": realized_return,
            }
            for fold, date in enumerate(dates)
            for ticker, expected_return, realized_return in (
                ("AAPL", 0.030, 0.012),
                ("MSFT", 0.020, 0.008),
                ("SPY", 0.010, 0.004),
            )
        ]
    )
    validation_dates = pd.date_range("2026-02-02", periods=5, freq="B")
    validation_summary = pd.DataFrame(
        {
            "fold": range(len(validation_dates)),
            "train_end": validation_dates - pd.Timedelta(days=7),
            "test_start": validation_dates,
            "is_oos": [False, False, False, False, True],
            "labeled_test_observations": [3] * len(validation_dates),
            "train_observations": [120] * len(validation_dates),
        }
    )
    equity_curve = pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": [0.030] * len(dates),
            "gross_return": [0.030] * len(dates),
            "cost_adjusted_return": [0.030] * len(dates),
            "benchmark_return": [0.004] * len(dates),
            "turnover": [0.10] * len(dates),
        }
    )
    full_model_row = _stage1_required_ablation_row(
        "all_features",
        "signal",
        "All features",
        sharpe=0.50,
        excess_return=0.050,
        cost_adjusted_return=0.080,
    )
    no_model_proxy_row = _stage1_required_ablation_row(
        "no_model_proxy",
        "pipeline_control",
        "No model proxy",
        sharpe=0.49,
        excess_return=0.048,
        cost_adjusted_return=0.078,
        feature_families=["price", "text", "sec"],
        model_proxy_enabled=False,
    )
    price_only_row = _stage1_required_ablation_row(
        "price_only",
        "data_channel",
        "Price only",
        sharpe=0.50,
        excess_return=0.050,
        cost_adjusted_return=0.080,
        feature_families=["price", "chronos", "granite_ttm"],
    )
    ablation_summary = [
        full_model_row,
        price_only_row,
        _stage1_required_ablation_row(
            "text_only",
            "data_channel",
            "Text only",
            sharpe=0.30,
            excess_return=0.020,
            cost_adjusted_return=0.030,
            feature_families=["text"],
        ),
        _stage1_required_ablation_row(
            "sec_only",
            "data_channel",
            "SEC only",
            sharpe=0.20,
            excess_return=0.010,
            cost_adjusted_return=0.020,
            feature_families=["sec"],
        ),
        no_model_proxy_row,
        _stage1_required_ablation_row(
            "no_costs",
            "cost",
            "No costs",
            sharpe=0.60,
            excess_return=0.060,
            cost_adjusted_return=0.090,
            cost_enabled=False,
            slippage_enabled=False,
            turnover_enabled=False,
            effective_cost_bps=0.0,
            effective_slippage_bps=0.0,
        ),
    ]

    report = build_validity_gate_report(
        predictions,
        validation_summary,
        equity_curve,
        SimpleNamespace(cagr=1.5, sharpe=1.2, max_drawdown=-0.04, turnover=0.10),
        ablation_summary=ablation_summary,
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_5",
            required_validation_horizon=5,
            benchmark_ticker="SPY",
            gap_periods=5,
            embargo_periods=5,
            cost_bps=0.0,
            slippage_bps=0.0,
        ),
    )

    model_value_gate = report.gate_results["model_value"]
    assert report.system_validity_status == "pass"
    assert report.strategy_candidate_status == "warning"
    assert report.warning is True
    assert model_value_gate["status"] == "warning"
    assert model_value_gate["reason_code"] == "model_value_too_similar_to_proxy_or_price"
    assert model_value_gate["warning_baselines"] == ["no_model_proxy", "price_only"]
    assert any("model-value warning" in warning for warning in report.warnings)
    assert [warning["code"] for warning in report.structured_warnings] == [
        "model_value_too_similar_to_proxy_or_price"
    ]

    json_path, markdown_path = write_validity_gate_artifacts(report, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert payload["gate_results"]["model_value"]["status"] == "warning"
    assert payload["metrics"]["model_value"] == payload["gate_results"]["model_value"]
    assert payload["evidence"]["model_value"] == payload["gate_results"]["model_value"]
    assert payload["structured_warnings"][0]["gate"] == "model_value"
    assert "| model_value | warning | model-value warning:" in markdown
    assert "model_value_too_similar_to_proxy_or_price" in markdown
    assert "<td>model_value</td>" in report.to_html()


def test_stage1_artifacts_surface_required_ablation_scenarios_and_comparable_metrics(
    tmp_path,
) -> None:
    required_scenarios = ("price_only", "text_only", "sec_only", "no_model_proxy", "no_costs")
    ablation_summary = [
        _stage1_required_ablation_row(
            "price_only",
            "data_channel",
            "Price only",
            sharpe=0.31,
            excess_return=0.011,
            cost_adjusted_return=0.021,
            feature_families=["price", "chronos", "granite_ttm"],
        ),
        _stage1_required_ablation_row(
            "text_only",
            "data_channel",
            "Text only",
            sharpe=0.22,
            excess_return=0.008,
            cost_adjusted_return=0.014,
            feature_families=["text"],
        ),
        _stage1_required_ablation_row(
            "sec_only",
            "data_channel",
            "SEC only",
            sharpe=0.18,
            excess_return=0.006,
            cost_adjusted_return=0.012,
            feature_families=["sec"],
        ),
        _stage1_required_ablation_row(
            "no_model_proxy",
            "pipeline_control",
            "No model proxy",
            sharpe=0.41,
            excess_return=0.015,
            cost_adjusted_return=0.028,
            feature_families=["price", "text", "sec"],
            model_proxy_enabled=False,
        ),
        _stage1_required_ablation_row(
            "no_costs",
            "cost",
            "No transaction costs",
            sharpe=0.52,
            excess_return=0.025,
            cost_adjusted_return=0.038,
            effective_cost_bps=0.0,
            effective_slippage_bps=0.0,
            cost_enabled=False,
            slippage_enabled=False,
            turnover_enabled=False,
        ),
    ]

    report = build_validity_gate_report(
        _comparison_predictions(),
        pd.DataFrame(),
        _comparison_equity_curve(),
        _comparison_metrics(),
        ablation_summary=ablation_summary,
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_1",
            benchmark_ticker="SPY",
            cost_bps=10.0,
            slippage_bps=5.0,
        ),
    )

    json_path, markdown_path = write_validity_gate_artifacts(report, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rows = {row["scenario"]: row for row in payload["ablation_results"]}

    assert tuple(rows) == required_scenarios
    for scenario in required_scenarios:
        row = rows[scenario]
        assert {
            "kind",
            "label",
            "cagr",
            "sharpe",
            "max_drawdown",
            "excess_return",
            "effective_cost_bps",
            "effective_slippage_bps",
            "validation_status",
            "validation_fold_count",
            "validation_oos_fold_count",
            "pipeline_controls",
            "deterministic_signal_evaluation_metrics",
        }.issubset(row)
        signal_metrics = row["deterministic_signal_evaluation_metrics"]
        assert signal_metrics["return_basis"] == "cost_adjusted_return"
        assert "cost_adjusted_cumulative_return" in signal_metrics
        assert "average_daily_turnover" in signal_metrics
        assert "total_cost_return" in signal_metrics

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## Stage 1 Ablation Scenario Comparison" in markdown
    assert (
        "| Scenario | Kind | Return Basis | CAGR | Sharpe | Max Drawdown | "
        "Excess Return | Cost-Adjusted Cumulative Return | Avg Daily Turnover | "
        "Total Cost Return | Effective Cost bps | Effective Slippage bps | "
        "Validation Status | Fold Count | OOS Fold Count | Feature Families | "
        "Model Proxy | Cost | Slippage | Turnover |"
    ) in markdown
    for scenario in required_scenarios:
        assert f"| {scenario} |" in markdown
    assert "| price_only | data_channel | cost_adjusted_return |" in markdown
    assert "| text_only | data_channel | cost_adjusted_return |" in markdown
    assert "| sec_only | data_channel | cost_adjusted_return |" in markdown
    assert "| no_model_proxy | pipeline_control | cost_adjusted_return |" in markdown
    assert "| no_costs | cost | cost_adjusted_return |" in markdown
    expected_no_cost_row = (
        "| no_costs | cost | cost_adjusted_return | 0.1100 | 0.5200 | "
        "-0.0400 | 0.0250 | 0.0380 | 0.1200 | 0.0000 | 0.0000 | "
        "0.0000 | pass | 5 | 1 | price, text, sec, chronos, granite_ttm | "
        "True | False | False | False |"
    )
    assert expected_no_cost_row in markdown


def test_validity_gate_baselines_match_strategy_evaluation_dates_and_horizon() -> None:
    evaluation_dates = pd.to_datetime(["2026-02-02", "2026-02-09", "2026-02-17"])
    predictions = pd.DataFrame(
        {
            "date": [
                evaluation_dates[0],
                evaluation_dates[0],
                evaluation_dates[1],
                evaluation_dates[1],
                evaluation_dates[2],
                evaluation_dates[2],
            ],
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT"],
            "expected_return": [0.03, 0.02, 0.04, 0.01, 0.02, 0.03],
            "forward_return_5": [0.05, 0.01, 0.02, 0.04, -0.01, 0.03],
        }
    )
    equity_curve = pd.DataFrame(
        {
            "date": evaluation_dates,
            "return_date": pd.to_datetime(["2026-02-09", "2026-02-17", "2026-02-24"]),
            "portfolio_return": [0.02, 0.03, 0.01],
            "cost_adjusted_return": [0.019, 0.03, 0.009],
            "benchmark_return": [0.004, -0.002, 0.003],
            "realized_return_column": ["forward_return_5"] * 3,
            "turnover": [1.0, 0.0, 0.5],
            "cost_bps": [10.0, 10.0, 10.0],
            "slippage_bps": [5.0, 5.0, 5.0],
        }
    )

    report = build_validity_gate_report(
        predictions,
        pd.DataFrame(),
        equity_curve,
        SimpleNamespace(cagr=1.5, sharpe=1.1, max_drawdown=-0.01, turnover=0.50),
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_5",
            benchmark_ticker="SPY",
            gap_periods=5,
            embargo_periods=5,
            cost_bps=10.0,
            slippage_bps=5.0,
        ),
    )

    comparison_rows = {row["name"]: row for row in report.cost_adjusted_metric_comparison}
    strategy_row = comparison_rows["strategy"]
    assert strategy_row["evaluation_observations"] == len(evaluation_dates)
    assert strategy_row["evaluation_start"] == "2026-02-02"
    assert strategy_row["evaluation_end"] == "2026-02-17"
    assert strategy_row["return_column"] == "forward_return_5"
    assert strategy_row["return_horizon"] == 5

    for baseline_name in ("SPY", "equal_weight"):
        baseline_row = comparison_rows[baseline_name]
        baseline_comparison = report.baseline_comparisons[baseline_name]
        benchmark_result = next(row for row in report.benchmark_results if row["name"] == baseline_name)

        assert baseline_row["evaluation_observations"] == strategy_row["evaluation_observations"]
        assert baseline_row["evaluation_start"] == strategy_row["evaluation_start"]
        assert baseline_row["evaluation_end"] == strategy_row["evaluation_end"]
        assert baseline_row["return_column"] == strategy_row["return_column"]
        assert baseline_row["return_horizon"] == strategy_row["return_horizon"]
        assert baseline_comparison["evaluation_observations"] == strategy_row["evaluation_observations"]
        assert baseline_comparison["evaluation_start"] == strategy_row["evaluation_start"]
        assert baseline_comparison["evaluation_end"] == strategy_row["evaluation_end"]
        assert baseline_comparison["return_column"] == strategy_row["return_column"]
        assert baseline_comparison["return_horizon"] == strategy_row["return_horizon"]
        assert benchmark_result["evaluation_observations"] == strategy_row["evaluation_observations"]
        assert benchmark_result["evaluation_start"] == strategy_row["evaluation_start"]
        assert benchmark_result["evaluation_end"] == strategy_row["evaluation_end"]
        assert benchmark_result["return_column"] == strategy_row["return_column"]
        assert benchmark_result["return_horizon"] == strategy_row["return_horizon"]


def test_stage1_comparison_filters_extra_baseline_dates_to_candidate_samples() -> None:
    evaluation_dates = pd.to_datetime(["2026-02-02", "2026-02-09", "2026-02-17"])
    extra_before = pd.Timestamp("2026-01-26")
    extra_after = pd.Timestamp("2026-02-24")
    predictions = pd.DataFrame(
        {
            "date": [
                evaluation_dates[0],
                evaluation_dates[0],
                evaluation_dates[1],
                evaluation_dates[1],
                evaluation_dates[2],
                evaluation_dates[2],
            ],
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT"],
            "expected_return": [0.03, 0.02, 0.04, 0.01, 0.02, 0.03],
            "forward_return_5": [0.05, 0.01, 0.02, 0.04, -0.01, 0.03],
        }
    )
    equity_curve = pd.DataFrame(
        {
            "date": evaluation_dates,
            "portfolio_return": [0.02, 0.03, -0.01],
            "cost_adjusted_return": [0.02, 0.03, -0.01],
            "realized_return_column": ["forward_return_5"] * 3,
            "turnover": [0.4, 0.2, 0.2],
        }
    )
    benchmark_return_series = pd.DataFrame(
        {
            "date": [extra_before, *evaluation_dates, extra_after],
            "benchmark_ticker": ["SPY"] * 5,
            "return_column": ["forward_return_5"] * 5,
            "return_horizon": [5] * 5,
            "benchmark_return": [0.99, 0.01, 0.02, 0.03, 0.99],
        }
    )
    equal_weight_baseline_return_series = pd.DataFrame(
        {
            "date": [extra_before, *evaluation_dates, extra_after],
            "baseline_name": ["equal_weight"] * 5,
            "return_column": ["forward_return_5"] * 5,
            "return_horizon": [5] * 5,
            "equal_weight_return": [0.88, 0.015, 0.025, 0.035, 0.88],
            "constituent_count": [2] * 5,
            "expected_constituent_count": [2] * 5,
        }
    )

    report = build_validity_gate_report(
        predictions,
        pd.DataFrame(),
        equity_curve,
        SimpleNamespace(cagr=1.5, sharpe=1.1, max_drawdown=-0.01, turnover=0.25),
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_5",
            benchmark_ticker="SPY",
            gap_periods=5,
            embargo_periods=5,
            cost_bps=0.0,
            slippage_bps=0.0,
        ),
        benchmark_return_series=benchmark_return_series,
        equal_weight_baseline_return_series=equal_weight_baseline_return_series,
    )

    expected_candidate_dates = ["2026-02-02", "2026-02-09", "2026-02-17"]
    alignment_gate = report.gate_results["baseline_sample_alignment"]
    assert alignment_gate["status"] == "pass"
    assert alignment_gate["candidate_dates"] == expected_candidate_dates
    assert alignment_gate["failed_baselines"] == []

    comparison_rows = {row["name"]: row for row in report.cost_adjusted_metric_comparison}
    assert comparison_rows["SPY"]["evaluation_observations"] == len(evaluation_dates)
    assert comparison_rows["SPY"]["evaluation_start"] == "2026-02-02"
    assert comparison_rows["SPY"]["evaluation_end"] == "2026-02-17"
    assert comparison_rows["SPY"]["cost_adjusted_cumulative_return"] == pytest.approx(
        1.01 * 1.02 * 1.03 - 1.0
    )
    assert comparison_rows["equal_weight"][
        "cost_adjusted_cumulative_return"
    ] == pytest.approx(1.015 * 1.025 * 1.035 - 1.0)

    for baseline_name in ("SPY", "equal_weight"):
        alignment = report.baseline_comparisons[baseline_name]["sample_alignment"]
        assert alignment["status"] == "pass"
        assert alignment["candidate_dates"] == expected_candidate_dates
        assert alignment["aligned_dates"] == expected_candidate_dates
        assert alignment["missing_candidate_dates"] == []
        assert alignment["extra_baseline_dates"] == ["2026-01-26", "2026-02-24"]
        assert alignment["aligned_sample_count"] == len(evaluation_dates)


def test_stage1_baseline_excess_returns_ignore_unaligned_baseline_samples() -> None:
    evaluation_dates = pd.to_datetime(["2026-02-02", "2026-02-09", "2026-02-17"])
    extra_before = pd.Timestamp("2026-01-26")
    extra_after = pd.Timestamp("2026-02-24")
    predictions = pd.DataFrame(
        {
            "date": [
                evaluation_dates[0],
                evaluation_dates[0],
                evaluation_dates[1],
                evaluation_dates[1],
                evaluation_dates[2],
                evaluation_dates[2],
            ],
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT"],
            "expected_return": [0.03, 0.02, 0.04, 0.01, 0.02, 0.03],
            "forward_return_5": [0.05, 0.01, 0.02, 0.04, -0.01, 0.03],
        }
    )
    equity_curve = pd.DataFrame(
        {
            "date": evaluation_dates,
            "portfolio_return": [0.02, 0.03, -0.01],
            "cost_adjusted_return": [0.02, 0.03, -0.01],
            "realized_return_column": ["forward_return_5"] * 3,
            "turnover": [0.4, 0.2, 0.2],
        }
    )
    benchmark_return_series = pd.DataFrame(
        {
            "date": [extra_before, *evaluation_dates, extra_after],
            "benchmark_ticker": ["SPY"] * 5,
            "return_column": ["forward_return_5"] * 5,
            "return_horizon": [5] * 5,
            "benchmark_return": [0.99, 0.01, 0.02, 0.03, -0.90],
        }
    )
    equal_weight_baseline_return_series = pd.DataFrame(
        {
            "date": [extra_before, *evaluation_dates, extra_after],
            "baseline_name": ["equal_weight"] * 5,
            "return_column": ["forward_return_5"] * 5,
            "return_horizon": [5] * 5,
            "equal_weight_return": [-0.90, 0.015, 0.025, 0.035, 0.88],
            "constituent_count": [2] * 5,
            "expected_constituent_count": [2] * 5,
        }
    )

    report = build_validity_gate_report(
        predictions,
        pd.DataFrame(),
        equity_curve,
        SimpleNamespace(cagr=1.5, sharpe=1.1, max_drawdown=-0.01, turnover=0.25),
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_5",
            benchmark_ticker="SPY",
            gap_periods=5,
            embargo_periods=5,
            cost_bps=0.0,
            slippage_bps=0.0,
        ),
        benchmark_return_series=benchmark_return_series,
        equal_weight_baseline_return_series=equal_weight_baseline_return_series,
    )

    strategy_return = 1.02 * 1.03 * 0.99 - 1.0
    spy_return = 1.01 * 1.02 * 1.03 - 1.0
    equal_weight_return = 1.015 * 1.025 * 1.035 - 1.0
    comparison_rows = {row["name"]: row for row in report.cost_adjusted_metric_comparison}
    baseline_comparisons = report.baseline_comparisons
    cost_gate = report.gate_results["cost_adjusted_performance"]

    assert comparison_rows["SPY"]["cost_adjusted_cumulative_return"] == pytest.approx(
        spy_return
    )
    assert comparison_rows["equal_weight"][
        "cost_adjusted_cumulative_return"
    ] == pytest.approx(equal_weight_return)
    assert comparison_rows["SPY"]["strategy_excess_return"] == pytest.approx(
        strategy_return - spy_return
    )
    assert comparison_rows["equal_weight"]["strategy_excess_return"] == pytest.approx(
        strategy_return - equal_weight_return
    )
    assert baseline_comparisons["SPY"]["strategy_excess_return"] == pytest.approx(
        strategy_return - spy_return
    )
    assert baseline_comparisons["equal_weight"][
        "strategy_excess_return"
    ] == pytest.approx(strategy_return - equal_weight_return)
    assert cost_gate["baseline_excess_returns"]["SPY"] == pytest.approx(
        strategy_return - spy_return
    )
    assert cost_gate["baseline_excess_returns"]["equal_weight"] == pytest.approx(
        strategy_return - equal_weight_return
    )

    for baseline_name in ("SPY", "equal_weight"):
        alignment = baseline_comparisons[baseline_name]["sample_alignment"]
        assert alignment["status"] == "pass"
        assert alignment["aligned_dates"] == [
            "2026-02-02",
            "2026-02-09",
            "2026-02-17",
        ]
        assert alignment["extra_baseline_dates"] == ["2026-01-26", "2026-02-24"]


def test_stage1_comparison_hard_fails_when_required_baseline_date_is_missing() -> None:
    evaluation_dates = pd.to_datetime(["2026-02-02", "2026-02-09", "2026-02-17"])
    predictions = pd.DataFrame(
        {
            "date": [
                evaluation_dates[0],
                evaluation_dates[0],
                evaluation_dates[1],
                evaluation_dates[1],
                evaluation_dates[2],
                evaluation_dates[2],
            ],
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT"],
            "expected_return": [0.03, 0.02, 0.04, 0.01, 0.02, 0.03],
            "forward_return_5": [0.05, 0.01, 0.02, 0.04, -0.01, 0.03],
        }
    )
    equity_curve = pd.DataFrame(
        {
            "date": evaluation_dates,
            "portfolio_return": [0.02, 0.03, -0.01],
            "cost_adjusted_return": [0.02, 0.03, -0.01],
            "realized_return_column": ["forward_return_5"] * 3,
            "turnover": [0.4, 0.2, 0.2],
        }
    )
    benchmark_return_series = pd.DataFrame(
        {
            "date": [evaluation_dates[0], evaluation_dates[2]],
            "benchmark_ticker": ["SPY", "SPY"],
            "return_column": ["forward_return_5", "forward_return_5"],
            "return_horizon": [5, 5],
            "benchmark_return": [0.01, 0.03],
        }
    )
    equal_weight_baseline_return_series = pd.DataFrame(
        {
            "date": evaluation_dates,
            "baseline_name": ["equal_weight"] * 3,
            "return_column": ["forward_return_5"] * 3,
            "return_horizon": [5] * 3,
            "equal_weight_return": [0.015, 0.025, 0.035],
            "constituent_count": [2] * 3,
            "expected_constituent_count": [2] * 3,
        }
    )

    report = build_validity_gate_report(
        predictions,
        pd.DataFrame(),
        equity_curve,
        SimpleNamespace(cagr=1.5, sharpe=1.1, max_drawdown=-0.01, turnover=0.25),
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_5",
            benchmark_ticker="SPY",
            gap_periods=5,
            embargo_periods=5,
            cost_bps=0.0,
            slippage_bps=0.0,
        ),
        benchmark_return_series=benchmark_return_series,
        equal_weight_baseline_return_series=equal_weight_baseline_return_series,
    )

    alignment_gate = report.gate_results["baseline_sample_alignment"]
    assert alignment_gate["status"] == "hard_fail"
    assert alignment_gate["failed_baselines"] == ["SPY"]
    assert alignment_gate["baselines"]["SPY"]["missing_candidate_dates"] == ["2026-02-09"]
    assert alignment_gate["baselines"]["SPY"]["aligned_dates"] == [
        "2026-02-02",
        "2026-02-17",
    ]
    assert "baseline sample alignment failed for required baseline(s): SPY" in (
        report.hard_fail_reasons
    )
    assert report.system_validity_status == "hard_fail"
    assert report.strategy_candidate_status == "not_evaluable"

    spy_comparison = report.baseline_comparisons["SPY"]
    assert spy_comparison["sample_alignment_status"] == "hard_fail"
    assert spy_comparison["excess_return"] is None
    assert spy_comparison["excess_return_status"] == "not_evaluable"
    assert report.gate_results["cost_adjusted_performance"][
        "baseline_excess_return_statuses"
    ]["SPY"] == "not_evaluable"


def test_stage1_output_artifact_contains_both_baseline_comparisons(tmp_path) -> None:
    report = build_validity_gate_report(
        _comparison_predictions(),
        pd.DataFrame(),
        _comparison_equity_curve(),
        _comparison_metrics(),
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_1",
            benchmark_ticker="SPY",
            cost_bps=10.0,
            slippage_bps=5.0,
        ),
    )

    json_path, markdown_path = write_validity_gate_artifacts(report, tmp_path)

    _assert_stage1_artifact_contains_both_baseline_comparisons(
        json.loads(json_path.read_text(encoding="utf-8"))
    )
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "| SPY | market_benchmark | cost_adjusted_benchmark_return |" in markdown
    assert "| equal_weight | equal_weight_universe | cost_adjusted_equal_weight_return |" in markdown


def test_validity_gate_artifacts_persist_cost_adjusted_comparison(tmp_path) -> None:
    report = build_validity_gate_report(
        _comparison_predictions(),
        pd.DataFrame(),
        _comparison_equity_curve(),
        _comparison_metrics(),
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_1",
            benchmark_ticker="SPY",
            cost_bps=10.0,
            slippage_bps=5.0,
        ),
    )

    json_path, markdown_path = write_validity_gate_artifacts(report, tmp_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert [row["name"] for row in payload["cost_adjusted_metric_comparison"]] == [
        "strategy",
        "SPY",
        "equal_weight",
    ]
    assert list(payload["side_by_side_metric_comparison"][0]) == [
        "metric",
        "metric_label",
        "strategy",
        "SPY",
        "equal_weight",
    ]
    side_by_side_columns = payload["side_by_side_metric_columns"]
    assert list(side_by_side_columns) == ["strategy", "SPY", "equal_weight"]
    assert set(side_by_side_columns["strategy"]) >= {
        "cagr",
        "sharpe",
        "cost_adjusted_cumulative_return",
    }
    assert list(payload["baseline_comparisons"]) == ["SPY", "equal_weight"]
    assert [row["name"] for row in payload["baseline_comparison_entries"]] == ["SPY", "equal_weight"]
    _assert_stage1_artifact_contains_both_baseline_comparisons(payload)
    assert "## Cost-Adjusted Strategy Comparison" in markdown_path.read_text(encoding="utf-8")
    assert "## Cost-Adjusted Side-by-Side Metrics" in markdown_path.read_text(encoding="utf-8")
    assert "## Baseline Comparisons" in markdown_path.read_text(encoding="utf-8")


def test_stage1_artifact_strategy_metrics_are_reported_after_cost_and_slippage(tmp_path) -> None:
    net_returns = pd.Series([0.010, -0.005, 0.020])
    gross_returns = pd.Series([0.015, 0.000, 0.025])
    total_costs = gross_returns - net_returns
    equity_curve = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-02", periods=3, freq="B"),
            "equity": (1 + net_returns).cumprod(),
            "benchmark_equity": [1.0, 1.0, 1.0],
            "portfolio_return": gross_returns,
            "gross_return": gross_returns,
            "cost_adjusted_return": net_returns,
            "transaction_cost_return": total_costs * 0.60,
            "slippage_cost_return": total_costs * 0.40,
            "total_cost_return": total_costs,
            "benchmark_return": [0.0, 0.0, 0.0],
            "turnover": [1.0, 1.0, 1.0],
            "exposure": [1.0, 1.0, 1.0],
        }
    )
    strategy_metrics = calculate_metrics(equity_curve)
    report = build_validity_gate_report(
        pd.DataFrame(),
        pd.DataFrame(),
        equity_curve,
        strategy_metrics,
    )

    json_path, markdown_path = write_validity_gate_artifacts(report, tmp_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    strategy_row = next(row for row in payload["cost_adjusted_metric_comparison"] if row["name"] == "strategy")
    expected_net_sharpe = float(net_returns.mean() / net_returns.std(ddof=0) * (252**0.5))
    expected_net_cumulative_return = float((1 + net_returns).prod() - 1)

    assert payload["metrics"]["strategy_sharpe"] == pytest.approx(expected_net_sharpe)
    assert payload["metrics"]["strategy_transaction_cost_return"] == pytest.approx(
        float((total_costs * 0.60).sum())
    )
    assert payload["metrics"]["strategy_slippage_cost_return"] == pytest.approx(
        float((total_costs * 0.40).sum())
    )
    assert payload["metrics"]["strategy_total_cost_return"] == pytest.approx(float(total_costs.sum()))
    assert strategy_row["return_basis"] == "cost_adjusted_return"
    assert strategy_row["cagr"] == pytest.approx(strategy_metrics.cagr)
    assert strategy_row["sharpe"] == pytest.approx(expected_net_sharpe)
    assert strategy_row["cost_adjusted_cumulative_return"] == pytest.approx(
        expected_net_cumulative_return
    )
    assert strategy_row["gross_cumulative_return"] > strategy_row["cost_adjusted_cumulative_return"]
    assert "## Cost-Adjusted Strategy Comparison" in markdown_path.read_text(encoding="utf-8")


def test_cost_adjusted_comparison_uses_same_metric_contract_for_strategy_spy_and_equal_weight() -> None:
    dates = pd.bdate_range("2026-02-02", periods=4)
    cost_bps = 10.0
    slippage_bps = 5.0
    strategy_gross_returns = pd.Series([0.020, -0.010, 0.015, 0.005])
    strategy_turnover = pd.Series([1.0, 0.4, 0.2, 0.3])
    strategy_costs = calculate_cost_adjusted_returns(
        strategy_gross_returns,
        strategy_turnover,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
    )
    equity_curve = pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": strategy_costs["cost_adjusted_return"],
            "cost_adjusted_return": strategy_costs["cost_adjusted_return"],
            "gross_return": strategy_gross_returns,
            "transaction_cost_return": strategy_costs["transaction_cost_return"],
            "slippage_cost_return": strategy_costs["slippage_cost_return"],
            "total_cost_return": strategy_costs["total_cost_return"],
            "benchmark_return": [0.010, -0.005, 0.008, 0.002],
            "turnover": strategy_turnover,
            "cost_bps": cost_bps,
            "slippage_bps": slippage_bps,
            "realized_return_column": "forward_return_1",
        }
    )
    strategy_metrics = calculate_metrics(equity_curve)

    benchmark_return_series = pd.DataFrame(
        {
            "date": dates,
            "benchmark_ticker": "SPY",
            "return_column": "forward_return_1",
            "return_horizon": 1,
            "benchmark_return": [0.010, -0.005, 0.008, 0.002],
        }
    )
    equal_weight_baseline_return_series = pd.DataFrame(
        {
            "date": dates,
            "baseline_name": "equal_weight",
            "return_column": "forward_return_1",
            "return_horizon": 1,
            "equal_weight_return": [0.012, -0.004, 0.006, 0.003],
            "constituent_count": [3, 3, 3, 3],
            "expected_constituent_count": [3, 3, 3, 3],
            "incomplete_ticker_universe": [False, False, False, False],
        }
    )
    report = build_validity_gate_report(
        pd.DataFrame(),
        pd.DataFrame(),
        equity_curve,
        strategy_metrics,
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT", "NVDA"],
            prediction_target_column="forward_return_1",
            benchmark_ticker="SPY",
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
        ),
        benchmark_return_series=benchmark_return_series,
        equal_weight_baseline_return_series=equal_weight_baseline_return_series,
    )

    rows = {row["name"]: row for row in report.cost_adjusted_metric_comparison}
    assert list(rows) == ["strategy", "SPY", "equal_weight"]
    expected_spy = _expected_metrics_from_gross_returns(
        dates,
        benchmark_return_series["benchmark_return"],
        pd.Series([1.0, 0.0, 0.0, 0.0]),
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
    )
    expected_equal_weight = _expected_metrics_from_gross_returns(
        dates,
        equal_weight_baseline_return_series["equal_weight_return"],
        pd.Series([1.0, 0.0, 0.0, 0.0]),
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
    )

    _assert_standard_cost_adjusted_metric_row(rows["strategy"], strategy_metrics)
    _assert_standard_cost_adjusted_metric_row(rows["SPY"], expected_spy)
    _assert_standard_cost_adjusted_metric_row(rows["equal_weight"], expected_equal_weight)
    assert rows["strategy"]["return_basis"] == "cost_adjusted_return"
    assert rows["SPY"]["return_basis"] == "cost_adjusted_benchmark_return"
    assert rows["equal_weight"]["return_basis"] == "cost_adjusted_equal_weight_return"
    assert {row["metric"] for row in report.side_by_side_metric_comparison} >= {
        "cagr",
        "sharpe",
        "max_drawdown",
        "average_daily_turnover",
        "transaction_cost_return",
        "slippage_cost_return",
        "total_cost_return",
    }
    for side_by_side_row in report.side_by_side_metric_comparison:
        assert {"strategy", "SPY", "equal_weight"}.issubset(side_by_side_row)


def _expected_metrics_from_gross_returns(
    dates: pd.Series | pd.DatetimeIndex,
    gross_returns: pd.Series,
    turnover: pd.Series,
    *,
    cost_bps: float,
    slippage_bps: float,
):
    costs = calculate_cost_adjusted_returns(
        gross_returns,
        turnover,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
    )
    return calculate_metrics(
        pd.DataFrame(
            {
                "date": dates,
                "portfolio_return": costs["cost_adjusted_return"],
                "cost_adjusted_return": costs["cost_adjusted_return"],
                "gross_return": gross_returns,
                "transaction_cost_return": costs["transaction_cost_return"],
                "slippage_cost_return": costs["slippage_cost_return"],
                "total_cost_return": costs["total_cost_return"],
                "turnover": turnover,
                "exposure": 1.0,
            }
        )
    )


def _assert_standard_cost_adjusted_metric_row(row: dict[str, object], expected) -> None:
    assert row["cagr"] == pytest.approx(expected.cagr)
    assert row["sharpe"] == pytest.approx(expected.sharpe)
    assert row["max_drawdown"] == pytest.approx(expected.max_drawdown)
    assert row["gross_cumulative_return"] == pytest.approx(expected.gross_cumulative_return)
    assert row["cost_adjusted_cumulative_return"] == pytest.approx(
        expected.cost_adjusted_cumulative_return
    )
    assert row["average_daily_turnover"] == pytest.approx(expected.turnover)
    assert row["transaction_cost_return"] == pytest.approx(expected.transaction_cost_return)
    assert row["slippage_cost_return"] == pytest.approx(expected.slippage_cost_return)
    assert row["total_cost_return"] == pytest.approx(expected.total_cost_return)
    assert row["evaluation_observations"] == 4
    assert row["cost_bps"] == pytest.approx(10.0)
    assert row["slippage_bps"] == pytest.approx(5.0)


def _stage1_required_ablation_row(
    scenario: str,
    kind: str,
    label: str,
    *,
    sharpe: float,
    excess_return: float,
    cost_adjusted_return: float,
    feature_families: list[str] | None = None,
    model_proxy_enabled: bool = True,
    cost_enabled: bool = True,
    slippage_enabled: bool = True,
    turnover_enabled: bool = True,
    effective_cost_bps: float = 10.0,
    effective_slippage_bps: float = 5.0,
) -> dict[str, object]:
    return {
        "scenario": scenario,
        "kind": kind,
        "label": label,
        "pipeline_controls": {
            "model_proxy": model_proxy_enabled,
            "cost": cost_enabled,
            "slippage": slippage_enabled,
            "turnover": turnover_enabled,
        },
        "permitted_feature_families": feature_families
        or ["price", "text", "sec", "chronos", "granite_ttm"],
        "effective_cost_bps": effective_cost_bps,
        "effective_slippage_bps": effective_slippage_bps,
        "cagr": 0.11,
        "sharpe": sharpe,
        "max_drawdown": -0.04,
        "turnover": 0.12,
        "excess_return": excess_return,
        "validation_status": "pass",
        "validation_fold_count": 5,
        "validation_oos_fold_count": 1,
        "deterministic_signal_evaluation_metrics": {
            "return_basis": "cost_adjusted_return",
            "cost_adjusted_cumulative_return": cost_adjusted_return,
            "average_daily_turnover": 0.12,
            "total_cost_return": 0.0 if not cost_enabled else 0.004,
            "action_counts": {"BUY": 3, "SELL": 1, "HOLD": 6},
        },
    }
