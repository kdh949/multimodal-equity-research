from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research.validation import (
    DETERMINISTIC_GATE_FINAL_DECISIONS,
    DETERMINISTIC_VALIDITY_GATE_ENGINE_ID,
    VALIDATION_GATE_DECISION_CRITERIA_POLICY_ID,
    VALIDATION_GATE_DECISION_CRITERIA_SCHEMA_VERSION,
    StrategyCandidateGatePolicy,
    StrategyCandidateMetricRule,
    ValidationGateThresholds,
    aggregate_deterministic_gate_results,
    build_validation_gate_decision_criteria,
    build_validity_gate_report,
    evaluate_strategy_candidate_gate_policy,
)


def test_strategy_candidate_policy_defines_canonical_inputs_and_rules() -> None:
    policy = StrategyCandidateGatePolicy()

    assert policy.target_horizon == "forward_return_20"
    assert policy.required_baselines == ("SPY", "equal_weight")
    assert set(policy.required_input_metrics) == {
        "mean_rank_ic",
        "oos_rank_ic",
        "positive_fold_ratio",
        "cost_adjusted_excess_return_vs_spy",
        "cost_adjusted_excess_return_vs_equal_weight",
        "max_drawdown",
        "average_daily_turnover",
        "proxy_ic_improvement",
    }

    rules = {rule.rule_id: rule for rule in policy.rules}
    assert rules["mean_rank_ic_positive"].operator == ">"
    assert rules["mean_rank_ic_positive"].threshold == 0.0
    assert rules["oos_rank_ic_positive"].operator == ">"
    assert rules["oos_rank_ic_positive"].threshold == 0.0
    assert rules["positive_fold_ratio_minimum"].operator == ">="
    assert rules["positive_fold_ratio_minimum"].threshold == pytest.approx(0.65)
    assert rules["spy_excess_return_positive"].operator == ">"
    assert rules["equal_weight_excess_return_positive"].operator == ">"
    assert rules["max_drawdown_floor"].operator == ">="
    assert rules["max_drawdown_floor"].threshold == pytest.approx(-0.20)
    assert rules["average_daily_turnover_budget"].operator == "<="
    assert rules["average_daily_turnover_budget"].threshold == pytest.approx(0.25)
    assert rules["proxy_ic_improvement_minimum"].operator == ">="
    assert rules["proxy_ic_improvement_minimum"].threshold == pytest.approx(0.01)


def test_validation_gate_decision_criteria_define_per_item_thresholds() -> None:
    thresholds = ValidationGateThresholds(
        min_folds=7,
        min_oos_folds=3,
        min_rank_ic=0.03,
        min_positive_fold_ratio=0.70,
        cost_adjusted_collapse_threshold=0.02,
        max_daily_turnover=0.25,
        max_monthly_turnover=1.50,
        sharpe_pass=0.90,
        sharpe_warning=0.60,
        drawdown_pass=-0.20,
        drawdown_warning=-0.30,
    )
    policy = StrategyCandidateGatePolicy(
        warning_rules=(
            StrategyCandidateMetricRule(
                "preferred_turnover_band",
                "average_daily_turnover",
                "<=",
                0.10,
                severity="warning",
            ),
        )
    )

    criteria = build_validation_gate_decision_criteria(thresholds, policy)
    rows = {row["criterion_id"]: row for row in criteria["criteria"]}

    assert criteria["policy_id"] == VALIDATION_GATE_DECISION_CRITERIA_POLICY_ID
    assert criteria["schema_version"] == VALIDATION_GATE_DECISION_CRITERIA_SCHEMA_VERSION
    assert criteria["status_contract"] == {
        "pass": "the validation item meets its configured pass condition",
        "warn": "the validation item is evaluable but falls into a review-required warning band",
        "fail": "the validation item violates a configured hard or strategy failure condition",
        "not_evaluable": "required metric or evidence is missing or insufficient",
    }
    assert criteria["status_aliases"] == {"warn": "warning"}
    assert criteria["status_precedence"] == ["fail", "warn", "not_evaluable", "pass"]

    assert rows["walk_forward_oos"]["threshold"] == {"min_folds": 7, "min_oos_folds": 3}
    assert rows["walk_forward_oos"]["stage"] == "system_validity"
    assert rows["walk_forward_oos"]["fail_status"] == "hard_fail"
    assert rows["rank_ic_signal_robustness"]["threshold"] == {
        "mean_rank_ic_pass": 0.03,
        "mean_rank_ic_fail_at_or_below": 0.0,
        "positive_fold_ratio_min": 0.70,
        "oos_rank_ic_min": 0.0,
    }
    assert rows["cost_adjusted_performance"]["threshold"]["collapse_threshold"] == 0.02
    assert rows["turnover"]["threshold"]["max_daily_turnover"] == 0.25
    assert rows["turnover"]["threshold"]["max_monthly_turnover"] == 1.50
    assert rows["benchmark_comparison"]["threshold"]["sharpe_pass"] == 0.90
    assert rows["drawdown"]["threshold"]["drawdown_pass"] == -0.20
    assert rows["strategy_candidate_policy_positive_fold_ratio_minimum"]["threshold"] == 0.65
    assert (
        rows["strategy_candidate_policy_preferred_turnover_band"]["fail_status"]
        == "warning"
    )


def test_strategy_candidate_policy_evaluation_returns_pass_fail_warning_and_missing_status() -> None:
    passing_metrics = {
        "mean_rank_ic": 0.03,
        "oos_rank_ic": 0.02,
        "positive_fold_ratio": 0.70,
        "cost_adjusted_excess_return_vs_spy": 0.02,
        "cost_adjusted_excess_return_vs_equal_weight": 0.01,
        "max_drawdown": -0.12,
        "average_daily_turnover": 0.20,
        "proxy_ic_improvement": 0.015,
    }
    assert evaluate_strategy_candidate_gate_policy(passing_metrics)["status"] == "pass"

    failed = evaluate_strategy_candidate_gate_policy(
        {**passing_metrics, "oos_rank_ic": 0.0}
    )
    assert failed["status"] == "fail"
    assert failed["failed_required_rules"] == ["oos_rank_ic_positive"]

    warning_policy = StrategyCandidateGatePolicy(
        warning_rules=(
            StrategyCandidateMetricRule(
                "preferred_turnover_band",
                "average_daily_turnover",
                "<=",
                0.10,
                severity="warning",
            ),
        )
    )
    warned = evaluate_strategy_candidate_gate_policy(
        passing_metrics,
        policy=warning_policy,
    )
    assert warned["status"] == "warning"
    assert warned["failed_warning_rules"] == ["preferred_turnover_band"]

    missing = evaluate_strategy_candidate_gate_policy(
        {key: value for key, value in passing_metrics.items() if key != "proxy_ic_improvement"}
    )
    assert missing["status"] == "not_evaluable"
    assert missing["missing_metrics"] == ["proxy_ic_improvement"]


@pytest.mark.parametrize(
    "metric",
    StrategyCandidateGatePolicy().required_input_metrics,
)
def test_strategy_candidate_policy_marks_each_missing_required_metric_not_evaluable(
    metric: str,
) -> None:
    metrics = _passing_strategy_candidate_metrics()
    metrics.pop(metric)

    result = evaluate_strategy_candidate_gate_policy(metrics)

    assert result["status"] == "not_evaluable"
    assert result["missing_metrics"] == [metric]
    assert result["input_metrics"][metric] is None
    assert "required Strategy Candidate Gate input metric(s) are missing" in result["reason"]


@pytest.mark.parametrize("missing_value", [None, float("nan")])
@pytest.mark.parametrize(
    "metric",
    [
        "cost_adjusted_excess_return_vs_spy",
        "cost_adjusted_excess_return_vs_equal_weight",
        "max_drawdown",
        "average_daily_turnover",
        "proxy_ic_improvement",
    ],
)
def test_strategy_candidate_policy_treats_missing_performance_and_risk_values_as_not_evaluable(
    metric: str,
    missing_value: float | None,
) -> None:
    metrics = _passing_strategy_candidate_metrics()
    metrics[metric] = missing_value

    result = evaluate_strategy_candidate_gate_policy(metrics)

    assert result["status"] == "not_evaluable"
    assert result["missing_metrics"] == [metric]
    assert result["failed_required_rules"] == []


def test_strategy_candidate_policy_missing_required_metric_takes_precedence_over_failed_rules() -> None:
    metrics = _passing_strategy_candidate_metrics()
    metrics["max_drawdown"] = None
    metrics["average_daily_turnover"] = 0.25 + 1e-6

    result = evaluate_strategy_candidate_gate_policy(metrics)

    assert result["status"] == "not_evaluable"
    assert result["missing_metrics"] == ["max_drawdown"]
    assert result["failed_required_rules"] == ["average_daily_turnover_budget"]
    assert result["status_precedence"][0] == "not_evaluable"


@pytest.mark.parametrize(
    ("metric", "boundary_value", "expected_status", "expected_failed_rule"),
    [
        ("mean_rank_ic", 0.0, "fail", "mean_rank_ic_positive"),
        ("oos_rank_ic", 0.0, "fail", "oos_rank_ic_positive"),
        ("positive_fold_ratio", 0.65, "pass", None),
        ("cost_adjusted_excess_return_vs_spy", 0.0, "fail", "spy_excess_return_positive"),
        (
            "cost_adjusted_excess_return_vs_equal_weight",
            0.0,
            "fail",
            "equal_weight_excess_return_positive",
        ),
        ("max_drawdown", -0.20, "pass", None),
        ("average_daily_turnover", 0.25, "pass", None),
        ("proxy_ic_improvement", 0.01, "pass", None),
    ],
)
def test_strategy_candidate_policy_required_rule_boundary_values(
    metric: str,
    boundary_value: float,
    expected_status: str,
    expected_failed_rule: str | None,
) -> None:
    metrics = _passing_strategy_candidate_metrics()
    metrics[metric] = boundary_value

    result = evaluate_strategy_candidate_gate_policy(metrics)

    assert result["status"] == expected_status
    if expected_failed_rule is None:
        assert result["failed_required_rules"] == []
    else:
        assert result["failed_required_rules"] == [expected_failed_rule]


@pytest.mark.parametrize(
    ("metric", "boundary_value", "failed_value", "failed_rule"),
    [
        ("positive_fold_ratio", 0.65, 0.65 - 1e-6, "positive_fold_ratio_minimum"),
        ("max_drawdown", -0.20, -0.20 - 1e-6, "max_drawdown_floor"),
        ("average_daily_turnover", 0.25, 0.25 + 1e-6, "average_daily_turnover_budget"),
        ("proxy_ic_improvement", 0.01, 0.01 - 1e-6, "proxy_ic_improvement_minimum"),
    ],
)
def test_strategy_candidate_policy_inclusive_thresholds_pass_exact_boundary_and_fail_beyond(
    metric: str,
    boundary_value: float,
    failed_value: float,
    failed_rule: str,
) -> None:
    boundary_metrics = _passing_strategy_candidate_metrics()
    boundary_metrics[metric] = boundary_value
    failed_metrics = _passing_strategy_candidate_metrics()
    failed_metrics[metric] = failed_value

    boundary_result = evaluate_strategy_candidate_gate_policy(boundary_metrics)
    failed_result = evaluate_strategy_candidate_gate_policy(failed_metrics)

    assert boundary_result["status"] == "pass"
    assert failed_result["status"] == "fail"
    assert failed_result["failed_required_rules"] == [failed_rule]


@pytest.mark.parametrize(
    ("operator", "threshold", "passing_value", "boundary_value", "failing_value"),
    [
        (">", 0.0, 1e-6, 0.0, -1e-6),
        (">=", 0.65, 0.65 + 1e-6, 0.65, 0.65 - 1e-6),
        ("<", 0.25, 0.25 - 1e-6, 0.25, 0.25 + 1e-6),
        ("<=", 0.25, 0.25 - 1e-6, 0.25, 0.25 + 1e-6),
        ("==", 0.01, 0.01, 0.01, 0.01 + 1e-6),
    ],
)
def test_strategy_candidate_policy_operator_threshold_boundaries_drive_pass_fail(
    operator: str,
    threshold: float,
    passing_value: float,
    boundary_value: float,
    failing_value: float,
) -> None:
    policy = StrategyCandidateGatePolicy(
        required_input_metrics=("test_metric",),
        rules=(
            StrategyCandidateMetricRule(
                "test_metric_rule",
                "test_metric",
                operator,
                threshold,
            ),
        ),
    )

    passing = evaluate_strategy_candidate_gate_policy(
        {"test_metric": passing_value},
        policy=policy,
    )
    boundary = evaluate_strategy_candidate_gate_policy(
        {"test_metric": boundary_value},
        policy=policy,
    )
    failing = evaluate_strategy_candidate_gate_policy(
        {"test_metric": failing_value},
        policy=policy,
    )

    assert passing["status"] == "pass"
    if operator in {">", "<"}:
        assert boundary["status"] == "fail"
        assert boundary["failed_required_rules"] == ["test_metric_rule"]
    else:
        assert boundary["status"] == "pass"
        assert boundary["failed_required_rules"] == []
    assert failing["status"] == "fail"
    assert failing["failed_required_rules"] == ["test_metric_rule"]


def test_strategy_candidate_policy_warning_rule_boundary_returns_pass_then_warning() -> None:
    policy = StrategyCandidateGatePolicy(
        warning_rules=(
            StrategyCandidateMetricRule(
                "preferred_turnover_band",
                "average_daily_turnover",
                "<=",
                0.10,
                severity="warning",
            ),
        )
    )
    boundary_metrics = _passing_strategy_candidate_metrics()
    boundary_metrics["average_daily_turnover"] = 0.10
    warning_metrics = _passing_strategy_candidate_metrics()
    warning_metrics["average_daily_turnover"] = 0.10 + 1e-6

    boundary_result = evaluate_strategy_candidate_gate_policy(
        boundary_metrics,
        policy=policy,
    )
    warning_result = evaluate_strategy_candidate_gate_policy(
        warning_metrics,
        policy=policy,
    )

    assert boundary_result["status"] == "pass"
    assert boundary_result["failed_warning_rules"] == []
    assert warning_result["status"] == "warning"
    assert warning_result["failed_warning_rules"] == ["preferred_turnover_band"]


def test_deterministic_gate_aggregation_returns_final_pass_warn_fail() -> None:
    passing = aggregate_deterministic_gate_results(
        {
            "leakage": {"status": "pass", "reason": "no leakage", "affects_system": True},
            "rank_ic": {"status": "pass", "reason": "positive IC", "affects_strategy": True},
            "rank_ic_1d_diagnostic": {
                "status": "warning",
                "reason": "diagnostic only",
                "affects_strategy": False,
                "affects_system": False,
                "affects_pass_fail": False,
            },
        },
        system_validity_status="pass",
        strategy_candidate_status="pass",
    )
    assert passing["engine_id"] == DETERMINISTIC_VALIDITY_GATE_ENGINE_ID
    assert tuple(passing["final_decision_contract"]) == DETERMINISTIC_GATE_FINAL_DECISIONS
    assert passing["final_decision"] == "PASS"
    assert passing["final_status"] == "pass"
    assert passing["blocking_items"] == []
    assert passing["warning_items"] == []

    warning = aggregate_deterministic_gate_results(
        {
            "leakage": {"status": "pass", "affects_system": True},
            "model_value": {"status": "warning", "affects_strategy": True},
        },
        system_validity_status="pass",
        strategy_candidate_status="warning",
    )
    assert warning["final_decision"] == "WARN"
    assert warning["final_status"] == "warning"
    assert warning["warning_items"] == ["model_value", "strategy_candidate_status"]

    failing = aggregate_deterministic_gate_results(
        {
            "leakage": {"status": "hard_fail", "affects_system": True},
            "walk_forward_oos": {"status": "insufficient_data", "affects_system": True},
            "turnover": {"status": "warning", "affects_strategy": True},
        },
        system_validity_status="hard_fail",
        strategy_candidate_status="not_evaluable",
    )
    assert failing["final_decision"] == "FAIL"
    assert failing["final_status"] == "fail"
    assert failing["blocking_items"] == [
        "leakage",
        "walk_forward_oos",
        "system_validity_status",
        "strategy_candidate_status",
    ]
    assert failing["not_evaluable_items"] == ["walk_forward_oos"]


def test_strategy_candidate_policy_is_validated() -> None:
    with pytest.raises(ValueError, match="target_horizon"):
        StrategyCandidateGatePolicy(target_horizon="forward_return_5")

    with pytest.raises(ValueError, match="duplicates"):
        StrategyCandidateGatePolicy(required_input_metrics=("mean_rank_ic", "mean_rank_ic"))

    with pytest.raises(ValueError, match="rule metrics"):
        StrategyCandidateGatePolicy(
            warning_rules=(
                StrategyCandidateMetricRule(
                    "unknown_metric_warning",
                    "unknown_metric",
                    ">=",
                    0.0,
                    severity="warning",
                ),
            )
        )


def test_validity_gate_report_records_strategy_candidate_policy() -> None:
    dates = pd.date_range("2026-01-02", periods=21, freq="B")
    predictions = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "fold": fold,
                "is_oos": fold >= 19,
                "expected_return": expected_return,
                "forward_return_20": realized_return,
            }
            for fold, date in enumerate(dates)
            for ticker, expected_return, realized_return in (
                ("AAPL", 0.03, 0.020),
                ("MSFT", 0.02, 0.010),
                ("SPY", 0.01, 0.001),
            )
        ]
    )
    validation_summary = pd.DataFrame(
        {
            "fold": range(len(dates)),
            "train_end": dates - pd.Timedelta(days=30),
            "test_start": dates,
            "is_oos": [idx >= 19 for idx in range(len(dates))],
            "labeled_test_observations": [3] * len(dates),
            "train_observations": [252] * len(dates),
            "purged_date_count": [20] * len(dates),
            "embargo_periods": [20] * len(dates),
            "purge_applied": [True] * len(dates),
            "embargo_applied": [True] * len(dates),
        }
    )
    equity_curve = pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": [0.025] * len(dates),
            "gross_return": [0.025] * len(dates),
            "cost_adjusted_return": [0.025] * len(dates),
            "benchmark_return": [0.001] * len(dates),
            "turnover": [0.10] * len(dates),
        }
    )

    report = build_validity_gate_report(
        predictions,
        validation_summary,
        equity_curve,
        SimpleNamespace(cagr=2.0, sharpe=1.2, max_drawdown=-0.05, turnover=0.10),
        ablation_summary=[
            {
                "scenario": "all_features",
                "sharpe": 1.0,
                "excess_return": 0.10,
                "rank_ic": 0.04,
            },
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
        ],
        config=SimpleNamespace(
            tickers=["AAPL", "MSFT"],
            prediction_target_column="forward_return_20",
            benchmark_ticker="SPY",
            gap_periods=20,
            embargo_periods=20,
            cost_bps=0.0,
            slippage_bps=0.0,
        ),
    )

    policy_gate = report.gate_results["strategy_candidate_policy"]
    assert report.strategy_candidate_policy["policy_id"] == "canonical_stage1_strategy_candidate_gate"
    assert report.metrics["strategy_candidate_policy"] == report.strategy_candidate_policy
    assert report.evidence["strategy_candidate_policy"] == report.strategy_candidate_policy
    assert (
        report.metrics["validation_gate_decision_criteria"]["policy_id"]
        == VALIDATION_GATE_DECISION_CRITERIA_POLICY_ID
    )
    assert (
        report.evidence["validation_gate_decision_criteria"]
        == report.metrics["validation_gate_decision_criteria"]
    )
    assert policy_gate["status"] == "pass"
    assert policy_gate["input_metrics"]["proxy_ic_improvement"] == pytest.approx(0.02)
    assert policy_gate["affects_strategy"] is False
    aggregation = report.gate_results["deterministic_gate_aggregation"]
    assert aggregation["final_decision"] == "PASS"
    assert aggregation["final_status"] == "pass"
    assert aggregation["affects_strategy"] is False
    assert aggregation["affects_pass_fail"] is False
    assert report.metrics["deterministic_gate_aggregation"]["final_decision"] == "PASS"
    assert report.metrics["final_gate_decision"] == "PASS"
    assert report.evidence["final_gate_status"] == "pass"


def _passing_strategy_candidate_metrics() -> dict[str, float]:
    return {
        "mean_rank_ic": 0.03,
        "oos_rank_ic": 0.02,
        "positive_fold_ratio": 0.70,
        "cost_adjusted_excess_return_vs_spy": 0.02,
        "cost_adjusted_excess_return_vs_equal_weight": 0.01,
        "max_drawdown": -0.12,
        "average_daily_turnover": 0.20,
        "proxy_ic_improvement": 0.015,
    }
