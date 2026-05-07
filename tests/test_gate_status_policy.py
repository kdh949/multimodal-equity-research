from __future__ import annotations

from quant_research.validation import (
    DEFAULT_GATE_STATUS_POLICY,
    GATE_STATUS_PRECEDENCE,
    NON_PASS_GATE_STATUSES,
    PASS_GATE_STATUS,
    GateStatusPolicy,
    aggregate_deterministic_gate_results,
    aggregate_deterministic_validity_gate,
)


def test_gate_status_policy_classifies_pass_and_non_pass_states() -> None:
    policy = GateStatusPolicy()

    assert policy.classify("PASS").to_dict() == {
        "raw_status": "PASS",
        "normalized_status": "pass",
        "decision": "PASS",
        "passed": True,
        "non_pass": False,
        "severity_rank": GATE_STATUS_PRECEDENCE.index("pass"),
    }
    for status in NON_PASS_GATE_STATUSES:
        classification = policy.classify(status)
        assert classification.passed is False
        assert classification.non_pass is True
        assert classification.normalized_status == status

    assert PASS_GATE_STATUS == "pass"
    assert policy.classify("WARN").normalized_status == "warning"
    assert policy.classify("hard-fail").normalized_status == "hard_fail"
    assert policy.classify("missing").normalized_status == "not_evaluable"
    assert policy.classify("unexpected-status").normalized_status == "not_evaluable"


def test_gate_status_policy_extracts_only_pass_fail_affecting_non_pass_results() -> None:
    gate_results = {
        "leakage": {
            "status": "pass",
            "affects_system": True,
            "affects_strategy": False,
        },
        "model_value": {
            "status": "warn",
            "reason": "proxy-like model value",
            "affects_strategy": True,
        },
        "rank_ic_5d_diagnostic": {
            "status": "fail",
            "reason": "diagnostic horizon only",
            "affects_pass_fail": False,
        },
        "walk_forward_oos": {
            "status": "hard_fail",
            "reason": "oos fold count below canonical minimum",
            "affects_system": True,
        },
    }

    non_pass = DEFAULT_GATE_STATUS_POLICY.non_pass_gate_results(gate_results)

    assert set(non_pass) == {"model_value", "walk_forward_oos"}
    assert non_pass["model_value"]["normalized_status"] == "warning"
    assert non_pass["model_value"]["decision"] == "WARN"
    assert non_pass["walk_forward_oos"]["normalized_status"] == "hard_fail"
    assert non_pass["walk_forward_oos"]["decision"] == "FAIL"

    with_diagnostic = DEFAULT_GATE_STATUS_POLICY.non_pass_gate_results(
        gate_results,
        include_unaffected=True,
    )
    assert "rank_ic_5d_diagnostic" in with_diagnostic


def test_deterministic_gate_aggregation_uses_common_status_policy_aliases() -> None:
    result = aggregate_deterministic_gate_results(
        {
            "leakage": {"status": "PASS", "affects_system": True},
            "model_value": {"status": "WARN", "affects_strategy": True},
            "diagnostic": {"status": "hard-fail", "affects_pass_fail": False},
        },
        system_validity_status="PASS",
        strategy_candidate_status="WARN",
    )

    assert result["final_decision"] == "WARN"
    assert result["blocking_items"] == []
    assert result["warning_items"] == ["model_value", "strategy_candidate_status"]
    assert result["item_results"][0]["normalized_status"] == "pass"
    assert result["item_results"][1]["normalized_status"] == "warn"


def test_deterministic_validity_engine_uses_common_policy_for_non_pass_aliases() -> None:
    result = aggregate_deterministic_validity_gate(
        {
            "leakage": {
                "status": "hard-fail",
                "reason": "embargo_periods=0",
                "affects_system": True,
                "affects_strategy": False,
            },
            "walk_forward_oos": {
                "status": "MISSING",
                "reason": "OOS folds unavailable",
                "affects_system": True,
                "affects_strategy": False,
            },
            "model_value": {
                "status": "WARN",
                "reason": "proxy-like model value",
                "affects_strategy": True,
            },
        }
    )

    assert result.system_validity_status == "hard_fail"
    assert result.strategy_candidate_status == "not_evaluable"
    assert result.hard_fail_rules == ("leakage",)
    assert result.insufficient_data_rules == ("walk_forward_oos",)
    assert result.warning_rules == ("model_value",)
