from __future__ import annotations

import pytest

from quant_research.validation import (
    DETERMINISTIC_VALIDITY_GATE_ENGINE_ID,
    DETERMINISTIC_VALIDITY_GATE_ENGINE_SCHEMA_VERSION,
    DeterministicValidityGateEngine,
    aggregate_deterministic_validity_gate,
)


def test_validity_gate_engine_passes_normal_case() -> None:
    result = aggregate_deterministic_validity_gate(
        {
            "leakage": {
                "status": "pass",
                "reason": "target horizon, feature cutoff, purge, and embargo passed",
                "affects_system": True,
                "affects_strategy": False,
            },
            "walk_forward_oos": {
                "status": "pass",
                "reason": "oos_fold_count >= 2",
                "affects_system": True,
                "affects_strategy": False,
            },
            "rank_ic": {
                "status": "pass",
                "reason": "rank IC is positive",
                "affects_strategy": True,
            },
            "cost_adjusted_performance": {
                "status": "pass",
                "reason": "cost-adjusted excess return is positive",
                "affects_strategy": True,
            },
        }
    )

    assert result.engine_id == DETERMINISTIC_VALIDITY_GATE_ENGINE_ID
    assert result.schema_version == DETERMINISTIC_VALIDITY_GATE_ENGINE_SCHEMA_VERSION
    assert result.system_validity_status == "pass"
    assert result.strategy_candidate_status == "pass"
    assert result.final_strategy_status == "pass"
    assert result.system_validity_pass is True
    assert result.strategy_pass is True
    assert result.hard_fail is False
    assert result.warning is False
    assert result.blocking_rules == ()
    assert result.warning_rules == ()
    assert result.insufficient_data_rules == ()
    assert result.reason == "all deterministic validity gate rules passed"


def test_validity_gate_engine_returns_warning_for_strategy_boundary_case() -> None:
    result = aggregate_deterministic_validity_gate(
        {
            "leakage": {
                "status": "pass",
                "affects_system": True,
                "affects_strategy": False,
            },
            "rank_ic": {"status": "pass", "affects_strategy": True},
            "model_value": {
                "status": "warning",
                "reason": "proxy IC improvement is below review threshold",
                "affects_strategy": True,
            },
            "rank_ic_1d_diagnostic": {
                "status": "warning",
                "reason": "diagnostic horizon only",
                "affects_system": False,
                "affects_strategy": False,
            },
        }
    )

    assert result.system_validity_status == "pass"
    assert result.strategy_candidate_status == "warning"
    assert result.final_strategy_status == "warning"
    assert result.system_validity_pass is True
    assert result.strategy_pass is False
    assert result.hard_fail is False
    assert result.warning is True
    assert result.warning_rules == ("model_value",)
    assert "rank_ic_1d_diagnostic" not in result.warning_rules
    assert result.rule_statuses["rank_ic_1d_diagnostic"] == "warning"
    assert result.reason == "deterministic strategy rule warning(s): model_value"


def test_validity_gate_engine_marks_insufficient_data_boundary_as_not_evaluable() -> None:
    result = aggregate_deterministic_validity_gate(
        {
            "leakage": {
                "status": "pass",
                "affects_system": True,
                "affects_strategy": False,
            },
            "walk_forward_oos": {
                "status": "insufficient_data",
                "reason": "oos_fold_count is unavailable",
                "affects_system": True,
                "affects_strategy": False,
            },
            "rank_ic": {"status": "pass", "affects_strategy": True},
        }
    )

    assert result.system_validity_status == "not_evaluable"
    assert result.strategy_candidate_status == "insufficient_data"
    assert result.final_strategy_status == "insufficient_data"
    assert result.system_validity_pass is False
    assert result.strategy_pass is False
    assert result.hard_fail is False
    assert result.warning is False
    assert result.insufficient_data_rules == ("walk_forward_oos",)
    assert result.hard_fail_rules == ()
    assert result.reason == "required validation data is insufficient: walk_forward_oos"


def test_validity_gate_engine_fails_strategy_without_system_hard_fail() -> None:
    result = aggregate_deterministic_validity_gate(
        {
            "leakage": {
                "status": "pass",
                "affects_system": True,
                "affects_strategy": False,
            },
            "walk_forward_oos": {
                "status": "pass",
                "affects_system": True,
                "affects_strategy": False,
            },
            "rank_ic": {
                "status": "fail",
                "reason": "mean_rank_ic <= 0",
                "affects_strategy": True,
            },
            "turnover": {
                "status": "pass",
                "affects_strategy": True,
            },
        }
    )

    assert result.system_validity_status == "pass"
    assert result.strategy_candidate_status == "fail"
    assert result.final_strategy_status == "fail"
    assert result.system_validity_pass is True
    assert result.strategy_pass is False
    assert result.hard_fail is False
    assert result.warning is False
    assert result.blocking_rules == ("rank_ic",)
    assert result.reason == "deterministic strategy rule(s) failed: rank_ic"


def test_validity_gate_engine_system_hard_fail_overrides_strategy_case() -> None:
    result = aggregate_deterministic_validity_gate(
        {
            "leakage": {
                "status": "hard_fail",
                "reason": "forward_return_20 cannot run with embargo_periods=0",
                "affects_system": True,
                "affects_strategy": False,
            },
            "rank_ic": {
                "status": "fail",
                "reason": "mean_rank_ic <= 0",
                "affects_strategy": True,
            },
            "model_value": {
                "status": "warning",
                "reason": "proxy-like model value",
                "affects_strategy": True,
            },
        }
    )

    assert result.system_validity_status == "hard_fail"
    assert result.strategy_candidate_status == "not_evaluable"
    assert result.final_strategy_status == "not_evaluable"
    assert result.system_validity_pass is False
    assert result.strategy_pass is False
    assert result.hard_fail is True
    assert result.warning is False
    assert result.hard_fail_rules == ("leakage",)
    assert result.blocking_rules == ("rank_ic",)
    assert result.warning_rules == ("model_value",)
    assert result.reason == "system validity hard-failed: leakage"


def test_validity_gate_engine_rejects_invalid_aggregation_inputs() -> None:
    with pytest.raises(TypeError, match="gate_results must be a mapping"):
        DeterministicValidityGateEngine().aggregate([("leakage", {"status": "pass"})])

