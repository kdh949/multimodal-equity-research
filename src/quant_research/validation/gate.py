from __future__ import annotations

import html
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from quant_research.backtest.metrics import (
    calculate_cost_adjusted_returns,
    calculate_portfolio_turnover,
)
from quant_research.data.timestamps import date_end_utc, timestamp_utc
from quant_research.performance import calculate_return_series_metrics
from quant_research.validation.ablation import NO_COST_ABLATION_SCENARIO
from quant_research.validation.benchmark_inputs import (
    EQUAL_WEIGHT_BASELINE_NAME,
    BaselineComparisonInput,
    StrategyEvaluationWindow,
    build_stage1_baseline_comparison_inputs,
    evaluate_stage1_baseline_comparison_inputs,
)
from quant_research.validation.comparison import (
    build_stage1_comparison_input_schema,
    build_stage1_comparison_result_schema,
)
from quant_research.validation.config import (
    CANONICAL_MIN_POSITIVE_FOLD_RATIO,
    DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG,
    default_model_comparison_config,
)
from quant_research.validation.gate_inputs import build_system_validity_gate_input_schema
from quant_research.validation.gate_outputs import (
    GateFailureReason,
    SerializableValidityGateReport,
    build_system_validity_gate_output_schema,
    build_system_validity_gate_report_schema,
)
from quant_research.validation.policy import DEFAULT_GATE_STATUS_POLICY

OFFICIAL_STRATEGY_FAIL_MESSAGE = "시스템은 유효하지만 현재 전략 후보는 배포/사용 부적합"
DEFAULT_HORIZONS = ("1d", "5d", "20d")
DIAGNOSTIC_ONLY_HORIZONS = ("1d", "5d")
ROBUSTNESS_HORIZONS: tuple[str, ...] = ()
# A 20d forward return needs the current observation plus 20 future observations.
MIN_WINDOWED_HORIZON_OBSERVATIONS = {20: 21}
MAX_DAILY_TURNOVER = 0.35
TRADING_DAYS_PER_MONTH = 21
MAX_MONTHLY_TURNOVER = MAX_DAILY_TURNOVER * TRADING_DAYS_PER_MONTH
STRATEGY_CANDIDATE_POLICY_ID = "canonical_stage1_strategy_candidate_gate"
STRATEGY_CANDIDATE_POLICY_SCHEMA_VERSION = "1.0"
STRATEGY_CANDIDATE_DEFAULT_REQUIRED_BASELINES = ("SPY", EQUAL_WEIGHT_BASELINE_NAME)
NO_MODEL_PROXY_ABLATION_SCENARIO = "no_model_proxy"
STAGE1_REQUIRED_ABLATION_SCENARIOS = (
    "price_only",
    "text_only",
    "sec_only",
    NO_MODEL_PROXY_ABLATION_SCENARIO,
    NO_COST_ABLATION_SCENARIO,
)
SIDE_BY_SIDE_METRIC_FIELDS = (
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
)
SIDE_BY_SIDE_NUMERIC_METRICS = {
    "cagr",
    "sharpe",
    "max_drawdown",
    "gross_cumulative_return",
    "cost_adjusted_cumulative_return",
    "average_daily_turnover",
    "transaction_cost_return",
    "slippage_cost_return",
    "total_cost_return",
    "excess_return",
}
SIDE_BY_SIDE_COUNT_METRICS = {"evaluation_observations", "return_horizon"}
MODEL_COMPARISON_LOWER_IS_BETTER_METRICS = {"turnover"}
INSUFFICIENT_DATA_GATE_STATUSES = {"insufficient_data", "skipped", "not_evaluable"}
MODEL_VALUE_WARNING_CODE = "model_value_too_similar_to_proxy_or_price"
MODEL_VALUE_COMPARISON_BASELINE_SCENARIOS = (
    NO_MODEL_PROXY_ABLATION_SCENARIO,
    "price_only",
)
MODEL_VALUE_MIN_MATERIAL_IMPROVEMENT_BY_METRIC = {
    "rank_ic": 0.005,
    "positive_fold_ratio": 0.05,
    "oos_rank_ic": 0.005,
    "sharpe": 0.05,
    "max_drawdown": 0.01,
    "cost_adjusted_cumulative_return": 0.005,
    "excess_return": 0.005,
}
type SystemValidityStatus = Literal["pass", "hard_fail", "not_evaluable"]
SYSTEM_VALIDITY_STATUSES: tuple[SystemValidityStatus, ...] = ("pass", "hard_fail", "not_evaluable")
SYSTEM_VALIDITY_GATE_POLICY_ID = "canonical_stage1_system_validity_gate"
SYSTEM_VALIDITY_GATE_POLICY_SCHEMA_VERSION = "1.0"
VALIDATION_GATE_DECISION_CRITERIA_POLICY_ID = "canonical_stage1_validation_gate_decision_criteria"
VALIDATION_GATE_DECISION_CRITERIA_SCHEMA_VERSION = "1.0"
DETERMINISTIC_VALIDITY_GATE_ENGINE_ID = "canonical_stage1_deterministic_validity_gate_engine"
DETERMINISTIC_VALIDITY_GATE_ENGINE_SCHEMA_VERSION = "1.0"
type StrategyCandidateStatus = Literal["pass", "warning", "fail", "insufficient_data", "not_evaluable"]
STRATEGY_CANDIDATE_STATUSES: tuple[StrategyCandidateStatus, ...] = (
    "pass",
    "warning",
    "fail",
    "insufficient_data",
    "not_evaluable",
)
type DeterministicGateFinalDecision = Literal["PASS", "WARN", "FAIL"]
DETERMINISTIC_GATE_FINAL_DECISIONS: tuple[DeterministicGateFinalDecision, ...] = (
    "PASS",
    "WARN",
    "FAIL",
)
type ValidationGateStage = Literal["system_validity", "strategy_candidate"]


@dataclass(frozen=True)
class DeterministicValidityGateAggregation:
    engine_id: str
    schema_version: str
    system_validity_status: SystemValidityStatus
    strategy_candidate_status: StrategyCandidateStatus
    final_strategy_status: StrategyCandidateStatus
    system_validity_pass: bool
    strategy_pass: bool
    hard_fail: bool
    warning: bool
    blocking_rules: tuple[str, ...]
    warning_rules: tuple[str, ...]
    insufficient_data_rules: tuple[str, ...]
    hard_fail_rules: tuple[str, ...]
    rule_statuses: dict[str, str]
    status_precedence: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        if self.system_validity_status not in SYSTEM_VALIDITY_STATUSES:
            allowed = ", ".join(SYSTEM_VALIDITY_STATUSES)
            raise ValueError(f"system_validity_status must be one of: {allowed}")
        if self.strategy_candidate_status not in STRATEGY_CANDIDATE_STATUSES:
            allowed = ", ".join(STRATEGY_CANDIDATE_STATUSES)
            raise ValueError(f"strategy_candidate_status must be one of: {allowed}")
        if self.final_strategy_status != self.strategy_candidate_status:
            raise ValueError("final_strategy_status must match strategy_candidate_status")

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class DeterministicValidityGateEngine:
    """Provider-free deterministic aggregation for validation rule results."""

    engine_id: str = DETERMINISTIC_VALIDITY_GATE_ENGINE_ID
    schema_version: str = DETERMINISTIC_VALIDITY_GATE_ENGINE_SCHEMA_VERSION
    status_precedence: tuple[str, ...] = (
        "hard_fail",
        "insufficient_data",
        "not_evaluable",
        "fail",
        "warning",
        "pass",
    )

    def aggregate(
        self,
        gate_results: Mapping[str, Mapping[str, Any]],
        *,
        hard_fail_reasons: Iterable[str] | None = None,
        insufficient_data: bool | None = None,
    ) -> DeterministicValidityGateAggregation:
        if not isinstance(gate_results, Mapping):
            raise TypeError("gate_results must be a mapping of rule name to result")

        rule_statuses = {
            str(name): str(_mapping_or_empty(result).get("status", "not_evaluable"))
            for name, result in gate_results.items()
        }
        system_hard_fail_rules = self._system_hard_fail_rules(gate_results)
        if hard_fail_reasons is None:
            hard_fail_list = [
                str(_mapping_or_empty(gate_results.get(name)).get("reason", name))
                for name in system_hard_fail_rules
            ]
        else:
            hard_fail_list = [str(reason) for reason in hard_fail_reasons if str(reason)]
            if hard_fail_list and not system_hard_fail_rules:
                system_hard_fail_rules = tuple(
                    name
                    for name, status in rule_statuses.items()
                    if status == "hard_fail"
                )

        insufficient_rules = self._insufficient_data_rules(gate_results)
        insufficient = bool(insufficient_data) if insufficient_data is not None else bool(
            insufficient_rules
        )
        system_status = _system_validity_status(hard_fail_list, insufficient)
        strategy_status = _strategy_status(system_status, dict(gate_results), insufficient)
        strategy_affecting_rules = self._strategy_affecting_rules(gate_results)
        blocking_rules = tuple(
            name
            for name in strategy_affecting_rules
            if DEFAULT_GATE_STATUS_POLICY.normalize(rule_statuses.get(name)) == "fail"
        )
        warning_rules = tuple(
            name
            for name in strategy_affecting_rules
            if DEFAULT_GATE_STATUS_POLICY.normalize(rule_statuses.get(name)) == "warning"
        )
        hard_fail_rules = tuple(
            name
            for name, status in rule_statuses.items()
            if DEFAULT_GATE_STATUS_POLICY.normalize(status) == "hard_fail"
        )

        return DeterministicValidityGateAggregation(
            engine_id=self.engine_id,
            schema_version=self.schema_version,
            system_validity_status=system_status,
            strategy_candidate_status=strategy_status,
            final_strategy_status=strategy_status,
            system_validity_pass=system_status == "pass",
            strategy_pass=strategy_status == "pass",
            hard_fail=system_status == "hard_fail",
            warning=strategy_status == "warning",
            blocking_rules=blocking_rules,
            warning_rules=warning_rules,
            insufficient_data_rules=insufficient_rules,
            hard_fail_rules=hard_fail_rules,
            rule_statuses=rule_statuses,
            status_precedence=self.status_precedence,
            reason=self._reason(
                system_status,
                strategy_status,
                blocking_rules=blocking_rules,
                warning_rules=warning_rules,
                insufficient_rules=insufficient_rules,
                hard_fail_rules=hard_fail_rules,
            ),
        )

    @staticmethod
    def _system_hard_fail_rules(
        gate_results: Mapping[str, Mapping[str, Any]],
    ) -> tuple[str, ...]:
        rules: list[str] = []
        for name, result in gate_results.items():
            payload = _mapping_or_empty(result)
            status = DEFAULT_GATE_STATUS_POLICY.normalize(payload.get("status"))
            affects_system = payload.get("affects_system")
            if status == "hard_fail" and affects_system is not False:
                rules.append(str(name))
            elif status == "fail" and affects_system is True:
                rules.append(str(name))
        return tuple(rules)

    @staticmethod
    def _strategy_affecting_rules(
        gate_results: Mapping[str, Mapping[str, Any]],
    ) -> tuple[str, ...]:
        return tuple(
            str(name)
            for name, result in gate_results.items()
            if _mapping_or_empty(result).get("affects_strategy", True) is not False
        )

    @staticmethod
    def _insufficient_data_rules(
        gate_results: Mapping[str, Mapping[str, Any]],
    ) -> tuple[str, ...]:
        return tuple(
            str(name)
            for name, result in gate_results.items()
            if DEFAULT_GATE_STATUS_POLICY.normalize(
                _mapping_or_empty(result).get("status")
            )
            in INSUFFICIENT_DATA_GATE_STATUSES
        )

    @staticmethod
    def _reason(
        system_status: SystemValidityStatus,
        strategy_status: StrategyCandidateStatus,
        *,
        blocking_rules: tuple[str, ...],
        warning_rules: tuple[str, ...],
        insufficient_rules: tuple[str, ...],
        hard_fail_rules: tuple[str, ...],
    ) -> str:
        if system_status == "hard_fail":
            return "system validity hard-failed: " + ", ".join(hard_fail_rules)
        if strategy_status == "insufficient_data":
            return "required validation data is insufficient: " + ", ".join(
                insufficient_rules
            )
        if strategy_status == "not_evaluable":
            return "strategy candidate is not evaluable under the current system status"
        if blocking_rules:
            return "deterministic strategy rule(s) failed: " + ", ".join(blocking_rules)
        if warning_rules:
            return "deterministic strategy rule warning(s): " + ", ".join(warning_rules)
        return "all deterministic validity gate rules passed"


def aggregate_deterministic_validity_gate(
    gate_results: Mapping[str, Mapping[str, Any]],
    *,
    hard_fail_reasons: Iterable[str] | None = None,
    insufficient_data: bool | None = None,
    engine: DeterministicValidityGateEngine | None = None,
) -> DeterministicValidityGateAggregation:
    engine = engine or DeterministicValidityGateEngine()
    return engine.aggregate(
        gate_results,
        hard_fail_reasons=hard_fail_reasons,
        insufficient_data=insufficient_data,
    )


@dataclass(frozen=True)
class SystemValidityGateCriterion:
    criterion_id: str
    gate_result: str
    pass_condition: str
    hard_fail_condition: str
    not_evaluable_condition: str
    required_evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.criterion_id.strip():
            raise ValueError("criterion_id must not be empty")
        if not self.gate_result.strip():
            raise ValueError("gate_result must not be empty")
        if not self.pass_condition.strip():
            raise ValueError("pass_condition must not be empty")
        if not self.hard_fail_condition.strip():
            raise ValueError("hard_fail_condition must not be empty")
        if not self.not_evaluable_condition.strip():
            raise ValueError("not_evaluable_condition must not be empty")
        if not self.required_evidence:
            raise ValueError("required_evidence must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class ValidationGateDecisionCriterion:
    criterion_id: str
    gate_result: str
    stage: ValidationGateStage
    pass_condition: str
    fail_condition: str
    threshold: Any
    metric: str | None = None
    operator: str | None = None
    warning_condition: str | None = None
    warning_threshold: Any | None = None
    not_evaluable_condition: str = "required metric or evidence is unavailable"
    pass_status: str = "pass"
    warning_status: str | None = "warning"
    fail_status: str = "fail"
    threshold_source: str = "ValidationGateThresholds"
    required_evidence: tuple[str, ...] = ()
    affects_system: bool = False
    affects_strategy: bool = True

    def __post_init__(self) -> None:
        if not self.criterion_id.strip():
            raise ValueError("criterion_id must not be empty")
        if not self.gate_result.strip():
            raise ValueError("gate_result must not be empty")
        if self.stage not in {"system_validity", "strategy_candidate"}:
            raise ValueError("stage must be system_validity or strategy_candidate")
        if not self.pass_condition.strip():
            raise ValueError("pass_condition must not be empty")
        if not self.fail_condition.strip():
            raise ValueError("fail_condition must not be empty")
        if not self.not_evaluable_condition.strip():
            raise ValueError("not_evaluable_condition must not be empty")
        valid_statuses = {
            "pass",
            "warning",
            "warn",
            "fail",
            "hard_fail",
            "insufficient_data",
            "not_evaluable",
        }
        for field_name, status in (
            ("pass_status", self.pass_status),
            ("warning_status", self.warning_status),
            ("fail_status", self.fail_status),
        ):
            if status is not None and status not in valid_statuses:
                raise ValueError(f"{field_name} must be a known gate status")

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class ValidationGateThresholds:
    min_folds: int = 5
    min_oos_folds: int = 2
    required_validation_horizon: int = 20
    min_rank_ic: float = 0.02
    min_positive_fold_ratio: float = CANONICAL_MIN_POSITIVE_FOLD_RATIO
    max_daily_turnover: float = MAX_DAILY_TURNOVER
    max_monthly_turnover: float = MAX_MONTHLY_TURNOVER
    monthly_turnover_budget: float | None = None
    sharpe_pass: float = 0.80
    sharpe_warning: float = 0.50
    benchmark_sharpe_margin: float = 0.20
    cost_adjusted_collapse_threshold: float = 0.0
    drawdown_pass: float = -0.25
    drawdown_warning: float = -0.35
    max_drawdown_spy_lag: float = 0.05


def build_validation_gate_decision_criteria(
    thresholds: ValidationGateThresholds | None = None,
    strategy_candidate_policy: StrategyCandidateGatePolicy | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or ValidationGateThresholds()
    strategy_candidate_policy = _strategy_candidate_gate_policy(strategy_candidate_policy)
    criteria: list[ValidationGateDecisionCriterion] = [
        ValidationGateDecisionCriterion(
            criterion_id="target_horizon_forward_return_20",
            gate_result="leakage",
            stage="system_validity",
            metric="target_horizon",
            operator="==",
            threshold="forward_return_20",
            pass_condition="decision target is forward_return_20 and target_horizon is 20",
            warning_condition=None,
            fail_condition="target horizon is not forward_return_20 or purge/embargo is horizon-inconsistent",
            fail_status="hard_fail",
            threshold_source="canonical experiment contract",
            required_evidence=("metrics.target_column", "metrics.target_horizon"),
            affects_system=True,
            affects_strategy=False,
        ),
        ValidationGateDecisionCriterion(
            criterion_id="feature_availability_cutoff",
            gate_result="leakage",
            stage="system_validity",
            metric="feature_availability_timestamp",
            operator="<=",
            threshold="sample_date_end",
            pass_condition="every feature availability timestamp is at or before the sample date end",
            warning_condition=None,
            fail_condition="any feature is available only after the decision sample date",
            fail_status="hard_fail",
            threshold_source="feature timing contract",
            required_evidence=(
                "system_validity_gate_input_schema.feature_availability_cutoff",
                "evidence.leakage",
            ),
            affects_system=True,
            affects_strategy=False,
        ),
        ValidationGateDecisionCriterion(
            criterion_id="purge_embargo_horizon_consistency",
            gate_result="leakage",
            stage="system_validity",
            metric="gap_periods_and_embargo_periods",
            operator=">=",
            threshold={
                "required_validation_horizon": int(thresholds.required_validation_horizon),
                "minimum_gap_periods": int(thresholds.required_validation_horizon),
                "minimum_embargo_periods": int(thresholds.required_validation_horizon),
            },
            pass_condition="gap_periods and embargo_periods are both at least the required horizon",
            warning_condition=None,
            fail_condition="forward_return_20 has embargo_periods=0 or purge/embargo below the horizon",
            fail_status="hard_fail",
            required_evidence=("evidence.purge_embargo_application", "metrics.embargo_periods"),
            affects_system=True,
            affects_strategy=False,
        ),
        ValidationGateDecisionCriterion(
            criterion_id="walk_forward_oos",
            gate_result="walk_forward_oos",
            stage="system_validity",
            metric="oos_fold_count",
            operator=">=",
            threshold={
                "min_folds": int(thresholds.min_folds),
                "min_oos_folds": int(thresholds.min_oos_folds),
            },
            pass_condition="fold_count and oos_fold_count meet configured minimums with labeled tests",
            warning_condition=None,
            fail_condition="chronology is invalid or required OOS folds are structurally impossible",
            fail_status="hard_fail",
            not_evaluable_condition="fold_count, oos_fold_count, train observations, or labeled test observations are insufficient",
            required_evidence=("evidence.walk_forward_oos", "metrics.oos_fold_count"),
            affects_system=True,
            affects_strategy=False,
        ),
        ValidationGateDecisionCriterion(
            criterion_id="baseline_sample_alignment",
            gate_result="baseline_sample_alignment",
            stage="system_validity",
            metric="baseline_evaluation_dates",
            operator="matches",
            threshold={"required_baselines": ["SPY", EQUAL_WEIGHT_BASELINE_NAME]},
            pass_condition="SPY and equal-weight baseline samples align to strategy evaluation samples",
            warning_condition=None,
            fail_condition="required baseline dates or horizon alignment do not match the strategy sample",
            fail_status="hard_fail",
            not_evaluable_condition="candidate or baseline return samples are unavailable",
            threshold_source="benchmark_config",
            required_evidence=("evidence.baseline_sample_alignment",),
            affects_system=True,
            affects_strategy=False,
        ),
        ValidationGateDecisionCriterion(
            criterion_id="artifact_reproducibility_contract",
            gate_result="system_validity_artifact_contract",
            stage="system_validity",
            metric="artifact_manifest",
            operator="contains",
            threshold=(
                "system_validity_gate_input_schema",
                "system_validity_gate_output_schema",
                "thresholds",
                "gate_results",
                "evidence",
            ),
            pass_condition="canonical report payload embeds schemas, thresholds, gate results, and evidence",
            warning_condition=None,
            fail_condition="required reproducibility schema or manifest evidence is missing",
            fail_status="hard_fail",
            threshold_source="artifact contract",
            required_evidence=(
                "system_validity_gate_input_schema",
                "system_validity_gate_output_schema",
                "evidence.thresholds",
            ),
            affects_system=True,
            affects_strategy=False,
        ),
        ValidationGateDecisionCriterion(
            criterion_id="rank_ic_signal_robustness",
            gate_result="rank_ic",
            stage="strategy_candidate",
            metric="mean_rank_ic_positive_fold_ratio_oos_rank_ic",
            operator="composite",
            threshold={
                "mean_rank_ic_pass": float(thresholds.min_rank_ic),
                "mean_rank_ic_fail_at_or_below": 0.0,
                "positive_fold_ratio_min": float(thresholds.min_positive_fold_ratio),
                "oos_rank_ic_min": 0.0,
            },
            pass_condition="mean_rank_ic >= min_rank_ic, positive_fold_ratio >= minimum, and oos_rank_ic > 0",
            warning_condition="rank IC is positive but below one or more pass thresholds",
            fail_condition="mean_rank_ic <= 0 or positive_fold_ratio is below the configured minimum",
            required_evidence=("evidence.rank_ic", "metrics.positive_fold_ratio_threshold"),
        ),
        ValidationGateDecisionCriterion(
            criterion_id="cost_adjusted_performance",
            gate_result="cost_adjusted_performance",
            stage="strategy_candidate",
            metric="cost_adjusted_cumulative_return_and_baseline_excess_return",
            operator=">",
            threshold={
                "collapse_threshold": float(thresholds.cost_adjusted_collapse_threshold),
                "minimum_excess_return_vs_each_required_baseline": 0.0,
            },
            pass_condition="cost-adjusted cumulative return is above collapse threshold and excess return is positive versus every required baseline",
            warning_condition=None,
            fail_condition="cost-adjusted return collapses or any required baseline excess return is not positive",
            required_evidence=("evidence.cost_adjusted_collapse_check", "metrics.baseline_comparisons"),
        ),
        ValidationGateDecisionCriterion(
            criterion_id="benchmark_comparison",
            gate_result="benchmark_comparison",
            stage="strategy_candidate",
            metric="strategy_sharpe",
            operator=">= or margin >=",
            threshold={
                "sharpe_pass": float(thresholds.sharpe_pass),
                "sharpe_warning": float(thresholds.sharpe_warning),
                "benchmark_sharpe_margin": float(thresholds.benchmark_sharpe_margin),
            },
            pass_condition="strategy Sharpe meets absolute pass threshold or beats benchmark Sharpe by configured margin",
            warning_condition="strategy Sharpe is at least the warning threshold",
            fail_condition="strategy Sharpe is below the warning threshold",
            required_evidence=("metrics.strategy_sharpe", "evidence.baseline_results"),
        ),
        ValidationGateDecisionCriterion(
            criterion_id="turnover",
            gate_result="turnover",
            stage="strategy_candidate",
            metric="average_daily_turnover",
            operator="<=",
            threshold={
                "max_daily_turnover": float(thresholds.max_daily_turnover),
                "max_monthly_turnover": float(thresholds.max_monthly_turnover),
                "monthly_turnover_budget": thresholds.monthly_turnover_budget,
            },
            pass_condition="average daily and monthly turnover stay within configured budgets",
            warning_condition="turnover exceeds budget without clear cost-collapse evidence",
            fail_condition="turnover exceeds budget and costs collapse otherwise positive performance",
            required_evidence=("metrics.strategy_turnover", "metrics.strategy_max_monthly_turnover"),
        ),
        ValidationGateDecisionCriterion(
            criterion_id="drawdown",
            gate_result="drawdown",
            stage="strategy_candidate",
            metric="max_drawdown",
            operator=">=",
            threshold={
                "drawdown_pass": float(thresholds.drawdown_pass),
                "drawdown_warning": float(thresholds.drawdown_warning),
                "max_drawdown_spy_lag": float(thresholds.max_drawdown_spy_lag),
            },
            pass_condition="drawdown is within absolute pass floor and no worse than SPY by configured lag",
            warning_condition="drawdown is between warning and pass floors",
            fail_condition="drawdown is worse than the warning floor",
            required_evidence=("metrics.strategy_max_drawdown", "evidence.baseline_results"),
        ),
        ValidationGateDecisionCriterion(
            criterion_id="model_value",
            gate_result="model_value",
            stage="strategy_candidate",
            metric="proxy_ic_improvement",
            operator=">=",
            threshold={"minimum_material_rank_ic_improvement": MODEL_VALUE_MIN_MATERIAL_IMPROVEMENT_BY_METRIC["rank_ic"]},
            pass_condition="full model shows material improvement over proxy/price baselines",
            warning_condition="model value is too similar to proxy or price-only alternatives",
            fail_condition="not used as a hard fail in v1; warning is review-blocking context only",
            fail_status="warning",
            threshold_source="MODEL_VALUE_MIN_MATERIAL_IMPROVEMENT_BY_METRIC",
            required_evidence=("evidence.model_value", "evidence.model_comparison_results"),
        ),
    ]
    for rule in strategy_candidate_policy.rules:
        criteria.append(
            ValidationGateDecisionCriterion(
                criterion_id=f"strategy_candidate_policy_{rule.rule_id}",
                gate_result="strategy_candidate_policy",
                stage="strategy_candidate",
                metric=rule.metric,
                operator=rule.operator,
                threshold=rule.threshold,
                pass_condition=f"{rule.metric} {rule.operator} {rule.threshold}",
                warning_condition=None,
                fail_condition=f"{rule.metric} does not satisfy required policy rule {rule.rule_id}",
                threshold_source=f"{strategy_candidate_policy.policy_id}.rules",
                required_evidence=("evidence.strategy_candidate_policy_evaluation",),
            )
        )
    for rule in strategy_candidate_policy.warning_rules:
        criteria.append(
            ValidationGateDecisionCriterion(
                criterion_id=f"strategy_candidate_policy_{rule.rule_id}",
                gate_result="strategy_candidate_policy",
                stage="strategy_candidate",
                metric=rule.metric,
                operator=rule.operator,
                threshold=rule.threshold,
                pass_condition=f"{rule.metric} {rule.operator} {rule.threshold}",
                warning_condition=f"{rule.metric} does not satisfy warning policy rule {rule.rule_id}",
                fail_condition=f"{rule.metric} does not satisfy warning policy rule {rule.rule_id}",
                fail_status="warning",
                threshold_source=f"{strategy_candidate_policy.policy_id}.warning_rules",
                required_evidence=("evidence.strategy_candidate_policy_evaluation",),
            )
        )
    return {
        "policy_id": VALIDATION_GATE_DECISION_CRITERIA_POLICY_ID,
        "schema_version": VALIDATION_GATE_DECISION_CRITERIA_SCHEMA_VERSION,
        "status_contract": {
            "pass": "the validation item meets its configured pass condition",
            "warn": "the validation item is evaluable but falls into a review-required warning band",
            "fail": "the validation item violates a configured hard or strategy failure condition",
            "not_evaluable": "required metric or evidence is missing or insufficient",
        },
        "status_aliases": {"warn": "warning"},
        "status_precedence": ["fail", "warn", "not_evaluable", "pass"],
        "criteria": [criterion.to_dict() for criterion in criteria],
    }


def build_system_validity_gate_criteria(
    thresholds: ValidationGateThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or ValidationGateThresholds()
    criteria = (
        SystemValidityGateCriterion(
            criterion_id="target_horizon_forward_return_20",
            gate_result="leakage",
            pass_condition="decision target resolves to forward_return_20 with target_horizon=20",
            hard_fail_condition="target horizon is below 20 or configured purge/embargo is horizon-inconsistent",
            not_evaluable_condition="predictions are empty or forward_return_20 cannot be evaluated",
            required_evidence=("metrics.target_column", "metrics.target_horizon"),
        ),
        SystemValidityGateCriterion(
            criterion_id="feature_availability_cutoff",
            gate_result="leakage",
            pass_condition="all feature availability timestamps used upstream are <= the sample date end",
            hard_fail_condition="any feature availability cutoff or event/availability ordering validation fails",
            not_evaluable_condition="feature rows are absent, so model and strategy gates cannot be evaluated",
            required_evidence=(
                "system_validity_gate_input_schema.feature_availability_cutoff",
                "evidence.leakage",
            ),
        ),
        SystemValidityGateCriterion(
            criterion_id="purge_embargo_horizon_consistency",
            gate_result="leakage",
            pass_condition="gap_periods and embargo_periods are both >= target_horizon and each fold records applied purge/embargo evidence",
            hard_fail_condition="forward_return_20 has embargo_periods=0, embargo/gap below 20, overlapping train/test windows, or missing applied purge/embargo per-fold evidence",
            not_evaluable_condition="walk-forward fold rows are unavailable",
            required_evidence=("evidence.purge_embargo_application", "metrics.embargo_periods"),
        ),
        SystemValidityGateCriterion(
            criterion_id="walk_forward_oos",
            gate_result="walk_forward_oos",
            pass_condition=f"fold_count >= {int(thresholds.min_folds)} and oos_fold_count >= {int(thresholds.min_oos_folds)} with labeled test observations",
            hard_fail_condition="train/test chronology is invalid or canonical forward_return_20 has fewer than two OOS folds after enough folds were produced",
            not_evaluable_condition="folds, OOS folds, training observations, or labeled test observations are insufficient",
            required_evidence=("evidence.walk_forward_oos", "metrics.oos_fold_count"),
        ),
        SystemValidityGateCriterion(
            criterion_id="benchmark_equal_weight_sample_alignment",
            gate_result="baseline_sample_alignment",
            pass_condition="SPY and equal-weight universe baselines align to the candidate evaluation sample",
            hard_fail_condition="a required baseline is missing candidate evaluation dates or has incompatible sample alignment",
            not_evaluable_condition="candidate or baseline return samples are unavailable",
            required_evidence=(
                "evidence.baseline_sample_alignment",
                "metrics.baseline_sample_alignment",
            ),
        ),
        SystemValidityGateCriterion(
            criterion_id="artifact_reproducibility_contract",
            gate_result="system_validity_artifact_contract",
            pass_condition="input schema, output schema, thresholds, gate results, and evidence are embedded in the report artifact payload",
            hard_fail_condition="required reproducibility schema or manifest evidence is missing from the canonical report payload",
            not_evaluable_condition="report artifact generation was not requested",
            required_evidence=(
                "system_validity_gate_input_schema",
                "system_validity_gate_output_schema",
                "evidence.thresholds",
            ),
        ),
    )
    return {
        "policy_id": SYSTEM_VALIDITY_GATE_POLICY_ID,
        "schema_version": SYSTEM_VALIDITY_GATE_POLICY_SCHEMA_VERSION,
        "status_contract": {
            "pass": "all system criteria pass and no required data criterion is insufficient",
            "hard_fail": "one or more structural validity criteria fail",
            "not_evaluable": "no structural failure exists, but required data/evidence is insufficient",
        },
        "status_precedence": ["hard_fail", "not_evaluable", "pass"],
        "criteria": [criterion.to_dict() for criterion in criteria],
    }


type StrategyCandidateRuleSeverity = Literal["required", "warning"]
type StrategyCandidateRuleOperator = Literal[">", ">=", "<", "<=", "=="]


@dataclass(frozen=True)
class StrategyCandidateMetricRule:
    rule_id: str
    metric: str
    operator: StrategyCandidateRuleOperator
    threshold: float
    severity: StrategyCandidateRuleSeverity = "required"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.rule_id.strip():
            raise ValueError("rule_id must not be empty")
        if not self.metric.strip():
            raise ValueError("metric must not be empty")
        if self.operator not in {">", ">=", "<", "<=", "=="}:
            raise ValueError("operator must be one of: >, >=, <, <=, ==")
        if self.severity not in {"required", "warning"}:
            raise ValueError("severity must be one of: required, warning")
        if not np.isfinite(float(self.threshold)):
            raise ValueError("threshold must be finite")

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class StrategyCandidateGatePolicy:
    """Configurable Stage 1 policy for strategy candidate PASS/WARN/FAIL.

    The policy is intentionally metric-based. LLM/model outputs are only inputs
    after deterministic scoring and validation; the policy never makes trading
    orders or position decisions.
    """

    policy_id: str = STRATEGY_CANDIDATE_POLICY_ID
    schema_version: str = STRATEGY_CANDIDATE_POLICY_SCHEMA_VERSION
    target_horizon: str = "forward_return_20"
    required_input_metrics: tuple[str, ...] = (
        "mean_rank_ic",
        "oos_rank_ic",
        "positive_fold_ratio",
        "cost_adjusted_excess_return_vs_spy",
        "cost_adjusted_excess_return_vs_equal_weight",
        "max_drawdown",
        "average_daily_turnover",
        "proxy_ic_improvement",
    )
    required_baselines: tuple[str, ...] = STRATEGY_CANDIDATE_DEFAULT_REQUIRED_BASELINES
    rules: tuple[StrategyCandidateMetricRule, ...] = (
        StrategyCandidateMetricRule(
            "mean_rank_ic_positive",
            "mean_rank_ic",
            ">",
            0.0,
            description="mean rank IC must be positive",
        ),
        StrategyCandidateMetricRule(
            "oos_rank_ic_positive",
            "oos_rank_ic",
            ">",
            0.0,
            description="OOS rank IC must be positive",
        ),
        StrategyCandidateMetricRule(
            "positive_fold_ratio_minimum",
            "positive_fold_ratio",
            ">=",
            CANONICAL_MIN_POSITIVE_FOLD_RATIO,
            description="fold_rank_ic > 0 ratio must meet the canonical minimum",
        ),
        StrategyCandidateMetricRule(
            "spy_excess_return_positive",
            "cost_adjusted_excess_return_vs_spy",
            ">",
            0.0,
            description="cost-adjusted excess return versus SPY must be positive",
        ),
        StrategyCandidateMetricRule(
            "equal_weight_excess_return_positive",
            "cost_adjusted_excess_return_vs_equal_weight",
            ">",
            0.0,
            description="cost-adjusted excess return versus equal-weight universe must be positive",
        ),
        StrategyCandidateMetricRule(
            "max_drawdown_floor",
            "max_drawdown",
            ">=",
            -0.20,
            description="maximum drawdown must be no worse than -20%",
        ),
        StrategyCandidateMetricRule(
            "average_daily_turnover_budget",
            "average_daily_turnover",
            "<=",
            0.25,
            description="average daily turnover must be at or below 25%",
        ),
        StrategyCandidateMetricRule(
            "proxy_ic_improvement_minimum",
            "proxy_ic_improvement",
            ">=",
            0.01,
            description="rank IC improvement over proxy must be at least 0.01",
        ),
    )
    warning_rules: tuple[StrategyCandidateMetricRule, ...] = ()

    def __post_init__(self) -> None:
        if not self.policy_id.strip():
            raise ValueError("policy_id must not be empty")
        if not self.schema_version.strip():
            raise ValueError("schema_version must not be empty")
        if self.target_horizon != "forward_return_20":
            raise ValueError("Strategy Candidate Gate policy target_horizon must be forward_return_20")
        if not self.required_input_metrics:
            raise ValueError("required_input_metrics must not be empty")
        duplicate_metrics = _duplicates(self.required_input_metrics)
        if duplicate_metrics:
            raise ValueError(f"required_input_metrics contains duplicates: {', '.join(duplicate_metrics)}")
        duplicate_rules = _duplicates(rule.rule_id for rule in (*self.rules, *self.warning_rules))
        if duplicate_rules:
            raise ValueError(f"rule_id contains duplicates: {', '.join(duplicate_rules)}")
        missing_rule_metrics = sorted(
            {
                rule.metric
                for rule in (*self.rules, *self.warning_rules)
                if rule.metric not in self.required_input_metrics
            }
        )
        if missing_rule_metrics:
            raise ValueError(
                "rule metrics must be listed in required_input_metrics: "
                + ", ".join(missing_rule_metrics)
            )
        if not self.rules:
            raise ValueError("rules must not be empty")
        if not self.required_baselines:
            raise ValueError("required_baselines must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class ValidationGateReport:
    system_validity_status: SystemValidityStatus
    strategy_candidate_status: StrategyCandidateStatus
    hard_fail: bool
    warning: bool
    strategy_pass: bool
    system_validity_pass: bool
    warnings: list[str]
    hard_fail_reasons: list[str]
    metrics: dict[str, Any]
    evidence: dict[str, Any]
    horizons: list[str]
    required_validation_horizon: str
    embargo_periods: dict[str, int]
    benchmark_results: list[dict[str, Any]]
    ablation_results: list[dict[str, Any]]
    gate_results: dict[str, dict[str, Any]]
    official_message: str
    baseline_comparison_inputs: list[dict[str, Any]] = field(default_factory=list)
    cost_adjusted_metric_comparison: list[dict[str, Any]] = field(default_factory=list)
    side_by_side_metric_comparison: list[dict[str, Any]] = field(default_factory=list)
    baseline_comparisons: dict[str, dict[str, Any]] = field(default_factory=dict)
    no_model_proxy_ablation: dict[str, Any] = field(default_factory=dict)
    model_comparison_results: list[dict[str, Any]] = field(default_factory=list)
    structured_warnings: list[dict[str, Any]] = field(default_factory=list)
    comparison_input_schema: dict[str, Any] = field(default_factory=dict)
    comparison_result_schema: dict[str, Any] = field(default_factory=dict)
    full_model_metrics: dict[str, Any] = field(default_factory=dict)
    baseline_metrics: list[dict[str, Any]] = field(default_factory=list)
    ablation_metrics: list[dict[str, Any]] = field(default_factory=list)
    structured_pass_fail_reasons: list[dict[str, Any]] = field(default_factory=list)
    system_validity_gate_input_schema: dict[str, Any] = field(default_factory=dict)
    system_validity_gate_output_schema: dict[str, Any] = field(default_factory=dict)
    system_validity_gate_report_schema: dict[str, Any] = field(default_factory=dict)
    strategy_candidate_policy: dict[str, Any] = field(default_factory=dict)
    strategy_candidate_policy_evaluation: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.system_validity_status not in SYSTEM_VALIDITY_STATUSES:
            allowed = ", ".join(SYSTEM_VALIDITY_STATUSES)
            raise ValueError(f"system_validity_status must be one of: {allowed}")
        if self.strategy_candidate_status not in STRATEGY_CANDIDATE_STATUSES:
            allowed = ", ".join(STRATEGY_CANDIDATE_STATUSES)
            raise ValueError(f"strategy_candidate_status must be one of: {allowed}")

    def to_dict(self) -> dict[str, Any]:
        payload = _json_safe(asdict(self))
        result_summary = _json_safe(self.validity_gate_result_summary)
        payload["validity_gate_result_summary"] = result_summary
        payload["metrics"]["validity_gate_result_summary"] = result_summary
        payload["evidence"]["validity_gate_result_summary"] = result_summary
        payload["final_strategy_status"] = _json_safe(self.final_strategy_status)
        payload["final_strategy_status_explanation"] = _json_safe(
            self.final_strategy_status_explanation
        )
        payload["rule_result_explanations"] = _json_safe(
            self.rule_result_explanations
        )
        payload["strategy_failure_summary"] = _json_safe(self.strategy_failure_summary)
        payload["collapse_failures"] = _json_safe(self.collapse_failures)
        payload["collapse_status"] = _json_safe(self.primary_collapse_status)
        payload["collapse_reason"] = _json_safe(self.primary_collapse_reason)
        payload["collapse_reason_code"] = _json_safe(self.primary_collapse_reason_code)
        payload["baseline_comparison_entries"] = _json_safe(self.baseline_comparison_entries)
        payload["side_by_side_metric_columns"] = _json_safe(self.side_by_side_metric_columns)
        payload["cost_adjusted_metric_comparison_side_by_side"] = _json_safe(
            self.side_by_side_metric_comparison
        )
        payload["stage1_comparison_input_schema"] = _json_safe(
            self.comparison_input_schema
        )
        payload["stage1_comparison_result_schema"] = _json_safe(
            self.comparison_result_schema
        )
        payload["system_validity_gate_input_schema"] = _json_safe(
            self.system_validity_gate_input_schema
        )
        payload["system_validity_gate_output_schema"] = _json_safe(
            self.system_validity_gate_output_schema
        )
        payload["system_validity_gate_report_schema"] = _json_safe(
            self.system_validity_gate_report_schema
        )
        serializable_report = self.serializable_gate_report.to_dict()
        payload["gate_failure_reasons"] = _json_safe(self.gate_failure_reasons)
        structured_failure_report = _json_safe(self.structured_gate_failure_report)
        payload["structured_gate_failure_report"] = structured_failure_report
        payload["serializable_gate_report"] = _json_safe(serializable_report)
        payload["metrics"]["gate_failure_reasons"] = _json_safe(self.gate_failure_reasons)
        payload["evidence"]["gate_failure_reasons"] = _json_safe(self.gate_failure_reasons)
        payload["metrics"]["structured_gate_failure_report"] = structured_failure_report
        payload["evidence"]["structured_gate_failure_report"] = structured_failure_report
        return payload

    @property
    def target_horizon(self) -> int | None:
        value = self.metrics.get("target_horizon")
        return int(value) if value is not None else None

    @property
    def requested_gap_periods(self) -> int | None:
        value = self.metrics.get("requested_gap_periods")
        return int(value) if value is not None else None

    @property
    def requested_embargo_periods(self) -> int | None:
        value = self.metrics.get("requested_embargo_periods")
        return int(value) if value is not None else None

    @property
    def effective_gap_periods(self) -> int | None:
        value = self.metrics.get("effective_gap_periods")
        return int(value) if value is not None else None

    @property
    def effective_embargo_periods(self) -> int | None:
        value = self.metrics.get("effective_embargo_periods")
        return int(value) if value is not None else None

    @property
    def baseline_comparison_entries(self) -> list[dict[str, Any]]:
        return list(self.baseline_comparisons.values())

    @property
    def side_by_side_metric_columns(self) -> dict[str, dict[str, Any]]:
        return _side_by_side_metric_columns(self.side_by_side_metric_comparison)

    @property
    def final_strategy_status(self) -> StrategyCandidateStatus:
        return self.strategy_candidate_status

    @property
    def final_strategy_status_explanation(self) -> dict[str, Any]:
        return _final_strategy_status_explanation(self)

    @property
    def validity_gate_result_summary(self) -> dict[str, Any]:
        return _validity_gate_result_summary(self)

    @property
    def gate_failure_reasons(self) -> list[dict[str, Any]]:
        return _gate_failure_reasons(self)

    @property
    def structured_gate_failure_report(self) -> dict[str, Any]:
        return _structured_gate_failure_report(self)

    @property
    def serializable_gate_report(self) -> SerializableValidityGateReport:
        return SerializableValidityGateReport(
            system_validity_status=self.system_validity_status,
            strategy_candidate_status=self.strategy_candidate_status,
            system_validity_pass=self.system_validity_pass,
            strategy_pass=self.strategy_pass,
            hard_fail=self.hard_fail,
            warning=self.warning,
            official_message=self.official_message,
            gate_failure_reasons=tuple(
                GateFailureReason.from_mapping(row) for row in self.gate_failure_reasons
            ),
            gate_results=_json_safe(self.gate_results),
            metrics=_json_safe(self.metrics),
            evidence=_json_safe(self.evidence),
            structured_gate_failure_report=_json_safe(self.structured_gate_failure_report),
            artifact_manifest={
                "json_artifact": "validity_gate_report.json",
                "markdown_artifact": "validity_gate_report.md",
                "schema_embedded_in_top_level_payload": True,
                "gate_failure_reasons_field": "gate_failure_reasons",
                "structured_gate_failure_report_field": "structured_gate_failure_report",
            },
            report_path="reports/validity_report.md",
        )

    @property
    def rule_result_explanations(self) -> list[dict[str, Any]]:
        return _rule_result_explanations(self.gate_results)

    @property
    def strategy_failure_summary(self) -> list[dict[str, Any]]:
        return _strategy_failure_summary(self.gate_results)

    @property
    def collapse_failures(self) -> list[dict[str, Any]]:
        return [
            row
            for row in self.strategy_failure_summary
            if row.get("collapse_status") == "fail"
        ]

    @property
    def primary_collapse_status(self) -> str | None:
        if self.collapse_failures:
            return str(self.collapse_failures[0]["collapse_status"])
        return None

    @property
    def primary_collapse_reason(self) -> str | None:
        if self.collapse_failures:
            return str(self.collapse_failures[0]["collapse_reason"])
        return None

    @property
    def primary_collapse_reason_code(self) -> str | None:
        if self.collapse_failures:
            return str(self.collapse_failures[0]["reason_code"])
        return None

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_markdown(self) -> str:
        lines = [
            "# Validity Gate Report",
            "",
            f"- System validity: `{self.system_validity_status}`",
            f"- Strategy candidate: `{self.strategy_candidate_status}`",
            f"- Hard fail: `{self.hard_fail}`",
            f"- Warning: `{self.warning}`",
            f"- Strategy pass: `{self.strategy_pass}`",
            f"- System validity pass: `{self.system_validity_pass}`",
            f"- Positive fold ratio: {_format_metric(self.metrics.get('positive_fold_ratio'))}",
            f"- Positive fold ratio threshold: {_format_metric(self.metrics.get('positive_fold_ratio_threshold'))}",
            f"- Positive fold ratio threshold passed: `{self.metrics.get('positive_fold_ratio_passed')}`",
            f"- Official message: {self.official_message}",
        ]
        if self.primary_collapse_status:
            lines.extend(
                [
                    f"- Collapse status: `{self.primary_collapse_status}`",
                    f"- Collapse reason: {self.primary_collapse_reason}",
                ]
            )
        lines.append("")

        result_summary = self.validity_gate_result_summary
        lines.extend(
            [
                "## Validity Gate Result Summary",
                "",
                "| Field | Value |",
                "|---|---|",
                f"| System validity status | `{result_summary.get('system_validity_status')}` |",
                f"| Strategy candidate status | `{result_summary.get('strategy_candidate_status')}` |",
                f"| Final gate decision | `{result_summary.get('final_gate_decision')}` |",
                f"| Failure reason count | {_format_count_metric(result_summary.get('failure_reason_count'))} |",
                f"| Warning count | {_format_count_metric(result_summary.get('warning_count'))} |",
                f"| Key metric count | {_format_count_metric(result_summary.get('key_metric_count'))} |",
                f"| Official message | {_markdown_cell(result_summary.get('official_message'))} |",
                "",
            ]
        )
        summary_failures = result_summary.get("failure_reasons")
        if summary_failures:
            lines.extend(["### Summary Failure Reasons", ""])
            lines.extend(
                f"- {row.get('gate')}: {_markdown_cell(row.get('reason'))}"
                for row in summary_failures
                if isinstance(row, Mapping)
            )
            lines.append("")
        summary_warnings = result_summary.get("warnings")
        if summary_warnings:
            lines.extend(["### Summary Warnings", ""])
            lines.extend(
                f"- {row.get('gate')}: {_markdown_cell(row.get('message'))}"
                for row in summary_warnings
                if isinstance(row, Mapping)
            )
            lines.append("")
        summary_metrics = _mapping_or_empty(result_summary.get("key_metrics"))
        if summary_metrics:
            lines.extend(
                [
                    "### Summary Key Metrics",
                    "",
                    "| Metric | Value |",
                    "|---|---:|",
                ]
            )
            for metric, value in summary_metrics.items():
                lines.append(f"| {metric} | {_format_metric(value)} |")
            lines.append("")

        failure_report = self.structured_gate_failure_report
        failure_gates = failure_report.get("gates", [])
        if failure_gates:
            lines.extend(
                [
                    "## Structured Gate Failure Report",
                    "",
                    "| Gate | Status | Severity | Reason Count | Metrics | Top Reason |",
                    "|---|---|---|---:|---|---|",
                ]
            )
            for row in failure_gates:
                metrics = ", ".join(
                    str(metric.get("metric"))
                    for metric in row.get("related_metrics", [])
                    if isinstance(metric, Mapping) and metric.get("metric") is not None
                )
                lines.append(
                    "| "
                    f"{row.get('gate', '')} | "
                    f"{row.get('status', '')} | "
                    f"{row.get('severity', '')} | "
                    f"{_format_count_metric(row.get('reason_count'))} | "
                    f"{_markdown_cell(metrics)} | "
                    f"{_markdown_cell(row.get('top_reason'))} |"
                )
            lines.append("")

        final_status_explanation = self.final_strategy_status_explanation
        lines.extend(
            [
                "## Final Strategy Status Explanation",
                "",
                "| Field | Value |",
                "|---|---|",
                f"| Final strategy status | `{self.final_strategy_status}` |",
                f"| System validity status | `{self.system_validity_status}` |",
                f"| Strategy pass | `{self.strategy_pass}` |",
                f"| Blocking rules | {_format_sequence(final_status_explanation.get('blocking_rules'))} |",
                f"| Warning rules | {_format_sequence(final_status_explanation.get('warning_rules'))} |",
                f"| Insufficient data rules | {_format_sequence(final_status_explanation.get('insufficient_data_rules'))} |",
                f"| Reason | {_markdown_cell(final_status_explanation.get('reason'))} |",
                f"| Official message | {_markdown_cell(final_status_explanation.get('official_message'))} |",
                "",
            ]
        )

        if self.rule_result_explanations:
            lines.extend(
                [
                    "## Rule Result Explanations",
                    "",
                    "| Rule | Status | Passed | Affects Strategy | Affects System | Reason | Reason Code | Metric | Value | Threshold | Operator |",
                    "|---|---|---|---:|---:|---|---|---|---:|---:|---|",
                ]
            )
            for row in self.rule_result_explanations:
                lines.append(
                    "| "
                    f"{row.get('rule', '')} | "
                    f"{row.get('status', '')} | "
                    f"{_markdown_cell(row.get('passed'))} | "
                    f"{row.get('affects_strategy')} | "
                    f"{row.get('affects_system')} | "
                    f"{_markdown_cell(row.get('reason'))} | "
                    f"{_markdown_cell(row.get('reason_code'))} | "
                    f"{_markdown_cell(row.get('metric'))} | "
                    f"{_format_metric(row.get('value'))} | "
                    f"{_format_metric(row.get('threshold'))} | "
                    f"{_markdown_cell(row.get('operator'))} |"
                )
            lines.append("")

        if self.strategy_failure_summary:
            lines.extend(
                [
                    "## Strategy Failure Summary",
                    "",
                    "| Gate | Status | Reason | Reason Code | Collapse Status | Collapse Reason | Metric | Value | Threshold | Operator |",
                    "|---|---|---|---|---|---|---|---:|---:|---|",
                ]
            )
            for row in self.strategy_failure_summary:
                lines.append(
                    "| "
                    f"{row.get('gate', '')} | "
                    f"{row.get('status', '')} | "
                    f"{_markdown_cell(row.get('reason'))} | "
                    f"{_markdown_cell(row.get('reason_code'))} | "
                    f"{_markdown_cell(row.get('collapse_status'))} | "
                    f"{_markdown_cell(row.get('collapse_reason'))} | "
                    f"{_markdown_cell(row.get('metric'))} | "
                    f"{_format_metric(row.get('value'))} | "
                    f"{_format_metric(row.get('threshold'))} | "
                    f"{_markdown_cell(row.get('operator'))} |"
                )
            lines.append("")

        lines.extend(
            [
                "## Gate Results",
                "",
                "| Gate | Status | Reason |",
                "|---|---|---|",
            ]
        )
        for name, result in self.gate_results.items():
            reason = str(result.get("reason", "")).replace("|", "\\|")
            lines.append(f"| {name} | {result.get('status', '')} | {reason} |")

        horizon_metrics = _mapping_or_empty(self.metrics.get("horizon_metrics"))
        if horizon_metrics:
            lines.extend(
                [
                    "",
                    "## Horizon Diagnostics",
                    "",
                    "| Horizon | Label | Role | Target | Affects Pass/Fail | Status | Insufficient Data | Insufficient Data Status | Insufficient Data Code | Insufficient Data Reason | Mean Rank IC | Positive Fold Ratio | Positive Fold Ratio Threshold | Positive Fold Ratio Passed | OOS Rank IC |",
                    "|---|---|---|---|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|",
                ]
            )
            for horizon in self.horizons:
                row = _mapping_or_empty(horizon_metrics.get(horizon))
                if not row:
                    continue
                lines.append(
                    "| "
                    f"{horizon} | "
                    f"{row.get('label', row.get('role', ''))} | "
                    f"{row.get('role', '')} | "
                    f"{row.get('target_column', '')} | "
                    f"{row.get('affects_pass_fail', False)} | "
                    f"{row.get('status', row.get('rank_ic_status', ''))} | "
                    f"{row.get('insufficient_data', False)} | "
                    f"{_markdown_cell(row.get('insufficient_data_status'))} | "
                    f"{_markdown_cell(row.get('insufficient_data_code'))} | "
                    f"{_markdown_cell(row.get('insufficient_data_reason'))} | "
                    f"{_format_metric(row.get('mean_rank_ic'))} | "
                    f"{_format_metric(row.get('positive_fold_ratio'))} | "
                    f"{_format_metric(row.get('positive_fold_ratio_threshold'))} | "
                    f"{_markdown_cell(row.get('positive_fold_ratio_passed'))} | "
                    f"{_format_metric(row.get('oos_rank_ic'))} |"
                )

        if self.cost_adjusted_metric_comparison:
            lines.extend(
                [
                    "",
                    "## Cost-Adjusted Strategy Comparison",
                    "",
                    "| Name | Role | Return Basis | CAGR | Sharpe | Max Drawdown | Cost-Adjusted Cumulative Return | Avg Daily Turnover | Total Cost Return | Excess Return | Excess Return Status |",
                    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
                ]
            )
            for row in self.cost_adjusted_metric_comparison:
                lines.append(
                    "| "
                    f"{row.get('name', '')} | "
                    f"{row.get('role', '')} | "
                    f"{row.get('return_basis', '')} | "
                    f"{_format_metric(row.get('cagr'))} | "
                    f"{_format_metric(row.get('sharpe'))} | "
                    f"{_format_metric(row.get('max_drawdown'))} | "
                    f"{_format_metric(row.get('cost_adjusted_cumulative_return'))} | "
                    f"{_format_metric(row.get('average_daily_turnover'))} | "
                    f"{_format_metric(row.get('total_cost_return'))} | "
                    f"{_format_metric(_row_excess_return(row))} | "
                    f"{_markdown_cell(row.get('excess_return_status'))} |"
                )

        report_only_research_metrics = _report_only_research_metric_rows(self.evidence)
        if report_only_research_metrics:
            lines.extend(
                [
                    "",
                    "## Report-Only Research Metrics",
                    "",
                    "| Metric | Status | Value | Target | Sample Scope | Report Only | Decision Use | Reason |",
                    "|---|---|---:|---|---|---:|---|---|",
                ]
            )
            for row in report_only_research_metrics:
                lines.append(
                    "| "
                    f"{_markdown_cell(row.get('metric'))} | "
                    f"{_markdown_cell(row.get('status'))} | "
                    f"{_format_metric(row.get('value'))} | "
                    f"{_markdown_cell(row.get('target_column'))} | "
                    f"{_markdown_cell(row.get('sample_scope'))} | "
                    f"{_markdown_cell(row.get('report_only'))} | "
                    f"{_markdown_cell(row.get('decision_use'))} | "
                    f"{_markdown_cell(row.get('reason'))} |"
                )

        covariance_risk = _mapping_or_empty(self.metrics.get("covariance_aware_risk"))
        if covariance_risk:
            parameters = _mapping_or_empty(covariance_risk.get("parameters"))
            realized = _mapping_or_empty(covariance_risk.get("realized_metrics"))
            lines.extend(
                [
                    "",
                    "## Covariance-Aware Risk",
                    "",
                    "| Field | Value |",
                    "|---|---:|",
                    f"| Applied | {covariance_risk.get('applied')} |",
                    f"| Configured enabled | {covariance_risk.get('configured_enabled')} |",
                    f"| Status | {_markdown_cell(covariance_risk.get('status'))} |",
                    f"| Return column | {_markdown_cell(parameters.get('return_column'))} |",
                    f"| Lookback periods | {_format_count_metric(parameters.get('lookback_periods'))} |",
                    f"| Min periods | {_format_count_metric(parameters.get('min_periods'))} |",
                    f"| Fallback | {_markdown_cell(parameters.get('fallback'))} |",
                    f"| Max holdings | {_format_count_metric(parameters.get('max_holdings'))} |",
                    f"| Max symbol weight | {_format_metric(parameters.get('max_symbol_weight'))} |",
                    f"| Max sector weight | {_format_metric(parameters.get('max_sector_weight'))} |",
                    f"| Portfolio volatility limit | {_format_metric(parameters.get('portfolio_volatility_limit'))} |",
                    f"| Max position risk contribution | {_format_metric(parameters.get('max_position_risk_contribution'))} |",
                    f"| Average covariance volatility | {_format_metric(realized.get('average_portfolio_volatility_estimate'))} |",
                    f"| Max covariance volatility | {_format_metric(realized.get('max_portfolio_volatility_estimate'))} |",
                    f"| Latest sizing validation | {_markdown_cell(realized.get('latest_position_sizing_validation_status'))} |",
                ]
            )

        if self.full_model_metrics:
            row = self.full_model_metrics
            metrics = _mapping_or_empty(row.get("metrics"))
            validation = _mapping_or_empty(row.get("validation_metrics"))
            lines.extend(
                [
                    "",
                    "## Full Model Metrics",
                    "",
                    "| Field | Value |",
                    "|---|---:|",
                    f"| Entity | {_markdown_cell(row.get('entity_id'))} |",
                    f"| Status | {_markdown_cell(row.get('status'))} |",
                    f"| Rank IC | {_format_metric(validation.get('mean_rank_ic'))} |",
                    f"| Positive Fold Ratio | {_format_metric(validation.get('positive_fold_ratio'))} |",
                    f"| Positive Fold Ratio Threshold | {_format_metric(validation.get('positive_fold_ratio_threshold'))} |",
                    f"| Positive Fold Ratio Passed | {_markdown_cell(validation.get('positive_fold_ratio_passed'))} |",
                    f"| OOS Rank IC | {_format_metric(validation.get('oos_rank_ic'))} |",
                    f"| Sharpe | {_format_metric(metrics.get('sharpe'))} |",
                    f"| Max Drawdown | {_format_metric(metrics.get('max_drawdown'))} |",
                    f"| Cost-Adjusted Cumulative Return | {_format_metric(metrics.get('cost_adjusted_cumulative_return'))} |",
                    f"| Average Daily Turnover | {_format_metric(metrics.get('average_daily_turnover'))} |",
                ]
            )

        if self.baseline_metrics:
            lines.extend(
                [
                    "",
                    "## Baseline Metrics",
                    "",
                    "| Entity | Role | Status | Sharpe | Max Drawdown | Cost-Adjusted Cumulative Return | Excess Return | Turnover |",
                    "|---|---|---|---:|---:|---:|---:|---:|",
                ]
            )
            for row in self.baseline_metrics:
                metrics = _mapping_or_empty(row.get("metrics"))
                lines.append(
                    "| "
                    f"{row.get('entity_id', '')} | "
                    f"{row.get('role', '')} | "
                    f"{row.get('status', '')} | "
                    f"{_format_metric(metrics.get('sharpe'))} | "
                    f"{_format_metric(metrics.get('max_drawdown'))} | "
                    f"{_format_metric(metrics.get('cost_adjusted_cumulative_return'))} | "
                    f"{_format_metric(metrics.get('excess_return'))} | "
                    f"{_format_metric(metrics.get('turnover', metrics.get('average_daily_turnover')))} |"
                )

        if self.ablation_metrics:
            lines.extend(
                [
                    "",
                    "## Ablation Metrics",
                    "",
                    "| Scenario | Role | Kind | Status | Sharpe | Rank IC | Cost-Adjusted Cumulative Return | Turnover |",
                    "|---|---|---|---|---:|---:|---:|---:|",
                ]
            )
            for row in self.ablation_metrics:
                metrics = _mapping_or_empty(row.get("metrics"))
                lines.append(
                    "| "
                    f"{row.get('entity_id', '')} | "
                    f"{row.get('role', '')} | "
                    f"{row.get('kind', '')} | "
                    f"{row.get('status', '')} | "
                    f"{_format_metric(metrics.get('sharpe'))} | "
                    f"{_format_metric(metrics.get('rank_ic', metrics.get('mean_rank_ic')))} | "
                    f"{_format_metric(metrics.get('cost_adjusted_cumulative_return'))} | "
                    f"{_format_metric(metrics.get('turnover', metrics.get('average_daily_turnover')))} |"
                )

        if self.structured_pass_fail_reasons:
            lines.extend(
                [
                    "",
                    "## Structured Pass/Fail Reasons",
                    "",
                    "| Category | Entity | Rule | Metric | Status | Passed | Reason Code | Reason |",
                    "|---|---|---|---|---|---|---|---|",
                ]
            )
            for row in self.structured_pass_fail_reasons:
                lines.append(
                    "| "
                    f"{row.get('category', '')} | "
                    f"{row.get('entity_id', '')} | "
                    f"{row.get('rule', '')} | "
                    f"{row.get('metric', '')} | "
                    f"{row.get('status', '')} | "
                    f"{_markdown_cell(row.get('passed'))} | "
                    f"{_markdown_cell(row.get('reason_code'))} | "
                    f"{_markdown_cell(row.get('reason'))} |"
                )

        if self.side_by_side_metric_comparison:
            columns = _side_by_side_metric_entity_columns(self.side_by_side_metric_comparison)
            header = "| Metric | " + " | ".join(_comparison_display_name(name) for name in columns) + " |"
            separator = "|---|" + "|".join("---:" if _side_by_side_column_is_numeric(self.side_by_side_metric_comparison, name) else "---" for name in columns) + "|"
            lines.extend(
                [
                    "",
                    "## Cost-Adjusted Side-by-Side Metrics",
                    "",
                    header,
                    separator,
                ]
            )
            for row in self.side_by_side_metric_comparison:
                metric = str(row.get("metric", ""))
                values = [
                    _format_side_by_side_metric_value(metric, row.get(name))
                    for name in columns
                ]
                lines.append(
                    "| "
                    f"{_markdown_cell(row.get('metric_label', metric))} | "
                    + " | ".join(values)
                    + " |"
                )

        if self.baseline_comparisons:
            lines.extend(
                [
                    "",
                    "## Baseline Comparisons",
                    "",
                    "| Baseline | Type | Return Basis | CAGR | Sharpe | Max Drawdown | Cost-Adjusted Cumulative Return | Avg Daily Turnover | Total Cost Return | Excess Return | Excess Return Status |",
                    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
                ]
            )
            for row in self.baseline_comparison_entries:
                lines.append(
                    "| "
                    f"{row.get('name', '')} | "
                    f"{row.get('baseline_type', '')} | "
                    f"{row.get('return_basis', '')} | "
                    f"{_format_metric(row.get('cagr'))} | "
                    f"{_format_metric(row.get('sharpe'))} | "
                    f"{_format_metric(row.get('max_drawdown'))} | "
                    f"{_format_metric(row.get('cost_adjusted_cumulative_return'))} | "
                    f"{_format_metric(row.get('average_daily_turnover'))} | "
                    f"{_format_metric(row.get('total_cost_return'))} | "
                    f"{_format_metric(_row_excess_return(row))} | "
                    f"{_markdown_cell(row.get('excess_return_status'))} |"
                )

        if self.baseline_comparison_inputs:
            lines.extend(
                [
                    "",
                    "## Baseline Comparison Inputs",
                    "",
                    "| Baseline | Type | Return Basis | Data Source | Return Column | Horizon | Required | Benchmark | Universe |",
                    "|---|---|---|---|---|---:|---:|---|---|",
                ]
            )
            for row in self.baseline_comparison_inputs:
                lines.append(
                    "| "
                    f"{row.get('name', '')} | "
                    f"{row.get('baseline_type', '')} | "
                    f"{row.get('return_basis', '')} | "
                    f"{row.get('data_source', '')} | "
                    f"{row.get('return_column', '')} | "
                    f"{row.get('return_horizon', '')} | "
                    f"{row.get('required_for_stage1', '')} | "
                    f"{row.get('benchmark_ticker') or ''} | "
                    f"{_format_sequence(row.get('universe_tickers'))} |"
                )

        if self.no_model_proxy_ablation:
            ablation = self.no_model_proxy_ablation
            controls = _mapping_or_empty(ablation.get("pipeline_controls"))
            performance = _mapping_or_empty(ablation.get("performance_metrics"))
            validation = _mapping_or_empty(ablation.get("validation_metrics"))
            signal_metrics = _mapping_or_empty(
                ablation.get("deterministic_signal_evaluation_metrics")
            )
            action_counts = _mapping_or_empty(signal_metrics.get("action_counts"))
            lines.extend(
                [
                    "",
                    "## No-Model-Proxy Ablation",
                    "",
                    "| Field | Value |",
                    "|---|---:|",
                    f"| Available | {ablation.get('available', False)} |",
                    f"| Model proxy enabled | {controls.get('model_proxy', '')} |",
                    f"| Validation status | {validation.get('validation_status', ablation.get('status', ''))} |",
                    f"| Fold count | {validation.get('validation_fold_count', '')} |",
                    f"| OOS fold count | {validation.get('validation_oos_fold_count', '')} |",
                    f"| Sharpe | {_format_metric(performance.get('sharpe'))} |",
                    f"| Excess return | {_format_metric(performance.get('excess_return'))} |",
                    f"| Cost-adjusted cumulative return | {_format_metric(signal_metrics.get('cost_adjusted_cumulative_return'))} |",
                    f"| Average daily turnover | {_format_metric(signal_metrics.get('average_daily_turnover'))} |",
                    f"| Buy / Sell / Hold | {action_counts.get('BUY', 0)} / {action_counts.get('SELL', 0)} / {action_counts.get('HOLD', 0)} |",
                ]
            )

        if self.model_comparison_results:
            lines.extend(
                [
                    "",
                    "## Model Comparison Results",
                    "",
                    "| Window | Metric | Candidate | Candidate Value | Comparison Target | Target Role | Target Value | Absolute Delta | Relative Delta | Pass/Fail |",
                    "|---|---|---|---:|---|---|---:|---:|---:|---|",
                ]
            )
            for row in self.model_comparison_results:
                lines.append(
                    "| "
                    f"{row.get('window_id', '')} | "
                    f"{row.get('metric', '')} | "
                    f"{row.get('candidate', '')} | "
                    f"{_format_metric(row.get('candidate_value'))} | "
                    f"{row.get('baseline', '')} | "
                    f"{row.get('baseline_role', '')} | "
                    f"{_format_metric(row.get('baseline_value'))} | "
                    f"{_format_metric(row.get('absolute_delta', row.get('delta')))} | "
                    f"{_format_metric(row.get('relative_delta'))} | "
                    f"{row.get('pass_fail', '')} |"
                )

        if self.comparison_input_schema:
            schema = self.comparison_input_schema
            full_model = _mapping_or_empty(schema.get("full_model"))
            lines.extend(
                [
                    "",
                    "## Stage 1 Comparison Input Schema",
                    "",
                    "| Field | Value |",
                    "|---|---|",
                    f"| Schema version | {_markdown_cell(schema.get('schema_version'))} |",
                    f"| Comparison ID | {_markdown_cell(schema.get('comparison_id'))} |",
                    f"| Full model | {_markdown_cell(full_model.get('entity_id'))} |",
                    f"| Baselines | {_format_count_metric(_sequence_length(schema.get('baselines')))} |",
                    f"| Ablations | {_format_count_metric(_sequence_length(schema.get('ablations')))} |",
                    f"| Metrics | {_format_count_metric(_sequence_length(schema.get('metrics')))} |",
                    f"| Validation windows | {_format_count_metric(_sequence_length(schema.get('validation_windows')))} |",
                    f"| Signal engine | {_markdown_cell(full_model.get('signal_engine'))} |",
                    f"| Model predictions are order signals | {full_model.get('model_predictions_are_order_signals')} |",
                    f"| LLM makes trading decisions | {full_model.get('llm_makes_trading_decisions')} |",
                ]
            )

        if self.comparison_result_schema:
            schema = self.comparison_result_schema
            full_model_result = _mapping_or_empty(schema.get("full_model_result"))
            lines.extend(
                [
                    "",
                    "## Stage 1 Comparison Result Schema",
                    "",
                    "| Field | Value |",
                    "|---|---|",
                    f"| Schema version | {_markdown_cell(schema.get('schema_version'))} |",
                    f"| Comparison ID | {_markdown_cell(schema.get('comparison_id'))} |",
                    f"| Full model result | {_markdown_cell(full_model_result.get('entity_id'))} |",
                    f"| Baseline results | {_format_count_metric(_sequence_length(schema.get('baseline_results')))} |",
                    f"| Ablation results | {_format_count_metric(_sequence_length(schema.get('ablation_results')))} |",
                    f"| Metric results | {_format_count_metric(_sequence_length(schema.get('metric_results')))} |",
                    f"| Validation windows | {_format_count_metric(_sequence_length(schema.get('validation_windows')))} |",
                ]
            )

        if self.warnings:
            lines.extend(["", "## Warnings", ""])
            lines.extend(f"- {warning}" for warning in self.warnings)

        if self.structured_warnings:
            lines.extend(
                [
                    "",
                    "## Structured Warnings",
                    "",
                    "| Code | Gate | Combined Gate | Severity | Metric | Realized Turnover | Budget | Value | Threshold | Operator | Message | Reason |",
                    "|---|---|---|---|---|---:|---:|---:|---:|---|---|---|",
                ]
            )
            for warning in self.structured_warnings:
                lines.append(
                    "| "
                    f"{warning.get('code', '')} | "
                    f"{warning.get('gate', '')} | "
                    f"{warning.get('combined_gate', '')} | "
                    f"{warning.get('severity', '')} | "
                    f"{warning.get('metric', '')} | "
                    f"{_format_metric(warning.get('realized_turnover'))} | "
                    f"{_format_metric(warning.get('budget'))} | "
                    f"{_format_metric(warning.get('value'))} | "
                    f"{_format_metric(warning.get('threshold'))} | "
                    f"{warning.get('operator', '')} | "
                    f"{_markdown_cell(warning.get('message'))} | "
                    f"{_markdown_cell(warning.get('reason'))} |"
                )

        if self.hard_fail_reasons:
            lines.extend(["", "## Hard Fail Reasons", ""])
            lines.extend(f"- {reason}" for reason in self.hard_fail_reasons)

        if self.benchmark_results:
            lines.extend(
                [
                    "",
                    "## Benchmark Results",
                    "",
                    "| Baseline | CAGR | Sharpe | Max Drawdown | Excess Return | Excess Return Status | Avg Daily Turnover |",
                    "|---|---:|---:|---:|---:|---|---:|",
                ]
            )
            for row in self.benchmark_results:
                lines.append(
                    "| "
                    f"{row.get('name', '')} | "
                    f"{_format_metric(row.get('cagr'))} | "
                    f"{_format_metric(row.get('sharpe'))} | "
                    f"{_format_metric(row.get('max_drawdown'))} | "
                    f"{_format_metric(row.get('excess_return'))} | "
                    f"{_markdown_cell(row.get('excess_return_status'))} | "
                    f"{_format_metric(row.get('average_daily_turnover'))} |"
                )

        stage1_ablation_rows = _stage1_ablation_comparison_results(self.ablation_results)
        if stage1_ablation_rows:
            lines.extend(
                [
                    "",
                    "## Stage 1 Ablation Scenario Comparison",
                    "",
                    "| Scenario | Kind | Return Basis | CAGR | Sharpe | Max Drawdown | Excess Return | Cost-Adjusted Cumulative Return | Avg Daily Turnover | Total Cost Return | Effective Cost bps | Effective Slippage bps | Validation Status | Fold Count | OOS Fold Count | Feature Families | Model Proxy | Cost | Slippage | Turnover |",
                    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---:|",
                ]
            )
            for row in stage1_ablation_rows:
                signal_metrics = _ablation_signal_metrics(row)
                controls = _mapping_or_empty(row.get("pipeline_controls"))
                lines.append(
                    "| "
                    f"{row.get('scenario', '')} | "
                    f"{row.get('kind', '')} | "
                    f"{signal_metrics.get('return_basis', row.get('signal_return_basis', ''))} | "
                    f"{_format_metric(row.get('cagr'))} | "
                    f"{_format_metric(row.get('sharpe'))} | "
                    f"{_format_metric(row.get('max_drawdown'))} | "
                    f"{_format_metric(row.get('excess_return'))} | "
                    f"{_format_metric(_ablation_signal_metric(row, signal_metrics, 'cost_adjusted_cumulative_return'))} | "
                    f"{_format_metric(_ablation_signal_metric(row, signal_metrics, 'average_daily_turnover'))} | "
                    f"{_format_metric(_ablation_signal_metric(row, signal_metrics, 'total_cost_return'))} | "
                    f"{_format_metric(row.get('effective_cost_bps'))} | "
                    f"{_format_metric(row.get('effective_slippage_bps'))} | "
                    f"{row.get('validation_status', '')} | "
                    f"{_format_count_metric(row.get('validation_fold_count'))} | "
                    f"{_format_count_metric(row.get('validation_oos_fold_count'))} | "
                    f"{_format_feature_families(row)} | "
                    f"{controls.get('model_proxy', '')} | "
                    f"{controls.get('cost', '')} | "
                    f"{controls.get('slippage', '')} | "
                    f"{controls.get('turnover', '')} |"
                )

        pipeline_controls = _pipeline_control_results(self.ablation_results)
        if pipeline_controls:
            lines.extend(
                [
                    "",
                    "## Pipeline Controls",
                    "",
                    "| Scenario | Model Proxy | Cost | Slippage | Turnover | Effective Cost bps | Effective Slippage bps | Signal Cost-Adjusted Return | Signal Turnover | Buy | Sell | Hold | Risk Stops |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in pipeline_controls:
                controls = row.get("pipeline_controls", {})
                if not isinstance(controls, Mapping):
                    controls = {}
                lines.append(
                    "| "
                    f"{row.get('scenario', '')} | "
                    f"{controls.get('model_proxy', '')} | "
                    f"{controls.get('cost', '')} | "
                    f"{controls.get('slippage', '')} | "
                    f"{controls.get('turnover', '')} | "
                    f"{row.get('effective_cost_bps', '')} | "
                    f"{row.get('effective_slippage_bps', '')} |"
                    f" {_format_metric(row.get('signal_cost_adjusted_cumulative_return'))} |"
                    f" {_format_metric(row.get('signal_average_daily_turnover'))} |"
                    f" {row.get('signal_buy_count', '')} |"
                    f" {row.get('signal_sell_count', '')} |"
                    f" {row.get('signal_hold_count', '')} |"
                    f" {row.get('signal_risk_stop_observation_count', '')} |"
                )

        cost_ablation_controls = _cost_ablation_results(self.ablation_results)
        if cost_ablation_controls:
            lines.extend(
                [
                    "",
                    "## Cost Ablations",
                    "",
                    "| Scenario | Cost | Slippage | Turnover | Effective Cost bps | Effective Slippage bps | Signal Cost-Adjusted Return | Signal Turnover | Total Cost Return |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in cost_ablation_controls:
                controls = row.get("pipeline_controls", {})
                if not isinstance(controls, Mapping):
                    controls = {}
                lines.append(
                    "| "
                    f"{row.get('scenario', '')} | "
                    f"{controls.get('cost', '')} | "
                    f"{controls.get('slippage', '')} | "
                    f"{controls.get('turnover', '')} | "
                    f"{row.get('effective_cost_bps', '')} | "
                    f"{row.get('effective_slippage_bps', '')} |"
                    f" {_format_metric(row.get('signal_cost_adjusted_cumulative_return'))} |"
                    f" {_format_metric(row.get('signal_average_daily_turnover'))} |"
                    f" {_format_metric(row.get('signal_total_cost_return'))} |"
                )

        lines.extend(
            [
                "",
                "## Metrics",
                "",
                "```json",
                json.dumps(_json_safe(self.metrics), ensure_ascii=False, indent=2),
                "```",
                "",
                "## Evidence",
                "",
                "```json",
                json.dumps(_json_safe(self.evidence), ensure_ascii=False, indent=2),
                "```",
            ]
        )
        return "\n".join(lines)

    def to_html(self) -> str:
        lines = [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            "<title>Validity Gate Report</title>",
            "</head>",
            "<body>",
            "<h1>Validity Gate Report</h1>",
        ]
        status_rows: list[tuple[str, object]] = [
            ("System validity", self.system_validity_status),
            ("Strategy candidate", self.strategy_candidate_status),
            ("Hard fail", self.hard_fail),
            ("Warning", self.warning),
            ("Strategy pass", self.strategy_pass),
            ("System validity pass", self.system_validity_pass),
            ("Positive fold ratio", _format_metric(self.metrics.get("positive_fold_ratio"))),
            (
                "Positive fold ratio threshold",
                _format_metric(self.metrics.get("positive_fold_ratio_threshold")),
            ),
            (
                "Positive fold ratio threshold passed",
                self.metrics.get("positive_fold_ratio_passed"),
            ),
            ("Official message", self.official_message),
        ]
        if self.primary_collapse_status:
            status_rows.extend(
                [
                    ("Collapse status", self.primary_collapse_status),
                    ("Collapse reason", self.primary_collapse_reason),
                ]
            )
        lines.extend(
            _html_table(
                "Status",
                ("Field", "Value"),
                status_rows,
            )
        )
        result_summary = self.validity_gate_result_summary
        lines.extend(
            _html_table(
                "Validity Gate Result Summary",
                ("Field", "Value"),
                (
                    ("System validity status", result_summary.get("system_validity_status")),
                    (
                        "Strategy candidate status",
                        result_summary.get("strategy_candidate_status"),
                    ),
                    ("Final gate decision", result_summary.get("final_gate_decision")),
                    ("Failure reason count", result_summary.get("failure_reason_count")),
                    ("Warning count", result_summary.get("warning_count")),
                    ("Key metric count", result_summary.get("key_metric_count")),
                    ("Official message", result_summary.get("official_message")),
                ),
            )
        )
        summary_failures = [
            row
            for row in result_summary.get("failure_reasons", [])
            if isinstance(row, Mapping)
        ]
        if summary_failures:
            lines.extend(
                _html_table(
                    "Summary Failure Reasons",
                    ("Gate", "Status", "Reason Code", "Reason", "Metric", "Value", "Threshold"),
                    (
                        (
                            row.get("gate", ""),
                            row.get("status", ""),
                            row.get("reason_code", ""),
                            row.get("reason", ""),
                            row.get("metric", ""),
                            _format_metric(row.get("value")),
                            _format_metric(row.get("threshold")),
                        )
                        for row in summary_failures
                    ),
                )
            )
        summary_warnings = [
            row
            for row in result_summary.get("warnings", [])
            if isinstance(row, Mapping)
        ]
        if summary_warnings:
            lines.extend(
                _html_table(
                    "Summary Warnings",
                    ("Gate", "Code", "Metric", "Message", "Value", "Threshold"),
                    (
                        (
                            row.get("gate", ""),
                            row.get("code", ""),
                            row.get("metric", ""),
                            row.get("message", ""),
                            _format_metric(row.get("value")),
                            _format_metric(row.get("threshold")),
                        )
                        for row in summary_warnings
                    ),
                )
            )
        summary_metrics = _mapping_or_empty(result_summary.get("key_metrics"))
        if summary_metrics:
            lines.extend(
                _html_table(
                    "Summary Key Metrics",
                    ("Metric", "Value"),
                    (
                        (metric, _format_metric(value))
                        for metric, value in summary_metrics.items()
                    ),
                )
            )
        failure_report = self.structured_gate_failure_report
        failure_gates = [
            row
            for row in failure_report.get("gates", [])
            if isinstance(row, Mapping)
        ]
        if failure_gates:
            lines.extend(
                _html_table(
                    "Structured Gate Failure Report",
                    ("Gate", "Status", "Severity", "Reason Count", "Metrics", "Top Reason"),
                    (
                        (
                            row.get("gate", ""),
                            row.get("status", ""),
                            row.get("severity", ""),
                            row.get("reason_count", ""),
                            ", ".join(
                                str(metric.get("metric"))
                                for metric in row.get("related_metrics", [])
                                if isinstance(metric, Mapping)
                                and metric.get("metric") is not None
                            ),
                            row.get("top_reason", ""),
                        )
                        for row in failure_gates
                    ),
                )
            )
        final_status_explanation = self.final_strategy_status_explanation
        lines.extend(
            _html_table(
                "Final Strategy Status Explanation",
                ("Field", "Value"),
                (
                    ("Final strategy status", self.final_strategy_status),
                    ("System validity status", self.system_validity_status),
                    ("Strategy pass", self.strategy_pass),
                    (
                        "Blocking rules",
                        _format_sequence(final_status_explanation.get("blocking_rules")),
                    ),
                    (
                        "Warning rules",
                        _format_sequence(final_status_explanation.get("warning_rules")),
                    ),
                    (
                        "Insufficient data rules",
                        _format_sequence(
                            final_status_explanation.get("insufficient_data_rules")
                        ),
                    ),
                    ("Reason", final_status_explanation.get("reason", "")),
                    ("Official message", final_status_explanation.get("official_message", "")),
                ),
            )
        )
        lines.extend(
            _html_table(
                "Rule Result Explanations",
                (
                    "Rule",
                    "Status",
                    "Passed",
                    "Affects Strategy",
                    "Affects System",
                    "Reason",
                    "Reason Code",
                    "Metric",
                    "Value",
                    "Threshold",
                    "Operator",
                ),
                (
                    (
                        row.get("rule", ""),
                        row.get("status", ""),
                        row.get("passed"),
                        row.get("affects_strategy"),
                        row.get("affects_system"),
                        row.get("reason", ""),
                        row.get("reason_code", ""),
                        row.get("metric", ""),
                        _format_metric(row.get("value")),
                        _format_metric(row.get("threshold")),
                        row.get("operator", ""),
                    )
                    for row in self.rule_result_explanations
                ),
            )
        )
        lines.extend(
            _html_table(
                "Strategy Failure Summary",
                (
                    "Gate",
                    "Status",
                    "Reason",
                    "Reason Code",
                    "Collapse Status",
                    "Collapse Reason",
                    "Metric",
                    "Value",
                    "Threshold",
                    "Operator",
                ),
                (
                    (
                        row.get("gate", ""),
                        row.get("status", ""),
                        row.get("reason", ""),
                        row.get("reason_code", ""),
                        row.get("collapse_status", ""),
                        row.get("collapse_reason", ""),
                        row.get("metric", ""),
                        _format_metric(row.get("value")),
                        _format_metric(row.get("threshold")),
                        row.get("operator", ""),
                    )
                    for row in self.strategy_failure_summary
                ),
            )
        )
        lines.extend(
            _html_table(
                "Gate Results",
                ("Gate", "Status", "Reason"),
                (
                    (name, result.get("status", ""), result.get("reason", ""))
                    for name, result in self.gate_results.items()
                ),
            )
        )

        horizon_metrics = _mapping_or_empty(self.metrics.get("horizon_metrics"))
        horizon_rows = []
        if horizon_metrics:
            for horizon in self.horizons:
                row = _mapping_or_empty(horizon_metrics.get(horizon))
                if not row:
                    continue
                horizon_rows.append(
                    (
                        horizon,
                        row.get("label", row.get("role", "")),
                        row.get("role", ""),
                        row.get("target_column", ""),
                        row.get("affects_pass_fail", False),
                        row.get("status", row.get("rank_ic_status", "")),
                        row.get("insufficient_data", False),
                        row.get("insufficient_data_status"),
                        row.get("insufficient_data_code"),
                        row.get("insufficient_data_reason"),
                        _format_metric(row.get("mean_rank_ic")),
                        _format_metric(row.get("positive_fold_ratio")),
                        _format_metric(row.get("positive_fold_ratio_threshold")),
                        row.get("positive_fold_ratio_passed"),
                        _format_metric(row.get("oos_rank_ic")),
                    )
                )
        lines.extend(
            _html_table(
                "Horizon Diagnostics",
                (
                    "Horizon",
                    "Label",
                    "Role",
                    "Target",
                    "Affects Pass/Fail",
                    "Status",
                    "Insufficient Data",
                    "Insufficient Data Status",
                    "Insufficient Data Code",
                    "Insufficient Data Reason",
                    "Mean Rank IC",
                    "Positive Fold Ratio",
                    "Positive Fold Ratio Threshold",
                    "Positive Fold Ratio Passed",
                    "OOS Rank IC",
                ),
                horizon_rows,
            )
        )

        covariance_risk = _mapping_or_empty(self.metrics.get("covariance_aware_risk"))
        if covariance_risk:
            parameters = _mapping_or_empty(covariance_risk.get("parameters"))
            realized = _mapping_or_empty(covariance_risk.get("realized_metrics"))
            lines.extend(
                _html_table(
                    "Covariance-Aware Risk",
                    ("Field", "Value"),
                    (
                        ("Applied", covariance_risk.get("applied")),
                        ("Configured enabled", covariance_risk.get("configured_enabled")),
                        ("Status", covariance_risk.get("status", "")),
                        ("Return column", parameters.get("return_column")),
                        ("Lookback periods", parameters.get("lookback_periods")),
                        ("Min periods", parameters.get("min_periods")),
                        ("Fallback", parameters.get("fallback")),
                        ("Max holdings", parameters.get("max_holdings")),
                        ("Max symbol weight", _format_metric(parameters.get("max_symbol_weight"))),
                        ("Max sector weight", _format_metric(parameters.get("max_sector_weight"))),
                        (
                            "Portfolio volatility limit",
                            _format_metric(parameters.get("portfolio_volatility_limit")),
                        ),
                        (
                            "Max position risk contribution",
                            _format_metric(
                                parameters.get("max_position_risk_contribution")
                            ),
                        ),
                        (
                            "Average covariance volatility",
                            _format_metric(
                                realized.get("average_portfolio_volatility_estimate")
                            ),
                        ),
                        (
                            "Max covariance volatility",
                            _format_metric(
                                realized.get("max_portfolio_volatility_estimate")
                            ),
                        ),
                        (
                            "Latest sizing validation",
                            realized.get("latest_position_sizing_validation_status", ""),
                        ),
                    ),
                )
            )

        if self.cost_adjusted_metric_comparison:
            lines.extend(
                _html_table(
                    "Cost-Adjusted Strategy Comparison",
                    (
                        "Name",
                        "Role",
                        "Return Basis",
                        "CAGR",
                        "Sharpe",
                        "Max Drawdown",
                        "Cost-Adjusted Cumulative Return",
                        "Avg Daily Turnover",
                        "Total Cost Return",
                        "Excess Return",
                        "Excess Return Status",
                    ),
                    (
                        (
                            row.get("name", ""),
                            row.get("role", ""),
                            row.get("return_basis", ""),
                            _format_metric(row.get("cagr")),
                            _format_metric(row.get("sharpe")),
                            _format_metric(row.get("max_drawdown")),
                            _format_metric(row.get("cost_adjusted_cumulative_return")),
                            _format_metric(row.get("average_daily_turnover")),
                            _format_metric(row.get("total_cost_return")),
                            _format_metric(_row_excess_return(row)),
                            row.get("excess_return_status", ""),
                        )
                        for row in self.cost_adjusted_metric_comparison
                    ),
                )
            )

        report_only_research_metrics = _report_only_research_metric_rows(self.evidence)
        if report_only_research_metrics:
            lines.extend(
                _html_table(
                    "Report-Only Research Metrics",
                    (
                        "Metric",
                        "Status",
                        "Value",
                        "Target",
                        "Sample Scope",
                        "Report Only",
                        "Decision Use",
                        "Reason",
                    ),
                    (
                        (
                            row.get("metric", ""),
                            row.get("status", ""),
                            _format_metric(row.get("value")),
                            row.get("target_column", ""),
                            row.get("sample_scope", ""),
                            row.get("report_only"),
                            row.get("decision_use", ""),
                            row.get("reason", ""),
                        )
                        for row in report_only_research_metrics
                    ),
                )
            )

        if self.full_model_metrics:
            row = self.full_model_metrics
            metrics = _mapping_or_empty(row.get("metrics"))
            validation = _mapping_or_empty(row.get("validation_metrics"))
            lines.extend(
                _html_table(
                    "Full Model Metrics",
                    ("Field", "Value"),
                    (
                        ("Entity", row.get("entity_id", "")),
                        ("Status", row.get("status", "")),
                        ("Rank IC", _format_metric(validation.get("mean_rank_ic"))),
                        (
                            "Positive Fold Ratio",
                            _format_metric(validation.get("positive_fold_ratio")),
                        ),
                        (
                            "Positive Fold Ratio Threshold",
                            _format_metric(validation.get("positive_fold_ratio_threshold")),
                        ),
                        (
                            "Positive Fold Ratio Passed",
                            validation.get("positive_fold_ratio_passed"),
                        ),
                        ("OOS Rank IC", _format_metric(validation.get("oos_rank_ic"))),
                        ("Sharpe", _format_metric(metrics.get("sharpe"))),
                        ("Max Drawdown", _format_metric(metrics.get("max_drawdown"))),
                        (
                            "Cost-Adjusted Cumulative Return",
                            _format_metric(
                                metrics.get("cost_adjusted_cumulative_return")
                            ),
                        ),
                        (
                            "Average Daily Turnover",
                            _format_metric(metrics.get("average_daily_turnover")),
                        ),
                    ),
                )
            )

        if self.baseline_metrics:
            lines.extend(
                _html_table(
                    "Baseline Metrics",
                    (
                        "Entity",
                        "Role",
                        "Status",
                        "Sharpe",
                        "Max Drawdown",
                        "Cost-Adjusted Cumulative Return",
                        "Excess Return",
                        "Turnover",
                    ),
                    (
                        (
                            row.get("entity_id", ""),
                            row.get("role", ""),
                            row.get("status", ""),
                            _format_metric(_mapping_or_empty(row.get("metrics")).get("sharpe")),
                            _format_metric(
                                _mapping_or_empty(row.get("metrics")).get("max_drawdown")
                            ),
                            _format_metric(
                                _mapping_or_empty(row.get("metrics")).get(
                                    "cost_adjusted_cumulative_return"
                                )
                            ),
                            _format_metric(
                                _mapping_or_empty(row.get("metrics")).get("excess_return")
                            ),
                            _format_metric(
                                _mapping_or_empty(row.get("metrics")).get(
                                    "turnover",
                                    _mapping_or_empty(row.get("metrics")).get(
                                        "average_daily_turnover"
                                    ),
                                )
                            ),
                        )
                        for row in self.baseline_metrics
                    ),
                )
            )

        if self.ablation_metrics:
            lines.extend(
                _html_table(
                    "Ablation Metrics",
                    (
                        "Scenario",
                        "Role",
                        "Kind",
                        "Status",
                        "Sharpe",
                        "Rank IC",
                        "Cost-Adjusted Cumulative Return",
                        "Turnover",
                    ),
                    (
                        (
                            row.get("entity_id", ""),
                            row.get("role", ""),
                            row.get("kind", ""),
                            row.get("status", ""),
                            _format_metric(_mapping_or_empty(row.get("metrics")).get("sharpe")),
                            _format_metric(
                                _mapping_or_empty(row.get("metrics")).get(
                                    "rank_ic",
                                    _mapping_or_empty(row.get("metrics")).get("mean_rank_ic"),
                                )
                            ),
                            _format_metric(
                                _mapping_or_empty(row.get("metrics")).get(
                                    "cost_adjusted_cumulative_return"
                                )
                            ),
                            _format_metric(
                                _mapping_or_empty(row.get("metrics")).get(
                                    "turnover",
                                    _mapping_or_empty(row.get("metrics")).get(
                                        "average_daily_turnover"
                                    ),
                                )
                            ),
                        )
                        for row in self.ablation_metrics
                    ),
                )
            )

        if self.structured_pass_fail_reasons:
            lines.extend(
                _html_table(
                    "Structured Pass/Fail Reasons",
                    (
                        "Category",
                        "Entity",
                        "Rule",
                        "Metric",
                        "Status",
                        "Passed",
                        "Reason Code",
                        "Reason",
                    ),
                    (
                        (
                            row.get("category", ""),
                            row.get("entity_id", ""),
                            row.get("rule", ""),
                            row.get("metric", ""),
                            row.get("status", ""),
                            row.get("passed"),
                            row.get("reason_code", ""),
                            row.get("reason", ""),
                        )
                        for row in self.structured_pass_fail_reasons
                    ),
                )
            )

        if self.side_by_side_metric_comparison:
            columns = _side_by_side_metric_entity_columns(self.side_by_side_metric_comparison)
            lines.extend(
                _html_table(
                    "Cost-Adjusted Side-by-Side Metrics",
                    ("Metric", *(_comparison_display_name(name) for name in columns)),
                    (
                        (
                            row.get("metric_label", row.get("metric", "")),
                            *(
                                _format_side_by_side_metric_value(
                                    str(row.get("metric", "")),
                                    row.get(name),
                                )
                                for name in columns
                            ),
                        )
                        for row in self.side_by_side_metric_comparison
                    ),
                )
            )

        if self.baseline_comparisons:
            lines.extend(
                _html_table(
                    "Baseline Comparisons",
                    (
                        "Baseline",
                        "Type",
                        "Return Basis",
                        "CAGR",
                        "Sharpe",
                        "Max Drawdown",
                        "Cost-Adjusted Cumulative Return",
                        "Avg Daily Turnover",
                        "Total Cost Return",
                        "Excess Return",
                        "Excess Return Status",
                    ),
                    (
                        (
                            row.get("name", ""),
                            row.get("baseline_type", ""),
                            row.get("return_basis", ""),
                            _format_metric(row.get("cagr")),
                            _format_metric(row.get("sharpe")),
                            _format_metric(row.get("max_drawdown")),
                            _format_metric(row.get("cost_adjusted_cumulative_return")),
                            _format_metric(row.get("average_daily_turnover")),
                            _format_metric(row.get("total_cost_return")),
                            _format_metric(_row_excess_return(row)),
                            row.get("excess_return_status", ""),
                        )
                        for row in self.baseline_comparison_entries
                    ),
                )
            )

        if self.baseline_comparison_inputs:
            lines.extend(
                _html_table(
                    "Baseline Comparison Inputs",
                    (
                        "Baseline",
                        "Type",
                        "Return Basis",
                        "Data Source",
                        "Return Column",
                        "Horizon",
                        "Required",
                        "Benchmark",
                        "Universe",
                    ),
                    (
                        (
                            row.get("name", ""),
                            row.get("baseline_type", ""),
                            row.get("return_basis", ""),
                            row.get("data_source", ""),
                            row.get("return_column", ""),
                            row.get("return_horizon", ""),
                            row.get("required_for_stage1", ""),
                            row.get("benchmark_ticker") or "",
                            _format_sequence(row.get("universe_tickers")),
                        )
                        for row in self.baseline_comparison_inputs
                    ),
                )
            )

        if self.no_model_proxy_ablation:
            ablation = self.no_model_proxy_ablation
            controls = _mapping_or_empty(ablation.get("pipeline_controls"))
            performance = _mapping_or_empty(ablation.get("performance_metrics"))
            validation = _mapping_or_empty(ablation.get("validation_metrics"))
            signal_metrics = _mapping_or_empty(
                ablation.get("deterministic_signal_evaluation_metrics")
            )
            action_counts = _mapping_or_empty(signal_metrics.get("action_counts"))
            lines.extend(
                _html_table(
                    "No-Model-Proxy Ablation",
                    ("Field", "Value"),
                    (
                        ("Available", ablation.get("available", False)),
                        ("Model proxy enabled", controls.get("model_proxy", "")),
                        (
                            "Validation status",
                            validation.get("validation_status", ablation.get("status", "")),
                        ),
                        ("Fold count", validation.get("validation_fold_count", "")),
                        ("OOS fold count", validation.get("validation_oos_fold_count", "")),
                        ("Sharpe", _format_metric(performance.get("sharpe"))),
                        ("Excess return", _format_metric(performance.get("excess_return"))),
                        (
                            "Cost-adjusted cumulative return",
                            _format_metric(
                                signal_metrics.get("cost_adjusted_cumulative_return")
                            ),
                        ),
                        (
                            "Average daily turnover",
                            _format_metric(signal_metrics.get("average_daily_turnover")),
                        ),
                        (
                            "Buy / Sell / Hold",
                            (
                                f"{action_counts.get('BUY', 0)} / "
                                f"{action_counts.get('SELL', 0)} / "
                                f"{action_counts.get('HOLD', 0)}"
                            ),
                        ),
                    ),
                )
            )

        if self.model_comparison_results:
            lines.extend(
                _html_table(
                    "Model Comparison Results",
                    (
                        "Window",
                        "Metric",
                        "Candidate",
                        "Candidate Value",
                        "Comparison Target",
                        "Target Role",
                        "Target Value",
                        "Absolute Delta",
                        "Relative Delta",
                        "Pass/Fail",
                    ),
                    (
                        (
                            row.get("window_id", ""),
                            row.get("metric", ""),
                            row.get("candidate", ""),
                            _format_metric(row.get("candidate_value")),
                            row.get("baseline", ""),
                            row.get("baseline_role", ""),
                            _format_metric(row.get("baseline_value")),
                            _format_metric(row.get("absolute_delta", row.get("delta"))),
                            _format_metric(row.get("relative_delta")),
                            row.get("pass_fail", ""),
                        )
                        for row in self.model_comparison_results
                    ),
                )
            )

        if self.comparison_input_schema:
            schema = self.comparison_input_schema
            full_model = _mapping_or_empty(schema.get("full_model"))
            lines.extend(
                _html_table(
                    "Stage 1 Comparison Input Schema",
                    ("Field", "Value"),
                    (
                        ("Schema version", schema.get("schema_version", "")),
                        ("Comparison ID", schema.get("comparison_id", "")),
                        ("Full model", full_model.get("entity_id", "")),
                        ("Baselines", _sequence_length(schema.get("baselines"))),
                        ("Ablations", _sequence_length(schema.get("ablations"))),
                        ("Metrics", _sequence_length(schema.get("metrics"))),
                        ("Validation windows", _sequence_length(schema.get("validation_windows"))),
                        ("Signal engine", full_model.get("signal_engine", "")),
                        (
                            "Model predictions are order signals",
                            full_model.get("model_predictions_are_order_signals"),
                        ),
                        (
                            "LLM makes trading decisions",
                            full_model.get("llm_makes_trading_decisions"),
                        ),
                    ),
                )
            )

        if self.comparison_result_schema:
            schema = self.comparison_result_schema
            full_model_result = _mapping_or_empty(schema.get("full_model_result"))
            lines.extend(
                _html_table(
                    "Stage 1 Comparison Result Schema",
                    ("Field", "Value"),
                    (
                        ("Schema version", schema.get("schema_version", "")),
                        ("Comparison ID", schema.get("comparison_id", "")),
                        ("Full model result", full_model_result.get("entity_id", "")),
                        ("Baseline results", _sequence_length(schema.get("baseline_results"))),
                        ("Ablation results", _sequence_length(schema.get("ablation_results"))),
                        ("Metric results", _sequence_length(schema.get("metric_results"))),
                        (
                            "Validation windows",
                            _sequence_length(schema.get("validation_windows")),
                        ),
                    ),
                )
            )

        if self.warnings:
            lines.extend(_html_list("Warnings", self.warnings))
        if self.structured_warnings:
            lines.extend(
                _html_table(
                    "Structured Warnings",
                    (
                        "Code",
                        "Gate",
                        "Combined Gate",
                        "Severity",
                        "Metric",
                        "Realized Turnover",
                        "Budget",
                        "Value",
                        "Threshold",
                        "Operator",
                        "Message",
                        "Reason",
                    ),
                    (
                        (
                            row.get("code", ""),
                            row.get("gate", ""),
                            row.get("combined_gate", ""),
                            row.get("severity", ""),
                            row.get("metric", ""),
                            _format_metric(row.get("realized_turnover")),
                            _format_metric(row.get("budget")),
                            _format_metric(row.get("value")),
                            _format_metric(row.get("threshold")),
                            row.get("operator", ""),
                            row.get("message", ""),
                            row.get("reason", ""),
                        )
                        for row in self.structured_warnings
                    ),
                )
            )
        if self.hard_fail_reasons:
            lines.extend(_html_list("Hard Fail Reasons", self.hard_fail_reasons))

        if self.benchmark_results:
            lines.extend(
                _html_table(
                    "Benchmark Results",
                    (
                        "Baseline",
                        "CAGR",
                        "Sharpe",
                        "Max Drawdown",
                        "Excess Return",
                        "Excess Return Status",
                        "Avg Daily Turnover",
                    ),
                    (
                        (
                            row.get("name", ""),
                            _format_metric(row.get("cagr")),
                            _format_metric(row.get("sharpe")),
                            _format_metric(row.get("max_drawdown")),
                            _format_metric(row.get("excess_return")),
                            row.get("excess_return_status", ""),
                            _format_metric(row.get("average_daily_turnover")),
                        )
                        for row in self.benchmark_results
                    ),
                )
            )

        stage1_ablation_rows = _stage1_ablation_comparison_results(self.ablation_results)
        if stage1_ablation_rows:
            lines.extend(
                _html_table(
                    "Stage 1 Ablation Scenario Comparison",
                    (
                        "Scenario",
                        "Kind",
                        "Return Basis",
                        "CAGR",
                        "Sharpe",
                        "Max Drawdown",
                        "Excess Return",
                        "Cost-Adjusted Cumulative Return",
                        "Avg Daily Turnover",
                        "Total Cost Return",
                        "Effective Cost bps",
                        "Effective Slippage bps",
                        "Validation Status",
                        "Fold Count",
                        "OOS Fold Count",
                        "Feature Families",
                        "Model Proxy",
                        "Cost",
                        "Slippage",
                        "Turnover",
                    ),
                    (
                        (
                            row.get("scenario", ""),
                            row.get("kind", ""),
                            _ablation_signal_metrics(row).get(
                                "return_basis",
                                row.get("signal_return_basis", ""),
                            ),
                            _format_metric(row.get("cagr")),
                            _format_metric(row.get("sharpe")),
                            _format_metric(row.get("max_drawdown")),
                            _format_metric(row.get("excess_return")),
                            _format_metric(
                                _ablation_signal_metric(
                                    row,
                                    _ablation_signal_metrics(row),
                                    "cost_adjusted_cumulative_return",
                                )
                            ),
                            _format_metric(
                                _ablation_signal_metric(
                                    row,
                                    _ablation_signal_metrics(row),
                                    "average_daily_turnover",
                                )
                            ),
                            _format_metric(
                                _ablation_signal_metric(
                                    row,
                                    _ablation_signal_metrics(row),
                                    "total_cost_return",
                                )
                            ),
                            _format_metric(row.get("effective_cost_bps")),
                            _format_metric(row.get("effective_slippage_bps")),
                            row.get("validation_status", ""),
                            _format_count_metric(row.get("validation_fold_count")),
                            _format_count_metric(row.get("validation_oos_fold_count")),
                            _format_feature_families(row),
                            _mapping_or_empty(row.get("pipeline_controls")).get(
                                "model_proxy",
                                "",
                            ),
                            _mapping_or_empty(row.get("pipeline_controls")).get("cost", ""),
                            _mapping_or_empty(row.get("pipeline_controls")).get(
                                "slippage",
                                "",
                            ),
                            _mapping_or_empty(row.get("pipeline_controls")).get(
                                "turnover",
                                "",
                            ),
                        )
                        for row in stage1_ablation_rows
                    ),
                )
            )

        pipeline_controls = _pipeline_control_results(self.ablation_results)
        if pipeline_controls:
            lines.extend(
                _html_table(
                    "Pipeline Controls",
                    (
                        "Scenario",
                        "Model Proxy",
                        "Cost",
                        "Slippage",
                        "Turnover",
                        "Effective Cost bps",
                        "Effective Slippage bps",
                        "Signal Cost-Adjusted Return",
                        "Signal Turnover",
                        "Buy",
                        "Sell",
                        "Hold",
                        "Risk Stops",
                    ),
                    (
                        (
                            row.get("scenario", ""),
                            _mapping_or_empty(row.get("pipeline_controls")).get(
                                "model_proxy",
                                "",
                            ),
                            _mapping_or_empty(row.get("pipeline_controls")).get("cost", ""),
                            _mapping_or_empty(row.get("pipeline_controls")).get(
                                "slippage",
                                "",
                            ),
                            _mapping_or_empty(row.get("pipeline_controls")).get(
                                "turnover",
                                "",
                            ),
                            row.get("effective_cost_bps", ""),
                            row.get("effective_slippage_bps", ""),
                            _format_metric(row.get("signal_cost_adjusted_cumulative_return")),
                            _format_metric(row.get("signal_average_daily_turnover")),
                            row.get("signal_buy_count", ""),
                            row.get("signal_sell_count", ""),
                            row.get("signal_hold_count", ""),
                            row.get("signal_risk_stop_observation_count", ""),
                        )
                        for row in pipeline_controls
                    ),
                )
            )

        cost_ablation_controls = _cost_ablation_results(self.ablation_results)
        if cost_ablation_controls:
            lines.extend(
                _html_table(
                    "Cost Ablations",
                    (
                        "Scenario",
                        "Cost",
                        "Slippage",
                        "Turnover",
                        "Effective Cost bps",
                        "Effective Slippage bps",
                        "Signal Cost-Adjusted Return",
                        "Signal Turnover",
                        "Total Cost Return",
                    ),
                    (
                        (
                            row.get("scenario", ""),
                            _mapping_or_empty(row.get("pipeline_controls")).get("cost", ""),
                            _mapping_or_empty(row.get("pipeline_controls")).get(
                                "slippage",
                                "",
                            ),
                            _mapping_or_empty(row.get("pipeline_controls")).get(
                                "turnover",
                                "",
                            ),
                            row.get("effective_cost_bps", ""),
                            row.get("effective_slippage_bps", ""),
                            _format_metric(row.get("signal_cost_adjusted_cumulative_return")),
                            _format_metric(row.get("signal_average_daily_turnover")),
                            _format_metric(row.get("signal_total_cost_return")),
                        )
                        for row in cost_ablation_controls
                    ),
                )
            )

        lines.extend(_html_json_section("Metrics", self.metrics))
        lines.extend(_html_json_section("Evidence", self.evidence))
        lines.extend(["</body>", "</html>"])
        return "\n".join(lines)


def build_validity_gate_report(
    predictions: pd.DataFrame,
    validation_summary: pd.DataFrame,
    equity_curve: pd.DataFrame,
    strategy_metrics: object,
    ablation_summary: list[dict[str, Any]] | None = None,
    *,
    config: object | None = None,
    walk_forward_config: object | None = None,
    thresholds: ValidationGateThresholds | None = None,
    strategy_candidate_policy: StrategyCandidateGatePolicy | Mapping[str, Any] | None = None,
    benchmark_return_series: pd.DataFrame | None = None,
    equal_weight_baseline_return_series: pd.DataFrame | None = None,
    baseline_comparison_inputs: Iterable[BaselineComparisonInput] | None = None,
) -> ValidationGateReport:
    thresholds = thresholds or ValidationGateThresholds()
    strategy_candidate_policy = _strategy_candidate_gate_policy(
        strategy_candidate_policy
        or getattr(config, "strategy_candidate_policy", None)
    )
    ablation_summary = ablation_summary or []
    configured_target_column = str(getattr(config, "prediction_target_column", "forward_return_20"))
    configured_target_horizon = (
        _horizon_from_target(configured_target_column)
        or thresholds.required_validation_horizon
    )
    required_horizon = max(
        int(getattr(config, "required_validation_horizon", thresholds.required_validation_horizon)),
        int(thresholds.required_validation_horizon),
    )
    target_column = _gate_decision_target_column(
        predictions,
        configured_target_column,
        required_horizon,
    )
    target_horizon = _horizon_from_target(target_column) or required_horizon
    reporting_target_column = (
        target_column if target_column in predictions.columns else configured_target_column
    )
    benchmark_ticker = str(getattr(config, "benchmark_ticker", "SPY"))

    gap_periods = int(getattr(walk_forward_config, "gap_periods", getattr(config, "gap_periods", 0)))
    embargo_periods = int(
        getattr(walk_forward_config, "embargo_periods", getattr(config, "embargo_periods", 0))
    )
    requested_gap_periods = int(
        getattr(walk_forward_config, "requested_gap_periods", gap_periods)
    )
    requested_embargo_periods = int(
        getattr(walk_forward_config, "requested_embargo_periods", embargo_periods)
    )
    effective_gap_periods = int(getattr(walk_forward_config, "effective_gap_periods", gap_periods))
    effective_embargo_periods = int(
        getattr(walk_forward_config, "effective_embargo_periods", embargo_periods)
    )
    gap_periods = effective_gap_periods
    embargo_periods = effective_embargo_periods
    min_train_observations = int(getattr(walk_forward_config, "min_train_observations", 0))

    gate_results: dict[str, dict[str, Any]] = {}
    hard_fail_reasons: list[str] = []
    warnings: list[str] = []
    structured_warnings: list[dict[str, Any]] = []
    insufficient_data = False

    baseline_input_objects = _stage1_baseline_comparison_inputs(
        baseline_comparison_inputs,
        equity_curve,
        predictions,
        config,
        benchmark_ticker=benchmark_ticker,
        return_column=reporting_target_column,
    )
    baseline_input_gate = evaluate_stage1_baseline_comparison_inputs(baseline_input_objects)
    gate_results["baseline_inputs"] = baseline_input_gate
    if baseline_input_gate["status"] == "hard_fail":
        hard_fail_reasons.extend(str(reason) for reason in baseline_input_gate["reasons"])
    baseline_input_rows = [
        baseline_input.to_dict()
        for baseline_input in baseline_input_objects
    ]
    requested_gap_embargo_warnings: list[str] = []
    if requested_gap_periods < gap_periods:
        requested_gap_embargo_warnings.append(
            f"requested_gap_periods={requested_gap_periods} raised to effective_gap_periods={gap_periods}"
        )
    if requested_embargo_periods < embargo_periods:
        requested_gap_embargo_warnings.append(
            "requested_embargo_periods="
            f"{requested_embargo_periods} raised to effective_embargo_periods={embargo_periods}"
        )
    if requested_gap_embargo_warnings:
        gate_results["requested_gap_embargo_adjustment"] = {
            "status": "warning",
            "reason": "; ".join(requested_gap_embargo_warnings),
            "affects_system": False,
            "affects_strategy": False,
            "affects_pass_fail": False,
        }

    horizon_metrics = _horizon_validation_metrics(
        predictions,
        thresholds,
        target_column,
        required_horizon,
        horizons=DEFAULT_HORIZONS,
    )

    purge_embargo_application = _purge_embargo_application_evidence(
        validation_summary,
        gap_periods=gap_periods,
        embargo_periods=embargo_periods,
        target_horizon=target_horizon,
    )
    leakage = _evaluate_leakage(
        validation_summary,
        gap_periods,
        embargo_periods,
        target_horizon,
        purge_embargo_application=purge_embargo_application,
    )
    gate_results["leakage"] = leakage
    if leakage["status"] == "hard_fail":
        hard_fail_reasons.extend(leakage["reasons"])

    walk_forward = _evaluate_walk_forward(
        validation_summary,
        thresholds,
        min_train_observations,
        target_horizon=target_horizon,
    )
    gate_results["walk_forward_oos"] = walk_forward
    if walk_forward["status"] == "hard_fail":
        hard_fail_reasons.extend(walk_forward["reasons"])
    elif walk_forward["status"] in INSUFFICIENT_DATA_GATE_STATUSES:
        insufficient_data = True

    rank_ic = _rank_ic_metrics_for_horizon(predictions, target_column, target_horizon)
    rank_gate = _evaluate_rank_ic(rank_ic, thresholds)
    gate_results["rank_ic"] = rank_gate
    if rank_gate["status"] == "fail":
        warnings.append(rank_gate["reason"])
    elif rank_gate["status"] in INSUFFICIENT_DATA_GATE_STATUSES:
        insufficient_data = True

    for diagnostic_horizon in DIAGNOSTIC_ONLY_HORIZONS:
        diagnostic_rank_gate = _diagnostic_horizon_gate_result(
            horizon_metrics.get(diagnostic_horizon)
        )
        if diagnostic_rank_gate:
            gate_results[f"rank_ic_{diagnostic_horizon}_diagnostic"] = diagnostic_rank_gate

    baseline_results = _benchmark_results(
        predictions,
        equity_curve,
        strategy_metrics,
        benchmark_ticker,
        reporting_target_column,
        config=config,
        benchmark_return_series=benchmark_return_series,
        equal_weight_baseline_return_series=equal_weight_baseline_return_series,
    )
    top_decile_20d = calculate_top_decile_20d_excess_return(
        predictions,
        universe=getattr(config, "tickers", None),
        benchmark_ticker=benchmark_ticker,
    )
    baseline_sample_alignment = _evaluate_baseline_sample_alignment(baseline_results)
    gate_results["baseline_sample_alignment"] = baseline_sample_alignment
    if baseline_sample_alignment["status"] == "hard_fail":
        hard_fail_reasons.append(str(baseline_sample_alignment["reason"]))
    pipeline_control_results = _pipeline_control_results(ablation_summary)
    cost_ablation_results = _cost_ablation_results(ablation_summary)
    cost_adjustment = _cost_adjustment_metrics(equity_curve)
    cost_adjusted_metric_comparison = _cost_adjusted_metric_comparison(
        equity_curve,
        strategy_metrics,
        baseline_results,
        cost_adjustment,
        config=config,
        default_return_column=reporting_target_column,
    )
    side_by_side_metric_comparison = _side_by_side_metric_comparison(
        cost_adjusted_metric_comparison
    )
    baseline_comparisons = _baseline_comparisons(
        baseline_results,
        cost_adjusted_metric_comparison,
    )
    no_model_proxy_ablation = _no_model_proxy_ablation_result(ablation_summary)
    model_comparison_results = _model_comparison_results(
        ablation_summary,
        config,
        baseline_comparisons=baseline_comparisons,
        cost_adjusted_metric_comparison=cost_adjusted_metric_comparison,
    )
    model_value_gate = _evaluate_model_value_warning(ablation_summary, config)
    gate_results["model_value"] = model_value_gate
    if model_value_gate["status"] == "warning":
        warnings.append(model_value_gate["reason"])
        structured_warning = _structured_warning_from_gate(model_value_gate)
        if structured_warning:
            structured_warnings.append(structured_warning)
    cost_gate = _evaluate_cost_adjusted(
        baseline_results,
        cost_adjustment=cost_adjustment,
        collapse_threshold=_configured_cost_adjusted_collapse_threshold(config, thresholds),
    )
    gate_results["cost_adjusted_performance"] = cost_gate
    if cost_gate["status"] == "warning":
        warnings.append(cost_gate["reason"])

    benchmark_gate = _evaluate_benchmark(baseline_results, thresholds)
    gate_results["benchmark_comparison"] = benchmark_gate
    if benchmark_gate["status"] == "warning":
        warnings.append(benchmark_gate["reason"])

    daily_turnover_gate = evaluate_average_daily_turnover_gate(
        _metric(strategy_metrics, "turnover"),
        baseline_results=baseline_results,
        ablation_summary=ablation_summary,
        thresholds=thresholds,
    )
    monthly_budget_override = _finite_or_none(
        getattr(config, "monthly_turnover_budget", getattr(config, "max_monthly_turnover", None))
    )
    monthly_turnover_gate = evaluate_monthly_turnover_budget_gate(
        equity_curve,
        thresholds=thresholds,
        monthly_turnover_budget=monthly_budget_override,
    )

    for turnover_component_gate in (daily_turnover_gate, monthly_turnover_gate):
        structured_warning = _structured_warning_from_gate(turnover_component_gate)
        if not structured_warning:
            continue
        structured_warnings.append(structured_warning)
        warnings.append(str(structured_warning["message"]))

    turnover_gate = _combine_turnover_validity_gates(daily_turnover_gate, monthly_turnover_gate)
    gate_results["turnover"] = turnover_gate
    gate_results["monthly_turnover_budget"] = {
        **monthly_turnover_gate,
        "affects_strategy": False,
        "combined_gate": "turnover",
    }
    if turnover_gate["status"] == "warning":
        warnings.append(turnover_gate["reason"])

    drawdown_gate = _evaluate_drawdown(baseline_results, thresholds)
    gate_results["drawdown"] = drawdown_gate
    if drawdown_gate["status"] == "warning":
        warnings.append(drawdown_gate["reason"])

    strategy_candidate_policy_evaluation = evaluate_strategy_candidate_gate_policy(
        _strategy_candidate_policy_input_metrics(
            rank_ic=rank_ic,
            baseline_comparisons=baseline_comparisons,
            strategy_metrics=strategy_metrics,
            model_value_gate=model_value_gate,
        ),
        policy=strategy_candidate_policy,
    )
    gate_results["strategy_candidate_policy"] = {
        **strategy_candidate_policy_evaluation,
        "affects_strategy": False,
        "affects_system": False,
        "affects_insufficient_data": False,
        "reason": (
            "strategy candidate policy evaluation recorded for canonical Stage 1 "
            "configuration; deterministic composite gate remains the active "
            "strategy decision path"
        ),
    }

    strategy_validity_gate = _evaluate_deterministic_strategy_validity(gate_results)
    gate_results["deterministic_strategy_validity"] = strategy_validity_gate

    ablation_gate = _evaluate_ablation(ablation_summary)
    gate_results["ablation"] = ablation_gate
    if ablation_gate["status"] == "warning":
        warnings.append(ablation_gate["reason"])

    system_validity_criteria = build_system_validity_gate_criteria(thresholds)
    validation_gate_decision_criteria = build_validation_gate_decision_criteria(
        thresholds,
        strategy_candidate_policy,
    )
    gate_results["system_validity_artifact_contract"] = {
        "status": "pass",
        "reason": (
            "system validity criteria, per-item PASS/WARN/FAIL decision criteria, "
            "input schema, output schema, thresholds, and evidence are embedded "
            "in the canonical report payload"
        ),
        "affects_system": True,
        "affects_strategy": False,
        "affects_insufficient_data": False,
        "policy_id": system_validity_criteria["policy_id"],
        "schema_version": system_validity_criteria["schema_version"],
    }
    hard_fail_reasons = _report_hard_fail_reasons(gate_results)
    warnings = _report_warning_messages(gate_results)
    system_status = _system_validity_status(hard_fail_reasons, insufficient_data)
    strategy_status = _strategy_status(system_status, gate_results, insufficient_data)
    official_message = _official_message(system_status, strategy_status)
    deterministic_gate_aggregation = aggregate_deterministic_gate_results(
        gate_results,
        system_validity_status=system_status,
        strategy_candidate_status=strategy_status,
    )
    gate_results["deterministic_gate_aggregation"] = {
        **deterministic_gate_aggregation,
        "affects_system": False,
        "affects_strategy": False,
        "affects_pass_fail": False,
    }
    system_validity_gate_input_schema = build_system_validity_gate_input_schema()
    system_validity_gate_output_schema = build_system_validity_gate_output_schema()
    system_validity_gate_report_schema = build_system_validity_gate_report_schema()
    model_comparison_config = _model_comparison_config_payload(config)
    comparison_input_schema = build_stage1_comparison_input_schema(
        config,
        baseline_inputs=baseline_input_objects,
    ).to_dict()
    comparison_result_schema = build_stage1_comparison_result_schema(
        config=config,
        model_comparison_results=model_comparison_results,
        baseline_comparisons=baseline_comparisons,
        ablation_results=ablation_summary,
        validation_summary=validation_summary,
    ).to_dict()
    full_model_metrics = _full_model_metric_contract(
        model_comparison_config,
        rank_ic,
        cost_adjusted_metric_comparison,
        ablation_summary,
        strategy_status=strategy_status,
        positive_fold_ratio_threshold=thresholds.min_positive_fold_ratio,
    )
    baseline_metrics = _baseline_metric_contract(
        comparison_result_schema,
        baseline_comparisons,
        ablation_summary,
    )
    ablation_metrics = _ablation_metric_contract(
        comparison_result_schema,
        ablation_summary,
    )
    structured_pass_fail_reasons = _structured_pass_fail_reasons(
        gate_results,
        model_comparison_results,
    )
    structured_warnings = _structured_warning_reasons(gate_results, structured_warnings)
    warnings = _deduplicate_preserving_order(
        [
            *warnings,
            *(
                str(warning.get("message") or warning.get("reason"))
                for warning in structured_warnings
                if str(warning.get("message") or warning.get("reason") or "").strip()
            ),
        ]
    )
    covariance_aware_risk = _covariance_aware_risk_report_payload(
        config,
        equity_curve,
        strategy_metrics,
    )
    label_columns = sorted(
        [
            column
            for column in predictions.columns
            if _horizon_from_target(str(column)) is not None
        ],
        key=lambda column: (_horizon_from_target(str(column)) or 0, str(column)),
    )
    label_coverage = {
        column: float(pd.to_numeric(predictions[column], errors="coerce").notna().mean())
        for column in label_columns
    }
    metrics = {
        "fold_count": _count_validation_folds(validation_summary),
        "oos_fold_count": _count_oos_folds(validation_summary),
        "insufficient_data": insufficient_data,
        "insufficient_data_reasons": _insufficient_data_reasons(gate_results),
        "target_column": target_column,
        "target_horizon": target_horizon,
        "target_horizon_periods": target_horizon,
        "realized_return_column": reporting_target_column,
        "label_columns": label_columns,
        "label_coverage": label_coverage,
        "reporting_target_column": reporting_target_column,
        "reporting_target_horizon": (
            _horizon_from_target(reporting_target_column) or target_horizon
        ),
        "configured_target_column": configured_target_column,
        "configured_target_horizon": configured_target_horizon,
        "required_validation_horizon": required_horizon,
        "horizon_metrics": horizon_metrics,
        "diagnostic_horizon_metrics": {
            horizon: metrics
            for horizon, metrics in horizon_metrics.items()
            if metrics.get("label") == "diagnostic"
        },
        "requested_gap_periods": requested_gap_periods,
        "requested_embargo_periods": requested_embargo_periods,
        "effective_gap_periods": gap_periods,
        "effective_embargo_periods": embargo_periods,
        "gap_periods": gap_periods,
        "embargo_periods": embargo_periods,
        "purge_embargo_application": purge_embargo_application,
        "strategy_cagr": _metric(strategy_metrics, "cagr"),
        "strategy_sharpe": _metric(strategy_metrics, "sharpe"),
        "strategy_max_drawdown": _metric(strategy_metrics, "max_drawdown"),
        "strategy_turnover": _metric(strategy_metrics, "turnover"),
        "strategy_gross_cumulative_return": cost_adjustment["gross_cumulative_return"],
        "strategy_cost_adjusted_cumulative_return": cost_adjustment[
            "cost_adjusted_cumulative_return"
        ],
        "strategy_transaction_cost_return": cost_adjustment["transaction_cost_return"],
        "strategy_slippage_cost_return": cost_adjustment["slippage_cost_return"],
        "strategy_total_cost_return": cost_adjustment["total_cost_return"],
        "cost_adjusted_collapse_threshold": cost_gate.get("collapse_threshold"),
        "strategy_max_monthly_turnover": monthly_turnover_gate.get("value"),
        "monthly_turnover_budget": monthly_turnover_gate.get("threshold"),
        "strategy_excess_return_vs_spy": baseline_results[0]["excess_return"],
        **_baseline_metric_snapshot("spy_baseline", baseline_results[0]),
        **_baseline_metric_snapshot("market_benchmark_baseline", baseline_results[0]),
        "strategy_excess_return_vs_equal_weight": baseline_results[1]["excess_return"],
        "equal_weight_baseline_cagr": baseline_results[1]["cagr"],
        "equal_weight_baseline_sharpe": baseline_results[1]["sharpe"],
        "equal_weight_baseline_max_drawdown": baseline_results[1]["max_drawdown"],
        "equal_weight_baseline_average_daily_turnover": baseline_results[1].get("average_daily_turnover"),
        "baseline_comparisons": baseline_comparisons,
        "baseline_comparison_inputs": baseline_input_rows,
        "baseline_sample_alignment": baseline_sample_alignment,
        "no_model_proxy_ablation": no_model_proxy_ablation,
        "mean_rank_ic": rank_ic.get("mean_rank_ic"),
        "positive_fold_ratio": rank_ic.get("positive_fold_ratio"),
        "positive_fold_ratio_threshold": thresholds.min_positive_fold_ratio,
        "positive_fold_ratio_passed": _metric_threshold_passed(
            rank_ic.get("positive_fold_ratio"),
            thresholds.min_positive_fold_ratio,
        ),
        "oos_rank_ic": rank_ic.get("oos_rank_ic"),
        "top_decile_20d_excess_return": top_decile_20d["top_decile_20d_excess_return"],
        "top_decile_20d_excess_return_status": top_decile_20d["status"],
        "top_decile_20d_excess_return_scope": top_decile_20d["sample_scope"],
        "top_decile_20d_excess_return_decision_use": top_decile_20d["decision_use"],
        "cost_adjusted_metric_comparison": cost_adjusted_metric_comparison,
        "side_by_side_metric_comparison": side_by_side_metric_comparison,
        "cost_adjusted_metric_comparison_side_by_side": side_by_side_metric_comparison,
        "model_comparison_config": model_comparison_config,
        "model_comparison_results": model_comparison_results,
        "comparison_input_schema": comparison_input_schema,
        "comparison_result_schema": comparison_result_schema,
        "full_model_metrics": full_model_metrics,
        "baseline_metrics": baseline_metrics,
        "ablation_metrics": ablation_metrics,
        "structured_pass_fail_reasons": structured_pass_fail_reasons,
        "model_value": model_value_gate,
        "deterministic_strategy_validity": strategy_validity_gate,
        "deterministic_gate_aggregation": deterministic_gate_aggregation,
        "final_gate_decision": deterministic_gate_aggregation["final_decision"],
        "final_gate_status": deterministic_gate_aggregation["final_status"],
        "covariance_aware_risk": covariance_aware_risk,
        "system_validity_gate_criteria": system_validity_criteria,
        "validation_gate_decision_criteria": validation_gate_decision_criteria,
        "system_validity_gate_input_schema": system_validity_gate_input_schema,
        "system_validity_gate_output_schema": system_validity_gate_output_schema,
        "system_validity_gate_report_schema": system_validity_gate_report_schema,
        "strategy_candidate_policy": strategy_candidate_policy.to_dict(),
        "strategy_candidate_policy_evaluation": strategy_candidate_policy_evaluation,
    }
    evidence = {
        "system_validity_gate_input_schema": system_validity_gate_input_schema,
        "system_validity_gate_output_schema": system_validity_gate_output_schema,
        "system_validity_gate_report_schema": system_validity_gate_report_schema,
        "thresholds": asdict(thresholds),
        "strategy_candidate_policy": strategy_candidate_policy.to_dict(),
        "strategy_candidate_policy_evaluation": strategy_candidate_policy_evaluation,
        "leakage": leakage,
        "purge_embargo_application": purge_embargo_application,
        "walk_forward_oos": walk_forward,
        "rank_ic": rank_ic,
        "top_decile_20d_excess_return": top_decile_20d,
        "insufficient_data": insufficient_data,
        "insufficient_data_reasons": _insufficient_data_reasons(gate_results),
        "horizon_metrics": horizon_metrics,
        "diagnostic_horizon_metrics": {
            horizon: metrics
            for horizon, metrics in horizon_metrics.items()
            if metrics.get("label") == "diagnostic"
        },
        "benchmark_ticker": benchmark_ticker,
        "baseline_results": baseline_results,
        "baseline_comparisons": baseline_comparisons,
        "baseline_comparison_inputs": baseline_input_rows,
        "baseline_input_validation": baseline_input_gate,
        "baseline_sample_alignment": baseline_sample_alignment,
        "cost_adjustment": cost_adjustment,
        "cost_adjusted_collapse_check": cost_gate.get("collapse_check", {}),
        "ablation_required_scenarios": list(STAGE1_REQUIRED_ABLATION_SCENARIOS),
        "pipeline_control_required_scenarios": [
            "no_model_proxy",
        ],
        "pipeline_control_results": pipeline_control_results,
        "pipeline_control_toggles": {
            str(row.get("scenario")): row.get("pipeline_controls", {})
            for row in pipeline_control_results
        },
        "cost_required_scenarios": [
            NO_COST_ABLATION_SCENARIO,
        ],
        "cost_ablation_results": cost_ablation_results,
        "cost_ablation_toggles": {
            str(row.get("scenario")): row.get("pipeline_controls", {})
            for row in cost_ablation_results
        },
        "no_model_proxy_ablation": no_model_proxy_ablation,
        "side_by_side_metric_comparison": side_by_side_metric_comparison,
        "model_comparison_config": model_comparison_config,
        "model_comparison_results": model_comparison_results,
        "comparison_input_schema": comparison_input_schema,
        "comparison_result_schema": comparison_result_schema,
        "full_model_metrics": full_model_metrics,
        "baseline_metrics": baseline_metrics,
        "ablation_metrics": ablation_metrics,
        "structured_pass_fail_reasons": structured_pass_fail_reasons,
        "model_value": model_value_gate,
        "deterministic_strategy_validity": strategy_validity_gate,
        "deterministic_gate_aggregation": deterministic_gate_aggregation,
        "final_gate_decision": deterministic_gate_aggregation["final_decision"],
        "final_gate_status": deterministic_gate_aggregation["final_status"],
        "covariance_aware_risk": covariance_aware_risk,
        "system_validity_gate_criteria": system_validity_criteria,
        "validation_gate_decision_criteria": validation_gate_decision_criteria,
    }
    required_horizon_text = f"{required_horizon}d"
    return ValidationGateReport(
        system_validity_status=system_status,
        strategy_candidate_status=strategy_status,
        hard_fail=bool(hard_fail_reasons),
        warning=bool(warnings),
        strategy_pass=strategy_status == "pass",
        system_validity_pass=system_status == "pass",
        warnings=warnings,
        hard_fail_reasons=hard_fail_reasons,
        metrics=metrics,
        evidence=evidence,
        horizons=list(DEFAULT_HORIZONS),
        required_validation_horizon=required_horizon_text,
        embargo_periods={horizon: int(horizon.rstrip("d")) for horizon in DEFAULT_HORIZONS},
        benchmark_results=baseline_results,
        ablation_results=list(ablation_summary),
        gate_results=gate_results,
        official_message=official_message,
        baseline_comparison_inputs=baseline_input_rows,
        cost_adjusted_metric_comparison=cost_adjusted_metric_comparison,
        side_by_side_metric_comparison=side_by_side_metric_comparison,
        baseline_comparisons=baseline_comparisons,
        no_model_proxy_ablation=no_model_proxy_ablation,
        model_comparison_results=model_comparison_results,
        structured_warnings=structured_warnings,
        comparison_input_schema=comparison_input_schema,
        comparison_result_schema=comparison_result_schema,
        full_model_metrics=full_model_metrics,
        baseline_metrics=baseline_metrics,
        ablation_metrics=ablation_metrics,
        structured_pass_fail_reasons=structured_pass_fail_reasons,
        system_validity_gate_input_schema=system_validity_gate_input_schema,
        system_validity_gate_output_schema=system_validity_gate_output_schema,
        system_validity_gate_report_schema=system_validity_gate_report_schema,
        strategy_candidate_policy=strategy_candidate_policy.to_dict(),
        strategy_candidate_policy_evaluation=strategy_candidate_policy_evaluation,
    )


def evaluate_average_daily_turnover_gate(
    average_daily_turnover: object,
    *,
    baseline_results: list[dict[str, Any]] | None = None,
    ablation_summary: list[dict[str, Any]] | None = None,
    thresholds: ValidationGateThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate the Stage 1 turnover gate.

    Average daily turnover is a ratio, so 0.35 means 35%. The gate passes at
    and below the threshold, including exactly 35%.
    """

    if isinstance(average_daily_turnover, Mapping) or hasattr(average_daily_turnover, "turnover"):
        metrics = average_daily_turnover
    else:
        metrics = {"turnover": average_daily_turnover}
    return _evaluate_turnover(
        metrics,
        baseline_results or [],
        ablation_summary or [],
        thresholds or ValidationGateThresholds(),
    )


def evaluate_turnover_gate(
    average_daily_turnover: object,
    *,
    baseline_results: list[dict[str, Any]] | None = None,
    ablation_summary: list[dict[str, Any]] | None = None,
    thresholds: ValidationGateThresholds | None = None,
) -> dict[str, Any]:
    return evaluate_average_daily_turnover_gate(
        average_daily_turnover,
        baseline_results=baseline_results,
        ablation_summary=ablation_summary,
        thresholds=thresholds,
    )


def evaluate_monthly_turnover_budget_gate(
    monthly_turnover: object,
    *,
    thresholds: ValidationGateThresholds | None = None,
    monthly_turnover_budget: float | None = None,
    budget: float | None = None,
) -> dict[str, Any]:
    """Evaluate whether monthly turnover stays within the configured budget.

    Inputs may be a scalar monthly turnover, a mapping/object with
    ``monthly_turnover`` or ``turnover``, or a frame/series of daily turnover
    values with dates. Daily turnover values are summed by calendar month.
    """

    thresholds = thresholds or ValidationGateThresholds()
    configured_budget = _monthly_turnover_budget(
        thresholds,
        monthly_turnover_budget if monthly_turnover_budget is not None else budget,
    )
    observations = _monthly_turnover_observations(monthly_turnover)
    if observations.empty:
        return {
            "status": "not_evaluable",
            "reason": "monthly turnover is unavailable",
            "value": None,
            "threshold": configured_budget,
            "budget": configured_budget,
            "max_monthly_turnover": None,
            "operator": "<=",
            "monthly_turnover": {},
        }

    max_turnover = float(observations["turnover"].max())
    monthly_values = {
        str(row["month"]): float(row["turnover"])
        for _, row in observations.sort_values("month").iterrows()
    }
    if max_turnover <= configured_budget or bool(
        np.isclose(max_turnover, configured_budget, rtol=1e-12, atol=1e-12)
    ):
        return {
            "status": "pass",
            "reason": "monthly turnover is within configured budget",
            "value": max_turnover,
            "threshold": configured_budget,
            "budget": configured_budget,
            "max_monthly_turnover": max_turnover,
            "operator": "<=",
            "monthly_turnover": monthly_values,
        }
    return {
        "status": "warning",
        "reason": "monthly turnover exceeds configured budget",
        "value": max_turnover,
        "threshold": configured_budget,
        "budget": configured_budget,
        "max_monthly_turnover": max_turnover,
        "operator": "<=",
        "monthly_turnover": monthly_values,
        "structured_warning": _turnover_budget_structured_warning(
            {
                "reason": "monthly turnover exceeds configured budget",
                "value": max_turnover,
                "threshold": configured_budget,
                "operator": "<=",
                "monthly_turnover": monthly_values,
            }
        ),
    }


def evaluate_monthly_turnover_gate(
    monthly_turnover: object,
    *,
    thresholds: ValidationGateThresholds | None = None,
    monthly_turnover_budget: float | None = None,
    budget: float | None = None,
) -> dict[str, Any]:
    return evaluate_monthly_turnover_budget_gate(
        monthly_turnover,
        thresholds=thresholds,
        monthly_turnover_budget=monthly_turnover_budget,
        budget=budget,
    )


def evaluate_turnover_validity_gate(
    average_daily_turnover: object,
    monthly_turnover: object,
    *,
    baseline_results: list[dict[str, Any]] | None = None,
    ablation_summary: list[dict[str, Any]] | None = None,
    thresholds: ValidationGateThresholds | None = None,
    monthly_turnover_budget: float | None = None,
    budget: float | None = None,
) -> dict[str, Any]:
    """Evaluate combined turnover validity for the Stage 1 gate.

    The turnover gate passes when either the average daily turnover limit or
    the monthly turnover budget condition passes.
    """

    thresholds = thresholds or ValidationGateThresholds()
    daily_gate = evaluate_average_daily_turnover_gate(
        average_daily_turnover,
        baseline_results=baseline_results,
        ablation_summary=ablation_summary,
        thresholds=thresholds,
    )
    monthly_gate = evaluate_monthly_turnover_budget_gate(
        monthly_turnover,
        thresholds=thresholds,
        monthly_turnover_budget=monthly_turnover_budget,
        budget=budget,
    )
    return _combine_turnover_validity_gates(daily_gate, monthly_gate)


def evaluate_strategy_candidate_gate_policy(
    metrics: Mapping[str, Any],
    *,
    policy: StrategyCandidateGatePolicy | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate Strategy Candidate Gate metrics against a configurable policy."""

    policy = _strategy_candidate_gate_policy(policy)
    metric_values = {
        metric: _finite_or_none(metrics.get(metric))
        for metric in policy.required_input_metrics
    }
    missing_metrics = [
        metric
        for metric, value in metric_values.items()
        if value is None
    ]
    required_rule_results = [
        _evaluate_strategy_candidate_metric_rule(rule, metric_values)
        for rule in policy.rules
    ]
    warning_rule_results = [
        _evaluate_strategy_candidate_metric_rule(rule, metric_values)
        for rule in policy.warning_rules
    ]
    failed_required_rules = [
        row["rule_id"]
        for row in required_rule_results
        if row["status"] == "fail"
    ]
    failed_warning_rules = [
        row["rule_id"]
        for row in warning_rule_results
        if row["status"] == "warning"
    ]

    if missing_metrics:
        status: StrategyCandidateStatus = "not_evaluable"
        reason = "required Strategy Candidate Gate input metric(s) are missing: " + ", ".join(
            missing_metrics
        )
    elif failed_required_rules:
        status = "fail"
        reason = "required Strategy Candidate Gate rule(s) failed: " + ", ".join(
            failed_required_rules
        )
    elif failed_warning_rules:
        status = "warning"
        reason = "Strategy Candidate Gate warning rule(s) failed: " + ", ".join(
            failed_warning_rules
        )
    else:
        status = "pass"
        reason = "all Strategy Candidate Gate policy rules passed"

    return {
        "policy_id": policy.policy_id,
        "schema_version": policy.schema_version,
        "status": status,
        "reason": reason,
        "input_metrics": metric_values,
        "missing_metrics": missing_metrics,
        "required_rules": [rule.to_dict() for rule in policy.rules],
        "warning_rules": [rule.to_dict() for rule in policy.warning_rules],
        "required_rule_results": required_rule_results,
        "warning_rule_results": warning_rule_results,
        "failed_required_rules": failed_required_rules,
        "failed_warning_rules": failed_warning_rules,
        "status_precedence": ["not_evaluable", "fail", "warning", "pass"],
        "target_horizon": policy.target_horizon,
        "required_baselines": list(policy.required_baselines),
    }


def write_validity_gate_artifacts(
    report: ValidationGateReport,
    output_dir: str | Path = "reports",
) -> tuple[Path, Path]:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    json_path = path / "validity_gate.json"
    markdown_path = path / "validity_report.md"
    json_path.write_text(report.to_json() + "\n", encoding="utf-8")
    markdown_path.write_text(report.to_markdown() + "\n", encoding="utf-8")
    return json_path, markdown_path


def _strategy_candidate_gate_policy(
    policy: StrategyCandidateGatePolicy | Mapping[str, Any] | None,
) -> StrategyCandidateGatePolicy:
    if policy is None:
        return StrategyCandidateGatePolicy()
    if isinstance(policy, StrategyCandidateGatePolicy):
        return policy
    if not isinstance(policy, Mapping):
        raise TypeError("strategy_candidate_policy must be a StrategyCandidateGatePolicy or mapping")
    payload = dict(policy)
    return StrategyCandidateGatePolicy(
        policy_id=str(payload.get("policy_id", STRATEGY_CANDIDATE_POLICY_ID)),
        schema_version=str(
            payload.get("schema_version", STRATEGY_CANDIDATE_POLICY_SCHEMA_VERSION)
        ),
        target_horizon=str(payload.get("target_horizon", "forward_return_20")),
        required_input_metrics=tuple(
            str(metric)
            for metric in payload.get(
                "required_input_metrics",
                StrategyCandidateGatePolicy().required_input_metrics,
            )
        ),
        required_baselines=tuple(
            str(baseline)
            for baseline in payload.get(
                "required_baselines",
                STRATEGY_CANDIDATE_DEFAULT_REQUIRED_BASELINES,
            )
        ),
        rules=tuple(
            _strategy_candidate_metric_rule(rule)
            for rule in payload.get("rules", StrategyCandidateGatePolicy().rules)
        ),
        warning_rules=tuple(
            _strategy_candidate_metric_rule(rule)
            for rule in payload.get("warning_rules", ())
        ),
    )


def _strategy_candidate_metric_rule(
    rule: StrategyCandidateMetricRule | Mapping[str, Any],
) -> StrategyCandidateMetricRule:
    if isinstance(rule, StrategyCandidateMetricRule):
        return rule
    if not isinstance(rule, Mapping):
        raise TypeError("strategy candidate policy rules must be mappings")
    return StrategyCandidateMetricRule(
        rule_id=str(rule.get("rule_id", "")),
        metric=str(rule.get("metric", "")),
        operator=str(rule.get("operator", ">=")),  # type: ignore[arg-type]
        threshold=float(rule.get("threshold", 0.0)),
        severity=str(rule.get("severity", "required")),  # type: ignore[arg-type]
        description=str(rule.get("description", "")),
    )


def _evaluate_strategy_candidate_metric_rule(
    rule: StrategyCandidateMetricRule,
    metrics: Mapping[str, float | None],
) -> dict[str, Any]:
    value = metrics.get(rule.metric)
    if value is None:
        return {
            "rule_id": rule.rule_id,
            "metric": rule.metric,
            "status": "not_evaluable",
            "value": None,
            "threshold": rule.threshold,
            "operator": rule.operator,
            "severity": rule.severity,
            "passed": None,
            "reason": f"{rule.metric} is unavailable",
        }
    passed = _compare_strategy_candidate_metric(
        value,
        rule.threshold,
        rule.operator,
    )
    failed_status = "warning" if rule.severity == "warning" else "fail"
    return {
        "rule_id": rule.rule_id,
        "metric": rule.metric,
        "status": "pass" if passed else failed_status,
        "value": value,
        "threshold": rule.threshold,
        "operator": rule.operator,
        "severity": rule.severity,
        "passed": bool(passed),
        "reason": (
            f"{rule.metric} {rule.operator} {rule.threshold:g} passed"
            if passed
            else f"{rule.metric} {rule.operator} {rule.threshold:g} failed"
        ),
        "description": rule.description,
    }


def _compare_strategy_candidate_metric(
    value: float,
    threshold: float,
    operator: StrategyCandidateRuleOperator,
) -> bool:
    if operator == ">":
        return value > threshold and not np.isclose(value, threshold, rtol=1e-12, atol=1e-12)
    if operator == ">=":
        return value >= threshold or bool(np.isclose(value, threshold, rtol=1e-12, atol=1e-12))
    if operator == "<":
        return value < threshold and not np.isclose(value, threshold, rtol=1e-12, atol=1e-12)
    if operator == "<=":
        return value <= threshold or bool(np.isclose(value, threshold, rtol=1e-12, atol=1e-12))
    return bool(np.isclose(value, threshold, rtol=1e-12, atol=1e-12))


def _strategy_candidate_policy_input_metrics(
    *,
    rank_ic: Mapping[str, Any],
    baseline_comparisons: Mapping[str, Mapping[str, Any]],
    strategy_metrics: object,
    model_value_gate: Mapping[str, Any],
) -> dict[str, Any]:
    spy_comparison = _mapping_or_empty(baseline_comparisons.get("SPY"))
    equal_weight_comparison = _mapping_or_empty(
        baseline_comparisons.get(EQUAL_WEIGHT_BASELINE_NAME)
    )
    return {
        "mean_rank_ic": rank_ic.get("mean_rank_ic"),
        "oos_rank_ic": rank_ic.get("oos_rank_ic"),
        "positive_fold_ratio": rank_ic.get("positive_fold_ratio"),
        "cost_adjusted_excess_return_vs_spy": spy_comparison.get(
            "strategy_excess_return",
            spy_comparison.get("excess_return"),
        ),
        "cost_adjusted_excess_return_vs_equal_weight": equal_weight_comparison.get(
            "strategy_excess_return",
            equal_weight_comparison.get("excess_return"),
        ),
        "max_drawdown": _metric(strategy_metrics, "max_drawdown"),
        "average_daily_turnover": _metric(strategy_metrics, "turnover"),
        "proxy_ic_improvement": _proxy_rank_ic_improvement(model_value_gate),
    }


def _proxy_rank_ic_improvement(model_value_gate: Mapping[str, Any]) -> float | None:
    best: float | None = None
    for comparison in model_value_gate.get("comparisons", []) or []:
        if not isinstance(comparison, Mapping):
            continue
        baseline = str(comparison.get("baseline_scenario") or comparison.get("baseline") or "")
        if baseline != NO_MODEL_PROXY_ABLATION_SCENARIO:
            continue
        for metric_row in comparison.get("metrics", []) or []:
            if not isinstance(metric_row, Mapping):
                continue
            if str(metric_row.get("metric")) != "rank_ic":
                continue
            improvement = _finite_or_none(metric_row.get("improvement"))
            if improvement is None:
                continue
            if best is None or improvement > best:
                best = improvement
    return best


def _model_comparison_config_payload(config: object | None) -> dict[str, Any]:
    model_config = getattr(config, "model_comparison_config", None)
    if model_config is None:
        model_config = default_model_comparison_config()

    to_dict = getattr(model_config, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        return _mapping_or_empty(payload)
    if isinstance(model_config, Mapping):
        return dict(model_config)
    return {}


def _covariance_aware_risk_report_payload(
    config: object | None,
    equity_curve: pd.DataFrame,
    strategy_metrics: object,
) -> dict[str, Any]:
    default_config = DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG
    configured_enabled = bool(
        getattr(config, "covariance_aware_risk_enabled", default_config.covariance_aware_risk.enabled)
    )
    parameters = {
        "schema_version": str(
            getattr(config, "risk_constraint_schema_version", default_config.schema_version)
        ),
        "max_holdings": int(getattr(config, "top_n", default_config.max_holdings)),
        "max_symbol_weight": _attribute_float(
            config,
            "max_symbol_weight",
            default_config.max_symbol_weight,
        ),
        "max_sector_weight": _attribute_float(
            config,
            "max_sector_weight",
            default_config.max_sector_weight,
        ),
        "max_position_risk_contribution": _attribute_float(
            config,
            "max_position_risk_contribution",
            default_config.max_position_risk_contribution,
        ),
        "portfolio_volatility_limit": _attribute_float(
            config,
            "portfolio_volatility_limit",
            default_config.portfolio_volatility_limit,
        ),
        "lookback_periods": int(
            getattr(
                config,
                "portfolio_covariance_lookback",
                default_config.portfolio_covariance_lookback,
            )
        ),
        "return_column": str(
            getattr(
                config,
                "covariance_return_column",
                default_config.covariance_aware_risk.return_column,
            )
        ),
        "min_periods": int(
            getattr(
                config,
                "covariance_min_periods",
                default_config.covariance_aware_risk.min_periods,
            )
        ),
        "fallback": default_config.covariance_aware_risk.fallback,
        "long_only": True,
        "v1_exclusions": list(default_config.v1_exclusions),
    }
    observed_metric_names = (
        "average_portfolio_volatility_estimate",
        "max_portfolio_volatility_estimate",
        "max_position_risk_contribution",
    )
    realized_metrics = {
        name: _metric(strategy_metrics, name)
        for name in observed_metric_names
    }
    latest = _latest_covariance_risk_observation(equity_curve)
    realized_metrics.update(latest)
    observed = any(_finite_or_none(value) is not None for value in realized_metrics.values())
    status = (
        "applied"
        if configured_enabled and observed
        else "configured_but_not_observed"
        if configured_enabled
        else "disabled"
    )
    return {
        "configured_enabled": configured_enabled,
        "applied": bool(configured_enabled and observed),
        "status": status,
        "parameters": parameters,
        "realized_metrics": realized_metrics,
        "application_evidence": {
            "has_portfolio_volatility_estimate": bool(
                "portfolio_volatility_estimate" in equity_curve.columns
                and equity_curve["portfolio_volatility_estimate"].notna().any()
            ),
            "has_position_sizing_validation": bool(
                "position_sizing_validation_status" in equity_curve.columns
                and equity_curve["position_sizing_validation_status"].notna().any()
            ),
            "has_risk_contribution_metric": bool(
                _finite_or_none(
                    realized_metrics.get("max_position_risk_contribution")
                )
                is not None
            ),
        },
    }


def _latest_covariance_risk_observation(equity_curve: pd.DataFrame) -> dict[str, Any]:
    if equity_curve.empty:
        return {}
    columns = [
        "portfolio_volatility_estimate",
        "position_sizing_validation_status",
        "position_sizing_validation_rule",
        "position_sizing_validation_reason",
        "position_count",
        "max_position_weight",
        "max_sector_exposure",
        "max_position_risk_contribution",
    ]
    available = [column for column in columns if column in equity_curve.columns]
    if not available:
        return {}
    latest = equity_curve[available].tail(1).to_dict("records")[0]
    return {
        f"latest_{key}": value
        for key, value in latest.items()
    }


def _attribute_float(obj: object | None, name: str, default: float) -> float:
    if obj is None:
        return float(default)
    value = getattr(obj, name, default)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(numeric):
        return float(default)
    return numeric


def _full_model_metric_contract(
    config_payload: Mapping[str, Any],
    rank_ic: Mapping[str, Any],
    cost_adjusted_metric_comparison: Iterable[Mapping[str, Any]],
    ablation_summary: Iterable[Mapping[str, Any]],
    *,
    strategy_status: str,
    positive_fold_ratio_threshold: float,
) -> dict[str, Any]:
    entity_id = str(
        config_payload.get("full_model_candidate_id")
        or config_payload.get("primary_candidate_id")
        or "all_features"
    )
    scenario_id = _model_candidate_ablation_scenario_id(config_payload, entity_id)
    strategy_row = _strategy_metric_comparison_row(cost_adjusted_metric_comparison)
    ablation_row = _row_by_scenario(ablation_summary).get(scenario_id, {})
    metrics = {
        **_metric_contract_values(ablation_row),
        **_metric_contract_values(strategy_row),
    }
    validation_metrics = {
        key: rank_ic.get(key)
        for key in (
            "mean_rank_ic",
            "positive_fold_ratio",
            "oos_rank_ic",
            "rank_ic_count",
            "insufficient_data",
            "insufficient_data_status",
            "insufficient_data_code",
            "insufficient_data_reason",
        )
        if key in rank_ic
    }
    validation_metrics["positive_fold_ratio_threshold"] = positive_fold_ratio_threshold
    validation_metrics["positive_fold_ratio_passed"] = _metric_threshold_passed(
        validation_metrics.get("positive_fold_ratio"),
        positive_fold_ratio_threshold,
    )
    return {
        "entity_id": entity_id,
        "scenario": scenario_id,
        "role": "full_model",
        "status": strategy_status,
        "metrics": metrics,
        "validation_metrics": validation_metrics,
        "source": "strategy_cost_adjusted_metrics_and_walk_forward_rank_ic",
    }


def _baseline_metric_contract(
    comparison_result_schema: Mapping[str, Any],
    baseline_comparisons: Mapping[str, Mapping[str, Any]],
    ablation_summary: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_scenario = _row_by_scenario(ablation_summary)
    rows: list[dict[str, Any]] = []
    for result in _iter_mapping_rows_from_object(comparison_result_schema.get("baseline_results")):
        entity_id = str(result.get("entity_id") or "")
        metrics = _mapping_or_empty(result.get("metrics"))
        if result.get("role") == "model_baseline":
            scenario = str(result.get("ablation_scenario_id") or entity_id)
            metrics = {
                **_metric_contract_values(rows_by_scenario.get(scenario, {})),
                **dict(metrics),
            }
        else:
            baseline_name = _baseline_name_from_entity_id(entity_id)
            baseline_row = baseline_comparisons.get(baseline_name, {})
            metrics = {
                **_metric_contract_values(baseline_row),
                **dict(metrics),
            }
        rows.append(
            {
                "entity_id": entity_id,
                "label": result.get("label"),
                "role": result.get("role"),
                "status": result.get("status"),
                "baseline_type": result.get("baseline_type"),
                "sample_count": result.get("sample_count"),
                "evaluation_window": result.get("evaluation_window"),
                "metrics": metrics,
                "source": result.get("result_source", "comparison_result_schema"),
            }
        )
    return rows


def _ablation_metric_contract(
    comparison_result_schema: Mapping[str, Any],
    ablation_summary: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_scenario = _row_by_scenario(ablation_summary)
    rows: list[dict[str, Any]] = []
    for result in _iter_mapping_rows_from_object(comparison_result_schema.get("ablation_results")):
        entity_id = str(result.get("entity_id") or "")
        source_row = rows_by_scenario.get(entity_id, {})
        metrics = {
            **_metric_contract_values(source_row),
            **_mapping_or_empty(result.get("metrics")),
        }
        rows.append(
            {
                "entity_id": entity_id,
                "label": result.get("label"),
                "role": result.get("role"),
                "kind": source_row.get("kind"),
                "status": result.get("status"),
                "sample_count": result.get("sample_count"),
                "metrics": metrics,
                "pipeline_controls": source_row.get("pipeline_controls", {}),
                "source": result.get("result_source", "ablation_summary"),
            }
        )
    return rows


def _structured_pass_fail_reasons(
    gate_results: Mapping[str, Any],
    model_comparison_results: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _rule_result_explanations(gate_results):
        rows.append(
            {
                "category": "gate",
                "entity_id": row.get("gate"),
                "rule": row.get("rule"),
                "metric": row.get("metric"),
                "status": row.get("status"),
                "passed": row.get("passed"),
                "reason_code": row.get("reason_code"),
                "reason": row.get("reason"),
                "value": row.get("value"),
                "threshold": row.get("threshold"),
                "operator": row.get("operator"),
                "affects_strategy": row.get("affects_strategy"),
                "affects_system": row.get("affects_system"),
            }
        )
    for row in model_comparison_results:
        if not isinstance(row, Mapping):
            continue
        status = str(row.get("status") or row.get("pass_fail") or "not_evaluable")
        rows.append(
            {
                "category": "model_comparison",
                "entity_id": row.get("baseline_id") or row.get("baseline"),
                "rule": "model_outperformance",
                "metric": row.get("metric"),
                "status": status,
                "passed": row.get("passed"),
                "reason_code": row.get("reason_code"),
                "reason": row.get("reason") or _model_comparison_reason(row),
                "candidate": row.get("candidate"),
                "baseline": row.get("baseline"),
                "window_id": row.get("window_id"),
                "value": row.get("improvement", row.get("absolute_delta")),
                "threshold": row.get("outperformance_threshold"),
                "operator": row.get("operator"),
                "candidate_value": row.get("candidate_value"),
                "baseline_value": row.get("baseline_value"),
            }
        )
    return rows


def _metric_contract_values(row: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        return {}
    metrics: dict[str, Any] = {}
    for metric in (
        "rank_ic",
        "mean_rank_ic",
        "positive_fold_ratio",
        "oos_rank_ic",
        "validation_mean_information_coefficient",
        "validation_positive_ic_fold_ratio",
        "validation_oos_information_coefficient",
        "cagr",
        "sharpe",
        "max_drawdown",
        "cumulative_return",
        "gross_cumulative_return",
        "cost_adjusted_cumulative_return",
        "excess_return",
        "turnover",
        "average_daily_turnover",
        "transaction_cost_return",
        "slippage_cost_return",
        "total_cost_return",
        "evaluation_observations",
        "evaluation_start",
        "evaluation_end",
        "return_column",
        "return_horizon",
    ):
        if metric in row and _contract_value_present(row.get(metric)):
            metrics[metric] = row.get(metric)
    signal_metrics = row.get("deterministic_signal_evaluation_metrics")
    if isinstance(signal_metrics, Mapping):
        for metric in (
            "return_basis",
            "cost_adjusted_cumulative_return",
            "average_daily_turnover",
            "total_cost_return",
            "excess_return",
            "action_counts",
        ):
            if metric in signal_metrics and _contract_value_present(signal_metrics.get(metric)):
                metrics[f"signal_{metric}"] = signal_metrics.get(metric)
                metrics.setdefault(metric, signal_metrics.get(metric))
    if "turnover" not in metrics and "signal_average_daily_turnover" in metrics:
        metrics["turnover"] = metrics["signal_average_daily_turnover"]
    return metrics


def _row_by_scenario(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("scenario")): row
        for row in rows
        if isinstance(row, Mapping) and str(row.get("scenario", "")).strip()
    }


def _contract_value_present(value: object) -> bool:
    if value is None:
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return True
    if isinstance(missing, (bool, np.bool_)):
        return not bool(missing)
    return True


def _iter_mapping_rows_from_object(value: object) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        return tuple(row for row in value if isinstance(row, Mapping))
    return ()


def _baseline_name_from_entity_id(entity_id: str) -> str:
    if entity_id == "return_baseline_equal_weight":
        return EQUAL_WEIGHT_BASELINE_NAME
    prefix = "return_baseline_"
    if entity_id.startswith(prefix):
        return entity_id.removeprefix(prefix).upper()
    return entity_id


def _model_comparison_reason(row: Mapping[str, Any]) -> str:
    status = str(row.get("status") or row.get("pass_fail") or "not_evaluable")
    metric = row.get("metric")
    baseline = row.get("baseline")
    if status == "pass":
        return f"full model materially outperformed {baseline} on {metric}"
    if status == "fail":
        return f"full model did not clear required improvement over {baseline} on {metric}"
    return f"model comparison is not evaluable for {baseline} on {metric}"


def _model_comparison_results(
    ablation_summary: list[dict[str, Any]],
    config: object | None,
    *,
    baseline_comparisons: Mapping[str, Mapping[str, Any]] | None = None,
    cost_adjusted_metric_comparison: Iterable[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    config_payload = _model_comparison_config_payload(config)
    candidate_id = str(
        config_payload.get("full_model_candidate_id")
        or config_payload.get("primary_candidate_id")
        or "all_features"
    )
    baseline_id = str(config_payload.get("baseline_candidate_id") or "no_model_proxy")
    metric_ids = [str(metric) for metric in config_payload.get("metrics", [])]
    if not metric_ids:
        return []

    candidate_scenario_id = _model_candidate_ablation_scenario_id(
        config_payload,
        candidate_id,
    )
    baseline_scenario_id = _model_candidate_ablation_scenario_id(
        config_payload,
        baseline_id,
    )
    rows_by_scenario = {
        str(row.get("scenario")): row
        for row in ablation_summary
        if str(row.get("scenario", "")).strip()
    }
    candidate_row = rows_by_scenario.get(candidate_scenario_id) or rows_by_scenario.get(
        candidate_id
    )
    candidate_missing = candidate_row is None
    if candidate_row is None:
        candidate_row = {
            "scenario": candidate_scenario_id,
            "label": candidate_id,
            "validation_status": "not_evaluable",
            "reason_code": "missing_candidate_model",
            "reason": f"candidate model result is missing: {candidate_scenario_id}",
        }

    strategy_row = _strategy_metric_comparison_row(cost_adjusted_metric_comparison)
    comparison_targets = _model_comparison_targets(
        rows_by_scenario,
        config_payload,
        baseline_comparisons or {},
        baseline_id=baseline_id,
        baseline_scenario_id=baseline_scenario_id,
        candidate_scenario_id=candidate_scenario_id,
    )
    results: list[dict[str, Any]] = []
    for target in comparison_targets:
        candidate_source = (
            strategy_row
            if target["role"] == "return_baseline" and strategy_row
            else candidate_row
        )
        candidate_windows = _model_comparison_windows(
            candidate_source,
            fallback_window_id="strategy_evaluation",
        )
        target_windows = _model_comparison_windows(
            target["row"],
            fallback_window_id="strategy_evaluation",
        )
        for window_id in _model_comparison_window_ids(candidate_windows, target_windows):
            candidate_window = candidate_windows.get(window_id) or {}
            target_window = target_windows.get(window_id) or {}
            window_metadata = _model_comparison_window_metadata(
                window_id,
                candidate_window,
                target_window,
            )
            for metric in metric_ids:
                candidate_value = _model_comparison_metric_value(candidate_window, metric)
                baseline_value = _model_comparison_metric_value(target_window, metric)
                threshold = _model_comparison_outperformance_threshold(
                    config_payload,
                    metric,
                )
                absolute_delta = _model_comparison_absolute_delta(
                    candidate_value,
                    baseline_value,
                )
                relative_delta = _model_comparison_relative_delta(
                    absolute_delta,
                    baseline_value,
                )
                improvement = _model_comparison_improvement(
                    metric,
                    candidate_value,
                    baseline_value,
                )
                pass_fail, passed = _model_comparison_pass_fail(
                    metric,
                    candidate_value,
                    baseline_value,
                    threshold,
                )
                coverage = _model_comparison_coverage_status(
                    metric=metric,
                    candidate_missing=candidate_missing,
                    target_missing=bool(target.get("missing_model")),
                    candidate_window_available=window_id in candidate_windows,
                    target_window_available=window_id in target_windows,
                    candidate_value=candidate_value,
                    baseline_value=baseline_value,
                    candidate_scenario_id=candidate_scenario_id,
                    baseline_scenario_id=str(target["scenario_id"]),
                    window_id=window_id,
                )
                results.append(
                    {
                        "window_id": window_id,
                        "window_label": window_metadata["label"],
                        "window_role": window_metadata["role"],
                        "metric": metric,
                        "candidate": candidate_id,
                        "candidate_id": candidate_id,
                        "candidate_scenario": candidate_scenario_id,
                        "candidate_value": candidate_value,
                        "baseline": target["entity_id"],
                        "baseline_id": target["entity_id"],
                        "baseline_label": target["label"],
                        "baseline_role": target["role"],
                        "baseline_scenario": target["scenario_id"],
                        "baseline_value": baseline_value,
                        "delta": absolute_delta,
                        "absolute_delta": absolute_delta,
                        "relative_delta": relative_delta,
                        "improvement": improvement,
                        "outperformance_threshold": threshold,
                        "operator": _model_comparison_operator(metric, threshold),
                        "pass_fail": pass_fail,
                        "status": pass_fail,
                        "passed": passed,
                        "coverage_status": coverage["coverage_status"],
                        "reason_code": coverage["reason_code"],
                        "reason": coverage["reason"],
                        "candidate_model_available": not candidate_missing,
                        "baseline_model_available": not bool(target.get("missing_model")),
                        "candidate_window_available": window_id in candidate_windows,
                        "baseline_window_available": window_id in target_windows,
                    }
                )
    return results


def _model_comparison_targets(
    rows_by_scenario: Mapping[str, Mapping[str, Any]],
    config_payload: Mapping[str, Any],
    baseline_comparisons: Mapping[str, Mapping[str, Any]],
    *,
    baseline_id: str,
    baseline_scenario_id: str,
    candidate_scenario_id: str,
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    baseline_row = rows_by_scenario.get(baseline_scenario_id) or rows_by_scenario.get(
        baseline_id
    )
    _append_model_comparison_target(
        targets,
        seen,
        entity_id=baseline_id,
        label=str((baseline_row or {}).get("label") or baseline_id),
        role="model_baseline",
        scenario_id=baseline_scenario_id,
        row=baseline_row or {
            "scenario": baseline_scenario_id,
            "label": baseline_id,
            "validation_status": "not_evaluable",
            "reason_code": "missing_baseline_model",
            "reason": f"baseline model result is missing: {baseline_scenario_id}",
        },
        missing_model=baseline_row is None,
    )

    for row in baseline_comparisons.values():
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("name") or row.get("baseline_name") or "").strip()
        if not name:
            continue
        entity_id = (
            "return_baseline_equal_weight"
            if name == EQUAL_WEIGHT_BASELINE_NAME
            else f"return_baseline_{_safe_identifier(name)}"
        )
        _append_model_comparison_target(
            targets,
            seen,
            entity_id=entity_id,
            label=f"{name} return baseline",
            role="return_baseline",
            scenario_id=name,
            row=row,
        )

    configured_candidates = _configured_comparison_candidates(config_payload)
    for candidate in configured_candidates:
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        if not candidate_id or candidate_id == baseline_id:
            continue
        scenario_id = str(candidate.get("ablation_scenario_id") or candidate_id)
        if scenario_id == candidate_scenario_id:
            continue
        row = rows_by_scenario.get(scenario_id) or rows_by_scenario.get(candidate_id)
        if row is None:
            continue
        role = "diagnostic" if candidate.get("role") == "diagnostic" else "ablation"
        _append_model_comparison_target(
            targets,
            seen,
            entity_id=candidate_id,
            label=str(candidate.get("label") or row.get("label") or candidate_id),
            role=role,
            scenario_id=scenario_id,
            row=row,
        )

    for scenario_id, row in rows_by_scenario.items():
        if scenario_id in {candidate_scenario_id, baseline_scenario_id}:
            continue
        role = "diagnostic" if row.get("kind") == "cost" else "ablation"
        _append_model_comparison_target(
            targets,
            seen,
            entity_id=scenario_id,
            label=str(row.get("label") or scenario_id),
            role=role,
            scenario_id=scenario_id,
            row=row,
        )
    return targets


def _append_model_comparison_target(
    targets: list[dict[str, Any]],
    seen: set[tuple[str, str]],
    *,
    entity_id: str,
    label: str,
    role: str,
    scenario_id: str,
    row: Mapping[str, Any],
    missing_model: bool = False,
) -> None:
    key = (str(role), str(entity_id))
    if key in seen:
        return
    seen.add(key)
    targets.append(
        {
            "entity_id": str(entity_id),
            "label": str(label),
            "role": str(role),
            "scenario_id": str(scenario_id),
            "row": row,
            "missing_model": bool(missing_model),
        }
    )


def _configured_comparison_candidates(
    config_payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    candidates = config_payload.get("candidates", ())
    if isinstance(candidates, Iterable) and not isinstance(candidates, (str, bytes, Mapping)):
        return tuple(candidate for candidate in candidates if isinstance(candidate, Mapping))
    return ()


def _strategy_metric_comparison_row(
    cost_adjusted_metric_comparison: Iterable[Mapping[str, Any]],
) -> Mapping[str, Any]:
    for row in cost_adjusted_metric_comparison:
        if isinstance(row, Mapping) and str(row.get("name")) == "strategy":
            return row
    return {}


def _model_comparison_windows(
    row: Mapping[str, Any],
    *,
    fallback_window_id: str,
) -> dict[str, Mapping[str, Any]]:
    explicit = _explicit_model_comparison_windows(row)
    if explicit:
        return explicit
    return {
        fallback_window_id: {
            **dict(row),
            "window_id": fallback_window_id,
            "window_label": str(row.get("window_label") or "Strategy evaluation"),
            "window_role": str(row.get("window_role") or "strategy_evaluation"),
        }
    }


def _explicit_model_comparison_windows(
    row: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    for key in (
        "per_window_metrics",
        "window_metrics",
        "validation_window_metrics",
        "validation_windows",
        "fold_metrics",
        "fold_results",
        "windows",
    ):
        value = row.get(key)
        windows = _normalize_model_comparison_windows(value)
        if windows:
            return windows
    return {}


def _normalize_model_comparison_windows(value: object) -> dict[str, Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    if isinstance(value, pd.DataFrame):
        rows = [row for row in value.to_dict("records") if isinstance(row, Mapping)]
    elif isinstance(value, Mapping):
        for window_id, metrics in value.items():
            if not isinstance(metrics, Mapping):
                continue
            row = dict(metrics)
            row.setdefault("window_id", str(window_id))
            rows.append(row)
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        rows = [row for row in value if isinstance(row, Mapping)]

    normalized: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        metrics = dict(_mapping_or_empty(row.get("metrics")))
        merged = {**dict(row), **metrics}
        window_id = _model_comparison_window_id(merged, index)
        merged["window_id"] = window_id
        merged.setdefault("window_label", merged.get("label") or f"Window {window_id}")
        merged.setdefault("window_role", merged.get("role") or merged.get("fold_type") or "")
        normalized[window_id] = merged
    return normalized


def _model_comparison_window_id(row: Mapping[str, Any], fallback_index: int) -> str:
    for key in ("window_id", "validation_window_id", "fold_id"):
        value = _model_comparison_window_id_value(row.get(key))
        if value is not None:
            return value
    fold = _model_comparison_window_id_value(row.get("fold"))
    if fold is not None:
        return f"fold_{fold}"
    return f"window_{fallback_index}"


def _model_comparison_window_id_value(value: object) -> str | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _model_comparison_window_ids(
    candidate_windows: Mapping[str, Mapping[str, Any]],
    target_windows: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    candidate_ids = tuple(candidate_windows)
    target_ids = tuple(target_windows)
    if candidate_ids == ("strategy_evaluation",) or target_ids == ("strategy_evaluation",):
        return ("strategy_evaluation",)
    return tuple(dict.fromkeys((*candidate_ids, *target_ids)))


def _model_comparison_window_metadata(
    window_id: str,
    candidate_window: Mapping[str, Any],
    target_window: Mapping[str, Any],
) -> dict[str, str]:
    source = candidate_window or target_window
    label = str(
        source.get("window_label")
        or source.get("label")
        or ("Strategy evaluation" if window_id == "strategy_evaluation" else window_id)
    )
    role = str(
        source.get("window_role")
        or source.get("role")
        or source.get("fold_type")
        or ("strategy_evaluation" if window_id == "strategy_evaluation" else "")
    )
    return {"label": label, "role": role}


def _model_comparison_absolute_delta(
    candidate_value: float | None,
    baseline_value: float | None,
) -> float | None:
    if candidate_value is None or baseline_value is None:
        return None
    return float(candidate_value - baseline_value)


def _model_comparison_relative_delta(
    absolute_delta: float | None,
    baseline_value: float | None,
) -> float | None:
    if absolute_delta is None or baseline_value is None:
        return None
    denominator = abs(float(baseline_value))
    if denominator <= 1e-12:
        return None
    return float(absolute_delta / denominator)


def _model_candidate_ablation_scenario_id(
    config_payload: Mapping[str, Any],
    candidate_id: str,
) -> str:
    candidates = config_payload.get("candidates", [])
    if not isinstance(candidates, Iterable) or isinstance(candidates, (str, bytes)):
        return candidate_id
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        if str(candidate.get("candidate_id")) != candidate_id:
            continue
        scenario_id = str(candidate.get("ablation_scenario_id") or "").strip()
        return scenario_id or candidate_id
    return candidate_id


def _model_comparison_metric_value(
    row: Mapping[str, Any],
    metric: str,
) -> float | None:
    signal_metrics = _ablation_signal_metrics(row)
    row_keys = {
        "rank_ic": (
            "rank_ic",
            "validation_mean_information_coefficient",
            "mean_rank_ic",
        ),
        "positive_fold_ratio": (
            "validation_positive_ic_fold_ratio",
            "positive_fold_ratio",
        ),
        "oos_rank_ic": (
            "validation_oos_information_coefficient",
            "oos_rank_ic",
        ),
        "sharpe": ("sharpe",),
        "max_drawdown": ("max_drawdown",),
        "cost_adjusted_cumulative_return": (
            "signal_cost_adjusted_cumulative_return",
            "cost_adjusted_cumulative_return",
        ),
        "excess_return": ("excess_return", "signal_excess_return"),
        "turnover": ("signal_average_daily_turnover", "turnover"),
    }
    signal_keys = {
        "rank_ic": ("rank_ic", "information_coefficient"),
        "positive_fold_ratio": ("positive_fold_ratio",),
        "oos_rank_ic": ("oos_rank_ic", "oos_information_coefficient"),
        "sharpe": ("sharpe",),
        "max_drawdown": ("max_drawdown",),
        "cost_adjusted_cumulative_return": ("cost_adjusted_cumulative_return",),
        "excess_return": ("excess_return",),
        "turnover": ("average_daily_turnover",),
    }

    for key in row_keys.get(metric, (metric,)):
        value = _finite_or_none(row.get(key))
        if value is not None:
            return value
    for key in signal_keys.get(metric, (metric,)):
        value = _finite_or_none(signal_metrics.get(key))
        if value is not None:
            return value
    return None


def _model_comparison_pass_fail(
    metric: str,
    candidate_value: float | None,
    baseline_value: float | None,
    threshold: float = 0.0,
) -> tuple[str, bool | None]:
    if candidate_value is None or baseline_value is None:
        return "not_evaluable", None
    improvement = _model_comparison_improvement(metric, candidate_value, baseline_value)
    passed = improvement > threshold and not bool(
        np.isclose(improvement, threshold, rtol=1e-12, atol=1e-12)
    )
    return ("pass" if passed else "fail", bool(passed))


def _model_comparison_operator(metric: str, threshold: float = 0.0) -> str:
    if metric in MODEL_COMPARISON_LOWER_IS_BETTER_METRICS:
        return f"baseline - candidate > {threshold:g}"
    return f"candidate - baseline > {threshold:g}"


def _model_comparison_improvement(
    metric: str,
    candidate_value: float | None,
    baseline_value: float | None,
) -> float | None:
    if candidate_value is None or baseline_value is None:
        return None
    if metric in MODEL_COMPARISON_LOWER_IS_BETTER_METRICS:
        return float(baseline_value - candidate_value)
    return float(candidate_value - baseline_value)


def _model_comparison_outperformance_threshold(
    config_payload: Mapping[str, Any],
    metric: str,
) -> float:
    thresholds = _mapping_or_empty(config_payload.get("outperformance_thresholds"))
    minimums = _mapping_or_empty(thresholds.get("minimum_metric_improvements"))
    value = _finite_or_none(minimums.get(metric))
    if value is None:
        value = _finite_or_none(MODEL_VALUE_MIN_MATERIAL_IMPROVEMENT_BY_METRIC.get(metric))
    return float(value if value is not None else 0.0)


def _model_comparison_outperformance_thresholds(
    config_payload: Mapping[str, Any],
) -> dict[str, float]:
    thresholds = _mapping_or_empty(config_payload.get("outperformance_thresholds"))
    minimums = _mapping_or_empty(thresholds.get("minimum_metric_improvements"))
    merged = dict(MODEL_VALUE_MIN_MATERIAL_IMPROVEMENT_BY_METRIC)
    for metric, value in minimums.items():
        finite_value = _finite_or_none(value)
        if finite_value is not None:
            merged[str(metric)] = float(finite_value)
    return merged


def _model_comparison_coverage_status(
    *,
    metric: str,
    candidate_missing: bool,
    target_missing: bool,
    candidate_window_available: bool,
    target_window_available: bool,
    candidate_value: float | None,
    baseline_value: float | None,
    candidate_scenario_id: str,
    baseline_scenario_id: str,
    window_id: str,
) -> dict[str, str | None]:
    if candidate_missing:
        return {
            "coverage_status": "not_evaluable",
            "reason_code": "missing_candidate_model",
            "reason": f"candidate model result is missing: {candidate_scenario_id}",
        }
    if target_missing:
        return {
            "coverage_status": "not_evaluable",
            "reason_code": "missing_baseline_model",
            "reason": f"baseline model result is missing: {baseline_scenario_id}",
        }
    if not candidate_window_available:
        return {
            "coverage_status": "not_evaluable",
            "reason_code": "missing_candidate_window",
            "reason": (
                f"candidate validation window is missing for {candidate_scenario_id}: "
                f"{window_id}"
            ),
        }
    if not target_window_available:
        return {
            "coverage_status": "not_evaluable",
            "reason_code": "missing_baseline_window",
            "reason": (
                f"baseline validation window is missing for {baseline_scenario_id}: "
                f"{window_id}"
            ),
        }
    if candidate_value is None:
        return {
            "coverage_status": "not_evaluable",
            "reason_code": "missing_candidate_metric",
            "reason": (
                f"candidate metric is missing for {candidate_scenario_id} "
                f"window={window_id} metric={metric}"
            ),
        }
    if baseline_value is None:
        return {
            "coverage_status": "not_evaluable",
            "reason_code": "missing_baseline_metric",
            "reason": (
                f"baseline metric is missing for {baseline_scenario_id} "
                f"window={window_id} metric={metric}"
            ),
        }
    return {"coverage_status": "pass", "reason_code": None, "reason": None}


def _evaluate_model_value_warning(
    ablation_summary: list[dict[str, Any]],
    config: object | None,
) -> dict[str, Any]:
    config_payload = _model_comparison_config_payload(config)
    candidate_id = str(
        config_payload.get("full_model_candidate_id")
        or config_payload.get("primary_candidate_id")
        or "all_features"
    )
    candidate_scenario = _model_candidate_ablation_scenario_id(
        config_payload,
        candidate_id,
    )
    rows_by_scenario = {
        str(row.get("scenario")): row
        for row in ablation_summary
        if str(row.get("scenario", "")).strip()
    }
    candidate_row = rows_by_scenario.get(candidate_scenario) or rows_by_scenario.get(
        candidate_id
    )
    if candidate_row is None:
        return {
            "status": "not_evaluable",
            "reason": "full model ablation result is unavailable for model-value comparison",
            "reason_code": "model_value_candidate_unavailable",
            "affects_strategy": False,
            "affects_insufficient_data": False,
            "candidate": candidate_id,
            "candidate_scenario": candidate_scenario,
            "baseline_scenarios": list(
                _model_value_baseline_scenarios(config_payload, candidate_scenario)
            ),
            "comparisons": [],
        }

    metric_ids = tuple(
        metric
        for metric in (str(metric) for metric in config_payload.get("metrics", ()))
        if _model_comparison_outperformance_threshold(config_payload, metric) > 0
    )
    if not metric_ids:
        metric_ids = tuple(MODEL_VALUE_MIN_MATERIAL_IMPROVEMENT_BY_METRIC)

    comparisons: list[dict[str, Any]] = []
    warning_baselines: list[str] = []
    best_warning_metric: dict[str, Any] | None = None
    for baseline_scenario in _model_value_baseline_scenarios(
        config_payload,
        candidate_scenario,
    ):
        baseline_row = rows_by_scenario.get(baseline_scenario)
        if baseline_row is None:
            continue

        metric_rows: list[dict[str, Any]] = []
        for metric in metric_ids:
            candidate_value = _model_comparison_metric_value(candidate_row, metric)
            baseline_value = _model_comparison_metric_value(baseline_row, metric)
            if candidate_value is None or baseline_value is None:
                continue
            threshold = _model_comparison_outperformance_threshold(
                config_payload,
                metric,
            )
            improvement = _model_value_metric_improvement(
                metric,
                candidate_value,
                baseline_value,
            )
            materially_better = improvement > threshold and not np.isclose(
                improvement,
                threshold,
                rtol=1e-12,
                atol=1e-12,
            )
            metric_row = {
                "metric": metric,
                "candidate_value": candidate_value,
                "baseline_value": baseline_value,
                "delta": float(candidate_value - baseline_value),
                "improvement": improvement,
                "material_improvement_threshold": threshold,
                "operator": _model_value_operator(metric),
                "materially_better": bool(materially_better),
            }
            metric_rows.append(metric_row)

        if not metric_rows:
            continue

        material_metrics = [row for row in metric_rows if row["materially_better"]]
        comparison_status = "pass" if material_metrics else "warning"
        if comparison_status == "warning":
            warning_baselines.append(baseline_scenario)
            baseline_best_metric = max(
                metric_rows,
                key=lambda row: float(row.get("improvement", float("-inf"))),
            )
            if best_warning_metric is None or float(
                baseline_best_metric.get("improvement", float("-inf"))
            ) > float(best_warning_metric.get("improvement", float("-inf"))):
                best_warning_metric = {
                    **baseline_best_metric,
                    "baseline_scenario": baseline_scenario,
                }
        comparisons.append(
            {
                "candidate": candidate_id,
                "candidate_scenario": candidate_scenario,
                "baseline": baseline_scenario,
                "baseline_scenario": baseline_scenario,
                "baseline_role": _model_value_baseline_role(baseline_scenario),
                "status": comparison_status,
                "metric_count": len(metric_rows),
                "material_improvement_count": len(material_metrics),
                "similar_metric_count": len(metric_rows) - len(material_metrics),
                "metrics": metric_rows,
            }
        )

    if not comparisons:
        return {
            "status": "not_evaluable",
            "reason": "proxy/simple-price ablation baselines are unavailable for model-value comparison",
            "reason_code": "model_value_baselines_unavailable",
            "affects_strategy": False,
            "affects_insufficient_data": False,
            "candidate": candidate_id,
            "candidate_scenario": candidate_scenario,
            "baseline_scenarios": list(
                _model_value_baseline_scenarios(config_payload, candidate_scenario)
            ),
            "comparisons": [],
        }

    if warning_baselines:
        best_metric = best_warning_metric or {}
        reason = (
            "model-value warning: full model has no material improvement versus "
            f"proxy/simple-price baseline(s): {', '.join(warning_baselines)}"
        )
        gate = {
            "status": "warning",
            "reason": reason,
            "reason_code": MODEL_VALUE_WARNING_CODE,
            "candidate": candidate_id,
            "candidate_scenario": candidate_scenario,
            "baseline_scenarios": [
                comparison["baseline_scenario"] for comparison in comparisons
            ],
            "warning_baselines": warning_baselines,
            "comparisons": comparisons,
            "material_improvement_thresholds": dict(
                _model_comparison_outperformance_thresholds(config_payload)
            ),
            "reason_metadata": {
                "code": MODEL_VALUE_WARNING_CODE,
                "metric": best_metric.get("metric", "model_value_delta"),
                "value": best_metric.get("improvement"),
                "threshold": best_metric.get("material_improvement_threshold"),
                "operator": best_metric.get("operator", ">"),
            },
        }
        gate["structured_warning"] = _model_value_structured_warning(gate)
        return gate

    return {
        "status": "pass",
        "reason": "full model shows material value lift versus proxy/simple-price baselines",
        "reason_code": None,
        "candidate": candidate_id,
        "candidate_scenario": candidate_scenario,
        "baseline_scenarios": [
            comparison["baseline_scenario"] for comparison in comparisons
        ],
        "warning_baselines": [],
        "comparisons": comparisons,
        "material_improvement_thresholds": dict(
            _model_comparison_outperformance_thresholds(config_payload)
        ),
    }


def _model_value_baseline_scenarios(
    config_payload: Mapping[str, Any],
    candidate_scenario: str,
) -> tuple[str, ...]:
    configured_baseline_id = str(config_payload.get("baseline_candidate_id") or "")
    configured_baseline_scenario = (
        _model_candidate_ablation_scenario_id(config_payload, configured_baseline_id)
        if configured_baseline_id
        else NO_MODEL_PROXY_ABLATION_SCENARIO
    )
    return tuple(
        scenario
        for scenario in _dedupe_strings_preserving_order(
            (
                configured_baseline_scenario,
                *MODEL_VALUE_COMPARISON_BASELINE_SCENARIOS,
            )
        )
        if scenario and scenario != candidate_scenario
    )


def _dedupe_strings_preserving_order(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return tuple(output)


def _duplicates(values: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        normalized = str(value)
        if normalized in seen and normalized not in duplicates:
            duplicates.append(normalized)
        seen.add(normalized)
    return duplicates


def _model_value_metric_improvement(
    metric: str,
    candidate_value: float,
    baseline_value: float,
) -> float:
    if metric in MODEL_COMPARISON_LOWER_IS_BETTER_METRICS:
        return float(baseline_value - candidate_value)
    return float(candidate_value - baseline_value)


def _model_value_operator(metric: str) -> str:
    if metric in MODEL_COMPARISON_LOWER_IS_BETTER_METRICS:
        return "baseline - candidate > threshold"
    return "candidate - baseline > threshold"


def _model_value_baseline_role(scenario: str) -> str:
    if scenario == NO_MODEL_PROXY_ABLATION_SCENARIO:
        return "proxy_or_no_model_proxy"
    if scenario == "price_only":
        return "simple_price_features"
    return "ablation_baseline"


def _model_value_structured_warning(gate_result: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _mapping_or_empty(gate_result.get("reason_metadata"))
    return {
        "code": MODEL_VALUE_WARNING_CODE,
        "severity": "warning",
        "gate": "model_value",
        "combined_gate": "model_value",
        "metric": metadata.get("metric", "model_value_delta"),
        "value": metadata.get("value"),
        "threshold": metadata.get("threshold"),
        "operator": metadata.get("operator", ">"),
        "reason": gate_result.get("reason"),
        "message": gate_result.get("reason"),
        "candidate": gate_result.get("candidate"),
        "candidate_scenario": gate_result.get("candidate_scenario"),
        "comparison_baselines": list(gate_result.get("warning_baselines", []) or []),
    }


def _gate_decision_target_column(
    predictions: pd.DataFrame,
    configured_target_column: str,
    required_horizon: int,
) -> str:
    required_target_column = _target_column_for_horizon(required_horizon)
    configured_horizon = _horizon_from_target(configured_target_column)
    if required_target_column not in predictions.columns and configured_target_column in predictions.columns:
        return configured_target_column
    if _is_diagnostic_only_horizon(configured_horizon):
        return required_target_column
    if configured_target_column in predictions.columns:
        return configured_target_column
    return required_target_column


def _horizon_validation_metrics(
    predictions: pd.DataFrame,
    thresholds: ValidationGateThresholds,
    decision_target_column: str,
    required_horizon: int,
    *,
    horizons: Iterable[str],
) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for horizon_key in horizons:
        horizon = _horizon_from_label(horizon_key)
        if horizon is None:
            continue
        target_column = _target_column_for_horizon(horizon)
        rank_ic = _rank_ic_metrics_for_horizon(predictions, target_column, horizon)
        rank_gate = _evaluate_rank_ic(rank_ic, thresholds)
        affects_pass_fail = target_column == decision_target_column
        output_label = _horizon_output_label(horizon, required_horizon)
        role = _horizon_output_role(horizon, affects_pass_fail)
        if _is_diagnostic_only_horizon(horizon):
            affects_pass_fail = False
        status = str(rank_gate["status"])
        insufficient_data = status == "insufficient_data" or bool(rank_ic.get("insufficient_data"))
        positive_fold_ratio = rank_ic.get("positive_fold_ratio")
        metrics[horizon_key] = {
            "horizon": horizon,
            "label": output_label,
            "target_column": target_column,
            "role": role,
            "status": status,
            "required_validation_horizon": horizon == required_horizon,
            "affects_pass_fail": affects_pass_fail,
            "insufficient_data": insufficient_data,
            "insufficient_data_status": "insufficient_data" if insufficient_data else None,
            "insufficient_data_reason": (
                str(rank_gate.get("reason", ""))
                if insufficient_data
                else None
            ),
            "insufficient_data_code": rank_gate.get("skip_code") if insufficient_data else None,
            "rank_ic_status": rank_gate["status"],
            "rank_ic_reason": rank_gate["reason"],
            "positive_fold_ratio_threshold": thresholds.min_positive_fold_ratio,
            "positive_fold_ratio_passed": _metric_threshold_passed(
                positive_fold_ratio,
                thresholds.min_positive_fold_ratio,
            ),
            **rank_ic,
        }
    return metrics


def _diagnostic_horizon_gate_result(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {}
    rank_ic_status = str(row.get("rank_ic_status", "not_evaluable"))
    horizon = row.get("horizon")
    horizon_label = f"{horizon}d" if horizon is not None else "diagnostic"
    return {
        "status": "diagnostic",
        "label": row.get("label", "diagnostic"),
        "rank_ic_status": rank_ic_status,
        "diagnostic_status": rank_ic_status,
        "reason": (
            f"{horizon_label} rank IC is diagnostic and ignored for Stage 1 pass/fail: "
            f"{row.get('rank_ic_reason', '')}"
        ),
        "affects_strategy": False,
        "affects_system": False,
        "affects_pass_fail": False,
        "target_column": row.get("target_column"),
        "horizon": row.get("horizon"),
        "mean_rank_ic": row.get("mean_rank_ic"),
        "positive_fold_ratio": row.get("positive_fold_ratio"),
        "oos_rank_ic": row.get("oos_rank_ic"),
    }


def _evaluate_leakage(
    validation_summary: pd.DataFrame,
    gap_periods: int,
    embargo_periods: int,
    target_horizon: int,
    *,
    purge_embargo_application: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not validation_summary.empty and {"train_end", "test_start"}.issubset(validation_summary.columns):
        train_end = pd.to_datetime(validation_summary["train_end"], errors="coerce")
        test_start = pd.to_datetime(validation_summary["test_start"], errors="coerce")
        chronology_violations = int((train_end >= test_start).fillna(False).sum())
        if chronology_violations:
            reasons.append(f"train/test chronology violation count={chronology_violations}")
    if gap_periods < target_horizon:
        reasons.append(f"gap_periods={gap_periods} is below target horizon={target_horizon}")
    if embargo_periods < target_horizon:
        reasons.append(f"embargo_periods={embargo_periods} is below target horizon={target_horizon}")
    application = purge_embargo_application or {}
    if application.get("status") == "hard_fail":
        reasons.extend(str(reason) for reason in application.get("reasons", []))
    status = "hard_fail" if reasons else "pass"
    return {
        "status": status,
        "reason": "; ".join(reasons) if reasons else "no leakage guard violation",
        "reasons": reasons,
        "purge_embargo_application": dict(application),
    }


def _purge_embargo_application_evidence(
    validation_summary: pd.DataFrame,
    *,
    gap_periods: int,
    embargo_periods: int,
    target_horizon: int,
) -> dict[str, Any]:
    fold_rows = _walk_forward_fold_rows(validation_summary)
    reasons: list[str] = []
    configured_purge_ok = int(gap_periods) >= int(target_horizon)
    configured_embargo_ok = int(embargo_periods) >= int(target_horizon)
    if not configured_purge_ok:
        reasons.append(f"configured purge gap={gap_periods} is below target horizon={target_horizon}")
    if not configured_embargo_ok:
        reasons.append(f"configured embargo={embargo_periods} is below target horizon={target_horizon}")

    fold_count = int(len(fold_rows))
    fold_evidence: list[dict[str, Any]] = []
    purge_date_counts = _numeric_column_or_empty(fold_rows, "purged_date_count")
    embargo_period_values = _numeric_column_or_empty(fold_rows, "embargo_periods")
    embargo_date_counts = _numeric_column_or_empty(fold_rows, "embargoed_date_count")
    purge_applied_flags = _boolean_column_or_empty(fold_rows, "purge_applied")
    embargo_applied_flags = _boolean_column_or_empty(fold_rows, "embargo_applied")

    if fold_count and "purged_date_count" in fold_rows:
        insufficient_purge = purge_date_counts < int(target_horizon)
        if bool(insufficient_purge.any()):
            failed_folds = fold_rows.loc[insufficient_purge, "fold"].tolist()
            reasons.append(
                "one or more folds applied purge windows below target horizon: "
                f"folds={failed_folds}"
            )
    if fold_count and "embargo_periods" in fold_rows:
        insufficient_embargo = embargo_period_values < int(target_horizon)
        if bool(insufficient_embargo.any()):
            failed_folds = fold_rows.loc[insufficient_embargo, "fold"].tolist()
            reasons.append(
                "one or more folds applied embargo settings below target horizon: "
                f"folds={failed_folds}"
            )
    if fold_count and "purge_applied" in fold_rows and not bool(purge_applied_flags.all()):
        failed_folds = fold_rows.loc[~purge_applied_flags, "fold"].tolist()
        reasons.append(f"one or more folds did not apply purge windows: folds={failed_folds}")
    if fold_count and "embargo_applied" in fold_rows and not bool(embargo_applied_flags.all()):
        failed_folds = fold_rows.loc[~embargo_applied_flags, "fold"].tolist()
        reasons.append(f"one or more folds did not apply embargo settings: folds={failed_folds}")

    for _, row in fold_rows.iterrows():
        fold_evidence.append(
            {
                "fold": _json_safe(row.get("fold")),
                "fold_type": _json_safe(row.get("fold_type")),
                "is_oos": _bool_or_false(row.get("is_oos", False)),
                "train_start": _json_safe(row.get("train_start")),
                "train_end": _json_safe(row.get("train_end")),
                "validation_start": _json_safe(row.get("validation_start")),
                "validation_end": _json_safe(row.get("validation_end")),
                "test_start": _json_safe(row.get("test_start")),
                "test_end": _json_safe(row.get("test_end")),
                "oos_test_start": _json_safe(row.get("oos_test_start")),
                "oos_test_end": _json_safe(row.get("oos_test_end")),
                "train_periods": _json_safe(row.get("train_periods")),
                "validation_periods": _json_safe(row.get("validation_periods")),
                "test_periods": _json_safe(row.get("test_periods")),
                "purge_periods": _json_safe(row.get("purge_periods", gap_periods)),
                "purge_gap_periods": _json_safe(row.get("purge_gap_periods", gap_periods)),
                "purged_date_count": _json_safe(row.get("purged_date_count")),
                "purge_start": _json_safe(row.get("purge_start")),
                "purge_end": _json_safe(row.get("purge_end")),
                "purge_applied": _json_safe(row.get("purge_applied")),
                "embargo_periods": _json_safe(row.get("embargo_periods", embargo_periods)),
                "embargoed_date_count": _json_safe(row.get("embargoed_date_count")),
                "embargo_start": _json_safe(row.get("embargo_start")),
                "embargo_end": _json_safe(row.get("embargo_end")),
                "embargo_applied": _json_safe(row.get("embargo_applied")),
            }
        )

    status = "hard_fail" if reasons else ("not_evaluable" if fold_count == 0 else "pass")
    if status == "not_evaluable":
        reason = "no walk-forward fold rows available for purge/embargo application evidence"
    elif reasons:
        reason = "; ".join(reasons)
    else:
        reason = "purge and embargo application evidence is horizon-consistent"
    return {
        "status": status,
        "reason": reason,
        "reasons": reasons,
        "target_horizon": int(target_horizon),
        "configured_purge_gap_periods": int(gap_periods),
        "configured_embargo_periods": int(embargo_periods),
        "configured_purge_horizon_consistent": configured_purge_ok,
        "configured_embargo_horizon_consistent": configured_embargo_ok,
        "fold_count": fold_count,
        "min_purged_date_count": _series_min_or_none(purge_date_counts),
        "min_embargo_periods": _series_min_or_none(embargo_period_values),
        "min_embargoed_date_count": _series_min_or_none(embargo_date_counts),
        "all_folds_purge_applied": _all_true_or_none(purge_applied_flags),
        "all_folds_embargo_applied": _all_true_or_none(embargo_applied_flags),
        "folds": fold_evidence,
    }


def _numeric_column_or_empty(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame.empty or column not in frame:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _boolean_column_or_empty(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame.empty or column not in frame:
        return pd.Series(dtype=bool)
    return frame[column].fillna(False).astype(bool)


def _series_min_or_none(series: pd.Series) -> float | None:
    finite = pd.to_numeric(series, errors="coerce").dropna()
    if finite.empty:
        return None
    return float(finite.min())


def _all_true_or_none(series: pd.Series) -> bool | None:
    if series.empty:
        return None
    return bool(series.fillna(False).astype(bool).all())


def _bool_or_false(value: object) -> bool:
    if value is pd.NA:
        return False
    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


def _evaluate_walk_forward(
    validation_summary: pd.DataFrame,
    thresholds: ValidationGateThresholds,
    min_train_observations: int,
    *,
    target_horizon: int,
) -> dict[str, Any]:
    if validation_summary.empty:
        return {
            "status": "not_evaluable",
            "reason": "walk-forward summary is empty",
            "reasons": ["walk-forward summary is empty"],
            "fold_count": 0,
            "oos_fold_count": 0,
            "required_min_folds": int(thresholds.min_folds),
            "required_min_oos_folds": int(thresholds.min_oos_folds),
        }

    structured_skip = _structured_insufficient_walk_forward_result(validation_summary, thresholds)
    fold_rows = _walk_forward_fold_rows(validation_summary)
    if structured_skip and fold_rows.empty:
        return structured_skip

    data_reasons: list[str] = []
    hard_reasons: list[str] = []
    fold_count = _count_validation_folds(fold_rows)
    oos_fold_count = _count_oos_folds(fold_rows)
    required_min_oos_folds = (
        int(thresholds.min_oos_folds)
        if int(target_horizon) == int(thresholds.required_validation_horizon)
        else 1
    )
    if fold_count < thresholds.min_folds:
        data_reasons.append(f"fold_count={fold_count} is below required={thresholds.min_folds}")
    if "is_oos" not in fold_rows or oos_fold_count == 0:
        data_reasons.append("last OOS fold is missing")
    elif oos_fold_count < required_min_oos_folds:
        reason = f"oos_fold_count={oos_fold_count} is below required={required_min_oos_folds}"
        if fold_count <= thresholds.min_folds:
            data_reasons.append(reason)
        else:
            hard_reasons.append(reason)
    if {"train_end", "test_start"}.issubset(fold_rows.columns):
        train_end = pd.to_datetime(fold_rows["train_end"], errors="coerce")
        test_start = pd.to_datetime(fold_rows["test_start"], errors="coerce")
        if bool((train_end >= test_start).fillna(False).any()):
            hard_reasons.append("train_end must be earlier than test_start for every fold")
    if min_train_observations and "train_observations" in fold_rows:
        too_small = fold_rows["train_observations"].fillna(0) < min_train_observations
        if bool(too_small.any()):
            data_reasons.append("one or more folds have too few train observations")
    if "labeled_test_observations" in fold_rows:
        if bool((fold_rows["labeled_test_observations"].fillna(0) <= 0).any()):
            data_reasons.append("one or more folds have no labeled test observations")
    if hard_reasons:
        return {
            "status": "hard_fail",
            "reason": "; ".join(hard_reasons),
            "reasons": hard_reasons,
            "fold_count": fold_count,
            "oos_fold_count": oos_fold_count,
            "required_min_folds": int(thresholds.min_folds),
            "required_min_oos_folds": required_min_oos_folds,
        }
    if data_reasons:
        return {
            "status": "insufficient_data",
            "reason": "; ".join(data_reasons),
            "reasons": data_reasons,
            "fold_count": fold_count,
            "oos_fold_count": oos_fold_count,
            "required_min_folds": int(thresholds.min_folds),
            "required_min_oos_folds": required_min_oos_folds,
        }
    return {
        "status": "pass",
        "reason": "walk-forward/OOS structure is valid",
        "reasons": [],
        "fold_count": fold_count,
        "oos_fold_count": oos_fold_count,
        "required_min_folds": int(thresholds.min_folds),
        "required_min_oos_folds": required_min_oos_folds,
    }


def _structured_insufficient_walk_forward_result(
    validation_summary: pd.DataFrame,
    thresholds: ValidationGateThresholds,
) -> dict[str, Any]:
    if "validation_status" not in validation_summary:
        return {}
    statuses = (
        validation_summary["validation_status"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
    )
    if statuses.empty or not statuses.isin(INSUFFICIENT_DATA_GATE_STATUSES).any():
        return {}

    row = validation_summary.iloc[0].to_dict()
    status = "insufficient_data"
    reason = str(row.get("reason") or row.get("validation_reason") or "walk-forward skipped")
    reasons = [reason]
    return {
        "status": status,
        "reason": reason,
        "reasons": reasons,
        "skip_status": row.get("skip_status", "skipped"),
        "skip_code": row.get("skip_code"),
        "validation_status": row.get("validation_status"),
        "fold_count": int(_finite_or_none(row.get("fold_count")) or 0),
        "oos_fold_count": int(_finite_or_none(row.get("oos_fold_count")) or 0),
        "candidate_fold_count": int(_finite_or_none(row.get("candidate_fold_count")) or 0),
        "candidate_date_count": int(_finite_or_none(row.get("candidate_date_count")) or 0),
        "labeled_date_count": int(_finite_or_none(row.get("labeled_date_count")) or 0),
        "labeled_observation_count": int(
            _finite_or_none(row.get("labeled_observation_count")) or 0
        ),
        "required_min_date_count": row.get("required_min_date_count"),
        "required_min_folds": row.get("required_min_folds", int(thresholds.min_folds)),
        "required_min_oos_folds": row.get(
            "required_min_oos_folds",
            int(thresholds.min_oos_folds),
        ),
        "min_train_observations": row.get("min_train_observations"),
    }


def _walk_forward_fold_rows(validation_summary: pd.DataFrame) -> pd.DataFrame:
    if validation_summary.empty:
        return validation_summary
    mask = pd.Series(True, index=validation_summary.index)
    if "skip_status" in validation_summary:
        skip_status = validation_summary["skip_status"].fillna("").astype(str).str.lower()
        mask &= skip_status != "skipped"
    if "validation_status" in validation_summary:
        validation_status = validation_summary["validation_status"].fillna("").astype(str).str.lower()
        mask &= ~validation_status.isin({"insufficient_data", "skipped"})
    if "fold" in validation_summary:
        mask &= validation_summary["fold"].notna()
    return validation_summary.loc[mask].copy()


def _rank_ic_metrics(predictions: pd.DataFrame, target_column: str) -> dict[str, Any]:
    if predictions.empty or not {"date", "expected_return", target_column}.issubset(predictions.columns):
        return {"mean_rank_ic": None, "positive_fold_ratio": None, "oos_rank_ic": None, "rank_ic_count": 0}
    frame = predictions.dropna(subset=["date", "expected_return", target_column]).copy()
    if frame.empty:
        return {"mean_rank_ic": None, "positive_fold_ratio": None, "oos_rank_ic": None, "rank_ic_count": 0}
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for date_value, day in frame.groupby("date"):
        rank_ic = _spearman(day["expected_return"], day[target_column])
        if rank_ic is None:
            continue
        rows.append(
            {
                "date": date_value,
                "fold": day["fold"].iloc[0] if "fold" in day else None,
                "is_oos": bool(day["is_oos"].fillna(False).any()) if "is_oos" in day else False,
                "rank_ic": rank_ic,
            }
        )
    rank_frame = pd.DataFrame(rows)
    if rank_frame.empty:
        return {"mean_rank_ic": None, "positive_fold_ratio": None, "oos_rank_ic": None, "rank_ic_count": 0}
    if "fold" in rank_frame and rank_frame["fold"].notna().any():
        fold_ics = rank_frame.groupby("fold")["rank_ic"].mean()
        positive_ratio = float((fold_ics > 0).mean())
    else:
        positive_ratio = float((rank_frame["rank_ic"] > 0).mean())
    oos = rank_frame[rank_frame["is_oos"]]
    return {
        "mean_rank_ic": float(rank_frame["rank_ic"].mean()),
        "positive_fold_ratio": positive_ratio,
        "oos_rank_ic": float(oos["rank_ic"].mean()) if not oos.empty else None,
        "rank_ic_count": int(len(rank_frame)),
    }


def _rank_ic_metrics_for_horizon(
    predictions: pd.DataFrame,
    target_column: str,
    horizon: int,
) -> dict[str, Any]:
    support = _windowed_horizon_observation_support(predictions, horizon)
    if not support["supported"]:
        return _insufficient_rank_ic_metrics(
            reason=(
                "rank IC is not evaluable because the dataset has "
                f"max_observations_per_ticker={support.get('max_observations_per_ticker')} "
                "but requires "
                f"{support.get('required_min_observations')} observations for the {horizon}d window"
            ),
            code="insufficient_window_observations",
            support=support,
        )

    metrics = _rank_ic_metrics(predictions, target_column)
    metrics = {**metrics, **support}
    if (
        metrics.get("mean_rank_ic") is None
        or metrics.get("positive_fold_ratio") is None
        or metrics.get("oos_rank_ic") is None
    ):
        reason, code = _rank_ic_insufficient_data_reason(predictions, target_column)
        return {
            **metrics,
            "insufficient_data": True,
            "insufficient_data_status": "insufficient_data",
            "insufficient_data_reason": reason,
            "insufficient_data_code": code,
        }
    return {
        **metrics,
        "insufficient_data": False,
        "insufficient_data_status": None,
        "insufficient_data_reason": None,
        "insufficient_data_code": None,
    }


def calculate_top_decile_20d_excess_return(
    predictions: pd.DataFrame,
    *,
    universe: Iterable[str] | None = None,
    benchmark_ticker: str = "SPY",
    expected_return_column: str = "expected_return",
    target_column: str = "forward_return_20",
) -> dict[str, Any]:
    """Calculate report-only top-decile 20d excess return from dated predictions.

    The decile membership is formed only from same-day prediction scores. Realized
    20d labels are used after the membership is fixed, and timestamp cutoffs are
    checked so unavailable features or predictions cannot enter the metric.
    """

    required_columns = {"date", "ticker", expected_return_column, target_column}
    missing_columns = sorted(required_columns.difference(predictions.columns))
    if predictions.empty:
        return _top_decile_20d_not_evaluable(
            "predictions are empty",
            "empty_predictions",
        )
    if missing_columns:
        return _top_decile_20d_not_evaluable(
            f"required columns are missing: {', '.join(missing_columns)}",
            "missing_top_decile_columns",
        )

    frame = predictions.copy()
    _validate_top_decile_20d_cutoffs(frame)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["ticker"] = frame["ticker"].fillna("").astype(str).str.strip().str.upper()
    frame[expected_return_column] = pd.to_numeric(
        frame[expected_return_column], errors="coerce"
    )
    frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
    frame = frame.dropna(subset=["date", "ticker", expected_return_column, target_column])
    frame = frame[frame["ticker"].ne("")]
    if frame.empty:
        return _top_decile_20d_not_evaluable(
            "no labeled prediction rows are available",
            "insufficient_labeled_predictions",
        )

    universe_set = {
        str(ticker).strip().upper()
        for ticker in (universe or [])
        if str(ticker).strip()
    }
    if universe_set:
        frame = frame[frame["ticker"].isin(universe_set)]
    else:
        benchmark = str(benchmark_ticker).strip().upper()
        if benchmark:
            frame = frame[frame["ticker"].ne(benchmark)]
    if frame.empty:
        return _top_decile_20d_not_evaluable(
            "no prediction rows remain after universe filtering",
            "empty_universe_predictions",
        )

    sample_scope = "all_labeled_predictions"
    if "is_oos" in frame:
        oos_mask = frame["is_oos"].fillna(False).astype(bool)
        if oos_mask.any():
            frame = frame[oos_mask].copy()
            sample_scope = "oos_labeled_predictions"

    rows: list[dict[str, Any]] = []
    for date_value, day in frame.groupby("date", sort=True):
        day = day.sort_values(
            [expected_return_column, "ticker"],
            ascending=[False, True],
        )
        if len(day) < 2:
            continue
        top_count = max(1, int(np.ceil(len(day) * 0.10)))
        top = day.head(top_count)
        universe_return = float(day[target_column].mean())
        top_return = float(top[target_column].mean())
        rows.append(
            {
                "date": date_value,
                "universe_count": int(len(day)),
                "top_decile_count": int(len(top)),
                "top_decile_20d_return": top_return,
                "universe_20d_return": universe_return,
                "top_decile_20d_excess_return": top_return - universe_return,
            }
        )

    daily = pd.DataFrame(rows)
    if daily.empty:
        return _top_decile_20d_not_evaluable(
            "not enough cross-sectional observations to form daily top-decile excess returns",
            "insufficient_cross_section",
        )

    value = float(daily["top_decile_20d_excess_return"].mean())
    return {
        "metric": "top_decile_20d_excess_return",
        "target_column": target_column,
        "selection_column": expected_return_column,
        "return_basis": "mean_daily_top_decile_forward_return_20_minus_universe_mean",
        "sample_scope": sample_scope,
        "status": "report_only",
        "reason": "report-only diagnostic; not used for scoring, action, ranking, thresholding, or gating",
        "report_only": True,
        "decision_use": "none",
        "top_decile_20d_excess_return": value,
        "date_count": int(len(daily)),
        "observation_count": int(daily["universe_count"].sum()),
        "top_decile_observation_count": int(daily["top_decile_count"].sum()),
        "mean_top_decile_20d_return": float(daily["top_decile_20d_return"].mean()),
        "mean_universe_20d_return": float(daily["universe_20d_return"].mean()),
    }


def _top_decile_20d_not_evaluable(
    reason: str,
    code: str,
) -> dict[str, Any]:
    return {
        "metric": "top_decile_20d_excess_return",
        "target_column": "forward_return_20",
        "selection_column": "expected_return",
        "return_basis": "mean_daily_top_decile_forward_return_20_minus_universe_mean",
        "sample_scope": "not_evaluable",
        "status": "not_evaluable",
        "reason": reason,
        "reason_code": code,
        "report_only": True,
        "decision_use": "none",
        "top_decile_20d_excess_return": None,
        "date_count": 0,
        "observation_count": 0,
        "top_decile_observation_count": 0,
        "mean_top_decile_20d_return": None,
        "mean_universe_20d_return": None,
    }


def _validate_top_decile_20d_cutoffs(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    sample_timestamp = date_end_utc(frame["date"])
    cutoff_columns = [
        column
        for column in frame.columns
        if column == "availability_timestamp"
        or str(column).endswith("_availability_timestamp")
        or column in {"prediction_date", "prediction_timestamp", "model_prediction_timestamp"}
    ]
    for column in cutoff_columns:
        cutoff = timestamp_utc(frame[column])
        violation = cutoff.notna() & (cutoff > sample_timestamp)
        if violation.any():
            first_index = int(np.flatnonzero(violation.to_numpy())[0])
            raise ValueError(
                "top_decile_20d_excess_return input contains unavailable data: "
                f"{column} is later than signal date {frame.loc[first_index, 'date']}"
            )

    signal_dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    for column in ("return_label_date", "holding_end_date", "return_date"):
        if column not in frame:
            continue
        label_dates = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
        violation = label_dates.notna() & signal_dates.notna() & (label_dates <= signal_dates)
        if violation.any():
            first_index = int(np.flatnonzero(violation.to_numpy())[0])
            raise ValueError(
                "top_decile_20d_excess_return requires forward labels after signal date: "
                f"{column}={frame.loc[first_index, column]} is not after "
                f"{frame.loc[first_index, 'date']}"
            )


def _report_only_research_metric_rows(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    top_decile = _mapping_or_empty(evidence.get("top_decile_20d_excess_return"))
    if not top_decile:
        return []
    status = top_decile.get("status")
    return [
        {
            "metric": top_decile.get("metric", "top_decile_20d_excess_return"),
            "status": status,
            "value": top_decile.get("top_decile_20d_excess_return"),
            "target_column": top_decile.get("target_column", "forward_return_20"),
            "sample_scope": top_decile.get("sample_scope"),
            "report_only": top_decile.get("report_only", True),
            "decision_use": top_decile.get("decision_use", "none"),
            "reason": top_decile.get("reason", ""),
        }
    ]


def _insufficient_rank_ic_metrics(
    *,
    reason: str,
    code: str,
    support: dict[str, Any],
) -> dict[str, Any]:
    return {
        "mean_rank_ic": None,
        "positive_fold_ratio": None,
        "oos_rank_ic": None,
        "rank_ic_count": 0,
        **support,
        "insufficient_data": True,
        "insufficient_data_status": "insufficient_data",
        "insufficient_data_reason": reason,
        "insufficient_data_code": code,
    }


def _rank_ic_insufficient_data_reason(
    predictions: pd.DataFrame,
    target_column: str,
) -> tuple[str, str]:
    required_columns = {"date", "expected_return", target_column}
    if predictions.empty:
        return "rank IC is not evaluable because predictions are empty", "empty_predictions"
    missing_columns = sorted(required_columns.difference(predictions.columns))
    if missing_columns:
        return (
            f"rank IC is not evaluable because required columns are missing: {', '.join(missing_columns)}",
            "missing_rank_ic_columns",
        )
    frame = predictions.dropna(subset=["date", "expected_return", target_column])
    if frame.empty:
        return (
            "rank IC is not evaluable because no labeled prediction rows are available",
            "insufficient_labeled_predictions",
        )
    return (
        "rank IC is not evaluable because labeled predictions do not contain enough cross-sectional/OOS variation",
        "insufficient_rank_ic_observations",
    )


def _windowed_horizon_observation_support(
    predictions: pd.DataFrame,
    horizon: int,
) -> dict[str, Any]:
    required = MIN_WINDOWED_HORIZON_OBSERVATIONS.get(int(horizon))
    if required is None:
        return {
            "minimum_observation_guard": False,
            "required_min_observations": None,
            "max_observations_per_ticker": None,
            "supported": True,
        }

    observed = _max_unique_dates_per_ticker(predictions)
    return {
        "minimum_observation_guard": True,
        "required_min_observations": required,
        "max_observations_per_ticker": observed,
        "supported": observed >= required,
    }


def _max_unique_dates_per_ticker(frame: pd.DataFrame) -> int:
    if frame.empty or "date" not in frame:
        return 0
    dates = pd.to_datetime(frame["date"], errors="coerce")
    valid = frame.loc[dates.notna()].copy()
    if valid.empty:
        return 0
    valid["date"] = dates[dates.notna()].dt.normalize()
    if "ticker" not in valid:
        return int(valid["date"].nunique())
    per_ticker = valid.groupby("ticker", dropna=False)["date"].nunique()
    if per_ticker.empty:
        return 0
    return int(per_ticker.max())


def _evaluate_rank_ic(metrics: dict[str, Any], thresholds: ValidationGateThresholds) -> dict[str, Any]:
    if metrics.get("insufficient_data"):
        return {
            "status": "insufficient_data",
            "reason": str(metrics.get("insufficient_data_reason") or "rank IC has insufficient data"),
            "skip_code": metrics.get("insufficient_data_code"),
        }
    mean_rank_ic = metrics.get("mean_rank_ic")
    positive_fold_ratio = metrics.get("positive_fold_ratio")
    oos_rank_ic = metrics.get("oos_rank_ic")
    if mean_rank_ic is None or positive_fold_ratio is None or oos_rank_ic is None:
        if metrics.get("minimum_observation_guard") and not metrics.get("supported", True):
            return {
                "status": "not_evaluable",
                "reason": (
                    "rank IC is not evaluable because the dataset has "
                    f"max_observations_per_ticker={metrics.get('max_observations_per_ticker')} "
                    "but requires "
                    f"{metrics.get('required_min_observations')} observations for the 20d window"
                ),
            }
        return {"status": "not_evaluable", "reason": "rank IC is not evaluable"}
    if mean_rank_ic <= 0:
        return {
            "status": "fail",
            "reason": f"mean_rank_ic={mean_rank_ic:.4f} is <= 0",
            "reason_metadata": {
                "code": "mean_rank_ic_non_positive",
                "metric": "mean_rank_ic",
                "value": mean_rank_ic,
                "threshold": 0.0,
                "operator": ">",
            },
        }
    if positive_fold_ratio < thresholds.min_positive_fold_ratio:
        return {
            "status": "fail",
            "reason": (
                f"positive_fold_ratio={positive_fold_ratio:.4f} is below "
                f"required={thresholds.min_positive_fold_ratio:.4f}"
            ),
            "reason_metadata": {
                "code": "positive_fold_ratio_below_minimum",
                "metric": "positive_fold_ratio",
                "value": positive_fold_ratio,
                "threshold": thresholds.min_positive_fold_ratio,
                "operator": ">=",
            },
        }
    if (
        mean_rank_ic >= thresholds.min_rank_ic
        and positive_fold_ratio >= thresholds.min_positive_fold_ratio
        and oos_rank_ic > 0
    ):
        return {"status": "pass", "reason": "rank IC thresholds passed"}
    return {"status": "warning", "reason": "rank IC is positive but below one or more pass thresholds"}


def _metric_threshold_passed(value: object, threshold: object) -> bool | None:
    numeric_value = _finite_or_none(value)
    numeric_threshold = _finite_or_none(threshold)
    if numeric_value is None or numeric_threshold is None:
        return None
    return numeric_value >= numeric_threshold


def _benchmark_results(
    predictions: pd.DataFrame,
    equity_curve: pd.DataFrame,
    strategy_metrics: object,
    benchmark_ticker: str,
    return_column: str = "forward_return_1",
    *,
    config: object | None = None,
    benchmark_return_series: pd.DataFrame | None = None,
    equal_weight_baseline_return_series: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    strategy_cagr = _metric(strategy_metrics, "cagr")
    strategy_sharpe = _metric(strategy_metrics, "sharpe")
    strategy_max_drawdown = _metric(strategy_metrics, "max_drawdown")
    strategy_cost_adjusted_return = _cost_adjustment_metrics(equity_curve)[
        "cost_adjusted_cumulative_return"
    ]
    baseline_cost_bps = _baseline_cost_assumption(config, equity_curve, "cost_bps")
    baseline_slippage_bps = _baseline_cost_assumption(config, equity_curve, "slippage_bps")
    evaluation_dates = _baseline_evaluation_dates(equity_curve, predictions)
    spy = _spy_baseline_metrics(
        predictions,
        equity_curve,
        benchmark_ticker,
        return_column,
        cost_bps=baseline_cost_bps,
        slippage_bps=baseline_slippage_bps,
        benchmark_return_series=benchmark_return_series,
        evaluation_dates=evaluation_dates,
    )
    equal_weight = _equal_weight_baseline_metrics(
        predictions,
        return_column,
        ticker_universe=_strategy_universe_tickers(config, predictions),
        evaluation_dates=evaluation_dates,
        cost_bps=baseline_cost_bps,
        slippage_bps=baseline_slippage_bps,
        baseline_return_series=equal_weight_baseline_return_series,
    )
    spy_alignment = _baseline_sample_alignment(
        benchmark_ticker,
        evaluation_dates,
        _market_benchmark_alignment_dates(
            predictions,
            equity_curve,
            benchmark_return_series,
            benchmark_ticker,
            return_column,
        ),
        strict=(
            benchmark_return_series is not None
            or (
                not equity_curve.empty
                and {"date", "benchmark_return"}.issubset(equity_curve.columns)
            )
        ),
    )
    equal_weight_alignment = _baseline_sample_alignment(
        EQUAL_WEIGHT_BASELINE_NAME,
        evaluation_dates,
        _equal_weight_alignment_dates(
            predictions,
            equal_weight_baseline_return_series,
            return_column,
            ticker_universe=_strategy_universe_tickers(config, predictions),
        ),
        strict=equal_weight_baseline_return_series is not None,
    )
    return_horizon = _horizon_from_target(return_column) or 1
    equal_weight_evaluation_observations = equal_weight["observations"]
    equal_weight_excess_return = _aligned_cost_adjusted_excess_return(
        equal_weight_alignment,
        strategy_cost_adjusted_return,
        equal_weight.get(
            "cost_adjusted_cumulative_return",
            equal_weight.get("cumulative_return"),
        ),
    )
    equal_weight_result = {
        "name": "equal_weight",
        "return_basis": "cost_adjusted_equal_weight_return",
        "return_column": return_column,
        "return_horizon": return_horizon,
        "evaluation_observations": equal_weight_evaluation_observations,
        "evaluation_start": equal_weight["evaluation_start"],
        "evaluation_end": equal_weight["evaluation_end"],
        "cagr": equal_weight["cagr"],
        "sharpe": equal_weight["sharpe"],
        "max_drawdown": equal_weight["max_drawdown"],
        "excess_return": equal_weight_excess_return,
        "excess_return_status": _baseline_excess_return_status(
            equal_weight_excess_return,
            equal_weight_evaluation_observations,
        ),
        "sample_alignment": equal_weight_alignment,
        "sample_alignment_status": equal_weight_alignment["status"],
        "cagr_excess_return": strategy_cagr - equal_weight["cagr"],
        "strategy_cost_adjusted_cumulative_return": strategy_cost_adjusted_return,
        "strategy_sharpe": strategy_sharpe,
        "strategy_max_drawdown": strategy_max_drawdown,
    }
    for key in (
        "universe_tickers",
        "expected_constituent_count",
        "min_constituent_count",
        "max_constituent_count",
        "incomplete_rebalance_count",
        "missing_return_count",
        "rebalance_count",
        "rebalance_assumption",
        "return_timing",
        "cumulative_return",
        "gross_cagr",
        "gross_cumulative_return",
        "cost_adjusted_cumulative_return",
        "average_daily_turnover",
        "transaction_cost_return",
        "slippage_cost_return",
        "total_cost_return",
        "cost_bps",
        "slippage_bps",
    ):
        equal_weight_result[key] = equal_weight.get(key)
    spy_evaluation_observations = spy["observations"]
    spy_excess_return = _aligned_cost_adjusted_excess_return(
        spy_alignment,
        strategy_cost_adjusted_return,
        spy.get("cost_adjusted_cumulative_return", spy.get("cumulative_return")),
    )
    spy_result = {
        "name": benchmark_ticker,
        "return_basis": "cost_adjusted_benchmark_return",
        "return_column": return_column,
        "return_horizon": return_horizon,
        "evaluation_observations": spy_evaluation_observations,
        "evaluation_start": spy["evaluation_start"],
        "evaluation_end": spy["evaluation_end"],
        "cagr": spy["cagr"],
        "sharpe": spy["sharpe"],
        "max_drawdown": spy["max_drawdown"],
        "cumulative_return": spy["cumulative_return"],
        "excess_return": spy_excess_return,
        "excess_return_status": _baseline_excess_return_status(
            spy_excess_return,
            spy_evaluation_observations,
        ),
        "sample_alignment": spy_alignment,
        "sample_alignment_status": spy_alignment["status"],
        "cagr_excess_return": strategy_cagr - spy["cagr"],
        "strategy_cost_adjusted_cumulative_return": strategy_cost_adjusted_return,
        "strategy_sharpe": strategy_sharpe,
        "strategy_max_drawdown": strategy_max_drawdown,
    }
    for key in (
        "gross_cagr",
        "gross_cumulative_return",
        "cost_adjusted_cumulative_return",
        "average_daily_turnover",
        "transaction_cost_return",
        "slippage_cost_return",
        "total_cost_return",
        "cost_bps",
        "slippage_bps",
        "rebalance_assumption",
        "return_timing",
    ):
        spy_result[key] = spy.get(key)
    return [spy_result, equal_weight_result]


def _aligned_cost_adjusted_excess_return(
    sample_alignment: Mapping[str, Any],
    strategy_cost_adjusted_return: object,
    baseline_cost_adjusted_return: object,
) -> float | None:
    if sample_alignment.get("status") == "hard_fail":
        return None
    return _cost_adjusted_excess_return(
        strategy_cost_adjusted_return,
        baseline_cost_adjusted_return,
    )


def _evaluate_baseline_sample_alignment(
    baseline_results: list[dict[str, Any]],
) -> dict[str, Any]:
    baselines = {
        str(result.get("name")): _mapping_or_empty(result.get("sample_alignment"))
        for result in baseline_results
        if str(result.get("name", "")).strip()
    }
    failed_baselines = [
        name
        for name, alignment in baselines.items()
        if alignment.get("status") == "hard_fail"
    ]
    not_evaluable_baselines = [
        name
        for name, alignment in baselines.items()
        if alignment.get("status") == "not_evaluable"
    ]
    candidate_dates: list[str] = []
    for alignment in baselines.values():
        dates = alignment.get("candidate_dates")
        if isinstance(dates, list):
            candidate_dates = [str(value) for value in dates]
            break

    base_result = {
        "affects_system": True,
        "affects_strategy": False,
        "affects_insufficient_data": False,
        "candidate_dates": candidate_dates,
        "baselines": baselines,
        "failed_baselines": failed_baselines,
        "not_evaluable_baselines": not_evaluable_baselines,
        "aligned_baselines": [
            name
            for name, alignment in baselines.items()
            if alignment.get("status") == "pass"
        ],
    }
    if failed_baselines:
        return {
            **base_result,
            "status": "hard_fail",
            "reason": (
                "baseline sample alignment failed for required baseline(s): "
                f"{', '.join(failed_baselines)}"
            ),
        }
    if not baselines or len(not_evaluable_baselines) == len(baselines):
        return {
            **base_result,
            "status": "not_evaluable",
            "reason": "baseline sample alignment is not evaluable",
        }
    return {
        **base_result,
        "status": "pass",
        "reason": "baseline samples align to candidate evaluation dates",
    }


def _baseline_sample_alignment(
    baseline_name: str,
    candidate_dates: object,
    baseline_dates: object,
    *,
    strict: bool,
) -> dict[str, Any]:
    candidate = _date_iso_values(_normalize_metric_dates(candidate_dates))
    baseline = _date_iso_values(_normalize_metric_dates(baseline_dates))
    candidate_set = set(candidate)
    baseline_set = set(baseline)
    missing_candidate_dates = [value for value in candidate if value not in baseline_set]
    extra_baseline_dates = [value for value in baseline if value not in candidate_set]
    aligned_dates = [value for value in candidate if value in baseline_set]

    base_result = {
        "baseline": str(baseline_name),
        "candidate_dates": candidate,
        "baseline_dates": baseline,
        "aligned_dates": aligned_dates,
        "missing_candidate_dates": missing_candidate_dates,
        "extra_baseline_dates": extra_baseline_dates,
        "candidate_sample_count": len(candidate),
        "baseline_sample_count": len(baseline),
        "aligned_sample_count": len(aligned_dates),
        "missing_candidate_sample_count": len(missing_candidate_dates),
        "extra_baseline_sample_count": len(extra_baseline_dates),
    }
    if not candidate:
        return {
            **base_result,
            "status": "not_evaluable",
            "reason": "candidate evaluation dates are unavailable",
        }
    if missing_candidate_dates and not strict:
        return {
            **base_result,
            "status": "not_evaluable",
            "reason": (
                "baseline sample dates do not cover candidate evaluation date(s): "
                f"{', '.join(missing_candidate_dates)}"
            ),
        }
    if strict and missing_candidate_dates:
        return {
            **base_result,
            "status": "hard_fail",
            "reason": (
                "baseline is missing required candidate evaluation date(s): "
                f"{', '.join(missing_candidate_dates)}"
            ),
        }
    if strict and not baseline:
        return {
            **base_result,
            "status": "hard_fail",
            "reason": "baseline has no samples for candidate evaluation dates",
        }
    if not baseline:
        return {
            **base_result,
            "status": "not_evaluable",
            "reason": "baseline sample dates are unavailable",
        }
    return {
        **base_result,
        "status": "pass",
        "reason": (
            "baseline samples align to candidate dates"
            if not extra_baseline_dates
            else "baseline samples align after ignoring dates outside candidate evaluation"
        ),
    }


def _market_benchmark_alignment_dates(
    predictions: pd.DataFrame,
    equity_curve: pd.DataFrame,
    benchmark_return_series: pd.DataFrame | None,
    benchmark_ticker: str,
    return_column: str,
) -> pd.Series:
    if benchmark_return_series is not None and not benchmark_return_series.empty:
        frame = benchmark_return_series.copy()
        if "date" not in frame or "benchmark_return" not in frame:
            return pd.Series(dtype="datetime64[ns]")
        frame = _filter_baseline_return_series_to_definition(frame, return_column)
        if "benchmark_ticker" in frame:
            frame["benchmark_ticker"] = frame["benchmark_ticker"].map(_normalize_ticker)
            frame = frame[frame["benchmark_ticker"] == _normalize_ticker(benchmark_ticker)]
        frame["benchmark_return"] = pd.to_numeric(frame["benchmark_return"], errors="coerce")
        return frame.loc[frame["benchmark_return"].notna(), "date"]

    if not equity_curve.empty and {"date", "benchmark_return"}.issubset(equity_curve.columns):
        returns = pd.to_numeric(equity_curve["benchmark_return"], errors="coerce")
        return equity_curve.loc[returns.notna(), "date"]

    if predictions.empty or not {"date", "ticker", return_column}.issubset(predictions.columns):
        return pd.Series(dtype="datetime64[ns]")
    tickers = predictions["ticker"].map(_normalize_ticker)
    returns = pd.to_numeric(predictions[return_column], errors="coerce")
    mask = (tickers == _normalize_ticker(benchmark_ticker)) & returns.notna()
    return predictions.loc[mask, "date"]


def _equal_weight_alignment_dates(
    predictions: pd.DataFrame,
    equal_weight_baseline_return_series: pd.DataFrame | None,
    return_column: str,
    *,
    ticker_universe: tuple[str, ...],
) -> pd.Series:
    if (
        equal_weight_baseline_return_series is not None
        and not equal_weight_baseline_return_series.empty
    ):
        frame = equal_weight_baseline_return_series.copy()
        if "date" not in frame or "equal_weight_return" not in frame:
            return pd.Series(dtype="datetime64[ns]")
        frame = _filter_baseline_return_series_to_definition(frame, return_column)
        returns = pd.to_numeric(frame["equal_weight_return"], errors="coerce")
        return frame.loc[returns.notna(), "date"]

    if predictions.empty or not {"date", "ticker", return_column}.issubset(predictions.columns):
        return pd.Series(dtype="datetime64[ns]")
    frame = predictions[["date", "ticker", return_column]].copy()
    frame["ticker"] = frame["ticker"].map(_normalize_ticker)
    universe = set(ticker_universe or _normalize_ticker_sequence(frame["ticker"]))
    if not universe:
        return pd.Series(dtype="datetime64[ns]")
    frame = frame[frame["ticker"].isin(universe)]
    frame[return_column] = pd.to_numeric(frame[return_column], errors="coerce")
    return frame.loc[frame[return_column].notna(), "date"]


def _date_iso_values(dates: pd.Series) -> list[str]:
    values = _normalize_metric_dates(dates)
    return [pd.Timestamp(value).date().isoformat() for value in values]


def _cost_adjusted_metric_comparison(
    equity_curve: pd.DataFrame,
    strategy_metrics: object,
    baseline_results: list[dict[str, Any]],
    cost_adjustment: dict[str, Any],
    *,
    config: object | None = None,
    default_return_column: str = "forward_return_1",
) -> list[dict[str, Any]]:
    strategy_cagr = _metric(strategy_metrics, "cagr")
    strategy_cost_adjusted_return = cost_adjustment["cost_adjusted_cumulative_return"]
    strategy_start, strategy_end = _date_bounds(_date_series(equity_curve))
    strategy_return_column = _strategy_return_column(
        equity_curve,
        config,
        default_return_column=default_return_column,
    )
    strategy_return_horizon = _horizon_from_target(strategy_return_column) or 1
    strategy_returns = _numeric_column(
        equity_curve,
        "cost_adjusted_return",
        "portfolio_return",
        "net_return",
    )
    rows: list[dict[str, Any]] = [
        {
            "name": "strategy",
            "role": "strategy",
            "return_basis": "cost_adjusted_return",
            "cagr": strategy_cagr,
            "sharpe": _metric(strategy_metrics, "sharpe"),
            "max_drawdown": _metric(strategy_metrics, "max_drawdown"),
            "cumulative_return": cost_adjustment["cost_adjusted_cumulative_return"],
            "gross_cumulative_return": cost_adjustment["gross_cumulative_return"],
            "cost_adjusted_cumulative_return": cost_adjustment[
                "cost_adjusted_cumulative_return"
            ],
            "average_daily_turnover": _metric(strategy_metrics, "turnover"),
            "transaction_cost_return": cost_adjustment["transaction_cost_return"],
            "slippage_cost_return": cost_adjustment["slippage_cost_return"],
            "total_cost_return": cost_adjustment["total_cost_return"],
            "excess_return": 0.0,
            "strategy_excess_return": 0.0,
            "cost_bps": _baseline_cost_assumption(config, equity_curve, "cost_bps"),
            "slippage_bps": _baseline_cost_assumption(config, equity_curve, "slippage_bps"),
            "evaluation_observations": int(len(strategy_returns)),
            "evaluation_start": strategy_start,
            "evaluation_end": strategy_end,
            "return_column": strategy_return_column,
            "return_horizon": strategy_return_horizon,
        }
    ]
    rows.extend(
        _baseline_metric_comparison_row(
            row,
            strategy_cost_adjusted_return,
            strategy_cagr,
        )
        for row in baseline_results
    )
    return rows


def _side_by_side_metric_comparison(
    cost_adjusted_metric_comparison: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered_names = [
        str(row.get("name", ""))
        for row in cost_adjusted_metric_comparison
        if str(row.get("name", "")).strip()
    ]
    if not ordered_names:
        return []
    comparison_by_name = {
        str(row.get("name", "")): row
        for row in cost_adjusted_metric_comparison
        if str(row.get("name", "")).strip()
    }

    rows: list[dict[str, Any]] = []
    for metric, label in SIDE_BY_SIDE_METRIC_FIELDS:
        row: dict[str, Any] = {
            "metric": metric,
            "metric_label": label,
        }
        for name in ordered_names:
            row[name] = comparison_by_name.get(name, {}).get(metric)
        rows.append(row)
    return rows


def _side_by_side_metric_columns(
    side_by_side_metric_comparison: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    columns: dict[str, dict[str, Any]] = {}
    for row in side_by_side_metric_comparison:
        metric = str(row.get("metric", "")).strip()
        if not metric:
            continue
        for name in _side_by_side_metric_entity_columns(side_by_side_metric_comparison):
            columns.setdefault(name, {})[metric] = row.get(name)
    return columns


def _side_by_side_metric_entity_columns(
    side_by_side_metric_comparison: list[dict[str, Any]],
) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in side_by_side_metric_comparison:
        for name in row:
            if name in {"metric", "metric_label"} or name in seen:
                continue
            seen.add(name)
            columns.append(name)
    return columns


def _baseline_comparisons(
    baseline_results: list[dict[str, Any]],
    cost_adjusted_metric_comparison: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    comparison_by_name = {
        str(row.get("name", "")): row
        for row in cost_adjusted_metric_comparison
        if row.get("role") == "baseline" and row.get("name")
    }
    comparisons: dict[str, dict[str, Any]] = {}
    for result in baseline_results:
        name = str(result.get("name", ""))
        if not name:
            continue
        comparison = comparison_by_name.get(name, {})
        entry = {
            "name": name,
            "baseline_type": "equal_weight_universe" if name == "equal_weight" else "market_benchmark",
            "return_basis": comparison.get("return_basis"),
            "cagr": comparison.get("cagr", result.get("cagr", 0.0)),
            "sharpe": comparison.get("sharpe", result.get("sharpe", 0.0)),
            "max_drawdown": comparison.get("max_drawdown", result.get("max_drawdown", 0.0)),
            "cumulative_return": comparison.get(
                "cumulative_return",
                result.get("cumulative_return", 0.0),
            ),
            "cost_adjusted_cumulative_return": comparison.get(
                "cost_adjusted_cumulative_return",
                result.get("cost_adjusted_cumulative_return", result.get("cumulative_return", 0.0)),
            ),
            "average_daily_turnover": comparison.get(
                "average_daily_turnover",
                result.get("average_daily_turnover", 0.0),
            ),
            "transaction_cost_return": comparison.get(
                "transaction_cost_return",
                result.get("transaction_cost_return", 0.0),
            ),
            "slippage_cost_return": comparison.get(
                "slippage_cost_return",
                result.get("slippage_cost_return", 0.0),
            ),
            "total_cost_return": comparison.get(
                "total_cost_return",
                result.get("total_cost_return", 0.0),
            ),
            "excess_return": comparison.get(
                "excess_return",
                result.get("excess_return", 0.0),
            ),
            "strategy_excess_return": comparison.get(
                "strategy_excess_return",
                result.get("excess_return", 0.0),
            ),
            "excess_return_status": comparison.get(
                "excess_return_status",
                _baseline_excess_return_status(
                    result.get("excess_return"),
                    result.get("evaluation_observations", result.get("observations", 0)),
                ),
            ),
            "cagr_excess_return": comparison.get(
                "cagr_excess_return",
                result.get("cagr_excess_return"),
            ),
            "cost_bps": comparison.get("cost_bps", result.get("cost_bps", 0.0)),
            "slippage_bps": comparison.get("slippage_bps", result.get("slippage_bps", 0.0)),
            "evaluation_observations": comparison.get(
                "evaluation_observations",
                result.get("evaluation_observations", 0),
            ),
            "evaluation_start": comparison.get("evaluation_start", result.get("evaluation_start")),
            "evaluation_end": comparison.get("evaluation_end", result.get("evaluation_end")),
            "return_column": comparison.get("return_column", result.get("return_column")),
            "return_horizon": comparison.get("return_horizon", result.get("return_horizon")),
            "sample_alignment": comparison.get(
                "sample_alignment",
                result.get("sample_alignment", {}),
            ),
            "sample_alignment_status": comparison.get(
                "sample_alignment_status",
                result.get("sample_alignment_status"),
            ),
        }
        for key in (
            "universe_tickers",
            "expected_constituent_count",
            "min_constituent_count",
            "max_constituent_count",
            "incomplete_rebalance_count",
            "missing_return_count",
            "rebalance_count",
            "rebalance_assumption",
            "return_timing",
            "gross_cagr",
            "gross_cumulative_return",
        ):
            if key in result:
                entry[key] = result[key]
        comparisons[name] = entry
    return comparisons


def _baseline_metric_snapshot(prefix: str, result: Mapping[str, Any]) -> dict[str, Any]:
    cost_adjusted_return = result.get(
        "cost_adjusted_cumulative_return",
        result.get("cumulative_return"),
    )
    return {
        f"{prefix}_name": result.get("name"),
        f"{prefix}_cagr": result.get("cagr"),
        f"{prefix}_sharpe": result.get("sharpe"),
        f"{prefix}_max_drawdown": result.get("max_drawdown"),
        f"{prefix}_cost_adjusted_cumulative_return": cost_adjusted_return,
        f"{prefix}_average_daily_turnover": result.get("average_daily_turnover"),
        f"{prefix}_total_cost_return": result.get("total_cost_return"),
        f"{prefix}_evaluation_observations": result.get("evaluation_observations"),
        f"{prefix}_evaluation_start": result.get("evaluation_start"),
        f"{prefix}_evaluation_end": result.get("evaluation_end"),
        f"{prefix}_return_column": result.get("return_column"),
        f"{prefix}_return_horizon": result.get("return_horizon"),
    }


def _baseline_metric_comparison_row(
    result: dict[str, Any],
    strategy_cost_adjusted_return: float,
    strategy_cagr: float,
) -> dict[str, Any]:
    name = str(result.get("name", ""))
    is_equal_weight = name == "equal_weight"
    baseline_cagr = _finite_or_none(result.get("cagr")) or 0.0
    cumulative_return = result.get("cost_adjusted_cumulative_return")
    if cumulative_return is None:
        cumulative_return = result.get("cumulative_return", 0.0)
    baseline_cost_adjusted_return = _finite_or_none(cumulative_return) or 0.0
    sample_alignment = _mapping_or_empty(result.get("sample_alignment"))
    excess_return = _aligned_cost_adjusted_excess_return(
        sample_alignment,
        strategy_cost_adjusted_return,
        baseline_cost_adjusted_return,
    )
    evaluation_observations = result.get("evaluation_observations", 0)
    return {
        "name": name,
        "role": "baseline",
        "return_basis": (
            "cost_adjusted_equal_weight_return"
            if is_equal_weight
            else "cost_adjusted_benchmark_return"
        ),
        "cagr": baseline_cagr,
        "sharpe": result.get("sharpe", 0.0),
        "max_drawdown": result.get("max_drawdown", 0.0),
        "cumulative_return": cumulative_return,
        "gross_cumulative_return": result.get(
            "gross_cumulative_return",
            result.get("cumulative_return", 0.0),
        ),
        "cost_adjusted_cumulative_return": cumulative_return,
        "average_daily_turnover": result.get("average_daily_turnover", 0.0),
        "transaction_cost_return": result.get("transaction_cost_return", 0.0),
        "slippage_cost_return": result.get("slippage_cost_return", 0.0),
        "total_cost_return": result.get("total_cost_return", 0.0),
        "excess_return": excess_return,
        "strategy_excess_return": excess_return,
        "excess_return_status": _baseline_excess_return_status(
            excess_return,
            evaluation_observations,
        ),
        "cagr_excess_return": result.get(
            "cagr_excess_return",
            strategy_cagr - baseline_cagr,
        ),
        "cost_bps": result.get("cost_bps", 0.0),
        "slippage_bps": result.get("slippage_bps", 0.0),
        "evaluation_observations": evaluation_observations,
        "evaluation_start": result.get("evaluation_start"),
        "evaluation_end": result.get("evaluation_end"),
        "return_column": result.get("return_column"),
        "return_horizon": result.get("return_horizon"),
        "sample_alignment": sample_alignment,
        "sample_alignment_status": result.get(
            "sample_alignment_status",
            sample_alignment.get("status"),
        ),
    }


def _cost_adjusted_excess_return(
    strategy_cost_adjusted_return: object,
    baseline_cost_adjusted_return: object,
) -> float:
    strategy_return = _finite_or_none(strategy_cost_adjusted_return) or 0.0
    baseline_return = _finite_or_none(baseline_cost_adjusted_return) or 0.0
    return float(strategy_return - baseline_return)


def _row_excess_return(row: Mapping[str, Any]) -> object:
    if "excess_return" in row:
        return row.get("excess_return")
    return row.get("strategy_excess_return")


def _baseline_excess_return_status(
    excess_return: object,
    evaluation_observations: object,
) -> str:
    observation_count = int(_finite_or_none(evaluation_observations) or 0)
    finite_excess_return = _finite_or_none(excess_return)
    if observation_count <= 0 or finite_excess_return is None:
        return "not_evaluable"
    if finite_excess_return > 0:
        return "pass"
    return "fail"


def _evaluate_cost_adjusted(
    baseline_results: list[dict[str, Any]],
    *,
    cost_adjustment: Mapping[str, Any] | None = None,
    thresholds: ValidationGateThresholds | None = None,
    collapse_threshold: float | None = None,
) -> dict[str, Any]:
    baseline_excess_returns: dict[str, float | None] = {}
    baseline_excess_return_statuses: dict[str, str] = {}
    baseline_evaluation_observations: dict[str, int] = {}
    passed_baselines: list[str] = []
    failed_baselines: list[str] = []
    collapse_check = _cost_adjusted_collapse_check(
        cost_adjustment,
        collapse_threshold=collapse_threshold,
        thresholds=thresholds,
    )
    for idx, result in enumerate(baseline_results):
        name = str(result.get("name") or f"baseline_{idx + 1}")
        excess_return = _finite_or_none(result.get("excess_return"))
        evaluation_observations = _baseline_result_evaluation_observation_count(result)
        status = _baseline_excess_return_status(excess_return, evaluation_observations)
        baseline_excess_returns[name] = excess_return
        baseline_excess_return_statuses[name] = status
        baseline_evaluation_observations[name] = evaluation_observations
        if status == "pass":
            passed_baselines.append(name)
        else:
            failed_baselines.append(name)

    base_result = {
        "required_baselines": list(baseline_excess_returns),
        "baseline_excess_returns": baseline_excess_returns,
        "baseline_excess_return_statuses": baseline_excess_return_statuses,
        "baseline_evaluation_observations": baseline_evaluation_observations,
        "passed_baselines": passed_baselines,
        "failed_baselines": failed_baselines,
        "operator": "excess_return > 0 for every required baseline",
        "cost_adjusted_cumulative_return": collapse_check["value"],
        "collapse_threshold": collapse_check["threshold"],
        "collapse_operator": collapse_check["operator"],
        "collapse_status": collapse_check["status"],
        "collapse_reason_code": collapse_check["reason_code"],
        "collapse_check": collapse_check,
    }
    if not baseline_results or sum(baseline_evaluation_observations.values()) == 0:
        return {
            **base_result,
            "status": "not_evaluable",
            "reason": "cost-adjusted baselines are unavailable",
        }
    if collapse_check["status"] == "fail":
        return {
            **base_result,
            "status": "fail",
            "reason": str(collapse_check["reason"]),
            "reason_metadata": collapse_check["reason_metadata"],
        }
    if not failed_baselines:
        return {
            **base_result,
            "status": "pass",
            "reason": "net excess return is positive versus all required baselines",
        }
    return {
        **base_result,
        "status": "fail",
        "reason": (
            "net excess return is not positive versus required baseline(s): "
            f"{', '.join(failed_baselines)}"
        ),
    }


def _configured_cost_adjusted_collapse_threshold(
    config: object | None,
    thresholds: ValidationGateThresholds,
) -> float:
    configured_value = getattr(config, "cost_adjusted_collapse_threshold", None)
    if configured_value is None:
        configured_value = getattr(config, "cost_adjusted_performance_collapse_threshold", None)
    finite_configured_value = _finite_or_none(configured_value)
    if finite_configured_value is not None:
        return finite_configured_value
    return _cost_adjusted_collapse_threshold(thresholds, None)


def _cost_adjusted_collapse_threshold(
    thresholds: ValidationGateThresholds | None,
    override: float | None,
) -> float:
    finite_override = _finite_or_none(override)
    if finite_override is not None:
        return finite_override
    threshold_value = getattr(thresholds, "cost_adjusted_collapse_threshold", 0.0)
    finite_threshold_value = _finite_or_none(threshold_value)
    if finite_threshold_value is None:
        return 0.0
    return finite_threshold_value


def _cost_adjusted_collapse_check(
    cost_adjustment: Mapping[str, Any] | None,
    *,
    collapse_threshold: float | None = None,
    thresholds: ValidationGateThresholds | None = None,
) -> dict[str, Any]:
    threshold = _cost_adjusted_collapse_threshold(thresholds, collapse_threshold)
    value = None
    if cost_adjustment is not None:
        value = _finite_or_none(cost_adjustment.get("cost_adjusted_cumulative_return"))
    base_metadata = {
        "code": None,
        "metric": "cost_adjusted_cumulative_return",
        "value": value,
        "threshold": threshold,
        "operator": ">",
    }
    if value is None:
        return {
            "status": "not_evaluable",
            "reason": "cost-adjusted cumulative return is unavailable",
            "reason_code": "cost_adjusted_cumulative_return_unavailable",
            "metric": "cost_adjusted_cumulative_return",
            "value": value,
            "threshold": threshold,
            "operator": ">",
            "reason_metadata": {
                **base_metadata,
                "code": "cost_adjusted_cumulative_return_unavailable",
            },
        }
    if value <= threshold:
        reason_code = "cost_adjusted_cumulative_return_at_or_below_collapse_threshold"
        return {
            "status": "fail",
            "reason": (
                "cost-adjusted cumulative return "
                f"{value:.4f} is at or below configured collapse threshold "
                f"{threshold:.4f}"
            ),
            "reason_code": reason_code,
            "metric": "cost_adjusted_cumulative_return",
            "value": value,
            "threshold": threshold,
            "operator": ">",
            "reason_metadata": {
                **base_metadata,
                "code": reason_code,
            },
        }
    return {
        "status": "pass",
        "reason": "cost-adjusted cumulative return is above configured collapse threshold",
        "reason_code": None,
        "metric": "cost_adjusted_cumulative_return",
        "value": value,
        "threshold": threshold,
        "operator": ">",
        "reason_metadata": base_metadata,
    }


def _evaluate_benchmark(
    baseline_results: list[dict[str, Any]],
    thresholds: ValidationGateThresholds,
) -> dict[str, Any]:
    if _baseline_evaluation_observation_count(baseline_results) == 0:
        return {"status": "not_evaluable", "reason": "benchmark comparison is unavailable"}
    strategy_sharpe = float(baseline_results[0].get("strategy_sharpe", np.nan))
    benchmark_sharpe = max(float(result.get("sharpe", 0.0)) for result in baseline_results)
    if not np.isfinite(strategy_sharpe):
        return {"status": "not_evaluable", "reason": "strategy Sharpe is unavailable"}
    if strategy_sharpe >= thresholds.sharpe_pass or strategy_sharpe - benchmark_sharpe >= thresholds.benchmark_sharpe_margin:
        return {"status": "pass", "reason": "strategy Sharpe passes absolute or benchmark-relative threshold"}
    if strategy_sharpe >= thresholds.sharpe_warning:
        return {"status": "warning", "reason": "strategy Sharpe is in warning band"}
    return {"status": "fail", "reason": "strategy Sharpe is below minimum warning threshold"}


def _baseline_evaluation_observation_count(baseline_results: list[dict[str, Any]]) -> int:
    total = 0
    for result in baseline_results:
        total += _baseline_result_evaluation_observation_count(result)
    return total


def _baseline_result_evaluation_observation_count(result: Mapping[str, Any]) -> int:
    return int(
        _finite_or_none(
            result.get("evaluation_observations", result.get("observations", 0))
        )
        or 0
    )


def _evaluate_turnover(
    strategy_metrics: object,
    baseline_results: list[dict[str, Any]],
    ablation_summary: list[dict[str, Any]],
    thresholds: ValidationGateThresholds,
) -> dict[str, Any]:
    turnover = _metric(strategy_metrics, "turnover")
    threshold = float(thresholds.max_daily_turnover)
    if turnover <= threshold:
        return {
            "status": "pass",
            "reason": "average daily turnover is within budget",
            "value": turnover,
            "threshold": threshold,
            "operator": "<=",
        }
    no_cost = next(
        (
            row
            for row in ablation_summary
            if row.get("scenario") == NO_COST_ABLATION_SCENARIO
        ),
        None,
    )
    cost_collapse = bool(
        no_cost
        and float(no_cost.get("excess_return", 0.0)) > 0
        and all(float(row.get("excess_return", 0.0)) <= 0 for row in baseline_results)
    )
    status = "fail" if cost_collapse else "warning"
    reason = (
        "average daily turnover exceeds budget and performance collapses after costs"
        if cost_collapse
        else "average daily turnover exceeds budget but no cost collapse was detected"
    )
    return {
        "status": status,
        "reason": reason,
        "value": turnover,
        "threshold": threshold,
        "operator": "<=",
        "structured_warning": _average_daily_turnover_structured_warning(
            {
                "reason": reason,
                "value": turnover,
                "threshold": threshold,
                "operator": "<=",
            }
        ),
    }


def _combine_turnover_validity_gates(
    daily_turnover_gate: dict[str, Any],
    monthly_turnover_gate: dict[str, Any],
) -> dict[str, Any]:
    daily_status = str(daily_turnover_gate.get("status", "not_evaluable"))
    monthly_status = str(monthly_turnover_gate.get("status", "not_evaluable"))
    components = {
        "average_daily_turnover": daily_turnover_gate,
        "monthly_turnover_budget": monthly_turnover_gate,
    }
    passed_by = [
        name
        for name, status in (
            ("average_daily_turnover", daily_status),
            ("monthly_turnover_budget", monthly_status),
        )
        if status == "pass"
    ]
    base_result = {
        "daily_status": daily_status,
        "monthly_status": monthly_status,
        "value": daily_turnover_gate.get("value"),
        "threshold": daily_turnover_gate.get("threshold"),
        "monthly_value": monthly_turnover_gate.get("value"),
        "monthly_threshold": monthly_turnover_gate.get("threshold"),
        "operator": "daily <= threshold OR monthly <= budget",
        "components": components,
        "passed_by": passed_by,
    }

    if passed_by:
        return {
            **base_result,
            "status": "pass",
            "reason": f"turnover validity passed via {' and '.join(passed_by)}",
        }
    if daily_status == "not_evaluable" and monthly_status == "not_evaluable":
        return {
            **base_result,
            "status": "not_evaluable",
            "reason": "daily and monthly turnover are unavailable",
        }
    return {
        **base_result,
        "status": "fail",
        "reason": "daily turnover limit and monthly budget both missed pass thresholds",
    }


def _monthly_turnover_budget(
    thresholds: ValidationGateThresholds,
    override: float | None,
) -> float:
    if override is not None:
        return float(override)
    if thresholds.monthly_turnover_budget is not None:
        return float(thresholds.monthly_turnover_budget)
    return float(thresholds.max_monthly_turnover)


def _turnover_budget_structured_warning(gate_result: Mapping[str, Any]) -> dict[str, Any]:
    value = gate_result.get("value")
    threshold = gate_result.get("threshold")
    operator = str(gate_result.get("operator", "<="))
    reason = str(gate_result.get("reason", "monthly turnover exceeds configured budget"))
    message = (
        "monthly_turnover_budget: realized max monthly turnover "
        f"{_format_metric(value)} exceeds configured budget {_format_metric(threshold)}"
    )
    return {
        "code": "monthly_turnover_budget_exceeded",
        "severity": "warning",
        "gate": "monthly_turnover_budget",
        "combined_gate": "turnover",
        "metric": "max_monthly_turnover",
        "value": value,
        "realized_turnover": value,
        "threshold": threshold,
        "budget": threshold,
        "operator": operator,
        "reason": reason,
        "message": message,
        "monthly_turnover": dict(gate_result.get("monthly_turnover", {}) or {}),
    }


def _average_daily_turnover_structured_warning(gate_result: Mapping[str, Any]) -> dict[str, Any]:
    value = gate_result.get("value")
    threshold = gate_result.get("threshold")
    operator = str(gate_result.get("operator", "<="))
    reason = str(gate_result.get("reason", "average daily turnover exceeds configured budget"))
    message = (
        "average_daily_turnover: realized average daily turnover "
        f"{_format_metric(value)} exceeds configured budget {_format_metric(threshold)}"
    )
    return {
        "code": "average_daily_turnover_budget_exceeded",
        "severity": "warning",
        "gate": "average_daily_turnover",
        "combined_gate": "turnover",
        "metric": "average_daily_turnover",
        "value": value,
        "realized_turnover": value,
        "threshold": threshold,
        "budget": threshold,
        "operator": operator,
        "reason": reason,
        "message": message,
    }


def _structured_warning_from_gate(gate_result: Mapping[str, Any]) -> dict[str, Any]:
    structured_warning = gate_result.get("structured_warning")
    if not isinstance(structured_warning, Mapping):
        return {}
    return dict(structured_warning)


def _report_hard_fail_reasons(gate_results: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    for gate_name, result in gate_results.items():
        if not isinstance(result, Mapping):
            continue
        if result.get("status") != "hard_fail":
            continue
        nested_reasons = result.get("reasons")
        if isinstance(nested_reasons, Iterable) and not isinstance(
            nested_reasons, str | bytes | Mapping
        ):
            for reason in nested_reasons:
                if str(reason).strip():
                    reasons.append(str(reason))
        else:
            reason = str(result.get("reason", "")).strip()
            reasons.append(reason if reason else str(gate_name))
    return _deduplicate_preserving_order(reasons)


def _report_warning_messages(gate_results: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    for gate_name, result in gate_results.items():
        if not isinstance(result, Mapping):
            continue
        if result.get("status") != "warning":
            continue
        structured_warning = _structured_warning_from_gate(result)
        message = str(
            structured_warning.get("message")
            or result.get("reason")
            or gate_name
        ).strip()
        if not message:
            continue
        warnings.append(message)
    return _deduplicate_preserving_order(warnings)


def _structured_warning_reasons(
    gate_results: Mapping[str, Any],
    existing_warnings: Iterable[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for warning in existing_warnings:
        if isinstance(warning, Mapping):
            rows.append(dict(warning))

    for gate_name, result in gate_results.items():
        if not isinstance(result, Mapping):
            continue
        if result.get("status") != "warning":
            continue
        if (
            result.get("affects_system") is False
            and result.get("affects_strategy") is False
            and result.get("affects_pass_fail") is False
        ):
            continue
        metadata = _gate_result_metadata(result)
        structured_warning = _structured_warning_from_gate(result)
        row = {
            "code": (
                structured_warning.get("code")
                or _gate_result_reason_code(result, metadata)
                or f"{gate_name}_warning"
            ),
            "severity": "warning",
            "gate": structured_warning.get("gate") or gate_name,
            "combined_gate": structured_warning.get("combined_gate") or result.get("combined_gate"),
            "metric": structured_warning.get("metric") or metadata.get("metric"),
            "message": (
                structured_warning.get("message")
                or result.get("reason")
                or f"{gate_name} emitted a warning"
            ),
            "reason": structured_warning.get("reason") or result.get("reason"),
            "value": structured_warning.get("value", metadata.get("value")),
            "threshold": structured_warning.get("threshold", metadata.get("threshold")),
            "operator": structured_warning.get("operator", metadata.get("operator")),
            "affects_system": bool(result.get("affects_system", True)),
            "affects_strategy": bool(result.get("affects_strategy", True)),
        }
        for key, value in structured_warning.items():
            row.setdefault(key, value)
        rows.append(row)

    deduplicated: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (
            str(row.get("gate", "")),
            str(row.get("code", "")),
            str(row.get("message", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(row)
    return deduplicated


def _monthly_turnover_observations(value: object) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return _monthly_turnover_from_frame(value)
    if isinstance(value, pd.Series):
        return _monthly_turnover_from_series(value)
    if isinstance(value, Mapping):
        for key in ("monthly_turnover", "monthly_turnover_max", "max_monthly_turnover", "turnover"):
            if key in value:
                return _monthly_turnover_observations(value[key])
        return _monthly_turnover_from_mapping_values(value)
    if hasattr(value, "equity_curve"):
        return _monthly_turnover_observations(value.equity_curve)
    for attr in ("monthly_turnover", "turnover"):
        if hasattr(value, attr):
            return _monthly_turnover_observations(getattr(value, attr))
    numeric = _finite_or_none(value)
    if numeric is None:
        return pd.DataFrame(columns=["month", "turnover"])
    return pd.DataFrame({"month": ["configured"], "turnover": [numeric]})


def _monthly_turnover_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["month", "turnover"])
    if "monthly_turnover" in frame:
        month = _month_labels(frame)
        turnover = pd.to_numeric(frame["monthly_turnover"], errors="coerce")
        return _clean_monthly_turnover(month, turnover)
    if "turnover" not in frame:
        return pd.DataFrame(columns=["month", "turnover"])

    turnover = pd.to_numeric(frame["turnover"], errors="coerce")
    if "date" not in frame:
        total = float(turnover.fillna(0.0).sum())
        return pd.DataFrame({"month": ["configured"], "turnover": [total]})

    dates = pd.to_datetime(frame["date"], errors="coerce")
    valid = dates.notna()
    if not bool(valid.any()):
        return pd.DataFrame(columns=["month", "turnover"])
    monthly = (
        pd.DataFrame({"month": dates[valid].dt.to_period("M").astype(str), "turnover": turnover[valid].fillna(0.0)})
        .groupby("month", as_index=False)["turnover"]
        .sum()
    )
    return monthly


def _monthly_turnover_from_series(series: pd.Series) -> pd.DataFrame:
    turnover = pd.to_numeric(series, errors="coerce")
    if isinstance(series.index, pd.DatetimeIndex):
        frame = pd.DataFrame(
            {
                "month": series.index.to_period("M").astype(str),
                "turnover": turnover.fillna(0.0).to_numpy(dtype=float),
            }
        )
        return frame.groupby("month", as_index=False)["turnover"].sum()
    labels = [str(label) for label in series.index]
    return _clean_monthly_turnover(pd.Series(labels), turnover)


def _monthly_turnover_from_mapping_values(values: Mapping[object, object]) -> pd.DataFrame:
    rows = [
        {"month": str(key), "turnover": numeric}
        for key, value in values.items()
        if (numeric := _finite_or_none(value)) is not None
    ]
    if not rows:
        return pd.DataFrame(columns=["month", "turnover"])
    return pd.DataFrame(rows)


def _month_labels(frame: pd.DataFrame) -> pd.Series:
    if "month" in frame:
        return frame["month"].astype(str)
    if "date" in frame:
        dates = pd.to_datetime(frame["date"], errors="coerce")
        return dates.dt.to_period("M").astype(str)
    return pd.Series([f"row_{idx}" for idx in range(len(frame))], index=frame.index)


def _clean_monthly_turnover(month: pd.Series, turnover: pd.Series) -> pd.DataFrame:
    clean = pd.DataFrame({"month": month, "turnover": turnover}).dropna(subset=["turnover"])
    if clean.empty:
        return pd.DataFrame(columns=["month", "turnover"])
    clean["month"] = clean["month"].astype(str)
    clean["turnover"] = pd.to_numeric(clean["turnover"], errors="coerce").fillna(0.0)
    return clean.groupby("month", as_index=False)["turnover"].sum()


def _finite_or_none(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _cost_adjustment_metrics(equity_curve: pd.DataFrame) -> dict[str, Any]:
    if equity_curve.empty:
        return {
            "gross_cumulative_return": 0.0,
            "cost_adjusted_cumulative_return": 0.0,
            "transaction_cost_return": 0.0,
            "slippage_cost_return": 0.0,
            "total_cost_return": 0.0,
            "mean_turnover": 0.0,
            "has_explicit_cost_breakdown": False,
        }

    net = _numeric_column(equity_curve, "cost_adjusted_return", "net_return", "portfolio_return")
    transaction_cost = _numeric_column(equity_curve, "transaction_cost_return")
    slippage_cost = _numeric_column(equity_curve, "slippage_cost_return")
    total_cost = _numeric_column(equity_curve, "total_cost_return", "turnover_cost_return")
    if total_cost.empty:
        total_cost = transaction_cost.add(slippage_cost, fill_value=0.0)
    gross = _numeric_column(equity_curve, "gross_return", "deterministic_strategy_return")
    if gross.empty:
        gross = net.add(total_cost, fill_value=0.0)
    turnover = _numeric_column(equity_curve, "turnover")

    return {
        "gross_cumulative_return": _compound_return(gross),
        "cost_adjusted_cumulative_return": _compound_return(net),
        "transaction_cost_return": float(transaction_cost.sum()) if not transaction_cost.empty else 0.0,
        "slippage_cost_return": float(slippage_cost.sum()) if not slippage_cost.empty else 0.0,
        "total_cost_return": float(total_cost.sum()) if not total_cost.empty else 0.0,
        "mean_turnover": float(turnover.mean()) if not turnover.empty else 0.0,
        "has_explicit_cost_breakdown": {
            "transaction_cost_return",
            "slippage_cost_return",
            "total_cost_return",
        }.issubset(equity_curve.columns),
    }


def _numeric_column(frame: pd.DataFrame, *columns: str) -> pd.Series:
    for column in columns:
        if column in frame:
            return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return pd.Series(dtype=float)


def _compound_return(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return float((1 + returns.fillna(0.0)).prod() - 1)


def _evaluate_drawdown(
    baseline_results: list[dict[str, Any]],
    thresholds: ValidationGateThresholds,
) -> dict[str, Any]:
    strategy_drawdown = float(baseline_results[0].get("strategy_max_drawdown", np.nan))
    if not np.isfinite(strategy_drawdown):
        return {"status": "not_evaluable", "reason": "strategy drawdown is unavailable"}
    spy_drawdown = float(baseline_results[0].get("max_drawdown", 0.0))
    if strategy_drawdown >= thresholds.drawdown_pass and strategy_drawdown >= spy_drawdown - thresholds.max_drawdown_spy_lag:
        return {"status": "pass", "reason": "drawdown is within absolute and SPY-relative limits"}
    if strategy_drawdown >= thresholds.drawdown_warning:
        return {"status": "warning", "reason": "drawdown is in warning band"}
    return {"status": "fail", "reason": "drawdown is at or beyond fail threshold"}


def _evaluate_deterministic_strategy_validity(
    gate_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    required_rules = (
        "cost_adjusted_performance",
        "benchmark_comparison",
        "turnover",
        "drawdown",
    )
    rule_statuses = {
        rule: str(_mapping_or_empty(gate_results.get(rule)).get("status", "not_evaluable"))
        for rule in required_rules
    }
    failed_rules = [
        rule
        for rule, status in rule_statuses.items()
        if status != "pass"
    ]
    passed_rules = [
        rule
        for rule, status in rule_statuses.items()
        if status == "pass"
    ]
    rule_reasons = {
        rule: str(_mapping_or_empty(gate_results.get(rule)).get("reason", ""))
        for rule in required_rules
    }
    gate = {
        "status": "pass" if not failed_rules else "fail",
        "required_rules": list(required_rules),
        "passed_rules": passed_rules,
        "failed_rules": failed_rules,
        "rule_statuses": rule_statuses,
        "rule_reasons": rule_reasons,
        "operator": "every required deterministic outperformance rule status == pass",
        "all_required_outperformance_rules_passed": not failed_rules,
        "affects_strategy": True,
    }
    if not failed_rules:
        gate.update(
            {
                "reason": "all required deterministic outperformance rules passed",
                "reason_code": None,
            }
        )
        return gate

    first_failed_rule = failed_rules[0]
    first_failed_status = rule_statuses[first_failed_rule]
    gate.update(
        {
            "reason": (
                "required deterministic outperformance rule(s) did not pass: "
                f"{', '.join(failed_rules)}"
            ),
            "reason_code": "required_outperformance_rule_not_passed",
            "reason_metadata": {
                "code": "required_outperformance_rule_not_passed",
                "metric": first_failed_rule,
                "value": first_failed_status,
                "threshold": "pass",
                "operator": "==",
            },
        }
    )
    return gate


def _evaluate_ablation(ablation_summary: list[dict[str, Any]]) -> dict[str, Any]:
    required = set(STAGE1_REQUIRED_ABLATION_SCENARIOS)
    scenarios = {str(row.get("scenario")) for row in ablation_summary}
    missing = sorted(required - scenarios)
    if missing:
        return {"status": "warning", "reason": f"missing ablation scenarios: {', '.join(missing)}"}
    full = [row for row in ablation_summary if str(row.get("scenario")) in {"all_features", "full_model_features"}]
    others = [
        row
        for row in ablation_summary
        if row not in full and row.get("scenario") != NO_COST_ABLATION_SCENARIO
    ]
    if not full or not others:
        return {"status": "not_evaluable", "reason": "full model or comparison ablations are unavailable"}
    full_sharpe = max(float(row.get("sharpe", 0.0)) for row in full)
    other_sharpe = max(float(row.get("sharpe", 0.0)) for row in others)
    if full_sharpe >= other_sharpe:
        return {"status": "pass", "reason": "full model matches or beats ablation alternatives"}
    return {"status": "warning", "reason": "full model does not beat one or more ablation alternatives"}


def _pipeline_control_results(ablation_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = {NO_MODEL_PROXY_ABLATION_SCENARIO}
    return [row for row in ablation_summary if str(row.get("scenario")) in required]


def _cost_ablation_results(ablation_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = {NO_COST_ABLATION_SCENARIO}
    return [row for row in ablation_summary if str(row.get("scenario")) in required]


def _stage1_ablation_comparison_results(
    ablation_summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_scenario = {
        str(row.get("scenario")): row
        for row in ablation_summary
        if str(row.get("scenario")) in STAGE1_REQUIRED_ABLATION_SCENARIOS
    }
    return [
        rows_by_scenario[scenario]
        for scenario in STAGE1_REQUIRED_ABLATION_SCENARIOS
        if scenario in rows_by_scenario
    ]


def _ablation_signal_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _mapping_or_empty(row.get("deterministic_signal_evaluation_metrics"))
    if metrics:
        return metrics
    return _mapping_or_empty(row.get("signal_evaluation_metrics"))


def _ablation_signal_metric(
    row: Mapping[str, Any],
    signal_metrics: Mapping[str, Any],
    field: str,
) -> object:
    if field in signal_metrics:
        return signal_metrics[field]
    return row.get(f"signal_{field}")


def _format_feature_families(row: Mapping[str, Any]) -> str:
    families = (
        row.get("input_feature_families")
        or row.get("permitted_feature_families")
        or row.get("feature_family_allowlist")
        or []
    )
    if isinstance(families, str):
        return families
    if isinstance(families, Iterable):
        return ", ".join(str(family) for family in families)
    return ""


def _no_model_proxy_ablation_result(ablation_summary: list[dict[str, Any]]) -> dict[str, Any]:
    row = next(
        (
            row
            for row in ablation_summary
            if str(row.get("scenario")) == NO_MODEL_PROXY_ABLATION_SCENARIO
        ),
        None,
    )
    if row is None:
        return {
            "scenario": NO_MODEL_PROXY_ABLATION_SCENARIO,
            "available": False,
            "status": "missing",
            "reason": "no_model_proxy ablation result is unavailable",
            "pipeline_controls": {},
            "performance_metrics": {},
            "validation_metrics": {},
            "deterministic_signal_evaluation_metrics": {},
        }

    signal_metrics = _mapping_or_empty(row.get("deterministic_signal_evaluation_metrics"))
    if not signal_metrics:
        signal_metrics = _mapping_or_empty(row.get("signal_evaluation_metrics"))
    validation_metrics = {
        key: row.get(key)
        for key in (
            "validation_status",
            "validation_reason",
            "validation_skip_status",
            "validation_skip_code",
            "validation_fold_count",
            "validation_oos_fold_count",
            "validation_prediction_count",
            "validation_labeled_prediction_count",
            "validation_labeled_date_count",
            "validation_required_min_date_count",
            "validation_candidate_fold_count",
            "validation_mean_mae",
            "validation_mean_directional_accuracy",
            "validation_mean_information_coefficient",
            "validation_positive_ic_fold_ratio",
            "validation_oos_information_coefficient",
        )
        if key in row
    }
    performance_metrics = {
        key: row.get(key)
        for key in (
            "cagr",
            "sharpe",
            "max_drawdown",
            "turnover",
            "excess_return",
            "effective_cost_bps",
            "effective_slippage_bps",
        )
        if key in row
    }
    controls = _mapping_or_empty(row.get("pipeline_controls"))
    return {
        "scenario": NO_MODEL_PROXY_ABLATION_SCENARIO,
        "available": True,
        "status": validation_metrics.get("validation_status", "available"),
        "kind": row.get("kind"),
        "label": row.get("label"),
        "description": row.get("description"),
        "model_proxy_enabled": controls.get("model_proxy"),
        "pipeline_controls": controls,
        "toggles": _mapping_or_empty(row.get("toggles")),
        "feature_sources": _mapping_or_empty(row.get("feature_sources")),
        "permitted_feature_families": list(row.get("permitted_feature_families", []) or []),
        "input_feature_families": list(row.get("input_feature_families", []) or []),
        "input_feature_columns": list(row.get("input_feature_columns", []) or []),
        "performance_metrics": performance_metrics,
        "validation_metrics": validation_metrics,
        "deterministic_signal_evaluation_metrics": signal_metrics,
    }


def _mapping_or_empty(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _safe_identifier(value: object) -> str:
    normalized = "".join(
        char.lower() if char.isalnum() else "_"
        for char in str(value).strip()
    ).strip("_")
    return normalized or "unknown"


def _system_validity_status(
    hard_fail_reasons: list[str],
    insufficient_data: bool,
) -> SystemValidityStatus:
    if hard_fail_reasons:
        return "hard_fail"
    if insufficient_data:
        return "not_evaluable"
    return "pass"


def aggregate_deterministic_gate_results(
    gate_results: Mapping[str, Mapping[str, Any]],
    *,
    system_validity_status: str | None = None,
    strategy_candidate_status: str | None = None,
) -> dict[str, Any]:
    """Aggregate per-item gate statuses into the final PASS/WARN/FAIL decision.

    This is intentionally provider-free and metric-free: it only consumes the
    item-level status payloads already produced by deterministic validation
    rules. Missing or non-evaluable required items cannot produce a PASS.
    """

    item_results = [
        _deterministic_gate_aggregation_item(name, result)
        for name, result in gate_results.items()
        if isinstance(result, Mapping)
        and bool(
            result.get(
                "affects_pass_fail",
                result.get(
                    "affects_system",
                    result.get("affects_strategy", True),
                ),
            )
        )
    ]
    blocking_items = [
        item["gate"]
        for item in item_results
        if item["normalized_status"] == "fail"
    ]
    warning_items = [
        item["gate"]
        for item in item_results
        if item["normalized_status"] == "warn"
    ]
    not_evaluable_items = [
        item["gate"]
        for item in item_results
        if DEFAULT_GATE_STATUS_POLICY.normalize(item["raw_status"])
        in INSUFFICIENT_DATA_GATE_STATUSES
    ]
    passed_items = [
        item["gate"]
        for item in item_results
        if item["normalized_status"] == "pass"
    ]

    if system_validity_status is not None:
        normalized_system_status = _normalize_deterministic_gate_item_status(
            system_validity_status
        )
        if normalized_system_status == "fail":
            blocking_items.append("system_validity_status")
        elif normalized_system_status == "warn":
            warning_items.insert(0, "system_validity_status")
    if strategy_candidate_status is not None:
        normalized_strategy_status = _normalize_deterministic_gate_item_status(
            strategy_candidate_status
        )
        if normalized_strategy_status == "fail":
            blocking_items.append("strategy_candidate_status")
        elif normalized_strategy_status == "warn":
            warning_items.append("strategy_candidate_status")

    blocking_items = _deduplicate_preserving_order(blocking_items)
    warning_items = _deduplicate_preserving_order(warning_items)
    not_evaluable_items = _deduplicate_preserving_order(not_evaluable_items)
    passed_items = _deduplicate_preserving_order(passed_items)

    if blocking_items:
        final_decision: DeterministicGateFinalDecision = "FAIL"
        final_status = "fail"
        reason = "one or more required deterministic gate items failed or were not evaluable"
    elif warning_items:
        final_decision = "WARN"
        final_status = "warning"
        reason = "all required deterministic gate items passed hard checks with warning item(s)"
    else:
        final_decision = "PASS"
        final_status = "pass"
        reason = "all required deterministic gate items passed"

    return {
        "engine_id": DETERMINISTIC_VALIDITY_GATE_ENGINE_ID,
        "schema_version": DETERMINISTIC_VALIDITY_GATE_ENGINE_SCHEMA_VERSION,
        "status": final_status,
        "final_status": final_status,
        "final_decision": final_decision,
        "final_decision_contract": list(DETERMINISTIC_GATE_FINAL_DECISIONS),
        "status_precedence": ["FAIL", "WARN", "PASS"],
        "status_aliases": {
            "warning": "WARN",
            "warn": "WARN",
            "hard_fail": "FAIL",
            "insufficient_data": "FAIL",
            "not_evaluable": "FAIL",
            "skipped": "FAIL",
        },
        "reason": reason,
        "system_validity_status": system_validity_status,
        "strategy_candidate_status": strategy_candidate_status,
        "item_results": item_results,
        "blocking_items": blocking_items,
        "warning_items": warning_items,
        "not_evaluable_items": not_evaluable_items,
        "passed_items": passed_items,
        "pass": final_decision == "PASS",
        "warn": final_decision == "WARN",
        "fail": final_decision == "FAIL",
    }


def _deterministic_gate_aggregation_item(
    gate_name: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    raw_status = str(result.get("status", "not_evaluable"))
    return {
        "gate": gate_name,
        "raw_status": raw_status,
        "normalized_status": _normalize_deterministic_gate_item_status(raw_status),
        "affects_system": bool(result.get("affects_system", True)),
        "affects_strategy": bool(result.get("affects_strategy", True)),
        "affects_pass_fail": bool(
            result.get(
                "affects_pass_fail",
                result.get("affects_system", result.get("affects_strategy", True)),
            )
        ),
        "reason": result.get("reason"),
        "reason_code": _gate_result_reason_code(result, _gate_result_metadata(result)),
    }


def _normalize_deterministic_gate_item_status(status: object) -> str:
    normalized = DEFAULT_GATE_STATUS_POLICY.normalize(status)
    if normalized == "pass":
        return "pass"
    if normalized == "warning":
        return "warn"
    return "fail"


def _deduplicate_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _strategy_status(
    system_status: SystemValidityStatus,
    gate_results: dict[str, dict[str, Any]],
    insufficient_data: bool,
) -> StrategyCandidateStatus:
    if system_status == "hard_fail":
        return "not_evaluable"
    if insufficient_data:
        return "insufficient_data"
    statuses = {
        DEFAULT_GATE_STATUS_POLICY.normalize(result.get("status"))
        for result in gate_results.values()
        if result.get("affects_strategy", True)
    }
    if "fail" in statuses:
        return "fail"
    if "hard_fail" in statuses:
        return "fail"
    if "warning" in statuses:
        return "warning"
    return "pass"


def _official_message(system_status: str, strategy_status: str) -> str:
    if system_status == "hard_fail":
        return "시스템 검증이 구조적 실패 상태입니다. 누수 또는 walk-forward/OOS 설정을 먼저 수정해야 합니다."
    if system_status == "not_evaluable" or strategy_status in {"insufficient_data", "not_evaluable"}:
        return "필수 데이터 또는 fold 조건이 부족해 시스템/전략 유효성을 판정할 수 없습니다."
    if strategy_status == "pass":
        return "시스템 검증과 현재 전략 후보가 모두 Stage 1 기준을 통과했습니다."
    return OFFICIAL_STRATEGY_FAIL_MESSAGE


def _rule_result_explanations(gate_results: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gate_name, result in gate_results.items():
        if not isinstance(result, Mapping):
            continue
        metadata = _gate_result_metadata(result)
        status = str(result.get("status", "not_evaluable"))
        rows.append(
            {
                "rule": gate_name,
                "gate": gate_name,
                "status": status,
                "passed": _gate_status_passed(status),
                "pass_fail": status if status in {"pass", "fail"} else None,
                "affects_strategy": bool(result.get("affects_strategy", True)),
                "affects_system": bool(result.get("affects_system", True)),
                "affects_pass_fail": bool(
                    result.get("affects_pass_fail", result.get("affects_strategy", True))
                ),
                "reason": result.get("reason"),
                "reason_code": _gate_result_reason_code(result, metadata),
                "metric": metadata.get("metric"),
                "value": metadata.get("value"),
                "threshold": metadata.get("threshold"),
                "operator": metadata.get("operator"),
                "collapse_status": _mapping_or_empty(result.get("collapse_check")).get(
                    "status"
                ),
                "combined_gate": result.get("combined_gate"),
            }
        )
    return rows


def _final_strategy_status_explanation(report: ValidationGateReport) -> dict[str, Any]:
    rule_rows = report.rule_result_explanations
    strategy_affecting_rows = [
        row for row in rule_rows if row.get("affects_strategy") is not False
    ]
    blocking_rules = [
        str(row["rule"])
        for row in strategy_affecting_rows
        if row.get("status") == "fail"
    ]
    warning_rules = [
        str(row["rule"])
        for row in strategy_affecting_rows
        if row.get("status") == "warning"
    ]
    insufficient_data_rules = [
        str(row["rule"])
        for row in strategy_affecting_rows
        if row.get("status") in INSUFFICIENT_DATA_GATE_STATUSES
    ]
    hard_fail_rules = [
        str(row["rule"])
        for row in rule_rows
        if row.get("status") == "hard_fail"
    ]
    if report.system_validity_status == "hard_fail":
        reason = "system validity hard-failed before strategy candidate approval"
    elif report.strategy_candidate_status == "insufficient_data":
        reason = "strategy candidate cannot be evaluated because required data is insufficient"
    elif report.strategy_candidate_status == "not_evaluable":
        reason = "strategy candidate is not evaluable under the current system status"
    elif blocking_rules:
        reason = (
            "strategy candidate failed because required deterministic rule(s) failed: "
            f"{', '.join(blocking_rules)}"
        )
    elif warning_rules:
        reason = (
            "strategy candidate passed hard checks but has warning rule(s): "
            f"{', '.join(warning_rules)}"
        )
    else:
        reason = "strategy candidate passed all strategy-affecting Stage 1 rules"

    return {
        "final_strategy_status": report.strategy_candidate_status,
        "strategy_candidate_status": report.strategy_candidate_status,
        "strategy_pass": report.strategy_pass,
        "system_validity_status": report.system_validity_status,
        "system_validity_pass": report.system_validity_pass,
        "hard_fail": report.hard_fail,
        "warning": report.warning,
        "official_message": report.official_message,
        "reason": reason,
        "blocking_rules": blocking_rules,
        "warning_rules": warning_rules,
        "insufficient_data_rules": insufficient_data_rules,
        "hard_fail_rules": hard_fail_rules,
        "rule_count": len(rule_rows),
        "strategy_affecting_rule_count": len(strategy_affecting_rows),
        "passed_rule_count": sum(1 for row in strategy_affecting_rows if row.get("status") == "pass"),
        "failed_rule_count": len(blocking_rules),
    }


def _validity_gate_result_summary(report: ValidationGateReport) -> dict[str, Any]:
    """Compact, serialization-friendly summary for report readers and automation."""

    failure_reasons = _summary_failure_reasons(report)
    warning_rows = _summary_warning_rows(report)
    key_metrics = {
        "fold_count": report.metrics.get("fold_count"),
        "oos_fold_count": report.metrics.get("oos_fold_count"),
        "target_horizon": report.metrics.get("target_horizon"),
        "embargo_periods": report.metrics.get("embargo_periods"),
        "mean_rank_ic": report.metrics.get("mean_rank_ic"),
        "oos_rank_ic": report.metrics.get("oos_rank_ic"),
        "positive_fold_ratio": report.metrics.get("positive_fold_ratio"),
        "positive_fold_ratio_threshold": report.metrics.get(
            "positive_fold_ratio_threshold"
        ),
        "strategy_cost_adjusted_cumulative_return": report.metrics.get(
            "strategy_cost_adjusted_cumulative_return"
        ),
        "strategy_excess_return_vs_spy": report.metrics.get(
            "strategy_excess_return_vs_spy"
        ),
        "strategy_excess_return_vs_equal_weight": report.metrics.get(
            "strategy_excess_return_vs_equal_weight"
        ),
        "strategy_max_drawdown": report.metrics.get("strategy_max_drawdown"),
        "strategy_turnover": report.metrics.get("strategy_turnover"),
        "strategy_max_monthly_turnover": report.metrics.get(
            "strategy_max_monthly_turnover"
        ),
        "monthly_turnover_budget": report.metrics.get("monthly_turnover_budget"),
    }
    key_metrics = {key: value for key, value in key_metrics.items() if value is not None}
    final_explanation = report.final_strategy_status_explanation
    deterministic_aggregation = _mapping_or_empty(
        report.metrics.get("deterministic_gate_aggregation")
    )
    return {
        "schema_version": "validity_gate_result_summary.v1",
        "system_validity_status": report.system_validity_status,
        "strategy_candidate_status": report.strategy_candidate_status,
        "final_strategy_status": report.final_strategy_status,
        "final_gate_status": report.metrics.get("final_gate_status"),
        "final_gate_decision": report.metrics.get("final_gate_decision"),
        "system_validity_pass": report.system_validity_pass,
        "strategy_pass": report.strategy_pass,
        "hard_fail": report.hard_fail,
        "warning": report.warning,
        "official_message": report.official_message,
        "failure_reasons": failure_reasons,
        "failure_reason_count": len(failure_reasons),
        "warnings": warning_rows,
        "warning_count": len(warning_rows),
        "key_metrics": key_metrics,
        "key_metric_count": len(key_metrics),
        "blocking_rules": final_explanation.get("blocking_rules", []),
        "warning_rules": final_explanation.get("warning_rules", []),
        "insufficient_data_rules": final_explanation.get("insufficient_data_rules", []),
        "hard_fail_rules": final_explanation.get("hard_fail_rules", []),
        "deterministic_gate": {
            "engine_id": deterministic_aggregation.get("engine_id"),
            "schema_version": deterministic_aggregation.get("schema_version"),
            "blocking_items": deterministic_aggregation.get("blocking_items", []),
            "warning_items": deterministic_aggregation.get("warning_items", []),
            "not_evaluable_items": deterministic_aggregation.get("not_evaluable_items", []),
            "reason": deterministic_aggregation.get("reason"),
        },
    }


def _summary_failure_reasons(report: ValidationGateReport) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for reason in report.hard_fail_reasons:
        rows.append(
            {
                "gate": "system_validity",
                "status": "hard_fail",
                "reason_code": "system_validity_hard_fail",
                "reason": reason,
                "affects_system": True,
                "affects_strategy": False,
            }
        )
    rows.extend(report.strategy_failure_summary)
    for row in report.rule_result_explanations:
        status = row.get("status")
        if status not in INSUFFICIENT_DATA_GATE_STATUSES:
            continue
        rows.append(
            {
                "gate": row.get("gate", row.get("rule")),
                "status": status,
                "reason_code": row.get("reason_code"),
                "reason": row.get("reason"),
                "metric": row.get("metric"),
                "value": row.get("value"),
                "threshold": row.get("threshold"),
                "operator": row.get("operator"),
                "affects_system": row.get("affects_system"),
                "affects_strategy": row.get("affects_strategy"),
            }
        )
    return _deduplicate_reason_rows(rows)


def _summary_warning_rows(report: ValidationGateReport) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for warning in report.structured_warnings:
        rows.append(dict(warning))
    for row in report.rule_result_explanations:
        if row.get("status") != "warning":
            continue
        rows.append(
            {
                "code": row.get("reason_code") or f"{row.get('gate')}_warning",
                "severity": "warning",
                "gate": row.get("gate", row.get("rule")),
                "metric": row.get("metric"),
                "message": row.get("reason"),
                "reason": row.get("reason"),
                "value": row.get("value"),
                "threshold": row.get("threshold"),
                "operator": row.get("operator"),
            }
        )
    for warning in report.warnings:
        rows.append(
            {
                "code": "warning",
                "severity": "warning",
                "gate": "validity_gate",
                "metric": None,
                "message": warning,
                "reason": warning,
            }
        )
    return _deduplicate_warning_rows(rows)


def _gate_failure_reasons(report: ValidationGateReport) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(report.validity_gate_result_summary.get("failure_reasons", []))
    rows.extend(report.validity_gate_result_summary.get("warnings", []))
    for row in report.structured_pass_fail_reasons:
        if not isinstance(row, Mapping):
            continue
        if row.get("status") == "pass":
            continue
        rows.append(dict(row))

    serialized: list[dict[str, Any]] = []
    seen: set[tuple[object, object, object]] = set()
    for row in rows:
        try:
            reason = GateFailureReason.from_mapping(dict(row))
        except ValueError:
            continue
        payload = reason.to_dict()
        key = (payload.get("gate"), payload.get("reason_code"), payload.get("reason"))
        if key in seen:
            continue
        seen.add(key)
        serialized.append(_json_safe(payload))
    return serialized


def _structured_gate_failure_report(report: ValidationGateReport) -> dict[str, Any]:
    """Group non-passing gate outcomes by gate with severity and metric context."""

    reasons = report.gate_failure_reasons
    grouped: dict[str, list[dict[str, Any]]] = {}
    for reason in reasons:
        gate = str(reason.get("gate") or reason.get("rule") or "unknown_gate")
        grouped.setdefault(gate, []).append(dict(reason))

    gate_rows: list[dict[str, Any]] = []
    for gate, gate_reasons in grouped.items():
        ordered_reasons = sorted(
            gate_reasons,
            key=lambda row: _severity_rank(row.get("severity") or row.get("status")),
            reverse=True,
        )
        top_reason = ordered_reasons[0] if ordered_reasons else {}
        related_metrics = _gate_failure_related_metrics(ordered_reasons)
        statuses = _deduplicate_preserving_order(
            str(row.get("status"))
            for row in ordered_reasons
            if str(row.get("status") or "").strip()
        )
        severities = _deduplicate_preserving_order(
            str(row.get("severity") or row.get("status"))
            for row in ordered_reasons
            if str(row.get("severity") or row.get("status") or "").strip()
        )
        gate_rows.append(
            {
                "gate": gate,
                "status": top_reason.get("status"),
                "severity": top_reason.get("severity") or top_reason.get("status"),
                "statuses": statuses,
                "severities": severities,
                "reason_count": len(ordered_reasons),
                "top_reason_code": top_reason.get("reason_code"),
                "top_reason": top_reason.get("reason"),
                "affects_system": any(
                    row.get("affects_system") is True for row in ordered_reasons
                ),
                "affects_strategy": any(
                    row.get("affects_strategy") is not False for row in ordered_reasons
                ),
                "related_metrics": related_metrics,
                "reasons": ordered_reasons,
            }
        )

    gate_rows.sort(
        key=lambda row: (
            _severity_rank(row.get("severity")),
            int(row.get("reason_count") or 0),
            str(row.get("gate") or ""),
        ),
        reverse=True,
    )
    severity_counts: dict[str, int] = {}
    for reason in reasons:
        severity = str(reason.get("severity") or reason.get("status") or "unknown")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    return {
        "schema_version": "structured_gate_failure_report.v1",
        "system_validity_status": report.system_validity_status,
        "strategy_candidate_status": report.strategy_candidate_status,
        "system_validity_pass": report.system_validity_pass,
        "strategy_pass": report.strategy_pass,
        "failed_gate_count": len(gate_rows),
        "reason_count": len(reasons),
        "severity_counts": severity_counts,
        "gates": gate_rows,
    }


def _gate_failure_related_metrics(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    seen: set[tuple[object, object, object, object]] = set()
    for row in rows:
        metric = row.get("metric")
        value = row.get("value")
        threshold = row.get("threshold")
        operator = row.get("operator")
        if metric is None and value is None and threshold is None and operator is None:
            continue
        key = (metric, value, threshold, operator)
        if key in seen:
            continue
        seen.add(key)
        metrics.append(
            {
                "metric": metric,
                "value": value,
                "threshold": threshold,
                "operator": operator,
            }
        )
    return metrics


def _severity_rank(severity: object) -> int:
    return {
        "hard_fail": 5,
        "fail": 4,
        "insufficient_data": 3,
        "not_evaluable": 2,
        "warning": 1,
    }.get(str(severity or ""), 0)


def _deduplicate_reason_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    deduplicated: list[dict[str, Any]] = []
    seen: set[tuple[object, object, object]] = set()
    for row in rows:
        key = (row.get("gate"), row.get("reason_code"), row.get("reason"))
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(dict(row))
    return deduplicated


def _deduplicate_warning_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    deduplicated: list[dict[str, Any]] = []
    seen: set[tuple[object, object, object]] = set()
    for row in rows:
        key = (row.get("gate"), row.get("code"), row.get("message"))
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(dict(row))
    return deduplicated


def _gate_result_metadata(result: Mapping[str, Any]) -> dict[str, Any]:
    reason_metadata = result.get("reason_metadata")
    if isinstance(reason_metadata, Mapping):
        return dict(reason_metadata)
    collapse_check = result.get("collapse_check")
    if isinstance(collapse_check, Mapping):
        return dict(collapse_check)
    return {}


def _gate_result_reason_code(
    result: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> object:
    collapse_check = result.get("collapse_check")
    if not isinstance(collapse_check, Mapping):
        collapse_check = {}
    return (
        result.get("reason_code")
        or result.get("collapse_reason_code")
        or collapse_check.get("reason_code")
        or metadata.get("code")
    )


def _gate_status_passed(status: str) -> bool | None:
    normalized = DEFAULT_GATE_STATUS_POLICY.normalize(status)
    if normalized == "pass":
        return True
    if normalized in {"fail", "hard_fail"}:
        return False
    return None


def _strategy_failure_summary(gate_results: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gate_name, result in gate_results.items():
        if not isinstance(result, Mapping):
            continue
        if result.get("status") != "fail":
            continue
        if result.get("affects_strategy", True) is False:
            continue

        reason_metadata = result.get("reason_metadata")
        if not isinstance(reason_metadata, Mapping):
            reason_metadata = {}
        collapse_check = result.get("collapse_check")
        if not isinstance(collapse_check, Mapping):
            collapse_check = {}
        metadata = reason_metadata or collapse_check
        reason_code = (
            result.get("reason_code")
            or result.get("collapse_reason_code")
            or collapse_check.get("reason_code")
            or metadata.get("code")
        )
        collapse_status = collapse_check.get("status")
        collapse_reason = collapse_check.get("reason") if collapse_status else None
        rows.append(
            {
                "gate": gate_name,
                "status": result.get("status"),
                "reason": result.get("reason"),
                "reason_code": reason_code,
                "collapse_status": collapse_status,
                "collapse_reason": collapse_reason,
                "metric": metadata.get("metric"),
                "value": metadata.get("value"),
                "threshold": metadata.get("threshold"),
                "operator": metadata.get("operator"),
            }
        )
    return rows


def _spy_baseline_metrics(
    predictions: pd.DataFrame,
    equity_curve: pd.DataFrame,
    benchmark_ticker: str,
    return_column: str = "forward_return_1",
    *,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    benchmark_return_series: pd.DataFrame | None = None,
    evaluation_dates: pd.Series | None = None,
) -> dict[str, Any]:
    if benchmark_return_series is not None and not benchmark_return_series.empty:
        return _market_benchmark_metrics_from_return_series(
            benchmark_return_series,
            benchmark_ticker=benchmark_ticker,
            evaluation_dates=evaluation_dates,
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
        )
    if not equity_curve.empty and "benchmark_return" in equity_curve:
        return _market_benchmark_metrics(
            equity_curve["benchmark_return"],
            _date_series(equity_curve),
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
            turnover=(
                equity_curve["benchmark_turnover"]
                if "benchmark_turnover" in equity_curve
                else None
            ),
        )
    if predictions.empty or return_column not in predictions:
        return _empty_market_benchmark_metrics(cost_bps, slippage_bps)
    frame = predictions[
        predictions["ticker"].astype(str).str.upper() == benchmark_ticker.upper()
    ]
    if frame.empty:
        return _empty_market_benchmark_metrics(cost_bps, slippage_bps)
    frame = frame.copy()
    frame["date"] = _normalized_date_series(frame["date"])
    frame = (
        frame.dropna(subset=["date"])
        .drop_duplicates("date", keep="last")
        .sort_values("date")
    )
    dates = _normalize_metric_dates(evaluation_dates)
    if not dates.empty:
        frame = pd.DataFrame({"date": dates}).merge(
            frame[["date", return_column]],
            on="date",
            how="left",
        )
    return _market_benchmark_metrics(
        frame[return_column],
        frame["date"],
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
    )


def _market_benchmark_metrics_from_return_series(
    benchmark_return_series: pd.DataFrame,
    *,
    benchmark_ticker: str,
    evaluation_dates: pd.Series | None,
    cost_bps: float,
    slippage_bps: float,
) -> dict[str, Any]:
    required = {"date", "benchmark_return"}
    if not required.issubset(benchmark_return_series.columns):
        return _empty_market_benchmark_metrics(cost_bps, slippage_bps)

    columns = ["date", "benchmark_return"]
    if "benchmark_ticker" in benchmark_return_series:
        columns.append("benchmark_ticker")
    frame = benchmark_return_series[columns].copy()
    if "benchmark_ticker" in frame:
        frame["benchmark_ticker"] = frame["benchmark_ticker"].map(_normalize_ticker)
        frame = frame[frame["benchmark_ticker"] == _normalize_ticker(benchmark_ticker)]
    frame["date"] = _normalized_date_series(frame["date"])
    frame["benchmark_return"] = pd.to_numeric(frame["benchmark_return"], errors="coerce")
    frame = frame.dropna(subset=["date"]).drop_duplicates("date", keep="last")
    dates = _normalize_metric_dates(evaluation_dates)
    if not dates.empty:
        frame = pd.DataFrame({"date": dates}).merge(frame, on="date", how="left")
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame.empty:
        return _empty_market_benchmark_metrics(cost_bps, slippage_bps)
    return _market_benchmark_metrics(
        frame["benchmark_return"],
        frame["date"],
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
    )


def _market_benchmark_metrics(
    gross_returns: object,
    dates: pd.Series | None,
    *,
    cost_bps: float,
    slippage_bps: float,
    turnover: object | None = None,
) -> dict[str, Any]:
    gross = pd.to_numeric(pd.Series(gross_returns), errors="coerce")
    if gross.empty:
        return _empty_market_benchmark_metrics(cost_bps, slippage_bps)
    if turnover is None:
        traded = _buy_and_hold_baseline_turnover(gross)
    else:
        traded = pd.to_numeric(pd.Series(turnover), errors="coerce").fillna(0.0)
    cost_adjusted = calculate_cost_adjusted_returns(
        gross,
        traded,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
    )
    net_returns = cost_adjusted["cost_adjusted_return"]
    metrics = _return_series_metrics(net_returns, dates)
    gross_metrics = _return_series_metrics(gross, dates)
    return {
        **metrics,
        "gross_cagr": gross_metrics["cagr"],
        "gross_cumulative_return": _compound_return(gross),
        "cost_adjusted_cumulative_return": _compound_return(net_returns),
        "average_daily_turnover": float(cost_adjusted["turnover"].mean())
        if not cost_adjusted.empty
        else 0.0,
        "transaction_cost_return": float(cost_adjusted["transaction_cost_return"].sum()),
        "slippage_cost_return": float(cost_adjusted["slippage_cost_return"].sum()),
        "total_cost_return": float(cost_adjusted["total_cost_return"].sum()),
        "cost_bps": float(cost_bps),
        "slippage_bps": float(slippage_bps),
        "rebalance_assumption": "single_ticker_buy_and_hold_with_initial_entry",
        "return_timing": "signal_date_returns_apply_to_configured_forward_return_horizon",
    }


def _empty_market_benchmark_metrics(cost_bps: float, slippage_bps: float) -> dict[str, Any]:
    return {
        **_empty_metrics(),
        "gross_cagr": 0.0,
        "gross_cumulative_return": 0.0,
        "cost_adjusted_cumulative_return": 0.0,
        "average_daily_turnover": 0.0,
        "transaction_cost_return": 0.0,
        "slippage_cost_return": 0.0,
        "total_cost_return": 0.0,
        "cost_bps": float(cost_bps),
        "slippage_bps": float(slippage_bps),
        "rebalance_assumption": "single_ticker_buy_and_hold_with_initial_entry",
        "return_timing": "signal_date_returns_apply_to_configured_forward_return_horizon",
    }


def _buy_and_hold_baseline_turnover(gross_returns: pd.Series) -> pd.Series:
    turnover = pd.Series(0.0, index=gross_returns.index, dtype=float)
    valid_returns = gross_returns.notna()
    if bool(valid_returns.any()):
        turnover.iloc[int(np.flatnonzero(valid_returns.to_numpy())[0])] = 1.0
    return turnover


def _equal_weight_baseline_metrics(
    predictions: pd.DataFrame,
    return_column: str = "forward_return_1",
    *,
    ticker_universe: tuple[str, ...] = (),
    evaluation_dates: pd.Series | None = None,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    baseline_return_series: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if baseline_return_series is not None and not baseline_return_series.empty:
        return _equal_weight_baseline_metrics_from_return_series(
            baseline_return_series,
            return_column=return_column,
            ticker_universe=ticker_universe,
            evaluation_dates=evaluation_dates,
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
        )

    required = {"date", "ticker", return_column}
    if not predictions.empty and required.issubset(predictions.columns):
        return _equal_weight_baseline_metrics_from_predictions(
            predictions,
            return_column,
            ticker_universe=ticker_universe,
            evaluation_dates=evaluation_dates,
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
        )
    return _empty_equal_weight_metrics(ticker_universe, cost_bps, slippage_bps)


def _equal_weight_baseline_metrics_from_predictions(
    predictions: pd.DataFrame,
    return_column: str,
    *,
    ticker_universe: tuple[str, ...],
    evaluation_dates: pd.Series | None,
    cost_bps: float,
    slippage_bps: float,
) -> dict[str, Any]:
    frame = predictions[["date", "ticker", return_column]].copy()
    frame["date"] = _normalized_date_series(frame["date"])
    frame["ticker"] = frame["ticker"].map(_normalize_ticker)
    frame[return_column] = pd.to_numeric(frame[return_column], errors="coerce")
    frame = frame.dropna(subset=["date"])
    frame = frame[frame["ticker"].astype(bool)]

    universe = ticker_universe or _normalize_ticker_sequence(frame["ticker"])
    if not universe:
        return _empty_equal_weight_metrics(universe, cost_bps, slippage_bps)
    frame = frame[frame["ticker"].isin(set(universe))]

    dates = _normalize_metric_dates(evaluation_dates)
    if dates.empty:
        dates = _normalize_metric_dates(frame["date"])
    if dates.empty:
        return _empty_equal_weight_metrics(universe, cost_bps, slippage_bps)

    rows: list[dict[str, Any]] = []
    previous_weights: dict[str, float] = {}
    for date_value in dates:
        day = frame[frame["date"] == date_value].dropna(subset=[return_column])
        returns_by_ticker = (
            day.drop_duplicates("ticker", keep="last")
            .set_index("ticker")[return_column]
            .to_dict()
        )
        available = [ticker for ticker in universe if ticker in returns_by_ticker]
        target_weights = {ticker: 1.0 / len(available) for ticker in available} if available else {}
        turnover = calculate_portfolio_turnover(previous_weights, target_weights)
        gross_return = (
            sum(target_weights[ticker] * float(returns_by_ticker[ticker]) for ticker in available)
            if available
            else np.nan
        )
        rows.append(
            {
                "date": date_value,
                "gross_return": gross_return,
                "turnover": turnover,
                "constituent_count": len(available),
                "incomplete_ticker_universe": len(available) < len(universe),
                "missing_equal_weight_return": not available,
            }
        )
        previous_weights = target_weights

    return _finalize_equal_weight_baseline_metrics(
        pd.DataFrame(rows),
        universe,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
    )


def _equal_weight_baseline_metrics_from_return_series(
    baseline_return_series: pd.DataFrame,
    *,
    return_column: str,
    ticker_universe: tuple[str, ...],
    evaluation_dates: pd.Series | None,
    cost_bps: float,
    slippage_bps: float,
) -> dict[str, Any]:
    frame = baseline_return_series.copy()
    if "date" not in frame or "equal_weight_return" not in frame:
        return _empty_equal_weight_metrics(ticker_universe, cost_bps, slippage_bps)
    frame = _filter_baseline_return_series_to_definition(frame, return_column)
    if frame.empty:
        return _empty_equal_weight_metrics(ticker_universe, cost_bps, slippage_bps)
    frame["date"] = _normalized_date_series(frame["date"])
    frame["gross_return"] = pd.to_numeric(frame["equal_weight_return"], errors="coerce")
    frame = frame.dropna(subset=["date"]).drop_duplicates("date", keep="last")
    dates = _normalize_metric_dates(evaluation_dates)
    if not dates.empty:
        frame = pd.DataFrame({"date": dates}).merge(frame, on="date", how="left")
    frame = frame.sort_values("date").reset_index(drop=True)
    if "constituent_count" not in frame:
        frame["constituent_count"] = len(ticker_universe)
    frame["constituent_count"] = (
        pd.to_numeric(frame["constituent_count"], errors="coerce").fillna(0).astype(int)
    )
    frame["turnover"] = _turnover_from_constituent_counts(frame["constituent_count"])
    frame["incomplete_ticker_universe"] = frame.get("incomplete_ticker_universe", False)
    frame["missing_equal_weight_return"] = frame["gross_return"].isna()
    universe = ticker_universe or tuple(
        f"EQUAL_WEIGHT_{idx + 1}"
        for idx in range(int(frame["constituent_count"].max()))
    )
    return _finalize_equal_weight_baseline_metrics(
        frame[
            [
                "date",
                "gross_return",
                "turnover",
                "constituent_count",
                "incomplete_ticker_universe",
                "missing_equal_weight_return",
            ]
        ],
        universe,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
    )


def _filter_baseline_return_series_to_definition(
    frame: pd.DataFrame,
    return_column: str,
) -> pd.DataFrame:
    filtered = frame.copy()
    if "return_column" in filtered:
        filtered = filtered[filtered["return_column"].astype(str) == str(return_column)]
    if "return_horizon" in filtered:
        return_horizon = _horizon_from_target(return_column) or 1
        horizons = pd.to_numeric(filtered["return_horizon"], errors="coerce")
        filtered = filtered[horizons == return_horizon]
    return filtered.copy()


def _finalize_equal_weight_baseline_metrics(
    frame: pd.DataFrame,
    universe: tuple[str, ...],
    *,
    cost_bps: float,
    slippage_bps: float,
) -> dict[str, Any]:
    if frame.empty:
        return _empty_equal_weight_metrics(universe, cost_bps, slippage_bps)

    gross_returns = pd.to_numeric(frame["gross_return"], errors="coerce")
    turnover = pd.to_numeric(frame["turnover"], errors="coerce").fillna(0.0)
    cost_adjusted = calculate_cost_adjusted_returns(
        gross_returns,
        turnover,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
    )
    dates = frame["date"]
    net_returns = cost_adjusted["cost_adjusted_return"]
    metrics = _return_series_metrics(net_returns, dates)
    gross_metrics = _return_series_metrics(gross_returns, dates)
    constituent_count = pd.to_numeric(frame["constituent_count"], errors="coerce").fillna(0).astype(int)
    return {
        **metrics,
        "universe_tickers": list(universe),
        "expected_constituent_count": len(universe),
        "min_constituent_count": int(constituent_count.min()) if not constituent_count.empty else 0,
        "max_constituent_count": int(constituent_count.max()) if not constituent_count.empty else 0,
        "incomplete_rebalance_count": int(pd.Series(frame["incomplete_ticker_universe"]).fillna(False).sum()),
        "missing_return_count": int(pd.Series(frame["missing_equal_weight_return"]).fillna(False).sum()),
        "rebalance_count": int(len(frame)),
        "rebalance_assumption": "equal_weight_rebalanced_on_strategy_evaluation_dates",
        "return_timing": "signal_date_returns_apply_to_configured_forward_return_horizon",
        "gross_cagr": gross_metrics["cagr"],
        "gross_cumulative_return": _compound_return(gross_returns),
        "cost_adjusted_cumulative_return": _compound_return(net_returns),
        "average_daily_turnover": float(turnover.mean()) if not turnover.empty else 0.0,
        "transaction_cost_return": float(cost_adjusted["transaction_cost_return"].sum()),
        "slippage_cost_return": float(cost_adjusted["slippage_cost_return"].sum()),
        "total_cost_return": float(cost_adjusted["total_cost_return"].sum()),
        "cost_bps": float(cost_bps),
        "slippage_bps": float(slippage_bps),
    }


def _return_series_metrics(
    returns: pd.Series,
    dates: pd.Series | None = None,
) -> dict[str, float | int | str | None]:
    return calculate_return_series_metrics(returns, dates).to_dict()


def _empty_metrics() -> dict[str, float | int | str | None]:
    return {
        "cagr": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "cumulative_return": 0.0,
        "observations": 0,
        "evaluation_start": None,
        "evaluation_end": None,
    }


def _empty_equal_weight_metrics(
    ticker_universe: tuple[str, ...],
    cost_bps: float,
    slippage_bps: float,
) -> dict[str, Any]:
    return {
        **_empty_metrics(),
        "universe_tickers": list(ticker_universe),
        "expected_constituent_count": len(ticker_universe),
        "min_constituent_count": 0,
        "max_constituent_count": 0,
        "incomplete_rebalance_count": 0,
        "missing_return_count": 0,
        "rebalance_count": 0,
        "rebalance_assumption": "equal_weight_rebalanced_on_strategy_evaluation_dates",
        "return_timing": "signal_date_returns_apply_to_configured_forward_return_horizon",
        "gross_cagr": 0.0,
        "gross_cumulative_return": 0.0,
        "cost_adjusted_cumulative_return": 0.0,
        "average_daily_turnover": 0.0,
        "transaction_cost_return": 0.0,
        "slippage_cost_return": 0.0,
        "total_cost_return": 0.0,
        "cost_bps": float(cost_bps),
        "slippage_bps": float(slippage_bps),
    }


def _strategy_universe_tickers(config: object | None, predictions: pd.DataFrame) -> tuple[str, ...]:
    configured = getattr(config, "tickers", None)
    configured_universe = (
        _normalize_ticker_sequence(configured)
        if configured is not None
        else ()
    )
    predicted_universe = (
        _normalize_ticker_sequence(predictions["ticker"])
        if not predictions.empty and "ticker" in predictions
        else ()
    )
    if configured_universe and predicted_universe:
        predicted_set = set(predicted_universe)
        evaluated_universe = tuple(
            ticker for ticker in configured_universe if ticker in predicted_set
        )
        if evaluated_universe:
            return evaluated_universe
    return configured_universe or predicted_universe


def _baseline_evaluation_dates(equity_curve: pd.DataFrame, predictions: pd.DataFrame) -> pd.Series:
    if not equity_curve.empty and "date" in equity_curve:
        dates = _normalize_metric_dates(equity_curve["date"])
        if not dates.empty:
            return dates
    if not predictions.empty and "date" in predictions:
        return _normalize_metric_dates(predictions["date"])
    return pd.Series(dtype="datetime64[ns]")


def _stage1_baseline_comparison_inputs(
    provided_inputs: Iterable[BaselineComparisonInput] | None,
    equity_curve: pd.DataFrame,
    predictions: pd.DataFrame,
    config: object | None,
    *,
    benchmark_ticker: str,
    return_column: str,
) -> tuple[BaselineComparisonInput, ...]:
    if provided_inputs is not None:
        return tuple(provided_inputs)
    return build_stage1_baseline_comparison_inputs(
        _baseline_input_evaluation_window(equity_curve, predictions, config),
        None,
        benchmark_ticker=benchmark_ticker,
        tickers=_strategy_universe_tickers(config, predictions),
        return_column=return_column,
        return_horizon=_horizon_from_target(return_column) or 1,
        cost_bps=_baseline_cost_assumption(config, equity_curve, "cost_bps"),
        slippage_bps=_baseline_cost_assumption(config, equity_curve, "slippage_bps"),
    )


def _baseline_input_evaluation_window(
    equity_curve: pd.DataFrame,
    predictions: pd.DataFrame,
    config: object | None,
) -> StrategyEvaluationWindow | None:
    dates = _baseline_evaluation_dates(equity_curve, predictions)
    start, end = _date_bounds(dates)
    if start is not None and end is not None:
        return StrategyEvaluationWindow.from_bounds(start, end)
    configured_start = getattr(config, "start", None)
    configured_end = getattr(config, "end", None)
    if configured_start is None or configured_end is None:
        return None
    try:
        return StrategyEvaluationWindow.from_bounds(configured_start, configured_end)
    except ValueError:
        return None


def _baseline_cost_assumption(
    config: object | None,
    equity_curve: pd.DataFrame,
    name: str,
) -> float:
    configured = _finite_or_none(getattr(config, name, None))
    if configured is not None:
        return configured
    if not equity_curve.empty and name in equity_curve:
        values = pd.to_numeric(equity_curve[name], errors="coerce").dropna()
        if not values.empty:
            return float(values.iloc[0])
    return 0.0


def _strategy_return_column(
    equity_curve: pd.DataFrame,
    config: object | None,
    *,
    default_return_column: str = "forward_return_1",
) -> str:
    if not equity_curve.empty and "realized_return_column" in equity_curve:
        values = equity_curve["realized_return_column"].dropna().astype(str).str.strip()
        values = values[values.astype(bool)]
        if not values.empty:
            return str(values.iloc[0])
    configured = str(getattr(config, "prediction_target_column", default_return_column))
    configured_horizon = _horizon_from_target(configured)
    if _is_diagnostic_only_horizon(configured_horizon):
        return default_return_column
    return configured


def _normalize_ticker_sequence(values: object) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        ticker = _normalize_ticker(value)
        if ticker and ticker not in seen:
            normalized.append(ticker)
            seen.add(ticker)
    return tuple(normalized)


def _normalize_ticker(value: object) -> str:
    return str(value).strip().upper()


def _normalize_metric_dates(values: object) -> pd.Series:
    dates = _normalized_date_series(values).dropna().drop_duplicates().sort_values()
    return dates.reset_index(drop=True)


def _normalized_date_series(values: object) -> pd.Series:
    dates = pd.Series(pd.to_datetime(values, errors="coerce"))
    if dates.empty:
        return dates
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    return dates.dt.normalize()


def _turnover_from_constituent_counts(counts: pd.Series) -> pd.Series:
    previous_count = 0
    turnovers: list[float] = []
    for value in counts:
        current_count = int(value) if pd.notna(value) else 0
        if current_count <= 0:
            turnovers.append(1.0 if previous_count > 0 else 0.0)
        elif previous_count <= 0:
            turnovers.append(1.0)
        elif current_count == previous_count:
            turnovers.append(0.0)
        else:
            shared = min(previous_count, current_count)
            exited = max(previous_count - current_count, 0)
            entered = max(current_count - previous_count, 0)
            turnover = (
                shared * abs((1.0 / current_count) - (1.0 / previous_count))
                + exited * (1.0 / previous_count)
                + entered * (1.0 / current_count)
            )
            turnovers.append(float(turnover))
        previous_count = current_count
    return pd.Series(turnovers, index=counts.index, dtype=float)


def _format_metric(value: object) -> str:
    numeric = _finite_or_none(value)
    if numeric is None:
        return ""
    return f"{numeric:.4f}"


def _format_side_by_side_metric_value(metric: str, value: object) -> str:
    if metric in SIDE_BY_SIDE_NUMERIC_METRICS:
        return _format_metric(value)
    if metric in SIDE_BY_SIDE_COUNT_METRICS:
        return _format_count_metric(value)
    return _markdown_cell(value)


def _side_by_side_column_is_numeric(
    side_by_side_metric_comparison: list[dict[str, Any]],
    name: str,
) -> bool:
    numeric_rows = [
        row
        for row in side_by_side_metric_comparison
        if row.get("metric") in SIDE_BY_SIDE_NUMERIC_METRICS | SIDE_BY_SIDE_COUNT_METRICS
    ]
    if not numeric_rows:
        return False
    return any(_finite_or_none(row.get(name)) is not None for row in numeric_rows)


def _comparison_display_name(name: str) -> str:
    if name == "strategy":
        return "Strategy"
    if name == "equal_weight":
        return "Equal Weight"
    return name


def _format_count_metric(value: object) -> str:
    numeric = _finite_or_none(value)
    if numeric is None:
        return ""
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.4f}"


def _format_sequence(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Iterable):
        return ", ".join(str(item) for item in value)
    return ""


def _sequence_length(value: object) -> int:
    if isinstance(value, Mapping):
        return len(value)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return len(tuple(value))
    return 0


def _markdown_cell(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("\n", " ").replace("|", "\\|")


def _html_text(value: object) -> str:
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def _html_table(
    title: str,
    headers: Iterable[object],
    rows: Iterable[Iterable[object]],
) -> list[str]:
    row_list = [tuple(row) for row in rows]
    if not row_list:
        return []
    lines = [
        "<section>",
        f"<h2>{_html_text(title)}</h2>",
        "<table>",
        "<thead>",
        "<tr>",
    ]
    lines.extend(f"<th>{_html_text(header)}</th>" for header in headers)
    lines.extend(["</tr>", "</thead>", "<tbody>"])
    for row in row_list:
        lines.append("<tr>")
        lines.extend(f"<td>{_html_text(cell)}</td>" for cell in row)
        lines.append("</tr>")
    lines.extend(["</tbody>", "</table>", "</section>"])
    return lines


def _html_list(title: str, items: Iterable[object]) -> list[str]:
    item_list = list(items)
    if not item_list:
        return []
    lines = ["<section>", f"<h2>{_html_text(title)}</h2>", "<ul>"]
    lines.extend(f"<li>{_html_text(item)}</li>" for item in item_list)
    lines.extend(["</ul>", "</section>"])
    return lines


def _html_json_section(title: str, payload: object) -> list[str]:
    return [
        "<section>",
        f"<h2>{_html_text(title)}</h2>",
        "<pre><code>",
        _html_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2)),
        "</code></pre>",
        "</section>",
    ]


def _date_series(frame: pd.DataFrame) -> pd.Series | None:
    if "date" not in frame:
        return None
    return frame["date"]


def _date_bounds(dates: pd.Series | None) -> tuple[str | None, str | None]:
    if dates is None:
        return None, None
    values = pd.to_datetime(dates, errors="coerce").dropna()
    if values.empty:
        return None, None
    return values.min().date().isoformat(), values.max().date().isoformat()


def _horizon_from_target(target_column: str) -> int | None:
    prefix = "forward_return_"
    if not target_column.startswith(prefix):
        return None
    suffix = target_column.removeprefix(prefix)
    try:
        return int(suffix)
    except ValueError:
        return None


def _target_column_for_horizon(horizon: int) -> str:
    return f"forward_return_{max(int(horizon), 1)}"


def _horizon_from_label(label: str) -> int | None:
    suffix = str(label).strip().lower().removesuffix("d")
    try:
        horizon = int(suffix)
    except ValueError:
        return None
    return horizon if horizon >= 1 else None


def _horizon_output_label(horizon: int, required_horizon: int) -> str:
    if horizon == required_horizon:
        return "required"
    if _is_diagnostic_only_horizon(horizon):
        return "diagnostic"
    if _is_robustness_horizon(horizon):
        return "robustness"
    return "supplemental"


def _horizon_output_role(horizon: int, affects_pass_fail: bool) -> str:
    if _is_diagnostic_only_horizon(horizon):
        return "diagnostic"
    if affects_pass_fail:
        return "decision"
    if _is_robustness_horizon(horizon):
        return "robustness"
    return "supplemental"


def _is_diagnostic_only_horizon(horizon: int | None) -> bool:
    if horizon is None:
        return False
    return f"{horizon}d" in DIAGNOSTIC_ONLY_HORIZONS


def _is_robustness_horizon(horizon: int | None) -> bool:
    if horizon is None:
        return False
    return f"{horizon}d" in ROBUSTNESS_HORIZONS


def _spearman(left: pd.Series, right: pd.Series) -> float | None:
    left_numeric = pd.to_numeric(left, errors="coerce")
    right_numeric = pd.to_numeric(right, errors="coerce")
    common = left_numeric.notna() & right_numeric.notna()
    if common.sum() < 2:
        return None
    left_rank = left_numeric.loc[common].rank()
    right_rank = right_numeric.loc[common].rank()
    if left_rank.nunique() <= 1 or right_rank.nunique() <= 1:
        return None
    value = left_rank.corr(right_rank)
    return float(value) if pd.notna(value) else None


def _count_oos_folds(validation_summary: pd.DataFrame) -> int:
    if validation_summary.empty or "is_oos" not in validation_summary:
        return 0
    fold_rows = _walk_forward_fold_rows(validation_summary)
    if fold_rows.empty or "is_oos" not in fold_rows:
        return 0
    oos_rows = fold_rows[fold_rows["is_oos"].fillna(False).astype(bool)]
    if oos_rows.empty:
        return 0
    if "fold" in oos_rows:
        return int(oos_rows["fold"].nunique(dropna=True))
    return int(len(oos_rows))


def _count_validation_folds(validation_summary: pd.DataFrame) -> int:
    if validation_summary.empty:
        return 0
    fold_rows = _walk_forward_fold_rows(validation_summary)
    if fold_rows.empty:
        return 0
    if "fold" in fold_rows:
        return int(fold_rows["fold"].nunique(dropna=True))
    return int(len(fold_rows))


def _insufficient_data_reasons(gate_results: dict[str, dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for name, result in gate_results.items():
        if result.get("affects_insufficient_data") is False:
            continue
        if result.get("status") not in INSUFFICIENT_DATA_GATE_STATUSES:
            continue
        reason = str(result.get("reason", "")).strip()
        reasons.append(f"{name}: {reason}" if reason else str(name))
    return reasons


def _metric(metrics: object, name: str) -> float:
    if isinstance(metrics, Mapping):
        value = metrics.get(name, 0.0)
    else:
        try:
            value = getattr(metrics, name)
        except AttributeError:
            value = 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (str, bytes, dict, list, tuple)) else False:
        return None
    return value
