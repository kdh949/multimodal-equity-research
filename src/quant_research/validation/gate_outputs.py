from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Literal

SYSTEM_VALIDITY_GATE_OUTPUT_SCHEMA_VERSION = "system_validity_gate_output.v1"
SYSTEM_VALIDITY_GATE_OUTPUT_SCHEMA_ID = "stage1_system_validity_gate_outputs"
SYSTEM_VALIDITY_GATE_REPORT_SCHEMA_VERSION = "system_validity_gate_report.v1"
SYSTEM_VALIDITY_GATE_REPORT_SCHEMA_ID = "stage1_system_validity_gate_report"
GATE_FAILURE_REASON_SCHEMA_VERSION = "gate_failure_reason.v1"
GATE_FAILURE_REASON_SCHEMA_ID = "stage1_gate_failure_reason"
SERIALIZABLE_VALIDITY_GATE_REPORT_SCHEMA_VERSION = "serializable_validity_gate_report.v1"
SERIALIZABLE_VALIDITY_GATE_REPORT_SCHEMA_ID = "stage1_serializable_validity_gate_report"
SYSTEM_VALIDITY_GATE_REQUIRED_OUTPUT_SECTIONS: tuple[str, ...] = (
    "statuses",
    "metrics",
    "validation_result_schemas",
    "gate_results",
    "evidence",
    "artifact_contract",
    "scope_bounds",
)

OutputFieldRequirement = Literal["required", "optional"]


class GateRuleStatus(StrEnum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    HARD_FAIL = "hard_fail"
    INSUFFICIENT_DATA = "insufficient_data"
    NOT_EVALUABLE = "not_evaluable"
    SKIPPED = "skipped"


GATE_RULE_STATUSES: tuple[str, ...] = tuple(status.value for status in GateRuleStatus)


@dataclass(frozen=True, slots=True)
class GateOutputField:
    name: str
    dtype: str
    requirement: OutputFieldRequirement = "required"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("output field name must not be blank")
        if not self.dtype.strip():
            raise ValueError("output field dtype must not be blank")
        if self.requirement not in {"required", "optional"}:
            raise ValueError("output field requirement must be required or optional")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GateOutputSectionSchema:
    section_id: str
    description: str
    fields: tuple[GateOutputField, ...]

    def __post_init__(self) -> None:
        if not self.section_id.strip():
            raise ValueError("output section_id must not be blank")
        if not self.description.strip():
            raise ValueError("output section description must not be blank")
        if not self.fields:
            raise ValueError("output section must include at least one field")
        names = [field.name for field in self.fields]
        if len(set(names)) != len(names):
            raise ValueError(f"output section {self.section_id} contains duplicate fields")

    def required_fields(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields if field.requirement == "required")

    def optional_fields(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields if field.requirement == "optional")

    def to_dict(self) -> dict[str, object]:
        return {
            "section_id": self.section_id,
            "description": self.description,
            "required_fields": list(self.required_fields()),
            "optional_fields": list(self.optional_fields()),
            "fields": [field.to_dict() for field in self.fields],
        }


@dataclass(frozen=True, slots=True)
class GateValidationResultSchema:
    result_id: str
    description: str
    status_enum: tuple[str, ...]
    fields: tuple[GateOutputField, ...]
    row_grain: str = "validation result"

    def __post_init__(self) -> None:
        if not self.result_id.strip():
            raise ValueError("validation result_id must not be blank")
        if not self.description.strip():
            raise ValueError("validation result description must not be blank")
        if not self.status_enum:
            raise ValueError("validation result status_enum must not be empty")
        if not self.fields:
            raise ValueError("validation result schema must include at least one field")
        if "status" not in [field.name for field in self.fields]:
            raise ValueError("validation result schema must include status field")
        names = [field.name for field in self.fields]
        if len(set(names)) != len(names):
            raise ValueError(f"validation result {self.result_id} contains duplicate fields")

    def required_fields(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields if field.requirement == "required")

    def optional_fields(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields if field.requirement == "optional")

    def to_dict(self) -> dict[str, object]:
        return {
            "result_id": self.result_id,
            "description": self.description,
            "row_grain": self.row_grain,
            "status_enum": list(self.status_enum),
            "required_fields": list(self.required_fields()),
            "optional_fields": list(self.optional_fields()),
            "fields": [field.to_dict() for field in self.fields],
        }


@dataclass(frozen=True, slots=True)
class GateReportStructuredReasonSchema:
    reason_id: str
    description: str
    fields: tuple[GateOutputField, ...]

    def __post_init__(self) -> None:
        if not self.reason_id.strip():
            raise ValueError("reason_id must not be blank")
        if not self.description.strip():
            raise ValueError("reason description must not be blank")
        if not self.fields:
            raise ValueError("reason schema must include at least one field")
        names = [field.name for field in self.fields]
        if len(set(names)) != len(names):
            raise ValueError(f"reason schema {self.reason_id} contains duplicate fields")

    def required_fields(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields if field.requirement == "required")

    def optional_fields(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields if field.requirement == "optional")

    def to_dict(self) -> dict[str, object]:
        return {
            "reason_id": self.reason_id,
            "description": self.description,
            "required_fields": list(self.required_fields()),
            "optional_fields": list(self.optional_fields()),
            "fields": [field.to_dict() for field in self.fields],
        }


@dataclass(frozen=True, slots=True)
class GateFailureReason:
    """Serializable, structured reason for a gate that did not cleanly pass."""

    gate: str
    status: str
    reason_code: str
    reason: str
    category: str = "gate"
    passed: bool | None = False
    severity: str | None = None
    metric: str | None = None
    value: Any | None = None
    threshold: Any | None = None
    operator: str | None = None
    affects_system: bool | None = None
    affects_strategy: bool | None = None
    entity_id: str | None = None
    rule: str | None = None
    schema_version: str = GATE_FAILURE_REASON_SCHEMA_VERSION
    schema_id: str = GATE_FAILURE_REASON_SCHEMA_ID

    def __post_init__(self) -> None:
        if not self.gate.strip():
            raise ValueError("gate failure reason must include gate")
        if self.status not in GATE_RULE_STATUSES:
            allowed = ", ".join(GATE_RULE_STATUSES)
            raise ValueError(f"gate failure reason status must be one of: {allowed}")
        if self.status in {"fail", "hard_fail"}:
            if not self.reason_code.strip():
                raise ValueError("failed gate reason must include reason_code")
            if not self.reason.strip():
                raise ValueError("failed gate reason must include reason")

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> GateFailureReason:
        gate = str(row.get("gate") or row.get("rule") or row.get("entity_id") or "")
        severity = row.get("severity")
        status = str(row.get("status") or severity or "not_evaluable")
        severity = row.get("severity")
        return cls(
            gate=gate,
            status=status,
            reason_code=str(row.get("reason_code") or row.get("code") or f"{gate}_not_passed"),
            reason=str(row.get("reason") or row.get("message") or status),
            category=str(row.get("category") or "gate"),
            passed=row.get("passed"),
            severity=str(severity) if severity is not None else status,
            metric=str(row.get("metric")) if row.get("metric") is not None else None,
            value=row.get("value"),
            threshold=row.get("threshold"),
            operator=str(row.get("operator")) if row.get("operator") is not None else None,
            affects_system=row.get("affects_system"),
            affects_strategy=row.get("affects_strategy"),
            entity_id=str(row.get("entity_id")) if row.get("entity_id") is not None else None,
            rule=str(row.get("rule")) if row.get("rule") is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SerializableValidityGateReport:
    """Minimal JSON report model for automation and artifact persistence."""

    system_validity_status: str
    strategy_candidate_status: str
    system_validity_pass: bool
    strategy_pass: bool
    hard_fail: bool
    warning: bool
    official_message: str
    gate_failure_reasons: tuple[GateFailureReason, ...]
    gate_results: dict[str, Any]
    metrics: dict[str, Any]
    evidence: dict[str, Any]
    structured_gate_failure_report: dict[str, Any] = field(default_factory=dict)
    artifact_manifest: dict[str, Any] = field(default_factory=dict)
    report_path: str | None = None
    schema_version: str = SERIALIZABLE_VALIDITY_GATE_REPORT_SCHEMA_VERSION
    schema_id: str = SERIALIZABLE_VALIDITY_GATE_REPORT_SCHEMA_ID

    def __post_init__(self) -> None:
        if self.system_validity_status not in {"pass", "hard_fail", "not_evaluable"}:
            raise ValueError("system_validity_status is not a valid system gate status")
        if self.strategy_candidate_status not in {
            "pass",
            "warning",
            "fail",
            "insufficient_data",
            "not_evaluable",
        }:
            raise ValueError("strategy_candidate_status is not a valid strategy gate status")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gate_failure_reasons"] = [
            reason.to_dict() for reason in self.gate_failure_reasons
        ]
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str)


@dataclass(frozen=True, slots=True)
class SystemValidityGateReportSchema:
    schema_id: str = SYSTEM_VALIDITY_GATE_REPORT_SCHEMA_ID
    schema_version: str = SYSTEM_VALIDITY_GATE_REPORT_SCHEMA_VERSION
    status_enum: tuple[str, ...] = GATE_RULE_STATUSES
    top_level_statuses: GateOutputSectionSchema = field(
        default_factory=lambda: _report_top_level_statuses_section()
    )
    gate_result: GateOutputSectionSchema = field(
        default_factory=lambda: _report_gate_result_section()
    )
    structured_reasons: dict[str, GateReportStructuredReasonSchema] = field(
        default_factory=lambda: _report_structured_reason_schemas()
    )
    artifact_contract: dict[str, object] = field(
        default_factory=lambda: _report_schema_artifact_contract()
    )

    def __post_init__(self) -> None:
        if self.schema_version != SYSTEM_VALIDITY_GATE_REPORT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {SYSTEM_VALIDITY_GATE_REPORT_SCHEMA_VERSION!r}"
            )
        if "pass_fail_reason" not in self.structured_reasons:
            raise ValueError("structured_reasons must include pass_fail_reason")
        if "warning_reason" not in self.structured_reasons:
            raise ValueError("structured_reasons must include warning_reason")
        if "gate_failure_reason" not in self.structured_reasons:
            raise ValueError("structured_reasons must include gate_failure_reason")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "status_enum": list(self.status_enum),
            "top_level_statuses": self.top_level_statuses.to_dict(),
            "gate_result": self.gate_result.to_dict(),
            "structured_reasons": {
                key: value.to_dict()
                for key, value in self.structured_reasons.items()
            },
            "artifact_contract": dict(self.artifact_contract),
        }


@dataclass(frozen=True, slots=True)
class SystemValidityGateOutputSchema:
    schema_id: str = SYSTEM_VALIDITY_GATE_OUTPUT_SCHEMA_ID
    schema_version: str = SYSTEM_VALIDITY_GATE_OUTPUT_SCHEMA_VERSION
    required_sections: tuple[str, ...] = SYSTEM_VALIDITY_GATE_REQUIRED_OUTPUT_SECTIONS
    statuses: GateOutputSectionSchema = field(default_factory=lambda: _statuses_section())
    metrics: GateOutputSectionSchema = field(default_factory=lambda: _metrics_section())
    validation_result_schemas: dict[str, GateValidationResultSchema] = field(
        default_factory=lambda: _validation_result_schemas_section()
    )
    gate_results: GateOutputSectionSchema = field(default_factory=lambda: _gate_results_section())
    evidence: GateOutputSectionSchema = field(default_factory=lambda: _evidence_section())
    artifact_contract: dict[str, object] = field(default_factory=lambda: _artifact_contract_section())
    scope_bounds: dict[str, object] = field(default_factory=lambda: _scope_bounds_section())

    def __post_init__(self) -> None:
        if self.schema_version != SYSTEM_VALIDITY_GATE_OUTPUT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {SYSTEM_VALIDITY_GATE_OUTPUT_SCHEMA_VERSION!r}"
            )
        missing = [
            section
            for section in SYSTEM_VALIDITY_GATE_REQUIRED_OUTPUT_SECTIONS
            if not hasattr(self, section)
        ]
        if missing:
            raise ValueError(f"missing system validity gate output sections: {missing}")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "required_sections": list(self.required_sections),
            "statuses": self.statuses.to_dict(),
            "metrics": self.metrics.to_dict(),
            "validation_result_schemas": {
                key: value.to_dict()
                for key, value in self.validation_result_schemas.items()
            },
            "gate_results": self.gate_results.to_dict(),
            "evidence": self.evidence.to_dict(),
            "artifact_contract": dict(self.artifact_contract),
            "scope_bounds": dict(self.scope_bounds),
        }


def default_system_validity_gate_output_schema() -> SystemValidityGateOutputSchema:
    return SystemValidityGateOutputSchema()


def build_system_validity_gate_output_schema() -> dict[str, object]:
    return default_system_validity_gate_output_schema().to_dict()


def default_system_validity_gate_report_schema() -> SystemValidityGateReportSchema:
    return SystemValidityGateReportSchema()


def build_system_validity_gate_report_schema() -> dict[str, object]:
    return default_system_validity_gate_report_schema().to_dict()


def _field(
    name: str,
    dtype: str,
    requirement: OutputFieldRequirement = "required",
    description: str = "",
) -> GateOutputField:
    return GateOutputField(
        name=name,
        dtype=dtype,
        requirement=requirement,
        description=description,
    )


def _statuses_section() -> GateOutputSectionSchema:
    return GateOutputSectionSchema(
        section_id="statuses",
        description="Top-level deterministic validity and strategy candidate decisions.",
        fields=(
            _field("system_validity_status", "enum[pass,hard_fail,not_evaluable]"),
            _field(
                "strategy_candidate_status",
                "enum[pass,warning,fail,insufficient_data,not_evaluable]",
            ),
            _field("system_validity_pass", "boolean"),
            _field("strategy_pass", "boolean"),
            _field("hard_fail", "boolean"),
            _field("warning", "boolean"),
            _field("official_message", "string"),
        ),
    )


def _report_top_level_statuses_section() -> GateOutputSectionSchema:
    return GateOutputSectionSchema(
        section_id="top_level_statuses",
        description="Top-level hard system validity and strategy candidate decisions.",
        fields=(
            _field("system_validity_status", "enum[pass,hard_fail,not_evaluable]"),
            _field(
                "strategy_candidate_status",
                "enum[pass,warning,fail,insufficient_data,not_evaluable]",
            ),
            _field("system_validity_pass", "boolean"),
            _field("strategy_pass", "boolean"),
            _field("hard_fail", "boolean"),
            _field("warning", "boolean"),
            _field("official_message", "string"),
        ),
    )


def _report_gate_result_section() -> GateOutputSectionSchema:
    return GateOutputSectionSchema(
        section_id="gate_result",
        description="Canonical per-gate result row used by gate_results and structured reason tables.",
        fields=(
            _field("gate", "string"),
            _field("status", "GateRuleStatus"),
            _field("passed", "boolean|null"),
            _field("severity", "enum[pass,warning,fail,hard_fail,info]|null"),
            _field("reason_code", "string|null"),
            _field("reason", "string|null"),
            _field("metric", "string|null"),
            _field("value", "number|string|boolean|null"),
            _field("threshold", "number|string|object|null"),
            _field("operator", "string|null"),
            _field("affects_system", "boolean"),
            _field("affects_strategy", "boolean"),
        ),
    )


def _report_structured_reason_schemas() -> dict[str, GateReportStructuredReasonSchema]:
    return {
        "pass_fail_reason": GateReportStructuredReasonSchema(
            reason_id="pass_fail_reason",
            description="Structured PASS/WARN/FAIL reason row for each gate and model comparison rule.",
            fields=(
                _field("category", "enum[gate,model_comparison]"),
                _field("entity_id", "string|null"),
                _field("rule", "string|null"),
                _field("metric", "string|null"),
                _field("status", "GateRuleStatus"),
                _field("passed", "boolean|null"),
                _field("reason_code", "string|null"),
                _field("reason", "string|null"),
                _field("value", "number|string|boolean|null", "optional"),
                _field("threshold", "number|string|object|null", "optional"),
                _field("operator", "string|null", "optional"),
                _field("affects_strategy", "boolean|null", "optional"),
                _field("affects_system", "boolean|null", "optional"),
                _field("candidate", "string|null", "optional"),
                _field("baseline", "string|null", "optional"),
                _field("window_id", "string|null", "optional"),
                _field("candidate_value", "number|null", "optional"),
                _field("baseline_value", "number|null", "optional"),
            ),
        ),
        "warning_reason": GateReportStructuredReasonSchema(
            reason_id="warning_reason",
            description="Structured warning emitted when a non-blocking gate warning needs human review.",
            fields=(
                _field("code", "string"),
                _field("severity", "enum[warning]"),
                _field("gate", "string"),
                _field("metric", "string|null"),
                _field("message", "string"),
                _field("value", "number|string|boolean|null", "optional"),
                _field("threshold", "number|string|object|null", "optional"),
                _field("operator", "string|null", "optional"),
            ),
        ),
        "gate_failure_reason": GateReportStructuredReasonSchema(
            reason_id="gate_failure_reason",
            description="Serializable structured reason for any hard-fail, fail, insufficient-data, not-evaluable, or warning gate outcome.",
            fields=(
                _field("schema_version", "string"),
                _field("schema_id", "string"),
                _field("category", "enum[gate,model_comparison,system_validity]"),
                _field("gate", "string"),
                _field("status", "GateRuleStatus"),
                _field("passed", "boolean|null"),
                _field("severity", "enum[warning,fail,hard_fail,insufficient_data,not_evaluable]|null"),
                _field("reason_code", "string"),
                _field("reason", "string"),
                _field("metric", "string|null", "optional"),
                _field("value", "number|string|boolean|null", "optional"),
                _field("threshold", "number|string|object|null", "optional"),
                _field("operator", "string|null", "optional"),
                _field("affects_system", "boolean|null", "optional"),
                _field("affects_strategy", "boolean|null", "optional"),
                _field("entity_id", "string|null", "optional"),
                _field("rule", "string|null", "optional"),
            ),
        ),
    }


def _report_schema_artifact_contract() -> dict[str, object]:
    return {
        "embedded_in_top_level_payload": True,
        "embedded_in_metrics": True,
        "embedded_in_evidence": True,
        "pass_fail_reasons_field": "structured_pass_fail_reasons",
        "gate_failure_reasons_field": "gate_failure_reasons",
        "structured_gate_failure_report_field": "structured_gate_failure_report",
        "warnings_field": "structured_warnings",
        "gate_results_field": "gate_results",
        "serializable_report_model_field": "serializable_gate_report",
    }


def _metrics_section() -> GateOutputSectionSchema:
    return GateOutputSectionSchema(
        section_id="metrics",
        description="Canonical Stage 1 OOS, leakage, comparison, risk, and turnover metrics.",
        fields=(
            _field("fold_count", "integer"),
            _field("oos_fold_count", "integer"),
            _field("target_column", "string", description="Must resolve to forward_return_20."),
            _field("target_horizon", "integer"),
            _field("mean_rank_ic", "float|null"),
            _field("oos_rank_ic", "float|null"),
            _field("positive_fold_ratio", "float|null"),
            _field("positive_fold_ratio_threshold", "float"),
            _field("strategy_cost_adjusted_cumulative_return", "float|null"),
            _field("strategy_excess_return_vs_spy", "float|null"),
            _field("strategy_excess_return_vs_equal_weight", "float|null"),
            _field("strategy_max_drawdown", "float|null"),
            _field("strategy_turnover", "float|null"),
            _field("baseline_sample_alignment", "object"),
            _field("horizon_metrics", "object"),
            _field("system_validity_gate_criteria", "object"),
            _field("system_validity_gate_input_schema", "object"),
            _field("system_validity_gate_output_schema", "object"),
            _field("system_validity_gate_report_schema", "object", "optional"),
        ),
    )


def _validation_result_schemas_section() -> dict[str, GateValidationResultSchema]:
    return {
        "backtest": GateValidationResultSchema(
            result_id="backtest",
            description="Backtest validation result with sample alignment, costs, slippage, turnover, and baseline excess returns.",
            row_grain="backtest evaluation summary",
            status_enum=GATE_RULE_STATUSES,
            fields=(
                _field("status", "GateRuleStatus"),
                _field("passed", "boolean"),
                _field("cost_adjusted_cumulative_return", "float|null"),
                _field("excess_return_vs_spy", "float|null"),
                _field("excess_return_vs_equal_weight", "float|null"),
                _field("average_daily_turnover", "float|null"),
                _field("max_drawdown", "float|null"),
                _field("sample_alignment", "object"),
                _field("reason_code", "string", "optional"),
                _field("reason", "string", "optional"),
            ),
        ),
        "walk_forward": GateValidationResultSchema(
            result_id="walk_forward",
            description="Walk-forward validation result with target horizon, purge, embargo, and fold chronology evidence.",
            row_grain="walk-forward validation summary",
            status_enum=GATE_RULE_STATUSES,
            fields=(
                _field("status", "GateRuleStatus"),
                _field("passed", "boolean"),
                _field("fold_count", "integer"),
                _field("oos_fold_count", "integer"),
                _field("target_column", "string"),
                _field("target_horizon", "integer"),
                _field("purge_periods", "integer|null"),
                _field("embargo_periods", "integer|null"),
                _field("reason_code", "string", "optional"),
                _field("reason", "string", "optional"),
            ),
        ),
        "out_of_sample": GateValidationResultSchema(
            result_id="out_of_sample",
            description="OOS validation result with rank IC, positive fold ratio, and minimum OOS fold evidence.",
            row_grain="OOS validation summary",
            status_enum=GATE_RULE_STATUSES,
            fields=(
                _field("status", "GateRuleStatus"),
                _field("passed", "boolean"),
                _field("oos_rank_ic", "float|null"),
                _field("mean_rank_ic", "float|null"),
                _field("positive_fold_ratio", "float|null"),
                _field("positive_fold_ratio_threshold", "float"),
                _field("oos_fold_count", "integer"),
                _field("reason_code", "string", "optional"),
                _field("reason", "string", "optional"),
            ),
        ),
        "risk_rules": GateValidationResultSchema(
            result_id="risk_rules",
            description="Portfolio risk rule validation result for long-only, holdings, symbol, sector, drawdown, and turnover constraints.",
            row_grain="risk rule validation summary",
            status_enum=GATE_RULE_STATUSES,
            fields=(
                _field("status", "GateRuleStatus"),
                _field("passed", "boolean"),
                _field("max_holdings_passed", "boolean"),
                _field("max_symbol_weight_passed", "boolean"),
                _field("max_sector_weight_passed", "boolean"),
                _field("drawdown_passed", "boolean"),
                _field("turnover_passed", "boolean"),
                _field("violations", "array[object]"),
                _field("reason_code", "string", "optional"),
                _field("reason", "string", "optional"),
            ),
        ),
    }


def _gate_results_section() -> GateOutputSectionSchema:
    return GateOutputSectionSchema(
        section_id="gate_results",
        description="Per-rule PASS/WARN/FAIL evidence with strategy/system impact flags.",
        fields=(
            _field("leakage", "object"),
            _field("walk_forward_oos", "object"),
            _field("rank_ic", "object"),
            _field("cost_adjusted_performance", "object"),
            _field("benchmark_comparison", "object"),
            _field("turnover", "object"),
            _field("drawdown", "object"),
            _field("ablation", "object"),
            _field("system_validity_artifact_contract", "object"),
            _field("deterministic_strategy_validity", "object"),
            _field("strategy_candidate_policy", "object"),
        ),
    )


def _evidence_section() -> GateOutputSectionSchema:
    return GateOutputSectionSchema(
        section_id="evidence",
        description="Reproducibility and audit details backing the top-level gate statuses.",
        fields=(
            _field("system_validity_gate_input_schema", "object"),
            _field("system_validity_gate_output_schema", "object"),
            _field("thresholds", "object"),
            _field("leakage", "object"),
            _field("purge_embargo_application", "object"),
            _field("walk_forward_oos", "object"),
            _field("rank_ic", "object"),
            _field("baseline_results", "array[object]"),
            _field("baseline_comparisons", "object"),
            _field("baseline_sample_alignment", "object"),
            _field("model_comparison_results", "array[object]"),
            _field("ablation_required_scenarios", "array[string]"),
            _field("structured_pass_fail_reasons", "array[object]"),
            _field("system_validity_gate_criteria", "object"),
            _field("system_validity_gate_report_schema", "object", "optional"),
        ),
    )


def _artifact_contract_section() -> dict[str, object]:
    return {
        "json_artifact": "validity_gate_report.json",
        "markdown_artifact": "validity_gate_report.md",
        "schema_embedded_in_top_level_payload": True,
        "schema_embedded_in_metrics": True,
        "schema_embedded_in_evidence": True,
        "artifact_manifest_required": True,
        "report_path_required": True,
    }


def _scope_bounds_section() -> dict[str, object]:
    return {
        "real_trading_orders": "excluded",
        "llm_trade_decisions": "excluded",
        "deterministic_signal_engine_required": True,
        "target_horizon": "forward_return_20",
        "point_in_time_universe": "v2",
        "correlation_cluster_weight": "v1_5",
        "top_decile_20d_excess_return": "report_only",
    }
