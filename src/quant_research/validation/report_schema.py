from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from quant_research.config import DEFAULT_BENCHMARK_TICKER
from quant_research.validation.config import (
    CANONICAL_STRUCTURED_TEXT_FEATURES,
    DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG,
    DEFAULT_TRANSACTION_COST_SENSITIVITY_CONFIG,
    DETERMINISTIC_SIGNAL_ENGINE_ID,
    REQUIRED_FULL_MODEL_OPTIONAL_ADAPTERS,
    VALID_MODEL_COMPARISON_ADAPTERS,
    PortfolioRiskConstraintConfig,
    TransactionCostSensitivityConfig,
)
from quant_research.validation.horizons import (
    DEFAULT_VALIDATION_HORIZONS,
    REQUIRED_VALIDATION_HORIZON_DAYS,
    forward_return_column,
)
from quant_research.validation.universe import (
    DEFAULT_UNIVERSE_SELECTION_COUNT,
    DEFAULT_UNIVERSE_SURVIVORSHIP_BIAS_DISCLOSURE,
    UniverseSnapshot,
)
from quant_research.validation.walk_forward import (
    CANONICAL_WALK_FORWARD_EMBARGO_PERIODS,
    CANONICAL_WALK_FORWARD_PURGE_PERIODS,
    CANONICAL_WALK_FORWARD_TEST_PERIODS,
    CANONICAL_WALK_FORWARD_TRAIN_PERIODS,
    DEFAULT_PURGE_EMBARGO_WALK_FORWARD_CONFIG,
    PurgeEmbargoWalkForwardConfig,
)

CANONICAL_REPORT_SCHEMA_VERSION = "canonical_report_metadata.v1"
CANONICAL_REPORT_SCHEMA_ID = "stage1_canonical_experiment_report"
REPORT_IDENTITY_SCHEMA_VERSION = "report_identity.v1"
REPORT_UNIVERSE_METADATA_SCHEMA_VERSION = "report_universe_metadata.v1"
REPORT_PERIOD_SCHEMA_VERSION = "report_period.v1"
REPORT_RUN_CONFIGURATION_SCHEMA_VERSION = "report_run_configuration.v1"
REPORT_DATA_PROVENANCE_SCHEMA_VERSION = "report_data_provenance.v1"
REPORT_DATA_SOURCE_SCHEMA_VERSION = "report_data_source.v1"
REPORT_BACKTEST_RESULTS_SCHEMA_VERSION = "report_backtest_results.v1"
REPORT_WALK_FORWARD_VALIDATION_METRICS_SCHEMA_VERSION = (
    "report_walk_forward_validation_metrics.v1"
)
REPORT_RISK_CHECKS_SCHEMA_VERSION = "report_risk_checks.v1"
REPORT_DETERMINISTIC_SIGNAL_SUMMARY_SCHEMA_VERSION = (
    "report_deterministic_signal_summary.v1"
)
REPORT_MODEL_FEATURE_SUMMARIES_SCHEMA_VERSION = "report_model_feature_summaries.v1"
REPORT_TRADE_COST_ASSUMPTIONS_SCHEMA_VERSION = "report_trade_cost_assumptions.v1"
REPORT_INPUT_CONTRACT_SCHEMA_ID = "stage1_canonical_report_input_contract"
REPORT_INPUT_CONTRACT_SCHEMA_VERSION = "canonical_report_input_contract.v1"
ARTIFACT_MANIFEST_SCHEMA_ID = "stage1_canonical_artifact_manifest"
ARTIFACT_MANIFEST_SCHEMA_VERSION = "canonical_artifact_manifest.v1"
DEFAULT_REPORT_TYPE = "canonical_experiment_report"
DEFAULT_REPORT_ARTIFACT_ROOT = "reports"
DEFAULT_ARTIFACT_ROOT = "artifacts"
DEFAULT_MIN_HISTORY_YEARS = 3
DEFAULT_MAX_HOLDINGS = 20
DEFAULT_MAX_SYMBOL_WEIGHT = 0.10
DEFAULT_MAX_SECTOR_WEIGHT = 0.30
DEFAULT_AVERAGE_DAILY_TURNOVER_LIMIT = 0.25
DEFAULT_COST_BPS = 5.0
DEFAULT_SLIPPAGE_BPS = 2.0
DEFAULT_V1_SCOPE_EXCLUSIONS: tuple[str, ...] = (
    "real_trading_orders",
    "llm_trade_decisions",
    "point_in_time_universe",
    "correlation_cluster_weight",
)
CANONICAL_REPORT_REQUIRED_METADATA_SECTIONS: tuple[str, ...] = (
    "identity",
    "universe",
    "period",
    "run_configuration",
    "data_provenance",
)
CANONICAL_REPORT_REQUIRED_RESULT_SECTIONS: tuple[str, ...] = (
    "backtest_results",
    "walk_forward_validation_metrics",
    "risk_checks",
    "model_feature_summaries",
    "deterministic_signal_summary",
    "trade_cost_assumptions",
)
CANONICAL_REPORT_REQUIRED_INPUT_SECTIONS: tuple[str, ...] = (
    "deterministic_signal_outputs",
    "backtest_evaluation_metrics",
    "walk_forward_validation_metrics",
    "risk_evaluation_metrics",
    "comparison_evaluation_metrics",
    "system_validity_gate",
    "strategy_candidate_gate",
    "artifact_manifest",
)
REPORT_INPUT_PROHIBITED_SECTION_IDS: tuple[str, ...] = (
    "predictions",
    "model_predictions",
    "raw_model_predictions",
    "adapter_predictions",
    "llm_predictions",
    "llm_trade_decisions",
)
REPORT_INPUT_PROHIBITED_FIELD_NAMES: tuple[str, ...] = (
    "expected_return",
    "predicted_return",
    "predicted_volatility",
    "downside_quantile",
    "model_prediction",
    "model_prediction_timestamp",
    "prediction_timestamp",
    "chronos_expected_return",
    "chronos_predicted_volatility",
    "granite_ttm_expected_return",
    "raw_llm_output",
    "llm_decision",
)
ARTIFACT_MANIFEST_REQUIRED_METADATA_FIELDS: tuple[str, ...] = (
    "schema_id",
    "schema_version",
    "manifest_id",
    "experiment_id",
    "run_id",
    "created_at",
    "report_path",
    "artifact_root",
    "report_artifact_root",
    "metadata_schema_id",
    "metadata_schema_version",
    "config_hash",
    "universe_snapshot_hash",
    "feature_availability_cutoff_hash",
    "data_snapshot_hash",
    "system_validity_status",
    "strategy_candidate_status",
    "survivorship_bias_allowed",
    "survivorship_bias_disclosure",
    "v1_scope_exclusions",
)


@dataclass(frozen=True, slots=True)
class ReportInputSectionContract:
    section_id: str
    description: str
    row_grain: str
    required_fields: tuple[ReportMetricField, ...]
    optional_fields: tuple[ReportMetricField, ...] = ()
    sample_alignment_key: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        section_id = _required_str(self.section_id, "section_id")
        if section_id in REPORT_INPUT_PROHIBITED_SECTION_IDS:
            raise ValueError(f"report input section {section_id!r} is prohibited")
        object.__setattr__(self, "section_id", section_id)
        object.__setattr__(self, "description", _required_str(self.description, "description"))
        object.__setattr__(self, "row_grain", _required_str(self.row_grain, "row_grain"))
        _validate_report_section_fields(
            section_id,
            self.required_fields,
            self.optional_fields,
        )
        prohibited = sorted(
            set(_field_names((*self.required_fields, *self.optional_fields))).intersection(
                REPORT_INPUT_PROHIBITED_FIELD_NAMES
            )
        )
        if prohibited:
            raise ValueError(
                f"report input section {section_id!r} contains prohibited model "
                f"prediction fields: {prohibited}"
            )
        object.__setattr__(
            self,
            "sample_alignment_key",
            tuple(_required_str(value, "sample_alignment_key") for value in self.sample_alignment_key),
        )

    def required_field_names(self) -> tuple[str, ...]:
        return _field_names(self.required_fields)

    def optional_field_names(self) -> tuple[str, ...]:
        return _field_names(self.optional_fields)

    def to_dict(self) -> dict[str, object]:
        return {
            "section_id": self.section_id,
            "description": self.description,
            "row_grain": self.row_grain,
            "sample_alignment_key": list(self.sample_alignment_key),
            "required_fields": list(self.required_field_names()),
            "optional_fields": list(self.optional_field_names()),
            "fields": [
                field_value.to_dict()
                for field_value in (*self.required_fields, *self.optional_fields)
            ],
        }


@dataclass(frozen=True, slots=True)
class CanonicalReportInputContract:
    schema_id: str = REPORT_INPUT_CONTRACT_SCHEMA_ID
    schema_version: str = REPORT_INPUT_CONTRACT_SCHEMA_VERSION
    required_sections: tuple[str, ...] = CANONICAL_REPORT_REQUIRED_INPUT_SECTIONS
    input_sections: tuple[ReportInputSectionContract, ...] = field(
        default_factory=lambda: (
            _deterministic_signal_outputs_input_section(),
            _backtest_evaluation_metrics_input_section(),
            _walk_forward_validation_metrics_input_section(),
            _risk_evaluation_metrics_input_section(),
            _comparison_evaluation_metrics_input_section(),
            _system_validity_gate_input_section(),
            _strategy_candidate_gate_input_section(),
            _artifact_manifest_input_section(),
        )
    )
    prohibited_input_sections: tuple[str, ...] = REPORT_INPUT_PROHIBITED_SECTION_IDS
    prohibited_field_names: tuple[str, ...] = REPORT_INPUT_PROHIBITED_FIELD_NAMES

    def __post_init__(self) -> None:
        if self.schema_id != REPORT_INPUT_CONTRACT_SCHEMA_ID:
            raise ValueError(f"schema_id must be {REPORT_INPUT_CONTRACT_SCHEMA_ID!r}")
        if self.schema_version != REPORT_INPUT_CONTRACT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {REPORT_INPUT_CONTRACT_SCHEMA_VERSION!r}"
            )
        sections = tuple(self.input_sections)
        if not sections:
            raise ValueError("input_sections must not be empty")
        for section in sections:
            if not isinstance(section, ReportInputSectionContract):
                raise TypeError("input_sections must contain ReportInputSectionContract")
        section_ids = tuple(section.section_id for section in sections)
        if len(set(section_ids)) != len(section_ids):
            raise ValueError("report input contract contains duplicate sections")
        missing = sorted(set(self.required_sections).difference(section_ids))
        if missing:
            raise ValueError(f"report input contract missing required sections: {missing}")
        prohibited_sections = sorted(
            set(section_ids).intersection(self.prohibited_input_sections)
        )
        if prohibited_sections:
            raise ValueError(
                "report input contract must not consume raw model prediction sections: "
                f"{prohibited_sections}"
            )
        object.__setattr__(self, "input_sections", sections)
        object.__setattr__(self, "required_sections", tuple(self.required_sections))
        object.__setattr__(
            self,
            "prohibited_input_sections",
            tuple(self.prohibited_input_sections),
        )
        object.__setattr__(
            self,
            "prohibited_field_names",
            tuple(self.prohibited_field_names),
        )

    def section(self, section_id: str) -> ReportInputSectionContract:
        section_id = _required_str(section_id, "section_id")
        for section in self.input_sections:
            if section.section_id == section_id:
                return section
        raise KeyError(section_id)

    def validate_payload_sections(self, payload: Mapping[str, object]) -> None:
        section_ids = set(payload)
        prohibited_sections = sorted(section_ids.intersection(self.prohibited_input_sections))
        if prohibited_sections:
            raise ValueError(
                "canonical report inputs must not include raw model prediction sections: "
                f"{prohibited_sections}"
            )
        missing = sorted(set(self.required_sections).difference(section_ids))
        if missing:
            raise ValueError(f"canonical report inputs missing required sections: {missing}")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "required_sections": list(self.required_sections),
            "input_sections": {
                section.section_id: section.to_dict()
                for section in self.input_sections
            },
            "consumes_deterministic_signal_outputs": True,
            "consumes_validation_backtest_evaluation_metrics": True,
            "consumes_model_predictions_directly": False,
            "model_predictions_are_order_signals": False,
            "llm_makes_trading_decisions": False,
            "prohibited_input_sections": list(self.prohibited_input_sections),
            "prohibited_field_names": list(self.prohibited_field_names),
        }


@dataclass(frozen=True, slots=True)
class ReportMetricField:
    name: str
    dtype: str
    requirement: str = "required"
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _required_str(self.name, "name"))
        object.__setattr__(self, "dtype", _required_str(self.dtype, "dtype"))
        if self.requirement not in {"required", "optional"}:
            raise ValueError("requirement must be required or optional")
        object.__setattr__(self, "description", str(self.description or "").strip())

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "requirement": self.requirement,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class CanonicalArtifactManifestSchema:
    schema_id: str = ARTIFACT_MANIFEST_SCHEMA_ID
    schema_version: str = ARTIFACT_MANIFEST_SCHEMA_VERSION
    manifest_grain: str = "canonical experiment run"
    required_metadata_fields: tuple[str, ...] = ARTIFACT_MANIFEST_REQUIRED_METADATA_FIELDS
    required_artifact_fields: tuple[ReportMetricField, ...] = field(
        default_factory=lambda: (
            ReportMetricField("artifact_id", "string"),
            ReportMetricField("artifact_type", "string"),
            ReportMetricField("path", "string"),
            ReportMetricField("created_at", "datetime64[ns, UTC]"),
            ReportMetricField("content_hash", "string"),
        )
    )
    optional_artifact_fields: tuple[ReportMetricField, ...] = field(
        default_factory=lambda: (
            ReportMetricField("schema_id", "string", "optional"),
            ReportMetricField("schema_version", "string", "optional"),
            ReportMetricField("row_count", "integer", "optional"),
            ReportMetricField("size_bytes", "integer", "optional"),
            ReportMetricField("modified_at", "datetime64[ns, UTC]", "optional"),
            ReportMetricField("modified_at_epoch_ns", "integer", "optional"),
            ReportMetricField("relative_path", "string", "optional"),
            ReportMetricField("absolute_path", "string", "optional"),
            ReportMetricField("is_directory", "boolean", "optional"),
            ReportMetricField("description", "string", "optional"),
        )
    )

    def __post_init__(self) -> None:
        if self.schema_id != ARTIFACT_MANIFEST_SCHEMA_ID:
            raise ValueError(f"schema_id must be {ARTIFACT_MANIFEST_SCHEMA_ID!r}")
        if self.schema_version != ARTIFACT_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {ARTIFACT_MANIFEST_SCHEMA_VERSION!r}"
            )
        metadata_fields = tuple(
            _required_str(value, "required_metadata_fields")
            for value in self.required_metadata_fields
        )
        missing = sorted(
            set(ARTIFACT_MANIFEST_REQUIRED_METADATA_FIELDS).difference(metadata_fields)
        )
        if missing:
            raise ValueError(f"artifact manifest schema missing metadata fields: {missing}")
        _validate_report_section_fields(
            "artifact_manifest",
            self.required_artifact_fields,
            self.optional_artifact_fields,
        )
        object.__setattr__(
            self,
            "manifest_grain",
            _required_str(self.manifest_grain, "manifest_grain"),
        )
        object.__setattr__(self, "required_metadata_fields", metadata_fields)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "manifest_grain": self.manifest_grain,
            "required_metadata_fields": list(self.required_metadata_fields),
            "required_artifact_fields": list(_field_names(self.required_artifact_fields)),
            "optional_artifact_fields": list(_field_names(self.optional_artifact_fields)),
            "artifact_fields": [
                field_value.to_dict()
                for field_value in (
                    *self.required_artifact_fields,
                    *self.optional_artifact_fields,
                )
            ],
            "validation_rules": {
                "raw_data_caches_committed": False,
                "model_artifacts_committed": False,
                "report_outputs_committed": False,
                "content_hash_required": True,
                "universe_snapshot_hash_required": True,
                "feature_availability_cutoff_hash_required": True,
                "survivorship_bias_disclosure_required": True,
            },
        }


@dataclass(frozen=True, slots=True)
class ReportBacktestResultsSchema:
    schema_version: str = REPORT_BACKTEST_RESULTS_SCHEMA_VERSION
    section_id: str = "backtest_results"
    description: str = (
        "Horizon-consistent long-only portfolio backtest results after deterministic "
        "signals, transaction costs, and slippage."
    )
    row_grain: str = "evaluation date"
    sample_alignment_key: tuple[str, ...] = ("date",)
    required_fields: tuple[ReportMetricField, ...] = field(
        default_factory=lambda: (
            ReportMetricField("date", "datetime64[ns]"),
            ReportMetricField("portfolio_return", "float"),
            ReportMetricField("cost_adjusted_return", "float"),
            ReportMetricField("benchmark_return", "float"),
            ReportMetricField("equal_weight_return", "float"),
            ReportMetricField("turnover", "float"),
            ReportMetricField("holdings_count", "integer"),
            ReportMetricField("max_symbol_weight", "float"),
            ReportMetricField("max_sector_weight", "float"),
            ReportMetricField("cost_bps", "float"),
            ReportMetricField("slippage_bps", "float"),
        )
    )
    optional_fields: tuple[ReportMetricField, ...] = field(
        default_factory=lambda: (
            ReportMetricField("gross_return", "float", "optional"),
            ReportMetricField("transaction_cost_return", "float", "optional"),
            ReportMetricField("slippage_cost_return", "float", "optional"),
            ReportMetricField("total_cost_return", "float", "optional"),
            ReportMetricField("rebalance_interval", "integer|string", "optional"),
            ReportMetricField("position_count", "integer", "optional"),
        )
    )

    def __post_init__(self) -> None:
        if self.schema_version != REPORT_BACKTEST_RESULTS_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {REPORT_BACKTEST_RESULTS_SCHEMA_VERSION!r}"
            )
        _validate_report_section_fields(
            self.section_id,
            self.required_fields,
            self.optional_fields,
        )
        if "date" not in self.sample_alignment_key:
            raise ValueError("backtest_results sample_alignment_key must include date")

    def to_dict(self) -> dict[str, object]:
        return _report_section_to_dict(
            schema_version=self.schema_version,
            section_id=self.section_id,
            description=self.description,
            row_grain=self.row_grain,
            sample_alignment_key=self.sample_alignment_key,
            required_fields=self.required_fields,
            optional_fields=self.optional_fields,
            validation_rules={
                "long_only": True,
                "benchmark_ticker": DEFAULT_BENCHMARK_TICKER,
                "equal_weight_universe_baseline_required": True,
                "cost_bps": DEFAULT_COST_BPS,
                "slippage_bps": DEFAULT_SLIPPAGE_BPS,
                "max_holdings": DEFAULT_MAX_HOLDINGS,
                "max_symbol_weight": DEFAULT_MAX_SYMBOL_WEIGHT,
                "max_sector_weight": DEFAULT_MAX_SECTOR_WEIGHT,
                "average_daily_turnover_limit": DEFAULT_AVERAGE_DAILY_TURNOVER_LIMIT,
            },
        )


@dataclass(frozen=True, slots=True)
class ReportWalkForwardValidationMetricsSchema:
    schema_version: str = REPORT_WALK_FORWARD_VALIDATION_METRICS_SCHEMA_VERSION
    section_id: str = "walk_forward_validation_metrics"
    description: str = (
        "Fold-level and aggregate walk-forward metrics for forward_return_20 "
        "out-of-sample validation with purge and embargo evidence."
    )
    row_grain: str = "walk-forward fold plus aggregate OOS summary"
    sample_alignment_key: tuple[str, ...] = ("fold",)
    required_fields: tuple[ReportMetricField, ...] = field(
        default_factory=lambda: (
            ReportMetricField("fold", "integer"),
            ReportMetricField("is_oos", "boolean"),
            ReportMetricField("target_column", "string"),
            ReportMetricField("prediction_horizon_periods", "integer"),
            ReportMetricField("train_start", "datetime64[ns]"),
            ReportMetricField("train_end", "datetime64[ns]"),
            ReportMetricField("test_start", "datetime64[ns]"),
            ReportMetricField("test_end", "datetime64[ns]"),
            ReportMetricField("purge_periods", "integer"),
            ReportMetricField("embargo_periods", "integer"),
            ReportMetricField("train_observations", "integer"),
            ReportMetricField("labeled_test_observations", "integer"),
            ReportMetricField("fold_rank_ic", "float"),
            ReportMetricField("positive_rank_ic", "boolean"),
        )
    )
    optional_fields: tuple[ReportMetricField, ...] = field(
        default_factory=lambda: (
            ReportMetricField("mean_rank_ic", "float", "optional"),
            ReportMetricField("positive_fold_ratio", "float", "optional"),
            ReportMetricField("oos_fold_count", "integer", "optional"),
            ReportMetricField("oos_rank_ic_mean", "float", "optional"),
            ReportMetricField("proxy_ic_improvement", "float", "optional"),
            ReportMetricField("status", "GateRuleStatus", "optional"),
        )
    )

    def __post_init__(self) -> None:
        if self.schema_version != REPORT_WALK_FORWARD_VALIDATION_METRICS_SCHEMA_VERSION:
            raise ValueError(
                "schema_version must be "
                f"{REPORT_WALK_FORWARD_VALIDATION_METRICS_SCHEMA_VERSION!r}"
            )
        _validate_report_section_fields(
            self.section_id,
            self.required_fields,
            self.optional_fields,
        )
        if "fold" not in self.sample_alignment_key:
            raise ValueError("walk_forward_validation_metrics sample_alignment_key must include fold")

    def to_dict(self) -> dict[str, object]:
        return _report_section_to_dict(
            schema_version=self.schema_version,
            section_id=self.section_id,
            description=self.description,
            row_grain=self.row_grain,
            sample_alignment_key=self.sample_alignment_key,
            required_fields=self.required_fields,
            optional_fields=self.optional_fields,
            validation_rules={
                "target_column": forward_return_column(REQUIRED_VALIDATION_HORIZON_DAYS),
                "prediction_horizon_periods": REQUIRED_VALIDATION_HORIZON_DAYS,
                "train_periods": CANONICAL_WALK_FORWARD_TRAIN_PERIODS,
                "test_periods": CANONICAL_WALK_FORWARD_TEST_PERIODS,
                "purge_periods": CANONICAL_WALK_FORWARD_PURGE_PERIODS,
                "embargo_periods": CANONICAL_WALK_FORWARD_EMBARGO_PERIODS,
                "embargo_zero_for_forward_return_20_is_hard_fail": True,
                "minimum_oos_fold_count": 2,
            },
        )


@dataclass(frozen=True, slots=True)
class ReportRiskChecksSchema:
    schema_version: str = REPORT_RISK_CHECKS_SCHEMA_VERSION
    section_id: str = "risk_checks"
    description: str = (
        "Portfolio-level risk rule checks that must be reviewed before canonical "
        "Stage 1 signals are considered usable."
    )
    row_grain: str = "evaluation date and risk rule"
    sample_alignment_key: tuple[str, ...] = ("date", "risk_check")
    required_fields: tuple[ReportMetricField, ...] = field(
        default_factory=lambda: (
            ReportMetricField("date", "datetime64[ns]"),
            ReportMetricField("risk_check", "string"),
            ReportMetricField("status", "PASS|WARN|FAIL"),
            ReportMetricField("observed_value", "float|boolean|string"),
            ReportMetricField("threshold_value", "float|boolean|string"),
            ReportMetricField("long_only", "boolean"),
            ReportMetricField("holdings_count", "integer"),
            ReportMetricField("max_symbol_weight", "float"),
            ReportMetricField("max_sector_weight", "float"),
            ReportMetricField("average_daily_turnover", "float"),
            ReportMetricField("max_drawdown", "float"),
            ReportMetricField("covariance_aware_risk_enabled", "boolean"),
        )
    )
    optional_fields: tuple[ReportMetricField, ...] = field(
        default_factory=lambda: (
            ReportMetricField("portfolio_volatility", "float", "optional"),
            ReportMetricField("max_position_risk_contribution", "float", "optional"),
            ReportMetricField("risk_stop_active", "boolean", "optional"),
            ReportMetricField("breach_reason", "string", "optional"),
            ReportMetricField("covariance_lookback_periods", "integer", "optional"),
        )
    )

    def __post_init__(self) -> None:
        if self.schema_version != REPORT_RISK_CHECKS_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {REPORT_RISK_CHECKS_SCHEMA_VERSION!r}"
            )
        _validate_report_section_fields(
            self.section_id,
            self.required_fields,
            self.optional_fields,
        )
        if set(self.sample_alignment_key) != {"date", "risk_check"}:
            raise ValueError("risk_checks sample_alignment_key must be date and risk_check")

    def to_dict(self) -> dict[str, object]:
        return _report_section_to_dict(
            schema_version=self.schema_version,
            section_id=self.section_id,
            description=self.description,
            row_grain=self.row_grain,
            sample_alignment_key=self.sample_alignment_key,
            required_fields=self.required_fields,
            optional_fields=self.optional_fields,
            validation_rules={
                "long_only": True,
                "max_holdings": DEFAULT_MAX_HOLDINGS,
                "max_symbol_weight": DEFAULT_MAX_SYMBOL_WEIGHT,
                "max_sector_weight": DEFAULT_MAX_SECTOR_WEIGHT,
                "average_daily_turnover_limit": DEFAULT_AVERAGE_DAILY_TURNOVER_LIMIT,
                "max_drawdown_floor": -0.20,
                "correlation_cluster_weight_in_v1": False,
                "risk_checks_required_before_signal_summary": True,
            },
        )


@dataclass(frozen=True, slots=True)
class ReportDeterministicSignalSummarySchema:
    schema_version: str = REPORT_DETERMINISTIC_SIGNAL_SUMMARY_SCHEMA_VERSION
    section_id: str = "deterministic_signal_summary"
    description: str = (
        "Ticker-date summary of deterministic signal engine outputs. Model and LLM "
        "predictions are inputs only and are not treated as trade decisions."
    )
    row_grain: str = "ticker and signal date"
    sample_alignment_key: tuple[str, ...] = ("date", "ticker")
    required_fields: tuple[ReportMetricField, ...] = field(
        default_factory=lambda: (
            ReportMetricField("date", "datetime64[ns]"),
            ReportMetricField("ticker", "string"),
            ReportMetricField("expected_return", "float"),
            ReportMetricField("predicted_volatility", "float"),
            ReportMetricField("downside_quantile", "float"),
            ReportMetricField("sentiment_score", "float"),
            ReportMetricField("text_risk_score", "float"),
            ReportMetricField("sec_risk_flag", "float"),
            ReportMetricField("model_confidence", "float"),
            ReportMetricField("signal_score", "float"),
            ReportMetricField("action", "BUY|SELL|HOLD"),
            ReportMetricField("signal_engine", "string"),
        )
    )
    optional_fields: tuple[ReportMetricField, ...] = field(
        default_factory=lambda: (
            ReportMetricField("event_tag", "string", "optional"),
            ReportMetricField("summary_ref", "string", "optional"),
            ReportMetricField("risk_metric_penalty", "float", "optional"),
            ReportMetricField("signal_generation_gate_status", "PASS|FAIL", "optional"),
            ReportMetricField("signal_generation_gate_decision", "string", "optional"),
        )
    )

    def __post_init__(self) -> None:
        if self.schema_version != REPORT_DETERMINISTIC_SIGNAL_SUMMARY_SCHEMA_VERSION:
            raise ValueError(
                "schema_version must be "
                f"{REPORT_DETERMINISTIC_SIGNAL_SUMMARY_SCHEMA_VERSION!r}"
            )
        _validate_report_section_fields(
            self.section_id,
            self.required_fields,
            self.optional_fields,
        )
        if set(self.sample_alignment_key) != {"date", "ticker"}:
            raise ValueError(
                "deterministic_signal_summary sample_alignment_key must be date and ticker"
            )

    def to_dict(self) -> dict[str, object]:
        return _report_section_to_dict(
            schema_version=self.schema_version,
            section_id=self.section_id,
            description=self.description,
            row_grain=self.row_grain,
            sample_alignment_key=self.sample_alignment_key,
            required_fields=self.required_fields,
            optional_fields=self.optional_fields,
            validation_rules={
                "signal_engine": DETERMINISTIC_SIGNAL_ENGINE_ID,
                "allowed_actions": ["BUY", "SELL", "HOLD"],
                "llm_makes_trading_decisions": False,
                "model_predictions_are_order_signals": False,
                "requires_system_validity_gate_pass": True,
                "feature_availability_cutoff_required": True,
            },
        )


@dataclass(frozen=True, slots=True)
class ReportModelFeatureSummariesSchema:
    schema_version: str = REPORT_MODEL_FEATURE_SUMMARIES_SCHEMA_VERSION
    section_id: str = "model_feature_summaries"
    description: str = (
        "Ticker-date feature provenance and model adapter output summaries used as "
        "inputs to deterministic scoring. Text model outputs are structured fields; "
        "optional heavy adapters may be absent or rules-fallback generated."
    )
    row_grain: str = "ticker, feature date, feature name, and source adapter"
    sample_alignment_key: tuple[str, ...] = (
        "date",
        "ticker",
        "feature_name",
        "source_adapter",
    )
    required_fields: tuple[ReportMetricField, ...] = field(
        default_factory=lambda: (
            ReportMetricField("date", "datetime64[ns]"),
            ReportMetricField("ticker", "string"),
            ReportMetricField("feature_family", "price|text|sec|chronos|granite_ttm"),
            ReportMetricField("feature_name", "string"),
            ReportMetricField("feature_value", "float|string|boolean|null"),
            ReportMetricField("feature_as_of", "datetime64[ns]"),
            ReportMetricField("feature_available_at", "datetime64[ns]"),
            ReportMetricField("source_adapter", "ModelComparisonAdapter"),
            ReportMetricField("structured_feature", "boolean"),
            ReportMetricField("used_by_signal_engine", "boolean"),
        )
    )
    optional_fields: tuple[ReportMetricField, ...] = field(
        default_factory=lambda: (
            ReportMetricField("sentiment_score", "float", "optional"),
            ReportMetricField("event_tag", "string", "optional"),
            ReportMetricField("risk_flag", "boolean|float|string", "optional"),
            ReportMetricField("confidence", "float", "optional"),
            ReportMetricField("summary_ref", "string", "optional"),
            ReportMetricField("adapter_output_name", "string", "optional"),
            ReportMetricField("adapter_output_value", "float|string|boolean|null", "optional"),
            ReportMetricField("adapter_confidence", "float", "optional"),
            ReportMetricField("adapter_fallback_used", "boolean", "optional"),
            ReportMetricField("adapter_artifact_ref", "string", "optional"),
            ReportMetricField("chronos_expected_return", "float", "optional"),
            ReportMetricField("chronos_predicted_volatility", "float", "optional"),
            ReportMetricField("granite_ttm_expected_return", "float", "optional"),
            ReportMetricField("finbert_sentiment_score", "float", "optional"),
            ReportMetricField("finma_event_tag", "string", "optional"),
            ReportMetricField("fingpt_risk_flag", "boolean|float|string", "optional"),
            ReportMetricField("ollama_summary_ref", "string", "optional"),
        )
    )

    def __post_init__(self) -> None:
        if self.schema_version != REPORT_MODEL_FEATURE_SUMMARIES_SCHEMA_VERSION:
            raise ValueError(
                "schema_version must be "
                f"{REPORT_MODEL_FEATURE_SUMMARIES_SCHEMA_VERSION!r}"
            )
        _validate_report_section_fields(
            self.section_id,
            self.required_fields,
            self.optional_fields,
        )
        if set(self.sample_alignment_key) != {
            "date",
            "ticker",
            "feature_name",
            "source_adapter",
        }:
            raise ValueError(
                "model_feature_summaries sample_alignment_key must be date, "
                "ticker, feature_name, and source_adapter"
            )

    def to_dict(self) -> dict[str, object]:
        return _report_section_to_dict(
            schema_version=self.schema_version,
            section_id=self.section_id,
            description=self.description,
            row_grain=self.row_grain,
            sample_alignment_key=self.sample_alignment_key,
            required_fields=self.required_fields,
            optional_fields=self.optional_fields,
            validation_rules={
                "structured_text_features": list(CANONICAL_STRUCTURED_TEXT_FEATURES),
                "optional_model_adapters": list(REQUIRED_FULL_MODEL_OPTIONAL_ADAPTERS),
                "allowed_source_adapters": sorted(VALID_MODEL_COMPARISON_ADAPTERS),
                "optional_adapters_may_be_missing": True,
                "rules_fallback_allowed": True,
                "feature_availability_cutoff_required": True,
                "feature_available_at_lte_signal_date_required": True,
                "llm_makes_trading_decisions": False,
                "adapter_outputs_are_order_signals": False,
                "signal_engine": DETERMINISTIC_SIGNAL_ENGINE_ID,
            },
        )


@dataclass(frozen=True, slots=True)
class ReportTradeCostAssumptionsSchema:
    schema_version: str = REPORT_TRADE_COST_ASSUMPTIONS_SCHEMA_VERSION
    section_id: str = "trade_cost_assumptions"
    description: str = (
        "Canonical transaction cost, slippage, and turnover assumptions used by "
        "the long-only portfolio backtest and sensitivity analysis."
    )
    row_grain: str = "cost scenario"
    sample_alignment_key: tuple[str, ...] = ("scenario_id",)
    required_fields: tuple[ReportMetricField, ...] = field(
        default_factory=lambda: (
            ReportMetricField("scenario_id", "string"),
            ReportMetricField("is_baseline", "boolean"),
            ReportMetricField("cost_bps", "float"),
            ReportMetricField("slippage_bps", "float"),
            ReportMetricField("total_cost_bps", "float"),
            ReportMetricField("average_daily_turnover_budget", "float"),
            ReportMetricField("max_daily_turnover", "float|null"),
            ReportMetricField("cost_model", "string"),
            ReportMetricField("applies_to_turnover", "boolean"),
        )
    )
    optional_fields: tuple[ReportMetricField, ...] = field(
        default_factory=lambda: (
            ReportMetricField("turnover_sensitivity_grid", "array", "optional"),
            ReportMetricField("cost_adjusted_return_delta", "float", "optional"),
            ReportMetricField("total_cost_return", "float", "optional"),
            ReportMetricField("breach_count", "integer", "optional"),
        )
    )

    def __post_init__(self) -> None:
        if self.schema_version != REPORT_TRADE_COST_ASSUMPTIONS_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {REPORT_TRADE_COST_ASSUMPTIONS_SCHEMA_VERSION!r}"
            )
        _validate_report_section_fields(
            self.section_id,
            self.required_fields,
            self.optional_fields,
        )
        if self.sample_alignment_key != ("scenario_id",):
            raise ValueError(
                "trade_cost_assumptions sample_alignment_key must be scenario_id"
            )

    def to_dict(self) -> dict[str, object]:
        return _report_section_to_dict(
            schema_version=self.schema_version,
            section_id=self.section_id,
            description=self.description,
            row_grain=self.row_grain,
            sample_alignment_key=self.sample_alignment_key,
            required_fields=self.required_fields,
            optional_fields=self.optional_fields,
            validation_rules={
                "baseline_cost_bps": DEFAULT_COST_BPS,
                "baseline_slippage_bps": DEFAULT_SLIPPAGE_BPS,
                "baseline_total_cost_bps": DEFAULT_COST_BPS + DEFAULT_SLIPPAGE_BPS,
                "average_daily_turnover_limit": DEFAULT_AVERAGE_DAILY_TURNOVER_LIMIT,
                "costs_and_slippage_required": True,
                "turnover_sensitivity_required": True,
            },
        )


@dataclass(frozen=True, slots=True)
class ReportIdentity:
    experiment_id: str
    run_id: str
    report_id: str | None = None
    report_type: str = DEFAULT_REPORT_TYPE
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    schema_version: str = REPORT_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        experiment_id = _required_str(self.experiment_id, "experiment_id")
        run_id = _required_str(self.run_id, "run_id")
        report_type = _required_str(self.report_type, "report_type")
        if self.schema_version != REPORT_IDENTITY_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {REPORT_IDENTITY_SCHEMA_VERSION!r}")
        created_at = _coerce_datetime(self.created_at, "created_at")
        report_id = _required_str(
            self.report_id or f"{experiment_id}:{run_id}:{report_type}",
            "report_id",
        )

        object.__setattr__(self, "experiment_id", experiment_id)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "report_id", report_id)
        object.__setattr__(self, "report_type", report_type)
        object.__setattr__(self, "created_at", created_at)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "report_id": self.report_id,
            "report_type": self.report_type,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ReportUniverseMetadata:
    universe_snapshot: UniverseSnapshot
    schema_version: str = REPORT_UNIVERSE_METADATA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REPORT_UNIVERSE_METADATA_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {REPORT_UNIVERSE_METADATA_SCHEMA_VERSION!r}")
        if not isinstance(self.universe_snapshot, UniverseSnapshot):
            raise TypeError("universe_snapshot must be a UniverseSnapshot")
        if self.universe_snapshot.selection_count != DEFAULT_UNIVERSE_SELECTION_COUNT:
            raise ValueError("canonical report universe must use the top 150 selection count")
        if self.universe_snapshot.point_in_time_membership:
            raise ValueError("v1 report universe must not claim point-in-time membership")
        if not self.universe_snapshot.survivorship_bias_allowed:
            raise ValueError("v1 report universe must explicitly allow survivorship bias")

    @property
    def tickers(self) -> tuple[str, ...]:
        return self.universe_snapshot.tickers

    def to_dict(self) -> dict[str, object]:
        snapshot = self.universe_snapshot.to_dict()
        return {
            "schema_version": self.schema_version,
            "universe": list(self.tickers),
            "universe_count": self.universe_snapshot.constituent_count,
            "universe_snapshot": snapshot,
            "universe_snapshot_schema_version": snapshot["schema_version"],
            "universe_snapshot_date": snapshot["snapshot_date"],
            "benchmark_ticker": self.universe_snapshot.benchmark_ticker,
            "fixed_at_experiment_start": self.universe_snapshot.fixed_at_experiment_start,
            "survivorship_bias_allowed": self.universe_snapshot.survivorship_bias_allowed,
            "survivorship_bias_disclosure": (
                self.universe_snapshot.survivorship_bias_disclosure
                or DEFAULT_UNIVERSE_SURVIVORSHIP_BIAS_DISCLOSURE
            ),
        }


@dataclass(frozen=True, slots=True)
class ReportPeriod:
    start_date: date
    end_date: date
    min_history_years: int = DEFAULT_MIN_HISTORY_YEARS
    timezone: str = "UTC"
    schema_version: str = REPORT_PERIOD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REPORT_PERIOD_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {REPORT_PERIOD_SCHEMA_VERSION!r}")
        start_date = _coerce_date(self.start_date, "start_date")
        end_date = _coerce_date(self.end_date, "end_date")
        if end_date <= start_date:
            raise ValueError("end_date must be after start_date")
        min_history_years = int(self.min_history_years)
        if min_history_years < DEFAULT_MIN_HISTORY_YEARS:
            raise ValueError("min_history_years must be at least 3")
        if (end_date - start_date).days < 365 * min_history_years:
            raise ValueError("report period must cover at least min_history_years")
        timezone_name = _required_str(self.timezone, "timezone")

        object.__setattr__(self, "start_date", start_date)
        object.__setattr__(self, "end_date", end_date)
        object.__setattr__(self, "min_history_years", min_history_years)
        object.__setattr__(self, "timezone", timezone_name)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "min_history_years": self.min_history_years,
            "timezone": self.timezone,
        }


@dataclass(frozen=True, slots=True)
class ReportRunConfiguration:
    walk_forward_config: PurgeEmbargoWalkForwardConfig = (
        DEFAULT_PURGE_EMBARGO_WALK_FORWARD_CONFIG
    )
    portfolio_constraints: PortfolioRiskConstraintConfig = (
        DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG
    )
    transaction_costs: TransactionCostSensitivityConfig = (
        DEFAULT_TRANSACTION_COST_SENSITIVITY_CONFIG
    )
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER
    equal_weight_universe_baseline: bool = True
    target_horizon: str = forward_return_column(REQUIRED_VALIDATION_HORIZON_DAYS)
    diagnostic_horizons: tuple[str, ...] = (
        forward_return_column(1),
        forward_return_column(5),
    )
    signal_engine: str = DETERMINISTIC_SIGNAL_ENGINE_ID
    model_predictions_are_order_signals: bool = False
    llm_makes_trading_decisions: bool = False
    average_daily_turnover_limit: float = DEFAULT_AVERAGE_DAILY_TURNOVER_LIMIT
    schema_version: str = REPORT_RUN_CONFIGURATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REPORT_RUN_CONFIGURATION_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {REPORT_RUN_CONFIGURATION_SCHEMA_VERSION!r}"
            )
        if not isinstance(self.walk_forward_config, PurgeEmbargoWalkForwardConfig):
            raise TypeError("walk_forward_config must be a PurgeEmbargoWalkForwardConfig")
        if self.walk_forward_config.system_validity_issues():
            raise ValueError(
                "walk_forward_config is not system valid: "
                + "; ".join(self.walk_forward_config.system_validity_issues())
            )
        if not isinstance(self.portfolio_constraints, PortfolioRiskConstraintConfig):
            raise TypeError("portfolio_constraints must be a PortfolioRiskConstraintConfig")
        if not isinstance(self.transaction_costs, TransactionCostSensitivityConfig):
            raise TypeError("transaction_costs must be a TransactionCostSensitivityConfig")
        benchmark_ticker = _required_str(self.benchmark_ticker, "benchmark_ticker").upper()
        if benchmark_ticker != DEFAULT_BENCHMARK_TICKER:
            raise ValueError("canonical report benchmark_ticker must be SPY")
        if self.target_horizon != forward_return_column(REQUIRED_VALIDATION_HORIZON_DAYS):
            raise ValueError("canonical report target_horizon must be forward_return_20")
        diagnostic_horizons = tuple(_required_str(value, "diagnostic_horizons") for value in self.diagnostic_horizons)
        expected_diagnostics = tuple(
            forward_return_column(horizon)
            for horizon in DEFAULT_VALIDATION_HORIZONS
            if horizon != REQUIRED_VALIDATION_HORIZON_DAYS
        )
        if diagnostic_horizons != expected_diagnostics:
            raise ValueError("diagnostic_horizons must be forward_return_1 and forward_return_5")
        if self.signal_engine != DETERMINISTIC_SIGNAL_ENGINE_ID:
            raise ValueError("signal_engine must be deterministic_signal_engine")
        if self.model_predictions_are_order_signals:
            raise ValueError("model predictions must not be treated as order signals")
        if self.llm_makes_trading_decisions:
            raise ValueError("LLM adapters must not make trading decisions")
        if not self.equal_weight_universe_baseline:
            raise ValueError("canonical reports must include the equal-weight universe baseline")
        turnover_limit = float(self.average_daily_turnover_limit)
        if turnover_limit != DEFAULT_AVERAGE_DAILY_TURNOVER_LIMIT:
            raise ValueError("average_daily_turnover_limit must be 0.25")

        base_costs = self.transaction_costs.get(self.transaction_costs.baseline_scenario_id)
        if base_costs.cost_bps != DEFAULT_COST_BPS or base_costs.slippage_bps != DEFAULT_SLIPPAGE_BPS:
            raise ValueError("canonical transaction costs must be 5 bps cost and 2 bps slippage")
        if self.portfolio_constraints.max_holdings != DEFAULT_MAX_HOLDINGS:
            raise ValueError("canonical portfolio max_holdings must be 20")
        if self.portfolio_constraints.max_symbol_weight != DEFAULT_MAX_SYMBOL_WEIGHT:
            raise ValueError("canonical portfolio max_symbol_weight must be 10%")
        if self.portfolio_constraints.max_sector_weight != DEFAULT_MAX_SECTOR_WEIGHT:
            raise ValueError("canonical portfolio max_sector_weight must be 30%")

        object.__setattr__(self, "benchmark_ticker", benchmark_ticker)
        object.__setattr__(self, "diagnostic_horizons", diagnostic_horizons)
        object.__setattr__(self, "average_daily_turnover_limit", turnover_limit)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "target_horizon": self.target_horizon,
            "diagnostic_horizons": list(self.diagnostic_horizons),
            "walk_forward_config": self.walk_forward_config.to_dict(),
            "portfolio_constraints": self.portfolio_constraints.to_dict(),
            "transaction_costs": self.transaction_costs.to_dict(),
            "benchmark_config": {
                "benchmark_ticker": self.benchmark_ticker,
                "benchmark_type": "market",
                "equal_weight_universe_baseline": self.equal_weight_universe_baseline,
                "sample_alignment_required": True,
            },
            "signal_engine": self.signal_engine,
            "model_predictions_are_order_signals": self.model_predictions_are_order_signals,
            "llm_makes_trading_decisions": self.llm_makes_trading_decisions,
            "average_daily_turnover_limit": self.average_daily_turnover_limit,
        }


@dataclass(frozen=True, slots=True)
class ReportDataSource:
    source_id: str
    provider: str
    dataset: str
    as_of_date: date | None = None
    available_at: datetime | None = None
    source_version: str | None = None
    schema_version: str = REPORT_DATA_SOURCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REPORT_DATA_SOURCE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {REPORT_DATA_SOURCE_SCHEMA_VERSION!r}")
        object.__setattr__(self, "source_id", _required_str(self.source_id, "source_id"))
        object.__setattr__(self, "provider", _required_str(self.provider, "provider"))
        object.__setattr__(self, "dataset", _required_str(self.dataset, "dataset"))
        if self.as_of_date is not None:
            object.__setattr__(self, "as_of_date", _coerce_date(self.as_of_date, "as_of_date"))
        if self.available_at is not None:
            object.__setattr__(
                self,
                "available_at",
                _coerce_datetime(self.available_at, "available_at"),
            )
        object.__setattr__(self, "source_version", _optional_str(self.source_version))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "provider": self.provider,
            "dataset": self.dataset,
            "as_of_date": self.as_of_date.isoformat() if self.as_of_date else None,
            "available_at": self.available_at.isoformat() if self.available_at else None,
            "source_version": self.source_version,
        }


@dataclass(frozen=True, slots=True)
class ReportDataProvenance:
    data_sources: tuple[ReportDataSource, ...]
    feature_availability_cutoff: Mapping[str, Any]
    raw_data_committed: bool = False
    model_artifacts_committed: bool = False
    report_outputs_committed: bool = False
    structured_text_feature_fields: tuple[str, ...] = CANONICAL_STRUCTURED_TEXT_FEATURES
    schema_version: str = REPORT_DATA_PROVENANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REPORT_DATA_PROVENANCE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {REPORT_DATA_PROVENANCE_SCHEMA_VERSION!r}")
        data_sources = tuple(self.data_sources)
        if not data_sources:
            raise ValueError("data_sources must contain at least one source")
        for source in data_sources:
            if not isinstance(source, ReportDataSource):
                raise TypeError("data_sources must contain ReportDataSource instances")
        cutoff = dict(self.feature_availability_cutoff)
        if not cutoff:
            raise ValueError("feature_availability_cutoff must not be empty")
        if not self.structured_text_feature_fields:
            raise ValueError("structured_text_feature_fields must not be empty")
        if self.raw_data_committed or self.model_artifacts_committed or self.report_outputs_committed:
            raise ValueError("raw data caches, model artifacts, and report outputs must not be committed")

        object.__setattr__(self, "data_sources", data_sources)
        object.__setattr__(self, "feature_availability_cutoff", cutoff)
        object.__setattr__(
            self,
            "structured_text_feature_fields",
            tuple(self.structured_text_feature_fields),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "data_sources": [source.to_dict() for source in self.data_sources],
            "feature_availability_cutoff": dict(self.feature_availability_cutoff),
            "structured_text_feature_fields": list(self.structured_text_feature_fields),
            "raw_data_committed": self.raw_data_committed,
            "model_artifacts_committed": self.model_artifacts_committed,
            "report_outputs_committed": self.report_outputs_committed,
        }


@dataclass(frozen=True, slots=True)
class CanonicalReportMetadata:
    identity: ReportIdentity
    universe: ReportUniverseMetadata
    period: ReportPeriod
    run_configuration: ReportRunConfiguration
    data_provenance: ReportDataProvenance
    report_input_contract: CanonicalReportInputContract = field(
        default_factory=lambda: CanonicalReportInputContract()
    )
    backtest_results_schema: ReportBacktestResultsSchema = field(
        default_factory=lambda: ReportBacktestResultsSchema()
    )
    walk_forward_validation_metrics_schema: ReportWalkForwardValidationMetricsSchema = field(
        default_factory=lambda: ReportWalkForwardValidationMetricsSchema()
    )
    risk_checks_schema: ReportRiskChecksSchema = field(
        default_factory=lambda: ReportRiskChecksSchema()
    )
    model_feature_summaries_schema: ReportModelFeatureSummariesSchema = field(
        default_factory=lambda: ReportModelFeatureSummariesSchema()
    )
    deterministic_signal_summary_schema: ReportDeterministicSignalSummarySchema = field(
        default_factory=lambda: ReportDeterministicSignalSummarySchema()
    )
    trade_cost_assumptions_schema: ReportTradeCostAssumptionsSchema = field(
        default_factory=lambda: ReportTradeCostAssumptionsSchema()
    )
    artifact_manifest_schema: CanonicalArtifactManifestSchema = field(
        default_factory=lambda: CanonicalArtifactManifestSchema()
    )
    artifact_root: str = DEFAULT_ARTIFACT_ROOT
    report_artifact_root: str = DEFAULT_REPORT_ARTIFACT_ROOT
    v1_scope_exclusions: tuple[str, ...] = DEFAULT_V1_SCOPE_EXCLUSIONS
    schema_version: str = CANONICAL_REPORT_SCHEMA_VERSION
    schema_id: str = CANONICAL_REPORT_SCHEMA_ID

    def __post_init__(self) -> None:
        if self.schema_version != CANONICAL_REPORT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {CANONICAL_REPORT_SCHEMA_VERSION!r}")
        if self.schema_id != CANONICAL_REPORT_SCHEMA_ID:
            raise ValueError(f"schema_id must be {CANONICAL_REPORT_SCHEMA_ID!r}")
        if not isinstance(self.identity, ReportIdentity):
            raise TypeError("identity must be a ReportIdentity")
        if not isinstance(self.universe, ReportUniverseMetadata):
            raise TypeError("universe must be a ReportUniverseMetadata")
        if not isinstance(self.period, ReportPeriod):
            raise TypeError("period must be a ReportPeriod")
        if not isinstance(self.run_configuration, ReportRunConfiguration):
            raise TypeError("run_configuration must be a ReportRunConfiguration")
        if not isinstance(self.data_provenance, ReportDataProvenance):
            raise TypeError("data_provenance must be a ReportDataProvenance")
        if not isinstance(self.report_input_contract, CanonicalReportInputContract):
            raise TypeError("report_input_contract must be a CanonicalReportInputContract")
        if not isinstance(self.backtest_results_schema, ReportBacktestResultsSchema):
            raise TypeError("backtest_results_schema must be a ReportBacktestResultsSchema")
        if not isinstance(
            self.walk_forward_validation_metrics_schema,
            ReportWalkForwardValidationMetricsSchema,
        ):
            raise TypeError(
                "walk_forward_validation_metrics_schema must be a "
                "ReportWalkForwardValidationMetricsSchema"
            )
        if not isinstance(self.risk_checks_schema, ReportRiskChecksSchema):
            raise TypeError("risk_checks_schema must be a ReportRiskChecksSchema")
        if not isinstance(
            self.model_feature_summaries_schema,
            ReportModelFeatureSummariesSchema,
        ):
            raise TypeError(
                "model_feature_summaries_schema must be a "
                "ReportModelFeatureSummariesSchema"
            )
        if not isinstance(
            self.deterministic_signal_summary_schema,
            ReportDeterministicSignalSummarySchema,
        ):
            raise TypeError(
                "deterministic_signal_summary_schema must be a "
                "ReportDeterministicSignalSummarySchema"
            )
        if not isinstance(
            self.trade_cost_assumptions_schema,
            ReportTradeCostAssumptionsSchema,
        ):
            raise TypeError(
                "trade_cost_assumptions_schema must be a ReportTradeCostAssumptionsSchema"
            )
        if not isinstance(self.artifact_manifest_schema, CanonicalArtifactManifestSchema):
            raise TypeError(
                "artifact_manifest_schema must be a CanonicalArtifactManifestSchema"
            )
        if self.identity.experiment_id != self.universe.universe_snapshot.experiment_id:
            raise ValueError("identity.experiment_id must match universe snapshot experiment_id")
        if self.universe.universe_snapshot.snapshot_date != self.period.start_date:
            raise ValueError("universe snapshot_date must equal report period start_date")
        exclusions = tuple(_required_str(value, "v1_scope_exclusions") for value in self.v1_scope_exclusions)
        missing_exclusions = sorted(set(DEFAULT_V1_SCOPE_EXCLUSIONS).difference(exclusions))
        if missing_exclusions:
            raise ValueError(f"v1_scope_exclusions missing required entries: {missing_exclusions}")
        object.__setattr__(self, "artifact_root", _required_str(self.artifact_root, "artifact_root"))
        object.__setattr__(
            self,
            "report_artifact_root",
            _required_str(self.report_artifact_root, "report_artifact_root"),
        )
        object.__setattr__(self, "v1_scope_exclusions", exclusions)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "schema_id": self.schema_id,
            "required_metadata_sections": list(CANONICAL_REPORT_REQUIRED_METADATA_SECTIONS),
            "required_result_sections": list(CANONICAL_REPORT_REQUIRED_RESULT_SECTIONS),
            "identity": self.identity.to_dict(),
            "universe": self.universe.to_dict(),
            "period": self.period.to_dict(),
            "run_configuration": self.run_configuration.to_dict(),
            "data_provenance": self.data_provenance.to_dict(),
            "report_input_contract": self.report_input_contract.to_dict(),
            "result_schema_sections": {
                "backtest_results": self.backtest_results_schema.to_dict(),
                "walk_forward_validation_metrics": (
                    self.walk_forward_validation_metrics_schema.to_dict()
                ),
                "risk_checks": self.risk_checks_schema.to_dict(),
                "model_feature_summaries": self.model_feature_summaries_schema.to_dict(),
                "deterministic_signal_summary": (
                    self.deterministic_signal_summary_schema.to_dict()
                ),
                "trade_cost_assumptions": self.trade_cost_assumptions_schema.to_dict(),
            },
            "artifact_manifest": {
                "schema": self.artifact_manifest_schema.to_dict(),
                "schema_id": self.artifact_manifest_schema.schema_id,
                "schema_version": self.artifact_manifest_schema.schema_version,
                "artifact_root": self.artifact_root,
                "report_artifact_root": self.report_artifact_root,
                "manifest_required": True,
                "reproducible_input_metadata_required": True,
                "required_metadata_fields": list(
                    self.artifact_manifest_schema.required_metadata_fields
                ),
                "result_sections_required": list(CANONICAL_REPORT_REQUIRED_RESULT_SECTIONS),
            },
            "v1_scope_exclusions": list(self.v1_scope_exclusions),
        }


def build_canonical_report_metadata(
    *,
    experiment_id: str,
    run_id: str,
    universe_snapshot: UniverseSnapshot,
    start_date: date,
    end_date: date,
    data_sources: tuple[ReportDataSource, ...],
    feature_availability_cutoff: Mapping[str, Any],
    created_at: datetime | None = None,
) -> CanonicalReportMetadata:
    return CanonicalReportMetadata(
        identity=ReportIdentity(
            experiment_id=experiment_id,
            run_id=run_id,
            created_at=created_at or datetime.now(UTC),
        ),
        universe=ReportUniverseMetadata(universe_snapshot),
        period=ReportPeriod(start_date=start_date, end_date=end_date),
        run_configuration=ReportRunConfiguration(),
        data_provenance=ReportDataProvenance(
            data_sources=data_sources,
            feature_availability_cutoff=feature_availability_cutoff,
        ),
    )


def default_canonical_report_input_contract() -> CanonicalReportInputContract:
    return CanonicalReportInputContract()


def build_canonical_report_input_contract() -> dict[str, object]:
    return default_canonical_report_input_contract().to_dict()


def default_artifact_manifest_schema() -> CanonicalArtifactManifestSchema:
    return CanonicalArtifactManifestSchema()


def build_artifact_manifest_schema() -> dict[str, object]:
    return default_artifact_manifest_schema().to_dict()


def default_report_backtest_results_schema() -> ReportBacktestResultsSchema:
    return ReportBacktestResultsSchema()


def build_report_backtest_results_schema() -> dict[str, object]:
    return default_report_backtest_results_schema().to_dict()


def default_report_walk_forward_validation_metrics_schema() -> (
    ReportWalkForwardValidationMetricsSchema
):
    return ReportWalkForwardValidationMetricsSchema()


def build_report_walk_forward_validation_metrics_schema() -> dict[str, object]:
    return default_report_walk_forward_validation_metrics_schema().to_dict()


def default_report_risk_checks_schema() -> ReportRiskChecksSchema:
    return ReportRiskChecksSchema()


def build_report_risk_checks_schema() -> dict[str, object]:
    return default_report_risk_checks_schema().to_dict()


def default_report_deterministic_signal_summary_schema() -> (
    ReportDeterministicSignalSummarySchema
):
    return ReportDeterministicSignalSummarySchema()


def build_report_deterministic_signal_summary_schema() -> dict[str, object]:
    return default_report_deterministic_signal_summary_schema().to_dict()


def default_report_model_feature_summaries_schema() -> ReportModelFeatureSummariesSchema:
    return ReportModelFeatureSummariesSchema()


def build_report_model_feature_summaries_schema() -> dict[str, object]:
    return default_report_model_feature_summaries_schema().to_dict()


def default_report_trade_cost_assumptions_schema() -> ReportTradeCostAssumptionsSchema:
    return ReportTradeCostAssumptionsSchema()


def build_report_trade_cost_assumptions_schema() -> dict[str, object]:
    return default_report_trade_cost_assumptions_schema().to_dict()


def _validate_report_section_fields(
    section_id: str,
    required_fields: tuple[ReportMetricField, ...],
    optional_fields: tuple[ReportMetricField, ...],
) -> None:
    if not _required_str(section_id, "section_id"):
        raise ValueError("section_id must not be blank")
    if not required_fields:
        raise ValueError(f"{section_id} must define required_fields")
    for field_value in (*required_fields, *optional_fields):
        if not isinstance(field_value, ReportMetricField):
            raise TypeError(f"{section_id} fields must be ReportMetricField instances")
    names = [field_value.name for field_value in (*required_fields, *optional_fields)]
    if len(names) != len(set(names)):
        raise ValueError(f"{section_id} contains duplicate fields")


def _field_names(fields: tuple[ReportMetricField, ...]) -> tuple[str, ...]:
    return tuple(field_value.name for field_value in fields)


def _deterministic_signal_outputs_input_section() -> ReportInputSectionContract:
    return ReportInputSectionContract(
        section_id="deterministic_signal_outputs",
        description=(
            "Outputs emitted by the deterministic signal engine after model features, "
            "costs, slippage, and risk rules have been converted into final scores."
        ),
        row_grain="ticker and signal date",
        sample_alignment_key=("date", "ticker"),
        required_fields=(
            ReportMetricField("date", "datetime64[ns]"),
            ReportMetricField("ticker", "string"),
            ReportMetricField("signal_score", "float"),
            ReportMetricField("action", "BUY|SELL|HOLD"),
            ReportMetricField("signal_engine", "string"),
            ReportMetricField("cost_bps", "float"),
            ReportMetricField("slippage_bps", "float"),
            ReportMetricField("risk_metric_penalty", "float"),
        ),
        optional_fields=(
            ReportMetricField("signal_generation_gate_status", "PASS|FAIL", "optional"),
            ReportMetricField("signal_generation_gate_decision", "string", "optional"),
            ReportMetricField("portfolio_weight", "float", "optional"),
            ReportMetricField("sector", "string", "optional"),
        ),
    )


def _backtest_evaluation_metrics_input_section() -> ReportInputSectionContract:
    return ReportInputSectionContract(
        section_id="backtest_evaluation_metrics",
        description=(
            "Horizon-consistent long-only backtest metrics produced from deterministic "
            "signals and aligned SPY/equal-weight baselines."
        ),
        row_grain="evaluation date",
        sample_alignment_key=("date",),
        required_fields=(
            ReportMetricField("date", "datetime64[ns]"),
            ReportMetricField("portfolio_return", "float"),
            ReportMetricField("cost_adjusted_return", "float"),
            ReportMetricField("benchmark_return", "float"),
            ReportMetricField("equal_weight_return", "float"),
            ReportMetricField("turnover", "float"),
            ReportMetricField("holdings_count", "integer"),
            ReportMetricField("max_symbol_weight", "float"),
            ReportMetricField("max_sector_weight", "float"),
        ),
        optional_fields=(
            ReportMetricField("gross_return", "float", "optional"),
            ReportMetricField("transaction_cost_return", "float", "optional"),
            ReportMetricField("slippage_cost_return", "float", "optional"),
            ReportMetricField("total_cost_return", "float", "optional"),
        ),
    )


def _walk_forward_validation_metrics_input_section() -> ReportInputSectionContract:
    return ReportInputSectionContract(
        section_id="walk_forward_validation_metrics",
        description=(
            "Fold-level and aggregate OOS validation metrics for forward_return_20 "
            "with purge and embargo evidence."
        ),
        row_grain="walk-forward fold or aggregate OOS summary",
        sample_alignment_key=("fold",),
        required_fields=(
            ReportMetricField("fold", "integer"),
            ReportMetricField("is_oos", "boolean"),
            ReportMetricField("target_column", "string"),
            ReportMetricField("prediction_horizon_periods", "integer"),
            ReportMetricField("purge_periods", "integer"),
            ReportMetricField("embargo_periods", "integer"),
            ReportMetricField("fold_rank_ic", "float"),
            ReportMetricField("positive_rank_ic", "boolean"),
        ),
        optional_fields=(
            ReportMetricField("mean_rank_ic", "float", "optional"),
            ReportMetricField("positive_fold_ratio", "float", "optional"),
            ReportMetricField("oos_fold_count", "integer", "optional"),
            ReportMetricField("proxy_ic_improvement", "float", "optional"),
        ),
    )


def _risk_evaluation_metrics_input_section() -> ReportInputSectionContract:
    return ReportInputSectionContract(
        section_id="risk_evaluation_metrics",
        description="Portfolio risk and constraint metrics evaluated after signal generation.",
        row_grain="evaluation date and risk rule",
        sample_alignment_key=("date", "risk_check"),
        required_fields=(
            ReportMetricField("date", "datetime64[ns]"),
            ReportMetricField("risk_check", "string"),
            ReportMetricField("status", "PASS|WARN|FAIL"),
            ReportMetricField("observed_value", "float|boolean|string"),
            ReportMetricField("threshold_value", "float|boolean|string"),
            ReportMetricField("long_only", "boolean"),
            ReportMetricField("average_daily_turnover", "float"),
            ReportMetricField("max_drawdown", "float"),
        ),
        optional_fields=(
            ReportMetricField("portfolio_volatility", "float", "optional"),
            ReportMetricField("breach_reason", "string", "optional"),
        ),
    )


def _comparison_evaluation_metrics_input_section() -> ReportInputSectionContract:
    return ReportInputSectionContract(
        section_id="comparison_evaluation_metrics",
        description=(
            "Comparative evaluation metrics versus SPY, equal-weight universe, proxy, "
            "and ablation baselines."
        ),
        row_grain="comparison entity and metric",
        sample_alignment_key=("entity_id", "metric"),
        required_fields=(
            ReportMetricField("entity_id", "string"),
            ReportMetricField("entity_role", "strategy|benchmark|baseline|proxy|ablation"),
            ReportMetricField("metric", "string"),
            ReportMetricField("value", "float|string|boolean|null"),
            ReportMetricField("sample_start", "datetime64[ns]"),
            ReportMetricField("sample_end", "datetime64[ns]"),
            ReportMetricField("sample_aligned", "boolean"),
        ),
        optional_fields=(
            ReportMetricField("delta_vs_strategy", "float", "optional"),
            ReportMetricField("status", "PASS|WARN|FAIL", "optional"),
        ),
    )


def _system_validity_gate_input_section() -> ReportInputSectionContract:
    return ReportInputSectionContract(
        section_id="system_validity_gate",
        description="Hard PASS/FAIL system validity gate results and failure reasons.",
        row_grain="gate criterion",
        sample_alignment_key=("criterion_id",),
        required_fields=(
            ReportMetricField("criterion_id", "string"),
            ReportMetricField("status", "PASS|FAIL"),
            ReportMetricField("hard_fail", "boolean"),
            ReportMetricField("reason_code", "string"),
            ReportMetricField("reason", "string"),
        ),
    )


def _strategy_candidate_gate_input_section() -> ReportInputSectionContract:
    return ReportInputSectionContract(
        section_id="strategy_candidate_gate",
        description="PASS/WARN/FAIL strategy candidate gate metrics and decisions.",
        row_grain="strategy candidate rule",
        sample_alignment_key=("rule_id",),
        required_fields=(
            ReportMetricField("rule_id", "string"),
            ReportMetricField("status", "PASS|WARN|FAIL"),
            ReportMetricField("metric", "string"),
            ReportMetricField("value", "float|string|boolean|null"),
            ReportMetricField("threshold", "float|string|boolean|null"),
            ReportMetricField("reason", "string"),
        ),
    )


def _artifact_manifest_input_section() -> ReportInputSectionContract:
    return ReportInputSectionContract(
        section_id="artifact_manifest",
        description="Reproducible input and output artifact manifest for the report run.",
        row_grain="artifact",
        sample_alignment_key=("artifact_id",),
        required_fields=(
            ReportMetricField("artifact_id", "string"),
            ReportMetricField("artifact_type", "string"),
            ReportMetricField("path", "string"),
            ReportMetricField("created_at", "datetime64[ns, UTC]"),
            ReportMetricField("content_hash", "string"),
        ),
        optional_fields=(
            ReportMetricField("schema_version", "string", "optional"),
            ReportMetricField("row_count", "integer", "optional"),
        ),
    )


def _report_section_to_dict(
    *,
    schema_version: str,
    section_id: str,
    description: str,
    row_grain: str,
    sample_alignment_key: tuple[str, ...],
    required_fields: tuple[ReportMetricField, ...],
    optional_fields: tuple[ReportMetricField, ...],
    validation_rules: Mapping[str, Any],
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "section_id": section_id,
        "description": description,
        "row_grain": row_grain,
        "sample_alignment_key": list(sample_alignment_key),
        "required_fields": [field_value.name for field_value in required_fields],
        "optional_fields": [field_value.name for field_value in optional_fields],
        "fields": [
            field_value.to_dict() for field_value in (*required_fields, *optional_fields)
        ],
        "validation_rules": dict(validation_rules),
    }


def _required_str(value: object, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must not be blank")
    return text


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_date(value: object, field_name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError(f"{field_name} must be a date or ISO date string")


def _coerce_datetime(value: object, field_name: str) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        result = datetime.fromisoformat(value)
    else:
        raise TypeError(f"{field_name} must be a datetime or ISO datetime string")
    if result.tzinfo is None:
        result = result.replace(tzinfo=UTC)
    return result.astimezone(UTC)
