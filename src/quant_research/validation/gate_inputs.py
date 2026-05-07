from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from quant_research.validation.config import (
    CANONICAL_DETERMINISTIC_VALIDATION_DEFAULTS,
    CANONICAL_MIN_POSITIVE_FOLD_RATIO,
    DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG,
    DEFAULT_TRANSACTION_COST_SENSITIVITY_CONFIG,
)
from quant_research.validation.horizons import REQUIRED_VALIDATION_HORIZON_DAYS
from quant_research.validation.universe import (
    DEFAULT_UNIVERSE_SELECTION_COUNT,
    DEFAULT_UNIVERSE_SELECTION_METHOD,
    DEFAULT_UNIVERSE_SURVIVORSHIP_BIAS_DISCLOSURE,
    UNIVERSE_SNAPSHOT_SCHEMA_VERSION,
)
from quant_research.validation.walk_forward import (
    CANONICAL_WALK_FORWARD_EMBARGO_PERIODS,
    CANONICAL_WALK_FORWARD_PURGE_PERIODS,
    CANONICAL_WALK_FORWARD_TARGET_COLUMN,
    CANONICAL_WALK_FORWARD_TEST_PERIODS,
    CANONICAL_WALK_FORWARD_TRAIN_PERIODS,
)

SYSTEM_VALIDITY_GATE_INPUT_SCHEMA_VERSION = "system_validity_gate_input.v1"
SYSTEM_VALIDITY_GATE_SCHEMA_ID = "stage1_system_validity_gate_inputs"
SYSTEM_VALIDITY_GATE_REQUIRED_INPUT_SECTIONS: tuple[str, ...] = (
    "experiment",
    "universe_snapshot",
    "predictions",
    "validation_summary",
    "equity_curve",
    "backtest_results",
    "walk_forward_results",
    "out_of_sample_results",
    "risk_rule_results",
    "strategy_metrics",
    "feature_availability_cutoff",
    "walk_forward_config",
    "portfolio_constraints",
    "transaction_costs",
    "benchmark_config",
    "comparison_inputs",
    "scope_bounds",
)

ColumnRequirement = Literal["required", "optional"]


@dataclass(frozen=True, slots=True)
class GateInputColumn:
    name: str
    dtype: str
    requirement: ColumnRequirement = "required"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("input column name must not be blank")
        if not self.dtype.strip():
            raise ValueError("input column dtype must not be blank")
        if self.requirement not in {"required", "optional"}:
            raise ValueError("input column requirement must be required or optional")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GateInputFrameSchema:
    frame_id: str
    description: str
    columns: tuple[GateInputColumn, ...]
    row_grain: str
    sample_alignment_key: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.frame_id.strip():
            raise ValueError("frame_id must not be blank")
        if not self.description.strip():
            raise ValueError("frame description must not be blank")
        if not self.columns:
            raise ValueError("frame schema must include at least one column")
        if not self.row_grain.strip():
            raise ValueError("row_grain must not be blank")
        names = [column.name for column in self.columns]
        if len(set(names)) != len(names):
            raise ValueError(f"frame {self.frame_id} contains duplicate columns")

    def required_columns(self) -> tuple[str, ...]:
        return tuple(
            column.name for column in self.columns if column.requirement == "required"
        )

    def optional_columns(self) -> tuple[str, ...]:
        return tuple(
            column.name for column in self.columns if column.requirement == "optional"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "description": self.description,
            "row_grain": self.row_grain,
            "sample_alignment_key": list(self.sample_alignment_key),
            "required_columns": list(self.required_columns()),
            "optional_columns": list(self.optional_columns()),
            "columns": [column.to_dict() for column in self.columns],
        }


@dataclass(frozen=True, slots=True)
class SystemValidityGateInputSchema:
    schema_id: str = SYSTEM_VALIDITY_GATE_SCHEMA_ID
    schema_version: str = SYSTEM_VALIDITY_GATE_INPUT_SCHEMA_VERSION
    required_sections: tuple[str, ...] = SYSTEM_VALIDITY_GATE_REQUIRED_INPUT_SECTIONS
    experiment: dict[str, object] = field(default_factory=lambda: _experiment_section())
    universe_snapshot: dict[str, object] = field(default_factory=lambda: _universe_section())
    predictions: GateInputFrameSchema = field(default_factory=lambda: _predictions_frame())
    validation_summary: GateInputFrameSchema = field(
        default_factory=lambda: _validation_summary_frame()
    )
    equity_curve: GateInputFrameSchema = field(default_factory=lambda: _equity_curve_frame())
    backtest_results: GateInputFrameSchema = field(
        default_factory=lambda: _backtest_results_frame()
    )
    walk_forward_results: GateInputFrameSchema = field(
        default_factory=lambda: _walk_forward_results_frame()
    )
    out_of_sample_results: GateInputFrameSchema = field(
        default_factory=lambda: _out_of_sample_results_frame()
    )
    risk_rule_results: GateInputFrameSchema = field(
        default_factory=lambda: _risk_rule_results_frame()
    )
    strategy_metrics: dict[str, object] = field(
        default_factory=lambda: _strategy_metrics_section()
    )
    feature_availability_cutoff: dict[str, object] = field(
        default_factory=lambda: _feature_availability_section()
    )
    walk_forward_config: dict[str, object] = field(default_factory=lambda: _walk_forward_section())
    portfolio_constraints: dict[str, object] = field(default_factory=lambda: _portfolio_section())
    transaction_costs: dict[str, object] = field(
        default_factory=lambda: _transaction_costs_section()
    )
    benchmark_config: dict[str, object] = field(default_factory=lambda: _benchmark_section())
    comparison_inputs: dict[str, object] = field(default_factory=lambda: _comparison_section())
    scope_bounds: dict[str, object] = field(default_factory=lambda: _scope_bounds_section())

    def __post_init__(self) -> None:
        if self.schema_version != SYSTEM_VALIDITY_GATE_INPUT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {SYSTEM_VALIDITY_GATE_INPUT_SCHEMA_VERSION!r}"
            )
        missing = [
            section for section in SYSTEM_VALIDITY_GATE_REQUIRED_INPUT_SECTIONS
            if not hasattr(self, section)
        ]
        if missing:
            raise ValueError(f"missing system validity gate input sections: {missing}")
        if self.walk_forward_config["target_column"] != CANONICAL_WALK_FORWARD_TARGET_COLUMN:
            raise ValueError("walk_forward_config target_column must be forward_return_20")
        if int(self.walk_forward_config["embargo_periods"]) < REQUIRED_VALIDATION_HORIZON_DAYS:
            raise ValueError("embargo_periods must be at least the target horizon")
        if self.benchmark_config["benchmark_ticker"] != "SPY":
            raise ValueError("benchmark_config benchmark_ticker must be SPY")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "required_sections": list(self.required_sections),
            "experiment": dict(self.experiment),
            "universe_snapshot": dict(self.universe_snapshot),
            "predictions": self.predictions.to_dict(),
            "validation_summary": self.validation_summary.to_dict(),
            "equity_curve": self.equity_curve.to_dict(),
            "backtest_results": self.backtest_results.to_dict(),
            "walk_forward_results": self.walk_forward_results.to_dict(),
            "out_of_sample_results": self.out_of_sample_results.to_dict(),
            "risk_rule_results": self.risk_rule_results.to_dict(),
            "strategy_metrics": dict(self.strategy_metrics),
            "feature_availability_cutoff": dict(self.feature_availability_cutoff),
            "walk_forward_config": dict(self.walk_forward_config),
            "portfolio_constraints": dict(self.portfolio_constraints),
            "transaction_costs": dict(self.transaction_costs),
            "benchmark_config": dict(self.benchmark_config),
            "comparison_inputs": dict(self.comparison_inputs),
            "scope_bounds": dict(self.scope_bounds),
        }


def default_system_validity_gate_input_schema() -> SystemValidityGateInputSchema:
    return SystemValidityGateInputSchema()


def build_system_validity_gate_input_schema() -> dict[str, object]:
    return default_system_validity_gate_input_schema().to_dict()


def _column(
    name: str,
    dtype: str,
    requirement: ColumnRequirement = "required",
    description: str = "",
) -> GateInputColumn:
    return GateInputColumn(
        name=name,
        dtype=dtype,
        requirement=requirement,
        description=description,
    )


def _experiment_section() -> dict[str, object]:
    return {
        "required_fields": [
            "experiment_id",
            "target_horizon",
            "diagnostic_horizons",
            "created_at",
        ],
        "target_horizon": "forward_return_20",
        "diagnostic_horizons": ["forward_return_1", "forward_return_5"],
        "minimum_history_years": 3,
        "model_predictions_are_order_signals": False,
        "llm_makes_trading_decisions": False,
        "signal_engine": CANONICAL_DETERMINISTIC_VALIDATION_DEFAULTS.signal_engine,
    }


def _universe_section() -> dict[str, object]:
    return {
        "required_artifact": "universe_snapshot",
        "schema_version": UNIVERSE_SNAPSHOT_SCHEMA_VERSION,
        "selection_method": DEFAULT_UNIVERSE_SELECTION_METHOD,
        "selection_count": DEFAULT_UNIVERSE_SELECTION_COUNT,
        "fixed_at_experiment_start": True,
        "survivorship_bias_allowed_v1": True,
        "survivorship_bias_disclosure": DEFAULT_UNIVERSE_SURVIVORSHIP_BIAS_DISCLOSURE,
        "point_in_time_universe_scope": "v2",
    }


def _predictions_frame() -> GateInputFrameSchema:
    return GateInputFrameSchema(
        frame_id="predictions",
        description="Model scores and realized forward returns used by the deterministic gate.",
        row_grain="date,ticker prediction observation",
        sample_alignment_key=("date", "ticker"),
        columns=(
            _column("date", "datetime64[ns]"),
            _column("ticker", "string"),
            _column("fold", "integer"),
            _column("is_oos", "boolean"),
            _column("expected_return", "float", description="Model score, not an order signal."),
            _column("forward_return_20", "float"),
            _column("forward_return_1", "float", "optional"),
            _column("forward_return_5", "float", "optional"),
            _column("feature_timestamp", "datetime64[ns, UTC]", "optional"),
            _column("feature_availability_timestamp", "datetime64[ns, UTC]", "optional"),
        ),
    )


def _validation_summary_frame() -> GateInputFrameSchema:
    return GateInputFrameSchema(
        frame_id="validation_summary",
        description="Walk-forward fold boundaries and purge/embargo evidence.",
        row_grain="walk-forward fold",
        sample_alignment_key=("fold",),
        columns=(
            _column("fold", "integer"),
            _column("train_end", "datetime64[ns]"),
            _column("test_start", "datetime64[ns]"),
            _column("is_oos", "boolean"),
            _column("labeled_test_observations", "integer"),
            _column("train_observations", "integer"),
            _column("target_column", "string", "optional"),
            _column("prediction_horizon_periods", "integer", "optional"),
            _column("gap_periods", "integer", "optional"),
            _column("purge_periods", "integer", "optional"),
            _column("purged_date_count", "integer", "optional"),
            _column("purge_applied", "boolean", "optional"),
            _column("embargo_periods", "integer", "optional"),
            _column("embargoed_date_count", "integer", "optional"),
            _column("embargo_applied", "boolean", "optional"),
        ),
    )


def _equity_curve_frame() -> GateInputFrameSchema:
    return GateInputFrameSchema(
        frame_id="equity_curve",
        description="Long-only strategy return series after deterministic signal conversion.",
        row_grain="evaluation date",
        sample_alignment_key=("date",),
        columns=(
            _column("date", "datetime64[ns]"),
            _column("portfolio_return", "float"),
            _column("cost_adjusted_return", "float"),
            _column("benchmark_return", "float"),
            _column("turnover", "float"),
            _column("gross_return", "float", "optional"),
            _column("transaction_cost_return", "float", "optional"),
            _column("slippage_cost_return", "float", "optional"),
            _column("total_cost_return", "float", "optional"),
            _column("position_count", "integer", "optional"),
            _column("max_position_weight", "float", "optional"),
            _column("max_sector_exposure", "float", "optional"),
        ),
    )


def _backtest_results_frame() -> GateInputFrameSchema:
    return GateInputFrameSchema(
        frame_id="backtest_results",
        description="Horizon-consistent long-only backtest results after deterministic signals, costs, and slippage.",
        row_grain="evaluation date",
        sample_alignment_key=("date",),
        columns=(
            _column("date", "datetime64[ns]"),
            _column("portfolio_return", "float"),
            _column("cost_adjusted_return", "float"),
            _column("benchmark_return", "float"),
            _column("equal_weight_return", "float"),
            _column("turnover", "float"),
            _column("holdings_count", "integer"),
            _column("max_symbol_weight", "float"),
            _column("max_sector_weight", "float"),
            _column("rebalance_interval", "integer|string", "optional"),
            _column("cost_bps", "float", "optional"),
            _column("slippage_bps", "float", "optional"),
        ),
    )


def _walk_forward_results_frame() -> GateInputFrameSchema:
    return GateInputFrameSchema(
        frame_id="walk_forward_results",
        description="Walk-forward fold validation results with purge and embargo evidence.",
        row_grain="walk-forward fold",
        sample_alignment_key=("fold",),
        columns=(
            _column("fold", "integer"),
            _column("train_start", "datetime64[ns]", "optional"),
            _column("train_end", "datetime64[ns]"),
            _column("test_start", "datetime64[ns]"),
            _column("test_end", "datetime64[ns]", "optional"),
            _column("is_oos", "boolean"),
            _column("target_column", "string"),
            _column("prediction_horizon_periods", "integer"),
            _column("purge_periods", "integer"),
            _column("embargo_periods", "integer"),
            _column("labeled_test_observations", "integer"),
            _column("train_observations", "integer"),
            _column("fold_rank_ic", "float", "optional"),
            _column("status", "GateRuleStatus", "optional"),
        ),
    )


def _out_of_sample_results_frame() -> GateInputFrameSchema:
    return GateInputFrameSchema(
        frame_id="out_of_sample_results",
        description="OOS fold and sample metrics used by the system and strategy candidate gates.",
        row_grain="OOS walk-forward fold",
        sample_alignment_key=("fold",),
        columns=(
            _column("fold", "integer"),
            _column("is_oos", "boolean"),
            _column("fold_rank_ic", "float"),
            _column("positive_rank_ic", "boolean"),
            _column("cost_adjusted_excess_return_vs_spy", "float"),
            _column("cost_adjusted_excess_return_vs_equal_weight", "float"),
            _column("max_drawdown", "float"),
            _column("average_daily_turnover", "float"),
            _column("status", "GateRuleStatus", "optional"),
        ),
    )


def _risk_rule_results_frame() -> GateInputFrameSchema:
    return GateInputFrameSchema(
        frame_id="risk_rule_results",
        description="Deterministic long-only portfolio risk rule results used before candidate approval.",
        row_grain="risk rule evaluation",
        sample_alignment_key=("rule_id",),
        columns=(
            _column("rule_id", "string"),
            _column("rule_group", "string"),
            _column("status", "GateRuleStatus"),
            _column("passed", "boolean"),
            _column("metric", "string"),
            _column("value", "float|integer|null"),
            _column("threshold", "float|integer|null"),
            _column("operator", "string"),
            _column("reason_code", "string", "optional"),
            _column("reason", "string", "optional"),
        ),
    )


def _strategy_metrics_section() -> dict[str, object]:
    return {
        "required_attributes": ["cagr", "sharpe", "max_drawdown", "turnover"],
        "turnover_field": "average_daily_turnover or turnover",
        "max_drawdown_pass_floor": -0.20,
        "average_daily_turnover_max": 0.25,
    }


def _feature_availability_section() -> dict[str, object]:
    return {
        "required": True,
        "rule": "features must be available at or before date_end(t); positions apply to t+1 or later returns",
        "required_cutoff_fields": [
            "feature_timestamp",
            "feature_availability_timestamp",
            "source_release_timestamp",
        ],
        "fail_on_future_availability": True,
    }


def _walk_forward_section() -> dict[str, object]:
    return {
        "train_periods": CANONICAL_WALK_FORWARD_TRAIN_PERIODS,
        "test_periods": CANONICAL_WALK_FORWARD_TEST_PERIODS,
        "purge_periods": CANONICAL_WALK_FORWARD_PURGE_PERIODS,
        "embargo_periods": CANONICAL_WALK_FORWARD_EMBARGO_PERIODS,
        "target_column": CANONICAL_WALK_FORWARD_TARGET_COLUMN,
        "target_horizon_periods": REQUIRED_VALIDATION_HORIZON_DAYS,
        "required_min_oos_folds": 2,
        "embargo_zero_for_forward_return_20_is_hard_fail": True,
        "evaluation_mode": "non_overlapping_or_horizon_consistent",
    }


def _portfolio_section() -> dict[str, object]:
    config = DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG
    payload = config.to_dict()
    payload["long_only"] = True
    payload["correlation_cluster_weight_scope"] = "v1_excluded"
    return payload


def _transaction_costs_section() -> dict[str, object]:
    config = DEFAULT_TRANSACTION_COST_SENSITIVITY_CONFIG
    return {
        "cost_bps": 5.0,
        "slippage_bps": 2.0,
        "average_daily_turnover_budget": 0.25,
        "sensitivity_config": config.to_dict(),
    }


def _benchmark_section() -> dict[str, object]:
    return {
        "benchmark_ticker": "SPY",
        "required_baselines": ["SPY", "equal_weight_universe"],
        "alignment_required": True,
        "strategy_excess_return_vs_spy_must_be_positive": True,
        "strategy_excess_return_vs_equal_weight_must_be_positive": True,
    }


def _comparison_section() -> dict[str, object]:
    return {
        "proxy_comparison_required": True,
        "proxy_ic_improvement_min": 0.01,
        "ablation_results_required": True,
        "required_ablation_scenarios": [
            "price_only",
            "text_only",
            "sec_only",
            "no_model_proxy",
            "no_costs",
        ],
        "positive_fold_ratio_min": CANONICAL_MIN_POSITIVE_FOLD_RATIO,
    }


def _scope_bounds_section() -> dict[str, object]:
    return {
        "real_trading_orders": "excluded",
        "llm_trade_decisions": "excluded",
        "point_in_time_universe": "v2",
        "correlation_cluster_weight": "v1_5",
        "top_decile_20d_excess_return": "report_only",
        "sec_cik_mapping_150": "separate_check",
        "llm_inference_time": "separate_check",
    }
