from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Literal

import pandas as pd

from quant_research.validation.benchmark_inputs import (
    DEFAULT_BENCHMARK_TICKER,
    EQUAL_WEIGHT_BASELINE_NAME,
    BaselineComparisonInput,
    StrategyEvaluationWindow,
    build_stage1_baseline_comparison_inputs,
)
from quant_research.validation.config import (
    CANONICAL_DETERMINISTIC_VALIDATION_DEFAULTS,
    DETERMINISTIC_SIGNAL_ENGINE_ID,
    MODEL_COMPARISON_ID_PATTERN,
    ModelComparisonConfig,
    default_model_comparison_config,
)
from quant_research.validation.horizons import (
    REQUIRED_VALIDATION_HORIZON_DAYS,
    forward_return_column,
)

STAGE1_COMPARISON_SCHEMA_VERSION = "stage1_validity_gate.v1"
STAGE1_COMPARISON_CONTRACT_ID = "stage1_validity_gate_comparison"
STAGE1_COMPARISON_RETURN_TIMING = "positions_apply_to_t_plus_1_or_later_returns"
REPORT_COMPARISON_TABLE_SCHEMA_VERSION = "report_comparison_table.v1"

ComparisonEntityRole = Literal[
    "full_model",
    "model_baseline",
    "return_baseline",
    "ablation",
    "diagnostic",
]
ComparisonStatus = Literal[
    "configured",
    "pass",
    "fail",
    "warning",
    "hard_fail",
    "not_evaluable",
    "insufficient_data",
    "skipped",
]
MetricDirection = Literal["higher_is_better", "lower_is_better"]
ValidationWindowRole = Literal[
    "configured_walk_forward",
    "walk_forward_fold",
    "oos_holdout",
    "strategy_evaluation",
]

VALID_COMPARISON_ENTITY_ROLES: frozenset[str] = frozenset(
    {"full_model", "model_baseline", "return_baseline", "ablation", "diagnostic"}
)
VALID_COMPARISON_STATUSES: frozenset[str] = frozenset(
    {
        "configured",
        "pass",
        "fail",
        "warning",
        "hard_fail",
        "not_evaluable",
        "insufficient_data",
        "skipped",
    }
)
VALID_METRIC_DIRECTIONS: frozenset[str] = frozenset(
    {"higher_is_better", "lower_is_better"}
)
VALID_VALIDATION_WINDOW_ROLES: frozenset[str] = frozenset(
    {"configured_walk_forward", "walk_forward_fold", "oos_holdout", "strategy_evaluation"}
)

LOWER_IS_BETTER_COMPARISON_METRICS: frozenset[str] = frozenset({"turnover"})
DEFAULT_COMPARISON_METRIC_LABELS: dict[str, str] = {
    "rank_ic": "Rank IC",
    "positive_fold_ratio": "Positive Fold Ratio",
    "oos_rank_ic": "OOS Rank IC",
    "sharpe": "Sharpe",
    "max_drawdown": "Max Drawdown",
    "cost_adjusted_cumulative_return": "Cost-Adjusted Cumulative Return",
    "excess_return": "Excess Return",
    "turnover": "Turnover",
    "average_daily_turnover": "Average Daily Turnover",
    "total_cost_return": "Total Cost Return",
    "mean_rank_ic": "Mean Rank IC",
}
REPORT_COMPARISON_DEFAULT_METRICS: tuple[str, ...] = (
    "cost_adjusted_cumulative_return",
    "excess_return",
    "sharpe",
    "max_drawdown",
    "average_daily_turnover",
    "total_cost_return",
    "mean_rank_ic",
    "positive_fold_ratio",
    "oos_rank_ic",
)
REPORT_COMPARISON_TABLE_COLUMNS: tuple[str, ...] = (
    "schema_version",
    "entity_id",
    "label",
    "role",
    "result_type",
    "result_source",
    "baseline_type",
    "ablation_scenario_id",
    "metric",
    "metric_label",
    "metric_direction",
    "value",
    "value_type",
    "status",
    "return_basis",
    "target_column",
    "target_horizon",
    "evaluation_start",
    "evaluation_end",
    "evaluation_observations",
    "cost_bps",
    "slippage_bps",
)


@dataclass(frozen=True, slots=True)
class ComparisonValidationWindow:
    window_id: str
    label: str
    role: ValidationWindowRole
    target_column: str
    target_horizon: int
    train_periods: int | None = None
    test_periods: int | None = None
    gap_periods: int = REQUIRED_VALIDATION_HORIZON_DAYS
    embargo_periods: int = REQUIRED_VALIDATION_HORIZON_DAYS
    start: date | str | pd.Timestamp | None = None
    end: date | str | pd.Timestamp | None = None
    train_start: date | str | pd.Timestamp | None = None
    train_end: date | str | pd.Timestamp | None = None
    test_start: date | str | pd.Timestamp | None = None
    test_end: date | str | pd.Timestamp | None = None
    is_oos: bool = False
    status: ComparisonStatus = "configured"
    train_observations: int | None = None
    test_observations: int | None = None
    labeled_test_observations: int | None = None
    prediction_count: int | None = None
    return_timing: str = STAGE1_COMPARISON_RETURN_TIMING
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.window_id).strip():
            raise ValueError("validation window_id must not be blank")
        if not str(self.label).strip():
            raise ValueError("validation window label must not be blank")
        if self.role not in VALID_VALIDATION_WINDOW_ROLES:
            raise ValueError(f"unsupported validation window role: {self.role}")
        target_horizon = max(int(self.target_horizon), 1)
        expected_target = forward_return_column(target_horizon)
        if self.target_column != expected_target:
            raise ValueError(
                f"target_column must be {expected_target!r} for target_horizon={target_horizon}"
            )
        if self.gap_periods < target_horizon:
            raise ValueError("gap_periods must be at least the target horizon")
        if self.embargo_periods < target_horizon:
            raise ValueError("embargo_periods must be at least the target horizon")
        if self.status not in VALID_COMPARISON_STATUSES:
            raise ValueError(f"unsupported validation window status: {self.status}")
        if "t_plus_1" not in self.return_timing:
            raise ValueError("return_timing must document t+1 or later return application")

        normalized_dates = {
            "start": _optional_iso_date(self.start),
            "end": _optional_iso_date(self.end),
            "train_start": _optional_iso_date(self.train_start),
            "train_end": _optional_iso_date(self.train_end),
            "test_start": _optional_iso_date(self.test_start),
            "test_end": _optional_iso_date(self.test_end),
        }
        start = normalized_dates["start"]
        end = normalized_dates["end"]
        if start is not None and end is not None and start > end:
            raise ValueError("validation window start must be on or before end")
        train_end = normalized_dates["train_end"]
        test_start = normalized_dates["test_start"]
        if train_end is not None and test_start is not None and train_end >= test_start:
            raise ValueError("training window must end before the test window starts")
        test_start_value = normalized_dates["test_start"]
        test_end_value = normalized_dates["test_end"]
        if (
            test_start_value is not None
            and test_end_value is not None
            and test_start_value > test_end_value
        ):
            raise ValueError("test window start must be on or before test window end")

        object.__setattr__(self, "target_horizon", target_horizon)
        for name, value in normalized_dates.items():
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "label": self.label,
            "role": self.role,
            "target_column": self.target_column,
            "target_horizon": self.target_horizon,
            "train_periods": self.train_periods,
            "test_periods": self.test_periods,
            "gap_periods": self.gap_periods,
            "embargo_periods": self.embargo_periods,
            "start": self.start,
            "end": self.end,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "is_oos": self.is_oos,
            "status": self.status,
            "train_observations": self.train_observations,
            "test_observations": self.test_observations,
            "labeled_test_observations": self.labeled_test_observations,
            "prediction_count": self.prediction_count,
            "return_timing": self.return_timing,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ComparisonMetricSchema:
    metric_id: str
    label: str
    direction: MetricDirection
    value_type: str = "float"
    source: str = "validity_gate"
    required_for_stage1: bool = True
    nullable: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        if not MODEL_COMPARISON_ID_PATTERN.fullmatch(self.metric_id):
            raise ValueError("metric_id must be stable snake_case starting with a letter")
        if not self.label.strip():
            raise ValueError("metric label must not be blank")
        if self.direction not in VALID_METRIC_DIRECTIONS:
            raise ValueError(f"unsupported metric direction: {self.direction}")
        if not self.value_type.strip():
            raise ValueError("metric value_type must not be blank")
        if not self.source.strip():
            raise ValueError("metric source must not be blank")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ComparisonEntityInputSchema:
    entity_id: str
    label: str
    role: ComparisonEntityRole
    comparison_candidate_id: str = ""
    model_config_id: str = ""
    ablation_scenario_id: str = ""
    baseline_type: str = ""
    return_basis: str = ""
    return_column: str = ""
    return_horizon: int | None = None
    data_source: str = ""
    evaluation_window: StrategyEvaluationWindow | Mapping[str, Any] | None = None
    benchmark_ticker: str | None = None
    universe_tickers: tuple[str, ...] = ()
    adapters: tuple[str, ...] = ()
    optional_adapters: tuple[str, ...] = ()
    fallback_adapters: tuple[str, ...] = ()
    modalities: tuple[str, ...] = ()
    feature_families: tuple[str, ...] = ()
    comparison_method: str = ""
    contribution_kind: str = ""
    required_for_stage1: bool = True
    affects_signal_engine: bool = True
    requires_heavy_model: bool = False
    signal_engine: str = DETERMINISTIC_SIGNAL_ENGINE_ID
    model_predictions_are_order_signals: bool = False
    llm_makes_trading_decisions: bool = False
    return_timing: str = STAGE1_COMPARISON_RETURN_TIMING
    description: str = ""

    def __post_init__(self) -> None:
        if not str(self.entity_id).strip():
            raise ValueError("comparison entity_id must not be blank")
        if not str(self.label).strip():
            raise ValueError("comparison entity label must not be blank")
        if self.role not in VALID_COMPARISON_ENTITY_ROLES:
            raise ValueError(f"unsupported comparison entity role: {self.role}")
        if self.requires_heavy_model:
            raise ValueError("Stage 1 comparison entities must keep heavy models optional")
        if self.signal_engine != DETERMINISTIC_SIGNAL_ENGINE_ID:
            raise ValueError("comparison entities must use deterministic_signal_engine")
        if self.model_predictions_are_order_signals:
            raise ValueError("model predictions must not be treated as order signals")
        if self.llm_makes_trading_decisions:
            raise ValueError("LLM adapters must not make trading decisions")
        if "t_plus_1" not in self.return_timing:
            raise ValueError("return_timing must document t+1 or later return application")
        if self.role == "full_model" and not self.comparison_candidate_id:
            raise ValueError("full_model comparison entity requires comparison_candidate_id")
        if self.role == "return_baseline" and (not self.baseline_type or not self.return_basis):
            raise ValueError("return baseline comparison entity requires type and return basis")
        object.__setattr__(self, "universe_tickers", _tuple_strings(self.universe_tickers))
        for field_name in (
            "adapters",
            "optional_adapters",
            "fallback_adapters",
            "modalities",
            "feature_families",
        ):
            object.__setattr__(self, field_name, _tuple_strings(getattr(self, field_name)))

    def to_dict(self) -> dict[str, Any]:
        evaluation_window = self.evaluation_window
        if hasattr(evaluation_window, "to_dict"):
            evaluation_window_payload = evaluation_window.to_dict()
        elif isinstance(evaluation_window, Mapping):
            evaluation_window_payload = dict(evaluation_window)
        else:
            evaluation_window_payload = None
        return {
            "entity_id": self.entity_id,
            "label": self.label,
            "role": self.role,
            "comparison_candidate_id": self.comparison_candidate_id,
            "model_config_id": self.model_config_id,
            "ablation_scenario_id": self.ablation_scenario_id,
            "baseline_type": self.baseline_type,
            "return_basis": self.return_basis,
            "return_column": self.return_column,
            "return_horizon": self.return_horizon,
            "data_source": self.data_source,
            "evaluation_window": evaluation_window_payload,
            "benchmark_ticker": self.benchmark_ticker,
            "universe_tickers": list(self.universe_tickers),
            "adapters": list(self.adapters),
            "optional_adapters": list(self.optional_adapters),
            "fallback_adapters": list(self.fallback_adapters),
            "modalities": list(self.modalities),
            "feature_families": list(self.feature_families),
            "comparison_method": self.comparison_method,
            "contribution_kind": self.contribution_kind,
            "required_for_stage1": self.required_for_stage1,
            "affects_signal_engine": self.affects_signal_engine,
            "requires_heavy_model": self.requires_heavy_model,
            "signal_engine": self.signal_engine,
            "model_predictions_are_order_signals": self.model_predictions_are_order_signals,
            "llm_makes_trading_decisions": self.llm_makes_trading_decisions,
            "return_timing": self.return_timing,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class ComparisonMetricResultSchema:
    metric: str
    candidate: str
    candidate_value: float | None
    baseline: str
    baseline_value: float | None
    delta: float | None
    operator: str
    status: ComparisonStatus
    passed: bool | None = None
    window_id: str = "strategy_evaluation"
    window_label: str = "Strategy evaluation"
    window_role: str = "strategy_evaluation"
    baseline_label: str = ""
    baseline_role: ComparisonEntityRole | str = ""
    absolute_delta: float | None = None
    relative_delta: float | None = None
    improvement: float | None = None
    outperformance_threshold: float | None = None
    candidate_scenario: str = ""
    baseline_scenario: str = ""

    def __post_init__(self) -> None:
        if not str(self.metric).strip():
            raise ValueError("metric result metric must not be blank")
        if not str(self.candidate).strip():
            raise ValueError("metric result candidate must not be blank")
        if not str(self.baseline).strip():
            raise ValueError("metric result baseline must not be blank")
        if self.status not in VALID_COMPARISON_STATUSES:
            raise ValueError(f"unsupported metric result status: {self.status}")

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> ComparisonMetricResultSchema:
        return cls(
            metric=str(row.get("metric", "")),
            candidate=str(row.get("candidate_id") or row.get("candidate") or ""),
            candidate_value=_optional_float(row.get("candidate_value")),
            baseline=str(row.get("baseline_id") or row.get("baseline") or ""),
            baseline_value=_optional_float(row.get("baseline_value")),
            delta=_optional_float(row.get("delta", row.get("absolute_delta"))),
            operator=str(row.get("operator") or ""),
            status=_normalize_status(row.get("status") or row.get("pass_fail")),
            passed=_optional_bool(row.get("passed")),
            window_id=str(row.get("window_id") or "strategy_evaluation"),
            window_label=str(row.get("window_label") or "Strategy evaluation"),
            window_role=str(row.get("window_role") or "strategy_evaluation"),
            baseline_label=str(row.get("baseline_label") or ""),
            baseline_role=str(row.get("baseline_role") or ""),
            absolute_delta=_optional_float(row.get("absolute_delta", row.get("delta"))),
            relative_delta=_optional_float(row.get("relative_delta")),
            improvement=_optional_float(row.get("improvement")),
            outperformance_threshold=_optional_float(row.get("outperformance_threshold")),
            candidate_scenario=str(row.get("candidate_scenario") or ""),
            baseline_scenario=str(row.get("baseline_scenario") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ComparisonEntityResultSchema:
    entity_id: str
    label: str
    role: ComparisonEntityRole
    status: ComparisonStatus
    metrics: dict[str, Any] = field(default_factory=dict)
    sample_count: int | None = None
    evaluation_window: dict[str, Any] | None = None
    ablation_scenario_id: str = ""
    baseline_type: str = ""
    result_source: str = ""

    def __post_init__(self) -> None:
        if not str(self.entity_id).strip():
            raise ValueError("comparison result entity_id must not be blank")
        if not str(self.label).strip():
            raise ValueError("comparison result label must not be blank")
        if self.role not in VALID_COMPARISON_ENTITY_ROLES:
            raise ValueError(f"unsupported comparison result role: {self.role}")
        if self.status not in VALID_COMPARISON_STATUSES:
            raise ValueError(f"unsupported comparison result status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "label": self.label,
            "role": self.role,
            "status": self.status,
            "metrics": dict(self.metrics),
            "sample_count": self.sample_count,
            "evaluation_window": self.evaluation_window,
            "ablation_scenario_id": self.ablation_scenario_id,
            "baseline_type": self.baseline_type,
            "result_source": self.result_source,
        }


@dataclass(frozen=True, slots=True)
class ComparisonInputSchema:
    schema_version: str
    comparison_id: str
    full_model: ComparisonEntityInputSchema
    baselines: tuple[ComparisonEntityInputSchema, ...]
    ablations: tuple[ComparisonEntityInputSchema, ...]
    metrics: tuple[ComparisonMetricSchema, ...]
    validation_windows: tuple[ComparisonValidationWindow, ...]

    def __post_init__(self) -> None:
        if self.schema_version != STAGE1_COMPARISON_SCHEMA_VERSION:
            raise ValueError("unsupported comparison schema_version")
        if not MODEL_COMPARISON_ID_PATTERN.fullmatch(self.comparison_id):
            raise ValueError("comparison_id must be stable snake_case starting with a letter")
        if self.full_model.role != "full_model":
            raise ValueError("full_model must use role='full_model'")
        if not self.baselines:
            raise ValueError("comparison input schema must include baselines")
        if not self.metrics:
            raise ValueError("comparison input schema must include metrics")
        if not self.validation_windows:
            raise ValueError("comparison input schema must include validation windows")
        _require_unique(
            (
                self.full_model.entity_id,
                *(baseline.entity_id for baseline in self.baselines),
                *(ablation.entity_id for ablation in self.ablations),
            ),
            "comparison entity_id",
        )
        _require_unique((metric.metric_id for metric in self.metrics), "comparison metric_id")
        _require_unique(
            (window.window_id for window in self.validation_windows),
            "comparison validation window_id",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "comparison_id": self.comparison_id,
            "full_model": self.full_model.to_dict(),
            "baselines": [baseline.to_dict() for baseline in self.baselines],
            "ablations": [ablation.to_dict() for ablation in self.ablations],
            "metrics": [metric.to_dict() for metric in self.metrics],
            "validation_windows": [
                validation_window.to_dict()
                for validation_window in self.validation_windows
            ],
        }


@dataclass(frozen=True, slots=True)
class ComparisonResultSchema:
    schema_version: str
    comparison_id: str
    full_model_result: ComparisonEntityResultSchema
    baseline_results: tuple[ComparisonEntityResultSchema, ...]
    ablation_results: tuple[ComparisonEntityResultSchema, ...]
    metric_results: tuple[ComparisonMetricResultSchema, ...]
    validation_windows: tuple[ComparisonValidationWindow, ...]

    def __post_init__(self) -> None:
        if self.schema_version != STAGE1_COMPARISON_SCHEMA_VERSION:
            raise ValueError("unsupported comparison schema_version")
        if not MODEL_COMPARISON_ID_PATTERN.fullmatch(self.comparison_id):
            raise ValueError("comparison_id must be stable snake_case starting with a letter")
        if self.full_model_result.role != "full_model":
            raise ValueError("full_model_result must use role='full_model'")
        _require_unique(
            (
                self.full_model_result.entity_id,
                *(baseline.entity_id for baseline in self.baseline_results),
                *(ablation.entity_id for ablation in self.ablation_results),
            ),
            "comparison result entity_id",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "comparison_id": self.comparison_id,
            "full_model_result": self.full_model_result.to_dict(),
            "baseline_results": [baseline.to_dict() for baseline in self.baseline_results],
            "ablation_results": [ablation.to_dict() for ablation in self.ablation_results],
            "metric_results": [metric.to_dict() for metric in self.metric_results],
            "validation_windows": [
                validation_window.to_dict()
                for validation_window in self.validation_windows
            ],
        }


def build_stage1_comparison_input_schema(
    config: object | None = None,
    *,
    baseline_inputs: Iterable[BaselineComparisonInput] | None = None,
    validation_windows: Iterable[ComparisonValidationWindow] | None = None,
) -> ComparisonInputSchema:
    model_config_payload = _model_comparison_config_payload(config)
    baseline_input_objects = tuple(baseline_inputs or _default_baseline_inputs(config))
    return ComparisonInputSchema(
        schema_version=STAGE1_COMPARISON_SCHEMA_VERSION,
        comparison_id=str(
            model_config_payload.get("config_id") or STAGE1_COMPARISON_CONTRACT_ID
        ),
        full_model=_full_model_input_entity(model_config_payload),
        baselines=(
            _model_baseline_input_entity(model_config_payload),
            *(
                _return_baseline_input_entity(baseline_input)
                for baseline_input in baseline_input_objects
            ),
        ),
        ablations=_ablation_input_entities(model_config_payload),
        metrics=_metric_schema_rows(model_config_payload),
        validation_windows=tuple(validation_windows or _configured_validation_windows(config)),
    )


def build_stage1_comparison_result_schema(
    *,
    config: object | None = None,
    model_comparison_results: Iterable[Mapping[str, Any]] = (),
    baseline_comparisons: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]] = (),
    ablation_results: Iterable[Mapping[str, Any]] = (),
    validation_summary: pd.DataFrame | None = None,
) -> ComparisonResultSchema:
    config_payload = _model_comparison_config_payload(config)
    full_model_id = str(
        config_payload.get("full_model_candidate_id")
        or config_payload.get("primary_candidate_id")
        or "all_features"
    )
    baseline_id = str(config_payload.get("baseline_candidate_id") or "no_model_proxy")
    ablation_rows = [
        row
        for row in ablation_results
        if isinstance(row, Mapping) and str(row.get("scenario", "")).strip()
    ]
    ablation_by_scenario = {str(row.get("scenario")): row for row in ablation_rows}
    full_model_row = ablation_by_scenario.get(full_model_id, {})
    model_baseline_row = ablation_by_scenario.get(baseline_id, {})

    baseline_result_rows: list[ComparisonEntityResultSchema] = []
    if model_baseline_row:
        baseline_result_rows.append(
            _ablation_entity_result(model_baseline_row, role="model_baseline")
        )
    baseline_result_rows.extend(
        _return_baseline_entity_result(row)
        for row in _baseline_comparison_rows(baseline_comparisons)
    )

    ablation_result_rows = tuple(
        _ablation_entity_result(row, role=_ablation_result_role(row))
        for row in ablation_rows
        if str(row.get("scenario")) not in {full_model_id, baseline_id}
    )
    metric_rows = tuple(
        ComparisonMetricResultSchema.from_mapping(row)
        for row in model_comparison_results
        if isinstance(row, Mapping)
    )

    return ComparisonResultSchema(
        schema_version=STAGE1_COMPARISON_SCHEMA_VERSION,
        comparison_id=str(config_payload.get("config_id") or STAGE1_COMPARISON_CONTRACT_ID),
        full_model_result=_ablation_entity_result(
            full_model_row,
            role="full_model",
            fallback_entity_id=full_model_id,
            fallback_label="Full model",
        ),
        baseline_results=tuple(baseline_result_rows),
        ablation_results=ablation_result_rows,
        metric_results=metric_rows,
        validation_windows=build_stage1_validation_window_schemas(
            validation_summary,
            config=config,
        ),
    )


def build_stage1_validation_window_schemas(
    validation_summary: pd.DataFrame | None,
    *,
    config: object | None = None,
) -> tuple[ComparisonValidationWindow, ...]:
    if validation_summary is None or validation_summary.empty:
        return tuple(_configured_validation_windows(config))

    target_horizon = _configured_target_horizon(config)
    target_column = _configured_target_column(config)
    gap_periods = int(
        getattr(config, "gap_periods", CANONICAL_DETERMINISTIC_VALIDATION_DEFAULTS.gap_periods)
    )
    embargo_periods = int(
        getattr(
            config,
            "embargo_periods",
            CANONICAL_DETERMINISTIC_VALIDATION_DEFAULTS.embargo_periods,
        )
    )
    gap_periods = max(gap_periods, target_horizon)
    embargo_periods = max(embargo_periods, target_horizon)
    windows: list[ComparisonValidationWindow] = []
    for index, row in validation_summary.reset_index(drop=True).iterrows():
        row_mapping = row.to_dict()
        fold_value = row_mapping.get("fold")
        fold_id = _fold_id(fold_value, index)
        is_oos = bool(row_mapping.get("is_oos", False))
        status = _window_status(row_mapping)
        role: ValidationWindowRole = "oos_holdout" if is_oos else "walk_forward_fold"
        windows.append(
            ComparisonValidationWindow(
                window_id=f"fold_{fold_id}",
                label=f"Fold {fold_id}" + (" OOS" if is_oos else ""),
                role=role,
                target_column=target_column,
                target_horizon=target_horizon,
                gap_periods=gap_periods,
                embargo_periods=embargo_periods,
                start=row_mapping.get("train_start"),
                end=row_mapping.get("test_end") or row_mapping.get("test_start"),
                train_start=row_mapping.get("train_start"),
                train_end=row_mapping.get("train_end"),
                test_start=row_mapping.get("test_start"),
                test_end=row_mapping.get("test_end") or row_mapping.get("test_start"),
                is_oos=is_oos,
                status=status,
                train_observations=_optional_int(row_mapping.get("train_observations")),
                test_observations=_optional_int(row_mapping.get("test_observations")),
                labeled_test_observations=_optional_int(
                    row_mapping.get("labeled_test_observations")
                ),
                prediction_count=_optional_int(row_mapping.get("prediction_count")),
                metadata={
                    key: value
                    for key, value in row_mapping.items()
                    if key
                    in {
                        "fold_type",
                        "model_name",
                        "validation_status",
                        "skip_status",
                        "skip_code",
                        "reason",
                    }
                    and _is_present(value)
                },
            )
        )
    return tuple(windows)


def default_stage1_comparison_input_schema() -> ComparisonInputSchema:
    return build_stage1_comparison_input_schema()


def default_stage1_comparison_result_schema() -> ComparisonResultSchema:
    return build_stage1_comparison_result_schema()


def build_report_comparison_table(
    original_strategy_result: Mapping[str, Any],
    equal_weight_benchmark_result: Mapping[str, Any],
    proxy_ablation_result: Mapping[str, Any],
    *,
    metrics: Iterable[str] | None = None,
    target_column: str = "forward_return_20",
    target_horizon: int = REQUIRED_VALIDATION_HORIZON_DAYS,
) -> pd.DataFrame:
    """Normalize strategy, benchmark, and proxy-ablation results for reports."""
    metric_ids = _report_metric_ids(metrics)
    entities = (
        _report_entity_descriptor(
            original_strategy_result,
            default_entity_id="all_features",
            default_label="Original strategy",
            role="full_model",
            result_type="original_strategy",
            result_source="original_strategy_backtest",
        ),
        _report_entity_descriptor(
            equal_weight_benchmark_result,
            default_entity_id="return_baseline_equal_weight",
            default_label="Equal-weight universe",
            role="return_baseline",
            result_type="equal_weight_benchmark",
            result_source="equal_weight_benchmark_backtest",
            default_baseline_type="equal_weight_universe",
        ),
        _report_entity_descriptor(
            proxy_ablation_result,
            default_entity_id="no_model_proxy",
            default_label="No-model proxy ablation",
            role="model_baseline",
            result_type="proxy_ablation",
            result_source="proxy_ablation_backtest",
            default_ablation_scenario_id="no_model_proxy",
        ),
    )

    rows: list[dict[str, Any]] = []
    for entity in entities:
        source_row = entity["source_row"]
        metric_values = _report_metric_values(source_row)
        for metric in metric_ids:
            value = metric_values.get(metric)
            rows.append(
                {
                    "schema_version": REPORT_COMPARISON_TABLE_SCHEMA_VERSION,
                    "entity_id": entity["entity_id"],
                    "label": entity["label"],
                    "role": entity["role"],
                    "result_type": entity["result_type"],
                    "result_source": entity["result_source"],
                    "baseline_type": entity["baseline_type"],
                    "ablation_scenario_id": entity["ablation_scenario_id"],
                    "metric": metric,
                    "metric_label": DEFAULT_COMPARISON_METRIC_LABELS.get(
                        metric,
                        metric.replace("_", " ").title(),
                    ),
                    "metric_direction": (
                        "lower_is_better"
                        if metric in LOWER_IS_BETTER_COMPARISON_METRICS
                        or metric in {"average_daily_turnover", "total_cost_return"}
                        else "higher_is_better"
                    ),
                    "value": _json_scalar(value),
                    "value_type": _report_value_type(value),
                    "status": entity["status"],
                    "return_basis": _first_present_string(
                        source_row,
                        ("return_basis", "signal_return_basis"),
                    ),
                    "target_column": str(source_row.get("return_column") or target_column),
                    "target_horizon": _optional_int(source_row.get("return_horizon"))
                    or int(target_horizon),
                    "evaluation_start": _optional_iso_date(source_row.get("evaluation_start")),
                    "evaluation_end": _optional_iso_date(source_row.get("evaluation_end")),
                    "evaluation_observations": _optional_int(
                        source_row.get("evaluation_observations")
                        or source_row.get("observations")
                        or source_row.get("validation_fold_count")
                    ),
                    "cost_bps": _optional_float(source_row.get("cost_bps")),
                    "slippage_bps": _optional_float(source_row.get("slippage_bps")),
                }
            )
    return pd.DataFrame(rows, columns=REPORT_COMPARISON_TABLE_COLUMNS)


def _configured_validation_windows(config: object | None) -> tuple[ComparisonValidationWindow, ...]:
    target_horizon = _configured_target_horizon(config)
    target_column = _configured_target_column(config)
    train_periods = int(
        getattr(config, "train_periods", CANONICAL_DETERMINISTIC_VALIDATION_DEFAULTS.train_periods)
    )
    test_periods = int(
        getattr(config, "test_periods", CANONICAL_DETERMINISTIC_VALIDATION_DEFAULTS.test_periods)
    )
    gap_periods = int(
        getattr(config, "gap_periods", CANONICAL_DETERMINISTIC_VALIDATION_DEFAULTS.gap_periods)
    )
    embargo_periods = int(
        getattr(
            config,
            "embargo_periods",
            CANONICAL_DETERMINISTIC_VALIDATION_DEFAULTS.embargo_periods,
        )
    )
    gap_periods = max(gap_periods, target_horizon)
    embargo_periods = max(embargo_periods, target_horizon)
    return (
        ComparisonValidationWindow(
            window_id="configured_walk_forward",
            label="Configured walk-forward train/test window",
            role="configured_walk_forward",
            target_column=target_column,
            target_horizon=target_horizon,
            train_periods=train_periods,
            test_periods=test_periods,
            gap_periods=gap_periods,
            embargo_periods=embargo_periods,
        ),
        ComparisonValidationWindow(
            window_id="configured_oos_holdout",
            label="Configured final fold out-of-sample holdout",
            role="oos_holdout",
            target_column=target_column,
            target_horizon=target_horizon,
            train_periods=train_periods,
            test_periods=test_periods,
            gap_periods=gap_periods,
            embargo_periods=embargo_periods,
            is_oos=True,
        ),
        ComparisonValidationWindow(
            window_id="strategy_evaluation",
            label="Strategy and baseline aligned evaluation samples",
            role="strategy_evaluation",
            target_column=target_column,
            target_horizon=target_horizon,
            gap_periods=gap_periods,
            embargo_periods=embargo_periods,
        ),
    )


def _model_comparison_config_payload(config: object | None) -> dict[str, Any]:
    model_config: object | None
    if isinstance(config, ModelComparisonConfig):
        model_config = config
    elif isinstance(config, Mapping):
        model_config = config.get("model_comparison_config", config)
    else:
        model_config = getattr(config, "model_comparison_config", None)
    if model_config is None:
        model_config = default_model_comparison_config()

    to_dict = getattr(model_config, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        return dict(payload) if isinstance(payload, Mapping) else {}
    if isinstance(model_config, Mapping):
        return dict(model_config)
    return default_model_comparison_config().to_dict()


def _default_baseline_inputs(config: object | None) -> tuple[BaselineComparisonInput, ...]:
    tickers = tuple(str(ticker) for ticker in getattr(config, "tickers", ()) or ())
    return build_stage1_baseline_comparison_inputs(
        None,
        None,
        return_column=_configured_target_column(config),
        return_horizon=_configured_target_horizon(config),
        benchmark_ticker=str(getattr(config, "benchmark_ticker", DEFAULT_BENCHMARK_TICKER)),
        tickers=tickers,
        cost_bps=_optional_float(getattr(config, "cost_bps", None)),
        slippage_bps=_optional_float(getattr(config, "slippage_bps", None)),
    )


def _full_model_input_entity(payload: Mapping[str, Any]) -> ComparisonEntityInputSchema:
    full_model_id = str(
        payload.get("full_model_candidate_id") or payload.get("primary_candidate_id") or "all_features"
    )
    model_config = _canonical_model_config_by_role(payload, "full_model")
    candidate = _candidate_by_id(payload, full_model_id)
    return ComparisonEntityInputSchema(
        entity_id=full_model_id,
        label=str(model_config.get("label") or candidate.get("label") or "Full model"),
        role="full_model",
        comparison_candidate_id=full_model_id,
        model_config_id=str(model_config.get("config_id") or ""),
        ablation_scenario_id=str(candidate.get("ablation_scenario_id") or full_model_id),
        adapters=_tuple_strings(model_config.get("adapters") or candidate.get("adapters") or ()),
        optional_adapters=_tuple_strings(model_config.get("optional_adapters") or ()),
        fallback_adapters=_tuple_strings(model_config.get("fallback_adapters") or ()),
        modalities=_tuple_strings(model_config.get("modalities") or ()),
        feature_families=_tuple_strings(
            model_config.get("feature_families") or candidate.get("feature_families") or ()
        ),
        description=str(model_config.get("description") or candidate.get("description") or ""),
    )


def _model_baseline_input_entity(payload: Mapping[str, Any]) -> ComparisonEntityInputSchema:
    baseline_id = str(payload.get("baseline_candidate_id") or "no_model_proxy")
    model_config = _canonical_model_config_by_role(payload, "baseline")
    candidate = _candidate_by_id(payload, baseline_id)
    return ComparisonEntityInputSchema(
        entity_id=baseline_id,
        label=str(model_config.get("label") or candidate.get("label") or "Model baseline"),
        role="model_baseline",
        comparison_candidate_id=baseline_id,
        model_config_id=str(model_config.get("config_id") or ""),
        ablation_scenario_id=str(candidate.get("ablation_scenario_id") or baseline_id),
        adapters=_tuple_strings(model_config.get("adapters") or candidate.get("adapters") or ()),
        optional_adapters=_tuple_strings(model_config.get("optional_adapters") or ()),
        fallback_adapters=_tuple_strings(model_config.get("fallback_adapters") or ()),
        modalities=_tuple_strings(model_config.get("modalities") or ()),
        feature_families=_tuple_strings(
            model_config.get("feature_families") or candidate.get("feature_families") or ()
        ),
        description=str(model_config.get("description") or candidate.get("description") or ""),
    )


def _return_baseline_input_entity(
    baseline_input: BaselineComparisonInput,
) -> ComparisonEntityInputSchema:
    baseline_name = baseline_input.name
    entity_id = (
        "return_baseline_equal_weight"
        if baseline_name == EQUAL_WEIGHT_BASELINE_NAME
        else f"return_baseline_{_safe_identifier(baseline_name)}"
    )
    return ComparisonEntityInputSchema(
        entity_id=entity_id,
        label=f"{baseline_name} return baseline",
        role="return_baseline",
        baseline_type=baseline_input.baseline_type,
        return_basis=baseline_input.return_basis,
        return_column=baseline_input.return_column,
        return_horizon=baseline_input.return_horizon,
        data_source=baseline_input.data_source,
        evaluation_window=baseline_input.evaluation_window,
        benchmark_ticker=baseline_input.benchmark_ticker,
        universe_tickers=baseline_input.universe_tickers,
        required_for_stage1=baseline_input.required_for_stage1,
        affects_signal_engine=False,
        description=baseline_input.construction_method,
    )


def _ablation_input_entities(payload: Mapping[str, Any]) -> tuple[ComparisonEntityInputSchema, ...]:
    entities: list[ComparisonEntityInputSchema] = []
    full_model_id = str(payload.get("full_model_candidate_id") or payload.get("primary_candidate_id") or "")
    baseline_id = str(payload.get("baseline_candidate_id") or "")
    for candidate in _iter_mapping_rows(payload.get("candidates")):
        candidate_id = str(candidate.get("candidate_id") or "")
        if candidate_id in {full_model_id, baseline_id}:
            continue
        role = "diagnostic" if candidate.get("role") == "diagnostic" else "ablation"
        entities.append(
            ComparisonEntityInputSchema(
                entity_id=candidate_id,
                label=str(candidate.get("label") or candidate_id),
                role=role,
                comparison_candidate_id=candidate_id,
                ablation_scenario_id=str(candidate.get("ablation_scenario_id") or candidate_id),
                adapters=_tuple_strings(candidate.get("adapters") or ()),
                feature_families=_tuple_strings(candidate.get("feature_families") or ()),
                description=str(candidate.get("description") or ""),
            )
        )

    existing_ids = {entity.entity_id for entity in entities}
    for ablation_config in _iter_mapping_rows(payload.get("named_ablation_configs")):
        entity_id = str(ablation_config.get("config_id") or "")
        if not entity_id or entity_id in existing_ids:
            continue
        existing_ids.add(entity_id)
        role = "diagnostic" if ablation_config.get("affects_signal_engine") is False else "ablation"
        entities.append(
            ComparisonEntityInputSchema(
                entity_id=entity_id,
                label=str(ablation_config.get("label") or entity_id),
                role=role,
                ablation_scenario_id=str(ablation_config.get("ablation_scenario_id") or ""),
                adapters=_tuple_strings(ablation_config.get("isolated_adapters") or ()),
                modalities=_tuple_strings(ablation_config.get("isolated_modalities") or ()),
                feature_families=_tuple_strings(ablation_config.get("feature_families") or ()),
                comparison_method=str(ablation_config.get("comparison_method") or ""),
                contribution_kind=str(ablation_config.get("contribution_kind") or ""),
                affects_signal_engine=bool(ablation_config.get("affects_signal_engine", True)),
                description=str(ablation_config.get("description") or ""),
            )
        )
    return tuple(entities)


def _metric_schema_rows(payload: Mapping[str, Any]) -> tuple[ComparisonMetricSchema, ...]:
    metrics = tuple(str(metric) for metric in payload.get("metrics", ()) or ())
    return tuple(
        ComparisonMetricSchema(
            metric_id=metric,
            label=DEFAULT_COMPARISON_METRIC_LABELS.get(metric, metric.replace("_", " ").title()),
            direction=(
                "lower_is_better"
                if metric in LOWER_IS_BETTER_COMPARISON_METRICS
                else "higher_is_better"
            ),
            source="model_comparison_results",
            description="Stage 1 full-model versus baseline comparison metric.",
        )
        for metric in metrics
    )


def _baseline_comparison_rows(
    baseline_comparisons: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if isinstance(baseline_comparisons, Mapping):
        return [
            row
            for row in baseline_comparisons.values()
            if isinstance(row, Mapping)
        ]
    return [row for row in baseline_comparisons if isinstance(row, Mapping)]


def _return_baseline_entity_result(row: Mapping[str, Any]) -> ComparisonEntityResultSchema:
    name = str(row.get("name") or row.get("baseline_name") or "")
    return ComparisonEntityResultSchema(
        entity_id=(
            "return_baseline_equal_weight"
            if name == EQUAL_WEIGHT_BASELINE_NAME
            else f"return_baseline_{_safe_identifier(name)}"
        ),
        label=f"{name} return baseline" if name else "Return baseline",
        role="return_baseline",
        status=_normalize_status(row.get("sample_alignment_status") or row.get("excess_return_status")),
        metrics=_selected_metrics(row),
        sample_count=_optional_int(row.get("evaluation_observations")),
        evaluation_window={
            "start": row.get("evaluation_start"),
            "end": row.get("evaluation_end"),
        },
        baseline_type=str(row.get("baseline_type") or ""),
        result_source="baseline_comparisons",
    )


def _ablation_entity_result(
    row: Mapping[str, Any],
    *,
    role: ComparisonEntityRole,
    fallback_entity_id: str = "",
    fallback_label: str = "",
) -> ComparisonEntityResultSchema:
    scenario = str(row.get("scenario") or fallback_entity_id or "unavailable")
    label = str(row.get("label") or fallback_label or scenario)
    return ComparisonEntityResultSchema(
        entity_id=scenario,
        label=label,
        role=role,
        status=_normalize_status(row.get("validation_status") or row.get("status")),
        metrics=_selected_metrics(row),
        sample_count=_optional_int(
            row.get("validation_fold_count") or row.get("evaluation_observations")
        ),
        ablation_scenario_id=scenario,
        result_source="ablation_summary",
    )


def _ablation_result_role(row: Mapping[str, Any]) -> ComparisonEntityRole:
    return "diagnostic" if row.get("kind") == "cost" else "ablation"


def _selected_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    metric_keys = (
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
        "cost_adjusted_cumulative_return",
        "signal_cost_adjusted_cumulative_return",
        "excess_return",
        "signal_excess_return",
        "turnover",
        "signal_average_daily_turnover",
        "total_cost_return",
        "signal_total_cost_return",
    )
    metrics = {
        key: row.get(key)
        for key in metric_keys
        if key in row and _is_present(row.get(key))
    }
    signal_metrics = row.get("deterministic_signal_evaluation_metrics")
    if isinstance(signal_metrics, Mapping):
        for key in (
            "cost_adjusted_cumulative_return",
            "average_daily_turnover",
            "total_cost_return",
            "excess_return",
            "return_basis",
        ):
            if key in signal_metrics and _is_present(signal_metrics.get(key)):
                metrics[f"signal_{key}"] = signal_metrics.get(key)
    return metrics


def _canonical_model_config_by_role(payload: Mapping[str, Any], role: str) -> dict[str, Any]:
    for row in _iter_mapping_rows(payload.get("canonical_model_configs")):
        if row.get("role") == role:
            return dict(row)
    return {}


def _candidate_by_id(payload: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    for row in _iter_mapping_rows(payload.get("candidates")):
        if str(row.get("candidate_id")) == candidate_id:
            return dict(row)
    return {}


def _iter_mapping_rows(value: object) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        return tuple(row for row in value if isinstance(row, Mapping))
    return ()


def _configured_target_column(config: object | None) -> str:
    return str(
        getattr(
            config,
            "prediction_target_column",
            CANONICAL_DETERMINISTIC_VALIDATION_DEFAULTS.prediction_target_column,
        )
    )


def _configured_target_horizon(config: object | None) -> int:
    target = _configured_target_column(config)
    prefix = "forward_return_"
    if target.startswith(prefix):
        try:
            return max(int(target.removeprefix(prefix)), 1)
        except ValueError:
            pass
    return int(
        getattr(
            config,
            "required_validation_horizon",
            CANONICAL_DETERMINISTIC_VALIDATION_DEFAULTS.required_validation_horizon_days,
        )
    )


def _optional_iso_date(value: object) -> str | None:
    if value is None or value is pd.NA:
        return None
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return None
    return timestamp.date().isoformat()


def _optional_float(value: object) -> float | None:
    if value is None or value is pd.NA:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


def _optional_int(value: object) -> int | None:
    if value is None or value is pd.NA:
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _optional_bool(value: object) -> bool | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, bool):
        return value
    return None


def _report_metric_ids(metrics: Iterable[str] | None) -> tuple[str, ...]:
    raw_metrics = metrics or REPORT_COMPARISON_DEFAULT_METRICS
    normalized: list[str] = []
    seen: set[str] = set()
    for metric in raw_metrics:
        metric_id = str(metric).strip()
        if not metric_id or metric_id in seen:
            continue
        if not MODEL_COMPARISON_ID_PATTERN.fullmatch(metric_id):
            raise ValueError("report comparison metric ids must be stable snake_case")
        normalized.append(metric_id)
        seen.add(metric_id)
    if not normalized:
        raise ValueError("report comparison table requires at least one metric")
    return tuple(normalized)


def _report_entity_descriptor(
    row: Mapping[str, Any],
    *,
    default_entity_id: str,
    default_label: str,
    role: ComparisonEntityRole,
    result_type: str,
    result_source: str,
    default_baseline_type: str = "",
    default_ablation_scenario_id: str = "",
) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise TypeError("report comparison results must be mappings")
    source_row = dict(row)
    entity_id = str(
        source_row.get("entity_id")
        or source_row.get("scenario")
        or default_entity_id
    ).strip()
    label = str(source_row.get("label") or source_row.get("name") or default_label).strip()
    if not entity_id:
        raise ValueError("report comparison entity_id must not be blank")
    if not label:
        raise ValueError("report comparison label must not be blank")
    return {
        "source_row": source_row,
        "entity_id": entity_id,
        "label": label,
        "role": role,
        "result_type": result_type,
        "result_source": str(source_row.get("result_source") or result_source),
        "baseline_type": str(source_row.get("baseline_type") or default_baseline_type),
        "ablation_scenario_id": str(
            source_row.get("ablation_scenario_id")
            or source_row.get("scenario")
            or default_ablation_scenario_id
        ),
        "status": _normalize_status(
            source_row.get("validation_status")
            or source_row.get("sample_alignment_status")
            or source_row.get("excess_return_status")
            or source_row.get("status")
        ),
    }


def _report_metric_values(row: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _selected_metrics(row)
    signal_metrics = row.get("deterministic_signal_evaluation_metrics")
    if isinstance(signal_metrics, Mapping):
        for key, value in signal_metrics.items():
            if _is_present(value):
                metrics.setdefault(str(key), value)
    aliases = {
        "cost_adjusted_cumulative_return": (
            "cost_adjusted_cumulative_return",
            "signal_cost_adjusted_cumulative_return",
        ),
        "excess_return": ("excess_return", "signal_excess_return"),
        "average_daily_turnover": (
            "average_daily_turnover",
            "signal_average_daily_turnover",
            "turnover",
            "signal_average_daily_turnover",
        ),
        "total_cost_return": ("total_cost_return", "signal_total_cost_return"),
        "mean_rank_ic": (
            "mean_rank_ic",
            "rank_ic",
            "validation_mean_information_coefficient",
        ),
        "positive_fold_ratio": (
            "positive_fold_ratio",
            "validation_positive_ic_fold_ratio",
        ),
        "oos_rank_ic": (
            "oos_rank_ic",
            "validation_oos_information_coefficient",
        ),
    }
    for canonical, candidates in aliases.items():
        if canonical in metrics and _is_present(metrics[canonical]):
            continue
        for candidate in candidates:
            if candidate in metrics and _is_present(metrics[candidate]):
                metrics[canonical] = metrics[candidate]
                break
            if candidate in row and _is_present(row[candidate]):
                metrics[canonical] = row[candidate]
                break
    return metrics


def _json_scalar(value: object) -> object:
    if not _is_present(value):
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if pd.isna(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return None if pd.isna(numeric) else numeric


def _report_value_type(value: object) -> str:
    if not _is_present(value):
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "float"
    try:
        float(value)
    except (TypeError, ValueError):
        return "string"
    return "float"


def _first_present_string(row: Mapping[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if _is_present(value):
            return str(value)
    signal_metrics = row.get("deterministic_signal_evaluation_metrics")
    if isinstance(signal_metrics, Mapping):
        value = signal_metrics.get("return_basis")
        if _is_present(value):
            return str(value)
    return ""


def _tuple_strings(values: object) -> tuple[str, ...]:
    if values is None or isinstance(values, (str, bytes)):
        raw_values = () if values is None else (values,)
    else:
        raw_values = tuple(values) if isinstance(values, Iterable) else (values,)
    normalized: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        normalized.append(item)
        seen.add(item)
    return tuple(normalized)


def _safe_identifier(value: str) -> str:
    normalized = "".join(
        char.lower() if char.isalnum() else "_"
        for char in str(value).strip()
    ).strip("_")
    return normalized or "unknown"


def _normalize_status(value: object) -> ComparisonStatus:
    status = str(value or "not_evaluable").strip()
    if status == "pass":
        return "pass"
    if status in {"fail", "failed"}:
        return "fail"
    if status in {"warning", "warn"}:
        return "warning"
    if status == "hard_fail":
        return "hard_fail"
    if status == "insufficient_data":
        return "insufficient_data"
    if status == "skipped":
        return "skipped"
    if status == "configured":
        return "configured"
    return "not_evaluable"


def _window_status(row: Mapping[str, Any]) -> ComparisonStatus:
    for key in ("validation_status", "skip_status", "status"):
        if _is_present(row.get(key)):
            return _normalize_status(row.get(key))
    if _is_present(row.get("skip_code")):
        return "skipped"
    return "pass"


def _fold_id(value: object, fallback: int) -> str:
    if value is None or value is pd.NA:
        return str(fallback)
    try:
        if pd.isna(value):
            return str(fallback)
    except TypeError:
        pass
    return str(value)


def _is_present(value: object) -> bool:
    if value is None:
        return False
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def _require_unique(values: Iterable[str], label: str) -> None:
    ordered = tuple(values)
    duplicates = sorted({value for value in ordered if ordered.count(value) > 1})
    if duplicates:
        raise ValueError(f"duplicate {label}: {duplicates}")
