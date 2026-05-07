from __future__ import annotations

import math
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field, replace
from typing import Literal

from quant_research.validation.ablation import DEFAULT_ABLATION_SCENARIO_IDS, FeatureFamily
from quant_research.validation.horizons import (
    DEFAULT_PURGE_EMBARGO_DAYS,
    DEFAULT_VALIDATION_HORIZONS,
    REQUIRED_VALIDATION_HORIZON_DAYS,
    forward_return_column,
)

ModelComparisonRole = Literal["primary", "baseline", "ablation", "diagnostic"]
ModelComparisonAdapter = Literal[
    "tabular",
    "chronos",
    "granite_ttm",
    "finbert",
    "finma",
    "fingpt",
    "ollama",
    "rules_fallback",
]
CanonicalModelRole = Literal["baseline", "full_model"]
CanonicalModelModality = Literal[
    "price",
    "news_text",
    "sec_filing",
    "time_series",
    "filing_text",
]
ModelComparisonMetric = Literal[
    "rank_ic",
    "positive_fold_ratio",
    "oos_rank_ic",
    "sharpe",
    "max_drawdown",
    "cost_adjusted_cumulative_return",
    "excess_return",
    "turnover",
]
AblationContributionKind = Literal["model", "modality"]
AblationComparisonMethod = Literal[
    "single_channel_refit",
    "full_minus_ablation_delta",
    "adapter_override",
    "reporting_only",
]

MODEL_COMPARISON_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
VALID_MODEL_COMPARISON_ROLES: frozenset[str] = frozenset(
    {"primary", "baseline", "ablation", "diagnostic"}
)
VALID_MODEL_COMPARISON_ADAPTERS: frozenset[str] = frozenset(
    {
        "tabular",
        "chronos",
        "granite_ttm",
        "finbert",
        "finma",
        "fingpt",
        "ollama",
        "rules_fallback",
    }
)
VALID_CANONICAL_MODEL_ROLES: frozenset[str] = frozenset({"baseline", "full_model"})
VALID_CANONICAL_MODEL_MODALITIES: frozenset[str] = frozenset(
    {"price", "news_text", "sec_filing", "time_series", "filing_text"}
)
DETERMINISTIC_SIGNAL_ENGINE_ID = "deterministic_signal_engine"
CANONICAL_STRUCTURED_TEXT_FEATURES: tuple[str, ...] = (
    "sentiment_score",
    "event_tag",
    "risk_flag",
    "confidence",
    "summary_ref",
)
REQUIRED_FULL_MODEL_OPTIONAL_ADAPTERS: tuple[ModelComparisonAdapter, ...] = (
    "chronos",
    "granite_ttm",
    "finbert",
    "finma",
    "fingpt",
    "ollama",
)
VALID_MODEL_COMPARISON_METRICS: frozenset[str] = frozenset(
    {
        "rank_ic",
        "positive_fold_ratio",
        "oos_rank_ic",
        "sharpe",
        "max_drawdown",
        "cost_adjusted_cumulative_return",
        "excess_return",
        "turnover",
    }
)
DEFAULT_MODEL_COMPARISON_METRICS: tuple[ModelComparisonMetric, ...] = (
    "rank_ic",
    "positive_fold_ratio",
    "oos_rank_ic",
    "sharpe",
    "max_drawdown",
    "cost_adjusted_cumulative_return",
    "excess_return",
    "turnover",
)
DEFAULT_STAGE1_OUTPERFORMANCE_MIN_DELTAS: dict[ModelComparisonMetric, float] = {
    "rank_ic": 0.005,
    "positive_fold_ratio": 0.05,
    "oos_rank_ic": 0.005,
    "sharpe": 0.05,
    "max_drawdown": 0.01,
    "cost_adjusted_cumulative_return": 0.005,
    "excess_return": 0.005,
    "turnover": 0.0,
}
CANONICAL_MIN_POSITIVE_FOLD_RATIO = 0.65
REQUIRED_STAGE1_ABLATION_MODEL_ADAPTERS: tuple[ModelComparisonAdapter, ...] = (
    "tabular",
    "chronos",
    "granite_ttm",
    "finbert",
    "finma",
    "fingpt",
    "ollama",
)
REQUIRED_STAGE1_ABLATION_MODALITIES: tuple[CanonicalModelModality, ...] = (
    "price",
    "news_text",
    "sec_filing",
    "time_series",
    "filing_text",
)
PORTFOLIO_RISK_CONSTRAINT_SCHEMA_VERSION = "portfolio_risk_constraints.v1"


@dataclass(frozen=True, slots=True)
class Stage1OutperformanceThresholds:
    minimum_metric_improvements: dict[ModelComparisonMetric, float] = field(
        default_factory=lambda: dict(DEFAULT_STAGE1_OUTPERFORMANCE_MIN_DELTAS)
    )
    require_all_configured_baselines: bool = True
    require_all_configured_windows: bool = True

    def __post_init__(self) -> None:
        if not self.minimum_metric_improvements:
            raise ValueError("minimum_metric_improvements must not be empty")
        unsupported_metrics = sorted(
            set(self.minimum_metric_improvements).difference(VALID_MODEL_COMPARISON_METRICS)
        )
        if unsupported_metrics:
            raise ValueError(
                f"unsupported outperformance threshold metrics: {unsupported_metrics}"
            )
        for metric, threshold in self.minimum_metric_improvements.items():
            if threshold < 0:
                raise ValueError(
                    f"outperformance threshold for {metric} must be non-negative"
                )

    def threshold_for(self, metric: str) -> float:
        return float(self.minimum_metric_improvements.get(metric, 0.0))

    def to_dict(self) -> dict[str, object]:
        return {
            "minimum_metric_improvements": dict(self.minimum_metric_improvements),
            "require_all_configured_baselines": self.require_all_configured_baselines,
            "require_all_configured_windows": self.require_all_configured_windows,
        }


DEFAULT_STAGE1_OUTPERFORMANCE_THRESHOLDS = Stage1OutperformanceThresholds()


@dataclass(frozen=True, slots=True)
class DeterministicValidationDefaults:
    required_validation_horizon_days: int = REQUIRED_VALIDATION_HORIZON_DAYS
    validation_horizons: tuple[int, ...] = DEFAULT_VALIDATION_HORIZONS
    prediction_target_column: str = forward_return_column(REQUIRED_VALIDATION_HORIZON_DAYS)
    gap_periods: int = DEFAULT_PURGE_EMBARGO_DAYS
    embargo_periods: int = DEFAULT_PURGE_EMBARGO_DAYS
    train_periods: int = 90
    test_periods: int = 20
    cost_bps: float = 5.0
    slippage_bps: float = 2.0
    max_daily_turnover: float = 0.35
    min_positive_fold_ratio: float = CANONICAL_MIN_POSITIVE_FOLD_RATIO
    benchmark_ticker: str = "SPY"
    signal_engine: str = DETERMINISTIC_SIGNAL_ENGINE_ID
    model_predictions_are_order_signals: bool = False
    llm_makes_trading_decisions: bool = False
    return_timing: str = "positions_apply_to_t_plus_1_or_later_returns"

    def __post_init__(self) -> None:
        if self.required_validation_horizon_days < 1:
            raise ValueError("required_validation_horizon_days must be positive")
        if not self.validation_horizons:
            raise ValueError("validation_horizons must not be empty")
        if self.required_validation_horizon_days not in self.validation_horizons:
            raise ValueError("validation_horizons must include the required validation horizon")
        if len(set(self.validation_horizons)) != len(self.validation_horizons):
            raise ValueError("validation_horizons must not contain duplicates")
        expected_target = forward_return_column(self.required_validation_horizon_days)
        if self.prediction_target_column != expected_target:
            raise ValueError(
                f"prediction_target_column must be {expected_target!r} for the required horizon"
            )
        if self.gap_periods < self.required_validation_horizon_days:
            raise ValueError("gap_periods must be at least the required validation horizon")
        if self.embargo_periods < self.required_validation_horizon_days:
            raise ValueError("embargo_periods must be at least the required validation horizon")
        if self.train_periods < 1 or self.test_periods < 1:
            raise ValueError("train_periods and test_periods must be positive")
        if self.cost_bps < 0 or self.slippage_bps < 0:
            raise ValueError("cost_bps and slippage_bps must be non-negative")
        if self.max_daily_turnover <= 0:
            raise ValueError("max_daily_turnover must be positive")
        if not 0 <= self.min_positive_fold_ratio <= 1:
            raise ValueError("min_positive_fold_ratio must be between 0 and 1")
        if not self.benchmark_ticker.strip():
            raise ValueError("benchmark_ticker must not be blank")
        if self.signal_engine != DETERMINISTIC_SIGNAL_ENGINE_ID:
            raise ValueError("signal_engine must be deterministic_signal_engine")
        if self.model_predictions_are_order_signals:
            raise ValueError("model predictions must not be treated as order signals")
        if self.llm_makes_trading_decisions:
            raise ValueError("LLM adapters must not make trading decisions")
        if "t_plus_1" not in self.return_timing:
            raise ValueError("return_timing must document t+1 or later return application")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["validation_horizons"] = list(self.validation_horizons)
        return payload


CANONICAL_DETERMINISTIC_VALIDATION_DEFAULTS = DeterministicValidationDefaults()


@dataclass(frozen=True, slots=True)
class TransactionCostSensitivityScenario:
    scenario_id: str
    label: str
    cost_bps: float
    slippage_bps: float
    average_daily_turnover_budget: float
    max_daily_turnover: float | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not MODEL_COMPARISON_ID_PATTERN.fullmatch(self.scenario_id):
            raise ValueError("scenario_id must be stable snake_case starting with a letter")
        if not self.label.strip():
            raise ValueError("label must not be blank")
        _validate_non_negative_float(self.cost_bps, "cost_bps")
        _validate_non_negative_float(self.slippage_bps, "slippage_bps")
        _validate_fraction(
            self.average_daily_turnover_budget,
            "average_daily_turnover_budget",
        )
        if self.max_daily_turnover is not None:
            _validate_long_only_turnover_limit(self.max_daily_turnover, "max_daily_turnover")

    @property
    def total_cost_bps(self) -> float:
        return float(self.cost_bps) + float(self.slippage_bps)

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "label": self.label,
            "cost_bps": float(self.cost_bps),
            "slippage_bps": float(self.slippage_bps),
            "total_cost_bps": self.total_cost_bps,
            "average_daily_turnover_budget": float(self.average_daily_turnover_budget),
            "max_daily_turnover": (
                None if self.max_daily_turnover is None else float(self.max_daily_turnover)
            ),
            "description": self.description,
        }


def _validate_non_negative_float(value: object, name: str) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be non-negative") from exc
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_fraction(value: object, name: str) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be greater than 0 and no more than 1") from exc
    if not math.isfinite(numeric) or not 0 < numeric <= 1:
        raise ValueError(f"{name} must be greater than 0 and no more than 1")


def _validate_unit_interval(value: object, name: str) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be between 0 and 1") from exc
    if not math.isfinite(numeric) or not 0 <= numeric <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def _validate_positive_float(value: object, name: str) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive") from exc
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_long_only_turnover_limit(value: object, name: str) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be between 0 and 2 for a long-only portfolio") from exc
    if not math.isfinite(numeric) or not 0 <= numeric <= 2:
        raise ValueError(f"{name} must be between 0 and 2 for a long-only portfolio")


DEFAULT_TRANSACTION_COST_SENSITIVITY_SCENARIOS: tuple[
    TransactionCostSensitivityScenario, ...
] = (
    TransactionCostSensitivityScenario(
        scenario_id="canonical_costs",
        label="Canonical costs",
        cost_bps=5.0,
        slippage_bps=2.0,
        average_daily_turnover_budget=0.25,
        description=(
            "Stage 1 base case: 5 bps transaction cost, 2 bps slippage, "
            "and 25% average daily turnover budget."
        ),
    ),
    TransactionCostSensitivityScenario(
        scenario_id="no_costs",
        label="No costs",
        cost_bps=0.0,
        slippage_bps=0.0,
        average_daily_turnover_budget=0.25,
        description="Diagnostic upper-bound run with transaction cost and slippage drag disabled.",
    ),
    TransactionCostSensitivityScenario(
        scenario_id="low_costs",
        label="Low costs",
        cost_bps=2.0,
        slippage_bps=1.0,
        average_daily_turnover_budget=0.25,
        description="Sensitivity run for highly liquid execution assumptions.",
    ),
    TransactionCostSensitivityScenario(
        scenario_id="high_costs",
        label="High costs",
        cost_bps=10.0,
        slippage_bps=5.0,
        average_daily_turnover_budget=0.25,
        description="Stress run for wider spreads and less favorable execution assumptions.",
    ),
    TransactionCostSensitivityScenario(
        scenario_id="tight_turnover_budget",
        label="Tight turnover budget",
        cost_bps=5.0,
        slippage_bps=2.0,
        average_daily_turnover_budget=0.15,
        max_daily_turnover=0.50,
        description="Sensitivity run that tightens average turnover while preserving base costs.",
    ),
    TransactionCostSensitivityScenario(
        scenario_id="loose_turnover_budget",
        label="Loose turnover budget",
        cost_bps=5.0,
        slippage_bps=2.0,
        average_daily_turnover_budget=0.25,
        max_daily_turnover=1.00,
        description=(
            "Execution stress run with base costs and looser single-day rebalance capacity; "
            "the strategy gate still evaluates the 25% average daily turnover limit."
        ),
    ),
)
DEFAULT_TRANSACTION_COST_SENSITIVITY_SCENARIO_IDS: tuple[str, ...] = tuple(
    scenario.scenario_id for scenario in DEFAULT_TRANSACTION_COST_SENSITIVITY_SCENARIOS
)


@dataclass(frozen=True, slots=True)
class TransactionCostSensitivityConfig:
    config_id: str = "stage1_cost_turnover_sensitivity"
    baseline_scenario_id: str = "canonical_costs"
    scenarios: tuple[
        TransactionCostSensitivityScenario, ...
    ] = DEFAULT_TRANSACTION_COST_SENSITIVITY_SCENARIOS
    require_canonical_base_case: bool = True

    def __post_init__(self) -> None:
        if not MODEL_COMPARISON_ID_PATTERN.fullmatch(self.config_id):
            raise ValueError("config_id must be stable snake_case starting with a letter")
        if not self.scenarios:
            raise ValueError("transaction cost sensitivity config must contain scenarios")
        by_id: dict[str, TransactionCostSensitivityScenario] = {}
        for scenario in self.scenarios:
            if scenario.scenario_id in by_id:
                raise ValueError(
                    f"duplicate transaction cost sensitivity scenario_id: {scenario.scenario_id}"
                )
            by_id[scenario.scenario_id] = scenario
        if self.baseline_scenario_id not in by_id:
            raise ValueError(
                "baseline_scenario_id must reference a configured sensitivity scenario"
            )
        if self.require_canonical_base_case:
            baseline = by_id[self.baseline_scenario_id]
            defaults = CANONICAL_DETERMINISTIC_VALIDATION_DEFAULTS
            if (
                baseline.cost_bps != defaults.cost_bps
                or baseline.slippage_bps != defaults.slippage_bps
                or baseline.average_daily_turnover_budget != 0.25
            ):
                raise ValueError(
                    "baseline sensitivity scenario must match canonical cost, slippage, "
                    "and 25% turnover defaults"
                )

    def scenario_ids(self) -> tuple[str, ...]:
        return tuple(scenario.scenario_id for scenario in self.scenarios)

    def get(self, scenario_id: str) -> TransactionCostSensitivityScenario:
        for scenario in self.scenarios:
            if scenario.scenario_id == scenario_id:
                return scenario
        raise KeyError(f"unknown transaction cost sensitivity scenario_id: {scenario_id}")

    def to_dict(self) -> dict[str, object]:
        return {
            "config_id": self.config_id,
            "baseline_scenario_id": self.baseline_scenario_id,
            "scenario_ids": list(self.scenario_ids()),
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
            "require_canonical_base_case": self.require_canonical_base_case,
        }


DEFAULT_TRANSACTION_COST_SENSITIVITY_CONFIG = TransactionCostSensitivityConfig()


@dataclass(frozen=True, slots=True)
class RiskConstraintAdjustmentConfig:
    volatility_scale_strength: float = 1.0
    concentration_scale_strength: float = 1.0
    risk_contribution_scale_strength: float = 1.0

    def __post_init__(self) -> None:
        _validate_unit_interval(
            self.volatility_scale_strength,
            "volatility_scale_strength",
        )
        _validate_unit_interval(
            self.concentration_scale_strength,
            "concentration_scale_strength",
        )
        _validate_unit_interval(
            self.risk_contribution_scale_strength,
            "risk_contribution_scale_strength",
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "volatility_scale_strength": float(self.volatility_scale_strength),
            "concentration_scale_strength": float(self.concentration_scale_strength),
            "risk_contribution_scale_strength": float(
                self.risk_contribution_scale_strength
            ),
        }


DEFAULT_RISK_CONSTRAINT_ADJUSTMENTS = RiskConstraintAdjustmentConfig()


@dataclass(frozen=True, slots=True)
class CovarianceAwareRiskConfig:
    enabled: bool = True
    return_column: str = "return_1"
    lookback_periods: int = DEFAULT_PURGE_EMBARGO_DAYS
    min_periods: int = 20
    fallback: str = "diagonal_predicted_volatility"

    def __post_init__(self) -> None:
        if not str(self.return_column).strip():
            raise ValueError("covariance return_column must not be blank")
        if int(self.lookback_periods) < 1:
            raise ValueError("covariance lookback_periods must be positive")
        if int(self.min_periods) < 2:
            raise ValueError("covariance min_periods must be at least 2")
        if int(self.min_periods) > int(self.lookback_periods):
            raise ValueError("covariance min_periods must not exceed lookback_periods")
        if self.fallback != "diagonal_predicted_volatility":
            raise ValueError("covariance fallback must be diagonal_predicted_volatility")

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": bool(self.enabled),
            "return_column": self.return_column,
            "lookback_periods": int(self.lookback_periods),
            "min_periods": int(self.min_periods),
            "fallback": self.fallback,
        }


DEFAULT_COVARIANCE_AWARE_RISK_CONFIG = CovarianceAwareRiskConfig()


@dataclass(frozen=True, slots=True)
class PortfolioRiskConstraintConfig:
    config_id: str = "stage1_long_only_portfolio_risk"
    schema_version: str = PORTFOLIO_RISK_CONSTRAINT_SCHEMA_VERSION
    max_holdings: int = 20
    max_symbol_weight: float = 0.10
    max_sector_weight: float = 0.30
    max_position_risk_contribution: float = 1.0
    portfolio_volatility_limit: float = 0.04
    portfolio_covariance_lookback: int = DEFAULT_PURGE_EMBARGO_DAYS
    covariance_aware_risk: CovarianceAwareRiskConfig = DEFAULT_COVARIANCE_AWARE_RISK_CONFIG
    max_drawdown_stop: float = 0.20
    adjustment: RiskConstraintAdjustmentConfig = DEFAULT_RISK_CONSTRAINT_ADJUSTMENTS
    long_only: bool = True
    v1_exclusions: tuple[str, ...] = ("correlation_cluster_weight",)
    description: str = (
        "Canonical Stage 1 long-only risk schema. Correlation cluster weights are "
        "intentionally excluded from v1 and remain a later enhancement."
    )

    def __post_init__(self) -> None:
        if not MODEL_COMPARISON_ID_PATTERN.fullmatch(self.config_id):
            raise ValueError("config_id must be stable snake_case starting with a letter")
        if self.schema_version != PORTFOLIO_RISK_CONSTRAINT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {PORTFOLIO_RISK_CONSTRAINT_SCHEMA_VERSION!r}"
            )
        if int(self.max_holdings) < 1:
            raise ValueError("max_holdings must be positive")
        _validate_fraction(self.max_symbol_weight, "max_symbol_weight")
        _validate_fraction(self.max_sector_weight, "max_sector_weight")
        _validate_fraction(
            self.max_position_risk_contribution,
            "max_position_risk_contribution",
        )
        _validate_positive_float(
            self.portfolio_volatility_limit,
            "portfolio_volatility_limit",
        )
        if int(self.portfolio_covariance_lookback) < 1:
            raise ValueError("portfolio_covariance_lookback must be positive")
        if int(self.portfolio_covariance_lookback) != int(
            self.covariance_aware_risk.lookback_periods
        ):
            if self.covariance_aware_risk == DEFAULT_COVARIANCE_AWARE_RISK_CONFIG:
                object.__setattr__(
                    self,
                    "covariance_aware_risk",
                    replace(
                        self.covariance_aware_risk,
                        lookback_periods=int(self.portfolio_covariance_lookback),
                        min_periods=min(
                            int(self.covariance_aware_risk.min_periods),
                            int(self.portfolio_covariance_lookback),
                        ),
                    ),
                )
            else:
                raise ValueError(
                    "portfolio_covariance_lookback must match "
                    "covariance_aware_risk.lookback_periods"
                )
        if int(self.covariance_aware_risk.min_periods) > int(
            self.portfolio_covariance_lookback
        ):
            raise ValueError(
                "covariance_aware_risk.min_periods must not exceed portfolio_covariance_lookback"
            )
        _validate_fraction(self.max_drawdown_stop, "max_drawdown_stop")
        if not self.long_only:
            raise ValueError("Stage 1 portfolio risk constraints must be long-only")
        if "correlation_cluster_weight" not in self.v1_exclusions:
            raise ValueError("v1_exclusions must document correlation_cluster_weight")

    def to_backtest_kwargs(self) -> dict[str, object]:
        return {
            "top_n": int(self.max_holdings),
            "max_symbol_weight": float(self.max_symbol_weight),
            "max_sector_weight": float(self.max_sector_weight),
            "portfolio_covariance_lookback": int(self.portfolio_covariance_lookback),
            "covariance_aware_risk_enabled": bool(self.covariance_aware_risk.enabled),
            "covariance_return_column": self.covariance_aware_risk.return_column,
            "covariance_min_periods": int(self.covariance_aware_risk.min_periods),
            "portfolio_volatility_limit": float(self.portfolio_volatility_limit),
            "max_drawdown_stop": float(self.max_drawdown_stop),
            "max_position_risk_contribution": float(
                self.max_position_risk_contribution
            ),
            "volatility_adjustment_strength": float(
                self.adjustment.volatility_scale_strength
            ),
            "concentration_adjustment_strength": float(
                self.adjustment.concentration_scale_strength
            ),
            "risk_contribution_adjustment_strength": float(
                self.adjustment.risk_contribution_scale_strength
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "config_id": self.config_id,
            "schema_version": self.schema_version,
            "max_holdings": int(self.max_holdings),
            "max_symbol_weight": float(self.max_symbol_weight),
            "max_sector_weight": float(self.max_sector_weight),
            "max_position_risk_contribution": float(
                self.max_position_risk_contribution
            ),
            "portfolio_volatility_limit": float(self.portfolio_volatility_limit),
            "portfolio_covariance_lookback": int(self.portfolio_covariance_lookback),
            "covariance_aware_risk": self.covariance_aware_risk.to_dict(),
            "max_drawdown_stop": float(self.max_drawdown_stop),
            "adjustment": self.adjustment.to_dict(),
            "long_only": self.long_only,
            "v1_exclusions": list(self.v1_exclusions),
            "description": self.description,
        }


DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG = PortfolioRiskConstraintConfig()


@dataclass(frozen=True, slots=True)
class NamedAblationConfiguration:
    config_id: str
    label: str
    contribution_kind: AblationContributionKind
    comparison_method: AblationComparisonMethod
    ablation_scenario_id: str
    isolated_adapters: tuple[ModelComparisonAdapter, ...] = ()
    isolated_modalities: tuple[CanonicalModelModality, ...] = ()
    feature_families: tuple[FeatureFamily, ...] = ()
    validation_defaults: DeterministicValidationDefaults = CANONICAL_DETERMINISTIC_VALIDATION_DEFAULTS
    affects_signal_engine: bool = True
    requires_heavy_model: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if not MODEL_COMPARISON_ID_PATTERN.fullmatch(self.config_id):
            raise ValueError("config_id must be stable snake_case starting with a letter")
        if not self.label.strip():
            raise ValueError("label must not be blank")
        if self.contribution_kind not in {"model", "modality"}:
            raise ValueError(
                f"unsupported ablation contribution kind: {self.contribution_kind}"
            )
        if self.comparison_method not in {
            "single_channel_refit",
            "full_minus_ablation_delta",
            "adapter_override",
            "reporting_only",
        }:
            raise ValueError(f"unsupported ablation comparison method: {self.comparison_method}")
        if self.ablation_scenario_id not in DEFAULT_ABLATION_SCENARIO_IDS:
            raise ValueError(
                "ablation_scenario_id must reference a configured ablation scenario"
            )
        self._validate_isolated_adapters()
        self._validate_isolated_modalities()
        self._validate_feature_families()
        if self.validation_defaults != CANONICAL_DETERMINISTIC_VALIDATION_DEFAULTS:
            raise ValueError("named ablation configurations must share Stage 1 validation semantics")
        if self.requires_heavy_model:
            raise ValueError("Stage 1 named ablations must keep heavy model adapters optional")
        if self.contribution_kind == "model" and not self.isolated_adapters:
            raise ValueError("model ablation configurations require isolated_adapters")
        if self.contribution_kind == "modality" and not self.isolated_modalities:
            raise ValueError("modality ablation configurations require isolated_modalities")
        if self.affects_signal_engine and not self.feature_families:
            raise ValueError("signal-affecting ablations require feature_families")

    def to_dict(self) -> dict[str, object]:
        return {
            "config_id": self.config_id,
            "label": self.label,
            "contribution_kind": self.contribution_kind,
            "comparison_method": self.comparison_method,
            "ablation_scenario_id": self.ablation_scenario_id,
            "isolated_adapters": list(self.isolated_adapters),
            "isolated_modalities": list(self.isolated_modalities),
            "feature_families": list(self.feature_families),
            "validation_defaults": self.validation_defaults.to_dict(),
            "affects_signal_engine": self.affects_signal_engine,
            "requires_heavy_model": self.requires_heavy_model,
            "description": self.description,
        }

    def _validate_isolated_adapters(self) -> None:
        unsupported = sorted(
            set(self.isolated_adapters).difference(VALID_MODEL_COMPARISON_ADAPTERS)
        )
        if unsupported:
            raise ValueError(f"unsupported isolated_adapters: {unsupported}")
        if len(set(self.isolated_adapters)) != len(self.isolated_adapters):
            raise ValueError("isolated_adapters must not contain duplicates")

    def _validate_isolated_modalities(self) -> None:
        unsupported = sorted(
            set(self.isolated_modalities).difference(VALID_CANONICAL_MODEL_MODALITIES)
        )
        if unsupported:
            raise ValueError(f"unsupported isolated_modalities: {unsupported}")
        if len(set(self.isolated_modalities)) != len(self.isolated_modalities):
            raise ValueError("isolated_modalities must not contain duplicates")

    def _validate_feature_families(self) -> None:
        unsupported = sorted(
            set(self.feature_families).difference(
                {"price", "text", "sec", "chronos", "granite_ttm"}
            )
        )
        if unsupported:
            raise ValueError(f"unsupported feature families: {unsupported}")
        if len(set(self.feature_families)) != len(self.feature_families):
            raise ValueError("feature_families must not contain duplicates")


STAGE1_NAMED_ABLATION_CONFIGS: tuple[NamedAblationConfiguration, ...] = (
    NamedAblationConfiguration(
        config_id="price_modality",
        label="Price modality",
        contribution_kind="modality",
        comparison_method="single_channel_refit",
        ablation_scenario_id="price_only",
        isolated_adapters=("tabular",),
        isolated_modalities=("price",),
        feature_families=("price",),
        description="Isolates OHLCV-derived tabular price features under the shared Stage 1 walk-forward contract.",
    ),
    NamedAblationConfiguration(
        config_id="news_text_modality",
        label="News text modality",
        contribution_kind="modality",
        comparison_method="single_channel_refit",
        ablation_scenario_id="text_only",
        isolated_adapters=("finbert", "rules_fallback"),
        isolated_modalities=("news_text",),
        feature_families=("text",),
        description="Isolates structured news sentiment, event, confidence, and risk features.",
    ),
    NamedAblationConfiguration(
        config_id="sec_filing_modality",
        label="SEC filing modality",
        contribution_kind="modality",
        comparison_method="single_channel_refit",
        ablation_scenario_id="sec_only",
        isolated_adapters=("rules_fallback",),
        isolated_modalities=("sec_filing",),
        feature_families=("sec",),
        description="Isolates SEC filing counts, filing metadata, and company facts features.",
    ),
    NamedAblationConfiguration(
        config_id="time_series_modality",
        label="Time-series model modality",
        contribution_kind="modality",
        comparison_method="full_minus_ablation_delta",
        ablation_scenario_id="tabular_without_ts_proxies",
        isolated_adapters=("chronos", "granite_ttm"),
        isolated_modalities=("time_series",),
        feature_families=("chronos", "granite_ttm"),
        description="Measures the incremental value of optional time-series proxy feature families.",
    ),
    NamedAblationConfiguration(
        config_id="filing_text_modality",
        label="Filing text modality",
        contribution_kind="modality",
        comparison_method="single_channel_refit",
        ablation_scenario_id="sec_only",
        isolated_adapters=("finma", "fingpt", "rules_fallback"),
        isolated_modalities=("filing_text",),
        feature_families=("sec",),
        description="Isolates structured event extraction from filing text while keeping outputs schema-bound.",
    ),
    NamedAblationConfiguration(
        config_id="tabular_model",
        label="Tabular model",
        contribution_kind="model",
        comparison_method="full_minus_ablation_delta",
        ablation_scenario_id="tabular_without_ts_proxies",
        isolated_adapters=("tabular",),
        isolated_modalities=("price", "news_text", "sec_filing", "filing_text"),
        feature_families=("price", "text", "sec"),
        description="Defines the non-heavy tabular baseline after time-series proxy features are removed.",
    ),
    NamedAblationConfiguration(
        config_id="chronos_model",
        label="Chronos-2 model",
        contribution_kind="model",
        comparison_method="full_minus_ablation_delta",
        ablation_scenario_id="no_chronos_features",
        isolated_adapters=("chronos",),
        isolated_modalities=("time_series",),
        feature_families=("chronos",),
        description="Measures Chronos feature contribution by comparing full features against no_chronos_features.",
    ),
    NamedAblationConfiguration(
        config_id="granite_ttm_model",
        label="Granite TTM model",
        contribution_kind="model",
        comparison_method="full_minus_ablation_delta",
        ablation_scenario_id="no_granite_features",
        isolated_adapters=("granite_ttm",),
        isolated_modalities=("time_series",),
        feature_families=("granite_ttm",),
        description="Measures Granite TTM feature contribution by comparing full features against no_granite_features.",
    ),
    NamedAblationConfiguration(
        config_id="finbert_model",
        label="FinBERT model",
        contribution_kind="model",
        comparison_method="adapter_override",
        ablation_scenario_id="text_only",
        isolated_adapters=("finbert", "rules_fallback"),
        isolated_modalities=("news_text",),
        feature_families=("text",),
        description="Defines the FinBERT news-text contribution with deterministic keyword fallback.",
    ),
    NamedAblationConfiguration(
        config_id="finma_model",
        label="FinMA model",
        contribution_kind="model",
        comparison_method="adapter_override",
        ablation_scenario_id="sec_only",
        isolated_adapters=("finma", "rules_fallback"),
        isolated_modalities=("filing_text",),
        feature_families=("sec",),
        description="Defines the FinMA filing event extraction contribution with rules fallback.",
    ),
    NamedAblationConfiguration(
        config_id="fingpt_model",
        label="FinGPT model",
        contribution_kind="model",
        comparison_method="adapter_override",
        ablation_scenario_id="sec_only",
        isolated_adapters=("fingpt", "rules_fallback"),
        isolated_modalities=("filing_text",),
        feature_families=("sec",),
        description="Defines the FinGPT filing event extraction contribution with rules fallback.",
    ),
    NamedAblationConfiguration(
        config_id="ollama_model",
        label="Ollama local model",
        contribution_kind="model",
        comparison_method="reporting_only",
        ablation_scenario_id="all_features",
        isolated_adapters=("ollama", "rules_fallback"),
        isolated_modalities=("filing_text",),
        affects_signal_engine=False,
        description="Records the reporting/explanation adapter separately; it must not affect deterministic signals.",
    ),
)
STAGE1_NAMED_ABLATION_CONFIG_IDS: tuple[str, ...] = tuple(
    config.config_id for config in STAGE1_NAMED_ABLATION_CONFIGS
)


@dataclass(frozen=True, slots=True)
class CanonicalModelConfiguration:
    config_id: str
    label: str
    role: CanonicalModelRole
    comparison_candidate_id: str
    modalities: tuple[CanonicalModelModality, ...]
    feature_families: tuple[FeatureFamily, ...]
    adapters: tuple[ModelComparisonAdapter, ...]
    optional_adapters: tuple[ModelComparisonAdapter, ...] = ()
    fallback_adapters: tuple[ModelComparisonAdapter, ...] = ("rules_fallback",)
    structured_text_features: tuple[str, ...] = CANONICAL_STRUCTURED_TEXT_FEATURES
    validation_defaults: DeterministicValidationDefaults = CANONICAL_DETERMINISTIC_VALIDATION_DEFAULTS
    requires_heavy_model: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if not MODEL_COMPARISON_ID_PATTERN.fullmatch(self.config_id):
            raise ValueError("config_id must be stable snake_case starting with a letter")
        if not self.label.strip():
            raise ValueError("label must not be blank")
        if self.role not in VALID_CANONICAL_MODEL_ROLES:
            raise ValueError(f"unsupported canonical model role: {self.role}")
        if not MODEL_COMPARISON_ID_PATTERN.fullmatch(self.comparison_candidate_id):
            raise ValueError(
                "comparison_candidate_id must be stable snake_case starting with a letter"
            )
        self._validate_modalities()
        self._validate_feature_families()
        self._validate_adapters()
        if self.role == "full_model":
            missing = sorted(set(REQUIRED_FULL_MODEL_OPTIONAL_ADAPTERS).difference(self.optional_adapters))
            if missing:
                raise ValueError(f"full_model config missing optional adapters: {missing}")
        if self.requires_heavy_model:
            raise ValueError("canonical Stage 1 configurations must use deterministic fallbacks by default")

    def to_dict(self) -> dict[str, object]:
        return {
            "config_id": self.config_id,
            "label": self.label,
            "role": self.role,
            "comparison_candidate_id": self.comparison_candidate_id,
            "modalities": list(self.modalities),
            "feature_families": list(self.feature_families),
            "adapters": list(self.adapters),
            "optional_adapters": list(self.optional_adapters),
            "fallback_adapters": list(self.fallback_adapters),
            "structured_text_features": list(self.structured_text_features),
            "validation_defaults": self.validation_defaults.to_dict(),
            "requires_heavy_model": self.requires_heavy_model,
            "description": self.description,
        }

    def _validate_modalities(self) -> None:
        if not self.modalities:
            raise ValueError("modalities must contain at least one modality")
        unsupported = sorted(set(self.modalities).difference(VALID_CANONICAL_MODEL_MODALITIES))
        if unsupported:
            raise ValueError(f"unsupported canonical model modalities: {unsupported}")
        if len(set(self.modalities)) != len(self.modalities):
            raise ValueError("modalities must not contain duplicates")

    def _validate_feature_families(self) -> None:
        if not self.feature_families:
            raise ValueError("feature_families must contain at least one family")
        unsupported = sorted(set(self.feature_families).difference({"price", "text", "sec", "chronos", "granite_ttm"}))
        if unsupported:
            raise ValueError(f"unsupported feature families: {unsupported}")
        if len(set(self.feature_families)) != len(self.feature_families):
            raise ValueError("feature_families must not contain duplicates")
        if {"text", "sec"}.intersection(self.feature_families) and not self.structured_text_features:
            raise ValueError("text and SEC configurations require structured text features")

    def _validate_adapters(self) -> None:
        if not self.adapters:
            raise ValueError("adapters must contain at least one adapter id")
        for field_name, adapters in (
            ("adapters", self.adapters),
            ("optional_adapters", self.optional_adapters),
            ("fallback_adapters", self.fallback_adapters),
        ):
            unsupported = sorted(set(adapters).difference(VALID_MODEL_COMPARISON_ADAPTERS))
            if unsupported:
                raise ValueError(f"unsupported {field_name}: {unsupported}")
            if len(set(adapters)) != len(adapters):
                raise ValueError(f"{field_name} must not contain duplicates")
        missing_optional = sorted(set(self.optional_adapters).difference(self.adapters))
        if missing_optional:
            raise ValueError(f"optional_adapters must be listed in adapters: {missing_optional}")
        missing_fallbacks = sorted(set(self.fallback_adapters).difference(self.adapters))
        if missing_fallbacks:
            raise ValueError(f"fallback_adapters must be listed in adapters: {missing_fallbacks}")


CANONICAL_BASELINE_MODEL_CONFIG = CanonicalModelConfiguration(
    config_id="baseline_no_model_proxy",
    label="Canonical no-model-proxy baseline",
    role="baseline",
    comparison_candidate_id="no_model_proxy",
    modalities=("price", "news_text", "sec_filing", "filing_text"),
    feature_families=("price", "text", "sec"),
    adapters=("tabular", "finbert", "finma", "fingpt", "rules_fallback"),
    optional_adapters=("finbert", "finma", "fingpt"),
    description=(
        "Stage 1 baseline keeps price, news/text, and SEC structured features while "
        "removing optional time-series proxy features. Heavy NLP adapters remain optional "
        "and fall back to deterministic rules."
    ),
)
CANONICAL_FULL_MODEL_CONFIG = CanonicalModelConfiguration(
    config_id="full_multimodal_model",
    label="Canonical full multimodal model",
    role="full_model",
    comparison_candidate_id="all_features",
    modalities=("price", "news_text", "sec_filing", "filing_text", "time_series"),
    feature_families=("price", "text", "sec", "chronos", "granite_ttm"),
    adapters=(
        "tabular",
        "chronos",
        "granite_ttm",
        "finbert",
        "finma",
        "fingpt",
        "ollama",
        "rules_fallback",
    ),
    optional_adapters=REQUIRED_FULL_MODEL_OPTIONAL_ADAPTERS,
    description=(
        "Stage 1 full-model candidate covers price, text, SEC, Chronos-2, Granite TTM, "
        "FinBERT, FinMA, FinGPT, and Ollama feature adapters while keeping deterministic "
        "fallback execution as the default."
    ),
)
CANONICAL_MODEL_CONFIGS: tuple[CanonicalModelConfiguration, ...] = (
    CANONICAL_BASELINE_MODEL_CONFIG,
    CANONICAL_FULL_MODEL_CONFIG,
)
CANONICAL_STAGE1_MODEL_CONFIGS = CANONICAL_MODEL_CONFIGS
CANONICAL_MODEL_CONFIG_IDS: tuple[str, ...] = tuple(
    model_config.config_id for model_config in CANONICAL_MODEL_CONFIGS
)


@dataclass(frozen=True, slots=True)
class ModelComparisonCandidateConfig:
    candidate_id: str
    label: str
    role: ModelComparisonRole
    adapters: tuple[ModelComparisonAdapter, ...]
    feature_families: tuple[FeatureFamily, ...] = ()
    ablation_scenario_id: str | None = None
    requires_heavy_model: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if not MODEL_COMPARISON_ID_PATTERN.fullmatch(self.candidate_id):
            raise ValueError("candidate_id must be stable snake_case starting with a letter")
        if not self.label.strip():
            raise ValueError("label must not be blank")
        if self.role not in VALID_MODEL_COMPARISON_ROLES:
            raise ValueError(f"unsupported model comparison role: {self.role}")
        if not self.adapters:
            raise ValueError("adapters must contain at least one adapter id")
        unsupported_adapters = sorted(set(self.adapters).difference(VALID_MODEL_COMPARISON_ADAPTERS))
        if unsupported_adapters:
            raise ValueError(f"unsupported model comparison adapters: {unsupported_adapters}")
        if len(set(self.adapters)) != len(self.adapters):
            raise ValueError("adapters must not contain duplicate adapter ids")
        if len(set(self.feature_families)) != len(self.feature_families):
            raise ValueError("feature_families must not contain duplicate families")
        if self.role == "ablation" and not self.ablation_scenario_id:
            raise ValueError("ablation model comparisons require ablation_scenario_id")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["adapters"] = list(self.adapters)
        payload["feature_families"] = list(self.feature_families)
        return payload


class ModelComparisonRegistry:
    def __init__(self, candidates: Iterable[ModelComparisonCandidateConfig]) -> None:
        ordered = tuple(candidates)
        if not ordered:
            raise ValueError("model comparison registry must contain at least one candidate")

        by_id: dict[str, ModelComparisonCandidateConfig] = {}
        for candidate in ordered:
            if candidate.candidate_id in by_id:
                raise ValueError(f"duplicate model comparison candidate_id: {candidate.candidate_id}")
            by_id[candidate.candidate_id] = candidate

        self._candidates = ordered
        self._by_id = by_id

    def __iter__(self) -> Iterator[ModelComparisonCandidateConfig]:
        return iter(self._candidates)

    def __len__(self) -> int:
        return len(self._candidates)

    def ids(self) -> tuple[str, ...]:
        return tuple(candidate.candidate_id for candidate in self._candidates)

    def get(self, candidate_id: str) -> ModelComparisonCandidateConfig:
        try:
            return self._by_id[candidate_id]
        except KeyError as exc:
            raise KeyError(f"unknown model comparison candidate_id: {candidate_id}") from exc

    def by_role(self, role: ModelComparisonRole) -> tuple[ModelComparisonCandidateConfig, ...]:
        if role not in VALID_MODEL_COMPARISON_ROLES:
            raise ValueError(f"unsupported model comparison role: {role}")
        return tuple(candidate for candidate in self._candidates if candidate.role == role)

    def to_dicts(self) -> list[dict[str, object]]:
        return [candidate.to_dict() for candidate in self._candidates]


@dataclass(frozen=True, slots=True)
class ModelComparisonConfig:
    config_id: str = "stage1_validity_gate"
    label: str = "Stage 1 validity gate model comparison"
    required_horizon_days: int = REQUIRED_VALIDATION_HORIZON_DAYS
    primary_candidate_id: str = "all_features"
    baseline_candidate_id: str = "no_model_proxy"
    full_model_candidate_id: str = "all_features"
    candidates: tuple[ModelComparisonCandidateConfig, ...] = ()
    metrics: tuple[ModelComparisonMetric, ...] = DEFAULT_MODEL_COMPARISON_METRICS
    canonical_model_configs: tuple[CanonicalModelConfiguration, ...] = CANONICAL_MODEL_CONFIGS
    named_ablation_configs: tuple[NamedAblationConfiguration, ...] = STAGE1_NAMED_ABLATION_CONFIGS
    validation_defaults: DeterministicValidationDefaults = CANONICAL_DETERMINISTIC_VALIDATION_DEFAULTS
    outperformance_thresholds: Stage1OutperformanceThresholds = DEFAULT_STAGE1_OUTPERFORMANCE_THRESHOLDS
    require_cost_adjusted_metrics: bool = True
    allow_heavy_model_candidates: bool = False

    def __post_init__(self) -> None:
        if not MODEL_COMPARISON_ID_PATTERN.fullmatch(self.config_id):
            raise ValueError("config_id must be stable snake_case starting with a letter")
        if not self.label.strip():
            raise ValueError("label must not be blank")
        if self.required_horizon_days < 1:
            raise ValueError("required_horizon_days must be positive")
        if not self.candidates:
            raise ValueError("model comparison config must contain candidates")
        if not self.metrics:
            raise ValueError("model comparison config must contain metrics")

        unsupported_metrics = sorted(set(self.metrics).difference(VALID_MODEL_COMPARISON_METRICS))
        if unsupported_metrics:
            raise ValueError(f"unsupported model comparison metrics: {unsupported_metrics}")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("metrics must not contain duplicate entries")

        if (
            not self.allow_heavy_model_candidates
            and any(candidate.requires_heavy_model for candidate in self.candidates)
        ):
            raise ValueError(
                "heavy model comparison candidates require allow_heavy_model_candidates=True"
            )
        registry = ModelComparisonRegistry(self.candidates)
        try:
            registry.get(self.primary_candidate_id)
        except KeyError as exc:
            raise ValueError(
                "primary_candidate_id must reference a configured model comparison candidate"
            ) from exc
        configured_candidate_ids = set(registry.ids())
        for field_name, candidate_id in (
            ("baseline_candidate_id", self.baseline_candidate_id),
            ("full_model_candidate_id", self.full_model_candidate_id),
        ):
            if not MODEL_COMPARISON_ID_PATTERN.fullmatch(candidate_id):
                raise ValueError(
                    f"{field_name} must be stable snake_case starting with a letter"
                )
            if candidate_id not in configured_candidate_ids:
                continue
        self._validate_canonical_model_configs()
        self._validate_named_ablation_configs()
        if self.validation_defaults.required_validation_horizon_days != self.required_horizon_days:
            raise ValueError(
                "validation_defaults must use the configured required_horizon_days"
            )

    @property
    def registry(self) -> ModelComparisonRegistry:
        return ModelComparisonRegistry(self.candidates)

    def ablation_scenario_ids(self) -> tuple[str, ...]:
        return tuple(
            candidate.ablation_scenario_id
            for candidate in self.candidates
            if candidate.ablation_scenario_id is not None
        )

    def required_candidate_ids(self) -> tuple[str, ...]:
        return tuple(candidate.candidate_id for candidate in self.candidates)

    def to_dict(self) -> dict[str, object]:
        return {
            "config_id": self.config_id,
            "label": self.label,
            "required_horizon_days": self.required_horizon_days,
            "primary_candidate_id": self.primary_candidate_id,
            "baseline_candidate_id": self.baseline_candidate_id,
            "full_model_candidate_id": self.full_model_candidate_id,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "metrics": list(self.metrics),
            "canonical_model_configs": [
                model_config.to_dict() for model_config in self.canonical_model_configs
            ],
            "named_ablation_configs": [
                ablation_config.to_dict() for ablation_config in self.named_ablation_configs
            ],
            "validation_defaults": self.validation_defaults.to_dict(),
            "outperformance_thresholds": self.outperformance_thresholds.to_dict(),
            "require_cost_adjusted_metrics": self.require_cost_adjusted_metrics,
            "allow_heavy_model_candidates": self.allow_heavy_model_candidates,
            "ablation_scenario_ids": list(self.ablation_scenario_ids()),
            "required_candidate_ids": list(self.required_candidate_ids()),
        }

    def _validate_canonical_model_configs(self) -> None:
        configs = tuple(self.canonical_model_configs)
        if not configs:
            raise ValueError("canonical_model_configs must contain baseline and full_model configs")
        by_role: dict[str, CanonicalModelConfiguration] = {}
        by_id: dict[str, CanonicalModelConfiguration] = {}
        for model_config in configs:
            if model_config.config_id in by_id:
                raise ValueError(f"duplicate canonical model config_id: {model_config.config_id}")
            by_id[model_config.config_id] = model_config
            if model_config.role in by_role:
                raise ValueError(f"duplicate canonical model role: {model_config.role}")
            by_role[model_config.role] = model_config
        missing_roles = sorted(VALID_CANONICAL_MODEL_ROLES.difference(by_role))
        if missing_roles:
            raise ValueError(f"missing canonical model role(s): {missing_roles}")
        if by_role["baseline"].comparison_candidate_id != self.baseline_candidate_id:
            raise ValueError(
                "baseline canonical model config must reference baseline_candidate_id"
            )
        if by_role["full_model"].comparison_candidate_id != self.full_model_candidate_id:
            raise ValueError(
                "full_model canonical model config must reference full_model_candidate_id"
            )

    def _validate_named_ablation_configs(self) -> None:
        configs = tuple(self.named_ablation_configs)
        if not configs:
            raise ValueError("named_ablation_configs must not be empty")
        by_id: dict[str, NamedAblationConfiguration] = {}
        for ablation_config in configs:
            if ablation_config.config_id in by_id:
                raise ValueError(
                    f"duplicate named ablation config_id: {ablation_config.config_id}"
                )
            by_id[ablation_config.config_id] = ablation_config
            if ablation_config.validation_defaults != self.validation_defaults:
                raise ValueError(
                    "named ablation configurations must share model comparison validation semantics"
                )

        covered_adapters = {
            adapter
            for ablation_config in configs
            if ablation_config.contribution_kind == "model"
            for adapter in ablation_config.isolated_adapters
            if adapter != "rules_fallback"
        }
        missing_adapters = sorted(
            set(REQUIRED_STAGE1_ABLATION_MODEL_ADAPTERS).difference(covered_adapters)
        )
        if missing_adapters:
            raise ValueError(
                f"missing Stage 1 named model ablation adapter coverage: {missing_adapters}"
            )

        covered_modalities = {
            modality
            for ablation_config in configs
            if ablation_config.contribution_kind == "modality"
            for modality in ablation_config.isolated_modalities
        }
        missing_modalities = sorted(
            set(REQUIRED_STAGE1_ABLATION_MODALITIES).difference(covered_modalities)
        )
        if missing_modalities:
            raise ValueError(
                f"missing Stage 1 named modality ablation coverage: {missing_modalities}"
            )


DEFAULT_MODEL_COMPARISON_CANDIDATES: tuple[ModelComparisonCandidateConfig, ...] = (
    ModelComparisonCandidateConfig(
        candidate_id="all_features",
        label="All structured features",
        role="primary",
        adapters=(
            "tabular",
            "chronos",
            "granite_ttm",
            "finbert",
            "finma",
            "fingpt",
            "ollama",
        ),
        feature_families=("price", "text", "sec", "chronos", "granite_ttm"),
        ablation_scenario_id="all_features",
        description=(
            "Primary deterministic signal candidate using structured price, text, SEC, "
            "and optional model proxy features."
        ),
    ),
    ModelComparisonCandidateConfig(
        candidate_id="no_model_proxy",
        label="No model proxy features",
        role="baseline",
        adapters=("tabular", "finbert", "finma", "fingpt", "rules_fallback"),
        feature_families=("price", "text", "sec"),
        ablation_scenario_id="no_model_proxy",
        description="Ablation refit that removes Chronos and Granite proxy feature families.",
    ),
    ModelComparisonCandidateConfig(
        candidate_id="price_only",
        label="Price only",
        role="ablation",
        adapters=("tabular", "chronos", "granite_ttm"),
        feature_families=("price", "chronos", "granite_ttm"),
        ablation_scenario_id="price_only",
        description="Ablation refit using price-derived feature families only.",
    ),
    ModelComparisonCandidateConfig(
        candidate_id="text_only",
        label="Text only",
        role="ablation",
        adapters=("finbert", "finma", "fingpt", "rules_fallback"),
        feature_families=("text",),
        ablation_scenario_id="text_only",
        description="Ablation refit using structured news and filing text features only.",
    ),
    ModelComparisonCandidateConfig(
        candidate_id="sec_only",
        label="SEC only",
        role="ablation",
        adapters=("finma", "fingpt", "rules_fallback"),
        feature_families=("sec",),
        ablation_scenario_id="sec_only",
        description="Ablation refit using structured SEC filing and facts features only.",
    ),
    ModelComparisonCandidateConfig(
        candidate_id="no_costs",
        label="No costs",
        role="diagnostic",
        adapters=("tabular", "rules_fallback"),
        feature_families=("price", "text", "sec", "chronos", "granite_ttm"),
        ablation_scenario_id="no_costs",
        description="Diagnostic comparison with transaction costs, slippage, and turnover costs disabled.",
    ),
)
DEFAULT_MODEL_COMPARISON_CANDIDATE_IDS: tuple[str, ...] = tuple(
    candidate.candidate_id for candidate in DEFAULT_MODEL_COMPARISON_CANDIDATES
)


def default_model_comparison_registry() -> ModelComparisonRegistry:
    return ModelComparisonRegistry(DEFAULT_MODEL_COMPARISON_CANDIDATES)


def default_model_comparison_config() -> ModelComparisonConfig:
    return ModelComparisonConfig(candidates=DEFAULT_MODEL_COMPARISON_CANDIDATES)


def default_deterministic_validation_defaults() -> DeterministicValidationDefaults:
    return CANONICAL_DETERMINISTIC_VALIDATION_DEFAULTS


def default_canonical_model_configs() -> tuple[CanonicalModelConfiguration, ...]:
    return CANONICAL_MODEL_CONFIGS


def default_stage1_model_configs() -> tuple[CanonicalModelConfiguration, ...]:
    return default_canonical_model_configs()


def default_named_ablation_configs() -> tuple[NamedAblationConfiguration, ...]:
    return STAGE1_NAMED_ABLATION_CONFIGS


def default_transaction_cost_sensitivity_config() -> TransactionCostSensitivityConfig:
    return DEFAULT_TRANSACTION_COST_SENSITIVITY_CONFIG
