from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from quant_research.validation.config import DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG
from quant_research.validation.gate import (
    StrategyCandidateGatePolicy,
    ValidationGateReport,
    ValidationGateThresholds,
    build_validity_gate_report,
)
from quant_research.validation.walk_forward import (
    CANONICAL_WALK_FORWARD_EMBARGO_PERIODS,
    CANONICAL_WALK_FORWARD_TARGET_COLUMN,
)

DETERMINISTIC_GATE_INTERFACE_SCHEMA_VERSION = "deterministic_gate_interface.v1"
DETERMINISTIC_GATE_INTERFACE_ID = "stage1_provider_free_deterministic_gate"


@dataclass(frozen=True, slots=True)
class DeterministicGateCostInputs:
    cost_bps: float = 5.0
    slippage_bps: float = 2.0
    average_daily_turnover_budget: float = 0.25
    turnover_sensitivity_results: Sequence[Mapping[str, Any]] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if float(self.cost_bps) < 0:
            raise ValueError("cost_bps must be non-negative")
        if float(self.slippage_bps) < 0:
            raise ValueError("slippage_bps must be non-negative")
        if not 0 < float(self.average_daily_turnover_budget) <= 2:
            raise ValueError("average_daily_turnover_budget must be in (0, 2]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cost_bps": float(self.cost_bps),
            "slippage_bps": float(self.slippage_bps),
            "total_cost_bps": float(self.cost_bps) + float(self.slippage_bps),
            "average_daily_turnover_budget": float(self.average_daily_turnover_budget),
            "turnover_sensitivity_results": [
                dict(row) for row in self.turnover_sensitivity_results
            ],
        }


@dataclass(frozen=True, slots=True)
class DeterministicGateBacktestInputs:
    equity_curve: pd.DataFrame
    strategy_metrics: object

    def __post_init__(self) -> None:
        _require_frame_columns(
            self.equity_curve,
            ("date", "cost_adjusted_return", "turnover"),
            "equity_curve",
        )
        for attribute in ("cagr", "sharpe", "max_drawdown", "turnover"):
            if not hasattr(self.strategy_metrics, attribute):
                raise ValueError(f"strategy_metrics must expose {attribute!r}")


@dataclass(frozen=True, slots=True)
class DeterministicGateWalkForwardInputs:
    validation_summary: pd.DataFrame
    walk_forward_config: object | None = None
    target_column: str = CANONICAL_WALK_FORWARD_TARGET_COLUMN
    embargo_periods: int = CANONICAL_WALK_FORWARD_EMBARGO_PERIODS

    def __post_init__(self) -> None:
        _require_frame_columns(
            self.validation_summary,
            ("fold", "train_end", "test_start", "is_oos"),
            "validation_summary",
        )
        if self.target_column != CANONICAL_WALK_FORWARD_TARGET_COLUMN:
            raise ValueError("target_column must be forward_return_20")
        if int(self.embargo_periods) <= 0:
            raise ValueError("embargo_periods must be positive for forward_return_20")


@dataclass(frozen=True, slots=True)
class DeterministicGateRiskRuleInputs:
    portfolio_constraints: Mapping[str, Any] = field(
        default_factory=lambda: DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG.to_dict()
    )
    risk_rule_results: Sequence[Mapping[str, Any]] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        constraints = dict(self.portfolio_constraints)
        if constraints.get("long_only") is not True:
            raise ValueError("portfolio_constraints must be long-only")
        if float(constraints.get("max_symbol_weight", 0.0)) > 0.10:
            raise ValueError("max_symbol_weight must be at most 10%")
        if float(constraints.get("max_sector_weight", 0.0)) > 0.30:
            raise ValueError("max_sector_weight must be at most 30%")
        if int(constraints.get("max_holdings", 0)) > 20:
            raise ValueError("max_holdings must be at most 20")

    def to_dict(self) -> dict[str, Any]:
        return {
            "portfolio_constraints": dict(self.portfolio_constraints),
            "risk_rule_results": [dict(row) for row in self.risk_rule_results],
        }


@dataclass(frozen=True, slots=True)
class DeterministicGateInputs:
    predictions: pd.DataFrame
    backtest: DeterministicGateBacktestInputs
    walk_forward: DeterministicGateWalkForwardInputs
    costs: DeterministicGateCostInputs = field(default_factory=DeterministicGateCostInputs)
    risk: DeterministicGateRiskRuleInputs = field(
        default_factory=DeterministicGateRiskRuleInputs
    )
    ablation_summary: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    config: object | None = None
    thresholds: ValidationGateThresholds | None = None
    strategy_candidate_policy: StrategyCandidateGatePolicy | Mapping[str, Any] | None = None
    benchmark_return_series: pd.DataFrame | None = None
    equal_weight_baseline_return_series: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        _require_frame_columns(
            self.predictions,
            ("date", "ticker", "expected_return", CANONICAL_WALK_FORWARD_TARGET_COLUMN),
            "predictions",
        )

    @property
    def schema_id(self) -> str:
        return DETERMINISTIC_GATE_INTERFACE_ID

    @property
    def schema_version(self) -> str:
        return DETERMINISTIC_GATE_INTERFACE_SCHEMA_VERSION

    def to_report_kwargs(self) -> dict[str, Any]:
        return {
            "predictions": self.predictions,
            "validation_summary": self.walk_forward.validation_summary,
            "equity_curve": self.backtest.equity_curve,
            "strategy_metrics": self.backtest.strategy_metrics,
            "ablation_summary": [dict(row) for row in self.ablation_summary],
            "config": self._report_config(),
            "walk_forward_config": self.walk_forward.walk_forward_config,
            "thresholds": self.thresholds,
            "strategy_candidate_policy": self.strategy_candidate_policy,
            "benchmark_return_series": self.benchmark_return_series,
            "equal_weight_baseline_return_series": self.equal_weight_baseline_return_series,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "provider_free": True,
            "decision_engine": "deterministic",
            "llm_makes_trading_decisions": False,
            "model_predictions_are_order_signals": False,
            "target_horizon": CANONICAL_WALK_FORWARD_TARGET_COLUMN,
            "input_sections": [
                "predictions",
                "backtest",
                "transaction_costs",
                "slippage",
                "turnover",
                "walk_forward",
                "out_of_sample",
                "risk_rules",
                "benchmark",
                "ablation",
            ],
            "costs": self.costs.to_dict(),
            "risk": self.risk.to_dict(),
            "walk_forward": {
                "target_column": self.walk_forward.target_column,
                "embargo_periods": int(self.walk_forward.embargo_periods),
                "oos_fold_count": _oos_fold_count(self.walk_forward.validation_summary),
            },
        }

    def _report_config(self) -> object:
        if self.config is not None:
            return self.config
        constraints = dict(self.risk.portfolio_constraints)
        return SimpleNamespace(
            prediction_target_column=CANONICAL_WALK_FORWARD_TARGET_COLUMN,
            benchmark_ticker="SPY",
            gap_periods=int(self.walk_forward.embargo_periods),
            embargo_periods=int(self.walk_forward.embargo_periods),
            cost_bps=float(self.costs.cost_bps),
            slippage_bps=float(self.costs.slippage_bps),
            average_daily_turnover_budget=float(
                self.costs.average_daily_turnover_budget
            ),
            max_holdings=int(constraints["max_holdings"]),
            max_symbol_weight=float(constraints["max_symbol_weight"]),
            max_sector_weight=float(constraints["max_sector_weight"]),
        )


@runtime_checkable
class ProviderFreeDeterministicGate(Protocol):
    def evaluate(self, inputs: DeterministicGateInputs) -> ValidationGateReport:
        """Evaluate deterministic gate inputs without calling market/data providers."""


@dataclass(frozen=True, slots=True)
class BuildValidityGateReportProviderFree:
    """Adapter exposing the existing gate builder through a provider-free interface."""

    def evaluate(self, inputs: DeterministicGateInputs) -> ValidationGateReport:
        return build_validity_gate_report(**inputs.to_report_kwargs())


def evaluate_provider_free_deterministic_gate(
    inputs: DeterministicGateInputs,
    evaluator: ProviderFreeDeterministicGate | None = None,
) -> ValidationGateReport:
    evaluator = evaluator or BuildValidityGateReportProviderFree()
    return evaluator.evaluate(inputs)


def _require_frame_columns(
    frame: pd.DataFrame,
    required_columns: Sequence[str],
    frame_name: str,
) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def _oos_fold_count(validation_summary: pd.DataFrame) -> int:
    if "is_oos" not in validation_summary:
        return 0
    return int(validation_summary.loc[validation_summary["is_oos"].astype(bool), "fold"].nunique())
