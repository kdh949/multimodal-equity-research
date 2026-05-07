from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from quant_research.backtest.metrics import (
    analyze_transaction_cost_scenarios,
    calculate_metrics,
)
from quant_research.validation.config import (
    TransactionCostSensitivityConfig,
    TransactionCostSensitivityScenario,
    default_transaction_cost_sensitivity_config,
)

TRANSACTION_COST_SENSITIVITY_BATCH_SCHEMA_VERSION = (
    "transaction_cost_sensitivity_batch.v1"
)
TRANSACTION_COST_SENSITIVITY_SUMMARY_SCHEMA_VERSION = (
    "transaction_cost_sensitivity_summary.v1"
)
TRANSACTION_COST_SENSITIVITY_BATCH_COLUMNS: tuple[str, ...] = (
    "schema_version",
    "batch_id",
    "config_id",
    "baseline_scenario_id",
    "scenario_id",
    "scenario",
    "label",
    "description",
    "is_baseline",
    "execution_mode",
    "status",
    "cost_bps",
    "slippage_bps",
    "total_cost_bps",
    "average_daily_turnover_budget",
    "max_daily_turnover",
    "observations",
    "return_basis",
    "cagr",
    "sharpe",
    "max_drawdown",
    "hit_rate",
    "turnover",
    "max_turnover",
    "turnover_budget_pass",
    "max_daily_turnover_pass",
    "gross_cumulative_return",
    "cost_adjusted_cumulative_return",
    "benchmark_cost_adjusted_cumulative_return",
    "excess_return",
    "transaction_cost_return",
    "slippage_cost_return",
    "total_cost_return",
    "baseline_cost_adjusted_cumulative_return_delta",
    "baseline_excess_return_delta",
    "baseline_total_cost_return_delta",
    "error_code",
    "error_message",
)
SENSITIVITY_STATUS_ORDER: tuple[str, ...] = (
    "pass",
    "warning",
    "insufficient_data",
    "error",
)

ScenarioRunner = Callable[[TransactionCostSensitivityScenario], object]


@dataclass(frozen=True)
class TransactionCostSensitivityBatchResult:
    summary: pd.DataFrame
    equity_curves: dict[str, pd.DataFrame]
    config: TransactionCostSensitivityConfig
    batch_id: str
    execution_mode: str
    summary_metrics: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": TRANSACTION_COST_SENSITIVITY_BATCH_SCHEMA_VERSION,
            "batch_id": self.batch_id,
            "execution_mode": self.execution_mode,
            "config": self.config.to_dict(),
            "scenario_count": len(self.summary),
            "summary_metrics": self.summary_metrics,
            "summary": self.summary.to_dict(orient="records"),
        }


def run_transaction_cost_sensitivity_batch(
    equity_curve_or_result: object | None = None,
    *,
    sensitivity_config: TransactionCostSensitivityConfig | None = None,
    scenario_runner: ScenarioRunner | None = None,
    batch_id: str | None = None,
) -> TransactionCostSensitivityBatchResult:
    """Run all configured cost/turnover sensitivity scenarios and return one table.

    Without ``scenario_runner`` this reprices a baseline equity curve, which is fast and
    keeps the signal path fixed. With ``scenario_runner`` each scenario can run a full
    backtest using its own cost, slippage, and turnover constraints.
    """

    config = sensitivity_config or default_transaction_cost_sensitivity_config()
    resolved_batch_id = _normalize_batch_id(batch_id, config)
    if scenario_runner is None:
        if equity_curve_or_result is None:
            raise ValueError(
                "equity_curve_or_result is required when scenario_runner is not provided"
            )
        analysis = analyze_transaction_cost_scenarios(
            equity_curve_or_result,
            sensitivity_config=config,
        )
        summary = build_transaction_cost_sensitivity_batch_table(
            analysis.summary,
            sensitivity_config=config,
            batch_id=resolved_batch_id,
            execution_mode="reprice",
        )
        return TransactionCostSensitivityBatchResult(
            summary=summary,
            equity_curves=analysis.equity_curves,
            config=config,
            batch_id=resolved_batch_id,
            execution_mode="reprice",
            summary_metrics=build_transaction_cost_sensitivity_summary_metrics(
                summary,
                sensitivity_config=config,
                batch_id=resolved_batch_id,
                execution_mode="reprice",
            ),
        )

    rows: list[dict[str, object]] = []
    equity_curves: dict[str, pd.DataFrame] = {}
    baseline_metrics: Any | None = None
    scenarios = tuple(config.scenarios)
    for scenario in scenarios:
        scenario_id = scenario.scenario_id
        try:
            curve = _coerce_runner_equity_curve(scenario_runner(scenario), scenario_id)
        except Exception as exc:  # pragma: no cover - exact exception type is runner-owned
            rows.append(
                _scenario_error_row(
                    scenario,
                    exc,
                    baseline_scenario_id=config.baseline_scenario_id,
                )
            )
            equity_curves[scenario_id] = pd.DataFrame()
            continue
        metrics = calculate_metrics(curve)
        equity_curves[scenario_id] = curve
        row = _scenario_metrics_row(
            scenario,
            metrics,
            curve,
            baseline_scenario_id=config.baseline_scenario_id,
        )
        if scenario_id == config.baseline_scenario_id:
            baseline_metrics = metrics
        rows.append(row)

    summary = pd.DataFrame(rows)
    if baseline_metrics is not None:
        summary["baseline_cost_adjusted_cumulative_return_delta"] = (
            pd.to_numeric(summary["cost_adjusted_cumulative_return"], errors="coerce")
            - float(baseline_metrics.cost_adjusted_cumulative_return)
        )
        summary["baseline_excess_return_delta"] = (
            pd.to_numeric(summary["excess_return"], errors="coerce")
            - float(baseline_metrics.excess_return)
        )
        summary["baseline_total_cost_return_delta"] = (
            pd.to_numeric(summary["total_cost_return"], errors="coerce")
            - float(baseline_metrics.total_cost_return)
        )
    summary = build_transaction_cost_sensitivity_batch_table(
        summary,
        sensitivity_config=config,
        batch_id=resolved_batch_id,
        execution_mode="runner",
    )
    return TransactionCostSensitivityBatchResult(
        summary=summary,
        equity_curves=equity_curves,
        config=config,
        batch_id=resolved_batch_id,
        execution_mode="runner",
        summary_metrics=build_transaction_cost_sensitivity_summary_metrics(
            summary,
            sensitivity_config=config,
            batch_id=resolved_batch_id,
            execution_mode="runner",
        ),
    )


def build_transaction_cost_sensitivity_batch_table(
    summary: object,
    *,
    sensitivity_config: TransactionCostSensitivityConfig | None = None,
    batch_id: str | None = None,
    execution_mode: str = "reprice",
) -> pd.DataFrame:
    config = sensitivity_config or default_transaction_cost_sensitivity_config()
    frame = pd.DataFrame(summary).copy()
    resolved_batch_id = _normalize_batch_id(batch_id, config)
    if frame.empty:
        return pd.DataFrame(columns=TRANSACTION_COST_SENSITIVITY_BATCH_COLUMNS)

    frame["scenario_id"] = frame.get("scenario_id", frame.get("scenario", "")).astype(str)
    frame["scenario"] = frame["scenario_id"]
    scenario_order = {scenario.scenario_id: idx for idx, scenario in enumerate(config.scenarios)}
    scenario_payloads = {scenario.scenario_id: scenario for scenario in config.scenarios}
    unknown = sorted(set(frame["scenario_id"]).difference(scenario_order))
    if unknown:
        raise ValueError(f"sensitivity summary contains unknown scenarios: {unknown}")
    if config.baseline_scenario_id not in set(frame["scenario_id"]):
        raise ValueError(
            "sensitivity summary must contain the configured baseline scenario "
            f"{config.baseline_scenario_id!r}"
        )

    frame["schema_version"] = TRANSACTION_COST_SENSITIVITY_BATCH_SCHEMA_VERSION
    frame["batch_id"] = resolved_batch_id
    frame["config_id"] = config.config_id
    frame["baseline_scenario_id"] = config.baseline_scenario_id
    frame["execution_mode"] = str(execution_mode)
    frame["description"] = frame["scenario_id"].map(
        lambda scenario_id: scenario_payloads[scenario_id].description
    )
    frame["status"] = frame.apply(_sensitivity_row_status, axis=1)
    frame["_scenario_order"] = frame["scenario_id"].map(scenario_order)
    frame = frame.sort_values("_scenario_order").drop(columns=["_scenario_order"])

    for column in TRANSACTION_COST_SENSITIVITY_BATCH_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    frame = _fill_baseline_delta_columns(frame, config.baseline_scenario_id)
    return frame.loc[:, TRANSACTION_COST_SENSITIVITY_BATCH_COLUMNS].reset_index(drop=True)


def build_transaction_cost_sensitivity_summary_metrics(
    summary: object,
    *,
    sensitivity_config: TransactionCostSensitivityConfig | None = None,
    batch_id: str | None = None,
    execution_mode: str | None = None,
) -> dict[str, object]:
    config = sensitivity_config or default_transaction_cost_sensitivity_config()
    frame = build_transaction_cost_sensitivity_batch_table(
        summary,
        sensitivity_config=config,
        batch_id=batch_id,
        execution_mode=execution_mode or "summary",
    )
    if frame.empty:
        return {
            "schema_version": TRANSACTION_COST_SENSITIVITY_SUMMARY_SCHEMA_VERSION,
            "batch_id": _normalize_batch_id(batch_id, config),
            "config_id": config.config_id,
            "baseline_scenario_id": config.baseline_scenario_id,
            "execution_mode": execution_mode or "summary",
            "scenario_order": [],
            "scenario_count": 0,
            "status_counts": {status: 0 for status in SENSITIVITY_STATUS_ORDER},
            "pass_count": 0,
            "warning_count": 0,
            "insufficient_data_count": 0,
            "error_count": 0,
            "all_scenarios_evaluable": False,
            "all_turnover_budgets_pass": False,
            "all_max_daily_turnover_limits_pass": False,
            "error_messages": [],
        }

    baseline = frame.loc[frame["scenario_id"] == config.baseline_scenario_id].iloc[0]
    status_counts = {
        status: int((frame["status"].astype(str) == status).sum())
        for status in SENSITIVITY_STATUS_ORDER
    }
    scenario_count = int(len(frame))
    turnover_budget_pass = _boolean_pass_series(frame["turnover_budget_pass"])
    max_daily_turnover_pass = _boolean_pass_series(frame["max_daily_turnover_pass"])

    return {
        "schema_version": TRANSACTION_COST_SENSITIVITY_SUMMARY_SCHEMA_VERSION,
        "batch_id": _normalize_batch_id(batch_id, config),
        "config_id": config.config_id,
        "baseline_scenario_id": config.baseline_scenario_id,
        "execution_mode": execution_mode or "summary",
        "scenario_order": frame["scenario_id"].astype(str).tolist(),
        "scenario_count": scenario_count,
        "status_counts": status_counts,
        "pass_count": status_counts["pass"],
        "warning_count": status_counts["warning"],
        "insufficient_data_count": status_counts["insufficient_data"],
        "error_count": status_counts["error"],
        "all_scenarios_evaluable": (
            status_counts["insufficient_data"] == 0 and status_counts["error"] == 0
        ),
        "all_turnover_budgets_pass": bool(turnover_budget_pass.all()),
        "all_max_daily_turnover_limits_pass": bool(max_daily_turnover_pass.all()),
        "turnover_budget_breach_count": int((~turnover_budget_pass).sum()),
        "max_daily_turnover_breach_count": int((~max_daily_turnover_pass).sum()),
        "baseline_status": str(baseline["status"]),
        "baseline_cost_adjusted_cumulative_return": _float_or_none(
            baseline["cost_adjusted_cumulative_return"]
        ),
        "baseline_excess_return": _float_or_none(baseline["excess_return"]),
        "baseline_total_cost_return": _float_or_none(baseline["total_cost_return"]),
        "best_cost_adjusted_scenario_id": _scenario_id_at_extreme(
            frame,
            "cost_adjusted_cumulative_return",
            largest=True,
        ),
        "worst_cost_adjusted_scenario_id": _scenario_id_at_extreme(
            frame,
            "cost_adjusted_cumulative_return",
            largest=False,
        ),
        "best_excess_return_scenario_id": _scenario_id_at_extreme(
            frame,
            "excess_return",
            largest=True,
        ),
        "largest_total_cost_scenario_id": _scenario_id_at_extreme(
            frame,
            "total_cost_return",
            largest=True,
        ),
        "max_cost_adjusted_return_loss_vs_baseline": _positive_loss_from_min_delta(
            frame,
            "baseline_cost_adjusted_cumulative_return_delta",
        ),
        "max_excess_return_loss_vs_baseline": _positive_loss_from_min_delta(
            frame,
            "baseline_excess_return_delta",
        ),
        "max_total_cost_increase_vs_baseline": _max_numeric(
            frame,
            "baseline_total_cost_return_delta",
        ),
        "error_messages": _sensitivity_error_messages(frame),
    }


def _scenario_metrics_row(
    scenario: TransactionCostSensitivityScenario,
    metrics: Any,
    equity_curve: pd.DataFrame,
    *,
    baseline_scenario_id: str,
) -> dict[str, object]:
    max_turnover = (
        float(pd.to_numeric(equity_curve["turnover"], errors="coerce").fillna(0.0).max())
        if "turnover" in equity_curve and not equity_curve.empty
        else 0.0
    )
    max_daily_turnover = scenario.max_daily_turnover
    return {
        "scenario_id": scenario.scenario_id,
        "scenario": scenario.scenario_id,
        "label": scenario.label,
        "is_baseline": scenario.scenario_id == baseline_scenario_id,
        "cost_bps": float(scenario.cost_bps),
        "slippage_bps": float(scenario.slippage_bps),
        "total_cost_bps": scenario.total_cost_bps,
        "average_daily_turnover_budget": float(scenario.average_daily_turnover_budget),
        "max_daily_turnover": None if max_daily_turnover is None else float(max_daily_turnover),
        "observations": int(len(equity_curve)),
        "return_basis": metrics.return_basis,
        "cagr": float(metrics.cagr),
        "sharpe": float(metrics.sharpe),
        "max_drawdown": float(metrics.max_drawdown),
        "hit_rate": float(metrics.hit_rate),
        "turnover": float(metrics.turnover),
        "max_turnover": max_turnover,
        "turnover_budget_pass": (
            float(metrics.turnover) <= float(scenario.average_daily_turnover_budget) + 1e-12
        ),
        "max_daily_turnover_pass": (
            True
            if max_daily_turnover is None
            else max_turnover <= float(max_daily_turnover) + 1e-12
        ),
        "gross_cumulative_return": float(metrics.gross_cumulative_return),
        "cost_adjusted_cumulative_return": float(metrics.cost_adjusted_cumulative_return),
        "benchmark_cost_adjusted_cumulative_return": float(
            metrics.benchmark_cost_adjusted_cumulative_return
        ),
        "excess_return": float(metrics.excess_return),
        "transaction_cost_return": float(metrics.transaction_cost_return),
        "slippage_cost_return": float(metrics.slippage_cost_return),
        "total_cost_return": float(metrics.total_cost_return),
        "error_code": None,
        "error_message": None,
    }


def _scenario_error_row(
    scenario: TransactionCostSensitivityScenario,
    exc: Exception,
    *,
    baseline_scenario_id: str,
) -> dict[str, object]:
    max_daily_turnover = scenario.max_daily_turnover
    return {
        "scenario_id": scenario.scenario_id,
        "scenario": scenario.scenario_id,
        "label": scenario.label,
        "is_baseline": scenario.scenario_id == baseline_scenario_id,
        "cost_bps": float(scenario.cost_bps),
        "slippage_bps": float(scenario.slippage_bps),
        "total_cost_bps": scenario.total_cost_bps,
        "average_daily_turnover_budget": float(scenario.average_daily_turnover_budget),
        "max_daily_turnover": None if max_daily_turnover is None else float(max_daily_turnover),
        "observations": 0,
        "turnover_budget_pass": False,
        "max_daily_turnover_pass": False,
        "error_code": exc.__class__.__name__,
        "error_message": str(exc) or exc.__class__.__name__,
    }


def _fill_baseline_delta_columns(
    frame: pd.DataFrame,
    baseline_scenario_id: str,
) -> pd.DataFrame:
    baseline = frame.loc[frame["scenario_id"] == baseline_scenario_id].iloc[0]
    delta_columns = {
        "baseline_cost_adjusted_cumulative_return_delta": "cost_adjusted_cumulative_return",
        "baseline_excess_return_delta": "excess_return",
        "baseline_total_cost_return_delta": "total_cost_return",
    }
    for delta_column, metric_column in delta_columns.items():
        if (
            delta_column in frame
            and frame[delta_column].notna().all()
            and not frame[delta_column].empty
        ):
            continue
        metric_values = pd.to_numeric(frame[metric_column], errors="coerce")
        baseline_value = pd.to_numeric(
            pd.Series([baseline[metric_column]]),
            errors="coerce",
        ).iloc[0]
        frame[delta_column] = metric_values - baseline_value
    return frame


def _boolean_pass_series(values: pd.Series) -> pd.Series:
    if values.empty:
        return pd.Series(dtype=bool)
    return values.fillna(False).astype(bool)


def _scenario_id_at_extreme(
    frame: pd.DataFrame,
    column: str,
    *,
    largest: bool,
) -> str | None:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.dropna().empty:
        return None
    idx = values.idxmax() if largest else values.idxmin()
    return str(frame.loc[idx, "scenario_id"])


def _positive_loss_from_min_delta(frame: pd.DataFrame, column: str) -> float | None:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return max(0.0, -float(values.min()))


def _max_numeric(frame: pd.DataFrame, column: str) -> float | None:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.max())


def _float_or_none(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _sensitivity_row_status(row: pd.Series) -> str:
    error_message = row.get("error_message")
    error_code = row.get("error_code")
    if pd.notna(error_message) or pd.notna(error_code):
        return "error"
    turnover_pass = bool(row.get("turnover_budget_pass", False))
    max_turnover_pass = bool(row.get("max_daily_turnover_pass", False))
    observations = pd.to_numeric(row.get("observations", 0), errors="coerce")
    if pd.isna(observations) or float(observations) <= 0:
        return "insufficient_data"
    return "pass" if turnover_pass and max_turnover_pass else "warning"


def _sensitivity_error_messages(frame: pd.DataFrame) -> list[str]:
    if "error_message" not in frame:
        return []
    errors = frame.loc[frame["error_message"].notna(), ["scenario_id", "error_message"]]
    return [
        f"{row.scenario_id}: {row.error_message}"
        for row in errors.itertuples(index=False)
        if str(row.error_message).strip()
    ]


def _coerce_runner_equity_curve(value: object, scenario_id: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if hasattr(value, "equity_curve") and isinstance(value.equity_curve, pd.DataFrame):
        return value.equity_curve.copy()
    raise TypeError(
        f"sensitivity scenario {scenario_id} runner must return a DataFrame or object "
        "with an equity_curve DataFrame"
    )


def _normalize_batch_id(
    batch_id: str | None,
    config: TransactionCostSensitivityConfig,
) -> str:
    resolved = str(batch_id or config.config_id).strip()
    if not resolved:
        raise ValueError("batch_id must not be blank")
    return resolved
