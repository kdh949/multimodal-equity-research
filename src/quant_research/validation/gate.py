from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OFFICIAL_STRATEGY_FAIL_MESSAGE = "시스템은 유효하지만 현재 전략 후보는 배포/사용 부적합"
DEFAULT_HORIZONS = ("1d", "5d", "20d")


@dataclass(frozen=True)
class ValidationGateThresholds:
    min_folds: int = 5
    required_validation_horizon: int = 5
    min_rank_ic: float = 0.02
    min_positive_fold_ratio: float = 0.60
    max_daily_turnover: float = 0.35
    sharpe_pass: float = 0.80
    sharpe_warning: float = 0.50
    benchmark_sharpe_margin: float = 0.20
    drawdown_pass: float = -0.25
    drawdown_warning: float = -0.35
    max_drawdown_spy_lag: float = 0.05


@dataclass(frozen=True)
class ValidationGateReport:
    system_validity_status: str
    strategy_candidate_status: str
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

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_markdown(self) -> str:
        lines = [
            "# Validity Gate Report",
            "",
            f"- System validity: `{self.system_validity_status}`",
            f"- Strategy candidate: `{self.strategy_candidate_status}`",
            f"- Hard fail: `{self.hard_fail}`",
            f"- Official message: {self.official_message}",
            "",
            "## Gate Results",
            "",
            "| Gate | Status | Reason |",
            "|---|---|---|",
        ]
        for name, result in self.gate_results.items():
            reason = str(result.get("reason", "")).replace("|", "\\|")
            lines.append(f"| {name} | {result.get('status', '')} | {reason} |")

        if self.warnings:
            lines.extend(["", "## Warnings", ""])
            lines.extend(f"- {warning}" for warning in self.warnings)

        if self.hard_fail_reasons:
            lines.extend(["", "## Hard Fail Reasons", ""])
            lines.extend(f"- {reason}" for reason in self.hard_fail_reasons)

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
) -> ValidationGateReport:
    thresholds = thresholds or ValidationGateThresholds()
    ablation_summary = ablation_summary or []
    target_column = str(getattr(config, "prediction_target_column", "forward_return_5"))
    target_horizon = _horizon_from_target(target_column) or thresholds.required_validation_horizon
    required_horizon = int(getattr(config, "required_validation_horizon", thresholds.required_validation_horizon))
    benchmark_ticker = str(getattr(config, "benchmark_ticker", "SPY"))

    gap_periods = int(getattr(walk_forward_config, "gap_periods", getattr(config, "gap_periods", 0)))
    embargo_periods = int(
        getattr(walk_forward_config, "embargo_periods", getattr(config, "embargo_periods", 0))
    )
    min_train_observations = int(getattr(walk_forward_config, "min_train_observations", 0))

    gate_results: dict[str, dict[str, Any]] = {}
    hard_fail_reasons: list[str] = []
    warnings: list[str] = []
    insufficient_data = False

    leakage = _evaluate_leakage(validation_summary, gap_periods, embargo_periods, target_horizon)
    gate_results["leakage"] = leakage
    if leakage["status"] == "hard_fail":
        hard_fail_reasons.extend(leakage["reasons"])

    walk_forward = _evaluate_walk_forward(validation_summary, thresholds, min_train_observations)
    gate_results["walk_forward_oos"] = walk_forward
    if walk_forward["status"] == "hard_fail":
        hard_fail_reasons.extend(walk_forward["reasons"])
    elif walk_forward["status"] == "not_evaluable":
        insufficient_data = True

    rank_ic = _rank_ic_metrics(predictions, target_column)
    rank_gate = _evaluate_rank_ic(rank_ic, thresholds)
    gate_results["rank_ic"] = rank_gate
    if rank_gate["status"] == "fail":
        warnings.append(rank_gate["reason"])
    elif rank_gate["status"] == "not_evaluable":
        insufficient_data = True

    baseline_results = _benchmark_results(predictions, equity_curve, strategy_metrics, benchmark_ticker)
    cost_gate = _evaluate_cost_adjusted(baseline_results)
    gate_results["cost_adjusted_performance"] = cost_gate
    if cost_gate["status"] == "warning":
        warnings.append(cost_gate["reason"])

    benchmark_gate = _evaluate_benchmark(baseline_results, thresholds)
    gate_results["benchmark_comparison"] = benchmark_gate
    if benchmark_gate["status"] == "warning":
        warnings.append(benchmark_gate["reason"])

    turnover_gate = _evaluate_turnover(strategy_metrics, baseline_results, ablation_summary, thresholds)
    gate_results["turnover"] = turnover_gate
    if turnover_gate["status"] == "warning":
        warnings.append(turnover_gate["reason"])

    drawdown_gate = _evaluate_drawdown(baseline_results, thresholds)
    gate_results["drawdown"] = drawdown_gate
    if drawdown_gate["status"] == "warning":
        warnings.append(drawdown_gate["reason"])

    ablation_gate = _evaluate_ablation(ablation_summary)
    gate_results["ablation"] = ablation_gate
    if ablation_gate["status"] == "warning":
        warnings.append(ablation_gate["reason"])

    system_status = "hard_fail" if hard_fail_reasons else ("not_evaluable" if insufficient_data else "pass")
    strategy_status = _strategy_status(system_status, gate_results, insufficient_data)
    official_message = _official_message(system_status, strategy_status)
    metrics = {
        "fold_count": int(len(validation_summary)),
        "oos_fold_count": _count_oos_folds(validation_summary),
        "target_column": target_column,
        "target_horizon": target_horizon,
        "required_validation_horizon": required_horizon,
        "gap_periods": gap_periods,
        "embargo_periods": embargo_periods,
        "strategy_cagr": _metric(strategy_metrics, "cagr"),
        "strategy_sharpe": _metric(strategy_metrics, "sharpe"),
        "strategy_max_drawdown": _metric(strategy_metrics, "max_drawdown"),
        "strategy_turnover": _metric(strategy_metrics, "turnover"),
        "strategy_excess_return_vs_spy": baseline_results[0]["excess_return"],
        "strategy_excess_return_vs_equal_weight": baseline_results[1]["excess_return"],
        "mean_rank_ic": rank_ic.get("mean_rank_ic"),
        "positive_fold_ratio": rank_ic.get("positive_fold_ratio"),
        "oos_rank_ic": rank_ic.get("oos_rank_ic"),
    }
    evidence = {
        "thresholds": asdict(thresholds),
        "leakage": leakage,
        "walk_forward_oos": walk_forward,
        "rank_ic": rank_ic,
        "benchmark_ticker": benchmark_ticker,
        "baseline_results": baseline_results,
        "ablation_required_scenarios": [
            "price_only",
            "text_only",
            "sec_only",
            "no_model_proxy",
            "no_costs",
        ],
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
    )


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


def _evaluate_leakage(
    validation_summary: pd.DataFrame,
    gap_periods: int,
    embargo_periods: int,
    target_horizon: int,
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
    status = "hard_fail" if reasons else "pass"
    return {"status": status, "reason": "; ".join(reasons) if reasons else "no leakage guard violation", "reasons": reasons}


def _evaluate_walk_forward(
    validation_summary: pd.DataFrame,
    thresholds: ValidationGateThresholds,
    min_train_observations: int,
) -> dict[str, Any]:
    if validation_summary.empty:
        return {
            "status": "not_evaluable",
            "reason": "walk-forward summary is empty",
            "reasons": ["walk-forward summary is empty"],
        }
    reasons: list[str] = []
    if len(validation_summary) < thresholds.min_folds:
        reasons.append(f"fold_count={len(validation_summary)} is below required={thresholds.min_folds}")
    if "is_oos" not in validation_summary or not validation_summary["is_oos"].fillna(False).any():
        reasons.append("last OOS fold is missing")
    if {"train_end", "test_start"}.issubset(validation_summary.columns):
        train_end = pd.to_datetime(validation_summary["train_end"], errors="coerce")
        test_start = pd.to_datetime(validation_summary["test_start"], errors="coerce")
        if bool((train_end >= test_start).fillna(False).any()):
            reasons.append("train_end must be earlier than test_start for every fold")
    if min_train_observations and "train_observations" in validation_summary:
        too_small = validation_summary["train_observations"].fillna(0) < min_train_observations
        if bool(too_small.any()):
            reasons.append("one or more folds have too few train observations")
    if "labeled_test_observations" in validation_summary:
        if bool((validation_summary["labeled_test_observations"].fillna(0) <= 0).any()):
            reasons.append("one or more folds have no labeled test observations")
    status = "hard_fail" if reasons else "pass"
    return {"status": status, "reason": "; ".join(reasons) if reasons else "walk-forward/OOS structure is valid", "reasons": reasons}


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


def _evaluate_rank_ic(metrics: dict[str, Any], thresholds: ValidationGateThresholds) -> dict[str, Any]:
    mean_rank_ic = metrics.get("mean_rank_ic")
    positive_fold_ratio = metrics.get("positive_fold_ratio")
    oos_rank_ic = metrics.get("oos_rank_ic")
    if mean_rank_ic is None or positive_fold_ratio is None or oos_rank_ic is None:
        return {"status": "not_evaluable", "reason": "rank IC is not evaluable"}
    if mean_rank_ic <= 0:
        return {"status": "fail", "reason": f"mean_rank_ic={mean_rank_ic:.4f} is <= 0"}
    if (
        mean_rank_ic >= thresholds.min_rank_ic
        and positive_fold_ratio >= thresholds.min_positive_fold_ratio
        and oos_rank_ic > 0
    ):
        return {"status": "pass", "reason": "rank IC thresholds passed"}
    return {"status": "warning", "reason": "rank IC is positive but below one or more pass thresholds"}


def _benchmark_results(
    predictions: pd.DataFrame,
    equity_curve: pd.DataFrame,
    strategy_metrics: object,
    benchmark_ticker: str,
) -> list[dict[str, Any]]:
    strategy = {
        "name": "strategy",
        "cagr": _metric(strategy_metrics, "cagr"),
        "sharpe": _metric(strategy_metrics, "sharpe"),
        "max_drawdown": _metric(strategy_metrics, "max_drawdown"),
        "excess_return": 0.0,
    }
    spy = _spy_baseline_metrics(predictions, equity_curve, benchmark_ticker)
    equal_weight = _equal_weight_baseline_metrics(predictions)
    return [
        {
            "name": benchmark_ticker,
            "cagr": spy["cagr"],
            "sharpe": spy["sharpe"],
            "max_drawdown": spy["max_drawdown"],
            "excess_return": strategy["cagr"] - spy["cagr"],
            "strategy_sharpe": strategy["sharpe"],
            "strategy_max_drawdown": strategy["max_drawdown"],
        },
        {
            "name": "equal_weight",
            "cagr": equal_weight["cagr"],
            "sharpe": equal_weight["sharpe"],
            "max_drawdown": equal_weight["max_drawdown"],
            "excess_return": strategy["cagr"] - equal_weight["cagr"],
            "strategy_sharpe": strategy["sharpe"],
            "strategy_max_drawdown": strategy["max_drawdown"],
        },
    ]


def _evaluate_cost_adjusted(baseline_results: list[dict[str, Any]]) -> dict[str, Any]:
    exceeded = [float(result.get("excess_return", 0.0)) > 0 for result in baseline_results]
    if all(exceeded):
        return {"status": "pass", "reason": "net excess return is positive versus SPY and equal-weight baselines"}
    if any(exceeded):
        return {"status": "warning", "reason": "net excess return is positive versus only one baseline"}
    return {"status": "fail", "reason": "net excess return misses both baselines"}


def _evaluate_benchmark(
    baseline_results: list[dict[str, Any]],
    thresholds: ValidationGateThresholds,
) -> dict[str, Any]:
    strategy_sharpe = float(baseline_results[0].get("strategy_sharpe", np.nan))
    benchmark_sharpe = max(float(result.get("sharpe", 0.0)) for result in baseline_results)
    if not np.isfinite(strategy_sharpe):
        return {"status": "not_evaluable", "reason": "strategy Sharpe is unavailable"}
    if strategy_sharpe >= thresholds.sharpe_pass or strategy_sharpe - benchmark_sharpe >= thresholds.benchmark_sharpe_margin:
        return {"status": "pass", "reason": "strategy Sharpe passes absolute or benchmark-relative threshold"}
    if strategy_sharpe >= thresholds.sharpe_warning:
        return {"status": "warning", "reason": "strategy Sharpe is in warning band"}
    return {"status": "fail", "reason": "strategy Sharpe is below minimum warning threshold"}


def _evaluate_turnover(
    strategy_metrics: object,
    baseline_results: list[dict[str, Any]],
    ablation_summary: list[dict[str, Any]],
    thresholds: ValidationGateThresholds,
) -> dict[str, Any]:
    turnover = _metric(strategy_metrics, "turnover")
    if turnover <= thresholds.max_daily_turnover:
        return {"status": "pass", "reason": "average turnover is within budget"}
    no_cost = next((row for row in ablation_summary if row.get("scenario") == "no_costs"), None)
    cost_collapse = bool(
        no_cost
        and float(no_cost.get("excess_return", 0.0)) > 0
        and all(float(row.get("excess_return", 0.0)) <= 0 for row in baseline_results)
    )
    if cost_collapse:
        return {"status": "fail", "reason": "turnover exceeds budget and performance collapses after costs"}
    return {"status": "warning", "reason": "turnover exceeds budget but no cost collapse was detected"}


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


def _evaluate_ablation(ablation_summary: list[dict[str, Any]]) -> dict[str, Any]:
    required = {"price_only", "text_only", "sec_only", "no_model_proxy", "no_costs"}
    scenarios = {str(row.get("scenario")) for row in ablation_summary}
    missing = sorted(required - scenarios)
    if missing:
        return {"status": "warning", "reason": f"missing ablation scenarios: {', '.join(missing)}"}
    full = [row for row in ablation_summary if str(row.get("scenario")) in {"all_features", "full_model_features"}]
    others = [row for row in ablation_summary if row not in full and row.get("scenario") != "no_costs"]
    if not full or not others:
        return {"status": "not_evaluable", "reason": "full model or comparison ablations are unavailable"}
    full_sharpe = max(float(row.get("sharpe", 0.0)) for row in full)
    other_sharpe = max(float(row.get("sharpe", 0.0)) for row in others)
    if full_sharpe >= other_sharpe:
        return {"status": "pass", "reason": "full model matches or beats ablation alternatives"}
    return {"status": "warning", "reason": "full model does not beat one or more ablation alternatives"}


def _strategy_status(
    system_status: str,
    gate_results: dict[str, dict[str, Any]],
    insufficient_data: bool,
) -> str:
    if system_status == "hard_fail":
        return "not_evaluable"
    if insufficient_data:
        return "insufficient_data"
    statuses = {result.get("status") for result in gate_results.values()}
    if "fail" in statuses:
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


def _spy_baseline_metrics(
    predictions: pd.DataFrame,
    equity_curve: pd.DataFrame,
    benchmark_ticker: str,
) -> dict[str, float]:
    if not equity_curve.empty and "benchmark_return" in equity_curve:
        return _return_series_metrics(equity_curve["benchmark_return"])
    if predictions.empty or "forward_return_1" not in predictions:
        return _empty_metrics()
    frame = predictions[predictions["ticker"].astype(str).str.upper() == benchmark_ticker.upper()]
    return _return_series_metrics(frame["forward_return_1"]) if not frame.empty else _empty_metrics()


def _equal_weight_baseline_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    if predictions.empty or not {"date", "forward_return_1"}.issubset(predictions.columns):
        return _empty_metrics()
    returns = predictions.groupby("date")["forward_return_1"].mean()
    return _return_series_metrics(returns)


def _return_series_metrics(returns: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if values.empty:
        return _empty_metrics()
    years = max(len(values) / 252, 1 / 252)
    equity = (1 + values).cumprod()
    cagr = float(equity.iloc[-1] ** (1 / years) - 1)
    std = float(values.std(ddof=0))
    sharpe = float(values.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    drawdown = equity / equity.cummax() - 1
    return {"cagr": cagr, "sharpe": sharpe, "max_drawdown": float(drawdown.min())}


def _empty_metrics() -> dict[str, float]:
    return {"cagr": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}


def _horizon_from_target(target_column: str) -> int | None:
    prefix = "forward_return_"
    if not target_column.startswith(prefix):
        return None
    suffix = target_column.removeprefix(prefix)
    try:
        return int(suffix)
    except ValueError:
        return None


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
    return int(validation_summary["is_oos"].fillna(False).sum())


def _metric(metrics: object, name: str) -> float:
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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (str, bytes, dict, list, tuple)) else False:
        return None
    return value
