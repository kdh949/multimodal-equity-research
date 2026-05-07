from __future__ import annotations

# ruff: noqa: E402, I001

import json
from types import SimpleNamespace

from quant_research.runtime import configure_local_runtime_defaults

configure_local_runtime_defaults()

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quant_research.config import DEFAULT_BENCHMARK_TICKER, DEFAULT_TICKERS
from quant_research.dashboard import build_beginner_research_dashboard
from quant_research.dashboard.streamlit import render_beginner_overview
from quant_research.pipeline import PipelineConfig, run_research_pipeline
from quant_research.signals.engine import (
    SignalGenerationBlockedError,
    require_signal_generation_gate_pass,
)
from quant_research.validation import (
    TRANSACTION_COST_SENSITIVITY_BATCH_COLUMNS,
    build_validity_gate_report,
    build_transaction_cost_sensitivity_summary_metrics,
    run_transaction_cost_sensitivity_batch,
)


HORIZON_DIAGNOSTIC_COLUMNS = [
    "horizon",
    "label",
    "role",
    "target_column",
    "affects_pass_fail",
    "status",
    "insufficient_data",
    "insufficient_data_status",
    "insufficient_data_code",
    "rank_ic_status",
    "insufficient_data_reason",
    "mean_rank_ic",
    "positive_fold_ratio",
    "positive_fold_ratio_threshold",
    "positive_fold_ratio_passed",
    "oos_rank_ic",
    "rank_ic_count",
]
HORIZON_STATUS_SUMMARY_LABELS = ("1d", "20d")
STRUCTURED_WARNING_COLUMNS = [
    "code",
    "severity",
    "gate",
    "combined_gate",
    "metric",
    "realized_turnover",
    "budget",
    "value",
    "threshold",
    "operator",
    "reason",
    "message",
    "monthly_turnover",
]
TURNOVER_BUDGET_WARNING_CODES = {
    "average_daily_turnover_budget_exceeded",
    "monthly_turnover_budget_exceeded",
}
TURNOVER_BUDGET_WARNING_COLUMNS = [
    "code",
    "severity",
    "gate",
    "combined_gate",
    "metric",
    "realized_turnover",
    "budget",
    "operator",
    "message",
    "reason",
    "monthly_turnover",
]
GATE_RESULT_COLUMNS = [
    "gate",
    "status",
    "reason",
    "reason_code",
    "metric",
    "value",
    "threshold",
    "operator",
]
RULE_EXPLANATION_COLUMNS = [
    "rule",
    "status",
    "passed",
    "affects_strategy",
    "affects_system",
    "reason",
    "reason_code",
    "metric",
    "value",
    "threshold",
    "operator",
]
FINAL_STRATEGY_STATUS_COLUMNS = [
    "final_strategy_status",
    "system_validity_status",
    "strategy_pass",
    "blocking_rules",
    "warning_rules",
    "insufficient_data_rules",
    "reason",
    "official_message",
]
STRATEGY_FAILURE_COLUMNS = [
    "gate",
    "status",
    "reason",
    "reason_code",
    "collapse_status",
    "collapse_reason",
    "metric",
    "value",
    "threshold",
    "operator",
]
SIDE_BY_SIDE_BASE_COLUMNS = ["metric", "metric_label", "strategy", "equal_weight"]
THREE_STRATEGY_COMPARISON_COLUMNS = [
    "strategy",
    "role",
    "return_basis",
    "cagr",
    "sharpe",
    "max_drawdown",
    "cost_adjusted_cumulative_return",
    "average_daily_turnover",
    "transaction_cost_return",
    "slippage_cost_return",
    "total_cost_return",
    "strategy_excess_return",
    "cost_bps",
    "slippage_bps",
    "evaluation_observations",
    "evaluation_start",
    "evaluation_end",
]
MODEL_COMPARISON_COLUMNS = [
    "window_id",
    "metric",
    "candidate",
    "candidate_value",
    "baseline",
    "baseline_role",
    "baseline_value",
    "absolute_delta",
    "relative_delta",
    "pass_fail",
    "operator",
]
METRIC_CONTRACT_COLUMNS = [
    "entity_id",
    "role",
    "kind",
    "status",
    "mean_rank_ic",
    "positive_fold_ratio",
    "positive_fold_ratio_threshold",
    "positive_fold_ratio_passed",
    "oos_rank_ic",
    "sharpe",
    "rank_ic",
    "max_drawdown",
    "cost_adjusted_cumulative_return",
    "excess_return",
    "turnover",
    "average_daily_turnover",
]
BACKTEST_RISK_SIZING_COLUMNS = [
    "date",
    "portfolio_volatility_estimate",
    "position_sizing_validation_status",
    "position_sizing_validation_rule",
    "position_sizing_validation_reason",
    "position_count",
    "max_position_weight",
    "max_sector_exposure",
    "gross_exposure",
    "net_exposure",
    "max_position_risk_contribution",
    "post_cost_validation_total_cost_return",
]
PORTFOLIO_RISK_CONFIG_COLUMNS = [
    "covariance_aware_risk_enabled",
    "covariance_return_column",
    "portfolio_covariance_lookback",
    "covariance_min_periods",
    "portfolio_volatility_limit",
    "max_symbol_weight",
    "max_sector_weight",
    "max_position_risk_contribution",
    "average_daily_turnover_budget",
    "max_daily_turnover",
    "max_drawdown_stop",
    "volatility_adjustment_strength",
    "concentration_adjustment_strength",
    "risk_contribution_adjustment_strength",
    "v1_exclusions",
]
STRUCTURED_PASS_FAIL_REASON_COLUMNS = [
    "category",
    "entity_id",
    "rule",
    "metric",
    "status",
    "passed",
    "reason_code",
    "reason",
    "value",
    "threshold",
    "operator",
    "candidate",
    "baseline",
    "window_id",
    "candidate_value",
    "baseline_value",
]
USER_GATE_STATUS_COLUMNS = [
    "scope",
    "gate",
    "display_status",
    "severity",
    "reason",
    "reason_code",
    "metric",
    "value",
    "threshold",
    "operator",
]
WALK_FORWARD_FOLD_PERIOD_COLUMNS = [
    "fold",
    "fold_type",
    "is_oos",
    "train_start",
    "train_end",
    "validation_start",
    "validation_end",
    "test_start",
    "test_end",
    "oos_test_start",
    "oos_test_end",
    "purge_periods",
    "purge_start",
    "purge_end",
    "purged_date_count",
    "purge_applied",
    "embargo_periods",
    "embargo_start",
    "embargo_end",
    "embargoed_date_count",
    "embargo_applied",
]
OOS_PERFORMANCE_SUMMARY_COLUMNS = [
    "oos_fold_count",
    "oos_start",
    "oos_end",
    "oos_rank_ic",
    "oos_rank_ic_positive_fold_ratio",
    "oos_rank_ic_count",
    "oos_prediction_count",
    "oos_labeled_prediction_count",
    "oos_mean_mae",
    "oos_mean_directional_accuracy",
    "oos_mean_information_coefficient",
    "walk_forward_fold_count",
    "walk_forward_mean_rank_ic",
    "walk_forward_positive_rank_ic_fold_ratio",
]
TRANSACTION_COST_SENSITIVITY_DISPLAY_COLUMNS = [
    "scenario_id",
    "label",
    "status",
    "is_baseline",
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
    "turnover",
    "max_turnover",
    "turnover_budget_pass",
    "max_daily_turnover_pass",
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
]
TRANSACTION_COST_SENSITIVITY_SUMMARY_COLUMNS = [
    "batch_id",
    "config_id",
    "baseline_scenario_id",
    "execution_mode",
    "scenario_count",
    "pass_count",
    "warning_count",
    "insufficient_data_count",
    "error_count",
    "all_scenarios_evaluable",
    "all_turnover_budgets_pass",
    "all_max_daily_turnover_limits_pass",
    "turnover_budget_breach_count",
    "max_daily_turnover_breach_count",
    "baseline_status",
    "baseline_cost_adjusted_cumulative_return",
    "baseline_excess_return",
    "baseline_total_cost_return",
    "best_cost_adjusted_scenario_id",
    "worst_cost_adjusted_scenario_id",
    "largest_total_cost_scenario_id",
    "max_cost_adjusted_return_loss_vs_baseline",
    "max_excess_return_loss_vs_baseline",
    "max_total_cost_increase_vs_baseline",
    "error_messages",
]
TRANSACTION_COST_SENSITIVITY_REVIEW_COLUMNS = [
    "check",
    "status",
    "scenario_id",
    "value",
    "threshold",
    "operator",
    "reason",
]
REPORT_ONLY_RESEARCH_METRIC_COLUMNS = [
    "metric",
    "status",
    "value",
    "target_column",
    "sample_scope",
    "report_only",
    "decision_use",
    "reason",
]
GATE_REPORT_ARTIFACT_COLUMNS = [
    "artifact",
    "format",
    "report_path",
    "system_validity_status",
    "strategy_candidate_status",
    "final_gate_decision",
    "gate_result_count",
    "structured_reason_count",
    "warning_count",
    "includes_gate_results",
    "includes_system_validity_gate",
    "includes_strategy_candidate_gate",
]
REPORT_APPROVAL_COLUMNS = [
    "approval_status",
    "approval_allowed",
    "final_gate_decision",
    "final_status",
    "system_validity_status",
    "strategy_candidate_status",
    "reason",
]


def _validity_horizon_rows(validity_report: object) -> list[dict[str, object]]:
    metrics = getattr(validity_report, "metrics", {})
    horizon_metrics = metrics.get("horizon_metrics", {}) if isinstance(metrics, dict) else {}
    if not isinstance(horizon_metrics, dict):
        return []

    ordered_horizons = getattr(validity_report, "horizons", None)
    if not isinstance(ordered_horizons, list | tuple):
        ordered_horizons = tuple(horizon_metrics)

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for horizon_label in [*ordered_horizons, *horizon_metrics.keys()]:
        horizon_key = str(horizon_label)
        if horizon_key in seen:
            continue
        seen.add(horizon_key)
        metric_row = horizon_metrics.get(horizon_key)
        if not isinstance(metric_row, dict):
            continue
        row = dict(metric_row)
        row["horizon_periods"] = row.get("horizon")
        row["horizon"] = horizon_key
        rows.append(row)
    return rows


def _validity_structured_warning_rows(validity_report: object) -> list[dict[str, object]]:
    warnings = getattr(validity_report, "structured_warnings", [])
    if not isinstance(warnings, list | tuple):
        return []
    return [dict(row) for row in warnings if isinstance(row, dict)]


def _validity_turnover_budget_warning_rows(validity_report: object) -> list[dict[str, object]]:
    rows = _validity_structured_warning_rows(validity_report)
    return [
        row
        for row in rows
        if row.get("combined_gate") == "turnover"
        or row.get("code") in TURNOVER_BUDGET_WARNING_CODES
    ]


def _validity_gate_result_rows(validity_report: object) -> list[dict[str, object]]:
    gate_results = getattr(validity_report, "gate_results", {})
    if not isinstance(gate_results, dict):
        return []

    rows: list[dict[str, object]] = []
    for gate, result in gate_results.items():
        if not isinstance(result, dict):
            continue
        metadata = result.get("reason_metadata")
        if not isinstance(metadata, dict):
            metadata = result.get("collapse_check", {})
        if not isinstance(metadata, dict):
            metadata = {}
        rows.append(
            {
                "gate": gate,
                "status": result.get("status"),
                "reason": result.get("reason"),
                "reason_code": result.get("reason_code")
                or result.get("collapse_reason_code")
                or metadata.get("code"),
                "metric": metadata.get("metric"),
                "value": metadata.get("value"),
                "threshold": metadata.get("threshold"),
                "operator": metadata.get("operator"),
            }
        )
    return rows


def _validity_rule_explanation_rows(validity_report: object) -> list[dict[str, object]]:
    rows = getattr(validity_report, "rule_result_explanations", [])
    if not isinstance(rows, list | tuple):
        return []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _validity_final_strategy_status_rows(validity_report: object) -> list[dict[str, object]]:
    explanation = getattr(validity_report, "final_strategy_status_explanation", {})
    if not isinstance(explanation, dict):
        return []
    return [dict(explanation)]


def _validity_report_only_research_metric_rows(
    validity_report: object,
) -> list[dict[str, object]]:
    evidence = getattr(validity_report, "evidence", {})
    if not isinstance(evidence, dict):
        return []
    top_decile = evidence.get("top_decile_20d_excess_return", {})
    if not isinstance(top_decile, dict) or not top_decile:
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
            "reason": top_decile.get("reason"),
        }
    ]


def _validity_metric_contract_rows(rows: object) -> list[dict[str, object]]:
    if not isinstance(rows, list | tuple):
        return []
    output: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        metrics = row.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        validation_metrics = row.get("validation_metrics", {})
        if not isinstance(validation_metrics, dict):
            validation_metrics = {}
        output.append({**row, **validation_metrics, **metrics})
    return output


def _validity_full_model_metric_rows(validity_report: object) -> list[dict[str, object]]:
    full_model_metrics = getattr(validity_report, "full_model_metrics", {})
    if not isinstance(full_model_metrics, dict) or not full_model_metrics:
        return []
    return _validity_metric_contract_rows([full_model_metrics])


def _validity_strategy_failure_rows(validity_report: object) -> list[dict[str, object]]:
    summary = getattr(validity_report, "strategy_failure_summary", None)
    if isinstance(summary, list | tuple):
        return [dict(row) for row in summary if isinstance(row, dict)]
    return [
        row
        for row in _validity_gate_result_rows(validity_report)
        if row.get("status") == "fail"
    ]


def _validity_gate_report_payload(validity_report: object) -> dict[str, object]:
    to_dict = getattr(validity_report, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
    else:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    gate_results = payload.get("gate_results", {})
    structured_reasons = payload.get("structured_pass_fail_reasons", [])
    structured_warnings = payload.get("structured_warnings", [])
    if not isinstance(gate_results, dict):
        gate_results = {}
    if not isinstance(structured_reasons, list):
        structured_reasons = []
    if not isinstance(structured_warnings, list):
        structured_warnings = []

    artifact_manifest = {
        "schema_version": "canonical_gate_report_artifact_manifest.v1",
        "artifacts": [
            {
                "artifact": "validity_gate.json",
                "format": "json",
                "report_path": "reports/validity_gate.json",
                "contains": [
                    "system_validity_status",
                    "strategy_candidate_status",
                    "gate_results",
                    "structured_pass_fail_reasons",
                    "structured_warnings",
                    "metrics",
                    "evidence",
                ],
            },
            {
                "artifact": "validity_report.md",
                "format": "markdown",
                "report_path": "reports/validity_report.md",
                "contains": [
                    "validity_gate_result_summary",
                    "final_strategy_status_explanation",
                    "gate_results",
                    "strategy_failure_summary",
                    "structured_warnings",
                ],
            },
        ],
    }
    payload = dict(payload)
    payload["artifact_manifest"] = artifact_manifest
    payload["report_path"] = "reports/validity_report.md"
    payload["gate_report_data_included"] = {
        "gate_result_count": len(gate_results),
        "structured_reason_count": len(structured_reasons),
        "warning_count": len(structured_warnings),
        "includes_gate_results": bool(gate_results),
        "includes_system_validity_gate": "system_validity_artifact_contract" in gate_results,
        "includes_strategy_candidate_gate": "deterministic_strategy_validity" in gate_results,
    }
    return payload


def _validity_gate_report_artifact_rows(validity_report: object) -> list[dict[str, object]]:
    payload = _validity_gate_report_payload(validity_report)
    included = payload.get("gate_report_data_included", {})
    if not isinstance(included, dict):
        included = {}
    manifest = payload.get("artifact_manifest", {})
    artifacts = manifest.get("artifacts", []) if isinstance(manifest, dict) else []
    if not isinstance(artifacts, list):
        artifacts = []
    final_gate = payload.get("final_gate_decision") or (
        payload.get("metrics", {}).get("final_gate_decision")
        if isinstance(payload.get("metrics"), dict)
        else None
    )
    rows = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        rows.append(
            {
                "artifact": artifact.get("artifact"),
                "format": artifact.get("format"),
                "report_path": artifact.get("report_path"),
                "system_validity_status": payload.get("system_validity_status"),
                "strategy_candidate_status": payload.get("strategy_candidate_status"),
                "final_gate_decision": final_gate,
                "gate_result_count": included.get("gate_result_count"),
                "structured_reason_count": included.get("structured_reason_count"),
                "warning_count": included.get("warning_count"),
                "includes_gate_results": included.get("includes_gate_results"),
                "includes_system_validity_gate": included.get("includes_system_validity_gate"),
                "includes_strategy_candidate_gate": included.get(
                    "includes_strategy_candidate_gate"
                ),
            }
        )
    return rows


def _validity_report_approval_gate(validity_report: object) -> dict[str, object]:
    try:
        gate_payload = require_signal_generation_gate_pass(
            validity_report,
            required=True,
        )
    except SignalGenerationBlockedError as exc:
        payload = _validity_gate_report_payload(validity_report)
        summary = payload.get("validity_gate_result_summary", {})
        if not isinstance(summary, dict):
            summary = {}
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        return {
            "approval_status": "blocked",
            "approval_allowed": False,
            "final_gate_decision": (
                summary.get("final_gate_decision")
                or metrics.get("final_gate_decision")
                or "FAIL"
            ),
            "final_status": (
                summary.get("final_status")
                or payload.get("strategy_candidate_status")
                or payload.get("system_validity_status")
                or "not_evaluable"
            ),
            "system_validity_status": payload.get("system_validity_status"),
            "strategy_candidate_status": payload.get("strategy_candidate_status"),
            "reason": str(exc),
        }

    gate_payload = gate_payload or {}
    return {
        "approval_status": "approved",
        "approval_allowed": True,
        "final_gate_decision": gate_payload.get("final_gate_decision", "PASS"),
        "final_status": gate_payload.get("final_status", "pass"),
        "system_validity_status": gate_payload.get("system_validity_status"),
        "strategy_candidate_status": gate_payload.get("strategy_candidate_status"),
        "reason": gate_payload.get("reason"),
    }


def _render_validity_report_approval(validity_report: object) -> None:
    st.subheader("Canonical Report Approval")
    approval = st.session_state.get("validity_report_approval")
    if isinstance(approval, dict):
        frame = pd.DataFrame([approval])
        st.dataframe(
            frame[[col for col in REPORT_APPROVAL_COLUMNS if col in frame.columns]],
            width="stretch",
            hide_index=True,
        )

    if st.button("Approve Canonical Report", width="stretch"):
        approval = _validity_report_approval_gate(validity_report)
        st.session_state["validity_report_approval"] = approval
        if approval["approval_allowed"]:
            st.success("Canonical report approved after common gate PASS.")
        else:
            st.error(str(approval["reason"]))


def _format_status_value(value: object) -> str:
    if value is None:
        return "not_evaluable"
    if isinstance(value, float) and pd.isna(value):
        return "not_evaluable"
    text = str(value)
    return text if text else "not_evaluable"


def _display_gate_status(status: object) -> str:
    normalized = _format_status_value(status).lower()
    return {
        "pass": "PASS",
        "warning": "WARN",
        "hard_fail": "FAIL",
        "fail": "FAIL",
        "insufficient_data": "NEEDS DATA",
        "not_evaluable": "NOT EVALUABLE",
        "diagnostic": "DIAGNOSTIC",
        "skipped": "SKIPPED",
    }.get(normalized, normalized.upper())


def _status_severity(status: object) -> str:
    normalized = _format_status_value(status).lower()
    if normalized in {"hard_fail", "fail"}:
        return "error"
    if normalized in {"warning", "insufficient_data", "not_evaluable"}:
        return "warning"
    if normalized == "pass":
        return "success"
    return "info"


def _first_related_metric(row: dict[str, object]) -> dict[str, object]:
    metrics = row.get("related_metrics")
    if isinstance(metrics, list):
        for metric in metrics:
            if isinstance(metric, dict):
                return metric
    return {}


def _validity_user_gate_status_rows(validity_report: object) -> list[dict[str, object]]:
    payload = _validity_gate_report_payload(validity_report)
    summary = payload.get("validity_gate_result_summary", {})
    if not isinstance(summary, dict):
        summary = {}
    failure_report = payload.get("structured_gate_failure_report", {})
    if not isinstance(failure_report, dict):
        failure_report = {}

    rows: list[dict[str, object]] = [
        {
            "scope": "system",
            "gate": "system_validity",
            "display_status": _display_gate_status(
                payload.get("system_validity_status")
            ),
            "severity": _status_severity(payload.get("system_validity_status")),
            "reason": payload.get("official_message"),
            "reason_code": None,
            "metric": None,
            "value": None,
            "threshold": None,
            "operator": None,
        },
        {
            "scope": "strategy",
            "gate": "strategy_candidate",
            "display_status": _display_gate_status(
                payload.get("strategy_candidate_status")
            ),
            "severity": _status_severity(payload.get("strategy_candidate_status")),
            "reason": summary.get("deterministic_gate", {}).get("reason")
            if isinstance(summary.get("deterministic_gate"), dict)
            else payload.get("official_message"),
            "reason_code": None,
            "metric": None,
            "value": None,
            "threshold": None,
            "operator": None,
        },
    ]

    gate_rows = failure_report.get("gates", [])
    if isinstance(gate_rows, list):
        for row in gate_rows:
            if not isinstance(row, dict):
                continue
            metric = _first_related_metric(row)
            status = row.get("status") or row.get("severity")
            rows.append(
                {
                    "scope": "rule",
                    "gate": row.get("gate"),
                    "display_status": _display_gate_status(status),
                    "severity": _status_severity(row.get("severity") or status),
                    "reason": row.get("top_reason"),
                    "reason_code": row.get("top_reason_code"),
                    "metric": metric.get("metric"),
                    "value": metric.get("value"),
                    "threshold": metric.get("threshold"),
                    "operator": metric.get("operator"),
                }
            )
    return rows


def _render_user_gate_status(validity_report: object) -> None:
    rows = _validity_user_gate_status_rows(validity_report)
    if not rows:
        return

    st.subheader("User Gate Status")
    primary_rows = [row for row in rows if row.get("scope") in {"system", "strategy"}]
    for row in primary_rows:
        message = (
            f"{row.get('gate')}: {row.get('display_status')}"
            f" - {row.get('reason') or 'No failure reason reported.'}"
        )
        severity = row.get("severity")
        if severity == "error":
            st.error(message)
        elif severity == "warning":
            st.warning(message)
        elif severity == "success":
            st.success(message)
        else:
            st.info(message)

    frame = pd.DataFrame(rows)
    st.dataframe(
        frame[[col for col in USER_GATE_STATUS_COLUMNS if col in frame.columns]],
        width="stretch",
        hide_index=True,
    )


def _render_horizon_status_summary(horizon_rows: list[dict[str, object]]) -> None:
    summary_rows = [
        row for row in horizon_rows if str(row.get("horizon")) in HORIZON_STATUS_SUMMARY_LABELS
    ]
    if not summary_rows:
        return

    summary_cols = st.columns(len(summary_rows))
    for column, row in zip(summary_cols, summary_rows, strict=True):
        horizon = _format_status_value(row.get("horizon"))
        label = _format_status_value(row.get("label")).replace("_", " ").title()
        status = _format_status_value(row.get("status"))
        column.metric(f"{horizon} {label}", status)
        insufficient_status = row.get("insufficient_data_status")
        if insufficient_status:
            column.caption(f"Insufficient data: {insufficient_status}")
        reason = row.get("insufficient_data_reason")
        if reason:
            column.caption(str(reason))


def _format_metric_value(value: object, *, percent: bool = False) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and pd.isna(value):
        return "N/A"
    if isinstance(value, pd.Timestamp) and pd.isna(value):
        return "N/A"
    if percent:
        return f"{float(value):.1%}"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _portfolio_risk_config_frame(config: PipelineConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "covariance_aware_risk_enabled": bool(config.covariance_aware_risk_enabled),
                "covariance_return_column": config.covariance_return_column,
                "portfolio_covariance_lookback": int(config.portfolio_covariance_lookback),
                "covariance_min_periods": int(config.covariance_min_periods),
                "portfolio_volatility_limit": float(config.portfolio_volatility_limit),
                "max_symbol_weight": float(config.max_symbol_weight),
                "max_sector_weight": float(config.max_sector_weight),
                "max_position_risk_contribution": float(
                    config.max_position_risk_contribution
                ),
                "average_daily_turnover_budget": float(
                    config.average_daily_turnover_budget
                ),
                "max_daily_turnover": config.max_daily_turnover,
                "max_drawdown_stop": float(config.max_drawdown_stop),
                "volatility_adjustment_strength": float(
                    config.volatility_adjustment_strength
                ),
                "concentration_adjustment_strength": float(
                    config.concentration_adjustment_strength
                ),
                "risk_contribution_adjustment_strength": float(
                    config.risk_contribution_adjustment_strength
                ),
                "v1_exclusions": "correlation_cluster_weight",
            }
        ],
        columns=PORTFOLIO_RISK_CONFIG_COLUMNS,
    )


def _walk_forward_oos_summary_row(validation_summary: pd.DataFrame) -> dict[str, object]:
    if validation_summary.empty:
        return {}
    summary_columns = [
        column
        for column in OOS_PERFORMANCE_SUMMARY_COLUMNS
        if column in validation_summary.columns
    ]
    if summary_columns:
        return validation_summary.iloc[-1][summary_columns].to_dict()

    if "is_oos" not in validation_summary.columns:
        return {}
    oos = validation_summary[validation_summary["is_oos"].fillna(False).astype(bool)]
    if oos.empty:
        return {"oos_fold_count": 0}
    row: dict[str, object] = {"oos_fold_count": int(len(oos))}
    if "test_start" in oos:
        row["oos_start"] = oos["test_start"].min()
    if "test_end" in oos:
        row["oos_end"] = oos["test_end"].max()
    if "mae" in oos:
        row["oos_mean_mae"] = pd.to_numeric(oos["mae"], errors="coerce").mean()
    if "directional_accuracy" in oos:
        row["oos_mean_directional_accuracy"] = pd.to_numeric(
            oos["directional_accuracy"],
            errors="coerce",
        ).mean()
    return row


def _render_walk_forward_periods_and_oos_summary(
    validation_summary: pd.DataFrame,
    config: PipelineConfig,
) -> None:
    st.subheader("Walk-Forward Fold Periods and Purge/Embargo")
    if validation_summary.empty:
        st.info("Walk-forward fold 기간과 purge/embargo 설정을 표시할 데이터가 아직 없습니다.")
        return

    setting_cols = st.columns(4)
    setting_cols[0].metric("Train Periods", int(config.train_periods))
    setting_cols[1].metric("Test Periods", int(config.test_periods))
    setting_cols[2].metric("Purge Periods", int(config.gap_periods))
    setting_cols[3].metric("Embargo Periods", int(config.embargo_periods))
    st.caption(
        "Fold별 train/test 기간과 t+1 이후 수익률 적용 전 purge/embargo 격리 구간을 함께 표시합니다."
    )

    fold_columns = [
        column for column in WALK_FORWARD_FOLD_PERIOD_COLUMNS if column in validation_summary.columns
    ]
    if fold_columns:
        st.dataframe(
            validation_summary[fold_columns],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("Fold 기간 컬럼이 아직 validation summary에 없습니다.")

    st.subheader("OOS Performance Summary")
    oos_summary = _walk_forward_oos_summary_row(validation_summary)
    if not oos_summary:
        st.info("OOS 성능 요약을 계산할 데이터가 아직 없습니다.")
        return

    oos_cols = st.columns(4)
    oos_cols[0].metric("OOS Folds", _format_metric_value(oos_summary.get("oos_fold_count")))
    oos_cols[1].metric("OOS Rank IC", _format_metric_value(oos_summary.get("oos_rank_ic")))
    oos_cols[2].metric(
        "Positive OOS Folds",
        _format_metric_value(oos_summary.get("oos_rank_ic_positive_fold_ratio"), percent=True),
    )
    oos_cols[3].metric("OOS MAE", _format_metric_value(oos_summary.get("oos_mean_mae")))

    oos_frame = pd.DataFrame([oos_summary])
    st.dataframe(
        oos_frame[
            [column for column in OOS_PERFORMANCE_SUMMARY_COLUMNS if column in oos_frame.columns]
        ],
        width="stretch",
        hide_index=True,
    )


def _transaction_cost_sensitivity_result(result: object, config: PipelineConfig) -> object | None:
    sensitivity_result = getattr(result, "transaction_cost_sensitivity", None)
    if sensitivity_result is not None:
        return sensitivity_result
    backtest = getattr(result, "backtest", None)
    equity_curve = getattr(backtest, "equity_curve", None)
    if not isinstance(equity_curve, pd.DataFrame) or equity_curve.empty:
        return None
    try:
        return run_transaction_cost_sensitivity_batch(
            equity_curve,
            sensitivity_config=config.transaction_cost_sensitivity_config,
        )
    except Exception as exc:
        return _transaction_cost_sensitivity_error_result(exc, config)


def _transaction_cost_sensitivity_error_result(
    exc: Exception,
    config: PipelineConfig,
) -> SimpleNamespace:
    sensitivity_config = config.transaction_cost_sensitivity_config
    summary = pd.DataFrame(columns=TRANSACTION_COST_SENSITIVITY_BATCH_COLUMNS)
    summary_metrics = build_transaction_cost_sensitivity_summary_metrics(
        summary,
        sensitivity_config=sensitivity_config,
        execution_mode="error",
    )
    summary_metrics.update(
        {
            "execution_mode": "error",
            "error_count": 1,
            "all_scenarios_evaluable": False,
            "calculation_status": "error",
            "calculation_error_code": exc.__class__.__name__,
            "calculation_error_message": str(exc) or exc.__class__.__name__,
            "error_messages": [str(exc) or exc.__class__.__name__],
        }
    )
    return SimpleNamespace(
        summary=summary,
        equity_curves={},
        config=sensitivity_config,
        batch_id=str(sensitivity_config.config_id),
        execution_mode="error",
        summary_metrics=summary_metrics,
    )


def _render_transaction_cost_sensitivity(sensitivity_result: object | None) -> None:
    st.subheader("Transaction Cost and Turnover Sensitivity")
    if sensitivity_result is None:
        st.info("거래비용/슬리피지/turnover 민감도 결과가 아직 없습니다.")
        return

    summary_metrics = getattr(sensitivity_result, "summary_metrics", {})
    summary = getattr(sensitivity_result, "summary", pd.DataFrame())
    if not isinstance(summary_metrics, dict):
        summary_metrics = {}
    if not isinstance(summary, pd.DataFrame):
        summary = pd.DataFrame(summary)

    error_messages = _transaction_cost_sensitivity_error_messages(summary_metrics)
    if error_messages:
        st.error("민감도 계산 실패: " + " | ".join(error_messages))

    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Sensitivity Scenarios",
        _format_metric_value(summary_metrics.get("scenario_count")),
    )
    metric_cols[1].metric(
        "Warnings",
        _format_metric_value(summary_metrics.get("warning_count")),
    )
    metric_cols[2].metric(
        "Baseline Status",
        _format_status_value(summary_metrics.get("baseline_status")),
    )
    metric_cols[3].metric(
        "Max Cost Loss",
        _format_metric_value(
            summary_metrics.get("max_cost_adjusted_return_loss_vs_baseline"),
            percent=True,
        ),
    )

    review_rows = _transaction_cost_sensitivity_review_rows(summary_metrics)
    if review_rows:
        st.subheader("Sensitivity Result Review Summary")
        st.dataframe(
            pd.DataFrame(review_rows)[TRANSACTION_COST_SENSITIVITY_REVIEW_COLUMNS],
            width="stretch",
            hide_index=True,
        )

    summary_frame = pd.DataFrame([summary_metrics])
    if not summary_frame.empty:
        st.dataframe(
            summary_frame[
                [
                    col
                    for col in TRANSACTION_COST_SENSITIVITY_SUMMARY_COLUMNS
                    if col in summary_frame.columns
                ]
            ],
            width="stretch",
            hide_index=True,
        )

    if summary.empty:
        if error_messages:
            st.info("민감도 시나리오별 성과 표를 생성하지 못했습니다.")
        else:
            st.info("민감도 시나리오별 성과 표가 아직 없습니다.")
        return
    sensitivity_figure = _build_transaction_cost_sensitivity_figure(summary)
    if sensitivity_figure is not None:
        st.subheader("Sensitivity Cost/Turnover Heatmap")
        st.caption(
            "비용+슬리피지 bps와 평균 turnover 예산 조합별 비용 차감 누적수익률을 비교합니다."
        )
        st.plotly_chart(sensitivity_figure, width="stretch")

    display_columns = [
        col
        for col in TRANSACTION_COST_SENSITIVITY_DISPLAY_COLUMNS
        if col in summary.columns
    ]
    if not display_columns:
        display_columns = [
            col
            for col in TRANSACTION_COST_SENSITIVITY_BATCH_COLUMNS
            if col in summary.columns
        ]
    st.dataframe(
        summary[display_columns],
        width="stretch",
        hide_index=True,
    )


def _build_transaction_cost_sensitivity_figure(summary: object) -> go.Figure | None:
    frame = pd.DataFrame(summary).copy()
    required_columns = {
        "scenario_id",
        "total_cost_bps",
        "average_daily_turnover_budget",
        "cost_adjusted_cumulative_return",
    }
    if frame.empty or not required_columns.issubset(frame.columns):
        return None

    plot_frame = frame.loc[:, list(required_columns)].copy()
    for column in (
        "total_cost_bps",
        "average_daily_turnover_budget",
        "cost_adjusted_cumulative_return",
    ):
        plot_frame[column] = pd.to_numeric(plot_frame[column], errors="coerce")
    plot_frame = plot_frame.dropna(
        subset=[
            "total_cost_bps",
            "average_daily_turnover_budget",
            "cost_adjusted_cumulative_return",
        ]
    )
    if plot_frame.empty:
        return None

    grouped = (
        plot_frame.groupby(
            ["average_daily_turnover_budget", "total_cost_bps"],
            as_index=False,
        )
        .agg(
            cost_adjusted_cumulative_return=("cost_adjusted_cumulative_return", "mean"),
            scenario_id=("scenario_id", lambda values: ", ".join(sorted(map(str, values)))),
        )
        .sort_values(["average_daily_turnover_budget", "total_cost_bps"])
    )
    turnover_budgets = sorted(grouped["average_daily_turnover_budget"].unique())
    total_costs = sorted(grouped["total_cost_bps"].unique())
    if not turnover_budgets or not total_costs:
        return None

    matrix = grouped.pivot(
        index="average_daily_turnover_budget",
        columns="total_cost_bps",
        values="cost_adjusted_cumulative_return",
    ).reindex(index=turnover_budgets, columns=total_costs)
    labels = grouped.pivot(
        index="average_daily_turnover_budget",
        columns="total_cost_bps",
        values="scenario_id",
    ).reindex(index=turnover_budgets, columns=total_costs)

    figure = go.Figure(
        data=go.Heatmap(
            x=[f"{cost:.1f}" for cost in total_costs],
            y=[f"{budget:.0%}" for budget in turnover_budgets],
            z=matrix.to_numpy(),
            text=labels.fillna("").to_numpy(),
            colorscale="RdYlGn",
            colorbar={"title": "Cost-adjusted return"},
            hovertemplate=(
                "total cost: %{x} bps<br>"
                "turnover budget: %{y}<br>"
                "cost-adjusted return: %{z:.2%}<br>"
                "scenario: %{text}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        height=320,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis_title="Cost + slippage (bps)",
        yaxis_title="Average daily turnover budget",
    )
    return figure


def _transaction_cost_sensitivity_review_rows(
    summary_metrics: dict[str, object],
) -> list[dict[str, object]]:
    if not summary_metrics:
        return []

    all_evaluable = bool(summary_metrics.get("all_scenarios_evaluable", False))
    turnover_pass = bool(summary_metrics.get("all_turnover_budgets_pass", False))
    max_turnover_pass = bool(summary_metrics.get("all_max_daily_turnover_limits_pass", False))
    error_count = summary_metrics.get("error_count", 0)
    rows = [
        {
            "check": "calculation_errors",
            "status": "pass" if int(error_count or 0) == 0 else "fail",
            "scenario_id": None,
            "value": error_count,
            "threshold": 0,
            "operator": "==",
            "reason": "sensitivity calculation must complete without errors",
        },
        {
            "check": "scenario_evaluability",
            "status": "pass" if all_evaluable else "fail",
            "scenario_id": None,
            "value": summary_metrics.get("insufficient_data_count"),
            "threshold": 0,
            "operator": "==",
            "reason": "all sensitivity scenarios must have enough observations",
        },
        {
            "check": "turnover_budget",
            "status": "pass" if turnover_pass else "warning",
            "scenario_id": None,
            "value": summary_metrics.get("turnover_budget_breach_count"),
            "threshold": 0,
            "operator": "==",
            "reason": "average daily turnover should stay within each scenario budget",
        },
        {
            "check": "max_daily_turnover",
            "status": "pass" if max_turnover_pass else "warning",
            "scenario_id": None,
            "value": summary_metrics.get("max_daily_turnover_breach_count"),
            "threshold": 0,
            "operator": "==",
            "reason": "daily turnover spikes should stay within each scenario limit",
        },
        {
            "check": "worst_cost_adjusted_loss",
            "status": "review",
            "scenario_id": summary_metrics.get("worst_cost_adjusted_scenario_id"),
            "value": summary_metrics.get("max_cost_adjusted_return_loss_vs_baseline"),
            "threshold": None,
            "operator": "report",
            "reason": "largest loss in cost-adjusted return versus canonical costs",
        },
        {
            "check": "largest_total_cost_increase",
            "status": "review",
            "scenario_id": summary_metrics.get("largest_total_cost_scenario_id"),
            "value": summary_metrics.get("max_total_cost_increase_vs_baseline"),
            "threshold": None,
            "operator": "report",
            "reason": "largest total cost increase versus canonical costs",
        },
    ]
    return rows


def _transaction_cost_sensitivity_error_messages(
    summary_metrics: dict[str, object],
) -> list[str]:
    raw_messages = summary_metrics.get("error_messages", [])
    if isinstance(raw_messages, str):
        return [raw_messages]
    if isinstance(raw_messages, list):
        return [str(message) for message in raw_messages if str(message).strip()]
    message = summary_metrics.get("calculation_error_message")
    if message:
        return [str(message)]
    return []


def _render_validity_failure_summary(validity_report: object) -> None:
    failure_rows = _validity_strategy_failure_rows(validity_report)
    if not failure_rows:
        return

    first_failure = failure_rows[0]
    reason = (
        first_failure.get("collapse_reason")
        or first_failure.get("reason")
        or "Stage 1 validity gate failure"
    )
    status = _format_status_value(
        getattr(validity_report, "strategy_candidate_status", first_failure.get("status"))
    )
    official_message = str(getattr(validity_report, "official_message", "") or "")
    if status == "fail" and official_message:
        st.error(official_message)
    else:
        st.error(f"Strategy candidate {status}: {reason}")
    st.subheader("Validity Gate Failure Summary")
    failure_frame = pd.DataFrame(failure_rows)
    st.dataframe(
        failure_frame[
            [col for col in STRATEGY_FAILURE_COLUMNS if col in failure_frame.columns]
        ],
        width="stretch",
        hide_index=True,
    )


def _render_turnover_budget_warnings(validity_report: object) -> None:
    turnover_warning_rows = _validity_turnover_budget_warning_rows(validity_report)
    if not turnover_warning_rows:
        return

    st.subheader("Validity Gate Turnover Budget Warnings")
    for row in turnover_warning_rows:
        message = row.get("message") or row.get("reason") or "Turnover budget warning"
        st.warning(str(message))

    warning_frame = pd.DataFrame(turnover_warning_rows)
    st.dataframe(
        warning_frame[
            [
                col
                for col in TURNOVER_BUDGET_WARNING_COLUMNS
                if col in warning_frame.columns
            ]
        ],
        width="stretch",
        hide_index=True,
    )


def _three_strategy_comparison_frame(validity_report: object) -> pd.DataFrame:
    rows = getattr(validity_report, "cost_adjusted_metric_comparison", [])
    if not isinstance(rows, list | tuple):
        return pd.DataFrame()

    comparison = pd.DataFrame([dict(row) for row in rows if isinstance(row, dict)])
    if comparison.empty or "name" not in comparison.columns:
        return pd.DataFrame()

    expected_order = ["strategy", "SPY", "equal_weight"]
    comparison = comparison[comparison["name"].isin(expected_order)].copy()
    if comparison.empty:
        return pd.DataFrame()

    comparison["strategy"] = pd.Categorical(
        comparison["name"],
        categories=expected_order,
        ordered=True,
    )
    comparison = comparison.sort_values("strategy").drop(columns=["name"])
    comparison["strategy"] = comparison["strategy"].astype(str)
    return comparison[
        [col for col in THREE_STRATEGY_COMPARISON_COLUMNS if col in comparison.columns]
    ]


def main() -> None:
    st.set_page_config(
        page_title="Quant Research",
        page_icon="Q",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Quant Research")

    with st.sidebar:
        defaults = PipelineConfig()
        st.header("Run")
        data_mode = st.selectbox("Data mode", ["synthetic", "live"], index=0)
        start_date = st.date_input("Evaluation start", value=defaults.start)
        end_date = st.date_input("Evaluation end", value=defaults.end)
        st.caption("Default workflow: 20-day US equity research target with SPY benchmark data.")
        tickers_text = st.text_area("Strategy tickers", ", ".join(DEFAULT_TICKERS), height=130)
        sidebar_tickers = [ticker.strip().upper() for ticker in tickers_text.split(",") if ticker.strip()]
        if not sidebar_tickers:
            sidebar_tickers = list(DEFAULT_TICKERS)
        focus_ticker = st.selectbox("Focus ticker", sidebar_tickers, index=0)
        benchmark_options = list(dict.fromkeys([DEFAULT_BENCHMARK_TICKER, defaults.benchmark_ticker, *sidebar_tickers]))
        benchmark_ticker = st.selectbox("Benchmark ticker", benchmark_options, index=0)
        top_n = st.slider("Top N", min_value=1, max_value=8, value=3)
        train_periods = st.slider("Train periods", min_value=30, max_value=300, value=90, step=10)
        test_periods = st.slider("Test periods", min_value=5, max_value=60, value=20, step=5)
        cost_bps = st.slider("Cost bps", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
        slippage_bps = st.slider("Slippage bps", min_value=0.0, max_value=50.0, value=2.0, step=0.5)
        sentiment_model = st.selectbox("Sentiment model", ["keyword", "finbert"], index=1)
        time_series_options = ["proxy", "local"]
        time_series_inference_mode = st.selectbox(
            "Time-series inference",
            time_series_options,
            index=time_series_options.index(defaults.time_series_inference_mode),
            help="Proxy features are diagnostic baselines; local mode uses optional cached Chronos/Granite adapters.",
        )
        max_ts_windows = st.number_input(
            "Local TS inference windows",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
            help="Only used when local time-series inference is selected.",
        )
        filing_extractor_model = st.selectbox("Filing extractor", ["rules", "finma", "fingpt"], index=2)
        enable_local_filing_llm = st.checkbox("Use local filing LLM", value=True)
        fingpt_runtime_options = ("transformers", "mlx", "llama-cpp")
        fingpt_default_runtime = defaults.fingpt_runtime
        if fingpt_default_runtime not in fingpt_runtime_options:
            fingpt_default_runtime = "transformers"
        with st.expander("Local model settings"):
            local_model_device_map = st.text_input("Device map", value="auto")
            local_model_offload_folder = st.text_input("Offload folder", value="artifacts/model_offload")
            chronos_model_id = st.text_input("Chronos-2 model", value="amazon/chronos-2")
            granite_ttm_model_id = st.text_input(
                "Granite TTM model",
                value="ibm-granite/granite-timeseries-ttm-r2",
            )
            granite_ttm_revision = st.text_input("Granite revision", value="")
            finma_model_id = st.text_input("FinMA model", value="ChanceFocus/finma-7b-nlp")
            fingpt_model_id = st.text_input("FinGPT adapter", value="FinGPT/fingpt-mt_llama3-8b_lora")
            fingpt_base_model_id = st.text_input("FinGPT base model", value="meta-llama/Meta-Llama-3-8B")
            fingpt_runtime = st.selectbox(
                "FinGPT runtime",
                options=fingpt_runtime_options,
                index=fingpt_runtime_options.index(fingpt_default_runtime),
                help="Transformers loads only base LoRA flow; MLX/llama-cpp requires a local quantized file path.",
            )
            fingpt_quantized_model_path = st.text_input(
                "FinGPT quantized runtime path",
                value=defaults.fingpt_quantized_model_path,
                help="Warmup and inference are optional heavy steps; keep blank if you do not use quantized runtime.",
            )
            fingpt_allow_unquantized_transformers = st.checkbox(
                "Allow unquantized Transformers 8B load",
                value=defaults.fingpt_allow_unquantized_transformers,
                help=(
                    "Only enable for explicit local experiments; keeps default safety guard off when false."
                ),
            )
            fingpt_single_load_lock_path = st.text_input(
                "FinGPT single-load lock file",
                value=defaults.fingpt_single_load_lock_path or "",
                help="Limits concurrent FinGPT local model load attempts to one at a time.",
            )
        with st.expander("Portfolio risk settings", expanded=True):
            covariance_aware_risk_enabled = st.checkbox(
                "Covariance-aware risk",
                value=defaults.covariance_aware_risk_enabled,
                help=(
                    "When enabled, deterministic scoring and sizing validation use "
                    "recent cross-asset covariance estimates before any signal is shown."
                ),
            )
            covariance_return_options = ["return_1", "return_5", "return_20"]
            covariance_return_default_index = (
                covariance_return_options.index(defaults.covariance_return_column)
                if defaults.covariance_return_column in covariance_return_options
                else 0
            )
            covariance_return_column = st.selectbox(
                "Covariance return column",
                covariance_return_options,
                index=covariance_return_default_index,
            )
            portfolio_covariance_lookback = st.slider(
                "Covariance lookback periods",
                min_value=2,
                max_value=252,
                value=int(defaults.portfolio_covariance_lookback),
                step=1,
            )
            covariance_min_periods = st.slider(
                "Covariance minimum periods",
                min_value=2,
                max_value=int(portfolio_covariance_lookback),
                value=min(
                    int(defaults.covariance_min_periods),
                    int(portfolio_covariance_lookback),
                ),
                step=1,
            )
            max_symbol_weight = st.slider(
                "Max symbol weight",
                min_value=0.05,
                max_value=1.0,
                value=float(defaults.max_symbol_weight),
                step=0.05,
            )
            max_sector_weight = st.slider(
                "Max sector weight",
                min_value=0.05,
                max_value=1.0,
                value=float(defaults.max_sector_weight),
                step=0.05,
            )
            max_position_risk_contribution = st.slider(
                "Max position risk contribution",
                min_value=0.05,
                max_value=1.0,
                value=float(defaults.max_position_risk_contribution),
                step=0.05,
            )
            portfolio_volatility_limit = st.slider(
                "Portfolio volatility limit",
                min_value=0.005,
                max_value=0.10,
                value=float(defaults.portfolio_volatility_limit),
                step=0.005,
            )
            average_daily_turnover_budget = st.slider(
                "Average daily turnover budget",
                min_value=0.05,
                max_value=1.0,
                value=float(defaults.average_daily_turnover_budget),
                step=0.05,
            )
            apply_max_daily_turnover = st.checkbox(
                "Apply max daily turnover limit",
                value=defaults.max_daily_turnover is not None,
            )
            max_daily_turnover_value = 0.50 if defaults.max_daily_turnover is None else float(defaults.max_daily_turnover)
            max_daily_turnover = None
            if apply_max_daily_turnover:
                max_daily_turnover = st.slider(
                    "Max daily turnover",
                    min_value=0.05,
                    max_value=2.0,
                    value=max_daily_turnover_value,
                    step=0.05,
                )
            max_drawdown_stop = st.slider(
                "Max drawdown stop",
                min_value=0.05,
                max_value=0.50,
                value=float(defaults.max_drawdown_stop),
                step=0.05,
            )
            volatility_adjustment_strength = st.slider(
                "Volatility adjustment strength",
                min_value=0.0,
                max_value=1.0,
                value=float(defaults.volatility_adjustment_strength),
                step=0.05,
            )
            concentration_adjustment_strength = st.slider(
                "Concentration adjustment strength",
                min_value=0.0,
                max_value=1.0,
                value=float(defaults.concentration_adjustment_strength),
                step=0.05,
            )
            risk_contribution_adjustment_strength = st.slider(
                "Risk contribution adjustment strength",
                min_value=0.0,
                max_value=1.0,
                value=float(defaults.risk_contribution_adjustment_strength),
                step=0.05,
            )
        enable_feature_model_ablation = st.checkbox("Run model feature ablation", value=False)
        run = st.button("Run research", type="primary", width="stretch")

    if data_mode == "live":
        st.info(
            "Live mode may call yfinance, GDELT, and SEC EDGAR. SEC requests use a User-Agent and local cache. "
            "Synthetic mode is recommended for offline verification."
        )

    tickers = [ticker.strip().upper() for ticker in tickers_text.split(",") if ticker.strip()]
    if not tickers:
        tickers = list(DEFAULT_TICKERS)
    config = PipelineConfig(
        tickers=tickers,
        data_mode=data_mode,
        start=start_date,
        end=end_date,
        train_periods=train_periods,
        test_periods=test_periods,
        top_n=top_n,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
        benchmark_ticker=benchmark_ticker,
        sentiment_model=sentiment_model,
        time_series_inference_mode=time_series_inference_mode,
        max_time_series_inference_windows=int(max_ts_windows),
        chronos_model_id=chronos_model_id,
        granite_ttm_model_id=granite_ttm_model_id,
        granite_ttm_revision=granite_ttm_revision or None,
        local_model_device_map=local_model_device_map,
        local_model_offload_folder=local_model_offload_folder or None,
        filing_extractor_model=filing_extractor_model,
        enable_local_filing_llm=enable_local_filing_llm,
        finma_model_id=finma_model_id,
        fingpt_model_id=fingpt_model_id,
        fingpt_base_model_id=fingpt_base_model_id or None,
        fingpt_runtime=fingpt_runtime,
        fingpt_quantized_model_path=fingpt_quantized_model_path,
        fingpt_allow_unquantized_transformers=fingpt_allow_unquantized_transformers,
        fingpt_single_load_lock_path=fingpt_single_load_lock_path or None,
        max_symbol_weight=max_symbol_weight,
        max_sector_weight=max_sector_weight,
        covariance_aware_risk_enabled=covariance_aware_risk_enabled,
        covariance_return_column=covariance_return_column,
        portfolio_covariance_lookback=int(portfolio_covariance_lookback),
        covariance_min_periods=int(covariance_min_periods),
        average_daily_turnover_budget=average_daily_turnover_budget,
        max_daily_turnover=max_daily_turnover,
        portfolio_volatility_limit=portfolio_volatility_limit,
        max_position_risk_contribution=max_position_risk_contribution,
        max_drawdown_stop=max_drawdown_stop,
        volatility_adjustment_strength=volatility_adjustment_strength,
        concentration_adjustment_strength=concentration_adjustment_strength,
        risk_contribution_adjustment_strength=risk_contribution_adjustment_strength,
        enable_feature_model_ablation=enable_feature_model_ablation,
    )

    if run:
        with st.spinner("Building features, validating models, and running the deterministic signal engine"):
            pipeline_result = run_research_pipeline(config)
            st.session_state["result"] = pipeline_result
            st.session_state["transaction_cost_sensitivity_result"] = (
                pipeline_result.transaction_cost_sensitivity
            )

    result = st.session_state.get("result")
    if result is None:
        st.info("기본 모드에서는 계산이 실행되지 않습니다. 먼저 'Run research' 버튼을 눌러서 결과를 생성하세요.")
        st.caption("선택 항목은 기본값으로 구성되며, 이후 동일 버튼 클릭 시 최신 값으로만 실행됩니다.")
        st.stop()

    metrics = result.backtest.metrics
    dashboard = build_beginner_research_dashboard(result, focus_ticker, config)
    validity_report = build_validity_gate_report(
        result.predictions,
        result.validation_summary,
        result.backtest.equity_curve,
        result.backtest.metrics,
        ablation_summary=result.ablation_summary,
        config=config,
        benchmark_return_series=result.benchmark_return_series,
        equal_weight_baseline_return_series=result.equal_weight_baseline_return_series,
        baseline_comparison_inputs=result.baseline_comparison_inputs or None,
    )
    sensitivity_result = _transaction_cost_sensitivity_result(result, config)
    st.session_state["transaction_cost_sensitivity_result"] = sensitivity_result

    render_beginner_overview(dashboard)

    metric_cols = st.columns(6)
    metric_cols[0].metric("Net CAGR", f"{metrics.net_cagr:.2%}")
    metric_cols[1].metric("Net Sharpe", f"{metrics.sharpe:.2f}")
    metric_cols[2].metric("Net Max DD", f"{metrics.max_drawdown:.2%}")
    metric_cols[3].metric("Hit Rate", f"{metrics.hit_rate:.2%}")
    metric_cols[4].metric("Turnover", f"{metrics.turnover:.2%}")
    metric_cols[5].metric("Exposure", f"{metrics.exposure:.2%}")

    tabs = st.tabs(["Backtest", "Signals", "Features", "Validation", "Data"])

    with tabs[0]:
        st.subheader("Equity Curve")
        equity = result.backtest.equity_curve.set_index("date")
        st.line_chart(equity[["equity", "benchmark_equity"]])
        st.subheader("Portfolio Risk Configuration")
        st.dataframe(
            _portfolio_risk_config_frame(config),
            width="stretch",
            hide_index=True,
        )
        st.subheader("Portfolio Returns")
        st.bar_chart(equity.set_index(equity.index)["portfolio_return"])
        risk_sizing_columns = [
            column
            for column in BACKTEST_RISK_SIZING_COLUMNS
            if column in result.backtest.equity_curve.columns
        ]
        if risk_sizing_columns:
            st.subheader("Covariance Risk and Post-Cost Sizing Validation")
            st.dataframe(
                result.backtest.equity_curve[risk_sizing_columns].tail(60),
                width="stretch",
                hide_index=True,
            )

    with tabs[1]:
        st.subheader("Latest Deterministic Signals")
        latest_date = result.signals["date"].max()
        signal_columns = [
            "date",
            "ticker",
            "action",
            "signal_score",
            "expected_return",
            "predicted_volatility",
            "downside_quantile",
            "text_risk_score",
            "sec_risk_flag",
        ]
        latest = (
            result.signals[result.signals["date"] == latest_date]
            .sort_values("signal_score", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(
            latest[[col for col in signal_columns if col in latest.columns]],
            width="stretch",
            hide_index=True,
        )

    with tabs[2]:
        st.subheader("Feature Fusion Sample")
        st.dataframe(result.features.tail(200), width="stretch", hide_index=True)

    with tabs[3]:
        st.subheader("Validity Gate Status")
        status_cols = st.columns(3)
        status_cols[0].metric("System Validity", validity_report.system_validity_status)
        status_cols[1].metric("Strategy Candidate", validity_report.strategy_candidate_status)
        status_cols[2].metric("Required Horizon", validity_report.required_validation_horizon)
        st.caption(validity_report.official_message)
        _render_user_gate_status(validity_report)
        _render_validity_failure_summary(validity_report)

        final_strategy_status_rows = _validity_final_strategy_status_rows(validity_report)
        if final_strategy_status_rows:
            st.subheader("Final Strategy Status Explanation")
            final_strategy_status_frame = pd.DataFrame(final_strategy_status_rows)
            st.dataframe(
                final_strategy_status_frame[
                    [
                        col
                        for col in FINAL_STRATEGY_STATUS_COLUMNS
                        if col in final_strategy_status_frame.columns
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

        rule_explanation_rows = _validity_rule_explanation_rows(validity_report)
        if rule_explanation_rows:
            st.subheader("Validity Gate Rule Result Explanations")
            rule_explanation_frame = pd.DataFrame(rule_explanation_rows)
            st.dataframe(
                rule_explanation_frame[
                    [
                        col
                        for col in RULE_EXPLANATION_COLUMNS
                        if col in rule_explanation_frame.columns
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

        gate_result_rows = _validity_gate_result_rows(validity_report)
        if gate_result_rows:
            st.subheader("Validity Gate Results")
            gate_result_frame = pd.DataFrame(gate_result_rows)
            st.dataframe(
                gate_result_frame[
                    [col for col in GATE_RESULT_COLUMNS if col in gate_result_frame.columns]
                ],
                width="stretch",
                hide_index=True,
            )

        st.subheader("Canonical Gate Report Artifacts")
        gate_report_payload = _validity_gate_report_payload(validity_report)
        gate_report_artifact_rows = _validity_gate_report_artifact_rows(validity_report)
        gate_report_artifact_frame = pd.DataFrame(gate_report_artifact_rows)
        if gate_report_artifact_frame.empty:
            st.info("canonical Gate report artifact manifest가 아직 없습니다.")
        else:
            st.dataframe(
                gate_report_artifact_frame[
                    [
                        col
                        for col in GATE_REPORT_ARTIFACT_COLUMNS
                        if col in gate_report_artifact_frame.columns
                    ]
                ],
                width="stretch",
                hide_index=True,
            )
            download_cols = st.columns(2)
            download_cols[0].download_button(
                "Download Gate JSON",
                data=json.dumps(gate_report_payload, ensure_ascii=False, indent=2) + "\n",
                file_name="validity_gate.json",
                mime="application/json",
                width="stretch",
            )
            download_cols[1].download_button(
                "Download Gate Markdown",
                data=validity_report.to_markdown() + "\n",
                file_name="validity_report.md",
                mime="text/markdown",
                width="stretch",
            )
        _render_validity_report_approval(validity_report)

        _render_turnover_budget_warnings(validity_report)

        structured_warning_rows = _validity_structured_warning_rows(validity_report)
        if structured_warning_rows:
            st.subheader("Validity Gate Structured Warnings")
            st.warning("Validity Gate warnings require review before trusting the strategy candidate.")
            structured_warning_frame = pd.DataFrame(structured_warning_rows)
            st.dataframe(
                structured_warning_frame[
                    [
                        col
                        for col in STRUCTURED_WARNING_COLUMNS
                        if col in structured_warning_frame.columns
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

        st.subheader("Validity Gate Horizon Diagnostics")
        horizon_rows = _validity_horizon_rows(validity_report)
        _render_horizon_status_summary(horizon_rows)
        horizon_frame = pd.DataFrame(horizon_rows)
        if horizon_frame.empty:
            st.info("기간별 진단 지표가 아직 없습니다.")
        else:
            st.dataframe(
                horizon_frame[
                    [col for col in HORIZON_DIAGNOSTIC_COLUMNS if col in horizon_frame.columns]
                ],
                width="stretch",
                hide_index=True,
            )

        st.subheader("Report-Only Research Metrics")
        report_only_research_rows = _validity_report_only_research_metric_rows(validity_report)
        if not report_only_research_rows:
            st.info("리포트 전용 연구 지표가 아직 없습니다.")
        else:
            st.caption(
                "top_decile_20d_excess_return는 전략 후보 판정, 점수, action, ranking, threshold에 쓰지 않는 report-only 메타데이터입니다."
            )
            report_only_research_frame = pd.DataFrame(report_only_research_rows)
            st.dataframe(
                report_only_research_frame[
                    [
                        col
                        for col in REPORT_ONLY_RESEARCH_METRIC_COLUMNS
                        if col in report_only_research_frame.columns
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

        st.subheader("Validity Gate Cost-Adjusted Comparison")
        comparison = pd.DataFrame(validity_report.cost_adjusted_metric_comparison)
        if comparison.empty:
            st.info("비교 가능한 전략, SPY, equal-weight 기준선 지표가 아직 없습니다.")
        else:
            comparison_columns = [
                "name",
                "role",
                "return_basis",
                "cagr",
                "sharpe",
                "max_drawdown",
                "cost_adjusted_cumulative_return",
                "average_daily_turnover",
                "total_cost_return",
                "excess_return",
                "excess_return_status",
                "strategy_excess_return",
                "evaluation_observations",
                "evaluation_start",
                "evaluation_end",
            ]
            st.dataframe(
                comparison[[col for col in comparison_columns if col in comparison.columns]],
                width="stretch",
                hide_index=True,
            )

        st.subheader("Three Strategy Performance and Cost Comparison")
        three_strategy_comparison = _three_strategy_comparison_frame(validity_report)
        if three_strategy_comparison.empty:
            st.info("전략, SPY, equal-weight 세 결과를 비교할 표준 성과/비용 지표가 아직 없습니다.")
        else:
            st.caption(
                "Deterministic strategy, SPY benchmark, equal-weight universe baseline을 동일 평가 구간에서 비교합니다."
            )
            st.dataframe(
                three_strategy_comparison,
                width="stretch",
                hide_index=True,
            )

        st.subheader("Validity Gate Side-by-Side Metrics")
        side_by_side = pd.DataFrame(validity_report.side_by_side_metric_comparison)
        if side_by_side.empty:
            st.info("전략, 벤치마크, equal-weight 기준선 지표를 나란히 표시할 데이터가 아직 없습니다.")
        else:
            preferred_columns = list(
                dict.fromkeys(
                    [
                        "metric",
                        "metric_label",
                        "strategy",
                        config.benchmark_ticker,
                        "SPY",
                        "equal_weight",
                        *SIDE_BY_SIDE_BASE_COLUMNS,
                    ]
                )
            )
            side_by_side_columns = [
                col for col in preferred_columns if col in side_by_side.columns
            ]
            side_by_side_columns.extend(
                col for col in side_by_side.columns if col not in side_by_side_columns
            )
            st.dataframe(
                side_by_side[side_by_side_columns],
                width="stretch",
                hide_index=True,
            )

        st.subheader("Validity Gate Baseline Comparisons")
        baseline_comparisons = pd.DataFrame(validity_report.baseline_comparison_entries)
        if baseline_comparisons.empty:
            st.info("SPY와 equal-weight 기준선 비교 결과가 아직 없습니다.")
        else:
            baseline_columns = [
                "name",
                "baseline_type",
                "return_basis",
                "cagr",
                "sharpe",
                "max_drawdown",
                "cost_adjusted_cumulative_return",
                "average_daily_turnover",
                "total_cost_return",
                "excess_return",
                "excess_return_status",
                "strategy_excess_return",
                "evaluation_observations",
                "evaluation_start",
                "evaluation_end",
            ]
            st.dataframe(
                baseline_comparisons[[col for col in baseline_columns if col in baseline_comparisons.columns]],
                width="stretch",
                hide_index=True,
            )

        st.subheader("Validity Gate Baseline Inputs")
        baseline_inputs = pd.DataFrame(validity_report.baseline_comparison_inputs)
        if baseline_inputs.empty:
            st.info("Stage 1 기준선 입력 계약이 아직 없습니다.")
        else:
            baseline_input_columns = [
                "name",
                "baseline_type",
                "return_basis",
                "data_source",
                "return_column",
                "return_horizon",
                "required_for_stage1",
                "benchmark_ticker",
                "universe_tickers",
                "cost_bps",
                "slippage_bps",
            ]
            st.dataframe(
                baseline_inputs[[col for col in baseline_input_columns if col in baseline_inputs.columns]],
                width="stretch",
                hide_index=True,
            )

        st.subheader("Validity Gate Comparison Schemas")
        if not validity_report.comparison_input_schema and not validity_report.comparison_result_schema:
            st.info("Stage 1 비교 입력/결과 스키마가 아직 없습니다.")
        else:
            schema_counts = pd.DataFrame(
                [
                    {
                        "schema": "input",
                        "schema_version": validity_report.comparison_input_schema.get(
                            "schema_version"
                        ),
                        "full_model": (
                            validity_report.comparison_input_schema.get(
                                "full_model",
                                {},
                            ).get("entity_id")
                            if isinstance(
                                validity_report.comparison_input_schema.get("full_model"),
                                dict,
                            )
                            else None
                        ),
                        "baselines": len(
                            validity_report.comparison_input_schema.get("baselines", [])
                        ),
                        "ablations": len(
                            validity_report.comparison_input_schema.get("ablations", [])
                        ),
                        "metrics": len(
                            validity_report.comparison_input_schema.get("metrics", [])
                        ),
                        "validation_windows": len(
                            validity_report.comparison_input_schema.get(
                                "validation_windows",
                                [],
                            )
                        ),
                    },
                    {
                        "schema": "result",
                        "schema_version": validity_report.comparison_result_schema.get(
                            "schema_version"
                        ),
                        "full_model": (
                            validity_report.comparison_result_schema.get(
                                "full_model_result",
                                {},
                            ).get("entity_id")
                            if isinstance(
                                validity_report.comparison_result_schema.get(
                                    "full_model_result"
                                ),
                                dict,
                            )
                            else None
                        ),
                        "baselines": len(
                            validity_report.comparison_result_schema.get(
                                "baseline_results",
                                [],
                            )
                        ),
                        "ablations": len(
                            validity_report.comparison_result_schema.get(
                                "ablation_results",
                                [],
                            )
                        ),
                        "metrics": len(
                            validity_report.comparison_result_schema.get(
                                "metric_results",
                                [],
                            )
                        ),
                        "validation_windows": len(
                            validity_report.comparison_result_schema.get(
                                "validation_windows",
                                [],
                            )
                        ),
                    },
                ]
            )
            st.dataframe(schema_counts, width="stretch", hide_index=True)
            with st.expander("Stage 1 comparison input schema"):
                st.json(validity_report.comparison_input_schema)
            with st.expander("Stage 1 comparison result schema"):
                st.json(validity_report.comparison_result_schema)

        st.subheader("Validity Gate Model Comparison Results")
        model_comparisons = pd.DataFrame(validity_report.model_comparison_results)
        if model_comparisons.empty:
            st.info("full model과 기준선/ablation을 비교할 지표가 아직 없습니다.")
        else:
            st.dataframe(
                model_comparisons[
                    [col for col in MODEL_COMPARISON_COLUMNS if col in model_comparisons.columns]
                ],
                width="stretch",
                hide_index=True,
            )

        st.subheader("Validity Gate Metric Contract")
        st.subheader("Full Model Metrics")
        full_model_metric_rows = _validity_full_model_metric_rows(validity_report)
        full_model_metrics = getattr(validity_report, "full_model_metrics", {})
        if full_model_metric_rows:
            full_model_metric_frame = pd.DataFrame(full_model_metric_rows)
            st.dataframe(
                full_model_metric_frame[
                    [col for col in METRIC_CONTRACT_COLUMNS if col in full_model_metric_frame.columns]
                ],
                width="stretch",
                hide_index=True,
            )
            with st.expander("Full model metric contract JSON"):
                st.json(full_model_metrics)
        else:
            st.info("full model metric contract가 아직 없습니다.")

        st.subheader("Baseline Metrics")
        baseline_metric_rows = _validity_metric_contract_rows(
            getattr(validity_report, "baseline_metrics", [])
        )
        if baseline_metric_rows:
            baseline_metric_frame = pd.DataFrame(baseline_metric_rows)
            st.dataframe(
                baseline_metric_frame[
                    [col for col in METRIC_CONTRACT_COLUMNS if col in baseline_metric_frame.columns]
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("baseline metric contract가 아직 없습니다.")

        st.subheader("Ablation Metrics")
        ablation_metric_rows = _validity_metric_contract_rows(
            getattr(validity_report, "ablation_metrics", [])
        )
        if ablation_metric_rows:
            ablation_metric_frame = pd.DataFrame(ablation_metric_rows)
            st.dataframe(
                ablation_metric_frame[
                    [col for col in METRIC_CONTRACT_COLUMNS if col in ablation_metric_frame.columns]
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("ablation metric contract가 아직 없습니다.")

        st.subheader("Structured Pass/Fail Reasons")
        pass_fail_reasons = pd.DataFrame(
            getattr(validity_report, "structured_pass_fail_reasons", [])
        )
        if not pass_fail_reasons.empty:
            st.dataframe(
                pass_fail_reasons[
                    [
                        col
                        for col in STRUCTURED_PASS_FAIL_REASON_COLUMNS
                        if col in pass_fail_reasons.columns
                    ]
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("structured pass/fail reason contract가 아직 없습니다.")

        st.subheader("No-Model-Proxy Ablation")
        no_model_proxy_ablation = validity_report.no_model_proxy_ablation
        if not no_model_proxy_ablation.get("available"):
            st.info("no_model_proxy ablation 결과가 아직 없습니다.")
        else:
            no_model_proxy_controls = no_model_proxy_ablation.get("pipeline_controls", {})
            no_model_proxy_performance = no_model_proxy_ablation.get("performance_metrics", {})
            no_model_proxy_validation = no_model_proxy_ablation.get("validation_metrics", {})
            no_model_proxy_signal = no_model_proxy_ablation.get(
                "deterministic_signal_evaluation_metrics",
                {},
            )
            no_model_proxy_actions = (
                no_model_proxy_signal.get("action_counts", {})
                if isinstance(no_model_proxy_signal, dict)
                else {}
            )
            if not isinstance(no_model_proxy_controls, dict):
                no_model_proxy_controls = {}
            if not isinstance(no_model_proxy_performance, dict):
                no_model_proxy_performance = {}
            if not isinstance(no_model_proxy_validation, dict):
                no_model_proxy_validation = {}
            if not isinstance(no_model_proxy_signal, dict):
                no_model_proxy_signal = {}
            if not isinstance(no_model_proxy_actions, dict):
                no_model_proxy_actions = {}
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "scenario": no_model_proxy_ablation.get("scenario"),
                            "model_proxy_enabled": no_model_proxy_controls.get("model_proxy"),
                            "validation_status": no_model_proxy_validation.get("validation_status"),
                            "validation_fold_count": no_model_proxy_validation.get(
                                "validation_fold_count"
                            ),
                            "validation_oos_fold_count": no_model_proxy_validation.get(
                                "validation_oos_fold_count"
                            ),
                            "sharpe": no_model_proxy_performance.get("sharpe"),
                            "excess_return": no_model_proxy_performance.get("excess_return"),
                            "cost_adjusted_cumulative_return": no_model_proxy_signal.get(
                                "cost_adjusted_cumulative_return"
                            ),
                            "average_daily_turnover": no_model_proxy_signal.get(
                                "average_daily_turnover"
                            ),
                            "buy_count": no_model_proxy_actions.get("BUY", 0),
                            "sell_count": no_model_proxy_actions.get("SELL", 0),
                            "hold_count": no_model_proxy_actions.get("HOLD", 0),
                        }
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

        _render_walk_forward_periods_and_oos_summary(result.validation_summary, config)
        _render_transaction_cost_sensitivity(sensitivity_result)

        st.subheader("Walk-Forward Summary")
        st.dataframe(result.validation_summary, width="stretch", hide_index=True)
        if result.benchmark_inputs is not None:
            st.subheader("Benchmark Construction Inputs")
            st.json(result.benchmark_inputs.to_dict())
        if result.benchmark_return_series is not None and not result.benchmark_return_series.empty:
            benchmark_label = result.benchmark_return_series["benchmark_ticker"].iloc[0]
            st.subheader(f"{benchmark_label} Baseline Return Series")
            st.dataframe(result.benchmark_return_series.tail(100), width="stretch", hide_index=True)
        if (
            result.equal_weight_baseline_return_series is not None
            and not result.equal_weight_baseline_return_series.empty
        ):
            st.subheader("Equal-Weight Baseline Return Series")
            st.dataframe(
                result.equal_weight_baseline_return_series.tail(100),
                width="stretch",
                hide_index=True,
            )
        if "is_oos" in result.validation_summary.columns:
            oos_summary = result.validation_summary[result.validation_summary["is_oos"]]
        else:
            oos_summary = pd.DataFrame()
        if not oos_summary.empty:
            st.subheader("Out-of-Sample Holdout")
            st.dataframe(oos_summary, width="stretch", hide_index=True)
        st.subheader("Ablation Summary")
        ablation_frame = pd.DataFrame(result.ablation_summary)
        if not ablation_frame.empty and "pipeline_controls" in ablation_frame:
            for control in ("model_proxy", "cost", "slippage", "turnover"):
                ablation_frame[f"control_{control}"] = ablation_frame["pipeline_controls"].map(
                    lambda value, key=control: value.get(key) if isinstance(value, dict) else None
                )
        st.dataframe(ablation_frame, width="stretch", hide_index=True)

    with tabs[4]:
        st.subheader("Raw Market Sample")
        st.dataframe(result.market_data.tail(200), width="stretch", hide_index=True)
        st.subheader("News Feature Sample")
        st.dataframe(result.news_features.tail(100), width="stretch", hide_index=True)
        st.subheader("SEC Feature Sample")
        st.dataframe(result.sec_features.tail(100), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
