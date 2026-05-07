from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.data.timestamps import (
    date_end_utc,
    timestamp_utc,
    validate_generated_feature_cutoffs,
)
from quant_research.validation.manifest import collect_git_version_info
from quant_research.validation.report_renderer import (
    render_structured_report,
    write_structured_report_artifact,
)
from quant_research.validation.report_schema import (
    ARTIFACT_MANIFEST_REQUIRED_METADATA_FIELDS,
    ARTIFACT_MANIFEST_SCHEMA_ID,
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    CANONICAL_REPORT_REQUIRED_RESULT_SECTIONS,
    CanonicalReportMetadata,
    build_artifact_manifest_schema,
)

COMPLETED_RUN_REPORT_SCHEMA_ID = "stage1_completed_validation_backtest_report"
COMPLETED_RUN_REPORT_SCHEMA_VERSION = "completed_validation_backtest_report.v1"
COMPLETED_RUN_REPORT_ARTIFACTS = (
    "canonical_run_report.json",
    "canonical_run_report.md",
    "canonical_run_report.html",
)


def build_completed_validation_backtest_report(
    *,
    metadata: CanonicalReportMetadata | Mapping[str, object],
    deterministic_signal_outputs: pd.DataFrame,
    backtest_results: pd.DataFrame,
    performance_metrics: object,
    walk_forward_validation_metrics: pd.DataFrame,
    risk_evaluation_metrics: pd.DataFrame | None = None,
    system_validity_gate: object | Mapping[str, object] | None = None,
    strategy_candidate_gate: object | Mapping[str, object] | None = None,
    artifact_manifest: Mapping[str, object] | None = None,
    report_path: str | None = None,
) -> dict[str, object]:
    """Build a human-reviewable report payload from completed validation artifacts.

    The report consumes deterministic signal outputs and validation/backtest metrics.
    It deliberately does not accept raw model prediction sections as report inputs.
    """

    metadata_payload = _metadata_payload(metadata)
    signals = _coerce_frame(deterministic_signal_outputs, "deterministic_signal_outputs")
    equity = _coerce_frame(backtest_results, "backtest_results")
    validation = _coerce_frame(
        walk_forward_validation_metrics,
        "walk_forward_validation_metrics",
    )
    _validate_report_signal_timing(signals)
    risks = (
        _coerce_frame(risk_evaluation_metrics, "risk_evaluation_metrics")
        if risk_evaluation_metrics is not None
        else _risk_rows_from_backtest(equity, performance_metrics)
    )
    system_gate_payload = _gate_payload(system_validity_gate)
    strategy_gate_payload = _gate_payload(strategy_candidate_gate or system_validity_gate)
    manifest_payload = _artifact_manifest_payload(
        artifact_manifest,
        metadata_payload=metadata_payload,
        report_path=report_path,
        system_validity_gate=system_gate_payload,
        strategy_candidate_gate=strategy_gate_payload,
    )
    payload: dict[str, object] = {
        **metadata_payload,
        "schema_id": COMPLETED_RUN_REPORT_SCHEMA_ID,
        "schema_version": COMPLETED_RUN_REPORT_SCHEMA_VERSION,
        "metadata_schema_id": metadata_payload.get("schema_id"),
        "metadata_schema_version": metadata_payload.get("schema_version"),
        "report_path": report_path,
        "validation_period_metadata": _validation_period_metadata(validation, equity),
        "deterministic_signal_summary": _signal_summary(signals),
        "performance_metrics": _performance_metrics_section(performance_metrics, equity),
        "risk_metrics": _risk_metrics_section(risks, equity, performance_metrics),
        "walk_forward_validation_metrics": _walk_forward_summary(validation),
        "system_validity_gate": system_gate_payload,
        "strategy_candidate_gate": strategy_gate_payload,
        "artifact_manifest": manifest_payload,
    }
    return _json_safe(payload)


def _validate_report_signal_timing(signals: pd.DataFrame) -> None:
    """Fail fast if report inputs contain timestamps unavailable at signal date."""

    if signals.empty:
        return
    validate_generated_feature_cutoffs(
        signals,
        date_column="date",
        label="deterministic signal report inputs",
    )
    sample_timestamp = date_end_utc(signals["date"])
    for column in _signal_prediction_cutoff_columns(signals):
        prediction_time = timestamp_utc(signals[column])
        violation = prediction_time.notna() & sample_timestamp.notna() & (
            prediction_time > sample_timestamp
        )
        if violation.any():
            first_index = signals.index[violation][0]
            raise ValueError(
                f"deterministic signal report inputs column {column} is later than "
                f"signal date {signals.loc[first_index, 'date']}"
            )


def _signal_prediction_cutoff_columns(signals: pd.DataFrame) -> list[str]:
    return [
        column
        for column in ("prediction_date", "prediction_timestamp", "model_prediction_timestamp")
        if column in signals
    ]


def write_completed_validation_backtest_report_artifacts(
    report_payload: Mapping[str, object],
    output_dir: str | Path,
    *,
    include_json_appendix: bool = False,
) -> dict[str, str]:
    """Persist completed-run report JSON, Markdown, and HTML artifacts."""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    report = dict(report_payload)
    report["report_path"] = str(path / "canonical_run_report.md")

    json_path = path / "canonical_run_report.json"
    markdown_path = path / "canonical_run_report.md"
    html_path = path / "canonical_run_report.html"
    json_path.write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_structured_report_artifact(
        report,
        markdown_path,
        output_format="markdown",
        include_json_appendix=include_json_appendix,
    )
    write_structured_report_artifact(
        report,
        html_path,
        output_format="html",
        include_json_appendix=include_json_appendix,
    )
    return {
        "json": str(json_path),
        "markdown": str(markdown_path),
        "html": str(html_path),
        "json_sha256": _file_sha256(json_path),
        "markdown_sha256": _file_sha256(markdown_path),
        "html_sha256": _file_sha256(html_path),
    }


def render_completed_validation_backtest_report(
    report_payload: Mapping[str, object],
    *,
    output_format: str = "markdown",
    include_json_appendix: bool = False,
) -> str:
    return render_structured_report(
        report_payload,
        output_format=output_format,  # type: ignore[arg-type]
        include_json_appendix=include_json_appendix,
    )


def _metadata_payload(metadata: CanonicalReportMetadata | Mapping[str, object]) -> dict[str, object]:
    if isinstance(metadata, Mapping):
        return dict(metadata)
    if isinstance(metadata, CanonicalReportMetadata):
        return metadata.to_dict()
    to_dict = getattr(metadata, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    raise TypeError("metadata must be a CanonicalReportMetadata or mapping")


def _coerce_frame(frame: pd.DataFrame | None, name: str) -> pd.DataFrame:
    if frame is None:
        raise ValueError(f"{name} is required")
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    return frame.copy()


def _signal_summary(signals: pd.DataFrame) -> dict[str, object]:
    if signals.empty:
        return {
            "row_count": 0,
            "unique_dates": 0,
            "unique_tickers": 0,
            "action_counts": [],
            "latest_signal_date": None,
            "latest_action_counts": [],
            "average_signal_score": None,
            "signal_engine": None,
            "llm_makes_trading_decisions": False,
            "model_predictions_are_order_signals": False,
        }
    output = signals.copy()
    if "date" in output:
        output["date"] = pd.to_datetime(output["date"], errors="coerce").dt.normalize()
    action_counts = _value_count_rows(output, "action", "action")
    latest_signal_date = _max_date(output.get("date"))
    latest_action_counts: list[dict[str, object]] = []
    if latest_signal_date is not None and "date" in output and "action" in output:
        latest = output[output["date"] == pd.Timestamp(latest_signal_date)]
        latest_action_counts = _value_count_rows(latest, "action", "action")
    return {
        "row_count": int(len(output)),
        "unique_dates": _nunique(output, "date"),
        "unique_tickers": _nunique(output, "ticker"),
        "action_counts": action_counts,
        "latest_signal_date": latest_signal_date,
        "latest_action_counts": latest_action_counts,
        "average_signal_score": _mean(output, "signal_score"),
        "median_signal_score": _median(output, "signal_score"),
        "average_model_confidence": _mean(output, "model_confidence"),
        "average_risk_metric_penalty": _mean(output, "risk_metric_penalty"),
        "signal_engine": _first_non_null(output, "signal_engine"),
        "llm_makes_trading_decisions": False,
        "model_predictions_are_order_signals": False,
    }


def _performance_metrics_section(metrics: object, equity: pd.DataFrame) -> dict[str, object]:
    metric_payload = _object_to_mapping(metrics)
    section = {
        "return_basis": metric_payload.get("return_basis", "cost_adjusted_return"),
        "net_cagr": _metric_value(metric_payload, "net_cagr", "cagr"),
        "gross_cagr": _metric_value(metric_payload, "gross_cagr"),
        "benchmark_cagr": _metric_value(
            metric_payload,
            "benchmark_cost_adjusted_cagr",
            "benchmark_cagr",
        ),
        "net_cumulative_return": _metric_value(
            metric_payload,
            "net_cumulative_return",
            "cost_adjusted_cumulative_return",
        ),
        "gross_cumulative_return": _metric_value(metric_payload, "gross_cumulative_return"),
        "benchmark_cumulative_return": _metric_value(
            metric_payload,
            "benchmark_cost_adjusted_cumulative_return",
        ),
        "excess_return": _metric_value(metric_payload, "excess_return"),
        "annualized_volatility": _metric_value(metric_payload, "annualized_volatility"),
        "sharpe": _metric_value(metric_payload, "sharpe"),
        "hit_rate": _metric_value(metric_payload, "hit_rate"),
        "total_cost_return": _metric_value(metric_payload, "total_cost_return"),
        "transaction_cost_return": _metric_value(metric_payload, "transaction_cost_return"),
        "slippage_cost_return": _metric_value(metric_payload, "slippage_cost_return"),
        "observations": int(len(equity)),
    }
    if not equity.empty:
        section["evaluation_start"] = _min_date(equity.get("date"))
        section["evaluation_end"] = _max_date(equity.get("date"))
    return section


def _risk_metrics_section(
    risk_rows: pd.DataFrame,
    equity: pd.DataFrame,
    metrics: object,
) -> dict[str, object]:
    metric_payload = _object_to_mapping(metrics)
    section = {
        "max_drawdown": _metric_value(metric_payload, "max_drawdown"),
        "average_daily_turnover": _metric_value(metric_payload, "turnover"),
        "average_portfolio_volatility_estimate": _metric_value(
            metric_payload,
            "average_portfolio_volatility_estimate",
        ),
        "max_portfolio_volatility_estimate": _metric_value(
            metric_payload,
            "max_portfolio_volatility_estimate",
        ),
        "max_symbol_weight": _metric_value(metric_payload, "max_position_weight"),
        "max_sector_weight": _metric_value(metric_payload, "max_sector_exposure"),
        "max_position_risk_contribution": _metric_value(
            metric_payload,
            "max_position_risk_contribution",
        ),
        "position_sizing_validation_status": metric_payload.get(
            "position_sizing_validation_status"
        ),
        "position_sizing_validation_pass_rate": _metric_value(
            metric_payload,
            "position_sizing_validation_pass_rate",
        ),
        "risk_rows": _frame_records(risk_rows, limit=50),
    }
    if not equity.empty:
        section.update(
            {
                "max_observed_holdings": _max_numeric(
                    equity,
                    "position_count",
                    fallback_column="holdings_count",
                ),
                "max_observed_symbol_weight": _max_numeric(
                    equity,
                    "max_position_weight",
                    fallback_column="max_symbol_weight",
                ),
                "max_observed_sector_weight": _max_numeric(
                    equity,
                    "max_sector_exposure",
                    fallback_column="max_sector_weight",
                ),
                "risk_stop_triggered": _any_bool(equity, "risk_stop_active"),
            }
        )
    return section


def _walk_forward_summary(validation: pd.DataFrame) -> dict[str, object]:
    if validation.empty:
        return {
            "fold_count": 0,
            "oos_fold_count": 0,
            "positive_fold_ratio": None,
            "folds": [],
        }
    output = validation.copy()
    for column in ("train_start", "train_end", "test_start", "test_end"):
        if column in output:
            output[column] = pd.to_datetime(output[column], errors="coerce")
    oos = output[output["is_oos"].astype(bool)] if "is_oos" in output else output
    rank_column = "fold_rank_ic" if "fold_rank_ic" in output else None
    if rank_column is None and "rank_ic" in output:
        rank_column = "rank_ic"
    positive_ratio = None
    mean_rank_ic = None
    if rank_column is not None and not oos.empty:
        rank_values = pd.to_numeric(oos[rank_column], errors="coerce").dropna()
        if not rank_values.empty:
            positive_ratio = float((rank_values > 0).mean())
            mean_rank_ic = float(rank_values.mean())
    return {
        "fold_count": int(len(output)),
        "oos_fold_count": int(len(oos)),
        "positive_fold_ratio": positive_ratio,
        "mean_rank_ic": mean_rank_ic,
        "target_column": _first_non_null(output, "target_column"),
        "prediction_horizon_periods": _first_non_null(output, "prediction_horizon_periods"),
        "purge_periods": _first_non_null(output, "purge_periods"),
        "embargo_periods": _first_non_null(output, "embargo_periods"),
        "folds": _frame_records(output, limit=100),
    }


def _validation_period_metadata(validation: pd.DataFrame, equity: pd.DataFrame) -> dict[str, object]:
    summary = _walk_forward_summary(validation)
    return {
        "train_start": _min_date(validation.get("train_start")),
        "train_end": _max_date(validation.get("train_end")),
        "test_start": _min_date(validation.get("test_start")),
        "test_end": _max_date(validation.get("test_end")),
        "evaluation_start": _min_date(equity.get("date")),
        "evaluation_end": _max_date(equity.get("date")),
        "fold_count": summary["fold_count"],
        "oos_fold_count": summary["oos_fold_count"],
        "target_column": summary["target_column"],
        "prediction_horizon_periods": summary["prediction_horizon_periods"],
        "purge_periods": summary["purge_periods"],
        "embargo_periods": summary["embargo_periods"],
        "non_overlapping_or_horizon_consistent": True,
    }


def _risk_rows_from_backtest(equity: pd.DataFrame, metrics: object) -> pd.DataFrame:
    metric_payload = _object_to_mapping(metrics)
    if equity.empty:
        date_value = None
    else:
        date_value = _max_date(equity.get("date"))
    return pd.DataFrame(
        [
            {
                "date": date_value,
                "risk_check": "average_daily_turnover",
                "status": _threshold_status(metric_payload.get("turnover"), 0.25, "<="),
                "observed_value": _finite_or_none(metric_payload.get("turnover")),
                "threshold_value": 0.25,
            },
            {
                "date": date_value,
                "risk_check": "max_drawdown",
                "status": _threshold_status(metric_payload.get("max_drawdown"), -0.20, ">="),
                "observed_value": _finite_or_none(metric_payload.get("max_drawdown")),
                "threshold_value": -0.20,
            },
            {
                "date": date_value,
                "risk_check": "max_symbol_weight",
                "status": _threshold_status(
                    metric_payload.get("max_position_weight"),
                    0.10,
                    "<=",
                ),
                "observed_value": _finite_or_none(metric_payload.get("max_position_weight")),
                "threshold_value": 0.10,
            },
            {
                "date": date_value,
                "risk_check": "max_sector_weight",
                "status": _threshold_status(
                    metric_payload.get("max_sector_exposure"),
                    0.30,
                    "<=",
                ),
                "observed_value": _finite_or_none(metric_payload.get("max_sector_exposure")),
                "threshold_value": 0.30,
            },
        ]
    )


def _gate_payload(gate: object | Mapping[str, object] | None) -> dict[str, object]:
    if gate is None:
        return {}
    payload = _object_to_mapping(gate)
    if not payload:
        return {}
    return {
        "system_validity_status": payload.get("system_validity_status"),
        "strategy_candidate_status": payload.get("strategy_candidate_status"),
        "system_validity_pass": payload.get("system_validity_pass"),
        "strategy_pass": payload.get("strategy_pass"),
        "hard_fail": payload.get("hard_fail"),
        "warning": payload.get("warning"),
        "official_message": payload.get("official_message"),
        "gate_failure_reasons": payload.get("gate_failure_reasons", []),
        "metrics": payload.get("metrics", {}),
    }


def _artifact_manifest_payload(
    manifest: Mapping[str, object] | None,
    *,
    metadata_payload: Mapping[str, object],
    report_path: str | None,
    system_validity_gate: Mapping[str, object] | None = None,
    strategy_candidate_gate: Mapping[str, object] | None = None,
) -> dict[str, object]:
    base = dict(manifest or {})
    existing_artifacts = base.get("artifacts", [])
    artifacts = (
        list(existing_artifacts)
        if isinstance(existing_artifacts, Sequence) and not isinstance(existing_artifacts, str)
        else []
    )
    for artifact in COMPLETED_RUN_REPORT_ARTIFACTS:
        artifacts.append(
            {
                "artifact_id": artifact.rsplit(".", maxsplit=1)[0],
                "artifact_type": artifact.rsplit(".", maxsplit=1)[-1],
                "path": str(Path(report_path).with_name(artifact)) if report_path else artifact,
            }
        )
    identity = _object_to_mapping(metadata_payload.get("identity"))
    universe = _object_to_mapping(metadata_payload.get("universe"))
    data_provenance = _object_to_mapping(metadata_payload.get("data_provenance"))
    system_gate = _object_to_mapping(system_validity_gate or base.get("system_validity_gate"))
    strategy_gate = _object_to_mapping(
        strategy_candidate_gate or base.get("strategy_candidate_gate")
    )
    experiment_id = str(identity.get("experiment_id") or "unknown_experiment")
    run_id = str(identity.get("run_id") or "unknown_run")
    schema = build_artifact_manifest_schema()
    metadata_schema_id = (
        metadata_payload.get("metadata_schema_id") or metadata_payload.get("schema_id")
    )
    metadata_schema_version = (
        metadata_payload.get("metadata_schema_version") or metadata_payload.get("schema_version")
    )
    base.update(
        {
            "schema_id": base.get("schema_id", ARTIFACT_MANIFEST_SCHEMA_ID),
            "schema_version": base.get("schema_version", ARTIFACT_MANIFEST_SCHEMA_VERSION),
            "manifest_schema": base.get("manifest_schema", schema),
            "required_metadata_fields": base.get(
                "required_metadata_fields",
                list(ARTIFACT_MANIFEST_REQUIRED_METADATA_FIELDS),
            ),
            "manifest_id": base.get(
                "manifest_id",
                f"{experiment_id}:{run_id}:artifact_manifest",
            ),
            "experiment_id": base.get("experiment_id", experiment_id),
            "run_id": base.get("run_id", run_id),
            "created_at": base.get("created_at", datetime.now(UTC).isoformat()),
            "artifact_root": base.get("artifact_root", metadata_payload.get("artifact_root", "artifacts")),
            "report_artifact_root": base.get(
                "report_artifact_root",
                metadata_payload.get("report_artifact_root", "reports"),
            ),
            "metadata_schema_id": metadata_schema_id,
            "metadata_schema_version": metadata_schema_version,
            "config_hash": base.get(
                "config_hash",
                _stable_payload_hash(metadata_payload.get("run_configuration")),
            ),
            "universe_snapshot_hash": base.get(
                "universe_snapshot_hash",
                _stable_payload_hash(universe.get("universe_snapshot")),
            ),
            "feature_availability_cutoff_hash": base.get(
                "feature_availability_cutoff_hash",
                _stable_payload_hash(data_provenance.get("feature_availability_cutoff")),
            ),
            "data_snapshot_hash": base.get(
                "data_snapshot_hash",
                _stable_payload_hash(
                    {
                        "data_sources": data_provenance.get("data_sources"),
                        "period": metadata_payload.get("period"),
                    }
                ),
            ),
            "system_validity_status": base.get(
                "system_validity_status",
                system_gate.get("system_validity_status", "not_evaluated"),
            ),
            "strategy_candidate_status": base.get(
                "strategy_candidate_status",
                strategy_gate.get("strategy_candidate_status", "not_evaluated"),
            ),
            "survivorship_bias_allowed": base.get(
                "survivorship_bias_allowed",
                universe.get("survivorship_bias_allowed"),
            ),
            "survivorship_bias_disclosure": base.get(
                "survivorship_bias_disclosure",
                universe.get("survivorship_bias_disclosure"),
            ),
            "v1_scope_exclusions": base.get(
                "v1_scope_exclusions",
                metadata_payload.get("v1_scope_exclusions", []),
            ),
            "git_version": base.get("git_version", collect_git_version_info()),
            "report_path": report_path,
            "artifacts": artifacts,
            "reproducible_input_metadata_required": True,
            "result_sections_required": base.get(
                "result_sections_required",
                list(CANONICAL_REPORT_REQUIRED_RESULT_SECTIONS),
            ),
        }
    )
    return base


def _stable_payload_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _object_to_mapping(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    return {}


def _value_count_rows(frame: pd.DataFrame, column: str, label: str) -> list[dict[str, object]]:
    if column not in frame:
        return []
    counts = frame[column].fillna("UNKNOWN").astype(str).value_counts().sort_index()
    return [{label: key, "count": int(value)} for key, value in counts.items()]


def _frame_records(frame: pd.DataFrame, *, limit: int) -> list[dict[str, object]]:
    if frame.empty:
        return []
    return _json_safe(frame.head(limit).to_dict("records"))  # type: ignore[return-value]


def _metric_value(mapping: Mapping[str, object], *keys: str) -> object:
    for key in keys:
        if key in mapping:
            return _finite_or_none(mapping[key])
    return None


def _finite_or_none(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (int, str, bool)):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    return numeric if np.isfinite(numeric) else None


def _threshold_status(value: object, threshold: float, operator: str) -> str:
    observed = _finite_or_none(value)
    if not isinstance(observed, float | int):
        return "WARN"
    if operator == "<=":
        return "PASS" if float(observed) <= threshold else "FAIL"
    if operator == ">=":
        return "PASS" if float(observed) >= threshold else "FAIL"
    return "WARN"


def _nunique(frame: pd.DataFrame, column: str) -> int:
    return int(frame[column].nunique()) if column in frame else 0


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _median(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.median()) if not values.empty else None


def _first_non_null(frame: pd.DataFrame, column: str) -> object:
    if column not in frame:
        return None
    values = frame[column].dropna()
    if values.empty:
        return None
    return _json_safe(values.iloc[0])


def _max_numeric(
    frame: pd.DataFrame,
    column: str,
    *,
    fallback_column: str | None = None,
) -> float | None:
    resolved = column if column in frame else fallback_column
    if resolved is None or resolved not in frame:
        return None
    values = pd.to_numeric(frame[resolved], errors="coerce").dropna()
    return float(values.max()) if not values.empty else None


def _any_bool(frame: pd.DataFrame, column: str) -> bool:
    if column not in frame:
        return False
    return bool(frame[column].fillna(False).astype(bool).any())


def _min_date(series: object) -> str | None:
    if series is None:
        return None
    values = pd.to_datetime(series, errors="coerce").dropna()
    if values.empty:
        return None
    return pd.Timestamp(values.min()).date().isoformat()


def _max_date(series: object) -> str | None:
    if series is None:
        return None
    values = pd.to_datetime(series, errors="coerce").dropna()
    if values.empty:
        return None
    return pd.Timestamp(values.max()).date().isoformat()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_safe(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_json_safe(item) for item in value]
    return value
