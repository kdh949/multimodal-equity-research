from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

import pandas as pd
import streamlit as st


def render_validity_gate(report: object) -> None:
    """
    Render a reusable Streamlit block for a Validity Gate report.

    The input may be either a dataclass-like object or a dict-like object.
    Missing fields are shown as graceful placeholders.
    """

    normalized = normalize_validity_gate_report(report)

    st.subheader("Validity Gate")
    if normalized["official_message"]:
        st.info(normalized["official_message"])

    status_columns = st.columns(3)
    _render_status_badge(
        status_columns[0],
        "System Validity",
        normalized["system_validity_status"],
    )
    _render_status_badge(
        status_columns[1],
        "Strategy Candidate",
        normalized["strategy_candidate_status"],
    )
    _render_status_badge(
        status_columns[2],
        "Hard Fail",
        "FAIL" if normalized["hard_fail"] else "PASS",
    )

    warnings = normalized["warnings"]
    if warnings:
        st.markdown("#### Warnings")
        for warning in warnings:
            st.warning(str(warning))

    metrics = normalized["metrics"]
    if not metrics.empty:
        st.markdown("#### Gate Metrics")
        st.dataframe(metrics, width="stretch", hide_index=True)

    evidence = normalized["evidence"]
    if not evidence.empty:
        st.markdown("#### Evidence")
        st.dataframe(evidence, width="stretch", hide_index=True)

    benchmark_results = normalized["benchmark_results"]
    if not benchmark_results.empty:
        st.markdown("#### Benchmark Results")
        st.dataframe(benchmark_results, width="stretch", hide_index=True)

    ablation_results = normalized["ablation_results"]
    if not ablation_results.empty:
        st.markdown("#### Ablation Results")
        st.dataframe(ablation_results, width="stretch", hide_index=True)


def normalize_validity_gate_report(report: object) -> dict[str, Any]:
    """Normalize a validity report object into a Streamlit-ready dict."""

    report_mapping = _to_mapping(report)
    normalized: dict[str, Any] = {
        "system_validity_status": _coerce_status(
            _get_field(
                report_mapping,
                "system_validity_status",
                ("system_status", "system_state", "system_validity"),
            )
        ),
        "strategy_candidate_status": _coerce_status(
            _get_field(
                report_mapping,
                "strategy_candidate_status",
                ("strategy_status", "strategy_state", "candidate_status"),
            )
        ),
        "hard_fail": _coerce_bool(
            _get_field(
                report_mapping,
                "hard_fail",
                ("is_hard_fail", "blocked", "is_blocked"),
            )
        ),
        "warnings": _coerce_warning_list(
            _get_field(report_mapping, "warnings", ("warning", "alerts", "alerts_list"))
        ),
        "metrics": _build_key_value_frame(
            _get_field(
                report_mapping,
                "metrics",
                ("metric", "gate_metrics", "score_card"),
            ),
            key_label="Metric",
        ),
        "evidence": _build_key_value_frame(
            _get_field(
                report_mapping,
                "evidence",
                ("evidence_summary", "supporting_evidence"),
            ),
            key_label="Evidence",
        ),
        "official_message": _coerce_text(
            _get_field(
                report_mapping,
                "official_message",
                ("message", "summary", "note"),
            )
        ),
        "benchmark_results": _to_result_frame(
            _get_field(
                report_mapping,
                "benchmark_results",
                ("benchmarks", "benchmark"),
            )
        ),
        "ablation_results": _to_result_frame(
            _get_field(
                report_mapping,
                "ablation_results",
                ("ablation", "ablation_study"),
            )
        ),
    }
    return normalized


def _render_status_badge(column: Any, title: str, status: str) -> None:
    tone = _status_tone(status)
    label = _status_label(status)
    column.metric(title, label)
    column.markdown(f"{tone} {label}")


def _status_label(status: str) -> str:
    status_text = status.upper()
    return status_text


def _status_tone(status: str) -> str:
    normalized = status.upper()
    if normalized in {"PASS", "OK", "READY", "CLEARED", "GOOD", "TRUE"}:
        return ":green[PASS]"
    if normalized in {"WARN", "WARNING", "CAUTION", "INSUFFICIENT_DATA", "NOT_EVALUABLE"}:
        return ":orange[WARN]"
    if normalized in {"FAIL", "HARD_FAIL", "NO", "BLOCK", "BLOCKED", "ERROR", "BAD", "FALSE"}:
        return ":red[FAIL]"
    return ":blue[UNKNOWN]"


def _to_mapping(report: object) -> dict[str, Any]:
    if isinstance(report, Mapping):
        return dict(report)
    if is_dataclass(report):
        return asdict(report)
    if hasattr(report, "__dict__"):
        raw = report.__dict__
        if isinstance(raw, dict):
            return dict(raw)
    slots = getattr(report, "__slots__", ())
    if slots:
        if isinstance(slots, str):
            slots = (slots,)
        return {slot: getattr(report, slot) for slot in slots if hasattr(report, slot)}
    as_dict = getattr(report, "_asdict", None)
    if callable(as_dict):
        data = as_dict()
        if isinstance(data, Mapping):
            return dict(data)
    return {}


def _get_field(
    source: dict[str, Any],
    key: str,
    aliases: tuple[str, ...] = (),
) -> Any:
    if key in source:
        return source[key]
    for alias in aliases:
        if alias in source:
            return source[alias]
    return None


def _coerce_status(value: Any, *, default: str = "UNKNOWN") -> str:
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if value is None:
        return default
    if isinstance(value, (int, float)):
        if value == 1:
            return "PASS"
        if value == 0:
            return "FAIL"
    text = str(value).strip()
    if not text:
        return default
    text_upper = text.upper()
    if text_upper in {"PASS", "PASSED", "PASSING", "OK", "READY", "TRUE", "SUCCESS"}:
        return "PASS"
    if text_upper in {"FAIL", "FAILED", "HARD_FAIL", "BLOCKED", "BLOCK", "NO", "FALSE", "ERROR"}:
        return "FAIL"
    if text_upper in {"WARN", "WARNING", "CAUTION"}:
        return "WARN"
    return text_upper


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        return text in {"true", "1", "yes", "y", "on", "fail", "failed", "blocked"}
    return False


def _coerce_warning_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        trimmed = value.strip()
        return [trimmed] if trimmed else []
    if isinstance(value, (list, tuple, set)):
        warnings: list[str] = []
        for item in value:
            warnings.extend(_coerce_warning_list(item))
        return warnings
    return [_coerce_text(value)]


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _build_key_value_frame(value: Any, *, key_label: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.reset_index(drop=True)
    if isinstance(value, Mapping):
        rows = _flatten_to_rows(value)
        return pd.DataFrame(rows, columns=[key_label, "Value"])
    return pd.DataFrame(columns=[key_label, "Value"])

def _to_result_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.reset_index(drop=True)
    if isinstance(value, Mapping):
        return _build_key_value_frame(value, key_label="Key")
    if isinstance(value, (list, tuple, set)):
        if not value:
            return pd.DataFrame()
        if all(isinstance(item, Mapping) for item in value):
            return pd.DataFrame(list(value))
        return pd.DataFrame({"value": [ _format_scalar(item) for item in value ]})
    return pd.DataFrame()


def _flatten_to_rows(
    value: Mapping[str, Any],
    *,
    parent: str = "",
) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for key, item in value.items():
        label = str(key) if not parent else f"{parent}.{key}"
        if isinstance(item, Mapping):
            rows.extend(_flatten_to_rows(item, parent=label))
        elif isinstance(item, (list, tuple)):
            for index, nested in enumerate(item):
                nested_label = f"{label}[{index}]"
                if isinstance(nested, Mapping):
                    rows.extend(_flatten_to_rows(nested, parent=nested_label))
                else:
                    rows.append((nested_label, _format_scalar(nested)))
        else:
            rows.append((label, _format_scalar(item)))
    return rows


def _format_scalar(value: Any) -> str:
    if isinstance(value, float):
        if abs(value) < 1 and "rate" not in str(value):
            return f"{value:.4f}"
        return f"{value:.4f}"
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().isoformat()
    return str(value)
