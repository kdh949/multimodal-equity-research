from __future__ import annotations

import html
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal

ReportRenderFormat = Literal["markdown", "html"]

CANONICAL_REPORT_SECTION_ORDER: tuple[str, ...] = (
    "identity",
    "universe",
    "period",
    "run_configuration",
    "data_provenance",
    "result_schema_sections",
    "artifact_manifest",
    "v1_scope_exclusions",
)


def render_structured_report_markdown(
    report_data: object,
    *,
    title: str | None = None,
    include_json_appendix: bool = False,
) -> str:
    """Render canonical structured report data as operator-readable Markdown."""

    payload = _coerce_mapping(report_data)
    report_title = title or _default_title(payload)
    lines = [f"# {_markdown_text(report_title)}", ""]
    lines.extend(_markdown_table("Report Summary", _summary_rows(payload)))

    for section_id in _ordered_sections(payload):
        section = payload.get(section_id)
        if section is None:
            continue
        lines.extend(_render_markdown_section(section_id, section))

    if include_json_appendix:
        lines.extend(
            [
                "## Structured JSON Appendix",
                "",
                "```json",
                json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def render_structured_report_html(
    report_data: object,
    *,
    title: str | None = None,
    include_json_appendix: bool = False,
) -> str:
    """Render canonical structured report data as standalone HTML."""

    payload = _coerce_mapping(report_data)
    report_title = title or _default_title(payload)
    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{_html_text(report_title)}</title>",
        "</head>",
        "<body>",
        f"<h1>{_html_text(report_title)}</h1>",
    ]
    lines.extend(_html_table("Report Summary", ("Field", "Value"), _summary_rows(payload)))

    for section_id in _ordered_sections(payload):
        section = payload.get(section_id)
        if section is None:
            continue
        lines.extend(_render_html_section(section_id, section))

    if include_json_appendix:
        lines.extend(
            [
                "<section>",
                "<h2>Structured JSON Appendix</h2>",
                f"<pre>{_html_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2))}</pre>",
                "</section>",
            ]
        )
    lines.extend(["</body>", "</html>"])
    return "\n".join(lines)


def render_structured_report(
    report_data: object,
    *,
    output_format: ReportRenderFormat = "markdown",
    title: str | None = None,
    include_json_appendix: bool = False,
) -> str:
    if output_format == "markdown":
        return render_structured_report_markdown(
            report_data,
            title=title,
            include_json_appendix=include_json_appendix,
        )
    if output_format == "html":
        return render_structured_report_html(
            report_data,
            title=title,
            include_json_appendix=include_json_appendix,
        )
    raise ValueError("output_format must be markdown or html")


def write_structured_report_artifact(
    report_data: object,
    output_path: str | Path,
    *,
    output_format: ReportRenderFormat | None = None,
    title: str | None = None,
    include_json_appendix: bool = False,
) -> Path:
    path = Path(output_path)
    resolved_format = output_format or _format_from_suffix(path)
    rendered = render_structured_report(
        report_data,
        output_format=resolved_format,
        title=title,
        include_json_appendix=include_json_appendix,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered + "\n", encoding="utf-8")
    return path


def _render_markdown_section(section_id: str, section: object) -> list[str]:
    title = _section_title(section_id)
    if section_id == "result_schema_sections" and isinstance(section, Mapping):
        return _render_result_schema_sections_markdown(section)
    if section_id == "v1_scope_exclusions" and isinstance(section, Sequence):
        return [f"## {title}", "", *[f"- {_markdown_cell(item)}" for item in section], ""]
    if section_id == "artifact_manifest" and isinstance(section, Mapping):
        return _render_artifact_manifest_markdown(section)
    if isinstance(section, Mapping):
        lines = _markdown_table(title, _scalar_mapping_rows(section))
        lines.extend(_render_nested_markdown_tables(section))
        return lines
    if _is_table(section):
        return _markdown_table(title, _rows_from_sequence(section))
    return [f"## {title}", "", _markdown_cell(section), ""]


def _render_html_section(section_id: str, section: object) -> list[str]:
    title = _section_title(section_id)
    if section_id == "result_schema_sections" and isinstance(section, Mapping):
        return _render_result_schema_sections_html(section)
    if section_id == "v1_scope_exclusions" and isinstance(section, Sequence):
        return _html_list(title, section)
    if section_id == "artifact_manifest" and isinstance(section, Mapping):
        return _render_artifact_manifest_html(section)
    if isinstance(section, Mapping):
        lines = _html_table(title, ("Field", "Value"), _scalar_mapping_rows(section))
        lines.extend(_render_nested_html_tables(section))
        return lines
    if _is_table(section):
        rows = _rows_from_sequence(section)
        headers = _table_headers(rows)
        return _html_table(title, headers, (_row_values(row, headers) for row in rows))
    return ["<section>", f"<h2>{_html_text(title)}</h2>", f"<p>{_html_text(section)}</p>", "</section>"]


def _render_result_schema_sections_markdown(section: Mapping[str, object]) -> list[str]:
    rows = []
    for section_name, raw_schema in section.items():
        schema = raw_schema if isinstance(raw_schema, Mapping) else {}
        rows.append(
            {
                "Section": section_name,
                "Row Grain": schema.get("row_grain", ""),
                "Sample Alignment Key": _format_value(schema.get("sample_alignment_key")),
                "Required Fields": _format_value(schema.get("required_fields")),
                "Optional Fields": _format_value(schema.get("optional_fields")),
            }
        )
    return _markdown_table("Result Schema Sections", rows)


def _render_result_schema_sections_html(section: Mapping[str, object]) -> list[str]:
    rows = []
    for section_name, raw_schema in section.items():
        schema = raw_schema if isinstance(raw_schema, Mapping) else {}
        rows.append(
            {
                "Section": section_name,
                "Row Grain": schema.get("row_grain", ""),
                "Sample Alignment Key": _format_value(schema.get("sample_alignment_key")),
                "Required Fields": _format_value(schema.get("required_fields")),
                "Optional Fields": _format_value(schema.get("optional_fields")),
            }
        )
    headers = _table_headers(rows)
    return _html_table(
        "Result Schema Sections",
        headers,
        (_row_values(row, headers) for row in rows),
    )


def _render_artifact_manifest_markdown(section: Mapping[str, object]) -> list[str]:
    lines = _markdown_table("Artifact Manifest", _scalar_mapping_rows(section))
    artifacts = section.get("artifacts")
    if _is_table(artifacts):
        lines.extend(_markdown_table("Artifact Files", _rows_from_sequence(artifacts)))
    result_sections = section.get("result_sections_required")
    if isinstance(result_sections, Sequence) and not isinstance(result_sections, str):
        lines.extend(["### Required Result Sections", ""])
        lines.extend(f"- {_markdown_cell(item)}" for item in result_sections)
        lines.append("")
    return lines


def _render_artifact_manifest_html(section: Mapping[str, object]) -> list[str]:
    lines = _html_table("Artifact Manifest", ("Field", "Value"), _scalar_mapping_rows(section))
    artifacts = section.get("artifacts")
    if _is_table(artifacts):
        rows = _rows_from_sequence(artifacts)
        headers = _table_headers(rows)
        lines.extend(_html_table("Artifact Files", headers, (_row_values(row, headers) for row in rows)))
    result_sections = section.get("result_sections_required")
    if isinstance(result_sections, Sequence) and not isinstance(result_sections, str):
        lines.extend(_html_list("Required Result Sections", result_sections))
    return lines


def _render_nested_markdown_tables(section: Mapping[str, object]) -> list[str]:
    lines: list[str] = []
    for key, value in section.items():
        if isinstance(value, Mapping):
            lines.extend(_markdown_table(_section_title(str(key)), _scalar_mapping_rows(value), level=3))
        elif _is_table(value):
            lines.extend(_markdown_table(_section_title(str(key)), _rows_from_sequence(value), level=3))
    return lines


def _render_nested_html_tables(section: Mapping[str, object]) -> list[str]:
    lines: list[str] = []
    for key, value in section.items():
        if isinstance(value, Mapping):
            lines.extend(_html_table(_section_title(str(key)), ("Field", "Value"), _scalar_mapping_rows(value)))
        elif _is_table(value):
            rows = _rows_from_sequence(value)
            headers = _table_headers(rows)
            lines.extend(_html_table(_section_title(str(key)), headers, (_row_values(row, headers) for row in rows)))
    return lines


def _markdown_table(title: str, rows: Sequence[Mapping[str, object]], *, level: int = 2) -> list[str]:
    if not rows:
        return []
    headers = _table_headers(rows)
    lines = [f"{'#' * level} {_markdown_text(title)}", ""]
    lines.append("| " + " | ".join(_markdown_cell(header) for header in headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_markdown_cell(_format_value(row.get(header))) for header in headers)
            + " |"
        )
    lines.append("")
    return lines


def _html_table(
    title: str,
    headers: Iterable[object],
    rows: Iterable[Iterable[object] | Mapping[str, object]],
) -> list[str]:
    header_tuple = tuple(str(header) for header in headers)
    row_values = [
        _row_values(row, header_tuple) if isinstance(row, Mapping) else tuple(row)
        for row in rows
    ]
    if not row_values:
        return []
    lines = ["<section>", f"<h2>{_html_text(title)}</h2>", "<table>", "<thead>", "<tr>"]
    lines.extend(f"<th>{_html_text(header)}</th>" for header in header_tuple)
    lines.extend(["</tr>", "</thead>", "<tbody>"])
    for row in row_values:
        lines.append("<tr>")
        lines.extend(f"<td>{_html_text(_format_value(value))}</td>" for value in row)
        lines.append("</tr>")
    lines.extend(["</tbody>", "</table>", "</section>"])
    return lines


def _html_list(title: str, items: Iterable[object]) -> list[str]:
    item_list = [item for item in items if not isinstance(item, Mapping)]
    if not item_list:
        return []
    lines = ["<section>", f"<h2>{_html_text(title)}</h2>", "<ul>"]
    lines.extend(f"<li>{_html_text(_format_value(item))}</li>" for item in item_list)
    lines.extend(["</ul>", "</section>"])
    return lines


def _summary_rows(payload: Mapping[str, object]) -> list[dict[str, object]]:
    identity = _mapping(payload.get("identity"))
    universe = _mapping(payload.get("universe"))
    snapshot = _mapping(universe.get("universe_snapshot"))
    period = _mapping(payload.get("period"))
    config = _mapping(payload.get("run_configuration"))
    rows = [
        ("Schema ID", payload.get("schema_id")),
        ("Schema Version", payload.get("schema_version")),
        ("Experiment ID", identity.get("experiment_id")),
        ("Run ID", identity.get("run_id")),
        ("Report Type", identity.get("report_type")),
        ("Created At", identity.get("created_at")),
        ("Period", _date_range(period.get("start_date"), period.get("end_date"))),
        ("Universe Count", snapshot.get("selection_count") or _sequence_length(universe.get("universe"))),
        ("Target Horizon", config.get("target_horizon")),
        ("System Validity", payload.get("system_validity_status")),
        ("Strategy Candidate", payload.get("strategy_candidate_status")),
        ("Report Path", payload.get("report_path")),
    ]
    return [{"Field": label, "Value": value} for label, value in rows if value not in (None, "")]


def _scalar_mapping_rows(mapping: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    for key, value in mapping.items():
        if isinstance(value, Mapping) or _is_table(value):
            continue
        rows.append({"Field": _section_title(str(key)), "Value": _format_value(value)})
    return rows


def _rows_from_sequence(value: object) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    rows: list[dict[str, object]] = []
    for item in value:
        if isinstance(item, Mapping):
            rows.append({str(key): item_value for key, item_value in item.items()})
        else:
            rows.append({"Value": item})
    return rows


def _table_headers(rows: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(str(key))
    return tuple(headers)


def _row_values(row: Iterable[object] | Mapping[str, object], headers: Sequence[str]) -> tuple[object, ...]:
    if isinstance(row, Mapping):
        return tuple(row.get(header, "") for header in headers)
    return tuple(row)


def _ordered_sections(payload: Mapping[str, object]) -> list[str]:
    ordered = [section for section in CANONICAL_REPORT_SECTION_ORDER if section in payload]
    ordered.extend(
        section
        for section in payload
        if section not in ordered
        and section
        not in {
            "schema_id",
            "schema_version",
            "required_metadata_sections",
            "required_result_sections",
        }
    )
    return ordered


def _coerce_mapping(report_data: object) -> Mapping[str, object]:
    if isinstance(report_data, Mapping):
        return report_data
    to_dict = getattr(report_data, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return payload
    raise TypeError("report_data must be a mapping or expose to_dict()")


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _is_table(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str)
        and bool(value)
        and all(not isinstance(item, (str, bytes)) for item in value)
    )


def _sequence_length(value: object) -> int:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return len(value)
    return 0


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, Mapping):
        return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
    if isinstance(value, Sequence) and not isinstance(value, str):
        return ", ".join(_format_value(item) for item in value)
    return str(value)


def _date_range(start: object, end: object) -> str:
    if start and end:
        return f"{start} to {end}"
    return ""


def _default_title(payload: Mapping[str, object]) -> str:
    identity = _mapping(payload.get("identity"))
    report_type = identity.get("report_type") or payload.get("schema_id")
    if report_type:
        return _section_title(str(report_type))
    return "Canonical Experiment Report"


def _section_title(section_id: str) -> str:
    return section_id.replace("_", " ").replace("-", " ").title()


def _format_from_suffix(path: Path) -> ReportRenderFormat:
    suffix = path.suffix.lower()
    if suffix in {".html", ".htm"}:
        return "html"
    if suffix in {".md", ".markdown", ""}:
        return "markdown"
    raise ValueError("output_path suffix must be .md, .markdown, .html, or .htm")


def _markdown_text(value: object) -> str:
    return str(value).replace("\n", " ").strip()


def _markdown_cell(value: object) -> str:
    return _markdown_text(value).replace("|", "\\|")


def _html_text(value: object) -> str:
    return html.escape(str(value), quote=True)


def _json_safe(value: object) -> object:
    try:
        json.dumps(value)
    except TypeError:
        if isinstance(value, Mapping):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, str):
            return [_json_safe(item) for item in value]
        return str(value)
    return value
