from __future__ import annotations

import json
import re
from fnmatch import fnmatch
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "docs" / "semantic-safety-evidence-manifest.json"
BROKER_ORDER_POLICY_LANGUAGE = re.compile(
    r"\b("
    r"broker|brokers|order|orders|place_order|submit_order|create_order|"
    r"market_order|limit_order|send_order|live_trade|execution"
    r")\b"
    r"|execute trade|place trade|submit trade|send order|"
    r"매수 주문|매도 주문|주문 전송",
    re.IGNORECASE,
)
ALLOWLIST_CLASSIFICATIONS = {"guardrail", "audit", "policy", "fixture", "documentation"}


def _manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text())


def _policy_language_hits() -> list[tuple[str, str, str]]:
    hits: list[tuple[str, str, str]] = []
    searched_roots = (REPOSITORY_ROOT / "docs", REPOSITORY_ROOT / "tests")
    searched_suffixes = {".csv", ".json", ".md", ".py", ".txt", ".yaml", ".yml"}

    for root in searched_roots:
        for path in root.rglob("*"):
            if path.suffix not in searched_suffixes:
                continue
            relative = path.relative_to(REPOSITORY_ROOT).as_posix()
            lines = path.read_text().splitlines()
            for index, line in enumerate(lines):
                if not BROKER_ORDER_POLICY_LANGUAGE.search(line):
                    continue
                start = max(0, index - 16)
                end = min(len(lines), index + 17)
                context = "\n".join(lines[start:end])
                hits.append((relative, line.strip(), context))
    return hits


def _entry_matches_hit(entry: dict[str, Any], path: str, line: str, context: str) -> bool:
    entry_path = str(entry["path"])
    path_matches = path == entry_path or fnmatch(path, entry_path)
    if not path_matches:
        return False
    if not re.search(str(entry["term"]), line, re.IGNORECASE):
        return False
    pattern = str(entry.get("line_or_pattern") or entry.get("context_excerpt") or "")
    if str(entry.get("line_number", "")).startswith("generated:"):
        return entry.get("generated_audit_report", {}).get("expands_to_exact_occurrences") is True
    return bool(pattern and re.search(pattern, context, re.IGNORECASE))


def test_manifest_maps_every_acceptance_criterion_to_verifiable_evidence() -> None:
    manifest = _manifest()
    records = manifest["evidence_manifest"]

    assert [record["ac_id"] for record in records] == list(range(1, 16))
    for record in records:
        for required_field in (
            "invariant",
            "independent_artifact_ids",
            "test_names",
            "commands",
            "expected_results",
            "static_references",
            "anti_self_reference_check",
            "reviewer_notes",
        ):
            assert record[required_field], f"AC {record['ac_id']} missing {required_field}"

        assert all(
            "pytest" in command or command.startswith("rg ")
            for command in record["commands"]
        )
        assert all(
            isinstance(name, str) and ("::" in name or name.endswith(".py"))
            for name in record["test_names"]
        )
        assert all(
            reference != "docs/semantic-safety-evidence-manifest.json"
            for reference in record["static_references"]
        ), f"AC {record['ac_id']} is self-referential only"
        assert any(
            artifact["external_to_manifest"] is True
            for artifact in record["independent_artifact_ids"]
        )
        assert "pass" in " ".join(record["expected_results"]).lower() or "fail" in " ".join(
            record["expected_results"]
        ).lower()


def test_manifest_documents_production_surface_boundaries() -> None:
    manifest = _manifest()
    surface = manifest["production_surface"]

    assert surface["included_executable_paths"] == [
        "app.py",
        "scripts/*.py",
        "src/quant_research/**/*.py",
    ]
    assert "src/quant_research/signals/**/*.py" in surface["src_quant_research_production_executable_paths"]
    assert "src/quant_research/validation/**/*.py" in surface["src_quant_research_production_executable_paths"]
    assert "src/quant_research/backtest/**/*.py" in surface["src_quant_research_production_executable_paths"]

    excluded = {entry["path"]: entry for entry in surface["excluded_non_production_paths"]}
    for path, classification in {
        "docs/**": "documentation",
        "tests/**": "guardrail",
        "tests/fixtures/**": "fixture",
        "reports/**": "report",
        ".pytest_cache/**": "generated_outputs",
    }.items():
        assert excluded[path]["classification"] == classification
        assert excluded[path]["rationale"]

    assert "No broker/order implementation exceptions" in surface["exception_policy"]
    assert manifest["production_enforced"]["behavior_change_expected"] is False


def test_broker_order_allowlist_entries_are_complete_and_cover_policy_hits() -> None:
    manifest = _manifest()
    allowlist = manifest["broker_order_term_allowlist"]

    assert allowlist
    for entry in allowlist:
        for required_field in (
            "term",
            "path",
            "stable_occurrence_id",
            "line_number",
            "matched_text",
            "context_excerpt",
            "allowed_context",
            "reason",
            "owner_invariant",
            "classification",
        ):
            assert entry[required_field], f"allowlist entry missing {required_field}: {entry}"
        assert entry["classification"] in ALLOWLIST_CLASSIFICATIONS
        if str(entry["line_number"]).startswith("generated:"):
            assert entry["generated_audit_report"]["test_name"]
            assert entry["generated_audit_report"]["expands_to_exact_occurrences"] is True

    misses = []
    for path, line, context in _policy_language_hits():
        if path == "docs/semantic-safety-evidence-manifest.json":
            continue
        if not any(_entry_matches_hit(entry, path, line, context) for entry in allowlist):
            misses.append(f"{path}: {line}")

    assert not misses, "\n".join(misses)


def test_manifest_defines_final_label_provenance_and_report_only_metric_isolation() -> None:
    manifest = _manifest()
    provenance = manifest["final_label_provenance"]
    report_metric = manifest["report_only_metrics"][0]
    label_inventory = manifest["label_context_inventory"]

    assert provenance["source_module"] == "src/quant_research/signals/engine.py"
    assert provenance["deterministic_engine_function"] == "DeterministicSignalEngine.generate"
    assert provenance["allowed_final_labels"] == ["BUY", "SELL", "HOLD"]
    assert any("features/text.py" in source for source in provenance["prohibited_upstream_sources"])
    assert any("test_signal_engine_does_not_pass_through" in test for test in provenance["tests_proving_provenance"])

    assert report_metric["metric_name"] == "top_decile_20d_excess_return"
    assert any("calculate_top_decile_20d_excess_return" in producer for producer in report_metric["allowed_producers"])
    assert "src/quant_research/signals/**/*.py" in report_metric["forbidden_consumers"]
    assert any("does_not_change_gate_decisions" in test for test in report_metric["tests_or_audits"])
    assert {entry["classification"] for entry in label_inventory} >= {
        "final_signal",
        "fixture_expectation",
        "report_label",
        "diagnostic",
        "audit_language",
        "documentation",
    }


def test_manifest_records_validation_non_impact_and_warning_policy() -> None:
    manifest = _manifest()
    validation_impact = manifest["validation_impact"]["baseline_comparison"]
    warning_policy = manifest["warning_policy"]

    for field in (
        "signal_outputs",
        "model_predictions",
        "backtest_metrics",
        "report_metrics",
        "validation_gates",
    ):
        assert validation_impact[field], field

    assert manifest["validation_impact"]["behavior_change_expected"] is False
    warning = warning_policy["warnings"][0]
    assert warning["warning_class"] == "FutureWarning"
    assert warning["source_path"] == ".venv/lib/python3.12/site-packages/sklearn/pipeline.py:61"
    assert warning_policy["triggering_command"] == "python3 -m pytest"
    assert warning["accepted_count"] == 2
    assert warning_policy["drift_trigger"]
    assert warning_policy["warning_reduction_changed_semantics"] is False
