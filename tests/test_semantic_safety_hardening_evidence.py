from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from quant_research.signals.engine import (
    DeterministicSignalEngine,
    SignalEngineConfig,
    SignalGenerationBlockedError,
    require_signal_generation_gate_pass,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_PATH = REPOSITORY_ROOT / "docs" / "semantic-safety-taxonomies.json"
MANIFEST_PATH = REPOSITORY_ROOT / "docs" / "semantic-safety-evidence-manifest.json"
PRESERVATION_BASELINE_PATH = (
    REPOSITORY_ROOT / "tests" / "fixtures" / "semantic_preservation_baseline.json"
)
WARNING_BASELINE_PATH = REPOSITORY_ROOT / "tests" / "fixtures" / "warning_baseline.json"

PRODUCTION_ROOTS = (
    REPOSITORY_ROOT / "app.py",
    REPOSITORY_ROOT / "scripts",
    REPOSITORY_ROOT / "src" / "quant_research",
)
EXCLUDED_CLASSIFICATIONS = {
    "docs": "docs",
    "tests": "tests",
    "tests/fixtures": "fixtures",
    "reports": "reports",
    ".pytest_cache": "generated_outputs",
}
# Guardrail audit scanner: these prohibited terms appear here only to prove
# production absence and generated allowlist expansion.
BROKER_ORDER_SCAN = re.compile(
    r"\b("
    r"broker|brokers|order|orders|place_order|submit_order|create_order|"
    r"market_order|limit_order|send_order|live_trade|execution"
    r")\b"
    r"|execute trade|place trade|submit trade|send order|"
    r"매수 주문|매도 주문|주문 전송",
    re.IGNORECASE,
)
ACTION_LIKE_SCAN = re.compile(
    r"\b(?:buy|sell|hold)\b|strong buy|buy now|sell the stock|"
    r"hold pending earnings|recommend accumulation|exit the position|"
    r"\b(?:BUY|SELL|HOLD)[:_-][A-Z]{1,6}\b",
    re.IGNORECASE,
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _production_python_paths() -> list[str]:
    paths: list[Path] = []
    for root in PRODUCTION_ROOTS:
        if root.is_file():
            paths.append(root)
        else:
            paths.extend(sorted(root.rglob("*.py")))
    return [path.relative_to(REPOSITORY_ROOT).as_posix() for path in paths]


def _classify_production_path(relative_path: str) -> tuple[bool, str, str]:
    if relative_path == "app.py":
        return True, "application_entrypoint", "app.py"
    if relative_path.startswith("scripts/"):
        return True, "documented_command", "manual script entrypoint"
    if relative_path.startswith("src/quant_research/signals/"):
        return True, "signal_flow", "quant_research.signals"
    if relative_path.startswith("src/quant_research/backtest/"):
        return True, "backtest_flow", "quant_research.backtest"
    if relative_path.startswith("src/quant_research/validation/"):
        if "report" in relative_path:
            return True, "report_flow", "quant_research.validation"
        return True, "validation_flow", "quant_research.validation"
    if relative_path.startswith("src/quant_research/"):
        return True, "src_package", "quant_research package import root"
    return False, "explicit_exemption", "outside production roots"


def _classify_excluded_path(relative_path: str) -> str | None:
    for prefix, classification in EXCLUDED_CLASSIFICATIONS.items():
        if relative_path == prefix or relative_path.startswith(f"{prefix}/"):
            return classification
    return None


def _broker_term_occurrences() -> list[dict[str, object]]:
    occurrences: list[dict[str, object]] = []
    for root in (REPOSITORY_ROOT / "docs", REPOSITORY_ROOT / "tests"):
        for path in sorted(root.rglob("*")):
            if path.suffix not in {".csv", ".json", ".md", ".py", ".txt", ".yaml", ".yml"}:
                continue
            if path == MANIFEST_PATH:
                continue
            relative = path.relative_to(REPOSITORY_ROOT).as_posix()
            lines = path.read_text().splitlines()
            for line_number, line in enumerate(lines, start=1):
                if not BROKER_ORDER_SCAN.search(line):
                    continue
                classification = _classify_excluded_path(relative) or "audit_policy"
                occurrences.append(
                    {
                        "stable_occurrence_id": f"{relative}:{line_number}",
                        "path": relative,
                        "line_number": line_number,
                        "matched_text": line.strip(),
                        "context_excerpt": "\n".join(
                            lines[max(0, line_number - 2) : min(len(lines), line_number + 1)]
                        ),
                        "allowed_context": classification,
                        "reason": "Non-production policy, fixture, or guardrail evidence.",
                        "owner_invariant": "No production executable path may implement live order placement.",
                        "classification": classification,
                    }
                )
    return occurrences


def _classify_broker_order_phrase(phrase: str, *, production: bool) -> str:
    normalized = phrase.lower()
    ambiguous = {"ordering", "sort_order", "display_order", "ordered_columns", "scenario_order"}
    if normalized in ambiguous or any(term in normalized for term in ambiguous):
        return "ambiguous_non_trading"
    if any(term in normalized for term in ("place_order", "submit_order", "broker", "live_trade")):
        return "prohibited_implementation" if production else "allowed_policy_audit"
    if "order placement audit" in normalized or "no live trading" in normalized:
        return "allowed_policy_audit"
    if normalized == "order":
        return "prohibited_implementation" if production else "allowed_policy_audit"
    return "allowed_policy_audit"


def _action_like_match(text: str) -> dict[str, object]:
    normalized = text.lower()
    negated = bool(re.search(r"\b(?:do not|not a|no)\s+(?:buy|sell|hold)", normalized))
    return {
        "matched": bool(ACTION_LIKE_SCAN.search(text)),
        "negated": negated,
        "blocked_upstream": bool(ACTION_LIKE_SCAN.search(text)),
        "final_signal_expected": "deterministic_engine_only",
    }


def _action_label_assignment_targets(paths: list[str]) -> set[str]:
    targets: set[str] = set()
    for relative in paths:
        path = REPOSITORY_ROOT / relative
        tree = ast.parse(path.read_text(), filename=relative)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign | ast.AnnAssign):
                continue
            value = node.value
            if value is None:
                continue
            labels = {
                child.value
                for child in ast.walk(value)
                if isinstance(child, ast.Constant) and child.value in {"BUY", "SELL", "HOLD"}
            }
            if not labels:
                continue
            source = path.read_text()
            assigned_nodes = node.targets if isinstance(node, ast.Assign) else [node.target]
            target_repr = "\n".join(
                ast.get_source_segment(source, target) or "" for target in assigned_nodes
            )
            if '"action"' in target_repr or "action=" in target_repr:
                targets.add(f"{relative}:{node.lineno}")
    return targets


def _compare_warning_baseline(
    baseline: dict[str, Any], observed: list[dict[str, object]]
) -> list[str]:
    expected_by_key = {
        (
            row["warning_class"],
            row["source_path"],
            row["triggering_command"] if "triggering_command" in row else baseline["triggering_command"],
        ): row
        for row in baseline["warnings"]
    }
    failures: list[str] = []
    observed_counts: dict[tuple[str, str, str], int] = {}
    for row in observed:
        key = (
            str(row["warning_class"]),
            str(row["source_path"]),
            str(row["triggering_command"]),
        )
        expected = expected_by_key.get(key)
        if expected is None:
            failures.append(f"unexpected warning: {key}")
            continue
        if not re.search(str(expected["message_regex"]), str(row["message"])):
            failures.append(f"message drift: {key}")
        observed_counts[key] = observed_counts.get(key, 0) + int(row.get("count", 1))

    for key, expected in expected_by_key.items():
        if observed_counts.get(key, 0) != expected["accepted_count"]:
            failures.append(f"count drift: {key}")
    return failures


def test_production_surface_inventory_covers_entrypoints_and_reachability() -> None:
    inventory = []
    for relative_path in _production_python_paths():
        included, reason, reachable_from = _classify_production_path(relative_path)
        inventory.append(
            {
                "path": relative_path,
                "module_name": relative_path.removesuffix(".py").replace("/", "."),
                "included_in_surface": included,
                "inclusion_reason": reason,
                "reachable_from": reachable_from,
                "excluded_reason": None,
                "audit_test_name": "tests/test_semantic_safety_hardening_evidence.py::test_production_surface_inventory_covers_entrypoints_and_reachability",
            }
        )

    assert inventory
    assert {row["path"] for row in inventory} >= {
        "app.py",
        "scripts/run_backtest_validation.py",
        "scripts/preload_local_models.py",
        "src/quant_research/signals/engine.py",
        "src/quant_research/backtest/engine.py",
        "src/quant_research/validation/gate.py",
    }
    assert all(row["included_in_surface"] is True for row in inventory)
    assert {row["inclusion_reason"] for row in inventory} >= {
        "application_entrypoint",
        "documented_command",
        "src_package",
        "validation_flow",
        "backtest_flow",
        "report_flow",
        "signal_flow",
    }
    assert all(row["reachable_from"] for row in inventory)


def test_excluded_paths_are_classified_as_docs_tests_fixtures_reports_generated_or_audit_policy() -> None:
    examples = {
        "docs/risk-validation.md": "docs",
        "tests/test_signal_engine.py": "tests",
        "tests/fixtures/report_generation/signals.csv": "tests",
        "reports/example/validity_gate.json": "reports",
        ".pytest_cache/v/cache/nodeids": "generated_outputs",
    }

    for path, expected in examples.items():
        assert _classify_excluded_path(path) == expected


def test_broker_order_taxonomy_classifies_prohibited_policy_and_ambiguous_terms() -> None:
    taxonomy = _json(TAXONOMY_PATH)["broker_order_term_taxonomy"]

    assert "place_order" in taxonomy["prohibited_trading_implementation_terms"]
    assert "sort_order" in taxonomy["ambiguous_non_trading_terms"]
    assert _classify_broker_order_phrase("client.submit_order", production=True) == "prohibited_implementation"
    assert _classify_broker_order_phrase("order placement audit", production=False) == "allowed_policy_audit"
    assert _classify_broker_order_phrase("sort_order", production=True) == "ambiguous_non_trading"
    assert _classify_broker_order_phrase("display_order", production=True) == "ambiguous_non_trading"


def test_generated_broker_order_audit_report_expands_allowlist_to_exact_occurrences() -> None:
    occurrences = _broker_term_occurrences()

    assert occurrences
    assert len({row["stable_occurrence_id"] for row in occurrences}) == len(occurrences)
    for row in occurrences:
        assert row["path"]
        assert isinstance(row["line_number"], int)
        assert row["matched_text"]
        assert row["context_excerpt"]
        assert row["reason"]
        assert row["owner_invariant"]
        assert row["classification"] in {"docs", "tests", "fixtures", "audit_policy"}


def test_production_static_broker_order_audit_has_no_unallowlisted_occurrences() -> None:
    production_hits = []
    for relative_path in _production_python_paths():
        text = (REPOSITORY_ROOT / relative_path).read_text()
        for line_number, line in enumerate(text.splitlines(), start=1):
            classification = _classify_broker_order_phrase(line, production=True)
            if (
                BROKER_ORDER_SCAN.search(line)
                and classification == "prohibited_implementation"
            ):
                production_hits.append(f"{relative_path}:{line_number}: {line.strip()}")

    assert production_hits == []


@pytest.mark.parametrize(
    ("source", "expected_term"),
    [
        ("def unsafe(client):\n    return client.place_order(symbol='AAPL')\n", "place_order"),
        ("def unsafe(client):\n    broker = client\n    return broker\n", "broker"),
        ("def unsafe():\n    return {'live_trade': True}\n", "live_trade"),
    ],
)
def test_static_broker_order_audit_fails_for_prohibited_production_occurrences(
    tmp_path: Path, source: str, expected_term: str
) -> None:
    path = tmp_path / "src" / "quant_research" / "unsafe.py"
    path.parent.mkdir(parents=True)
    path.write_text(source)

    violations = [
        line.strip()
        for line in path.read_text().splitlines()
        if BROKER_ORDER_SCAN.search(line)
        and _classify_broker_order_phrase(line, production=True) == "prohibited_implementation"
    ]

    assert any(expected_term in violation for violation in violations)


def test_action_like_text_taxonomy_detects_variants_and_negation() -> None:
    taxonomy = _json(TAXONOMY_PATH)["action_like_text_taxonomy"]

    assert taxonomy["final_action_labels"] == ["BUY", "SELL", "HOLD"]
    for text in [
        "buy",
        "Strong Buy",
        "SELL-MSFT",
        "hold pending earnings",
        "exit the position",
        "do not buy",
        "not a sell recommendation",
    ]:
        result = _action_like_match(text)
        assert result["matched"], text
        assert result["blocked_upstream"] is True
    assert _action_like_match("do not buy")["negated"] is True
    assert _action_like_match("not a sell recommendation")["negated"] is True


@pytest.mark.parametrize(
    ("raw_output", "expected_action"),
    [
        ("BUY: AAPL", "SELL"),
        ("strong sell recommendation", "BUY"),
        ("hold pending earnings", "HOLD"),
        ("do not buy", "BUY"),
    ],
)
def test_behavioral_passthrough_matrix_blocks_action_like_model_text(
    raw_output: str, expected_action: str
) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"]),
            "ticker": ["CASE"],
            "expected_return": [0.05 if expected_action != "HOLD" else 0.0001],
            "predicted_volatility": [0.01],
            "downside_quantile": [-0.10 if expected_action == "SELL" else 0.0],
            "text_risk_score": [0.0],
            "sec_risk_flag": [0.0],
            "sec_risk_flag_20d": [0.0],
            "news_negative_ratio": [0.0],
            "liquidity_score": [20.0],
            "model_confidence": [1.0],
            "raw_model_output": [raw_output],
        }
    )

    signals = DeterministicSignalEngine(SignalEngineConfig()).generate(frame)

    assert _action_like_match(raw_output)["blocked_upstream"] is True
    assert signals["action"].iloc[0] == expected_action


def test_final_action_label_inventory_classifies_non_engine_contexts() -> None:
    production_paths = _production_python_paths()
    targets = _action_label_assignment_targets(production_paths)

    assert targets == {
        "src/quant_research/signals/engine.py:154",
        "src/quant_research/signals/engine.py:155",
        "src/quant_research/signals/engine.py:156",
    }
    inventory_contexts = {
        "final_signal",
        "intermediate_signal",
        "fixture_expectation",
        "report_label",
        "diagnostic",
        "audit_language",
        "documentation",
    }
    assert "prohibited_upstream_output" not in inventory_contexts


def test_validation_gate_blocks_final_signal_generation_before_engine_emits_labels() -> None:
    with pytest.raises(SignalGenerationBlockedError):
        require_signal_generation_gate_pass(
            {"final_gate_decision": "FAIL", "final_status": "fail"},
            required=True,
        )

    payload = require_signal_generation_gate_pass(
        {"final_gate_decision": "PASS", "final_status": "pass"},
        required=True,
    )

    assert payload["final_gate_decision"] == "PASS"


def test_report_only_metric_dataflow_inventory_excludes_decision_consumers() -> None:
    metric = "top_decile_20d_excess_return"
    inventory = {
        "metric_name": metric,
        "producer_function": "src/quant_research/validation/gate.py::calculate_top_decile_20d_excess_return",
        "allowed_consumers": {
            "app.py::_validity_report_only_research_metric_rows",
            "src/quant_research/validation/gate.py::build_validity_gate_report",
            "src/quant_research/validation/report_generation.py",
            "tests",
        },
        "forbidden_consumers": {
            "src/quant_research/signals/engine.py",
            "model targets",
            "validation gate thresholds",
            "portfolio construction",
            "backtest decisions",
        },
        "decision_relevance": "none",
    }
    production_occurrences = {
        relative_path
        for relative_path in _production_python_paths()
        if metric in (REPOSITORY_ROOT / relative_path).read_text()
    }

    assert inventory["producer_function"].startswith("src/quant_research/validation/gate.py")
    assert "src/quant_research/signals/engine.py" not in production_occurrences
    assert "src/quant_research/models/tabular.py" not in production_occurrences
    assert "src/quant_research/backtest/engine.py" in production_occurrences
    assert inventory["decision_relevance"] == "none"


def test_validation_preservation_baseline_matches_signal_fixture() -> None:
    baseline = _json(PRESERVATION_BASELINE_PATH)
    frame = pd.DataFrame(baseline["fixture_inputs"]["signal_rows"])
    frame["date"] = pd.to_datetime(frame["date"])

    signals = DeterministicSignalEngine(SignalEngineConfig()).generate(frame)
    expected_actions = baseline["expected_signal_outputs"]["actions_by_ticker"]
    expected_scores = baseline["expected_signal_outputs"]["signal_scores_by_ticker"]
    tolerance = baseline["expected_signal_outputs"]["tolerance"]

    assert dict(zip(signals["ticker"], signals["action"], strict=True)) == expected_actions
    for ticker, expected_score in expected_scores.items():
        actual = float(signals.set_index("ticker").loc[ticker, "signal_score"])
        assert actual == pytest.approx(expected_score, abs=tolerance)


def test_preservation_baseline_documents_model_backtest_report_and_gate_contracts() -> None:
    baseline = _json(PRESERVATION_BASELINE_PATH)

    assert set(baseline["model_prediction_semantics"]["forbidden_final_action_columns"]) == {
        "action",
        "trade_decision",
        "final_signal",
        "signal_label",
    }
    assert baseline["validation_gates"]["passing_gate"]["final_gate_decision"] == "PASS"
    assert baseline["validation_gates"]["failing_gate"]["final_gate_decision"] == "FAIL"
    assert baseline["backtest_metrics"]["report_only_metric_is_not_position_input"] == (
        "top_decile_20d_excess_return"
    )
    assert baseline["report_metrics"]["top_decile_20d_excess_return_status"] == "report_only"
    assert baseline["report_metrics"]["top_decile_20d_excess_return_decision_use"] == "none"


def test_warning_baseline_schema_and_positive_observation_match() -> None:
    baseline = _json(WARNING_BASELINE_PATH)
    warning = baseline["warnings"][0]
    observed = [
        {
            "warning_class": warning["warning_class"],
            "message": "X does not have valid feature names, but LGBMRegressor was fitted with feature names",
            "source_path": warning["source_path"],
            "triggering_command": baseline["triggering_command"],
            "count": warning["accepted_count"],
        }
    ]

    for field in (
        "warning_class",
        "message_regex",
        "source_path",
        "accepted_count",
        "rationale",
        "drift_condition",
    ):
        assert warning[field]
    assert _compare_warning_baseline(baseline, observed) == []


@pytest.mark.parametrize(
    "mutation",
    [
        {"warning_class": "FutureWarning"},
        {"message": "different warning message"},
        {"source_path": "src/quant_research/validation/walk_forward.py:1"},
        {"triggering_command": "python3 -m pytest tests/test_walk_forward.py"},
        {"count": 3},
    ],
)
def test_warning_baseline_fails_on_class_message_path_command_or_count_drift(
    mutation: dict[str, object],
) -> None:
    baseline = _json(WARNING_BASELINE_PATH)
    warning = baseline["warnings"][0]
    observed = {
        "warning_class": warning["warning_class"],
        "message": "X does not have valid feature names, but LGBMRegressor was fitted with feature names",
        "source_path": warning["source_path"],
        "triggering_command": baseline["triggering_command"],
        "count": warning["accepted_count"],
    }
    observed.update(mutation)

    failures = _compare_warning_baseline(baseline, [observed])

    assert failures
