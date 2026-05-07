from __future__ import annotations

import re
from pathlib import Path

# Guardrail/policy audit: broker/order vocabulary is allowed in docs/tests only
# when nearby text makes the reference an explicit prohibition or evidence check.
BROKER_ORDER_POLICY_LANGUAGE = re.compile(
    r"\b("
    r"broker|brokers|order|orders|place_order|submit_order|create_order|"
    r"market_order|limit_order|send_order|live_trade"
    r")\b"
    r"|execute trade|place trade|submit trade|send order|"
    r"매수 주문|매도 주문|주문 전송",
    re.IGNORECASE,
)

POLICY_CONTEXT_MARKERS = (
    "absence",
    "allowlist",
    "assert",
    "audit",
    "block",
    "차단",
    "evidence",
    "forbid",
    "금지",
    "guard",
    "guardrail",
    "hard-fail",
    "invariant",
    "isolation",
    "must not",
    "no ",
    "not ",
    "policy",
    "prohibit",
    "report-only",
    "검증",
    "감사",
    "범위 밖",
    "부재",
    "불변식",
    "정책",
    "증거",
    "허용",
)
APPROVED_POLICY_LANGUAGE_PREFIXES = ("docs/", "tests/")
APPROVED_POLICY_LANGUAGE_CONTEXTS = (
    "docs/**",
    "tests/**",
    "tests/fixtures/**",
    "deterministic signal engine tests",
    "report/schema tests",
    "policy docs",
)
EXECUTABLE_POLICY_LANGUAGE_CONTEXTS = (
    "app.py",
    "src/quant_research/**/*.py",
)


def _markdown_table_rows(markdown: str, heading: str) -> list[list[str]]:
    lines = markdown.splitlines()
    try:
        heading_index = lines.index(heading)
    except ValueError:
        return []

    rows: list[list[str]] = []
    in_table = False
    for line in lines[heading_index + 1 :]:
        if not line.strip():
            if in_table:
                break
            continue
        if not line.startswith("|"):
            if in_table:
                break
            continue
        in_table = True
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and set(cells[0]) <= {"-", ":"}:
            continue
        rows.append(cells)
    return rows


def _policy_language_hits(repo_root: Path) -> list[tuple[Path, int, str]]:
    searched_roots = (repo_root / "docs", repo_root / "tests")
    searched_suffixes = {".csv", ".json", ".md", ".py", ".txt", ".yaml", ".yml"}
    hits: list[tuple[Path, int, str]] = []

    for root in searched_roots:
        for path in root.rglob("*"):
            if path.suffix not in searched_suffixes:
                continue

            for index, line in enumerate(path.read_text().splitlines(), start=1):
                if BROKER_ORDER_POLICY_LANGUAGE.search(line):
                    hits.append((path.relative_to(repo_root), index, line.strip()))

    return hits


def test_risk_validation_docs_define_narrow_policy_language_exceptions() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    markdown = (repo_root / "docs" / "risk-validation.md").read_text()

    assert "### Narrow Policy-Language Exceptions" in markdown
    assert "| Exception type | Allowed paths or surfaces | Required invariant |" in markdown
    for exception_type in (
        "Non-executable guidance",
        "Validation audit text",
        "Report evidence text",
        "Final-action labels in evidence",
    ):
        assert exception_type in markdown

    for invariant in (
        "The exception does not cover `app.py` or `src/quant_research/**/*.py`.",
        "Executable-path enforcement remains hard-fail",
        "These documentation, report, and validation-text",
        "exceptions are evidence-only",
        "have no validation impact",
    ):
        assert invariant in markdown


def test_risk_validation_docs_define_no_live_trading_boundary() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    markdown = (repo_root / "docs" / "risk-validation.md").read_text()

    assert "## No-Live-Trading Boundary" in markdown
    for scope_item in (
        "research-only quantitative validation app",
        "data collection, feature generation, optional model",
        "walk-forward validation, long-only",
        "backtest simulation, static report generation",
        "Streamlit review surfaces",
    ):
        assert scope_item in markdown

    for prohibited_behavior in (
        "must not include live trading behavior",
        "broker account connectivity",
        "order routing",
        "order ticket",
        "order status polling",
        "live execution payloads",
        "place, submit, create, send, or manage orders",
        "authorize executable broker adapters or runtime branches",
    ):
        assert prohibited_behavior in markdown

    for boundary in (
        "Predictions remain structured research features, not orders or final action",
        "`BUY`, `SELL`, and `HOLD` are deterministic signal-engine labels",
        "Backtests are simulations that apply labels to t+1-or-later returns",
        "Reports and dashboards summarize predictions, validation status",
        "they must not render",
        "broker/order instructions, live execution payloads, or order-management",
    ):
        assert boundary in markdown

    for evidence in (
        "tests/test_architecture_guards.py::test_repository_has_no_live_trading_or_order_execution_modules",
        "tests/test_architecture_guards.py::test_executable_source_does_not_import_live_trading_broker_sdks",
        "tests/test_architecture_guards.py::test_source_does_not_call_broker_order_apis",
        "tests/test_architecture_guards.py::test_executable_source_does_not_define_live_execution_payload_keys",
        "tests/test_report_only_execution_isolation.py::test_report_only_outputs_contain_no_live_order_or_execution_payloads",
        "tests/test_policy_language_guardrails.py::test_risk_validation_docs_define_no_live_trading_boundary",
        "uv --cache-dir .uv-cache run pytest tests/test_architecture_guards.py tests/test_report_only_execution_isolation.py tests/test_policy_language_guardrails.py",
    ):
        assert evidence in markdown

    assert "Expected validation impact: none." in markdown
    assert "must not alter validation behavior, signal" in markdown
    assert "model predictions, backtest results, report metrics" in markdown


def test_risk_validation_docs_define_final_action_label_gate_sequence() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    markdown = (repo_root / "docs" / "risk-validation.md").read_text()

    assert "## Final Action Label Gate Sequence" in markdown
    for invariant in (
        "`BUY`, `SELL`, and `HOLD` are final action labels, not model prediction",
        "feature availability timestamps and model",
        "realized return",
        "labels are excluded from signal-engine inputs",
        "structured research features",
        "These values remain inputs only.",
        "after subtracting configured transaction cost and",
        "slippage plus volatility",
        "canonical `forward_return_20` target with purge/embargo protection",
        "`final_gate_decision` is `PASS`",
        "missing, `WARN`, `FAIL`, hard-fail, or",
        "not-evaluable gate results block final label emission",
        "`BUY` requires expected return, score, volatility, downside,",
        "`SELL` is emitted",
        "all remaining rows stay",
        "`HOLD`",
        "Positions use t+1-or-later returns",
        "Expected validation impact: none.",
        "does not change validation semantics, signal labels, model predictions, backtest",
        "returns, or report metrics",
    ):
        assert invariant in markdown

    for evidence in (
        "src/quant_research/signals/engine.py::require_signal_generation_gate_pass",
        "src/quant_research/backtest/engine.py::run_long_only_backtest",
        "tests/test_backtest_risk.py::test_backtest_blocks_final_signal_path_when_common_gate_is_not_pass",
        "tests/test_backtest_risk.py::test_backtest_records_common_gate_pass_on_final_signals",
        "tests/test_backtest_risk.py::test_backtest_signal_engine_receives_no_realized_return_columns",
        "python3 -m pytest tests/test_backtest_risk.py tests/test_signal_engine.py tests/test_policy_language_guardrails.py",
    ):
        assert evidence in markdown


def test_risk_validation_docs_define_audit_evidence_reproduction() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    markdown = (repo_root / "docs" / "risk-validation.md").read_text()

    assert "### Audit Evidence Reproduction" in markdown
    assert "| Evidence area | Reproduction command | Expected result | Evidence reviewed or stored |" in markdown
    for command in (
        "uv --cache-dir .uv-cache run pytest tests/test_architecture_guards.py tests/test_report_only_execution_isolation.py",
        "uv --cache-dir .uv-cache run pytest tests/test_policy_language_guardrails.py",
        "uv --cache-dir .uv-cache run pytest tests/test_signal_engine.py tests/test_backtest_risk.py tests/test_architecture_guards.py",
        "uv --cache-dir .uv-cache run pytest tests/test_validity_gate_metric_formulas.py tests/test_report_generation.py tests/test_report_only_execution_isolation.py",
        "uv --cache-dir .uv-cache run pytest -q",
        "rg -n --glob 'src/quant_research/**/*.py' --glob 'app.py'",
    ):
        assert command in markdown

    for evidence_target in (
        "docs/live-order-placement-audit.md",
        "tests/fixtures/report_generation",
        "tests/test_architecture_guards.py",
        "tests/test_policy_language_guardrails.py",
        "docs/final-action-label-inventory.md",
        "src/quant_research/signals/engine.py",
        "tests/test_warning_baseline_documentation.py",
    ):
        assert evidence_target in markdown

    for expected_result in (
        "no new generated artifacts are written",
        "evidence is stored as committed docs/tests only",
        "generated report artifacts are not committed",
        "Current integrated baseline is `832 passed, 12 warnings`",
        "Expected result: no output.",
        "Validation impact: none.",
        "must not change validation behavior, signal semantics, model",
        "predictions, backtest results, report metrics",
    ):
        assert expected_result in markdown


def test_live_order_placement_absence_audit_artifact_is_reviewable() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    markdown = (repo_root / "docs" / "live-order-placement-audit.md").read_text()
    normalized_markdown = " ".join(markdown.split())

    assert "# Live Order Placement Absence Audit" in markdown
    assert "documentation-only evidence for semantic QA hardening" in normalized_markdown
    assert "does not change validation behavior, signal semantics" in normalized_markdown
    assert "model predictions, backtest results, report metrics" in normalized_markdown

    for required_section in (
        "## Audit Scope",
        "## Verified Absence Summary",
        "## Reproduction Commands",
        "## Allowlist",
        "## Review Notes",
    ):
        assert required_section in markdown

    for affected_path in (
        "`app.py`, `src/quant_research/**/*.py`",
        "`src/quant_research/validation/**`, `tests/fixtures/report_generation/**`",
        "`docs/**`, `tests/**`, `tests/fixtures/**`",
    ):
        assert affected_path in markdown

    for invariant in (
        "must not import broker SDKs",
        "define broker/order execution modules",
        "call order placement APIs",
        "construct live execution payloads",
        "Reports are review artifacts only",
        "must not contain live order payloads",
        "Broker/order terminology is allowed only to describe prohibited behavior",
    ):
        assert invariant in markdown

    for evidence in (
        "Static architecture guard tests plus ad hoc text scan.",
        "Report-only artifact tests over JSON, Markdown, HTML, and fixture files.",
        "Documentation allowlist and policy-language tests.",
        "uv --cache-dir .uv-cache run pytest tests/test_architecture_guards.py tests/test_report_only_execution_isolation.py tests/test_policy_language_guardrails.py",
        "rg -n --glob 'src/quant_research/**/*.py' --glob 'app.py'",
    ):
        assert evidence in markdown

    for validation_impact in (
        "None: no validation formulas, signal thresholds, model outputs",
        "None: report writers still emit static review files only.",
        "None: allowlisted words are non-executable evidence only.",
    ):
        assert validation_impact in markdown


def test_documented_policy_language_exceptions_name_only_non_executable_contexts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    markdown = (repo_root / "docs" / "risk-validation.md").read_text()
    allowlist_rows = _markdown_table_rows(markdown, "### Broker/Order Term Allowlist")
    exception_rows = _markdown_table_rows(markdown, "### Narrow Policy-Language Exceptions")

    assert allowlist_rows
    assert exception_rows

    documented_contexts = [
        row[1]
        for row in allowlist_rows[1:] + exception_rows[1:]
        if len(row) >= 2
    ]
    assert documented_contexts

    for context in documented_contexts:
        assert any(
            approved in context for approved in APPROVED_POLICY_LANGUAGE_CONTEXTS
        ), context
        assert not any(
            executable in context for executable in EXECUTABLE_POLICY_LANGUAGE_CONTEXTS
        ), context


def test_policy_language_matches_are_limited_to_approved_non_executable_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    violations = []

    for relative_path, line_number, line in _policy_language_hits(repo_root):
        relative = relative_path.as_posix()
        if not relative.startswith(APPROVED_POLICY_LANGUAGE_PREFIXES):
            violations.append(f"{relative}:{line_number}: {line}")

    assert not violations, "\n".join(violations)


def test_docs_and_tests_use_broker_order_terms_only_as_policy_language() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    searched_roots = (repo_root / "docs", repo_root / "tests")
    searched_suffixes = {".csv", ".json", ".md", ".py", ".txt", ".yaml", ".yml"}
    violations: list[str] = []

    for root in searched_roots:
        for path in root.rglob("*"):
            if path.suffix not in searched_suffixes:
                continue

            lines = path.read_text().splitlines()
            for index, line in enumerate(lines):
                if not BROKER_ORDER_POLICY_LANGUAGE.search(line):
                    continue

                start = max(0, index - 16)
                end = min(len(lines), index + 17)
                context = "\n".join(lines[start:end]).lower()
                if not any(marker in context for marker in POLICY_CONTEXT_MARKERS):
                    relative_path = path.relative_to(repo_root)
                    violations.append(f"{relative_path}:{index + 1}: {line.strip()}")

    assert not violations, "\n".join(violations)
