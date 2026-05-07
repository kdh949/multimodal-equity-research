from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
APPLICATION_ENTRYPOINT = REPOSITORY_ROOT / "app.py"
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "quant_research"
EXECUTABLE_CODE_SCAN_SCOPE = (APPLICATION_ENTRYPOINT, PACKAGE_ROOT)
PRODUCTION_POLICY_GUARD_PATHS = (
    PACKAGE_ROOT / "signals" / "engine.py",
    PACKAGE_ROOT / "validation",
    PACKAGE_ROOT / "runtime.py",
)
MODEL_LLM_OUTPUT_SCAN_SCOPE = (
    PACKAGE_ROOT / "models",
    PACKAGE_ROOT / "features" / "text.py",
    PACKAGE_ROOT / "features" / "sec.py",
)
PROHIBITED_LIVE_TRADING_MODULE_TERMS = {
    "broker",
    "brokers",
    "order",
    "orders",
    "execution",
    "trading",
}
PROHIBITED_ORDER_API_CALL_TERMS = {
    "place_order",
    "submit_order",
    "create_order",
    "market_order",
    "limit_order",
    "send_order",
    "live_trade",
}
PROHIBITED_LIVE_TRADING_DEPENDENCIES = {
    "alpaca-py",
    "alpaca-trade-api",
    "ccxt",
    "etrade",
    "ib-insync",
    "ibapi",
    "interactive-brokers",
    "oandapyv20",
    "robin-stocks",
    "robinhood",
    "schwab-py",
    "tda-api",
    "tradier",
}
PROHIBITED_LIVE_TRADING_IMPORT_ROOTS = {
    "alpaca",
    "ccxt",
    "etrade",
    "ib_insync",
    "ibapi",
    "oandapyV20",
    "robin_stocks",
    "robinhood",
    "schwab",
    "tda",
    "tradier",
}
PROHIBITED_LIVE_TRADING_IMPLEMENTATION_NAME_TERMS = {
    "broker",
    "brokerage",
    "broker_client",
    "execution_adapter",
    "live_trading",
    "live_execution",
    "order_client",
    "order_executor",
    "place_order",
    "submit_order",
    "trade_executor",
}
PROHIBITED_EXECUTABLE_SOURCE_TERMS = (
    "broker",
    "place_order",
    "submit_order",
    "create_order",
    "market_order",
    "limit_order",
    "send_order",
    "live_trade",
)
PROHIBITED_LIVE_EXECUTION_PAYLOAD_KEYS = (
    "broker",
    "order",
    "orders",
    "execution",
    "live_execution",
    "live_trade",
)
PROHIBITED_MODEL_LLM_FINAL_ACTION_OUTPUT_KEYS = (
    "action",
    "trade_decision",
    "final_signal",
    "signal_label",
)
PROHIBITED_EXECUTABLE_SOURCE_PATTERN = re.compile(
    r"\b("
    + "|".join(re.escape(term) for term in PROHIBITED_EXECUTABLE_SOURCE_TERMS)
    + r")\b",
    re.IGNORECASE,
)
FINAL_ACTION_LABELS = {"BUY", "SELL", "HOLD"}
ALLOWED_FINAL_ACTION_LABEL_EMISSION_TARGETS = {
    "src/quant_research/signals/engine.py:154",
    "src/quant_research/signals/engine.py:155",
    "src/quant_research/signals/engine.py:156",
}
ALLOWED_FINAL_ACTION_ASSIGNMENT_TARGETS = ALLOWED_FINAL_ACTION_LABEL_EMISSION_TARGETS | {
    "src/quant_research/backtest/engine.py:222",
}


def _python_files_in_scope(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
    return tuple(files)


def _called_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name):
            names.add(function.id)
        elif isinstance(function, ast.Attribute):
            names.add(function.attr)
    return names


def _configured_dependency_names() -> set[str]:
    pyproject = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())
    dependency_groups = [pyproject["project"].get("dependencies", [])]
    dependency_groups.extend(pyproject["project"].get("optional-dependencies", {}).values())

    names: set[str] = set()
    for group in dependency_groups:
        for dependency in group:
            names.add(re.split(r"[<>=!~;,\[\]\s]", dependency, maxsplit=1)[0].lower())
    return names


def _import_root_violations(paths: tuple[Path, ...]) -> list[str]:
    prohibited_roots = {root.lower() for root in PROHIBITED_LIVE_TRADING_IMPORT_ROOTS}
    violations: list[str] = []
    for path in _python_files_in_scope(paths):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            imported_roots: list[str] = []
            if isinstance(node, ast.Import):
                imported_roots.extend(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.append(node.module.split(".", maxsplit=1)[0])
            if prohibited_roots.intersection(root.lower() for root in imported_roots):
                violations.append(_relative_node_location(path, node))
    return violations


def _implementation_name_violations(paths: tuple[Path, ...]) -> list[str]:
    prohibited_terms = {
        _normalize_identifier_for_policy(term)
        for term in PROHIBITED_LIVE_TRADING_IMPLEMENTATION_NAME_TERMS
    }
    violations: list[str] = []
    for path in _python_files_in_scope(paths):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            normalized_name = _normalize_identifier_for_policy(node.name)
            if any(term in normalized_name for term in prohibited_terms):
                violations.append(f"{_relative_node_location(path, node)}: {node.name}")
    return violations


def _normalize_identifier_for_policy(identifier: str) -> str:
    with_separators = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", identifier)
    return re.sub(r"[^a-z0-9]+", "_", with_separators.lower()).strip("_")


def _module_term_violations(package_root: Path) -> list[str]:
    violations: list[str] = []
    for path in package_root.rglob("*.py"):
        relative_path = path.relative_to(package_root)
        for part in relative_path.with_suffix("").parts:
            normalized = part.lower()
            if normalized in PROHIBITED_LIVE_TRADING_MODULE_TERMS:
                violations.append(f"{relative_path}: prohibited module term {part}")
    return violations


def _source_term_violations(paths: tuple[Path, ...]) -> list[str]:
    violations: list[str] = []
    for path in _python_files_in_scope(paths):
        for index, line in enumerate(path.read_text().splitlines(), start=1):
            if PROHIBITED_EXECUTABLE_SOURCE_PATTERN.search(line):
                try:
                    relative_path = path.relative_to(REPOSITORY_ROOT)
                except ValueError:
                    relative_path = path
                violations.append(f"{relative_path}:{index}: {line.strip()}")
    return violations


def _live_execution_payload_key_targets(paths: tuple[Path, ...]) -> set[str]:
    prohibited_keys = set(PROHIBITED_LIVE_EXECUTION_PAYLOAD_KEYS)
    return _payload_key_targets(paths, prohibited_keys)


def _payload_key_targets(paths: tuple[Path, ...], prohibited_keys: set[str]) -> set[str]:
    targets: set[str] = set()
    for path in _python_files_in_scope(paths):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key in node.keys:
                    if (
                        isinstance(key, ast.Constant)
                        and isinstance(key.value, str)
                        and key.value.lower() in prohibited_keys
                    ):
                        targets.add(_relative_node_location(path, node))
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg and keyword.arg.lower() in prohibited_keys:
                        targets.add(_relative_node_location(path, node))
    return targets


def _source_text(path: str) -> str:
    return (REPOSITORY_ROOT / path).read_text()


def _is_action_subscript(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == "action"
    )


def _contains_action_subscript(node: ast.AST) -> bool:
    return any(
        _is_action_subscript(child)
        or (isinstance(child, ast.Constant) and child.value == "action")
        for child in ast.walk(node)
    )


def _contains_final_action_label_literal(node: ast.AST | None) -> bool:
    if node is None:
        return False
    return any(
        isinstance(child, ast.Constant) and child.value in FINAL_ACTION_LABELS
        for child in ast.walk(node)
    )


def _relative_node_location(path: Path, node: ast.AST) -> str:
    try:
        relative_path = path.relative_to(REPOSITORY_ROOT)
    except ValueError:
        relative_path = path
    return f"{relative_path}:{node.lineno}"


def _action_assignment_targets(paths: tuple[Path, ...]) -> set[str]:
    targets: set[str] = set()
    for path in _python_files_in_scope(paths):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                assigned_nodes = node.targets
            elif isinstance(node, ast.AnnAssign):
                assigned_nodes = [node.target]
            elif isinstance(node, ast.AugAssign):
                assigned_nodes = [node.target]
            else:
                continue
            if not any(_contains_action_subscript(target) for target in assigned_nodes):
                continue
            targets.add(_relative_node_location(path, node))
    return targets


def _final_action_label_emission_targets(paths: tuple[Path, ...]) -> set[str]:
    targets: set[str] = set()
    for path in _python_files_in_scope(paths):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                assigned_nodes = node.targets
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                assigned_nodes = [node.target]
                value = node.value
            elif isinstance(node, ast.AugAssign):
                assigned_nodes = [node.target]
                value = node.value
            else:
                assigned_nodes = []
                value = None

            if (
                assigned_nodes
                and any(_contains_action_subscript(target) for target in assigned_nodes)
                and _contains_final_action_label_literal(value)
            ):
                targets.add(_relative_node_location(path, node))

            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == "action" and _contains_final_action_label_literal(
                        keyword.value
                    ):
                        targets.add(_relative_node_location(path, node))

            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values, strict=True):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value == "action"
                        and _contains_final_action_label_literal(value)
                    ):
                        targets.add(_relative_node_location(path, node))
    return targets


def test_live_trading_order_audit_scope_covers_application_and_package_code() -> None:
    assert EXECUTABLE_CODE_SCAN_SCOPE == (APPLICATION_ENTRYPOINT, PACKAGE_ROOT)
    assert APPLICATION_ENTRYPOINT.is_file()
    assert PACKAGE_ROOT.is_dir()


def test_prohibited_live_trading_order_term_lists_are_explicit() -> None:
    assert PROHIBITED_LIVE_TRADING_MODULE_TERMS == {
        "broker",
        "brokers",
        "order",
        "orders",
        "execution",
        "trading",
    }
    assert PROHIBITED_ORDER_API_CALL_TERMS == {
        "place_order",
        "submit_order",
        "create_order",
        "market_order",
        "limit_order",
        "send_order",
        "live_trade",
    }
    assert PROHIBITED_LIVE_TRADING_DEPENDENCIES == {
        "alpaca-py",
        "alpaca-trade-api",
        "ccxt",
        "etrade",
        "ib-insync",
        "ibapi",
        "interactive-brokers",
        "oandapyv20",
        "robin-stocks",
        "robinhood",
        "schwab-py",
        "tda-api",
        "tradier",
    }
    assert PROHIBITED_LIVE_TRADING_IMPORT_ROOTS == {
        "alpaca",
        "ccxt",
        "etrade",
        "ib_insync",
        "ibapi",
        "oandapyV20",
        "robin_stocks",
        "robinhood",
        "schwab",
        "tda",
        "tradier",
    }
    assert PROHIBITED_LIVE_TRADING_IMPLEMENTATION_NAME_TERMS == {
        "broker",
        "brokerage",
        "broker_client",
        "execution_adapter",
        "live_trading",
        "live_execution",
        "order_client",
        "order_executor",
        "place_order",
        "submit_order",
        "trade_executor",
    }
    assert PROHIBITED_EXECUTABLE_SOURCE_TERMS == (
        "broker",
        "place_order",
        "submit_order",
        "create_order",
        "market_order",
        "limit_order",
        "send_order",
        "live_trade",
    )


def test_repository_has_no_live_trading_or_order_execution_modules() -> None:
    violations = _module_term_violations(PACKAGE_ROOT)

    assert violations == []


def test_project_dependencies_do_not_include_live_trading_broker_sdks() -> None:
    dependency_names = _configured_dependency_names()

    assert dependency_names.isdisjoint(PROHIBITED_LIVE_TRADING_DEPENDENCIES)


def test_executable_source_does_not_import_live_trading_broker_sdks() -> None:
    violations = _import_root_violations(EXECUTABLE_CODE_SCAN_SCOPE)

    assert violations == []


def test_executable_source_does_not_define_live_trading_implementation_surfaces() -> None:
    violations = _implementation_name_violations(EXECUTABLE_CODE_SCAN_SCOPE)

    assert violations == []


def test_source_does_not_call_broker_order_apis() -> None:
    called_names = {
        name
        for path in _python_files_in_scope(EXECUTABLE_CODE_SCAN_SCOPE)
        for name in _called_names(path)
    }

    assert called_names.isdisjoint(PROHIBITED_ORDER_API_CALL_TERMS)


@pytest.mark.parametrize("dependency", sorted(PROHIBITED_LIVE_TRADING_DEPENDENCIES))
def test_dependency_guard_flags_prohibited_broker_sdk(dependency: str) -> None:
    assert dependency in PROHIBITED_LIVE_TRADING_DEPENDENCIES


@pytest.mark.parametrize("import_root", sorted(PROHIBITED_LIVE_TRADING_IMPORT_ROOTS))
def test_import_guard_flags_prohibited_broker_sdk_roots(
    tmp_path: Path,
    import_root: str,
) -> None:
    source_path = tmp_path / "src" / "quant_research" / "unsafe_adapter.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(f"import {import_root}\n")

    violations = _import_root_violations((source_path,))

    assert violations == [f"{source_path}:1"]


@pytest.mark.parametrize(
    "implementation_name",
    (
        "BrokerClient",
        "LiveTradingAdapter",
        "OrderExecutor",
        "submitOrder",
        "place_order",
    ),
)
def test_implementation_name_guard_flags_live_trading_surfaces(
    tmp_path: Path,
    implementation_name: str,
) -> None:
    source_path = tmp_path / "src" / "quant_research" / "unsafe_surface.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    if implementation_name[0].isupper():
        source_path.write_text(f"class {implementation_name}:\n    pass\n")
    else:
        source_path.write_text(f"def {implementation_name}():\n    return None\n")

    violations = _implementation_name_violations((source_path,))

    assert violations == [f"{source_path}:1: {implementation_name}"]


def test_order_api_call_guard_flags_prohibited_implementation_terms(tmp_path: Path) -> None:
    source_path = tmp_path / "strategy_research.py"
    source_path.write_text(
        "\n".join(
            (
                "def unsafe_live_path(client):",
                "    return client.submit_order(symbol='AAPL', quantity=1)",
            )
        )
    )

    called_names = _called_names(source_path)

    assert "submit_order" in called_names
    assert not called_names.isdisjoint(PROHIBITED_ORDER_API_CALL_TERMS)


def test_executable_source_does_not_contain_broker_order_api_terms() -> None:
    violations = _source_term_violations(EXECUTABLE_CODE_SCAN_SCOPE)

    assert not violations, "\n".join(violations)


def test_executable_source_does_not_define_live_execution_payload_keys() -> None:
    payload_key_targets = _live_execution_payload_key_targets(EXECUTABLE_CODE_SCAN_SCOPE)

    assert payload_key_targets == set()


def test_signal_validation_and_runtime_paths_reject_prohibited_policy_terms() -> None:
    assert PRODUCTION_POLICY_GUARD_PATHS == (
        PACKAGE_ROOT / "signals" / "engine.py",
        PACKAGE_ROOT / "validation",
        PACKAGE_ROOT / "runtime.py",
    )
    for path in PRODUCTION_POLICY_GUARD_PATHS:
        assert path.exists()

    violations = _source_term_violations(PRODUCTION_POLICY_GUARD_PATHS)

    assert not violations, "\n".join(violations)


def test_production_path_policy_guard_flags_each_executable_surface(tmp_path: Path) -> None:
    signal_path = tmp_path / "src" / "quant_research" / "signals" / "engine.py"
    validation_path = tmp_path / "src" / "quant_research" / "validation" / "gate.py"
    runtime_path = tmp_path / "src" / "quant_research" / "runtime.py"
    signal_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    signal_path.write_text("def unsafe_signal_path(client):\n    return client.place_order()\n")
    validation_path.write_text("def unsafe_validation_path(client):\n    return client.submit_order()\n")
    runtime_path.write_text("def unsafe_runtime_path():\n    live_trade = True\n    return live_trade\n")

    violations = _source_term_violations((signal_path, validation_path, runtime_path))

    assert len(violations) == 4
    assert any(
        "signals/engine.py:2" in violation and "place_order" in violation
        for violation in violations
    )
    assert any(
        "validation/gate.py:2" in violation and "submit_order" in violation
        for violation in violations
    )
    assert any(
        "runtime.py:2" in violation and "live_trade" in violation
        for violation in violations
    )
    assert any(
        "runtime.py:3" in violation and "live_trade" in violation
        for violation in violations
    )


def test_executable_source_term_guard_flags_prohibited_implementation_terms(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "live_trading_adapter.py"
    source_path.write_text(
        "\n".join(
            (
                "def unsafe_live_path(broker_client):",
                "    broker = broker_client",
                "    return broker.place_order(symbol='AAPL', quantity=1)",
            )
        )
    )

    violations = _source_term_violations((source_path,))

    assert len(violations) == 2
    assert "broker =" in violations[0]
    assert "place_order" in violations[1]


@pytest.mark.parametrize(
    "module_term",
    sorted(PROHIBITED_LIVE_TRADING_MODULE_TERMS),
)
def test_documented_exception_module_terms_fail_in_production_modules(
    tmp_path: Path,
    module_term: str,
) -> None:
    package_root = tmp_path / "src" / "quant_research"
    module_path = package_root / f"{module_term}.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text("def research_only_placeholder():\n    return None\n")

    violations = _module_term_violations(package_root)

    assert violations == [f"{module_term}.py: prohibited module term {module_term}"]


@pytest.mark.parametrize("source_term", PROHIBITED_EXECUTABLE_SOURCE_TERMS)
def test_documented_exception_source_terms_fail_in_executable_production_context(
    tmp_path: Path,
    source_term: str,
) -> None:
    source_path = tmp_path / "src" / "quant_research" / "unsafe_policy_context.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    if source_term == "broker":
        source = "def unsafe_path(client):\n    broker = client\n    return broker\n"
    elif source_term == "live_trade":
        source = "def unsafe_path():\n    live_trade = True\n    return live_trade\n"
    else:
        source = f"def unsafe_path(client):\n    return client.{source_term}()\n"
    source_path.write_text(source)

    violations = _source_term_violations((source_path,))

    assert violations
    assert any(source_term in violation for violation in violations)


@pytest.mark.parametrize("payload_key", PROHIBITED_LIVE_EXECUTION_PAYLOAD_KEYS)
def test_documented_exception_payload_terms_fail_as_production_payload_fields(
    tmp_path: Path,
    payload_key: str,
) -> None:
    source_path = tmp_path / "src" / "quant_research" / "validation" / "unsafe_payload.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        "\n".join(
            (
                "def unsafe_payload():",
                f"    return {{{payload_key!r}: True}}",
            )
        )
    )

    payload_key_targets = _live_execution_payload_key_targets((source_path,))

    assert payload_key_targets == {f"{source_path}:2"}


@pytest.mark.parametrize("payload_key", PROHIBITED_LIVE_EXECUTION_PAYLOAD_KEYS)
def test_documented_exception_payload_terms_fail_as_production_keyword_fields(
    tmp_path: Path,
    payload_key: str,
) -> None:
    source_path = tmp_path / "src" / "quant_research" / "runtime.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        "\n".join(
            (
                "def build_payload(**kwargs):",
                "    return kwargs",
                "",
                "def unsafe_runtime_payload():",
                f"    return build_payload({payload_key}=True)",
            )
        )
    )

    payload_key_targets = _live_execution_payload_key_targets((source_path,))

    assert payload_key_targets == {f"{source_path}:5"}


def test_executable_source_term_guard_passes_documented_non_executable_context(
    tmp_path: Path,
) -> None:
    docs_path = tmp_path / "risk-validation.md"
    docs_path.write_text(
        "\n".join(
            (
                "### Broker/Order Term Allowlist",
                "`broker` and `place_order` are allowed here only as audit evidence.",
                "This documentation does not add a runtime order path.",
            )
        )
    )

    violations = _source_term_violations((docs_path,))

    assert violations == []


def test_production_guard_scope_excludes_allowlisted_docs_and_tests() -> None:
    scope_strings = {str(path.relative_to(REPOSITORY_ROOT)) for path in EXECUTABLE_CODE_SCAN_SCOPE}

    assert scope_strings == {"app.py", "src/quant_research"}
    assert "docs" not in scope_strings
    assert "tests" not in scope_strings


def test_risk_validation_docs_explain_broker_order_term_allowlist() -> None:
    docs_path = REPOSITORY_ROOT / "docs" / "risk-validation.md"
    markdown = docs_path.read_text()

    assert "### Broker/Order Term Allowlist" in markdown
    assert "| Allowed reference | Permitted paths | Why it is allowed |" in markdown
    for term in (
        "`broker`",
        "`order`, `orders`, `place_order`",
        "`execution`, `live execution`, `live_trade`",
        "`BUY`, `SELL`, `HOLD`",
    ):
        assert term in markdown


def test_final_action_label_inventory_covers_all_audited_surfaces() -> None:
    docs_path = REPOSITORY_ROOT / "docs" / "final-action-label-inventory.md"
    markdown = docs_path.read_text()

    assert "Final action labels are produced only by the deterministic signal engine" in markdown
    assert "| `src/quant_research/signals/engine.py` | `DeterministicSignalEngine.generate` |" in markdown

    for section in (
        "## Source Of Final Labels",
        "## Package API Surfaces",
        "## UI Surfaces",
        "## Report Surfaces",
        "## Export Surfaces",
        "## Explicit Non-Emission Paths",
        "## Emission Path Provenance Evidence",
        "## Audit Commands",
    ):
        assert section in markdown

    for reference in (
        "`app.py`",
        "`scripts/run_backtest_validation.py`",
        "`src/quant_research/backtest/engine.py`",
        "`src/quant_research/dashboard/beginner.py`",
        "`src/quant_research/dashboard/streamlit.py`",
        "`src/quant_research/pipeline.py`",
        "`src/quant_research/signals/__init__.py`",
        "`src/quant_research/validation/gate.py`",
        "`src/quant_research/validation/report_generation.py`",
        "`src/quant_research/validation/report_renderer.py`",
        "`src/quant_research/validation/report_schema.py`",
    ):
        assert reference in markdown

    for invariant in (
        "does not change validation behavior",
        "does not accept raw model prediction sections",
        "no live order path",
        "only `src/quant_research/signals/engine.py` assigns final",
        "Each audited emission path receives labels from the deterministic signal output",
        "defensive `result.predictions[\"action\"]` fallback is documented as a non-emission",
    ):
        assert invariant in markdown


def test_final_action_assignment_sources_are_limited_to_engine_and_backtest_passthrough() -> None:
    action_assignments = _action_assignment_targets(EXECUTABLE_CODE_SCAN_SCOPE)

    assert action_assignments == ALLOWED_FINAL_ACTION_ASSIGNMENT_TARGETS

    engine_source = _source_text("src/quant_research/signals/engine.py")
    assert 'scored["action"] = "HOLD"' in engine_source
    assert 'scored.loc[sell_mask, "action"] = "SELL"' in engine_source
    assert 'scored.loc[buy_mask, "action"] = "BUY"' in engine_source

    backtest_source = _source_text("src/quant_research/backtest/engine.py")
    assert "generated_signals = signal_engine.generate(" in backtest_source
    assert 'signals["action"] = generated_signals["action"].to_numpy()' in backtest_source


def test_final_action_label_emissions_are_created_only_by_signal_engine() -> None:
    label_emissions = _final_action_label_emission_targets(EXECUTABLE_CODE_SCAN_SCOPE)

    assert label_emissions == ALLOWED_FINAL_ACTION_LABEL_EMISSION_TARGETS


def test_model_llm_output_surfaces_do_not_emit_final_action_labels() -> None:
    assert MODEL_LLM_OUTPUT_SCAN_SCOPE == (
        PACKAGE_ROOT / "models",
        PACKAGE_ROOT / "features" / "text.py",
        PACKAGE_ROOT / "features" / "sec.py",
    )
    for path in MODEL_LLM_OUTPUT_SCAN_SCOPE:
        assert path.exists()

    label_emissions = _final_action_label_emission_targets(MODEL_LLM_OUTPUT_SCAN_SCOPE)
    action_assignments = _action_assignment_targets(MODEL_LLM_OUTPUT_SCAN_SCOPE)
    output_key_targets = _payload_key_targets(
        MODEL_LLM_OUTPUT_SCAN_SCOPE,
        set(PROHIBITED_MODEL_LLM_FINAL_ACTION_OUTPUT_KEYS),
    )

    assert label_emissions == set()
    assert action_assignments == set()
    assert output_key_targets == set()


def test_model_llm_output_guard_flags_final_action_label_bypasses(tmp_path: Path) -> None:
    adapter_path = tmp_path / "src" / "quant_research" / "models" / "unsafe_llm.py"
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_path.write_text(
        "\n".join(
            (
                "def unsafe_generation_payload():",
                "    return {'trade_decision': 'BUY'}",
                "",
                "def unsafe_prediction_frame(frame):",
                "    return frame.assign(action='SELL')",
            )
        )
    )

    output_key_targets = _payload_key_targets(
        (adapter_path,),
        set(PROHIBITED_MODEL_LLM_FINAL_ACTION_OUTPUT_KEYS),
    )
    label_emissions = _final_action_label_emission_targets((adapter_path,))

    assert output_key_targets == {f"{adapter_path}:2", f"{adapter_path}:5"}
    assert label_emissions == {f"{adapter_path}:5"}


def test_final_action_label_emission_guard_flags_bypasses_outside_engine(
    tmp_path: Path,
) -> None:
    bypass_path = tmp_path / "bypass_signal_engine.py"
    bypass_path.write_text(
        "\n".join(
            (
                "def bypass_with_assignment(frame):",
                "    frame['action'] = 'BUY'",
                "    return frame",
                "",
                "def bypass_with_assign(frame):",
                "    return frame.assign(action='SELL')",
                "",
                "def bypass_with_payload():",
                "    return {'action': 'HOLD'}",
            )
        )
    )

    label_emissions = _final_action_label_emission_targets((bypass_path,))
    bypass_location = str(bypass_path)

    assert label_emissions == {
        f"{bypass_location}:2",
        f"{bypass_location}:6",
        f"{bypass_location}:9",
    }


def test_documented_emission_paths_use_deterministic_signal_sources() -> None:
    source_expectations = {
        "src/quant_research/pipeline.py": (
            "signals=backtest.signals",
            "result.signals",
            '"action_counts": action_counts',
        ),
        "app.py": (
            'result.signals["date"].max()',
            'result.signals[result.signals["date"] == latest_date]',
            'no_model_proxy_signal.get("action_counts", {})',
        ),
        "src/quant_research/validation/report_generation.py": (
            "deterministic_signal_outputs: pd.DataFrame",
            '_coerce_frame(deterministic_signal_outputs, "deterministic_signal_outputs")',
            '_value_count_rows(output, "action", "action")',
        ),
        "scripts/run_backtest_validation.py": (
            "result.backtest.signals.to_csv",
            "deterministic_signal_outputs=result.backtest.signals",
        ),
        "src/quant_research/validation/gate.py": (
            'signal_metrics.get("action_counts")',
            "Buy / Sell / Hold",
        ),
        "src/quant_research/validation/report_renderer.py": (
            "render_structured_report",
            "write_structured_report_artifact",
        ),
    }

    for relative_path, snippets in source_expectations.items():
        source = _source_text(relative_path)
        for snippet in snippets:
            assert snippet in source, f"{relative_path} missing provenance snippet: {snippet}"


def test_beginner_dashboard_prediction_action_fallback_is_non_rendered_and_documented() -> None:
    dashboard_source = _source_text("src/quant_research/dashboard/beginner.py")
    streamlit_source = _source_text("src/quant_research/dashboard/streamlit.py")
    inventory = _source_text("docs/final-action-label-inventory.md")

    assert "if not result.signals.empty:" in dashboard_source
    assert "if not result.predictions.empty:" in dashboard_source
    assert '"raw_signal_visible": False' in dashboard_source
    assert "raw_signal" not in streamlit_source
    assert "defensive `result.predictions[\"action\"]` fallback" in inventory
    assert "non-emission compatibility fallback" in inventory
