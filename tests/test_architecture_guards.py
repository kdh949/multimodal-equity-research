from __future__ import annotations

from pathlib import Path


def test_repository_has_no_live_trading_or_order_execution_modules() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "quant_research"
    forbidden_path_terms = {"broker", "brokers", "order", "orders", "execution", "trading"}

    paths = {
        part.lower()
        for path in root.rglob("*.py")
        for part in path.relative_to(root).with_suffix("").parts
    }

    assert paths.isdisjoint(forbidden_path_terms)


def test_source_does_not_call_broker_order_apis() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "quant_research"
    forbidden_phrases = (
        "place_order",
        "submit_order",
        "create_order",
        "market_order",
        "limit_order",
    )

    source = "\n".join(path.read_text() for path in root.rglob("*.py"))

    assert all(phrase not in source for phrase in forbidden_phrases)
