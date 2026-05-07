from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_heavy_quant_validation.py"
SPEC = importlib.util.spec_from_file_location("run_heavy_quant_validation", SCRIPT_PATH)
assert SPEC is not None
heavy_runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = heavy_runner
SPEC.loader.exec_module(heavy_runner)


def test_request_budget_respects_provider_min_interval() -> None:
    clock = {"value": 100.0}
    sleeps: list[float] = []

    def now() -> float:
        return clock["value"]

    def sleeper(seconds: float) -> None:
        sleeps.append(seconds)
        clock["value"] += seconds

    budget = heavy_runner.RequestBudget({"sec": 2.0}, now=now, sleeper=sleeper)

    budget.acquire("sec")
    budget.acquire("sec")

    assert sleeps == [pytest.approx(0.5)]
    assert budget.request_count("sec") == 2


def test_market_cache_hit_avoids_downloader_call(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    artifact = raw_dir / "market_history.parquet"
    artifact.write_text("cached artifact placeholder", encoding="utf-8")
    request = heavy_runner.BackfillRequest(
        tickers=("AAPL", "MSFT"),
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
    )
    expected = {"phase": "market", "request": request.to_manifest_key()}
    manifest = {
        "schema_version": "heavy_backfill_phase.v1",
        "expected": expected,
        "rows": 123,
        "artifact_path": str(artifact),
        "artifact_sha256": "placeholder",
    }
    (raw_dir / "heavy_backfill_market.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    def fail_downloader(_: object) -> object:
        raise AssertionError("downloader must not run for cache hits")

    result = heavy_runner.ensure_market_data(
        raw_dir,
        request,
        heavy_runner.RequestBudget({"yfinance_market": 100.0}),
        downloader=fail_downloader,
    )

    assert result.cache_hit is True
    assert result.rows == 123
    assert result.request_count == 0


def test_heavy_report_includes_every_spec_test_id() -> None:
    checks = heavy_runner.build_spec_checks(None, {})

    assert tuple(check.test_id for check in checks) == heavy_runner.SPEC_TEST_IDS
    assert len(checks) == 19


def test_unimplemented_capabilities_are_added_to_todo() -> None:
    checks = heavy_runner.build_spec_checks(None, {})
    todos = {check.test_id: check.todo for check in checks if check.todo is not None}

    for test_id in {"DATA-01", "FACTOR-01", "OVERFIT-01", "EXEC-01", "PAPER-01", "MON-01", "CAP-01"}:
        assert test_id in todos
        assert todos[test_id] is not None
        assert todos[test_id].missing_capability
        assert todos[test_id].recommended_implementation


def test_generated_report_manifest_hashes_validate(tmp_path: Path) -> None:
    checks = heavy_runner.build_spec_checks(None, {})

    outputs = heavy_runner.write_heavy_report(tmp_path, checks, run_id="unit_test_run")

    manifest_path = Path(outputs["heavy_validation_manifest"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_hashes = {item["artifact_id"]: item["sha256"] for item in payload["artifacts"]}

    for artifact_id, output_key in {
        "final_report_markdown": "final_report_markdown",
        "final_report_json": "final_report_json",
    }.items():
        artifact_path = Path(outputs[output_key])
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        assert artifact_hashes[artifact_id] == digest
        assert outputs[f"{output_key}_sha256"] == digest

    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert outputs["heavy_validation_manifest_sha256"] == manifest_digest
