from __future__ import annotations

import importlib.util
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from quant_research.backtest.metrics import PerformanceMetrics
from quant_research.pipeline import PipelineConfig, run_research_pipeline
from quant_research.signals.engine import DeterministicSignalEngine, SignalEngineConfig
from quant_research.validation import (
    ARTIFACT_MANIFEST_REQUIRED_METADATA_FIELDS,
    ARTIFACT_MANIFEST_SCHEMA_ID,
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    CANONICAL_REPORT_REQUIRED_INPUT_SECTIONS,
    CANONICAL_REPORT_REQUIRED_RESULT_SECTIONS,
    COMPLETED_RUN_REPORT_SCHEMA_ID,
    COMPLETED_RUN_REPORT_SCHEMA_VERSION,
    ReportDataSource,
    UniverseSnapshot,
    build_canonical_report_input_contract,
    build_canonical_report_metadata,
    build_completed_validation_backtest_report,
    load_artifact_manifest_json,
    render_completed_validation_backtest_report,
    validate_artifact_manifest_schema,
    write_completed_validation_backtest_report_artifacts,
)

REPORT_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "report_generation"
FUTURE_LEAKAGE_REPORT_INDICATORS = (
    "raw_expected_return",
    "expected_return",
    "predicted_return",
    "predicted_volatility",
    "downside_quantile",
    "model_prediction_timestamp",
    "prediction_timestamp",
    "text_availability_timestamp",
    "forward_return_1",
    "forward_return_5",
)
# Guardrail vocabulary: these strings are forbidden in rendered report payloads.
TRADE_ORDER_LANGUAGE = (
    "place_order",
    "submit_order",
    "create_order",
    "send_order",
    "broker",
    "execution_mode",
    "live_trade",
    "execute trade",
    "place trade",
    "submit trade",
    "send order",
    "매수 주문",
    "매도 주문",
    "주문 전송",
)


def _load_backtest_validation_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_backtest_validation.py"
    spec = importlib.util.spec_from_file_location("run_backtest_validation", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_completed_run_report_builds_signal_performance_risk_and_period_sections() -> None:
    report = build_completed_validation_backtest_report(
        metadata=_metadata(),
        deterministic_signal_outputs=_signals(),
        backtest_results=_equity_curve(),
        performance_metrics=_metrics(),
        walk_forward_validation_metrics=_walk_forward_summary(),
        system_validity_gate={
            "system_validity_status": "pass",
            "strategy_candidate_status": "warning",
            "system_validity_pass": True,
            "strategy_pass": False,
            "hard_fail": False,
            "warning": True,
            "official_message": "strategy candidate requires review",
            "metrics": {"positive_fold_ratio": 0.5},
        },
        report_path="reports/run_001/canonical_run_report.md",
    )

    assert report["schema_id"] == COMPLETED_RUN_REPORT_SCHEMA_ID
    assert report["schema_version"] == COMPLETED_RUN_REPORT_SCHEMA_VERSION
    assert report["identity"]["experiment_id"] == "stage1_exp"
    assert report["report_path"] == "reports/run_001/canonical_run_report.md"

    period = report["validation_period_metadata"]
    assert period["train_start"] == "2022-01-03"
    assert period["test_end"] == "2025-02-28"
    assert period["evaluation_start"] == "2024-12-02"
    assert period["evaluation_end"] == "2025-02-28"
    assert period["oos_fold_count"] == 2
    assert period["target_column"] == "forward_return_20"
    assert period["embargo_periods"] == 20
    assert period["non_overlapping_or_horizon_consistent"] is True

    signals = report["deterministic_signal_summary"]
    assert signals["row_count"] == 4
    assert signals["unique_tickers"] == 2
    assert signals["latest_signal_date"] == "2025-02-28"
    assert signals["average_signal_score"] == 0.35
    assert signals["llm_makes_trading_decisions"] is False
    assert signals["model_predictions_are_order_signals"] is False
    assert {"action": "BUY", "count": 2} in signals["action_counts"]

    performance = report["performance_metrics"]
    assert performance["return_basis"] == "cost_adjusted_return"
    assert performance["net_cagr"] == 0.12
    assert performance["benchmark_cagr"] == 0.08
    assert performance["excess_return"] == 0.04
    assert performance["observations"] == 2

    risk = report["risk_metrics"]
    assert risk["max_drawdown"] == -0.08
    assert risk["average_daily_turnover"] == 0.18
    assert risk["max_symbol_weight"] == 0.1
    assert risk["max_sector_weight"] == 0.22
    assert risk["max_observed_holdings"] == 20
    assert risk["risk_stop_triggered"] is False
    assert {row["risk_check"] for row in risk["risk_rows"]} >= {
        "average_daily_turnover",
        "max_drawdown",
    }

    walk_forward = report["walk_forward_validation_metrics"]
    assert walk_forward["fold_count"] == 2
    assert walk_forward["oos_fold_count"] == 2
    assert walk_forward["positive_fold_ratio"] == 0.5
    assert walk_forward["mean_rank_ic"] == 0.01

    gate = report["system_validity_gate"]
    assert gate["system_validity_status"] == "pass"
    assert gate["strategy_candidate_status"] == "warning"
    assert gate["official_message"] == "strategy candidate requires review"
    manifest = report["artifact_manifest"]
    assert manifest["reproducible_input_metadata_required"] is True
    assert manifest["schema_id"] == ARTIFACT_MANIFEST_SCHEMA_ID
    assert manifest["schema_version"] == ARTIFACT_MANIFEST_SCHEMA_VERSION
    assert manifest["manifest_id"] == "stage1_exp:run_001:artifact_manifest"
    assert manifest["experiment_id"] == "stage1_exp"
    assert manifest["run_id"] == "run_001"
    assert manifest["system_validity_status"] == "pass"
    assert manifest["strategy_candidate_status"] == "warning"
    assert manifest["survivorship_bias_allowed"] is True
    assert "survivorship bias" in manifest["survivorship_bias_disclosure"]
    for field_name in ARTIFACT_MANIFEST_REQUIRED_METADATA_FIELDS:
        assert field_name in manifest


def test_completed_run_report_exposes_required_sections_metrics_and_rendered_content() -> None:
    gate_payload = {
        "system_validity_status": "pass",
        "strategy_candidate_status": "pass",
        "system_validity_pass": True,
        "strategy_pass": True,
        "hard_fail": False,
        "warning": False,
        "official_message": "canonical experiment passed pre-trading research checks",
        "metrics": {
            "oos_fold_count": 2,
            "mean_rank_ic": 0.012,
            "positive_fold_ratio": 0.67,
            "max_drawdown": -0.08,
            "average_daily_turnover": 0.18,
            "strategy_excess_return_vs_spy": 0.03,
            "strategy_excess_return_vs_equal_weight": 0.02,
            "proxy_ic_improvement": 0.015,
            "cost_bps": 5.0,
            "slippage_bps": 2.0,
        },
    }
    report = build_completed_validation_backtest_report(
        metadata=_metadata(),
        deterministic_signal_outputs=_signals(),
        backtest_results=_equity_curve(),
        performance_metrics=_metrics(),
        walk_forward_validation_metrics=_walk_forward_summary(),
        system_validity_gate=gate_payload,
        strategy_candidate_gate=gate_payload,
        artifact_manifest={
            "created_at": "2025-03-04T00:00:00+00:00",
            "result_sections_required": [
                "backtest_results",
                "walk_forward_validation_metrics",
                "risk_checks",
                "model_feature_summaries",
                "deterministic_signal_summary",
                "trade_cost_assumptions",
            ],
        },
        report_path="reports/run_001/canonical_run_report.md",
    )

    required_sections = {
        "identity",
        "universe",
        "period",
        "run_configuration",
        "data_provenance",
        "validation_period_metadata",
        "deterministic_signal_summary",
        "performance_metrics",
        "risk_metrics",
        "walk_forward_validation_metrics",
        "system_validity_gate",
        "strategy_candidate_gate",
        "artifact_manifest",
        "v1_scope_exclusions",
    }
    assert required_sections <= set(report)
    assert set(report["artifact_manifest"]["result_sections_required"]) == {
        "backtest_results",
        "walk_forward_validation_metrics",
        "risk_checks",
        "model_feature_summaries",
        "deterministic_signal_summary",
        "trade_cost_assumptions",
    }

    assert report["run_configuration"]["target_horizon"] == "forward_return_20"
    assert report["run_configuration"]["walk_forward_config"]["embargo_periods"] == 20
    assert report["run_configuration"]["benchmark_config"]["benchmark_ticker"] == "SPY"
    assert report["run_configuration"]["benchmark_config"]["equal_weight_universe_baseline"]
    assert report["validation_period_metadata"]["oos_fold_count"] == 2
    assert report["walk_forward_validation_metrics"]["positive_fold_ratio"] == 0.5
    assert report["walk_forward_validation_metrics"]["mean_rank_ic"] == 0.01
    assert report["performance_metrics"]["return_basis"] == "cost_adjusted_return"
    assert report["performance_metrics"]["benchmark_cagr"] == 0.08
    assert report["performance_metrics"]["excess_return"] == 0.04
    assert report["risk_metrics"]["max_drawdown"] == -0.08
    assert report["risk_metrics"]["average_daily_turnover"] == 0.18
    assert report["risk_metrics"]["max_symbol_weight"] == 0.1
    assert report["risk_metrics"]["max_sector_weight"] == 0.22
    assert report["system_validity_gate"]["metrics"]["strategy_excess_return_vs_spy"] == 0.03
    assert (
        report["strategy_candidate_gate"]["metrics"]["strategy_excess_return_vs_equal_weight"]
        == 0.02
    )
    assert report["strategy_candidate_gate"]["metrics"]["proxy_ic_improvement"] == 0.015

    markdown = render_completed_validation_backtest_report(report)
    html = render_completed_validation_backtest_report(report, output_format="html")

    for expected in (
        "## Universe",
        "## Run Configuration",
        "## Data Provenance",
        "## Validation Period Metadata",
        "## Deterministic Signal Summary",
        "## Performance Metrics",
        "## Risk Metrics",
        "## Walk Forward Validation Metrics",
        "## System Validity Gate",
        "## Strategy Candidate Gate",
        "## Artifact Manifest",
        "## V1 Scope Exclusions",
        "forward_return_20",
        "SPY",
        "Equal Weight Universe Baseline",
        "Positive Fold Ratio",
        "Strategy Excess Return Vs Equal Weight",
        "Proxy Ic Improvement",
        "real_trading_orders",
        "point_in_time_universe",
        "correlation_cluster_weight",
    ):
        assert expected in markdown

    for expected in (
        "<h2>Universe</h2>",
        "<h2>Run Configuration</h2>",
        "<h2>Validation Period Metadata</h2>",
        "<h2>System Validity Gate</h2>",
        "<h2>Strategy Candidate Gate</h2>",
        "<h2>Artifact Manifest</h2>",
        "forward_return_20",
        "Strategy Excess Return Vs Spy",
        "Proxy Ic Improvement",
    ):
        assert expected in html


def test_completed_run_report_rendering_keeps_canonical_section_order_and_headings() -> None:
    report = build_completed_validation_backtest_report(
        metadata=_metadata(),
        deterministic_signal_outputs=_signals(),
        backtest_results=_equity_curve(),
        performance_metrics=_metrics(),
        walk_forward_validation_metrics=_walk_forward_summary(),
        system_validity_gate={
            "system_validity_status": "pass",
            "strategy_candidate_status": "warning",
            "official_message": "review required",
        },
        report_path="reports/run_001/canonical_run_report.md",
    )

    markdown = render_completed_validation_backtest_report(report)
    html = render_completed_validation_backtest_report(report, output_format="html")

    ordered_markdown_sections = [
        "## Report Summary",
        "## Identity",
        "## Universe",
        "## Period",
        "## Run Configuration",
        "## Data Provenance",
        "## Result Schema Sections",
        "## Artifact Manifest",
        "## V1 Scope Exclusions",
        "## Report Input Contract",
        "## Metadata Schema Id",
        "## Metadata Schema Version",
        "## Report Path",
        "## Validation Period Metadata",
        "## Deterministic Signal Summary",
        "## Performance Metrics",
        "## Risk Metrics",
        "## Walk Forward Validation Metrics",
        "## System Validity Gate",
        "## Strategy Candidate Gate",
    ]
    positions = [markdown.index(section) for section in ordered_markdown_sections]

    assert positions == sorted(positions)
    assert markdown.count("## Report Summary") == 1
    assert markdown.count("## Artifact Manifest") == 1
    assert markdown.count("## System Validity Gate") == 1
    assert "| Field | Value |" in markdown
    assert "| Target Horizon | forward_return_20 |" in markdown
    assert "| Embargo Periods | 20 |" in markdown
    assert "| Average Daily Turnover | 0.1800 |" in markdown
    assert "| Max Drawdown | -0.0800 |" in markdown
    assert "<h2>Report Summary</h2>" in html
    assert "<h2>Deterministic Signal Summary</h2>" in html
    assert "<td>forward_return_20</td>" in html
    assert "<td>0.1800</td>" in html


def test_completed_run_report_structure_matches_declared_schema_and_input_contract() -> None:
    report = build_completed_validation_backtest_report(
        metadata=_metadata(),
        deterministic_signal_outputs=_signals(),
        backtest_results=_equity_curve(),
        performance_metrics=_metrics(),
        walk_forward_validation_metrics=_walk_forward_summary(),
        system_validity_gate={
            "system_validity_status": "pass",
            "strategy_candidate_status": "pass",
            "system_validity_pass": True,
            "strategy_pass": True,
            "hard_fail": False,
            "warning": False,
            "official_message": "passed",
            "metrics": {
                "oos_fold_count": 2,
                "mean_rank_ic": 0.012,
                "positive_fold_ratio": 0.67,
                "strategy_excess_return_vs_spy": 0.03,
                "strategy_excess_return_vs_equal_weight": 0.02,
                "proxy_ic_improvement": 0.015,
            },
        },
        report_path="reports/run_001/canonical_run_report.md",
    )
    input_contract = build_canonical_report_input_contract()
    generated_sections = {
        "deterministic_signal_summary": report["deterministic_signal_summary"],
        "performance_metrics": report["performance_metrics"],
        "risk_metrics": report["risk_metrics"],
        "walk_forward_validation_metrics": report["walk_forward_validation_metrics"],
        "system_validity_gate": report["system_validity_gate"],
        "strategy_candidate_gate": report["strategy_candidate_gate"],
    }
    serialized_generated_sections = json.dumps(generated_sections, ensure_ascii=False)

    assert set(report["required_result_sections"]) == set(
        CANONICAL_REPORT_REQUIRED_RESULT_SECTIONS
    )
    assert set(report["result_schema_sections"]) == set(
        CANONICAL_REPORT_REQUIRED_RESULT_SECTIONS
    )
    assert set(report["report_input_contract"]["required_sections"]) == set(
        CANONICAL_REPORT_REQUIRED_INPUT_SECTIONS
    )
    assert report["report_input_contract"] == input_contract
    assert report["artifact_manifest"]["reproducible_input_metadata_required"] is True
    assert report["artifact_manifest"]["schema_id"] == ARTIFACT_MANIFEST_SCHEMA_ID
    assert report["artifact_manifest"]["schema_version"] == ARTIFACT_MANIFEST_SCHEMA_VERSION
    assert report["artifact_manifest"]["required_metadata_fields"] == list(
        ARTIFACT_MANIFEST_REQUIRED_METADATA_FIELDS
    )
    assert report["artifact_manifest"]["manifest_schema"]["schema_id"] == (
        ARTIFACT_MANIFEST_SCHEMA_ID
    )
    assert len(report["artifact_manifest"]["config_hash"]) == 64
    assert len(report["artifact_manifest"]["universe_snapshot_hash"]) == 64
    assert len(report["artifact_manifest"]["feature_availability_cutoff_hash"]) == 64
    assert len(report["artifact_manifest"]["data_snapshot_hash"]) == 64
    assert set(report["artifact_manifest"]["result_sections_required"]) == set(
        CANONICAL_REPORT_REQUIRED_RESULT_SECTIONS
    )
    assert report["deterministic_signal_summary"]["signal_engine"] == (
        "deterministic_signal_engine"
    )
    assert report["deterministic_signal_summary"]["model_predictions_are_order_signals"] is False
    assert report["system_validity_gate"]["metrics"]["oos_fold_count"] == 2
    assert report["system_validity_gate"]["metrics"]["proxy_ic_improvement"] == 0.015
    assert report["strategy_candidate_gate"]["metrics"]["positive_fold_ratio"] == 0.67
    assert "raw_model_predictions" not in serialized_generated_sections
    assert "llm_trade_decisions" not in serialized_generated_sections
    assert "place_order" not in serialized_generated_sections


def test_lightweight_report_fixture_data_generates_deterministic_report_payload() -> None:
    inputs = _load_report_fixture_inputs()

    first_report = build_completed_validation_backtest_report(**inputs)
    second_report = build_completed_validation_backtest_report(**inputs)

    assert first_report == second_report
    assert first_report["identity"]["run_id"] == "fixture_run_001"
    assert first_report["deterministic_signal_summary"]["row_count"] == 4
    assert first_report["validation_period_metadata"]["oos_fold_count"] == 2
    assert first_report["walk_forward_validation_metrics"]["target_column"] == "forward_return_20"
    assert first_report["artifact_manifest"]["created_at"] == "2025-03-04T00:00:00+00:00"

    markdown = render_completed_validation_backtest_report(first_report)
    html = render_completed_validation_backtest_report(first_report, output_format="html")

    assert markdown == render_completed_validation_backtest_report(second_report)
    assert html == render_completed_validation_backtest_report(second_report, output_format="html")
    assert "fixture_run_001" in markdown
    assert "<h2>Walk Forward Validation Metrics</h2>" in html
    generated_sections = {
        "deterministic_signal_summary": first_report["deterministic_signal_summary"],
        "performance_metrics": first_report["performance_metrics"],
        "risk_metrics": first_report["risk_metrics"],
        "walk_forward_validation_metrics": first_report["walk_forward_validation_metrics"],
    }
    assert "raw_model_predictions" not in json.dumps(generated_sections, ensure_ascii=False)


def test_completed_run_report_is_generated_from_deterministic_signals_and_evaluation_outputs() -> None:
    signal_inputs = _signal_engine_inputs()
    deterministic_signals = DeterministicSignalEngine(
        SignalEngineConfig(covariance_aware_risk_enabled=False)
    ).generate(signal_inputs)
    deterministic_signals["signal_engine"] = "deterministic_signal_engine"
    deterministic_signals["cost_bps"] = 5.0
    deterministic_signals["slippage_bps"] = 2.0

    report = build_completed_validation_backtest_report(
        metadata=_metadata(),
        deterministic_signal_outputs=deterministic_signals,
        backtest_results=_equity_curve(),
        performance_metrics=_metrics(),
        walk_forward_validation_metrics=_walk_forward_summary(),
    )

    summary = report["deterministic_signal_summary"]
    assert summary["row_count"] == len(deterministic_signals)
    assert summary["signal_engine"] == "deterministic_signal_engine"
    assert summary["llm_makes_trading_decisions"] is False
    assert summary["model_predictions_are_order_signals"] is False
    assert {row["action"] for row in summary["action_counts"]} <= {"BUY", "SELL", "HOLD"}
    assert report["performance_metrics"]["return_basis"] == "cost_adjusted_return"
    assert report["walk_forward_validation_metrics"]["target_column"] == "forward_return_20"
    generated_sections = {
        "deterministic_signal_summary": report["deterministic_signal_summary"],
        "performance_metrics": report["performance_metrics"],
        "risk_metrics": report["risk_metrics"],
        "walk_forward_validation_metrics": report["walk_forward_validation_metrics"],
    }
    serialized_sections = json.dumps(generated_sections, ensure_ascii=False)
    assert "raw_model_predictions" not in serialized_sections
    assert "adapter_predictions" not in serialized_sections
    assert "llm_trade_decisions" not in serialized_sections


def test_completed_run_report_result_sections_exclude_future_leakage_indicators() -> None:
    signal_inputs = _signal_engine_inputs()
    deterministic_signals = DeterministicSignalEngine(
        SignalEngineConfig(covariance_aware_risk_enabled=False)
    ).generate(signal_inputs)
    deterministic_signals["signal_engine"] = "deterministic_signal_engine"

    report = build_completed_validation_backtest_report(
        metadata=_metadata(),
        deterministic_signal_outputs=deterministic_signals,
        backtest_results=_equity_curve(),
        performance_metrics=_metrics(),
        walk_forward_validation_metrics=_walk_forward_summary(),
    )

    operator_result_sections = {
        "deterministic_signal_summary": report["deterministic_signal_summary"],
        "performance_metrics": report["performance_metrics"],
        "risk_metrics": report["risk_metrics"],
        "system_validity_gate": report["system_validity_gate"],
        "strategy_candidate_gate": report["strategy_candidate_gate"],
    }
    serialized_sections = json.dumps(
        operator_result_sections,
        ensure_ascii=False,
        sort_keys=True,
    ).lower()
    rendered_signal_section = _rendered_markdown_section(
        render_completed_validation_backtest_report(report),
        "## Deterministic Signal Summary",
    ).lower()

    for indicator in FUTURE_LEAKAGE_REPORT_INDICATORS:
        assert indicator not in serialized_sections
        assert indicator not in rendered_signal_section


def test_completed_run_report_markdown_and_html_exclude_trade_order_language() -> None:
    report = build_completed_validation_backtest_report(
        metadata=_metadata(),
        deterministic_signal_outputs=_signals(),
        backtest_results=_equity_curve(),
        performance_metrics=_metrics(),
        walk_forward_validation_metrics=_walk_forward_summary(),
    )

    rendered_report = "\n".join(
        [
            render_completed_validation_backtest_report(report),
            render_completed_validation_backtest_report(report, output_format="html"),
        ]
    ).lower()

    assert "real_trading_orders" in rendered_report
    assert report["deterministic_signal_summary"]["llm_makes_trading_decisions"] is False
    assert report["deterministic_signal_summary"]["model_predictions_are_order_signals"] is False
    for phrase in TRADE_ORDER_LANGUAGE:
        assert phrase.lower() not in rendered_report


def test_completed_run_report_rejects_future_data_leakage_in_signal_inputs() -> None:
    leaky_signals = _signals()
    leaky_signals["text_availability_timestamp"] = pd.to_datetime(
        [
            "2025-01-31 12:00:00+00:00",
            "2025-01-31 12:00:00+00:00",
            "2025-03-01 00:00:00+00:00",
            "2025-02-28 12:00:00+00:00",
        ],
        utc=True,
    )

    with pytest.raises(
        ValueError,
        match="text_availability_timestamp.*unavailable at feature date",
    ):
        build_completed_validation_backtest_report(
            metadata=_metadata(),
            deterministic_signal_outputs=leaky_signals,
            backtest_results=_equity_curve(),
            performance_metrics=_metrics(),
            walk_forward_validation_metrics=_walk_forward_summary(),
        )


def test_completed_run_report_rejects_future_model_prediction_timestamp() -> None:
    leaky_signals = _signals()
    leaky_signals["model_prediction_timestamp"] = pd.to_datetime(
        [
            "2025-01-31 12:00:00+00:00",
            "2025-01-31 12:00:00+00:00",
            "2025-03-01 00:00:00+00:00",
            "2025-02-28 12:00:00+00:00",
        ],
        utc=True,
    )

    with pytest.raises(
        ValueError,
        match="model_prediction_timestamp.*later than signal date",
    ):
        build_completed_validation_backtest_report(
            metadata=_metadata(),
            deterministic_signal_outputs=leaky_signals,
            backtest_results=_equity_curve(),
            performance_metrics=_metrics(),
            walk_forward_validation_metrics=_walk_forward_summary(),
        )


def test_completed_run_report_renders_and_writes_artifacts(tmp_path) -> None:
    report = build_completed_validation_backtest_report(
        metadata=_metadata(),
        deterministic_signal_outputs=_signals(),
        backtest_results=_equity_curve(),
        performance_metrics=_metrics(),
        walk_forward_validation_metrics=_walk_forward_summary(),
    )

    markdown = render_completed_validation_backtest_report(report)
    html = render_completed_validation_backtest_report(report, output_format="html")
    artifacts = write_completed_validation_backtest_report_artifacts(report, tmp_path)

    assert markdown.startswith("# Canonical Experiment Report")
    assert "## Validation Period Metadata" in markdown
    assert "## Deterministic Signal Summary" in markdown
    assert "## Performance Metrics" in markdown
    assert "## Risk Metrics" in markdown
    assert "<h2>Validation Period Metadata</h2>" in html
    assert set(artifacts) == {
        "json",
        "markdown",
        "html",
        "json_sha256",
        "markdown_sha256",
        "html_sha256",
    }
    assert (tmp_path / "canonical_run_report.json").exists()
    assert (tmp_path / "canonical_run_report.md").read_text(encoding="utf-8").startswith(
        "# Canonical Experiment Report"
    )
    assert (tmp_path / "canonical_run_report.html").read_text(encoding="utf-8").startswith(
        "<!doctype html>"
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "canonical_run_report.html",
        "canonical_run_report.json",
        "canonical_run_report.md",
    ]
    rendered_payload = "\n".join(
        [
            (tmp_path / "canonical_run_report.json").read_text(encoding="utf-8"),
            (tmp_path / "canonical_run_report.md").read_text(encoding="utf-8"),
            (tmp_path / "canonical_run_report.html").read_text(encoding="utf-8"),
        ]
    ).lower()
    assert "place_order" not in rendered_payload
    assert "submit_order" not in rendered_payload
    assert "create_order" not in rendered_payload
    assert "broker" not in rendered_payload
    assert "execution_mode" not in rendered_payload
    assert "live_trade" not in rendered_payload


def test_successful_backtest_validation_save_outputs_writes_canonical_report_artifacts(
    tmp_path,
) -> None:
    config = PipelineConfig(
        tickers=["SPY", "AAPL", "MSFT"],
        data_mode="synthetic",
        start=date(2022, 1, 3),
        end=date(2025, 1, 3),
        train_periods=60,
        test_periods=15,
        top_n=2,
        native_tabular_isolation=False,
    )
    result = run_research_pipeline(config)
    out_dir = tmp_path / "nested" / "validation_run"

    manifest = _load_backtest_validation_script().save_outputs(result, out_dir, config)

    assert set(manifest) >= {
        "validity_gate_json",
        "validity_gate_markdown",
        "canonical_report_json",
        "canonical_report_markdown",
        "canonical_report_html",
        "canonical_report_json_sha256",
        "canonical_report_markdown_sha256",
        "canonical_report_html_sha256",
        "artifact_manifest",
    }
    assert (out_dir / "market_data.csv").exists()
    assert (out_dir / "features.csv").exists()
    assert (out_dir / "signals.csv").exists()
    assert (out_dir / "equity_curve.csv").exists()
    assert (out_dir / "pipeline_config.json").exists()
    assert (out_dir / "canonical_metadata.json").exists()
    assert (out_dir / "universe_snapshot.json").exists()
    assert (out_dir / "feature_availability_cutoff.json").exists()
    assert (out_dir / "validity_gate.json").exists()
    assert (out_dir / "validity_report.md").exists()
    assert (out_dir / "canonical_run_report.json").exists()
    assert (out_dir / "canonical_run_report.md").exists()
    assert (out_dir / "canonical_run_report.html").exists()
    assert (out_dir / "artifact_manifest.json").exists()

    artifact_manifest = load_artifact_manifest_json(manifest["artifact_manifest"])
    validate_artifact_manifest_schema(artifact_manifest)
    assert artifact_manifest["experiment_id"] == "stage1_canonical_experiment"
    assert artifact_manifest["report_path"] == str(out_dir / "canonical_run_report.md")
    assert artifact_manifest["system_validity_status"] != "not_evaluated"
    assert artifact_manifest["strategy_candidate_status"] != "not_evaluated"
    artifact_ids = {
        artifact["artifact_id"] for artifact in artifact_manifest["artifacts"]
    }
    assert {
        "market_data",
        "model_feature_matrix",
        "pipeline_config",
        "canonical_metadata",
        "universe_snapshot",
        "feature_availability_cutoff",
        "model_predictions",
        "deterministic_signals",
        "equity_curve",
        "walk_forward_summary",
        "performance_metrics",
        "validity_gate",
        "report:canonical_run_report",
    } <= artifact_ids

    report = pd.read_json(manifest["canonical_report_json"], typ="series").to_dict()
    assert report["schema_id"] == COMPLETED_RUN_REPORT_SCHEMA_ID
    assert report["report_path"] == str(out_dir / "canonical_run_report.md")
    assert report["system_validity_gate"]
    assert report["strategy_candidate_gate"]
    assert report["deterministic_signal_summary"]["llm_makes_trading_decisions"] is False


def _metadata():
    return build_canonical_report_metadata(
        experiment_id="stage1_exp",
        run_id="run_001",
        universe_snapshot=UniverseSnapshot.from_tickers(
            ["AAPL", "MSFT"],
            experiment_id="stage1_exp",
            snapshot_date=date(2022, 1, 3),
        ),
        start_date=date(2022, 1, 3),
        end_date=date(2025, 3, 3),
        data_sources=(
            ReportDataSource(
                source_id="prices",
                provider="synthetic",
                dataset="ohlcv",
                as_of_date=date(2025, 3, 3),
                available_at=datetime(2025, 3, 3, tzinfo=UTC),
            ),
        ),
        feature_availability_cutoff={
            "price": "date <= t",
            "news_text": "published_at <= t",
            "sec_filing": "accepted_at <= t",
        },
        created_at=datetime(2025, 3, 4, tzinfo=UTC),
    )


def _signals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-01-31", "2025-01-31", "2025-02-28", "2025-02-28"]
            ),
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "action": ["BUY", "HOLD", "BUY", "SELL"],
            "signal_score": [0.5, 0.1, 0.6, 0.2],
            "model_confidence": [0.7, 0.6, 0.8, 0.5],
            "risk_metric_penalty": [0.0, 0.1, 0.0, 0.2],
            "signal_engine": ["deterministic_signal_engine"] * 4,
        }
    )


def _signal_engine_inputs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-01-31", "2025-01-31", "2025-02-28", "2025-02-28"]
            ),
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "expected_return": [0.03, 0.001, 0.04, 0.03],
            "predicted_volatility": [0.02, 0.03, 0.02, 0.02],
            "downside_quantile": [-0.01, -0.02, -0.01, -0.01],
            "sentiment_score": [0.3, 0.1, 0.4, -0.5],
            "text_risk_score": [0.1, 0.2, 0.1, 0.9],
            "sec_risk_flag": [0.0, 0.0, 0.0, 0.0],
            "sec_risk_flag_20d": [0.0, 0.0, 0.0, 0.0],
            "news_negative_ratio": [0.1, 0.1, 0.1, 0.1],
            "liquidity_score": [20.0, 20.0, 20.0, 20.0],
            "model_confidence": [0.7, 0.6, 0.8, 0.5],
            "model_prediction_timestamp": pd.to_datetime(
                [
                    "2025-01-31 12:00:00+00:00",
                    "2025-01-31 12:00:00+00:00",
                    "2025-02-28 12:00:00+00:00",
                    "2025-02-28 12:00:00+00:00",
                ],
                utc=True,
            ),
            "text_availability_timestamp": pd.to_datetime(
                [
                    "2025-01-31 11:00:00+00:00",
                    "2025-01-31 11:00:00+00:00",
                    "2025-02-28 11:00:00+00:00",
                    "2025-02-28 11:00:00+00:00",
                ],
                utc=True,
            ),
        }
    )


def _equity_curve() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-12-02", "2025-02-28"]),
            "cost_adjusted_return": [0.03, 0.04],
            "benchmark_return": [0.01, 0.02],
            "turnover": [0.15, 0.21],
            "position_count": [18, 20],
            "max_position_weight": [0.09, 0.10],
            "max_sector_exposure": [0.18, 0.22],
            "risk_stop_active": [False, False],
        }
    )


def _walk_forward_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": [0, 1],
            "is_oos": [True, True],
            "target_column": ["forward_return_20", "forward_return_20"],
            "prediction_horizon_periods": [20, 20],
            "train_start": pd.to_datetime(["2022-01-03", "2022-04-04"]),
            "train_end": pd.to_datetime(["2023-01-03", "2023-04-04"]),
            "test_start": pd.to_datetime(["2024-12-02", "2025-01-02"]),
            "test_end": pd.to_datetime(["2025-01-31", "2025-02-28"]),
            "purge_periods": [20, 20],
            "embargo_periods": [20, 20],
            "fold_rank_ic": [0.04, -0.02],
        }
    )


def _metrics() -> PerformanceMetrics:
    return PerformanceMetrics(
        cagr=0.12,
        annualized_volatility=0.16,
        sharpe=0.75,
        max_drawdown=-0.08,
        hit_rate=0.55,
        turnover=0.18,
        exposure=0.95,
        benchmark_cagr=0.08,
        excess_return=0.04,
        gross_cagr=0.14,
        gross_cumulative_return=0.18,
        net_cagr=0.12,
        net_cumulative_return=0.15,
        benchmark_cost_adjusted_cagr=0.08,
        benchmark_cost_adjusted_cumulative_return=0.10,
        transaction_cost_return=0.01,
        slippage_cost_return=0.004,
        total_cost_return=0.014,
        average_portfolio_volatility_estimate=0.13,
        max_portfolio_volatility_estimate=0.19,
        max_position_weight=0.10,
        max_sector_exposure=0.22,
        max_position_risk_contribution=0.18,
        position_sizing_validation_pass_rate=1.0,
        position_sizing_validation_status="pass",
    )


def _load_report_fixture_inputs() -> dict[str, object]:
    metadata_payload = json.loads((REPORT_FIXTURE_DIR / "metadata.json").read_text())
    metrics_payload = json.loads(
        (REPORT_FIXTURE_DIR / "performance_metrics.json").read_text()
    )
    return {
        "metadata": _metadata_from_fixture(metadata_payload),
        "deterministic_signal_outputs": pd.read_csv(
            REPORT_FIXTURE_DIR / "signals.csv",
            parse_dates=["date"],
        ),
        "backtest_results": pd.read_csv(
            REPORT_FIXTURE_DIR / "equity_curve.csv",
            parse_dates=["date"],
        ),
        "performance_metrics": PerformanceMetrics(**metrics_payload),
        "walk_forward_validation_metrics": pd.read_csv(
            REPORT_FIXTURE_DIR / "walk_forward_summary.csv",
            parse_dates=["train_start", "train_end", "test_start", "test_end"],
        ),
        "system_validity_gate": json.loads(
            (REPORT_FIXTURE_DIR / "system_validity_gate.json").read_text()
        ),
        "artifact_manifest": json.loads(
            (REPORT_FIXTURE_DIR / "artifact_manifest.json").read_text()
        ),
        "report_path": "reports/fixture_run_001/canonical_run_report.md",
    }


def _metadata_from_fixture(payload: dict[str, object]):
    data_sources = tuple(
        ReportDataSource(
            source_id=str(source["source_id"]),
            provider=str(source["provider"]),
            dataset=str(source["dataset"]),
            as_of_date=date.fromisoformat(str(source["as_of_date"])),
            available_at=datetime.fromisoformat(str(source["available_at"])),
        )
        for source in payload["data_sources"]
    )
    return build_canonical_report_metadata(
        experiment_id=str(payload["experiment_id"]),
        run_id=str(payload["run_id"]),
        universe_snapshot=UniverseSnapshot.from_tickers(
            list(payload["universe"]),
            experiment_id=str(payload["experiment_id"]),
            snapshot_date=date.fromisoformat(str(payload["start_date"])),
        ),
        start_date=date.fromisoformat(str(payload["start_date"])),
        end_date=date.fromisoformat(str(payload["end_date"])),
        data_sources=data_sources,
        feature_availability_cutoff=dict(payload["feature_availability_cutoff"]),
        created_at=datetime.fromisoformat(str(payload["created_at"])),
    )


def _rendered_markdown_section(markdown: str, heading: str) -> str:
    start = markdown.index(heading)
    next_heading = markdown.find("\n## ", start + len(heading))
    if next_heading == -1:
        return markdown[start:]
    return markdown[start:next_heading]
