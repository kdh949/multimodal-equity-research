from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime

import pytest

from quant_research.validation import (
    ARTIFACT_MANIFEST_REQUIRED_METADATA_FIELDS,
    ARTIFACT_MANIFEST_SCHEMA_ID,
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    CANONICAL_REPORT_REQUIRED_INPUT_SECTIONS,
    CANONICAL_REPORT_REQUIRED_METADATA_SECTIONS,
    CANONICAL_REPORT_REQUIRED_RESULT_SECTIONS,
    CANONICAL_REPORT_SCHEMA_ID,
    CANONICAL_REPORT_SCHEMA_VERSION,
    DEFAULT_PURGE_EMBARGO_WALK_FORWARD_CONFIG,
    REPORT_INPUT_CONTRACT_SCHEMA_ID,
    REPORT_INPUT_CONTRACT_SCHEMA_VERSION,
    REPORT_INPUT_PROHIBITED_FIELD_NAMES,
    REPORT_INPUT_PROHIBITED_SECTION_IDS,
    CanonicalArtifactManifestSchema,
    CanonicalReportInputContract,
    CanonicalReportMetadata,
    ReportDataProvenance,
    ReportDataSource,
    ReportDeterministicSignalSummarySchema,
    ReportIdentity,
    ReportInputSectionContract,
    ReportMetricField,
    ReportModelFeatureSummariesSchema,
    ReportPeriod,
    ReportRiskChecksSchema,
    ReportRunConfiguration,
    ReportTradeCostAssumptionsSchema,
    ReportUniverseMetadata,
    UniverseSnapshot,
    build_artifact_manifest_schema,
    build_canonical_report_input_contract,
    build_canonical_report_metadata,
    build_report_backtest_results_schema,
    build_report_deterministic_signal_summary_schema,
    build_report_model_feature_summaries_schema,
    build_report_risk_checks_schema,
    build_report_trade_cost_assumptions_schema,
    build_report_walk_forward_validation_metrics_schema,
    render_structured_report,
    render_structured_report_html,
    render_structured_report_markdown,
    write_structured_report_artifact,
)


def test_canonical_report_metadata_serializes_identity_universe_period_config_and_provenance() -> None:
    metadata = _canonical_metadata()

    payload = metadata.to_dict()

    assert payload["schema_id"] == CANONICAL_REPORT_SCHEMA_ID
    assert payload["schema_version"] == CANONICAL_REPORT_SCHEMA_VERSION
    assert payload["required_metadata_sections"] == list(
        CANONICAL_REPORT_REQUIRED_METADATA_SECTIONS
    )
    assert payload["required_result_sections"] == list(
        CANONICAL_REPORT_REQUIRED_RESULT_SECTIONS
    )
    assert payload["identity"]["experiment_id"] == "stage1_exp"
    assert payload["identity"]["run_id"] == "run_001"
    assert payload["identity"]["report_type"] == "canonical_experiment_report"
    assert payload["universe"]["universe"] == ["AAPL", "MSFT"]
    assert payload["universe"]["universe_snapshot"]["selection_count"] == 150
    assert payload["universe"]["survivorship_bias_allowed"] is True
    assert "survivorship bias" in payload["universe"]["survivorship_bias_disclosure"]
    assert payload["period"]["start_date"] == "2022-01-03"
    assert payload["period"]["end_date"] == "2025-01-03"
    assert payload["run_configuration"]["target_horizon"] == "forward_return_20"
    assert payload["run_configuration"]["diagnostic_horizons"] == [
        "forward_return_1",
        "forward_return_5",
    ]
    assert payload["run_configuration"]["walk_forward_config"]["train_periods"] == 252
    assert payload["run_configuration"]["walk_forward_config"]["test_periods"] == 60
    assert payload["run_configuration"]["walk_forward_config"]["embargo_periods"] == 20
    assert payload["run_configuration"]["portfolio_constraints"]["max_holdings"] == 20
    assert payload["run_configuration"]["portfolio_constraints"]["max_symbol_weight"] == 0.10
    assert payload["run_configuration"]["portfolio_constraints"]["max_sector_weight"] == 0.30
    assert payload["run_configuration"]["transaction_costs"]["scenarios"][0]["cost_bps"] == 5.0
    assert (
        payload["run_configuration"]["transaction_costs"]["scenarios"][0]["slippage_bps"]
        == 2.0
    )
    assert payload["run_configuration"]["benchmark_config"]["benchmark_ticker"] == "SPY"
    assert payload["run_configuration"]["benchmark_config"]["equal_weight_universe_baseline"]
    assert payload["data_provenance"]["feature_availability_cutoff"] == {
        "price": "date <= t",
        "news_text": "published_at <= t",
        "sec_filing": "accepted_at <= t",
    }
    assert payload["data_provenance"]["structured_text_feature_fields"] == [
        "sentiment_score",
        "event_tag",
        "risk_flag",
        "confidence",
        "summary_ref",
    ]
    assert payload["artifact_manifest"]["manifest_required"] is True
    assert payload["artifact_manifest"]["schema_id"] == ARTIFACT_MANIFEST_SCHEMA_ID
    assert payload["artifact_manifest"]["schema_version"] == ARTIFACT_MANIFEST_SCHEMA_VERSION
    assert payload["artifact_manifest"]["schema"] == build_artifact_manifest_schema()
    assert payload["artifact_manifest"]["required_metadata_fields"] == list(
        ARTIFACT_MANIFEST_REQUIRED_METADATA_FIELDS
    )
    assert payload["artifact_manifest"]["result_sections_required"] == [
        "backtest_results",
        "walk_forward_validation_metrics",
        "risk_checks",
        "model_feature_summaries",
        "deterministic_signal_summary",
        "trade_cost_assumptions",
    ]
    input_contract = payload["report_input_contract"]
    assert input_contract["schema_id"] == REPORT_INPUT_CONTRACT_SCHEMA_ID
    assert input_contract["schema_version"] == REPORT_INPUT_CONTRACT_SCHEMA_VERSION
    assert input_contract["required_sections"] == list(
        CANONICAL_REPORT_REQUIRED_INPUT_SECTIONS
    )
    assert input_contract["consumes_deterministic_signal_outputs"] is True
    assert input_contract["consumes_validation_backtest_evaluation_metrics"] is True
    assert input_contract["consumes_model_predictions_directly"] is False
    assert input_contract["model_predictions_are_order_signals"] is False
    assert input_contract["llm_makes_trading_decisions"] is False
    assert "predictions" in input_contract["prohibited_input_sections"]
    assert "expected_return" in input_contract["prohibited_field_names"]
    report_inputs = input_contract["input_sections"]
    assert set(report_inputs) == set(CANONICAL_REPORT_REQUIRED_INPUT_SECTIONS)
    signal_inputs = report_inputs["deterministic_signal_outputs"]
    assert signal_inputs["sample_alignment_key"] == ["date", "ticker"]
    assert "signal_score" in signal_inputs["required_fields"]
    assert "action" in signal_inputs["required_fields"]
    assert "expected_return" not in signal_inputs["required_fields"]
    assert "predicted_volatility" not in signal_inputs["required_fields"]
    assert "model_prediction_timestamp" not in signal_inputs["optional_fields"]
    backtest_schema = payload["result_schema_sections"]["backtest_results"]
    assert backtest_schema["section_id"] == "backtest_results"
    assert backtest_schema["sample_alignment_key"] == ["date"]
    assert backtest_schema["required_fields"] == [
        "date",
        "portfolio_return",
        "cost_adjusted_return",
        "benchmark_return",
        "equal_weight_return",
        "turnover",
        "holdings_count",
        "max_symbol_weight",
        "max_sector_weight",
        "cost_bps",
        "slippage_bps",
    ]
    assert backtest_schema["validation_rules"]["long_only"] is True
    assert backtest_schema["validation_rules"]["benchmark_ticker"] == "SPY"
    assert backtest_schema["validation_rules"]["average_daily_turnover_limit"] == 0.25
    walk_forward_schema = payload["result_schema_sections"][
        "walk_forward_validation_metrics"
    ]
    assert walk_forward_schema["section_id"] == "walk_forward_validation_metrics"
    assert walk_forward_schema["sample_alignment_key"] == ["fold"]
    assert walk_forward_schema["validation_rules"]["target_column"] == "forward_return_20"
    assert walk_forward_schema["validation_rules"]["embargo_periods"] == 20
    assert walk_forward_schema["validation_rules"]["minimum_oos_fold_count"] == 2
    risk_schema = payload["result_schema_sections"]["risk_checks"]
    assert risk_schema["section_id"] == "risk_checks"
    assert risk_schema["sample_alignment_key"] == ["date", "risk_check"]
    assert risk_schema["validation_rules"]["long_only"] is True
    assert risk_schema["validation_rules"]["max_holdings"] == 20
    assert risk_schema["validation_rules"]["max_symbol_weight"] == 0.10
    assert risk_schema["validation_rules"]["max_sector_weight"] == 0.30
    assert risk_schema["validation_rules"]["correlation_cluster_weight_in_v1"] is False
    feature_schema = payload["result_schema_sections"]["model_feature_summaries"]
    assert feature_schema["section_id"] == "model_feature_summaries"
    assert feature_schema["sample_alignment_key"] == [
        "date",
        "ticker",
        "feature_name",
        "source_adapter",
    ]
    assert "feature_available_at" in feature_schema["required_fields"]
    assert "sentiment_score" in feature_schema["optional_fields"]
    assert "chronos_expected_return" in feature_schema["optional_fields"]
    assert feature_schema["validation_rules"]["structured_text_features"] == [
        "sentiment_score",
        "event_tag",
        "risk_flag",
        "confidence",
        "summary_ref",
    ]
    assert feature_schema["validation_rules"]["optional_model_adapters"] == [
        "chronos",
        "granite_ttm",
        "finbert",
        "finma",
        "fingpt",
        "ollama",
    ]
    assert feature_schema["validation_rules"]["adapter_outputs_are_order_signals"] is False
    assert feature_schema["validation_rules"]["feature_availability_cutoff_required"]
    signal_schema = payload["result_schema_sections"]["deterministic_signal_summary"]
    assert signal_schema["section_id"] == "deterministic_signal_summary"
    assert signal_schema["sample_alignment_key"] == ["date", "ticker"]
    assert "signal_score" in signal_schema["required_fields"]
    assert "action" in signal_schema["required_fields"]
    assert signal_schema["validation_rules"]["signal_engine"] == "deterministic_signal_engine"
    assert signal_schema["validation_rules"]["allowed_actions"] == ["BUY", "SELL", "HOLD"]
    assert signal_schema["validation_rules"]["llm_makes_trading_decisions"] is False
    assert signal_schema["validation_rules"]["model_predictions_are_order_signals"] is False
    cost_schema = payload["result_schema_sections"]["trade_cost_assumptions"]
    assert cost_schema["section_id"] == "trade_cost_assumptions"
    assert cost_schema["sample_alignment_key"] == ["scenario_id"]
    assert cost_schema["validation_rules"]["baseline_cost_bps"] == 5.0
    assert cost_schema["validation_rules"]["baseline_slippage_bps"] == 2.0
    assert cost_schema["validation_rules"]["baseline_total_cost_bps"] == 7.0
    assert cost_schema["validation_rules"]["turnover_sensitivity_required"] is True
    assert "real_trading_orders" in payload["v1_scope_exclusions"]
    assert "point_in_time_universe" in payload["v1_scope_exclusions"]
    assert "correlation_cluster_weight" in payload["v1_scope_exclusions"]


def test_report_run_configuration_rejects_invalid_canonical_embargo() -> None:
    invalid_walk_forward = replace(DEFAULT_PURGE_EMBARGO_WALK_FORWARD_CONFIG, embargo_periods=0)

    with pytest.raises(ValueError, match="walk_forward_config is not system valid"):
        ReportRunConfiguration(walk_forward_config=invalid_walk_forward)


def test_report_run_configuration_rejects_llm_trading_decisions() -> None:
    with pytest.raises(ValueError, match="LLM adapters must not make trading decisions"):
        ReportRunConfiguration(llm_makes_trading_decisions=True)


def test_canonical_report_input_contract_consumes_signals_and_metrics_not_predictions() -> None:
    contract = CanonicalReportInputContract()
    payload = build_canonical_report_input_contract()

    assert payload["schema_id"] == REPORT_INPUT_CONTRACT_SCHEMA_ID
    assert payload["required_sections"] == list(CANONICAL_REPORT_REQUIRED_INPUT_SECTIONS)
    assert payload["consumes_model_predictions_directly"] is False
    assert payload["prohibited_input_sections"] == list(
        REPORT_INPUT_PROHIBITED_SECTION_IDS
    )
    assert payload["prohibited_field_names"] == list(
        REPORT_INPUT_PROHIBITED_FIELD_NAMES
    )
    assert "predictions" not in payload["input_sections"]
    assert contract.section("deterministic_signal_outputs").required_field_names() == (
        "date",
        "ticker",
        "signal_score",
        "action",
        "signal_engine",
        "cost_bps",
        "slippage_bps",
        "risk_metric_penalty",
    )
    assert "backtest_evaluation_metrics" in payload["input_sections"]
    assert "walk_forward_validation_metrics" in payload["input_sections"]


def test_artifact_manifest_schema_defines_required_reproducibility_metadata() -> None:
    schema = CanonicalArtifactManifestSchema()
    payload = schema.to_dict()

    assert payload["schema_id"] == ARTIFACT_MANIFEST_SCHEMA_ID
    assert payload["schema_version"] == ARTIFACT_MANIFEST_SCHEMA_VERSION
    assert payload["required_metadata_fields"] == list(
        ARTIFACT_MANIFEST_REQUIRED_METADATA_FIELDS
    )
    for field_name in (
        "experiment_id",
        "run_id",
        "config_hash",
        "universe_snapshot_hash",
        "feature_availability_cutoff_hash",
        "data_snapshot_hash",
        "system_validity_status",
        "strategy_candidate_status",
        "survivorship_bias_disclosure",
        "v1_scope_exclusions",
    ):
        assert field_name in payload["required_metadata_fields"]
    assert payload["required_artifact_fields"] == [
        "artifact_id",
        "artifact_type",
        "path",
        "created_at",
        "content_hash",
    ]
    assert payload["validation_rules"]["content_hash_required"] is True
    assert payload["validation_rules"]["universe_snapshot_hash_required"] is True

    with pytest.raises(ValueError, match="schema_version"):
        CanonicalArtifactManifestSchema(schema_version="canonical_artifact_manifest.v0")

    with pytest.raises(ValueError, match="missing metadata fields"):
        CanonicalArtifactManifestSchema(required_metadata_fields=("experiment_id",))


def test_canonical_report_input_contract_rejects_prediction_sections_and_fields() -> None:
    with pytest.raises(ValueError, match="prohibited"):
        ReportInputSectionContract(
            section_id="predictions",
            description="raw model predictions",
            row_grain="date,ticker",
            required_fields=(ReportMetricField("date", "datetime64[ns]"),),
        )

    with pytest.raises(ValueError, match="prohibited model prediction fields"):
        ReportInputSectionContract(
            section_id="deterministic_signal_outputs",
            description="invalid direct model prediction input",
            row_grain="date,ticker",
            required_fields=(
                ReportMetricField("date", "datetime64[ns]"),
                ReportMetricField("expected_return", "float"),
            ),
        )

    contract = CanonicalReportInputContract()
    valid_payload = {section: [] for section in CANONICAL_REPORT_REQUIRED_INPUT_SECTIONS}
    contract.validate_payload_sections(valid_payload)
    with pytest.raises(ValueError, match="raw model prediction sections"):
        contract.validate_payload_sections({**valid_payload, "model_predictions": []})
    with pytest.raises(ValueError, match="missing required sections"):
        contract.validate_payload_sections({"deterministic_signal_outputs": []})


def test_report_period_requires_at_least_three_years() -> None:
    with pytest.raises(ValueError, match="at least min_history_years"):
        ReportPeriod(start_date=date(2024, 1, 3), end_date=date(2025, 1, 3))


def test_report_data_provenance_rejects_committed_artifacts() -> None:
    with pytest.raises(ValueError, match="must not be committed"):
        ReportDataProvenance(
            data_sources=(_source(),),
            feature_availability_cutoff={"price": "date <= t"},
            model_artifacts_committed=True,
        )


def test_canonical_report_metadata_requires_matching_experiment_and_snapshot_start() -> None:
    snapshot = _snapshot()
    with pytest.raises(ValueError, match="experiment_id must match"):
        CanonicalReportMetadata(
            identity=ReportIdentity("other_exp", "run_001"),
            universe=ReportUniverseMetadata(snapshot),
            period=ReportPeriod(start_date=date(2022, 1, 3), end_date=date(2025, 1, 3)),
            run_configuration=ReportRunConfiguration(),
            data_provenance=_provenance(),
        )

    with pytest.raises(ValueError, match="snapshot_date must equal"):
        CanonicalReportMetadata(
            identity=ReportIdentity("stage1_exp", "run_001"),
            universe=ReportUniverseMetadata(snapshot),
            period=ReportPeriod(start_date=date(2022, 1, 4), end_date=date(2025, 1, 4)),
            run_configuration=ReportRunConfiguration(),
            data_provenance=_provenance(),
        )


def test_build_canonical_report_metadata_helper_uses_schema_models() -> None:
    metadata = build_canonical_report_metadata(
        experiment_id="stage1_exp",
        run_id="run_001",
        universe_snapshot=_snapshot(),
        start_date=date(2022, 1, 3),
        end_date=date(2025, 1, 3),
        data_sources=(_source(),),
        feature_availability_cutoff={"price": "date <= t"},
        created_at=datetime(2025, 1, 4, tzinfo=UTC),
    )

    assert metadata.identity.report_id == "stage1_exp:run_001:canonical_experiment_report"
    assert metadata.period.min_history_years == 3
    assert metadata.run_configuration.target_horizon == "forward_return_20"


def test_report_result_schema_builders_define_backtest_and_walk_forward_metrics() -> None:
    backtest_schema = build_report_backtest_results_schema()
    walk_forward_schema = build_report_walk_forward_validation_metrics_schema()
    risk_schema = build_report_risk_checks_schema()
    feature_schema = build_report_model_feature_summaries_schema()
    signal_schema = build_report_deterministic_signal_summary_schema()
    cost_schema = build_report_trade_cost_assumptions_schema()

    assert backtest_schema["section_id"] == "backtest_results"
    assert backtest_schema["row_grain"] == "evaluation date"
    assert "cost_adjusted_return" in backtest_schema["required_fields"]
    assert "equal_weight_return" in backtest_schema["required_fields"]
    assert backtest_schema["validation_rules"]["cost_bps"] == 5.0
    assert backtest_schema["validation_rules"]["slippage_bps"] == 2.0

    assert walk_forward_schema["section_id"] == "walk_forward_validation_metrics"
    assert "fold_rank_ic" in walk_forward_schema["required_fields"]
    assert "positive_fold_ratio" in walk_forward_schema["optional_fields"]
    assert walk_forward_schema["validation_rules"][
        "embargo_zero_for_forward_return_20_is_hard_fail"
    ]

    assert risk_schema["section_id"] == "risk_checks"
    assert "average_daily_turnover" in risk_schema["required_fields"]
    assert "max_drawdown" in risk_schema["required_fields"]
    assert risk_schema["validation_rules"]["average_daily_turnover_limit"] == 0.25

    assert feature_schema["section_id"] == "model_feature_summaries"
    assert "source_adapter" in feature_schema["required_fields"]
    assert "adapter_output_value" in feature_schema["optional_fields"]
    assert "fingpt_risk_flag" in feature_schema["optional_fields"]
    assert feature_schema["validation_rules"]["rules_fallback_allowed"]
    assert feature_schema["validation_rules"]["llm_makes_trading_decisions"] is False

    assert signal_schema["section_id"] == "deterministic_signal_summary"
    assert "expected_return" in signal_schema["required_fields"]
    assert "signal_score" in signal_schema["required_fields"]
    assert signal_schema["validation_rules"]["requires_system_validity_gate_pass"]

    assert cost_schema["section_id"] == "trade_cost_assumptions"
    assert "cost_bps" in cost_schema["required_fields"]
    assert "slippage_bps" in cost_schema["required_fields"]
    assert cost_schema["validation_rules"]["costs_and_slippage_required"]


def test_structured_report_renderer_outputs_human_readable_markdown_sections_and_tables() -> None:
    metadata = _canonical_metadata()

    markdown = render_structured_report_markdown(metadata)

    assert markdown.startswith("# Canonical Experiment Report")
    assert "## Report Summary" in markdown
    assert "| Experiment ID | stage1_exp |" in markdown
    assert "| Run ID | run_001 |" in markdown
    assert "| Target Horizon | forward_return_20 |" in markdown
    assert "## Identity" in markdown
    assert "## Universe" in markdown
    assert "## Run Configuration" in markdown
    assert "### Walk Forward Config" in markdown
    assert "### Feature Availability Cutoff" in markdown
    assert "## Result Schema Sections" in markdown
    assert "| backtest_results | evaluation date | date |" in markdown
    assert "| walk_forward_validation_metrics | walk-forward fold plus aggregate OOS summary | fold |" in markdown
    assert "## Artifact Manifest" in markdown
    assert "### Required Result Sections" in markdown
    assert "- backtest_results" in markdown
    assert "## V1 Scope Exclusions" in markdown
    assert "- real_trading_orders" in markdown
    assert "- correlation_cluster_weight" in markdown


def test_structured_report_renderer_outputs_standalone_html_with_escaped_tables() -> None:
    payload = _canonical_metadata().to_dict()
    payload["identity"]["run_id"] = "run_<001>"

    html = render_structured_report_html(payload)

    assert html.startswith("<!doctype html>")
    assert "<h1>Canonical Experiment Report</h1>" in html
    assert "<h2>Report Summary</h2>" in html
    assert "<table>" in html
    assert "<td>Experiment ID</td>" in html
    assert "<td>stage1_exp</td>" in html
    assert "<td>run_&lt;001&gt;</td>" in html
    assert "<h2>Result Schema Sections</h2>" in html
    assert "<td>backtest_results</td>" in html
    assert "<h2>V1 Scope Exclusions</h2>" in html
    assert "<li>point_in_time_universe</li>" in html


def test_structured_report_renderer_dispatches_format_and_writes_artifact(tmp_path) -> None:
    metadata = _canonical_metadata()

    markdown = render_structured_report(metadata, output_format="markdown")
    html = render_structured_report(metadata, output_format="html")
    markdown_path = write_structured_report_artifact(metadata, tmp_path / "canonical_report.md")
    html_path = write_structured_report_artifact(metadata, tmp_path / "canonical_report.html")

    assert markdown.startswith("# Canonical Experiment Report")
    assert html.startswith("<!doctype html>")
    assert markdown_path.read_text(encoding="utf-8").startswith("# Canonical Experiment Report")
    assert html_path.read_text(encoding="utf-8").startswith("<!doctype html>")
    with pytest.raises(ValueError, match="output_format"):
        render_structured_report(metadata, output_format="json")  # type: ignore[arg-type]


def test_report_schema_sections_reject_invalid_versions_and_alignment_keys() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        ReportRiskChecksSchema(schema_version="report_risk_checks.v0")

    with pytest.raises(ValueError, match="sample_alignment_key"):
        ReportRiskChecksSchema(sample_alignment_key=("date",))

    with pytest.raises(ValueError, match="schema_version"):
        ReportDeterministicSignalSummarySchema(
            schema_version="report_deterministic_signal_summary.v0"
        )

    with pytest.raises(ValueError, match="sample_alignment_key"):
        ReportDeterministicSignalSummarySchema(sample_alignment_key=("date",))

    with pytest.raises(ValueError, match="schema_version"):
        ReportModelFeatureSummariesSchema(
            schema_version="report_model_feature_summaries.v0"
        )

    with pytest.raises(ValueError, match="sample_alignment_key"):
        ReportModelFeatureSummariesSchema(sample_alignment_key=("date", "ticker"))

    with pytest.raises(ValueError, match="schema_version"):
        ReportTradeCostAssumptionsSchema(
            schema_version="report_trade_cost_assumptions.v0"
        )

    with pytest.raises(ValueError, match="sample_alignment_key"):
        ReportTradeCostAssumptionsSchema(sample_alignment_key=("date",))


def _canonical_metadata() -> CanonicalReportMetadata:
    return build_canonical_report_metadata(
        experiment_id="stage1_exp",
        run_id="run_001",
        universe_snapshot=_snapshot(),
        start_date=date(2022, 1, 3),
        end_date=date(2025, 1, 3),
        data_sources=(_source(),),
        feature_availability_cutoff={
            "price": "date <= t",
            "news_text": "published_at <= t",
            "sec_filing": "accepted_at <= t",
        },
        created_at=datetime(2025, 1, 4, tzinfo=UTC),
    )


def _snapshot() -> UniverseSnapshot:
    return UniverseSnapshot.from_tickers(
        ["AAPL", "MSFT"],
        experiment_id="stage1_exp",
        snapshot_date=date(2022, 1, 3),
    )


def _source() -> ReportDataSource:
    return ReportDataSource(
        source_id="prices",
        provider="yfinance",
        dataset="ohlcv",
        as_of_date=date(2025, 1, 3),
        available_at=datetime(2025, 1, 3, 22, tzinfo=UTC),
    )


def _provenance() -> ReportDataProvenance:
    return ReportDataProvenance(
        data_sources=(_source(),),
        feature_availability_cutoff={"price": "date <= t"},
    )
