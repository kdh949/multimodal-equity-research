from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from streamlit.testing.v1 import AppTest

import quant_research.pipeline as pipeline
from quant_research.backtest.engine import BacktestResult
from quant_research.backtest.metrics import PerformanceMetrics
from quant_research.pipeline import PipelineResult


def _load_app_module() -> object:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("quant_research_streamlit_app", app_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Streamlit app module from {app_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_stub_result() -> PipelineResult:
    dates = pd.date_range("2026-01-01", periods=6, freq="D")
    focus_ticker = "AAPL"
    market_data = pd.DataFrame(
        {
            "date": dates.repeat(2),
            "ticker": [focus_ticker, focus_ticker] * len(dates),
            "close": [440, 441] * len(dates),
            "forward_return_1": [0.005, 0.004] * len(dates),
        }
    )
    news_features = pd.DataFrame(
        {
            "date": dates,
            "ticker": [focus_ticker] * len(dates),
            "news_article_count": [1] * len(dates),
            "news_sentiment_mean": [0.02] * len(dates),
            "news_negative_ratio": [0.0] * len(dates),
            "news_source_count": [1] * len(dates),
            "news_source_diversity": [1.0] * len(dates),
            "news_event_count": [0] * len(dates),
            "news_top_event": ["none"] * len(dates),
            "text_risk_score": [0.0] * len(dates),
            "news_confidence_mean": [0.6] * len(dates),
            "news_token_count_mean": [12.0] * len(dates),
            "news_text_length": [40.0] * len(dates),
            "news_full_text_available_ratio": [1.0] * len(dates),
            "news_recency_decay": [0.0] * len(dates),
            "news_staleness_days": [0.0] * len(dates),
            "news_coverage_5d": [3.0] * len(dates),
            "news_coverage_20d": [12.0] * len(dates),
        }
    )
    sec_features = pd.DataFrame(
        {
            "date": dates,
            "ticker": [focus_ticker] * len(dates),
            "sec_event_tag": ["none"] * len(dates),
            "sec_risk_flag": [0.0] * len(dates),
            "sec_risk_flag_20d": [0.0] * len(dates),
            "sec_event_confidence": [0.0] * len(dates),
            "sec_summary_ref": [""] * len(dates),
        }
    )
    predictions = pd.DataFrame(
        {
            "date": dates,
            "ticker": [focus_ticker] * len(dates),
            "action": ["HOLD"] * len(dates),
            "signal_score": [0.0] * len(dates),
            "expected_return": [0.01] * len(dates),
            "predicted_volatility": [0.02] * len(dates),
            "downside_quantile": [-0.015] * len(dates),
            "upside_quantile": [0.035] * len(dates),
            "text_risk_score": [0.0] * len(dates),
            "sec_risk_flag": [0.0] * len(dates),
            "sec_risk_flag_20d": [0.0] * len(dates),
            "text_risk_flag": [0.0] * len(dates),
            "sec_event_confidence": [0.0] * len(dates),
        }
    )
    signals = predictions.copy()
    features = market_data.copy()
    validation_summary = pd.DataFrame(
        {
            "fold": [0, 1],
            "fold_type": ["validation", "oos"],
            "is_oos": [False, True],
            "train_start": [dates[0], dates[1]],
            "train_end": [dates[2], dates[3]],
            "validation_start": [dates[3], pd.NaT],
            "validation_end": [dates[4], pd.NaT],
            "test_start": [dates[3], dates[4]],
            "test_end": [dates[4], dates[5]],
            "oos_test_start": [pd.NaT, dates[4]],
            "oos_test_end": [pd.NaT, dates[5]],
            "purge_periods": [20, 20],
            "purge_start": [dates[2], dates[3]],
            "purge_end": [dates[2], dates[3]],
            "purged_date_count": [20, 20],
            "purge_applied": [True, True],
            "embargo_periods": [20, 20],
            "embargo_start": [dates[4], dates[5]],
            "embargo_end": [dates[4], dates[5]],
            "embargoed_date_count": [20, 20],
            "embargo_applied": [True, True],
            "mae": [0.012, 0.011],
            "directional_accuracy": [0.60, 0.62],
            "oos_fold_count": [1, 1],
            "oos_start": [dates[4], dates[4]],
            "oos_end": [dates[5], dates[5]],
            "oos_rank_ic": [0.04, 0.04],
            "oos_rank_ic_positive_fold_ratio": [1.0, 1.0],
            "oos_rank_ic_count": [2, 2],
            "oos_prediction_count": [6, 6],
            "oos_labeled_prediction_count": [6, 6],
            "oos_mean_mae": [0.011, 0.011],
            "oos_mean_directional_accuracy": [0.62, 0.62],
            "oos_mean_information_coefficient": [0.05, 0.05],
            "walk_forward_fold_count": [2, 2],
            "walk_forward_mean_rank_ic": [0.035, 0.035],
            "walk_forward_positive_rank_ic_fold_ratio": [1.0, 1.0],
        }
    )
    ablation_summary = [
        {
            "scenario": "all_features",
            "kind": "signal",
            "cagr": 0.01,
            "sharpe": 0.3,
            "max_drawdown": -0.01,
            "excess_return": 0.008,
        },
        {
            "scenario": "price_only",
            "kind": "data_channel",
            "label": "Price only",
            "cagr": 0.015,
            "sharpe": 0.35,
            "max_drawdown": -0.015,
            "turnover": 0.08,
            "excess_return": 0.009,
            "validation_status": "pass",
            "validation_fold_count": 1,
            "validation_oos_fold_count": 1,
            "validation_metrics": {
                "mean_rank_ic": 0.03,
                "positive_fold_ratio": 0.7,
                "oos_rank_ic": 0.025,
            },
            "deterministic_signal_evaluation_metrics": {
                "cost_adjusted_cumulative_return": 0.015,
                "average_daily_turnover": 0.08,
            },
        },
        {
            "scenario": "no_model_proxy",
            "kind": "pipeline_control",
            "label": "No model proxy",
            "pipeline_controls": {
                "model_proxy": False,
                "cost": True,
                "slippage": True,
                "turnover": True,
            },
            "effective_cost_bps": 5.0,
            "effective_slippage_bps": 2.0,
            "cagr": 0.02,
            "sharpe": 0.4,
            "max_drawdown": -0.02,
            "turnover": 0.10,
            "excess_return": 0.01,
            "validation_status": "pass",
            "validation_fold_count": 1,
            "validation_oos_fold_count": 1,
            "deterministic_signal_evaluation_metrics": {
                "action_counts": {"BUY": 0, "SELL": 0, "HOLD": 6},
                "cost_adjusted_cumulative_return": 0.02,
                "average_daily_turnover": 0.10,
            },
        }
    ]
    backtest = BacktestResult(
        equity_curve=pd.DataFrame(
            {
                "date": dates,
                "equity": [1.0, 1.01, 1.02, 1.03, 1.04, 1.05],
                "benchmark_equity": [1.0, 1.0, 1.01, 1.01, 1.01, 1.02],
                "portfolio_return": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                "benchmark_return": [0.0, 0.001, 0.0, 0.0, 0.0, 0.001],
                "turnover": [0.0] * len(dates),
                "exposure": [0.2] * len(dates),
                "gross_return": [0.01] * len(dates),
                "return_date": dates,
                "portfolio_volatility_estimate": [0.015] * len(dates),
                "position_sizing_validation_status": ["pass"] * len(dates),
                "position_sizing_validation_rule": [
                    "post_cost_position_sizing_constraints_v1"
                ]
                * len(dates),
                "position_sizing_validation_reason": [
                    "risk_concentration_and_leverage_limits_passed_after_costs"
                ]
                * len(dates),
                "position_count": [1] * len(dates),
                "max_position_weight": [0.10] * len(dates),
                "max_sector_exposure": [0.20] * len(dates),
                "gross_exposure": [0.20] * len(dates),
                "net_exposure": [0.20] * len(dates),
                "max_position_risk_contribution": [1.0] * len(dates),
                "post_cost_validation_total_cost_return": [0.0007] * len(dates),
            }
        ),
        weights=pd.DataFrame(
            {
                "signal_date": pd.to_datetime(dates),
                "effective_date": pd.to_datetime(dates),
                "ticker": [focus_ticker] * len(dates),
                "weight": [0.0] * len(dates),
            }
        ),
        signals=signals,
        metrics=PerformanceMetrics(
            cagr=0.03,
            annualized_volatility=0.12,
            sharpe=0.9,
            max_drawdown=-0.02,
            hit_rate=0.58,
            turnover=0.0,
            exposure=0.2,
            benchmark_cagr=0.02,
            excess_return=0.01,
        ),
    )
    return PipelineResult(
        market_data=market_data,
        news_features=news_features,
        sec_features=sec_features,
        features=features,
        predictions=predictions,
        signals=signals,
        validation_summary=validation_summary,
        ablation_summary=ablation_summary,
        backtest=backtest,
    )


def test_transaction_cost_sensitivity_heatmap_figure_builds_from_summary() -> None:
    app_module = _load_app_module()
    summary = pd.DataFrame(
        {
            "scenario_id": ["canonical_costs", "high_costs", "tight_turnover_budget"],
            "total_cost_bps": [7.0, 15.0, 7.0],
            "average_daily_turnover_budget": [0.25, 0.25, 0.15],
            "cost_adjusted_cumulative_return": [0.04, 0.02, 0.035],
        }
    )

    figure = app_module._build_transaction_cost_sensitivity_figure(summary)

    assert figure is not None
    assert figure.data[0].type == "heatmap"
    assert list(figure.data[0].x) == ["7.0", "15.0"]
    assert list(figure.data[0].y) == ["15%", "25%"]
    assert "canonical_costs" in figure.data[0].text[1][0]


def test_report_only_research_metric_rows_mark_top_decile_as_non_decision_metadata() -> None:
    app_module = _load_app_module()
    report = SimpleNamespace(
        evidence={
            "top_decile_20d_excess_return": {
                "metric": "top_decile_20d_excess_return",
                "status": "report_only",
                "top_decile_20d_excess_return": 0.027,
                "target_column": "forward_return_20",
                "sample_scope": "oos_labeled_predictions",
                "report_only": True,
                "decision_use": "none",
                "reason": "report-only diagnostic; not used for scoring, action, ranking, thresholding, or gating",
            }
        }
    )

    rows = app_module._validity_report_only_research_metric_rows(report)

    assert rows == [
        {
            "metric": "top_decile_20d_excess_return",
            "status": "report_only",
            "value": 0.027,
            "target_column": "forward_return_20",
            "sample_scope": "oos_labeled_predictions",
            "report_only": True,
            "decision_use": "none",
            "reason": "report-only diagnostic; not used for scoring, action, ranking, thresholding, or gating",
        }
    ]


def test_gate_report_artifact_rows_include_canonical_gate_data() -> None:
    app_module = _load_app_module()
    report = SimpleNamespace(
        to_dict=lambda: {
            "system_validity_status": "pass",
            "strategy_candidate_status": "fail",
            "metrics": {"final_gate_decision": "FAIL"},
            "gate_results": {
                "system_validity_artifact_contract": {"status": "pass"},
                "deterministic_strategy_validity": {"status": "fail"},
            },
            "structured_pass_fail_reasons": [
                {"category": "gate", "rule": "deterministic_strategy_validity"}
            ],
            "structured_warnings": [{"gate": "turnover"}],
        }
    )

    payload = app_module._validity_gate_report_payload(report)
    rows = app_module._validity_gate_report_artifact_rows(report)

    assert payload["report_path"] == "reports/validity_report.md"
    assert payload["gate_report_data_included"] == {
        "gate_result_count": 2,
        "structured_reason_count": 1,
        "warning_count": 1,
        "includes_gate_results": True,
        "includes_system_validity_gate": True,
        "includes_strategy_candidate_gate": True,
    }
    assert [row["artifact"] for row in rows] == [
        "validity_gate.json",
        "validity_report.md",
    ]
    assert {row["format"] for row in rows} == {"json", "markdown"}
    assert all(row["includes_gate_results"] is True for row in rows)
    assert all(row["includes_system_validity_gate"] is True for row in rows)
    assert all(row["includes_strategy_candidate_gate"] is True for row in rows)


def test_report_approval_gate_allows_only_common_gate_pass() -> None:
    app_module = _load_app_module()
    report = SimpleNamespace(
        to_dict=lambda: {
            "system_validity_status": "pass",
            "strategy_candidate_status": "pass",
            "validity_gate_result_summary": {
                "final_gate_decision": "PASS",
                "final_status": "pass",
                "reason": "all required deterministic gate items passed",
            },
            "gate_results": {
                "system_validity_artifact_contract": {"status": "pass"},
                "deterministic_strategy_validity": {"status": "pass"},
            },
            "structured_pass_fail_reasons": [],
            "structured_warnings": [],
        }
    )

    approval = app_module._validity_report_approval_gate(report)

    assert approval["approval_status"] == "approved"
    assert approval["approval_allowed"] is True
    assert approval["final_gate_decision"] == "PASS"
    assert approval["final_status"] == "pass"


def test_report_approval_gate_blocks_non_pass_common_gate() -> None:
    app_module = _load_app_module()
    report = SimpleNamespace(
        to_dict=lambda: {
            "system_validity_status": "pass",
            "strategy_candidate_status": "warning",
            "validity_gate_result_summary": {
                "final_gate_decision": "WARN",
                "final_status": "warning",
                "reason": "all required deterministic gate items passed hard checks with warning item(s)",
            },
            "gate_results": {
                "system_validity_artifact_contract": {"status": "pass"},
                "deterministic_strategy_validity": {"status": "warning"},
            },
            "structured_pass_fail_reasons": [],
            "structured_warnings": [{"gate": "model_value"}],
        }
    )

    approval = app_module._validity_report_approval_gate(report)

    assert approval["approval_status"] == "blocked"
    assert approval["approval_allowed"] is False
    assert approval["final_gate_decision"] == "WARN"
    assert approval["final_status"] == "warning"
    assert "validation gate blocked final signal generation" in approval["reason"]


def test_user_gate_status_rows_render_structured_gate_failures() -> None:
    app_module = _load_app_module()
    report = SimpleNamespace(
        to_dict=lambda: {
            "system_validity_status": "pass",
            "strategy_candidate_status": "fail",
            "official_message": "시스템은 유효하지만 현재 전략 후보는 배포/사용 부적합",
            "validity_gate_result_summary": {
                "deterministic_gate": {
                    "reason": "deterministic strategy rule(s) failed: rank_ic"
                }
            },
            "structured_gate_failure_report": {
                "gates": [
                    {
                        "gate": "rank_ic",
                        "status": "fail",
                        "severity": "fail",
                        "top_reason_code": "positive_fold_ratio_below_minimum",
                        "top_reason": "positive fold ratio is below threshold",
                        "related_metrics": [
                            {
                                "metric": "positive_fold_ratio",
                                "value": 0.5,
                                "threshold": 0.65,
                                "operator": ">=",
                            }
                        ],
                    }
                ]
            },
            "gate_results": {"rank_ic": {"status": "fail"}},
            "structured_pass_fail_reasons": [],
            "structured_warnings": [],
        }
    )

    rows = app_module._validity_user_gate_status_rows(report)

    assert rows[0]["gate"] == "system_validity"
    assert rows[0]["display_status"] == "PASS"
    assert rows[0]["severity"] == "success"
    assert rows[1]["gate"] == "strategy_candidate"
    assert rows[1]["display_status"] == "FAIL"
    assert rows[1]["severity"] == "error"
    assert rows[1]["reason"] == "deterministic strategy rule(s) failed: rank_ic"
    assert rows[2] == {
        "scope": "rule",
        "gate": "rank_ic",
        "display_status": "FAIL",
        "severity": "error",
        "reason": "positive fold ratio is below threshold",
        "reason_code": "positive_fold_ratio_below_minimum",
        "metric": "positive_fold_ratio",
        "value": 0.5,
        "threshold": 0.65,
        "operator": ">=",
    }


class _FakeSensitivityMetricColumn:
    def __init__(self) -> None:
        self.metrics: list[tuple[str, object]] = []

    def metric(self, label: str, value: object) -> None:
        self.metrics.append((label, value))


class _FakeSensitivityStreamlit:
    def __init__(self) -> None:
        self.dataframes: list[pd.DataFrame] = []
        self.errors: list[str] = []
        self.infos: list[str] = []
        self.subheaders: list[str] = []

    def columns(self, count: int) -> list[_FakeSensitivityMetricColumn]:
        return [_FakeSensitivityMetricColumn() for _ in range(count)]

    def dataframe(self, value: pd.DataFrame, **_: object) -> None:
        self.dataframes.append(value)

    def error(self, value: str) -> None:
        self.errors.append(str(value))

    def info(self, value: str) -> None:
        self.infos.append(str(value))

    def subheader(self, value: str) -> None:
        self.subheaders.append(str(value))


def test_transaction_cost_sensitivity_renderer_surfaces_calculation_failure(
    monkeypatch,
) -> None:
    app_module = _load_app_module()
    fake_streamlit = _FakeSensitivityStreamlit()
    monkeypatch.setattr(app_module, "st", fake_streamlit)

    result = app_module._transaction_cost_sensitivity_error_result(
        RuntimeError("cost sensitivity failed"),
        pipeline.PipelineConfig(),
    )
    app_module._render_transaction_cost_sensitivity(result)

    assert fake_streamlit.errors
    assert "민감도 계산 실패" in fake_streamlit.errors[0]
    assert "cost sensitivity failed" in fake_streamlit.errors[0]
    assert fake_streamlit.infos == ["민감도 시나리오별 성과 표를 생성하지 못했습니다."]
    review_tables = [
        table for table in fake_streamlit.dataframes if "check" in table.columns
    ]
    assert review_tables
    calculation_row = review_tables[0].set_index("check").loc["calculation_errors"]
    assert calculation_row["status"] == "fail"
    assert calculation_row["value"] == 1


def _child_nodes(node: object) -> list[object]:
    children = getattr(node, "children", None)
    if isinstance(children, dict):
        return list(children.values())
    if isinstance(children, list):
        return children
    return []


def _collect_dataframes(node: object) -> list[object]:
    if node.__class__.__name__ == "Dataframe":
        return [node]

    tables: list[object] = []
    for child in _child_nodes(node):
        tables.extend(_collect_dataframes(child))
    return tables


def _is_tab_container(node: object) -> bool:
    return node.__class__.__name__ == "Block" and getattr(node, "type", None) == "tab_container"


def _has_evidence(captions: list[str], keys: tuple[str, ...]) -> bool:
    return any(any(key in line for key in keys) for line in captions)


def _run_fake_research_pipeline(
    captured: dict[str, object],
    config: pipeline.PipelineConfig,
) -> PipelineResult:
    captured["calls"] = captured.get("calls", 0) + 1
    captured["config"] = config
    return _build_stub_result()


def _assert_full_stack_defaults(
    app: object,
    *,
    data_mode_select_value: str = "synthetic",
    sentiment_select_value: str = "finbert",
    time_series_select_value: str = "proxy",
    filing_extractor_value: str = "fingpt",
) -> None:
    data_mode_select = next(node for node in app.selectbox if node.label == "Data mode")
    sentiment_select = next(node for node in app.selectbox if node.label == "Sentiment model")
    time_series_select = next(node for node in app.selectbox if node.label == "Time-series inference")
    filing_extractor_select = next(node for node in app.selectbox if node.label == "Filing extractor")

    assert data_mode_select.value == data_mode_select_value
    assert sentiment_select.value == sentiment_select_value
    assert time_series_select.value == time_series_select_value
    assert filing_extractor_select.value == filing_extractor_value
    assert any(node.label == "Use local filing LLM" and node.value is True for node in app.checkbox)


def _assert_beginner_dashboard_rendered(app: object, captured: dict[str, object]) -> None:
    data_mode_select = next(node for node in app.selectbox if node.label == "Data mode")
    sentiment_select = next(node for node in app.selectbox if node.label == "Sentiment model")
    time_series_select = next(node for node in app.selectbox if node.label == "Time-series inference")
    filing_extractor_select = next(node for node in app.selectbox if node.label == "Filing extractor")
    config = captured["config"]
    defaults = pipeline.PipelineConfig()
    assert isinstance(config, pipeline.PipelineConfig)
    assert config.sentiment_model == "finbert"
    assert config.filing_extractor_model == "fingpt"
    assert config.enable_local_filing_llm is True
    assert config.time_series_inference_mode == "proxy"
    assert config.covariance_aware_risk_enabled is defaults.covariance_aware_risk_enabled
    assert config.covariance_return_column == defaults.covariance_return_column
    assert config.portfolio_covariance_lookback == defaults.portfolio_covariance_lookback
    assert config.covariance_min_periods == defaults.covariance_min_periods
    assert config.average_daily_turnover_budget == defaults.average_daily_turnover_budget
    assert config.max_symbol_weight == defaults.max_symbol_weight
    assert config.max_sector_weight == defaults.max_sector_weight

    assert data_mode_select.value == "synthetic"
    assert sentiment_select.value == "finbert"
    assert time_series_select.value == "proxy"
    assert filing_extractor_select.value == "fingpt"
    markdown_values = [str(markdown.value) for markdown in app.markdown]
    caption_values = [str(caption.value) for caption in app.caption]
    metric_labels = {metric.label for metric in app.metric}

    assert any("Beginner Research Overview" in markdown for markdown in markdown_values)
    assert any(
        "연구용 리서치 화면이며 투자 권고가 아닙니다. 실거래 주문 기능은 제공하지 않습니다." in markdown
        for markdown in markdown_values
    )

    assert any("**방향성**" in markdown for markdown in markdown_values)
    assert any("**위험도**" in markdown for markdown in markdown_values)
    assert any("**공시 영향**" in markdown for markdown in markdown_values)
    assert any("**검증 신뢰도**" in markdown for markdown in markdown_values)

    assert _has_evidence(caption_values, ("expected_return", "downside_quantile"))
    assert _has_evidence(caption_values, ("predicted_volatility", "max_drawdown"))
    assert _has_evidence(caption_values, ("risk_flag", "event_tag", "confidence"))
    assert _has_evidence(caption_values, ("is_oos", "sharpe", "hit_rate"))

    assert any("Forecast Interval" in subheader.value for subheader in app.subheader)
    assert any("SEC Filing Impact" in subheader.value for subheader in app.subheader)
    assert any("Backtest Validation Snapshot" in subheader.value for subheader in app.subheader)
    assert any(
        "Covariance Risk and Post-Cost Sizing Validation" in subheader.value
        for subheader in app.subheader
    )
    assert any("Portfolio Risk Configuration" in subheader.value for subheader in app.subheader)
    assert any(
        "Validity Gate Cost-Adjusted Comparison" in subheader.value
        for subheader in app.subheader
    )
    assert any(
        "Three Strategy Performance and Cost Comparison" in subheader.value
        for subheader in app.subheader
    )
    assert any(
        "Validity Gate Side-by-Side Metrics" in subheader.value
        for subheader in app.subheader
    )
    assert any(
        "Validity Gate Baseline Comparisons" in subheader.value
        for subheader in app.subheader
    )
    assert any(
        "Validity Gate Baseline Inputs" in subheader.value
        for subheader in app.subheader
    )
    assert any("Report-Only Research Metrics" in subheader.value for subheader in app.subheader)
    assert any(
        "Validity Gate Model Comparison Results" in subheader.value
        for subheader in app.subheader
    )
    assert any(
        "Canonical Gate Report Artifacts" in subheader.value
        for subheader in app.subheader
    )
    assert any("User Gate Status" in subheader.value for subheader in app.subheader)
    assert any("Full Model Metrics" in subheader.value for subheader in app.subheader)
    assert any("Baseline Metrics" in subheader.value for subheader in app.subheader)
    assert any("Ablation Metrics" in subheader.value for subheader in app.subheader)
    assert any(
        "Structured Pass/Fail Reasons" in subheader.value
        for subheader in app.subheader
    )
    assert any("No-Model-Proxy Ablation" in subheader.value for subheader in app.subheader)
    assert any(
        "Walk-Forward Fold Periods and Purge/Embargo" in subheader.value
        for subheader in app.subheader
    )
    assert any("OOS Performance Summary" in subheader.value for subheader in app.subheader)
    assert any(
        "Transaction Cost and Turnover Sensitivity" in subheader.value
        for subheader in app.subheader
    )
    assert any(
        "Sensitivity Result Review Summary" in subheader.value
        for subheader in app.subheader
    )
    assert any(
        "Sensitivity Cost/Turnover Heatmap" in subheader.value
        for subheader in app.subheader
    )
    assert any("Validation Metrics" in markdown for markdown in markdown_values)
    assert {"Net CAGR", "Net Sharpe", "Net Max DD", "Hit Rate", "Exposure", "Turnover"}.issubset(
        metric_labels
    )
    risk_sizing_tables = [
        dataframe.value
        for dataframe in app.dataframe
        if {
            "portfolio_volatility_estimate",
            "position_sizing_validation_status",
            "position_sizing_validation_rule",
            "max_position_weight",
            "max_sector_exposure",
            "post_cost_validation_total_cost_return",
        }.issubset(dataframe.value.columns)
    ]
    assert risk_sizing_tables
    assert risk_sizing_tables[0]["position_sizing_validation_status"].eq("pass").all()
    assert (
        risk_sizing_tables[0]["position_sizing_validation_rule"]
        .eq("post_cost_position_sizing_constraints_v1")
        .all()
    )
    risk_config_tables = [
        dataframe.value
        for dataframe in app.dataframe
        if {
            "covariance_aware_risk_enabled",
            "covariance_return_column",
            "portfolio_covariance_lookback",
            "covariance_min_periods",
            "portfolio_volatility_limit",
            "max_symbol_weight",
            "max_sector_weight",
            "max_position_risk_contribution",
        }.issubset(dataframe.value.columns)
    ]
    assert risk_config_tables
    risk_config_row = risk_config_tables[0].iloc[0].to_dict()
    assert bool(risk_config_row["covariance_aware_risk_enabled"]) is True
    assert risk_config_row["covariance_return_column"] == defaults.covariance_return_column
    assert risk_config_row["portfolio_covariance_lookback"] == defaults.portfolio_covariance_lookback
    assert risk_config_row["covariance_min_periods"] == defaults.covariance_min_periods
    assert risk_config_row["max_symbol_weight"] == defaults.max_symbol_weight
    assert risk_config_row["max_sector_weight"] == defaults.max_sector_weight
    assert {"1d Diagnostic", "20d Required"}.issubset(metric_labels)
    horizon_tables = [
        dataframe.value
        for dataframe in app.dataframe
        if {"horizon", "label", "role", "status", "insufficient_data_status"}.issubset(
            dataframe.value.columns
        )
    ]
    assert horizon_tables
    assert {"positive_fold_ratio_threshold", "positive_fold_ratio_passed"}.issubset(
        horizon_tables[0].columns
    )
    horizon_rows = horizon_tables[0].set_index("horizon").to_dict("index")
    assert horizon_rows["1d"]["label"] == "diagnostic"
    assert horizon_rows["1d"]["role"] == "diagnostic"
    assert horizon_rows["1d"]["status"] == "insufficient_data"
    assert horizon_rows["1d"]["insufficient_data_status"] == "insufficient_data"
    assert horizon_rows["20d"]["label"] == "required"
    assert horizon_rows["20d"]["role"] == "decision"
    assert horizon_rows["20d"]["status"] == "insufficient_data"
    assert horizon_rows["20d"]["insufficient_data_status"] == "insufficient_data"
    report_only_research_tables = [
        dataframe.value
        for dataframe in app.dataframe
        if {
            "metric",
            "status",
            "value",
            "target_column",
            "sample_scope",
            "report_only",
            "decision_use",
        }.issubset(dataframe.value.columns)
        and "top_decile_20d_excess_return" in set(dataframe.value["metric"])
    ]
    assert report_only_research_tables
    assert report_only_research_tables[0].loc[0, "status"] in {"report_only", "not_evaluable"}
    assert "threshold" not in report_only_research_tables[0].columns
    assert "operator" not in report_only_research_tables[0].columns
    assert report_only_research_tables[0].loc[0, "decision_use"] == "none"
    assert bool(report_only_research_tables[0].loc[0, "report_only"]) is True
    comparison_tables = [
        dataframe.value
        for dataframe in app.dataframe
        if {"name", "role", "return_basis", "cagr"}.issubset(dataframe.value.columns)
    ]
    assert comparison_tables
    assert {"strategy", "SPY", "equal_weight"}.issubset(set(comparison_tables[0]["name"]))
    three_strategy_tables = [
        dataframe.value
        for dataframe in app.dataframe
        if {
            "strategy",
            "role",
            "return_basis",
            "cagr",
            "sharpe",
            "max_drawdown",
            "cost_adjusted_cumulative_return",
            "average_daily_turnover",
            "transaction_cost_return",
            "slippage_cost_return",
            "total_cost_return",
        }.issubset(dataframe.value.columns)
    ]
    assert three_strategy_tables
    three_strategy_rows = three_strategy_tables[0].set_index("strategy").to_dict("index")
    assert {"strategy", "SPY", "equal_weight"}.issubset(set(three_strategy_rows))
    assert three_strategy_rows["strategy"]["return_basis"] == "cost_adjusted_return"
    assert three_strategy_rows["strategy"]["average_daily_turnover"] is not None
    assert three_strategy_rows["strategy"]["total_cost_return"] is not None
    assert three_strategy_rows["SPY"]["cost_adjusted_cumulative_return"] is not None
    assert three_strategy_rows["equal_weight"]["cost_adjusted_cumulative_return"] is not None
    side_by_side_tables = [
        dataframe.value
        for dataframe in app.dataframe
        if {"metric", "metric_label", "strategy", "SPY", "equal_weight"}.issubset(
            dataframe.value.columns
        )
    ]
    assert side_by_side_tables
    side_by_side_rows = side_by_side_tables[0].set_index("metric").to_dict("index")
    assert {"cagr", "sharpe", "cost_adjusted_cumulative_return"}.issubset(
        set(side_by_side_rows)
    )
    assert side_by_side_rows["cagr"]["strategy"] is not None
    assert side_by_side_rows["cagr"]["SPY"] is not None
    assert side_by_side_rows["cagr"]["equal_weight"] is not None
    baseline_tables = [
        dataframe.value
        for dataframe in app.dataframe
        if {"name", "baseline_type", "return_basis", "cagr"}.issubset(dataframe.value.columns)
    ]
    assert baseline_tables
    assert {"SPY", "equal_weight"}.issubset(set(baseline_tables[0]["name"]))
    assert {"market_benchmark", "equal_weight_universe"}.issubset(
        set(baseline_tables[0]["baseline_type"])
    )
    baseline_input_tables = [
        dataframe.value
        for dataframe in app.dataframe
        if {"name", "baseline_type", "return_basis", "data_source"}.issubset(
            dataframe.value.columns
        )
    ]
    assert baseline_input_tables
    assert {"SPY", "equal_weight"}.issubset(set(baseline_input_tables[0]["name"]))
    assert {"benchmark_return_series", "equal_weight_baseline_return_series"}.issubset(
        set(baseline_input_tables[0]["data_source"])
    )
    model_comparison_tables = [
        dataframe.value
        for dataframe in app.dataframe
        if {
            "window_id",
            "metric",
            "baseline",
            "absolute_delta",
            "relative_delta",
            "pass_fail",
        }.issubset(dataframe.value.columns)
    ]
    assert model_comparison_tables
    assert {"no_model_proxy", "return_baseline_spy", "return_baseline_equal_weight"}.issubset(
        set(model_comparison_tables[0]["baseline"])
    )
    full_model_metric_tables = [
        dataframe.value
        for dataframe in app.dataframe
        if {"entity_id", "role", "status", "sharpe", "mean_rank_ic"}.issubset(
            dataframe.value.columns
        )
    ]
    assert full_model_metric_tables
    assert full_model_metric_tables[0].loc[0, "entity_id"] == "all_features"
    assert full_model_metric_tables[0].loc[0, "role"] == "full_model"
    baseline_metric_tables = [
        dataframe.value
        for dataframe in app.dataframe
        if {"entity_id", "role", "status", "sharpe", "cost_adjusted_cumulative_return"}.issubset(
            dataframe.value.columns
        )
        and "model_baseline" in set(dataframe.value["role"].dropna().astype(str))
    ]
    assert baseline_metric_tables
    assert {"no_model_proxy", "return_baseline_spy", "return_baseline_equal_weight"}.issubset(
        set(baseline_metric_tables[0]["entity_id"])
    )
    ablation_metric_tables = [
        dataframe.value
        for dataframe in app.dataframe
        if {"entity_id", "role", "kind", "status", "sharpe"}.issubset(dataframe.value.columns)
        and "ablation" in set(dataframe.value["role"].dropna().astype(str))
    ]
    assert ablation_metric_tables
    assert {"price_only"}.issubset(set(ablation_metric_tables[0]["entity_id"]))
    pass_fail_reason_tables = [
        dataframe.value
        for dataframe in app.dataframe
        if {"category", "entity_id", "rule", "metric", "status", "passed", "reason"}.issubset(
            dataframe.value.columns
        )
    ]
    assert pass_fail_reason_tables
    assert {"gate", "model_comparison"}.issubset(set(pass_fail_reason_tables[0]["category"]))
    assert any(
        row["rule"] == "model_outperformance" and row["baseline"] == "no_model_proxy"
        for row in pass_fail_reason_tables[0].to_dict("records")
    )
    gate_report_artifact_tables = [
        dataframe.value
        for dataframe in app.dataframe
        if {
            "artifact",
            "format",
            "report_path",
            "system_validity_status",
            "strategy_candidate_status",
            "gate_result_count",
            "includes_gate_results",
            "includes_system_validity_gate",
            "includes_strategy_candidate_gate",
        }.issubset(dataframe.value.columns)
    ]
    assert gate_report_artifact_tables
    gate_report_rows = gate_report_artifact_tables[0].set_index("artifact").to_dict("index")
    assert {"validity_gate.json", "validity_report.md"}.issubset(gate_report_rows)
    assert gate_report_rows["validity_gate.json"]["format"] == "json"
    assert gate_report_rows["validity_report.md"]["format"] == "markdown"
    assert gate_report_rows["validity_gate.json"]["gate_result_count"] > 0
    assert bool(gate_report_rows["validity_gate.json"]["includes_gate_results"]) is True
    assert bool(gate_report_rows["validity_gate.json"]["includes_system_validity_gate"]) is True
    assert bool(gate_report_rows["validity_gate.json"]["includes_strategy_candidate_gate"]) is True
    user_gate_status_tables = [
        dataframe.value
        for dataframe in app.dataframe
        if {
            "scope",
            "gate",
            "display_status",
            "severity",
            "reason",
            "reason_code",
            "metric",
        }.issubset(dataframe.value.columns)
    ]
    assert user_gate_status_tables
    user_gate_rows = user_gate_status_tables[0].set_index("gate").to_dict("index")
    assert "system_validity" in user_gate_rows
    assert "strategy_candidate" in user_gate_rows
    assert user_gate_rows["system_validity"]["display_status"] in {
        "PASS",
        "FAIL",
        "NOT EVALUABLE",
    }
    no_model_proxy_tables = [
        dataframe.value
        for dataframe in app.dataframe
        if {"scenario", "model_proxy_enabled", "validation_status", "sharpe"}.issubset(
            dataframe.value.columns
        )
    ]
    assert no_model_proxy_tables
    assert no_model_proxy_tables[0].loc[0, "scenario"] == "no_model_proxy"
    assert not bool(no_model_proxy_tables[0].loc[0, "model_proxy_enabled"])
    fold_period_tables = [
        dataframe.value
        for dataframe in app.dataframe
        if {
            "fold",
            "fold_type",
            "train_start",
            "test_start",
            "purge_periods",
            "embargo_periods",
        }.issubset(dataframe.value.columns)
    ]
    assert fold_period_tables
    assert set(fold_period_tables[0]["fold_type"]) == {"validation", "oos"}
    assert fold_period_tables[0]["purge_periods"].eq(20).all()
    assert fold_period_tables[0]["embargo_periods"].eq(20).all()
    oos_summary_tables = [
        dataframe.value
        for dataframe in app.dataframe
        if {
            "oos_fold_count",
            "oos_start",
            "oos_end",
            "oos_rank_ic",
            "oos_mean_mae",
        }.issubset(dataframe.value.columns)
    ]
    assert oos_summary_tables
    assert int(oos_summary_tables[0].loc[0, "oos_fold_count"]) == 1
    sensitivity_summary_tables = [
        dataframe.value
        for dataframe in app.dataframe
        if {
            "batch_id",
            "config_id",
            "baseline_scenario_id",
            "scenario_count",
            "baseline_status",
        }.issubset(dataframe.value.columns)
    ]
    assert sensitivity_summary_tables
    assert int(sensitivity_summary_tables[0].loc[0, "scenario_count"]) >= 1
    sensitivity_review_tables = [
        dataframe.value
        for dataframe in app.dataframe
        if {
            "check",
            "status",
            "scenario_id",
            "value",
            "threshold",
            "operator",
            "reason",
        }.issubset(dataframe.value.columns)
        and "turnover_budget" in set(dataframe.value["check"])
    ]
    assert sensitivity_review_tables
    assert {
        "scenario_evaluability",
        "turnover_budget",
        "max_daily_turnover",
        "worst_cost_adjusted_loss",
        "largest_total_cost_increase",
    }.issubset(set(sensitivity_review_tables[0]["check"]))
    sensitivity_tables = [
        dataframe.value
        for dataframe in app.dataframe
        if {
            "scenario_id",
            "label",
            "status",
            "cost_bps",
            "slippage_bps",
            "turnover_budget_pass",
            "cost_adjusted_cumulative_return",
            "baseline_cost_adjusted_cumulative_return_delta",
        }.issubset(dataframe.value.columns)
    ]
    assert sensitivity_tables
    assert "canonical_costs" in set(sensitivity_tables[0]["scenario_id"])
    assert "transaction_cost_sensitivity_result" in app.session_state

    runtime_options = [node.value for node in app.selectbox if node.label == "FinGPT runtime"]
    assert runtime_options and runtime_options[0] in {"transformers", "mlx", "llama-cpp"}
    assert any(
        node.label == "Allow unquantized Transformers 8B load" and node.value is False for node in app.checkbox
    )
    assert any(node.label == "Covariance-aware risk" and node.value is True for node in app.checkbox)
    assert any(node.label == "Apply max daily turnover limit" and node.value is False for node in app.checkbox)
    assert any(node.label == "FinGPT quantized runtime path" for node in app.text_input)

    forecast_column = app.main.children[4].children[0]
    forecast_nodes = _child_nodes(forecast_column)
    has_forecast_chart = any(node.__class__.__name__ == "UnknownElement" for node in forecast_nodes)
    has_forecast_fallback = any(
        node.__class__.__name__ == "Caption"
        and (
            "자료 부족" in str(node.value)
            or ":gray" in str(node.value)
            or "expected_return" in str(node.value)
            or "downside_quantile" in str(node.value)
        )
        for node in forecast_nodes
    )
    has_forecast_fallback = has_forecast_fallback or any(
        node.__class__.__name__ == "Markdown"
        and ("expected_return" in str(node.value) or "downside_quantile" in str(node.value))
        for node in forecast_nodes
    )
    assert has_forecast_chart or has_forecast_fallback

    sec_column = app.main.children[4].children[1]
    sec_nodes = _child_nodes(sec_column)
    has_sec_events = any(node.__class__.__name__ == "Markdown" and "·" in str(node.value) for node in sec_nodes)
    has_sec_fallback = any(
        node.__class__.__name__ == "Caption" and "표시할 SEC 이벤트 카드가 없습니다." in str(node.value)
        for node in sec_nodes
    )
    has_sec_fallback = has_sec_fallback or any(
        node.__class__.__name__ == "Caption" and "predicted_volatility" in str(node.value)
        for node in sec_nodes
    )
    has_sec_fallback = has_sec_fallback or any(
        node.__class__.__name__ == "Markdown" and "위험도" in str(node.value)
        for node in sec_nodes
    )
    assert has_sec_events or has_sec_fallback

    detail_tables = [dataframe.value for dataframe in app.dataframe if "action" in dataframe.value.columns]
    assert detail_tables
    assert detail_tables[0]["action"].dropna().astype(str).str.upper().isin({"BUY", "SELL", "HOLD"}).all()

    first_screen_nodes = [
        node
        for key, node in app.main.children.items()
        if not _is_tab_container(node)
    ]
    first_screen_tables = [
        table
        for node in first_screen_nodes
        for table in _collect_dataframes(node)
    ]
    for table in first_screen_tables:
        assert "action" not in table.value.columns
        assert "raw_signal" not in table.value.columns


def test_streamlit_app_does_not_auto_run_on_first_render(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(pipeline, "run_research_pipeline", lambda config: _run_fake_research_pipeline(captured, config))
    app = AppTest.from_file("app.py", default_timeout=90)
    app.run()

    assert not app.exception
    assert "result" not in app.session_state
    assert captured.get("calls", 0) == 0
    _assert_full_stack_defaults(app)

    caption_values = [str(caption.value) for caption in app.caption]
    assert any(
        "선택 항목은 기본값으로 구성되며" in caption
        for caption in caption_values
    )


def test_streamlit_app_runs_pipeline_only_when_run_button_clicked(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(pipeline, "run_research_pipeline", lambda config: _run_fake_research_pipeline(captured, config))
    app = AppTest.from_file("app.py", default_timeout=90)
    app.run()

    run_button = next(node for node in app.button if node.label == "Run research")
    run_button.click().run()

    assert captured.get("calls", 0) == 1
    assert "result" in app.session_state
    _assert_full_stack_defaults(app)
    _assert_beginner_dashboard_rendered(app, captured)

    # Re-run without click should keep latest result and not rerun pipeline.
    app.run()
    assert captured.get("calls", 0) == 1
