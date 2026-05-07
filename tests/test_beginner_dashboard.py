from __future__ import annotations

from dataclasses import replace

import pandas as pd

from quant_research.backtest.engine import BacktestConfig, run_long_only_backtest
from quant_research.dashboard import (
    build_beginner_decision_coach_report,
    build_beginner_research_dashboard,
)
from quant_research.dashboard import streamlit as dashboard_streamlit
from quant_research.pipeline import PipelineConfig, PipelineResult


def test_beginner_dashboard_builds_badges_and_keeps_raw_signal_in_details() -> None:
    result = _result(
        predictions=_prediction_frame(
            expected_return=0.02,
            predicted_volatility=0.015,
            downside_quantile=-0.01,
            sec_risk_flag=0.0,
            sec_event_tag="earnings",
            sec_event_confidence=0.82,
        ),
        validation_summary=pd.DataFrame(
            {
                "fold": [0],
                "is_oos": [True],
                "directional_accuracy": [0.63],
            }
        ),
    )

    dashboard = build_beginner_research_dashboard(
        result,
        "AAPL",
        PipelineConfig(portfolio_volatility_limit=0.04, max_drawdown_stop=0.2),
    )

    assert dashboard.ticker == "AAPL"
    assert dashboard.direction_badge.label == "상승"
    assert dashboard.risk_badge.label == "낮음"
    assert dashboard.sec_impact_badge.label == "긍정"
    assert dashboard.validation_badge.label == "강함"
    assert dashboard.raw_signal in {"BUY", "SELL", "HOLD"}
    assert dashboard.fallback_state["direction"]["status"] == "정상"
    assert dashboard.fallback_state["forecast_interval_chart"]["status"] == "정상"
    assert dashboard.fallback_state["sec_events"]["status"] == "정상"
    assert not dashboard.fallback_state["forecast_interval_chart"]["reason"]
    assert isinstance(dashboard.backtest_result["equity_curve"], pd.DataFrame)
    assert "equity" in dashboard.backtest_result["equity_curve"].columns
    assert not dashboard.backtest_result["equity_curve"].empty
    assert dashboard.fallback_state["backtest_equity_curve"]["status"] == "정상"
    assert dashboard.research_summary["raw_signal_visible"] is False
    assert "투자 권고가 아닙니다" in dashboard.disclaimer
    assert not dashboard.forecast_interval_chart["interval"].empty


def test_beginner_dashboard_reports_fallbacks_without_hiding_sections() -> None:
    result = _result(
        predictions=pd.DataFrame(columns=["date", "ticker"]),
        validation_summary=pd.DataFrame(),
        sec_features=pd.DataFrame(),
    )

    dashboard = build_beginner_research_dashboard(result, "AAPL", PipelineConfig())

    assert dashboard.direction_badge.label == "자료 부족"
    assert dashboard.sec_impact_badge.label == "자료 부족"
    assert dashboard.validation_badge.label == "검증 불가"
    assert dashboard.fallback_state["direction"]["status"] == "자료 부족"
    assert dashboard.fallback_state["validation"]["status"] == "검증 불가"
    assert dashboard.fallback_state["sec_events"]["status"] == "자료 부족"
    assert dashboard.fallback_state["forecast_interval_chart"]["status"] == "자료 부족"
    assert dashboard.fallback_state["backtest_equity_curve"]["status"] == "자료 부족"
    assert dashboard.fallback_state["validation"]["next_needed_data"]
    assert dashboard.forecast_interval_chart["history"].empty
    assert dashboard.raw_signal == "HOLD"


def test_beginner_dashboard_marks_negative_sec_risk() -> None:
    result = _result(
        predictions=_prediction_frame(
            expected_return=0.01,
            predicted_volatility=0.02,
            downside_quantile=-0.01,
            sec_risk_flag=1.0,
            sec_event_tag="legal",
            sec_event_confidence=0.9,
        ),
        validation_summary=pd.DataFrame({"fold": [0], "is_oos": [True], "directional_accuracy": [0.52]}),
    )

    dashboard = build_beginner_research_dashboard(result, "AAPL", PipelineConfig())

    assert dashboard.sec_impact_badge.label == "부정"
    assert dashboard.sec_impact_badge.evidence["event_tag"] == "legal"
    assert dashboard.sec_events


def test_beginner_dashboard_extracts_sec_events_when_predictions_lack_sec_fields() -> None:
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            "ticker": ["AAPL", "AAPL"],
            "expected_return": [0.01, 0.01],
            "predicted_volatility": [0.02, 0.02],
            "downside_quantile": [-0.01, -0.01],
            "upside_quantile": [0.03, 0.03],
            "forward_return_1": [0.004, 0.006],
            "is_oos": [False, True],
        }
    )
    sec_features = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-04", "2026-01-05"]),
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "sec_risk_flag": [0.0, 0.0, 1.0],
            "sec_risk_flag_20d": [0.0, 0.0, 1.0],
            "sec_event_tag": ["quarterly_report", "earnings", "legal"],
            "sec_event_confidence": [0.6, 0.25, 0.88],
            "sec_summary_ref": ["Form 10-Q", "Form 8-K", "Form 8-K"],
        }
    )

    result = _result(predictions=predictions, validation_summary=pd.DataFrame(), sec_features=sec_features)
    dashboard = build_beginner_research_dashboard(result, "AAPL", PipelineConfig())

    assert dashboard.sec_impact_badge.label == "부정"
    assert dashboard.sec_impact_badge.evidence["event_tag"] == "legal"
    assert dashboard.sec_events
    assert dashboard.sec_events[0]["event_tag"] == "legal"
    assert dashboard.sec_events[0]["risk_flag"] is True
    assert dashboard.raw_signal in {"BUY", "SELL", "HOLD"}


def test_beginner_dashboard_forecast_interval_matches_expected_close_projection() -> None:
    result = _result(
        predictions=_prediction_frame(
            expected_return=0.02,
            predicted_volatility=0.015,
            downside_quantile=-0.01,
            sec_risk_flag=0.0,
            sec_event_tag="earnings",
            sec_event_confidence=0.82,
        ),
        validation_summary=pd.DataFrame(
            {
                "fold": [0],
                "is_oos": [True],
                "directional_accuracy": [0.63],
            }
        ),
    )
    dashboard = build_beginner_research_dashboard(result, "AAPL")
    interval = dashboard.forecast_interval_chart["interval"].iloc[0]
    history = dashboard.forecast_interval_chart["history"]
    latest_close = history["close"].iloc[-1]

    assert interval["date"] > history["date"].iloc[-1]
    assert interval["expected"] == latest_close * 1.02
    assert interval["downside"] == latest_close * 0.99
    assert interval["upside"] == latest_close * 1.04


def test_beginner_dashboard_falls_back_interval_when_forecast_inputs_missing() -> None:
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            "ticker": ["AAPL", "AAPL"],
            "expected_return": [0.015, 0.015],
            "predicted_volatility": [0.015, 0.015],
            "downside_quantile": [-0.01, -0.01],
            "forward_return_1": [0.004, 0.006],
        }
    )
    result = _result(predictions=predictions, validation_summary=pd.DataFrame())
    dashboard = build_beginner_research_dashboard(result, "AAPL")

    assert dashboard.forecast_interval_chart["interval"].empty
    assert dashboard.fallback_state["forecast_interval_chart"]["status"] == "모델 비활성"
    assert not dashboard.fallback_state["forecast_interval_chart"]["reason"] == ""
    assert len(dashboard.forecast_interval_chart["history"]) >= 1


def test_beginner_backtest_equity_curve_is_renderable_or_fallback_safe() -> None:
    result = _result(
        predictions=_prediction_frame(
            expected_return=0.02,
            predicted_volatility=0.015,
            downside_quantile=-0.01,
            sec_risk_flag=0.0,
            sec_event_tag="earnings",
            sec_event_confidence=0.82,
        ),
        validation_summary=pd.DataFrame({"fold": [0], "is_oos": [True], "directional_accuracy": [0.63]}),
    )
    dashboard = build_beginner_research_dashboard(result, "AAPL")

    figure = dashboard_streamlit._build_backtest_equity_curve_figure(
        dashboard.backtest_result["equity_curve"]
    )
    assert figure is not None
    assert len(figure.data) >= 1

    assert dashboard_streamlit._build_backtest_equity_curve_figure(pd.DataFrame()) is None


def test_beginner_decision_coach_uses_deterministic_signal_action_not_prediction_action() -> None:
    predictions = _prediction_frame(
        expected_return=0.04,
        predicted_volatility=0.015,
        downside_quantile=-0.01,
        sec_risk_flag=0.0,
        sec_event_tag="earnings",
        sec_event_confidence=0.82,
    )
    predictions["action"] = "SELL"
    result = _result(
        predictions=predictions,
        validation_summary=pd.DataFrame({"fold": [0], "is_oos": [True], "directional_accuracy": [0.63]}),
    )
    signals = result.signals.copy()
    signals["action"] = "BUY"
    signals["signal_score"] = 0.04
    result = replace(result, predictions=predictions, signals=signals)

    report = build_beginner_decision_coach_report(result, "AAPL", _validity_gate("PASS"))

    assert report.research_signal == "BUY"
    assert report.display_label == "긍정적"
    assert report.beginner_signal == "positive"
    assert report.decision_source == "deterministic_signal_engine"
    assert "action_buy" in report.reason_codes
    assert "validation_gate_pass" in report.reason_codes
    assert "deterministic signal engine" in report.plain_language_explanation
    assert report.advanced_disclosure["raw_signal"] == "BUY"
    assert report.advanced_disclosure["raw_signal_visible"] is False
    assert report.not_investment_advice is True


def test_beginner_decision_coach_downgrades_gate_fail_to_composite_label() -> None:
    result = _result(
        predictions=_prediction_frame(
            expected_return=0.04,
            predicted_volatility=0.015,
            downside_quantile=-0.01,
            sec_risk_flag=0.0,
            sec_event_tag="earnings",
            sec_event_confidence=0.82,
        ),
        validation_summary=pd.DataFrame({"fold": [0], "is_oos": [True], "directional_accuracy": [0.63]}),
    )
    signals = result.signals.copy()
    signals["action"] = "BUY"
    result = replace(result, signals=signals)

    report = build_beginner_decision_coach_report(
        result,
        "AAPL",
        _validity_gate("FAIL", system_status="pass", strategy_status="fail"),
    )

    assert report.display_label == "긍정적이지만 검증 불충분"
    assert report.beginner_signal == "insufficient_validation"
    assert report.validation_gate_status == "fail"
    assert report.visual_tone == "blocked"
    assert report.confidence_level == "insufficient_validation"
    assert "검증 불충분" in report.plain_language_explanation
    assert "확정 신호처럼 읽으면 안 됩니다" in report.why_it_might_be_wrong


def test_beginner_decision_coach_does_not_fallback_to_prediction_action_when_signals_missing() -> None:
    predictions = _prediction_frame(
        expected_return=0.04,
        predicted_volatility=0.015,
        downside_quantile=-0.01,
        sec_risk_flag=0.0,
        sec_event_tag="earnings",
        sec_event_confidence=0.82,
    )
    predictions["action"] = "BUY"
    result = _result(
        predictions=predictions,
        validation_summary=pd.DataFrame({"fold": [0], "is_oos": [True], "directional_accuracy": [0.63]}),
    )
    result = replace(result, signals=pd.DataFrame(columns=["date", "ticker", "action"]))

    report = build_beginner_decision_coach_report(result, "AAPL", _validity_gate("PASS"))

    assert report.research_signal == ""
    assert report.display_label == "검증 불충분"
    assert report.beginner_signal == "insufficient_validation"
    assert "action_missing" in report.reason_codes
    assert report.advanced_disclosure["raw_signal_disclosure_visible"] is False


def test_final_signal_strip_renders_beginner_labels_without_raw_action(monkeypatch) -> None:
    result = _result(
        predictions=_prediction_frame(
            expected_return=0.04,
            predicted_volatility=0.015,
            downside_quantile=-0.01,
            sec_risk_flag=0.0,
            sec_event_tag="earnings",
            sec_event_confidence=0.82,
        ),
        validation_summary=pd.DataFrame({"fold": [0], "is_oos": [True], "directional_accuracy": [0.63]}),
    )
    signals = result.signals.copy()
    signals["action"] = "BUY"
    result = replace(result, signals=signals)
    report = build_beginner_decision_coach_report(result, "AAPL", _validity_gate("PASS"))
    fake_streamlit = _FakeSignalStripStreamlit()
    monkeypatch.setattr(dashboard_streamlit, "st", fake_streamlit)

    dashboard_streamlit._render_final_signal_strip(report)

    rendered_text = fake_streamlit.rendered_text()
    assert "AAPL 최종 연구 신호: 긍정적" in rendered_text
    assert ("예측 방향", "상승") in fake_streamlit.metrics
    assert ("검증 Gate", "통과") in fake_streamlit.metrics
    assert "신호 출처: `deterministic_signal_engine`" in rendered_text
    assert "LLM/텍스트 모델은 feature extractor로만 사용" in rendered_text
    assert "BUY" not in rendered_text
    assert "SELL" not in rendered_text
    assert "HOLD" not in rendered_text


def test_final_signal_strip_uses_warning_tone_when_gate_blocks_label(monkeypatch) -> None:
    result = _result(
        predictions=_prediction_frame(
            expected_return=0.04,
            predicted_volatility=0.015,
            downside_quantile=-0.01,
            sec_risk_flag=0.0,
            sec_event_tag="earnings",
            sec_event_confidence=0.82,
        ),
        validation_summary=pd.DataFrame({"fold": [0], "is_oos": [True], "directional_accuracy": [0.63]}),
    )
    signals = result.signals.copy()
    signals["action"] = "BUY"
    result = replace(result, signals=signals)
    report = build_beginner_decision_coach_report(result, "AAPL", _validity_gate("FAIL"))
    fake_streamlit = _FakeSignalStripStreamlit()
    monkeypatch.setattr(dashboard_streamlit, "st", fake_streamlit)

    dashboard_streamlit._render_final_signal_strip(report)

    rendered_text = fake_streamlit.rendered_text()
    assert fake_streamlit.warnings == ["AAPL 최종 연구 신호: 긍정적이지만 검증 불충분"]
    assert ("신뢰도", "검증 불충분") in fake_streamlit.metrics
    assert ("검증 Gate", "실패") in fake_streamlit.metrics
    assert "BUY" not in rendered_text


def test_decision_evidence_frames_split_available_and_missing_values() -> None:
    result = _result(
        predictions=pd.DataFrame(columns=["date", "ticker"]),
        validation_summary=pd.DataFrame(),
    )
    result = replace(result, signals=pd.DataFrame(columns=["date", "ticker", "action"]))
    report = build_beginner_decision_coach_report(result, "AAPL", _validity_gate("PASS"))

    available, missing = dashboard_streamlit._decision_evidence_frames(report)

    assert {"거래 비용", "슬리피지", "Validation gate", "Gate reason"}.issubset(
        set(available["근거"])
    )
    assert {"예상 수익률", "Signal score", "하방 분위수", "예측 변동성", "Data cutoff"}.issubset(
        set(missing["근거"])
    )
    assert missing["값"].eq("미확인").all()


def test_decision_evidence_marks_absent_optional_features_as_missing() -> None:
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            "ticker": ["AAPL", "AAPL"],
            "expected_return": [0.02, 0.02],
            "predicted_volatility": [0.015, 0.015],
            "downside_quantile": [-0.01, -0.01],
            "upside_quantile": [0.04, 0.04],
            "model_confidence": [0.8, 0.8],
            "liquidity_score": [20.0, 20.0],
            "forward_return_1": [0.004, 0.006],
            "forward_return_20": [0.04, 0.06],
            "is_oos": [False, True],
        }
    )
    result = _result(
        predictions=predictions,
        validation_summary=pd.DataFrame({"fold": [0], "is_oos": [True], "directional_accuracy": [0.63]}),
        sec_features=pd.DataFrame(),
    )
    report = build_beginner_decision_coach_report(result, "AAPL", _validity_gate("PASS"))

    available, missing = dashboard_streamlit._decision_evidence_frames(report)

    assert {"예상 수익률", "Signal score", "하방 분위수", "예측 변동성"}.issubset(
        set(available["근거"])
    )
    assert "SEC 이벤트 태그" in set(missing["근거"])
    missing_rows = missing.set_index("근거")
    assert missing_rows.loc["SEC 이벤트 태그", "값"] == "미확인"


def test_decision_coach_details_render_explanation_evidence_and_closed_disclosure(
    monkeypatch,
) -> None:
    result = _result(
        predictions=_prediction_frame(
            expected_return=0.04,
            predicted_volatility=0.015,
            downside_quantile=-0.01,
            sec_risk_flag=0.0,
            sec_event_tag="earnings",
            sec_event_confidence=0.82,
        ),
        validation_summary=pd.DataFrame({"fold": [0], "is_oos": [True], "directional_accuracy": [0.63]}),
    )
    signals = result.signals.copy()
    signals["action"] = "BUY"
    result = replace(result, signals=signals)
    report = build_beginner_decision_coach_report(result, "AAPL", _validity_gate("PASS"))
    fake_streamlit = _FakeSignalStripStreamlit()
    monkeypatch.setattr(dashboard_streamlit, "st", fake_streamlit)

    dashboard_streamlit._render_decision_coach_details(report)

    rendered_text = fake_streamlit.rendered_text()
    assert report.plain_language_explanation in rendered_text
    assert report.why_it_might_be_wrong in rendered_text
    assert fake_streamlit.expanders == [("고급: 원천 action provenance", False)]
    assert "이 값은 주문 신호가 아니라 deterministic engine의 원천 action입니다." in rendered_text
    evidence_tables = [
        frame for frame in fake_streamlit.dataframes if {"상태", "근거", "값", "상세"}.issubset(frame.columns)
    ]
    assert evidence_tables
    assert {
        "예상 수익률",
        "Signal score",
        "하방 분위수",
        "예측 변동성",
        "텍스트 리스크",
        "SEC 위험 flag",
        "SEC 20일 위험 flag",
        "거래 비용",
        "슬리피지",
        "Data cutoff",
    }.issubset(set(evidence_tables[0]["근거"]))
    disclosure_tables = [
        frame for frame in fake_streamlit.dataframes if {"항목", "값", "설명"}.issubset(frame.columns)
    ]
    assert disclosure_tables
    raw_action_row = disclosure_tables[0].set_index("항목").loc["raw action"]
    assert raw_action_row["값"] == "BUY"
    assert "주문 신호가 아니라 deterministic engine의 원천 action" in raw_action_row["설명"]


class _FakeSignalStripColumn:
    def __init__(self, owner: _FakeSignalStripStreamlit) -> None:
        self._owner = owner

    def metric(self, label: str, value: object) -> None:
        self._owner.metrics.append((label, str(value)))


class _FakeSignalStripExpander:
    def __enter__(self) -> _FakeSignalStripExpander:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        return False


class _FakeSignalStripStreamlit:
    def __init__(self) -> None:
        self.markdowns: list[str] = []
        self.captions: list[str] = []
        self.infos: list[str] = []
        self.successes: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.metrics: list[tuple[str, str]] = []
        self.dataframes: list[pd.DataFrame] = []
        self.expanders: list[tuple[str, bool]] = []

    def markdown(self, value: str) -> None:
        self.markdowns.append(str(value))

    def caption(self, value: str) -> None:
        self.captions.append(str(value))

    def info(self, value: str) -> None:
        self.infos.append(str(value))

    def success(self, value: str) -> None:
        self.successes.append(str(value))

    def warning(self, value: str) -> None:
        self.warnings.append(str(value))

    def error(self, value: str) -> None:
        self.errors.append(str(value))

    def columns(self, count: int) -> list[_FakeSignalStripColumn]:
        return [_FakeSignalStripColumn(self) for _ in range(count)]

    def dataframe(self, value: pd.DataFrame, **_: object) -> None:
        self.dataframes.append(value)

    def expander(self, label: str, *, expanded: bool = False) -> _FakeSignalStripExpander:
        self.expanders.append((label, expanded))
        return _FakeSignalStripExpander()

    def rendered_text(self) -> str:
        metric_text = [f"{label}: {value}" for label, value in self.metrics]
        dataframe_text = [dataframe.to_string(index=False) for dataframe in self.dataframes]
        return "\n".join(
            [
                *self.markdowns,
                *self.successes,
                *self.warnings,
                *self.errors,
                *self.infos,
                *metric_text,
                *self.captions,
                *dataframe_text,
            ]
        )


def _result(
    predictions: pd.DataFrame,
    validation_summary: pd.DataFrame,
    sec_features: pd.DataFrame | None = None,
) -> PipelineResult:
    market_data = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=5, freq="B"),
            "ticker": ["AAPL"] * 5,
            "open": [99.0, 100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [98.0, 99.0, 100.0, 101.0, 102.0],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "adj_close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "volume": [1_000_000] * 5,
        }
    )
    if sec_features is None:
        sec_features = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
                "ticker": ["AAPL", "AAPL"],
                "sec_risk_flag": [0.0, float(predictions["sec_risk_flag"].max()) if "sec_risk_flag" in predictions else 0.0],
                "sec_risk_flag_20d": [
                    0.0,
                    float(predictions["sec_risk_flag_20d"].max()) if "sec_risk_flag_20d" in predictions else 0.0,
                ],
                "sec_event_tag": [
                    "quarterly_report",
                    str(predictions["sec_event_tag"].iloc[-1])
                    if "sec_event_tag" in predictions and not predictions.empty
                    else "none",
                ],
                "sec_event_confidence": [
                    0.6,
                    float(predictions["sec_event_confidence"].iloc[-1])
                    if "sec_event_confidence" in predictions and not predictions.empty
                    else 0.0,
                ],
                "sec_summary_ref": ["Form 10-Q", "Form 8-K"],
            }
        )
    frame = predictions.copy()
    if not frame.empty and "forward_return_1" not in frame:
        frame["forward_return_1"] = 0.005
    if not frame.empty and "forward_return_20" not in frame:
        frame["forward_return_20"] = 0.05
    backtest = run_long_only_backtest(frame, BacktestConfig(top_n=1)) if not frame.empty else run_long_only_backtest(
        pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "expected_return",
                "predicted_volatility",
                "downside_quantile",
                "forward_return_1",
            ]
        )
    )
    return PipelineResult(
        market_data=market_data,
        news_features=pd.DataFrame(),
        sec_features=sec_features,
        features=pd.DataFrame(),
        predictions=predictions,
        signals=backtest.signals,
        validation_summary=validation_summary,
        ablation_summary=[],
        backtest=backtest,
    )


def _validity_gate(
    final_decision: str,
    *,
    system_status: str = "pass",
    strategy_status: str = "pass",
) -> dict[str, object]:
    return {
        "system_validity_status": system_status,
        "strategy_candidate_status": strategy_status,
        "official_message": "Gate ready" if final_decision == "PASS" else "Gate blocked",
        "validity_gate_result_summary": {
            "final_gate_decision": final_decision,
            "system_validity_status": system_status,
            "strategy_candidate_status": strategy_status,
            "reason": "Gate ready" if final_decision == "PASS" else "Gate blocked",
        },
    }


def _prediction_frame(
    expected_return: float,
    predicted_volatility: float,
    downside_quantile: float,
    sec_risk_flag: float,
    sec_event_tag: str,
    sec_event_confidence: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            "ticker": ["AAPL", "AAPL"],
            "expected_return": [expected_return, expected_return],
            "predicted_volatility": [predicted_volatility, predicted_volatility],
            "downside_quantile": [downside_quantile, downside_quantile],
            "upside_quantile": [0.04, 0.04],
            "model_confidence": [0.8, 0.8],
            "text_risk_score": [0.0, 0.0],
            "sec_risk_flag": [0.0, sec_risk_flag],
            "sec_risk_flag_20d": [0.0, sec_risk_flag],
            "sec_event_tag": ["none", sec_event_tag],
            "sec_event_confidence": [0.0, sec_event_confidence],
            "sec_summary_ref": ["", "Form 8-K"],
            "news_negative_ratio": [0.0, 0.0],
            "liquidity_score": [20.0, 20.0],
            "forward_return_1": [0.004, 0.006],
            "forward_return_20": [0.04, 0.06],
            "is_oos": [False, True],
        }
    )
