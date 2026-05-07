from __future__ import annotations

import pandas as pd

from quant_research.backtest.engine import BacktestConfig, run_long_only_backtest
from quant_research.dashboard import build_beginner_research_dashboard
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
