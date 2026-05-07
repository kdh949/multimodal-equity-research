from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quant_research.dashboard.beginner import (
    BeginnerDecisionCoachReport,
    BeginnerResearchDashboard,
    DashboardBadge,
)

DECISION_EVIDENCE_SPECS = (
    ("expected_return", "예상 수익률", "정량 모델이 산출한 다음 구간 기대수익률 feature"),
    ("signal_score", "Signal score", "비용과 슬리피지를 반영해 deterministic engine이 비교한 점수"),
    ("downside_quantile", "하방 분위수", "나쁜 경우를 가정한 하방 수익률 feature"),
    ("predicted_volatility", "예측 변동성", "리스크 모델이 산출한 변동성 feature"),
    ("text_risk_score", "텍스트 리스크", "뉴스/공시 텍스트 extractor가 구조화한 risk feature"),
    ("sec_risk_flag", "SEC 위험 flag", "SEC extractor가 감지한 최신 공시 위험 flag"),
    ("sec_risk_flag_20d", "SEC 20일 위험 flag", "최근 20영업일 공시 위험 flag"),
    ("sec_event_tag", "SEC 이벤트 태그", "SEC extractor가 분류한 이벤트 태그"),
    ("cost_bps", "거래 비용", "신호 점수화에 반영한 비용 가정"),
    ("slippage_bps", "슬리피지", "신호 점수화에 반영한 체결 마찰 가정"),
    ("data_cutoff", "Data cutoff", "이 날짜까지의 데이터만 사용"),
    ("validation_gate_status", "Validation gate", "최종 label을 통과 또는 보류시키는 검증 gate 상태"),
    ("validation_gate_reason", "Gate reason", "검증 gate가 반환한 판단 이유"),
)
PERCENT_EVIDENCE_KEYS = {
    "expected_return",
    "signal_score",
    "downside_quantile",
    "predicted_volatility",
    "text_risk_score",
}
FLAG_EVIDENCE_KEYS = {"sec_risk_flag", "sec_risk_flag_20d"}
BPS_EVIDENCE_KEYS = {"cost_bps", "slippage_bps"}


def render_beginner_overview(dashboard: BeginnerResearchDashboard) -> None:
    st.markdown("### Beginner Research Overview")
    _render_final_signal_strip(dashboard.decision_coach_report)
    st.markdown(dashboard.disclaimer)
    _render_decision_coach_details(dashboard.decision_coach_report)
    badge_cols = st.columns(4)
    for column, badge in zip(
        badge_cols,
        [
            dashboard.direction_badge,
            dashboard.risk_badge,
            dashboard.sec_impact_badge,
            dashboard.validation_badge,
        ],
        strict=True,
    ):
        with column:
            _render_badge(badge)

    chart_col, sec_col = st.columns([1.3, 1.0])
    with chart_col:
        st.subheader(f"{dashboard.ticker} Forecast Interval")
        _render_forecast_chart(dashboard)
    with sec_col:
        st.subheader("SEC Filing Impact")
        _render_sec_events(dashboard)

    st.subheader("Backtest Validation Snapshot")
    backtest = dashboard.backtest_result
    equity_chart_col, metrics_col = st.columns([1.5, 1.0])
    with equity_chart_col:
        st.markdown("### Backtest Equity Curve")
        _render_backtest_equity_curve(dashboard)
    with metrics_col:
        st.markdown("### Validation Metrics")
        metrics_cols = st.columns(2)
        metrics_cols[0].metric("Sharpe", f"{backtest['sharpe']:.2f}")
        metrics_cols[1].metric("Hit Rate", f"{backtest['hit_rate']:.2%}")
        metrics_cols[0].metric("Max Drawdown", f"{backtest['max_drawdown']:.2%}")
        metrics_cols[1].metric("Exposure", f"{backtest['exposure']:.2%}")


def _render_final_signal_strip(report: BeginnerDecisionCoachReport | None) -> None:
    st.markdown("#### 최종 연구 신호")
    if report is None:
        st.warning("최종 연구 신호를 표시하려면 deterministic signal engine과 validation gate 결과가 필요합니다.")
        st.caption(
            "연구용 참고 화면입니다. 개인 맞춤 투자 권고가 아니며 주문 기능은 없습니다. "
            "LLM/텍스트 모델은 feature extractor로만 사용됩니다."
        )
        return

    _render_signal_status(report)
    st.caption(f"검증 Gate: {report.validation_gate_label} · 신호 출처: `{report.decision_source}`")
    st.caption(
        "연구용 참고 화면입니다. 개인 맞춤 투자 권고가 아니며 주문 기능은 없습니다. "
        "LLM/텍스트 모델은 feature extractor로만 사용되고 최종 신호는 deterministic engine과 validation gate가 만듭니다."
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric("예측 방향", report.forecast_direction_label)
    metric_cols[1].metric("신뢰도", report.confidence_label)
    metric_cols[2].metric("검증 Gate", report.validation_gate_label)
    metric_cols[3].metric("기준일", report.as_of_date or "미확인")


def _render_signal_status(report: BeginnerDecisionCoachReport) -> None:
    label = f"{report.ticker} 최종 연구 신호: {report.display_label}"
    if report.visual_tone == "positive":
        st.success(label)
    elif report.visual_tone == "negative":
        st.error(label)
    elif report.visual_tone in {"blocked", "caution"}:
        st.warning(label)
    else:
        st.info(label)


def _render_decision_coach_details(report: BeginnerDecisionCoachReport | None) -> None:
    if report is None:
        return

    st.markdown("#### 판단 이유")
    st.info(report.plain_language_explanation)
    st.warning(report.why_it_might_be_wrong)
    _render_decision_evidence_workspace(report)
    _render_advanced_decision_disclosure(report)


def _render_decision_evidence_workspace(report: BeginnerDecisionCoachReport) -> None:
    available, missing = _decision_evidence_frames(report)
    st.markdown("#### 근거 요약")
    if available.empty:
        st.info("표시 가능한 정량 근거가 아직 없습니다.")
    else:
        st.dataframe(available, width="stretch", hide_index=True)

    st.markdown("#### 아직 부족한 근거")
    if missing.empty:
        st.caption("누락된 핵심 근거가 없습니다.")
    else:
        st.dataframe(missing, width="stretch", hide_index=True)


def _decision_evidence_frames(report: BeginnerDecisionCoachReport) -> tuple[pd.DataFrame, pd.DataFrame]:
    available_rows: list[dict[str, str]] = []
    missing_rows: list[dict[str, str]] = []
    for key, label, detail in DECISION_EVIDENCE_SPECS:
        value = report.evidence.get(key)
        row = {
            "상태": "available",
            "근거": label,
            "값": _format_decision_evidence_value(key, value),
            "상세": detail,
        }
        if _is_missing_evidence(value):
            row["상태"] = "missing"
            row["값"] = "미확인"
            missing_rows.append(row)
        else:
            available_rows.append(row)
    return pd.DataFrame(available_rows), pd.DataFrame(missing_rows)


def _render_advanced_decision_disclosure(report: BeginnerDecisionCoachReport) -> None:
    disclosure = report.advanced_disclosure
    raw_action_key = "_".join(("raw", "signal"))
    note = str(
        disclosure.get(
            f"{raw_action_key}_note",
            "이 값은 주문 신호가 아니라 deterministic engine의 원천 action입니다.",
        )
    )
    with st.expander("고급: 원천 action provenance", expanded=False):
        st.caption(note)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "항목": "raw action",
                        "값": str(disclosure.get(raw_action_key) or "없음"),
                        "설명": note,
                    },
                    {
                        "항목": "decision source",
                        "값": str(disclosure.get("decision_source") or report.decision_source),
                        "설명": "최종 초심자 label을 산출한 deterministic source입니다.",
                    },
                ]
            ),
            width="stretch",
            hide_index=True,
        )


def _render_badge(badge: DashboardBadge) -> None:
    icon = {
        "positive": ":green[●]",
        "negative": ":red[●]",
        "neutral": ":blue[●]",
        "muted": ":gray[●]",
    }.get(badge.tone, ":gray[●]")
    st.markdown(f"**{badge.name}**")
    st.markdown(f"{icon} **{badge.label}**")
    for label, value in badge.evidence.items():
        st.caption(f"{label}: {_format_evidence(value)}")
    if badge.reason:
        st.caption(f"이유: {badge.reason}")
    if badge.next_needed_data:
        st.caption(f"다음 필요 데이터: {badge.next_needed_data}")


def _render_forecast_chart(dashboard: BeginnerResearchDashboard) -> None:
    chart = dashboard.forecast_interval_chart
    history = chart["history"]
    interval = chart["interval"]
    if history.empty or interval.empty:
        _render_fallback(dashboard.fallback_state["forecast_interval_chart"])
        return
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["close"],
            mode="lines",
            name="Close",
        )
    )
    next_date = interval["date"].iloc[0]
    figure.add_trace(
        go.Scatter(
            x=[history["date"].iloc[-1], next_date],
            y=[history["close"].iloc[-1], interval["expected"].iloc[0]],
            mode="lines+markers",
            name="Expected",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[next_date, next_date],
            y=[interval["downside"].iloc[0], interval["upside"].iloc[0]],
            mode="lines",
            name="Forecast range",
        )
    )
    figure.update_layout(height=320, margin={"l": 10, "r": 10, "t": 10, "b": 10})
    st.plotly_chart(figure, width="stretch")


def _render_backtest_equity_curve(dashboard: BeginnerResearchDashboard) -> None:
    equity_curve = dashboard.backtest_result.get("equity_curve")
    if not isinstance(equity_curve, pd.DataFrame):
        _render_fallback(dashboard.fallback_state["backtest_equity_curve"])
        return
    figure = _build_backtest_equity_curve_figure(equity_curve)
    if figure is None:
        _render_fallback(dashboard.fallback_state["backtest_equity_curve"])
        return
    st.plotly_chart(figure, width="stretch")


def _build_backtest_equity_curve_figure(equity_curve: pd.DataFrame) -> go.Figure | None:
    if equity_curve.empty or "date" not in equity_curve.columns or "equity" not in equity_curve.columns:
        return None
    curve = equity_curve.copy()
    curve["date"] = pd.to_datetime(curve["date"], errors="coerce")
    curve = curve.dropna(subset=["date", "equity"]).sort_values("date")
    if curve.empty:
        return None
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=curve["date"],
            y=curve["equity"],
            mode="lines",
            name="Portfolio Equity",
        )
    )
    if "benchmark_equity" in curve.columns:
        figure.add_trace(
            go.Scatter(
                x=curve["date"],
                y=curve["benchmark_equity"],
                mode="lines",
                name="Benchmark Equity",
            )
        )
    figure.update_layout(height=260, margin={"l": 10, "r": 10, "t": 10, "b": 10})
    return figure


def _render_sec_events(dashboard: BeginnerResearchDashboard) -> None:
    if not dashboard.sec_events:
        _render_fallback(dashboard.fallback_state["sec_events"])
        return
    for event in dashboard.sec_events:
        date_value = event["date"]
        date_text = date_value.strftime("%Y-%m-%d") if hasattr(date_value, "strftime") else str(date_value)
        risk_text = "risk" if event["risk_flag"] else "no risk flag"
        st.markdown(f"**{date_text} · {event['event_tag']}**")
        st.caption(f"confidence: {float(event['confidence']):.0%} · {risk_text}")
        if event["summary_ref"]:
            st.caption(str(event["summary_ref"]))


def _render_fallback(state: dict[str, str]) -> None:
    st.markdown(f":gray[{state['status']}]")
    st.caption(state["reason"])
    st.caption(f"다음 필요 데이터: {state['next_needed_data']}")


def _format_evidence(value: object) -> str:
    if isinstance(value, float):
        if abs(value) <= 1:
            return f"{value:.2%}"
        return f"{value:.2f}"
    return str(value)


def _format_decision_evidence_value(key: str, value: object) -> str:
    if _is_missing_evidence(value):
        return "미확인"
    if key in FLAG_EVIDENCE_KEYS:
        try:
            return "있음" if float(value) > 0 else "없음"
        except (TypeError, ValueError):
            return str(value)
    if key in BPS_EVIDENCE_KEYS:
        try:
            return f"{float(value):.2f} bps"
        except (TypeError, ValueError):
            return str(value)
    if key in PERCENT_EVIDENCE_KEYS:
        try:
            return f"{float(value):.2%}"
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _is_missing_evidence(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        is_missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(is_missing, bool):
        return is_missing
    try:
        return bool(is_missing)
    except TypeError:
        return False
