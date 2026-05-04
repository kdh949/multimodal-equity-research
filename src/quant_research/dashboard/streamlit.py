from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quant_research.dashboard.beginner import BeginnerResearchDashboard, DashboardBadge


def render_beginner_overview(dashboard: BeginnerResearchDashboard) -> None:
    st.markdown("### Beginner Research Overview")
    st.markdown(dashboard.disclaimer)
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
    with st.expander("Deterministic signal detail"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "ticker": dashboard.ticker,
                        "raw_signal": dashboard.raw_signal,
                        "signal_source": "DeterministicSignalEngine",
                    }
                ]
            ),
            hide_index=True,
            use_container_width=True,
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
    st.plotly_chart(figure, use_container_width=True)


def _render_backtest_equity_curve(dashboard: BeginnerResearchDashboard) -> None:
    equity_curve = dashboard.backtest_result.get("equity_curve")
    if not isinstance(equity_curve, pd.DataFrame):
        _render_fallback(dashboard.fallback_state["backtest_equity_curve"])
        return
    figure = _build_backtest_equity_curve_figure(equity_curve)
    if figure is None:
        _render_fallback(dashboard.fallback_state["backtest_equity_curve"])
        return
    st.plotly_chart(figure, use_container_width=True)


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
