from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from quant_research.backtest.metrics import PerformanceMetrics
from quant_research.signals.engine import SignalEngineConfig

if TYPE_CHECKING:
    from quant_research.pipeline import PipelineConfig, PipelineResult


DISCLAIMER = "연구용 리서치 화면이며 투자 권고가 아닙니다. 실거래 주문 기능은 제공하지 않습니다."
NO_EVENT_TAGS = {"", "none", "0.0", "nan"}


@dataclass(frozen=True)
class DashboardBadge:
    name: str
    label: str
    tone: str
    evidence: dict[str, object] = field(default_factory=dict)
    status: str = "정상"
    reason: str = ""
    next_needed_data: str = ""


@dataclass(frozen=True)
class BeginnerResearchDashboard:
    ticker: str
    research_summary: dict[str, object]
    direction_badge: DashboardBadge
    risk_badge: DashboardBadge
    sec_impact_badge: DashboardBadge
    validation_badge: DashboardBadge
    forecast_interval_chart: dict[str, pd.DataFrame]
    sec_events: list[dict[str, object]]
    backtest_result: dict[str, object]
    raw_signal: str
    fallback_state: dict[str, dict[str, str]]
    disclaimer: str = DISCLAIMER


def build_beginner_research_dashboard(
    result: PipelineResult,
    ticker: str,
    config: PipelineConfig | None = None,
) -> BeginnerResearchDashboard:
    ticker = ticker.upper()
    signal_config = SignalEngineConfig(
        cost_bps=getattr(config, "cost_bps", SignalEngineConfig.cost_bps),
        slippage_bps=getattr(config, "slippage_bps", SignalEngineConfig.slippage_bps),
    )
    latest = _latest_signal_row(result, ticker)
    metrics = result.backtest.metrics

    direction_badge = _direction_badge(latest, signal_config)
    risk_badge = _risk_badge(latest, metrics, config)
    sec_impact_badge = _sec_impact_badge(latest, result.sec_features, ticker)
    validation_badge = _validation_badge(result.validation_summary, metrics)
    forecast_chart = _forecast_interval_chart(result.market_data, latest, ticker)
    sec_events = _sec_events(result.sec_features, ticker)
    fallback_state = {
        "direction": _fallback_from_badge(direction_badge),
        "risk": _fallback_from_badge(risk_badge),
        "sec_impact": _fallback_from_badge(sec_impact_badge),
        "validation": _fallback_from_badge(validation_badge),
        "forecast_interval_chart": _chart_fallback(forecast_chart),
        "backtest_equity_curve": _backtest_curve_fallback(result.backtest.equity_curve),
        "sec_events": _events_fallback(sec_events),
    }

    raw_signal = _latest_action(result, ticker)
    research_summary = {
        "ticker": ticker,
        "direction": direction_badge.label,
        "risk": risk_badge.label,
        "sec_impact": sec_impact_badge.label,
        "validation": validation_badge.label,
        "raw_signal_visible": False,
    }
    return BeginnerResearchDashboard(
        ticker=ticker,
        research_summary=research_summary,
        direction_badge=direction_badge,
        risk_badge=risk_badge,
        sec_impact_badge=sec_impact_badge,
        validation_badge=validation_badge,
        forecast_interval_chart=forecast_chart,
        sec_events=sec_events,
        backtest_result=_backtest_result(metrics, result.backtest.equity_curve),
        raw_signal=raw_signal,
        fallback_state=fallback_state,
    )


def _latest_signal_row(result: PipelineResult, ticker: str) -> pd.Series | None:
    candidates: list[pd.Series] = []
    if not result.signals.empty:
        row = _latest_ticker_row(result.signals, ticker)
        if row is not None:
            row = row.copy()
            row["_source_priority"] = 0
            candidates.append(row)
    if not result.predictions.empty:
        row = _latest_ticker_row(result.predictions, ticker)
        if row is not None:
            row = row.copy()
            row["_source_priority"] = 1
            candidates.append(row)
    if not candidates:
        return None
    candidates_df = pd.DataFrame(candidates).copy()
    candidates_df["date"] = pd.to_datetime(candidates_df["date"], errors="coerce")
    candidates_df = candidates_df.dropna(subset=["date"])
    if candidates_df.empty:
        return None
    candidates_df["_source_priority"] = -candidates_df["_source_priority"].astype(int)
    return candidates_df.sort_values(["date", "_source_priority"]).iloc[-1].drop(labels=["_source_priority"])


def _latest_ticker_row(frame: pd.DataFrame, ticker: str) -> pd.Series | None:
    if frame.empty or not {"date", "ticker"}.issubset(frame.columns):
        return None
    selected = frame[frame["ticker"].astype(str).str.upper() == ticker].copy()
    if selected.empty:
        return None
    selected["date"] = pd.to_datetime(selected["date"], errors="coerce")
    selected = selected.dropna(subset=["date"])
    if selected.empty:
        return None
    return selected.sort_values("date").iloc[-1]


def _latest_action(result: PipelineResult, ticker: str) -> str:
    if not result.signals.empty:
        row = _latest_ticker_row(result.signals, ticker)
        if row is not None:
            action = _text_value(row, "action", "")
            action = action.upper()
            if action in {"BUY", "SELL", "HOLD"}:
                return action
    if not result.predictions.empty:
        row = _latest_ticker_row(result.predictions, ticker)
        if row is not None:
            action = _text_value(row, "action", "")
            action = action.upper()
            if action in {"BUY", "SELL", "HOLD"}:
                return action
    return "HOLD"


def _direction_badge(row: pd.Series | None, config: SignalEngineConfig) -> DashboardBadge:
    if row is None or not _has_values(row, "expected_return", "downside_quantile"):
        return _fallback_badge(
            "방향성",
            "자료 부족",
            "예상수익률과 하방 분위수 예측값이 없습니다.",
            "walk-forward 예측값을 생성할 수 있는 가격 feature와 학습 구간",
        )
    expected = _float_value(row, "expected_return")
    downside = _float_value(row, "downside_quantile")
    evidence = {"expected_return": expected, "downside_quantile": downside}
    if expected >= config.min_expected_return and downside >= config.min_downside_quantile:
        return DashboardBadge("방향성", "상승", "positive", evidence)
    if expected < 0 or downside < config.min_downside_quantile:
        return DashboardBadge("방향성", "하락", "negative", evidence)
    return DashboardBadge("방향성", "중립", "neutral", evidence)


def _risk_badge(
    row: pd.Series | None,
    metrics: PerformanceMetrics,
    config: PipelineConfig | None,
) -> DashboardBadge:
    if row is None or not _has_values(row, "predicted_volatility"):
        return _fallback_badge(
            "위험도",
            "자료 부족",
            "예측 변동성 값이 없습니다.",
            "가격 feature, 예측 변동성, 백테스트 equity curve",
        )
    volatility = _float_value(row, "predicted_volatility")
    max_drawdown = float(metrics.max_drawdown)
    volatility_limit = float(getattr(config, "portfolio_volatility_limit", 0.04))
    drawdown_stop = abs(float(getattr(config, "max_drawdown_stop", 0.20)))
    evidence = {"predicted_volatility": volatility, "max_drawdown": max_drawdown}
    if volatility <= volatility_limit * 0.5 and abs(max_drawdown) <= drawdown_stop * 0.5:
        return DashboardBadge("위험도", "낮음", "positive", evidence)
    if volatility > volatility_limit or abs(max_drawdown) > drawdown_stop:
        return DashboardBadge("위험도", "높음", "negative", evidence)
    return DashboardBadge("위험도", "보통", "neutral", evidence)


def _sec_impact_badge(row: pd.Series | None, sec_features: pd.DataFrame, ticker: str) -> DashboardBadge:
    row = _row_with_latest_sec(row, sec_features, ticker)
    if row is None or not _has_values(row, "sec_event_confidence"):
        return _fallback_badge(
            "공시 영향",
            "자료 부족",
            "SEC 공시 이벤트 feature가 없습니다.",
            "SEC filing event tag, risk flag, confidence",
        )
    event_tag = _text_value(row, "sec_event_tag", "none")
    confidence = _float_value(row, "sec_event_confidence")
    risk_flag = _float_value(row, "sec_risk_flag")
    risk_flag_20d = _float_value(row, "sec_risk_flag_20d")
    evidence = {
        "risk_flag": risk_flag,
        "risk_flag_20d": risk_flag_20d,
        "event_tag": _normalize_event_tag(event_tag),
        "confidence": confidence,
    }
    if confidence <= 0 and _is_no_event_tag(event_tag):
        return DashboardBadge(
            "공시 영향",
            "중립",
            "neutral",
            evidence,
            status="정상",
            reason="",
            next_needed_data="",
        )
    if risk_flag > 0 or risk_flag_20d > 0 or _contains_any(_normalize_event_tag(event_tag), {"risk", "restatement", "legal"}):
        return DashboardBadge("공시 영향", "부정", "negative", evidence)
    if confidence < 0.4 and not _is_no_event_tag(event_tag):
        return DashboardBadge("공시 영향", "불확실", "muted", evidence)
    if _is_no_event_tag(event_tag):
        return DashboardBadge("공시 영향", "중립", "neutral", evidence)
    if confidence >= 0.4:
        return DashboardBadge("공시 영향", "긍정", "positive", evidence)
    return DashboardBadge("공시 영향", "불확실", "muted", evidence)


def _validation_badge(validation_summary: pd.DataFrame, metrics: PerformanceMetrics) -> DashboardBadge:
    if validation_summary.empty or "is_oos" not in validation_summary.columns:
        return _fallback_badge(
            "검증 신뢰도",
            "검증 불가",
            "out-of-sample fold가 없습니다.",
            "walk-forward 검증이 가능한 충분한 기간의 가격/feature 데이터",
        )
    has_oos = bool(validation_summary["is_oos"].fillna(False).any())
    if not has_oos:
        return _fallback_badge(
            "검증 신뢰도",
            "검증 불가",
            "마지막 fold가 out-of-sample로 표시되지 않았습니다.",
            "검증 구간을 포함한 walk-forward split",
        )
    oos = validation_summary[validation_summary["is_oos"].fillna(False)]
    hit_rate = float(metrics.hit_rate)
    directional_accuracy = _safe_mean(oos, "directional_accuracy")
    evidence = {
        "is_oos": True,
        "sharpe": float(metrics.sharpe),
        "hit_rate": hit_rate,
        "directional_accuracy": directional_accuracy,
    }
    if metrics.sharpe >= 1.0 and hit_rate >= 0.53:
        return DashboardBadge("검증 신뢰도", "강함", "positive", evidence)
    if metrics.sharpe >= 0.25 and hit_rate >= 0.48:
        return DashboardBadge("검증 신뢰도", "보통", "neutral", evidence)
    return DashboardBadge("검증 신뢰도", "약함", "negative", evidence)


def _forecast_interval_chart(
    market_data: pd.DataFrame,
    row: pd.Series | None,
    ticker: str,
) -> dict[str, pd.DataFrame]:
    empty = {"history": pd.DataFrame(columns=["date", "close"]), "interval": pd.DataFrame()}
    if market_data.empty or not {"date", "ticker", "close"}.issubset(market_data.columns) or row is None:
        return empty
    history = market_data[market_data["ticker"].astype(str).str.upper() == ticker].copy()
    if history.empty:
        return empty
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history.dropna(subset=["date", "close"]).sort_values("date")
    anchor = _to_datetime_or_none(_text_value(row, "date", ""))
    if anchor is not None:
        history = history[history["date"] <= anchor]
    history = history.tail(60)
    if history.empty or not _has_values(row, "expected_return", "downside_quantile", "upside_quantile"):
        return {"history": history[["date", "close"]], "interval": pd.DataFrame()}
    last_close = float(history["close"].iloc[-1])
    next_date = pd.bdate_range(history["date"].iloc[-1], periods=2)[-1]
    interval = pd.DataFrame(
        {
            "date": [next_date],
            "expected": [last_close * (1 + _float_value(row, "expected_return"))],
            "downside": [last_close * (1 + _float_value(row, "downside_quantile"))],
            "upside": [last_close * (1 + _float_value(row, "upside_quantile"))],
        }
    )
    return {"history": history[["date", "close"]], "interval": interval}


def _sec_events(sec_features: pd.DataFrame, ticker: str) -> list[dict[str, object]]:
    if sec_features.empty or not {"date", "ticker"}.issubset(sec_features.columns):
        return []
    frame = sec_features[sec_features["ticker"].astype(str).str.upper() == ticker].copy()
    if frame.empty:
        return []
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["sec_event_tag"] = frame.get("sec_event_tag", "none")
    frame["sec_event_confidence"] = _numeric_series(frame, "sec_event_confidence")
    event_tag_series = frame["sec_event_tag"].astype(str).str.lower()
    frame["has_event_tag"] = event_tag_series.ne("none") & event_tag_series.ne("nan") & frame["sec_event_tag"].notna()
    event_mask = (
        frame["has_event_tag"]
        | (_numeric_series(frame, "sec_risk_flag") > 0)
        | (_numeric_series(frame, "sec_8k_count") > 0)
        | (_numeric_series(frame, "sec_10q_count") > 0)
        | (_numeric_series(frame, "sec_10k_count") > 0)
        | (_numeric_series(frame, "sec_form4_count") > 0)
    )
    events = frame[event_mask].sort_values("date", ascending=False).head(6)
    rows: list[dict[str, object]] = []
    for _, event in events.iterrows():
        event_tag = _normalize_event_tag(_text_value(event, "sec_event_tag", "none"))
        rows.append(
            {
                "date": event["date"],
                "event_tag": event_tag,
                "risk_flag": _float_value(event, "sec_risk_flag") > 0,
                "risk_flag_20d": _float_value(event, "sec_risk_flag_20d") > 0,
                "confidence": _float_value(event, "sec_event_confidence"),
                "summary_ref": _text_value(event, "sec_summary_ref", ""),
            }
        )
    return rows


def _backtest_result(metrics: PerformanceMetrics, equity_curve: pd.DataFrame) -> dict[str, object]:
    return {
        "cagr": float(metrics.cagr),
        "sharpe": float(metrics.sharpe),
        "max_drawdown": float(metrics.max_drawdown),
        "hit_rate": float(metrics.hit_rate),
        "turnover": float(metrics.turnover),
        "exposure": float(metrics.exposure),
        "equity_curve": equity_curve.copy(),
    }


def _row_with_latest_sec(row: pd.Series | None, sec_features: pd.DataFrame, ticker: str) -> pd.Series | None:
    if row is not None and "sec_event_tag" in row.index:
        return row
    events = _sec_events(sec_features, ticker)
    if not events:
        return row
    latest_event = events[0]
    data = row.copy() if row is not None else pd.Series(dtype=object)
    data["sec_event_tag"] = latest_event["event_tag"]
    data["sec_event_confidence"] = latest_event["confidence"]
    data["sec_summary_ref"] = latest_event["summary_ref"]
    data["sec_risk_flag"] = 1.0 if latest_event["risk_flag"] else _float_value(data, "sec_risk_flag")
    data["sec_risk_flag_20d"] = 1.0 if latest_event["risk_flag_20d"] else _float_value(data, "sec_risk_flag_20d")
    return pd.Series(data)


def _fallback_badge(
    name: str,
    label: str,
    reason: str,
    next_needed_data: str,
    evidence: dict[str, object] | None = None,
) -> DashboardBadge:
    return DashboardBadge(
        name=name,
        label=label,
        tone="muted",
        evidence=evidence or {},
        status=label,
        reason=reason,
        next_needed_data=next_needed_data,
    )


def _fallback_from_badge(badge: DashboardBadge) -> dict[str, str]:
    return {
        "status": badge.status,
        "reason": badge.reason,
        "next_needed_data": badge.next_needed_data,
    }


def _chart_fallback(chart: dict[str, pd.DataFrame]) -> dict[str, str]:
    if chart["history"].empty or chart["interval"].empty:
        return {
            "status": "자료 부족",
            "reason": "가격 history 또는 예측 구간 데이터가 부족합니다.",
            "next_needed_data": "최근 가격 데이터와 expected/downside/upside 예측값",
        }
    return {"status": "정상", "reason": "", "next_needed_data": ""}


def _backtest_curve_fallback(equity_curve: pd.DataFrame) -> dict[str, str]:
    if equity_curve.empty or "date" not in equity_curve.columns or "equity" not in equity_curve.columns:
        return {
            "status": "자료 부족",
            "reason": "백테스트 equity 곡선이 비어 있습니다.",
            "next_needed_data": "백테스트 결과 equity와 날짜 데이터",
        }
    return {"status": "정상", "reason": "", "next_needed_data": ""}


def _events_fallback(events: list[dict[str, object]]) -> dict[str, str]:
    if not events:
        return {
            "status": "자료 부족",
            "reason": "표시할 SEC 이벤트 카드가 없습니다.",
            "next_needed_data": "분석 기간 내 SEC filing 이벤트",
        }
    return {"status": "정상", "reason": "", "next_needed_data": ""}


def _has_values(row: pd.Series, *columns: str) -> bool:
    return all(column in row.index and pd.notna(row[column]) for column in columns)


def _float_value(row: pd.Series, column: str, default: float = 0.0) -> float:
    if column not in row.index or pd.isna(row[column]):
        return default
    try:
        return float(row[column])
    except (TypeError, ValueError):
        return default


def _text_value(row: pd.Series, column: str, default: str = "") -> str:
    if column not in row.index or pd.isna(row[column]):
        return default
    return str(row[column])


def _contains_any(value: str, needles: set[str]) -> bool:
    lowered = value.lower()
    return any(needle in lowered for needle in needles)


def _is_no_event_tag(event_tag: str) -> bool:
    return _normalize_event_tag(event_tag) in NO_EVENT_TAGS


def _normalize_event_tag(event_tag: object) -> str:
    if pd.isna(event_tag):
        return "none"
    raw = str(event_tag).strip()
    if not raw:
        return "none"
    normalized_tags = [part.strip() for part in raw.split(",") if part.strip()]
    if not normalized_tags:
        return "none"
    normalized = ",".join(part for part in normalized_tags)
    if normalized.lower() in NO_EVENT_TAGS:
        return "none"
    return normalized


def _safe_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return 0.0
    value = pd.to_numeric(frame[column], errors="coerce").mean()
    return 0.0 if pd.isna(value) else float(value)


def _numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index)


def _to_datetime_or_none(value: object) -> pd.Timestamp | None:
    timestamp = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(timestamp) else timestamp
