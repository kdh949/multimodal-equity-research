from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from quant_research.backtest.metrics import PerformanceMetrics
from quant_research.signals.engine import SignalEngineConfig

if TYPE_CHECKING:
    from quant_research.pipeline import PipelineConfig, PipelineResult


DISCLAIMER = "연구용 리서치 화면이며 투자 권고가 아닙니다. 실거래 주문 기능은 제공하지 않습니다."
DECISION_SOURCE = "deterministic_signal_engine"
NO_EVENT_TAGS = {"", "none", "0.0", "nan"}
DIRECTION_REQUIRED_COLUMNS = ("expected_return", "downside_quantile")
FORECAST_REQUIRED_COLUMNS = ("expected_return", "downside_quantile", "upside_quantile")
RISK_REQUIRED_COLUMNS = ("predicted_volatility",)
SEC_REQUIRED_COLUMNS = ("sec_event_confidence", "sec_event_tag", "sec_risk_flag", "sec_risk_flag_20d")
ACTION_LABELS = {"BUY": "긍정적", "SELL": "부정적", "HOLD": "보류"}
ACTION_BEGINNER_SIGNALS = {"BUY": "positive", "SELL": "negative", "HOLD": "hold"}
ACTION_TONES = {"BUY": "positive", "SELL": "negative", "HOLD": "neutral"}
VALIDATION_GATE_LABELS = {
    "pass": "통과",
    "warning": "경고",
    "fail": "실패",
    "missing": "미확인",
    "unknown": "미확인",
}
CONFIDENCE_LABELS = {
    "high": "높음",
    "medium": "보통",
    "low": "낮음",
    "insufficient_validation": "검증 불충분",
}


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
    decision_coach_report: BeginnerDecisionCoachReport | None = None
    disclaimer: str = DISCLAIMER


@dataclass(frozen=True)
class BeginnerDecisionCoachReport:
    ticker: str
    as_of_date: str
    beginner_signal: str
    research_signal: str
    display_label: str
    visual_tone: str
    forecast_direction: str
    forecast_direction_label: str
    confidence_level: str
    confidence_label: str
    decision_source: str
    validation_gate_status: str
    validation_gate_label: str
    reason_codes: tuple[str, ...]
    plain_language_explanation: str
    why_it_might_be_wrong: str
    evidence: dict[str, object]
    advanced_disclosure: dict[str, object]
    not_investment_advice: bool = True
    disclaimer: str = DISCLAIMER


def build_beginner_decision_coach_report(
    result: PipelineResult,
    ticker: str,
    validation_gate_report: object | None = None,
    config: PipelineConfig | None = None,
) -> BeginnerDecisionCoachReport:
    ticker = ticker.upper()
    latest = _latest_signal_row(result, ticker)
    signal_config = SignalEngineConfig(
        cost_bps=getattr(config, "cost_bps", SignalEngineConfig.cost_bps),
        slippage_bps=getattr(config, "slippage_bps", SignalEngineConfig.slippage_bps),
    )
    metrics = result.backtest.metrics
    direction_badge = _direction_badge(latest, signal_config, result.predictions, result.signals)
    risk_badge = _risk_badge(latest, metrics, config, result.predictions, result.signals)
    sec_badge = _sec_impact_badge(latest, result.sec_features, ticker)
    validation_badge = _validation_badge(result.validation_summary, metrics)
    research_signal, signal_row = _strict_latest_signal_action(result.signals, ticker)
    source_row = signal_row if signal_row is not None else latest
    gate = _validation_gate_summary(validation_gate_report)
    evidence = _decision_evidence(source_row, signal_config, gate)
    reason_codes = _decision_reason_codes(
        research_signal,
        direction_badge,
        risk_badge,
        sec_badge,
        validation_badge,
        gate,
        evidence,
    )
    display_label = _decision_display_label(research_signal, gate)
    confidence_level = _decision_confidence_level(gate, direction_badge, risk_badge, sec_badge, evidence)
    explanation = _plain_language_explanation(
        ticker,
        display_label,
        research_signal,
        direction_badge,
        risk_badge,
        sec_badge,
        gate,
        reason_codes,
    )
    why_wrong = _why_decision_might_be_wrong(gate, risk_badge, sec_badge, evidence, reason_codes)
    return BeginnerDecisionCoachReport(
        ticker=ticker,
        as_of_date=str(evidence.get("data_cutoff", "")),
        beginner_signal=_beginner_signal(research_signal, gate),
        research_signal=research_signal,
        display_label=display_label,
        visual_tone=_decision_visual_tone(research_signal, gate),
        forecast_direction=_forecast_direction(direction_badge),
        forecast_direction_label=direction_badge.label,
        confidence_level=confidence_level,
        confidence_label=CONFIDENCE_LABELS[confidence_level],
        decision_source=DECISION_SOURCE,
        validation_gate_status=gate["status"],
        validation_gate_label=VALIDATION_GATE_LABELS[gate["status"]],
        reason_codes=reason_codes,
        plain_language_explanation=explanation,
        why_it_might_be_wrong=why_wrong,
        evidence=evidence,
        advanced_disclosure={
            "raw_signal_visible": False,
            "raw_signal_disclosure_visible": bool(research_signal),
            "raw_signal": research_signal,
            "raw_signal_note": "이 값은 주문 신호가 아니라 deterministic engine의 원천 action입니다.",
            "decision_source": DECISION_SOURCE,
        },
    )


def build_beginner_research_dashboard(
    result: PipelineResult,
    ticker: str,
    config: PipelineConfig | None = None,
    decision_coach_report: BeginnerDecisionCoachReport | None = None,
) -> BeginnerResearchDashboard:
    ticker = ticker.upper()
    signal_config = SignalEngineConfig(
        cost_bps=getattr(config, "cost_bps", SignalEngineConfig.cost_bps),
        slippage_bps=getattr(config, "slippage_bps", SignalEngineConfig.slippage_bps),
    )
    latest = _latest_signal_row(result, ticker)
    metrics = result.backtest.metrics

    direction_badge = _direction_badge(latest, signal_config, result.predictions, result.signals)
    risk_badge = _risk_badge(
        latest,
        metrics,
        config,
        result.predictions,
        result.signals,
    )
    sec_impact_badge = _sec_impact_badge(latest, result.sec_features, ticker)
    validation_badge = _validation_badge(result.validation_summary, metrics)
    forecast_chart = _forecast_interval_chart(result.market_data, latest, ticker)
    sec_events = _sec_events(result.sec_features, ticker)
    fallback_state = {
        "direction": _fallback_from_badge(direction_badge),
        "risk": _fallback_from_badge(risk_badge),
        "sec_impact": _fallback_from_badge(sec_impact_badge),
        "validation": _fallback_from_badge(validation_badge),
        "forecast_interval_chart": _chart_fallback(
            forecast_chart,
            latest,
            result.predictions,
            result.signals,
        ),
        "backtest_equity_curve": _backtest_curve_fallback(result.backtest.equity_curve),
        "sec_events": _events_fallback(sec_events, result.sec_features),
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
        decision_coach_report=decision_coach_report,
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


def _strict_latest_signal_action(signals: pd.DataFrame, ticker: str) -> tuple[str, pd.Series | None]:
    row = _latest_ticker_row(signals, ticker)
    if row is None:
        return "", None
    action = _text_value(row, "action", "").upper()
    if action in ACTION_LABELS:
        return action, row
    return "", row


def _validation_gate_summary(validation_gate_report: object | None) -> dict[str, str]:
    if validation_gate_report is None:
        return {
            "status": "missing",
            "decision": "MISSING",
            "system_validity_status": "unknown",
            "strategy_candidate_status": "unknown",
            "reason": "validation gate report is missing",
        }
    mapping = _object_mapping(validation_gate_report)
    summary = _object_mapping(
        _first_present_from_mapping(
            mapping,
            ("validity_gate_result_summary", "result_summary", "summary"),
        )
    )
    decision = str(
        _first_present_from_mapping(summary, ("final_gate_decision", "final_decision", "decision"))
        or _first_present_from_mapping(mapping, ("final_gate_decision", "final_decision", "decision"))
        or ""
    ).upper()
    system_status = _normalize_status(
        _first_present_from_mapping(summary, ("system_validity_status",))
        or _first_present_from_mapping(mapping, ("system_validity_status",))
        or getattr(validation_gate_report, "system_validity_status", "")
    )
    strategy_status = _normalize_status(
        _first_present_from_mapping(summary, ("strategy_candidate_status", "final_strategy_status", "final_status"))
        or _first_present_from_mapping(mapping, ("strategy_candidate_status", "final_strategy_status", "final_status"))
        or getattr(validation_gate_report, "strategy_candidate_status", "")
    )
    status = _validation_status_from_decision(decision, system_status, strategy_status)
    reason = str(
        _first_present_from_mapping(summary, ("reason", "official_message"))
        or _first_present_from_mapping(mapping, ("official_message", "reason"))
        or getattr(validation_gate_report, "official_message", "")
        or "validation gate did not return PASS"
    )
    return {
        "status": status,
        "decision": decision or "UNKNOWN",
        "system_validity_status": system_status or "unknown",
        "strategy_candidate_status": strategy_status or "unknown",
        "reason": reason,
    }


def _validation_status_from_decision(decision: str, system_status: str, strategy_status: str) -> str:
    if decision == "PASS":
        return "pass"
    if decision == "WARN":
        return "warning"
    if decision == "FAIL":
        return "fail"
    statuses = {system_status, strategy_status}
    if statuses <= {"", "unknown"}:
        return "unknown"
    if "hard_fail" in statuses or "fail" in statuses or "not_evaluable" in statuses:
        return "fail"
    if "warning" in statuses or "warn" in statuses:
        return "warning"
    if system_status == "pass" and strategy_status == "pass":
        return "pass"
    if "insufficient_data" in statuses or "not_evaluated" in statuses:
        return "missing"
    return "unknown"


def _decision_display_label(research_signal: str, gate: Mapping[str, str]) -> str:
    base_label = ACTION_LABELS.get(research_signal, "")
    if gate["status"] == "pass" and base_label:
        return base_label
    if base_label:
        return f"{base_label}이지만 검증 불충분"
    return "검증 불충분"


def _beginner_signal(research_signal: str, gate: Mapping[str, str]) -> str:
    if gate["status"] != "pass":
        return "insufficient_validation"
    return ACTION_BEGINNER_SIGNALS.get(research_signal, "insufficient_validation")


def _decision_visual_tone(research_signal: str, gate: Mapping[str, str]) -> str:
    if gate["status"] == "pass":
        return ACTION_TONES.get(research_signal, "blocked")
    if gate["status"] == "warning":
        return "caution"
    return "blocked"


def _forecast_direction(direction_badge: DashboardBadge) -> str:
    if direction_badge.label == "상승":
        return "up"
    if direction_badge.label == "하락":
        return "down"
    return "uncertain"


def _decision_confidence_level(
    gate: Mapping[str, str],
    direction_badge: DashboardBadge,
    risk_badge: DashboardBadge,
    sec_badge: DashboardBadge,
    evidence: Mapping[str, object],
) -> str:
    if gate["status"] in {"fail", "missing", "unknown"}:
        return "insufficient_validation"
    if gate["status"] == "warning" or any(badge.status != "정상" for badge in (direction_badge, risk_badge, sec_badge)):
        return "low"
    if risk_badge.tone == "negative" or sec_badge.tone == "negative":
        return "low"
    signal_score = _float_from_mapping(evidence, "signal_score")
    volatility = _float_from_mapping(evidence, "predicted_volatility")
    downside = _float_from_mapping(evidence, "downside_quantile")
    if signal_score >= 0.02 and volatility <= 0.02 and downside >= -0.02:
        return "high"
    return "medium"


def _decision_evidence(
    row: pd.Series | None,
    signal_config: SignalEngineConfig,
    gate: Mapping[str, str],
) -> dict[str, object]:
    if row is None:
        return {
            "expected_return": None,
            "signal_score": None,
            "downside_quantile": None,
            "predicted_volatility": None,
            "text_risk_score": None,
            "sec_risk_flag": None,
            "sec_risk_flag_20d": None,
            "sec_event_tag": "none",
            "cost_bps": float(signal_config.cost_bps),
            "slippage_bps": float(signal_config.slippage_bps),
            "data_cutoff": "",
            "validation_gate_status": gate["status"],
            "validation_gate_reason": gate["reason"],
        }
    data_cutoff = _text_value(row, "date", "")
    timestamp = _to_datetime_or_none(data_cutoff)
    if timestamp is not None:
        data_cutoff = timestamp.strftime("%Y-%m-%d")
    return {
        "expected_return": _optional_float_value(row, "expected_return"),
        "signal_score": _optional_float_value(row, "signal_score"),
        "downside_quantile": _optional_float_value(row, "downside_quantile"),
        "predicted_volatility": _optional_float_value(row, "predicted_volatility"),
        "text_risk_score": _optional_float_value(row, "text_risk_score"),
        "sec_risk_flag": _optional_float_value(row, "sec_risk_flag"),
        "sec_risk_flag_20d": _optional_float_value(row, "sec_risk_flag_20d"),
        "sec_event_tag": _optional_event_tag(row, "sec_event_tag"),
        "sec_event_confidence": _optional_float_value(row, "sec_event_confidence"),
        "cost_bps": float(signal_config.cost_bps),
        "slippage_bps": float(signal_config.slippage_bps),
        "data_cutoff": data_cutoff,
        "validation_gate_status": gate["status"],
        "validation_gate_reason": gate["reason"],
    }


def _decision_reason_codes(
    research_signal: str,
    direction_badge: DashboardBadge,
    risk_badge: DashboardBadge,
    sec_badge: DashboardBadge,
    validation_badge: DashboardBadge,
    gate: Mapping[str, str],
    evidence: Mapping[str, object],
) -> tuple[str, ...]:
    codes: list[str] = [f"action_{research_signal.lower()}"] if research_signal else ["action_missing"]
    codes.append(f"validation_gate_{gate['status']}")
    codes.append(f"direction_{_forecast_direction(direction_badge)}")
    codes.append(f"risk_{risk_badge.tone}")
    codes.append(f"sec_{sec_badge.tone}")
    codes.append(f"oos_{validation_badge.tone}")
    if _float_from_mapping(evidence, "expected_return") > 0:
        codes.append("expected_return_positive")
    if _float_from_mapping(evidence, "downside_quantile") < 0:
        codes.append("downside_loss_possible")
    if _float_from_mapping(evidence, "sec_risk_flag") > 0 or _float_from_mapping(evidence, "sec_risk_flag_20d") > 0:
        codes.append("sec_risk_flag")
    if _float_from_mapping(evidence, "text_risk_score") > 0:
        codes.append("text_risk_present")
    return tuple(dict.fromkeys(codes))


def _plain_language_explanation(
    ticker: str,
    display_label: str,
    research_signal: str,
    direction_badge: DashboardBadge,
    risk_badge: DashboardBadge,
    sec_badge: DashboardBadge,
    gate: Mapping[str, str],
    reason_codes: tuple[str, ...],
) -> str:
    variant = _template_variant(ticker, reason_codes)
    source = "이 판단은 LLM 문장이 아니라 deterministic signal engine의 정량 규칙에서 나온 연구 신호입니다."
    if not research_signal:
        return (
            f"쉽게 말해, {ticker}는 아직 최종 연구 신호를 보여줄 만큼 deterministic signal 근거가 충분하지 않습니다. "
            f"검증 상태는 {VALIDATION_GATE_LABELS[gate['status']]}이며, 필요한 신호와 검증 자료가 더 필요합니다. {source}"
        )
    opener_pool = (
        f"쉽게 말해, {ticker}의 현재 표시는 {display_label}입니다.",
        f"현재 데이터만 놓고 보면 {ticker}의 초심자용 표시는 {display_label}입니다.",
        f"이번 연구 실행에서 {ticker}는 {display_label}로 정리됩니다.",
    )
    connector_pool = (
        f"예상 방향은 {direction_badge.label}, 위험도는 {risk_badge.label}, 공시/뉴스 영향은 {sec_badge.label}로 읽혔습니다.",
        f"핵심 근거는 {direction_badge.label} 방향성, {risk_badge.label} 위험도, {sec_badge.label} 공시 영향입니다.",
        f"가격/리스크/공시 근거를 함께 보면 방향성은 {direction_badge.label}, 위험도는 {risk_badge.label}입니다.",
    )
    gate_sentence = (
        "검증 gate가 통과되어 standalone label로 보여줍니다."
        if gate["status"] == "pass"
        else f"다만 검증 gate가 {VALIDATION_GATE_LABELS[gate['status']]} 상태라서 standalone 신호가 아니라 검증 불충분 label로 낮춰 보여줍니다."
    )
    return f"{opener_pool[variant]} {connector_pool[variant]} {gate_sentence} {source}"


def _why_decision_might_be_wrong(
    gate: Mapping[str, str],
    risk_badge: DashboardBadge,
    sec_badge: DashboardBadge,
    evidence: Mapping[str, object],
    reason_codes: tuple[str, ...],
) -> str:
    variant = _template_variant(str(evidence.get("data_cutoff", "")), reason_codes)
    if gate["status"] != "pass":
        return (
            "확인할 점: 검증 gate가 완전히 통과하지 않았기 때문에 이 표시는 최종 확정 신호처럼 읽으면 안 됩니다. "
            f"gate reason: {gate['reason']}"
        )
    warnings = []
    if risk_badge.tone == "negative" or _float_from_mapping(evidence, "predicted_volatility") > 0.04:
        warnings.append("변동성이 커지면 작은 예상수익률은 쉽게 사라질 수 있습니다")
    if sec_badge.tone == "negative" or "sec_risk_flag" in reason_codes:
        warnings.append("SEC나 뉴스 리스크가 새로 커지면 판단이 빠르게 바뀔 수 있습니다")
    if _float_from_mapping(evidence, "downside_quantile") < -0.03:
        warnings.append("하방 구간이 넓어지면 손실 가능성을 먼저 봐야 합니다")
    fallback = (
        "확인할 점: 이 판단은 현재 data cutoff까지의 feature만 사용하므로, 이후 가격 급변이나 새 공시가 나오면 달라질 수 있습니다.",
        "다만 이 신호는 개인 포트폴리오나 현금 상황을 반영하지 않습니다. 새 데이터가 들어오면 같은 규칙으로 다시 확인해야 합니다.",
        "주의할 점: 비용, 슬리피지, 변동성 조건이 나빠지면 같은 방향성이라도 실제 연구 판단은 약해질 수 있습니다.",
    )
    if warnings:
        return "확인할 점: " + "; ".join(warnings) + "."
    return fallback[variant]


def _template_variant(seed: str, reason_codes: tuple[str, ...]) -> int:
    value = sum(ord(ch) for ch in f"{seed}|{'|'.join(reason_codes)}")
    return value % 3


def _direction_badge(
    row: pd.Series | None,
    config: SignalEngineConfig,
    predictions: pd.DataFrame,
    signals: pd.DataFrame,
) -> DashboardBadge:
    if row is None or not _has_values(row, *DIRECTION_REQUIRED_COLUMNS):
        if _model_inactive(row, DIRECTION_REQUIRED_COLUMNS, predictions, signals):
            return _fallback_badge(
                "방향성",
                "모델 비활성",
                "방향성 예측 컬럼이 현재 비활성된 모델 출력에서 누락되었습니다.",
                "예측 모델을 활성화하고 expected_return, downside_quantile 를 산출",
            )
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
    predictions: pd.DataFrame,
    signals: pd.DataFrame,
) -> DashboardBadge:
    if row is None or not _has_values(row, *RISK_REQUIRED_COLUMNS):
        if _model_inactive(row, RISK_REQUIRED_COLUMNS, predictions, signals):
            return _fallback_badge(
                "위험도",
                "모델 비활성",
                "위험도 예측 모델이 비활성되어 변동성 값이 누락되었습니다.",
                "예측 모델을 활성화하고 predicted_volatility 를 산출",
            )
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
        if _model_inactive(
            row,
            SEC_REQUIRED_COLUMNS,
            sec_features,
        ):
            return _fallback_badge(
                "공시 영향",
                "모델 비활성",
                "SEC 이벤트 추출 모델에서 필요한 컬럼이 없어 상태를 계산할 수 없습니다.",
                "Filing event extractor 또는 sec_event_* 컬럼이 포함된 SEC feature 파이프라인 활성화",
            )
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


def _chart_fallback(
    chart: dict[str, pd.DataFrame],
    row: pd.Series | None,
    predictions: pd.DataFrame,
    signals: pd.DataFrame,
) -> dict[str, str]:
    if chart["history"].empty or chart["interval"].empty:
        if _model_inactive(row, FORECAST_REQUIRED_COLUMNS, predictions, signals):
            return {
                "status": "모델 비활성",
                "reason": "예측 구간을 그리려면 타임시리즈 예측 모델 결과가 필요합니다.",
                "next_needed_data": "expected_return, downside_quantile, upside_quantile 를 함께 산출하는 예측 컬럼",
            }
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


def _events_fallback(events: list[dict[str, object]], sec_features: pd.DataFrame) -> dict[str, str]:
    if not events:
        if _model_inactive(None, SEC_REQUIRED_COLUMNS, sec_features):
            return {
                "status": "모델 비활성",
                "reason": "SEC 이벤트 추출 모델이 비활성되어 이벤트 타임라인을 만들 수 없습니다.",
                "next_needed_data": "sec_event_confidence, sec_event_tag, sec_risk_flag_20d 출력을 포함하는 SEC 모델",
            }
        return {
            "status": "자료 부족",
            "reason": "표시할 SEC 이벤트 카드가 없습니다.",
            "next_needed_data": "분석 기간 내 SEC filing 이벤트",
        }
    return {"status": "정상", "reason": "", "next_needed_data": ""}


def _has_values(row: pd.Series, *columns: str) -> bool:
    return all(column in row.index and pd.notna(row[column]) for column in columns)


def _model_inactive(row: pd.Series | None, required_columns: tuple[str, ...], *frames: pd.DataFrame) -> bool:
    if row is not None and not set(required_columns).issubset(set(row.index)):
        return True
    for frame in frames:
        if frame.empty:
            continue
        if not set(required_columns).issubset(set(frame.columns)):
            return True
    return False


def _float_value(row: pd.Series, column: str, default: float = 0.0) -> float:
    if column not in row.index or pd.isna(row[column]):
        return default
    try:
        return float(row[column])
    except (TypeError, ValueError):
        return default


def _optional_float_value(row: pd.Series, column: str) -> float | None:
    if column not in row.index or pd.isna(row[column]):
        return None
    try:
        return float(row[column])
    except (TypeError, ValueError):
        return None


def _text_value(row: pd.Series, column: str, default: str = "") -> str:
    if column not in row.index or pd.isna(row[column]):
        return default
    return str(row[column])


def _optional_event_tag(row: pd.Series, column: str) -> str | None:
    if column not in row.index or pd.isna(row[column]):
        return None
    return _normalize_event_tag(str(row[column]))


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


def _object_mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return payload
    return {}


def _first_present_from_mapping(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _normalize_status(value: object) -> str:
    return str(value or "").strip().lower()


def _float_from_mapping(mapping: Mapping[str, object], key: str, default: float = 0.0) -> float:
    value = mapping.get(key, default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_datetime_or_none(value: object) -> pd.Timestamp | None:
    timestamp = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(timestamp) else timestamp
