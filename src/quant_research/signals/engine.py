from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class SignalGenerationBlockedError(RuntimeError):
    """Raised when the validation gate does not permit final signal generation."""


@dataclass(frozen=True)
class SignalEngineConfig:
    min_expected_return: float = 0.005
    min_signal_score: float = 0.003
    max_predicted_volatility: float = 0.08
    min_downside_quantile: float = -0.05
    max_text_risk_score: float = 0.65
    max_sec_risk_flag: float = 2.0
    max_sec_risk_20d: float = 4.0
    block_buy_sec_risk_flag: float = 1.0
    block_buy_sec_risk_20d: float = 1.0
    moderate_sec_risk_flag: float = 0.5
    moderate_sec_risk_20d: float = 0.5
    max_news_negative_ratio: float = 0.75
    min_liquidity_score: float = 10.0
    cost_bps: float = 5.0
    slippage_bps: float = 2.0
    confidence_weight: float = 0.003
    volatility_penalty_weight: float = 0.20
    text_risk_penalty_weight: float = 0.012
    sec_risk_penalty_weight: float = 0.010
    sec_risk_20d_penalty_weight: float = 0.004
    news_negative_penalty_weight: float = 0.006
    downside_penalty_weight: float = 0.20
    portfolio_volatility_limit: float = 0.04
    average_daily_turnover_budget: float = 0.25
    max_symbol_weight: float = 0.10
    max_sector_weight: float = 0.30
    covariance_aware_risk_enabled: bool = True
    portfolio_covariance_lookback: int = 20
    covariance_return_column: str = "return_1"
    covariance_min_periods: int = 20
    max_drawdown_floor: float = -0.20
    portfolio_volatility_penalty_weight: float = 0.50
    turnover_penalty_weight: float = 0.030
    concentration_penalty_weight: float = 0.040
    drawdown_penalty_weight: float = 0.050
    risk_stop_penalty: float = 1.0

    def __post_init__(self) -> None:
        if self.portfolio_covariance_lookback < 1:
            raise ValueError("portfolio_covariance_lookback must be positive")
        if not str(self.covariance_return_column).strip():
            raise ValueError("covariance_return_column must not be blank")
        if self.covariance_min_periods < 2:
            raise ValueError("covariance_min_periods must be at least 2")
        if self.covariance_min_periods > self.portfolio_covariance_lookback:
            raise ValueError("covariance_min_periods must not exceed portfolio_covariance_lookback")


class DeterministicSignalEngine:
    def __init__(self, config: SignalEngineConfig | None = None) -> None:
        self.config = config or SignalEngineConfig()

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        _validate_signal_input_timing(output)
        cost = (self.config.cost_bps + self.config.slippage_bps) / 10_000
        expected = output["expected_return"].fillna(0.0)
        volatility = _series(output, "predicted_volatility", 0.02).fillna(0.02)
        downside = _series(output, "downside_quantile", -0.02).fillna(-0.02)
        text_risk = _series(output, "text_risk_score", 0.0).fillna(0.0)
        sec_risk = _series(output, "sec_risk_flag", 0.0).fillna(0.0)
        sec_risk_20d = _series(output, "sec_risk_flag_20d", 0.0).fillna(0.0)
        news_negative = _series(output, "news_negative_ratio", 0.0).fillna(0.0)
        confidence = _series(output, "model_confidence", 0.0).fillna(0.0)
        portfolio_risk_penalty = self._portfolio_risk_penalty(output)

        moderate_sec_risk = (
            np.maximum(0, sec_risk - self.config.moderate_sec_risk_flag)
            + np.maximum(0, sec_risk_20d - self.config.moderate_sec_risk_20d)
        )
        risk_penalty = (
            self.config.volatility_penalty_weight * volatility
            + self.config.text_risk_penalty_weight * text_risk
            + self.config.sec_risk_penalty_weight * sec_risk
            + self.config.sec_risk_20d_penalty_weight * sec_risk_20d
            + self.config.news_negative_penalty_weight * news_negative
            + self.config.sec_risk_penalty_weight * moderate_sec_risk
        )
        downside_penalty = (
            np.maximum(0, self.config.min_downside_quantile - downside)
            * self.config.downside_penalty_weight
        )
        output["signal_score"] = (
            expected
            + self.config.confidence_weight * confidence
            - risk_penalty
            - downside_penalty
            - portfolio_risk_penalty
            - cost
        )
        output["risk_metric_penalty"] = portfolio_risk_penalty
        output["covariance_aware_risk_enabled"] = self.config.covariance_aware_risk_enabled
        output["portfolio_covariance_lookback"] = self.config.portfolio_covariance_lookback
        output["covariance_return_column"] = self.config.covariance_return_column
        output["covariance_min_periods"] = self.config.covariance_min_periods
        output["portfolio_volatility_limit"] = self.config.portfolio_volatility_limit
        output["average_daily_turnover_budget"] = self.config.average_daily_turnover_budget
        output["configured_max_symbol_weight"] = self.config.max_symbol_weight
        output["configured_max_sector_weight"] = self.config.max_sector_weight
        return output

    def generate(
        self,
        frame: pd.DataFrame,
        *,
        validation_gate: object | None = None,
        require_validation_gate: bool = False,
    ) -> pd.DataFrame:
        gate_payload = require_signal_generation_gate_pass(
            validation_gate,
            required=require_validation_gate,
        )
        scored = self.score(frame)
        cfg = self.config
        text_risk = _series(scored, "text_risk_score", 0.0)
        sec_risk = _series(scored, "sec_risk_flag", 0.0)
        sec_risk_20d = _series(scored, "sec_risk_flag_20d", 0.0)
        news_negative = _series(scored, "news_negative_ratio", 0.0)
        liquidity = _series(scored, "liquidity_score", cfg.min_liquidity_score)
        buy_mask = (
            (scored["expected_return"] >= cfg.min_expected_return)
            & (scored["signal_score"] >= cfg.min_signal_score)
            & (scored["predicted_volatility"] <= cfg.max_predicted_volatility)
            & (scored["downside_quantile"] >= cfg.min_downside_quantile)
            & (text_risk <= cfg.max_text_risk_score)
            & (sec_risk < cfg.block_buy_sec_risk_flag)
            & (sec_risk_20d < cfg.block_buy_sec_risk_20d)
            & (news_negative <= cfg.max_news_negative_ratio)
            & (liquidity >= cfg.min_liquidity_score)
        )
        sell_mask = (
            (text_risk > cfg.max_text_risk_score)
            | (sec_risk >= cfg.max_sec_risk_flag)
            | (sec_risk_20d >= cfg.max_sec_risk_20d)
            | (news_negative > cfg.max_news_negative_ratio)
            | (scored["downside_quantile"] < cfg.min_downside_quantile)
        )
        scored["action"] = "HOLD"
        scored.loc[sell_mask, "action"] = "SELL"
        scored.loc[buy_mask, "action"] = "BUY"
        if gate_payload is not None:
            scored["signal_generation_gate_decision"] = gate_payload["final_gate_decision"]
            scored["signal_generation_gate_status"] = gate_payload["final_status"]
        return scored

    def _portfolio_risk_penalty(self, frame: pd.DataFrame) -> pd.Series:
        cfg = self.config
        if not cfg.covariance_aware_risk_enabled:
            return pd.Series(0.0, index=frame.index)
        volatility = _first_series(
            frame,
            ("portfolio_volatility_estimate", "portfolio_volatility"),
            0.0,
        )
        turnover = _first_series(
            frame,
            ("average_daily_turnover", "period_turnover", "turnover"),
            0.0,
        )
        symbol_weight = _first_series(frame, ("max_symbol_weight",), 0.0)
        sector_weight = _first_series(frame, ("max_sector_weight",), 0.0)
        drawdown = _first_series(frame, ("current_drawdown", "drawdown"), 0.0)
        risk_stop_active = _first_series(frame, ("risk_stop_active",), 0.0)

        volatility_excess = np.maximum(0.0, volatility - cfg.portfolio_volatility_limit)
        turnover_excess = np.maximum(0.0, turnover - cfg.average_daily_turnover_budget)
        symbol_excess = np.maximum(0.0, symbol_weight - cfg.max_symbol_weight)
        sector_excess = np.maximum(0.0, sector_weight - cfg.max_sector_weight)
        drawdown_excess = np.maximum(0.0, cfg.max_drawdown_floor - drawdown)
        risk_stop = (risk_stop_active > 0).astype(float)

        return (
            cfg.portfolio_volatility_penalty_weight * volatility_excess
            + cfg.turnover_penalty_weight * turnover_excess
            + cfg.concentration_penalty_weight * (symbol_excess + sector_excess)
            + cfg.drawdown_penalty_weight * drawdown_excess
            + cfg.risk_stop_penalty * risk_stop
        )


def _series(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in frame:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(default, index=frame.index)


def _first_series(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    default: float,
) -> pd.Series:
    for column in columns:
        if column in frame:
            return _series(frame, column, default).fillna(default)
    return pd.Series(default, index=frame.index)


def _validate_signal_input_timing(frame: pd.DataFrame) -> None:
    if frame.empty or "date" not in frame:
        return
    from quant_research.data.timestamps import (
        date_end_utc,
        timestamp_utc,
        validate_generated_feature_cutoffs,
    )

    validate_generated_feature_cutoffs(frame, label="signal engine input")
    sample_timestamp = date_end_utc(frame["date"])
    for column in _signal_prediction_cutoff_columns(frame):
        prediction_time = timestamp_utc(frame[column])
        violation = prediction_time.notna() & sample_timestamp.notna() & (prediction_time > sample_timestamp)
        if violation.any():
            first_index = int(np.flatnonzero(violation.to_numpy())[0])
            raise ValueError(
                f"signal engine input column {column} is later than signal date "
                f"{frame.loc[first_index, 'date']}"
            )


def _signal_prediction_cutoff_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in ("prediction_date", "prediction_timestamp", "model_prediction_timestamp")
        if column in frame
    ]


def require_signal_generation_gate_pass(
    validation_gate: object | None,
    *,
    required: bool = False,
) -> dict[str, str] | None:
    """Block final signal generation unless the common validity gate is PASS.

    Low-level research scoring can still call the engine without a gate. Production-like
    final signal paths should pass a ValidationGateReport or its serialized dict and set
    ``required=True`` so missing, WARN, FAIL, hard-fail, or not-evaluable gates cannot
    emit BUY/SELL/HOLD actions.
    """

    if validation_gate is None:
        if required:
            raise SignalGenerationBlockedError(
                "validation gate is required before final signal generation"
            )
        return None

    payload = _signal_generation_gate_payload(validation_gate)
    final_decision = payload["final_gate_decision"]
    if final_decision != "PASS":
        status = payload["final_status"]
        reason = payload["reason"]
        raise SignalGenerationBlockedError(
            "validation gate blocked final signal generation: "
            f"final_gate_decision={final_decision}, final_status={status}, reason={reason}"
        )
    return payload


def _signal_generation_gate_payload(validation_gate: object) -> dict[str, str]:
    mapping = _gate_mapping(validation_gate)
    summary = _gate_summary(mapping, validation_gate)
    final_decision = _normalize_gate_decision(
        _first_present(
            summary,
            mapping,
            ("final_gate_decision", "final_decision", "decision"),
        )
    )
    system_status = _normalize_gate_status(
        _first_present(summary, mapping, ("system_validity_status",))
        or getattr(validation_gate, "system_validity_status", None)
    )
    strategy_status = _normalize_gate_status(
        _first_present(
            summary,
            mapping,
            ("strategy_candidate_status", "final_strategy_status", "final_status"),
        )
        or getattr(validation_gate, "strategy_candidate_status", None)
    )
    if final_decision is None:
        final_decision = (
            "PASS"
            if system_status == "pass" and strategy_status == "pass"
            else "FAIL"
        )
    final_status = strategy_status or system_status or final_decision.lower()
    reason = str(
        _first_present(summary, mapping, ("reason", "official_message"))
        or getattr(validation_gate, "official_message", "")
        or "validation gate did not return PASS"
    )
    return {
        "final_gate_decision": final_decision,
        "final_status": final_status,
        "system_validity_status": system_status or "unknown",
        "strategy_candidate_status": strategy_status or "unknown",
        "reason": reason,
    }


def _gate_mapping(validation_gate: object) -> Mapping[str, Any]:
    if isinstance(validation_gate, Mapping):
        return validation_gate
    to_dict = getattr(validation_gate, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return payload
    return {}


def _gate_summary(
    mapping: Mapping[str, Any],
    validation_gate: object,
) -> Mapping[str, Any]:
    summary = mapping.get("validity_gate_result_summary")
    if isinstance(summary, Mapping):
        return summary
    metrics = mapping.get("metrics")
    if isinstance(metrics, Mapping):
        nested = metrics.get("validity_gate_result_summary")
        if isinstance(nested, Mapping):
            return nested
    property_summary = getattr(validation_gate, "validity_gate_result_summary", None)
    if isinstance(property_summary, Mapping):
        return property_summary
    return {}


def _first_present(
    primary: Mapping[str, Any],
    secondary: Mapping[str, Any],
    keys: tuple[str, ...],
) -> Any:
    for source in (primary, secondary):
        for key in keys:
            value = source.get(key)
            if value is not None:
                return value
    return None


def _normalize_gate_decision(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    if normalized in {"PASS", "WARN", "WARNING", "FAIL", "HARD_FAIL", "NOT_EVALUABLE"}:
        if normalized == "WARNING":
            return "WARN"
        if normalized in {"HARD_FAIL", "NOT_EVALUABLE"}:
            return "FAIL"
        return normalized
    if normalized in {"TRUE", "1"}:
        return "PASS"
    if normalized in {"FALSE", "0"}:
        return "FAIL"
    return normalized or None


def _normalize_gate_status(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized == "warn":
        return "warning"
    return normalized or None
