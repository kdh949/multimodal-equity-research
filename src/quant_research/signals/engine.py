from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalEngineConfig:
    min_expected_return: float = 0.001
    min_signal_score: float = 0.0005
    max_predicted_volatility: float = 0.08
    min_downside_quantile: float = -0.05
    max_text_risk_score: float = 0.65
    max_sec_risk_flag: float = 0.0
    min_liquidity_score: float = 10.0
    cost_bps: float = 5.0
    slippage_bps: float = 2.0


class DeterministicSignalEngine:
    def __init__(self, config: SignalEngineConfig | None = None) -> None:
        self.config = config or SignalEngineConfig()

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        cost = (self.config.cost_bps + self.config.slippage_bps) / 10_000
        expected = output["expected_return"].fillna(0.0)
        volatility = output.get("predicted_volatility", pd.Series(0.02, index=output.index)).fillna(0.02)
        downside = output.get("downside_quantile", pd.Series(-0.02, index=output.index)).fillna(-0.02)
        text_risk = output.get("text_risk_score", pd.Series(0.0, index=output.index)).fillna(0.0)
        sec_risk = output.get("sec_risk_flag", pd.Series(0.0, index=output.index)).fillna(0.0)
        confidence = output.get("model_confidence", pd.Series(0.0, index=output.index)).fillna(0.0)

        risk_penalty = 0.25 * volatility + 0.01 * text_risk + 0.015 * sec_risk
        downside_penalty = np.maximum(0, self.config.min_downside_quantile - downside) * 0.25
        output["signal_score"] = expected + 0.002 * confidence - risk_penalty - downside_penalty - cost
        return output

    def generate(self, frame: pd.DataFrame) -> pd.DataFrame:
        scored = self.score(frame)
        cfg = self.config
        buy_mask = (
            (scored["expected_return"] >= cfg.min_expected_return)
            & (scored["signal_score"] >= cfg.min_signal_score)
            & (scored["predicted_volatility"] <= cfg.max_predicted_volatility)
            & (scored["downside_quantile"] >= cfg.min_downside_quantile)
            & (scored.get("text_risk_score", 0) <= cfg.max_text_risk_score)
            & (scored.get("sec_risk_flag", 0) <= cfg.max_sec_risk_flag)
            & (scored.get("liquidity_score", cfg.min_liquidity_score) >= cfg.min_liquidity_score)
        )
        sell_mask = (
            (scored.get("text_risk_score", 0) > cfg.max_text_risk_score)
            | (scored.get("sec_risk_flag", 0) > cfg.max_sec_risk_flag)
            | (scored["downside_quantile"] < cfg.min_downside_quantile)
        )
        scored["action"] = "HOLD"
        scored.loc[sell_mask, "action"] = "SELL"
        scored.loc[buy_mask, "action"] = "BUY"
        return scored
