from __future__ import annotations

import pandas as pd

from quant_research.signals.engine import DeterministicSignalEngine, SignalEngineConfig


def test_signal_engine_buys_only_after_risk_checks() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"] * 3),
            "ticker": ["GOOD", "RISK", "WEAK"],
            "expected_return": [0.02, 0.03, 0.0001],
            "predicted_volatility": [0.02, 0.02, 0.02],
            "downside_quantile": [-0.01, -0.01, -0.01],
            "text_risk_score": [0.0, 0.9, 0.0],
            "sec_risk_flag": [0.0, 0.0, 0.0],
            "liquidity_score": [20.0, 20.0, 20.0],
            "model_confidence": [0.8, 0.8, 0.8],
        }
    )

    signals = DeterministicSignalEngine(SignalEngineConfig()).generate(frame)
    actions = dict(zip(signals["ticker"], signals["action"], strict=True))

    assert actions["GOOD"] == "BUY"
    assert actions["RISK"] == "SELL"
    assert actions["WEAK"] == "HOLD"


def test_signal_engine_blocks_low_liquidity() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"]),
            "ticker": ["THIN"],
            "expected_return": [0.05],
            "predicted_volatility": [0.01],
            "downside_quantile": [0.0],
            "text_risk_score": [0.0],
            "sec_risk_flag": [0.0],
            "liquidity_score": [1.0],
            "model_confidence": [1.0],
        }
    )

    signals = DeterministicSignalEngine(SignalEngineConfig()).generate(frame)

    assert signals["action"].iloc[0] == "HOLD"
