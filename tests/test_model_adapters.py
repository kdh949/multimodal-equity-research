from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import requests

from quant_research.models.ollama import OllamaAgent
from quant_research.models.tabular import TabularReturnModel
from quant_research.models.text import (
    FilingEventExtractor,
    FinBERTSentimentAnalyzer,
    FinGPTEventExtractor,
)
from quant_research.models.timeseries import Chronos2Adapter, GraniteTTMAdapter


def test_chronos2_proxy_forecast_columns_are_added() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                ]
            ),
            "ticker": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"],
            "return_20": [0.0, 0.05, -0.02, -0.03, 0.01, 0.02],
            "volatility_20": [0.12, 0.10, 0.11, 0.14, 0.13, None],
        }
    )

    output = Chronos2Adapter().add_proxy_forecasts(frame)

    assert {
        "chronos_expected_return",
        "chronos_downside_quantile",
        "chronos_upside_quantile",
        "chronos_quantile_width",
    } <= set(output.columns)

    momentum = frame.groupby("ticker")["return_20"].transform(lambda series: series.shift(1).fillna(0) / 20)
    volatility = frame["volatility_20"].fillna(frame["volatility_20"].median()).fillna(0.02)

    pd.testing.assert_series_equal(output["chronos_expected_return"], momentum, check_names=False)
    pd.testing.assert_series_equal(
        output["chronos_downside_quantile"],
        momentum - 1.28 * volatility,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        output["chronos_upside_quantile"],
        momentum + 1.28 * volatility,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        output["chronos_quantile_width"],
        output["chronos_upside_quantile"] - output["chronos_downside_quantile"],
        check_names=False,
    )


def test_granite_ttm_proxy_forecast_columns_are_added() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                ]
            ),
            "ticker": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"],
            "return_5": [0.01, 0.00, 0.04, -0.01, 0.03, 0.02],
            "high_low_range": [1.2, 0.8, 1.0, 0.5, 0.6, 0.9],
        }
    )

    output = GraniteTTMAdapter().add_proxy_forecasts(frame)

    assert {"granite_ttm_expected_return", "granite_ttm_confidence"} <= set(output.columns)

    short_signal = frame.groupby("ticker")["return_5"].transform(lambda series: series.shift(1).fillna(0) / 5)
    pd.testing.assert_series_equal(
        output["granite_ttm_expected_return"],
        short_signal - 0.1 * frame["high_low_range"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        output["granite_ttm_confidence"],
        (1 / (1 + np.exp(-abs(short_signal) * 100))).clip(0, 1),
        check_names=False,
    )


def test_tabular_model_fallback_schema_when_lightgbm_is_absent(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "lightgbm", types.ModuleType("lightgbm"))

    train = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=40, freq="D"),
            "ticker": ["AAPL"] * 20 + ["MSFT"] * 20,
            "volatility_20": [0.01 + i * 0.001 for i in range(40)],
            "return_5": [0.001 * (i - 20) for i in range(40)],
            "return_20": [0.002 * (i - 10) for i in range(40)],
            "forward_return_1": [0.001 * (i % 5 - 2) for i in range(40)],
        }
    )

    model = TabularReturnModel(model_name="lightgbm", random_state=7).fit(train)
    assert model.actual_model_name == "HistGradientBoostingRegressor"

    predictions = model.predict(train)
    assert {
        "date",
        "ticker",
        "expected_return",
        "predicted_volatility",
        "downside_quantile",
        "upside_quantile",
        "quantile_width",
        "model_confidence",
        "model_name",
    } <= set(predictions.columns)


def test_finbert_uses_no_network_fallback_contract(monkeypatch) -> None:
    fake_transformers = types.ModuleType("transformers")

    def failing_pipeline(*args, **kwargs):
        raise RuntimeError("transformers unavailable in this environment")

    fake_transformers.pipeline = failing_pipeline
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    analyzer = FinBERTSentimentAnalyzer()
    result = analyzer.score("Company announced stronger guidance and improving demand.")

    assert set(result) == {"sentiment_score", "negative_flag", "label", "confidence", "event_tag", "risk_flag"}
    assert isinstance(result["sentiment_score"], float)
    assert isinstance(result["confidence"], float)
    assert isinstance(result["negative_flag"], bool)
    assert isinstance(result["risk_flag"], bool)


def test_filing_and_fingpt_extractor_follow_structured_contract_without_services() -> None:
    for extractor in (FilingEventExtractor(), FinGPTEventExtractor()):
        result = extractor.extract("Company reported an earnings guidance update with potential legal risk.")
        assert set(result) == {"event_tag", "risk_flag", "confidence", "summary_ref"}
        assert isinstance(result["event_tag"], str)
        assert isinstance(result["risk_flag"], bool)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["summary_ref"], str)


def test_ollama_agent_falls_back_when_server_unreachable(monkeypatch) -> None:
    def failing_request(*args, **kwargs):
        raise requests.RequestException("server unreachable")

    monkeypatch.setattr(requests, "post", failing_request)

    response = OllamaAgent().explain("Summarize risk in this sample text.")
    assert response == "Ollama is unavailable; deterministic metrics and signals were generated locally."
