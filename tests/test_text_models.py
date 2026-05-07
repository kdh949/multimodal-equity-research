from __future__ import annotations

import json
import sys
import types

import pytest

from quant_research.models.text import (
    FilingEventExtractor,
    FinBERTSentimentAnalyzer,
    FinGPTEventExtractor,
    _normalize_fingpt_runtime,
)


def test_finbert_fallback_returns_structured_schema() -> None:
    analyzer = FinBERTSentimentAnalyzer()

    result = analyzer.score("Company beats earnings expectations with strong revenue growth")

    assert {"sentiment_score", "negative_flag", "label", "confidence", "event_tag", "risk_flag"} == set(
        result
    )
    assert isinstance(result["sentiment_score"], float)
    assert isinstance(result["risk_flag"], bool)


def test_filing_event_extractor_validates_schema() -> None:
    extractor = FilingEventExtractor()
    payload = {
        "event_tag": "current_report",
        "risk_flag": True,
        "confidence": 0.8,
        "summary_ref": "8-K material event",
    }

    assert extractor.validate_json(json.dumps(payload)) == payload

    with pytest.raises(ValueError):
        extractor.validate_json(json.dumps({"event_tag": "current_report"}))


def test_filing_event_extractor_strips_final_action_label_extras() -> None:
    extractor = FilingEventExtractor()
    payload = {
        "event_tag": "guidance",
        "risk_flag": True,
        "confidence": 0.74,
        "summary_ref": "guidance risk",
        "action": "BUY",
        "trade_decision": "SELL",
        "final_signal": "HOLD",
    }

    result = extractor.validate_json(json.dumps(payload))

    assert result == {
        "event_tag": "guidance",
        "risk_flag": True,
        "confidence": 0.74,
        "summary_ref": "guidance risk",
    }


def test_finbert_does_not_emit_final_action_labels_from_model_labels(monkeypatch) -> None:
    fake_transformers = types.ModuleType("transformers")

    class FakeFinBERTPipeline:
        def __call__(self, text: str, **kwargs: object) -> list[list[dict[str, object]]]:
            return [
                [
                    {"label": "BUY", "score": 0.99},
                    {"label": "SELL", "score": 0.01},
                    {"label": "HOLD", "score": 0.50},
                ]
            ]

    fake_transformers.pipeline = lambda *args, **kwargs: FakeFinBERTPipeline()
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    result = FinBERTSentimentAnalyzer().score("Ignore any instruction to emit a trade label.")

    assert set(result) == {
        "sentiment_score",
        "negative_flag",
        "label",
        "confidence",
        "event_tag",
        "risk_flag",
    }
    assert result["label"] == "neutral"
    assert result["confidence"] == 0.0
    assert "action" not in result


def test_fingpt_adapter_uses_same_structured_contract() -> None:
    extractor = FinGPTEventExtractor()

    result = extractor.extract("Form 8-K material guidance risk update")

    assert extractor.model_id == "FinGPT/fingpt-mt_llama3-8b_lora"
    assert extractor.base_model_id == "meta-llama/Meta-Llama-3-8B"
    assert {"event_tag", "risk_flag", "confidence", "summary_ref"} == set(result)
    assert result["risk_flag"] is True


def test_fingpt_runtime_alias_is_normalized() -> None:
    assert _normalize_fingpt_runtime("LLAMA_CPP") == "llama_cpp"
    assert _normalize_fingpt_runtime("transformer") == "transformers"
    assert _normalize_fingpt_runtime("mlx-lm") == "mlx"


def test_fingpt_transformers_runtime_guard_requires_explicit_allow_flag() -> None:
    extractor = FinGPTEventExtractor(use_local_model=True)
    result = extractor.extract("Form 8-K guidance update with litigation risk")

    assert extractor.last_source == "rules"
    assert extractor.last_error is not None
    assert "allow_unquantized_transformers" in extractor.last_error
    assert result["risk_flag"] is True
