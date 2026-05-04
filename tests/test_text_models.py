from __future__ import annotations

import json

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
