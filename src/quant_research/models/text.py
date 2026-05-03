from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from quant_research.features.text import KeywordSentimentAnalyzer


@dataclass
class FinBERTSentimentAnalyzer:
    model_id: str = "ProsusAI/finbert"
    _pipeline: Any = None
    _fallback: KeywordSentimentAnalyzer = field(default_factory=KeywordSentimentAnalyzer)

    def available(self) -> bool:
        try:
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    def score(self, text: str) -> dict[str, float | str | bool]:
        if self._pipeline is None:
            try:
                from transformers import pipeline

                self._pipeline = pipeline("text-classification", model=self.model_id, top_k=None)
            except Exception:
                return self._fallback.score(text)

        result = self._pipeline(text[:3000])
        labels = {item["label"].lower(): float(item["score"]) for item in result[0]}
        positive = labels.get("positive", 0.0)
        negative = labels.get("negative", 0.0)
        neutral = labels.get("neutral", 0.0)
        sentiment = positive - negative
        label = max(labels, key=labels.get) if labels else "neutral"
        return {
            "sentiment_score": sentiment,
            "negative_flag": negative > max(positive, neutral),
            "label": label,
            "confidence": max(labels.values()) if labels else 0.0,
            "event_tag": "",
            "risk_flag": negative > 0.5,
        }


@dataclass
class FilingEventExtractor:
    model_id: str = "TheFinAI/finma-7b-nlp"

    def extract(self, text: str) -> dict[str, object]:
        lower = text.lower()
        event_tags = []
        for keyword, tag in {
            "material": "material_event",
            "restatement": "restatement",
            "guidance": "guidance",
            "earnings": "earnings",
            "risk": "risk_factor",
            "lawsuit": "legal",
            "insider": "insider_activity",
        }.items():
            if keyword in lower:
                event_tags.append(tag)
        risk_flag = bool(re.search(r"\b(risk|restatement|investigation|lawsuit|impairment)\b", lower))
        return {
            "event_tag": ",".join(sorted(set(event_tags))) or "none",
            "risk_flag": risk_flag,
            "confidence": 0.65 if event_tags else 0.35,
            "summary_ref": _stable_summary_ref(text),
        }

    def validate_json(self, raw: str) -> dict[str, object]:
        payload = json.loads(raw)
        required = {"event_tag", "risk_flag", "confidence", "summary_ref"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"Missing filing extraction keys: {sorted(missing)}")
        return payload


def _stable_summary_ref(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned[:160]
