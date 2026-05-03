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
    model_id: str = "ChanceFocus/finma-7b-nlp"
    use_local_model: bool = False
    device_map: str = "auto"
    local_files_only: bool = False
    trust_remote_code: bool = False
    offload_folder: str | None = None
    max_new_tokens: int = 192
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    last_source: str = field(default="rules", init=False)
    last_error: str | None = field(default=None, init=False)

    def extract(self, text: str) -> dict[str, object]:
        if self.use_local_model:
            try:
                payload = self.validate_json(self._generate_with_local_model(text))
                self.last_source = "local"
                self.last_error = None
                return payload
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
        self.last_source = "rules"
        return self.extract_with_rules(text)

    def extract_with_rules(self, text: str) -> dict[str, object]:
        lower = text.lower()
        event_tags = []
        for keyword, tag in {
            "8-k": "current_report",
            "10-q": "quarterly_report",
            "10-k": "annual_report",
            "form 4": "insider_activity",
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
        payload = json.loads(_extract_json_object(raw))
        required = {"event_tag", "risk_flag", "confidence", "summary_ref"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"Missing filing extraction keys: {sorted(missing)}")
        if not isinstance(payload["event_tag"], str):
            raise ValueError("event_tag must be a string")
        if not isinstance(payload["risk_flag"], bool):
            raise ValueError("risk_flag must be a boolean")
        if isinstance(payload["confidence"], bool) or not isinstance(payload["confidence"], int | float):
            raise ValueError("confidence must be numeric")
        if not 0 <= float(payload["confidence"]) <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if not isinstance(payload["summary_ref"], str):
            raise ValueError("summary_ref must be a string")
        payload["confidence"] = float(payload["confidence"])
        return payload

    def _generate_with_local_model(self, text: str) -> str:
        tokenizer, model = self._load_local_model()
        prompt = _event_extraction_prompt(text)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        device = _model_device(model)
        if device is not None:
            inputs = {key: value.to(device) for key, value in inputs.items()}
        output_ids = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(generated, skip_special_tokens=True)

    def _load_local_model(self) -> tuple[Any, Any]:
        if self._tokenizer is None or self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            kwargs = _from_pretrained_kwargs(self.local_files_only, self.trust_remote_code)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
            _ensure_pad_token(self._tokenizer)
            model_kwargs = _model_from_pretrained_kwargs(self.device_map, self.offload_folder, kwargs)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs,
            )
        return self._tokenizer, self._model


@dataclass
class FinGPTEventExtractor(FilingEventExtractor):
    model_id: str = "FinGPT/fingpt-mt_llama2-7b_lora"
    base_model_id: str | None = "meta-llama/Llama-2-7b-hf"

    def _load_local_model(self) -> tuple[Any, Any]:
        if self._tokenizer is None or self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer_id = self.base_model_id or self.model_id
            kwargs = _from_pretrained_kwargs(self.local_files_only, self.trust_remote_code)
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, **kwargs)
            _ensure_pad_token(self._tokenizer)
            model_kwargs = _model_from_pretrained_kwargs(self.device_map, self.offload_folder, kwargs)
            if self.base_model_id:
                from peft import PeftModel

                base = AutoModelForCausalLM.from_pretrained(
                    self.base_model_id,
                    **model_kwargs,
                )
                peft_kwargs = {"local_files_only": True} if self.local_files_only else {}
                self._model = PeftModel.from_pretrained(base, self.model_id, **peft_kwargs)
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    **model_kwargs,
                )
        return self._tokenizer, self._model


def _stable_summary_ref(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned[:160]


def _event_extraction_prompt(text: str) -> str:
    return (
        "You are a financial filing event extraction model. "
        "Return only valid JSON with keys event_tag, risk_flag, confidence, summary_ref. "
        "event_tag must be a comma-separated string or 'none'; risk_flag must be boolean; "
        "confidence must be a number between 0 and 1; summary_ref must be a short source-grounded phrase.\n\n"
        f"Document:\n{text[:4000]}\n\nJSON:"
    )


def _extract_json_object(raw: str) -> str:
    stripped = raw.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        return match.group(0)
    raise ValueError("No JSON object found in model output")


def _from_pretrained_kwargs(local_files_only: bool, trust_remote_code: bool) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if local_files_only:
        kwargs["local_files_only"] = True
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    return kwargs


def _model_from_pretrained_kwargs(
    device_map: str,
    offload_folder: str | None,
    base_kwargs: dict[str, object],
) -> dict[str, object]:
    kwargs = {"device_map": device_map, **base_kwargs}
    if offload_folder:
        kwargs["offload_folder"] = offload_folder
    return kwargs


def _ensure_pad_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token


def _model_device(model: Any) -> Any | None:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return getattr(model, "device", None)
