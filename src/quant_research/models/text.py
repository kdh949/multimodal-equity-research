from __future__ import annotations

import contextlib
import hashlib
import json
import multiprocessing as mp
import os
import re
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from quant_research.features.text import KeywordSentimentAnalyzer

try:
    import fcntl

    _HAS_FCNTL = True
except Exception:  # pragma: no cover - platform-specific
    fcntl = None
    _HAS_FCNTL = False

_LOCK_CATALOG: dict[str, threading.Lock] = {}
_LOCK_CATALOG_GUARD = threading.Lock()


def _mlx_worker_loop(model_path: str, req_queue: Any, resp_queue: Any) -> None:
    """MLX 추론 워커 — spawn된 별도 프로세스에서 실행 (Metal 컨텍스트 격리).

    fork()를 사용하지 않는 spawn 방식으로 시작되므로 PyTorch MPS 상태가 상속되지 않아
    Metal 컨텍스트 충돌(SIGSEGV)이 발생하지 않는다.
    """
    try:
        from mlx_lm import generate, load

        model, tokenizer = load(model_path)
        resp_queue.put({"status": "ready"})
    except Exception as exc:
        resp_queue.put({"error": str(exc)})
        return

    while True:
        prompt = req_queue.get()
        if prompt is None:
            break
        try:
            response = generate(model, tokenizer, prompt=prompt, max_tokens=192, verbose=False)
            resp_queue.put({"response": response})
        except Exception as exc:
            resp_queue.put({"error": str(exc)})


@dataclass
class FinBERTSentimentAnalyzer:
    model_id: str = "ProsusAI/finbert"
    device: str | None = None
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

                kwargs: dict[str, object] = {"top_k": None}
                if self.device is not None:
                    kwargs["device"] = self.device
                self._pipeline = pipeline("text-classification", model=self.model_id, **kwargs)
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
    single_load_lock_path: str | None = None
    max_new_tokens: int = 192
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    last_source: str = field(default="rules", init=False)
    last_error: str | None = field(default=None, init=False)

    def _load_lock_path(self) -> str | None:
        if self.single_load_lock_path:
            return _explicit_model_lock_path(self.single_load_lock_path)
        lock_root = self.offload_folder
        if not lock_root:
            return _fallback_model_lock_path(_default_runtime_lock_id(self.model_id, self.__class__.__name__))
        return _build_model_lock_path(lock_root, _default_runtime_lock_id(self.model_id, self.__class__.__name__))

    def _load_local_model(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        lock_path = self._load_lock_path()
        with _acquire_model_load_lock(lock_path):
            if self._tokenizer is not None and self._model is not None:
                return self._tokenizer, self._model
            tokenizer, model = self._load_local_model_no_lock()
            self._tokenizer = tokenizer
            self._model = model
            return self._tokenizer, self._model

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

    def _load_local_model_no_lock(self) -> tuple[Any, Any]:
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
    model_id: str = "FinGPT/fingpt-mt_llama3-8b_lora"
    base_model_id: str | None = "meta-llama/Meta-Llama-3-8B"
    runtime: str = "transformers"
    runtime_model_path: str | None = None
    allow_unquantized_transformers: bool = False
    allow_unquantized_fingpt: bool = False
    ollama_model: str = "fingpt"
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: float = 60.0
    # MLX spawn 프로세스 (fork 없이 Metal 컨텍스트 격리, 내부 상태)
    _mlx_proc: Any = field(default=None, init=False, repr=False, compare=False)
    _mlx_req_queue: Any = field(default=None, init=False, repr=False, compare=False)
    _mlx_resp_queue: Any = field(default=None, init=False, repr=False, compare=False)

    def _load_lock_path(self) -> str | None:
        if self.single_load_lock_path:
            return _explicit_model_lock_path(self.single_load_lock_path)
        lock_root = self.offload_folder or (
            os.path.dirname(os.path.expanduser(self.runtime_model_path))
            if self.runtime_model_path
            else None
        )
        return _build_model_lock_path(
            lock_root,
            _default_runtime_lock_id(
                self.model_id,
                self.__class__.__name__,
                base_model_id=self.base_model_id,
                runtime=self._normalized_runtime(),
                runtime_model_path=self.runtime_model_path,
            ),
        )

    @property
    def runtime_label(self) -> str:
        return _normalize_fingpt_runtime(self.runtime)

    def _normalized_runtime(self) -> str:
        return self.runtime_label

    def _is_default_unquantized_guarded_transformers(self) -> bool:
        if self.allow_unquantized_transformers or self.allow_unquantized_fingpt:
            return False
        base = (self.base_model_id or "").lower()
        return "meta-llama/llama-3-8b" in base or ("meta-llama" in base and "3-8b" in base)

    def _load_transformer_runtime(self) -> tuple[Any, Any]:
        if self._is_default_unquantized_guarded_transformers():
            raise RuntimeError(
                "Refusing to load default unquantized FinGPT Transformers runtime. "
                "Set allow_unquantized_transformers=True or allow_unquantized_fingpt=True to enable."
            )
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required for FinGPT transformers runtime.") from exc
        tokenizer_id = self.base_model_id or self.model_id
        kwargs = _from_pretrained_kwargs(self.local_files_only, self.trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, **kwargs)
        _ensure_pad_token(tokenizer)
        model_kwargs = _model_from_pretrained_kwargs(self.device_map, self.offload_folder, kwargs)
        if self.base_model_id:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("peft is required for FinGPT transformers LoRA adapter loading.") from exc
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                **model_kwargs,
            )
            peft_kwargs = {"local_files_only": True} if self.local_files_only else {}
            model = PeftModel.from_pretrained(base, self.model_id, **peft_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs,
            )
        return tokenizer, model

    def _load_mlx_runtime(self) -> tuple[Any, Any]:
        path = self.runtime_model_path
        if not path:
            raise ValueError("runtime_model_path is required for FinGPT MLX runtime.")
        try:
            from mlx_lm import load
        except ImportError as exc:
            raise RuntimeError("mlx_lm is required for FinGPT MLX runtime.") from exc
        loaded = load(path)
        if not isinstance(loaded, tuple) or len(loaded) < 2:
            raise RuntimeError(f"Unexpected MLX load response for: {path}")
        model, tokenizer = loaded[:2]
        if tokenizer is None or model is None:
            raise RuntimeError(f"MLX runtime failed to load model from: {path}")
        return tokenizer, model

    def _load_llamacpp_runtime(self) -> tuple[Any, Any]:
        path = self.runtime_model_path
        if not path:
            raise ValueError("runtime_model_path is required for FinGPT llama.cpp runtime.")
        if not (str(path).lower().endswith(".gguf") or str(path).lower().endswith(".ggml")):
            raise ValueError(f"LLama.cpp runtime expects a GGUF/quantized model file path; got: {path}")
        try:
            from llama_cpp import Llama
        except Exception as exc:
            raise RuntimeError("llama_cpp is required for FinGPT llama.cpp runtime.") from exc
        return None, Llama(
            model_path=str(path),
            n_ctx=4096,
            verbose=False,
        )

    def _load_local_model_no_lock(self) -> tuple[Any, Any]:
        runtime = self._normalized_runtime()
        if runtime == "transformers":
            return self._load_transformer_runtime()
        if runtime == "mlx":
            return self._load_mlx_runtime()
        if runtime in {"llama_cpp", "llama-cpp", "llamacpp"}:
            return self._load_llamacpp_runtime()
        raise ValueError(f"Unsupported FinGPT runtime: {runtime}")

    def _start_mlx_worker(self) -> None:
        path = self.runtime_model_path
        if not path:
            raise ValueError("runtime_model_path is required for FinGPT MLX runtime.")
        # spawn 방식으로 프로세스를 시작해 fork()를 완전히 우회
        # (fork는 macOS Metal/ObjC 상태를 상속하여 SIGSEGV 유발)
        ctx = mp.get_context("spawn")
        self._mlx_req_queue = ctx.Queue()
        self._mlx_resp_queue = ctx.Queue()
        self._mlx_proc = ctx.Process(
            target=_mlx_worker_loop,
            args=(str(Path(path).resolve()), self._mlx_req_queue, self._mlx_resp_queue),
            daemon=True,
        )
        self._mlx_proc.start()
        ready = self._mlx_resp_queue.get(timeout=180)
        if "error" in ready:
            self._mlx_proc.terminate()
            raise RuntimeError(f"MLX worker 시작 실패: {ready['error']}")

    def _generate_with_mlx(self, text: str) -> str:
        if self._mlx_proc is None or not self._mlx_proc.is_alive():
            self._start_mlx_worker()
        prompt = _event_extraction_prompt(text)
        self._mlx_req_queue.put(prompt)
        response = self._mlx_resp_queue.get(timeout=120)
        if "error" in response:
            raise RuntimeError(f"MLX worker 추론 오류: {response['error']}")
        return response.get("response", "")

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            if self._mlx_req_queue is not None:
                self._mlx_req_queue.put(None)
        with contextlib.suppress(Exception):
            if self._mlx_proc is not None and self._mlx_proc.is_alive():
                self._mlx_proc.join(timeout=5)
                if self._mlx_proc.is_alive():
                    self._mlx_proc.terminate()

    def _generate_with_llamacpp(self, text: str) -> str:
        _, model = self._load_local_model()
        prompt = _event_extraction_prompt(text)
        generated = model(prompt, max_tokens=self.max_new_tokens, stop=["</s>"], temperature=0.0)
        return _coerce_generation_text(generated)

    def _generate_with_ollama(self, text: str) -> str:
        prompt = _event_extraction_prompt(text)
        response = requests.post(
            f"{self.ollama_base_url}/api/generate",
            json={"model": self.ollama_model, "prompt": prompt, "stream": False},
            timeout=self.ollama_timeout,
        )
        response.raise_for_status()
        return str(response.json().get("response", "")).strip()

    def _generate_with_local_model(self, text: str) -> str:
        runtime = self._normalized_runtime()
        if runtime == "transformers":
            return super()._generate_with_local_model(text)
        if runtime == "mlx":
            return self._generate_with_mlx(text)
        if runtime in {"llama_cpp", "llama-cpp", "llamacpp"}:
            return self._generate_with_llamacpp(text)
        if runtime == "ollama":
            return self._generate_with_ollama(text)
        raise ValueError(f"Unsupported FinGPT runtime: {runtime}")


@contextlib.contextmanager
def _acquire_model_load_lock(lock_path: str | None):
    if not lock_path:
        yield
        return

    try:
        lock_file = Path(lock_path)
        lock_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        yield
        return

    with _LOCK_CATALOG_GUARD:
        lock = _LOCK_CATALOG.setdefault(lock_path, threading.Lock())
    if not lock.acquire(blocking=False):
        raise RuntimeError(
            "Another local filing model load is in progress in this process. Falling back to rules until it completes."
        )
    opened = None
    try:
        try:
            opened = lock_file.open("a+")
        except OSError:
            yield
            return
        if _HAS_FCNTL:
            try:
                fcntl.flock(opened.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(
                    "Another process is loading the local filing LLM; skipping to avoid duplicate loading."
                ) from exc
        yield
    finally:
        if _HAS_FCNTL and opened is not None:
            try:
                fcntl.flock(opened.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        if opened is not None:
            try:
                opened.close()
            except OSError:
                pass
        lock.release()


def _build_model_lock_path(lock_root: str | None, lock_id: str) -> str | None:
    if not lock_root:
        return None
    try:
        root = Path(lock_root).expanduser()
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return str(root / f"{lock_id}.fingpt-load.lock")


def _explicit_model_lock_path(lock_path: str) -> str | None:
    try:
        path = Path(lock_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return str(path)


def _fallback_model_lock_path(lock_id: str) -> str | None:
    return _build_model_lock_path(tempfile.gettempdir(), lock_id)


def _default_runtime_lock_id(
    *parts: str | None,
    base_model_id: str | None = None,
    runtime: str | None = None,
    runtime_model_path: str | None = None,
) -> str:
    values = [str(part or "") for part in (*parts, base_model_id or "", runtime or "", runtime_model_path or "")]
    return hashlib.md5(
        "|".join(values).encode(),
    ).hexdigest()[:16]



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


def _normalize_fingpt_runtime(raw: str | None) -> str:
    normalized = (raw or "").strip().replace(" ", "").lower()
    normalized = normalized.replace("_", "-")
    if normalized in {"transformer", "transformers", "hf", "huggingface", "transformersruntime"}:
        return "transformers"
    if normalized in {"llamacpp", "llama-cpp", "llama_cpp", "llama.cpp", "gguf", "ggml"}:
        return "llama_cpp"
    if normalized in {"mlx", "mlxlm", "mlx-lm"}:
        return "mlx"
    if normalized == "ollama":
        return "ollama"
    return normalized


def _coerce_generation_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list) and payload:
        for item in payload:
            converted = _coerce_generation_text(item)
            if converted:
                return converted
        return ""
    if isinstance(payload, tuple) and payload:
        return _coerce_generation_text(payload[0])
    if isinstance(payload, dict):
        for key in ("text", "response", "output", "result", "generated_text"):
            if isinstance(payload.get(key), str):
                return str(payload[key])
        if isinstance(payload.get("choices"), list):
            first = payload["choices"][0]
            if isinstance(first, dict):
                for key in ("text", "content", "response"):
                    if isinstance(first.get(key), str):
                        return str(first[key])
                message = first.get("message")
                if isinstance(message, dict):
                    for key in ("content", "text"):
                        if isinstance(message.get(key), str):
                            return str(message[key])
    return str(payload)


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
