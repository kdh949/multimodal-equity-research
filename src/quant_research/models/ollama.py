from __future__ import annotations

from dataclasses import dataclass

import requests


@dataclass
class OllamaAgent:
    model: str = "qwen3-coder:30b"
    base_url: str = "http://localhost:11434"
    timeout_seconds: float = 30.0

    def explain(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            return str(response.json().get("response", "")).strip()
        except Exception:
            return "Ollama is unavailable; deterministic metrics and signals were generated locally."
