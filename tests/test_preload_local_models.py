from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

preload = importlib.import_module("scripts.preload_local_models")


def test_fingpt_runtime_flag_accepts_llama_cpp_alias(monkeypatch: object) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "--fingpt", "--fingpt-runtime", "llama_cpp"])
    args = preload._parse_args()
    assert args.fingpt_runtime == "llama-cpp"


def test_fingpt_transformers_warmup_is_blocked_without_explicit_allow(monkeypatch: object, capsys: object) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "--mode", "warmup", "--fingpt", "--fingpt-runtime", "transformers"])
    monkeypatch.setattr(preload, "_warmup_selected", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("warmup should not run")))

    return_code = preload.main()
    output = capsys.readouterr().out

    assert return_code == 2
    assert "Refusing unquantized default FinGPT warmup path for Llama-3 8B" in output


def test_fingpt_transformers_warmup_runs_with_allow_flag(monkeypatch: object) -> None:
    monkeypatch.setattr(sys, "argv", [
        "prog",
        "--mode",
        "warmup",
        "--fingpt",
        "--fingpt-runtime",
        "transformers",
        "--allow-unquantized-fingpt-transformers",
    ])
    monkeypatch.setattr(preload, "_warmup_selected", lambda *_args, **_kwargs: None)

    return_code = preload.main()

    assert return_code == 0


def test_verify_mode_checks_fingpt_cache_without_model_load(monkeypatch: object, tmp_path: Path, capsys: object) -> None:
    quantized_model = tmp_path / "fingpt-cpu-int4.gguf"
    quantized_model.write_text("quantized-model-placeholder")
    calls: list[tuple[str, Path | None, bool]] = []

    def fake_snapshot_download(model_id: str, cache_dir: Path | None, local_files_only: bool, revision: str | None = None) -> str:
        assert local_files_only is True
        calls.append((model_id, cache_dir, local_files_only))
        return str(tmp_path / "snapshot")

    monkeypatch.setattr(preload, "_snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(sys, "argv", [
        "prog",
        "--mode",
        "verify",
        "--fingpt",
        "--fingpt-runtime",
        "llama-cpp",
        "--fingpt-quantized-model-path",
        str(quantized_model),
    ])

    return_code = preload.main()
    output = capsys.readouterr().out

    assert return_code == 0
    assert "[OK] FinGPT adapter:" in output
    expected_model_ids = {
        "FinGPT/fingpt-mt_llama3-8b_lora",
        "meta-llama/Meta-Llama-3-8B",
    }
    assert expected_model_ids <= {call[0] for call in calls}
    assert any("[OK]" in line for line in output.splitlines())


def test_verify_mode_requires_quantized_model_path_for_quantized_runtime(monkeypatch: object, capsys: object) -> None:
    monkeypatch.setattr(preload, "_snapshot_download", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no download should happen")))
    monkeypatch.setattr(sys, "argv", ["prog", "--mode", "verify", "--fingpt", "--fingpt-runtime", "llama-cpp"])

    return_code = preload.main()
    output = capsys.readouterr().out

    assert return_code == 2
    assert "Quantized runtime requested but --fingpt-quantized-model-path is missing" in output


def test_fingpt_quantized_warmup_passes_runtime_settings(monkeypatch: object, tmp_path: Path) -> None:
    quantized_model = tmp_path / "fingpt-cpu-int4.gguf"
    quantized_model.write_text("quantized-model-placeholder")
    captured: dict[str, object] = {}

    class FakeFinGPTExtractor:
        last_source = "local"
        last_error = None

        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def extract(self, text: str) -> dict[str, object]:
            return {
                "event_tag": "guidance",
                "risk_flag": True,
                "confidence": 0.8,
                "summary_ref": text,
            }

    import quant_research.models.text as text_models

    monkeypatch.setattr(text_models, "FinGPTEventExtractor", FakeFinGPTExtractor)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--mode",
            "warmup",
            "--fingpt",
            "--fingpt-runtime",
            "llama-cpp",
            "--fingpt-quantized-model-path",
            str(quantized_model),
            "--offload-folder",
            str(tmp_path / "offload"),
        ],
    )

    return_code = preload.main()

    assert return_code == 0
    assert captured["runtime"] == "llama-cpp"
    assert captured["runtime_model_path"] == str(quantized_model)
    assert captured["allow_unquantized_transformers"] is False
