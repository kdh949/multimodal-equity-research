from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_research.runtime import configure_local_runtime_defaults

configure_local_runtime_defaults()

import pandas as pd

FINGPT_PROFILES = {
    "mt-llama3": {
        "adapter": "FinGPT/fingpt-mt_llama3-8b_lora",
        "base": "meta-llama/Meta-Llama-3-8B",
        "description": "Official FinGPT multi-task Llama-3 8B LoRA profile.",
    },
    "mt-llama2": {
        "adapter": "FinGPT/fingpt-mt_llama2-7b_lora",
        "base": "meta-llama/Llama-2-7b-hf",
        "description": "Official FinGPT multi-task Llama-2 7B LoRA profile.",
    },
    "forecaster": {
        "adapter": "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora",
        "base": "meta-llama/Llama-2-7b-chat-hf",
        "description": "Official FinGPT-Forecaster DOW30 Llama-2 7B chat LoRA profile.",
    },
    "sentiment-v3": {
        "adapter": "FinGPT/fingpt-sentiment_llama2-13b_lora",
        "base": "NousResearch/Llama-2-13b-hf",
        "description": "Official FinGPT v3.3 sentiment Llama-2 13B LoRA profile.",
    },
}

FINGPT_DEFAULT_RUNTIME = "llama-cpp"
FINGPT_UNQUANTIZED_BLOCKED_BASES = {"meta-llama/meta-llama-3-8b"}


def main() -> int:
    args = _parse_args()
    selected = _selected_models(args)
    if not any(selected.values()):
        print("Select at least one model with --chronos, --granite, --finma, --fingpt, or --all.")
        return 2
    if selected["fingpt"] and args.mode == "warmup" and args.fingpt_adapter_only:
        print("FinGPT warmup requires a base model; remove --fingpt-adapter-only.")
        return 2

    if args.mode == "download":
        _download_selected(args, selected)
        return 0
    if args.mode == "verify":
        return _verify_selected(args, selected)
    if _should_block_unquantized_fingpt_warmup(args, selected["fingpt"]):
        return 2
    _warmup_selected(args, selected)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download or warm up optional local Hugging Face models for the quant research app."
    )
    parser.add_argument("--mode", choices=["download", "warmup", "verify"], default="download")
    parser.add_argument("--all", action="store_true", help="Select every supported local model.")
    parser.add_argument("--chronos", action="store_true", help="Select Chronos-2.")
    parser.add_argument("--granite", action="store_true", help="Select Granite TTM.")
    parser.add_argument("--finma", action="store_true", help="Select FinMA.")
    parser.add_argument("--fingpt", action="store_true", help="Select FinGPT.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--local-files-only", action="store_true", help="Do not contact Hugging Face.")
    parser.add_argument("--device-map", default="auto", help="Transformers/Chronos device map.")
    parser.add_argument("--offload-folder", type=Path, default=Path("artifacts/model_offload"))
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--chronos-id", default="amazon/chronos-2")
    parser.add_argument("--granite-id", default="ibm-granite/granite-timeseries-ttm-r2")
    parser.add_argument("--granite-revision", default=None)
    parser.add_argument("--finma-id", default="ChanceFocus/finma-7b-nlp")
    parser.add_argument("--fingpt-profile", choices=sorted(FINGPT_PROFILES), default="mt-llama3")
    parser.add_argument("--fingpt-id", default=None, help="Override the selected FinGPT LoRA adapter.")
    parser.add_argument("--fingpt-base-id", default=None, help="Override the selected FinGPT base model.")
    parser.add_argument(
        "--fingpt-adapter-only",
        action="store_true",
        help="Download only the FinGPT LoRA adapter. Warmup still requires a base model.",
    )
    parser.add_argument(
        "--fingpt-runtime",
        choices=["transformers", "mlx", "llama-cpp", "llama_cpp"],
        default=FINGPT_DEFAULT_RUNTIME,
        help="FinGPT warmup runtime (default favors quantized path).",
    )
    parser.add_argument(
        "--fingpt-quantized-model-path",
        type=Path,
        default=None,
        help="Path to quantized GGUF/MLX model for FinGPT warmup.",
    )
    parser.add_argument(
        "--allow-unquantized-fingpt-transformers",
        action="store_true",
        help="Allow forcing FinGPT warmup through transformers with default Llama-3 8B.",
    )
    parser.add_argument(
        "--fail-on-fingpt-base-error",
        action="store_true",
        help="Fail the command if the gated FinGPT base model cannot be downloaded.",
    )
    args = parser.parse_args()
    args.fingpt_runtime = _normalize_fingpt_runtime(args.fingpt_runtime)
    return args


def _selected_models(args: argparse.Namespace) -> dict[str, bool]:
    return {
        "chronos": args.all or args.chronos,
        "granite": args.all or args.granite,
        "finma": args.all or args.finma,
        "fingpt": args.all or args.fingpt,
    }


def _download_selected(args: argparse.Namespace, selected: dict[str, bool]) -> None:
    if selected["chronos"]:
        _snapshot_download(args.chronos_id, args.cache_dir, args.local_files_only)
    if selected["granite"]:
        _snapshot_download(
            args.granite_id,
            args.cache_dir,
            args.local_files_only,
            revision=args.granite_revision,
        )
    if selected["finma"]:
        _snapshot_download(args.finma_id, args.cache_dir, args.local_files_only)
    if selected["fingpt"]:
        adapter_id, base_id, profile_description = _resolve_fingpt_model_ids(args)
        print(f"FinGPT profile: {args.fingpt_profile} ({profile_description})")
        if args.fingpt_adapter_only:
            print("Skipping FinGPT base model download because --fingpt-adapter-only was set.")
        elif base_id:
            try:
                _snapshot_download(base_id, args.cache_dir, args.local_files_only)
            except Exception as exc:
                print(f"FinGPT base download failed: {type(exc).__name__}: {exc}")
                if args.fail_on_fingpt_base_error:
                    raise
        _snapshot_download(adapter_id, args.cache_dir, args.local_files_only)


def _snapshot_download(
    model_id: str,
    cache_dir: Path | None,
    local_files_only: bool,
    revision: str | None = None,
) -> str:
    from huggingface_hub import snapshot_download

    kwargs: dict[str, object] = {"repo_id": model_id, "local_files_only": local_files_only}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if revision:
        kwargs["revision"] = revision
    path = snapshot_download(**kwargs)
    print(f"{model_id}: {path}")
    return path


def _verify_model_cache(
    model_id: str,
    cache_dir: Path | None,
    label: str,
    revision: str | None = None,
) -> bool:
    try:
        path = _snapshot_download(model_id, cache_dir, local_files_only=True, revision=revision)
        print(f"[OK] {label}: local cache found ({path})")
        return True
    except Exception as exc:
        print(f"[FAIL] {label}: local cache unavailable ({type(exc).__name__}: {exc})")
        return False


def _verify_selected(args: argparse.Namespace, selected: dict[str, bool]) -> int:
    if not selected["fingpt"]:
        print("verify mode currently checks only FinGPT in this script.")
        return 0
    if _should_block_missing_fingpt_runtime_requirements(args):
        return 2

    adapter_id, base_id, profile_description = _resolve_fingpt_model_ids(args)
    print(f"FinGPT profile: {args.fingpt_profile} ({profile_description})")
    ok = True
    ok &= _verify_model_cache(adapter_id, args.cache_dir, "FinGPT adapter")
    if not args.fingpt_adapter_only and base_id:
        ok &= _verify_model_cache(base_id, args.cache_dir, "FinGPT base model")
    return 0 if ok else 2


def _should_block_missing_fingpt_runtime_requirements(args: argparse.Namespace) -> bool:
    if args.fingpt_runtime in {"mlx", "llama-cpp"} and args.fingpt_quantized_model_path is None:
        print(
            "[FINGPT] Quantized runtime requested but --fingpt-quantized-model-path is missing.\n"
            "[금융 LLM] llama.cpp 또는 MLX 런타임은 양자화 모델 경로가 필요합니다.\n"
            "Provide --fingpt-quantized-model-path to continue."
        )
        return True

    if (
        args.fingpt_runtime in {"mlx", "llama-cpp"}
        and args.fingpt_quantized_model_path is not None
        and not args.fingpt_quantized_model_path.exists()
    ):
        print(
            f"[FINGPT] Quantized model file not found: {args.fingpt_quantized_model_path}\n"
            "[금융 LLM] 지정한 양자화 모델 파일 경로를 확인해 주세요."
        )
        return True

    return False


def _should_block_unquantized_fingpt_warmup(args: argparse.Namespace, is_selected: bool) -> bool:
    if not is_selected:
        return False

    adapter_id, base_id, _ = _resolve_fingpt_model_ids(args)
    if (
        args.fingpt_runtime == "transformers"
        and not args.allow_unquantized_fingpt_transformers
        and _is_blocked_unquantized_fingpt_base(base_id)
    ):
        print(
            "[FINGPT] Refusing unquantized default FinGPT warmup path for Llama-3 8B.\n"
            "[금융 LLM] 기본 mt-llama3 (Llama-3 8B) LoRA는 안전상 기본적으로 블록됩니다.\n"
            "기본 동작은 양자화 런타임(llama-cpp)이며, Transformers를 강제로 사용하려면 "
            "--allow-unquantized-fingpt-transformers 플래그를 추가하세요."
        )
        print(f"FinGPT profile: {args.fingpt_profile}, adapter={adapter_id}, base={base_id}")
        print("FinGPT runtime selected: transformers")
        return True

    if _should_block_missing_fingpt_runtime_requirements(args):
        return True

    return False


def _is_blocked_unquantized_fingpt_base(base_id: str | None) -> bool:
    if base_id is None:
        return False
    normalized = str(base_id).strip().lower()
    return normalized in FINGPT_UNQUANTIZED_BLOCKED_BASES


def _normalize_fingpt_runtime(runtime: str) -> str:
    return "llama-cpp" if runtime == "llama_cpp" else runtime


def _warmup_selected(args: argparse.Namespace, selected: dict[str, bool]) -> None:
    from quant_research.models.text import FilingEventExtractor, FinGPTEventExtractor
    from quant_research.models.timeseries import Chronos2Adapter, GraniteTTMAdapter

    sample = _sample_features()
    if selected["chronos"]:
        output = Chronos2Adapter(
            model_id=args.chronos_id,
            device_map=args.device_map,
            local_files_only=args.local_files_only,
        ).add_local_forecasts(sample, min_context=16, max_inference_windows=1)
        _print_columns(output, "Chronos-2", "chronos_expected_return", "chronos_quantile_width")
    if selected["granite"]:
        output = GraniteTTMAdapter(
            model_id=args.granite_id,
            revision=args.granite_revision,
        ).add_local_forecasts(sample, min_context=16, max_inference_windows=1)
        _print_columns(output, "Granite TTM", "granite_ttm_expected_return", "granite_ttm_confidence")
    if selected["finma"]:
        args.offload_folder.mkdir(parents=True, exist_ok=True)
        extractor = FilingEventExtractor(
            model_id=args.finma_id,
            use_local_model=True,
            device_map=args.device_map,
            local_files_only=args.local_files_only,
            offload_folder=str(args.offload_folder),
            max_new_tokens=args.max_new_tokens,
        )
        output = extractor.extract("Form 8-K material earnings guidance update with potential legal risk.")
        print(f"FinMA: {output}")
        print(f"FinMA source: {extractor.last_source}")
        if extractor.last_error:
            print(f"FinMA fallback error: {extractor.last_error}")
    if selected["fingpt"]:
        adapter_id, base_id, profile_description = _resolve_fingpt_model_ids(args)
        print(f"FinGPT profile: {args.fingpt_profile} ({profile_description})")
        args.offload_folder.mkdir(parents=True, exist_ok=True)
        extractor = FinGPTEventExtractor(
            model_id=adapter_id,
            base_model_id=base_id,
            use_local_model=True,
            device_map=args.device_map,
            local_files_only=args.local_files_only,
            offload_folder=str(args.offload_folder),
            max_new_tokens=args.max_new_tokens,
            runtime=args.fingpt_runtime,
            runtime_model_path=(
                str(args.fingpt_quantized_model_path)
                if args.fingpt_quantized_model_path is not None
                else None
            ),
            allow_unquantized_transformers=args.allow_unquantized_fingpt_transformers,
            allow_unquantized_fingpt=args.allow_unquantized_fingpt_transformers,
        )
        output = extractor.extract("Form 8-K material earnings guidance update with potential legal risk.")
        print(f"FinGPT: {output}")
        print(f"FinGPT source: {extractor.last_source}")
        if extractor.last_error:
            print(f"FinGPT fallback error: {extractor.last_error}")


def _resolve_fingpt_model_ids(args: argparse.Namespace) -> tuple[str, str | None, str]:
    profile = FINGPT_PROFILES[args.fingpt_profile]
    adapter_id = args.fingpt_id or str(profile["adapter"])
    base_id = args.fingpt_base_id or str(profile["base"])
    return adapter_id, base_id, str(profile["description"])


def _sample_features() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=40, freq="D")
    rows: list[dict[str, object]] = []
    for ticker, drift in [("AAPL", 0.0008), ("MSFT", 0.0004)]:
        for index, day in enumerate(dates):
            value = drift + (index % 5 - 2) * 0.0005
            rows.append(
                {
                    "date": day,
                    "ticker": ticker,
                    "return_1": value,
                    "return_5": value * 5,
                    "return_20": value * 20,
                    "volatility_20": 0.02 + abs(value),
                    "high_low_range": 0.01 + abs(value),
                }
            )
    return pd.DataFrame(rows)


def _print_columns(output: pd.DataFrame, label: str, *columns: str) -> None:
    latest = output.sort_values("date").groupby("ticker").tail(1)
    print(f"{label}:")
    print(latest[["date", "ticker", *columns]].to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main())
