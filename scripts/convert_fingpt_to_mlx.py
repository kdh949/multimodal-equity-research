"""FinGPT (Llama-3-8B + LoRA) → MLX 4-bit 양자화 변환 스크립트."""
from __future__ import annotations

# ruff: noqa: E402, I001

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from quant_research.runtime import configure_local_runtime_defaults

configure_local_runtime_defaults()

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_PATH = str(ROOT / "artifacts/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920")
LORA_PATH = str(ROOT / "artifacts/huggingface/hub/models--FinGPT--fingpt-mt_llama3-8b_lora/snapshots/5b5850574ec13e4ce7c102e24f763205992711b7")
MERGED_PATH = str(ROOT / "artifacts/model_cache/fingpt-merged-temp")
MLX_PATH = str(ROOT / "artifacts/model_cache/fingpt-mt-llama3-8b-mlx")


def main() -> None:
    print("[1/4] 베이스 모델 로드 중 (Llama-3-8B, float16)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_PATH,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    print("[2/4] LoRA 어댑터 병합 중 (FinGPT-MT)...")
    model = PeftModel.from_pretrained(model, LORA_PATH, local_files_only=True)
    model = model.merge_and_unload()

    print(f"[3/4] 병합된 모델 저장 중 → {MERGED_PATH}")
    Path(MERGED_PATH).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MERGED_PATH, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_PATH)
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"[4/4] MLX 4-bit 양자화 변환 중 → {MLX_PATH}")
    from mlx_lm import convert
    convert(hf_path=MERGED_PATH, mlx_path=MLX_PATH, quantize=True, q_bits=4)

    print(f"\n임시 병합 파일 정리 중 → {MERGED_PATH}")
    shutil.rmtree(MERGED_PATH, ignore_errors=True)

    mlx_files = list(Path(MLX_PATH).iterdir())
    total_mb = sum(f.stat().st_size for f in mlx_files if f.is_file()) / 1e6
    print(f"\n변환 완료: {MLX_PATH}")
    print(f"파일 수: {len(mlx_files)}, 전체 크기: {total_mb:.0f} MB")


if __name__ == "__main__":
    main()
