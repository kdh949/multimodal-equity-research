"""MLX FinGPT 추론 워커 — 메인 프로세스(PyTorch)와 Metal 컨텍스트 격리용 서브프로세스."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"error": "모델 경로 인자 필요: fingpt_mlx_worker.py <model_path>"}), flush=True)
        sys.exit(1)

    model_path = sys.argv[1]
    try:
        from mlx_lm import generate, load
    except ImportError as exc:
        print(json.dumps({"error": f"mlx_lm import 실패: {exc}"}), flush=True)
        sys.exit(1)

    try:
        model, tokenizer = load(model_path)
        # 준비 완료 신호
        print(json.dumps({"status": "ready"}), flush=True)
    except Exception as exc:
        print(json.dumps({"error": f"모델 로드 실패: {exc}"}), flush=True)
        sys.exit(1)

    for line in sys.stdin:
        prompt = line.rstrip("\n")
        if not prompt:
            continue
        try:
            response = generate(model, tokenizer, prompt=prompt, max_tokens=192, verbose=False)
            print(json.dumps({"response": response}), flush=True)
        except Exception as exc:
            print(json.dumps({"error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
