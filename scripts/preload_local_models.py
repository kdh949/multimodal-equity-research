from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    args = _parse_args()
    selected = _selected_models(args)
    if not any(selected.values()):
        print("Select at least one model with --chronos, --granite, --finma, --fingpt, or --all.")
        return 2
    if selected["fingpt"] and not args.fingpt_base_id:
        print("FinGPT LoRA inference requires --fingpt-base-id, for example meta-llama/Meta-Llama-3-8B.")
        return 2

    if args.mode == "download":
        _download_selected(args, selected)
    else:
        _warmup_selected(args, selected)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download or warm up optional local Hugging Face models for the quant research app."
    )
    parser.add_argument("--mode", choices=["download", "warmup"], default="download")
    parser.add_argument("--all", action="store_true", help="Select every supported local model.")
    parser.add_argument("--chronos", action="store_true", help="Select Chronos-2.")
    parser.add_argument("--granite", action="store_true", help="Select Granite TTM.")
    parser.add_argument("--finma", action="store_true", help="Select FinMA.")
    parser.add_argument("--fingpt", action="store_true", help="Select FinGPT.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--local-files-only", action="store_true", help="Do not contact Hugging Face.")
    parser.add_argument("--device-map", default="auto", help="Transformers/Chronos device map.")
    parser.add_argument("--chronos-id", default="amazon/chronos-2")
    parser.add_argument("--granite-id", default="ibm-granite/granite-timeseries-ttm-r2")
    parser.add_argument("--granite-revision", default=None)
    parser.add_argument("--finma-id", default="ChanceFocus/finma-7b-nlp")
    parser.add_argument("--fingpt-id", default="FinGPT/fingpt-mt_llama3-8b_lora")
    parser.add_argument("--fingpt-base-id", default="meta-llama/Meta-Llama-3-8B")
    return parser.parse_args()


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
        _snapshot_download(args.fingpt_base_id, args.cache_dir, args.local_files_only)
        _snapshot_download(args.fingpt_id, args.cache_dir, args.local_files_only)


def _snapshot_download(
    model_id: str,
    cache_dir: Path | None,
    local_files_only: bool,
    revision: str | None = None,
) -> None:
    from huggingface_hub import snapshot_download

    kwargs: dict[str, object] = {"repo_id": model_id, "local_files_only": local_files_only}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if revision:
        kwargs["revision"] = revision
    path = snapshot_download(**kwargs)
    print(f"{model_id}: {path}")


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
        output = FilingEventExtractor(
            model_id=args.finma_id,
            use_local_model=True,
            device_map=args.device_map,
            local_files_only=args.local_files_only,
        ).extract("Form 8-K material earnings guidance update with potential legal risk.")
        print(f"FinMA: {output}")
    if selected["fingpt"]:
        output = FinGPTEventExtractor(
            model_id=args.fingpt_id,
            base_model_id=args.fingpt_base_id,
            use_local_model=True,
            device_map=args.device_map,
            local_files_only=args.local_files_only,
        ).extract("Form 8-K material earnings guidance update with potential legal risk.")
        print(f"FinGPT: {output}")


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
