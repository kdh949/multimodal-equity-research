from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import dataclasses
import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from quant_research.runtime import configure_local_runtime_defaults  # noqa: E402

configure_local_runtime_defaults()

import pandas as pd

from quant_research.config import DEFAULT_TICKERS  # noqa: E402
from quant_research.pipeline import PipelineConfig, PipelineResult, run_research_pipeline  # noqa: E402

TICKERS = list(DEFAULT_TICKERS)
DATE_RANGE_YEARS = 2
REPORT_ROOT = ROOT / "reports"
SEP_WIDE = "=" * 60
SEP_THIN = "-" * 60


MLX_MODEL_PATH = "artifacts/model_cache/fingpt-mt-llama3-8b-mlx"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="백테스트 검증 스크립트 — yfinance 실제 데이터로 모델 예측력을 검증합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
실행 예시:
  uv run python scripts/run_backtest_validation.py                        # 경량 모드 (기본)
  uv run python scripts/run_backtest_validation.py --mode full            # 실전 모드 MLX (기본)
  uv run python scripts/run_backtest_validation.py --mode full --runtime ollama  # 실전 모드 Ollama
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["lightweight", "full"],
        default="lightweight",
        help="lightweight: 빠른 검증 (기본). full: FinBERT + FinGPT 포함 실전 스택.",
    )
    parser.add_argument(
        "--runtime",
        choices=["mlx", "ollama"],
        default="mlx",
        help="full 모드 FinGPT 런타임. mlx: Apple Silicon MLX (기본). ollama: Ollama 서버.",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=DATE_RANGE_YEARS,
        metavar="N",
        help=f"분석할 과거 데이터 기간(년). 기본값: {DATE_RANGE_YEARS}",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        metavar="TICKER",
        help="분석할 종목 코드 목록. 기본값: DEFAULT_TICKERS (10개)",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    end = date.today()
    start = end - timedelta(days=365 * args.years)
    tickers = args.tickers if args.tickers else TICKERS

    if args.mode == "full":
        runtime = args.runtime
        extra: dict = {}
        if runtime == "mlx":
            # MLX는 Metal GPU를 점유하므로 FinBERT(PyTorch MPS)와 충돌 방지를 위해 CPU 사용
            extra["fingpt_quantized_model_path"] = MLX_MODEL_PATH
            extra["local_model_device_map"] = "cpu"
        return PipelineConfig(
            tickers=tickers,
            data_mode="live",
            start=start,
            end=end,
            sentiment_model="finbert",
            filing_extractor_model="fingpt",
            enable_local_filing_llm=True,
            fingpt_runtime=runtime,
            time_series_inference_mode="proxy",
            **extra,
        )

    return PipelineConfig(
        tickers=tickers,
        data_mode="live",
        start=start,
        end=end,
        sentiment_model="keyword",
        filing_extractor_model="rules",
        time_series_inference_mode="proxy",
    )


def prepare_output_dir(run_date: date) -> Path:
    out_dir = REPORT_ROOT / f"backtest_validation_{run_date.strftime('%Y%m%d')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def print_header(config: PipelineConfig, result: PipelineResult) -> None:
    tickers_str = ", ".join(config.tickers)
    market_rows = len(result.market_data)
    print(SEP_WIDE)
    print("  백테스트 검증 리포트 (Backtest Validation Report)")
    print(SEP_WIDE)
    print(f"  실행 날짜   : {date.today()}")
    print("  데이터 소스 : yfinance (실제 인터넷 데이터)")
    print(f"  분석 기간   : {config.start} → {config.end}")
    print(f"  종목        : {tickers_str}")
    print(f"  시장 데이터 : {market_rows:,}행")
    print(f"  예측 모델   : {config.model_name}  |  훈련: {config.train_periods}일  |  테스트: {config.test_periods}일")
    print(f"  감성 모델   : {config.sentiment_model}  |  공시: {config.filing_extractor_model}  |  시계열: {config.time_series_inference_mode}")
    print("  (SPY/QQQ는 SEC 데이터 없음 - CIK 미등록 ETF)")
    print()


def print_walk_forward_summary(result: PipelineResult) -> None:
    print(SEP_THIN)
    print("  [1] Walk-Forward 검증 결과 (예측 정확도)")
    print(SEP_THIN)

    summary = result.validation_summary
    if summary.empty:
        print("  [경고] Walk-Forward fold가 생성되지 않았습니다. 데이터가 부족합니다.")
        print()
        return

    print(f"  {'Fold':>4}  {'훈련 시작':>12}  {'훈련 종료':>12}  {'테스트 시작':>12}  {'테스트 종료':>12}  {'MAE':>8}  {'방향성':>8}  {'OOS?':>5}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*5}")
    for _, row in summary.iterrows():
        oos_marker = "YES" if row["is_oos"] else "-"
        train_start = pd.Timestamp(row["train_start"]).strftime("%Y-%m-%d")
        train_end = pd.Timestamp(row["train_end"]).strftime("%Y-%m-%d")
        test_start = pd.Timestamp(row["test_start"]).strftime("%Y-%m-%d")
        test_end = pd.Timestamp(row["test_end"]).strftime("%Y-%m-%d")
        print(
            f"  {int(row['fold']):>4}  {train_start:>12}  {train_end:>12}  "
            f"{test_start:>12}  {test_end:>12}  "
            f"{row['mae']:>8.5f}  {row['directional_accuracy']:>7.1%}  {oos_marker:>5}"
        )

    overall_mae = summary["mae"].mean()
    overall_dir = summary["directional_accuracy"].mean()
    n_folds = len(summary)

    if overall_dir >= 0.55:
        verdict = "STRONG PASS - 모델에 유의미한 예측력이 있습니다"
    elif overall_dir >= 0.50:
        verdict = "PASS - 50% 이상의 방향성 정확도 (실제 예측력 존재)"
    else:
        verdict = "FAIL - 방향성 정확도가 50% 미만 (랜덤 추정과 유사)"

    print()
    print(f"  전체 평균 MAE              : {overall_mae:.5f}")
    print(f"  전체 평균 방향성 정확도    : {overall_dir:.1%}")
    print(f"  완료된 Fold 수             : {n_folds}")
    print()
    print(f"  판정: {verdict}")
    print()


def print_per_ticker_accuracy(result: PipelineResult) -> None:
    print(SEP_THIN)
    print("  [2] 종목별 예측 정확도")
    print(SEP_THIN)

    preds = result.predictions
    if preds.empty:
        print("  [경고] 예측 데이터가 없습니다.")
        print()
        return

    preds = preds.copy()
    preds["correct_direction"] = (preds["expected_return"] * preds["forward_return_1"]) > 0
    preds["abs_error"] = (preds["expected_return"] - preds["forward_return_1"]).abs()

    ticker_stats = (
        preds.groupby("ticker")
        .agg(
            n_predictions=("expected_return", "count"),
            mae=("abs_error", "mean"),
            directional_accuracy=("correct_direction", "mean"),
            avg_predicted=("expected_return", "mean"),
            avg_actual=("forward_return_1", "mean"),
        )
        .reset_index()
        .sort_values("directional_accuracy", ascending=False)
    )

    print(f"  {'종목':>6}  {'예측수':>6}  {'MAE':>8}  {'방향성':>8}  {'평균 예측':>10}  {'평균 실제':>10}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}")
    for _, row in ticker_stats.iterrows():
        print(
            f"  {row['ticker']:>6}  {int(row['n_predictions']):>6}  "
            f"{row['mae']:>8.5f}  {row['directional_accuracy']:>7.1%}  "
            f"{row['avg_predicted']:>+9.4%}  {row['avg_actual']:>+9.4%}"
        )
    print()


def print_portfolio_metrics(result: PipelineResult) -> None:
    print(SEP_THIN)
    print("  [3] 포트폴리오 백테스트 지표")
    print(SEP_THIN)

    m = result.backtest.metrics
    if result.backtest.equity_curve.empty:
        print("  [경고] 백테스트 결과가 없습니다.")
        print()
        return

    excess_sign = "+" if m.excess_return >= 0 else ""
    print(f"  CAGR (연환산 수익률)       : {m.cagr:>+.2%}")
    print(f"  벤치마크 CAGR (SPY)        : {m.benchmark_cagr:>+.2%}")
    print(f"  초과 수익률                : {excess_sign}{m.excess_return:.2%}")
    print(f"  연환산 변동성              : {m.annualized_volatility:.2%}")
    print(f"  샤프 비율                  : {m.sharpe:.2f}")
    print(f"  최대 낙폭 (Max Drawdown)   : {m.max_drawdown:.2%}")
    print(f"  적중률 (Hit Rate)          : {m.hit_rate:.1%}")
    print(f"  평균 일간 회전율           : {m.turnover:.1%}")
    print(f"  평균 포트폴리오 노출도     : {m.exposure:.1%}")
    print()
    print(f"  거래 비용: {5:.0f}bps + {2:.0f}bps 슬리피지  |  최대 종목수: top-{3}  |  최대 종목 비중: 35%")
    print()


def print_sample_predictions(result: PipelineResult, n: int = 20) -> None:
    print(SEP_THIN)
    print(f"  [4] 최근 {n}개 OOS(Out-of-Sample) 예측 vs 실제")
    print(SEP_THIN)

    preds = result.predictions
    if preds.empty:
        print("  [경고] 예측 데이터가 없습니다.")
        print()
        return

    oos = preds[preds["is_oos"]].tail(n) if "is_oos" in preds.columns else preds.tail(n)
    if oos.empty:
        print("  OOS 예측 데이터가 없습니다. 전체 예측 데이터를 사용합니다.")
        oos = preds.tail(n)

    print(f"  {'날짜':>12}  {'종목':>6}  {'예측 수익률':>12}  {'실제 수익률':>12}  {'오차':>9}  {'방향 맞춤?':>10}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*9}  {'-'*10}")
    for _, row in oos.iterrows():
        predicted = float(row["expected_return"])
        actual = float(row["forward_return_1"]) if pd.notna(row.get("forward_return_1")) else float("nan")
        error = abs(predicted - actual) if pd.notna(actual) else float("nan")
        correct = "O" if (predicted * actual > 0) else "X"
        date_str = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
        actual_str = f"{actual:>+.4%}" if pd.notna(actual) else "N/A"
        error_str = f"{error:.4%}" if pd.notna(actual) else "N/A"
        print(
            f"  {date_str:>12}  {str(row['ticker']):>6}  "
            f"{predicted:>+11.4%}  {actual_str:>12}  {error_str:>9}  {correct:>10}"
        )
    print()


def save_outputs(result: PipelineResult, out_dir: Path) -> None:
    result.predictions.to_csv(out_dir / "predictions.csv", index=False)
    result.validation_summary.to_csv(out_dir / "validation_summary.csv", index=False)

    metrics_dict = dataclasses.asdict(result.backtest.metrics)
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)


def print_file_summary(out_dir: Path) -> None:
    print(SEP_THIN)
    print("  저장된 파일")
    print(SEP_THIN)
    for p in sorted(out_dir.iterdir()):
        size_kb = p.stat().st_size / 1024
        print(f"  {p}  ({size_kb:.1f} KB)")
    print()


def main() -> int:
    args = _parse_args()
    config = build_config(args)

    print()
    if args.mode == "full":
        runtime_label = "MLX (Apple Silicon)" if args.runtime == "mlx" else "Ollama"
        print(f"[ 실전 모드 ] FinBERT(감성 분석) + FinGPT(공시 이벤트 추출) 활성화  |  런타임: {runtime_label}")
    else:
        print("[ 경량 모드 ] 키워드 감성 + 규칙 기반 공시 (대형 모델 다운로드 없음)")
    print(f"             {len(config.tickers)}개 종목, {config.start} ~ {config.end} ({args.years}년)")
    print()

    result = run_research_pipeline(config)
    run_date = date.today()
    out_dir = prepare_output_dir(run_date)

    print_header(config, result)
    print_walk_forward_summary(result)
    print_per_ticker_accuracy(result)
    print_portfolio_metrics(result)
    print_sample_predictions(result, n=20)
    save_outputs(result, out_dir)
    print_file_summary(out_dir)

    print(SEP_WIDE)
    print("  검증 완료.")
    print(SEP_WIDE)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
