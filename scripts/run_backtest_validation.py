from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import dataclasses
import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from quant_research.runtime import configure_local_runtime_defaults  # noqa: E402

configure_local_runtime_defaults()

import pandas as pd

from quant_research.config import DEFAULT_TICKERS  # noqa: E402
from quant_research.pipeline import PipelineConfig, PipelineResult, run_research_pipeline  # noqa: E402
from quant_research.validation import (  # noqa: E402
    ReportDataSource,
    UniverseSnapshot,
    build_artifact_manifest_from_paths,
    build_canonical_report_metadata,
    build_completed_validation_backtest_report,
    build_validity_gate_report,
    write_artifact_manifest_json,
    write_completed_validation_backtest_report_artifacts,
    write_validity_gate_artifacts,
)

TICKERS = list(DEFAULT_TICKERS)
DATE_RANGE_YEARS = 3
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
    parser.add_argument(
        "--skip-manifest-regeneration",
        action="store_true",
        help="리서치/백테스트 산출물 저장 후 artifact_manifest.json 재생성을 건너뜁니다.",
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
    print(f"  성과 산출 기준             : {m.return_basis} (거래비용/슬리피지 차감 후 순수익률)")
    print(f"  순 CAGR (연환산 수익률)    : {m.net_cagr:>+.2%}")
    print(f"  총 CAGR (비용 차감 전)     : {m.gross_cagr:>+.2%}")
    print(f"  벤치마크 순 CAGR (SPY)     : {m.benchmark_cost_adjusted_cagr:>+.2%}")
    print(f"  순 초과 수익률             : {excess_sign}{m.excess_return:.2%}")
    print(f"  순 누적 수익률             : {m.net_cumulative_return:>+.2%}")
    print(f"  총 누적 수익률             : {m.gross_cumulative_return:>+.2%}")
    print(f"  누적 비용 차감             : {m.total_cost_return:.2%}")
    print(f"  순 연환산 변동성           : {m.annualized_volatility:.2%}")
    print(f"  순 샤프 비율               : {m.sharpe:.2f}")
    print(f"  순 최대 낙폭 (Max DD)      : {m.max_drawdown:.2%}")
    print(f"  적중률 (Hit Rate)          : {m.hit_rate:.1%}")
    print(f"  평균 일간 회전율           : {m.turnover:.1%}")
    print(f"  평균 포트폴리오 노출도     : {m.exposure:.1%}")
    print(f"  평균 covariance 변동성     : {m.average_portfolio_volatility_estimate:.2%}")
    print(f"  최대 covariance 변동성     : {m.max_portfolio_volatility_estimate:.2%}")
    print(f"  최대 종목 비중             : {m.max_position_weight:.1%}")
    print(f"  최대 섹터 노출             : {m.max_sector_exposure:.1%}")
    print(f"  최대 리스크 기여도         : {m.max_position_risk_contribution:.1%}")
    print(
        "  비용 조정 sizing 검증      : "
        f"{m.position_sizing_validation_status.upper()} "
        f"({m.position_sizing_validation_pass_rate:.1%}, {m.position_sizing_validation_rule})"
    )
    print()
    print(
        f"  거래 비용: {m.transaction_cost_return:.2%} transaction + "
        f"{m.slippage_cost_return:.2%} slippage  |  "
        "포지션 sizing은 비용 차감 후 long-only/risk 제약으로 검증"
    )
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


def save_outputs(
    result: PipelineResult,
    out_dir: Path,
    config: PipelineConfig,
    *,
    regenerate_manifest: bool = True,
) -> dict[str, str]:
    metadata = _canonical_report_metadata(config)

    result.market_data.to_csv(out_dir / "market_data.csv", index=False)
    result.features.to_csv(out_dir / "features.csv", index=False)
    result.predictions.to_csv(out_dir / "predictions.csv", index=False)
    result.backtest.signals.to_csv(out_dir / "signals.csv", index=False)
    result.backtest.equity_curve.to_csv(out_dir / "equity_curve.csv", index=False)
    result.validation_summary.to_csv(out_dir / "validation_summary.csv", index=False)

    _write_json_artifact(dataclasses.asdict(config), out_dir / "pipeline_config.json")
    _write_json_artifact(metadata.to_dict(), out_dir / "canonical_metadata.json")
    _write_json_artifact(
        metadata.universe.universe_snapshot.to_dict(),
        out_dir / "universe_snapshot.json",
    )
    _write_json_artifact(
        dict(metadata.data_provenance.feature_availability_cutoff),
        out_dir / "feature_availability_cutoff.json",
    )

    metrics_dict = dataclasses.asdict(result.backtest.metrics)
    _write_json_artifact(metrics_dict, out_dir / "metrics.json")

    risk_sizing_validation = _risk_sizing_validation_summary(result, config)
    _write_json_artifact(risk_sizing_validation, out_dir / "risk_sizing_validation.json")

    validity_report = build_validity_gate_report(
        result.predictions,
        result.validation_summary,
        result.backtest.equity_curve,
        result.backtest.metrics,
        ablation_summary=result.ablation_summary,
        config=config,
        benchmark_return_series=result.benchmark_return_series,
        equal_weight_baseline_return_series=result.equal_weight_baseline_return_series,
        baseline_comparison_inputs=result.baseline_comparison_inputs or None,
    )
    validity_gate_json_path, validity_gate_markdown_path = write_validity_gate_artifacts(
        validity_report,
        out_dir,
    )
    completed_report = build_completed_validation_backtest_report(
        metadata=metadata,
        deterministic_signal_outputs=result.backtest.signals,
        backtest_results=result.backtest.equity_curve,
        performance_metrics=result.backtest.metrics,
        walk_forward_validation_metrics=result.validation_summary,
        system_validity_gate=validity_report,
        strategy_candidate_gate=validity_report,
        artifact_manifest={
            "artifacts": [
                {"artifact_id": "predictions", "artifact_type": "csv", "path": "predictions.csv"},
                {
                    "artifact_id": "validation_summary",
                    "artifact_type": "csv",
                    "path": "validation_summary.csv",
                },
                {"artifact_id": "metrics", "artifact_type": "json", "path": "metrics.json"},
                {
                    "artifact_id": "validity_gate",
                    "artifact_type": "json",
                    "path": "validity_gate.json",
                },
            ]
        },
        report_path=str(out_dir / "canonical_run_report.md"),
    )
    report_artifacts = write_completed_validation_backtest_report_artifacts(
        completed_report,
        out_dir,
    )
    output_manifest = {
        "market_data": str(out_dir / "market_data.csv"),
        "features": str(out_dir / "features.csv"),
        "predictions": str(out_dir / "predictions.csv"),
        "signals": str(out_dir / "signals.csv"),
        "equity_curve": str(out_dir / "equity_curve.csv"),
        "validation_summary": str(out_dir / "validation_summary.csv"),
        "pipeline_config": str(out_dir / "pipeline_config.json"),
        "canonical_metadata": str(out_dir / "canonical_metadata.json"),
        "universe_snapshot": str(out_dir / "universe_snapshot.json"),
        "feature_availability_cutoff": str(out_dir / "feature_availability_cutoff.json"),
        "metrics": str(out_dir / "metrics.json"),
        "risk_sizing_validation": str(out_dir / "risk_sizing_validation.json"),
        "validity_gate_json": str(validity_gate_json_path),
        "validity_gate_markdown": str(validity_gate_markdown_path),
        "canonical_report_json": report_artifacts["json"],
        "canonical_report_markdown": report_artifacts["markdown"],
        "canonical_report_html": report_artifacts["html"],
        "canonical_report_json_sha256": report_artifacts["json_sha256"],
        "canonical_report_markdown_sha256": report_artifacts["markdown_sha256"],
        "canonical_report_html_sha256": report_artifacts["html_sha256"],
    }
    if regenerate_manifest:
        manifest_path = regenerate_artifact_manifest(
            out_dir=out_dir,
            config=config,
            metadata=metadata,
            validity_report=validity_report,
        )
        output_manifest["artifact_manifest"] = str(manifest_path)
    return output_manifest


def regenerate_artifact_manifest(
    *,
    out_dir: Path,
    config: PipelineConfig,
    metadata: object,
    validity_report: object,
) -> Path:
    """Regenerate the reproducibility manifest from completed run artifacts."""

    gate_payload = validity_report.to_dict() if hasattr(validity_report, "to_dict") else {}
    system_status = str(gate_payload.get("system_validity_status") or "not_evaluated")
    strategy_status = str(gate_payload.get("strategy_candidate_status") or "not_evaluated")
    metadata_payload = metadata.to_dict() if hasattr(metadata, "to_dict") else {}
    identity = metadata_payload.get("identity", {})
    experiment_id = str(identity.get("experiment_id") or "stage1_canonical_experiment")
    run_id = str(identity.get("run_id") or out_dir.name)

    manifest = build_artifact_manifest_from_paths(
        experiment_id=experiment_id,
        run_id=run_id,
        dataset_paths=[
            {
                "path": out_dir / "market_data.csv",
                "artifact_id": "market_data",
                "schema_id": "market_data_ohlcv",
                "description": "Research input market data snapshot used for feature generation.",
            },
            {
                "path": out_dir / "features.csv",
                "artifact_id": "model_feature_matrix",
                "schema_id": "multimodal_feature_matrix",
                "description": "Features available at t; used for research and validation only.",
            },
        ],
        config_paths=[
            {
                "path": out_dir / "pipeline_config.json",
                "artifact_id": "pipeline_config",
                "schema_id": "pipeline_config",
            },
            {
                "path": out_dir / "canonical_metadata.json",
                "artifact_id": "canonical_metadata",
                "schema_id": "canonical_report_metadata",
            },
            {
                "path": out_dir / "risk_sizing_validation.json",
                "artifact_id": "risk_sizing_validation",
                "schema_id": "portfolio_risk_sizing_validation",
            },
        ],
        model_output_paths=[
            {
                "path": out_dir / "predictions.csv",
                "artifact_id": "model_predictions",
                "description": "Model prediction features; not order signals.",
            },
        ],
        backtest_output_paths=[
            {"path": out_dir / "signals.csv", "artifact_id": "deterministic_signals"},
            {"path": out_dir / "equity_curve.csv", "artifact_id": "equity_curve"},
            {"path": out_dir / "validation_summary.csv", "artifact_id": "walk_forward_summary"},
            {"path": out_dir / "metrics.json", "artifact_id": "performance_metrics"},
            {"path": out_dir / "validity_gate.json", "artifact_id": "validity_gate"},
        ],
        universe_snapshot_path={
            "path": out_dir / "universe_snapshot.json",
            "artifact_id": "universe_snapshot",
            "schema_id": "canonical_universe_snapshot",
        },
        feature_availability_cutoff_path={
            "path": out_dir / "feature_availability_cutoff.json",
            "artifact_id": "feature_availability_cutoff",
            "schema_id": "feature_availability_cutoff",
        },
        report_path=out_dir / "canonical_run_report.md",
        artifact_root=ROOT / "artifacts",
        report_artifact_root=REPORT_ROOT,
        metadata_schema_id=str(metadata_payload.get("schema_id") or "canonical_report_metadata"),
        metadata_schema_version=str(
            metadata_payload.get("schema_version") or "canonical_report_metadata.v1"
        ),
        system_validity_status=system_status,
        strategy_candidate_status=strategy_status,
        repo_path=ROOT,
    )
    return write_artifact_manifest_json(manifest, out_dir / "artifact_manifest.json")


def _write_json_artifact(payload: object, path: Path) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _canonical_report_metadata(config: PipelineConfig):
    benchmark_ticker = str(config.benchmark_ticker).strip().upper()
    universe_tickers = [
        ticker
        for ticker in config.tickers
        if str(ticker).strip().upper() != benchmark_ticker
    ]
    return build_canonical_report_metadata(
        experiment_id="stage1_canonical_experiment",
        run_id=f"backtest_validation_{date.today().strftime('%Y%m%d')}",
        universe_snapshot=UniverseSnapshot.from_tickers(
            universe_tickers,
            experiment_id="stage1_canonical_experiment",
            snapshot_date=config.start,
            benchmark_ticker=config.benchmark_ticker,
        ),
        start_date=config.start,
        end_date=config.end,
        data_sources=(
            ReportDataSource(
                source_id="market_data",
                provider=config.data_mode,
                dataset="ohlcv",
                as_of_date=config.end,
                available_at=datetime.now(UTC),
            ),
            ReportDataSource(
                source_id="news_text",
                provider=config.data_mode,
                dataset="news_items",
                as_of_date=config.end,
                available_at=datetime.now(UTC),
            ),
            ReportDataSource(
                source_id="sec_filings",
                provider=config.data_mode,
                dataset="edgar_filings_and_facts",
                as_of_date=config.end,
                available_at=datetime.now(UTC),
            ),
        ),
        feature_availability_cutoff={
            "price": "date <= t",
            "news_text": "published_at <= t",
            "sec_filing": "accepted_at <= t",
            "model_adapter_outputs": "feature_available_at <= t",
        },
        created_at=datetime.now(UTC),
    )


def _risk_sizing_validation_summary(
    result: PipelineResult,
    config: PipelineConfig,
) -> dict[str, object]:
    metrics = dataclasses.asdict(result.backtest.metrics)
    equity_curve = result.backtest.equity_curve
    latest: dict[str, object] = {}
    if not equity_curve.empty:
        columns = [
            "date",
            "portfolio_volatility_estimate",
            "position_sizing_validation_status",
            "position_sizing_validation_rule",
            "position_sizing_validation_reason",
            "position_count",
            "max_position_weight",
            "max_sector_exposure",
            "gross_exposure",
            "net_exposure",
            "max_position_risk_contribution",
            "post_cost_validation_total_cost_return",
        ]
        latest = (
            equity_curve[[column for column in columns if column in equity_curve.columns]]
            .tail(1)
            .to_dict("records")[0]
        )
    return {
        "covariance_aware_risk": {
            "applied": bool(
                config.covariance_aware_risk_enabled
                and not equity_curve.empty
                and "portfolio_volatility_estimate" in equity_curve.columns
                and equity_curve["portfolio_volatility_estimate"].notna().any()
            ),
            "configured_enabled": bool(config.covariance_aware_risk_enabled),
            "parameters": {
                "return_column": config.covariance_return_column,
                "lookback_periods": int(config.portfolio_covariance_lookback),
                "min_periods": int(config.covariance_min_periods),
                "fallback": "diagonal_predicted_volatility",
                "max_holdings": int(config.top_n),
                "max_symbol_weight": float(config.max_symbol_weight),
                "max_sector_weight": float(config.max_sector_weight),
                "portfolio_volatility_limit": float(config.portfolio_volatility_limit),
                "max_position_risk_contribution": float(
                    config.max_position_risk_contribution
                ),
            },
        },
        "summary": {
            "average_portfolio_volatility_estimate": metrics[
                "average_portfolio_volatility_estimate"
            ],
            "max_portfolio_volatility_estimate": metrics[
                "max_portfolio_volatility_estimate"
            ],
            "max_position_weight": metrics["max_position_weight"],
            "max_sector_exposure": metrics["max_sector_exposure"],
            "max_position_risk_contribution": metrics["max_position_risk_contribution"],
            "position_sizing_validation_pass_rate": metrics[
                "position_sizing_validation_pass_rate"
            ],
            "position_sizing_validation_status": metrics[
                "position_sizing_validation_status"
            ],
            "position_sizing_validation_rule": metrics["position_sizing_validation_rule"],
            "transaction_cost_return": metrics["transaction_cost_return"],
            "slippage_cost_return": metrics["slippage_cost_return"],
            "total_cost_return": metrics["total_cost_return"],
        },
        "latest_observation": latest,
    }


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
    output_manifest = save_outputs(
        result,
        out_dir,
        config,
        regenerate_manifest=not args.skip_manifest_regeneration,
    )
    print_file_summary(out_dir)
    print(f"  Canonical report: {output_manifest['canonical_report_markdown']}")
    if "artifact_manifest" in output_manifest:
        print(f"  Artifact manifest: {output_manifest['artifact_manifest']}")
    print()

    print(SEP_WIDE)
    print("  검증 완료.")
    print(SEP_WIDE)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
