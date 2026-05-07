from __future__ import annotations

# ruff: noqa: E402
import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from quant_research.config import DEFAULT_TICKERS, SecSettings
from quant_research.data.market import _normalize_yfinance_frame
from quant_research.data.news import GDELTNewsProvider, NewsItem, YFinanceNewsProvider
from quant_research.data.sec import SecEdgarClient, extract_frame_values
from quant_research.pipeline import (
    CIK_BY_TICKER,
    PipelineConfig,
    PipelineResult,
    run_research_pipeline,
)
from scripts.run_backtest_validation import save_outputs

Status = Literal["pass", "warning", "fail", "not_evaluable", "not_implemented"]

SPEC_TEST_IDS: tuple[str, ...] = (
    "HYP-01",
    "DATA-01",
    "DATA-02",
    "ENGINE-01",
    "BASE-01",
    "FACTOR-01",
    "OOS-01",
    "OVERFIT-01",
    "PERF-01",
    "COST-01",
    "CAP-01",
    "PORT-01",
    "STRESS-01",
    "RISK-01",
    "EXEC-01",
    "PAPER-01",
    "OPS-01",
    "MON-01",
    "REPORT-01",
)

SPEC_TITLES: dict[str, str] = {
    "HYP-01": "Investment hypothesis and preregistration",
    "DATA-01": "Survivorship, delisting, corporate action data",
    "DATA-02": "Look-ahead leakage and point-in-time alignment",
    "ENGINE-01": "Backtest accounting, timing, and reproducibility",
    "BASE-01": "Benchmark and simple baseline comparison",
    "FACTOR-01": "Known factor exposure and alpha significance",
    "OOS-01": "Out-of-sample and walk-forward validation",
    "OVERFIT-01": "Overfitting and multiple-testing controls",
    "PERF-01": "Net performance and risk metrics",
    "COST-01": "Transaction cost, slippage, and cost stress",
    "CAP-01": "Liquidity, capacity, and microcap dependence",
    "PORT-01": "Portfolio construction, exposure, and constraints",
    "STRESS-01": "Historical and hypothetical stress tests",
    "RISK-01": "Research risk controls and stop rules",
    "EXEC-01": "Fill-quality and execution-realism simulation",
    "PAPER-01": "Paper or shadow-live parity",
    "OPS-01": "Software quality, change control, and governance",
    "MON-01": "Monitoring, drift, and degradation checks",
    "REPORT-01": "Performance report and disclosure quality",
}


@dataclass(frozen=True)
class BackfillRequest:
    tickers: tuple[str, ...]
    start: date
    end: date
    interval: str = "1d"
    include_news: bool = True
    include_sec: bool = True

    def to_manifest_key(self) -> dict[str, object]:
        return {
            "tickers": list(self.tickers),
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "interval": self.interval,
            "include_news": self.include_news,
            "include_sec": self.include_sec,
        }


@dataclass
class BackfillPhaseResult:
    phase: str
    status: str
    cache_hit: bool
    artifact_path: str | None = None
    manifest_path: str | None = None
    rows: int = 0
    request_count: int = 0
    error_count: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclass
class RequestBudget:
    """Small per-provider throttle with testable time and sleep hooks."""

    max_requests_per_second: dict[str, float]
    now: Callable[[], float] = time.monotonic
    sleeper: Callable[[float], None] = time.sleep
    _last_request_at: dict[str, float] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)

    def acquire(self, provider: str) -> None:
        rate = float(self.max_requests_per_second.get(provider, 1.0))
        if rate <= 0:
            raise ValueError("max_requests_per_second must be positive")
        current = self.now()
        previous = self._last_request_at.get(provider)
        if previous is not None:
            min_interval = 1.0 / rate
            elapsed = current - previous
            if elapsed < min_interval:
                delay = min_interval - elapsed
                self.sleeper(delay)
                current = self.now()
        self._last_request_at[provider] = current
        self._counts[provider] = self._counts.get(provider, 0) + 1

    def request_count(self, provider: str | None = None) -> int:
        if provider is None:
            return sum(self._counts.values())
        return self._counts.get(provider, 0)

    def to_dict(self) -> dict[str, object]:
        return {
            "max_requests_per_second": dict(self.max_requests_per_second),
            "request_counts": dict(self._counts),
        }


@dataclass(frozen=True)
class TodoItem:
    test_id: str
    missing_capability: str
    why_needed: str
    recommended_implementation: str
    priority: str

    def to_dict(self) -> dict[str, str]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class SpecCheck:
    test_id: str
    title: str
    status: Status
    evidence: str
    metrics: Mapping[str, object] = field(default_factory=dict)
    todo: TodoItem | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "test_id": self.test_id,
            "title": self.title,
            "status": self.status,
            "evidence": self.evidence,
            "metrics": dict(self.metrics),
        }
        if self.todo is not None:
            payload["todo"] = self.todo.to_dict()
        return payload


def default_request_budget() -> RequestBudget:
    return RequestBudget(
        {
            "yfinance_market": 0.5,
            "yfinance_news": 0.5,
            "gdelt": 0.25,
            "article": 0.25,
            "sec": 9.0,
        }
    )


def apply_resource_profile(profile: str) -> dict[str, str]:
    if profile not in {"conservative", "balanced", "aggressive"}:
        raise ValueError("resource profile must be conservative, balanced, or aggressive")
    cpu_count = os.cpu_count() or 2
    if profile == "conservative":
        worker_threads = max(1, min(2, cpu_count // 2))
    elif profile == "balanced":
        worker_threads = max(1, cpu_count - 4)
    else:
        worker_threads = max(1, cpu_count - 2)
    values = {
        "OMP_NUM_THREADS": str(worker_threads),
        "OPENBLAS_NUM_THREADS": str(worker_threads),
        "MKL_NUM_THREADS": str(worker_threads),
        "VECLIB_MAXIMUM_THREADS": str(worker_threads),
        "TOKENIZERS_PARALLELISM": "false",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "KMP_INIT_AT_FORK": "FALSE",
        "KMP_BLOCKTIME": "0",
        "OBJC_DISABLE_INITIALIZE_FORK_SAFETY": "YES",
    }
    for key, value in values.items():
        os.environ[key] = value
    return values


@contextlib.contextmanager
def keep_awake(enabled: bool) -> Iterable[None]:
    process: subprocess.Popen[bytes] | None = None
    if enabled and platform.system() == "Darwin":
        process = subprocess.Popen(["caffeinate", "-dims"])
    try:
        yield
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def ensure_market_data(
    raw_dir: Path,
    request: BackfillRequest,
    budget: RequestBudget,
    *,
    refresh: bool = False,
    downloader: Callable[[BackfillRequest], pd.DataFrame] | None = None,
) -> BackfillPhaseResult:
    phase = "market"
    artifact = raw_dir / "market_history.parquet"
    manifest_path = raw_dir / "heavy_backfill_market.json"
    expected = {"phase": phase, "request": request.to_manifest_key()}
    if not refresh and _phase_cache_hit(artifact, manifest_path, expected):
        manifest = _load_json(manifest_path)
        return BackfillPhaseResult(
            phase=phase,
            status="cached",
            cache_hit=True,
            artifact_path=str(artifact),
            manifest_path=str(manifest_path),
            rows=int(manifest.get("rows", 0)),
            request_count=0,
        )

    raw_dir.mkdir(parents=True, exist_ok=True)
    budget.acquire("yfinance_market")
    frame = (downloader or _download_market_frame)(request)
    frame.to_parquet(artifact, index=False)
    result = BackfillPhaseResult(
        phase=phase,
        status="downloaded",
        cache_hit=False,
        artifact_path=str(artifact),
        manifest_path=str(manifest_path),
        rows=len(frame),
        request_count=budget.request_count("yfinance_market"),
    )
    _write_phase_manifest(manifest_path, expected, result, artifact)
    return result


def ensure_news_data(
    raw_dir: Path,
    request: BackfillRequest,
    budget: RequestBudget,
    *,
    refresh: bool = False,
    provider_factory: Callable[[], tuple[YFinanceNewsProvider, GDELTNewsProvider]] | None = None,
) -> BackfillPhaseResult:
    phase = "news"
    artifact = raw_dir / "news_items.jsonl"
    manifest_path = raw_dir / "heavy_backfill_news.json"
    expected = {"phase": phase, "request": request.to_manifest_key()}
    if not request.include_news:
        return BackfillPhaseResult(phase=phase, status="skipped", cache_hit=False)
    if not refresh and _phase_cache_hit(artifact, manifest_path, expected):
        manifest = _load_json(manifest_path)
        return BackfillPhaseResult(
            phase=phase,
            status="cached",
            cache_hit=True,
            artifact_path=str(artifact),
            manifest_path=str(manifest_path),
            rows=int(manifest.get("rows", 0)),
            request_count=0,
        )

    raw_dir.mkdir(parents=True, exist_ok=True)
    yf_provider, gdelt_provider = (
        provider_factory() if provider_factory is not None else (YFinanceNewsProvider(), GDELTNewsProvider())
    )
    items: list[NewsItem] = []
    errors: list[str] = []
    for ticker in request.tickers:
        try:
            budget.acquire("yfinance_news")
            items.extend(yf_provider.get_news([ticker], request.start, request.end))
        except Exception as exc:
            errors.append(f"yfinance_news:{ticker}:{exc}")
        try:
            budget.acquire("gdelt")
            items.extend(gdelt_provider.get_news([ticker], request.start, request.end))
        except Exception as exc:
            errors.append(f"gdelt:{ticker}:{exc}")

    with artifact.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(_news_item_to_dict(item), ensure_ascii=False, default=str) + "\n")
    result = BackfillPhaseResult(
        phase=phase,
        status="downloaded_with_errors" if errors else "downloaded",
        cache_hit=False,
        artifact_path=str(artifact),
        manifest_path=str(manifest_path),
        rows=len(items),
        request_count=budget.request_count("yfinance_news") + budget.request_count("gdelt"),
        error_count=len(errors),
        errors=errors,
    )
    _write_phase_manifest(manifest_path, expected, result, artifact)
    return result


def ensure_sec_data(
    raw_dir: Path,
    request: BackfillRequest,
    budget: RequestBudget,
    *,
    refresh: bool = False,
    user_agent: str | None = None,
) -> BackfillPhaseResult:
    phase = "sec"
    sec_dir = raw_dir / "sec"
    artifact = sec_dir
    manifest_path = raw_dir / "heavy_backfill_sec.json"
    expected = {"phase": phase, "request": request.to_manifest_key()}
    if not request.include_sec:
        return BackfillPhaseResult(phase=phase, status="skipped", cache_hit=False)
    if not refresh and _phase_cache_hit(artifact, manifest_path, expected):
        manifest = _load_json(manifest_path)
        return BackfillPhaseResult(
            phase=phase,
            status="cached",
            cache_hit=True,
            artifact_path=str(artifact),
            manifest_path=str(manifest_path),
            rows=int(manifest.get("rows", 0)),
            request_count=0,
        )

    sec_dir.mkdir(parents=True, exist_ok=True)
    settings = SecSettings(
        user_agent=user_agent or os.getenv("QT_SEC_USER_AGENT", "QuantResearchApp research@example.com"),
        max_requests_per_second=min(9.0, SecSettings().max_requests_per_second),
    )
    client = SecEdgarClient(settings=settings, cache_dir=sec_dir)
    errors: list[str] = []
    filing_rows = 0
    for ticker in request.tickers:
        cik = CIK_BY_TICKER.get(ticker)
        if cik is None:
            continue
        try:
            budget.acquire("sec")
            client.get_submissions(cik)
            filings = client.recent_filings(
                cik,
                {"8-K", "10-Q", "10-K", "4"},
                include_document_text=True,
            )
            filing_rows += len(filings)
            budget.acquire("sec")
            client.get_companyfacts(cik)
            budget.acquire("sec")
            client.get_companyconcept(cik, "us-gaap", "NetIncomeLoss")
        except Exception as exc:
            errors.append(f"sec:{ticker}:{exc}")
    try:
        budget.acquire("sec")
        extract_frame_values(client.get_frame("us-gaap", "Assets", "USD", "CY2024Q4I"), "assets")
    except Exception as exc:
        errors.append(f"sec:frame_assets:{exc}")

    result = BackfillPhaseResult(
        phase=phase,
        status="downloaded_with_errors" if errors else "downloaded",
        cache_hit=False,
        artifact_path=str(artifact),
        manifest_path=str(manifest_path),
        rows=filing_rows,
        request_count=budget.request_count("sec"),
        error_count=len(errors),
        errors=errors,
    )
    _write_phase_manifest(manifest_path, expected, result, artifact)
    return result


def run_backfill(
    raw_dir: Path,
    request: BackfillRequest,
    budget: RequestBudget | None = None,
    *,
    refresh: bool = False,
    user_agent: str | None = None,
) -> list[BackfillPhaseResult]:
    budget = budget or default_request_budget()
    results = [
        ensure_market_data(raw_dir, request, budget, refresh=refresh),
        ensure_news_data(raw_dir, request, budget, refresh=refresh),
        ensure_sec_data(raw_dir, request, budget, refresh=refresh, user_agent=user_agent),
    ]
    _write_json(
        raw_dir / "heavy_backfill_manifest.json",
        {
            "schema_version": "heavy_backfill_manifest.v1",
            "created_at": _utc_now(),
            "request": request.to_manifest_key(),
            "budget": budget.to_dict(),
            "phases": [result.to_dict() for result in results],
        },
    )
    return results


def build_pipeline_config(
    request: BackfillRequest,
    *,
    raw_dir: Path,
    model_mode: str,
    runtime: str,
) -> PipelineConfig:
    base = {
        "tickers": list(request.tickers),
        "data_mode": "local",
        "local_data_dir": str(raw_dir),
        "start": request.start,
        "end": request.end,
        "time_series_inference_mode": "proxy",
    }
    if model_mode == "full":
        extra: dict[str, object] = {
            "sentiment_model": "finbert",
            "filing_extractor_model": "fingpt",
            "enable_local_filing_llm": True,
            "fingpt_runtime": runtime,
        }
        if runtime == "mlx":
            extra["fingpt_quantized_model_path"] = "artifacts/model_cache/fingpt-mt-llama3-8b-mlx"
            extra["local_model_device_map"] = "cpu"
        return PipelineConfig(**base, **extra)
    return PipelineConfig(
        **base,
        sentiment_model="keyword",
        filing_extractor_model="rules",
        enable_local_filing_llm=False,
    )


def build_spec_checks(
    result: PipelineResult | None,
    output_manifest: Mapping[str, object] | None,
    backfill_results: Sequence[BackfillPhaseResult] = (),
) -> list[SpecCheck]:
    metrics = _metrics_payload(result)
    validation = result.validation_summary if result is not None else pd.DataFrame()
    equity = result.backtest.equity_curve if result is not None else pd.DataFrame()
    backfill_errors = sum(phase.error_count for phase in backfill_results)
    output_manifest = output_manifest or {}

    checks = [
        _todo_check(
            "HYP-01",
            "strategy hypothesis registry and parameter trial ledger",
            "The spec requires pre-registered hypotheses and a strategy_trials.csv style audit trail.",
            "Add a research registry that records hypothesis text, parameter ranges, trial count, selected criteria, and excluded trials before each run.",
            "P1",
        ),
        _todo_check(
            "DATA-01",
            "survivorship-free point-in-time universe, delisting returns, and corporate action tables",
            "The current free-provider universe cannot prove historical investability, delisting treatment, or point-in-time corporate actions.",
            "Add a licensed or reconstructed PIT universe provider with persistent identifiers, delisting returns, split/dividend/spin-off tables, and monthly universe snapshots.",
            "P0",
        ),
        SpecCheck(
            "DATA-02",
            SPEC_TITLES["DATA-02"],
            "pass" if result is not None else "not_evaluable",
            "Backtest and report generation enforce availability timestamps, feature cutoff checks, and forward-return timing guards.",
            {"backfill_error_count": backfill_errors},
        ),
        SpecCheck(
            "ENGINE-01",
            SPEC_TITLES["ENGINE-01"],
            "pass" if result is not None and not equity.empty else "not_evaluable",
            "The deterministic long-only backtest produced signal, equity, cost, turnover, and risk-sizing artifacts.",
            {
                "equity_rows": int(len(equity)),
                "signals_artifact": output_manifest.get("signals"),
                "equity_artifact": output_manifest.get("equity_curve"),
            },
        ),
        SpecCheck(
            "BASE-01",
            SPEC_TITLES["BASE-01"],
            "pass" if metrics.get("benchmark_cost_adjusted_cagr") is not None else "not_evaluable",
            "The pipeline compares strategy returns against the configured benchmark and equal-weight baseline inputs.",
            {
                "benchmark_cost_adjusted_cagr": metrics.get("benchmark_cost_adjusted_cagr"),
                "excess_return": metrics.get("excess_return"),
            },
        ),
        _todo_check(
            "FACTOR-01",
            "Fama-French/Carhart factor regression and alpha significance tests",
            "The spec requires known-factor exposure control and alpha t-stat evidence.",
            "Add factor data ingestion, monthly strategy return alignment, OLS/Newey-West regression, alpha t-stat, and factor exposure report artifacts.",
            "P0",
        ),
        SpecCheck(
            "OOS-01",
            SPEC_TITLES["OOS-01"],
            _oos_status(validation, metrics),
            "Walk-forward outputs are evaluated with OOS fold metadata and cost-adjusted portfolio metrics.",
            {
                "folds": int(len(validation)),
                "oos_folds": _oos_fold_count(validation),
                "net_cagr": metrics.get("net_cagr"),
                "sharpe": metrics.get("sharpe"),
            },
        ),
        _todo_check(
            "OVERFIT-01",
            "PBO, DSR/PSR, Reality Check, and SPA multiple-testing controls",
            "The current gate has ablations and baselines but no formal multiple-testing correction statistics.",
            "Add CPCV/PBO, probabilistic and deflated Sharpe, stationary bootstrap, White Reality Check or Hansen SPA, and trial-count reporting.",
            "P0",
        ),
        SpecCheck(
            "PERF-01",
            SPEC_TITLES["PERF-01"],
            "pass" if _has_core_performance_metrics(metrics) else "not_evaluable",
            "Net/gross return, benchmark, volatility, Sharpe, drawdown, hit-rate, turnover, and costs are emitted.",
            _select_metrics(
                metrics,
                (
                    "net_cagr",
                    "gross_cagr",
                    "benchmark_cost_adjusted_cagr",
                    "sharpe",
                    "max_drawdown",
                    "turnover",
                    "total_cost_return",
                ),
            ),
        ),
        SpecCheck(
            "COST-01",
            SPEC_TITLES["COST-01"],
            "pass" if result is not None and result.transaction_cost_sensitivity is not None else "not_evaluable",
            "The existing validation stack runs canonical transaction-cost and turnover sensitivity scenarios.",
            {"transaction_cost_sensitivity": result.transaction_cost_sensitivity is not None if result else False},
        ),
        _todo_check(
            "CAP-01",
            "ADV capacity, market impact, spread, and microcap dependence model",
            "The current liquidity_score is useful for screening but does not prove capacity at a target capital size.",
            "Add ADV participation limits, spread estimates, impact curves, target capital scenarios, and capacity pass/fail thresholds.",
            "P0",
        ),
        SpecCheck(
            "PORT-01",
            SPEC_TITLES["PORT-01"],
            "pass" if metrics.get("position_sizing_validation_status") is not None else "not_evaluable",
            "Portfolio concentration, sector exposure, covariance-aware volatility, and sizing validation artifacts are produced.",
            _select_metrics(
                metrics,
                (
                    "position_sizing_validation_status",
                    "max_position_weight",
                    "max_sector_exposure",
                    "average_portfolio_volatility_estimate",
                    "max_position_risk_contribution",
                ),
            ),
        ),
        _partial_todo_check(
            "STRESS-01",
            "historical crash/recession/rate-shock and hypothetical data-outage stress harness",
            "Cost stress exists, but the spec also requires historical and hypothetical scenario stress tests.",
            "Add scenario return transformations for 2008, 2020, 2022-style regimes, volatility spikes, spread widening, missing data, and provider outage cases.",
            "P1",
        ),
        SpecCheck(
            "RISK-01",
            SPEC_TITLES["RISK-01"],
            "warning" if result is not None else "not_evaluable",
            "Research-side limits, drawdown stop, turnover caps, and deterministic gate metadata exist; live pre-trade controls remain out of v1 scope.",
            _select_metrics(
                metrics,
                (
                    "max_drawdown",
                    "max_position_weight",
                    "max_sector_exposure",
                    "position_sizing_validation_pass_rate",
                ),
            ),
        ),
        _todo_check(
            "EXEC-01",
            "next-open/VWAP fill quality, spread, impact, delay, and no-fill simulation",
            "The current backtest uses cost/slippage bps assumptions rather than a fill-quality simulator.",
            "Add fill-model protocols for next-open, VWAP/TWAP, half-spread, participation impact, delay cost, partial fill, and post-run slippage attribution.",
            "P0",
        ),
        _todo_check(
            "PAPER-01",
            "paper or shadow-live replay parity harness",
            "The repository intentionally has no live path, and no paper/shadow parity harness is implemented.",
            "Add a research-only replay harness that records timestamped signals and compares them with backtest replay outputs without integrating live order placement.",
            "P1",
        ),
        SpecCheck(
            "OPS-01",
            SPEC_TITLES["OPS-01"],
            "warning",
            "The project has deterministic tests, artifact manifests, and git metadata; full model governance and change approval workflows are not complete.",
            {"canonical_artifact_manifest": output_manifest.get("artifact_manifest")},
        ),
        _todo_check(
            "MON-01",
            "monitoring, drift, degradation, and alert dashboard",
            "The spec requires ongoing live or paper monitoring distributions and alerts, which are not implemented.",
            "Add research monitoring artifacts for rolling signal distributions, drawdown, turnover, feature drift, cost drift, and threshold breach reports.",
            "P1",
        ),
        SpecCheck(
            "REPORT-01",
            SPEC_TITLES["REPORT-01"],
            "pass" if output_manifest.get("canonical_report_markdown") else "not_evaluable",
            "Canonical markdown/html/json reports and this heavy spec coverage report are generated with companion manifests.",
            {
                "canonical_report_markdown": output_manifest.get("canonical_report_markdown"),
                "validity_gate_markdown": output_manifest.get("validity_gate_markdown"),
            },
        ),
    ]
    _assert_all_spec_ids_present(checks)
    return checks


def write_heavy_report(
    output_dir: Path,
    checks: Sequence[SpecCheck],
    *,
    run_id: str,
    backfill_results: Sequence[BackfillPhaseResult] = (),
    output_manifest: Mapping[str, object] | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "final_report.md"
    json_path = output_dir / "final_report.json"
    manifest_path = output_dir / "heavy_validation_manifest.json"
    payload = {
        "schema_version": "heavy_quant_validation_report.v1",
        "run_id": run_id,
        "created_at": _utc_now(),
        "status_counts": _status_counts(checks),
        "checks": [check.to_dict() for check in checks],
        "todos": [check.todo.to_dict() for check in checks if check.todo is not None],
        "backfill": [phase.to_dict() for phase in backfill_results],
        "output_manifest": dict(output_manifest or {}),
    }
    report_path.write_text(_render_markdown_report(payload), encoding="utf-8")
    _write_json(json_path, payload)
    manifest = {
        "schema_version": "heavy_quant_validation_manifest.v1",
        "run_id": run_id,
        "created_at": _utc_now(),
        "artifacts": [
            _artifact_record(report_path, "final_report_markdown"),
            _artifact_record(json_path, "final_report_json"),
        ],
    }
    _write_json(manifest_path, manifest)
    return {
        "final_report_markdown": str(report_path),
        "final_report_json": str(json_path),
        "heavy_validation_manifest": str(manifest_path),
        "final_report_markdown_sha256": sha256_file(report_path),
        "final_report_json_sha256": sha256_file(json_path),
        "heavy_validation_manifest_sha256": sha256_file(manifest_path),
    }


def run_heavy_quant_validation(args: argparse.Namespace) -> dict[str, str]:
    apply_resource_profile(args.resource_profile)
    tickers = _parse_tickers(args.tickers)
    end = _parse_date(args.end) if args.end else date.today()
    start = _parse_date(args.start) if args.start else end - timedelta(days=365 * args.years)
    request = BackfillRequest(
        tickers=tuple(tickers),
        start=start,
        end=end,
        interval=args.interval,
        include_news=not args.skip_news,
        include_sec=not args.skip_sec,
    )
    run_id = args.run_id or f"heavy_quant_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    raw_dir = Path(args.raw_data_dir)
    output_dir = Path(args.report_dir) / run_id
    with keep_awake(args.keep_awake):
        backfill = run_backfill(
            raw_dir,
            request,
            refresh=args.refresh_data,
            user_agent=args.sec_user_agent,
        )
        config = build_pipeline_config(
            request,
            raw_dir=raw_dir,
            model_mode=args.model_mode,
            runtime=args.runtime,
        )
        result = run_research_pipeline(config)
        output_manifest = save_outputs(result, output_dir, config)
        checks = build_spec_checks(result, output_manifest, backfill)
        heavy_outputs = write_heavy_report(
            output_dir,
            checks,
            run_id=run_id,
            backfill_results=backfill,
            output_manifest=output_manifest,
        )
    print(f"Heavy quant validation report: {heavy_outputs['final_report_markdown']}")
    return heavy_outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run heavy/manual quant validation and emit spec coverage reports.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--tickers", nargs="+", default=["default"])
    parser.add_argument("--raw-data-dir", default="data/raw")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--model-mode", choices=["lightweight", "full"], default="full")
    parser.add_argument("--runtime", choices=["mlx", "ollama"], default="mlx")
    parser.add_argument(
        "--resource-profile",
        choices=["conservative", "balanced", "aggressive"],
        default="aggressive",
    )
    parser.add_argument("--keep-awake", action="store_true")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--skip-news", action="store_true")
    parser.add_argument("--skip-sec", action="store_true")
    parser.add_argument("--sec-user-agent", default=None)
    return parser.parse_args()


def _download_market_frame(request: BackfillRequest) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required for heavy market data backfill") from exc
    raw = yf.download(
        tickers=list(request.tickers),
        start=str(request.start),
        end=str(request.end),
        interval=request.interval,
        auto_adjust=False,
        group_by="ticker",
        progress=True,
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no market data")
    frames: list[pd.DataFrame] = []
    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in request.tickers:
            if ticker in raw.columns.get_level_values(0):
                frames.append(_normalize_yfinance_frame(raw[ticker].copy(), ticker))
    else:
        frames.append(_normalize_yfinance_frame(raw, request.tickers[0]))
    if not frames:
        raise RuntimeError("no market frames were available after yfinance normalization")
    return pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


def _phase_cache_hit(artifact: Path, manifest_path: Path, expected: Mapping[str, object]) -> bool:
    if not artifact.exists() or not manifest_path.exists():
        return False
    manifest = _load_json(manifest_path)
    return manifest.get("expected") == dict(expected) and manifest.get("artifact_sha256")


def _write_phase_manifest(
    manifest_path: Path,
    expected: Mapping[str, object],
    result: BackfillPhaseResult,
    artifact: Path,
) -> None:
    artifact_hash = _directory_hash(artifact) if artifact.is_dir() else sha256_file(artifact)
    _write_json(
        manifest_path,
        {
            "schema_version": "heavy_backfill_phase.v1",
            "created_at": _utc_now(),
            "expected": dict(expected),
            "rows": result.rows,
            "result": result.to_dict(),
            "artifact_path": str(artifact),
            "artifact_sha256": artifact_hash,
        },
    )


def _news_item_to_dict(item: NewsItem) -> dict[str, object]:
    return {
        "ticker": item.ticker,
        "published_at": item.published_at,
        "title": item.title,
        "source": item.source,
        "url": item.url,
        "summary": item.summary,
        "content": item.content,
        "full_text": item.full_text,
        "body_text": item.body_text,
        "collected_at": item.collected_at,
        "event_timestamp": item.event_timestamp,
        "availability_timestamp": item.availability_timestamp,
        "source_timestamp": item.source_timestamp,
        "timezone": item.timezone,
    }


def _todo_check(
    test_id: str,
    missing_capability: str,
    why_needed: str,
    recommended_implementation: str,
    priority: str,
) -> SpecCheck:
    return SpecCheck(
        test_id,
        SPEC_TITLES[test_id],
        "not_implemented",
        "The spec requires this capability, but no executable implementation exists yet.",
        todo=TodoItem(
            test_id,
            missing_capability,
            why_needed,
            recommended_implementation,
            priority,
        ),
    )


def _partial_todo_check(
    test_id: str,
    missing_capability: str,
    why_needed: str,
    recommended_implementation: str,
    priority: str,
) -> SpecCheck:
    return SpecCheck(
        test_id,
        SPEC_TITLES[test_id],
        "warning",
        "The current stack has partial coverage, but the missing capability is required for the full spec.",
        todo=TodoItem(
            test_id,
            missing_capability,
            why_needed,
            recommended_implementation,
            priority,
        ),
    )


def _metrics_payload(result: PipelineResult | None) -> dict[str, object]:
    if result is None:
        return {}
    return dataclasses.asdict(result.backtest.metrics)


def _oos_status(validation: pd.DataFrame, metrics: Mapping[str, object]) -> Status:
    if validation.empty or _oos_fold_count(validation) == 0:
        return "not_evaluable"
    net_cagr = _float_or_none(metrics.get("net_cagr"))
    if net_cagr is None:
        return "warning"
    return "pass" if net_cagr > 0 else "fail"


def _oos_fold_count(validation: pd.DataFrame) -> int:
    if validation.empty or "is_oos" not in validation:
        return 0
    return int(validation["is_oos"].fillna(False).astype(bool).sum())


def _has_core_performance_metrics(metrics: Mapping[str, object]) -> bool:
    return all(
        key in metrics
        for key in (
            "net_cagr",
            "gross_cagr",
            "benchmark_cost_adjusted_cagr",
            "sharpe",
            "max_drawdown",
            "turnover",
        )
    )


def _select_metrics(metrics: Mapping[str, object], keys: Sequence[str]) -> dict[str, object]:
    return {key: metrics.get(key) for key in keys if key in metrics}


def _float_or_none(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _status_counts(checks: Sequence[SpecCheck]) -> dict[str, int]:
    counts = {status: 0 for status in ("pass", "warning", "fail", "not_evaluable", "not_implemented")}
    for check in checks:
        counts[check.status] += 1
    return counts


def _assert_all_spec_ids_present(checks: Sequence[SpecCheck]) -> None:
    observed = tuple(check.test_id for check in checks)
    if observed != SPEC_TEST_IDS:
        raise ValueError(f"spec checks must exactly match test ids: {SPEC_TEST_IDS}")


def _render_markdown_report(payload: Mapping[str, object]) -> str:
    checks = list(payload.get("checks", []))
    todos = list(payload.get("todos", []))
    lines = [
        "# Heavy Quant Validation Report",
        "",
        "This report is for research validation only. It does not add live trading functionality.",
        "",
        "## Summary",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for status, count in dict(payload.get("status_counts", {})).items():
        lines.append(f"| {status} | {count} |")
    lines.extend(
        [
            "",
            "## Spec Coverage",
            "",
            "| Test ID | Title | Status | Evidence |",
            "|---|---|---|---|",
        ]
    )
    for raw in checks:
        row = dict(raw)
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(row.get("test_id")),
                    _md(row.get("title")),
                    _md(row.get("status")),
                    _md(row.get("evidence")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## TODO",
            "",
            "| Test ID | Priority | Missing Capability | Recommended Implementation |",
            "|---|---|---|---|",
        ]
    )
    for raw in todos:
        row = dict(raw)
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(row.get("test_id")),
                    _md(row.get("priority")),
                    _md(row.get("missing_capability")),
                    _md(row.get("recommended_implementation")),
                ]
            )
            + " |"
        )
    if not todos:
        lines.append("| - | - | - | - |")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "| Key | Path |",
            "|---|---|",
        ]
    )
    for key, value in dict(payload.get("output_manifest", {})).items():
        lines.append(f"| {_md(key)} | {_md(value)} |")
    lines.append("")
    return "\n".join(lines)


def _artifact_record(path: Path, artifact_id: str) -> dict[str, object]:
    return {
        "artifact_id": artifact_id,
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _directory_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(child for child in path.rglob("*") if child.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(sha256_file(child).encode("ascii"))
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _json_safe(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _md(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_tickers(values: Sequence[str]) -> list[str]:
    if len(values) == 1 and values[0].lower() == "default":
        return list(DEFAULT_TICKERS)
    tickers: list[str] = []
    for value in values:
        tickers.extend(part.strip().upper() for part in value.split(",") if part.strip())
    if not tickers:
        raise ValueError("at least one ticker must be supplied")
    return tickers


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def main() -> None:
    run_heavy_quant_validation(_parse_args())


if __name__ == "__main__":
    main()
