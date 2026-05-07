from __future__ import annotations

# ruff: noqa: E402, I001

import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from inspect import Parameter, signature

from quant_research.runtime import configure_local_runtime_defaults

configure_local_runtime_defaults()

import numpy as np
import pandas as pd

from quant_research.backtest.engine import BacktestConfig, BacktestResult, run_long_only_backtest
from quant_research.backtest.metrics import PerformanceMetrics
from quant_research.config import DEFAULT_BENCHMARK_TICKER, DEFAULT_TICKERS
from quant_research.data.market import (
    LocalMarketDataProvider,
    SyntheticMarketDataProvider,
    YFinanceMarketDataProvider,
)
from quant_research.data.news import (
    GDELTNewsProvider,
    LocalNewsProvider,
    SyntheticNewsProvider,
    YFinanceNewsProvider,
)
from quant_research.data.sec import (
    SecEdgarClient,
    SyntheticSecProvider,
    extract_companyconcept_frame,
    extract_companyfacts_frame,
    extract_frame_values,
    merge_fact_frames,
)
from quant_research.features.fusion import fuse_features
from quant_research.features.price import build_price_features
from quant_research.features.sec import DEFAULT_FILING_EVENT_CACHE_PATH, build_sec_features
from quant_research.features.text import KeywordSentimentAnalyzer, build_news_features
from quant_research.models.text import (
    FilingEventExtractor,
    FinBERTSentimentAnalyzer,
    FinGPTEventExtractor,
)
from quant_research.models.tabular import infer_feature_columns
from quant_research.models.timeseries import Chronos2Adapter, GraniteTTMAdapter
from quant_research.validation.ablation import (
    AblationScenarioConfig,
    AblationScenarioRegistry,
    AblationToggles,
    DEFAULT_VALIDITY_GATE_ABLATION_MODE_IDS,
    FeatureFamily,
    NO_COST_ABLATION_SCENARIO,
    default_ablation_registry,
    feature_family_columns,
    feature_family_for_column,
    normalize_validity_gate_ablation_mode_ids,
)
from quant_research.validation.benchmark_inputs import (
    BaselineComparisonInput,
    BenchmarkConstructionInputs,
    TickerUniverse,
    build_benchmark_construction_inputs,
    build_benchmark_return_series,
    build_equal_weight_baseline_equity_curve,
    build_equal_weight_baseline_return_series,
    calculate_equal_weight_baseline_performance_metrics,
)
from quant_research.validation.config import (
    DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG,
    ModelComparisonConfig,
    PortfolioRiskConstraintConfig,
    TransactionCostSensitivityConfig,
    default_model_comparison_config,
    default_transaction_cost_sensitivity_config,
)
from quant_research.validation.horizons import (
    DEFAULT_PURGE_EMBARGO_DAYS,
    DEFAULT_VALIDATION_HORIZONS,
    REQUIRED_VALIDATION_HORIZON_DAYS,
    forward_return_column,
)
from quant_research.validation.sensitivity import (
    TransactionCostSensitivityBatchResult,
    run_transaction_cost_sensitivity_batch,
)
from quant_research.validation.walk_forward import (
    PurgeEmbargoWalkForwardConfig,
    PurgeEmbargoWalkForwardSplitter,
    WalkForwardConfig,
    WalkForwardSplitter,
    walk_forward_predict,
)

CIK_BY_TICKER = {
    "AAPL": "320193",
    "MSFT": "789019",
    "NVDA": "1045810",
    "AMZN": "1018724",
    "GOOGL": "1652044",
    "META": "1326801",
    "TSLA": "1318605",
    "JPM": "19617",
    "V": "1403161",
    "MA": "1141391",
    "UNH": "731766",
    "XOM": "34088",
    "JNJ": "200406",
    "PG": "80424",
    "HD": "354950",
    "COST": "909832",
    "ABBV": "1551152",
    "BAC": "70858",
    "KO": "21344",
    "PEP": "77476",
    "WMT": "104169",
    "AVGO": "1730168",
    "LLY": "59478",
    "MRK": "310158",
    "CVX": "93410",
    "CRM": "1108524",
    "AMD": "2488",
    "NFLX": "1065280",
    "ORCL": "1341439",
    "ADBE": "796343",
}


def _is_macos() -> bool:
    return sys.platform == "darwin"


def _default_fingpt_runtime() -> str:
    return "mlx" if _is_macos() else "llama-cpp"


def _default_fingpt_quantized_model_path() -> str:
    if _is_macos():
        return "artifacts/model_cache/fingpt-mt-llama3-8b-mlx"
    return "artifacts/model_cache/fingpt-mt-llama3-8b-lora-q4_0.gguf"


@dataclass(frozen=True)
class PipelineConfig:
    tickers: list[str] = field(default_factory=lambda: list(DEFAULT_TICKERS))
    data_mode: str = "synthetic"  # "synthetic" | "live" | "local"
    local_data_dir: str = "data/raw"  # used when data_mode="local"
    start: date = field(default_factory=lambda: date.today() - timedelta(days=365 * 2))
    end: date = field(default_factory=date.today)
    interval: str = "1d"
    train_periods: int = 90
    test_periods: int = 20
    gap_periods: int = DEFAULT_PURGE_EMBARGO_DAYS
    embargo_periods: int = DEFAULT_PURGE_EMBARGO_DAYS
    model_name: str = "lightgbm"
    prediction_target_column: str = forward_return_column(REQUIRED_VALIDATION_HORIZON_DAYS)
    required_validation_horizon: int = REQUIRED_VALIDATION_HORIZON_DAYS
    top_n: int = DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG.max_holdings
    cost_bps: float = 5.0
    slippage_bps: float = 2.0
    average_daily_turnover_budget: float = 0.25
    cost_adjusted_collapse_threshold: float = 0.0
    monthly_turnover_budget: float | None = None
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER
    sentiment_model: str = "finbert"
    time_series_inference_mode: str = "proxy"
    time_series_target_column: str = "return_20"
    time_series_min_context: int = 64
    max_time_series_inference_windows: int | None = None
    chronos_model_id: str = "amazon/chronos-2"
    granite_ttm_model_id: str = "ibm-granite/granite-timeseries-ttm-r2"
    granite_ttm_revision: str | None = None
    local_model_device_map: str = "auto"
    local_model_offload_folder: str | None = "artifacts/model_offload"
    filing_extractor_model: str = "fingpt"
    enable_local_filing_llm: bool = True
    finma_model_id: str = "ChanceFocus/finma-7b-nlp"
    fingpt_model_id: str = "FinGPT/fingpt-mt_llama3-8b_lora"
    fingpt_base_model_id: str | None = "meta-llama/Meta-Llama-3-8B"
    fingpt_runtime: str = field(default_factory=_default_fingpt_runtime)
    fingpt_quantized_model_path: str = field(default_factory=_default_fingpt_quantized_model_path)
    fingpt_allow_unquantized_transformers: bool = False
    fingpt_single_load_lock_path: str | None = "artifacts/model_locks/fingpt-local-load.lock"
    max_symbol_weight: float = DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG.max_symbol_weight
    max_sector_weight: float = DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG.max_sector_weight
    max_correlation_cluster_weight: float = 0.70
    correlation_cluster_threshold: float = 0.80
    portfolio_covariance_lookback: int = (
        DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG.portfolio_covariance_lookback
    )
    covariance_aware_risk_enabled: bool = (
        DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG.covariance_aware_risk.enabled
    )
    covariance_return_column: str = (
        DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG.covariance_aware_risk.return_column
    )
    covariance_min_periods: int = (
        DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG.covariance_aware_risk.min_periods
    )
    max_daily_turnover: float | None = None
    portfolio_volatility_limit: float = (
        DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG.portfolio_volatility_limit
    )
    max_drawdown_stop: float = DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG.max_drawdown_stop
    max_position_risk_contribution: float = (
        DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG.max_position_risk_contribution
    )
    volatility_adjustment_strength: float = (
        DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG.adjustment.volatility_scale_strength
    )
    concentration_adjustment_strength: float = (
        DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG.adjustment.concentration_scale_strength
    )
    risk_contribution_adjustment_strength: float = (
        DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG.adjustment.risk_contribution_scale_strength
    )
    portfolio_risk_constraint_config: PortfolioRiskConstraintConfig = (
        DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG
    )
    sec_frame_period: str = "CY2024Q4I"
    sec_filing_event_cache_path: str | None = str(DEFAULT_FILING_EVENT_CACHE_PATH)
    validity_gate_ablation_modes: tuple[str, ...] = DEFAULT_VALIDITY_GATE_ABLATION_MODE_IDS
    enable_feature_model_ablation: bool = False
    validation_horizons: tuple[int, ...] = DEFAULT_VALIDATION_HORIZONS
    model_comparison_config: ModelComparisonConfig = field(
        default_factory=default_model_comparison_config
    )
    transaction_cost_sensitivity_config: TransactionCostSensitivityConfig = field(
        default_factory=default_transaction_cost_sensitivity_config
    )
    native_tabular_isolation: bool = True
    native_model_timeout_seconds: int = 180
    tabular_num_threads: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "validity_gate_ablation_modes",
            normalize_validity_gate_ablation_mode_ids(self.validity_gate_ablation_modes),
        )
        minimum_gap_periods = _minimum_validation_gap_periods(
            self.prediction_target_column,
            self.required_validation_horizon,
        )
        object.__setattr__(self, "gap_periods", max(int(self.gap_periods), minimum_gap_periods))
        object.__setattr__(
            self, "embargo_periods", max(int(self.embargo_periods), minimum_gap_periods)
        )


@dataclass(frozen=True)
class PipelineResult:
    market_data: pd.DataFrame
    news_features: pd.DataFrame
    sec_features: pd.DataFrame
    features: pd.DataFrame
    predictions: pd.DataFrame
    signals: pd.DataFrame
    validation_summary: pd.DataFrame
    ablation_summary: list[dict[str, object]]
    backtest: BacktestResult
    benchmark_inputs: BenchmarkConstructionInputs | None = None
    benchmark_return_series: pd.DataFrame | None = None
    equal_weight_baseline_return_series: pd.DataFrame | None = None
    equal_weight_baseline_equity_curve: pd.DataFrame | None = None
    equal_weight_baseline_metrics: PerformanceMetrics | None = None
    transaction_cost_sensitivity: TransactionCostSensitivityBatchResult | None = None

    @property
    def baseline_comparison_inputs(self) -> tuple[BaselineComparisonInput, ...]:
        if self.benchmark_inputs is None:
            return ()
        return self.benchmark_inputs.baseline_comparison_inputs


def run_research_pipeline(config: PipelineConfig | None = None) -> PipelineResult:
    config = config or PipelineConfig()
    ticker_universe = TickerUniverse(tuple(config.tickers), config.benchmark_ticker)
    tickers = list(ticker_universe.tickers)
    data_tickers = list(ticker_universe.data_tickers)

    market_data = _load_market_data(config, data_tickers)
    price_features_all = build_price_features(market_data)
    price_features = price_features_all[
        price_features_all["ticker"].isin(set(tickers))
    ].reset_index(drop=True)
    chronos_adapter = Chronos2Adapter(
        model_id=config.chronos_model_id,
        device_map=config.local_model_device_map,
        local_files_only=True,
    )
    price_features = chronos_adapter.add_forecasts(
        price_features,
        mode=_resolve_timeseries_mode(config.time_series_inference_mode, chronos_adapter),
        target_column=config.time_series_target_column,
        min_context=config.time_series_min_context,
        max_inference_windows=config.max_time_series_inference_windows,
    )
    granite_adapter = GraniteTTMAdapter(
        model_id=config.granite_ttm_model_id,
        revision=config.granite_ttm_revision,
    )
    price_features = granite_adapter.add_forecasts(
        price_features,
        mode=_resolve_timeseries_mode(config.time_series_inference_mode, granite_adapter),
        target_column=config.time_series_target_column,
        min_context=config.time_series_min_context,
        max_inference_windows=config.max_time_series_inference_windows,
    )

    news_items = _load_news_items(config, tickers)
    news_features = build_news_features(
        news_items, _sentiment_analyzer(config.sentiment_model, config.local_model_device_map)
    )

    filings_by_ticker, facts_by_ticker = _load_sec_data(config, tickers)
    filing_extractor = _filing_extractor(config)
    sec_features = build_sec_features(
        filings_by_ticker,
        facts_by_ticker,
        price_features,
        filing_extractor=filing_extractor,
        filing_cache_path=(
            config.sec_filing_event_cache_path
            if config.enable_local_filing_llm
            and bool(getattr(filing_extractor, "use_local_model", False))
            else None
        ),
    )

    features = fuse_features(price_features, news_features, sec_features)
    target_column = _available_prediction_target_column(features, config.prediction_target_column)
    features = features.dropna(subset=[target_column]).reset_index(drop=True)

    walk_config = _walk_forward_config(config)
    splitter = _walk_forward_splitter(config, target_column=target_column)
    predictions, validation_summary = _predict_walk_forward_with_splitter(
        features,
        walk_config,
        target=target_column,
        splitter=splitter,
    )
    predictions = _attach_signal_features(predictions, features)
    predictions = _ensure_backtest_prediction_columns(predictions, target_column)
    evaluation_dates = (
        predictions["date"] if not predictions.empty and "date" in predictions else []
    )
    benchmark_return_series = build_benchmark_return_series(
        price_features_all,
        evaluation_dates,
        benchmark_ticker=config.benchmark_ticker,
        return_column=target_column,
    )
    equal_weight_baseline_return_series = build_equal_weight_baseline_return_series(
        price_features_all,
        evaluation_dates,
        _evaluated_ticker_universe(predictions, tickers),
        return_column=target_column,
    )
    backtest = run_long_only_backtest(
        predictions,
        _backtest_config(config, realized_return_column=target_column),
        benchmark_returns=benchmark_return_series,
    )
    equal_weight_baseline_equity_curve = build_equal_weight_baseline_equity_curve(
        equal_weight_baseline_return_series,
        benchmark_return_series=benchmark_return_series,
        cost_bps=config.cost_bps,
        slippage_bps=config.slippage_bps,
    )
    equal_weight_baseline_metrics = calculate_equal_weight_baseline_performance_metrics(
        equal_weight_baseline_return_series,
        benchmark_return_series=benchmark_return_series,
        cost_bps=config.cost_bps,
        slippage_bps=config.slippage_bps,
    )
    benchmark_inputs = build_benchmark_construction_inputs(
        config,
        evaluation_frame=_benchmark_evaluation_frame(backtest, predictions),
    )
    ablation_summary = _run_ablation_summary(
        predictions,
        features,
        config,
        benchmark_return_series=benchmark_return_series,
    )
    if config.enable_feature_model_ablation:
        ablation_summary.extend(
            _run_optional_model_feature_ablation_summary(
                features,
                config,
                benchmark_return_series=benchmark_return_series,
                exclude_scenario_ids={str(row.get("scenario")) for row in ablation_summary},
            )
        )
    transaction_cost_sensitivity = run_transaction_cost_sensitivity_batch(
        backtest.equity_curve,
        sensitivity_config=config.transaction_cost_sensitivity_config,
    )

    return PipelineResult(
        market_data=market_data,
        news_features=news_features,
        sec_features=sec_features,
        features=features,
        predictions=predictions,
        signals=backtest.signals,
        validation_summary=validation_summary,
        ablation_summary=ablation_summary,
        backtest=backtest,
        benchmark_inputs=benchmark_inputs,
        benchmark_return_series=benchmark_return_series,
        equal_weight_baseline_return_series=equal_weight_baseline_return_series,
        equal_weight_baseline_equity_curve=equal_weight_baseline_equity_curve,
        equal_weight_baseline_metrics=equal_weight_baseline_metrics,
        transaction_cost_sensitivity=transaction_cost_sensitivity,
    )


def _load_market_data(config: PipelineConfig, tickers: list[str]) -> pd.DataFrame:
    if config.data_mode == "local":
        import os

        path = os.path.join(config.local_data_dir, "market_history.parquet")
        provider = LocalMarketDataProvider(data_path=path)
        return provider.get_history(tickers, config.start, config.end, config.interval)
    if config.data_mode == "live":
        provider = YFinanceMarketDataProvider()
        live = provider.get_history(tickers, config.start, config.end, config.interval)
        if not live.empty:
            return live
    return SyntheticMarketDataProvider(periods=260).get_history(
        tickers, config.start, config.end, config.interval
    )


def _load_news_items(config: PipelineConfig, tickers: list[str]):
    if config.data_mode == "local":
        import os

        path = os.path.join(config.local_data_dir, "news_items.jsonl")
        return LocalNewsProvider(data_path=path).get_news(tickers, config.start, config.end)
    if config.data_mode == "live":
        items = []
        for provider in (YFinanceNewsProvider(), GDELTNewsProvider()):
            try:
                items.extend(provider.get_news(tickers, config.start, config.end))
            except Exception:
                continue
        if items:
            return items
    return SyntheticNewsProvider().get_news(tickers, config.start, config.end)


def _load_sec_data(
    config: PipelineConfig, tickers: list[str]
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    if config.data_mode == "local":
        import os

        sec_cache_dir = os.path.join(config.local_data_dir, "sec")
        client = SecEdgarClient(cache_dir=sec_cache_dir)  # type: ignore[arg-type]
        filings_by_ticker: dict[str, pd.DataFrame] = {}
        facts_by_ticker: dict[str, pd.DataFrame] = {}
        frame_assets = _load_sec_frame_assets(client, config)
        for ticker in tickers:
            cik = CIK_BY_TICKER.get(ticker)
            if cik is None:
                continue
            try:
                filings_by_ticker[ticker] = client.recent_filings(
                    cik,
                    {"8-K", "10-Q", "10-K", "4"},
                    include_document_text=True,
                )
                facts_frame = extract_companyfacts_frame(client.get_companyfacts(cik))
                concept_frame = extract_companyconcept_frame(
                    client.get_companyconcept(cik, "us-gaap", "NetIncomeLoss"),
                    "net_income",
                )
                merged_facts = merge_fact_frames(facts_frame, concept_frame)
                cik10 = "".join(char for char in str(cik) if char.isdigit()).zfill(10)
                if cik10 in frame_assets:
                    merged_facts["sec_frame_assets"] = frame_assets[cik10]
                facts_by_ticker[ticker] = merged_facts
            except Exception:
                continue
        return _neutral_sec_mappings_for_missing_tickers(
            tickers, filings_by_ticker, facts_by_ticker
        )
    if config.data_mode == "live":
        client = SecEdgarClient()
        filings_by_ticker: dict[str, pd.DataFrame] = {}
        facts_by_ticker: dict[str, pd.DataFrame] = {}
        frame_assets = _load_sec_frame_assets(client, config)
        for ticker in tickers:
            cik = CIK_BY_TICKER.get(ticker)
            if cik is None:
                continue
            try:
                filings_by_ticker[ticker] = client.recent_filings(
                    cik,
                    {"8-K", "10-Q", "10-K", "4"},
                    include_document_text=True,
                )
                facts_frame = extract_companyfacts_frame(client.get_companyfacts(cik))
                concept_frame = extract_companyconcept_frame(
                    client.get_companyconcept(cik, "us-gaap", "NetIncomeLoss"),
                    "net_income",
                )
                merged_facts = merge_fact_frames(facts_frame, concept_frame)
                cik10 = "".join(char for char in str(cik) if char.isdigit()).zfill(10)
                if cik10 in frame_assets:
                    merged_facts["sec_frame_assets"] = frame_assets[cik10]
                facts_by_ticker[ticker] = merged_facts
            except Exception:
                continue
        return _neutral_sec_mappings_for_missing_tickers(
            tickers, filings_by_ticker, facts_by_ticker
        )

    provider = SyntheticSecProvider()
    filings = {
        ticker: provider.recent_filings(
            CIK_BY_TICKER.get(ticker, "0"),
            include_document_text=True,
        )
        for ticker in tickers
    }
    facts = {
        ticker: provider.companyfacts_frame(CIK_BY_TICKER.get(ticker, "0")) for ticker in tickers
    }
    return filings, facts


def _neutral_sec_mappings_for_missing_tickers(
    tickers: list[str],
    filings_by_ticker: dict[str, pd.DataFrame],
    facts_by_ticker: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    filings = dict(filings_by_ticker)
    facts = dict(facts_by_ticker)
    for ticker in tickers:
        filings.setdefault(ticker, pd.DataFrame())
        facts.setdefault(ticker, pd.DataFrame())
    return filings, facts


def _load_sec_frame_assets(client: SecEdgarClient, config: PipelineConfig) -> dict[str, float]:
    try:
        frame = extract_frame_values(
            client.get_frame("us-gaap", "Assets", "USD", config.sec_frame_period),
            "sec_frame_assets",
        )
    except Exception:
        return {}
    return dict(zip(frame["cik"], frame["sec_frame_assets"], strict=False))


def _attach_signal_features(predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    if features.empty or not {"date", "ticker"}.issubset(features.columns):
        enriched = predictions.copy()
    else:
        merge_columns = [
            column
            for column in features.columns
            if column in {"date", "ticker"} or column not in predictions.columns
        ]
        enriched = predictions.merge(
            features[merge_columns],
            on=["date", "ticker"],
            how="left",
        )

    defaults_numeric = {
        "text_risk_score": 0.0,
        "sec_risk_flag": 0.0,
        "sec_risk_flag_20d": 0.0,
        "news_negative_ratio": 0.0,
        "liquidity_score": 0.0,
        "sec_event_confidence": 0.0,
    }
    for column, default_value in defaults_numeric.items():
        if column in enriched:
            enriched[column] = pd.to_numeric(enriched[column], errors="coerce").fillna(
                default_value
            )
        else:
            enriched[column] = default_value

    defaults_text = {
        "news_top_event": "none",
        "sec_event_tag": "none",
        "sec_summary_ref": "",
    }
    for column, default_value in defaults_text.items():
        if column in enriched:
            merged_series = enriched[column].fillna(default_value).astype(str)
            merged_series = merged_series.where(~merged_series.isin({"nan", "None"}), default_value)
            if default_value == "none":
                merged_series = merged_series.where(merged_series.str.len() > 0, default_value)
            enriched[column] = merged_series
        else:
            enriched[column] = default_value

    return enriched


def _sentiment_analyzer(model_name: str, device_map: str = "auto"):
    if model_name.lower() == "finbert":
        # device_map="cpu" when MLX is active to avoid PyTorch MPS + Metal conflict
        device = "cpu" if device_map == "cpu" else None
        analyzer = FinBERTSentimentAnalyzer(device=device, local_files_only=True)
        try:
            if analyzer.available():
                return analyzer
        except Exception:
            return KeywordSentimentAnalyzer()
        return KeywordSentimentAnalyzer()
    return KeywordSentimentAnalyzer()


def _resolve_timeseries_mode(requested_mode: str, adapter: object) -> str:
    if requested_mode != "local":
        return requested_mode
    if hasattr(adapter, "available"):
        try:
            if adapter.available():
                return requested_mode
        except Exception:
            return "proxy"
        return "proxy"
    return "proxy"


def _filing_extractor(config: PipelineConfig):
    model_name = config.filing_extractor_model.lower()
    if model_name == "finma":
        extractor = FilingEventExtractor(
            model_id=config.finma_model_id,
            use_local_model=config.enable_local_filing_llm,
            local_files_only=True,
            device_map=config.local_model_device_map,
            offload_folder=config.local_model_offload_folder,
        )
        try:
            is_available = getattr(extractor, "available", None)
            if is_available is None:
                return extractor
            return extractor if is_available() else FilingEventExtractor(use_local_model=False)
        except Exception:
            return FilingEventExtractor(use_local_model=False)
    if model_name == "fingpt":
        # Keep compatibility with both the current and planned FinGPT extractor constructor
        # fields by only passing arguments that are accepted.
        runtime_args = {
            "runtime": config.fingpt_runtime,
            "runtime_model_path": config.fingpt_quantized_model_path,
            "quantized_model_path": config.fingpt_quantized_model_path,
            "allow_unquantized_transformers": config.fingpt_allow_unquantized_transformers,
            "allow_unquantized_fingpt": config.fingpt_allow_unquantized_transformers,
            "single_load_lock_path": config.fingpt_single_load_lock_path,
        }
        init_signature = signature(FinGPTEventExtractor)
        signature_allows_var_kwargs = any(
            param.kind == Parameter.VAR_KEYWORD for param in init_signature.parameters.values()
        )
        if signature_allows_var_kwargs:
            runtime_kwargs = runtime_args
        else:
            accepted_keys = set(init_signature.parameters.keys())
            runtime_kwargs = {
                key: value for key, value in runtime_args.items() if key in accepted_keys
            }
        extractor = FinGPTEventExtractor(
            model_id=config.fingpt_model_id,
            use_local_model=config.enable_local_filing_llm,
            device_map=config.local_model_device_map,
            local_files_only=True,
            base_model_id=config.fingpt_base_model_id,
            offload_folder=config.local_model_offload_folder,
            **runtime_kwargs,
        )
        try:
            is_available = getattr(extractor, "available", None)
            if is_available is None:
                return extractor
            return extractor if is_available() else FilingEventExtractor(use_local_model=False)
        except Exception:
            return FilingEventExtractor(use_local_model=False)
    return FilingEventExtractor(use_local_model=False)


def _run_ablation_summary(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    config: PipelineConfig,
    *,
    benchmark_return_series: pd.DataFrame | None = None,
) -> list[dict[str, object]]:
    if predictions.empty:
        return []
    target_column = _available_prediction_target_column(
        features,
        _available_prediction_target_column(predictions, config.prediction_target_column),
    )
    original_evaluation_dates = _evaluation_date_index(predictions)

    scenarios: list[dict[str, object]] = []
    registry = default_ablation_registry()
    selected_by_kind = _selected_validity_gate_ablation_scenarios_by_kind(config, registry)
    for scenario in selected_by_kind["signal"]:
        backtest_config = _backtest_config_for_ablation(
            config, scenario, target_column=target_column
        )
        result = run_long_only_backtest(
            _apply_signal_ablation_toggles(predictions, scenario.toggles),
            backtest_config,
            benchmark_returns=benchmark_return_series,
        )
        scenarios.append(_ablation_metrics_row(scenario, result, backtest_config))

    for scenario in selected_by_kind["cost"]:
        backtest_config = _backtest_config_for_ablation(
            config, scenario, target_column=target_column
        )
        result = run_long_only_backtest(
            _apply_signal_ablation_toggles(predictions, scenario.toggles),
            backtest_config,
            benchmark_returns=benchmark_return_series,
        )
        scenarios.append(_ablation_metrics_row(scenario, result, backtest_config))

    for scenario in selected_by_kind["pipeline_control"]:
        backtest_config = _backtest_config_for_ablation(
            config, scenario, target_column=target_column
        )
        scenario_validation_summary = pd.DataFrame()
        if _requires_model_refit(scenario.toggles):
            variant = _apply_model_feature_ablation_scenario(features, scenario)
            variant = variant.dropna(subset=[target_column]).reset_index(drop=True)
            model_input_variant = _model_input_features_for_ablation(variant, scenario)
            scenario_predictions, scenario_validation_summary = _predict_walk_forward_with_splitter(
                model_input_variant,
                _walk_forward_config(config),
                target=target_column,
                splitter=_walk_forward_splitter(config, target_column=target_column),
            )
            scenario_predictions = _attach_signal_features(scenario_predictions, variant)
        else:
            scenario_predictions = _apply_signal_ablation_toggles(predictions, scenario.toggles)
        scenario_predictions = _align_predictions_to_evaluation_dates(
            scenario_predictions,
            original_evaluation_dates,
        )
        result = run_long_only_backtest(
            _ensure_backtest_prediction_columns(scenario_predictions, target_column),
            backtest_config,
            benchmark_returns=benchmark_return_series,
        )
        row = _ablation_metrics_row(
            scenario,
            result,
            backtest_config,
            input_features=(
                model_input_variant if _requires_model_refit(scenario.toggles) else None
            ),
            target_column=target_column,
        )
        if _requires_model_refit(scenario.toggles):
            row.update(
                _ablation_validation_metrics(
                    scenario_validation_summary,
                    scenario_predictions,
                    target_column=target_column,
                )
            )
        scenarios.append(row)

    scenarios.extend(
        _run_model_refit_ablation_summary(
            selected_by_kind["data_channel"] + selected_by_kind["model_feature"],
            features,
            config,
            target_column=target_column,
            benchmark_return_series=benchmark_return_series,
            evaluation_dates=original_evaluation_dates,
        )
    )

    return scenarios


def _selected_validity_gate_ablation_scenarios_by_kind(
    config: PipelineConfig,
    registry: AblationScenarioRegistry,
) -> dict[str, tuple[AblationScenarioConfig, ...]]:
    selected_ids = set(config.validity_gate_ablation_modes)
    return {
        kind: tuple(
            scenario for scenario in registry.by_kind(kind) if scenario.scenario_id in selected_ids
        )
        for kind in ("signal", "cost", "pipeline_control", "data_channel", "model_feature")
    }


def _run_feature_model_ablation_summary(
    features: pd.DataFrame,
    config: PipelineConfig,
    *,
    benchmark_return_series: pd.DataFrame | None = None,
) -> list[dict[str, object]]:
    registry = default_ablation_registry()
    return _run_model_refit_ablation_summary(
        registry.by_kind("data_channel") + registry.by_kind("model_feature"),
        features,
        config,
        benchmark_return_series=benchmark_return_series,
    )


def _run_optional_model_feature_ablation_summary(
    features: pd.DataFrame,
    config: PipelineConfig,
    *,
    benchmark_return_series: pd.DataFrame | None = None,
    exclude_scenario_ids: set[str] | None = None,
) -> list[dict[str, object]]:
    registry = default_ablation_registry()
    excluded = exclude_scenario_ids or set()
    return _run_model_refit_ablation_summary(
        tuple(
            scenario
            for scenario in registry.by_kind("model_feature")
            if scenario.scenario_id not in excluded
        ),
        features,
        config,
        benchmark_return_series=benchmark_return_series,
    )


def _run_model_refit_ablation_summary(
    scenarios_to_run: tuple[AblationScenarioConfig, ...],
    features: pd.DataFrame,
    config: PipelineConfig,
    *,
    target_column: str | None = None,
    benchmark_return_series: pd.DataFrame | None = None,
    evaluation_dates: pd.Index | None = None,
) -> list[dict[str, object]]:
    scenarios: list[dict[str, object]] = []
    target_column = target_column or _available_prediction_target_column(
        features, config.prediction_target_column
    )
    backtest_config = _backtest_config(config, realized_return_column=target_column)
    for scenario in scenarios_to_run:
        variant = _apply_model_feature_ablation_scenario(features, scenario)
        variant = variant.dropna(subset=[target_column]).reset_index(drop=True)
        model_input_variant = _model_input_features_for_ablation(variant, scenario)
        predictions, validation_summary = _predict_walk_forward_with_splitter(
            model_input_variant,
            _walk_forward_config(config),
            target=target_column,
            splitter=_walk_forward_splitter(config, target_column=target_column),
        )
        predictions = _attach_signal_features(predictions, variant)
        predictions = _align_predictions_to_evaluation_dates(predictions, evaluation_dates)
        result = run_long_only_backtest(
            _ensure_backtest_prediction_columns(predictions, target_column),
            backtest_config,
            benchmark_returns=benchmark_return_series,
        )
        row = _ablation_metrics_row(
            scenario,
            result,
            backtest_config,
            input_features=model_input_variant,
            target_column=target_column,
        )
        row.update(
            _ablation_validation_metrics(
                validation_summary,
                predictions,
                target_column=target_column,
            )
        )
        scenarios.append(row)
    return scenarios


def _apply_signal_ablation_toggles(
    predictions: pd.DataFrame,
    toggles: AblationToggles,
) -> pd.DataFrame:
    frame = predictions
    if not toggles.include_text_features or not toggles.include_text_risk:
        text_assignments: dict[str, object] = {
            "text_risk_score": 0.0,
            "news_negative_ratio": 0.0,
        }
        if not toggles.include_text_features:
            text_assignments["news_top_event"] = "none"
        frame = frame.assign(**text_assignments)
    if not toggles.include_sec_features or not toggles.include_sec_risk:
        sec_assignments: dict[str, object] = {
            "sec_risk_flag": 0.0,
            "sec_risk_flag_20d": 0.0,
        }
        if not toggles.include_sec_features:
            sec_assignments.update(
                {
                    "sec_event_tag": "none",
                    "sec_event_confidence": 0.0,
                    "sec_summary_ref": "",
                }
            )
        frame = frame.assign(**sec_assignments)
    return frame


def _apply_model_feature_ablation_toggles(
    features: pd.DataFrame,
    toggles: AblationToggles,
) -> pd.DataFrame:
    frame = _apply_signal_ablation_toggles(features, toggles)
    drop_columns: set[str] = set()
    if not toggles.include_price_features:
        drop_columns.update(_price_source_feature_columns(frame))
    if not toggles.include_text_features:
        drop_columns.update(_text_source_feature_columns(frame))
    if not toggles.include_sec_features:
        drop_columns.update(_sec_source_feature_columns(frame))
    if not toggles.proxy_features_enabled():
        drop_columns.update(_model_proxy_feature_columns(frame))
    if (
        not toggles.include_price_features
        or not toggles.proxy_features_enabled()
        or not toggles.include_chronos_features
    ):
        drop_columns.update(column for column in frame.columns if column.startswith("chronos_"))
    if (
        not toggles.include_price_features
        or not toggles.proxy_features_enabled()
        or not toggles.include_granite_ttm_features
    ):
        drop_columns.update(column for column in frame.columns if column.startswith("granite_ttm_"))
    return frame.drop(columns=sorted(drop_columns), errors="ignore")


def _apply_model_feature_ablation_scenario(
    features: pd.DataFrame,
    scenario: AblationScenarioConfig,
) -> pd.DataFrame:
    frame = _apply_model_feature_ablation_toggles(features, scenario.toggles)
    allowed_families = set(scenario.permitted_feature_families)
    drop_columns = [
        column
        for column in frame.columns
        if (family := feature_family_for_column(column)) is not None
        and family not in allowed_families
    ]
    filtered = frame.drop(columns=drop_columns, errors="ignore")
    _validate_model_feature_ablation_families(filtered, scenario)
    return filtered


def _model_input_features_for_ablation(
    features: pd.DataFrame,
    scenario: AblationScenarioConfig,
) -> pd.DataFrame:
    if scenario.toggles.proxy_model_inputs_enabled():
        return features
    return features.drop(
        columns=sorted(_model_proxy_feature_columns(features)),
        errors="ignore",
    )


def _validate_model_feature_ablation_families(
    features: pd.DataFrame,
    scenario: AblationScenarioConfig,
) -> None:
    allowed_families = set(scenario.permitted_feature_families)
    disallowed: dict[FeatureFamily, list[str]] = {}
    for column in features.columns:
        family = feature_family_for_column(column)
        if family is not None and family not in allowed_families:
            disallowed.setdefault(family, []).append(str(column))
    if disallowed:
        details = ", ".join(f"{family}={columns}" for family, columns in sorted(disallowed.items()))
        raise ValueError(
            f"ablation scenario {scenario.scenario_id} received disallowed feature families: {details}"
        )


def _price_source_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if not _is_identifier_or_label_column(column)
        and not _is_text_source_feature_column(column)
        and not _is_sec_source_feature_column(column)
    ]


def _text_source_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if not _is_identifier_or_label_column(column) and _is_text_source_feature_column(column)
    ]


def _sec_source_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if not _is_identifier_or_label_column(column) and _is_sec_source_feature_column(column)
    ]


def _model_proxy_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if str(column).startswith(("proxy_", "chronos_", "granite_ttm_"))
    ]


def _is_identifier_or_label_column(column: str) -> bool:
    return column in {"date", "ticker"} or column.startswith("forward_return_")


def _is_text_source_feature_column(column: str) -> bool:
    return column.startswith(("news_", "text_"))


def _is_sec_source_feature_column(column: str) -> bool:
    return column.startswith(("sec_", "revenue_", "net_income_", "assets_"))


def _backtest_config_for_ablation(
    config: PipelineConfig,
    scenario: AblationScenarioConfig | AblationToggles,
    *,
    target_column: str | None = None,
) -> BacktestConfig:
    scenario_id = scenario.scenario_id if isinstance(scenario, AblationScenarioConfig) else None
    no_cost_scenario_active = scenario_id == NO_COST_ABLATION_SCENARIO
    return _backtest_config(
        config,
        cost_bps=0.0 if no_cost_scenario_active else config.cost_bps,
        slippage_bps=0.0 if no_cost_scenario_active else config.slippage_bps,
        realized_return_column=target_column,
    )


def _evaluation_date_index(predictions: pd.DataFrame) -> pd.Index:
    if predictions.empty or "date" not in predictions:
        return pd.Index([], dtype="datetime64[ns]")
    dates = pd.to_datetime(predictions["date"], errors="coerce").dt.normalize().dropna()
    return pd.Index(dates.drop_duplicates().sort_values())


def _align_predictions_to_evaluation_dates(
    predictions: pd.DataFrame,
    evaluation_dates: pd.Index | None,
) -> pd.DataFrame:
    if (
        evaluation_dates is None
        or len(evaluation_dates) == 0
        or predictions.empty
        or "date" not in predictions
    ):
        return predictions
    frame = predictions.copy()
    normalized_dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    return frame.loc[normalized_dates.isin(evaluation_dates)].reset_index(drop=True)


def _backtest_control_payload(backtest_config: BacktestConfig) -> dict[str, object]:
    return {
        "top_n": backtest_config.top_n,
        "cost_bps": backtest_config.cost_bps,
        "slippage_bps": backtest_config.slippage_bps,
        "average_daily_turnover_budget": backtest_config.average_daily_turnover_budget,
        "benchmark_ticker": backtest_config.benchmark_ticker,
        "realized_return_column": backtest_config.realized_return_column,
        "max_symbol_weight": backtest_config.max_symbol_weight,
        "max_sector_weight": backtest_config.max_sector_weight,
        "max_correlation_cluster_weight": backtest_config.max_correlation_cluster_weight,
        "correlation_cluster_threshold": backtest_config.correlation_cluster_threshold,
        "portfolio_covariance_lookback": backtest_config.portfolio_covariance_lookback,
        "covariance_aware_risk_enabled": backtest_config.covariance_aware_risk_enabled,
        "covariance_return_column": backtest_config.covariance_return_column,
        "covariance_min_periods": backtest_config.covariance_min_periods,
        "max_daily_turnover": backtest_config.max_daily_turnover,
        "portfolio_volatility_limit": backtest_config.portfolio_volatility_limit,
        "max_drawdown_stop": backtest_config.max_drawdown_stop,
        "max_position_risk_contribution": backtest_config.max_position_risk_contribution,
        "volatility_adjustment_strength": backtest_config.volatility_adjustment_strength,
        "concentration_adjustment_strength": (
            backtest_config.concentration_adjustment_strength
        ),
        "risk_contribution_adjustment_strength": (
            backtest_config.risk_contribution_adjustment_strength
        ),
        "portfolio_risk_constraints": {
            "schema_version": backtest_config.risk_constraint_schema_version,
            "max_holdings": backtest_config.top_n,
            "max_symbol_weight": backtest_config.max_symbol_weight,
            "max_sector_weight": backtest_config.max_sector_weight,
            "max_position_risk_contribution": (
                backtest_config.max_position_risk_contribution
            ),
            "portfolio_volatility_limit": backtest_config.portfolio_volatility_limit,
            "portfolio_covariance_lookback": backtest_config.portfolio_covariance_lookback,
            "covariance_aware_risk": {
                "enabled": backtest_config.covariance_aware_risk_enabled,
                "return_column": backtest_config.covariance_return_column,
                "lookback_periods": backtest_config.portfolio_covariance_lookback,
                "min_periods": backtest_config.covariance_min_periods,
                "fallback": "diagonal_predicted_volatility",
            },
            "max_drawdown_stop": backtest_config.max_drawdown_stop,
            "adjustment": {
                "volatility_scale_strength": (
                    backtest_config.volatility_adjustment_strength
                ),
                "concentration_scale_strength": (
                    backtest_config.concentration_adjustment_strength
                ),
                "risk_contribution_scale_strength": (
                    backtest_config.risk_contribution_adjustment_strength
                ),
            },
            "v1_exclusions": ["correlation_cluster_weight"],
        },
    }


def _ablation_metrics_row(
    scenario: AblationScenarioConfig,
    result: BacktestResult,
    backtest_config: BacktestConfig,
    *,
    input_features: pd.DataFrame | None = None,
    target_column: str | None = None,
) -> dict[str, object]:
    signal_evaluation_metrics = _deterministic_signal_evaluation_metrics(result, backtest_config)
    position_level_metrics = _position_level_net_return_metrics(result.weights)
    evaluation_start, evaluation_end = _evaluation_date_bounds(result.equity_curve)
    row: dict[str, object] = {
        "scenario": scenario.scenario_id,
        "scenario_id": scenario.scenario_id,
        "kind": scenario.kind,
        "label": scenario.label,
        "description": scenario.description,
        "toggles": scenario.toggles.to_dict(),
        "feature_sources": scenario.toggles.feature_source_toggles(),
        "feature_family_allowlist": list(scenario.feature_family_allowlist),
        "permitted_feature_families": list(scenario.permitted_feature_families),
        "pipeline_controls": scenario.toggles.pipeline_control_toggles(),
        "proxy_removal_options": scenario.toggles.proxy_removal_options(),
        "effective_cost_bps": backtest_config.cost_bps,
        "effective_slippage_bps": backtest_config.slippage_bps,
        "evaluation_start": evaluation_start,
        "evaluation_end": evaluation_end,
        "evaluation_observation_count": int(len(result.equity_curve)),
        "backtest_controls": _backtest_control_payload(backtest_config),
        "inherited_backtest_controls": _backtest_control_payload(backtest_config),
        "cagr": result.metrics.cagr,
        "sharpe": result.metrics.sharpe,
        "max_drawdown": result.metrics.max_drawdown,
        "turnover": result.metrics.turnover,
        "excess_return": result.metrics.excess_return,
        "position_level_metrics": position_level_metrics,
        "signal_evaluation_metrics": signal_evaluation_metrics,
        "deterministic_signal_evaluation_metrics": signal_evaluation_metrics,
        **_ablation_validation_metrics(
            pd.DataFrame(),
            result.signals,
            target_column=target_column or backtest_config.realized_return_column,
        ),
        **_flatten_signal_evaluation_metrics(signal_evaluation_metrics),
    }
    if input_features is not None and target_column is not None:
        input_feature_columns = infer_feature_columns(input_features, target_column)
        input_family_columns = feature_family_columns(input_feature_columns)
        row["input_feature_columns"] = input_feature_columns
        row["input_feature_families"] = [
            family
            for family in scenario.permitted_feature_families
            if family in input_family_columns
        ]
    return row


def _deterministic_signal_evaluation_metrics(
    result: BacktestResult,
    backtest_config: BacktestConfig,
) -> dict[str, object]:
    equity_curve = result.equity_curve
    signals = result.signals
    net_returns = _numeric_backtest_column(
        equity_curve,
        "cost_adjusted_return",
        "portfolio_return",
        "net_return",
    )
    gross_returns = _numeric_backtest_column(
        equity_curve,
        "gross_return",
        "deterministic_strategy_return",
    )
    transaction_cost = _numeric_backtest_column(equity_curve, "transaction_cost_return")
    slippage_cost = _numeric_backtest_column(equity_curve, "slippage_cost_return")
    total_cost = _numeric_backtest_column(equity_curve, "total_cost_return", "turnover_cost_return")
    if total_cost.empty:
        total_cost = transaction_cost.add(slippage_cost, fill_value=0.0)
    if gross_returns.empty:
        gross_returns = net_returns.add(total_cost, fill_value=0.0)

    turnover = _numeric_backtest_column(equity_curve, "turnover")
    exposure = _numeric_backtest_column(equity_curve, "exposure")
    portfolio_volatility = _numeric_backtest_column(equity_curve, "portfolio_volatility_estimate")
    risk_stop_active = _boolean_backtest_column(equity_curve, "risk_stop_active")
    action_counts = _signal_action_counts(signals)
    signal_observations = int(len(signals))
    evaluation_observations = int(len(equity_curve))
    evaluation_start, evaluation_end = _evaluation_date_bounds(equity_curve)
    risk_stop_count = int(risk_stop_active.sum()) if not risk_stop_active.empty else 0

    return {
        "engine": "deterministic_signal_engine",
        "return_basis": "cost_adjusted_return",
        "realized_return_column": backtest_config.realized_return_column,
        "effective_cost_bps": float(backtest_config.cost_bps),
        "effective_slippage_bps": float(backtest_config.slippage_bps),
        "signal_observations": signal_observations,
        "evaluation_observations": evaluation_observations,
        "evaluation_start": evaluation_start,
        "evaluation_end": evaluation_end,
        "action_counts": action_counts,
        "action_ratios": _signal_action_ratios(action_counts, signal_observations),
        "cagr": float(result.metrics.cagr),
        "annualized_volatility": float(result.metrics.annualized_volatility),
        "sharpe": float(result.metrics.sharpe),
        "max_drawdown": float(result.metrics.max_drawdown),
        "hit_rate": float(result.metrics.hit_rate),
        "average_daily_turnover": float(result.metrics.turnover),
        "max_daily_turnover": _finite_max(turnover),
        "exposure": float(result.metrics.exposure),
        "average_exposure": _finite_mean(exposure),
        "benchmark_cagr": float(result.metrics.benchmark_cagr),
        "excess_return": float(result.metrics.excess_return),
        "gross_cumulative_return": _compound_returns(gross_returns),
        "cost_adjusted_cumulative_return": _compound_returns(net_returns),
        "transaction_cost_return": float(transaction_cost.sum())
        if not transaction_cost.empty
        else 0.0,
        "slippage_cost_return": float(slippage_cost.sum()) if not slippage_cost.empty else 0.0,
        "total_cost_return": float(total_cost.sum()) if not total_cost.empty else 0.0,
        "risk_stop_observation_count": risk_stop_count,
        "risk_stop_observation_ratio": (
            risk_stop_count / evaluation_observations if evaluation_observations else 0.0
        ),
        "max_portfolio_volatility_estimate": _finite_max(portfolio_volatility),
    }


def _flatten_signal_evaluation_metrics(metrics: dict[str, object]) -> dict[str, object]:
    action_counts = metrics.get("action_counts")
    if not isinstance(action_counts, dict):
        action_counts = {}
    action_ratios = metrics.get("action_ratios")
    if not isinstance(action_ratios, dict):
        action_ratios = {}
    return {
        "signal_engine": metrics.get("engine"),
        "signal_return_basis": metrics.get("return_basis"),
        "signal_realized_return_column": metrics.get("realized_return_column"),
        "signal_observation_count": metrics.get("signal_observations"),
        "signal_evaluation_observation_count": metrics.get("evaluation_observations"),
        "signal_evaluation_start": metrics.get("evaluation_start"),
        "signal_evaluation_end": metrics.get("evaluation_end"),
        "signal_buy_count": action_counts.get("BUY", 0),
        "signal_sell_count": action_counts.get("SELL", 0),
        "signal_hold_count": action_counts.get("HOLD", 0),
        "signal_buy_ratio": action_ratios.get("BUY", 0.0),
        "signal_sell_ratio": action_ratios.get("SELL", 0.0),
        "signal_hold_ratio": action_ratios.get("HOLD", 0.0),
        "signal_hit_rate": metrics.get("hit_rate"),
        "signal_exposure": metrics.get("exposure"),
        "signal_average_daily_turnover": metrics.get("average_daily_turnover"),
        "signal_max_daily_turnover": metrics.get("max_daily_turnover"),
        "signal_gross_cumulative_return": metrics.get("gross_cumulative_return"),
        "signal_cost_adjusted_cumulative_return": metrics.get("cost_adjusted_cumulative_return"),
        "signal_transaction_cost_return": metrics.get("transaction_cost_return"),
        "signal_slippage_cost_return": metrics.get("slippage_cost_return"),
        "signal_total_cost_return": metrics.get("total_cost_return"),
        "signal_risk_stop_observation_count": metrics.get("risk_stop_observation_count"),
        "signal_risk_stop_observation_ratio": metrics.get("risk_stop_observation_ratio"),
        "signal_max_portfolio_volatility_estimate": metrics.get(
            "max_portfolio_volatility_estimate"
        ),
    }


def _position_level_net_return_metrics(positions: pd.DataFrame) -> dict[str, object]:
    if positions.empty:
        return {
            "position_count": 0,
            "position_gross_return_contribution": 0.0,
            "position_net_return_contribution": 0.0,
            "position_total_cost_return": 0.0,
            "position_return_basis": "position_net_return",
        }
    gross = _numeric_backtest_column(positions, "gross_return_contribution")
    net = _numeric_backtest_column(
        positions,
        "net_return_contribution",
        "position_net_return",
    )
    total_cost = _numeric_backtest_column(positions, "total_cost_return")
    if net.empty and not gross.empty:
        net = gross.sub(total_cost.reindex(gross.index, fill_value=0.0), fill_value=0.0)
    return {
        "position_count": int(len(positions)),
        "position_gross_return_contribution": float(gross.sum()) if not gross.empty else 0.0,
        "position_net_return_contribution": float(net.sum()) if not net.empty else 0.0,
        "position_total_cost_return": float(total_cost.sum()) if not total_cost.empty else 0.0,
        "position_return_basis": "position_net_return",
    }


def _signal_action_counts(signals: pd.DataFrame) -> dict[str, int]:
    if signals.empty or "action" not in signals:
        return {"BUY": 0, "SELL": 0, "HOLD": 0}
    actions = signals["action"].fillna("").astype(str).str.upper()
    return {action: int((actions == action).sum()) for action in ("BUY", "SELL", "HOLD")}


def _signal_action_ratios(
    action_counts: dict[str, int], signal_observations: int
) -> dict[str, float]:
    if signal_observations <= 0:
        return {action: 0.0 for action in ("BUY", "SELL", "HOLD")}
    return {
        action: float(action_counts.get(action, 0) / signal_observations)
        for action in ("BUY", "SELL", "HOLD")
    }


def _numeric_backtest_column(frame: pd.DataFrame, *columns: str) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    for column in columns:
        if column in frame:
            return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float)
    return pd.Series(dtype=float)


def _boolean_backtest_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame.empty or column not in frame:
        return pd.Series(dtype=bool)
    return frame[column].fillna(False).astype(bool)


def _compound_returns(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return float((1.0 + returns.fillna(0.0)).prod() - 1.0)


def _finite_max(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    finite = values[np.isfinite(values)]
    if finite.empty:
        return 0.0
    return float(finite.max())


def _evaluation_date_bounds(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    if frame.empty or "date" not in frame:
        return None, None
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if dates.empty:
        return None, None
    return dates.min().date().isoformat(), dates.max().date().isoformat()


def _ablation_validation_metrics(
    validation_summary: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    target_column: str,
) -> dict[str, object]:
    structured_skip = _structured_insufficient_validation_metrics(
        validation_summary,
        predictions,
        target_column=target_column,
    )
    if structured_skip is not None:
        return structured_skip

    if validation_summary.empty:
        return {
            "validation_status": "not_evaluable",
            "validation_fold_count": 0,
            "validation_oos_fold_count": 0,
            "validation_prediction_count": int(len(predictions)),
            "validation_labeled_prediction_count": _labeled_prediction_count(
                predictions, target_column
            ),
            "validation_mean_mae": None,
            "validation_mean_directional_accuracy": None,
            "validation_mean_information_coefficient": None,
            "validation_positive_ic_fold_ratio": None,
            "validation_oos_information_coefficient": None,
        }

    summary = validation_summary.copy()
    ic = _numeric_metric_series(summary.get("information_coefficient"), index=summary.index)
    oos_mask = (
        summary.get("is_oos", pd.Series(False, index=summary.index)).fillna(False).astype(bool)
    )
    return {
        "validation_status": "pass" if bool(oos_mask.any()) else "not_evaluable",
        "validation_fold_count": int(len(summary)),
        "validation_oos_fold_count": int(oos_mask.sum()),
        "validation_prediction_count": int(len(predictions)),
        "validation_labeled_prediction_count": _labeled_prediction_count(
            predictions, target_column
        ),
        "validation_mean_mae": _finite_mean(summary.get("mae")),
        "validation_mean_directional_accuracy": _finite_mean(summary.get("directional_accuracy")),
        "validation_mean_information_coefficient": _finite_mean(ic),
        "validation_positive_ic_fold_ratio": (
            float((ic.dropna() > 0).mean()) if not ic.dropna().empty else None
        ),
        "validation_oos_information_coefficient": _finite_mean(ic[oos_mask]),
    }


def _structured_insufficient_validation_metrics(
    validation_summary: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    target_column: str,
) -> dict[str, object] | None:
    if validation_summary.empty or "validation_status" not in validation_summary:
        return None
    statuses = validation_summary["validation_status"].dropna().astype(str).str.strip().str.lower()
    if statuses.empty or not statuses.isin({"insufficient_data", "skipped"}).any():
        return None

    row = validation_summary.iloc[0].to_dict()
    fold_count = int(_finite_number(row.get("fold_count")) or 0)
    oos_count = 0
    if "is_oos" in validation_summary:
        oos_count = int(validation_summary["is_oos"].fillna(False).sum())
    return {
        "validation_status": "insufficient_data",
        "validation_reason": row.get("reason"),
        "validation_skip_status": row.get("skip_status", "skipped"),
        "validation_skip_code": row.get("skip_code"),
        "validation_fold_count": fold_count,
        "validation_oos_fold_count": oos_count,
        "validation_prediction_count": int(len(predictions)),
        "validation_labeled_prediction_count": _labeled_prediction_count(
            predictions, target_column
        ),
        "validation_labeled_date_count": int(_finite_number(row.get("labeled_date_count")) or 0),
        "validation_required_min_date_count": row.get("required_min_date_count"),
        "validation_candidate_fold_count": int(
            _finite_number(row.get("candidate_fold_count")) or 0
        ),
        "validation_mean_mae": None,
        "validation_mean_directional_accuracy": None,
        "validation_mean_information_coefficient": None,
        "validation_positive_ic_fold_ratio": None,
        "validation_oos_information_coefficient": None,
    }


def _labeled_prediction_count(predictions: pd.DataFrame, target_column: str) -> int:
    if predictions.empty or target_column not in predictions:
        return 0
    return int(pd.to_numeric(predictions[target_column], errors="coerce").notna().sum())


def _finite_number(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _finite_mean(values: object) -> float | None:
    if values is None:
        return None
    series = pd.to_numeric(values, errors="coerce")
    if not isinstance(series, pd.Series):
        series = pd.Series([series])
    finite = series[np.isfinite(series)]
    if finite.empty:
        return None
    return float(finite.mean())


def _numeric_metric_series(values: object, *, index: pd.Index) -> pd.Series:
    if values is None:
        return pd.Series(np.nan, index=index, dtype=float)
    series = pd.to_numeric(values, errors="coerce")
    if isinstance(series, pd.Series):
        return series
    return pd.Series(series, index=index, dtype=float)


def _ensure_backtest_prediction_columns(
    predictions: pd.DataFrame, target_column: str
) -> pd.DataFrame:
    frame = predictions.copy()
    defaults: dict[str, object] = {
        "date": pd.NaT,
        "ticker": "",
        "expected_return": 0.0,
        "predicted_volatility": 0.0,
        "downside_quantile": 0.0,
        "upside_quantile": 0.0,
        "quantile_width": 0.0,
        "model_confidence": 0.0,
        target_column: np.nan,
    }
    for column, default in defaults.items():
        if column not in frame:
            frame[column] = pd.Series(default, index=frame.index)
    return frame


def _requires_model_refit(toggles: AblationToggles) -> bool:
    return (
        not toggles.include_price_features
        or not toggles.include_text_features
        or not toggles.include_sec_features
        or not toggles.include_model_proxy_features
        or not toggles.include_proxy_features
        or not toggles.include_proxy_model_inputs
        or not toggles.include_chronos_features
        or not toggles.include_granite_ttm_features
    )


def _backtest_config(
    config: PipelineConfig,
    cost_bps: float | None = None,
    slippage_bps: float | None = None,
    realized_return_column: str | None = None,
) -> BacktestConfig:
    return BacktestConfig(
        top_n=config.top_n,
        cost_bps=config.cost_bps if cost_bps is None else cost_bps,
        slippage_bps=config.slippage_bps if slippage_bps is None else slippage_bps,
        average_daily_turnover_budget=config.average_daily_turnover_budget,
        benchmark_ticker=config.benchmark_ticker,
        realized_return_column=realized_return_column or config.prediction_target_column,
        max_symbol_weight=config.max_symbol_weight,
        max_sector_weight=config.max_sector_weight,
        max_correlation_cluster_weight=config.max_correlation_cluster_weight,
        correlation_cluster_threshold=config.correlation_cluster_threshold,
        portfolio_covariance_lookback=config.portfolio_covariance_lookback,
        covariance_aware_risk_enabled=config.covariance_aware_risk_enabled,
        covariance_return_column=config.covariance_return_column,
        covariance_min_periods=config.covariance_min_periods,
        max_daily_turnover=config.max_daily_turnover,
        portfolio_volatility_limit=config.portfolio_volatility_limit,
        max_drawdown_stop=config.max_drawdown_stop,
        max_position_risk_contribution=config.max_position_risk_contribution,
        volatility_adjustment_strength=config.volatility_adjustment_strength,
        concentration_adjustment_strength=config.concentration_adjustment_strength,
        risk_contribution_adjustment_strength=config.risk_contribution_adjustment_strength,
    )


def _walk_forward_config(config: PipelineConfig) -> WalkForwardConfig:
    target_horizon = (
        _target_horizon(config.prediction_target_column) or config.required_validation_horizon
    )
    required_horizon = _minimum_validation_gap_periods(
        config.prediction_target_column,
        config.required_validation_horizon,
    )
    return WalkForwardConfig(
        train_periods=config.train_periods,
        test_periods=config.test_periods,
        gap_periods=max(config.gap_periods, required_horizon),
        model_name=config.model_name,
        native_tabular_isolation=config.native_tabular_isolation,
        native_model_timeout_seconds=config.native_model_timeout_seconds,
        tabular_num_threads=config.tabular_num_threads,
        embargo_periods=max(config.embargo_periods, required_horizon),
        prediction_horizon_periods=target_horizon,
    )


def _walk_forward_splitter(
    config: PipelineConfig,
    *,
    target_column: str | None = None,
) -> WalkForwardSplitter:
    target = target_column or config.prediction_target_column
    required_horizon = _minimum_validation_gap_periods(
        target,
        config.required_validation_horizon,
    )
    splitter_config = PurgeEmbargoWalkForwardConfig(
        train_periods=config.train_periods,
        test_periods=config.test_periods,
        purge_periods=max(config.gap_periods, required_horizon),
        embargo_periods=max(config.embargo_periods, required_horizon),
        target_column=target,
    )
    return PurgeEmbargoWalkForwardSplitter(splitter_config)


def _predict_walk_forward_with_splitter(
    frame: pd.DataFrame,
    walk_config: WalkForwardConfig,
    *,
    target: str,
    splitter: WalkForwardSplitter,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    walk_forward_signature = signature(walk_forward_predict)
    parameters = walk_forward_signature.parameters
    supports_splitter = "splitter" in parameters or any(
        parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    if supports_splitter:
        return walk_forward_predict(frame, walk_config, target=target, splitter=splitter)
    return walk_forward_predict(frame, walk_config, target=target)


def _minimum_validation_gap_periods(
    prediction_target_column: str, required_validation_horizon: int
) -> int:
    target_horizon = _target_horizon(prediction_target_column) or 1
    return max(int(required_validation_horizon), target_horizon, 1)


def _available_prediction_target_column(frame: pd.DataFrame, preferred: str) -> str:
    if preferred in frame.columns:
        return preferred
    forward_columns = [
        str(column) for column in frame.columns if str(column).startswith("forward_return_")
    ]
    if not forward_columns:
        return preferred
    return max(forward_columns, key=lambda column: _target_horizon(column) or 0)


def _target_horizon(target_column: str) -> int | None:
    prefix = "forward_return_"
    if not target_column.startswith(prefix):
        return None
    try:
        return int(target_column.removeprefix(prefix))
    except ValueError:
        return None


def _benchmark_evaluation_frame(
    backtest: BacktestResult, predictions: pd.DataFrame
) -> pd.DataFrame | None:
    if not backtest.equity_curve.empty and "date" in backtest.equity_curve:
        return backtest.equity_curve
    if not predictions.empty and "date" in predictions:
        return predictions
    return None


def _evaluated_ticker_universe(
    predictions: pd.DataFrame,
    configured_tickers: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    configured = _normalize_ticker_sequence(configured_tickers)
    if predictions.empty or "ticker" not in predictions:
        return configured

    predicted = _normalize_ticker_sequence(predictions["ticker"])
    if not configured:
        return predicted
    if not predicted:
        return configured

    predicted_set = set(predicted)
    evaluated = tuple(ticker for ticker in configured if ticker in predicted_set)
    return evaluated or configured


def _normalize_ticker_sequence(tickers: object) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        value = str(ticker).strip().upper()
        if value and value not in seen:
            normalized.append(value)
            seen.add(value)
    return tuple(normalized)
