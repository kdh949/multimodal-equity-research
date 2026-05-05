from __future__ import annotations

# ruff: noqa: E402, I001

import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from inspect import Parameter, signature

from quant_research.runtime import configure_local_runtime_defaults

configure_local_runtime_defaults()

import pandas as pd

from quant_research.backtest.engine import BacktestConfig, BacktestResult, run_long_only_backtest
from quant_research.config import DEFAULT_TICKERS
from quant_research.data.market import LocalMarketDataProvider, SyntheticMarketDataProvider, YFinanceMarketDataProvider
from quant_research.data.news import GDELTNewsProvider, LocalNewsProvider, SyntheticNewsProvider, YFinanceNewsProvider
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
from quant_research.models.timeseries import Chronos2Adapter, GraniteTTMAdapter
from quant_research.validation.gate import ValidationGateReport, build_validity_gate_report
from quant_research.validation.walk_forward import WalkForwardConfig, walk_forward_predict

CIK_BY_TICKER = {
    "AAPL": "320193",
    "MSFT": "789019",
    "NVDA": "1045810",
    "AMZN": "1018724",
    "GOOGL": "1652044",
    "META": "1326801",
    "TSLA": "1318605",
    "JPM": "19617",
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
    gap_periods: int = 5
    embargo_periods: int = 5
    model_name: str = "lightgbm"
    prediction_target_column: str = "forward_return_5"
    required_validation_horizon: int = 5
    top_n: int = 3
    cost_bps: float = 5.0
    slippage_bps: float = 2.0
    sentiment_model: str = "finbert"
    time_series_inference_mode: str = "local"
    time_series_target_column: str = "return_1"
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
    max_symbol_weight: float = 0.35
    portfolio_volatility_limit: float = 0.04
    max_drawdown_stop: float = 0.20
    sec_frame_period: str = "CY2024Q4I"
    sec_filing_event_cache_path: str | None = str(DEFAULT_FILING_EVENT_CACHE_PATH)
    enable_feature_model_ablation: bool = False
    native_tabular_isolation: bool = True
    native_model_timeout_seconds: int = 180
    tabular_num_threads: int = 1


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
    validity_report: ValidationGateReport | None = None


def run_research_pipeline(config: PipelineConfig | None = None) -> PipelineResult:
    config = config or PipelineConfig()
    tickers = [ticker.upper() for ticker in config.tickers]

    market_data = _load_market_data(config, tickers)
    price_features = build_price_features(market_data)
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
    news_features = build_news_features(news_items, _sentiment_analyzer(config.sentiment_model, config.local_model_device_map))

    filings_by_ticker, facts_by_ticker = _load_sec_data(config, tickers)
    filing_extractor = _filing_extractor(config)
    sec_features = build_sec_features(
        filings_by_ticker,
        facts_by_ticker,
        price_features,
        filing_extractor=filing_extractor,
        filing_cache_path=(
            config.sec_filing_event_cache_path
            if config.enable_local_filing_llm and bool(getattr(filing_extractor, "use_local_model", False))
            else None
        ),
    )

    features = fuse_features(price_features, news_features, sec_features)
    features = features.dropna(subset=[config.prediction_target_column]).reset_index(drop=True)

    walk_config = _walk_forward_config(config)
    predictions, validation_summary = walk_forward_predict(
        features,
        walk_config,
        target=config.prediction_target_column,
    )
    predictions = _attach_signal_features(predictions, features)
    backtest = run_long_only_backtest(
        predictions,
        _backtest_config(config),
    )
    ablation_summary = _run_ablation_summary(predictions, config)
    if config.enable_feature_model_ablation:
        ablation_summary.extend(_run_feature_model_ablation_summary(features, config))
    validity_report = build_validity_gate_report(
        predictions,
        validation_summary,
        backtest.equity_curve,
        backtest.metrics,
        ablation_summary,
        config=config,
        walk_forward_config=walk_config,
    )

    return PipelineResult(
        market_data=market_data,
        news_features=news_features,
        sec_features=sec_features,
        features=features,
        predictions=predictions,
        signals=backtest.signals,
        validation_summary=validation_summary,
        validity_report=validity_report,
        ablation_summary=ablation_summary,
        backtest=backtest,
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
    return SyntheticMarketDataProvider(periods=260).get_history(tickers, config.start, config.end, config.interval)


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


def _load_sec_data(config: PipelineConfig, tickers: list[str]) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
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
        if filings_by_ticker or facts_by_ticker:
            return filings_by_ticker, facts_by_ticker
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
        if filings_by_ticker or facts_by_ticker:
            return filings_by_ticker, facts_by_ticker

    provider = SyntheticSecProvider()
    filings = {
        ticker: provider.recent_filings(
            CIK_BY_TICKER.get(ticker, "0"),
            include_document_text=True,
        )
        for ticker in tickers
    }
    facts = {ticker: provider.companyfacts_frame(CIK_BY_TICKER.get(ticker, "0")) for ticker in tickers}
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
            enriched[column] = pd.to_numeric(enriched[column], errors="coerce").fillna(default_value)
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
                key: value
                for key, value in runtime_args.items()
                if key in accepted_keys
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


def _run_ablation_summary(predictions: pd.DataFrame, config: PipelineConfig) -> list[dict[str, object]]:
    if predictions.empty:
        return []

    scenarios = []
    scenario_frames = {
        "all_features": predictions,
        "no_text_risk": predictions.assign(text_risk_score=0.0, news_negative_ratio=0.0),
        "no_sec_risk": predictions.assign(sec_risk_flag=0.0, sec_risk_flag_20d=0.0),
    }
    for name, frame in scenario_frames.items():
        result = run_long_only_backtest(
            frame,
            _backtest_config(config),
        )
        scenarios.append(
            {
                "scenario": name,
                "kind": "signal",
                "cagr": result.metrics.cagr,
                "sharpe": result.metrics.sharpe,
                "max_drawdown": result.metrics.max_drawdown,
                "excess_return": result.metrics.excess_return,
            }
        )

    no_cost = run_long_only_backtest(
        predictions,
        _backtest_config(config, cost_bps=0.0, slippage_bps=0.0),
    )
    scenarios.append(
        {
            "scenario": "no_costs",
            "kind": "cost",
            "cagr": no_cost.metrics.cagr,
            "sharpe": no_cost.metrics.sharpe,
            "max_drawdown": no_cost.metrics.max_drawdown,
            "excess_return": no_cost.metrics.excess_return,
        }
    )
    return scenarios


def _run_feature_model_ablation_summary(
    features: pd.DataFrame,
    config: PipelineConfig,
) -> list[dict[str, object]]:
    scenarios: list[dict[str, object]] = []
    base_columns = _base_feature_columns(features)
    variants = {
        "full_model_features": features,
        "price_only": _feature_subset(features, base_columns, _price_feature_columns(features)),
        "text_only": _feature_subset(features, base_columns, _text_feature_columns(features)),
        "sec_only": _feature_subset(features, base_columns, _sec_feature_columns(features)),
        "no_chronos_features": features.drop(
            columns=[column for column in features.columns if column.startswith("chronos_")],
            errors="ignore",
        ),
        "no_granite_features": features.drop(
            columns=[column for column in features.columns if column.startswith("granite_ttm_")],
            errors="ignore",
        ),
        "no_model_proxy": features.drop(
            columns=[
                column
                for column in features.columns
                if column.startswith("chronos_") or column.startswith("granite_ttm_")
            ],
            errors="ignore",
        ),
        "tabular_without_ts_proxies": features.drop(
            columns=[
                column
                for column in features.columns
                if column.startswith("chronos_") or column.startswith("granite_ttm_")
            ],
            errors="ignore",
        ),
    }
    for scenario, variant in variants.items():
        predictions, _summary = walk_forward_predict(
            variant,
            _walk_forward_config(config),
            target=config.prediction_target_column,
        )
        predictions = _attach_signal_features(predictions, variant)
        result = run_long_only_backtest(predictions, _backtest_config(config))
        scenarios.append(
            {
                "scenario": scenario,
                "kind": "model_feature",
                "cagr": result.metrics.cagr,
                "sharpe": result.metrics.sharpe,
                "max_drawdown": result.metrics.max_drawdown,
                "excess_return": result.metrics.excess_return,
            }
        )
    return scenarios


def _backtest_config(
    config: PipelineConfig,
    cost_bps: float | None = None,
    slippage_bps: float | None = None,
) -> BacktestConfig:
    return BacktestConfig(
        top_n=config.top_n,
        cost_bps=config.cost_bps if cost_bps is None else cost_bps,
        slippage_bps=config.slippage_bps if slippage_bps is None else slippage_bps,
        realized_return_column=config.prediction_target_column,
        max_symbol_weight=config.max_symbol_weight,
        portfolio_volatility_limit=config.portfolio_volatility_limit,
        max_drawdown_stop=config.max_drawdown_stop,
    )


def _walk_forward_config(config: PipelineConfig) -> WalkForwardConfig:
    horizon = _target_horizon(config.prediction_target_column) or config.required_validation_horizon
    return WalkForwardConfig(
        train_periods=config.train_periods,
        test_periods=config.test_periods,
        gap_periods=max(config.gap_periods, horizon),
        model_name=config.model_name,
        native_tabular_isolation=config.native_tabular_isolation,
        native_model_timeout_seconds=config.native_model_timeout_seconds,
        tabular_num_threads=config.tabular_num_threads,
        embargo_periods=max(config.embargo_periods, horizon),
        target_horizon=horizon,
        requested_gap_periods=config.gap_periods,
        requested_embargo_periods=config.embargo_periods,
    )


def _target_horizon(target_column: str) -> int | None:
    prefix = "forward_return_"
    if not target_column.startswith(prefix):
        return None
    try:
        return int(target_column.removeprefix(prefix))
    except ValueError:
        return None


def _base_feature_columns(features: pd.DataFrame) -> list[str]:
    return [
        column
        for column in features.columns
        if column in {"date", "ticker"} or str(column).startswith("forward_return_")
    ]


def _price_feature_columns(features: pd.DataFrame) -> list[str]:
    prefixes = (
        "return_",
        "volatility_",
        "volume_",
        "ma_ratio_",
        "rsi_",
        "high_low_",
        "dollar_volume",
        "realized_volatility",
        "liquidity_score",
    )
    return [column for column in features.columns if str(column).startswith(prefixes)]


def _text_feature_columns(features: pd.DataFrame) -> list[str]:
    prefixes = ("news_", "sentiment_", "text_")
    return [column for column in features.columns if str(column).startswith(prefixes)]


def _sec_feature_columns(features: pd.DataFrame) -> list[str]:
    return [column for column in features.columns if str(column).startswith("sec_")]


def _feature_subset(
    features: pd.DataFrame,
    base_columns: list[str],
    selected_columns: list[str],
) -> pd.DataFrame:
    columns = list(dict.fromkeys([*base_columns, *selected_columns]))
    return features[[column for column in columns if column in features.columns]].copy()
