from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from quant_research.backtest.engine import BacktestConfig, BacktestResult, run_long_only_backtest
from quant_research.config import DEFAULT_TICKERS
from quant_research.data.market import SyntheticMarketDataProvider, YFinanceMarketDataProvider
from quant_research.data.news import GDELTNewsProvider, SyntheticNewsProvider, YFinanceNewsProvider
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
from quant_research.features.sec import build_sec_features
from quant_research.features.text import KeywordSentimentAnalyzer, build_news_features
from quant_research.models.text import (
    FilingEventExtractor,
    FinBERTSentimentAnalyzer,
    FinGPTEventExtractor,
)
from quant_research.models.timeseries import Chronos2Adapter, GraniteTTMAdapter
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


@dataclass(frozen=True)
class PipelineConfig:
    tickers: list[str] = field(default_factory=lambda: list(DEFAULT_TICKERS))
    data_mode: str = "synthetic"
    start: date = field(default_factory=lambda: date.today() - timedelta(days=365 * 2))
    end: date = field(default_factory=date.today)
    interval: str = "1d"
    train_periods: int = 90
    test_periods: int = 20
    gap_periods: int = 1
    model_name: str = "lightgbm"
    top_n: int = 3
    cost_bps: float = 5.0
    slippage_bps: float = 2.0
    sentiment_model: str = "finbert"
    time_series_inference_mode: str = "proxy"
    time_series_target_column: str = "return_1"
    time_series_min_context: int = 64
    max_time_series_inference_windows: int | None = None
    chronos_model_id: str = "amazon/chronos-2"
    granite_ttm_model_id: str = "ibm-granite/granite-timeseries-ttm-r2"
    granite_ttm_revision: str | None = None
    local_model_device_map: str = "auto"
    filing_extractor_model: str = "rules"
    enable_local_filing_llm: bool = False
    finma_model_id: str = "ChanceFocus/finma-7b-nlp"
    fingpt_model_id: str = "FinGPT/fingpt-mt_llama3-8b_lora"
    fingpt_base_model_id: str | None = "meta-llama/Meta-Llama-3-8B"
    max_symbol_weight: float = 0.35
    portfolio_volatility_limit: float = 0.04
    max_drawdown_stop: float = 0.20
    sec_frame_period: str = "CY2024Q4I"
    enable_feature_model_ablation: bool = False


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


def run_research_pipeline(config: PipelineConfig | None = None) -> PipelineResult:
    config = config or PipelineConfig()
    tickers = [ticker.upper() for ticker in config.tickers]

    market_data = _load_market_data(config, tickers)
    price_features = build_price_features(market_data)
    price_features = Chronos2Adapter(
        model_id=config.chronos_model_id,
        device_map=config.local_model_device_map,
    ).add_forecasts(
        price_features,
        mode=config.time_series_inference_mode,
        target_column=config.time_series_target_column,
        min_context=config.time_series_min_context,
        max_inference_windows=config.max_time_series_inference_windows,
    )
    price_features = GraniteTTMAdapter(
        model_id=config.granite_ttm_model_id,
        revision=config.granite_ttm_revision,
    ).add_forecasts(
        price_features,
        mode=config.time_series_inference_mode,
        target_column=config.time_series_target_column,
        min_context=config.time_series_min_context,
        max_inference_windows=config.max_time_series_inference_windows,
    )

    news_items = _load_news_items(config, tickers)
    news_features = build_news_features(news_items, _sentiment_analyzer(config.sentiment_model))

    filings_by_ticker, facts_by_ticker = _load_sec_data(config, tickers)
    sec_features = build_sec_features(
        filings_by_ticker,
        facts_by_ticker,
        price_features,
        filing_extractor=_filing_extractor(config),
    )

    features = fuse_features(price_features, news_features, sec_features)
    features = features.dropna(subset=["forward_return_1"]).reset_index(drop=True)

    predictions, validation_summary = walk_forward_predict(
        features,
        WalkForwardConfig(
            train_periods=config.train_periods,
            test_periods=config.test_periods,
            gap_periods=config.gap_periods,
            model_name=config.model_name,
        ),
    )
    predictions = _attach_signal_features(predictions, features)
    backtest = run_long_only_backtest(
        predictions,
        _backtest_config(config),
    )
    ablation_summary = _run_ablation_summary(predictions, config)
    if config.enable_feature_model_ablation:
        ablation_summary.extend(_run_feature_model_ablation_summary(features, config))

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
    )


def _load_market_data(config: PipelineConfig, tickers: list[str]) -> pd.DataFrame:
    if config.data_mode == "live":
        provider = YFinanceMarketDataProvider()
        live = provider.get_history(tickers, config.start, config.end, config.interval)
        if not live.empty:
            return live
    return SyntheticMarketDataProvider(periods=260).get_history(tickers, config.start, config.end, config.interval)


def _load_news_items(config: PipelineConfig, tickers: list[str]):
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
                filings_by_ticker[ticker] = client.recent_filings(cik, {"8-K", "10-Q", "10-K", "4"})
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
    filings = {ticker: provider.recent_filings(CIK_BY_TICKER.get(ticker, "0")) for ticker in tickers}
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
    signal_columns = [
        "date",
        "ticker",
        "forward_return_1",
        "text_risk_score",
        "sec_risk_flag",
        "sec_risk_flag_20d",
        "liquidity_score",
        "news_sentiment_mean",
        "news_negative_ratio",
        "sec_event_confidence",
        "revenue_growth",
        "net_income_growth",
        "assets_growth",
    ]
    available = [column for column in signal_columns if column in features.columns]
    enriched = predictions.drop(columns=["forward_return_1"], errors="ignore").merge(
        features[available],
        on=["date", "ticker"],
        how="left",
    )
    for column in ["text_risk_score", "sec_risk_flag", "liquidity_score"]:
        if column not in enriched:
            enriched[column] = 0.0
    return enriched


def _sentiment_analyzer(model_name: str):
    if model_name.lower() == "finbert":
        return FinBERTSentimentAnalyzer()
    return KeywordSentimentAnalyzer()


def _filing_extractor(config: PipelineConfig):
    model_name = config.filing_extractor_model.lower()
    if model_name == "finma":
        return FilingEventExtractor(
            model_id=config.finma_model_id,
            use_local_model=config.enable_local_filing_llm,
            device_map=config.local_model_device_map,
        )
    if model_name == "fingpt":
        return FinGPTEventExtractor(
            model_id=config.fingpt_model_id,
            use_local_model=config.enable_local_filing_llm,
            device_map=config.local_model_device_map,
            base_model_id=config.fingpt_base_model_id,
        )
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
    variants = {
        "full_model_features": features,
        "no_chronos_features": features.drop(
            columns=[column for column in features.columns if column.startswith("chronos_")],
            errors="ignore",
        ),
        "no_granite_features": features.drop(
            columns=[column for column in features.columns if column.startswith("granite_ttm_")],
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
            WalkForwardConfig(
                train_periods=config.train_periods,
                test_periods=config.test_periods,
                gap_periods=config.gap_periods,
                model_name=config.model_name,
            ),
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
        max_symbol_weight=config.max_symbol_weight,
        portfolio_volatility_limit=config.portfolio_volatility_limit,
        max_drawdown_stop=config.max_drawdown_stop,
    )
