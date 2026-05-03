from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from quant_research.backtest.engine import BacktestConfig, BacktestResult, run_long_only_backtest
from quant_research.config import DEFAULT_TICKERS
from quant_research.data.market import SyntheticMarketDataProvider, YFinanceMarketDataProvider
from quant_research.data.news import GDELTNewsProvider, SyntheticNewsProvider, YFinanceNewsProvider
from quant_research.data.sec import SecEdgarClient, SyntheticSecProvider, extract_companyfacts_frame
from quant_research.features.fusion import fuse_features
from quant_research.features.price import build_price_features
from quant_research.features.sec import build_sec_features
from quant_research.features.text import KeywordSentimentAnalyzer, build_news_features
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
    price_features = Chronos2Adapter().add_proxy_forecasts(price_features)
    price_features = GraniteTTMAdapter().add_proxy_forecasts(price_features)

    news_items = _load_news_items(config, tickers)
    news_features = build_news_features(news_items, KeywordSentimentAnalyzer())

    filings_by_ticker, facts_by_ticker = _load_sec_data(config, tickers)
    sec_features = build_sec_features(filings_by_ticker, facts_by_ticker, price_features)

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
        BacktestConfig(top_n=config.top_n, cost_bps=config.cost_bps, slippage_bps=config.slippage_bps),
    )
    ablation_summary = _run_ablation_summary(predictions, config)

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
        for ticker in tickers:
            cik = CIK_BY_TICKER.get(ticker)
            if cik is None:
                continue
            try:
                filings_by_ticker[ticker] = client.recent_filings(cik, {"8-K", "10-Q", "10-K", "4"})
                facts_by_ticker[ticker] = extract_companyfacts_frame(client.get_companyfacts(cik))
            except Exception:
                continue
        if filings_by_ticker or facts_by_ticker:
            return filings_by_ticker, facts_by_ticker

    provider = SyntheticSecProvider()
    filings = {ticker: provider.recent_filings(CIK_BY_TICKER.get(ticker, "0")) for ticker in tickers}
    facts = {ticker: provider.companyfacts_frame(CIK_BY_TICKER.get(ticker, "0")) for ticker in tickers}
    return filings, facts


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
            BacktestConfig(top_n=config.top_n, cost_bps=config.cost_bps, slippage_bps=config.slippage_bps),
        )
        scenarios.append(
            {
                "scenario": name,
                "cagr": result.metrics.cagr,
                "sharpe": result.metrics.sharpe,
                "max_drawdown": result.metrics.max_drawdown,
                "excess_return": result.metrics.excess_return,
            }
        )

    no_cost = run_long_only_backtest(
        predictions,
        BacktestConfig(top_n=config.top_n, cost_bps=0.0, slippage_bps=0.0),
    )
    scenarios.append(
        {
            "scenario": "no_costs",
            "cagr": no_cost.metrics.cagr,
            "sharpe": no_cost.metrics.sharpe,
            "max_drawdown": no_cost.metrics.max_drawdown,
            "excess_return": no_cost.metrics.excess_return,
        }
    )
    return scenarios
