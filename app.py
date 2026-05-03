from __future__ import annotations

import pandas as pd
import streamlit as st

from quant_research.config import DEFAULT_TICKERS
from quant_research.pipeline import PipelineConfig, run_research_pipeline

st.set_page_config(
    page_title="Quant Research",
    page_icon="Q",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Quant Research")

with st.sidebar:
    st.header("Run")
    data_mode = st.selectbox("Data mode", ["synthetic", "live"], index=0)
    tickers_text = st.text_area("Tickers", ", ".join(DEFAULT_TICKERS), height=90)
    top_n = st.slider("Top N", min_value=1, max_value=8, value=3)
    train_periods = st.slider("Train periods", min_value=30, max_value=300, value=90, step=10)
    test_periods = st.slider("Test periods", min_value=5, max_value=60, value=20, step=5)
    cost_bps = st.slider("Cost bps", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
    slippage_bps = st.slider("Slippage bps", min_value=0.0, max_value=50.0, value=2.0, step=0.5)
    sentiment_model = st.selectbox("Sentiment model", ["finbert", "keyword"], index=0)
    time_series_inference_mode = st.selectbox("Time-series inference", ["proxy", "local"], index=0)
    max_ts_windows = st.number_input(
        "Local TS inference windows",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
        help="Only used when local time-series inference is selected.",
    )
    filing_extractor_model = st.selectbox("Filing extractor", ["rules", "finma", "fingpt"], index=0)
    enable_local_filing_llm = st.checkbox("Use local filing LLM", value=False)
    with st.expander("Local model settings"):
        local_model_device_map = st.text_input("Device map", value="auto")
        chronos_model_id = st.text_input("Chronos-2 model", value="amazon/chronos-2")
        granite_ttm_model_id = st.text_input(
            "Granite TTM model",
            value="ibm-granite/granite-timeseries-ttm-r2",
        )
        granite_ttm_revision = st.text_input("Granite revision", value="")
        finma_model_id = st.text_input("FinMA model", value="ChanceFocus/finma-7b-nlp")
        fingpt_model_id = st.text_input("FinGPT adapter", value="FinGPT/fingpt-mt_llama3-8b_lora")
        fingpt_base_model_id = st.text_input("FinGPT base model", value="meta-llama/Meta-Llama-3-8B")
    max_symbol_weight = st.slider("Max symbol weight", min_value=0.05, max_value=1.0, value=0.35, step=0.05)
    portfolio_volatility_limit = st.slider(
        "Portfolio volatility limit",
        min_value=0.005,
        max_value=0.10,
        value=0.04,
        step=0.005,
    )
    max_drawdown_stop = st.slider("Max drawdown stop", min_value=0.05, max_value=0.50, value=0.20, step=0.05)
    enable_feature_model_ablation = st.checkbox("Run model feature ablation", value=False)
    run = st.button("Run research", type="primary", use_container_width=True)

if data_mode == "live":
    st.info(
        "Live mode may call yfinance, GDELT, and SEC EDGAR. SEC requests use a User-Agent and local cache. "
        "Synthetic mode is recommended for offline verification."
    )

tickers = [ticker.strip().upper() for ticker in tickers_text.split(",") if ticker.strip()]
config = PipelineConfig(
    tickers=tickers,
    data_mode=data_mode,
    train_periods=train_periods,
    test_periods=test_periods,
    top_n=top_n,
    cost_bps=cost_bps,
    slippage_bps=slippage_bps,
    sentiment_model=sentiment_model,
    time_series_inference_mode=time_series_inference_mode,
    max_time_series_inference_windows=int(max_ts_windows),
    chronos_model_id=chronos_model_id,
    granite_ttm_model_id=granite_ttm_model_id,
    granite_ttm_revision=granite_ttm_revision or None,
    local_model_device_map=local_model_device_map,
    filing_extractor_model=filing_extractor_model,
    enable_local_filing_llm=enable_local_filing_llm,
    finma_model_id=finma_model_id,
    fingpt_model_id=fingpt_model_id,
    fingpt_base_model_id=fingpt_base_model_id or None,
    max_symbol_weight=max_symbol_weight,
    portfolio_volatility_limit=portfolio_volatility_limit,
    max_drawdown_stop=max_drawdown_stop,
    enable_feature_model_ablation=enable_feature_model_ablation,
)

if run or "result" not in st.session_state:
    with st.spinner("Building features, validating models, and running the deterministic signal engine"):
        st.session_state["result"] = run_research_pipeline(config)

result = st.session_state["result"]
metrics = result.backtest.metrics

metric_cols = st.columns(6)
metric_cols[0].metric("CAGR", f"{metrics.cagr:.2%}")
metric_cols[1].metric("Sharpe", f"{metrics.sharpe:.2f}")
metric_cols[2].metric("Max DD", f"{metrics.max_drawdown:.2%}")
metric_cols[3].metric("Hit Rate", f"{metrics.hit_rate:.2%}")
metric_cols[4].metric("Turnover", f"{metrics.turnover:.2%}")
metric_cols[5].metric("Exposure", f"{metrics.exposure:.2%}")

tabs = st.tabs(["Backtest", "Signals", "Features", "Validation", "Data"])

with tabs[0]:
    st.subheader("Equity Curve")
    equity = result.backtest.equity_curve.set_index("date")
    st.line_chart(equity[["equity", "benchmark_equity"]])
    st.subheader("Portfolio Returns")
    st.bar_chart(equity.set_index(equity.index)["portfolio_return"])

with tabs[1]:
    st.subheader("Latest Deterministic Signals")
    latest_date = result.signals["date"].max()
    latest = (
        result.signals[result.signals["date"] == latest_date]
        .sort_values(["action", "signal_score"], ascending=[True, False])
        .reset_index(drop=True)
    )
    st.dataframe(
        latest[
            [
                "date",
                "ticker",
                "action",
                "signal_score",
                "expected_return",
                "predicted_volatility",
                "downside_quantile",
                "text_risk_score",
                "sec_risk_flag",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

with tabs[2]:
    st.subheader("Feature Fusion Sample")
    st.dataframe(result.features.tail(200), use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Walk-Forward Summary")
    st.dataframe(result.validation_summary, use_container_width=True, hide_index=True)
    if "is_oos" in result.validation_summary.columns:
        oos_summary = result.validation_summary[result.validation_summary["is_oos"]]
    else:
        oos_summary = pd.DataFrame()
    if not oos_summary.empty:
        st.subheader("Out-of-Sample Holdout")
        st.dataframe(oos_summary, use_container_width=True, hide_index=True)
    st.subheader("Ablation Summary")
    st.dataframe(pd.DataFrame(result.ablation_summary), use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Raw Market Sample")
    st.dataframe(result.market_data.tail(200), use_container_width=True, hide_index=True)
    st.subheader("News Feature Sample")
    st.dataframe(result.news_features.tail(100), use_container_width=True, hide_index=True)
    st.subheader("SEC Feature Sample")
    st.dataframe(result.sec_features.tail(100), use_container_width=True, hide_index=True)
