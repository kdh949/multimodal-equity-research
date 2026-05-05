from __future__ import annotations

# ruff: noqa: E402, I001

from quant_research.runtime import configure_local_runtime_defaults

configure_local_runtime_defaults()

import pandas as pd
import streamlit as st

from quant_research.config import DEFAULT_TICKERS
from quant_research.dashboard import build_beginner_research_dashboard
from quant_research.dashboard.streamlit import render_beginner_overview
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
    sidebar_tickers = [ticker.strip().upper() for ticker in tickers_text.split(",") if ticker.strip()]
    if not sidebar_tickers:
        sidebar_tickers = list(DEFAULT_TICKERS)
    focus_ticker = st.selectbox("Focus ticker", sidebar_tickers, index=0)
    top_n = st.slider("Top N", min_value=1, max_value=8, value=3)
    train_periods = st.slider("Train periods", min_value=30, max_value=300, value=90, step=10)
    test_periods = st.slider("Test periods", min_value=5, max_value=60, value=20, step=5)
    cost_bps = st.slider("Cost bps", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
    slippage_bps = st.slider("Slippage bps", min_value=0.0, max_value=50.0, value=2.0, step=0.5)
    sentiment_model = st.selectbox("Sentiment model", ["keyword", "finbert"], index=1)
    time_series_inference_mode = st.selectbox("Time-series inference", ["proxy", "local"], index=1)
    max_ts_windows = st.number_input(
        "Local TS inference windows",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
        help="Only used when local time-series inference is selected.",
    )
    filing_extractor_model = st.selectbox("Filing extractor", ["rules", "finma", "fingpt"], index=2)
    enable_local_filing_llm = st.checkbox("Use local filing LLM", value=True)
    defaults = PipelineConfig()
    fingpt_runtime_options = ("transformers", "mlx", "llama-cpp")
    fingpt_default_runtime = defaults.fingpt_runtime
    if fingpt_default_runtime not in fingpt_runtime_options:
        fingpt_default_runtime = "transformers"
    with st.expander("Local model settings"):
        local_model_device_map = st.text_input("Device map", value="auto")
        local_model_offload_folder = st.text_input("Offload folder", value="artifacts/model_offload")
        chronos_model_id = st.text_input("Chronos-2 model", value="amazon/chronos-2")
        granite_ttm_model_id = st.text_input(
            "Granite TTM model",
            value="ibm-granite/granite-timeseries-ttm-r2",
        )
        granite_ttm_revision = st.text_input("Granite revision", value="")
        finma_model_id = st.text_input("FinMA model", value="ChanceFocus/finma-7b-nlp")
        fingpt_model_id = st.text_input("FinGPT adapter", value="FinGPT/fingpt-mt_llama3-8b_lora")
        fingpt_base_model_id = st.text_input("FinGPT base model", value="meta-llama/Meta-Llama-3-8B")
        fingpt_runtime = st.selectbox(
            "FinGPT runtime",
            options=fingpt_runtime_options,
            index=fingpt_runtime_options.index(fingpt_default_runtime),
            help="Transformers loads only base LoRA flow; MLX/llama-cpp requires a local quantized file path.",
        )
        fingpt_quantized_model_path = st.text_input(
            "FinGPT quantized runtime path",
            value=defaults.fingpt_quantized_model_path,
            help="Warmup and inference are optional heavy steps; keep blank if you do not use quantized runtime.",
        )
        fingpt_allow_unquantized_transformers = st.checkbox(
            "Allow unquantized Transformers 8B load",
            value=defaults.fingpt_allow_unquantized_transformers,
            help=(
                "Only enable for explicit local experiments; keeps default safety guard off when false."
            ),
        )
        fingpt_single_load_lock_path = st.text_input(
            "FinGPT single-load lock file",
            value=defaults.fingpt_single_load_lock_path or "",
            help="Limits concurrent FinGPT local model load attempts to one at a time.",
        )
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
    run = st.button("Run research", type="primary", width="stretch")

if data_mode == "live":
    st.info(
        "Live mode may call yfinance, GDELT, and SEC EDGAR. SEC requests use a User-Agent and local cache. "
        "Synthetic mode is recommended for offline verification."
    )

tickers = [ticker.strip().upper() for ticker in tickers_text.split(",") if ticker.strip()]
if not tickers:
    tickers = list(DEFAULT_TICKERS)
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
    local_model_offload_folder=local_model_offload_folder or None,
    filing_extractor_model=filing_extractor_model,
    enable_local_filing_llm=enable_local_filing_llm,
    finma_model_id=finma_model_id,
    fingpt_model_id=fingpt_model_id,
    fingpt_base_model_id=fingpt_base_model_id or None,
    fingpt_runtime=fingpt_runtime,
    fingpt_quantized_model_path=fingpt_quantized_model_path,
    fingpt_allow_unquantized_transformers=fingpt_allow_unquantized_transformers,
    fingpt_single_load_lock_path=fingpt_single_load_lock_path or None,
    max_symbol_weight=max_symbol_weight,
    portfolio_volatility_limit=portfolio_volatility_limit,
    max_drawdown_stop=max_drawdown_stop,
    enable_feature_model_ablation=enable_feature_model_ablation,
)

if run:
    with st.spinner("Building features, validating models, and running the deterministic signal engine"):
        st.session_state["result"] = run_research_pipeline(config)

result = st.session_state.get("result")
if result is None:
    st.info("기본 모드에서는 계산이 실행되지 않습니다. 먼저 'Run research' 버튼을 눌러서 결과를 생성하세요.")
    st.caption("선택 항목은 기본값으로 구성되며, 이후 동일 버튼 클릭 시 최신 값으로만 실행됩니다.")
    st.stop()

metrics = result.backtest.metrics
dashboard = build_beginner_research_dashboard(result, focus_ticker, config)

render_beginner_overview(dashboard)

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
    signal_columns = [
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
    latest = (
        result.signals[result.signals["date"] == latest_date]
        .sort_values("signal_score", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(
        latest[[col for col in signal_columns if col in latest.columns]],
        width="stretch",
        hide_index=True,
    )

with tabs[2]:
    st.subheader("Feature Fusion Sample")
    st.dataframe(result.features.tail(200), width="stretch", hide_index=True)

with tabs[3]:
    st.subheader("Walk-Forward Summary")
    st.dataframe(result.validation_summary, width="stretch", hide_index=True)
    if "is_oos" in result.validation_summary.columns:
        oos_summary = result.validation_summary[result.validation_summary["is_oos"]]
    else:
        oos_summary = pd.DataFrame()
    if not oos_summary.empty:
        st.subheader("Out-of-Sample Holdout")
        st.dataframe(oos_summary, width="stretch", hide_index=True)
    st.subheader("Ablation Summary")
    st.dataframe(pd.DataFrame(result.ablation_summary), width="stretch", hide_index=True)

with tabs[4]:
    st.subheader("Raw Market Sample")
    st.dataframe(result.market_data.tail(200), width="stretch", hide_index=True)
    st.subheader("News Feature Sample")
    st.dataframe(result.news_features.tail(100), width="stretch", hide_index=True)
    st.subheader("SEC Feature Sample")
    st.dataframe(result.sec_features.tail(100), width="stretch", hide_index=True)
