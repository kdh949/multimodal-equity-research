from __future__ import annotations

import pandas as pd
from streamlit.testing.v1 import AppTest

import quant_research.pipeline as pipeline
from quant_research.backtest.engine import BacktestResult
from quant_research.backtest.metrics import PerformanceMetrics
from quant_research.pipeline import PipelineResult


def _build_stub_result() -> PipelineResult:
    dates = pd.date_range("2026-01-01", periods=6, freq="D")
    focus_ticker = "SPY"
    market_data = pd.DataFrame(
        {
            "date": dates.repeat(2),
            "ticker": [focus_ticker, focus_ticker] * len(dates),
            "close": [440, 441] * len(dates),
            "forward_return_1": [0.005, 0.004] * len(dates),
        }
    )
    news_features = pd.DataFrame(
        {
            "date": dates,
            "ticker": [focus_ticker] * len(dates),
            "news_article_count": [1] * len(dates),
            "news_sentiment_mean": [0.02] * len(dates),
            "news_negative_ratio": [0.0] * len(dates),
            "news_source_count": [1] * len(dates),
            "news_source_diversity": [1.0] * len(dates),
            "news_event_count": [0] * len(dates),
            "news_top_event": ["none"] * len(dates),
            "text_risk_score": [0.0] * len(dates),
            "news_confidence_mean": [0.6] * len(dates),
            "news_token_count_mean": [12.0] * len(dates),
            "news_text_length": [40.0] * len(dates),
            "news_full_text_available_ratio": [1.0] * len(dates),
            "news_recency_decay": [0.0] * len(dates),
            "news_staleness_days": [0.0] * len(dates),
            "news_coverage_5d": [3.0] * len(dates),
            "news_coverage_20d": [12.0] * len(dates),
        }
    )
    sec_features = pd.DataFrame(
        {
            "date": dates,
            "ticker": [focus_ticker] * len(dates),
            "sec_event_tag": ["none"] * len(dates),
            "sec_risk_flag": [0.0] * len(dates),
            "sec_risk_flag_20d": [0.0] * len(dates),
            "sec_event_confidence": [0.0] * len(dates),
            "sec_summary_ref": [""] * len(dates),
        }
    )
    predictions = pd.DataFrame(
        {
            "date": dates,
            "ticker": [focus_ticker] * len(dates),
            "action": ["HOLD"] * len(dates),
            "signal_score": [0.0] * len(dates),
            "expected_return": [0.01] * len(dates),
            "predicted_volatility": [0.02] * len(dates),
            "downside_quantile": [-0.015] * len(dates),
            "upside_quantile": [0.035] * len(dates),
            "text_risk_score": [0.0] * len(dates),
            "sec_risk_flag": [0.0] * len(dates),
            "sec_risk_flag_20d": [0.0] * len(dates),
            "text_risk_flag": [0.0] * len(dates),
            "sec_event_confidence": [0.0] * len(dates),
        }
    )
    signals = predictions.copy()
    features = market_data.copy()
    validation_summary = pd.DataFrame({"is_oos": [True], "directional_accuracy": [0.62]})
    ablation_summary = [
        {
            "scenario": "all_features",
            "kind": "signal",
            "cagr": 0.01,
            "sharpe": 0.3,
            "max_drawdown": -0.01,
            "excess_return": 0.008,
        }
    ]
    backtest = BacktestResult(
        equity_curve=pd.DataFrame(
            {
                "date": dates,
                "equity": [1.0, 1.01, 1.02, 1.03, 1.04, 1.05],
                "benchmark_equity": [1.0, 1.0, 1.01, 1.01, 1.01, 1.02],
                "portfolio_return": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                "benchmark_return": [0.0, 0.001, 0.0, 0.0, 0.0, 0.001],
                "turnover": [0.0] * len(dates),
                "exposure": [0.2] * len(dates),
                "gross_return": [0.01] * len(dates),
                "return_date": dates,
            }
        ),
        weights=pd.DataFrame(
            {
                "signal_date": pd.to_datetime(dates),
                "effective_date": pd.to_datetime(dates),
                "ticker": [focus_ticker] * len(dates),
                "weight": [0.0] * len(dates),
            }
        ),
        signals=signals,
        metrics=PerformanceMetrics(
            cagr=0.03,
            annualized_volatility=0.12,
            sharpe=0.9,
            max_drawdown=-0.02,
            hit_rate=0.58,
            turnover=0.0,
            exposure=0.2,
            benchmark_cagr=0.02,
            excess_return=0.01,
        ),
    )
    return PipelineResult(
        market_data=market_data,
        news_features=news_features,
        sec_features=sec_features,
        features=features,
        predictions=predictions,
        signals=signals,
        validation_summary=validation_summary,
        ablation_summary=ablation_summary,
        backtest=backtest,
    )

def _child_nodes(node: object) -> list[object]:
    children = getattr(node, "children", None)
    if isinstance(children, dict):
        return list(children.values())
    if isinstance(children, list):
        return children
    return []


def _collect_dataframes(node: object) -> list[object]:
    if node.__class__.__name__ == "Dataframe":
        return [node]

    tables: list[object] = []
    for child in _child_nodes(node):
        tables.extend(_collect_dataframes(child))
    return tables


def _is_tab_container(node: object) -> bool:
    return node.__class__.__name__ == "Block" and getattr(node, "type", None) == "tab_container"


def _has_evidence(captions: list[str], keys: tuple[str, ...]) -> bool:
    return any(any(key in line for key in keys) for line in captions)


def _run_fake_research_pipeline(
    captured: dict[str, object],
    config: pipeline.PipelineConfig,
) -> PipelineResult:
    captured["calls"] = captured.get("calls", 0) + 1
    captured["config"] = config
    return _build_stub_result()


def _assert_full_stack_defaults(
    app: object,
    *,
    data_mode_select_value: str = "synthetic",
    sentiment_select_value: str = "finbert",
    time_series_select_value: str = "local",
    filing_extractor_value: str = "fingpt",
) -> None:
    data_mode_select = next(node for node in app.selectbox if node.label == "Data mode")
    sentiment_select = next(node for node in app.selectbox if node.label == "Sentiment model")
    time_series_select = next(node for node in app.selectbox if node.label == "Time-series inference")
    filing_extractor_select = next(node for node in app.selectbox if node.label == "Filing extractor")

    assert data_mode_select.value == data_mode_select_value
    assert sentiment_select.value == sentiment_select_value
    assert time_series_select.value == time_series_select_value
    assert filing_extractor_select.value == filing_extractor_value
    assert any(node.label == "Use local filing LLM" and node.value is True for node in app.checkbox)


def _assert_beginner_dashboard_rendered(app: object, captured: dict[str, object]) -> None:
    data_mode_select = next(node for node in app.selectbox if node.label == "Data mode")
    sentiment_select = next(node for node in app.selectbox if node.label == "Sentiment model")
    time_series_select = next(node for node in app.selectbox if node.label == "Time-series inference")
    filing_extractor_select = next(node for node in app.selectbox if node.label == "Filing extractor")
    config = captured["config"]
    assert isinstance(config, pipeline.PipelineConfig)
    assert config.sentiment_model == "finbert"
    assert config.filing_extractor_model == "fingpt"
    assert config.enable_local_filing_llm is True
    assert config.time_series_inference_mode == "local"

    assert data_mode_select.value == "synthetic"
    assert sentiment_select.value == "finbert"
    assert time_series_select.value == "local"
    assert filing_extractor_select.value == "fingpt"
    markdown_values = [str(markdown.value) for markdown in app.markdown]
    caption_values = [str(caption.value) for caption in app.caption]
    metric_labels = {metric.label for metric in app.metric}

    assert any("Beginner Research Overview" in markdown for markdown in markdown_values)
    assert any(
        "연구용 리서치 화면이며 투자 권고가 아닙니다. 실거래 주문 기능은 제공하지 않습니다." in markdown
        for markdown in markdown_values
    )

    assert any("**방향성**" in markdown for markdown in markdown_values)
    assert any("**위험도**" in markdown for markdown in markdown_values)
    assert any("**공시 영향**" in markdown for markdown in markdown_values)
    assert any("**검증 신뢰도**" in markdown for markdown in markdown_values)

    assert _has_evidence(caption_values, ("expected_return", "downside_quantile"))
    assert _has_evidence(caption_values, ("predicted_volatility", "max_drawdown"))
    assert _has_evidence(caption_values, ("risk_flag", "event_tag", "confidence"))
    assert _has_evidence(caption_values, ("is_oos", "sharpe", "hit_rate"))

    assert any("Forecast Interval" in subheader.value for subheader in app.subheader)
    assert any("SEC Filing Impact" in subheader.value for subheader in app.subheader)
    assert any("Backtest Validation Snapshot" in subheader.value for subheader in app.subheader)
    assert any("Validation Metrics" in markdown for markdown in markdown_values)
    assert {"CAGR", "Sharpe", "Max DD", "Hit Rate", "Exposure", "Turnover"}.issubset(metric_labels)

    runtime_options = [node.value for node in app.selectbox if node.label == "FinGPT runtime"]
    assert runtime_options and runtime_options[0] in {"transformers", "mlx", "llama-cpp"}
    assert any(
        node.label == "Allow unquantized Transformers 8B load" and node.value is False for node in app.checkbox
    )
    assert any(node.label == "FinGPT quantized runtime path" for node in app.text_input)

    forecast_column = app.main.children[4].children[0]
    forecast_nodes = _child_nodes(forecast_column)
    has_forecast_chart = any(node.__class__.__name__ == "UnknownElement" for node in forecast_nodes)
    has_forecast_fallback = any(
        node.__class__.__name__ == "Caption"
        and (
            "자료 부족" in str(node.value)
            or ":gray" in str(node.value)
            or "expected_return" in str(node.value)
            or "downside_quantile" in str(node.value)
        )
        for node in forecast_nodes
    )
    has_forecast_fallback = has_forecast_fallback or any(
        node.__class__.__name__ == "Markdown"
        and ("expected_return" in str(node.value) or "downside_quantile" in str(node.value))
        for node in forecast_nodes
    )
    assert has_forecast_chart or has_forecast_fallback

    sec_column = app.main.children[4].children[1]
    sec_nodes = _child_nodes(sec_column)
    has_sec_events = any(node.__class__.__name__ == "Markdown" and "·" in str(node.value) for node in sec_nodes)
    has_sec_fallback = any(
        node.__class__.__name__ == "Caption" and "표시할 SEC 이벤트 카드가 없습니다." in str(node.value)
        for node in sec_nodes
    )
    has_sec_fallback = has_sec_fallback or any(
        node.__class__.__name__ == "Caption" and "predicted_volatility" in str(node.value)
        for node in sec_nodes
    )
    has_sec_fallback = has_sec_fallback or any(
        node.__class__.__name__ == "Markdown" and "위험도" in str(node.value)
        for node in sec_nodes
    )
    assert has_sec_events or has_sec_fallback

    detail_tables = [dataframe.value for dataframe in app.dataframe if "action" in dataframe.value.columns]
    assert detail_tables
    assert detail_tables[0]["action"].dropna().astype(str).str.upper().isin({"BUY", "SELL", "HOLD"}).all()

    first_screen_nodes = [
        node
        for key, node in app.main.children.items()
        if not _is_tab_container(node)
    ]
    first_screen_tables = [
        table
        for node in first_screen_nodes
        for table in _collect_dataframes(node)
    ]
    for table in first_screen_tables:
        assert "action" not in table.value.columns
        assert "raw_signal" not in table.value.columns


def test_streamlit_app_does_not_auto_run_on_first_render(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(pipeline, "run_research_pipeline", lambda config: _run_fake_research_pipeline(captured, config))
    app = AppTest.from_file("app.py", default_timeout=90)
    app.run()

    assert not app.exception
    assert "result" not in app.session_state
    assert captured.get("calls", 0) == 0
    _assert_full_stack_defaults(app)

    caption_values = [str(caption.value) for caption in app.caption]
    assert any(
        "선택 항목은 기본값으로 구성되며" in caption
        for caption in caption_values
    )


def test_streamlit_app_runs_pipeline_only_when_run_button_clicked(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(pipeline, "run_research_pipeline", lambda config: _run_fake_research_pipeline(captured, config))
    app = AppTest.from_file("app.py", default_timeout=90)
    app.run()

    run_button = next(node for node in app.button if node.label == "Run research")
    run_button.click().run()

    assert captured.get("calls", 0) == 1
    assert "result" in app.session_state
    _assert_full_stack_defaults(app)
    _assert_beginner_dashboard_rendered(app, captured)

    # Re-run without click should keep latest result and not rerun pipeline.
    app.run()
    assert captured.get("calls", 0) == 1
