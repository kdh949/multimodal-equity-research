from __future__ import annotations

from streamlit.testing.v1 import AppTest


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


def test_streamlit_app_renders_beginner_dashboard_in_synthetic_mode() -> None:
    app = AppTest.from_file("app.py", default_timeout=90)
    app.run()

    assert not app.exception

    data_mode_select = next(node for node in app.selectbox if node.label == "Data mode")
    sentiment_select = next(node for node in app.selectbox if node.label == "Sentiment model")
    filing_extractor_select = next(node for node in app.selectbox if node.label == "Filing extractor")

    assert data_mode_select.value == "synthetic"
    assert sentiment_select.value == "keyword"
    assert filing_extractor_select.value == "rules"

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
        and ("자료 부족" in str(node.value) or ":gray" in str(node.value))
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
