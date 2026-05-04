from __future__ import annotations

from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_beginner_dashboard_in_synthetic_mode() -> None:
    app = AppTest.from_file("app.py", default_timeout=90)

    app.run()

    assert not app.exception

    assert app.selectbox[0].value == "synthetic"
    assert app.selectbox[2].value == "keyword"
    assert app.selectbox[4].value == "rules"

    markdown_values = [markdown.value for markdown in app.markdown]
    assert any("Beginner Research Overview" in markdown.value for markdown in app.markdown)
    assert any("연구용 리서치 화면이며 투자 권고가 아닙니다" in markdown for markdown in markdown_values)
    assert any("**방향성**" in markdown for markdown in markdown_values)
    assert any("**위험도**" in markdown for markdown in markdown_values)
    assert any("**공시 영향**" in markdown for markdown in markdown_values)
    assert any("**검증 신뢰도**" in markdown for markdown in markdown_values)

    assert any("Forecast Interval" in subheader.value for subheader in app.subheader)
    assert any("SEC Filing Impact" in subheader.value for subheader in app.subheader)

    forecast_column = app.main.children[4].children[0]
    forecast_nodes = list(forecast_column.children.values())
    has_forecast_chart = any(node.__class__.__name__ == "UnknownElement" for node in forecast_nodes)
    has_forecast_fallback = any(
        node.__class__.__name__ == "Markdown" and "자료 부족" in node.value for node in forecast_nodes
    )
    has_forecast_fallback = has_forecast_fallback or any(
        node.__class__.__name__ == "Markdown" and ":gray" in node.value and "자료 부족" in node.value
        for node in forecast_nodes
    )
    assert has_forecast_chart or has_forecast_fallback

    sec_column = app.main.children[4].children[1]
    sec_nodes = list(sec_column.children.values())
    has_sec_events = any(
        node.__class__.__name__ == "Markdown" and "·" in str(node.value) for node in sec_nodes
    )
    has_sec_fallback = any(
        node.__class__.__name__ == "Caption" and "표시할 SEC 이벤트 카드가 없습니다." in str(node.value)
        for node in sec_nodes
    )
    assert has_sec_events or has_sec_fallback

    detail_tables = [dataframe.value for dataframe in app.dataframe if "raw_signal" in dataframe.value.columns]
    assert detail_tables
    assert detail_tables[0]["raw_signal"].iloc[0] in {"BUY", "SELL", "HOLD"}

    for dataframe in app.dataframe:
        assert "action" not in dataframe.value.columns
