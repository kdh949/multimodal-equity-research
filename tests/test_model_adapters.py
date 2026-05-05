from __future__ import annotations

import sys
import types
from contextlib import contextmanager

import numpy as np
import pandas as pd
import requests

import quant_research.models.tabular as tabular
import quant_research.pipeline as pipeline
from quant_research.features.text import KeywordSentimentAnalyzer
from quant_research.models.ollama import OllamaAgent
from quant_research.models.tabular import TabularReturnModel
from quant_research.models.text import (
    FilingEventExtractor,
    FinBERTSentimentAnalyzer,
    FinGPTEventExtractor,
)
from quant_research.models.timeseries import Chronos2Adapter, GraniteTTMAdapter

PipelineConfig = pipeline.PipelineConfig


def test_chronos2_proxy_forecast_columns_are_added() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                ]
            ),
            "ticker": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"],
            "return_20": [0.0, 0.05, -0.02, -0.03, 0.01, 0.02],
            "volatility_20": [0.12, 0.10, 0.11, 0.14, 0.13, None],
        }
    )

    output = Chronos2Adapter().add_proxy_forecasts(frame)

    assert {
        "chronos_expected_return",
        "chronos_downside_quantile",
        "chronos_upside_quantile",
        "chronos_quantile_width",
    } <= set(output.columns)

    momentum = frame.groupby("ticker")["return_20"].transform(lambda series: series.shift(1).fillna(0) / 20)
    volatility = frame["volatility_20"].fillna(frame["volatility_20"].median()).fillna(0.02)

    pd.testing.assert_series_equal(output["chronos_expected_return"], momentum, check_names=False)
    pd.testing.assert_series_equal(
        output["chronos_downside_quantile"],
        momentum - 1.28 * volatility,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        output["chronos_upside_quantile"],
        momentum + 1.28 * volatility,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        output["chronos_quantile_width"],
        output["chronos_upside_quantile"] - output["chronos_downside_quantile"],
        check_names=False,
    )


def test_chronos2_local_inference_normalizes_pipeline_output(monkeypatch) -> None:
    class FakeChronosPipeline:
        def predict_df(self, context_df, **kwargs):
            assert {"id", "timestamp", "target"} <= set(context_df.columns)
            assert kwargs["prediction_length"] == 1
            return pd.DataFrame(
                {
                    "id": ["AAPL", "MSFT"],
                    "timestamp": pd.to_datetime(["2026-02-01", "2026-02-01"]),
                    "0.1": [-0.02, -0.03],
                    "0.5": [0.01, 0.02],
                    "0.9": [0.04, 0.05],
                }
            )

    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    frame = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["AAPL"] * 10 + ["MSFT"] * 10,
            "return_1": [0.001] * 20,
            "return_20": [0.01] * 20,
            "volatility_20": [0.02] * 20,
        }
    )
    adapter = Chronos2Adapter()
    monkeypatch.setattr(adapter, "_load_pipeline", lambda: FakeChronosPipeline())

    output = adapter.add_local_forecasts(frame, min_context=5, max_inference_windows=1)
    latest = output[output["date"] == dates[-1]].set_index("ticker")

    assert latest.loc["AAPL", "chronos_expected_return"] == 0.01
    assert latest.loc["MSFT", "chronos_downside_quantile"] == -0.03
    assert latest.loc["MSFT", "chronos_upside_quantile"] == 0.05


def test_granite_ttm_proxy_forecast_columns_are_added() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                ]
            ),
            "ticker": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"],
            "return_5": [0.01, 0.00, 0.04, -0.01, 0.03, 0.02],
            "high_low_range": [1.2, 0.8, 1.0, 0.5, 0.6, 0.9],
        }
    )

    output = GraniteTTMAdapter().add_proxy_forecasts(frame)

    assert {"granite_ttm_expected_return", "granite_ttm_confidence"} <= set(output.columns)

    short_signal = frame.groupby("ticker")["return_5"].transform(lambda series: series.shift(1).fillna(0) / 5)
    pd.testing.assert_series_equal(
        output["granite_ttm_expected_return"],
        short_signal - 0.1 * frame["high_low_range"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        output["granite_ttm_confidence"],
        (1 / (1 + np.exp(-abs(short_signal) * 100))).clip(0, 1),
        check_names=False,
    )


def test_granite_ttm_local_inference_uses_forecaster_contract(monkeypatch) -> None:
    class FakeForecaster:
        def fit(self, series):
            assert len(series) >= 5
            return self

        def predict(self, fh):
            assert fh == [1]
            return pd.Series([0.0123])

    dates = pd.date_range("2026-01-01", periods=8, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAPL"] * len(dates),
            "return_1": [0.001] * len(dates),
            "return_5": [0.002] * len(dates),
            "high_low_range": [0.01] * len(dates),
        }
    )
    adapter = GraniteTTMAdapter()
    monkeypatch.setattr(adapter, "_load_forecaster", lambda: FakeForecaster())

    output = adapter.add_local_forecasts(frame, min_context=5, max_inference_windows=1)
    latest = output[output["date"] == dates[-1]].iloc[0]

    assert latest["granite_ttm_expected_return"] == 0.0123
    assert latest["granite_ttm_confidence"] == 1.0


def test_tabular_model_fallback_schema_when_lightgbm_is_absent(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "lightgbm", types.ModuleType("lightgbm"))

    train = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=40, freq="D"),
            "ticker": ["AAPL"] * 20 + ["MSFT"] * 20,
            "volatility_20": [0.01 + i * 0.001 for i in range(40)],
            "return_5": [0.001 * (i - 20) for i in range(40)],
            "return_20": [0.002 * (i - 10) for i in range(40)],
            "forward_return_1": [0.001 * (i % 5 - 2) for i in range(40)],
        }
    )

    model = TabularReturnModel(model_name="lightgbm", random_state=7).fit(train)
    assert model.actual_model_name == "HistGradientBoostingRegressor"

    predictions = model.predict(train)
    assert {
        "date",
        "ticker",
        "expected_return",
        "predicted_volatility",
        "downside_quantile",
        "upside_quantile",
        "quantile_width",
        "model_confidence",
        "model_name",
    } <= set(predictions.columns)


def test_tabular_model_falls_back_when_lightgbm_native_library_fails(monkeypatch) -> None:
    class BrokenLightGBM(types.ModuleType):
        def __getattr__(self, name):
            if name == "LGBMRegressor":
                raise OSError("libomp.dylib missing")
            raise AttributeError(name)

    monkeypatch.setitem(sys.modules, "lightgbm", BrokenLightGBM("lightgbm"))
    train = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=40, freq="D"),
            "ticker": ["AAPL"] * 40,
            "volatility_20": [0.02] * 40,
            "return_5": [0.001] * 40,
            "return_20": [0.002] * 40,
            "forward_return_1": [0.001] * 40,
        }
    )

    model = TabularReturnModel(model_name="lightgbm", random_state=7).fit(train)

    assert model.actual_model_name == "HistGradientBoostingRegressor"


def test_tabular_model_falls_back_when_lightgbm_subprocess_fails(monkeypatch) -> None:
    FakeLGBM = type(
        "LGBMRegressor",
        (),
        {
            "fit": lambda self, X, y, sample_weight=None: self,
            "predict": lambda self, X: np.zeros(len(X)),
        },
    )

    def fake_make_estimator(model_name: str, random_state: int, num_threads: int = 1):
        del model_name, random_state, num_threads
        return FakeLGBM()

    def fake_lightgbm_subprocess(**kwargs):
        del kwargs
        return {"success": False, "reason": "child exception: simulated"}

    monkeypatch.setattr(tabular, "_make_estimator", fake_make_estimator)
    monkeypatch.setattr(tabular, "_run_lightgbm_subprocess", fake_lightgbm_subprocess)

    train = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=40, freq="D"),
            "ticker": ["AAPL"] * 40,
            "volatility_20": [0.02] * 40,
            "return_5": [0.001] * 40,
            "return_20": [0.002] * 40,
            "forward_return_1": [0.001] * 40,
        }
    )

    model = TabularReturnModel(model_name="lightgbm", random_state=7).fit(train)

    assert model.actual_model_name == "HistGradientBoostingRegressor"
    assert model.training_metadata["tabular_fallback_reason"] == "child exception: simulated"
    assert model.training_metadata["fit_reason"] == "lightgbm_fallback"


def test_tabular_model_falls_back_when_lightgbm_subprocess_times_out(monkeypatch) -> None:
    FakeLGBM = type(
        "LGBMRegressor",
        (),
        {
            "fit": lambda self, X, y, sample_weight=None: self,
            "predict": lambda self, X: np.zeros(len(X)),
        },
    )

    def fake_make_estimator(model_name: str, random_state: int, num_threads: int = 1):
        del model_name, random_state, num_threads
        return FakeLGBM()

    def fake_lightgbm_subprocess(**kwargs):
        del kwargs
        return {"success": False, "reason": "timeout"}

    monkeypatch.setattr(tabular, "_make_estimator", fake_make_estimator)
    monkeypatch.setattr(tabular, "_run_lightgbm_subprocess", fake_lightgbm_subprocess)

    train = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=40, freq="D"),
            "ticker": ["AAPL"] * 40,
            "volatility_20": [0.02] * 40,
            "return_5": [0.001] * 40,
            "return_20": [0.002] * 40,
            "forward_return_1": [0.001] * 40,
        }
    )

    model = TabularReturnModel(model_name="lightgbm", random_state=7, native_model_timeout_seconds=5).fit(train)

    assert model.actual_model_name == "HistGradientBoostingRegressor"
    assert model.training_metadata["tabular_fallback_reason"] == "timeout"


def test_tabular_model_returns_calibration_metadata_and_raw_predictions() -> None:
    train = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=80, freq="D"),
            "ticker": ["AAPL"] * 80,
            "volatility_20": [0.01 + (0.0001 * i if i < 40 else 40) for i in range(80)],
            "return_5": [0.001 * (i % 7 - 3) for i in range(80)],
            "return_20": [0.002 * (i % 5 - 2) for i in range(80)],
            "forward_return_1": [0.001 * (i % 3 - 1) for i in range(80)],
        }
    )
    model = TabularReturnModel(model_name="auto", recent_weight_power=1.2, random_state=11).fit(train)
    preds = model.predict(train.tail(10))

    assert {"raw_expected_return", "expected_return", "model_calibration_scale", "model_calibration_bias"}.issubset(
        set(preds.columns)
    )
    assert {"winsorized_feature_count", "calibration_scale", "calibration_bias"}.issubset(model.training_metadata)
    assert float(model.training_metadata["winsorized_feature_count"]) >= 0


def test_finbert_uses_no_network_fallback_contract(monkeypatch) -> None:
    fake_transformers = types.ModuleType("transformers")

    def failing_pipeline(*args, **kwargs):
        raise RuntimeError("transformers unavailable in this environment")

    fake_transformers.pipeline = failing_pipeline
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    analyzer = FinBERTSentimentAnalyzer()
    result = analyzer.score("Company announced stronger guidance and improving demand.")

    assert set(result) == {
        "sentiment_score",
        "negative_flag",
        "label",
        "confidence",
        "event_tag",
        "risk_flag",
    }
    assert isinstance(result["sentiment_score"], float)
    assert isinstance(result["confidence"], float)
    assert isinstance(result["negative_flag"], bool)
    assert isinstance(result["risk_flag"], bool)


def test_filing_and_fingpt_extractor_follow_structured_contract_without_services() -> None:
    for extractor in (FilingEventExtractor(), FinGPTEventExtractor()):
        result = extractor.extract("Company reported an earnings guidance update with potential legal risk.")
        assert set(result) == {"event_tag", "risk_flag", "confidence", "summary_ref"}
        assert isinstance(result["event_tag"], str)
        assert isinstance(result["risk_flag"], bool)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["summary_ref"], str)


def test_finma_local_generation_output_is_validated(monkeypatch) -> None:
    extractor = FilingEventExtractor(use_local_model=True)
    monkeypatch.setattr(
        extractor,
        "_generate_with_local_model",
        lambda text: '{"event_tag":"guidance","risk_flag":true,"confidence":0.82,"summary_ref":"guidance risk"}',
    )

    result = extractor.extract("The filing mentions guidance risk.")

    assert result == {
        "event_tag": "guidance",
        "risk_flag": True,
        "confidence": 0.82,
        "summary_ref": "guidance risk",
    }
    assert extractor.last_source == "local"
    assert extractor.last_error is None


def test_fingpt_local_generation_falls_back_to_rules_on_bad_json(monkeypatch) -> None:
    extractor = FinGPTEventExtractor(use_local_model=True)
    monkeypatch.setattr(extractor, "_generate_with_local_model", lambda text: "not json")

    result = extractor.extract("Form 8-K material legal risk")

    assert result["event_tag"] != "none"
    assert result["risk_flag"] is True
    assert extractor.last_source == "rules"
    assert "ValueError" in str(extractor.last_error)


def test_fingpt_mlx_requires_runtime_path_or_repo_id(monkeypatch) -> None:
    extractor = FinGPTEventExtractor(
        use_local_model=True,
        runtime="mlx",
        runtime_model_path=None,
        model_id="",
    )
    result = extractor.extract("Form 8-K material update")

    assert result["event_tag"] != "none"
    assert extractor.last_source == "rules"
    assert "runtime_model_path is required for FinGPT MLX runtime" in str(extractor.last_error)


def test_fingpt_llama_cpp_path_validation(monkeypatch) -> None:
    extractor = FinGPTEventExtractor(
        use_local_model=True,
        runtime="llama.cpp",
        runtime_model_path="not-a-quantized-model.bin",
    )
    result = extractor.extract("Material risk disclosure in the filing.")

    assert extractor.last_source == "rules"
    assert "GGUF/quantized model file path" in str(extractor.last_error)
    assert isinstance(result["event_tag"], str)


def test_fingpt_single_load_lock_fails_fast_without_duplicate_load(monkeypatch) -> None:
    @contextmanager
    def locked_context(_: str | None):
        raise RuntimeError("another process is loading the local filing LLM")
        yield

    extractor = FinGPTEventExtractor(use_local_model=True, allow_unquantized_transformers=True)
    monkeypatch.setattr("quant_research.models.text._acquire_model_load_lock", locked_context)
    monkeypatch.setattr(extractor, "_load_transformer_runtime", lambda: ("tokenizer", "model"))

    result = extractor.extract("Guidance risk guidance guidance")

    assert extractor.last_source == "rules"
    assert "another process is loading" in str(extractor.last_error)
    assert result["risk_flag"] is True


def test_fingpt_explicit_single_load_lock_path_is_used(monkeypatch, tmp_path) -> None:
    seen: dict[str, str | None] = {}

    @contextmanager
    def capture_lock_path(lock_path: str | None):
        seen["path"] = lock_path
        yield

    lock_path = tmp_path / "locks" / "fingpt.lock"
    extractor = FinGPTEventExtractor(
        use_local_model=True,
        allow_unquantized_transformers=True,
        single_load_lock_path=str(lock_path),
    )
    monkeypatch.setattr("quant_research.models.text._acquire_model_load_lock", capture_lock_path)
    monkeypatch.setattr(extractor, "_load_transformer_runtime", lambda: ("tokenizer", "model"))

    assert extractor._load_local_model() == ("tokenizer", "model")
    assert seen["path"] == str(lock_path)


def test_ollama_agent_falls_back_when_server_unreachable(monkeypatch) -> None:
    def failing_request(*args, **kwargs):
        raise requests.RequestException("server unreachable")

    monkeypatch.setattr(requests, "post", failing_request)

    response = OllamaAgent().explain("Summarize risk in this sample text.")
    assert response == "Ollama is unavailable; deterministic metrics and signals were generated locally."


def test_sentiment_analyzer_prefers_finbert_when_available(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeFinBERT:
        def __init__(self, *args, **kwargs) -> None:
            captured["args"] = kwargs

        def available(self) -> bool:
            return True

        def score(self, text: str) -> dict[str, float | str | bool]:
            return {
                "sentiment_score": 0.42,
                "negative_flag": False,
                "label": "positive",
                "confidence": 0.88,
                "event_tag": "none",
                "risk_flag": False,
            }

    monkeypatch.setattr(pipeline, "FinBERTSentimentAnalyzer", FakeFinBERT)
    analyzer = pipeline._sentiment_analyzer("finbert")

    assert isinstance(analyzer, FakeFinBERT)
    assert captured["args"]["local_files_only"] is True


def test_sentiment_analyzer_falls_back_to_keyword_when_finbert_unavailable(monkeypatch) -> None:
    class UnavailableFinBERT:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def available(self) -> bool:
            return False

        def score(self, text: str) -> dict[str, float | str | bool]:
            return {
                "sentiment_score": 0.0,
                "negative_flag": False,
                "label": "neutral",
                "confidence": 0.0,
                "event_tag": "",
                "risk_flag": False,
            }

    monkeypatch.setattr(pipeline, "FinBERTSentimentAnalyzer", UnavailableFinBERT)

    analyzer = pipeline._sentiment_analyzer("finbert")

    assert isinstance(analyzer, KeywordSentimentAnalyzer)


def test_filing_extractor_prefers_fingpt_adapter_when_available(monkeypatch) -> None:
    class FakeFinGPT:
        def __init__(self, *args, **kwargs) -> None:
            self.kwargs = kwargs

        def available(self) -> bool:
            return True

        def extract(self, text: str) -> dict[str, object]:
            return {
                "event_tag": "none",
                "risk_flag": False,
                "confidence": 0.8,
                "summary_ref": text[:20],
            }

    monkeypatch.setattr(pipeline, "FinGPTEventExtractor", FakeFinGPT)
    extractor = pipeline._filing_extractor(
        PipelineConfig(
            filing_extractor_model="fingpt",
            enable_local_filing_llm=True,
            fingpt_runtime="llama-cpp",
            fingpt_quantized_model_path="artifacts/model_cache/fingpt-test.gguf",
        )
    )

    assert isinstance(extractor, FakeFinGPT)
    assert extractor.kwargs["runtime"] == "llama-cpp"


def test_filing_extractor_falls_back_to_rules_when_fingpt_unavailable(monkeypatch) -> None:
    class UnavailableFinGPT:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def available(self) -> bool:
            return False

        def extract(self, text: str) -> dict[str, object]:
            return {
                "event_tag": "none",
                "risk_flag": False,
                "confidence": 0.5,
                "summary_ref": "fallback",
            }

    monkeypatch.setattr(pipeline, "FinGPTEventExtractor", UnavailableFinGPT)
    extractor = pipeline._filing_extractor(
        PipelineConfig(
            filing_extractor_model="fingpt",
            enable_local_filing_llm=True,
            fingpt_runtime="transformers",
            fingpt_base_model_id=None,
        )
    )

    assert isinstance(extractor, FilingEventExtractor)
    assert extractor.use_local_model is False


def test_resolve_timeseries_mode_forces_proxy_when_adapter_unavailable() -> None:
    class OfflineAdapter:
        def available(self) -> bool:
            return False

    assert pipeline._resolve_timeseries_mode("local", OfflineAdapter()) == "proxy"


def test_resolve_timeseries_mode_keeps_local_when_adapter_available() -> None:
    class OnlineAdapter:
        def available(self) -> bool:
            return True

    assert pipeline._resolve_timeseries_mode("local", OnlineAdapter()) == "local"
