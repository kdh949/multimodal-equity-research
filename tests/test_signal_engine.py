from __future__ import annotations

import pandas as pd
import pytest

from quant_research.signals.engine import (
    DeterministicSignalEngine,
    SignalEngineConfig,
    SignalGenerationBlockedError,
    require_signal_generation_gate_pass,
)


def test_signal_engine_buys_only_after_risk_checks() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"] * 3),
            "ticker": ["GOOD", "RISK", "WEAK"],
            "expected_return": [0.02, 0.03, 0.0001],
            "predicted_volatility": [0.02, 0.02, 0.02],
            "downside_quantile": [-0.01, -0.01, -0.01],
            "text_risk_score": [0.0, 0.9, 0.0],
            "sec_risk_flag": [0.0, 0.0, 0.0],
            "liquidity_score": [20.0, 20.0, 20.0],
            "model_confidence": [0.8, 0.8, 0.8],
        }
    )

    signals = DeterministicSignalEngine(SignalEngineConfig()).generate(frame)
    actions = dict(zip(signals["ticker"], signals["action"], strict=True))

    assert actions["GOOD"] == "BUY"
    assert actions["RISK"] == "SELL"
    assert actions["WEAK"] == "HOLD"


def test_signal_engine_buy_label_threshold_boundaries_are_inclusive() -> None:
    config = SignalEngineConfig(
        min_expected_return=0.02,
        min_signal_score=0.02,
        max_predicted_volatility=0.05,
        min_downside_quantile=-0.03,
        max_text_risk_score=0.60,
        block_buy_sec_risk_flag=1.0,
        block_buy_sec_risk_20d=1.0,
        max_news_negative_ratio=0.50,
        min_liquidity_score=10.0,
        cost_bps=0.0,
        slippage_bps=0.0,
        confidence_weight=0.0,
        volatility_penalty_weight=0.0,
        text_risk_penalty_weight=0.0,
        sec_risk_penalty_weight=0.0,
        sec_risk_20d_penalty_weight=0.0,
        news_negative_penalty_weight=0.0,
        downside_penalty_weight=0.0,
        covariance_aware_risk_enabled=False,
    )
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"] * 4),
            "ticker": ["AT_THRESHOLD", "LOW_EXPECTED", "HIGH_VOL", "LOW_LIQUIDITY"],
            "expected_return": [0.02, 0.0199, 0.02, 0.02],
            "predicted_volatility": [0.05, 0.05, 0.0501, 0.05],
            "downside_quantile": [-0.03, -0.03, -0.03, -0.03],
            "text_risk_score": [0.60, 0.60, 0.60, 0.60],
            "sec_risk_flag": [0.0, 0.0, 0.0, 0.0],
            "sec_risk_flag_20d": [0.0, 0.0, 0.0, 0.0],
            "news_negative_ratio": [0.50, 0.50, 0.50, 0.50],
            "liquidity_score": [10.0, 10.0, 10.0, 9.99],
            "model_confidence": [0.0, 0.0, 0.0, 0.0],
        }
    )

    signals = DeterministicSignalEngine(config).generate(frame)
    actions = dict(zip(signals["ticker"], signals["action"], strict=True))

    assert actions == {
        "AT_THRESHOLD": "BUY",
        "LOW_EXPECTED": "HOLD",
        "HIGH_VOL": "HOLD",
        "LOW_LIQUIDITY": "HOLD",
    }


def test_signal_engine_risk_label_threshold_boundaries_are_directional() -> None:
    config = SignalEngineConfig(
        min_expected_return=0.02,
        min_signal_score=0.02,
        max_predicted_volatility=0.05,
        min_downside_quantile=-0.03,
        max_text_risk_score=0.60,
        max_sec_risk_flag=2.0,
        max_sec_risk_20d=4.0,
        block_buy_sec_risk_flag=1.0,
        block_buy_sec_risk_20d=1.0,
        max_news_negative_ratio=0.50,
        min_liquidity_score=10.0,
        cost_bps=0.0,
        slippage_bps=0.0,
        confidence_weight=0.0,
        volatility_penalty_weight=0.0,
        text_risk_penalty_weight=0.0,
        sec_risk_penalty_weight=0.0,
        sec_risk_20d_penalty_weight=0.0,
        news_negative_penalty_weight=0.0,
        downside_penalty_weight=0.0,
        covariance_aware_risk_enabled=False,
    )
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"] * 6),
            "ticker": [
                "TEXT_AT_MAX",
                "TEXT_ABOVE_MAX",
                "SEC_AT_BUY_BLOCK",
                "SEC_AT_SELL",
                "SEC_20D_AT_SELL",
                "DOWNSIDE_BELOW_MIN",
            ],
            "expected_return": [0.02] * 6,
            "predicted_volatility": [0.05] * 6,
            "downside_quantile": [-0.03, -0.03, -0.03, -0.03, -0.03, -0.0301],
            "text_risk_score": [0.60, 0.6001, 0.0, 0.0, 0.0, 0.0],
            "sec_risk_flag": [0.0, 0.0, 1.0, 2.0, 0.0, 0.0],
            "sec_risk_flag_20d": [0.0, 0.0, 0.0, 0.0, 4.0, 0.0],
            "news_negative_ratio": [0.50] * 6,
            "liquidity_score": [10.0] * 6,
            "model_confidence": [0.0] * 6,
        }
    )

    signals = DeterministicSignalEngine(config).generate(frame)
    actions = dict(zip(signals["ticker"], signals["action"], strict=True))

    assert actions == {
        "TEXT_AT_MAX": "BUY",
        "TEXT_ABOVE_MAX": "SELL",
        "SEC_AT_BUY_BLOCK": "HOLD",
        "SEC_AT_SELL": "SELL",
        "SEC_20D_AT_SELL": "SELL",
        "DOWNSIDE_BELOW_MIN": "SELL",
    }


def test_signal_engine_blocks_low_liquidity() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"]),
            "ticker": ["THIN"],
            "expected_return": [0.05],
            "predicted_volatility": [0.01],
            "downside_quantile": [0.0],
            "text_risk_score": [0.0],
            "sec_risk_flag": [0.0],
            "liquidity_score": [1.0],
            "model_confidence": [1.0],
        }
    )

    signals = DeterministicSignalEngine(SignalEngineConfig()).generate(frame)

    assert signals["action"].iloc[0] == "HOLD"


def test_signal_engine_throttles_sec_risk_without_immediate_sell() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-01"]),
            "ticker": ["LOW_SEC", "MODERATE_SEC", "SEVERE_SEC"],
            "expected_return": [0.05, 0.05, 0.05],
            "predicted_volatility": [0.01, 0.01, 0.01],
            "downside_quantile": [0.0, 0.0, 0.0],
            "text_risk_score": [0.0, 0.0, 0.0],
            "sec_risk_flag": [0.0, 1.0, 0.0],
            "sec_risk_flag_20d": [0.5, 1.0, 4.0],
            "news_negative_ratio": [0.0, 0.0, 0.0],
            "liquidity_score": [20.0, 20.0, 20.0],
            "model_confidence": [1.0, 1.0, 1.0],
        }
    )

    signals = DeterministicSignalEngine(SignalEngineConfig()).generate(frame)
    actions = dict(zip(signals["ticker"], signals["action"], strict=True))

    assert actions["LOW_SEC"] == "BUY"
    assert actions["MODERATE_SEC"] == "HOLD"
    assert actions["SEVERE_SEC"] == "SELL"


def test_signal_engine_still_sells_on_severe_text_or_news_risk() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "ticker": ["TEXT", "NEWS"],
            "expected_return": [0.05, 0.05],
            "predicted_volatility": [0.01, 0.01],
            "downside_quantile": [0.0, 0.0],
            "text_risk_score": [0.9, 0.0],
            "sec_risk_flag": [0.0, 0.0],
            "sec_risk_flag_20d": [0.0, 0.0],
            "news_negative_ratio": [0.0, 1.0],
            "liquidity_score": [20.0, 20.0],
            "model_confidence": [1.0, 1.0],
        }
    )

    signals = DeterministicSignalEngine(SignalEngineConfig()).generate(frame)

    assert set(signals["action"]) == {"SELL"}


def test_signal_engine_score_weights_are_configurable() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"]),
            "ticker": ["AAPL"],
            "expected_return": [0.03],
            "predicted_volatility": [0.02],
            "downside_quantile": [0.0],
            "text_risk_score": [0.0],
            "sec_risk_flag": [0.0],
            "sec_risk_flag_20d": [0.0],
            "news_negative_ratio": [0.0],
            "liquidity_score": [20.0],
            "model_confidence": [1.0],
        }
    )

    neutral = DeterministicSignalEngine(
        SignalEngineConfig(volatility_penalty_weight=0.0)
    ).score(frame)
    penalized = DeterministicSignalEngine(
        SignalEngineConfig(volatility_penalty_weight=1.0)
    ).score(frame)

    assert neutral["signal_score"].iloc[0] > penalized["signal_score"].iloc[0]


def test_signal_engine_applies_portfolio_risk_metric_penalty() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "ticker": ["CALM", "STRESSED"],
            "expected_return": [0.02, 0.02],
            "predicted_volatility": [0.01, 0.01],
            "downside_quantile": [0.0, 0.0],
            "text_risk_score": [0.0, 0.0],
            "sec_risk_flag": [0.0, 0.0],
            "sec_risk_flag_20d": [0.0, 0.0],
            "news_negative_ratio": [0.0, 0.0],
            "liquidity_score": [20.0, 20.0],
            "model_confidence": [1.0, 1.0],
            "portfolio_volatility_estimate": [0.02, 0.10],
            "average_daily_turnover": [0.10, 0.50],
            "max_symbol_weight": [0.08, 0.20],
            "max_sector_weight": [0.20, 0.50],
            "current_drawdown": [-0.05, -0.30],
        }
    )

    signals = DeterministicSignalEngine(SignalEngineConfig()).generate(frame)
    calm = signals.set_index("ticker").loc["CALM"]
    stressed = signals.set_index("ticker").loc["STRESSED"]

    assert calm["risk_metric_penalty"] == pytest.approx(0.0)
    assert stressed["risk_metric_penalty"] == pytest.approx(0.0545)
    assert calm["signal_score"] > stressed["signal_score"]
    assert calm["action"] == "BUY"
    assert stressed["action"] == "HOLD"


def test_signal_engine_can_disable_covariance_aware_risk_penalty() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"]),
            "ticker": ["AAPL"],
            "expected_return": [0.05],
            "predicted_volatility": [0.01],
            "downside_quantile": [0.0],
            "text_risk_score": [0.0],
            "sec_risk_flag": [0.0],
            "sec_risk_flag_20d": [0.0],
            "news_negative_ratio": [0.0],
            "liquidity_score": [20.0],
            "model_confidence": [1.0],
            "portfolio_volatility_estimate": [0.20],
            "average_daily_turnover": [0.75],
            "max_symbol_weight": [0.40],
            "max_sector_weight": [0.80],
            "current_drawdown": [-0.40],
        }
    )

    signals = DeterministicSignalEngine(
        SignalEngineConfig(covariance_aware_risk_enabled=False)
    ).generate(frame)

    assert signals["risk_metric_penalty"].iloc[0] == pytest.approx(0.0)
    assert bool(signals["covariance_aware_risk_enabled"].iloc[0]) is False


def test_signal_engine_risk_scoring_uses_only_point_in_time_portfolio_metrics() -> None:
    base = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"]),
            "ticker": ["AAPL"],
            "expected_return": [0.05],
            "predicted_volatility": [0.01],
            "downside_quantile": [0.0],
            "text_risk_score": [0.0],
            "sec_risk_flag": [0.0],
            "sec_risk_flag_20d": [0.0],
            "news_negative_ratio": [0.0],
            "liquidity_score": [20.0],
            "model_confidence": [1.0],
            "portfolio_volatility_estimate": [0.02],
            "average_daily_turnover": [0.10],
            "max_symbol_weight": [0.08],
            "max_sector_weight": [0.20],
            "current_drawdown": [-0.05],
            "forward_return_20": [-0.99],
            "max_drawdown": [-0.99],
            "future_portfolio_volatility_estimate": [0.99],
            "future_average_daily_turnover": [2.0],
        }
    )
    changed_future_metrics = base.assign(
        forward_return_20=0.99,
        max_drawdown=0.0,
        future_portfolio_volatility_estimate=0.0,
        future_average_daily_turnover=0.0,
    )

    engine = DeterministicSignalEngine(SignalEngineConfig())
    base_signal = engine.generate(base)
    changed_signal = engine.generate(changed_future_metrics)

    assert base_signal["risk_metric_penalty"].iloc[0] == pytest.approx(0.0)
    assert changed_signal["risk_metric_penalty"].iloc[0] == pytest.approx(0.0)
    assert base_signal["signal_score"].iloc[0] == changed_signal["signal_score"].iloc[0]
    assert base_signal["action"].iloc[0] == changed_signal["action"].iloc[0] == "BUY"

    posthoc_drawdown_only = base.drop(columns=["current_drawdown"])
    changed_posthoc_drawdown = changed_future_metrics.drop(columns=["current_drawdown"])
    posthoc_signal = engine.generate(posthoc_drawdown_only)
    changed_posthoc_signal = engine.generate(changed_posthoc_drawdown)

    assert posthoc_signal["risk_metric_penalty"].iloc[0] == pytest.approx(0.0)
    assert changed_posthoc_signal["risk_metric_penalty"].iloc[0] == pytest.approx(0.0)
    assert posthoc_signal["signal_score"].iloc[0] == changed_posthoc_signal["signal_score"].iloc[0]


def test_signal_engine_generate_preserves_output_contract_with_risk_scoring() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"]),
            "ticker": ["AAPL"],
            "expected_return": [0.05],
            "predicted_volatility": [0.01],
            "downside_quantile": [0.0],
            "text_risk_score": [0.0],
            "sec_risk_flag": [0.0],
            "sec_risk_flag_20d": [0.0],
            "news_negative_ratio": [0.0],
            "liquidity_score": [20.0],
            "model_confidence": [1.0],
            "portfolio_volatility_estimate": [0.02],
            "average_daily_turnover": [0.10],
            "max_symbol_weight": [0.08],
            "max_sector_weight": [0.20],
            "current_drawdown": [-0.05],
        }
    )

    signals = DeterministicSignalEngine(SignalEngineConfig()).generate(frame)

    assert frame.columns.tolist() == [
        column for column in signals.columns if column in frame.columns
    ]
    assert set(signals.columns).difference(frame.columns) == {
        "signal_score",
        "risk_metric_penalty",
        "covariance_aware_risk_enabled",
        "portfolio_covariance_lookback",
        "covariance_return_column",
        "covariance_min_periods",
        "portfolio_volatility_limit",
        "average_daily_turnover_budget",
        "configured_max_symbol_weight",
        "configured_max_sector_weight",
        "action",
    }
    assert pd.api.types.is_numeric_dtype(signals["signal_score"])
    assert pd.api.types.is_numeric_dtype(signals["risk_metric_penalty"])
    assert set(signals["action"]).issubset({"BUY", "SELL", "HOLD"})


def test_signal_engine_penalizes_active_risk_stop_without_realized_returns() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"]),
            "ticker": ["STOPPED"],
            "expected_return": [0.05],
            "predicted_volatility": [0.01],
            "downside_quantile": [0.0],
            "text_risk_score": [0.0],
            "sec_risk_flag": [0.0],
            "sec_risk_flag_20d": [0.0],
            "news_negative_ratio": [0.0],
            "liquidity_score": [20.0],
            "model_confidence": [1.0],
            "risk_stop_active": [True],
            "forward_return_20": [0.99],
        }
    )

    signals = DeterministicSignalEngine(SignalEngineConfig()).generate(frame)

    assert signals["risk_metric_penalty"].iloc[0] == pytest.approx(1.0)
    assert signals["action"].iloc[0] == "HOLD"


def test_signal_engine_does_not_use_realized_return_columns_for_actions() -> None:
    base = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"]),
            "ticker": ["AAPL"],
            "expected_return": [0.05],
            "predicted_volatility": [0.01],
            "downside_quantile": [0.0],
            "text_risk_score": [0.0],
            "sec_risk_flag": [0.0],
            "sec_risk_flag_20d": [0.0],
            "news_negative_ratio": [0.0],
            "liquidity_score": [20.0],
            "model_confidence": [1.0],
            "return_1": [-0.99],
            "forward_return_1": [-0.99],
            "forward_return_20": [-0.99],
        }
    )
    changed_realized_returns = base.assign(
        return_1=0.99,
        forward_return_1=0.99,
        forward_return_20=0.99,
    )

    engine = DeterministicSignalEngine(SignalEngineConfig())
    base_signal = engine.generate(base)
    changed_signal = engine.generate(changed_realized_returns)

    assert base_signal["action"].iloc[0] == "BUY"
    assert changed_signal["action"].iloc[0] == "BUY"
    assert base_signal["signal_score"].iloc[0] == changed_signal["signal_score"].iloc[0]


def test_signal_engine_ignores_report_only_top_decile_metric_for_scores_and_actions() -> None:
    base = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "ticker": ["AAPL", "MSFT"],
            "expected_return": [0.05, 0.04],
            "predicted_volatility": [0.01, 0.01],
            "downside_quantile": [0.0, 0.0],
            "text_risk_score": [0.0, 0.0],
            "sec_risk_flag": [0.0, 0.0],
            "sec_risk_flag_20d": [0.0, 0.0],
            "news_negative_ratio": [0.0, 0.0],
            "liquidity_score": [20.0, 20.0],
            "model_confidence": [1.0, 1.0],
        }
    )
    with_report_only_metric = base.assign(
        top_decile_20d_excess_return=[-100.0, 100.0],
    )

    engine = DeterministicSignalEngine(SignalEngineConfig())
    base_signal = engine.generate(base)
    changed_signal = engine.generate(with_report_only_metric)

    assert base_signal["signal_score"].tolist() == changed_signal["signal_score"].tolist()
    assert base_signal["action"].tolist() == changed_signal["action"].tolist()


def test_signal_engine_rejects_features_unavailable_at_signal_date() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-05"]),
            "ticker": ["AAPL"],
            "expected_return": [0.05],
            "predicted_volatility": [0.01],
            "downside_quantile": [0.0],
            "text_risk_score": [0.0],
            "sec_risk_flag": [0.0],
            "sec_risk_flag_20d": [0.0],
            "news_negative_ratio": [0.0],
            "liquidity_score": [20.0],
            "model_confidence": [1.0],
            "news_availability_timestamp": [
                pd.Timestamp("2026-01-06 00:00:00", tz="UTC")
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="signal engine input column news_availability_timestamp contains data unavailable",
    ):
        DeterministicSignalEngine(SignalEngineConfig()).generate(frame)


def test_signal_engine_rejects_predictions_created_after_signal_date() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-05"]),
            "ticker": ["AAPL"],
            "expected_return": [0.05],
            "predicted_volatility": [0.01],
            "downside_quantile": [0.0],
            "text_risk_score": [0.0],
            "sec_risk_flag": [0.0],
            "sec_risk_flag_20d": [0.0],
            "news_negative_ratio": [0.0],
            "liquidity_score": [20.0],
            "model_confidence": [1.0],
            "model_prediction_timestamp": [
                pd.Timestamp("2026-01-06 00:00:00", tz="UTC")
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="model_prediction_timestamp is later than signal date",
    ):
        DeterministicSignalEngine(SignalEngineConfig()).generate(frame)


def test_signal_engine_requires_common_gate_when_final_generation_demands_it() -> None:
    with pytest.raises(SignalGenerationBlockedError, match="validation gate is required"):
        DeterministicSignalEngine(SignalEngineConfig()).generate(
            _passing_signal_frame(),
            require_validation_gate=True,
        )


def test_signal_engine_blocks_final_generation_when_common_gate_is_not_pass() -> None:
    gate = {
        "validity_gate_result_summary": {
            "final_gate_decision": "WARN",
            "strategy_candidate_status": "warning",
            "reason": "proxy improvement requires review",
        }
    }

    with pytest.raises(
        SignalGenerationBlockedError,
        match="final_gate_decision=WARN",
    ):
        DeterministicSignalEngine(SignalEngineConfig()).generate(
            _passing_signal_frame(),
            validation_gate=gate,
            require_validation_gate=True,
        )


@pytest.mark.parametrize(
    ("gate", "expected_decision", "expected_status"),
    [
        (
            {
                "validity_gate_result_summary": {
                    "final_gate_decision": "WARN",
                    "final_status": "warning",
                    "strategy_candidate_status": "warning",
                    "reason": "model value warning requires human review",
                }
            },
            "WARN",
            "warning",
        ),
        (
            {
                "validity_gate_result_summary": {
                    "final_gate_decision": "FAIL",
                    "final_status": "fail",
                    "strategy_candidate_status": "fail",
                    "reason": "strategy candidate rule failed",
                }
            },
            "FAIL",
            "fail",
        ),
        (
            {
                "system_validity_status": "hard_fail",
                "strategy_candidate_status": "not_evaluable",
                "official_message": "system validity hard-failed",
            },
            "FAIL",
            "not_evaluable",
        ),
        (
            {
                "validity_gate_result_summary": {
                    "final_gate_decision": "FAIL",
                    "final_status": "insufficient_data",
                    "system_validity_status": "not_evaluable",
                    "strategy_candidate_status": "insufficient_data",
                    "reason": "required OOS folds are unavailable",
                }
            },
            "FAIL",
            "insufficient_data",
        ),
        (
            {
                "system_validity_status": "not_evaluable",
                "strategy_candidate_status": "not_evaluable",
                "official_message": "required validation data is missing",
            },
            "FAIL",
            "not_evaluable",
        ),
    ],
)
def test_signal_engine_blocks_final_generation_for_each_non_pass_common_gate_state(
    gate: dict[str, object],
    expected_decision: str,
    expected_status: str,
) -> None:
    with pytest.raises(SignalGenerationBlockedError) as exc_info:
        DeterministicSignalEngine(SignalEngineConfig()).generate(
            _passing_signal_frame(),
            validation_gate=gate,
            require_validation_gate=True,
        )

    message = str(exc_info.value)
    assert "validation gate blocked final signal generation" in message
    assert f"final_gate_decision={expected_decision}" in message
    assert f"final_status={expected_status}" in message


def test_signal_engine_allows_final_generation_only_after_common_gate_pass() -> None:
    gate = {
        "validity_gate_result_summary": {
            "final_gate_decision": "PASS",
            "system_validity_status": "pass",
            "strategy_candidate_status": "pass",
        }
    }

    signals = DeterministicSignalEngine(SignalEngineConfig()).generate(
        _passing_signal_frame(),
        validation_gate=gate,
        require_validation_gate=True,
    )

    assert signals["action"].iloc[0] == "BUY"
    assert signals["signal_generation_gate_decision"].iloc[0] == "PASS"
    assert signals["signal_generation_gate_status"].iloc[0] == "pass"


def test_common_signal_gate_accepts_serialized_report_without_final_decision() -> None:
    payload = require_signal_generation_gate_pass(
        {
            "system_validity_status": "pass",
            "strategy_candidate_status": "pass",
        },
        required=True,
    )

    assert payload is not None
    assert payload["final_gate_decision"] == "PASS"


def _passing_signal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"]),
            "ticker": ["PASSING"],
            "expected_return": [0.05],
            "predicted_volatility": [0.01],
            "downside_quantile": [0.0],
            "text_risk_score": [0.0],
            "sec_risk_flag": [0.0],
            "sec_risk_flag_20d": [0.0],
            "news_negative_ratio": [0.0],
            "liquidity_score": [20.0],
            "model_confidence": [1.0],
        }
    )
