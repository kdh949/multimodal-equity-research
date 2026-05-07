from __future__ import annotations

import pandas as pd
import pytest

import quant_research.backtest.engine as backtest_engine
from quant_research.backtest.engine import BacktestConfig, run_long_only_backtest
from quant_research.signals.engine import (
    DeterministicSignalEngine,
    SignalGenerationBlockedError,
)


def test_backtest_caps_symbol_weight() -> None:
    frame = _prediction_frame(
        returns=[0.01, 0.01],
        tickers=["AAPL", "MSFT"],
        dates=["2026-01-01", "2026-01-01"],
    )

    result = run_long_only_backtest(
        frame,
        BacktestConfig(top_n=2, max_symbol_weight=0.25, portfolio_volatility_limit=1.0),
    )

    assert not result.weights.empty
    assert result.weights["weight"].max() <= 0.25
    assert result.equity_curve["exposure"].iloc[0] == 0.5


def test_backtest_scales_portfolio_volatility() -> None:
    frame = _prediction_frame(
        returns=[0.01, 0.01],
        tickers=["AAPL", "MSFT"],
        dates=["2026-01-01", "2026-01-01"],
        volatility=0.10,
    )

    result = run_long_only_backtest(
        frame,
        BacktestConfig(
            top_n=2,
            max_symbol_weight=1.0,
            portfolio_volatility_limit=0.02,
        ),
    )

    assert result.equity_curve["portfolio_volatility_estimate"].iloc[0] <= 0.0200001


def test_backtest_applies_configurable_sector_concentration_limit() -> None:
    frame = _prediction_frame(
        returns=[0.01, 0.01],
        tickers=["AAPL", "MSFT"],
        dates=["2026-01-01", "2026-01-01"],
    )
    frame["sector"] = ["Information Technology", "Information Technology"]

    result = run_long_only_backtest(
        frame,
        BacktestConfig(
            top_n=2,
            max_symbol_weight=0.50,
            max_sector_weight=0.30,
            portfolio_volatility_limit=1.0,
        ),
    )

    assert result.equity_curve["exposure"].iloc[0] == pytest.approx(0.30)
    assert result.weights["weight"].sum() == pytest.approx(0.30)


def test_backtest_records_post_cost_position_sizing_validation_rule() -> None:
    frame = _prediction_frame(
        returns=[0.01, 0.02],
        tickers=["AAPL", "MSFT"],
        dates=["2026-01-01", "2026-01-01"],
    )
    frame["sector"] = ["Information Technology", "Communication Services"]

    result = run_long_only_backtest(
        frame,
        BacktestConfig(
            top_n=2,
            max_symbol_weight=0.10,
            max_sector_weight=0.30,
            cost_bps=5.0,
            slippage_bps=2.0,
            portfolio_volatility_limit=1.0,
        ),
    )

    row = result.equity_curve.iloc[0]
    assert row["position_sizing_validation_status"] == "pass"
    assert row["position_sizing_validation_rule"] == "post_cost_position_sizing_constraints_v1"
    assert row["position_count"] == 2
    assert row["max_position_weight"] <= 0.10
    assert row["max_sector_exposure"] <= 0.30
    assert row["gross_exposure"] <= 1.0
    assert row["net_exposure"] == pytest.approx(row["gross_exposure"])
    assert row["post_cost_validation_total_cost_return"] == pytest.approx(
        row["total_cost_return"]
    )
    assert result.metrics.position_sizing_validation_status == "pass"
    assert result.metrics.position_sizing_validation_pass_rate == pytest.approx(1.0)
    assert (
        result.metrics.position_sizing_validation_rule
        == "post_cost_position_sizing_constraints_v1"
    )
    assert result.metrics.max_position_weight <= 0.10
    assert result.metrics.max_sector_exposure <= 0.30
    assert result.metrics.max_position_risk_contribution >= 0.0
    assert result.metrics.max_portfolio_volatility_estimate == pytest.approx(
        result.equity_curve["portfolio_volatility_estimate"].max()
    )
    assert result.metrics.average_portfolio_volatility_estimate == pytest.approx(
        result.equity_curve["portfolio_volatility_estimate"].mean()
    )


def test_backtest_rejects_post_cost_position_sizing_leverage_breach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _prediction_frame(
        returns=[0.01],
        tickers=["AAPL"],
        dates=["2026-01-01"],
    )

    def overlevered_targets(
        previous_weights: dict[str, float],
        target_weights: dict[str, float],
        config: BacktestConfig,
    ) -> dict[str, float]:
        return {"AAPL": 1.20}

    monkeypatch.setattr(backtest_engine, "_apply_turnover_limit", overlevered_targets)

    with pytest.raises(ValueError, match="post-cost position sizing validation failed"):
        run_long_only_backtest(
            frame,
            BacktestConfig(
                top_n=1,
                max_symbol_weight=1.0,
                max_sector_weight=1.0,
                portfolio_volatility_limit=1.0,
            ),
        )


def test_backtest_risk_contribution_limit_has_configurable_adjustment_strength() -> None:
    frame = _prediction_frame(
        returns=[0.01, 0.01],
        tickers=["AAPL", "MSFT"],
        dates=["2026-01-01", "2026-01-01"],
    )
    frame["predicted_volatility"] = [0.07, 0.01]

    result = run_long_only_backtest(
        frame,
        BacktestConfig(
            top_n=2,
            max_symbol_weight=0.50,
            max_sector_weight=1.0,
            max_position_risk_contribution=0.50,
            risk_contribution_adjustment_strength=1.0,
            portfolio_volatility_limit=1.0,
        ),
    )

    weights = result.weights.set_index("ticker")["weight"].to_dict()
    assert weights["AAPL"] < 0.50
    assert weights["MSFT"] == pytest.approx(0.50)


def test_backtest_rejects_invalid_risk_adjustment_strength() -> None:
    with pytest.raises(ValueError, match="risk_contribution_adjustment_strength"):
        BacktestConfig(risk_contribution_adjustment_strength=1.1)


def test_backtest_drawdown_stop_forces_cash_after_breach() -> None:
    frame = _prediction_frame(
        returns=[-0.50, 0.10, 0.10],
        tickers=["AAPL", "AAPL", "AAPL"],
        dates=["2026-01-01", "2026-01-02", "2026-01-05"],
    )

    result = run_long_only_backtest(
        frame,
        BacktestConfig(top_n=1, max_symbol_weight=1.0, max_drawdown_stop=0.20),
    )

    assert result.equity_curve["risk_stop_active"].iloc[1]
    assert result.equity_curve["exposure"].iloc[1] == 0.0


def test_backtest_records_next_period_weight_timing() -> None:
    frame = _prediction_frame(
        returns=[0.01, 0.01],
        tickers=["AAPL", "AAPL"],
        dates=["2026-01-01", "2026-01-02"],
    )

    result = run_long_only_backtest(
        frame,
        BacktestConfig(top_n=1, max_symbol_weight=1.0, realized_return_column="forward_return_1"),
    )

    assert {"signal_date", "effective_date", "ticker", "weight"}.issubset(result.weights.columns)
    assert result.weights["effective_date"].iloc[0] > result.weights["signal_date"].iloc[0]


def test_backtest_applies_signal_date_position_to_configured_forward_return_only() -> None:
    dates = pd.bdate_range("2026-01-01", periods=2)
    frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAPL", "AAPL"],
            "expected_return": [0.05, 0.05],
            "predicted_volatility": [0.01, 0.01],
            "downside_quantile": [0.0, 0.0],
            "model_confidence": [1.0, 1.0],
            "text_risk_score": [0.0, 0.0],
            "sec_risk_flag": [0.0, 0.0],
            "sec_risk_flag_20d": [0.0, 0.0],
            "news_negative_ratio": [0.0, 0.0],
            "liquidity_score": [20.0, 20.0],
            "return_1": [-0.99, -0.99],
            "forward_return_1": [0.07, 0.0],
        }
    )

    result = run_long_only_backtest(
        frame,
        BacktestConfig(
            top_n=1,
            max_symbol_weight=1.0,
            cost_bps=0.0,
            slippage_bps=0.0,
            realized_return_column="forward_return_1",
        ),
    )

    assert result.equity_curve["return_date"].iloc[0] == dates[1]
    assert result.equity_curve["return_date"].iloc[0] > result.equity_curve["date"].iloc[0]
    assert result.equity_curve["holding_start_date"].iloc[0] == dates[1]
    assert result.equity_curve["holding_start_date"].iloc[0] > result.equity_curve["date"].iloc[0]
    assert result.equity_curve["portfolio_return"].iloc[0] == pytest.approx(0.07)


def test_backtest_position_returns_are_labeled_after_signal_date() -> None:
    dates = pd.bdate_range("2026-01-01", periods=3)
    frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "expected_return": [0.05, 0.05, 0.05],
            "predicted_volatility": [0.01, 0.01, 0.01],
            "downside_quantile": [0.0, 0.0, 0.0],
            "model_confidence": [1.0, 1.0, 1.0],
            "text_risk_score": [0.0, 0.0, 0.0],
            "sec_risk_flag": [0.0, 0.0, 0.0],
            "sec_risk_flag_20d": [0.0, 0.0, 0.0],
            "news_negative_ratio": [0.0, 0.0, 0.0],
            "liquidity_score": [20.0, 20.0, 20.0],
            "return_1": [-0.99, -0.99, -0.99],
            "forward_return_1": [0.07, 0.11, 0.13],
        }
    )

    result = run_long_only_backtest(
        frame,
        BacktestConfig(
            top_n=1,
            max_symbol_weight=1.0,
            cost_bps=0.0,
            slippage_bps=0.0,
            realized_return_column="forward_return_1",
        ),
    )

    assert result.weights["holding_start_date"].gt(result.weights["signal_date"]).all()
    assert result.weights["effective_date"].ge(result.weights["holding_start_date"]).all()
    assert result.equity_curve["holding_start_date"].gt(result.equity_curve["date"]).all()
    assert result.equity_curve["return_date"].ge(result.equity_curve["holding_start_date"]).all()
    assert result.weights["realized_return"].tolist() == pytest.approx([0.07, 0.11, 0.13])
    assert result.equity_curve["portfolio_return"].tolist() == pytest.approx([0.07, 0.11, 0.13])


def test_backtest_lags_single_day_signal_to_next_business_day_forward_label() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02"]),
            "ticker": ["AAPL"],
            "expected_return": [0.05],
            "predicted_volatility": [0.01],
            "downside_quantile": [0.0],
            "model_confidence": [1.0],
            "text_risk_score": [0.0],
            "sec_risk_flag": [0.0],
            "sec_risk_flag_20d": [0.0],
            "news_negative_ratio": [0.0],
            "liquidity_score": [20.0],
            "return_1": [-0.99],
            "forward_return_1": [0.07],
        }
    )

    result = run_long_only_backtest(
        frame,
        BacktestConfig(
            top_n=1,
            max_symbol_weight=1.0,
            cost_bps=0.0,
            slippage_bps=0.0,
            realized_return_column="forward_return_1",
        ),
    )

    assert result.equity_curve["holding_start_date"].iloc[0] == pd.Timestamp("2026-01-05")
    assert result.equity_curve["return_date"].iloc[0] == pd.Timestamp("2026-01-05")
    assert result.equity_curve["holding_start_date"].iloc[0] > result.equity_curve["date"].iloc[0]
    assert result.equity_curve["portfolio_return"].iloc[0] == pytest.approx(0.07)


def test_backtest_rejects_same_day_position_application_metadata() -> None:
    frame = _prediction_frame(
        returns=[0.01],
        tickers=["AAPL"],
        dates=["2026-01-02"],
    ).assign(
        signal_date=[pd.Timestamp("2026-01-02")],
        holding_start_date=[pd.Timestamp("2026-01-02")],
        return_label_date=[pd.Timestamp("2026-01-05")],
        realized_return_column=["forward_return_1"],
        return_horizon=[1],
    )

    with pytest.raises(ValueError, match="holding_start_date must be after signal_date"):
        run_long_only_backtest(
            frame,
            BacktestConfig(
                top_n=1,
                max_symbol_weight=1.0,
                realized_return_column="forward_return_1",
            ),
        )


def test_backtest_rejects_return_label_before_future_holding_window() -> None:
    frame = _prediction_frame(
        returns=[0.01],
        tickers=["AAPL"],
        dates=["2026-01-02"],
    ).assign(
        signal_date=[pd.Timestamp("2026-01-02")],
        holding_start_date=[pd.Timestamp("2026-01-05")],
        return_label_date=[pd.Timestamp("2026-01-02")],
        realized_return_column=["forward_return_1"],
        return_horizon=[1],
    )

    with pytest.raises(ValueError, match="return label date must be on or after holding_start_date"):
        run_long_only_backtest(
            frame,
            BacktestConfig(
                top_n=1,
                max_symbol_weight=1.0,
                realized_return_column="forward_return_1",
            ),
        )


def test_backtest_rejects_same_day_return_label_metadata_without_holding_start() -> None:
    frame = _prediction_frame(
        returns=[0.01],
        tickers=["AAPL"],
        dates=["2026-01-02"],
    ).assign(
        signal_date=[pd.Timestamp("2026-01-02")],
        return_date=[pd.Timestamp("2026-01-02")],
        realized_return_column=["forward_return_1"],
        return_horizon=[1],
    )

    with pytest.raises(ValueError, match="return label date must be after signal_date"):
        run_long_only_backtest(
            frame,
            BacktestConfig(
                top_n=1,
                max_symbol_weight=1.0,
                realized_return_column="forward_return_1",
            ),
        )


def test_backtest_rejects_return_horizon_metadata_mismatch() -> None:
    frame = _prediction_frame(
        returns=[0.01],
        tickers=["AAPL"],
        dates=["2026-01-02"],
    ).assign(
        signal_date=[pd.Timestamp("2026-01-02")],
        holding_start_date=[pd.Timestamp("2026-01-05")],
        return_label_date=[pd.Timestamp("2026-01-30")],
        realized_return_column=["forward_return_20"],
        return_horizon=[1],
    )

    with pytest.raises(ValueError, match="return_horizon must match configured forward_return_20"):
        run_long_only_backtest(
            frame,
            BacktestConfig(
                top_n=1,
                max_symbol_weight=1.0,
                realized_return_column="forward_return_20",
            ),
        )


def test_backtest_normalizes_prediction_rows_before_signal_generation() -> None:
    frame = _prediction_frame(
        returns=[0.02, 0.03, 0.04],
        tickers=[" MSFT ", "AAPL", "AAPL"],
        dates=["2026-01-05", "2026-01-02", "2026-01-05"],
    )

    result = run_long_only_backtest(
        frame.sample(frac=1.0, random_state=7),
        BacktestConfig(top_n=1, max_symbol_weight=1.0, cost_bps=0.0, slippage_bps=0.0),
    )

    assert result.signals["date"].tolist() == sorted(result.signals["date"].tolist())
    assert " MSFT " not in result.signals["ticker"].tolist()
    assert result.equity_curve["date"].tolist() == sorted(result.equity_curve["date"].tolist())


def test_backtest_rejects_features_unavailable_at_signal_date() -> None:
    frame = _prediction_frame(
        returns=[0.01],
        tickers=["AAPL"],
        dates=["2026-01-02"],
    ).assign(
        news_availability_timestamp=[pd.Timestamp("2026-01-03 00:00:00", tz="UTC")],
    )

    with pytest.raises(ValueError, match="unavailable at signal date"):
        run_long_only_backtest(frame, BacktestConfig(top_n=1, max_symbol_weight=1.0))


def test_backtest_rejects_predictions_generated_after_signal_date() -> None:
    frame = _prediction_frame(
        returns=[0.01],
        tickers=["AAPL"],
        dates=["2026-01-02"],
    ).assign(
        prediction_timestamp=[pd.Timestamp("2026-01-03 00:00:00", tz="UTC")],
    )

    with pytest.raises(ValueError, match="later than signal date"):
        run_long_only_backtest(frame, BacktestConfig(top_n=1, max_symbol_weight=1.0))


def test_backtest_signal_engine_receives_no_realized_return_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_columns: list[str] = []
    original_generate = DeterministicSignalEngine.generate

    def capture_generate(self: DeterministicSignalEngine, frame: pd.DataFrame) -> pd.DataFrame:
        captured_columns.extend(frame.columns.tolist())
        return original_generate(self, frame)

    monkeypatch.setattr(DeterministicSignalEngine, "generate", capture_generate)
    frame = _prediction_frame(
        returns=[-0.99],
        tickers=["AAPL"],
        dates=["2026-01-02"],
    ).assign(
        top_decile_20d_excess_return=[0.99],
    )

    run_long_only_backtest(frame, BacktestConfig(top_n=1, max_symbol_weight=1.0))

    assert "return_1" not in captured_columns
    assert "forward_return_20" not in captured_columns
    assert "forward_return_1" not in captured_columns
    assert "top_decile_20d_excess_return" not in captured_columns


def test_backtest_blocks_final_signal_path_when_common_gate_is_not_pass() -> None:
    frame = _prediction_frame(
        returns=[0.01],
        tickers=["AAPL"],
        dates=["2026-01-02"],
    )
    gate = {
        "validity_gate_result_summary": {
            "final_gate_decision": "FAIL",
            "strategy_candidate_status": "fail",
            "reason": "cost-adjusted excess return did not pass",
        }
    }

    with pytest.raises(
        SignalGenerationBlockedError,
        match="validation gate blocked final signal generation",
    ):
        run_long_only_backtest(
            frame,
            BacktestConfig(top_n=1, max_symbol_weight=1.0),
            validation_gate=gate,
            require_validation_gate=True,
        )


def test_backtest_records_common_gate_pass_on_final_signals() -> None:
    frame = _prediction_frame(
        returns=[0.01],
        tickers=["AAPL"],
        dates=["2026-01-02"],
    )
    gate = {
        "validity_gate_result_summary": {
            "final_gate_decision": "PASS",
            "system_validity_status": "pass",
            "strategy_candidate_status": "pass",
        }
    }

    result = run_long_only_backtest(
        frame,
        BacktestConfig(top_n=1, max_symbol_weight=1.0),
        validation_gate=gate,
        require_validation_gate=True,
    )

    assert result.signals["signal_generation_gate_decision"].iloc[0] == "PASS"
    assert result.signals["signal_generation_gate_status"].iloc[0] == "pass"


def test_backtest_generated_validation_signals_are_unchanged_by_warning_metadata() -> None:
    frame = _prediction_frame(
        returns=[0.01, 0.01, 0.01],
        tickers=["BUY_CANDIDATE", "SELL_CANDIDATE", "HOLD_CANDIDATE"],
        dates=["2026-01-02", "2026-01-02", "2026-01-02"],
    )
    frame["expected_return"] = [0.05, 0.05, 0.0001]
    frame["text_risk_score"] = [0.0, 0.9, 0.0]

    pass_gate = {
        "validity_gate_result_summary": {
            "final_gate_decision": "PASS",
            "system_validity_status": "pass",
            "strategy_candidate_status": "pass",
            "reason": "common validation gate passed",
        }
    }
    pass_gate_with_warning_evidence = {
        **pass_gate,
        "warning": True,
        "warnings": [
            "monthly_turnover_budget: realized max monthly turnover exceeded review budget"
        ],
        "structured_warnings": [
            {
                "code": "monthly_turnover_budget_exceeded",
                "severity": "warning",
                "gate": "monthly_turnover_budget",
                "metric": "max_monthly_turnover",
                "value": 0.55,
                "threshold": 0.50,
                "operator": "<=",
            }
        ],
        "gate_results": {
            "monthly_turnover_budget": {
                "status": "warning",
                "reason": "realized max monthly turnover exceeded review budget",
            }
        },
    }
    config = BacktestConfig(top_n=1, max_symbol_weight=1.0)

    baseline = run_long_only_backtest(
        frame,
        config,
        validation_gate=pass_gate,
        require_validation_gate=True,
    )
    with_warning_evidence = run_long_only_backtest(
        frame,
        config,
        validation_gate=pass_gate_with_warning_evidence,
        require_validation_gate=True,
    )

    signal_columns = [
        "date",
        "ticker",
        "signal_score",
        "risk_metric_penalty",
        "action",
    ]
    assert baseline.signals[signal_columns].to_dict("records") == (
        with_warning_evidence.signals[signal_columns].to_dict("records")
    )
    assert with_warning_evidence.signals["signal_generation_gate_decision"].eq("PASS").all()
    assert with_warning_evidence.signals["signal_generation_gate_status"].eq("pass").all()


def test_backtest_ignores_report_only_top_decile_metric_for_position_eligibility() -> None:
    frame = _prediction_frame(
        returns=[0.01, 0.02],
        tickers=["AAPL", "MSFT"],
        dates=["2026-01-02", "2026-01-02"],
    )
    with_report_only_metric = frame.assign(
        top_decile_20d_excess_return=[100.0, -100.0],
    )

    config = BacktestConfig(
        top_n=1,
        max_symbol_weight=1.0,
        cost_bps=0.0,
        slippage_bps=0.0,
        portfolio_volatility_limit=1.0,
    )
    base = run_long_only_backtest(frame, config)
    changed = run_long_only_backtest(with_report_only_metric, config)

    assert base.signals[["ticker", "signal_score", "action"]].to_dict("records") == (
        changed.signals[["ticker", "signal_score", "action"]].to_dict("records")
    )
    assert base.weights[["ticker", "weight"]].to_dict("records") == (
        changed.weights[["ticker", "weight"]].to_dict("records")
    )
    assert base.equity_curve["exposure"].tolist() == pytest.approx(
        changed.equity_curve["exposure"].tolist()
    )


def test_backtest_report_only_top_decile_metric_does_not_change_signals_or_risk_rules() -> None:
    frame = _prediction_frame(
        returns=[-0.50, 0.10, 0.10],
        tickers=["AAPL", "AAPL", "AAPL"],
        dates=["2026-01-01", "2026-01-02", "2026-01-05"],
    )
    base_metric = frame.assign(
        top_decile_20d_excess_return=[100.0, -100.0, 0.0],
    )
    changed_metric = frame.assign(
        top_decile_20d_excess_return=[-100.0, 100.0, 50.0],
    )

    config = BacktestConfig(
        top_n=1,
        max_symbol_weight=1.0,
        cost_bps=0.0,
        slippage_bps=0.0,
        portfolio_volatility_limit=1.0,
        max_drawdown_stop=0.20,
    )
    base = run_long_only_backtest(base_metric, config)
    changed = run_long_only_backtest(changed_metric, config)

    signal_columns = [
        "date",
        "ticker",
        "signal_score",
        "risk_metric_penalty",
        "action",
    ]
    risk_rule_columns = [
        "date",
        "position_sizing_validation_status",
        "position_sizing_validation_rule",
        "position_sizing_validation_reason",
        "risk_stop_active",
        "exposure",
        "position_count",
        "max_position_weight",
        "max_sector_exposure",
        "gross_exposure",
        "net_exposure",
    ]

    assert base.signals[signal_columns].to_dict("records") == (
        changed.signals[signal_columns].to_dict("records")
    )
    assert base.equity_curve[risk_rule_columns].to_dict("records") == (
        changed.equity_curve[risk_rule_columns].to_dict("records")
    )
    assert base.equity_curve["risk_stop_active"].tolist() == [False, True, True]


def test_backtest_uses_external_benchmark_return_series() -> None:
    frame = _prediction_frame(
        returns=[0.01, 0.01],
        tickers=["AAPL", "AAPL"],
        dates=["2026-01-01", "2026-01-02"],
    )
    benchmark_returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "benchmark_return": [0.02, -0.01],
        }
    )

    result = run_long_only_backtest(
        frame,
        BacktestConfig(top_n=1, max_symbol_weight=1.0),
        benchmark_returns=benchmark_returns,
    )

    assert result.equity_curve["benchmark_return"].tolist() == [0.02, -0.01]
    assert result.equity_curve["benchmark_equity"].iloc[-1] == pytest.approx(
        (1.0 + 0.02) * (1.0 - 0.01)
    )
    assert result.equity_curve["benchmark_turnover"].tolist() == pytest.approx([1.0, 0.0])
    assert result.equity_curve["benchmark_total_cost_return"].tolist() == pytest.approx(
        [0.0007, 0.0]
    )
    assert result.equity_curve["cost_adjusted_benchmark_return"].tolist() == pytest.approx(
        [0.02 - 0.0007, -0.01]
    )
    expected_cost_adjusted_benchmark_equity = (1.0 + 0.02 - 0.0007) * (1.0 - 0.01)
    assert result.equity_curve["cost_adjusted_benchmark_equity"].iloc[-1] == pytest.approx(
        expected_cost_adjusted_benchmark_equity
    )
    assert result.metrics.benchmark_cagr == pytest.approx(
        expected_cost_adjusted_benchmark_equity ** (252 / 2) - 1
    )


def test_backtest_returns_structured_empty_result_for_no_predictions() -> None:
    result = run_long_only_backtest(
        pd.DataFrame(),
        BacktestConfig(realized_return_column="forward_return_5"),
    )

    assert result.signals.empty
    assert {"date", "ticker", "expected_return", "action", "forward_return_5"}.issubset(
        result.signals.columns
    )
    assert result.equity_curve.empty
    assert {"date", "equity", "benchmark_equity", "portfolio_return"}.issubset(
        result.equity_curve.columns
    )
    assert result.metrics.cagr == 0


def test_backtest_uses_configured_realized_return_column_and_horizon() -> None:
    dates = pd.bdate_range("2026-01-01", periods=6)
    frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAPL"] * len(dates),
            "expected_return": [0.05] * len(dates),
            "predicted_volatility": [0.01] * len(dates),
            "downside_quantile": [0.0] * len(dates),
            "model_confidence": [1.0] * len(dates),
            "text_risk_score": [0.0] * len(dates),
            "sec_risk_flag": [0.0] * len(dates),
            "sec_risk_flag_20d": [0.0] * len(dates),
            "news_negative_ratio": [0.0] * len(dates),
            "liquidity_score": [20.0] * len(dates),
            "forward_return_1": [0.01] * len(dates),
            "forward_return_5": [0.10, 0.20, 0.0, 0.0, 0.0, 0.0],
        }
    )

    result = run_long_only_backtest(
        frame,
        BacktestConfig(
            top_n=1,
            max_symbol_weight=1.0,
            cost_bps=0.0,
            slippage_bps=0.0,
            realized_return_column="forward_return_5",
        ),
    )

    assert result.equity_curve["portfolio_return"].iloc[:2].tolist() == pytest.approx([0.10, 0.20])
    assert result.equity_curve["realized_return_column"].eq("forward_return_5").all()
    assert result.weights["effective_date"].iloc[0] == dates[5]


def test_backtest_uses_covariance_aware_volatility_for_correlated_holdings() -> None:
    dates = pd.bdate_range("2026-01-01", periods=70)
    market_return = [0.02 if index % 2 == 0 else -0.02 for index in range(len(dates))]
    frame = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "expected_return": 0.05,
                "predicted_volatility": 0.02,
                "downside_quantile": 0.0,
                "model_confidence": 1.0,
                "text_risk_score": 0.0,
                "sec_risk_flag": 0.0,
                "sec_risk_flag_20d": 0.0,
                "news_negative_ratio": 0.0,
                "liquidity_score": 20.0,
                "return_1": market_return[date_index],
                "forward_return_20": 0.01,
            }
            for date_index, date in enumerate(dates)
            for ticker in ("AAPL", "MSFT")
        ]
    )

    result = run_long_only_backtest(
        frame,
        BacktestConfig(
            top_n=2,
            max_symbol_weight=0.5,
            max_correlation_cluster_weight=1.0,
            portfolio_volatility_limit=1.0,
            realized_return_column="forward_return_20",
        ),
    )

    observed = result.equity_curve["portfolio_volatility_estimate"].iloc[-1]
    diagonal = (2 * (0.5 * 0.02) ** 2) ** 0.5
    assert observed > diagonal


def test_backtest_volatility_estimate_falls_back_to_diagonal_without_return_history() -> None:
    frame = _prediction_frame(
        returns=[0.01, 0.01],
        tickers=["AAPL", "MSFT"],
        dates=["2026-01-01", "2026-01-01"],
        volatility=0.02,
    ).drop(columns=["return_1"])

    result = run_long_only_backtest(
        frame,
        BacktestConfig(
            top_n=2,
            max_symbol_weight=0.5,
            portfolio_volatility_limit=1.0,
        ),
    )

    expected_diagonal = (2 * (0.5 * 0.02) ** 2) ** 0.5
    assert result.equity_curve["portfolio_volatility_estimate"].iloc[0] == pytest.approx(
        expected_diagonal
    )


def test_backtest_can_disable_covariance_aware_risk_from_config() -> None:
    dates = pd.bdate_range("2026-01-01", periods=70)
    frame = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "expected_return": 0.05,
                "predicted_volatility": 0.02,
                "downside_quantile": 0.0,
                "model_confidence": 1.0,
                "text_risk_score": 0.0,
                "sec_risk_flag": 0.0,
                "sec_risk_flag_20d": 0.0,
                "news_negative_ratio": 0.0,
                "liquidity_score": 20.0,
                "return_1": 0.02 if date_index % 2 == 0 else -0.02,
                "forward_return_20": 0.01,
            }
            for date_index, date in enumerate(dates)
            for ticker in ("AAPL", "MSFT")
        ]
    )

    result = run_long_only_backtest(
        frame,
        BacktestConfig(
            top_n=2,
            max_symbol_weight=0.5,
            portfolio_volatility_limit=1.0,
            covariance_aware_risk_enabled=False,
        ),
    )

    expected_diagonal = (2 * (0.5 * 0.02) ** 2) ** 0.5
    assert result.equity_curve["portfolio_volatility_estimate"].iloc[-1] == pytest.approx(
        expected_diagonal
    )


def test_backtest_passes_covariance_aware_risk_config_to_signal_engine() -> None:
    frame = _prediction_frame(
        returns=[0.01],
        tickers=["AAPL"],
        dates=["2026-01-01"],
        volatility=0.01,
    )
    frame["portfolio_volatility_estimate"] = [0.08]
    frame["average_daily_turnover"] = [0.10]
    frame["max_symbol_weight"] = [0.40]
    frame["max_sector_weight"] = [0.50]

    result = run_long_only_backtest(
        frame,
        BacktestConfig(
            top_n=1,
            max_symbol_weight=0.50,
            max_sector_weight=0.60,
            portfolio_volatility_limit=0.20,
            average_daily_turnover_budget=0.20,
            portfolio_covariance_lookback=5,
            covariance_min_periods=3,
            covariance_return_column="return_1",
            covariance_aware_risk_enabled=True,
        ),
    )

    signal = result.signals.iloc[0]
    assert signal["risk_metric_penalty"] == pytest.approx(0.0)
    assert bool(signal["covariance_aware_risk_enabled"]) is True
    assert signal["portfolio_covariance_lookback"] == 5
    assert signal["covariance_return_column"] == "return_1"
    assert signal["covariance_min_periods"] == 3
    assert signal["portfolio_volatility_limit"] == pytest.approx(0.20)
    assert signal["average_daily_turnover_budget"] == pytest.approx(0.20)
    assert signal["configured_max_symbol_weight"] == pytest.approx(0.50)
    assert signal["configured_max_sector_weight"] == pytest.approx(0.60)
    assert signal["action"] == "BUY"


def _prediction_frame(
    returns: list[float],
    tickers: list[str],
    dates: list[str],
    volatility: float = 0.01,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "ticker": tickers,
            "expected_return": [0.05] * len(dates),
            "predicted_volatility": [volatility] * len(dates),
            "downside_quantile": [0.0] * len(dates),
            "model_confidence": [1.0] * len(dates),
            "text_risk_score": [0.0] * len(dates),
            "sec_risk_flag": [0.0] * len(dates),
            "sec_risk_flag_20d": [0.0] * len(dates),
            "news_negative_ratio": [0.0] * len(dates),
            "liquidity_score": [20.0] * len(dates),
            "return_1": returns,
            "forward_return_1": returns,
            "forward_return_20": returns,
        }
    )
