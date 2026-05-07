from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from quant_research.backtest.metrics import calculate_metrics
from quant_research.pipeline import PipelineConfig, run_research_pipeline
from quant_research.validation import (
    BASELINE_ALIGNMENT_RULES_SCHEMA_VERSION,
    EQUAL_WEIGHT_BASELINE_TYPE,
    MARKET_BENCHMARK_BASELINE_TYPE,
    STAGE1_REQUIRED_BASELINE_TYPES,
    BaselineComparisonInput,
    build_benchmark_construction_inputs,
    build_benchmark_return_series,
    build_equal_weight_baseline_equity_curve,
    build_equal_weight_baseline_return_series,
    calculate_equal_weight_baseline_performance_metrics,
    construct_spy_baseline_return_series,
    validate_baseline_alignment_rules,
    validate_stage1_baseline_comparison_inputs,
)


def test_benchmark_inputs_from_config_normalize_window_and_ticker_universe() -> None:
    config = PipelineConfig(
        tickers=["aapl", " MSFT ", "AAPL"],
        start=date(2025, 1, 1),
        end=date(2025, 12, 31),
        benchmark_ticker="qqq",
        data_mode="local",
        interval="1d",
    )

    inputs = build_benchmark_construction_inputs(config)

    assert inputs.evaluation_window.start == date(2025, 1, 1)
    assert inputs.evaluation_window.end == date(2025, 12, 31)
    assert inputs.ticker_universe.tickers == ("AAPL", "MSFT")
    assert inputs.ticker_universe.benchmark_ticker == "QQQ"
    assert inputs.ticker_universe.data_tickers == ("AAPL", "MSFT", "QQQ")
    assert inputs.return_column == "forward_return_20"
    assert inputs.return_horizon == 20
    json.dumps(inputs.to_dict())


def test_benchmark_inputs_preserve_required_stage1_baselines_as_separate_inputs() -> None:
    config = PipelineConfig(
        tickers=["aapl", "MSFT", "AAPL"],
        start=date(2025, 1, 1),
        end=date(2025, 12, 31),
        benchmark_ticker="qqq",
        cost_bps=12.0,
        slippage_bps=3.0,
    )

    inputs = build_benchmark_construction_inputs(config)

    baseline_inputs = inputs.baseline_comparison_inputs
    assert tuple(row.baseline_type for row in baseline_inputs) == STAGE1_REQUIRED_BASELINE_TYPES
    assert [row.name for row in baseline_inputs] == ["QQQ", "equal_weight"]

    market_benchmark, equal_weight = baseline_inputs
    assert market_benchmark.baseline_type == MARKET_BENCHMARK_BASELINE_TYPE
    assert market_benchmark.return_basis == "cost_adjusted_benchmark_return"
    assert market_benchmark.data_source == "benchmark_return_series"
    assert market_benchmark.benchmark_ticker == "QQQ"
    assert market_benchmark.universe_tickers == ()
    assert market_benchmark.cost_bps == pytest.approx(12.0)
    assert market_benchmark.slippage_bps == pytest.approx(3.0)

    assert equal_weight.baseline_type == EQUAL_WEIGHT_BASELINE_TYPE
    assert equal_weight.return_basis == "cost_adjusted_equal_weight_return"
    assert equal_weight.data_source == "equal_weight_baseline_return_series"
    assert equal_weight.benchmark_ticker is None
    assert equal_weight.universe_tickers == ("AAPL", "MSFT")
    assert equal_weight.cost_bps == pytest.approx(12.0)
    assert equal_weight.slippage_bps == pytest.approx(3.0)

    payload = inputs.to_dict()
    payload_inputs = payload["baseline_comparison_inputs"]
    assert isinstance(payload_inputs, list)
    assert [row["baseline_type"] for row in payload_inputs] == list(STAGE1_REQUIRED_BASELINE_TYPES)
    assert [row["name"] for row in payload_inputs] == ["QQQ", "equal_weight"]
    assert payload["required_baseline_types"] == list(STAGE1_REQUIRED_BASELINE_TYPES)
    json.dumps(payload)


def test_baseline_alignment_rules_validate_dates_horizon_and_strategy_universe() -> None:
    candidate_dates = pd.to_datetime(["2025-01-03", "2025-01-02", "2025-01-02"])
    extra_date = pd.Timestamp("2025-01-06")
    benchmark = pd.DataFrame(
        {
            "date": [candidate_dates[1], candidate_dates[0], extra_date],
            "benchmark_ticker": ["spy", "SPY", "SPY"],
            "return_column": ["forward_return_20"] * 3,
            "return_horizon": [20] * 3,
            "benchmark_return": [0.01, 0.02, 0.99],
        }
    )
    equal_weight = pd.DataFrame(
        {
            "date": [candidate_dates[1], candidate_dates[0], extra_date],
            "baseline_name": ["equal_weight"] * 3,
            "return_column": ["forward_return_20"] * 3,
            "return_horizon": [20] * 3,
            "equal_weight_return": [0.03, 0.04, 0.88],
        }
    )

    result = validate_baseline_alignment_rules(
        candidate_dates,
        benchmark_return_series=benchmark,
        equal_weight_baseline_return_series=equal_weight,
        strategy_tickers=["msft", "AAPL", "MSFT"],
        actual_equal_weight_tickers=["AAPL", "MSFT"],
        benchmark_ticker="spy",
    )

    expected_candidate_dates = ["2025-01-02", "2025-01-03"]
    assert result["schema_version"] == BASELINE_ALIGNMENT_RULES_SCHEMA_VERSION
    assert result["status"] == "pass"
    assert result["candidate_dates"] == expected_candidate_dates
    assert result["strategy_tickers"] == ["MSFT", "AAPL"]
    assert result["benchmark_ticker"] == "SPY"
    assert result["benchmark_in_strategy_universe"] is False
    assert {rule["rule_id"] for rule in result["rules"]} == {
        "candidate_dates",
        "benchmark_dates",
        "equal_weight_dates",
        "ticker_universe",
    }

    spy_alignment = result["baselines"]["SPY"]
    equal_weight_alignment = result["baselines"]["equal_weight"]
    assert spy_alignment["aligned_dates"] == expected_candidate_dates
    assert spy_alignment["extra_baseline_dates"] == ["2025-01-06"]
    assert equal_weight_alignment["aligned_dates"] == expected_candidate_dates
    assert equal_weight_alignment["extra_baseline_dates"] == ["2025-01-06"]
    assert equal_weight_alignment["ticker_universe_alignment"]["status"] == "pass"
    assert equal_weight_alignment["ticker_universe_alignment"]["benchmark_is_data_only"] is True


def test_baseline_alignment_rules_hard_fail_missing_date_or_wrong_equal_weight_universe() -> None:
    candidate_dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    benchmark = pd.DataFrame(
        {
            "date": [candidate_dates[0], candidate_dates[1]],
            "benchmark_ticker": ["SPY", "SPY"],
            "return_column": ["forward_return_20", "forward_return_20"],
            "return_horizon": [20, 20],
            "benchmark_return": [0.01, 0.02],
        }
    )
    equal_weight = pd.DataFrame(
        {
            "date": [candidate_dates[0]],
            "baseline_name": ["equal_weight"],
            "return_column": ["forward_return_20"],
            "return_horizon": [20],
            "equal_weight_return": [0.03],
        }
    )

    result = validate_baseline_alignment_rules(
        candidate_dates,
        benchmark_return_series=benchmark,
        equal_weight_baseline_return_series=equal_weight,
        strategy_tickers=["AAPL", "MSFT"],
        actual_equal_weight_tickers=["AAPL", "NVDA"],
        benchmark_ticker="SPY",
    )

    assert result["status"] == "hard_fail"
    assert result["failed_checks"] == ["equal_weight", "ticker_universe"]
    equal_weight_alignment = result["baselines"]["equal_weight"]
    universe_alignment = result["baselines"]["ticker_universe"]
    assert equal_weight_alignment["missing_candidate_dates"] == ["2025-01-03"]
    assert equal_weight_alignment["ticker_universe_alignment"]["missing_universe_tickers"] == [
        "MSFT"
    ]
    assert equal_weight_alignment["ticker_universe_alignment"]["extra_universe_tickers"] == [
        "NVDA"
    ]
    assert universe_alignment["status"] == "hard_fail"


def test_stage1_baseline_input_validation_rejects_missing_required_baseline() -> None:
    market_only = BaselineComparisonInput(
        name="SPY",
        baseline_type=MARKET_BENCHMARK_BASELINE_TYPE,
        return_basis="cost_adjusted_benchmark_return",
        return_column="forward_return_5",
        return_horizon=5,
        data_source="benchmark_return_series",
        benchmark_ticker="SPY",
    )

    with pytest.raises(ValueError, match="missing required Stage 1 baseline inputs"):
        validate_stage1_baseline_comparison_inputs([market_only])


def test_benchmark_inputs_prefer_actual_strategy_evaluation_dates() -> None:
    config = PipelineConfig(
        tickers=["SPY", "AAPL"],
        start=date(2025, 1, 1),
        end=date(2025, 12, 31),
    )
    evaluation_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-03-03", "2025-04-04", None]),
            "equity": [1.0, 1.02, 1.03],
        }
    )

    inputs = build_benchmark_construction_inputs(config, evaluation_frame=evaluation_frame)

    assert inputs.evaluation_window.start == date(2025, 3, 3)
    assert inputs.evaluation_window.end == date(2025, 4, 4)


def test_benchmark_return_series_aligns_spy_returns_to_strategy_dates() -> None:
    price_data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-06",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-06",
                ]
            ),
            "ticker": ["SPY", "SPY", "SPY", "AAPL", "AAPL", "AAPL"],
            "adj_close": [100.0, 102.0, 101.0, 50.0, 60.0, 70.0],
        }
    )
    evaluation_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-03", "2025-01-02", "2025-01-02"]),
            "equity": [1.02, 1.0, 1.0],
        }
    )

    series = build_benchmark_return_series(price_data, evaluation_frame)

    assert list(series["date"]) == list(pd.to_datetime(["2025-01-02", "2025-01-03"]))
    assert list(series["return_date"]) == list(pd.to_datetime(["2025-01-03", "2025-01-06"]))
    assert series["benchmark_ticker"].tolist() == ["SPY", "SPY"]
    assert series["benchmark_return"].tolist() == pytest.approx([0.02, 101.0 / 102.0 - 1])
    assert not series["missing_benchmark_return"].any()
    assert series["benchmark_equity"].iloc[-1] == pytest.approx((1.0 + 0.02) * (101.0 / 102.0))


def test_equal_weight_baseline_equity_curve_uses_backtest_metric_contract() -> None:
    dates = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    equal_weight = pd.DataFrame(
        {
            "date": dates,
            "return_date": dates.shift(1, freq="B"),
            "baseline_name": ["equal_weight"] * 3,
            "return_column": ["forward_return_1"] * 3,
            "return_horizon": [1, 1, 1],
            "equal_weight_return": [0.04, -0.02, 0.03],
            "constituent_count": [2, 2, 2],
            "expected_constituent_count": [2, 2, 2],
        }
    )
    benchmark = pd.DataFrame(
        {
            "date": dates,
            "benchmark_ticker": ["SPY"] * 3,
            "return_column": ["forward_return_1"] * 3,
            "return_horizon": [1, 1, 1],
            "benchmark_return": [0.01, 0.01, 0.01],
        }
    )

    curve = build_equal_weight_baseline_equity_curve(
        equal_weight,
        benchmark_return_series=benchmark,
        cost_bps=10.0,
        slippage_bps=5.0,
    )
    metrics = calculate_equal_weight_baseline_performance_metrics(
        equal_weight,
        benchmark_return_series=benchmark,
        cost_bps=10.0,
        slippage_bps=5.0,
    )

    assert list(curve["date"]) == list(dates)
    assert curve["portfolio_return"].tolist() == pytest.approx([0.0385, -0.02, 0.03])
    assert curve["cost_adjusted_return"].tolist() == pytest.approx(curve["portfolio_return"].tolist())
    assert curve["turnover"].tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert curve["exposure"].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert curve["benchmark_return"].tolist() == pytest.approx([0.01, 0.01, 0.01])
    assert curve["cost_adjusted_benchmark_return"].tolist() == pytest.approx([0.0085, 0.01, 0.01])
    expected_metrics = calculate_metrics(curve)
    assert metrics.cagr == pytest.approx(expected_metrics.cagr)
    assert metrics.sharpe == pytest.approx(expected_metrics.sharpe)
    assert metrics.max_drawdown == pytest.approx(expected_metrics.max_drawdown)
    assert metrics.turnover == pytest.approx(1.0 / 3.0)
    assert metrics.exposure == pytest.approx(1.0)
    assert metrics.benchmark_cagr != 0.0


def test_benchmark_return_series_uses_selected_return_horizon_on_strategy_dates() -> None:
    dates = pd.bdate_range("2025-01-02", periods=8)
    price_data = pd.DataFrame(
        {
            "date": list(dates) + list(dates),
            "ticker": ["SPY"] * len(dates) + ["AAPL"] * len(dates),
            "adj_close": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                110.0,
                111.0,
                112.0,
                50.0,
                51.0,
                52.0,
                53.0,
                54.0,
                55.0,
                56.0,
                57.0,
            ],
        }
    )
    evaluation_frame = pd.DataFrame({"date": [dates[0], dates[1], dates[0]]})

    series = build_benchmark_return_series(
        price_data,
        evaluation_frame,
        return_column="forward_return_5",
    )

    assert list(series["date"]) == [dates[0], dates[1]]
    assert list(series["return_date"]) == [dates[5], dates[6]]
    assert series["return_column"].tolist() == ["forward_return_5", "forward_return_5"]
    assert series["return_horizon"].tolist() == [5, 5]
    assert series["benchmark_return"].tolist() == pytest.approx(
        [110.0 / 100.0 - 1.0, 111.0 / 101.0 - 1.0]
    )


def test_spy_baseline_return_series_uses_strategy_window_even_without_spy_predictions() -> None:
    dates = pd.bdate_range("2025-01-02", periods=8)
    price_data = pd.DataFrame(
        {
            "date": list(dates) + list(dates),
            "ticker": ["SPY"] * len(dates) + ["AAPL"] * len(dates),
            "adj_close": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
                107.0,
                50.0,
                70.0,
                90.0,
                110.0,
                130.0,
                150.0,
                170.0,
                190.0,
            ],
        }
    )
    strategy_evaluation_frame = pd.DataFrame(
        {
            "date": [dates[2], dates[0], dates[2]],
            "ticker": ["AAPL", "AAPL", "MSFT"],
            "equity": [1.02, 1.00, 1.02],
        }
    )

    series = construct_spy_baseline_return_series(
        price_data,
        strategy_evaluation_frame,
        return_column="forward_return_5",
    )

    assert list(series["date"]) == [dates[0], dates[2]]
    assert list(series["return_date"]) == [dates[5], dates[7]]
    assert series["benchmark_ticker"].tolist() == ["SPY", "SPY"]
    assert series["return_column"].tolist() == ["forward_return_5", "forward_return_5"]
    assert series["return_horizon"].tolist() == [5, 5]
    assert series["benchmark_return"].tolist() == pytest.approx(
        [105.0 / 100.0 - 1.0, 107.0 / 102.0 - 1.0]
    )
    assert not series["missing_benchmark_return"].any()


def test_equal_weight_baseline_series_uses_strategy_universe_on_strategy_dates() -> None:
    price_data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-06",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-06",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-06",
                ]
            ),
            "ticker": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT", "SPY", "SPY", "SPY"],
            "forward_return_1": [0.10, 0.20, None, -0.05, 0.04, None, 0.99, 0.99, None],
        }
    )
    evaluation_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-03", "2025-01-02", "2025-01-02"]),
            "equity": [1.02, 1.0, 1.0],
        }
    )

    series = build_equal_weight_baseline_return_series(
        price_data,
        evaluation_frame,
        tickers=["msft", "AAPL"],
    )

    assert list(series["date"]) == list(pd.to_datetime(["2025-01-02", "2025-01-03"]))
    assert list(series["return_date"]) == list(pd.to_datetime(["2025-01-03", "2025-01-06"]))
    assert series["baseline_name"].tolist() == ["equal_weight", "equal_weight"]
    assert series["equal_weight_return"].tolist() == pytest.approx([0.025, 0.12])
    assert series["constituent_count"].tolist() == [2, 2]
    assert series["expected_constituent_count"].tolist() == [2, 2]
    assert not series["missing_equal_weight_return"].any()
    assert not series["incomplete_ticker_universe"].any()
    assert series["equal_weight_equity"].iloc[-1] == pytest.approx((1.0 + 0.025) * (1.0 + 0.12))


def test_equal_weight_baseline_series_uses_selected_forward_horizon_on_strategy_dates() -> None:
    dates = pd.bdate_range("2025-01-02", periods=8)
    price_data = pd.DataFrame(
        {
            "date": list(dates) * 3,
            "ticker": (
                ["AAPL"] * len(dates)
                + ["MSFT"] * len(dates)
                + ["SPY"] * len(dates)
            ),
            "adj_close": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                110.0,
                111.0,
                112.0,
                200.0,
                202.0,
                204.0,
                206.0,
                208.0,
                220.0,
                222.0,
                224.0,
                300.0,
                450.0,
                450.0,
                450.0,
                450.0,
                450.0,
                450.0,
                450.0,
            ],
        }
    )
    evaluation_frame = pd.DataFrame({"date": [dates[1], dates[0], dates[0]]})

    series = build_equal_weight_baseline_return_series(
        price_data,
        evaluation_frame,
        tickers=["msft", "AAPL"],
        return_column="forward_return_5",
    )

    assert list(series["date"]) == [dates[0], dates[1]]
    assert list(series["return_date"]) == [dates[5], dates[6]]
    assert series["return_column"].tolist() == ["forward_return_5", "forward_return_5"]
    assert series["return_horizon"].tolist() == [5, 5]
    assert series["equal_weight_return"].tolist() == pytest.approx(
        [
            ((110.0 / 100.0 - 1.0) + (220.0 / 200.0 - 1.0)) / 2.0,
            ((111.0 / 101.0 - 1.0) + (222.0 / 202.0 - 1.0)) / 2.0,
        ]
    )
    assert series["constituent_count"].tolist() == [2, 2]
    assert not series["incomplete_ticker_universe"].any()


def test_both_baselines_use_strategy_window_alignment_and_expected_universe() -> None:
    price_data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-06",
                    "2025-01-07",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-06",
                    "2025-01-07",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-06",
                    "2025-01-07",
                    "2025-01-03",
                    "2025-01-06",
                ]
            ),
            "ticker": [
                "QQQ",
                "QQQ",
                "QQQ",
                "QQQ",
                "AAPL",
                "AAPL",
                "AAPL",
                "AAPL",
                "MSFT",
                "MSFT",
                "MSFT",
                "MSFT",
                "SPY",
                "SPY",
            ],
            "forward_return_1": [
                0.50,
                0.03,
                -0.02,
                0.40,
                0.91,
                0.10,
                0.20,
                0.92,
                0.93,
                -0.04,
                0.06,
                0.94,
                0.99,
                0.99,
            ],
        }
    )
    strategy_evaluation_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-06", "2025-01-03", "2025-01-03"]),
            "equity": [1.03, 1.00, 1.00],
        }
    )
    expected_strategy_dates = list(pd.to_datetime(["2025-01-03", "2025-01-06"]))
    expected_return_dates = list(pd.to_datetime(["2025-01-06", "2025-01-07"]))

    benchmark = build_benchmark_return_series(
        price_data,
        strategy_evaluation_frame,
        benchmark_ticker="qqq",
    )
    equal_weight = build_equal_weight_baseline_return_series(
        price_data,
        strategy_evaluation_frame,
        tickers=["msft", "AAPL", "AAPL"],
    )

    assert list(benchmark["date"]) == expected_strategy_dates
    assert list(equal_weight["date"]) == expected_strategy_dates
    assert list(benchmark["return_date"]) == expected_return_dates
    assert list(equal_weight["return_date"]) == expected_return_dates

    assert benchmark["benchmark_ticker"].tolist() == ["QQQ", "QQQ"]
    assert benchmark["benchmark_return"].tolist() == pytest.approx([0.03, -0.02])

    assert equal_weight["baseline_name"].tolist() == ["equal_weight", "equal_weight"]
    assert equal_weight["equal_weight_return"].tolist() == pytest.approx([0.03, 0.13])
    assert equal_weight["constituent_count"].tolist() == [2, 2]
    assert equal_weight["expected_constituent_count"].tolist() == [2, 2]
    assert not equal_weight["incomplete_ticker_universe"].any()


def test_baseline_series_preserve_strategy_dates_and_flag_missing_returns() -> None:
    dates = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    price_data = pd.DataFrame(
        {
            "date": list(dates) * 3,
            "ticker": ["SPY"] * 3 + ["AAPL"] * 3 + ["MSFT"] * 3,
            "forward_return_1": [
                0.01,
                None,
                0.03,
                0.02,
                0.04,
                None,
                0.06,
                None,
                None,
            ],
        }
    )
    strategy_evaluation_frame = pd.DataFrame(
        {"date": [dates[2], dates[0], dates[1], dates[0]]}
    )

    benchmark = build_benchmark_return_series(price_data, strategy_evaluation_frame)
    equal_weight = build_equal_weight_baseline_return_series(
        price_data,
        strategy_evaluation_frame,
        tickers=["AAPL", "MSFT"],
    )

    assert list(benchmark["date"]) == list(dates)
    assert benchmark["benchmark_return"].iloc[[0, 2]].tolist() == pytest.approx([0.01, 0.03])
    assert pd.isna(benchmark["benchmark_return"].iloc[1])
    assert benchmark["missing_benchmark_return"].tolist() == [False, True, False]
    assert benchmark["benchmark_equity"].tolist() == pytest.approx(
        [1.01, 1.01, 1.01 * 1.03]
    )

    assert list(equal_weight["date"]) == list(dates)
    assert equal_weight["equal_weight_return"].iloc[:2].tolist() == pytest.approx([0.04, 0.04])
    assert pd.isna(equal_weight["equal_weight_return"].iloc[2])
    assert equal_weight["constituent_count"].tolist() == [2, 1, 0]
    assert equal_weight["expected_constituent_count"].tolist() == [2, 2, 2]
    assert equal_weight["missing_equal_weight_return"].tolist() == [False, False, True]
    assert equal_weight["incomplete_ticker_universe"].tolist() == [False, True, True]
    assert equal_weight["equal_weight_equity"].tolist() == pytest.approx(
        [1.04, 1.04 * 1.04, 1.04 * 1.04]
    )


def test_benchmark_and_equal_weight_use_same_sample_window_with_missing_nontrading_and_partial_data() -> None:
    strategy_dates = pd.to_datetime(
        [
            "2025-01-02",
            "2025-01-03",
            "2025-01-06",
            "2025-01-07",
            "2025-01-08",
        ]
    )
    price_data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-06",
                    "2025-01-08",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-06",
                    "2025-01-07",
                    "2025-01-08",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-06",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-06",
                    "2025-01-07",
                ]
            ),
            "ticker": [
                "SPY",
                "SPY",
                "SPY",
                "SPY",
                "AAPL",
                "AAPL",
                "AAPL",
                "AAPL",
                "AAPL",
                "MSFT",
                "MSFT",
                "MSFT",
                "NVDA",
                "NVDA",
                "NVDA",
                "NVDA",
            ],
            "forward_return_1": [
                0.010,
                None,
                0.012,
                None,
                0.020,
                0.030,
                None,
                0.010,
                None,
                -0.010,
                0.020,
                None,
                0.040,
                None,
                0.010,
                None,
            ],
        }
    )
    strategy_evaluation_frame = pd.DataFrame(
        {
            "date": [
                strategy_dates[3],
                strategy_dates[0],
                pd.Timestamp("2025-01-04"),
                strategy_dates[2],
                strategy_dates[1],
                strategy_dates[4],
                strategy_dates[0],
            ],
            "ticker": ["AAPL", "MSFT", "AAPL", "NVDA", "AAPL", "AAPL", "NVDA"],
        }
    )
    expected_window = pd.to_datetime(
        [
            "2025-01-02",
            "2025-01-03",
            "2025-01-04",
            "2025-01-06",
            "2025-01-07",
            "2025-01-08",
        ]
    )

    benchmark = build_benchmark_return_series(price_data, strategy_evaluation_frame)
    equal_weight = build_equal_weight_baseline_return_series(
        price_data,
        strategy_evaluation_frame,
        tickers=["AAPL", "MSFT", "NVDA"],
    )

    assert list(benchmark["date"]) == list(expected_window)
    assert list(equal_weight["date"]) == list(expected_window)
    assert benchmark["date"].tolist() == equal_weight["date"].tolist()
    assert benchmark["return_column"].eq("forward_return_1").all()
    assert equal_weight["return_column"].eq("forward_return_1").all()
    assert benchmark["return_horizon"].eq(1).all()
    assert equal_weight["return_horizon"].eq(1).all()

    assert benchmark["benchmark_return"].tolist() == pytest.approx(
        [0.010, float("nan"), float("nan"), 0.012, float("nan"), float("nan")],
        nan_ok=True,
    )
    assert benchmark["missing_benchmark_return"].tolist() == [
        False,
        True,
        True,
        False,
        True,
        True,
    ]
    assert equal_weight["equal_weight_return"].tolist() == pytest.approx(
        [
            (0.020 - 0.010 + 0.040) / 3.0,
            (0.030 + 0.020) / 2.0,
            float("nan"),
            0.010,
            0.010,
            float("nan"),
        ],
        nan_ok=True,
    )
    assert equal_weight["constituent_count"].tolist() == [3, 2, 0, 1, 1, 0]
    assert equal_weight["expected_constituent_count"].tolist() == [3, 3, 3, 3, 3, 3]
    assert equal_weight["missing_equal_weight_return"].tolist() == [
        False,
        False,
        True,
        False,
        False,
        True,
    ]
    assert equal_weight["incomplete_ticker_universe"].tolist() == [
        False,
        True,
        True,
        True,
        True,
        True,
    ]


def test_pipeline_result_exposes_reusable_benchmark_inputs() -> None:
    config = PipelineConfig(
        tickers=["SPY", "AAPL"],
        data_mode="synthetic",
        train_periods=60,
        test_periods=15,
        gap_periods=1,
        embargo_periods=1,
        prediction_target_column="forward_return_1",
        required_validation_horizon=1,
        top_n=1,
        sentiment_model="keyword",
        time_series_inference_mode="proxy",
        filing_extractor_model="rules",
        enable_local_filing_llm=False,
    )

    result = run_research_pipeline(config)

    assert result.benchmark_inputs is not None
    assert result.benchmark_inputs.ticker_universe.tickers == ("SPY", "AAPL")
    assert result.benchmark_inputs.ticker_universe.benchmark_ticker == "SPY"
    assert result.benchmark_inputs.evaluation_window.start == pd.Timestamp(
        result.backtest.equity_curve["date"].min()
    ).date()
    assert result.benchmark_inputs.evaluation_window.end == pd.Timestamp(
        result.backtest.equity_curve["date"].max()
    ).date()


def test_pipeline_constructs_spy_baseline_when_spy_is_not_strategy_ticker() -> None:
    config = PipelineConfig(
        tickers=["AAPL", "MSFT"],
        data_mode="synthetic",
        train_periods=60,
        test_periods=15,
        gap_periods=1,
        embargo_periods=1,
        prediction_target_column="forward_return_1",
        required_validation_horizon=1,
        top_n=1,
        sentiment_model="keyword",
        time_series_inference_mode="proxy",
        filing_extractor_model="rules",
        enable_local_filing_llm=False,
    )

    result = run_research_pipeline(config)

    assert set(result.predictions["ticker"].unique()) == {"AAPL", "MSFT"}
    assert result.benchmark_return_series is not None
    assert not result.benchmark_return_series.empty
    assert result.benchmark_return_series["benchmark_ticker"].eq("SPY").all()
    assert not result.benchmark_return_series["missing_benchmark_return"].any()
    assert list(result.benchmark_return_series["date"]) == list(result.backtest.equity_curve["date"])
    assert result.backtest.equity_curve["benchmark_return"].tolist() == pytest.approx(
        result.benchmark_return_series["benchmark_return"].tolist()
    )
    assert result.backtest.equity_curve["benchmark_return"].abs().sum() > 0
    assert result.equal_weight_baseline_return_series is not None
    assert not result.equal_weight_baseline_return_series.empty
    assert result.equal_weight_baseline_equity_curve is not None
    assert not result.equal_weight_baseline_equity_curve.empty
    assert result.equal_weight_baseline_metrics is not None
    assert list(result.equal_weight_baseline_return_series["date"]) == list(result.backtest.equity_curve["date"])
    assert list(result.equal_weight_baseline_equity_curve["date"]) == list(result.backtest.equity_curve["date"])
    assert not result.equal_weight_baseline_return_series["missing_equal_weight_return"].any()
    assert not result.equal_weight_baseline_return_series["incomplete_ticker_universe"].any()
    assert result.equal_weight_baseline_metrics.cagr == pytest.approx(
        calculate_metrics(result.equal_weight_baseline_equity_curve).cagr
    )
    assert result.equal_weight_baseline_metrics.turnover == pytest.approx(
        result.equal_weight_baseline_equity_curve["turnover"].mean()
    )

    expected_equal_weight_returns = (
        result.features[
            result.features["date"].isin(result.equal_weight_baseline_return_series["date"])
        ]
        .groupby("date")["forward_return_1"]
        .mean()
        .reindex(result.equal_weight_baseline_return_series["date"])
    )
    assert result.equal_weight_baseline_return_series["equal_weight_return"].tolist() == pytest.approx(
        expected_equal_weight_returns.tolist()
    )


def test_pipeline_baseline_series_match_strategy_dates_and_return_horizon() -> None:
    config = PipelineConfig(
        tickers=["AAPL", "MSFT"],
        data_mode="synthetic",
        train_periods=70,
        test_periods=18,
        gap_periods=5,
        embargo_periods=5,
        prediction_target_column="forward_return_5",
        required_validation_horizon=5,
        top_n=1,
        sentiment_model="keyword",
        time_series_inference_mode="proxy",
        filing_extractor_model="rules",
        enable_local_filing_llm=False,
    )

    result = run_research_pipeline(config)

    strategy_dates = list(result.backtest.equity_curve["date"])
    strategy_return_columns = result.backtest.equity_curve["realized_return_column"].unique().tolist()
    assert strategy_return_columns == ["forward_return_5"]
    assert result.benchmark_return_series is not None
    assert result.equal_weight_baseline_return_series is not None

    assert list(result.benchmark_return_series["date"]) == strategy_dates
    assert list(result.equal_weight_baseline_return_series["date"]) == strategy_dates
    assert result.benchmark_return_series["return_column"].eq("forward_return_5").all()
    assert result.equal_weight_baseline_return_series["return_column"].eq("forward_return_5").all()
    assert result.benchmark_return_series["return_horizon"].eq(5).all()
    assert result.equal_weight_baseline_return_series["return_horizon"].eq(5).all()
