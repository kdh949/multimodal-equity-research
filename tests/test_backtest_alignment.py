from __future__ import annotations

import pandas as pd
import pytest

from quant_research.backtest import (
    BacktestHorizonAlignment,
    align_backtest_horizon_inputs,
    forward_return_horizon,
)


def test_backtest_horizon_alignment_adds_explicit_signal_and_holding_dates() -> None:
    dates = pd.bdate_range("2026-01-01", periods=6)
    frame = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["AAPL"] * len(dates) + ["MSFT"] * len(dates),
            "expected_return": [0.01] * (len(dates) * 2),
            "forward_return_5": [0.10, None, None, None, None, None] * 2,
        }
    )

    aligned = align_backtest_horizon_inputs(frame, return_column="forward_return_5")

    assert len(aligned) == 2
    assert aligned["ticker"].tolist() == ["AAPL", "MSFT"]
    assert aligned["prediction_date"].eq(aligned["date"]).all()
    assert aligned["signal_date"].eq(aligned["date"]).all()
    assert aligned["holding_start_date"].eq(dates[1]).all()
    assert aligned["holding_end_date"].eq(dates[5]).all()
    assert aligned["return_label_date"].eq(aligned["holding_end_date"]).all()
    assert aligned["realized_return_column"].eq("forward_return_5").all()
    assert aligned["return_horizon"].eq(5).all()
    assert aligned["holding_periods"].eq(5).all()
    assert aligned["horizon_complete"].all()


def test_backtest_horizon_alignment_drops_incomplete_tail_labels_by_default() -> None:
    dates = pd.bdate_range("2026-01-01", periods=7)
    frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAPL"] * len(dates),
            "forward_return_5": [0.10, 0.20, None, None, None, None, None],
        }
    )

    aligned = align_backtest_horizon_inputs(frame, return_column="forward_return_5")

    assert aligned["date"].tolist() == [dates[0], dates[1]]
    assert aligned["holding_end_date"].tolist() == [dates[5], dates[6]]


def test_backtest_horizon_alignment_can_keep_incomplete_rows_for_diagnostics() -> None:
    dates = pd.bdate_range("2026-01-01", periods=3)
    frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAPL"] * len(dates),
            "forward_return_5": [0.10, None, None],
        }
    )
    alignment = BacktestHorizonAlignment(
        return_column="forward_return_5",
        require_complete_holding_period=False,
    )

    aligned = align_backtest_horizon_inputs(frame, alignment)

    assert len(aligned) == 3
    assert aligned["horizon_complete"].tolist() == [False, False, False]
    assert pd.isna(aligned["holding_end_date"]).all()


def test_backtest_horizon_alignment_selects_non_overlapping_signal_dates() -> None:
    dates = pd.bdate_range("2026-01-01", periods=11)
    frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAPL"] * len(dates),
            "forward_return_5": [0.10] * len(dates),
        }
    )

    aligned = align_backtest_horizon_inputs(
        frame,
        return_column="forward_return_5",
        mode="non_overlapping",
    )

    assert aligned["date"].tolist() == [dates[0], dates[5]]
    assert aligned["holding_start_date"].tolist() == [dates[1], dates[6]]
    assert aligned["holding_end_date"].tolist() == [dates[5], dates[10]]
    assert aligned["horizon_alignment_mode"].eq("non_overlapping").all()


def test_backtest_horizon_alignment_selects_non_overlapping_per_symbol_strategy_scope() -> None:
    dates = pd.bdate_range("2026-01-01", periods=12)
    frame = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "strategy_id": strategy_id,
                "forward_return_5": 0.10,
            }
            for ticker, strategy_id, ticker_dates in (
                ("AAPL", "stage1_long_only", dates),
                ("AAPL", "proxy_baseline", dates),
                ("MSFT", "stage1_long_only", dates[2:]),
            )
            for date in ticker_dates
        ]
    )

    aligned = align_backtest_horizon_inputs(
        frame,
        return_column="forward_return_5",
        mode="non_overlapping",
    )

    rows = {
        (row["ticker"], row["strategy_id"]): list(group["date"])
        for (row_ticker, row_strategy), group in aligned.groupby(["ticker", "strategy_id"])
        for row in [{"ticker": row_ticker, "strategy_id": row_strategy}]
    }
    assert rows[("AAPL", "stage1_long_only")] == [dates[0], dates[5]]
    assert rows[("AAPL", "proxy_baseline")] == [dates[0], dates[5]]
    assert rows[("MSFT", "stage1_long_only")] == [dates[2]]


def test_forward_return_horizon_requires_explicit_forward_return_label() -> None:
    assert forward_return_horizon("forward_return_20") == 20

    with pytest.raises(ValueError, match="forward_return_<horizon>"):
        forward_return_horizon("return_20")

    with pytest.raises(ValueError, match="at least 1"):
        forward_return_horizon("forward_return_0")
