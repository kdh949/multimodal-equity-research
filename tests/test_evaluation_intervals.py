from __future__ import annotations

import pandas as pd
import pytest

from quant_research.validation import (
    EvaluationInterval,
    build_evaluation_intervals,
    generate_non_overlapping_rebalance_schedule,
    select_non_overlapping_evaluation_samples,
    validate_evaluation_interval_frame,
    validate_non_overlapping_evaluation_intervals,
)


def test_evaluation_interval_overlap_is_scoped_to_same_symbol_and_strategy() -> None:
    base = EvaluationInterval(
        symbol="AAPL",
        strategy_id="stage1_long_only",
        start=pd.Timestamp("2026-01-02"),
        end=pd.Timestamp("2026-01-30"),
    )
    same_scope = EvaluationInterval(
        symbol="AAPL",
        strategy_id="stage1_long_only",
        start=pd.Timestamp("2026-01-15"),
        end=pd.Timestamp("2026-02-10"),
    )
    different_symbol = EvaluationInterval(
        symbol="MSFT",
        strategy_id="stage1_long_only",
        start=pd.Timestamp("2026-01-15"),
        end=pd.Timestamp("2026-02-10"),
    )
    different_strategy = EvaluationInterval(
        symbol="AAPL",
        strategy_id="proxy_baseline",
        start=pd.Timestamp("2026-01-15"),
        end=pd.Timestamp("2026-02-10"),
    )

    assert base.overlaps(same_scope)
    assert not base.overlaps(different_symbol)
    assert not base.overlaps(different_strategy)
    assert base.overlaps(different_symbol, same_scope_only=False)


def test_validate_non_overlapping_evaluation_intervals_reports_same_scope_overlap() -> None:
    intervals = [
        EvaluationInterval("AAPL", "stage1_long_only", "2026-01-02", "2026-01-30", fold=0),
        EvaluationInterval("AAPL", "stage1_long_only", "2026-01-30", "2026-02-27", fold=1),
        EvaluationInterval("AAPL", "proxy_baseline", "2026-01-15", "2026-02-12", fold=1),
        EvaluationInterval("MSFT", "stage1_long_only", "2026-01-15", "2026-02-12", fold=1),
    ]

    result = validate_non_overlapping_evaluation_intervals(intervals)

    assert not result.passed
    assert result.interval_count == 4
    assert result.overlap_count == 1
    overlap = result.overlaps[0]
    assert overlap.left.fold == 0
    assert overlap.right.fold == 1
    assert overlap.overlap_start == pd.Timestamp("2026-01-30")
    assert overlap.overlap_end == pd.Timestamp("2026-01-30")
    assert result.to_frame()["strategy_id"].tolist() == ["stage1_long_only"]


def test_validate_evaluation_interval_frame_accepts_non_overlapping_horizon_consistent_rows() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "MSFT"],
            "strategy_id": ["stage1_long_only"] * 3,
            "holding_start_date": pd.to_datetime(
                ["2026-01-02", "2026-02-02", "2026-01-15"]
            ),
            "holding_end_date": pd.to_datetime(
                ["2026-01-30", "2026-02-27", "2026-02-12"]
            ),
            "fold": [0, 1, 0],
        }
    )

    result = validate_evaluation_interval_frame(frame)

    assert result.passed
    assert result.interval_count == 3
    assert result.overlap_count == 0


def test_select_non_overlapping_evaluation_samples_is_scoped_by_symbol_and_strategy() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL", "AAPL", "MSFT"],
            "strategy_id": [
                "stage1_long_only",
                "stage1_long_only",
                "stage1_long_only",
                "proxy_baseline",
                "stage1_long_only",
            ],
            "holding_start_date": pd.to_datetime(
                [
                    "2026-01-02",
                    "2026-01-15",
                    "2026-02-02",
                    "2026-01-15",
                    "2026-01-15",
                ]
            ),
            "holding_end_date": pd.to_datetime(
                [
                    "2026-01-30",
                    "2026-02-12",
                    "2026-02-27",
                    "2026-02-12",
                    "2026-02-12",
                ]
            ),
            "sample_id": ["a0", "a1_overlap", "a2", "proxy_overlap_ok", "msft_overlap_ok"],
        }
    )

    selected = select_non_overlapping_evaluation_samples(frame)

    assert selected["sample_id"].tolist() == [
        "a0",
        "a2",
        "proxy_overlap_ok",
        "msft_overlap_ok",
    ]
    result = validate_evaluation_interval_frame(selected)
    assert result.passed


def test_generate_non_overlapping_rebalance_schedule_uses_horizon_stride() -> None:
    dates = pd.bdate_range("2026-01-01", periods=13)

    schedule = generate_non_overlapping_rebalance_schedule(dates, return_horizon=5)

    assert schedule["rebalance_number"].tolist() == [0, 1]
    assert schedule["signal_date"].tolist() == [dates[0], dates[5]]
    assert schedule["holding_start_date"].tolist() == [dates[1], dates[6]]
    assert schedule["holding_end_date"].tolist() == [dates[5], dates[10]]
    assert schedule["return_label_date"].tolist() == [dates[5], dates[10]]
    assert schedule["return_horizon"].eq(5).all()
    assert schedule["holding_periods"].eq(5).all()
    assert schedule["horizon_complete"].all()

    interval_frame = schedule.assign(ticker="AAPL", strategy_id="stage1_long_only")
    assert validate_evaluation_interval_frame(interval_frame).passed


def test_generated_schedule_excludes_overlapping_candidates_and_validates_non_overlapping() -> None:
    dates = pd.bdate_range("2026-01-01", periods=16)
    schedule = generate_non_overlapping_rebalance_schedule(dates, return_horizon=5)
    overlapping_candidates = pd.DataFrame(
        {
            "rebalance_number": [100, 101],
            "signal_date": [dates[1], dates[6]],
            "holding_start_date": [dates[2], dates[7]],
            "holding_end_date": [dates[6], dates[11]],
            "return_label_date": [dates[6], dates[11]],
            "return_horizon": [5, 5],
            "holding_periods": [5, 5],
            "horizon_complete": [True, True],
            "horizon_alignment_mode": ["non_overlapping", "non_overlapping"],
        }
    )
    candidate_frame = pd.concat([schedule, overlapping_candidates], ignore_index=True).assign(
        ticker="AAPL",
        strategy_id="stage1_long_only",
    )

    assert not validate_evaluation_interval_frame(candidate_frame).passed

    selected = select_non_overlapping_evaluation_samples(candidate_frame)

    assert selected["signal_date"].tolist() == [dates[0], dates[5], dates[10]]
    assert selected["rebalance_number"].tolist() == [0, 1, 2]
    assert set(overlapping_candidates["signal_date"]).isdisjoint(set(selected["signal_date"]))
    result = validate_evaluation_interval_frame(selected)
    assert result.passed
    assert result.interval_count == 3
    assert result.overlap_count == 0


def test_generate_non_overlapping_rebalance_schedule_can_keep_incomplete_tail() -> None:
    dates = pd.bdate_range("2026-01-01", periods=8)

    schedule = generate_non_overlapping_rebalance_schedule(
        dates,
        return_horizon=5,
        require_complete_holding_period=False,
    )

    assert schedule["signal_date"].tolist() == [dates[0], dates[5]]
    assert schedule["holding_end_date"].tolist()[0] == dates[5]
    assert pd.isna(schedule["holding_end_date"].iloc[1])
    assert schedule["horizon_complete"].tolist() == [True, False]


def test_build_evaluation_intervals_can_use_default_strategy_id_and_skip_incomplete_rows() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "holding_start_date": [pd.Timestamp("2026-01-02"), pd.NaT],
            "holding_end_date": [pd.Timestamp("2026-01-30"), pd.NaT],
        }
    )

    intervals = build_evaluation_intervals(frame, strategy_column=None)

    assert len(intervals) == 1
    assert intervals[0].strategy_id == "deterministic_signal_engine"
    assert intervals[0].source_index == 0


def test_evaluation_interval_rejects_invalid_bounds_and_missing_columns() -> None:
    with pytest.raises(ValueError, match="start must be on or before end"):
        EvaluationInterval("AAPL", "stage1_long_only", "2026-02-01", "2026-01-01")

    with pytest.raises(ValueError, match="missing required columns"):
        build_evaluation_intervals(pd.DataFrame({"ticker": ["AAPL"]}))
