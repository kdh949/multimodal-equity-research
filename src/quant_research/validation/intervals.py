from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class EvaluationInterval:
    """Inclusive evaluation interval for one symbol and deterministic strategy."""

    symbol: str
    strategy_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    fold: int | str | None = None
    source_index: int | None = None

    def __post_init__(self) -> None:
        symbol = str(self.symbol).strip()
        strategy_id = str(self.strategy_id).strip()
        if not symbol:
            raise ValueError("symbol must not be blank")
        if not strategy_id:
            raise ValueError("strategy_id must not be blank")

        start = _normalize_timestamp(self.start, field_name="start")
        end = _normalize_timestamp(self.end, field_name="end")
        if start > end:
            raise ValueError("start must be on or before end")

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "strategy_id", strategy_id)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def scope(self) -> tuple[str, str]:
        return (self.symbol, self.strategy_id)

    def overlaps(self, other: EvaluationInterval, *, same_scope_only: bool = True) -> bool:
        if same_scope_only and self.scope != other.scope:
            return False
        return max(self.start, other.start) <= min(self.end, other.end)


@dataclass(frozen=True, slots=True)
class EvaluationIntervalOverlap:
    left: EvaluationInterval
    right: EvaluationInterval
    overlap_start: pd.Timestamp
    overlap_end: pd.Timestamp

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.left.symbol,
            "strategy_id": self.left.strategy_id,
            "left_start": self.left.start,
            "left_end": self.left.end,
            "left_fold": self.left.fold,
            "left_source_index": self.left.source_index,
            "right_start": self.right.start,
            "right_end": self.right.end,
            "right_fold": self.right.fold,
            "right_source_index": self.right.source_index,
            "overlap_start": self.overlap_start,
            "overlap_end": self.overlap_end,
        }


@dataclass(frozen=True, slots=True)
class EvaluationIntervalValidationResult:
    interval_count: int
    overlap_count: int
    overlaps: tuple[EvaluationIntervalOverlap, ...]

    @property
    def passed(self) -> bool:
        return self.overlap_count == 0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([overlap.to_dict() for overlap in self.overlaps])

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "interval_count": self.interval_count,
            "overlap_count": self.overlap_count,
            "overlaps": [overlap.to_dict() for overlap in self.overlaps],
        }


def generate_non_overlapping_rebalance_schedule(
    dates: Iterable[object] | pd.Series | pd.Index,
    *,
    return_horizon: int,
    start_offset: int = 0,
    require_complete_holding_period: bool = True,
) -> pd.DataFrame:
    """Generate horizon-consistent signal dates with non-overlapping holdings.

    For a ``forward_return_N`` label, a signal emitted at date index ``t`` is
    evaluated over the interval ``t+1`` through ``t+N``. The next scheduled
    signal is therefore ``N`` observations later, so its holding interval starts
    after the previous interval ends.
    """

    horizon = int(return_horizon)
    if horizon < 1:
        raise ValueError("return_horizon must be at least 1")

    offset = int(start_offset)
    if offset < 0:
        raise ValueError("start_offset must be non-negative")

    calendar = _normalized_unique_dates(dates)
    columns = [
        "rebalance_number",
        "signal_date",
        "holding_start_date",
        "holding_end_date",
        "return_label_date",
        "return_horizon",
        "holding_periods",
        "horizon_complete",
        "horizon_alignment_mode",
    ]
    if offset >= len(calendar):
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    rebalance_number = 0
    for signal_index in range(offset, len(calendar), horizon):
        holding_start_index = signal_index + 1
        holding_end_index = signal_index + horizon
        complete = holding_end_index < len(calendar)
        if require_complete_holding_period and not complete:
            continue

        holding_start = (
            calendar[holding_start_index] if holding_start_index < len(calendar) else pd.NaT
        )
        holding_end = calendar[holding_end_index] if complete else pd.NaT
        rows.append(
            {
                "rebalance_number": rebalance_number,
                "signal_date": calendar[signal_index],
                "holding_start_date": holding_start,
                "holding_end_date": holding_end,
                "return_label_date": holding_end,
                "return_horizon": horizon,
                "holding_periods": (
                    holding_end_index - signal_index
                    if complete
                    else max(len(calendar) - 1 - signal_index, 0)
                ),
                "horizon_complete": bool(complete),
                "horizon_alignment_mode": "non_overlapping",
            }
        )
        rebalance_number += 1

    return pd.DataFrame(rows, columns=columns)


def build_evaluation_intervals(
    frame: pd.DataFrame,
    *,
    symbol_column: str = "ticker",
    strategy_column: str | None = "strategy_id",
    default_strategy_id: str = "deterministic_signal_engine",
    start_column: str = "holding_start_date",
    end_column: str = "holding_end_date",
    fold_column: str | None = "fold",
    skip_incomplete: bool = True,
) -> tuple[EvaluationInterval, ...]:
    """Build evaluation intervals from aligned prediction/backtest rows."""

    required = {symbol_column, start_column, end_column}
    if strategy_column is not None:
        required.add(strategy_column)
    if fold_column is not None and fold_column in frame.columns:
        required.add(fold_column)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"evaluation interval input missing required columns: {sorted(missing)}")

    intervals: list[EvaluationInterval] = []
    for source_index, (_, row) in enumerate(frame.iterrows()):
        start = pd.to_datetime(row[start_column], errors="coerce")
        end = pd.to_datetime(row[end_column], errors="coerce")
        if pd.isna(start) or pd.isna(end):
            if skip_incomplete:
                continue
            raise ValueError("evaluation interval start/end must not be null")

        strategy_id = (
            row[strategy_column]
            if strategy_column is not None
            else default_strategy_id
        )
        intervals.append(
            EvaluationInterval(
                symbol=row[symbol_column],
                strategy_id=strategy_id,
                start=start,
                end=end,
                fold=row[fold_column] if fold_column is not None and fold_column in frame.columns else None,
                source_index=source_index,
            )
        )
    return tuple(intervals)


def find_overlapping_evaluation_intervals(
    intervals: Iterable[EvaluationInterval],
) -> tuple[EvaluationIntervalOverlap, ...]:
    """Return all inclusive overlaps within each symbol/strategy scope."""

    grouped: dict[tuple[str, str], list[EvaluationInterval]] = {}
    for interval in intervals:
        grouped.setdefault(interval.scope, []).append(interval)

    overlaps: list[EvaluationIntervalOverlap] = []
    for scoped_intervals in grouped.values():
        ordered = sorted(scoped_intervals, key=lambda item: (item.start, item.end))
        active: list[EvaluationInterval] = []
        for current in ordered:
            active = [candidate for candidate in active if candidate.end >= current.start]
            for candidate in active:
                if candidate.overlaps(current):
                    overlaps.append(
                        EvaluationIntervalOverlap(
                            left=candidate,
                            right=current,
                            overlap_start=max(candidate.start, current.start),
                            overlap_end=min(candidate.end, current.end),
                        )
                    )
            active.append(current)
    return tuple(overlaps)


def validate_non_overlapping_evaluation_intervals(
    intervals: Iterable[EvaluationInterval],
) -> EvaluationIntervalValidationResult:
    materialized = tuple(intervals)
    overlaps = find_overlapping_evaluation_intervals(materialized)
    return EvaluationIntervalValidationResult(
        interval_count=len(materialized),
        overlap_count=len(overlaps),
        overlaps=overlaps,
    )


def validate_evaluation_interval_frame(
    frame: pd.DataFrame,
    **kwargs: object,
) -> EvaluationIntervalValidationResult:
    intervals = build_evaluation_intervals(frame, **kwargs)
    return validate_non_overlapping_evaluation_intervals(intervals)


def select_non_overlapping_evaluation_samples(
    frame: pd.DataFrame,
    *,
    symbol_column: str = "ticker",
    strategy_column: str | None = "strategy_id",
    default_strategy_id: str = "deterministic_signal_engine",
    start_column: str = "holding_start_date",
    end_column: str = "holding_end_date",
    skip_incomplete: bool = True,
) -> pd.DataFrame:
    """Select candidate evaluation rows whose holding intervals do not overlap.

    Selection is scoped to the same symbol and deterministic strategy. Intervals
    from different symbols or strategies may overlap because they represent
    separate candidate evaluation streams.
    """

    required = {symbol_column, start_column, end_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"evaluation sample input missing required columns: {sorted(missing)}")

    if frame.empty:
        return frame.copy()

    candidates: list[tuple[int, EvaluationInterval]] = []
    for source_index, (_, row) in enumerate(frame.iterrows()):
        start = pd.to_datetime(row[start_column], errors="coerce")
        end = pd.to_datetime(row[end_column], errors="coerce")
        if pd.isna(start) or pd.isna(end):
            if skip_incomplete:
                continue
            raise ValueError("evaluation sample start/end must not be null")

        strategy_id = (
            row[strategy_column]
            if strategy_column is not None and strategy_column in frame.columns
            else default_strategy_id
        )
        candidates.append(
            (
                source_index,
                EvaluationInterval(
                    symbol=row[symbol_column],
                    strategy_id=strategy_id,
                    start=start,
                    end=end,
                    source_index=source_index,
                ),
            )
        )

    selected_positions: set[int] = set()
    grouped: dict[tuple[str, str], list[tuple[int, EvaluationInterval]]] = {}
    for source_index, interval in candidates:
        grouped.setdefault(interval.scope, []).append((source_index, interval))

    for scoped_candidates in grouped.values():
        last_end: pd.Timestamp | None = None
        ordered = sorted(
            scoped_candidates,
            key=lambda item: (item[1].start, item[1].end, item[0]),
        )
        for source_index, interval in ordered:
            if last_end is not None and interval.start <= last_end:
                continue
            selected_positions.add(source_index)
            last_end = interval.end

    return frame.iloc[
        [position for position in range(len(frame)) if position in selected_positions]
    ].copy()


def _normalized_unique_dates(dates: Iterable[object] | pd.Series | pd.Index) -> list[pd.Timestamp]:
    series = pd.Series(list(dates) if not isinstance(dates, pd.Series) else dates)
    normalized = pd.to_datetime(series, errors="coerce").dropna().dt.normalize()
    return [pd.Timestamp(date).normalize() for date in sorted(normalized.unique())]


def _normalize_timestamp(value: object, *, field_name: str) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError(f"{field_name} must be a valid timestamp")
    return pd.Timestamp(timestamp).normalize()
