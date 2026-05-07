from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from quant_research.validation.intervals import (
    generate_non_overlapping_rebalance_schedule,
    select_non_overlapping_evaluation_samples,
)

AlignmentMode = Literal["daily", "non_overlapping"]


@dataclass(frozen=True)
class BacktestHorizonAlignment:
    return_column: str = "forward_return_20"
    date_column: str = "date"
    ticker_column: str = "ticker"
    strategy_column: str | None = "strategy_id"
    mode: AlignmentMode = "daily"
    require_complete_holding_period: bool = True

    def __post_init__(self) -> None:
        if not str(self.return_column).strip():
            raise ValueError("return_column must not be blank")
        if not str(self.date_column).strip():
            raise ValueError("date_column must not be blank")
        if not str(self.ticker_column).strip():
            raise ValueError("ticker_column must not be blank")
        if self.mode not in {"daily", "non_overlapping"}:
            raise ValueError("mode must be 'daily' or 'non_overlapping'")

    @property
    def return_horizon(self) -> int:
        return forward_return_horizon(self.return_column)


def align_backtest_horizon_inputs(
    frame: pd.DataFrame,
    alignment: BacktestHorizonAlignment | None = None,
    *,
    return_column: str | None = None,
    mode: AlignmentMode | None = None,
    require_complete_holding_period: bool | None = None,
) -> pd.DataFrame:
    """Align prediction rows with the forward-return holding period used in backtests.

    The returned rows keep the original signal date and add explicit timing columns:
    predictions are made on ``signal_date``, positions are assumed to start on the
    next available observation, and the realized label must end exactly ``horizon``
    observations after the signal date.
    """

    alignment = _resolve_alignment(
        alignment,
        return_column=return_column,
        mode=mode,
        require_complete_holding_period=require_complete_holding_period,
    )
    _require_columns(frame, alignment)

    if frame.empty:
        return _empty_aligned_frame(frame, alignment)

    output = frame.copy()
    output[alignment.date_column] = pd.to_datetime(
        output[alignment.date_column],
        errors="coerce",
    ).dt.normalize()
    output = output.dropna(subset=[alignment.date_column])
    sort_columns = [alignment.ticker_column]
    if alignment.strategy_column is not None and alignment.strategy_column in output.columns:
        sort_columns.append(alignment.strategy_column)
    sort_columns.append(alignment.date_column)
    output = output.sort_values(sort_columns).reset_index(drop=True)

    if output.empty:
        return _empty_aligned_frame(frame, alignment)

    horizon = alignment.return_horizon
    date_positions = _date_positions(output[alignment.date_column])
    selected_signal_dates = _selected_signal_dates(
        date_positions,
        alignment.mode,
        horizon=horizon,
        require_complete_holding_period=alignment.require_complete_holding_period,
    )

    rows: list[pd.DataFrame] = []
    group_columns = [alignment.ticker_column]
    if alignment.strategy_column is not None and alignment.strategy_column in output.columns:
        group_columns.append(alignment.strategy_column)
    for _, ticker_frame in output.groupby(group_columns, sort=False):
        rows.append(_align_ticker_frame(ticker_frame, alignment, horizon, selected_signal_dates))

    aligned = pd.concat(rows, ignore_index=True) if rows else _empty_aligned_frame(frame, alignment)
    if alignment.require_complete_holding_period:
        aligned = aligned[aligned["horizon_complete"]].reset_index(drop=True)
    if alignment.mode == "non_overlapping" and not aligned.empty:
        aligned = select_non_overlapping_evaluation_samples(
            aligned,
            symbol_column=alignment.ticker_column,
            strategy_column=alignment.strategy_column,
            start_column="holding_start_date",
            end_column="holding_end_date",
        ).reset_index(drop=True)
    _validate_horizon_aligned_return_timing(aligned, alignment)
    return aligned


def forward_return_horizon(return_column: str) -> int:
    prefix = "forward_return_"
    column = str(return_column)
    if not column.startswith(prefix):
        raise ValueError("return_column must use the forward_return_<horizon> naming convention")
    try:
        horizon = int(column.removeprefix(prefix))
    except ValueError as exc:
        raise ValueError("return_column horizon must be an integer") from exc
    if horizon < 1:
        raise ValueError("return_column horizon must be at least 1")
    return horizon


def _resolve_alignment(
    alignment: BacktestHorizonAlignment | None,
    *,
    return_column: str | None,
    mode: AlignmentMode | None,
    require_complete_holding_period: bool | None,
) -> BacktestHorizonAlignment:
    if alignment is None:
        return BacktestHorizonAlignment(
            return_column=return_column or "forward_return_20",
            mode=mode or "daily",
            require_complete_holding_period=(
                True
                if require_complete_holding_period is None
                else bool(require_complete_holding_period)
            ),
        )
    if return_column is None and mode is None and require_complete_holding_period is None:
        return alignment
    return BacktestHorizonAlignment(
        return_column=return_column or alignment.return_column,
        date_column=alignment.date_column,
        ticker_column=alignment.ticker_column,
        strategy_column=alignment.strategy_column,
        mode=mode or alignment.mode,
        require_complete_holding_period=(
            alignment.require_complete_holding_period
            if require_complete_holding_period is None
            else bool(require_complete_holding_period)
        ),
    )


def _require_columns(frame: pd.DataFrame, alignment: BacktestHorizonAlignment) -> None:
    required = {alignment.date_column, alignment.ticker_column, alignment.return_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"backtest alignment input missing required columns: {sorted(missing)}")


def _empty_aligned_frame(frame: pd.DataFrame, alignment: BacktestHorizonAlignment) -> pd.DataFrame:
    columns = list(frame.columns)
    for column in _alignment_columns():
        if column not in columns:
            columns.append(column)
    return pd.DataFrame(columns=columns)


def _alignment_columns() -> list[str]:
    return [
        "prediction_date",
        "signal_date",
        "holding_start_date",
        "holding_end_date",
        "return_label_date",
        "realized_return_column",
        "return_horizon",
        "holding_periods",
        "horizon_complete",
        "horizon_alignment_mode",
    ]


def _date_positions(dates: pd.Series) -> dict[pd.Timestamp, int]:
    unique_dates = sorted(pd.to_datetime(dates, errors="coerce").dropna().unique())
    return {pd.Timestamp(date).normalize(): index for index, date in enumerate(unique_dates)}


def _selected_signal_dates(
    date_positions: dict[pd.Timestamp, int],
    mode: AlignmentMode,
    *,
    horizon: int,
    require_complete_holding_period: bool,
) -> set[pd.Timestamp] | None:
    if mode == "daily":
        return set(date_positions)

    del date_positions, horizon, require_complete_holding_period
    return None


def _align_ticker_frame(
    frame: pd.DataFrame,
    alignment: BacktestHorizonAlignment,
    horizon: int,
    selected_signal_dates: set[pd.Timestamp] | None,
) -> pd.DataFrame:
    output = frame.copy().reset_index(drop=True)
    ticker_dates = [pd.Timestamp(date).normalize() for date in output[alignment.date_column]]
    scoped_signal_dates = (
        selected_signal_dates
        if selected_signal_dates is not None
        else _scheduled_signal_dates(
            ticker_dates,
            horizon=horizon,
            require_complete_holding_period=alignment.require_complete_holding_period,
        )
    )
    holding_start_dates: list[pd.Timestamp | pd.NaT] = []
    holding_end_dates: list[pd.Timestamp | pd.NaT] = []
    holding_periods: list[int] = []
    horizon_complete: list[bool] = []

    for index, signal_date in enumerate(ticker_dates):
        start_index = index + 1
        end_index = index + horizon
        holding_start = ticker_dates[start_index] if start_index < len(ticker_dates) else pd.NaT
        holding_end = ticker_dates[end_index] if end_index < len(ticker_dates) else pd.NaT
        complete = (
            signal_date in scoped_signal_dates
            and end_index < len(ticker_dates)
            and pd.notna(output.loc[index, alignment.return_column])
        )

        holding_start_dates.append(holding_start)
        holding_end_dates.append(holding_end)
        holding_periods.append(max(min(end_index, len(ticker_dates) - 1) - index, 0))
        horizon_complete.append(bool(complete))

    output["prediction_date"] = output[alignment.date_column]
    output["signal_date"] = output[alignment.date_column]
    output["holding_start_date"] = holding_start_dates
    output["holding_end_date"] = holding_end_dates
    output["return_label_date"] = holding_end_dates
    output["realized_return_column"] = alignment.return_column
    output["return_horizon"] = horizon
    output["holding_periods"] = holding_periods
    output["horizon_complete"] = horizon_complete
    output["horizon_alignment_mode"] = alignment.mode
    return output


def _validate_horizon_aligned_return_timing(
    frame: pd.DataFrame,
    alignment: BacktestHorizonAlignment,
) -> None:
    if frame.empty:
        return
    signal_dates = pd.to_datetime(frame["signal_date"], errors="coerce").dt.normalize()
    holding_start = pd.to_datetime(
        frame["holding_start_date"], errors="coerce"
    ).dt.normalize()
    return_label = pd.to_datetime(
        frame["return_label_date"], errors="coerce"
    ).dt.normalize()
    complete = frame["horizon_complete"].fillna(False).astype(bool)

    invalid_start = complete & holding_start.notna() & signal_dates.notna() & (
        holding_start <= signal_dates
    )
    if invalid_start.any():
        raise ValueError("backtest alignment holding_start_date must be after signal_date")

    invalid_label = complete & return_label.notna() & signal_dates.notna() & (
        return_label <= signal_dates
    )
    if invalid_label.any():
        raise ValueError("backtest alignment return_label_date must be after signal_date")

    observed_horizon = pd.to_numeric(frame["return_horizon"], errors="coerce")
    mismatched_horizon = complete & observed_horizon.notna() & observed_horizon.ne(
        alignment.return_horizon
    )
    if mismatched_horizon.any():
        raise ValueError(
            "backtest alignment return_horizon must match realized forward return column"
        )

    realized_column = frame["realized_return_column"].dropna().astype(str)
    if realized_column.ne(alignment.return_column).any():
        raise ValueError(
            "backtest alignment realized_return_column must match configured return column"
        )


def _scheduled_signal_dates(
    dates: Iterable[pd.Timestamp],
    *,
    horizon: int,
    require_complete_holding_period: bool,
) -> set[pd.Timestamp]:
    schedule = generate_non_overlapping_rebalance_schedule(
        dates,
        return_horizon=horizon,
        require_complete_holding_period=require_complete_holding_period,
    )
    if schedule.empty:
        return set()
    return {
        pd.Timestamp(signal_date).normalize()
        for signal_date in schedule["signal_date"]
        if pd.notna(signal_date)
    }
