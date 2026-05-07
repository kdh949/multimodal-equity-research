from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

import pandas as pd

from quant_research.backtest.metrics import (
    PerformanceMetrics,
    calculate_cost_adjusted_returns,
    calculate_metrics,
)

DEFAULT_BENCHMARK_TICKER = "SPY"
EQUAL_WEIGHT_BASELINE_NAME = "equal_weight"
MARKET_BENCHMARK_BASELINE_TYPE = "market_benchmark"
EQUAL_WEIGHT_BASELINE_TYPE = "equal_weight_universe"
BASELINE_ALIGNMENT_RULES_SCHEMA_VERSION = "baseline_alignment_rules.v1"
STAGE1_REQUIRED_BASELINE_TYPES = (
    MARKET_BENCHMARK_BASELINE_TYPE,
    EQUAL_WEIGHT_BASELINE_TYPE,
)
BASELINE_ALIGNMENT_RULES = (
    {
        "rule_id": "candidate_dates",
        "description": (
            "Benchmark and equal-weight baselines are evaluated on the unique "
            "normalized strategy evaluation dates only."
        ),
    },
    {
        "rule_id": "benchmark_dates",
        "description": (
            "The market benchmark must provide the selected return column and "
            "horizon for every candidate evaluation date; dates outside the "
            "candidate sample are ignored for metric comparison."
        ),
    },
    {
        "rule_id": "equal_weight_dates",
        "description": (
            "The equal-weight universe baseline must provide the selected "
            "return column and horizon for every candidate evaluation date; "
            "dates outside the candidate sample are ignored for metric comparison."
        ),
    },
    {
        "rule_id": "ticker_universe",
        "description": (
            "The equal-weight baseline uses exactly the normalized strategy "
            "ticker universe. The benchmark ticker is a separate market "
            "benchmark input unless it is explicitly part of that strategy universe."
        ),
    },
)
BENCHMARK_RETURN_COLUMNS = (
    "date",
    "return_date",
    "benchmark_ticker",
    "return_column",
    "return_horizon",
    "benchmark_return",
    "benchmark_equity",
    "missing_benchmark_return",
)
EQUAL_WEIGHT_BASELINE_RETURN_COLUMNS = (
    "date",
    "return_date",
    "baseline_name",
    "return_column",
    "return_horizon",
    "equal_weight_return",
    "equal_weight_equity",
    "constituent_count",
    "expected_constituent_count",
    "missing_equal_weight_return",
    "incomplete_ticker_universe",
)
EQUAL_WEIGHT_BASELINE_EQUITY_COLUMNS = (
    "date",
    "return_date",
    "baseline_name",
    "portfolio_return",
    "cost_adjusted_return",
    "net_return",
    "equal_weight_return",
    "gross_return",
    "transaction_cost_return",
    "slippage_cost_return",
    "total_cost_return",
    "turnover_cost_return",
    "cost_bps",
    "slippage_bps",
    "realized_return_column",
    "return_column",
    "return_horizon",
    "benchmark_return",
    "cost_adjusted_benchmark_return",
    "benchmark_transaction_cost_return",
    "benchmark_slippage_cost_return",
    "benchmark_total_cost_return",
    "benchmark_turnover",
    "equity",
    "benchmark_equity",
    "cost_adjusted_benchmark_equity",
    "turnover",
    "exposure",
    "constituent_count",
    "expected_constituent_count",
    "missing_equal_weight_return",
    "incomplete_ticker_universe",
)


@dataclass(frozen=True)
class StrategyEvaluationWindow:
    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError("evaluation window start must be on or before end")

    @classmethod
    def from_bounds(cls, start: date | str | pd.Timestamp, end: date | str | pd.Timestamp) -> StrategyEvaluationWindow:
        return cls(start=_as_date(start), end=_as_date(end))

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame | None,
        *,
        date_column: str = "date",
        fallback: StrategyEvaluationWindow | None = None,
    ) -> StrategyEvaluationWindow:
        if frame is None or frame.empty or date_column not in frame:
            if fallback is None:
                raise ValueError("evaluation frame must contain at least one date")
            return fallback

        dates = pd.to_datetime(frame[date_column], errors="coerce").dropna()
        if dates.empty:
            if fallback is None:
                raise ValueError("evaluation frame must contain at least one valid date")
            return fallback
        return cls(start=dates.min().date(), end=dates.max().date())

    def to_dict(self) -> dict[str, str]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
        }


@dataclass(frozen=True)
class TickerUniverse:
    tickers: tuple[str, ...]
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER

    def __post_init__(self) -> None:
        normalized = _normalize_tickers(self.tickers)
        if not normalized:
            raise ValueError("ticker universe must contain at least one ticker")

        benchmark = _normalize_ticker(self.benchmark_ticker)
        if not benchmark:
            raise ValueError("benchmark ticker must not be blank")

        object.__setattr__(self, "tickers", normalized)
        object.__setattr__(self, "benchmark_ticker", benchmark)

    @property
    def benchmark_in_universe(self) -> bool:
        return self.benchmark_ticker in self.tickers

    @property
    def data_tickers(self) -> tuple[str, ...]:
        return _normalize_tickers((*self.tickers, self.benchmark_ticker))

    def to_dict(self) -> dict[str, object]:
        return {
            "tickers": list(self.tickers),
            "benchmark_ticker": self.benchmark_ticker,
            "benchmark_in_universe": self.benchmark_in_universe,
            "data_tickers": list(self.data_tickers),
        }


@dataclass(frozen=True)
class BaselineComparisonInput:
    name: str
    baseline_type: str
    return_basis: str
    return_column: str
    return_horizon: int
    data_source: str
    evaluation_window: StrategyEvaluationWindow | None = None
    benchmark_ticker: str | None = None
    universe_tickers: tuple[str, ...] = ()
    required_for_stage1: bool = True
    cost_bps: float | None = None
    slippage_bps: float | None = None
    construction_method: str = ""
    return_timing: str = "signal_date_returns_apply_to_configured_forward_return_horizon"

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("baseline comparison input name must not be blank")
        baseline_type = str(self.baseline_type).strip()
        if not baseline_type:
            raise ValueError("baseline comparison input type must not be blank")
        return_basis = str(self.return_basis).strip()
        if not return_basis:
            raise ValueError("baseline comparison input return basis must not be blank")
        return_column = str(self.return_column).strip()
        if not return_column:
            raise ValueError("baseline comparison input return column must not be blank")
        return_horizon = max(int(self.return_horizon), 1)
        data_source = str(self.data_source).strip()
        if not data_source:
            raise ValueError("baseline comparison input data source must not be blank")
        benchmark_ticker = (
            _normalize_ticker(self.benchmark_ticker)
            if self.benchmark_ticker is not None
            else None
        )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "baseline_type", baseline_type)
        object.__setattr__(self, "return_basis", return_basis)
        object.__setattr__(self, "return_column", return_column)
        object.__setattr__(self, "return_horizon", return_horizon)
        object.__setattr__(self, "data_source", data_source)
        object.__setattr__(self, "benchmark_ticker", benchmark_ticker)
        object.__setattr__(self, "universe_tickers", _normalize_tickers(self.universe_tickers))
        object.__setattr__(
            self,
            "cost_bps",
            _optional_float(self.cost_bps),
        )
        object.__setattr__(
            self,
            "slippage_bps",
            _optional_float(self.slippage_bps),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "baseline_type": self.baseline_type,
            "return_basis": self.return_basis,
            "return_column": self.return_column,
            "return_horizon": self.return_horizon,
            "data_source": self.data_source,
            "evaluation_window": (
                self.evaluation_window.to_dict()
                if self.evaluation_window is not None
                else None
            ),
            "benchmark_ticker": self.benchmark_ticker,
            "universe_tickers": list(self.universe_tickers),
            "required_for_stage1": self.required_for_stage1,
            "cost_bps": self.cost_bps,
            "slippage_bps": self.slippage_bps,
            "construction_method": self.construction_method,
            "return_timing": self.return_timing,
        }


@dataclass(frozen=True)
class BenchmarkConstructionInputs:
    evaluation_window: StrategyEvaluationWindow
    ticker_universe: TickerUniverse
    data_mode: str
    interval: str
    return_column: str = "forward_return_1"
    return_horizon: int = 1
    cost_bps: float | None = None
    slippage_bps: float | None = None
    baseline_comparison_inputs: tuple[BaselineComparisonInput, ...] = ()

    def __post_init__(self) -> None:
        baseline_inputs = self.baseline_comparison_inputs
        if not baseline_inputs:
            baseline_inputs = build_stage1_baseline_comparison_inputs(
                self.evaluation_window,
                self.ticker_universe,
                return_column=self.return_column,
                return_horizon=self.return_horizon,
                cost_bps=self.cost_bps,
                slippage_bps=self.slippage_bps,
            )
        object.__setattr__(
            self,
            "baseline_comparison_inputs",
            validate_stage1_baseline_comparison_inputs(baseline_inputs),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "evaluation_window": self.evaluation_window.to_dict(),
            "ticker_universe": self.ticker_universe.to_dict(),
            "data_mode": self.data_mode,
            "interval": self.interval,
            "return_column": self.return_column,
            "return_horizon": self.return_horizon,
            "cost_bps": self.cost_bps,
            "slippage_bps": self.slippage_bps,
            "required_baseline_types": list(STAGE1_REQUIRED_BASELINE_TYPES),
            "baseline_comparison_inputs": [
                baseline_input.to_dict()
                for baseline_input in self.baseline_comparison_inputs
            ],
        }


def build_stage1_baseline_comparison_inputs(
    evaluation_window: StrategyEvaluationWindow | None,
    ticker_universe: TickerUniverse | None,
    *,
    return_column: str = "forward_return_1",
    return_horizon: int | None = None,
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER,
    tickers: Iterable[str] = (),
    cost_bps: float | None = None,
    slippage_bps: float | None = None,
) -> tuple[BaselineComparisonInput, ...]:
    if ticker_universe is not None:
        universe_tickers = ticker_universe.tickers
        benchmark = ticker_universe.benchmark_ticker
    else:
        universe_tickers = _normalize_tickers(tickers)
        benchmark = _normalize_ticker(benchmark_ticker)
    horizon = return_horizon if return_horizon is not None else _return_horizon(return_column)

    return validate_stage1_baseline_comparison_inputs(
        (
            BaselineComparisonInput(
                name=benchmark,
                baseline_type=MARKET_BENCHMARK_BASELINE_TYPE,
                return_basis="cost_adjusted_benchmark_return",
                return_column=return_column,
                return_horizon=horizon,
                data_source="benchmark_return_series",
                evaluation_window=evaluation_window,
                benchmark_ticker=benchmark,
                cost_bps=cost_bps,
                slippage_bps=slippage_bps,
                construction_method="single_ticker_benchmark_buy_and_hold_aligned_to_strategy_dates",
            ),
            BaselineComparisonInput(
                name=EQUAL_WEIGHT_BASELINE_NAME,
                baseline_type=EQUAL_WEIGHT_BASELINE_TYPE,
                return_basis="cost_adjusted_equal_weight_return",
                return_column=return_column,
                return_horizon=horizon,
                data_source="equal_weight_baseline_return_series",
                evaluation_window=evaluation_window,
                universe_tickers=universe_tickers,
                cost_bps=cost_bps,
                slippage_bps=slippage_bps,
                construction_method="equal_weight_strategy_universe_aligned_to_strategy_dates",
            ),
        )
    )


def validate_stage1_baseline_comparison_inputs(
    baseline_inputs: Iterable[BaselineComparisonInput],
) -> tuple[BaselineComparisonInput, ...]:
    inputs = tuple(baseline_inputs)
    validation = evaluate_stage1_baseline_comparison_inputs(inputs)
    if validation["status"] != "pass":
        raise ValueError(str(validation["reason"]))
    return tuple(
        next(
            baseline_input
            for baseline_input in inputs
            if baseline_input.baseline_type == baseline_type
        )
        for baseline_type in STAGE1_REQUIRED_BASELINE_TYPES
    )


def evaluate_stage1_baseline_comparison_inputs(
    baseline_inputs: Iterable[BaselineComparisonInput],
) -> dict[str, object]:
    inputs = tuple(baseline_inputs)
    baseline_types = [baseline_input.baseline_type for baseline_input in inputs]
    missing = [
        baseline_type
        for baseline_type in STAGE1_REQUIRED_BASELINE_TYPES
        if baseline_type not in baseline_types
    ]
    duplicates = sorted(
        {
            baseline_type
            for baseline_type in baseline_types
            if baseline_types.count(baseline_type) > 1
        }
    )
    unknown = sorted(
        {
            baseline_type
            for baseline_type in baseline_types
            if baseline_type not in STAGE1_REQUIRED_BASELINE_TYPES
        }
    )
    reasons: list[str] = []
    if missing:
        reasons.append(f"missing required Stage 1 baseline inputs: {', '.join(missing)}")
    if duplicates:
        reasons.append(f"duplicate Stage 1 baseline inputs: {', '.join(duplicates)}")
    if unknown:
        reasons.append(f"unknown Stage 1 baseline inputs: {', '.join(unknown)}")

    return {
        "status": "hard_fail" if reasons else "pass",
        "reason": "; ".join(reasons) if reasons else "required Stage 1 baseline inputs are present",
        "reasons": reasons,
        "affects_system": True,
        "affects_strategy": False,
        "required_baseline_types": list(STAGE1_REQUIRED_BASELINE_TYPES),
        "provided_baseline_types": baseline_types,
        "baseline_names": [baseline_input.name for baseline_input in inputs],
        "baseline_input_count": len(inputs),
    }


def validate_baseline_alignment_rules(
    candidate_evaluation_dates: pd.DataFrame | pd.Series | Iterable[object],
    *,
    benchmark_return_series: pd.DataFrame | None,
    equal_weight_baseline_return_series: pd.DataFrame | None,
    strategy_tickers: Iterable[str],
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER,
    return_column: str = "forward_return_20",
    date_column: str = "date",
    actual_equal_weight_tickers: Iterable[str] | None = None,
    strict: bool = True,
) -> dict[str, object]:
    """Validate date and ticker-universe alignment for required return baselines.

    This utility defines the Stage 1 comparison contract used by the validity
    gate: SPY/market benchmark and equal-weight baseline metrics are comparable
    only when both baselines cover the same candidate strategy dates and the
    equal-weight baseline is constructed from the configured strategy universe.
    """

    candidate_dates = _normalize_evaluation_dates(
        candidate_evaluation_dates,
        date_column=date_column,
    )
    expected_universe = _normalize_tickers(strategy_tickers)
    actual_universe = (
        _normalize_tickers(actual_equal_weight_tickers)
        if actual_equal_weight_tickers is not None
        else expected_universe
    )
    benchmark = _normalize_ticker(benchmark_ticker)
    horizon = _return_horizon(return_column)

    benchmark_dates = _baseline_series_dates(
        benchmark_return_series,
        date_column=date_column,
        return_column=return_column,
        return_horizon=horizon,
        value_column="benchmark_return",
        ticker_column="benchmark_ticker",
        ticker_value=benchmark,
    )
    equal_weight_dates = _baseline_series_dates(
        equal_weight_baseline_return_series,
        date_column=date_column,
        return_column=return_column,
        return_horizon=horizon,
        value_column="equal_weight_return",
        name_column="baseline_name",
        name_value=EQUAL_WEIGHT_BASELINE_NAME,
    )
    benchmark_alignment = _evaluate_date_alignment(
        candidate_dates,
        benchmark_dates,
        baseline_name=benchmark,
        strict=strict,
    )
    equal_weight_alignment = _evaluate_date_alignment(
        candidate_dates,
        equal_weight_dates,
        baseline_name=EQUAL_WEIGHT_BASELINE_NAME,
        strict=strict,
    )
    universe_alignment = _evaluate_equal_weight_universe_alignment(
        expected_universe,
        actual_universe,
        benchmark,
        strict=strict,
    )
    checks = {
        benchmark: benchmark_alignment,
        EQUAL_WEIGHT_BASELINE_NAME: {
            **equal_weight_alignment,
            "ticker_universe_alignment": universe_alignment,
        },
        "ticker_universe": universe_alignment,
    }
    failed_checks = [
        name
        for name, check in checks.items()
        if isinstance(check, dict) and check.get("status") == "hard_fail"
    ]
    not_evaluable_checks = [
        name
        for name, check in checks.items()
        if isinstance(check, dict) and check.get("status") == "not_evaluable"
    ]
    status = "pass"
    if failed_checks:
        status = "hard_fail"
    elif not_evaluable_checks:
        status = "not_evaluable"

    return {
        "schema_version": BASELINE_ALIGNMENT_RULES_SCHEMA_VERSION,
        "status": status,
        "reason": _baseline_alignment_reason(status, failed_checks, not_evaluable_checks),
        "rules": list(BASELINE_ALIGNMENT_RULES),
        "return_column": return_column,
        "return_horizon": horizon,
        "candidate_dates": _date_iso_values(candidate_dates),
        "benchmark_ticker": benchmark,
        "strategy_tickers": list(expected_universe),
        "benchmark_in_strategy_universe": benchmark in expected_universe,
        "baselines": checks,
        "failed_checks": failed_checks,
        "not_evaluable_checks": not_evaluable_checks,
    }


def build_benchmark_construction_inputs(
    config: object,
    *,
    evaluation_frame: pd.DataFrame | None = None,
    date_column: str = "date",
) -> BenchmarkConstructionInputs:
    requested_window = StrategyEvaluationWindow.from_bounds(
        config.start,
        config.end,
    )
    evaluation_window = StrategyEvaluationWindow.from_frame(
        evaluation_frame,
        date_column=date_column,
        fallback=requested_window,
    )
    return BenchmarkConstructionInputs(
        evaluation_window=evaluation_window,
        ticker_universe=TickerUniverse(
            tickers=tuple(config.tickers),
            benchmark_ticker=getattr(config, "benchmark_ticker", DEFAULT_BENCHMARK_TICKER),
        ),
        data_mode=str(config.data_mode),
        interval=str(config.interval),
        return_column=str(getattr(config, "prediction_target_column", "forward_return_20")),
        return_horizon=_return_horizon(str(getattr(config, "prediction_target_column", "forward_return_20"))),
        cost_bps=_optional_float(getattr(config, "cost_bps", None)),
        slippage_bps=_optional_float(getattr(config, "slippage_bps", None)),
    )


def build_benchmark_return_series(
    price_data: pd.DataFrame,
    evaluation_dates: pd.DataFrame | pd.Series | Iterable[object],
    *,
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER,
    date_column: str = "date",
    return_column: str = "forward_return_1",
) -> pd.DataFrame:
    """Build benchmark returns on exactly the strategy evaluation dates and horizon."""

    dates = _normalize_evaluation_dates(evaluation_dates, date_column=date_column)
    benchmark = _normalize_ticker(benchmark_ticker)
    return_horizon = _return_horizon(return_column)
    aligned = pd.DataFrame({"date": dates})
    if aligned.empty:
        return pd.DataFrame(columns=BENCHMARK_RETURN_COLUMNS)

    aligned["benchmark_ticker"] = benchmark
    aligned["return_column"] = return_column
    aligned["return_horizon"] = return_horizon
    if price_data.empty:
        return _finalize_benchmark_return_series(aligned)

    missing = {"date", "ticker"}.difference(price_data.columns)
    if missing:
        raise ValueError(f"benchmark price data missing required columns: {sorted(missing)}")

    frame = price_data.copy()
    frame["date"] = _normalize_datetime_values(frame["date"])
    frame["ticker"] = frame["ticker"].map(_normalize_ticker)
    frame = frame[frame["ticker"] == benchmark].sort_values("date")
    if frame.empty:
        return _finalize_benchmark_return_series(aligned)

    if return_column not in frame.columns:
        price_column = "adj_close" if "adj_close" in frame.columns else "close"
        if price_column not in frame.columns:
            raise ValueError(
                f"benchmark price data must include '{return_column}', 'adj_close', or 'close'"
            )
        prices = pd.to_numeric(frame[price_column], errors="coerce")
        frame[return_column] = prices.shift(-return_horizon) / prices - 1

    frame["return_date"] = frame["date"].shift(-return_horizon)
    frame[return_column] = pd.to_numeric(frame[return_column], errors="coerce")
    returns = frame[["date", "return_date", return_column]].drop_duplicates("date", keep="last")
    returns = returns.rename(columns={return_column: "benchmark_return"})

    aligned = aligned.merge(returns, on="date", how="left")
    return _finalize_benchmark_return_series(aligned)


def build_equal_weight_baseline_return_series(
    price_data: pd.DataFrame,
    evaluation_dates: pd.DataFrame | pd.Series | Iterable[object],
    tickers: Iterable[str],
    *,
    date_column: str = "date",
    return_column: str = "forward_return_1",
) -> pd.DataFrame:
    """Build equal-weight returns for the strategy ticker universe.

    The output is aligned to exactly the strategy evaluation dates. Returns are
    averaged across strategy tickers only, so external benchmark symbols loaded
    for comparison, such as SPY, do not enter the equal-weight baseline unless
    they are also part of the configured strategy universe.
    """

    dates = _normalize_evaluation_dates(evaluation_dates, date_column=date_column)
    ticker_universe = _normalize_tickers(tickers)
    return_horizon = _return_horizon(return_column)
    if not ticker_universe:
        raise ValueError("strategy ticker universe must contain at least one ticker")

    aligned = pd.DataFrame({"date": dates})
    if aligned.empty:
        return pd.DataFrame(columns=EQUAL_WEIGHT_BASELINE_RETURN_COLUMNS)

    aligned["baseline_name"] = EQUAL_WEIGHT_BASELINE_NAME
    aligned["return_column"] = return_column
    aligned["return_horizon"] = return_horizon
    aligned["expected_constituent_count"] = len(ticker_universe)
    if price_data.empty:
        return _finalize_equal_weight_baseline_return_series(aligned)

    missing = {"date", "ticker"}.difference(price_data.columns)
    if missing:
        raise ValueError(f"equal-weight price data missing required columns: {sorted(missing)}")

    frame = price_data.copy()
    frame["date"] = _normalize_datetime_values(frame["date"])
    frame["ticker"] = frame["ticker"].map(_normalize_ticker)
    frame = frame[frame["ticker"].isin(set(ticker_universe))].sort_values(["ticker", "date"])
    if frame.empty:
        return _finalize_equal_weight_baseline_return_series(aligned)

    if return_column not in frame.columns:
        price_column = "adj_close" if "adj_close" in frame.columns else "close"
        if price_column not in frame.columns:
            raise ValueError(
                f"equal-weight price data must include '{return_column}', 'adj_close', or 'close'"
            )
        frame[price_column] = pd.to_numeric(frame[price_column], errors="coerce")
        frame[return_column] = frame.groupby("ticker", group_keys=False)[price_column].transform(
            lambda series: series.shift(-return_horizon) / series - 1
        )

    frame["return_date"] = frame.groupby("ticker")["date"].shift(-return_horizon)
    frame[return_column] = pd.to_numeric(frame[return_column], errors="coerce")
    returns = frame[["date", "ticker", "return_date", return_column]].drop_duplicates(
        ["date", "ticker"],
        keep="last",
    )
    returns_by_date = (
        returns.groupby("date", as_index=False)
        .agg(
            return_date=("return_date", "max"),
            equal_weight_return=(return_column, "mean"),
            constituent_count=(return_column, "count"),
        )
        .reset_index(drop=True)
    )

    aligned = aligned.merge(returns_by_date, on="date", how="left")
    return _finalize_equal_weight_baseline_return_series(aligned)


def construct_benchmark_return_series(
    price_data: pd.DataFrame,
    evaluation_dates: pd.DataFrame | pd.Series | Iterable[object],
    *,
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER,
    date_column: str = "date",
    return_column: str = "forward_return_1",
) -> pd.DataFrame:
    return build_benchmark_return_series(
        price_data,
        evaluation_dates,
        benchmark_ticker=benchmark_ticker,
        date_column=date_column,
        return_column=return_column,
    )


def construct_equal_weight_baseline_return_series(
    price_data: pd.DataFrame,
    evaluation_dates: pd.DataFrame | pd.Series | Iterable[object],
    tickers: Iterable[str],
    *,
    date_column: str = "date",
    return_column: str = "forward_return_1",
) -> pd.DataFrame:
    return build_equal_weight_baseline_return_series(
        price_data,
        evaluation_dates,
        tickers,
        date_column=date_column,
        return_column=return_column,
    )


def construct_spy_baseline_return_series(
    price_data: pd.DataFrame,
    evaluation_dates: pd.DataFrame | pd.Series | Iterable[object],
    *,
    date_column: str = "date",
    return_column: str = "forward_return_1",
) -> pd.DataFrame:
    return build_benchmark_return_series(
        price_data,
        evaluation_dates,
        benchmark_ticker=DEFAULT_BENCHMARK_TICKER,
        date_column=date_column,
        return_column=return_column,
    )


def build_equal_weight_baseline_equity_curve(
    equal_weight_baseline_return_series: pd.DataFrame,
    *,
    benchmark_return_series: pd.DataFrame | None = None,
    cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> pd.DataFrame:
    """Convert equal-weight universe returns to a backtest-like equity curve.

    The resulting frame uses the same return/cost column contract consumed by
    ``calculate_metrics`` for strategy backtests, making the equal-weight
    baseline directly comparable to existing backtest and validity outputs.
    """

    if equal_weight_baseline_return_series.empty:
        return pd.DataFrame(columns=EQUAL_WEIGHT_BASELINE_EQUITY_COLUMNS)
    required = {"date", "equal_weight_return"}
    missing = required.difference(equal_weight_baseline_return_series.columns)
    if missing:
        raise ValueError(f"equal-weight baseline series missing required columns: {sorted(missing)}")

    frame = equal_weight_baseline_return_series.copy()
    frame["date"] = _normalize_datetime_values(frame["date"])
    frame = frame.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date")
    if frame.empty:
        return pd.DataFrame(columns=EQUAL_WEIGHT_BASELINE_EQUITY_COLUMNS)
    if "return_date" not in frame:
        frame["return_date"] = pd.NaT
    if "baseline_name" not in frame:
        frame["baseline_name"] = EQUAL_WEIGHT_BASELINE_NAME
    if "return_column" not in frame:
        frame["return_column"] = "forward_return_1"
    if "return_horizon" not in frame:
        frame["return_horizon"] = frame["return_column"].map(_return_horizon)
    if "constituent_count" not in frame:
        frame["constituent_count"] = 0
    if "expected_constituent_count" not in frame:
        frame["expected_constituent_count"] = frame["constituent_count"]
    if "missing_equal_weight_return" not in frame:
        frame["missing_equal_weight_return"] = frame["equal_weight_return"].isna()
    if "incomplete_ticker_universe" not in frame:
        frame["incomplete_ticker_universe"] = (
            pd.to_numeric(frame["constituent_count"], errors="coerce").fillna(0)
            < pd.to_numeric(frame["expected_constituent_count"], errors="coerce").fillna(0)
        )

    gross_returns = pd.to_numeric(frame["equal_weight_return"], errors="coerce").fillna(0.0)
    constituent_count = pd.to_numeric(frame["constituent_count"], errors="coerce").fillna(0).astype(int)
    turnover = _equal_weight_turnover_from_constituent_counts(constituent_count)
    cost_adjusted = calculate_cost_adjusted_returns(
        gross_returns,
        turnover,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
    ).reset_index(drop=True)
    benchmark = _aligned_benchmark_cost_adjustment(
        frame,
        benchmark_return_series=benchmark_return_series,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
    )
    output = pd.DataFrame(
        {
            "date": frame["date"].reset_index(drop=True),
            "return_date": _normalize_datetime_values(frame["return_date"]).reset_index(drop=True),
            "baseline_name": frame["baseline_name"].astype(str).reset_index(drop=True),
            "portfolio_return": cost_adjusted["cost_adjusted_return"],
            "cost_adjusted_return": cost_adjusted["cost_adjusted_return"],
            "net_return": cost_adjusted["cost_adjusted_return"],
            "equal_weight_return": gross_returns.reset_index(drop=True),
            "gross_return": gross_returns.reset_index(drop=True),
            "transaction_cost_return": cost_adjusted["transaction_cost_return"],
            "slippage_cost_return": cost_adjusted["slippage_cost_return"],
            "total_cost_return": cost_adjusted["total_cost_return"],
            "turnover_cost_return": cost_adjusted["turnover_cost_return"],
            "cost_bps": float(cost_bps),
            "slippage_bps": float(slippage_bps),
            "realized_return_column": frame["return_column"].astype(str).reset_index(drop=True),
            "return_column": frame["return_column"].astype(str).reset_index(drop=True),
            "return_horizon": pd.to_numeric(frame["return_horizon"], errors="coerce").fillna(1).astype(int).reset_index(drop=True),
            "benchmark_return": benchmark["benchmark_return"],
            "cost_adjusted_benchmark_return": benchmark["cost_adjusted_benchmark_return"],
            "benchmark_transaction_cost_return": benchmark["benchmark_transaction_cost_return"],
            "benchmark_slippage_cost_return": benchmark["benchmark_slippage_cost_return"],
            "benchmark_total_cost_return": benchmark["benchmark_total_cost_return"],
            "benchmark_turnover": benchmark["benchmark_turnover"],
            "equity": (1.0 + cost_adjusted["cost_adjusted_return"]).cumprod(),
            "benchmark_equity": (1.0 + benchmark["benchmark_return"]).cumprod(),
            "cost_adjusted_benchmark_equity": (1.0 + benchmark["cost_adjusted_benchmark_return"]).cumprod(),
            "turnover": turnover.reset_index(drop=True),
            "exposure": (constituent_count > 0).astype(float).reset_index(drop=True),
            "constituent_count": constituent_count.reset_index(drop=True),
            "expected_constituent_count": pd.to_numeric(frame["expected_constituent_count"], errors="coerce").fillna(0).astype(int).reset_index(drop=True),
            "missing_equal_weight_return": frame["missing_equal_weight_return"].fillna(False).astype(bool).reset_index(drop=True),
            "incomplete_ticker_universe": frame["incomplete_ticker_universe"].fillna(False).astype(bool).reset_index(drop=True),
        }
    )
    return output.loc[:, EQUAL_WEIGHT_BASELINE_EQUITY_COLUMNS].reset_index(drop=True)


def calculate_equal_weight_baseline_performance_metrics(
    equal_weight_baseline_return_series: pd.DataFrame,
    *,
    benchmark_return_series: pd.DataFrame | None = None,
    cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> PerformanceMetrics:
    equity_curve = build_equal_weight_baseline_equity_curve(
        equal_weight_baseline_return_series,
        benchmark_return_series=benchmark_return_series,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
    )
    return calculate_metrics(equity_curve)


def _aligned_benchmark_cost_adjustment(
    equal_weight_frame: pd.DataFrame,
    *,
    benchmark_return_series: pd.DataFrame | None,
    cost_bps: float,
    slippage_bps: float,
) -> pd.DataFrame:
    returns = pd.Series(0.0, index=equal_weight_frame.index, dtype=float)
    has_benchmark_observation = False
    if benchmark_return_series is not None and not benchmark_return_series.empty:
        if {"date", "benchmark_return"}.issubset(benchmark_return_series.columns):
            benchmark = benchmark_return_series.copy()
            benchmark["date"] = _normalize_datetime_values(benchmark["date"])
            benchmark["benchmark_return"] = pd.to_numeric(
                benchmark["benchmark_return"],
                errors="coerce",
            )
            if "return_column" in equal_weight_frame and "return_column" in benchmark:
                return_columns = set(equal_weight_frame["return_column"].dropna().astype(str))
                benchmark = benchmark[benchmark["return_column"].astype(str).isin(return_columns)]
            if "return_horizon" in equal_weight_frame and "return_horizon" in benchmark:
                return_horizons = set(
                    pd.to_numeric(equal_weight_frame["return_horizon"], errors="coerce")
                    .dropna()
                    .astype(int)
                )
                benchmark_horizons = pd.to_numeric(benchmark["return_horizon"], errors="coerce")
                benchmark = benchmark[benchmark_horizons.isin(return_horizons)]
            benchmark = benchmark.dropna(subset=["date"]).drop_duplicates("date", keep="last")
            aligned = equal_weight_frame[["date"]].merge(
                benchmark[["date", "benchmark_return"]],
                on="date",
                how="left",
            )
            has_benchmark_observation = bool(aligned["benchmark_return"].notna().any())
            returns = aligned["benchmark_return"].fillna(0.0).astype(float)

    turnover = (
        _buy_and_hold_turnover(returns)
        if has_benchmark_observation
        else pd.Series(0.0, index=returns.index, dtype=float)
    )
    costs = calculate_cost_adjusted_returns(
        returns,
        turnover,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
    ).reset_index(drop=True)
    return pd.DataFrame(
        {
            "benchmark_return": returns.reset_index(drop=True),
            "cost_adjusted_benchmark_return": costs["cost_adjusted_return"],
            "benchmark_transaction_cost_return": costs["transaction_cost_return"],
            "benchmark_slippage_cost_return": costs["slippage_cost_return"],
            "benchmark_total_cost_return": costs["total_cost_return"],
            "benchmark_turnover": turnover.reset_index(drop=True),
        }
    )


def _buy_and_hold_turnover(gross_returns: pd.Series) -> pd.Series:
    turnover = pd.Series(0.0, index=gross_returns.index, dtype=float)
    if not gross_returns.empty:
        turnover.iloc[0] = 1.0
    return turnover


def _equal_weight_turnover_from_constituent_counts(counts: pd.Series) -> pd.Series:
    previous_count = 0
    turnovers: list[float] = []
    for value in counts:
        current_count = int(value) if pd.notna(value) else 0
        if current_count <= 0:
            turnovers.append(1.0 if previous_count > 0 else 0.0)
        elif previous_count <= 0:
            turnovers.append(1.0)
        elif current_count == previous_count:
            turnovers.append(0.0)
        else:
            shared = min(previous_count, current_count)
            exited = max(previous_count - current_count, 0)
            entered = max(current_count - previous_count, 0)
            turnovers.append(
                float(
                    shared * abs((1.0 / current_count) - (1.0 / previous_count))
                    + exited * (1.0 / previous_count)
                    + entered * (1.0 / current_count)
                )
            )
        previous_count = current_count
    return pd.Series(turnovers, index=counts.index, dtype=float)


def _baseline_series_dates(
    frame: pd.DataFrame | None,
    *,
    date_column: str,
    return_column: str,
    return_horizon: int,
    value_column: str,
    ticker_column: str | None = None,
    ticker_value: str | None = None,
    name_column: str | None = None,
    name_value: str | None = None,
) -> pd.Series:
    if frame is None or frame.empty or date_column not in frame or value_column not in frame:
        return pd.Series(dtype="datetime64[ns]")

    aligned = frame.copy()
    aligned[date_column] = _normalize_datetime_values(aligned[date_column])
    if "return_column" in aligned:
        aligned = aligned[aligned["return_column"].astype(str) == str(return_column)]
    if "return_horizon" in aligned:
        horizons = pd.to_numeric(aligned["return_horizon"], errors="coerce")
        aligned = aligned[horizons == int(return_horizon)]
    if ticker_column is not None and ticker_column in aligned and ticker_value is not None:
        aligned[ticker_column] = aligned[ticker_column].map(_normalize_ticker)
        aligned = aligned[aligned[ticker_column] == _normalize_ticker(ticker_value)]
    if name_column is not None and name_column in aligned and name_value is not None:
        aligned = aligned[aligned[name_column].astype(str) == str(name_value)]

    values = pd.to_numeric(aligned[value_column], errors="coerce")
    return _normalize_datetime_values(aligned.loc[values.notna(), date_column])


def _evaluate_date_alignment(
    candidate_dates: pd.Series,
    baseline_dates: pd.Series,
    *,
    baseline_name: str,
    strict: bool,
) -> dict[str, object]:
    candidate = _date_iso_values(candidate_dates)
    baseline = _date_iso_values(baseline_dates)
    candidate_set = set(candidate)
    baseline_set = set(baseline)
    missing_candidate_dates = [value for value in candidate if value not in baseline_set]
    extra_baseline_dates = [value for value in baseline if value not in candidate_set]
    aligned_dates = [value for value in candidate if value in baseline_set]
    base_result = {
        "baseline": str(baseline_name),
        "candidate_dates": candidate,
        "baseline_dates": baseline,
        "aligned_dates": aligned_dates,
        "missing_candidate_dates": missing_candidate_dates,
        "extra_baseline_dates": extra_baseline_dates,
        "candidate_sample_count": len(candidate),
        "baseline_sample_count": len(baseline),
        "aligned_sample_count": len(aligned_dates),
        "missing_candidate_sample_count": len(missing_candidate_dates),
        "extra_baseline_sample_count": len(extra_baseline_dates),
    }
    if not candidate:
        return {
            **base_result,
            "status": "not_evaluable",
            "reason": "candidate evaluation dates are unavailable",
        }
    if missing_candidate_dates and strict:
        return {
            **base_result,
            "status": "hard_fail",
            "reason": (
                "baseline is missing required candidate evaluation date(s): "
                f"{', '.join(missing_candidate_dates)}"
            ),
        }
    if missing_candidate_dates:
        return {
            **base_result,
            "status": "not_evaluable",
            "reason": (
                "baseline sample dates do not cover candidate evaluation date(s): "
                f"{', '.join(missing_candidate_dates)}"
            ),
        }
    if strict and not baseline:
        return {
            **base_result,
            "status": "hard_fail",
            "reason": "baseline has no samples for candidate evaluation dates",
        }
    if not baseline:
        return {
            **base_result,
            "status": "not_evaluable",
            "reason": "baseline sample dates are unavailable",
        }
    return {
        **base_result,
        "status": "pass",
        "reason": (
            "baseline samples align to candidate dates"
            if not extra_baseline_dates
            else "baseline samples align after ignoring dates outside candidate evaluation"
        ),
    }


def _evaluate_equal_weight_universe_alignment(
    expected_universe: tuple[str, ...],
    actual_universe: tuple[str, ...],
    benchmark_ticker: str,
    *,
    strict: bool,
) -> dict[str, object]:
    expected_set = set(expected_universe)
    actual_set = set(actual_universe)
    missing_tickers = [ticker for ticker in expected_universe if ticker not in actual_set]
    extra_tickers = [ticker for ticker in actual_universe if ticker not in expected_set]
    base_result = {
        "expected_universe_tickers": list(expected_universe),
        "actual_universe_tickers": list(actual_universe),
        "benchmark_ticker": benchmark_ticker,
        "benchmark_in_strategy_universe": benchmark_ticker in expected_set,
        "benchmark_is_data_only": benchmark_ticker not in expected_set,
        "missing_universe_tickers": missing_tickers,
        "extra_universe_tickers": extra_tickers,
        "expected_universe_count": len(expected_universe),
        "actual_universe_count": len(actual_universe),
    }
    if not expected_universe:
        return {
            **base_result,
            "status": "hard_fail" if strict else "not_evaluable",
            "reason": "strategy ticker universe is unavailable",
        }
    if missing_tickers or extra_tickers:
        status = "hard_fail" if strict else "not_evaluable"
        return {
            **base_result,
            "status": status,
            "reason": (
                "equal-weight baseline ticker universe must match the strategy universe"
            ),
        }
    return {
        **base_result,
        "status": "pass",
        "reason": "equal-weight baseline ticker universe matches the strategy universe",
    }


def _baseline_alignment_reason(
    status: str,
    failed_checks: list[str],
    not_evaluable_checks: list[str],
) -> str:
    if status == "hard_fail":
        return f"baseline alignment failed for required check(s): {', '.join(failed_checks)}"
    if status == "not_evaluable":
        return (
            "baseline alignment is not evaluable for check(s): "
            f"{', '.join(not_evaluable_checks)}"
        )
    return "benchmark and equal-weight baselines align to candidate dates and strategy universe"


def _date_iso_values(dates: pd.Series) -> list[str]:
    values = _normalize_datetime_values(dates).dropna().drop_duplicates().sort_values()
    return [pd.Timestamp(value).date().isoformat() for value in values]


def _normalize_tickers(tickers: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        value = _normalize_ticker(ticker)
        if value and value not in seen:
            normalized.append(value)
            seen.add(value)
    return tuple(normalized)


def _normalize_ticker(ticker: str | object) -> str:
    return str(ticker).strip().upper()


def _return_horizon(return_column: str) -> int:
    prefix = "forward_return_"
    if not str(return_column).startswith(prefix):
        return 1
    try:
        horizon = int(str(return_column).removeprefix(prefix))
    except ValueError:
        return 1
    return max(horizon, 1)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_date(value: date | str | pd.Timestamp) -> date:
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    return pd.Timestamp(value).date()


def _normalize_evaluation_dates(
    evaluation_dates: pd.DataFrame | pd.Series | Iterable[object],
    *,
    date_column: str,
) -> pd.Series:
    if isinstance(evaluation_dates, pd.DataFrame):
        if date_column not in evaluation_dates:
            raise ValueError(f"evaluation dates frame must include '{date_column}'")
        raw_dates = evaluation_dates[date_column]
    elif isinstance(evaluation_dates, pd.Series):
        raw_dates = evaluation_dates
    else:
        raw_dates = list(evaluation_dates)

    dates = _normalize_datetime_values(raw_dates).dropna().drop_duplicates().sort_values()
    return dates.reset_index(drop=True)


def _normalize_datetime_values(values: object) -> pd.Series:
    dates = pd.Series(pd.to_datetime(values, errors="coerce"))
    if dates.empty:
        return dates
    dates = dates.map(_drop_timezone)
    return pd.to_datetime(dates, errors="coerce").dt.normalize()


def _drop_timezone(value: object) -> pd.Timestamp | object:
    if pd.isna(value):
        return pd.NaT
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        return timestamp.tz_convert(None)
    return timestamp


def _finalize_benchmark_return_series(aligned: pd.DataFrame) -> pd.DataFrame:
    frame = aligned.copy()
    if "return_date" not in frame:
        frame["return_date"] = pd.NaT
    if "return_column" not in frame:
        frame["return_column"] = "forward_return_1"
    if "return_horizon" not in frame:
        frame["return_horizon"] = _return_horizon(str(frame["return_column"].iloc[0]))
    if "benchmark_return" not in frame:
        frame["benchmark_return"] = pd.NA
    frame["benchmark_return"] = pd.to_numeric(frame["benchmark_return"], errors="coerce")
    frame["return_horizon"] = pd.to_numeric(frame["return_horizon"], errors="coerce").fillna(1).astype(int)
    frame["missing_benchmark_return"] = frame["benchmark_return"].isna()
    frame["benchmark_equity"] = (1 + frame["benchmark_return"].fillna(0.0)).cumprod()
    frame["date"] = _normalize_datetime_values(frame["date"])
    frame["return_date"] = _normalize_datetime_values(frame["return_date"])
    return frame.loc[:, BENCHMARK_RETURN_COLUMNS].reset_index(drop=True)


def _finalize_equal_weight_baseline_return_series(aligned: pd.DataFrame) -> pd.DataFrame:
    frame = aligned.copy()
    if "return_date" not in frame:
        frame["return_date"] = pd.NaT
    if "baseline_name" not in frame:
        frame["baseline_name"] = EQUAL_WEIGHT_BASELINE_NAME
    if "return_column" not in frame:
        frame["return_column"] = "forward_return_1"
    if "return_horizon" not in frame:
        frame["return_horizon"] = _return_horizon(str(frame["return_column"].iloc[0]))
    if "equal_weight_return" not in frame:
        frame["equal_weight_return"] = pd.NA
    if "constituent_count" not in frame:
        frame["constituent_count"] = 0
    if "expected_constituent_count" not in frame:
        frame["expected_constituent_count"] = 0

    frame["equal_weight_return"] = pd.to_numeric(frame["equal_weight_return"], errors="coerce")
    frame["return_horizon"] = pd.to_numeric(frame["return_horizon"], errors="coerce").fillna(1).astype(int)
    frame["constituent_count"] = (
        pd.to_numeric(frame["constituent_count"], errors="coerce").fillna(0).astype(int)
    )
    frame["expected_constituent_count"] = (
        pd.to_numeric(frame["expected_constituent_count"], errors="coerce").fillna(0).astype(int)
    )
    frame["missing_equal_weight_return"] = frame["equal_weight_return"].isna()
    frame["incomplete_ticker_universe"] = frame["constituent_count"] < frame["expected_constituent_count"]
    frame["equal_weight_equity"] = (1 + frame["equal_weight_return"].fillna(0.0)).cumprod()
    frame["date"] = _normalize_datetime_values(frame["date"])
    frame["return_date"] = _normalize_datetime_values(frame["return_date"])
    return frame.loc[:, EQUAL_WEIGHT_BASELINE_RETURN_COLUMNS].reset_index(drop=True)
