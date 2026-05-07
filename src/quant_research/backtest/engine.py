from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_research.backtest.alignment import forward_return_horizon
from quant_research.backtest.covariance import (
    estimate_correlation_matrix,
    estimate_covariance_matrix,
)
from quant_research.backtest.metrics import (
    PerformanceMetrics,
    calculate_cost_adjusted_returns,
    calculate_metrics,
    calculate_portfolio_turnover,
)
from quant_research.data.timestamps import (
    date_end_utc,
    timestamp_utc,
    validate_event_availability_order,
)
from quant_research.signals.engine import DeterministicSignalEngine, SignalEngineConfig
from quant_research.validation.config import DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG


@dataclass(frozen=True)
class BacktestConfig:
    risk_constraint_schema_version: str = (
        DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG.schema_version
    )
    top_n: int = DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG.max_holdings
    cost_bps: float = 5.0
    slippage_bps: float = 2.0
    average_daily_turnover_budget: float = 0.25
    benchmark_ticker: str = "SPY"
    realized_return_column: str = "forward_return_20"
    max_symbol_weight: float = DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG.max_symbol_weight
    max_sector_weight: float = DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG.max_sector_weight
    max_correlation_cluster_weight: float = 0.70
    correlation_cluster_threshold: float = 0.80
    portfolio_covariance_lookback: int = (
        DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG.portfolio_covariance_lookback
    )
    covariance_aware_risk_enabled: bool = (
        DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG.covariance_aware_risk.enabled
    )
    covariance_return_column: str = (
        DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG.covariance_aware_risk.return_column
    )
    covariance_min_periods: int = (
        DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG.covariance_aware_risk.min_periods
    )
    max_daily_turnover: float | None = None
    portfolio_volatility_limit: float = (
        DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG.portfolio_volatility_limit
    )
    max_drawdown_stop: float = DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG.max_drawdown_stop
    max_position_risk_contribution: float = (
        DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG.max_position_risk_contribution
    )
    volatility_adjustment_strength: float = (
        DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG.adjustment.volatility_scale_strength
    )
    concentration_adjustment_strength: float = (
        DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG.adjustment.concentration_scale_strength
    )
    risk_contribution_adjustment_strength: float = (
        DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG.adjustment.risk_contribution_scale_strength
    )

    def __post_init__(self) -> None:
        if self.risk_constraint_schema_version != (
            DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG.schema_version
        ):
            raise ValueError(
                "risk_constraint_schema_version must match the canonical portfolio "
                "risk constraint schema"
            )
        object.__setattr__(self, "top_n", _positive_int(self.top_n, "top_n"))
        object.__setattr__(self, "cost_bps", _non_negative_bps(self.cost_bps, "cost_bps"))
        object.__setattr__(
            self,
            "slippage_bps",
            _non_negative_bps(self.slippage_bps, "slippage_bps"),
        )
        object.__setattr__(
            self,
            "average_daily_turnover_budget",
            _fraction(self.average_daily_turnover_budget, "average_daily_turnover_budget"),
        )
        benchmark_ticker = str(self.benchmark_ticker).strip().upper()
        if not benchmark_ticker:
            raise ValueError("benchmark_ticker must not be blank")
        object.__setattr__(self, "benchmark_ticker", benchmark_ticker)
        _validate_realized_return_column(self.realized_return_column)
        object.__setattr__(
            self, "max_symbol_weight", _fraction(self.max_symbol_weight, "max_symbol_weight")
        )
        object.__setattr__(
            self, "max_sector_weight", _fraction(self.max_sector_weight, "max_sector_weight")
        )
        object.__setattr__(
            self,
            "max_correlation_cluster_weight",
            _fraction(self.max_correlation_cluster_weight, "max_correlation_cluster_weight"),
        )
        object.__setattr__(
            self,
            "correlation_cluster_threshold",
            _fraction(self.correlation_cluster_threshold, "correlation_cluster_threshold"),
        )
        object.__setattr__(
            self,
            "portfolio_covariance_lookback",
            _positive_int(self.portfolio_covariance_lookback, "portfolio_covariance_lookback"),
        )
        covariance_return_column = str(self.covariance_return_column).strip()
        if not covariance_return_column:
            raise ValueError("covariance_return_column must not be blank")
        object.__setattr__(self, "covariance_return_column", covariance_return_column)
        object.__setattr__(
            self,
            "covariance_min_periods",
            _positive_int(self.covariance_min_periods, "covariance_min_periods"),
        )
        if (
            self.covariance_min_periods
            == DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG.covariance_aware_risk.min_periods
            and self.portfolio_covariance_lookback < self.covariance_min_periods
        ):
            object.__setattr__(
                self,
                "covariance_min_periods",
                max(int(self.portfolio_covariance_lookback), 2),
            )
        if self.covariance_min_periods < 2:
            raise ValueError("covariance_min_periods must be at least 2")
        if self.covariance_min_periods > self.portfolio_covariance_lookback:
            raise ValueError("covariance_min_periods must not exceed portfolio_covariance_lookback")
        if self.max_daily_turnover is not None:
            object.__setattr__(
                self,
                "max_daily_turnover",
                _long_only_turnover_limit(self.max_daily_turnover, "max_daily_turnover"),
            )
        object.__setattr__(
            self,
            "portfolio_volatility_limit",
            _positive_float(self.portfolio_volatility_limit, "portfolio_volatility_limit"),
        )
        object.__setattr__(
            self, "max_drawdown_stop", _fraction(self.max_drawdown_stop, "max_drawdown_stop")
        )
        object.__setattr__(
            self,
            "max_position_risk_contribution",
            _fraction(self.max_position_risk_contribution, "max_position_risk_contribution"),
        )
        object.__setattr__(
            self,
            "volatility_adjustment_strength",
            _unit_interval(
                self.volatility_adjustment_strength,
                "volatility_adjustment_strength",
            ),
        )
        object.__setattr__(
            self,
            "concentration_adjustment_strength",
            _unit_interval(
                self.concentration_adjustment_strength,
                "concentration_adjustment_strength",
            ),
        )
        object.__setattr__(
            self,
            "risk_contribution_adjustment_strength",
            _unit_interval(
                self.risk_contribution_adjustment_strength,
                "risk_contribution_adjustment_strength",
            ),
        )


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.DataFrame
    weights: pd.DataFrame
    signals: pd.DataFrame
    metrics: PerformanceMetrics


def run_long_only_backtest(
    frame: pd.DataFrame,
    config: BacktestConfig | None = None,
    *,
    benchmark_returns: pd.DataFrame | pd.Series | None = None,
    validation_gate: object | None = None,
    require_validation_gate: bool = False,
) -> BacktestResult:
    config = config or BacktestConfig()
    _validate_realized_return_column(config.realized_return_column)
    frame = _ensure_signal_input_columns(frame, config.realized_return_column)
    frame = _normalize_backtest_input_frame(frame)
    _validate_backtest_feature_cutoffs(frame)
    _validate_backtest_return_timing(frame, config.realized_return_column)
    signal_config = _signal_config_from_backtest_config(config)
    signal_engine = DeterministicSignalEngine(signal_config)
    signal_inputs = _signal_generation_frame(frame)
    if validation_gate is None and not require_validation_gate:
        generated_signals = signal_engine.generate(signal_inputs)
    else:
        generated_signals = signal_engine.generate(
            signal_inputs,
            validation_gate=validation_gate,
            require_validation_gate=require_validation_gate,
        )
    signals = frame.copy()
    signals["signal_score"] = generated_signals["signal_score"].to_numpy()
    signals["action"] = generated_signals["action"].to_numpy()
    for column in _signal_engine_metadata_columns():
        if column in generated_signals:
            signals[column] = generated_signals[column].to_numpy()
    benchmark_by_date = _benchmark_returns_by_date(benchmark_returns)

    dates = sorted(signals["date"].dropna().unique())
    previous_weights: dict[str, float] = {}
    equity = 1.0
    peak_equity = 1.0
    benchmark_equity = 1.0
    cost_adjusted_benchmark_equity = 1.0
    benchmark_position_open = False
    cumulative_turnover = 0.0
    risk_stop_active = False
    rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []

    for idx, current_date in enumerate(dates):
        day = signals[signals["date"] == current_date].copy()
        return_horizon = _return_horizon(config.realized_return_column)
        holding_start_idx = idx + 1
        effective_idx = idx + return_horizon
        holding_start_date = _future_execution_date(dates, holding_start_idx, current_date, 1)
        effective_date = _future_execution_date(dates, effective_idx, current_date, return_horizon)

        current_drawdown = equity / peak_equity - 1
        if current_drawdown <= -abs(config.max_drawdown_stop):
            risk_stop_active = True

        if risk_stop_active:
            target_weights: dict[str, float] = {}
        else:
            target_weights = _select_stateful_targets(day, previous_weights, config)
            return_history = _recent_return_history(signals, current_date, config)
            target_weights = _apply_concentration_limits(
                day, target_weights, return_history, config
            )
            target_weights = _apply_risk_contribution_limit(
                day,
                target_weights,
                config,
                return_history=return_history,
            )
            target_weights = _apply_portfolio_volatility_limit(
                day,
                target_weights,
                config,
                return_history=return_history,
            )
            target_weights = _apply_turnover_limit(previous_weights, target_weights, config)

        turnover = calculate_portfolio_turnover(previous_weights, target_weights)
        cumulative_turnover += turnover
        gross_return = 0.0
        position_rows: list[dict[str, object]] = []
        for ticker, weight in target_weights.items():
            realized = _future_realized_returns(
                day,
                ticker,
                config.realized_return_column,
                signal_date=current_date,
                holding_start_date=holding_start_date,
                return_date=effective_date,
            )
            realized_return = 0.0
            if not realized.empty and pd.notna(realized.iloc[0]):
                realized_return = float(realized.iloc[0])
                gross_return += weight * realized_return
            position_rows.append(
                {
                    "date": holding_start_date,
                    "signal_date": current_date,
                    "holding_start_date": holding_start_date,
                    "effective_date": effective_date,
                    "ticker": ticker,
                    "weight": weight,
                    "previous_weight": float(previous_weights.get(ticker, 0.0)),
                    "realized_return": realized_return,
                    "gross_return_contribution": float(weight * realized_return),
                    "position_turnover": abs(
                        float(weight) - float(previous_weights.get(ticker, 0.0))
                    ),
                }
            )

        cost_adjustment = calculate_cost_adjusted_returns(
            gross_return,
            turnover,
            cost_bps=config.cost_bps,
            slippage_bps=config.slippage_bps,
        ).iloc[0]
        sizing_validation = _validate_post_cost_position_sizing(
            day,
            target_weights,
            config,
            return_history=_recent_return_history(signals, current_date, config),
            cost_adjustment=cost_adjustment,
        )
        weight_rows.extend(
            _allocate_position_costs(
                position_rows,
                transaction_cost_return=float(cost_adjustment["transaction_cost_return"]),
                slippage_cost_return=float(cost_adjustment["slippage_cost_return"]),
            )
        )
        net_return = float(cost_adjustment["cost_adjusted_return"])
        benchmark_return, benchmark_return_available = _benchmark_return_observation(
            day,
            config.benchmark_ticker,
            benchmark_by_date,
            current_date,
            config.realized_return_column,
            holding_start_date=holding_start_date,
            return_date=effective_date,
        )
        benchmark_turnover = 0.0
        if benchmark_return_available and not benchmark_position_open:
            benchmark_turnover = 1.0
            benchmark_position_open = True
        benchmark_cost_adjustment = calculate_cost_adjusted_returns(
            benchmark_return,
            benchmark_turnover,
            cost_bps=config.cost_bps,
            slippage_bps=config.slippage_bps,
        ).iloc[0]
        cost_adjusted_benchmark_return = float(benchmark_cost_adjustment["cost_adjusted_return"])
        equity *= 1 + net_return
        peak_equity = max(peak_equity, equity)
        benchmark_equity *= 1 + benchmark_return
        cost_adjusted_benchmark_equity *= 1 + cost_adjusted_benchmark_return
        rows.append(
            {
                "date": current_date,
                "holding_start_date": holding_start_date,
                "return_date": effective_date,
                "portfolio_return": net_return,
                "cost_adjusted_return": net_return,
                "net_return": net_return,
                "deterministic_strategy_return": gross_return,
                "gross_return": gross_return,
                "transaction_cost_return": float(cost_adjustment["transaction_cost_return"]),
                "slippage_cost_return": float(cost_adjustment["slippage_cost_return"]),
                "total_cost_return": float(cost_adjustment["total_cost_return"]),
                "turnover_cost_return": float(cost_adjustment["turnover_cost_return"]),
                "cost_bps": config.cost_bps,
                "slippage_bps": config.slippage_bps,
                "average_daily_turnover_budget": config.average_daily_turnover_budget,
                "realized_return_column": config.realized_return_column,
                "benchmark_return": benchmark_return,
                "benchmark_gross_return": benchmark_return,
                "cost_adjusted_benchmark_return": cost_adjusted_benchmark_return,
                "benchmark_transaction_cost_return": float(
                    benchmark_cost_adjustment["transaction_cost_return"]
                ),
                "benchmark_slippage_cost_return": float(
                    benchmark_cost_adjustment["slippage_cost_return"]
                ),
                "benchmark_total_cost_return": float(
                    benchmark_cost_adjustment["total_cost_return"]
                ),
                "benchmark_turnover": benchmark_turnover,
                "equity": equity,
                "benchmark_equity": benchmark_equity,
                "cost_adjusted_benchmark_equity": cost_adjusted_benchmark_equity,
                "period_turnover": turnover,
                "turnover": turnover,
                "cumulative_turnover": cumulative_turnover,
                "exposure": sum(target_weights.values()),
                "portfolio_volatility_estimate": _portfolio_volatility_estimate(
                    day,
                    target_weights,
                    return_history=_recent_return_history(signals, current_date, config),
                    min_periods=config.covariance_min_periods,
                ),
                "position_sizing_validation_status": sizing_validation["status"],
                "position_sizing_validation_rule": sizing_validation["rule"],
                "position_sizing_validation_reason": sizing_validation["reason"],
                "position_count": sizing_validation["position_count"],
                "max_position_weight": sizing_validation["max_position_weight"],
                "max_sector_exposure": sizing_validation["max_sector_exposure"],
                "gross_exposure": sizing_validation["gross_exposure"],
                "net_exposure": sizing_validation["net_exposure"],
                "max_position_risk_contribution": sizing_validation[
                    "max_position_risk_contribution"
                ],
                "leverage_limit": sizing_validation["leverage_limit"],
                "post_cost_validation_total_cost_return": sizing_validation[
                    "total_cost_return"
                ],
                "risk_stop_active": risk_stop_active,
            }
        )
        previous_weights = target_weights

    equity_curve = pd.DataFrame(rows, columns=_equity_curve_columns())
    weights = pd.DataFrame(
        weight_rows,
        columns=_weight_columns(),
    )
    metrics = calculate_metrics(equity_curve)
    return BacktestResult(
        equity_curve=equity_curve, weights=weights, signals=signals, metrics=metrics
    )


def _ensure_signal_input_columns(frame: pd.DataFrame, realized_return_column: str) -> pd.DataFrame:
    output = frame.copy()
    defaults: dict[str, object] = {
        "date": pd.NaT,
        "ticker": "",
        "expected_return": 0.0,
        "predicted_volatility": 0.0,
        "downside_quantile": 0.0,
        "model_confidence": 0.0,
        "text_risk_score": 0.0,
        "sec_risk_flag": 0.0,
        "sec_risk_flag_20d": 0.0,
        "news_negative_ratio": 0.0,
        "liquidity_score": 0.0,
        realized_return_column: np.nan,
    }
    for column, default in defaults.items():
        if column not in output:
            output[column] = pd.Series(default, index=output.index)
    return output


def _signal_config_from_backtest_config(config: BacktestConfig) -> SignalEngineConfig:
    return SignalEngineConfig(
        cost_bps=config.cost_bps,
        slippage_bps=config.slippage_bps,
        portfolio_volatility_limit=config.portfolio_volatility_limit,
        average_daily_turnover_budget=config.average_daily_turnover_budget,
        max_symbol_weight=config.max_symbol_weight,
        max_sector_weight=config.max_sector_weight,
        covariance_aware_risk_enabled=config.covariance_aware_risk_enabled,
        portfolio_covariance_lookback=config.portfolio_covariance_lookback,
        covariance_return_column=config.covariance_return_column,
        covariance_min_periods=config.covariance_min_periods,
        max_drawdown_floor=-abs(config.max_drawdown_stop),
    )


def _signal_engine_metadata_columns() -> tuple[str, ...]:
    return (
        "signal_generation_gate_decision",
        "signal_generation_gate_status",
        "risk_metric_penalty",
        "covariance_aware_risk_enabled",
        "portfolio_covariance_lookback",
        "covariance_return_column",
        "covariance_min_periods",
        "portfolio_volatility_limit",
        "average_daily_turnover_budget",
        "configured_max_symbol_weight",
        "configured_max_sector_weight",
    )


def _normalize_backtest_input_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if output.empty:
        return output
    output["date"] = pd.to_datetime(output["date"], errors="coerce").dt.normalize()
    output["ticker"] = output["ticker"].fillna("").astype(str).str.strip()
    invalid = output["date"].isna() | output["ticker"].eq("")
    if invalid.any():
        raise ValueError(
            "backtest input rows must have non-null date and ticker before signal generation"
        )
    sort_columns = ["date", "ticker"]
    if "strategy_id" in output:
        sort_columns.append("strategy_id")
    return output.sort_values(sort_columns).reset_index(drop=True)


def _validate_backtest_feature_cutoffs(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    sample_timestamp = date_end_utc(frame["date"])
    availability_columns = _availability_cutoff_columns(frame)
    for column in availability_columns:
        availability = timestamp_utc(frame[column])
        violation = availability.notna() & (availability > sample_timestamp)
        if violation.any():
            first_index = int(np.flatnonzero(violation.to_numpy())[0])
            raise ValueError(
                f"backtest input column {column} contains data unavailable at signal date "
                f"{frame.loc[first_index, 'date']}"
            )

    prediction_columns = _prediction_cutoff_columns(frame)
    for column in prediction_columns:
        prediction_time = timestamp_utc(frame[column])
        violation = prediction_time.notna() & (prediction_time > sample_timestamp)
        if violation.any():
            first_index = int(np.flatnonzero(violation.to_numpy())[0])
            raise ValueError(
                f"backtest prediction column {column} is later than signal date "
                f"{frame.loc[first_index, 'date']}"
            )
    validate_event_availability_order(frame, label="backtest input")


def _validate_realized_return_column(realized_return_column: str) -> None:
    forward_return_horizon(realized_return_column)


def _validate_backtest_return_timing(frame: pd.DataFrame, realized_return_column: str) -> None:
    if frame.empty:
        return
    horizon = forward_return_horizon(realized_return_column)
    if "realized_return_column" in frame:
        realized = frame["realized_return_column"].dropna().astype(str)
        mismatched = realized.ne(realized_return_column)
        if mismatched.any():
            raise ValueError(
                "backtest input realized_return_column must match configured "
                f"{realized_return_column}"
            )
    if "return_horizon" in frame:
        observed_horizon = pd.to_numeric(frame["return_horizon"], errors="coerce")
        mismatched_horizon = observed_horizon.notna() & observed_horizon.ne(horizon)
        if mismatched_horizon.any():
            raise ValueError(
                f"backtest input return_horizon must match configured {realized_return_column}"
            )

    signal_dates = pd.to_datetime(
        frame.get("signal_date", frame["date"]), errors="coerce"
    ).dt.normalize()
    holding_start = _optional_timing_dates(frame, "holding_start_date")
    if holding_start is not None:
        invalid_start = (
            holding_start.notna() & signal_dates.notna() & (holding_start <= signal_dates)
        )
        if invalid_start.any():
            raise ValueError("backtest holding_start_date must be after signal_date")

    holding_end = _optional_timing_dates(frame, "return_label_date")
    if holding_end is None:
        holding_end = _optional_timing_dates(frame, "holding_end_date")
    if holding_end is None:
        holding_end = _optional_timing_dates(frame, "return_date")
    if holding_start is not None and holding_end is not None:
        invalid_end = holding_end.notna() & holding_start.notna() & (holding_end < holding_start)
        if invalid_end.any():
            raise ValueError("backtest return label date must be on or after holding_start_date")
    if holding_end is not None:
        invalid_label_date = (
            holding_end.notna() & signal_dates.notna() & (holding_end <= signal_dates)
        )
        if invalid_label_date.any():
            raise ValueError("backtest return label date must be after signal_date")


def _optional_timing_dates(frame: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in frame:
        return None
    return pd.to_datetime(frame[column], errors="coerce").dt.normalize()


def _availability_cutoff_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column == "availability_timestamp" or str(column).endswith("_availability_timestamp")
    ]


def _prediction_cutoff_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in ("prediction_date", "prediction_timestamp", "model_prediction_timestamp")
        if column in frame
    ]


def _signal_generation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    label_columns = [
        column
        for column in frame.columns
        if str(column).startswith("forward_return_") or str(column).startswith("return_")
    ]
    return frame.drop(
        columns=[*label_columns, *_report_only_research_metric_columns()],
        errors="ignore",
    )


def _report_only_research_metric_columns() -> tuple[str, ...]:
    return ("top_decile_20d_excess_return",)


def _equity_curve_columns() -> list[str]:
    return [
        "date",
        "holding_start_date",
        "return_date",
        "portfolio_return",
        "cost_adjusted_return",
        "net_return",
        "deterministic_strategy_return",
        "gross_return",
        "transaction_cost_return",
        "slippage_cost_return",
        "total_cost_return",
        "turnover_cost_return",
        "cost_bps",
        "slippage_bps",
        "average_daily_turnover_budget",
        "realized_return_column",
        "benchmark_return",
        "benchmark_gross_return",
        "cost_adjusted_benchmark_return",
        "benchmark_transaction_cost_return",
        "benchmark_slippage_cost_return",
        "benchmark_total_cost_return",
        "benchmark_turnover",
        "equity",
        "benchmark_equity",
        "cost_adjusted_benchmark_equity",
        "period_turnover",
        "turnover",
        "cumulative_turnover",
        "exposure",
        "portfolio_volatility_estimate",
        "position_sizing_validation_status",
        "position_sizing_validation_rule",
        "position_sizing_validation_reason",
        "position_count",
        "max_position_weight",
        "max_sector_exposure",
        "gross_exposure",
        "net_exposure",
        "max_position_risk_contribution",
        "leverage_limit",
        "post_cost_validation_total_cost_return",
        "risk_stop_active",
    ]


def _weight_columns() -> list[str]:
    return [
        "date",
        "signal_date",
        "holding_start_date",
        "effective_date",
        "ticker",
        "weight",
        "previous_weight",
        "realized_return",
        "gross_return_contribution",
        "position_turnover",
        "transaction_cost_return",
        "slippage_cost_return",
        "total_cost_return",
        "net_return_contribution",
        "position_net_return",
    ]


def _allocate_position_costs(
    position_rows: list[dict[str, object]],
    *,
    transaction_cost_return: float,
    slippage_cost_return: float,
) -> list[dict[str, object]]:
    if not position_rows:
        return []
    allocated_rows: list[dict[str, object]] = []
    row_turnover = sum(float(row.get("position_turnover", 0.0) or 0.0) for row in position_rows)
    allocation_denominator = row_turnover if row_turnover > 0 else 0.0
    equal_share = 1.0 / len(position_rows)
    for row in position_rows:
        position_turnover = float(row.get("position_turnover", 0.0) or 0.0)
        if allocation_denominator > 0:
            cost_share = position_turnover / allocation_denominator
        else:
            cost_share = equal_share
        transaction_cost = transaction_cost_return * cost_share
        slippage_cost = slippage_cost_return * cost_share
        total_cost = transaction_cost + slippage_cost
        gross_contribution = float(row.get("gross_return_contribution", 0.0) or 0.0)
        net_contribution = gross_contribution - total_cost
        allocated = dict(row)
        allocated.update(
            {
                "transaction_cost_return": transaction_cost,
                "slippage_cost_return": slippage_cost,
                "total_cost_return": total_cost,
                "net_return_contribution": net_contribution,
                "position_net_return": net_contribution,
            }
        )
        allocated_rows.append(allocated)
    return allocated_rows


def _benchmark_return(
    day: pd.DataFrame,
    benchmark_ticker: str,
    benchmark_by_date: dict[pd.Timestamp, float] | None = None,
    current_date: object | None = None,
    realized_return_column: str = "forward_return_1",
    holding_start_date: object | None = None,
    return_date: object | None = None,
) -> float:
    return _benchmark_return_observation(
        day,
        benchmark_ticker,
        benchmark_by_date,
        current_date,
        realized_return_column,
        holding_start_date=holding_start_date,
        return_date=return_date,
    )[0]


def _benchmark_return_observation(
    day: pd.DataFrame,
    benchmark_ticker: str,
    benchmark_by_date: dict[pd.Timestamp, float] | None = None,
    current_date: object | None = None,
    realized_return_column: str = "forward_return_1",
    holding_start_date: object | None = None,
    return_date: object | None = None,
) -> tuple[float, bool]:
    if benchmark_by_date is not None and current_date is not None:
        value = benchmark_by_date.get(pd.Timestamp(current_date).normalize())
        if value is not None and pd.notna(value):
            return float(value), True
        return 0.0, False

    if current_date is not None and holding_start_date is not None and return_date is not None:
        benchmark = _future_realized_returns(
            day,
            benchmark_ticker,
            realized_return_column,
            signal_date=current_date,
            holding_start_date=holding_start_date,
            return_date=return_date,
        )
    else:
        benchmark = _realized_returns(day, benchmark_ticker, realized_return_column)
    if benchmark.empty or pd.isna(benchmark.iloc[0]):
        return 0.0, False
    return float(benchmark.iloc[0]), True


def _benchmark_returns_by_date(
    benchmark_returns: pd.DataFrame | pd.Series | None,
) -> dict[pd.Timestamp, float] | None:
    if benchmark_returns is None:
        return None

    if isinstance(benchmark_returns, pd.Series):
        if not isinstance(benchmark_returns.index, pd.DatetimeIndex):
            raise ValueError("benchmark return series must use a DatetimeIndex")
        frame = benchmark_returns.rename("benchmark_return").reset_index()
        frame = frame.rename(columns={frame.columns[0]: "date"})
    else:
        frame = benchmark_returns.copy()

    missing = {"date", "benchmark_return"}.difference(frame.columns)
    if missing:
        raise ValueError(f"benchmark returns missing required columns: {sorted(missing)}")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["benchmark_return"] = pd.to_numeric(frame["benchmark_return"], errors="coerce")
    frame = frame.dropna(subset=["date"]).drop_duplicates("date", keep="last")
    return {
        pd.Timestamp(row["date"]).normalize(): float(row["benchmark_return"])
        for _, row in frame.iterrows()
        if pd.notna(row["benchmark_return"])
    }


def _realized_returns(day: pd.DataFrame, ticker: str, return_column: str) -> pd.Series:
    if return_column not in day:
        return pd.Series(dtype=float)
    return day.loc[day["ticker"] == ticker, return_column]


def _future_execution_date(
    dates: list[object],
    target_index: int,
    signal_date: object,
    business_day_offset: int,
) -> pd.Timestamp:
    if target_index < len(dates):
        return pd.Timestamp(dates[target_index]).normalize()
    return pd.Timestamp(signal_date).normalize() + pd.offsets.BDay(max(int(business_day_offset), 1))


def _future_realized_returns(
    day: pd.DataFrame,
    ticker: str,
    return_column: str,
    *,
    signal_date: object,
    holding_start_date: object,
    return_date: object,
) -> pd.Series:
    if not _is_future_execution_window(signal_date, holding_start_date, return_date):
        return pd.Series(dtype=float)
    return _realized_returns(day, ticker, return_column)


def _is_future_execution_window(
    signal_date: object,
    holding_start_date: object,
    return_date: object,
) -> bool:
    signal = pd.Timestamp(signal_date).normalize()
    start = pd.Timestamp(holding_start_date).normalize() if pd.notna(holding_start_date) else pd.NaT
    end = pd.Timestamp(return_date).normalize() if pd.notna(return_date) else pd.NaT
    return pd.notna(start) and pd.notna(end) and start > signal and end >= start


def _return_horizon(return_column: str) -> int:
    return forward_return_horizon(return_column)


def _positive_int(value: object, name: str) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric) or int(numeric) != numeric or int(numeric) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def _non_negative_bps(value: object, name: str) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric) or not np.isfinite(float(numeric)) or float(numeric) < 0:
        raise ValueError(f"{name} must be non-negative")
    return float(numeric)


def _positive_float(value: object, name: str) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric) or not np.isfinite(float(numeric)) or float(numeric) <= 0:
        raise ValueError(f"{name} must be positive")
    return float(numeric)


def _fraction(value: object, name: str) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric) or not np.isfinite(float(numeric)) or not 0 < float(numeric) <= 1:
        raise ValueError(f"{name} must be greater than 0 and no more than 1")
    return float(numeric)


def _unit_interval(value: object, name: str) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric) or not np.isfinite(float(numeric)) or not 0 <= float(numeric) <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return float(numeric)


def _long_only_turnover_limit(value: object, name: str) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric) or not np.isfinite(float(numeric)) or not 0 <= float(numeric) <= 2:
        raise ValueError(f"{name} must be between 0 and 2 for a long-only portfolio")
    return float(numeric)


def _select_stateful_targets(
    day: pd.DataFrame,
    previous_weights: dict[str, float],
    config: BacktestConfig,
) -> dict[str, float]:
    sell_tickers = set(day.loc[day["action"] == "SELL", "ticker"])
    candidates = [ticker for ticker in previous_weights if ticker not in sell_tickers]

    buy = day[day["action"] == "BUY"].sort_values("signal_score", ascending=False)
    for ticker in buy["ticker"]:
        if ticker not in candidates:
            candidates.append(ticker)

    if not candidates:
        return {}

    score_by_ticker = day.set_index("ticker")["signal_score"].to_dict()
    selected = sorted(
        candidates, key=lambda ticker: score_by_ticker.get(ticker, -np.inf), reverse=True
    )[: config.top_n]
    weight = min(1.0 / len(selected), config.max_symbol_weight)
    return {ticker: weight for ticker in selected}


def _apply_portfolio_volatility_limit(
    day: pd.DataFrame,
    target_weights: dict[str, float],
    config: BacktestConfig,
    *,
    return_history: pd.DataFrame | None = None,
) -> dict[str, float]:
    estimate = _portfolio_volatility_estimate(
        day,
        target_weights,
        return_history=return_history,
        min_periods=config.covariance_min_periods,
    )
    if estimate <= config.portfolio_volatility_limit or estimate == 0:
        return target_weights
    raw_scale = config.portfolio_volatility_limit / estimate
    scale = _blend_scale(raw_scale, config.volatility_adjustment_strength)
    return {ticker: weight * scale for ticker, weight in target_weights.items()}


def _apply_concentration_limits(
    day: pd.DataFrame,
    target_weights: dict[str, float],
    return_history: pd.DataFrame | None,
    config: BacktestConfig,
) -> dict[str, float]:
    if not target_weights:
        return target_weights
    limited = dict(target_weights)
    if "sector" in day.columns:
        limited = _apply_group_weight_limit(day, limited, "sector", config.max_sector_weight)
    else:
        limited = _apply_correlation_cluster_limit(return_history, limited, config)
    adjusted = _blend_weights(
        target_weights,
        limited,
        config.concentration_adjustment_strength,
    )
    return {ticker: weight for ticker, weight in adjusted.items() if weight > 0}


def _apply_group_weight_limit(
    day: pd.DataFrame,
    target_weights: dict[str, float],
    group_column: str,
    max_group_weight: float,
) -> dict[str, float]:
    if max_group_weight <= 0:
        return {}
    group_by_ticker = day.set_index("ticker")[group_column].fillna("unknown").astype(str).to_dict()
    output = dict(target_weights)
    for group in sorted(set(group_by_ticker.get(ticker, "unknown") for ticker in output)):
        tickers = [ticker for ticker in output if group_by_ticker.get(ticker, "unknown") == group]
        exposure = sum(output[ticker] for ticker in tickers)
        if exposure <= max_group_weight or exposure <= 0:
            continue
        scale = max_group_weight / exposure
        for ticker in tickers:
            output[ticker] *= scale
    return output


def _apply_correlation_cluster_limit(
    return_history: pd.DataFrame | None,
    target_weights: dict[str, float],
    config: BacktestConfig,
) -> dict[str, float]:
    clusters = _correlation_clusters(
        return_history, tuple(target_weights), config.correlation_cluster_threshold
    )
    if not clusters:
        return target_weights
    output = dict(target_weights)
    for cluster in clusters:
        exposure = sum(output.get(ticker, 0.0) for ticker in cluster)
        if exposure <= config.max_correlation_cluster_weight or exposure <= 0:
            continue
        scale = config.max_correlation_cluster_weight / exposure
        for ticker in cluster:
            if ticker in output:
                output[ticker] *= scale
    return output


def _apply_risk_contribution_limit(
    day: pd.DataFrame,
    target_weights: dict[str, float],
    config: BacktestConfig,
    *,
    return_history: pd.DataFrame | None = None,
) -> dict[str, float]:
    if not target_weights:
        return target_weights
    contributions = _position_risk_contribution_pct(
        day,
        target_weights,
        return_history=return_history,
        min_periods=config.covariance_min_periods,
    )
    if not contributions:
        return target_weights
    limited = dict(target_weights)
    for ticker, contribution in contributions.items():
        if contribution <= config.max_position_risk_contribution or contribution <= 0:
            continue
        raw_scale = config.max_position_risk_contribution / contribution
        limited[ticker] = limited[ticker] * raw_scale
    adjusted = _blend_weights(
        target_weights,
        limited,
        config.risk_contribution_adjustment_strength,
    )
    return {ticker: weight for ticker, weight in adjusted.items() if weight > 0}


def _apply_turnover_limit(
    previous_weights: dict[str, float],
    target_weights: dict[str, float],
    config: BacktestConfig,
) -> dict[str, float]:
    if config.max_daily_turnover is None:
        return target_weights
    max_turnover = max(float(config.max_daily_turnover), 0.0)
    turnover = calculate_portfolio_turnover(previous_weights, target_weights)
    if turnover <= max_turnover or turnover == 0:
        return target_weights
    scale = max_turnover / turnover
    tickers = set(previous_weights).union(target_weights)
    limited = {
        ticker: previous_weights.get(ticker, 0.0)
        + scale * (target_weights.get(ticker, 0.0) - previous_weights.get(ticker, 0.0))
        for ticker in tickers
    }
    return {ticker: weight for ticker, weight in limited.items() if weight > 1e-12}


def _validate_post_cost_position_sizing(
    day: pd.DataFrame,
    target_weights: dict[str, float],
    config: BacktestConfig,
    *,
    return_history: pd.DataFrame | None,
    cost_adjustment: pd.Series,
) -> dict[str, object]:
    tolerance = 1e-9
    weights = {ticker: float(weight) for ticker, weight in target_weights.items()}
    failures: list[str] = []

    position_count = len(weights)
    max_position_weight = max((abs(weight) for weight in weights.values()), default=0.0)
    gross_exposure = float(sum(abs(weight) for weight in weights.values()))
    net_exposure = float(sum(weights.values()))
    leverage_limit = 1.0

    if any(weight < -tolerance for weight in weights.values()):
        failures.append("long_only")
    if max_position_weight > config.max_symbol_weight + tolerance:
        failures.append("max_symbol_weight")
    if gross_exposure > leverage_limit + tolerance:
        failures.append("gross_leverage")
    if net_exposure < -tolerance or net_exposure - gross_exposure > tolerance:
        failures.append("net_leverage")

    max_sector_exposure = 0.0
    if weights and "sector" in day.columns:
        sector_by_ticker = (
            day.set_index("ticker")["sector"].fillna("unknown").astype(str).to_dict()
        )
        sector_exposures: dict[str, float] = {}
        for ticker, weight in weights.items():
            sector = sector_by_ticker.get(ticker, "unknown")
            sector_exposures[sector] = sector_exposures.get(sector, 0.0) + abs(weight)
        max_sector_exposure = max(sector_exposures.values(), default=0.0)
        if max_sector_exposure > config.max_sector_weight + tolerance:
            failures.append("max_sector_weight")

    risk_contributions = _position_risk_contribution_pct(
        day,
        weights,
        return_history=return_history,
        min_periods=config.covariance_min_periods,
    )
    max_risk_contribution = max(risk_contributions.values(), default=0.0)
    portfolio_volatility = _portfolio_volatility_estimate(
        day,
        weights,
        return_history=return_history,
        min_periods=config.covariance_min_periods,
    )
    if portfolio_volatility > config.portfolio_volatility_limit + tolerance:
        failures.append("portfolio_volatility_limit")

    gross_return = _finite_cost_value(cost_adjustment, "gross_return")
    transaction_cost_return = _finite_cost_value(cost_adjustment, "transaction_cost_return")
    slippage_cost_return = _finite_cost_value(cost_adjustment, "slippage_cost_return")
    total_cost_return = _finite_cost_value(cost_adjustment, "total_cost_return")
    net_return = _finite_cost_value(cost_adjustment, "cost_adjusted_return")
    if transaction_cost_return < -tolerance:
        failures.append("transaction_cost_return")
    if slippage_cost_return < -tolerance:
        failures.append("slippage_cost_return")
    if abs((gross_return - total_cost_return) - net_return) > tolerance:
        failures.append("post_cost_return_accounting")

    if failures:
        reason = ",".join(sorted(set(failures)))
        raise ValueError(
            "post-cost position sizing validation failed: "
            f"{reason}; position_count={position_count}, "
            f"max_position_weight={max_position_weight:.6f}, "
            f"max_sector_exposure={max_sector_exposure:.6f}, "
            f"gross_exposure={gross_exposure:.6f}, "
            f"portfolio_volatility={portfolio_volatility:.6f}, "
            f"max_position_risk_contribution={max_risk_contribution:.6f}"
        )

    return {
        "status": "pass",
        "rule": "post_cost_position_sizing_constraints_v1",
        "reason": "risk_concentration_and_leverage_limits_passed_after_costs",
        "position_count": position_count,
        "max_position_weight": max_position_weight,
        "max_sector_exposure": max_sector_exposure,
        "gross_exposure": gross_exposure,
        "net_exposure": net_exposure,
        "portfolio_volatility": portfolio_volatility,
        "max_position_risk_contribution": max_risk_contribution,
        "leverage_limit": leverage_limit,
        "total_cost_return": total_cost_return,
    }


def _finite_cost_value(cost_adjustment: pd.Series, column: str) -> float:
    value = pd.to_numeric(cost_adjustment.get(column, 0.0), errors="coerce")
    if pd.isna(value) or not np.isfinite(float(value)):
        raise ValueError(f"post-cost position sizing validation failed: {column} must be finite")
    return float(value)


def _recent_return_history(
    signals: pd.DataFrame,
    current_date: object,
    config: BacktestConfig,
) -> pd.DataFrame | None:
    if not config.covariance_aware_risk_enabled:
        return None
    return_column = config.covariance_return_column
    if return_column not in signals or "date" not in signals or "ticker" not in signals:
        return None
    dates = pd.to_datetime(signals["date"], errors="coerce")
    current = pd.Timestamp(current_date).normalize()
    lookback = max(int(config.portfolio_covariance_lookback), 2)
    history = signals.loc[dates <= current, ["date", "ticker", return_column]].copy()
    if history.empty:
        return None
    history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.normalize()
    history["return_1"] = pd.to_numeric(history[return_column], errors="coerce")
    history = history.dropna(subset=["date", "ticker", "return_1"]).sort_values("date")
    if history.empty:
        return None
    keep_dates = sorted(history["date"].dropna().unique())[-lookback:]
    return history[history["date"].isin(keep_dates)].reset_index(drop=True)


def _portfolio_volatility_estimate(
    day: pd.DataFrame,
    weights: dict[str, float],
    *,
    return_history: pd.DataFrame | None = None,
    min_periods: int = 2,
) -> float:
    if not weights:
        return 0.0
    covariance_estimate = _covariance_portfolio_volatility(
        return_history,
        weights,
        min_periods=min_periods,
    )
    if covariance_estimate is not None:
        return covariance_estimate
    volatility_by_ticker = day.set_index("ticker")["predicted_volatility"].fillna(0.0).to_dict()
    return float(
        np.sqrt(
            sum(
                (weight * float(volatility_by_ticker.get(ticker, 0.0))) ** 2
                for ticker, weight in weights.items()
            )
        )
    )


def _position_risk_contribution_pct(
    day: pd.DataFrame,
    weights: dict[str, float],
    *,
    return_history: pd.DataFrame | None = None,
    min_periods: int = 2,
) -> dict[str, float]:
    tickers = [ticker for ticker in weights if ticker]
    if not tickers:
        return {}
    covariance = _risk_contribution_covariance(
        day,
        return_history,
        tickers,
        min_periods=min_periods,
    )
    weight_vector = np.array([weights[ticker] for ticker in covariance.columns], dtype=float)
    covariance_values = covariance.to_numpy(dtype=float)
    variance = float(weight_vector @ covariance_values @ weight_vector.T)
    if not np.isfinite(variance) or variance <= 0:
        return {}
    marginal = covariance_values @ weight_vector
    contribution = weight_vector * marginal
    contribution_pct = contribution / variance
    return {
        ticker: max(float(value), 0.0)
        for ticker, value in zip(covariance.columns, contribution_pct, strict=False)
    }


def _risk_contribution_covariance(
    day: pd.DataFrame,
    return_history: pd.DataFrame | None,
    tickers: list[str],
    *,
    min_periods: int = 2,
) -> pd.DataFrame:
    if return_history is not None and not return_history.empty and len(tickers) >= 2:
        try:
            covariance = estimate_covariance_matrix(
                return_history,
                tickers=tickers,
                min_periods=max(int(min_periods), 2),
            )
        except ValueError:
            covariance = pd.DataFrame()
        if not covariance.empty:
            return covariance
    volatility_by_ticker = day.set_index("ticker")["predicted_volatility"].fillna(0.0).to_dict()
    volatility = np.array(
        [max(float(volatility_by_ticker.get(ticker, 0.0)), 0.0) for ticker in tickers],
        dtype=float,
    )
    return pd.DataFrame(np.diag(np.square(volatility)), index=tickers, columns=tickers)


def _blend_scale(raw_scale: float, strength: float) -> float:
    bounded_scale = min(max(float(raw_scale), 0.0), 1.0)
    bounded_strength = min(max(float(strength), 0.0), 1.0)
    return 1.0 - bounded_strength * (1.0 - bounded_scale)


def _blend_weights(
    original: dict[str, float],
    limited: dict[str, float],
    strength: float,
) -> dict[str, float]:
    bounded_strength = min(max(float(strength), 0.0), 1.0)
    tickers = set(original).union(limited)
    return {
        ticker: original.get(ticker, 0.0)
        + bounded_strength * (limited.get(ticker, 0.0) - original.get(ticker, 0.0))
        for ticker in tickers
    }


def _covariance_portfolio_volatility(
    return_history: pd.DataFrame | None,
    weights: dict[str, float],
    *,
    min_periods: int = 2,
) -> float | None:
    if return_history is None or return_history.empty or len(weights) < 2:
        return None
    if not {"date", "ticker", "return_1"}.issubset(return_history.columns):
        return None
    selected = [ticker for ticker in weights if ticker]
    if len(selected) < 2:
        return None
    try:
        covariance = estimate_covariance_matrix(
            return_history,
            tickers=selected,
            min_periods=max(int(min_periods), 2),
        )
    except ValueError:
        return None
    if covariance.empty:
        return None
    weight_vector = np.array([weights[ticker] for ticker in covariance.columns], dtype=float)
    variance = float(weight_vector @ covariance.to_numpy(dtype=float) @ weight_vector.T)
    if not np.isfinite(variance) or variance < 0:
        return None
    return float(np.sqrt(variance))


def _correlation_clusters(
    return_history: pd.DataFrame | None,
    tickers: tuple[str, ...],
    threshold: float,
) -> list[set[str]]:
    if return_history is None or return_history.empty or len(tickers) < 2:
        return []
    if not {"date", "ticker", "return_1"}.issubset(return_history.columns):
        return []
    selected = [ticker for ticker in tickers if ticker]
    if len(selected) < 2:
        return []
    try:
        corr = estimate_correlation_matrix(return_history, tickers=selected, min_periods=2)
    except ValueError:
        return []
    parent = {ticker: ticker for ticker in corr.columns}

    def find(ticker: str) -> str:
        while parent[ticker] != ticker:
            parent[ticker] = parent[parent[ticker]]
            ticker = parent[ticker]
        return ticker

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in corr.columns:
        for right in corr.columns:
            if left >= right:
                continue
            if float(corr.loc[left, right]) >= threshold:
                union(left, right)

    grouped: dict[str, set[str]] = {}
    for ticker in corr.columns:
        grouped.setdefault(find(ticker), set()).add(ticker)
    return [cluster for cluster in grouped.values() if len(cluster) > 1]
