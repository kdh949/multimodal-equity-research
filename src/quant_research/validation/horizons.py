from __future__ import annotations

DEFAULT_VALIDATION_HORIZONS: tuple[int, int, int] = (1, 5, 20)
REQUIRED_VALIDATION_HORIZON_DAYS: int = 20
LONGEST_PRICE_FEATURE_LOOKBACK_DAYS: int = 60
DEFAULT_PURGE_EMBARGO_DAYS: int = max(
    REQUIRED_VALIDATION_HORIZON_DAYS,
    LONGEST_PRICE_FEATURE_LOOKBACK_DAYS,
)


def forward_return_column(horizon: int) -> str:
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    return f"forward_return_{horizon}"


def horizon_label(horizon: int) -> str:
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    return f"{horizon}d"


def default_horizon_labels() -> tuple[str, ...]:
    return tuple(horizon_label(horizon) for horizon in DEFAULT_VALIDATION_HORIZONS)


def required_validation_horizon_label() -> str:
    return horizon_label(REQUIRED_VALIDATION_HORIZON_DAYS)
