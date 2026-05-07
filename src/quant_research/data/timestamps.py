from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

PRICE_TIMEZONE = "America/New_York"
UTC_TIMEZONE = "UTC"
FEATURE_AVAILABILITY_SCHEMA_VERSION = "feature-availability-v1"
STANDARD_TIMESTAMP_COLUMNS = (
    "event_timestamp",
    "availability_timestamp",
    "source_timestamp",
    "timezone",
)
STANDARD_FEATURE_AVAILABILITY_COLUMNS = (
    "as_of_timestamp",
    "publication_timestamp",
    "availability_timestamp",
    "timezone",
)


@dataclass(frozen=True)
class FeatureAvailabilitySchema:
    """Column contract used to validate point-in-time feature inputs."""

    source_family: str
    as_of_column: str = "as_of_timestamp"
    publication_column: str = "publication_timestamp"
    availability_column: str = "availability_timestamp"
    timezone_column: str = "timezone"
    sample_timestamp_column: str = "sample_timestamp"
    schema_version: str = FEATURE_AVAILABILITY_SCHEMA_VERSION
    cutoff_rule: str = "availability_timestamp <= sample_timestamp"
    null_policy: str = "availability_timestamp and as_of_timestamp must be non-null"

    def manifest(self, feature_names: list[str] | tuple[str, ...] = ()) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_family": self.source_family,
            "feature_names": list(feature_names),
            "as_of_column": self.as_of_column,
            "publication_column": self.publication_column,
            "availability_column": self.availability_column,
            "timezone_column": self.timezone_column,
            "sample_timestamp_column": self.sample_timestamp_column,
            "cutoff_rule": self.cutoff_rule,
            "null_policy": self.null_policy,
        }


@dataclass(frozen=True)
class FeatureAvailabilityIssue:
    row_index: object
    code: str
    column: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "row_index": self.row_index,
            "code": self.code,
            "column": self.column,
            "message": self.message,
        }


@dataclass(frozen=True)
class FeatureAvailabilityValidationResult:
    schema: FeatureAvailabilitySchema
    row_count: int
    issues: tuple[FeatureAvailabilityIssue, ...] = field(default_factory=tuple)
    null_counts: dict[str, int] = field(default_factory=dict)
    cutoff_violation_count: int = 0

    @property
    def passed(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema.manifest(),
            "passed": self.passed,
            "row_count": self.row_count,
            "null_counts": dict(self.null_counts),
            "cutoff_violation_count": self.cutoff_violation_count,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def add_price_timestamps(frame: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Add canonical timestamp columns for daily OHLCV rows."""
    normalized = frame.copy()
    event_timestamp = daily_market_close_utc(normalized[date_column])
    normalized["event_timestamp"] = event_timestamp
    normalized["availability_timestamp"] = event_timestamp
    normalized["source_timestamp"] = pd.NaT
    normalized["timezone"] = PRICE_TIMEZONE
    return normalized


def standardize_feature_availability_metadata(
    frame: pd.DataFrame,
    *,
    as_of_column: str | None = None,
    publication_column: str | None = None,
    availability_column: str | None = None,
    timezone_column: str | None = None,
    default_timezone: str = UTC_TIMEZONE,
) -> pd.DataFrame:
    """Add canonical as-of/publication/availability columns to feature inputs.

    The original provider-specific columns are preserved. Missing publication
    timestamps are allowed, but ``availability_timestamp`` and
    ``as_of_timestamp`` must be present or derivable.
    """
    normalized = frame.copy()
    timezone_values = _timezone_values(normalized, timezone_column, default_timezone)

    as_of_source = _first_existing_series(
        normalized,
        [as_of_column, "as_of_timestamp", "as_of", "event_timestamp", "date"],
    )
    publication_source = _first_existing_series(
        normalized,
        [publication_column, "publication_timestamp", "published_at", "source_timestamp"],
    )
    availability_source = _first_existing_series(
        normalized,
        [
            availability_column,
            "availability_timestamp",
            "available_at",
            "available_timestamp",
            publication_column,
            "publication_timestamp",
            "published_at",
            "source_timestamp",
            as_of_column,
            "as_of_timestamp",
            "as_of",
            "event_timestamp",
            "date",
        ],
    )

    normalized["as_of_timestamp"] = _timestamp_utc_per_row_timezone(as_of_source, timezone_values)
    normalized["publication_timestamp"] = _timestamp_utc_per_row_timezone(
        publication_source,
        timezone_values,
        allow_all_null=True,
    )
    normalized["availability_timestamp"] = _timestamp_utc_per_row_timezone(
        availability_source,
        timezone_values,
    )
    normalized["timezone"] = timezone_values.fillna(default_timezone).astype(str)
    return normalized


def validate_feature_availability(
    frame: pd.DataFrame,
    schema: FeatureAvailabilitySchema,
    *,
    feature_names: list[str] | tuple[str, ...] = (),
    sample_timestamp_mode: Literal["literal", "date_end"] = "literal",
    max_issues: int = 50,
) -> FeatureAvailabilityValidationResult:
    """Validate that feature rows have point-in-time metadata and pass cutoff."""
    issues: list[FeatureAvailabilityIssue] = []
    required_columns = [
        schema.as_of_column,
        schema.availability_column,
        schema.timezone_column,
        schema.sample_timestamp_column,
    ]
    if feature_names:
        required_columns.extend(feature_names)
    missing = [column for column in required_columns if column not in frame.columns]
    for column in missing:
        issues.append(
            FeatureAvailabilityIssue(
                row_index=None,
                code="missing_column",
                column=column,
                message=f"Missing required feature availability column: {column}",
            )
        )
    if missing:
        return FeatureAvailabilityValidationResult(
            schema=schema,
            row_count=len(frame),
            issues=tuple(issues),
            null_counts={},
            cutoff_violation_count=0,
        )

    null_counts = {
        column: int(frame[column].isna().sum())
        for column in [schema.as_of_column, schema.availability_column, schema.timezone_column]
    }
    for column, count in null_counts.items():
        if count:
            for index in frame.index[frame[column].isna()].tolist()[:max_issues]:
                issues.append(
                    FeatureAvailabilityIssue(
                        row_index=index,
                        code="null_required_metadata",
                        column=column,
                        message=f"{column} must be non-null for feature availability validation",
                    )
                )

    availability = timestamp_utc(frame[schema.availability_column], UTC_TIMEZONE)
    as_of = timestamp_utc(frame[schema.as_of_column], UTC_TIMEZONE)
    sample_timestamp = _sample_timestamp(frame[schema.sample_timestamp_column], sample_timestamp_mode)

    issues.extend(
        _timestamp_dtype_issues(frame, schema.as_of_column, "as_of_not_utc", max_issues=max_issues)
    )
    issues.extend(
        _timestamp_dtype_issues(
            frame,
            schema.availability_column,
            "availability_not_utc",
            max_issues=max_issues,
        )
    )

    cutoff_mask = availability.notna() & sample_timestamp.notna() & (availability > sample_timestamp)
    cutoff_violation_count = int(cutoff_mask.sum())
    for index in frame.index[cutoff_mask].tolist()[:max_issues]:
        issues.append(
            FeatureAvailabilityIssue(
                row_index=index,
                code="cutoff_violation",
                column=schema.availability_column,
                message=(
                    f"{schema.availability_column} is later than "
                    f"{schema.sample_timestamp_column}"
                ),
            )
        )

    impossible_order_mask = availability.notna() & as_of.notna() & (availability < as_of)
    for index in frame.index[impossible_order_mask].tolist()[:max_issues]:
        issues.append(
            FeatureAvailabilityIssue(
                row_index=index,
                code="availability_before_as_of",
                column=schema.availability_column,
                message=f"{schema.availability_column} is earlier than {schema.as_of_column}",
            )
        )

    return FeatureAvailabilityValidationResult(
        schema=schema,
        row_count=len(frame),
        issues=tuple(issues[:max_issues]),
        null_counts=null_counts,
        cutoff_violation_count=cutoff_violation_count,
    )


def filter_available_as_of(
    frame: pd.DataFrame,
    sample_timestamps: object,
    *,
    availability_column: str = "availability_timestamp",
    sample_timestamp_mode: Literal["literal", "date_end"] = "literal",
) -> pd.DataFrame:
    """Keep provider rows whose availability is known by the sample cutoff.

    Provider outputs can contain late-published or late-corrected rows. Feature
    builders should call this before deriving features from rows keyed to a
    sample timestamp so data with ``availability_timestamp > t`` never enters a
    feature row for ``t``.
    """
    if frame.empty:
        return frame.copy()
    if availability_column not in frame.columns:
        raise ValueError(f"Missing availability cutoff column: {availability_column}")

    availability = timestamp_utc(frame[availability_column], UTC_TIMEZONE)
    sample_timestamp = _sample_timestamp(
        _align_sample_timestamps(sample_timestamps, frame.index),
        sample_timestamp_mode,
    )
    keep = availability.notna() & sample_timestamp.notna() & (availability <= sample_timestamp)
    return frame.loc[keep].copy()


def validate_generated_feature_cutoffs(
    frame: pd.DataFrame,
    *,
    date_column: str = "date",
    label: str = "feature frame",
) -> None:
    """Fail fast when generated features contain values unavailable at sample date."""
    if frame.empty:
        return
    if date_column not in frame.columns:
        raise ValueError(f"{label} must include {date_column!r} for feature cutoff validation")

    sample_timestamp = date_end_utc(frame[date_column], UTC_TIMEZONE)
    for column in _availability_cutoff_columns(frame):
        availability = timestamp_utc(frame[column], UTC_TIMEZONE)
        violation = availability.notna() & sample_timestamp.notna() & (availability > sample_timestamp)
        if violation.any():
            first_index = frame.index[violation][0]
            sample_date = frame.loc[first_index, date_column]
            raise ValueError(
                f"{label} column {column} contains data unavailable at feature date {sample_date}"
            )
    validate_event_availability_order(frame, label=label)


def validate_event_availability_order(
    frame: pd.DataFrame,
    *,
    label: str = "feature frame",
) -> None:
    """Fail fast when event timestamps appear after their public availability."""
    if frame.empty:
        return
    for event_column, availability_column in _event_availability_column_pairs(frame):
        event_timestamp = timestamp_utc(frame[event_column], UTC_TIMEZONE)
        availability = timestamp_utc(frame[availability_column], UTC_TIMEZONE)
        violation = (
            event_timestamp.notna()
            & availability.notna()
            & (event_timestamp > availability)
        )
        if violation.any():
            first_index = frame.index[violation][0]
            raise ValueError(
                f"{label} column {event_column} contains an event timestamp after "
                f"{availability_column} at row {first_index}"
            )


def daily_market_close_utc(values: object) -> pd.Series:
    dates = _as_datetime_series(values).dt.tz_localize(None).dt.normalize()
    market_close = dates + pd.Timedelta(hours=16)
    return market_close.dt.tz_localize(PRICE_TIMEZONE, ambiguous=False, nonexistent="shift_forward").dt.tz_convert(
        UTC_TIMEZONE
    )


def date_end_utc(values: object, timezone: str = UTC_TIMEZONE) -> pd.Series:
    dates = _as_datetime_series(values).dt.tz_localize(None).dt.normalize()
    end_of_day = dates + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return end_of_day.dt.tz_localize(timezone, ambiguous=False, nonexistent="shift_forward").dt.tz_convert(UTC_TIMEZONE)


def timestamp_utc(values: object, timezone: str = UTC_TIMEZONE) -> pd.Series:
    if isinstance(values, pd.Series):
        raw = values
    elif isinstance(values, pd.Index):
        raw = pd.Series(values, index=values)
    else:
        raw = pd.Series(values)
    if raw.dtype == object:
        timezones = pd.Series(timezone, index=raw.index, dtype=object)
        return _timestamp_utc_per_row_timezone(raw, timezones, allow_all_null=True)

    series = pd.to_datetime(raw, errors="coerce")
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_convert(UTC_TIMEZONE)
    return series.dt.tz_localize(timezone, ambiguous=False, nonexistent="shift_forward").dt.tz_convert(UTC_TIMEZONE)


def coalesce_timestamps(*series: pd.Series) -> pd.Series:
    usable = [item for item in series if item is not None]
    if not usable:
        return pd.Series(dtype="datetime64[ns, UTC]")
    result = usable[0].copy()
    for item in usable[1:]:
        result = result.fillna(item)
    return result


def max_timestamps(*series: pd.Series) -> pd.Series:
    usable = [item for item in series if item is not None]
    if not usable:
        return pd.Series(dtype="datetime64[ns, UTC]")
    frame = pd.concat(usable, axis=1)
    return frame.max(axis=1)


def _first_existing_series(
    frame: pd.DataFrame,
    columns: list[str | None],
) -> pd.Series:
    for column in columns:
        if column and column in frame:
            return frame[column]
    return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")


def _timezone_values(
    frame: pd.DataFrame,
    timezone_column: str | None,
    default_timezone: str,
) -> pd.Series:
    candidates = [timezone_column, "timezone"]
    for column in candidates:
        if column and column in frame:
            return frame[column].fillna(default_timezone).astype(str)
    return pd.Series(default_timezone, index=frame.index, dtype=object)


def _timestamp_utc_per_row_timezone(
    values: pd.Series,
    timezones: pd.Series,
    *,
    allow_all_null: bool = False,
) -> pd.Series:
    if allow_all_null and values.isna().all():
        return pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    rows: list[pd.Timestamp] = []
    for index, value in values.items():
        if pd.isna(value):
            rows.append(pd.NaT)
            continue
        timezone = str(timezones.loc[index]) if index in timezones.index else UTC_TIMEZONE
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            rows.append(pd.NaT)
        elif isinstance(parsed, pd.Timestamp) and parsed.tzinfo is not None:
            rows.append(parsed.tz_convert(UTC_TIMEZONE))
        else:
            rows.append(pd.Timestamp(parsed).tz_localize(timezone).tz_convert(UTC_TIMEZONE))
    return pd.Series(rows, index=values.index, dtype="datetime64[ns, UTC]")


def _sample_timestamp(
    values: pd.Series,
    mode: Literal["literal", "date_end"],
) -> pd.Series:
    if mode == "date_end":
        return date_end_utc(values, UTC_TIMEZONE)
    return timestamp_utc(values, UTC_TIMEZONE)


def _align_sample_timestamps(values: object, index: pd.Index) -> pd.Series:
    if isinstance(values, pd.Series):
        if values.index.equals(index):
            return values
        if len(values) == len(index):
            return pd.Series(values.to_numpy(), index=index)
        return values.reindex(index)
    if isinstance(values, pd.Index):
        values = pd.Series(values)
    if isinstance(values, list | tuple):
        if len(values) != len(index):
            raise ValueError("sample_timestamps length must match frame length")
        return pd.Series(values, index=index)
    return pd.Series(values, index=index)


def _availability_cutoff_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column == "availability_timestamp" or str(column).endswith("_availability_timestamp")
    ]


def _event_availability_column_pairs(frame: pd.DataFrame) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for event_column in frame.columns:
        event_name = str(event_column)
        if event_name == "event_timestamp":
            availability_column = "availability_timestamp"
        elif event_name.endswith("_event_timestamp"):
            availability_column = event_name.removesuffix("_event_timestamp") + "_availability_timestamp"
        else:
            continue
        if availability_column in frame.columns:
            pairs.append((event_name, availability_column))
    return pairs


def _timestamp_dtype_issues(
    frame: pd.DataFrame,
    column: str,
    code: str,
    *,
    max_issues: int,
) -> list[FeatureAvailabilityIssue]:
    series = frame[column]
    if isinstance(series.dtype, pd.DatetimeTZDtype) and str(series.dt.tz) == UTC_TIMEZONE:
        return []
    return [
        FeatureAvailabilityIssue(
            row_index=None,
            code=code,
            column=column,
            message=f"{column} must be stored as timezone-aware UTC timestamps",
        )
    ][:max_issues]


def _as_datetime_series(values: object) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_datetime(values, errors="coerce")
    if isinstance(values, pd.Index):
        return pd.Series(pd.to_datetime(values, errors="coerce"), index=values)
    return pd.Series(pd.to_datetime(values, errors="coerce"))
