from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Literal

from quant_research.config import DEFAULT_BENCHMARK_TICKER

UNIVERSE_SNAPSHOT_SCHEMA_VERSION = "universe_snapshot.v1"
DEFAULT_UNIVERSE_INDEX_NAME = "S&P 500"
DEFAULT_UNIVERSE_NAME = "stage1_sp500_top150"
DEFAULT_UNIVERSE_SELECTION_METHOD = "sp500_market_cap_top_150_at_experiment_start"
DEFAULT_UNIVERSE_SELECTION_COUNT = 150
UNIVERSE_SELECTION_CONFIG_SCHEMA_VERSION = "universe_selection_config.v1"
UNIVERSE_DATA_VALIDATION_RULES_VERSION = "universe_data_validation_rules.v1"
DEFAULT_UNIVERSE_PROVIDER_ID = "configured_sp500_top_market_cap"
DEFAULT_UNIVERSE_CONFIG_ID = "stage1_sp500_top150"
DEFAULT_UNIVERSE_CONSTITUENT_IDENTIFIER_FIELDS = ("ticker", "cik")
DEFAULT_UNIVERSE_REQUIRED_CONSTITUENT_FIELDS = (
    "ticker",
    "exchange",
    "security_type",
    "listing_status",
    "market_cap_rank",
)
DEFAULT_UNIVERSE_METADATA_CUTOFF_FIELDS = ("available_at", "as_of_date", "effective_date")
DEFAULT_ALLOWED_UNIVERSE_EXCHANGES = ("NYSE", "NASDAQ")
DEFAULT_ALLOWED_UNIVERSE_SECURITY_TYPES = ("common_stock",)
DEFAULT_ALLOWED_UNIVERSE_LISTING_STATUSES = ("active",)
DEFAULT_EXCLUDED_UNIVERSE_SECURITY_TYPES = (
    "adr",
    "closed_end_fund",
    "etf",
    "fund",
    "preferred_stock",
    "reit_preferred",
    "unit",
    "warrant",
)
DEFAULT_EXCLUDED_UNIVERSE_TICKERS = (DEFAULT_BENCHMARK_TICKER, "QQQ", "DIA", "IWM")
DEFAULT_MIN_UNIVERSE_PRICE_USD = 5.0
DEFAULT_MIN_UNIVERSE_AVERAGE_DAILY_DOLLAR_VOLUME_USD = 25_000_000.0
DEFAULT_MIN_UNIVERSE_MARKET_CAP_USD = 5_000_000_000.0
DEFAULT_MIN_UNIVERSE_LIQUIDITY_SCORE = 0.0
DEFAULT_UNIVERSE_SURVIVORSHIP_BIAS_DISCLOSURE = (
    "v1 fixes the research universe at experiment start from provider-available "
    "S&P 500 constituents and explicitly allows survivorship bias; point-in-time "
    "membership reconstruction is deferred to v2."
)

UniverseProviderMode = Literal["configured", "point_in_time"]


@dataclass(frozen=True, slots=True)
class UniverseSelectionConfig:
    """Canonical Stage 1 universe construction contract.

    The provider is responsible for fetching or loading constituents; this
    schema defines the reproducible selection rules the provider must satisfy.
    """

    config_id: str = DEFAULT_UNIVERSE_CONFIG_ID
    provider_id: str = DEFAULT_UNIVERSE_PROVIDER_ID
    provider_mode: UniverseProviderMode = "configured"
    index_name: str = DEFAULT_UNIVERSE_INDEX_NAME
    selection_method: str = DEFAULT_UNIVERSE_SELECTION_METHOD
    selection_count: int = DEFAULT_UNIVERSE_SELECTION_COUNT
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER
    fixed_at_experiment_start: bool = True
    point_in_time_membership: bool = False
    survivorship_bias_allowed: bool = True
    survivorship_bias_disclosure: str = DEFAULT_UNIVERSE_SURVIVORSHIP_BIAS_DISCLOSURE
    required_min_history_years: int = 3
    market_cap_rank_basis: str = "provider_available_market_cap_at_experiment_start"
    allowed_exchanges: tuple[str, ...] = DEFAULT_ALLOWED_UNIVERSE_EXCHANGES
    allowed_security_types: tuple[str, ...] = DEFAULT_ALLOWED_UNIVERSE_SECURITY_TYPES
    allowed_listing_statuses: tuple[str, ...] = DEFAULT_ALLOWED_UNIVERSE_LISTING_STATUSES
    excluded_security_types: tuple[str, ...] = DEFAULT_EXCLUDED_UNIVERSE_SECURITY_TYPES
    excluded_tickers: tuple[str, ...] = DEFAULT_EXCLUDED_UNIVERSE_TICKERS
    min_price_usd: float = DEFAULT_MIN_UNIVERSE_PRICE_USD
    min_average_daily_dollar_volume_usd: float = (
        DEFAULT_MIN_UNIVERSE_AVERAGE_DAILY_DOLLAR_VOLUME_USD
    )
    min_market_cap_usd: float = DEFAULT_MIN_UNIVERSE_MARKET_CAP_USD
    min_liquidity_score: float = DEFAULT_MIN_UNIVERSE_LIQUIDITY_SCORE

    def __post_init__(self) -> None:
        config_id = str(self.config_id).strip()
        provider_id = str(self.provider_id).strip()
        index_name = str(self.index_name).strip()
        selection_method = str(self.selection_method).strip()
        rank_basis = str(self.market_cap_rank_basis).strip()

        if not config_id:
            raise ValueError("config_id must not be blank")
        if not provider_id:
            raise ValueError("provider_id must not be blank")
        if self.provider_mode not in {"configured", "point_in_time"}:
            raise ValueError(f"unsupported universe provider_mode: {self.provider_mode}")
        if index_name != DEFAULT_UNIVERSE_INDEX_NAME:
            raise ValueError("Stage 1 universe index_name must be S&P 500")
        if selection_method != DEFAULT_UNIVERSE_SELECTION_METHOD:
            raise ValueError(
                "Stage 1 universe must use S&P 500 top market-cap selection at experiment start"
            )
        if self.selection_count != DEFAULT_UNIVERSE_SELECTION_COUNT:
            raise ValueError("Stage 1 universe selection_count must be 150")
        benchmark = str(self.benchmark_ticker or "").strip().upper()
        if not benchmark:
            raise ValueError("benchmark_ticker must not be blank")
        if benchmark != DEFAULT_BENCHMARK_TICKER:
            raise ValueError("Stage 1 benchmark_ticker must be SPY")
        if not self.fixed_at_experiment_start:
            raise ValueError("universe selection config must fix membership at experiment start")
        if self.point_in_time_membership:
            raise ValueError("v1 universe config must not claim point-in-time membership")
        if not self.survivorship_bias_allowed:
            raise ValueError("v1 universe config must explicitly allow survivorship bias")
        disclosure = str(self.survivorship_bias_disclosure).strip()
        if "survivorship bias" not in disclosure.lower():
            raise ValueError(
                "survivorship_bias_disclosure must explicitly mention survivorship bias"
            )
        if self.required_min_history_years < 3:
            raise ValueError("required_min_history_years must be at least 3")
        if not rank_basis:
            raise ValueError("market_cap_rank_basis must not be blank")
        allowed_exchanges = _normalize_exchange_values(self.allowed_exchanges)
        allowed_security_types = _normalize_rule_values(self.allowed_security_types)
        allowed_listing_statuses = _normalize_rule_values(self.allowed_listing_statuses)
        excluded_security_types = _normalize_rule_values(self.excluded_security_types)
        excluded_tickers = _normalize_tickers(self.excluded_tickers)
        min_price_usd = _required_float("min_price_usd", self.min_price_usd)
        min_average_daily_dollar_volume_usd = _required_float(
            "min_average_daily_dollar_volume_usd",
            self.min_average_daily_dollar_volume_usd,
        )
        min_market_cap_usd = _required_float("min_market_cap_usd", self.min_market_cap_usd)
        min_liquidity_score = _required_float("min_liquidity_score", self.min_liquidity_score)
        if not allowed_exchanges:
            raise ValueError("allowed_exchanges must contain at least one exchange")
        if not allowed_security_types:
            raise ValueError("allowed_security_types must contain at least one type")
        if not allowed_listing_statuses:
            raise ValueError("allowed_listing_statuses must contain at least one status")
        if benchmark not in excluded_tickers:
            raise ValueError(
                "excluded_tickers must exclude the benchmark ticker from strategy universe"
            )
        overlapping_types = sorted(
            set(allowed_security_types).intersection(excluded_security_types)
        )
        if overlapping_types:
            raise ValueError(
                f"security types cannot be both allowed and excluded: {overlapping_types}"
            )
        _validate_non_negative_threshold("min_price_usd", min_price_usd)
        _validate_non_negative_threshold(
            "min_average_daily_dollar_volume_usd",
            min_average_daily_dollar_volume_usd,
        )
        _validate_non_negative_threshold("min_market_cap_usd", min_market_cap_usd)
        _validate_non_negative_threshold("min_liquidity_score", min_liquidity_score)

        object.__setattr__(self, "config_id", config_id)
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "index_name", index_name)
        object.__setattr__(self, "selection_method", selection_method)
        object.__setattr__(self, "benchmark_ticker", benchmark)
        object.__setattr__(self, "survivorship_bias_disclosure", disclosure)
        object.__setattr__(self, "market_cap_rank_basis", rank_basis)
        object.__setattr__(self, "allowed_exchanges", allowed_exchanges)
        object.__setattr__(self, "allowed_security_types", allowed_security_types)
        object.__setattr__(self, "allowed_listing_statuses", allowed_listing_statuses)
        object.__setattr__(self, "excluded_security_types", excluded_security_types)
        object.__setattr__(self, "excluded_tickers", excluded_tickers)
        object.__setattr__(self, "min_price_usd", min_price_usd)
        object.__setattr__(
            self,
            "min_average_daily_dollar_volume_usd",
            min_average_daily_dollar_volume_usd,
        )
        object.__setattr__(self, "min_market_cap_usd", min_market_cap_usd)
        object.__setattr__(self, "min_liquidity_score", min_liquidity_score)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": UNIVERSE_SELECTION_CONFIG_SCHEMA_VERSION,
            "config_id": self.config_id,
            "provider_id": self.provider_id,
            "provider_mode": self.provider_mode,
            "index_name": self.index_name,
            "selection_method": self.selection_method,
            "selection_count": self.selection_count,
            "benchmark_ticker": self.benchmark_ticker,
            "fixed_at_experiment_start": self.fixed_at_experiment_start,
            "point_in_time_membership": self.point_in_time_membership,
            "survivorship_bias_allowed": self.survivorship_bias_allowed,
            "survivorship_bias_disclosure": self.survivorship_bias_disclosure,
            "required_min_history_years": self.required_min_history_years,
            "market_cap_rank_basis": self.market_cap_rank_basis,
            "allowed_exchanges": list(self.allowed_exchanges),
            "allowed_security_types": list(self.allowed_security_types),
            "allowed_listing_statuses": list(self.allowed_listing_statuses),
            "excluded_security_types": list(self.excluded_security_types),
            "excluded_tickers": list(self.excluded_tickers),
            "min_price_usd": self.min_price_usd,
            "min_average_daily_dollar_volume_usd": self.min_average_daily_dollar_volume_usd,
            "min_market_cap_usd": self.min_market_cap_usd,
            "min_liquidity_score": self.min_liquidity_score,
        }


@dataclass(frozen=True, slots=True)
class UniverseConstituent:
    ticker: str
    name: str | None = None
    sector: str | None = None
    exchange: str = "NASDAQ"
    security_type: str = "common_stock"
    listing_status: str = "active"
    market_cap_rank: int | None = None
    market_cap_usd: float | None = None
    price_usd: float | None = None
    average_daily_volume: float | None = None
    average_daily_dollar_volume_usd: float | None = None
    liquidity_score: float | None = None
    cik: str | None = None

    def __post_init__(self) -> None:
        ticker = _normalize_ticker(self.ticker)
        if not ticker:
            raise ValueError("universe constituent ticker must not be blank")
        object.__setattr__(self, "ticker", ticker)
        object.__setattr__(self, "name", _optional_str(self.name))
        object.__setattr__(self, "sector", _optional_str(self.sector))
        object.__setattr__(self, "exchange", _normalize_exchange(self.exchange))
        object.__setattr__(self, "security_type", _normalize_rule_value(self.security_type))
        object.__setattr__(self, "listing_status", _normalize_rule_value(self.listing_status))
        object.__setattr__(self, "cik", _optional_str(self.cik))
        for field_name in (
            "market_cap_usd",
            "price_usd",
            "average_daily_volume",
            "average_daily_dollar_volume_usd",
            "liquidity_score",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_float(field_name, getattr(self, field_name)),
            )

        if self.market_cap_rank is not None and self.market_cap_rank < 1:
            raise ValueError("market_cap_rank must be positive when provided")
        for field_name in (
            "market_cap_usd",
            "price_usd",
            "average_daily_volume",
            "average_daily_dollar_volume_usd",
            "liquidity_score",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be non-negative when provided")
        if (
            self.average_daily_dollar_volume_usd is None
            and self.price_usd is not None
            and self.average_daily_volume is not None
        ):
            object.__setattr__(
                self,
                "average_daily_dollar_volume_usd",
                self.price_usd * self.average_daily_volume,
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "ticker": self.ticker,
            "name": self.name,
            "sector": self.sector,
            "exchange": self.exchange,
            "security_type": self.security_type,
            "listing_status": self.listing_status,
            "market_cap_rank": self.market_cap_rank,
            "market_cap_usd": self.market_cap_usd,
            "price_usd": self.price_usd,
            "average_daily_volume": self.average_daily_volume,
            "average_daily_dollar_volume_usd": self.average_daily_dollar_volume_usd,
            "liquidity_score": self.liquidity_score,
            "cik": self.cik,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> UniverseConstituent:
        if not isinstance(payload, dict):
            raise TypeError("universe constituent payload must be a dictionary")
        return cls(
            ticker=_required_payload_str(payload, "ticker"),
            name=payload.get("name"),
            sector=payload.get("sector"),
            exchange=str(payload.get("exchange", "NASDAQ")),
            security_type=str(payload.get("security_type", "common_stock")),
            listing_status=str(payload.get("listing_status", "active")),
            market_cap_rank=_optional_int("market_cap_rank", payload.get("market_cap_rank")),
            market_cap_usd=payload.get("market_cap_usd"),
            price_usd=payload.get("price_usd"),
            average_daily_volume=payload.get("average_daily_volume"),
            average_daily_dollar_volume_usd=payload.get("average_daily_dollar_volume_usd"),
            liquidity_score=payload.get("liquidity_score"),
            cik=payload.get("cik"),
        )


@dataclass(frozen=True, slots=True)
class UniverseSnapshot:
    experiment_id: str
    snapshot_date: date
    constituents: tuple[UniverseConstituent, ...]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    universe_name: str = DEFAULT_UNIVERSE_NAME
    constituent_identifier_fields: tuple[str, ...] = (
        DEFAULT_UNIVERSE_CONSTITUENT_IDENTIFIER_FIELDS
    )
    required_constituent_fields: tuple[str, ...] = DEFAULT_UNIVERSE_REQUIRED_CONSTITUENT_FIELDS
    metadata_cutoff_fields: tuple[str, ...] = DEFAULT_UNIVERSE_METADATA_CUTOFF_FIELDS
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER
    index_name: str = DEFAULT_UNIVERSE_INDEX_NAME
    selection_method: str = DEFAULT_UNIVERSE_SELECTION_METHOD
    selection_count: int = DEFAULT_UNIVERSE_SELECTION_COUNT
    fixed_at_experiment_start: bool = True
    point_in_time_membership: bool = False
    survivorship_bias_allowed: bool = True
    survivorship_bias_disclosure: str = DEFAULT_UNIVERSE_SURVIVORSHIP_BIAS_DISCLOSURE
    source: str = "configured_universe"
    source_version: str | None = None
    allowed_exchanges: tuple[str, ...] = DEFAULT_ALLOWED_UNIVERSE_EXCHANGES
    allowed_security_types: tuple[str, ...] = DEFAULT_ALLOWED_UNIVERSE_SECURITY_TYPES
    allowed_listing_statuses: tuple[str, ...] = DEFAULT_ALLOWED_UNIVERSE_LISTING_STATUSES
    excluded_security_types: tuple[str, ...] = DEFAULT_EXCLUDED_UNIVERSE_SECURITY_TYPES
    excluded_tickers: tuple[str, ...] = DEFAULT_EXCLUDED_UNIVERSE_TICKERS
    min_price_usd: float = DEFAULT_MIN_UNIVERSE_PRICE_USD
    min_average_daily_dollar_volume_usd: float = (
        DEFAULT_MIN_UNIVERSE_AVERAGE_DAILY_DOLLAR_VOLUME_USD
    )
    min_market_cap_usd: float = DEFAULT_MIN_UNIVERSE_MARKET_CAP_USD
    min_liquidity_score: float = DEFAULT_MIN_UNIVERSE_LIQUIDITY_SCORE

    def __post_init__(self) -> None:
        experiment_id = str(self.experiment_id).strip()
        if not experiment_id:
            raise ValueError("experiment_id must not be blank")
        snapshot_date = _coerce_snapshot_date(self.snapshot_date, "snapshot_date")
        created_at = _coerce_snapshot_created_at(self.created_at)
        universe_name = str(self.universe_name).strip()
        if not universe_name:
            raise ValueError("universe_name must not be blank")
        constituent_identifier_fields = _normalize_rule_values(
            self.constituent_identifier_fields
        )
        required_constituent_fields = _normalize_rule_values(self.required_constituent_fields)
        metadata_cutoff_fields = _normalize_rule_values(self.metadata_cutoff_fields)
        if "ticker" not in constituent_identifier_fields:
            raise ValueError("constituent_identifier_fields must include ticker")
        if "ticker" not in required_constituent_fields:
            raise ValueError("required_constituent_fields must include ticker")
        if not metadata_cutoff_fields:
            raise ValueError("metadata_cutoff_fields must contain at least one cutoff field")
        if self.selection_count < 1:
            raise ValueError("selection_count must be positive")
        if not self.fixed_at_experiment_start:
            raise ValueError("universe snapshot must be fixed at experiment start")
        if self.point_in_time_membership:
            raise ValueError("v1 universe snapshot must not claim point-in-time membership")
        if not self.survivorship_bias_allowed:
            raise ValueError("v1 universe snapshot must explicitly allow survivorship bias")

        constituents = tuple(
            constituent
            if isinstance(constituent, UniverseConstituent)
            else UniverseConstituent(str(constituent))
            for constituent in self.constituents
        )
        if not constituents:
            raise ValueError("universe snapshot must contain at least one constituent")
        if len(constituents) > self.selection_count:
            raise ValueError("universe snapshot cannot exceed selection_count")

        tickers = [constituent.ticker for constituent in constituents]
        duplicate_tickers = sorted({ticker for ticker in tickers if tickers.count(ticker) > 1})
        if duplicate_tickers:
            raise ValueError(f"universe snapshot contains duplicate tickers: {duplicate_tickers}")

        benchmark = _normalize_ticker(self.benchmark_ticker)
        if not benchmark:
            raise ValueError("benchmark_ticker must not be blank")
        if benchmark in tickers:
            raise ValueError("benchmark_ticker must not be a strategy universe constituent")
        allowed_exchanges = _normalize_exchange_values(self.allowed_exchanges)
        allowed_security_types = _normalize_rule_values(self.allowed_security_types)
        allowed_listing_statuses = _normalize_rule_values(self.allowed_listing_statuses)
        excluded_security_types = _normalize_rule_values(self.excluded_security_types)
        excluded_tickers = _normalize_tickers(self.excluded_tickers)
        min_price_usd = _required_float("min_price_usd", self.min_price_usd)
        min_average_daily_dollar_volume_usd = _required_float(
            "min_average_daily_dollar_volume_usd",
            self.min_average_daily_dollar_volume_usd,
        )
        min_market_cap_usd = _required_float("min_market_cap_usd", self.min_market_cap_usd)
        min_liquidity_score = _required_float("min_liquidity_score", self.min_liquidity_score)
        _validate_non_negative_threshold("min_price_usd", min_price_usd)
        _validate_non_negative_threshold(
            "min_average_daily_dollar_volume_usd",
            min_average_daily_dollar_volume_usd,
        )
        _validate_non_negative_threshold("min_market_cap_usd", min_market_cap_usd)
        _validate_non_negative_threshold("min_liquidity_score", min_liquidity_score)
        if benchmark not in excluded_tickers:
            raise ValueError(
                "excluded_tickers must exclude the benchmark ticker from strategy universe"
            )
        ineligible = _ineligible_constituent_reasons(
            constituents,
            allowed_exchanges=allowed_exchanges,
            allowed_security_types=allowed_security_types,
            allowed_listing_statuses=allowed_listing_statuses,
            excluded_security_types=excluded_security_types,
            excluded_tickers=excluded_tickers,
            min_price_usd=min_price_usd,
            min_average_daily_dollar_volume_usd=min_average_daily_dollar_volume_usd,
            min_market_cap_usd=min_market_cap_usd,
            min_liquidity_score=min_liquidity_score,
        )
        if ineligible:
            reason_text = "; ".join(
                f"{ticker}: {', '.join(reasons)}" for ticker, reasons in ineligible.items()
            )
            raise ValueError(f"universe snapshot contains ineligible constituents: {reason_text}")

        ranks = [
            constituent.market_cap_rank
            for constituent in constituents
            if constituent.market_cap_rank is not None
        ]
        duplicate_ranks = sorted({rank for rank in ranks if ranks.count(rank) > 1})
        if duplicate_ranks:
            raise ValueError(
                f"universe snapshot contains duplicate market_cap_rank values: {duplicate_ranks}"
            )
        if ranks and max(ranks) > self.selection_count:
            raise ValueError("market_cap_rank values must not exceed selection_count")

        disclosure = str(self.survivorship_bias_disclosure).strip()
        if "survivorship bias" not in disclosure.lower():
            raise ValueError(
                "survivorship_bias_disclosure must explicitly mention survivorship bias"
            )

        object.__setattr__(self, "experiment_id", experiment_id)
        object.__setattr__(self, "snapshot_date", snapshot_date)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "universe_name", universe_name)
        object.__setattr__(
            self,
            "constituent_identifier_fields",
            constituent_identifier_fields,
        )
        object.__setattr__(self, "required_constituent_fields", required_constituent_fields)
        object.__setattr__(self, "metadata_cutoff_fields", metadata_cutoff_fields)
        object.__setattr__(self, "constituents", constituents)
        object.__setattr__(self, "benchmark_ticker", benchmark)
        object.__setattr__(self, "index_name", str(self.index_name).strip())
        object.__setattr__(self, "selection_method", str(self.selection_method).strip())
        object.__setattr__(self, "survivorship_bias_disclosure", disclosure)
        object.__setattr__(self, "source", str(self.source).strip())
        object.__setattr__(self, "source_version", _optional_str(self.source_version))
        object.__setattr__(self, "allowed_exchanges", allowed_exchanges)
        object.__setattr__(self, "allowed_security_types", allowed_security_types)
        object.__setattr__(self, "allowed_listing_statuses", allowed_listing_statuses)
        object.__setattr__(self, "excluded_security_types", excluded_security_types)
        object.__setattr__(self, "excluded_tickers", excluded_tickers)
        object.__setattr__(self, "min_price_usd", min_price_usd)
        object.__setattr__(
            self,
            "min_average_daily_dollar_volume_usd",
            min_average_daily_dollar_volume_usd,
        )
        object.__setattr__(self, "min_market_cap_usd", min_market_cap_usd)
        object.__setattr__(self, "min_liquidity_score", min_liquidity_score)

    @classmethod
    def from_tickers(
        cls,
        tickers: Iterable[str],
        *,
        experiment_id: str,
        snapshot_date: date,
        benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER,
        selection_count: int = DEFAULT_UNIVERSE_SELECTION_COUNT,
        source: str = "configured_universe",
        source_version: str | None = None,
    ) -> UniverseSnapshot:
        return cls(
            experiment_id=experiment_id,
            snapshot_date=snapshot_date,
            constituents=tuple(
                UniverseConstituent(ticker=ticker, market_cap_rank=index)
                for index, ticker in enumerate(_normalize_tickers(tickers), start=1)
            ),
            benchmark_ticker=benchmark_ticker,
            selection_count=selection_count,
            source=source,
            source_version=source_version,
        )

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(constituent.ticker for constituent in self.constituents)

    @property
    def constituent_count(self) -> int:
        return len(self.constituents)

    @property
    def ticker_universe(self):
        from quant_research.validation.benchmark_inputs import TickerUniverse

        return TickerUniverse(self.tickers, self.benchmark_ticker)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": UNIVERSE_SNAPSHOT_SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "snapshot_date": self.snapshot_date.isoformat(),
            "created_at": self.created_at.isoformat(),
            "universe_name": self.universe_name,
            "snapshot_metadata": {
                "created_at": self.created_at.isoformat(),
                "reference_date": self.snapshot_date.isoformat(),
                "universe_name": self.universe_name,
                "universe_identifier": (
                    f"{self.universe_name}:{self.snapshot_date.isoformat()}"
                ),
                "index_name": self.index_name,
                "selection_method": self.selection_method,
                "constituent_identifier_fields": list(self.constituent_identifier_fields),
                "primary_constituent_identifier": "ticker",
            },
            "index_name": self.index_name,
            "selection_method": self.selection_method,
            "selection_count": self.selection_count,
            "constituent_count": self.constituent_count,
            "constituents": [constituent.to_dict() for constituent in self.constituents],
            "tickers": list(self.tickers),
            "benchmark_ticker": self.benchmark_ticker,
            "data_tickers": list(self.ticker_universe.data_tickers),
            "fixed_at_experiment_start": self.fixed_at_experiment_start,
            "point_in_time_membership": self.point_in_time_membership,
            "survivorship_bias_allowed": self.survivorship_bias_allowed,
            "survivorship_bias_disclosure": self.survivorship_bias_disclosure,
            "source": self.source,
            "source_version": self.source_version,
            "eligibility_rules": {
                "allowed_exchanges": list(self.allowed_exchanges),
                "allowed_security_types": list(self.allowed_security_types),
                "allowed_listing_statuses": list(self.allowed_listing_statuses),
                "excluded_security_types": list(self.excluded_security_types),
                "excluded_tickers": list(self.excluded_tickers),
                "min_price_usd": self.min_price_usd,
                "min_average_daily_dollar_volume_usd": self.min_average_daily_dollar_volume_usd,
                "min_market_cap_usd": self.min_market_cap_usd,
                "min_liquidity_score": self.min_liquidity_score,
            },
            "data_validation_rules": {
                "schema_version": UNIVERSE_DATA_VALIDATION_RULES_VERSION,
                "snapshot_date_field": "snapshot_date",
                "reference_date_field": "snapshot_metadata.reference_date",
                "snapshot_date_must_equal_experiment_start": True,
                "fixed_universe_required": True,
                "unique_ticker_required": True,
                "benchmark_ticker_excluded": self.benchmark_ticker,
                "max_constituent_count": self.selection_count,
                "required_constituent_fields": list(self.required_constituent_fields),
                "constituent_identifier_fields": list(self.constituent_identifier_fields),
                "metadata_availability_cutoff_fields": list(self.metadata_cutoff_fields),
                "metadata_cutoff_rule": "cutoff dates must be on or before snapshot_date",
                "eligibility_rules": {
                    "allowed_exchanges": list(self.allowed_exchanges),
                    "allowed_security_types": list(self.allowed_security_types),
                    "allowed_listing_statuses": list(self.allowed_listing_statuses),
                    "excluded_security_types": list(self.excluded_security_types),
                    "excluded_tickers": list(self.excluded_tickers),
                    "min_price_usd": self.min_price_usd,
                    "min_average_daily_dollar_volume_usd": (
                        self.min_average_daily_dollar_volume_usd
                    ),
                    "min_market_cap_usd": self.min_market_cap_usd,
                    "min_liquidity_score": self.min_liquidity_score,
                },
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> UniverseSnapshot:
        if not isinstance(payload, dict):
            raise TypeError("universe snapshot payload must be a dictionary")
        schema_version = payload.get("schema_version")
        if schema_version != UNIVERSE_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                "universe snapshot schema_version must be "
                f"{UNIVERSE_SNAPSHOT_SCHEMA_VERSION}"
            )
        constituents_payload = payload.get("constituents")
        if not isinstance(constituents_payload, list):
            raise ValueError("universe snapshot constituents must be a list")
        eligibility_rules = payload.get("eligibility_rules", {})
        if not isinstance(eligibility_rules, dict):
            raise ValueError("universe snapshot eligibility_rules must be a dictionary")
        snapshot_metadata = payload.get("snapshot_metadata", {})
        if snapshot_metadata is not None and not isinstance(snapshot_metadata, dict):
            raise ValueError("universe snapshot snapshot_metadata must be a dictionary")
        snapshot_metadata = snapshot_metadata or {}
        data_validation_rules = payload.get("data_validation_rules", {})
        if data_validation_rules is not None and not isinstance(data_validation_rules, dict):
            raise ValueError("universe snapshot data_validation_rules must be a dictionary")
        data_validation_rules = data_validation_rules or {}
        data_validation_rules_version = data_validation_rules.get(
            "schema_version",
            UNIVERSE_DATA_VALIDATION_RULES_VERSION,
        )
        if data_validation_rules_version != UNIVERSE_DATA_VALIDATION_RULES_VERSION:
            raise ValueError(
                "universe snapshot data_validation_rules.schema_version must be "
                f"{UNIVERSE_DATA_VALIDATION_RULES_VERSION}"
            )

        return cls(
            experiment_id=_required_payload_str(payload, "experiment_id"),
            snapshot_date=_coerce_snapshot_date(payload.get("snapshot_date"), "snapshot_date"),
            constituents=tuple(
                UniverseConstituent.from_dict(_require_mapping(row, "constituents"))
                for row in constituents_payload
            ),
            created_at=_coerce_snapshot_created_at(
                payload.get("created_at", snapshot_metadata.get("created_at"))
            ),
            universe_name=str(
                payload.get(
                    "universe_name",
                    snapshot_metadata.get("universe_name", DEFAULT_UNIVERSE_NAME),
                )
            ),
            constituent_identifier_fields=_payload_tuple(
                data_validation_rules,
                "constituent_identifier_fields",
                _payload_tuple(
                    snapshot_metadata,
                    "constituent_identifier_fields",
                    DEFAULT_UNIVERSE_CONSTITUENT_IDENTIFIER_FIELDS,
                ),
            ),
            required_constituent_fields=_payload_tuple(
                data_validation_rules,
                "required_constituent_fields",
                DEFAULT_UNIVERSE_REQUIRED_CONSTITUENT_FIELDS,
            ),
            metadata_cutoff_fields=_payload_tuple(
                data_validation_rules,
                "metadata_availability_cutoff_fields",
                DEFAULT_UNIVERSE_METADATA_CUTOFF_FIELDS,
            ),
            benchmark_ticker=str(payload.get("benchmark_ticker", DEFAULT_BENCHMARK_TICKER)),
            index_name=str(payload.get("index_name", DEFAULT_UNIVERSE_INDEX_NAME)),
            selection_method=str(
                payload.get("selection_method", DEFAULT_UNIVERSE_SELECTION_METHOD)
            ),
            selection_count=_required_int(
                "selection_count",
                payload.get("selection_count", DEFAULT_UNIVERSE_SELECTION_COUNT),
            ),
            fixed_at_experiment_start=bool(payload.get("fixed_at_experiment_start", True)),
            point_in_time_membership=bool(payload.get("point_in_time_membership", False)),
            survivorship_bias_allowed=bool(payload.get("survivorship_bias_allowed", True)),
            survivorship_bias_disclosure=str(
                payload.get(
                    "survivorship_bias_disclosure",
                    DEFAULT_UNIVERSE_SURVIVORSHIP_BIAS_DISCLOSURE,
                )
            ),
            source=str(payload.get("source", "configured_universe")),
            source_version=payload.get("source_version"),
            allowed_exchanges=_payload_tuple(
                eligibility_rules,
                "allowed_exchanges",
                DEFAULT_ALLOWED_UNIVERSE_EXCHANGES,
            ),
            allowed_security_types=_payload_tuple(
                eligibility_rules,
                "allowed_security_types",
                DEFAULT_ALLOWED_UNIVERSE_SECURITY_TYPES,
            ),
            allowed_listing_statuses=_payload_tuple(
                eligibility_rules,
                "allowed_listing_statuses",
                DEFAULT_ALLOWED_UNIVERSE_LISTING_STATUSES,
            ),
            excluded_security_types=_payload_tuple(
                eligibility_rules,
                "excluded_security_types",
                DEFAULT_EXCLUDED_UNIVERSE_SECURITY_TYPES,
            ),
            excluded_tickers=_payload_tuple(
                eligibility_rules,
                "excluded_tickers",
                DEFAULT_EXCLUDED_UNIVERSE_TICKERS,
            ),
            min_price_usd=eligibility_rules.get("min_price_usd", DEFAULT_MIN_UNIVERSE_PRICE_USD),
            min_average_daily_dollar_volume_usd=eligibility_rules.get(
                "min_average_daily_dollar_volume_usd",
                DEFAULT_MIN_UNIVERSE_AVERAGE_DAILY_DOLLAR_VOLUME_USD,
            ),
            min_market_cap_usd=eligibility_rules.get(
                "min_market_cap_usd",
                DEFAULT_MIN_UNIVERSE_MARKET_CAP_USD,
            ),
            min_liquidity_score=eligibility_rules.get(
                "min_liquidity_score",
                DEFAULT_MIN_UNIVERSE_LIQUIDITY_SCORE,
            ),
        )


def _normalize_tickers(tickers: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        value = _normalize_ticker(ticker)
        if value and value not in seen:
            normalized.append(value)
            seen.add(value)
    return tuple(normalized)


def _normalize_ticker(ticker: str | None) -> str:
    return str(ticker or "").strip().upper()


def _normalize_exchange(exchange: str | None) -> str:
    return str(exchange or "").strip().upper()


def _normalize_exchange_values(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(value for value in (_normalize_exchange(value) for value in values) if value)


def _normalize_rule_value(value: str | None) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def _normalize_rule_values(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(value for value in (_normalize_rule_value(value) for value in values) if value)


def _ineligible_constituent_reasons(
    constituents: Iterable[UniverseConstituent],
    *,
    allowed_exchanges: tuple[str, ...],
    allowed_security_types: tuple[str, ...],
    allowed_listing_statuses: tuple[str, ...],
    excluded_security_types: tuple[str, ...],
    excluded_tickers: tuple[str, ...],
    min_price_usd: float = DEFAULT_MIN_UNIVERSE_PRICE_USD,
    min_average_daily_dollar_volume_usd: float = (
        DEFAULT_MIN_UNIVERSE_AVERAGE_DAILY_DOLLAR_VOLUME_USD
    ),
    min_market_cap_usd: float = DEFAULT_MIN_UNIVERSE_MARKET_CAP_USD,
    min_liquidity_score: float = DEFAULT_MIN_UNIVERSE_LIQUIDITY_SCORE,
) -> dict[str, tuple[str, ...]]:
    ineligible: dict[str, tuple[str, ...]] = {}
    for constituent in constituents:
        reasons: list[str] = []
        if constituent.ticker in excluded_tickers:
            reasons.append("excluded ticker")
        if constituent.exchange not in allowed_exchanges:
            reasons.append(f"exchange {constituent.exchange} is not allowed")
        if constituent.security_type not in allowed_security_types:
            reasons.append(f"security_type {constituent.security_type} is not allowed")
        if constituent.security_type in excluded_security_types:
            reasons.append(f"security_type {constituent.security_type} is excluded")
        if constituent.listing_status not in allowed_listing_statuses:
            reasons.append(f"listing_status {constituent.listing_status} is not allowed")
        if (
            constituent.price_usd is not None
            and constituent.price_usd < min_price_usd
        ):
            reasons.append(f"price_usd {constituent.price_usd:g} is below {min_price_usd:g}")
        if (
            constituent.average_daily_dollar_volume_usd is not None
            and constituent.average_daily_dollar_volume_usd
            < min_average_daily_dollar_volume_usd
        ):
            reasons.append(
                "average_daily_dollar_volume_usd "
                f"{constituent.average_daily_dollar_volume_usd:g} is below "
                f"{min_average_daily_dollar_volume_usd:g}"
            )
        if (
            constituent.market_cap_usd is not None
            and constituent.market_cap_usd < min_market_cap_usd
        ):
            reasons.append(
                f"market_cap_usd {constituent.market_cap_usd:g} is below {min_market_cap_usd:g}"
            )
        if (
            constituent.liquidity_score is not None
            and constituent.liquidity_score < min_liquidity_score
        ):
            reasons.append(
                f"liquidity_score {constituent.liquidity_score:g} is below "
                f"{min_liquidity_score:g}"
            )
        if reasons:
            ineligible[constituent.ticker] = tuple(reasons)
    return ineligible


def _validate_non_negative_threshold(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _optional_float(name: str, value: object | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric when provided") from exc


def _required_float(name: str, value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _optional_int(name: str, value: object | None) -> int | None:
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer when provided") from exc
    return result


def _required_int(name: str, value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _required_payload_str(payload: dict[str, object], key: str) -> str:
    value = str(payload.get(key, "")).strip()
    if not value:
        raise ValueError(f"universe snapshot {key} must not be blank")
    return value


def _coerce_snapshot_date(value: object, field_name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO date") from exc


def _coerce_snapshot_created_at(value: object | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if isinstance(value, datetime):
        result = value
    else:
        raw = str(value).strip()
        if not raw:
            return datetime.now(UTC)
        try:
            result = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("created_at must be an ISO datetime") from exc
    if result.tzinfo is None:
        result = result.replace(tzinfo=UTC)
    return result.astimezone(UTC)


def _require_mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} entries must be dictionaries")
    return value


def _payload_tuple(
    payload: dict[str, Any],
    key: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    value = payload.get(key, default)
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise ValueError(f"eligibility_rules.{key} must be a list")
    return tuple(str(item) for item in value)


DEFAULT_UNIVERSE_SELECTION_CONFIG = UniverseSelectionConfig()
