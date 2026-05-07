from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Protocol

from quant_research.config import DEFAULT_TICKERS
from quant_research.validation.universe import (
    DEFAULT_UNIVERSE_INDEX_NAME,
    DEFAULT_UNIVERSE_SELECTION_CONFIG,
    DEFAULT_UNIVERSE_SELECTION_METHOD,
    UniverseConstituent,
    UniverseSelectionConfig,
    UniverseSnapshot,
)


class UniverseProvider(Protocol):
    def get_universe_snapshot(
        self,
        *,
        experiment_id: str,
        experiment_start: date,
        config: UniverseSelectionConfig = DEFAULT_UNIVERSE_SELECTION_CONFIG,
    ) -> UniverseSnapshot:
        """Return the strategy universe fixed at the experiment start date."""


@dataclass(frozen=True, slots=True)
class UniverseConstructionRequest:
    """Input boundary for constructing the fixed research universe."""

    experiment_id: str
    as_of_date: date | datetime | str
    definition: UniverseSelectionConfig = DEFAULT_UNIVERSE_SELECTION_CONFIG

    def __post_init__(self) -> None:
        experiment_id = str(self.experiment_id).strip()
        if not experiment_id:
            raise ValueError("experiment_id must not be blank")
        if not isinstance(self.definition, UniverseSelectionConfig):
            raise TypeError("definition must be a UniverseSelectionConfig")

        object.__setattr__(self, "experiment_id", experiment_id)
        object.__setattr__(
            self,
            "as_of_date",
            _coerce_metadata_date(self.as_of_date, "as_of_date"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "as_of_date": self.as_of_date.isoformat(),
            "definition": self.definition.to_dict(),
        }


class UniverseService(Protocol):
    def build_universe(
        self,
        request: UniverseConstructionRequest,
    ) -> UniverseSnapshot:
        """Return a reproducible snapshot for the supplied universe definition."""

    def list_constituents(
        self,
        request: UniverseConstructionRequest,
    ) -> tuple[UniverseConstituent, ...]:
        """Return the constructed strategy constituent list."""


class UniverseSnapshotRepository(Protocol):
    def save(self, snapshot: UniverseSnapshot) -> Path:
        """Persist a schema-versioned universe snapshot and return its local path."""

    def load(self, experiment_id: str) -> UniverseSnapshot:
        """Reload a universe snapshot by experiment id."""

    def path_for(self, experiment_id: str) -> Path:
        """Return the canonical local path for an experiment snapshot."""


@dataclass(frozen=True, slots=True)
class FileUniverseSnapshotRepository:
    """JSON-backed local repository for reproducible universe snapshots."""

    root_dir: str | Path = Path("artifacts") / "universe_snapshots"

    def save(self, snapshot: UniverseSnapshot) -> Path:
        path = self.path_for(snapshot.experiment_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def load(self, experiment_id: str) -> UniverseSnapshot:
        path = self.path_for(experiment_id)
        if not path.exists():
            raise FileNotFoundError(f"universe snapshot not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"universe snapshot is not valid JSON: {path}") from exc
        return UniverseSnapshot.from_dict(payload)

    def path_for(self, experiment_id: str) -> Path:
        normalized = _snapshot_file_stem(experiment_id)
        return Path(self.root_dir) / f"{normalized}.json"


@dataclass(frozen=True, slots=True)
class ConfiguredUniverseProvider:
    """Offline universe provider backed by configured tickers.

    This keeps tests and CI independent of live index membership or market-cap
    feeds while preserving the stage-1 universe snapshot contract.
    """

    tickers: tuple[str, ...] = DEFAULT_TICKERS
    constituent_metadata: Mapping[str, Mapping[str, object]] | None = None
    index_name: str = DEFAULT_UNIVERSE_INDEX_NAME
    selection_method: str = DEFAULT_UNIVERSE_SELECTION_METHOD
    source: str = "configured_universe"
    source_version: str | None = None

    def get_universe_snapshot(
        self,
        *,
        experiment_id: str,
        experiment_start: date,
        config: UniverseSelectionConfig = DEFAULT_UNIVERSE_SELECTION_CONFIG,
    ) -> UniverseSnapshot:
        metadata_by_ticker = {
            str(ticker).strip().upper(): dict(metadata)
            for ticker, metadata in (self.constituent_metadata or {}).items()
        }
        candidate_constituents: list[UniverseConstituent] = []
        for ticker in self.tickers:
            raw_metadata = metadata_by_ticker.get(str(ticker).strip().upper(), {})
            if not _metadata_available_by_experiment_start(raw_metadata, experiment_start):
                continue
            metadata = _constituent_metadata(raw_metadata)
            candidate = UniverseConstituent(ticker=ticker, **metadata)
            if not _is_eligible_constituent(candidate, config):
                continue
            candidate_constituents.append(candidate)
        candidate_constituents = _rank_by_provider_market_cap(candidate_constituents)
        candidate_constituents = candidate_constituents[: config.selection_count]

        constituents = tuple(
            UniverseConstituent(
                ticker=constituent.ticker,
                name=constituent.name,
                sector=constituent.sector,
                exchange=constituent.exchange,
                security_type=constituent.security_type,
                listing_status=constituent.listing_status,
                market_cap_rank=rank,
                market_cap_usd=constituent.market_cap_usd,
                price_usd=constituent.price_usd,
                average_daily_volume=constituent.average_daily_volume,
                average_daily_dollar_volume_usd=constituent.average_daily_dollar_volume_usd,
                liquidity_score=constituent.liquidity_score,
                cik=constituent.cik,
            )
            for rank, constituent in enumerate(candidate_constituents, start=1)
        )
        return UniverseSnapshot(
            experiment_id=experiment_id,
            snapshot_date=experiment_start,
            constituents=constituents,
            benchmark_ticker=config.benchmark_ticker,
            index_name=config.index_name,
            selection_method=config.selection_method,
            selection_count=config.selection_count,
            fixed_at_experiment_start=config.fixed_at_experiment_start,
            point_in_time_membership=config.point_in_time_membership,
            survivorship_bias_allowed=config.survivorship_bias_allowed,
            survivorship_bias_disclosure=config.survivorship_bias_disclosure,
            source=self.source,
            source_version=self.source_version,
            allowed_exchanges=config.allowed_exchanges,
            allowed_security_types=config.allowed_security_types,
            allowed_listing_statuses=config.allowed_listing_statuses,
            excluded_security_types=config.excluded_security_types,
            excluded_tickers=config.excluded_tickers,
            min_price_usd=config.min_price_usd,
            min_average_daily_dollar_volume_usd=config.min_average_daily_dollar_volume_usd,
            min_market_cap_usd=config.min_market_cap_usd,
            min_liquidity_score=config.min_liquidity_score,
        )


@dataclass(frozen=True, slots=True)
class ProviderBackedUniverseService:
    """Universe construction service that delegates data access to a provider."""

    provider: UniverseProvider

    def build_universe(
        self,
        request: UniverseConstructionRequest,
    ) -> UniverseSnapshot:
        return self.provider.get_universe_snapshot(
            experiment_id=request.experiment_id,
            experiment_start=request.as_of_date,
            config=request.definition,
        )

    def list_constituents(
        self,
        request: UniverseConstructionRequest,
    ) -> tuple[UniverseConstituent, ...]:
        return self.build_universe(request).constituents


def _is_eligible_constituent(
    constituent: UniverseConstituent,
    config: UniverseSelectionConfig,
) -> bool:
    return (
        constituent.ticker not in config.excluded_tickers
        and constituent.exchange in config.allowed_exchanges
        and constituent.security_type in config.allowed_security_types
        and constituent.security_type not in config.excluded_security_types
        and constituent.listing_status in config.allowed_listing_statuses
        and (
            constituent.price_usd is None
            or constituent.price_usd >= config.min_price_usd
        )
        and (
            constituent.average_daily_dollar_volume_usd is None
            or constituent.average_daily_dollar_volume_usd
            >= config.min_average_daily_dollar_volume_usd
        )
        and (
            constituent.market_cap_usd is None
            or constituent.market_cap_usd >= config.min_market_cap_usd
        )
        and (
            constituent.liquidity_score is None
            or constituent.liquidity_score >= config.min_liquidity_score
        )
    )


def _rank_by_provider_market_cap(
    constituents: list[UniverseConstituent],
) -> list[UniverseConstituent]:
    ranked = sorted(
        enumerate(constituents),
        key=lambda item: (
            item[1].market_cap_rank is None,
            item[1].market_cap_rank or 0,
            -(item[1].market_cap_usd or 0.0),
            item[0],
        ),
    )
    return [constituent for _, constituent in ranked]


def _constituent_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    allowed_keys = {
        "name",
        "sector",
        "exchange",
        "security_type",
        "listing_status",
        "market_cap_rank",
        "market_cap_usd",
        "price_usd",
        "average_daily_volume",
        "average_daily_dollar_volume_usd",
        "liquidity_score",
        "cik",
    }
    return {key: value for key, value in metadata.items() if key in allowed_keys}


def _metadata_available_by_experiment_start(
    metadata: Mapping[str, object],
    experiment_start: date,
) -> bool:
    for key in ("available_at", "as_of_date", "effective_date"):
        if key not in metadata:
            continue
        observed_at = _coerce_metadata_date(metadata[key], key)
        if observed_at > experiment_start:
            return False
    return True


def _coerce_metadata_date(value: object, field_name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO date when provided") from exc


def _snapshot_file_stem(experiment_id: str) -> str:
    normalized = str(experiment_id).strip()
    if not normalized:
        raise ValueError("experiment_id must not be blank")
    safe = "".join(character if character.isalnum() or character in "._-" else "_" for character in normalized)
    safe = safe.strip("._-")
    if not safe:
        raise ValueError("experiment_id must contain at least one file-safe character")
    return safe
