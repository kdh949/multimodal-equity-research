from __future__ import annotations

import json
from datetime import date

import pytest

from quant_research.config import DEFAULT_BENCHMARK_TICKER, DEFAULT_TICKERS
from quant_research.data import (
    ConfiguredUniverseProvider,
    FileUniverseSnapshotRepository,
    ProviderBackedUniverseService,
    UniverseConstructionRequest,
    UniverseProvider,
    UniverseService,
    UniverseSnapshotRepository,
)
from quant_research.data.market import SyntheticMarketDataProvider
from quant_research.validation import (
    DEFAULT_ALLOWED_UNIVERSE_EXCHANGES,
    DEFAULT_ALLOWED_UNIVERSE_LISTING_STATUSES,
    DEFAULT_ALLOWED_UNIVERSE_SECURITY_TYPES,
    DEFAULT_EXCLUDED_UNIVERSE_SECURITY_TYPES,
    DEFAULT_EXCLUDED_UNIVERSE_TICKERS,
    DEFAULT_MIN_UNIVERSE_AVERAGE_DAILY_DOLLAR_VOLUME_USD,
    DEFAULT_MIN_UNIVERSE_LIQUIDITY_SCORE,
    DEFAULT_MIN_UNIVERSE_MARKET_CAP_USD,
    DEFAULT_MIN_UNIVERSE_PRICE_USD,
    DEFAULT_UNIVERSE_CONSTITUENT_IDENTIFIER_FIELDS,
    DEFAULT_UNIVERSE_METADATA_CUTOFF_FIELDS,
    DEFAULT_UNIVERSE_NAME,
    DEFAULT_UNIVERSE_REQUIRED_CONSTITUENT_FIELDS,
    DEFAULT_UNIVERSE_SELECTION_CONFIG,
    DEFAULT_UNIVERSE_SELECTION_COUNT,
    DEFAULT_UNIVERSE_SELECTION_METHOD,
    UNIVERSE_DATA_VALIDATION_RULES_VERSION,
    UNIVERSE_SELECTION_CONFIG_SCHEMA_VERSION,
    UNIVERSE_SNAPSHOT_SCHEMA_VERSION,
    TickerUniverse,
    UniverseConstituent,
    UniverseSelectionConfig,
    UniverseSnapshot,
)


@pytest.fixture
def canonical_universe_snapshot_fixture() -> dict[str, object]:
    return {
        "experiment_id": "stage1/live readiness 2026-01-02",
        "as_of_date": "2026-01-02",
        "tickers": ("AAPL", "MSFT", "FUTUREGIANT", "SPY"),
        "metadata": {
            "AAPL": {
                "market_cap_usd": 50_000_000_000,
                "price_usd": 150.0,
                "average_daily_dollar_volume_usd": 500_000_000,
                "as_of_date": "2026-01-02",
                "sector": "Information Technology",
                "cik": "0000320193",
            },
            "MSFT": {
                "market_cap_usd": 60_000_000_000,
                "price_usd": 300.0,
                "average_daily_dollar_volume_usd": 600_000_000,
                "as_of_date": "2026-01-02",
                "sector": "Information Technology",
                "cik": "0000789019",
            },
            "FUTUREGIANT": {
                "market_cap_rank": 1,
                "market_cap_usd": 5_000_000_000_000,
                "price_usd": 500.0,
                "average_daily_dollar_volume_usd": 10_000_000_000,
                "as_of_date": "2026-01-03",
            },
            "SPY": {
                "security_type": "etf",
                "market_cap_usd": 500_000_000_000,
                "price_usd": 500.0,
                "average_daily_dollar_volume_usd": 20_000_000_000,
                "as_of_date": "2026-01-02",
            },
        },
    }


def test_default_universe_is_broad_liquid_equity_candidate_set() -> None:
    assert len(DEFAULT_TICKERS) >= 25
    assert len(DEFAULT_TICKERS) == len(set(DEFAULT_TICKERS))
    assert all(ticker == ticker.strip().upper() for ticker in DEFAULT_TICKERS)
    assert DEFAULT_BENCHMARK_TICKER == "SPY"
    assert "SPY" not in DEFAULT_TICKERS
    assert "QQQ" not in DEFAULT_TICKERS


def test_ticker_universe_keeps_benchmark_for_data_only() -> None:
    universe = TickerUniverse(DEFAULT_TICKERS, DEFAULT_BENCHMARK_TICKER)

    assert universe.benchmark_in_universe is False
    assert universe.benchmark_ticker == "SPY"
    assert universe.data_tickers[-1] == "SPY"
    assert set(universe.tickers).isdisjoint({"SPY", "QQQ"})


def test_default_universe_is_offline_synthetic_data_compatible() -> None:
    universe = TickerUniverse(DEFAULT_TICKERS, DEFAULT_BENCHMARK_TICKER)

    frame = SyntheticMarketDataProvider(periods=5).get_history(list(universe.data_tickers))

    assert set(frame["ticker"].unique()) == set(universe.data_tickers)
    assert frame.groupby("ticker").size().min() == 5


def test_universe_snapshot_model_captures_stage1_selection_contract() -> None:
    snapshot = UniverseSnapshot.from_tickers(
        [" aapl ", "MSFT", "aapl"],
        experiment_id="stage1_2026_01",
        snapshot_date=date(2026, 1, 2),
        source="synthetic_test_fixture",
        source_version="unit-test",
    )

    assert snapshot.experiment_id == "stage1_2026_01"
    assert snapshot.snapshot_date == date(2026, 1, 2)
    assert snapshot.tickers == ("AAPL", "MSFT")
    assert snapshot.constituent_count == 2
    assert snapshot.selection_count == DEFAULT_UNIVERSE_SELECTION_COUNT
    assert snapshot.selection_method == DEFAULT_UNIVERSE_SELECTION_METHOD
    assert snapshot.fixed_at_experiment_start is True
    assert snapshot.point_in_time_membership is False
    assert snapshot.survivorship_bias_allowed is True
    assert "survivorship bias" in snapshot.survivorship_bias_disclosure.lower()
    assert snapshot.ticker_universe.data_tickers == ("AAPL", "MSFT", "SPY")

    payload = snapshot.to_dict()
    assert payload["schema_version"] == UNIVERSE_SNAPSHOT_SCHEMA_VERSION
    assert payload["universe_name"] == DEFAULT_UNIVERSE_NAME
    assert payload["snapshot_metadata"] == {
        "created_at": snapshot.created_at.isoformat(),
        "reference_date": "2026-01-02",
        "universe_name": DEFAULT_UNIVERSE_NAME,
        "universe_identifier": "stage1_sp500_top150:2026-01-02",
        "index_name": "S&P 500",
        "selection_method": DEFAULT_UNIVERSE_SELECTION_METHOD,
        "constituent_identifier_fields": list(
            DEFAULT_UNIVERSE_CONSTITUENT_IDENTIFIER_FIELDS
        ),
        "primary_constituent_identifier": "ticker",
    }
    assert payload["selection_count"] == 150
    assert payload["constituents"][0]["market_cap_rank"] == 1
    assert payload["data_tickers"] == ["AAPL", "MSFT", "SPY"]
    assert payload["data_validation_rules"]["schema_version"] == (
        UNIVERSE_DATA_VALIDATION_RULES_VERSION
    )
    assert payload["data_validation_rules"]["reference_date_field"] == (
        "snapshot_metadata.reference_date"
    )
    assert payload["data_validation_rules"]["required_constituent_fields"] == list(
        DEFAULT_UNIVERSE_REQUIRED_CONSTITUENT_FIELDS
    )
    assert payload["data_validation_rules"]["constituent_identifier_fields"] == list(
        DEFAULT_UNIVERSE_CONSTITUENT_IDENTIFIER_FIELDS
    )
    assert payload["data_validation_rules"]["metadata_availability_cutoff_fields"] == list(
        DEFAULT_UNIVERSE_METADATA_CUTOFF_FIELDS
    )
    json.dumps(payload)


def test_universe_snapshot_schema_round_trips_from_dict() -> None:
    snapshot = UniverseSnapshot(
        experiment_id="stage1_2026_01",
        snapshot_date=date(2026, 1, 2),
        constituents=(
            UniverseConstituent(
                "MSFT",
                name="Microsoft",
                sector="Information Technology",
                exchange="NASDAQ",
                market_cap_rank=1,
                market_cap_usd=3_000_000_000_000,
                price_usd=400.0,
                average_daily_volume=20_000_000,
                liquidity_score=99.0,
                cik="0000789019",
            ),
            UniverseConstituent(
                "AAPL",
                name="Apple",
                sector="Information Technology",
                exchange="NASDAQ",
                market_cap_rank=2,
                market_cap_usd=2_800_000_000_000,
                price_usd=190.0,
                average_daily_dollar_volume_usd=1_000_000_000,
                liquidity_score=98.0,
                cik="0000320193",
            ),
        ),
        source="unit_test_config",
        source_version="2026-01-02",
    )

    reloaded = UniverseSnapshot.from_dict(snapshot.to_dict())

    assert reloaded == snapshot
    assert reloaded.to_dict() == snapshot.to_dict()


def test_universe_snapshot_schema_rejects_wrong_version() -> None:
    payload = UniverseSnapshot.from_tickers(
        ["AAPL"],
        experiment_id="stage1_2026_01",
        snapshot_date=date(2026, 1, 2),
    ).to_dict()
    payload["schema_version"] = "universe_snapshot.v0"

    with pytest.raises(ValueError, match="schema_version"):
        UniverseSnapshot.from_dict(payload)


def test_file_universe_snapshot_repository_saves_and_reloads_schema_snapshot(tmp_path) -> None:
    repository: UniverseSnapshotRepository = FileUniverseSnapshotRepository(tmp_path)
    snapshot = UniverseSnapshot.from_tickers(
        ["MSFT", "AAPL"],
        experiment_id="stage1/2026 01",
        snapshot_date=date(2026, 1, 2),
        source="unit_test_config",
        source_version="2026-01-02",
    )

    path = repository.save(snapshot)
    payload = json.loads(path.read_text(encoding="utf-8"))
    reloaded = repository.load("stage1/2026 01")

    assert path == tmp_path / "stage1_2026_01.json"
    assert payload["schema_version"] == UNIVERSE_SNAPSHOT_SCHEMA_VERSION
    assert payload["experiment_id"] == "stage1/2026 01"
    assert payload["survivorship_bias_allowed"] is True
    assert "survivorship bias" in payload["survivorship_bias_disclosure"].lower()
    assert reloaded == snapshot


def test_specific_reference_date_snapshot_is_created_and_saved_to_canonical_path(
    tmp_path,
    canonical_universe_snapshot_fixture,
) -> None:
    provider = ConfiguredUniverseProvider(
        tickers=canonical_universe_snapshot_fixture["tickers"],
        constituent_metadata=canonical_universe_snapshot_fixture["metadata"],
        source="unit_test_config",
        source_version=canonical_universe_snapshot_fixture["as_of_date"],
    )
    service = ProviderBackedUniverseService(provider)
    repository = FileUniverseSnapshotRepository(tmp_path / "snapshots")
    request = UniverseConstructionRequest(
        experiment_id=canonical_universe_snapshot_fixture["experiment_id"],
        as_of_date=canonical_universe_snapshot_fixture["as_of_date"],
    )

    snapshot = service.build_universe(request)
    path = repository.save(snapshot)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert snapshot.snapshot_date == date(2026, 1, 2)
    assert snapshot.tickers == ("MSFT", "AAPL")
    assert "FUTUREGIANT" not in snapshot.tickers
    assert "SPY" not in snapshot.tickers
    assert path == tmp_path / "snapshots" / "stage1_live_readiness_2026-01-02.json"
    assert repository.path_for(request.experiment_id) == path
    assert payload["snapshot_metadata"]["reference_date"] == "2026-01-02"
    assert payload["snapshot_metadata"]["universe_identifier"] == (
        "stage1_sp500_top150:2026-01-02"
    )
    assert payload["data_validation_rules"]["metadata_cutoff_rule"] == (
        "cutoff dates must be on or before snapshot_date"
    )
    assert repository.load(request.experiment_id) == snapshot


def test_universe_snapshot_generation_path_persists_reproducible_contract_payload(
    tmp_path,
    canonical_universe_snapshot_fixture,
) -> None:
    provider = ConfiguredUniverseProvider(
        tickers=canonical_universe_snapshot_fixture["tickers"],
        constituent_metadata=canonical_universe_snapshot_fixture["metadata"],
        source="unit_test_config",
        source_version=canonical_universe_snapshot_fixture["as_of_date"],
    )
    service: UniverseService = ProviderBackedUniverseService(provider)
    repository: UniverseSnapshotRepository = FileUniverseSnapshotRepository(
        tmp_path / "universe_snapshots"
    )
    request = UniverseConstructionRequest(
        experiment_id=canonical_universe_snapshot_fixture["experiment_id"],
        as_of_date=canonical_universe_snapshot_fixture["as_of_date"],
        definition=DEFAULT_UNIVERSE_SELECTION_CONFIG,
    )

    snapshot = service.build_universe(request)
    artifact_path = repository.save(snapshot)
    artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert artifact_path == (
        tmp_path
        / "universe_snapshots"
        / "stage1_live_readiness_2026-01-02.json"
    )
    assert artifact_payload["schema_version"] == UNIVERSE_SNAPSHOT_SCHEMA_VERSION
    assert artifact_payload["experiment_id"] == request.experiment_id
    assert artifact_payload["snapshot_date"] == request.as_of_date.isoformat()
    assert artifact_payload["snapshot_metadata"]["reference_date"] == (
        request.as_of_date.isoformat()
    )
    assert artifact_payload["selection_method"] == DEFAULT_UNIVERSE_SELECTION_METHOD
    assert artifact_payload["selection_count"] == DEFAULT_UNIVERSE_SELECTION_COUNT
    assert artifact_payload["constituent_count"] == 2
    assert artifact_payload["tickers"] == ["MSFT", "AAPL"]
    assert artifact_payload["benchmark_ticker"] == "SPY"
    assert artifact_payload["data_tickers"] == ["MSFT", "AAPL", "SPY"]
    assert artifact_payload["fixed_at_experiment_start"] is True
    assert artifact_payload["point_in_time_membership"] is False
    assert artifact_payload["survivorship_bias_allowed"] is True
    assert "survivorship bias" in artifact_payload["survivorship_bias_disclosure"].lower()
    assert artifact_payload["data_validation_rules"]["metadata_cutoff_rule"] == (
        "cutoff dates must be on or before snapshot_date"
    )
    assert artifact_payload["data_validation_rules"]["benchmark_ticker_excluded"] == "SPY"
    assert artifact_payload["eligibility_rules"]["excluded_tickers"] == [
        "SPY",
        "QQQ",
        "DIA",
        "IWM",
    ]
    assert repository.load(request.experiment_id).to_dict() == snapshot.to_dict()


def test_file_universe_snapshot_repository_reports_missing_and_corrupt_files(tmp_path) -> None:
    repository = FileUniverseSnapshotRepository(tmp_path)

    with pytest.raises(FileNotFoundError, match="universe snapshot not found"):
        repository.load("missing")

    corrupt_path = repository.path_for("corrupt")
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_text("{broken-json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        repository.load("corrupt")


def test_universe_selection_config_schema_captures_stage1_provider_contract() -> None:
    config = DEFAULT_UNIVERSE_SELECTION_CONFIG

    assert config.config_id == "stage1_sp500_top150"
    assert config.provider_mode == "configured"
    assert config.index_name == "S&P 500"
    assert config.selection_method == DEFAULT_UNIVERSE_SELECTION_METHOD
    assert config.selection_count == 150
    assert config.benchmark_ticker == "SPY"
    assert config.fixed_at_experiment_start is True
    assert config.point_in_time_membership is False
    assert config.survivorship_bias_allowed is True
    assert config.required_min_history_years == 3
    assert "experiment_start" in config.selection_method
    assert config.allowed_exchanges == DEFAULT_ALLOWED_UNIVERSE_EXCHANGES
    assert config.allowed_security_types == DEFAULT_ALLOWED_UNIVERSE_SECURITY_TYPES
    assert config.allowed_listing_statuses == DEFAULT_ALLOWED_UNIVERSE_LISTING_STATUSES
    assert config.excluded_security_types == DEFAULT_EXCLUDED_UNIVERSE_SECURITY_TYPES
    assert config.excluded_tickers == DEFAULT_EXCLUDED_UNIVERSE_TICKERS
    assert config.min_price_usd == DEFAULT_MIN_UNIVERSE_PRICE_USD
    assert (
        config.min_average_daily_dollar_volume_usd
        == DEFAULT_MIN_UNIVERSE_AVERAGE_DAILY_DOLLAR_VOLUME_USD
    )
    assert config.min_market_cap_usd == DEFAULT_MIN_UNIVERSE_MARKET_CAP_USD
    assert config.min_liquidity_score == DEFAULT_MIN_UNIVERSE_LIQUIDITY_SCORE

    payload = config.to_dict()
    assert payload["schema_version"] == UNIVERSE_SELECTION_CONFIG_SCHEMA_VERSION
    assert payload["selection_count"] == 150
    assert payload["provider_id"] == "configured_sp500_top_market_cap"
    assert payload["allowed_exchanges"] == ["NYSE", "NASDAQ"]
    assert payload["allowed_security_types"] == ["common_stock"]
    assert payload["allowed_listing_statuses"] == ["active"]
    assert payload["excluded_security_types"] == [
        "adr",
        "closed_end_fund",
        "etf",
        "fund",
        "preferred_stock",
        "reit_preferred",
        "unit",
        "warrant",
    ]
    assert payload["excluded_tickers"] == ["SPY", "QQQ", "DIA", "IWM"]
    assert payload["min_price_usd"] == 5.0
    assert payload["min_average_daily_dollar_volume_usd"] == 25_000_000.0
    assert payload["min_market_cap_usd"] == 5_000_000_000.0
    assert payload["min_liquidity_score"] == 0.0
    json.dumps(payload)


def test_universe_provider_interface_returns_fixed_experiment_start_snapshot() -> None:
    provider: UniverseProvider = ConfiguredUniverseProvider(
        tickers=("msft", "aapl", "nvda"),
        source="unit_test_config",
        source_version="2026-01-02",
    )

    snapshot = provider.get_universe_snapshot(
        experiment_id="stage1_2026_01",
        experiment_start=date(2026, 1, 2),
    )

    assert snapshot.snapshot_date == date(2026, 1, 2)
    assert snapshot.tickers == ("MSFT", "AAPL", "NVDA")
    assert snapshot.constituent_count == 3
    assert snapshot.selection_count == 150
    assert snapshot.selection_method == DEFAULT_UNIVERSE_SELECTION_METHOD
    assert snapshot.index_name == "S&P 500"
    assert snapshot.fixed_at_experiment_start is True
    assert snapshot.point_in_time_membership is False
    assert snapshot.survivorship_bias_allowed is True
    assert snapshot.source == "unit_test_config"
    assert snapshot.source_version == "2026-01-02"
    assert [row.market_cap_rank for row in snapshot.constituents] == [1, 2, 3]
    assert snapshot.to_dict()["eligibility_rules"]["allowed_exchanges"] == ["NYSE", "NASDAQ"]
    assert snapshot.to_dict()["snapshot_metadata"]["reference_date"] == "2026-01-02"
    assert snapshot.to_dict()["data_validation_rules"]["metadata_cutoff_rule"] == (
        "cutoff dates must be on or before snapshot_date"
    )


def test_universe_service_accepts_reference_date_and_definition() -> None:
    provider = ConfiguredUniverseProvider(
        tickers=("AAPL", "MSFT", "SPY", "FUTURE"),
        constituent_metadata={
            "AAPL": {"market_cap_rank": 2, "as_of_date": "2026-01-02"},
            "MSFT": {"market_cap_rank": 1, "as_of_date": "2026-01-02"},
            "SPY": {"security_type": "etf", "as_of_date": "2026-01-02"},
            "FUTURE": {
                "market_cap_rank": 1,
                "market_cap_usd": 5_000_000_000_000,
                "as_of_date": "2026-01-03",
            },
        },
        source="unit_test_config",
    )
    service: UniverseService = ProviderBackedUniverseService(provider)
    request = UniverseConstructionRequest(
        experiment_id=" stage1_2026_01 ",
        as_of_date="2026-01-02T15:30:00",
        definition=DEFAULT_UNIVERSE_SELECTION_CONFIG,
    )

    snapshot = service.build_universe(request)
    constituents = service.list_constituents(request)

    assert request.experiment_id == "stage1_2026_01"
    assert request.as_of_date == date(2026, 1, 2)
    assert request.to_dict()["definition"]["config_id"] == "stage1_sp500_top150"
    assert snapshot.snapshot_date == date(2026, 1, 2)
    assert snapshot.tickers == ("MSFT", "AAPL")
    assert tuple(constituent.ticker for constituent in constituents) == ("MSFT", "AAPL")
    assert "SPY" not in snapshot.tickers
    assert "FUTURE" not in snapshot.tickers


def test_universe_construction_request_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="experiment_id"):
        UniverseConstructionRequest(experiment_id=" ", as_of_date=date(2026, 1, 2))

    with pytest.raises(ValueError, match="as_of_date must be an ISO date"):
        UniverseConstructionRequest(experiment_id="stage1", as_of_date="not-a-date")

    with pytest.raises(TypeError, match="definition"):
        UniverseConstructionRequest(
            experiment_id="stage1",
            as_of_date=date(2026, 1, 2),
            definition=object(),  # type: ignore[arg-type]
        )


def test_configured_universe_provider_applies_basic_include_exclude_rules() -> None:
    provider = ConfiguredUniverseProvider(
        tickers=("AAPL", "SPY", "QQQ", "OTCM", "ETF1", "OLD", "MSFT", "NVDA"),
        constituent_metadata={
            "AAPL": {
                "exchange": "NASDAQ",
                "security_type": "common stock",
                "listing_status": "ACTIVE",
            },
            "SPY": {"exchange": "NYSEARCA", "security_type": "etf", "listing_status": "active"},
            "QQQ": {"exchange": "NASDAQ", "security_type": "etf", "listing_status": "active"},
            "OTCM": {
                "exchange": "OTC",
                "security_type": "common_stock",
                "listing_status": "active",
            },
            "ETF1": {"exchange": "NYSE", "security_type": "ETF", "listing_status": "active"},
            "OLD": {
                "exchange": "NYSE",
                "security_type": "common_stock",
                "listing_status": "delisted",
            },
            "MSFT": {
                "exchange": "NASDAQ",
                "security_type": "common_stock",
                "listing_status": "active",
            },
            "NVDA": {
                "exchange": "NASDAQ",
                "security_type": "common_stock",
                "listing_status": "active",
            },
        },
    )

    snapshot = provider.get_universe_snapshot(
        experiment_id="stage1_2026_01",
        experiment_start=date(2026, 1, 2),
    )

    assert snapshot.tickers == ("AAPL", "MSFT", "NVDA")
    assert [constituent.market_cap_rank for constituent in snapshot.constituents] == [1, 2, 3]
    assert [constituent.security_type for constituent in snapshot.constituents] == [
        "common_stock",
        "common_stock",
        "common_stock",
    ]


def test_configured_universe_provider_filters_price_value_market_cap_and_liquidity() -> None:
    config = UniverseSelectionConfig(min_liquidity_score=5.0)
    provider = ConfiguredUniverseProvider(
        tickers=("BIG", "LOWP", "LOWADV", "SMALL", "LOWLIQ", "DERIVED", "KEEP2"),
        constituent_metadata={
            "BIG": {
                "market_cap_rank": 2,
                "market_cap_usd": 900_000_000_000,
                "price_usd": 150.0,
                "average_daily_dollar_volume_usd": 3_000_000_000,
                "liquidity_score": 20.0,
            },
            "LOWP": {
                "market_cap_usd": 20_000_000_000,
                "price_usd": 4.99,
                "average_daily_dollar_volume_usd": 100_000_000,
                "liquidity_score": 12.0,
            },
            "LOWADV": {
                "market_cap_usd": 20_000_000_000,
                "price_usd": 25.0,
                "average_daily_dollar_volume_usd": 24_999_999,
                "liquidity_score": 12.0,
            },
            "SMALL": {
                "market_cap_usd": 4_999_999_999,
                "price_usd": 25.0,
                "average_daily_dollar_volume_usd": 100_000_000,
                "liquidity_score": 12.0,
            },
            "LOWLIQ": {
                "market_cap_usd": 20_000_000_000,
                "price_usd": 25.0,
                "average_daily_dollar_volume_usd": 100_000_000,
                "liquidity_score": 4.99,
            },
            "DERIVED": {
                "market_cap_rank": 1,
                "market_cap_usd": 1_000_000_000_000,
                "price_usd": 50.0,
                "average_daily_volume": 1_000_000,
                "liquidity_score": 18.0,
            },
            "KEEP2": {
                "market_cap_usd": 7_000_000_000,
                "price_usd": 30.0,
                "average_daily_dollar_volume_usd": 30_000_000,
                "liquidity_score": 6.0,
            },
        },
    )

    snapshot = provider.get_universe_snapshot(
        experiment_id="stage1_2026_01",
        experiment_start=date(2026, 1, 2),
        config=config,
    )

    assert snapshot.tickers == ("DERIVED", "BIG", "KEEP2")
    assert snapshot.constituents[0].average_daily_dollar_volume_usd == 50_000_000.0
    assert [constituent.market_cap_rank for constituent in snapshot.constituents] == [1, 2, 3]
    assert snapshot.to_dict()["eligibility_rules"] == {
        "allowed_exchanges": ["NYSE", "NASDAQ"],
        "allowed_security_types": ["common_stock"],
        "allowed_listing_statuses": ["active"],
        "excluded_security_types": [
            "adr",
            "closed_end_fund",
            "etf",
            "fund",
            "preferred_stock",
            "reit_preferred",
            "unit",
            "warrant",
        ],
        "excluded_tickers": ["SPY", "QQQ", "DIA", "IWM"],
        "min_price_usd": 5.0,
        "min_average_daily_dollar_volume_usd": 25_000_000.0,
        "min_market_cap_usd": 5_000_000_000.0,
        "min_liquidity_score": 5.0,
    }


def test_configured_universe_provider_selects_top_150_by_experiment_start_market_cap() -> None:
    tickers = tuple(f"TICKER{index:03d}" for index in range(155))
    metadata = {
        ticker: {
            "market_cap_usd": 200_000_000_000 - index * 1_000_000_000,
            "price_usd": 25.0,
            "average_daily_dollar_volume_usd": 100_000_000,
            "as_of_date": "2026-01-02",
        }
        for index, ticker in enumerate(tickers)
    }
    provider = ConfiguredUniverseProvider(tickers=tickers, constituent_metadata=metadata)

    snapshot = provider.get_universe_snapshot(
        experiment_id="stage1_2026_01",
        experiment_start=date(2026, 1, 2),
    )

    assert snapshot.constituent_count == 150
    assert snapshot.selection_count == 150
    assert snapshot.tickers == tickers[:150]
    assert snapshot.tickers[-1] == "TICKER149"
    assert "TICKER150" not in snapshot.tickers
    assert [constituent.market_cap_rank for constituent in snapshot.constituents] == list(
        range(1, 151)
    )
    assert snapshot.snapshot_date == date(2026, 1, 2)
    assert snapshot.fixed_at_experiment_start is True
    assert snapshot.point_in_time_membership is False
    assert snapshot.survivorship_bias_allowed is True


def test_universe_provider_does_not_leak_future_available_metadata_into_snapshot() -> None:
    provider = ConfiguredUniverseProvider(
        tickers=("AAPL", "MSFT", "FUTUREGIANT"),
        constituent_metadata={
            "AAPL": {
                "market_cap_usd": 50_000_000_000,
                "price_usd": 150.0,
                "average_daily_dollar_volume_usd": 500_000_000,
                "available_at": "2026-01-02",
            },
            "MSFT": {
                "market_cap_usd": 60_000_000_000,
                "price_usd": 300.0,
                "average_daily_dollar_volume_usd": 600_000_000,
                "effective_date": date(2026, 1, 2),
            },
            "FUTUREGIANT": {
                "market_cap_rank": 1,
                "market_cap_usd": 5_000_000_000_000,
                "price_usd": 500.0,
                "average_daily_dollar_volume_usd": 10_000_000_000,
                "as_of_date": "2026-01-03",
            },
        },
    )

    snapshot = provider.get_universe_snapshot(
        experiment_id="stage1_2026_01",
        experiment_start=date(2026, 1, 2),
    )

    assert snapshot.tickers == ("MSFT", "AAPL")
    assert "FUTUREGIANT" not in snapshot.tickers
    assert snapshot.to_dict()["snapshot_date"] == "2026-01-02"
    assert snapshot.to_dict()["point_in_time_membership"] is False


def test_universe_provider_rejects_unparseable_metadata_availability_dates() -> None:
    provider = ConfiguredUniverseProvider(
        tickers=("AAPL",),
        constituent_metadata={"AAPL": {"available_at": "not-a-date"}},
    )

    with pytest.raises(ValueError, match="available_at must be an ISO date"):
        provider.get_universe_snapshot(
            experiment_id="stage1_2026_01",
            experiment_start=date(2026, 1, 2),
        )


def test_universe_snapshot_rejects_contract_violations() -> None:
    with pytest.raises(ValueError, match="benchmark_ticker"):
        UniverseSnapshot.from_tickers(
            ["AAPL", "SPY"],
            experiment_id="stage1_2026_01",
            snapshot_date=date(2026, 1, 2),
        )

    with pytest.raises(ValueError, match="point-in-time"):
        UniverseSnapshot(
            experiment_id="stage1_2026_01",
            snapshot_date=date(2026, 1, 2),
            constituents=(UniverseConstituent("AAPL", market_cap_rank=1),),
            point_in_time_membership=True,
        )

    with pytest.raises(ValueError, match="survivorship bias"):
        UniverseSnapshot(
            experiment_id="stage1_2026_01",
            snapshot_date=date(2026, 1, 2),
            constituents=(UniverseConstituent("AAPL", market_cap_rank=1),),
            survivorship_bias_disclosure="v1 allows current provider membership.",
        )

    with pytest.raises(ValueError, match="ineligible constituents"):
        UniverseSnapshot(
            experiment_id="stage1_2026_01",
            snapshot_date=date(2026, 1, 2),
            constituents=(UniverseConstituent("OTCM", exchange="OTC", market_cap_rank=1),),
        )

    with pytest.raises(ValueError, match="security_type etf"):
        UniverseSnapshot(
            experiment_id="stage1_2026_01",
            snapshot_date=date(2026, 1, 2),
            constituents=(
                UniverseConstituent(
                    "ETF1",
                    exchange="NYSE",
                    security_type="ETF",
                    market_cap_rank=1,
                ),
            ),
        )

    with pytest.raises(ValueError, match="listing_status delisted"):
        UniverseSnapshot(
            experiment_id="stage1_2026_01",
            snapshot_date=date(2026, 1, 2),
            constituents=(
                UniverseConstituent("OLD", listing_status="delisted", market_cap_rank=1),
            ),
        )

    with pytest.raises(ValueError, match="price_usd"):
        UniverseSnapshot(
            experiment_id="stage1_2026_01",
            snapshot_date=date(2026, 1, 2),
            constituents=(UniverseConstituent("PENNY", price_usd=4.99, market_cap_rank=1),),
        )

    with pytest.raises(ValueError, match="average_daily_dollar_volume_usd"):
        UniverseSnapshot(
            experiment_id="stage1_2026_01",
            snapshot_date=date(2026, 1, 2),
            constituents=(
                UniverseConstituent(
                    "THIN",
                    price_usd=10.0,
                    average_daily_volume=2_000_000,
                    market_cap_rank=1,
                ),
            ),
        )

    with pytest.raises(ValueError, match="market_cap_usd"):
        UniverseSnapshot(
            experiment_id="stage1_2026_01",
            snapshot_date=date(2026, 1, 2),
            constituents=(
                UniverseConstituent("SMALL", market_cap_usd=4_999_999_999, market_cap_rank=1),
            ),
        )

    with pytest.raises(ValueError, match="universe_name"):
        UniverseSnapshot(
            experiment_id="stage1_2026_01",
            snapshot_date=date(2026, 1, 2),
            universe_name=" ",
            constituents=(UniverseConstituent("AAPL", market_cap_rank=1),),
        )

    with pytest.raises(ValueError, match="constituent_identifier_fields"):
        UniverseSnapshot(
            experiment_id="stage1_2026_01",
            snapshot_date=date(2026, 1, 2),
            constituent_identifier_fields=("cik",),
            constituents=(UniverseConstituent("AAPL", market_cap_rank=1),),
        )

    with pytest.raises(ValueError, match="metadata_cutoff_fields"):
        UniverseSnapshot(
            experiment_id="stage1_2026_01",
            snapshot_date=date(2026, 1, 2),
            metadata_cutoff_fields=(),
            constituents=(UniverseConstituent("AAPL", market_cap_rank=1),),
        )


def test_universe_selection_config_rejects_non_canonical_stage1_rules() -> None:
    with pytest.raises(ValueError, match="selection_count must be 150"):
        UniverseSelectionConfig(selection_count=149)

    with pytest.raises(ValueError, match="point-in-time"):
        UniverseSelectionConfig(point_in_time_membership=True)

    with pytest.raises(ValueError, match="at least 3"):
        UniverseSelectionConfig(required_min_history_years=2)

    with pytest.raises(ValueError, match="allowed_exchanges"):
        UniverseSelectionConfig(allowed_exchanges=())

    with pytest.raises(ValueError, match="both allowed and excluded"):
        UniverseSelectionConfig(
            allowed_security_types=("common_stock",), excluded_security_types=("common stock",)
        )

    with pytest.raises(ValueError, match="benchmark ticker"):
        UniverseSelectionConfig(excluded_tickers=("QQQ",))

    with pytest.raises(ValueError, match="min_price_usd"):
        UniverseSelectionConfig(min_price_usd=-0.01)
