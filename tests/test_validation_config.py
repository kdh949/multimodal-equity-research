from __future__ import annotations

import pytest

from quant_research.pipeline import PipelineConfig
from quant_research.validation import (
    CANONICAL_BASELINE_MODEL_CONFIG,
    CANONICAL_DETERMINISTIC_VALIDATION_DEFAULTS,
    CANONICAL_FULL_MODEL_CONFIG,
    CANONICAL_MIN_POSITIVE_FOLD_RATIO,
    CANONICAL_MODEL_CONFIG_IDS,
    CANONICAL_MODEL_CONFIGS,
    CANONICAL_STRUCTURED_TEXT_FEATURES,
    DEFAULT_COVARIANCE_AWARE_RISK_CONFIG,
    DEFAULT_MODEL_COMPARISON_CANDIDATE_IDS,
    DEFAULT_MODEL_COMPARISON_METRICS,
    DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG,
    DEFAULT_RISK_CONSTRAINT_ADJUSTMENTS,
    DEFAULT_STAGE1_OUTPERFORMANCE_MIN_DELTAS,
    DEFAULT_STAGE1_OUTPERFORMANCE_THRESHOLDS,
    DEFAULT_TRANSACTION_COST_SENSITIVITY_CONFIG,
    DEFAULT_TRANSACTION_COST_SENSITIVITY_SCENARIO_IDS,
    DEFAULT_TRANSACTION_COST_SENSITIVITY_SCENARIOS,
    DETERMINISTIC_SIGNAL_ENGINE_ID,
    PORTFOLIO_RISK_CONSTRAINT_SCHEMA_VERSION,
    REQUIRED_STAGE1_ABLATION_MODALITIES,
    REQUIRED_STAGE1_ABLATION_MODEL_ADAPTERS,
    REQUIRED_VALIDATION_HORIZON_DAYS,
    STAGE1_NAMED_ABLATION_CONFIG_IDS,
    STAGE1_NAMED_ABLATION_CONFIGS,
    CanonicalModelConfiguration,
    CovarianceAwareRiskConfig,
    DeterministicValidationDefaults,
    ModelComparisonCandidateConfig,
    ModelComparisonConfig,
    NamedAblationConfiguration,
    PortfolioRiskConstraintConfig,
    RiskConstraintAdjustmentConfig,
    Stage1OutperformanceThresholds,
    TransactionCostSensitivityConfig,
    TransactionCostSensitivityScenario,
    ValidationGateThresholds,
    default_canonical_model_configs,
    default_deterministic_validation_defaults,
    default_model_comparison_config,
    default_model_comparison_registry,
    default_named_ablation_configs,
    default_stage1_model_configs,
    default_transaction_cost_sensitivity_config,
)


def test_default_model_comparison_config_is_stage1_typed_contract() -> None:
    config = default_model_comparison_config()

    assert config.config_id == "stage1_validity_gate"
    assert config.required_horizon_days == REQUIRED_VALIDATION_HORIZON_DAYS
    assert config.primary_candidate_id == "all_features"
    assert config.baseline_candidate_id == "no_model_proxy"
    assert config.full_model_candidate_id == "all_features"
    assert config.metrics == DEFAULT_MODEL_COMPARISON_METRICS
    assert config.validation_defaults == CANONICAL_DETERMINISTIC_VALIDATION_DEFAULTS
    assert config.outperformance_thresholds == DEFAULT_STAGE1_OUTPERFORMANCE_THRESHOLDS
    assert config.canonical_model_configs == CANONICAL_MODEL_CONFIGS
    assert config.named_ablation_configs == STAGE1_NAMED_ABLATION_CONFIGS
    assert config.require_cost_adjusted_metrics is True
    assert config.allow_heavy_model_candidates is False
    assert config.registry.ids() == DEFAULT_MODEL_COMPARISON_CANDIDATE_IDS
    assert default_model_comparison_registry().ids() == DEFAULT_MODEL_COMPARISON_CANDIDATE_IDS
    assert config.ablation_scenario_ids() == DEFAULT_MODEL_COMPARISON_CANDIDATE_IDS
    assert config.required_candidate_ids() == DEFAULT_MODEL_COMPARISON_CANDIDATE_IDS

    payload = config.to_dict()
    assert payload["config_id"] == "stage1_validity_gate"
    assert payload["required_horizon_days"] == 20
    assert payload["primary_candidate_id"] == "all_features"
    assert payload["baseline_candidate_id"] == "no_model_proxy"
    assert payload["full_model_candidate_id"] == "all_features"
    assert payload["metrics"] == list(DEFAULT_MODEL_COMPARISON_METRICS)
    assert payload["required_candidate_ids"] == list(DEFAULT_MODEL_COMPARISON_CANDIDATE_IDS)
    assert payload["canonical_model_configs"][0]["config_id"] == "baseline_no_model_proxy"
    assert payload["canonical_model_configs"][1]["config_id"] == "full_multimodal_model"
    assert [row["config_id"] for row in payload["named_ablation_configs"]] == list(
        STAGE1_NAMED_ABLATION_CONFIG_IDS
    )
    assert payload["validation_defaults"]["prediction_target_column"] == "forward_return_20"
    assert payload["validation_defaults"]["min_positive_fold_ratio"] == 0.65
    assert CANONICAL_MIN_POSITIVE_FOLD_RATIO == 0.65
    assert ValidationGateThresholds().min_positive_fold_ratio == 0.65
    assert payload["outperformance_thresholds"]["minimum_metric_improvements"] == (
        DEFAULT_STAGE1_OUTPERFORMANCE_MIN_DELTAS
    )
    assert payload["outperformance_thresholds"]["require_all_configured_baselines"] is True
    assert payload["outperformance_thresholds"]["require_all_configured_windows"] is True
    assert payload["validation_defaults"]["signal_engine"] == DETERMINISTIC_SIGNAL_ENGINE_ID
    assert payload["validation_defaults"]["model_predictions_are_order_signals"] is False
    assert payload["validation_defaults"]["llm_makes_trading_decisions"] is False
    assert all(candidate["requires_heavy_model"] is False for candidate in payload["candidates"])
    assert payload["candidates"][0]["adapters"] == [
        "tabular",
        "chronos",
        "granite_ttm",
        "finbert",
        "finma",
        "fingpt",
        "ollama",
    ]


def test_model_comparison_config_groups_candidates_by_role() -> None:
    registry = default_model_comparison_registry()

    assert tuple(candidate.candidate_id for candidate in registry.by_role("primary")) == (
        "all_features",
    )
    assert tuple(candidate.candidate_id for candidate in registry.by_role("baseline")) == (
        "no_model_proxy",
    )
    assert tuple(candidate.candidate_id for candidate in registry.by_role("ablation")) == (
        "price_only",
        "text_only",
        "sec_only",
    )
    assert tuple(candidate.candidate_id for candidate in registry.by_role("diagnostic")) == (
        "no_costs",
    )
    assert registry.get("text_only").feature_families == ("text",)


def test_stage1_outperformance_thresholds_are_configurable_and_validated() -> None:
    thresholds = Stage1OutperformanceThresholds(
        minimum_metric_improvements={"sharpe": 0.10, "turnover": 0.02},
        require_all_configured_baselines=False,
        require_all_configured_windows=False,
    )
    config = ModelComparisonConfig(
        candidates=default_model_comparison_config().candidates,
        outperformance_thresholds=thresholds,
    )

    assert config.outperformance_thresholds.threshold_for("sharpe") == 0.10
    assert config.outperformance_thresholds.threshold_for("rank_ic") == 0.0
    assert config.to_dict()["outperformance_thresholds"] == {
        "minimum_metric_improvements": {"sharpe": 0.10, "turnover": 0.02},
        "require_all_configured_baselines": False,
        "require_all_configured_windows": False,
    }

    with pytest.raises(ValueError, match="unsupported outperformance threshold metrics"):
        Stage1OutperformanceThresholds(
            minimum_metric_improvements={"unknown_metric": 0.01}  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="must be non-negative"):
        Stage1OutperformanceThresholds(
            minimum_metric_improvements={"sharpe": -0.01}
        )


def test_portfolio_risk_constraint_config_schema_is_configurable() -> None:
    config = PortfolioRiskConstraintConfig(
        max_holdings=12,
        max_symbol_weight=0.08,
        max_sector_weight=0.25,
        max_position_risk_contribution=0.18,
        portfolio_volatility_limit=0.03,
        portfolio_covariance_lookback=42,
        covariance_aware_risk=CovarianceAwareRiskConfig(
            enabled=True,
            return_column="return_1",
            lookback_periods=42,
            min_periods=10,
        ),
        max_drawdown_stop=0.15,
        adjustment=RiskConstraintAdjustmentConfig(
            volatility_scale_strength=0.75,
            concentration_scale_strength=0.50,
            risk_contribution_scale_strength=0.25,
        ),
    )

    payload = config.to_dict()
    assert payload["schema_version"] == PORTFOLIO_RISK_CONSTRAINT_SCHEMA_VERSION
    assert payload["max_holdings"] == 12
    assert payload["max_symbol_weight"] == 0.08
    assert payload["max_sector_weight"] == 0.25
    assert payload["max_position_risk_contribution"] == 0.18
    assert payload["portfolio_volatility_limit"] == 0.03
    assert payload["covariance_aware_risk"] == {
        "enabled": True,
        "return_column": "return_1",
        "lookback_periods": 42,
        "min_periods": 10,
        "fallback": "diagonal_predicted_volatility",
    }
    assert payload["adjustment"] == {
        "volatility_scale_strength": 0.75,
        "concentration_scale_strength": 0.50,
        "risk_contribution_scale_strength": 0.25,
    }
    assert "correlation_cluster_weight" in payload["v1_exclusions"]
    assert config.to_backtest_kwargs()["top_n"] == 12
    assert config.to_backtest_kwargs()["max_sector_weight"] == 0.25
    assert config.to_backtest_kwargs()["covariance_aware_risk_enabled"] is True
    assert config.to_backtest_kwargs()["covariance_min_periods"] == 10


def test_default_portfolio_risk_constraint_config_matches_stage1_limits() -> None:
    config = DEFAULT_PORTFOLIO_RISK_CONSTRAINT_CONFIG

    assert config.schema_version == PORTFOLIO_RISK_CONSTRAINT_SCHEMA_VERSION
    assert config.max_holdings == 20
    assert config.max_symbol_weight == 0.10
    assert config.max_sector_weight == 0.30
    assert config.max_drawdown_stop == 0.20
    assert config.covariance_aware_risk == DEFAULT_COVARIANCE_AWARE_RISK_CONFIG
    assert config.covariance_aware_risk.enabled is True
    assert config.covariance_aware_risk.return_column == "return_1"
    assert config.covariance_aware_risk.lookback_periods == config.portfolio_covariance_lookback
    assert config.covariance_aware_risk.min_periods == 20
    assert config.adjustment == DEFAULT_RISK_CONSTRAINT_ADJUSTMENTS
    assert "correlation_cluster_weight" in config.v1_exclusions

    pipeline_config = PipelineConfig()
    assert pipeline_config.portfolio_risk_constraint_config == config
    assert pipeline_config.top_n == config.max_holdings
    assert pipeline_config.max_symbol_weight == config.max_symbol_weight
    assert pipeline_config.max_sector_weight == config.max_sector_weight
    assert pipeline_config.covariance_aware_risk_enabled == config.covariance_aware_risk.enabled
    assert pipeline_config.covariance_return_column == config.covariance_aware_risk.return_column
    assert pipeline_config.covariance_min_periods == config.covariance_aware_risk.min_periods
    assert pipeline_config.portfolio_volatility_limit == config.portfolio_volatility_limit


def test_portfolio_risk_constraint_config_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="max_symbol_weight"):
        PortfolioRiskConstraintConfig(max_symbol_weight=0.0)

    with pytest.raises(ValueError, match="volatility_scale_strength"):
        RiskConstraintAdjustmentConfig(volatility_scale_strength=1.5)

    with pytest.raises(ValueError, match="correlation_cluster_weight"):
        PortfolioRiskConstraintConfig(v1_exclusions=())

    with pytest.raises(ValueError, match="covariance min_periods"):
        CovarianceAwareRiskConfig(min_periods=1)

    with pytest.raises(ValueError, match="portfolio_covariance_lookback"):
        PortfolioRiskConstraintConfig(
            portfolio_covariance_lookback=60,
            covariance_aware_risk=CovarianceAwareRiskConfig(lookback_periods=40),
        )


def test_stage1_named_ablation_configs_isolate_each_model_and_modality() -> None:
    configs = default_named_ablation_configs()

    assert configs == STAGE1_NAMED_ABLATION_CONFIGS
    assert tuple(config.config_id for config in configs) == STAGE1_NAMED_ABLATION_CONFIG_IDS
    assert STAGE1_NAMED_ABLATION_CONFIG_IDS == (
        "price_modality",
        "news_text_modality",
        "sec_filing_modality",
        "time_series_modality",
        "filing_text_modality",
        "tabular_model",
        "chronos_model",
        "granite_ttm_model",
        "finbert_model",
        "finma_model",
        "fingpt_model",
        "ollama_model",
    )

    model_configs = [
        config for config in configs if config.contribution_kind == "model"
    ]
    modality_configs = [
        config for config in configs if config.contribution_kind == "modality"
    ]
    covered_adapters = {
        adapter
        for config in model_configs
        for adapter in config.isolated_adapters
        if adapter != "rules_fallback"
    }
    covered_modalities = {
        modality
        for config in modality_configs
        for modality in config.isolated_modalities
    }

    assert covered_adapters == set(REQUIRED_STAGE1_ABLATION_MODEL_ADAPTERS)
    assert covered_modalities == set(REQUIRED_STAGE1_ABLATION_MODALITIES)
    assert all(
        config.validation_defaults == CANONICAL_DETERMINISTIC_VALIDATION_DEFAULTS
        for config in configs
    )
    assert all(config.requires_heavy_model is False for config in configs)
    assert all(
        config.validation_defaults.to_dict()
        == CANONICAL_DETERMINISTIC_VALIDATION_DEFAULTS.to_dict()
        for config in configs
    )
    assert {
        config.config_id: config.ablation_scenario_id
        for config in configs
        if config.config_id in {"chronos_model", "granite_ttm_model", "tabular_model"}
    } == {
        "chronos_model": "no_chronos_features",
        "granite_ttm_model": "no_granite_features",
        "tabular_model": "tabular_without_ts_proxies",
    }

    payloads = [config.to_dict() for config in configs]
    assert all(payload["validation_defaults"]["prediction_target_column"] == "forward_return_20" for payload in payloads)
    assert all(payload["validation_defaults"]["signal_engine"] == DETERMINISTIC_SIGNAL_ENGINE_ID for payload in payloads)
    assert next(
        payload for payload in payloads if payload["config_id"] == "ollama_model"
    )["affects_signal_engine"] is False


def test_named_ablation_configs_reject_changed_validation_semantics() -> None:
    with pytest.raises(ValueError, match="share Stage 1 validation semantics"):
        NamedAblationConfiguration(
            config_id="changed_semantics",
            label="Changed semantics",
            contribution_kind="model",
            comparison_method="adapter_override",
            ablation_scenario_id="all_features",
            isolated_adapters=("tabular",),
            isolated_modalities=("price",),
            feature_families=("price",),
            validation_defaults=DeterministicValidationDefaults(train_periods=120),
        )

    with pytest.raises(ValueError, match="heavy model adapters optional"):
        NamedAblationConfiguration(
            config_id="heavy_chronos",
            label="Heavy Chronos",
            contribution_kind="model",
            comparison_method="adapter_override",
            ablation_scenario_id="no_chronos_features",
            isolated_adapters=("chronos",),
            isolated_modalities=("time_series",),
            feature_families=("chronos",),
            requires_heavy_model=True,
        )


def test_model_comparison_config_requires_named_ablation_coverage() -> None:
    base = default_model_comparison_config()
    missing_ollama = tuple(
        config for config in base.named_ablation_configs if config.config_id != "ollama_model"
    )

    with pytest.raises(ValueError, match="adapter coverage"):
        ModelComparisonConfig(
            candidates=base.candidates,
            named_ablation_configs=missing_ollama,
        )


def test_model_comparison_candidate_config_validates_contract() -> None:
    with pytest.raises(ValueError, match="stable snake_case"):
        ModelComparisonCandidateConfig(
            candidate_id="Bad-ID",
            label="Bad",
            role="primary",
            adapters=("tabular",),
        )

    with pytest.raises(ValueError, match="ablation_scenario_id"):
        ModelComparisonCandidateConfig(
            candidate_id="missing_ablation",
            label="Missing ablation scenario",
            role="ablation",
            adapters=("tabular",),
        )

    with pytest.raises(ValueError, match="duplicate adapter"):
        ModelComparisonCandidateConfig(
            candidate_id="duplicate_adapters",
            label="Duplicate adapters",
            role="primary",
            adapters=("tabular", "tabular"),
        )


def test_model_comparison_config_rejects_heavy_candidates_unless_enabled() -> None:
    heavy_candidate = ModelComparisonCandidateConfig(
        candidate_id="chronos_local",
        label="Chronos local inference",
        role="diagnostic",
        adapters=("chronos",),
        feature_families=("chronos",),
        requires_heavy_model=True,
    )

    with pytest.raises(ValueError, match="allow_heavy_model_candidates=True"):
        ModelComparisonConfig(
            primary_candidate_id="chronos_local",
            candidates=(heavy_candidate,),
        )

    config = ModelComparisonConfig(
        primary_candidate_id="chronos_local",
        candidates=(heavy_candidate,),
        allow_heavy_model_candidates=True,
    )
    assert config.registry.get("chronos_local").requires_heavy_model is True


def test_pipeline_config_carries_default_model_comparison_config() -> None:
    config = PipelineConfig()

    assert config.model_comparison_config.required_horizon_days == 20
    assert config.model_comparison_config.primary_candidate_id == "all_features"
    assert config.model_comparison_config.baseline_candidate_id == "no_model_proxy"
    assert config.model_comparison_config.registry.ids() == DEFAULT_MODEL_COMPARISON_CANDIDATE_IDS


def test_canonical_stage1_model_configs_cover_baseline_full_model_and_defaults() -> None:
    configs = default_canonical_model_configs()

    assert configs == CANONICAL_MODEL_CONFIGS
    assert default_stage1_model_configs() == configs
    assert tuple(config.config_id for config in configs) == CANONICAL_MODEL_CONFIG_IDS
    assert CANONICAL_BASELINE_MODEL_CONFIG in configs
    assert CANONICAL_FULL_MODEL_CONFIG in configs

    baseline = CANONICAL_BASELINE_MODEL_CONFIG
    assert baseline.role == "baseline"
    assert baseline.comparison_candidate_id == "no_model_proxy"
    assert baseline.modalities == ("price", "news_text", "sec_filing", "filing_text")
    assert baseline.feature_families == ("price", "text", "sec")
    assert baseline.fallback_adapters == ("rules_fallback",)
    assert baseline.requires_heavy_model is False

    full_model = CANONICAL_FULL_MODEL_CONFIG
    assert full_model.role == "full_model"
    assert full_model.comparison_candidate_id == "all_features"
    assert full_model.feature_families == ("price", "text", "sec", "chronos", "granite_ttm")
    assert {
        "chronos",
        "granite_ttm",
        "finbert",
        "finma",
        "fingpt",
        "ollama",
    }.issubset(full_model.optional_adapters)
    assert full_model.fallback_adapters == ("rules_fallback",)
    assert full_model.structured_text_features == CANONICAL_STRUCTURED_TEXT_FEATURES
    assert full_model.requires_heavy_model is False


def test_deterministic_validation_defaults_encode_stage1_contract() -> None:
    defaults = default_deterministic_validation_defaults()

    assert defaults.required_validation_horizon_days == 20
    assert defaults.validation_horizons == (1, 5, 20)
    assert defaults.prediction_target_column == "forward_return_20"
    assert defaults.gap_periods == 60
    assert defaults.embargo_periods == 60
    assert defaults.cost_bps == 5.0
    assert defaults.slippage_bps == 2.0
    assert defaults.signal_engine == DETERMINISTIC_SIGNAL_ENGINE_ID
    assert defaults.model_predictions_are_order_signals is False
    assert defaults.llm_makes_trading_decisions is False
    assert "t_plus_1" in defaults.return_timing

    payload = defaults.to_dict()
    assert payload["validation_horizons"] == [1, 5, 20]
    assert payload["benchmark_ticker"] == "SPY"


def test_default_transaction_cost_sensitivity_config_defines_stage1_scenarios() -> None:
    config = default_transaction_cost_sensitivity_config()

    assert config == DEFAULT_TRANSACTION_COST_SENSITIVITY_CONFIG
    assert config.config_id == "stage1_cost_turnover_sensitivity"
    assert config.baseline_scenario_id == "canonical_costs"
    assert config.scenario_ids() == DEFAULT_TRANSACTION_COST_SENSITIVITY_SCENARIO_IDS
    assert config.scenarios == DEFAULT_TRANSACTION_COST_SENSITIVITY_SCENARIOS
    assert config.scenario_ids() == (
        "canonical_costs",
        "no_costs",
        "low_costs",
        "high_costs",
        "tight_turnover_budget",
        "loose_turnover_budget",
    )

    baseline = config.get("canonical_costs")
    assert baseline.cost_bps == 5.0
    assert baseline.slippage_bps == 2.0
    assert baseline.total_cost_bps == 7.0
    assert baseline.average_daily_turnover_budget == 0.25
    assert baseline.max_daily_turnover is None

    no_costs = config.get("no_costs")
    assert no_costs.cost_bps == 0.0
    assert no_costs.slippage_bps == 0.0
    assert no_costs.average_daily_turnover_budget == 0.25

    assert config.get("low_costs").total_cost_bps < baseline.total_cost_bps
    assert config.get("high_costs").total_cost_bps > baseline.total_cost_bps
    assert config.get("tight_turnover_budget").average_daily_turnover_budget == 0.15
    assert config.get("loose_turnover_budget").max_daily_turnover == 1.0

    payload = config.to_dict()
    assert payload["baseline_scenario_id"] == "canonical_costs"
    assert payload["scenario_ids"] == list(DEFAULT_TRANSACTION_COST_SENSITIVITY_SCENARIO_IDS)
    assert payload["scenarios"][0] == {
        "scenario_id": "canonical_costs",
        "label": "Canonical costs",
        "cost_bps": 5.0,
        "slippage_bps": 2.0,
        "total_cost_bps": 7.0,
        "average_daily_turnover_budget": 0.25,
        "max_daily_turnover": None,
        "description": baseline.description,
    }


def test_pipeline_config_carries_default_transaction_cost_sensitivity_config() -> None:
    config = PipelineConfig()

    assert (
        config.transaction_cost_sensitivity_config
        == DEFAULT_TRANSACTION_COST_SENSITIVITY_CONFIG
    )
    assert config.transaction_cost_sensitivity_config.baseline_scenario_id == "canonical_costs"
    assert config.transaction_cost_sensitivity_config.get("canonical_costs").cost_bps == (
        config.cost_bps
    )
    assert config.transaction_cost_sensitivity_config.get(
        "canonical_costs"
    ).slippage_bps == config.slippage_bps
    assert config.transaction_cost_sensitivity_config.get(
        "canonical_costs"
    ).average_daily_turnover_budget == config.average_daily_turnover_budget


def test_transaction_cost_sensitivity_config_validates_schema() -> None:
    with pytest.raises(ValueError, match="scenario_id must be stable snake_case"):
        TransactionCostSensitivityScenario(
            scenario_id="Bad Scenario",
            label="Bad",
            cost_bps=5.0,
            slippage_bps=2.0,
            average_daily_turnover_budget=0.25,
        )

    with pytest.raises(ValueError, match="cost_bps must be non-negative"):
        TransactionCostSensitivityScenario(
            scenario_id="bad_cost",
            label="Bad cost",
            cost_bps=-0.1,
            slippage_bps=2.0,
            average_daily_turnover_budget=0.25,
        )

    with pytest.raises(ValueError, match="slippage_bps must be non-negative"):
        TransactionCostSensitivityScenario(
            scenario_id="bad_slippage",
            label="Bad slippage",
            cost_bps=5.0,
            slippage_bps=-0.1,
            average_daily_turnover_budget=0.25,
        )

    with pytest.raises(ValueError, match="average_daily_turnover_budget"):
        TransactionCostSensitivityScenario(
            scenario_id="bad_turnover_budget",
            label="Bad turnover budget",
            cost_bps=5.0,
            slippage_bps=2.0,
            average_daily_turnover_budget=0.0,
        )

    with pytest.raises(ValueError, match="max_daily_turnover"):
        TransactionCostSensitivityScenario(
            scenario_id="bad_max_turnover",
            label="Bad max turnover",
            cost_bps=5.0,
            slippage_bps=2.0,
            average_daily_turnover_budget=0.25,
            max_daily_turnover=2.01,
        )

    with pytest.raises(ValueError, match="duplicate transaction cost sensitivity"):
        TransactionCostSensitivityConfig(
            scenarios=(
                TransactionCostSensitivityScenario(
                    scenario_id="canonical_costs",
                    label="Canonical costs",
                    cost_bps=5.0,
                    slippage_bps=2.0,
                    average_daily_turnover_budget=0.25,
                ),
                TransactionCostSensitivityScenario(
                    scenario_id="canonical_costs",
                    label="Duplicate canonical costs",
                    cost_bps=5.0,
                    slippage_bps=2.0,
                    average_daily_turnover_budget=0.25,
                ),
            )
        )

    with pytest.raises(ValueError, match="baseline_scenario_id"):
        TransactionCostSensitivityConfig(
            baseline_scenario_id="missing",
            scenarios=(
                TransactionCostSensitivityScenario(
                    scenario_id="canonical_costs",
                    label="Canonical costs",
                    cost_bps=5.0,
                    slippage_bps=2.0,
                    average_daily_turnover_budget=0.25,
                ),
            ),
        )

    with pytest.raises(ValueError, match="must match canonical cost"):
        TransactionCostSensitivityConfig(
            scenarios=(
                TransactionCostSensitivityScenario(
                    scenario_id="canonical_costs",
                    label="Changed canonical costs",
                    cost_bps=6.0,
                    slippage_bps=2.0,
                    average_daily_turnover_budget=0.25,
                ),
            )
        )


def test_canonical_model_config_rejects_unsafe_defaults() -> None:
    with pytest.raises(ValueError, match="LLM adapters must not make trading decisions"):
        DeterministicValidationDefaults(llm_makes_trading_decisions=True)

    with pytest.raises(ValueError, match="order signals"):
        DeterministicValidationDefaults(model_predictions_are_order_signals=True)

    with pytest.raises(ValueError, match="full_model config missing optional adapters"):
        CanonicalModelConfiguration(
            config_id="bad_full_model",
            label="Bad full model",
            role="full_model",
            comparison_candidate_id="all_features",
            modalities=("price", "time_series"),
            feature_families=("price", "chronos"),
            adapters=("tabular", "chronos", "rules_fallback"),
            optional_adapters=("chronos",),
        )
