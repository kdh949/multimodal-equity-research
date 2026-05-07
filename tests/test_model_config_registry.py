from __future__ import annotations

from collections.abc import Iterable

from quant_research.validation import (
    DEFAULT_MODEL_COMPARISON_CANDIDATE_IDS,
    REQUIRED_FULL_MODEL_OPTIONAL_ADAPTERS,
    REQUIRED_STAGE1_ABLATION_MODEL_ADAPTERS,
    VALID_MODEL_COMPARISON_ADAPTERS,
    default_model_comparison_config,
    default_model_comparison_registry,
)


def test_default_model_config_registry_is_complete_for_stage1_adapters() -> None:
    registry = default_model_comparison_registry()
    config = default_model_comparison_config()

    assert registry.ids() == DEFAULT_MODEL_COMPARISON_CANDIDATE_IDS
    assert config.required_candidate_ids() == DEFAULT_MODEL_COMPARISON_CANDIDATE_IDS

    registered_adapters = {
        adapter
        for candidate in registry
        for adapter in candidate.adapters
    }
    named_ablation_adapters = {
        adapter
        for ablation_config in config.named_ablation_configs
        if ablation_config.contribution_kind == "model"
        for adapter in ablation_config.isolated_adapters
        if adapter != "rules_fallback"
    }

    assert set(REQUIRED_STAGE1_ABLATION_MODEL_ADAPTERS) <= registered_adapters
    assert set(REQUIRED_STAGE1_ABLATION_MODEL_ADAPTERS) == named_ablation_adapters
    assert set(REQUIRED_FULL_MODEL_OPTIONAL_ADAPTERS) <= registered_adapters
    assert registered_adapters <= set(VALID_MODEL_COMPARISON_ADAPTERS)


def test_default_model_config_registry_names_are_unique() -> None:
    config = default_model_comparison_config()
    registry = default_model_comparison_registry()

    assert_unique((candidate.candidate_id for candidate in registry), "candidate_id")
    assert_unique((candidate.label for candidate in registry), "candidate label")
    assert_unique(
        (model_config.config_id for model_config in config.canonical_model_configs),
        "canonical config_id",
    )
    assert_unique(
        (model_config.label for model_config in config.canonical_model_configs),
        "canonical label",
    )
    assert_unique(
        (ablation_config.config_id for ablation_config in config.named_ablation_configs),
        "named ablation config_id",
    )
    assert_unique(
        (ablation_config.label for ablation_config in config.named_ablation_configs),
        "named ablation label",
    )


def test_default_model_config_registry_keeps_heavy_adapters_optional_and_safe() -> None:
    config = default_model_comparison_config()
    optional_adapters = set(REQUIRED_FULL_MODEL_OPTIONAL_ADAPTERS)
    full_model_config = next(
        model_config
        for model_config in config.canonical_model_configs
        if model_config.role == "full_model"
    )

    assert config.allow_heavy_model_candidates is False
    assert set(full_model_config.optional_adapters) == optional_adapters
    assert optional_adapters <= set(full_model_config.adapters)
    assert "rules_fallback" in full_model_config.adapters
    assert "rules_fallback" in full_model_config.fallback_adapters

    assert all(candidate.requires_heavy_model is False for candidate in config.registry)
    assert all(
        model_config.requires_heavy_model is False
        for model_config in config.canonical_model_configs
    )
    assert all(
        ablation_config.requires_heavy_model is False
        for ablation_config in config.named_ablation_configs
    )
    assert all(
        set(model_config.optional_adapters) <= set(model_config.adapters)
        for model_config in config.canonical_model_configs
    )
    assert all(
        set(model_config.fallback_adapters) <= set(model_config.adapters)
        for model_config in config.canonical_model_configs
    )


def assert_unique(values: Iterable[str], label: str) -> None:
    ordered = tuple(values)
    duplicates = sorted({value for value in ordered if ordered.count(value) > 1})
    assert duplicates == [], f"duplicate {label}: {duplicates}"
