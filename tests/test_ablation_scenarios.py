from __future__ import annotations

import pytest

from quant_research.validation import (
    DATA_CHANNEL_FEATURE_FAMILY_ALLOWLISTS,
    DEFAULT_ABLATION_SCENARIO_IDS,
    DEFAULT_VALIDITY_GATE_ABLATION_MODE_IDS,
    NO_COST_ABLATION_SCENARIO,
    PRICE_ONLY_FEATURE_FAMILY_ALLOWLIST,
    SEC_ONLY_FEATURE_FAMILY_ALLOWLIST,
    TEXT_ONLY_FEATURE_FAMILY_ALLOWLIST,
    VALID_ABLATION_SCENARIO_KINDS,
    VALID_FEATURE_FAMILIES,
    VALIDITY_GATE_ABLATION_MODE_IDS,
    AblationScenarioConfig,
    AblationScenarioRegistry,
    AblationToggles,
    default_ablation_registry,
    normalize_validity_gate_ablation_mode_ids,
)


def test_default_ablation_scenario_ids_are_stable_and_ordered() -> None:
    assert DEFAULT_ABLATION_SCENARIO_IDS == (
        "all_features",
        "no_text_risk",
        "no_sec_risk",
        "no_model_proxy",
        "no_costs",
        "price_only",
        "text_only",
        "sec_only",
        "full_model_features",
        "no_chronos_features",
        "no_granite_features",
        "tabular_without_ts_proxies",
    )
    assert default_ablation_registry().ids() == DEFAULT_ABLATION_SCENARIO_IDS


def test_default_validity_gate_ablation_modes_preserve_stage1_default_contract() -> None:
    assert DEFAULT_VALIDITY_GATE_ABLATION_MODE_IDS == (
        "all_features",
        "no_text_risk",
        "no_sec_risk",
        "no_model_proxy",
        "no_costs",
        "price_only",
        "text_only",
        "sec_only",
    )
    assert set(DEFAULT_VALIDITY_GATE_ABLATION_MODE_IDS).issubset(VALIDITY_GATE_ABLATION_MODE_IDS)


def test_validity_gate_ablation_mode_ids_normalize_no_model_proxy_alias() -> None:
    assert normalize_validity_gate_ablation_mode_ids(("no-model-proxy",)) == ("no_model_proxy",)
    assert normalize_validity_gate_ablation_mode_ids(("no-cost",)) == (
        NO_COST_ABLATION_SCENARIO,
    )
    assert normalize_validity_gate_ablation_mode_ids("default") == DEFAULT_VALIDITY_GATE_ABLATION_MODE_IDS
    assert normalize_validity_gate_ablation_mode_ids(
        ("default", "no-model-proxy", "no_model_proxy")
    ) == DEFAULT_VALIDITY_GATE_ABLATION_MODE_IDS

    with pytest.raises(ValueError, match="unsupported validity gate ablation mode"):
        normalize_validity_gate_ablation_mode_ids(("unknown_mode",))


def test_default_ablation_scenario_names_and_kinds_are_stable() -> None:
    registry = default_ablation_registry()

    assert tuple((scenario.scenario_id, scenario.kind, scenario.label) for scenario in registry) == (
        ("all_features", "signal", "All signal features"),
        ("no_text_risk", "signal", "No text risk"),
        ("no_sec_risk", "signal", "No SEC risk"),
        ("no_model_proxy", "pipeline_control", "No model proxy"),
        (NO_COST_ABLATION_SCENARIO, "cost", "No transaction costs"),
        ("price_only", "data_channel", "Price only"),
        ("text_only", "data_channel", "Text only"),
        ("sec_only", "data_channel", "SEC only"),
        ("full_model_features", "model_feature", "Full model features"),
        ("no_chronos_features", "model_feature", "No Chronos features"),
        ("no_granite_features", "model_feature", "No Granite TTM features"),
        (
            "tabular_without_ts_proxies",
            "model_feature",
            "Tabular without time-series proxies",
        ),
    )


def test_default_ablation_registry_groups_scenarios_by_execution_kind() -> None:
    registry = default_ablation_registry()

    assert tuple(scenario.scenario_id for scenario in registry.by_kind("signal")) == (
        "all_features",
        "no_text_risk",
        "no_sec_risk",
    )
    assert tuple(scenario.scenario_id for scenario in registry.by_kind("pipeline_control")) == (
        "no_model_proxy",
    )
    assert tuple(scenario.scenario_id for scenario in registry.by_kind("cost")) == (
        NO_COST_ABLATION_SCENARIO,
    )
    assert tuple(scenario.scenario_id for scenario in registry.by_kind("data_channel")) == (
        "price_only",
        "text_only",
        "sec_only",
    )
    assert tuple(scenario.scenario_id for scenario in registry.by_kind("model_feature")) == (
        "full_model_features",
        "no_chronos_features",
        "no_granite_features",
        "tabular_without_ts_proxies",
    )


def test_ablation_toggles_are_explicit_deterministic_bool_fields() -> None:
    expected_toggle_fields = (
        "include_price_features",
        "include_text_features",
        "include_sec_features",
        "include_text_risk",
        "include_sec_risk",
        "include_model_proxy_features",
        "include_proxy_features",
        "include_proxy_model_inputs",
        "include_chronos_features",
        "include_granite_ttm_features",
        "include_transaction_costs",
        "include_slippage",
        "include_turnover_costs",
    )

    for scenario in default_ablation_registry():
        toggle_payload = scenario.toggles.to_dict()
        assert tuple(toggle_payload) == expected_toggle_fields
        assert all(type(value) is bool for value in toggle_payload.values())

    assert default_ablation_registry().get("no_text_risk").toggles.include_text_risk is False
    assert default_ablation_registry().get("no_sec_risk").toggles.include_sec_risk is False
    assert default_ablation_registry().get("no_model_proxy").toggles.include_model_proxy_features is False
    assert default_ablation_registry().get("no_model_proxy").toggles.include_proxy_features is False
    assert (
        default_ablation_registry().get("no_model_proxy").toggles.include_proxy_model_inputs
        is False
    )
    assert default_ablation_registry().get("no_costs").toggles.include_transaction_costs is False
    assert default_ablation_registry().get("no_costs").toggles.include_slippage is False
    assert default_ablation_registry().get("no_costs").toggles.include_turnover_costs is False
    assert (
        default_ablation_registry().get("tabular_without_ts_proxies").toggles.include_chronos_features
        is False
    )
    assert (
        default_ablation_registry()
        .get("tabular_without_ts_proxies")
        .toggles.include_granite_ttm_features
        is False
    )


def test_default_ablation_scenarios_have_expected_toggle_values() -> None:
    true_toggles = AblationToggles().to_dict()
    registry = default_ablation_registry()

    expected_by_scenario = {
        "all_features": true_toggles,
        "no_text_risk": true_toggles | {"include_text_risk": False},
        "no_sec_risk": true_toggles | {"include_sec_risk": False},
        "no_model_proxy": true_toggles
        | {
            "include_model_proxy_features": False,
            "include_proxy_features": False,
            "include_proxy_model_inputs": False,
            "include_chronos_features": False,
            "include_granite_ttm_features": False,
        },
        "no_costs": true_toggles
        | {
            "include_transaction_costs": False,
            "include_slippage": False,
            "include_turnover_costs": False,
        },
        "price_only": true_toggles
        | {
            "include_text_features": False,
            "include_sec_features": False,
            "include_text_risk": False,
            "include_sec_risk": False,
        },
        "text_only": true_toggles
        | {
            "include_price_features": False,
            "include_sec_features": False,
            "include_sec_risk": False,
            "include_chronos_features": False,
            "include_granite_ttm_features": False,
        },
        "sec_only": true_toggles
        | {
            "include_price_features": False,
            "include_text_features": False,
            "include_text_risk": False,
            "include_chronos_features": False,
            "include_granite_ttm_features": False,
        },
        "full_model_features": true_toggles,
        "no_chronos_features": true_toggles | {"include_chronos_features": False},
        "no_granite_features": true_toggles | {"include_granite_ttm_features": False},
        "tabular_without_ts_proxies": true_toggles
        | {
            "include_model_proxy_features": False,
            "include_proxy_features": False,
            "include_proxy_model_inputs": False,
            "include_chronos_features": False,
            "include_granite_ttm_features": False,
        },
    }

    assert {
        scenario.scenario_id: scenario.toggles.to_dict() for scenario in registry
    } == expected_by_scenario


def test_data_channel_scenarios_enable_exactly_one_source() -> None:
    registry = default_ablation_registry()

    assert registry.get("price_only").toggles.feature_source_toggles() == {
        "price": True,
        "text": False,
        "sec": False,
    }
    assert registry.get("text_only").toggles.feature_source_toggles() == {
        "price": False,
        "text": True,
        "sec": False,
    }
    assert registry.get("sec_only").toggles.feature_source_toggles() == {
        "price": False,
        "text": False,
        "sec": True,
    }


def test_data_channel_scenarios_have_explicit_feature_family_allowlists() -> None:
    registry = default_ablation_registry()

    assert DATA_CHANNEL_FEATURE_FAMILY_ALLOWLISTS == {
        "price_only": PRICE_ONLY_FEATURE_FAMILY_ALLOWLIST,
        "text_only": TEXT_ONLY_FEATURE_FAMILY_ALLOWLIST,
        "sec_only": SEC_ONLY_FEATURE_FAMILY_ALLOWLIST,
    }
    assert registry.get("price_only").feature_family_allowlist == (
        "price",
        "chronos",
        "granite_ttm",
    )
    assert registry.get("price_only").permitted_feature_families == (
        "price",
        "chronos",
        "granite_ttm",
    )
    assert registry.get("text_only").feature_family_allowlist == ("text",)
    assert registry.get("sec_only").feature_family_allowlist == ("sec",)
    assert set().union(
        *(
            set(registry.get(scenario_id).feature_family_allowlist)
            for scenario_id in DATA_CHANNEL_FEATURE_FAMILY_ALLOWLISTS
        )
    ).issubset(VALID_FEATURE_FAMILIES)


def test_ablation_scenarios_compute_permitted_feature_families_from_toggles() -> None:
    registry = default_ablation_registry()

    assert {
        scenario.scenario_id: scenario.permitted_feature_families for scenario in registry
    } == {
        "all_features": ("price", "text", "sec", "chronos", "granite_ttm"),
        "no_text_risk": ("price", "text", "sec", "chronos", "granite_ttm"),
        "no_sec_risk": ("price", "text", "sec", "chronos", "granite_ttm"),
        "no_model_proxy": ("price", "text", "sec"),
        "no_costs": ("price", "text", "sec", "chronos", "granite_ttm"),
        "price_only": ("price", "chronos", "granite_ttm"),
        "text_only": ("text",),
        "sec_only": ("sec",),
        "full_model_features": ("price", "text", "sec", "chronos", "granite_ttm"),
        "no_chronos_features": ("price", "text", "sec", "granite_ttm"),
        "no_granite_features": ("price", "text", "sec", "chronos"),
        "tabular_without_ts_proxies": ("price", "text", "sec"),
    }


def test_ablation_scenarios_export_json_ready_contract_payloads() -> None:
    payloads = default_ablation_registry().to_dicts()

    assert {payload["scenario"] for payload in payloads} == set(DEFAULT_ABLATION_SCENARIO_IDS)
    assert {payload["scenario_id"] for payload in payloads} == set(DEFAULT_ABLATION_SCENARIO_IDS)
    assert all(isinstance(payload["label"], str) for payload in payloads)
    assert all(isinstance(payload["description"], str) for payload in payloads)
    assert all(isinstance(payload["toggles"], dict) for payload in payloads)
    assert all(isinstance(payload["feature_sources"], dict) for payload in payloads)
    assert all(isinstance(payload["feature_family_allowlist"], list) for payload in payloads)
    assert all(isinstance(payload["permitted_feature_families"], list) for payload in payloads)
    assert all(isinstance(payload["proxy_removal_options"], dict) for payload in payloads)
    assert {
        payload["scenario"]: payload["feature_family_allowlist"]
        for payload in payloads
        if payload["scenario"] in {"price_only", "text_only", "sec_only"}
    } == {
        "price_only": ["price", "chronos", "granite_ttm"],
        "text_only": ["text"],
        "sec_only": ["sec"],
    }
    assert all(isinstance(payload["pipeline_controls"], dict) for payload in payloads)
    assert default_ablation_registry().get("no_model_proxy").toggles.pipeline_control_toggles() == {
        "model_proxy": False,
        "proxy_features": False,
        "proxy_model_inputs": False,
        "cost": True,
        "slippage": True,
        "turnover": True,
    }
    assert default_ablation_registry().get("no_model_proxy").toggles.proxy_removal_options() == {
        "remove_proxy_features": True,
        "remove_proxy_model_inputs": True,
    }
    assert default_ablation_registry().get("no_costs").toggles.pipeline_control_toggles() == {
        "model_proxy": True,
        "proxy_features": True,
        "proxy_model_inputs": True,
        "cost": False,
        "slippage": False,
        "turnover": False,
    }


def test_pipeline_control_and_cost_are_first_class_scenario_kinds() -> None:
    assert "pipeline_control" in VALID_ABLATION_SCENARIO_KINDS
    assert "cost" in VALID_ABLATION_SCENARIO_KINDS


def test_ablation_registry_lookup_is_deterministic_and_exact() -> None:
    registry = default_ablation_registry()
    scenarios_by_id = {scenario.scenario_id: scenario for scenario in registry}

    assert tuple(registry.get(scenario_id) for scenario_id in registry.ids()) == tuple(registry)
    assert tuple(registry.get(scenario_id) for scenario_id in reversed(registry.ids())) == tuple(
        scenarios_by_id[scenario_id] for scenario_id in reversed(DEFAULT_ABLATION_SCENARIO_IDS)
    )
    assert default_ablation_registry().to_dicts() == registry.to_dicts()
    assert registry.get("sec_only") is scenarios_by_id["sec_only"]

    with pytest.raises(KeyError, match="unknown ablation scenario_id: SEC_ONLY"):
        registry.get("SEC_ONLY")


def test_ablation_registry_rejects_unstable_or_duplicate_scenario_ids() -> None:
    scenario = AblationScenarioConfig(
        scenario_id="stable_case",
        kind="signal",
        label="Stable case",
        toggles=AblationToggles(),
    )

    with pytest.raises(ValueError, match="duplicate"):
        AblationScenarioRegistry([scenario, scenario])

    with pytest.raises(ValueError, match="stable snake_case"):
        AblationScenarioConfig(
            scenario_id="unstable-case",
            kind="signal",
            label="Bad case",
            toggles=AblationToggles(),
        )

    with pytest.raises(TypeError, match="include_text_risk"):
        AblationToggles(include_text_risk=1)


def test_data_channel_ablation_rejects_missing_or_inconsistent_feature_family_allowlists() -> None:
    with pytest.raises(ValueError, match="explicit feature_family_allowlist"):
        AblationScenarioConfig(
            scenario_id="custom_channel",
            kind="data_channel",
            label="Custom channel",
            toggles=AblationToggles(),
        )

    with pytest.raises(ValueError, match="text_only feature_family_allowlist"):
        AblationScenarioConfig(
            scenario_id="text_only",
            kind="data_channel",
            label="Text only",
            toggles=AblationToggles(
                include_price_features=False,
                include_sec_features=False,
                include_sec_risk=False,
                include_chronos_features=False,
                include_granite_ttm_features=False,
            ),
            feature_family_allowlist=("price",),
        )

    with pytest.raises(ValueError, match="disabled feature families"):
        AblationScenarioConfig(
            scenario_id="custom_channel",
            kind="data_channel",
            label="Custom channel",
            toggles=AblationToggles(include_text_features=False),
            feature_family_allowlist=("text",),
        )
