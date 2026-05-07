from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, fields
from typing import Literal

AblationScenarioKind = Literal["signal", "cost", "pipeline_control", "data_channel", "model_feature"]
FeatureFamily = Literal["price", "text", "sec", "chronos", "granite_ttm"]

FEATURE_FAMILY_ORDER: tuple[FeatureFamily, ...] = (
    "price",
    "text",
    "sec",
    "chronos",
    "granite_ttm",
)
VALID_ABLATION_SCENARIO_KINDS: frozenset[str] = frozenset(
    {"signal", "cost", "pipeline_control", "data_channel", "model_feature"}
)
VALID_FEATURE_FAMILIES: frozenset[str] = frozenset(FEATURE_FAMILY_ORDER)
ABLATION_SCENARIO_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
ABLATION_NON_FEATURE_COLUMNS: frozenset[str] = frozenset({"date", "ticker"})
NO_COST_ABLATION_SCENARIO = "no_costs"

PRICE_ONLY_FEATURE_FAMILY_ALLOWLIST: tuple[FeatureFamily, ...] = (
    "price",
    "chronos",
    "granite_ttm",
)
TEXT_ONLY_FEATURE_FAMILY_ALLOWLIST: tuple[FeatureFamily, ...] = ("text",)
SEC_ONLY_FEATURE_FAMILY_ALLOWLIST: tuple[FeatureFamily, ...] = ("sec",)

DATA_CHANNEL_FEATURE_FAMILY_ALLOWLISTS: dict[str, tuple[FeatureFamily, ...]] = {
    "price_only": PRICE_ONLY_FEATURE_FAMILY_ALLOWLIST,
    "text_only": TEXT_ONLY_FEATURE_FAMILY_ALLOWLIST,
    "sec_only": SEC_ONLY_FEATURE_FAMILY_ALLOWLIST,
}


def feature_family_for_column(column: str) -> FeatureFamily | None:
    column = str(column)
    if column in ABLATION_NON_FEATURE_COLUMNS or column.startswith("forward_return_"):
        return None
    if column.startswith("chronos_"):
        return "chronos"
    if column.startswith("granite_ttm_"):
        return "granite_ttm"
    if column.startswith(("news_", "text_")):
        return "text"
    if column.startswith(("sec_", "revenue_", "net_income_", "assets_")):
        return "sec"
    return "price"


def feature_family_columns(columns: Iterable[str]) -> dict[FeatureFamily, tuple[str, ...]]:
    grouped: dict[FeatureFamily, list[str]] = {family: [] for family in FEATURE_FAMILY_ORDER}
    for column in columns:
        family = feature_family_for_column(column)
        if family is not None:
            grouped[family].append(str(column))
    return {
        family: tuple(grouped[family])
        for family in FEATURE_FAMILY_ORDER
        if grouped[family]
    }


@dataclass(frozen=True, slots=True)
class AblationToggles:
    include_price_features: bool = True
    include_text_features: bool = True
    include_sec_features: bool = True
    include_text_risk: bool = True
    include_sec_risk: bool = True
    include_model_proxy_features: bool = True
    include_proxy_features: bool = True
    include_proxy_model_inputs: bool = True
    include_chronos_features: bool = True
    include_granite_ttm_features: bool = True
    include_transaction_costs: bool = True
    include_slippage: bool = True
    include_turnover_costs: bool = True

    def __post_init__(self) -> None:
        for field in fields(self):
            if not isinstance(getattr(self, field.name), bool):
                raise TypeError(f"{field.name} must be a bool")

    def to_dict(self) -> dict[str, bool]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def feature_source_toggles(self) -> dict[str, bool]:
        return {
            "price": self.include_price_features,
            "text": self.include_text_features,
            "sec": self.include_sec_features,
        }

    def pipeline_control_toggles(self) -> dict[str, bool]:
        return {
            "model_proxy": self.proxy_model_inputs_enabled(),
            "proxy_features": self.proxy_features_enabled(),
            "proxy_model_inputs": self.proxy_model_inputs_enabled(),
            "cost": self.include_transaction_costs,
            "slippage": self.include_slippage,
            "turnover": self.include_turnover_costs,
        }

    def proxy_features_enabled(self) -> bool:
        return self.include_model_proxy_features and self.include_proxy_features

    def proxy_model_inputs_enabled(self) -> bool:
        return self.include_model_proxy_features and self.include_proxy_model_inputs

    def proxy_removal_options(self) -> dict[str, bool]:
        return {
            "remove_proxy_features": not self.proxy_features_enabled(),
            "remove_proxy_model_inputs": not self.proxy_model_inputs_enabled(),
        }

    def enabled_feature_families(self) -> tuple[FeatureFamily, ...]:
        families: list[FeatureFamily] = []
        if self.include_price_features:
            families.append("price")
        if self.include_text_features:
            families.append("text")
        if self.include_sec_features:
            families.append("sec")
        if (
            self.include_price_features
            and self.proxy_features_enabled()
            and self.include_chronos_features
        ):
            families.append("chronos")
        if (
            self.include_price_features
            and self.proxy_features_enabled()
            and self.include_granite_ttm_features
        ):
            families.append("granite_ttm")
        return tuple(families)


@dataclass(frozen=True, slots=True)
class AblationScenarioConfig:
    scenario_id: str
    kind: AblationScenarioKind
    label: str
    toggles: AblationToggles
    description: str = ""
    feature_family_allowlist: tuple[FeatureFamily, ...] = ()

    def __post_init__(self) -> None:
        if not ABLATION_SCENARIO_ID_PATTERN.fullmatch(self.scenario_id):
            raise ValueError("scenario_id must be stable snake_case starting with a letter")
        if self.kind not in VALID_ABLATION_SCENARIO_KINDS:
            raise ValueError(f"unsupported ablation scenario kind: {self.kind}")
        if not self.label.strip():
            raise ValueError("label must not be blank")
        self._validate_feature_family_allowlist()

    @property
    def permitted_feature_families(self) -> tuple[FeatureFamily, ...]:
        return self.feature_family_allowlist or self.toggles.enabled_feature_families()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["scenario"] = self.scenario_id
        payload["toggles"] = self.toggles.to_dict()
        payload["feature_sources"] = self.toggles.feature_source_toggles()
        payload["pipeline_controls"] = self.toggles.pipeline_control_toggles()
        payload["proxy_removal_options"] = self.toggles.proxy_removal_options()
        payload["feature_family_allowlist"] = list(self.feature_family_allowlist)
        payload["permitted_feature_families"] = list(self.permitted_feature_families)
        return payload

    def _validate_feature_family_allowlist(self) -> None:
        families = self.feature_family_allowlist
        unsupported = sorted(set(families).difference(VALID_FEATURE_FAMILIES))
        if unsupported:
            raise ValueError(f"unsupported feature families: {unsupported}")
        if len(set(families)) != len(families):
            raise ValueError("feature_family_allowlist must not contain duplicate families")
        expected = DATA_CHANNEL_FEATURE_FAMILY_ALLOWLISTS.get(self.scenario_id)
        if expected is not None and families != expected:
            raise ValueError(
                f"{self.scenario_id} feature_family_allowlist must be {expected}"
            )
        if self.kind == "data_channel" and not families:
            raise ValueError("data_channel scenarios require an explicit feature_family_allowlist")
        self._validate_feature_family_toggle_consistency()

    def _validate_feature_family_toggle_consistency(self) -> None:
        enabled_families = set(self.toggles.enabled_feature_families())
        disabled = [
            family
            for family in self.feature_family_allowlist
            if family not in enabled_families
        ]
        if disabled:
            raise ValueError(
                "feature_family_allowlist includes disabled feature families: "
                f"{', '.join(disabled)}"
            )


class AblationScenarioRegistry:
    def __init__(self, scenarios: Iterable[AblationScenarioConfig]) -> None:
        ordered = tuple(scenarios)
        if not ordered:
            raise ValueError("ablation scenario registry must contain at least one scenario")

        by_id: dict[str, AblationScenarioConfig] = {}
        for scenario in ordered:
            if scenario.scenario_id in by_id:
                raise ValueError(f"duplicate ablation scenario_id: {scenario.scenario_id}")
            by_id[scenario.scenario_id] = scenario

        self._scenarios = ordered
        self._by_id = by_id

    def __iter__(self) -> Iterator[AblationScenarioConfig]:
        return iter(self._scenarios)

    def __len__(self) -> int:
        return len(self._scenarios)

    def ids(self) -> tuple[str, ...]:
        return tuple(scenario.scenario_id for scenario in self._scenarios)

    def get(self, scenario_id: str) -> AblationScenarioConfig:
        try:
            return self._by_id[scenario_id]
        except KeyError as exc:
            raise KeyError(f"unknown ablation scenario_id: {scenario_id}") from exc

    def by_kind(self, kind: AblationScenarioKind) -> tuple[AblationScenarioConfig, ...]:
        if kind not in VALID_ABLATION_SCENARIO_KINDS:
            raise ValueError(f"unsupported ablation scenario kind: {kind}")
        return tuple(scenario for scenario in self._scenarios if scenario.kind == kind)

    def to_dicts(self) -> list[dict[str, object]]:
        return [scenario.to_dict() for scenario in self._scenarios]


DEFAULT_ABLATION_SCENARIOS: tuple[AblationScenarioConfig, ...] = (
    AblationScenarioConfig(
        scenario_id="all_features",
        kind="signal",
        label="All signal features",
        description="Deterministic signal scoring with all structured signal features and realistic costs.",
        toggles=AblationToggles(),
    ),
    AblationScenarioConfig(
        scenario_id="no_text_risk",
        kind="signal",
        label="No text risk",
        description="Structured text risk features are zeroed before deterministic signal scoring.",
        toggles=AblationToggles(include_text_risk=False),
    ),
    AblationScenarioConfig(
        scenario_id="no_sec_risk",
        kind="signal",
        label="No SEC risk",
        description="Structured SEC event risk features are zeroed before deterministic signal scoring.",
        toggles=AblationToggles(include_sec_risk=False),
    ),
    AblationScenarioConfig(
        scenario_id="no_model_proxy",
        kind="pipeline_control",
        label="No model proxy",
        description="Walk-forward model refit after removing optional Chronos and Granite proxy features.",
        toggles=AblationToggles(
            include_model_proxy_features=False,
            include_proxy_features=False,
            include_proxy_model_inputs=False,
            include_chronos_features=False,
            include_granite_ttm_features=False,
        ),
    ),
    AblationScenarioConfig(
        scenario_id=NO_COST_ABLATION_SCENARIO,
        kind="cost",
        label="No transaction costs",
        description="Transaction costs, slippage, and turnover cost drag are disabled.",
        toggles=AblationToggles(
            include_transaction_costs=False,
            include_slippage=False,
            include_turnover_costs=False,
        ),
    ),
    AblationScenarioConfig(
        scenario_id="price_only",
        kind="data_channel",
        label="Price only",
        description="Walk-forward model refit using price and price-derived proxy features only.",
        toggles=AblationToggles(
            include_text_features=False,
            include_sec_features=False,
            include_text_risk=False,
            include_sec_risk=False,
        ),
        feature_family_allowlist=PRICE_ONLY_FEATURE_FAMILY_ALLOWLIST,
    ),
    AblationScenarioConfig(
        scenario_id="text_only",
        kind="data_channel",
        label="Text only",
        description="Walk-forward model refit using structured news/text features only.",
        toggles=AblationToggles(
            include_price_features=False,
            include_sec_features=False,
            include_sec_risk=False,
            include_chronos_features=False,
            include_granite_ttm_features=False,
        ),
        feature_family_allowlist=TEXT_ONLY_FEATURE_FAMILY_ALLOWLIST,
    ),
    AblationScenarioConfig(
        scenario_id="sec_only",
        kind="data_channel",
        label="SEC only",
        description="Walk-forward model refit using structured SEC filing and facts features only.",
        toggles=AblationToggles(
            include_price_features=False,
            include_text_features=False,
            include_text_risk=False,
            include_chronos_features=False,
            include_granite_ttm_features=False,
        ),
        feature_family_allowlist=SEC_ONLY_FEATURE_FAMILY_ALLOWLIST,
    ),
    AblationScenarioConfig(
        scenario_id="full_model_features",
        kind="model_feature",
        label="Full model features",
        description="Walk-forward model refit with all price, text, SEC, and time-series proxy features.",
        toggles=AblationToggles(),
    ),
    AblationScenarioConfig(
        scenario_id="no_chronos_features",
        kind="model_feature",
        label="No Chronos features",
        description="Walk-forward model refit after removing Chronos proxy forecast features.",
        toggles=AblationToggles(include_chronos_features=False),
    ),
    AblationScenarioConfig(
        scenario_id="no_granite_features",
        kind="model_feature",
        label="No Granite TTM features",
        description="Walk-forward model refit after removing Granite TTM proxy forecast features.",
        toggles=AblationToggles(include_granite_ttm_features=False),
    ),
    AblationScenarioConfig(
        scenario_id="tabular_without_ts_proxies",
        kind="model_feature",
        label="Tabular without time-series proxies",
        description="Walk-forward model refit after removing Chronos and Granite TTM proxy features.",
        toggles=AblationToggles(
            include_model_proxy_features=False,
            include_proxy_features=False,
            include_proxy_model_inputs=False,
            include_chronos_features=False,
            include_granite_ttm_features=False,
        ),
    ),
)

DEFAULT_ABLATION_SCENARIO_IDS: tuple[str, ...] = tuple(
    scenario.scenario_id for scenario in DEFAULT_ABLATION_SCENARIOS
)
DEFAULT_VALIDITY_GATE_ABLATION_MODE_IDS: tuple[str, ...] = (
    "all_features",
    "no_text_risk",
    "no_sec_risk",
    "no_model_proxy",
    NO_COST_ABLATION_SCENARIO,
    "price_only",
    "text_only",
    "sec_only",
)
VALIDITY_GATE_ABLATION_MODE_IDS: tuple[str, ...] = DEFAULT_ABLATION_SCENARIO_IDS
_VALIDITY_GATE_ABLATION_MODE_ALIASES: dict[str, str] = {
    "default": "default",
    "stage1_default": "default",
    "stage_1_default": "default",
    "stage1": "default",
    "no_model_proxy": "no_model_proxy",
    "no-model-proxy": "no_model_proxy",
    "no_cost": NO_COST_ABLATION_SCENARIO,
}


def default_ablation_registry() -> AblationScenarioRegistry:
    return AblationScenarioRegistry(DEFAULT_ABLATION_SCENARIOS)


def normalize_validity_gate_ablation_mode_ids(
    modes: Iterable[str] | str | None,
) -> tuple[str, ...]:
    if modes is None:
        return DEFAULT_VALIDITY_GATE_ABLATION_MODE_IDS
    raw_modes = (modes,) if isinstance(modes, str) else tuple(modes)
    if not raw_modes:
        return ()

    normalized: list[str] = []
    valid_ids = set(VALIDITY_GATE_ABLATION_MODE_IDS)
    for raw_mode in raw_modes:
        mode = _normalize_ablation_mode_id(raw_mode)
        if mode == "default":
            normalized.extend(DEFAULT_VALIDITY_GATE_ABLATION_MODE_IDS)
            continue
        if mode not in valid_ids:
            allowed = ", ".join(("default", *VALIDITY_GATE_ABLATION_MODE_IDS))
            raise ValueError(f"unsupported validity gate ablation mode: {raw_mode!r}; allowed: {allowed}")
        normalized.append(mode)
    return _dedupe_preserving_order(normalized)


def _normalize_ablation_mode_id(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    return _VALIDITY_GATE_ABLATION_MODE_ALIASES.get(normalized, normalized)


def _dedupe_preserving_order(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return tuple(output)
