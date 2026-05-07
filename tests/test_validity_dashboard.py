from __future__ import annotations

from dataclasses import dataclass

from quant_research.dashboard import validity as validity_dashboard


@dataclass(frozen=True)
class DummyValidityReport:
    system_validity_status: str
    strategy_candidate_status: str
    hard_fail: bool
    warnings: list[str]
    metrics: dict[str, object]
    evidence: dict[str, object]
    official_message: str
    benchmark_results: list[dict[str, object]]
    ablation_results: list[dict[str, object]]


def test_normalize_validity_gate_report_supports_dict_and_dataclass_input() -> None:
    payload = {
        "system_validity_status": "pass",
        "strategy_candidate_status": "warn",
        "hard_fail": False,
        "warnings": ["ok", "light warning"],
        "metrics": {"a": 1.0, "nested": {"b": 2.0}},
        "evidence": {"source": "unit"},
        "official_message": "Gate ready",
        "benchmark_results": [
            {"name": "bench_a", "score": 0.81},
        ],
        "ablation_results": [
            {"name": "full", "value": 0.42},
        ],
    }
    payload_dataclass = DummyValidityReport(**payload)

    normalized_dict = validity_dashboard.normalize_validity_gate_report(payload)
    normalized_object = validity_dashboard.normalize_validity_gate_report(payload_dataclass)

    assert normalized_dict["system_validity_status"] == "PASS"
    assert normalized_dict["strategy_candidate_status"] == "WARN"
    assert normalized_dict["hard_fail"] is False
    assert normalized_dict["warnings"] == ["ok", "light warning"]
    assert "Unit" not in normalized_dict["warnings"]
    assert not normalized_dict["benchmark_results"].empty
    assert not normalized_dict["ablation_results"].empty

    assert normalized_object["system_validity_status"] == "PASS"
    assert normalized_object["strategy_candidate_status"] == "WARN"
    assert normalized_object["hard_fail"] is False
    assert normalized_object["warnings"] == ["ok", "light warning"]
    assert normalized_object["official_message"] == "Gate ready"
    assert normalized_object["metrics"].equals(normalized_dict["metrics"])
    assert normalized_object["evidence"].equals(normalized_dict["evidence"])
    assert normalized_object["benchmark_results"].equals(normalized_dict["benchmark_results"])
    assert normalized_object["ablation_results"].equals(normalized_dict["ablation_results"])


def test_normalize_validity_gate_report_uses_defaults_for_missing_fields() -> None:
    normalized = validity_dashboard.normalize_validity_gate_report({})

    assert normalized["system_validity_status"] == "UNKNOWN"
    assert normalized["strategy_candidate_status"] == "UNKNOWN"
    assert normalized["hard_fail"] is False
    assert normalized["warnings"] == []
    assert normalized["official_message"] == ""
    assert normalized["metrics"].empty
    assert normalized["evidence"].empty
    assert normalized["benchmark_results"].empty
    assert normalized["ablation_results"].empty


def test_coerce_warning_list_flattens_nested_inputs() -> None:
    nested_warnings = ["first", ["second", ("third", 4.0)], "", {"ignore": "value"}]
    normalized = validity_dashboard.normalize_validity_gate_report({"warnings": nested_warnings})

    assert normalized["warnings"] == ["first", "second", "third", "4.0", "{'ignore': 'value'}"]


def test_flatten_and_frames_from_nested_mapping() -> None:
    nested = {"alpha": {"beta": 1, "gamma": 0.25}, "tag": "base"}
    frame = validity_dashboard._build_key_value_frame(nested, key_label="Metric")

    assert list(frame.columns) == ["Metric", "Value"]
    values = {row["Metric"]: row["Value"] for _, row in frame.iterrows()}
    assert values["alpha.beta"] == "1"
    assert values["alpha.gamma"] == "0.2500"
    assert values["tag"] == "base"


def test_to_result_frame_handles_list_of_rows_and_nested_mapping() -> None:
    nested_results = [{"scenario": "base", "metric": 0.1}, {"scenario": "alt", "metric": 0.2}]
    result_frame = validity_dashboard._to_result_frame(nested_results)
    assert list(result_frame.columns) == ["scenario", "metric"]
    assert len(result_frame) == 2
    assert result_frame["scenario"].tolist() == ["base", "alt"]

    mapping_results = validity_dashboard._to_result_frame({"one": {"two": 2}, "three": 3})
    assert set(mapping_results.columns) == {"Key", "Value"}
    assert mapping_results.loc[0, "Key"] == "one.two"
    assert mapping_results.loc[0, "Value"] == "2"
