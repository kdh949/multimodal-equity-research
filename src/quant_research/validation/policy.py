from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

GateDecision = Literal["PASS", "WARN", "FAIL"]
NormalizedGateStatus = Literal[
    "pass",
    "warning",
    "fail",
    "hard_fail",
    "insufficient_data",
    "not_evaluable",
    "skipped",
]

PASS_GATE_STATUS = "pass"
NON_PASS_GATE_STATUSES: tuple[NormalizedGateStatus, ...] = (
    "warning",
    "fail",
    "hard_fail",
    "insufficient_data",
    "not_evaluable",
    "skipped",
)
GATE_STATUS_PRECEDENCE: tuple[NormalizedGateStatus, ...] = (
    "hard_fail",
    "fail",
    "insufficient_data",
    "not_evaluable",
    "skipped",
    "warning",
    "pass",
)
GATE_STATUS_ALIASES: dict[str, NormalizedGateStatus] = {
    "pass": "pass",
    "passed": "pass",
    "ok": "pass",
    "warn": "warning",
    "warning": "warning",
    "fail": "fail",
    "failed": "fail",
    "hard_fail": "hard_fail",
    "hard-fail": "hard_fail",
    "hard fail": "hard_fail",
    "insufficient": "insufficient_data",
    "insufficient_data": "insufficient_data",
    "insufficient data": "insufficient_data",
    "not_evaluable": "not_evaluable",
    "not evaluable": "not_evaluable",
    "not-evaluable": "not_evaluable",
    "missing": "not_evaluable",
    "skipped": "skipped",
    "skip": "skipped",
}


@dataclass(frozen=True)
class GateStatusClassification:
    raw_status: str
    normalized_status: NormalizedGateStatus
    decision: GateDecision
    passed: bool
    non_pass: bool
    severity_rank: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateStatusPolicy:
    """Common PASS/non-PASS status policy for validation gates.

    The policy is deliberately provider-free and metric-free. Gate evaluators
    produce rule-specific statuses; this class only normalizes aliases and
    determines whether a status can contribute to a PASS.
    """

    status_aliases: Mapping[str, NormalizedGateStatus] | None = None
    status_precedence: tuple[NormalizedGateStatus, ...] = GATE_STATUS_PRECEDENCE

    def normalize(self, status: object) -> NormalizedGateStatus:
        raw = str(status or "not_evaluable").strip().lower()
        aliases = self.status_aliases or GATE_STATUS_ALIASES
        return aliases.get(raw, "not_evaluable")

    def classify(self, status: object) -> GateStatusClassification:
        normalized = self.normalize(status)
        return GateStatusClassification(
            raw_status=str(status or "not_evaluable"),
            normalized_status=normalized,
            decision=self.decision_for_status(normalized),
            passed=normalized == PASS_GATE_STATUS,
            non_pass=normalized != PASS_GATE_STATUS,
            severity_rank=self.severity_rank(normalized),
        )

    def decision_for_status(self, status: object) -> GateDecision:
        normalized = self.normalize(status)
        if normalized == "pass":
            return "PASS"
        if normalized == "warning":
            return "WARN"
        return "FAIL"

    def is_pass(self, status: object) -> bool:
        return self.normalize(status) == PASS_GATE_STATUS

    def is_non_pass(self, status: object) -> bool:
        return not self.is_pass(status)

    def severity_rank(self, status: object) -> int:
        normalized = self.normalize(status)
        try:
            return self.status_precedence.index(normalized)
        except ValueError:
            return self.status_precedence.index("not_evaluable")

    def worst_status(self, statuses: object) -> NormalizedGateStatus:
        if isinstance(statuses, (str, bytes)) or not hasattr(statuses, "__iter__"):
            return self.normalize(statuses)
        normalized_statuses = [self.normalize(status) for status in statuses]
        if not normalized_statuses:
            return "not_evaluable"
        return min(normalized_statuses, key=self.severity_rank)

    def non_pass_gate_results(
        self,
        gate_results: Mapping[str, Mapping[str, Any]],
        *,
        include_unaffected: bool = False,
    ) -> dict[str, dict[str, Any]]:
        non_pass_results: dict[str, dict[str, Any]] = {}
        for gate_name, result in gate_results.items():
            if not isinstance(result, Mapping):
                continue
            if not include_unaffected and not bool(
                result.get(
                    "affects_pass_fail",
                    result.get("affects_system", result.get("affects_strategy", True)),
                )
            ):
                continue
            classification = self.classify(result.get("status"))
            if not classification.non_pass:
                continue
            payload = dict(result)
            payload["normalized_status"] = classification.normalized_status
            payload["decision"] = classification.decision
            payload["non_pass"] = True
            non_pass_results[str(gate_name)] = payload
        return non_pass_results


DEFAULT_GATE_STATUS_POLICY = GateStatusPolicy()
