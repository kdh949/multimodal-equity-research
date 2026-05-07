# Live Order Placement Absence Audit

This artifact summarizes the reviewed evidence that the repository has no live
order placement behavior. It is documentation-only evidence for semantic QA
hardening and does not change validation behavior, signal semantics, model
predictions, backtest results, report metrics, SEC EDGAR request behavior, or
report artifact schemas.

## Audit Scope

| Surface | Affected paths | Invariant |
| --- | --- | --- |
| Executable application code | `app.py`, `src/quant_research/**/*.py` | Research code must not import broker SDKs, define broker/order execution modules, call order placement APIs, or construct live execution payloads. |
| Static report outputs | `src/quant_research/validation/**`, `tests/fixtures/report_generation/**` | Reports are review artifacts only; they must not contain live order payloads or order-management controls. |
| Policy evidence text | `docs/**`, `tests/**`, `tests/fixtures/**` | Broker/order terminology is allowed only to describe prohibited behavior, audit commands, synthetic failing examples, or report-only isolation evidence. |

## Verified Absence Summary

| Gap | Evidence type | Production enforced | Behavioral test | Validation impact |
| --- | --- | ---: | ---: | --- |
| Live order placement behavior could be introduced silently. | Static architecture guard tests plus ad hoc text scan. | Yes | Yes | None: no validation formulas, signal thresholds, model outputs, backtest accounting, or report metrics are changed. |
| Report generation could look like an execution surface. | Report-only artifact tests over JSON, Markdown, HTML, and fixture files. | Yes | Yes | None: report writers still emit static review files only. |
| Policy words could be mistaken for implementation permission. | Documentation allowlist and policy-language tests. | No | Yes | None: allowlisted words are non-executable evidence only. |

## Reproduction Commands

Run from the repository root:

```bash
uv --cache-dir .uv-cache run pytest tests/test_architecture_guards.py tests/test_report_only_execution_isolation.py tests/test_policy_language_guardrails.py
```

Expected result: tests pass. The checks verify no production broker SDK
dependencies/imports, no broker/order execution modules, no order placement API
calls, no live execution payload keys, no report-only execution payloads, and no
policy-language usage outside approved non-executable evidence paths.

For a reviewer-readable source scan:

```bash
rg -n --glob 'src/quant_research/**/*.py' --glob 'app.py' \
  '(place_order|submit_order|create_order|market_order|limit_order|send_order|live_trade|broker)'
```

Expected result: no output. `rg` returns exit code 1 when it finds no matches,
so the audit signal is the empty output rather than a zero exit status.

## Allowlist

| Allowlisted context | Allowed paths | Required invariant |
| --- | --- | --- |
| Non-executable policy docs | `docs/**` | Text must describe prohibition, audit evidence, or expected absence only. |
| Guardrail tests and synthetic unsafe examples | `tests/**` | Tests must fail if executable production code gains live order placement behavior. |
| Report fixtures and report isolation tests | `tests/fixtures/**`, `tests/test_report_only_execution_isolation.py` | Fixtures may prove report-only behavior but must not become generated production artifacts. |

## Review Notes

- The production-enforced checks are hard-fail tests; this document adds no
  runtime branches and no adapter interfaces.
- The behavioral proof is intentionally small and reproducible through committed
  tests and fixtures only.
- No API keys, raw data caches, model artifacts, generated reports, or bulky
  command-output snapshots are part of this evidence.
