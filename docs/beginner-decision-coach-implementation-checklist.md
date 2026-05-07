# Beginner Decision Coach Implementation Checklist

This checklist compresses the approved gstack design into implementation stages so the work can proceed without reopening the full design document on every step.

Source design:

- `/Users/donghyunkim/.gstack/projects/Quantitative-Trading/donghyunkim-main-design-20260508-010021.md`

Working rule:

- Finish one stage, run the listed verification, report to the user, then wait for approval before starting the next stage.
- Keep real trading, broker connection, order buttons, and personalized investment advice out of scope.
- LLM/text models remain feature extractors only. Final visible decision labels must come from deterministic signal outputs plus validation gate state.
- Do not expose raw `BUY / SELL / HOLD` in the beginner top block. Raw action may appear only in a default-closed advanced disclosure as provenance.

## Sub-worker Strategy

Use sub-workers only for simple coding tasks with clearly separated file ownership.

- Simple tasks with disjoint file/module ownership may be delegated to sub-workers.
- To reduce token cost, prefer `GPT-5.3-Codex-Spark` with xhigh reasoning for these workers.
- Multiple workers may run in parallel only when their write scopes do not overlap.
- Give each worker explicit ownership of the files/modules they may edit.
- Tell each worker not to revert changes made by other workers or by the user.
- The main agent owns shared interfaces, signal/validation semantics, final integration decisions, and final test review.

Recommended split:

- Worker A: draft unit coverage in `tests/test_beginner_dashboard.py`.
- Worker B: draft audit documentation and guard updates in `docs/final-action-label-inventory.md` and `tests/test_architecture_guards.py`.
- Main agent: implement `src/quant_research/dashboard/` contracts, wire `app.py`, integrate all changes, and run final tests.

## Stage Status

| Stage | Scope | Status | Report Gate |
|---|---|---|---|
| 1 | Core report contract and pure deterministic mapping | Done | Reported |
| 2 | Pipeline/app wiring and validity gate order | Done | Reported |
| 3 | Streamlit Final Signal Strip renderer | Done | Reported |
| 4 | Evidence workspace, explanation, and advanced disclosure rendering | Done | Reported |
| 5 | Documentation and architecture guard updates | Done | Reported |
| 6 | Browser QA and responsive/accessibility checks | Done | Final implementation checkpoint |

## Stage 1: Core Report Contract

Goal: Add the pure data contract and deterministic label logic without changing visible UI.

Files:

- `src/quant_research/dashboard/beginner.py`
- `src/quant_research/dashboard/__init__.py`
- `tests/test_beginner_dashboard.py`

Checklist:

- [x] Add `BeginnerDecisionCoachReport`.
- [x] Add `build_beginner_decision_coach_report()`.
- [x] Read final raw action from `result.signals` only.
- [x] Ignore `result.predictions["action"]` for final decision labels.
- [x] Map `BUY -> 긍정적`, `SELL -> 부정적`, `HOLD -> 보류`.
- [x] Map gate fail/missing/error to `{mapped label}이지만 검증 불충분`.
- [x] Keep visual tone `blocked` or `caution` when validation is insufficient.
- [x] Add deterministic reason codes and deterministic Korean template text.
- [x] Keep raw action hidden from top block and available only as advanced provenance metadata.
- [x] Add focused unit tests for action source, gate downgrade, and missing signal behavior.

Verification:

- [x] `.venv/bin/ruff check src/quant_research/dashboard/beginner.py src/quant_research/dashboard/__init__.py tests/test_beginner_dashboard.py`
- [x] `.venv/bin/python -m pytest tests/test_beginner_dashboard.py tests/test_architecture_guards.py`

Stage 1 notes:

- UI rendering is intentionally not wired yet.
- Existing untracked design/research docs under `docs/` were left untouched.

## Stage 2: Pipeline/App Wiring

Goal: Ensure the validity gate report exists before beginner decision UI data is built.

Files:

- `app.py`
- `src/quant_research/dashboard/beginner.py` only if the Stage 1 builder needs a small signature adjustment
- `tests/test_app_smoke.py` or a focused app-level smoke test

Checklist:

- [x] Reorder `app.py` so `build_validity_gate_report(...)` runs before beginner dashboard/coach construction.
- [x] Build `BeginnerDecisionCoachReport` after `validity_report` is available.
- [x] Keep existing `build_beginner_research_dashboard(...)` behavior intact until renderer migration is explicit.
- [x] Store or pass the coach report through the same local render boundary as the current beginner overview.
- [x] Verify `validity_report` missing/fail states cannot produce standalone `긍정적 / 부정적 / 보류` in the coach report.
- [x] Add smoke coverage that the app constructs validity data before the coach report.

Verification:

- [x] `.venv/bin/python -m pytest tests/test_beginner_dashboard.py tests/test_app_smoke.py`
- [x] `.venv/bin/python -m pytest tests/test_architecture_guards.py`

Report before continuing:

- [x] Summarize changed wiring.
- [x] Confirm no visible UI render change beyond data availability, unless Stage 3 is explicitly approved.

## Stage 3: Final Signal Strip Renderer

Goal: Render the first-viewport decision strip in Streamlit.

Files:

- `src/quant_research/dashboard/streamlit.py`
- `tests/test_beginner_dashboard.py`
- `tests/test_app_smoke.py` or Streamlit AppTest coverage if practical

Checklist:

- [x] Add a renderer for `Final Signal Strip`.
- [x] Show `display_label`, `forecast_direction_label`, `confidence_label`, and `validation_gate_label` in the same first-screen area.
- [x] Show `decision_source = deterministic_signal_engine` near the label.
- [x] Use `blocked/caution` tone for composite insufficient-validation labels.
- [x] Do not show raw `BUY / SELL / HOLD` in the top block.
- [x] Preserve the existing disclaimer meaning: research aid, no personalized advice, no orders, no LLM final signal.
- [x] Keep layout calm and app-like, not a generic card mosaic.

Verification:

- [x] `.venv/bin/python -m pytest tests/test_beginner_dashboard.py tests/test_app_smoke.py`
- [x] `.venv/bin/ruff check src/quant_research/dashboard/streamlit.py tests/test_beginner_dashboard.py tests/test_app_smoke.py`

Report before continuing:

- [x] Include the exact labels rendered for pass/fail gate examples.

## Stage 4: Evidence, Explanation, And Disclosure Rendering

Goal: Render the beginner explanation, evidence workspace, and default-closed provenance disclosure.

Files:

- `src/quant_research/dashboard/streamlit.py`
- `tests/test_beginner_dashboard.py`
- `tests/test_app_smoke.py`

Checklist:

- [x] Render `plain_language_explanation`.
- [x] Render `why_it_might_be_wrong`.
- [x] Render evidence as label/value/detail rows, not decorative cards.
- [x] Include expected return, signal score, downside, volatility, text/SEC risk, cost/slippage, and data cutoff where available.
- [x] Separate available evidence from missing evidence.
- [x] Add default-closed advanced disclosure for raw action provenance.
- [x] Include the raw action warning text: `이 값은 주문 신호가 아니라 deterministic engine의 원천 action입니다`.
- [x] Avoid implying price targets or guaranteed returns.

Verification:

- [x] `.venv/bin/python -m pytest tests/test_beginner_dashboard.py tests/test_app_smoke.py`
- [x] `.venv/bin/python -m pytest tests/test_architecture_guards.py`

Report before continuing:

- [x] Summarize evidence fields shown and missing-data behavior.

## Stage 5: Documentation And Guard Updates

Goal: Keep semantic safety documentation aligned with the new visible label surface.

Files:

- `docs/final-action-label-inventory.md`
- `tests/test_architecture_guards.py`
- Optional: `README.md` only if the UI disclaimer or project overview needs the agreed sentence

Checklist:

- [x] Add Beginner Decision Coach visible label surface to the final action label inventory.
- [x] Document that Korean beginner labels are mapped from deterministic signals plus validation gate state.
- [x] Document that raw action appears only in advanced disclosure as provenance.
- [x] Ensure architecture guards allow the new mapping surface but still forbid model/LLM final action emission.
- [x] Update tests to cover new inventory/guard expectations.

Verification:

- [x] `.venv/bin/python -m pytest tests/test_architecture_guards.py`
- [x] `.venv/bin/python -m pytest tests/test_beginner_dashboard.py`

Report before continuing:

- [x] Summarize new inventory entries and any guard allowlist changes.

## Stage 6: Browser QA And Responsive/Accessibility Checks

Goal: Verify the implemented UI against the deferred design-review checks.

Files:

- No planned code files unless QA finds an issue.
- Browser/QA artifacts should not be committed unless explicitly requested.

Checklist:

- [x] Run the Streamlit app locally.
- [x] Verify desktop first viewport shows final signal, validation state, source, and disclaimer context.
- [x] Verify 375px/mobile width does not hide validation/source/disclaimer behind only the final label.
- [x] Verify final strip text does not overlap or truncate badly.
- [x] Verify keyboard-only ticker input and run action.
- [x] Verify color is not the only carrier of signal meaning.
- [x] Verify chart fallback and error states include readable text.
- [x] Verify there are no order, trade, broker, or personalized advice controls.

Verification:

- [x] `.venv/bin/python -m pytest tests/test_beginner_dashboard.py tests/test_app_smoke.py tests/test_architecture_guards.py`
- [x] Browser QA notes reported to the user.

Report before finishing:

- [x] State any responsive/accessibility concerns that remain.
- [x] Recommend whether to run `/design-review` after implementation.

## Commit Plan

Follow the repository convention: `<type>: <한국어 제목>`, no scope.

Suggested commits:

- [ ] `test: 초심자 결정 코치 신호 출처 검증 추가`
- [ ] `feat: 초심자 결정 코치 리포트 계약 추가`
- [ ] `feat: 초심자 결정 코치 화면 연결`
- [ ] `docs: 초심자 결정 라벨 감사 문서 갱신`
- [ ] `test: 초심자 결정 화면 스모크 검증 추가`

## Current Deferred Items

- Detailed responsive/accessibility spec was intentionally deferred to browser QA.
- gstack designer mockups were intentionally skipped because designer authentication was not ready.
- No `TODOS.md` item was created for either deferred item by user choice.
