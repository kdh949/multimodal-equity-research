# Risk And Validation

## Fixed Validation Flow

1. 모델이 예측값을 생성한다.
2. 예측값을 바로 주문하지 않는다.
3. 예측값과 텍스트/SEC feature를 신호 점수로 변환한다.
4. 거래비용, 슬리피지, turnover를 반영한다.
5. walk-forward 검증을 수행한다.
6. out-of-sample 성능을 확인한다.
7. deterministic signal engine이 매수/매도/관망을 판단한다.

## No-Live-Trading Boundary

This repository is a research-only quantitative validation app. The supported
v1 scope is limited to data collection, feature generation, optional model
prediction, deterministic signal scoring, walk-forward validation, long-only
backtest simulation, static report generation, and Streamlit review surfaces.

The production application must not include live trading behavior. Prohibited
behavior includes broker account connectivity, order routing, order ticket
construction, order status polling, live execution payloads, and API calls or
adapters that place, submit, create, send, or manage orders. Any broker/order
language in this document or in tests is policy evidence only; it does not
authorize executable broker adapters or runtime branches.

Research outputs stop at auditable validation artifacts:

- Predictions remain structured research features, not orders or final action
  labels.
- `BUY`, `SELL`, and `HOLD` are deterministic signal-engine labels emitted only
  after scoring, cost/slippage, turnover, validation, out-of-sample, and risk
  checks.
- Backtests are simulations that apply labels to t+1-or-later returns and record
  costs, slippage, turnover, risk metrics, and benchmark comparisons.
- Reports and dashboards summarize predictions, validation status, deterministic
  signal aggregates, and backtest metrics for review only; they must not render
  broker/order instructions, live execution payloads, or order-management
  controls.

Auditable evidence:

- Production enforcement:
  `tests/test_architecture_guards.py::test_repository_has_no_live_trading_or_order_execution_modules`,
  `tests/test_architecture_guards.py::test_executable_source_does_not_import_live_trading_broker_sdks`,
  `tests/test_architecture_guards.py::test_source_does_not_call_broker_order_apis`,
  `tests/test_architecture_guards.py::test_executable_source_does_not_define_live_execution_payload_keys`,
  and
  `tests/test_report_only_execution_isolation.py::test_report_only_outputs_contain_no_live_order_or_execution_payloads`.
- Documentation guard:
  `tests/test_policy_language_guardrails.py::test_risk_validation_docs_define_no_live_trading_boundary`.
- Reproducible command:

```bash
uv --cache-dir .uv-cache run pytest tests/test_architecture_guards.py tests/test_report_only_execution_isolation.py tests/test_policy_language_guardrails.py
```

Expected validation impact: none. This section documents existing scope and
hard-fail guardrails only; it must not alter validation behavior, signal
semantics, model predictions, backtest results, report metrics, or generated
artifact schemas.

## Final Action Label Gate Sequence

`BUY`, `SELL`, and `HOLD` are final action labels, not model prediction
classes. They may be emitted only after the validation and deterministic signal
layers have passed the following sequence:

1. Point-in-time input checks pass: feature availability timestamps and model
   prediction timestamps are not later than the signal date, and realized return
   labels are excluded from signal-engine inputs.
2. Model and optional LLM/text adapters produce structured research features
   such as expected return, volatility, downside quantile, confidence,
   sentiment, event, and SEC risk fields. These values remain inputs only.
3. The deterministic signal engine converts the structured prediction/features
   into `signal_score` after subtracting configured transaction cost and
   slippage plus volatility, downside, text/news, SEC, turnover, concentration,
   drawdown, and portfolio-risk penalties.
4. Walk-forward validation uses past training windows, a future holdout window,
   and the canonical `forward_return_20` target with purge/embargo protection.
5. Out-of-sample and baseline comparisons pass the common validity gate. Final
   signal-generation paths must provide a gate payload whose
   `final_gate_decision` is `PASS`; missing, `WARN`, `FAIL`, hard-fail, or
   not-evaluable gate results block final label emission when the gate is
   required.
6. The deterministic signal engine applies action-label thresholds and risk
   blockers: `BUY` requires expected return, score, volatility, downside,
   text/news risk, SEC throttle, and liquidity checks to pass; `SELL` is emitted
   only by deterministic severe risk/downside rules; all remaining rows stay
   `HOLD`.
7. The long-only backtest can size positions only after labels are generated,
   then applies turnover limits, concentration/risk-contribution/volatility
   limits, transaction costs, slippage, and post-cost position-sizing
   validation. Positions use t+1-or-later returns, not same-day or future
   features.

Auditable evidence:

- Production enforcement:
  `src/quant_research/signals/engine.py::require_signal_generation_gate_pass`
  and `src/quant_research/backtest/engine.py::run_long_only_backtest`.
- Behavioral tests:
  `tests/test_backtest_risk.py::test_backtest_blocks_final_signal_path_when_common_gate_is_not_pass`,
  `tests/test_backtest_risk.py::test_backtest_records_common_gate_pass_on_final_signals`,
  `tests/test_backtest_risk.py::test_backtest_signal_engine_receives_no_realized_return_columns`,
  and `tests/test_signal_engine.py` deterministic threshold/risk cases.
- Reproducible command:

```bash
python3 -m pytest tests/test_backtest_risk.py tests/test_signal_engine.py tests/test_policy_language_guardrails.py
```

Expected validation impact: none. This section documents existing behavior and
does not change validation semantics, signal labels, model predictions, backtest
returns, or report metrics.

## Walk-Forward

- train window는 과거 구간만 사용한다.
- 기본 검증 target은 `forward_return_20`이다.
- 1일/5일 horizon은 pass/fail을 만들지 않는 diagnostic 지표로만 사용한다.
- gap과 embargo는 target horizon과 긴 가격 lookback을 모두 덮도록 기본 60 거래일 이상으로 둔다.
- test window는 train 이후 구간만 사용한다.
- 마지막 구간은 out-of-sample summary로 별도 표시한다.

## Backtest Rules

- 기본 전략은 상위 점수 종목 동일가중 롱온리다.
- 포지션은 신호 생성 다음 기간 수익률부터 적용한다.
- 비용은 turnover 기준으로 차감한다.
- 슬리피지는 거래된 notional 기준으로 차감한다.
- 백테스트 산출물은 gross return, transaction cost, slippage cost, total turnover cost, net cost-adjusted return을 분리해 기록한다.
- 벤치마크는 `SPY`다.

## Risk Rules

- 종목별 최대 비중
- 포트폴리오 변동성 한도와 최근 실현수익률 공분산 기반 상관관계 반영
- sector 데이터가 없을 때는 수익률 상관 cluster별 집중도 제한
- 최대 낙폭 중단 룰
- SEC 이벤트 리스크는 graded throttle로 처리하고, severe/recent risk일 때만 강한 매도 조건으로 쓴다.
- 최소 유동성 조건
- 텍스트 리스크 과열 차단

## Metrics

- CAGR
- annualized volatility
- Sharpe ratio
- maximum drawdown
- hit rate
- turnover
- exposure
- benchmark excess return
- OOS score stability
- feature ablation delta

## Semantic QA Audit Commands

금지된 broker/order 실행 기능은 두 단계로 감사한다.

Executable-code scan scope:

- `app.py`
- `src/quant_research/**/*.py`

Prohibited live-trading/order module terms:

- `broker`
- `brokers`
- `order`
- `orders`
- `execution`
- `trading`

Prohibited live-trading/order API call terms:

- `place_order`
- `submit_order`
- `create_order`
- `market_order`
- `limit_order`
- `send_order`
- `live_trade`

Prohibited broker SDK dependency/import examples:

- `alpaca-py`
- `alpaca-trade-api`
- `ib-insync`
- `ibapi`
- `ccxt`
- `tradier`
- `tda-api`
- `schwab-py`

Prohibited live-trading implementation name examples:

- `BrokerClient`
- `LiveTradingAdapter`
- `OrderExecutor`
- `place_order`
- `submit_order`

1. 실행 가능한 production 경로는 hard-fail 테스트로 감사한다.

```bash
python3 -m pytest tests/test_architecture_guards.py tests/test_report_only_execution_isolation.py
```

이 명령은 `pyproject.toml` dependency groups에 broker SDK가 추가되지
않았는지, `src/quant_research` 아래에 live trading, broker, order
execution 모듈명이 생기지 않았는지, `app.py`와
`src/quant_research/**/*.py` 안에서 production source가 broker SDK를
import하거나 `BrokerClient`, `LiveTradingAdapter`, `OrderExecutor` 같은
실행 surface를 정의하지 않는지를 검증한다. 또한 `place_order`,
`submit_order`, `create_order`, `market_order`, `limit_order`, `send_order`,
`live_trade` 같은 주문 API 호출과 live execution payload key가 없는지,
report-only 산출물이 정적 리포트 파일만 만들고 live execution payload나
broker/order 문구를 렌더링하지 않는지 확인한다.

2. 리뷰어가 변경분을 빠르게 재현할 수 있도록 동일한 production surface에
   대해 보조 텍스트 스캔을 실행한다.

```bash
rg -n --glob 'src/quant_research/**/*.py' --glob 'app.py' \
  '(place_order|submit_order|create_order|market_order|limit_order|send_order|live_trade|broker)'
```

정상 결과는 매치 없음이다. `rg`는 매치가 없으면 exit code 1을 반환하므로,
이 명령의 감사 의미는 출력이 비어 있는지 확인하는 것이다.

### Audit Evidence Reproduction

Run these commands from the repository root when reviewing semantic QA evidence.
The evidence is intentionally small and reviewable in source control: tests
assert the invariants, docs name the policy scope, and generated report fixtures
remain under `tests/fixtures/report_generation`. Do not commit raw data caches,
API keys, model artifacts, generated report directories, or bulky command-output
snapshots.

| Evidence area | Reproduction command | Expected result | Evidence reviewed or stored |
| --- | --- | --- | --- |
| Production live-trading/order absence | `uv --cache-dir .uv-cache run pytest tests/test_architecture_guards.py tests/test_report_only_execution_isolation.py` plus the ad hoc `rg` command below | Tests pass with no production broker SDK imports, live-trading modules, order API calls, live execution payload keys, or report payloads. | Review `docs/live-order-placement-audit.md`, test assertions in `tests/test_architecture_guards.py`, and report-only fixtures in `tests/fixtures/report_generation`; no new generated artifacts are written. |
| Forbidden policy-language allowlist | `uv --cache-dir .uv-cache run pytest tests/test_policy_language_guardrails.py` | Tests pass and broker/order terms remain confined to docs, tests, and fixtures as explicit prohibition or audit evidence. | Review this document's allowlist tables plus `tests/test_policy_language_guardrails.py`; evidence is stored as committed docs/tests only. |
| Deterministic signal-only final labels | `uv --cache-dir .uv-cache run pytest tests/test_signal_engine.py tests/test_backtest_risk.py tests/test_architecture_guards.py` | Tests pass and final `BUY`, `SELL`, and `HOLD` labels are emitted only after deterministic scoring, gate, cost/slippage, and risk checks. | Review `docs/final-action-label-inventory.md`, `src/quant_research/signals/engine.py`, and the named tests; no model output artifact is treated as a final label source. |
| Report-only metric isolation | `uv --cache-dir .uv-cache run pytest tests/test_validity_gate_metric_formulas.py tests/test_report_generation.py tests/test_report_only_execution_isolation.py` | Tests pass and `top_decile_20d_excess_return` remains a report-only out-of-sample diagnostic with no gate status, threshold, or final-signal effect. | Review report metric tests and static fixtures under `tests/fixtures/report_generation`; generated report artifacts are not committed. |
| Optional warning baseline | `uv --cache-dir .uv-cache run pytest -q` | Current integrated baseline is `832 passed, 12 warnings`; optional warning metadata remains documented as non-semantic dependency-surface evidence. | Review this section and `tests/test_warning_baseline_documentation.py`; keep only concise warning metadata in docs/tests. |

For ad hoc text review, run:

```bash
rg -n --glob 'src/quant_research/**/*.py' --glob 'app.py' \
  '(place_order|submit_order|create_order|market_order|limit_order|send_order|live_trade|broker)'
```

Expected result: no output. Because `rg` returns exit code 1 when no matches are
found, reviewers should evaluate this audit by the empty output, not by a zero
exit status.

Validation impact: none. These reproduction steps only prove existing
guardrails. They must not change validation behavior, signal semantics, model
predictions, backtest results, report metrics, SEC EDGAR request behavior, or
report-only artifact schemas.

### Broker/Order Term Allowlist

정책 문서, 테스트, fixture는 금지 문구 자체를 설명하거나 차단을 검증하기
위해 아래 broker/order 계열 용어를 포함할 수 있다. 이 allowlist는 production
경로의 주문 실행 기능을 허용하지 않으며, 모델 예측값이나 LLM 출력이 최종
`BUY`/`SELL`/`HOLD` 신호를 직접 만들 수 없다는 deterministic signal-only
경계를 바꾸지 않는다.

| Allowed reference | Permitted paths | Why it is allowed |
| --- | --- | --- |
| `broker` | `docs/**`, `tests/**`, `tests/fixtures/**` | 문서와 테스트에서 broker 연동이 v1 범위 밖임을 설명하고 production 코드에 broker surface가 없는지 검증하기 위한 증거 문구다. |
| `order`, `orders`, `place_order`, `submit_order`, `create_order`, `market_order`, `limit_order`, `send_order` | `docs/**`, `tests/**`, `tests/fixtures/**` | 주문 API 명칭을 차단 목록으로 고정하고 정적/행동 테스트가 해당 API 호출 부재를 검증하기 위해 필요하다. |
| `execution`, `live execution`, `live_trade` | `docs/**`, `tests/**`, `tests/fixtures/**` | report-only 산출물이 live execution payload나 live trading branch를 만들지 않는다는 감사 문맥에서만 허용된다. |
| `BUY`, `SELL`, `HOLD` | deterministic signal engine tests, report/schema tests, policy docs | 최종 action label은 deterministic signal engine의 출력으로만 허용된다. LLM/model fixture에 나타나는 경우에는 신호 격리 테스트의 부정 예제로만 허용된다. |

### Narrow Policy-Language Exceptions

Allowlisted policy language is intentionally narrow. The exception applies only
to non-executable text that proves or explains a prohibition:

| Exception type | Allowed paths or surfaces | Required invariant |
| --- | --- | --- |
| Non-executable guidance | `docs/**` policy, architecture, risk, and validation guidance | The text must describe scope limits, audit commands, or expected absence of live-trading/order behavior. |
| Validation audit text | `tests/**` assertions and synthetic unsafe examples | The text must be part of a guard that fails if production code gains broker/order modules, API calls, payload fields, or runtime branches. |
| Report evidence text | `tests/fixtures/**` and report/schema tests | The text must verify report-only isolation or deterministic action-label provenance; generated production reports must not introduce broker/order payloads. |
| Final-action labels in evidence | deterministic signal engine tests, report/schema tests, and policy docs | `BUY`, `SELL`, and `HOLD` may appear only to prove that final labels come from deterministic signal generation, not model/LLM text. |

The exception does not cover `app.py` or `src/quant_research/**/*.py`.
Executable-path enforcement remains hard-fail through
`tests/test_architecture_guards.py`: production code must not contain
live-trading adapters, broker/order API calls, live execution payload fields, or
runtime branches for orders. These documentation, report, and validation-text exceptions are evidence-only and have no validation impact.

### Forbidden Term Invariants

금지된 broker/order 계열 용어가 문서, 테스트, fixture allowlist에 나타나는
경우에도 아래 불변식은 유지되어야 한다.

- Production source under `src/quant_research` must not define live-trading,
  broker, order, execution, or trading modules and must not call order API
  phrases such as `place_order`, `submit_order`, `create_order`,
  `market_order`, or `limit_order`.
- Allowlisted references are evidence text only. They may describe prohibited
  behavior or assert its absence, but they must not add executable adapters,
  payload fields, side effects, or runtime branches for live orders.
- Model and LLM outputs remain structured research features. Even if an
  allowlisted test fixture contains `BUY`, `SELL`, or `HOLD` strings in model
  text, those strings are not read as final actions.
- Final action labels are emitted only by the deterministic signal engine after
  score, cost/slippage, turnover, walk-forward, out-of-sample, and risk-rule
  checks.
- Report-only diagnostics, including forbidden-term audit evidence, must not
  change signal scores, action labels, gate pass/fail decisions, backtest
  returns, model predictions, or published report metrics.

Expected validation impact: none. A change that only adds or edits allowlisted
audit wording must leave validation behavior, signal semantics, model
predictions, backtest results, and report metrics unchanged.

## Optional Dependency Warning Baseline

현재 검증 workflow는 optional dependency upgrade 경고를 허용된
non-semantic baseline evidence로 기록한다. 이 baseline은 경고를 숨기거나
validation 결과로 승격하지 않으며, dependency upgrade 때 제거 대상을
추적하기 위한 감사 증거다.

Reproducible command:

```bash
uv --cache-dir .uv-cache run pytest -q
```

Observed baseline on 2026-05-07 after merging all active validation branches,
fixing pandas SEC feature call sites, and adding audit-evidence reproduction
documentation:

- `832 passed`
- `12 warnings`
- Affected production path: none
- Affected tests: `tests/test_model_adapters.py` (2 warnings),
  `tests/test_report_generation.py` (10 warnings)

Documented warning families:

| Source | Warning class | Count | Invariant |
| --- | --- | ---: | --- |
| `.venv/lib/python3.12/site-packages/sklearn/utils/validation.py:2749` | sklearn `UserWarning` for LightGBM prediction inputs without fitted feature names | 12 | Optional LightGBM-backed tests exercise prediction paths; warning is dependency-surface evidence only. |

Pandas compatibility fixes applied in `src/quant_research/features/sec.py`:

| Source | Fix | Invariant |
| --- | --- | --- |
| SEC numeric fill call sites | Convert merged object columns with `pd.to_numeric(...).fillna(0.0)` before rolling calculations. | SEC feature missing values stay neutral and point-in-time. |
| SEC timestamp forward-fill call sites | Normalize timestamp columns with `timestamp_utc(...).ffill()` before propagation. | SEC timestamp propagation remains point-in-time. |
| SEC per-ticker concatenation | Normalize SEC numeric and timestamp dtypes before concatenating ticker frames. | Empty SEC inputs remain neutral and sorted by date/ticker. |

Intentionally addressed in this hardening pass:

- Added this validation-suite inventory as auditable evidence.
- Fixed pandas SEC feature call sites without changing generated values.
- Added tests that assert the documented warning counts, sources, and invariants
  stay visible during review.
- Did not suppress, filter, or silence pandas warnings; the SEC call sites now
  avoid the pandas deprecations directly.
- Did not change SEC feature semantics, walk-forward preprocessing,
  signal semantics, model predictions, backtest results, or report metrics.

Allowed warning evidence types:

- pytest warning summary from the command above
- documentation references in this section
- tests that assert the warning baseline is documented

Production-enforced impact: none. These warnings do not add hard enforcement
inside production paths and do not change signal labels, validation gate status,
model predictions, backtest returns, report metrics, SEC EDGAR request behavior,
or report-only artifact generation.

### Warning Triage And Semantic Impact

This hardening pass separates reduced warnings from intentionally tracked
warnings so reviewers can audit the remaining pytest baseline without treating
it as validation evidence drift.

Warnings reduced:

| Warning family | Affected paths | Reduction reason | Semantic impact |
| --- | --- | --- | --- |
| pandas object-dtype fill/forward-fill deprecations in SEC feature generation | `src/quant_research/features/sec.py`, `tests/test_sec_features.py` | SEC numeric and timestamp columns are normalized before fill and rolling operations. | None: missing SEC numeric values remain neutral, SEC timestamps remain point-in-time, and no feature/date ordering rule changes. |
| downstream validation-suite warning count | pytest full suite | Removing the pandas deprecations leaves the expected baseline at the documented sklearn LightGBM `UserWarning` rows. | None: the reduction removes compatibility noise only; it does not re-score signals, re-gate candidates, or alter report metrics. |

Warnings intentionally tracked:

| Warning family | Affected paths | Why it remains tracked | Removal trigger |
| --- | --- | --- | --- |
| sklearn `UserWarning` for LightGBM prediction inputs without fitted feature names | dependency source `.venv/lib/python3.12/site-packages/sklearn/utils/validation.py:2749`; observed through `tests/test_model_adapters.py` and `tests/test_report_generation.py` | Optional LightGBM-backed tests exercise prediction paths where sklearn emits feature-name metadata warnings. The warning is dependency-surface evidence, not a validation status. | Remove this baseline only when sklearn/LightGBM or the test helpers no longer emit the warning while preserving prediction-output assertions. |

Why semantics are unchanged:

- The reduced pandas warnings are addressed by dtype normalization before the
  same SEC neutral-value and timestamp-propagation operations; no thresholds,
  formulas, validation gates, signal labels, or report fields are changed.
- The intentionally tracked sklearn warning remains in documentation and tests
  only. It is not filtered, promoted to a gate result, or used as an input to
  model prediction, signal generation, backtest accounting, or report metric
  calculation.
- The expected validation impact for both categories is none: signal semantics,
  model predictions, backtest results, report metrics, SEC EDGAR request policy,
  and report-only artifact generation remain unchanged.

## Stage 1 Validity Gate Schema Contract

Stage 1 defines the comparison contract before expanding model complexity. The
machine-readable `validity_gate.json` and human-readable report include:

- full model input schema: structured price/text/SEC plus optional Chronos-2,
  Granite TTM, FinBERT, FinMA, FinGPT, and Ollama adapters with rules fallback
- baseline input schema: no-model-proxy model baseline, SPY market benchmark,
  and equal-weight universe benchmark
- ablation input schema: price/text/SEC channels, no-cost diagnostic, and named
  model/modality ablations
- metric schema: rank IC, positive fold ratio, OOS rank IC, Sharpe, max drawdown,
  cost-adjusted cumulative return, excess return, and turnover
- validation window schema: walk-forward folds, OOS holdout, gap/embargo, target
  horizon, and t+1-or-later return timing
- comparison result rows: per-window full-model comparisons against model
  baselines, return baselines, and ablations with absolute delta, relative delta,
  and direction-aware pass/fail status
- explicit metric contract rows: top-level full-model metrics, baseline metrics,
  ablation metrics, and structured pass/fail reasons mirrored into JSON
- rule result explanations: normalized rows for each gate with status, pass/fail
  boolean when applicable, reason, reason code, metric, threshold, operator, and
  whether the rule affects final strategy status
- final strategy status explanation: one summary object tying system validity,
  blocking rules, warning rules, insufficient-data rules, and official operator
  message to the final strategy candidate status

## Six-Stage Roadmap

1. Stage 1 Validity Gate: schema contract, JSON/Markdown/HTML/Streamlit outputs,
   and tests for comparison inputs/results.
2. Stage 2 Data Timing: stricter provider contracts, publication-time alignment,
   and feature availability checks.
3. Stage 3 Model Adapter Evidence: optional heavy adapter preload path, fallback
   parity tests, and structured text/model feature provenance.
4. Stage 4 Signal And Risk Realism: deterministic signal scoring, costs,
   slippage, turnover, drawdown, and benchmark gates.
5. Stage 5 Ablation And Incremental Value: Chronos/Granite/LLM/model-channel
   ablations with material improvement thresholds.
6. Stage 6 Reporting And Reproducibility: reproducible artifacts, operator
   reports, run manifests, and release-ready validation summaries.
