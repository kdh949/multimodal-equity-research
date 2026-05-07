# 테스트 가이드 (Tests Guide)

본 문서는 `tests/` 폴더에 포함된 테스트 스위트의 목적, 카테고리별 설명, 실행 방법, 그리고 **결과를 해석하는 방법**을 설명합니다.
이 프로젝트(`quant-research`)는 **로컬 멀티모달 미국 주식 정량 연구·백테스트 시스템**으로, 테스트는 단순한 단위 테스트를 넘어 **연구 결과의 신뢰성·재현성·정책 준수**를 강제하는 *게이트(gate)* 역할을 합니다.

---

## 1. 빠른 시작

### 1.1 전체 테스트 실행

```bash
# 프로젝트 루트에서
pytest -q
```

### 1.2 카테고리별 실행

```bash
# 유효성 게이트(Validity Gate) 관련 테스트만
pytest tests/test_validity_gate*.py -q

# 백테스트 관련 테스트만
pytest tests/test_backtest_*.py tests/test_walk_forward.py -q

# 단일 테스트 + 자세한 로그
pytest tests/test_validity_gate.py::test_validity_gate_passes_structural_contract_for_clean_inputs -vv
```

### 1.3 보고 옵션

```bash
# 실패한 첫 케이스에서 즉시 중단
pytest -x

# 실패 추적용 짧은 traceback
pytest --tb=short

# 가장 느린 테스트 10개 출력
pytest --durations=10
```

> **참고**: `pyproject.toml`에 `testpaths = ["tests"]`, `pythonpath = ["src"]`가 정의되어 있으므로 별도 설정 없이 루트에서 `pytest`만 실행하면 됩니다. `conftest.py`는 OpenMP·Accelerate·VECLIB 등 멀티스레딩 환경 변수를 결정론적 기본값으로 고정합니다.

---

## 2. 테스트 카테고리 개요

`tests/` 폴더의 ~55개 테스트 파일은 **8개 도메인**으로 구분됩니다. 도메인별로 어떤 *불변식(invariant)* 을 강제하는지 아래 표로 요약합니다.

| 카테고리 | 파일 접두 | 보장하는 핵심 불변식 |
|---|---|---|
| 유효성 게이트 | `test_validity_gate*`, `test_gate_status_policy`, `test_deterministic_gate_interface`, `test_system_validity_gate_*_schema` | 모델/전략은 정해진 통계·리스크 임계를 만족할 때만 *통과(pass)* 한다 |
| 백테스트·검증 | `test_backtest_*`, `test_walk_forward`, `test_evaluation_intervals`, `test_validation_*`, `test_pipeline` | 학습-테스트 분리, purge/embargo, 거래비용 반영이 정확하다 |
| 데이터·피처 | `test_data_providers`, `test_provider_timestamp_normalization`, `test_price_features`, `test_sec_features`, `test_feature_leakage_guards`, `test_universe`, `test_text_models`, `test_covariance_inputs`, `test_benchmark_inputs` | 미래 정보가 과거 피처에 누설되지 않는다 |
| 모델 | `test_model_*`, `test_preload_local_models`, `test_signal_engine`, `test_no_model_proxy_ablation`, `test_strategy_candidate_policy` | 모델 어댑터 계약이 일관되며 프록시-온리 사용이 감지된다 |
| Ablation·민감도 | `test_ablation_*`, `test_sensitivity_batch` | 피처/비용 시나리오 변동 시 비교 가능한 메트릭이 산출된다 |
| 보고서·대시보드 | `test_report_*`, `test_artifact_manifest_builder`, `test_validity_dashboard`, `test_beginner_dashboard`, `test_comparison_schemas` | 산출물(보고서·매니페스트·대시보드)이 스키마를 따른다 |
| 의미 안전성·정책 | `test_semantic_safety_*`, `test_policy_language_guardrails`, `test_warning_baseline_documentation` | 실거래(execute/broker/order) 어휘가 코드·문서에 등장하지 않는다 |
| 앱·런타임·아키텍처 | `test_app_*`, `test_runtime`, `test_architecture_guards`, `test_report_only_execution_isolation`, `test_portfolio_risk_metrics`, `test_return_series_metrics` | 앱이 부팅되고, 모듈 경계가 유지되며, 메트릭이 정확하다 |

---

## 3. 카테고리별 상세 설명

### 3.1 유효성 게이트(Validity Gate)

이 시스템에서 **가장 중요한 테스트군**입니다. 검증 게이트는 백테스트가 끝난 뒤 *“이 전략을 보고서 단계로 넘겨도 되는가?”* 를 결정론적으로 판정합니다.

| 파일 | 검증 내용 |
|---|---|
| `test_validity_gate.py` | 게이트의 기본 계약: 깨끗한 입력에서 `system_validity_pass=True`, 위반 시 `hard_fail` |
| `test_validity_gate_engine_cases.py` | pass/warning/hard_fail 세 갈래의 상태 집계 로직 |
| `test_validity_gate_metric_formulas.py` | Rank IC, Sharpe, 회전율 등의 **수식**이 정의대로 계산되는지 |
| `test_validity_gate_combination_matrix.py` | 여러 규칙 조합 매트릭스에서 최종 상태가 일관되는지 |
| `test_validity_gate_comparison.py` | 동일가중·SPY 등 baseline 대비 비교 결과 |
| `test_validity_gate_outputs.py` | 대시보드용 출력 스키마(메타·지표·룰 결과) |
| `test_validity_gate_synthetic_metric_tables.py` | 합성 데이터에 대한 end-to-end 게이트 실행 |
| `test_validity_gate_turnover.py` | 회전율과 비용 시나리오 반영 |
| `test_validity_gate_system_status.py` | 시스템 단위 상태(여러 룰 → 최종 pass/fail) 집약 |
| `test_gate_status_policy.py` | `pass`/`warn`/`hard_fail` 정규화·우선순위 정책 |
| `test_deterministic_gate_interface.py` | provider-free 결정론적 인터페이스 보존 |
| `test_system_validity_gate_input_schema.py` / `_output_schema.py` | I/O 스키마 필드 누락 방지 |

**해석 원칙**: 이 카테고리에서 **단 하나라도 실패하면**, 게이트의 신뢰성 자체가 무너진 것이므로 다른 결과(예: 백테스트 메트릭)도 *재현·해석할 수 없다*고 보아야 합니다.

### 3.2 백테스트·검증

| 파일 | 검증 내용 |
|---|---|
| `test_backtest_alignment.py` | 신호 일자, 보유기간, 라벨이 정확히 정렬되는지 |
| `test_backtest_turnover.py` | 거래량과 거래비용 시나리오의 정확성 |
| `test_backtest_risk.py` | 종목/포트폴리오 가중치 상한, 변동성 제약 강제 |
| `test_walk_forward.py` | 학습/테스트 윈도, purge, embargo 기본값 |
| `test_evaluation_intervals.py` | 종목·전략별 비중복 평가 구간 생성 |
| `test_validation_config.py` | 모든 baseline·비용 시나리오가 설정에 존재 |
| `test_validation_horizons.py` | 1d/5d/20d 등 표준 지평선과 gap/embargo |
| `test_pipeline.py` | 데이터→피처→신호 파이프라인 end-to-end |

### 3.3 데이터·피처

미래 정보 누설(look-ahead leakage)은 정량 연구에서 **가장 흔한 치명적 오류**입니다. 이 카테고리는 이를 자동으로 차단합니다.

특히 `test_feature_leakage_guards.py`와 `test_price_features.py`가 핵심입니다. 이 둘이 통과한다는 것은 *t* 시점 피처 계산에 *t* 이후의 가격·뉴스·SEC 데이터가 사용되지 않는다는 뜻입니다.

### 3.4 모델

`test_model_adapters.py`는 TabularReturnModel·FinBERT·FinGPT·Chronos·Granite 어댑터의 **공통 인터페이스**를 강제합니다. `test_no_model_proxy_ablation.py`는 *“실모델 없이 프록시만으로 통과되는 부정행위”* 를 차단합니다.

### 3.5 Ablation·민감도

`test_ablation_scenarios.py`는 12가지 표준 ablation(가격만, 텍스트만, no-model 등)을 정의하고, `test_ablation_metric_comparability.py`는 ablation 간 메트릭이 서로 비교 가능한 척도(예: 동일 평가구간·동일 비용)인지 검증합니다.

### 3.6 보고서·대시보드

`test_report_schema.py`와 `test_artifact_manifest_builder.py`는 보고서 산출물이 *재현 가능한 매니페스트*(원본 데이터 소스 + 코드 버전 + 설정 해시)를 동반하는지 강제합니다. `test_validity_dashboard.py`·`test_beginner_dashboard.py`는 Streamlit 대시보드의 입력 정규화를 검증합니다.

### 3.7 의미 안전성·정책

본 시스템은 **연구 전용**이며 실거래(라이브 트레이딩)를 수행하지 않습니다. 그래서 코드·문서 어디에도 `broker`, `place_order`, `execute_trade` 등의 어휘가 등장해선 안 됩니다. `test_semantic_safety_hardening_evidence.py`·`test_policy_language_guardrails.py`가 이 정책을 정적 감사합니다. `test_warning_baseline_documentation.py`는 “용인된 경고”의 화이트리스트 문서가 최신 상태인지 점검합니다.

### 3.8 앱·런타임·아키텍처

`test_app_smoke.py`·`test_app_regression_001.py`는 Streamlit 앱이 부팅되고 알려진 시나리오에서 동일한 결과를 내는지 확인합니다. `test_architecture_guards.py`는 모듈 경계(예: 정책 코드는 `policy/` 모듈에만 존재)를 강제합니다.

---

## 4. 결과 해석 방법

### 4.1 Pytest 출력 한눈에 읽기

표준 출력은 다음 형식입니다.

```
tests/test_validity_gate.py ........F.....                       [ 25%]
tests/test_backtest_risk.py ...x.s..........                     [ 47%]
=================== FAILURES ===================
___________ test_validity_gate_hard_fails_horizon_embargo_violation ___________
...
=========== 1 failed, 312 passed, 4 skipped, 1 xfailed in 28.41s ===========
```

각 점·문자의 의미:

| 기호 | 의미 | 조치 |
|---|---|---|
| `.` | pass | 정상 |
| `F` | fail | **즉시 조사**. 어떤 불변식이 깨졌는지 traceback 확인 |
| `E` | error (테스트 코드 자체 예외) | 픽스처/임포트 문제 — 코드 변경이 인터페이스를 깬 경우 |
| `s` | skipped | 의도적 건너뛰기. 이유가 표시되는지 확인 |
| `x` | expected fail (xfail) | 알려진 버그·미구현. 정상 |
| `X` | unexpectedly passed (xpass) | xfail 마커를 제거할 때 |
| `w` | warning 발생 | 경고 메시지 검토 권장 |

### 4.2 도메인별 “실패가 의미하는 것”

| 실패 카테고리 | 신호 | 권장 대응 |
|---|---|---|
| 유효성 게이트 | 게이트 자체의 수식·스키마가 깨짐 | 다른 테스트 결과 신뢰 금지. 게이트부터 복구 |
| 백테스트·검증 | 학습/테스트 누수, purge·embargo 위반 | 데이터 정렬·시간 처리부터 의심 |
| 데이터·피처 | look-ahead leakage 가능성 | 새 피처·롤링 윈도 코드 즉시 검토 |
| 모델 | 어댑터 인터페이스 변경 | 모델 추가·업그레이드 시 가장 흔함 |
| Ablation·민감도 | 시나리오 간 비교 불가 상태 | 메트릭·평가 구간 정규화 확인 |
| 보고서·대시보드 | 산출물 스키마/매니페스트 누락 | 재현성 손상. 보고서 발행 보류 |
| 의미 안전성·정책 | 실거래 어휘·금지 패턴 유입 | 즉시 어휘 제거 (PR 차단 사유) |
| 앱·런타임·아키텍처 | 부팅 실패·모듈 경계 위반 | 의존성·import 그래프 점검 |

### 4.3 “경고이지만 통과”의 의미

`warning=True`이면서 `hard_fail=False`인 게이트 결과는 **결과를 사용해도 좋지만, 보고서에 경고 사유가 명시되어야** 합니다. `test_warning_baseline_documentation.py`가 이 사유 목록을 강제하므로, 새로운 경고를 도입할 때는 baseline 문서도 함께 업데이트해야 합니다.

### 4.4 재현성 점검

테스트가 **로컬에서는 통과하지만 CI에서는 실패**한다면 다음을 우선 확인합니다.

```bash
# 환경 변수가 conftest 기본값과 같은지
env | grep -E "OMP|KMP|MKL|VECLIB|OPENBLAS"

# 캐시·아티팩트가 오래된 결과를 가리지 않는지
rm -rf .pytest_cache artifacts/cache
pytest -q
```

`test_runtime.py`가 통과한다면 멀티스레딩 비결정성은 배제할 수 있습니다.

### 4.5 흔한 실패 패턴과 디버깅

1. **`AssertionError: hard_fail is True` (실수로 hard_fail이 발생)**
   → 입력 데이터의 NaN·시간 정렬을 먼저 의심. `test_provider_timestamp_normalization.py`도 함께 실행.
2. **`KeyError` in 보고서 스키마 테스트**
   → 새 메트릭을 추가하면서 스키마 정의를 빠뜨린 경우. `quant_research/validation/gate.py`의 출력 dict 키 갱신.
3. **`test_feature_leakage_guards.py` 실패**
   → 새 롤링/시프트 연산이 `shift(-k)`처럼 미래를 참조했을 가능성. 단위 테스트 케이스의 입력 시계열을 출력해 스왑된 인덱스를 확인.
4. **`test_semantic_safety_*` 실패**
   → README·docstring·로그 문자열에 금지어가 들어간 경우. `git grep -nE "place_order|broker|execute_trade"` 로 위치 파악.

---

## 5. 새 테스트를 추가할 때

새 기능을 추가할 때는 **카테고리 일관성**을 유지합니다.

새 데이터 제공자를 추가했다면 `test_data_providers.py`·`test_provider_timestamp_normalization.py`·`test_feature_leakage_guards.py`에 케이스를 추가하고, 새 모델 어댑터를 추가했다면 `test_model_adapters.py`·`test_no_model_proxy_ablation.py`에 등록합니다. 새 룰을 게이트에 추가했다면 `test_validity_gate_combination_matrix.py`의 매트릭스를 갱신해야 합니다.

테스트 데이터는 `tests/fixtures/` 하위 합성 데이터 또는 `conftest.py` 픽스처를 재사용합니다. **실시장 데이터를 테스트에 직접 끌어오면 결정론이 깨집니다.**

---

## 6. CI/CD 통합 권장 사항

```yaml
# 예시: .github/workflows/tests.yml
- name: Run safety + leakage gates first (fail fast)
  run: pytest tests/test_semantic_safety_*.py tests/test_feature_leakage_guards.py tests/test_architecture_guards.py -q

- name: Run validity gate suite
  run: pytest tests/test_validity_gate*.py tests/test_gate_status_policy.py -q

- name: Run remaining suite
  run: pytest --ignore=tests/test_validity_gate*.py --ignore=tests/test_semantic_safety_*.py -q
```

이 순서는 **가장 치명적인 정책·누설 검사 → 게이트 → 나머지** 의 우선순위를 따릅니다. 앞 두 단계가 실패하면 뒤를 실행할 의미가 없습니다.

---

## 7. 요약

이 테스트 스위트의 목적은 단순히 “코드가 동작하는가?”가 아니라 *“이 시스템이 산출한 연구 결과를 신뢰해도 되는가?”* 를 자동으로 답하는 것입니다. 따라서 통과(`pass`)는 단순한 그린 라이트가 아니라 **데이터 누설 없음 + 수식·스키마 일치 + 정책 준수 + 재현 가능 매니페스트** 라는 네 가지 보증을 동시에 의미합니다. 어떤 테스트가 실패하든, 그 실패는 *해당 카테고리의 보증 하나가 무너졌음* 을 뜻하므로 보고서·대시보드의 결과를 사용하기 전에 반드시 복구해야 합니다.
