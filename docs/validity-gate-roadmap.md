# Validity Gate Stage 1~6 로드맵

이 문서는 US 멀티모달 퀀트 리서치 앱에서 생성되는 Validity Gate seed의 실행 계획이다.

## Stage 1 (1차 OOO Run Scope)
가장 먼저, 모델 학습/백테스트 전반을 Gate로 통제한다.

- **초점**: `Validity Gate outputs`, `contract`, `tests` 정의 및 자동 검증
- **출력물**
  - `validity_gate_summary.json`: run id, 입력 범위, 누수 검사 결과, OOS pass 여부, pass/fail 이유
  - `validity_gate_contract.md`(또는 스키마 파일): 필수 필드/포맷/허용 범위
  - `validity_gate_test_report.json`: 정합성 테스트/실패 케이스/재현성 해시
- **테스트**
  - 스키마 적합성 테스트 (필수 필드/타입/범위)
  - 타임라인 누수 검사 테스트 (feature 시점 ≤ t, label 적용 시점 = t+1)
  - OOS/Walk-forward fold 완성도 테스트
  - 비용·슬리피지·turnover 파라미터 존재성 테스트
- **승인 조건**
  - Gate 실패 시 Stage 2 이상 진입 금지
  - stage 1은 seed의 “출력-계약-검증 루프”가 완전 동작할 때만 통과

## Stage 2 — 멀티타겟 + Embargo/Purge
예측 타깃을 1일 외에 5일/20일 등으로 확장하고, 데이터 분할 시 lookahead를 막기 위해 embargo/purge를 적용한다.

- `forward_return_1`, `forward_return_5`, `forward_return_20` 라벨을 생성하되, 한 번의 파이프라인 실행은 설정된 `prediction_target_column` 하나만 학습/검증 타깃으로 사용한다.
- 라벨 컬럼은 모델 입력 feature에서 항상 제외하고, 리포트/coverage metadata로만 추적한다.
- `requested_gap_periods`, `requested_embargo_periods`는 사용자가 요청한 값으로 보존한다.
- 실제 분할에는 `effective_gap_periods=max(requested_gap_periods, target_horizon)`, `effective_embargo_periods=max(requested_embargo_periods, target_horizon)`를 적용한다.
- 요청 gap/embargo가 horizon보다 짧으면 warning으로 기록하되, effective 값이 horizon 이상이면 hard fail로 보지 않는다.
- train label interval이 test label interval과 겹치지 않도록 purge 검사를 수행하고, 남은 overlap은 hard fail 처리한다.
- 백테스트 실현 수익률, SPY benchmark, equal-weight benchmark는 모두 선택된 horizon의 `forward_return_*` 컬럼을 사용한다.
- Stage 2 통과 후 Stage 3 유니버스 확장으로 넘어간다.

## Stage 3 — 유니버스 확장
기초 종목군을 벗어나 표준 유니버스(예: S&P 500 하위, 섹터 분할)로 확장한다.

- 상장사/ETF/섹터별 누락 규칙을 문서화
- 소액·비유동 종목 제외 규칙 정교화
- 시그널/피처 파이프라인이 종목 수 증가 시에도 동일하게 동작하는지 성능/안정성 점검

## Stage 4 — 검증된 Deterministic Signal Rules
예측값을 직접 주문 신호로 쓰지 않고, 규칙 기반 점수화로 결정한다.

- 리스크 룰, 비용 룰, turnover 룰을 일관되게 적용한 점수 계산식 확정
- `BUY/SELL/HOLD`는 gate 통과 후에만 생성
- 규칙 변경 시 `docs` + `tests`로 계약 기반 버전관리
- OOS 성능 저하 시 규칙을 즉시 롤백할 수 있는 절차 정의

## Stage 5 — 상관관계 인식 포트폴리오
종목 간 상관/공분산을 반영해 과도한 집중을 억제한다.

- 상관 임계치 기반 포지션 상한 규칙 적용
- sector/스타일 중복 노출 제어
- 포트폴리오 최적화 전 단계에서 risk-budget 기반 필터 추가
- correlation-aware 규칙도 gate 계약 안에서 추적(재현성 hash/입력/출력)

## Stage 6 — Chronos/Granite/LLM Ablation & Value Proof
복잡 모델 의존도를 수치로 검증하고, 실제 가치 기여를 증명한다.

- Chronos-2, Granite TTM, FinBERT 계열 adapter별 ablation 비교
- 텍스트 감성/이벤트/리스크 피처만 유무 조합 실험
- “LLM은 신호 결정이 아닌 feature 생성 보조” 가정의 실증
- value proof: 비용·슬리피지·OOS 성능에서 개선폭이 있는 조합만 채택

## 실행 원칙
- 각 Stage는 이전 Stage의 산출물/계약/테스트가 통과했다는 전제에서 순차 실행한다.
- Stage별 변경은 기능 또는 의미 단위로 커밋한다.
- Scope 충돌 가능성이 있으면 즉시 구현을 중단하고 설계를 재확인한다.

## 운영 제약
- 실거래 주문 기능은 구현하지 않는다. v1은 리서치/백테스트/검증/리포트까지.
- LLM은 `BUY/SELL/HOLD` 최종 결정을 내리지 않는다.
- 텍스트 모델 출력은 구조화 feature(예: `sentiment_score`, `event_tag`, `risk_flag`)로만 사용한다.

## 커밋 규약
커밋 메시지 형식은 다음만 허용한다.

```text
<type>: <한국어 제목>
```

허용 `type`:
`feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `style`, `perf`, `build`, `ci`, `revert`

스코프(`feat: ui/...`)는 쓰지 않는다.
