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

- 멀티 타임스케일(예: 1d/5d/20d/1m) 예측 아웃풋 저장
- Embargo 구간과 purge window 고정 정책 설정
- Fold 간 leakage 경로 재점검
- 다중 타깃에서도 동일 gate 계약이 유지되는지 확인

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

## 실행 오케스트레이션 정책
- 메인 에이전트가 전체 작업 흐름을 분해해 Stage별 책임자를 배정한다.
- 단순하고 분리 가능한 작업은 GPT-5.3-Codex-Spark `xhigh` 워커로 병렬 실행한다.
- 메인 에이전트가 최종 설계 판단, 통합, 테스트 실행, 충돌 조정, 병합을 수행한다.
- Scope 충돌 가능성이 있으면 즉시 통합 중단 후 설계 재합의.

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
