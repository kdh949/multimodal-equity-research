# Agent Development Guide

이 저장소는 미국 주식 멀티모달 퀀트 리서치 앱을 만든다. 모든 에이전트와 개발자는 아래 원칙을 지켜야 한다.

## Non-Negotiables

- 실거래 주문 기능은 구현하지 않는다. v1은 리서치, 백테스트, 검증, 리포트까지만 다룬다.
- LLM은 매수/매도/관망 결정을 내리지 않는다. 최종 신호는 deterministic signal engine이 만든다.
- 모델 예측값은 바로 주문 신호가 아니다. 예측값은 점수화, 비용, 슬리피지, 리스크 룰, walk-forward, out-of-sample 검증을 통과해야 한다.
- 미래 데이터 누수를 금지한다. 피처는 t 시점까지의 데이터만 사용하고, 포지션은 t+1 이후 수익률에 적용한다.
- SEC EDGAR 요청은 명시적 User-Agent와 10 req/s 이하 rate limit을 사용한다.
- API 키, 원천 데이터 캐시, 모델 아티팩트, 리포트 산출물은 커밋하지 않는다.

## Architecture Rules

- UI는 Streamlit에 두고, 데이터/모델/검증/신호 로직은 `src/quant_research/` 패키지에 둔다.
- 가격 데이터, 텍스트 데이터, SEC 데이터는 provider 인터페이스 뒤에 둔다.
- Chronos-2, Granite TTM, FinBERT, FinMA, FinGPT, Ollama는 optional adapter로 구현한다. 의존성이 없거나 모델이 내려받아지지 않은 환경에서도 테스트는 돌아가야 한다.
- 실제 heavy model 추론은 `scripts/preload_local_models.py`로 모델 캐시를 준비한 뒤 명시적으로 켠다. 기본 테스트와 CI는 proxy/rules fallback 경로를 유지한다.
- LightGBM이 없으면 scikit-learn gradient boosting fallback을 사용한다.
- 텍스트 모델 출력은 자유 서술 대신 `sentiment_score`, `event_tag`, `risk_flag`, `confidence`, `summary_ref` 같은 구조화 feature로 저장한다.

## Parallel Agent Development

- 작업을 병렬로 나눌 수 있고, 각 작업이 간단하며 GPT-5.3-Codex-Spark가 구현하기 적합한 범위라면 여러 개의 멀티 에이전트를 생성해 개발해도 된다.
- 병렬 에이전트에는 `GPT-5.3-Codex-Spark` 모델과 매우 높은 추론 성능 설정을 우선 사용한다.
- 병렬화는 파일/모듈 소유권이 분리되는 경우에만 사용한다. 예: provider 테스트, 모델 adapter 테스트, 문서 보강처럼 write scope가 서로 겹치지 않는 작업.
- 각 에이전트에게는 담당 파일 또는 담당 서브시스템을 명확히 지정하고, 다른 에이전트나 사용자의 변경을 되돌리지 말라고 지시한다.
- 설계 판단, 공통 인터페이스 변경, 충돌 가능성이 큰 통합 작업, 리스크/검증 의미가 바뀌는 작업은 메인 에이전트가 직접 처리한다.
- 병렬 에이전트 결과는 메인 에이전트가 최종 리뷰, 통합, 테스트, 커밋 정리를 수행한 뒤 반영한다.

## Validation Flow

1. 모델이 수익률, 변동성, 분위수, 텍스트 리스크 예측값을 생성한다.
2. 예측값을 바로 주문하지 않는다.
3. feature를 deterministic signal score로 변환한다.
4. 거래비용, 슬리피지, turnover를 반영한다.
5. walk-forward 검증을 수행한다.
6. out-of-sample 성능을 확인한다.
7. 리스크 룰을 통과한 경우에만 `BUY`, `SELL`, `HOLD` 신호를 만든다.

## Git Workflow

- `main`은 항상 실행 가능한 기준선으로 유지한다.
- 기능 작업은 `feature/<name>` 브랜치에서 진행한다.
- 커밋 메시지는 `<type>: <한국어 제목>` 형식을 사용하고 스코프는 쓰지 않는다.
- 허용 타입: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `style`, `perf`, `build`, `ci`, `revert`.
- 모호한 커밋 제목을 금지한다. 예: `fix: 버그 수정` 대신 `fix: SEC 공시 이벤트 날짜 정렬 오류 수정`.

## Verification Commands

```bash
uv --cache-dir .uv-cache sync --all-extras
uv --cache-dir .uv-cache run pytest
uv --cache-dir .uv-cache run streamlit run app.py
```

네트워크나 heavy model 의존성이 없는 환경에서는 아래 최소 검증을 우선한다.

```bash
python3 -m pytest
```
