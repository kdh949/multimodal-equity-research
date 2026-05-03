# Modeling

## Price Models

### Chronos-2

수익률, 변동성, 분위수 예측을 위한 time-series foundation model adapter로 둔다. v1에서는 optional dependency로 구현하며, 모델이 없으면 비활성화한다.

### LightGBM / XGBoost / CatBoost

전통적인 tabular ML baseline이다. 기본 우선순위는 다음과 같다.

1. LightGBM
2. XGBoost
3. CatBoost
4. scikit-learn fallback

baseline은 feature importance와 빠른 walk-forward 검증에 사용한다.

### Granite TTM

초경량 시계열 모델 adapter다. 빠른 로컬 실험, intraday PoC, daily PoC에 사용한다.

## Text Models

### FinBERT

금융 뉴스와 헤드라인의 positive, neutral, negative score를 산출한다. 앱 feature는 `sentiment_score`, `negative_ratio`, `confidence`로 정규화한다.

### FinMA / FinGPT

SEC 공시, 실적발표, 긴 문서의 event extraction adapter로 둔다. 모델 출력은 자유 문장이 아니라 구조화 JSON으로 검증한다.

### Ollama Agent

기본 로컬 모델은 `qwen3-coder:30b`로 둔다. 역할은 데이터 조회 보조, 모델 호출 상태 설명, 결과 리포트 요약이다. 매수/매도 판단은 하지 않는다.

## Targets

- `forward_return_1`: 다음 기간 수익률
- `realized_volatility`: rolling realized volatility
- `downside_quantile`: 하방 분위수 proxy 또는 Chronos quantile
- `risk_score`: 가격/텍스트/SEC feature 결합 리스크 점수

## Feature Fusion

날짜와 티커를 기준으로 가격 feature, 모델 예측값, 뉴스 feature, SEC feature를 left join한다. 텍스트 feature는 event timestamp 이후 다음 거래 가능 시점부터 적용한다.
