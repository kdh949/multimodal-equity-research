# Modeling

## Price Models

### Chronos-2

수익률, 변동성, 분위수 예측을 위한 time-series foundation model adapter로 둔다. 기본 UI는 `proxy` feature를 사용하지만, `Time-series inference=local`을 선택하면 `chronos.Chronos2Pipeline.from_pretrained("amazon/chronos-2")`와 `predict_df`를 호출해 `chronos_expected_return`, `chronos_downside_quantile`, `chronos_upside_quantile`, `chronos_quantile_width`를 채운다.

운영 원칙:

- 로컬 추론은 optional dependency(`chronos-forecasting>=2.0`)가 있을 때만 사용한다.
- 예측 윈도우는 UI의 `Local TS inference windows`로 제한해 노트북 환경에서 과도한 추론을 막는다.
- 모델 로딩/추론이 실패하면 기존 proxy feature로 fallback한다.

### LightGBM / XGBoost / CatBoost

전통적인 tabular ML baseline이다. 기본 우선순위는 다음과 같다.

1. LightGBM
2. XGBoost
3. CatBoost
4. scikit-learn fallback

baseline은 feature importance와 빠른 walk-forward 검증에 사용한다.

### Granite TTM

초경량 시계열 모델 adapter다. 빠른 로컬 실험, intraday PoC, daily PoC에 사용한다. 로컬 모드에서는 `sktime.forecasting.ttm.TinyTimeMixerForecaster`로 `ibm-granite/granite-timeseries-ttm-r2`를 불러와 ticker별 다음 기간 수익률을 추론하고 `granite_ttm_expected_return`, `granite_ttm_confidence`를 채운다.

## Text Models

### FinBERT

금융 뉴스와 헤드라인의 positive, neutral, negative score를 산출한다. 앱 feature는 `sentiment_score`, `negative_ratio`, `confidence`로 정규화한다.

### FinMA / FinGPT

SEC 공시, 실적발표, 긴 문서의 event extraction adapter로 둔다. 모델 출력은 자유 문장이 아니라 구조화 JSON으로 검증한다.

- FinMA 기본 모델: `ChanceFocus/finma-7b-nlp`
- FinGPT 기본 adapter: `FinGPT/fingpt-mt_llama3-8b_lora`
- FinGPT 기본 base model: `meta-llama/Meta-Llama-3-8B`
- 구조화 출력 스키마: `event_tag`, `risk_flag`, `confidence`, `summary_ref`
- JSON 파싱 또는 스키마 검증이 실패하면 deterministic rules로 fallback한다.

FinGPT는 LoRA adapter이므로 base model 다운로드 권한과 Hugging Face 인증이 필요할 수 있다. 이 모델들은 feature 생성과 리포트 요약 보조에만 사용하며, 매매 판단은 하지 않는다.

### Ollama Agent

기본 로컬 모델은 `qwen3-coder:30b`로 둔다. 역할은 데이터 조회 보조, 모델 호출 상태 설명, 결과 리포트 요약이다. 매수/매도 판단은 하지 않는다.

## Targets

- `forward_return_1`: 다음 기간 수익률
- `realized_volatility`: rolling realized volatility
- `downside_quantile`: 하방 분위수 proxy 또는 Chronos quantile
- `risk_score`: 가격/텍스트/SEC feature 결합 리스크 점수

## Feature Fusion

날짜와 티커를 기준으로 가격 feature, 모델 예측값, 뉴스 feature, SEC feature를 left join한다. 텍스트 feature는 event timestamp 이후 다음 거래 가능 시점부터 적용한다.

## Local Model Bootstrap

로컬 모델 캐시는 다음 스크립트로 준비한다.

```bash
uv --cache-dir .uv-cache run python scripts/preload_local_models.py --chronos --granite
uv --cache-dir .uv-cache run python scripts/preload_local_models.py --finma --mode download
uv --cache-dir .uv-cache run python scripts/preload_local_models.py --fingpt --mode download --fingpt-base-id meta-llama/Meta-Llama-3-8B
```

이미 캐시된 모델만 검증하려면 `--local-files-only --mode warmup`을 붙인다. 이 경로는 실제 모델 파일과 런타임 환경에서만 완전히 검증된다.
FinMA/FinGPT 같은 7B 계열 모델은 `--offload-folder artifacts/model_offload`로 disk offload를 허용한다. warmup 출력의 `source`가 `local`이면 모델 JSON이 검증을 통과한 것이고, `rules`면 모델 오류 또는 JSON 스키마 실패 후 deterministic fallback이 사용된 것이다.
