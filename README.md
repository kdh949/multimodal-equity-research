# Quantitative Trading Research App

미국 주식 가격 데이터, 뉴스/공시 텍스트, SEC EDGAR 재무 데이터를 결합해 로컬에서 모델링, 백테스트, walk-forward 검증을 수행하는 멀티모달 퀀트 리서치 앱이다.

이 앱은 투자 조언이나 실거래 시스템이 아니다. v1은 리서치와 검증 전용이며, 최종 매수/매도/관망 신호는 LLM이 아니라 deterministic signal engine이 산출한다.

## 목표

- OHLCV, 기술지표, 변동성, 모멘텀, SEC 재무지표 수집
- Chronos-2, Granite TTM, LightGBM 계열 모델을 통한 가격/리스크 예측
- FinBERT, FinMA/FinGPT, Ollama 기반 뉴스/공시 분석 adapter
- 가격 feature와 텍스트/SEC feature 결합
- 거래비용, 슬리피지, 리스크 룰을 포함한 백테스트
- walk-forward 및 out-of-sample 검증

## 기본 흐름

```text
가격/OHLCV/거래량/지표 데이터
        |
시계열 예측 모델 / 전통 ML 모델
        |
수익률·변동성·리스크 예측값

뉴스/공시/실적발표 텍스트
        |
FinBERT / FinGPT / FinMA / 로컬 LLM
        |
감성 점수·이벤트 태그·리스크 요약

두 결과를 feature로 결합
        |
백테스트 + 워크포워드 검증 + 리스크 룰
        |
최종 매수/매도/관망 신호
```

## 실행

```bash
uv --cache-dir .uv-cache sync --all-extras
uv --cache-dir .uv-cache run streamlit run app.py
```

테스트:

```bash
uv --cache-dir .uv-cache run pytest
```

## 로컬 heavy model 추론

기본 실행은 `proxy`/`rules` fallback으로 빠르게 동작한다. 기본적으로는 **다운로드/캐시 없는 경량 모드**로 시작되어 실제 추론을 트리거하지 않는다.
실제 Chronos-2, Granite TTM, FinMA, FinGPT를 쓰려면 optional dependency와 Hugging Face 모델 캐시가 필요하다.

```bash
uv --cache-dir .uv-cache sync --all-extras
uv --cache-dir .uv-cache run python scripts/preload_local_models.py --chronos --granite
uv --cache-dir .uv-cache run python scripts/preload_local_models.py --finma --mode download
uv --cache-dir .uv-cache run python scripts/preload_local_models.py --fingpt --mode download --fingpt-profile mt-llama3
uv --cache-dir .uv-cache run python scripts/preload_local_models.py --fingpt --mode verify --fingpt-profile mt-llama3 --fingpt-runtime transformers
uv --cache-dir .uv-cache run python scripts/preload_local_models.py --fingpt --mode download --fingpt-profile forecaster --fingpt-adapter-only
```

Streamlit 사이드바에서 `Time-series inference=local`, `Filing extractor=finma|fingpt`, `Use local filing LLM`을 켜면 실제 로컬 어댑터를 호출한다. 기본 FinGPT 경로는 `rules`/`keyword`이고, 경량 모드는 그대로 유지된다.

FinGPT는 공식 repo의 base model + LoRA adapter 구조를 따른다. `mt-llama3` 기본 profile은 `FinGPT/fingpt-mt_llama3-8b_lora`와 `meta-llama/Meta-Llama-3-8B`를 사용한다.
`FinGPT base model`은 접근 권한과 `HF_TOKEN` 또는 Hugging Face 로그인 상태가 필요할 수 있다.

`Local model settings`에는 런타임 전용 경로가 추가되며, mac 기준 기본 런타임은 MLX/llama.cpp 계열 양자화 경로(`artifacts/model_cache/...q4...`)를 선호한다.
`verify`는 로컬 캐시만 확인하고 모델 가중치를 메모리에 올리지 않는다. 무거운 warmup/추론은 선택적 동작이며, 기본 `Allow unquantized Transformers 8B load`는 꺼져 있어서 기본 보안 가드를 유지한다.
단일 장비에서 로컬 모델을 동시에 여러 번 적재하지 않도록 `FinGPT single-load lock file`이 기본으로 준비되어 있다.
7B 계열 모델이 메모리 부족으로 disk offload를 사용할 때는 `--offload-folder artifacts/model_offload`와 `--max-new-tokens`로 warmup 비용을 조절한다.

## 기본 종목군

기본 전략 후보는 대표적인 미국 대형 유동주 30개 종목이다.

`AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, JPM, V, MA, UNH, XOM, JNJ, PG, HD, COST, ABBV, BAC, KO, PEP, WMT, AVGO, LLY, MRK, CVX, CRM, AMD, NFLX, ORCL, ADBE`

`SPY`는 기본 벤치마크 데이터로만 포함하며, `SPY`, `QQQ`, `DIA`, `IWM` 같은 benchmark ETF는 기본 전략 후보에서 제외한다.

## 문서

- [Architecture](docs/architecture.md)
- [Data Sources](docs/data-sources.md)
- [Data Timestamp Schema](docs/data-timestamp-schema.md)
- [Modeling](docs/modeling.md)
- [Risk And Validation](docs/risk-validation.md)
- [Git Workflow](docs/git-workflow.md)
