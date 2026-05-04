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

기본 실행은 `proxy`/`rules` fallback으로 빠르게 동작한다. 실제 Chronos-2, Granite TTM, FinMA, FinGPT 로컬 추론을 쓰려면 optional dependency와 Hugging Face 모델 캐시가 필요하다.

```bash
uv --cache-dir .uv-cache sync --all-extras
uv --cache-dir .uv-cache run python scripts/preload_local_models.py --chronos --granite
uv --cache-dir .uv-cache run python scripts/preload_local_models.py --finma --mode download
uv --cache-dir .uv-cache run python scripts/preload_local_models.py --fingpt --mode download --fingpt-profile mt-llama3
uv --cache-dir .uv-cache run python scripts/preload_local_models.py --fingpt --mode download --fingpt-profile forecaster --fingpt-adapter-only
```

Streamlit 사이드바에서 `Time-series inference=local`, `Filing extractor=finma|fingpt`, `Use local filing LLM`을 켜면 실제 로컬 어댑터를 호출한다. FinGPT는 공식 repo의 base model + LoRA adapter 구조를 따른다. `mt-llama3` 기본 profile은 `FinGPT/fingpt-mt_llama3-8b_lora`와 `meta-llama/Meta-Llama-3-8B`를 사용하고, 이전 `mt-llama2` profile도 명시적으로 선택할 수 있다. Forecaster profile은 `FinGPT/fingpt-forecaster_dow30_llama2-7b_lora`와 `meta-llama/Llama-2-7b-chat-hf`를 사용한다. Meta Llama base model은 접근 권한과 `HF_TOKEN` 또는 Hugging Face 로그인 상태가 필요할 수 있다.
7B 계열 모델이 메모리 부족으로 disk offload를 사용할 때는 `--offload-folder artifacts/model_offload`와 `--max-new-tokens`로 warmup 비용을 조절한다.

## 기본 종목군

`SPY, QQQ, AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, JPM`

## 문서

- [Architecture](docs/architecture.md)
- [Data Sources](docs/data-sources.md)
- [Modeling](docs/modeling.md)
- [Risk And Validation](docs/risk-validation.md)
- [Git Workflow](docs/git-workflow.md)
