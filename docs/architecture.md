# Architecture

## Layers

이 앱은 UI와 리서치 엔진을 분리한다.

- `app.py`: Streamlit 대시보드 진입점
- `src/quant_research/data`: 가격, 뉴스, SEC EDGAR provider
- `src/quant_research/features`: 가격/텍스트/SEC feature 생성과 feature fusion
- `src/quant_research/models`: tabular ML, time-series foundation model, NLP model adapter
- `src/quant_research/validation`: walk-forward split, out-of-sample 평가, ablation
- `src/quant_research/backtest`: portfolio simulation, cost/slippage, performance metrics
- `src/quant_research/signals`: deterministic signal engine과 리스크 룰

## Data Flow

```text
MarketDataProvider -> PriceFeatureBuilder -> PriceModelAdapter
NewsProvider       -> TextFeatureBuilder  -> TextModelAdapter
SecEdgarProvider   -> SecFeatureBuilder   -> FilingModelAdapter

PricePredictions + TextSignals + SecFeatures
        |
FeatureFusion
        |
WalkForwardValidator
        |
Backtester
        |
DeterministicSignalEngine
```

## Deterministic Decision Boundary

LLM과 문서 이해 모델은 feature와 설명을 만든다. 투자 판단은 다음 순서로 고정한다.

1. 모델 예측값 생성
2. feature score 생성
3. 비용과 슬리피지 차감
4. 리스크 룰 적용
5. 검증 통과 여부 확인
6. signal engine이 `BUY`, `SELL`, `HOLD` 생성

## Optional Model Policy

Chronos-2, Granite TTM, FinBERT, FinMA, FinGPT, Ollama는 optional adapter로 둔다. 앱은 모델이 없을 때도 synthetic 데이터와 lightweight fallback으로 실행되어야 한다.

실제 local heavy model 추론은 명시적으로 켜야 한다. `scripts/preload_local_models.py`는 Hugging Face 캐시 다운로드와 warmup을 담당하고, Streamlit 설정은 모델 ID, device map, FinGPT base model을 주입한다. 모델 로딩/추론 실패는 리서치 파이프라인 전체 실패가 아니라 해당 adapter의 deterministic fallback으로 처리한다.

FinGPT의 로컬 경로는 `PipelineConfig`에서 주입되는 안전한 기본값을 사용한다.
기본값은 경량 모드에서 변하지 않으며, Mac 기준 `mlx` 또는 `llama-cpp` 양자화 경로를 우선하고, 안전 장치가 켜진 상태에서는 unquantized Transformers 8B를 기본적으로 로드하지 않는다.
`single_load_lock_path`는 로컬 FinGPT 로딩 동시 실행을 막아 한 번에 하나의 로컬 LLM 로드만 허용한다.
