# Quant Validity Interview Prep

Date: 2026-05-07

## Working Answer

이 저장소는 "바로 쓸 수 있는 매매 전략"으로는 아직 충분히 유효하지 않다. 하지만 "퀀트 리서치 후보를 검증하고, 쓸모없는 전략을 걸러내는 시스템"으로는 이미 쓸모가 있다.

인터뷰에서는 이 구분을 먼저 고정해야 한다.

- System validity: 리서치/검증 프레임워크로는 의미가 있다.
- Strategy candidate validity: 현재 전략 후보는 사용 부적합에 가깝다.
- Product positioning: 알파 생성기가 아니라 alpha due-diligence harness로 설명하는 편이 맞다.

## Evidence

현재 시스템이 유효한 부분:

- UI와 리서치 엔진이 분리되어 있고, 가격/뉴스/SEC feature fusion -> walk-forward -> backtest -> ablation 흐름이 있다. See `src/quant_research/pipeline.py`.
- `PipelineConfig`의 기본 target은 `forward_return_20`이고, gap/embargo는 target horizon 이상으로 보정된다.
- Walk-forward split은 train window 이후 gap을 둔 test window를 만들고 마지막 fold를 OOS로 표시한다. See `src/quant_research/validation/walk_forward.py`.
- LLM/text output은 직접 매매 판단이 아니라 feature로 들어가며, 최종 action은 deterministic signal engine과 backtest layer가 만든다. See `src/quant_research/signals/engine.py` and `src/quant_research/backtest/engine.py`.
- Validity gate에는 system status와 strategy candidate status가 분리되어 있다. `pass`, `hard_fail`, `not_evaluable` 같은 시스템 상태와 전략 후보의 `pass/warning/fail/insufficient_data/not_evaluable` 계약이 있다.
- Baseline 비교, equal-weight 비교, cost-adjusted return, turnover, drawdown, rank IC, ablation 결과를 JSON/Markdown/HTML/Streamlit로 내보내는 구조가 있다.
- 최소 검증 결과: `uv --cache-dir .uv-cache run pytest` passed: 290 tests, 152 warnings, 216.20s.

현재 전략 후보가 약한 부분:

- `reports/backtest_validation_20260505/metrics.json`: CAGR -5.24%, Sharpe -0.36, max drawdown -21.93%, hit rate 18.63%, benchmark CAGR +17.78%, excess return -23.03%.
- `reports/backtest_validation_20260505/validation_summary.csv`: 마지막 OOS fold는 2026-04-22 to 2026-05-01, 80 observations, directional accuracy 45.0%.
- `reports/backtest_validation_20260504_tabular_model_matrix/summary.json`: LightGBM/XGBoost/CatBoost 모두 mean directional accuracy가 약 49.5% 수준이고 OOS directional accuracy도 42.9% to 48.6%다. 세 모델 모두 CAGR과 excess return이 음수다.
- `reports/backtest_validation_matrix/summary_20260504.csv`: FinBERT/FinMA/FinGPT/Ollama 변형이 결과를 거의 개선하지 못했다. 현재 heavy/text adapter는 "추가 가치" 증거가 부족하다.

## Main Risks To Discuss

1. Return horizon semantics
   - 현재 설계는 20일 forward return을 핵심 target으로 둔다.
   - 그런데 reporting script 일부는 `forward_return_1`을 직접 참조한다.
   - backtest가 `forward_return_20`을 매 signal date마다 compounding하는 방식이면, 20일 중첩 수익률을 일별 수익처럼 쓸 위험이 있다.
   - 인터뷰에서 먼저 결정할 것: 20d signal을 20일 보유 포지션으로 시뮬레이션할지, 1d realized return으로 일별 보유를 추적할지, non-overlapping 20d rebalance로 평가할지.

2. Data timing
   - 가격 feature는 과거 rolling 기반으로 안전한 편이다.
   - 뉴스/SEC feature는 publication time, filing acceptance time, market close 기준 적용 lag가 더 엄격해야 한다.
   - "feature available at t" 계약을 provider/output schema에 넣어야 한다.

3. Model value
   - 현재 모델/adapter가 baseline proxy, price-only, equal-weight, SPY 대비 유의미한 개선을 보였다는 증거가 약하다.
   - "모델이 복잡하다"가 아니라 "모델이 proxy보다 낫다"를 gate에서 보여줘야 한다.

4. Reproducibility
   - live yfinance/GDELT/EDGAR 실행 결과는 재실행 시 변할 수 있다.
   - 중요한 run은 raw snapshot hash, config, universe, artifact manifest를 함께 저장해야 한다.

## Interview Agenda

1. "쓸모있다"의 정의를 확정한다.
   - 리서치 시스템인가?
   - 전략 후보 검증기인가?
   - 실제 운용 가능한 전략인가?

2. 현재 결론을 분리한다.
   - 시스템: 유효성 검증 장치로는 쓸모 있음.
   - 전략 후보: 현재 리포트 기준으로는 fail.

3. 가장 먼저 고칠 축을 고른다.
   - P0: horizon/backtest timing semantics.
   - P0: report target column consistency.
   - P0: feature availability timestamp and lag.
   - P1: validity gate artifact를 실제 live run의 기본 산출물로 만들기.
   - P1: model/proxy ablation 결과를 UI와 report 첫 화면에 노출.

4. 다음 작업의 acceptance criteria를 정한다.
   - 어떤 run이 "system pass / strategy fail"을 명확히 보여야 한다.
   - 어떤 run이 "strategy warning/pass"로 올라가려면 SPY와 equal-weight 대비 cost-adjusted excess return이 양수여야 한다.
   - 모델 추가는 no-model proxy와 price-only 대비 material improvement가 있어야만 성공으로 친다.

## Suggested Next Changes

P0:

- `scripts/run_backtest_validation.py`에서 `forward_return_1` 하드코딩을 제거하고, 실제 `config.prediction_target_column` 또는 backtest `realized_return_column`을 사용한다.
- 20d target backtest semantics를 명시한다. 중첩 20d return을 매일 compounding하지 않도록 설계/테스트를 추가한다.
- 뉴스/SEC feature에 `available_at` 또는 `effective_date`를 추가하고, train/test input에서 signal date 이후 feature를 차단한다.

P1:

- live validation script가 `validity_gate.json`, `validity_gate.md`, `validity_gate.html`을 기본 저장하게 한다.
- Streamlit 첫 화면을 "System validity"와 "Strategy candidate status"로 나눈다.
- model/proxy/channel ablation matrix를 one-screen summary로 만든다.

P2:

- report artifact manifest에 data snapshot hash, config, git commit, universe, provider fallback reason을 저장한다.
- universe별 안정성 테스트를 추가한다: mega-cap 30, sector-balanced, liquidity top N, ex-tech.
- cost/slippage sensitivity matrix를 추가한다.

## Interview Questions

- 이 프로젝트의 성공 기준은 "수익 전략 발견"인가, "나쁜 전략을 빠르게 탈락시키는 검증 시스템"인가?
- 20일 target을 유지할 것인가, 아니면 backtest는 1일 realized return으로 포지션을 추적할 것인가?
- live data 재현성을 어느 수준까지 요구할 것인가?
- heavy model은 feature generator로만 남길 것인가, 아니면 proxy 대비 성능 개선이 없으면 기본 off로 둘 것인가?
- 다음 milestone의 demo는 Streamlit UX인가, validity artifact인가, 실제 live run report인가?
