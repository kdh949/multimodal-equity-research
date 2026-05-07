# Risk And Validation

## Fixed Validation Flow

1. 모델이 예측값을 생성한다.
2. 예측값을 바로 주문하지 않는다.
3. 예측값과 텍스트/SEC feature를 신호 점수로 변환한다.
4. 거래비용, 슬리피지, turnover를 반영한다.
5. walk-forward 검증을 수행한다.
6. out-of-sample 성능을 확인한다.
7. deterministic signal engine이 매수/매도/관망을 판단한다.

## Walk-Forward

- train window는 과거 구간만 사용한다.
- 기본 검증 target은 `forward_return_20`이다.
- 1일/5일 horizon은 pass/fail을 만들지 않는 diagnostic 지표로만 사용한다.
- gap과 embargo는 target horizon과 긴 가격 lookback을 모두 덮도록 기본 60 거래일 이상으로 둔다.
- test window는 train 이후 구간만 사용한다.
- 마지막 구간은 out-of-sample summary로 별도 표시한다.

## Backtest Rules

- 기본 전략은 상위 점수 종목 동일가중 롱온리다.
- 포지션은 신호 생성 다음 기간 수익률부터 적용한다.
- 비용은 turnover 기준으로 차감한다.
- 슬리피지는 거래된 notional 기준으로 차감한다.
- 백테스트 산출물은 gross return, transaction cost, slippage cost, total turnover cost, net cost-adjusted return을 분리해 기록한다.
- 벤치마크는 `SPY`다.

## Risk Rules

- 종목별 최대 비중
- 포트폴리오 변동성 한도와 최근 실현수익률 공분산 기반 상관관계 반영
- sector 데이터가 없을 때는 수익률 상관 cluster별 집중도 제한
- 최대 낙폭 중단 룰
- SEC 이벤트 리스크는 graded throttle로 처리하고, severe/recent risk일 때만 강한 매도 조건으로 쓴다.
- 최소 유동성 조건
- 텍스트 리스크 과열 차단

## Metrics

- CAGR
- annualized volatility
- Sharpe ratio
- maximum drawdown
- hit rate
- turnover
- exposure
- benchmark excess return
- OOS score stability
- feature ablation delta

## Stage 1 Validity Gate Schema Contract

Stage 1 defines the comparison contract before expanding model complexity. The
machine-readable `validity_gate.json` and human-readable report include:

- full model input schema: structured price/text/SEC plus optional Chronos-2,
  Granite TTM, FinBERT, FinMA, FinGPT, and Ollama adapters with rules fallback
- baseline input schema: no-model-proxy model baseline, SPY market benchmark,
  and equal-weight universe benchmark
- ablation input schema: price/text/SEC channels, no-cost diagnostic, and named
  model/modality ablations
- metric schema: rank IC, positive fold ratio, OOS rank IC, Sharpe, max drawdown,
  cost-adjusted cumulative return, excess return, and turnover
- validation window schema: walk-forward folds, OOS holdout, gap/embargo, target
  horizon, and t+1-or-later return timing
- comparison result rows: per-window full-model comparisons against model
  baselines, return baselines, and ablations with absolute delta, relative delta,
  and direction-aware pass/fail status
- explicit metric contract rows: top-level full-model metrics, baseline metrics,
  ablation metrics, and structured pass/fail reasons mirrored into JSON
- rule result explanations: normalized rows for each gate with status, pass/fail
  boolean when applicable, reason, reason code, metric, threshold, operator, and
  whether the rule affects final strategy status
- final strategy status explanation: one summary object tying system validity,
  blocking rules, warning rules, insufficient-data rules, and official operator
  message to the final strategy candidate status

## Six-Stage Roadmap

1. Stage 1 Validity Gate: schema contract, JSON/Markdown/HTML/Streamlit outputs,
   and tests for comparison inputs/results.
2. Stage 2 Data Timing: stricter provider contracts, publication-time alignment,
   and feature availability checks.
3. Stage 3 Model Adapter Evidence: optional heavy adapter preload path, fallback
   parity tests, and structured text/model feature provenance.
4. Stage 4 Signal And Risk Realism: deterministic signal scoring, costs,
   slippage, turnover, drawdown, and benchmark gates.
5. Stage 5 Ablation And Incremental Value: Chronos/Granite/LLM/model-channel
   ablations with material improvement thresholds.
6. Stage 6 Reporting And Reproducibility: reproducible artifacts, operator
   reports, run manifests, and release-ready validation summaries.
