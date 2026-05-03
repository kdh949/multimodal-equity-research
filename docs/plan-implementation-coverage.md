# Plan Implementation Coverage (vs `/Users/donghyunkim/Downloads/PLAN.md`)

Last checked: 2026-05-04

Legend: **Implemented / Partial / Remaining**  

## 1) Foundations and setup

| Item | Status | Evidence |
| --- | --- | --- |
| `AGENTS.md`, `README.md`, 핵심 docs 파일들이 사전 작성돼 있는지 | Implemented | [AGENTS.md](/Users/donghyunkim/Documents/Quantitative-Trading/AGENTS.md:1), [README.md](/Users/donghyunkim/Documents/Quantitative-Trading/README.md:1), [docs/architecture.md](/Users/donghyunkim/Documents/Quantitative-Trading/docs/architecture.md:1), [docs/data-sources.md](/Users/donghyunkim/Documents/Quantitative-Trading/docs/data-sources.md:1), [docs/modeling.md](/Users/donghyunkim/Documents/Quantitative-Trading/docs/modeling.md:1), [docs/risk-validation.md](/Users/donghyunkim/Documents/Quantitative-Trading/docs/risk-validation.md:1), [docs/git-workflow.md](/Users/donghyunkim/Documents/Quantitative-Trading/docs/git-workflow.md:1) |
| Streamlit UI + `src/quant_research/` 패키지 분리 | Implemented | [app.py](/Users/donghyunkim/Documents/Quantitative-Trading/app.py:16), [src/quant_research/pipeline.py](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/pipeline.py:74), [src/quant_research/features/fusion.py](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/features/fusion.py:8), [src/quant_research/models/tabular.py](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/models/tabular.py:20), [src/quant_research/backtest/engine.py](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/backtest/engine.py:31), [src/quant_research/signals/engine.py](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/signals/engine.py:24) |
| 실거래 주문 기능 제외, LLM 결정 금지 | Implemented | [AGENTS.md](/Users/donghyunkim/Documents/Quantitative-Trading/AGENTS.md:5), [README.md](/Users/donghyunkim/Documents/Quantitative-Trading/README.md:5), [src/quant_research/signals/engine.py](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/signals/engine.py:24), [src/quant_research/backtest/engine.py](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/backtest/engine.py:31) |
| 미래 데이터 누수 방지 | Partial | `forward_return_1` 생성은 시프트 기반으로 다음 시점 타깃 사용([src/quant_research/features/price.py](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/features/price.py:21)), walk-forward 분리는 과거 구간→미래 구간 원칙 적용([src/quant_research/validation/walk_forward.py](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/validation/walk_forward.py:30)). 하지만 지표 생성 후 `test` 구간에 대한 추가 타임라인 가드(예: feature lag/공개일정 오프셋 강제)가 일부 엔드투엔드에서 완전 엄격히 명시되지 않음 |

## 2) Architecture & data layer

| Item | Status | Evidence |
| --- | --- | --- |
| 가격 계층: OHLCV, 거래량, 규격화 | Implemented | [MarketDataProvider protocol](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/data/market.py:11), [SyntheticMarketDataProvider](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/data/market.py:22), [YFinanceMarketDataProvider](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/data/market.py:69), [normalize](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/data/market.py:114) |
| 뉴스 + SEC provider 인터페이스 | Implemented | [NewsProvider protocol](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/data/news.py:22), [SecEdgarClient](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/data/sec.py:16) |
| 기본 종목군 정의 | Implemented | [DEFAULT_TICKERS](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/config.py:7), [README.md](/Users/donghyunkim/Documents/Quantitative-Trading/README.md:51), [src/quant_research/pipeline.py](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/pipeline.py:42) |
| 가격 + 뉴스 + SEC feature 결합 | Implemented | [fuse_features](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/features/fusion.py:8), [build_news_features](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/features/text.py:70), [build_sec_features](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/features/sec.py:10), [pipeline flow](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/pipeline.py:79) |
| 파이프라인 동작: synthetic/live 모드 + 기본 통합 | Implemented | [PipelineConfig](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/pipeline.py:40), [market loader](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/pipeline.py:121), [news loader](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/pipeline.py:130), [sec loader](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/pipeline.py:143) |

## 3) Modeling layer

| Item | Status | Evidence |
| --- | --- | --- |
| Chronos-2 (return/vol/quantile 후보) | Partial | [Chronos2Adapter](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/models/timeseries.py:10) only adds proxy features and does not invoke 실제 챗봇/모델 추론 |
| Granite TTM adapter | Partial | [GraniteTTMAdapter](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/models/timeseries.py:33) is proxy/feature-only implementation |
| LightGBM baseline 우선순위 + XGBoost/CatBoost + fallback | Implemented | [_make_estimator](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/models/tabular.py:71), model fallback chain 및 `HistGradientBoostingRegressor` fallback |
| FinBERT sentiment | Partial | [FinBERTSentimentAnalyzer](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/models/text.py:12) uses transformers fallback to keyword analyzer |
| FinMA / FinGPT event extraction | Partial | [FilingEventExtractor](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/models/text.py:52), [FinGPTEventExtractor](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/models/text.py:101) are rule-based/shared-contract, not live model adapters |
| Ollama local model for explanation/summary | Implemented | [OllamaAgent](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/models/ollama.py:9) with default `qwen3-coder:30b` |
| 구조화 feature 저장 (sentiment_score/event_tag/risk_flag/confidence/summary_ref) | Implemented | Text path: [KeywordSentimentAnalyzer](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/features/text.py:50), [news aggregation](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/features/text.py:70), SEC path: [FilingEventExtractor](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/models/text.py:55), [sec features](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/features/sec.py:62), [pipeline attach](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/pipeline.py:187) |

## 4) Validation, signals, and backtest

| Item | Status | Evidence |
| --- | --- | --- |
| 예측값을 바로 주문으로 쓰지 않음 | Implemented | [run_long_only_backtest uses DeterministicSignalEngine](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/backtest/engine.py:33) before signaling |
| signal score + BUY/SELL/HOLD 규칙 + 비용/슬리피지 반영 | Implemented | [SignalEngineConfig](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/signals/engine.py:10), [generate masks](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/signals/engine.py:59), [cost/slippage in backtest](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/backtest/engine.py:45) |
| Walk-forward + final fold OOS 표기 | Implemented | [walk_forward_splits](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/validation/walk_forward.py:30), [walk_forward_predict OOS flag](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/validation/walk_forward.py:85) |
| 리스크 룰: max weight/portfolio vol/max drawdown stop/event/유동성 | Implemented | [BacktestConfig](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/backtest/engine.py:13), [signal engine risk masks](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/signals/engine.py:59), [drawdown stop logic](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/backtest/engine.py:51) |
| 비용·슬리피지·turnover 반영 | Implemented | [net_return calc](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/backtest/engine.py:77), [cost config](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/backtest/engine.py:13) |
| Signal/portfolio 시차 적용 | Implemented | [effective_date shift](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/backtest/engine.py:49), [coverage in tests](/Users/donghyunkim/Documents/Quantitative-Trading/tests/test_backtest_risk.py:61) |
| Ablation (텍스트/SEC/비용) | Implemented | [pipeline _run_ablation_summary](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/pipeline.py:223), [pipeline smoke test checks scenarios](/Users/donghyunkim/Documents/Quantitative-Trading/tests/test_pipeline.py:23) |

## 5) Test coverage vs plan

| Plan test target | Status | Evidence |
| --- | --- | --- |
| Leakage/time-shift tests | Implemented | [test_forward_return_uses_next_period_only](/Users/donghyunkim/Documents/Quantitative-Trading/tests/test_price_features.py:9), [test_walk_forward_splits_never_train_on_future_dates](/Users/donghyunkim/Documents/Quantitative-Trading/tests/test_walk_forward.py:12) |
| SEC 파서·필링 정규화 테스트 | Implemented | [test_sec_features_normalize_filings_and_facts](/Users/donghyunkim/Documents/Quantitative-Trading/tests/test_sec_features.py:17), [test_sec_companyconcept_and_frame_extractors_normalize_payloads](/Users/donghyunkim/Documents/Quantitative-Trading/tests/test_sec_features.py:47) |
| 텍스트/LLM 출력 스키마 검증 | Implemented | [test_finbert_fallback_returns_structured_schema](/Users/donghyunkim/Documents/Quantitative-Trading/tests/test_text_models.py:14), [test_filing_event_extractor_validates_schema](/Users/donghyunkim/Documents/Quantitative-Trading/tests/test_text_models.py:26) |
| signal engine BUY/SELL/HOLD 규칙 | Implemented | [test_signal_engine_buys_only_after_risk_checks](/Users/donghyunkim/Documents/Quantitative-Trading/tests/test_signal_engine.py:8), [test_signal_engine_blocks_low_liquidity](/Users/donghyunkim/Documents/Quantitative-Trading/tests/test_signal_engine.py:31) |
| 통합 파이프라인 + walk-forward smoke | Implemented | [test_synthetic_pipeline_runs_end_to_end](/Users/donghyunkim/Documents/Quantitative-Trading/tests/test_pipeline.py:6), [test_backtest_*](/Users/donghyunkim/Documents/Quantitative-Trading/tests/test_backtest_risk.py:8) |
| 성능 비교/모델별 개선/안정성 지표(Chronos/TTM/기본) | Partial | 비용/텍스트/SEC ablation과 Chronos/Granite proxy feature 제거 비교는 구현됨([pipeline](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/pipeline.py:223)). 다만 실제 Chronos-2/Granite 모델 추론 기반 통계 검정과 장기 OOS 안정성 리포트는 아직 proxy 수준 |

## 6) SEC 규칙 준수

| 항목 | Status | Evidence |
| --- | --- | --- |
| User-Agent 명시 | Implemented | [SecSettings.user_agent](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/config.py:21), [session header](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/data/sec.py:24) |
| 10 req/s 이하 throttling | Implemented | [max_requests_per_second=9.0](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/config.py:25), [throttle interval](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/data/sec.py:92) |
| local cache and no commit of raw artifacts | Implemented | [cache_dir + file cache](/Users/donghyunkim/Documents/Quantitative-Trading/src/quant_research/data/sec.py:18), [data/artifacts gitignore](/Users/donghyunkim/Documents/Quantitative-Trading/.gitignore:13) |

## GitHub Flow + Commit Convention evidence

- Branch naming policy is documented as GitHub Flow in docs, with `main`, `feature/*`, and `fix/*` conventions: [docs/git-workflow.md](/Users/donghyunkim/Documents/Quantitative-Trading/docs/git-workflow.md:3).  
- Current repo shows branch usage `main`, `feature/bootstrap-quant-research-app`, and current fix branch: `fix/plan-qa-risk-validation-gaps` via `git branch` state.
- Commit messages currently follow `<type>: <한국어 제목>` pattern used in recent history, e.g. `feat: 멀티모달 퀀트 리서치 앱 초기 구현`, `test: 워크포워드 검증 테스트 추가` (from `git log --oneline`), matching the allowed list in [docs/git-workflow.md](/Users/donghyunkim/Documents/Quantitative-Trading/docs/git-workflow.md:21).

## Remaining priority (to fully match PLAN)

1. 실제 추론 파이프라인으로 Chronos-2 / Granite TTM / FinMA / FinGPT를 선택적 Adapter로 추가 (현재는 대체로 proxy/fallback 기반).
2. 텍스트·SEC feature 타이밍(특히 이벤트 발생-적용 lag)과 공개 시점 기준 반영을 더 엄격하게 문서-코드 동기화.
3. 성능 계획 대비 항목인 실제 모델 추론 기반 비교/개선 분석(Chronos-2/Granite TTM/기본 모델, OOS score stability)을 리포트 형태로 정규화.
