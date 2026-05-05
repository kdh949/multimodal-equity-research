# 백테스트 검증 개요 및 결과 구조

## 목적

이 검증 스크립트는 LightGBM 기반 주식 수익률 예측 모델이 합성(Synthetic) 데이터가 아닌
**실제 시장 데이터**에서도 유효한 예측력을 갖는지 확인한다.

## 검증 대상 모델

| 컴포넌트 | 설명 |
|---|---|
| 예측 모델 | LightGBM (TabularReturnModel) |
| 검증 방식 | Walk-Forward Cross-Validation |
| 예측 대상 | 설정된 선행 수익률 horizon (`forward_return_1`, `forward_return_5`, `forward_return_20` 중 하나) |
| 입력 피처 | 가격·기술지표·거래량·뉴스감성·SEC공시 등 융합 피처 |

## 사용 데이터

| 항목 | 내용 |
|---|---|
| 데이터 소스 | yfinance (인터넷 실시간) |
| 분석 종목 | SPY, QQQ, AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, JPM |
| 분석 기간 | 실행일 기준 과거 2년 |
| 뉴스 소스 | yfinance News API + GDELT |
| 공시 소스 | SEC EDGAR API (SPY·QQQ 제외, CIK 미등록 ETF) |

## 검증 파이프라인 흐름

```
yfinance 실제 데이터 수집
        ↓
가격 피처 생성 (return_1/5/20, RSI, 변동성, 거래량 등)
        ↓
뉴스 감성 피처 생성 (키워드 기반)
        ↓
SEC 공시 피처 생성 (규칙 기반)
        ↓
피처 융합 (fuse_features)
        ↓
Walk-Forward 분할 (훈련 90일 / 테스트 20일 / effective gap/embargo ≥ 선택 horizon)
        ↓
각 Fold: LightGBM 학습 → OOS 예측
        ↓
예측 정확도 측정 (MAE, 방향성 정확도)
        ↓
시그널 생성 → 포트폴리오 백테스트
        ↓
최종 리포트 출력 + 파일 저장
```

## 생성되는 산출물

스크립트 실행 후 `reports/backtest_validation_<YYYYMMDD>/` 폴더에 저장된다.

| 파일 | 내용 |
|---|---|
| `predictions.csv` | 모든 fold의 날짜·종목별 예측값, 실제값, fold 번호, OOS 여부 |
| `validation_summary.csv` | fold별 훈련 기간, 테스트 기간, MAE, 방향성 정확도 |
| `metrics.json` | 포트폴리오 백테스트 최종 지표 (CAGR, Sharpe, 최대 낙폭 등) |

### predictions.csv 주요 컬럼

| 컬럼 | 의미 |
|---|---|
| `date` | 예측 대상 날짜 |
| `ticker` | 종목 코드 |
| `expected_return` | 모델이 예측한 선택 horizon 수익률 |
| `forward_return_*` | 실제 발생한 horizon별 수익률 라벨. 모델 입력에는 사용하지 않고, 선택된 하나만 검증/백테스트 실현 수익률로 사용 |
| `fold` | 속한 Walk-Forward fold 번호 |
| `is_oos` | 마지막 fold(Out-of-Sample) 여부 |

### validation_summary.csv 주요 컬럼

| 컬럼 | 의미 |
|---|---|
| `target_column` | 해당 실행에서 학습/검증에 사용한 `forward_return_*` 컬럼 |
| `target_horizon` | 선택 target의 거래일 horizon |
| `requested_gap_periods` | 사용자가 요청한 train/test gap |
| `requested_embargo_periods` | 사용자가 요청한 fold 이후 embargo |
| `effective_gap_periods` | 실제 분할에 적용한 gap. 최소 선택 horizon 이상 |
| `effective_embargo_periods` | 실제 분할에 적용한 embargo. 최소 선택 horizon 이상 |
| `label_overlap_violations` | purge 이후에도 train/test label interval이 겹친 수 |

### metrics.json 주요 키

| 키 | 의미 |
|---|---|
| `cagr` | 연환산 전략 수익률 |
| `benchmark_cagr` | SPY 연환산 수익률 |
| `excess_return` | 전략 CAGR − SPY CAGR |
| `sharpe` | 샤프 비율 |
| `max_drawdown` | 최대 낙폭 (음수) |
| `hit_rate` | 수익 발생 거래일 비율 |
| `annualized_volatility` | 연환산 전략 변동성 |
