# 백테스트 검증 개요 및 결과 구조

## 목적

이 검증 스크립트는 LightGBM 기반 주식 수익률 예측 모델이 합성(Synthetic) 데이터가 아닌
**실제 시장 데이터**에서도 유효한 예측력을 갖는지 확인한다.

## 검증 대상 모델

| 컴포넌트 | 설명 |
|---|---|
| 예측 모델 | LightGBM (TabularReturnModel) |
| 검증 방식 | Walk-Forward Cross-Validation |
| 예측 대상 | 20일 선행 수익률 (`forward_return_20`) |
| 입력 피처 | 가격·기술지표·거래량·뉴스감성·SEC공시 등 융합 피처 |

## 사용 데이터

| 항목 | 내용 |
|---|---|
| 데이터 소스 | yfinance (인터넷 실시간) |
| 분석 종목 | 대표 미국 대형 유동주 30개 기본 유니버스 |
| 분석 기간 | 실행일 기준 과거 2년 |
| 뉴스 소스 | yfinance News API + GDELT |
| 공시 소스 | SEC EDGAR API (CIK 미등록 종목은 중립 SEC feature로 계속 진행) |
| 벤치마크 | SPY (전략 후보가 아니라 비교 기준 데이터) |

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
Walk-Forward 분할 (기본 20일 target, purge/embargo 최소 60 거래일)
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
| `expected_return` | 모델이 예측한 20일 수익률 |
| `forward_return_20` | 실제 발생한 20일 수익률 |
| `forward_return_1`, `forward_return_5` | 진단용 단기 horizon 수익률 |
| `fold` | 속한 Walk-Forward fold 번호 |
| `is_oos` | 마지막 fold(Out-of-Sample) 여부 |

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
