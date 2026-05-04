# 백테스트 검증 실행 방법

## 사전 요건

- Python 패키지 매니저 `uv` 설치 완료
- 인터넷 연결 (yfinance, SEC EDGAR, GDELT 데이터 수집)
- `QT_SEC_USER_AGENT` 환경변수 설정 권장 (미설정 시 기본값 사용)

## 기본 실행

프로젝트 루트 디렉터리에서 실행한다.

```bash
uv run python scripts/run_backtest_validation.py
```

실행 시간은 인터넷 속도와 종목 수에 따라 **3~10분** 소요된다.
대형 LLM 모델을 다운로드하지 않으므로 첫 실행도 빠르다.

## 실행 모드 설명

스크립트는 다음 경량 설정으로 고정 실행된다. 무거운 모델 없이 핵심 LightGBM 예측만 검증한다.

| 설정 | 값 | 이유 |
|---|---|---|
| `data_mode` | `live` | yfinance 실제 데이터 |
| `sentiment_model` | `keyword` | FinBERT 다운로드 없이 빠른 실행 |
| `filing_extractor_model` | `rules` | FinGPT/FinMA 없이 규칙 기반 |
| `time_series_inference_mode` | `proxy` | Chronos/GraniteTTM 없이 proxy 신호 |

## 설정 변경 방법

`scripts/run_backtest_validation.py` 내 `build_config()` 함수를 수정한다.

```python
def build_config() -> PipelineConfig:
    end = date.today()
    start = end - timedelta(days=365 * DATE_RANGE_YEARS)  # 기간 변경: DATE_RANGE_YEARS 수정
    return PipelineConfig(
        tickers=TICKERS,          # 종목 변경: TICKERS 상수 수정
        data_mode="live",
        start=start,
        end=end,
        sentiment_model="keyword",     # "finbert"으로 변경 시 FinBERT 활성화
        filing_extractor_model="rules", # "finma" 또는 "fingpt"로 변경 가능
        time_series_inference_mode="proxy",  # "local"로 변경 시 Chronos 활성화
    )
```

### 자주 쓰는 변경 예시

```python
# 분석 기간을 3년으로 늘리기
DATE_RANGE_YEARS = 3

# 특정 종목만 검증
TICKERS = ["AAPL", "MSFT", "NVDA"]

# FinBERT 감성 모델 사용 (다운로드 필요)
sentiment_model="finbert"
```

## 터미널 출력 구조

실행하면 4개 섹션이 순서대로 출력된다.

```
[1] Walk-Forward 검증 결과 — fold별 MAE + 방향성 정확도 + PASS/FAIL 판정
[2] 종목별 예측 정확도      — 종목별 MAE, 방향성 정확도, 평균 예측/실제 수익률
[3] 포트폴리오 백테스트 지표 — CAGR, Sharpe, 최대 낙폭, 초과 수익률 등
[4] OOS 예측 vs 실제 샘플  — 최근 20개 Out-of-Sample 예측과 실제 비교
```

## 결과 파일 확인

실행 후 생성된 파일을 확인한다.

```bash
# 파일 목록 확인
ls reports/backtest_validation_$(date +%Y%m%d)/

# 포트폴리오 지표 확인
cat reports/backtest_validation_$(date +%Y%m%d)/metrics.json

# 예측 데이터 상위 10행 확인
head -n 11 reports/backtest_validation_$(date +%Y%m%d)/predictions.csv

# fold별 정확도 확인
cat reports/backtest_validation_$(date +%Y%m%d)/validation_summary.csv
```

## 문제 해결

| 증상 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError: pandas` | 시스템 Python 사용 | `uv run python` 으로 실행 |
| yfinance 데이터 빈 DataFrame | 네트워크 오류 | 재시도 또는 인터넷 연결 확인 (합성 데이터로 자동 폴백됨) |
| `Walk-Forward fold가 생성되지 않았습니다` | 데이터 기간 부족 | `DATE_RANGE_YEARS`를 늘리거나 `train_periods`를 줄임 |
| SEC 요청 실패 | Rate limit 초과 | 자동 재시도 내장, 기다리면 해결 |
