# 백테스트 검증 실행 방법

두 가지 실행 모드를 제공한다.

| 모드 | 실행 시간 | 사용 모델 |
|---|---|---|
| **경량 모드** | 3~10분 | LightGBM만 사용 (기본) |
| **실전 모드** | 30~90분 | LightGBM + FinBERT + FinGPT (전체 스택) |

---

## 경량 모드 (빠른 검증)

### 사전 요건

- `uv` 설치 완료
- 인터넷 연결

### 실행

```bash
uv run python scripts/run_backtest_validation.py
```

대형 LLM 모델을 다운로드하지 않는다. LightGBM 예측력만 빠르게 검증할 때 사용한다.

---

## 실전 모드 (FinGPT 포함 전체 스택)

FinBERT(뉴스 감성 분석)와 FinGPT(SEC 공시 이벤트 추출)를 실제로 활성화해서 검증한다.
경량 모드보다 피처 품질이 높고, 실제 운영 환경과 동일한 조건으로 백테스트가 진행된다.

### 1단계: 의존성 설치

```bash
# NLP 모델 (FinBERT용: torch, transformers)
uv sync --extra nlp

# LLM 모델 (FinGPT용: peft, accelerate, sentencepiece 등)
uv sync --extra llm

# Apple Silicon(M1/M2/M3) macOS — MLX 런타임
uv add mlx-lm
```

> **Apple Silicon이 아닌 경우** (Intel Mac, Linux 등): `mlx-lm` 대신 `llama-cpp-python`을 설치하고, 아래 FinGPT 설정에서 `fingpt_runtime="llama-cpp"`를 사용한다.

### 2단계: 환경 변수 설정

FinGPT의 베이스 모델(`meta-llama/Meta-Llama-3-8B`)은 Meta 라이선스가 필요한 게이티드(gated) 모델이다.

**2-1. HuggingFace에서 Llama 3 라이선스 동의**

아래 링크에서 로그인 후 "Agree and access repository" 클릭:
```
https://huggingface.co/meta-llama/Meta-Llama-3-8B
```

**2-2. HuggingFace Access Token 발급 및 환경 변수 설정**

```bash
# .env 파일에 토큰 추가
echo 'HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"' >> .env

# 또는 셸에서 직접 설정
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
```

**2-3. SEC User-Agent 설정 (권장)**

```bash
export QT_SEC_USER_AGENT="YourName your@email.com"
```

### 3단계: FinGPT 모델 다운로드

```bash
# Apple Silicon macOS — MLX 형식으로 다운로드
uv run python scripts/preload_local_models.py \
    --fingpt \
    --mode download \
    --fingpt-runtime mlx \
    --fingpt-quantized-model-path artifacts/model_cache/fingpt-mt-llama3-8b-mlx
```

```bash
# Intel Mac / Linux — llama.cpp GGUF 형식으로 다운로드
uv run python scripts/preload_local_models.py \
    --fingpt \
    --mode download \
    --fingpt-runtime llama-cpp \
    --fingpt-quantized-model-path artifacts/model_cache/fingpt-mt-llama3-8b-lora-q4_0.gguf
```

FinBERT는 첫 실행 시 자동으로 다운로드(`ProsusAI/finbert`, 약 440MB)되므로 별도 다운로드 불필요.

### 4단계: 실행

`--runtime` 플래그로 FinGPT 런타임을 선택한다. 코드 수정 불필요.

```bash
# Apple Silicon — MLX (기본값, 가장 빠름)
uv run python scripts/run_backtest_validation.py --mode full --runtime mlx

# Ollama 서버 사용 (ollama에 fingpt 모델 설치 필요)
uv run python scripts/run_backtest_validation.py --mode full --runtime ollama
```

> **런타임 비교**
>
> | 런타임 | 하드웨어 가속 | 비고 |
> |---|---|---|
> | `mlx` | Apple Neural Engine + Metal (기본) | `scripts/convert_fingpt_to_mlx.py` 변환 선행 필요 |
> | `ollama` | Metal GPU (llama.cpp 기반) | `ollama pull fingpt` 선행 필요 |

| 단계 | 예상 시간 | 비고 |
|---|---|---|
| yfinance 데이터 수집 | 1~3분 | 10개 종목 2년치 |
| FinBERT 첫 로드 | 1~2분 | 이후 캐시됨 |
| FinGPT 공시 분석 | 20~60분 | 종목·기간에 따라 다름 |
| Walk-Forward 학습/예측 | 5~15분 | fold 수에 비례 |
| 백테스트 | 1~2분 | |

### 6단계: 결과 파일 확인

```bash
# 저장된 파일 목록
ls reports/backtest_validation_$(date +%Y%m%d)/

# 포트폴리오 지표 (JSON)
cat reports/backtest_validation_$(date +%Y%m%d)/metrics.json

# fold별 예측 정확도
cat reports/backtest_validation_$(date +%Y%m%d)/validation_summary.csv

# 예측 데이터 상위 10행
head -n 11 reports/backtest_validation_$(date +%Y%m%d)/predictions.csv
```

---

## 터미널 출력 구조 (공통)

두 모드 모두 동일한 4개 섹션을 출력한다.

```
[1] Walk-Forward 검증 결과 — fold별 MAE + 방향성 정확도 + PASS/FAIL 판정
[2] 종목별 예측 정확도      — 종목별 MAE, 방향성 정확도, 평균 예측/실제 수익률
[3] 포트폴리오 백테스트 지표 — CAGR, Sharpe, 최대 낙폭, 초과 수익률 등
[4] OOS 예측 vs 실제 샘플  — 최근 20개 Out-of-Sample 예측과 실제 비교
```

---

## 문제 해결

| 증상 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError: pandas` | 시스템 Python 사용 | `uv run python`으로 실행 |
| `ModuleNotFoundError: mlx_lm` | mlx-lm 미설치 | `uv add mlx-lm` |
| `ModuleNotFoundError: peft` | llm extras 미설치 | `uv sync --extra llm` |
| `401 Unauthorized` (HuggingFace) | HF_TOKEN 미설정 또는 라이선스 미동의 | 2단계 참고 |
| `RuntimeError: mlx_lm is required` | mlx-lm 미설치 | `uv add mlx-lm` |
| `Walk-Forward fold가 생성되지 않았습니다` | 데이터 기간 부족 | `DATE_RANGE_YEARS` 증가 또는 `train_periods` 감소 |
| SEC 요청 실패 | Rate limit 초과 | 자동 재시도 내장, 잠시 대기 |
| yfinance 빈 DataFrame | 네트워크 오류 | 재시도 (합성 데이터로 자동 폴백됨) |
| FinGPT 오류 후 규칙 기반으로 폴백 | 모델 경로 오류 | `fingpt_quantized_model_path` 경로 확인 |
