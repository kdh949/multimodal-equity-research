# 데이터 타임스탬프 스키마

이 문서는 가격, 펀더멘털, 뉴스/텍스트, SEC EDGAR 데이터에 공통 적용하는
타임스탬프 스키마를 정의한다. 목적은 모든 feature가 `t` 시점에 실제로 사용
가능했던 데이터만 사용하도록 강제하고, `forward_return_20` 검증에서 미래 데이터
누수를 방지하는 것이다.

## 공통 원칙

모든 원천 데이터 row와 feature row는 아래 세 가지 시각을 구분한다.

| 컬럼 | 필수 | 타입 | 의미 |
|---|---:|---|---|
| `event_timestamp` | 필수 | timezone-aware UTC timestamp | 데이터가 설명하는 경제적 사건, 시장 관측치, 공시 기간 또는 뉴스 사건의 기준 시각 |
| `availability_timestamp` | 필수 | timezone-aware UTC timestamp | 해당 row가 리서치 시스템에서 feature로 사용 가능해진 가장 이른 시각 |
| `source_timestamp` | 조건부 | timezone-aware UTC timestamp 또는 null | 원천 provider가 명시한 발행, 접수, 수정, 업데이트 시각 |
| `timezone` | 필수 | string | 원천 시각을 해석할 때 사용한 IANA timezone 또는 `UTC` |

누수 방지 규칙:

- feature는 평가 기준 시각 `t`에 대해 `availability_timestamp <= t`인 row만 사용할 수 있다.
- `event_timestamp`가 과거라도 `availability_timestamp`가 미래이면 사용할 수 없다.
- provider가 시각 없이 날짜만 제공하면, 원천 timezone의 해당 날짜 종료 시각으로 해석한 뒤 UTC로 변환한다.
- 원천 timezone을 알 수 없으면 `UTC`로 해석하고, `timezone`에는 `UTC`를 기록한다.
- 변환 후 저장되는 세 timestamp는 모두 timezone-aware UTC 값이어야 한다.
- naive timestamp는 저장하지 않는다. 입력이 naive이면 provider adapter에서 명시적으로 timezone을 부여한 뒤 UTC로 변환한다.

## 컬럼 의미

### event_timestamp

`event_timestamp`는 row가 표현하는 사실이 발생했거나 귀속되는 시각이다.

- 가격 일봉: 해당 거래일의 정규장 종가 시각
- 분봉/시간봉 가격: 해당 bar 종료 시각
- 펀더멘털 기간값: 회계 기간 종료일의 종료 시각
- 뉴스 기사: 기사 본문이 다루는 사건 시각을 알 수 있으면 그 시각, 없으면 기사 발행 시각
- SEC filing 이벤트: 공시가 설명하는 회계 기간 종료일 또는 거래/보고 이벤트 시각

`event_timestamp`는 모델 sample 정렬과 horizon 계산의 기준이지만, 단독으로 feature 사용 가능성을 보장하지 않는다.

### availability_timestamp

`availability_timestamp`는 feature engineering과 walk-forward 검증에서 가장 중요한 시각이다.
시스템은 이 값이 평가 기준 시각보다 늦은 row를 반드시 제외해야 한다.

보수적 산정 규칙:

- 원천 발행/접수/공개 시각이 있으면 그 시각을 UTC로 변환한다.
- 원천 시각이 날짜 단위만 있으면 해당 날짜의 원천 timezone 종료 시각으로 둔다.
- 원천 공개 시각을 알 수 없고 provider fetch 시각만 있으면 fetch 시각을 사용한다.
- 가격 일봉처럼 시장 종료 전 확정될 수 없는 데이터는 적어도 해당 bar 종료 시각 이후로 둔다.
- 수정 데이터가 뒤늦게 반영된 경우 원천 수정 시각 또는 fetch 시각 중 더 늦은 값을 사용한다.

### source_timestamp

`source_timestamp`는 provider가 제공한 원천 발행, 접수, 수정 또는 업데이트 시각이다.
원천이 값을 제공하지 않으면 null을 허용한다. null인 경우에도 `availability_timestamp`는 반드시 채워야 한다.

예:

- SEC EDGAR `acceptanceDateTime`
- 뉴스 provider의 `published_at` 또는 `updated_at`
- 펀더멘털 provider의 `asOfDate`, `reportedDate`, `filedDate`
- 가격 vendor의 bar update timestamp

## 데이터 유형별 규칙

| 데이터 유형 | `event_timestamp` | `availability_timestamp` | `source_timestamp` | 기본 timezone |
|---|---|---|---|---|
| 가격 OHLCV | bar 종료 시각. 일봉은 미국 정규장 종가 시각 | bar 종료 시각과 provider 공개/fetch 시각 중 더 늦은 값 | vendor update 시각이 있으면 사용, 없으면 null | `America/New_York` |
| 펀더멘털 | 회계 기간 종료 시각 또는 metric 기준일 | filing, report, vendor 공개 시각 중 가장 보수적인 값 | provider의 report/file/update 시각 | provider 명시값, 없으면 `UTC` |
| 뉴스/텍스트 | 사건 시각이 있으면 사건 시각, 없으면 기사 발행 시각 | 기사 발행/수정 시각과 수집 시각 중 더 늦은 값 | 기사 발행/수정 시각 | provider 명시값, 없으면 `UTC` |
| SEC filing metadata | filing이 보고하는 기간 종료 또는 보고 이벤트 시각 | EDGAR 접수 공개 시각 | EDGAR `acceptanceDateTime` | `UTC` |
| SEC XBRL facts | XBRL fact의 period end 시각 | 해당 fact를 포함한 filing의 EDGAR 접수 공개 시각 | filing `acceptanceDateTime` 또는 fact update 시각 | `UTC` |

## Null 허용 정책

| 컬럼 | 원천 row | feature row | null 허용 여부 |
|---|---:|---:|---|
| `symbol` | 필수 | 필수 | 불가. market-wide row는 명시적 universe key를 사용한다 |
| `event_timestamp` | 필수 | 필수 | 불가 |
| `availability_timestamp` | 필수 | 필수 | 불가 |
| `source_timestamp` | 조건부 | 조건부 | provider가 원천 시각을 제공하지 않을 때만 허용 |
| `timezone` | 필수 | 필수 | 불가 |
| feature value | 선택 | 선택 | 원천 결측 또는 adapter fallback에서 허용. 결측 처리 정책을 feature metadata에 기록한다 |

`availability_timestamp`가 null인 row는 학습, 검증, 백테스트, report artifact에 포함할 수 없다.
이 경우 adapter 단계에서 row를 제외하거나 명시적인 validation error를 발생시킨다.

## Feature Availability Cutoff

각 feature builder는 산출 feature마다 아래 metadata를 남겨야 한다.

| 필드 | 의미 |
|---|---|
| `feature_name` | feature 컬럼명 |
| `source_family` | `price`, `fundamental`, `text`, `sec` 중 하나 |
| `event_timestamp_column` | 기준 event 컬럼명 |
| `availability_timestamp_column` | cutoff 검증에 사용할 availability 컬럼명 |
| `max_allowed_timestamp` | sample 기준 시각 `t` |
| `cutoff_rule` | `availability_timestamp <= t` 같은 적용 규칙 |
| `null_policy` | null 처리 방식 |

walk-forward 검증과 system validity gate는 fold별 sample에 대해 모든 feature row가
`availability_timestamp <= sample_timestamp`를 만족하는지 확인해야 한다.

코드 레벨 표준 스키마는 `quant_research.data.timestamps.FeatureAvailabilitySchema`에
정의한다. provider별 원천 컬럼은
`standardize_feature_availability_metadata()`로 `as_of_timestamp`,
`publication_timestamp`, `availability_timestamp`, `timezone` 컬럼으로 정규화하고,
`validate_feature_availability()`로 sample cutoff, null metadata, UTC timestamp 저장 여부를
검증한다. 검증 결과는 `FeatureAvailabilityValidationResult.to_dict()`로 artifact manifest에
저장할 수 있는 구조화 dict를 반환한다.

## Artifact 요구사항

canonical experiment artifact에는 최소한 아래 항목을 포함한다.

- 이 문서의 스키마 버전 또는 문서 경로
- 데이터 family별 timestamp normalization 규칙
- feature availability cutoff manifest
- `source_timestamp` null row 수와 비율
- `availability_timestamp` null row 수. 이 값은 항상 0이어야 한다
- timezone 변환 실패 또는 naive timestamp reject 건수

## v1 범위 메모

v1은 생존편향 있는 universe snapshot을 허용하지만 artifact와 report에 명시한다.
타임스탬프 스키마는 생존편향 허용과 별개로 모든 row-level feature availability를 검증해야 한다.
point-in-time universe 재구성은 v2 확인 대상이다.
