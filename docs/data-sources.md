# Data Sources

모든 원천 데이터와 feature row는 [데이터 타임스탬프 스키마](data-timestamp-schema.md)의
`event_timestamp`, `availability_timestamp`, `source_timestamp`, timezone, null 허용 정책을 따른다.

## Market Data

v1은 `yfinance`를 기본 무료 provider로 사용한다.

- OHLCV
- adjusted close
- volume
- interval 기반 일봉/시간봉 확장 지점

무료 데이터는 연구/개인 검증용으로 제한한다. provider 인터페이스를 통해 Polygon, Tiingo, Nasdaq Data Link 같은 유료 소스로 교체 가능하게 설계한다.

## News Data

v1 무료 뉴스 소스:

- `yfinance` ticker news
- GDELT DOC API

뉴스 feature:

- 기사 수
- 평균 감성
- 부정 감성 비율
- 소스 다양성
- 이벤트 키워드
- recency decay score

## SEC EDGAR

공식 SEC API를 사용한다.

- `submissions`: filing history, 8-K/10-Q/10-K/Form 4 이벤트
- `companyfacts`: 기업별 XBRL fact 전체
- `companyconcept`: 특정 US-GAAP concept 시계열
- `frames`: calendar period 기준 cross-company fact

SEC 접근 규칙:

- 인증키는 필요 없다.
- `User-Agent` 헤더를 명시한다. 로컬에서는 `QT_SEC_USER_AGENT` 환경변수로 연락처를 설정한다.
- 요청 속도는 10 req/s 이하로 제한한다.
- 원천 응답은 로컬 캐시에 저장하고 커밋하지 않는다.

## Cache Policy

`data/raw`, `data/processed`, `artifacts`, `reports`, `.uv-cache`는 커밋하지 않는다.
