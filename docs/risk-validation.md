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
- gap을 둘 수 있어야 한다.
- test window는 train 이후 구간만 사용한다.
- 마지막 구간은 out-of-sample summary로 별도 표시한다.

## Backtest Rules

- 기본 전략은 상위 점수 종목 동일가중 롱온리다.
- 포지션은 신호 생성 다음 기간 수익률부터 적용한다.
- 비용은 turnover 기준으로 차감한다.
- 슬리피지는 거래된 notional 기준으로 차감한다.
- 벤치마크는 `SPY`다.

## Risk Rules

- 종목별 최대 비중
- 포트폴리오 변동성 한도
- 최대 낙폭 중단 룰
- 이벤트 리스크 차단 룰
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
