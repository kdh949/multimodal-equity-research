from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Protocol

import numpy as np
import pandas as pd


class MarketDataProvider(Protocol):
    def get_history(
        self,
        tickers: list[str],
        start: str | date | None = None,
        end: str | date | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Return normalized OHLCV data with date, ticker, open, high, low, close, adj_close, volume."""


@dataclass
class SyntheticMarketDataProvider:
    periods: int = 260
    seed: int = 42

    def get_history(
        self,
        tickers: list[str],
        start: str | date | None = None,
        end: str | date | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        del start, interval
        rng = np.random.default_rng(self.seed)
        end_date = pd.Timestamp(end or date.today()).normalize()
        dates = pd.bdate_range(end=end_date, periods=self.periods)
        frames: list[pd.DataFrame] = []

        market_shock = rng.normal(0.00025, 0.009, len(dates))
        for idx, ticker in enumerate(tickers):
            beta = 0.75 + idx * 0.04
            idiosyncratic = rng.normal(0.00015 + idx * 0.00002, 0.012 + idx * 0.0007, len(dates))
            returns = beta * market_shock + idiosyncratic
            close = 100 * np.exp(np.cumsum(returns))
            open_ = close * (1 + rng.normal(0, 0.002, len(dates)))
            high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.004, 0.002, len(dates))))
            low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.004, 0.002, len(dates))))
            volume = rng.lognormal(mean=15.5 + idx * 0.025, sigma=0.25, size=len(dates)).astype(int)

            frames.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "ticker": ticker,
                        "open": open_,
                        "high": high,
                        "low": low,
                        "close": close,
                        "adj_close": close,
                        "volume": volume,
                    }
                )
            )

        return pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


@dataclass
class YFinanceMarketDataProvider:
    auto_adjust: bool = False
    progress: bool = False

    def get_history(
        self,
        tickers: list[str],
        start: str | date | None = None,
        end: str | date | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise RuntimeError("yfinance is required for live market data") from exc

        start = start or (date.today() - timedelta(days=365 * 3))
        end = end or date.today()
        raw = yf.download(
            tickers=tickers,
            start=str(start),
            end=str(end),
            interval=interval,
            auto_adjust=self.auto_adjust,
            group_by="ticker",
            progress=self.progress,
            threads=True,
        )
        if raw.empty:
            return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"])

        frames: list[pd.DataFrame] = []
        if isinstance(raw.columns, pd.MultiIndex):
            for ticker in tickers:
                if ticker not in raw.columns.get_level_values(0):
                    continue
                part = raw[ticker].copy()
                frames.append(_normalize_yfinance_frame(part, ticker))
        else:
            frames.append(_normalize_yfinance_frame(raw, tickers[0]))

        return pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


@dataclass
class LocalMarketDataProvider:
    """Reads pre-downloaded OHLCV data from a Parquet file (no network calls)."""

    data_path: str = "data/raw/market_history.parquet"

    def get_history(
        self,
        tickers: list[str],
        start: str | date | None = None,
        end: str | date | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        from pathlib import Path

        path = Path(self.data_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Local market data not found at '{path}'. "
                "Run: uv run python scripts/download_backtest_data.py"
            )
        frame = pd.read_parquet(path)
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        if tickers:
            frame = frame[frame["ticker"].isin(tickers)]
        if start:
            frame = frame[frame["date"] >= pd.Timestamp(start).normalize()]
        if end:
            frame = frame[frame["date"] <= pd.Timestamp(end).normalize()]
        return frame.sort_values(["date", "ticker"]).reset_index(drop=True)


def _normalize_yfinance_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    renamed = {column: str(column).lower().replace(" ", "_") for column in frame.columns}
    normalized = frame.rename(columns=renamed).reset_index()
    date_col = "date" if "date" in normalized.columns else normalized.columns[0]
    normalized = normalized.rename(columns={date_col: "date"})
    if "adj_close" not in normalized.columns:
        normalized["adj_close"] = normalized.get("close")
    normalized["ticker"] = ticker
    columns = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    for column in columns:
        if column not in normalized.columns:
            normalized[column] = np.nan
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.tz_localize(None).dt.normalize()
    return normalized[columns].dropna(subset=["close"])
