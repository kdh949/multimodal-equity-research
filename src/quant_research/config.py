from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BENCHMARK_TICKER = "SPY"
DEFAULT_EXCLUDED_STRATEGY_TICKERS = ("SPY", "QQQ", "DIA", "IWM")
DEFAULT_TICKERS = (
    "AAPL",
    "ABBV",
    "ABT",
    "ACN",
    "ADBE",
    "AIG",
    "AMD",
    "AMGN",
    "AMT",
    "AMZN",
    "AVGO",
    "AXP",
    "BA",
    "BAC",
    "BK",
    "BKNG",
    "BLK",
    "BMY",
    "C",
    "CAT",
    "CHTR",
    "CL",
    "CMCSA",
    "COF",
    "COP",
    "COST",
    "CRM",
    "CSCO",
    "CVS",
    "CVX",
    "DE",
    "DHR",
    "DIS",
    "DUK",
    "EMR",
    "F",
    "FDX",
    "GD",
    "GE",
    "GILD",
    "GM",
    "GOOG",
    "GOOGL",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC",
    "JNJ",
    "JPM",
    "KHC",
    "KO",
    "LIN",
    "LLY",
    "LMT",
    "LOW",
    "MA",
    "MCD",
    "MDLZ",
    "MDT",
    "MET",
    "META",
    "MMM",
    "MO",
    "MRK",
    "MS",
    "MSFT",
    "NEE",
    "NFLX",
    "NKE",
    "NVDA",
    "ORCL",
    "PEP",
    "PFE",
    "PG",
    "PM",
    "PYPL",
    "QCOM",
    "RTX",
    "SBUX",
    "SCHW",
    "SO",
    "SPG",
    "T",
    "TGT",
    "TMO",
    "TMUS",
    "TSLA",
    "TXN",
    "UNH",
    "UNP",
    "UPS",
    "USB",
    "V",
    "VZ",
    "WBA",
    "WFC",
    "WMT",
    "XOM",
)


@dataclass(frozen=True)
class AppPaths:
    root: Path = Path(".")
    raw_data: Path = Path("data/raw")
    processed_data: Path = Path("data/processed")
    artifacts: Path = Path("artifacts")
    reports: Path = Path("reports")
    sec_cache: Path = Path("data/raw/sec")


@dataclass(frozen=True)
class SecSettings:
    user_agent: str = field(
        default_factory=lambda: os.getenv("QT_SEC_USER_AGENT", "QuantResearchApp research@example.com")
    )
    max_requests_per_second: float = 9.0
