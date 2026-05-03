from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TICKERS = ("SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM")


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
