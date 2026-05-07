"""One-time data download script for offline backtesting.

Downloads all market, news, and SEC data to disk so subsequent runs
use data_mode="local" without any network calls.

Usage:
    uv run python scripts/download_backtest_data.py
    uv run python scripts/download_backtest_data.py --start 2022-01-01 --end 2024-12-31
    uv run python scripts/download_backtest_data.py --tickers AAPL MSFT NVDA
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from quant_research.config import DEFAULT_TICKERS
from quant_research.pipeline import CIK_BY_TICKER

OUTPUT_DIR = Path("data/raw")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-download all backtest data for offline use.")
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS), metavar="TICKER")
    parser.add_argument("--start", default=str(date.today() - timedelta(days=365 * 2)), metavar="YYYY-MM-DD")
    parser.add_argument("--end", default=str(date.today()), metavar="YYYY-MM-DD")
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR), metavar="DIR")
    parser.add_argument("--skip-news", action="store_true", help="Skip news download (fastest to skip)")
    parser.add_argument("--skip-sec", action="store_true", help="Skip SEC filing download")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

def download_market(tickers: list[str], start: str, end: str, out_dir: Path) -> None:
    print(f"\n[1/3] Market data — {len(tickers)} tickers from {start} to {end}")
    try:
        import yfinance as yf
    except ImportError:
        print("  SKIP: yfinance not installed")
        return

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        progress=True,
        threads=True,
    )
    if raw.empty:
        print("  WARNING: yfinance returned no data")
        return

    from quant_research.data.market import _normalize_yfinance_frame

    frames: list[pd.DataFrame] = []
    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker not in raw.columns.get_level_values(0):
                print(f"  WARNING: no data for {ticker}")
                continue
            frames.append(_normalize_yfinance_frame(raw[ticker].copy(), ticker))
    else:
        frames.append(_normalize_yfinance_frame(raw, tickers[0]))

    if not frames:
        print("  WARNING: no frames to save")
        return

    combined = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    out_path = out_dir / "market_history.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"  Saved {len(combined):,} rows → {out_path}")


# ---------------------------------------------------------------------------
# News data
# ---------------------------------------------------------------------------

def _news_item_to_dict(item: object) -> dict:
    return {
        "ticker": item.ticker,
        "published_at": str(item.published_at),
        "title": item.title,
        "source": item.source,
        "url": item.url,
        "summary": item.summary,
        "content": item.content,
        "full_text": item.full_text,
        "body_text": item.body_text,
    }


def download_news(tickers: list[str], start: str, end: str, out_dir: Path) -> None:
    print(f"\n[2/3] News data — {len(tickers)} tickers")
    from quant_research.data.news import GDELTNewsProvider, YFinanceNewsProvider

    all_items: list[object] = []

    print("  Fetching yfinance news…")
    yf_provider = YFinanceNewsProvider()
    for ticker in tickers:
        try:
            items = yf_provider.get_news([ticker], start, end)
            all_items.extend(items)
            print(f"    {ticker}: {len(items)} items")
        except Exception as exc:
            print(f"    {ticker}: ERROR — {exc}")
        time.sleep(0.5)

    print("  Fetching GDELT news…")
    gdelt = GDELTNewsProvider()
    for ticker in tickers:
        try:
            items = gdelt.get_news([ticker], start, end)
            all_items.extend(items)
            print(f"    {ticker}: {len(items)} items")
        except Exception as exc:
            print(f"    {ticker}: ERROR — {exc}")
        time.sleep(1.0)

    out_path = out_dir / "news_items.jsonl"
    with open(out_path, "w") as fh:
        for item in all_items:
            fh.write(json.dumps(_news_item_to_dict(item), ensure_ascii=False) + "\n")
    print(f"  Saved {len(all_items):,} news items → {out_path}")


# ---------------------------------------------------------------------------
# SEC data  (SecEdgarClient already auto-caches to disk)
# ---------------------------------------------------------------------------

def download_sec(tickers: list[str], start: str, end: str, out_dir: Path) -> None:  # noqa: ARG001
    print(f"\n[3/3] SEC data — {len(tickers)} tickers (auto-cached to {out_dir / 'sec'})")
    from quant_research.data.sec import (
        SecEdgarClient,
        extract_frame_values,
    )

    sec_dir = out_dir / "sec"
    client = SecEdgarClient(cache_dir=sec_dir)

    for ticker in tickers:
        cik = CIK_BY_TICKER.get(ticker)
        if cik is None:
            print(f"  {ticker}: no CIK mapping, skipping")
            continue
        print(f"  {ticker} (CIK {cik})…", end=" ", flush=True)
        try:
            client.get_submissions(cik)
            filings = client.recent_filings(cik, {"8-K", "10-Q", "10-K", "4"}, include_document_text=True)
            client.get_companyfacts(cik)
            client.get_companyconcept(cik, "us-gaap", "NetIncomeLoss")
            print(f"OK ({len(filings)} filings cached)")
        except Exception as exc:
            print(f"ERROR — {exc}")
        time.sleep(0.15)

    print("  Fetching SEC frame (assets)…", end=" ", flush=True)
    try:
        extract_frame_values(client.get_frame("us-gaap", "Assets", "USD", "CY2024Q4I"))
        print("OK")
    except Exception as exc:
        print(f"ERROR — {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Backtest Data Downloader")
    print(f"  tickers  : {args.tickers}")
    print(f"  period   : {args.start} → {args.end}")
    print(f"  output   : {out_dir.resolve()}")
    print("=" * 60)

    download_market(args.tickers, args.start, args.end, out_dir)

    if not args.skip_news:
        download_news(args.tickers, args.start, args.end, out_dir)
    else:
        print("\n[2/3] News — skipped")

    if not args.skip_sec:
        download_sec(args.tickers, args.start, args.end, out_dir)
    else:
        print("\n[3/3] SEC — skipped")

    print("\nDone! Run backtests with data_mode='local' to use cached data.")
    print(f"  e.g. PipelineConfig(data_mode='local', local_data_dir='{out_dir}')")


if __name__ == "__main__":
    main()
