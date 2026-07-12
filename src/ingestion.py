"""
Data ingestion for VN30 stocks via vnstock.

Fetches daily OHLCV from VCI source, normalizes columns, handles
adjusted prices (already adjusted by vnstock for splits/dividends),
and persists to parquet under data/raw/.

Usage:
    python -m src.ingestion --ticker FPT
    python -m src.ingestion --universe vn30
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from vnstock import Vnstock

# Resolve project root regardless of CWD
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# VN30 universe (as of late 2024 - representative large-cap, liquid names).
# Using 10 names keeps the project tractable while still showing universe-level thinking.
VN30_UNIVERSE = [
    "FPT", "VCB", "VHM", "VIC", "VNM",
    "HPG", "MWG", "MSN", "TCB", "VPB",
]


def fetch_one(ticker: str, start: str, end: str, source: str = "VCI") -> pd.DataFrame:
    """Fetch OHLCV for a single ticker.

    Returns DataFrame with columns: time, open, high, low, close, volume.
    Sorted ascending by time. Adjusted close is already handled by vnstock.
    """
    stock = Vnstock().stock(symbol=ticker, source=source)
    df = stock.quote.history(start=start, end=end)

    expected = {"time", "open", "high", "low", "close", "volume"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{ticker}: missing columns {missing}")

    df = df[list(expected)].copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df = df.dropna(subset=["close"])
    df = df[df["volume"] >= 0]
    return df


def fetch_universe(
    tickers: Iterable[str],
    start: str,
    end: str,
    sleep_s: float = 0.6,
) -> dict[str, pd.DataFrame]:
    """Fetch multiple tickers sequentially with a small delay to be polite to the API."""
    out: dict[str, pd.DataFrame] = {}
    for tkr in tickers:
        try:
            df = fetch_one(tkr, start=start, end=end)
            out[tkr] = df
            print(f"  {tkr}: {len(df):>5d} rows  [{df['time'].min().date()} -> {df['time'].max().date()}]")
        except Exception as e:  # noqa: BLE001 - we want to keep going on a bad ticker
            print(f"  {tkr}: FAILED ({e})", file=sys.stderr)
        time.sleep(sleep_s)
    return out


def save_parquet(data: dict[str, pd.DataFrame], out_dir: Path = RAW_DIR) -> list[Path]:
    """Save each ticker to its own parquet file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for tkr, df in data.items():
        p = out_dir / f"{tkr}.parquet"
        df.to_parquet(p, index=False)
        paths.append(p)
    return paths


def load_one(ticker: str, in_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load a single ticker from parquet."""
    p = in_dir / f"{ticker}.parquet"
    return pd.read_parquet(p)


def load_universe(tickers: Iterable[str], in_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    return {tkr: load_one(tkr, in_dir) for tkr in tickers if (in_dir / f"{tkr}.parquet").exists()}


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest VN30 OHLCV via vnstock")
    p.add_argument("--ticker", type=str, default=None, help="Single ticker, e.g. FPT")
    p.add_argument("--universe", action="store_true", help="Fetch the full VN30 subset")
    p.add_argument("--start", type=str, default="2019-01-01")
    p.add_argument("--end", type=str, default="2025-12-31")
    p.add_argument("--source", type=str, default="VCI")
    return p.parse_args()


def main() -> None:
    args = build_args()
    if args.ticker:
        tickers = [args.ticker.upper()]
    elif args.universe:
        tickers = VN30_UNIVERSE
    else:
        raise SystemExit("Specify --ticker FPT or --universe")

    print(f"Fetching {len(tickers)} tickers from {args.start} to {args.end} via {args.source} ...")
    data = fetch_universe(tickers, start=args.start, end=args.end)
    paths = save_parquet(data)
    print(f"Saved {len(paths)} parquet files to {RAW_DIR}")


if __name__ == "__main__":
    main()