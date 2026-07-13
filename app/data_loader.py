"""Helpers that turn the artifacts under `reports/` into DataFrames for the
dashboard. All readers are tolerant to missing keys and NaN values; the dashboard
should still render if one block is missing.

The data shape mirrors what `src.robustness.py` and `src.report.py` emit, so
re-running either upstream task regenerates the inputs transparently.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
EQUITY_FILE = REPORT_DIR / "portfolio_equity_curve.csv"

# ---------------------------------------------------------------------------
# JSON loaders
# ---------------------------------------------------------------------------


def _read_json(name: str, default: Any = None) -> Any:
    path = REPORT_DIR / name
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_portfolio_metrics() -> dict[str, Any]:
    return _read_json("portfolio_metrics.json", {})


def load_buy_hold_metrics() -> dict[str, Any]:
    return _read_json("buy_hold_metrics.json", {})


def load_subperiods() -> dict[str, Any]:
    return _read_json("robustness_subperiods.json", {})


def load_significance() -> dict[str, Any]:
    return _read_json("robustness_significance.json", {})


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------


def load_equity_curve() -> pd.DataFrame:
    """Return the portfolio day-by-day equity curve (1.0 == starting capital)."""
    if not EQUITY_FILE.exists():
        return pd.DataFrame(columns=["time", "position", "pnl", "cost", "equity", "exposure_on"])
    df = pd.read_csv(EQUITY_FILE)
    df["time"] = pd.to_datetime(df["time"])
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def buy_hold_curve(tickers: Iterable[str] | None = None) -> pd.DataFrame | None:
    """If per-ticker buy-hold parquet files exist under ``data/processed/``,
    return a wide DataFrame (one column per ticker) of buy-and-hold equity
    normalised to 1.0 at each ticker's first observation. Falls back to None
    if any input is missing.
    """
    data_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    if not data_dir.exists():
        return None
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        return None
    series: dict[str, pd.Series] = {}
    for f in files:
        try:
            d = pd.read_parquet(f, columns=["time", "close"])
        except Exception:
            continue
        d["time"] = pd.to_datetime(d["time"])
        d = d.sort_values("time")
        if d["close"].iloc[0] == 0 or pd.isna(d["close"].iloc[0]):
            continue
        eq = d["close"] / float(d["close"].iloc[0])
        series[f.stem] = pd.Series(eq.values, index=d["time"].values)

    if not series:
        return None
    wide = pd.DataFrame(series)
    wide.index.name = "time"
    wide = wide.sort_index()
    if tickers is not None:
        keep = [c for c in tickers if c in wide.columns]
        if keep:
            wide = wide[keep]
        else:
            return None
    wide = wide.reset_index()
    return wide


# ---------------------------------------------------------------------------
# Derived analytics used by the dashboard
# ---------------------------------------------------------------------------


def daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().fillna(0.0)


def rolling_sharpe(returns: pd.Series, window: int = 63, periods_per_year: int = 252) -> pd.Series:
    """63-trading-day rolling Sharpe (≈ 3 months), annualized."""
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std(ddof=0)
    return (mean / std.replace(0.0, np.nan)) * math.sqrt(periods_per_year)


def drawdown_curve(equity: pd.Series) -> pd.DataFrame:
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return pd.DataFrame({"equity": equity, "drawdown": dd})


def monthly_returns(equity: pd.Series) -> pd.DataFrame:
    """Pivot daily equity into a (year x month) matrix of monthly returns.

    Used by the dashboard's "monthly returns calendar" heatmap. Empty entries
    (no data for that month) are returned as NaN so the heatmap can render
    them grey.
    """
    if equity.empty:
        return pd.DataFrame()
    daily_rets = equity.pct_change().fillna(0.0)
    monthly = (1.0 + daily_rets).resample("ME").prod() - 1.0
    # ``monthly`` is a Series whose index is the end-of-month timestamps; we
    # don't rely on a particular index name because upstream callers might
    # strip it (``.rename`` etc.).
    df = monthly.rename("ret").reset_index()
    df.columns = ["time", "ret"]
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    pivot = df.pivot(index="year", columns="month", values="ret")
    pivot.columns = [int(c) for c in pivot.columns]
    pivot = pivot.reindex(columns=range(1, 13))
    return pivot


def underwater_intensity(equity: pd.Series) -> pd.DataFrame:
    """Build a long-form DataFrame ``(year, day_of_year, drawdown)`` for the
    underwater heatmap. Each row is one trading day, drawdown in [-1, 0].
    """
    if equity.empty:
        return pd.DataFrame(columns=["year", "doy", "drawdown"])
    dd = drawdown_curve(equity)["drawdown"]
    df = dd.reset_index()
    df.columns = ["time", "drawdown"]
    df["year"] = df["time"].dt.year
    df["doy"] = df["time"].dt.dayofyear
    return df[["year", "doy", "drawdown"]]


def fmt_pct(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x * 100:.{digits}f}%"


def fmt_num(x: float, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.{digits}f}"


def subperiods_frame(block: list[dict[str, Any]] | None) -> pd.DataFrame:
    if not block:
        return pd.DataFrame()
    df = pd.DataFrame(block)
    if "start" in df.columns:
        df["start"] = pd.to_datetime(df["start"])
    if "end" in df.columns:
        df["end"] = pd.to_datetime(df["end"])
    return df


def annual_return(equity: pd.Series) -> float:
    """CAGR-style annualization between the first and last equity point.

    Returns the total return when the span is shorter than ~30 days — CAGR is
    numerically unstable for short windows (e.g. two points one day apart
    would imply astronomic annualised growth).
    """
    if equity.empty or equity.iloc[0] <= 0:
        return 0.0
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    if isinstance(equity.index, pd.DatetimeIndex):
        days = (equity.index[-1] - equity.index[0]).days
    else:
        days = len(equity)
    if days < 30:
        # Treat as "I don't know how to annualise this"; report the raw
        # total return instead so callers still get a number, not NaN.
        return end / start - 1.0
    years = days / 365.25
    try:
        return (end / start) ** (1.0 / years) - 1.0
    except (ValueError, ZeroDivisionError, OverflowError):
        return 0.0
