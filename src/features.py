"""
Feature engineering for momentum + volatility regime models.

Pitfall #1 from the plan — LOOKAHEAD BIAS:
    Every rolling feature computed at time t must use information available
    BEFORE the close of t. We achieve this by:
      (a) Computing all rolling indicators on past data only (rolling().mean() etc.
          uses window [t-w, t-1] by definition)
      (b) Then shift(1) the entire feature frame so that feature row at time t
          only sees info up to t-1.

    Without shift(1), RSI/ATR at row t would incorporate today's close -> leak.

We expose a single entrypoint `build_features(ohlcv_df, cfg)` that:
    - Adds RSI, MACD, Bollinger width, ATR
    - Adds rolling volatility (5/20/60d), volume z-score
    - Adds lag returns (1/2/5d)
    - Adds realized volatility regime label (high/low) as a separate column,
      NOT as a feature (regime is what we want to PREDICT / use to scale).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange, BollingerBands

from .config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---- technical indicators ---------------------------------------------------

def _add_technical_features(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()

    f = cfg["features"]

    # RSI
    rsi = RSIIndicator(close=out["close"], window=f["rsi_len"], fillna=False)
    out[f"rsi_{f['rsi_len']}"] = rsi.rsi()

    # MACD
    macd = MACD(
        close=out["close"],
        window_slow=f["macd_slow"],
        window_fast=f["macd_fast"],
        window_sign=f["macd_signal"],
    )
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_diff"] = macd.macd_diff()

    # Bollinger Bands -> use the WIDTH as a feature (normalised by middle band)
    bb = BollingerBands(close=out["close"], window=f["bb_len"], window_dev=f["bb_std"])
    out["bb_high"] = bb.bollinger_hband()
    out["bb_low"] = bb.bollinger_lband()
    out["bb_mavg"] = bb.bollinger_mavg()
    out["bb_width"] = (out["bb_high"] - out["bb_low"]) / out["bb_mavg"]

    # ATR (volatility)
    atr = AverageTrueRange(
        high=out["high"], low=out["low"], close=out["close"], window=f["atr_len"], fillna=False
    )
    out[f"atr_{f['atr_len']}"] = atr.average_true_range()
    # ATR as % of close — scale-invariant
    out["atr_pct"] = out[f"atr_{f['atr_len']}"] / out["close"]

    return out


def _add_return_features(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    f = cfg["features"]

    # Daily log return (NOT shifted — this IS today's move, the label target uses it).
    # As a feature we use past returns only via lag.
    out["ret_1d"] = np.log(out["close"] / out["close"].shift(1))

    # Lag returns — lag 1 means "return from t-2 to t-1" -> safe at row t
    for lag in f["return_lags"]:
        out[f"ret_lag_{lag}"] = out["ret_1d"].shift(lag)

    # Rolling realised volatility (annualised). Use raw squared log returns.
    vol_windows = f["vol_windows"]
    for w in vol_windows:
        rolling = out["ret_1d"].rolling(w).std()
        out[f"vol_{w}d"] = rolling * np.sqrt(252)

    # Volume z-score
    zwin = f["vol_zscore_window"]
    vol_mean = out["volume"].rolling(zwin).mean()
    vol_std = out["volume"].rolling(zwin).std()
    out["vol_zscore"] = (out["volume"] - vol_mean) / vol_std.replace(0, np.nan)

    return out


def _shift_features(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """shift(1) every FEATURE column to avoid lookahead.

    Returns are NOT shifted here — `ret_1d` is the realised move from t-1 to t,
    which is what we want to predict (the label uses future returns, defined
    in labeling.py).
    """
    if not cfg["features"].get("shift_features", True):
        return df

    feature_cols = [c for c in df.columns if c not in {"time", "open", "high", "low", "close", "volume", "ret_1d"}]
    df = df.copy()
    for c in feature_cols:
        df[c] = df[c].shift(1)
    return df


# ---- regime labelling -------------------------------------------------------

def add_regime_label(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Label each row as low / high volatility regime based on rolling 60d vol.

    Method: rolling 60d vol (annualised) -> percentile threshold.
    A row is 'high_vol' if its 60d vol is in the top tercile of the
    *training* distribution. For simplicity here we use the full-sample
    percentile; in production you would refit this per-fold.
    """
    out = df.copy()
    r = cfg["regime"]
    lookback = r["lookback"]
    thr = r["high_vol_percentile"]

    if "vol_60d" not in out.columns:
        out["vol_60d"] = out["ret_1d"].rolling(lookback).std() * np.sqrt(252)

    cutoff = out["vol_60d"].quantile(thr)
    out["regime_high_vol"] = (out["vol_60d"] >= cutoff).astype(int)

    # shift the regime label too — at time t we only know vol up to t-1
    out["regime_high_vol"] = out["regime_high_vol"].shift(1)

    return out


# ---- main entrypoint --------------------------------------------------------

def build_features(ohlcv: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Build the full feature frame for a single ticker.

    Parameters
    ----------
    ohlcv : DataFrame
        Must have columns time, open, high, low, close, volume (sorted asc).
    cfg : dict, optional
        Strategy config. If None, loaded from disk.

    Returns
    -------
    DataFrame sorted by time, with all indicator features + ret_1d + regime label.
    """
    if cfg is None:
        cfg = load_config()

    df = ohlcv.copy()
    df["time"] = pd.to_datetime(df["time"])

    df = _add_technical_features(df, cfg)
    df = _add_return_features(df, cfg)
    df = _shift_features(df, cfg)
    df = add_regime_label(df, cfg)

    # Drop warmup NaNs (anything before enough history is available)
    required_cols = [c for c in df.columns if c not in {"regime_high_vol", "ret_1d"}]
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of columns usable as model features (X)."""
    excluded = {
        "time", "open", "high", "low", "close", "volume", "ret_1d",
        "regime_high_vol",
        # label-related columns MUST never be features
        "label", "label_ret", "label_hdays",
    }
    return [c for c in df.columns if c not in excluded]


def build_universe_features(
    universe_dfs: dict[str, pd.DataFrame],
    cfg: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build features for each ticker in the universe."""
    if cfg is None:
        cfg = load_config()
    return {tkr: build_features(df, cfg) for tkr, df in universe_dfs.items()}


def process_universe_to_parquet(cfg: dict[str, Any] | None = None) -> dict[str, Path]:
    """Load raw OHLCV from data/raw, build features, persist to data/processed."""
    from .ingestion import RAW_DIR, load_universe

    if cfg is None:
        cfg = load_config()

    tickers = cfg["universe"]["tickers"]
    raw = load_universe(tickers)

    out_dir = PROJECT_ROOT / cfg["data"]["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for tkr, df in raw.items():
        feat = build_features(df, cfg)
        p = out_dir / f"{tkr}_features.parquet"
        feat.to_parquet(p, index=False)
        paths[tkr] = p
        print(f"  {tkr}: {len(feat):>5d} feature rows -> {p.name}")
    return paths


if __name__ == "__main__":
    process_universe_to_parquet()