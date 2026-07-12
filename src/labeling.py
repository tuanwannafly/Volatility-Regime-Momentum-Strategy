"""
Triple-barrier labeling (Marcos López de Prado, "Advances in Financial ML").

For each entry date t we look forward up to time_barrier_days and label
the row with whichever barrier is hit FIRST:
    - upper barrier hit -> label = 1 (long winner)
    - lower barrier hit -> label = -1 (long loser)
    - time barrier hit  -> label = 0 (flat / undecided)
    - "neither" because move too small -> label = 0

Barriers are DYNAMIC: entry * (1 +/- k * ATR%) — k configurable.
Using ATR-scaled barriers means the threshold adapts to recent volatility,
so the model learns regime-relative moves rather than absolute %.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_config


def triple_barrier_labels(
    df: pd.DataFrame,
    upper_atr_mult: float = 2.0,
    lower_atr_mult: float = 2.0,
    time_barrier_days: int = 10,
    min_return: float = 0.0,
    atr_col: str = "atr_14",
    close_col: str = "close",
) -> pd.DataFrame:
    """Compute triple-barrier labels.

    Parameters
    ----------
    df : DataFrame
        Must contain close, atr columns, sorted ascending by time.
    upper_atr_mult : float
        Upper barrier = entry * (1 + mult * atr_pct).
    lower_atr_mult : float
        Lower barrier = entry * (1 - mult * atr_pct).
    time_barrier_days : int
        Max horizon. If neither price barrier hit, label = 0 (time out).
    min_return : float
        Minimum |return| at the hit day for it to count as 1/-1;
        below this we call it flat. Suppresses label noise.
    atr_col : str
        Column with the ATR (raw, not %).
    close_col : str
        Column with close price.

    Returns
    -------
    Input df with new columns: `label` (int in {-1, 0, 1}), `label_ret` (float,
    return realised on the day the barrier was hit).
    """
    out = df.copy().reset_index(drop=True)
    n = len(out)
    closes = out[close_col].to_numpy(dtype=float)
    atrs = out[atr_col].to_numpy(dtype=float)

    labels = np.zeros(n, dtype=np.int8)
    label_rets = np.zeros(n, dtype=float)
    label_hdays = np.zeros(n, dtype=np.int16)  # holding days
    invalid = np.zeros(n, dtype=bool)  # not enough future data

    for i in range(n):
        entry = closes[i]
        atr = atrs[i]
        if not np.isfinite(entry) or not np.isfinite(atr) or atr <= 0 or entry <= 0:
            invalid[i] = True
            continue

        upper = entry * (1.0 + upper_atr_mult * atr / entry)
        lower = entry * (1.0 - lower_atr_mult * atr / entry)
        # Make sure lower > 0
        lower = max(lower, entry * 0.5)

        # Walk forward
        last_idx = min(i + time_barrier_days, n - 1)
        hit = 0
        hit_ret = 0.0
        hit_day = 0
        for j in range(i + 1, last_idx + 1):
            if closes[j] >= upper:
                hit = 1
                hit_ret = (closes[j] - entry) / entry
                hit_day = j - i
                break
            if closes[j] <= lower:
                hit = -1
                hit_ret = (closes[j] - entry) / entry
                hit_day = j - i
                break

        if hit == 0:
            # time barrier — label with realised return at last day
            hit_ret = (closes[last_idx] - entry) / entry
            hit_day = last_idx - i

        # Apply min_return de-noise
        if abs(hit_ret) < min_return:
            labels[i] = 0
        else:
            labels[i] = hit
        label_rets[i] = hit_ret
        label_hdays[i] = hit_day

        if i + time_barrier_days > n - 1:
            invalid[i] = True

    out["label"] = labels
    out["label_ret"] = label_rets
    out["label_hdays"] = label_hdays
    # Drop rows where we don't have enough future to label
    out = out[~invalid].reset_index(drop=True)
    return out


def build_labels_for_ticker(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    if cfg is None:
        cfg = load_config()
    p = cfg["labeling"]
    return triple_barrier_labels(
        df,
        upper_atr_mult=p["upper_atr_mult"],
        lower_atr_mult=p["lower_atr_mult"],
        time_barrier_days=p["time_barrier_days"],
        min_return=p["min_return"],
    )


if __name__ == "__main__":
    cfg = load_config()
    p = cfg["data"]["processed_dir"]
    from pathlib import Path
    from .features import build_features
    from .ingestion import load_one
    root = Path(__file__).resolve().parents[1]
    raw = load_one("FPT")
    feat = build_features(raw, cfg)
    labelled = build_labels_for_ticker(feat, cfg)
    print("Label distribution:")
    print(labelled["label"].value_counts(normalize=True).round(3))
    print("Avg holding days:", labelled["label_hdays"].mean())
    print("Avg realised return by label:")
    print(labelled.groupby("label")["label_ret"].agg(["mean", "std", "count"]).round(4))