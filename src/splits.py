"""
Walk-forward (expanding-window) split generator.

CRITICAL pitfall from the plan: random train_test_split on time series is
data leakage. We use expanding-window splits:

    Fold 1: train = [start, t1],           test = [t1, t1 + test_window]
    Fold 2: train = [start, t2],           test = [t2, t2 + test_window]
    ...
    Fold k: train = [start, tk],           test = [tk, tk + test_window]

Each fold is strictly separated by an embargo_days gap to avoid information
leak between train end and test start.

Returns a list of (train_idx, test_idx) integer index arrays relative to
the input dataframe.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

from .config import load_config


def walk_forward_splits(
    df: pd.DataFrame,
    cfg: dict | None = None,
) -> list[dict]:
    """Generate walk-forward folds.

    Parameters
    ----------
    df : DataFrame
        Must be sorted ascending by time and have a `time` column.
    cfg : dict
        Strategy config.

    Returns
    -------
    List of dicts, one per fold:
        { "train_idx": np.array, "test_idx": np.array,
          "train_start": date, "train_end": date, "test_start": date, "test_end": date }
    """
    if cfg is None:
        cfg = load_config()
    sp = cfg["split"]

    df = df.sort_values("time").reset_index(drop=True)
    n = len(df)
    times = pd.to_datetime(df["time"])
    start = times.iloc[0]
    end = times.iloc[-1]

    train_days = int(sp["train_years"] * 252)
    test_days = int(sp["test_months"] * 21)  # ~21 trading days / month
    embargo = int(sp["embargo_days"])
    n_splits = int(sp["n_splits"])
    min_train = int(sp["min_train_size"])

    if train_days + test_days > n:
        raise ValueError("Not enough data for one full walk-forward fold")

    # Choose fold start dates evenly across the available range AFTER the initial
    # train window. The last fold's test must end at or before `end`.
    first_test_start_idx = max(train_days, min_train)
    last_test_start_idx = n - test_days - 1
    if last_test_start_idx <= first_test_start_idx:
        raise ValueError("Cannot build any walk-forward folds with current params")

    test_starts = np.linspace(first_test_start_idx, last_test_start_idx, n_splits).astype(int)

    folds = []
    for ts in test_starts:
        # Train = [0, ts - embargo - 1]
        train_end = ts - embargo - 1
        train_idx = np.arange(0, train_end + 1)
        test_idx = np.arange(ts, min(ts + test_days, n))

        if len(train_idx) < min_train or len(test_idx) < 50:
            continue

        folds.append({
            "train_idx": train_idx,
            "test_idx": test_idx,
            "train_start": times.iloc[train_idx[0]],
            "train_end": times.iloc[train_idx[-1]],
            "test_start": times.iloc[test_idx[0]],
            "test_end": times.iloc[test_idx[-1]],
        })
    return folds


def iter_walk_forward(
    df: pd.DataFrame,
    cfg: dict | None = None,
) -> Iterator[dict]:
    for fold in walk_forward_splits(df, cfg):
        yield fold


if __name__ == "__main__":
    cfg = load_config()
    from pathlib import Path
    from .features import build_features
    from .ingestion import load_one
    from .labeling import build_labels_for_ticker

    root = Path(__file__).resolve().parents[1]
    raw = load_one("FPT")
    feat = build_features(raw, cfg)
    labelled = build_labels_for_ticker(feat, cfg)

    folds = walk_forward_splits(labelled, cfg)
    for i, f in enumerate(folds):
        print(f"Fold {i+1}:")
        print(f"  train: {f['train_start'].date()} -> {f['train_end'].date()}  ({len(f['train_idx'])} rows)")
        print(f"  test : {f['test_start'].date()} -> {f['test_end'].date()}  ({len(f['test_idx'])} rows)")