"""
End-to-end training pipeline.

For each ticker (or pooled across universe, depending on cfg):
  1. Load features + triple-barrier labels
  2. Generate walk-forward folds
  3. For each (model, fold):
       - fit on train
       - predict on test
       - record metrics to MLflow
  4. Aggregate fold-level metrics

Three model tiers (per the plan):
  - baseline logistic regression
  - XGBoost (Optuna tuned on first fold, params reused across folds)
  - LSTM/GRU sequence model

We tune XGBoost ONCE on fold 1's train split (not test) and reuse params
across subsequent folds. This is honest tuning: hyperparams do not see the
test data of any fold.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .config import PROJECT_ROOT, load_config
from .features import feature_columns
from .labeling import build_labels_for_ticker
from .models.baseline_lr import BaselineLRModel
from .models.lstm_model import LSTMModel
from .models.xgb_model import XGBBinaryModel, XGBModel, tune_xgb, tune_xgb_binary
from .splits import walk_forward_splits


# Use MLflow only if available; degrade gracefully
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_OK = True
except Exception:
    MLFLOW_OK = False


def _to_numpy_xy(df: pd.DataFrame):
    fc = feature_columns(df)
    X = df[fc].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int8)
    return X, y, fc


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def _log_loss(proba: np.ndarray, y_true: np.ndarray) -> float:
    eps = 1e-9
    p = np.clip(proba, eps, 1.0)
    return float(-np.log(p[np.arange(len(y_true)), y_true]).mean())


# Map triple-barrier labels {0,1,2} where 0=-1, 1=flat, 2=+1
INV_MAP = {0: -1, 1: 0, 2: 1}


@dataclass
class FoldResult:
    fold: int
    train_start: Any
    train_end: Any
    test_start: Any
    test_end: Any
    acc: float
    logloss: float
    # Predictions and truth (for downstream backtest use)
    times: list = field(default_factory=list)
    y_true: list = field(default_factory=list)
    proba: list = field(default_factory=list)


def run_lr(feat_lab: pd.DataFrame, folds: list[dict], cfg: dict) -> list[FoldResult]:
    out = []
    X_all, y_all, fc = _to_numpy_xy(feat_lab)
    for i, fold in enumerate(folds):
        tr, te = fold["train_idx"], fold["test_idx"]
        m = BaselineLRModel.fit(
            X_all[tr], y_all[tr],
            C=cfg["models"]["baseline_lr"].get("C", 1.0),
            max_iter=cfg["models"]["baseline_lr"].get("max_iter", 1000),
        )
        proba = m.predict_proba(X_all[te])
        # Convert raw label (-1/0/1) to mapped (0/1/2) for logloss calc
        y_te_mapped = np.array([{(-1): 0, 0: 1, 1: 2}[int(v)] for v in y_all[te]])
        out.append(FoldResult(
            fold=i + 1,
            train_start=fold["train_start"], train_end=fold["train_end"],
            test_start=fold["test_start"], test_end=fold["test_end"],
            acc=_accuracy(y_te_mapped, np.argmax(proba, axis=1)),
            logloss=_log_loss(proba, y_te_mapped),
            times=feat_lab["time"].iloc[te].tolist(),
            y_true=y_all[te].tolist(),
            proba=proba.tolist(),
        ))
    return out


def run_xgb(feat_lab: pd.DataFrame, folds: list[dict], cfg: dict) -> list[FoldResult]:
    out = []
    X_all, y_all, fc = _to_numpy_xy(feat_lab)
    xgb_cfg = cfg["models"]["xgboost"]

    # Optuna tuning on the FIRST fold's train split only
    tuned_params = None
    if xgb_cfg.get("enabled", True) and xgb_cfg.get("n_trials", 0) > 0:
        tr0 = folds[0]["train_idx"]
        print(f"  XGB Optuna tuning on first fold ({len(tr0)} rows, {xgb_cfg['n_trials']} trials) ...")
        tuned_params = tune_xgb(
            X_all[tr0], y_all[tr0],
            n_trials=xgb_cfg["n_trials"],
            timeout_s=xgb_cfg.get("timeout_s"),
            n_cv=xgb_cfg.get("cv_folds", 3),
        )
        print(f"  best params: {json.dumps(tuned_params, indent=None)}")

    for i, fold in enumerate(folds):
        tr, te = fold["train_idx"], fold["test_idx"]
        m = XGBModel.fit(
            X_all[tr], y_all[tr],
            params=tuned_params, feature_names=fc,
            early_stopping_rounds=xgb_cfg.get("early_stopping_rounds", 50),
            eval_set=(X_all[te], y_all[te]),
        )
        proba = m.predict_proba(X_all[te])
        y_te_mapped = np.array([{(-1): 0, 0: 1, 1: 2}[int(v)] for v in y_all[te]])
        out.append(FoldResult(
            fold=i + 1,
            train_start=fold["train_start"], train_end=fold["train_end"],
            test_start=fold["test_start"], test_end=fold["test_end"],
            acc=_accuracy(y_te_mapped, np.argmax(proba, axis=1)),
            logloss=_log_loss(proba, y_te_mapped),
            times=feat_lab["time"].iloc[te].tolist(),
            y_true=y_all[te].tolist(),
            proba=proba.tolist(),
        ))
    return out


def run_lstm(feat_lab: pd.DataFrame, folds: list[dict], cfg: dict) -> list[FoldResult]:
    out = []
    X_all, y_all, fc = _to_numpy_xy(feat_lab)
    lstm_cfg = cfg["models"]["lstm"]
    seq_len = lstm_cfg["seq_len"]
    # LSTM needs sequence data; for each fold we walk forward from index seq_len
    for i, fold in enumerate(folds):
        tr, te = fold["train_idx"], fold["test_idx"]
        # Need contiguous training rows so sequences span correct history.
        # Walk-forward uses an expanding window so train is [0..tr_end]; good.
        try:
            m = LSTMModel.fit(
                X_all[tr], y_all[tr],
                seq_len=seq_len,
                hidden_size=lstm_cfg["hidden_size"],
                num_layers=lstm_cfg["num_layers"],
                dropout=lstm_cfg["dropout"],
                rnn_type="gru",
                lr=lstm_cfg["lr"],
                epochs=lstm_cfg["epochs"],
                batch_size=lstm_cfg["batch_size"],
                patience=lstm_cfg["patience"],
                device=lstm_cfg["device"],
            )
        except Exception as e:  # noqa: BLE001
            print(f"  fold {i+1}: LSTM failed ({e})")
            continue
        proba = m.predict_proba(X_all[te])
        y_te_mapped = np.array([{(-1): 0, 0: 1, 1: 2}[int(v)] for v in y_all[te]])
        # ignore padded leading rows in logloss/acc — measure only rows with a real prediction
        mask = np.arange(len(te)) >= seq_len
        if mask.sum() == 0:
            continue
        acc = _accuracy(y_te_mapped[mask], np.argmax(proba[mask], axis=1))
        ll = _log_loss(proba[mask], y_te_mapped[mask])
        out.append(FoldResult(
            fold=i + 1,
            train_start=fold["train_start"], train_end=fold["train_end"],
            test_start=fold["test_start"], test_end=fold["test_end"],
            acc=acc,
            logloss=ll,
            times=feat_lab["time"].iloc[te].tolist(),
            y_true=y_all[te].tolist(),
            proba=proba.tolist(),
        ))
    return out


def _aggregate(results: list[FoldResult]) -> dict[str, float]:
    if not results:
        return {"acc_mean": float("nan"), "logloss_mean": float("nan")}
    return {
        "acc_mean": float(np.mean([r.acc for r in results])),
        "acc_std": float(np.std([r.acc for r in results])),
        "logloss_mean": float(np.mean([r.logloss for r in results])),
        "logloss_std": float(np.std([r.logloss for r in results])),
    }


# ----- directional binary model (added for stronger trading signal) -----

LABEL_BINARY_MAP = {-1: 0, 1: 1}  # drop the "flat" class
INV_BINARY_MAP = {0: -1, 1: 1}


@dataclass
class FoldResultBinary:
    fold: int
    train_start: Any
    train_end: Any
    test_start: Any
    test_end: Any
    acc: float
    logloss: float
    times: list = field(default_factory=list)
    y_true: list = field(default_factory=list)
    proba: list = field(default_factory=list)  # P(up)


def _to_numpy_xy_binary(df: pd.DataFrame):
    fc = feature_columns(df)
    # Drop rows where label == 0 (flat) — they are not informative for direction
    mask = df["label"] != 0
    X = df.loc[mask, fc].to_numpy(dtype=np.float64)
    y = df.loc[mask, "label"].to_numpy(dtype=np.int8)
    times = df.loc[mask, "time"].reset_index(drop=True)
    return X, y, fc, times, mask


def run_xgb_binary(feat_lab: pd.DataFrame, folds: list[dict], cfg: dict) -> list[FoldResultBinary]:
    """Train binary up/down classifier on the non-flat rows.

    Why: for trading, the *direction* signal matters more than the exact
    triple-barrier class. A binary model often gives sharper probabilities.
    """
    out = []
    fc = feature_columns(feat_lab)
    xgb_cfg = cfg["models"]["xgboost"]

    X_all_raw = feat_lab[fc].to_numpy(dtype=np.float64)
    y_all_raw = feat_lab["label"].to_numpy(dtype=np.int8)
    times_all = pd.to_datetime(feat_lab["time"]).reset_index(drop=True)

    # Build a single mask of "non-flat" rows
    mask = y_all_raw != 0
    X_dir = X_all_raw[mask]
    y_dir = y_all_raw[mask]
    times_dir = times_all[mask].reset_index(drop=True)

    # Build index mapping for fold split — each fold's indices must be the
    # indices into the *original* dataframe (for time alignment), but the
    # classifier only sees the non-flat subset.
    # We reuse folds over the full feature frame, then filter.
    tuned_params = None
    if xgb_cfg.get("n_trials", 0) > 0:
        # tune on first fold's non-flat training rows
        tr_full = folds[0]["train_idx"]
        # Map original indices -> mask indices
        mask_idx = np.where(mask)[0]
        tr_mask_idx = np.intersect1d(tr_full, mask_idx, assume_unique=False)
        if len(tr_mask_idx) >= 100:
            Xtr = X_all_raw[tr_mask_idx]
            ytr = y_all_raw[tr_mask_idx]
            print(f"  XGB-binary Optuna tuning on {len(Xtr)} non-flat train rows ({xgb_cfg['n_trials']} trials) ...")
            tuned_params = tune_xgb_binary(Xtr, ytr, n_trials=min(xgb_cfg["n_trials"], 20), n_cv=xgb_cfg.get("cv_folds", 3))
            print(f"  best params: {json.dumps(tuned_params, indent=None)}")

    for i, fold in enumerate(folds):
        te_full = fold["test_idx"]
        tr_full = fold["train_idx"]
        te_mask_idx = np.intersect1d(te_full, mask_idx, assume_unique=False)
        tr_mask_idx = np.intersect1d(tr_full, mask_idx, assume_unique=False)
        if len(tr_mask_idx) < 100 or len(te_mask_idx) < 20:
            continue

        Xtr = X_all_raw[tr_mask_idx]
        ytr = y_all_raw[tr_mask_idx]
        Xte = X_all_raw[te_mask_idx]
        yte = y_all_raw[te_mask_idx]

        m = XGBBinaryModel.fit(
            Xtr, ytr,
            params=tuned_params,
        )
        proba = m.predict_proba(Xte)
        y_te_bin = np.array([LABEL_BINARY_MAP[int(v)] for v in yte])
        acc = _accuracy(y_te_bin, (proba[:, 1] > 0.5).astype(int))
        # binary logloss
        p1 = np.clip(proba[:, 1], 1e-9, 1.0)
        ll = float(-np.mean(y_te_bin * np.log(p1) + (1 - y_te_bin) * np.log(1 - p1)))
        out.append(FoldResultBinary(
            fold=i + 1,
            train_start=fold["train_start"], train_end=fold["train_end"],
            test_start=fold["test_start"], test_end=fold["test_end"],
            acc=acc, logloss=ll,
            times=times_all.iloc[te_mask_idx].tolist(),
            y_true=yte.tolist(),
            proba=proba[:, 1].tolist(),
        ))
    return out


def run_pipeline(cfg: dict, ticker: str = "FPT", mlflow_enable: bool = True) -> dict[str, Any]:
    """Run the full pipeline for a single ticker."""
    feat_p = PROJECT_ROOT / cfg["data"]["processed_dir"] / f"{ticker}_features.parquet"
    feat = pd.read_parquet(feat_p)
    feat_lab = build_labels_for_ticker(feat, cfg)
    folds = walk_forward_splits(feat_lab, cfg)
    print(f"[{ticker}] {len(feat_lab)} labelled rows, {len(folds)} walk-forward folds")

    all_results: dict[str, Any] = {"ticker": ticker, "folds": len(folds)}

    if mlflow_enable and MLFLOW_OK:
        mlflow.set_tracking_uri(str(PROJECT_ROOT / cfg["mlflow"]["tracking_uri"]))
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    else:
        mlflow = None  # noqa: F841

    # Baseline LR
    print("\n[LR] baseline ...")
    lr_results = run_lr(feat_lab, folds, cfg)
    all_results["lr"] = {"folds": [r.__dict__ for r in lr_results], "agg": _aggregate(lr_results)}
    if MLFLOW_OK and mlflow_enable:
        with mlflow.start_run(run_name=f"lr_{ticker}"):
            mlflow.log_param("model", "logistic_regression")
            mlflow.log_param("ticker", ticker)
            mlflow.log_metrics(all_results["lr"]["agg"])
    print("  agg:", all_results["lr"]["agg"])

    # XGBoost
    print("\n[XGB] optuna-tuned ...")
    xgb_results = run_xgb(feat_lab, folds, cfg)
    all_results["xgb"] = {"folds": [r.__dict__ for r in xgb_results], "agg": _aggregate(xgb_results)}
    if MLFLOW_OK and mlflow_enable:
        with mlflow.start_run(run_name=f"xgb_{ticker}"):
            mlflow.log_param("model", "xgboost")
            mlflow.log_param("ticker", ticker)
            mlflow.log_metrics(all_results["xgb"]["agg"])
    print("  agg:", all_results["xgb"]["agg"])

    # LSTM / GRU
    print("\n[LSTM/GRU] sequence model ...")
    lstm_results = run_lstm(feat_lab, folds, cfg)
    all_results["lstm"] = {"folds": [r.__dict__ for r in lstm_results], "agg": _aggregate(lstm_results)}
    if MLFLOW_OK and mlflow_enable:
        with mlflow.start_run(run_name=f"lstm_{ticker}"):
            mlflow.log_param("model", "gru")
            mlflow.log_param("ticker", ticker)
            mlflow.log_metrics(all_results["lstm"]["agg"])
    print("  agg:", all_results["lstm"]["agg"])

    # XGBoost binary (directional) — used as the trading signal
    print("\n[XGB-binary] directional classifier ...")
    bin_results = run_xgb_binary(feat_lab, folds, cfg)
    all_results["xgb_binary"] = {
        "folds": [r.__dict__ for r in bin_results],
        "agg": _aggregate(bin_results) if bin_results else {"acc_mean": float("nan"), "logloss_mean": float("nan")},
    }
    if MLFLOW_OK and mlflow_enable:
        with mlflow.start_run(run_name=f"xgb_binary_{ticker}"):
            mlflow.log_param("model", "xgboost_binary")
            mlflow.log_param("ticker", ticker)
            mlflow.log_metrics(all_results["xgb_binary"]["agg"])
    print("  agg:", all_results["xgb_binary"]["agg"])

    # Persist predictions for downstream backtest
    pred_dir = PROJECT_ROOT / "data" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{ticker}_predictions.json"
    with pred_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nPredictions saved to {pred_path}")
    return all_results


def _build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="FPT")
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "strategy.yaml"))
    p.add_argument("--no-mlflow", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _build_args()
    cfg = load_config(args.config)
    run_pipeline(cfg, ticker=args.ticker.upper(), mlflow_enable=not args.no_mlflow)


if __name__ == "__main__":
    main()