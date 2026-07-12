"""
XGBoost classifier for triple-barrier labels, with optional Optuna tuning.

Uses gradient-boosted trees with multiclass loss (-1/0/1 -> 0/1/2).
Tuning: n_trials Optuna runs on a validation fold (NOT the test fold) to
avoid leak through hyperparameter selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from .baseline_lr import INV_LABEL_MAP, LABEL_MAP


# Suppress optuna's verbose logging unless asked
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class XGBModel:
    model: XGBClassifier
    feature_names: list[str] = field(default_factory=list)

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y_raw: np.ndarray,
        params: dict | None = None,
        feature_names: list[str] | None = None,
        early_stopping_rounds: int = 50,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "XGBModel":
        y = np.array([LABEL_MAP[int(v)] for v in y_raw])
        defaults = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "max_depth": 5,
            "learning_rate": 0.05,
            "n_estimators": 400,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "n_jobs": -1,
            "verbosity": 0,
            "random_state": 42,
        }
        if params:
            defaults.update(params)
        m = XGBClassifier(**defaults)
        if eval_set is not None:
            Xv, yv = eval_set
            m.fit(X, y, eval_set=[(Xv, np.array([LABEL_MAP[int(v)] for v in yv]))], verbose=False)
        else:
            m.fit(X, y)
        return cls(model=m, feature_names=feature_names or [])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict_sign(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return np.array([INV_LABEL_MAP[int(v)] for v in idx])


def objective_factory(X: np.ndarray, y_raw: np.ndarray, n_cv: int = 3):
    """Build an Optuna objective that uses TimeSeriesSplit inside the training
    data only. This protects against leak during hyperparameter selection.
    """
    y = np.array([LABEL_MAP[int(v)] for v in y_raw])

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "tree_method": "hist",
            "n_jobs": -1,
            "verbosity": 0,
            "random_state": 42,
        }
        m = XGBClassifier(**params)

        # Use time-series CV inside the training set only
        tscv = TimeSeriesSplit(n_splits=n_cv)
        scores = []
        for tr_idx, va_idx in tscv.split(X):
            m.fit(X[tr_idx], y[tr_idx], verbose=False)
            proba = m.predict_proba(X[va_idx])
            preds = np.argmax(proba, axis=1)
            # multi-class logloss
            p = np.clip(proba, 1e-9, 1.0)
            ll = -np.log(p[np.arange(len(va_idx)), y[va_idx]]).mean()
            scores.append(ll)
        return float(np.mean(scores))

    return objective


def tune_xgb(
    X: np.ndarray,
    y_raw: np.ndarray,
    n_trials: int = 30,
    timeout_s: int | None = None,
    n_cv: int = 3,
) -> dict:
    """Run Optuna search. Returns best params dict."""
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        objective_factory(X, y_raw, n_cv=n_cv),
        n_trials=n_trials,
        timeout=timeout_s,
        show_progress_bar=False,
    )
    return study.best_params


# ---- Binary {-1, +1} classifier (used for stronger trading signal) ----

@dataclass
class XGBBinaryModel:
    model: XGBClassifier
    feature_names: list[str] = field(default_factory=list)

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y_raw: np.ndarray,
        params: dict | None = None,
        feature_names: list[str] | None = None,
    ) -> "XGBBinaryModel":
        # map {-1: 0, +1: 1}
        y = (np.asarray(y_raw) == 1).astype(int)
        defaults = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 4,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "n_jobs": -1,
            "verbosity": 0,
            "random_state": 42,
        }
        if params:
            defaults.update(params)
        m = XGBClassifier(**defaults)
        m.fit(X, y)
        return cls(model=m, feature_names=feature_names or [])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


def _binary_objective_factory(X: np.ndarray, y_raw: np.ndarray, n_cv: int = 3):
    y = (np.asarray(y_raw) == 1).astype(int)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "tree_method": "hist",
            "n_jobs": -1,
            "verbosity": 0,
            "random_state": 42,
        }
        m = XGBClassifier(**params)
        tscv = TimeSeriesSplit(n_splits=n_cv)
        scores = []
        for tr_idx, va_idx in tscv.split(X):
            m.fit(X[tr_idx], y[tr_idx], verbose=False)
            p = m.predict_proba(X[va_idx])[:, 1]
            p = np.clip(p, 1e-9, 1 - 1e-9)
            scores.append(-np.log(p).mean() if False else float(-np.mean(y[va_idx] * np.log(p) + (1 - y[va_idx]) * np.log(1 - p))))
        return float(np.mean(scores))

    return objective


def tune_xgb_binary(
    X: np.ndarray,
    y_raw: np.ndarray,
    n_trials: int = 20,
    timeout_s: int | None = None,
    n_cv: int = 3,
) -> dict:
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        _binary_objective_factory(X, y_raw, n_cv=n_cv),
        n_trials=n_trials,
        timeout=timeout_s,
        show_progress_bar=False,
    )
    return study.best_params