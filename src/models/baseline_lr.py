"""
Baseline logistic regression classifier for triple-barrier labels.

Remap {-1, 0, 1} -> {0, 1, 2} for sklearn, then inverse on prediction.
Use class_weight='balanced' so the rare up-move class isn't drowned out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# Mapping between triple-barrier labels and sklearn-friendly integers
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


@dataclass
class BaselineLRModel:
    model: LogisticRegression
    scaler: StandardScaler

    @classmethod
    def fit(cls, X: np.ndarray, y_raw: np.ndarray, C: float = 1.0, max_iter: int = 1000) -> "BaselineLRModel":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        y = np.array([LABEL_MAP[int(v)] for v in y_raw])
        m = LogisticRegression(
            C=C, max_iter=max_iter, class_weight="balanced", solver="lbfgs", multi_class="auto"
        )
        m.fit(Xs, y)
        return cls(model=m, scaler=scaler)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.model.predict_proba(Xs)

    def predict_sign(self, X: np.ndarray) -> np.ndarray:
        """Return {-1, 0, 1} by taking argmax of predict_proba and remapping.

        For trading we usually only use P(class=up) - P(class=down) as a continuous
        signal, but this method gives a discrete fallback for reporting.
        """
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return np.array([INV_LABEL_MAP[int(v)] for v in idx])