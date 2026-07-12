"""
LSTM / GRU sequence model for triple-barrier labels.

We treat a rolling window of length `seq_len` as one input sample.
The model predicts P(label in {-1, 0, 1}) for the LAST day in the window.

Note: GRU tends to train faster and overfit less than LSTM on small tabular-ish
sequences. We default to GRU but allow LSTM via the `rnn_type` config flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .baseline_lr import INV_LABEL_MAP, LABEL_MAP


class SequenceClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        rnn_type: str = "gru",
        n_classes: int = 3,
    ):
        super().__init__()
        rnn_cls = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last)


def make_sequences(X: np.ndarray, y_raw: np.ndarray | None, seq_len: int):
    """Build rolling-window sequences.

    Returns:
        X_seq: (N, T, F)
        y_seq: (N,) of mapped labels, or None
    """
    n = len(X)
    if n <= seq_len:
        return np.empty((0, seq_len, X.shape[1])), np.empty((0,))
    Xs, ys = [], []
    for i in range(seq_len, n):
        Xs.append(X[i - seq_len : i])
        if y_raw is not None:
            ys.append(LABEL_MAP[int(y_raw[i])])
    Xs = np.asarray(Xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.int64) if y_raw is not None else None
    return Xs, ys


@dataclass
class LSTMModel:
    model: SequenceClassifier
    scaler_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    scaler_std: np.ndarray = field(default_factory=lambda: np.array([]))
    seq_len: int = 30
    device: str = "cpu"

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y_raw: np.ndarray,
        seq_len: int = 30,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        rnn_type: str = "gru",
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 64,
        patience: int = 5,
        device: str = "cpu",
        verbose: bool = False,
    ) -> "LSTMModel":
        # Standardise features
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        Xn = (X - mu) / sd

        Xs, ys = make_sequences(Xn, y_raw, seq_len=seq_len)
        if len(Xs) == 0:
            raise ValueError("Not enough rows for one sequence")

        # Train/val split inside training data: last 15% as val
        n = len(Xs)
        cut = int(n * 0.85)
        Xtr, ytr = Xs[:cut], ys[:cut]
        Xva, yva = Xs[cut:], ys[cut:]

        tr_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=batch_size, shuffle=True)
        va_loader = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)), batch_size=batch_size, shuffle=False)

        device_t = torch.device(device)
        model = SequenceClassifier(
            n_features=Xs.shape[-1], hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout, rnn_type=rnn_type, n_classes=3,
        ).to(device_t)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        loss_fn = nn.CrossEntropyLoss()

        best_val = float("inf")
        best_state = None
        bad = 0
        history = []
        for ep in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for xb, yb in tr_loader:
                xb = xb.to(device_t); yb = yb.to(device_t)
                opt.zero_grad()
                out = model(xb)
                l = loss_fn(out, yb)
                l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                train_loss += l.item() * xb.size(0)
            train_loss /= max(1, len(tr_loader.dataset))

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in va_loader:
                    xb = xb.to(device_t); yb = yb.to(device_t)
                    out = model(xb)
                    l = loss_fn(out, yb)
                    val_loss += l.item() * xb.size(0)
            val_loss /= max(1, len(va_loader.dataset))
            history.append((ep, train_loss, val_loss))
            if verbose:
                print(f"  epoch {ep:02d}  train={train_loss:.4f}  val={val_loss:.4f}")

            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    if verbose:
                        print(f"  early stop at epoch {ep}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        return cls(model=model, scaler_mean=mu, scaler_std=sd, seq_len=seq_len, device=device)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xn = (X - self.scaler_mean) / self.scaler_std
        Xs, _ = make_sequences(Xn, None, seq_len=self.seq_len)
        if len(Xs) == 0:
            return np.empty((0, 3))
        self.model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(Xs).to(self.device)
            out = self.model(xb)
            proba = torch.softmax(out, dim=1).cpu().numpy()
        # The first seq_len rows have no prediction -> pad with NaN probs (zeros for argmax fallback)
        padded = np.zeros((len(X), 3), dtype=np.float32)
        padded[self.seq_len:] = proba
        return padded

    def predict_sign(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return np.array([INV_LABEL_MAP[int(v)] for v in idx])