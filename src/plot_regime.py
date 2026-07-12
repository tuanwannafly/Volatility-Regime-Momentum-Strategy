"""Plot sanity check: close price + high-vol regime shading.

Outputs reports/figures/regime_check.png for one ticker.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def plot_regime(ticker: str, cfg: dict | None = None) -> Path:
    if cfg is None:
        cfg = load_config()

    p = PROJECT_ROOT / cfg["data"]["processed_dir"] / f"{ticker}_features.parquet"
    df = pd.read_parquet(p)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(df["time"], df["close"], color="black", linewidth=1.2)
    ax1.set_title(f"{ticker} close price with high-vol regime (shaded)")
    ax1.set_ylabel("Close (VND)")
    ax1.grid(True, alpha=0.3)

    # Shade high-vol periods
    in_high = False
    start = None
    for t, flag in zip(df["time"], df["regime_high_vol"]):
        if flag == 1 and not in_high:
            start = t
            in_high = True
        elif flag == 0 and in_high:
            ax1.axvspan(start, t, alpha=0.2, color="red")
            in_high = False
    if in_high:
        ax1.axvspan(start, df["time"].iloc[-1], alpha=0.2, color="red")

    ax2.plot(df["time"], df["vol_60d"], color="steelblue", linewidth=1.0, label="60d realised vol (ann.)")
    ax2.set_ylabel("Annualised vol")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")

    out_dir = PROJECT_ROOT / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"regime_check_{ticker}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for tkr in ["FPT", "VHM", "HPG"]:
        p = plot_regime(tkr)
        print(f"  wrote {p}")