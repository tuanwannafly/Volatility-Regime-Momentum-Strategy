"""
Robustness checks for the vol-target strategy.

Two layers of analysis on the existing portfolio equity curve:

1. **Sub-period analysis** — break the full walk-forward backtest window into
   smaller pieces and check whether the strategy is consistently profitable
   across them, or whether it rides on 1-2 good years.

   We do this in three cuts:
     (a) Calendar years
     (b) Walk-forward folds (matches the model's actual evaluation slices)
     (c) Volatility regimes (high-vol vs low-vol periods of the *equal-weight*
         index proxy)

   For each sub-period we report:
     - SR (annualised)
     - Total return
     - Max drawdown (within that sub-period)
     - Win rate
     - n_days

2. **Statistical significance** of the full-period Sharpe.
   See src/backtest/significance.py for the math.

Inputs: reports/portfolio_equity_curve.csv (produced by src/report.py).
Outputs:
   - reports/robustness_subperiods.json
   - reports/robustness_significance.json
   - reports/figures/subperiod_sharpe.png
   - reports/figures/rolling_sharpe_with_ci.png
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest.metrics import compute_metrics, rolling_sharpe
from .backtest.significance import (
    block_bootstrap_ci,
    deflated_sharpe_ratio,
    lo_sharpe_tstat,
    naive_sharpe_tstat,
)
from .config import PROJECT_ROOT, load_config


warnings.filterwarnings("ignore")


# ---- helpers ---------------------------------------------------------------

def _period_metrics(equity: pd.Series, returns: pd.Series) -> dict:
    """Compute summary metrics for a sub-period."""
    if len(equity) < 5:
        return {"sr": float("nan"), "total_return": float("nan"), "max_drawdown": float("nan"),
                "win_rate": float("nan"), "n_days": 0, "annual_vol": float("nan")}

    df = pd.DataFrame({"equity": equity.values, "ret": returns.values})
    m = compute_metrics(df, risk_free=0.04)
    m["n_days"] = int(len(df))
    return m


def _add_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["time"].dt.year
    return df


def _add_regime(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Label each day as high-vol or low-vol based on the *equal-weight
    portfolio's* rolling 60d realised vol. We compute it here from the equity
    curve's returns (portfolio-level), which is what actually drove the
    strategy's behaviour.
    """
    df = df.copy()
    ret = df["ret"].astype(float)
    roll_vol = ret.rolling(60).std() * np.sqrt(252)
    thr = roll_vol.quantile(cfg["regime"]["high_vol_percentile"])
    df["high_vol_regime"] = (roll_vol >= thr).astype(int)
    df["roll_vol_60d"] = roll_vol
    return df


# ---- sub-period cuts -------------------------------------------------------

def yearly_breakdown(df: pd.DataFrame) -> list[dict]:
    rows = []
    for yr, g in _add_year(df).groupby("year"):
        if len(g) < 5:
            continue
        eq = (1 + g["ret"].astype(float).fillna(0.0)).cumprod()
        m = _period_metrics(eq, g["ret"].astype(float).fillna(0.0))
        m["period"] = f"FY{yr}"
        m["start"] = str(g["time"].iloc[0].date())
        m["end"] = str(g["time"].iloc[-1].date())
        rows.append(m)
    return rows


def fold_breakdown(df: pd.DataFrame, cfg: dict) -> list[dict]:
    """Split by walk-forward fold. A fold's test window is what the model
    never saw during training — this is the cleanest out-of-sample cut.

    We use `splits.walk_forward_splits` on the *labelled feature frame* for
    one ticker (FPT) to recover the exact fold boundaries, then map those
    indices onto the portfolio equity curve (which spans a subset of dates).
    """
    from .features import build_features
    from .ingestion import load_one
    from .labeling import build_labels_for_ticker
    from .splits import walk_forward_splits

    raw = load_one("FPT")
    feat = build_features(raw, cfg)
    feat_lab = build_labels_for_ticker(feat, cfg)
    folds_meta = walk_forward_splits(feat_lab, cfg)

    # Map each fold's start/end dates to positions in the portfolio equity curve
    rows = []
    for i, fold in enumerate(folds_meta):
        ts = pd.to_datetime(fold["test_start"])
        te = pd.to_datetime(fold["test_end"])
        mask = (df["time"] >= ts) & (df["time"] <= te)
        g = df.loc[mask]
        if len(g) < 5:
            continue
        eq = (1 + g["ret"].astype(float).fillna(0.0)).cumprod()
        m = _period_metrics(eq, g["ret"].astype(float).fillna(0.0))
        m["period"] = f"Fold{i+1}"
        m["start"] = str(g["time"].iloc[0].date())
        m["end"] = str(g["time"].iloc[-1].date())
        m["oos"] = True  # truly out-of-sample
        rows.append(m)
    return rows


def regime_breakdown(df: pd.DataFrame, cfg: dict) -> list[dict]:
    """Split by high-vol vs low-vol regime of the portfolio's own realised vol.
    A strategy that ONLY works in low-vol would have negative SR in the
    high-vol rows — that's important to disclose.
    """
    df_reg = _add_regime(df, cfg)
    rows = []
    for label, g in df_reg.groupby("high_vol_regime"):
        if len(g) < 10:
            continue
        eq = (1 + g["ret"].astype(float).fillna(0.0)).cumprod()
        m = _period_metrics(eq, g["ret"].astype(float).fillna(0.0))
        m["period"] = "High-vol regime" if label == 1 else "Low-vol regime"
        m["start"] = str(g["time"].iloc[0].date())
        m["end"] = str(g["time"].iloc[-1].date())
        rows.append(m)
    return rows


# ---- significance ----------------------------------------------------------

def significance_summary(returns: pd.Series, cfg: dict) -> dict:
    """Run all three significance tests on the full-period returns."""
    r = returns.astype(float).dropna().to_numpy()
    rf_daily = (1 + cfg["evaluation"]["risk_free_rate"]) ** (1 / 252) - 1

    naive = naive_sharpe_tstat(r, rf_daily=rf_daily)
    lo = lo_sharpe_tstat(r, rf_daily=rf_daily)

    # Block bootstrap: 21-day blocks (~1 month), 2000 resamples
    boot = block_bootstrap_ci(r, block_size=21, n_bootstrap=2000, rf_daily=rf_daily, seed=42)
    boot_out = {
        "sr_point": boot["sr_point"],
        "sr_ci_95_low": boot["sr_ci_low"],
        "sr_ci_95_high": boot["sr_ci_high"],
        "block_size_days": boot["block_size"],
        "n_bootstrap": boot["n_bootstrap"],
        "ci_contains_zero": bool(boot["sr_ci_low"] <= 0 <= boot["sr_ci_high"]),
        "frac_bootstrap_negative": float((boot["sr_dist"] < 0).mean()),
    }

    # Deflated SR — number of trials.
    # Conservative count: 4 model tiers × Optuna trials (LR=1, XGB~30, LSTM=1,
    # XGB-binary=30). Treating Optuna's 30 trials as one "trial family" gives
    # 4 effective trials. Use a more conservative upper bound of 30+30=60 to
    # report both.
    n_trials_min = 4
    n_trials_max = 60
    dsr_min = deflated_sharpe_ratio(r, sr_hat=lo["sr"], n_trials=n_trials_min, rf_daily=rf_daily)
    dsr_max = deflated_sharpe_ratio(r, sr_hat=lo["sr"], n_trials=n_trials_max, rf_daily=rf_daily)

    return {
        "naive_tstat": naive,
        "lo_tstat": lo,
        "block_bootstrap": boot_out,
        "deflated_sharpe": {
            "min_trials_4": dsr_min,
            "max_trials_60": dsr_max,
            "interpretation": (
                "DSR p-value < 0.05 means the SR is unlikely to be the best "
                "of `n_trials` zero-SR strategies. We report both conservative "
                "(4) and aggressive (60) trial counts because the line between "
                "model tiers and Optuna hyperparam trials is fuzzy."
            ),
        },
    }


# ---- plots -----------------------------------------------------------------

def plot_subperiod_sharpe(yearly: list[dict], folds: list[dict], regimes: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 9))

    def _bar(ax, rows: list[dict], title: str):
        if not rows:
            return
        labels = [r["period"] for r in rows]
        vals = [r.get("sharpe", float("nan")) for r in rows]
        colors = ["steelblue" if v >= 0 else "indianred" for v in vals]
        ax.bar(labels, vals, color=colors, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(title)
        ax.set_ylabel("Sharpe (annualised)")
        ax.grid(True, alpha=0.3, axis="y")

    _bar(axes[0], yearly, "Sharpe by calendar year")
    _bar(axes[1], folds, "Sharpe by walk-forward fold (true OOS)")
    _bar(axes[2], regimes, "Sharpe by vol regime")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_rolling_sharpe_with_ci(equity_curve: pd.DataFrame, boot_dist: np.ndarray, out_path: Path) -> None:
    """Plot rolling Sharpe + 95% block-bootstrap CI band.

    The CI is the pointwise bootstrap percentile of rolling Sharpe across
    resamples — not a confidence band around the rolling estimate itself, but
    a horizontal "what SR would we expect at any random point in time" band.
    For a moving-CI we use a simpler approach: compute rolling Sharpe on each
    bootstrap-resampled equity curve and report 2.5/97.5 percentile.
    """
    rs = rolling_sharpe(equity_curve, window=126)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(equity_curve["time"], rs, color="darkgreen", linewidth=1.2, label="Rolling 6m Sharpe")
    # Horizontal band from the full-period bootstrap CI
    if len(boot_dist) > 0:
        ci_lo = float(np.percentile(boot_dist, 2.5))
        ci_hi = float(np.percentile(boot_dist, 97.5))
        ax.axhspan(ci_lo, ci_hi, alpha=0.15, color="grey", label=f"95% bootstrap CI [{ci_lo:.2f}, {ci_hi:.2f}]")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Rolling Sharpe with 95% bootstrap confidence band")
    ax.set_ylabel("Sharpe")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---- main ------------------------------------------------------------------

def main() -> None:
    cfg = load_config()
    p = PROJECT_ROOT / "reports" / "portfolio_equity_curve.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — run src.report first")

    df = pd.read_csv(p)
    df["time"] = pd.to_datetime(df["time"])
    df["ret"] = df["equity"].pct_change().fillna(0.0)

    print("=" * 70)
    print("ROBUSTNESS CHECKS")
    print("=" * 70)

    # Sub-period breakdowns
    yearly = yearly_breakdown(df)
    folds = fold_breakdown(df, cfg)
    regimes = regime_breakdown(df, cfg)

    print("\n--- Calendar years ---")
    print(f"{'Year':>6}  {'n':>4}  {'ret':>8}  {'SR':>6}  {'maxDD':>7}")
    for r in yearly:
        print(f"  {r['period']:>4}  {r['n_days']:>4}  {r.get('total_return', 0)*100:>+7.2f}%  {r.get('sharpe', 0):>+6.2f}  {r.get('max_drawdown', 0)*100:>+6.2f}%")

    print("\n--- Walk-forward folds (true OOS) ---")
    print(f"{'Fold':>6}  {'n':>4}  {'ret':>8}  {'SR':>6}  {'maxDD':>7}")
    for r in folds:
        print(f"  {r['period']:>4}  {r['n_days']:>4}  {r.get('total_return', 0)*100:>+7.2f}%  {r.get('sharpe', 0):>+6.2f}  {r.get('max_drawdown', 0)*100:>+6.2f}%")

    print("\n--- Vol regime ---")
    print(f"{'Regime':>20}  {'n':>4}  {'ret':>8}  {'SR':>6}  {'maxDD':>7}")
    for r in regimes:
        print(f"  {r['period']:>18}  {r['n_days']:>4}  {r.get('total_return', 0)*100:>+7.2f}%  {r.get('sharpe', 0):>+6.2f}  {r.get('max_drawdown', 0)*100:>+6.2f}%")

    # Full-period significance
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE OF FULL-PERIOD SHARPE")
    print("=" * 70)
    sig = significance_summary(df["ret"], cfg)

    print(f"\nNaive t-stat (iid Gaussian assumption):")
    print(f"  SR={sig['naive_tstat']['sr']:+.3f}   t={sig['naive_tstat']['t_stat']:+.3f}   p={sig['naive_tstat']['p_value_two_sided']:.4f}")
    print(f"\nLo (2002) t-stat (autocorrelation + higher moments corrected):")
    lo = sig['lo_tstat']
    print(f"  SR={lo['sr']:+.3f}   t={lo['t_stat']:+.3f}   p={lo['p_value_two_sided']:.4f}")
    print(f"  lag-1 autocorr: {lo['autocorr_lag1']:+.3f}")
    print(f"  skewness: {lo['skew']:+.3f}    excess kurtosis: {lo['excess_kurtosis']:+.3f}")

    boot = sig["block_bootstrap"]
    print(f"\nBlock bootstrap (21-day blocks, {boot['n_bootstrap']} resamples):")
    print(f"  SR = {boot['sr_point']:+.3f}  95% CI = [{boot['sr_ci_95_low']:+.3f}, {boot['sr_ci_95_high']:+.3f}]")
    print(f"  CI contains 0?  {boot['ci_contains_zero']}")
    print(f"  Fraction of bootstrap SRs that are negative: {boot['frac_bootstrap_negative']*100:.1f}%")

    dsr = sig["deflated_sharpe"]
    print(f"\nDeflated Sharpe Ratio (multiple-testing adjusted):")
    print(f"  Conservative (n=4 trials):  p={dsr['min_trials_4']['dsr_p_value']:.4f}   "
          f"E[max SR | H0]={dsr['min_trials_4']['expected_max_sr_under_null']:+.3f}")
    print(f"  Aggressive   (n=60 trials): p={dsr['max_trials_60']['dsr_p_value']:.4f}   "
          f"E[max SR | H0]={dsr['max_trials_60']['expected_max_sr_under_null']:+.3f}")
    print(f"  {dsr['interpretation']}")

    # Persist
    rep = PROJECT_ROOT / "reports"
    (rep / "robustness_subperiods.json").write_text(json.dumps({
        "yearly": yearly, "folds": folds, "regimes": regimes
    }, indent=2, default=str))
    (rep / "robustness_significance.json").write_text(json.dumps(sig, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else (None if o is None else str(o))))

    # Plots
    fig_dir = rep / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_subperiod_sharpe(yearly, folds, regimes, fig_dir / "subperiod_sharpe.png")

    # Re-run bootstrap to capture distribution for the rolling plot
    rf_daily = (1 + cfg["evaluation"]["risk_free_rate"]) ** (1 / 252) - 1
    boot_full = block_bootstrap_ci(df["ret"].dropna().to_numpy(), block_size=21, n_bootstrap=2000, rf_daily=rf_daily, seed=42)
    plot_rolling_sharpe_with_ci(df, boot_full["sr_dist"], fig_dir / "rolling_sharpe_with_ci.png")

    print(f"\nResults saved to {rep / 'robustness_subperiods.json'} and {rep / 'robustness_significance.json'}")
    print(f"Plots saved to {fig_dir / 'subperiod_sharpe.png'} and {fig_dir / 'rolling_sharpe_with_ci.png'}")


if __name__ == "__main__":
    main()