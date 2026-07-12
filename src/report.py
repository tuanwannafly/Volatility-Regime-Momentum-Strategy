"""
Generate tear-sheet + plots for the strategy.

Outputs:
  - reports/figures/equity_curve.png
  - reports/figures/rolling_sharpe.png
  - reports/tearsheets/<ticker>_tearsheet.html
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest.metrics import compute_metrics, rolling_sharpe
from .backtest.run import (
    buy_and_hold_benchmark,
    buy_and_hold_portfolio,
    run_ticker,
    run_universe,
)
from .config import PROJECT_ROOT, load_config


warnings.filterwarnings("ignore")


def plot_equity_curves(port_result: dict, bn_benchmark: dict, out_path: Path) -> None:
    port_curve = port_result["portfolio"]["equity_curve"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(port_curve["time"], port_curve["equity"], color="darkblue", linewidth=1.5, label="Strategy (vol-target)")
    # Build buy-hold benchmark equity
    bn_eq = pd.DataFrame()
    for r in bn_benchmark.get("per_ticker", []):
        bn_eq[r["ticker"]] = pd.Series(dtype=float)
    # Build buy-hold equity for each ticker and align
    eqs = []
    for r in bn_benchmark.get("per_ticker", []):
        from src.ingestion import load_one
        raw = load_one(r["ticker"])
        raw["time"] = pd.to_datetime(raw["time"])
        raw = raw.sort_values("time")
        ret = raw["close"].pct_change().fillna(0.0)
        eq = (1 + ret).cumprod()
        eqs.append(pd.DataFrame({"time": raw["time"], "eq": eq.values}).set_index("time")["eq"])
    if eqs:
        bn = pd.concat(eqs, axis=1).ffill()
        bn["mean_eq"] = bn.mean(axis=1)
        # Normalise both to start at the strategy start
        first = port_curve["time"].iloc[0]
        bn = bn[bn.index >= first]
        bn = bn / bn["mean_eq"].iloc[0] if len(bn) > 0 else bn
        ax1.plot(bn.index, bn["mean_eq"], color="grey", linewidth=1.2, linestyle="--", label="Buy & hold (avg)")
    ax1.set_title("Vol-target strategy vs Buy & Hold")
    ax1.set_ylabel("Equity (start = 1.0)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Drawdown subplot
    eq = port_curve["equity"]
    peak = eq.cummax()
    dd = eq / peak - 1
    ax2.fill_between(port_curve["time"], dd * 100, 0, color="steelblue", alpha=0.4)
    ax2.set_title("Strategy drawdown (%)")
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_rolling_sharpe(port_curve: pd.DataFrame, out_path: Path, window: int = 126) -> None:
    rs = rolling_sharpe(port_curve, window=window)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(port_curve["time"], rs, color="darkgreen", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(1, color="grey", linewidth=0.5, linestyle="--", alpha=0.6)
    ax.set_title(f"Rolling Sharpe ({window//21} months)")
    ax.set_ylabel("Sharpe")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    tickers = cfg["universe"]["tickers"]
    print(f"Running backtest on {len(tickers)} tickers ...")
    result = run_universe(cfg, tickers)
    bn = buy_and_hold_portfolio(tickers, cfg)

    print("\nPortfolio metrics:")
    print(json.dumps(result["portfolio"]["metrics"], indent=2))
    print("\nBuy & hold benchmark (avg):")
    print(json.dumps(bn["avg"], indent=2))

    # Save portfolio metrics JSON
    rep_dir = PROJECT_ROOT / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "portfolio_metrics.json").write_text(json.dumps(result["portfolio"]["metrics"], indent=2))
    (rep_dir / "buy_hold_metrics.json").write_text(json.dumps(bn, indent=2, default=str))

    # Save equity curve CSV
    result["portfolio"]["equity_curve"].to_csv(rep_dir / "portfolio_equity_curve.csv", index=False)

    # Plots
    fig_dir = rep_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_equity_curves(result, bn, fig_dir / "equity_curve.png")
    plot_rolling_sharpe(result["portfolio"]["equity_curve"], fig_dir / "rolling_sharpe.png")
    print(f"\nPlots saved to {fig_dir}")

    # Per-ticker plots
    for tkr, metrics in result.get("per_ticker", {}).items():
        print(f"\n[{tkr}]")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()