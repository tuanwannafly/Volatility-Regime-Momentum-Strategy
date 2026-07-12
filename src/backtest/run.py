"""
Driver: load model predictions for the universe, run backtest for each,
aggregate to an equal-weight portfolio, compare to buy-and-hold benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import run_backtest
from ..backtest.metrics import compute_metrics, rolling_sharpe
from ..config import PROJECT_ROOT, load_config
from ..ingestion import load_one


PRED_DIR = PROJECT_ROOT / "data" / "predictions"


def load_predictions(ticker: str) -> pd.DataFrame:
    p = PRED_DIR / f"{ticker}_predictions.json"
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Prefer the binary (directional) classifier for trading — it gives a
    # sharper P(up) signal that the multi-class model.
    if "xgb_binary" in data and data["xgb_binary"]["folds"]:
        model_key = "xgb_binary"
        rows = []
        for fold in data[model_key]["folds"]:
            for t, y, p in zip(fold["times"], fold["y_true"], fold["proba"]):
                # proba is a scalar P(up) in [0,1]; expand to 3 columns
                p_up = float(p)
                p_down = 1.0 - p_up
                p_flat = 0.0
                rows.append({
                    "time": pd.to_datetime(t), "label": y,
                    "proba": [p_down, p_flat, p_up],
                })
        df = pd.DataFrame(rows)
        print(f"  [{ticker}] using XGB-binary ({data[model_key]['agg']})")
    else:
        # fallback: best of LR / XGB / LSTM by logloss
        best_model = min(["lr", "xgb", "lstm"], key=lambda m: data[m]["agg"]["logloss_mean"])
        print(f"  [{ticker}] fallback best model by logloss: {best_model}  ({data[best_model]['agg']})")
        rows = []
        for fold in data[best_model]["folds"]:
            for t, y, p in zip(fold["times"], fold["y_true"], fold["proba"]):
                rows.append({"time": pd.to_datetime(t), "label": y, "proba": p})
        df = pd.DataFrame(rows)

    df["proba_down"] = df["proba"].apply(lambda v: v[0])
    df["proba_flat"] = df["proba"].apply(lambda v: v[1])
    df["proba_up"] = df["proba"].apply(lambda v: v[2])
    df = df.drop(columns=["proba"]).sort_values("time").reset_index(drop=True)
    return df


def merge_with_returns(pred_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    raw = load_one(ticker)
    raw = raw[["time", "close"]].copy()
    raw["time"] = pd.to_datetime(raw["time"])
    raw["ret"] = raw["close"].pct_change()
    raw = raw.dropna().reset_index(drop=True)
    out = pred_df.merge(raw[["time", "ret"]], on="time", how="inner").sort_values("time").reset_index(drop=True)
    return out


def run_ticker(ticker: str, cfg: dict) -> dict:
    pred = load_predictions(ticker)
    merged = merge_with_returns(pred, ticker)
    if len(merged) < 60:
        return {"ticker": ticker, "error": "too few rows"}
    proba = np.stack([merged["proba_down"].to_numpy(), merged["proba_flat"].to_numpy(), merged["proba_up"].to_numpy()], axis=1)

    result = run_backtest(
        times=merged["time"],
        returns=merged["ret"],
        proba=proba,
        cfg=cfg,
    )
    ec = result.equity_curve.copy()
    ec["ret"] = merged["ret"].values
    metrics = result.summary(risk_free=cfg["evaluation"]["risk_free_rate"])
    metrics["ticker"] = ticker
    return {"ticker": ticker, "equity_curve": ec, "metrics": metrics}


def run_universe(cfg: dict, tickers: list[str]) -> dict:
    """Run each ticker individually, then build an equal-weight portfolio.

    Portfolio construction:
      - Use UNION of trading dates across tickers (each ticker contributes
        when it has a prediction; forward-fill positions to handle gaps).
      - For each date, position = mean of available ticker positions.
      - PnL  = position_prev * mean(available ticker returns) on that day.
      - Apply same cost model as per-ticker.
    """
    per_ticker = {}
    ticker_data: list[tuple[str, pd.DataFrame]] = []  # (ticker, df with time, ret, pos)
    metrics_rows = []
    for t in tickers:
        try:
            r = run_ticker(t, cfg)
            per_ticker[t] = r
            if "equity_curve" in r:
                ec = r["equity_curve"].copy()
                ec["ret"] = ec.get("ret", 0.0)
                ticker_data.append((t, ec[["time", "ret", "position"]].copy()))
                metrics_rows.append(r["metrics"])
        except Exception as e:  # noqa: BLE001
            print(f"  {t}: backtest failed ({e})")

    if not ticker_data:
        return {"per_ticker": per_ticker, "portfolio": None}

    # Build per-ticker wide frames
    pos_by_t = {t: df.set_index("time")["position"] for t, df in ticker_data}
    ret_by_t = {t: df.set_index("time")["ret"] for t, df in ticker_data}

    # Union of dates
    all_dates = sorted(set().union(*[set(s.index) for s in pos_by_t.values()]))
    pos_df = pd.DataFrame({t: pos_by_t[t].reindex(all_dates) for t in pos_by_t}).ffill().fillna(0.0)
    ret_df = pd.DataFrame({t: ret_by_t[t].reindex(all_dates) for t in ret_by_t}).fillna(0.0)

    # Equal weight: each ticker's vol-target position, portfolio = sum of
    # positions (NOT mean) so we get full exposure to all N assets.
    # Cap total exposure at max_position_size * N to prevent over-leverage.
    pos_df_filled = pos_df.ffill().fillna(0.0)
    ret_df_filled = ret_df.fillna(0.0)
    # Only count a ticker as "available" when its position is non-zero (i.e. actively traded)
    avail_mask = pos_df_filled.abs() > 1e-9
    # If no ticker active, treat as cash
    port_pos = pos_df_filled.where(avail_mask, 0.0).sum(axis=1)
    # Returns: weighted by position magnitude (so larger positions contribute more)
    # Use simple mean for diversification
    port_ret = ret_df_filled.where(avail_mask, 0.0).sum(axis=1) / avail_mask.sum(axis=1).replace(0, 1)

    # Cap at N * max_position_size for safety
    b = cfg["backtest"]
    n = len(ticker_data)
    cap = b["max_position_size"] * n
    port_pos = port_pos.clip(-cap, cap)

    # Build portfolio equity curve with cost
    b = cfg["backtest"]
    eq = np.ones(len(all_dates), dtype=float)
    cost_paid = np.zeros(len(all_dates), dtype=float)
    pnl_only = np.zeros(len(all_dates), dtype=float)
    pos_prev = 0.0
    from .engine import bps_cost
    for i in range(len(all_dates)):
        if i == 0:
            eq[i] = 1.0
            pos_prev = float(port_pos.iloc[i])
            continue
        pnl = pos_prev * float(port_ret.iloc[i])
        dpos = abs(float(port_pos.iloc[i]) - pos_prev)
        side = bps_cost(dpos, b["cost_bps_per_side"], b["sell_tax_bps"], b["slippage_bps"], pos_prev, float(port_pos.iloc[i]))
        cost_paid[i] = side
        pnl_only[i] = pnl
        eq[i] = eq[i - 1] * (1 + pnl) - side * eq[i - 1]
        pos_prev = float(port_pos.iloc[i])

    # Apply drawdown circuit breaker
    from .risk import drawdown_circuit_breaker
    breaker_on = drawdown_circuit_breaker(pd.Series(eq), threshold=b["max_drawdown_threshold"], cooldown_days=b["drawdown_cooldown_days"])
    # Re-run with breaker applied to positions
    final_pos = port_pos * breaker_on.astype(float).values
    eq2 = np.ones(len(all_dates), dtype=float)
    pos_prev = 0.0
    for i in range(len(all_dates)):
        if i == 0:
            eq2[i] = 1.0
            pos_prev = float(final_pos.iloc[i])
            continue
        pnl = pos_prev * float(port_ret.iloc[i])
        dpos = abs(float(final_pos.iloc[i]) - pos_prev)
        side = bps_cost(dpos, b["cost_bps_per_side"], b["sell_tax_bps"], b["slippage_bps"], pos_prev, float(final_pos.iloc[i]))
        eq2[i] = eq2[i - 1] * (1 + pnl) - side * eq2[i - 1]
        pos_prev = float(final_pos.iloc[i])

    port_curve = pd.DataFrame({
        "time": pd.to_datetime(all_dates),
        "position": final_pos.values,
        "pnl": pnl_only,
        "cost": cost_paid,
        "equity": eq2,
        "exposure_on": breaker_on.astype(int).values,
    })

    metrics = compute_metrics(port_curve, risk_free=cfg["evaluation"]["risk_free_rate"])
    metrics["n_assets"] = len(ticker_data)

    return {
        "per_ticker": {k: v.get("metrics") for k, v in per_ticker.items()},
        "per_ticker_metrics": metrics_rows,
        "portfolio": {"equity_curve": port_curve, "metrics": metrics},
    }


def buy_and_hold_benchmark(ticker: str, cfg: dict) -> dict:
    """Compute the equity curve if we had just bought and held the ticker."""
    raw = load_one(ticker)
    raw["time"] = pd.to_datetime(raw["time"])
    raw["ret"] = raw["close"].pct_change()
    rets = raw["ret"].fillna(0.0)
    eq = (1 + rets).cumprod()
    df = pd.DataFrame({"time": raw["time"], "equity": eq})
    m = compute_metrics(df, risk_free=cfg["evaluation"]["risk_free_rate"])
    m["ticker"] = ticker
    return m


def buy_and_hold_portfolio(tickers: list[str], cfg: dict) -> dict:
    rows = []
    for t in tickers:
        try:
            rows.append(buy_and_hold_benchmark(t, cfg))
        except Exception:
            continue
    # Average Sharpe / max DD / CAGR across names as a simple summary
    if not rows:
        return {}
    avg = {
        "sharpe": float(np.mean([r["sharpe"] for r in rows])),
        "cagr": float(np.mean([r["cagr"] for r in rows])),
        "max_drawdown": float(np.mean([r["max_drawdown"] for r in rows])),
        "annual_vol": float(np.mean([r["annual_vol"] for r in rows])),
    }
    return {"per_ticker": rows, "avg": avg}


if __name__ == "__main__":
    cfg = load_config()
    tickers = cfg["universe"]["tickers"]
    print(f"Backtesting {len(tickers)} tickers ...")
    res = run_universe(cfg, tickers)
    print("\nPortfolio metrics:")
    print(json.dumps(res["portfolio"]["metrics"], indent=2))
    print("\nPer-ticker metrics:")
    for r in res["per_ticker_metrics"]:
        print(f"  {r['ticker']:5s}  Sharpe {r['sharpe']:+.2f}  CAGR {r['cagr']*100:+.1f}%  MaxDD {r['max_drawdown']*100:+.1f}%")

    bn = buy_and_hold_portfolio(tickers, cfg)
    print("\nBuy & hold benchmark (avg across universe):")
    print(json.dumps(bn["avg"], indent=2))