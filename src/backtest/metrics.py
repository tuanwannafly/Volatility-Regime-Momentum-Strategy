"""
Performance metrics used for evaluating the strategy.

Per the plan we focus on industry-standard metrics, NOT raw accuracy:
  - Sharpe, Sortino, Calmar
  - Max drawdown
  - Win rate, avg win/loss
  - Turnover
  - Rolling Sharpe
  - Annualised return / vol
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def compute_metrics(equity_curve: pd.DataFrame, risk_free: float = 0.04) -> dict:
    eq = equity_curve["equity"].astype(float).reset_index(drop=True)
    rets = eq.pct_change().fillna(0.0)

    # Annualised return / vol
    n_days = len(eq)
    if n_days < 2 or eq.iloc[0] <= 0:
        return {"error": "insufficient data"}

    total_ret = eq.iloc[-1] / eq.iloc[0] - 1.0
    years = n_days / TRADING_DAYS
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0.0
    vol = rets.std() * np.sqrt(TRADING_DAYS)
    rf_daily = (1 + risk_free) ** (1 / TRADING_DAYS) - 1
    excess = rets - rf_daily

    sharpe = (excess.mean() / rets.std()) * np.sqrt(TRADING_DAYS) if rets.std() > 0 else 0.0
    downside = rets[rets < 0].std()
    sortino = (excess.mean() / downside) * np.sqrt(TRADING_DAYS) if downside and downside > 0 else 0.0

    # Drawdown
    peak = eq.cummax()
    dd = eq / peak - 1.0
    max_dd = float(dd.min())

    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    # Win / loss
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    win_rate = float(len(wins) / max(1, len(wins) + len(losses)))
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    payoff = (avg_win / abs(avg_loss)) if avg_loss < 0 else 0.0

    # Turnover — sum of |dpos| over period (optional, portfolio curve may not have it)
    if "position" in equity_curve.columns:
        turnover = float(equity_curve["position"].diff().abs().sum())
    else:
        turnover = float("nan")

    return {
        "total_return": float(total_ret),
        "cagr": float(cagr),
        "annual_vol": float(vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "payoff": float(payoff),
        "turnover": float(turnover),
        "n_days": int(n_days),
    }


def rolling_sharpe(equity_curve: pd.DataFrame, window: int = 126, risk_free: float = 0.04) -> pd.Series:
    """Rolling Sharpe over `window` trading days (default 6 months)."""
    eq = equity_curve["equity"].astype(float).reset_index(drop=True)
    rets = eq.pct_change().fillna(0.0)
    rf_daily = (1 + risk_free) ** (1 / TRADING_DAYS) - 1
    excess = rets - rf_daily
    roll_mean = excess.rolling(window).mean()
    roll_std = rets.rolling(window).std()
    rs = (roll_mean / roll_std) * np.sqrt(TRADING_DAYS)
    return rs.replace([np.inf, -np.inf], np.nan)