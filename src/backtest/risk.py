"""
Risk management helpers: vol-target sizing + drawdown circuit breaker.

vol_target_position_size(target_annual_vol, realised_vol)
    -> position weight in [0, max_size] that targets a constant risk budget.

drawdown_circuit_breaker(equity, threshold, cooldown_days)
    -> boolean mask of "exposure on/off" — when DD > threshold, cut to 0
       for cooldown_days trading days.

Both functions are pure (no side effects) — easier to test and reason about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_vol(returns: pd.Series, halflife: int = 20, annualise: bool = True) -> pd.Series:
    """Exponentially-weighted realised volatility.

    Uses the standard RiskMetrics formula:
        sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_{t-1}^2
    with lambda = exp(-ln(2)/halflife).
    """
    r = returns.astype(float).fillna(0.0)
    lam = np.exp(-np.log(2) / halflife)
    var = pd.Series(index=r.index, dtype=float)
    prev = float(r.iloc[0] ** 2) if len(r) else 0.0
    var.iloc[0] = prev
    for i in range(1, len(r)):
        prev = lam * prev + (1 - lam) * float(r.iloc[i - 1]) ** 2
        var.iloc[i] = prev
    out = np.sqrt(var)
    if annualise:
        out = out * np.sqrt(252)
    return out


def vol_target_position_size(
    realised_vol: pd.Series,
    target_vol: float,
    max_size: float = 1.0,
) -> pd.Series:
    """Compute position weight = target_vol / realised_vol, clipped to [0, max_size].

    `realised_vol` is annualised. Position < 1 means we use cash / safe asset.
    Position > 1 means leverage (we cap at max_size).
    """
    rv = realised_vol.astype(float).replace(0, np.nan)
    w = target_vol / rv
    w = w.clip(lower=0.0, upper=max_size)
    return w.fillna(0.0)


def drawdown_circuit_breaker(
    equity: pd.Series,
    threshold: float = 0.15,
    cooldown_days: int = 10,
) -> pd.Series:
    """Return a boolean mask = True means exposure ALLOWED.

    Logic:
      - Track running peak of equity.
      - When drawdown = (equity / peak - 1) falls below -threshold, turn off exposure.
      - Stay off for `cooldown_days` trading days, then resume.

    Note: "exposure off" is implemented as position weight = 0, not as
    forced liquidation — the strategy keeps whatever it currently holds
    in cash-equivalent during the cooldown.
    """
    eq = equity.astype(float)
    peak = eq.cummax()
    dd = eq / peak - 1.0

    allowed = pd.Series(True, index=eq.index)
    cooldown_until = None
    for i, (t, d) in enumerate(zip(eq.index, dd)):
        if cooldown_until is not None and t < cooldown_until:
            allowed.iloc[i] = False
            continue
        cooldown_until = None
        if d < -threshold:
            allowed.iloc[i] = False
            cooldown_until = eq.index[min(i + cooldown_days, len(eq) - 1)]
        else:
            allowed.iloc[i] = True
    return allowed