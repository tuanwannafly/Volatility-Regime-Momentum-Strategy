"""
Backtest engine: signal -> position -> PnL.

Inputs:
  - times (pd.DatetimeIndex-like): one row per trading day
  - proba_up (pd.Series): P(label == +1) from model
  - proba_down (pd.Series): P(label == -1)
  - returns (pd.Series): realised daily log-returns of the asset
  - realised_vol (pd.Series): annualised realised vol for sizing
  - cfg (dict): strategy config (costs, vol-target params, drawdown control)

Cost model (Vietnamese market, conservative):
  - cost_bps_per_side applied on every buy AND sell (brokerage + fees)
  - sell_tax_bps extra on sell only (Vietnamese capital-gains tax on equities)
  - slippage_bps as additional adverse fill

Signal -> position:
  - Expected sign = proba_up - proba_down in [-1, +1]
  - Vol-target size:  weight = clip(target_vol / realised_vol, 0, max_size)
  - Combined weight:  pos = sign * weight
  - Multiply by drawdown_allowed (0/1 mask)

PnL accounting:
  - We track equity and position. When position changes, we pay cost on
    |delta_position| * price.
  - Daily PnL = position_{t-1} * (close_t / close_{t-1} - 1) - cost paid today

Output:
  - DataFrame with: time, signal, weight, position, cost, pnl, equity
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..config import load_config
from .risk import drawdown_circuit_breaker, vol_target_position_size


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame  # columns: time, signal, weight, position, cost, pnl, equity, exposure_on

    def summary(self, risk_free: float = 0.04) -> dict:
        from .metrics import compute_metrics
        return compute_metrics(self.equity_curve, risk_free=risk_free)


def run_backtest(
    times: pd.Series,
    returns: pd.Series,
    proba: np.ndarray,
    cfg: dict | None = None,
    realised_vol: Optional[pd.Series] = None,
) -> BacktestResult:
    """Run the vol-targeting long-only momentum backtest.

    Parameters
    ----------
    times : pd.Series
        Date index of length N.
    returns : pd.Series
        Realised daily simple returns (close_t / close_{t-1} - 1).
    proba : np.ndarray of shape (N, 3)
        Softmax probabilities for classes [down, flat, up].
    cfg : dict
        Strategy config.
    realised_vol : pd.Series, optional
        If None, computed from returns with EWMA.

    Returns
    -------
    BacktestResult
    """
    if cfg is None:
        cfg = load_config()
    b = cfg["backtest"]

    times = pd.to_datetime(pd.Series(times).reset_index(drop=True))
    rets = pd.Series(returns).reset_index(drop=True).astype(float)
    n = len(times)
    assert len(rets) == n and proba.shape[0] == n, (len(rets), n, proba.shape)

    proba = np.asarray(proba, dtype=float)
    # Class order from labeling: {-1: 0, 0: 1, 1: 2} -> proba columns: [down, flat, up]
    proba_up = proba[:, 2]
    proba_down = proba[:, 0]

    # Rescale the raw (up - down) signal so that a typical "edge" of 0.15
    # becomes a meaningful position. We use a soft stretching:
    #   signal = tanh(k * (up - down))
    # which is in (-1, +1) and amplifies confident calls.
    raw_signal = proba_up - proba_down
    signal = pd.Series(np.tanh(2.5 * raw_signal), index=range(n))

    if realised_vol is None:
        from .risk import ewma_vol
        realised_vol = ewma_vol(rets, halflife=b.get("ewma_halflife", 20))
    realised_vol = realised_vol.reset_index(drop=True)

    weight = vol_target_position_size(
        realised_vol,
        target_vol=b["target_annual_vol"],
        max_size=b["max_position_size"],
    )

    # Drawdown control: only meaningful once we have an equity curve.
    # We do a first pass to get raw equity, then apply the breaker.
    raw_pos = signal * weight
    # Build a preliminary equity to compute drawdown
    eq_raw = _compute_equity(rets, raw_pos, cost_bps=b["cost_bps_per_side"], sell_tax_bps=b["sell_tax_bps"], slippage_bps=b["slippage_bps"])
    breaker_on = drawdown_circuit_breaker(eq_raw, threshold=b["max_drawdown_threshold"], cooldown_days=b["drawdown_cooldown_days"])
    breaker_on = breaker_on.reset_index(drop=True)

    final_pos = raw_pos * breaker_on.astype(float)
    eq = _compute_equity(rets, final_pos, cost_bps=b["cost_bps_per_side"], sell_tax_bps=b["sell_tax_bps"], slippage_bps=b["slippage_bps"])

    out = pd.DataFrame({
        "time": times,
        "signal": signal,
        "weight": weight,
        "exposure_on": breaker_on.astype(int),
        "position": final_pos,
        "cost": _compute_cost_series(final_pos, cost_bps=b["cost_bps_per_side"], sell_tax_bps=b["sell_tax_bps"], slippage_bps=b["slippage_bps"]),
        "pnl": _compute_pnl_series(rets, final_pos),
        "equity": eq,
    })
    return BacktestResult(equity_curve=out)


def _compute_equity(rets: pd.Series, pos: pd.Series, cost_bps: float, sell_tax_bps: float, slippage_bps: float) -> pd.Series:
    """Equity curve starting from 1.0, applying pnl - cost on each day."""
    eq = np.ones(len(rets), dtype=float)
    cost_paid = np.zeros(len(rets), dtype=float)
    pnl_only = np.zeros(len(rets), dtype=float)

    p = pos.to_numpy()
    r = rets.to_numpy()

    for i in range(1, len(rets)):
        # daily PnL from yesterday's position -> today's return
        pnl = p[i - 1] * r[i]
        # cost from changing position today
        dpos = abs(p[i] - p[i - 1])
        # cost is on traded notional. We approximate by treating |dpos| as fraction
        # of capital traded. Total bps = cost_bps + slippage_bps on both sides,
        # plus sell_tax_bps when the side is "sell" (dpos and new pos lower than old).
        side = bps_cost(dpos, cost_bps, sell_tax_bps, slippage_bps, p[i - 1], p[i])
        cost_paid[i] = side
        pnl_only[i] = pnl
        # cost scaled by current equity (multiplicative, not absolute subtraction)
        eq[i] = eq[i - 1] * (1 + pnl) - side * eq[i - 1]

    return pd.Series(eq)


def _compute_cost_series(pos: pd.Series, cost_bps: float, sell_tax_bps: float, slippage_bps: float) -> pd.Series:
    p = pos.to_numpy()
    cost = np.zeros(len(p), dtype=float)
    for i in range(1, len(p)):
        dpos = abs(p[i] - p[i - 1])
        cost[i] = bps_cost(dpos, cost_bps, sell_tax_bps, slippage_bps, p[i - 1], p[i])
    return pd.Series(cost)


def _compute_pnl_series(rets: pd.Series, pos: pd.Series) -> pd.Series:
    p = pos.to_numpy()
    r = rets.to_numpy()
    pnl = np.zeros(len(p), dtype=float)
    for i in range(1, len(p)):
        pnl[i] = p[i - 1] * r[i]
    return pd.Series(pnl)


def bps_cost(dpos: float, cost_bps: float, sell_tax_bps: float, slippage_bps: float, pos_prev: float, pos_new: float) -> float:
    """Total cost of a position change.

    - brokerage: cost_bps on both sides of the trade (buy + sell) -> dpos * cost_bps/1e4
    - sell tax:   sell_tax_bps when decreasing position (selling more than buying)
    - slippage:   slippage_bps on both sides
    """
    if dpos <= 0:
        return 0.0
    brokerage = dpos * (cost_bps / 1e4)
    slippage = dpos * (slippage_bps / 1e4)
    sell_side = max(0.0, pos_prev - pos_new)
    tax = sell_side * (sell_tax_bps / 1e4)
    return brokerage + slippage + tax