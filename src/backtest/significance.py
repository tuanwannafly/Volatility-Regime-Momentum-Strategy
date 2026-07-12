"""
Statistical significance of the Sharpe ratio.

Three flavours, from least to most rigorous:

1. **Naive t-stat**   — `SR * sqrt(N)`. Only valid under iid Gaussian returns.
                        Reported for completeness, NOT recommended.

2. **Lo (2002) t-stat** — corrects for autocorrelation and higher moments
                          (skewness, kurtosis). Paper: "The Statistics of
                          Sharpe Ratios", FAJ 2002.
                          t = SR * sqrt(1 / (1 - phi) - 2 * phi / (N * (1 - phi)^3))
                          where phi is the lag-1 autocorrelation of squared
                          returns (variance autocorrelation proxy).

3. **Block bootstrap CI** — resample BLOCKS of returns (not individual days)
                           to preserve serial dependence. Recompute Sharpe on
                           each resample. 95% CI = [2.5th, 97.5th] percentile.

4. **Deflated Sharpe Ratio (DSR)** — Bailey & Lopez de Prado, 2014.
                                      Adjusts the SR for the number of
                                      trials tried (in our case: 4 model
                                      variants x hyperparam tuning). A SR of
                                      0.86 from 1 trial is impressive; the
                                      same SR from 100 trials is much less so.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def _to_daily_returns(equity: pd.Series) -> pd.ndarray:
    eq = equity.astype(float).reset_index(drop=True)
    rets = eq.pct_change().dropna().to_numpy()
    return rets


def naive_sharpe_tstat(returns: Sequence[float], rf_daily: float = 0.0) -> dict:
    """Naive t-stat assuming iid Gaussian returns. `returns` are already excess
    if rf_daily=0, otherwise raw returns minus risk-free."""
    r = np.asarray(returns, dtype=float)
    r = r - rf_daily
    n = len(r)
    sr = r.mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else 0.0
    t = sr * np.sqrt(n / TRADING_DAYS)  # same as SR * sqrt(n_days / 252)
    p = 2 * (1 - _normal_cdf(abs(t)))
    return {"sr": float(sr), "t_stat": float(t), "p_value_two_sided": float(p), "n_days": int(n)}


def _normal_cdf(x: float) -> float:
    """Standard normal CDF using math.erf."""
    from math import erf, sqrt
    return 0.5 * (1 + erf(x / sqrt(2)))


def lo_sharpe_tstat(returns: Sequence[float], rf_daily: float = 0.0) -> dict:
    """Lo (2002) t-stat: corrects for autocorrelation and higher moments.

    Reference formula (Lo, 2002, eq 7):
        t = sqrt(v) * (SR - SR*) / sqrt(1 - gamma1 * SR + (gamma2 - 1) * SR^2 / 4)

    where v = number of observations per year, SR* = benchmark (0 if we test
    against zero), and:
        gamma1 = skewness of returns
        gamma2 = kurtosis of returns

    For autocorrelation, Lo notes the variance of the SR estimator itself is
    inflated by serial dependence. A pragmatic approximation:

        var(SR) approx= (1 + 2*sum(rho_k)) / n  where rho_k = autocorr at lag k

    We compute it directly on daily returns, summed up to lag 50.
    """
    r = np.asarray(returns, dtype=float)
    r = r - rf_daily
    n = len(r)
    if n < 30 or r.std() == 0:
        return {"sr": float("nan"), "t_stat": float("nan"), "p_value_two_sided": float("nan"),
                "autocorr_lag1": float("nan"), "skew": float("nan"), "kurt": float("nan"), "n_days": int(n)}

    sr = r.mean() / r.std() * np.sqrt(TRADING_DAYS)

    # Serial correlation correction
    # Variance of SR-hat is approx:
    #   Var(SR) = (SR^2 + 1) * (1 + 2*sum_k rho_k) / n
    # sum of autocorrelations up to lag L (Lo uses L=22 for daily in his paper)
    autocorrs = []
    r_centered = r - r.mean()
    var = r_centered.var()
    for k in range(1, 50):
        if k >= n:
            break
        c = np.corrcoef(r_centered[:-k], r_centered[k:])[0, 1]
        autocorrs.append(c)
    # Lo's recommendation: only sum positive autocorrelations, or use Newey-West
    pos_sum = sum(max(0.0, a) for a in autocorrs)

    # Higher moments
    s = r.std()
    if s > 0:
        skew = float((((r - r.mean()) / s) ** 3).mean())
        kurt = float((((r - r.mean()) / s) ** 4).mean())
    else:
        skew = 0.0
        kurt = 3.0
    excess_kurt = kurt - 3.0

    # Variance of SR (annualised)
    var_sr = (1 + pos_sum) * (1 + sr ** 2 / TRADING_DAYS ** 2 * 0) / n
    # Conservative correction using Lo's exact formula (gamma1=skew, gamma2=kurt)
    var_sr_lo = (
        1.0
        - skew * sr / np.sqrt(TRADING_DAYS)
        + (excess_kurt - 1) * sr ** 2 / (4 * TRADING_DAYS)
    ) / (n - 1) if n > 1 else 1.0
    var_sr_lo = max(var_sr_lo, 1e-12)
    t = sr / np.sqrt(var_sr_lo * TRADING_DAYS)  # multiply by sqrt(252) to annualise

    # Two-sided p-value using normal approx (Lo shows t is approx normal for n>60)
    p = 2 * (1 - _normal_cdf(abs(t)))
    return {
        "sr": float(sr),
        "t_stat": float(t),
        "p_value_two_sided": float(p),
        "autocorr_lag1": float(autocorrs[0]) if autocorrs else 0.0,
        "pos_autocorr_sum": float(pos_sum),
        "skew": float(skew),
        "excess_kurtosis": float(excess_kurt),
        "n_days": int(n),
    }


def block_bootstrap_ci(
    returns: Sequence[float],
    block_size: int = 21,
    n_bootstrap: int = 2000,
    rf_daily: float = 0.0,
    seed: int = 42,
) -> dict:
    """Block bootstrap confidence interval for annualised Sharpe.

    We resample WITH REPLACEMENT blocks of `block_size` consecutive returns
    (default ~1 trading month). This preserves the serial dependence of
    returns within a block, which the iid bootstrap would destroy.

    Returns
    -------
    dict with:
        sr_point: point estimate
        sr_ci_low / sr_ci_high: 2.5/97.5 percentile CI
        sr_dist: full distribution of bootstrapped SRs
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, dtype=float) - rf_daily
    n = len(r)
    if n < block_size * 3:
        return {"sr_point": float("nan"), "sr_ci_low": float("nan"), "sr_ci_high": float("nan"),
                "sr_dist": np.array([])}

    sr_point = r.mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else 0.0

    n_blocks = (n + block_size - 1) // block_size
    blocks = [r[i * block_size : (i + 1) * block_size] for i in range(n_blocks)]
    # Pad the last block so all are equal length
    last = blocks[-1]
    if len(last) < block_size:
        last = np.concatenate([last, np.zeros(block_size - len(last))])
        blocks[-1] = last

    sr_dist = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        sample_blocks = rng.choice(n_blocks, size=n_blocks, replace=True)
        sample = np.concatenate([blocks[i] for i in sample_blocks])[:n]
        std = sample.std()
        if std <= 0:
            sr_dist[b] = 0.0
        else:
            sr_dist[b] = sample.mean() / std * np.sqrt(TRADING_DAYS)

    ci_low, ci_high = np.percentile(sr_dist, [2.5, 97.5])
    return {
        "sr_point": float(sr_point),
        "sr_ci_low": float(ci_low),
        "sr_ci_high": float(sr_high := ci_high),
        "sr_dist": sr_dist,
        "block_size": block_size,
        "n_bootstrap": n_bootstrap,
    }


def deflated_sharpe_ratio(
    returns: Sequence[float],
    sr_hat: float,
    n_trials: int,
    rf_daily: float = 0.0,
    skew: float | None = None,
    kurt: float | None = None,
) -> dict:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    The classical SR test assumes ONE trial. If we tried N strategies and
    picked the best, the SR of the best is biased upward by multiple testing.

    DSR computes the probability that the true SR exceeds a benchmark SR*
    (typically 0), adjusted for the number of trials and the higher moments of
    the return distribution.

    Parameters
    ----------
    returns : array of daily returns
    sr_hat : the observed (annualised) SR of the chosen strategy
    n_trials : number of strategies tried (model variants * hyperparam configs)
    skew, kurt : higher moments; if None, estimated from `returns`
    """
    r = np.asarray(returns, dtype=float) - rf_daily
    n = len(r)
    if n < 30 or r.std() == 0:
        return {"dsr": float("nan"), "p_value": float("nan"), "expected_max_sr": float("nan")}

    sr = sr_hat
    if skew is None:
        s = r.std()
        if s > 0:
            skew = float((((r - r.mean()) / s) ** 3).mean())
        else:
            skew = 0.0
    if kurt is None:
        s = r.std()
        if s > 0:
            kurt = float((((r - r.mean()) / s) ** 4).mean())
        else:
            kurt = 3.0
    excess_kurt = kurt - 3.0

    # Variance of SR estimate (Lo-style)
    autocorrs = []
    r_centered = r - r.mean()
    for k in range(1, 50):
        if k >= n:
            break
        c = np.corrcoef(r_centered[:-k], r_centered[k:])[0, 1]
        autocorrs.append(max(0.0, c))
    pos_sum = sum(autocorrs)

    # se(SR) — Lo's formula
    var_sr = (
        1.0
        - skew * sr / np.sqrt(TRADING_DAYS)
        + (excess_kurt - 1) * sr ** 2 / (4 * TRADING_DAYS)
    ) / (n - 1)
    se_sr = np.sqrt(max(var_sr, 1e-12))

    # Expected maximum SR under the null (all trials have true SR=0)
    # E[max] = se * (1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N*e))
    # Approximation from BLP 2014 (eq 4):
    #   e_max_z = (1 - gamma) * z_{1-1/N} + gamma * z_{1-1/(N*e)}
    #   E[max SR] = e_max_z * se_SR + sr*
    from math import log, sqrt
    gamma = 0.5772156649  # Euler-Mascheroni
    z = _normal_quantile
    if n_trials <= 1:
        e_max_z = 0.0
    else:
        e_max_z = (1 - gamma) * z(1 - 1.0 / n_trials) + gamma * z(1 - 1.0 / (n_trials * np.e))
    expected_max_sr = e_max_z * se_sr * np.sqrt(TRADING_DAYS)

    # DSR p-value: probability that true SR > SR* (often 0) given the observed SR_hat
    # is the better of N trials.
    # Phi( (SR_hat - SR*) / se_SR ) — but adjusted so that the threshold
    # accounts for multiple testing.
    # We compute: probability that SR > expected_max_sr under null
    dsr_z = (sr - expected_max_sr) / (se_sr * np.sqrt(TRADING_DAYS))
    dsr_p = 1 - _normal_cdf(dsr_z)
    return {
        "dsr_p_value": float(dsr_p),
        "expected_max_sr_under_null": float(expected_max_sr),
        "n_trials": int(n_trials),
        "se_sr": float(se_sr),
    }


def _normal_quantile(p: float) -> float:
    """Inverse normal CDF using rational approximation (Abramowitz & Stegun)."""
    from math import log, sqrt
    if p <= 0:
        return -6.0
    if p >= 1:
        return 6.0
    # Beasley-Springer-Moro
    a = [
        -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
        1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
        6.680131188771972e01, -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
        -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = sqrt(-2 * log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = sqrt(-2 * log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


if __name__ == "__main__":
    # Quick sanity test with iid Gaussian
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, 1260)  # ~5y daily
    naive = naive_sharpe_tstat(r)
    lo = lo_sharpe_tstat(r)
    boot = block_bootstrap_ci(r, block_size=21, n_bootstrap=500)
    dsr = deflated_sharpe_ratio(r, sr_hat=lo["sr"], n_trials=4)
    print("Naive t-stat:", naive)
    print("Lo t-stat:", lo)
    print(f"Bootstrap CI: {boot['sr_point']:.3f}  [{boot['sr_ci_low']:.3f}, {boot['sr_ci_high']:.3f}]")
    print("DSR:", dsr)