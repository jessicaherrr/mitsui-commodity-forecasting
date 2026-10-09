"""Out-of-sample evaluation of variance forecasts.

Loss functions (f = variance forecast > 0, s = variance proxy >= 0)

    QLIKE(s, f) = log f + s / f          (equivalent to s/f - log(s/f) - 1 up to a term in s
                                          only, but well defined when s = 0)
    MSE(s, f)   = (s - f)^2

Both are *robust* in the sense of Patton (2011): with a conditionally unbiased proxy
(E_t[s] = true variance) they rank forecasts as the true variance would. QLIKE differences
are scale-free (multiplying f and s by c changes neither s/f nor log(f1/f2)), so they can be
averaged across assets; MSE is reported per asset as ratios.

Proxies
-------
* squared returns summed over the forecast window: conditionally unbiased but extremely noisy
  for one day (Andersen & Bollerslev 1998); the primary proxy;
* Garman-Klass (1980) range variance 0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2, available for US
  and JPX instruments with OHLC: far less noisy but *biased downward* because it omits the
  overnight move, so model rankings under it are a sensitivity check only.

Tests
-----
* Diebold-Mariano: d_t = L_a,t - L_b,t; t = mean(d) / NW-se(d) (HAC; lags >= window length
  because multi-day proxies overlap);
* panel DM: d_t averaged across assets on each date first, then the same HAC t-test
  (cross-sectional dependence is absorbed by averaging);
* Model Confidence Set (Hansen, Lunde & Nason 2011): stationary block bootstrap, alpha = 0.10;
* Mincer-Zarnowitz: s_t = a + b f_t + u_t with HAC; ideal a = 0, b = 1.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..evaluation.metrics import newey_west_se


def qlike(s: np.ndarray, f: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(f > 0, np.log(f) + s / f, np.nan)


def mse(s: np.ndarray, f: np.ndarray) -> np.ndarray:
    return (s - f) ** 2


def dm_test(d: np.ndarray, lags: int) -> tuple[float, float, float, int]:
    d = d[np.isfinite(d)]
    if len(d) < 30:
        return np.nan, np.nan, np.nan, len(d)
    se = newey_west_se(d, lags)
    t = d.mean() / se if se > 0 else np.nan
    return float(d.mean()), float(t), float(2 * norm.sf(abs(t))) if np.isfinite(t) else np.nan, len(d)


def mcs_included(losses: pd.DataFrame, alpha: float = 0.10, reps: int = 500, block: int = 10, seed: int = 0) -> list[str]:
    from arch.bootstrap import MCS

    L = losses.dropna()
    if len(L) < 50:
        return []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mcs = MCS(L, size=alpha, reps=reps, block_size=block, method="R", bootstrap="stationary", seed=seed)
        mcs.compute()
    return [str(m) for m in mcs.included]


def mincer_zarnowitz(s: np.ndarray, f: np.ndarray, lags: int) -> dict:
    import statsmodels.api as sm

    ok = np.isfinite(s) & np.isfinite(f)
    if ok.sum() < 50:
        return {"mz_a": np.nan, "mz_b": np.nan, "mz_r2": np.nan, "mz_p_unbiased": np.nan}
    X = sm.add_constant(f[ok])
    res = sm.OLS(s[ok], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    diff = np.asarray(res.params) - np.array([0.0, 1.0])  # H0: a = 0, b = 1
    stat = float(diff @ np.linalg.solve(np.asarray(res.cov_params()), diff))
    from scipy.stats import chi2

    return {"mz_a": float(res.params[0]), "mz_b": float(res.params[1]), "mz_r2": float(res.rsquared),
            "mz_p_unbiased": float(chi2.sf(stat, 2))}


def garman_klass(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = (h > 0) & (l > 0) & (o > 0) & (c > 0) & (h >= np.maximum(o, c)) & (l <= np.minimum(o, c)) & (h >= l)
        v = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    return np.where(ok & (v >= 0), v, np.nan)


def window_proxy_rows(daily: np.ndarray, traded: np.ndarray, a: int, b: int) -> np.ndarray:
    """Sum of a daily proxy over trading days a..b ahead of each trading row (NaN if incomplete)."""
    pos = np.flatnonzero(traded)
    x = daily[pos]
    n = len(x)
    out = np.full(len(daily), np.nan)
    finite = np.isfinite(x)
    c = np.concatenate([[0.0], np.cumsum(np.where(finite, x, 0.0))])
    cf = np.concatenate([[0], np.cumsum(finite)])
    i = np.arange(n)
    lo, hi = i + a, i + b
    ok = hi < n
    full = np.zeros(n, dtype=bool)
    full[ok] = (cf[hi[ok] + 1] - cf[lo[ok]]) == (b - a + 1)
    vals = np.full(n, np.nan)
    vals[full] = c[hi[full] + 1] - c[lo[full]]
    out[pos] = vals
    return out
