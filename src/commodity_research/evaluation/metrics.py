"""Finance-appropriate forecast evaluation with dependence-aware inference.

Competition metric (MITSUI)
---------------------------
For each date d, the Spearman correlation between predictions and realised labels across all
targets with a non-null label on d (computed as the Pearson correlation of average ranks);
the score is

    score = mean_d(IC_d) / std_d(IC_d)          (population std, ddof = 0)

i.e. an information ratio of the daily rank IC.  The official implementation raises an
error on dates where predictions or labels have zero dispersion.  Here such dates are
*excluded and counted* (``n_degenerate``) -- never converted to a favourable value.

Inference
---------
Daily ICs of multi-day horizons overlap (the h-day labels of consecutive dates share h-1
daily returns), so the IC series is autocorrelated.  Mean-IC t-statistics use Newey-West
HAC standard errors; confidence intervals use a circular moving-block bootstrap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata


# ------------------------------------------------------------------ daily cross-sections
def daily_rank_ic(pred: np.ndarray, label: np.ndarray, min_valid: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Per-date Spearman IC across columns.  Returns (ic (D,), n_valid (D,)).

    ``ic`` is NaN when fewer than ``min_valid`` targets are valid or either side has zero
    dispersion (degenerate date).
    """
    pred = np.asarray(pred, dtype=float)
    label = np.asarray(label, dtype=float)
    D = pred.shape[0]
    ic = np.full(D, np.nan)
    n_valid = np.zeros(D, dtype=int)
    for d in range(D):
        m = np.isfinite(label[d]) & np.isfinite(pred[d])
        n = int(m.sum())
        n_valid[d] = n
        if n < min_valid:
            continue
        rp, rl = rankdata(pred[d, m]), rankdata(label[d, m])
        sp, sl = rp.std(), rl.std()
        if sp == 0 or sl == 0:
            continue
        ic[d] = float(np.mean((rp - rp.mean()) * (rl - rl.mean())) / (sp * sl))
    return ic, n_valid


def competition_score(pred: np.ndarray, label: np.ndarray) -> float:
    ic, _ = daily_rank_ic(pred, label, min_valid=2)
    ic = ic[np.isfinite(ic)]
    if len(ic) == 0 or ic.std() == 0:
        return np.nan
    return float(ic.mean() / ic.std())


# ------------------------------------------------------------------------- inference
def newey_west_se(x: np.ndarray, lags: int | None = None) -> float:
    """HAC standard error of the sample mean (Bartlett kernel)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return np.nan
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    e = x - x.mean()
    var = e @ e / n
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1 - lag / (lags + 1)
        var += 2 * w * (e[lag:] @ e[:-lag]) / n
    return float(np.sqrt(max(var, 0.0) / n))


def block_bootstrap(x: np.ndarray, stat, block: int = 20, reps: int = 1000, seed: int = 0) -> np.ndarray:
    """Circular moving-block bootstrap distribution of ``stat(x)`` (NaNs dropped first)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2 * block:
        return np.full(reps, np.nan)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    out = np.empty(reps)
    xx = np.concatenate([x, x[:block]])
    for r in range(reps):
        starts = rng.integers(0, n, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        out[r] = stat(xx[idx])
    return out


def _ir(x: np.ndarray) -> float:
    s = x.std()
    return float(x.mean() / s) if s > 0 else np.nan


def summarize_ic(ic: np.ndarray, hac_lags: int | None = None, block: int = 20, reps: int = 1000, seed: int = 0) -> dict:
    """Summary statistics of a daily IC series."""
    ic = np.asarray(ic, dtype=float)
    valid = ic[np.isfinite(ic)]
    n = len(valid)
    out = {"n_dates": int(n), "n_degenerate": int(np.sum(~np.isfinite(ic)))}
    if n < 3:
        return out | {k: np.nan for k in ["mean_ic", "std_ic", "ic_ir", "frac_pos", "t_hac", "p_hac", "mean_ci_lo", "mean_ci_hi", "ir_ci_lo", "ir_ci_hi"]}
    se = newey_west_se(valid, hac_lags)
    t = valid.mean() / se if se and se > 0 else np.nan
    boot_mean = block_bootstrap(valid, np.mean, block, reps, seed)
    boot_ir = block_bootstrap(valid, _ir, block, reps, seed)
    return out | {
        "mean_ic": float(valid.mean()),
        "std_ic": float(valid.std()),
        "ic_ir": _ir(valid),
        "frac_pos": float(np.mean(valid > 0)),
        "t_hac": float(t),
        "p_hac": float(2 * norm.sf(abs(t))) if np.isfinite(t) else np.nan,
        "mean_ci_lo": float(np.nanpercentile(boot_mean, 2.5)),
        "mean_ci_hi": float(np.nanpercentile(boot_mean, 97.5)),
        "ir_ci_lo": float(np.nanpercentile(boot_ir, 2.5)),
        "ir_ci_hi": float(np.nanpercentile(boot_ir, 97.5)),
    }


def paired_ic_test(ic_a: np.ndarray, ic_b: np.ndarray, hac_lags: int | None = None, block: int = 20,
                   reps: int = 1000, seed: int = 0) -> dict:
    """Test mean(IC_a - IC_b) = 0 on dates where both are defined (same universe & dates)."""
    a, b = np.asarray(ic_a, float), np.asarray(ic_b, float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    if len(d) < 3:
        return {"n_dates": int(len(d)), "mean_diff": np.nan, "t_hac": np.nan, "p_hac": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    se = newey_west_se(d, hac_lags)
    t = d.mean() / se if se > 0 else np.nan
    boot = block_bootstrap(d, np.mean, block, reps, seed)
    return {
        "n_dates": int(len(d)),
        "mean_diff": float(d.mean()),
        "t_hac": float(t),
        "p_hac": float(2 * norm.sf(abs(t))) if np.isfinite(t) else np.nan,
        "ci_lo": float(np.nanpercentile(boot, 2.5)),
        "ci_hi": float(np.nanpercentile(boot, 97.5)),
    }


# ------------------------------------------------------------------ secondary metrics
def error_metrics(pred: np.ndarray, label: np.ndarray) -> dict:
    m = np.isfinite(pred) & np.isfinite(label)
    p, y = pred[m], label[m]
    nz = (p != 0) & (y != 0)
    return {
        "mae_bps": float(np.mean(np.abs(p - y)) * 1e4),
        "rmse_bps": float(np.sqrt(np.mean((p - y) ** 2)) * 1e4),
        "rmse_zero_bps": float(np.sqrt(np.mean(y**2)) * 1e4),
        "dir_acc": float(np.mean(np.sign(p[nz]) == np.sign(y[nz]))) if nz.any() else np.nan,
    }


def rank_stability(pred: np.ndarray) -> float:
    """Mean day-over-day Spearman correlation of the forecast cross-section (1 = static)."""
    vals = []
    for d in range(1, pred.shape[0]):
        m = np.isfinite(pred[d]) & np.isfinite(pred[d - 1])
        if m.sum() < 5:
            continue
        a, b = rankdata(pred[d, m]), rankdata(pred[d - 1, m])
        if a.std() == 0 or b.std() == 0:
            continue
        vals.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(vals)) if vals else np.nan


def time_series_ic_by_target(pred: np.ndarray, label: np.ndarray, min_obs: int = 60) -> np.ndarray:
    """Per-target Spearman correlation over time (J,)."""
    out = np.full(pred.shape[1], np.nan)
    for j in range(pred.shape[1]):
        m = np.isfinite(pred[:, j]) & np.isfinite(label[:, j])
        if m.sum() < min_obs:
            continue
        a, b = rankdata(pred[m, j]), rankdata(label[m, j])
        if a.std() > 0 and b.std() > 0:
            out[j] = np.corrcoef(a, b)[0, 1]
    return out


def ic_frame(ic: np.ndarray, dates: pd.Index, name: str) -> pd.Series:
    return pd.Series(ic, index=dates, name=name)
