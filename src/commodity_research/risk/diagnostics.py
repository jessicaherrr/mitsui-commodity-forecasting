"""Risk-aware forecast diagnostics.

Three separate questions, kept separate on purpose:

1. *Market volatility* (how much prices move) -- measured by the causal EWMA volatility.
2. *Predictive uncertainty* (how wrong the forecast is likely to be) -- measured by
   out-of-sample residuals; prediction intervals are calibrated on development-period
   residuals (split-conformal style) and their coverage is checked on the holdout.
3. *Model disagreement* (how much ensemble members differ) -- NOT a confidence interval; we
   only test whether it is associated with larger realised errors.

The hypothetical backtest converts single-asset horizon-1 forecasts into a daily-rebalanced,
dollar-neutral portfolio. It is a signal diagnostic, not evidence of tradable profitability:
it ignores financing, borrow, FX conversion of non-USD futures, market impact, contract
rolls and the different closing times of LME, JPX, US and FX markets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


# ------------------------------------------------------------------ prediction intervals
def interval_coverage(labels: np.ndarray, pred: np.ndarray, scale: np.ndarray, active: np.ndarray,
                      calib_dates: np.ndarray, test_dates: np.ndarray, levels=(0.8, 0.9),
                      groups: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """Coverage of vol-scaled vs constant-width intervals for asset-level forecasts.

    labels/pred/scale: (T, N, H).  ``groups`` maps a label to a boolean date mask (e.g. volatility
    regimes) over which holdout coverage is broken down.
    """
    rows = []
    H = labels.shape[2]
    for k in range(H):
        def resid(dates, scaled):
            y, p, s = labels[dates, :, k], pred[dates, :, k], scale[dates, :, k]
            ok = np.isfinite(y) & np.isfinite(p) & active[dates]
            e = np.abs(y - p) / (s if scaled else 1.0)
            return np.where(ok, e, np.nan)

        cal_s, cal_r = resid(calib_dates, True), resid(calib_dates, False)
        for level in levels:
            q_s, q_r = np.nanquantile(cal_s, level), np.nanquantile(cal_r, level)
            test_sets = {"all": np.ones(len(test_dates), bool)}
            if groups:
                test_sets.update({g: m[test_dates] for g, m in groups.items()})
            for g, m in test_sets.items():
                d = test_dates[m]
                if len(d) == 0:
                    continue
                es, er = resid(d, True), resid(d, False)
                n = np.isfinite(es).sum()
                rows.append({
                    "horizon_idx": k, "level": level, "subset": g, "n_obs": int(n),
                    "coverage_volscaled": float(np.nanmean(es <= q_s)),
                    "coverage_constant": float(np.nanmean(er <= q_r)),
                    "avg_width_volscaled_bps": float(2 * q_s * np.nanmean(np.where(np.isfinite(es), scale[d, :, k], np.nan)) * 1e4),
                    "width_constant_bps": float(2 * q_r * 1e4),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- disagreement
def disagreement_analysis(disagreement: np.ndarray, abs_err_z: np.ndarray, target_scale: np.ndarray,
                          daily_ic: np.ndarray, dates: np.ndarray, n_bins: int = 5) -> dict:
    """Is cross-model disagreement associated with larger (vol-scaled) realised errors?"""
    xs_corr, xs_corr_vol = [], []
    for d in dates:
        dis, err, sc = disagreement[d], abs_err_z[d], target_scale[d]
        m = np.isfinite(dis) & np.isfinite(err) & np.isfinite(sc)
        if m.sum() < 20 or np.std(dis[m]) == 0:
            continue
        xs_corr.append(spearmanr(dis[m], err[m]).statistic)
        xs_corr_vol.append(spearmanr(dis[m], sc[m]).statistic)
    D = disagreement[dates].ravel()
    E = abs_err_z[dates].ravel()
    ok = np.isfinite(D) & np.isfinite(E)
    q = pd.qcut(rankdata(D[ok]), n_bins, labels=False)
    by_bin = pd.Series(E[ok]).groupby(q).mean()
    mean_dis = np.nanmean(disagreement[dates], axis=1)
    ic = daily_ic[dates]
    okd = np.isfinite(mean_dis) & np.isfinite(ic)
    return {
        "mean_xs_spearman_disagreement_vs_abs_error_z": float(np.nanmean(xs_corr)) if xs_corr else np.nan,
        "frac_dates_positive": float(np.mean(np.array(xs_corr) > 0)) if xs_corr else np.nan,
        "mean_xs_spearman_disagreement_vs_volatility": float(np.nanmean(xs_corr_vol)) if xs_corr_vol else np.nan,
        "abs_error_z_by_disagreement_quintile": [float(v) for v in by_bin.to_numpy()],
        "ts_spearman_mean_disagreement_vs_daily_ic": float(spearmanr(mean_dis[okd], ic[okd]).statistic) if okd.sum() > 10 else np.nan,
        "n_dates": int(len(xs_corr)),
    }


# ------------------------------------------------------------------------- backtest
def long_short_backtest(pred_h1: np.ndarray, ret_h1: np.ndarray, vol: np.ndarray, active: np.ndarray,
                        dates: np.ndarray, costs_bps=(0, 2, 5, 10), vol_scaled: bool = True,
                        periods_per_year: float = 252.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily dollar-neutral portfolio from horizon-1 asset forecasts.

    Timing: forecast formed at close t; position entered at close t+1 and held to close t+2,
    i.e. the portfolio earns exactly the horizon-1 label r_{i}(t) = log(P_{t+2}/P_{t+1}).
    Weights: s_i = rhat_i / sigma_i (risk-adjusted) or rhat_i (raw), cross-sectionally
    demeaned and scaled to unit gross exposure.  An asset with a missing label (no trade on
    t+1 or t+2) contributes zero return.  Costs: c * sum_i |w_{i,t} - w_{i,t-1}| per day.
    """
    W = np.zeros((len(dates), pred_h1.shape[1]))
    gross = np.zeros(len(dates))
    for n, t in enumerate(dates):
        p = pred_h1[t]
        ok = np.isfinite(p) & active[t]
        s = np.where(ok, p / vol[t] if vol_scaled else p, 0.0)
        s = np.where(ok, s - s[ok].mean() if ok.any() else 0.0, 0.0)
        tot = np.abs(s).sum()
        if tot > 0:
            W[n] = s / tot
        r = np.nan_to_num(ret_h1[t])
        gross[n] = W[n] @ r
    turnover = np.abs(np.diff(W, axis=0, prepend=np.zeros((1, W.shape[1])))).sum(axis=1)
    turnover[0] = np.abs(W[0]).sum()
    daily = pd.DataFrame({"gross": gross, "turnover": turnover}, index=dates)
    rows = []
    for c in costs_bps:
        net = gross - turnover * c / 1e4
        daily[f"net_{c}bps"] = net
        ann_ret = net.mean() * periods_per_year
        ann_vol = net.std() * np.sqrt(periods_per_year)
        curve = np.cumsum(net)
        mdd = float(np.max(np.maximum.accumulate(curve) - curve)) if len(curve) else np.nan
        rows.append({
            "cost_bps": c, "ann_return": float(ann_ret), "ann_vol": float(ann_vol),
            "sharpe": float(ann_ret / ann_vol) if ann_vol > 0 else np.nan,
            "max_drawdown": mdd, "avg_daily_turnover": float(turnover.mean()), "n_days": int(len(net)),
        })
    return pd.DataFrame(rows), daily
