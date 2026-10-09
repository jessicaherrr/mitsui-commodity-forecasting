"""Financial time-series diagnostics.

Stylised-fact tests on a daily log-return series r_t (non-trading days removed):

* Jarque-Bera:  JB = n/6 (S^2 + K^2/4) ~ chi2(2) under normality (S skewness, K excess kurtosis)
* Ljung-Box:    Q(m) = n(n+2) sum_{k<=m} rho_k^2 / (n-k) ~ chi2(m - model_df), on r_t (linear
                predictability) and on r_t^2 (volatility clustering)
* ARCH-LM:      n R^2 of r_t^2 on r_{t-1}^2..r_{t-q}^2 ~ chi2(q)                     (Engle, 1982)
* Engle-Ng sign/size-bias test on standardised residuals z_t of a symmetric GARCH:
                z_t^2 = c0 + c1 S-_{t-1} + c2 S-_{t-1} z_{t-1} + c3 S+_{t-1} z_{t-1} + u_t
                with S- = 1[z<0], S+ = 1 - S-. H0: c1=c2=c3=0 (no neglected asymmetry).
* GARCH residual diagnostics: Ljung-Box on z and z^2 (model_df = 2 for the two dynamic GARCH
  parameters, Li & Mak 1994), ARCH-LM on z, Jarque-Bera on z, and a likelihood-ratio test of
  GJR-GARCH vs GARCH (nested, 1 restriction).

All p-values are per series; across many series use ``bh_fdr`` (Benjamini-Hochberg).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats


def _clean(r) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    return r[np.isfinite(r)]


def ljung_box(x: np.ndarray, lags: int, model_df: int = 0) -> tuple[float, float]:
    from statsmodels.stats.diagnostic import acorr_ljungbox

    res = acorr_ljungbox(x, lags=[lags], model_df=model_df, return_df=True)
    return float(res["lb_stat"].iloc[0]), float(res["lb_pvalue"].iloc[0])


def arch_lm(x: np.ndarray, lags: int = 5) -> tuple[float, float]:
    from statsmodels.stats.diagnostic import het_arch

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        lm, p, _, _ = het_arch(x - x.mean(), nlags=lags)
    return float(lm), float(p)


def stylised_facts(r, lags=(5, 10, 21), arch_lags: int = 5) -> dict:
    x = _clean(r)
    n = len(x)
    jb, jbp = stats.jarque_bera(x)
    out = {
        "n": n, "mean_ann": float(x.mean() * 252), "vol_ann": float(x.std(ddof=1) * np.sqrt(252)),
        "skew": float(stats.skew(x)), "exkurt": float(stats.kurtosis(x)), "jb": float(jb), "jb_p": float(jbp),
    }
    for m in lags:
        out[f"lb_r_{m}"], out[f"lb_r_{m}_p"] = ljung_box(x, m)
        out[f"lb_r2_{m}"], out[f"lb_r2_{m}_p"] = ljung_box(x**2, m)
    out["arch_lm"], out["arch_lm_p"] = arch_lm(x, arch_lags)
    from statsmodels.tsa.stattools import acf

    a1 = acf(x, nlags=5, fft=True)
    a2 = acf(x**2, nlags=5, fft=True)
    for k in range(1, 6):
        out[f"acf_r_{k}"], out[f"acf_r2_{k}"] = float(a1[k]), float(a2[k])
    return out


def acf_pacf(x, nlags: int = 21) -> dict:
    from statsmodels.tsa.stattools import acf, pacf

    x = _clean(x)
    return {"acf": acf(x, nlags=nlags, fft=True), "pacf": pacf(x, nlags=nlags, method="ywm"), "n": len(x)}


def sign_bias_test(z) -> dict:
    """Engle & Ng (1993) sign, negative-size and positive-size bias test (HC1 covariance)."""
    import statsmodels.api as sm

    z = _clean(z)
    y = z[1:] ** 2
    zl = z[:-1]
    sm_ = (zl < 0).astype(float)
    X = np.column_stack([np.ones_like(zl), sm_, sm_ * zl, (1 - sm_) * zl])
    res = sm.OLS(y, X).fit(cov_type="HC1")
    R = np.eye(4)[1:]
    w = res.wald_test(R, scalar=True)
    return {
        "sign_t": float(res.tvalues[1]), "neg_size_t": float(res.tvalues[2]), "pos_size_t": float(res.tvalues[3]),
        "joint_stat": float(w.statistic), "joint_p": float(w.pvalue),
    }


def fit_full(x_pct: np.ndarray, spec: dict):
    from arch import arch_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return arch_model(x_pct, mean="Zero", **spec).fit(disp="off", show_warning=False)


def garch_residual_diagnostics(r, lags: int = 10) -> dict:
    """In-sample GARCH(1,1)-t and GJR-GARCH(1,1)-t fits on one series (percentage returns)."""
    from .models import GARCH_SPECS

    x = _clean(r) * 100
    g = fit_full(x, GARCH_SPECS["garch_t"])
    gj = fit_full(x, GARCH_SPECS["gjr_t"])
    gn = fit_full(x, GARCH_SPECS["garch_n"])
    z = np.asarray(g.std_resid)
    z = z[np.isfinite(z)]
    p = g.params
    out = {
        "omega": float(p["omega"]), "alpha": float(p["alpha[1]"]), "beta": float(p["beta[1]"]), "nu": float(p["nu"]),
        "persistence": float(p["alpha[1]"] + p["beta[1]"]),
        "half_life_days": float(np.log(0.5) / np.log(min(p["alpha[1]"] + p["beta[1]"], 0.9999))),
        "gjr_gamma": float(gj.params["gamma[1]"]),
        "lr_gjr_vs_garch": float(2 * (gj.loglikelihood - g.loglikelihood)),
        "lr_t_vs_normal_loglik_gain": float(g.loglikelihood - gn.loglikelihood),
        "garch_n_alpha": float(gn.params["alpha[1]"]), "garch_n_beta": float(gn.params["beta[1]"]),
        "bic_garch_t": float(g.bic), "bic_gjr_t": float(gj.bic), "bic_garch_n": float(gn.bic),
    }
    out["lr_gjr_p"] = float(stats.chi2.sf(max(out["lr_gjr_vs_garch"], 0.0), 1))
    out["lb_z"], out["lb_z_p"] = ljung_box(z, lags)
    out["lb_z2"], out["lb_z2_p"] = ljung_box(z**2, lags, model_df=2)
    out["arch_lm_z"], out["arch_lm_z_p"] = arch_lm(z, 5)
    jb, jbp = stats.jarque_bera(z)
    out["jb_z"], out["jb_z_p"] = float(jb), float(jbp)
    out["skew_z"], out["exkurt_z"] = float(stats.skew(z)), float(stats.kurtosis(z))
    # Kolmogorov-Smirnov distance of z to the fitted standardised Student-t and to N(0,1)
    nu = out["nu"]
    scale = np.sqrt((nu - 2) / nu) if nu > 2 else 1.0
    out["ks_t"] = float(stats.kstest(z, lambda v: stats.t.cdf(v / scale, nu)).statistic)
    out["ks_normal"] = float(stats.kstest(z, "norm").statistic)
    out.update({f"sb_{k}": v for k, v in sign_bias_test(z).items()})
    return out | {"_z": z}


def bh_fdr(p: np.ndarray, q: float = 0.10) -> np.ndarray:
    """Benjamini-Hochberg rejections at FDR level q (NaN p-values never rejected)."""
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    rej = np.zeros(len(p), dtype=bool)
    if ok.sum() == 0:
        return rej
    ps = p[ok]
    order = np.argsort(ps)
    m = len(ps)
    thresh = q * np.arange(1, m + 1) / m
    passed = ps[order] <= thresh
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    r = np.zeros(m, dtype=bool)
    r[order[:k]] = True
    rej[np.flatnonzero(ok)] = r
    return rej


def qq_points(z: np.ndarray, dist: str = "normal", nu: float | None = None, n_points: int = 400) -> tuple[np.ndarray, np.ndarray]:
    z = np.sort(_clean(z))
    probs = (np.arange(1, len(z) + 1) - 0.5) / len(z)
    if dist == "normal":
        theo = stats.norm.ppf(probs)
    else:
        scale = np.sqrt((nu - 2) / nu)
        theo = stats.t.ppf(probs, nu) * scale
    idx = np.unique(np.linspace(0, len(z) - 1, n_points).astype(int))
    return theo[idx], z[idx]


def summarise_panel(df: pd.DataFrame, tests: dict[str, str], q: float = 0.10) -> pd.DataFrame:
    """Share of series rejecting each test at 5% (raw) and after BH-FDR at level q."""
    rows = []
    for label, col in tests.items():
        p = df[col].to_numpy(dtype=float)
        rows.append({"test": label, "n_series": int(np.isfinite(p).sum()), "share_p<0.05": float(np.nanmean(p < 0.05)),
                     f"share_BH_q{q}": float(bh_fdr(p, q).mean()), "median_p": float(np.nanmedian(p))})
    return pd.DataFrame(rows)
