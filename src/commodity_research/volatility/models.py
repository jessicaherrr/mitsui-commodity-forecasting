"""Causal volatility forecasting: historical, EWMA and GARCH-family models.

All models forecast the conditional variance of *daily log returns* on an asset's own trading
days (non-trading rows are skipped, so "k days ahead" means k trading days):

    f_k(i) = E[ r_{i+k}^2 | r_1..r_i ]           (zero-mean convention, decimal units)

Models
------
const     expanding mean of r^2                                   (no dynamics; benchmark)
hv21      mean of the last 21 r^2                                  (rolling historical vol)
hv63      mean of the last 63 r^2
ewma      s_{i+1} = lam s_i + (1 - lam) r_i^2, lam = 0.94           (RiskMetrics)
garch_n   GARCH(1,1), Gaussian QML:   s_{i+1} = w + a r_i^2 + b s_i
garch_t   GARCH(1,1), Student-t ML (fat-tailed innovations)
gjr_t     GJR-GARCH(1,1)-t: s_{i+1} = w + (a + g 1[r_i < 0]) r_i^2 + b s_i
          (asymmetric; nested in garch_t, so an LR test applies; closed-form multi-step)

Multi-step forecasts for the GARCH family use the exact recursion (symmetric innovations)

    E_i[s_{i+k+1}] = w + phi E_i[s_{i+k}],   phi = a + b + g/2

which equals sbar + phi^{k-1}(s_{i+1} - sbar) with sbar = w/(1-phi) when phi < 1, and remains
valid for integrated (phi = 1) or explosive estimates. Historical/EWMA forecasts are flat in k.

Causality
---------
GARCH parameters are re-estimated every ``refit_every`` trading days on all returns up to and
including the refit day (expanding window, at least ``min_obs`` observations). Between refits
the parameters are frozen and the variance recursion is filtered forward with realised returns,
so the forecast made on day i uses returns up to day i only.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

SCALE = 100.0  # estimate on percentage returns for numerical stability
GARCH_SPECS = {
    "garch_n": dict(vol="GARCH", p=1, o=0, q=1, dist="normal"),
    "garch_t": dict(vol="GARCH", p=1, o=0, q=1, dist="t"),
    "gjr_t": dict(vol="GARCH", p=1, o=1, q=1, dist="t"),
}
ALL_MODELS = ["const", "hv21", "hv63", "ewma", "garch_n", "garch_t", "gjr_t"]


@dataclass
class VolEngineConfig:
    models: list[str] = field(default_factory=lambda: list(ALL_MODELS))
    refit_every: int = 21
    min_obs: int = 250
    ewma_lambda: float = 0.94
    k_max: int = 21  # longest horizon (trading days) for which forecasts are produced


@dataclass
class SeriesForecast:
    """Forecasts for one series on its trading-day grid."""

    pos: np.ndarray  # (n,) row positions of the trading days in the full date index
    f: dict[str, np.ndarray]  # model -> (n, k_max) forecasts of r^2 at k = 1..k_max (decimal)
    phi: dict[str, np.ndarray]  # model -> (n,) persistence a + b + g/2 in force on each day (NaN if not GARCH)
    params: dict[str, list[dict]]  # model -> list of {"refit_pos", "omega", "alpha", "gamma", "beta", "nu", "converged"}
    resid: dict[str, np.ndarray]  # model -> (n,) standardised one-step residuals r_i / sqrt(s_i)


def _flat(f1: np.ndarray, k_max: int) -> np.ndarray:
    return np.repeat(f1[:, None], k_max, axis=1)


def _historical(x2: np.ndarray, kind: str, lam: float) -> np.ndarray:
    n = len(x2)
    f1 = np.full(n, np.nan)
    if kind == "const":
        c = np.cumsum(x2)
        idx = np.arange(1, n + 1)
        f1[19:] = (c / idx)[19:]
    elif kind.startswith("hv"):
        w = int(kind[2:])
        c = np.concatenate([[0.0], np.cumsum(x2)])
        f1[w - 1:] = (c[w:] - c[:-w]) / w
    elif kind == "ewma":
        if n >= 20:
            s = x2[:20].mean()
            f1[19] = s
            for i in range(20, n):
                s = lam * s + (1 - lam) * x2[i]
                f1[i] = s
    return f1


def _fit_garch(x: np.ndarray, spec: dict):
    from arch import arch_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(x, mean="Zero", **spec).fit(disp="off", show_warning=False)
    p = res.params
    out = {
        "omega": float(p["omega"]), "alpha": float(p["alpha[1]"]), "beta": float(p["beta[1]"]),
        "gamma": float(p.get("gamma[1]", 0.0)), "nu": float(p.get("nu", np.nan)),
        "converged": bool(res.convergence_flag == 0), "loglik": float(res.loglikelihood),
    }
    s_last = float(res.conditional_volatility[-1] ** 2)
    return out, s_last


def _garch_series(x: np.ndarray, spec: dict, cfg: VolEngineConfig):
    """One-step forecasts with periodic re-estimation; returns (f1, phi, omega, params, sigma2)."""
    n = len(x)
    f1 = np.full(n, np.nan)
    phi = np.full(n, np.nan)
    omega = np.full(n, np.nan)
    sig2 = np.full(n, np.nan)  # conditional variance of x[i] given x[..i-1]
    params = []
    prev = None
    for m in range(cfg.min_obs, n + 1, cfg.refit_every):
        try:
            par, s_last = _fit_garch(x[:m], spec)
        except Exception as e:  # pragma: no cover - optimiser failure
            log.debug("garch fit failed at %d: %s", m, e)
            if prev is None:
                continue
            par, s_last = prev[0], None
        if not np.isfinite(par["omega"]) or par["alpha"] < 0 or par["beta"] < 0:
            if prev is None:
                continue
            par = prev[0]
        params.append({"refit_obs": m, **par})
        w, a, g, b = par["omega"], par["alpha"], par["gamma"], par["beta"]
        ph = a + b + 0.5 * g
        # conditional variance of x[m-1]: from the fit, or (failed fit) the previous block's
        # one-step forecast made on day m-2
        if s_last is not None:
            s = s_last
        elif m >= 2 and np.isfinite(f1[m - 2]):
            s = f1[m - 2]
        else:
            s = float(np.var(x[:m]))
        end = min(m - 1 + cfg.refit_every, n)
        for i in range(m - 1, end):
            if i > m - 1:
                s = f1[i - 1]
            sig2[i] = s
            e2 = x[i] ** 2
            f1[i] = w + (a + (g if x[i] < 0 else 0.0)) * e2 + b * s
            phi[i], omega[i] = ph, w
        prev = (par, None)
    return f1, phi, omega, params, sig2


def forecast_series(r: np.ndarray, cfg: VolEngineConfig) -> SeriesForecast:
    """Run every configured model on one daily log-return series (NaN = no trade)."""
    pos = np.flatnonzero(np.isfinite(r))
    x = r[pos] * SCALE
    x2 = x**2
    n, K = len(x), cfg.k_max
    f, params, resid, persist = {}, {}, {}, {}
    for m in cfg.models:
        if m in GARCH_SPECS:
            if n < cfg.min_obs:
                f[m] = np.full((n, K), np.nan)
                params[m], resid[m], persist[m] = [], np.full(n, np.nan), np.full(n, np.nan)
                continue
            f1, phi, om, par, sig2 = _garch_series(x, GARCH_SPECS[m], cfg)
            fk = np.empty((n, K))
            fk[:, 0] = f1
            for k in range(1, K):
                fk[:, k] = om + phi * fk[:, k - 1]
            f[m] = fk / SCALE**2
            params[m] = par
            resid[m] = x / np.sqrt(sig2)
            persist[m] = phi
        else:
            f1 = _historical(x2, m, cfg.ewma_lambda)
            f[m] = _flat(f1, K) / SCALE**2
            params[m] = []
            prev = np.concatenate([[np.nan], f1[:-1]])
            resid[m] = x / np.sqrt(prev)
            persist[m] = np.full(n, np.nan)
    return SeriesForecast(pos=pos, f=f, phi=persist, params=params, resid=resid)


def to_rows(sf: SeriesForecast, n_rows: int, arr: np.ndarray, ffill_limit: int = 5) -> np.ndarray:
    """Map a trading-day array (n, ...) to the full row index; non-trading rows carry the last
    forecast forward for at most ``ffill_limit`` rows (information is unchanged on those rows)."""
    out = np.full((n_rows,) + arr.shape[1:], np.nan)
    out[sf.pos] = arr
    last = -10**9
    j = -1
    pos_set = np.zeros(n_rows, dtype=bool)
    pos_set[sf.pos] = True
    src = np.full(n_rows, -1)
    for t in range(n_rows):
        if pos_set[t]:
            j += 1
            last = t
        if j >= 0 and t - last <= ffill_limit:
            src[t] = j
    ok = src >= 0
    out[ok] = arr[src[ok]]
    return out


def window_sum(fk: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Sum of forecasts over k = start..stop (1-based, inclusive) -> forecast variance of the window."""
    return fk[:, start - 1:stop].sum(axis=1)
