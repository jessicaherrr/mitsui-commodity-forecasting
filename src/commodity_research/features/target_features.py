"""Target-level features for the volatility research phase (docs/VOLATILITY_PREREGISTRATION.md).

For target j with legs (A, B) and horizon h, let l_t = p^A_t - p^B_t (p^A_t for single assets) be
its log price ratio (forward-filled at most 5 rows) and d_t = l_t - l_{t-1} its daily spread
return (NaN unless both legs traded). Everything below uses rows <= t only.

Groups
------
R   return / rank (no second moments)
      lag_label    l_t - l_{t-h}         = the most recent *observable* label of this target
                                           (MITSUI delivers label(t-1-h) at date t; verified on
                                           the competition's lagged-label files)
      lag_label_2  l_{t-h} - l_{t-2h}     previous non-overlapping observable label
      sret_k       l_t - l_{t-k}, k in {1, 5, 21, 63}
      madev_20     l_t - mean_20(l)
      rank_*       within-horizon cross-sectional percentile rank (minus 0.5) of lag_label and
                   sret_{5,21,63}  ("historical rank features" used by competition entrants)
Vn  volatility-normalised signals
      z_lag_label, z_sret_{5,21,63} = R signal / (sigma_EWMA * sqrt(window)),
      spread_z_63 = (l_t - mean_63(l)) / sd_63(l),  garch_shock = d_t / sigma_{t|t-1} (GARCH-t),
      rank_z_sret_21
Vs  volatility state
      log_garch_lw   0.5 log of the GARCH-t forecast of the label-window variance (k = 2..h+1)
      log_ewma_vol   log sigma_EWMA of d
      log_rv_ratio   log(realised vol 5d / 63d)
      garch_term     log( (cum21/21) / k1 ): expected mean reversion of variance
      garch_innov    log(k1_t / k1_{t-1}): variance news on day t
      garch_persist  a + b in force
      vol_of_vol     sd_63 of daily changes in log sigma_EWMA
      vol_pct_own    expanding percentile of sigma_EWMA in its own history
      rank_vol       within-horizon cross-sectional rank of sigma_EWMA
      leg_vol_ratio  0.5 log(k1_A / k1_B) from asset GARCH-t (single target: vs universe median)
      mkt_vol_pct, mkt_vol_chg   market volatility state (VT percentile, 5/63 vol change)
C   cross-asset: leg A minus leg B of factor exposures, home/equity correlations, group-relative
    and factor-residual momentum; weekly factor returns; trend and cross-correlation regimes
"""

from __future__ import annotations

import bisect
import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .engineering import ewma_vol
from .panel import ModelingPanel, build_pair_features, target_level_labels

log = logging.getLogger(__name__)

C_ASSET_FIXED = ["beta_home_126", "corr_home_63", "corr_eq_63", "home_expo_5d_z", "grp_dev_mom_21", "resid_mom_21_z"]
C_DATE_FIXED = ["reg_eq_trend", "reg_xcorr_63"]


def c_feature_names(names: list[str]) -> tuple[list[str], list[str]]:
    """Cross-asset features present in an asset-level feature set, in feature-set order.

    Asset-specific: every factor exposure ``expo_*`` plus the fixed list above (leg difference is
    taken later). Date-level: every weekly factor return ``fac_*`` plus trend / cross-correlation
    regimes. Volatility-based asset features (log_vol_rel_home, xs_rank_vol) are excluded on purpose.
    """
    asset = [n for n in names if n.startswith("expo_") or n in C_ASSET_FIXED]
    date = [n for n in names if n.startswith("fac_") or n in C_DATE_FIXED]
    return asset, date
GROUPS = ("R", "Vn", "Vs", "C")


@dataclass
class TargetFeatureSet:
    names: list[str]
    groups: list[str]
    values: np.ndarray  # (T, J, F) float32
    scales: dict[str, np.ndarray]  # label/forecast scale options -> (T, J)
    labels: np.ndarray  # (T, J) target labels recomputed from prices (training only)
    spread_vol: np.ndarray  # (T, J) EWMA spread volatility (for target-volatility terciles)

    def mask(self, groups: list[str]) -> np.ndarray:
        unknown = set(groups) - set(GROUPS)
        if unknown:
            raise ValueError(f"unknown feature groups {unknown}")
        return np.array([g in groups for g in self.groups])


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(a, np.nan)
    if k < len(a):
        out[k:] = a[: len(a) - k]
    return out


def expanding_percentile_fast(x: np.ndarray, min_periods: int = 126) -> np.ndarray:
    """Share of past-and-current finite values <= current value (causal), O(T log T)."""
    out = np.full(len(x), np.nan)
    hist: list[float] = []
    for i, v in enumerate(x):
        if np.isfinite(v):
            bisect.insort(hist, v)
            if len(hist) >= min_periods:
                out[i] = bisect.bisect_right(hist, v) / len(hist)
    return out


def _xs_rank(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Within-group cross-sectional percentile rank minus 0.5 (NaN preserved)."""
    out = np.full_like(X, np.nan)
    for g in np.unique(groups):
        cols = groups == g
        out[:, cols] = pd.DataFrame(X[:, cols]).rank(axis=1, pct=True).to_numpy() - 0.5
    return out


def build_target_features(panel: ModelingPanel, spread_eng: dict, asset_eng: dict, lam: float = 0.94) -> TargetFeatureSet:
    fs = panel.fs
    T, J = panel.n_dates, len(panel.targets)
    hz = np.asarray(panel.horizons)[panel.h_idx]  # (J,)
    hgrp = panel.h_idx
    lp = fs.logp_ff[fs.assets].to_numpy()
    obs = panel.md.close[fs.assets].notna().to_numpy()
    la, lb = panel.leg_a, panel.leg_b
    L = lp[:, la] - np.where(lb[None, :] >= 0, lp[:, np.maximum(lb, 0)], 0.0)
    both = obs[:, la] & np.where(lb[None, :] >= 0, obs[:, np.maximum(lb, 0)], True)
    d = np.where(both, np.diff(L, axis=0, prepend=np.nan), np.nan)
    feats: dict[str, tuple[str, np.ndarray]] = {}

    def add(name, group, arr):
        feats[name] = (group, np.asarray(arr, dtype=np.float64))

    # ------------------------------------------------------------------ R
    lag = np.full((T, J), np.nan)
    lag2 = np.full((T, J), np.nan)
    for h in np.unique(hz):
        cols = hz == h
        lag[:, cols] = L[:, cols] - _shift(L[:, cols], h)
        lag2[:, cols] = _shift(L[:, cols], h) - _shift(L[:, cols], 2 * h)
    add("lag_label", "R", lag)
    add("lag_label_2", "R", lag2)
    sret = {k: L - _shift(L, k) for k in (1, 5, 21, 63)}
    for k, v in sret.items():
        add(f"sret_{k}", "R", v)
    Ldf = pd.DataFrame(L)
    add("madev_20", "R", L - Ldf.rolling(20, min_periods=10).mean().to_numpy())
    add("rank_lag_label", "R", _xs_rank(lag, hgrp))
    for k in (5, 21, 63):
        add(f"rank_sret_{k}", "R", _xs_rank(sret[k], hgrp))

    # ------------------------------------------------------------------ Vn
    sig = ewma_vol(pd.DataFrame(d), lam).to_numpy()  # causal EWMA vol of spread returns
    add("z_lag_label", "Vn", lag / (sig * np.sqrt(hz)[None, :]))
    for k in (5, 21, 63):
        add(f"z_sret_{k}", "Vn", sret[k] / (sig * np.sqrt(k)))
    mu63, sd63 = Ldf.rolling(63, min_periods=40).mean().to_numpy(), Ldf.rolling(63, min_periods=40).std().to_numpy()
    add("spread_z_63", "Vn", np.clip((L - mu63) / (sd63 + 1e-12), -6, 6))
    zs = spread_eng["Z"][:, :, spread_eng["models"].index("garch_t")].T.astype(float)  # (T, J)
    add("garch_shock", "Vn", np.clip(zs, -8, 8))
    add("rank_z_sret_21", "Vn", _xs_rank(sret[21] / (sig * np.sqrt(21)), hgrp))

    # ------------------------------------------------------------------ Vs
    m = spread_eng["models"].index("garch_t")
    A = {a: k for k, a in enumerate(spread_eng["aggs"])}
    Fs = spread_eng["F"][:, :, m, :].astype(float)  # (J, T, aggs)
    lw = np.stack([Fs[j, :, A[f"lw{hz[j]}"]] for j in range(J)], axis=1)  # (T, J)
    k1 = Fs[:, :, A["k1"]].T
    c21 = Fs[:, :, A["cum21"]].T
    with np.errstate(divide="ignore", invalid="ignore"):
        add("log_garch_lw", "Vs", 0.5 * np.log(lw))
        add("log_ewma_vol", "Vs", np.log(sig))
        rv5 = pd.DataFrame(d).rolling(5, min_periods=4).std().to_numpy()
        rv63 = pd.DataFrame(d).rolling(63, min_periods=40).std().to_numpy()
        add("log_rv_ratio", "Vs", np.log((rv5 + 1e-12) / (rv63 + 1e-12)))
        add("garch_term", "Vs", np.log((c21 / 21) / k1))
        lk1 = np.log(k1)
        add("garch_innov", "Vs", lk1 - _shift(lk1, 1))
        add("garch_persist", "Vs", Fs[:, :, A["phi"]].T)
        dls = pd.DataFrame(np.log(sig)).diff()
        add("vol_of_vol", "Vs", dls.rolling(63, min_periods=40).std().to_numpy())
    vp = np.column_stack([expanding_percentile_fast(sig[:, j], 126) for j in range(J)])
    add("vol_pct_own", "Vs", vp)
    add("rank_vol", "Vs", _xs_rank(sig, hgrp))
    ma = asset_eng["models"].index("garch_t")
    Aa = {a: k for k, a in enumerate(asset_eng["aggs"])}
    k1a = asset_eng["F"][:, :, ma, Aa["k1"]].T.astype(float)  # (T, N), same asset order as fs.assets
    if list(asset_eng["names"]) != list(fs.assets):
        raise ValueError("asset engine order does not match the panel")
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows before the first GARCH fit
        lka = np.log(k1a)
        med = np.nanmedian(lka, axis=1, keepdims=True)
        lvb = np.where(lb[None, :] >= 0, lka[:, np.maximum(lb, 0)], med)
        add("leg_vol_ratio", "Vs", 0.5 * (lka[:, la] - lvb))
    reg = fs.regime
    add("mkt_vol_pct", "Vs", np.repeat(reg["reg_eq_vol_pct"].to_numpy()[:, None], J, axis=1))
    add("mkt_vol_chg", "Vs", np.repeat(reg["reg_eq_vol_chg"].to_numpy()[:, None], J, axis=1))

    # ------------------------------------------------------------------ C
    idx = {n: k for k, n in enumerate(fs.names)}
    c_asset, c_date = c_feature_names(fs.names)
    for n in c_asset:
        v = fs.values[:, :, idx[n]].astype(float)
        add(f"diff_{n}", "C", v[:, la] - np.where(lb[None, :] >= 0, v[:, np.maximum(lb, 0)], 0.0))
    for n in c_date:
        add(n, "C", np.repeat(fs.values[:, 0, idx[n]].astype(float)[:, None], J, axis=1))

    names = list(feats)
    values = np.stack([np.where(np.isfinite(feats[n][1]), feats[n][1], np.nan) for n in names], axis=-1).astype(np.float32)
    _, _, tscale = build_pair_features(panel, lam)
    with np.errstate(invalid="ignore"):
        scales = {"ewma": tscale, "garch_t": np.sqrt(lw), "none": np.ones((T, J))}
    log.info("target features: %d targets x %d dates x %d features (%s)", J, T, len(names),
             {g: sum(feats[n][0] == g for n in names) for g in GROUPS})
    return TargetFeatureSet(names=names, groups=[feats[n][0] for n in names], values=values, scales=scales,
                            labels=target_level_labels(panel), spread_vol=sig)
