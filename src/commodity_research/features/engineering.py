"""Causal, economically interpretable feature families.

Every feature at row t is a function of prices at rows <= t only (verified by
``tests/test_causality.py``, which perturbs future prices and checks earlier features are
bit-for-bit unchanged).  Price *levels* are never used directly: US prices are vendor
adjusted closes whose level depends on later corporate actions, whereas log-price
*differences* are invariant to proportional back-adjustment.

Feature groups (used for ablations)
-----------------------------------
own       momentum / trend / reversal and volatility / distribution of the asset itself
cross     exposures to, and recent moves of, cross-asset factors (equity, USD, metals, gold,
          energy, rates), rolling correlations and relative volatility
relative  cross-sectional (relative-value) standing of the asset versus the universe and its
          market group, and factor-residual momentum
regime    date-level market state: equity volatility percentile, volatility change, trend
          strength and cross-factor correlation (identical for all assets on a date)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.schema import MarketData

log = logging.getLogger(__name__)

EPS = 1e-12


# --------------------------------------------------------------------------- primitives
def log_prices(close: pd.DataFrame, ffill_limit: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (forward-filled log price, daily log return).

    Forward filling uses only past values (bounded by ``ffill_limit`` rows).  The daily
    return is NaN on dates the instrument did not trade; the first return after a gap spans
    the whole gap.
    """
    logp = np.log(close)
    logp_ff = logp.ffill(limit=ffill_limit)
    ret = logp_ff.diff().where(close.notna())
    return logp_ff, ret


def ewma_vol(ret: pd.DataFrame | pd.Series, lam: float = 0.94, min_periods: int = 20, floor: float = 1e-4):
    var = (ret**2).ewm(alpha=1.0 - lam, adjust=False, ignore_na=True, min_periods=min_periods).mean()
    return np.sqrt(var).ffill().clip(lower=floor)


def expanding_percentile(s: pd.Series, min_periods: int = 126) -> pd.Series:
    """Fraction of *past and current* observations <= current value (causal)."""
    vals = s.to_numpy()
    out = np.full(len(vals), np.nan)
    hist: list[float] = []
    for i, v in enumerate(vals):
        if np.isfinite(v):
            hist.append(v)
            if len(hist) >= min_periods:
                arr = np.asarray(hist)
                out[i] = np.mean(arr <= v)
    return pd.Series(out, index=s.index)


def factor_returns(ret: pd.DataFrame, factor_defs: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Daily factor return = sign-oriented mean of available constituent returns."""
    out = {}
    for name, members in factor_defs.items():
        cols = list(members)
        w = pd.Series(members, dtype=float)
        sub = ret[cols] * w
        out[name] = sub.mean(axis=1, skipna=True)
    return pd.DataFrame(out, index=ret.index)


def home_factor(md: MarketData, asset: str, factors: list[str]) -> str:
    """Economically natural reference factor for an asset (falls back to the first factor)."""
    row = md.assets.loc[asset]
    preference = {
        "base_metal_futures": "METAL",
        "precious_metal_futures": "GOLD",
        "fx": "USD",
        "us_equity_broad": "EQ",
        "us_industrial_fin": "EQ",
        "us_bond": "RATES",
        "us_energy": "ENERGY",
        "us_precious": "GOLD",
        "us_metals_materials": "METAL",
    }.get(row["asset_class"])
    if preference in factors:
        return preference
    cls_factor = f"CLS_{row['asset_class']}".upper()
    if cls_factor in factors:
        return cls_factor
    return factors[0]


# ------------------------------------------------------------------------- the panel
@dataclass
class FeatureSet:
    dates: pd.Index
    assets: list[str]
    names: list[str]
    groups: list[str]
    values: np.ndarray  # (T, N, F) float32, NaN allowed
    vol: np.ndarray  # (T, N) causal EWMA daily volatility
    active: np.ndarray  # (T, N) bool: asset has a usable, recent price at t
    regime: pd.DataFrame  # (T x k) date-level regime variables and discrete labels
    logp_ff: pd.DataFrame  # forward-filled log prices (all assets), for pair features
    returns: pd.DataFrame  # daily log returns (all assets)

    def group_mask(self, groups: list[str]) -> np.ndarray:
        return np.array([g in groups for g in self.groups])

    def frame(self, t: int) -> pd.DataFrame:
        """Feature cross-section at row t (debug helper)."""
        return pd.DataFrame(self.values[t], index=self.assets, columns=self.names)


def build_features(md: MarketData, cfg: dict | None = None) -> FeatureSet:
    cfg = cfg or {}
    ffill_limit = int(cfg.get("ffill_limit", 5))
    lam = float(cfg.get("ewma_lambda", 0.94))
    universe = md.universe

    logp_ff, ret = log_prices(md.close, ffill_limit)
    sigma_all = ewma_vol(ret, lam)
    feats: dict[str, tuple[str, pd.DataFrame]] = {}

    def add(name: str, group: str, frame: pd.DataFrame) -> None:
        feats[name] = (group, frame.reindex(columns=universe).replace([np.inf, -np.inf], np.nan))

    lp, r, sig = logp_ff[universe], ret[universe], sigma_all[universe]
    sig_prev = sig.shift(1)

    # ---------------------------------------------------------------- A. momentum / trend
    add("ret_1_z", "own", r / sig_prev)
    add("ret_1_lag1_z", "own", (r / sig_prev).shift(1))
    for k in (5, 21, 63):
        add(f"mom_{k}_z", "own", (lp - lp.shift(k)) / (sig * np.sqrt(k)))
    add("mom_126_21_z", "own", (lp.shift(21) - lp.shift(126)) / (sig * np.sqrt(105)))
    for k in (20, 60):
        add(f"madev_{k}_z", "own", (lp - lp.rolling(k, min_periods=k // 2).mean()) / (sig * np.sqrt(k)))
    abs_path = r.abs().rolling(63, min_periods=40).sum()
    add("trend_eff_63", "own", (lp - lp.shift(63)) / (abs_path + EPS))

    # --------------------------------------------------------- B. volatility / distribution
    vol21 = r.rolling(21, min_periods=15).std()
    vol63 = r.rolling(63, min_periods=40).std()
    add("log_vol_ewma", "own", np.log(sig))
    add("log_vol_ratio_21_63", "own", np.log((vol21 + EPS) / (vol63 + EPS)))
    downside = np.sqrt((r.clip(upper=0) ** 2).rolling(63, min_periods=40).mean())
    add("log_downside_ratio_63", "own", np.log((downside + EPS) / (vol63 + EPS)))
    add("skew_63", "own", r.rolling(63, min_periods=40).skew())
    add("exkurt_63", "own", r.rolling(63, min_periods=40).kurt().clip(-5, 30))

    # -------------------------------------------------------------------- C. cross-asset
    fac = factor_returns(ret, md.factor_defs)
    fac_names = list(fac.columns)
    fac_sig = ewma_vol(fac, lam)
    fac_lp = fac.fillna(0.0).cumsum()
    for name in fac_names:
        beta = _rolling_beta_frame(r, _broadcast(fac[name], universe), 126, 60)
        add(f"expo_{name}_1d", "cross", beta * fac[name].to_numpy()[:, None] / sig.to_numpy())
    # Home-factor relationships
    home = {a: home_factor(md, a, fac_names) for a in universe}
    home_ret = pd.DataFrame({a: fac[home[a]] for a in universe}, index=fac.index)
    home_sig = pd.DataFrame({a: fac_sig[home[a]] for a in universe}, index=fac.index)
    home_lp = pd.DataFrame({a: fac_lp[home[a]] for a in universe}, index=fac.index)
    beta_home = _rolling_beta_frame(r, home_ret, 126, 60)
    corr_home = _rolling_corr_frame(r, home_ret, 63, 40)
    eq = fac["EQ"] if "EQ" in fac else fac.iloc[:, 0]
    add("beta_home_126", "cross", beta_home)
    add("corr_home_63", "cross", corr_home)
    add("corr_eq_63", "cross", _rolling_corr_frame(r, _broadcast(eq, universe), 63, 40))
    add("home_expo_5d_z", "cross", beta_home * (home_lp - home_lp.shift(5)) / (sig * np.sqrt(5)))
    add("log_vol_rel_home", "cross", np.log(sig / (home_sig + EPS)))
    for name in [n for n in ("EQ", "USD", "METAL") if n in fac_names]:
        z = (fac_lp[name] - fac_lp[name].shift(5)) / (fac_sig[name] * np.sqrt(5))
        add(f"fac_{name}_5d_z", "cross", _broadcast(z, universe))

    # ------------------------------------------------------------------- D. relative value
    mom = {k: feats[f"mom_{k}_z"][1] for k in (5, 21, 63)}
    for k, m in mom.items():
        add(f"xs_rank_mom_{k}", "relative", m.rank(axis=1, pct=True) - 0.5)
    groups = md.assets.loc[universe, "group"]
    m21 = mom[21]
    grp_mean = pd.concat({g: m21[cols].mean(axis=1) for g, cols in groups.groupby(groups).groups.items()}, axis=1)
    add("grp_dev_mom_21", "relative", m21 - pd.DataFrame({a: grp_mean[groups[a]] for a in universe}, index=m21.index))
    resid_mom = ((lp - lp.shift(21)) - beta_home * (home_lp - home_lp.shift(21))) / (sig * np.sqrt(21))
    add("resid_mom_21_z", "relative", resid_mom)
    add("xs_rank_vol", "relative", sig.rank(axis=1, pct=True) - 0.5)

    # -------------------------------------------------------------------------- E. regime
    regime = regime_frame(fac, fac_sig)
    for col in ("reg_eq_vol_pct", "reg_eq_vol_chg", "reg_eq_trend", "reg_xcorr_63"):
        add(col, "regime", _broadcast(regime[col], universe))

    names = list(feats)
    values = np.stack([feats[n][1].to_numpy(dtype=np.float32) for n in names], axis=-1)
    observed = md.close[universe].notna()
    recent = observed.rolling(ffill_limit + 1, min_periods=1).max().astype(bool)
    history = r.notna().cumsum() >= 63
    active = (recent & history).to_numpy()
    log.info("built %d features for %d assets x %d dates", len(names), len(universe), len(md.close))
    return FeatureSet(
        dates=md.close.index,
        assets=universe,
        names=names,
        groups=[feats[n][0] for n in names],
        values=values,
        vol=sig.to_numpy(dtype=np.float64),
        active=active,
        regime=regime,
        logp_ff=logp_ff,
        returns=ret,
    )


def regime_frame(fac: pd.DataFrame, fac_sig: pd.DataFrame) -> pd.DataFrame:
    """Date-level, causally observable market-state variables and discrete regime labels."""
    eq = fac["EQ"] if "EQ" in fac else fac.iloc[:, 0]
    vol21 = eq.rolling(21, min_periods=15).std()
    vol5 = eq.rolling(5, min_periods=4).std()
    vol63 = eq.rolling(63, min_periods=40).std()
    eq_lp = eq.fillna(0.0).cumsum()
    out = pd.DataFrame(index=fac.index)
    out["eq_vol_21_ann"] = vol21 * np.sqrt(252)
    out["reg_eq_vol_pct"] = expanding_percentile(vol21, min_periods=126)
    out["reg_eq_vol_chg"] = np.log((vol5 + EPS) / (vol63 + EPS))
    out["reg_eq_trend"] = (eq_lp - eq_lp.shift(63)).abs() / (vol63 * np.sqrt(63) + EPS)
    corr = fac.rolling(63, min_periods=40).corr()
    k = fac.shape[1]
    off = ~np.eye(k, dtype=bool)
    xcorr = corr.groupby(level=0).apply(lambda m: np.nanmean(np.abs(m.to_numpy()[off])) if m.notna().any().any() else np.nan)
    out["reg_xcorr_63"] = xcorr.reindex(fac.index)
    pct = out["reg_eq_vol_pct"]
    out["vol_regime"] = pd.cut(pct, [-np.inf, 1 / 3, 2 / 3, np.inf], labels=["low", "mid", "high"]).astype(object)
    out["trend_regime"] = np.where(out["reg_eq_trend"].isna(), None, np.where(out["reg_eq_trend"] > 1.0, "trend", "range"))
    xc_med = out["reg_xcorr_63"].expanding(min_periods=126).median()
    out["corr_regime"] = np.where(xc_med.isna(), None, np.where(out["reg_xcorr_63"] > xc_med, "high_corr", "low_corr"))
    return out


# ---------------------------------------------------------------------------- helpers
def _broadcast(s: pd.Series, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(np.repeat(s.to_numpy()[:, None], len(columns), axis=1), index=s.index, columns=columns)


def _rolling_moments(x: pd.DataFrame, y: pd.DataFrame, window: int, min_periods: int):
    mask = x.notna() & y.notna()
    xv, yv = x.where(mask), y.where(mask)
    roll = lambda d: d.rolling(window, min_periods=min_periods)
    n = roll(mask.astype(float)).sum()
    sx, sy = roll(xv).sum(), roll(yv).sum()
    cxy = (roll(xv * yv).sum() - sx * sy / n) / (n - 1)
    vx = (roll(xv * xv).sum() - sx * sx / n) / (n - 1)
    vy = (roll(yv * yv).sum() - sy * sy / n) / (n - 1)
    valid = n >= min_periods
    return cxy.where(valid), vx.where(valid), vy.where(valid)


def _rolling_beta_frame(x, y, window, min_periods):
    cxy, vx, vy = _rolling_moments(x, y, window, min_periods)
    return (cxy / (vy + EPS)).clip(-5, 5)


def _rolling_corr_frame(x, y, window, min_periods):
    cxy, vx, vy = _rolling_moments(x, y, window, min_periods)
    return (cxy / np.sqrt(vx * vy + EPS)).clip(-1, 1)
