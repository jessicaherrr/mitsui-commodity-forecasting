"""Volatility research pipeline: diagnostics (Part I) and volatility-forecast evaluation (Part II).

Return-forecasting experiments with volatility features (Part III) run through the standard
``train/evaluate/report`` commands with ``configs/vol.yaml``; the volatility engine output they
need is precomputed and cached here.

Sample discipline: every statistic that informs a modelling decision (e.g. whether to include an
asymmetric GARCH) is computed on the *estimation sample* (rows < holdout_start). Holdout-period
statistics are reported separately as confirmation only.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import results_dir
from ..volatility import diagnostics as D
from ..volatility.engine import load_or_run, series_from_panel
from ..volatility.models import VolEngineConfig
from .runner import data_fingerprint, prepare, resolve_split

log = logging.getLogger(__name__)

# Representative series chosen by economic role (not by test outcome).
REPRESENTATIVE = [
    "LME_CA_Close", "LME_AH_Close", "LME_ZS_Close", "LME_PB_Close",
    "JPX_Gold_Standard_Futures_Close", "JPX_Platinum_Standard_Futures_Close",
    "US_Stock_GLD_adj_close", "US_Stock_XLE_adj_close", "US_Stock_FCX_adj_close", "US_Stock_VT_adj_close",
    "FX_USDJPY", "FX_AUDUSD", "FX_ZARUSD",
]


def vol_dir(cfg: dict) -> Path:
    d = results_dir(cfg) / "vol"
    d.mkdir(parents=True, exist_ok=True)
    return d


def engine_config(cfg: dict) -> VolEngineConfig:
    v = cfg.get("volatility", {})
    return VolEngineConfig(models=v.get("models", ["const", "hv21", "hv63", "ewma", "garch_n", "garch_t", "gjr_t"]),
                           refit_every=int(v.get("refit_every", 21)), min_obs=int(v.get("min_obs", 250)),
                           ewma_lambda=float(cfg.get("features", {}).get("ewma_lambda", 0.94)))


# ------------------------------------------------------------------------- Part I
def run_diagnostics(cfg: dict) -> dict:
    t0 = time.time()
    md, fs, panel = prepare(cfg)
    split = resolve_split(cfg, panel)
    out = vol_dir(cfg)
    hs = split["holdout_start"]
    assets, spreads = series_from_panel(panel)
    meta = md.assets

    def facts(series: dict, rows: slice, with_garch: bool, min_obs: int = 250):
        recs = []
        for name, r in series.items():
            x = r[rows]
            if np.isfinite(x).sum() < min_obs:
                continue
            rec = {"series": name, **D.stylised_facts(x)}
            if with_garch:
                g = D.garch_residual_diagnostics(x)
                g.pop("_z")
                rec.update(g)
            recs.append(rec)
        return pd.DataFrame(recs).set_index("series") if recs else pd.DataFrame(index=pd.Index([], name="series"))

    est = slice(0, hs)
    hold = slice(hs, split["end"])
    a_est = facts(assets, est, with_garch=True)
    a_est.insert(0, "asset_class", meta.loc[a_est.index, "asset_class"])
    a_est.insert(0, "market", meta.loc[a_est.index, "market"])
    a_est.to_csv(out / "diagnostics_assets_estimation.csv")
    a_hold = facts(assets, hold, with_garch=False, min_obs=100)  # holdouts can be short (FRED 2026: 197 rows)
    a_hold.to_csv(out / "diagnostics_assets_holdout.csv")
    s_est = facts(spreads, est, with_garch=False)
    s_est.insert(0, "horizon", panel.targets.loc[s_est.index, "horizon"])
    s_est.insert(1, "category", panel.targets.loc[s_est.index, "category"])
    s_est.to_csv(out / "diagnostics_spreads_estimation.csv")

    tests = {"Jarque-Bera (normality)": "jb_p", "Ljung-Box r, 10 lags": "lb_r_10_p", "Ljung-Box r^2, 10 lags": "lb_r2_10_p",
             "ARCH-LM, 5 lags": "arch_lm_p"}
    gtests = {"GARCH-t resid: Ljung-Box z, 10": "lb_z_p", "GARCH-t resid: Ljung-Box z^2, 10 (df-2)": "lb_z2_p",
              "GARCH-t resid: ARCH-LM z, 5": "arch_lm_z_p", "GARCH-t resid: Jarque-Bera z": "jb_z_p",
              "Engle-Ng joint sign/size bias": "sb_joint_p", "LR GJR vs GARCH (1 df)": "lr_gjr_p"}
    summ = pd.concat([
        D.summarise_panel(a_est, tests).assign(sample="assets, estimation"),
        D.summarise_panel(a_hold, tests).assign(sample="assets, holdout"),
        D.summarise_panel(s_est, tests).assign(sample="target spreads, estimation"),
        D.summarise_panel(a_est, gtests).assign(sample="assets, estimation (GARCH residuals)"),
    ])
    summ.to_csv(out / "diagnostics_summary.csv", index=False)

    # by market group (estimation sample)
    grp = a_est.groupby("market").agg(
        n=("jb_p", "size"), exkurt_median=("exkurt", "median"), skew_median=("skew", "median"),
        share_lb_r2=("lb_r2_10_p", lambda p: float((p < 0.05).mean())), share_lb_r=("lb_r_10_p", lambda p: float((p < 0.05).mean())),
        persistence_median=("persistence", "median"), half_life_median=("half_life_days", "median"),
        nu_median=("nu", "median"), gjr_gamma_median=("gjr_gamma", "median"),
        share_sign_bias=("sb_joint_p", lambda p: float((p < 0.05).mean())),
        share_lr_gjr=("lr_gjr_p", lambda p: float((p < 0.05).mean())),
        loglik_gain_t_median=("lr_t_vs_normal_loglik_gain", "median"))
    grp.to_csv(out / "diagnostics_by_market.csv")

    # sub-period stability of the two key facts (config windows, else four equal estimation chunks)
    windows = cfg.get("volatility", {}).get("subperiods") or {
        f"estimation part {k + 1}": [int(a), int(b)] for k, (a, b) in enumerate(
            zip(np.linspace(split["first_train"], hs, 5)[:-1], np.linspace(split["first_train"], hs, 5)[1:]))}
    windows = {**windows, "holdout": [hs, split["end"]]}
    sub = []
    for name, (a_, b_) in windows.items():
        p1, p2 = [], []
        for r in assets.values():
            x = r[int(a_):int(b_)]
            x = x[np.isfinite(x)]
            if len(x) < 100:
                continue
            p1.append(D.ljung_box(x, 10)[1])
            p2.append(D.ljung_box(x**2, 10)[1])
        sub.append({"window": name, "rows": f"{int(a_)}-{int(b_) - 1}", "n_series": len(p1),
                    "share_BH_lb_returns": float(D.bh_fdr(np.array(p1)).mean()),
                    "share_BH_lb_squared": float(D.bh_fdr(np.array(p2)).mean())})
    pd.DataFrame(sub).to_csv(out / "diagnostics_subperiods.csv", index=False)

    # figure data for representative series
    figdata = {}
    for name in [s for s in REPRESENTATIVE if s in assets] or list(assets)[:6]:
        x = assets[name][est]
        ap = D.acf_pacf(x, 21)
        ap2 = D.acf_pacf(np.where(np.isfinite(x), x**2, np.nan), 21)
        g = D.garch_residual_diagnostics(x)
        z = g.pop("_z")
        xs = x[np.isfinite(x)]
        qn = D.qq_points(xs / xs.std(), "normal")
        qz = D.qq_points(z, "normal")
        qt = D.qq_points(z, "t", g["nu"])
        figdata[name] = {"acf_r": ap["acf"].tolist(), "pacf_r": ap["pacf"].tolist(), "acf_r2": ap2["acf"].tolist(),
                         "pacf_r2": ap2["pacf"].tolist(), "n": ap["n"], "qq_raw_normal": [qn[0].tolist(), qn[1].tolist()],
                         "qq_z_normal": [qz[0].tolist(), qz[1].tolist()], "qq_z_t": [qt[0].tolist(), qt[1].tolist()], "nu": g["nu"]}
    (out / "diagnostics_figdata.json").write_text(json.dumps(figdata))
    log.info("diagnostics done in %.1fs -> %s", time.time() - t0, out)
    return {"summary": summ, "by_market": grp}


# ------------------------------------------------------------------------- engine
def engine(cfg: dict, kind: str, prepared=None) -> dict:
    md, fs, panel = prepared or prepare(cfg)
    n_jobs = int(cfg.get("runtime", {}).get("n_jobs", -1))
    return load_or_run(panel, engine_config(cfg), results_dir(cfg) / "cache", data_fingerprint(cfg) | {"dataset": md.name},
                       kind, n_jobs=n_jobs)


# ------------------------------------------------------------------------- Part II
# Pre-registered contrasts (docs/VOLATILITY_PREREGISTRATION.md, V1-V4): (model_a, model_b)
CONTRASTS = [
    ("V1", "hv21", "const"), ("V1", "hv63", "const"), ("V1", "ewma", "const"), ("V1", "garch_n", "const"),
    ("V1", "garch_t", "const"), ("V1", "gjr_t", "const"),
    ("V2", "garch_t", "ewma"), ("V2", "garch_t", "hv21"), ("V2", "garch_t", "hv63"),
    ("V3", "gjr_t", "garch_t"),
    ("V4", "garch_t", "garch_n"),
]
WINDOW_LAGS = {"k1": 5, "cum5": 10, "cum21": 42}
WINDOWS = {"k1": (1, 1), "cum5": (1, 5), "cum21": (1, 21)}


def load_ohlc(md, path) -> dict[str, tuple]:
    """(open, high, low, close) arrays for MITSUI assets that have them (US adjusted, JPX)."""
    from ..config import ROOT

    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    raw = pd.read_csv(path / "train.csv").set_index("date_id")
    out = {}
    for a in md.close.columns:
        if a.startswith("US_Stock_") and a.endswith("_adj_close"):
            base = a[: -len("close")]
            cols = [base + s for s in ("open", "high", "low", "close")]
        elif a.startswith("JPX_") and a.endswith("_Close"):
            base = a[: -len("Close")]
            cols = [base + s for s in ("Open", "High", "Low", "Close")]
        else:
            continue
        if all(c in raw.columns for c in cols):
            out[a] = tuple(raw[c].to_numpy(dtype=float) for c in cols)
    return out


def _asset_losses(name, r, Fa, models, aggs, ohlc, periods, regime):
    """Per-asset loss panels and per-asset statistics for every period and window."""
    from ..volatility import evaluation as E

    traded = np.isfinite(r)
    r2 = np.where(traded, r**2, np.nan)
    gk = E.garman_klass(*ohlc) if ohlc is not None else None
    recs, panel = [], {}
    for agg, (a, b) in WINDOWS.items():
        k = aggs.index(agg)
        proxies = {"sq": E.window_proxy_rows(r2, traded, a, b)}
        if gk is not None and agg in ("k1", "cum5"):
            proxies["gk"] = E.window_proxy_rows(np.where(traded, gk, np.nan), traded, a, b)
        for pname, rows in periods.items():
            sel = np.zeros(len(r), dtype=bool)
            sel[rows] = True
            sel &= traded
            for prox, s in proxies.items():
                f = Fa[:, :, k]  # (rows, models)
                ok = sel & np.isfinite(s) & np.all(np.isfinite(f) & (f > 0), axis=1)
                if ok.sum() < 60:
                    continue
                Lq = E.qlike(s[ok, None], f[ok])
                Lm = E.mse(s[ok, None], f[ok]) * 1e8  # (bp^2)^2 scale for readability
                idx = np.flatnonzero(ok)
                panel[(pname, agg, prox)] = (idx, Lq)
                rec = {"series": name, "period": pname, "window": agg, "proxy": prox, "n": int(ok.sum())}
                for j, m in enumerate(models):
                    rec[f"qlike_{m}"] = float(Lq[:, j].mean())
                    rec[f"mse_ratio_{m}"] = float(Lm[:, j].mean() / Lm[:, models.index("ewma")].mean())
                for cid, ma, mb in CONTRASTS:
                    d = Lq[:, models.index(ma)] - Lq[:, models.index(mb)]
                    mean, t, p, _ = E.dm_test(d, WINDOW_LAGS[agg])
                    rec[f"dm_{ma}_vs_{mb}_t"], rec[f"dm_{ma}_vs_{mb}_p"] = t, p
                if prox == "sq":
                    for m in ("ewma", "garch_t"):
                        mz = E.mincer_zarnowitz(s[ok], f[ok, models.index(m)], WINDOW_LAGS[agg])
                        rec.update({f"{kk}_{m}": v for kk, v in mz.items()})
                    block = max(10, 2 * (b - a + 1))
                    inc = E.mcs_included(pd.DataFrame(Lq, columns=models), alpha=0.10, reps=300, block=block)
                    for m in models:
                        rec[f"mcs_{m}"] = m in inc
                recs.append(rec)
    return recs, panel


def run_vol_forecast(cfg: dict) -> dict:
    """Part II: out-of-sample comparison of variance forecasts on every universe asset."""
    from joblib import Parallel, delayed

    from ..volatility import evaluation as E

    t0 = time.time()
    md, fs, panel = prepared = prepare(cfg)
    split = resolve_split(cfg, panel)
    out = vol_dir(cfg)
    eng = engine(cfg, "assets", prepared)
    models, aggs = eng["models"], eng["aggs"]
    periods = {"dev": np.arange(split["dev_start"], split["holdout_start"]),
               "holdout": np.arange(split["holdout_start"], split["end"])}
    ohlc = load_ohlc(md, cfg["data"].get("path", "data/raw/mitsui")) if md.name == "mitsui" else {}
    regime = fs.regime["vol_regime"].to_numpy()
    rets = {a: fs.returns[a].to_numpy(dtype=float) for a in fs.assets}
    jobs = Parallel(n_jobs=int(cfg.get("runtime", {}).get("n_jobs", -1)))(
        delayed(_asset_losses)(name, rets[name], eng["F"][s], models, aggs, ohlc.get(name), periods, regime)
        for s, name in enumerate(eng["names"]))
    per_asset = pd.DataFrame([r for recs, _ in jobs for r in recs])
    per_asset.insert(1, "market", md.assets.loc[per_asset["series"], "market"].to_numpy())
    per_asset.to_csv(out / "volforecast_per_asset.csv", index=False)

    # panel Diebold-Mariano: average QLIKE differences across assets per date
    T = panel.n_dates
    prow = []
    markets = md.assets.loc[eng["names"], "market"].to_numpy()
    for key in sorted({k for _, p in jobs for k in p}):
        pname, agg, prox = key
        stacks = {}
        for s, (_, pan) in enumerate(jobs):
            if key in pan:
                idx, L = pan[key]
                full = np.full((T, len(models)), np.nan)
                full[idx] = L
                stacks[s] = full
        if not stacks:
            continue
        S = np.stack(list(stacks.values()))  # (assets, T, models)
        mk = markets[list(stacks)]
        groups = {"all": np.ones(len(mk), bool)} | {f"market={m}": mk == m for m in np.unique(mk)}
        for gname, gm in groups.items():
            for subset, rows in [("all", None), ("vol_regime=low", "low"), ("vol_regime=mid", "mid"), ("vol_regime=high", "high")]:
                if gname != "all" and subset != "all":
                    continue
                for cid, ma, mb in CONTRASTS:
                    d = S[gm, :, models.index(ma)] - S[gm, :, models.index(mb)]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)  # dates with no asset
                        dt = np.nanmean(d, axis=0)
                    if rows is not None:
                        dt = np.where(regime == rows, dt, np.nan)
                    mean, t, p, n = E.dm_test(dt, WINDOW_LAGS[agg])
                    prow.append({"period": pname, "window": agg, "proxy": prox, "assets": gname, "subset": subset,
                                 "hypothesis": cid, "model_a": ma, "model_b": mb, "n_dates": n, "n_assets": int(gm.sum()),
                                 "mean_qlike_diff": mean, "t_hac": t, "p": p})
    pan = pd.DataFrame(prow)
    pan.to_csv(out / "volforecast_panel_dm.csv", index=False)

    # summary per model: average QLIKE relative to EWMA, MCS inclusion share, per-asset DM shares
    srow = []
    sq = per_asset[per_asset.proxy == "sq"]
    for (pname, agg), g in sq.groupby(["period", "window"]):
        for m in models:
            rec = {"period": pname, "window": agg, "model": m, "n_assets": len(g),
                   "mean_qlike_minus_ewma": float((g[f"qlike_{m}"] - g["qlike_ewma"]).mean()),
                   "median_mse_ratio_vs_ewma": float(g[f"mse_ratio_{m}"].median()),
                   "mcs_inclusion_share": float(g[f"mcs_{m}"].mean())}
            srow.append(rec)
    summary = pd.DataFrame(srow)
    # per-asset DM shares (with BH-FDR) for each pre-registered contrast
    drow = []
    for (pname, agg, prox), g in per_asset.groupby(["period", "window", "proxy"]):
        for cid, ma, mb in CONTRASTS:
            t, p = g[f"dm_{ma}_vs_{mb}_t"].to_numpy(), g[f"dm_{ma}_vs_{mb}_p"].to_numpy()
            rej = D.bh_fdr(p, 0.10)
            drow.append({"period": pname, "window": agg, "proxy": prox, "hypothesis": cid, "model_a": ma, "model_b": mb,
                         "n_assets": int(np.isfinite(p).sum()),
                         "share_a_better_BH": float(np.mean(rej & (t < 0))), "share_a_worse_BH": float(np.mean(rej & (t > 0))),
                         "share_a_lower_loss": float(np.nanmean(t < 0))})
    dshare = pd.DataFrame(drow)
    summary.to_csv(out / "volforecast_summary.csv", index=False)
    dshare.to_csv(out / "volforecast_dm_shares.csv", index=False)
    mz = sq.groupby(["period", "window"])[[c for c in sq.columns if c.startswith("mz_")]].median()
    mz.to_csv(out / "volforecast_mincer_zarnowitz_median.csv")
    log.info("volatility-forecast evaluation done in %.1fs -> %s", time.time() - t0, out)
    return {"per_asset": per_asset, "panel": pan, "summary": summary, "dm_shares": dshare}



# ------------------------------------------------------------------------- Part III
PRIMARY = [("R1", "tgbm_RV", "tgbm_R"), ("R2", "tgbm_RVC", "tgbm_RC"), ("R3", "tgbm_R_garch", "tgbm_R")]
VOL_ARMS = ["tgbm_RV", "tgbm_RVs", "tgbm_RVC", "tgbm_R_garch"]


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def _stability(ic: np.ndarray) -> dict:
    v = ic[np.isfinite(ic)]
    roll = pd.Series(ic).rolling(126, min_periods=63).mean()
    return {"mean_ic": float(v.mean()), "std_ic": float(v.std()), "ic_ir": float(v.mean() / v.std()),
            "frac_pos": float((v > 0).mean()), "worst_rolling126": float(roll.min()), "best_rolling126": float(roll.max()),
            "share_rolling126_negative": float((roll.dropna() < 0).mean())}


def run_vol_return_analysis(cfg: dict, periods: tuple[str, ...] = ("dev", "holdout")) -> dict:
    """Part III statistics from saved walk-forward forecasts (daily IC over all targets).

    ``periods=("dev",)`` touches development dates only (used before the design freeze).
    """
    from arch.bootstrap import MCS, SPA

    from ..evaluation.metrics import daily_rank_ic, paired_ic_test, summarize_ic
    from .runner import load_predictions

    md, fs, panel = prepare(cfg)
    split = resolve_split(cfg, panel)
    out = vol_dir(cfg)
    tag = "_".join(periods)
    manifest = json.loads((results_dir(cfg) / "manifest.json").read_text())
    arms = [m for m, v in manifest["models"].items() if v.get("status") == "ok"]
    P = {"dev": np.arange(split["dev_start"], split["holdout_start"]),
         "holdout": np.arange(split["holdout_start"], split["end"])}
    labels = panel.eval_labels.to_numpy()
    reps = int(cfg.get("evaluation", {}).get("bootstrap_reps", 1000))

    # forecasts -> target level; daily IC overall, by market-vol regime, by target-vol tercile
    from ..models.target_models import get_target_features

    tf = get_target_features(panel, cfg)
    svol = tf.spread_vol / np.sqrt(np.asarray(panel.horizons)[panel.h_idx])[None, :]  # per-day spread vol
    tpred, ic, ic_terc = {}, {}, {}
    for m in arms:
        p, lev = load_predictions(cfg, m)
        tp = np.nan_to_num(panel.to_targets(p) if lev == "asset" else p)
        tpred[m] = tp
        ic[m] = np.full(panel.n_dates, np.nan)
        ic_terc[m] = np.full((panel.n_dates, 3), np.nan)
        for pname in periods:
            d = P[pname]
            ic[m][d], _ = daily_rank_ic(tp[d], labels[d], min_valid=2)
            for t in d:  # target-volatility terciles, assigned ex ante with the date-t EWMA spread vol
                v = svol[t]
                ok = np.isfinite(v)
                if ok.sum() < 30:
                    continue
                q = np.nanquantile(v[ok], [1 / 3, 2 / 3])
                bins = np.digitize(v, q)
                for b in range(3):
                    sel = ok & (bins == b)
                    i_, _ = daily_rank_ic(tp[t:t + 1, sel], labels[t:t + 1, sel], min_valid=10)
                    ic_terc[m][t, b] = i_[0]

    rows, stab, regime_rows, terc_rows = [], [], [], []
    reg = fs.regime["vol_regime"].to_numpy()
    for pname in periods:
        d = P[pname]
        for m in arms:
            s = summarize_ic(ic[m][d], reps=reps)
            stab.append({"period": pname, "model": m, **_stability(ic[m][d]), "t_hac": s["t_hac"],
                         "mean_ci_lo": s["mean_ci_lo"], "mean_ci_hi": s["mean_ci_hi"]})
        for exp, a, b in [tuple(x) for x in cfg["evaluation"]["paired"]]:
            if a in ic and b in ic:
                rows.append({"period": pname, "comparison": exp, "model_a": a, "model_b": b, **paired_ic_test(ic[a][d], ic[b][d], reps=reps)})
                for r in ("low", "mid", "high"):
                    sel = d[reg[d] == r]
                    t = paired_ic_test(ic[a][sel], ic[b][sel], reps=200)
                    regime_rows.append({"period": pname, "comparison": exp, "model_a": a, "model_b": b, "regime": r, **t})
                for k, name in enumerate(("low", "mid", "high")):
                    t = paired_ic_test(ic_terc[a][d, k], ic_terc[b][d, k], reps=200)
                    terc_rows.append({"period": pname, "comparison": exp, "model_a": a, "model_b": b, "target_vol_tercile": name, **t})
    paired = pd.DataFrame(rows)
    # Holm across the three pre-registered primary tests, within each period
    paired["holm_p"] = np.nan
    for pname in periods:
        sel = [(paired.period == pname) & (paired.model_a == a) & (paired.model_b == b) for _, a, b in PRIMARY]
        idx = [paired.index[s][0] for s in sel if s.any()]
        if idx:
            paired.loc[idx, "holm_p"] = holm(paired.loc[idx, "p_hac"].tolist())
    if set(periods) == {"dev", "holdout"}:
        # Secondary (not pre-registered as primary): all out-of-sample dates pooled. Legitimate for
        # arms without development-period tuning (all tgbm_* arms use a-priori LightGBM settings).
        d = np.concatenate([P["dev"], P["holdout"]])
        extra = []
        for exp, a, b in [tuple(x) for x in cfg["evaluation"]["paired"]]:
            if a in ic and b in ic:
                extra.append({"period": "pooled_oos", "comparison": exp, "model_a": a, "model_b": b,
                              **paired_ic_test(ic[a][d], ic[b][d], reps=reps)})
        paired = pd.concat([paired, pd.DataFrame(extra)], ignore_index=True)
        # holdout halves (robustness to the evaluation window)
        h = P["holdout"]
        halves = {"holdout_1st_half": h[: len(h) // 2], "holdout_2nd_half": h[len(h) // 2:]}
        extra = []
        for hname, hd in halves.items():
            for exp, a, b in [tuple(x) for x in cfg["evaluation"]["paired"]]:
                if a in ic and b in ic:
                    extra.append({"period": hname, "comparison": exp, "model_a": a, "model_b": b,
                                  **paired_ic_test(ic[a][hd], ic[b][hd], reps=200)})
        paired = pd.concat([paired, pd.DataFrame(extra)], ignore_index=True)
    # by horizon group: daily IC within each horizon's targets
    hrows = []
    for pname in periods:
        d = P[pname]
        for k, hval in enumerate(panel.horizons):
            cols = panel.h_idx == k
            ich = {}
            for m in arms:
                ich[m], _ = daily_rank_ic(tpred[m][d][:, cols], labels[d][:, cols], min_valid=5)
            for exp, a, b in [tuple(x) for x in cfg["evaluation"]["paired"]]:
                if a in ich and b in ich:
                    hrows.append({"period": pname, "horizon": hval, "comparison": exp, "model_a": a, "model_b": b,
                                  **paired_ic_test(ich[a], ich[b], reps=200)})
    pd.DataFrame(hrows).to_csv(out / f"returns_by_horizon_{tag}.csv", index=False)
    paired.to_csv(out / f"returns_paired_{tag}.csv", index=False)
    pd.DataFrame(stab).to_csv(out / f"returns_stability_{tag}.csv", index=False)
    pd.DataFrame(regime_rows).to_csv(out / f"returns_regime_{tag}.csv", index=False)
    pd.DataFrame(terc_rows).to_csv(out / f"returns_target_vol_tercile_{tag}.csv", index=False)

    # Model Confidence Set over the target-level arms (loss = -daily IC) and SPA vs tgbm_R
    mrows = []
    tarms = [m for m in arms if m.startswith(("tgbm", "tridge"))]
    for pname in periods:
        d = P[pname]
        L = pd.DataFrame({m: -ic[m][d] for m in tarms}).dropna()
        mcs = MCS(L, size=0.10, reps=reps, block_size=20, method="R", bootstrap="stationary", seed=0)
        mcs.compute()
        inc = set(map(str, mcs.included))
        pv = mcs.pvalues["Pvalue"].to_dict()
        for m in tarms:
            mrows.append({"period": pname, "model": m, "in_mcs_90": m in inc, "mcs_pvalue": float(pv.get(m, np.nan))})
        if "tgbm_R" in L and all(a in L for a in VOL_ARMS):
            spa = SPA(L["tgbm_R"], L[VOL_ARMS], reps=reps, block_size=20, bootstrap="stationary", seed=0)
            spa.compute()
            mrows.append({"period": pname, "model": "SPA: any of " + "/".join(VOL_ARMS) + " beats tgbm_R",
                          "in_mcs_90": None, "mcs_pvalue": float(spa.pvalues["consistent"])})
    pd.DataFrame(mrows).to_csv(out / f"returns_mcs_spa_{tag}.csv", index=False)
    pd.DataFrame({m: ic[m] for m in arms}, index=fs.dates).to_csv(out / f"returns_daily_ic_{tag}.csv")
    log.info("Part III analysis (%s) -> %s", tag, out)
    return {"paired": paired, "stability": pd.DataFrame(stab), "mcs": pd.DataFrame(mrows)}
