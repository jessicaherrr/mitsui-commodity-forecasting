"""Out-of-sample evaluation of saved walk-forward forecasts.

Every model is evaluated on exactly the same decision dates and target universe: forecasts
are composed to the targets (424 for MITSUI), a missing forecast is replaced by 0 for *every* model, and
dates/targets without a realised label are excluded for every model.

Periods
-------
dev      development walk-forward dates (out of sample per fold, but hyper-parameters and
         ensemble weights were selected on this period -> mildly optimistic)
holdout  final holdout; untouched by any selection step -> the headline numbers
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from ..config import results_dir
from ..evaluation.metrics import (daily_rank_ic, error_metrics, paired_ic_test, rank_stability,
                                  summarize_ic, time_series_ic_by_target)
from ..features import build_pair_features
from ..models import FitContext, make_model
from ..models.ensemble import member_disagreement
from ..risk.diagnostics import disagreement_analysis, interval_coverage, long_short_backtest
from .runner import feature_mask, load_predictions, prepare

log = logging.getLogger(__name__)

PAIRED = [
    ("A: model class", "lgbm", "ridge"),
    ("A: model class", "lstm", "ridge"),
    ("A: model class", "lstm", "lgbm"),
    ("A: model class", "ridge", "hist_mean"),
    ("A: model class", "lgbm", "hist_mean"),
    ("A: model class", "lstm", "hist_mean"),
    ("B: cross-asset info", "ridge", "ridge_own"),
    ("B: cross-asset info", "lgbm", "lgbm_own"),
    ("D: ensembles", "ens_static_ic", "ens_equal"),
    ("D: ensembles", "ens_regime", "ens_equal"),
    ("D: ensembles", "ens_online", "ens_equal"),
    ("D: ensembles", "ens_equal", "ridge"),
    ("D: ensembles", "ens_regime", "ens_static_ic"),
    ("D: ensembles", "ens_best_equal", "hist_mean"),
    ("D: ensembles", "ens_best_regime", "ens_best_equal"),
    ("D: ensembles", "ens_best_equal", "ens_equal"),
    ("F: pooled vs target-specific", "ridge", "ridge_direct_pooled"),
    ("F: pooled vs target-specific", "ridge", "ridge_direct_target"),
    ("F: pooled vs target-specific", "ridge_direct_pooled", "ridge_direct_target"),
    ("Design: label mode", "ridge", "ridge_rawlabel"),
]

REGIME_COLUMNS = ["vol_regime", "trend_regime", "corr_regime"]


def _periods(split: dict) -> dict[str, np.ndarray]:
    return {
        "dev": np.arange(split["dev_start"], split["holdout_start"]),
        "holdout": np.arange(split["holdout_start"], split["end"]),
    }


def run_evaluation(cfg: dict) -> dict:
    t0 = time.time()
    out = results_dir(cfg)
    met = out / "metrics"
    met.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((out / "manifest.json").read_text())
    md, fs, panel = prepare(cfg)
    split = manifest["split"]
    ecfg = cfg.get("evaluation", {})
    reps, block = int(ecfg.get("bootstrap_reps", 1000)), int(ecfg.get("block", 20))
    seed = int(cfg.get("seed", 42))
    labels = panel.eval_labels.to_numpy()
    periods = _periods(split)
    oos = np.arange(split["dev_start"], split["end"])
    dates_index = panel.fs.dates
    tt = panel.targets

    models = [m for m, v in manifest["models"].items() if v.get("status") == "ok"]
    preds, levels, tps = {}, {}, {}
    for m in models:
        p, lev = load_predictions(cfg, m)
        preds[m], levels[m] = p, lev
        tp = panel.to_targets(p) if lev == "asset" else p
        tp = np.where(np.isnan(tp), 0.0, tp)
        tps[m] = tp
    log.info("evaluating %d models on %d OOS dates x %d targets", len(models), len(oos), len(tt))

    # ----------------------------------------------------------- daily IC series
    daily = {}
    daily_h = {h: {} for h in panel.horizons}
    daily_asset = {h: {} for h in panel.horizons}
    for m in models:
        ic = np.full(panel.n_dates, np.nan)
        ic[oos], _ = daily_rank_ic(tps[m][oos], labels[oos], min_valid=2)
        daily[m] = ic
        for k, h in enumerate(panel.horizons):
            cols = panel.h_idx == k
            ich = np.full(panel.n_dates, np.nan)
            ich[oos], _ = daily_rank_ic(tps[m][oos][:, cols], labels[oos][:, cols], min_valid=5)
            daily_h[h][m] = ich
            if levels[m] == "asset":
                ica = np.full(panel.n_dates, np.nan)
                lab_a = np.where(panel.fs.active[oos], panel.labels[oos, :, k], np.nan)
                ica[oos], _ = daily_rank_ic(preds[m][oos, :, k], lab_a, min_valid=5)
                daily_asset[h][m] = ica
    pd.DataFrame(daily, index=dates_index).iloc[oos].to_csv(met / "daily_ic.csv")
    pd.concat({h: pd.DataFrame(v, index=dates_index).iloc[oos] for h, v in daily_h.items()}, names=["horizon", "date_id"]).to_csv(met / "daily_ic_by_horizon.csv")

    # ----------------------------------------------------------- summary tables
    summary = {m: {} for m in models}
    rows = []
    for m in models:
        for pname, d in periods.items():
            s = summarize_ic(daily[m][d], block=block, reps=reps, seed=seed)
            s["score"] = s.get("ic_ir")
            entry = {"overall": s, "by_horizon": {}, "asset_level": {}}
            for h in panel.horizons:
                entry["by_horizon"][h] = summarize_ic(daily_h[h][m][d], block=block, reps=reps, seed=seed)
                if m in daily_asset[h]:
                    entry["asset_level"][h] = summarize_ic(daily_asset[h][m][d], block=block, reps=reps, seed=seed)
            entry["errors"] = error_metrics(tps[m][d], labels[d])
            entry["rank_stability"] = rank_stability(tps[m][d])
            summary[m][pname] = entry
            rows.append({"model": m, "period": pname, **{k: s[k] for k in ["score", "mean_ic", "std_ic", "frac_pos", "t_hac", "p_hac", "mean_ci_lo", "mean_ci_hi", "ir_ci_lo", "ir_ci_hi", "n_dates", "n_degenerate"]},
                         **entry["errors"], "rank_stability": entry["rank_stability"]})
    comp = pd.DataFrame(rows)
    comp.to_csv(met / "model_comparison.csv", index=False)

    hrows = []
    J, H = len(tt), len(panel.horizons)
    target_universe = f"targets: {J} ({J // H} per horizon)"
    asset_universe = f"single assets: {panel.n_assets}"
    for m in models:
        for pname in periods:
            for h in panel.horizons:
                s = summary[m][pname]["by_horizon"][h]
                hrows.append({"model": m, "period": pname, "horizon": h, "universe": target_universe, **s})
                if h in summary[m][pname]["asset_level"]:
                    s = summary[m][pname]["asset_level"][h]
                    hrows.append({"model": m, "period": pname, "horizon": h, "universe": asset_universe, **s})
    pd.DataFrame(hrows).to_csv(met / "by_horizon.csv", index=False)

    # ----------------------------------------------------------- paired tests
    prow = []
    paired = [tuple(x) for x in ecfg["paired"]] if ecfg.get("paired") else PAIRED
    for exp, a, b in paired:
        if a in daily and b in daily:
            for pname, d in periods.items():
                prow.append({"experiment": exp, "model_a": a, "model_b": b, "period": pname,
                             **paired_ic_test(daily[a][d], daily[b][d], block=block, reps=reps, seed=seed)})
    pd.DataFrame(prow).to_csv(met / "paired_tests.csv", index=False)

    # ----------------------------------------------------------- regimes & time
    reg = panel.fs.regime
    rrows = []
    for col in REGIME_COLUMNS:
        values = reg[col].to_numpy()
        for pname, d in {**periods, "all_oos": oos}.items():
            for r in [v for v in pd.unique(values[d]) if v is not None and v == v]:
                sel = d[values[d] == r]
                for m in models:
                    s = summarize_ic(daily[m][sel], block=min(block, max(2, len(sel) // 10)), reps=200, seed=seed)
                    rrows.append({"regime_type": col, "regime": r, "period": pname, "model": m, "n_dates": s["n_dates"],
                                  "mean_ic": s.get("mean_ic"), "std_ic": s.get("std_ic"), "ic_ir": s.get("ic_ir"),
                                  "frac_pos": s.get("frac_pos"), "t_hac": s.get("t_hac")})
    pd.DataFrame(rrows).to_csv(met / "regime_ic.csv", index=False)
    regime_dates = reg.loc[dates_index[oos], REGIME_COLUMNS + ["eq_vol_21_ann", "reg_eq_vol_pct"]]
    regime_dates.to_csv(met / "regime_by_date.csv")

    blk = 126
    trows = []
    for start in range(split["dev_start"], split["end"], blk):
        sel = np.arange(start, min(start + blk, split["end"]))
        for m in models:
            v = daily[m][sel]
            trows.append({"block_start": int(start), "block_end": int(sel[-1]), "model": m,
                          "period": "holdout" if start >= split["holdout_start"] else "dev",
                          "mean_ic": float(np.nanmean(v)), "ic_ir": float(np.nanmean(v) / np.nanstd(v)) if np.nanstd(v) > 0 else np.nan})
    pd.DataFrame(trows).to_csv(met / "time_blocks.csv", index=False)

    crow = []
    hold = periods["holdout"]
    for m in models:
        ts_ic = time_series_ic_by_target(tps[m][hold], labels[hold])
        df = pd.DataFrame({"category": tt["category"].to_numpy(), "horizon": tt["horizon"].to_numpy(), "ts_ic": ts_ic})
        for cat, g in df.groupby("category"):
            crow.append({"model": m, "category": cat, "n_targets": len(g), "mean_ts_ic": g["ts_ic"].mean(), "frac_pos": (g["ts_ic"] > 0).mean()})
    pd.DataFrame(crow).to_csv(met / "category_ts_ic.csv", index=False)

    # ----------------------------------------------------------- risk diagnostics
    risk = run_risk(cfg, panel, manifest, preds, tps, daily, periods, met)

    # ----------------------------------------------------------- interpretation & timing
    interp = interpretability(cfg, panel, fs, manifest, split, met)
    timing = timing_diagnostic(md, fs, periods, met)

    result = {"summary": summary, "risk": risk, "interpretation": interp, "timing": timing, "models": models,
              "periods": {k: [int(v[0]), int(v[-1])] for k, v in periods.items()},
              "is_synthetic": manifest.get("is_synthetic", False), "seconds": round(time.time() - t0, 1)}
    (met / "summary.json").write_text(json.dumps(result, indent=2, default=_jd))
    log.info("evaluation finished in %.1fs -> %s", time.time() - t0, met)
    return result


def run_risk(cfg, panel, manifest, preds, tps, daily, periods, met) -> dict:
    ecfg = cfg.get("evaluation", {})
    dev, hold = periods["dev"], periods["holdout"]
    out: dict = {}
    reg = panel.fs.regime["vol_regime"].to_numpy()
    groups = {f"vol_regime={r}": (reg == r) for r in ["low", "mid", "high"]}
    asset_models = [m for m in preds if manifest["models"][m].get("level") == "asset"]

    # 1) prediction intervals (calibrated on dev residuals, evaluated on holdout)
    rows = []
    for m in [x for x in ("ens_equal", "ridge", "lgbm", "lstm") if x in asset_models]:
        cov = interval_coverage(panel.labels, preds[m], panel.scale, panel.fs.active, dev, hold,
                                levels=tuple(ecfg.get("interval_levels", [0.8, 0.9])), groups=groups)
        cov.insert(0, "model", m)
        cov["horizon"] = [panel.horizons[k] for k in cov["horizon_idx"]]
        rows.append(cov)
    if rows:
        pd.concat(rows).to_csv(met / "interval_coverage.csv", index=False)

    # 2) disagreement among ensemble members
    ens = manifest["models"].get("ens_equal")
    if ens and all(m in preds for m in ens["members"]):
        members = ens["members"]
        scale = np.array(ens["member_scales"])
        dis = member_disagreement(preds, members, scale, panel.to_targets)
        _, _, tscale = build_pair_features(panel)
        y = panel.eval_labels.to_numpy()
        err = np.abs(y - tps["ens_equal"]) / np.where(tscale > 0, tscale, np.nan)
        out["disagreement_holdout"] = disagreement_analysis(dis, err, tscale, daily["ens_equal"], hold)
        out["disagreement_dev"] = disagreement_analysis(dis, err, tscale, daily["ens_equal"], dev)

    # 3) hypothetical dollar-neutral backtest on single assets, horizon 1
    bt_rows, curves = [], {}
    for m in [x for x in ("hist_mean", "reversal_5", "ridge", "lgbm", "lstm", "ens_equal", "ens_regime") if x in asset_models]:
        for vs in (True, False):
            if not vs and m not in ("ens_equal", "ridge"):
                continue
            stats, daily_bt = long_short_backtest(preds[m][:, :, 0], panel.labels[:, :, 0], panel.fs.vol, panel.fs.active,
                                                  hold, costs_bps=tuple(ecfg.get("backtest_costs_bps", [0, 2, 5, 10])), vol_scaled=vs)
            tag = f"{m}{'' if vs else '_rawweights'}"
            stats.insert(0, "signal", tag)
            bt_rows.append(stats)
            curves[tag] = daily_bt["net_0bps"].cumsum()
    if bt_rows:
        pd.concat(bt_rows).to_csv(met / "backtest_holdout.csv", index=False)
        pd.DataFrame(curves).to_csv(met / "backtest_curves_holdout.csv")
    return out


def timing_diagnostic(md, fs, periods, met) -> pd.DataFrame | dict:
    """Why the 1-date gap matters: lead-lag from asynchronous market closes.

    US equities close after LME/JPX on the same date_id, so today's global-equity move is news
    for tomorrow's LME/JPX/FX close-to-close return (gap 0).  The competition labels start at
    t+1 (gap 1), which removes that return.  We compare the asset-level daily IC of simple
    known-at-t signals against both label definitions for the non-US assets.
    """
    assets = [a for a in fs.assets if md.assets.loc[a, "market"] in ("LME", "JPX", "FX")]
    if not assets:  # only meaningful for datasets with asynchronous non-US markets
        return {}
    idx = [fs.assets.index(a) for a in assets]
    close = md.close[assets]
    from ..targets import forward_log_return

    lab = {g: np.column_stack([forward_log_return(close[a], g, g + 1).to_numpy() for a in assets]) for g in (0, 1)}
    signals = {name: fs.values[:, idx, fs.names.index(name)] for name in ("expo_EQ_1d", "ret_1_z", "expo_METAL_1d") if name in fs.names}
    rows = []
    for name, sig in signals.items():
        for pname, d in periods.items():
            for g, y in lab.items():
                ic, _ = daily_rank_ic(np.nan_to_num(sig[d]), np.where(fs.active[d][:, idx], y[d], np.nan), min_valid=5)
                s = summarize_ic(ic, reps=200)
                rows.append({"signal": name, "period": pname, "label_gap": g, "universe": f"{len(assets)} LME/JPX/FX assets, h=1",
                             "mean_ic": s.get("mean_ic"), "t_hac": s.get("t_hac"), "frac_pos": s.get("frac_pos"), "n_dates": s.get("n_dates")})
    df = pd.DataFrame(rows)
    df.to_csv(met / "timing_diagnostic.csv", index=False)
    return df.to_dict(orient="records")


def interpretability(cfg, panel, fs, manifest, split, met) -> dict:
    """Refit selected ridge / LightGBM once with information available at the holdout start."""
    tau = split["holdout_start"]
    res = {}
    for name in ("ridge", "lgbm"):
        spec = next((s for s in cfg["models"] if s["name"] == name), None)
        info = manifest["models"].get(name, {})
        if spec is None or info.get("status") != "ok":
            continue
        mask = feature_mask(fs, cfg, spec.get("features", "all"))
        model = make_model(spec["type"], name, info["selected_params"], n_jobs=1)
        model.fit(panel, FitContext(tau=tau, first_train=split["first_train"], feature_mask=mask, seed=int(cfg.get("seed", 42))))
        names = [n for n, k in zip(fs.names, mask) if k]
        groups = [g for g, k in zip(fs.groups, mask) if k]
        vals = model.coefficients() if name == "ridge" else model.feature_importance()
        if name == "lgbm":
            vals = vals / vals.sum(axis=1, keepdims=True)
        df = pd.DataFrame(vals.T, index=names, columns=[f"h{h}" for h in panel.horizons])
        df.insert(0, "group", groups)
        df.to_csv(met / f"interpret_{name}.csv")
        res[name] = {"fitted_at": int(tau), "kind": "standardized coefficients" if name == "ridge" else "gain share"}
    return res


def _jd(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
