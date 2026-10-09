"""``report`` command: figures + machine-generated markdown result tables.

Everything here is read from ``results/<profile>/`` (produced by ``train`` and ``evaluate``);
no number is typed by hand.  Interpretation lives in docs/*.md.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT, results_dir
from ..visualization import figures as F

log = logging.getLogger(__name__)

CORE = ["hist_mean", "momentum_21", "reversal_5", "ridge", "lgbm", "lstm", "ens_equal", "ens_regime", "ens_best_equal"]
CAL_NOTE = "date_id is an anonymised trading-date index; inferred calendar ~2018-01..~2025-06 (~260 rows/year)"


def report_dir(cfg: dict) -> Path:
    base = Path(cfg.get("report_root", "reports"))
    if not base.is_absolute():
        base = ROOT / base
    return base / cfg["name"]


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)

    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def run_report(cfg: dict) -> Path:
    res = results_dir(cfg)
    met = res / "metrics"
    out = report_dir(cfg)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((res / "manifest.json").read_text())
    summary = json.loads((met / "summary.json").read_text())
    synthetic = bool(manifest.get("is_synthetic"))
    split = manifest["split"]
    hold_label = f"date_id {split['holdout_start']}-{split['end'] - 1}"

    comp = pd.read_csv(met / "model_comparison.csv")
    byh = pd.read_csv(met / "by_horizon.csv")
    paired = pd.read_csv(met / "paired_tests.csv")
    reg = pd.read_csv(met / "regime_ic.csv")
    daily = pd.read_csv(met / "daily_ic.csv", index_col=0)
    blocks = pd.read_csv(met / "time_blocks.csv")
    cats = pd.read_csv(met / "category_ts_ic.csv")
    cov = pd.read_csv(met / "interval_coverage.csv") if (met / "interval_coverage.csv").exists() else pd.DataFrame()
    bt = pd.read_csv(met / "backtest_holdout.csv") if (met / "backtest_holdout.csv").exists() else pd.DataFrame()
    curves = pd.read_csv(met / "backtest_curves_holdout.csv", index_col=0) if (met / "backtest_curves_holdout.csv").exists() else pd.DataFrame()

    # ------------------------------------------------------------------ figures
    from .runner import prepare  # regime timeline needs the full-sample regime frame

    _, fs, _ = prepare(cfg)
    # Row positions -> index labels (date_id for MITSUI, timestamps for the generic adapter).
    dates = fs.dates
    if not pd.api.types.is_numeric_dtype(dates):
        dates = pd.to_datetime(dates)
        daily.index = pd.to_datetime(daily.index)
        if not curves.empty:
            curves.index = dates[curves.index.to_numpy()]
    marks = {k: dates[split[k]] for k in ("dev_start", "holdout_start")}
    regime = fs.regime.set_axis(dates)
    F.fig_regime_timeline(regime, marks, figdir / "01_regime_timeline.png", synthetic)
    F.fig_ic_over_time(daily, marks["holdout_start"], figdir / "02_ic_over_time.png", synthetic)
    F.fig_model_comparison(comp, figdir / "03_model_comparison.png", synthetic, hold_label)
    F.fig_by_horizon(byh, figdir / "04_by_horizon.png", synthetic)
    F.fig_regimes(reg, figdir / "05_vol_regimes_holdout.png", synthetic, "holdout")
    F.fig_regimes(reg, figdir / "05b_vol_regimes_all_oos.png", synthetic, "all_oos")
    F.fig_ablation(comp, figdir / "06_cross_asset_ablation.png", synthetic)
    weights = {}
    for scheme in ("ens_online", "ens_regime"):
        p = res / "predictions" / f"{scheme}_weights.npz"
        if p.exists():
            z = np.load(p, allow_pickle=False)
            weights[scheme] = pd.DataFrame(z["weights"], index=z["dates"], columns=[str(m) for m in z["members"]])
    F.fig_ensembles(comp, weights, figdir / "07_ensembles.png", synthetic)
    if not cov.empty:
        F.fig_uncertainty(cov, summary.get("risk", {}).get("disagreement_holdout"), figdir / "08_uncertainty.png", synthetic)
    if not curves.empty:
        F.fig_backtest(curves, figdir / "09_backtest_hypothetical.png", synthetic)
    ridge_i = pd.read_csv(met / "interpret_ridge.csv", index_col=0) if (met / "interpret_ridge.csv").exists() else None
    lgbm_i = pd.read_csv(met / "interpret_lgbm.csv", index_col=0) if (met / "interpret_lgbm.csv").exists() else None
    F.fig_features(ridge_i, lgbm_i, figdir / "10_feature_attribution.png", synthetic)

    # ------------------------------------------------------------------- tables
    L = []
    title = f"# Result tables - profile `{cfg['name']}`"
    L.append(title + (" - SYNTHETIC DATA, NOT MARKET EVIDENCE" if synthetic else ""))
    L.append("")
    L.append("_Machine-generated by `python -m commodity_research.cli report`. Do not edit by hand._")
    L.append("")
    L.append(f"- Dataset: `{manifest['dataset']}` ({manifest.get('provenance', '')})")
    L.append(f"- Synthetic: **{synthetic}**")
    L.append(f"- Targets evaluated: {manifest['n_targets']} ({manifest['n_assets']} assets x horizons {manifest['horizons']}, gap {manifest['gap']})")
    L.append(f"- Development OOS: date_id {split['dev_start']}-{split['holdout_start'] - 1}; final holdout: {hold_label}; refit every {split['refit_every']} dates, expanding window")
    if manifest["dataset"] == "mitsui":
        L.append(f"- {CAL_NOTE}")
    else:
        L.append(f"- Calendar: development {fs.dates[split['dev_start']]} .. {fs.dates[split['holdout_start'] - 1]}; holdout {fs.dates[split['holdout_start']]} .. {fs.dates[split['end'] - 1]}")
    L.append(f"- Config hash `{manifest['config_hash']}`, seed {manifest['seed']}, data fingerprint `{manifest.get('data_fingerprint')}`")
    L.append(f"- Training wall time: {manifest.get('total_seconds', float('nan')) / 60:.1f} min")
    L.append("")

    L.append(f"## 1. Model comparison (competition-style score = mean/std of daily rank IC over all {manifest['n_targets']} targets)\n")
    h = comp[comp.period == "holdout"].set_index("model")
    d = comp[comp.period == "dev"].set_index("model")
    models = [m for m in F.ORDER if m in h.index] + [m for m in h.index if m not in F.ORDER]
    t1 = pd.DataFrame({
        "model": models,
        "family": [F.FAMILY.get(m, "") for m in models],
        "dev score": [d.loc[m, "score"] if m in d.index else np.nan for m in models],
        "holdout score": [h.loc[m, "score"] for m in models],
        "holdout score 95% CI": [f"[{h.loc[m, 'ir_ci_lo']:.3f}, {h.loc[m, 'ir_ci_hi']:.3f}]" for m in models],
        "holdout mean IC": [h.loc[m, "mean_ic"] for m in models],
        "mean IC 95% CI": [f"[{h.loc[m, 'mean_ci_lo']:.4f}, {h.loc[m, 'mean_ci_hi']:.4f}]" for m in models],
        "t (HAC)": [h.loc[m, "t_hac"] for m in models],
        "% days IC>0": [f"{100 * h.loc[m, 'frac_pos']:.1f}" for m in models],
        "dir. acc.": [h.loc[m, "dir_acc"] for m in models],
        "rank stability": [h.loc[m, "rank_stability"] for m in models],
    })
    L.append(md_table(t1, "{:.4f}"))
    L.append(f"\nHoldout dates with a defined IC: {int(h['n_dates'].iloc[0])}; degenerate dates excluded: {int(h['n_degenerate'].max())}. "
             "CIs: circular moving-block bootstrap (block 20). t (HAC): Newey-West. Dev scores are optimistic for tuned models and in-sample for ensemble weights.\n")

    L.append("## 2. By horizon (holdout)\n")
    hh = byh[(byh.period == "holdout") & byh.model.isin(CORE)]
    piv = hh.pivot_table(index=["model", "universe"], columns="horizon", values="mean_ic").reset_index()
    piv.columns = [f"h={c} mean IC" if isinstance(c, (int, np.integer)) else c for c in piv.columns]
    L.append(md_table(piv))
    piv2 = hh[hh.universe.str.startswith("targets")].pivot_table(index="model", columns="horizon", values="ic_ir").reset_index()
    piv2.columns = [f"h={c} IC IR" if isinstance(c, (int, np.integer)) else c for c in piv2.columns]
    L.append("\nIC information ratio (mean/std of daily IC) within each horizon group:\n")
    L.append(md_table(piv2))
    L.append("")

    L.append("## 3. Pre-specified paired comparisons (daily IC difference, model_a - model_b)\n")
    pt = paired.copy()
    pt["95% CI"] = [f"[{a:.4f}, {b:.4f}]" for a, b in zip(pt.ci_lo, pt.ci_hi)]
    L.append(md_table(pt[["experiment", "model_a", "model_b", "period", "n_dates", "mean_diff", "95% CI", "t_hac", "p_hac"]]))
    L.append("\nNo multiple-testing adjustment is applied; with 17 comparisons x 2 periods, isolated p < 0.05 results should be read as exploratory.\n")

    L.append("## 4. Volatility regimes (mean daily IC; regime known at decision date)\n")
    for period in ("holdout", "all_oos"):
        r = reg[(reg.regime_type == "vol_regime") & (reg.period == period) & reg.model.isin(CORE)]
        if r.empty:
            continue
        pv = r.pivot_table(index="model", columns="regime", values="mean_ic")
        pv = pv[[c for c in ["low", "mid", "high"] if c in pv.columns]]
        n = r.drop_duplicates("regime").set_index("regime")["n_dates"]
        pv.columns = [f"{c} (n={int(n[c])})" for c in pv.columns]
        L.append(f"**{period}**\n")
        L.append(md_table(pv.reset_index()))
        L.append("")
    for col in ("trend_regime", "corr_regime"):
        r = reg[(reg.regime_type == col) & (reg.period == "holdout") & reg.model.isin(CORE)]
        if r.empty:
            continue
        pv = r.pivot_table(index="model", columns="regime", values="mean_ic")
        n = r.drop_duplicates("regime").set_index("regime")["n_dates"]
        pv.columns = [f"{c} (n={int(n[c])})" for c in pv.columns]
        L.append(f"**{col} (holdout)**\n")
        L.append(md_table(pv.reset_index()))
        L.append("")

    L.append("## 5. Ensemble weights\n")
    erows = []
    for m in ("ens_equal", "ens_static_ic", "ens_regime", "ens_online", "ens_best_equal", "ens_best_regime"):
        info = manifest["models"].get(m)
        if not info:
            continue
        scheme, members = info["scheme"], info["members"]
        if scheme == "equal":
            used = str([round(1 / len(members), 4)] * len(members))
        elif scheme == "static_ic":
            used = str(info["static_weights"])
        elif scheme == "regime":
            used = json.dumps(info.get("regime_weights", {})) + " (fallback " + str(info["static_weights"]) + ")"
        else:
            used = "time-varying: trailing observable IC (see figure 07)"
        erows.append({"ensemble": m, "scheme": scheme, "members": ", ".join(members), "weights used": used})
    if erows:
        L.append(md_table(pd.DataFrame(erows)))
    L.append("")

    if not cov.empty:
        L.append("## 6. Prediction-interval coverage (holdout; calibrated on development residuals)\n")
        c = cov.groupby(["model", "level", "subset"])[["coverage_volscaled", "coverage_constant", "avg_width_volscaled_bps", "width_constant_bps"]].mean().reset_index()
        L.append(md_table(c))
        L.append("")
    risk = summary.get("risk", {})
    if risk.get("disagreement_holdout"):
        L.append("## 7. Model disagreement vs realised error\n")
        rows = []
        for per in ("dev", "holdout"):
            x = risk.get(f"disagreement_{per}")
            if x:
                rows.append({"period": per, **{k: (str([round(v, 3) for v in val]) if isinstance(val, list) else val) for k, val in x.items()}})
        L.append(md_table(pd.DataFrame(rows)))
        L.append("")
    if not bt.empty:
        L.append("## 8. Hypothetical long-short diagnostic (holdout, horizon-1 single assets)\n")
        L.append("Not evidence of tradable profitability: ignores financing, borrow, FX conversion, impact, contract rolls and asynchronous market closes.\n")
        L.append(md_table(bt))
        L.append("")

    if (met / "timing_diagnostic.csv").exists():
        L.append("## 8b. Information timing: next-day (gap 0) vs competition (gap 1) labels\n")
        L.append("Asset-level daily IC of known-at-t signals for LME/JPX/FX assets at horizon 1. Gap 0 = return from close t to t+1 "
                 "(contains the asynchronous-close lead-lag); gap 1 = competition definition (t+1 to t+2).\n")
        L.append(md_table(pd.read_csv(met / "timing_diagnostic.csv")))
        L.append("")

    L.append("## 9. Time-series predictability by target category (holdout; mean per-target Spearman over time)\n")
    cc = cats[cats.model.isin(CORE)].pivot_table(index="category", columns="model", values="mean_ts_ic")
    n = cats.drop_duplicates("category").set_index("category")["n_targets"]
    cc.insert(0, "n_targets", n)
    L.append(md_table(cc.reset_index()))
    L.append("")

    L.append("## 10. Mean daily IC by 126-date block (all out-of-sample dates)\n")
    tb = blocks[blocks.model.isin(CORE)].pivot_table(index=["block_start", "period"], columns="model", values="mean_ic").reset_index()
    L.append(md_table(tb))
    L.append("")

    L.append("## 11. Hyper-parameter selection (development period only)\n")
    grows = []
    for m, info in manifest["models"].items():
        for g in info.get("grid", []):
            grows.append({"model": m, "params": json.dumps(g["params"]), "dev score": g["dev_score"], "dev mean IC": g["dev_mean_ic"],
                          "selected": "yes" if g["params"] == info["selected_params"] else ""})
    L.append(md_table(pd.DataFrame(grows)))
    L.append("")

    for name, d_i in (("ridge (standardized coefficients)", ridge_i), ("LightGBM (gain share)", lgbm_i)):
        if d_i is not None:
            L.append(f"## 12. Feature attribution - {name}, refit at holdout start\n")
            top = d_i.reindex(d_i["h1"].abs().sort_values(ascending=False).index).head(12).reset_index().rename(columns={"index": "feature"})
            L.append(md_table(top))
            L.append("")

    L.append("## Figures\n")
    for f in sorted(figdir.glob("*.png")):
        L.append(f"- `{f.relative_to(out)}`")
    path = out / "results_tables.md"
    path.write_text("\n".join(L) + "\n")
    log.info("report written to %s", out)
    return path
