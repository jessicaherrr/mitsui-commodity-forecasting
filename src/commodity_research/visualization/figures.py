"""Publication figures generated from saved evaluation outputs (never from placeholders).

Style follows a fixed, colour-vision-deficiency-validated categorical order; each model keeps
its colour in every figure; thin marks, hairline solid grid, one y-axis per panel, a legend
whenever two or more series are shown.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e0"
CONTEXT = "#a3a29c"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]

MODEL_COLOR = {
    "ridge": SLOTS[0], "lgbm": SLOTS[1], "lstm": SLOTS[2], "hist_mean": SLOTS[3], "ens_equal": SLOTS[5],
}
FAMILY = {
    "hist_mean": "benchmark", "momentum_21": "benchmark", "reversal_5": "benchmark",
    "ridge": "linear", "ridge_own": "linear", "ridge_rawlabel": "linear",
    "lgbm": "gradient boosting", "lgbm_own": "gradient boosting", "lstm": "LSTM",
    "ridge_direct_pooled": "direct pair model", "ridge_direct_target": "direct pair model",
    "ens_equal": "ensemble", "ens_static_ic": "ensemble", "ens_regime": "ensemble", "ens_online": "ensemble",
    "ens_best_equal": "ensemble", "ens_best_regime": "ensemble",
}
FAMILY_COLOR = {"benchmark": SLOTS[3], "linear": SLOTS[0], "gradient boosting": SLOTS[1], "LSTM": SLOTS[2],
                "direct pair model": SLOTS[4], "ensemble": SLOTS[5]}
ORDER = ["hist_mean", "momentum_21", "reversal_5", "ridge", "ridge_own", "ridge_rawlabel", "lgbm", "lgbm_own",
         "lstm", "ridge_direct_pooled", "ridge_direct_target", "ens_equal", "ens_static_ic", "ens_regime", "ens_online",
         "ens_best_equal", "ens_best_regime"]


def color(model: str, i: int = 0) -> str:
    return MODEL_COLOR.get(model, FAMILY_COLOR.get(FAMILY.get(model, ""), CONTEXT))


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True, "font.size": 9.5, "axes.titlesize": 10.5,
        "axes.titleweight": "semibold", "legend.frameon": False, "lines.linewidth": 1.8, "figure.dpi": 150,
    })


def _finish(fig, ax_or_axes, path: Path, caption: str, synthetic: bool) -> None:
    note = caption + ("\nSYNTHETIC DATA - NOT MARKET EVIDENCE" if synthetic else "")
    fig.text(0.01, 0.005, note, fontsize=7.5, color=INK2, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.06 if "\n" in note else 0.045, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def present(models, available):
    return [m for m in models if m in available]


# ---------------------------------------------------------------------------- figures
def fig_ic_over_time(daily_ic: pd.DataFrame, holdout_start: int, path: Path, synthetic: bool, window: int = 126):
    _style()
    models = present(["hist_mean", "ridge", "lgbm", "lstm", "ens_equal"], daily_ic.columns)
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    for m in models:
        r = daily_ic[m].rolling(window, min_periods=window // 2).mean()
        ax.plot(r.index, r.to_numpy(), color=color(m), label=m)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvspan(holdout_start, daily_ic.index.max(), color=GRID, alpha=0.6, lw=0)
    ax.text(holdout_start, ax.get_ylim()[1] * 0.92, "  final holdout", color=INK2, fontsize=8, va="top")
    ax.set_xlabel(_xlabel(daily_ic.index) + " (decision date)")
    ax.set_ylabel(f"rolling {window}-date mean daily rank IC")
    ax.set_title("Walk-forward out-of-sample rank IC over time (all targets)")
    ax.legend(ncol=len(models), loc="upper left", bbox_to_anchor=(0, -0.18), fontsize=8)
    _finish(fig, ax, path, "Daily Spearman IC across all targets with a realised label; every point is out of sample "
            "(models refit on an expanding window with horizon-specific label-availability purging).", synthetic)


def fig_model_comparison(comp: pd.DataFrame, path: Path, synthetic: bool, hold_label: str):
    _style()
    hold = comp[comp.period == "holdout"].set_index("model")
    dev = comp[comp.period == "dev"].set_index("model")
    models = [m for m in ORDER if m in hold.index] + [m for m in hold.index if m not in ORDER]
    y = np.arange(len(models))[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 0.36 * len(models) + 1.6))
    for yi, m in zip(y, models):
        c = FAMILY_COLOR.get(FAMILY.get(m, ""), CONTEXT)  # colour = model family (legend)
        r = hold.loc[m]
        ax.plot([r.mean_ci_lo, r.mean_ci_hi], [yi, yi], color=c, lw=1.6)
        ax.plot(r.mean_ic, yi, "o", color=c, ms=6.5, mec=SURFACE, mew=1.5)
        if m in dev.index:
            ax.plot(dev.loc[m].mean_ic, yi, "o", mfc=SURFACE, mec=INK2, ms=5.5, mew=1.1)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, models)
    ax.set_xlabel("mean daily rank IC (all targets)")
    ax.set_title("Model comparison: holdout mean IC with 95% block-bootstrap CI")
    handles = [plt.Line2D([], [], marker="o", ls="", color=FAMILY_COLOR[f], label=f) for f in FAMILY_COLOR
               if any(FAMILY.get(m) == f for m in models)]
    handles.append(plt.Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=INK2, label="development period"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    _finish(fig, ax, path, f"Filled: final holdout ({hold_label}); hollow: development walk-forward (used for "
            "hyper-parameter and ensemble-weight selection, hence mildly optimistic).", synthetic)


def fig_by_horizon(byh: pd.DataFrame, path: Path, synthetic: bool):
    _style()
    d = byh[(byh.period == "holdout") & byh.universe.str.startswith("targets")]
    models = present(["hist_mean", "ridge", "lgbm", "lstm", "ens_equal"], d.model.unique())
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    for m in models:
        g = d[d.model == m].sort_values("horizon")
        axes[0].plot(g.horizon, g.mean_ic, "-o", color=color(m), ms=5, label=m)
        axes[1].plot(g.horizon, g.ic_ir, "-o", color=color(m), ms=5, label=m)
    for ax, lab in zip(axes, ["mean daily IC", "IC information ratio (mean/std)"]):
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(sorted(d.horizon.unique()))
        ax.set_xlabel("forecast horizon (dates)")
        ax.set_ylabel(lab)
    axes[0].set_title("Accuracy by horizon")
    axes[1].set_title("Stability by horizon")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    _finish(fig, axes, path, "Final holdout; cross-sectional IC computed within each horizon group of targets.", synthetic)


def fig_regimes(reg: pd.DataFrame, path: Path, synthetic: bool, period: str = "holdout"):
    _style()
    d = reg[(reg.regime_type == "vol_regime") & (reg.period == period)]
    models = present(["hist_mean", "ridge", "lgbm", "lstm", "ens_equal"], d.model.unique())
    levels = [lv for lv in ["low", "mid", "high"] if lv in d.regime.unique()]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    w = 0.8 / max(len(models), 1)
    for k, m in enumerate(models):
        g = d[d.model == m].set_index("regime").reindex(levels)
        x = np.arange(len(levels)) + (k - (len(models) - 1) / 2) * w
        ax.bar(x, g.mean_ic, width=w - 0.03, color=color(m), label=m)
    ax.axhline(0, color=INK2, lw=0.8)
    n = d.drop_duplicates("regime").set_index("regime").reindex(levels).n_dates
    ax.set_xticks(np.arange(len(levels)), [f"{lv} vol\n(n={int(v)} dates)" for lv, v in zip(levels, n)])
    ax.set_ylabel("mean daily rank IC")
    ax.set_title(f"Forecast accuracy by equity-volatility regime ({period})")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    _finish(fig, ax, path, "Regime = tercile of the causal expanding percentile of 21-day realised volatility of VT "
            "(global equity), observable at the decision date.", synthetic)


def fig_ablation(comp: pd.DataFrame, path: Path, synthetic: bool):
    _style()
    pairs = [("ridge_own", "ridge"), ("lgbm_own", "lgbm")]
    pairs = [p for p in pairs if p[0] in comp.model.values and p[1] in comp.model.values]
    if not pairs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
    for ax, period in zip(axes, ["dev", "holdout"]):
        d = comp[comp.period == period].set_index("model")
        for k, (own, full) in enumerate(pairs):
            for j, m in enumerate((own, full)):
                r = d.loc[m]
                x = k + (j - 0.5) * 0.3
                c = color(full)
                ax.plot([x, x], [r.mean_ci_lo, r.mean_ci_hi], color=c, lw=1.5, alpha=0.55 if j == 0 else 1)
                ax.plot(x, r.mean_ic, "o", ms=6.5, color=c if j == 1 else SURFACE, mec=c, mew=1.6)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(len(pairs)), [p[1] for p in pairs])
        ax.set_title(period)
    axes[0].set_ylabel("mean daily rank IC (95% CI)")
    h = [plt.Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=INK2, label="own-asset features only"),
         plt.Line2D([], [], marker="o", ls="", color=INK2, label="+ cross-asset, relative, regime")]
    axes[1].legend(handles=h, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.suptitle("Experiment B: does cross-asset information help?", fontsize=10.5, fontweight="semibold")
    _finish(fig, axes, path, "Same model class, hyper-parameter grid, walk-forward schedule and target universe; only "
            "the feature set differs.", synthetic)


def fig_ensembles(comp: pd.DataFrame, weights: dict[str, pd.DataFrame], path: Path, synthetic: bool):
    _style()
    ens = present(["ens_equal", "ens_static_ic", "ens_regime", "ens_online", "ens_best_equal", "ens_best_regime"], comp.model.unique())
    if not ens:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), gridspec_kw={"width_ratios": [1, 1.4]})
    d = comp[comp.period == "holdout"].set_index("model")
    ref = present(["hist_mean", "ridge", "lgbm", "lstm"], d.index)
    names = ref + ens
    y = np.arange(len(names))[::-1]
    for yi, m in zip(y, names):
        r = d.loc[m]
        c = color(m) if m in MODEL_COLOR else FAMILY_COLOR["ensemble"]
        axes[0].plot([r.mean_ci_lo, r.mean_ci_hi], [yi, yi], color=c, lw=1.5)
        axes[0].plot(r.mean_ic, yi, "o", color=c, ms=6, mec=SURFACE, mew=1.4)
    axes[0].axvline(0, color=INK2, lw=0.8)
    axes[0].set_yticks(y, names)
    axes[0].set_xlabel("holdout mean daily IC (95% CI)")
    axes[0].set_title("Members vs combinations")
    if "ens_online" in weights:
        w = weights["ens_online"]
        bottom = np.zeros(len(w))
        for m in w.columns:
            axes[1].fill_between(w.index, bottom, bottom + w[m].to_numpy(), color=color(m), lw=0, label=m, step="post")
            bottom += w[m].to_numpy()
        axes[1].set_ylim(0, 1)
        axes[1].set_xlabel("row position (date index)")
        axes[1].set_ylabel("weight")
        axes[1].set_title("Causal online weights (trailing realised IC)")
        axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    _finish(fig, axes, path, "Weights use only ICs whose labels were observable at the decision date; static/regime "
            "weights were estimated on development dates only.", synthetic)


def fig_uncertainty(cov: pd.DataFrame, dis: dict | None, path: Path, synthetic: bool, model: str = "ens_equal"):
    _style()
    d = cov[(cov.model == model) & (cov.level == 0.9)]
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
    subsets = [s for s in ["vol_regime=low", "vol_regime=mid", "vol_regime=high", "all"] if s in d.subset.unique()]
    g = d.groupby("subset")[["coverage_volscaled", "coverage_constant"]].mean().reindex(subsets)
    x = np.arange(len(subsets))
    axes[0].bar(x - 0.18, g.coverage_volscaled, 0.34, color=SLOTS[0], label="vol-scaled interval")
    axes[0].bar(x + 0.18, g.coverage_constant, 0.34, color=CONTEXT, label="constant-width interval")
    axes[0].axhline(0.9, color=INK, lw=1)
    axes[0].text(-0.45, 0.905, "nominal 90%", fontsize=8, ha="left", va="bottom", color=INK)
    axes[0].set_xticks(x, [s.replace("vol_regime=", "") for s in subsets])
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_ylabel("holdout coverage")
    axes[0].set_title("90% interval coverage by vol regime")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2, fontsize=8)
    if dis and dis.get("abs_error_z_by_disagreement_quintile"):
        q = dis["abs_error_z_by_disagreement_quintile"]
        axes[1].plot(np.arange(1, len(q) + 1), q, "-o", color=SLOTS[5], ms=5)
        axes[1].set_xticks(np.arange(1, len(q) + 1))
        axes[1].set_xlabel("model-disagreement quintile (1 = lowest)")
        axes[1].set_ylabel("mean |error| / target volatility")
        axes[1].set_title("Does disagreement flag larger errors?")
    _finish(fig, axes, path, f"Intervals for {model} asset forecasts, calibrated on development residuals (split-conformal), "
            "evaluated on the holdout, averaged over horizons. Disagreement = std of scale-normalised member forecasts.", synthetic)


def fig_backtest(curves: pd.DataFrame, path: Path, synthetic: bool):
    _style()
    cols = present(["hist_mean", "ridge", "lgbm", "lstm", "ens_equal"], curves.columns)
    if not cols:
        return
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    for m in cols:
        ax.plot(curves.index, curves[m].to_numpy() * 100, color=color(m), label=m)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel(_xlabel(curves.index) + " (signal date)")
    ax.set_ylabel("cumulative log return (%), gross")
    ax.set_title("Hypothetical dollar-neutral 1-day portfolio on single assets (holdout)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    _finish(fig, ax, path, "Paper diagnostic, not a tradable result: enter at close t+1, exit t+2, unit gross exposure, "
            "no costs/financing/FX conversion; see backtest table for cost sensitivity.", synthetic)


def fig_features(ridge: pd.DataFrame | None, lgbm: pd.DataFrame | None, path: Path, synthetic: bool, top: int = 12):
    _style()
    panels = [(n, d) for n, d in (("ridge: standardized coefficient (h=1)", ridge), ("LightGBM: gain share (h=1)", lgbm)) if d is not None]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.0))
    axes = np.atleast_1d(axes)
    gcol = {"own": SLOTS[0], "cross": SLOTS[1], "relative": SLOTS[2], "regime": SLOTS[3]}
    for ax, (title, d) in zip(axes, panels):
        s = d.reindex(d["h1"].abs().sort_values(ascending=False).index).head(top)[::-1]
        ax.barh(np.arange(len(s)), s["h1"], color=[gcol.get(g, CONTEXT) for g in s["group"]], height=0.7)
        ax.set_yticks(np.arange(len(s)), s.index, fontsize=8)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(title)
    h = [plt.Rectangle((0, 0), 1, 1, color=c, label=g) for g, c in gcol.items()]
    axes[-1].legend(handles=h, loc="lower right", fontsize=8, title="feature group", title_fontsize=8)
    _finish(fig, axes, path, "Models refit once on all labels observable at the holdout start; features are standardized "
            "on the training window.", synthetic)


def _xlabel(index) -> str:
    return "date_id" if pd.api.types.is_numeric_dtype(index) else "date"


def fig_regime_timeline(reg: pd.DataFrame, marks: dict, path: Path, synthetic: bool):
    _style()
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    v = reg["eq_vol_21_ann"] * 100
    ax.plot(v.index, v.to_numpy(), color=INK2, lw=1.1, label="21-day realised vol (EQ factor, else first factor)")
    for lv, c in [("high", SLOTS[1]), ("low", SLOTS[0])]:
        m = (reg["vol_regime"] == lv).to_numpy()
        ax.fill_between(v.index, 0, v.max() * 1.05, where=m, color=c, alpha=0.12, lw=0, label=f"{lv}-vol regime")
    for k, lab in [("dev_start", "dev OOS starts"), ("holdout_start", "holdout starts")]:
        ax.axvline(marks[k], color=INK, lw=1)
        ax.text(marks[k], v.max() * 0.98, "  " + lab, fontsize=8, va="top")
    ax.set_ylim(0, v.max() * 1.05)
    ax.set_xlabel(_xlabel(reg.index))
    ax.set_ylabel("annualised vol (%)")
    ax.set_title("Market-state context: reference-factor volatility and causal regime labels")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    _finish(fig, ax, path, "Regime labels use only data up to each date (expanding percentile of the reference "
            "factor's 21-day volatility; VT for MITSUI).", synthetic)
