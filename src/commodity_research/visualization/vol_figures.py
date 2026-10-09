"""Figures for the volatility research phase (generated from saved results only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .figures import CONTEXT, GRID, INK, INK2, SLOTS, SURFACE, _finish, _style, plt

VOL_COLOR = {"const": CONTEXT, "hv21": SLOTS[4], "hv63": "#c45a8a", "ewma": SLOTS[3], "garch_n": SLOTS[1],
             "garch_t": SLOTS[0], "gjr_t": SLOTS[5]}
SHORT = {"LME_CA_Close": "LME copper", "LME_AH_Close": "LME aluminium", "JPX_Gold_Standard_Futures_Close": "JPX gold",
         "US_Stock_XLE_adj_close": "XLE (energy equity)", "FX_AUDUSD": "AUD/USD", "FX_USDJPY": "USD/JPY",
         "US_Stock_GLD_adj_close": "GLD", "US_Stock_FCX_adj_close": "FCX", "US_Stock_VT_adj_close": "VT"}


def fig_acf(figdata: dict, series: list[str], path: Path, sample_label: str):
    _style()
    series = [s for s in series if s in figdata]
    fig, axes = plt.subplots(len(series), 2, figsize=(8.6, 1.9 * len(series) + 0.8), sharex=True)
    axes = np.atleast_2d(axes)
    for i, s in enumerate(series):
        d = figdata[s]
        band = 1.96 / np.sqrt(d["n"])
        for j, (key, lab) in enumerate((("acf_r", "returns $r_t$"), ("acf_r2", "squared returns $r_t^2$"))):
            ax = axes[i, j]
            v = np.asarray(d[key])[1:]
            ax.bar(np.arange(1, len(v) + 1), v, width=0.6, color=SLOTS[0] if j == 0 else SLOTS[1], zorder=2)
            ax.axhspan(-band, band, color=GRID, alpha=0.9, lw=0, zorder=0)
            ax.axhline(0, color=INK2, lw=0.6)
            ax.set_ylim(min(-0.15, v.min() - 0.02), max(0.35, v.max() + 0.03))
            if i == 0:
                ax.set_title(f"ACF of {lab}")
            if j == 0:
                ax.set_ylabel(SHORT.get(s, s), fontsize=8.5)
    for ax in axes[-1]:
        ax.set_xlabel("lag (trading days)")
    _finish(fig, axes, path, f"Sample autocorrelations, {sample_label}. Grey band: approximate 95% bounds under white "
            "noise (±1.96/√n). Squared returns are strongly autocorrelated (volatility clustering); returns are not.", False)


def fig_qq(figdata: dict, series: list[str], path: Path, sample_label: str):
    _style()
    series = [s for s in series if s in figdata]
    fig, axes = plt.subplots(len(series), 3, figsize=(8.6, 2.2 * len(series) + 0.8))
    axes = np.atleast_2d(axes)
    titles = ("raw returns vs Normal", "GARCH-t residuals vs Normal", "GARCH-t residuals vs fitted Student-t")
    for i, s in enumerate(series):
        d = figdata[s]
        for j, key in enumerate(("qq_raw_normal", "qq_z_normal", "qq_z_t")):
            ax = axes[i, j]
            x, y = np.asarray(d[key][0]), np.asarray(d[key][1])
            lim = max(abs(x).max(), abs(y).max())
            ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=0.8)
            ax.plot(x, y, "o", ms=2.4, color=SLOTS[0] if j < 2 else SLOTS[2])
            if i == 0:
                ax.set_title(titles[j], fontsize=9.5)
            if j == 0:
                ax.set_ylabel(SHORT.get(s, s), fontsize=8.5)
            if j == 2:
                ax.text(0.04, 0.92, f"ν = {d['nu']:.1f}", transform=ax.transAxes, fontsize=8, color=INK2)
    for ax in axes[-1]:
        ax.set_xlabel("theoretical quantile")
    _finish(fig, axes, path, f"Q-Q plots, {sample_label}. GARCH removes volatility clustering but the standardised "
            "residuals remain fat-tailed relative to the Normal; the fitted Student-t matches the tails far better.", False)


def fig_diag_shares(summary: pd.DataFrame, path: Path, caption: str = ""):
    _style()
    d = summary.copy()
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    samples = [s for s in ["assets, estimation", "assets, holdout", "target spreads, estimation", "assets, estimation (GARCH residuals)"] if s in d["sample"].unique()]
    labels, vals, cols = [], [], []
    for k, s in enumerate(samples):
        g = d[d["sample"] == s]
        for _, r in g.iterrows():
            labels.append(f"{r['test']}  [{s}]")
            vals.append(r["share_BH_q0.1"])
            cols.append(SLOTS[k])
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=cols, height=0.7)
    ax.set_yticks(y, labels, fontsize=7.2)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of series rejecting H0 after Benjamini-Hochberg (q = 0.10)")
    ax.set_title("Stylised-fact tests across the universe")
    _finish(fig, ax, path, caption, False)


def fig_vol_paths(dates, realised: dict, forecasts: dict, series: list[str], holdout_start, path: Path):
    _style()
    fig, axes = plt.subplots(len(series), 1, figsize=(8.4, 2.3 * len(series) + 0.6), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, series):
        ax.plot(dates, realised[s] * 100, color=CONTEXT, lw=1.0, label="realised 21-day vol (ex post)")
        for m in ("ewma", "garch_t"):
            ax.plot(dates, forecasts[s][m] * 100, color=VOL_COLOR[m], lw=1.3, label=f"{m} forecast (ex ante)")
        ax.axvline(holdout_start, color=INK, lw=0.8)
        ax.set_ylabel(f"{SHORT.get(s, s)}\nann. vol (%)", fontsize=8.5)
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    axes[-1].set_xlabel("date_id" if np.issubdtype(np.asarray(dates).dtype, np.number) else "date")
    axes[0].set_title("21-day-ahead volatility forecasts vs realised volatility")
    _finish(fig, axes, path, "Forecasts made at each date for the next 21 trading days (annualised); realised = root of the "
            "sum of the next 21 squared returns. Vertical line: holdout start.", False)


def fig_qlike(summary: pd.DataFrame, path: Path, models: list[str], caption: str = ""):
    _style()
    windows = [w for w in ("k1", "cum5", "cum21") if w in summary["window"].unique()]
    fig, axes = plt.subplots(1, len(windows), figsize=(9.0, 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, w in zip(axes, windows):
        for j, period in enumerate(("dev", "holdout")):
            g = summary[(summary.window == w) & (summary.period == period)].set_index("model").reindex(models)
            y = np.arange(len(models))[::-1] + (0.15 if j == 0 else -0.15)
            ax.plot(g["mean_qlike_minus_ewma"], y, "o", ms=6 if j else 5, color=SLOTS[0] if j else SURFACE,
                    mec=SLOTS[0], mew=1.4, ls="", label=period)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title({"k1": "1 day", "cum5": "5 days", "cum21": "21 days"}[w])
    fig.supxlabel("mean QLIKE(model) − QLIKE(EWMA) across assets (lower = more accurate)", fontsize=9, color=INK2, y=0.1)
    axes[0].set_yticks(np.arange(len(models))[::-1], models)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    _finish(fig, axes, path, caption, False)


def fig_mcs(summary: pd.DataFrame, path: Path, models: list[str], period: str = "holdout"):
    _style()
    g = summary[summary.period == period]
    windows = [w for w in ("k1", "cum5", "cum21") if w in g["window"].unique()]
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    w_ = 0.8 / len(windows)
    for k, w in enumerate(windows):
        v = g[g.window == w].set_index("model").reindex(models)["mcs_inclusion_share"]
        ax.bar(np.arange(len(models)) + (k - (len(windows) - 1) / 2) * w_, v, width=w_ - 0.03,
               color=SLOTS[k], label={"k1": "1 day", "cum5": "5 days", "cum21": "21 days"}[w])
    ax.set_xticks(np.arange(len(models)), models)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share of assets with model in 90% MCS")
    ax.set_title(f"Model Confidence Set inclusion ({period})")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    _finish(fig, ax, path, "Hansen-Lunde-Nason MCS per asset (QLIKE, stationary bootstrap). High inclusion for many models "
            "reflects the low power of noisy daily proxies, not equal accuracy.", False)


def fig_arms(stab: pd.DataFrame, path: Path, arms: list[str], periods: list[str]):
    _style()
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(arms) + 1.6))
    y = np.arange(len(arms))[::-1]
    for j, p in enumerate(periods):
        g = stab[stab.period == p].set_index("model").reindex(arms)
        yy = y + (0.15 if j == 0 else -0.15) * (len(periods) > 1)
        c = SLOTS[0] if p == "holdout" else INK2
        ax.hlines(yy, g["mean_ci_lo"], g["mean_ci_hi"], color=c, lw=1.4, alpha=0.8)
        ax.plot(g["mean_ic"], yy, "o", ms=6, color=c if p == "holdout" else SURFACE, mec=c, mew=1.4, ls="", label=p)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, arms)
    ax.set_xlabel("mean daily rank IC (all targets), 95% block-bootstrap CI")
    ax.set_title("Return forecasting: volatility arms vs return/rank baseline")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    _finish(fig, ax, path, "Target-level pooled models (one per horizon). R = return/rank, Vn = vol-normalised signals, "
            "Vs = volatility state, C = cross-asset. _garch/_none = label scaling.", False)


def fig_delta_by(df: pd.DataFrame, col: str, levels: list[str], path: Path, title: str, comparisons: list[str]):
    _style()
    fig, ax = plt.subplots(figsize=(7.8, 3.2))
    w_ = 0.8 / len(comparisons)
    for k, cmp in enumerate(comparisons):
        g = df[df.comparison == cmp].set_index(col).reindex(levels)
        x = np.arange(len(levels)) + (k - (len(comparisons) - 1) / 2) * w_
        ax.bar(x, g["mean_diff"], width=w_ - 0.03, color=SLOTS[k], label=cmp)
        ax.vlines(x, g["ci_lo"], g["ci_hi"], color=INK, lw=0.9)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(np.arange(len(levels)), levels)
    ax.set_ylabel("ΔIC (model_a − model_b)")
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7.5)
    _finish(fig, ax, path, "Paired daily IC differences on identical dates/targets; whiskers = 95% block-bootstrap CI.", False)


def fig_rolling(daily: pd.DataFrame, arms: list[str], holdout_start, path: Path, window: int = 126):
    _style()
    fig, ax = plt.subplots(figsize=(8.2, 3.3))
    for k, m in enumerate(arms):
        if m in daily:
            r = daily[m].rolling(window, min_periods=window // 2).mean()
            ax.plot(r.index, r.to_numpy(), color=SLOTS[k], label=m)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(holdout_start, color=INK, lw=0.8)
    ax.set_ylabel(f"rolling {window}-date mean IC")
    ax.set_xlabel("date_id" if pd.api.types.is_numeric_dtype(daily.index) else "date")
    ax.set_title("Stability over time: does adding volatility information smooth the IC path?")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    _finish(fig, ax, path, "Out-of-sample daily rank IC over all targets, rolling mean. Vertical line: holdout start.", False)


def fig_group_importance(imp: pd.DataFrame, path: Path):
    _style()
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    g = imp.groupby("group")["gain_share"].sum().reindex(["R", "Vn", "Vs", "C"]).fillna(0)
    ax.bar(g.index, g.values, color=[SLOTS[0], SLOTS[2], SLOTS[1], SLOTS[3]], width=0.6)
    for x, v in zip(g.index, g.values):
        ax.text(x, v + 0.01, f"{v:.0%}", ha="center", fontsize=8.5, color=INK)
    ax.set_ylabel("share of total LightGBM gain")
    ax.set_title("Where tgbm_RVC splits: feature-group gain shares")
    _finish(fig, ax, path, "Refit once on all labels observable at the holdout start; gain averaged over the four horizons.", False)
