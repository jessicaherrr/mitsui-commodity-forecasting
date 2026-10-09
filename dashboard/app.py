"""Research dashboard (Streamlit) - reads saved results only; no live data, no new numbers.

    streamlit run dashboard/app.py

Requires ``python -m commodity_research.cli all --profile fast`` (and/or ``full``) to have run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS, REPORTS = ROOT / "results", ROOT / "reports"
CORE = ["hist_mean", "momentum_21", "reversal_5", "ridge", "lgbm", "lstm", "ens_equal", "ens_regime", "ens_best_equal"]

st.set_page_config(page_title="Commodity cross-asset research", layout="wide")


@st.cache_data
def load(profile: str):
    met = RESULTS / profile / "metrics"
    out = {"manifest": json.loads((RESULTS / profile / "manifest.json").read_text())}
    for name in ["model_comparison", "by_horizon", "paired_tests", "regime_ic", "time_blocks", "category_ts_ic",
                 "interval_coverage", "backtest_holdout", "timing_diagnostic"]:
        p = met / f"{name}.csv"
        out[name] = pd.read_csv(p) if p.exists() else pd.DataFrame()
    out["daily_ic"] = pd.read_csv(met / "daily_ic.csv", index_col=0)
    out["summary"] = json.loads((met / "summary.json").read_text())
    return out


profiles = [p.name for p in RESULTS.iterdir() if (p / "metrics" / "summary.json").exists()] if RESULTS.exists() else []
if not profiles:
    st.error("No evaluated results found. Run `python -m commodity_research.cli all --profile fast` first.")
    st.stop()

st.sidebar.title("Controls")
profile = st.sidebar.selectbox("Profile", sorted(profiles), index=sorted(profiles).index("fast") if "fast" in profiles else 0)
D = load(profile)
man = D["manifest"]
split = man["split"]
if man.get("is_synthetic"):
    st.warning("SYNTHETIC DATA - these results are not market evidence.")

st.title("Commodity Forecasting & Cross-Asset Quantitative Research")
st.caption("Regime-aware multi-asset return prediction and risk-aware signal analysis. Every number on this page is "
           "read from saved out-of-sample results; nothing is live, simulated or typed by hand.")

tab_task, tab_models, tab_time, tab_regime, tab_risk, tab_fail, tab_vol, tab_gallery = st.tabs(
    ["What is predicted", "Model comparison", "Over time & horizon", "Regimes", "Risk & uncertainty", "Where it fails",
     "Volatility (phase 2)", "Figure gallery"])

with tab_task:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Targets", man["n_targets"])
    c2.metric("Underlying assets", man["n_assets"])
    c3.metric("Horizons (dates)", ", ".join(map(str, man["horizons"])))
    c4.metric("Holdout dates", split["end"] - split["holdout_start"])
    if man["dataset"] == "mitsui":
        st.markdown("""
**Target.** For decision date *t*, target *j* is the log return of asset A from close *t+1* to close *t+1+h*,
minus the same return of asset B (420 pairs) or alone (4 single assets). Every pair has a commodity leg
(LME base metals or JPX gold/platinum) against a US equity/ETF, an FX rate or another commodity.
""")
    else:
        st.markdown(f"**Target.** Forward log returns and return spreads generated from configuration "
                    f"(gap {man['gap']}, horizons {man['horizons']}).")
    st.markdown(f"""
**How forecasts are produced.** Models forecast every asset's return per horizon from 38 causal features
(own momentum/volatility, cross-asset factor exposures, relative value, market regime); pair forecasts are
differences of asset forecasts. Models are refit every {split['refit_every']} dates on an expanding window,
using only labels observable at the refit date.

**Evaluation.** Each date, the 424 forecasts are rank-correlated with the realised labels (Spearman IC);
the competition score is mean(IC)/std(IC). Development period: date_id {split['dev_start']}-{split['holdout_start'] - 1};
final holdout: {split['holdout_start']}-{split['end'] - 1} (used once, after all design choices).
""")
    st.caption(f"Dataset: {man.get('provenance', '')}")

with tab_models:
    comp = D["model_comparison"]
    period = st.radio("Period", ["holdout", "dev"], horizontal=True)
    t = comp[comp.period == period][["model", "score", "mean_ic", "mean_ci_lo", "mean_ci_hi", "t_hac", "frac_pos", "dir_acc", "rank_stability"]]
    st.dataframe(t.sort_values("score", ascending=False).style.format(precision=4), width="stretch", hide_index=True)
    fig = REPORTS / profile / "figures" / "03_model_comparison.png"
    if fig.exists():
        st.image(str(fig))
    st.subheader("Paired tests (same dates & targets)")
    pt = D["paired_tests"]
    st.dataframe(pt[pt.period == period].style.format(precision=4), width="stretch", hide_index=True)

with tab_time:
    models = st.multiselect("Models", list(D["daily_ic"].columns), default=[m for m in ["hist_mean", "ridge", "lgbm", "lstm"] if m in D["daily_ic"].columns])
    window = st.slider("Rolling window (dates)", 21, 252, 126, 21)
    if models:
        st.line_chart(D["daily_ic"][models].rolling(window, min_periods=window // 2).mean())
        st.caption(f"Rolling mean daily rank IC; holdout starts at date_id {split['holdout_start']}.")
    byh = D["by_horizon"]
    uni = st.radio("Universe", sorted(byh.universe.unique()), horizontal=True)
    h = byh[(byh.period == "holdout") & (byh.universe == uni) & byh.model.isin(CORE)]
    st.dataframe(h.pivot_table(index="model", columns="horizon", values="mean_ic").style.format(precision=4), width="stretch")

with tab_regime:
    reg = D["regime_ic"]
    rtype = st.selectbox("Regime variable", sorted(reg.regime_type.unique()))
    rper = st.radio("Dates", ["holdout", "all_oos", "dev"], horizontal=True)
    r = reg[(reg.regime_type == rtype) & (reg.period == rper) & reg.model.isin(CORE)]
    st.dataframe(r.pivot_table(index="model", columns="regime", values="mean_ic").style.format(precision=4), width="stretch")
    st.caption("Counts of dates per regime: " + ", ".join(f"{k}: {int(v)}" for k, v in r.drop_duplicates('regime').set_index('regime').n_dates.items()))
    for f in ["01_regime_timeline.png", "05_vol_regimes_holdout.png"]:
        p = REPORTS / profile / "figures" / f
        if p.exists():
            st.image(str(p))

with tab_risk:
    cov = D["interval_coverage"]
    if not cov.empty:
        st.subheader("Prediction-interval coverage (holdout)")
        st.dataframe(cov.groupby(["model", "level", "subset"])[["coverage_volscaled", "coverage_constant"]].mean().style.format(precision=3), width="stretch")
    dis = D["summary"].get("risk", {}).get("disagreement_holdout")
    if dis:
        st.subheader("Model disagreement vs realised error (holdout)")
        st.json(dis)
    if not D["backtest_holdout"].empty:
        st.subheader("Hypothetical long-short diagnostic (holdout, not a tradable result)")
        st.dataframe(D["backtest_holdout"].style.format(precision=3), width="stretch", hide_index=True)
    p = REPORTS / profile / "figures" / "08_uncertainty.png"
    if p.exists():
        st.image(str(p))

with tab_fail:
    comp_h = D["model_comparison"].query("period == 'holdout'").set_index("model")
    comp_d = D["model_comparison"].query("period == 'dev'").set_index("model")
    both = comp_h.join(comp_d, lsuffix="_holdout", rsuffix="_dev")[["mean_ic_dev", "mean_ic_holdout", "t_hac_holdout"]]
    both["decay"] = both["mean_ic_holdout"] - both["mean_ic_dev"]
    st.markdown("**Development vs holdout mean IC** - how much of the development-period signal survived:")
    st.dataframe(both.sort_values("decay").style.format(precision=4), width="stretch")
    st.markdown("""
Known failure modes documented in `docs/RESEARCH_REPORT.md`:
- signals that look best on the development period decay on the holdout (non-stationarity);
- learned models do not reliably beat a 1-year historical-mean benchmark;
- ensemble disagreement does not flag larger errors; interval coverage falls short of nominal;
- gross paper profits of fast signals disappear at a few basis points of trading cost.
""")


# ----------------------------------------------------------------------------- volatility
VOL = RESULTS / "vol" / "vol"
VOL_FIG = REPORTS / "vol" / "figures"
WINDOW_LABEL = {"k1": "1 day", "cum5": "5 days", "cum21": "21 days"}
VOL_ORDER = ["const", "hv21", "hv63", "ewma", "garch_n", "garch_t", "gjr_t"]


@st.cache_data
def load_vol():
    out = {}
    for name in ["volforecast_summary", "volforecast_panel_dm", "diagnostics_summary", "diagnostics_subperiods",
                 "returns_paired_dev_holdout", "returns_stability_dev_holdout"]:
        p = VOL / f"{name}.csv"
        out[name] = pd.read_csv(p) if p.exists() else pd.DataFrame()
    for prof in ["vol_robust", "vol_rolling"]:
        p = RESULTS / prof / "vol" / "returns_paired_dev_holdout.csv"
        out[prof] = pd.read_csv(p) if p.exists() else pd.DataFrame()
    return out


def _labelled_columns(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t.columns = [f"{p} {WINDOW_LABEL.get(w, w)}" for p, w in t.columns]
    return t


with tab_vol:
    st.markdown(
        "Volatility dynamics and return predictability (phase 2). The hypotheses and decision rules were fixed in "
        "`docs/VOLATILITY_PREREGISTRATION.md` before evaluation. Full write-up: `docs/VOLATILITY_AND_DIAGNOSTICS.md`. "
        "Everything below is read from saved results.")
    V = load_vol()
    if V["volforecast_summary"].empty:
        st.info("Volatility results not found. They ship in `results/vol/`; "
                "regenerate with `python -m commodity_research.cli vol-all --profile vol` (needs the MITSUI data).")
    else:
        st.subheader("1. Which volatility model forecasts best? (out of sample)")
        s = V["volforecast_summary"]
        t = s.pivot_table(index="model", columns=["period", "window"], values="mean_qlike_minus_ewma")
        t = _labelled_columns(t.reindex(VOL_ORDER)[[c for c in t.columns]])
        st.markdown("Mean QLIKE of each model minus EWMA, averaged over 106 assets. **Negative = more accurate than EWMA.**")
        st.dataframe(t.style.format(precision=4), width="stretch")

        dm = V["volforecast_panel_dm"]
        if not dm.empty:
            dm = dm[(dm.assets == "all") & (dm.subset == "all") & (dm.proxy == "sq")]
            tt = dm.pivot_table(index=["hypothesis", "model_a", "model_b"], columns=["period", "window"], values="t_hac")
            tt = _labelled_columns(tt)
            st.markdown("Panel Diebold-Mariano t-statistics (squared-return proxy; negative = model A more accurate). "
                        "Pre-registered hypotheses V1-V4.")
            st.dataframe(tt.style.format(precision=2), width="stretch")
        for fig_name, cap in [("v04_qlike_vs_ewma.png", "QLIKE relative to EWMA"),
                              ("v05_mcs_inclusion.png", "Model Confidence Set inclusion"),
                              ("v06_vol_forecast_paths.png", "21-day forecasts vs realised volatility")]:
            if (VOL_FIG / fig_name).exists():
                st.image(str(VOL_FIG / fig_name), caption=cap)

        st.subheader("2. Does volatility information improve return forecasts?")
        P = V["returns_paired_dev_holdout"]
        if not P.empty:
            pv = P.pivot_table(index="comparison", columns="period", values="mean_diff")
            cols = [c for c in ["dev", "holdout", "holdout_1st_half", "holdout_2nd_half", "pooled_oos"] if c in pv.columns]
            st.markdown("Daily rank-IC difference (model A minus model B) on identical dates and targets.")
            st.dataframe(pv[cols].style.format(precision=4), width="stretch")
            tv = P.pivot_table(index="comparison", columns="period", values="t_hac")
            st.markdown("HAC t-statistics for the same differences.")
            st.dataframe(tv[[c for c in cols if c in tv.columns]].style.format(precision=2), width="stretch")
        st_tab = V["returns_stability_dev_holdout"]
        if not st_tab.empty:
            st.markdown("Accuracy and stability on the holdout (daily rank IC over all targets).")
            hold = st_tab[st_tab.period == "holdout"][["model", "mean_ic", "mean_ci_lo", "mean_ci_hi", "t_hac",
                                                        "std_ic", "worst_rolling126"]]
            st.dataframe(hold.set_index("model").style.format(precision=4), width="stretch")
        for fig_name, cap in [("v07_return_arms.png", "Return arms: daily IC with 95% block-bootstrap CIs"),
                              ("v10_rolling_ic.png", "Rolling 126-date IC over time")]:
            if (VOL_FIG / fig_name).exists():
                st.image(str(VOL_FIG / fig_name), caption=cap)

        st.subheader("3. Robustness of the return result")
        rows = []
        for prof, label in [("vol_robust", "refit every 42 dates, 15-leaf LightGBM"),
                            ("vol_rolling", "rolling 750-date training window")]:
            R = V[prof]
            if not R.empty:
                sel = R[R.comparison.str.startswith(("R1", "R5"))]
                for _, r in sel.iterrows():
                    rows.append({"specification": label, "comparison": r["comparison"], "period": r["period"],
                                 "mean_diff": r["mean_diff"], "t_hac": r["t_hac"]})
        if rows:
            rb = pd.DataFrame(rows)
            rb = rb[rb.period.isin(["dev", "holdout", "pooled_oos"])]
            st.dataframe(rb.set_index(["specification", "comparison", "period"]).style.format(precision=4),
                         width="stretch")
            st.caption("A rolling training window removes the holdout gain; the development gain remains.")

        st.subheader("4. Diagnostics: are volatility dynamics present?")
        D = V["diagnostics_summary"]
        if not D.empty:
            st.dataframe(D[D["sample"].isin(["assets, estimation", "assets, holdout"])]
                         .set_index(["sample", "test"])[["n_series", "share_p<0.05", "share_BH_q0.1"]]
                         .style.format(precision=3), width="stretch")
        sp = V["diagnostics_subperiods"]
        if not sp.empty:
            st.markdown("Return autocorrelation appears only around the 2020 crash: share of assets rejecting "
                        "independence after FDR control.")
            st.dataframe(sp.set_index("window").style.format(precision=3), width="stretch")
        by_market = VOL / "diagnostics_by_market.csv"
        if by_market.exists():
            st.markdown("GARCH(1,1)-t fits by market: tail thickness, persistence, half-life and asymmetry.")
            bm = pd.read_csv(by_market).set_index("market")
            st.dataframe(bm[["n", "exkurt_median", "persistence_median", "half_life_median", "nu_median",
                             "gjr_gamma_median", "share_lr_gjr"]].style.format(precision=3), width="stretch")
        for fig_name, cap in [("v01_acf_returns_vs_squared.png", "ACF of returns vs squared returns (volatility clustering)"),
                              ("v02_qq_plots.png", "Q-Q plots: raw returns, GARCH residuals vs Normal, GARCH residuals vs Student-t"),
                              ("v03_test_rejection_shares.png", "Share of instruments rejecting each diagnostic test (BH-FDR)")]:
            if (VOL_FIG / fig_name).exists():
                st.image(str(VOL_FIG / fig_name), caption=cap)
        st.subheader("5. Regimes and feature groups")
        for fig_name, cap in [("v08_delta_by_market_vol.png", "Gain from volatility features by market-volatility regime"),
                              ("v09_delta_by_target_vol.png", "Gain from volatility features by target volatility"),
                              ("v11_group_importance.png", "LightGBM feature-group importance")]:
            if (VOL_FIG / fig_name).exists():
                st.image(str(VOL_FIG / fig_name), caption=cap)


# ----------------------------------------------------------------------------- figure gallery
CAPTIONS = {
    "01_regime_timeline.png": "Equity volatility and the volatility regimes",
    "02_ic_over_time.png": "Rolling out-of-sample rank correlation",
    "03_model_comparison.png": "Holdout accuracy of every model with 95% CIs",
    "04_by_horizon.png": "Accuracy and stability by horizon",
    "05_vol_regimes_holdout.png": "Accuracy by volatility regime (holdout)",
    "05b_vol_regimes_all_oos.png": "Accuracy by volatility regime (all out-of-sample dates)",
    "06_cross_asset_ablation.png": "Own-asset vs cross-asset features",
    "07_ensembles.png": "Ensembles and online weights",
    "08_uncertainty.png": "Interval coverage and model disagreement",
    "09_backtest_hypothetical.png": "Hypothetical long-short portfolio (paper diagnostic, not tradable)",
    "10_feature_attribution.png": "Feature attribution (Ridge and LightGBM)",
    "v01_acf_returns_vs_squared.png": "ACF of returns vs squared returns",
    "v02_qq_plots.png": "Q-Q plots of returns and GARCH residuals",
    "v03_test_rejection_shares.png": "Diagnostic-test rejection shares",
    "v04_qlike_vs_ewma.png": "Volatility model QLIKE relative to EWMA",
    "v05_mcs_inclusion.png": "Model Confidence Set inclusion",
    "v06_vol_forecast_paths.png": "21-day volatility forecasts vs realised volatility",
    "v07_return_arms.png": "Return models with and without volatility information",
    "v08_delta_by_market_vol.png": "Volatility-feature gain by market regime",
    "v09_delta_by_target_vol.png": "Volatility-feature gain by target volatility",
    "v10_rolling_ic.png": "Rolling accuracy with and without volatility information",
    "v11_group_importance.png": "LightGBM feature-group importance",
}

with tab_gallery:
    folders = sorted(p.parent.name for p in REPORTS.glob("*/figures") if any(p.glob("*.png")))
    labels = {"fast": "fast: return models", "full": "full: return models (3x refits)", "demo_fred_energy": "demo_fred_energy: public energy data",
              "vol": "vol: volatility study (MITSUI)", "vol_fred": "vol_fred: volatility replication (FRED, fresh 2026)",
              "vol_robust": "vol_robust: refit 42 / high-capacity LightGBM", "vol_rolling": "vol_rolling: rolling 750-date window",
              "vol_explore": "vol_explore: exploratory market vs own volatility"}
    default = folders.index("vol") if "vol" in folders else 0
    pick = st.selectbox("Report", folders, index=default, format_func=lambda f: labels.get(f, f))
    st.caption("Every figure is a saved output of this repository's code. The same images are indexed in `docs/FIGURE_GALLERY.md`.")
    imgs = sorted((REPORTS / pick / "figures").glob("*.png"))
    cols = st.columns(2)
    for i, img in enumerate(imgs):
        with cols[i % 2]:
            st.image(str(img), caption=CAPTIONS.get(img.name, img.name))

