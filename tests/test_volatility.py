"""Tests for the volatility research module: correctness of forecasts, losses, tests and causality."""

import copy
import warnings

import numpy as np
import pandas as pd
import pytest

from commodity_research.volatility import diagnostics as D
from commodity_research.volatility import evaluation as E
from commodity_research.volatility.models import GARCH_SPECS, VolEngineConfig, forecast_series

warnings.filterwarnings("ignore")


def simulate_garch(n=1500, w=0.02, a=0.08, b=0.9, nu=7.0, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu)
    s2 = w / (1 - a - b)
    x = np.empty(n)
    for i in range(n):
        x[i] = np.sqrt(s2) * z[i]
        s2 = w + a * x[i] ** 2 + b * s2
    return x / 100  # decimal returns


CFG = VolEngineConfig(refit_every=50, min_obs=300, k_max=10)


def test_forecasts_are_causal():
    r = simulate_garch(900)
    r[[100, 400, 401]] = np.nan  # non-trading days
    sf = forecast_series(r, CFG)
    r2 = r.copy()
    rng = np.random.default_rng(1)
    t0 = 600
    future = np.arange(len(r)) > t0
    r2[future] = r2[future] * rng.uniform(0.1, 5, future.sum())
    sf2 = forecast_series(r2, CFG)
    cut = np.searchsorted(sf.pos, t0, side="right")
    for m in CFG.models:
        np.testing.assert_array_equal(np.isnan(sf.f[m][:cut]), np.isnan(sf2.f[m][:cut]))
        np.testing.assert_allclose(np.nan_to_num(sf.f[m][:cut]), np.nan_to_num(sf2.f[m][:cut]), rtol=0, atol=1e-15)


def test_historical_and_ewma_match_reference():
    r = simulate_garch(400, seed=3)
    sf = forecast_series(r, VolEngineConfig(models=["const", "hv21", "ewma"], k_max=3))
    x2 = (r * 100) ** 2
    hv = pd.Series(x2).rolling(21).mean().to_numpy() / 1e4
    np.testing.assert_allclose(sf.f["hv21"][20:, 0], hv[20:], rtol=1e-12)
    np.testing.assert_allclose(sf.f["const"][19:, 0], (np.cumsum(x2) / np.arange(1, 401))[19:] / 1e4, rtol=1e-12)
    s = x2[:20].mean()
    ref = [s]
    for v in x2[20:]:
        s = 0.94 * s + 0.06 * v
        ref.append(s)
    np.testing.assert_allclose(sf.f["ewma"][19:, 0], np.array(ref) / 1e4, rtol=1e-12)
    assert np.all(sf.f["ewma"][19:, 0] == sf.f["ewma"][19:, 2])  # flat term structure


@pytest.mark.parametrize("model", ["garch_t", "gjr_t"])
def test_multistep_matches_arch_analytic_forecast(model):
    """At a refit day the engine's k-step forecasts equal arch's own analytic forecasts."""
    from arch import arch_model

    r = simulate_garch(800, seed=5)
    cfg = VolEngineConfig(models=[model], refit_every=10_000, min_obs=800, k_max=5)  # one fit on all data
    sf = forecast_series(r, cfg)
    res = arch_model(r * 100, mean="Zero", **GARCH_SPECS[model]).fit(disp="off")
    ref = res.forecast(horizon=5, reindex=False).variance.to_numpy()[-1] / 1e4
    np.testing.assert_allclose(sf.f[model][-1], ref, rtol=1e-6)


def test_qlike_valid_for_zero_proxy_and_ranks_true_variance_best():
    rng = np.random.default_rng(0)
    true = rng.uniform(0.5, 2.0, 5000)
    proxy = true * rng.chisquare(1, 5000)  # E[proxy | true] = true (unbiased, noisy)
    assert np.isfinite(E.qlike(np.array([0.0]), np.array([1.0]))).all()
    l_true = E.qlike(proxy, true).mean()
    l_flat = E.qlike(proxy, np.full(5000, true.mean())).mean()
    l_biased = E.qlike(proxy, true * 1.5).mean()
    assert l_true < l_flat and l_true < l_biased


def test_dm_and_mcs_identify_better_forecast():
    rng = np.random.default_rng(1)
    base = rng.normal(1.0, 0.3, 600)
    losses = pd.DataFrame({"good": base, "bad": base + 0.2 + rng.normal(0, 0.05, 600), "worse": base + 0.5})
    mean, t, p, n = E.dm_test((losses["good"] - losses["bad"]).to_numpy(), 5)
    assert t < -5 and p < 1e-4
    assert E.mcs_included(losses, reps=200) == ["good"]


def test_mincer_zarnowitz_unbiased_forecast():
    rng = np.random.default_rng(2)
    f = rng.uniform(1, 3, 3000)
    s = f * rng.chisquare(1, 3000)
    mz = E.mincer_zarnowitz(s, f, 5)
    assert abs(mz["mz_b"] - 1) < 0.2 and mz["mz_p_unbiased"] > 0.01


def test_garman_klass_rejects_inconsistent_bars():
    o, h, l, c = (np.array(v, float) for v in ([10, 10], [11, 9], [9, 9.5], [10.5, 10]))
    gk = E.garman_klass(o, h, l, c)
    assert np.isfinite(gk[0]) and gk[0] > 0 and np.isnan(gk[1])  # high < open is invalid


def test_window_proxy_skips_non_trading_rows():
    daily = np.array([1.0, np.nan, 2.0, 3.0, 4.0])
    traded = np.isfinite(daily)
    w = E.window_proxy_rows(daily, traded, 1, 2)
    assert w[0] == 5.0 and np.isnan(w[1]) and w[2] == 7.0 and np.isnan(w[4])


def test_diagnostics_detect_arch_effects_and_bh():
    x = simulate_garch(2000, seed=7)
    f = D.stylised_facts(x)
    assert f["arch_lm_p"] < 0.01 and f["lb_r2_10_p"] < 0.01 and f["jb_p"] < 0.01
    iid = np.random.default_rng(3).normal(0, 0.01, 2000)
    assert D.stylised_facts(iid)["arch_lm_p"] > 0.001
    from statsmodels.stats.multitest import multipletests

    p = np.random.default_rng(4).uniform(0, 0.2, 60)
    np.testing.assert_array_equal(D.bh_fdr(p, 0.1), multipletests(p, 0.1, "fdr_bh")[0])


def test_sign_bias_detects_leverage():
    rng = np.random.default_rng(8)
    n = 4000
    z = np.empty(n)
    z[0] = rng.normal()
    for i in range(1, n):  # variance larger after negative shocks: neglected asymmetry
        z[i] = rng.normal() * (1.5 if z[i - 1] < 0 else 0.7)
    assert D.sign_bias_test(z)["joint_p"] < 0.01


# ------------------------------------------------------------------- target-level features
def _engines(panel):
    from commodity_research.volatility.engine import run_engine, series_from_panel

    cfg = VolEngineConfig(models=["ewma", "garch_t"], refit_every=40, min_obs=120)
    assets, spreads = series_from_panel(panel)
    return (run_engine(spreads, VolEngineConfig(models=["garch_t"], refit_every=40, min_obs=120), panel.gap, panel.horizons, n_jobs=1),
            run_engine(assets, cfg, panel.gap, panel.horizons, n_jobs=1))


@pytest.mark.slow
def test_target_features_are_causal(synth):
    from commodity_research.features import build_features, build_panel
    from commodity_research.features.target_features import build_target_features

    from .conftest import perturb_future

    md, fs, panel = synth
    tf = build_target_features(panel, *_engines(panel))
    t0 = 400
    md2 = perturb_future(md, t0)
    fs2 = build_features(md2)
    p2 = build_panel(md2, fs2)
    tf2 = build_target_features(p2, *_engines(p2))
    a, b = tf.values[: t0 + 1], tf2.values[: t0 + 1]
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_allclose(np.nan_to_num(a), np.nan_to_num(b), rtol=0, atol=1e-6)
    for k in ("ewma", "garch_t"):
        np.testing.assert_allclose(np.nan_to_num(tf.scales[k][: t0 + 1]), np.nan_to_num(tf2.scales[k][: t0 + 1]), atol=1e-12)
    assert set(tf.groups) == {"R", "Vn", "Vs", "C"}


@pytest.mark.slow
@pytest.mark.parametrize("estimator,scale", [("lgbm", "ewma"), ("ridge", "garch_t")])
def test_target_model_ignores_unobservable_labels(synth, estimator, scale):
    from commodity_research.features.target_features import build_target_features
    from commodity_research.models import FitContext
    from commodity_research.models import target_models as TM

    md, fs, panel = synth
    tf = build_target_features(panel, *_engines(panel))
    tau = 420
    ctx = FitContext(tau=tau, first_train=130, feature_mask=None, seed=0)
    params = {"lgbm_params": {"n_estimators": 20, "min_child_samples": 50}} if estimator == "lgbm" else {}
    TM._TF.clear()
    TM._TF[id(panel)] = tf
    m1 = TM.TargetPooledModel(["R", "Vn", "Vs"], estimator=estimator, scale=scale, n_jobs=1, **params)
    m1.fit(panel, ctx)
    p1 = m1.predict(panel, np.arange(tau, tau + 10))
    tf2 = copy.copy(tf)
    lab = tf.labels.copy()
    for k, h in enumerate(panel.horizons):
        cols = panel.h_idx == k
        lab[tau - panel.gap - h + 1:, cols] = np.random.default_rng(0).normal(0, 0.1, lab[tau - panel.gap - h + 1:, cols].shape)
    tf2.labels = lab
    TM._TF.clear()
    TM._TF[id(panel)] = tf2
    m2 = TM.TargetPooledModel(["R", "Vn", "Vs"], estimator=estimator, scale=scale, n_jobs=1, **params)
    m2.fit(panel, ctx)
    np.testing.assert_allclose(p1, m2.predict(panel, np.arange(tau, tau + 10)), rtol=0, atol=1e-10)
    TM._TF.clear()


@pytest.mark.slow
def test_target_model_include_exclude(synth):
    from commodity_research.features.target_features import build_target_features
    from commodity_research.models import FitContext
    from commodity_research.models import target_models as TM

    md, fs, panel = synth
    tf = build_target_features(panel, *_engines(panel))
    TM._TF.clear()
    TM._TF[id(panel)] = tf
    ctx = FitContext(tau=420, first_train=130, feature_mask=None, seed=0)
    m = TM.TargetPooledModel(["R"], estimator="ridge", include=["mkt_vol_pct"], n_jobs=1)
    m.fit(panel, ctx)
    chosen = {n for n, k in zip(tf.names, m.mask_) if k}
    assert "mkt_vol_pct" in chosen and all(tf.groups[tf.names.index(n)] == "R" for n in chosen - {"mkt_vol_pct"})
    m2 = TM.TargetPooledModel(["R", "Vs"], estimator="ridge", exclude=["mkt_vol_pct", "mkt_vol_chg"], n_jobs=1)
    m2.fit(panel, ctx)
    chosen2 = {n for n, k in zip(tf.names, m2.mask_) if k}
    assert not chosen2 & {"mkt_vol_pct", "mkt_vol_chg"} and "log_garch_lw" in chosen2
    with pytest.raises(ValueError):
        TM.TargetPooledModel(["R"], include=["no_such_feature"], n_jobs=1).fit(panel, ctx)
    TM._TF.clear()
