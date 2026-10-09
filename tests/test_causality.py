"""Anti-leakage tests: future information must never change earlier features or forecasts."""

import numpy as np
import pytest

from commodity_research.features import build_features, build_panel
from commodity_research.models import FitContext, make_model
from commodity_research.models.base import fit_scaler

from .conftest import HAS_MITSUI, perturb_future


def _assert_same_until(a, b, t0):
    np.testing.assert_array_equal(np.isnan(a[: t0 + 1]), np.isnan(b[: t0 + 1]))
    np.testing.assert_allclose(np.nan_to_num(a[: t0 + 1]), np.nan_to_num(b[: t0 + 1]), rtol=0, atol=1e-6)


@pytest.mark.parametrize("t0", [200, 350])
def test_features_unchanged_by_future_prices_synthetic(synth, t0):
    md, fs, _ = synth
    fs2 = build_features(perturb_future(md, t0))
    _assert_same_until(fs.values, fs2.values, t0)
    _assert_same_until(fs.vol, fs2.vol, t0)
    num = fs.regime.select_dtypes("number").to_numpy()
    num2 = fs2.regime.select_dtypes("number").to_numpy()
    _assert_same_until(num, num2, t0)
    assert (fs.active[: t0 + 1] == fs2.active[: t0 + 1]).all()


@pytest.mark.realdata
@pytest.mark.skipif(not HAS_MITSUI, reason="MITSUI data not available")
def test_features_unchanged_by_future_prices_mitsui(mitsui):
    md, fs, _ = mitsui
    t0 = 1200
    fs2 = build_features(perturb_future(md, t0))
    _assert_same_until(fs.values, fs2.values, t0)
    _assert_same_until(fs.regime.select_dtypes("number").to_numpy(), fs2.regime.select_dtypes("number").to_numpy(), t0)


def _fit_predict(panel, kind, params, tau, dates, seed=0):
    m = make_model(kind, kind, params, n_jobs=1)
    m.fit(panel, FitContext(tau=tau, first_train=130, feature_mask=np.ones(len(panel.fs.names), bool), seed=seed))
    return m.predict(panel, dates)


MODELS = [
    ("hist_mean", {}),
    ("reversal", {}),
    ("ridge", {"alpha": 100.0}),
    ("lgbm", {"n_estimators": 20, "num_leaves": 7, "min_child_samples": 50}),
    ("direct_ridge", {"alpha": 10.0, "pooled": True}),
    ("direct_ridge", {"alpha": 10.0, "pooled": False}),
    ("lstm", {"window": 5, "hidden": 8, "max_epochs": 2, "max_samples": 2000, "val_len": 40, "batch_size": 256}),
]


@pytest.mark.parametrize("kind,params", MODELS)
def test_predictions_unchanged_by_future_data(synth, kind, params):
    """Model fitted at tau and predicting dates <= t0: perturbing prices after t0 changes nothing."""
    md, _, panel = synth
    tau, t0 = 400, 430
    dates = np.arange(tau, t0 + 1)
    md2 = perturb_future(md, t0)
    fs2 = build_features(md2)
    panel2 = build_panel(md2, fs2)
    p1 = _fit_predict(panel, kind, params, tau, dates)
    p2 = _fit_predict(panel2, kind, params, tau, dates)
    np.testing.assert_allclose(p1, p2, rtol=0, atol=1e-6)


@pytest.mark.parametrize("kind,params", [m for m in MODELS if m[0] not in ("hist_mean", "reversal")])
def test_unobservable_labels_are_not_used(synth, kind, params):
    """Labels whose information interval ends after tau must not influence the fit at tau."""
    md, fs, panel = synth
    tau = 400
    p1 = _fit_predict(panel, kind, params, tau, np.arange(tau, tau + 10))
    import copy

    panel2 = copy.copy(panel)
    labels = panel.labels.copy()
    rng = np.random.default_rng(1)
    for k, h in enumerate(panel.horizons):
        last_ok = tau - panel.gap - h  # last observable decision date for this horizon
        labels[last_ok + 1:, :, k] = rng.normal(0, 0.05, labels[last_ok + 1:, :, k].shape)
    panel2.labels = labels
    p2 = _fit_predict(panel2, kind, params, tau, np.arange(tau, tau + 10))
    np.testing.assert_allclose(p1, p2, rtol=0, atol=1e-6)


def test_observable_labels_are_used(synth):
    """Sanity check of the previous test: changing an *observable* label does change the fit."""
    import copy

    md, fs, panel = synth
    tau = 400
    p1 = _fit_predict(panel, "ridge", {"alpha": 1.0}, tau, np.arange(tau, tau + 5))
    panel2 = copy.copy(panel)
    labels = panel.labels.copy()
    labels[200:300] *= -3.0
    panel2.labels = labels
    p2 = _fit_predict(panel2, "ridge", {"alpha": 1.0}, tau, np.arange(tau, tau + 5))
    assert not np.allclose(p1, p2)


def test_scaler_ignores_test_period_features(synth):
    md, fs, panel = synth
    import copy

    ctx = FitContext(tau=300, first_train=130, feature_mask=np.ones(len(fs.names), bool))
    s1 = fit_scaler(panel, ctx)
    panel2 = copy.copy(panel)
    fs2 = copy.copy(fs)
    vals = fs.values.copy()
    vals[ctx.tau - panel.gap - min(panel.horizons) + 1:] = 1e6  # everything at/after the test start
    fs2.values = vals
    panel2.fs = fs2
    s2 = fit_scaler(panel2, ctx)
    np.testing.assert_array_equal(s1.mean_, s2.mean_)
    np.testing.assert_array_equal(s1.std_, s2.std_)
