import numpy as np
import pandas as pd
import pytest
import torch

from commodity_research.models import FitContext, ForecastCombiner, make_model
from commodity_research.models.lstm import LSTMNet, make_sequences


def test_lstm_output_shape_and_input_check():
    net = LSTMNet(n_features=7, n_outputs=4, hidden=8)
    out = net(torch.randn(16, 20, 7))
    assert out.shape == (16, 4)
    with pytest.raises(ValueError):
        net(torch.randn(16, 7))


def test_sequences_only_use_past_rows():
    T, N, F = 50, 3, 2
    Xs = np.zeros((T, N, F), dtype=np.float32)
    Xs[:, :, 0] = np.arange(T)[:, None]  # feature 0 = row index
    Xs[:, :, 1] = np.arange(N)[None, :]  # feature 1 = asset index
    t = np.array([10, 30, 49])
    i = np.array([0, 2, 1])
    seq = make_sequences(Xs, t, i, window=6)
    assert seq.shape == (3, 6, 2)
    assert (seq[:, -1, 0] == t).all()  # last step is the decision date
    assert (seq[:, :, 0] <= t[:, None]).all()  # never a future row
    assert (seq[:, :, 1] == i[:, None]).all()  # never another asset


def _ctx(panel, seed):
    return FitContext(tau=420, first_train=130, feature_mask=np.ones(len(panel.fs.names), bool), seed=seed)


@pytest.mark.parametrize("kind,params", [
    ("ridge", {"alpha": 10.0}),
    ("lgbm", {"n_estimators": 30, "num_leaves": 7, "min_child_samples": 50}),
    ("lstm", {"window": 5, "hidden": 8, "max_epochs": 2, "max_samples": 2000, "val_len": 40, "batch_size": 256}),
])
def test_same_seed_same_predictions(synth, kind, params):
    _, _, panel = synth
    dates = np.arange(420, 440)
    preds = []
    for _ in range(2):
        m = make_model(kind, kind, params, n_jobs=1)
        m.fit(panel, _ctx(panel, seed=123))
        preds.append(m.predict(panel, dates))
    np.testing.assert_allclose(preds[0], preds[1], rtol=0, atol=1e-7)
    assert preds[0].shape == (len(dates), panel.n_assets, len(panel.horizons))
    assert np.isfinite(preds[0]).all()


def test_cross_sectional_forecasts_are_demeaned(synth):
    _, _, panel = synth
    m = make_model("ridge", "ridge", {"alpha": 10.0})
    m.fit(panel, _ctx(panel, 0))
    dates = np.arange(420, 430)
    z = m.predict(panel, dates) / panel.scale[dates]
    act = panel.fs.active[dates]
    for k in range(z.shape[2]):
        means = [z[d, act[d], k].mean() for d in range(len(dates))]
        np.testing.assert_allclose(means, 0, atol=1e-8)


def test_target_composition(synth):
    _, _, panel = synth
    rng = np.random.default_rng(0)
    p = rng.normal(size=(panel.n_dates, panel.n_assets, len(panel.horizons)))
    tp = panel.to_targets(p)
    for j in range(len(panel.targets)):
        a, b, k = panel.leg_a[j], panel.leg_b[j], panel.h_idx[j]
        expected = p[:, a, k] - (p[:, b, k] if b >= 0 else 0)
        np.testing.assert_allclose(tp[:, j], expected)


def test_online_ensemble_weights_only_use_observable_ics():
    T, N, H = 300, 4, 2
    rng = np.random.default_rng(0)
    preds = {m: rng.normal(size=(T, N, H)) for m in ("a", "b")}
    ic = {"a": rng.normal(0.05, 0.1, T), "b": rng.normal(0.0, 0.1, T)}
    dev = np.arange(50, 200)
    comb = ForecastCombiner(["a", "b"], "online", gap=1, h_max=2, lookback=60).fit(preds, ic, dev)
    s = 250
    w1 = comb.weights(s)
    ic2 = {k: v.copy() for k, v in ic.items()}
    for v in ic2.values():
        v[s - 1 - 2 + 1:] = 10.0  # ICs not yet observable at s
    comb2 = ForecastCombiner(["a", "b"], "online", gap=1, h_max=2, lookback=60).fit(preds, ic2, dev)
    np.testing.assert_array_equal(w1, comb2.weights(s))
    assert np.isclose(w1.sum(), 1.0)


def test_regime_weights_fallback_and_equal():
    T, N, H = 200, 3, 1
    rng = np.random.default_rng(1)
    preds = {m: rng.normal(size=(T, N, H)) for m in ("a", "b", "c")}
    ic = {m: rng.normal(0.02, 0.1, T) for m in preds}
    reg = pd.Series(np.where(np.arange(T) % 2 == 0, "low", "high"), dtype=object)
    reg.iloc[150] = None
    comb = ForecastCombiner(list(preds), "regime").fit(preds, ic, np.arange(20, 120), reg)
    assert set(comb.regime_w_) == {"low", "high"}
    np.testing.assert_array_equal(comb.weights(150), comb.static_w_)  # unknown regime -> static
    eq = ForecastCombiner(list(preds), "equal").fit(preds, ic, np.arange(20, 120))
    np.testing.assert_allclose(eq.weights(10), 1 / 3)
