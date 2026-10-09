"""Pooled tabular forecasters (one model per horizon, pooled over assets and dates).

Training objective: squared error on volatility-scaled, cross-sectionally demeaned labels

    z_{i,t,h} = clip(r_{i,t,h} / (sigma_{i,t} * sqrt(h)), -5, 5)
    ztilde_{i,t,h} = z_{i,t,h} - mean_{j active at t} z_{j,t,h}       (label_mode="xs_demean")

where sigma_{i,t} is the causal EWMA volatility known at t.  Forecasts are demeaned across
assets in the same way and mapped back to return units with the same (known-at-t) scale:
rhat = zhat * sigma_{i,t} * sqrt(h).  With demeaned labels, purely date-level features
(regime variables) can only act through interactions (trees/LSTM), not as linear tilts.  Vol
scaling stops high-volatility equities from dominating the pooled loss and makes the
prediction target comparable across FX, metals and equities.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from ..features.panel import ModelingPanel
from .base import FitContext, PanelModel, finalize_z, fit_scaler, gather_rows


class _PooledTabular(PanelModel):
    level = "asset"
    label_mode = "xs_demean"

    def _make(self, seed: int):
        raise NotImplementedError

    def fit(self, panel: ModelingPanel, ctx: FitContext) -> None:
        self.mask_ = ctx.feature_mask
        self.scaler_ = fit_scaler(panel, ctx)
        self.models_ = []
        for k, h in enumerate(panel.horizons):
            t, i, z = gather_rows(panel, ctx.dates(panel, h), k, self.label_mode)
            X = self.scaler_.transform(panel.fs.values[t, i][:, self.mask_])
            model = self._make(ctx.seed + k)
            model.fit(X, z)
            self.models_.append(model)

    def predict(self, panel: ModelingPanel, dates: np.ndarray) -> np.ndarray:
        n_d, n_a = len(dates), panel.n_assets
        X = self.scaler_.transform(panel.fs.values[dates][:, :, self.mask_].reshape(n_d * n_a, -1))
        out = np.empty((n_d, n_a, len(panel.horizons)))
        for k, model in enumerate(self.models_):
            zhat = finalize_z(model.predict(X).reshape(n_d, n_a), panel.fs.active[dates], self.label_mode)
            out[:, :, k] = zhat * panel.scale[dates, :, k]
        return out


class RidgeModel(_PooledTabular):
    def __init__(self, alpha: float = 100.0, label_mode: str = "xs_demean", name: str = "ridge"):
        self.alpha, self.label_mode, self.name = alpha, label_mode, name

    def _make(self, seed):
        return Ridge(alpha=self.alpha)

    def coefficients(self) -> np.ndarray:
        return np.stack([m.coef_ for m in self.models_])  # (H, F)


class LightGBMModel(_PooledTabular):
    def __init__(self, params: dict | None = None, name: str = "lgbm", n_jobs: int = -1):
        params = dict(params or {})
        self.label_mode = params.pop("label_mode", "xs_demean")
        self.params, self.name, self.n_jobs = params, name, n_jobs

    def _make(self, seed):
        import sys

        import lightgbm as lgb

        n_jobs = self.n_jobs
        if "torch" in sys.modules and n_jobs != 1:
            # Two OpenMP runtimes (torch's and LightGBM's) in one process crash on macOS;
            # single-threaded LightGBM is safe. The experiment runner isolates models in
            # separate processes, so this only triggers in mixed interactive/test sessions.
            n_jobs = 1
        p = {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "min_child_samples": 500,
            "subsample": 0.7,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 5.0,
        }
        p.update(self.params)
        return lgb.LGBMRegressor(
            **p, random_state=seed, n_jobs=n_jobs, deterministic=True, force_row_wise=True, verbose=-1
        )

    def feature_importance(self) -> np.ndarray:
        return np.stack([m.booster_.feature_importance("gain") for m in self.models_])
