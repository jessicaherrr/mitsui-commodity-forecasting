"""Target-level pooled forecasters for the volatility research phase.

One model per horizon, pooled over that horizon's targets (106 for MITSUI) and dates. Each row
is (decision date t, target j) with target-level features (``features.target_features``).

Training label (``scale`` controls the normaliser channel):

    z_{j,t} = clip( y_{j,t} / s_{j,t}, -5, 5 ),   demeaned across the horizon's targets on date t

    s = sigma_EWMA(spread) * sqrt(h)              scale="ewma"    (default, all predictor arms)
    s = sqrt(GARCH-t label-window variance)       scale="garch_t"
    s = constant (median EWMA scale, train dates) scale="none"    (ranking is then on raw returns)

Forecast: yhat_{j,t} = zhat_{j,t} * s_{j,t}, with zhat demeaned within horizon on each date.
"""

from __future__ import annotations

import numpy as np

from ..features.panel import ModelingPanel
from .base import Z_CLIP, FitContext, PanelModel, TrainOnlyScaler, xs_demean

_TF: dict = {}


def get_target_features(panel: ModelingPanel, cfg: dict | None):
    """Target features for this panel (cached per process; built from the cached vol engine)."""
    key = id(panel)
    if key not in _TF:
        from ..experiments.volatility import engine
        from ..features.target_features import build_target_features

        if cfg is None:
            raise ValueError("target-level models need the experiment config to locate the volatility engine")
        prepared = (panel.md, panel.fs, panel)
        spread_cfg = {**cfg, "volatility": {**cfg.get("volatility", {}), "models": ["garch_t"]}}
        _TF.clear()
        _TF[key] = build_target_features(panel, engine(spread_cfg, "spreads", prepared), engine(cfg, "assets", prepared))
    return _TF[key]


class TargetPooledModel(PanelModel):
    level = "target"

    def __init__(self, groups: list[str], estimator: str = "lgbm", scale: str = "ewma", alpha: float = 1e4,
                 lgbm_params: dict | None = None, include: list[str] | None = None, exclude: list[str] | None = None,
                 name: str = "target_model", n_jobs: int = -1, cfg: dict | None = None):
        if estimator not in ("lgbm", "ridge"):
            raise ValueError("estimator must be lgbm or ridge")
        if scale not in ("ewma", "garch_t", "none"):
            raise ValueError("scale must be ewma, garch_t or none")
        self.groups, self.estimator, self.scale, self.alpha = list(groups), estimator, scale, alpha
        self.include, self.exclude = list(include or []), list(exclude or [])
        self.lgbm_params, self.name, self.n_jobs, self.cfg = dict(lgbm_params or {}), name, n_jobs, cfg

    # ----------------------------------------------------------------- helpers
    def _scale(self, tf, dates):
        if self.scale == "none":
            return np.full((len(dates), tf.values.shape[1]), self.const_)
        return tf.scales[self.scale][dates]

    def _make(self, seed: int):
        if self.estimator == "ridge":
            from sklearn.linear_model import Ridge

            return Ridge(alpha=self.alpha)
        import sys

        import lightgbm as lgb

        p = {"n_estimators": 200, "learning_rate": 0.03, "num_leaves": 7, "min_child_samples": 2000,
             "subsample": 0.7, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 5.0}
        p.update(self.lgbm_params)
        n_jobs = 1 if "torch" in sys.modules else self.n_jobs
        return lgb.LGBMRegressor(**p, random_state=seed, n_jobs=n_jobs, deterministic=True, force_row_wise=True, verbose=-1)

    # --------------------------------------------------------------------- fit
    def fit(self, panel: ModelingPanel, ctx: FitContext) -> None:
        tf = get_target_features(panel, self.cfg)
        self.mask_ = tf.mask(self.groups)
        unknown = set(self.include + self.exclude) - set(tf.names)
        if unknown:
            raise ValueError(f"unknown features {unknown}")
        names = np.array(tf.names)
        self.mask_ = (self.mask_ | np.isin(names, self.include)) & ~np.isin(names, self.exclude)
        F = int(self.mask_.sum())
        d1 = ctx.dates(panel, min(panel.horizons))
        self.scaler_ = TrainOnlyScaler().fit(tf.values[d1][:, :, self.mask_].reshape(-1, F))
        self.const_ = float(np.nanmedian(tf.scales["ewma"][d1]))
        self.models_ = {}
        for k, h in enumerate(panel.horizons):
            cols = np.flatnonzero(panel.h_idx == k)
            dates = ctx.dates(panel, h)
            y = tf.labels[np.ix_(dates, cols)]
            s = self._scale(tf, dates)[:, cols]
            valid = np.isfinite(y) & np.isfinite(s) & (s > 0)
            z = np.clip(np.where(valid, y, 0.0) / np.where(valid, s, 1.0), -Z_CLIP, Z_CLIP)
            z = xs_demean(z, valid)
            di, jj = np.nonzero(valid)
            if len(di) < 100:
                continue
            X = self.scaler_.transform(tf.values[dates[di], cols[jj]][:, self.mask_])
            model = self._make(ctx.seed + k)
            model.fit(X, z[di, jj])
            self.models_[k] = model

    # ----------------------------------------------------------------- predict
    def predict(self, panel: ModelingPanel, dates: np.ndarray) -> np.ndarray:
        tf = get_target_features(panel, self.cfg)
        J = len(panel.targets)
        out = np.zeros((len(dates), J))
        s = self._scale(tf, dates)
        for k, model in self.models_.items():
            cols = np.flatnonzero(panel.h_idx == k)
            X = self.scaler_.transform(tf.values[dates][:, cols][:, :, self.mask_].reshape(-1, int(self.mask_.sum())))
            zhat = model.predict(X).reshape(len(dates), len(cols))
            zhat = zhat - zhat.mean(axis=1, keepdims=True)
            out[:, cols] = zhat * np.nan_to_num(s[:, cols])
        return out

    def importance(self) -> dict[int, np.ndarray]:
        if self.estimator == "ridge":
            return {k: m.coef_ for k, m in self.models_.items()}
        return {k: m.booster_.feature_importance("gain") for k, m in self.models_.items()}
