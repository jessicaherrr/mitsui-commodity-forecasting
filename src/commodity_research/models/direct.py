"""Direct target-level comparators for the asset-factorised design.

direct_pooled   one ridge per horizon, pooled over all targets of that horizon, trained on
                pair features (leg-A minus leg-B own features + log-price-ratio z-score and
                relative momentum) and the target label itself
direct_target   one ridge per target (424 small regressions per refit), same features

Both train on vol-scaled *target* labels recomputed from prices (identical to the official
labels inside the sample) with the same horizon-specific availability rule.  With
``label_mode="xs_demean"`` the labels (and forecasts) are demeaned across the targets of the
same horizon on each date, mirroring the asset-level models.
"""

from __future__ import annotations

import numpy as np

from ..features.panel import ModelingPanel, build_pair_features, target_level_labels
from .base import LABEL_MODES, Z_CLIP, FitContext, PanelModel, TrainOnlyScaler, xs_demean


class _PairCache:
    """Pair features are deterministic functions of the panel; compute once per panel."""

    def __init__(self):
        self.key = None

    def get(self, panel: ModelingPanel, mode: str):
        if self.key != (id(panel), mode):
            self.X, self.names, self.tscale = build_pair_features(panel)
            y = target_level_labels(panel)
            valid = np.isfinite(y) & (self.tscale > 0)
            z = np.clip(np.where(valid, y, 0.0) / np.where(valid, self.tscale, 1.0), -Z_CLIP, Z_CLIP)
            z = np.where(valid, z, np.nan)
            if mode == "xs_demean":
                for k in np.unique(panel.h_idx):
                    cols = panel.h_idx == k
                    z[:, cols] = xs_demean(z[:, cols], valid[:, cols])
            self.z, self.valid = z, valid
            self.key = (id(panel), mode)
        return self


class DirectRidge(PanelModel):
    level = "target"

    def __init__(self, alpha: float = 100.0, pooled: bool = True, label_mode: str = "xs_demean",
                 name: str | None = None):
        if label_mode not in LABEL_MODES:
            raise ValueError(f"label mode must be one of {LABEL_MODES}")
        self.alpha, self.pooled, self.label_mode = alpha, pooled, label_mode
        self.name = name or ("ridge_direct_pooled" if pooled else "ridge_direct_target")
        self._cache = _PairCache()

    def _rows(self, c, dates, j_sel):
        di, jj = np.nonzero(c.valid[np.ix_(dates, j_sel)])
        t, j = dates[di], np.asarray(j_sel)[jj]
        return t, j, c.z[t, j]

    def fit(self, panel: ModelingPanel, ctx: FitContext) -> None:
        c = self._cache.get(panel, self.label_mode)
        J, F = c.X.shape[1], c.X.shape[2]
        d1 = ctx.dates(panel, min(panel.horizons))
        self.scaler_ = TrainOnlyScaler().fit(c.X[d1].reshape(-1, F))
        self.coef_ = np.zeros((J, F))
        self.intercept_ = np.zeros(J)
        groups = (
            [np.flatnonzero(panel.h_idx == k) for k in range(len(panel.horizons))]
            if self.pooled else [np.array([j]) for j in range(J)]
        )
        for j_sel in groups:
            h = panel.horizons[panel.h_idx[j_sel[0]]]
            t, j, z = self._rows(c, ctx.dates(panel, h), j_sel)
            if len(z) < F + 5:
                continue
            X = self.scaler_.transform(c.X[t, j])
            beta, b0 = _ridge(X, z, self.alpha)
            self.coef_[j_sel] = beta
            self.intercept_[j_sel] = b0

    def predict(self, panel: ModelingPanel, dates: np.ndarray) -> np.ndarray:
        c = self._cache.get(panel, self.label_mode)
        D, J, F = len(dates), c.X.shape[1], c.X.shape[2]
        X = self.scaler_.transform(c.X[dates].reshape(-1, F)).reshape(D, J, F)
        z = np.einsum("djf,jf->dj", X, self.coef_) + self.intercept_[None, :]
        if self.label_mode == "xs_demean":
            for k in np.unique(panel.h_idx):
                cols = panel.h_idx == k
                z[:, cols] -= z[:, cols].mean(axis=1, keepdims=True)
        return z * np.nan_to_num(c.tscale[dates])


def _ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Closed-form ridge with an unpenalised intercept."""
    xm, ym = X.mean(axis=0), y.mean()
    Xc, yc = X - xm, y - ym
    A = Xc.T @ Xc + alpha * np.eye(X.shape[1])
    beta = np.linalg.solve(A, Xc.T @ yc)
    return beta, float(ym - xm @ beta)
