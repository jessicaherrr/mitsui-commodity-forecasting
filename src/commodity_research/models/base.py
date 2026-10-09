"""Common model interface, train-only preprocessing and panel row gathering."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..features.panel import ModelingPanel
from ..validation import train_dates

Z_CLIP = 5.0


@dataclass
class FitContext:
    """Everything a model may know when it is fitted at the close of ``tau``."""

    tau: int
    first_train: int
    feature_mask: np.ndarray
    seed: int = 42
    window: int | None = None  # None = expanding window
    extra: dict = field(default_factory=dict)

    def dates(self, panel: ModelingPanel, h: int) -> np.ndarray:
        return train_dates(self.tau, h, panel.gap, self.first_train, self.window)


class TrainOnlyScaler:
    """Winsorise + standardise each feature with statistics from training rows only.

    Missing values are imputed with 0 *after* standardisation, i.e. with the training mean.
    """

    def __init__(self, lower_q: float = 0.5, upper_q: float = 99.5):
        self.lower_q, self.upper_q = lower_q, upper_q

    def fit(self, X: np.ndarray) -> "TrainOnlyScaler":
        X = np.asarray(X, dtype=np.float64)
        self.lo_ = np.nanpercentile(X, self.lower_q, axis=0)
        self.hi_ = np.nanpercentile(X, self.upper_q, axis=0)
        Xc = np.clip(X, self.lo_, self.hi_)
        self.mean_ = np.nan_to_num(np.nanmean(Xc, axis=0))
        sd = np.nanstd(Xc, axis=0)
        self.std_ = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
        self.lo_ = np.nan_to_num(self.lo_, nan=-np.inf)
        self.hi_ = np.nan_to_num(self.hi_, nan=np.inf)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (np.clip(X, self.lo_, self.hi_) - self.mean_) / self.std_
        return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fit_scaler(panel: ModelingPanel, ctx: FitContext) -> TrainOnlyScaler:
    """Fit on features of active assets at the training decision dates (shortest horizon)."""
    dates = ctx.dates(panel, min(panel.horizons))
    X = panel.fs.values[dates][:, :, ctx.feature_mask]
    act = panel.fs.active[dates]
    return TrainOnlyScaler().fit(X[act])


LABEL_MODES = ("xs_demean", "raw")


def xs_demean(z: np.ndarray, valid: np.ndarray, axis: int = 1) -> np.ndarray:
    """Subtract the cross-sectional mean of the valid entries (other entries become NaN)."""
    z = np.where(valid, z, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cross-sections -> mean 0
        mu = np.nanmean(z, axis=axis, keepdims=True)
    return z - np.nan_to_num(mu)


def vol_scaled_labels(panel: ModelingPanel, dates: np.ndarray, k: int, mode: str = "xs_demean"):
    """(D, N) clipped vol-scaled labels and validity mask for horizon index k.

    ``xs_demean`` removes the date-wide component (the equal-weighted average vol-scaled move
    of all assets).  Because the evaluation is a cross-sectional rank correlation, that
    component is pure noise for the ranking task; removing it is standard practice in
    cross-sectional return prediction.  ``raw`` keeps it (ablation).
    """
    if mode not in LABEL_MODES:
        raise ValueError(f"label mode must be one of {LABEL_MODES}")
    lab = panel.labels[dates, :, k]
    valid = panel.fs.active[dates] & np.isfinite(lab)
    z = np.clip(np.where(valid, lab, 0.0) / panel.scale[dates, :, k], -Z_CLIP, Z_CLIP)
    z = xs_demean(z, valid) if mode == "xs_demean" else np.where(valid, z, np.nan)
    return z, valid


def gather_rows(panel: ModelingPanel, dates: np.ndarray, k: int, mode: str = "xs_demean"):
    """Rows (t, i) with an active asset and a finite horizon-k label; returns (t, i, z)."""
    z, valid = vol_scaled_labels(panel, dates, k, mode)
    ti, ii = np.nonzero(valid)
    return dates[ti], ii, z[ti, ii]


def finalize_z(zhat: np.ndarray, active: np.ndarray, mode: str) -> np.ndarray:
    """Map raw model output (D, N) to the forecast used downstream (demeaned if trained so)."""
    if mode == "xs_demean":
        return zhat - _xs_mean(zhat, active)
    return zhat


def _xs_mean(z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nan_to_num(np.nanmean(np.where(valid, z, np.nan), axis=1, keepdims=True))


class PanelModel(ABC):
    """A forecaster refitted at each walk-forward refit date.

    ``level == "asset"`` models return (len(dates), N, H) forecasts of asset log returns that
    are composed into target forecasts; ``level == "target"`` models return (len(dates), J).
    """

    name: str = "model"
    level: str = "asset"

    @abstractmethod
    def fit(self, panel: ModelingPanel, ctx: FitContext) -> None: ...

    @abstractmethod
    def predict(self, panel: ModelingPanel, dates: np.ndarray) -> np.ndarray: ...
