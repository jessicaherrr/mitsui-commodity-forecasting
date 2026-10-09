"""Level A benchmarks that require no fitting (pre-specified, so they cannot be overfit).

All forecasts are in log-return units for horizon h and therefore produce valid
cross-sectional rankings for both single-asset and pair targets.

historical_mean   rhat = h * mean(daily log return over the trailing 252 rows)
momentum_21       rhat = h * (p_t - p_{t-21}) / 21         (time-series momentum)
reversal_5        rhat = -h * (p_t - p_{t-5}) / 5          (short-term reversal)
"""

from __future__ import annotations

import numpy as np

from ..features.panel import ModelingPanel
from .base import FitContext, PanelModel


class _NoFit(PanelModel):
    level = "asset"

    def fit(self, panel: ModelingPanel, ctx: FitContext) -> None:  # nothing to learn
        return None

    def _per_day(self, panel: ModelingPanel) -> np.ndarray:  # (T, N) daily-rate forecast
        raise NotImplementedError

    def predict(self, panel: ModelingPanel, dates: np.ndarray) -> np.ndarray:
        if getattr(self, "_cache_key", None) != id(panel):
            self._cache = np.nan_to_num(self._per_day(panel))
            self._cache_key = id(panel)
        h = np.asarray(panel.horizons, dtype=float)
        return self._cache[dates][:, :, None] * h[None, None, :]


class HistoricalMean(_NoFit):
    name = "hist_mean"

    def __init__(self, window: int = 252, min_periods: int = 126):
        self.window, self.min_periods = window, min_periods

    def _per_day(self, panel):
        r = panel.fs.returns[panel.fs.assets]
        return r.rolling(self.window, min_periods=self.min_periods).mean().to_numpy()


class Momentum(_NoFit):
    name = "momentum_21"

    def __init__(self, lookback: int = 21):
        self.lookback = lookback

    def _per_day(self, panel):
        lp = panel.fs.logp_ff[panel.fs.assets]
        return ((lp - lp.shift(self.lookback)) / self.lookback).to_numpy()


class Reversal(_NoFit):
    name = "reversal_5"

    def __init__(self, lookback: int = 5):
        self.lookback = lookback

    def _per_day(self, panel):
        lp = panel.fs.logp_ff[panel.fs.assets]
        return (-(lp - lp.shift(self.lookback)) / self.lookback).to_numpy()
