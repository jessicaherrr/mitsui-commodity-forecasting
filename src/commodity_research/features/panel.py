"""Modelling panel: asset-level features + labels + information-availability bookkeeping.

Design choice: *asset-factorised* forecasting
--------------------------------------------
All 424 MITSUI targets are linear combinations of 106 asset log returns at 4 horizons:

    y_{j,t} = r^{A_j}_{t,h_j} - r^{B_j}_{t,h_j}     (or r^{A_j}_{t,h_j} for single-asset targets)

so models forecast r^{i}_{t,h} for every asset i and horizon h, and target forecasts are
composed linearly: yhat_{j,t} = rhat^{A_j}_{t,h_j} - rhat^{B_j}_{t,h_j}.  This keeps forecasts
internally consistent (e.g. yhat(A-B) + yhat(B-C) = yhat(A-C)), shares statistical strength
across the 424 targets, and turns 424 tiny problems into 4 pooled panels.  Direct pair-level
models are implemented as comparators (see ``models.direct``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.schema import MarketData
from ..targets import forward_log_return
from .engineering import FeatureSet, ewma_vol


@dataclass
class ModelingPanel:
    md: MarketData
    fs: FeatureSet
    horizons: list[int]
    gap: int
    labels: np.ndarray  # (T, N, H) asset forward log returns from RAW prices (NaN-propagating)
    scale: np.ndarray  # (T, N, H) sigma_t * sqrt(h): causal label/prediction scale
    targets: pd.DataFrame  # target table (index = target name)
    leg_a: np.ndarray  # (J,) asset index of leg A
    leg_b: np.ndarray  # (J,) asset index of leg B, -1 for single-asset targets
    h_idx: np.ndarray  # (J,) horizon index of each target
    eval_labels: pd.DataFrame  # (T x J) labels used for evaluation

    @property
    def n_dates(self) -> int:
        return len(self.fs.dates)

    @property
    def n_assets(self) -> int:
        return len(self.fs.assets)

    def last_label_date(self, tau: int, h: int) -> int:
        """Latest decision date whose horizon-h label is observable at the close of ``tau``."""
        return tau - self.gap - h

    def to_targets(self, asset_pred: np.ndarray) -> np.ndarray:
        """Compose (T, N, H) asset forecasts into (T, J) target forecasts."""
        a = asset_pred[:, self.leg_a, self.h_idx]
        b = np.where(self.leg_b[None, :] >= 0, asset_pred[:, np.maximum(self.leg_b, 0), self.h_idx], 0.0)
        return a - b


def build_panel(md: MarketData, fs: FeatureSet) -> ModelingPanel:
    horizons, gap = md.horizons, md.gap
    assets = fs.assets
    close = md.close[assets]
    labels = np.stack(
        [np.column_stack([forward_log_return(close[a], gap, gap + h).to_numpy() for a in assets]) for h in horizons],
        axis=-1,
    )
    scale = fs.vol[:, :, None] * np.sqrt(np.asarray(horizons, dtype=float))[None, None, :]

    tt = md.target_table()
    pos = {a: i for i, a in enumerate(assets)}
    hpos = {h: k for k, h in enumerate(horizons)}
    leg_a = np.array([pos[a] for a in tt["leg_a"]])
    leg_b = np.array([pos[b] if isinstance(b, str) else -1 for b in tt["leg_b"]])
    h_idx = np.array([hpos[h] for h in tt["horizon"]])
    return ModelingPanel(
        md=md, fs=fs, horizons=horizons, gap=gap, labels=labels, scale=scale, targets=tt,
        leg_a=leg_a, leg_b=leg_b, h_idx=h_idx, eval_labels=md.labels(),
    )


def build_pair_features(panel: ModelingPanel, lam: float = 0.94) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Target-level features for *direct* pair models.

    Returns (X (T, J, F), names, target_scale (T, J)).  Features are the leg-A minus leg-B
    difference of the asset's own-group features plus relative-value features of the log
    price ratio l_t = p^A_t - p^B_t:

        spread_z_63    (l_t - mean_63(l)) / std_63(l)           mean-reversion indicator
        spread_mom_21  (l_t - l_{t-21}) / (sigma_l * sqrt(21))  relative momentum

    Single-asset targets use the leg-A features and zero spread features.
    """
    fs = panel.fs
    own = fs.group_mask(["own"])
    own_names = [n for n, m in zip(fs.names, own) if m]
    Xa = fs.values[:, panel.leg_a][:, :, own]
    Xb = np.where((panel.leg_b >= 0)[None, :, None], fs.values[:, np.maximum(panel.leg_b, 0)][:, :, own], 0.0)
    diff = Xa - Xb

    T, J = diff.shape[:2]
    spread_z = np.zeros((T, J), dtype=np.float32)
    spread_mom = np.zeros((T, J), dtype=np.float32)
    tscale = np.zeros((T, J))
    assets = fs.assets
    lp = fs.logp_ff[assets].to_numpy()
    for j in range(J):
        a = panel.leg_a[j]
        b = panel.leg_b[j]
        l = pd.Series(lp[:, a] - (lp[:, b] if b >= 0 else 0.0))
        dl = l.diff()
        sig = ewma_vol(dl, lam).to_numpy()
        tscale[:, j] = sig * np.sqrt(panel.horizons[panel.h_idx[j]])
        if b >= 0:
            mu, sd = l.rolling(63, min_periods=40).mean(), l.rolling(63, min_periods=40).std()
            spread_z[:, j] = ((l - mu) / (sd + 1e-12)).clip(-6, 6).to_numpy()
            spread_mom[:, j] = ((l - l.shift(21)) / (sig * np.sqrt(21))).to_numpy()
    X = np.concatenate([diff, spread_z[:, :, None], spread_mom[:, :, None]], axis=-1).astype(np.float32)
    names = [f"diff_{n}" for n in own_names] + ["spread_z_63", "spread_mom_21"]
    return X, names, tscale


def target_level_labels(panel: ModelingPanel) -> np.ndarray:
    """(T, J) target labels recomputed from raw prices (used to *train* direct models)."""
    lab = panel.labels
    a = lab[:, panel.leg_a, panel.h_idx]
    b = np.where(panel.leg_b[None, :] >= 0, lab[:, np.maximum(panel.leg_b, 0), panel.h_idx], 0.0)
    return a - b
