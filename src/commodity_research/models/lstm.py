"""Level C: a deliberately small, multi-horizon LSTM sequence model.

Redesign of the legacy ``AdvancedLSTMRegressor`` (see docs/LEGACY_AUDIT.md):

* input  x: (batch, window, n_features) -- the asset's own last ``window`` rows of
  train-scaled features, ending at the decision date t (no future rows);
* output   : (batch, n_horizons) -- vol-scaled forecasts for every horizon at once
  (multi-output; missing / not-yet-observable labels are masked out of the loss);
* unidirectional LSTM -> last hidden state -> dropout -> linear head.  The legacy
  bidirectional+attention+skip design produced an output of the wrong shape, could not be
  constructed (Xavier init on 1-D LayerNorm weights) and is unnecessary at this data size;
* AdamW, gradient clipping, early stopping on a purged internal validation block, best
  weights restored with a deep copy, deterministic seeds, CPU by default.
"""

from __future__ import annotations

import copy
import logging

import numpy as np
import torch
from torch import nn

from ..features.panel import ModelingPanel
from ..validation import internal_validation
from .base import Z_CLIP, FitContext, PanelModel, finalize_z, fit_scaler, xs_demean

log = logging.getLogger(__name__)


class LSTMNet(nn.Module):
    def __init__(self, n_features: int, n_outputs: int, hidden: int = 32, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected (batch, window, features), got {tuple(x.shape)}")
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])  # (batch, n_outputs)


def make_sequences(Xs: np.ndarray, t: np.ndarray, i: np.ndarray, window: int) -> np.ndarray:
    """Gather (len(t), window, F) sequences ending at rows t for assets i (rows <= t only)."""
    offsets = np.arange(-window + 1, 1)
    rows = np.clip(t[:, None] + offsets[None, :], 0, None)
    return Xs[rows, i[:, None]]


class LSTMModel(PanelModel):
    level = "asset"

    def __init__(self, window: int = 20, hidden: int = 32, num_layers: int = 1, dropout: float = 0.2,
                 lr: float = 1e-3, weight_decay: float = 1e-4, batch_size: int = 512, max_epochs: int = 15,
                 patience: int = 3, max_samples: int | None = 60000, val_len: int = 126, n_seeds: int = 1,
                 device: str = "cpu", num_threads: int | None = None, label_mode: str = "xs_demean",
                 name: str = "lstm"):
        self.window, self.hidden, self.num_layers, self.dropout = window, hidden, num_layers, dropout
        self.lr, self.weight_decay, self.batch_size = lr, weight_decay, batch_size
        self.max_epochs, self.patience, self.max_samples = max_epochs, patience, max_samples
        self.val_len, self.n_seeds, self.device, self.name = val_len, n_seeds, device, name
        self.num_threads, self.label_mode = num_threads, label_mode

    # ------------------------------------------------------------------ data
    def _samples(self, panel: ModelingPanel, per_h_dates: list[np.ndarray]):
        """Rows (t, i) with at least one usable label; per-horizon availability masks."""
        H = len(panel.horizons)
        all_dates = np.unique(np.concatenate(per_h_dates)) if per_h_dates else np.arange(0)
        allowed = np.zeros((panel.n_dates, H), dtype=bool)
        for k, d in enumerate(per_h_dates):
            allowed[d, k] = True
        lab = panel.labels[all_dates]  # (D, N, H)
        usable = np.isfinite(lab) & allowed[all_dates][:, None, :] & panel.fs.active[all_dates][:, :, None]
        zfull = np.clip(np.where(usable, lab, 0.0) / panel.scale[all_dates], -Z_CLIP, Z_CLIP)
        for k in range(H):
            zfull[:, :, k] = (xs_demean(zfull[:, :, k], usable[:, :, k]) if self.label_mode == "xs_demean"
                              else np.where(usable[:, :, k], zfull[:, :, k], np.nan))
        ti, ii = np.nonzero(usable.any(axis=2))
        t = all_dates[ti]
        m = usable[ti, ii]
        z = np.where(m, np.nan_to_num(zfull[ti, ii]), 0.0)
        return t, ii, z.astype(np.float32), m.astype(np.float32)

    def _loss(self, net, Xs, t, i, z, m, idx=None):
        if idx is not None:
            t, i, z, m = t[idx], i[idx], z[idx], m[idx]
        x = torch.from_numpy(make_sequences(Xs, t, i, self.window)).to(self.device)
        pred = net(x)
        zt, mt = torch.from_numpy(z).to(self.device), torch.from_numpy(m).to(self.device)
        return ((pred - zt) ** 2 * mt).sum() / mt.sum().clamp(min=1.0)

    # ------------------------------------------------------------------- fit
    def fit(self, panel: ModelingPanel, ctx: FitContext) -> None:
        if self.num_threads:
            torch.set_num_threads(self.num_threads)
        self.mask_ = ctx.feature_mask
        self.scaler_ = fit_scaler(panel, ctx)
        T, N, _ = panel.fs.values.shape
        self.Xs_ = self.scaler_.transform(panel.fs.values[:, :, self.mask_].reshape(T * N, -1)).reshape(T, N, -1)
        h_max = max(panel.horizons)
        fit_d, val_d = [], []
        for h in panel.horizons:
            f, v = internal_validation(ctx.tau, h, panel.gap, ctx.first_train, self.val_len, h_max, ctx.window)
            fit_d.append(f)
            val_d.append(v)
        tr = self._samples(panel, fit_d)
        va = self._samples(panel, val_d)
        rng = np.random.default_rng(ctx.seed)
        if self.max_samples and len(tr[0]) > self.max_samples:
            keep = np.sort(rng.choice(len(tr[0]), self.max_samples, replace=False))
            tr = tuple(a[keep] for a in tr)
        self.nets_ = []
        self.history_ = []
        for s in range(self.n_seeds):
            net, hist = self._train_one(tr, va, len(panel.horizons), ctx.seed + 1000 * s)
            self.nets_.append(net)
            self.history_.append(hist)

    def _train_one(self, tr, va, n_out, seed):
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        net = LSTMNet(self.Xs_.shape[-1], n_out, self.hidden, self.num_layers, self.dropout).to(self.device)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best, best_state, bad, hist = np.inf, copy.deepcopy(net.state_dict()), 0, []
        n = len(tr[0])
        for epoch in range(self.max_epochs):
            net.train()
            perm = torch.randperm(n, generator=gen).numpy()
            tot = 0.0
            for b in range(0, n, self.batch_size):
                idx = perm[b:b + self.batch_size]
                opt.zero_grad()
                loss = self._loss(net, self.Xs_, *tr, idx=idx)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                tot += loss.item() * len(idx)
            val = self._evaluate(net, va)
            hist.append({"epoch": epoch, "train_loss": tot / max(n, 1), "val_loss": val})
            if val < best - 1e-5:
                best, best_state, bad = val, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        return net, hist

    @torch.no_grad()
    def _evaluate(self, net, va) -> float:
        net.eval()
        n = len(va[0])
        if n == 0:
            return np.nan
        num = den = 0.0
        for b in range(0, n, 4096):
            idx = np.arange(b, min(b + 4096, n))
            t, i, z, m = (a[idx] for a in va)
            pred = net(torch.from_numpy(make_sequences(self.Xs_, t, i, self.window)).to(self.device)).cpu().numpy()
            num += float(((pred - z) ** 2 * m).sum())
            den += float(m.sum())
        return num / max(den, 1.0)

    # --------------------------------------------------------------- predict
    @torch.no_grad()
    def predict(self, panel: ModelingPanel, dates: np.ndarray) -> np.ndarray:
        N, H = panel.n_assets, len(panel.horizons)
        t = np.repeat(dates, N)
        i = np.tile(np.arange(N), len(dates))
        out = np.zeros((len(t), H))
        for b in range(0, len(t), 4096):
            sl = slice(b, b + 4096)
            x = torch.from_numpy(make_sequences(self.Xs_, t[sl], i[sl], self.window)).to(self.device)
            out[sl] = np.mean([net(x).cpu().numpy() for net in self.nets_], axis=0)
        z = out.reshape(len(dates), N, H)
        for k in range(H):
            z[:, :, k] = finalize_z(z[:, :, k], panel.fs.active[dates], self.label_mode)
        return z * panel.scale[dates]
