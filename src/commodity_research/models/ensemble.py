"""Level D: forecast combination, including regime-aware weighting.

All schemes combine *asset-level* forecasts of the same member set.  Members are first put on
a common scale with a per-(model, horizon) scale ``s_{m,h}`` = std of the model's
development-period forecasts, combined, and mapped back with the mean member scale so that
the cross-horizon magnitude (which matters for the mixed-horizon competition cross-section)
is preserved:

    rhat^ens_{i,t,h} = sbar_h * sum_m w_{m,t} * rhat^m_{i,t,h} / s_{m,h}

Weighting schemes (none of them ever sees a holdout label before it is observable):

equal       w_m = 1/M
static_ic   w_m from development-period mean daily IC (clipped at 0), shrunk 50% to equal
regime      static_ic computed separately for each volatility regime on development dates;
            on date s the weights of the regime observed at s are used
online      w_{m,s} from the trailing ``lookback`` daily ICs whose labels are observable at s
            (decision dates t <= s - gap - h_max), clipped at 0, shrunk 50% to equal.  This is
            what a live system could do; it uses holdout labels only after they are public.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _shrunk_ic_weights(mean_ics: np.ndarray, shrink: float) -> np.ndarray:
    M = len(mean_ics)
    raw = np.clip(np.nan_to_num(mean_ics), 0.0, None)
    eq = np.full(M, 1.0 / M)
    if raw.sum() <= 0:
        return eq
    return shrink * eq + (1 - shrink) * raw / raw.sum()


class ForecastCombiner:
    def __init__(self, members: list[str], scheme: str = "equal", shrink: float = 0.5, lookback: int = 126,
                 gap: int = 1, h_max: int = 4):
        if scheme not in {"equal", "static_ic", "regime", "online"}:
            raise ValueError(f"unknown ensemble scheme {scheme}")
        self.members, self.scheme, self.shrink, self.lookback = members, scheme, shrink, lookback
        self.gap, self.h_max = gap, h_max

    def fit(self, preds: dict[str, np.ndarray], daily_ic: dict[str, np.ndarray], dev_dates: np.ndarray,
            regimes: pd.Series | None = None) -> "ForecastCombiner":
        """Estimate scales and static/regime weights from *development* dates only."""
        H = preds[self.members[0]].shape[2]
        self.scale_ = np.array([[np.nanstd(preds[m][dev_dates, :, k]) for k in range(H)] for m in self.members])
        self.scale_ = np.where(self.scale_ > 0, self.scale_, 1.0)
        dev_ic = np.array([np.nanmean(daily_ic[m][dev_dates]) for m in self.members])
        self.static_w_ = _shrunk_ic_weights(dev_ic, self.shrink)
        self.regime_w_ = {}
        if regimes is not None:
            reg = regimes.to_numpy()
            for r in pd.unique(reg[dev_dates]):
                if r is None or (isinstance(r, float) and np.isnan(r)):
                    continue
                sel = dev_dates[reg[dev_dates] == r]
                if len(sel) < 20:
                    continue
                ics = np.array([np.nanmean(daily_ic[m][sel]) for m in self.members])
                self.regime_w_[r] = _shrunk_ic_weights(ics, self.shrink)
        self.daily_ic_ = np.column_stack([daily_ic[m] for m in self.members])
        self.regimes_ = regimes
        return self

    def weights(self, s: int) -> np.ndarray:
        M = len(self.members)
        if self.scheme == "equal":
            return np.full(M, 1.0 / M)
        if self.scheme == "static_ic":
            return self.static_w_
        if self.scheme == "regime":
            r = self.regimes_.iloc[s] if self.regimes_ is not None else None
            return self.regime_w_.get(r, self.static_w_)
        last = s - self.gap - self.h_max  # last decision date with all-horizon labels observable at s
        lo = max(0, last - self.lookback + 1)
        if last < lo:
            return np.full(M, 1.0 / M)
        window = self.daily_ic_[lo:last + 1]
        if np.isfinite(window).sum() < 20 * M:
            return self.static_w_
        return _shrunk_ic_weights(np.nanmean(window, axis=0), self.shrink)

    def combine(self, preds: dict[str, np.ndarray], dates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (combined (T,N,H) with NaN outside ``dates``, weights (len(dates), M))."""
        T, N, H = preds[self.members[0]].shape
        out = np.full((T, N, H), np.nan)
        sbar = self.scale_.mean(axis=0)
        W = np.zeros((len(dates), len(self.members)))
        for n, s in enumerate(dates):
            w = self.weights(int(s))
            W[n] = w
            acc = np.zeros((N, H))
            for k, m in enumerate(self.members):
                acc += w[k] * np.nan_to_num(preds[m][s]) / self.scale_[k]
            out[s] = acc * sbar
        return out, W


def member_disagreement(preds: dict[str, np.ndarray], members: list[str], scale: np.ndarray, to_targets) -> np.ndarray:
    """Cross-model std of scale-normalised *target* forecasts, (T, J)."""
    stack = []
    for k, m in enumerate(members):
        p = preds[m] / scale[k][None, None, :]
        stack.append(to_targets(p))
    return np.nanstd(np.stack(stack), axis=0)
