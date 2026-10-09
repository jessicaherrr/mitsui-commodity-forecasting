"""Walk-forward splits that respect *label availability*, not just feature timestamps.

Timing convention (row positions):

* the forecast for decision date ``s`` is formed at the close of ``s`` from features <= s;
* a horizon-h label for decision date ``t`` uses prices up to ``t + gap + h`` and is therefore
  observable only from the close of ``t + gap + h`` onward;
* a model refitted at the close of ``tau`` may train on decision dates
  ``t <= tau - gap - h``  (the purge is horizon-specific: 2..5 dates for MITSUI);
* it then predicts decision dates ``tau, tau+1, ...`` until the next refit.

Because training labels end at price ``tau`` and test labels start at price ``tau + gap``,
training and test label information intervals never overlap; no separate embargo is needed
for an expanding (or rolling) window that only looks backward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    refit: int  # tau: close at which the model is (re)fitted
    test: np.ndarray  # decision dates predicted by this model


def walk_forward_folds(start: int, end: int, refit_every: int) -> list[Fold]:
    """Folds covering decision dates [start, end) with a refit every ``refit_every`` rows."""
    if refit_every < 1:
        raise ValueError("refit_every must be >= 1")
    folds = []
    for tau in range(start, end, refit_every):
        folds.append(Fold(refit=tau, test=np.arange(tau, min(tau + refit_every, end))))
    return folds


def train_dates(tau: int, h: int, gap: int, first: int, window: int | None = None) -> np.ndarray:
    """Decision dates whose horizon-h labels are observable at the close of ``tau``."""
    last = tau - gap - h
    lo = first if window is None else max(first, last - window + 1)
    if last < lo:
        return np.arange(0)
    return np.arange(lo, last + 1)


def internal_validation(tau: int, h: int, gap: int, first: int, val_len: int, h_max: int,
                        window: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Split the information available at ``tau`` into (fit, early-stopping validation) dates.

    The validation block is the most recent ``val_len`` decision dates whose labels (for the
    longest horizon) are observable at ``tau``; the fit block only contains labels that were
    observable *before* the validation block starts, so the early-stopping signal never
    overlaps the fitting labels.
    """
    v_end = tau - gap - h_max
    v_start = v_end - val_len + 1
    val = np.arange(v_start, tau - gap - h + 1)
    fit = train_dates(v_start - 1, h, gap, first, window)
    return fit, val
