"""Explicit, dataset-independent target definitions.

Notation (positions are row indices of the chronologically sorted date index):

    p_t^A  = log P_t^A                      log price of asset A at the close of date t
    g      = gap                            dates between the decision date and return start
    h      = horizon                        return length in dates

Target kinds
------------
LOG_RETURN            y_t = p^A_{t+g+h} - p^A_{t+g}
RETURN_SPREAD         y_t = (p^A_{t+g+h} - p^A_{t+g}) - (p^B_{t+g+h} - p^B_{t+g})
                          = l_{t+g+h} - l_{t+g},   with l_t = p^A_t - p^B_t (log price ratio)
PRICE_SPREAD_CHANGE   y_t = S_{t+g+h} - S_{t+g},   with S_t = P^A_t - P^B_t   (NOT scale free)
DIRECTION             y_t = sign(LOG_RETURN or RETURN_SPREAD)

The four objects are different and must not be confused:

* a *price spread* S_t = P^A - P^B depends on units/currencies and is not a return;
* a *log price ratio* l_t = log(P^A / P^B) is a level (a feature, never a label);
* a *return spread* r^A - r^B is the change of the log price ratio over the horizon;
* an *individual return* r^A is a single-asset label.

The MITSUI competition uses g = 1 (verified to 1e-6 against train_labels.csv):
target(t) = log(P_{t+1+lag} / P_{t+1}) for single assets, and the difference of two such
log returns for "A - B" pairs.

Information timing
------------------
A label for decision date t is only known after the close of date t + g + h, so it is
*available* at position ``t + g + h``.  Any model trained at the end of date tau may only use
labels with ``t + g + h <= tau``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


class TargetKind(str, Enum):
    LOG_RETURN = "log_return"
    RETURN_SPREAD = "return_spread"
    PRICE_SPREAD_CHANGE = "price_spread_change"
    DIRECTION = "direction"


@dataclass(frozen=True)
class TargetSpec:
    """A forward-looking label defined on close prices."""

    name: str
    kind: TargetKind
    legs: tuple[str, ...]
    horizon: int
    gap: int = 1
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"{self.name}: horizon must be >= 1")
        if self.gap < 0:
            raise ValueError(f"{self.name}: gap must be >= 0")
        n_legs = len(self.legs)
        if self.kind in (TargetKind.RETURN_SPREAD, TargetKind.PRICE_SPREAD_CHANGE) and n_legs != 2:
            raise ValueError(f"{self.name}: {self.kind.value} needs exactly two legs")
        if self.kind == TargetKind.LOG_RETURN and n_legs != 1:
            raise ValueError(f"{self.name}: log_return needs exactly one leg")
        if self.kind == TargetKind.DIRECTION and n_legs not in (1, 2):
            raise ValueError(f"{self.name}: direction needs one or two legs")

    @property
    def is_pair(self) -> bool:
        return len(self.legs) == 2

    @property
    def signs(self) -> tuple[float, ...]:
        """Linear weights on leg log-returns (used to compose pair forecasts)."""
        return (1.0, -1.0) if self.is_pair else (1.0,)

    # ------------------------------------------------------------------ timing
    def information_interval(self, t: int) -> tuple[int, int]:
        """Positions of the first and last prices the label at ``t`` depends on."""
        return t + self.gap, t + self.gap + self.horizon

    def available_position(self, t: int | np.ndarray) -> int | np.ndarray:
        """First position at whose close the label for decision date ``t`` is observable."""
        return t + self.gap + self.horizon

    # ------------------------------------------------------------- computation
    def compute(self, close: pd.DataFrame) -> pd.Series:
        """Compute the label series from *raw* (un-filled) close prices.

        Missing prices propagate to NaN labels, matching the competition convention that a
        label is undefined when either leg did not trade on a required date.
        """
        missing = [leg for leg in self.legs if leg not in close.columns]
        if missing:
            raise KeyError(f"{self.name}: legs not found in price panel: {missing}")
        start, end = self.gap, self.gap + self.horizon
        if self.kind == TargetKind.PRICE_SPREAD_CHANGE:
            spread = close[self.legs[0]] - close[self.legs[1]]
            y = spread.shift(-end) - spread.shift(-start)
        else:
            y = sum(
                sign * forward_log_return(close[leg], start, end)
                for sign, leg in zip(self.signs, self.legs)
            )
            if self.kind == TargetKind.DIRECTION:
                y = np.sign(y)
        return pd.Series(np.asarray(y, dtype=float), index=close.index, name=self.name)


def forward_log_return(price: pd.Series, start: int, end: int) -> pd.Series:
    """log(P_{t+end} / P_{t+start}) computed on raw prices (NaN-propagating)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.log(price.astype(float))
    return logp.shift(-end) - logp.shift(-start)


def compute_labels(close: pd.DataFrame, targets: list[TargetSpec]) -> pd.DataFrame:
    return pd.concat([t.compute(close) for t in targets], axis=1)


def parse_mitsui_pair(target: str, pair: str, lag: int) -> TargetSpec:
    """Parse a row of MITSUI ``target_pairs.csv`` into a :class:`TargetSpec`.

    ``"A - B"`` denotes the difference of log returns (verified against official labels),
    not a difference of price levels.
    """
    legs = tuple(part.strip() for part in pair.split(" - "))
    kind = TargetKind.RETURN_SPREAD if len(legs) == 2 else TargetKind.LOG_RETURN
    return TargetSpec(name=target, kind=kind, legs=legs, horizon=int(lag), gap=1)
