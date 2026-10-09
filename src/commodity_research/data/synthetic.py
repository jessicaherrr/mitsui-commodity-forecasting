"""Small deterministic SYNTHETIC dataset for unit and smoke tests ONLY.

Nothing produced from this module is market evidence; the returned ``MarketData`` carries
``is_synthetic=True`` and every report generated from it is stamped as synthetic.

Generating process (per asset i, date t)::

    r_{i,t} = beta_i * f_t + u_{i,t},     u_{i,t} = -0.05 * u_{i,t-1} + e_{i,t}

with a common factor f_t, asset-specific noise scales and ~3% randomly missing (holiday)
prices.
The weak planted AR structure lets smoke tests confirm the pipeline runs end to end; its
magnitude is arbitrary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..targets import TargetKind, TargetSpec
from .schema import MarketData


def make_synthetic_market(
    n_dates: int = 600,
    n_assets: int = 8,
    horizons: tuple[int, ...] = (1, 2),
    seed: int = 7,
    missing_frac: float = 0.03,
) -> MarketData:
    rng = np.random.default_rng(seed)
    classes = ["metal", "fx", "equity"]
    ids = [f"SYN_{classes[i % 3].upper()}_{i}" for i in range(n_assets)]
    vol = rng.uniform(0.005, 0.02, n_assets)
    beta = rng.uniform(0.3, 1.2, n_assets)
    f = rng.standard_normal(n_dates) * 0.008
    idio = np.zeros((n_dates, n_assets))
    eps = rng.standard_normal((n_dates, n_assets)) * vol
    for t in range(1, n_dates):
        idio[t] = -0.05 * idio[t - 1] + eps[t]
    r = beta * f[:, None] + idio
    logp = np.log(100.0) + np.cumsum(r, axis=0)
    close = pd.DataFrame(np.exp(logp), index=pd.RangeIndex(n_dates, name="date_id"), columns=ids)
    mask = rng.random(close.shape) < missing_frac
    mask[:5] = False
    close = close.mask(mask)

    assets = pd.DataFrame(index=pd.Index(ids, name="asset"))
    assets["market"] = "SYN"
    assets["asset_class"] = [classes[i % 3] for i in range(n_assets)]
    assets["group"] = assets["asset_class"].str.upper()
    assets["fx_base"] = None
    assets["fx_quote"] = None

    targets: list[TargetSpec] = []
    k = 0
    for h in horizons:
        targets.append(TargetSpec(f"target_{k}", TargetKind.LOG_RETURN, (ids[0],), h, 1)); k += 1
        for i in range(1, n_assets):
            targets.append(TargetSpec(f"target_{k}", TargetKind.RETURN_SPREAD, (ids[i], ids[i - 1]), h, 1)); k += 1

    md = MarketData(
        name="synthetic",
        close=close,
        assets=assets,
        targets=targets,
        official_labels=None,
        factor_defs={"MKT": {a: 1.0 for a in ids}, "METAL": {a: 1.0 for a in ids if "METAL" in a}},
        is_synthetic=True,
        provenance="SYNTHETIC deterministic test data - not real market evidence",
    )
    md.validate()
    return md
