"""Adapter for the MITSUI & CO. Commodity Prediction Challenge (Kaggle) files.

Files (``data/raw/mitsui``):

* ``train.csv``          date_id + 557 raw market columns (LME, JPX, US equities/ETFs, FX)
* ``train_labels.csv``   date_id + target_0 ... target_423
* ``target_pairs.csv``   target, lag, pair   ("A" or "A - B")
* ``test.csv``           NOT an independent test set: rows 1827-1960 of train.csv repeated
                         (verified identical) plus an ``is_scored`` flag. Never used for
                         evaluation in this project.

Verified label definition (max abs error <= 1e-6 over all 424 targets, see ``audit``)::

    target(t) = log(P^A_{t+1+lag} / P^A_{t+1})  [- log(P^B_{t+1+lag} / P^B_{t+1})]
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..targets import compute_labels, parse_mitsui_pair
from .schema import MarketData

log = logging.getLogger(__name__)

# Hand-curated, approximate classification of the US tickers in the dataset (by primary
# exposure).  Used only for grouping/factor construction, never as a label.
US_GROUPS: dict[str, list[str]] = {
    "us_equity_broad": [
        "ACWI", "VT", "VXUS", "VEA", "VWO", "IEMG", "EEM", "EFA", "EWJ", "EWT", "EWY", "EWZ",
        "FXI", "VGK", "RSP", "SPYV", "VTV", "VYM", "YINN",
    ],
    "us_bond": [
        "AGG", "BND", "BNDX", "BSV", "IEF", "IGSB", "JNK", "LQD", "MBB", "SHY", "SPIB", "SPTL",
        "TIP", "VCIT", "VCSH", "VGIT", "VGLT", "VGSH", "EMB",
    ],
    "us_precious": [
        "GLD", "IAU", "SLV", "GDX", "GDXJ", "NUGT", "AEM", "AG", "FNV", "GOLD", "HL", "KGC",
        "NEM", "PAAS", "WPM",
    ],
    "us_metals_materials": [
        "FCX", "SCCO", "TECK", "RIO", "VALE", "CLF", "X", "NUE", "STLD", "ALB", "LYB", "XLB",
        "URA", "CCJ",
    ],
    "us_energy": [
        "BP", "COP", "CVE", "CVX", "DVN", "EOG", "HAL", "HES", "KMI", "MPC", "OIH", "OKE",
        "OXY", "SHEL", "SLB", "TRGP", "WMB", "XLE", "XOM", "ENB", "BKR",
    ],
    "us_industrial_fin": ["CAT", "DE", "AMP", "BCS", "MS", "RY", "TD"],
}
_US_LOOKUP = {tic: grp for grp, tics in US_GROUPS.items() for tic in tics}

# Coarse leg groups used to categorise targets (pair categories such as "COMMOD-EQ").
COARSE = {"LME": "COMMOD", "JPX": "COMMOD", "US": "EQ", "FX": "FX"}


def _close_columns(raw: pd.DataFrame) -> list[str]:
    cols = []
    for c in raw.columns:
        if c.startswith("LME_") and c.endswith("_Close"):
            cols.append(c)
        elif c.startswith("JPX_") and c.endswith("_Close"):
            cols.append(c)
        elif c.startswith("US_Stock_") and c.endswith("_adj_close"):
            cols.append(c)
        elif c.startswith("FX_"):
            cols.append(c)
    return cols


def _asset_metadata(asset: str) -> dict:
    meta = {"market": None, "asset_class": None, "group": None, "fx_base": None, "fx_quote": None}
    if asset.startswith("LME_"):
        meta.update(market="LME", asset_class="base_metal_futures")
    elif asset.startswith("JPX_"):
        meta.update(market="JPX", asset_class="precious_metal_futures" if ("Gold" in asset or "Platinum" in asset) else "other_futures")
    elif asset.startswith("US_Stock_"):
        tic = asset[len("US_Stock_"):].replace("_adj_close", "")
        meta.update(market="US", asset_class=_US_LOOKUP.get(tic, "us_other"))
    elif asset.startswith("FX_"):
        pair = asset[3:]
        meta.update(market="FX", asset_class="fx", fx_base=pair[:3], fx_quote=pair[3:6])
    meta["group"] = COARSE.get(meta["market"], "other")
    return meta


def _factor_definitions(assets: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Economically motivated cross-asset factors (daily log-return composites)."""
    def present(ids):
        return [a for a in ids if a in assets.index]

    fx = assets[assets["market"] == "FX"]
    usd = {}
    for aid, row in fx.iterrows():
        if row["fx_base"] == "USD":
            usd[aid] = 1.0  # USDJPY up = USD stronger
        elif row["fx_quote"] == "USD":
            usd[aid] = -1.0  # AUDUSD up = USD weaker
    factors = {
        "EQ": {a: 1.0 for a in present(["US_Stock_VT_adj_close"])},
        "USD": usd,
        "METAL": {a: 1.0 for a in assets.index[assets["market"] == "LME"]},
        "GOLD": {a: 1.0 for a in present(["US_Stock_GLD_adj_close", "US_Stock_IAU_adj_close"])},
        "RATES": {a: 1.0 for a in present(["US_Stock_IEF_adj_close", "US_Stock_VGIT_adj_close"])},
        "ENERGY": {a: 1.0 for a in present(["US_Stock_XLE_adj_close"])},
    }
    return {k: v for k, v in factors.items() if v}


def load_mitsui(path: str | Path = "data/raw/mitsui") -> MarketData:
    path = Path(path)
    required = ["train.csv", "train_labels.csv", "target_pairs.csv"]
    missing = [f for f in required if not (path / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"MITSUI files {missing} not found in {path}. Download them from "
            "https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/data"
        )
    raw = pd.read_csv(path / "train.csv")
    labels = pd.read_csv(path / "train_labels.csv")
    pairs = pd.read_csv(path / "target_pairs.csv")

    for df, name, cols in [
        (raw, "train.csv", ["date_id"]),
        (labels, "train_labels.csv", ["date_id"]),
        (pairs, "target_pairs.csv", ["target", "lag", "pair"]),
    ]:
        absent = [c for c in cols if c not in df.columns]
        if absent:
            raise ValueError(f"{name} is missing required columns {absent}")

    raw = raw.sort_values("date_id").set_index("date_id")
    labels = labels.sort_values("date_id").set_index("date_id")
    if not raw.index.equals(labels.index):
        raise ValueError("train.csv and train_labels.csv have different date_id sets")
    if not np.all(np.diff(raw.index.to_numpy()) == 1):
        raise ValueError("date_id is expected to be a gap-free integer sequence")

    close_cols = _close_columns(raw)
    close = raw[close_cols].apply(pd.to_numeric, errors="coerce").astype(float)
    # Non-positive prices are data errors; treat as missing rather than crash.
    close = close.where(close > 0)

    assets = pd.DataFrame([_asset_metadata(a) for a in close_cols], index=pd.Index(close_cols, name="asset"))
    targets = [parse_mitsui_pair(r.target, r.pair, r.lag) for r in pairs.itertuples()]

    md = MarketData(
        name="mitsui",
        close=close,
        assets=assets,
        targets=targets,
        official_labels=labels[[t.name for t in targets]].astype(float),
        factor_defs=_factor_definitions(assets),
        is_synthetic=False,
        provenance=(
            "Kaggle MITSUI&CO. Commodity Prediction Challenge (2025) training files; "
            "date_id is an anonymised trading-date index (calendar dates not provided)."
        ),
    )
    for w in md.validate():
        log.warning(w)
    return md


def verify_official_labels(md: MarketData) -> pd.DataFrame:
    """Recompute every target from prices and compare with the official labels.

    Returns one row per target with max abs error and NaN-pattern diagnostics. Official
    labels near the end of the sample use prices beyond the last row of train.csv, so those
    cells cannot be recomputed and are reported separately (``n_beyond_sample``).
    """
    if md.official_labels is None:
        raise ValueError("dataset has no official labels")
    recomputed = compute_labels(md.close, md.targets)
    rows = []
    n = md.n_dates
    pos = np.arange(n)
    for t in md.targets:
        y, c = md.official_labels[t.name].to_numpy(), recomputed[t.name].to_numpy()
        both = np.isfinite(y) & np.isfinite(c)
        beyond = t.available_position(pos) > n - 1
        rows.append(
            {
                "target": t.name,
                "horizon": t.horizon,
                "max_abs_err": float(np.max(np.abs(y[both] - c[both]))) if both.any() else np.nan,
                "n_compared": int(both.sum()),
                "n_official_only": int((np.isfinite(y) & ~np.isfinite(c)).sum()),
                "n_official_only_beyond_sample": int((np.isfinite(y) & ~np.isfinite(c) & beyond).sum()),
                "n_recomputed_only": int((~np.isfinite(y) & np.isfinite(c)).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("target")
