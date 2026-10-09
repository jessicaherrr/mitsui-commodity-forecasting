"""Generic adapter for ordinary long-format daily price data.

Input: one CSV (or DataFrame) with columns

    timestamp, asset_id, close            (required; ``price`` is accepted for ``close``)
    open, high, low, volume               (optional, carried but unused by default)
    market, asset_class                   (optional metadata)

Targets are generated from configuration rather than read from a file:

    horizons:  [1, 5]           forward log-return horizons in rows (trading dates)
    gap:       1                decision at close t, return measured from close t+gap
    singles:   true             one LOG_RETURN target per asset and horizon
    pairs:     [[A, B], ...]    RETURN_SPREAD targets r^A - r^B per horizon
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..targets import TargetKind, TargetSpec
from .schema import MarketData


def build_targets(
    assets: list[str],
    horizons: list[int],
    gap: int = 1,
    singles: bool = True,
    pairs: list[tuple[str, str]] | None = None,
) -> list[TargetSpec]:
    targets: list[TargetSpec] = []
    for h in horizons:
        if singles:
            for a in assets:
                targets.append(TargetSpec(f"ret_{a}_h{h}", TargetKind.LOG_RETURN, (a,), h, gap))
        for a, b in pairs or []:
            targets.append(TargetSpec(f"spread_{a}_minus_{b}_h{h}", TargetKind.RETURN_SPREAD, (a, b), h, gap))
    return targets


def load_long_format(
    source: str | Path | pd.DataFrame,
    horizons: list[int] = (1, 5),
    gap: int = 1,
    singles: bool = True,
    pairs: list[tuple[str, str]] | None = None,
    name: str = "generic",
    provenance: str = "",
    is_synthetic: bool = False,
) -> MarketData:
    df = pd.read_csv(source) if not isinstance(source, pd.DataFrame) else source.copy()
    if "close" not in df.columns and "price" in df.columns:
        df = df.rename(columns={"price": "close"})
    required = ["timestamp", "asset_id", "close"]
    absent = [c for c in required if c not in df.columns]
    if absent:
        raise ValueError(f"long-format data is missing required columns {absent}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if df.duplicated(["timestamp", "asset_id"]).any():
        raise ValueError("duplicate (timestamp, asset_id) rows")

    close = df.pivot(index="timestamp", columns="asset_id", values="close").sort_index()
    close = close.where(close > 0).astype(float)
    close.columns.name = None

    meta = df.groupby("asset_id").first()
    assets = pd.DataFrame(index=pd.Index(close.columns, name="asset"))
    assets["market"] = meta["market"].reindex(assets.index) if "market" in meta else "generic"
    assets["asset_class"] = meta["asset_class"].reindex(assets.index) if "asset_class" in meta else "generic"
    assets["market"] = assets["market"].fillna("generic")
    assets["asset_class"] = assets["asset_class"].fillna("generic")
    assets["group"] = assets["asset_class"]
    assets["fx_base"] = None
    assets["fx_quote"] = None

    # One equal-weighted factor per asset class (only when it has >= 2 members), plus a
    # market-wide factor.
    factor_defs = {"MKT": {a: 1.0 for a in close.columns}}
    for cls, members in assets.groupby("asset_class").groups.items():
        if len(members) >= 2:
            factor_defs[f"CLS_{cls}".upper()] = {a: 1.0 for a in members}

    md = MarketData(
        name=name,
        close=close,
        assets=assets,
        targets=build_targets(list(close.columns), list(horizons), gap, singles, pairs),
        official_labels=None,
        factor_defs=factor_defs,
        is_synthetic=is_synthetic,
        provenance=provenance,
    )
    md.validate()
    return md


def to_long_format(close: pd.DataFrame, assets: pd.DataFrame | None = None) -> pd.DataFrame:
    """Inverse helper (used in tests): wide close panel -> long format."""
    long = close.rename_axis("timestamp").reset_index().melt(id_vars="timestamp", var_name="asset_id", value_name="close")
    long = long.dropna(subset=["close"])
    if assets is not None:
        long = long.merge(assets[["market", "asset_class"]], left_on="asset_id", right_index=True, how="left")
    return long.reset_index(drop=True)


__all__ = ["build_targets", "load_long_format", "to_long_format"]
