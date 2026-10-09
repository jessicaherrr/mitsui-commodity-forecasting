"""Dataset-independent container for daily multi-asset market data."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..targets import TargetSpec, compute_labels

ASSET_META_COLUMNS = ["market", "asset_class", "group", "fx_base", "fx_quote"]


@dataclass
class MarketData:
    """Close-price panel plus metadata and target definitions.

    Attributes
    ----------
    close:
        (dates x assets) *raw* close prices. NaN means the instrument did not trade / was not
        observed on that date. The index is chronological; horizons are counted in rows.
    assets:
        Metadata indexed by asset id with columns ``ASSET_META_COLUMNS``.
    targets:
        Forward-looking label definitions.
    official_labels:
        Optional externally supplied labels (dates x target names), e.g. ``train_labels.csv``.
        When present they are used for evaluation; models train on labels recomputed from
        prices so that availability can be controlled per horizon.
    factor_defs:
        name -> {asset_id: weight}; daily factor return = weighted mean of available
        constituent log returns (weights are signs/orientation, renormalised per date).
    """

    name: str
    close: pd.DataFrame
    assets: pd.DataFrame
    targets: list[TargetSpec]
    official_labels: pd.DataFrame | None = None
    factor_defs: dict[str, dict[str, float]] = field(default_factory=dict)
    is_synthetic: bool = False
    provenance: str = ""

    # ----------------------------------------------------------------- helpers
    @property
    def universe(self) -> list[str]:
        """Assets that appear as a leg of at least one target (the modelling universe)."""
        legs: list[str] = []
        for t in self.targets:
            for leg in t.legs:
                if leg not in legs:
                    legs.append(leg)
        return legs

    @property
    def horizons(self) -> list[int]:
        return sorted({t.horizon for t in self.targets})

    @property
    def gap(self) -> int:
        gaps = {t.gap for t in self.targets}
        if len(gaps) != 1:
            raise ValueError(f"mixed gaps are not supported: {gaps}")
        return gaps.pop()

    @property
    def n_dates(self) -> int:
        return len(self.close)

    def target_table(self) -> pd.DataFrame:
        """One row per target with timing and grouping metadata."""
        rows = []
        cls = self.assets["group"].to_dict()
        for t in self.targets:
            leg_groups = [cls.get(leg, "other") for leg in t.legs]
            category = "single:" + leg_groups[0] if not t.is_pair else "-".join(sorted(leg_groups))
            rows.append(
                {
                    "target": t.name,
                    "kind": t.kind.value,
                    "horizon": t.horizon,
                    "gap": t.gap,
                    "leg_a": t.legs[0],
                    "leg_b": t.legs[1] if t.is_pair else None,
                    "is_pair": t.is_pair,
                    "category": category,
                }
            )
        return pd.DataFrame(rows).set_index("target")

    def labels(self) -> pd.DataFrame:
        """Labels used for *evaluation*: official labels if provided, else recomputed."""
        if self.official_labels is not None:
            return self.official_labels[[t.name for t in self.targets]]
        return compute_labels(self.close, self.targets)

    # -------------------------------------------------------------- validation
    def validate(self) -> list[str]:
        """Raise on structural errors; return a list of non-fatal warnings."""
        warnings: list[str] = []
        idx = self.close.index
        if not idx.is_unique:
            raise ValueError("date index contains duplicates")
        if not idx.is_monotonic_increasing:
            raise ValueError("date index is not sorted chronologically")
        values = self.close.to_numpy(dtype=float)
        if np.any(values[np.isfinite(values)] <= 0):
            raise ValueError("close prices must be strictly positive")
        missing_meta = set(self.close.columns) - set(self.assets.index)
        if missing_meta:
            raise ValueError(f"assets without metadata: {sorted(missing_meta)[:5]}")
        for col in ASSET_META_COLUMNS:
            if col not in self.assets.columns:
                raise ValueError(f"asset metadata missing column {col!r}")
        names = [t.name for t in self.targets]
        if len(set(names)) != len(names):
            raise ValueError("duplicate target names")
        for t in self.targets:
            for leg in t.legs:
                if leg not in self.close.columns:
                    raise ValueError(f"target {t.name}: leg {leg} has no price series")
        if self.official_labels is not None:
            if not self.official_labels.index.equals(idx):
                raise ValueError("official labels are not aligned to the price index")
            absent = set(names) - set(self.official_labels.columns)
            if absent:
                raise ValueError(f"official labels missing targets: {sorted(absent)[:5]}")
        for name, legs in self.factor_defs.items():
            unknown = set(legs) - set(self.close.columns)
            if unknown:
                raise ValueError(f"factor {name} references unknown assets {sorted(unknown)}")
        sparse = self.close.columns[self.close.notna().mean() < 0.5]
        if len(sparse):
            warnings.append(f"{len(sparse)} assets observed on <50% of dates: {list(sparse)[:5]}")
        if self.is_synthetic:
            warnings.append("SYNTHETIC DATA: not real market evidence")
        return warnings
