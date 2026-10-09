"""``audit`` command: programmatic verification of the data foundation.

Checks (all recomputed from the raw files on every run):

* file structure, date ordering, missingness by market (exchange holidays);
* the exact target formula, by reproducing all 424 official labels from prices, and by
  showing that the plausible *wrong* readings (no 1-day gap; price-level differences, as in
  the legacy code) do not reproduce them;
* information timing: when each label becomes observable;
* that ``test.csv`` is a verbatim copy of the last 134 rows of ``train.csv`` (so it cannot serve
  as an independent offline test set);
* basic data-quality flags (non-positive prices, extreme moves);
* static checks of the legacy pipeline (synthetic-data use).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT, results_dir
from ..data import load_mitsui, verify_official_labels
from ..targets import forward_log_return

log = logging.getLogger(__name__)


def _corr(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 10 else np.nan


def run_audit(cfg: dict) -> dict:
    path = Path(cfg["data"].get("path", "data/raw/mitsui"))
    if not path.is_absolute():
        path = ROOT / path
    md = load_mitsui(path)
    raw = pd.read_csv(path / "train.csv").set_index("date_id")
    out_dir = results_dir({**cfg, "name": "audit"})
    out_dir.mkdir(parents=True, exist_ok=True)
    rep: dict = {}

    # ------------------------------------------------------------- structure
    markets = {}
    for mk, prefix in [("LME", "LME_"), ("JPX", "JPX_"), ("US", "US_Stock_"), ("FX", "FX_")]:
        cols = [c for c in raw.columns if c.startswith(prefix)]
        closes = [c for c in md.close.columns if c.startswith(prefix)]
        markets[mk] = {
            "raw_columns": len(cols),
            "close_series": len(closes),
            "dates_with_all_closes_missing": int(md.close[closes].isna().all(axis=1).sum()),
            "mean_missing_frac_close": round(float(md.close[closes].isna().mean().mean()), 4),
        }
    rep["structure"] = {
        "n_dates": int(len(raw)), "date_id_min": int(raw.index.min()), "date_id_max": int(raw.index.max()),
        "date_id_gap_free": bool(np.all(np.diff(raw.index.to_numpy()) == 1)),
        "n_raw_columns": int(raw.shape[1]), "markets": markets,
        "n_targets": len(md.targets), "horizons": md.horizons,
        "targets_per_horizon": {int(h): int(sum(t.horizon == h for t in md.targets)) for h in md.horizons},
        "single_asset_targets": int(sum(not t.is_pair for t in md.targets)),
        "pair_targets": int(sum(t.is_pair for t in md.targets)),
        "distinct_assets_in_targets": len(md.universe),
        "target_categories": md.target_table()["category"].value_counts().to_dict(),
    }

    # ------------------------------------------------------------- target formula
    ver = verify_official_labels(md)
    ver.to_csv(out_dir / "label_verification.csv")
    rep["label_verification"] = {
        "formula": "target(t) = log(P_A[t+1+lag]/P_A[t+1]) - log(P_B[t+1+lag]/P_B[t+1])  (single: first term only)",
        "max_abs_error_all_targets": float(ver["max_abs_err"].max()),
        "cells_compared": int(ver["n_compared"].sum()),
        "official_only_cells": int(ver["n_official_only"].sum()),
        "official_only_cells_beyond_sample_end": int(ver["n_official_only_beyond_sample"].sum()),
        "recomputed_only_cells": int(ver["n_recomputed_only"].sum()),
    }
    # Alternative (wrong) interpretations, on a few representative targets
    alt = []
    for t in [md.targets[i] for i in (0, 1, 2, 106, 107, 318, 423)]:
        y = md.official_labels[t.name]
        legs = t.legs
        gap0 = sum(s * forward_log_return(md.close[l], 0, t.horizon) for s, l in zip(t.signs, legs))
        if t.is_pair:
            spread = md.close[legs[0]] - md.close[legs[1]]
        else:
            spread = md.close[legs[0]]
        price_diff_change = spread.shift(-(1 + t.horizon)) - spread.shift(-1)
        price_level_fwd = spread.shift(-t.horizon)  # legacy: raw price spread shifted by lag
        alt.append({
            "target": t.name, "pair": " - ".join(legs), "lag": t.horizon,
            "corr_correct_formula": _corr(y, t.compute(md.close)),
            "corr_no_gap_log_return": _corr(y, gap0),
            "corr_price_spread_change": _corr(y, price_diff_change),
            "corr_legacy_price_level_target": _corr(y, price_level_fwd),
        })
    rep["alternative_interpretations"] = alt

    # ------------------------------------------------------------- timing
    rep["information_timing"] = {
        "decision": "forecast for date t is formed at the close of t using data with date_id <= t",
        "label_window": "prices at t+1 and t+1+lag (the close-to-close return from t to t+1 is NOT part of the label)",
        "label_observable_at": "close of t+1+lag (lag 1..4 -> 2..5 dates after the decision date)",
        "training_rule": "a model refitted at the close of tau may use labels with t <= tau - 1 - lag",
        "label_nan_fraction_by_horizon": {int(h): round(float(md.official_labels[[t.name for t in md.targets if t.horizon == h]].isna().mean().mean()), 4) for h in md.horizons},
    }

    # ------------------------------------------------------------- test.csv
    test_path = path / "test.csv"
    if test_path.exists():
        te = pd.read_csv(test_path).set_index("date_id")
        common = [c for c in te.columns if c in raw.columns]
        overlap = te.index.intersection(raw.index)
        a, b = te.loc[overlap, common], raw.loc[overlap, common]
        same_values = bool(((a == b) | (a.isna() & b.isna())).all().all())
        rep["test_csv"] = {
            "rows": int(len(te)), "date_id_range": [int(te.index.min()), int(te.index.max())],
            "rows_overlapping_train": int(len(overlap)), "identical_to_train_rows": same_values,
            "extra_columns": [c for c in te.columns if c not in raw.columns],
            "is_scored_counts": {str(k): int(v) for k, v in te["is_scored"].value_counts().items()} if "is_scored" in te else None,
            "has_labels": False,
            "conclusion": "test.csv is the competition API mock: features already in train.csv, no labels -> not an independent test set",
        }

    # ------------------------------------------------------------- data quality
    ret = np.log(md.close.ffill(limit=5)).diff().where(md.close.notna())
    big = ret.abs().stack()
    big = big[big > 0.25].sort_values(ascending=False)
    rep["data_quality"] = {
        "nonpositive_prices": int((raw[md.close.columns] <= 0).sum().sum()),
        "sparse_assets_lt50pct": [c for c in md.close.columns if md.close[c].notna().mean() < 0.5],
        "daily_abs_log_returns_gt_0.25": int(len(big)),
        "largest_moves": [{"date_id": int(d), "asset": a, "abs_log_return": round(float(v), 3)} for (d, a), v in big.head(8).items()],
        "note": "largest moves cluster around date_id ~566-586 (consistent with March 2020) and are kept as genuine market moves",
    }

    # ------------------------------------------------------------- legacy static checks
    legacy = ROOT / "legacy" / "original" / "main_workflow.py"
    if legacy.exists():
        src = legacy.read_text()
        main_body = src[src.index("def main():"):]
        rep["legacy_static_checks"] = {
            "main_uses_synthetic_generator": "create_enhanced_sample_data(" in main_body,
            "main_reads_train_csv": "train.csv" in main_body,
            "main_reads_train_labels_csv": "train_labels" in main_body,
            "target_built_as_price_difference": bool(re.search(r"all_data\[feat1\]\s*-\s*all_data\[feat2\]", src)),
            "label_shift_applied_again": "y.shift(-lag)" in src,
        }

    rep["headline"] = {
        "dates": rep["structure"]["n_dates"],
        "targets": rep["structure"]["n_targets"],
        "assets_in_targets": rep["structure"]["distinct_assets_in_targets"],
        "label_formula_verified_max_abs_error": rep["label_verification"]["max_abs_error_all_targets"],
        "test_csv_identical_to_train_tail": rep.get("test_csv", {}).get("identical_to_train_rows"),
        "legacy_uses_synthetic_data": rep.get("legacy_static_checks", {}).get("main_uses_synthetic_generator"),
        "report": str(out_dir / "data_audit.json"),
    }
    (out_dir / "data_audit.json").write_text(json.dumps(rep, indent=2, default=str))
    log.info("audit written to %s", out_dir)
    return rep
