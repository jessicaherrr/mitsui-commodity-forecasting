"""EXPLORATORY (post-hoc) analyses used in docs/RESEARCH_REPORT.md section 7.

Nothing here feeds back into model selection; results are labelled exploratory in the report.

1. Structure of the competition cross-section: daily IC of *oracle* forecasts that know the
   realised returns of only (a) the commodity legs or (b) the non-commodity legs of each target.
   Shows how much of the 424-target ranking is determined by the 6 commodity instruments.
2. Target-level historical mean: an expanding-window mean of each target's own vol-scaled,
   horizon-demeaned label (observable labels only) - the "intercept" component of the
   per-target direct ridge - evaluated on development and holdout dates.

Usage: python scripts/exploratory_analysis.py [--profile fast]
"""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd

from commodity_research.config import load_config, results_dir
from commodity_research.evaluation.metrics import daily_rank_ic, summarize_ic
from commodity_research.experiments.runner import prepare
from commodity_research.features import build_pair_features, target_level_labels


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="fast")
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    cfg = load_config(a.profile)
    md, fs, panel = prepare(cfg)
    man = json.loads((results_dir(cfg) / "manifest.json").read_text())
    sp = man["split"]
    periods = {"dev": np.arange(sp["dev_start"], sp["holdout_start"]), "holdout": np.arange(sp["holdout_start"], sp["end"])}
    labels = panel.eval_labels.to_numpy()
    out_dir = results_dir(cfg) / "exploratory"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    # 1) oracle leg decomposition --------------------------------------------------------
    commod = np.array([md.assets.loc[a, "group"] == "COMMOD" for a in fs.assets])
    lab = panel.labels  # (T, N, H) realised asset returns
    only_commod = np.where(commod[None, :, None], lab, 0.0)
    only_other = np.where(~commod[None, :, None], lab, 0.0)
    for name, p in [("oracle_commodity_legs_only", only_commod), ("oracle_noncommodity_legs_only", only_other)]:
        tp = np.nan_to_num(panel.to_targets(p))
        for per, d in periods.items():
            ic, _ = daily_rank_ic(tp[d], labels[d], min_valid=2)
            s = summarize_ic(ic, reps=200)
            rows.append({"analysis": name, "period": per, "mean_ic": s["mean_ic"], "ic_ir": s["ic_ir"], "t_hac": s["t_hac"]})

    # 2) target-level historical mean (expanding, availability-respecting) ---------------
    y = target_level_labels(panel)
    _, _, tscale = build_pair_features(panel)
    z = np.where(np.isfinite(y) & (tscale > 0), y / np.where(tscale > 0, tscale, 1), np.nan)
    for k in np.unique(panel.h_idx):
        cols = panel.h_idx == k
        z[:, cols] = z[:, cols] - np.nanmean(z[:, cols], axis=1, keepdims=True)
    T, J = z.shape
    first = sp["first_train"]
    pred = np.full((T, J), np.nan)
    csum = np.nancumsum(np.where(np.isfinite(z), z, 0), axis=0)
    ccnt = np.cumsum(np.isfinite(z), axis=0)
    for j in range(J):
        h = panel.horizons[panel.h_idx[j]]
        for t in range(sp["dev_start"], sp["end"]):
            last = t - panel.gap - h  # observable at t
            n = ccnt[last, j] - ccnt[first - 1, j]
            if n > 20:
                pred[t, j] = (csum[last, j] - csum[first - 1, j]) / n * tscale[t, j]
    for per, d in periods.items():
        ic, _ = daily_rank_ic(np.nan_to_num(pred[d]), labels[d], min_valid=2)
        s = summarize_ic(ic, reps=200)
        rows.append({"analysis": "target_level_expanding_mean", "period": per, "mean_ic": s["mean_ic"], "ic_ir": s["ic_ir"], "t_hac": s["t_hac"]})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "exploratory.csv", index=False)
    print(df.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
