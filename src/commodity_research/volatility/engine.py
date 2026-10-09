"""Run the volatility models over many series in parallel and cache the results.

Two families of series are forecast:

* **assets**: daily log returns of every asset in the modelling universe (106 for MITSUI);
* **spreads**: daily changes of each target's log price ratio l_t = p^A_t - p^B_t (or p^A_t for
  single-asset targets); the variance of a target label is the variance of the sum of these
  daily spread returns over the label window.

Stored aggregates per (series, row, model), all in decimal variance units:

    k1      one-step-ahead variance  E_t[r_{t+1}^2]
    cum5    variance of the next 5 trading days
    cum21   variance of the next 21 trading days
    lw{h}   variance of the MITSUI label window, k = 2..h+1 (gap 1), h = 1..4
    phi     GARCH persistence a + b + g/2 in force (NaN for non-GARCH models)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .models import VolEngineConfig, forecast_series, to_rows, window_sum

log = logging.getLogger(__name__)


def aggregates(gap: int, horizons: list[int]) -> dict[str, tuple[int, int]]:
    out = {"k1": (1, 1), "cum5": (1, 5), "cum21": (1, 21)}
    for h in horizons:
        out[f"lw{h}"] = (gap + 1, gap + h)
    return out


def _run_one(name: str, r: np.ndarray, cfg: VolEngineConfig, n_rows: int, aggs: dict, ffill: int):
    sf = forecast_series(r, cfg)
    agg = {}
    for m, fk in sf.f.items():
        chans = [to_rows(sf, n_rows, window_sum(fk, a, b), ffill) for a, b in aggs.values()]
        chans.append(to_rows(sf, n_rows, sf.phi[m], ffill))
        agg[m] = np.stack(chans, axis=-1)
    resid = {m: to_rows(sf, n_rows, v, 0) for m, v in sf.resid.items()}
    return name, agg, sf.params, resid


def run_engine(series: dict[str, np.ndarray], cfg: VolEngineConfig, gap: int, horizons: list[int],
               n_jobs: int = -1, ffill: int = 5) -> dict:
    from joblib import Parallel, delayed

    aggs = aggregates(gap, horizons)
    names = list(series)
    n_rows = len(next(iter(series.values())))
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs, batch_size=4)(
        delayed(_run_one)(n, series[n], cfg, n_rows, aggs, ffill) for n in names)
    try:  # release joblib's reusable workers now; otherwise a spawned model process waits for
        # their 300 s idle timeout before it can exit
        from joblib.externals.loky import get_reusable_executor

        get_reusable_executor().shutdown(wait=True)
    except Exception:  # pragma: no cover
        pass
    M, A = len(cfg.models), len(aggs) + 1
    F = np.full((len(names), n_rows, M, A), np.nan, dtype=np.float32)
    Z = np.full((len(names), n_rows, M), np.nan, dtype=np.float32)
    params = {}
    for s, (name, agg, par, resid) in enumerate(res):
        for k, m in enumerate(cfg.models):
            F[s, :, k, :] = agg[m]
            Z[s, :, k] = resid[m]
        params[name] = par
    log.info("volatility engine: %d series x %d models in %.1fs", len(names), M, time.time() - t0)
    return {"names": names, "models": list(cfg.models), "aggs": list(aggs) + ["phi"], "F": F, "Z": Z, "params": params}


def series_from_panel(panel) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """(asset daily returns, target spread daily returns) as row-aligned arrays."""
    fs = panel.fs
    ret = fs.returns[fs.assets]
    assets = {a: ret[a].to_numpy(dtype=float) for a in fs.assets}
    lp = fs.logp_ff[fs.assets].to_numpy()
    obs = panel.md.close[fs.assets].notna().to_numpy()
    spreads = {}
    for j, name in enumerate(panel.targets.index):
        a, b = panel.leg_a[j], panel.leg_b[j]
        l = lp[:, a] - (lp[:, b] if b >= 0 else 0.0)
        both = obs[:, a] & (obs[:, b] if b >= 0 else True)
        d = np.diff(l, prepend=np.nan)
        spreads[name] = np.where(both, d, np.nan)
    return assets, spreads


def cache_key(cfg: VolEngineConfig, fingerprint: dict, kind: str) -> str:
    blob = json.dumps({"cfg": asdict(cfg), "data": fingerprint, "kind": kind, "v": ENGINE_VERSION}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


ENGINE_VERSION = 3  # bump when cached outputs change (3: exact multi-step recursion for phi >= 1)


def load_or_run(panel, cfg: VolEngineConfig, cache_dir: Path, fingerprint: dict, kind: str, n_jobs: int = -1) -> dict:
    """Cached engine output for ``kind`` in {"assets", "spreads"}."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(cfg, fingerprint, kind)
    path = cache_dir / f"vol_{kind}_{key}.npz"
    meta_path = cache_dir / f"vol_{kind}_{key}_params.json"
    if path.exists() and meta_path.exists():
        z = np.load(path, allow_pickle=False)
        params = json.loads(meta_path.read_text())
        return {"names": list(z["names"]), "models": list(z["models"]), "aggs": list(z["aggs"]),
                "F": z["F"], "Z": z["Z"], "params": params}
    assets, spreads = series_from_panel(panel)
    series = assets if kind == "assets" else spreads
    out = run_engine(series, cfg, panel.gap, panel.horizons, n_jobs=n_jobs)
    np.savez_compressed(path, names=np.array(out["names"]), models=np.array(out["models"]),
                        aggs=np.array(out["aggs"]), F=out["F"], Z=out["Z"])
    meta_path.write_text(json.dumps(out["params"]))
    return out


def realised(r: np.ndarray, aggs: dict[str, tuple[int, int]]) -> np.ndarray:
    """Realised proxy for each aggregate on the trading-day grid mapped to rows: sum of squared
    future returns over the same window (NaN if any return in the window is missing)."""
    pos = np.flatnonzero(np.isfinite(r))
    x2 = r[pos] ** 2
    n = len(x2)
    c = np.concatenate([[0.0], np.cumsum(x2)])
    out = np.full((len(r), len(aggs)), np.nan)
    for k, (a, b) in enumerate(aggs.values()):
        i = np.arange(n)
        lo, hi = i + a, i + b  # trading-day positions of the window
        ok = hi < n
        vals = np.full(n, np.nan)
        vals[ok] = c[hi[ok] + 1] - c[lo[ok]]
        out[pos, k] = vals
    return out


def frame(engine: dict, model: str, agg: str) -> pd.DataFrame:
    """(rows x series) DataFrame of one model/aggregate."""
    m, a = engine["models"].index(model), engine["aggs"].index(agg)
    return pd.DataFrame(engine["F"][:, :, m, a].T, columns=engine["names"])
