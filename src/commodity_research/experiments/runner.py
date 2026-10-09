"""Walk-forward training: development-period tuning, then an untouched final holdout.

Protocol
--------
1. Development walk-forward over decision dates [dev_start, holdout_start): every candidate
   hyper-parameter configuration is refitted every ``refit_every`` dates on an expanding
   window and predicts the following block out of sample.
2. The configuration with the best development competition-style score is selected.  The
   holdout is not touched in this step.
3. The selected configuration is run walk-forward over [holdout_start, end) with the same
   refit rule.  Holdout labels enter a training set only once they are observable
   (``t + gap + h <= tau``), exactly as they would in live use.
4. Ensemble scales/weights are estimated from development-period out-of-sample forecasts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np

from ..config import config_hash, results_dir
from ..data import MarketData, load_dataset
from ..evaluation.metrics import competition_score, daily_rank_ic
from ..features import FeatureSet, ModelingPanel, build_features, build_panel
from ..models import FitContext, ForecastCombiner, make_model
from ..validation import walk_forward_folds

log = logging.getLogger(__name__)


# ------------------------------------------------------------------------ preparation
def prepare(cfg: dict) -> tuple[MarketData, FeatureSet, ModelingPanel]:
    md = load_dataset(cfg["data"])
    fs = build_features(md, cfg.get("features", {}))
    panel = build_panel(md, fs)
    return md, fs, panel


def resolve_split(cfg: dict, panel: ModelingPanel) -> dict:
    s = dict(cfg["split"])
    T = panel.n_dates
    end = s.get("end") or T
    out = {
        "first_train": int(s["first_train"]),
        "dev_start": int(s["dev_start"]),
        "holdout_start": int(s["holdout_start"]),
        "end": int(min(end, T)),
        "refit_every": int(s["refit_every"]),
        "window": s.get("window"),
    }
    if not out["first_train"] < out["dev_start"] < out["holdout_start"] < out["end"]:
        raise ValueError(f"invalid split ordering: {out}")
    return out


def feature_mask(fs: FeatureSet, cfg: dict, key: str) -> np.ndarray:
    groups = cfg["feature_sets"][key]
    mask = fs.group_mask(groups)
    if not mask.any():
        raise ValueError(f"feature set {key} selects no features")
    return mask


def data_fingerprint(cfg: dict) -> dict:
    d = cfg["data"]
    if d.get("adapter") == "generic":
        p = Path(d["path"])
        return {"adapter": "generic", "file": hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None,
                "targets": {k: d.get(k) for k in ("horizons", "gap", "singles", "pairs")}}
    if d.get("adapter") != "mitsui":
        return {"adapter": d.get("adapter"), **{k: v for k, v in d.items() if k != "adapter"}}
    path = Path(d.get("path", "data/raw/mitsui"))
    out = {}
    for f in ["train.csv", "train_labels.csv", "target_pairs.csv"]:
        p = path / f
        if p.exists():
            out[f] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


# ------------------------------------------------------------------------ walk-forward
def walk_forward(spec: dict, params: dict, panel: ModelingPanel, folds, split: dict, mask: np.ndarray,
                 seed: int, n_jobs: int, cfg: dict | None = None) -> tuple[np.ndarray, str, list[dict]]:
    """Run one model configuration over ``folds``; return (predictions, level, fold logs)."""
    T, N, H, J = panel.n_dates, panel.n_assets, len(panel.horizons), len(panel.targets)
    model = make_model(spec["type"], spec["name"], params, n_jobs=n_jobs, cfg=cfg)
    level = model.level
    pred = np.full((T, N, H) if level == "asset" else (T, J), np.nan, dtype=np.float64)
    logs = []
    for fold in folds:
        t0 = time.time()
        model = make_model(spec["type"], spec["name"], params, n_jobs=n_jobs, cfg=cfg)
        ctx = FitContext(tau=fold.refit, first_train=split["first_train"], feature_mask=mask, seed=seed,
                         window=split["window"])
        model.fit(panel, ctx)
        pred[fold.test] = model.predict(panel, fold.test)
        entry = {"refit": int(fold.refit), "n_test": int(len(fold.test)), "seconds": round(time.time() - t0, 2)}
        if hasattr(model, "history_"):
            entry["epochs"] = [len(h) for h in model.history_]
        logs.append(entry)
    return pred, level, logs


def target_predictions(pred: np.ndarray, level: str, panel: ModelingPanel) -> np.ndarray:
    tp = panel.to_targets(pred) if level == "asset" else pred
    return tp


def score_on(dates: np.ndarray, tp: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    p = np.nan_to_num(tp[dates])
    ic, _ = daily_rank_ic(p, labels[dates], min_valid=2)
    return competition_score(p, labels[dates]), float(np.nanmean(ic))


# ------------------------------------------------------------------------------- main
def run_training(cfg: dict, only: list[str] | None = None) -> dict:
    t_start = time.time()
    out_dir = results_dir(cfg)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)
    budget = float(cfg.get("runtime", {}).get("max_minutes", 60)) * 60

    md, fs, panel = prepare(cfg)
    split = resolve_split(cfg, panel)
    dev_folds = walk_forward_folds(split["dev_start"], split["holdout_start"], split["refit_every"])
    hold_folds = walk_forward_folds(split["holdout_start"], split["end"], split["refit_every"])
    log.info("dataset=%s synthetic=%s dates=%d assets=%d targets=%d | dev folds=%d holdout folds=%d",
             md.name, md.is_synthetic, panel.n_dates, panel.n_assets, len(panel.targets), len(dev_folds), len(hold_folds))

    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if (only is not None and manifest_path.exists()) else {}
    manifest.update({
        "profile": cfg["name"], "config_hash": config_hash(cfg), "seed": seed, "dataset": md.name,
        "is_synthetic": md.is_synthetic, "provenance": md.provenance, "data_fingerprint": data_fingerprint(cfg),
        "split": split, "n_dates": panel.n_dates, "n_assets": panel.n_assets, "n_targets": len(panel.targets),
        "horizons": panel.horizons, "gap": panel.gap, "feature_names": fs.names, "feature_groups": fs.groups,
    })
    manifest.setdefault("models", {})

    for spec in cfg["models"]:
        name = spec["name"]
        if only is not None and name not in only:
            continue
        if spec.get("enabled", True) is False:
            continue
        elapsed = time.time() - t_start
        if spec.get("optional", False) and elapsed > budget:
            log.warning("skipping optional model %s: time budget exhausted (%.0fs)", name, elapsed)
            manifest["models"][name] = {"status": "skipped_time_budget"}
            continue
        if cfg.get("runtime", {}).get("isolate_models", True):
            manifest["models"][name] = run_isolated(cfg, spec)
        else:
            manifest["models"][name] = train_model(cfg, spec, (md, fs, panel))
        manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default))

    run_ensembles(cfg, panel, split, manifest)
    manifest["total_seconds"] = round(time.time() - t_start, 1)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default))
    log.info("training finished in %.1f min -> %s", manifest["total_seconds"] / 60, out_dir)
    return manifest


def train_model(cfg: dict, spec: dict, prepared=None) -> dict:
    """Tune on the development walk-forward, then run the untouched holdout; save forecasts."""
    t0 = time.time()
    md, fs, panel = prepared or prepare(cfg)
    split = resolve_split(cfg, panel)
    labels = panel.eval_labels.to_numpy()
    seed = int(cfg.get("seed", 42))
    n_jobs = int(cfg.get("runtime", {}).get("n_jobs", -1))
    dev_dates = np.arange(split["dev_start"], split["holdout_start"])
    dev_folds = walk_forward_folds(split["dev_start"], split["holdout_start"], split["refit_every"])
    hold_folds = walk_forward_folds(split["holdout_start"], split["end"], split["refit_every"])
    name = spec["name"]
    mask = feature_mask(fs, cfg, spec.get("features", "all"))
    grid = spec.get("grid") or [spec.get("params", {})]
    grid_results, best = [], None
    for params in grid:
        dev_pred, level, dev_logs = walk_forward(spec, params, panel, dev_folds, split, mask, seed, n_jobs, cfg)
        sc, mic = score_on(dev_dates, target_predictions(dev_pred, level, panel), labels)
        grid_results.append({"params": params, "dev_score": sc, "dev_mean_ic": mic, "fold_log": dev_logs})
        log.info("%-22s dev  params=%s score=%.4f meanIC=%.4f", name, params, sc, mic)
        if best is None or (np.isfinite(sc) and sc > best[0]):
            best = (sc, params, dev_pred)
    _, best_params, dev_pred = best
    hold_pred, level, hold_logs = walk_forward(spec, best_params, panel, hold_folds, split, mask, seed, n_jobs, cfg)
    pred = np.where(np.isnan(hold_pred), dev_pred, hold_pred)
    pred_dir = results_dir(cfg) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_dir / f"{name}.npz", pred=pred.astype(np.float32), level=level)
    log.info("%-22s done in %.1fs (selected %s)", name, time.time() - t0, best_params)
    return {
        "status": "ok", "kind": spec["type"], "level": level, "features": spec.get("features", "all"),
        "selected_params": best_params, "grid": grid_results, "holdout_fold_log": hold_logs,
        "seconds": round(time.time() - t0, 1),
    }


def _isolated_job(cfg: dict, spec: dict) -> dict:
    from ..logging_utils import setup_logging

    setup_logging(results_dir(cfg) / "run.log")
    return train_model(cfg, spec)


def run_isolated(cfg: dict, spec: dict) -> dict:
    """Run ``train_model`` in a fresh spawned process.

    Each model gets a clean interpreter, so torch (LSTM) and LightGBM never share a process
    (their OpenMP runtimes conflict on macOS) and memory is released between models.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
        try:
            return ex.submit(_isolated_job, cfg, spec).result()
        except Exception as e:  # keep the remaining experiments running; record the failure
            log.exception("model %s failed", spec["name"])
            return {"status": "failed", "error": f"{type(e).__name__}: {e}"}


def load_predictions(cfg: dict, name: str) -> tuple[np.ndarray, str]:
    z = np.load(results_dir(cfg) / "predictions" / f"{name}.npz", allow_pickle=False)
    return z["pred"].astype(np.float64), str(z["level"])


def ensemble_groups(cfg: dict) -> list[dict]:
    """Ensemble groups from config (``ensembles`` list; legacy single ``ensemble`` dict accepted)."""
    if cfg.get("ensembles"):
        return cfg["ensembles"]
    if cfg.get("ensemble"):
        return [{"prefix": "ens", **cfg["ensemble"]}]
    return []


def select_members(group: dict, manifest: dict) -> list[str]:
    """Explicit member list, or ``dev_positive``: candidates with a positive development score."""
    ok = lambda m: manifest["models"].get(m, {}).get("status") == "ok" and manifest["models"][m].get("level") == "asset"
    if group["members"] == "dev_positive":
        out = []
        for m in group["candidates"]:
            if not ok(m):
                continue
            info = manifest["models"][m]
            best = next(g for g in info["grid"] if g["params"] == info["selected_params"])
            if np.isfinite(best["dev_score"]) and best["dev_score"] > 0:
                out.append(m)
        return out
    return [m for m in group["members"] if ok(m)]


def run_ensembles(cfg: dict, panel: ModelingPanel, split: dict, manifest: dict) -> None:
    labels = panel.eval_labels.to_numpy()
    dev_dates = np.arange(split["dev_start"], split["holdout_start"])
    oos_dates = np.arange(split["dev_start"], split["end"])
    pred_dir = results_dir(cfg) / "predictions"
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for group in ensemble_groups(cfg):
        members = select_members(group, manifest)
        if len(members) < 2:
            log.warning("ensemble group %s skipped: fewer than two members (%s)", group["prefix"], members)
            continue
        preds, daily_ic = {}, {}
        for m in members:
            if m not in cache:
                p, _ = load_predictions(cfg, m)
                tp = np.nan_to_num(panel.to_targets(p))
                ic = np.full(panel.n_dates, np.nan)
                ic[oos_dates], _ = daily_rank_ic(tp[oos_dates], labels[oos_dates], min_valid=2)
                cache[m] = (p, ic)
            preds[m], daily_ic[m] = cache[m]
        regimes = panel.fs.regime[group.get("regime_column", "vol_regime")]
        for scheme in group.get("schemes", ["equal"]):
            comb = ForecastCombiner(members, scheme, shrink=group.get("shrink", 0.5), lookback=group.get("lookback", 126),
                                    gap=panel.gap, h_max=max(panel.horizons))
            comb.fit(preds, daily_ic, dev_dates, regimes)
            out, W = comb.combine(preds, oos_dates)
            name = f"{group['prefix']}_{scheme}"
            np.savez_compressed(pred_dir / f"{name}.npz", pred=out.astype(np.float32), level="asset")
            np.savez_compressed(pred_dir / f"{name}_weights.npz", weights=W.astype(np.float32), dates=oos_dates, members=np.array(members))
            manifest["models"][name] = {
                "status": "ok", "kind": "ensemble", "level": "asset", "scheme": scheme, "members": members,
                "member_rule": group["members"] if isinstance(group["members"], str) else "fixed",
                "static_weights": comb.static_w_.round(4).tolist(),
                "regime_weights": {str(k): v.round(4).tolist() for k, v in comb.regime_w_.items()},
                "member_scales": comb.scale_.tolist(),
                "note": "weights/scales estimated on development dates only; development-period ensemble metrics are in-sample for the weights",
            }
            log.info("ensemble %-18s members=%s static w=%s", name, members, np.round(comb.static_w_, 3))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
