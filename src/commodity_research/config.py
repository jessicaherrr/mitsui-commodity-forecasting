"""Configuration loading: ``configs/base.yaml`` deep-merged with ``configs/<profile>.yaml``."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(profile: str | Path) -> dict:
    path = Path(profile)
    if not path.suffix:
        path = CONFIG_DIR / f"{profile}.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base_path = CONFIG_DIR / "base.yaml"
    if path.resolve() != base_path.resolve() and base_path.exists() and not cfg.get("no_base", False):
        with open(base_path) as f:
            cfg = deep_merge(yaml.safe_load(f) or {}, cfg)
    cfg.setdefault("name", path.stem)
    cfg["_config_path"] = str(path)
    return cfg


def config_hash(cfg: dict) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest()[:12]


def results_dir(cfg: dict) -> Path:
    base = Path(cfg.get("output_root", "results"))
    if not base.is_absolute():
        base = ROOT / base
    return base / cfg["name"]
