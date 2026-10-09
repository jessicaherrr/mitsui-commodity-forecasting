"""Shared fixtures.

torch is imported first on purpose: with torch loaded, LightGBM models automatically run
single-threaded, which avoids the macOS OpenMP-runtime conflict between the two libraries
inside a single pytest process (the experiment runner isolates them in separate processes).
"""

from pathlib import Path

import numpy as np
import pytest

try:  # noqa: SIM105
    import torch  # noqa: F401
except ImportError:  # pragma: no cover
    pass

from commodity_research.data import make_synthetic_market
from commodity_research.features import build_features, build_panel

ROOT = Path(__file__).resolve().parents[1]
MITSUI = ROOT / "data" / "raw" / "mitsui"
HAS_MITSUI = (MITSUI / "train.csv").exists() and (MITSUI / "train_labels.csv").exists()


@pytest.fixture(scope="session")
def synth():
    md = make_synthetic_market(n_dates=520, n_assets=8, horizons=(1, 2), seed=7)
    fs = build_features(md)
    return md, fs, build_panel(md, fs)


@pytest.fixture(scope="session")
def mitsui():
    if not HAS_MITSUI:
        pytest.skip("MITSUI csv files not available in data/raw/mitsui")
    from commodity_research.data import load_mitsui

    md = load_mitsui(MITSUI)
    fs = build_features(md)
    return md, fs, build_panel(md, fs)


def perturb_future(md, t0: int, seed: int = 0):
    """Copy of ``md`` whose prices strictly after row t0 are randomly distorted."""
    import copy

    md2 = copy.deepcopy(md)
    rng = np.random.default_rng(seed)
    noise = np.exp(rng.normal(0, 0.05, md2.close.iloc[t0 + 1:].shape))
    md2.close.iloc[t0 + 1:] = md2.close.iloc[t0 + 1:].to_numpy() * noise
    if md2.official_labels is not None:
        md2.official_labels.iloc[t0 + 1:] = rng.normal(0, 1, md2.official_labels.iloc[t0 + 1:].shape)
    return md2
