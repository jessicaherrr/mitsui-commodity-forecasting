from .generic import build_targets, load_long_format, to_long_format
from .mitsui import load_mitsui, verify_official_labels
from .schema import MarketData
from .synthetic import make_synthetic_market

__all__ = [
    "MarketData",
    "build_targets",
    "load_long_format",
    "load_mitsui",
    "make_synthetic_market",
    "to_long_format",
    "verify_official_labels",
]


def load_dataset(cfg: dict) -> MarketData:
    """Instantiate the dataset described by the ``data`` section of a config."""
    kind = cfg.get("adapter", "mitsui")
    if kind == "mitsui":
        return load_mitsui(cfg.get("path", "data/raw/mitsui"))
    if kind == "generic":
        return load_long_format(
            cfg["path"],
            horizons=cfg.get("horizons", [1, 5]),
            gap=cfg.get("gap", 1),
            singles=cfg.get("singles", True),
            pairs=[tuple(p) for p in cfg.get("pairs", [])],
            name=cfg.get("name", "generic"),
            provenance=cfg.get("provenance", ""),
        )
    if kind == "synthetic":
        return make_synthetic_market(
            n_dates=cfg.get("n_dates", 600),
            n_assets=cfg.get("n_assets", 8),
            horizons=tuple(cfg.get("horizons", [1, 2])),
            seed=cfg.get("seed", 7),
        )
    raise ValueError(f"unknown data adapter {kind!r}")
