from .base import FitContext, PanelModel, TrainOnlyScaler
from .baselines import HistoricalMean, Momentum, Reversal
from .direct import DirectRidge
from .ensemble import ForecastCombiner
from .tabular import LightGBMModel, RidgeModel

# NOTE: torch is imported lazily (models.lstm) so that a process running LightGBM never loads
# torch's bundled OpenMP runtime: on macOS the two runtimes segfault/deadlock in one process.


def make_model(kind: str, name: str, params: dict | None = None, n_jobs: int = -1, cfg: dict | None = None) -> PanelModel:
    """Factory used by the experiment runner (``kind`` comes from the config)."""
    params = dict(params or {})
    if kind == "hist_mean":
        m = HistoricalMean(**params)
    elif kind == "momentum":
        m = Momentum(**params)
    elif kind == "reversal":
        m = Reversal(**params)
    elif kind == "ridge":
        m = RidgeModel(**params)
    elif kind == "lgbm":
        m = LightGBMModel(params=params, n_jobs=n_jobs)
    elif kind == "lstm":
        from .lstm import LSTMModel

        m = LSTMModel(**params)
    elif kind == "direct_ridge":
        m = DirectRidge(**params)
    elif kind == "target_pooled":
        from .target_models import TargetPooledModel

        m = TargetPooledModel(**params, n_jobs=n_jobs, cfg=cfg)
    else:
        raise ValueError(f"unknown model kind {kind!r}")
    m.name = name
    return m


__all__ = [
    "DirectRidge", "FitContext", "ForecastCombiner", "HistoricalMean", "LightGBMModel", "Momentum", "PanelModel", "Reversal", "RidgeModel", "TrainOnlyScaler", "make_model",
]
