"""Model registry; importing baselines never requires PyTorch."""

from .persistence import Persistence
from .ridge_ar import RidgeAR

MODEL_NAMES = ("persistence", "ridge_ar", "local", "fixed_graph", "static_gate", "dynamic_gate")


def make_model(name, data, config=None, seed=0):
    config = dict(config or {})
    if name == "persistence":
        return Persistence(data, config, seed)
    if name == "ridge_ar":
        return RidgeAR(data, config, seed)
    if name in MODEL_NAMES[2:]:
        try:
            from .gated_forecaster import GatedForecaster
        except ImportError as exc:
            if exc.name == "torch":
                raise RuntimeError("Neural models require PyTorch; persistence and ridge_ar run without it.") from exc
            raise
        return GatedForecaster(name, data, config, seed)
    raise ValueError(f"Unknown model {name!r}; choose one of {MODEL_NAMES}.")
