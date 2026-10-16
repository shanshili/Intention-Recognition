"""Model registry; importing baselines never requires PyTorch."""

from .persistence import Persistence
from .ridge_ar import RidgeAR
from .deig_variants import VARIANT_NAMES

BASELINE_NAMES = ("persistence", "ridge_ar", "local", "fixed_graph", "static_gate", "dynamic_gate")
CONSTRUCTION_NAMES = tuple(f"m{version}" for version in range(1, 8)) + VARIANT_NAMES
MODEL_NAMES = BASELINE_NAMES + CONSTRUCTION_NAMES
NEURAL_MODEL_NAMES = BASELINE_NAMES[2:] + CONSTRUCTION_NAMES
MODEL_ALIASES = {**{f"b{index + 1}": name for index, name in enumerate(BASELINE_NAMES)},
                 "m0": "dynamic_gate", "m7_l0": "m7", "m7_dynamic_intent": "m7",
                 "m7_no_dynamic_edge": "m5_attention", "m7_no_temporal_attention": "m6"}


def canonical_model_name(name):
    name = str(name).lower()
    return MODEL_ALIASES.get(name, name)


def make_model(name, data, config=None, seed=0):
    name = canonical_model_name(name)
    config = dict(config or {})
    feature_set = config.pop("input_features", None)
    if feature_set not in {None, "level", "full"}:
        raise ValueError("input_features must be level or full")
    if name == "persistence":
        return Persistence(data, config, seed)
    if name == "ridge_ar":
        return RidgeAR(data, config, seed)
    if name in NEURAL_MODEL_NAMES:
        try:
            if name in CONSTRUCTION_NAMES:
                from .deig_tcn import DeigTCNForecaster
                return DeigTCNForecaster(name, data, config, seed)
            from .gated_forecaster import GatedForecaster
        except ImportError as exc:
            if exc.name == "torch":
                raise RuntimeError("Neural models require PyTorch; persistence and ridge_ar run without it.") from exc
            raise
        return GatedForecaster(name, data, config, seed)
    raise ValueError(f"Unknown model {name!r}; choose one of {MODEL_NAMES}.")
