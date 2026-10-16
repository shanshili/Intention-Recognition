"""Resolve shared model settings and enforce the M7-L0 ablation protocol."""

from copy import deepcopy

from .models import canonical_model_name
from .models.deig_variants import VARIANT_NAMES, resolve_model_spec


STRICT_ABLATION_NAMES = ("m7", *VARIANT_NAMES, "m6")


def resolve_model_configs(configured):
    """Resolve local `inherits` links; persisted run configs contain no links."""
    canonical = {}
    for requested, config in configured.items():
        name = canonical_model_name(requested)
        if name in canonical:
            raise ValueError(f"Duplicate aliases for {name}; aliases must share one run")
        if not isinstance(config, dict):
            raise ValueError(f"Model configuration for {name} must be an object")
        canonical[name] = dict(config)
    resolved, visiting = {}, set()

    def resolve(name):
        if name in resolved:
            return resolved[name]
        if name not in canonical:
            raise ValueError(f"Unknown inherited model configuration {name!r}")
        if name in visiting:
            raise ValueError(f"Cyclic model configuration inheritance at {name}")
        visiting.add(name)
        config = dict(canonical[name])
        parent = config.pop("inherits", None)
        if parent is not None:
            if not isinstance(parent, str):
                raise ValueError("Model inherits must name a configured model")
            config = {**deepcopy(resolve(canonical_model_name(parent))), **config}
        visiting.remove(name)
        resolved[name] = config
        return config

    return {name: deepcopy(resolve(name)) for name in canonical}


def validate_strict_ablation(configs, selected, training, model_overrides=None):
    """Fail before data preparation if structure and loss changes are mixed."""
    if "m7" not in configs:
        raise ValueError("Strict ablations require an m7 mother model configuration")
    if any(name not in STRICT_ABLATION_NAMES for name in selected):
        raise ValueError("Strict ablations must use M7-derived variants, not incremental baselines")
    overrides = dict(model_overrides or {})
    if "encoder" in overrides:
        raise ValueError("Strict ablations fix encoder substitutions; --encoder is not applicable")
    mother = {**configs["m7"], **overrides}
    if mother.get("input_features") != "full":
        raise ValueError("Strict ablations retain full history; node-input ablation slices inside GAT")
    for name in selected:
        if {**configs[name], **overrides} != mother:
            raise ValueError(f"Strict ablation {name} must share all M7 hyperparameters")
    loss = training.get("loss", {})
    if any(float(loss.get(key, 0)) != 0 for key in ("sparse", "direction", "temporal")):
        raise ValueError("Strict structure ablations use M7-L0: all graph loss weights must be zero")
    return {"kind": "strict_m7_l0", "mother_model": "m7",
            "model_roles": {name: ("mother" if name == "m7" else
                                    "without_temporal_attention" if name == "m6" else
                                    resolve_model_spec(name)["ablation"]) for name in selected},
            "shared_model_config": deepcopy(mother),
            "training_policy": "independent_training_same_hyperparameters_and_seed_list",
            "initialization_policy": "same_seed_not_identical_shared_parameter_initialization",
            "checkpoint_policy": "train_from_scratch_not_parent_checkpoint_finetuning"}
