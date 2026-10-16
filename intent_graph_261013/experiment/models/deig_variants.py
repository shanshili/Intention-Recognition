"""Torch-free, named DEIG model specifications.

Strict ablations start from the same M7 structure and change only the named
component. Components are not user-overridable flags: this table is the
experiment definition, while model configuration supplies numeric dimensions.
"""


VARIANT_NAMES = (
    "m5_attention",
    "m7_no_gat",
    "m7_no_tcn",
    "m7_no_extra_node_features",
    "m7_no_gat_edge",
    "m7_no_score_edge",
    "m7_no_intent",
    "m7_fixed_intent",
    "m7_static_intent",
)


_M7_SPEC = {
    "version": 7,
    "input_features": 7,
    "spatial_input_features": 7,
    "has_pre_gat": True,
    "encoder_kind": "tcn",
    "temporal_pooling": "attention",
    "static_in_gat": True,
    "static_in_score": True,
    "dynamic_in_score": True,
    "score_kind": "dynamic",
    "parent_model": "m7",
    "ablation": None,
}


_VARIANT_CHANGES = {
    "m5_attention": {"dynamic_in_score": False, "ablation": "without_dynamic_edge_features"},
    "m7_no_gat": {"has_pre_gat": False, "static_in_gat": False, "ablation": "without_gat"},
    "m7_no_tcn": {"encoder_kind": "pointwise", "ablation": "without_temporal_convolution"},
    "m7_no_extra_node_features": {
        "spatial_input_features": 1, "ablation": "without_extra_spatial_node_features"},
    "m7_no_gat_edge": {"static_in_gat": False, "ablation": "without_static_edge_in_gat"},
    "m7_no_score_edge": {"static_in_score": False, "ablation": "without_static_edge_in_score"},
    "m7_no_intent": {"score_kind": "none", "ablation": "without_intent_message"},
    "m7_fixed_intent": {"score_kind": "fixed", "ablation": "fixed_intent_scores"},
    "m7_static_intent": {"score_kind": "static", "ablation": "static_intent_scores"},
}


def resolve_model_spec(name):
    """Return a fresh specification for a canonical construction/ablation name.

When Edge MLP is replaced or removed, e/d have no scoring path. The static
edge prior in GAT remains active in all three intent-message controls.
Aliases are intentionally owned by the public model registry.
"""
    name = str(name).lower()
    if name in {f"m{i}" for i in range(1, 8)}:
        version = int(name[1:])
        features = 1 if version < 3 else 7
        return {
            "version": version,
            "input_features": features,
            "spatial_input_features": features,
            "has_pre_gat": True,
            "encoder_kind": "mlp" if version == 1 else "tcn",
            "temporal_pooling": ("history_mlp" if version == 1 else
                                 "attention" if version == 7 else "last"),
            "static_in_gat": version >= 5,
            "static_in_score": version >= 4,
            "dynamic_in_score": version >= 6,
            "score_kind": "dynamic",
            "parent_model": "m7" if version == 7 else None,
            "ablation": None,
        }
    if name not in _VARIANT_CHANGES:
        raise ValueError(f"Unknown DEIG construction/ablation model {name!r}.")
    spec = dict(_M7_SPEC)
    spec.update(_VARIANT_CHANGES[name])
    if spec["score_kind"] != "dynamic":
        spec["static_in_score"] = False
        spec["dynamic_in_score"] = False
    return spec
