"""Model-specific input views over one prepared data/evaluation protocol.

Feature selection does not rebuild statistics, graphs, targets, or split windows.
The canonical protocol ID remains shared while the effective input channels are
recorded explicitly in each view's resolved protocol and metadata.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .data import PreparedData


FULL_CHANNELS = (
    "level", "delta", "periodic_residual", "time_sin", "time_cos",
    "weekday_sin", "weekday_cos",
)


def select_model_inputs(data: PreparedData, feature_set: str = "level") -> PreparedData:
    """Return level-only or ordered seven-channel inputs sharing all labels.

``full`` requires preparation of delta, periodic residual, and calendar inputs;
it reorders channels by semantic names rather than assuming their source order.
Missing channels are a configuration error, not silently synthesized features.
"""
    if feature_set not in {"level", "full"}:
        raise ValueError("feature_set must be level or full")
    source_channels = list(data.protocol["feature_channels"])
    if len(source_channels) != data.features.shape[-1] or len(set(source_channels)) != len(source_channels):
        raise ValueError("feature_channels must uniquely identify every prepared channel")
    selected = ["level"] if feature_set == "level" else list(FULL_CHANNELS)
    missing = [name for name in selected if name not in source_channels]
    if missing:
        raise ValueError(f"Cannot select {feature_set} inputs; missing channels: {missing}")
    indices = [source_channels.index(name) for name in selected]
    # Level is a view in the usual layout. Full inputs require a copy only when
    # semantic channel ordering differs from the prepared order.
    if indices == list(range(len(source_channels))):
        features = data.features
    elif indices == [0]:
        features = data.features[..., :1]
    else:
        features = np.take(data.features, indices, axis=-1)
    protocol = dict(data.protocol)
    protocol.update({
        "features": ["level"] if feature_set == "level" else ["level", "delta", "periodic_residual", "time"],
        "feature_channels": selected,
        "model_input_feature_set": feature_set,
        "source_feature_channels": source_channels,
    })
    metadata = dict(data.metadata)
    metadata.update({
        "feature_channels": selected,
        "model_input_feature_set": feature_set,
        "source_feature_channels": source_channels,
    })
    return replace(data, features=features, protocol=protocol, metadata=metadata)
