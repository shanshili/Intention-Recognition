"""Masked prediction loss and explicitly enabled graph regularizers."""

import math


def validate_loss_config(config, model):
    config = dict(config or {})
    unknown = set(config) - {"prediction", "huber_delta", "sparse", "direction", "temporal"}
    if unknown:
        raise ValueError(f"Unknown loss parameters: {sorted(unknown)}")
    prediction = config.get("prediction", "huber")
    if prediction not in {"huber", "mae"}:
        raise ValueError("prediction loss must be huber or mae.")
    delta = float(config.get("huber_delta", 1.0))
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("huber_delta must be finite and positive.")
    result = {"prediction": prediction, "huber_delta": delta}
    for name in ("sparse", "direction", "temporal"):
        weight = float(config.get(name, 0.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{name} weight must be finite and nonnegative.")
        result[name] = weight
    if any(result[name] > 0 for name in ("sparse", "direction", "temporal")):
        if model.capabilities["score_kind"] not in {"static", "dynamic"}:
            raise ValueError("Graph regularization requires learned static or dynamic scores.")
    if result["temporal"] > 0 and model.capabilities["score_kind"] != "dynamic":
        raise ValueError("Temporal regularization requires dynamic scores.")
    return result


def compute_loss(model, output, batch, config):
    import torch
    from torch.nn import functional as F

    target = torch.as_tensor(batch["target"], dtype=torch.float32, device=model.device)
    mask = torch.as_tensor(batch["target_mask"], dtype=torch.bool, device=model.device)
    if not mask.any():
        raise ValueError("Training batch has no observed targets.")
    if config["prediction"] == "mae":
        prediction_loss = torch.abs(output["prediction"] - target)[mask].mean()
    else:
        prediction_loss = F.huber_loss(output["prediction"][mask], target[mask],
                                       delta=config["huber_delta"], reduction="mean")
    zero = prediction_loss.new_zeros(())
    sparse, direction, temporal = zero, zero, zero
    scores = output["edge_scores"]
    if config["sparse"] > 0 and scores.numel():
        sparse = scores.mean()
    if config["direction"] > 0 and len(model.module.pair_a):
        direction = (scores[:, model.module.pair_a] * scores[:, model.module.pair_b]).mean()
    if config["temporal"] > 0:
        if "previous_history" not in batch or "previous_mask" not in batch:
            raise ValueError("Temporal loss requires previous_history and previous_mask from training data.")
        previous_mask = torch.as_tensor(batch["previous_mask"], dtype=torch.bool, device=model.device)
        if previous_mask.ndim != 1 or previous_mask.shape[0] != len(scores):
            raise ValueError("previous_mask must have shape [B].")
        if previous_mask.any() and scores.numel():
            previous_history = torch.as_tensor(batch["previous_history"], dtype=torch.float32,
                                               device=model.device)
            previous_scores = model.module.raw_scores(model.module.encode(previous_history))
            temporal = torch.abs(scores[previous_mask] - previous_scores[previous_mask]).mean()
    total = prediction_loss + config["sparse"] * sparse + config["direction"] * direction + config["temporal"] * temporal
    pieces = {"prediction": prediction_loss, "sparse": sparse, "direction": direction,
              "temporal": temporal, "total": total}
    return total, {name: float(value.detach().cpu()) for name, value in pieces.items()}, int(mask.sum())
