"""One masked, complete-target evaluator for every model and intervention."""

from __future__ import annotations

import time

import numpy as np


def validate_evaluation_config(config):
    result = {"batch_size": 128, "top_fractions": [0.05, 0.1, 0.2],
              "random_seeds": list(range(20)), "threshold_quantile": 0.9,
              "absolute_threshold": None, "near_zero": 1e-8,
              "max_plot_windows": 3, "graph_interventions": True,
              "graph_intervention_splits": ["test"]}
    unknown = set(config) - set(result)
    if unknown:
        raise ValueError(f"Unknown evaluation settings: {sorted(unknown)}")
    result.update(config)
    for key in ("batch_size", "max_plot_windows"):
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    fractions = result["top_fractions"]
    if not fractions or any(not np.isfinite(p) or not 0 < p <= 1 for p in fractions):
        raise ValueError("top_fractions must be nonempty and in (0,1]")
    if len(set(fractions)) != len(fractions):
        raise ValueError("top_fractions cannot contain duplicates")
    seeds = result["random_seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("random_seeds must be unique nonnegative integers")
    q = result["threshold_quantile"]
    if q is not None and (not np.isfinite(q) or not 0 <= q <= 1):
        raise ValueError("threshold_quantile must be null or in [0,1]")
    threshold = result["absolute_threshold"]
    if threshold is not None and (not np.isfinite(threshold) or threshold < 0):
        raise ValueError("absolute_threshold must be null or nonnegative")
    if not np.isfinite(result["near_zero"]) or result["near_zero"] < 0:
        raise ValueError("near_zero must be finite and nonnegative")
    if not isinstance(result["graph_interventions"], bool):
        raise ValueError("graph_interventions must be boolean")
    splits = result["graph_intervention_splits"]
    if (not isinstance(splits, list) or len(set(splits)) != len(splits)
            or not set(splits) <= {"train", "val", "test"}):
        raise ValueError("graph_intervention_splits must list unique train/val/test names")
    return result


def _window_mae(prediction, target, mask):
    counts = mask.sum(axis=(1, 2))
    sums = np.where(mask, np.abs(prediction - target), 0).sum(axis=(1, 2), dtype=np.float64)
    return np.divide(sums, counts, out=np.full(len(counts), np.nan), where=counts > 0)


def _metric_arrays(prediction, target, mask):
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Prediction, target and mask must have identical [M,N,H] shapes")
    if not np.isfinite(prediction).all() or not np.isfinite(target[mask]).all():
        raise FloatingPointError("Nonfinite prediction or observed target")
    count = int(mask.sum())
    if count == 0:
        raise ValueError("No observed targets in evaluation")
    error = np.where(mask, prediction.astype(np.float64) - target, 0)
    counts_h = mask.sum(axis=(0, 1))
    absolute_h = np.abs(error).sum(axis=(0, 1), dtype=np.float64)
    square_h = np.square(error).sum(axis=(0, 1), dtype=np.float64)
    mae_h = np.divide(absolute_h, counts_h, out=np.full(len(counts_h), np.nan), where=counts_h > 0)
    rmse_h = np.sqrt(np.divide(square_h, counts_h, out=np.full(len(counts_h), np.nan), where=counts_h > 0))
    return {"mae": float(np.abs(error).sum(dtype=np.float64) / count),
            "rmse": float(np.sqrt(np.square(error).sum(dtype=np.float64) / count)),
            "per_horizon_mae": mae_h.tolist(), "per_horizon_rmse": rmse_h.tolist(),
            "valid_targets": count, "per_horizon_valid_targets": counts_h.tolist(),
            "windows": len(prediction)}


def collect_predictions(model, dataset, batch_size, edge_masks=None):
    chunks = {key: [] for key in ("prediction", "target", "target_mask", "raw_target",
                                  "last", "window_id", "target_times", "edge_scores", "effective_weight")}
    offset = 0
    for batch in dataset.iter_batches(batch_size):
        B = len(batch["window_id"])
        mask = None if edge_masks is None else edge_masks[offset:offset + B]
        result = model.predict(batch, edge_mask=mask)
        prediction = np.asarray(result["prediction"])
        if prediction.shape != batch["target"].shape or not np.isfinite(prediction).all():
            raise ValueError("Model returned invalid prediction shape or nonfinite values")
        chunks["prediction"].append(prediction)
        for key in ("target", "target_mask", "raw_target", "last", "window_id", "target_times"):
            chunks[key].append(batch[key])
        for key in ("edge_scores", "effective_weight"):
            value = result.get(key)
            if value is not None:
                value = np.asarray(value)
                if value.shape != (B, dataset.data.graph["E"]) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid {key} shape or nonfinite values")
                chunks[key].append(value)
        offset += B
    if offset != len(dataset) or offset == 0:
        raise ValueError("Incomplete or empty evaluation dataset")
    combined = {key: np.concatenate(values, axis=0) if values else None for key, values in chunks.items()}
    for key in ("edge_scores", "effective_weight"):
        if combined[key] is not None and len(combined[key]) != offset:
            raise ValueError(f"Model returned {key} for only some batches")
    return combined


def _score_summary(scores, near_zero):
    quantiles = [0, 0.5, 0.9, 0.99, 1]
    return {"minimum": float(scores.min()), "maximum": float(scores.max()),
            "mean": float(scores.mean()), "quantile_levels": quantiles,
            "quantiles": np.quantile(scores, quantiles).tolist(),
            "zero_fraction": float(np.mean(scores == 0)),
            "near_zero_fraction": float(np.mean(scores <= near_zero)),
            "all_zero_window_fraction": float(np.mean(scores.max(axis=1) == 0)),
            "near_zero_cutoff": near_zero,
            "constant_window_fraction": float(np.mean(np.ptp(scores, axis=1) == 0))}


def _selection(scores, k, mode, seed=None):
    selected = np.zeros(scores.shape, dtype=bool)
    if mode in {"top", "low"}:
        order = np.argsort(-scores if mode == "top" else scores, axis=1, kind="stable")
        np.put_along_axis(selected, order[:, :k], True, axis=1)
    elif mode == "random":
        rng = np.random.default_rng(seed)
        for row in range(len(scores)):
            selected[row, rng.choice(scores.shape[1], k, replace=False)] = True
    else:
        raise ValueError(mode)
    return selected


def _jaccard_pairs(selections, adjacent):
    before, after = selections[:-1][adjacent], selections[1:][adjacent]
    intersection = np.logical_and(before, after).sum(1)
    union = np.logical_or(before, after).sum(1)
    return np.divide(intersection, union, out=np.ones(len(union), dtype=float), where=union > 0)


def _intervention(model, dataset, config, edge_masks, mean, std):
    error_sum, count, offset, window_errors = 0.0, 0, 0, []
    for batch in dataset.iter_batches(config["batch_size"]):
        B = len(batch["window_id"])
        prediction = np.asarray(model.predict(batch, edge_mask=edge_masks[offset:offset + B])["prediction"])
        if prediction.shape != batch["target"].shape:
            raise ValueError("Intervention prediction shape differs from target")
        raw = prediction.astype(np.float64) * std[None, :, None] + mean[None, :, None]
        if not np.isfinite(raw).all():
            raise FloatingPointError("Nonfinite intervention prediction")
        mask = batch["target_mask"]
        error_sum += float(np.where(mask, np.abs(raw - batch["raw_target"]), 0).sum(dtype=np.float64))
        count += int(mask.sum())
        window_errors.append(_window_mae(raw, batch["raw_target"], mask))
        offset += B
    return error_sum / count, np.concatenate(window_errors)


def _threshold(model, validation, config, scores=None):
    explicit = config["absolute_threshold"]
    if explicit is not None:
        return {"enabled": True, "value": float(explicit), "rule": "predeclared_absolute",
                "positive_scores_required": True}
    if validation is None or config["threshold_quantile"] is None:
        return {"enabled": False, "value": None, "rule": "disabled"}
    if scores is None:
        scores = collect_predictions(model, validation, config["batch_size"])["edge_scores"]
    scores = np.asarray(scores)
    if (scores.shape != (len(validation), validation.data.graph["E"])
            or not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1)):
        raise ValueError("Invalid validation gate scores")
    positive = scores[scores > 0]
    if not len(positive):
        return {"enabled": False, "value": None, "rule": "validation_positive_quantile",
                "reason": "No positive validation scores"}
    return {"enabled": True, "value": float(np.quantile(positive, config["threshold_quantile"])),
            "quantile": config["threshold_quantile"], "rule": "validation_positive_quantile",
            "positive_scores_required": True, "exploratory_not_calibrated": True,
            "calibration_split": "val"}


def evaluate_model(model, dataset, evaluation_cfg, validation=None, threshold_spec=None):
    config = validate_evaluation_config(evaluation_cfg)
    started = time.perf_counter()
    data = dataset.data
    split = getattr(dataset, "split", "test")
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown evaluation split: {split}")
    interventions_enabled = config["graph_interventions"] and split in config["graph_intervention_splits"]
    prediction_started = time.perf_counter()
    arrays = collect_predictions(model, dataset, config["batch_size"])
    prediction_seconds = time.perf_counter() - prediction_started
    standard = _metric_arrays(arrays["prediction"], arrays["target"], arrays["target_mask"])
    raw_prediction = arrays["prediction"].astype(np.float64) * data.std[None, :, None] + data.mean[None, :, None]
    raw_metrics = _metric_arrays(raw_prediction, arrays["raw_target"], arrays["target_mask"])
    persistence = np.repeat(arrays["last"][..., None], data.protocol["H"], axis=-1)
    persistence_raw = persistence.astype(np.float64) * data.std[None, :, None] + data.mean[None, :, None]
    persistence_metrics = _metric_arrays(persistence_raw, arrays["raw_target"], arrays["target_mask"])
    persistence_standard = _metric_arrays(persistence, arrays["target"], arrays["target_mask"])
    metrics = {**raw_metrics, "mae_standardized": standard["mae"],
               "rmse_standardized": standard["rmse"],
               "per_horizon_mae_standardized": standard["per_horizon_mae"],
               "per_horizon_rmse_standardized": standard["per_horizon_rmse"],
               "persistence_mae": persistence_metrics["mae"],
               "persistence_rmse": persistence_metrics["rmse"],
               "persistence_mae_standardized": persistence_standard["mae"],
               "persistence_rmse_standardized": persistence_standard["rmse"],
               "skill": (1 - raw_metrics["mae"] / persistence_metrics["mae"]
                         if persistence_metrics["mae"] > 0 else None),
               "skill_standardized": (1 - standard["mae"] / persistence_standard["mae"]
                                      if persistence_standard["mae"] > 0 else None),
               "prediction_seconds": prediction_seconds,
               "prediction_windows_per_second": len(dataset) / prediction_seconds,
               "split": split}
    arrays["raw_prediction"] = raw_prediction
    arrays["per_window_mae"] = _window_mae(raw_prediction, arrays["raw_target"], arrays["target_mask"])
    arrays["per_window_persistence_mae"] = _window_mae(persistence_raw, arrays["raw_target"], arrays["target_mask"])
    capabilities = model.capabilities
    graph_metrics = {"available": False, "intervention_scope": capabilities.get("intervention_scope"),
                     "score_kind": capabilities.get("score_kind"), "budgets": [],
                     "threshold": {"enabled": False, "value": None}, "split": split,
                     "interventions_computed": bool(interventions_enabled and capabilities["supports_intervention"]),
                     "intervention_status": ("computed" if interventions_enabled and capabilities["supports_intervention"]
                                             else "not_applicable" if not capabilities["supports_intervention"]
                                             else "disabled_by_split_policy")}
    graph_arrays = {}
    scores = arrays["edge_scores"]
    if scores is not None and scores.size:
        if np.any(scores < 0) or np.any(scores > 1):
            raise ValueError("Gate scores must be in [0,1]")
        graph_metrics["diagnostics"] = {split: _score_summary(scores, config["near_zero"])}
        graph_arrays.update({"window_id": arrays["window_id"],
                             "score_minimum": scores.min(1), "score_maximum": scores.max(1),
                             "score_mean": scores.mean(1), "score_zero_fraction": (scores == 0).mean(1),
                             "score_near_zero_fraction": (scores <= config["near_zero"]).mean(1),
                             "score_quantile_levels": np.asarray([0, 0.5, 0.9, 0.99, 1]),
                             "score_quantiles": np.quantile(scores, [0, 0.5, 0.9, 0.99, 1], axis=1).T})
        stride = data.protocol["train_stride" if split == "train" else "eval_stride"]
        adjacent = np.diff(arrays["window_id"]) == stride
        changes = np.abs(np.diff(scores, axis=0)).mean(1)[adjacent]
        graph_metrics["temporal"] = {"adjacent_pairs": int(adjacent.sum()),
                                     "mean_score_change": float(changes.mean()) if len(changes) else None}
        graph_arrays["adjacent_window_id"] = arrays["window_id"][1:][adjacent]
        graph_arrays["adjacent_score_change"] = changes
        if capabilities["supports_ranking"]:
            graph_metrics["available"] = True
            validation_scores = (scores if validation is dataset else
                                 collect_predictions(model, validation, config["batch_size"])["edge_scores"]
                                 if validation is not None else None)
            graph_metrics["threshold"] = (dict(threshold_spec) if threshold_spec is not None else
                                          _threshold(model, validation, config, scores=validation_scores))
            threshold = graph_metrics["threshold"]
            if threshold["enabled"]:
                kept = (scores > 0) & (scores >= threshold["value"])
                graph_arrays["absolute_edge_count"] = kept.sum(1)
                graph_metrics["threshold"]["mean_candidate_density"] = float(kept.mean())
                graph_metrics["threshold"]["empty_window_fraction"] = float(np.mean(~kept.any(1)))
            if validation is not None:
                graph_metrics["diagnostics"]["val"] = _score_summary(validation_scores, config["near_zero"])
            for index, fraction in enumerate(config["top_fractions"]):
                k = min(scores.shape[1], max(1, int(np.ceil(fraction * scores.shape[1]))))
                top = _selection(scores, k, "top")
                jaccard = _jaccard_pairs(top, adjacent)
                budget = {"fraction": fraction, "k": k, "actual_fraction": k / scores.shape[1],
                          "adjacent_top_jaccard": float(jaccard.mean()) if len(jaccard) else None}
                graph_arrays[f"budget{index}_top_indices"] = np.argsort(-scores, axis=1, kind="stable")[:, :k]
                graph_arrays[f"budget{index}_jaccard"] = jaccard
                budget["interventions_computed"] = interventions_enabled
                budget["intervention_status"] = "computed" if interventions_enabled else "disabled_by_split_policy"
                if interventions_enabled:
                    print(f"[graph/{split}] {model.name}: budget={fraction:g}, K={k}, random={len(config['random_seeds'])}", flush=True)
                    for mode in ("top", "low"):
                        selection = top if mode == "top" else _selection(scores, k, "low")
                        for action in ("delete", "keep"):
                            masks = ~selection if action == "delete" else selection
                            value, per_window = _intervention(model, dataset, config, masks, data.mean, data.std)
                            key = f"{action}_{mode}"
                            budget[key] = value
                            budget[key + "_delta"] = value - metrics["mae"]
                            graph_arrays[f"budget{index}_{key}_window_delta"] = per_window - arrays["per_window_mae"]
                    random_values = {"delete": [], "keep": []}
                    for seed in config["random_seeds"]:
                        selection = _selection(scores, k, "random", seed)
                        for action in random_values:
                            masks = ~selection if action == "delete" else selection
                            value, per_window = _intervention(model, dataset, config, masks, data.mean, data.std)
                            random_values[action].append(value)
                            graph_arrays[f"budget{index}_{action}_random_seed{seed}_window_delta"] = per_window - arrays["per_window_mae"]
                    for action, values in random_values.items():
                        budget[f"{action}_random_mean"] = float(np.mean(values))
                        budget[f"{action}_random_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
                        budget[f"{action}_random_delta"] = float(np.mean(values)) - metrics["mae"]
                        budget[f"{action}_excess"] = budget[f"{action}_top"] - float(np.mean(values))
                        graph_arrays[f"budget{index}_{action}_random_seed_mae"] = np.asarray(values)
                    budget["random_seeds"] = config["random_seeds"]
                graph_metrics["budgets"].append(budget)
        else:
            graph_metrics["reason"] = "Fixed all-one gates have no learned edge ranking"
        if capabilities["supports_intervention"] and interventions_enabled:
            value, per_window = _intervention(model, dataset, config, np.zeros_like(scores, dtype=bool), data.mean, data.std)
            graph_metrics["all_messages_off_mae"] = value
            graph_metrics["all_messages_off_delta"] = value - metrics["mae"]
            graph_arrays["all_messages_off_window_delta"] = per_window - arrays["per_window_mae"]
    else:
        graph_metrics["interventions_computed"] = False
        graph_metrics["intervention_status"] = "not_applicable"
        graph_metrics["reason"] = "Model has no edge scores, or candidate graph is empty"
    metrics["evaluation_seconds"] = time.perf_counter() - started
    return {"metrics": metrics, "arrays": arrays, "graph_metrics": graph_metrics,
            "graph_arrays": graph_arrays, "evaluation_config": config}
