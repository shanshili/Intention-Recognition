"""Save numerical evidence before rendering; figures never trigger inference."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, (Path, np.datetime64)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: str | Path, payload) -> None:
    Path(path).write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding="utf-8")


def target_fingerprint(arrays: dict) -> str:
    """Actual ordered evaluation targets, masks, horizon times, and window IDs."""
    digest = hashlib.sha256()
    for key in ("window_id", "target_times", "target_mask"):
        array = np.asarray(arrays[key])
        digest.update(key.encode())
        digest.update(str(array.shape).encode())
        if key == "target_times":
            array = array.astype("datetime64[ns]").astype("<i8")
        elif key == "window_id":
            array = array.astype("<i8")
        else:
            array = array.astype(np.uint8)
        digest.update(np.ascontiguousarray(array).tobytes())
    if "raw_target" in arrays:
        raw = np.asarray(arrays["raw_target"], dtype="<f8")
        mask = np.asarray(arrays["target_mask"], dtype=bool)
        if raw.shape != mask.shape:
            raise ValueError("raw_target and target_mask shapes differ")
        values = np.where(mask, raw, 0.0)
        digest.update(b"raw_target")
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def save_run(output_dir, data, model, fit, evaluation, summary, make_plots=True) -> dict:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.asarray(value) for key, value in evaluation["arrays"].items()
              if value is not None}
    summary = dict(summary)
    summary["target_fingerprint"] = target_fingerprint(arrays)
    common = {"mean": data.mean, "std": data.std,
              "node_ids": np.asarray(data.node_ids, dtype=str),
              "coords": data.coords, "candidate_edges": data.graph["edges"],
              "protocol_id": np.asarray(data.protocol_id)}
    prediction_arrays = {key: value for key, value in arrays.items()
                         if key not in {"edge_scores", "effective_weight"}}
    np.savez_compressed(destination / "predictions.npz", **prediction_arrays, **common)
    write_json(destination / "resolved_config.json", {
        "protocol": data.protocol, "model": summary.get("model"),
        "model_config": summary.get("model_config", {}),
        "training_config": summary.get("training_config", {}),
        "evaluation_config": summary.get("evaluation_config", {}),
        "protocol_id": data.protocol_id,
        "evaluation_signature": summary.get("evaluation_signature"),
    })
    write_json(destination / "training_history.json", fit.get("history", []))
    graph_metrics = evaluation.get("graph_metrics", {})
    write_json(destination / "graph_evaluation.json", graph_metrics)
    with (destination / "per_window_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["window_id", "first_target_time", "last_target_time", "mae_raw",
                         "persistence_mae_raw", "valid_targets"])
        mask = arrays["target_mask"]
        for row, window_id in enumerate(arrays["window_id"]):
            writer.writerow([int(window_id), str(arrays["target_times"][row, 0]),
                             str(arrays["target_times"][row, -1]),
                             json_safe(arrays["per_window_mae"][row]),
                             json_safe(arrays["per_window_persistence_mae"][row]),
                             int(mask[row].sum())])
    if "edge_scores" in arrays:
        metadata = getattr(model, "capabilities", {})
        edge_arrays = {"edge_scores": arrays["edge_scores"],
                       "window_id": arrays["window_id"], "target_times": arrays["target_times"],
                       "src": data.graph["src"], "dst": data.graph["dst"],
                       "degree": data.graph["degree"], "edge_features": data.graph["edge_feats"],
                       "score_kind": np.asarray(metadata.get("score_kind", "unknown")),
                       "intervention_scope": np.asarray(metadata.get("intervention_scope") or "none"),
                       **common}
        if "effective_weight" in arrays:
            edge_arrays["effective_weight"] = arrays["effective_weight"]
        np.savez_compressed(destination / "edge_scores.npz", **edge_arrays)
    if evaluation.get("graph_arrays"):
        np.savez_compressed(destination / "graph_evaluation.npz",
                            **{key: np.asarray(value) for key, value in evaluation["graph_arrays"].items()
                               if value is not None})
    plots = []
    if make_plots:
        from .visualization import make_run_plots
        plots = make_run_plots(destination, fit, evaluation, data)
    write_json(destination / "run_summary.json", summary)
    return {"output_dir": str(destination.resolve()), "target_fingerprint": summary["target_fingerprint"],
            "plots": plots}


def save_multisplit_run(output_dir, data, model, fit, evaluations, summary,
                       make_plots=True) -> dict:
    """Save every partition without duplicating the test arrays at run root.

    The root summary retains test metrics for compatibility and points to each
    split's complete evidence. It is written only after every split finishes.
    Rendering is limited to the test split; numerical arrays are never sampled.
    """
    split_order = ("train", "val", "test")
    if set(evaluations) != set(split_order):
        raise ValueError("Multisplit artifacts require exactly train, val, and test evaluations")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    root_summary = dict(summary)
    root_summary["artifact_layout"] = "split_v2"
    root_summary["metrics"] = evaluations["test"]["metrics"]
    root_summary["graph_metrics"] = evaluations["test"].get("graph_metrics", {})
    split_results, plots = {}, []
    for split in split_order:
        evaluation = evaluations[split]
        split_config = dict(evaluation.get("evaluation_config", summary.get("evaluation_config", {})))
        split_config["split"] = split
        split_summary = {
            **summary,
            "artifact_layout": "single_split_v2",
            "split": split,
            "metrics": evaluation["metrics"],
            "graph_metrics": evaluation.get("graph_metrics", {}),
            "evaluation_config": split_config,
        }
        saved = save_run(destination / "splits" / split, data, model, fit, evaluation,
                         split_summary, make_plots=bool(make_plots and split == "test"))
        result = {"metrics": evaluation["metrics"],
                  "graph_metrics": evaluation.get("graph_metrics", {}),
                  "prediction_path": f"splits/{split}/predictions.npz",
                  "target_fingerprint": saved["target_fingerprint"]}
        split_results[split] = result
        plots.extend(saved["plots"])
    root_summary["target_fingerprint"] = split_results["test"]["target_fingerprint"]
    root_summary["split_results"] = split_results
    write_json(destination / "resolved_config.json", {
        "protocol": data.protocol, "model": summary.get("model"),
        "model_config": summary.get("model_config", {}),
        "training_config": summary.get("training_config", {}),
        "evaluation_config": summary.get("evaluation_config", {}),
        "protocol_id": data.protocol_id,
        "evaluation_signature": summary.get("evaluation_signature"),
        "artifact_layout": "split_v2", "saved_splits": list(split_order),
    })
    write_json(destination / "training_history.json", fit.get("history", []))
    write_json(destination / "split_metrics.json", split_results)
    with (destination / "per_seed_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        metric_names = ("mae", "rmse", "mae_standardized", "rmse_standardized", "skill",
                        "persistence_mae", "persistence_rmse", "windows", "valid_targets")
        fields = ["model", "seed", "split", "protocol_id", "evaluation_signature", *metric_names]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split in split_order:
            metrics = split_results[split]["metrics"]
            row = {"model": summary.get("model"), "seed": summary.get("seed"), "split": split,
                   "protocol_id": data.protocol_id,
                   "evaluation_signature": summary.get("evaluation_signature")}
            row.update({key: json_safe(metrics.get(key)) for key in metric_names})
            writer.writerow(row)
    # A completed root summary is the final artifact, so a failed split/render
    # cannot leave a run claiming it saved all three partitions.
    write_json(destination / "run_summary.json", root_summary)
    return {"output_dir": str(destination.resolve()),
            "target_fingerprint": root_summary["target_fingerprint"],
            "split_results": split_results, "plots": plots}
