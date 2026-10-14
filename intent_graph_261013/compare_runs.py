"""Strict comparison of completed runs sharing ordered targets and protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiment.artifacts import target_fingerprint, write_json


SCALARS = ("mae", "rmse", "mae_standardized", "rmse_standardized", "skill", "skill_standardized",
           "persistence_mae", "persistence_rmse", "persistence_mae_standardized", "persistence_rmse_standardized")
SPLIT_ORDER = ("train", "val", "test")


def _split_results(summary):
    """Legacy artifacts contain test only; never invent their training results."""
    if summary.get("artifact_layout") == "split_v2":
        splits = summary.get("split_results", {})
        if set(splits) != set(SPLIT_ORDER):
            raise ValueError("split_v2 artifacts must include train, val, and test results")
        return splits
    if summary.get("split_results"):
        splits = summary["split_results"]
        if "test" not in splits or not set(splits) <= set(SPLIT_ORDER):
            raise ValueError("Split results need test and may contain only train, val, test")
        return splits
    return {"test": {"metrics": summary["metrics"], "graph_metrics": summary.get("graph_metrics", {}),
                     "prediction_path": "predictions.npz",
                     "target_fingerprint": summary.get("target_fingerprint")}}


def _fingerprint_split(path, summary, split, result):
    supplied = result.get("prediction_path", f"splits/{split}/predictions.npz")
    prediction_path = (path / supplied).resolve()
    if not prediction_path.is_relative_to(path):
        raise ValueError(f"Prediction path escapes run directory: {prediction_path}")
    with np.load(prediction_path, allow_pickle=False) as stored:
        required = {name: stored[name] for name in ("window_id", "target_times", "target_mask", "raw_target")}
        fingerprint = target_fingerprint(required)
        if "protocol_id" in stored and str(stored["protocol_id"].item()) != summary["protocol_id"]:
            raise ValueError(f"Prediction protocol ID disagrees with summary: {path}/{split}")
        mask = required["target_mask"]
        if (mask.ndim != 3 or mask.dtype != np.bool_ or required["raw_target"].shape != mask.shape
                or len(required["window_id"]) != mask.shape[0]
                or required["target_times"].shape != (mask.shape[0], mask.shape[-1])):
            raise ValueError(f"Malformed evaluation arrays: {path}/{split}")
        metrics = result["metrics"]
        if int(metrics.get("windows", -1)) != len(required["window_id"]):
            raise ValueError(f"Saved window count disagrees with metrics: {path}/{split}")
        if int(metrics.get("valid_targets", -1)) != int(mask.sum()):
            raise ValueError(f"Saved valid target count disagrees with metrics: {path}/{split}")
    if result.get("target_fingerprint") and result["target_fingerprint"] != fingerprint:
        raise ValueError(f"Actual targets disagree with saved fingerprint: {path}/{split}")
    return fingerprint


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def _budget_label(fraction):
    percentage = float(fraction) * 100
    return "p" + (f"{int(percentage):02d}" if percentage.is_integer() else f"{percentage:g}".replace(".", "_"))


def _wide_test_graph(graph):
    """Flat numeric fields retain seed SD as an explicitly named statistic."""
    result = {}
    for key in ("all_messages_off_mae", "all_messages_off_delta"):
        result["graph_" + key] = graph.get(key)
    diagnostics = graph.get("diagnostics", {})
    # New evaluator names this split's distribution 'current'; older ones
    # called it 'test'. Never accidentally use the validation distribution.
    current = diagnostics.get("current", diagnostics.get("test", {}))
    for key in ("minimum", "maximum", "mean", "zero_fraction", "near_zero_fraction",
                "all_zero_window_fraction", "constant_window_fraction"):
        result["gate_" + key] = current.get(key)
    for level, value in zip(current.get("quantile_levels", []), current.get("quantiles", [])):
        result["gate_q" + f"{100 * level:g}".replace(".", "_")] = value
    temporal = graph.get("temporal", {})
    result["gate_mean_score_change"] = temporal.get("mean_score_change")
    result["gate_adjacent_pairs"] = temporal.get("adjacent_pairs")
    threshold = graph.get("threshold", {})
    result["gate_threshold"] = threshold.get("value")
    result["gate_threshold_density"] = threshold.get("mean_candidate_density")
    result["gate_threshold_empty_window_fraction"] = threshold.get("empty_window_fraction")
    for budget in graph.get("budgets", []):
        prefix = _budget_label(budget["fraction"]) + "_"
        for key, value in budget.items():
            if key not in {"random_seeds", "fraction"} and (_finite_number(value) or value is None):
                result[prefix + key] = value
        if "random_seeds" in budget:
            result[prefix + "random_seed_count"] = len(budget["random_seeds"])
    return result


def _configuration_key(summary):
    training = {key: value for key, value in summary.get("training_config", {}).items()
                if key not in {"seed", "seeds"}}
    payload = {"model": summary["model"], "model_config": summary.get("model_config", {}),
               "training_config": training,
               "declared_signature": summary.get("run_config_signature"),
               "fit_kind": summary.get("fit", {}).get("kind", "unknown"),
               "actual_device": summary.get("fit", {}).get("device", "cpu"),
               "score_kind": summary.get("capabilities", {}).get("score_kind"),
               "intervention_scope": summary.get("capabilities", {}).get("intervention_scope"),
               "source_sha256": summary.get("code_version", {}).get("experiment_source_sha256"),
               "package_versions": summary.get("environment", {}).get("packages", {})}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()[:20]


def compare_runs(run_paths) -> dict:
    runs = []
    reference_fingerprints = {}
    for supplied in run_paths:
        path = Path(supplied).resolve()
        if path.name == "run_summary.json":
            path = path.parent
        with (path / "run_summary.json").open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if not summary.get("protocol_id") or not summary.get("evaluation_signature"):
            raise ValueError(f"Run lacks protocol/evaluation identity: {path}")
        if summary.get("status", "completed") != "completed":
            raise ValueError(f"Only completed runs can be compared: {path}")
        splits = _split_results(summary)
        fingerprints = {split: _fingerprint_split(path, summary, split, result)
                        for split, result in splits.items()}
        fingerprint = fingerprints["test"]
        if runs:
            reference = runs[0]
            if summary["protocol_id"] != reference["summary"]["protocol_id"]:
                raise ValueError("Runs have different protocol IDs and cannot be compared")
            if summary["evaluation_signature"] != reference["summary"]["evaluation_signature"]:
                raise ValueError("Runs have different evaluation signatures and cannot be compared")
        for split, value in fingerprints.items():
            if split in reference_fingerprints and value != reference_fingerprints[split]:
                raise ValueError(f"Runs have different ordered windows/targets/masks for {split} and cannot be compared")
            reference_fingerprints.setdefault(split, value)
        runs.append({"path": str(path), "summary": summary, "fingerprint": fingerprint,
                     "splits": splits, "split_fingerprints": fingerprints})
    if not runs:
        raise ValueError("At least one run is required")
    rows, groups, details = [], {}, []
    for run in runs:
        summary = run["summary"]
        config_key = _configuration_key(summary)
        metrics = run["splits"]["test"]["metrics"]
        graph = run["splits"]["test"].get("graph_metrics", {})
        fit = summary.get("fit", {})
        row = {"run": run["path"], "model": summary["model"], "seed": summary.get("seed"),
               "config_signature": config_key, "protocol_id": summary["protocol_id"],
               "evaluation_signature": summary["evaluation_signature"],
               "fit_kind": summary.get("fit", {}).get("kind", "unknown"),
               "device": summary.get("fit", {}).get("device", "cpu"),
               "intervention_scope": summary.get("capabilities", {}).get("intervention_scope"),
               "fit_seconds": summary.get("fit", {}).get("fit_seconds"),
               "fit_peak_gpu_allocated_mb": fit.get("peak_gpu_allocated_mb"),
               "parameter_count": summary.get("fit", {}).get("parameter_count"),
               "windows": metrics["windows"], "valid_targets": metrics["valid_targets"],
               "epochs": fit.get("epochs"), "best_epoch": fit.get("best_epoch"),
               "prediction_seconds": metrics.get("prediction_seconds"),
               "prediction_windows_per_second": metrics.get("prediction_windows_per_second"),
               "peak_gpu_allocated_mb": metrics.get("peak_gpu_allocated_mb"),
               "evaluation_seconds": metrics.get("evaluation_seconds"),
               "available_splits": ",".join(split for split in SPLIT_ORDER if split in run["splits"]),
               "graph_ranking_status": "available" if graph.get("available") else "not_applicable",
               "graph_reason": graph.get("reason"),
               "graph_intervention_status": graph.get("intervention_status", "computed" if any(
                   "delete_top" in budget for budget in graph.get("budgets", [])) else "not_computed")}
        row["graph_message_ablation_status"] = ("computed" if "all_messages_off_mae" in graph else
                                                "not_computed" if summary.get("capabilities", {}).get("supports_intervention")
                                                else "not_applicable")
        row["graph_ranking_intervention_status"] = ("not_applicable" if not graph.get("available") else
                                                     "computed" if any("delete_top" in budget for budget in graph.get("budgets", []))
                                                     else "not_computed")
        row.update({key: metrics.get(key) for key in SCALARS})
        row.update(_wide_test_graph(graph))
        for split in SPLIT_ORDER:
            split_metrics = run["splits"].get(split, {}).get("metrics", {})
            row.update({f"{split}_{key}": split_metrics.get(key) for key in SCALARS})
        rows.append(row)
        groups.setdefault((summary["model"], config_key), []).append(row)
        details.append({"run": run["path"], "model": summary["model"], "seed": summary.get("seed"),
                        "config_signature": config_key, "protocol_id": summary["protocol_id"],
                        "evaluation_signature": summary["evaluation_signature"],
                        "artifact_layout": summary.get("artifact_layout", "legacy_test_only"),
                        "fit": {key: value for key, value in fit.items() if key != "history"},
                        "capabilities": summary.get("capabilities", {}), "data": summary.get("data", {}),
                        "evaluation_config": summary.get("evaluation_config", {}),
                        "split_results": run["splits"], "split_fingerprints": run["split_fingerprints"]})
    aggregates = []
    for (model, signature), members in groups.items():
        deterministic = all(member["fit_kind"] in {"none", "closed_form"} for member in members)
        if deterministic:
            # Duplicate deterministic fits are records, not independent seeds.
            predictive_keys = list(SCALARS) + [f"{split}_{key}" for split in SPLIT_ORDER for key in SCALARS]
            for key in predictive_keys:
                values = [member[key] for member in members if member[key] is not None]
                if len(values) > 1 and not np.allclose(values, values[0], rtol=1e-10, atol=1e-12):
                    raise ValueError(f"Deterministic duplicates disagree for {model}: {key}")
            effective = members[:1]
        else:
            seeds = [member["seed"] for member in members]
            if len(set(seeds)) != len(seeds):
                raise ValueError(f"Repeated training seeds in configuration {model}/{signature}")
            effective = members
        aggregate = {"model": model, "config_signature": signature, "runs": len(members),
                     "device": members[0]["device"], "intervention_scope": members[0]["intervention_scope"],
                     "independent_seeds": 0 if deterministic else len(effective),
                     "deterministic": deterministic,
                     "available_splits": ",".join(split for split in SPLIT_ORDER if any(
                         split in member["available_splits"].split(",") for member in members)),
                     "graph_ranking_status": members[0]["graph_ranking_status"],
                     "graph_intervention_status": members[0]["graph_intervention_status"],
                     "graph_message_ablation_status": members[0]["graph_message_ablation_status"],
                     "graph_ranking_intervention_status": members[0]["graph_ranking_intervention_status"],
                     "graph_reason": members[0]["graph_reason"],
                     "seed_std_kind": "sample_SD_across_independent_training_seeds"}
        numeric_keys = list(SCALARS)
        for member in members:
            for key, value in member.items():
                if key not in {"seed"} and key not in numeric_keys and _finite_number(value):
                    numeric_keys.append(key)
        for key in numeric_keys:
            source_members = members if key in {"fit_seconds", "prediction_seconds", "prediction_windows_per_second",
                                               "evaluation_seconds", "peak_gpu_allocated_mb"} else effective
            values = [member.get(key) for member in source_members if _finite_number(member.get(key))]
            aggregate[f"{key}_mean"] = float(np.mean(values)) if values else None
            aggregate[f"{key}_std"] = (float(np.std(values, ddof=1))
                                         if len(values) > 1 and not deterministic else None)
            aggregate[f"{key}_n"] = len(values)
        aggregates.append(aggregate)
    return {"protocol_id": runs[0]["summary"]["protocol_id"],
            "evaluation_signature": runs[0]["summary"]["evaluation_signature"],
            "target_fingerprint": runs[0]["fingerprint"], "split_fingerprints": reference_fingerprints,
            "split_coverage": "all_runs_three_splits" if all(set(run["splits"]) == set(SPLIT_ORDER) for run in runs)
                              else "partial_legacy_split_coverage",
            "runs": rows, "groups": aggregates, "run_details": details}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="Completed run directories")
    parser.add_argument("--out", type=Path, help="Exact CSV or JSON output filename; otherwise print JSON")
    args = parser.parse_args(argv)
    comparison = compare_runs(args.runs)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.suffix.lower() == ".json":
            write_json(args.out, comparison)
        elif args.out.suffix.lower() == ".csv":
            with args.out.open("w", encoding="utf-8", newline="") as handle:
                columns = list(dict.fromkeys(key for row in comparison["groups"] for key in row))
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(comparison["groups"])
        else:
            parser.error("--out must end in .csv or .json")
    else:
        print(json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False))
    return comparison


if __name__ == "__main__":
    main()
