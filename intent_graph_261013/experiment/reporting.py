"""Suite-wide mathematical evidence tables; undefined metrics stay blank.

This module consumes validated comparison summaries only, without loading a
model, retraining, generating scores, or filling uncomputed graph experiments.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .artifacts import write_json


SPLITS = ("train", "val", "test")
PREDICTION_FIELDS = (
    "windows", "valid_targets", "mae", "rmse", "mae_standardized", "rmse_standardized",
    "skill", "skill_standardized", "persistence_mae", "persistence_rmse",
    "persistence_mae_standardized", "persistence_rmse_standardized",
    "prediction_seconds", "prediction_windows_per_second", "evaluation_seconds", "peak_gpu_allocated_mb",
)
HORIZON_FIELDS = ("mae", "rmse", "mae_standardized", "rmse_standardized", "valid_targets")
GRAPH_BUDGET_FIELDS = (
    "k", "actual_fraction", "adjacent_top_jaccard",
    "delete_top", "delete_top_delta", "delete_low", "delete_low_delta",
    "delete_random_mean", "delete_random_std", "delete_random_delta", "delete_excess",
    "keep_top", "keep_top_delta", "keep_low", "keep_low_delta",
    "keep_random_mean", "keep_random_std", "keep_random_delta", "keep_excess",
)
BASE_FIELDS = ("run", "model", "seed", "config_signature", "protocol_id", "evaluation_signature")


def _base(detail):
    return {key: detail.get(key) for key in BASE_FIELDS}


def _write_csv(path, rows, initial_columns=()):
    columns = list(dict.fromkeys([*initial_columns, *(key for row in rows for key in row)]))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _split_records(detail):
    for split in SPLITS:
        if split in detail["split_results"]:
            yield split, detail["split_results"][split]


def _distribution(graph, split):
    diagnostics = graph.get("diagnostics", {})
    # Version 1 used 'test' and 'val'; split-aware evaluators can use current
    # or the actual split name. Do not substitute another split's distribution.
    return diagnostics.get("current", diagnostics.get(split, {}))


def _intervention_status(graph):
    if not graph.get("available"):
        return "not_applicable"
    if any("delete_top" in budget for budget in graph.get("budgets", [])):
        return "computed"
    return "not_computed"


def _message_ablation_status(graph, capabilities):
    if "all_messages_off_mae" in graph:
        return "computed", ""
    if not capabilities.get("supports_intervention"):
        return "not_applicable", "model_does_not_support_message_intervention"
    return "not_computed", graph.get("intervention_status", "interventions_not_computed")


def build_suite_reports(comparison):
    """Return long-format rows for each available run and evaluated split."""
    tables = {name: [] for name in ("prediction_summary", "horizon_summary", "graph_summary",
                                   "diagnostics_summary", "efficiency_summary")}
    for detail in comparison.get("run_details", []):
        base = _base(detail)
        fit = detail.get("fit", {})
        capabilities = detail.get("capabilities", {})
        for split, result in _split_records(detail):
            metrics = result.get("metrics", {})
            prediction = {**base, "split": split, "status": "computed",
                          "target_fingerprint": detail.get("split_fingerprints", {}).get(split)}
            prediction.update({key: metrics.get(key) for key in PREDICTION_FIELDS})
            prediction["skill_reason"] = ("zero_persistence_error" if metrics.get("persistence_mae") == 0
                                           else "not_saved" if metrics.get("skill") is None else "")
            prediction["skill_standardized_reason"] = (
                "zero_persistence_error" if metrics.get("persistence_mae_standardized") == 0
                else "not_saved" if metrics.get("skill_standardized") is None else "")
            tables["prediction_summary"].append(prediction)
            horizon_arrays = {
                "mae": metrics.get("per_horizon_mae", []),
                "rmse": metrics.get("per_horizon_rmse", []),
                "mae_standardized": metrics.get("per_horizon_mae_standardized", []),
                "rmse_standardized": metrics.get("per_horizon_rmse_standardized", []),
                "valid_targets": metrics.get("per_horizon_valid_targets", []),
            }
            horizon_count = max((len(values) for values in horizon_arrays.values()), default=0)
            minutes = result.get("step_minutes", detail.get("data", {}).get("step_minutes"))
            for horizon in range(horizon_count):
                row = {**base, "split": split, "horizon": horizon + 1,
                       "horizon_minutes": None if minutes is None else (horizon + 1) * minutes,
                       "status": "computed"}
                row.update({key: values[horizon] if horizon < len(values) else None
                            for key, values in horizon_arrays.items()})
                row["reason"] = ("zero_observed_targets" if row["valid_targets"] == 0 else
                                 "some_horizon_metrics_not_saved" if any(row[field] is None for field in HORIZON_FIELDS) else "")
                tables["horizon_summary"].append(row)
            graph = result.get("graph_metrics", {})
            status = _intervention_status(graph)
            message_status, message_reason = _message_ablation_status(graph, capabilities)
            reason = (graph.get("reason") if status == "not_applicable" else
                      "interventions_disabled_for_split" if status == "not_computed" else "")
            budgets = graph.get("budgets", [])
            if not budgets:
                fractions = detail.get("evaluation_config", {}).get("top_fractions", [0.05, 0.1, 0.2])
                budgets = [{"fraction": fraction} for fraction in fractions]
            for budget in budgets:
                row = {**base, "split": split, "fraction": budget["fraction"],
                       "score_kind": graph.get("score_kind", capabilities.get("score_kind")),
                       "intervention_scope": graph.get("intervention_scope", capabilities.get("intervention_scope")),
                       "ranking_status": "computed" if graph.get("available") else "not_applicable",
                       "intervention_status": status, "reason": reason,
                       "ranking_intervention_status": status,
                       "message_ablation_status": message_status, "message_ablation_reason": message_reason,
                       "all_messages_off_mae": graph.get("all_messages_off_mae"),
                       "all_messages_off_delta": graph.get("all_messages_off_delta"),
                       "random_seed_count": len(budget["random_seeds"]) if "random_seeds" in budget else None,
                       "random_std_kind": "sample_SD_across_selection_seeds" if "random_seeds" in budget else None}
                row.update({key: budget.get(key) for key in GRAPH_BUDGET_FIELDS})
                tables["graph_summary"].append(row)
            distribution = _distribution(graph, split)
            diagnostic = {**base, "split": split,
                          "score_kind": graph.get("score_kind", capabilities.get("score_kind")),
                          "status": "computed" if distribution else "not_applicable",
                          "reason": "" if distribution else graph.get("reason", "no_edge_scores")}
            for key in ("minimum", "maximum", "mean", "zero_fraction", "near_zero_fraction",
                        "all_zero_window_fraction", "constant_window_fraction", "near_zero_cutoff"):
                diagnostic["gate_" + key] = distribution.get(key)
            quantiles = dict(zip(distribution.get("quantile_levels", []), distribution.get("quantiles", [])))
            for quantile in (0, 0.5, 0.9, 0.99, 1):
                diagnostic["gate_q" + f"{100 * quantile:g}".replace(".", "_")] = quantiles.get(quantile)
            temporal, threshold = graph.get("temporal", {}), graph.get("threshold", {})
            diagnostic.update({"adjacent_pairs": temporal.get("adjacent_pairs"),
                               "mean_score_change": temporal.get("mean_score_change"),
                               "threshold_status": "computed" if threshold.get("enabled") else "disabled",
                               "threshold_value": threshold.get("value"),
                               "threshold_rule": threshold.get("rule"),
                               "threshold_reason": threshold.get("reason"),
                               "candidate_density": threshold.get("mean_candidate_density"),
                               "empty_window_fraction": threshold.get("empty_window_fraction")})
            tables["diagnostics_summary"].append(diagnostic)
        efficiency = {**base, "device": fit.get("device", "cpu"), "gpu_name": fit.get("gpu_name"),
                      "fit_kind": fit.get("kind", "unknown"), "fit_seconds": fit.get("fit_seconds"),
                      "parameter_count": fit.get("parameter_count"), "epochs": fit.get("epochs"),
                      "best_epoch": fit.get("best_epoch"), "fit_peak_gpu_allocated_mb": fit.get("peak_gpu_allocated_mb"),
                      "status": "computed", "available_splits": ",".join(split for split, _ in _split_records(detail))}
        split_metrics = {split: result.get("metrics", {}) for split, result in _split_records(detail)}
        for split in SPLITS:
            for key in ("prediction_seconds", "prediction_windows_per_second", "evaluation_seconds", "peak_gpu_allocated_mb"):
                efficiency[f"{split}_{key}"] = split_metrics.get(split, {}).get(key)
        for split in ("val", "test"):
            for metric in ("mae", "rmse", "mae_standardized", "rmse_standardized"):
                train_value = split_metrics.get("train", {}).get(metric)
                split_value = split_metrics.get(split, {}).get(metric)
                efficiency[f"{split}_minus_train_{metric}"] = (
                    None if train_value is None or split_value is None else split_value - train_value)
        efficiency["gap_reason"] = "" if "train" in split_metrics else "legacy_training_split_not_evaluated"
        tables["efficiency_summary"].append(efficiency)
    return tables


def write_reports(comparison, suite_path):
    """Incrementally overwrite suite tables using completed, validated runs."""
    destination = Path(suite_path)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "comparison.json", comparison)
    _write_csv(destination / "comparison.csv", comparison.get("groups", []),
               ("model", "config_signature", "runs", "independent_seeds", "device"))
    tables = build_suite_reports(comparison)
    files = {"comparison_json": str(destination / "comparison.json"),
             "comparison_csv": str(destination / "comparison.csv")}
    for name, rows in tables.items():
        path = destination / (name + ".csv")
        _write_csv(path, rows, (*BASE_FIELDS, "split") if name != "efficiency_summary" else BASE_FIELDS)
        files[name] = str(path)
    write_json(destination / "report_manifest.json", {
        "protocol_id": comparison.get("protocol_id"),
        "evaluation_signature": comparison.get("evaluation_signature"),
        "split_coverage": comparison.get("split_coverage"),
        "completed_runs": len(comparison.get("run_details", [])),
        "rows": {key: len(value) for key, value in tables.items()},
        "blank_meaning": "undefined, unsupported, not computed, or not saved; consult status and reason columns",
        "training_seed_sd": "comparison *_std is sample SD across independent training seeds",
        "selection_seed_sd": "graph *_random_std is sample SD across random edge-selection seeds",
        "aggregated_selection_seed_sd": "comparison *_random_std_mean averages selection-seed SD across training runs; *_random_std_std is its across-training-seed SD",
        "truth_metrics": "precision, recall and structural recovery metrics are not fabricated without edge truth",
    })
    files["manifest"] = str(destination / "report_manifest.json")
    return files


save_suite_reports = write_reports
