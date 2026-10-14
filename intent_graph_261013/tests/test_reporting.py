"""Suite reports preserve split ownership, exact zeroes, and missing evidence."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_runs import compare_runs
from experiment.artifacts import target_fingerprint
from experiment.reporting import build_suite_reports, write_reports


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, name, model="dynamic_gate", seed=0, legacy=False, val_shift=0, fit_seconds=1.0):
        root = self.root / name
        root.mkdir()
        splits = {}
        for split, start in (("train", 1), ("val", 10 + val_shift), ("test", 20)):
            if legacy and split != "test":
                continue
            output = root if legacy else root / "splits" / split
            output.mkdir(parents=True, exist_ok=True)
            window_id = np.arange(start, start + 3)
            arrays = {"window_id": window_id,
                      "target_times": (window_id[:, None] + np.arange(1, 3)).astype("datetime64[m]"),
                      "target_mask": np.ones((3, 2, 2), dtype=bool),
                      "raw_target": np.zeros((3, 2, 2))}
            np.savez_compressed(output / "predictions.npz", **arrays, protocol_id=np.asarray("same"))
            error = {"train": 1.0, "val": 2.0, "test": 3.0}[split] + seed
            metrics = {"windows": 3, "valid_targets": 12, "mae": error, "rmse": error + 0.5,
                       "mae_standardized": error / 10, "rmse_standardized": (error + 0.5) / 10,
                       "skill": 0.1, "skill_standardized": 0.2,
                       "persistence_mae": 5, "persistence_rmse": 6,
                       "persistence_mae_standardized": 0.5, "persistence_rmse_standardized": 0.6,
                       "per_horizon_mae": [error, error + 0.2],
                       "per_horizon_rmse": [error + 0.5, error + 0.7],
                       "per_horizon_mae_standardized": [error / 10, (error + 0.2) / 10],
                       "per_horizon_rmse_standardized": [(error + 0.5) / 10, (error + 0.7) / 10],
                       "per_horizon_valid_targets": [6, 6],
                       "prediction_seconds": 0.01, "prediction_windows_per_second": 300.0,
                       "evaluation_seconds": 0.02, "peak_gpu_allocated_mb": 1.25}
            if model == "persistence":
                graph = {"available": False, "score_kind": "none", "budgets": [],
                         "reason": "Model has no edge scores"}
            else:
                graph = {"available": True, "score_kind": "static" if model == "static_gate" else "dynamic",
                         "intervention_scope": "single_message",
                         "diagnostics": {"current": {"minimum": 0.5, "maximum": 0.5, "mean": 0.5,
                            "zero_fraction": 0, "near_zero_fraction": 0,
                            "all_zero_window_fraction": 0, "constant_window_fraction": 1,
                            "quantile_levels": [0, 0.5, 0.9, 0.99, 1], "quantiles": [0.5] * 5}},
                         "temporal": {"adjacent_pairs": 2, "mean_score_change": 0},
                         "threshold": {"enabled": True, "value": 0.5, "rule": "validation_positive_quantile",
                                       "mean_candidate_density": 1, "empty_window_fraction": 0},
                         "budgets": []}
                for fraction in (0.05, 0.1, 0.2):
                    budget = {"fraction": fraction, "k": 1, "actual_fraction": 0.5,
                              "adjacent_top_jaccard": 1}
                    if split == "test":
                        budget.update({"delete_top": error, "delete_top_delta": 0.0,
                            "delete_low": error, "delete_low_delta": 0.0,
                            "delete_random_mean": error, "delete_random_std": 0.0,
                            "delete_random_delta": 0.0, "delete_excess": 0.0,
                            "keep_top": error, "keep_top_delta": 0.0,
                            "keep_low": error, "keep_low_delta": 0.0,
                            "keep_random_mean": error, "keep_random_std": 0.0,
                            "keep_random_delta": 0.0, "keep_excess": 0.0, "random_seeds": [0, 1]})
                    graph["budgets"].append(budget)
                if split == "test":
                    graph["all_messages_off_mae"] = error
                    graph["all_messages_off_delta"] = 0.0
            splits[split] = {"metrics": metrics, "graph_metrics": graph,
                             "target_fingerprint": target_fingerprint(arrays),
                             "prediction_path": str((output / "predictions.npz").relative_to(root))}
        summary = {"status": "completed", "protocol_id": "same", "evaluation_signature": "same_eval",
                   "model": model, "seed": seed, "model_config": {}, "training_config": {},
                   "fit": {"kind": "none" if model == "persistence" else "gradient", "device": "cpu",
                           "fit_seconds": fit_seconds, "parameter_count": 30, "epochs": 5, "best_epoch": 2},
                   "metrics": splits["test"]["metrics"], "graph_metrics": splits["test"]["graph_metrics"],
                   "target_fingerprint": splits["test"]["target_fingerprint"],
                   "evaluation_config": {"top_fractions": [0.05, 0.1, 0.2]},
                   "data": {"step_minutes": 5}}
        if not legacy:
            summary.update({"artifact_layout": "split_v2", "split_results": splits})
        (root / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return root

    def test_all_splits_and_horizons_export_with_zero_graph_values(self):
        first = self.fixture("a", model="static_gate")
        unsupported = self.fixture("b", model="persistence")
        comparison = compare_runs([first, unsupported])
        self.assertEqual(comparison["split_coverage"], "all_runs_three_splits")
        tables = build_suite_reports(comparison)
        self.assertEqual(len(tables["prediction_summary"]), 6)
        self.assertEqual(len(tables["horizon_summary"]), 12)
        self.assertEqual(len(tables["graph_summary"]), 18)
        self.assertEqual(len(tables["diagnostics_summary"]), 6)
        self.assertEqual(len(tables["efficiency_summary"]), 2)
        test = [row for row in tables["graph_summary"] if row["model"] == "static_gate" and row["split"] == "test"]
        self.assertTrue(all(row["delete_top_delta"] == 0 for row in test))
        self.assertTrue(all(row["intervention_status"] == "computed" for row in test))
        training = [row for row in tables["graph_summary"] if row["model"] == "static_gate" and row["split"] == "train"]
        self.assertTrue(all(row["intervention_status"] == "not_computed" and row["delete_top_delta"] is None for row in training))
        none = [row for row in tables["graph_summary"] if row["model"] == "persistence"]
        self.assertTrue(all(row["ranking_status"] == "not_applicable" and row["delete_top_delta"] is None for row in none))
        self.assertEqual(tables["efficiency_summary"][0]["test_minus_train_mae"], 2)
        output = self.root / "reports"
        files = write_reports(comparison, output)
        self.assertEqual(len(files), 8)
        with Path(files["graph_summary"]).open(encoding="utf-8-sig", newline="") as handle:
            saved = list(csv.DictReader(handle))
        self.assertEqual(len(saved), 18)
        unsupported_saved = next(row for row in saved if row["model"] == "persistence")
        self.assertEqual(unsupported_saved["delete_top_delta"], "")
        static_saved = next(row for row in saved if row["model"] == "static_gate" and row["split"] == "test")
        self.assertEqual(float(static_saved["delete_top_delta"]), 0.0)

    def test_validation_targets_checked_even_when_test_matches(self):
        first = self.fixture("a")
        changed = self.fixture("b", seed=1, val_shift=1)
        with self.assertRaisesRegex(ValueError, "for val"):
            compare_runs([first, changed])

    def test_legacy_reports_test_only_and_does_not_invent_train_gap(self):
        legacy = self.fixture("old", legacy=True)
        comparison = compare_runs([legacy])
        self.assertEqual(comparison["split_coverage"], "partial_legacy_split_coverage")
        tables = build_suite_reports(comparison)
        self.assertEqual(len(tables["prediction_summary"]), 1)
        self.assertEqual(tables["prediction_summary"][0]["split"], "test")
        self.assertIsNone(tables["efficiency_summary"][0]["test_minus_train_mae"])
        self.assertEqual(tables["efficiency_summary"][0]["gap_reason"], "legacy_training_split_not_evaluated")

    def test_training_seed_and_selection_seed_sd_remain_separate(self):
        first = self.fixture("a", seed=0)
        second = self.fixture("b", seed=1)
        comparison = compare_runs([first, second])
        group = comparison["groups"][0]
        self.assertAlmostEqual(group["mae_std"], np.sqrt(0.5))
        self.assertEqual(group["p10_delete_random_std_mean"], 0)
        self.assertEqual(group["p10_delete_random_std_std"], 0)
        self.assertEqual(group["independent_seeds"], 2)
        self.assertEqual(group["train_mae_mean"], 1.5)
        output = self.root / "report"
        write_reports(comparison, output)
        manifest = json.loads((output / "report_manifest.json").read_text())
        self.assertIn("training", manifest["aggregated_selection_seed_sd"])

    def test_deterministic_duplicates_ignore_different_elapsed_times(self):
        first = self.fixture("a", model="persistence", seed=0, fit_seconds=0.1)
        second = self.fixture("b", model="persistence", seed=0, fit_seconds=0.9)
        comparison = compare_runs([first, second])
        group = comparison["groups"][0]
        self.assertEqual(group["independent_seeds"], 0)
        self.assertIsNone(group["mae_std"])
        self.assertAlmostEqual(group["fit_seconds_mean"], 0.5)
        self.assertIsNone(group["fit_seconds_std"])

    def test_fixed_graph_separates_ranking_and_message_ablation_status(self):
        path = self.fixture("fixed", model="fixed_graph")
        summary_path = path / "run_summary.json"
        summary = json.loads(summary_path.read_text())
        summary["capabilities"] = {"supports_ranking": False, "supports_intervention": True,
                                   "score_kind": "fixed", "intervention_scope": "single_message"}
        for split, result in summary["split_results"].items():
            graph = result["graph_metrics"]
            graph["available"], graph["budgets"], graph["score_kind"] = False, [], "fixed"
            graph["reason"] = "Fixed all-one gates have no learned edge ranking"
            graph["intervention_status"] = "computed" if split == "test" else "disabled_by_split_policy"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        comparison = compare_runs([path])
        tables = build_suite_reports(comparison)
        test = next(row for row in tables["graph_summary"] if row["split"] == "test")
        self.assertEqual(test["ranking_intervention_status"], "not_applicable")
        self.assertEqual(test["message_ablation_status"], "computed")
        self.assertEqual(test["all_messages_off_delta"], 0)
        training = next(row for row in tables["graph_summary"] if row["split"] == "train")
        self.assertEqual(training["message_ablation_status"], "not_computed")
        group = comparison["groups"][0]
        self.assertEqual(group["graph_message_ablation_status"], "computed")
        self.assertEqual(group["graph_ranking_intervention_status"], "not_applicable")
        self.assertEqual(group["prediction_windows_per_second_mean"], 300)
        self.assertEqual(group["peak_gpu_allocated_mb_mean"], 1.25)
        efficiency = tables["efficiency_summary"][0]
        self.assertEqual(efficiency["test_prediction_windows_per_second"], 300)
        self.assertEqual(efficiency["train_peak_gpu_allocated_mb"], 1.25)


if __name__ == "__main__":
    unittest.main()
