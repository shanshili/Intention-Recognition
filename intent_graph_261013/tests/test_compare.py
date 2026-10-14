"""Comparisons reject protocol drift and partial/different saved targets."""

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from compare_runs import compare_runs
from experiment.artifacts import save_multisplit_run, save_run, target_fingerprint


class CompareRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def save_fixture(self, name, protocol="same", windows=None, seed=1, kind="gradient", mae=2.0):
        path = self.root / name
        path.mkdir()
        windows = np.asarray([12, 13, 14] if windows is None else windows)
        count = len(windows)
        arrays = {"window_id": windows, "target_times": (windows[:, None] + np.arange(1, 4)).astype("datetime64[m]"),
                  "target_mask": np.ones((count, 2, 3), dtype=bool),
                  "raw_target": np.zeros((count, 2, 3))}
        np.savez_compressed(path / "predictions.npz", **arrays, protocol_id=np.asarray(protocol))
        summary = {"protocol_id": protocol, "evaluation_signature": "eval1", "model": "model",
                   "model_config": {}, "training_config": {"seeds": [1, 2]}, "seed": seed,
                   "fit": {"kind": kind}, "target_fingerprint": target_fingerprint(arrays),
                   "metrics": {"windows": count, "valid_targets": count * 6, "mae": mae,
                               "rmse": 3, "mae_standardized": 0.2, "rmse_standardized": 0.3, "skill": 0.1}}
        (path / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return path

    def test_same_protocol_aggregates_sample_seed_sd(self):
        first = self.save_fixture("a", seed=1, mae=2)
        second = self.save_fixture("b", seed=2, mae=4)
        result = compare_runs([first, second])
        self.assertEqual(result["groups"][0]["mae_mean"], 3)
        self.assertAlmostEqual(result["groups"][0]["mae_std"], np.sqrt(2))
        self.assertEqual(result["groups"][0]["independent_seeds"], 2)

    def test_protocol_id_mismatch_is_rejected(self):
        first = self.save_fixture("a")
        second = self.save_fixture("b", protocol="other", seed=2)
        with self.assertRaisesRegex(ValueError, "protocol IDs"):
            compare_runs([first, second])

    def test_actual_window_mismatch_and_tail_loss_are_rejected(self):
        first = self.save_fixture("a")
        changed = self.save_fixture("b", windows=[12, 13, 15], seed=2)
        truncated = self.save_fixture("c", windows=[12, 13], seed=2)
        for alternative in (changed, truncated):
            with self.assertRaisesRegex(ValueError, "different ordered"):
                compare_runs([first, alternative])

    def test_deterministic_duplicates_do_not_manufacture_seed_statistics(self):
        first = self.save_fixture("a", kind="closed_form", seed=1)
        second = self.save_fixture("b", kind="closed_form", seed=2)
        group = compare_runs([first, second])["groups"][0]
        self.assertEqual(group["independent_seeds"], 0)
        self.assertIsNone(group["mae_std"])

    def test_actual_device_and_intervention_scope_separate_seed_groups(self):
        first = self.save_fixture("a", seed=1)
        second = self.save_fixture("b", seed=2)
        third = self.save_fixture("c", seed=3)
        for path, device, scope in [(first, "cpu", "message"), (second, "cuda:0", "message"),
                                    (third, "cpu", "different_layer")]:
            summary_path = path / "run_summary.json"
            summary = json.loads(summary_path.read_text())
            summary["run_config_signature"] = "same_declared_auto_device_configuration"
            summary["fit"]["device"] = device
            summary["capabilities"] = {"intervention_scope": scope, "score_kind": "gate"}
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
        result = compare_runs([first, second, third])
        self.assertEqual(len(result["groups"]), 3)
        self.assertTrue(all(group["independent_seeds"] == 1 for group in result["groups"]))

    def test_artifact_saving_preserves_tail_without_model_calls(self):
        arrays = {"window_id": np.asarray([11, 12, 13]),
                  "target_times": np.asarray([[12, 13], [13, 14], [14, 15]], dtype="datetime64[m]"),
                  "prediction": np.ones((3, 2, 2)), "target": np.zeros((3, 2, 2)),
                  "raw_prediction": np.ones((3, 2, 2)), "raw_target": np.zeros((3, 2, 2)),
                  "target_mask": np.ones((3, 2, 2), dtype=bool),
                  "edge_scores": np.asarray([[0, 1], [0.5, 1], [0, 0]]),
                  "effective_weight": np.asarray([[0, 1], [0.5, 1], [0, 0]]),
                  "per_window_mae": np.ones(3), "per_window_persistence_mae": np.ones(3)}
        graph = {"edges": np.asarray([[0, 1], [1, 0]]), "src": np.asarray([0, 1]),
                 "dst": np.asarray([1, 0]), "degree": np.ones(2), "edge_feats": np.zeros((2, 8))}
        data = SimpleNamespace(mean=np.zeros(2), std=np.ones(2), coords=np.zeros((2, 2)),
                               node_ids=["a", "b"], graph=graph, protocol_id="same", protocol={})
        model = SimpleNamespace(capabilities={"score_kind": "gate", "intervention_scope": "single_message"})
        summary = {"model": "mock", "protocol_id": "same", "evaluation_signature": "eval1"}
        output = self.root / "saved"
        result = save_run(output, data, model, {"history": []},
                          {"arrays": arrays, "metrics": {}, "graph_metrics": {"available": True}},
                          summary, make_plots=False)
        self.assertEqual(result["target_fingerprint"], target_fingerprint(arrays))
        with np.load(output / "predictions.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["window_id"], [11, 12, 13])
        with np.load(output / "edge_scores.npz", allow_pickle=False) as saved:
            self.assertEqual(saved["score_kind"].item(), "gate")
            self.assertEqual(saved["intervention_scope"].item(), "single_message")
        self.assertEqual(len((output / "per_window_metrics.csv").read_text().splitlines()), 4)

    def _multisplit_fixture(self):
        graph = {"edges": np.asarray([[0, 1], [1, 0]]), "src": np.asarray([0, 1]),
                 "dst": np.asarray([1, 0]), "degree": np.ones(2), "edge_feats": np.zeros((2, 8))}
        data = SimpleNamespace(mean=np.zeros(2), std=np.ones(2), coords=np.zeros((2, 2)),
                               node_ids=["a", "b"], graph=graph, protocol_id="same", protocol={})
        model = SimpleNamespace(capabilities={"score_kind": "gate", "intervention_scope": "single_message"})
        summary = {"status": "completed", "model": "mock", "seed": 0, "protocol_id": "same",
                   "evaluation_signature": "eval1", "evaluation_config": {"batch_size": 10}}
        evaluations = {}
        for split in ("train", "val", "test"):
            arrays = {"window_id": np.asarray([11, 12, 13]),
                      "target_times": np.asarray([[12, 13], [13, 14], [14, 15]], dtype="datetime64[m]"),
                      "prediction": np.ones((3, 2, 2)), "target": np.zeros((3, 2, 2)),
                      "raw_prediction": np.ones((3, 2, 2)), "raw_target": np.zeros((3, 2, 2)),
                      "target_mask": np.asarray([[[1, 0], [1, 1]]] * 3, dtype=bool),
                      "edge_scores": np.asarray([[0, 1], [0.5, 1], [0, 0]]),
                      "effective_weight": np.asarray([[0, 1], [0.5, 1], [0, 0]]),
                      "per_window_mae": np.ones(3), "per_window_persistence_mae": np.ones(3)}
            evaluations[split] = {"arrays": arrays, "metrics": {"mae": 1, "windows": 3, "valid_targets": 9},
                                  "graph_metrics": {"available": True},
                                  "evaluation_config": {"batch_size": 10, "graph_interventions": split == "test"}}
        return data, model, summary, evaluations

    def test_multisplit_saving_complete_arrays_and_test_only_plots(self):
        data, model, summary, evaluations = self._multisplit_fixture()
        output = self.root / "multisplit"
        plotted = []

        def fake_plots(destination, fit, evaluation, received_data):
            plotted.append(Path(destination).name)
            return ["test_plot.png"]

        with patch("experiment.visualization.make_run_plots", side_effect=fake_plots):
            saved = save_multisplit_run(output, data, model, {"history": []},
                                       evaluations, summary, make_plots=True)
        self.assertEqual(plotted, ["test"])
        self.assertFalse((output / "predictions.npz").exists())
        self.assertFalse((output / "edge_scores.npz").exists())
        root_summary = json.loads((output / "run_summary.json").read_text())
        self.assertEqual(root_summary["artifact_layout"], "split_v2")
        self.assertEqual(root_summary["metrics"], evaluations["test"]["metrics"])
        self.assertEqual(set(root_summary["split_results"]), {"train", "val", "test"})
        self.assertEqual(saved["target_fingerprint"], root_summary["split_results"]["test"]["target_fingerprint"])
        for split in ("train", "val", "test"):
            nested = output / "splits" / split
            self.assertEqual(root_summary["split_results"][split]["prediction_path"], f"splits/{split}/predictions.npz")
            with np.load(nested / "predictions.npz", allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays["target_mask"], evaluations[split]["arrays"]["target_mask"])
                np.testing.assert_array_equal(arrays["window_id"], [11, 12, 13])
            nested_summary = json.loads((nested / "run_summary.json").read_text())
            self.assertEqual(nested_summary["split"], split)
            self.assertEqual(nested_summary["evaluation_config"]["split"], split)
            self.assertEqual(nested_summary["evaluation_config"]["graph_interventions"], split == "test")
            self.assertTrue((nested / "edge_scores.npz").exists())
            self.assertEqual(len((nested / "per_window_metrics.csv").read_text().splitlines()), 4)
        self.assertEqual(len((output / "per_seed_summary.csv").read_text().splitlines()), 4)
        self.assertTrue((output / "split_metrics.json").exists())
        self.assertTrue((output / "resolved_config.json").exists())
        self.assertTrue((output / "training_history.json").exists())

    def test_multisplit_failure_does_not_write_completed_root_summary(self):
        data, model, summary, evaluations = self._multisplit_fixture()
        output = self.root / "failed"
        evaluations["val"]["arrays"].pop("target_mask")
        with self.assertRaises(KeyError):
            save_multisplit_run(output, data, model, {"history": []}, evaluations, summary, make_plots=False)
        self.assertFalse((output / "run_summary.json").exists())

    def test_multisplit_requires_all_three_partitions(self):
        data, model, summary, evaluations = self._multisplit_fixture()
        evaluations.pop("train")
        with self.assertRaisesRegex(ValueError, "exactly train"):
            save_multisplit_run(self.root / "partial", data, model, {}, evaluations, summary, make_plots=False)


if __name__ == "__main__":
    unittest.main()
