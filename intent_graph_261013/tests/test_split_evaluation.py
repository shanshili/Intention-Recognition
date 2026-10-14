"""Independent split-policy checks; no training or accelerator dependency."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiment.evaluation import evaluate_model, validate_evaluation_config


class SplitDataset:
    def __init__(self, split, ids, edges=4, target=1.0):
        self.split = split
        self.ids = np.asarray(ids, dtype=np.int64)
        self.target = float(target)
        self.data = SimpleNamespace(mean=np.zeros(1), std=np.ones(1),
                                    protocol={"H": 2, "train_stride": 3, "eval_stride": 1},
                                    graph={"E": edges})

    def __len__(self):
        return len(self.ids)

    def iter_batches(self, batch_size):
        for start in range(0, len(self.ids), batch_size):
            ids = self.ids[start:start + batch_size]
            target = np.full((len(ids), 1, 2), self.target)
            yield {"window_id": ids, "target_times": (ids[:, None] + [1, 2]).astype("datetime64[m]"),
                   "target": target, "raw_target": target.copy(),
                   "target_mask": np.ones(target.shape, dtype=bool), "last": np.zeros((len(ids), 1))}


class LoggedGateModel:
    name = "logged_gate"

    def __init__(self, edges=4, ranked=True, intervenable=True, has_scores=True):
        self.edges = edges
        self.has_scores = has_scores
        self.calls = []
        self.levels = {"train": 0.8, "val": 0.2, "test": 0.5}
        self.capabilities = {"supports_ranking": ranked, "supports_intervention": intervenable,
                             "score_kind": "gate" if has_scores else "none",
                             "intervention_scope": "single_message" if intervenable else None}

    def predict(self, batch, edge_mask=None):
        ids = batch["window_id"]
        split = "train" if ids[0] < 10 else "val" if ids[0] < 20 else "test"
        self.calls.append((split, edge_mask is not None, ids.tolist()))
        if not self.has_scores:
            return {"prediction": np.zeros_like(batch["target"]), "edge_scores": None,
                    "effective_weight": None}
        # Scores are deterministic by input ID, so invocation order cannot
        # create apparent temporal variation or interfere with calibration.
        scores = np.asarray([[self.levels[split] + 0.01 * (int(window) % 3)] * self.edges
                             for window in ids], dtype=float).reshape(len(ids), self.edges)
        weights = scores if edge_mask is None else scores * edge_mask
        signal = weights.sum(1)
        prediction = np.repeat(signal[:, None, None], 2, axis=2)
        return {"prediction": prediction, "edge_scores": scores, "effective_weight": weights}


class SplitEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.config = {"batch_size": 2, "top_fractions": [0.5], "random_seeds": [0, 1],
                       "threshold_quantile": 0.5}
        self.datasets = {"train": SplitDataset("train", [1, 4, 7]),
                         "val": SplitDataset("val", [11, 12, 13]),
                         "test": SplitDataset("test", [21, 22, 23])}

    def test_default_train_val_retain_gates_without_interventions_test_computes_them(self):
        for split in ("train", "val", "test"):
            model = LoggedGateModel()
            result = evaluate_model(model, self.datasets[split], self.config)
            metrics, graph = result["metrics"], result["graph_metrics"]
            self.assertEqual(metrics["split"], split)
            self.assertEqual(graph["split"], split)
            self.assertEqual(result["arrays"]["edge_scores"].shape, (3, 4))
            self.assertEqual(result["arrays"]["effective_weight"].shape, (3, 4))
            self.assertIn(split, graph["diagnostics"])
            self.assertIn("budget0_top_indices", result["graph_arrays"])
            self.assertEqual(graph["interventions_computed"], split == "test")
            masked_calls = [entry for entry in model.calls if entry[1]]
            if split == "test":
                self.assertTrue(masked_calls)
                self.assertIn("delete_top", graph["budgets"][0])
                self.assertIn("all_messages_off_mae", graph)
                self.assertEqual(graph["intervention_status"], "computed")
            else:
                self.assertEqual(masked_calls, [])
                self.assertNotIn("delete_top", graph["budgets"][0])
                self.assertNotIn("all_messages_off_mae", graph)
                self.assertEqual(graph["intervention_status"], "disabled_by_split_policy")

    def test_validation_threshold_is_frozen_and_not_recalibrated_per_split(self):
        model = LoggedGateModel()
        val = evaluate_model(model, self.datasets["val"], self.config, validation=self.datasets["val"])
        threshold = val["graph_metrics"]["threshold"]
        original = dict(threshold)
        expected = np.median([0.22] * 4 + [0.20] * 4 + [0.21] * 4)
        self.assertAlmostEqual(threshold["value"], expected)
        self.assertEqual(threshold["calibration_split"], "val")
        # Change validation outputs after calibration. Reusing the frozen
        # specification must preserve its threshold and avoid validation calls.
        model.levels["val"] = 0.9
        val_call_count = sum(split == "val" for split, masked, ids in model.calls)
        for split in ("train", "test"):
            result = evaluate_model(model, self.datasets[split], self.config, threshold_spec=threshold)
            self.assertAlmostEqual(result["graph_metrics"]["threshold"]["value"], expected)
            self.assertEqual(result["graph_metrics"]["threshold"]["calibration_split"], "val")
            np.testing.assert_array_equal(result["graph_arrays"]["absolute_edge_count"], [4, 4, 4])
        self.assertEqual(sum(split == "val" for split, masked, ids in model.calls), val_call_count)
        self.assertEqual(threshold, original)

    def test_training_stride_defines_adjacency_and_breaks_are_excluded(self):
        dataset = SplitDataset("train", [1, 4, 7, 13])
        result = evaluate_model(LoggedGateModel(), dataset, self.config)
        self.assertEqual(result["graph_metrics"]["temporal"]["adjacent_pairs"], 2)
        np.testing.assert_array_equal(result["graph_arrays"]["adjacent_window_id"], [4, 7])
        np.testing.assert_allclose(result["graph_arrays"]["adjacent_score_change"], [0, 0])
        self.assertEqual(len(result["graph_arrays"]["budget0_jaccard"]), 2)
        self.assertAlmostEqual(result["graph_metrics"]["budgets"][0]["adjacent_top_jaccard"], 1)

    def test_explicit_all_splits_enables_interventions_and_global_false_disables_them(self):
        explicit = {**self.config, "graph_intervention_splits": ["train", "val", "test"]}
        for split, dataset in self.datasets.items():
            active_model = LoggedGateModel()
            active = evaluate_model(active_model, dataset, explicit)
            self.assertTrue(active["graph_metrics"]["interventions_computed"])
            self.assertIn("delete_top", active["graph_metrics"]["budgets"][0])
            self.assertTrue(any(masked for name, masked, ids in active_model.calls))
            disabled_model = LoggedGateModel()
            disabled = evaluate_model(disabled_model, dataset, {**explicit, "graph_interventions": False})
            self.assertFalse(disabled["graph_metrics"]["interventions_computed"])
            self.assertFalse(any(masked for name, masked, ids in disabled_model.calls))
            self.assertIn("budget0_top_indices", disabled["graph_arrays"])

    def test_no_graph_and_zero_persistence_are_na_and_timing_separates_prediction(self):
        model = LoggedGateModel(ranked=False, intervenable=False, has_scores=False)
        dataset = SplitDataset("test", [21, 22, 23], target=0)
        with patch("experiment.evaluation.time.perf_counter", side_effect=[100.0, 101.0, 104.0, 112.0]):
            result = evaluate_model(model, dataset, self.config)
        metrics, graph = result["metrics"], result["graph_metrics"]
        self.assertEqual(metrics["prediction_seconds"], 3.0)
        self.assertEqual(metrics["evaluation_seconds"], 12.0)
        self.assertEqual(metrics["prediction_windows_per_second"], 1.0)
        self.assertIsNone(metrics["skill"])
        self.assertIsNone(metrics["skill_standardized"])
        self.assertFalse(graph["available"])
        self.assertFalse(graph["interventions_computed"])
        self.assertEqual(graph["intervention_status"], "not_applicable")
        self.assertEqual(graph["budgets"], [])
        self.assertIsNone(result["arrays"]["edge_scores"])

    def test_fixed_graph_reports_only_all_messages_off_on_test(self):
        for split in ("train", "test"):
            result = evaluate_model(LoggedGateModel(ranked=False), self.datasets[split], self.config)
            graph = result["graph_metrics"]
            self.assertFalse(graph["available"])
            self.assertEqual(graph["budgets"], [])
            self.assertEqual(graph["interventions_computed"], split == "test")
            self.assertEqual("all_messages_off_mae" in graph, split == "test")

    def test_empty_candidate_graph_does_not_claim_interventions_were_computed(self):
        result = evaluate_model(LoggedGateModel(edges=0), SplitDataset("test", [21, 22, 23], edges=0), self.config)
        graph = result["graph_metrics"]
        self.assertFalse(graph["available"])
        self.assertFalse(graph["interventions_computed"])
        self.assertEqual(graph["intervention_status"], "not_applicable")
        self.assertNotIn("all_messages_off_mae", graph)

    def test_split_policy_config_rejects_duplicates_and_unknown_names(self):
        self.assertEqual(validate_evaluation_config({})["graph_intervention_splits"], ["test"])
        for splits in (["test", "test"], ["unknown"], "test"):
            with self.assertRaises(ValueError):
                validate_evaluation_config({"graph_intervention_splits": splits})


if __name__ == "__main__":
    unittest.main()
