"""Evaluator checks use explicit toy predictions and independently computed errors."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch
import tempfile

import numpy as np

from experiment.evaluation import evaluate_model


class ToyDataset:
    def __init__(self, ids, target, mask=None, mean=None, std=None, last=None, edges=4):
        self.ids = np.asarray(ids, dtype=np.int64)
        self.target = np.asarray(target, dtype=float)
        self.mask = np.ones_like(self.target, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        M, N, H = self.target.shape
        self.last = np.zeros((M, N)) if last is None else np.asarray(last, dtype=float)
        self.data = SimpleNamespace(mean=np.zeros(N) if mean is None else np.asarray(mean, dtype=float),
                                    std=np.ones(N) if std is None else np.asarray(std, dtype=float),
                                    protocol={"H": H, "eval_stride": 1}, graph={"E": edges})
        self.raw = self.target * self.data.std[None, :, None] + self.data.mean[None, :, None]
        self.raw = np.where(self.mask, self.raw, np.nan)

    def __len__(self):
        return len(self.ids)

    def iter_batches(self, batch_size):
        for start in range(0, len(self), batch_size):
            end = start + batch_size
            ids = self.ids[start:end]
            times = (ids[:, None] + np.arange(1, self.target.shape[-1] + 1)).astype("datetime64[m]")
            yield {"target": self.target[start:end], "raw_target": self.raw[start:end],
                   "target_mask": self.mask[start:end], "last": self.last[start:end],
                   "window_id": ids, "target_times": times}


class OffsetModel:
    name = "offset"
    capabilities = {"supports_ranking": False, "supports_intervention": False,
                    "score_kind": "none", "intervention_scope": None}

    def __init__(self, errors):
        self.errors = errors
        self.batch_sizes = []

    def predict(self, batch, edge_mask=None):
        self.batch_sizes.append(len(batch["window_id"]))
        error = np.stack([self.errors[int(window)] for window in batch["window_id"]])
        return {"prediction": batch["target"] + error, "edge_scores": None, "effective_weight": None}


class ToyGateModel:
    name = "toy_gate"
    capabilities = {"supports_ranking": True, "supports_intervention": True,
                    "score_kind": "sigmoid_gate", "intervention_scope": "fixed_original_messages"}

    def __init__(self, scores):
        self.scores = {key: np.asarray(value, dtype=float) for key, value in scores.items()}
        self.coefficient = np.asarray([1.0, 2.0, 4.0, 8.0])

    def predict(self, batch, edge_mask=None):
        scores = np.stack([self.scores[int(window)] for window in batch["window_id"]])
        masks = np.ones_like(scores) if edge_mask is None else np.asarray(edge_mask)
        signal = (scores * masks * self.coefficient).sum(axis=1)
        prediction = np.broadcast_to(signal[:, None, None], batch["target"].shape).copy()
        return {"prediction": prediction, "edge_scores": scores,
                "effective_weight": scores * masks}


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.graph_config = {"batch_size": 2, "top_fractions": [0.5],
                             "random_seeds": [0, 3, 7], "threshold_quantile": 0.75}
        self.graph_model = ToyGateModel({
            1: [0, 0, 0, 0], 2: [0.1, 0.2, 0.3, 0.4],
            11: [0, 0, 0, 0], 12: [0.8, 0.8, 0.1, 0.1], 13: [0.3, 0.3, 0.2, 0.2],
        })
        self.test = ToyDataset([11, 12, 13], np.ones((3, 1, 2)))
        self.validation = ToyDataset([1, 2], np.ones((2, 1, 2)))

    def test_weighted_mask_tail_horizons_and_full_persistence(self):
        target = np.asarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=float)
        mask = np.asarray([[[1, 1], [1, 1]], [[1, 0], [0, 0]], [[1, 1], [1, 0]]], dtype=bool)
        errors = {11: np.ones((2, 2)), 12: np.full((2, 2), 4), 13: np.full((2, 2), 2)}
        dataset = ToyDataset([11, 12, 13], target, mask=mask, mean=[10, 20], std=[2, 5])
        model = OffsetModel(errors)
        result = evaluate_model(model, dataset, {"batch_size": 2})
        raw_errors = np.stack([errors[window] for window in [11, 12, 13]]) * np.asarray([2, 5])[None, :, None]
        valid_errors = raw_errors[mask]
        self.assertEqual(model.batch_sizes, [2, 1])
        self.assertEqual(result["metrics"]["windows"], 3)
        self.assertEqual(result["metrics"]["valid_targets"], 8)
        self.assertAlmostEqual(result["metrics"]["mae"], valid_errors.mean())
        self.assertAlmostEqual(result["metrics"]["rmse"], np.sqrt(np.square(valid_errors).mean()))
        self.assertAlmostEqual(result["metrics"]["mae_standardized"],
                               np.stack([errors[window] for window in [11, 12, 13]])[mask].mean())
        expected_horizon = [raw_errors[:, :, h][mask[:, :, h]].mean() for h in range(2)]
        np.testing.assert_allclose(result["metrics"]["per_horizon_mae"], expected_horizon)
        persistence_error = np.abs(target * np.asarray([2, 5])[None, :, None])
        expected_persistence = persistence_error[mask].mean()
        self.assertAlmostEqual(result["metrics"]["persistence_mae"], expected_persistence)
        self.assertAlmostEqual(result["metrics"]["skill"], 1 - valid_errors.mean() / expected_persistence)
        self.assertNotAlmostEqual(expected_persistence, persistence_error[-1][mask[-1]].mean())
        np.testing.assert_array_equal(result["arrays"]["window_id"], [11, 12, 13])

    def test_exact_top_with_zero_ties_and_random_seed_sd(self):
        result = evaluate_model(self.graph_model, self.test, self.graph_config, validation=self.validation)
        graph_arrays = result["graph_arrays"]
        top = graph_arrays["budget0_top_indices"]
        self.assertEqual(top.shape, (3, 2))
        # All-zero and tied positive rows still select exactly K in candidate order.
        np.testing.assert_array_equal(top, [[0, 1], [0, 1], [0, 1]])
        budget = result["graph_metrics"]["budgets"][0]
        self.assertEqual(budget["k"], 2)
        self.assertEqual(budget["random_seeds"], [0, 3, 7])
        seed_errors = graph_arrays["budget0_delete_random_seed_mae"]
        self.assertGreater(float(np.ptp(seed_errors)), 0)
        self.assertAlmostEqual(budget["delete_random_mean"], seed_errors.mean())
        self.assertAlmostEqual(budget["delete_random_std"], seed_errors.std(ddof=1))
        # Each seed's statistic is total test error, not SD across windows.
        for i, seed in enumerate(budget["random_seeds"]):
            delta = graph_arrays[f"budget0_delete_random_seed{seed}_window_delta"]
            expected = (delta + result["arrays"]["per_window_mae"]).mean()
            self.assertAlmostEqual(seed_errors[i], expected)

    def test_threshold_uses_validation_positive_scores_only(self):
        config = {**self.graph_config, "graph_interventions": False}
        result = evaluate_model(self.graph_model, self.test, config, validation=self.validation)
        threshold = result["graph_metrics"]["threshold"]
        self.assertTrue(threshold["enabled"])
        self.assertAlmostEqual(threshold["value"], np.quantile([0.1, 0.2, 0.3, 0.4], 0.75))
        self.assertEqual(threshold["rule"], "validation_positive_quantile")
        self.assertNotIn("delete_top", result["graph_metrics"]["budgets"][0])
        np.testing.assert_array_equal(result["graph_arrays"]["absolute_edge_count"], [0, 2, 0])

    def test_no_positive_validation_disables_threshold_and_zero_guard(self):
        zeros = ToyDataset([1], np.ones((1, 1, 2)))
        config = {**self.graph_config, "graph_interventions": False}
        result = evaluate_model(self.graph_model, self.test, config, validation=zeros)
        self.assertFalse(result["graph_metrics"]["threshold"]["enabled"])
        self.assertIsNone(result["graph_metrics"]["threshold"]["value"])
        explicit = evaluate_model(self.graph_model, self.test,
                                  {**config, "absolute_threshold": 0}, validation=zeros)
        np.testing.assert_array_equal(explicit["graph_arrays"]["absolute_edge_count"], [0, 4, 4])

    def test_disabled_intervention_plots_respect_configured_window_count(self):
        import matplotlib.pyplot as plt
        from experiment.visualization import make_run_plots

        result = evaluate_model(self.graph_model, self.test,
                                {**self.graph_config, "graph_interventions": False, "max_plot_windows": 1},
                                validation=self.validation)
        data = SimpleNamespace(coords=np.asarray([[0, 0], [1, 1]], dtype=float),
                               graph={"edges": np.asarray([[0, 1], [1, 0], [0, 1], [1, 0]])},
                               protocol={"step_minutes": 5})
        rendered = []

        def capture(fig, output, stem, timestamp):
            rendered.append((stem, len(fig.axes)))
            plt.close(fig)
            return []

        with tempfile.TemporaryDirectory() as destination:
            with patch("experiment.visualization.save_all_formats", side_effect=capture):
                make_run_plots(destination, {"history": []}, result, data)
        self.assertNotIn("edge_deletion_delta", [stem for stem, count in rendered])
        gates = [(stem, count) for stem, count in rendered if stem.startswith("gate_")]
        self.assertEqual(len(gates), 2)
        # One graph panel plus its shared colorbar; no hidden extra windows.
        self.assertTrue(all(count == 2 for stem, count in gates))


if __name__ == "__main__":
    unittest.main()
