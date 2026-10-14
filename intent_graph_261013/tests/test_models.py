"""Behavioral checks for masks, fixed normalization, and checkpoint selection."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment.models import make_model
from experiment.trainer import validation_mae


class FakeDataset:
    def __init__(self, count=9, seed=123, window=3, nodes=3, horizon=2):
        rng = np.random.default_rng(seed)
        self.history = rng.normal(size=(count, window, nodes, 1)).astype(np.float32)
        self.last = self.history[:, -1, :, 0].copy()
        self.target = self.last[..., None] + np.full((count, nodes, horizon), 0.2, dtype=np.float32)
        self.mask = np.ones_like(self.target, dtype=bool)

    def iter_batches(self, batch_size, shuffle_seed=None, with_previous=False):
        indices = np.arange(len(self.history))
        if shuffle_seed is not None:
            np.random.default_rng(shuffle_seed).shuffle(indices)
        for start in range(0, len(indices), batch_size):
            chosen = indices[start:start + batch_size]
            result = {"history": self.history[chosen], "last": self.last[chosen],
                      "target": self.target[chosen], "target_mask": self.mask[chosen]}
            if with_previous:
                result["previous_history"] = self.history[np.maximum(chosen - 1, 0)]
                result["previous_mask"] = chosen > 0
            yield result


def prepared():
    return SimpleNamespace(protocol={"W": 3, "H": 2}, features=np.zeros((15, 3, 1)),
                           graph={"src": np.array([0, 1, 2]), "dst": np.array([2, 2, 1]),
                                  "degree": np.array([0, 1, 2])})


class BaselineTests(unittest.TestCase):
    def test_persistence_broadcasts_last_and_rejects_edges(self):
        model = make_model("persistence", prepared())
        batch = next(FakeDataset().iter_batches(4))
        np.testing.assert_allclose(model.predict(batch)["prediction"],
                                   np.repeat(batch["last"][..., None], 2, axis=-1))
        self.assertIsNone(model.predict(batch)["edge_scores"])
        with self.assertRaises(ValueError):
            model.predict(batch, np.ones((4, 3)))

    def test_validation_includes_tail_batch_and_missing_mask(self):
        dataset = FakeDataset(count=5)
        dataset.last[:] = 0
        dataset.target[:] = 1
        dataset.target[-1] = 10
        dataset.mask[0, 0, 0] = False
        model = make_model("persistence", prepared())
        expected = (4 * 3 * 2 - 1 + 10 * 3 * 2) / (5 * 3 * 2 - 1)
        self.assertAlmostEqual(validation_mae(model, dataset, 4), expected)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is optional for this test")
    def test_ridge_fits_direct_horizons_and_excludes_missing_targets(self):
        train, val = FakeDataset(count=80), FakeDataset(count=13, seed=456)
        for dataset in (train, val):
            residual = 0.35 * dataset.history[:, 0, :, 0] - 0.2 * dataset.history[:, 1, :, 0]
            dataset.target = dataset.last[..., None] + np.stack([residual + 0.1, 2 * residual - 0.2], axis=-1)
        train.mask[0, 0, 0] = False
        train.target[0, 0, 0] = 1e8
        model = make_model("ridge_ar", prepared(), {"alpha_candidates": [0.0001, 100]})
        metadata = model.fit(train, val, {"batch_size": 7})
        self.assertEqual(metadata["selected_alpha"], 0.0001)
        self.assertLess(validation_mae(model, val, 4), 1e-4)
        self.assertFalse(model.capabilities["supports_intervention"])

    def test_unknown_models_and_loss_keys_do_not_silently_pass(self):
        from experiment.losses import validate_loss_config
        with self.assertRaises(ValueError):
            make_model("not_a_model", prepared())
        with self.assertRaises(ValueError):
            validate_loss_config({"misspelled_sparse": 1}, make_model("persistence", prepared()))


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional for baselines")
class NeuralTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        self.torch = torch
        self.batch = next(FakeDataset().iter_batches(4))

    def test_zero_mask_removes_aggregation_and_keeps_raw_scores(self):
        model = make_model("dynamic_gate", prepared(), seed=11)
        original = model.predict(self.batch)
        zero_mask = np.zeros((4, 3))
        output = model.forward(self.batch, zero_mask)
        np.testing.assert_allclose(output["aggregation"].detach().numpy(), 0)
        np.testing.assert_allclose(model.predict(self.batch, zero_mask)["edge_scores"], original["edge_scores"])
        np.testing.assert_allclose(model.predict(self.batch, zero_mask)["effective_weight"], 0)
        # The prediction head can now access only the node's own representation.
        representation = model.module.encode(self.torch.as_tensor(self.batch["history"]))
        expected = model.module.head(self.torch.cat([representation,
                    self.torch.zeros((4, 3, model.module.msg_dim))], dim=-1))
        expected += self.torch.as_tensor(self.batch["last"])[..., None]
        np.testing.assert_allclose(output["prediction"].detach().numpy(), expected.detach().numpy())

    def test_deleting_edge_does_not_renormalize_other_incoming_edges(self):
        model = make_model("fixed_graph", prepared())
        mask = np.ones((4, 3))
        mask[:, 0] = 0
        unmasked, masked = model.predict(self.batch), model.predict(self.batch, mask)
        np.testing.assert_allclose(unmasked["effective_weight"][:, :2], 0.5)
        np.testing.assert_allclose(masked["effective_weight"][:, 1], 0.5)
        np.testing.assert_allclose(masked["effective_weight"][:, 0], 0)
        self.assertFalse(model.capabilities["supports_ranking"])

    def test_static_scores_invariant_and_dynamic_sigmoid_finite(self):
        static = make_model("static_gate", prepared())
        changed = {**self.batch, "history": self.batch["history"] * 1e5}
        np.testing.assert_array_equal(static.predict(self.batch)["edge_scores"],
                                      static.predict(changed)["edge_scores"])
        dynamic = make_model("dynamic_gate", prepared())
        scores = dynamic.predict(changed)["edge_scores"]
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())

    def test_device_selection_is_explicit_and_moves_graph_buffers(self):
        model = make_model("dynamic_gate", prepared())
        metadata = model.configure_device("cpu")
        self.assertEqual(metadata["device"], "cpu")
        self.assertIsNone(metadata["gpu_name"])
        self.assertEqual(model.module.src.device.type, "cpu")
        self.assertEqual(model.module.degree.device.type, "cpu")
        with mock.patch.object(self.torch.cuda, "is_available", return_value=False):
            self.assertEqual(model.configure_device("auto")["device"], "cpu")
            with self.assertRaises(RuntimeError):
                model.configure_device("cuda")
        with self.assertRaises(ValueError):
            model.configure_device("mps")
        if self.torch.cuda.is_available():
            model.configure_device("cuda")
            output = model.predict(self.batch)
            self.assertEqual(model.module.src.device.type, "cuda")
            self.assertTrue(np.isfinite(output["prediction"]).all())
            self.assertIsInstance(output["prediction"], np.ndarray)

    def test_all_variants_start_at_persistence_and_tcn_is_causal(self):
        for name in ("local", "fixed_graph", "static_gate", "dynamic_gate"):
            model = make_model(name, prepared(), {"encoder": "tcn"})
            expected = np.repeat(self.batch["last"][..., None], 2, axis=-1)
            np.testing.assert_array_equal(model.predict(self.batch)["prediction"], expected)
        from experiment.models.gated_forecaster import CausalTCN
        tcn = CausalTCN(1, 5)
        with self.torch.no_grad():
            sequence = self.torch.as_tensor(self.batch["history"])
            short = tcn(sequence[:, :2])
            # Last representation of a two-step prefix cannot depend on an
            # appended third time step: causal layers consume the prefix alone.
            layer_input = sequence.permute(0, 2, 3, 1).reshape(12, 1, 3)
            for layer, padding in zip(tcn.layers, tcn.left_padding):
                layer_input = self.torch.relu(layer(self.torch.nn.functional.pad(layer_input, (padding, 0))))
            np.testing.assert_allclose(short.numpy(), layer_input[:, :, 1].reshape(4, 3, 5).numpy(), atol=1e-6)

    def test_trainer_restores_deepcopied_best_checkpoint_and_counts_tail(self):
        model = make_model("local", prepared(), seed=12)
        train, val = FakeDataset(count=9), FakeDataset(count=5)
        snapshots = []
        scores = iter([0.5, 0.2, 0.4, 0.6])
        def fake_validation(*args):
            snapshots.append({key: value.clone() for key, value in model.module.state_dict().items()})
            return next(scores)
        with mock.patch("experiment.trainer.validation_mae", side_effect=fake_validation):
            result = model.fit(train, val, {"epochs": 10, "patience": 2, "batch_size": 4,
                                          "learning_rate": 0.01, "cpu_threads": 1, "device": "cpu"})
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(result["epochs"], 3)
        self.assertEqual(result["history"][1]["batches"], 3)
        self.assertEqual(result["history"][1]["train_observed_targets"], 9 * 3 * 2)
        for key, value in model.module.state_dict().items():
            self.torch.testing.assert_close(value, snapshots[1][key])
        self.assertTrue(any(not self.torch.equal(snapshots[1][key], snapshots[3][key])
                            for key in snapshots[1]))

    def test_temporal_regularizer_uses_paired_training_windows(self):
        from experiment.losses import compute_loss, validate_loss_config
        model = make_model("dynamic_gate", prepared())
        batch = next(FakeDataset().iter_batches(4, with_previous=True))
        config = validate_loss_config({"temporal": 0.3}, model)
        _, pieces, _ = compute_loss(model, model.forward(batch), batch, config)
        self.assertGreater(pieces["temporal"], 0)
        batch["previous_mask"][:] = False
        _, pieces, _ = compute_loss(model, model.forward(batch), batch, config)
        self.assertEqual(pieces["temporal"], 0)
        with self.assertRaises(ValueError):
            validate_loss_config({"temporal": 1}, make_model("static_gate", prepared()))


if __name__ == "__main__":
    unittest.main()
