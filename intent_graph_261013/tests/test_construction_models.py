"""Behavioral contracts for the incremental DEIG versions, not score claims."""

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def prepared(version, window=12, empty=False):
    channels = ["level"] if version < 3 else ["level", "delta", "periodic_residual",
                                             "time_sin", "time_cos", "weekday_sin", "weekday_cos"]
    src, dst = ([], []) if empty else ([0, 1, 2], [1, 0, 1])
    return SimpleNamespace(protocol={"W": window, "H": 3, "feature_channels": channels},
        features=np.zeros((30, 3, len(channels)), dtype=np.float32),
        graph={"src": np.asarray(src), "dst": np.asarray(dst),
               "edge_feats": np.arange(len(src) * 8, dtype=np.float32).reshape(-1, 8) / 20})


def batch(version, window=12, count=2):
    rng = np.random.default_rng(42)
    history = rng.normal(size=(count, window, 3, 1 if version < 3 else 7)).astype(np.float32)
    return {"history": history, "last": history[:, -1, :, 0].copy()}


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required for construction models")
class ConstructionModelTests(unittest.TestCase):
    def setUp(self):
        import torch
        from experiment.models.deig_tcn import DeigTCNForecaster
        torch.set_num_threads(1)
        self.torch, self.model_class = torch, DeigTCNForecaster

    def test_all_versions_start_at_persistence_and_export_separate_scores(self):
        for version in range(1, 8):
            with self.subTest(version=version):
                model = self.model_class(f"m{version}", prepared(version), seed=17)
                example = batch(version)
                output = model.predict(example)
                np.testing.assert_array_equal(output["prediction"],
                                               np.repeat(example["last"][..., None], 3, axis=-1))
                self.assertEqual(output["edge_scores"].shape, (2, 3))
                self.assertTrue(np.isfinite(output["edge_scores"]).all())
                np.testing.assert_allclose(output["effective_weight"][:, [0, 2]],
                                           output["edge_scores"][:, [0, 2]] / 2)
                self.assertEqual(model.capabilities["intervention_scope"], "post_message_only")
                self.assertEqual(model.parameter_count,
                                 sum(p.numel() for p in model.module.parameters()))

    def test_components_follow_the_exact_incremental_matrix(self):
        for version in range(1, 8):
            model = self.model_class(f"m{version}", prepared(version))
            metadata = model.construction_metadata
            self.assertEqual(metadata["input_features"], 1 if version < 3 else 7)
            self.assertEqual(metadata["static_edge_in_score"], version >= 4)
            self.assertEqual(metadata["static_edge_in_gat"], version >= 5)
            self.assertEqual(metadata["dynamic_edge_in_score"], version >= 6)
            self.assertEqual(model.module.gat.edge_projection is not None, version >= 5)
            if version >= 2:
                self.assertEqual(metadata["tcn_receptive_field"], 15)
                self.assertEqual(model.module.encoder.dilations, (1, 2, 4))
            if version == 7:
                self.assertEqual(metadata["temporal_pooling"], "attention")

    def test_tcn_full_sequence_is_causal_and_covers_oldest_window_position(self):
        from experiment.models.deig_tcn import FullSequenceCausalTCN
        tcn = FullSequenceCausalTCN(1, 2)
        with self.torch.no_grad():
            for layer in tcn.layers:
                layer.weight.fill_(0.25)
                layer.bias.fill_(0.1)
            history = self.torch.ones(2, 12, 3, 1)
            baseline = tcn(history)
            modified = history.clone()
            modified[:, 8:] += 20
            self.torch.testing.assert_close(tcn(modified)[:, :8], baseline[:, :8])
            oldest = history.clone()
            oldest[:, 0] += 1
            self.assertTrue(self.torch.all(tcn(oldest)[:, -1] > baseline[:, -1]))
            self.assertEqual(tuple(baseline.shape), (2, 12, 3, 2))

    def test_gat_direction_and_self_path_are_not_intention_edges(self):
        from experiment.models.deig_tcn import DirectedGAT
        gat = DirectedGAT(1, 1, 3, [0], [1])
        with self.torch.no_grad():
            gat.projection.weight.fill_(1)
            gat.attention_src.zero_()
            gat.attention_dst.zero_()
            history = self.torch.tensor([[[[2.0], [6.0], [10.0]]]])
            output, attention = gat(history, return_attention=True)
            self.torch.testing.assert_close(output.squeeze(), self.torch.tensor([2.0, 4.0, 10.0]))
            self.assertEqual(tuple(attention.shape), (1, 1, 4))
            changed = history.clone()
            changed[:, :, 2] += 100
            self.torch.testing.assert_close(gat(changed)[:, :, :2], output[:, :, :2])
            changed = history.clone()
            changed[:, :, 1] += 100
            self.torch.testing.assert_close(gat(changed)[:, :, 0], output[:, :, 0])

    def test_static_prior_enters_gat_only_from_m5(self):
        for version in (4, 5):
            left_data, right_data = prepared(version), prepared(version)
            right_data.graph["edge_feats"][0] += 20
            left = self.model_class(f"m{version}", left_data, seed=31)
            right = self.model_class(f"m{version}", right_data, seed=31)
            history = self.torch.as_tensor(batch(version)["history"])
            left_spatial, right_spatial = left.module.gat(history), right.module.gat(history)
            if version == 4:
                self.torch.testing.assert_close(left_spatial, right_spatial)
            else:
                self.assertFalse(self.torch.allclose(left_spatial, right_spatial))
            self.assertFalse(self.torch.allclose(left.module.score_history(history),
                                                right.module.score_history(history)))

    def test_dynamic_features_are_directed_and_zero_variance_is_finite(self):
        from experiment.models.deig_tcn import dynamic_edge_features
        torch = self.torch
        history = torch.zeros(1, 12, 3, 7)
        source = torch.tensor([0.2, -1.1, 0.6, 1.8, -0.4, 1.0, -1.8, 0.4, 0.9, -0.7, 1.4, 0.3])
        history[0, :, 0, 2] = source
        history[0, 1:, 1, 2] = source[:-1]
        history[0, :, 0, 1] = 2
        history[0, :, 1, 1] = -3
        history[0, 0, :2, 1] = 99  # The outside-window transition is excluded.
        src, dst = torch.tensor([0, 1, 2]), torch.tensor([1, 0, 1])
        features = dynamic_edge_features(history, src, dst, max_lag=4)
        self.assertEqual(tuple(features.shape), (1, 3, 6))
        self.assertTrue(torch.isfinite(features).all())
        self.assertAlmostEqual(float(features[0, 0, 0]), 1, places=5)
        self.assertGreater(float(features[0, 0, 1]), 0)
        torch.testing.assert_close(features[0, 0, 4:], torch.tensor([2.0, -3.0]))
        self.assertEqual(float(features[0, 2, 0]), 0)
        short = dynamic_edge_features(history[:, :2], src, dst)
        self.assertTrue(torch.isfinite(short).all())
        torch.testing.assert_close(short[..., :2], torch.zeros_like(short[..., :2]))
        opposite = torch.zeros_like(history)
        opposite[0, :, 0, 2] = torch.arange(12)
        opposite[0, :, 1, 2] = -torch.arange(12)
        negative = dynamic_edge_features(opposite, src, dst)
        torch.testing.assert_close(negative[0, :2, 0], torch.tensor([-1.0, -1.0]))

    def test_post_message_mask_keeps_gat_representation_and_raw_scores(self):
        model = self.model_class("m7", prepared(7), seed=12)
        example = batch(7)
        original = model.forward(example)
        masked = model.forward(example, np.zeros((2, 3), dtype=np.float32))
        self.torch.testing.assert_close(masked["representation"], original["representation"])
        self.torch.testing.assert_close(masked["edge_scores"], original["edge_scores"])
        self.torch.testing.assert_close(masked["aggregation"], self.torch.zeros_like(masked["aggregation"]))
        self.torch.testing.assert_close(masked["effective_weight"], self.torch.zeros_like(masked["effective_weight"]))
        self.torch.testing.assert_close(model.score_history(example["history"]), original["edge_scores"])

    def test_m7_attention_is_normalized_over_historical_positions(self):
        model = self.model_class("m7", prepared(7))
        output = model.forward(batch(7))
        attention = output["temporal_attention"]
        self.assertEqual(tuple(attention.shape), (2, 12, 3))
        self.assertTrue(self.torch.all(attention > 0))
        self.torch.testing.assert_close(attention.sum(dim=1), self.torch.ones(2, 3))

    def test_learned_prediction_loss_reaches_both_spatial_routes_and_time_pool(self):
        model = self.model_class("m7", prepared(7), seed=8)
        with self.torch.no_grad():
            model.module.head[-1].weight.fill_(0.1)
        output = model.forward(batch(7))
        output["prediction"].square().mean().backward()
        for component in (model.module.gat.projection, model.module.gat.edge_projection,
                          model.module.encoder.layers[0], model.module.gate[0],
                          model.module.message, model.module.temporal_attention[0]):
            self.assertIsNotNone(component.weight.grad)
            self.assertTrue(self.torch.isfinite(component.weight.grad).all())
            self.assertGreater(float(component.weight.grad.abs().sum()), 0)

    def test_zero_edge_graph_preserves_self_encoding_and_shape(self):
        model = self.model_class("m7", prepared(7, empty=True))
        output = model.predict(batch(7))
        self.assertEqual(output["edge_scores"].shape, (2, 0))
        self.assertTrue(np.isfinite(output["prediction"]).all())

    def test_checkpoint_graph_buffers_are_portable_and_device_moves_together(self):
        model = self.model_class("m7", prepared(7), seed=5)
        self.assertEqual(model.configure_device("cpu")["device"], "cpu")
        for _, value in model.module.named_buffers():
            self.assertEqual(value.device.type, "cpu")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            model.save(checkpoint)
            saved = self.torch.load(checkpoint, map_location="cpu", weights_only=False)
            other = self.model_class("m7", prepared(7), saved["config"], saved["seed"])
            other.module.load_state_dict(saved["state_dict"])
            np.testing.assert_array_equal(model.predict(batch(7))["edge_scores"],
                                           other.predict(batch(7))["edge_scores"])

    def test_misconfigured_channels_short_receptive_field_and_unknown_keys_raise(self):
        with self.assertRaises(ValueError):
            self.model_class("m3", prepared(2))
        data = prepared(7)
        data.protocol["feature_channels"][1:3] = ["periodic_residual", "delta"]
        with self.assertRaises(ValueError):
            self.model_class("m7", data)
        with self.assertRaises(ValueError):
            self.model_class("m2", prepared(2), {"tcn_dilations": [1, 2]})
        with self.assertRaises(ValueError):
            self.model_class("m7", prepared(7), {"typo_attention": 8})


if __name__ == "__main__":
    unittest.main()
