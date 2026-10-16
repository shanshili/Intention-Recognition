"""Behavioral contracts for section 6's M7-parent structural ablations."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_construction_models import batch, prepared


VARIANTS = (
    "m7", "m5_attention", "m7_no_gat", "m7_no_tcn",
    "m7_no_extra_node_features", "m7_no_gat_edge", "m7_no_score_edge",
    "m7_no_intent", "m7_fixed_intent", "m7_static_intent",
)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required for strict ablations")
class StrictAblationTests(unittest.TestCase):
    def setUp(self):
        import torch
        from experiment.models.deig_tcn import DeigTCNForecaster
        torch.set_num_threads(1)
        self.torch = torch
        self.model_class = DeigTCNForecaster

    def model(self, name, data=None, seed=17):
        return self.model_class(name, prepared(7) if data is None else data, seed=seed)

    def test_all_ablations_preserve_prediction_contract_and_honest_score_capabilities(self):
        for name in VARIANTS:
            with self.subTest(model=name):
                model = self.model(name)
                example = batch(7)
                output = model.predict(example)
                np.testing.assert_array_equal(
                    output["prediction"], np.repeat(example["last"][..., None], 3, axis=-1))
                self.assertEqual(model.parameter_count,
                                 sum(parameter.numel() for parameter in model.module.parameters()))
                if name == "m7_no_intent":
                    self.assertIsNone(output["edge_scores"])
                    self.assertIsNone(output["effective_weight"])
                    self.assertEqual(model.capabilities["score_kind"], "none")
                    self.assertFalse(model.capabilities["supports_ranking"])
                    self.assertFalse(model.capabilities["supports_intervention"])
                else:
                    scores = output["edge_scores"]
                    self.assertEqual(scores.shape, (2, 3))
                    self.assertTrue(np.isfinite(scores).all())
                    self.assertTrue(np.all((scores >= 0) & (scores <= 1)))
                    np.testing.assert_allclose(output["effective_weight"][:, [0, 2]],
                                               scores[:, [0, 2]] / 2)
                    kind = {"m7_fixed_intent": "fixed", "m7_static_intent": "static"}.get(
                        name, "dynamic")
                    self.assertEqual(model.capabilities["score_kind"], kind)
                    self.assertEqual(model.capabilities["supports_ranking"], kind != "fixed")
                    self.assertTrue(model.capabilities["supports_intervention"])
                    self.assertEqual(model.capabilities["intervention_scope"], "post_message_only")

    def test_attention_is_kept_and_normalized_for_every_new_variant(self):
        for name in VARIANTS:
            with self.subTest(model=name):
                attention = self.model(name).forward(batch(7))["temporal_attention"]
                self.assertEqual(tuple(attention.shape), (2, 12, 3))
                self.assertTrue(self.torch.all(attention > 0))
                self.torch.testing.assert_close(attention.sum(dim=1), self.torch.ones(2, 3))

    def test_m5_attention_never_constructs_explicit_dynamic_features(self):
        model = self.model("m5_attention")
        with patch("experiment.models.deig_tcn.dynamic_edge_features",
                   side_effect=AssertionError("Removed dynamic features must not be computed")):
            output = model.forward(batch(7))
            scores = model.score_history(batch(7)["history"])
        self.torch.testing.assert_close(output["edge_scores"], scores)
        # Removing explicit d does not turn a window-conditioned scorer static.
        self.assertFalse(self.torch.allclose(scores[0], scores[1]))

    def test_no_extra_features_changes_only_encoder_input_not_dynamic_feature_history(self):
        from experiment.models.deig_tcn import dynamic_edge_features
        model = self.model("m7_no_extra_node_features")
        example = batch(7)
        altered = {key: value.copy() for key, value in example.items()}
        altered["history"][..., 1:] += 10
        original_representation = model.representation_method(example["history"])
        altered_representation = model.representation_method(altered["history"])
        self.torch.testing.assert_close(original_representation, altered_representation)
        with patch("experiment.models.deig_tcn.dynamic_edge_features",
                   wraps=dynamic_edge_features) as feature_function:
            model.forward(altered)
        feature_function.assert_called_once()
        supplied_history = feature_function.call_args.args[0]
        self.assertEqual(tuple(supplied_history.shape), (2, 12, 3, 7))
        np.testing.assert_array_equal(supplied_history.detach().cpu().numpy(), altered["history"])
        # The single-channel condition is internal: deleting raw channels would
        # also delete the full model's explicit dynamic-edge feature pathway.
        with self.assertRaises(ValueError):
            self.model_class("m7_no_extra_node_features", prepared(2))

    def test_no_tcn_maps_each_time_position_independently_before_attention(self):
        model = self.model("m7_no_tcn")
        history = self.torch.as_tensor(batch(7)["history"])
        baseline = model.module.temporal_sequence(history)
        changed = history.clone()
        changed[:, 0] += 20
        modified = model.module.temporal_sequence(changed)
        self.assertEqual(tuple(baseline.shape[:3]), (2, 12, 3))
        self.torch.testing.assert_close(modified[:, 1:], baseline[:, 1:])
        self.assertFalse(self.torch.allclose(modified[:, 0], baseline[:, 0]))
        self.assertFalse(any(isinstance(layer, self.torch.nn.Conv1d)
                             for layer in model.module.modules()))
        self.assertIsNotNone(model.forward(batch(7))["temporal_attention"])

    def test_no_gat_representation_is_node_local_but_post_message_path_remains(self):
        model = self.model("m7_no_gat")
        example = batch(7)
        altered = {key: value.copy() for key, value in example.items()}
        altered["history"][:, :, 0] += 20
        original = model.forward(example)
        changed = model.forward(altered)
        self.torch.testing.assert_close(original["representation"][:, 1:],
                                        changed["representation"][:, 1:])
        self.assertFalse(self.torch.allclose(original["representation"][:, 0],
                                            changed["representation"][:, 0]))
        self.assertGreater(float(original["aggregation"].abs().sum()), 0)
        masked = model.forward(example, np.zeros((2, 3), dtype=np.float32))
        self.torch.testing.assert_close(masked["representation"], original["representation"])
        self.torch.testing.assert_close(masked["edge_scores"], original["edge_scores"])
        self.torch.testing.assert_close(masked["aggregation"],
                                        self.torch.zeros_like(masked["aggregation"]))
        self.assertFalse(model.capabilities["has_pre_gat"])

    def test_static_edge_removal_is_specific_to_the_selected_path(self):
        history = self.torch.as_tensor(batch(7)["history"])
        for name in ("m7_no_gat_edge", "m7_no_score_edge"):
            with self.subTest(model=name):
                left_data, right_data = prepared(7), prepared(7)
                right_data.graph["edge_feats"][0] += 20
                left, right = self.model(name, left_data), self.model(name, right_data)
                left_rep = left.representation_method(history)
                right_rep = right.representation_method(history)
                # Hold Z and d fixed when isolating the e -> Edge MLP route.
                left_score = left.module.raw_scores(left_rep, history)
                right_score = right.module.raw_scores(left_rep, history)
                if name == "m7_no_gat_edge":
                    self.torch.testing.assert_close(left_rep, right_rep)
                    self.assertFalse(self.torch.allclose(left_score, right_score))
                else:
                    self.assertFalse(self.torch.allclose(left_rep, right_rep))
                    self.torch.testing.assert_close(left_score, right_score)

    def test_no_intent_does_not_export_untrained_scores_or_accept_interventions(self):
        model = self.model("m7_no_intent")
        with patch("experiment.models.deig_tcn.dynamic_edge_features",
                   side_effect=AssertionError("No scorer should compute dynamic edge features")):
            output = model.forward(batch(7))
            self.assertIsNone(model.score_history(batch(7)["history"]))
        self.assertIsNone(output["edge_scores"])
        self.torch.testing.assert_close(output["aggregation"],
                                        self.torch.zeros_like(output["aggregation"]))
        self.assertTrue(model.capabilities["has_pre_gat"])
        with self.assertRaises(ValueError):
            model.forward(batch(7), np.zeros((2, 3), dtype=np.float32))

    def test_fixed_and_static_scores_do_not_depend_on_windows(self):
        fixed = self.model("m7_fixed_intent")
        np.testing.assert_array_equal(fixed.predict(batch(7))["edge_scores"], np.ones((2, 3)))
        static = self.model("m7_static_intent")
        before = static.score_history(batch(7)["history"])
        self.torch.testing.assert_close(before[0], before[1])
        optimizer = self.torch.optim.SGD(static.module.parameters(), lr=0.1)
        optimizer.zero_grad()
        (before * self.torch.tensor([1.0, 2.0, 3.0])).sum().backward()
        optimizer.step()
        after = static.score_history(batch(7)["history"] * 10)
        self.torch.testing.assert_close(after[0], after[1])
        self.assertFalse(self.torch.allclose(before.detach(), after.detach()))
        self.assertGreater(float(after[0].std()), 0)
        with patch("experiment.models.deig_tcn.dynamic_edge_features",
                   side_effect=AssertionError("Non-dynamic intent does not use d")):
            fixed.forward(batch(7))
            static.forward(batch(7))

    def test_prediction_training_reaches_every_retained_parameter(self):
        for name in VARIANTS:
            with self.subTest(model=name):
                model = self.model(name, seed=8)
                with self.torch.no_grad():
                    # The deliberate persistence initialization initially blocks
                    # upstream gradients until the residual head learns.
                    model.module.head[-1].weight.fill_(0.1)
                model.forward(batch(7))["prediction"].square().mean().backward()
                for parameter_name, parameter in model.module.named_parameters():
                    self.assertIsNotNone(parameter.grad, f"{name}: unused {parameter_name}")
                    self.assertTrue(self.torch.isfinite(parameter.grad).all(),
                                    f"{name}: nonfinite {parameter_name}")
                self.assertGreater(sum(float(parameter.grad.abs().sum())
                                       for parameter in model.module.parameters()), 0)

    def test_aliases_reuse_the_exact_model_instead_of_creating_duplicates(self):
        from experiment.models import make_model
        aliases = {"m7_no_dynamic_edge": "m5_attention", "m7_no_temporal_attention": "m6",
                   "m7_l0": "m7", "m7_dynamic_intent": "m7"}
        for alias, canonical in aliases.items():
            with self.subTest(alias=alias):
                left = make_model(alias, prepared(7), seed=14)
                right = make_model(canonical, prepared(7), seed=14)
                self.assertEqual(left.name, canonical)
                self.assertEqual(left.parameter_count, right.parameter_count)
                for key in ("prediction", "edge_scores", "effective_weight"):
                    np.testing.assert_array_equal(left.predict(batch(7))[key],
                                                  right.predict(batch(7))[key])

    def test_every_variant_supports_empty_graphs_and_checkpoint_roundtrips(self):
        from experiment.models import make_model
        for name in VARIANTS:
            with self.subTest(model=name):
                empty = self.model(name, prepared(7, empty=True)).predict(batch(7))
                self.assertTrue(np.isfinite(empty["prediction"]).all())
                if name != "m7_no_intent":
                    self.assertEqual(empty["edge_scores"].shape, (2, 0))
                model = self.model(name)
                model.configure_device("cpu")
                with tempfile.TemporaryDirectory() as directory:
                    checkpoint = Path(directory) / "model.pt"
                    model.save(checkpoint)
                    saved = self.torch.load(checkpoint, map_location="cpu", weights_only=False)
                    loaded = make_model(saved["model"], prepared(7), saved["config"], saved["seed"])
                    loaded.module.load_state_dict(saved["state_dict"])
                    original, restored = model.predict(batch(7)), loaded.predict(batch(7))
                    for key in ("prediction", "edge_scores", "effective_weight"):
                        if original[key] is None:
                            self.assertIsNone(restored[key])
                        else:
                            np.testing.assert_array_equal(original[key], restored[key])


if __name__ == "__main__":
    unittest.main()
