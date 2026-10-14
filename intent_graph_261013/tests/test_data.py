"""Regression tests for leakage prevention, split ownership, and batch coverage."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from experiment.data import prepare_data
from experiment.dataset import WindowDataset
from experiment.graph import build_candidate_graph


class DataProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.flow = self.root / "flow.csv"
        self.coords = self.root / "coords.csv"
        self.protocol_path = self.root / "protocol.json"
        self.values = np.stack([np.arange(60, dtype=float), 2 * np.arange(60, dtype=float)], axis=1)
        pd.DataFrame(self.values, columns=["a", "b"]).to_csv(self.flow, index=False)
        pd.DataFrame({"node_id": ["a", "b"], "x": [0, 1], "y": [0, 1]}).to_csv(self.coords, index=False)
        self.config = {"flow_path": str(self.flow), "coord_path": str(self.coords),
                       "W": 4, "H": 3, "train_stride": 3, "eval_stride": 1,
                       "features": ["level", "delta", "time", "periodic_residual"],
                       "candidate_graph": {"k": 1, "tau_max": 2}}
        self._write_config()

    def tearDown(self):
        self.temp.cleanup()

    def _write_config(self):
        self.protocol_path.write_text(json.dumps(self.config), encoding="utf-8")

    def test_no_future_interpolation_or_target_clipping(self):
        self.values[37, 0] = np.nan
        self.values[38, 0] = np.nan
        self.values[39, 0] = 10000
        pd.DataFrame(self.values, columns=["a", "b"]).to_csv(self.flow, index=False)
        original = prepare_data(self.protocol_path)
        self.values[39:, 0] = 20000
        pd.DataFrame(self.values, columns=["a", "b"]).to_csv(self.flow, index=False)
        changed = prepare_data(self.protocol_path)
        np.testing.assert_allclose(original.mean, changed.mean)
        np.testing.assert_allclose(original.std, changed.std)
        np.testing.assert_allclose(original.features[:39], changed.features[:39])
        np.testing.assert_allclose(original.graph["edge_feats"], changed.graph["edge_feats"])
        self.assertEqual(float(original.features[37, 0, 0]), float(original.features[36, 0, 0]))
        self.assertFalse(original.observed[37, 0])
        self.assertEqual(float(original.raw[39, 0]), 10000)
        self.assertAlmostEqual(float(original.targets[39, 0]),
                               (10000 - original.mean[0]) / original.std[0], places=3)
        self.assertNotEqual(original.protocol_id, changed.protocol_id)

    def test_split_horizon_boundaries_and_tail_coverage(self):
        data = prepare_data(self.protocol_path)
        covered = set()
        for split, indices in data.splits.items():
            lo, hi = data.protocol["split_bounds"][split]
            self.assertTrue(np.all(indices + 1 >= lo))
            self.assertTrue(np.all(indices + data.protocol["H"] < hi))
            self.assertTrue(np.all(indices >= data.protocol["W"] - 1))
            target_times = {int(t) for end in indices for t in range(int(end) + 1, int(end) + 4)}
            self.assertTrue(covered.isdisjoint(target_times))
            covered.update(target_times)
            dataset = WindowDataset(data, split)
            batches = list(dataset.iter_batches(7, with_previous=True))
            self.assertEqual(sum(len(batch["window_id"]) for batch in batches), len(dataset))
            np.testing.assert_array_equal(np.concatenate([batch["window_id"] for batch in batches]), indices)
            self.assertFalse(batches[0]["previous_mask"][0])
            self.assertEqual(batches[0]["history"].shape[1:], (4, 2, 7))
            self.assertEqual(batches[0]["target"].shape[1:], (2, 3))
        self.assertEqual(data.splits["val"][0] + 1, data.metadata["train_end"])

    def test_external_mask_is_explicit_and_strict(self):
        mask_path = self.root / "mask.csv"
        missing = np.zeros_like(self.values)
        missing[36:39, 0] = 1
        pd.DataFrame(missing, columns=["a", "b"]).to_csv(mask_path, index=False)
        self.config["missing_mask_path"] = str(mask_path)
        self._write_config()
        data = prepare_data(self.protocol_path)
        self.assertTrue(data.metadata["external_missing_mask_used"])
        self.assertFalse(data.observed[36:39, 0].any())
        self.assertTrue(np.all(data.targets[36:39, 0] == 0))
        missing[0, 1] = 2
        pd.DataFrame(missing, columns=["a", "b"]).to_csv(mask_path, index=False)
        with self.assertRaisesRegex(ValueError, "only 0"):
            prepare_data(self.protocol_path)
        pd.DataFrame(missing[:-1], columns=["a", "b"]).to_csv(mask_path, index=False)
        with self.assertRaisesRegex(ValueError, "row count"):
            prepare_data(self.protocol_path)

    def test_explicit_missing_interval_and_max_steps_change_protocol(self):
        original = prepare_data(self.protocol_path)
        interval = {"start": "2025-01-01 03:00:00", "end": "2025-01-01 03:15:00", "node_ids": ["a"]}
        amended = prepare_data(self.protocol_path, {"known_missing_intervals": [interval]})
        self.assertFalse(amended.observed[36:39, 0].any())
        self.assertTrue(amended.observed[36:39, 1].all())
        self.assertNotEqual(original.protocol_id, amended.protocol_id)
        shortened = prepare_data(self.protocol_path, {"max_steps": 50})
        self.assertEqual(len(shortened.raw), 50)
        self.assertNotEqual(original.protocol_id, shortened.protocol_id)

    def test_constant_and_empty_graph_are_finite(self):
        graph = build_candidate_graph(np.array([[0.0, 0.0]]), np.ones((1, 20)), 12, k=4)
        self.assertEqual(graph["edges"].shape, (0, 2))
        np.testing.assert_array_equal(graph["degree"], [1])
        graph = build_candidate_graph(np.zeros((3, 2)), np.ones((3, 20)), 12, k=1)
        self.assertTrue(np.isfinite(graph["edge_feats"]).all())
        self.assertTrue(np.all(graph["src"] != graph["dst"]))
        self.assertEqual(graph["edges"].tolist(), sorted(graph["edges"].tolist()))


if __name__ == "__main__":
    unittest.main()
