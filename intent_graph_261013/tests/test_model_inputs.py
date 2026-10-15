"""Input views retain shared labels, windows, graph and canonical identity."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from experiment.data import prepare_data
from experiment.dataset import WindowDataset
from experiment.model_inputs import FULL_CHANNELS, select_model_inputs


class ModelInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.flow = self.root / "flow.csv"
        self.coords = self.root / "coords.csv"
        self.protocol_path = self.root / "protocol.json"
        values = np.stack([np.arange(96, dtype=float), np.arange(96, dtype=float) * 2], axis=1)
        pd.DataFrame(values, columns=["a", "b"]).to_csv(self.flow, index=False)
        pd.DataFrame({"node_id": ["a", "b"], "x": [0, 1], "y": [0, 1]}).to_csv(self.coords, index=False)
        self.config = {"flow_path": str(self.flow), "coord_path": str(self.coords),
                       "W": 4, "H": 3, "features": ["level", "time", "delta", "periodic_residual"],
                       "candidate_graph": {"k": 1, "tau_max": 2},
                       "periodic_residual_resolution": "hour",
                       "timestamp_source": "user_confirmed_start_regular_grid"}
        self.protocol_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.data = prepare_data(self.protocol_path)

    def tearDown(self):
        self.temp.cleanup()

    def test_full_reorders_by_semantic_names_and_keeps_canonical_identity(self):
        view = select_model_inputs(self.data, "full")
        self.assertEqual(view.protocol["feature_channels"], list(FULL_CHANNELS))
        self.assertEqual(view.features.shape, (96, 2, 7))
        for index, name in enumerate(FULL_CHANNELS):
            source_index = self.data.protocol["feature_channels"].index(name)
            np.testing.assert_array_equal(view.features[..., index], self.data.features[..., source_index])
        self.assertEqual(view.protocol_id, self.data.protocol_id)
        self.assertEqual(view.protocol["protocol_id"], self.data.protocol_id)
        for name in ("raw", "targets", "observed", "mean", "std", "coords", "timestamps", "splits", "graph"):
            self.assertIs(getattr(view, name), getattr(self.data, name))
        self.assertEqual(self.data.protocol["feature_channels"][1], "time_sin")
        self.assertEqual(view.metadata["model_input_feature_set"], "full")

    def test_level_view_matches_full_targets_and_batch_windows(self):
        level = select_model_inputs(self.data, "level")
        full = select_model_inputs(self.data, "full")
        self.assertTrue(np.shares_memory(level.features, self.data.features))
        self.assertEqual(level.protocol["features"], ["level"])
        for split in self.data.splits:
            ends = self.data.splits[split][:2]
            left = WindowDataset(level, split).batch(ends)
            right = WindowDataset(full, split).batch(ends)
            np.testing.assert_array_equal(left["history"][..., 0], right["history"][..., 0])
            for key in ("target", "target_mask", "last", "raw_target", "window_id", "target_times"):
                np.testing.assert_array_equal(left[key], right[key])

    def test_missing_channels_and_unknown_mode_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "feature_set"):
            select_model_inputs(self.data, "extra")
        level = select_model_inputs(self.data, "level")
        with self.assertRaisesRegex(ValueError, "missing channels"):
            select_model_inputs(level, "full")

    def test_timestamp_source_records_user_confirmation_not_independent_verification(self):
        self.assertEqual(self.data.timestamps[0], np.datetime64("2025-01-01T00:00:00"))
        self.assertEqual(self.data.metadata["timestamp_source"], "user_confirmed_start_regular_grid")
        self.assertTrue(self.data.metadata["timestamp_start_user_confirmed"])
        self.assertFalse(self.data.metadata["timestamps_independently_verified"])
        with self.assertRaisesRegex(ValueError, "timestamp_source"):
            prepare_data(self.protocol_path, {"timestamp_source": "invented"})
        with self.assertRaisesRegex(ValueError, "periodic_residual_resolution"):
            prepare_data(self.protocol_path, {"periodic_residual_resolution": "day"})

    def test_hourly_expectation_uses_observed_train_values_only(self):
        values = pd.read_csv(self.flow).to_numpy()
        values[3, 0] = np.nan
        values[60, 0] = np.nan
        pd.DataFrame(values, columns=["a", "b"]).to_csv(self.flow, index=False)
        original = prepare_data(self.protocol_path)
        train_end = original.metadata["train_end"]
        residual_index = original.protocol["feature_channels"].index("periodic_residual")
        calendar = pd.DatetimeIndex(original.timestamps)
        slot = calendar.dayofweek.to_numpy() * 24 + calendar.hour.to_numpy()
        normalized = (original.raw - original.mean) / original.std
        for key in np.unique(slot):
            train_rows = np.flatnonzero(slot[:train_end] == key)
            expectation = np.zeros(2)
            if len(train_rows):
                for node in range(2):
                    valid = train_rows[original.observed[train_rows, node]]
                    if len(valid):
                        expectation[node] = normalized[valid, node].mean()
            rows = np.flatnonzero(slot == key)
            np.testing.assert_allclose(original.features[rows, :, residual_index],
                                       original.features[rows, :, 0] - expectation,
                                       atol=2e-7)
        values[train_end:] = 50000
        pd.DataFrame(values, columns=["a", "b"]).to_csv(self.flow, index=False)
        changed = prepare_data(self.protocol_path)
        np.testing.assert_array_equal(original.features[:train_end], changed.features[:train_end])
        # The missing held-out input is carried forward, not interpolated.
        np.testing.assert_allclose(original.features[60, 0, 0], original.features[59, 0, 0])
        self.assertFalse(original.observed[60, 0])


if __name__ == "__main__":
    unittest.main()
