"""
Unit tests for TrafficIntentDataset and compute_future_labels.
"""
import numpy as np
import torch
import pytest


class TestTrafficIntentDataset260601:
    """Tests for the dataset class in 260601.py (uses time_labels parameter)."""

    def test_basic_creation(self, mod_260601, sample_data_5nodes):
        time_labels = mod_260601.generate_time_labels(100, start_hour=0)
        ds = mod_260601.TrafficIntentDataset(
            sample_data_5nodes, time_labels, target_node_idx=0,
            window=24, pred_len=6, stride=1
        )
        assert len(ds) > 0

    def test_length_formula(self, mod_260601, sample_data_5nodes):
        time_labels = mod_260601.generate_time_labels(100, start_hour=0)
        ds = mod_260601.TrafficIntentDataset(
            sample_data_5nodes, time_labels, target_node_idx=0,
            window=24, pred_len=6, stride=1
        )
        # T=100, window=24, pred_len=6 -> valid starts: 0..70 -> 71 samples
        expected = 100 - 24 - 6 + 1
        assert len(ds) == expected

    def test_getitem_shapes(self, mod_260601, sample_data_5nodes):
        time_labels = mod_260601.generate_time_labels(100, start_hour=0)
        ds = mod_260601.TrafficIntentDataset(
            sample_data_5nodes, time_labels, target_node_idx=0,
            window=24, pred_len=6, stride=1
        )
        x, y_cls, y_pred = ds[0]
        assert x.shape == (5, 24)   # (N, window)
        assert y_cls.shape == ()    # scalar
        assert y_pred.shape == (6,) # (pred_len,)

    def test_stride_reduces_samples(self, mod_260601, sample_data_5nodes):
        time_labels = mod_260601.generate_time_labels(100, start_hour=0)
        ds_s1 = mod_260601.TrafficIntentDataset(
            sample_data_5nodes, time_labels, target_node_idx=0,
            window=24, pred_len=6, stride=1
        )
        ds_s3 = mod_260601.TrafficIntentDataset(
            sample_data_5nodes, time_labels, target_node_idx=0,
            window=24, pred_len=6, stride=3
        )
        assert len(ds_s3) < len(ds_s1)


class TestTrafficIntentDataset260604:
    """Tests for the dataset class in 260604.py (label-free, uses external labels)."""

    def test_basic_creation(self, mod_260604, sample_data_5nodes):
        ds = mod_260604.TrafficIntentDataset(
            sample_data_5nodes, target_node_idx=0,
            window=24, pred_len=6, stride=1
        )
        assert len(ds) > 0
        assert ds.labels is None

    def test_length_formula(self, mod_260604, sample_data_5nodes):
        ds = mod_260604.TrafficIntentDataset(
            sample_data_5nodes, target_node_idx=0,
            window=24, pred_len=6, stride=1
        )
        expected = 100 - 24 - 6 + 1
        assert len(ds) == expected

    def test_getitem_shapes(self, mod_260604, sample_data_5nodes):
        ds = mod_260604.TrafficIntentDataset(
            sample_data_5nodes, target_node_idx=0,
            window=24, pred_len=6, stride=1
        )
        x, y_cls, y_pred = ds[0]
        assert x.shape == (5, 24)
        assert y_cls == -1  # No labels assigned yet
        assert y_pred.shape == (6,)

    def test_with_labels(self, mod_260604, sample_data_5nodes):
        ds = mod_260604.TrafficIntentDataset(
            sample_data_5nodes, target_node_idx=0,
            window=24, pred_len=6, stride=1
        )
        # Assign labels
        labels = torch.zeros(len(ds), dtype=torch.long)
        labels[0] = 2
        ds.labels = labels
        _, y_cls, _ = ds[0]
        assert y_cls == 2

    def test_target_node_prediction(self, mod_260604, sample_data_5nodes):
        """Verify y_pred corresponds to the target node's future values."""
        target_idx = 2
        ds = mod_260604.TrafficIntentDataset(
            sample_data_5nodes, target_node_idx=target_idx,
            window=24, pred_len=6, stride=1
        )
        x, _, y_pred = ds[0]
        # y_pred should be data[target_idx, 24:30]
        expected = sample_data_5nodes[target_idx, 24:30]
        torch.testing.assert_close(y_pred, expected)

    def test_window_content(self, mod_260604, sample_data_5nodes):
        """Verify x corresponds to data[:, start:start+window]."""
        ds = mod_260604.TrafficIntentDataset(
            sample_data_5nodes, target_node_idx=0,
            window=24, pred_len=6, stride=3
        )
        # Second sample starts at index 3 (stride=3)
        x, _, _ = ds[1]
        expected = sample_data_5nodes[:, 3:27]
        torch.testing.assert_close(x, expected)


class TestComputeFutureLabels:
    def test_basic_label_generation(self, mod_260604, sample_data_5nodes):
        ds = mod_260604.TrafficIntentDataset(
            sample_data_5nodes, target_node_idx=0,
            window=10, pred_len=4, stride=1
        )
        total = len(ds)
        train_indices = list(range(total // 2))
        labels = mod_260604.compute_future_labels(ds, train_indices, num_classes=3)
        assert labels.shape == (total,)
        assert set(np.unique(labels)).issubset({0, 1, 2})

    def test_three_classes_present_in_training(self, mod_260604, sample_data_5nodes):
        ds = mod_260604.TrafficIntentDataset(
            sample_data_5nodes, target_node_idx=0,
            window=10, pred_len=4, stride=1
        )
        total = len(ds)
        train_indices = list(range(total))
        labels = mod_260604.compute_future_labels(ds, train_indices, num_classes=3)
        # With enough samples and random data, all 3 classes should appear
        assert len(np.unique(labels)) == 3

    def test_label_dtype(self, mod_260604, sample_data_5nodes):
        ds = mod_260604.TrafficIntentDataset(
            sample_data_5nodes, target_node_idx=0,
            window=10, pred_len=4, stride=1
        )
        total = len(ds)
        train_indices = list(range(total // 2))
        labels = mod_260604.compute_future_labels(ds, train_indices)
        assert labels.dtype == np.int64
