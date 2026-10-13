"""
Unit tests for metric computation and utility functions:
compute_metrics, compute_persistence_metrics, save_figure
"""
import os
import numpy as np
import torch
import pytest
from unittest.mock import patch, MagicMock
from torch.utils.data import DataLoader, TensorDataset


class TestComputeMetrics:
    def test_perfect_prediction(self, mod_260604, tmp_path):
        all_true = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        all_pred = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with patch.object(mod_260604, "TIMESTAMP", "test"):
            mse, mae = mod_260604.compute_metrics(all_true, all_pred, str(tmp_path))
        assert mse == 0.0
        assert mae == 0.0

    def test_known_error(self, mod_260604, tmp_path):
        all_true = np.array([[0.0, 0.0], [0.0, 0.0]])
        all_pred = np.array([[1.0, 1.0], [1.0, 1.0]])
        with patch.object(mod_260604, "TIMESTAMP", "test"):
            mse, mae = mod_260604.compute_metrics(all_true, all_pred, str(tmp_path))
        assert abs(mse - 1.0) < 1e-6
        assert abs(mae - 1.0) < 1e-6

    def test_writes_file(self, mod_260604, tmp_path):
        all_true = np.array([[1.0, 2.0]])
        all_pred = np.array([[1.5, 2.5]])
        with patch.object(mod_260604, "TIMESTAMP", "test"):
            mod_260604.compute_metrics(all_true, all_pred, str(tmp_path))
        assert os.path.exists(tmp_path / "metrics_test.txt")


class TestComputePersistenceMetrics:
    def test_constant_series_zero_error(self, mod_260604):
        """If all values are the same, persistence baseline should have 0 error."""
        N, T, pred_len = 5, 50, 6
        # All values = 1.0 -> last historical value = 1.0, future = 1.0
        data = torch.ones(N, T)
        ds = mod_260604.TrafficIntentDataset(data, target_node_idx=0, window=10, pred_len=pred_len, stride=1)
        ds.labels = torch.zeros(len(ds), dtype=torch.long)
        loader = DataLoader(ds, batch_size=8)
        with patch.object(mod_260604, "PRED_LEN", pred_len):
            mse, mae = mod_260604.compute_persistence_metrics(loader, target_idx=0)
        assert abs(mse) < 1e-6
        assert abs(mae) < 1e-6

    def test_nonconstant_series(self, mod_260604):
        """With varying data, persistence metrics should be non-zero."""
        N, T, pred_len = 5, 100, 6
        torch.manual_seed(42)
        data = torch.randn(N, T)
        ds = mod_260604.TrafficIntentDataset(data, target_node_idx=0, window=10, pred_len=pred_len, stride=1)
        ds.labels = torch.zeros(len(ds), dtype=torch.long)
        loader = DataLoader(ds, batch_size=16)
        with patch.object(mod_260604, "PRED_LEN", pred_len):
            mse, mae = mod_260604.compute_persistence_metrics(loader, target_idx=0)
        assert mse > 0
        assert mae > 0


class TestSaveFigure:
    def test_creates_files(self, mod_260604, tmp_path):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        with patch.object(mod_260604, "TIMESTAMP", "test"):
            mod_260604.save_figure(fig, "test_plot", str(tmp_path), dpi=72)
        assert os.path.exists(tmp_path / "test_plot_test.png")
        assert os.path.exists(tmp_path / "test_plot_test.svg")
        assert os.path.exists(tmp_path / "test_plot_test.pdf")

    def test_closes_figure(self, mod_260604, tmp_path):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        num_figs_before = len(plt.get_fignums())
        with patch.object(mod_260604, "TIMESTAMP", "test2"):
            mod_260604.save_figure(fig, "test_close", str(tmp_path), dpi=72)
        num_figs_after = len(plt.get_fignums())
        assert num_figs_after < num_figs_before
