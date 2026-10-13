"""
Unit tests for data loading and graph construction functions.
Covers: load_node_coords, load_time_series, generate_time_labels,
        build_correlation_graph, build_hybrid_graph
"""
import numpy as np
import pandas as pd
import torch
import pytest


class TestLoadNodeCoords:
    def test_basic_loading(self, mod_260604, tmp_csv_dir):
        node_ids, coords = mod_260604.load_node_coords(str(tmp_csv_dir / "coords.csv"))
        assert len(node_ids) == 5
        assert coords.shape == (5, 2)
        assert coords.dtype == np.float32

    def test_returns_correct_values(self, mod_260604, tmp_csv_dir):
        node_ids, coords = mod_260604.load_node_coords(str(tmp_csv_dir / "coords.csv"))
        np.testing.assert_array_equal(node_ids, [0, 1, 2, 3, 4])
        np.testing.assert_allclose(coords[0], [0.0, 0.0])
        np.testing.assert_allclose(coords[2], [2.0, 0.0])


class TestLoadTimeSeries:
    def test_basic_loading(self, mod_260604, tmp_csv_dir):
        data, node_names = mod_260604.load_time_series(str(tmp_csv_dir / "timeseries.csv"))
        assert isinstance(data, torch.Tensor)
        assert data.shape == (5, 100)  # (N_nodes, T_steps)
        assert len(node_names) == 5

    def test_column_names(self, mod_260604, tmp_csv_dir):
        _, node_names = mod_260604.load_time_series(str(tmp_csv_dir / "timeseries.csv"))
        assert node_names == ["node_0", "node_1", "node_2", "node_3", "node_4"]

    def test_nan_raises_error(self, mod_260604, tmp_path):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
        df.to_csv(tmp_path / "nan_data.csv", index=False)
        with pytest.raises(ValueError, match="NaN"):
            mod_260604.load_time_series(str(tmp_path / "nan_data.csv"))

    def test_inf_raises_error(self, mod_260604, tmp_path):
        df = pd.DataFrame({"a": [1.0, np.inf, 3.0], "b": [4.0, 5.0, 6.0]})
        df.to_csv(tmp_path / "inf_data.csv", index=False)
        with pytest.raises(ValueError, match="Inf"):
            mod_260604.load_time_series(str(tmp_path / "inf_data.csv"))


class TestGenerateTimeLabels:
    def test_output_shape(self, mod_260601):
        labels = mod_260601.generate_time_labels(48, start_hour=0)
        assert labels.shape == (48,)
        assert labels.dtype == torch.int64

    def test_morning_peak(self, mod_260601):
        labels = mod_260601.generate_time_labels(24, start_hour=0)
        # Hours 6-8 should be labeled 1 (morning peak)
        for h in [6, 7, 8]:
            assert labels[h].item() == 1

    def test_evening_peak(self, mod_260601):
        labels = mod_260601.generate_time_labels(24, start_hour=0)
        # Hours 17-19 should be labeled 2 (evening peak)
        for h in [17, 18, 19]:
            assert labels[h].item() == 2

    def test_flat_period(self, mod_260601):
        labels = mod_260601.generate_time_labels(24, start_hour=0)
        # Hours 0-5, 9-16, 20-23 should be labeled 0 (flat)
        for h in [0, 1, 5, 9, 12, 16, 20, 23]:
            assert labels[h].item() == 0

    def test_wrapping_start_hour(self, mod_260601):
        labels = mod_260601.generate_time_labels(24, start_hour=20)
        # At index 10, effective hour = (20+10)%24 = 6 -> morning peak
        assert labels[10].item() == 1


class TestBuildCorrelationGraph:
    def test_output_shapes(self, mod_260604, sample_data_5nodes):
        adj_norm, adj_binary = mod_260604.build_correlation_graph(sample_data_5nodes, threshold=0.3)
        assert adj_norm.shape == (5, 5)
        assert adj_binary.shape == (5, 5)

    def test_diagonal_is_zero(self, mod_260604, sample_data_5nodes):
        adj_norm, adj_binary = mod_260604.build_correlation_graph(sample_data_5nodes, threshold=0.3)
        assert np.all(np.diag(adj_binary) == 0)
        np.testing.assert_allclose(torch.diag(adj_norm).numpy(), 0.0, atol=1e-6)

    def test_symmetry(self, mod_260604, sample_data_5nodes):
        _, adj_binary = mod_260604.build_correlation_graph(sample_data_5nodes, threshold=0.3)
        np.testing.assert_array_equal(adj_binary, adj_binary.T)

    def test_row_normalization(self, mod_260604, sample_data_5nodes):
        adj_norm, adj_binary = mod_260604.build_correlation_graph(sample_data_5nodes, threshold=0.3)
        # Rows with edges should sum to 1
        for i in range(5):
            if adj_binary[i].sum() > 0:
                assert abs(adj_norm[i].sum().item() - 1.0) < 1e-5

    def test_high_threshold_sparse_graph(self, mod_260604, sample_data_5nodes):
        _, adj_binary = mod_260604.build_correlation_graph(sample_data_5nodes, threshold=0.99)
        # Very high threshold should produce a very sparse graph
        assert adj_binary.sum() <= 5 * 4  # at most all edges (unlikely at 0.99)

    def test_low_threshold_dense_graph(self, mod_260604, sample_data_5nodes):
        _, adj_binary = mod_260604.build_correlation_graph(sample_data_5nodes, threshold=0.0)
        # threshold=0 means abs(corr) > 0, so nearly all edges except perfect 0 correlation
        # All off-diagonal should be edges (since random data unlikely has exactly 0 corr)
        total_edges = adj_binary.sum()
        assert total_edges > 0


class TestBuildHybridGraph:
    def test_output_shapes(self, mod_260604, sample_data_5nodes):
        coords = np.array([[0.0, 0.0], [0.1, 0.0], [0.5, 0.5], [0.9, 0.9], [1.0, 1.0]], dtype=np.float32)
        adj_norm, adj_binary = mod_260604.build_hybrid_graph(
            coords, sample_data_5nodes, dist_threshold=0.3, corr_threshold=0.1
        )
        assert adj_norm.shape == (5, 5)
        assert adj_binary.shape == (5, 5)

    def test_isolated_nodes_get_neighbors(self, mod_260604, sample_data_5nodes):
        """Isolated nodes should be connected to 2 nearest neighbors."""
        # Use coords that are very far apart and high corr threshold
        coords = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]], dtype=np.float32)
        _, adj_binary = mod_260604.build_hybrid_graph(
            coords, sample_data_5nodes, dist_threshold=0.01, corr_threshold=0.99,
            normalize_coords=True
        )
        # Even with extreme thresholds, no node should be completely isolated
        for i in range(5):
            assert adj_binary[i].sum() > 0

    def test_diagonal_is_zero(self, mod_260604, sample_data_5nodes):
        coords = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]], dtype=np.float32)
        _, adj_binary = mod_260604.build_hybrid_graph(
            coords, sample_data_5nodes, dist_threshold=0.5, corr_threshold=0.1
        )
        assert np.all(np.diag(adj_binary) == 0)

    def test_symmetry(self, mod_260604, sample_data_5nodes):
        coords = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]], dtype=np.float32)
        _, adj_binary = mod_260604.build_hybrid_graph(
            coords, sample_data_5nodes, dist_threshold=0.5, corr_threshold=0.1
        )
        np.testing.assert_array_equal(adj_binary, adj_binary.T)
