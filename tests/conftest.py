"""
Test configuration: provides importable modules and shared fixtures.
"""
import sys
import os
import importlib.util
from unittest.mock import patch

import pytest
import numpy as np
import torch

# Add repo root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def _import_module(filename):
    """Import a module from repo root, suppressing side effects (makedirs, print)."""
    filepath = os.path.join(REPO_ROOT, filename)
    spec = importlib.util.spec_from_file_location(filename.replace(".", "_"), filepath)
    with patch("builtins.print"), patch("os.makedirs"):
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def mod_260601():
    return _import_module("260601.py")


@pytest.fixture(scope="session")
def mod_260604():
    return _import_module("260604.py")


@pytest.fixture(scope="session")
def mod_260604_tcn():
    return _import_module("260604_TCN.py")


@pytest.fixture
def sample_data_5nodes():
    """Synthetic time series data: 5 nodes, 100 time steps."""
    torch.manual_seed(42)
    np.random.seed(42)
    data = torch.randn(5, 100)
    return data


@pytest.fixture
def sample_adj_5nodes():
    """Simple adjacency matrix for 5 nodes (ring graph)."""
    adj = np.zeros((5, 5), dtype=np.float32)
    for i in range(5):
        adj[i, (i + 1) % 5] = 1.0
        adj[(i + 1) % 5, i] = 1.0
    rowsum = adj.sum(1, keepdims=True)
    adj_norm = adj / rowsum
    return torch.FloatTensor(adj_norm)


@pytest.fixture
def tmp_csv_dir(tmp_path):
    """Create temporary CSV files for data loading tests."""
    import pandas as pd

    # Node coordinates
    coords_df = pd.DataFrame({
        "node_id": [0, 1, 2, 3, 4],
        "x": [0.0, 1.0, 2.0, 3.0, 4.0],
        "y": [0.0, 1.0, 0.0, 1.0, 0.0],
    })
    coords_df.to_csv(tmp_path / "coords.csv", index=False)

    # Time series (5 nodes, 100 time steps)
    np.random.seed(42)
    ts_data = np.random.randn(100, 5)
    ts_df = pd.DataFrame(ts_data, columns=["node_0", "node_1", "node_2", "node_3", "node_4"])
    ts_df.to_csv(tmp_path / "timeseries.csv", index=False)

    return tmp_path
