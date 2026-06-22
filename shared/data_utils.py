"""
Shared data loading, graph construction, and dataset utilities.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.spatial import distance_matrix


def load_node_coords(filepath):
    df = pd.read_csv(filepath)
    node_ids = df["node_id"].values
    coords = df[["x", "y"]].values.astype(np.float32)
    return node_ids, coords


def load_time_series(filepath, check_validity=True):
    df = pd.read_csv(filepath)
    node_names = df.columns.tolist()
    data = df.values.T  # (N, T)
    if check_validity:
        if np.isnan(data).any():
            raise ValueError("时间序列含有NaN值")
        if np.isinf(data).any():
            raise ValueError("时间序列含有Inf值")
    return torch.FloatTensor(data), node_names


def generate_time_labels(T, start_hour=0):
    labels = np.zeros(T, dtype=int)
    for t in range(T):
        hour = (start_hour + t) % 24
        if 6 <= hour < 9:
            labels[t] = 1
        elif 17 <= hour < 20:
            labels[t] = 2
    return torch.LongTensor(labels)


def build_correlation_graph(data, threshold=0.5):
    data_np = data.numpy()
    corr = np.corrcoef(data_np)
    adj = (np.abs(corr) > threshold).astype(np.float32)
    np.fill_diagonal(adj, 0)
    adj = (adj + adj.T) > 0
    adj = adj.astype(np.float32)
    rowsum = adj.sum(1, keepdims=True)
    rowsum = np.where(rowsum == 0, 1, rowsum)
    adj_norm = adj / rowsum
    return torch.FloatTensor(adj_norm), adj


def build_hybrid_graph(coords, data, dist_threshold=0.1, corr_threshold=0.7,
                       normalize_coords=True):
    if normalize_coords:
        coords_min = coords.min(axis=0, keepdims=True)
        coords_max = coords.max(axis=0, keepdims=True)
        coords = (coords - coords_min) / (coords_max - coords_min + 1e-8)
    dist = distance_matrix(coords, coords)
    data_np = data.numpy()
    corr = np.corrcoef(data_np)
    adj_dist = (dist < dist_threshold).astype(np.float32)
    adj_corr = (np.abs(corr) > corr_threshold).astype(np.float32)
    adj = adj_dist * adj_corr
    np.fill_diagonal(adj, 0)
    adj = (adj + adj.T) > 0
    adj = adj.astype(np.float32)
    for i in range(adj.shape[0]):
        if adj[i].sum() == 0:
            idx = np.argpartition(dist[i], 3)[:3]
            for j in idx:
                if i != j:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
    rowsum = adj.sum(1, keepdims=True)
    rowsum = np.where(rowsum == 0, 1, rowsum)
    adj_norm = adj / rowsum
    return torch.FloatTensor(adj_norm), adj


class TrafficIntentDataset(Dataset):
    """Unified dataset supporting both time-label and future-label modes.

    Parameters
    ----------
    data : Tensor (N, T)
    target_node_idx : int
    time_labels : optional LongTensor (T,) for time-period labeling mode (260601)
    window, pred_len, stride : sliding window params
    """

    def __init__(self, data, target_node_idx, window=24, pred_len=6,
                 stride=1, time_labels=None):
        self.data = data
        self.target_idx = target_node_idx
        self.window = window
        self.pred_len = pred_len
        self.time_labels = time_labels
        self.samples = []
        T = data.shape[1]
        for start in range(0, T - window - pred_len + 1, stride):
            self.samples.append(start)
        self.labels = None  # can be set externally for future-label mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start = self.samples[idx]
        end = start + self.window
        pred_end = end + self.pred_len
        x = self.data[:, start:end]
        y_pred = self.data[self.target_idx, end:pred_end]
        if self.time_labels is not None:
            y_cls = self.time_labels[end - 1]
        elif self.labels is not None:
            y_cls = self.labels[idx]
        else:
            y_cls = -1
        return x, y_cls, y_pred


def compute_future_labels(dataset, train_indices, num_classes=3):
    train_means = []
    for idx in train_indices:
        _, _, y_pred = dataset[idx]
        train_means.append(y_pred.mean().item())
    train_means = np.array(train_means)
    thresholds = np.percentile(train_means, [100 / 3, 200 / 3])
    print(f"标签阈值 (训练集未来平均流量分位数): {thresholds}")
    all_labels = np.zeros(len(dataset), dtype=np.int64)
    for i in range(len(dataset)):
        _, _, y_pred = dataset[i]
        mean_val = y_pred.mean().item()
        if mean_val <= thresholds[0]:
            all_labels[i] = 0
        elif mean_val <= thresholds[1]:
            all_labels[i] = 1
        else:
            all_labels[i] = 2
    return all_labels
