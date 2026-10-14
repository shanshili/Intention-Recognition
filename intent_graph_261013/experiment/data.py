"""Shared causal preprocessing and target-time split construction.

CSV loading, train-only statistics, calendar/difference/residual features are
adapted from 260628/deig/data.py. Future interpolation, clipping of evaluation
labels, and fabricated missing-value assumptions are deliberately removed.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .graph import build_candidate_graph
from .protocol import load_protocol, protocol_hash, sha256_file


@dataclass
class PreparedData:
    raw: np.ndarray
    observed: np.ndarray
    features: np.ndarray
    targets: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    coords: np.ndarray
    node_ids: list[str]
    timestamps: np.ndarray
    splits: dict[str, np.ndarray]
    graph: dict
    protocol: dict
    protocol_id: str
    metadata: dict


def _csv_rows(path: str | Path) -> int:
    """Count records rather than line breaks, including quoted CSV records."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.reader(handle)
        next(rows, None)
        return sum(1 for row in rows if row)


def _causal_fill(raw: np.ndarray, observed: np.ndarray, mean: np.ndarray) -> np.ndarray:
    result = np.empty_like(raw, dtype=np.float64)
    last = mean.copy()
    for t in range(len(raw)):
        last = np.where(observed[t], raw[t], last)
        result[t] = last
    return result


def _split_indices(T: int, W: int, H: int, ratios: list, train_stride: int,
                   eval_stride: int) -> tuple[dict, dict]:
    train_end = int(T * ratios[0])
    val_end = int(T * (ratios[0] + ratios[1]))
    bounds = {"train": [0, train_end], "val": [train_end, val_end], "test": [val_end, T]}
    splits = {}
    for split, (lo, hi) in bounds.items():
        # Target interval is [t+1,t+H]; the complete interval stays in a split.
        first = max(W - 1, lo - 1)
        last_exclusive = hi - H
        stride = train_stride if split == "train" else eval_stride
        splits[split] = np.arange(first, last_exclusive, stride, dtype=np.int64)
        if len(splits[split]) == 0:
            raise ValueError(f"{split} has no complete windows for T={T}, W={W}, H={H}")
    return splits, bounds


def _apply_known_missing(observed: np.ndarray, timestamps: np.ndarray, node_ids: list[str],
                         intervals: list) -> None:
    for entry in intervals:
        if not isinstance(entry, dict) or "start" not in entry or "end" not in entry:
            raise ValueError("Known missing intervals require start and end")
        start, end = np.datetime64(entry["start"]), np.datetime64(entry["end"])
        if start >= end:
            raise ValueError("Known missing interval must have start < end")
        selected = entry.get("node_ids", node_ids)
        if not isinstance(selected, list) or any(node not in node_ids for node in selected):
            raise ValueError("Known missing interval contains unknown node_ids")
        rows = np.flatnonzero((timestamps >= start) & (timestamps < end))
        cols = [node_ids.index(node) for node in selected]
        observed[np.ix_(rows, cols)] = False


def prepare_data(protocol_path: str | Path, overrides: dict | None = None) -> PreparedData:
    protocol = load_protocol(protocol_path, overrides)
    flow_path, coord_path = Path(protocol["flow_path"]), Path(protocol["coord_path"])
    flow = pd.read_csv(flow_path, nrows=protocol["max_steps"])
    coordinates = pd.read_csv(coord_path)
    node_ids = flow.columns.astype(str).tolist()
    if len(set(node_ids)) != len(node_ids) or not {"node_id", "x", "y"} <= set(coordinates.columns):
        raise ValueError("Flow node IDs must be unique and coordinates need node_id,x,y")
    if coordinates["node_id"].astype(str).tolist() != node_ids:
        raise ValueError("Flow headers must exactly match coordinate node_id order")
    raw = flow.to_numpy(dtype=np.float64)
    coords = coordinates[["x", "y"]].to_numpy(dtype=np.float64)
    if not np.isfinite(coords).all():
        raise ValueError("Coordinates must be finite")
    T, N = raw.shape
    if N == 0:
        raise ValueError("Dataset has no nodes")
    observed = np.isfinite(raw)
    source_hashes = {"flow": sha256_file(flow_path), "coordinates": sha256_file(coord_path)}
    mask_path = protocol["missing_mask_path"]
    if mask_path:
        if _csv_rows(mask_path) != _csv_rows(flow_path):
            raise ValueError("Missing mask must match the complete flow CSV row count")
        missing = pd.read_csv(mask_path, nrows=protocol["max_steps"])
        if missing.columns.astype(str).tolist() != node_ids or missing.shape != raw.shape:
            raise ValueError("Missing mask header and used-prefix shape must match flow exactly")
        mask_values = missing.to_numpy(dtype=np.float64)
        if not np.isin(mask_values, [0, 1]).all():
            raise ValueError("Missing mask must contain only 0=observed and 1=missing")
        observed &= mask_values == 0
        source_hashes["missing_mask"] = sha256_file(mask_path)
    timestamps = pd.date_range(protocol["start_time"], periods=T,
                               freq=f"{protocol['step_minutes']}min").to_numpy(dtype="datetime64[ns]")
    _apply_known_missing(observed, timestamps, node_ids, protocol["known_missing_intervals"])
    splits, bounds = _split_indices(T, protocol["W"], protocol["H"], protocol["split_ratios"],
                                   protocol["train_stride"], protocol["eval_stride"])
    train_end = bounds["train"][1]
    train_mask = observed[:train_end]
    train_counts = train_mask.sum(axis=0)
    if np.any(train_counts == 0):
        names = [node_ids[i] for i in np.flatnonzero(train_counts == 0)]
        raise ValueError(f"Nodes without observed training data: {names}")
    train_values = np.where(train_mask, raw[:train_end], 0.0)
    mean = train_values.sum(axis=0) / train_counts
    centered = np.where(train_mask, raw[:train_end] - mean, 0.0)
    empirical_std = np.sqrt((centered ** 2).sum(axis=0) / train_counts)
    std = np.where(empirical_std > protocol["std_epsilon"], empirical_std, 1.0)
    filled = _causal_fill(raw, observed, mean)
    level = (filled - mean) / std
    targets = np.where(observed, (raw - mean) / std, 0.0).astype(np.float32)
    channels, channel_names = [level[..., None]], ["level"]
    calendar = pd.DatetimeIndex(timestamps)
    minute = calendar.hour.to_numpy() * 60 + calendar.minute.to_numpy()
    weekday = calendar.dayofweek.to_numpy()
    for feature in protocol["features"][1:]:
        if feature == "delta":
            delta = np.zeros_like(level)
            delta[1:] = level[1:] - level[:-1]
            channels.append(delta[..., None])
            channel_names.append("delta")
        elif feature == "time":
            time_encoding = np.stack([np.sin(2 * np.pi * minute / 1440),
                                      np.cos(2 * np.pi * minute / 1440),
                                      np.sin(2 * np.pi * weekday / 7),
                                      np.cos(2 * np.pi * weekday / 7)], axis=-1)
            channels.append(np.broadcast_to(time_encoding[:, None, :], (T, N, 4)))
            channel_names.extend(["time_sin", "time_cos", "weekday_sin", "weekday_cos"])
        elif feature == "periodic_residual":
            residual = level.copy()
            slot = weekday * 1440 + minute
            for key in np.unique(slot):
                training_rows = np.flatnonzero(slot[:train_end] == key)
                if len(training_rows):
                    masks = observed[training_rows]
                    counts = masks.sum(axis=0)
                    sums = np.where(masks, targets[training_rows], 0).sum(axis=0)
                    expectation = np.divide(sums, counts, out=np.zeros(N), where=counts > 0)
                    residual[slot == key] -= expectation
            channels.append(residual[..., None])
            channel_names.append("periodic_residual")
    features = np.concatenate(channels, axis=-1).astype(np.float32)
    graph = build_candidate_graph(coords, level.T, train_end, **protocol["candidate_graph"])
    protocol["actual_steps"] = T
    protocol["node_count"] = N
    protocol["split_bounds"] = bounds
    protocol["feature_channels"] = channel_names
    protocol["source_sha256"] = source_hashes
    identity = {key: value for key, value in protocol.items()
                if key not in {"name", "protocol_source", "flow_path", "coord_path", "missing_mask_path"}}
    identity.update({"node_ids": node_ids, "candidate_edges": graph["edges"].tolist(),
                     "split_windows": {key: values.tolist() for key, values in splits.items()}})
    protocol_id = protocol_hash(identity)
    protocol["protocol_id"] = protocol_id
    metadata = {"T": T, "N": N, "E": graph["E"], "train_end": train_end,
                "val_end": bounds["val"][1], "split_bounds": bounds,
                "split_sizes": {key: len(values) for key, values in splits.items()},
                "observed_fraction": float(observed.mean()),
                "external_missing_mask_used": bool(mask_path),
                "mask_validation_scope": "full_row_count_header_and_used_values" if mask_path else "none",
                "timestamp_source": "configured_regular_grid_not_verified_source_timestamps",
                "feature_channels": channel_names,
                "provenance": {"data": "adapted from 260628/deig/data.py",
                               "graph": "copied and adapted from 260628/deig/graph.py"}}
    return PreparedData(raw=raw, observed=observed, features=features, targets=targets,
                        mean=mean, std=std, coords=coords, node_ids=node_ids,
                        timestamps=timestamps, splits=splits, graph=graph,
                        protocol=protocol, protocol_id=protocol_id, metadata=metadata)
