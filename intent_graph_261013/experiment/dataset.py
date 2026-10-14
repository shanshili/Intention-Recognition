"""Lazy windows shared by all models; incomplete target horizons are excluded."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from .data import PreparedData


class WindowDataset:
    def __init__(self, data: PreparedData, split: str):
        if split not in data.splits:
            raise ValueError(f"Unknown split: {split}")
        self.data, self.split = data, split
        self.indices = data.splits[split]
        self.W, self.H = data.protocol["W"], data.protocol["H"]
        self._index_set = set(self.indices.tolist())

    def __len__(self) -> int:
        return len(self.indices)

    def batch(self, indices: np.ndarray, with_previous: bool = False) -> dict:
        ends = np.asarray(indices, dtype=np.int64)
        history_rows = ends[:, None] - np.arange(self.W - 1, -1, -1)[None, :]
        target_rows = ends[:, None] + np.arange(1, self.H + 1)[None, :]
        result = {
            "history": self.data.features[history_rows],
            "target": self.data.targets[target_rows].transpose(0, 2, 1),
            "target_mask": self.data.observed[target_rows].transpose(0, 2, 1),
            "last": self.data.features[ends, :, 0],
            "raw_target": self.data.raw[target_rows].transpose(0, 2, 1),
            "window_id": ends.copy(),
            "target_times": self.data.timestamps[target_rows],
        }
        if with_previous:
            stride = self.data.protocol["train_stride" if self.split == "train" else "eval_stride"]
            previous_ends = ends - stride
            previous_mask = np.asarray([int(t) in self._index_set for t in previous_ends], dtype=bool)
            paired_ends = np.where(previous_mask, previous_ends, ends)
            rows = paired_ends[:, None] - np.arange(self.W - 1, -1, -1)[None, :]
            result["previous_history"] = self.data.features[rows]
            result["previous_mask"] = previous_mask
        return result

    def iter_batches(self, batch_size: int, shuffle_seed: int | None = None,
                     with_previous: bool = False) -> Iterator[dict]:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        indices = self.indices.copy()
        if shuffle_seed is not None:
            np.random.default_rng(shuffle_seed).shuffle(indices)
        for start in range(0, len(indices), batch_size):
            yield self.batch(indices[start:start + batch_size], with_previous=with_previous)
