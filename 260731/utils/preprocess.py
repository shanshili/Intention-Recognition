# -*- coding: utf-8 -*-
"""交通数据结构与预处理。"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class TrafficData:
    X: np.ndarray
    coords: np.ndarray
    hour: np.ndarray
    weekday: np.ndarray
    true_chains: List = field(default_factory=list)
    # 与 X 的第二维严格同长；pos=0 对应配置中的 start_datetime。
    timestamps: Optional[np.ndarray] = None

    @property
    def num_nodes(self) -> int:
        return self.X.shape[0]

    @property
    def num_steps(self) -> int:
        return self.X.shape[1]


class Preprocessor:
    """仅使用训练段统计量，对每个节点执行 z-score。"""

    def __init__(self, train_end: int):
        self.train_end = int(train_end)
        self.mean_ = None
        self.std_ = None

    def fit_transform(self, data: TrafficData) -> dict:
        X = np.asarray(data.X, dtype=float)
        train = X[:, : self.train_end]
        self.mean_ = np.nanmean(train, axis=1, keepdims=True)
        std = np.nanstd(train, axis=1, keepdims=True)
        std[std < 1e-6] = 1.0
        self.std_ = std
        Z = (X - self.mean_) / self.std_
        Z = np.nan_to_num(Z, nan=0.0)
        return {
            "Z": Z,
            "hour": data.hour,
            "weekday": data.weekday,
            "timestamps": data.timestamps,
            "mean": self.mean_,
            "std": self.std_,
        }

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        return Z * self.std_ + self.mean_
