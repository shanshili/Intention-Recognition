# -*- coding: utf-8 -*-
"""Estructura de datos de trafico y preprocesado (z-score con stats de train)."""
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class TrafficData:
    X: np.ndarray                 # [N, T] flujo (crudo o normalizado)
    coords: np.ndarray            # [N, 2] coordenadas de sensores
    hour: np.ndarray              # [T]
    weekday: np.ndarray           # [T]
    true_chains: List = field(default_factory=list)

    @property
    def num_nodes(self) -> int:
        return self.X.shape[0]

    @property
    def num_steps(self) -> int:
        return self.X.shape[1]


class Preprocessor:
    """Normaliza cada nodo con media/desviacion calculadas solo en train."""

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
            "mean": self.mean_,
            "std": self.std_,
        }

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        return Z * self.std_ + self.mean_
