# -*- coding: utf-8 -*-
r"""Construccion de muestras: mapea cada paso mostrado c a una ventana temporal.

Se asume que los n_show pasos de intencion corresponden a los ultimos n_show
instantes de la serie: pos(c) = (T - n_show) + c.
"""
from typing import Dict, List

import numpy as np
import torch


class SampleBuilder:
    def __init__(self, bundle: Dict, cfg, device):
        self.cfg = cfg
        self.device = device
        self.N = bundle["N"]
        self.X = bundle["data"].X.astype(np.float32)          # [N, T]
        self.coords = bundle["data"].coords.astype(np.float32)  # [N, 2]
        self.A_C = bundle["A_C"].astype(np.float32)             # [N, N]
        self.A_I_list = [a.astype(np.float32) for a in bundle["A_I_list"]]
        # caracteristicas iniciales de nodo independientes (opcional) [N, F0]
        nf = bundle.get("node_feat")
        if nf is not None:
            nf = np.asarray(nf, dtype=np.float32)
            nmin = nf.min(0, keepdims=True)
            nmax = nf.max(0, keepdims=True)
            nf = (nf - nmin) / (nmax - nmin + 1e-8)          # normalizacion min-max
        self.node_feat = nf                                   # [N, F0] o None
        self.n_show = len(self.A_I_list)
        self.T = self.X.shape[1]
        self.L = cfg.model.L
        self.r = cfg.model.r
        self.base = self.T - self.n_show                       # offset

        # coords normalizadas [N,2] en [0,1]
        cmin = self.coords.min(0, keepdims=True)
        cmax = self.coords.max(0, keepdims=True)
        self.coord_norm = (self.coords - cmin) / (cmax - cmin + 1e-8)

        self._A_C_t = torch.tensor(self.A_C, device=device)
        self._coord_t = torch.tensor(self.coord_norm, device=device)

        self.valid_centers = self._compute_valid_centers()

    def in_dim(self) -> int:
        d = 1                                          # flujo del instante
        if self.cfg.model.node_extra_feat:
            d += 2                                     # coordenadas (x, y)
        if self.node_feat is not None:
            d += self.node_feat.shape[1]               # caracteristicas iniciales
        return d

    def _pos(self, c: int) -> int:
        return self.base + c

    def _compute_valid_centers(self) -> List[int]:
        centers = []
        for c in range(self.n_show):
            if c - self.r < 0 or c + self.r >= self.n_show:
                continue
            pos = self._pos(c)
            if pos - self.L + 1 < 0:
                continue
            if pos + 1 > self.T - 1:
                continue
            centers.append(c)
        return centers

    def split(self, train_ratio: float = 0.6):
        n = len(self.valid_centers)
        k = int(train_ratio * n)
        return self.valid_centers[:k], self.valid_centers[k:]

    def build(self, c: int) -> Dict:
        pos = self._pos(c)
        L = self.L
        # ventana de features [L, N, F]
        window = self.X[:, pos - L + 1: pos + 1]               # [N, L]
        feats = []
        for t in range(L):
            f = window[:, t:t + 1]                              # [N,1] flujo
            if self.cfg.model.node_extra_feat:
                f = np.concatenate([f, self.coord_norm], axis=1)  # + coords
            if self.node_feat is not None:
                f = np.concatenate([f, self.node_feat], axis=1)   # + feat iniciales
            feats.append(f)
        Xseq = np.stack(feats, axis=0)                         # [L, N, F]

        target = self.X[:, pos + 1]                            # [N]

        sample = {
            "c": c, "pos": pos,
            "Xseq": torch.tensor(Xseq, dtype=torch.float32, device=self.device),
            "A_C": self._A_C_t,
            "A_I_prev": torch.tensor(self.A_I_list[c - self.r], device=self.device),
            "A_I_cur": torch.tensor(self.A_I_list[c], device=self.device),
            "A_I_next": torch.tensor(self.A_I_list[c + self.r], device=self.device),
            "target": torch.tensor(target, dtype=torch.float32, device=self.device),
        }
        return sample
