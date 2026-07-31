# -*- coding: utf-8 -*-
r"""样本构造与严格时间映射。

符号约定：
- c：意图图序列的局部索引，范围 0..n_show-1；
- pos：原始流量序列位置，pos=0 严格对应 2025-01-01 00:00:00；
- 部署时刻：t=pos；
- 监督预测目标：t+1=pos+1。
"""
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


class SampleBuilder:
    def __init__(self, bundle: Dict, cfg, device):
        self.cfg = cfg
        self.device = device
        self.N = bundle["N"]
        self.X = bundle["data"].X.astype(np.float32)
        self.coords = bundle["data"].coords.astype(np.float32)
        self.timestamps = pd.DatetimeIndex(bundle["timestamps"])
        self.A_C = bundle["A_C"].astype(np.float32)
        self.A_I_list = [a.astype(np.float32) for a in bundle["A_I_list"]]
        self.intent_positions = np.asarray(bundle["intent_positions"], dtype=int)
        self.intent_alignment = dict(bundle.get("intent_alignment", {}))

        nf = bundle.get("node_feat")
        if nf is not None:
            nf = np.asarray(nf, dtype=np.float32)
            nmin = nf.min(0, keepdims=True)
            nmax = nf.max(0, keepdims=True)
            nf = (nf - nmin) / (nmax - nmin + 1e-8)
        self.node_feat = nf

        self.n_show = len(self.A_I_list)
        self.T = self.X.shape[1]
        self.L = cfg.model.L
        self.r = cfg.model.r
        if len(self.timestamps) != self.T:
            raise ValueError("timestamps 长度必须等于流量序列 T")
        if len(self.intent_positions) != self.n_show:
            raise ValueError("intent_positions 长度必须等于 A_I_list 长度")

        # 仅为兼容旧代码；非连续显式时间索引时请使用 _pos(c)。
        self.base = int(self.intent_positions[0])

        cmin = self.coords.min(0, keepdims=True)
        cmax = self.coords.max(0, keepdims=True)
        self.coord_norm = (self.coords - cmin) / (cmax - cmin + 1e-8)

        self._A_C_t = torch.tensor(self.A_C, dtype=torch.float32, device=device)
        self.valid_centers = self._compute_valid_centers()

    def in_dim(self) -> int:
        d = 1
        if self.cfg.model.node_extra_feat:
            d += 2
        if self.node_feat is not None:
            d += self.node_feat.shape[1]
        return d

    def _pos(self, c: int) -> int:
        c = int(c)
        if c < 0 or c >= self.n_show:
            raise IndexError(f"意图局部索引 c={c} 越界")
        return int(self.intent_positions[c])

    def timestamp_at_pos(self, pos: int) -> pd.Timestamp:
        pos = int(pos)
        if pos < 0 or pos >= self.T:
            raise IndexError(f"原始时间位置 pos={pos} 越界")
        return pd.Timestamp(self.timestamps[pos])

    def time_str_at_pos(self, pos: int, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
        return self.timestamp_at_pos(pos).strftime(fmt)

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
        c = int(c)
        pos = self._pos(c)
        target_pos = pos + 1
        L = self.L

        window = self.X[:, pos - L + 1: pos + 1]
        feats = []
        for t in range(L):
            f = window[:, t:t + 1]
            if self.cfg.model.node_extra_feat:
                f = np.concatenate([f, self.coord_norm], axis=1)
            if self.node_feat is not None:
                f = np.concatenate([f, self.node_feat], axis=1)
            feats.append(f)
        Xseq = np.stack(feats, axis=0)

        target = self.X[:, target_pos]
        return {
            "c": c,
            "pos": pos,
            "time": self.time_str_at_pos(pos),
            "target_pos": target_pos,
            "target_time": self.time_str_at_pos(target_pos),
            "window_start_pos": pos - L + 1,
            "window_start_time": self.time_str_at_pos(pos - L + 1),
            "Xseq": torch.tensor(Xseq, dtype=torch.float32, device=self.device),
            "A_C": self._A_C_t,
            "A_I_prev": torch.tensor(
                self.A_I_list[c - self.r], dtype=torch.float32, device=self.device),
            "A_I_cur": torch.tensor(
                self.A_I_list[c], dtype=torch.float32, device=self.device),
            "A_I_next": torch.tensor(
                self.A_I_list[c + self.r], dtype=torch.float32, device=self.device),
            "target": torch.tensor(target, dtype=torch.float32, device=self.device),
        }
