# -*- coding: utf-8 -*-
r"""模块四：预算约束下的部署子图 G_D^t。

保留补充文件中的 edge-first 策略：候选边按 S_D[i,j] 降序选择；一条边
被接受时，其尚未部署的端点同时加入，确保 q_ij <= p_i, q_ij <= p_j。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

EPS = 1e-9


class ScoreHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, h_D: torch.Tensor) -> torch.Tensor:
        return self.net(h_D).squeeze(-1)


def node_scores(S_D: torch.Tensor, mlp_term: torch.Tensor) -> torch.Tensor:
    return S_D.sum(dim=1) + S_D.sum(dim=0) + mlp_term


def _binary_jaccard(A: np.ndarray, B: np.ndarray) -> float:
    a = np.asarray(A) > 0
    b = np.asarray(B) > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / (union + EPS))


def greedy_deploy(S_D: np.ndarray, scores: np.ndarray,
                  A_I: np.ndarray, A_P: np.ndarray,
                  dep_cfg, A_CI: Optional[np.ndarray] = None) -> Dict:
    """以边为中心执行贪心部署，并返回绘图/汇总所需的完整字段。"""
    S = np.nan_to_num(np.asarray(S_D, dtype=float),
                      nan=0.0, posinf=0.0, neginf=0.0)
    S = np.maximum(S, 0.0)
    score_vec = np.nan_to_num(np.asarray(scores, dtype=float).reshape(-1),
                              nan=0.0, posinf=0.0, neginf=0.0)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"S_D 必须为方阵，实际 {S.shape}")
    if score_vec.size != S.shape[0]:
        raise ValueError("scores 长度必须等于节点数")

    N = S.shape[0]
    B = float(dep_cfg.budget)
    nc = float(dep_cfg.node_cost)
    ec = float(dep_cfg.edge_cost)
    if B <= 0 or nc <= 0 or ec < 0:
        raise ValueError("budget/node_cost 必须为正，edge_cost 必须非负")

    max_nodes = int(getattr(dep_cfg, "top_k_nodes", N) or N)
    max_edges = getattr(dep_cfg, "max_edges", None)
    if max_edges is not None:
        max_edges = int(max_edges)

    ii, jj = np.nonzero(S)
    candidates = []
    for i, j in zip(ii.tolist(), jj.tolist()):
        if i == j:
            continue
        s = float(S[i, j])
        if s <= 0:
            continue
        tie = float(score_vec[i] + score_vec[j])
        candidates.append((s, tie, int(i), int(j)))
    candidates.sort(key=lambda x: (-x[0], -x[1], x[2], x[3]))

    node_set, chosen_nodes, chosen_edges = set(), [], []
    cost = 0.0
    for s, _tie, i, j in candidates:
        if max_edges is not None and len(chosen_edges) >= max_edges:
            break
        new_nodes = [k for k in (i, j) if k not in node_set]
        if len(node_set) + len(new_nodes) > max_nodes:
            continue
        increment = ec + nc * len(new_nodes)
        if cost + increment > B + EPS:
            continue
        for k in new_nodes:
            node_set.add(k)
            chosen_nodes.append(k)
        chosen_edges.append((i, j, s))
        cost += increment

    A_dep = np.zeros((N, N), dtype=float)
    A_dep_weighted = np.zeros((N, N), dtype=float)
    for i, j, s in chosen_edges:
        A_dep[i, j] = 1.0
        A_dep_weighted[i, j] = s

    j_intent = _binary_jaccard(A_dep, A_I)
    j_persist = _binary_jaccard(A_dep, A_P)
    u_track = 0.5 * (j_intent + j_persist)
    consensus_hits = 0
    if A_CI is not None:
        consensus_hits = int(
            np.logical_and(A_dep > 0, np.asarray(A_CI) > 0).sum())

    return {
        "nodes": sorted(chosen_nodes),
        "edges": chosen_edges,
        "A_dep": A_dep,
        "A_dep_weighted": A_dep_weighted,
        "cost": float(cost),
        "budget": B,
        "budget_ratio": float(cost / B),
        "u_track": float(u_track),
        "jaccard_intent": float(j_intent),
        "jaccard_persist": float(j_persist),
        "consensus_hits": consensus_hits,
        "deployed_score": float(A_dep_weighted.sum()),
        "num_nodes": len(chosen_nodes),
        "num_edges": len(chosen_edges),
    }
