# -*- coding: utf-8 -*-
r"""Modulo cuatro: generacion y validacion del subgrafo de despliegue G_D^t.

A partir de S_D^t se calculan puntuaciones de nodo y se resuelve una version
greedy / top-K del problema de despliegue con restriccion de presupuesto.
"""
from typing import Dict

import numpy as np
import torch
import torch.nn as nn


class ScoreHead(nn.Module):
    r"""MLP_score(h_{i,D}^t) usado en Score_i^t y como cabeza de prediccion."""

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h_D: torch.Tensor) -> torch.Tensor:
        return self.net(h_D).squeeze(-1)         # [N]


def node_scores(S_D: torch.Tensor, mlp_term: torch.Tensor) -> torch.Tensor:
    r"""Score_i^t = sum_j S_D[i,j] + sum_j S_D[j,i] + MLP_score(h_{i,D})."""
    out_s = S_D.sum(dim=1)
    in_s = S_D.sum(dim=0)
    return out_s + in_s + mlp_term


# def greedy_deploy(S_D: np.ndarray, scores: np.ndarray,
#                   A_I: np.ndarray, A_P: np.ndarray,
#                   dep_cfg) -> Dict:
#     r"""Genera G_D^t con estrategia greedy bajo presupuesto B.
#
#     Reparte el presupuesto: hasta node_budget_ratio*B para nodos (por Score_i),
#     el resto para aristas (por S_D). Si no quedan aristas entre los nodos
#     elegidos, se expande de forma edge-centrica para garantizar un subgrafo util.
#     - q_ij<=p_i, q_ij<=p_j: una arista solo se anade si ambos extremos existen.
#     """
#     N = S_D.shape[0]
#     B = dep_cfg.budget
#     nc, ec = dep_cfg.node_cost, dep_cfg.edge_cost
#     node_budget = dep_cfg.node_budget_ratio * B
#
#     # --- 1. nodos por Score_i (con presupuesto reservado para aristas) ---
#     order = np.argsort(-scores)
#     chosen_nodes, cost = [], 0.0
#     for n in order:
#         if len(chosen_nodes) >= dep_cfg.top_k_nodes:
#             break
#         if cost + nc > node_budget:
#             break
#         chosen_nodes.append(int(n))
#         cost += nc
#     node_set = set(chosen_nodes)
#
#     # --- 2. aristas entre nodos elegidos, por S_D, hasta agotar B ---
#     cand = [(S_D[i, j], i, j)
#             for i in chosen_nodes for j in chosen_nodes
#             if i != j and S_D[i, j] > 0]
#     cand.sort(reverse=True)
#     chosen_edges = []
#     for s, i, j in cand:
#         if cost + ec > B:
#             break
#         chosen_edges.append((i, j, float(s)))
#         cost += ec
#
#     # --- 3. fallback edge-centrico: si no hay aristas, expandir con presupuesto ---
#     if not chosen_edges:
#         ii, jj = np.nonzero(S_D)
#         all_edges = sorted(((S_D[i, j], int(i), int(j))
#                             for i, j in zip(ii, jj)), reverse=True)
#         for s, i, j in all_edges:
#             need = [k for k in (i, j) if k not in node_set]
#             add = len(need) * nc + ec
#             if cost + add > B:
#                 continue
#             for k in need:
#                 node_set.add(k)
#                 chosen_nodes.append(k)
#             chosen_edges.append((i, j, float(s)))
#             cost += add
#
#     # --- adyacencia del subgrafo desplegado ---
#     A_dep = np.zeros((N, N), dtype=float)
#     for i, j, s in chosen_edges:
#         A_dep[i, j] = s
#
#     def _jaccard(a, b):
#         inter = np.minimum(a, b).sum()
#         union = np.maximum(a, b).sum()
#         return float(inter / (union + 1e-8))
#
#     A_dep_bin = (A_dep > 0).astype(float)
#     u_track = _jaccard(A_dep_bin, (A_I > 0).astype(float)) + \
#         _jaccard(A_dep_bin, (A_P > 0).astype(float))
#
#     return {
#         "nodes": chosen_nodes,
#         "edges": chosen_edges,
#         "A_dep": A_dep,
#         "cost": cost,
#         "u_track": u_track,
#         "num_nodes": len(chosen_nodes),
#         "num_edges": len(chosen_edges),
#     }
def greedy_deploy(S_D: np.ndarray, scores: np.ndarray,
                  A_I: np.ndarray, A_P: np.ndarray,
                  dep_cfg) -> Dict:
    r"""以边为中心的贪心部署 G_D^t（edge-first）。

    完全由融合边分数 S_D 驱动：按 S_D[i,j] 降序选边，节点仅在其
    作为被选边的端点时才被部署（q_ij<=p_i, q_ij<=p_j）。
    scores 仅用于同分边的次序（tie-break），不单独决定节点。
    """
    N = S_D.shape[0]
    B = dep_cfg.budget
    nc, ec = dep_cfg.node_cost, dep_cfg.edge_cost
    max_nodes = getattr(dep_cfg, "top_k_nodes", N)          # 可选：节点数上限
    max_edges = getattr(dep_cfg, "max_edges", None)         # 可选：边数上限

    # --- 1. 候选边：S_D>0，按分数降序（同分用端点 Score 之和 tie-break）---
    ii, jj = np.nonzero(S_D)
    cand = []
    for i, j in zip(ii.tolist(), jj.tolist()):
        if i == j:
            continue
        s = float(S_D[i, j])
        if s <= 0:
            continue
        tie = float(scores[i] + scores[j]) if scores is not None else 0.0
        cand.append((s, tie, i, j))
    cand.sort(key=lambda x: (-x[0], -x[1]))

    # --- 2. 边驱动贪心：接受一条边即把其新端点纳入部署 ---
    node_set, chosen_nodes, chosen_edges, cost = set(), [], [], 0.0
    for s, _tie, i, j in cand:
        if max_edges is not None and len(chosen_edges) >= max_edges:
            break
        new_nodes = [k for k in (i, j) if k not in node_set]
        if len(node_set) + len(new_nodes) > max_nodes:
            continue                                        # 超节点上限，跳过看更便宜的边
        inc = ec + nc * len(new_nodes)                      # 增量成本：边 + 新端点
        if cost + inc > B:
            continue                                        # 超预算，继续找端点已就位的边
        for k in new_nodes:
            node_set.add(k)
            chosen_nodes.append(k)
        chosen_edges.append((i, j, s))
        cost += inc

    # --- 3. 部署子图邻接 ---
    A_dep = np.zeros((N, N), dtype=float)
    for i, j, s in chosen_edges:
        A_dep[i, j] = s

    def _jaccard(a, b):
        inter = np.minimum(a, b).sum()
        union = np.maximum(a, b).sum()
        return float(inter / (union + 1e-8))

    A_dep_bin = (A_dep > 0).astype(float)
    u_track = _jaccard(A_dep_bin, (A_I > 0).astype(float)) + \
        _jaccard(A_dep_bin, (A_P > 0).astype(float))

    return {
        "nodes": chosen_nodes,
        "edges": chosen_edges,
        "A_dep": A_dep,
        "cost": cost,
        "u_track": u_track,
        "num_nodes": len(chosen_nodes),
        "num_edges": len(chosen_edges),
    }
