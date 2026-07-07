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


def greedy_deploy(S_D: np.ndarray, scores: np.ndarray,
                  A_I: np.ndarray, A_P: np.ndarray,
                  dep_cfg) -> Dict:
    r"""Genera G_D^t con estrategia greedy bajo presupuesto B.

    Reparte el presupuesto: hasta node_budget_ratio*B para nodos (por Score_i),
    el resto para aristas (por S_D). Si no quedan aristas entre los nodos
    elegidos, se expande de forma edge-centrica para garantizar un subgrafo util.
    - q_ij<=p_i, q_ij<=p_j: una arista solo se anade si ambos extremos existen.
    """
    N = S_D.shape[0]
    B = dep_cfg.budget
    nc, ec = dep_cfg.node_cost, dep_cfg.edge_cost
    node_budget = dep_cfg.node_budget_ratio * B

    # --- 1. nodos por Score_i (con presupuesto reservado para aristas) ---
    order = np.argsort(-scores)
    chosen_nodes, cost = [], 0.0
    for n in order:
        if len(chosen_nodes) >= dep_cfg.top_k_nodes:
            break
        if cost + nc > node_budget:
            break
        chosen_nodes.append(int(n))
        cost += nc
    node_set = set(chosen_nodes)

    # --- 2. aristas entre nodos elegidos, por S_D, hasta agotar B ---
    cand = [(S_D[i, j], i, j)
            for i in chosen_nodes for j in chosen_nodes
            if i != j and S_D[i, j] > 0]
    cand.sort(reverse=True)
    chosen_edges = []
    for s, i, j in cand:
        if cost + ec > B:
            break
        chosen_edges.append((i, j, float(s)))
        cost += ec

    # --- 3. fallback edge-centrico: si no hay aristas, expandir con presupuesto ---
    if not chosen_edges:
        ii, jj = np.nonzero(S_D)
        all_edges = sorted(((S_D[i, j], int(i), int(j))
                            for i, j in zip(ii, jj)), reverse=True)
        for s, i, j in all_edges:
            need = [k for k in (i, j) if k not in node_set]
            add = len(need) * nc + ec
            if cost + add > B:
                continue
            for k in need:
                node_set.add(k)
                chosen_nodes.append(k)
            chosen_edges.append((i, j, float(s)))
            cost += add

    # --- adyacencia del subgrafo desplegado ---
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
