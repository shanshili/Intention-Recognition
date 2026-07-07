"""
deploy.py  --  Module 4: deployment sub-graph generation under a budget
======================================================================

Given the fused edge-score matrix S_D^t and node scores Score_i^t, a greedy /
top-K strategy selects nodes p_i and edges q_ij subject to

    sum_i c_i p_i + sum_ij c_ij q_ij  <=  B          (budget)
    q_ij <= p_i ,  q_ij <= p_j                        (edge needs both endpoints)

maximising the deployment utility (prediction + elastic + tracking) minus cost.
The greedy heuristic stands in for the exact integer program (spec section 5).
"""

from __future__ import annotations

import numpy as np


def jaccard(A, B, eps=1e-9):
    return np.minimum(A, B).sum() / (np.maximum(A, B).sum() + eps)


def greedy_deploy(node_scores, S_dense, view_dict, cfg):
    """
    Returns a dict describing G_D^t:
        nodes      : sorted list of deployed node ids
        edges      : list of (i, j) deployed edges
        cost       : total consumed budget
        util       : reported utility breakdown
    """
    N = cfg.N
    B = cfg.budget
    c_node, c_edge = cfg.node_cost, cfg.edge_cost

    # ---- node selection: greedily add highest-scoring nodes -------------- #
    order = np.argsort(-node_scores)
    selected = []
    cost = 0.0
    for i in order:
        if cost + c_node <= B * 0.7:                 # reserve part of budget for edges
            selected.append(int(i))
            cost += c_node
        if cost >= B * 0.7:
            break
    sel_set = set(selected)

    # ---- edge selection: rank candidate edges within selected nodes ------ #
    ii, jj = np.where(S_dense > 0)
    cand = [(int(a), int(b), float(S_dense[a, b])) for a, b in zip(ii, jj)]
    cand.sort(key=lambda x: -x[2])
    edges = []
    for i, j, s in cand:
        if i in sel_set and j in sel_set:
            if cost + c_edge <= B:
                edges.append((i, j))
                cost += c_edge
            if cost + c_edge > B:
                break

    # ---- utility reporting ---------------------------------------------- #
    A_dep = np.zeros((N, N))
    for i, j in edges:
        A_dep[i, j] = 1.0
    u_track = jaccard(A_dep, (view_dict["I"] > 0).astype(float)) + \
              jaccard(A_dep, (view_dict["P"] > 0).astype(float))
    u_consensus = A_dep[(view_dict["CI"] > 0)].sum()
    util = {
        "track": float(u_track),
        "consensus_hits": float(u_consensus),
        "deployed_score": float(sum(S_dense[i, j] for i, j in edges)),
    }

    return {
        "nodes": sorted(selected),
        "edges": edges,
        "cost": float(cost),
        "budget": float(B),
        "util": util,
        "A_dep": A_dep,
    }
