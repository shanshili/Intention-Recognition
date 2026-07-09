"""
views.py  --  模块 1：残差感知因果意图多视图构建
===========================================================================

给定归一化的因果邻接矩阵 \tilde A_C 和每步归一化的意图邻接矩阵
\tilde A_I^t，此模块为时间步 t 构建候选视图集

    M = { C, I, CI, R (=R_I), CmI (=R_C), P, N }

以及消退视图 A_E（仅用作门控特征）、全局结构统计向量 q_struct^t (9维)、
逐节点局部统计 (5维) 和逐边特征栈 b_ij (8维)。

此处的所有量相对于可学习参数都是*常数*；它们作为固定张量进入自动求导图。
"""

from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0.0)


def jaccard(A, B, eps=1e-9):
    return np.minimum(A, B).sum() / (np.maximum(A, B).sum() + eps)


class ViewBuilder:
    """构建时间步 t 的候选视图集"""

    def __init__(self, cfg, A_C_norm, A_I_norm_stack):
        self.cfg = cfg
        self.A_C = A_C_norm                          # [N, N]  \tilde A_C
        self.A_I = A_I_norm_stack                    # [n_show, N, N]  \tilde A_I^t
        self.n_show = A_I_norm_stack.shape[0]
        # causal mask is time-invariant
        self.M_C = sigmoid((self.A_C - cfg.delta_C) / cfg.eta)

    # ------------------------------------------------------------------ #
    def _AI(self, t):
        t = int(np.clip(t, 0, self.n_show - 1))
        return self.A_I[t]

    def build(self, t):
        """构建时间步 t 的候选视图集"""
        cfg = self.cfg
        AC = self.A_C
        AIt = self._AI(t)
        AI_prev = self._AI(t - cfg.r)
        AI_next = self._AI(t + cfg.r)

        M_C = self.M_C
        M_I = sigmoid((AIt - cfg.delta_I) / cfg.eta)

        views = {
            "C": AC.copy(),                          # A_C^t
            "I": AIt.copy(),                         # A_I^t
            "CI": AC * AIt,                          # consensus
            "R": (1.0 - M_C) * AIt,                  # intent residual R_I
            "CmI": AC * (1.0 - M_I),                 # causal-inactive R_C
            "P": np.minimum(AIt, AI_prev),           # persistence A_P
            "N": relu(AIt - AI_prev),                # new A_N
        }
        A_E = relu(AI_prev - AIt)                    # fade A_E (gating feature only)

        stats = self._global_stats(AC, AIt, AI_prev, AI_next, views, A_E)
        node_stats = self._node_stats(AC, AIt, AI_prev, views)

        return {
            "views": views,
            "A_E": A_E,
            "q_struct": stats,                       # [9]
            "q_node": node_stats,                    # [N, 5]
            "M_C": M_C, "M_I": M_I,
        }

    # ------------------------------------------------------------------ #
    def _global_stats(self, AC, AIt, AIp, AIn, views, A_E):
        eps = 1e-9
        o_CI = np.minimum(AC, AIt).sum() / (np.maximum(AC, AIt).sum() + eps)
        r_ImC = views["R"].sum() / (AIt.sum() + eps)
        r_CmI = views["CmI"].sum() / (AC.sum() + eps)
        p_minus = jaccard(AIt, AIp)
        p_plus = jaccard(AIt, AIn)
        d_minus = np.abs(AIt - AIp).sum() / (np.abs(AIp).sum() + eps)
        d_plus = np.abs(AIn - AIt).sum() / (np.abs(AIt).sum() + eps)
        n_I = views["N"].sum() / (AIt.sum() + eps)
        e_I = A_E.sum() / (np.abs(AIp).sum() + eps)
        return np.array([o_CI, r_ImC, r_CmI, p_minus, p_plus,
                         d_minus, d_plus, n_I, e_I], dtype=float)

    # ------------------------------------------------------------------ #
    def _node_stats(self, AC, AIt, AIp, views):
        """每个节点的自我边上的局部结构统计。"""
        eps = 1e-9
        N = AC.shape[0]

        def node_ego(M):                             # in + out incident weight per node
            return M.sum(axis=1) + M.sum(axis=0)

        ego_min = node_ego(np.minimum(AC, AIt))
        ego_max = node_ego(np.maximum(AC, AIt)) + eps
        o_CI_i = ego_min / ego_max

        ego_R = node_ego(views["R"])
        ego_I = node_ego(AIt) + eps
        r_ImC_i = ego_R / ego_I

        ego_CmI = node_ego(views["CmI"])
        ego_C = node_ego(AC) + eps
        r_CmI_i = ego_CmI / ego_C

        ego_P = node_ego(np.minimum(AIt, AIp))
        ego_U = node_ego(np.maximum(AIt, AIp)) + eps
        p_minus_i = ego_P / ego_U

        ego_N = node_ego(views["N"])
        n_I_i = ego_N / ego_I

        return np.stack([o_CI_i, r_ImC_i, r_CmI_i, p_minus_i, n_I_i], axis=1)  # [N,5]


def edge_features(view_dict, A_E, i, j):
    """为单条边组合 8 维边特征 b_ij。"""
    v = view_dict
    return np.array([
        v["C"][i, j], v["I"][i, j], v["CI"][i, j], v["R"][i, j],
        v["CmI"][i, j], v["P"][i, j], v["N"][i, j], A_E[i, j],
    ], dtype=float)


def candidate_edges(view_dict, max_edges=None):
    """
    存在于任何部署视图中的边并集 -> 候选部署边。
    返回形状为 [E, 2] 的整型数组，其中 i = 源行, j = 目标列。
    """
    support = np.zeros_like(view_dict["C"])
    for key in ("C", "I", "CI", "R", "CmI", "P", "N"):
        support = support + view_dict[key]
    ii, jj = np.where(support > 1e-6)
    edges = np.stack([ii, jj], axis=1)
    if max_edges is not None and len(edges) > max_edges:
        strength = support[ii, jj]
        keep = np.argsort(-strength)[:max_edges]
        edges = edges[keep]
    return edges.astype(int)
