# -*- coding: utf-8 -*-
r"""Modulo tres: gating dinamico de dominancia de grafo.

Combina las estadisticas estructurales y las representaciones espacio-temporales
multivista para producir:
  pi^t      -> gating a nivel de grafo   [M]
  gamma_i^t -> gating a nivel de nodo     [N, M]
  omega_ij  -> gating a nivel de arista   [E, M]
  S_D^t     -> matriz de puntuacion de aristas fusionada
  h_{i,D}^t -> representacion de nodo fusionada
"""
from typing import Dict, List

import torch
import torch.nn as nn

EPS = 1e-8


def _mlp(in_dim: int, out_dim: int, hidden: int = 64, dropout: float = 0.1):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


def node_local_stats(views: Dict[str, torch.Tensor]) -> torch.Tensor:
    r"""q_{i,struct}^t = [o_CI,i, r_{I\C},i, r_{C\I},i, p_I,i^-, n_I,i]  -> [N,5]."""
    A_C, A_I = views["C"], views["I"]
    R_I, R_C = views["R"], views["CmI"]
    A_P, A_N = views["P"], views["N"]

    def incident(A):                       # grado in+out por nodo
        return A.sum(dim=1) + A.sum(dim=0)

    c = incident(A_C)
    ii = incident(A_I)
    inter = incident(torch.minimum(A_C, A_I))
    union = incident(torch.maximum(A_C, A_I))
    o_CI = inter / (union + EPS)
    r_ImC = incident(R_I) / (ii + EPS)
    r_CmI = incident(R_C) / (c + EPS)
    p_prev = incident(A_P) / (ii + EPS)
    n_I = incident(A_N) / (ii + EPS)
    return torch.stack([o_CI, r_ImC, r_CmI, p_prev, n_I], dim=1)


class DominanceGating(nn.Module):
    def __init__(self, cfg, view_names: List[str]):
        super().__init__()
        self.view_names = view_names
        self.M = len(view_names)
        m = cfg.model
        d_h, d_g = m.tcn_hidden, m.graph_emb

        # gating a nivel de grafo: [q_struct(9) || concat g_m (M*d_g)]
        g_in = 9 + self.M * d_g
        self.mlp_g = _mlp(g_in, self.M, dropout=m.dropout)

        # gating a nivel de nodo: [q_i(5) || h_i_all(M*d_h) || pi(M)]
        n_in = 5 + self.M * d_h + self.M
        self.mlp_node = _mlp(n_in, self.M, dropout=m.dropout)

        # gating a nivel de arista:
        # [b_ij(8) || gamma_i(M) || gamma_j(M) || h_iD(d_h) || h_jD(d_h) || pi(M)]
        e_in = 8 + 2 * self.M + 2 * d_h + self.M
        self.mlp_edge = _mlp(e_in, self.M, dropout=m.dropout)

    def forward(self, q_struct, H, g, views, decay, edge_index, b_ij):
        N = views["C"].shape[0]
        names = self.view_names

        # --- gating de grafo pi^t ---
        g_all = torch.cat([g[n] for n in names], dim=0)          # [M*d_g]
        x_g = torch.cat([q_struct, g_all], dim=0)
        pi = torch.softmax(self.mlp_g(x_g), dim=0)               # [M]

        # --- gating de nodo gamma_i^t ---
        h_all = torch.cat([H[n] for n in names], dim=1)          # [N, M*d_h]
        q_i = node_local_stats(views)                            # [N, 5]
        pi_row = pi.unsqueeze(0).expand(N, -1)                   # [N, M]
        x_node = torch.cat([q_i, h_all, pi_row], dim=1)
        gamma = torch.softmax(self.mlp_node(x_node), dim=1)      # [N, M]

        # --- representacion de nodo fusionada h_{i,D}^t ---
        H_stack = torch.stack([H[n] for n in names], dim=0)      # [M, N, d_h]
        h_D = (pi.view(-1, 1, 1) * H_stack).sum(dim=0)           # [N, d_h]

        # --- gating de arista omega_ij^t ---
        i_idx, j_idx = edge_index[0], edge_index[1]
        E = i_idx.shape[0]
        gi = gamma[i_idx]                                        # [E, M]
        gj = gamma[j_idx]
        hi = h_D[i_idx]                                          # [E, d_h]
        hj = h_D[j_idx]
        pi_e = pi.unsqueeze(0).expand(E, -1)
        x_edge = torch.cat([b_ij, gi, gj, hi, hj, pi_e], dim=1)
        omega = torch.softmax(self.mlp_edge(x_edge), dim=1)      # [E, M]

        # --- gating conjunto global-local alpha_ijm ---
        num = pi_e * omega                                       # [E, M]
        alpha = num / (num.sum(dim=1, keepdim=True) + EPS)       # [E, M]

        # A_m^t[i,j] por arista y vista -> [E, M]
        A_edge = torch.stack([views[n][i_idx, j_idx] for n in names], dim=1)
        s_edge = (alpha * A_edge).sum(dim=1)                     # [E]

        # matriz densa S_D^t
        S_D = torch.zeros(N, N, device=views["C"].device)
        S_D[i_idx, j_idx] = s_edge

        return {
            "pi": pi, "gamma": gamma, "omega": omega, "alpha": alpha,
            "S_D": S_D, "s_edge": s_edge, "h_D": h_D,
            "edge_index": edge_index,
        }
