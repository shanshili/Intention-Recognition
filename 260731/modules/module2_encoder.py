# -*- coding: utf-8 -*-
r"""Modulo dos: codificacion espacio-temporal multivista (GNN + TCN).

Para cada vista m se aplica una GNN por instante de la ventana temporal y
luego una TCN a nivel de nodo (-> H_m^t) y a nivel de grafo (-> g_m^t).
"""
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


# ---------------------------------------------------------------------------
# Capas GNN densas (N pequeno => adyacencia densa es suficiente)
# ---------------------------------------------------------------------------
class DenseGCN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        N = A.shape[0]
        Ah = A + torch.eye(N, device=A.device)
        deg = Ah.sum(dim=1)
        dinv = torch.pow(deg + EPS, -0.5)
        Anorm = dinv.unsqueeze(1) * Ah * dinv.unsqueeze(0)
        return self.lin(Anorm @ X)


class DenseSAGE(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        deg = A.sum(dim=1, keepdim=True) + EPS
        neigh = (A @ X) / deg
        return self.lin_self(X) + self.lin_neigh(neigh)


class DenseGAT(nn.Module):
    """GAT denso con peso de arista concatenado (A_m^tau[i,j])."""

    def __init__(self, in_dim: int, out_dim: int, alpha: float = 0.2):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(out_dim))
        self.att_dst = nn.Parameter(torch.empty(out_dim))
        self.att_edge = nn.Parameter(torch.empty(1))
        self.leaky = nn.LeakyReLU(alpha)
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.normal_(self.att_src, std=0.1)
        nn.init.normal_(self.att_dst, std=0.1)
        nn.init.normal_(self.att_edge, std=0.1)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        N = A.shape[0]
        Z = self.lin(X)                                 # [N, out]
        e_src = (Z * self.att_src).sum(-1, keepdim=True)   # [N,1]
        e_dst = (Z * self.att_dst).sum(-1, keepdim=True)   # [N,1]
        scores = e_src + e_dst.t() + self.att_edge * A     # [N,N]
        scores = self.leaky(scores)
        mask = (A > 0) | torch.eye(N, dtype=torch.bool, device=A.device)
        scores = scores.masked_fill(~mask, float("-inf"))
        att = torch.softmax(scores, dim=1)
        att = torch.nan_to_num(att, nan=0.0)
        return att @ Z


def make_gnn_layer(gnn_type: str, in_dim: int, out_dim: int) -> nn.Module:
    gnn_type = gnn_type.lower()
    if gnn_type == "gcn":
        return DenseGCN(in_dim, out_dim)
    if gnn_type == "sage":
        return DenseSAGE(in_dim, out_dim)
    return DenseGAT(in_dim, out_dim)


class GNNStack(nn.Module):
    def __init__(self, gnn_type: str, in_dim: int, hidden: int, layers: int,
                 dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        d = in_dim
        for _ in range(layers):
            self.layers.append(make_gnn_layer(gnn_type, d, hidden))
            d = hidden
        self.drop = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        h = X
        for i, layer in enumerate(self.layers):
            h = layer(h, A)
            if i < len(self.layers) - 1:
                h = self.drop(F.elu(h))
        return h                                        # [N, hidden]


# ---------------------------------------------------------------------------
# TCN (convoluciones causales dilatadas)
# ---------------------------------------------------------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, : -self.chomp] if self.chomp > 0 else x


class TCN(nn.Module):
    """Devuelve el estado final de la secuencia: [B, C_out]."""

    def __init__(self, in_dim: int, hidden: int, layers: int, kernel: int,
                 dropout: float = 0.1):
        super().__init__()
        blocks = []
        c_in = in_dim
        for i in range(layers):
            dil = 2 ** i
            pad = (kernel - 1) * dil
            blocks += [
                nn.Conv1d(c_in, hidden, kernel, padding=pad, dilation=dil),
                Chomp1d(pad),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            c_in = hidden
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, L] -> [B, hidden]
        y = self.net(x)
        return y[:, :, -1]


# ---------------------------------------------------------------------------
# Lectura por atencion (READOUT)
# ---------------------------------------------------------------------------
class AttentionReadout(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.w = nn.Linear(dim, 1, bias=False)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        # Z: [N, dim] -> u: [dim]
        beta = torch.softmax(self.w(torch.tanh(self.W(Z))), dim=0)  # [N,1]
        return (beta * Z).sum(dim=0)


# ---------------------------------------------------------------------------
# Codificador por vista
# ---------------------------------------------------------------------------
class ViewEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        self.gnn = GNNStack(m.gnn_type, cfg._in_dim, m.gnn_hidden,
                            m.gnn_layers, m.dropout)
        self.node_tcn = TCN(m.gnn_hidden, m.tcn_hidden, m.tcn_layers,
                            m.tcn_kernel, m.dropout)
        self.readout = AttentionReadout(m.gnn_hidden)
        self.graph_tcn = TCN(m.gnn_hidden, m.graph_emb, m.tcn_layers,
                             m.tcn_kernel, m.dropout)

    def forward(self, Xseq: torch.Tensor, A: torch.Tensor):
        """Xseq: [L, N, F], A: [N,N]. Devuelve H_m [N,d_h], g_m [d_g]."""
        L, N, _ = Xseq.shape
        Zs = [self.gnn(Xseq[t], A) for t in range(L)]        # cada [N, d]
        Zstack = torch.stack(Zs, dim=0)                       # [L, N, d]

        # TCN a nivel de nodo: nodos como batch -> [N, d, L]
        node_in = Zstack.permute(1, 2, 0)                     # [N, d, L]
        H_m = self.node_tcn(node_in)                          # [N, d_h]

        # READOUT por instante -> [L, d] -> TCN de grafo
        U = torch.stack([self.readout(Zs[t]) for t in range(L)], dim=0)  # [L,d]
        graph_in = U.t().unsqueeze(0)                         # [1, d, L]
        g_m = self.graph_tcn(graph_in).squeeze(0)             # [d_g]
        return H_m, g_m


class MultiViewEncoder(nn.Module):
    """Un ViewEncoder por vista de VIEW_NAMES."""

    def __init__(self, cfg, view_names: List[str]):
        super().__init__()
        self.view_names = view_names
        self.encoders = nn.ModuleDict(
            {name: ViewEncoder(cfg) for name in view_names})

    def forward(self, Xseq: torch.Tensor, view_adj: Dict[str, torch.Tensor]):
        H, g = {}, {}
        for name in self.view_names:
            H[name], g[name] = self.encoders[name](Xseq, view_adj[name])
        return H, g
