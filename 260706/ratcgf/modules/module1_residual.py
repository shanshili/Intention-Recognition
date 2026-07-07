# -*- coding: utf-8 -*-
r"""Modulo uno: construccion de vistas residuales causal-intencion.

Dadas las adyacencias normalizadas \tilde{A}_C y \tilde{A}_I^{t-r,t,t+r},
construye el conjunto de 7 vistas de despliegue mas la vista de desvanecimiento
A_E^t, junto con el vector de estadisticas estructurales q_struct^t y las
caracteristicas por arista b_ij^t.
"""
from typing import Dict

import torch

from ..config import VIEW_NAMES

EPS = 1e-8


def _mask(A: torch.Tensor, delta: float, eta: float) -> torch.Tensor:
    r"""Mascara suave  M[i,j] = sigmoid((A[i,j]-delta)/eta)."""
    return torch.sigmoid((A - delta) / eta)


def build_views(A_C: torch.Tensor,
                A_I_prev: torch.Tensor,
                A_I_cur: torch.Tensor,
                A_I_next: torch.Tensor,
                delta_C: float, delta_I: float, eta: float) -> Dict:
    """Construye A^t = {A_C, A_I, A_CI, R_I, R_C, A_P, A_N} y A_E, q_struct.

    Todas las entradas son [N, N] en [0,1]. Devuelve un dict con:
        views    : dict nombre -> [N,N]
        decay    : A_E^t [N,N]
        q_struct : [9] estadisticas estructurales de grafo
    """
    M_C = _mask(A_C, delta_C, eta)
    M_I = _mask(A_I_cur, delta_I, eta)

    A_view_C = A_C
    A_view_I = A_I_cur
    A_view_CI = A_C * A_I_cur                      # consenso
    R_I = (1.0 - M_C) * A_I_cur                    # residuo de intencion
    R_C = A_C * (1.0 - M_I)                        # causal no activada (C \ I)
    A_P = torch.minimum(A_I_cur, A_I_prev)         # intencion persistente
    A_N = torch.clamp(A_I_cur - A_I_prev, min=0.0) # intencion nueva
    A_E = torch.clamp(A_I_prev - A_I_cur, min=0.0) # intencion desvanecida

    views = {
        "C": A_view_C, "I": A_view_I, "CI": A_view_CI,
        "R": R_I, "CmI": R_C, "P": A_P, "N": A_N,
    }

    # --- estadisticas estructurales de grafo q_struct^t (dim 9) ---
    def _jaccard(a, b):
        inter = torch.minimum(a, b).sum()
        union = torch.maximum(a, b).sum()
        return inter / (union + EPS)

    o_CI = _jaccard(A_C, A_I_cur)
    r_ImC = R_I.sum() / (A_I_cur.sum() + EPS)
    r_CmI = R_C.sum() / (A_C.sum() + EPS)
    p_prev = _jaccard(A_I_cur, A_I_prev)
    p_next = _jaccard(A_I_cur, A_I_next)
    d_prev = (A_I_cur - A_I_prev).abs().sum() / (A_I_prev.sum() + EPS)
    d_next = (A_I_next - A_I_cur).abs().sum() / (A_I_cur.sum() + EPS)
    n_I = A_N.sum() / (A_I_cur.sum() + EPS)
    e_I = A_E.sum() / (A_I_prev.sum() + EPS)

    q_struct = torch.stack([o_CI, r_ImC, r_CmI, p_prev, p_next,
                            d_prev, d_next, n_I, e_I])

    return {"views": views, "decay": A_E, "q_struct": q_struct}


def edge_feature_tensor(views: Dict, decay: torch.Tensor,
                        edge_index: torch.Tensor) -> torch.Tensor:
    r"""Caracteristica por arista b_ij^t = [A_C, A_I, A_CI, R_I, R_C, A_P, A_N, A_E].

    edge_index: [2, E] con filas (i, j). Devuelve [E, 8].
    """
    i, j = edge_index[0], edge_index[1]
    cols = [views[name][i, j] for name in VIEW_NAMES]
    cols.append(decay[i, j])
    return torch.stack(cols, dim=1)


def candidate_edge_index(views: Dict, decay: torch.Tensor,
                         thresh: float = 1e-6) -> torch.Tensor:
    """Aristas candidatas: union de soportes de todas las vistas (+decay).

    Devuelve edge_index [2, E]. Limita el gasto del gating por arista.
    """
    support = torch.zeros_like(views["C"])
    for name in VIEW_NAMES:
        support = support + views[name]
    support = support + decay
    idx = (support > thresh).nonzero(as_tuple=False)  # [E, 2] (i, j)
    if idx.numel() == 0:                              # fallback: diagonal off
        n = views["C"].shape[0]
        idx = torch.tensor([[0], [min(1, n - 1)]]).t()
    return idx.t().contiguous()
