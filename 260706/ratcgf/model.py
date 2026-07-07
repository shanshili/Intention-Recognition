# -*- coding: utf-8 -*-
r"""Modelo RA-TCGF completo: encadena los cuatro modulos.

(G_C, G_I^{t-r}, G_I^t, G_I^{t+r}, X_{t-L+1:t})
   -> A^{t-r:t+r} (modulo 1)
   -> {H_m^t, g_m^t} (modulo 2)
   -> S_D^t (modulo 3)
   -> G_D^t (modulo 4)
"""
from typing import Dict

import torch
import torch.nn as nn

from .config import VIEW_NAMES
from .modules.module1_residual import (
    build_views, edge_feature_tensor, candidate_edge_index)
from .modules.module2_encoder import MultiViewEncoder
from .modules.module3_gating import DominanceGating
from .modules.module4_deploy import ScoreHead, node_scores


class RATCGF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.view_names = VIEW_NAMES
        self.encoder = MultiViewEncoder(cfg, self.view_names)
        self.gating = DominanceGating(cfg, self.view_names)
        d_h = cfg.model.tcn_hidden
        self.score_head = ScoreHead(d_h)          # termino MLP de Score_i^t
        self.predictor = ScoreHead(d_h)           # cabeza de prediccion (U_pred)

    def forward(self, sample: Dict) -> Dict:
        r"""sample contiene tensores ya en el dispositivo:
            Xseq [L,N,F], A_C, A_I_prev, A_I_cur, A_I_next [N,N].
        """
        m = self.cfg.model
        # --- Modulo 1: vistas residuales ---
        built = build_views(sample["A_C"], sample["A_I_prev"],
                            sample["A_I_cur"], sample["A_I_next"],
                            m.delta_C, m.delta_I, m.eta)
        views, decay, q_struct = built["views"], built["decay"], built["q_struct"]
        edge_index = candidate_edge_index(views, decay)
        b_ij = edge_feature_tensor(views, decay, edge_index)

        # --- Modulo 2: codificacion espacio-temporal ---
        H, g = self.encoder(sample["Xseq"], views)

        # --- Modulo 3: gating dinamico -> S_D^t ---
        gate = self.gating(q_struct, H, g, views, decay, edge_index, b_ij)

        # --- puntuacion de nodo y prediccion ---
        h_D = gate["h_D"]
        mlp_term = self.score_head(h_D)
        node_sc = node_scores(gate["S_D"], mlp_term)
        y_hat = self.predictor(h_D)               # prediccion por nodo [N]

        out = {
            "views": views, "decay": decay, "q_struct": q_struct,
            "S_D": gate["S_D"], "s_edge": gate["s_edge"],
            "edge_index": edge_index,
            "pi": gate["pi"], "gamma": gate["gamma"], "omega": gate["omega"],
            "h_D": h_D, "node_scores": node_sc, "y_hat": y_hat,
        }
        return out
