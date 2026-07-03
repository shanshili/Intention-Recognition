"""
Per-window input assembly for the *trainable* GAT-TCN encoder (Sections 5-7).

Key change vs. the previous version
-----------------------------------
Previously this file computed a FIXED spatial-smoothing ("GAT role") and FIXED
temporal statistics ("TCN role") and handed the model a finished vector z_i^t.
That meant GAT-TCN was never actually trained -- only the edge head / predictor
were. We now hand the model the RAW window sequence

    F_seq[tau] = f_i^tau  in R^{F0}            for tau = t-W+1 .. t

and let the trainable GAT-TCN encoder (gat_tcn.py) learn the spatial attention
and temporal convolutions end-to-end from the prediction loss.

We still precompute the *explicit* dynamic edge features d_ij^t (Sec 7.1):
those are genuine hand-crafted edge descriptors that feed the edge-intent head
as side information, not a substitute for the learned encoder.
"""

from __future__ import annotations

import numpy as np


def _windowed_lag_corr(a: np.ndarray, b: np.ndarray, tau_max: int = 3):
    """max over small lags of Corr(a[:-tau], b[tau:]) within a window."""
    best = 0.0
    for tau in range(1, tau_max + 1):
        x = a[:-tau]
        y = b[tau:]
        if x.size < 4 or x.std() < 1e-8 or y.std() < 1e-8:
            continue
        c = np.corrcoef(x, y)[0, 1]
        if c > best:
            best = c
    return best


def window_sequence(pre: dict, t: int, W: int):
    """Raw node-feature sequence for the window ending at t (inclusive).

    Returns (W, N, F0) where F0 = 7 is the per-node feature
    f_i^tau = [z, dz, r, hsin, hcos, wsin, wcos] built in data.Preprocessor.
    """
    return pre["feats"][t - W + 1:t + 1]                # (W, N, F0)


def dynamic_edge_features(pre: dict, cand: dict, t: int, W: int):
    """Explicit dynamic edge features d_ij^t (Sec 7.1)."""
    R = pre["R"]; dZ = pre["dZ"]
    sl = slice(t - W + 1, t + 1)
    r = R[:, sl]
    A = np.abs(r).mean(axis=1)                  # node anomaly strength A_i
    G = dZ[:, sl].mean(axis=1)                  # node trend G_i

    src, dst = cand["src"], cand["dst"]
    E = cand["E"]
    Dij = np.zeros(E)
    Dji = np.zeros(E)
    for e in range(E):
        i, j = src[e], dst[e]
        Dij[e] = _windowed_lag_corr(r[i], r[j])
        Dji[e] = _windowed_lag_corr(r[j], r[i])
    d = np.stack([Dij, Dij - Dji, A[src], A[dst], G[src], G[dst]], axis=1)
    return d                                     # (E, 6)


def build_window_inputs(pre: dict, cand: dict, t: int, W: int):
    """Assemble inputs for one window.

    Returns dict with:
        F_seq      (W, N, F0)   raw node feature sequence for GAT-TCN encoder
        edge_block (E, Fe)      static edge feats e_ij + dynamic edge feats d_ij^t
        z_last     (N,)         last standardised level (for residual prediction)
    """
    F_seq = window_sequence(pre, t, W)                 # (W, N, F0)
    d_dyn = dynamic_edge_features(pre, cand, t, W)      # (E, 6)
    edge_block = np.concatenate([cand["edge_feats"], d_dyn], axis=1)  # (E, Fe)
    z_last = pre["Z"][:, t]
    return {"F_seq": F_seq, "edge_block": edge_block, "z_last": z_last}
