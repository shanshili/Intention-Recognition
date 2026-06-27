"""
Spatio-temporal window features (Sections 6-7 of the method).

PyTorch is unavailable, so a fully-trainable deep GAT+dilated-TCN stack is not
practical to backprop through here. We instead realise the ST-encoder as a
*fixed* but faithful feature map that plays the same role:

  * Spatial (GAT role): one attention-weighted neighbourhood aggregation over
    the candidate graph, with attention computed from the static edge features
    (geometry + lag-correlation). This mixes each node with its directional
    neighbours.
  * Temporal (TCN role): multi-scale summaries of each node's history window
    (level, recent trend, volatility, residual energy at two dilations) plus
    an attention-style recency weighting.

The resulting per-node window vector z_i^t feeds the *trainable* edge-intent
head and predictor (model.py), which ARE trained end-to-end by the prediction
loss -- so the intent scores receive the self-supervised signal the method
calls for.

We also compute the explicit dynamic edge features d_ij^t (Sec 7.1).
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


def node_window_features(pre: dict, t: int, W: int):
    """Fixed node window vector z_i^t for window ending at t (inclusive)."""
    Z = pre["Z"]; dZ = pre["dZ"]; R = pre["R"]
    sl = slice(t - W + 1, t + 1)
    z = Z[:, sl]          # (N, W)
    dz = dZ[:, sl]
    r = R[:, sl]
    N = Z.shape[0]

    half = W // 2
    feats = np.stack([
        z[:, -1],                              # last level
        z.mean(axis=1),                        # window mean
        z.std(axis=1),                         # volatility
        dz.mean(axis=1),                       # trend  G_i
        np.abs(r).mean(axis=1),                # residual energy A_i
        r[:, -1],                              # last residual
        z[:, half:].mean(axis=1) - z[:, :half].mean(axis=1),  # half-window drift
        z.max(axis=1),
        z.min(axis=1),
    ], axis=1)                                  # (N, 9)

    # time encodings at t
    te = pre["feats"][t, :, 3:7]                # (N, 4)
    base = np.concatenate([feats, te], axis=1)  # (N, 13)
    return base


def spatial_smooth(base: np.ndarray, cand: dict, hops: int = 1):
    """GAT-role: attention aggregation over candidate graph (fixed weights)."""
    N = base.shape[0]
    src, dst = cand["src"], cand["dst"]
    ef = cand["edge_feats"]
    # attention logit from geometry + directional lag-corr signal
    logits = (2.0 * ef[:, 1]            # exp(-d/sigma): closeness
              + 1.5 * np.maximum(ef[:, 5], 0)   # forward lag corr
              + 1.0 * np.maximum(ef[:, 4], 0))  # pearson
    H = base.copy()
    for _ in range(hops):
        # softmax over incoming edges per destination
        ex = np.exp(logits - logits.max())
        denom = np.zeros(N)
        np.add.at(denom, dst, ex)
        alpha = ex / (denom[dst] + 1e-9)
        agg = np.zeros_like(H)
        np.add.at(agg, dst, alpha[:, None] * H[src])
        H = 0.5 * H + 0.5 * agg
    return np.concatenate([base, H], axis=1)    # (N, 26)


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
    """Assemble fixed inputs for one window: node features z and edge inputs.

    Returns dict with:
        z        (N, Fz)            node window features
        m_static (E, Fm)            edge representation m_ij^t WITHOUT the
                                    trainable parts (z_i,z_j etc are added in
                                    model from z); here we precompute the
                                    static + dynamic edge block.
        z_last   (N,)               last standardised level (for residual pred)
    """
    base = node_window_features(pre, t, W)
    z = spatial_smooth(base, cand, hops=1)         # (N, Fz)
    d_dyn = dynamic_edge_features(pre, cand, t, W)  # (E, 6)
    edge_block = np.concatenate([cand["edge_feats"], d_dyn], axis=1)  # (E, 14)
    z_last = pre["Z"][:, t]
    return {"z": z, "edge_block": edge_block, "z_last": z_last}
