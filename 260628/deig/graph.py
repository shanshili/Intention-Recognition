"""
Candidate directed-graph construction (Section 4 of the method).

Builds G0 = (V, E0) using bidirectional KNN over sensor coordinates, then
attaches static edge features e_ij from geometry + train-set correlation and
lag-correlation statistics. The final intent graph is a dynamic activated
subgraph of G0.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _norm_coords(coords: np.ndarray):
    cmin = coords.min(axis=0)
    cmax = coords.max(axis=0)
    return (coords - cmin) / (cmax - cmin + 1e-9)


def lag_correlation(Z_train: np.ndarray, i: int, j: int, tau_max: int):
    """max_tau Corr(z_i^{t-tau}, z_j^t) and argmax tau, over train window."""
    best, best_tau = -2.0, 1
    zj = Z_train[j]
    for tau in range(1, tau_max + 1):
        a = Z_train[i, : -tau] if tau > 0 else Z_train[i]
        b = zj[tau:]
        if a.std() < 1e-8 or b.std() < 1e-8:
            continue
        c = np.corrcoef(a, b)[0, 1]
        if c > best:
            best, best_tau = c, tau
    return best, best_tau


def build_candidate_graph(coords: np.ndarray, Z: np.ndarray, train_end: int,
                          k: int = 4, tau_max: int = 4):
    """Return edges, edge features, mask matrix and per-edge metadata."""
    N = coords.shape[0]
    cn = _norm_coords(coords)
    Z_train = Z[:, :train_end]
    # 确保 Z_train 的维度正确
    assert Z_train.shape[0] == N, f"Expected {N} sensors, got {Z_train.shape[0]}"

    # Pearson correlation on the training window
    rho = np.corrcoef(Z_train)
    rho = np.nan_to_num(rho)
    # 检查 rho 的维度
    assert rho.shape == (N, N), f"rho shape should be ({N}, {N}), got {rho.shape}"

    nn = NearestNeighbors(n_neighbors=min(k + 1, N)).fit(cn)
    _, knn_idx = nn.kneighbors(cn)

    edge_set = set()
    for i in range(N):
        for j in knn_idx[i, 1:]:        # skip self
            edge_set.add((i, int(j)))
            edge_set.add((int(j), i))   # keep both directions (Sec 4.4)

    edges = sorted(edge_set)
    E = len(edges)
    sigma_d = np.median([np.linalg.norm(cn[i] - cn[j]) for i, j in edges]) + 1e-9

    # cache lag-correlations both directions
    lag_cache = {}

    def get_lag(i, j):
        if (i, j) not in lag_cache:
            lag_cache[(i, j)] = lag_correlation(Z_train, i, j, tau_max)
        return lag_cache[(i, j)]

    feats = np.zeros((E, 8))
    meta = []
    for e, (i, j) in enumerate(edges):
        d = np.linalg.norm(cn[i] - cn[j])
        u = cn[j] - cn[i]
        theta = np.arctan2(u[1], u[0])
        rho_ij = rho[i, j]
        rlag_ij, tau_star = get_lag(i, j)
        rlag_ji, _ = get_lag(j, i)
        feats[e] = [
            d,
            np.exp(-d / sigma_d),
            np.cos(theta),
            np.sin(theta),
            rho_ij,
            rlag_ij,
            tau_star / tau_max,
            rlag_ij - rlag_ji,
        ]
        meta.append({"i": i, "j": j, "d": d, "theta": theta,
                     "tau_star": tau_star, "rlag": rlag_ij})

    edges = np.array(edges, dtype=np.int64)        # (E, 2)
    src = edges[:, 0].copy()
    dst = edges[:, 1].copy()

    mask = np.zeros((N, N))
    mask[src, dst] = 1.0

    return {
        "edges": edges, "src": src, "dst": dst, "E": E,
        "edge_feats": feats, "mask": mask, "meta": meta,
        "coords_norm": cn, "sigma_d": sigma_d, "rho": rho,
    }
