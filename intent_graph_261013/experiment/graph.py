"""Candidate graph adapted from 260628/deig/graph.py (copied 2026-10-13).

The geometric KNN, eight static edge features, and sorted directed-pair order
are retained. Empty graphs, coincident coordinates, constant signals, and
short training prefixes are handled explicitly; features use training only.
"""

from __future__ import annotations

import numpy as np


def _norm_coords(coords: np.ndarray) -> np.ndarray:
    low = coords.min(axis=0)
    span = coords.max(axis=0) - low
    return (coords - low) / np.where(span > 0, span, 1.0)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.clip(np.dot(a, b) / norm, -1, 1)) if norm > 1e-12 else 0.0


def lag_correlation(Z_train: np.ndarray, i: int, j: int, tau_max: int) -> tuple[float, int]:
    """Best positive-lag correlation; unavailable/constant inputs return zero."""
    valid = []
    for tau in range(1, min(tau_max, Z_train.shape[1] - 2) + 1):
        a, b = Z_train[i, :-tau], Z_train[j, tau:]
        if a.std() > 1e-8 and b.std() > 1e-8:
            valid.append((_correlation(a, b), tau))
    return max(valid, key=lambda pair: pair[0]) if valid else (0.0, 0)


def build_candidate_graph(coords: np.ndarray, Z: np.ndarray, train_end: int,
                          k: int = 4, tau_max: int = 4) -> dict:
    coords = np.asarray(coords, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError("coords must be finite [N,2]")
    N = coords.shape[0]
    if N == 0 or Z.ndim != 2 or Z.shape[0] != N or not 1 <= train_end <= Z.shape[1]:
        raise ValueError("Invalid graph signal shape or training boundary")
    if k < 0 or tau_max < 1:
        raise ValueError("k must be nonnegative and tau_max positive")
    cn = _norm_coords(coords)
    Z_train = Z[:, :train_end]
    if not np.isfinite(Z_train).all():
        raise ValueError("Training graph signals must already be causally filled")
    # Equivalent Euclidean KNN; exclude self by identity, not by sorted position.
    # Stable sorting fixes ties for coincident coordinates.
    distances = np.linalg.norm(cn[:, None, :] - cn[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1, kind="stable")[:, :min(k, N - 1)]
    edge_set = {(i, int(j)) for i in range(N) for j in neighbors[i]}
    edge_set |= {(j, i) for i, j in edge_set.copy()}
    edges = np.asarray(sorted(edge_set), dtype=np.int64).reshape(-1, 2)
    src, dst = edges[:, 0].copy(), edges[:, 1].copy()
    E = len(edges)
    edge_distances = np.linalg.norm(cn[src] - cn[dst], axis=1)
    sigma_d = float(np.median(edge_distances)) + 1e-9 if E else 1.0
    rho = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i, N):
            rho[i, j] = rho[j, i] = _correlation(Z_train[i], Z_train[j])
    lag_cache = {}
    for i, j in edges:
        lag_cache[int(i), int(j)] = lag_correlation(Z_train, int(i), int(j), tau_max)
    feats = np.zeros((E, 8), dtype=np.float32)
    meta = []
    for e, (i, j) in enumerate(edges):
        d = float(edge_distances[e])
        direction = cn[j] - cn[i]
        theta = float(np.arctan2(direction[1], direction[0]))
        rlag, tau = lag_cache[int(i), int(j)]
        reverse, _ = lag_cache[int(j), int(i)]
        feats[e] = [d, np.exp(-d / sigma_d), np.cos(theta), np.sin(theta),
                    rho[i, j], rlag, tau / tau_max, rlag - reverse]
        meta.append({"i": int(i), "j": int(j), "d": d, "theta": theta,
                     "tau_star": int(tau), "rlag": float(rlag)})
    mask = np.zeros((N, N), dtype=np.float32)
    mask[src, dst] = 1.0
    degree = np.maximum(np.bincount(dst, minlength=N), 1).astype(np.float32)
    return {"edges": edges, "src": src, "dst": dst, "E": E,
            "edge_feats": feats, "degree": degree, "mask": mask,
            "meta": meta, "coords_norm": cn, "sigma_d": sigma_d, "rho": rho}
