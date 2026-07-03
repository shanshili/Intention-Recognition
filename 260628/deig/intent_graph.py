"""
Dynamic intent graph construction + hotspot extraction (Sections 14-16).
"""

from __future__ import annotations

import numpy as np
import networkx as nx


def build_intent_graph(s_vec: np.ndarray, cand: dict, q: float = 0.90):
    """Threshold edge intent scores at quantile q to form G_I^t.

    Returns kept edge indices, their strengths, and an N x N adjacency.
    """
    src, dst = cand["src"], cand["dst"]
    N = cand["mask"].shape[0]
    if s_vec.size == 0:
        delta = 0.0
    else:
        delta = np.quantile(s_vec, q)
    keep = np.where(s_vec >= delta)[0]
    A = np.zeros((N, N))
    A[src[keep], dst[keep]] = s_vec[keep]
    return {"keep": keep, "delta": delta, "A": A,
            "edges": np.stack([src[keep], dst[keep]], axis=1),
            "strength": s_vec[keep]}


def node_intent_attributes(A: np.ndarray):
    I_out = A.sum(axis=1)
    I_in = A.sum(axis=0)
    return {"I_in": I_in, "I_out": I_out,
            "I_act": I_in + I_out,
            "I_src": I_out - I_in,
            "I_sink": I_in - I_out}


def extract_hotspots(graph: dict, min_size: int = 2):
    """Weakly-connected components of G_I^t as local hotspot intent subgraphs."""
    G = nx.DiGraph()
    for (i, j), w in zip(graph["edges"], graph["strength"]):
        G.add_edge(int(i), int(j), weight=float(w))
    comps = [c for c in nx.weakly_connected_components(G) if len(c) >= min_size]
    hotspots = []
    for c in comps:
        sub = G.subgraph(c)
        strength = sum(d["weight"] for _, _, d in sub.edges(data=True))
        hotspots.append({"nodes": sorted(c),
                         "n_edges": sub.number_of_edges(),
                         "strength": strength})
    hotspots.sort(key=lambda h: h["strength"], reverse=True)
    return hotspots
