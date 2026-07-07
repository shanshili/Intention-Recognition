# -*- coding: utf-8 -*-
r"""Visualizacion de los GRAFOS DE ENTRADA (no las vistas derivadas).

Dibuja:
  - Grafo causal G_C  (dirigido, ponderado y con SIGNO tomado de val_matrix),
    respetando su naturaleza de subgrafo con mask_threshold=0.5 y top_nodes.
  - Grafo de intencion G_I^t (dirigido y ponderado por la intensidad s_ij).

Uso:
    python -m ratcgf.viz_inputs                 # dibuja ambos (paso por defecto)
    python -m ratcgf.viz_inputs --intent-step 50
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx

from .config import Config
from .utils.io_utils import RunPaths
from .utils.data_loading import (
    load_all, load_causal, causal_to_adj, intent_to_adj_list)


# ---------------------------------------------------------------------------
# Construccion de la matriz causal CON SIGNO (para colorear + / -)
# ---------------------------------------------------------------------------
def causal_signed_matrix(causal: dict) -> np.ndarray:
    r"""[N,N] con el valor con signo de la arista causal i-->j.

    Para cada (i,j) se toma el lag tau con |val| maximo entre los que tienen
    graph[i,j,tau]=='-->', conservando el SIGNO del coeficiente.
    """
    graph = np.asarray(causal["graph"])
    val = np.asarray(causal["val_matrix"], dtype=float)
    if graph.ndim == 2:
        graph = graph[:, :, None]
        val = val[:, :, None]
    is_edge = (graph == "-->")
    masked = np.where(is_edge, val, 0.0)          # conserva signo
    absmax_lag = np.abs(masked).argmax(axis=2)    # tau de |val| maximo
    N = graph.shape[0]
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    W = masked[ii, jj, absmax_lag]
    W[~is_edge.any(axis=2)] = 0.0
    np.fill_diagonal(W, 0.0)
    return W


# ---------------------------------------------------------------------------
# Dibujo generico de grafo DIRIGIDO y PONDERADO sobre coordenadas
# ---------------------------------------------------------------------------
def draw_directed_weighted(ax, W: np.ndarray, coords: np.ndarray, title: str,
                           node_scores=None, highlight=None,
                           signed=False, pos_color="#1d4e89",
                           neg_color="#c1121f", uni_color="#2a9d8f"):
    """W[i,j]!=0 -> arista dirigida i->j. Grosor ~ |peso|; color por signo."""
    N = W.shape[0]
    pos = {i: (float(coords[i, 0]), float(coords[i, 1])) for i in range(N)}

    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    ii, jj = np.nonzero(W)
    for i, j in zip(ii.tolist(), jj.tolist()):
        G.add_edge(i, j, w=float(W[i, j]))

    # tamano de nodo: por node_scores si se da, si no por grado ponderado
    if node_scores is not None:
        ns = np.asarray(node_scores, dtype=float)
    else:
        ns = np.abs(W).sum(1) + np.abs(W).sum(0)
    rng = ns.max() - ns.min()
    sizes = 25 + 130 * (ns - ns.min()) / (rng + 1e-8)

    # aristas
    if G.number_of_edges() > 0:
        w = np.array([abs(d["w"]) for *_, d in G.edges(data=True)])
        wmax = w.max() + 1e-8
        widths = 0.4 + 2.8 * w / wmax
        if signed:
            colors = [neg_color if d["w"] < 0 else pos_color
                      for *_, d in G.edges(data=True)]
        else:
            colors = uni_color
        nx.draw_networkx_edges(
            G, pos, ax=ax, width=widths, edge_color=colors,
            arrows=True, arrowstyle="-|>", arrowsize=9,
            connectionstyle="arc3,rad=0.08", alpha=0.6, node_size=sizes)

    # nodos (resalta highlight si se da)
    if highlight is not None:
        hset = set(int(h) for h in highlight)
        ncolor = ["#ffb703" if i in hset else "#8d99ae" for i in range(N)]
    else:
        ncolor = "#e63946"
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=sizes, node_color=ncolor,
        edgecolors="white", linewidths=0.5)

    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    lat0 = float(np.mean(coords[:, 1]))
    ax.set_aspect(1.0 / max(np.cos(np.deg2rad(lat0)), 1e-3), adjustable="box")
    ax.margins(0.06)



# ---------------------------------------------------------------------------
# Figuras concretas
# ---------------------------------------------------------------------------
def plot_causal_graph(causal: dict, coords: np.ndarray, paths: RunPaths,
                      base="input_causal_graph"):
    """Grafo causal dirigido, ponderado y con signo; resalta top_nodes."""
    W = causal_signed_matrix(causal)
    meta = causal.get("meta", {})
    thr = meta.get("mask_threshold", 0.5)
    top_nodes = meta.get("top_nodes")
    node_scores = np.asarray(meta["node_scores"], dtype=float) \
        if meta.get("node_scores") is not None else None

    n_edges = int((W != 0).sum())
    n_pos = int((W > 0).sum()); n_neg = int((W < 0).sum())

    fig, ax = plt.subplots(figsize=(8, 7.5))
    draw_directed_weighted(
        ax, W, coords,
        title=(f"Input causal graph $G_C$  (directed, signed weights)\n"
               f"mask_threshold={thr}  |E|={n_edges}  (+{n_pos} / -{n_neg})"
               + ("  gold=top_nodes" if top_nodes is not None else "")),
        node_scores=node_scores, highlight=top_nodes, signed=True)

    legend = [
        Line2D([0], [0], color="#1d4e89", lw=3, label="positive effect (+)"),
        Line2D([0], [0], color="#c1121f", lw=3, label="negative effect (-)"),
    ]
    if top_nodes is not None:
        legend.append(Line2D([0], [0], marker="o", color="w",
                             markerfacecolor="#ffb703", markersize=9,
                             label="top_nodes (core)"))
    ax.legend(handles=legend, loc="upper right", fontsize=8)
    return paths.save_figure(fig, paths.figures, base)


def plot_intent_graph(A_I: np.ndarray, coords: np.ndarray, paths: RunPaths,
                      step: int, delta=None, base="input_intent_graph"):
    """Grafo de intencion dirigido y ponderado por intensidad s_ij."""
    n_edges = int((A_I > 0).sum())
    title = (f"Input intent graph $G_I^t$  (directed, weighted)\n"
             f"t = step {step}   |E|={n_edges}"
             + (f"   $\\delta$={delta:.3f}" if delta is not None else ""))
    fig, ax = plt.subplots(figsize=(8, 7.5))
    draw_directed_weighted(ax, A_I, coords, title=title, signed=False,
                           uni_color="#2a9d8f")
    legend = [Line2D([0], [0], color="#2a9d8f", lw=3,
                     label="intent strength $s_{ij}$ (width $\\propto s$)")]
    ax.legend(handles=legend, loc="upper right", fontsize=8)
    return paths.save_figure(fig, paths.figures, base)


def plot_side_by_side(causal: dict, A_I: np.ndarray, coords: np.ndarray,
                      paths: RunPaths, step: int, delta=None,
                      base="input_graphs_side_by_side"):
    """Panel doble: causal (izq) e intencion (der) para comparacion."""
    W = causal_signed_matrix(causal)
    meta = causal.get("meta", {})
    top_nodes = meta.get("top_nodes")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    draw_directed_weighted(
        axes[0], W, coords,
        title=f"Causal $G_C$  (mask={meta.get('mask_threshold', 0.5)})",
        highlight=top_nodes, signed=True)
    draw_directed_weighted(
        axes[1], A_I, coords,
        title=f"Intent $G_I^t$  (t=step {step})", signed=False,
        uni_color="#2a9d8f")
    fig.tight_layout()
    return paths.save_figure(fig, paths.figures, base)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run(cfg: Config, intent_step: int):
    paths = RunPaths(cfg.out_root)
    print(f"[viz-inputs] salida={paths.root}")

    bundle = load_all(cfg)
    N = bundle["N"]
    coords = bundle["data"].coords
    A_I_list = bundle["A_I_list"]
    delta_list = bundle.get("delta_list")

    # recarga el dict causal completo (para signo + meta); si es sintetico,
    # reconstruye un dict minimo desde load_all no lo guarda -> usa fichero.
    import os
    if os.path.exists(cfg.causal_path()):
        causal = load_causal(cfg.causal_path())
    else:
        # sintetico: envuelve la adyacencia en un dict compatible
        A = bundle["A_C"]
        g = np.where(A > 0, "-->", "").astype(object)[:, :, None]
        v = A[:, :, None]
        causal = {"graph": g, "val_matrix": v, "meta": bundle.get("causal_meta", {})}

    step = max(0, min(intent_step, len(A_I_list) - 1))
    A_I = A_I_list[step]
    delta = float(delta_list[step]) if delta_list is not None and \
        len(delta_list) > step else None

    p1 = plot_causal_graph(causal, coords, paths)
    p2 = plot_intent_graph(A_I, coords, paths, step=step, delta=delta)
    p3 = plot_side_by_side(causal, A_I, coords, paths, step=step, delta=delta)
    print(f"[viz-inputs] causal   -> {p1.get('png')}")
    print(f"[viz-inputs] intent   -> {p2.get('png')}")
    print(f"[viz-inputs] combinado-> {p3.get('png')}")
    print(f"[viz-inputs] formatos: svg/png/pdf/eps en {paths.figures}")
    return paths.root


def build_argparser():
    p = argparse.ArgumentParser(description="Dibuja los grafos de entrada")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--causal-pkl", default=None)
    p.add_argument("--intent-npz", default=None)
    p.add_argument("--coord-csv", default=None)
    p.add_argument("--intent-step", type=int, default=0,
                   help="indice de paso t del grafo de intencion a dibujar")
    return p


def main():
    args = build_argparser().parse_args()
    cfg = Config()
    if args.data_dir: cfg.data.data_dir = args.data_dir.strip()
    if args.causal_pkl: cfg.data.causal_pkl = args.causal_pkl.strip()
    if args.intent_npz: cfg.data.intent_npz = args.intent_npz.strip()
    if args.coord_csv: cfg.data.coord_csv = args.coord_csv.strip()
    run(cfg, args.intent_step)


if __name__ == "__main__":
    # arranque para ejecucion directa: python viz_inputs.py
    if __package__ in (None, ""):
        import os as _os, sys as _sys
        _here = _os.path.dirname(_os.path.abspath(__file__))
        _sys.path.insert(0, _os.path.dirname(_here))
        __package__ = _os.path.basename(_here)
    main()
