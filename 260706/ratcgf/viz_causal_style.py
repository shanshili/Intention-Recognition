# -*- coding: utf-8 -*-
r"""Estilo de visualizacion dirigida "PCMCI MultiDiGraph" para RA-TCGF.

Reproduce el aspecto de referencia del proyecto (pcmci_draw_MultiDiGraph2 /
visualize_mask_subgraph) SIN depender de esos modulos ni de cartopy:

  - Grafo dirigido MultiDiGraph con flechas.
  - Nodos: color = suma de |efecto| de aristas conectadas (YlGn), tamano
    proporcional a esa fuerza; SOLO se etiqueta el 30 % de nodos mas fuertes.
  - Aristas: color = efecto causal con signo (seismic, azul=- / rojo=+) para el
    grafo causal; para vistas/intencion (sin signo) se usa un mapa secuencial.
  - Auto-lazos (self-loops) dibujados para los nodos etiquetados.
  - Doble colorbar: fuerza de nodo (izq.) y efecto de arista (der.).

Convenciones de indice (¡importantes para la direccion de la flecha!):
  - Causal (.pkl PCMCI):   graph[target, source, tau] == '-->'  =>  arista source->target
                           (identico a build_multi_di_graph de referencia).
  - Intencion / vistas:    A[i, j] > 0                          =>  arista i->j
                           (identico a intent_to_adj_list de RA-TCGF).
"""
from typing import Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

_EPS = 1e-8


# --------------------------------------------------------------------------- #
#  Constructores de MultiDiGraph                                              #
# --------------------------------------------------------------------------- #
def build_causal_digraph(graph, val_matrix):
    r"""PCMCI [N,N,tau+1] -> MultiDiGraph (source->target, con signo y lag)."""
    graph = np.asarray(graph, dtype=object)
    val = np.asarray(val_matrix, dtype=float)
    if graph.ndim == 2:
        graph = graph[:, :, None]
        val = val[:, :, None]
    N, _, L = graph.shape
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(N))
    for tgt in range(N):
        for src in range(N):
            for tau in range(L):
                if graph[tgt, src, tau] == "-->":
                    ce = float(val[tgt, src, tau])
                    G.add_edge(int(src), int(tgt), key=int(tau),
                               lag=int(tau), causal_effect=ce, weight=abs(ce))
    return G


def build_weighted_digraph(A, self_loops=False):
    r"""Adyacencia ponderada [N,N] -> MultiDiGraph (A[i,j]>0 => i->j, sin signo)."""
    A = np.asarray(A, dtype=float)
    N = A.shape[0]
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(N))
    ii, jj = np.nonzero(A)
    for i, j in zip(ii.tolist(), jj.tolist()):
        if i == j and not self_loops:
            continue
        w = float(A[i, j])
        G.add_edge(i, j, key=0, weight=w, causal_effect=w)
    return G


# --------------------------------------------------------------------------- #
#  Dibujo (nucleo compartido)                                                  #
# --------------------------------------------------------------------------- #
def _node_strength(G, N):
    s = np.zeros(N, dtype=float)
    for u, v, _k, d in G.edges(keys=True, data=True):
        w = abs(float(d.get("causal_effect", d.get("weight", 0.0))))
        if u == v:
            s[u] += w
        else:
            s[u] += w
            s[v] += w
    return s


def _draw_reference_style(G, coords, ax, fig, signed,
                          edge_cmap, node_cmap=plt.cm.YlGn,
                          label_top_ratio=0.30, base_width=1.0,
                          width_scale=5.0, edge_label="Edge weight",
                          curved=True):
    N = G.number_of_nodes()
    pos = {i: (float(coords[i, 0]), float(coords[i, 1])) for i in range(N)}

    # --- nodos por fuerza conectada ---
    strength = _node_strength(G, N)
    vmax_n = strength.max() if strength.max() > 0 else 1.0
    sizes = 100 + (strength / vmax_n) * 300
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=strength,
                           cmap=node_cmap, edgecolors="white", linewidths=1.0,
                           alpha=0.9, ax=ax)

    # etiquetas solo para el top-30 %
    if N > 0:
        thr = np.quantile(strength, 1.0 - label_top_ratio)
        labels = {i: i for i in range(N) if strength[i] >= thr and strength[i] > 0}
    else:
        labels = {}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8,
                            font_color="black", ax=ax)

    # --- rango de color de aristas ---
    effs = [float(d.get("causal_effect", d.get("weight", 0.0)))
            for *_e, d in G.edges(keys=True, data=True)]
    if signed:
        amax = max((abs(x) for x in effs), default=1.0) or 1.0
        norm = plt.Normalize(-amax, amax)
    else:
        vmax_e = max((abs(x) for x in effs), default=1.0) or 1.0
        norm = plt.Normalize(0.0, vmax_e)

    self_edges, normal = [], []
    for e in G.edges(keys=True, data=True):
        (self_edges if e[0] == e[1] else normal).append(e)

    cs = "arc3,rad=0.06" if curved else "arc3,rad=0.0"
    for u, v, _k, d in normal:
        val = float(d.get("causal_effect", d.get("weight", 0.0)))
        color = edge_cmap(norm(val))
        width = base_width + abs(val) * width_scale
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width,
                               edge_color=[color], alpha=0.75, arrows=True,
                               arrowsize=10, ax=ax, node_size=sizes,
                               connectionstyle=cs)

    # auto-lazos solo en nodos etiquetados
    for u, _v, _k, d in self_edges:
        if u not in labels:
            continue
        val = float(d.get("causal_effect", d.get("weight", 0.0)))
        color = edge_cmap(norm(val))
        width = base_width + abs(val) * width_scale
        x, y = pos[u]
        dx = 0.012 * (ax.get_xlim()[1] - ax.get_xlim()[0] + _EPS)
        dy = 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0] + _EPS)
        ax.annotate("", xy=(x - dx, y - dx), xytext=(x - dy, y - dy),
                    arrowprops=dict(arrowstyle="simple", color=color,
                                    lw=1, alpha=0.7,
                                    mutation_scale=width * 1.1 + 5))

    # --- colorbars ---
    sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label(edge_label, rotation=270, labelpad=12)

    sm_n = plt.cm.ScalarMappable(cmap=node_cmap,
                                 norm=plt.Normalize(0, vmax_n)); sm_n.set_array([])
    cbar_n = fig.colorbar(sm_n, ax=ax, fraction=0.030, pad=0.06, location="left")
    cbar_n.set_label(r"$\Sigma$|effect| of connected edges",
                     rotation=90, labelpad=8)

    ax.set_xticks([])
    ax.set_yticks([])
    # 经纬度：用 cos(lat) 校正宽高比，保留数据范围，不把点压到中间
    lat0 = float(np.mean(coords[:, 1]))
    ax.set_aspect(1.0 / max(np.cos(np.deg2rad(lat0)), 1e-3), adjustable="box")
    ax.margins(0.06)
    ax.grid(True, linestyle="--", alpha=0.3)


# --------------------------------------------------------------------------- #
#  API de alto nivel (guardan via RunPaths.save_figure)                        #
# --------------------------------------------------------------------------- #
def plot_input_causal(causal: Dict, coords: np.ndarray, paths,
                      base="input_causal_graph_ref"):
    r"""Grafo causal G_C dirigido con signo (estilo PCMCI de referencia)."""
    G = build_causal_digraph(causal["graph"], causal["val_matrix"])
    n_edges = G.number_of_edges()
    meta = causal.get("meta", {}) if isinstance(causal, dict) else {}
    thr = meta.get("mask_threshold", None)
    fig, ax = plt.subplots(figsize=(10, 10))
    _draw_reference_style(G, coords, ax, fig, signed=True,
                          edge_cmap=plt.cm.seismic, edge_label="Causal Effect",
                          curved=False)
    ttl = (f"Input causal graph $G_C$ (directed, signed) | |E|={n_edges}"
           + (f", mask_thr={thr}" if thr is not None else "")
           + "\n(node labels: top 30% $\\Sigma$|effect|)")
    ax.set_title(ttl, fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    return paths.save_figure(fig, paths.figures, base)


def plot_input_intent(A_I: np.ndarray, coords: np.ndarray, paths, step: int,
                      delta=None, base="input_intent_graph_ref"):
    r"""Grafo de intencion G_I^t dirigido y ponderado (sin signo)."""
    G = build_weighted_digraph(A_I, self_loops=False)
    n_edges = G.number_of_edges()
    fig, ax = plt.subplots(figsize=(10, 10))
    _draw_reference_style(G, coords, ax, fig, signed=False,
                          edge_cmap=plt.cm.plasma,
                          edge_label=r"intent strength $s_{ij}$")
    ttl = (f"Input intent graph $G_I^t$ (directed, weighted) | step t={step}, "
           f"|E|={n_edges}"
           + (f", $\\delta$={delta:.3f}" if delta is not None else "")
           + "\n(node labels: top 30% $\\Sigma s$)")
    ax.set_title(ttl, fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    return paths.save_figure(fig, paths.figures, base)


def plot_view_reference(A: np.ndarray, coords: np.ndarray, paths,
                        base: str, title: str, directory=None):
    r"""Una vista derivada [N,N] (>=0) como grafo dirigido, estilo referencia."""
    G = build_weighted_digraph(A, self_loops=False)
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_reference_style(G, coords, ax, fig, signed=False,
                          edge_cmap=plt.cm.viridis, edge_label="edge weight")
    ax.set_title(f"{title}\n|E|={G.number_of_edges()} "
                 f"(node labels: top 30% $\\Sigma w$)", fontsize=10)
    directory = directory if directory is not None else paths.figures
    return paths.save_figure(fig, directory, base)
