# -*- coding: utf-8 -*-
r"""RA-TCGF 的 "PCMCI MultiDiGraph" 有向可视化样式。

复现项目的参考外观 (pcmci_draw_MultiDiGraph2 /
visualize_mask_subgraph) 而不依赖这些模块或 cartopy：

  - 带有箭头的有向图 MultiDiGraph。
  - 节点：颜色 = 连接边的 |效应| 之和，大小与该强度成比例；
    仅标记强度前 30% 的节点。
  - 边：颜色 = 带符号的因果效应（seismic，蓝=- / 红=+）用于因果图；
    对于视图/意图（无符号）使用顺序色彩映射。
  - 仅为标记的节点绘制自环。
  - 双色彩条：节点强度（左）和边效应（右）。

索引约定（对箭头方向很重要！）：
  - 因果 (.pkl PCMCI)：   graph[target, source, tau] == '-->'  =>  边 source->target
                           (与参考的 build_multi_di_graph 相同)。
  - 意图 / 视图：         A[i, j] > 0                          =>  边 i->j
                           (与 RA-TCGF 的 intent_to_adj_list 相同)。
"""

from typing import Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

_EPS = 1e-8


# --------------------------------------------------------------------------- #
#  MultiDiGraph 构造器                                                        #
# --------------------------------------------------------------------------- #
def build_causal_digraph(graph, val_matrix):
    r"""PCMCI [N,N,tau+1] -> MultiDiGraph (source->target，带符号和滞后)。"""
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
    r"""加权邻接 [N,N] -> MultiDiGraph (A[i,j]>0 => i->j，无符号)。"""
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
#  绘图（共享核心）                                                            #
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

    # --- 按连接强度的节点 ---
    strength = _node_strength(G, N)
    vmax_n = strength.max() if strength.max() > 0 else 1.0
    sizes = 100 + (strength / vmax_n) * 300

    # 判断节点是否为选中节点（有边连接）
    active_nodes = set()
    for u, v, _k, d in G.edges(keys=True, data=True):
        active_nodes.add(u)
        active_nodes.add(v)

    ncolor = []
    for i in range(N):
        if i in active_nodes:
            ncolor.append("#e63946")  # 选中节点：红底
        else:
            ncolor.append("#cccccc")  # 未选中节点：灰色

    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=ncolor,
                           edgecolors="white", linewidths=1.0,
                           alpha=0.9, ax=ax)

    # 仅前 30% 的标签（白字）
    if N > 0:
        thr = np.quantile(strength, 1.0 - label_top_ratio)
        labels = {i: i for i in range(N) if strength[i] >= thr and strength[i] > 0}
    else:
        labels = {}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8,
                            font_color="white", ax=ax)

    # --- 边颜色范围 ---
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

    # --- 自环边（箭头）---
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

    # --- 边颜色条 ---
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
    # 匹配节点坐标分布的横纵比例，不使用正方形
    ax.set_aspect('auto')
    ax.margins(0.06)
    ax.grid(True, linestyle="--", alpha=0.3)




# --------------------------------------------------------------------------- #
#  高级 API（通过 RunPaths.save_figure 保存）                                  #
# --------------------------------------------------------------------------- #
def plot_input_causal(causal: Dict, coords: np.ndarray, paths,
                      base="input_causal_graph_ref", dt_str=None):
    r"""带符号的有向因果图 G_C（参考 PCMCI 样式）。"""
    G = build_causal_digraph(causal["graph"], causal["val_matrix"])
    n_edges = G.number_of_edges()
    meta = causal.get("meta", {}) if isinstance(causal, dict) else {}
    thr = meta.get("mask_threshold", None)
    fig, ax = plt.subplots(figsize=(12, 8))
    _draw_reference_style(G, coords, ax, fig, signed=True,
                          edge_cmap=plt.cm.RdBu_r, edge_label="Causal Effect",
                          curved=False)



def plot_input_intent(A_I: np.ndarray, coords: np.ndarray, paths, step: int,
                      delta=None, base="input_intent_graph_ref", dt_str=None):
    r"""有向加权的意图图 G_I^t（无符号）。"""
    G = build_weighted_digraph(A_I, self_loops=False)
    n_edges = G.number_of_edges()
    fig, ax = plt.subplots(figsize=(12, 8))
    _draw_reference_style(G, coords, ax, fig, signed=False,
                          edge_cmap=plt.cm.plasma,
                          edge_label=r"intent strength $s_{ij}$")
    dt_label = f", time: {dt_str}" if dt_str else ""
    ttl = (f"Input intent graph $G_I^t$ (directed, weighted) | step t={step}, "
           f"|E|={n_edges}"
           + (f", $\\delta$={delta:.3f}" if delta is not None else "")
           + dt_label
           + "\n(node labels: top 30% $\\Sigma s$)")
    ax.set_title(ttl, fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    return paths.save_figure(fig, paths.figures, base)


def plot_view_reference(A: np.ndarray, coords: np.ndarray, paths,
                        base: str, title: str, directory=None, dt_str=None):
    r"""派生视图 [N,N] (>=0) 作为有向图，参考样式。"""
    G = build_weighted_digraph(A, self_loops=False)
    fig, ax = plt.subplots(figsize=(12, 8))
    _draw_reference_style(G, coords, ax, fig, signed=False,
                          edge_cmap=plt.cm.viridis, edge_label="edge weight")
    dt_label = f"\ntime: {dt_str}" if dt_str else ""
    ax.set_title(f"{title}\n|E|={G.number_of_edges()} "
                 f"(node labels: top 30% $\\Sigma w$){dt_label}", fontsize=10)
    directory = directory if directory is not None else paths.figures
    return paths.save_figure(fig, directory, base)
