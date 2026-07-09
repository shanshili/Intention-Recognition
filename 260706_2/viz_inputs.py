# -*- coding: utf-8 -*-
"""输入图的可视化（非派生视图）。

绘制：
  - 因果图 G_C（有向、加权且带有取自 val_matrix 的符号），
    遵循其作为 mask_threshold=0.5 和 top_nodes 子图的性质。
  - 意图图 G_I^t（有向且按强度 s_ij 加权）。

因果图（体现正负）的边用 seismic（红=正、蓝=负）；意图（不体现符号）的边用 Reds（越红越大）。

用法：
    python -m ratcgf.viz_inputs                 # 绘制两者（默认步）
    python -m ratcgf.viz_inputs --intent-step 50
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
import networkx as nx

from .config import Config
from .utils.io_utils import RunPaths
from .utils.data_loading import (
    load_all, load_causal, causal_to_adj, intent_to_adj_list)


# 全局字体与放大后的字号（对所有图片生效）
_LABEL_FONT = "Times New Roman"
_TITLE_FS = 15
_LABEL_FS = 13
_TICK_FS = 11
_NODE_LABEL_FS = 12

# 边色条：因果图（体现正负）用 seismic；意图（不体现符号，>=0）用 Reds
_SIGNED_CMAP = plt.cm.seismic      # 红=正、蓝=负
_UNSIGNED_CMAP = plt.cm.Reds       # 越红表示权重越大


def _apply_rc():
    r"""设置全局字体族与放大后的默认字号。"""
    matplotlib.rcParams.update({
        "font.family": _LABEL_FONT,
        "axes.unicode_minus": False,
        "font.size": _LABEL_FS,
        "axes.titlesize": _TITLE_FS,
        "axes.labelsize": _LABEL_FS,
        "xtick.labelsize": _TICK_FS,
        "ytick.labelsize": _TICK_FS,
        "legend.fontsize": _TICK_FS,
    })


# ---------------------------------------------------------------------------
# 构造带符号的因果矩阵（用于着色 + / -）
# ---------------------------------------------------------------------------
def causal_signed_matrix(causal: dict) -> np.ndarray:
    r"""[N,N] 带有因果边 i-->j 符号值的矩阵。

    对于每个 (i,j)，取在 graph[i,j,tau]=='-->' 的边中 |val| 最大的滞后 tau，
    并保留系数的符号。
    """
    graph = np.asarray(causal["graph"])
    val = np.asarray(causal["val_matrix"], dtype=float)
    if graph.ndim == 2:
        graph = graph[:, :, None]
        val = val[:, :, None]
    is_edge = (graph == "-->")
    masked = np.where(is_edge, val, 0.0)          # 保留符号
    absmax_lag = np.abs(masked).argmax(axis=2)    # |val| 最大的滞后 tau
    N = graph.shape[0]
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    W = masked[ii, jj, absmax_lag]
    W[~is_edge.any(axis=2)] = 0.0
    np.fill_diagonal(W, 0.0)
    return W


# ---------------------------------------------------------------------------
# 基于坐标的有向加权图通用绘制（seismic/Reds 色条 + 选中节点标签）
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 基于坐标的有向加权图通用绘制（seismic/Reds 色条 + 选中节点标签）
# ---------------------------------------------------------------------------
def _geo_axes(ax, nbins=6, xlabel="Longitude", ylabel="Latitude"):
    r"""把坐标轴显示为经纬度刻度（x=经度, y=纬度）。

    coords[:,0]=经度、coords[:,1]=纬度。若你的坐标其实是投影米制而非度，
    去掉下面的度符号格式化器即可（改用默认数字刻度）。
    """
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
    _fmt = FuncFormatter(lambda v, _p: f"{v:.2f}\u00b0")
    ax.xaxis.set_major_formatter(_fmt)
    ax.yaxis.set_major_formatter(_fmt)
    ax.tick_params(axis="both", labelsize=_TICK_FS)
    ax.set_xlabel(xlabel, fontsize=_LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=_LABEL_FS)

def draw_directed_weighted(ax, W: np.ndarray, coords: np.ndarray, title: str,
                           node_scores=None, highlight=None,
                           signed=False, fig=None, add_cbar=True,
                           edge_label="edge weight"):
    r"""W[i,j]!=0 -> 有向边 i->j。线宽 ~ |权重|；
    颜色：signed 用 seismic（红=正/蓝=负，对称）；否则用 Reds（[0,vmax] 顺序色）。"""
    N = W.shape[0]
    pos = {i: (float(coords[i, 0]), float(coords[i, 1])) for i in range(N)}

    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    ii, jj = np.nonzero(W)
    for i, j in zip(ii.tolist(), jj.tolist()):
        G.add_edge(i, j, w=float(W[i, j]))

    if node_scores is not None:
        ns = np.asarray(node_scores, dtype=float)
    else:
        ns = np.abs(W).sum(1) + np.abs(W).sum(0)
    rng = ns.max() - ns.min()
    sizes = 25 + 130 * (ns - ns.min()) / (rng + 1e-8)

    # 边着色：signed -> seismic 对称；unsigned -> Reds 单侧 [0, vmax]
    cmap = _SIGNED_CMAP if signed else _UNSIGNED_CMAP
    if G.number_of_edges() > 0:
        w = np.array([d["w"] for *_, d in G.edges(data=True)], dtype=float)
        wabs = np.abs(w).max() + 1e-8
        widths = 1 + 2 * np.abs(w) / wabs
        if signed:
            norm = plt.Normalize(-wabs, wabs)
        else:
            vmax = (w.max() if w.max() > 0 else 1.0)
            norm = plt.Normalize(0.0, vmax)
        colors = [cmap(norm(val)) for val in w]
        nx.draw_networkx_edges(
            G, pos, ax=ax, width=widths, edge_color=colors,
            arrows=True, arrowstyle="-|>", arrowsize=11,
            connectionstyle="arc3,rad=0.08", alpha=0.8, node_size=sizes)
        # 补充边的颜色映射色条（图例）
        if fig is not None and add_cbar:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.02)
            cbar.set_label(edge_label, fontsize=_LABEL_FS)
            cbar.ax.tick_params(labelsize=_TICK_FS)

    # 节点：选中(有边)标红白框，未选中灰色
    active_nodes = set()
    for u, v in G.edges():
        active_nodes.add(u)
        active_nodes.add(v)

    if highlight is not None:
        hset = set(int(h) for h in highlight)
        ncolor = ["#e63946" if i in hset else "#cccccc" for i in range(N)]
    else:
        ncolor = ["#e63946" if i in active_nodes else "#cccccc" for i in range(N)]

    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=sizes, node_color=ncolor,
        edgecolors="white", linewidths=0.6)

    # 选中节点加标签（白字，Times New Roman）
    labels = {i: i for i in range(N) if i in active_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=_NODE_LABEL_FS,
                            font_color="white", font_family=_LABEL_FONT, ax=ax)

    ax.set_title(title, fontsize=_TITLE_FS)
    _geo_axes(ax)                 # 经纬度刻度（x=经度, y=纬度）
    ax.set_aspect('equal')
    ax.margins(0.06)


# ---------------------------------------------------------------------------
# 具体图形
# ---------------------------------------------------------------------------
def plot_causal_graph(causal: dict, coords: np.ndarray, paths: RunPaths,
                      base="input_causal_graph", dt_str=None):
    r"""有向、加权且带符号的因果图；高亮 top_nodes。"""
    _apply_rc()
    W = causal_signed_matrix(causal)
    meta = causal.get("meta", {})
    thr = meta.get("mask_threshold", 0.5)
    top_nodes = meta.get("top_nodes")
    node_scores = np.asarray(meta["node_scores"], dtype=float) \
        if meta.get("node_scores") is not None else None

    n_edges = int((W != 0).sum())
    n_pos = int((W > 0).sum()); n_neg = int((W < 0).sum())

    fig, ax = plt.subplots(figsize=(12, 8))
    draw_directed_weighted(
        ax, W, coords,
        title=(f"Input causal graph $G_C$  (directed, signed weights)\n"
               f"mask_threshold={thr}  |E|={n_edges}  (+{n_pos} / -{n_neg})"
               + ("  gold=top_nodes" if top_nodes is not None else "")
               + (f", time: {dt_str}" if dt_str else "")),
        node_scores=node_scores, highlight=top_nodes, signed=True,
        fig=fig, edge_label="Causal Effect")

    legend = []
    if top_nodes is not None:
        legend.append(Line2D([0], [0], marker="o", color="w",
                             markerfacecolor="#ffb703", markersize=9,
                             label="top_nodes (core)"))
    if legend:
        ax.legend(handles=legend, loc="upper right", fontsize=_TICK_FS)
    return paths.save_figure(fig, paths.figures, base)


def plot_intent_graph(A_I: np.ndarray, coords: np.ndarray, paths: RunPaths,
                      step: int, delta=None, base="input_intent_graph", dt_str=None):
    r"""按强度 s_ij 加权的有向意图图。"""
    _apply_rc()
    n_edges = int((A_I > 0).sum())
    title = (f"Input intent graph $G_I^t$  (directed, weighted)\n"
             f"t = step {step}   |E|={n_edges}"
             + (f"   $\\delta$={delta:.3f}" if delta is not None else "")
             + (f", time: {dt_str}" if dt_str else ""))
    fig, ax = plt.subplots(figsize=(12, 8))
    draw_directed_weighted(ax, A_I, coords, title=title, signed=False,
                           fig=fig, edge_label=r"intent strength $s_{ij}$")
    return paths.save_figure(fig, paths.figures, base)


def plot_side_by_side(causal: dict, A_I: np.ndarray, coords: np.ndarray,
                      paths: RunPaths, step: int, delta=None,
                      base="input_graphs_side_by_side", dt_str=None):
    r"""双面板：因果（左）与意图（右）对比。"""
    _apply_rc()
    W = causal_signed_matrix(causal)
    meta = causal.get("meta", {})
    top_nodes = meta.get("top_nodes")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    draw_directed_weighted(
        axes[0], W, coords,
        title=f"Causal $G_C$  (mask={meta.get('mask_threshold', 0.5)})",
        highlight=top_nodes, signed=True, fig=fig, edge_label="Causal Effect")
    draw_directed_weighted(
        axes[1], A_I, coords,
        title=f"Intent $G_I^t$  (t=step {step})", signed=False,
        fig=fig, edge_label=r"intent strength $s_{ij}$")
    fig.tight_layout()
    return paths.save_figure(fig, paths.figures, base)


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------
def run(cfg: Config, intent_step: int):
    paths = RunPaths(cfg.out_root)
    print(f"[viz-inputs] 输出={paths.root}")

    bundle = load_all(cfg)
    N = bundle["N"]
    coords = bundle["data"].coords
    A_I_list = bundle["A_I_list"]
    delta_list = bundle.get("delta_list")

    # 重新加载完整的因果字典（带符号 + 元数据）；如果是合成的，
    # 从不保存的 load_all 重建最小字典 -> 使用文件。
    import os
    if os.path.exists(cfg.causal_path()):
        causal = load_causal(cfg.causal_path())
    else:
        # 合成：将邻接矩阵包装在兼容的字典中
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
    print(f"[viz-inputs] 因果   -> {p1.get('png')}")
    print(f"[viz-inputs] 意图   -> {p2.get('png')}")
    print(f"[viz-inputs] 组合   -> {p3.get('png')}")
    print(f"[viz-inputs] 格式: svg/png/pdf/eps 位于 {paths.figures}")
    return paths.root


def build_argparser():
    p = argparse.ArgumentParser(description="绘制输入图")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--causal-pkl", default=None)
    p.add_argument("--intent-npz", default=None)
    p.add_argument("--coord-csv", default=None)
    p.add_argument("--intent-step", type=int, default=0,
                   help="要绘制的意图图时间步 t 的索引")
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
    # 直接运行时的引导：python viz_inputs.py
    if __package__ in (None, ""):
        import os as _os, sys as _sys
        _here = _os.path.dirname(_os.path.abspath(__file__))
        _sys.path.insert(0, _os.path.dirname(_here))
        __package__ = _os.path.basename(_here)
    main()
