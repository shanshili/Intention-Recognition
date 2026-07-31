# -*- coding: utf-8 -*-
"""项目的所有可视化。

每张图同时保存为带时间戳的 SVG、PNG、PDF 和 EPS 格式。

注意 (v3)：
  - 派生视图、融合图和部署子图均绘制为有向加权图（箭头 i->j）。
  - 因果-意图融合类的边（不体现符号）统一使用 Reds 顺序色条（越红越大），
    并补充边的颜色映射色条（图例）。
  - 97 步导出的图片不再是正方形，按节点坐标真实比例绘制为长方形，并给选中节点加标签。
  - 部署子图的选中节点使用白色边框，箭头同样使用 Reds 色条。
边约定：A[i, j] != 0  =>  从节点 i 到节点 j 的箭头。
"""
import os
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np
import matplotlib.colors as mcolors  # 文件顶部加这个 import

# ---------------------------------------------------------------------------
# 时间步 <-> 真实时间的换算
# ---------------------------------------------------------------------------
# 数据集从 2025-01-01 00:00 开始，每小时一条 => 索引 c 对应第 c 个小时。
_BASE_DT = datetime(2025, 1, 1, 0, 0)   # 兼容旧调用的默认起点
_STEP_MINUTES = 60                         # 兼容旧调用的默认采样间隔


def step_to_dt_str(step: int, fmt: str = "%Y-%m-%d %H:%M:%S",
                   base_dt=None, step_minutes: int = _STEP_MINUTES) -> str:
    r"""把原始时间序列位置 ``step/pos`` 换算成真实时间字符串。

    注意：SampleBuilder 中的 ``c`` 是意图序列局部索引，真正对应原始
    时间序列的索引是 ``sample["pos"]``。调用者应传 pos，而不是 c。
    ``base_dt`` 可传 datetime 或 ``"YYYY-mm-dd HH:MM"`` 字符串。
    """
    if base_dt is None:
        base = _BASE_DT
    elif isinstance(base_dt, datetime):
        base = base_dt
    else:
        base = datetime.fromisoformat(str(base_dt))
    return (base + timedelta(minutes=int(step_minutes) * int(step))).strftime(fmt)

# 内部名 -> (子文件夹, 标题)
VIEW_EXPORTS = [
    ("CI", "consensus", "Causal-Intent Consensus (A_CI)"),
    ("R", "intent_residual", "Intent Residual (R_I)"),
    ("CmI", "causal_inactive", "Causal Inactive (R_C)"),
    ("P", "intent_persist", "Intent Persistent (A_P)"),
    ("N", "intent_new", "Intent New (A_N)"),
    ("E", "intent_decay", "Intent Decay (A_E)"),
]

_EPS = 1e-8

# 全局字体与放大后的字号（对所有图片生效）
_LABEL_FONT = "Times New Roman"
_TITLE_FS = 15
_LABEL_FS = 13
_TICK_FS = 11
_NODE_LABEL_FS = 12

# 因果-意图融合类（不体现符号，边权>=0）统一使用 Reds 顺序色条（越红越大）
_EDGE_CMAP = "Reds"


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


def _coords_figsize(coords: np.ndarray, base: float = 6.5,
                    cbar_pad: float = 1.28):
    r"""根据节点坐标的真实横纵范围返回长方形画布尺寸（宽, 高）。

    宽度固定为 base（再乘以 cbar_pad 给色条留白），高度按 y/x 范围比例缩放。
    """
    x = coords[:, 0]
    y = coords[:, 1]
    xr = float(x.max() - x.min()) or 1.0
    yr = float(y.max() - y.min()) or 1.0
    ratio = yr / xr
    h = base * ratio
    h = float(np.clip(h, 2.6, base * 1.6))   # 限制高度，避免过扁 / 过高
    return (base * cbar_pad, h)


# ---------------------------------------------------------------------------
def _quiver_directed(ax, A: np.ndarray, coords: np.ndarray,
                     cmap: str = _EDGE_CMAP, shrink: float = 0.90,
                     width: float = 0.006, alpha: float = 0.85, zorder=1):
    r"""将 A 绘制为有向边 i->j；颜色 = 权重（Reds 顺序色，[0, vmax] 归一化）。

    返回 quiver 句柄（若没有边则返回 None）。"""
    ii, jj = np.nonzero(A)
    if ii.size == 0:
        return None
    w = A[ii, jj].astype(float)
    x0 = coords[ii, 0]
    y0 = coords[ii, 1]
    dx = (coords[jj, 0] - x0) * shrink   # 稍微缩短箭身，
    dy = (coords[jj, 1] - y0) * shrink   # 使箭头不被目标节点遮挡
    q = ax.quiver(x0, y0, dx, dy, w,
                  angles="xy", scale_units="xy", scale=1.0,
                  cmap=cmap, width=width,
                  headwidth=4.0, headlength=5.0, headaxislength=4.5,
                  alpha=alpha, zorder=zorder)
    vmax = float(np.abs(w).max()) or 1.0      # 非负边权：Reds 顺序色 [0, vmax]
    q.set_clim(0.0, vmax)
    return q

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


def _frame(ax, coords, title, fontsize=_TITLE_FS, aspect="equal"):
    ax.set_title(title, fontsize=fontsize)
    _geo_axes(ax)                 # 经纬度刻度（x=经度, y=纬度）
    # aspect="equal" 时按数据真实比例等比绘制（x、y 单位等长，不拉伸/压扁）；
    # "auto" 则把数据拉满坐标框（可能变形）。
    ax.set_aspect(aspect)
    ax.margins(0.06)


def _label_active_nodes(ax, coords, active_mask, fontsize=_NODE_LABEL_FS):
    r"""给选中（有边连接）的节点添加白色标签（Times New Roman）。"""
    for idx in np.where(active_mask)[0]:
        ax.text(float(coords[idx, 0]), float(coords[idx, 1]), str(int(idx)),
                fontsize=fontsize, color="white", ha="center", va="center",
                fontfamily=_LABEL_FONT, zorder=5)


def _draw_adj(ax, A: np.ndarray, coords: np.ndarray, title: str,
              edge_cmap: str = _EDGE_CMAP, fig=None, add_cbar=True,
              edge_label="edge weight"):
    r"""在坐标上绘制有向加权邻接矩阵（Reds 色条 + 选中节点标签）。"""
    q = _quiver_directed(ax, A, coords, cmap=edge_cmap, zorder=3)
    deg = A.sum(1) + A.sum(0)
    sizes = 100 + 300 * (deg / (deg.max() + _EPS))

    # 选中节点红底白框，未选中灰色
    active = (A.sum(1) + A.sum(0)) > 0
    ncolor = np.where(active, "#e63946", "#cccccc")

    ax.scatter(coords[:, 0], coords[:, 1], s=sizes, c=ncolor,
               edgecolors="white", linewidths=0.6, zorder=2)
    _label_active_nodes(ax, coords, active)     # 选中节点加标签
    _frame(ax, coords, title)

    # 补充边的颜色映射色条（图例）
    if fig is not None and add_cbar and q is not None:
        cbar = fig.colorbar(q, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(edge_label, fontsize=_LABEL_FS)
        cbar.ax.tick_params(labelsize=_TICK_FS)


# ---------------------------------------------------------------------------
def plot_loss_curve(history: Dict, paths, base="loss_curve"):
    _apply_rc()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ep = range(1, len(history["train_loss"]) + 1)
    ax.plot(ep, history["train_loss"], label="train loss", lw=2)
    ax.plot(ep, history["val_loss"], label="val loss", lw=2)
    if history.get("train_pred"):
        ax.plot(ep, history["train_pred"], "--", label="train pred (U_pred)", lw=1.2)
    ax.set_xlabel("epoch", fontsize=_LABEL_FS)
    ax.set_ylabel("loss", fontsize=_LABEL_FS)
    ax.set_title("RA-TCGF training loss", fontsize=_TITLE_FS)
    ax.legend(); ax.grid(alpha=0.3)
    return paths.save_figure(fig, paths.figures, base)


# def plot_fused_visualization(S_D: np.ndarray, node_scores: np.ndarray,
#                              coords: np.ndarray, paths, base="fused_SD", dt_str=None):
#     r"""S_D 的热力图 + 坐标上的有向融合网络（边用 Reds 色条）。"""
#     _apply_rc()
#     fig, axes = plt.subplots(1, 2, figsize=(16, 6))
#     im = axes[0].imshow(S_D, cmap="viridis", aspect="auto")
#     axes[0].set_title("Fused edge score matrix  $S_D^t$", fontsize=_TITLE_FS)
#     axes[0].set_xlabel("dst node", fontsize=_LABEL_FS)
#     axes[0].set_ylabel("src node", fontsize=_LABEL_FS)
#     cb0 = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
#     cb0.ax.tick_params(labelsize=_TICK_FS)
#
#     q = _quiver_directed(axes[1], S_D, coords, cmap=_EDGE_CMAP, alpha=0.85)
#     if q is not None:
#         cbq = fig.colorbar(q, ax=axes[1], fraction=0.046, pad=0.04)
#         cbq.set_label("edge score", fontsize=_LABEL_FS)
#         cbq.ax.tick_params(labelsize=_TICK_FS)
#     sc = node_scores
#     sizes = 100 + 200 * (sc - sc.min()) / (sc.max() - sc.min() + _EPS)
#     edge_deg = S_D.sum(0) + S_D.sum(1)
#     active = (edge_deg > _EPS) | (sc > sc.min() + _EPS)
#     ncolor = np.where(active, "#e63946", "#cccccc")
#     axes[1].scatter(coords[:, 0], coords[:, 1], s=sizes, c=ncolor,
#                     edgecolors="white", linewidths=0.6, zorder=2)
#     _label_active_nodes(axes[1], coords, active)
#     dt_label = f", time: {dt_str}" if dt_str else ""
#     _frame(axes[1], coords,
#            f"Fused deployment graph (directed; node score = $Score_i^t$){dt_label}")
#     fig.tight_layout()
#     return paths.save_figure(fig, paths.figures, base)

# def plot_fused_visualization(S_D: np.ndarray, node_scores: np.ndarray,
#                              coords: np.ndarray, paths, base="fused_SD",
#                              dt_str=None, edge_quantile=0.0):
#     r"""左：S_D 热力图；右：以边融合分数为中心的有向网络。
#
#     右图完全由 S_D 驱动：
#       - 边颜色 = S_D[i,j]（Reds，越红分数越高）；
#       - 节点大小/高亮 = 该节点关联的边分数之和（edge strength）；
#       - edge_quantile>0 时，仅绘制分数在该分位以上的高分边。
#     node_scores 仅用于可选参考，不再决定节点显示。
#     """
#     _apply_rc()
#     fig, axes = plt.subplots(1, 2, figsize=(16, 6))
#
#     # ---- 左：融合边分数矩阵热力图（不变）----
#     im = axes[0].imshow(S_D, cmap="viridis", aspect="auto")
#     axes[0].set_title("Fused edge score matrix  $S_D^t$", fontsize=_TITLE_FS)
#     axes[0].set_xlabel("dst node", fontsize=_LABEL_FS)
#     axes[0].set_ylabel("src node", fontsize=_LABEL_FS)
#     cb0 = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
#     cb0.ax.tick_params(labelsize=_TICK_FS)
#
#     # ---- 右：以边融合分数为中心的有向图 ----
#     S_draw = S_D.copy()
#     # 可选：只保留高分边（按非零边分数的分位数阈值）
#     if edge_quantile and edge_quantile > 0.0:
#         nz = S_draw[S_draw > 0]
#         if nz.size > 0:
#             thr = np.quantile(nz, edge_quantile)
#             S_draw = np.where(S_draw >= thr, S_draw, 0.0)
#
#     q = _quiver_directed(axes[1], S_draw, coords, cmap=_EDGE_CMAP, alpha=0.85)
#     if q is not None:
#         cbq = fig.colorbar(q, ax=axes[1], fraction=0.046, pad=0.04)
#         cbq.set_label("edge score $S_D^t[i,j]$", fontsize=_LABEL_FS)
#         cbq.ax.tick_params(labelsize=_TICK_FS)
#
#     # 节点大小 / 高亮：均由"关联边分数之和"决定（edge-centric）
#     edge_strength = S_draw.sum(0) + S_draw.sum(1)          # 每节点入+出边分数
#     smax = edge_strength.max()
#     sizes = 100 + 300 * (edge_strength / (smax + _EPS))    # 分数越高点越大
#     active = edge_strength > _EPS                          # 有高分边相连才高亮
#     ncolor = np.where(active, "#e63946", "#cccccc")
#
#     axes[1].scatter(coords[:, 0], coords[:, 1], s=sizes, c=ncolor,
#                     edgecolors="white", linewidths=0.6, zorder=2)
#     _label_active_nodes(axes[1], coords, active)
#     dt_label = f", time: {dt_str}" if dt_str else ""
#     _frame(axes[1], coords,
#            f"Fused deployment graph (directed; edge/node = $S_D^t$ strength)"
#            f"{dt_label}")
#     fig.tight_layout()
#     return paths.save_figure(fig, paths.figures, base)

def plot_fused_visualization(S_D: np.ndarray, node_scores: np.ndarray,
                             coords: np.ndarray, paths, base="fused_SD",
                             dt_str=None,
                             edge_quantile=0.0, heatmap_scale="log",
                             mask_zeros=True, directory=None):
    r"""左：S_D 热力图（可选非零掩码 + 对数着色）；右：以边分数为中心的网络。

    heatmap_scale: "log" 用对数着色压缩动态范围突出高分边；"linear" 用线性。
    mask_zeros:    True 时把 S_D==0 设为背景色（不参与配色），只给真实边上色。
    """
    _apply_rc()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ---- 左：融合边分数矩阵热力图 ----
    cmap = plt.cm.get_cmap("viridis").copy()
    cmap.set_bad(color="#f0f0f0")          # 被掩码的 0 元素显示为浅灰背景

    S_plot = S_D.astype(float).copy()
    nz = S_plot[S_plot > 0]

    if mask_zeros:                          # 把 0（无边）掩掉，不参与配色
        S_plot = np.ma.masked_less_equal(S_plot, 0.0)

    if heatmap_scale == "log" and nz.size > 0:
        vmin = float(nz.min())
        vmax = float(nz.max())
        norm = mcolors.LogNorm(vmin=vmin, vmax=max(vmax, vmin * (1 + _EPS)))
    else:
        norm = mcolors.Normalize(vmin=0.0,
                                 vmax=float(nz.max()) if nz.size > 0 else 1.0)

    im = axes[0].imshow(S_plot, cmap=cmap, norm=norm, aspect="auto")
    scale_tag = "log" if heatmap_scale == "log" else "linear"
    matrix_time_label = f"\ntime t: {dt_str}" if dt_str else ""
    axes[0].set_title(
        f"Fused edge score matrix  $S_D^t$  ({scale_tag}){matrix_time_label}",
        fontsize=_TITLE_FS)
    axes[0].set_xlabel("dst node", fontsize=_LABEL_FS)
    axes[0].set_ylabel("src node", fontsize=_LABEL_FS)
    cb0 = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.set_label("edge score $S_D^t[i,j]$", fontsize=_LABEL_FS)
    cb0.ax.tick_params(labelsize=_TICK_FS)

    # ---- 右：以边融合分数为中心的有向图（同上一版）----
    S_draw = S_D.copy()
    if edge_quantile and edge_quantile > 0.0:
        nzd = S_draw[S_draw > 0]
        if nzd.size > 0:
            thr = np.quantile(nzd, edge_quantile)
            S_draw = np.where(S_draw >= thr, S_draw, 0.0)

    q = _quiver_directed(axes[1], S_draw, coords, cmap=_EDGE_CMAP, alpha=0.85)
    if q is not None:
        cbq = fig.colorbar(q, ax=axes[1], fraction=0.046, pad=0.04)
        cbq.set_label("edge score $S_D^t[i,j]$", fontsize=_LABEL_FS)
        cbq.ax.tick_params(labelsize=_TICK_FS)

    edge_strength = S_draw.sum(0) + S_draw.sum(1)
    smax = float(edge_strength.max())
    sizes = 100 + 3000 * (edge_strength / (smax + _EPS))
    # 相对阈值：强度达到全局最大的某个很小比例即算活跃
    active = edge_strength > 1e-6 * smax  # 关键：相对 smax，而非绝对 _EPS
    ncolor = np.where(active, "#e63946", "#cccccc")

    axes[1].scatter(coords[:, 0], coords[:, 1], s=sizes, c=ncolor,
                    edgecolors="white", linewidths=0.6, zorder=2)
    _label_active_nodes(axes[1], coords, active)
    dt_label = f"\ntime t: {dt_str}" if dt_str else ""
    _frame(axes[1], coords,
           f"Fused edge-score graph $S_D^t$ (directed)"
           f"{dt_label}")
    fig.tight_layout()
    save_dir = directory or paths.figures
    os.makedirs(save_dir, exist_ok=True)
    return paths.save_figure(fig, save_dir, base)



def plot_deployment_subgraph(deploy: Dict, coords: np.ndarray,
                             paths, base="deployment_subgraph", dt_str=None,
                             directory=None):
    r"""绘制部署子图 G_D^t（选中节点白色边框，箭头用 Reds 色条）。"""
    _apply_rc()
    fig, ax = plt.subplots(figsize=(12, 8))
    # 候选节点为灰色
    ax.scatter(coords[:, 0], coords[:, 1], s=100, c="#cccccc",
               zorder=1, label="candidate")
    nodes = deploy["nodes"]
    edges = deploy["edges"]
    if edges:
        ii = np.array([i for i, _, _ in edges])
        jj = np.array([j for _, j, _ in edges])
        w = np.array([s for _, _, s in edges], dtype=float)
        x0 = coords[ii, 0]; y0 = coords[ii, 1]
        dx = (coords[jj, 0] - x0) * 0.9
        dy = (coords[jj, 1] - y0) * 0.9
        q = ax.quiver(x0, y0, dx, dy, w, angles="xy", scale_units="xy",
                      scale=1.0, cmap=_EDGE_CMAP, width=0.006,
                      headwidth=4.0, headlength=5.0, alpha=0.9, zorder=2)
        vmax = float(np.abs(w).max()) or 1.0     # 非负边权：Reds 顺序色 [0, vmax]
        q.set_clim(0.0, vmax)
        cbar = fig.colorbar(q, ax=ax, fraction=0.030, pad=0.02)
        cbar.set_label("edge score", fontsize=_LABEL_FS)
        cbar.ax.tick_params(labelsize=_TICK_FS)
    if nodes:
        c = coords[nodes]
        ax.scatter(c[:, 0], c[:, 1], s=400, c="#e63946",
                   edgecolors="white", linewidths=1.0, zorder=3,
                   label="deployed")          # 选中节点：白色边框
        _label_active_nodes(ax, coords, np.isin(np.arange(len(coords)), nodes))
    dt_label = f"\ndeployment time t: {dt_str}" if dt_str else ""
    ax.set_title(f"Deployment subgraph $G_D^t$  (directed)  "
                 f"(|V|={deploy['num_nodes']}, |E|={deploy['num_edges']}, "
                 f"cost={deploy['cost']:.1f}){dt_label}",
                 fontsize=_TITLE_FS)
    ax.legend()
    _geo_axes(ax)                 # 经纬度刻度（x=经度, y=纬度）
    ax.set_aspect('equal')
    ax.margins(0.06)
    save_dir = directory or paths.figures
    os.makedirs(save_dir, exist_ok=True)
    return paths.save_figure(fig, save_dir, base)


def plot_deployment_metrics(rows: List[Dict], paths,
                            base="deployment_metrics", directory=None):
    r"""绘制多时间步部署规模、预算与效用的汇总趋势。"""
    if not rows:
        return None
    _apply_rc()
    x = np.arange(len(rows))
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(x, [r["num_nodes"] for r in rows], marker="o", label="nodes")
    axes[0].plot(x, [r["num_edges"] for r in rows], marker="s", label="edges")
    axes[0].set_ylabel("count"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(x, [r["cost"] for r in rows], marker="o", label="cost")
    axes[1].plot(x, [r["budget"] for r in rows], "--", label="budget")
    axes[1].set_ylabel("budget"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot(x, [r["u_track"] for r in rows], marker="o", label="U_track")
    axes[2].plot(x, [r["u_elastic"] for r in rows], marker="s", label="U_elastic")
    axes[2].plot(x, [r["total_utility"] for r in rows], marker="^", label="total")
    axes[2].set_ylabel("utility"); axes[2].set_xlabel("deployment time")
    axes[2].legend(); axes[2].grid(alpha=0.3)
    tick_count = min(8, len(rows))
    tick_idx = np.unique(np.linspace(0, len(rows) - 1, tick_count, dtype=int))
    axes[2].set_xticks(tick_idx)
    axes[2].set_xticklabels(
        [rows[i]["deployment_time"][5:16] for i in tick_idx],
        rotation=35, ha="right")
    fig.suptitle("Multi-step deployment metrics (strict dataset timeline)")
    fig.tight_layout()
    save_dir = directory or paths.figures
    os.makedirs(save_dir, exist_ok=True)
    return paths.save_figure(fig, save_dir, base)


def plot_deployment_sequence(records: List[Dict], coords: np.ndarray, paths,
                             base="deployment_sequence", max_panels=12,
                             directory=None):
    r"""把多个部署时间步放在一张总览图中；过多时均匀抽取最多 max_panels 步。"""
    if not records:
        return None
    _apply_rc()
    n_all = len(records)
    if n_all > max_panels:
        idx = np.unique(np.linspace(0, n_all - 1, max_panels, dtype=int))
        shown = [records[i] for i in idx]
    else:
        shown = records
    n = len(shown)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows),
                             squeeze=False)
    for ax, rec in zip(axes.flat, shown):
        ax.scatter(coords[:, 0], coords[:, 1], s=28, c="#d0d0d0", zorder=1)
        edges = rec.get("edges", [])
        if edges:
            ii = np.asarray([e[0] for e in edges], dtype=int)
            jj = np.asarray([e[1] for e in edges], dtype=int)
            w = np.asarray([e[2] for e in edges], dtype=float)
            x0, y0 = coords[ii, 0], coords[ii, 1]
            dx = (coords[jj, 0] - x0) * 0.9
            dy = (coords[jj, 1] - y0) * 0.9
            q = ax.quiver(x0, y0, dx, dy, w, angles="xy",
                          scale_units="xy", scale=1.0, cmap=_EDGE_CMAP,
                          width=0.004, alpha=0.85, zorder=2)
            q.set_clim(0.0, float(w.max()) or 1.0)
        nodes = rec.get("nodes", [])
        if nodes:
            c = coords[np.asarray(nodes, dtype=int)]
            ax.scatter(c[:, 0], c[:, 1], s=90, c="#e63946",
                       edgecolors="white", linewidths=0.5, zorder=3)
        ax.set_title(f"#{rec['sequence_index']}  c={rec['intent_index']}  pos={rec['pos']}\n"
                     f"{rec['deployment_time']}  |V|={rec['num_nodes']} |E|={rec['num_edges']}")
        _geo_axes(ax, nbins=4)
        ax.set_aspect("equal"); ax.margins(0.05)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(f"Deployment sequence overview ({n_all} steps; showing {n})")
    fig.tight_layout()
    save_dir = directory or paths.figures
    os.makedirs(save_dir, exist_ok=True)
    return paths.save_figure(fig, save_dir, base)


# ---------------------------------------------------------------------------
def export_97_views(model, builder, centers: List[int], coords: np.ndarray,
                    paths, cfg, device):
    r"""导出已由 main.py 完成时间范围/数量选择的派生视图。

    函数名保留 export_97_views 仅用于兼容旧调用；实际数量由 cfg.views.num_steps
    和传入的 centers 决定，不再在本函数内部二次截取“最后 97 步”。
    """
    import torch
    from .modules.module1_residual import build_views

    _apply_rc()
    m = cfg.model
    subdirs = {}
    for _, folder, _ in VIEW_EXPORTS:
        d = os.path.join(paths.views, folder)
        os.makedirs(d, exist_ok=True)
        subdirs[folder] = d

    sel = list(centers)
    print(f"[views] 正在导出 {len(sel)} 步 x {len(VIEW_EXPORTS)} 视图 "
          f"x 4 格式 = {len(sel)*len(VIEW_EXPORTS)*4} 个文件")

    figsize = _coords_figsize(coords)      # 长方形画布，匹配节点分布真实比例

    model.eval()
    manifest = []
    for step_idx, c in enumerate(sel):
        sample = builder.build(c)
        with torch.no_grad():
            built = build_views(sample["A_C"], sample["A_I_prev"],
                                sample["A_I_cur"], sample["A_I_next"],
                                m.delta_C, m.delta_I, m.eta)
        views = {k: v.detach().cpu().numpy() for k, v in built["views"].items()}
        views["E"] = built["decay"].detach().cpu().numpy()

        dt_str = sample["time"]
        for key, folder, title in VIEW_EXPORTS:
            A = views[key]
            # 因果派生视图（causal_inactive / R_C）转置以保持因果方向一致
            A_draw = A.T if key == "CmI" else A
            fig, ax = plt.subplots(figsize=figsize)
            _draw_adj(ax, A_draw, coords,
                      f"{title}\nstep {step_idx} (t=c{c}, pos={sample['pos']}), "
                      f"time: {dt_str}",
                      fig=fig, edge_label="edge weight")
            fig.tight_layout()
            time_tag = dt_str.replace("-", "").replace(":", "").replace(" ", "T")
            base = (f"pos{sample['pos']:06d}_{time_tag}_"
                    f"c{c:04d}_step{step_idx:03d}_{folder}")
            paths.save_figure(fig, subdirs[folder], base)
        manifest.append({
            "step": step_idx,
            "intent_index": int(c),
            "pos": int(sample["pos"]),
            "view_time": dt_str,
        })
    return manifest
