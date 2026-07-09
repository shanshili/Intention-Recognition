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

# ---------------------------------------------------------------------------
# 时间步 <-> 真实时间的换算
# ---------------------------------------------------------------------------
# 数据集从 2025-01-01 00:00 开始，每小时一条 => 索引 c 对应第 c 个小时。
_BASE_DT = datetime(2025, 1, 1, 0, 0)   # 数据集起始时间
_STEP_HOURS = 1                          # 每个时间步的间隔（小时）


def step_to_dt_str(c: int, fmt: str = "%Y-%m-%d %H:%M") -> str:
    r"""把时间序列的索引 c（中心步）换算成真实时间字符串。

    c 是原始每小时序列中的行索引（即距起点的小时数），
    因此真实时间 = 起点 + c 小时（自动跨天/跨月）。
    """
    return (_BASE_DT + timedelta(hours=_STEP_HOURS * int(c))).strftime(fmt)

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


def plot_fused_visualization(S_D: np.ndarray, node_scores: np.ndarray,
                             coords: np.ndarray, paths, base="fused_SD", dt_str=None):
    r"""S_D 的热力图 + 坐标上的有向融合网络（边用 Reds 色条）。"""
    _apply_rc()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    im = axes[0].imshow(S_D, cmap="viridis", aspect="auto")
    axes[0].set_title("Fused edge score matrix  $S_D^t$", fontsize=_TITLE_FS)
    axes[0].set_xlabel("dst node", fontsize=_LABEL_FS)
    axes[0].set_ylabel("src node", fontsize=_LABEL_FS)
    cb0 = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.ax.tick_params(labelsize=_TICK_FS)

    q = _quiver_directed(axes[1], S_D, coords, cmap=_EDGE_CMAP, alpha=0.85)
    if q is not None:
        cbq = fig.colorbar(q, ax=axes[1], fraction=0.046, pad=0.04)
        cbq.set_label("edge score", fontsize=_LABEL_FS)
        cbq.ax.tick_params(labelsize=_TICK_FS)
    sc = node_scores
    sizes = 100 + 200 * (sc - sc.min()) / (sc.max() - sc.min() + _EPS)
    edge_deg = S_D.sum(0) + S_D.sum(1)
    active = (edge_deg > _EPS) | (sc > sc.min() + _EPS)
    ncolor = np.where(active, "#e63946", "#cccccc")
    axes[1].scatter(coords[:, 0], coords[:, 1], s=sizes, c=ncolor,
                    edgecolors="white", linewidths=0.6, zorder=2)
    _label_active_nodes(axes[1], coords, active)
    dt_label = f", time: {dt_str}" if dt_str else ""
    _frame(axes[1], coords,
           f"Fused deployment graph (directed; node score = $Score_i^t$){dt_label}")
    fig.tight_layout()
    return paths.save_figure(fig, paths.figures, base)


def plot_deployment_subgraph(deploy: Dict, coords: np.ndarray,
                             paths, base="deployment_subgraph", dt_str=None):
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
    dt_label = f", time: {dt_str}" if dt_str else ""
    ax.set_title(f"Deployment subgraph $G_D^t$  (directed)  "
                 f"(|V|={deploy['num_nodes']}, |E|={deploy['num_edges']}, "
                 f"cost={deploy['cost']:.1f}){dt_label}", fontsize=_TITLE_FS)
    ax.legend()
    _geo_axes(ax)                 # 经纬度刻度（x=经度, y=纬度）
    ax.set_aspect('equal')
    ax.margins(0.06)
    return paths.save_figure(fig, paths.figures, base)


# ---------------------------------------------------------------------------
def export_97_views(model, builder, centers: List[int], coords: np.ndarray,
                    paths, cfg, device):
    r"""导出 97 个（时间步）视图；按坐标真实比例绘制为长方形。"""
    import torch
    from .modules.module1_residual import build_views

    _apply_rc()
    m = cfg.model
    subdirs = {}
    for _, folder, _ in VIEW_EXPORTS:
        d = os.path.join(paths.views, folder)
        os.makedirs(d, exist_ok=True)
        subdirs[folder] = d

    max_steps = min(cfg.train.num_show_steps, len(centers))
    sel = centers[:max_steps]
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

        dt_str = step_to_dt_str(c)          # 由中心索引 c 换算真实时间
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
            base = f"{folder}_step{step_idx:03d}_c{c}"
            paths.save_figure(fig, subdirs[folder], base)
        manifest.append({"step": step_idx, "center": int(c),
                         "pos": int(sample["pos"]), "time": dt_str})
    return manifest
