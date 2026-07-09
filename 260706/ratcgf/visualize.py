# -*- coding: utf-8 -*-
"""项目的所有可视化。

每张图同时保存为带time戳的 SVG、PNG、PDF 和 EPS 格式。

注意 (v2)：派生视图、融合图和部署子图现在绘制为有向加权图
（箭头 i->j，颜色/线宽按权重），使用 `quiver`（向量化且快速，便于导出多步）。
边约定：A[i, j] != 0  =>  从节点 i 到节点 j 的箭头。
"""
import os
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


# ---------------------------------------------------------------------------
def _quiver_directed(ax, A: np.ndarray, coords: np.ndarray,
                     cmap="viridis", shrink: float = 0.90,
                     width: float = 0.004, alpha: float = 0.75, zorder=1):
    """将 A 绘制为有向边 i->j；颜色 = 权重。返回 quiver 句柄
   （如果没有边则返回 None）。"""
    ii, jj = np.nonzero(A)
    if ii.size == 0:
        return None
    w = A[ii, jj].astype(float)
    x0 = coords[ii, 0]
    y0 = coords[ii, 1]
    dx = (coords[jj, 0] - x0) * shrink   # acorta un poco para que la cabeza
    dy = (coords[jj, 1] - y0) * shrink   # de flecha no quede bajo el nodo dst
    q = ax.quiver(x0, y0, dx, dy, w,
                  angles="xy", scale_units="xy", scale=1.0,
                  cmap=cmap, width=width,
                  headwidth=4.0, headlength=5.0, headaxislength=4.5,
                  alpha=alpha, zorder=zorder)
    return q


def _frame(ax, coords, title, fontsize=10):
    ax.set_title(title, fontsize=fontsize)
    ax.set_xticks([]); ax.set_yticks([])
    # 匹配节点坐标分布的横纵比例，不使用正方形
    ax.set_aspect('auto')
    ax.margins(0.06)


def _draw_adj(ax, A: np.ndarray, coords: np.ndarray, title: str,
              edge_cmap="viridis", node_color="#e63946"):
    """在坐标上绘制有向加权邻接矩阵。"""
    _quiver_directed(ax, A, coords, cmap=edge_cmap, zorder=3)
    deg = A.sum(1) + A.sum(0)
    sizes = 10 + 60 * (deg / (deg.max() + _EPS))

    # 选中节点红底白框，未选中灰色
    active = (A.sum(1) + A.sum(0)) > 0
    ncolor = np.where(active, "#e63946", "#cccccc")

    ax.scatter(coords[:, 0], coords[:, 1], s=sizes, c=ncolor,
               edgecolors="white", linewidths=0.4, zorder=2)
    _frame(ax, coords, title)


# ---------------------------------------------------------------------------
def plot_loss_curve(history: Dict, paths, base="loss_curve"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ep = range(1, len(history["train_loss"]) + 1)
    ax.plot(ep, history["train_loss"], label="train loss", lw=2)
    ax.plot(ep, history["val_loss"], label="val loss", lw=2)
    if history.get("train_pred"):
        ax.plot(ep, history["train_pred"], "--", label="train pred (U_pred)", lw=1.2)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("RA-TCGF training loss")
    ax.legend(); ax.grid(alpha=0.3)
    return paths.save_figure(fig, paths.figures, base)


def plot_fused_visualization(S_D: np.ndarray, node_scores: np.ndarray,
                             coords: np.ndarray, paths, base="fused_SD", dt_str=None):
    """S_D 的热力图 + 坐标上的有向融合网络。"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    im = axes[0].imshow(S_D, cmap="viridis", aspect="auto")
    axes[0].set_title("Fused edge score matrix  $S_D^t$")
    axes[0].set_xlabel("dst node"); axes[0].set_ylabel("src node")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    q = _quiver_directed(axes[1], S_D, coords, cmap="viridis", alpha=0.6)
    if q is not None:
        fig.colorbar(q, ax=axes[1], fraction=0.046, pad=0.04, label="edge score")
    sc = node_scores
    sizes = 15 + 120 * (sc - sc.min()) / (sc.max() - sc.min() + _EPS)
    active = sc > 0
    ncolor = np.where(active, "#e63946", "#cccccc")
    sp = axes[1].scatter(coords[:, 0], coords[:, 1], s=sizes, c=ncolor,
                         edgecolors="white", linewidths=0.4, zorder=2)
    fig.colorbar(sp, ax=axes[1], fraction=0.046, pad=0.04, label="node score")
    dt_label = f", time: {dt_str}" if dt_str else ""
    _frame(axes[1], coords,
           "Fused deployment graph (directed; node score = $Score_i^t$){dt_label}")
    fig.tight_layout()
    return paths.save_figure(fig, paths.figures, base)


def plot_deployment_subgraph(deploy: Dict, coords: np.ndarray,
                             paths, base="deployment_subgraph", dt_str=None):
    """绘制部署子图 G_D^t。"""
    fig, ax = plt.subplots(figsize=(12, 8))
    # candidatos en gris
    ax.scatter(coords[:, 0], coords[:, 1], s=12, c="#cccccc",
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
        ax.quiver(x0, y0, dx, dy, w, angles="xy", scale_units="xy", scale=1.0,
                  cmap="viridis", width=0.005, headwidth=4.0, headlength=5.0,
                  alpha=0.85, zorder=2)
    if nodes:
        c = coords[nodes]
        ax.scatter(c[:, 0], c[:, 1], s=90, c="#e63946",
                   edgecolors="black", linewidths=0.6, zorder=3, label="deployed")
    dt_label = f", time: {dt_str}" if dt_str else ""
    ax.set_title(f"Deployment subgraph $G_D^t$  (directed)  "
                 f"(|V|={deploy['num_nodes']}, |E|={deploy['num_edges']}, "
                 f"cost={deploy['cost']:.1f}){dt_label}")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('auto')
    ax.margins(0.06)

    return paths.save_figure(fig, paths.figures, base)


# ---------------------------------------------------------------------------
def export_97_views(model, builder, centers: List[int], coords: np.ndarray,
                    paths, cfg, device):
    """导出 97 个视图。"""
    import torch
    from .modules.module1_residual import build_views

    m = cfg.model
    subdirs = {}
    for _, folder, _ in VIEW_EXPORTS:
        d = os.path.join(paths.views, folder)
        os.makedirs(d, exist_ok=True)
        subdirs[folder] = d

    max_steps = min(cfg.train.num_show_steps, len(centers))
    sel = centers[:max_steps]
    print(f"[views] exportando {len(sel)} pasos x {len(VIEW_EXPORTS)} vistas "
          f"x 4 formatos = {len(sel)*len(VIEW_EXPORTS)*4} ficheros")

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

        for key, folder, title in VIEW_EXPORTS:
            A = views[key]
            fig, ax = plt.subplots(figsize=(5.5, 5.5))
            _draw_adj(ax, A, coords,
                      f"{title}\nstep t=c{c} (pos={sample['pos']})")
            base = f"{folder}_step{step_idx:03d}_c{c}"
            paths.save_figure(fig, subdirs[folder], base)
        manifest.append({"step": step_idx, "center": int(c),
                         "pos": int(sample["pos"])})
    return manifest
