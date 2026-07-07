# -*- coding: utf-8 -*-
r"""Todas las visualizaciones del proyecto.

Cada figura se guarda simultaneamente en SVG, PNG, PDF y EPS con timestamp.

NOTA (v2): las vistas derivadas, el grafo fusionado y el subgrafo de despliegue
se dibujan ahora como grafos DIRIGIDOS y PONDERADOS (flechas i->j, color/anchura
por peso) usando `quiver` (vectorizado y rapido para exportar muchos pasos).
Convencion de arista: A[i, j] != 0  =>  flecha del nodo i al nodo j.
"""
import os
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# nombre chino -> (clave interna, subcarpeta, titulo)
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
    """Dibuja A como aristas DIRIGIDAS i->j; color = peso. Devuelve el handle
    de quiver (o None si no hay aristas)."""
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
    lat0 = float(np.mean(coords[:, 1]))
    ax.set_aspect(1.0 / max(np.cos(np.deg2rad(lat0)), 1e-3), adjustable="box")
    ax.margins(0.06)


def _draw_adj(ax, A: np.ndarray, coords: np.ndarray, title: str,
              edge_cmap="viridis", node_color="#d1495b"):
    """Dibuja una adyacencia ponderada DIRIGIDA sobre las coordenadas."""
    _quiver_directed(ax, A, coords, cmap=edge_cmap)
    deg = A.sum(1) + A.sum(0)
    sizes = 10 + 60 * (deg / (deg.max() + _EPS))
    ax.scatter(coords[:, 0], coords[:, 1], s=sizes, c=node_color,
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
                             coords: np.ndarray, paths, base="fused_SD"):
    """Heatmap de S_D + red fusionada DIRIGIDA sobre coordenadas."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    im = axes[0].imshow(S_D, cmap="viridis", aspect="auto")
    axes[0].set_title("Fused edge score matrix  $S_D^t$")
    axes[0].set_xlabel("dst node"); axes[0].set_ylabel("src node")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    q = _quiver_directed(axes[1], S_D, coords, cmap="viridis", alpha=0.6)
    if q is not None:
        fig.colorbar(q, ax=axes[1], fraction=0.046, pad=0.04, label="edge score")
    sc = node_scores
    sizes = 15 + 120 * (sc - sc.min()) / (sc.max() - sc.min() + _EPS)
    sp = axes[1].scatter(coords[:, 0], coords[:, 1], s=sizes, c=sc,
                         cmap="magma", edgecolors="white", linewidths=0.4, zorder=2)
    fig.colorbar(sp, ax=axes[1], fraction=0.046, pad=0.04, label="node score")
    _frame(axes[1], coords,
           "Fused deployment graph (directed; node score = $Score_i^t$)")
    fig.tight_layout()
    return paths.save_figure(fig, paths.figures, base)


def plot_deployment_subgraph(deploy: Dict, coords: np.ndarray,
                             paths, base="deployment_subgraph"):
    """Subgrafo de despliegue G_D^t con aristas DIRIGIDAS i->j."""
    fig, ax = plt.subplots(figsize=(7.5, 7))
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
    ax.set_title(f"Deployment subgraph $G_D^t$  (directed)  "
                 f"(|V|={deploy['num_nodes']}, |E|={deploy['num_edges']}, "
                 f"cost={deploy['cost']:.1f})")
    ax.set_xticks([]); ax.set_yticks([])
    lat0 = float(np.mean(coords[:, 1]))
    ax.set_aspect(1.0 / max(np.cos(np.deg2rad(lat0)), 1e-3), adjustable="box")
    ax.margins(0.06)
    ax.legend(loc="upper right")

    return paths.save_figure(fig, paths.figures, base)


# ---------------------------------------------------------------------------
def export_97_views(model, builder, centers: List[int], coords: np.ndarray,
                    paths, cfg, device):
    """Exporta las 6 vistas DIRIGIDAS para cada paso t en views_97steps/."""
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
