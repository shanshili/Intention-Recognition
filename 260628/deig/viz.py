"""
Visualisation (candidate graph + dynamic intent graphs).

Requirements honoured:
  * every output filename embeds a run timestamp;
  * every figure is saved simultaneously as .png, .svg and .pdf.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np


def save_all_formats(fig, out_dir, stem, timestamp):
    """Save `fig` as png + svg + pdf with timestamped names. Returns paths."""
    paths = []
    for ext in ("png", "svg", "pdf"):
        name = f"{stem}_{timestamp}.{ext}"
        p = os.path.join(out_dir, name)
        fig.savefig(p, bbox_inches="tight", dpi=300)
        paths.append(p)
    plt.close(fig)
    return paths


def _draw_directed_edges(ax, coords, src, dst, strength, cmap, norm,
                         base_lw=0.6, max_lw=3.2, alpha_floor=0.15):
    """Draw arrows i->j with colour AND alpha encoding intent strength."""
    for e in range(len(src)):
        i, j = src[e], dst[e]
        s = strength[e]
        color = cmap(norm(s))
        alpha = alpha_floor + (1 - alpha_floor) * norm(s)
        lw = base_lw + (max_lw - base_lw) * norm(s)
        arrow = FancyArrowPatch(
            coords[i], coords[j],
            arrowstyle="-|>", mutation_scale=8,
            connectionstyle="arc3,rad=0.08",
            color=color, alpha=float(np.clip(alpha, 0, 1)),
            linewidth=lw, zorder=2,
        )
        ax.add_patch(arrow)


def plot_candidate_graph(coords, cand, out_dir, timestamp, chains=None):
    fig, ax = plt.subplots(figsize=(8, 7))
    src, dst = cand["src"], cand["dst"]
    # candidate edges in light grey
    segs = [[coords[i], coords[j]] for i, j in zip(src, dst)]
    lc = LineCollection(segs, colors="0.7", linewidths=0.5, alpha=0.6, zorder=1)
    ax.add_collection(lc)
    # direction hint with tiny arrows
    for i, j in zip(src, dst):
        ax.annotate("", xy=coords[j], xytext=coords[i],
                    arrowprops=dict(arrowstyle="-|>", color="0.55",
                                    alpha=0.5, lw=0.4,
                                    connectionstyle="arc3,rad=0.08"),
                    zorder=1)
    # ground-truth corridors (if known) overlaid faintly
    if chains:
        for ch in chains:
            pts = coords[ch]
            ax.plot(pts[:, 0], pts[:, 1], "-", color="tab:orange",
                    lw=1.2, alpha=0.35, zorder=1)
    ax.scatter(coords[:, 0], coords[:, 1], s=120, c="tab:blue",
               edgecolors="white", linewidths=1.0, zorder=3)
    for n, (x, y) in enumerate(coords):
        ax.text(x, y, str(n), ha="center", va="center",
                fontsize=7, color="white", zorder=4)
    ax.set_title(f"Candidate Directed Graph  G0  "
                 f"(N={coords.shape[0]}, |E0|={cand['E']})")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.2)
    return save_all_formats(fig, out_dir, "candidate_directed_graph", timestamp)


def plot_intent_windows(coords, cand, scores_list, window_labels,
                        out_dir, timestamp, q=0.90):
    """One multi-panel figure with the final intent graph of each window.

    Directed edges = intent; colour/alpha shade = intent strength.
    """
    from .intent_graph import build_intent_graph

    n = len(scores_list)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.8 * nrows))
    axes = np.atleast_1d(axes).ravel()

    cmap = cm.get_cmap("plasma")
    # shared colour scale across panels for comparability
    all_kept = []
    graphs = []
    for s in scores_list:
        g = build_intent_graph(s, cand, q=q)
        graphs.append(g)
        all_kept.append(g["strength"])
    flat = np.concatenate([a for a in all_kept if a.size]) if any(
        a.size for a in all_kept) else np.array([0.0, 1.0])
    norm = mcolors.Normalize(vmin=flat.min(), vmax=flat.max() + 1e-9)

    for k in range(len(axes)):
        ax = axes[k]
        if k >= n:
            ax.axis("off")
            continue
        g = graphs[k]
        ax.scatter(coords[:, 0], coords[:, 1], s=45, c="0.85",
                   edgecolors="0.5", linewidths=0.4, zorder=1)
        if g["edges"].size:
            _draw_directed_edges(ax, coords, g["edges"][:, 0], g["edges"][:, 1],
                                 g["strength"], cmap, norm)
        ax.set_title(f"{window_labels[k]}\n"
                     f"|E_I|={len(g['strength'])}  δ={g['delta']:.3g}",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("intent strength  s_ij")
    fig.suptitle(
        f"Final Intent Graphs over {n} Consecutive Prediction Windows "
        "(directed edges; shade = intent strength)",
        fontsize=13, y=1.0)
    return save_all_formats(fig, out_dir, "intent_graphs_10windows", timestamp)


def plot_single_intent_window(coords, cand, scores, label, out_dir,
                              timestamp, q=0.90, idx=0):
    from .intent_graph import build_intent_graph
    g = build_intent_graph(scores, cand, q=q)
    fig, ax = plt.subplots(figsize=(7, 6.2))
    cmap = cm.get_cmap("plasma")
    strength = g["strength"] if g["strength"].size else np.array([0.0, 1.0])
    norm = mcolors.Normalize(vmin=strength.min(), vmax=strength.max() + 1e-9)
    ax.scatter(coords[:, 0], coords[:, 1], s=90, c="0.85",
               edgecolors="0.5", linewidths=0.6, zorder=1)
    if g["edges"].size:
        _draw_directed_edges(ax, coords, g["edges"][:, 0], g["edges"][:, 1],
                             g["strength"], cmap, norm)
    for nn, (x, y) in enumerate(coords):
        ax.text(x, y, str(nn), ha="center", va="center", fontsize=6,
                color="0.3", zorder=4)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("intent strength  s_ij")
    ax.set_title(f"Intent Graph — {label}  (|E_I|={len(g['strength'])})")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.2)
    return save_all_formats(fig, out_dir, f"intent_graph_window_{idx:02d}",
                            timestamp)


def plot_training_curve(history, out_dir, timestamp):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["pred"], label="L_pred (Huber)")
    ax.plot(history["total"], label="L_total", linestyle="--")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("DEIG-TCN training loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    return save_all_formats(fig, out_dir, "training_loss", timestamp)


def plot_edge_mask_ablation(results, out_dir, timestamp):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    labels = ["mask top-k", "mask random", "mask low-k", "full model"]
    vals = [results["mask_top"], results["mask_random"],
            results["mask_low"], results["full"]]
    colors = ["tab:red", "tab:gray", "tab:green", "tab:blue"]
    ax.bar(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("test MAE (standardised)")
    ax.set_title("High-intent edge masking experiment\n"
                 "(masking top intent edges should hurt prediction most)")
    ax.grid(True, axis="y", alpha=0.3)
    return save_all_formats(fig, out_dir, "edge_mask_ablation", timestamp)


def plot_intent_density(densities, labels, out_dir, timestamp):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(range(len(densities)), densities, "o-", color="tab:purple")
    n = len(densities)
    # 最多显示约12个横轴标签，97个连续窗口时仍保持可读。
    tick_step = max(1, int(np.ceil(n / 12)))
    tick_idx = np.arange(0, n, tick_step)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([labels[i] for i in tick_idx],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("intent graph density")
    ax.set_title(f"Dynamic intent-graph density across {len(densities)} consecutive windows")
    ax.grid(True, alpha=0.3)
    return save_all_formats(fig, out_dir, "intent_graph_density", timestamp)
