"""Static plots adapted from 260628/deig/viz.py; render stored arrays only.

Retains PNG/SVG/PDF saving and directed arrows. Shared robust positive-score
log normalization replaces independent linear scales and zero-score drawing.
"""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import FancyArrowPatch
import numpy as np


def save_all_formats(fig, out_dir, stem, timestamp):
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("png", "svg", "pdf"):
        path = destination / f"{stem}_{timestamp}.{extension}"
        fig.savefig(path, bbox_inches="tight", dpi=180)
        paths.append(str(path.resolve()))
    plt.close(fig)
    return paths


def _positive_norm(scores):
    positive = np.asarray(scores)[np.isfinite(scores) & (np.asarray(scores) > 0)]
    if not len(positive):
        return colors.LogNorm(vmin=1e-12, vmax=1, clip=True), False
    low, high = np.quantile(positive, [0.01, 0.99])
    if high <= low:
        low, high = max(float(low) * 0.5, np.finfo(float).tiny), float(high) * 1.5
    return colors.LogNorm(vmin=max(float(low), np.finfo(float).tiny),
                          vmax=float(high), clip=True), True


def _draw_graph(ax, coords, edges, scores, keep, norm):
    ax.scatter(coords[:, 0], coords[:, 1], s=18, color="0.65", zorder=3)
    cmap = plt.get_cmap("plasma")
    drawn = keep & np.isfinite(scores) & (scores > 0)
    for edge in np.flatnonzero(drawn):
        i, j = edges[edge]
        magnitude = float(norm(scores[edge]))
        arrow = FancyArrowPatch(coords[i], coords[j], arrowstyle="-|>", mutation_scale=7,
                                connectionstyle="arc3,rad=0.08", color=cmap(magnitude),
                                linewidth=0.5 + 1.7 * magnitude, alpha=0.25 + 0.7 * magnitude,
                                zorder=2)
        ax.add_patch(arrow)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    return int(keep.sum()), int(drawn.sum())


def plot_gate_windows(coords, edges, scores, times, output, timestamp, threshold=None, maxcount=3):
    if scores.ndim != 2 or len(scores) == 0 or scores.shape[1] == 0:
        return []
    picks = np.unique(np.linspace(0, len(scores) - 1, min(maxcount, len(scores))).astype(int))
    norm, has_positive = _positive_norm(scores)
    modes = [("top10", None)]
    if threshold is not None and np.isfinite(threshold):
        modes.append(("absolute_threshold", float(threshold)))
    paths = []
    E = scores.shape[1]
    k = min(E, max(1, int(np.ceil(E * 0.1))))
    for mode, value in modes:
        fig, axes = plt.subplots(1, len(picks), figsize=(5 * len(picks), 4.4), squeeze=False)
        fig.subplots_adjust(top=0.78, bottom=0.06, wspace=0.22)
        for ax, row in zip(axes.ravel(), picks):
            current = scores[row]
            if mode == "top10":
                keep = np.zeros(E, dtype=bool)
                keep[np.argsort(-current, kind="stable")[:k]] = True
            else:
                keep = np.isfinite(current) & (current > 0) & (current >= value)
            selected, drawn = _draw_graph(ax, coords, edges, current, keep, norm)
            label = str(times[row, 0]).replace("T", " ")[:19]
            ax.set_title(f"{label}\nselected={selected}; positive drawn={drawn}", fontsize=9)
            if drawn == 0:
                ax.text(0.5, 0.04, "No positive selected gates", transform=ax.transAxes,
                        ha="center", fontsize=9)
        fig.suptitle("Exact Top-10% gate ranking" if mode == "top10"
                     else f"Validation absolute threshold: {value:.4g}", y=0.98)
        if has_positive:
            mappable = plt.cm.ScalarMappable(norm=norm, cmap="plasma")
            fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.7,
                         label="Gate score (shared log scale; 1%-99% clipped)")
        paths += save_all_formats(fig, output, f"gate_{mode}", timestamp)
    return paths


def make_run_plots(output_dir, fit, evaluation, data):
    destination = Path(output_dir)
    output = destination / "figures"
    timestamp = re.sub(r"[^A-Za-z0-9_-]", "_", destination.name)
    paths = []
    history = fit.get("history", [])
    epoch_records = [record for record in history if isinstance(record, dict) and "epoch" in record]
    if epoch_records:
        fig, ax = plt.subplots(figsize=(6, 4))
        epochs = [record["epoch"] for record in epoch_records]
        ax.plot(epochs, [record.get("val_mae", np.nan) for record in epoch_records], label="Validation MAE")
        train_values = [(record.get("train_loss") or {}).get("prediction", np.nan) for record in epoch_records]
        ax.plot(epochs, train_values, label="Train prediction loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Standardized error / loss")
        ax.grid(alpha=0.25)
        ax.legend()
        paths += save_all_formats(fig, output, "training_curve", timestamp)
    metrics = evaluation["metrics"]
    per_horizon = metrics.get("per_horizon_mae", [])
    if per_horizon:
        fig, ax = plt.subplots(figsize=(6, 4))
        horizons = np.arange(1, len(per_horizon) + 1) * data.protocol["step_minutes"]
        ax.plot(horizons, per_horizon, marker="o")
        ax.set_xlabel("Forecast horizon (minutes)")
        ax.set_ylabel("MAE (raw units)")
        ax.grid(alpha=0.25)
        paths += save_all_formats(fig, output, "per_horizon_mae", timestamp)
    graph_metrics = evaluation.get("graph_metrics", {})
    budgets = graph_metrics.get("budgets", [])
    if budgets and all("delete_top" in entry for entry in budgets):
        fig, ax = plt.subplots(figsize=(7, 4))
        fractions = np.asarray([entry["fraction"] for entry in budgets]) * 100
        base = metrics["mae"]
        for key, label in [("delete_top", "Delete top"), ("delete_low", "Delete low")]:
            values = [entry.get(key, np.nan) for entry in budgets]
            if all(np.isscalar(value) for value in values):
                ax.plot(fractions, np.asarray(values, dtype=float) - base, marker="o", label=label)
        means = np.asarray([entry.get("delete_random_mean", np.nan) for entry in budgets], dtype=float)
        errors = np.asarray([entry.get("delete_random_std") or 0 for entry in budgets], dtype=float)
        ax.errorbar(fractions, means - base, yerr=errors, marker="o", capsize=3,
                    label="Delete random (seed SD)")
        ax.axhline(0, color="0.5", linewidth=0.8)
        ax.set_xlabel("Candidate edges removed (%)")
        ax.set_ylabel("MAE change (raw units)")
        ax.grid(alpha=0.25)
        ax.legend()
        paths += save_all_formats(fig, output, "edge_deletion_delta", timestamp)
    arrays = evaluation["arrays"]
    scores = arrays.get("edge_scores")
    if scores is not None and graph_metrics.get("available"):
        threshold = graph_metrics.get("threshold", {})
        value = threshold.get("value") if threshold.get("enabled") else None
        paths += plot_gate_windows(data.coords, data.graph["edges"], np.asarray(scores),
                                   arrays["target_times"], output, timestamp, threshold=value,
                                   maxcount=evaluation.get("evaluation_config", {}).get("max_plot_windows", 3))
    return paths
