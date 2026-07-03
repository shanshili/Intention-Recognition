"""
Shared visualization utilities for intent recognition experiments.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report


def save_figure(fig, base_name, output_dir, timestamp, dpi=600):
    for ext in ["png", "svg", "pdf"]:
        filepath = os.path.join(output_dir, f"{base_name}_{timestamp}.{ext}")
        fig.savefig(filepath, dpi=dpi, format=ext, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_graph(adj, coords, node_ids, output_dir, timestamp,
                           corr_threshold=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    edges = np.argwhere(adj > 0)
    for i, j in edges:
        ax.plot(
            [coords[i, 0], coords[j, 0]],
            [coords[i, 1], coords[j, 1]],
            "k-", alpha=0.3, linewidth=0.5,
        )
    ax.scatter(coords[:, 0], coords[:, 1], c="red", s=20, zorder=5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    title = "Correlation Graph"
    if corr_threshold is not None:
        title += f" (threshold={corr_threshold})"
    ax.set_title(title)
    save_figure(fig, "correlation_graph", output_dir, timestamp)


def plot_loss_curves(train_losses, val_losses, train_cls, val_cls,
                     train_pred, val_pred, output_dir, timestamp):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(train_losses, label="Train")
    axes[0].plot(val_losses, label="Val")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(train_cls, label="Train")
    axes[1].plot(val_cls, label="Val")
    axes[1].set_title("Classification Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[2].plot(train_pred, label="Train")
    axes[2].plot(val_pred, label="Val")
    axes[2].set_title("Prediction Loss (MSE)")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    plt.tight_layout()
    save_figure(fig, "loss_curves", output_dir, timestamp)


def visualize_intent_tsne(model, dataloader, adj, device, output_dir,
                          timestamp, class_names=None):
    """t-SNE visualization with dynamic perplexity."""
    model.eval()
    if not isinstance(adj, torch.Tensor):
        adj = torch.FloatTensor(adj)
    adj = adj.to(device)
    intent_vecs, true_labels = [], []
    with torch.no_grad():
        for x, y_cls, _ in dataloader:
            x = x.to(device)
            intent, _, _ = model(x, adj)
            intent_vecs.append(intent.cpu().numpy())
            true_labels.append(y_cls.numpy() if isinstance(y_cls, torch.Tensor) else y_cls)
    intent_vecs = np.concatenate(intent_vecs, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    n = len(intent_vecs)
    perp = min(30, max(5, n // 3))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
    vec_2d = tsne.fit_transform(intent_vecs)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(vec_2d[:, 0], vec_2d[:, 1], c=true_labels,
                         cmap="viridis", alpha=0.6)
    if class_names is None:
        class_names = ["Class 0", "Class 1", "Class 2"]
    cbar = plt.colorbar(scatter, ticks=list(range(len(class_names))),
                        label="Class")
    cbar.ax.set_yticklabels(class_names)
    ax.set_title("t-SNE of Intent Vectors")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    save_figure(fig, "tsne_intent", output_dir, timestamp)


def plot_confusion_matrix(model, dataloader, adj, device, output_dir,
                          timestamp, class_names=None):
    model.eval()
    if not isinstance(adj, torch.Tensor):
        adj = torch.FloatTensor(adj)
    adj = adj.to(device)
    all_preds, all_true = [], []
    with torch.no_grad():
        for x, y_cls, _ in dataloader:
            x = x.to(device)
            _, logits, _ = model(x, adj)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(y_cls.numpy() if isinstance(y_cls, torch.Tensor) else y_cls)
    cm = confusion_matrix(all_true, all_preds)
    if class_names is None:
        class_names = ["Class 0", "Class 1", "Class 2"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    save_figure(fig, "confusion_matrix", output_dir, timestamp)
    report = classification_report(all_true, all_preds, target_names=class_names)
    with open(os.path.join(output_dir, f"classification_report_{timestamp}.txt"), "w") as f:
        f.write(report)
    print(report)
    return all_true, all_preds


def plot_predictions(model, test_loader, adj, device, output_dir, timestamp,
                     pred_len, target_idx=None, num_samples=3,
                     include_persistence=False):
    """Unified prediction comparison plot.

    Parameters
    ----------
    target_idx : int or None
        If not None, also compute and plot persistence baseline.
    include_persistence : bool
        Whether to include persistence baseline in the average plot.
    """
    model.eval()
    if not isinstance(adj, torch.Tensor):
        adj = torch.FloatTensor(adj)
    adj = adj.to(device)
    all_true, all_pred_model = [], []
    all_pred_pers = [] if include_persistence else None
    with torch.no_grad():
        for x, _, y_pred in test_loader:
            x = x.to(device)
            _, _, pred_flow = model(x, adj)
            all_true.append(y_pred.cpu().numpy())
            all_pred_model.append(pred_flow.cpu().numpy())
            if include_persistence and target_idx is not None:
                last = x[:, target_idx, -1].unsqueeze(1).cpu().numpy()
                all_pred_pers.append(np.repeat(last, pred_len, axis=1))
    all_true = np.concatenate(all_true, axis=0)
    all_pred_model = np.concatenate(all_pred_model, axis=0)

    mean_true = all_true.mean(axis=0)
    mean_model = all_pred_model.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(range(pred_len), mean_true, "b-o", label="True (avg)")
    axes[0].plot(range(pred_len), mean_model, "r--s", label="Model (avg)")
    if include_persistence and all_pred_pers:
        all_pred_pers = np.concatenate(all_pred_pers, axis=0)
        mean_pers = all_pred_pers.mean(axis=0)
        axes[0].plot(range(pred_len), mean_pers, "g--x", label="Persistence")
    axes[0].set_title("Average Prediction over Test Set")
    axes[0].set_xlabel("Future Steps")
    axes[0].set_ylabel("Normalized Flow")
    axes[0].legend()

    indices = np.random.choice(len(all_true), num_samples, replace=False)
    for i, idx in enumerate(indices):
        axes[1].plot(range(pred_len), all_true[idx], "--o", label=f"True {i+1}")
        axes[1].plot(range(pred_len), all_pred_model[idx], "-s", label=f"Model {i+1}")
    axes[1].set_title("Individual Sample Predictions")
    axes[1].set_xlabel("Future Steps")
    axes[1].legend()
    plt.tight_layout()
    save_figure(fig, "prediction_comparison", output_dir, timestamp)
