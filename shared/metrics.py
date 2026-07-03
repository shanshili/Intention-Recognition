"""
Shared metric computation utilities.
"""

import os
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error


def compute_metrics(all_true, all_pred, output_dir, timestamp):
    mse = mean_squared_error(all_true, all_pred)
    mae = mean_absolute_error(all_true, all_pred)
    print(f"Prediction Metrics on Test Set: MSE={mse:.6f}, MAE={mae:.6f}")
    with open(os.path.join(output_dir, f"metrics_{timestamp}.txt"), "w") as f:
        f.write(f"Prediction MSE: {mse:.6f}\n")
        f.write(f"Prediction MAE: {mae:.6f}\n")
    return mse, mae


def compute_persistence_metrics(loader, pred_len, target_idx=0):
    all_true, all_pred_base = [], []
    for x, _, y_true in loader:
        last_hist = x[:, target_idx, -1].unsqueeze(1)
        pred = last_hist.repeat(1, pred_len)
        all_true.append(y_true.numpy())
        all_pred_base.append(pred.numpy())
    all_true = np.concatenate(all_true, axis=0)
    all_pred_base = np.concatenate(all_pred_base, axis=0)
    mse = mean_squared_error(all_true, all_pred_base)
    mae = mean_absolute_error(all_true, all_pred_base)
    return mse, mae


def collect_predictions(model, loader, adj, device):
    """Collect all test predictions and ground truths."""
    model.eval()
    if not isinstance(adj, torch.Tensor):
        adj = torch.FloatTensor(adj)
    adj = adj.to(device)
    all_true, all_pred = [], []
    with torch.no_grad():
        for x, _, y_pred in loader:
            x = x.to(device)
            _, _, pred_flow = model(x, adj)
            all_true.append(y_pred.cpu().numpy())
            all_pred.append(pred_flow.cpu().numpy())
    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    return all_true, all_pred
