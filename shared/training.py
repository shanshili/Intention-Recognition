"""
Shared training loop with configurable early stopping and loss weighting.
"""

import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


def train_model(model, train_loader, val_loader, adj, epochs, lr, device,
                output_dir, timestamp, patience=None,
                use_adaptive_loss=False, weight_cls=1.0, weight_pred=1.0):
    """Unified training loop.

    Parameters
    ----------
    patience : int or None
        Early stopping patience. None disables early stopping (260601 mode).
    use_adaptive_loss : bool
        Use learned log-variance weighting (260604 mode).
    weight_cls, weight_pred : float
        Fixed loss weights when ``use_adaptive_loss=False``.
    """
    model.to(device)
    adj = adj.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_pred = nn.MSELoss()

    log_var_cls = None
    log_var_pred = None
    if use_adaptive_loss:
        log_var_cls = nn.Parameter(torch.zeros(1, device=device))
        log_var_pred = nn.Parameter(torch.zeros(1, device=device))
        optimizer.add_param_group({"params": [log_var_cls, log_var_pred], "lr": lr * 0.1})

    train_losses, val_losses = [], []
    train_cls_losses, train_pred_losses = [], []
    val_cls_losses, val_pred_losses = [], []
    val_accuracies = []

    best_val_loss = float("inf")
    best_epoch = 0
    best_model_path = os.path.join(output_dir, f"best_model_{timestamp}.pth")
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_cls = 0.0
        total_pred = 0.0
        for x, y_cls, y_pred in train_loader:
            x, y_cls, y_pred = x.to(device), y_cls.to(device), y_pred.to(device)
            _, logits, pred_flow = model(x, adj)
            loss_cls = criterion_cls(logits, y_cls)
            loss_pred = criterion_pred(pred_flow, y_pred)

            if use_adaptive_loss:
                lv_cls = torch.clamp(log_var_cls, -5, 5)
                lv_pred = torch.clamp(log_var_pred, -5, 5)
                precision_cls = torch.exp(-lv_cls)
                precision_pred = torch.exp(-lv_pred)
                loss = precision_cls * loss_cls + lv_cls + precision_pred * loss_pred + lv_pred
            else:
                loss = weight_cls * loss_cls + weight_pred * loss_pred

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cls += loss_cls.item()
            total_pred += loss_pred.item()

        n_batches = len(train_loader)
        avg_train_loss = total_loss / n_batches
        avg_train_cls = total_cls / n_batches
        avg_train_pred = total_pred / n_batches
        train_losses.append(avg_train_loss)
        train_cls_losses.append(avg_train_cls)
        train_pred_losses.append(avg_train_pred)

        # Validation
        model.eval()
        val_loss = 0.0
        val_cls_loss = 0.0
        val_pred_loss = 0.0
        all_preds, all_true = [], []
        with torch.no_grad():
            for x, y_cls, y_pred in val_loader:
                x, y_cls, y_pred = x.to(device), y_cls.to(device), y_pred.to(device)
                _, logits, pred_flow = model(x, adj)
                loss_cls = criterion_cls(logits, y_cls)
                loss_pred = criterion_pred(pred_flow, y_pred)
                if use_adaptive_loss:
                    loss = loss_cls + loss_pred
                else:
                    loss = weight_cls * loss_cls + weight_pred * loss_pred
                val_loss += loss.item()
                val_cls_loss += loss_cls.item()
                val_pred_loss += loss_pred.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(y_cls.cpu().numpy())

        n_val = len(val_loader)
        avg_val_loss = val_loss / n_val
        avg_val_cls = val_cls_loss / n_val
        avg_val_pred = val_pred_loss / n_val
        val_acc = accuracy_score(all_true, all_preds)
        val_losses.append(avg_val_loss)
        val_cls_losses.append(avg_val_cls)
        val_pred_losses.append(avg_val_pred)
        val_accuracies.append(val_acc)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} (cls:{avg_train_cls:.3f} pred:{avg_train_pred:.3f}) | "
                f"Val Loss: {avg_val_loss:.4f} (cls:{avg_val_cls:.3f} pred:{avg_val_pred:.3f}) | "
                f"Val Acc: {val_acc:.4f}"
            )

        if patience is not None and no_improve >= patience:
            print(f"早停触发，最佳epoch: {best_epoch+1}, val_loss: {best_val_loss:.4f}")
            break

    print(f"训练完成，最佳模型保存在 {best_model_path}")
    return (
        train_losses, val_losses,
        train_cls_losses, val_cls_losses,
        train_pred_losses, val_pred_losses,
        val_accuracies, best_model_path,
    )
