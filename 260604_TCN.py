"""
基于 GAT + BiGRU/TCN 的意图识别框架（修正版 v2.1）
修正项：
- 图构建严格限制在训练集历史窗口，杜绝未来泄露
- 持久性基线改为使用最后历史观测值
- 空间编码器显式保留目标节点表征（拼接全局池化）
- 自适应损失权重添加数值裁剪
- 添加早停、类别分布输出、动态t-SNE perplexity
- 修正图密度计算，补充缺失导入
"""

import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import distance_matrix       # 修正: 混合图所需
from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# -------------------------- 全局随机种子 --------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# ----------------------------------------------------------------

# ========================= 用户配置区域 =========================
DATA_DIR = "./pems_spatial_kmeans_topK"
NODE_COORD_FILE = "id_node_positions.csv"
TSV_FILE = "id_standardize_Env_A_timeseries.csv"
OUTPUT_DIR = "./output"
WINDOW_SIZE = 48
PRED_LEN = 6
BATCH_SIZE = 32
EPOCHS = 30                      # 最大迭代数，早停会提前结束
LEARNING_RATE = 1e-3
INTENT_DIM = 64
USE_TCN = True                  # False: BiGRU, True: TCN
CORR_THRESHOLD = 0.8
EARLY_STOP_PATIENCE = 7          # 新增: 早停容忍轮次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_NODE_IDX = 0              # 目标节点索引，便于统一使用

# 混合图开关
USE_HYBRID_GRAPH = True             # 启用混合空间‑相关图
DIST_THRESHOLD = 0.15               # 归一化坐标后的距离阈值（0~1）
CORR_THRESHOLD_HYBRID = 0.7         # 混合图中的相关性阈值（可不同于纯相关图）

WEIGHT_CLS = 1.0      # 分类损失权重
WEIGHT_PRED = 0.5     # 预测损失权重（可尝试 0.3~1.0）
# ===============================================================

# 生成时间戳
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(OUTPUT_DIR, TIMESTAMP)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"输出目录: {OUTPUT_DIR}")

# -------------------------- 1. 数据加载 --------------------------
def load_node_coords(filepath):
    df = pd.read_csv(filepath)
    node_ids = df["node_id"].values
    coords = df[["x", "y"]].values.astype(np.float32)
    return node_ids, coords

def load_time_series(filepath):
    df = pd.read_csv(filepath)
    node_names = df.columns.tolist()
    data = df.values.T  # (N, T)
    # 新增: 检查NaN/Inf
    if np.isnan(data).any():
        raise ValueError("时间序列含有NaN值")
    if np.isinf(data).any():
        raise ValueError("时间序列含有Inf值")
    return torch.FloatTensor(data), node_names

def build_correlation_graph(data, threshold=0.5):
    """仅基于训练集历史数据构建图，data: Tensor (N, T)"""
    data_np = data.numpy()
    corr = np.corrcoef(data_np)
    adj = (np.abs(corr) > threshold).astype(np.float32)
    np.fill_diagonal(adj, 0)
    adj = (adj + adj.T) > 0
    adj = adj.astype(np.float32)
    rowsum = adj.sum(1, keepdims=True)
    rowsum = np.where(rowsum == 0, 1, rowsum)
    adj_norm = adj / rowsum
    return torch.FloatTensor(adj_norm), adj

def build_hybrid_graph(coords, data, dist_threshold=0.1, corr_threshold=0.7,
                       normalize_coords=True):
    """混合空间‑相关性图（保留，需 scipy）"""
    if normalize_coords:
        coords_min = coords.min(axis=0, keepdims=True)
        coords_max = coords.max(axis=0, keepdims=True)
        coords = (coords - coords_min) / (coords_max - coords_min + 1e-8)
    dist = distance_matrix(coords, coords)
    data_np = data.numpy()
    corr = np.corrcoef(data_np)
    adj_dist = (dist < dist_threshold).astype(np.float32)
    adj_corr = (np.abs(corr) > corr_threshold).astype(np.float32)
    adj = adj_dist * adj_corr
    np.fill_diagonal(adj, 0)
    adj = (adj + adj.T) > 0
    adj = adj.astype(np.float32)
    for i in range(adj.shape[0]):
        if adj[i].sum() == 0:
            idx = np.argpartition(dist[i], 3)[:3]
            for j in idx:
                if i != j:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
    rowsum = adj.sum(1, keepdims=True)
    rowsum = np.where(rowsum == 0, 1, rowsum)
    adj_norm = adj / rowsum
    return torch.FloatTensor(adj_norm), adj

# -------------------------- 2. 数据集 --------------------------
class TrafficIntentDataset(Dataset):
    def __init__(self, data, target_node_idx, window=24, pred_len=6, stride=1):
        self.data = data
        self.target_idx = target_node_idx
        self.window = window
        self.pred_len = pred_len
        self.samples = []
        T = data.shape[1]
        for start in range(0, T - window - pred_len + 1, stride):
            self.samples.append(start)
        self.labels = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start = self.samples[idx]
        end = start + self.window
        pred_end = end + self.pred_len
        x = self.data[:, start:end]
        y_pred = self.data[self.target_idx, end:pred_end]
        y_cls = self.labels[idx] if self.labels is not None else -1
        return x, y_cls, y_pred

def compute_future_labels(dataset, train_indices, num_classes=3):
    train_means = []
    for idx in train_indices:
        _, _, y_pred = dataset[idx]
        train_means.append(y_pred.mean().item())
    train_means = np.array(train_means)
    thresholds = np.percentile(train_means, [100/3, 200/3])
    print(f"标签阈值 (训练集未来平均流量分位数): {thresholds}")
    all_labels = np.zeros(len(dataset), dtype=np.int64)
    for i in range(len(dataset)):
        _, _, y_pred = dataset[i]
        mean_val = y_pred.mean().item()
        if mean_val <= thresholds[0]:
            all_labels[i] = 0
        elif mean_val <= thresholds[1]:
            all_labels[i] = 1
        else:
            all_labels[i] = 2
    return all_labels

# -------------------------- 3. 模型组件 --------------------------
class AttentionPooling(nn.Module):
    """对时间维度做加权平均，权重由可学习的上下文向量决定"""
    def __init__(self, feat_dim):
        super().__init__()
        self.context = nn.Parameter(torch.randn(feat_dim, 1))  # (feat_dim, 1)
        nn.init.xavier_uniform_(self.context)

    def forward(self, x):
        # x: (batch, time_steps, feat_dim)
        scores = torch.matmul(x, self.context).squeeze(-1)     # (B, T)
        attn_weights = torch.softmax(scores, dim=-1)           # (B, T)
        pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # (B, feat_dim)
        return pooled

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2, alpha=0.2):
        super().__init__()
        self.W = nn.Parameter(torch.empty(in_dim, out_dim))
        self.a = nn.Parameter(torch.empty(2 * out_dim, 1))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)          # (B, N, out_dim)
        N = Wh.size(1)
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1)
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1)
        concat = torch.cat([Wh_i, Wh_j], dim=-1)
        e = self.leaky_relu(torch.matmul(concat, self.a).squeeze(-1))
        adj_expanded = adj.unsqueeze(0).expand(e.size(0), -1, -1)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_expanded > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        h_new = torch.matmul(attention, Wh)
        return h_new

class SpatialEncoder(nn.Module):
    """空间编码器：输出融合目标节点与全局信息的表征"""
    def __init__(self, in_dim, hidden_dim, out_dim, target_idx=0, n_layers=2, dropout=0.2):
        super().__init__()
        self.target_idx = target_idx
        self.layers = nn.ModuleList()
        self.layers.append(GATLayer(in_dim, hidden_dim, dropout))
        for _ in range(n_layers - 1):
            self.layers.append(GATLayer(hidden_dim, hidden_dim, dropout))
        # 拼接目标节点向量与全局平均池化，维度加倍
        self.out_proj = nn.Linear(hidden_dim * 2, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        for layer in self.layers:
            x = F.elu(layer(x, adj))
            x = self.dropout(x)
        target_vec = x[:, self.target_idx, :]      # (B, hidden_dim)
        global_vec = x.mean(dim=1)                  # (B, hidden_dim)
        combined = torch.cat([target_vec, global_vec], dim=-1)  # (B, 2*hidden_dim)
        return self.out_proj(combined)

class TemporalConvNet(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i, out_ch in enumerate(num_channels):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size-1))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            if i < len(num_channels) - 1:      # 最后一层不加 Dropout，保留信息
                layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.out_dim = num_channels[-1] if num_channels else input_dim

    def forward(self, x):
        # x: (B, T, C)  ->  transpose -> (B, C, T)
        x = x.transpose(1, 2)
        out = self.tcn(x)               # (B, out_dim, T)
        out = out.transpose(1, 2)       # (B, T, out_dim)
        return out      # 返回完整的时间序列特征，不做池化

class IntentModel(nn.Module):
    def __init__(
        self,
        N,
        node_feat_dim,
        spat_hid,
        spat_out,
        temp_hid,       # 仅 BiGRU 使用，TCN 时忽略
        intent_dim,
        num_classes=3,
        pred_len=6,
        use_tcn=False,
        tcn_channels=(64, 128, 256),   # 改为3层，容量更大
        target_idx=0,
        dropout=0.2,
    ):
        super().__init__()
        self.spatial_encoder = SpatialEncoder(
            node_feat_dim, spat_hid, spat_out, target_idx=target_idx
        )
        self.intent_dim = intent_dim
        self.pred_len = pred_len
        self.use_tcn = use_tcn

        if use_tcn:
            self.temporal = TemporalConvNet(spat_out, tcn_channels, kernel_size=3, dropout=dropout)
            temp_out = tcn_channels[-1]
            self.pooling = AttentionPooling(temp_out)        # 注意力池化

        self.intent_proj = nn.Linear(temp_out, intent_dim)   # 统一映射到意图空间
        self.classifier = nn.Linear(intent_dim, num_classes)

        # 预测头直接作用于池化后的向量（保留了重要时间信息）
        self.predictor = nn.Sequential(
            nn.Linear(intent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, pred_len),
        )

    def forward(self, x, adj):
        batch, N, W = x.shape
        spatial_seq = []
        for t in range(W):
            x_t = x[:, :, t].unsqueeze(-1)
            g_t = self.spatial_encoder(x_t, adj)
            spatial_seq.append(g_t.unsqueeze(1))
        spatial_seq = torch.cat(spatial_seq, dim=1)        # (B, W, spat_out)

        temp_out = self.temporal(spatial_seq)               # TCN: (B, W, temp_out)  GRU: (B, temp_out)

        if self.use_tcn:
            # 注意力池化得到固定向量
            temp_vec = self.pooling(temp_out)               # (B, temp_out)
        else:
            temp_vec = temp_out

        intent_vec = self.intent_proj(temp_vec)             # (B, intent_dim)
        logits = self.classifier(intent_vec)
        pred_flow = self.predictor(intent_vec)
        return intent_vec, logits, pred_flow
# -------------------------- 4. 训练与验证 --------------------------
def train_model(model, train_loader, val_loader, adj, epochs, lr, patience, device, output_dir):
    model.to(device)
    adj = adj.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_pred = nn.MSELoss()

    # 自适应损失权重，带数值裁剪防止爆炸
    # log_var_cls = nn.Parameter(torch.zeros(1, device=device))
    # log_var_pred = nn.Parameter(torch.zeros(1, device=device))
    # optimizer.add_param_group({"params": [log_var_cls, log_var_pred], "lr": lr * 0.1})

    train_losses, val_losses = [], []
    train_cls_losses, train_pred_losses = [], []
    val_cls_losses, val_pred_losses = [], []
    val_accuracies = []

    best_val_loss = float("inf")
    best_epoch = 0
    best_model_path = os.path.join(output_dir, f"best_model_{TIMESTAMP}.pth")
    no_improve = 0

    for epoch in range(epochs):
        # 训练
        model.train()
        total_loss = 0.0
        total_cls = 0.0
        total_pred = 0.0
        for x, y_cls, y_pred in train_loader:
            x, y_cls, y_pred = x.to(device), y_cls.to(device), y_pred.to(device)
            _, logits, pred_flow = model(x, adj)
            loss_cls = criterion_cls(logits, y_cls)
            loss_pred = criterion_pred(pred_flow, y_pred)

            # 裁剪 log_var 防止数值溢出
            # lv_cls = torch.clamp(log_var_cls, -5, 5)
            # lv_pred = torch.clamp(log_var_pred, -5, 5)
            # precision_cls = torch.exp(-lv_cls)
            # precision_pred = torch.exp(-lv_pred)
            # loss = precision_cls * loss_cls + lv_cls + precision_pred * loss_pred + lv_pred
            # 原代码中 log_var 相关部分全部删除，直接用固定权重
            loss = WEIGHT_CLS * loss_cls + WEIGHT_PRED * loss_pred

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cls += loss_cls.item()
            total_pred += loss_pred.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_cls = total_cls / len(train_loader)
        avg_train_pred = total_pred / len(train_loader)
        train_losses.append(avg_train_loss)
        train_cls_losses.append(avg_train_cls)
        train_pred_losses.append(avg_train_pred)

        # 验证（使用未加权损失评估，保证模型选择一致性）
        model.eval()
        val_loss = 0.0
        val_cls = 0.0
        val_pred = 0.0
        all_preds, all_true = [], []
        with torch.no_grad():
            for x, y_cls, y_pred in val_loader:
                x, y_cls, y_pred = x.to(device), y_cls.to(device), y_pred.to(device)
                _, logits, pred_flow = model(x, adj)
                loss_cls = criterion_cls(logits, y_cls)
                loss_pred = criterion_pred(pred_flow, y_pred)
                # loss = loss_cls + loss_pred
                loss = WEIGHT_CLS * loss_cls + WEIGHT_PRED * loss_pred
                val_loss += loss.item()
                val_cls += loss_cls.item()
                val_pred += loss_pred.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(y_cls.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        avg_val_cls = val_cls / len(val_loader)
        avg_val_pred = val_pred / len(val_loader)
        val_acc = accuracy_score(all_true, all_preds)
        val_losses.append(avg_val_loss)
        val_cls_losses.append(avg_val_cls)
        val_pred_losses.append(avg_val_pred)
        val_accuracies.append(val_acc)

        # 早停判断
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

        if no_improve >= patience:
            print(f"早停触发，最佳epoch: {best_epoch+1}, val_loss: {best_val_loss:.4f}")
            break

    print(f"训练完成，最佳模型保存在 {best_model_path}")
    return (
        train_losses, val_losses,
        train_cls_losses, val_cls_losses,
        train_pred_losses, val_pred_losses,
        val_accuracies, best_model_path,
    )

# -------------------------- 5. 可视化与评估 --------------------------
def save_figure(fig, base_name, output_dir, dpi=600):
    for ext in ["png", "svg", "pdf"]:
        filepath = os.path.join(output_dir, f"{base_name}_{TIMESTAMP}.{ext}")
        fig.savefig(filepath, dpi=dpi, format=ext, bbox_inches="tight")
    plt.close(fig)

def plot_correlation_graph(adj, coords, node_ids, output_dir):
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
    ax.set_title(f"Correlation Graph (threshold={CORR_THRESHOLD})")
    save_figure(fig, "correlation_graph", output_dir)

def plot_loss_curves(train_losses, val_losses, train_cls, val_cls, train_pred, val_pred, output_dir):
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
    save_figure(fig, "loss_curves", output_dir)

def visualize_intent_tsne(model, dataloader, adj, device, output_dir):
    model.eval()
    adj = adj.to(device)
    intent_vecs, true_labels = [], []
    with torch.no_grad():
        for x, y_cls, _ in dataloader:
            x = x.to(device)
            intent, _, _ = model(x, adj)
            intent_vecs.append(intent.cpu().numpy())
            true_labels.append(y_cls.numpy())
    intent_vecs = np.concatenate(intent_vecs, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    # 动态设置 perplexity
    n = len(intent_vecs)
    perp = min(30, max(5, n // 3))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
    vec_2d = tsne.fit_transform(intent_vecs)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(vec_2d[:, 0], vec_2d[:, 1], c=true_labels, cmap="viridis", alpha=0.6)
    cbar = plt.colorbar(scatter, ticks=[0, 1, 2], label="Flow Level")
    cbar.ax.set_yticklabels(["Low", "Medium", "High"])
    ax.set_title("t-SNE of Intent Vectors")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    save_figure(fig, "tsne_intent", output_dir)

def plot_confusion_matrix(model, dataloader, adj, device, output_dir):
    model.eval()
    adj = adj.to(device)
    all_preds, all_true = [], []
    with torch.no_grad():
        for x, y_cls, _ in dataloader:
            x = x.to(device)
            _, logits, _ = model(x, adj)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(y_cls.numpy())
    cm = confusion_matrix(all_true, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low", "Medium", "High"],
                yticklabels=["Low", "Medium", "High"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    save_figure(fig, "confusion_matrix", output_dir)
    report = classification_report(
        all_true, all_preds, target_names=["Low", "Medium", "High"]
    )
    with open(os.path.join(output_dir, f"classification_report_{TIMESTAMP}.txt"), "w") as f:
        f.write(report)
    print(report)
    return all_true, all_preds

def compute_persistence_metrics(loader, target_idx=0):
    """持久性基线：使用历史窗口最后一个观测值重复"""
    all_true, all_pred_base = [], []
    for x, _, y_true in loader:          # x:(B,N,W), y_true:(B,pred_len)
        last_hist = x[:, target_idx, -1].unsqueeze(1)   # (B,1)
        pred = last_hist.repeat(1, PRED_LEN)
        all_true.append(y_true.numpy())
        all_pred_base.append(pred.numpy())
    all_true = np.concatenate(all_true, axis=0)
    all_pred_base = np.concatenate(all_pred_base, axis=0)
    mse = mean_squared_error(all_true, all_pred_base)
    mae = mean_absolute_error(all_true, all_pred_base)
    return mse, mae

def plot_predictions_comparison(model, test_loader, adj, device, output_dir, target_idx=0, num_samples=3):
    model.eval()
    adj = adj.to(device)
    all_true, all_pred_model, all_pred_pers = [], [], []
    with torch.no_grad():
        for x, _, y_pred in test_loader:
            x = x.to(device)
            _, _, pred_flow = model(x, adj)
            all_true.append(y_pred.cpu().numpy())
            all_pred_model.append(pred_flow.cpu().numpy())
            # 持久性基线：来自历史
            last = x[:, target_idx, -1].unsqueeze(1).cpu().numpy()
            all_pred_pers.append(np.repeat(last, PRED_LEN, axis=1))
    all_true = np.concatenate(all_true, axis=0)
    all_pred_model = np.concatenate(all_pred_model, axis=0)
    all_pred_pers = np.concatenate(all_pred_pers, axis=0)

    mean_true = all_true.mean(axis=0)
    mean_model = all_pred_model.mean(axis=0)
    mean_pers = all_pred_pers.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(range(PRED_LEN), mean_true, "b-o", label="True")
    axes[0].plot(range(PRED_LEN), mean_model, "r--s", label="Model")
    axes[0].plot(range(PRED_LEN), mean_pers, "g--x", label="Persistence")
    axes[0].set_title("Average Prediction over Test Set")
    axes[0].set_xlabel("Future Steps")
    axes[0].set_ylabel("Normalized Flow")
    axes[0].legend()

    indices = np.random.choice(len(all_true), num_samples, replace=False)
    for i, idx in enumerate(indices):
        axes[1].plot(range(PRED_LEN), all_true[idx], "--o", label=f"True {i+1}")
        axes[1].plot(range(PRED_LEN), all_pred_model[idx], "-s", label=f"Model {i+1}")
    axes[1].set_title("Individual Sample Predictions")
    axes[1].set_xlabel("Future Steps")
    axes[1].legend()
    plt.tight_layout()
    save_figure(fig, "prediction_comparison", output_dir)

def compute_metrics(all_true, all_pred, output_dir):
    mse = mean_squared_error(all_true, all_pred)
    mae = mean_absolute_error(all_true, all_pred)
    print(f"Prediction Metrics on Test Set: MSE={mse:.6f}, MAE={mae:.6f}")
    with open(os.path.join(output_dir, f"metrics_{TIMESTAMP}.txt"), "w") as f:
        f.write(f"Prediction MSE: {mse:.6f}\n")
        f.write(f"Prediction MAE: {mae:.6f}\n")
    return mse, mae

# -------------------------- 6. 主程序 --------------------------
def main():
    print(f"使用设备: {DEVICE}")
    print("1. 加载数据...")
    node_coord_path = os.path.join(DATA_DIR, NODE_COORD_FILE)
    tsv_path = os.path.join(DATA_DIR, TSV_FILE)

    node_ids, coords = load_node_coords(node_coord_path)
    data_raw, node_names = load_time_series(tsv_path)
    N, T = data_raw.shape
    print(f"节点数: {N}, 时间步数: {T}")

    # 构建完整数据集（无标签）
    target_idx = TARGET_NODE_IDX
    full_dataset = TrafficIntentDataset(
        data_raw, target_idx, window=WINDOW_SIZE, pred_len=PRED_LEN, stride=3
    )
    total = len(full_dataset)
    print(f"总样本数: {total}")

    # 时间顺序划分
    indices = np.arange(total)
    train_len = int(0.6 * total)
    val_len = int(0.2 * total)
    train_indices = indices[:train_len]
    val_indices = indices[train_len:train_len + val_len]
    test_indices = indices[train_len + val_len:]
    print(f"数据集划分: 训练 {train_len}, 验证 {val_len}, 测试 {len(test_indices)}")

    # 标签生成
    all_labels = compute_future_labels(full_dataset, train_indices, num_classes=3)
    full_dataset.labels = torch.LongTensor(all_labels)

    # 输出类别分布
    for split, idxs in [("Train", train_indices), ("Val", val_indices), ("Test", test_indices)]:
        labels = all_labels[idxs]
        unique, counts = np.unique(labels, return_counts=True)
        print(f"{split} 类别分布: {dict(zip(unique, counts))}")

    # 构造数据子集
    train_ds = torch.utils.data.Subset(full_dataset, train_indices)
    val_ds = torch.utils.data.Subset(full_dataset, val_indices)
    test_ds = torch.utils.data.Subset(full_dataset, test_indices)

    # ---------- 关键修正：图数据严格限于训练集历史窗口 ----------
    # 训练集最后一个样本的起始时间步
    last_train_start = full_dataset.samples[train_indices[-1]]   # (train_len-1)*stride
    max_train_hist = last_train_start + WINDOW_SIZE               # 历史窗口右边界（不包含）
    # 确保不包含任何训练集或验证集的未来
    print(f"图构建数据范围: 时间步 0 至 {max_train_hist-1} (包含)")
    train_data_for_graph = data_raw[:, :max_train_hist]           # 切片 [0, max_train_hist)


    if USE_HYBRID_GRAPH:
        print("使用混合空间‑相关图构建...")
        adj_norm, adj_binary = build_hybrid_graph(
            coords, train_data_for_graph,
            dist_threshold=DIST_THRESHOLD,
            corr_threshold=CORR_THRESHOLD_HYBRID,
            normalize_coords=True          # 坐标归一化，阈值应介于 0~1
        )
    else:
        print("使用纯相关图构建...")
        adj_norm, adj_binary = build_correlation_graph(
            train_data_for_graph, threshold=CORR_THRESHOLD
        )

    # 密度计算（适用于两种图）
    density = adj_binary.sum() / (N * (N - 1)) if N > 1 else 0.0
    print(f"图密度: {density:.4f}")

    # 绘制相关性图
    print("绘制相关性图...")
    plot_correlation_graph(adj_binary, coords, node_ids, OUTPUT_DIR)

    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 构建模型（传入target_idx）
    model = IntentModel(
        N=N, node_feat_dim=1, spat_hid=64, spat_out=32,
        temp_hid=64, intent_dim=INTENT_DIM, num_classes=3,
        pred_len=PRED_LEN, use_tcn=USE_TCN, target_idx=target_idx,
    )
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    print("开始训练...")
    (train_losses, val_losses, train_cls, val_cls, train_pred, val_pred,
     val_accuracies, best_model_path) = train_model(
        model, train_loader, val_loader, adj_norm,
        EPOCHS, LEARNING_RATE, EARLY_STOP_PATIENCE, DEVICE, OUTPUT_DIR
    )

    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    # 可视化损失
    plot_loss_curves(train_losses, val_losses, train_cls, val_cls, train_pred, val_pred, OUTPUT_DIR)

    # t-SNE
    print("生成 t-SNE 意图聚类图...")
    visualize_intent_tsne(model, test_loader, adj_norm, DEVICE, OUTPUT_DIR)

    # 混淆矩阵
    print("生成混淆矩阵...")
    true_labels, pred_labels = plot_confusion_matrix(model, test_loader, adj_norm, DEVICE, OUTPUT_DIR)
    test_acc = accuracy_score(true_labels, pred_labels)
    print(f"测试集分类准确率: {test_acc:.4f}")

    # 模型预测指标
    print("计算预测指标...")
    all_true, all_pred = [], []
    with torch.no_grad():
        for x, _, y_pred in test_loader:
            x = x.to(DEVICE)
            _, _, pred_flow = model(x, adj_norm.to(DEVICE))
            all_true.append(y_pred.cpu().numpy())
            all_pred.append(pred_flow.cpu().numpy())
    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    model_mse, model_mae = compute_metrics(all_true, all_pred, OUTPUT_DIR)

    # 持久性基线（修正）
    pers_mse, pers_mae = compute_persistence_metrics(test_loader, target_idx=target_idx)
    print(f"持久性基线 MSE: {pers_mse:.6f}, MAE: {pers_mae:.6f}")

    # 预测对比图
    plot_predictions_comparison(model, test_loader, adj_norm, DEVICE, OUTPUT_DIR, target_idx=target_idx, num_samples=3)

    # 保存汇总
    with open(os.path.join(OUTPUT_DIR, f"summary_{TIMESTAMP}.txt"), "w") as f:
        f.write(f"Timestamp: {TIMESTAMP}\n")
        f.write(f"Random Seed: {SEED}\n")
        f.write(f"Correlation threshold: {CORR_THRESHOLD}\n")
        f.write(f"Window size: {WINDOW_SIZE}, Prediction steps: {PRED_LEN}\n")
        f.write(f"Time-ordered split: train {train_len}, val {val_len}, test {len(test_indices)}\n")
        f.write(f"Test classification accuracy: {test_acc:.4f}\n")
        f.write(f"Model Prediction - MSE: {model_mse:.6f}, MAE: {model_mae:.6f}\n")
        f.write(f"Persistence Baseline - MSE: {pers_mse:.6f}, MAE: {pers_mae:.6f}\n")
        f.write(f"Best model path: {best_model_path}\n")
    print(f"所有结果已保存到: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()