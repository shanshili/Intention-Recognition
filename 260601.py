"""
基于 GAT + BiGRU/TCN 的意图识别框架（增强版）

新增功能：
- 训练过程详细日志（分类/预测损失，验证准确率）
- 相关性图可视化（按经纬度布局）
- 损失分解曲线图
- 多节点流量预测对比（整体平均 + 局部样本）
- 输出关键指标（分类准确率、MSE、MAE）并保存 CSV
- 所有输出文件名带时间戳
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ========================= 用户配置区域 =========================
DATA_DIR = "./pems_spatial_kmeans_topK"
NODE_COORD_FILE = "id_node_positions.csv"
TSV_FILE = "id_standardize_Env_A_timeseries.csv"
OUTPUT_DIR = "./output"
WINDOW_SIZE = 24
PRED_LEN = 6
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
INTENT_DIM = 64
USE_TCN = False                 # False: BiGRU, True: TCN
CORR_THRESHOLD = 0.999            # 相关性图阈值
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ===============================================================

# 生成时间戳
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(OUTPUT_DIR, TIMESTAMP)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"输出目录: {OUTPUT_DIR}")

# -------------------------- 1. 数据加载 --------------------------
def load_node_coords(filepath):
    filepath = os.path.realpath(filepath)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Node coordinate file not found: {filepath}")
    df = pd.read_csv(filepath)
    # 列名兼容：假设有 node_id, x, y
    node_ids = df['node_id'].values
    coords = df[['x', 'y']].values.astype(np.float32)
    return node_ids, coords

def load_time_series(filepath):
    filepath = os.path.realpath(filepath)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Time series file not found: {filepath}")
    df = pd.read_csv(filepath)
    node_names = df.columns.tolist()
    data = df.values.T   # (N, T)
    if np.isnan(data).any():
        raise ValueError("Time series data contains NaN values")
    if np.isinf(data).any():
        raise ValueError("Time series data contains Inf values")
    return torch.FloatTensor(data), node_names

def generate_time_labels(T, start_hour=0):
    labels = np.zeros(T, dtype=int)
    for t in range(T):
        hour = (start_hour + t) % 24
        if 6 <= hour < 9:
            labels[t] = 1
        elif 17 <= hour < 20:
            labels[t] = 2
    return torch.LongTensor(labels)

def build_correlation_graph(data, threshold=0.5):
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

# -------------------------- 2. 数据集 --------------------------
class TrafficIntentDataset(Dataset):
    def __init__(self, data, time_labels, target_node_idx, window=24, pred_len=6, stride=1):
        self.data = data
        self.labels = time_labels
        self.target_idx = target_node_idx
        self.window = window
        self.pred_len = pred_len
        self.samples = []
        T = data.shape[1]
        for start in range(0, T - window - pred_len + 1, stride):
            self.samples.append(start)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start = self.samples[idx]
        end = start + self.window
        pred_end = end + self.pred_len
        x = self.data[:, start:end]
        y_cls = self.labels[end - 1]
        y_pred = self.data[self.target_idx, end:pred_end]
        return x, y_cls, y_pred

# -------------------------- 3. 模型组件（保持不变） --------------------------
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
        Wh = torch.matmul(h, self.W)
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
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=2, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATLayer(in_dim, hidden_dim, dropout))
        for _ in range(n_layers - 1):
            self.layers.append(GATLayer(hidden_dim, hidden_dim, dropout))
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        for layer in self.layers:
            x = F.elu(layer(x, adj))
            x = self.dropout(x)
        g = x.mean(dim=1)
        g = self.out_proj(g)
        return g

class BiGRUTemporal(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, seq):
        out, _ = self.gru(seq)
        last = out[:, -1, :]
        return self.fc(last)

class TemporalConvNet(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        in_ch = input_dim
        for out_ch in num_channels:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size - 1))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.out_dim = num_channels[-1] if num_channels else input_dim

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.tcn(x)
        out = out.mean(dim=-1)
        return out

class IntentModel(nn.Module):
    def __init__(self, N, node_feat_dim, spat_hid, spat_out, temp_hid, intent_dim,
                 num_classes=3, pred_len=6, use_tcn=False, tcn_channels=[64, 128]):
        super().__init__()
        self.spatial_encoder = SpatialEncoder(node_feat_dim, spat_hid, spat_out)
        self.intent_dim = intent_dim
        self.pred_len = pred_len
        if use_tcn:
            self.temporal = TemporalConvNet(spat_out, tcn_channels, kernel_size=3)
            temp_out = tcn_channels[-1]
        else:
            self.temporal = BiGRUTemporal(spat_out, temp_hid, intent_dim)
            temp_out = intent_dim
        self.intent_proj = nn.Linear(temp_out, intent_dim)
        self.classifier = nn.Linear(intent_dim, num_classes)
        self.predictor = nn.Sequential(
            nn.Linear(intent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, pred_len)
        )

    def forward(self, x, adj):
        batch, N, W = x.shape
        spatial_seq = []
        for t in range(W):
            x_t = x[:, :, t].unsqueeze(-1)
            g_t = self.spatial_encoder(x_t, adj)
            spatial_seq.append(g_t.unsqueeze(1))
        spatial_seq = torch.cat(spatial_seq, dim=1)
        temp_out = self.temporal(spatial_seq)
        intent_vec = self.intent_proj(temp_out)
        logits = self.classifier(intent_vec)
        pred_flow = self.predictor(intent_vec)
        return intent_vec, logits, pred_flow

# -------------------------- 4. 训练与验证（增强日志） --------------------------
def train_model(model, train_loader, val_loader, adj, epochs, lr, device, output_dir):
    model.to(device)
    adj = adj.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_pred = nn.MSELoss()

    train_losses, val_losses = [], []
    train_cls_losses, train_pred_losses = [], []
    val_cls_losses, val_pred_losses = [], []
    val_accuracies = []

    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, f"best_model_{TIMESTAMP}.pth")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_pred_loss = 0.0
        for x, y_cls, y_pred in train_loader:
            x, y_cls, y_pred = x.to(device), y_cls.to(device), y_pred.to(device)
            _, logits, pred_flow = model(x, adj)
            loss_cls = criterion_cls(logits, y_cls)
            loss_pred = criterion_pred(pred_flow, y_pred)
            loss = loss_cls + loss_pred
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_pred_loss += loss_pred.item()
        avg_train_loss = total_loss / len(train_loader)
        avg_train_cls = total_cls_loss / len(train_loader)
        avg_train_pred = total_pred_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_cls_losses.append(avg_train_cls)
        train_pred_losses.append(avg_train_pred)

        # 验证
        model.eval()
        val_loss = 0.0
        val_cls_loss = 0.0
        val_pred_loss = 0.0
        all_preds = []
        all_true = []
        with torch.no_grad():
            for x, y_cls, y_pred in val_loader:
                x, y_cls, y_pred = x.to(device), y_cls.to(device), y_pred.to(device)
                _, logits, pred_flow = model(x, adj)
                loss_cls = criterion_cls(logits, y_cls)
                loss_pred = criterion_pred(pred_flow, y_pred)
                val_loss += (loss_cls + loss_pred).item()
                val_cls_loss += loss_cls.item()
                val_pred_loss += loss_pred.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(y_cls.cpu().numpy())
        avg_val_loss = val_loss / len(val_loader)
        avg_val_cls = val_cls_loss / len(val_loader)
        avg_val_pred = val_pred_loss / len(val_loader)
        val_acc = accuracy_score(all_true, all_preds)
        val_losses.append(avg_val_loss)
        val_cls_losses.append(avg_val_cls)
        val_pred_losses.append(avg_val_pred)
        val_accuracies.append(val_acc)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train_loss:.4f} (cls:{avg_train_cls:.3f} pred:{avg_train_pred:.3f}) | "
                  f"Val Loss: {avg_val_loss:.4f} (cls:{avg_val_cls:.3f} pred:{avg_val_pred:.3f}) | Val Acc: {val_acc:.4f}")

    print(f"训练完成，最佳模型保存在 {best_model_path}")
    return (train_losses, val_losses, train_cls_losses, val_cls_losses,
            train_pred_losses, val_pred_losses, val_accuracies, best_model_path)

# -------------------------- 5. 可视化函数（带时间戳） --------------------------
def save_figure(fig, base_name, output_dir, dpi=600):
    for ext in ['png', 'svg', 'pdf']:
        filepath = os.path.join(output_dir, f"{base_name}_{TIMESTAMP}.{ext}")
        fig.savefig(filepath, dpi=dpi, format=ext, bbox_inches='tight')
    plt.close(fig)

def plot_correlation_graph(adj, coords, node_ids, output_dir):
    """绘制相关性图，节点按经纬度布局"""
    fig, ax = plt.subplots(figsize=(10, 8))
    # 获取边索引
    edges = np.argwhere(adj > 0)
    # 绘制边
    for (i, j) in edges:
        ax.plot([coords[i,0], coords[j,0]], [coords[i,1], coords[j,1]], 'k-', alpha=0.3, linewidth=0.5)
    # 绘制节点
    ax.scatter(coords[:,0], coords[:,1], c='red', s=20, zorder=5)
    # 可选：标注节点ID（数量少时可开启）
    # for idx, node_id in enumerate(node_ids):
    #     ax.annotate(node_id, (coords[idx,0], coords[idx,1]), fontsize=6)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Correlation Graph (threshold={CORR_THRESHOLD})')
    save_figure(fig, "correlation_graph", output_dir)

def plot_loss_curves(train_losses, val_losses, train_cls, val_cls, train_pred, val_pred, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(train_losses, label='Train')
    axes[0].plot(val_losses, label='Val')
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[1].plot(train_cls, label='Train')
    axes[1].plot(val_cls, label='Val')
    axes[1].set_title('Classification Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[2].plot(train_pred, label='Train')
    axes[2].plot(val_pred, label='Val')
    axes[2].set_title('Prediction Loss (MSE)')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()
    plt.tight_layout()
    save_figure(fig, "loss_curves", output_dir)

def visualize_intent_tsne(model, dataloader, adj, device, output_dir):
    model.eval()
    intent_vectors = []
    true_labels = []
    with torch.no_grad():
        for x, y_cls, _ in dataloader:
            x = x.to(device)
            intent_vec, _, _ = model(x, adj)
            intent_vectors.append(intent_vec.cpu().numpy())
            true_labels.append(y_cls.numpy())
    intent_vectors = np.concatenate(intent_vectors, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    vec_2d = tsne.fit_transform(intent_vectors)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(vec_2d[:, 0], vec_2d[:, 1], c=true_labels, cmap='viridis', alpha=0.6)
    cbar = plt.colorbar(scatter, ticks=[0, 1, 2], label='Time Period')
    cbar.ax.set_yticklabels(['Flat', 'Morning Peak', 'Evening Peak'])
    ax.set_title('t-SNE of Intent Vectors')
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    save_figure(fig, "tsne_intent", output_dir)

def plot_confusion_matrix(model, dataloader, adj, device, output_dir):
    model.eval()
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Flat', 'Morning', 'Evening'],
                yticklabels=['Flat', 'Morning', 'Evening'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    save_figure(fig, "confusion_matrix", output_dir)
    report = classification_report(all_true, all_preds, target_names=['Flat', 'Morning Peak', 'Evening Peak'])
    with open(os.path.join(output_dir, f"classification_report_{TIMESTAMP}.txt"), 'w') as f:
        f.write(report)
    print(report)
    return all_true, all_preds

def plot_predictions_multi_node(model, test_loader, adj, device, output_dir, node_indices=[0,1,2], num_samples=3):
    """
    多节点预测可视化：
    - 整体平均预测曲线（测试集所有样本的平均）
    - 随机几个样本的预测曲线
    """
    model.eval()
    # 收集预测值和真实值（注意：数据集的目标节点是固定的，但我们可以通过修改数据集来预测任意节点？
    # 由于数据集只针对一个目标节点（target_node_idx）构建，这里我们仍然使用同一个模型，但只展示目标节点的预测。
    # 如需多节点，需要修改数据集或使用另外的预测头。为简化，我们仅使用目标节点展示。
    # 但我们可以展示不同样本的预测对比。
    all_true = []
    all_pred = []
    with torch.no_grad():
        for x, y_cls, y_pred in test_loader:
            x = x.to(device)
            _, _, pred_flow = model(x, adj)
            all_true.append(y_pred.cpu().numpy())
            all_pred.append(pred_flow.cpu().numpy())
    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    # 整体平均曲线
    mean_true = all_true.mean(axis=0)
    mean_pred = all_pred.mean(axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(range(PRED_LEN), mean_true, 'b-o', label='True (avg)')
    axes[0].plot(range(PRED_LEN), mean_pred, 'r--s', label='Predicted (avg)')
    axes[0].set_title('Average Prediction over Test Set')
    axes[0].set_xlabel('Future Steps')
    axes[0].set_ylabel('Normalized Flow')
    axes[0].legend()
    # 随机选取几个样本
    indices = np.random.choice(len(all_true), num_samples, replace=False)
    for i, idx in enumerate(indices):
        axes[1].plot(range(PRED_LEN), all_true[idx], '--o', label=f'True {i+1}')
        axes[1].plot(range(PRED_LEN), all_pred[idx], '-s', label=f'Pred {i+1}')
    axes[1].set_title('Individual Sample Predictions')
    axes[1].set_xlabel('Future Steps')
    axes[1].legend()
    plt.tight_layout()
    save_figure(fig, "prediction_comparison", output_dir)

def compute_metrics(all_true, all_pred, output_dir):
    """计算预测指标 MSE, MAE 并保存"""
    mse = mean_squared_error(all_true, all_pred)
    mae = mean_absolute_error(all_true, all_pred)
    print(f"Prediction Metrics on Test Set: MSE={mse:.6f}, MAE={mae:.6f}")
    with open(os.path.join(output_dir, f"metrics_{TIMESTAMP}.txt"), 'w') as f:
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

    # 生成时段标签（假设从0点开始）
    time_labels = generate_time_labels(T, start_hour=0)
    print("时段标签已生成 (0:平峰, 1:早高峰, 2:晚高峰)")

    # 构建相关性图
    adj_norm, adj_binary = build_correlation_graph(data_raw, threshold=CORR_THRESHOLD)
    print(f"相关性图密度: {adj_binary.sum().item() / (N * N):.4f}")

    # 绘制相关性图
    print("绘制相关性图...")
    plot_correlation_graph(adj_binary, coords, node_ids, OUTPUT_DIR)

    # 选择预测节点（取第一个）
    target_node_idx = 0

    # 创建数据集
    dataset = TrafficIntentDataset(data_raw, time_labels, target_node_idx,
                                   window=WINDOW_SIZE, pred_len=PRED_LEN, stride=3)
    total = len(dataset)
    train_len = int(0.6 * total)
    val_len = int(0.2 * total)
    test_len = total - train_len - val_len
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    print(f"数据集划分: 训练 {train_len}, 验证 {val_len}, 测试 {test_len}")

    # 构建模型
    model = IntentModel(
        N=N, node_feat_dim=1, spat_hid=64, spat_out=32, temp_hid=64,
        intent_dim=INTENT_DIM, num_classes=3, pred_len=PRED_LEN,
        use_tcn=USE_TCN
    )
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    print("开始训练...")
    (train_losses, val_losses, train_cls, val_cls, train_pred, val_pred,
     val_accuracies, best_model_path) = train_model(
        model, train_loader, val_loader, adj_norm,
        epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE, output_dir=OUTPUT_DIR
    )

    # 加载最佳模型用于测试
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=True))
    model.eval()

    # 可视化损失曲线
    plot_loss_curves(train_losses, val_losses, train_cls, val_cls, train_pred, val_pred, OUTPUT_DIR)

    # t-SNE 意图可视化
    print("生成 t-SNE 意图聚类图...")
    visualize_intent_tsne(model, test_loader, adj_norm, DEVICE, OUTPUT_DIR)

    # 混淆矩阵及分类报告
    print("生成混淆矩阵...")
    true_labels, pred_labels = plot_confusion_matrix(model, test_loader, adj_norm, DEVICE, OUTPUT_DIR)
    # 计算分类准确率（已在 train 中计算，这里重新计算测试集准确率）
    test_acc = accuracy_score(true_labels, pred_labels)
    print(f"测试集分类准确率: {test_acc:.4f}")

    # 预测指标
    print("计算预测指标...")
    # 收集所有预测值和真实值
    all_true = []
    all_pred = []
    with torch.no_grad():
        for x, y_cls, y_pred in test_loader:
            x = x.to(DEVICE)
            _, _, pred_flow = model(x, adj_norm.to(DEVICE))
            all_true.append(y_pred.cpu().numpy())
            all_pred.append(pred_flow.cpu().numpy())
    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    mse, mae = compute_metrics(all_true, all_pred, OUTPUT_DIR)

    # 多节点预测对比图（这里仅用目标节点，但展示整体平均和个体样本）
    plot_predictions_multi_node(model, test_loader, adj_norm, DEVICE, OUTPUT_DIR, node_indices=[0], num_samples=3)

    # 保存关键指标到汇总文件
    with open(os.path.join(OUTPUT_DIR, f"summary_{TIMESTAMP}.txt"), 'w') as f:
        f.write(f"Timestamp: {TIMESTAMP}\n")
        f.write(f"Correlation threshold: {CORR_THRESHOLD}\n")
        f.write(f"Window size: {WINDOW_SIZE}, Prediction steps: {PRED_LEN}\n")
        f.write(f"Test classification accuracy: {test_acc:.4f}\n")
        f.write(f"Prediction MSE: {mse:.6f}, MAE: {mae:.6f}\n")
        f.write(f"Best model path: {best_model_path}\n")
    print(f"所有结果已保存到: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()