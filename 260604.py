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
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

from shared import (
    load_node_coords,
    load_time_series,
    build_correlation_graph,
    build_hybrid_graph,
    TrafficIntentDataset,
    compute_future_labels,
    IntentModel,
    train_model,
    plot_correlation_graph,
    plot_loss_curves,
    visualize_intent_tsne,
    plot_confusion_matrix,
    plot_predictions,
    compute_metrics,
    compute_persistence_metrics,
    collect_predictions,
)

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
WINDOW_SIZE = 24
PRED_LEN = 6
BATCH_SIZE = 64
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
# ===============================================================

# 生成时间戳
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(OUTPUT_DIR, TIMESTAMP)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"输出目录: {OUTPUT_DIR}")

# -------------------------- 主程序 --------------------------
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
    last_train_start = full_dataset.samples[train_indices[-1]]
    max_train_hist = last_train_start + WINDOW_SIZE
    print(f"图构建数据范围: 时间步 0 至 {max_train_hist-1} (包含)")
    train_data_for_graph = data_raw[:, :max_train_hist]

    if USE_HYBRID_GRAPH:
        print("使用混合空间‑相关图构建...")
        adj_norm, adj_binary = build_hybrid_graph(
            coords, train_data_for_graph,
            dist_threshold=DIST_THRESHOLD,
            corr_threshold=CORR_THRESHOLD_HYBRID,
            normalize_coords=True
        )
    else:
        print("使用纯相关图构建...")
        adj_norm, adj_binary = build_correlation_graph(
            train_data_for_graph, threshold=CORR_THRESHOLD
        )

    # 密度计算
    density = adj_binary.sum() / (N * (N - 1)) if N > 1 else 0.0
    print(f"图密度: {density:.4f}")

    # 绘制相关性图
    print("绘制相关性图...")
    plot_correlation_graph(adj_binary, coords, node_ids, OUTPUT_DIR, TIMESTAMP,
                           corr_threshold=CORR_THRESHOLD)

    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 构建模型（传入target_idx）
    model = IntentModel(
        N=N, node_feat_dim=1, spat_hid=64, spat_out=32,
        temp_hid=64, intent_dim=INTENT_DIM, num_classes=3,
        pred_len=PRED_LEN, use_tcn=USE_TCN, target_idx=target_idx,
        predictor_layers=3,
    )
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    print("开始训练...")
    (train_losses, val_losses, train_cls, val_cls, train_pred, val_pred,
     val_accuracies, best_model_path) = train_model(
        model, train_loader, val_loader, adj_norm,
        EPOCHS, LEARNING_RATE, DEVICE, OUTPUT_DIR, TIMESTAMP,
        patience=EARLY_STOP_PATIENCE, use_adaptive_loss=True,
    )

    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    # 可视化损失
    plot_loss_curves(train_losses, val_losses, train_cls, val_cls,
                     train_pred, val_pred, OUTPUT_DIR, TIMESTAMP)

    # t-SNE
    print("生成 t-SNE 意图聚类图...")
    visualize_intent_tsne(model, test_loader, adj_norm, DEVICE, OUTPUT_DIR,
                          TIMESTAMP, class_names=["Low", "Medium", "High"])

    # 混淆矩阵
    print("生成混淆矩阵...")
    true_labels, pred_labels = plot_confusion_matrix(
        model, test_loader, adj_norm, DEVICE, OUTPUT_DIR, TIMESTAMP,
        class_names=["Low", "Medium", "High"],
    )
    test_acc = accuracy_score(true_labels, pred_labels)
    print(f"测试集分类准确率: {test_acc:.4f}")

    # 模型预测指标
    print("计算预测指标...")
    all_true, all_pred = collect_predictions(model, test_loader, adj_norm, DEVICE)
    model_mse, model_mae = compute_metrics(all_true, all_pred, OUTPUT_DIR, TIMESTAMP)

    # 持久性基线（修正）
    pers_mse, pers_mae = compute_persistence_metrics(
        test_loader, pred_len=PRED_LEN, target_idx=target_idx
    )
    print(f"持久性基线 MSE: {pers_mse:.6f}, MAE: {pers_mae:.6f}")

    # 预测对比图
    plot_predictions(model, test_loader, adj_norm, DEVICE, OUTPUT_DIR, TIMESTAMP,
                     pred_len=PRED_LEN, target_idx=target_idx, num_samples=3,
                     include_persistence=True)

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
