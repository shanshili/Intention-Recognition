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
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

from shared import (
    load_node_coords,
    load_time_series,
    generate_time_labels,
    build_correlation_graph,
    TrafficIntentDataset,
    IntentModel,
    train_model,
    plot_correlation_graph,
    plot_loss_curves,
    visualize_intent_tsne,
    plot_confusion_matrix,
    plot_predictions,
    compute_metrics,
    collect_predictions,
)

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

# -------------------------- 主程序 --------------------------
def main():
    print(f"使用设备: {DEVICE}")
    print("1. 加载数据...")
    node_coord_path = os.path.join(DATA_DIR, NODE_COORD_FILE)
    tsv_path = os.path.join(DATA_DIR, TSV_FILE)

    node_ids, coords = load_node_coords(node_coord_path)
    data_raw, node_names = load_time_series(tsv_path, check_validity=False)
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
    plot_correlation_graph(adj_binary, coords, node_ids, OUTPUT_DIR, TIMESTAMP,
                           corr_threshold=CORR_THRESHOLD)

    # 选择预测节点（取第一个）
    target_node_idx = 0

    # 创建数据集
    dataset = TrafficIntentDataset(data_raw, target_node_idx,
                                   window=WINDOW_SIZE, pred_len=PRED_LEN,
                                   stride=3, time_labels=time_labels)
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
        use_tcn=USE_TCN, target_idx=None, predictor_layers=2,
    )
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    print("开始训练...")
    (train_losses, val_losses, train_cls, val_cls, train_pred, val_pred,
     val_accuracies, best_model_path) = train_model(
        model, train_loader, val_loader, adj_norm,
        epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE,
        output_dir=OUTPUT_DIR, timestamp=TIMESTAMP,
        patience=None, use_adaptive_loss=False,
    )

    # 加载最佳模型用于测试
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    # 可视化损失曲线
    plot_loss_curves(train_losses, val_losses, train_cls, val_cls,
                     train_pred, val_pred, OUTPUT_DIR, TIMESTAMP)

    # t-SNE 意图可视化
    print("生成 t-SNE 意图聚类图...")
    visualize_intent_tsne(model, test_loader, adj_norm, DEVICE, OUTPUT_DIR,
                          TIMESTAMP, class_names=["Flat", "Morning Peak", "Evening Peak"])

    # 混淆矩阵及分类报告
    print("生成混淆矩阵...")
    true_labels, pred_labels = plot_confusion_matrix(
        model, test_loader, adj_norm, DEVICE, OUTPUT_DIR, TIMESTAMP,
        class_names=["Flat", "Morning", "Evening"],
    )
    # 计算分类准确率
    test_acc = accuracy_score(true_labels, pred_labels)
    print(f"测试集分类准确率: {test_acc:.4f}")

    # 预测指标
    print("计算预测指标...")
    all_true, all_pred = collect_predictions(model, test_loader, adj_norm, DEVICE)
    mse, mae = compute_metrics(all_true, all_pred, OUTPUT_DIR, TIMESTAMP)

    # 多节点预测对比图
    plot_predictions(model, test_loader, adj_norm, DEVICE, OUTPUT_DIR, TIMESTAMP,
                     pred_len=PRED_LEN, num_samples=3, include_persistence=False)

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
