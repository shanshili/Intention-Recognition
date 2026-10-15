# 版本

| 版本                         | 实际用途                                               | 与后续版本的关系                                     | 建议用途             |
| ---------------------------- | ------------------------------------------------------ | ---------------------------------------------------- | -------------------- |
| `260601.py`                  | GAT + 时间编码器，时段标签分类及流量预测               | 早期节点级意图原型                                   | 历史参考             |
| `260604.py`、`260604_TCN.py` | 改为未来流量水平分类并预测流量；窗口、损失等配置有差异 | 仍是早期分类任务，不产生当前所需的动态边级意图图     | 基线参考             |
| `260627/`                    | 第一版 DEIG-TCN；边级意图头与预测器可训练              | GAT/TCN 部分实际为固定特征映射                       | 旧版对照             |
| `260628/`                    | 可训练 GAT + TCN + 边级意图头；导出全部有效窗口        | 意图图生成主版本                                     | 保留为上游           |
| `260706/ratcgf/`             | RA-TCGF 融合；先节点、后边的部署选择                   | 融合主模型早期实现                                   | 旧版对照             |
| `260706_2/`                  | 主要调整绘图和部署，改成优先选边并带入端点             | 核心模型、模块 1–3、训练器与 260706 相同             | 旧版对照             |
| `260731/`                    | 增加时间映射、范围选择、多时刻部署与汇总输出           | 核心模型、模块 1–3、训练器与 260706_2 **逐字节相同** | 保留为下游修正基础。 |

# log

## 260603

### 测试260601.py

#### cpu-20260602_121306

> Timestamp: 20260602_121306
> Correlation threshold: 0.999
> Window size: 24, Prediction steps: 6
> Test classification accuracy: 1.0000
> Prediction MSE: 0.017758, MAE: 0.096653

![prediction_comparison_20260602_121306](log.assets/prediction_comparison_20260602_121306.png)

<img src="log.assets/confusion_matrix_20260602_121306.png" alt="confusion_matrix_20260602_121306" style="zoom:10%;" />

<img src="log.assets/tsne_intent_20260602_121306.png" alt="tsne_intent_20260602_121306" style="zoom:10%;" />

![loss_curves_20260602_121306](log.assets/loss_curves_20260602_121306.png)

<img src="log.assets/correlation_graph_20260602_121306.png" alt="correlation_graph_20260602_121306" style="zoom:10%;" />



> 节点数: 100, 时间步数: 5088
> 时段标签已生成 (0:平峰, 1:早高峰, 2:晚高峰)
> 相关性图密度: 0.0544
> 绘制相关性图...
> 数据集划分: 训练 1012, 验证 337, 测试 338
> 模型参数量: 135,785
> 开始训练...
> Epoch   1/30 | Train Loss: 1.8255 (cls:0.819 pred:1.007) | Val Loss: 1.6502 (cls:0.717 pred:0.933) | Val Acc: 0.7240
> Epoch   5/30 | Train Loss: 0.3697 (cls:0.267 pred:0.103) | Val Loss: 0.2956 (cls:0.216 pred:0.079) | Val Acc: 0.9436
> Epoch  10/30 | Train Loss: 0.0612 (cls:0.009 pred:0.052) | Val Loss: 0.0441 (cls:0.004 pred:0.040) | Val Acc: 1.0000
> Epoch  15/30 | Train Loss: 0.0421 (cls:0.002 pred:0.041) | Val Loss: 0.0326 (cls:0.001 pred:0.032) | Val Acc: 1.0000
> Epoch  20/30 | Train Loss: 0.0338 (cls:0.001 pred:0.033) | Val Loss: 0.0241 (cls:0.000 pred:0.024) | Val Acc: 1.0000
> Epoch  25/30 | Train Loss: 0.0271 (cls:0.000 pred:0.027) | Val Loss: 0.0235 (cls:0.000 pred:0.023) | Val Acc: 1.0000
> Epoch  30/30 | Train Loss: 0.0220 (cls:0.000 pred:0.022) | Val Loss: 0.0188 (cls:0.000 pred:0.019) | Val Acc: 1.0000
> 训练完成，最佳模型保存在 ./output\20260602_121306\best_model_20260602_121306.pth
> 生成 t-SNE 意图聚类图...
> 生成混淆矩阵...
>               precision    recall  f1-score   support
>
>   Flat       1.00      1.00      1.00       265
>
> Morning Peak       1.00      1.00      1.00        37
> Evening Peak       1.00      1.00      1.00        36
>
> accuracy                                         1.00       338
>
>    macro avg       1.00      1.00      1.00       338
> weighted avg       1.00      1.00      1.00       338
>
> 测试集分类准确率: 1.0000
> 计算预测指标...
> Prediction Metrics on Test Set: MSE=0.017758, MAE=0.096653

#### gpu-20260602_133235

> ```
> Timestamp: 20260602_133235
> Correlation threshold: 0.999
> Window size: 24, Prediction steps: 6
> Test classification accuracy: 1.0000
> Prediction MSE: 0.020117, MAE: 0.099520
> ```



![prediction_comparison_20260602_133235](log.assets/prediction_comparison_20260602_133235.png)

![loss_curves_20260602_133235](log.assets/loss_curves_20260602_133235.png)

<img src="log.assets/tsne_intent_20260602_133235.png" alt="tsne_intent_20260602_133235" style="zoom:10%;" />

<img src="log.assets/confusion_matrix_20260602_133235.png" alt="confusion_matrix_20260602_133235" style="zoom:10%;" />

<img src="log.assets/correlation_graph_20260602_133235.png" alt="correlation_graph_20260602_133235" style="zoom:10%;" />

> 节点数: 100, 时间步数: 5088
> 时段标签已生成 (0:平峰, 1:早高峰, 2:晚高峰)
> 相关性图密度: 0.0544
> 绘制相关性图...
> 数据集划分: 训练 1012, 验证 337, 测试 338
> 模型参数量: 135,785
> 开始训练...
> Epoch   1/30 | Train Loss: 1.8058 (cls:0.814 pred:0.992) | Val Loss: 1.5738 (cls:0.621 pred:0.952) | Val Acc: 0.7715
> Epoch   5/30 | Train Loss: 0.3681 (cls:0.267 pred:0.101) | Val Loss: 0.3361 (cls:0.230 pred:0.106) | Val Acc: 0.9080
> Epoch  10/30 | Train Loss: 0.0740 (cls:0.009 pred:0.065) | Val Loss: 0.0603 (cls:0.005 pred:0.055) | Val Acc: 1.0000
> Epoch  15/30 | Train Loss: 0.0475 (cls:0.002 pred:0.046) | Val Loss: 0.0446 (cls:0.001 pred:0.044) | Val Acc: 1.0000
> Epoch  20/30 | Train Loss: 0.0401 (cls:0.001 pred:0.039) | Val Loss: 0.0329 (cls:0.001 pred:0.032) | Val Acc: 1.0000
> Epoch  25/30 | Train Loss: 0.0313 (cls:0.001 pred:0.031) | Val Loss: 0.0274 (cls:0.001 pred:0.027) | Val Acc: 1.0000
> Epoch  30/30 | Train Loss: 0.0259 (cls:0.000 pred:0.026) | Val Loss: 0.0220 (cls:0.000 pred:0.022) | Val Acc: 1.0000
> 训练完成，最佳模型保存在 ./output\20260602_133235\best_model_20260602_133235.pth
> 生成 t-SNE 意图聚类图...
> 生成混淆矩阵...
>               precision    recall  f1-score   support
>
> Flat       1.00      1.00      1.00       255
>
> Morning Peak       1.00      1.00      1.00        35
> Evening Peak       1.00      1.00      1.00        48
>
> accuracy                               1.00       338
>
>    macro avg       1.00      1.00      1.00       338
> weighted avg       1.00      1.00      1.00       338
>
> 测试集分类准确率: 1.0000
> 计算预测指标...
> Prediction Metrics on Test Set: MSE=0.020117, MAE=0.099520

### 测试260604.py

#### gpu-20260604_141445 

> ```
> WINDOW_SIZE = 24
> PRED_LEN = 6
> BATCH_SIZE = 64
> EPOCHS = 30                      # 最大迭代数，早停会提前结束
> LEARNING_RATE = 1e-3
> INTENT_DIM = 64
> USE_TCN = False                  # False: BiGRU, True: TCN
> CORR_THRESHOLD = 0.8
> EARLY_STOP_PATIENCE = 7          # 新增: 早停容忍轮次
> DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
> TARGET_NODE_IDX = 0              # 目标节点索引，便于统一使用
> 
> # 混合图开关
> USE_HYBRID_GRAPH = True             # 启用混合空间‑相关图
> DIST_THRESHOLD = 0.15               # 归一化坐标后的距离阈值（0~1）
> CORR_THRESHOLD_HYBRID = 0.7         # 混合图中的相关性阈值（可不同于纯相关图）
> ```

![correlation_graph_20260604_141445](log.assets/correlation_graph_20260604_141445.png)

![tsne_intent_20260604_145233](log.assets/tsne_intent_20260604_145233.png)

![prediction_comparison_20260604_145233](log.assets/prediction_comparison_20260604_145233.png)

![loss_curves_20260604_145233](log.assets/loss_curves_20260604_145233.png)

![confusion_matrix_20260604_145233](log.assets/confusion_matrix_20260604_145233.png)

> 节点数: 100, 时间步数: 5088
> 总样本数: 1687
> 数据集划分: 训练 1012, 验证 337, 测试 338
> 标签阈值 (训练集未来平均流量分位数): [-0.22940786  0.7092247 ]
> Train 类别分布: {np.int64(0): np.int64(343), np.int64(1): np.int64(343), np.int64(2): np.int64(326)}
> Val 类别分布: {np.int64(0): np.int64(114), np.int64(1): np.int64(114), np.int64(2): np.int64(109)}
> Test 类别分布: {np.int64(0): np.int64(114), np.int64(1): np.int64(115), np.int64(2): np.int64(109)}
> 图构建数据范围: 时间步 0 至 3056 (包含)
> 使用混合空间‑相关图构建...
> 图密度: 0.0562
> 绘制相关性图...
> 模型参数量: 139,721
> 开始训练...
> Epoch   1/30 | Train Loss: 2.0503 (cls:1.058 pred:0.992) | Val Loss: 1.8127 (cls:0.913 pred:0.899) | Val Acc: 0.6350
> Epoch   5/30 | Train Loss: 0.5720 (cls:0.469 pred:0.111) | Val Loss: 0.4903 (cls:0.424 pred:0.066) | Val Acc: 0.8042
> Epoch  10/30 | Train Loss: 0.3951 (cls:0.348 pred:0.071) | Val Loss: 0.3382 (cls:0.301 pred:0.038) | Val Acc: 0.8368
> Epoch  15/30 | Train Loss: 0.3395 (cls:0.323 pred:0.056) | Val Loss: 0.3879 (cls:0.349 pred:0.038) | Val Acc: 0.8101
> Epoch  20/30 | Train Loss: 0.3242 (cls:0.314 pred:0.065) | Val Loss: 0.2831 (cls:0.257 pred:0.026) | Val Acc: 0.8516
> Epoch  25/30 | Train Loss: 0.2696 (cls:0.290 pred:0.049) | Val Loss: 0.2844 (cls:0.257 pred:0.028) | Val Acc: 0.8427
> 早停触发，最佳epoch: 18, val_loss: 0.2827
> 训练完成，最佳模型保存在 ./output\20260604_145233\best_model_20260604_145233.pth
> 生成 t-SNE 意图聚类图...
> 生成混淆矩阵...
>               precision    recall  f1-score   support
>
>          Low       0.96      0.95      0.95       114
>       Medium       0.75      0.90      0.82       115
>         High       0.93      0.74      0.83       109
>    
>     accuracy                           0.87       338
>    macro avg       0.88      0.86      0.87       338
> weighted avg       0.88      0.87      0.87       338
>
> 测试集分类准确率: 0.8669
> 计算预测指标...
> Prediction Metrics on Test Set: MSE=0.028360, MAE=0.124661
> 持久性基线 MSE: 1.046382, MAE: 0.731023

#### gpu-20260604_161905  

==USE_TCN = True==   

> ```
> WINDOW_SIZE = 24
> PRED_LEN = 6
> BATCH_SIZE = 64
> EPOCHS = 30                      # 最大迭代数，早停会提前结束
> LEARNING_RATE = 1e-3
> INTENT_DIM = 64
> USE_TCN = True                  # False: BiGRU, True: TCN
> CORR_THRESHOLD = 0.8
> EARLY_STOP_PATIENCE = 7          # 新增: 早停容忍轮次
> DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
> TARGET_NODE_IDX = 0              # 目标节点索引，便于统一使用
> 
> # 混合图开关
> USE_HYBRID_GRAPH = True             # 启用混合空间‑相关图
> DIST_THRESHOLD = 0.15               # 归一化坐标后的距离阈值（0~1）
> CORR_THRESHOLD_HYBRID = 0.7         # 混合图中的相关性阈值（可不同于纯相关图）
> ```

![confusion_matrix_20260604_161905](log.assets/confusion_matrix_20260604_161905.png)

![loss_curves_20260604_161905](log.assets/loss_curves_20260604_161905.png)

![prediction_comparison_20260604_161905](log.assets/prediction_comparison_20260604_161905.png)

![tsne_intent_20260604_161905](log.assets/tsne_intent_20260604_161905.png)

> 节点数: 100, 时间步数: 5088
> 总样本数: 1687
> 数据集划分: 训练 1012, 验证 337, 测试 338
> 标签阈值 (训练集未来平均流量分位数): [-0.22940786  0.7092247 ]
> Train 类别分布: {np.int64(0): np.int64(343), np.int64(1): np.int64(343), np.int64(2): np.int64(326)}
> Val 类别分布: {np.int64(0): np.int64(114), np.int64(1): np.int64(114), np.int64(2): np.int64(109)}
> Test 类别分布: {np.int64(0): np.int64(114), np.int64(1): np.int64(115), np.int64(2): np.int64(109)}
> 图构建数据范围: 时间步 0 至 3056 (包含)
> 使用混合空间‑相关图构建...
> 图密度: 0.0562
> 绘制相关性图...
> 模型参数量: 54,729
> 开始训练...
> Epoch   1/30 | Train Loss: 2.0965 (cls:1.094 pred:1.002) | Val Loss: 2.0963 (cls:1.098 pred:0.998) | Val Acc: 0.3383
> Epoch   5/30 | Train Loss: 0.9934 (cls:0.600 pred:0.399) | Val Loss: 0.8891 (cls:0.563 pred:0.327) | Val Acc: 0.7507
> Epoch  10/30 | Train Loss: 0.6233 (cls:0.461 pred:0.183) | Val Loss: 0.4940 (cls:0.392 pred:0.102) | Val Acc: 0.8546
> Epoch  15/30 | Train Loss: 0.5435 (cls:0.414 pred:0.166) | Val Loss: 0.4606 (cls:0.384 pred:0.076) | Val Acc: 0.8309
> Epoch  20/30 | Train Loss: 0.5089 (cls:0.416 pred:0.144) | Val Loss: 0.4452 (cls:0.347 pred:0.098) | Val Acc: 0.8398
> Epoch  25/30 | Train Loss: 0.4081 (cls:0.353 pred:0.123) | Val Loss: 0.3879 (cls:0.333 pred:0.055) | Val Acc: 0.8605
> Epoch  30/30 | Train Loss: 0.3639 (cls:0.334 pred:0.113) | Val Loss: 0.3457 (cls:0.280 pred:0.066) | Val Acc: 0.8843
> 训练完成，最佳模型保存在 ./output\20260604_161905\best_model_20260604_161905.pth
> 生成 t-SNE 意图聚类图...
> 生成混淆矩阵...
>               precision    recall  f1-score   support
>
>          Low       0.97      0.86      0.91       114
>       Medium       0.79      0.86      0.82       115
>         High       0.88      0.90      0.89       109
>    
>     accuracy                           0.87       338
>    macro avg       0.88      0.87      0.87       338
> weighted avg       0.88      0.87      0.87       338
>
> 测试集分类准确率: 0.8728
> 计算预测指标...
> Prediction Metrics on Test Set: MSE=0.062099, MAE=0.183796
> 持久性基线 MSE: 1.046382, MAE: 0.731023



> 从你切换到 TCN 的结果来看，分类性能（特别是 High 类的召回）有明显改善，但**预测能力却出现了严重退化**。整体呈现“此消彼长”的局面，需要根据任务优先级来权衡。
>
> ### 核心指标对比
>
> | 指标                    | BiGRU (原方案) | TCN (新方案) | 变化趋势 |
> | ----------------------- | -------------- | ------------ | -------- |
> | **模型参数量**          | 139,721        | 54,729       | -61%     |
> | **最佳验证损失**        | 0.2827         | 0.3457       | ↑ (变差) |
> | **测试分类准确率**      | 0.8669         | 0.8728       | +0.59%   |
> | **Macro F1**            | 0.87           | 0.87         | 持平     |
> | **High 类 Recall**      | 0.74           | **0.90**     | ↑+21.6%  |
> | **High 类 Precision**   | 0.93           | 0.88         | ↓        |
> | **Medium 类 Precision** | 0.75           | 0.79         | ↑        |
> | **Low 类 Recall**       | 0.95           | 0.86         | ↓-9.5%   |
> | **预测 MSE**            | **0.028**      | **0.062**    | ↑+121%   |
> | **预测 MAE**            | **0.125**      | **0.184**    | ↑+47%    |
>
> ### 详细分析
>
> #### ✅ 改进之处：分类均衡性提升
> - **High 类召回率从 0.74 跃升至 0.90**，有效解决了之前“高流量样本大量漏检”的核心痛点。  
> - Medium 类的精确率也略有提高（0.75→0.79）。  
> - 训练过程更平滑，没有触发早停，验证损失持续下降，说明 TCN 在该任务上收敛更稳定。
>
> #### ❌ 退化之处：预测性能大幅跳水
> - 预测 MSE 从 0.028 升到 0.062，MAE 从 0.125 升到 0.184，**翻了一倍多**。虽然仍远好于持久性基线，但绝对精度损失很大。  
> - Low 类召回率下降（0.95→0.86），部分低流量样本被误判为其他类。
>
> #### 🔍 退化原因推测
> 1. **TCN 的全局平均池化破坏时序细节**  
>    你实现的 `TemporalConvNet` 最后对时间维度做了 `mean(dim=-1)`，将所有历史信息压缩成一个向量。这种操作丢失了未来步的局部趋势、转折点等细粒度信息，对分类影响有限，但对需要精确多步数值预测的任务是致命的。BiGRU 取最后时刻的输出则隐含保留了顺序先验（最后时刻对未来最近步影响最大）。
>
> 2. **多任务损失失衡**  
>    验证损失中，分类损失 `cls=0.280` 与 BiGRU 的 `0.257` 相差不大，但预测损失 `pred=0.066` 是 BiGRU `0.026` 的 2.5 倍。自适应权重未能压制分类、拉升预测，模型在优化时偏向了分类，牺牲了预测精度。
>
> 3. **参数量减半可能欠拟合**  
>    TCN 参数量仅 54k，而 BiGRU 约 140k，时序建模能力受限，难以同时学好分类与回归两个目标。
>
> ---
>
> ### 优化建议（按优先级排序）
>
> 1. **保留 BiGRU 并针对性提升 High 召回**  
>    TCN 的分类收益主要来自 High 召回，这可以通过我们在之前讨论中提出的 Focal Loss、调整分类阈值、加深分类头等方法在 BiGRU 上复现，而不必牺牲预测性能。建议回退到 BiGRU，并采取以下措施：
>    - 使用 `FocalLoss(gamma=2)` 作为分类损失；
>    - 将早停指标改为验证集 `macro F1` 或分类损失，保存分类最优模型；
>    - 对 High 类适当降低 softmax 阈值（如从 0.33 调至 0.25）以提升召回。
>
> 2. **若坚持使用 TCN，必须改进池化方式**  
>    去掉 `mean(dim=-1)`，改用**注意力池化**或保留多个时间步的特征，再输入预测头。例如：
>    - 在 TCN 输出后加一个 `nn.MultiheadAttention` 或简单的 `self-attention` 层，生成上下文向量；
>    - 或者直接将 TCN 最终输出的完整序列馈入一个解码器（如一层 GRU）来生成多步预测。
>    
> 3. **显式控制多任务权重**  
>    无论使用哪种编码器，都应手动设定 `loss = w_cls * loss_cls + w_pred * loss_pred`。建议从 `w_cls=1, w_pred=0.5` 开始调整，逐步提高 `w_pred` 直到预测 MSE 回到可接受范围。
>
> 4. **增大 TCN 容量并尝试双向结构**  
>    如果一定要用卷积类架构，可考虑：
>    - 增大 `tcn_channels` 至 `(128, 256)` 或更多层；
>    - 采用 **双向 TCN**（虽然会引入未来信息，但历史窗口内是可行的）；
>    - 在 TCN 后串联一个小型 GRU，融合卷积的局部模式提取和循环网络的顺序建模。
>
> ### 总体评价
> **TCN 方案在分类公平性上取得了可喜的进步，但以大幅度牺牲预测精度为代价，整体实用性反而不如 BiGRU。**  
> 建议优先“复用 BiGRU + 加强分类策略”的组合，它更有可能在保持低预测误差的同时，将 High 类召回提升至 0.85 以上，达到均衡的最优。如果任务中分类的优先级远高于预测（例如只关心流量等级预警，不关心具体数值），那么当前 TCN 方案是可接受的，但需要告知下游业务预测误差已翻倍。

#### gpu-20260604_172914

==WINDOW_SIZE = 48==

==USE_TCN = True==  

==BATCH_SIZE = 32==

> ```
> WINDOW_SIZE = 48
> PRED_LEN = 6
> BATCH_SIZE = 32
> EPOCHS = 30                      # 最大迭代数，早停会提前结束
> LEARNING_RATE = 1e-3
> INTENT_DIM = 64
> USE_TCN = True                  # False: BiGRU, True: TCN
> CORR_THRESHOLD = 0.8
> EARLY_STOP_PATIENCE = 7          # 新增: 早停容忍轮次
> DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
> TARGET_NODE_IDX = 0              # 目标节点索引，便于统一使用
> 
> # 混合图开关
> USE_HYBRID_GRAPH = True             # 启用混合空间‑相关图
> DIST_THRESHOLD = 0.15               # 归一化坐标后的距离阈值（0~1）
> CORR_THRESHOLD_HYBRID = 0.7         # 混合图中的相关性阈值（可不同于纯相关图）
> ```

![confusion_matrix_20260604_172914](log.assets/confusion_matrix_20260604_172914.png)

![loss_curves_20260604_172914](log.assets/loss_curves_20260604_172914.png)

![prediction_comparison_20260604_172914](log.assets/prediction_comparison_20260604_172914.png)

![tsne_intent_20260604_172914](log.assets/tsne_intent_20260604_172914.png)



> 节点数: 100, 时间步数: 5088
> 总样本数: 1679
> 数据集划分: 训练 1007, 验证 335, 测试 337
> 标签阈值 (训练集未来平均流量分位数): [-0.22940786  0.7092247 ]
> Train 类别分布: {np.int64(0): np.int64(341), np.int64(1): np.int64(342), np.int64(2): np.int64(324)}
> Val 类别分布: {np.int64(0): np.int64(114), np.int64(1): np.int64(113), np.int64(2): np.int64(108)}
> Test 类别分布: {np.int64(0): np.int64(114), np.int64(1): np.int64(115), np.int64(2): np.int64(108)}
> 图构建数据范围: 时间步 0 至 3065 (包含)
> 使用混合空间‑相关图构建...
> 图密度: 0.0562
> 绘制相关性图...
> 模型参数量: 162,249
> 开始训练...
> Epoch   1/30 | Train Loss: 1.2714 (cls:0.861 pred:0.821) | Val Loss: 1.5495 (cls:1.163 pred:0.772) | Val Acc: 0.3642
> Epoch   5/30 | Train Loss: 0.6921 (cls:0.552 pred:0.281) | Val Loss: 0.6235 (cls:0.513 pred:0.222) | Val Acc: 0.7642
> Epoch  10/30 | Train Loss: 0.5556 (cls:0.473 pred:0.166) | Val Loss: 0.4450 (cls:0.397 pred:0.096) | Val Acc: 0.8418
> Epoch  15/30 | Train Loss: 0.4372 (cls:0.375 pred:0.125) | Val Loss: 0.3986 (cls:0.342 pred:0.114) | Val Acc: 0.8597
> Epoch  20/30 | Train Loss: 0.4117 (cls:0.349 pred:0.125) | Val Loss: 0.3808 (cls:0.338 pred:0.086) | Val Acc: 0.8687
> Epoch  25/30 | Train Loss: 0.3855 (cls:0.335 pred:0.100) | Val Loss: 0.3344 (cls:0.308 pred:0.052) | Val Acc: 0.8776
> Epoch  30/30 | Train Loss: 0.3763 (cls:0.326 pred:0.100) | Val Loss: 0.2672 (cls:0.242 pred:0.051) | Val Acc: 0.9075
> 训练完成，最佳模型保存在 ./output\20260604_172914\best_model_20260604_172914.pth
> 生成 t-SNE 意图聚类图...
> 生成混淆矩阵...
>               precision    recall  f1-score   support
>
>          Low       0.95      0.95      0.95       114
>       Medium       0.94      0.77      0.84       115
>         High       0.83      0.99      0.90       108
>    
>     accuracy                           0.90       337
>    macro avg       0.90      0.90      0.90       337
> weighted avg       0.91      0.90      0.90       337
>
> 测试集分类准确率: 0.8991
> 计算预测指标...
> Prediction Metrics on Test Set: MSE=0.055492, MAE=0.162793
> 持久性基线 MSE: 1.049264, MAE: 0.732757



> 从实验指标来看，这是一个**质的飞跃**，成功解决了 BiGRU 版本“分类召回不均衡”与 TCN 平均池化版“预测精度崩坏”的双重问题，实现了**分类与预测的兼顾**。下面是详细点评和与之前版本的对比。
>
> ---
>
> ### 1. 核心指标对比
>
> | 指标               | BiGRU (原 v2.1) | TCN 平均池化 | **新版 TCN (注意力池化)** | 变化分析                               |
> | ------------------ | --------------- | ------------ | ------------------------- | -------------------------------------- |
> | 测试分类准确率     | 0.8669          | 0.8728       | **0.8991**                | ↑ 大幅提升                             |
> | Macro F1           | 0.87            | 0.87         | **0.90**                  | ↑ 0.03                                 |
> | **High 类 Recall** | 0.74            | 0.90         | **0.99**                  | ↑ 惊人（仅漏1%）                       |
> | High 类 Precision  | 0.93            | 0.88         | 0.83                      | ↓ 但可接受，几乎无漏检                 |
> | Medium 类 Recall   | 0.90            | 0.86         | 0.77                      | ↓ 部分 Medium 被分到 High              |
> | Low 类 Recall      | 0.95            | 0.86         | **0.95**                  | 回到原始优秀水平                       |
> | **预测 MSE**       | **0.028**       | 0.062        | **0.055**                 | 比原 TCN 大幅好转，但未回到 BiGRU 水平 |
> | **预测 MAE**       | **0.125**       | 0.184        | **0.163**                 | 同上                                   |
> | 持久性基线 MSE     | 1.046           | 1.046        | 1.049                     | 稳定                                   |
>
> ---
>
> ### 2. 定性分析
>
> **✅ 巨大进步：**
> - **High 类召回率 0.99**，几乎抓齐了所有高流量事件，这对于安全预警、负荷管理等下游应用意义重大，BiGRU 的 0.74 曾是最大短板。
> - **整体分类指标（准确率、宏 F1）均突破 0.89/0.90**，相比前两个版本有明显提升，证实了注意力池化保留时序信息对意图区分的价值。
> - **预测精度较平均池化版 TCN 改善约 10%**（MSE 0.062→0.055），回归任务没有被完全放弃，且仍远超持久性基线。
> - **训练过程健康**：验证损失稳定下降，第 30 轮未出现早停，说明模型仍在学习，有进一步微调的空间。
>
> **⚠️ 仍可优化的点：**
> - **预测 MSE 0.055 vs BiGRU 的 0.028**，回归性能仍有差距。虽然损失权重已固定（推测你用了 `WEIGHT_PRED=0.5`），但 TCN 的归纳偏置可能天生不如 RNN 适合逐点预测。这可以通过增大预测损失权重（如 `WEIGHT_PRED=1.0`）或使用 Focal 形式的预测损失进一步拉回。
> - **Medium 类召回下降（0.90→0.77）**：一些中等流量样本被误判为 High。这从混淆矩阵逻辑看是可理解的——High 召回极高必然导致一些边界 Medium 被划入 High，属于“宁可错报，不可漏报”的倾向。若业务上 Medium 误报为 High 成本较高，可适当降低 High 类的决策阈值（温度缩放后调整），或对分类损失引入类别加权。
> - **图密度仍然低（0.0562）**，可能限制空间编码的信息传递。后续可尝试 KNN 图或基于注意力自动学习边权重。
>
> ---
>
> ### 3. 与 BiGRU 版本的深度比较
>
> 这次新 TCN 版本在**分类能力（尤其高危样本识别）上完胜 BiGRU**，预测能力则接近 BiGRU 的 50% 性能（0.055 vs 0.028）。如果你场景中**漏掉高流量（High）的代价远大于预测数值的微小偏差**，那么当前模型已经非常理想。比如需要提前预警拥堵、启动资源调度，那么高召回率至关重要，0.99 的召回几乎可以确保万无一失。
>
> 但若同时要求预测非常精确的流量值（如用于精细控制），则仍需考虑 BiGRU，或者尝试在 TCN 的基础上**增强预测头**（如加入自回归解码、增大预测损失权重至 1.5）。
>
> ---
>
> ### 4. 结论
>
> > **总体评价：非常成功。模型在保持良好回归能力的同时，分类性能全面超越原始 BiGRU，尤其解决了高危样本漏检的致命弱点。**  
>
> 你可以向导师/团队展示：新架构将高流量事件捕获率从 **74% 提升至 99%**，整体意图识别准确率从 **86.7% 提升至 89.9%**，同时预测均方根误差（RMSE）仅从 0.167（BiGRU）轻微增至 0.235（仍然远优于基线 1.024），实现了安全性与精确性的最佳平衡。如果后续微调权重能再将预测 MSE 拉回 0.04 以内，则堪称完美。

## 260627

### 测试260627

#### 20260627_224528

w=24 h=6

![candidate_directed_graph_20260627_224528](log.assets/candidate_directed_graph_20260627_224528.png)

![edge_mask_ablation_20260627_224528](log.assets/edge_mask_ablation_20260627_224528.png)

![intent_graphs_10windows_20260627_224528](log.assets/intent_graphs_10windows_20260627_224528.png)

#### 20260628_021428

w=12 h=3

![edge_mask_ablation_20260628_021428](log.assets/edge_mask_ablation_20260628_021428.png)

![intent_graphs_10windows_20260628_021428](log.assets/intent_graphs_10windows_20260628_021428.png)

> 文件位置：[run_summary_20260627_224528.json](./260627/outputs/20260627_224528/run_summary_20260627_224528.json)
> 文件位置：[run_summary_20260628_022125.json](./260627/outputs/20260628_022125/run_summary_20260628_022125.json)
>
> 通过对比两个文件中的 hotspots_per_window 数据，我们可以观察到以下关键差异和可能含义：
>
> 1. 数值范围分布不同：
> - 第一个文件 (W=24, H=6)：hotspots_per_window 数值为 [5, 10, 3, 3, 5, 8, 9, 3, 5, 10]
> - 第二个文件 (W=3, H=1)：hotspots_per_window 数值为 [7, 6, 8, 8, 9, 8, 10, 10, 6, 10]
>
> 2. 窗口大小影响：
> - 第一个文件使用更大的窗口尺寸 (W=24, H=6)
> - 第二个文件使用更小的窗口尺寸 (W=3, H=1)
>
> 3. 可能的含义：
> - 小窗口 (W=3, H=1) 的热点分布更加均匀，数值集中在6-10之间，没有低于5的值
> - 大窗口 (W=24, H=6) 的热点分布波动更大，最小值只有3，最大值为10
> - 这表明较小的窗口可能更能捕捉局部热点特征，而较大的窗口可能会平滑掉一些局部热点
>
> 4. 性能相关性：
> - 小窗口模型的测试MAE (0.09157) 明显优于大窗口模型 (0.15520)
> - 这 suggests 热点分布的均匀性与模型性能可能存在正相关关系
>
> 5. 数据密度：
> - 两个文件的 intent_density_per_window 相同 (0.004949)
> - 这表明窗口大小变化没有改变整体的数据密度，但影响了热点的分布模式
>
> **这些差异表明窗口大小的选择对热点检测和模型性能有显著影响，较小的窗口可能更适合捕捉局部特征。**

#### 20260628_024449

w=3 h=1

![intent_graphs_10windows_20260628_024449](log.assets/intent_graphs_10windows_20260628_024449.png)

#### 20260628_150956

不多次标准化：没啥区别

#### 20260628_152134

96天的展示

确实每小时的重点意图区域不同

#### 问题：

1. 算法原理：260627是固定特征+边分数学习→须升级成GAT-TCN
2. 怎么验证识别的有效
3. 是否需要换数据集
4. 图是颜色更换
5. 

## 260628

### 测试260628

#### 20260628_162448

##### d169是2025年6月18日 星期三

d173有极高的交通流 （）

##### 加入GAT-TCN后的结果对比

> 文件位置：[260628/outputs/20260628_162448/run_summary_20260628_162448.json](./260628/outputs/20260628_162448/run_summary_20260628_162448.json) 和 [260627/outputs/20260628_152134/run_summary_20260628_152134.json](./260627/outputs/20260628_152134/run_summary_20260628_152134.json)
>
> 这两个JSON文件记录了两个不同版本的DEIG-TCN模型的实验结果，对比分析如下：
>
> 1. **模型性能对比**：
>    - 260628版本（可训练GAT-TCN）：
>      - 测试MAE：0.0841
>      - 持久性基线MAE：0.2582
>    
>    - 260627版本（固定特征映射）：
>      - 测试MAE：0.0892
>      - 持久性基线MAE：0.2564
>
>    结论：260628版本（可训练GAT-TCN）的预测性能更好，MAE降低了约5.7%。
>
> 2. **边掩消融实验**：
>    - 260628版本：
>      - 完整模型：0.0841
>      - 掩码顶部（mask_top）：0.0856
>      - 随机掩码（mask_random）：0.0842
>      - 掩码底部（mask_low）：0.0841
>    
>    - 260627版本：
>      - 完整模型：0.0892
>      - 掩码顶部（mask_top）：0.0906
>      - 随机掩码（mask_random）：0.0895
>      - 掩码底部（mask_low）：0.0893
>
>    结论：两个版本都显示移除高意图边会降低性能，而260628版本对边掩消融的敏感性略低。
>
> 3. **意图密度分析**：
>    - 两个版本的意图密度完全相同：0.0049（约0.5%）
>    - 这表明两个模型在边选择策略上保持了一致性
>
> 4. **热点分析**：
>    - 260628版本每窗口热点数：3-12个，平均约7.2个
>    - 260627版本每窗口热点数：3-17个，平均约10.6个
>    
>    结论：260628版本识别的热点数量更少，但更集中，这可能反映了其时空编码器能够更精准地定位重要节点。
>
> 5. **实验设置一致性**：
>    - 两个实验使用相同的：
>      - 节点数(N)：100
>      - 时间步长(T)：5088
>      - 窗口大小(W)：3
>      - 预测步长(H)：1
>      - 候选边数：488
>      - 训练/验证/测试窗口数：3051/1017/1017
>      - 意图分位数阈值：0.9
>
> 总结：260628版本通过引入可训练的GAT-TCN编码器，不仅提高了预测精度，还使模型能够更精准地识别时空中的重要节点（热点），同时保持了相似的意图密度。这表明端到端的训练方式能够更好地捕捉时空数据的动态特征。

多了更多双向边

##### 验证方法

> # 1. 当前已经使用的验证方法
>
> ## 1.1 预测性能验证：DEIG-TCN vs persistence baseline
>
> 测试阶段先计算模型预测误差：
>
> $$
> \mathrm{MAE}_{\mathrm{DEIG}}
> =
> \frac{1}{NH}
> \sum_{i=1}^{N}
> \sum_{h=1}^{H}
> \left|
> \widehat z_i^{t+h}-z_i^{t+h}
> \right|.
> $$
>
> 同时用 persistence baseline 比较：
>
> $$
> \widehat z_i^{t+h}=z_i^t,
> $$
>
> 即假设未来等于当前。代码中会输出
>
> $$
> \mathrm{MAE}_{\mathrm{DEIG}}
> \quad\text{和}\quad
> \mathrm{MAE}_{\mathrm{persist}}.
> $$
>
> 如果
>
> $$
> \mathrm{MAE}_{\mathrm{DEIG}}
> <
> \mathrm{MAE}_{\mathrm{persist}},
> $$
>
> 说明整体模型确实从时空结构中学到了有用信息。这个验证是**模型整体有效性验证**，但还不是专门验证意图子图。fileciteturn0file0
>
> ---
>
> ## 1.2 边遮蔽消融：mask top / random / low
>
> 这是当前代码中最直接验证意图边有效性的方法。
>
> 测试阶段先得到每条候选边的意图分数：
>
> $$
> s_{ij}^t = \mathrm{EdgeIntentHead}
> \left(
> \mathbf z_i^t,\mathbf z_j^t,\mathbf e_{ij}^t
> \right).
> $$
>
> 然后按照分数排序，分别遮蔽：
>
> $$
> \text{Top }10\%\text{ 高意图边},
> $$
>
> $$
> \text{Random }10\%\text{ 随机边},
> $$
>
> $$
> \text{Low }10\%\text{ 低意图边}.
> $$
>
> 遮蔽方式是令对应边分数变为 0：
>
> $$
> s_{ij}^t \leftarrow 0.
> $$
>
> 然后重新预测并计算 MAE。理想结果应该满足：
>
> $$
> \boxed{
> \mathrm{MAE}_{\mathrm{mask\ top}}
> >
> \mathrm{MAE}_{\mathrm{mask\ random}}
> \gtrsim
> \mathrm{MAE}_{\mathrm{mask\ low}}.
> }
> $$
>
> 含义是：如果删除高意图边造成的预测性能下降最大，那么这些边确实对预测有关键贡献。当前 `main.py` 里正是这样计算 `mask_top`、`mask_random` 和 `mask_low` 的。fileciteturn0file0
>
> 这个方法验证的是：
>
> $$
> \boxed{
> \text{高 }s_{ij}^t\text{ 的边是否对预测更重要。}
> }
> $$
>
> 因此它是当前最核心的**有效性验证**。
>
> ---
>
> ## 1.3 意图图可视化与热点分析
>
> 当前代码还会选取连续测试窗口，计算每个窗口的边分数，然后用分位数阈值保留高分边：
>
> $$
> \delta_t=Q_{0.90}
> \left(
> \{s_{ij}^t:(i,j)\in E_0\}
> \right),
> $$
>
> $$
> E_I^t
> =
> \left\{
> (i,j)\in E_0:
> s_{ij}^t\geq \delta_t
> \right\}.
> $$
>
> 也就是说，当前意图子图大约保留候选边中最高的 $10\%$。代码还会计算意图图密度、热点子图数量，并输出连续窗口的可视化结果。fileciteturn0file0 fileciteturn0file6
>
> 这部分主要是**解释性和可视化验证**，用于观察：
>
> $$
> G_I^t
> $$
>
> 是否具有空间聚集性、方向性、热点结构和时间变化规律。
>
> ---
>
> ## 1.4 稀疏性与方向性约束
>
> 训练目标中还加入了两个结构正则项：
>
> $$
> \mathcal L_{\mathrm{sparse}}
> =
> \frac{1}{|E_0|}
> \sum_{(i,j)\in E_0}
> s_{ij}^t,
> $$
>
> 用于鼓励意图图稀疏；以及
>
> $$
> \mathcal L_{\mathrm{dir}}
> =
> \frac{1}{|\mathcal P|}
> \sum_{\{i,j\}\in\mathcal P}
> s_{ij}^t s_{ji}^t,
> $$
>
> 用于惩罚双向边同时高分。fileciteturn0file7
>
> 不过要注意：这两个正则项是**结构诱导机制**，不是严格意义上的验证方法。它们让意图图更稀疏、更有方向性，但不能单独证明意图边一定真实可靠。
>
> ---
>
> # 2. 当前验证的不足
>
> 当前验证已经有一个不错的闭环：
>
> $$
> s_{ij}^t
> \rightarrow
> \widetilde s_{ij}^t
> \rightarrow
> \mathbf g_j^t
> \rightarrow
> \widehat z_j^{t+h}
> \rightarrow
> \mathrm{MAE}.
> $$
>
> 但还存在几个不足。
>
> 第一，目前主要只做了 **10% 边遮蔽**，验证点比较单一。更稳健的做法应该看不同遮蔽比例下的完整曲线。
>
> 第二，随机遮蔽当前只做一次时，偶然性较大。更好的是重复 $R$ 次随机遮蔽，报告均值和置信区间：
>
> $$
> \mu_{\mathrm{rand}}(p)
> \pm
> 1.96
> \frac{\sigma_{\mathrm{rand}}(p)}{\sqrt R}.
> $$
>
> 第三，目前更多是在验证“高分边对预测有用”，但还没有充分验证：
>
> $$
> \text{意图边是否稳定、是否可复现、是否符合交通传播规律。}
> $$
>
> 第四，由于预测时使用入边归一化：
>
> $$
> \widetilde s_{ij}^t
> =
> \frac{s_{ij}^t}
> {\sum_{p:(p,j)\in E_0}s_{pj}^t+\varepsilon},
> $$
>
> 所以原始分数 $s_{ij}^t$ 的绝对值可能存在尺度问题。实践中更应该关注：
>
> $$
> \text{边分数排序}
> \quad\text{和}\quad
> \text{归一化后的相对权重}.
> $$
>
> ---
>
> # 3. 可以增加的可靠性与有效性验证方法
>
> 下面这些方法建议分成四类：**预测贡献验证、稳定性验证、结构合理性验证、外部一致性验证**。
>
> ---
>
> # A. 预测贡献类验证
>
> ## A1. 多比例删除曲线
>
> 不要只删除 top 10%，而是设置多个比例：
>
> $$
> p\in\{1\%,5\%,10\%,20\%,30\%,50\%\}.
> $$
>
> 分别计算：
>
> $$
> \Delta_{\mathrm{top}}(p)
> =
> \mathrm{MAE}_{\mathrm{mask\ top}(p)}
> -
> \mathrm{MAE}_{\mathrm{full}},
> $$
>
> $$
> \Delta_{\mathrm{rand}}(p)
> =
> \mathrm{MAE}_{\mathrm{mask\ rand}(p)}
> -
> \mathrm{MAE}_{\mathrm{full}},
> $$
>
> $$
> \Delta_{\mathrm{low}}(p)
> =
> \mathrm{MAE}_{\mathrm{mask\ low}(p)}
> -
> \mathrm{MAE}_{\mathrm{full}}.
> $$
>
> 如果意图边有效，应观察到：
>
> $$
> \boxed{
> \Delta_{\mathrm{top}}(p)
> >
> \Delta_{\mathrm{rand}}(p)
> >
> \Delta_{\mathrm{low}}(p)
> }
> $$
>
> 在多数 $p$ 上成立。
>
> 还可以计算删除曲线面积：
>
> $$
> \mathrm{AUC}_{\mathrm{del}}
> =
> \int_0^1
> \Delta_{\mathrm{top}}(p)\,dp.
> $$
>
> 越大表示高分边越关键。
>
> ---
>
> ## A2. 插入曲线：只保留 top 边
>
> 删除实验问的是：
>
> $$
> \text{去掉高分边会不会变差？}
> $$
>
> 插入实验问的是：
>
> $$
> \text{只保留高分边是否已经足够预测？}
> $$
>
> 定义只保留 top $p\%$ 边：
>
> $$
> s_{ij}^{t,p}
> =
> \begin{cases}
> s_{ij}^t,
> &
> (i,j)\in \mathrm{Top}_p,
> \\
> 0,
> &
> \text{otherwise}.
> \end{cases}
> $$
>
> 然后计算：
>
> $$
> \mathrm{MAE}_{\mathrm{keep\ top}(p)}.
> $$
>
> 如果较小的 $p$ 就能接近完整模型：
>
> $$
> \mathrm{MAE}_{\mathrm{keep\ top}(p)}
> \approx
> \mathrm{MAE}_{\mathrm{full}},
> $$
>
> 说明意图子图具有较强的**充分性**。
>
> 删除实验验证必要性，插入实验验证充分性。两者结合更完整：
>
> $$
> \boxed{
> \text{高意图边既必要，又充分。}
> }
> $$
>
> ---
>
> ## A3. 分数分桶校准验证
>
> 把边按照 $s_{ij}^t$ 分成若干分位桶：
>
> $$
> B_1,B_2,\ldots,B_K,
> $$
>
> 其中 $B_K$ 是最高分边，$B_1$ 是最低分边。分别遮蔽每个桶，计算误差增量：
>
> $$
> \Delta_k
> =
> \mathrm{MAE}_{\mathrm{mask}(B_k)}
> -
> \mathrm{MAE}_{\mathrm{full}}.
> $$
>
> 如果意图分数可靠，应当满足近似单调关系：
>
> $$
> \boxed{
> s\text{ 越高的桶}
> \Rightarrow
> \Delta_k\text{ 越大}.
> }
> $$
>
> 也就是：
>
> $$
> \Delta_1
> \leq
> \Delta_2
> \leq
> \cdots
> \leq
> \Delta_K.
> $$
>
> 这个方法比只看 top 10% 更有说服力。
>
> ---
>
> ## A4. 置乱检验
>
> 保持模型不变，只打乱边分数：
>
> $$
> s_{ij}^t
> \rightarrow
> s_{\pi(i,j)}^t.
> $$
>
> 如果原始意图分数是有意义的，则应满足：
>
> $$
> \mathrm{MAE}_{\mathrm{shuffle}}
> >
> \mathrm{MAE}_{\mathrm{full}}.
> $$
>
> 还可以做时间置乱：
>
> $$
> s_{ij}^t
> \rightarrow
> s_{ij}^{t'},
> \qquad t'\neq t.
> $$
>
> 若动态意图确实有时间意义，则：
>
> $$
> \mathrm{MAE}_{\mathrm{time\ shuffle}}
> >
> \mathrm{MAE}_{\mathrm{full}}.
> $$
>
> 这可以验证：
>
> $$
> \boxed{
> s_{ij}^t\text{ 不只是静态边重要性，而是有动态时间含义。}
> }
> $$
>
> ---
>
> ## A5. 反向边检验
>
> 把每条边的分数换成反向边分数：
>
> $$
> s_{ij}^t
> \leftarrow
> s_{ji}^t.
> $$
>
> 然后计算：
>
> $$
> \mathrm{MAE}_{\mathrm{reverse}}.
> $$
>
> 如果方向性真的有效，应该有：
>
> $$
> \boxed{
> \mathrm{MAE}_{\mathrm{reverse}}
> >
> \mathrm{MAE}_{\mathrm{full}}.
> }
> $$
>
> 这个实验专门验证：
>
> $$
> \text{模型学到的是有向传播，而不是无向相关。}
> $$
>
> ---
>
> ## A6. 与无意图模型对比
>
> 增加几个基线模型：
>
> ### 均匀入边权重
>
> $$
> \widetilde s_{ij}^t
> =
> \frac{1}{|\mathcal N_j^-|}.
> $$
>
> ### 只用静态候选图
>
> $$
> s_{ij}^t
> =
> \mathrm{constant}
> \quad\text{or}\quad
> s_{ij}=f(d_{ij},\rho_{ij}).
> $$
>
> ### 随机意图图
>
> $$
> s_{ij}^t\sim \mathrm{Uniform}(0,1).
> $$
>
> ### 无边消息传播
>
> $$
> \mathbf g_j^t=\mathbf 0.
> $$
>
> 如果完整模型显著优于这些版本：
>
> $$
> \mathrm{MAE}_{\mathrm{DEIG}}
> <
> \mathrm{MAE}_{\mathrm{uniform}},
> \quad
> \mathrm{MAE}_{\mathrm{static}},
> \quad
> \mathrm{MAE}_{\mathrm{random}},
> \quad
> \mathrm{MAE}_{\mathrm{no\ edge}},
> $$
>
> 就能证明动态意图子图确实带来了预测收益。
>
> ---
>
> # B. 稳定性与可复现性验证
>
> ## B1. 多随机种子一致性
>
> 用不同随机种子训练 $R$ 次，得到：
>
> $$
> G_{I,r}^t,
> \qquad r=1,\ldots,R.
> $$
>
> 比较 top 意图边集合的 Jaccard 相似度：
>
> $$
> J(r_1,r_2)
> =
> \frac{
> |E_{I,r_1}^t\cap E_{I,r_2}^t|
> }{
> |E_{I,r_1}^t\cup E_{I,r_2}^t|
> }.
> $$
>
> 如果平均相似度较高：
>
> $$
> \overline J
> =
> \frac{2}{R(R-1)}
> \sum_{r_1<r_2}J(r_1,r_2),
> $$
>
> 说明意图子图对初始化不敏感。
>
> 也可以比较边分数排序的 Spearman 相关：
>
> $$
> \rho_{\mathrm{rank}}
> =
> \operatorname{Spearman}
> \left(
> s^{(r_1)},s^{(r_2)}
> \right).
> $$
>
> ---
>
> ## B2. 时间邻近窗口稳定性
>
> 连续时间窗口的意图图不应该完全随机跳变。可以计算：
>
> $$
> J_t
> =
> \frac{
> |E_I^t\cap E_I^{t+1}|
> }{
> |E_I^t\cup E_I^{t+1}|
> }.
> $$
>
> 正常情况下，交通传播结构应该具有一定连续性：
>
> $$
> J_t
> $$
>
> 不应长期接近 0。
>
> 但也不能要求它恒定很高，因为早晚高峰、突发拥堵、异常波动会导致意图图变化。因此可以同时观察：
>
> $$
> J_t
> \quad\text{与}\quad
> \mathrm{traffic\ volatility}_t
> $$
>
> 之间的关系。若交通波动越大，意图图变化越明显，这是合理现象。
>
> ---
>
> ## B3. 数据扰动鲁棒性
>
> 对输入数据加入轻微扰动：
>
> $$
> \widetilde z_i^t
> =
> z_i^t+\epsilon_i^t,
> \qquad
> \epsilon_i^t\sim \mathcal N(0,\sigma^2).
> $$
>
> 或随机遮蔽少量观测值。重新推理得到：
>
> $$
> \widetilde s_{ij}^t.
> $$
>
> 计算：
>
> $$
> \operatorname{Corr}
> \left(
> s_{ij}^t,
> \widetilde s_{ij}^t
> \right)
> $$
>
> 以及 top 边 Jaccard 相似度。如果轻微扰动不会导致意图子图剧烈变化，说明可靠性更强。
>
> ---
>
> ## B4. 不同训练时间段验证
>
> 例如分别用不同月份或不同时间片训练模型：
>
> $$
> \mathcal D_1,\mathcal D_2,\ldots,\mathcal D_R.
> $$
>
> 比较它们在相同测试窗口上的意图图：
>
> $$
> G_{I,1}^t,G_{I,2}^t,\ldots,G_{I,R}^t.
> $$
>
> 如果主要高意图边在不同训练集上仍然稳定出现，说明不是偶然拟合。
>
> ---
>
> # C. 结构合理性验证
>
> ## C1. 方向非对称性指标
>
> 当前模型有方向正则，但可以额外报告方向非对称程度：
>
> $$
> A_{\mathrm{dir}}
> =
> \frac{
> \sum_{\{i,j\}}
> |s_{ij}^t-s_{ji}^t|
> }{
> \sum_{\{i,j\}}
> (s_{ij}^t+s_{ji}^t)+\varepsilon
> }.
> $$
>
> 如果
>
> $$
> A_{\mathrm{dir}}
> $$
>
> 过低，说明图接近无向；如果过高但预测没有提升，可能说明方向正则过强。
>
> 理想情况是：
>
> $$
> A_{\mathrm{dir}}\text{ 较高}
> \quad\text{且}\quad
> \mathrm{MAE}\text{ 较低}.
> $$
>
> ---
>
> ## C2. 稀疏性与性能联合曲线
>
> 改变阈值分位数：
>
> $$
> q\in\{0.70,0.80,0.85,0.90,0.95\}.
> $$
>
> 得到不同密度的意图图：
>
> $$
> E_I^t(q)
> =
> \{(i,j):s_{ij}^t\geq Q_q(s^t)\}.
> $$
>
> 观察热点、主干边和预测性能是否稳定。若只在某一个阈值下看起来合理，说明解释结果比较脆弱。
>
> ---
>
> ## C3. 节点级贡献验证
>
> 当前代码已经可以计算节点入意图、出意图和活跃度：
>
> $$
> I_i^{\mathrm{out}}
> =
> \sum_j A_{ij}^t,
> $$
>
> $$
> I_i^{\mathrm{in}}
> =
> \sum_j A_{ji}^t,
> $$
>
> $$
> I_i^{\mathrm{act}}
> =
> I_i^{\mathrm{out}}+I_i^{\mathrm{in}}.
> $$
>
> 可以进一步验证：遮蔽高活跃节点的出边，是否比遮蔽低活跃节点的出边造成更大误差。
>
> 定义：
>
> $$
> \Delta_i^{\mathrm{out}}
> =
> \mathrm{MAE}_{\mathrm{mask\ out}(i)}
> -
> \mathrm{MAE}_{\mathrm{full}}.
> $$
>
> 如果
>
> $$
> I_i^{\mathrm{out}}
> $$
>
> 与
>
> $$
> \Delta_i^{\mathrm{out}}
> $$
>
> 正相关，说明“源节点”解释具有预测意义。
>
> ---
>
> # D. 外部一致性验证
>
> 如果有道路拓扑、车道方向、路网距离或真实传播链，可以做更强验证。
>
> ## D1. 与滞后相关一致性
>
> 候选边本身已经包含方向滞后相关：
>
> $$
> \rho_{i\to j}^{\mathrm{lag}}
> =
> \max_{\tau}
> \operatorname{Corr}
> \left(
> z_i^{t-\tau},z_j^t
> \right).
> $$
>
> 可以检验：
>
> $$
> s_{ij}^t
> $$
>
> 是否与
>
> $$
> \rho_{i\to j}^{\mathrm{lag}}
> $$
>
> 在统计上正相关：
>
> $$
> \operatorname{Spearman}
> \left(
> s_{ij}^t,
> \rho_{i\to j}^{\mathrm{lag}}
> \right)>0.
> $$
>
> 更重要的是比较方向差异：
>
> $$
> \Delta \rho_{ij}
> =
> \rho_{i\to j}^{\mathrm{lag}}
> -
> \rho_{j\to i}^{\mathrm{lag}}.
> $$
>
> 如果
>
> $$
> \Delta \rho_{ij}>0
> $$
>
> 时更容易出现
>
> $$
> s_{ij}^t>s_{ji}^t,
> $$
>
> 则说明意图方向与滞后传播方向一致。
>
> ---
>
> ## D2. 与道路物理方向一致性
>
> 如果有道路上下游关系，可以定义真实方向标签：
>
> $$
> y_{ij}
> =
> \begin{cases}
> 1,& i\to j\text{ 是真实下游传播方向},\\
> 0,& \text{otherwise}.
> \end{cases}
> $$
>
> 然后用 $s_{ij}^t$ 做边分类分数，计算：
>
> $$
> \mathrm{AUC},
> \quad
> \mathrm{Precision@K},
> \quad
> \mathrm{Recall@K}.
> $$
>
> 例如：
>
> $$
> \mathrm{Precision@K}
> =
> \frac{
> |\mathrm{TopK}(s)\cap E_{\mathrm{true}}|
> }{
> K
> }.
> $$
>
> 这是最强的可靠性验证之一。
>
> ---
>
> ## D3. 与 Granger 因果或 Transfer Entropy 对比
>
> 可以用传统时间序列方法独立构造参考图：
>
> $$
> E_{\mathrm{GC}},
> \quad
> E_{\mathrm{TE}}.
> $$
>
> 然后比较：
>
> $$
> \mathrm{Overlap}
> =
> \frac{
> |E_I^t\cap E_{\mathrm{GC}}|
> }{
> |E_I^t|
> }.
> $$
>
> 或者比较排序相关：
>
> $$
> \operatorname{Spearman}
> \left(
> s_{ij}^t,
> \mathrm{GC}_{ij}
> \right).
> $$
>
> 注意这不是要求二者完全一致，因为 DEIG-TCN 的 $s_{ij}^t$ 是预测任务驱动的动态依赖，不等同于严格因果。但如果两者高度冲突，就需要进一步检查模型解释是否可信。
>
> ---
>
> # 4. 我最建议你补充的验证组合
>
> 如果要让论文或实验更有说服力，我建议至少补这五个：
>
> ## 第一，删除曲线
>
> $$
> p=1\%,5\%,10\%,20\%,30\%,50\%.
> $$
>
> 报告：
>
> $$
> \Delta_{\mathrm{top}}(p),
> \quad
> \Delta_{\mathrm{rand}}(p),
> \quad
> \Delta_{\mathrm{low}}(p).
> $$
>
> ## 第二，插入曲线
>
> 只保留 top $p\%$ 意图边，看少量高分边是否足以维持预测性能。
>
> ## 第三，随机重复与置信区间
>
> 随机遮蔽重复 $R=30$ 或 $R=50$ 次：
>
> $$
> \mathrm{MAE}_{\mathrm{rand}}
> =
> \mu\pm 95\%\mathrm{CI}.
> $$
>
> ## 第四，多种模型消融
>
> 至少比较：
>
> $$
> \text{Full DEIG-TCN},
> $$
>
> $$
> \text{without dynamic intent},
> $$
>
> $$
> \text{uniform edge weights},
> $$
>
> $$
> \text{static KNN only},
> $$
>
> $$
> \text{shuffled scores},
> $$
>
> $$
> \text{reverse scores}.
> $$
>
> ## 第五，多随机种子稳定性
>
> 报告：
>
> $$
> \mathrm{Jaccard@TopK},
> \quad
> \mathrm{Spearman}(s^{(r_1)},s^{(r_2)}).
> $$
>
> ---
>
> # 5. 可以形成的最终验证结论模板
>
> 如果补完这些实验，意图子图可靠性可以从三个层面证明：
>
> $$
> \boxed{
> \text{有效性}
> }
> $$
>
> 高意图边被删除后预测误差显著上升：
>
> $$
> \Delta_{\mathrm{top}}
> >
> \Delta_{\mathrm{random}}
> >
> \Delta_{\mathrm{low}}.
> $$
>
> $$
> \boxed{
> \text{稳定性}
> }
> $$
>
> 不同随机种子、不同训练切分、轻微数据扰动下，top 意图边保持较高重合：
>
> $$
> Jaccard@K \text{ 较高},
> \quad
> Spearman(s)\text{ 较高}.
> $$
>
> $$
> \boxed{
> \text{合理性}
> }
> $$
>
> 高意图边与滞后相关、道路方向、物理邻近、热点传播区域相一致：
>
> $$
> s_{ij}^t
> \leftrightarrow
> \rho_{i\to j}^{\mathrm{lag}},
> \quad
> \text{road direction},
> \quad
> \text{traffic hotspots}.
> $$
>
> ---
>
> # 6. 一句话总结
>
> 当前方法主要通过：
>
> $$
> \boxed{
> \text{mask top / random / low 意图边后的 MAE 变化}
> }
> $$
>
> 来验证意图子图是否有效。这个方向是正确的，但还不够充分。更完整的验证应增加：
>
> $$
> \boxed{
> \text{删除曲线、插入曲线、置乱检验、反向边检验、多种模型消融、多随机种子稳定性、外部交通规律一致性}
> }
> $$
>
> 这样才能同时证明意图子图的：
>
> $$
> \boxed{
> \text{预测有效性}
> +
> \text{统计可靠性}
> +
> \text{交通解释合理性}.
> }
> $$
>
> -----
>
> > [文件不支持导出]
> > 这个实验结果你觉得如何
>
> 我先把这次运行的关键指标、边遮蔽结果和意图图稳定性一起检查一下，再给你一个“现在能说明什么、还不能说明什么”的判断。
>
> 目前已经有一个很清楚的积极信号：删除高分边的影响远大于随机边，而删除低分边几乎没有影响。不过，我还在核对这个差异到底有多强，以及哪些现象其实是阈值设定“固定出来”的。
>
> 整体看，我会把这次结果评价为：
>
> $$
> \boxed{\text{预测效果很好，意图边有效性出现了明确正信号，但可靠性证据还属于初步阶段。}}
> $$
>
> 作为一次可行性实验，我认为是成功的；但要作为论文中“意图子图可靠”的核心证据，目前还不够扎实。
>
> ## 1. 最值得肯定的结果
>
> 测试集上：
>
> $$
> \mathrm{MAE}_{\mathrm{DEIG}}=0.08411,
> \qquad
> \mathrm{MAE}_{\mathrm{Persistence}}=0.25824.
> $$
>
> 相对 persistence，误差降低了
>
> $$
> \frac{0.25824-0.08411}{0.25824}
> \approx 67.43\%.
> $$
>
> 也就是说，模型误差只有 persistence 的约
>
> $$
> 32.57\%.
> $$
>
> 这个提升非常明显，说明模型确实学习到了比“未来等于当前”更强的预测规律。当前实验包含 100 个节点、488 条候选边和 1017 个测试窗口，测试规模也不算太小。fileciteturn1file0L11-L27
>
> 不过，MAE 是标准化空间中的误差，所以暂时还不能直接解释为实际交通流量单位中的误差。
>
> ## 2. 意图边消融结果是“方向正确”的
>
> 消融结果为：
>
> | 实验                |      MAE | 相对完整模型变化 |
> | ------------------- | -------: | ---------------: |
> | 完整模型            | 0.084111 |                — |
> | 删除 Top 10% 意图边 | 0.085581 |       $+1.748\%$ |
> | 删除随机 10% 边     | 0.084201 |       $+0.107\%$ |
> | 删除 Low 10% 意图边 | 0.084111 |         约 $0\%$ |
>
> 而且，高分边删除造成的误差增量约为随机删除的
>
> $$
> \frac{0.085581-0.084111}
> {0.084201-0.084111}
> \approx 16.39
> $$
>
> 倍。
>
> 因此出现了非常理想的排序关系：
>
> $$
> \boxed{
> \mathrm{MAE}_{\mathrm{mask\ top}}
> >
> \mathrm{MAE}_{\mathrm{mask\ random}}
> >
> \mathrm{MAE}_{\mathrm{mask\ low}}
> }
> $$
>
> 这说明边分数至少具备不错的**预测贡献排序能力**：模型给高分的边确实比低分边更重要。fileciteturn1file0L21-L27
>
> 但也要客观看待：删除最高 10% 边只造成约 $1.75\%$ 的相对退化，属于“明确但幅度不大”。所以目前更适合表述为：
>
> $$
> \boxed{\text{高意图边具有显著更高的相对重要性}}
> $$
>
> 而不是直接说：
>
> $$
> \text{意图子图已经完全恢复了真实交通传播结构。}
> $$
>
> ## 3. Low 边完全无影响既是好现象，也需要警惕
>
> 删除最低 10% 边后，MAE 与完整模型完全相同。这可能意味着模型成功把大量无用边压到了接近零，高分边和低分边分离得很好。
>
> 但还存在另一种解释：预测时使用的是入边归一化权重，
>
> $$
> \widetilde s_{ij}
> =
> \frac{s_{ij}}
> {\sum_p s_{pj}+\varepsilon},
> $$
>
> 而训练又对原始分数施加稀疏正则。模型可能把所有原始分数整体压小，只保留相对排序。因此，最低分边可能已经小到几乎没有任何数值贡献。
>
> 所以还需要同时检查：
>
> $$
> s_{ij}\text{ 的分布},
> \qquad
> \widetilde s_{ij}\text{ 的分布},
> $$
>
> 特别是最大值、均值、分位数、熵和每个节点的有效入边数量。
>
> ## 4. 意图图密度恒定不是稳定性的证据
>
> 97 个窗口中的意图图密度全部是
>
> $$
> 0.00494949.
> $$
>
> 对于 $N=100$，可能的非自环有向边数为
>
> $$
> 100\times99=9900.
> $$
>
> 因此每个窗口保留的边数是
>
> $$
> 0.00494949\times9900=49.
> $$
>
> 候选边总数为 488，所以每个窗口固定保留约
>
> $$
> \frac{49}{488}=10.04\%
> $$
>
> 的候选边。这恰好来自 $q=0.90$ 的分位数阈值，而不是模型自动学出了稳定密度。换句话说：
>
> $$
> \boxed{\text{密度恒定是阈值机制决定的，不是实验发现。}}
> $$
>
> 代码也是对每个窗口使用第 90 百分位阈值后，再用保留边数除以 $N(N-1)$。fileciteturn2file0L49-L57
>
> 热点数量在 97 个窗口中从 3 到 12 变化，平均约为
>
> $$
> 6.54\pm1.79,
> $$
>
> 说明虽然边数固定，但边的空间组合和连通结构可能在变化。这是一个不错的动态性信号。fileciteturn1file1L60-L157
>
> 不过，仅看热点数量还不能证明图真的在合理变化。需要进一步计算连续窗口的边集合重合度：
>
> $$
> J_t=
> \frac{|E_I^t\cap E_I^{t+1}|}
> {|E_I^t\cup E_I^{t+1}|},
> $$
>
> 以及边分数时间相关性。
>
> ## 5. 当前最重要的技术问题：动态滞后特征实际没有生效
>
> 这次实验设置为：
>
> $$
> W=3,\qquad H=1.
> $$
>
> 也就是只使用 3 个历史时刻预测下一个时刻。fileciteturn3file1L63-L74
>
> 但是动态滞后相关函数规定：参与相关系数计算的序列长度小于 4 时，直接跳过：
>
> $$
> \texttt{if x.size < 4: continue}.
> $$
>
> 当 $W=3$ 时，即使滞后量为 1，也只有
>
> $$
> |x|=W-1=2<4.
> $$
>
> 因此所有窗口中都会得到
>
> $$
> D_{ij}^t=0,
> \qquad
> D_{ij}^t-D_{ji}^t=0.
> $$
>
> 也就是说，当前 6 维动态边特征中的前两维：
>
> $$
> \left[
> D_{ij}^t,\,
> D_{ij}^t-D_{ji}^t
> \right]
> $$
>
> 实际上始终为零。fileciteturn3file0L11-L22 fileciteturn3file0L34-L50
>
> 这是当前实验最值得优先修正的问题。现在的模型仍然具有动态性，因为节点时空表示、异常强度和趋势特征随窗口变化，但它并没有真正验证你设计的“窗口级方向滞后相关”机制。
>
> 至少需要：
>
> $$
> W\geq5
> $$
>
> 才能让滞后 1 的相关系数有机会计算；若希望滞后 1、2、3 都有至少 4 个样本，则应满足：
>
> $$
> W-\tau_{\max}\geq4
> \quad\Longrightarrow\quad
> W\geq7.
> $$
>
> 更合理的设置可以是：
>
> $$
> W\in\{12,24\},
> \qquad
> H\in\{1,3,6\}.
> $$
>
> ## 6. 预处理存在重复执行
>
> `load_csv()` 内部已经执行了一次 `Preprocessor.fit_transform()`，并将标准化后的 $Z$ 重新作为 `TrafficData.X` 返回。fileciteturn4file0L33-L51
>
> 随后主程序又执行了一次：
>
> $$
> \texttt{Preprocessor(...).fit\_transform(data)}.
> $$
>
> fileciteturn4file1L74-L82
>
> 因此当前数据至少经过了两次预处理。第二次标准化本身可能接近恒等变换，但分位数截断也会被重复执行，可能进一步压缩异常值，并影响最终 MAE 和异常传播特征。
>
> 建议让 `load_csv()` 只负责：
>
> $$
> \text{读取数据}+\text{缺失值基础处理},
> $$
>
> 所有标准化、截断和周期残差计算只在主程序中统一执行一次。
>
> ## 7. 当前遮蔽方式还需要改进
>
> 现在是把所有测试窗口的边分数拼成一个整体，然后从全部“窗口—边实例”中统一选 Top 10%、Low 10% 和一次随机 10%。fileciteturn2file0L18-L35
>
> 这可能导致高分边主要集中在少数窗口，最终 MAE 增量不能完全代表每个窗口的稳定规律。
>
> 更稳健的方法是对每个测试窗口分别执行：
>
> $$
> \Delta_t^{\mathrm{top}}
> =
> \mathrm{MAE}_t^{\mathrm{mask\ top}}
> -
> \mathrm{MAE}_t^{\mathrm{full}},
> $$
>
> 然后报告：
>
> $$
> \overline{\Delta}^{\mathrm{top}},
> \quad
> \operatorname{median}(\Delta_t^{\mathrm{top}}),
> \quad
> 95\%\text{置信区间}.
> $$
>
> 随机遮蔽应重复至少 30 次：
>
> $$
> \Delta_{\mathrm{random}}^{(1)},\ldots,
> \Delta_{\mathrm{random}}^{(30)}.
> $$
>
> 由于这些窗口高度重叠，置信区间最好使用时间块 bootstrap，而不是把 1017 个窗口视为完全独立样本。
>
> ## 综合评价
>
> 我会分别给出两个评价：
>
> $$
> \boxed{\text{作为模型可行性实验：7.5/10}}
> $$
>
> 原因是预测提升非常大，而且遮蔽结果满足理想排序，高意图边明显比随机边和低意图边重要。
>
> $$
> \boxed{\text{作为意图子图可靠性证明：5/10}}
> $$
>
> 原因是目前只有一次运行、一个预测步长、一个遮蔽比例、一次随机遮蔽；图密度是阈值固定的；更关键的是 $W=3$ 导致动态滞后相关特征完全失效，同时预处理还执行了两次。
>
> 最准确的实验结论可以写成：
>
> > 当前实验表明，DEIG-TCN 在单步预测任务上明显优于 persistence 基线，且边意图分数能够有效区分高贡献边和低贡献边。删除高意图边造成的误差增量约为随机边的 16.4 倍，说明意图评分具有初步的预测有效性。但由于窗口长度较短、动态滞后特征未实际激活、随机消融次数不足以及缺少多种子和外部结构验证，目前结果应视为意图子图可靠性的初步证据，而非最终结论。





#### 20260628_181022

w=12,h=1

效果极差

## 260708

### 测试260706

**已加入因果意图融合**

当前意图图保存的 intent_graphs_data 是只有show时间步的

修改了读取的数据集的问题，但是输出的演示图片不太理解，且节点布局有问题

绘制箭头方向反了

`@file ratcgf/viz_inputs.py` `@file ratcgf/viz_causal_style.py` `@file ratcgf/visualize.py` `@file ratcgf/views.py` `@file ratcgf/main.py`  1. 260706\ratcgf\outputs\run_20260709_110315\figures\views_target_ref 下的 view_ref_causal_inactive因果方向不对 2. 输入的意图图、意图残差图、因果意图共识图、意图持续图、意图新增图、意图消退图都标明日期和小时  3.所有绘图，被选中的节点（有边的）都标红，白框红底白字，大小表示节点初度、入度，出度、入度越大节点越大；未选中的节点一律用灰色表示； 因果边用红白蓝过度色条，红色表示正因果影响，蓝色表示负因果影响，0处是白色  3. 画框不要用正方形，匹配节点坐标分布的横纵比例，应该是个长方向

### 测试260706_2

#### **调整绘图问题**

> ## 一、节点大小
>
> 节点大小都是"基础值 + 按某个强度归一化后的增量",公式各图不同:
>
> | 图 / 函数                                   | 文件:行                     | 设置代码                                      | 说明                                                    |
> | ------------------------------------------- | --------------------------- | --------------------------------------------- | ------------------------------------------------------- |
> | 参考样式(输入因果/意图、`views_target_ref`) | `viz_causal_style.py:130`   | `sizes = 100 + (strength / vmax_n) * 300`     | 最小 100,最大 400;`strength`=该节点所连边 \|效应\| 之和 |
> | `views_97steps` 各视图                      | `visualize.py:139`          | `sizes = 10 + 60 * (deg / (deg.max()+_EPS))`  | 最小 10,最大 70;按节点度                                |
> | 融合图 `fused_SD`(右图)                     | `visualize.py:191`          | `sizes = 15 + 120 * (sc-sc.min())/(...)`      | 按 node score                                           |
> | 部署子图 候选点/已部署点                    | `visualize.py:210` / `:231` | `s=12`(灰色候选) / `s=90`(红色部署)           | 固定值                                                  |
> | `viz_inputs.py` 因果/意图/并排              | `viz_inputs.py:105`         | `sizes = 25 + 130 * (ns-ns.min())/(rng+1e-8)` | 最小 25,最大 155                                        |
>
> 改法举例:想让参考样式节点整体变大,把 `viz_causal_style.py:130` 的 `100` 和 `300` 调大即可(前者是下限,后者是浮动幅度)。
>
> ## 二、线宽
>
> 要分清两种"线":**箭头/边线本身的粗细** 和 **节点白色描边(边框)的粗细**。
>
> **边线粗细:**
>
> | 图 / 函数                         | 文件:行                                                | 设置代码                                                     |
> | --------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------ |
> | 参考样式(nx 画边)                 | `viz_causal_style.py:118-119`(默认参数) + `:179`(计算) | `base_width=1.0`、`width_scale=5.0`;`width = base_width + abs(val)*width_scale` |
> | `views_97steps`/融合(quiver 箭头) | `visualize.py:87`(默认 `width=0.004`)、`:109`          | quiver 的 `width` 是相对轴宽的比例,不是磅值                  |
> | 部署子图(quiver 箭头)             | `visualize.py:222`                                     | `width=0.005`                                                |
> | `viz_inputs.py`(nx 画边)          | `viz_inputs.py:112`                                    | `widths = 0.4 + 2.8 * np.abs(w)/wabs`                        |
>
> **节点白色描边粗细(第 5 点要求的白框):**
>
> | 图                       | 文件:行                     | 代码             |
> | ------------------------ | --------------------------- | ---------------- |
> | 参考样式节点             | `viz_causal_style.py:146`   | `linewidths=1.0` |
> | `views_97steps`/融合散点 | `visualize.py:146` / `:195` | `linewidths=0.6` |
> | 部署点白框               | `visualize.py:232`          | `linewidths=1.0` |
> | `viz_inputs.py` 节点     | `viz_inputs.py:145`         | `linewidths=0.6` |
>
> 另外箭头头部大小在 quiver 的 `headwidth/headlength`(如 `visualize.py:110`)、nx 边的 `arrowsize`(`viz_causal_style.py:182` 为 12)控制,和线宽是分开的参数。
>
> ## 三、字体大小
>
> 字体是**两层**控制,建议优先改常量:
>
> **第 1 层——每个模块顶部的字号常量(改这里最省事,影响该模块所有图):**
>
> - `viz_causal_style.py:36-39` → `_TITLE_FS=16`、`_LABEL_FS=14`、`_NODE_LABEL_FS=14`、`_TICK_FS=12`
> - `visualize.py:53-56` → `_TITLE_FS=15`、`_LABEL_FS=13`、`_TICK_FS=11`、`_NODE_LABEL_FS=12`
> - `viz_inputs.py:33-36` → 同 visualize 一套
>
> 这四个常量分别对应:标题、坐标轴/色条标签、节点标签、刻度/色条刻度。它们被各处的 `fontsize=_XXX_FS` 引用(标题如 `viz_causal_style.py:234`;节点标签如 `:155` 和 `visualize.py:129`、`viz_inputs.py:149`;色条如 `:204-205`)。
>
> **第 2 层——全局 rcParams 默认值(兜底,凡是没显式指定 fontsize 的文字都吃这套):**
>
> - `main.py:92-98`(主流程运行时设置):`font.size=15`、`axes.titlesize=17`、`axes.labelsize=15`、`xtick/ytick=13`、`legend=13`、`figure.titlesize=19`
> - 各模块内的 `_apply_rc()`:`viz_causal_style.py:52-58`、`visualize.py:67-72`、`viz_inputs.py:48-53`——把上面的常量写进 rcParams,保证单独运行某个模块时也生效。
>
> 一个注意点:`main.py` 的 rcParams(15/17/…)和各模块常量(16/14/…、15/13/…)是**两套数值**,标题这类显式写了 `fontsize=_TITLE_FS` 的地方以模块常量为准,`main.py` 的 rcParams 只在没显式指定时兜底。要全局统一放大,最稳妥是同时把模块常量和 `main.py:92-98` 一起调,或者干脆把各处 `fontsize=_XXX_FS` 去掉、只留 rcParams 一套来源。
>
> 需要的话,我可以把这三类参数统一抽到一个配置区(比如一个 `style.py` 或 config 里的字段),这样只改一处就能全局生效,避免现在多套数值分散的情况。

#### run_20260709_162857

![deployment_subgraph_20260709_162857](log.assets/deployment_subgraph_20260709_162857.png)

![fused_SD_20260709_162857](log.assets/fused_SD_20260709_162857.png)

1. fused_SD 是怎么来的，获得的融合权重和部署图有什么关系

> $S_D^t$ 的**来源**以及它和**部署图$G_D^t$** 的关系说清楚。
>
> ## 一、$S_D^t$ 是怎么来的
>
> (S_D^t)（融合边分数矩阵）是整个 RA-TCGF 模型前向传播四个模块串联的产物。看 `model.py` 的 `RATCGF.forward`，链路如下：
>
> **模块 1（残差视图构建）** — `build_views`
> 输入因果图 (A_C) 和三个时刻的意图图 $A_I^{t-r}, A_I^t, A_I^{t+r}$，构造出候选视图集 `views`$C、I、CI、R、CmI、P、N$、消退视图 `decay`、结构统计 `q_struct`。同时用 `candidate_edge_index` 取候选边、`edge_feature_tensor` 生成逐边特征 (b_{ij})。
> **模块 2（时空编码）** — `MultiViewEncoder`
> 输入历史流量序列 `Xseq [L,N,F]` 和各视图，输出每视图的节点表征 (H_m^t) 和视图级摘要 (g_m^t)：
>
> ```python
> H, g = self.encoder(sample["Xseq"], views)
> ```
>
> **模块 3（主导性门控 → (S_D^t)）** — `DominanceGating`
> 这是 (S_D^t) 真正生成的地方。它把结构统计 `q_struct`、节点表征 `H`、视图摘要 `g`、视图、消退、候选边、边特征 (b_{ij}) 全部喂进去，输出门控权重和融合边分数：
>
> ```python
> gate = self.gating(q_struct, H, g, views, decay, edge_index, b_ij)
> ...
> "S_D": gate["S_D"], "s_edge": gate["s_edge"],
> "pi": gate["pi"], "gamma": gate["gamma"], "omega": gate["omega"],
> ```
>
> 这里的 `pi`、`gamma`、`omega` 就是你问的"融合权重"——它们是门控网络算出的、对各视图/各边的加权系数，用来把多个视图的边证据融合成单一的 (S_D^t)。（`module3_gating.py` 你没上传，所以具体融合公式看不到，但从接口能确定：(S_D^t) = 门控权重对多视图边证据的加权融合。）
> **节点分数** — `node_scores`
>
> ```python
> h_D = gate["h_D"]
> mlp_term = self.score_head(h_D)
> node_sc = node_scores(gate["S_D"], mlp_term)
> ```
>
> 节点分数 $Score_i^t$ 由融合表征 $h_D$ 过 MLP 的项，加上 $S_D^t$ 的某种聚合共同决定。
>
> 所以一句话：**$S_D^t$ = 四模块前向的输出，核心是模块 3 用门控权重$(\pi,\gamma,\omega)$把多视图边证据融合成的单一有向加权边分数矩阵。**
>
> 补充一点训练侧的背景（`train.py`）：模型是自监督训练的，损失为下一时刻流量预测的 MSE 加上 $S_D^t$ 的 L1 稀疏正则：
> $
> \mathcal{L} = \text{MSE}(\hat y,\ y) + \lambda \cdot \frac{1}{|S_D|}\sum |S_D^t|
> $
> 这个稀疏项会压制 (S_D^t) 里的弱边，让融合分数矩阵更"干净"，间接影响部署时的候选边质量。
>
> ## 二、融合权重 / $S_D^t$ 和部署图 $G_D^t$ 的关系
>
> 关系是**上游 → 下游**：$S_D^t$ 和 $Score_i^t$ 是部署算法的**输入打分**，`greedy_deploy`（`deploy.py`）拿它们在预算约束下贪心地挑节点和边，产出 $G_D^t$。
>
> 看 `deploy.py` 的 `greedy_deploy`，分两阶段：
> **第一阶段——选节点（用 $Score_i^t$）**
> 按节点分数从高到低排序，逐个加入，直到花掉预算的 70%（预留剩下给边）：
>
> ```python
> order = np.argsort(-node_scores)
> for i in order:
>     if cost + c_node <= B * 0.7:
>         selected.append(int(i)); cost += c_node
>     if cost >= B * 0.7:
>         break
> ```
>
> 即节点选择只看 $Score_i^t$，受节点成本 $c_i$ 和预算上限 (0.7B) 约束。
>
> **第二阶段——选边（用 $S_D^t$）**
> 把 $S_D^t > 0$的候选边按融合分数降序排，只有**两个端点都已被选中**的边才允许部署，且总花费不能超预算 (B)：
>
> ```python
> ii, jj = np.where(S_dense > 0)
> cand = [(a, b, S_dense[a,b]) ...]; cand.sort(key=lambda x: -x[2])
> for i, j, s in cand:
>     if i in sel_set and j in sel_set:
>         if cost + c_edge <= B:
>             edges.append((i, j)); cost += c_edge
> ```
>
> 这正好对应文件头部注释里的整数规划约束：
> $
> \sum_i c_i p_i + \sum_{ij} c_{ij} q_{ij} \le B,\qquad q_{ij}\le p_i,\ \ q_{ij}\le p_j
> $
> （边要被部署，其两端节点必须都被部署）。贪心策略是这个精确整数规划的近似替代。
>
> **效用回报**——用 $S_D^t$ 和视图算部署质量：
>
> ```python
> u_track = jaccard(A_dep, I>0) + jaccard(A_dep, P>0)
> "deployed_score": sum(S_dense[i,j] for i,j in edges)
> ```
>
> `deployed_score` 就是被部署边的融合分数之和，衡量"部署方案吃到了多少融合证据"。
>
> ## 三、把整条链路串起来
>
> $$
> \underbrace{(A_C,\ A_I^{t-r:t+r},\ X)}*{\text{输入}}
> \xrightarrow{\text{模块1-2}}
> \underbrace{(H,\ g,\ b*{ij})}*{\text{多视图表征}}
> \xrightarrow[\pi,\gamma,\omega]{\text{模块3 门控融合}}
> \underbrace{S_D^t,\ Score_i^t}*{\text{融合打分}}
> \xrightarrow[\text{预算 }B]{\text{模块4 贪心}}
> \underbrace{G_D^t}_{\text{部署图}}
> $$
>
> 要点对照：
>
> | 量                           | 生成于                            | 在部署里的角色                                     |
> | ---------------------------- | --------------------------------- | -------------------------------------------------- |
> | 融合权重 (\pi,\gamma,\omega) | 模块 3 门控                       | 决定各视图边证据如何融合成 (S_D^t)（不直接进部署） |
> | (S_D^t)                      | 模块 3 输出                       | 部署第二阶段**选边**的打分 + 效用计算              |
> | (Score_i^t)                  | `node_scores`（(S_D^t) + MLP 项） | 部署第一阶段**选节点**的打分                       |
> | (G_D^t)                      | `greedy_deploy`                   | 最终部署方案（受预算 (B) 约束）                    |
>
> 这也回答了你上一个问题的悬念：`plot_deployment_subgraph` 里画的 `deploy["nodes"]`，正是这里第一阶段按 (Score_i^t) 贪心选出来的节点；`deploy["edges"]` 是第二阶段按 (S_D^t) 选出的边。而 `fused_SD` 右图那个 `active` 掩码跟这套预算贪心毫无关系，只是显示高亮——所以两张图的红点通常对不上。
>
> 需要我画一张从输入到 (G_D^t) 的模块数据流示意图吗？可以把 (S_D^t)、(Score_i^t)、融合权重和部署两阶段的依赖关系可视化出来。



2. run_20260709_162857\figures\views_target_ref==当前的掩码具体是怎么计算实现的==，怎么有的深线条粗，有的细 (先暂不管这个问题，run_20260709_162857\views_97steps看不出来)

   第一个时间步的意图持续、意图新增图、意图消退图都是怎么算出来的

   ![input_causal_graph_ref_20260709_162857](log.assets/input_causal_graph_ref_20260709_162857.png)

   ![input_intent_graph_ref_20260709_162857](log.assets/input_intent_graph_ref_20260709_162857.png)

   ![view_ref_causal_inactive_c168_20260709_162857](log.assets/view_ref_causal_inactive_c168_20260709_162857.png)

   ![view_ref_consensus_c168_20260709_162857](log.assets/view_ref_consensus_c168_20260709_162857.png)

   ![view_ref_intent_decay_c168_20260709_162857](log.assets/view_ref_intent_decay_c168_20260709_162857.png)

   ![view_ref_intent_new_c168_20260709_162857](log.assets/view_ref_intent_new_c168_20260709_162857.png)

   ![view_ref_intent_persist_c168_20260709_162857](log.assets/view_ref_intent_persist_c168_20260709_162857.png)

   ![view_ref_intent_residual_c168_20260709_162857](log.assets/view_ref_intent_residual_c168_20260709_162857.png)

   

![causal_inactive_step000_c1_20260709_162857](log.assets/causal_inactive_step000_c1_20260709_162857.png)

![consensus_step000_c1_20260709_162857](log.assets/consensus_step000_c1_20260709_162857.png)

![intent_decay_step000_c1_20260709_162857](log.assets/intent_decay_step000_c1_20260709_162857.png)

![intent_new_step000_c1_20260709_162857](log.assets/intent_new_step000_c1_20260709_162857.png)

![intent_persist_step000_c1_20260709_162857](log.assets/intent_persist_step000_c1_20260709_162857.png)

![intent_residual_step000_c1_20260709_162857](log.assets/intent_residual_step000_c1_20260709_162857.png)

#### run_20260709_175116

把主循环从"先节点后边"翻转为"边驱动、节点跟随"：

1. 取所有 $S_D^t[i,j]>0$ 的候选边，按分数降序排（可用 $Score_i+Score_j$ 做同分次序的 tie-break）。
2. 逐条尝试加入。加入边 ((i,j)) 的**增量成本**为
   $
   \Delta = c_{\text{edge}} + c_{\text{node}}\cdot\big|{k\in{i,j}: k\notin V_{\text{dep}}}\big|
   $
   即边成本，加上"这条边引入的、之前还没部署的端点"的节点成本。已经因别的边而部署过的端点不再重复计费——这自然实现了 (q_{ij}\le p_i,p_j)。
3. 若 $ \text{cost}+\Delta\le B$就接受这条边，并把新端点登记为已部署节点。
4. 最终节点集 $V_{\text{dep}}=\bigcup_{(i,j)\in E_{\text{dep}}}{i,j}$——**节点完全是被边带进来的**。

问题： 

1. ==成本计算的细节原理==

2. ==融合的细节==

3. ==为什么最终融合边的权重聚集在因果和意图的交集上，只覆盖了非常少的边，这个是不是需要调整模型==

   ==部署图高度聚焦到因果和意图的重合边上==



## 260731

1. 意图图不是所有时间步

   但是意图图好像没必要是所有时间步（但是可以计算一下有效的意图图时间步是多少，主要是要对齐时间标注）

2. 因果意图融合部分也需要订正一下时间步对齐

3. 融合图和部署图的阈值设定有问题，需要详细思考一下怎么定义

#### ==260628 已修正时间对齐方式为 按预测目标时间标记==

##### 20260731_205742_all

修改意图图title

![intent_graph_window_00_20260731_205742](log.assets/intent_graph_window_00_20260731_205742.png)

##### intent_graphs_data_20260731_205742.npz

保存所有时间步意图图

### 测试260731

输出多时间步部署

对齐、可定义 部署、中间过程可视化的时间步、窗口选择

发现意图图的输出只有168个时间步

#### run_20260731_205314 budget=30

目前先用部分意图图做实验，还未修正意图图时间步定义

调整融合图阈值，但是效果不合理：固定预算，仅在极少数时刻非常稀疏，其他时刻区别不大，需要详细梳理内部步骤流程

![deployment_metrics_20260731_205314](log.assets/deployment_metrics_20260731_205314.png)

上图是什么含义

![pos004919_20250724T230000_c0001_seq000_deployment_subgraph_20260731_205314](log.assets/pos004919_20250724T230000_c0001_seq000_deployment_subgraph_20260731_205314.png)

![pos004919_20250724T230000_c0001_seq000_fused_SD_20260731_205314](log.assets/pos004919_20250724T230000_c0001_seq000_fused_SD_20260731_205314.png)

问题： 

1. ==成本计算的细节原理==

2. ==融合的细节==

3. ==为什么最终融合边的权重聚集在因果和意图的交集上，只覆盖了非常少的边，这个是不是需要调整模型==

   ==部署图高度聚焦到因果和意图的重合边上==

4. ==验证意图图现在是否是全部时间步==

5. ==部署图的选择方法、阈值设计==



## 260930

1. 换数据集，5min颗粒
2. 重新构造实验思路，梳理实验计划，逐步完善算法框架

## 261011

1. 因果图：在 1h 数据中，`τ=1` 表示前 **60 分钟**；

已完成5min数据集构造

| 缺失区间               | 每个传感器缺失点数 | 涉及传感器  |
| ---------------------- | ------------------ | ----------- |
| 2025-02-01 05:45—23:55 | 219                | 全部 100 个 |
| 2025-03-09 02:00—02:55 | 12                 | 全部 100 个 |

## 261012

1. 更新意图图
   1. 重跑旧代码：
      1. 结果分析，结果的含义
      2. 代码和方案是否一致 
   2. 度量指标 
   3. 参考基线
   4. 消融*

#### gpt

| 项目       | `intent_5m_smoke`                | `intent_5m_full_stride12`        |
| ---------- | -------------------------------- | -------------------------------- |
| 使用数据   | 前 **2 天**，576 个采样值        | 全部 **212 天**，61,056 个采样值 |
| 窗口间隔   | 每 **5 分钟**一个窗口            | 每 **60 分钟**一个窗口           |
| 保存窗口数 | 573                              | 5,088                            |
| 训练轮数   | 2                                | 2                                |
| 绘制图片   | 10 个测试窗口                    | 10 个测试窗口                    |
| 用途       | 检查程序、时间标签和输出是否正常 | 检查完整数据范围下的旧模型表现   |

| 项目         | smoke               | full                |
| ------------ | ------------------- | ------------------- |
| 实际读取时间 | 2 天                | 212 天              |
| `max_steps`  | 576                 | 未设置              |
| stride       | 1                   | 12                  |
| 训练窗口数   | 342                 | 3053                |
| 每个窗口     | 3 个历史步预测 1 步 | 3 个历史步预测 1 步 |

| 文件                             | 数量 | 内容                               |
| -------------------------------- | ---- | ---------------------------------- |
| `candidate_directed_graph`       | 3    | 候选图，PNG/SVG/PDF                |
| `intent_graphs_10windows`        | 3    | 10 个窗口的组合图                  |
| `intent_graph_window_00` 至 `09` | 30   | 10 张独立意图图，各 3 种格式       |
| `training_loss`                  | 3    | 训练损失图                         |
| `edge_mask_ablation`             | 3    | 旧代码已有的删边检查图             |
| `intent_graph_density`           | 3    | 图密度曲线                         |
| `sensor_timeseries`              | 3    | 传感器序列图                       |
| `edge_intent_scores.npz`         | 1    | 所有运行窗口的候选边分数与时间信息 |
| `intent_graphs_data.npz`         | 1    | 阈值筛选后的图数据                 |
| `model_params.npz`               | 1    | 本次训练的参数                     |
| `run_summary.json`               | 1    | 配置和评价结果                     |
| `data_spec.json`                 | 1    | 数据规格                           |
| `run_log.txt`                    | 1    | 控制台日志                         |
| `timeseries_data.csv`            | 1    | 预处理后的序列副本                 |
| `sensor_coords.csv`              | 1    | 坐标副本                           |

##### 1. 流程分析与结果含义

1. 能给出24年往后的指标参考文献和基线方法参考文献吗

   > **1. 2024 年及之后的指标和基线文献**
   >
   > 按照我们的定义——“当前可得历史中，对未来预测有贡献的动态边关系”——更合适的是图解释忠实度、结构恢复和时间事件评价，而不是驾驶员行为意图分类指标。
   >
   > 指标参考：
   >
   > | 文献                                                         | 年份与发表渠道            | 对我们有什么用                                               |
   > | ------------------------------------------------------------ | ------------------------- | ------------------------------------------------------------ |
   > | Zheng 等，*Towards Robust Fidelity for Evaluating Explainability of Graph Neural Networks* | **ICLR 2024**             | 指出直接删除解释子图可能造成分布偏移；通过随机、部分删除/保留来评价边的重要性。[论文](https://proceedings.iclr.cc/paper_files/paper/2024/file/34293d684b1012ed45c3274b4a7edc00-Paper-Conference.pdf) |
   > | Li 等，*Graph Neural Network Explanations are Fragile*       | **ICML 2024**             | 说明预测仍正确时，解释也可能因轻微结构扰动而明显变化；适合参考其鲁棒性检验。[论文](https://proceedings.mlr.press/v235/li24bd.html) |
   > | Boniol 等，*VUS: Effective and Efficient Accuracy Measures for Time-Series Anomaly Detection* | **The VLDB Journal 2025** | 提供 VUS-PR/VUS-ROC，考虑连续事件区间和时间错位；可用于有真值的边活动区间评价，不能直接用来评价无标签意图图。[论文](https://link.springer.com/article/10.1007/s00778-025-00907-x) |
   >
   > 建议短期优先采用第一篇的思路。以高分边集合 $Q_t$ 为对象，随机抽取其中比例为 $\alpha$ 的边 $R_{\alpha,t}$，计算：
   >
   > $ D_\alpha= \frac1M\sum_t \mathbb E_{R_{\alpha,t}} \left[ L_t(f(X_t,S_t^{(-R_{\alpha,t})})) - L_t(f(X_t,S_t)) \right]. $
   >
   > 这是**我们对交通回归任务的适配公式，不是论文原分类公式**。同预算随机边作为对照，评价“高分边是否比随机边更有预测贡献”。不能把它叫作真实意图识别准确率。
   >
   > 较新的基线：
   >
   > | 方法           | 文献                                                         | 适用范围                                                     |
   > | -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
   > | **STDN**       | Cao 等，*Spatiotemporal-aware Trend-Seasonality Decomposition Network for Traffic Flow Forecasting*，**AAAI 2025**。[论文](https://ojs.aaai.org/index.php/AAAI/article/view/33247)、[作者代码](https://github.com/roarer008/STDN) | 动态图与趋势、周期建模。**建议作为当前意图图模块的第一个外部基线候选** |
   > | **DLF / DSTG** | Wang 等，*Towards Dynamic Spatial-Temporal Graph Learning: A Decoupled Perspective*，**AAAI 2024**。[论文](https://ojs.aaai.org/index.php/AAAI/article/view/28759)、[作者代码](https://github.com/UnderReview24/DLF) | 面向演化网络和分布变化，包含持续适应训练；与固定节点短窗动态图并非完全相同 |
   > | **MoGERNN**    | Zhou 等，*MoGERNN: An Inductive Traffic Predictor for Unobserved Locations*，**TRC 2025**。[论文](https://www.sciencedirect.com/science/article/abs/pii/S0968090X25000841)、[作者代码](https://github.com/ZJU-TSELab/MoGERNN) | 部分观测到全路网预测，更贴近后续有限观测和故障实验           |
   >
   > 我的选择建议更新为：**当前建图模块先考察 STDN；后续有限观测系统考察 MoGERNN。** Graph WaveNet 可以保留为经典辅助对照，但满足不了你提出的近期文献要求。
   >
   > 这些方法学习的是任务相关依赖，不能直接视为提供了真实“意图标签”。

2. 之前说 5min数据有缺失，现在意图图生成这里有考虑这个问题吗，数据集是怎么划分的

   | **缺失区间**           | **每个传感器缺失点数** | **涉及传感器** |
   | ---------------------- | ---------------------- | -------------- |
   | 2025-02-01 05:45—23:55 | 219                    | 全部 100 个    |
   | 2025-03-09 02:00—02:55 | 12                     | 全部 100 个    |

   > 划分依据是原始时间轴的 **60% / 20% / 20%**，不是随机划分：
   >
   > | 划分 | 完整数据的时间范围                   |
   > | ---- | ------------------------------------ |
   > | 训练 | 2025-01-01 00:00 至 2025-05-08 04:40 |
   > | 验证 | 2025-05-08 04:45 至 2025-06-19 14:15 |
   > | 测试 | 2025-06-19 14:20 至 2025-07-31 23:55 |

3. stride 和短程数据上限参数是什么意思；新增采样间隔和读取行数参数是什么用

   > | 参数              | 含义                          | 本次例子                      |
   > | ----------------- | ----------------------------- | ----------------------------- |
   > | `--step-minutes`  | 相邻 CSV 两行实际相隔多少分钟 | `5` 表示每行一个 5 分钟采样值 |
   > | `--stride`        | 相邻输入窗口向前移动多少行    | `12` 表示每次移动 60 分钟     |
   > | `--max-steps`     | 最多读取 CSV 开头多少行       | `576` 表示只读取前两天        |
   > | `nrows=max_steps` | 实现读取上限的 pandas 参数    | 和 `max-steps` 是同一机制     |

4. `intent_5m_full_stride12`  不合理吧，按照数据集划分，按照实际运行逻辑，我们只能获得前一小段时间的数据，不可能用全部数据吧，你觉得呢

5. 当前的窗口到底是怎么划分的，多少数据预测多少未来值，滑动的步数是？相关的配置在哪里

   > 两次运行都用了：
   >
   > $ W=3,\qquad H=1. $
   >
   > 即：**100 个节点各自的 3 个历史值，预测各自下一步 1 个未来值**。
   >
   > | 运行            | 窗口移动          |
   > | --------------- | ----------------- |
   > | `smoke`         | 1 行，即 5 分钟   |
   > | `full_stride12` | 12 行，即 60 分钟 |
   >
   > 配置在 [main.py 参数区 (line 98)](/D:/Tjnu-p/Mp/Intention Recognition/260628/main.py:98)，窗口生成在 [make_windows (line 53)](/D:/Tjnu-p/Mp/Intention Recognition/260628/main.py:53)。

6. 你的输出为什么没有放在Intention Recognition\260628\outputs下；只输出10张图也太少了，至少要统计观测1周-2周的量看看规律啊，到底有没有捕捉到局部的规律，和1h的数据集的区别

   > 只输出 10 张是我传了 `--viz-count 10`
   >
   > 对于连续 5 分钟输出：
   >
   > - 1 周约 **2,016 个窗口**；
   > - 2 周约 **4,032 个窗口**。

7. 解释一下什么训练窗口、完整运行数量，什么意思，还有什么批的大小。这些具体流程都是什么

8. 不是都用了5min间隔的数据吗，你的意图图不是5min间隔的意图图吗，为什么“完整运行展示的是 6 月、相隔 1 小时的窗口” 什么意思

9. `intent_graphs_10windows`  的各子图上标的参数的含义，什么target、input、E_I、符号，是怎么得来的，是否可靠

   > | 标记     | 含义                               | 如何得到                    |
   > | -------- | ---------------------------------- | --------------------------- |
   > | `input`  | 输入窗口首尾采样时间               | 起始时间加输入索引乘 5 分钟 |
   > | `target` | 要预测的未来时间                   | 当前输入末尾索引加 1        |
   > | (        | E_I                                | )                           |
   > | $\delta$ | 当前窗口筛边阈值                   | 全部候选边分数的 90% 分位数 |
   > | $s_{ij}$ | 从节点 \(i\) 指向 \(j\) 的模型分数 | 边评分网络的 sigmoid 输出   |
   > | 节点数字 | 数组中的节点索引                   | 0–99；不是原始站点编号      |

10. intent_5m_full_stride12 比intent_5m_smoke效果差的太多了

11. `training_loss`  现在是完全线性的下降

12. `edge_mask_ablation`  的含义是什么？作用是什么？能看出什么？

    > 它是在**同一个已训练模型**上，把部分边分数设为零，再观察测试 MAE：
    >
    > - `mask top-k`：屏蔽最高分的 10%；
    > - `mask random`：屏蔽等数量随机分数；
    > - `mask low-k`：屏蔽最低分的 10%；
    > - `full model`：不屏蔽。
    >
    > 理想迹象是：屏蔽高分边比随机边更伤害预测。
    >
    > 但当前实现有限制：
    >
    > 1. 把整个测试集的“窗口—边”分数混在一起排序，**不是每个窗口固定删 10%**。
    > 2. 随机对照只做一次。
    > 3. 只关闭意图消息权重，GAT 编码器仍使用候选图。
    > 4. 内部预测使用的数值精度与 `float32` 导出图不完全一致。
    > 5. 模型原本已弱于持续性，不能凭删边柱状图认定意图算法有效。
    >
    > 它是**固定模型的贡献检查**，不是重新训练后的结构消融，也不是实际传感器失效实验。

13. intent_5m_smoke的曲线密度相对稳定，这是真实情况，还是由于什么代码内部机制决定

    > **主要是代码机制决定的。**
    >
    > 488 条候选边按 90% 分位数筛选，分数无大量并列时通常保留 49 条。
    >
    > 密度定义为：
    >
    > $ density_t=\frac{|E_I^t|}{N(N-1)} =\frac{49}{100\times99} \approx0.004949. $
    >
    > smoke 展示的 10 个窗口都正好是这个值。
    >
    > 因此平稳密度不说明交通意图稳定。边数固定时，边身份和权重仍可能变化，应看边集合重合率、方向和热点位置。

14. intent_5m_full_stride12的曲线密度不稳定，第一个图特别低，后面几张图也没有区分，都完全一样

15. `sensor_timeseries`  是啥意思呢，随便选的传感器和时间段吗

    > 代码固定选：
    >
    > - 数组顺序的前 10 个节点；
    > - 数据开头的前 200 个采样值；
    > - 绘制入口预处理后的 \(Z\)。
    >
    > 不是随机选，也不是按热点选，主要用于输入检查。

16. `intent_graphs_data.npz`  `edge_intent_scores.npz`  是所有节点，所有时间窗口意图图吗

    > | 内容 | `edge_intent_scores.npz`        | `intent_graphs_data.npz` |
    > | ---- | ------------------------------- | ------------------------ |
    > | 节点 | 100 个节点坐标                  | 100 个节点坐标           |
    > | 边   | 固定候选图的 **488 条边**       | 各窗口筛选后的候选边     |
    > | 分数 | `scores`，形状为窗口数 × 488    | 每窗口保留边的权重       |
    > | 时间 | 输入、目标时间及索引            | 同样保存                 |
    > | 划分 | `split_id`：0训练、1验证、2测试 | 同样保存                 |
    > | 其他 | 候选边特征                      | 阈值、密度、热点数量     |

##### 2. 流程分析与结果含义

1. 这个模型，到底是用全部的前 60%数据进行训练，还是只用窗口数据进行预测，就是说模型预测性能到底和训练数据的量有没有关系，还是只和窗口的量有关系

2. 双向时间线性插值这个的作用是？？我哪有缺失吗，为什么要插值

3. 我有`missing_mask.csv`  吗，哪来的

4. 那不行换一下训练、测试、验证数据集的顺序呢，看看能不能把缺失数据避开，

5. “纯仿射变换通常会被再次标准化抵消”啥意思

6. `stride`  是12的话，那怎么得到的以5min间隔的意图图的，不得是以60min为间隔的意图图吗

7. “因此不是直接拿测试未来值训练。但是训练后还回头导出训练期的图，这些图使用了训练期后续数据拟合出的参数，**属于回顾性输出，不能当作当时在线产生的图**。”没看懂，就是说现在还是有监督呗，会用未来值监督当前预测？？还是什么意思，看不懂，不是**100 个节点各自的 3 个历史值，预测各自下一步 1 个未来值**吗

8. “已有长期全观测历史的离线模块检查”是什么意思，到底是什么流程

9. 输入值仍是 5 分钟粒度，但**每小时才抽一个窗口生成图**，中间 11 个窗口没有生成。  ，但是为什么统计下来的窗口量反而多得多呢

10. 涉及到批的计算我还是没理解，实际训练中到底是怎么运行的，一次用最多 16 个窗口计算损失，是同一节点的吗，还是不同节点的，这样是怎么更新的模型参数。完整运行到底为什么更新的更多？不都是W=3,H=1吗，完整运行到底多在哪了。批处理的话，损失到底是和谁算的 ，怎么算的

    > 当前：
    >
    > ```
    > bs = 16
    > ```
    >
    > 表示一次取最多 16 个窗口。
    >
    > 每个窗口都包含：
    >
    > ```
    > 100 个节点 × 3 个历史步
    > ```
    >
    > 所以一个 batch 的输入大致是：
    >
    > ```
    > 16 个窗口 × 100 个节点 × 3 个时间步
    > ```
    >
    > 这 16 个窗口通常来自不同时间点，但每个窗口都包含全部 100 个节点。
    >
    > 模型对每个窗口做：
    >
    > 1. 编码全部 100 个节点；
    > 2. 为候选边计算分数；
    > 3. 用边分数聚合消息；
    > 4. 预测全部 100 个节点的下一步值；
    > 5. 将 16 个窗口、共 1,600 个节点预测的损失合并。
    >
    > 例如每个窗口的预测损失是：
    >
    > $ L_b = \frac{1}{16\times100} \sum_{w=1}^{16}\sum_{i=1}^{100} \left|y_{w,i}-\hat y_{w,i}\right| $
    >
    > 然后：
    >
    > ```
    > 反向传播 batch 总损失
    > → 计算所有参数的梯度
    > → Adam 更新一次模型参数
    > ```

11. 当前窗口筛边阈值每个图都不同欸，全部候选边分数的 90% 分位数，那岂不是一定会固定边数

12. 这个没看懂

    | 检查                      | smoke 测试集 | 完整运行测试集  |
    | ------------------------- | ------------ | --------------- |
    | 全部边分数为零的窗口      | 0 / 116      | **244 / 1,017** |
    | 筛选出全部 488 条边的窗口 | 0            | **307**         |
    | 啥意思                    |              |                 |

13. 只训练了 **2 轮**  ？他意思是epoch=2吗

14. `mask top-k`  是高于另外3个，但是高出的并不多，另外3个还都一样

15. “把整个测试集的“窗口—边”分数混在一起排序”什么意思

16. “488 条候选边按 90% 分位数筛选，分数无大量并列时通常保留 49 条。”这样合理吗，是否考虑用绝对分数阈值？获得不固定边数的图

17. 意思是`edge_intent_scores.npz`  是一张图的边分数，那这个边分数取的是什么时刻的呢

18. `intent_graphs_data.npz`  各窗口筛选后的候选边，也就是只保留49条？

##### 3. 流程分析与结果含义

1. 排除缺失窗口会有什么影响吗
2. 所以，原始数据标准化过了，现在数据处理是不是可以不再标准化
3. "smoke 只覆盖 2 天，所以完整运行每轮看到的训练样本更多"? 什么意思，我理解了“模型对每个窗口做：
   1. 编码全部 100 个节点；
   2. 为候选边计算分数；
   3. 用边分数聚合消息；
   4. 预测全部 100 个节点的下一步值；
   5. 将 16 个窗口、共 1,600 个节点预测的损失合并。”

​        但是为什么会有只覆盖 2 天和完整运行的区别，不都是16个窗口吗，

1. “更合理的做法是对每个窗口分别：

   取该窗口最高 10% 边
   取该窗口最低 10% 边
   取该窗口随机 10% 边


   再逐窗口计算 MAE，最后求平均。” 这个对，之后改成这个

2. "应在验证集用以下方式确定：

   - 训练集/验证集分数的固定分位点；
   - 验证集删边贡献最大化；
   - 验证集预测损失与稀疏度的 Pareto 选择；
   - 或先做分数校准。
     我建议正式报告：

   ```
   Top-K：K=5%、10%、20%
   绝对阈值：δ_abs 由验证集预先确定"
   ```

   有道理



## 261013

检查：
修改后的数据集划分
设计方案和代码的出入（说是基本一致，可能需要按照代码整理方案）

初步实验结果：结果有问题，没有区分。用30天的数据开展，7个月的数据量太大

计划：代码新建文件夹“intent_graph_261013”，可以复用的代码就复制，各基线、初级方案、指标参数计算也给出详细文档md，文档中不要代码，用数学公式描述

## 261014

1. 指标是否充分：有矛盾点，6个基线不全都是构图的
   指标的对比核心到底是什么
2. 图表是否合理、充分

### comparison.csv

| 后缀    | 含义                                   |
| ------- | -------------------------------------- |
| `_mean` | 多次独立训练运行的平均值               |
| `_std`  | 多次独立训练 seed 之间的**样本标准差** |
| `_n`    | 这个指标实际参与汇总的独立运行数量     |

文档规定神经模型默认使用 3 个训练 seed，并用样本标准差汇总；Persistence 等确定性模型不需要重复相同训练。 

例如：

`test_mae_mean` = 各训练 seed 的测试集 MAE 平均值
 `test_mae_std` = 这些 seed 的测试集 MAE 样本标准差
 `test_mae_n` = 有多少个 seed 有有效测试 MAE。

| 列名                                | 含义                                                         |
| ----------------------------------- | ------------------------------------------------------------ |
| `model`                             | 模型名称，如 persistence、ridge_ar、local、fixed_graph、static_gate、dynamic_gate |
| `config_signature`                  | 本次模型/实验配置的唯一签名，用于判断是不是同一配置          |
| `runs`                              | 汇总了多少次运行                                             |
| `independent_seeds`                 | 独立训练 seed 数；主要针对神经模型                           |
| `device`                            | 实际训练计算设备，如 CPU / CUDA                              |
| `intervention_scope`                | 图干预作用范围的配置标记；当前图模型为 `single_message`      |
| `deterministic`                     | 是否属于确定性运行                                           |
| `available_splits`                  | 本次产物实际包含哪些分区，一般为 `train,val,test`            |
| `graph_ranking_status`              | 是否存在可用于 Top/Low 排名的边分数                          |
| `graph_intervention_status`         | 图干预是否实际计算                                           |
| `graph_message_ablation_status`     | “关闭全部图消息”实验是否计算                                 |
| `graph_ranking_intervention_status` | Top/Low/Random 排名干预是否实际计算                          |
| `graph_reason`                      | 上述图指标不适用时的原因                                     |
| `seed_std_kind`                     | `_std` 的定义，目前是独立训练 seed 之间的样本标准差          |

比如 Local 没有边门控，因此图排名属于 `not_applicable`；FixedGraph 虽有图消息，但所有边都是全一门控，没有学习出的 Top/Low 排序。文档对此作了明确区分。

| 列名根              | 含义                                | 趋势         |
| ------------------- | ----------------------------------- | ------------ |
| `mae`               | Mean Absolute Error，平均绝对误差   | **越小越好** |
| `rmse`              | Root Mean Squared Error，均方根误差 | **越小越好** |
| `mae_standardized`  | 标准化尺度 MAE                      | 越小越好     |
| `rmse_standardized` | 标准化尺度 RMSE                     | 越小越好     |

MAE 是文档规定的**主要预测指标**；RMSE 对少数大误差更加敏感。

| 列名                      | 含义                          |
| ------------------------- | ----------------------------- |
| `skill_mean`              | 相对于 Persistence 的预测技能 |
| `skill_standardized_mean` | 标准化尺度下的技能            |
| `train_skill_*`           | 训练集 Skill                  |
| `val_skill_*`             | 验证集 Skill                  |
| `test_skill_*`            | 测试集 Skill                  |

persistence_* ： 给 `skill` 提供基准分母

| 列名根                          | 含义                               |
| ------------------------------- | ---------------------------------- |
| `fit_seconds`                   | 模型拟合/训练耗时                  |
| `parameter_count`               | 可训练参数数量                     |
| `windows`                       | 主评价中的窗口数量                 |
| `valid_targets`                 | 实际参与误差计算的有效目标元素总数 |
| `epochs`                        | 实际训练 epoch 数                  |
| `best_epoch`                    | 验证指标最好的 epoch               |
| `prediction_seconds`            | 完整预测所需时间                   |
| `prediction_windows_per_second` | 每秒处理多少预测窗口               |
| `evaluation_seconds`            | 整个评价过程耗时                   |
| `fit_peak_gpu_allocated_mb`     | 训练时峰值 GPU 已分配显存，单位 MB |

| 列名根         | 含义                    |
| -------------- | ----------------------- |
| `gate_minimum` | 门控分数最小值          |
| `gate_maximum` | 最大值                  |
| `gate_mean`    | 平均门控分数            |
| `gate_q0`      | 0% 分位，也就是最小值   |
| `gate_q50`     | 中位数                  |
| `gate_q90`     | 90% 分位                |
| `gate_q99`     | 99% 分位                |
| `gate_q100`    | 100% 分位，也就是最大值 |

| 想回答的问题                   | 主要看这些列                                                 |
| ------------------------------ | ------------------------------------------------------------ |
| 谁预测最准？                   | `test_mae_mean`, `test_rmse_mean`, `test_skill_mean`         |
| 稳不稳定？                     | `test_mae_std`, `test_rmse_std`                              |
| 是否泛化？                     | `train_mae_mean`、`val_mae_mean`、`test_mae_mean`            |
| 图消息有没有被模型利用？       | `graph_all_messages_off_delta_mean`                          |
| 门控有没有塌缩？               | `gate_mean`, `gate_zero_fraction`, `gate_near_zero_fraction`, `gate_constant_window_fraction` |
| Dynamic 是否真的在变化？       | `gate_mean_score_change`, `p05/p10/p20_adjacent_top_jaccard` |
| Top 边是否比随机边重要？       | **`delete_excess`，希望 > 0**                                |
| Top 少量边是否足够？           | `keep_top_delta`，希望接近 0                                 |
| Top 是否比随机保留更好？       | **`keep_excess`，希望 < 0**                                  |
| 结果是不是随机抽边碰巧造成的？ | `delete_random_std`, `keep_random_std`                       |

### suite_20261013_221703_583502

使用 30 天、100 个节点的数据，历史窗口 12 步，预测未来 3 步；四个神经模型各运行 3 个种子，使用 GPU，训练上限 20 轮，部分因早停提前结束。

共保存 14 次模型运行、42 组三分区结果。我检查了各模型的数据指纹，并从保存的预测数组重新计算全部三分区 MAE，均与摘要一致。

测试集结果如下，± 表示三个训练种子的样本标准差：

| 模型        | MAE ↓              | RMSE ↓        | 相对 Persistence 的 MAE 改善 |
| ----------- | ------------------ | ------------- | ---------------------------- |
| Persistence | 16.303             | 30.729        | —                            |
| Ridge-AR    | 15.103             | 29.387        | 7.36%                        |
| Local       | 14.083 ± 0.210     | 27.037        | 13.62%                       |
| FixedGraph  | 13.722 ± 0.083     | **26.321**    | 15.83%                       |
| StaticGate  | 13.782 ± 0.179     | 26.408        | 15.46%                       |
| DynamicGate | **13.701 ± 0.109** | <u>26.397</u> | **15.96%**                   |

来源：[模型比较汇总](C:/Work-lcr/Tjnu-p/Mp/Intention-Recognition/intent_graph_261013/outputs/suite_20261013_221703_583502/comparison.csv)。

**本节点历史预测有效。** Local 明显优于 Persistence 和 Ridge，说明神经预测器学到了有用的时间规律。

**空间信息有增益。** FixedGraph 比 Local 的 MAE 再下降约 2.56%，支持加入邻居消息。但新增消息参数也改变了模型容量，所以不能把全部增益严格归因于图结构。

**动态门控的额外收益很小。** DynamicGate 比 FixedGraph 的 MAE 仅低约 **0.15%**，RMSE 反而略高；三个种子中，动态模型并非每次都胜出。因此，现在更准确的结论是：

> 固定图已经较有效，动态门控与固定图表现接近，尚无稳定优势。

未来 5、10、15 分钟的平均 MAE，DynamicGate 分别为 11.686、13.940、15.477，FixedGraph 为 11.674、13.981、15.510；差距都很小。

## 261015

### 绘图代码与训练接入

直接运行入口是 [plot_intent_sequences.py](C:/Work-lcr/Tjnu-p/Mp/Intention-Recognition/intent_graph_261013/plot_intent_sequences.py)。在 `intent_graph_261013` 目录运行：

```
python plot_intent_sequences.py
```

默认读取你指定的 `suite_20261013_221703_583502`，遍历全部模型、全部种子，保存连续 10 帧的 PNG／SVG／PDF。调整起点、数量和格式，例如：

```
python plot_intent_sequences.py --viz-start-test 20 --viz-count 97 --formats png
```

只绘制 DynamicGate 的 seed 0，使用已保存的验证阈值：

```
python plot_intent_sequences.py --models dynamic_gate --seeds 0 --mode saved_threshold
```

支持旧版 `--q` 分位数方式、精确 Top-K、节点标签、线宽、画布、DPI、坐标范围和固定颜色范围。每组输出包含逐帧图片、连续拼图、候选图、密度曲线、窗口索引 CSV 和 JSON 清单；同一序列共用颜色尺度。

.\run_construction.ps1 --protocol configs/protocol_construction_full.json --device cuda --epochs 50 --patience 8 --min-delta 0

.\run_construction.ps1 --protocol configs/protocol_construction_dev.json --models M3 M4 M5 M6 M7 --device cuda --batch-size 4 --epochs 100 --patience 15

### 绘图可以选择三种筛选方法

| 模式              | 规则                       | 特点                               |
| ----------------- | -------------------------- | ---------------------------------- |
| `topk`            | 每窗口保留最高的固定比例   | 精确预算；默认10%，488条中选49条   |
| `saved_threshold` | 使用该模型已保存的验证阈值 | 各窗口边数可变化，适合观察激活密度 |
| `window_quantile` | 每窗口重新取分位数         | 旧版方式；并列时可能保留更多边     |

在 `intent_graph_261013` 目录运行，例如绘制新实验 M7 seed 0：

```
python plot_intent_sequences.py --suite outputs/suite_20261015_090833_339433 --models m7 --seeds 0 --mode topk --fraction 0.1 --formats png
```

使用保存的验证阈值：

```
python plot_intent_sequences.py --suite outputs/suite_20261015_090833_339433 --models m7 --seeds 0 --mode saved_threshold --formats png
```

使用每窗口90%分位数：

```
python plot_intent_sequences.py --suite outputs/suite_20261015_090833_339433 --models m7 --seeds 0 --mode window_quantile --q 0.9 --formats png
```

删除 `--models m7 --seeds 0` 就会批量绘制该实验所有模型和种子。各模式生成新的图片目录，**不会改写原始权重或重新训练**。

目前独立绘图入口尚不支持直接输入任意固定阈值，例如 `--threshold 0.5`；`saved_threshold` 使用的是运行中已经保存的阈值。`--vmin/--vmax` 只调颜色范围，不负责筛边。

### suite_20261015_022730_638592

61,056 个五分钟时刻，约 212 天，从 2025-01-01 开始。训练配置是 50 轮上限、早停耐心 8、batch=32、CUDA。

已完成 21 个模型／种子运行：

- Persistence、Ridge-AR：各 1 次；
- Local、FixedGraph、StaticGate、DynamicGate、M1、M2：各 3 个种子；
- M3：只有 seed 0。

每个已完成运行都有训练、验证、测试分区结果。目录在 [本次 suite](C:/Work-lcr/Tjnu-p/Mp/Intention-Recognition/intent_graph_261013/outputs/suite_20261015_022730_638592)，当前状态见 [suite_summary.json](C:/Work-lcr/Tjnu-p/Mp/Intention-Recognition/intent_graph_261013/outputs/suite_20261015_022730_638592/suite_summary.json)。

**已完成模型的测试结果**

| 模型              | 测试 MAE ↓         | 测试 RMSE ↓ | 完成种子 |
| ----------------- | ------------------ | ----------- | -------- |
| Persistence       | 14.756             | 26.028      | 1        |
| Ridge-AR          | 13.311             | 23.600      | 1        |
| Local             | 12.004 ± 0.049     | 22.023      | 3        |
| FixedGraph        | 11.569 ± 0.057     | **21.186**  | 3        |
| StaticGate        | 11.690 ± 0.057     | 21.210      | 3        |
| DynamicGate（M0） | **11.545 ± 0.125** | 21.135      | 3        |
| M1                | 13.054 ± 0.251     | 23.718      | 3        |
| M2                | 12.973 ± 0.140     | 23.575      | 3        |
| M3                | 12.841             | 23.211      | 1        |

± 是三个种子的样本标准差。基线结果显示，神经模型和固定图整体优于 Persistence、Ridge；但 M1、M2 在这次运行中明显落后于 FixedGraph。M3 只有一个种子，不能和三种子结果等量比较。

**目前能得出的结论**

**1. M1、M2 这次没有显示出增加前置 GAT／TCN 的收益。**
两者测试 MAE 都约 12.97–13.05，而 FixedGraph 约 11.57。M3 seed 0 的 MAE 是 12.84，仍明显高于固定图。原因不能简单归结为七维特征：M1、M2 用 X1，M3 才切换到 X7；同时新增了前置 GAT，改变了信息路径与优化难度。

**2. M3 的门控排名有一定模型内作用，但稀疏子图不够充分。**
M3 seed 0 的 Top-10% 删除使 MAE 增加约 **0.852**，相对同预算随机删除额外增加约 **0.483**，说明排序不是完全随机。但只保留 Top-10% 时 MAE 增加约 **7.62**，表示这部分边无法维持完整模型预测。它支持“某些高分边对该模型有用”，不支持“Top-10% 已形成足够的预测子图”。而且这只是一个种子。

**3. 预测稳定不代表解释稳定。**
在更早完成的 DynamicGate 中，三个种子的测试 MAE 很接近，但关闭全部后置图消息后性能损失差异较大。这提示不同种子可能依赖图通道的程度不同；不能只看平均 MAE 就说学到稳定意图图。

各模型汇总见 [comparison.csv](C:/Work-lcr/Tjnu-p/Mp/Intention-Recognition/intent_graph_261013/outputs/suite_20261015_022730_638592/comparison.csv)，M3 的逐种子、逐分区信息见 [M3 seed 0 摘要](C:/Work-lcr/Tjnu-p/Mp/Intention-Recognition/intent_graph_261013/outputs/suite_20261015_022730_638592/m3/seed_0/run_summary.json)。

### suite_20261015_022730_638592

最新的 `suite_20261015_090833_339433` 已完整完成：**M3–M7 各 3 个种子，共 15 次运行**。目前最明确的结果是：**M5 的边感知 GAT 带来了主要预测改善，M7 平均最好，但动态边特征和时间注意力的收益并不均衡。**

**1. 两次实验可以联合解读，但不能合并排名**

| 项目                 | 旧实验 `022730`             | 新实验 `090833`   |
| -------------------- | --------------------------- | ----------------- |
| 数据范围             | 全量 212 天                 | 前 30 天          |
| 测试目标时段         | 6月19日—7月31日             | 1月25日—1月30日   |
| 训练 batch           | 32                          | 4                 |
| 最大 epoch／早停耐心 | 50／8                       | 100／15           |
| 完成情况             | 21 次，整体中断             | 15 次，全部完成   |
| 模型覆盖             | B1–B6、M1、M2及 M3 一个种子 | M3–M7，均三个种子 |

因此，旧 M0 的 MAE 11.545 和新 M7 的 12.802 **不能直接比较优劣**。数据时段、训练量和优化设置都不同；Skill 的分母也来自各自测试集，不能消除这些差异。

我检查了两次实验内各已完成运行的三分区目标指纹，内部一致。

**2. 新实验的预测结果**

以下为三个训练种子的均值；MAE 的 ± 为样本标准差。

| 模型 | 训练 MAE | 验证 MAE   | 测试 MAE ↓         | 测试 RMSE ↓ | 相对 Persistence 改善 |
| ---- | -------- | ---------- | ------------------ | ----------- | --------------------- |
| M3   | 11.390   | 12.691     | 14.782 ± 0.118     | 28.981      | 9.33%                 |
| M4   | 11.154   | 12.577     | 14.926 ± 0.218     | 29.784      | 8.45%                 |
| M5   | 9.495    | 11.533     | 12.875 ± 0.068     | 25.893      | 21.03%                |
| M6   | 9.458    | 11.677     | 13.121 ± 0.209     | 26.447      | 19.52%                |
| M7   | 9.646    | **11.450** | **12.802 ± 0.039** | **25.855**  | **21.47%**            |

来源：[新实验比较汇总](C:/Work-lcr/Tjnu-p/Mp/Intention-Recognition/intent_graph_261013/outputs/suite_20261015_090833_339433/comparison.csv)。

逐步解释：

- **M3→M4：静态边特征只加入评分器，没有带来测试收益。**训练、验证误差略降，测试 MAE 反而增加约 0.97%。有泛化不一致的迹象，不能只根据训练下降判定模块有效。
- **M4→M5：静态边特征加入前置 GAT，改善最明显。**测试 MAE 下降约 **13.74%**，三个种子均改善，验证集也明显改善。这是本轮最有力的增量证据。
- **M5→M6：加入显式动态边特征后平均变差。**测试 MAE 增加约 **1.92%**，种子波动也增大。目前没有支持这组特征提升预测的证据，但尚不能确定具体是噪声、冗余还是优化问题。
- **M6→M7：时间注意力汇总改善了结果。**测试 MAE 下降约 **2.43%**。不过，M7 相对更简单的 M5 只改善约 **0.56%**，一个种子还略逊于 M5；目前应同时保留 M5、M7，而不是认定所有组件都必要。

所有运行都因早停正常结束：实际训练 32–80 轮，最佳轮次为 17–65。不是因为没到 100 轮就算没完成。

联合旧实验来看，旧结果中 **M0 与 FixedGraph 很接近，M1/M2 加入普通 GAT 后反而变差**；新结果中 M5 加入边属性后明显恢复预测能力。这提示：**空间聚合采用什么信息和权重，可能比单纯增加 GAT／TCN 模块更关键。**这是跨实验的研究线索，还需要在相同协议下验证。

**3. 图权重的解释效果，比预测结果弱一些**

新实验 Top-10% 的结果如下：

| 模型 | 删除高分边的 MAE 增量 | 相对随机删除的额外损害 ↑ | 只保留高分边的 MAE 增量 | 保留高分相对随机的差异 ↓ |
| ---- | --------------------- | ------------------------ | ----------------------- | ------------------------ |
| M3   | +2.597                | +2.506                   | +5.339                  | −1.663                   |
| M4   | +1.382                | +1.254                   | +6.382                  | −1.959                   |
| M5   | +0.237                | +0.126                   | +5.883                  | +0.318                   |
| M6   | +0.828                | +0.408                   | +13.451                 | +0.079                   |
| M7   | +0.777                | +0.472                   | +9.130                  | −0.319                   |

这里有两个值得关注的现象：

- **预测最好，不等于门控排序最有解释力。**M5 预测很好，但高分边删除相对随机的优势较弱；仅保留高分边还略逊随机保留。
- **M7 有一定高分排序作用，但 Top-10% 子图仍不充分。**只保留这些边后，MAE 从约 12.802 升到 **21.932**，虽然略好于随机保留，仍远离完整模型。

这些干预只关闭**后置消息**，前置 GAT 仍然工作，不能解释为整个模型只使用所保留的图。

另外，M7 三个种子的预测很稳定，但全后置消息关闭后的 MAE 增量分别约 **24.55、4.99、4.95**。这说明预测相近时，内部图通道依赖仍可能很不同，解释稳定性尚未解决。

当前适合的下一步是：**在新实验相同的 30 天协议下补跑 B1–B6、M1、M2，重点比较 FixedGraph、M0、M5、M7。**这样才能回答完整结构是否真正优于简单图模型。

**4. 保存的边权没有经过阈值筛选**

是的。`edge_scores.npz` 中：

- `edge_scores`：完整候选图上，每个窗口的原始 sigmoid 门控分数；
- `effective_weight`：完整前向计算中的实际后置传播系数，即  \[ w_{ij}^{t}=s_{ij}^{t}/\max(1,d_j) \]
- 二者均未经过绘图 Top-K／阈值筛选。

我实际检查了新实验 M3 seed 0：`edge_scores` 形状为 **1726×488**，不是每窗只有49条筛选边；`effective_weight` 与上述公式完全一致。

要注意：这里是**全部候选边**，不是全部 \(N(N-1)\) 个节点对。原始分数偶尔为0，也不代表经过了阈值裁剪。

