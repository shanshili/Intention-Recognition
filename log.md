## 260603

### cpu-20260602_121306

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

### gpu-