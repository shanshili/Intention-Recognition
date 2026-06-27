# DEIG-TCN — Dynamic Edge-level Intent Graph Temporal Convolutional Network

预测驱动的动态边级交通意图图构建算法。意图被建模为 **当前窗口中对未来交通状态有预测贡献的有向边强度** \(s_{ij}^{t,H}\in[0,1]\)，而非离散类别。

## 运行

```bash
pip install numpy scipy matplotlib networkx scikit-learn   # 本环境已预装
python main.py --out ./outputs            # 完整运行 (N=30, 210天)
python main.py --quick --out ./outputs    # 快速测试 (N=20, 90天)
python main.py --epochs 60                 # 自定义训练轮数
```

无需 GPU / PyTorch。本实现用 NumPy 自带的最小自动微分引擎训练可学习部分。

## 输出文件（命名均带运行时间戳，图片同时存 png / svg / pdf）

| 文件 | 说明 |
|------|------|
| `candidate_directed_graph_<ts>.{png,svg,pdf}` | **候选有向图 G0**（要求项） |
| `intent_graphs_10windows_<ts>.{png,svg,pdf}` | **10 个连续时间窗口的最终意图图**（要求项，有向边 + 深浅表示强度） |
| `intent_graph_window_00..09_<ts>.{png,svg,pdf}` | 每个窗口单独的高清意图图 |
| `training_loss_<ts>.{png,svg,pdf}` | 训练损失曲线 |
| `edge_mask_ablation_<ts>.{png,svg,pdf}` | 高意图边遮蔽实验 |
| `intent_graph_density_<ts>.{png,svg,pdf}` | 各窗口意图图稀疏度 |
| `edge_intent_scores_<ts>.npz` | 边、坐标、各窗口意图分数等数据导出 |
| `run_summary_<ts>.json` | 指标汇总（MAE、消融、密度等） |

## 代码结构

```
main.py                 端到端流程编排
deig/
  autograd.py           最小反向模式自动微分引擎 + Adam（已做梯度数值校验）
  data.py               合成交通数据 + 预处理（标准化/差分/周期残差/时间编码）
  graph.py              候选有向图 G0（双向 KNN + 相关性 + 时滞相关性 + 边特征）
  features.py           时空窗口特征（GAT 角色的空间注意力聚合 + TCN 角色的多尺度时间摘要）+ 动态边特征 d_ij^t
  model.py              可训练：边级意图头 + 意图加权消息传递 + 预测器 + 损失 + 批处理
  intent_graph.py       动态阈值建图 G_I^t + 弱连通分量热点子图
  viz.py                带时间戳的 png/svg/pdf 保存、候选图、意图图面板
```

## 与论文规范的对应与取舍

完整实现的部分：
- 数据预处理（按时间顺序划分、训练集统计量、缺失/异常处理、变化率、周期残差、时间编码）；
- 候选有向图构建（双向 KNN、Pearson、时滞相关性 \(\rho_{ij}^{lag}\)、最优时滞、边静态特征 \(e_{ij}\)）；
- 显式动态边特征 \(d_{ij}^t\)（方向传播 \(D_{ij}^t\)、异常强度 \(A_i\)、近期趋势 \(G_i\)）；
- 边级意图头 \(s_{ij}^{t,H}=\sigma(\mathrm{MLP}(m_{ij}^t))\)；
- **预测自监督**：意图强度作为动态邻接权重做消息传递并预测全图未来 \(\hat Y_t\in\mathbb R^{N\times H}\)，预测损失反传到意图头；
- 第一版推荐损失 \(L=L_{pred}+\lambda_s L_{sparse}+\lambda_d L_{dir}\)；
- 分位数阈值动态建图 \(G_I^t\)、弱连通分量热点提取、节点入/出/汇/源意图属性；
- 高意图边遮蔽验证实验（top / random / low 对比）。

为在无 PyTorch 的纯 NumPy 环境内可训练、可收敛而做的简化（已在代码注释标明）：
- GAT 空间编码器与 TCN 时间编码器以 **固定特征映射** 实现其作用（一跳基于边特征的注意力空间聚合 + 多尺度时间摘要），可训练参数集中在意图头与预测器；这正是意图信号所在、并由预测损失训练。
- 时间平滑 \(L_{temp}\)、边遮蔽蒸馏 \(L_{edge}\)、节点弱监督 \(L_{node}\)、软重叠子图与跨窗口轨迹为规范中的后续增强项，接口已预留，第一版默认不启用以保持最小闭环。

要接入真实数据：用 `data.load_csv(flow_csv, coord_csv)` 替换 `generate_synthetic`，其余流程不变。
