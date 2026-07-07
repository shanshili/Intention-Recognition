# RA-TCGF：残差感知时序因果-意图图融合

**Residual-Aware Temporal Causal-intent Graph Fusion**，用于弹性传感网动态部署优化。

整体流程为：

\[
(G_C,\,G_I^{t-r},\,G_I^t,\,G_I^{t+r},\,X_{t-L+1:t})
\rightarrow
\mathcal{A}^{t-r:t+r}
\rightarrow
\{H_m^t,\mathbf{g}_m^t\}_{m\in\mathcal{M}}
\rightarrow
S_D^t
\rightarrow
G_D^t
\]

其中每个视图 \(m\in\mathcal{M}=\{C,I,CI,R,C\setminus I,P,N\}\)。

---

## 1. 目录结构

```
ra_tcgf/
├── requirements.txt
├── run_example.sh
└── ratcgf/
    ├── config.py                # 全部超参与路径
    ├── main.py                  # 入口：数据→训练→推理→部署→可视化
    ├── dataset.py               # 展示步 c → 时间窗 pos 的样本构造
    ├── model.py                 # RATCGF 主模型（串联模块 1-4）
    ├── train.py                 # 训练循环
    ├── visualize.py             # 所有可视化（多格式保存）
    ├── utils/
    │   ├── io_utils.py          # 时间戳目录 + SVG/PNG/PDF/EPS 保存
    │   ├── preprocess.py        # TrafficData + Preprocessor（z-score）
    │   └── data_loading.py      # 读 pkl / npz / csv（缺失时合成）
    └── modules/
        ├── module1_residual.py  # 残差多视图构造 + q_struct + b_ij
        ├── module2_encoder.py   # GNN(GCN/GAT/SAGE) + 节点/图级 TCN
        ├── module3_gating.py    # 图/节点/边级门控 → S_D^t
        └── module4_deploy.py    # 部署子图 G_D^t（贪心 + 预算约束）
```

## 2. 输入文件（四/五个**独立**文件，结构各不相同）

每个文件有独立的 loader（见 `utils/data_loading.py`），共享同一套节点索引 `0..N-1`。
加载时会自动把各文件的 \(N\) 对齐到最小公共值并给出警告（`load_all` 中）。

| 文件（默认名） | 独立 loader | 数据结构 |
| :--- | :--- | :--- |
| `id_standardize_Env_A_timeseries.csv` | `load_flow` | 节点原始数据（时间序列），行=时间列=节点 → `[N,T]` |
| `id_node_positions.csv` | `load_coords` | 节点坐标，列 `x`,`y` → `[N,2]` |
| `causal_subgraph.pkl` | `load_causal` | 因果图 dict：`graph[N,N,tau+1]`、`val_matrix`、`meta` |
| `intent_graphs.npz` | `load_intent` | 意图图：`edges_list`、`strength_list`、`delta_list`（97 步） |
| *(可选)* `node_features.npy/.csv` | `load_node_features` | 节点初始特征 → `[N,F0]`，若提供则拼进节点特征 |

> 每个文件都可用命令行单独指定路径；若某文件缺失且 `allow_synthetic=True`，仅该项会被合成，其余仍读真实文件。

命令行分别指定各独立文件：

```bash
python -m ratcgf.main \
    --data-dir ../pems_spatial_kmeans_topK \
    --flow-csv id_standardize_Env_A_timeseries.csv \
    --coord-csv id_node_positions.csv \
    --causal-pkl causal_subgraph.pkl \
    --intent-npz intent_graphs.npz \
    --node-feat node_features.npy \
    --epochs 40 --show-steps 97
```

## 3. 运行

```bash
pip install -r requirements.txt
python -m ratcgf.main --data-dir ../pems_spatial_kmeans_topK --epochs 40 --show-steps 97
```

## 4. 输出（全部带时间戳，位于 `outputs/run_<时间戳>/`）

| 路径 | 内容 |
| :--- | :--- |
| `models/ratcgf_model_<ts>.pt` | **最终训练好的模型**（权重 + 配置 + 训练历史） |
| `figures/fused_SD_<ts>.{svg,png,pdf,eps}` | **融合后的可视化图**（\(S_D^t\) 热图 + 网络图） |
| `figures/loss_curve_<ts>.{...}` | **损失函数曲线** |
| `figures/deployment_subgraph_<ts>.{...}` | \(S_D^t\) 生成的**部署子图** \(G_D^t\) |
| `arrays/SD_and_deploy_<ts>.npz` | \(S_D^t\) 矩阵、节点分数、部署节点/边 |
| `views_97steps/` | **97 个时间步的 6 类视图**（每类一个子文件夹） |
| `logs/run_summary_<ts>.json` | 运行摘要与指标 |

`views_97steps/` 下 6 个子文件夹分别对应：

- `consensus/` —— 因果-意图共识视图 \(A_{CI}^t=\tilde A_C\odot\tilde A_I^t\)
- `intent_residual/` —— 意图残差视图 \(R_I^t=(1-M_C)\odot\tilde A_I^t\)
- `causal_inactive/` —— 因果未激活视图 \(R_C^t=\tilde A_C\odot(1-M_I^t)\)
- `intent_persist/` —— 意图持续图 \(A_P^t=\min(\tilde A_I^t,\tilde A_I^{t-r})\)
- `intent_new/` —— 意图新增图 \(A_N^t=[\tilde A_I^t-\tilde A_I^{t-r}]_+\)
- `intent_decay/` —— 意图消退图 \(A_E^t=[\tilde A_I^{t-r}-\tilde A_I^t]_+\)

每张图均同时保存为 `svg / png / pdf / eps` 四种格式。

## 5. 模块与公式对应

- **模块一**（`module1_residual.py`）：掩码
  \(M_C=\sigma((\tilde A_C-\delta_C)/\eta)\)，
  \(M_I^\tau=\sigma((\tilde A_I^\tau-\delta_I)/\eta)\)，
  构造 7 视图 + \(A_E^t\)，并输出 \(\mathbf q_{struct}^t\)（9 维）与 \(\mathbf b_{ij}^t\)（8 维）。
- **模块二**（`module2_encoder.py`）：
  \(Z_m^\tau=\mathrm{GNN}_m(X_\tau,A_m^\tau)\)，
  \(H_m^t=\mathrm{TCN}_{node,m}([Z_m^\tau]_\tau)\)，
  \(\mathbf g_m^t=\mathrm{TCN}_{graph,m}([\mathrm{READOUT}(Z_m^\tau)]_\tau)\)。
- **模块三**（`module3_gating.py`）：
  \(\boldsymbol\pi^t=\mathrm{softmax}(\mathrm{MLP}_g([\mathbf q_{struct}^t\Vert\mathbf g_{all}^t]))\)，
  \(\boldsymbol\gamma_i^t\)、\(\boldsymbol\omega_{ij}^t\)，
  \(\alpha_{ij,m}^t=\dfrac{\pi_m^t\omega_{ij,m}^t}{\sum_n\pi_n^t\omega_{ij,n}^t+\varepsilon}\)，
  \(S_D^t[i,j]=\sum_m\alpha_{ij,m}^tA_m^t[i,j]\)。
- **模块四**（`module4_deploy.py`）：
  \(Score_i^t=\sum_j S_D^t[i,j]+\sum_j S_D^t[j,i]+\mathrm{MLP}_{score}(\mathbf h_{i,D}^t)\)，
  在预算 \(\sum_i c_ip_i^t+\sum_{ij}c_{ij}q_{ij}^t\le B\) 下贪心生成 \(G_D^t\)。

## 6. 训练目标

采用自监督预测效用 \(U_{pred}=-\ell(\widehat Y_t^{G(p,q)},Y_t)\)：
以融合节点表示 \(\mathbf h_{i,D}^t\) 预测下一时刻流量，损失为 MSE 加 \(S_D^t\) 的 L1 稀疏正则；
弹性效用 \(U_{elastic}\) 与追踪效用 \(U_{track}\) 在部署阶段计算并写入日志。
