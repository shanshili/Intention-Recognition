# -*- coding: utf-8 -*-
"""RA-TCGF 项目的全局配置。

RA-TCGF: 残差感知时序因果意图图融合。
项目的所有超参数和路径均在此集中管理。
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


# 部署的 7 个视图规范顺序 (模块一 / M)。
#   C    : 因果视图          A_C^t
#   I    : 意图视图          A_I^t
#   CI   : 因果-意图共识     A_CI^t
#   R    : 意图残差          R_I^t
#   CmI  : 未激活因果        R_C^t (C \ I)
#   P    : 持续意图          A_P^t
#   N    : 新增意图          A_N^t
VIEW_NAMES: List[str] = ["C", "I", "CI", "R", "CmI", "P", "N"]

# La vista de intencion que se desvanece (A_E^t) NO es fuente de aristas de
# despliegue: entra solo como caracteristica de estado del gating.
DECAY_VIEW = "E"


@dataclass
class DataConfig:
    data_dir: str = "../pems_spatial_kmeans_topK"
    # --- 4 个独立的输入文件 (结构各不相同) ---
    flow_csv: str = "id_standardize_Env_A_timeseries.csv"   # 节点原始数据 (时间序列)
    coord_csv: str = "id_node_positions.csv"                # 节点坐标
    causal_pkl: str = "causal_subgraph.pkl"                 # 因果图 {graph, val_matrix, meta}
    intent_npz: str = "intent_graphs.npz"                   # 意图图 {edges_list, strength_list, delta_list}
    # --- 可选的初始节点特征文件 ---
    # 可为 .npy [N, F0] 或 .csv (每行一个节点)。若为 None 或不存在，
    # 则不使用 (节点特征将由流量 + 坐标生成)。
    node_feat_file: str = None
    train_ratio: float = 0.6
    # 若缺少某些真实文件，将生成一致的合成数据。
    allow_synthetic: bool = True
    synth_num_nodes: int = 100
    synth_num_steps: int = 240
    synth_tau_max: int = 3
    synth_n_show: int = 97



@dataclass
class ModelConfig:
    L: int = 12                 # 时间窗口长度
    r: int = 1                  # 意图视图偏移量 (t-r, t, t+r)
    gnn_type: str = "gat"       # "gcn" | "gat" | "sage"
    gnn_layers: int = 2
    gnn_hidden: int = 32        # d  (每个视图的空间嵌入维度)
    tcn_hidden: int = 32        # d_h (节点-时间嵌入维度)
    tcn_layers: int = 2
    tcn_kernel: int = 3
    graph_emb: int = 32         # d_g (图-时间嵌入维度)
    node_extra_feat: bool = True   # 是否拼接归一化坐标
    # 模块一中掩码的阈值 / 温度。
    delta_C: float = 0.30
    delta_I: float = 0.30
    eta: float = 0.10
    dropout: float = 0.1



@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    sparsity_lambda: float = 1e-3   # S_D 上的 L1 正则化
    seed: int = 2025
    device: str = "cuda"            # "cuda" | "cpu" (自动回退到 cpu)
    num_show_steps: int = 97        # 最终导出多少个时间步 t

@dataclass
class DeployConfig:
    budget: float = 20.0        # 预算 B
    node_cost: float = 1.0      # 均匀节点成本 c_i
    edge_cost: float = 0.2      # 均匀边成本 c_ij
    lambda_el: float = 0.5      # 弹性效用权重
    lambda_tr: float = 0.5      # 追踪效用权重
    lambda_c: float = 1.0       # 成本权重
    top_k_nodes: int = 20       # 子图节点上限 (贪心算法)
    node_budget_ratio: float = 0.6  # 分配给节点的预算比例 (剩余分配给边)
    fail_samples: int = 5       # 用于 U_elastic 的故障样本数


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)
    out_root: str = "outputs"

    def flow_path(self) -> str:
        return os.path.join(self.data.data_dir, self.data.flow_csv)

    def coord_path(self) -> str:
        return os.path.join(self.data.data_dir, self.data.coord_csv)

    def causal_path(self) -> str:
        return os.path.join(self.data.data_dir, self.data.causal_pkl)

    def intent_path(self) -> str:
        return os.path.join(self.data.data_dir, self.data.intent_npz)

    def node_feat_path(self):
        if not self.data.node_feat_file:
            return None
        return os.path.join(self.data.data_dir, self.data.node_feat_file)
