# -*- coding: utf-8 -*-
"""RA-TCGF 项目的全局配置。"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


VIEW_NAMES: List[str] = ["C", "I", "CI", "R", "CmI", "P", "N"]
DECAY_VIEW = "E"


@dataclass
class DataConfig:
    data_dir: str = "../pems_spatial_kmeans_topK"
    flow_csv: str = "id_standardize_Env_A_timeseries.csv"
    coord_csv: str = "id_node_positions.csv"
    causal_pkl: str = "causal_subgraph.pkl"
    intent_npz: str = "intent_graphs.npz"
    node_feat_file: Optional[str] = None
    train_ratio: float = 0.6

    allow_synthetic: bool = True
    synth_num_nodes: int = 100
    synth_num_steps: int = 240
    synth_tau_max: int = 3
    synth_n_show: int = 97

    # 唯一时间轴：pos=0 严格对应 2025-01-01 00:00:00。
    start_datetime: str = "2025-01-01 00:00:00"
    step_minutes: int = 60

    # A_I_list[0] 对应的原始流量位置。None 表示末尾对齐。
    intent_start_pos: Optional[int] = None


@dataclass
class ModelConfig:
    L: int = 12
    r: int = 1
    gnn_type: str = "gat"
    gnn_layers: int = 2
    gnn_hidden: int = 32
    tcn_hidden: int = 32
    tcn_layers: int = 2
    tcn_kernel: int = 3
    graph_emb: int = 32
    node_extra_feat: bool = True
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
    sparsity_lambda: float = 1e-3
    seed: int = 2025
    device: str = "cuda"


@dataclass
class ViewExportConfig:
    """views_steps（派生视图批量导出）的时间范围与数量。"""

    # 这里就是原来的“97”。<=0 表示导出筛选范围内全部有效时间步。
    num_steps: int = 97

    # 采用闭区间；None 表示不限制。格式示例："2025-01-07 00:00:00"。
    start_time: Optional[str] = None
    end_time: Optional[str] = None

    # 先在时间范围内按此间隔抽样，再按 selection/num_steps 取值。
    step_stride: int = 1
    selection: str = "latest"      # latest | earliest | uniform


@dataclass
class DeployConfig:
    budget: float = 30.0
    node_cost: float = 1.0
    edge_cost: float = 0.2
    lambda_el: float = 0.5
    lambda_tr: float = 0.5
    lambda_c: float = 1.0
    top_k_nodes: int = 30
    node_budget_ratio: float = 0.6
    fail_samples: int = 5
    max_edges: Optional[int] = None

    # 最终部署时间范围与数量。
    num_steps: int = 24
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    step_stride: int = 1
    selection: str = "latest"      # latest | earliest | uniform
    source_split: str = "val"      # train | val | all
    export_fused_graph: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    views: ViewExportConfig = field(default_factory=ViewExportConfig)
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
