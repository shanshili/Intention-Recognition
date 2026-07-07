# -*- coding: utf-8 -*-
"""Global configuration for the RA-TCGF project.

RA-TCGF: Residual-Aware Temporal Causal-intent Graph Fusion.
Todos los hiperparametros y rutas del proyecto se centralizan aqui.
"""
import os
from dataclasses import dataclass, field
from typing import List, Tuple


# Orden CANONICO de las 7 vistas de despliegue (modulo uno / M).
#   C    : vista causal          A_C^t
#   I    : vista intencion       A_I^t
#   CI   : consenso causal-int.  A_CI^t
#   R    : residuo de intencion  R_I^t
#   CmI  : causal no activada    R_C^t (C \ I)
#   P    : intencion persistente A_P^t
#   N    : intencion nueva       A_N^t
VIEW_NAMES: List[str] = ["C", "I", "CI", "R", "CmI", "P", "N"]

# La vista de intencion que se desvanece (A_E^t) NO es fuente de aristas de
# despliegue: entra solo como caracteristica de estado del gating.
DECAY_VIEW = "E"


@dataclass
class DataConfig:
    data_dir: str = "../pems_spatial_kmeans_topK"
    # --- 4 ficheros de entrada INDEPENDIENTES (estructuras distintas) ---
    flow_csv: str = "id_standardize_Env_A_timeseries.csv"   # datos crudos de nodo (serie temporal)
    coord_csv: str = "id_node_positions.csv"                 # coordenadas de nodo (x, y)
    causal_pkl: str = "causal_subgraph.pkl"                  # grafo causal {graph, val_matrix, meta}
    intent_npz: str = "intent_graphs.npz"                    # grafo intencion {edges_list, strength_list, delta_list}
    # --- fichero OPCIONAL de caracteristicas iniciales de nodo ---
    # Puede ser .npy [N, F0] o .csv (una fila por nodo). Si es None o no existe,
    # no se usa (las features de nodo salen del flujo + coordenadas).
    node_feat_file: str = None
    train_ratio: float = 0.6
    # Si falta algun fichero real, se generan datos sinteticos coherentes.
    allow_synthetic: bool = True
    synth_num_nodes: int = 100
    synth_num_steps: int = 240
    synth_tau_max: int = 3
    synth_n_show: int = 97


@dataclass
class ModelConfig:
    L: int = 12                 # longitud de la ventana temporal
    r: int = 1                  # desfase de vistas de intencion (t-r, t, t+r)
    gnn_type: str = "gat"       # "gcn" | "gat" | "sage"
    gnn_layers: int = 2
    gnn_hidden: int = 32        # d  (dim del embedding espacial por vista)
    tcn_hidden: int = 32        # d_h (dim del embedding nodo-temporal)
    tcn_layers: int = 2
    tcn_kernel: int = 3
    graph_emb: int = 32         # d_g (dim del embedding grafo-temporal)
    node_extra_feat: bool = True   # concatenar coords normalizadas
    # Umbrales / temperatura de las mascaras del modulo uno.
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
    sparsity_lambda: float = 1e-3   # regularizacion L1 sobre S_D
    seed: int = 2025
    device: str = "cuda"            # "cuda" | "cpu" (auto-fallback a cpu)
    num_show_steps: int = 97        # cuantos pasos t se exportan al final


@dataclass
class DeployConfig:
    budget: float = 20.0        # presupuesto B
    node_cost: float = 1.0      # c_i homogeneo
    edge_cost: float = 0.2      # c_ij homogeneo
    lambda_el: float = 0.5      # peso utilidad elastica
    lambda_tr: float = 0.5      # peso utilidad de seguimiento
    lambda_c: float = 1.0       # peso de coste
    top_k_nodes: int = 20       # tope de nodos del subgrafo (greedy)
    node_budget_ratio: float = 0.6  # fraccion de B reservada a nodos (resto: aristas)
    fail_samples: int = 5       # muestras de fallo para U_elastic


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
