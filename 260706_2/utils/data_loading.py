# -*- coding: utf-8 -*-
r"""Carga de las ENTRADAS INDEPENDIENTES del proyecto.

Cada entrada es un fichero separado con su propia estructura y su propio
loader dedicado. Todas comparten el mismo indice de nodo 0..N-1:

  1. Datos crudos de nodo (serie temporal)  -> load_flow      (.csv)   [N, T]
  2. Coordenadas de nodo                     -> load_coords    (.csv)   [N, 2]
  3. Grafo causal                            -> load_causal    (.pkl)   {graph,val_matrix,meta}
  4. Grafo de intencion                      -> load_intent    (.npz)   {edges,strength,delta}
  5. (opcional) Caracteristicas iniciales    -> load_node_features (.npy/.csv) [N, F0]

`load_all` compone los cinco, alinea N al minimo comun y valida coherencia.
Si algun fichero falta y allow_synthetic=True, se sintetiza esa entrada.
"""
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .preprocess import TrafficData, Preprocessor
from ..config import Config


# ===========================================================================
# 1. Datos crudos de nodo (serie temporal)  ->  [N, T]
# ===========================================================================
def load_flow(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lee la serie temporal cruda. Devuelve (X [N,T], hour [T], weekday [T])."""
    flow_df = pd.read_csv(path)
    X = flow_df.values.T.astype(float)                 # filas=nodo, col=tiempo
    col_mean = np.nanmean(X, axis=1, keepdims=True)    # relleno de faltantes
    np.copyto(X, col_mean, where=np.isnan(X))
    T = X.shape[1]
    ts = pd.date_range("2025-01-01", periods=T, freq="h")
    return X, ts.hour.to_numpy(), ts.dayofweek.to_numpy()


# ===========================================================================
# 2. Coordenadas de nodo  ->  [N, 2]
# ===========================================================================
def load_coords(path: str) -> np.ndarray:
    coord_df = pd.read_csv(path)
    return coord_df[["x", "y"]].values.astype(np.float32)


# ===========================================================================
# 3. Grafo causal (.pkl)  ->  dict {graph, val_matrix, meta}
# ===========================================================================
def load_causal(path: str) -> Dict:
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    assert isinstance(obj, dict) and "graph" in obj and "val_matrix" in obj
    return obj


def causal_to_adj(causal: Dict) -> np.ndarray:
    r"""Colapsa [N,N,tau+1] -> adyacencia ponderada [N,N] normalizada a [0,1].

    A_C[i,j] = max_tau |val_matrix[i,j,tau]| sobre aristas graph[i,j,tau]=='-->'.
    """
    graph = np.asarray(causal["graph"])
    val = np.asarray(causal["val_matrix"], dtype=float)
    if graph.ndim == 2:
        graph = graph[:, :, None]
        val = val[:, :, None]
    A = np.where(graph == "-->", np.abs(val), 0.0).max(axis=2)
    np.fill_diagonal(A, 0.0)
    m = A.max()
    return A / m if m > 1e-9 else A


# ===========================================================================
# 4. Grafo de intencion (.npz)  ->  dict {edges_list, strength_list, delta_list}
# ===========================================================================
def load_intent(path: str) -> Dict:
    data = np.load(path, allow_pickle=True)
    return {
        "edges_list": list(data["edges_list"]),
        "strength_list": list(data["strength_list"]),
        "delta_list": np.asarray(data["delta_list"], dtype=float),
    }


def intent_to_adj_list(intent: Dict, N: int) -> List[np.ndarray]:
    """Reconstruye A_I^t [N,N] (normalizado a [0,1]) para cada paso mostrado."""
    adjs = []
    for edges, strength in zip(intent["edges_list"], intent["strength_list"]):
        A = np.zeros((N, N), dtype=float)
        edges = np.asarray(edges)
        strength = np.asarray(strength, dtype=float)
        if edges.size > 0:
            src = edges[:, 0].astype(int)
            dst = edges[:, 1].astype(int)
            valid = (src < N) & (dst < N) & (src >= 0) & (dst >= 0)
            A[src[valid], dst[valid]] = strength[valid]
        m = A.max()
        adjs.append(A / m if m > 1e-9 else A)
    return adjs


# ===========================================================================
# 5. (opcional) Caracteristicas iniciales de nodo  ->  [N, F0]
# ===========================================================================
def load_node_features(path: Optional[str]) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    if path.endswith(".npy"):
        F = np.load(path)
    else:                                              # csv
        F = pd.read_csv(path).values
    return np.asarray(F, dtype=np.float32)


# ===========================================================================
# Utilidad heredada: leer flujo+coords y devolver TrafficData preprocesado
# ===========================================================================
def load_csv(flow_path: str, coord_path: str, train_ratio: float = 0.6) -> TrafficData:
    X, hour, weekday = load_flow(flow_path)
    coords = load_coords(coord_path)
    train_end = int(train_ratio * X.shape[1])
    raw = TrafficData(X=X, coords=coords, hour=hour, weekday=weekday, true_chains=[])
    proc = Preprocessor(train_end=train_end).fit_transform(raw)
    return TrafficData(X=proc["Z"], coords=coords,
                       hour=proc["hour"], weekday=proc["weekday"], true_chains=[])


# ===========================================================================
# Generadores sinteticos (uno por entrada independiente)
# ===========================================================================
def _synth_flow(cfg: Config):
    rng = np.random.default_rng(cfg.train.seed)
    N, T = cfg.data.synth_num_nodes, cfg.data.synth_num_steps
    t = np.arange(T)
    daily = np.sin(2 * np.pi * t / 24.0)
    base = rng.uniform(0.5, 1.5, size=(N, 1)) * daily[None, :]
    X = 10 + 5 * base + 0.3 * rng.standard_normal((N, T))
    ts = pd.date_range("2025-01-01", periods=T, freq="h")
    return X, ts.hour.to_numpy(), ts.dayofweek.to_numpy()


def _synth_coords(cfg: Config):
    rng = np.random.default_rng(cfg.train.seed + 3)
    return rng.uniform(0, 100, size=(cfg.data.synth_num_nodes, 2)).astype(np.float32)


def _synth_causal(cfg: Config) -> Dict:
    rng = np.random.default_rng(cfg.train.seed + 1)
    N, tau = cfg.data.synth_num_nodes, cfg.data.synth_tau_max
    graph = np.full((N, N, tau + 1), "", dtype=object)
    val = np.zeros((N, N, tau + 1), dtype=float)
    for lag in range(tau + 1):
        m = rng.random((N, N)) < 0.03
        np.fill_diagonal(m, False)
        graph[:, :, lag][m] = "-->"
        val[:, :, lag][m] = rng.uniform(0.2, 0.9, size=m.sum())
    meta = {"mask_threshold": 0.5, "top_ratio": 0.30,
            "node_score_mode": "inout", "edge_keep_mode": "or"}
    return {"graph": graph, "val_matrix": val, "meta": meta}


def _synth_intent(cfg: Config) -> Dict:
    rng = np.random.default_rng(cfg.train.seed + 2)
    N, n_show = cfg.data.synth_num_nodes, cfg.data.synth_n_show
    edges_list, strength_list, delta_list = [], [], []
    for _ in range(n_show):
        ne = rng.integers(20, 60)
        src = rng.integers(0, N, size=ne)
        dst = rng.integers(0, N, size=ne)
        keep = src != dst
        edges_list.append(np.stack([src[keep], dst[keep]], axis=1).astype(int))
        s = rng.uniform(0.1, 1.0, size=keep.sum())
        strength_list.append(s.astype(float))
        delta_list.append(float(np.quantile(s, 0.90)) if s.size else 0.0)
    return {"edges_list": edges_list, "strength_list": strength_list,
            "delta_list": np.asarray(delta_list, dtype=float)}


# ===========================================================================
# API principal: carga independiente + alineacion + validacion
# ===========================================================================
def _load_or_synth(name, exists, real_fn, synth_fn, cfg, used):
    if exists:
        return real_fn()
    if cfg.data.allow_synthetic:
        used[name] = True
        return synth_fn()
    raise FileNotFoundError(f"Falta el fichero de entrada: {name}")


def load_all(cfg: Config) -> Dict:
    used = {"flow": False, "coords": False, "causal": False,
            "intent": False, "node_feat": False}

    # --- cada entrada se carga de forma INDEPENDIENTE ---
    X, hour, weekday = _load_or_synth(
        "flow", os.path.exists(cfg.flow_path()),
        lambda: load_flow(cfg.flow_path()), lambda: _synth_flow(cfg), cfg, used)

    coords = _load_or_synth(
        "coords", os.path.exists(cfg.coord_path()),
        lambda: load_coords(cfg.coord_path()), lambda: _synth_coords(cfg), cfg, used)

    causal = _load_or_synth(
        "causal", os.path.exists(cfg.causal_path()),
        lambda: load_causal(cfg.causal_path()), lambda: _synth_causal(cfg), cfg, used)

    intent = _load_or_synth(
        "intent", os.path.exists(cfg.intent_path()),
        lambda: load_intent(cfg.intent_path()), lambda: _synth_intent(cfg), cfg, used)

    node_feat = load_node_features(cfg.node_feat_path())
    if node_feat is not None:
        used["node_feat"] = False  # cargado real

    A_C_full = causal_to_adj(causal)

    # --- alineacion de N al minimo comun entre las 4/5 fuentes ---
    n_candidates = [X.shape[0], coords.shape[0], A_C_full.shape[0]]
    if node_feat is not None:
        n_candidates.append(node_feat.shape[0])
    N = int(min(n_candidates))

    dims = {"flow": X.shape[0], "coords": coords.shape[0],
            "causal": A_C_full.shape[0]}
    if node_feat is not None:
        dims["node_feat"] = node_feat.shape[0]
    if len(set(dims.values())) > 1:
        print(f"[warn] N difiere entre ficheros {dims}; se recorta a N={N}")

    X = X[:N]
    coords = coords[:N]
    A_C = A_C_full[:N, :N]
    if node_feat is not None:
        node_feat = node_feat[:N]

    # --- preprocesado (z-score con estadisticas de train) ---
    train_end = int(cfg.data.train_ratio * X.shape[1])
    raw = TrafficData(X=X, coords=coords, hour=hour, weekday=weekday, true_chains=[])
    proc = Preprocessor(train_end=train_end).fit_transform(raw)
    data = TrafficData(X=proc["Z"], coords=coords,
                       hour=proc["hour"], weekday=proc["weekday"], true_chains=[])

    # --- grafos de intencion reconstruidos con el N alineado ---
    A_I_list = intent_to_adj_list(intent, N)

    return {
        "data": data,
        "A_C": A_C,
        "A_I_list": A_I_list,
        "node_feat": node_feat,             # [N, F0] o None
        "causal_meta": causal.get("meta", {}),
        "delta_list": intent["delta_list"],
        "N": N,
        "node_dims": dims,
        "used_synthetic": used,
    }
