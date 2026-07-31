# -*- coding: utf-8 -*-
r"""RA-TCGF 独立输入加载与严格时间对齐。

原始流量序列位置 pos=0 严格对应 cfg.data.start_datetime；默认值为
2025-01-01 00:00:00。所有后续时间均使用同一 timestamps 数组，不再由
局部意图索引 c 单独推算。
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .preprocess import TrafficData, Preprocessor
from ..config import Config


def build_time_axis(cfg: Config, T: int) -> pd.DatetimeIndex:
    """构造唯一时间轴，并验证 pos=0 的精确锚点。"""
    if int(cfg.data.step_minutes) <= 0:
        raise ValueError("data.step_minutes 必须为正整数")
    start = pd.Timestamp(cfg.data.start_datetime)
    expected = pd.Timestamp("2025-01-01 00:00:00")
    if start != expected:
        raise ValueError(
            "本数据集要求 pos=0 严格对应 2025-01-01 00:00:00；"
            f"当前配置为 {start}。")
    return pd.date_range(
        start=start,
        periods=int(T),
        freq=pd.Timedelta(minutes=int(cfg.data.step_minutes)),
    )


# ===========================================================================
# 1. 节点流量 [N, T]
# ===========================================================================
def load_flow(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取流量 CSV，兼容原 API，返回 (X [N,T], hour [T], weekday [T])。"""
    flow_df = pd.read_csv(path)
    X = flow_df.values.T.astype(float)
    col_mean = np.nanmean(X, axis=1, keepdims=True)
    np.copyto(X, col_mean, where=np.isnan(X))
    T = X.shape[1]
    ts = pd.date_range("2025-01-01 00:00:00", periods=T, freq="h")
    return X, ts.hour.to_numpy(), ts.dayofweek.to_numpy()


# ===========================================================================
# 2. 坐标 [N, 2]
# ===========================================================================
def load_coords(path: str) -> np.ndarray:
    coord_df = pd.read_csv(path)
    return coord_df[["x", "y"]].values.astype(np.float32)


# ===========================================================================
# 3. 因果图
# ===========================================================================
def load_causal(path: str) -> Dict:
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    assert isinstance(obj, dict) and "graph" in obj and "val_matrix" in obj
    return obj


def causal_to_adj(causal: Dict) -> np.ndarray:
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
# 4. 意图图，支持可选时间元数据
# ===========================================================================
def _optional_npz_value(data, key):
    if key not in data.files:
        return None
    value = data[key]
    if np.asarray(value).ndim == 0:
        return np.asarray(value).item()
    return value


def load_intent(path: str) -> Dict:
    data = np.load(path, allow_pickle=True)
    result = {
        "edges_list": list(data["edges_list"]),
        "strength_list": list(data["strength_list"]),
        "delta_list": np.asarray(data["delta_list"], dtype=float),
    }
    # 支持三种可选元数据；存在时优先于“末尾 n_show 步”约定。
    for key in ("timestamps", "time_index", "start_pos"):
        value = _optional_npz_value(data, key)
        if value is not None:
            result[key] = value
    return result


def intent_to_adj_list(intent: Dict, N: int) -> List[np.ndarray]:
    adjs = []
    for edges, strength in zip(intent["edges_list"], intent["strength_list"]):
        A = np.zeros((N, N), dtype=float)
        edges = np.asarray(edges)
        strength = np.asarray(strength, dtype=float)
        if edges.size > 0:
            edges = edges.reshape(-1, 2)
            src = edges[:, 0].astype(int)
            dst = edges[:, 1].astype(int)
            valid = (src < N) & (dst < N) & (src >= 0) & (dst >= 0)
            A[src[valid], dst[valid]] = strength[valid]
        m = A.max()
        adjs.append(A / m if m > 1e-9 else A)
    return adjs


def resolve_intent_positions(intent: Dict, timestamps: pd.DatetimeIndex,
                             cfg: Config, n_show: int) -> Tuple[np.ndarray, Dict]:
    """把 A_I_list[c] 映射到原始流量位置 pos，返回位置和对齐说明。"""
    T = len(timestamps)
    method = None

    if "timestamps" in intent:
        intent_ts = pd.to_datetime(np.asarray(intent["timestamps"]).reshape(-1))
        if len(intent_ts) != n_show:
            raise ValueError("intent timestamps 数量与意图图时间步数不一致")
        positions = timestamps.get_indexer(intent_ts)
        if np.any(positions < 0):
            bad = intent_ts[np.where(positions < 0)[0][0]]
            raise ValueError(f"意图时间 {bad} 不在流量时间轴中")
        method = "intent_npz.timestamps"
    elif "time_index" in intent:
        positions = np.asarray(intent["time_index"], dtype=int).reshape(-1)
        method = "intent_npz.time_index"
    else:
        if "start_pos" in intent:
            start_pos = int(intent["start_pos"])
            method = "intent_npz.start_pos"
        elif cfg.data.intent_start_pos is not None:
            start_pos = int(cfg.data.intent_start_pos)
            method = "config.intent_start_pos"
        else:
            start_pos = T - n_show
            method = "explicit_tail_alignment"
        positions = start_pos + np.arange(n_show, dtype=int)

    positions = np.asarray(positions, dtype=int).reshape(-1)
    if positions.size != n_show:
        raise ValueError("意图时间位置数量与 A_I_list 数量不一致")
    if np.any(positions < 0) or np.any(positions >= T):
        raise ValueError(
            f"意图位置超出流量范围 [0,{T - 1}]："
            f"min={positions.min()}, max={positions.max()}")
    if positions.size > 1 and np.any(np.diff(positions) <= 0):
        raise ValueError("意图时间位置必须严格递增且不能重复")

    meta = {
        "method": method,
        "intent_start_pos": int(positions[0]),
        "intent_end_pos": int(positions[-1]),
        "intent_start_time": timestamps[positions[0]].strftime("%Y-%m-%d %H:%M:%S"),
        "intent_end_time": timestamps[positions[-1]].strftime("%Y-%m-%d %H:%M:%S"),
        "n_show": int(n_show),
    }
    return positions, meta


# ===========================================================================
# 5. 可选节点特征
# ===========================================================================
def load_node_features(path: Optional[str]) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    if path.endswith(".npy"):
        F = np.load(path)
    else:
        F = pd.read_csv(path).values
    return np.asarray(F, dtype=np.float32)


def load_csv(flow_path: str, coord_path: str, train_ratio: float = 0.6) -> TrafficData:
    """兼容旧 API；仍严格使用 2025-01-01 00:00:00 的小时轴。"""
    X, _hour, _weekday = load_flow(flow_path)
    coords = load_coords(coord_path)
    T = X.shape[1]
    timestamps = pd.date_range("2025-01-01 00:00:00", periods=T, freq="h")
    train_end = int(train_ratio * T)
    raw = TrafficData(
        X=X, coords=coords,
        hour=timestamps.hour.to_numpy(),
        weekday=timestamps.dayofweek.to_numpy(),
        true_chains=[], timestamps=timestamps.to_numpy(),
    )
    proc = Preprocessor(train_end=train_end).fit_transform(raw)
    return TrafficData(
        X=proc["Z"], coords=coords,
        hour=proc["hour"], weekday=proc["weekday"], true_chains=[],
        timestamps=proc["timestamps"],
    )


# ===========================================================================
# 合成数据
# ===========================================================================
def _synth_flow(cfg: Config):
    rng = np.random.default_rng(cfg.train.seed)
    N, T = cfg.data.synth_num_nodes, cfg.data.synth_num_steps
    t = np.arange(T)
    daily = np.sin(2 * np.pi * t / 24.0)
    base = rng.uniform(0.5, 1.5, size=(N, 1)) * daily[None, :]
    X = 10 + 5 * base + 0.3 * rng.standard_normal((N, T))
    ts = pd.date_range("2025-01-01 00:00:00", periods=T, freq="h")
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


def _load_or_synth(name, exists, real_fn, synth_fn, cfg, used):
    if exists:
        return real_fn()
    if cfg.data.allow_synthetic:
        used[name] = True
        return synth_fn()
    raise FileNotFoundError(f"缺少输入文件: {name}")


def load_all(cfg: Config) -> Dict:
    used = {"flow": False, "coords": False, "causal": False,
            "intent": False, "node_feat": False}

    flow_pack = _load_or_synth(
        "flow", os.path.exists(cfg.flow_path()),
        lambda: load_flow(cfg.flow_path()), lambda: _synth_flow(cfg), cfg, used)
    X, _legacy_hour, _legacy_weekday = flow_pack
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
    A_C_full = causal_to_adj(causal)

    n_candidates = [X.shape[0], coords.shape[0], A_C_full.shape[0]]
    if node_feat is not None:
        n_candidates.append(node_feat.shape[0])
    N = int(min(n_candidates))
    dims = {"flow": X.shape[0], "coords": coords.shape[0],
            "causal": A_C_full.shape[0]}
    if node_feat is not None:
        dims["node_feat"] = node_feat.shape[0]
    if len(set(dims.values())) > 1:
        print(f"[warn] N 不一致 {dims}；统一裁剪到 N={N}")

    X = X[:N]
    coords = coords[:N]
    A_C = A_C_full[:N, :N]
    if node_feat is not None:
        node_feat = node_feat[:N]

    timestamps = build_time_axis(cfg, X.shape[1])
    hour = timestamps.hour.to_numpy()
    weekday = timestamps.dayofweek.to_numpy()

    train_end = int(cfg.data.train_ratio * X.shape[1])
    raw = TrafficData(
        X=X, coords=coords, hour=hour, weekday=weekday,
        true_chains=[], timestamps=timestamps.to_numpy(),
    )
    proc = Preprocessor(train_end=train_end).fit_transform(raw)
    data = TrafficData(
        X=proc["Z"], coords=coords,
        hour=proc["hour"], weekday=proc["weekday"], true_chains=[],
        timestamps=proc["timestamps"],
    )

    A_I_list = intent_to_adj_list(intent, N)
    intent_positions, intent_alignment = resolve_intent_positions(
        intent, timestamps, cfg, len(A_I_list))

    return {
        "data": data,
        "timestamps": timestamps.to_numpy(),
        "A_C": A_C,
        "A_I_list": A_I_list,
        "intent_positions": intent_positions,
        "intent_alignment": intent_alignment,
        "node_feat": node_feat,
        "causal_meta": causal.get("meta", {}),
        "delta_list": intent["delta_list"],
        "N": N,
        "node_dims": dims,
        "used_synthetic": used,
    }
