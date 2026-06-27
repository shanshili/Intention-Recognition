"""
Data layer for DEIG-TCN.

Because no real dataset is provided and the sandbox has no network access, this
module synthesises a traffic-sensor network whose flows contain genuine
*directional, lagged* propagation between nearby sensors (plus daily / weekly
seasonality and noise). That directional structure is what makes candidate
edges and learned intent edges meaningful.

It also implements the preprocessing described in the method:
standardisation, first-difference (delta), periodic residual, and cyclical
time encodings -> per-node feature vector f_i^t in R^7.

If a real CSV is ever available it can be loaded via `load_csv` and fed through
`Preprocessor` in exactly the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

@dataclass
class TrafficData:
    X: np.ndarray            # (N, T) raw flow
    coords: np.ndarray       # (N, 2) lon/lat-like coordinates
    hour: np.ndarray         # (T,) hour of day 0..23
    weekday: np.ndarray      # (T,) day of week 0..6
    true_chains: list        # list of node-index chains used to generate flow


# def generate_synthetic(N=30, days=210, seed=7) -> TrafficData:
#     """Create N sensors on a road-like layout with hourly flow over `days`.
#
#     Flow propagates downstream along several chains with a 1-3 hour lag, so the
#     ground-truth directed structure is known (useful for sanity-checking the
#     recovered intent graph).
#     """
#     rng = np.random.default_rng(seed)
#     T = days * 24
#
#     # --- sensor coordinates: a few corridors laid out in 2D ----------------
#     coords = []
#     n_corridors = 4
#     per = int(np.ceil(N / n_corridors))
#     for c in range(n_corridors):
#         base = rng.uniform(0, 1, size=2)
#         direction = rng.standard_normal(2)
#         direction /= np.linalg.norm(direction) + 1e-9
#         for k in range(per):
#             if len(coords) >= N:
#                 break
#             p = base + direction * 0.16 * k + rng.normal(0, 0.012, size=2)
#             coords.append(p)
#     coords = np.array(coords[:N])
#     # scale to a pseudo lon/lat box
#     coords = coords * np.array([0.35, 0.25]) + np.array([116.30, 39.85])
#
#     # --- build downstream chains following corridor order ------------------
#     chains = []
#     idx = 0
#     for c in range(n_corridors):
#         chain = list(range(idx, min(idx + per, N)))
#         if len(chain) >= 2:
#             chains.append(chain)
#         idx += per
#         if idx >= N:
#             break
#
#     t = np.arange(T)
#     hour = t % 24
#     weekday = (t // 24) % 7
#
#     # seasonal base shared shape: morning + evening peaks, weekend damping
#     daily = (np.exp(-0.5 * ((hour - 8) / 2.2) ** 2)
#              + 0.9 * np.exp(-0.5 * ((hour - 18) / 2.6) ** 2))
#     weekly = np.where(weekday >= 5, 0.6, 1.0)
#     season = daily * weekly
#
#     X = np.zeros((N, T))
#     # exogenous "demand shocks" that travel down each chain
#     for ci, chain in enumerate(chains):
#         head = chain[0]
#         amp = rng.uniform(40, 90)
#         shock = rng.gamma(1.2, 1.0, size=T)
#         shock = np.convolve(shock, np.ones(3) / 3, mode="same")
#         head_flow = amp * (0.6 + season) * (0.8 + 0.5 * shock)
#         head_flow += rng.normal(0, amp * 0.05, size=T)
#         X[head] = np.clip(head_flow, 0, None)
#         prev = head
#         for node in chain[1:]:
#             lag = rng.integers(1, 4)
#             decay = rng.uniform(0.7, 0.95)
#             propagated = np.zeros(T)
#             propagated[lag:] = X[prev][:-lag] * decay
#             local = rng.uniform(15, 35) * (0.5 + season)
#             noise = rng.normal(0, 6, size=T)
#             X[node] = np.clip(0.65 * propagated + 0.55 * local + noise, 0, None)
#             prev = node
#
#     # any sensors not assigned to a chain get independent seasonal flow
#     assigned = {n for ch in chains for n in ch}
#     for n in range(N):
#         if n not in assigned:
#             X[n] = np.clip(rng.uniform(20, 50) * (0.5 + season)
#                            + rng.normal(0, 5, size=T), 0, None)
#
#     # inject a couple of missing / anomalous segments to exercise cleaning
#     X[2, 1000:1010] = np.nan
#     X[5, 2000] = 1e4
#     return TrafficData(X=X, coords=coords, hour=hour, weekday=weekday,
#                        true_chains=chains)


def load_csv(flow_path, coord_path):
    # 1. 读取坐标
    coord_df = pd.read_csv(coord_path)
    coords = coord_df[["x", "y"]].values.astype(np.float32)
    # print( coords.shape[0])

    # 2. 读取流量数据
    flow_df = pd.read_csv(flow_path)
    X = flow_df.values.T.astype(float)
    # print(X.shape[0])

    # 3. 缺失值处理
    col_mean = np.nanmean(X, axis=1, keepdims=True)
    np.copyto(X, col_mean, where=np.isnan(X))

    # 4. 时间特征
    T = X.shape[1]
    timestamps = pd.date_range("2020-01-01", periods=T, freq="h")
    hour = timestamps.hour.to_numpy()
    weekday = timestamps.dayofweek.to_numpy()
    train_end = int(0.6 * T)

    # 5. 初始化 TrafficData
    traffic_data = TrafficData(
        X=X,
        coords=coords,
        hour=hour,
        weekday=weekday,
        true_chains=[]
    )

    # 6. 预处理并重新封装
    processed_dict = Preprocessor(train_end=train_end).fit_transform(traffic_data)
    data = TrafficData(
        X=processed_dict["Z"],
        coords=coords,
        hour=processed_dict["hour"],
        weekday=processed_dict["weekday"],
        true_chains=[]
    )
    return data


# --------------------------------------------------------------------------
class Preprocessor:
    """Cleaning, standardisation and feature construction (train-set stats)."""

    def __init__(self, train_end: int, eps: float = 1e-6):
        self.train_end = train_end
        self.eps = eps

    def fit_transform(self, data: TrafficData):
        X = data.X.copy()
        N, T = X.shape
        hour, weekday = data.hour, data.weekday

        # 1. clean inf
        X[~np.isfinite(X) & ~np.isnan(X)] = np.nan

        # 2/3. fill missing + clip anomalies using TRAIN quantiles only
        for i in range(N):
            xi = X[i]
            train_vals = xi[: self.train_end]
            train_vals = train_vals[np.isfinite(train_vals)]
            q01, q99 = np.quantile(train_vals, [0.01, 0.99])
            # linear interpolation for short gaps / fill remaining
            nans = np.isnan(xi)
            if nans.any():
                good = ~nans
                xi[nans] = np.interp(np.flatnonzero(nans),
                                     np.flatnonzero(good), xi[good])
            xi = np.clip(xi, q01, q99)
            X[i] = xi

        # 4. per-node standardisation with train mean/std
        mu = X[:, : self.train_end].mean(axis=1, keepdims=True)
        sd = X[:, : self.train_end].std(axis=1, keepdims=True) + self.eps
        Z = (X - mu) / sd
        self.mu, self.sd = mu, sd

        # 5. first difference
        dZ = np.zeros_like(Z)
        dZ[:, 1:] = Z[:, 1:] - Z[:, :-1]

        # 6. periodic residual: subtract train mean per (hour, weekday)
        R = np.zeros_like(Z)
        period_mean = {}
        for h in range(24):
            for w in range(7):
                mask = (hour == w * 0 + h) & (weekday == w)  # hour h & weekday w
                train_mask = mask.copy()
                train_mask[self.train_end:] = False
                if train_mask.any():
                    pm = Z[:, train_mask].mean(axis=1)
                else:
                    pm = np.zeros(N)
                period_mean[(h, w)] = pm
                R[:, mask] = Z[:, mask] - pm[:, None]
        self.period_mean = period_mean

        # 7. cyclical time encodings
        hsin = np.sin(2 * np.pi * hour / 24)
        hcos = np.cos(2 * np.pi * hour / 24)
        wsin = np.sin(2 * np.pi * weekday / 7)
        wcos = np.cos(2 * np.pi * weekday / 7)
        time_enc = np.stack([hsin, hcos, wsin, wcos], axis=0)   # (4, T)

        # 8. per-node feature tensor f_i^t in R^7  -> (T, N, 7)
        F0 = 7
        feats = np.zeros((T, N, F0))
        feats[..., 0] = Z.T
        feats[..., 1] = dZ.T
        feats[..., 2] = R.T
        feats[..., 3] = time_enc[0][:, None]
        feats[..., 4] = time_enc[1][:, None]
        feats[..., 5] = time_enc[2][:, None]
        feats[..., 6] = time_enc[3][:, None]

        return {
            "Z": Z, "dZ": dZ, "R": R, "feats": feats,
            "mu": mu, "sd": sd, "hour": hour, "weekday": weekday,
        }
