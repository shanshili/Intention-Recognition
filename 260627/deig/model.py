"""
DEIG-TCN trainable model (Sections 7-9, 12-13).

Trainable components (the rest of the pipeline is fixed feature engineering):
  * EdgeIntentHead : MLP(m_ij^t) -> sigmoid -> s_ij in [0,1]
  * W_m            : message projection for intent-weighted propagation
  * Predictor      : MLP([z_j || g_j]) -> H-step residual forecast

The edge intent scores s_ij are used as dynamic adjacency weights in the
predictor's message passing, so the prediction loss back-propagates into the
edge head -- giving the intent scores a self-supervised training signal.

First-version objective (recommended minimal closed loop):
    L = L_pred + lambda_s * L_sparse + lambda_d * L_dir
"""

from __future__ import annotations

import numpy as np

from .autograd import Tensor, Adam, concat, huber_loss, parameter, _safe_div


def segment_sum_normalize(s: Tensor, seg_idx: np.ndarray, num_seg: int,
                          eps: float = 1e-9) -> Tensor:
    """s_tilde_e = s_e / (sum_{e' in seg(e)} s_e' + eps)  (Sec 8)."""
    denom = s.scatter_add(seg_idx, num_seg) + eps
    return _safe_div(s, denom.gather_rows(seg_idx))


class DEIG_TCN:
    def __init__(self, Fz, Fedge, H, hidden=32, msg_dim=16, seed=0):
        self.H = H
        Fm = 4 * Fz + Fedge          # [z_i,z_j,z_i-z_j,z_i*z_j, edge_block]
        # edge intent head
        self.W1 = parameter((Fm, hidden), seed=seed)
        self.b1 = Tensor(np.zeros(hidden), requires_grad=True)
        self.W2 = parameter((hidden, 1), seed=seed + 1)
        self.b2 = Tensor(np.zeros(1), requires_grad=True)
        # message projection
        self.Wm = parameter((Fz, msg_dim), seed=seed + 2)
        # predictor
        Fp = Fz + msg_dim
        self.Wp1 = parameter((Fp, hidden), seed=seed + 3)
        self.bp1 = Tensor(np.zeros(hidden), requires_grad=True)
        self.Wp2 = parameter((hidden, H), seed=seed + 4)
        self.bp2 = Tensor(np.zeros(H), requires_grad=True)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.Wm,
                self.Wp1, self.bp1, self.Wp2, self.bp2]

    # ---- forward pieces -------------------------------------------------
    def edge_scores(self, Z: Tensor, src, dst, edge_block: Tensor) -> Tensor:
        zi = Z.gather_rows(src)
        zj = Z.gather_rows(dst)
        m = concat([zi, zj, zi - zj, zi * zj, edge_block], axis=1)
        h = (m @ self.W1 + self.b1).relu()
        u = h @ self.W2 + self.b2          # (E,1)
        return u.sigmoid().reshape(-1)     # (E,)

    def predict(self, Z: Tensor, s: Tensor, src, dst, num_nodes,
                z_last: np.ndarray) -> Tensor:
        s_tilde = segment_sum_normalize(s, dst, num_nodes)     # (E,)
        msg = (Z @ self.Wm).gather_rows(src)                   # (E, msg_dim)
        weighted = msg * s_tilde.reshape(-1, 1)                # (E, msg_dim)
        g = weighted.scatter_add(dst, num_nodes)               # (num_nodes, msg)
        zbar = concat([Z, g], axis=1)
        hp = (zbar @ self.Wp1 + self.bp1).relu()
        delta = hp @ self.Wp2 + self.bp2                       # (num_nodes, H)
        return Tensor(z_last.reshape(-1, 1)) + delta           # residual form

    # ---- full forward + loss -------------------------------------------
    def forward_loss(self, batch, lambdas):
        Z = Tensor(batch["Z"])
        edge_block = Tensor(batch["edge_block"])
        src, dst = batch["src"], batch["dst"]
        num_nodes = batch["num_nodes"]

        s = self.edge_scores(Z, src, dst, edge_block)
        pred = self.predict(Z, s, src, dst, num_nodes, batch["z_last"])

        L_pred = huber_loss(pred, batch["target"], kappa=1.0)
        L_sparse = s.mean()
        # direction antisymmetry on reciprocal pairs
        pa, pb = batch["pair_a"], batch["pair_b"]
        if len(pa) > 0:
            L_dir = (s.gather_rows(pa) * s.gather_rows(pb)).mean()
        else:
            L_dir = Tensor(0.0)
        total = (L_pred
                 + lambdas["sparse"] * L_sparse
                 + lambdas["dir"] * L_dir)
        return total, {"pred": L_pred.data, "sparse": L_sparse.data,
                       "dir": L_dir.data, "total": total.data}, s, pred

    # ---- inference: edge scores only (no grad needed) ------------------
    def infer_scores(self, batch):
        Z = Tensor(batch["Z"])
        edge_block = Tensor(batch["edge_block"])
        s = self.edge_scores(Z, batch["src"], batch["dst"], edge_block)
        return s.data

    def infer_pred(self, batch, s_override=None):
        Z = Tensor(batch["Z"])
        if s_override is None:
            s = self.edge_scores(Z, batch["src"], batch["dst"],
                                 Tensor(batch["edge_block"]))
        else:
            s = Tensor(s_override)
        pred = self.predict(Z, s, batch["src"], batch["dst"],
                            batch["num_nodes"], batch["z_last"])
        return pred.data


# --------------------------------------------------------------------------
def assemble_batch(window_inputs, targets, cand, reverse_index):
    """Stack a list of per-window inputs into one concatenated graph batch.

    window_inputs : list of dicts from features.build_window_inputs
    targets       : list of (N, H) arrays
    reverse_index : array mapping edge e -> reverse edge index (or -1)
    """
    B = len(window_inputs)
    N = window_inputs[0]["z"].shape[0]
    E = cand["E"]
    src0, dst0 = cand["src"], cand["dst"]

    Z = np.concatenate([w["z"] for w in window_inputs], axis=0)        # (B*N,Fz)
    edge_block = np.concatenate([w["edge_block"] for w in window_inputs], axis=0)
    z_last = np.concatenate([w["z_last"] for w in window_inputs], axis=0)
    target = np.concatenate(targets, axis=0)                            # (B*N,H)

    src = np.concatenate([src0 + b * N for b in range(B)])
    dst = np.concatenate([dst0 + b * N for b in range(B)])

    # reciprocal pairs (i<j) with batch offsets
    pa, pb = [], []
    seen = set()
    for e in range(E):
        r = reverse_index[e]
        if r >= 0 and e < r and (e, r) not in seen:
            seen.add((e, r))
            for b in range(B):
                pa.append(e + b * E)
                pb.append(r + b * E)
    return {
        "Z": Z, "edge_block": edge_block, "z_last": z_last, "target": target,
        "src": src, "dst": dst, "num_nodes": B * N,
        "pair_a": np.array(pa, dtype=np.int64),
        "pair_b": np.array(pb, dtype=np.int64),
        "B": B, "N": N, "E": E,
    }


def compute_reverse_index(cand):
    """For each edge (i,j) find index of (j,i) in the edge list, else -1."""
    edges = cand["edges"]
    lookup = {(int(i), int(j)): e for e, (i, j) in enumerate(edges)}
    rev = np.full(cand["E"], -1, dtype=np.int64)
    for e, (i, j) in enumerate(edges):
        rev[e] = lookup.get((int(j), int(i)), -1)
    return rev
