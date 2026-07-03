"""
Trainable GAT-TCN spatio-temporal encoder  (Section 6 of the method spec).

This replaces the previous *fixed* feature map in features.py. Every operation
here goes through the NumPy autograd engine, so the prediction loss
back-propagates through BOTH the GAT spatial attention and the TCN temporal
convolutions -- i.e. GAT-TCN is now genuinely learned end-to-end, not a frozen
hand-crafted feature transform.

Pipeline for one batched window (B windows stacked as B*N graph nodes):

    F_seq : list of W tensors, each (B*N, F0)      # raw node features f_i^tau
        |  per time step tau:  L_g GAT layers on the candidate graph G0
        v
    H_seq : list of W tensors, each (B*N, F_g)     # H^{tau,G}   (Sec 6.1)
        |  per node:  dilated causal TCN over the W time steps
        v
    Q_seq : list of W tensors, each (B*N, F_t)     # Q_i^t       (Sec 6.2)
        |  temporal attention pooling (softmax over tau)
        v
    Z     : (B*N, F_t)                             # z_i^t window representation

The GAT attention alpha_ij here is the *spatial encoding* attention; it is NOT
the edge intent score (produced later by EdgeIntentHead in model.py), exactly
as the spec notes.
"""

from __future__ import annotations

import numpy as np

from .autograd import (Tensor, parameter, concat, segment_softmax,
                       softmax_lastdim)


# --------------------------------------------------------------------------
class GATLayer:
    """Single-head graph attention layer over the directed candidate graph.

        z_i      = W h_i
        e_ij     = LeakyReLU( a^T [ z_i || z_j || U e_ij^static ] )
        alpha_ij = softmax_{i : (i,j) in E0} e_ij        (over incoming edges)
        h_j'     = relu( sum_i alpha_ij z_i  +  W_res h_j )   (residual)
    """

    def __init__(self, f_in, f_out, f_edge, seed=0):
        self.W = parameter((f_in, f_out), seed=seed)            # node linear
        self.U = parameter((f_edge, f_out), seed=seed + 1)      # edge linear
        self.a = parameter((3 * f_out, 1), seed=seed + 2)       # attention vec
        self.res = parameter((f_in, f_out), seed=seed + 3)      # residual proj
        self.f_out = f_out

    def params(self):
        return [self.W, self.U, self.a, self.res]

    def __call__(self, H, src, dst, edge_block, num_nodes):
        Z = H @ self.W                                  # (BN, f_out)
        Ue = edge_block @ self.U                        # (E,  f_out)
        zi = Z.gather_rows(src)                         # (E, f_out)
        zj = Z.gather_rows(dst)                         # (E, f_out)
        logit = (concat([zi, zj, Ue], axis=1) @ self.a).leaky_relu(0.2)  # (E,1)
        alpha = segment_softmax(logit.reshape(-1), dst, num_nodes)       # (E,)
        msg = zi * alpha.reshape(-1, 1)                 # (E, f_out)
        agg = msg.scatter_add(dst, num_nodes)           # (BN, f_out)
        return (agg + H @ self.res).relu()              # residual + nonlinearity


class GATEncoder:
    """L_g GAT layers, applied with shared weights to each time step."""

    def __init__(self, f_in, f_hidden, f_edge, layers=2, seed=0):
        self.layers = []
        d_in = f_in
        for l in range(layers):
            self.layers.append(GATLayer(d_in, f_hidden, f_edge,
                                        seed=seed + 10 * l))
            d_in = f_hidden
        self.f_out = f_hidden

    def params(self):
        return [p for layer in self.layers for p in layer.params()]

    def encode_step(self, F_tau, src, dst, edge_block, num_nodes):
        H = F_tau
        for layer in self.layers:
            H = layer(H, src, dst, edge_block, num_nodes)
        return H                                        # (BN, f_out)


# --------------------------------------------------------------------------
class TCNLayer:
    """Dilated *causal* conv, kernel size 2 (combines step tau and tau-d),
    applied per node over the time axis.

        q_tau = relu( h_tau @ Wc_self + h_{tau-d} @ Wc_lag + b )  (+ residual)

    Causal padding: for tau < d the lagged term clamps to the earliest step,
    so no future information leaks.
    """

    def __init__(self, f_in, f_out, dilation, seed=0):
        self.Wc_self = parameter((f_in, f_out), seed=seed)
        self.Wc_lag = parameter((f_in, f_out), seed=seed + 1)
        self.b = Tensor(np.zeros(f_out), requires_grad=True)
        self.dilation = dilation
        self.residual = (f_in == f_out)

    def params(self):
        return [self.Wc_self, self.Wc_lag, self.b]

    def __call__(self, H_seq):
        W = len(H_seq)
        d = self.dilation
        out = []
        for tau in range(W):
            lag = H_seq[max(tau - d, 0)]                # causal, clamp at 0
            q = (H_seq[tau] @ self.Wc_self + lag @ self.Wc_lag + self.b).relu()
            if self.residual:
                q = q + H_seq[tau]
            out.append(q)
        return out


class TCNEncoder:
    """Dilated causal conv stack (dilations 1,2,4,... capped to W) + temporal
    attention pooling -> per-node window vector z_i^t."""

    def __init__(self, f_in, f_hidden, W, seed=0):
        self.layers = []
        d_in, dil, l = f_in, 1, 0
        while dil < max(W, 1):                          # cover full receptive field
            self.layers.append(TCNLayer(d_in, f_hidden, dilation=dil,
                                        seed=seed + 10 * l))
            d_in = f_hidden
            dil *= 2
            l += 1
        if not self.layers:                             # W == 1 edge case
            self.layers.append(TCNLayer(d_in, f_hidden, dilation=1, seed=seed))
            d_in = f_hidden
        self.f_out = d_in
        self.Wq = parameter((self.f_out, self.f_out), seed=seed + 500)
        self.vq = parameter((self.f_out, 1), seed=seed + 501)

    def params(self):
        return ([p for layer in self.layers for p in layer.params()]
                + [self.Wq, self.vq])

    def __call__(self, H_seq):
        Q_seq = H_seq
        for layer in self.layers:
            Q_seq = layer(Q_seq)                        # list of (BN, f_out)
        # temporal attention: u_tau = v^T tanh(Wq q_tau)
        scores = [(q @ self.Wq).tanh() @ self.vq for q in Q_seq]   # each (BN,1)
        beta = softmax_lastdim(concat(scores, axis=1))             # (BN, W)
        z = None
        for tau in range(len(Q_seq)):                   # z = sum_tau beta_tau q_tau
            term = Q_seq[tau] * _col(beta, tau)
            z = term if z is None else (z + term)
        return z                                        # (BN, f_out)


def _col(M, tau):
    """Differentiable extraction of column `tau` of (BN, W) as (BN, 1) via a
    one-hot matmul (flows through the existing autograd matmul rule)."""
    Wdim = M.shape[1]
    sel = np.zeros((Wdim, 1)); sel[tau, 0] = 1.0
    return M @ Tensor(sel)


# --------------------------------------------------------------------------
class STEncoder:
    """Full spatio-temporal encoder:  F_seq + G0  ->  Z  (Section 6)."""

    def __init__(self, F0, f_gat, f_tcn, f_edge, W, gat_layers=2, seed=0):
        self.gat = GATEncoder(F0, f_gat, f_edge, layers=gat_layers, seed=seed)
        self.tcn = TCNEncoder(f_gat, f_tcn, W, seed=seed + 1000)
        self.Fz = self.tcn.f_out

    def params(self):
        return self.gat.params() + self.tcn.params()

    def __call__(self, F_seq, src, dst, edge_block, num_nodes):
        """F_seq: list of W tensors (BN, F0). Returns Z (BN, Fz)."""
        H_seq = [self.gat.encode_step(F_tau, src, dst, edge_block, num_nodes)
                 for F_tau in F_seq]                    # GAT per time step
        return self.tcn(H_seq)                          # TCN + temporal pooling
