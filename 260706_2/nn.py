"""
nn.py
=====
Neural-network building blocks used by the RA-TCGF model, implemented on top of
the custom autograd `Tensor`.

Layers
------
Linear         : affine transform  y = xW + b
GraphConv      : one message-passing layer  Z = act( A_hat @ X @ W + X @ W_self )
GNNEncoder     : stack of GraphConv layers (the spatial encoder, Module 2)
TCN            : dilated causal temporal conv network (the temporal encoder)
AttnReadout    : attention pooling  u = sum_i beta_i z_i   (graph-level readout)
MLP            : generic multi-layer perceptron (used by the gating heads)
"""

from __future__ import annotations

import numpy as np

from .autograd import Tensor, cat, softmax, conv1d_causal


# --------------------------------------------------------------------------- #
#  Parameter initialisation                                                    #
# --------------------------------------------------------------------------- #
def glorot(shape, rng):
    fan_in, fan_out = shape[0], shape[-1]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return Tensor(rng.uniform(-limit, limit, size=shape), requires_grad=True)


def zeros(shape):
    return Tensor(np.zeros(shape), requires_grad=True)


class Module:
    """Base class: collects parameters recursively for the optimizer."""

    def parameters(self):
        params = []
        for v in self.__dict__.values():
            if isinstance(v, Tensor) and v.requires_grad:
                params.append(v)
            elif isinstance(v, Module):
                params.extend(v.parameters())
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
                    elif isinstance(item, Tensor) and item.requires_grad:
                        params.append(item)
        return params


# --------------------------------------------------------------------------- #
#  Basic layers                                                                #
# --------------------------------------------------------------------------- #
class Linear(Module):
    def __init__(self, d_in, d_out, rng, bias=True):
        self.W = glorot((d_in, d_out), rng)
        self.b = zeros((d_out,)) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        y = x @ self.W
        if self.b is not None:
            y = y + self.b
        return y


class MLP(Module):
    def __init__(self, dims, rng, act="relu", out_act=None):
        self.layers = [Linear(dims[i], dims[i + 1], rng) for i in range(len(dims) - 1)]
        self.act = act
        self.out_act = out_act

    def __call__(self, x: Tensor) -> Tensor:
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < n - 1:
                x = x.relu() if self.act == "relu" else x.tanh()
            else:
                if self.out_act == "relu":
                    x = x.relu()
                elif self.out_act == "tanh":
                    x = x.tanh()
        return x


# --------------------------------------------------------------------------- #
#  Graph convolution (GraphSAGE / GCN style, edge-weight aware)                #
# --------------------------------------------------------------------------- #
class GraphConv(Module):
    """
    Z = act( A_hat @ (X W_neigh) + X W_self + b )

    `A_hat` is the (constant) row-normalised weighted adjacency of a view.  Edge
    strengths therefore modulate the aggregation, which is what the scheme's
    A_m^tau[i,j] entries are meant to do.
    """

    def __init__(self, d_in, d_out, rng, act="relu"):
        self.W_neigh = glorot((d_in, d_out), rng)
        self.W_self = glorot((d_in, d_out), rng)
        self.b = zeros((d_out,))
        self.act = act

    def __call__(self, X: Tensor, A_hat: Tensor) -> Tensor:
        msg = A_hat @ (X @ self.W_neigh)
        out = msg + (X @ self.W_self) + self.b
        if self.act == "relu":
            out = out.relu()
        elif self.act == "tanh":
            out = out.tanh()
        return out


class GNNEncoder(Module):
    """Stack of GraphConv layers producing node-level spatial embeddings."""

    def __init__(self, d_in, d_hidden, n_layers, rng):
        dims = [d_in] + [d_hidden] * n_layers
        self.convs = [
            GraphConv(dims[i], dims[i + 1], rng, act="relu")
            for i in range(n_layers)
        ]

    def __call__(self, X: Tensor, A_hat: Tensor) -> Tensor:
        h = X
        for conv in self.convs:
            h = conv(h, A_hat)
        return h  # [N, d_hidden]


# --------------------------------------------------------------------------- #
#  Attention readout  (graph-level pooling)                                     #
# --------------------------------------------------------------------------- #
class AttnReadout(Module):
    def __init__(self, d, rng):
        self.Wr = glorot((d, d), rng)
        self.wr = glorot((d, 1), rng)

    def __call__(self, Z: Tensor) -> Tensor:
        # Z: [N, d]
        scores = (Z @ self.Wr).tanh() @ self.wr        # [N, 1]
        beta = softmax(scores, axis=0)                 # [N, 1]
        u = (beta * Z).sum(axis=0)                     # [d]
        return u


# --------------------------------------------------------------------------- #
#  Temporal Convolutional Network (dilated, causal)                            #
# --------------------------------------------------------------------------- #
class TCN(Module):
    """
    A small dilated causal TCN.  Input/output tensors use the [B, C, T]
    convention.  The representation of the *last* time step is returned as the
    temporal summary h^t.
    """

    def __init__(self, c_in, c_hidden, c_out, rng, kernel=2, dilations=(1, 2)):
        self.kernel = kernel
        self.dilations = dilations
        self.weights = []
        self.biases = []
        chans = [c_in] + [c_hidden] * (len(dilations) - 1) + [c_out]
        for i, d in enumerate(dilations):
            w = glorot((chans[i + 1], chans[i], kernel), rng)
            b = zeros((chans[i + 1],))
            self.weights.append(w)
            self.biases.append(b)

    def __call__(self, x: Tensor) -> Tensor:
        # x : [B, C_in, T]
        h = x
        for i, d in enumerate(self.dilations):
            h = conv1d_causal(h, self.weights[i], self.biases[i], dilation=d)
            h = h.relu()
        # take last time step -> [B, C_out]
        B, C, T = h.data.shape
        last = h.reshape(B, C, T)                       # no-op, kept for clarity
        # slice last timestep via matmul with a one-hot selector (keeps autograd)
        sel = np.zeros((T, 1)); sel[-1, 0] = 1.0
        sel_t = Tensor(sel)
        out = (last @ sel_t).reshape(B, C)              # [B, C]
        return out
