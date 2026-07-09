"""
autograd.py
===========
A compact reverse-mode automatic differentiation engine built on top of NumPy.

This module exists because the target environment has no deep-learning framework
(PyTorch / TensorFlow / JAX are unavailable and the network is disabled).  It
provides a tensor-valued autograd `Tensor` type together with the operations
required by the RA-TCGF model: dense linear algebra, elementwise non-linearities,
reductions, concatenation and a causal dilated 1-D convolution (for the TCN).

The design follows the classic "define-by-run" pattern: every operation records
a local backward closure and its parent tensors; `Tensor.backward()` performs a
topological sort and propagates gradients.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum `grad` so that its shape matches `shape` (reverse of broadcasting)."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


# --------------------------------------------------------------------------- #
#  Tensor                                                                      #
# --------------------------------------------------------------------------- #
class Tensor:
    """A minimal autograd tensor wrapping a float64 NumPy array."""

    __slots__ = ("data", "grad", "_backward", "_prev", "requires_grad")

    def __init__(self, data, requires_grad: bool = False, _prev=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_prev)

    # -- construction helpers ------------------------------------------------ #
    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"

    def _child(self, data, parents):
        req = any(p.requires_grad for p in parents)
        return Tensor(data, requires_grad=req, _prev=parents)

    # -- elementwise binary -------------------------------------------------- #
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._child(self.data + other.data, (self, other))

        def _backward():
            if self.requires_grad:
                self.grad = (0 if self.grad is None else self.grad) + _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad = (0 if other.grad is None else other.grad) + _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._child(self.data * other.data, (self, other))

        def _backward():
            if self.requires_grad:
                self.grad = (0 if self.grad is None else self.grad) + _unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad = (0 if other.grad is None else other.grad) + _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __neg__(self):
        out = self._child(-self.data, (self,))

        def _backward():
            if self.requires_grad:
                self.grad = (0 if self.grad is None else self.grad) - out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._child(self.data / other.data, (self, other))

        def _backward():
            if self.requires_grad:
                self.grad = (0 if self.grad is None else self.grad) + _unbroadcast(out.grad / other.data, self.data.shape)
            if other.requires_grad:
                g = -out.grad * self.data / (other.data ** 2)
                other.grad = (0 if other.grad is None else other.grad) + _unbroadcast(g, other.data.shape)

        out._backward = _backward
        return out

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other):
        return (-self) + other

    # -- matmul -------------------------------------------------------------- #
    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._child(self.data @ other.data, (self, other))

        def _backward():
            if self.requires_grad:
                self.grad = (0 if self.grad is None else self.grad) + out.grad @ other.data.swapaxes(-1, -2)
            if other.requires_grad:
                other.grad = (0 if other.grad is None else other.grad) + self.data.swapaxes(-1, -2) @ out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    # -- reductions ---------------------------------------------------------- #
    def sum(self, axis=None, keepdims=False):
        out = self._child(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def _backward():
            if self.requires_grad:
                g = out.grad
                if axis is not None and not keepdims:
                    g = np.expand_dims(g, axis=axis)
                self.grad = (0 if self.grad is None else self.grad) + np.broadcast_to(g, self.data.shape).copy()

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # -- unary non-linearities ---------------------------------------------- #
    def relu(self):
        out = self._child(np.maximum(self.data, 0.0), (self,))

        def _backward():
            if self.requires_grad:
                self.grad = (0 if self.grad is None else self.grad) + out.grad * (self.data > 0)

        out._backward = _backward
        return out

    def leaky_relu(self, slope=0.2):
        out = self._child(np.where(self.data > 0, self.data, slope * self.data), (self,))

        def _backward():
            if self.requires_grad:
                self.grad = (0 if self.grad is None else self.grad) + out.grad * np.where(self.data > 0, 1.0, slope)

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = self._child(s, (self,))

        def _backward():
            if self.requires_grad:
                self.grad = (0 if self.grad is None else self.grad) + out.grad * s * (1 - s)

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = self._child(t, (self,))

        def _backward():
            if self.requires_grad:
                self.grad = (0 if self.grad is None else self.grad) + out.grad * (1 - t ** 2)

        out._backward = _backward
        return out

    def exp(self):
        e = np.exp(self.data)
        out = self._child(e, (self,))

        def _backward():
            if self.requires_grad:
                self.grad = (0 if self.grad is None else self.grad) + out.grad * e

        out._backward = _backward
        return out

    def log(self):
        out = self._child(np.log(self.data + 1e-12), (self,))

        def _backward():
            if self.requires_grad:
                self.grad = (0 if self.grad is None else self.grad) + out.grad / (self.data + 1e-12)

        out._backward = _backward
        return out

    # -- shape ops ----------------------------------------------------------- #
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = self._child(self.data.reshape(shape), (self,))

        def _backward():
            if self.requires_grad:
                self.grad = (0 if self.grad is None else self.grad) + out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, *axes):
        axes = axes if axes else None
        out = self._child(np.transpose(self.data, axes), (self,))

        def _backward():
            if self.requires_grad:
                inv = np.argsort(axes) if axes is not None else None
                g = np.transpose(out.grad, inv) if inv is not None else out.grad.T
                self.grad = (0 if self.grad is None else self.grad) + g

        out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose()

    # -- backward pass ------------------------------------------------------- #
    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build(parent)
                topo.append(v)

        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


# --------------------------------------------------------------------------- #
#  Functional ops                                                              #
# --------------------------------------------------------------------------- #
def cat(tensors, axis=-1):
    """Concatenate a list of Tensors along `axis`."""
    datas = [t.data for t in tensors]
    out_data = np.concatenate(datas, axis=axis)
    parents = tuple(tensors)
    req = any(t.requires_grad for t in tensors)
    out = Tensor(out_data, requires_grad=req, _prev=parents)

    sizes = [t.data.shape[axis] for t in tensors]
    splits = np.cumsum(sizes)[:-1]

    def _backward():
        grads = np.split(out.grad, splits, axis=axis)
        for t, g in zip(tensors, grads):
            if t.requires_grad:
                t.grad = (0 if t.grad is None else t.grad) + g

    out._backward = _backward
    return out


def softmax(t: Tensor, axis=-1):
    """Numerically-stable softmax expressed through primitive ops."""
    m = Tensor(t.data.max(axis=axis, keepdims=True))       # constant shift
    e = (t - m).exp()
    s = e.sum(axis=axis, keepdims=True)
    return e / s


def conv1d_causal(x: Tensor, weight: Tensor, bias: Tensor, dilation: int = 1):
    """
    Causal dilated 1-D convolution.

    x      : Tensor [B, C_in, T]
    weight : Tensor [C_out, C_in, K]
    bias   : Tensor [C_out]
    returns: Tensor [B, C_out, T]  (length preserved via left/causal padding)
    """
    B, C_in, T = x.data.shape
    C_out, _, K = weight.data.shape
    pad = dilation * (K - 1)
    xp = np.pad(x.data, ((0, 0), (0, 0), (pad, 0)))          # left pad only -> causal

    # im2col: build [B, T, C_in*K]
    cols = np.empty((B, T, C_in * K), dtype=np.float64)
    for k in range(K):
        start = k * dilation
        cols[:, :, k * C_in:(k + 1) * C_in] = xp[:, :, start:start + T].transpose(0, 2, 1)
    w2d = weight.data.transpose(2, 1, 0).reshape(C_in * K, C_out)   # [C_in*K, C_out]
    out_data = cols @ w2d + bias.data.reshape(1, 1, C_out)          # [B, T, C_out]
    out_data = out_data.transpose(0, 2, 1)                          # [B, C_out, T]

    req = x.requires_grad or weight.requires_grad or bias.requires_grad
    out = Tensor(out_data, requires_grad=req, _prev=(x, weight, bias))

    def _backward():
        g = out.grad.transpose(0, 2, 1)                            # [B, T, C_out]
        if bias.requires_grad:
            bias.grad = (0 if bias.grad is None else bias.grad) + g.sum(axis=(0, 1))
        if weight.requires_grad:
            gw = np.einsum("btc,btk->kc", g, cols)                 # [C_in*K, C_out]
            gw = gw.reshape(K, C_in, C_out).transpose(2, 1, 0)     # [C_out, C_in, K]
            weight.grad = (0 if weight.grad is None else weight.grad) + gw
        if x.requires_grad:
            gcols = g @ w2d.T                                      # [B, T, C_in*K]
            gxp = np.zeros_like(xp)
            for k in range(K):
                start = k * dilation
                gxp[:, :, start:start + T] += gcols[:, :, k * C_in:(k + 1) * C_in].transpose(0, 2, 1)
            x.grad = (0 if x.grad is None else x.grad) + gxp[:, :, pad:]

    out._backward = _backward
    return out


# --------------------------------------------------------------------------- #
#  Optimizer                                                                   #
# --------------------------------------------------------------------------- #
class Adam:
    """Adam optimizer operating on a list of parameter Tensors."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad + self.wd * p.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
