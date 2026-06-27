"""
Minimal reverse-mode automatic differentiation engine built on NumPy.

This is intentionally small: it supports exactly the operations needed by the
DEIG-TCN trainable head (edge-level intent scoring + intent-weighted message
passing + predictor). PyTorch is not available in this environment, so we roll
our own tiny autograd so that the edge-intent scores receive a real, prediction
driven (self-supervised) training signal as described in the method spec.

Supported ops: +, -, *, matmul, sum, mean, relu, sigmoid, tanh, exp, log,
gather(rows), concat(axis=1), scatter_add (segment sum), segment_softmax,
and a Huber loss helper.
"""

from __future__ import annotations

import numpy as np


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Reduce `grad` so its shape matches `shape` (reverse of broadcasting)."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_backward", "_parents")

    def __init__(self, data, requires_grad: bool = False, _parents=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._parents = _parents

    # ---- helpers -------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def _ensure_grad(self):
        if self.grad is None:
            self.grad = np.zeros_like(self.data)

    @staticmethod
    def _as_tensor(x):
        return x if isinstance(x, Tensor) else Tensor(x)

    # ---- elementwise ---------------------------------------------------
    def __add__(self, other):
        other = self._as_tensor(other)
        out = Tensor(self.data + other.data,
                     self.requires_grad or other.requires_grad,
                     (self, other))

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other._ensure_grad()
                other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        out = Tensor(-self.data, self.requires_grad, (self,))

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += -out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-self._as_tensor(other))

    def __rsub__(self, other):
        return self._as_tensor(other) + (-self)

    def __mul__(self, other):
        other = self._as_tensor(other)
        out = Tensor(self.data * other.data,
                     self.requires_grad or other.requires_grad,
                     (self, other))

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other._ensure_grad()
                other.grad += _unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def matmul(self, other):
        other = self._as_tensor(other)
        out = Tensor(self.data @ other.data,
                     self.requires_grad or other.requires_grad,
                     (self, other))

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other._ensure_grad()
                other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    # ---- reductions ----------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     self.requires_grad, (self,))

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                g = out.grad
                if axis is not None and not keepdims:
                    g = np.expand_dims(g, axis)
                self.grad += np.broadcast_to(g, self.data.shape).copy()
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ---- nonlinearities ------------------------------------------------
    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), self.requires_grad, (self,))

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, self.requires_grad, (self,))

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, self.requires_grad, (self,))

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, self.requires_grad, (self,))

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), self.requires_grad, (self,))

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    # ---- structural ----------------------------------------------------
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape), self.requires_grad, (self,))

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def gather_rows(self, idx: np.ndarray):
        """out = self.data[idx] along axis 0."""
        idx = np.asarray(idx, dtype=np.int64)
        out = Tensor(self.data[idx], self.requires_grad, (self,))

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                np.add.at(self.grad, idx, out.grad)
        out._backward = _backward
        return out

    def scatter_add(self, idx: np.ndarray, size: int):
        """Segment sum: out[k] = sum_{e: idx[e]==k} self.data[e]."""
        idx = np.asarray(idx, dtype=np.int64)
        shape = (size,) + self.data.shape[1:]
        acc = np.zeros(shape, dtype=np.float64)
        np.add.at(acc, idx, self.data)
        out = Tensor(acc, self.requires_grad, (self,))

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad[idx]
        out._backward = _backward
        return out

    # ---- graph traversal ----------------------------------------------
    def backward(self):
        topo, visited = [], set()

        def build(v):
            if id(v) in visited:
                return
            visited.add(id(v))
            for p in v._parents:
                build(p)
            topo.append(v)
        build(self)

        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


# ---- functional helpers ------------------------------------------------
def concat(tensors, axis=1):
    datas = [t.data for t in tensors]
    out = Tensor(np.concatenate(datas, axis=axis),
                 any(t.requires_grad for t in tensors), tuple(tensors))
    sizes = [t.data.shape[axis] for t in tensors]
    bounds = np.cumsum([0] + sizes)

    def _backward():
        for k, t in enumerate(tensors):
            if t.requires_grad:
                t._ensure_grad()
                sl = [slice(None)] * out.grad.ndim
                sl[axis] = slice(bounds[k], bounds[k + 1])
                t.grad += out.grad[tuple(sl)]
    out._backward = _backward
    return out


def segment_softmax(scores: Tensor, seg_idx: np.ndarray, num_seg: int,
                    eps: float = 1e-9) -> Tensor:
    """Softmax of `scores` (shape (E,)) within groups given by seg_idx.

    Used to normalise edge intent weights per receiving node:
        s_tilde_ij = s_ij / sum_{p->j} s_pj   (here a softmax variant).
    """
    e = scores.exp()                                   # (E,)
    denom = e.scatter_add(seg_idx, num_seg) + eps      # (num_seg,)
    denom_per_edge = denom.gather_rows(seg_idx)         # (E,)
    return _safe_div(e, denom_per_edge)


def _safe_div(num: Tensor, den: Tensor) -> Tensor:
    out = Tensor(num.data / den.data, num.requires_grad or den.requires_grad,
                 (num, den))

    def _backward():
        if num.requires_grad:
            num._ensure_grad()
            num.grad += out.grad / den.data
        if den.requires_grad:
            den._ensure_grad()
            den.grad += -out.grad * num.data / (den.data ** 2)
    out._backward = _backward
    return out


def huber_loss(pred: Tensor, target: np.ndarray, kappa: float = 1.0) -> Tensor:
    """Mean Huber loss; differentiable wrt `pred`."""
    diff = pred + (-Tensor(target))            # pred - target
    a = np.abs(diff.data)
    quad = a <= kappa
    sq = diff * diff * 0.5
    quad_mask = Tensor(quad.astype(np.float64))
    lin_mask = Tensor((~quad).astype(np.float64))
    abs_diff = _abs(diff)
    lin_term = (abs_diff - 0.5 * kappa) * kappa
    loss = (sq * quad_mask) + (lin_term * lin_mask)
    return loss.mean()


def _abs(x: Tensor) -> Tensor:
    out = Tensor(np.abs(x.data), x.requires_grad, (x,))

    def _backward():
        if x.requires_grad:
            x._ensure_grad()
            x.grad += np.sign(x.data) * out.grad
    out._backward = _backward
    return out


class Adam:
    """Adam optimizer over a list of Tensor parameters."""

    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
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


def parameter(shape, scale=None, seed=None):
    rng = np.random.default_rng(seed)
    if scale is None:
        fan_in = shape[0] if len(shape) >= 1 else 1
        scale = np.sqrt(2.0 / max(fan_in, 1))
    return Tensor(rng.standard_normal(shape) * scale, requires_grad=True)
