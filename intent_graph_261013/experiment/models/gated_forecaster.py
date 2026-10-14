"""One controlled forecasting skeleton with local, fixed, static, or dynamic gates.

PyTorch is imported only after the registry selects a neural model. Spatial
information has exactly one route: gated messages with a fixed graph indegree.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn


class CausalTCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim, kernel_size, dilation=1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, dilation=2),
        ])
        self.left_padding = [(kernel_size - 1), 2 * (kernel_size - 1)]

    def forward(self, history):
        # Each node is an independent sequence; there is no graph mixing here.
        batch, window, nodes, features = history.shape
        sequence = history.permute(0, 2, 3, 1).reshape(batch * nodes, features, window)
        for layer, padding in zip(self.layers, self.left_padding):
            sequence = torch.relu(layer(torch.nn.functional.pad(sequence, (padding, 0))))
        return sequence[:, :, -1].reshape(batch, nodes, -1)


class ForecastModule(nn.Module):
    def __init__(self, name, data, config):
        super().__init__()
        self.name = name
        self.window = int(data.protocol["W"])
        self.horizon = int(data.protocol["H"])
        self.n_nodes = int(data.features.shape[1])
        input_features = int(data.features.shape[2])
        hidden = int(config.get("encoder_hidden", 24))
        msg_dim = int(config.get("msg_dim", 16))
        head_hidden = int(config.get("head_hidden", 32))
        encoder = config.get("encoder", "mlp")
        if min(hidden, msg_dim, head_hidden) <= 0:
            raise ValueError("Neural dimensions must be positive.")
        if encoder == "mlp":
            self.encoder = nn.Sequential(nn.Linear(self.window * input_features, hidden),
                                         nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        elif encoder == "tcn":
            self.encoder = CausalTCN(input_features, hidden)
        else:
            raise ValueError(f"Unknown encoder {encoder!r}; use mlp or tcn.")
        self.encoder_kind = encoder
        src = np.asarray(data.graph["src"], dtype=np.int64)
        dst = np.asarray(data.graph["dst"], dtype=np.int64)
        if src.shape != dst.shape or src.ndim != 1:
            raise ValueError("Candidate graph src/dst must be one-dimensional equal arrays.")
        if np.any(src < 0) or np.any(dst < 0) or np.any(src >= self.n_nodes) or np.any(dst >= self.n_nodes):
            raise ValueError("Candidate graph contains out-of-range nodes.")
        degree = np.maximum(1, np.bincount(dst, minlength=self.n_nodes))
        self.register_buffer("src", torch.as_tensor(src, dtype=torch.long))
        self.register_buffer("dst", torch.as_tensor(dst, dtype=torch.long))
        self.register_buffer("degree", torch.as_tensor(degree, dtype=torch.float32))
        lookup = {(int(i), int(j)): edge for edge, (i, j) in enumerate(zip(src, dst))}
        pairs = [(edge, lookup[(int(j), int(i))]) for edge, (i, j) in enumerate(zip(src, dst))
                 if (int(j), int(i)) in lookup and edge < lookup[(int(j), int(i))]]
        self.register_buffer("pair_a", torch.as_tensor([a for a, _ in pairs], dtype=torch.long))
        self.register_buffer("pair_b", torch.as_tensor([b for _, b in pairs], dtype=torch.long))
        self.msg_dim = msg_dim
        # Consume the same initialization draws for shared components, even
        # when the local control has no trainable message projection.
        message = nn.Linear(hidden, msg_dim, bias=False)
        self.message = None if name == "local" else message
        self.head = nn.Sequential(nn.Linear(hidden + msg_dim, head_hidden), nn.ReLU(),
                                  nn.Linear(head_hidden, self.horizon))
        if name == "static_gate":
            self.gate_logits = nn.Parameter(torch.zeros(len(src)))
        elif name == "dynamic_gate":
            gate_hidden = int(config.get("gate_hidden", 32))
            if gate_hidden <= 0:
                raise ValueError("gate_hidden must be positive.")
            self.gate = nn.Sequential(nn.Linear(4 * hidden, gate_hidden), nn.ReLU(),
                                      nn.Linear(gate_hidden, 1))
        # Every control model starts at persistence, before learned residuals.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def encode(self, history):
        if self.encoder_kind == "mlp":
            batch, window, nodes, features = history.shape
            own_history = history.permute(0, 2, 1, 3).reshape(batch, nodes, window * features)
            return self.encoder(own_history)
        return self.encoder(history)

    def raw_scores(self, representation):
        batch = representation.shape[0]
        if self.name == "local":
            return None
        if self.name == "fixed_graph":
            return representation.new_ones((batch, len(self.src)))
        if self.name == "static_gate":
            return torch.sigmoid(self.gate_logits).unsqueeze(0).expand(batch, -1)
        zi, zj = representation[:, self.src], representation[:, self.dst]
        return torch.sigmoid(self.gate(torch.cat([zi, zj, zi - zj, zi * zj], dim=-1))).squeeze(-1)

    def forward(self, history, last, edge_mask=None):
        representation = self.encode(history)
        scores = self.raw_scores(representation)
        batch = history.shape[0]
        aggregation = representation.new_zeros((batch, self.n_nodes, self.msg_dim))
        effective = None
        if scores is not None:
            masked = scores if edge_mask is None else scores * edge_mask
            effective = masked / self.degree[self.dst].unsqueeze(0)
            message = self.message(representation)[:, self.src] * effective.unsqueeze(-1)
            aggregation.index_add_(1, self.dst, message)
        residual = self.head(torch.cat([representation, aggregation], dim=-1))
        return {"prediction": last.unsqueeze(-1) + residual,
                "edge_scores": scores, "effective_weight": effective,
                "aggregation": aggregation}


class GatedForecaster:
    kind = "gradient"

    def __init__(self, name, data, config=None, seed=0):
        self.name = name
        self.config = dict(config or {})
        unknown = set(self.config) - {"encoder", "encoder_hidden", "msg_dim", "head_hidden", "gate_hidden"}
        if unknown:
            raise ValueError(f"Unknown neural model parameters: {sorted(unknown)}")
        self.seed = int(seed)
        self.device = torch.device("cpu")
        # A local RNG context keeps model initialization reproducible without
        # coupling it to run ordering or another model's initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.module = ForecastModule(name, data, self.config).to(self.device)
        score_kind = {"local": "none", "fixed_graph": "fixed", "static_gate": "static",
                      "dynamic_gate": "dynamic"}[name]
        self.capabilities = {"supports_ranking": name in {"static_gate", "dynamic_gate"},
                             "supports_intervention": name != "local", "score_kind": score_kind,
                             "intervention_scope": "single_message" if name != "local" else None}
        self.parameter_count = sum(parameter.numel() for parameter in self.module.parameters())

    def configure_device(self, requested="auto"):
        """Resolve and move all parameters and graph buffers together."""
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            device = torch.device(requested)
        except (RuntimeError, TypeError) as exc:
            raise ValueError("device must be auto, cpu, cuda, or a CUDA index such as cuda:0.") from exc
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("Only CPU and CUDA devices are supported.")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was explicitly requested but is unavailable in this PyTorch runtime.")
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise RuntimeError(f"CUDA device index {device.index} is unavailable.")
        self.device = device
        self.module.to(device)
        return self.device_metadata()

    def device_metadata(self):
        return {"device": str(self.device),
                "gpu_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
                "torch_version": str(torch.__version__), "cuda_runtime": torch.version.cuda,
                "precision": "float32", "mixed_precision": False}

    def _tensor_batch(self, batch, edge_mask=None):
        history = torch.as_tensor(batch["history"], dtype=torch.float32, device=self.device)
        last = torch.as_tensor(batch["last"], dtype=torch.float32, device=self.device)
        if history.ndim != 4 or last.shape != (history.shape[0], self.module.n_nodes):
            raise ValueError("Expected history[B,W,N,F] and last[B,N].")
        mask = None
        if edge_mask is not None:
            if not self.capabilities["supports_intervention"]:
                raise ValueError(f"{self.name} does not support edge interventions.")
            mask = torch.as_tensor(edge_mask, dtype=torch.float32, device=self.device)
            if mask.shape != (history.shape[0], len(self.module.src)):
                raise ValueError("edge_mask must have shape [B,E].")
            if not torch.isfinite(mask).all() or torch.any(mask < 0) or torch.any(mask > 1):
                raise ValueError("edge_mask must be finite and lie in [0,1].")
        return history, last, mask

    def forward(self, batch, edge_mask=None):
        history, last, mask = self._tensor_batch(batch, edge_mask)
        return self.module(history, last, mask)

    def predict(self, batch, edge_mask=None):
        self.module.eval()
        with torch.no_grad():
            output = self.forward(batch, edge_mask)
        return {name: None if output[name] is None else output[name].detach().cpu().numpy()
                for name in ("prediction", "edge_scores", "effective_weight")}

    def fit(self, train_dataset, val_dataset, training_config=None):
        from ..trainer import fit_neural
        return fit_neural(self, train_dataset, val_dataset, dict(training_config or {}))

    def save(self, path):
        torch.save({"model": self.name, "config": self.config, "seed": self.seed,
                    "device": str(self.device),
                    "state_dict": {name: value.detach().cpu() for name, value in self.module.state_dict().items()}},
                   Path(path))
