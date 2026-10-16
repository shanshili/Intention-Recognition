"""M1--M7 incremental DEIG construction, sharing the existing training API.

The first directed spatial attention route and the post-score message route
are distinct. ``edge_mask`` intervenes on the second route only. Self paths
exist in GAT attention but are not candidate intention edges.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .gated_forecaster import GatedForecaster
from .deig_variants import resolve_model_spec


FULL_FEATURE_CHANNELS = ("level", "delta", "periodic_residual", "time_sin",
                         "time_cos", "weekday_sin", "weekday_cos")


class DirectedGAT(nn.Module):
    """Single-head directed attention normalized over each receiver's inputs."""

    def __init__(self, input_dim, hidden_dim, nodes, src, dst, edge_features=None,
                 negative_slope=0.2):
        super().__init__()
        self.nodes = int(nodes)
        self.hidden_dim = int(hidden_dim)
        self.negative_slope = float(negative_slope)
        self.projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.attention_src = nn.Parameter(torch.empty(hidden_dim))
        self.attention_dst = nn.Parameter(torch.empty(hidden_dim))
        nn.init.xavier_uniform_(self.attention_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.attention_dst.unsqueeze(0))
        self.register_buffer("src", torch.as_tensor(src, dtype=torch.long))
        self.register_buffer("dst", torch.as_tensor(dst, dtype=torch.long))
        own = torch.arange(nodes, dtype=torch.long)
        self.register_buffer("attention_src_nodes", torch.cat([self.src, own]))
        self.register_buffer("attention_dst_nodes", torch.cat([self.dst, own]))
        self.edge_projection = None
        if edge_features is not None:
            edge_features = torch.as_tensor(edge_features, dtype=torch.float32)
            self.register_buffer("edge_features", edge_features)
            self.edge_projection = nn.Linear(edge_features.shape[1], 1, bias=False)

    def forward(self, history, return_attention=False):
        batch, window, nodes, _ = history.shape
        if nodes != self.nodes:
            raise ValueError("GAT history node count differs from candidate graph.")
        projected = self.projection(history)
        source = projected[:, :, self.attention_src_nodes]
        receiver = projected[:, :, self.attention_dst_nodes]
        logits = ((source * self.attention_src).sum(dim=-1)
                  + (receiver * self.attention_dst).sum(dim=-1))
        if self.edge_projection is not None:
            prior = self.edge_projection(self.edge_features).squeeze(-1)
            # A self-attention path is explicit, with a zero edge-prior term.
            prior = torch.cat([prior, prior.new_zeros(self.nodes)])
            logits = logits + prior.view(1, 1, -1)
        logits = F.leaky_relu(logits, negative_slope=self.negative_slope)
        dst_index = self.attention_dst_nodes.view(1, 1, -1).expand(batch, window, -1)
        maximum = logits.new_full((batch, window, nodes), -torch.inf)
        maximum.scatter_reduce_(2, dst_index, logits, reduce="amax", include_self=True)
        exponent = torch.exp(logits - maximum.gather(2, dst_index))
        denominator = exponent.new_zeros((batch, window, nodes))
        denominator.scatter_add_(2, dst_index, exponent)
        attention = exponent / denominator.gather(2, dst_index)
        aggregate = projected.new_zeros((batch, window, nodes, self.hidden_dim))
        aggregate.index_add_(2, self.attention_dst_nodes, source * attention.unsqueeze(-1))
        output = F.elu(aggregate)
        return (output, attention) if return_attention else output


class FullSequenceCausalTCN(nn.Module):
    """Causal convolutions returning every historical position, not only last."""

    def __init__(self, input_dim, hidden_dim, kernel_size=3, dilations=(1, 2, 4)):
        super().__init__()
        if kernel_size < 1 or not dilations or any(int(d) != d or d <= 0 for d in dilations):
            raise ValueError("TCN requires a positive kernel and positive integer dilations.")
        self.kernel_size = int(kernel_size)
        self.dilations = tuple(int(d) for d in dilations)
        self.receptive_field = 1 + (self.kernel_size - 1) * sum(self.dilations)
        self.layers = nn.ModuleList([
            nn.Conv1d(input_dim if index == 0 else hidden_dim, hidden_dim,
                      self.kernel_size, dilation=dilation)
            for index, dilation in enumerate(self.dilations)
        ])
        self.left_padding = [(self.kernel_size - 1) * d for d in self.dilations]

    def forward(self, history):
        batch, window, nodes, channels = history.shape
        sequence = history.permute(0, 2, 3, 1).reshape(batch * nodes, channels, window)
        for layer, padding in zip(self.layers, self.left_padding):
            sequence = torch.relu(layer(F.pad(sequence, (padding, 0))))
        return sequence.reshape(batch, nodes, -1, window).permute(0, 3, 1, 2)


def _max_lag_correlation(residual, src, dst, max_lag):
    """History-only i(past)->j(later) Pearson maximum, constant pairs zero."""
    batch, window, _ = residual.shape
    available = min(int(max_lag), window - 2)
    if available < 1 or not len(src):
        return residual.new_zeros((batch, len(src)))
    correlations = []
    valid_pairs = []
    for lag in range(1, available + 1):
        a, b = residual[:, :-lag, src], residual[:, lag:, dst]
        a, b = a - a.mean(dim=1, keepdim=True), b - b.mean(dim=1, keepdim=True)
        norm_a, norm_b = a.square().sum(dim=1), b.square().sum(dim=1)
        valid = (norm_a > 1e-12) & (norm_b > 1e-12)
        denominator = torch.sqrt((norm_a * norm_b).clamp_min(1e-24))
        correlation = ((a * b).sum(dim=1) / denominator).clamp(-1, 1)
        correlations.append(correlation.masked_fill(~valid, -torch.inf))
        valid_pairs.append(valid)
    result = torch.stack(correlations, dim=-1).amax(dim=-1)
    valid = torch.stack(valid_pairs, dim=-1).any(dim=-1)
    return torch.where(valid, result, torch.zeros_like(result))


def dynamic_edge_features(history, src, dst, max_lag=4):
    """The six explicit window features from residuals and level differences."""
    if history.shape[-1] != 7:
        raise ValueError("Explicit dynamic edge features require the full seven node channels.")
    residual, delta = history[..., 2], history[..., 1]
    forward = _max_lag_correlation(residual, src, dst, max_lag)
    reverse = _max_lag_correlation(residual, dst, src, max_lag)
    anomaly = residual.abs().mean(dim=1)
    # The first delta references a point before this window. Trends use only
    # its W-1 internal transitions, matching the original window definition.
    trend = delta[:, 1:].mean(dim=1) if history.shape[1] > 1 else torch.zeros_like(delta[:, 0])
    return torch.stack([forward, forward - reverse, anomaly[:, src], anomaly[:, dst],
                        trend[:, src], trend[:, dst]], dim=-1)


class DeigTCNModule(nn.Module):
    def __init__(self, name, data, config):
        super().__init__()
        self.name = name
        self.spec = resolve_model_spec(name)
        self.version = self.spec["version"]
        self.window, self.horizon = int(data.protocol["W"]), int(data.protocol["H"])
        self.n_nodes = int(data.features.shape[1])
        self.input_features = int(data.features.shape[2])
        expected_features = self.spec["input_features"]
        self.spatial_input_features = self.spec["spatial_input_features"]
        if self.input_features != expected_features:
            raise ValueError(f"{name} requires {expected_features} node channels; got {self.input_features}.")
        channels = data.protocol.get("feature_channels")
        if self.version >= 3 and channels is not None and tuple(channels) != FULL_FEATURE_CHANNELS:
            raise ValueError(f"{name} requires channel order {FULL_FEATURE_CHANNELS}.")
        hidden = int(config.get("encoder_hidden", 24))
        gat_hidden = int(config.get("gat_hidden", hidden))
        tcn_hidden = int(config.get("tcn_hidden", hidden))
        self.msg_dim = int(config.get("msg_dim", 16))
        head_hidden = int(config.get("head_hidden", 32))
        gate_hidden = int(config.get("gate_hidden", 32))
        attention_hidden = int(config.get("temporal_attention_hidden", hidden))
        if min(hidden, gat_hidden, tcn_hidden, self.msg_dim, head_hidden, gate_hidden, attention_hidden) <= 0:
            raise ValueError("All neural dimensions must be positive.")
        slope = float(config.get("negative_slope", 0.2))
        if not math.isfinite(slope) or slope < 0:
            raise ValueError("negative_slope must be finite and nonnegative.")
        self.max_lag = int(config.get("max_lag", data.protocol.get("candidate_graph", {}).get("tau_max", 4)))
        if self.max_lag < 1:
            raise ValueError("max_lag must be positive.")
        src = np.asarray(data.graph["src"], dtype=np.int64)
        dst = np.asarray(data.graph["dst"], dtype=np.int64)
        if src.ndim != 1 or src.shape != dst.shape:
            raise ValueError("Candidate src/dst must be one-dimensional equal arrays.")
        if (np.any(src < 0) or np.any(dst < 0) or np.any(src >= self.n_nodes)
                or np.any(dst >= self.n_nodes) or np.any(src == dst)):
            raise ValueError("Candidate graph requires in-range non-self directed edges.")
        if len(set(zip(src.tolist(), dst.tolist()))) != len(src):
            raise ValueError("Candidate graph contains duplicate directed edges.")
        self.register_buffer("src", torch.as_tensor(src, dtype=torch.long))
        self.register_buffer("dst", torch.as_tensor(dst, dtype=torch.long))
        degree = np.maximum(1, np.bincount(dst, minlength=self.n_nodes))
        self.register_buffer("degree", torch.as_tensor(degree, dtype=torch.float32))
        lookup = {(int(i), int(j)): e for e, (i, j) in enumerate(zip(src, dst))}
        pairs = [(e, lookup[(int(j), int(i))]) for e, (i, j) in enumerate(zip(src, dst))
                 if (int(j), int(i)) in lookup and e < lookup[(int(j), int(i))]]
        self.register_buffer("pair_a", torch.as_tensor([a for a, _ in pairs], dtype=torch.long))
        self.register_buffer("pair_b", torch.as_tensor([b for _, b in pairs], dtype=torch.long))
        self.static_in_score = self.spec["static_in_score"]
        self.static_in_gat = self.spec["static_in_gat"]
        self.dynamic_in_score = self.spec["dynamic_in_score"]
        self.temporal_pooling = self.spec["temporal_pooling"]
        self.has_pre_gat = self.spec["has_pre_gat"]
        self.score_kind = self.spec["score_kind"]
        edge_features = None
        if self.static_in_score or self.static_in_gat:
            edge_features = np.asarray(data.graph.get("edge_feats"), dtype=np.float32)
            if edge_features.shape != (len(src), 8) or not np.isfinite(edge_features).all():
                raise ValueError(f"{name} requires finite training-only edge features [E,8].")
            self.register_buffer("edge_features", torch.as_tensor(edge_features))
        if self.has_pre_gat:
            self.gat = DirectedGAT(self.spatial_input_features, gat_hidden, self.n_nodes, src, dst,
                                   edge_features if self.static_in_gat else None, slope)
        else:
            # No neighborhood mixing, but retain the GAT output width and
            # pointwise nonlinearity for the otherwise unchanged temporal path.
            self.gat = nn.Sequential(nn.Linear(self.spatial_input_features, gat_hidden, bias=False),
                                     nn.ELU())
        self.encoder_kind = self.spec["encoder_kind"]
        if self.encoder_kind == "mlp":
            self.encoder = nn.Sequential(nn.Linear(self.window * gat_hidden, hidden), nn.ReLU(),
                                         nn.Linear(hidden, hidden), nn.ReLU())
            self.representation_projection = nn.Identity()
        else:
            if self.encoder_kind == "tcn":
                self.encoder = FullSequenceCausalTCN(gat_hidden, tcn_hidden,
                    kernel_size=int(config.get("tcn_kernel_size", 3)),
                    dilations=config.get("tcn_dilations", [1, 2, 4]))
                if self.encoder.receptive_field < self.window:
                    raise ValueError("TCN receptive field must cover W for every TCN model.")
            else:
                # Operates independently at each historical position; attention
                # still pools across W, so this is no convolution, not no time.
                self.encoder = nn.Sequential(nn.Linear(gat_hidden, tcn_hidden), nn.ReLU())
            self.representation_projection = (nn.Identity() if tcn_hidden == hidden
                                               else nn.Linear(tcn_hidden, hidden))
            if self.temporal_pooling == "attention":
                self.temporal_attention = nn.Sequential(nn.Linear(tcn_hidden, attention_hidden),
                                                        nn.Tanh(), nn.Linear(attention_hidden, 1))
        gate_inputs = 4 * hidden + (8 if self.static_in_score else 0) + (6 if self.dynamic_in_score else 0)
        self.gate = None
        if self.score_kind == "dynamic":
            self.gate = nn.Sequential(nn.Linear(gate_inputs, gate_hidden), nn.ReLU(), nn.Linear(gate_hidden, 1))
        elif self.score_kind == "static":
            self.gate_logits = nn.Parameter(torch.zeros(len(src)))
        self.message = None if self.score_kind == "none" else nn.Linear(hidden, self.msg_dim, bias=False)
        self.head = nn.Sequential(nn.Linear(hidden + self.msg_dim, head_hidden), nn.ReLU(),
                                  nn.Linear(head_hidden, self.horizon))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def temporal_sequence(self, history):
        if self.encoder_kind == "mlp":
            raise ValueError("M1 uses a history MLP and has no TCN sequence.")
        return self.encoder(self.gat(history[..., :self.spatial_input_features]))

    def encode(self, history, return_attention=False):
        if history.ndim != 4 or tuple(history.shape[1:]) != (self.window, self.n_nodes, self.input_features):
            raise ValueError("History shape differs from the configured W, N, or feature channels.")
        attention = None
        if self.encoder_kind == "mlp":
            spatial = self.gat(history[..., :self.spatial_input_features])
            batch, window, nodes, channels = spatial.shape
            representation = self.encoder(spatial.permute(0, 2, 1, 3).reshape(batch, nodes, window * channels))
        else:
            sequence = self.temporal_sequence(history)
            if self.temporal_pooling == "attention":
                attention = torch.softmax(self.temporal_attention(sequence).squeeze(-1), dim=1)
                representation = (sequence * attention.unsqueeze(-1)).sum(dim=1)
            else:
                representation = sequence[:, -1]
            representation = self.representation_projection(representation)
        return (representation, attention) if return_attention else representation

    def raw_scores(self, representation, history=None):
        if self.score_kind == "none":
            return None
        if self.score_kind == "fixed":
            return representation.new_ones((len(representation), len(self.src)))
        if self.score_kind == "static":
            return torch.sigmoid(self.gate_logits).unsqueeze(0).expand(len(representation), -1)
        zi, zj = representation[:, self.src], representation[:, self.dst]
        pieces = [zi, zj, zi - zj, zi * zj]
        if self.static_in_score:
            pieces.append(self.edge_features.unsqueeze(0).expand(len(representation), -1, -1))
        if self.dynamic_in_score:
            if history is None:
                raise ValueError(f"{self.name} scores also require their full history window.")
            pieces.append(dynamic_edge_features(history, self.src, self.dst, self.max_lag))
        return torch.sigmoid(self.gate(torch.cat(pieces, dim=-1))).squeeze(-1)

    def score_history(self, history):
        return self.raw_scores(self.encode(history), history)

    def forward(self, history, last, edge_mask=None):
        representation, temporal_attention = self.encode(history, return_attention=True)
        scores = self.raw_scores(representation, history)
        aggregation = representation.new_zeros((len(history), self.n_nodes, self.msg_dim))
        effective = None
        if scores is not None:
            masked = scores if edge_mask is None else scores * edge_mask
            effective = masked / self.degree[self.dst].unsqueeze(0)
            messages = self.message(representation)[:, self.src] * effective.unsqueeze(-1)
            aggregation.index_add_(1, self.dst, messages)
        residual = self.head(torch.cat([representation, aggregation], dim=-1))
        return {"prediction": last.unsqueeze(-1) + residual, "edge_scores": scores,
                "effective_weight": effective, "aggregation": aggregation,
                "representation": representation, "temporal_attention": temporal_attention}


class DeigTCNForecaster(GatedForecaster):
    """Wrapper compatible with data batches, training, saving and intervention."""

    def __init__(self, name, data, config=None, seed=0):
        name = str(name).lower()
        spec = resolve_model_spec(name)
        self.name, self.config, self.seed = name, dict(config or {}), int(seed)
        valid_keys = {"encoder_hidden", "gat_hidden", "tcn_hidden", "msg_dim", "head_hidden",
                      "gate_hidden", "tcn_kernel_size", "tcn_dilations", "temporal_attention_hidden",
                      "negative_slope", "max_lag"}
        unknown = set(self.config) - valid_keys
        if unknown:
            raise ValueError(f"Unknown construction model parameters: {sorted(unknown)}")
        self.device = torch.device("cpu")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.module = DeigTCNModule(name, data, self.config).to(self.device)
        has_message = spec["score_kind"] != "none"
        intervention_scope = "post_message_only" if has_message else None
        self.capabilities = {"supports_ranking": spec["score_kind"] in {"static", "dynamic"},
                             "supports_intervention": has_message,
                             "score_kind": spec["score_kind"], "intervention_scope": intervention_scope,
                             "has_pre_gat": self.module.has_pre_gat, "gat_self_path": self.module.has_pre_gat,
                             "message_normalization": "fixed_candidate_indegree"}
        self.parameter_count = sum(parameter.numel() for parameter in self.module.parameters())
        self.construction_metadata = {
            "version": name.upper(), "input_features": self.module.input_features,
            "feature_channels": list(FULL_FEATURE_CHANNELS if self.module.input_features == 7 else ["level"]),
            "raw_history_features": self.module.input_features,
            "raw_history_feature_channels": list(FULL_FEATURE_CHANNELS if self.module.input_features == 7 else ["level"]),
            "spatial_input_features": self.module.spatial_input_features,
            "spatial_feature_channels": list(FULL_FEATURE_CHANNELS if self.module.spatial_input_features == 7 else ["level"]),
            "parent_model": spec["parent_model"], "ablation": spec["ablation"],
            "pre_spatial": ("node_local_projection" if not self.module.has_pre_gat else
                            "edge_aware_gat" if self.module.static_in_gat else "ordinary_gat"),
            "temporal_encoder": self.module.encoder_kind,
            "temporal_pooling": self.module.temporal_pooling,
            "tcn_receptive_field": getattr(self.module.encoder, "receptive_field", None),
            "static_edge_in_score": self.module.static_in_score,
            "static_edge_in_gat": self.module.static_in_gat,
            "dynamic_edge_in_score": self.module.dynamic_in_score,
            "intent_score_kind": spec["score_kind"],
            "has_intent_message": has_message,
            "prediction_head_input": "representation_concat_message_or_zero",
            "message_normalization": "fixed_candidate_indegree",
            "message_normalization_active": has_message,
            "intervention_scope": intervention_scope,
            "dynamic_features_method": "history_residual_max_positive_lag_pearson_no_shrinkage"
                if self.module.dynamic_in_score else None,
        }

    def representation_method(self, history):
        history = torch.as_tensor(history, dtype=torch.float32, device=self.device)
        return self.module.encode(history)

    def score_history(self, history):
        history = torch.as_tensor(history, dtype=torch.float32, device=self.device)
        return self.module.score_history(history)
