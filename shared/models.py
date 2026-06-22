"""
Shared model components: GATLayer, SpatialEncoder, BiGRUTemporal,
TemporalConvNet, AttentionPooling, IntentModel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2, alpha=0.2):
        super().__init__()
        self.W = nn.Parameter(torch.empty(in_dim, out_dim))
        self.a = nn.Parameter(torch.empty(2 * out_dim, 1))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)
        N = Wh.size(1)
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1)
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1)
        concat = torch.cat([Wh_i, Wh_j], dim=-1)
        e = self.leaky_relu(torch.matmul(concat, self.a).squeeze(-1))
        adj_expanded = adj.unsqueeze(0).expand(e.size(0), -1, -1)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_expanded > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        h_new = torch.matmul(attention, Wh)
        return h_new


class SpatialEncoder(nn.Module):
    """Spatial encoder with optional target-node aware pooling.

    When ``target_idx`` is None (260601 mode), output = Linear(mean_pool).
    Otherwise, output = Linear(concat(target_vec, mean_pool)).
    """

    def __init__(self, in_dim, hidden_dim, out_dim, target_idx=None,
                 n_layers=2, dropout=0.2):
        super().__init__()
        self.target_idx = target_idx
        self.layers = nn.ModuleList()
        self.layers.append(GATLayer(in_dim, hidden_dim, dropout))
        for _ in range(n_layers - 1):
            self.layers.append(GATLayer(hidden_dim, hidden_dim, dropout))
        proj_in = hidden_dim * 2 if target_idx is not None else hidden_dim
        self.out_proj = nn.Linear(proj_in, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        for layer in self.layers:
            x = F.elu(layer(x, adj))
            x = self.dropout(x)
        global_vec = x.mean(dim=1)
        if self.target_idx is not None:
            target_vec = x[:, self.target_idx, :]
            combined = torch.cat([target_vec, global_vec], dim=-1)
            return self.out_proj(combined)
        return self.out_proj(global_vec)


class BiGRUTemporal(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, seq):
        out, _ = self.gru(seq)
        last = out[:, -1, :]
        return self.fc(last)


class TemporalConvNet(nn.Module):
    """TCN with configurable pooling behaviour.

    Parameters
    ----------
    pool_output : bool
        If True (default, 260601/260604 behaviour), apply global mean pooling
        over time and return (B, out_dim).
        If False (260604_TCN behaviour), return full sequence (B, T, out_dim).
    drop_last : bool
        If False (260604_TCN behaviour), skip Dropout on the last conv block.
    """

    def __init__(self, input_dim, num_channels, kernel_size=2, dropout=0.2,
                 pool_output=True, drop_last=True):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i, out_ch in enumerate(num_channels):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size - 1))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            if drop_last or i < len(num_channels) - 1:
                layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.out_dim = num_channels[-1] if num_channels else input_dim
        self.pool_output = pool_output

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.tcn(x)
        if self.pool_output:
            return out.mean(dim=-1)
        return out.transpose(1, 2)


class AttentionPooling(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.context = nn.Parameter(torch.randn(feat_dim, 1))
        nn.init.xavier_uniform_(self.context)

    def forward(self, x):
        scores = torch.matmul(x, self.context).squeeze(-1)
        attn_weights = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)
        return pooled


class IntentModel(nn.Module):
    """Unified intent recognition model.

    Parameters
    ----------
    use_tcn : bool
        Use TCN temporal encoder instead of BiGRU.
    use_attention_pooling : bool
        Use AttentionPooling after TCN (260604_TCN mode). Only relevant
        when ``use_tcn=True``.
    target_idx : int or None
        Target node index for spatial encoder. None = global-pool only (260601).
    predictor_layers : int
        2 for simple predictor (260601), 3 for deeper predictor (260604+).
    """

    def __init__(self, N, node_feat_dim, spat_hid, spat_out, temp_hid,
                 intent_dim, num_classes=3, pred_len=6, use_tcn=False,
                 tcn_channels=(64, 128), target_idx=None,
                 use_attention_pooling=False, predictor_layers=2,
                 dropout=0.2):
        super().__init__()
        self.spatial_encoder = SpatialEncoder(
            node_feat_dim, spat_hid, spat_out, target_idx=target_idx
        )
        self.intent_dim = intent_dim
        self.pred_len = pred_len
        self.use_tcn = use_tcn
        self.use_attention_pooling = use_attention_pooling

        if use_tcn:
            pool_output = not use_attention_pooling
            drop_last = not use_attention_pooling
            self.temporal = TemporalConvNet(
                spat_out, tcn_channels, kernel_size=3, dropout=dropout,
                pool_output=pool_output, drop_last=drop_last,
            )
            temp_out = tcn_channels[-1]
            if use_attention_pooling:
                self.pooling = AttentionPooling(temp_out)
        else:
            self.temporal = BiGRUTemporal(spat_out, temp_hid, intent_dim)
            temp_out = intent_dim

        self.intent_proj = nn.Linear(temp_out, intent_dim)
        self.classifier = nn.Linear(intent_dim, num_classes)

        if predictor_layers == 3:
            self.predictor = nn.Sequential(
                nn.Linear(intent_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, pred_len),
            )
        else:
            self.predictor = nn.Sequential(
                nn.Linear(intent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, pred_len),
            )

    def forward(self, x, adj):
        batch, N, W = x.shape
        spatial_seq = []
        for t in range(W):
            x_t = x[:, :, t].unsqueeze(-1)
            g_t = self.spatial_encoder(x_t, adj)
            spatial_seq.append(g_t.unsqueeze(1))
        spatial_seq = torch.cat(spatial_seq, dim=1)
        temp_out = self.temporal(spatial_seq)
        if self.use_tcn and self.use_attention_pooling:
            temp_out = self.pooling(temp_out)
        intent_vec = self.intent_proj(temp_out)
        logits = self.classifier(intent_vec)
        pred_flow = self.predictor(intent_vec)
        return intent_vec, logits, pred_flow
