"""
Unit tests for model components:
GATLayer, SpatialEncoder, BiGRUTemporal, TemporalConvNet, IntentModel, AttentionPooling
"""
import torch
import pytest


# ============ GATLayer Tests ============
class TestGATLayer:
    def test_output_shape(self, mod_260604, sample_adj_5nodes):
        layer = mod_260604.GATLayer(in_dim=8, out_dim=16, dropout=0.0)
        h = torch.randn(2, 5, 8)  # (batch=2, nodes=5, features=8)
        out = layer(h, sample_adj_5nodes)
        assert out.shape == (2, 5, 16)

    def test_single_sample(self, mod_260604, sample_adj_5nodes):
        layer = mod_260604.GATLayer(in_dim=4, out_dim=8, dropout=0.0)
        h = torch.randn(1, 5, 4)
        out = layer(h, sample_adj_5nodes)
        assert out.shape == (1, 5, 8)

    def test_deterministic_without_dropout(self, mod_260604, sample_adj_5nodes):
        layer = mod_260604.GATLayer(in_dim=4, out_dim=8, dropout=0.0)
        layer.eval()
        h = torch.randn(1, 5, 4)
        out1 = layer(h, sample_adj_5nodes)
        out2 = layer(h, sample_adj_5nodes)
        torch.testing.assert_close(out1, out2)

    def test_gradient_flow(self, mod_260604, sample_adj_5nodes):
        layer = mod_260604.GATLayer(in_dim=4, out_dim=8, dropout=0.0)
        h = torch.randn(1, 5, 4, requires_grad=True)
        out = layer(h, sample_adj_5nodes)
        loss = out.sum()
        loss.backward()
        assert h.grad is not None
        assert h.grad.shape == (1, 5, 4)


# ============ SpatialEncoder Tests ============
class TestSpatialEncoder:
    def test_output_shape_260601(self, mod_260601, sample_adj_5nodes):
        """260601 SpatialEncoder uses mean pooling (no target_idx)."""
        encoder = mod_260601.SpatialEncoder(in_dim=1, hidden_dim=16, out_dim=8)
        x = torch.randn(2, 5, 1)  # (batch, nodes, features)
        out = encoder(x, sample_adj_5nodes)
        assert out.shape == (2, 8)

    def test_output_shape_260604(self, mod_260604, sample_adj_5nodes):
        """260604 SpatialEncoder uses target+global concatenation."""
        encoder = mod_260604.SpatialEncoder(
            in_dim=1, hidden_dim=16, out_dim=8, target_idx=0
        )
        x = torch.randn(2, 5, 1)
        out = encoder(x, sample_adj_5nodes)
        assert out.shape == (2, 8)

    def test_different_target_idx(self, mod_260604, sample_adj_5nodes):
        encoder = mod_260604.SpatialEncoder(
            in_dim=1, hidden_dim=16, out_dim=8, target_idx=3
        )
        x = torch.randn(2, 5, 1)
        out = encoder(x, sample_adj_5nodes)
        assert out.shape == (2, 8)


# ============ BiGRUTemporal Tests ============
class TestBiGRUTemporal:
    def test_output_shape(self, mod_260604):
        bigru = mod_260604.BiGRUTemporal(
            input_dim=32, hidden_dim=64, output_dim=16, num_layers=2, dropout=0.0
        )
        seq = torch.randn(4, 24, 32)  # (batch, time_steps, features)
        out = bigru(seq)
        assert out.shape == (4, 16)

    def test_single_step(self, mod_260604):
        bigru = mod_260604.BiGRUTemporal(
            input_dim=8, hidden_dim=16, output_dim=4, num_layers=1, dropout=0.0
        )
        seq = torch.randn(2, 1, 8)  # single time step
        out = bigru(seq)
        assert out.shape == (2, 4)

    def test_gradient_flow(self, mod_260604):
        bigru = mod_260604.BiGRUTemporal(
            input_dim=8, hidden_dim=16, output_dim=4, num_layers=1, dropout=0.0
        )
        seq = torch.randn(2, 10, 8, requires_grad=True)
        out = bigru(seq)
        loss = out.sum()
        loss.backward()
        assert seq.grad is not None


# ============ TemporalConvNet Tests ============
class TestTemporalConvNet:
    def test_output_shape_260601(self, mod_260601):
        """260601 TCN does mean pooling over time -> (B, out_channels)."""
        tcn = mod_260601.TemporalConvNet(input_dim=32, num_channels=[64, 128], kernel_size=2)
        x = torch.randn(4, 24, 32)  # (batch, time, features)
        out = tcn(x)
        assert out.shape == (4, 128)

    def test_output_shape_260604_tcn(self, mod_260604_tcn):
        """260604_TCN version returns full time sequence (B, T, out_channels)."""
        tcn = mod_260604_tcn.TemporalConvNet(input_dim=32, num_channels=[64, 128], kernel_size=3)
        x = torch.randn(4, 24, 32)
        out = tcn(x)
        assert out.shape[0] == 4
        assert out.shape[2] == 128
        # Time dimension might differ due to padding

    def test_single_channel(self, mod_260601):
        tcn = mod_260601.TemporalConvNet(input_dim=16, num_channels=[32], kernel_size=2)
        x = torch.randn(2, 10, 16)
        out = tcn(x)
        assert out.shape == (2, 32)


# ============ AttentionPooling Tests (260604_TCN only) ============
class TestAttentionPooling:
    def test_output_shape(self, mod_260604_tcn):
        pool = mod_260604_tcn.AttentionPooling(feat_dim=64)
        x = torch.randn(4, 24, 64)  # (batch, time, features)
        out = pool(x)
        assert out.shape == (4, 64)

    def test_single_timestep(self, mod_260604_tcn):
        pool = mod_260604_tcn.AttentionPooling(feat_dim=32)
        x = torch.randn(2, 1, 32)
        out = pool(x)
        assert out.shape == (2, 32)
        # With single timestep, output should equal input (softmax=1)
        torch.testing.assert_close(out, x.squeeze(1), atol=1e-5, rtol=1e-5)

    def test_gradient_flow(self, mod_260604_tcn):
        pool = mod_260604_tcn.AttentionPooling(feat_dim=16)
        x = torch.randn(2, 10, 16, requires_grad=True)
        out = pool(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


# ============ IntentModel Tests ============
class TestIntentModel:
    def test_forward_bigru(self, mod_260604, sample_adj_5nodes):
        model = mod_260604.IntentModel(
            N=5, node_feat_dim=1, spat_hid=16, spat_out=8,
            temp_hid=16, intent_dim=8, num_classes=3,
            pred_len=6, use_tcn=False, target_idx=0
        )
        x = torch.randn(2, 5, 24)  # (batch, nodes, window)
        intent_vec, logits, pred_flow = model(x, sample_adj_5nodes)
        assert intent_vec.shape == (2, 8)
        assert logits.shape == (2, 3)
        assert pred_flow.shape == (2, 6)

    def test_forward_tcn_260604(self, mod_260604, sample_adj_5nodes):
        model = mod_260604.IntentModel(
            N=5, node_feat_dim=1, spat_hid=16, spat_out=8,
            temp_hid=16, intent_dim=8, num_classes=3,
            pred_len=6, use_tcn=True, tcn_channels=(16, 32), target_idx=0
        )
        x = torch.randn(2, 5, 24)
        intent_vec, logits, pred_flow = model(x, sample_adj_5nodes)
        assert intent_vec.shape == (2, 8)
        assert logits.shape == (2, 3)
        assert pred_flow.shape == (2, 6)

    def test_forward_tcn_260604_tcn(self, mod_260604_tcn, sample_adj_5nodes):
        """The 260604_TCN version uses AttentionPooling."""
        model = mod_260604_tcn.IntentModel(
            N=5, node_feat_dim=1, spat_hid=16, spat_out=8,
            temp_hid=16, intent_dim=8, num_classes=3,
            pred_len=6, use_tcn=True, tcn_channels=(16, 32, 64), target_idx=0
        )
        x = torch.randn(2, 5, 24)
        intent_vec, logits, pred_flow = model(x, sample_adj_5nodes)
        assert intent_vec.shape == (2, 8)
        assert logits.shape == (2, 3)
        assert pred_flow.shape == (2, 6)

    def test_forward_260601_bigru(self, mod_260601, sample_adj_5nodes):
        model = mod_260601.IntentModel(
            N=5, node_feat_dim=1, spat_hid=16, spat_out=8,
            temp_hid=16, intent_dim=8, num_classes=3,
            pred_len=6, use_tcn=False
        )
        x = torch.randn(2, 5, 24)
        intent_vec, logits, pred_flow = model(x, sample_adj_5nodes)
        assert intent_vec.shape == (2, 8)
        assert logits.shape == (2, 3)
        assert pred_flow.shape == (2, 6)

    def test_gradient_flow(self, mod_260604, sample_adj_5nodes):
        model = mod_260604.IntentModel(
            N=5, node_feat_dim=1, spat_hid=16, spat_out=8,
            temp_hid=16, intent_dim=8, num_classes=3,
            pred_len=6, use_tcn=False, target_idx=0
        )
        x = torch.randn(2, 5, 24, requires_grad=True)
        intent_vec, logits, pred_flow = model(x, sample_adj_5nodes)
        loss = logits.sum() + pred_flow.sum()
        loss.backward()
        assert x.grad is not None

    def test_different_pred_lengths(self, mod_260604, sample_adj_5nodes):
        for pred_len in [1, 3, 12]:
            model = mod_260604.IntentModel(
                N=5, node_feat_dim=1, spat_hid=16, spat_out=8,
                temp_hid=16, intent_dim=8, num_classes=3,
                pred_len=pred_len, use_tcn=False, target_idx=0
            )
            x = torch.randn(1, 5, 24)
            _, _, pred_flow = model(x, sample_adj_5nodes)
            assert pred_flow.shape == (1, pred_len)

    def test_different_num_classes(self, mod_260604, sample_adj_5nodes):
        for num_classes in [2, 5, 10]:
            model = mod_260604.IntentModel(
                N=5, node_feat_dim=1, spat_hid=16, spat_out=8,
                temp_hid=16, intent_dim=8, num_classes=num_classes,
                pred_len=6, use_tcn=False, target_idx=0
            )
            x = torch.randn(1, 5, 24)
            _, logits, _ = model(x, sample_adj_5nodes)
            assert logits.shape == (1, num_classes)
