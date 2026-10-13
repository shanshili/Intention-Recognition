"""Read-only architecture probes; synthetic inputs, no training/checkpoint edits."""
from pathlib import Path
import importlib
import json
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

Config = importlib.import_module("260731.config").Config
RATCGF = importlib.import_module("260731.model").RATCGF

torch.set_num_threads(1)
torch.manual_seed(20260928)
cfg = Config()
cfg._in_dim = 3
model = RATCGF(cfg).eval()
n = 6

def graph():
    a = torch.rand(n, n)
    a[a < .4] = 0
    a.fill_diagonal_(0)
    return a

sample = {
    "Xseq": torch.randn(cfg.model.L, n, cfg._in_dim),
    "A_C": graph(), "A_I_prev": graph(),
    "A_I_cur": graph(), "A_I_next": graph(),
    "target": torch.randn(n),
}

def gradient_summary():
    groups = {
        "encoder": model.encoder,
        "global_gate": model.gating.mlp_g,
        "node_gate": model.gating.mlp_node,
        "edge_gate": model.gating.mlp_edge,
        "node_score_head": model.score_head,
        "predictor": model.predictor,
    }
    return {
        name: {
            "parameter_tensors": len(list(module.parameters())),
            "tensors_with_gradient": sum(p.grad is not None for p in module.parameters()),
            "gradient_l1": sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None),
        } for name, module in groups.items()
    }

out = model(sample)
F.mse_loss(out["y_hat"], sample["target"]).backward()
pred_grads = gradient_summary()
model.zero_grad(set_to_none=True)
out = model(sample)
(F.mse_loss(out["y_hat"], sample["target"])
 + cfg.train.sparsity_lambda * out["S_D"].abs().mean()).backward()
total_grads = gradient_summary()

with torch.no_grad():
    before = model(sample)
    for p in model.gating.mlp_edge.parameters():
        p.add_(torch.randn_like(p) * 2)
    after = model(sample)

result = {
    "scope": "Synthetic forward/backward architecture diagnostic, not a performance experiment",
    "torch_version": torch.__version__,
    "prediction_loss_gradients": pred_grads,
    "total_loss_gradients": total_grads,
    "edge_gate_perturbation": {
        "max_prediction_change": float((before["y_hat"] - after["y_hat"]).abs().max()),
        "max_fused_edge_change": float((before["S_D"] - after["S_D"]).abs().max()),
    },
}
path = Path(__file__).with_name("probe_result.json")
path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
