"""Small formula checks; no training or edits to algorithm modules."""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "deig_features_readonly", ROOT / "260628/deig/features.py"
)
features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features)

rng = np.random.default_rng(20260929)
x, y = rng.normal(size=(11, 5)), rng.normal(size=(11, 5))
A, B, C = [rng.normal(size=(5, 7)) for _ in range(3)]
bias = rng.normal(size=7)
with_difference = np.maximum(x @ A + y @ B + (x-y) @ C + bias, 0)
folded = np.maximum(x @ (A+C) + y @ (B-C) + bias, 0)

scores = np.array([-2., .4, 1.3])
def attention(shift):
    v = scores + shift
    v = np.where(v >= 0, v, .2*v)
    p = np.exp(v-v.max())
    return (p / p.sum()).tolist()

u = np.array([2., 1., 4., 3.])
report = {
    "scope": "Algebra and implementation checks, not forecasting performance",
    "W3_dynamic_lag_corr": float(features._windowed_lag_corr(
        np.array([1., 3., 2.]), np.array([0., 2., 1.]))),
    "difference_feature_fold_max_error": float(np.max(np.abs(with_difference-folded))),
    "shared_calendar_logit_shift": {
        "base_logits": scores.tolist(),
        "weights_shift_0": attention(0.),
        "weights_shift_3": attention(3.),
        "weights_shift_4": attention(4.),
        "ranking_all_cases": np.argsort(scores).tolist(),
    },
    "trend_window_mismatch": {
        "sequence_including_one_previous_step": u.tolist(),
        "W": 3,
        "note_internal_differences": float(np.diff(u[-3:]).mean()),
        "code_window_of_precomputed_differences": float(np.diff(u).mean()),
    },
}
out = Path(__file__).with_name("formula_checks.json")
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))

