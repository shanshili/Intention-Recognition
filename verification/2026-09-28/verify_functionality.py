"""Reproducible functional audit; never edits application code or old outputs.

Run from any directory with the project's Python interpreter. Synthetic fixtures
and all new outputs stay beside this file. No old checkpoints are deserialized.
PASS/FAIL compare implementation behavior with the stated functional contract;
an audit process that completes may deliberately contain FAIL findings.
"""
from pathlib import Path
import ast
import contextlib
import hashlib
import importlib
import io
import json
import os
import subprocess
import sys
import traceback
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
torch.manual_seed(20260928)
np.random.seed(20260928)
RESULT = {"scope": "Functional audit with small synthetic fixtures, not performance validation",
          "environment": {"python": sys.version, "torch": torch.__version__,
                          "numpy": np.__version__, "cuda": torch.cuda.is_available()},
          "checks": [], "evidence": {}}


def check(name, passed, detail):
    RESULT["checks"].append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})
    print(f"[{RESULT['checks'][-1]['status']}] {name}", flush=True)


def section(fn):
    try:
        fn()
    except Exception:
        detail = traceback.format_exc()
        check(fn.__name__, False, detail)
    (OUT / "results.json").write_text(json.dumps(RESULT, ensure_ascii=False, indent=2), encoding="utf-8")


def mod(name):
    return importlib.import_module(name)


Config = mod("260731.config").Config
loader = mod("260731.utils.data_loading")
Builder = mod("260731.dataset").SampleBuilder
RATCGF = mod("260731.model").RATCGF
deploy_module = mod("260731.modules.module4_deploy")
main_module = mod("260731.main")


def small_config():
    cfg = Config()
    cfg.data.data_dir = str(OUT / "deliberately_missing_inputs")
    cfg.data.synth_num_nodes = 6
    cfg.data.synth_num_steps = 48
    cfg.data.synth_n_show = 12
    cfg.model.L = 3
    cfg.model.gnn_hidden = cfg.model.tcn_hidden = cfg.model.graph_emb = 8
    cfg.model.gnn_layers = cfg.model.tcn_layers = 1
    cfg.model.dropout = 0.0
    cfg.train.device = "cpu"
    cfg.train.epochs = 2
    cfg.train.batch_size = 4
    return cfg


def version_inventory():
    paths = [p for p in ROOT.glob("*.py")]
    versions = ["260627", "260628", "260706/ratcgf", "260706_2", "260731"]
    for version in versions:
        paths.extend(p for p in (ROOT / version).rglob("*.py") if "outputs" not in p.parts)
    failures = []
    for path in paths:
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except Exception as exc:
            failures.append({"file": str(path.relative_to(ROOT)), "error": str(exc)})
    check("all_version_python_syntax", not failures, {"files": len(paths), "failures": failures})
    compare = {}
    for left, right in zip(versions, versions[1:]):
        lp, rp = ROOT / left, ROOT / right
        common = sorted({p.relative_to(lp).as_posix() for p in lp.rglob("*.py") if "outputs" not in p.parts}
                        & {p.relative_to(rp).as_posix() for p in rp.rglob("*.py") if "outputs" not in p.parts})
        compare[f"{left} -> {right}"] = {
            "identical": [p for p in common if (lp / p).read_bytes() == (rp / p).read_bytes()],
            "changed": [p for p in common if (lp / p).read_bytes() != (rp / p).read_bytes()]}
    RESULT["evidence"]["version_comparison"] = compare
    RESULT["evidence"]["source_sha256"] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    help_results = []
    for version in versions:
        proc = subprocess.run([sys.executable, "-B", str(ROOT / version / "main.py"), "--help"],
                              cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=45)
        help_results.append({"version": version, "returncode": proc.returncode, "stderr": proc.stderr[-1500:]})
    check("version_main_help_imports", all(p["returncode"] == 0 for p in help_results), help_results)


def time_and_data_contract():
    cfg = Config()
    cfg.data.data_dir = str(OUT / "deliberately_missing_inputs")
    b = Builder(loader.load_all(cfg), cfg, torch.device("cpu"))
    sample = b.build(b.valid_centers[0])
    check("latest_time_alignment_contract", b._pos(0) == 143 and sample["pos"] == 144 and sample["target_pos"] == 145,
          {"first_intent_pos": b._pos(0), "first_sample_pos": sample["pos"], "target_pos": sample["target_pos"],
           "deployment_time": sample["time"], "target_time": sample["target_time"]})
    proc = subprocess.run([sys.executable, "-B", str(ROOT / "260731/tests/test_time_alignment.py")],
                          cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=30)
    check("existing_time_test_directly_runnable", proc.returncode == 0,
          {"returncode": proc.returncode, "stderr": proc.stderr[-1800:]})
    fixture = OUT / "sparse_intent_export.npz"
    # The exact metadata names exported by 260628/main.py, with non-unit stride.
    np.savez_compressed(fixture, edges_list=np.array([[[0, 1]]] * 3),
                        strength_list=np.ones((3, 1)), delta_list=np.zeros(3),
                        input_end_t=np.array([2, 5, 8]), target_start_t=np.array([3, 6, 9]),
                        intent_time_t=np.array([3, 6, 9]))
    intent = loader.load_intent(str(fixture))
    positions, meta = loader.resolve_intent_positions(intent, loader.build_time_axis(cfg, 12), cfg, 3)
    check("upstream_timestamp_metadata_preserved", np.array_equal(positions, [3, 6, 9]),
          {"upstream_intent_time_t": [3, 6, 9], "actual": positions.tolist(), "method": meta["method"],
           "loaded_keys": list(intent)})
    args = main_module.build_argparser().parse_args([])
    default_cfg = Config()
    default_cfg.data.data_dir = args.data_dir
    default_cfg.data.causal_pkl = args.causal_pkl
    default_cfg.data.intent_npz = args.intent_npz
    bundle = loader.load_all(default_cfg)
    builder = Builder(bundle, default_cfg, torch.device("cpu"))
    centers = main_module._filter_centers_by_time(builder, builder.valid_centers, args.deploy_start_time, args.deploy_end_time)
    check("default_configuration_has_deployable_inputs", bool(centers),
          {"data_dir": args.data_dir, "path_exists": Path(args.data_dir).exists(),
           "used_synthetic": bundle["used_synthetic"], "selected_centers": len(centers),
           "time_range": [args.deploy_start_time, args.deploy_end_time]})
    cfg.data.allow_synthetic = False
    try:
        loader.load_all(cfg)
        strict_ok = False
    except FileNotFoundError:
        strict_ok = True
    check("strict_data_rejects_missing_inputs", strict_ok, "Strict mode rejects missing flow/graph files.")
    check("time_selection_modes", main_module._select_centers(range(10), 3, 2, "latest") == [4, 6, 8]
          and main_module._select_centers(range(10), 3, 2, "earliest") == [0, 2, 4]
          and main_module._select_centers(range(10), 3, 2, "uniform") == [0, 4, 8],
          "stride then latest/earliest/uniform selection")


def model_contract():
    cfg = small_config()
    cfg._in_dim = 3
    torch.manual_seed(20260928)
    model = RATCGF(cfg).eval()
    n = 6
    def graph():
        a = torch.rand(n, n)
        a[a < .4] = 0
        a.fill_diagonal_(0)
        return a
    sample = {"Xseq": torch.randn(cfg.model.L, n, 3), "A_C": graph(),
              "A_I_prev": graph(), "A_I_cur": graph(), "A_I_next": graph(), "target": torch.randn(n)}
    groups = {"encoder": model.encoder, "global_gate": model.gating.mlp_g,
              "node_gate": model.gating.mlp_node, "edge_gate": model.gating.mlp_edge,
              "score_head": model.score_head, "predictor": model.predictor}
    def grads():
        return {name: {"with_grad": sum(p.grad is not None for p in module.parameters()),
                       "l1": sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)}
                for name, module in groups.items()}
    out = model(sample)
    F.mse_loss(out["y_hat"], sample["target"]).backward()
    pred_grads = grads()
    model.zero_grad(set_to_none=True)
    out = model(sample)
    (F.mse_loss(out["y_hat"], sample["target"]) + cfg.train.sparsity_lambda * out["S_D"].abs().mean()).backward()
    total_grads = grads()
    RESULT["evidence"]["fusion_gradients"] = {"prediction_loss": pred_grads, "total_loss": total_grads}
    check("fusion_encoder_receives_prediction_gradient", pred_grads["encoder"]["l1"] > 0, pred_grads["encoder"])
    check("local_edge_gates_receive_prediction_gradient", pred_grads["node_gate"]["l1"] > 0 and pred_grads["edge_gate"]["l1"] > 0,
          {k: pred_grads[k] for k in ["node_gate", "edge_gate"]})
    check("node_score_head_is_trained", total_grads["score_head"]["l1"] > 0, total_grads["score_head"])
    with torch.no_grad():
        before = model(sample)
        for p in model.gating.mlp_edge.parameters():
            p.add_(torch.randn_like(p) * 2)
        after = model(sample)
    delta_y = float((before["y_hat"] - after["y_hat"]).abs().max())
    delta_s = float((before["S_D"] - after["S_D"]).abs().max())
    check("fused_edge_changes_affect_prediction", delta_y > 1e-8,
          {"max_prediction_change": delta_y, "max_fused_edge_change": delta_s})
    changed = dict(sample, A_I_next=torch.zeros_like(sample["A_I_next"]))
    with torch.no_grad():
        changed_out = model(changed)
    RESULT["evidence"]["next_intent_dependency"] = {
        "max_prediction_change": float((after["y_hat"] - changed_out["y_hat"]).abs().max()),
        "max_fused_edge_change": float((after["S_D"] - changed_out["S_D"]).abs().max()),
        "interpretation": "Design explicitly uses t+r. Online availability depends on intent timestamp semantics and upstream training cutoffs."}
    histories = []
    handle = model.encoder.encoders["I"].gnn.register_forward_pre_hook(lambda module, inputs: histories.append(inputs[1].detach().clone()))
    with torch.no_grad():
        model(sample)
    handle.remove()
    check("encoder_uses_time_varying_view_sequence", any(not torch.equal(histories[0], a) for a in histories[1:]),
          {"time_steps": len(histories), "identical_adjacency_at_all_steps": all(torch.equal(histories[0], a) for a in histories)})
    # Capture the actual alpha emitted by the gate under conflicting finite logits.
    captured = []
    h = model.gating.register_forward_hook(lambda module, inputs, output: captured.append(output))
    with torch.no_grad():
        for module in [model.gating.mlp_g, model.gating.mlp_edge]:
            for p in module.parameters():
                p.zero_()
        model.gating.mlp_g[-1].bias[0] = 80
        model.gating.mlp_edge[-1].bias[1] = 80
        model(sample)
    h.remove()
    alpha_sums = captured[-1]["alpha"].sum(1).detach().numpy()
    check("joint_gate_stays_normalized_under_saturation", bool(np.allclose(alpha_sums, 1.0, atol=1e-5)),
          {"min_alpha_sum": float(alpha_sums.min()), "max_alpha_sum": float(alpha_sums.max()), "logit_gap": 80})
    # Directed data contract is A[src,dst]; incoming aggregation must affect dst.
    enc = mod("260731.modules.module2_encoder")
    sage = enc.DenseSAGE(1, 1)
    with torch.no_grad():
        sage.lin_self.weight.zero_(); sage.lin_self.bias.zero_()
        sage.lin_neigh.weight.fill_(1); sage.lin_neigh.bias.zero_()
    adjacency = torch.tensor([[0., 1.], [0., 0.]])
    x = torch.tensor([[5.], [2.]])
    actual = sage(x, adjacency).detach().numpy().reshape(-1).tolist()
    check("directed_message_flows_source_to_destination", np.allclose(actual, [0, 5]),
          {"edge": "0 -> 1", "node_features": [5, 2], "expected_incoming_neighbor_message": [0, 5], "actual": actual})


def deployment_contract():
    rng = np.random.default_rng(41)
    failures = []
    for case in range(200):
        n = int(rng.integers(2, 12))
        s = rng.random((n, n)); s[s < .65] = 0; np.fill_diagonal(s, 0)
        cfg = Config().deploy
        cfg.budget = float(rng.uniform(.01, 25)); cfg.node_cost = float(rng.uniform(.2, 2))
        cfg.edge_cost = float(rng.uniform(0, 1)); cfg.top_k_nodes = int(rng.integers(1, n + 1))
        cfg.max_edges = None if case % 2 else int(rng.integers(0, n * n))
        dep = deploy_module.greedy_deploy(s, rng.random(n), s, s, cfg)
        endpoints = {k for i, j, _ in dep["edges"] for k in [i, j]}
        ok = (dep["cost"] <= cfg.budget + 1e-8 and set(dep["nodes"]) == endpoints
              and len(dep["nodes"]) <= cfg.top_k_nodes
              and abs(dep["cost"] - (len(dep["nodes"]) * cfg.node_cost + len(dep["edges"]) * cfg.edge_cost)) < 1e-8
              and (cfg.max_edges is None or len(dep["edges"]) <= cfg.max_edges))
        if not ok: failures.append(case)
    check("budget_endpoint_and_cost_invariants", not failures, {"random_cases": 200, "failures": failures})
    cfg = Config().deploy
    cfg.budget = 4.4
    s = np.array([[0, .9, 0], [0, 0, .8], [.7, 0, 0]])
    before = deploy_module.greedy_deploy(s, np.ones(3), s, s, cfg)
    cfg.lambda_el = 1000; cfg.lambda_tr = 1000; cfg.lambda_c = 1000
    after = deploy_module.greedy_deploy(s, np.ones(3), s, s, cfg)
    RESULT["evidence"]["deployment_utility_weights"] = {"unchanged_selection": before["edges"] == after["edges"],
        "interpretation": "Greedy function never reads lambda_el/lambda_tr/lambda_c; utilities are reported after selection."}
    # Equal scores, different topology: bridge vulnerability differs, but utility is identical.
    a = np.zeros((5, 5)); b = np.zeros((5, 5))
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4)]: a[i, j] = 1
    for i, j in [(0, 1), (1, 2), (2, 0), (0, 2)]: b[i, j] = 1
    RESULT["evidence"]["elastic_utility_proxy"] = {"chain": main_module.elastic_utility(torch.tensor(a), cfg),
        "cycle": main_module.elastic_utility(torch.tensor(b), cfg),
        "interpretation": "Score retention, not prediction loss under failures."}


def upstream_model_contract():
    dm = mod("260628.deig.model")
    auto = mod("260628.deig.autograd")
    features = mod("260628.deig.features")
    rng = np.random.default_rng(12)
    n, w = 4, 3
    edges = np.array([(i, j) for i in range(n) for j in range(n) if i != j])
    e = len(edges)
    cand = {"edges": edges, "src": edges[:, 0], "dst": edges[:, 1], "E": e}
    windows = [{"F_seq": rng.normal(size=(w, n, 7)), "edge_block": rng.normal(size=(e, 14)), "z_last": rng.normal(size=n)} for _ in range(2)]
    targets = [rng.normal(size=(n, 1)) for _ in windows]
    batch = dm.assemble_batch(windows, targets, cand, dm.compute_reverse_index(cand))
    model = dm.DEIG_TCN(F0=7, Fedge=14, H=1, W=w, f_gat=4, f_tcn=4, gat_layers=1, hidden=5, msg_dim=3, seed=4)
    loss, logs, scores, pred = model.forward_loss(batch, {"sparse": 0., "dir": 0.})
    loss.backward()
    groups = {"gat": model.encoder.gat.params(), "tcn": model.encoder.tcn.params(), "edge_head": [model.W1, model.W2]}
    gradnorm = {k: sum(float(np.abs(p.grad).sum()) for p in ps if p.grad is not None) for k, ps in groups.items()}
    check("upstream_gat_tcn_edge_head_prediction_gradients", all(v > 0 for v in gradnorm.values()), gradnorm)
    numerical = []
    for name, p in [("gat", model.encoder.gat.params()[0]), ("tcn", model.encoder.tcn.params()[0]), ("edge_head", model.W1)]:
        index = np.unravel_index(np.argmax(np.abs(p.grad)), p.grad.shape)
        analytic = float(p.grad[index]); original = float(p.data[index]); eps = 1e-5
        p.data[index] = original + eps
        plus = float(model.forward_loss(batch, {"sparse": 0., "dir": 0.})[0].data)
        p.data[index] = original - eps
        minus = float(model.forward_loss(batch, {"sparse": 0., "dir": 0.})[0].data)
        p.data[index] = original
        numeric = (plus - minus) / (2 * eps)
        numerical.append({"group": name, "analytic": analytic, "numeric": numeric, "abs_error": abs(analytic - numeric)})
    check("upstream_sampled_gradient_finite_difference", all(x["abs_error"] < 1e-5 for x in numerical), numerical)
    before = model.infer_pred(batch)
    scores_np = model.infer_scores(batch)
    mask = scores_np.copy(); mask[:3] = 0
    after = model.infer_pred(batch, s_override=mask)
    check("upstream_intent_edge_ablation_changes_prediction", float(np.max(np.abs(before - after))) > 1e-8,
          {"max_prediction_change": float(np.max(np.abs(before - after)))})
    # Directly exercise the lag feature with short and sufficiently long windows.
    a = np.array([2., 1., 4., 0., 5., 3., 7., 2.])
    b = np.r_[0, a[:-1]]
    short = features._windowed_lag_corr(a[-3:], b[-3:])
    long = features._windowed_lag_corr(a, b)
    check("default_three_step_window_has_lag_features", abs(short) > 0,
          {"default_W": 3, "minimum_samples_after_lag": 4, "short_result": short, "long_result": long})


def upstream_end_to_end():
    fixtures = OUT / "fixtures"
    data_dir = fixtures / "pems_spatial_kmeans_topK"
    workdir = fixtures / "intent_workdir"
    data_dir.mkdir(parents=True, exist_ok=True); workdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(30)
    n, t = 6, 96
    x = 10 + np.sin(np.arange(t)[:, None] / 4 + np.arange(n)[None, :] / 3) + rng.normal(0, .1, (t, n))
    pd.DataFrame(x, columns=[str(i) for i in range(n)]).to_csv(data_dir / "id_standardize_Env_A_timeseries.csv", index=False)
    pd.DataFrame({"x": rng.uniform(116, 117, n), "y": rng.uniform(39, 40, n)}).to_csv(data_dir / "id_node_positions.csv", index=False)
    proc = subprocess.run([sys.executable, "-B", str(ROOT / "260628/main.py"), "--epochs", "1", "--viz-count", "2",
                           "--out", str(OUT / "intent_smoke")], cwd=workdir,
                          capture_output=True, text=True, encoding="utf-8", timeout=150)
    (OUT / "intent_smoke.log").write_text(proc.stdout + "\nSTDERR\n" + proc.stderr, encoding="utf-8")
    summaries = sorted((OUT / "intent_smoke").glob("*/run_summary*.json"))
    summary = json.loads(summaries[-1].read_text(encoding="utf-8")) if summaries else {}
    check("upstream_end_to_end_train_export", proc.returncode == 0 and summary.get("intent_windows_saved") == t - 3,
          {"returncode": proc.returncode, "saved_windows": summary.get("intent_windows_saved"), "expected_windows": t - 3,
           "log": str(OUT / "intent_smoke.log"), "stderr_tail": proc.stderr[-1200:]})
    if summaries:
        exported = next(summaries[-1].parent.glob("intent_graphs_data*.npz"))
        intent = loader.load_intent(str(exported))
        pos, meta = loader.resolve_intent_positions(intent, loader.build_time_axis(Config(), t), Config(), len(intent["edges_list"]))
        with np.load(exported, allow_pickle=True) as data:
            expected = data["intent_time_t"].copy()
            upstream_keys = list(data.files)
        check("unit_stride_full_export_alignment", np.array_equal(pos, expected),
              {"method": meta["method"], "actual_first": int(pos[0]), "expected_first": int(expected[0]),
               "note": "Full unit-stride export aligns by tail length despite metadata being ignored; sparse fixture is tested separately."})
        RESULT["evidence"]["actual_upstream_export_keys"] = upstream_keys


def fusion_end_to_end():
    cfg = small_config()
    cfg.out_root = str(OUT / "fusion_smoke")
    cfg.deploy.num_steps = 2; cfg.deploy.source_split = "val"
    cfg.views.num_steps = 1
    io_utils = mod("260731.utils.io_utils")
    formats = io_utils.IMAGE_FORMATS
    io_utils.IMAGE_FORMATS = ["png"]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            path = Path(main_module.run(cfg, SimpleNamespace(export_input_graphs=True)))
    finally:
        io_utils.IMAGE_FORMATS = formats
        (OUT / "fusion_smoke.log").write_text(buf.getvalue(), encoding="utf-8")
    summary = json.loads(next((path / "logs").glob("run_summary*.json")).read_text(encoding="utf-8"))
    arrays = list((path / "arrays").glob("*.npz"))
    images = list(path.rglob("*.png"))
    check("fusion_end_to_end_train_deploy_export", summary["deploy_steps"] == 2 and summary["views_exported_steps"] == 1
          and len(arrays) == 2 and len(images) >= 13,
          {"output": str(path), "deploy_steps": summary["deploy_steps"], "view_steps": summary["views_exported_steps"],
           "arrays": len(arrays), "png_images": len(images), "best_val": summary["best_val"],
           "used_synthetic": summary["used_synthetic"], "figure_formats_tested": ["png"]})


def realtime_contract():
    cfg = small_config()
    bundle = loader.load_all(cfg)
    builder = Builder(bundle, cfg, torch.device("cpu"))
    latest_pos = builder.T - 1
    last_valid_pos = builder._pos(builder.valid_centers[-1])
    try:
        builder.build(builder.n_show - 1)
        error = None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    check("realtime_latest_observation_can_be_deployed_without_label", last_valid_pos == latest_pos and error is None,
          {"latest_observed_pos": latest_pos, "last_deployable_pos": last_valid_pos, "build_latest_error": error,
           "reason": "SampleBuilder shares supervised target and next-graph requirements with deployment inference."})
    dm = mod("260628.deig.data")
    values = np.array([[0., 1., 2., 3., 4., 5., 4., 3., np.nan, 1., 0., 1.],
                       [1., 2., 3., 4., 5., 6., 5., 4., np.nan, 2., 1., 2.]])
    changed = values.copy(); changed[:, 9] += 3
    def transform(v):
        data = dm.TrafficData(X=v, coords=np.zeros((2, 2)), hour=np.arange(12) % 24,
                              weekday=np.zeros(12, dtype=int), true_chains=[])
        return dm.Preprocessor(train_end=6).fit_transform(data)["Z"]
    before, after = transform(values), transform(changed)
    difference = float(np.max(np.abs(before[:, :9] - after[:, :9])))
    check("upstream_missing_value_fill_is_causal", difference < 1e-10,
          {"query_time": 8, "only_changed_time": 9, "max_change_at_or_before_query_time": difference,
           "condition": "Missing observation at t=8 is linearly interpolated using t=9."})
    # The fusion CSV reader fills missing values with the whole-series mean.
    csv_a, csv_b = OUT / "missing_a.csv", OUT / "missing_b.csv"
    pd.DataFrame(values.T).to_csv(csv_a, index=False)
    pd.DataFrame(changed.T).to_csv(csv_b, index=False)
    before = loader.load_flow(str(csv_a))[0]
    after = loader.load_flow(str(csv_b))[0]
    difference = float(np.max(np.abs(before[:, :9] - after[:, :9])))
    check("fusion_missing_value_fill_is_causal", difference < 1e-10,
          {"query_time": 8, "only_changed_time": 9, "max_change_at_or_before_query_time": difference,
           "condition": "load_flow fills missing values using the whole-series mean."})
    data_dir = ROOT / "pems_spatial_kmeans_topK"
    frame = pd.read_csv(data_dir / "id_standardize_Env_A_timeseries.csv")
    coords = pd.read_csv(data_dir / "id_node_positions.csv")
    RESULT["evidence"]["current_local_input_inventory"] = {
        "flow_shape": list(frame.shape), "coords_rows": len(coords),
        "flow_missing_values": int(frame.isna().sum().sum()),
        "flow_nonfinite_values": int((~np.isfinite(frame.to_numpy(dtype=float))).sum()),
        "graph_inputs": [p.name for p in data_dir.iterdir() if p.suffix in (".pkl", ".npz")],
        "interpretation": "Missing-value leakage is a reproducible conditional defect; clean current CSV does not trigger it."}
    # Verify the availability nuance using metadata from the real smoke export.
    exported = sorted((OUT / "intent_smoke").glob("*/intent_graphs_data*.npz"))[-1]
    with np.load(exported, allow_pickle=True) as data:
        times, ends = data["intent_time_t"], data["input_end_t"]
        metadata_check = bool(np.all(ends[1:] <= times[:-1]))
    check("r1_next_target_graph_is_available_with_latest_upstream_semantics", metadata_check,
          {"interpretation": "For H=1, stride=1 and intent_time=input_end+1, graph labelled t+1 only needs observations through t. A_I_next alone is not proof of future leakage.",
           "metadata_enforced_by_fusion_loader": False})


def connected_pipeline_contract():
    import pickle
    cfg = small_config()
    data_dir = OUT / "fixtures/pems_spatial_kmeans_topK"
    exported = sorted((OUT / "intent_smoke").glob("*/intent_graphs_data*.npz"))[-1]
    graph = np.full((6, 6, 1), "", dtype=object)
    weights = np.zeros((6, 6, 1))
    for i in range(6):
        graph[i, (i + 1) % 6, 0] = "-->"
        weights[i, (i + 1) % 6, 0] = .7
    causal_path = data_dir / "synthetic_causal_fixture.pkl"
    with causal_path.open("wb") as f:
        pickle.dump({"graph": graph, "val_matrix": weights, "meta": {"synthetic_audit_fixture": True}}, f)
    cfg.data.data_dir = str(data_dir)
    cfg.data.causal_pkl = causal_path.name
    cfg.data.intent_npz = str(exported)
    cfg.data.allow_synthetic = False
    cfg.train.epochs = 1
    cfg.deploy.num_steps = 2; cfg.deploy.source_split = "val"
    cfg.views.num_steps = 1
    cfg.out_root = str(OUT / "connected_smoke")
    io_utils = mod("260731.utils.io_utils")
    formats = io_utils.IMAGE_FORMATS
    io_utils.IMAGE_FORMATS = ["png"]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            path = Path(main_module.run(cfg, SimpleNamespace(export_input_graphs=False)))
    finally:
        io_utils.IMAGE_FORMATS = formats
        (OUT / "connected_smoke.log").write_text(buf.getvalue(), encoding="utf-8")
    summary = json.loads(next((path / "logs").glob("run_summary*.json")).read_text(encoding="utf-8"))
    check("actual_upstream_npz_to_fusion_train_deploy", summary["n_show"] == 93 and summary["deploy_steps"] == 2
          and not any(summary["used_synthetic"].values()),
          {"output": str(path), "n_show": summary["n_show"], "valid_centers": summary["valid_centers"],
           "used_synthetic_fallback": summary["used_synthetic"],
           "note": "Every input is a locally generated test fixture; no fallback and no real-data performance claim."})
    sys.path.insert(0, str(ROOT / "260628"))
    upstream_main = mod("260628.main")
    checks = []
    for horizon in [1, 3, 6]:
        train, val, test, train_cut, val_cut = upstream_main.make_windows(96, 3, horizon, stride=1)
        checks.append(all(t + horizon < train_cut for t in train)
                      and all(t + 1 >= train_cut and t + horizon < val_cut for t in val)
                      and all(t + 1 >= val_cut for t in test))
    check("upstream_target_intervals_do_not_cross_splits", all(checks), {"horizons_tested": [1, 3, 6]})


if __name__ == "__main__":
    if "--realtime-only" in sys.argv or "--integration-only" in sys.argv:
        RESULT = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
        functions = [realtime_contract] if "--realtime-only" in sys.argv else [connected_pipeline_contract]
    else:
        functions = [version_inventory, time_and_data_contract, model_contract, deployment_contract,
                     upstream_model_contract, upstream_end_to_end, fusion_end_to_end, realtime_contract, connected_pipeline_contract]
    for fn in functions:
        section(fn)
    print(json.dumps({"counts": {status: sum(c["status"] == status for c in RESULT["checks"]) for status in ["PASS", "FAIL"]},
                      "result": str(OUT / "results.json")}, ensure_ascii=False), flush=True)
