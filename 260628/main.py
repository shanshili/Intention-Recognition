"""
DEIG-TCN end-to-end pipeline.

Run:  python main.py [--out DIR] [--epochs N] [--quick]

Produces (all filenames carry a run timestamp; every figure saved as
png + svg + pdf):
  * candidate_directed_graph_<ts>.{png,svg,pdf}        (REQUIRED)
  * intent_graphs_10windows_<ts>.{png,svg,pdf}         (REQUIRED, 10 windows)
  * intent_graph_window_00.._09_<ts>.{png,svg,pdf}     (per-window detail)
  * training_loss_<ts>.{png,svg,pdf}
  * edge_mask_ablation_<ts>.{png,svg,pdf}
  * intent_graph_density_<ts>.{png,svg,pdf}
  * edge_intent_scores_<ts>.npz / run_summary_<ts>.json (data exports)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

from deig.data import  Preprocessor
from deig.graph import build_candidate_graph
from deig.features import build_window_inputs
from deig.model import (DEIG_TCN, assemble_batch, compute_reverse_index)
from deig.autograd import Adam
from deig.intent_graph import (build_intent_graph, extract_hotspots,
                               node_intent_attributes)
from deig import viz
import pandas as pd

class Logger:
    """将stdout重定向，同时输出到控制台和文件"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def make_windows(T, W, H, train_frac=0.6, val_frac=0.2, stride=4):
    """按预测目标时间划分窗口，而不是按窗口列表位置切分。

    t 是输入窗口结束索引：输入为 [t-W+1, ..., t]；
    预测目标区间为 [t+1, ..., t+H]。

    原始时间轴：
      train: [0, train_cut-1]
      val:   [train_cut, val_cut-1]
      test:  [val_cut, T-1]

    只有当整个 H 步目标区间落在同一数据段时，窗口才归入该段。
    历史输入可以跨越段边界，这是滚动预测中的正常做法。
    """
    all_t = np.arange(W - 1, T - H, stride, dtype=np.int64)
    train_cut = int(train_frac * T)
    val_cut = int((train_frac + val_frac) * T)

    target_start = all_t + 1
    target_end = all_t + H

    tr_t = all_t[target_end < train_cut]
    va_t = all_t[(target_start >= train_cut) & (target_end < val_cut)]
    te_t = all_t[target_start >= val_cut]

    return (tr_t.tolist(), va_t.tolist(), te_t.tolist(),
            train_cut, val_cut)


def build_inputs_for(pre, cand, t_list, W, H):
    Z = pre["Z"]
    inputs, targets = [], []
    for t in t_list:
        wi = build_window_inputs(pre, cand, t, W)
        inputs.append(wi)
        targets.append(Z[:, t + 1:t + 1 + H])
    return inputs, targets


def mae(pred, target):
    return float(np.mean(np.abs(pred - target)))



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--quick", action="store_true",
                    help="smaller/faster run for testing")
    ap.add_argument("--data-start", default="2025-01-01 00:00:00",
                    help="原始索引0对应的真实时间")
    ap.add_argument("--viz-count", type=int, default=97,
                    help="连续可视化的测试窗口数量")
    ap.add_argument("--viz-start-test", type=int, default=0,
                    help="从测试集第几个窗口开始连续可视化")
    ap.add_argument("--intent-batch-size", type=int, default=64,
                    help="全部窗口意图得分推理批大小")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[run] timestamp = {timestamp}")

    # 创建以时间戳命名的子文件夹
    out_dir = os.path.join(args.out, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    # 启用日志记录
    log_path = os.path.join(out_dir, f"run_log_{timestamp}.txt")
    sys.stdout = Logger(log_path)

    print(f"[run] timestamp = {timestamp}")
    print(f"[run] output directory = {out_dir}")

    # ---------------- data ----------------
    # 生成传感器时空数据
    # if args.quick:
    #     data = generate_synthetic(N=20, days=90, seed=7)
    #     W, H, stride, epochs = 24, 6, 8, 12
    # else:
    #     data = generate_synthetic(N=30, days=210, seed=7)
    #     W, H, stride, epochs = 24, 6, 6, args.epochs
    # N, T = data.X.shape
    # ---------------- data ----------------
    from deig.data import load_csv

    # 指向你的真实数据目录
    DATA_DIR = "../pems_spatial_kmeans_topK"
    flow_csv = os.path.join(DATA_DIR, "id_standardize_Env_A_timeseries.csv")  # 根据实际文件名调整
    coord_csv = os.path.join(DATA_DIR, "id_node_positions.csv")  # 根据实际文件名调整

    data = load_csv(flow_csv, coord_csv, start_time=args.data_start)

    # 真实数据集下，窗口和训练轮次需根据数据规模重新设定
    W, H, stride, epochs = 3, 1, 1, args.epochs
    N, T = data.X.shape
    start_datetime = pd.Timestamp(args.data_start)

    print(f"[data] N={N} sensors, T={T} hourly steps "
          f"({T/24:.0f} days), {len(data.true_chains)} corridors")

    train_end = int(0.6 * T)
    pre = Preprocessor(train_end=train_end).fit_transform(data)

    # ---------------- candidate graph ----------------
    # 构建候选图
    cand = build_candidate_graph(data.coords, pre["Z"], train_end,
                                 k=4, tau_max=4)
    rev = compute_reverse_index(cand)  # 反向索引
    print(f"[graph] candidate edges |E0|={cand['E']}")

    # ---------------- windows ----------------
    # 构建窗口
    tr_t, va_t, te_t, train_cut, val_cut = make_windows(
        T, W, H, stride=stride)
    print(f"[windows] train={len(tr_t)} val={len(va_t)} "
          f"test={len(te_t)} (split by prediction target time)")
    print(f"[windows] target-time cuts: train < {train_cut}, "
          f"val < {val_cut}, test >= {val_cut}")

    print("[feat] building window features (train/val/test)...")
    tr_in, tr_y = build_inputs_for(pre, cand, tr_t, W, H) # 训练数据
    va_in, va_y = build_inputs_for(pre, cand, va_t, W, H)
    te_in, te_y = build_inputs_for(pre, cand, te_t, W, H)

    F0 = tr_in[0]["F_seq"].shape[-1]   # 每步原始节点特征维度 (=7)
    Fedge = tr_in[0]["edge_block"].shape[1] # 边特征维度

    # GAT-TCN 现在端到端训练；编码器在内部决定 z_i^t 的维度 Fz
    model = DEIG_TCN(F0=F0, Fedge=Fedge, H=H, W=W,
                     f_gat=24, f_tcn=24, gat_layers=2,
                     hidden=32, msg_dim=16, seed=0)
    print(f"[model] GAT-TCN encoder: F0={F0} -> Fz={model.Fz}, Fedge={Fedge}, "
          f"#params={sum(p.data.size for p in model.params())}")
    opt = Adam(model.params(), lr=5e-3, weight_decay=1e-5)
    lambdas = {"sparse": 0.02, "dir": 0.05}

    # mini-batch the training windows by concatenation
    rng = np.random.default_rng(0) # 随机数生成器
    idx_all = np.arange(len(tr_in))
    bs = 16
    history = {"pred": [], "total": []}

    print("[train] starting...")
    for ep in range(epochs):
        rng.shuffle(idx_all) # 打乱训练数据顺序
        ep_pred, ep_total, nb = 0.0, 0.0, 0
        for b0 in range(0, len(idx_all), bs):
            bidx = idx_all[b0:b0 + bs]
            batch = assemble_batch([tr_in[i] for i in bidx], # 组装batch
                                   [tr_y[i] for i in bidx], cand, rev)
            opt.zero_grad()
            total, logs, s, pred = model.forward_loss(batch, lambdas)
            total.backward()
            opt.step()
            ep_pred += float(logs["pred"]); ep_total += float(logs["total"])
            nb += 1
        history["pred"].append(ep_pred / nb)
        history["total"].append(ep_total / nb)
        if ep % max(1, epochs // 8) == 0 or ep == epochs - 1:
            # quick val MAE
            vb = assemble_batch(va_in, va_y, cand, rev)
            vpred = model.infer_pred(vb)
            print(f"  epoch {ep:3d}  L_pred={history['pred'][-1]:.4f}  "
                  f"L_total={history['total'][-1]:.4f}  "
                  f"val_MAE={mae(vpred, vb['target']):.4f}")

    # ---------------- test prediction ----------------
    tb = assemble_batch(te_in, te_y, cand, rev) # 组装测试batch
    tpred = model.infer_pred(tb)
    test_mae = mae(tpred, tb["target"])
    persist = np.repeat(tb["z_last"].reshape(-1, 1), H, axis=1)
    test_mae_persist = mae(persist, tb["target"])
    print(f"[test] DEIG-TCN MAE={test_mae:.4f}  "
          f"persistence MAE={test_mae_persist:.4f}")

    # ---------------- edge-mask ablation ----------------
    s_test = model.infer_scores(tb)          # per-edge scores for full test batch
    E_total = s_test.size
    k = max(1, int(0.1 * E_total))
    order = np.argsort(s_test)
    low_idx, top_idx = order[:k], order[-k:]
    rand_idx = rng.choice(E_total, size=k, replace=False)

    def mask_and_eval(mask_idx):
        s2 = s_test.copy()
        s2[mask_idx] = 0.0
        p = model.infer_pred(tb, s_override=s2)
        return mae(p, tb["target"])

    abl = {"full": test_mae,
           "mask_top": mask_and_eval(top_idx),
           "mask_random": mask_and_eval(rand_idx),
           "mask_low": mask_and_eval(low_idx)}
    print(f"[ablation] {abl}")

    # ---------------- all valid-window intent graphs ----------------
    # 保存范围与可视化范围分离：
    #   1) 全部有效 train/val/test 窗口都计算并保存；
    #   2) 仅从测试集选取 97 个连续窗口进行可视化。
    intent_q = 0.90

    all_t = np.asarray(tr_t + va_t + te_t, dtype=np.int64)
    all_in = tr_in + va_in + te_in
    all_y = tr_y + va_y + te_y
    all_split = np.concatenate([
        np.zeros(len(tr_t), dtype=np.int8),
        np.ones(len(va_t), dtype=np.int8),
        np.full(len(te_t), 2, dtype=np.int8),
    ])

    M_all = len(all_t)
    E = int(cand["E"])
    infer_bs = max(1, int(args.intent_batch_size))
    score_chunks = []

    print(f"[intent] inferring all valid windows: M={M_all}, E={E}, "
          f"batch={infer_bs}...")
    for b0 in range(0, M_all, infer_bs):
        b1 = min(b0 + infer_bs, M_all)
        batch = assemble_batch(all_in[b0:b1], all_y[b0:b1], cand, rev)
        s_batch = np.asarray(model.infer_scores(batch), dtype=np.float32)
        expected_size = (b1 - b0) * E
        if s_batch.size != expected_size:
            raise ValueError(
                f"infer_scores返回{s_batch.size}个分数，但期望"
                f"{b1-b0}×{E}={expected_size}个。"
            )
        score_chunks.append(s_batch.reshape(b1 - b0, E))
        if b0 == 0 or b1 == M_all or b1 % (infer_bs * 10) == 0:
            print(f"  [intent] {b1}/{M_all}")

    all_scores = np.concatenate(score_chunks, axis=0)  # (M_all, E)

    # 每个窗口对应的完整时间语义。
    input_start_t = all_t - W + 1
    input_end_t = all_t
    target_start_t = all_t + 1
    target_end_t = all_t + H

    base_time = np.datetime64(start_datetime.to_datetime64(), "h")
    input_start_time = base_time + input_start_t.astype("timedelta64[h]")
    input_end_time = base_time + input_end_t.astype("timedelta64[h]")
    target_start_time = base_time + target_start_t.astype("timedelta64[h]")
    target_end_time = base_time + target_end_t.astype("timedelta64[h]")

    all_densities = []
    all_hotspot_counts = []
    intent_graphs_data = []

    for idx, s in enumerate(all_scores):
        g = build_intent_graph(s, cand, q=intent_q)
        density = len(g["strength"]) / (N * (N - 1))
        hotspots = extract_hotspots(g, min_size=2)

        all_densities.append(density)
        all_hotspot_counts.append(len(hotspots))
        intent_graphs_data.append({
            "input_end_t": int(input_end_t[idx]),
            "target_start_t": int(target_start_t[idx]),
            "target_end_t": int(target_end_t[idx]),
            "edges": g["edges"],
            "strength": g["strength"],
            "delta": g["delta"],
        })

    # ---------------- 97 consecutive test windows for visualization ----------------
    viz_start_in_test = max(0, int(args.viz_start_test))
    requested_viz_count = max(1, int(args.viz_count))
    if viz_start_in_test >= len(te_t):
        raise ValueError(
            f"--viz-start-test={viz_start_in_test}超出测试窗口范围"
            f"0..{len(te_t)-1}。"
        )

    n_show = min(requested_viz_count, len(te_t) - viz_start_in_test)
    test_start_all_idx = len(tr_t) + len(va_t)
    viz_start_all_idx = test_start_all_idx + viz_start_in_test
    viz_indices = np.arange(
        viz_start_all_idx,
        viz_start_all_idx + n_show,
        dtype=np.int64,
    )

    # stride=1时，下面这些索引必然是连续时间窗。
    if n_show > 1 and not np.all(np.diff(all_t[viz_indices]) == stride):
        raise RuntimeError("可视化窗口不是连续窗口，请检查窗口排序。")

    viz_scores = [all_scores[i] for i in viz_indices]
    viz_densities = [all_densities[i] for i in viz_indices]
    viz_hotspot_counts = [all_hotspot_counts[i] for i in viz_indices]
    viz_labels = []

    for all_idx in viz_indices:
        in_start = pd.Timestamp(input_start_time[all_idx])
        in_end = pd.Timestamp(input_end_time[all_idx])
        tar_start = pd.Timestamp(target_start_time[all_idx])
        tar_end = pd.Timestamp(target_end_time[all_idx])

        if H == 1:
            label = (
                f"target={tar_start:%Y-%m-%d %H:%M}\n"
                f"input={in_start:%m-%d %H:%M}–{in_end:%m-%d %H:%M}"
            )
        else:
            label = (
                f"target={tar_start:%Y-%m-%d %H:%M}–"
                f"{tar_end:%Y-%m-%d %H:%M}\n"
                f"input={in_start:%m-%d %H:%M}–{in_end:%m-%d %H:%M}"
            )
        viz_labels.append(label)

    print(
        f"[viz] {n_show} consecutive test windows: "
        f"target {pd.Timestamp(target_start_time[viz_indices[0]])} -> "
        f"{pd.Timestamp(target_end_time[viz_indices[-1]])}"
    )

    # ---------------- figures (all timestamped, png+svg+pdf) ----------------
    print("[viz] writing figures...")
    produced = {}
    produced["candidate"] = viz.plot_candidate_graph(
        data.coords, cand, out_dir, timestamp)

    # 组合图只显示连续97个中的前10个，避免超大画布。
    n_panel = min(10, n_show)
    produced["intent_panel"] = viz.plot_intent_windows(
        data.coords, cand, viz_scores[:n_panel], viz_labels[:n_panel],
        out_dir, timestamp, q=intent_q)

    # 连续97个窗口分别各保存一张图。
    produced["intent_single"] = []
    for k2, (score, label) in enumerate(zip(viz_scores, viz_labels)):
        produced["intent_single"].append(viz.plot_single_intent_window(
            data.coords, cand, score, label,
            out_dir, timestamp, q=intent_q, idx=k2))

    produced["training"] = viz.plot_training_curve(history, out_dir, timestamp)
    produced["ablation"] = viz.plot_edge_mask_ablation(abl, out_dir, timestamp)
    produced["density"] = viz.plot_intent_density(
        viz_densities, viz_labels, out_dir, timestamp)

    # ---------------- data exports: all valid windows ----------------
    # 1) 保存全部窗口、全部候选边的原始意图得分。
    npz_path = os.path.join(out_dir, f"edge_intent_scores_{timestamp}.npz")
    np.savez_compressed(
        npz_path,
        candidate_edges=cand["edges"],
        coords=data.coords,
        scores=all_scores,
        split_id=all_split,  # 0=train, 1=val, 2=test
        input_start_t=input_start_t,
        input_end_t=input_end_t,
        target_start_t=target_start_t,
        target_end_t=target_end_t,
        intent_time_t=target_start_t,
        input_start_time=input_start_time,
        input_end_time=input_end_time,
        target_start_time=target_start_time,
        target_end_time=target_end_time,
        intent_time=target_start_time,
        candidate_edge_feats=cand["edge_feats"],
        intent_quantile_q=np.float64(intent_q),
    )
    print(f"[save] All-window edge intent scores saved to {npz_path}")

    # 2) 保存全部阈值化意图图。边数随窗口变化，因此使用object数组。
    intent_g_npz_path = os.path.join(
        out_dir, f"intent_graphs_data_{timestamp}.npz")
    intent_edges_list = [d["edges"] for d in intent_graphs_data]
    intent_strength_list = [d["strength"] for d in intent_graphs_data]
    intent_delta_list = [d["delta"] for d in intent_graphs_data]

    np.savez_compressed(
        intent_g_npz_path,
        split_id=all_split,
        input_start_t=input_start_t,
        input_end_t=input_end_t,
        target_start_t=target_start_t,
        target_end_t=target_end_t,
        intent_time_t=target_start_t,
        input_start_time=input_start_time,
        input_end_time=input_end_time,
        target_start_time=target_start_time,
        target_end_time=target_end_time,
        intent_time=target_start_time,
        edges_list=np.asarray(intent_edges_list, dtype=object),
        strength_list=np.asarray(intent_strength_list, dtype=object),
        delta_list=np.asarray(intent_delta_list, dtype=object),
        density=np.asarray(all_densities, dtype=np.float32),
        hotspot_count=np.asarray(all_hotspot_counts, dtype=np.int32),
        candidate_edges=cand["edges"],
        coords=data.coords,
        intent_quantile_q=np.float64(intent_q),
    )
    print(f"[save] All-window intent graph data saved to {intent_g_npz_path}")

    def time_text(dt64):
        return np.datetime_as_string(dt64, unit="m").replace("T", " ")

    summary = {
        "timestamp": timestamp,
        "data_start": str(start_datetime),
        "N": int(N), "T": int(T), "W": W, "H": H,
        "stride": stride,
        "candidate_edges": int(cand["E"]),
        "split_rule": "entire prediction target interval",
        "train_target_cut_index": int(train_cut),
        "val_target_cut_index": int(val_cut),
        "train_windows": len(tr_t),
        "val_windows": len(va_t),
        "test_windows": len(te_t),
        "test_MAE": test_mae,
        "persistence_MAE": test_mae_persist,
        "edge_mask_ablation": abl,
        "intent_windows_saved": int(M_all),
        "intent_time_definition": "prediction target start time",
        "intent_quantile_q": intent_q,
        "visualization_split": "test",
        "visualization_contiguous": True,
        "visualization_count": int(n_show),
        "visualization_start_test_index": int(viz_start_in_test),
        "visualization_target_start": time_text(
            target_start_time[viz_indices[0]]),
        "visualization_target_end": time_text(
            target_end_time[viz_indices[-1]]),
        "visualization_density": [float(v) for v in viz_densities],
        "visualization_hotspot_count": [int(v) for v in viz_hotspot_counts],
    }
    sum_path = os.path.join(out_dir, f"run_summary_{timestamp}.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---------------- 保存模型参数 ----------------
    model_params_path = os.path.join(out_dir, f"model_params_{timestamp}.npz")
    model_params = {f"param_{i}": p.data for i, p in enumerate(model.params())}
    np.savez_compressed(model_params_path, **model_params)
    print(f"[save] Model parameters saved to {model_params_path}")

    # ---------------- 前10个传感器前200时间步可视化 ----------------
    n_sensors_plot = min(10, N)
    n_steps_plot = min(200, T)
    Z_plot = pre["Z"][:n_sensors_plot, :n_steps_plot]  # 使用标准化后的数据

    fig, axes = plt.subplots(n_sensors_plot, 1, figsize=(15, 2 * n_sensors_plot), sharex=True)
    if n_sensors_plot == 1:
        axes = [axes]
    for i in range(n_sensors_plot):
        axes[i].plot(Z_plot[i, :], linewidth=0.8)
        axes[i].set_ylabel(f"S{i}", rotation=0, labelpad=20)
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time Step")
    fig.suptitle(f"Time Series for First {n_sensors_plot} Sensors (First {n_steps_plot} Steps)", y=1.01)
    plt.tight_layout()

    ts_fig_path = os.path.join(out_dir, f"sensor_timeseries_{timestamp}")
    fig.savefig(f"{ts_fig_path}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{ts_fig_path}.svg", bbox_inches="tight")
    fig.savefig(f"{ts_fig_path}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[save] Sensor time series plot saved to {ts_fig_path}")

    # ---------------- 保存时间序列和传感器数据为 CSV ----------------
    # 保存时间序列数据 (行: 传感器, 列: 时间步)
    ts_csv_path = os.path.join(out_dir, f"timeseries_data_{timestamp}.csv")
    np.savetxt(ts_csv_path, pre["Z"], delimiter=",", fmt="%.6f")
    print(f"[save] Time series data saved to {ts_csv_path}")

    # 保存传感器坐标数据
    coord_csv_path = os.path.join(out_dir, f"sensor_coords_{timestamp}.csv")
    np.savetxt(coord_csv_path, data.coords, delimiter=",", fmt="%.6f", header="x,y")
    print(f"[save] Sensor coordinates saved to {coord_csv_path}")

    # ---------------- 保存生成数据规格信息 ----------------
    data_spec = {
        "timestamp": timestamp,
        "N_sensors": int(N),
        "T_timesteps": int(T),
        "days": float(T / 24),
        "W_window_size": W,
        "H_horizon": H,
        "stride": stride,
        "true_chains_count": len(data.true_chains),
        "quick_mode": args.quick
    }
    spec_path = os.path.join(out_dir, f"data_spec_{timestamp}.json")
    with open(spec_path, "w") as f:
        json.dump(data_spec, f, indent=2)
    print(f"[save] Data specification saved to {spec_path}")

    print("[done] outputs in", out_dir)

    # 恢复 sys.stdout 以防后续其他操作受影响
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal

    return produced, summary, out_dir, timestamp


if __name__ == "__main__":
    main()
