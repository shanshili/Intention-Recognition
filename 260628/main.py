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
    starts = list(range(W - 1, T - H, stride))
    M = len(starts)
    tr = int(train_frac * M)
    va = int((train_frac + val_frac) * M)
    return starts[:tr], starts[tr:va], starts[va:]


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
    ap.add_argument("--out", default="/mnt/user-data/outputs")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--quick", action="store_true",
                    help="smaller/faster run for testing")
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

    data = load_csv(flow_csv, coord_csv)

    # 真实数据集下，窗口和训练轮次需根据数据规模重新设定
    W, H, stride, epochs = 3, 1, 1, args.epochs
    N, T = data.X.shape

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
    tr_t, va_t, te_t = make_windows(T, W, H, stride=stride)
    print(f"[windows] train={len(tr_t)} val={len(va_t)} test={len(te_t)}")

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

    # ---------------- 10 consecutive test windows ----------------
    n_show = 170
    if len(te_t) >= n_show:
        show_t = te_t[:n_show]
        show_in = te_in[:n_show]
        show_y = te_y[:n_show]
    else:  # pad by reusing
        show_t = (te_t * n_show)[:n_show]
        show_in = (te_in * n_show)[:n_show]
        show_y = (te_y * n_show)[:n_show]

    window_labels, scores_list, densities, hotspot_counts = [], [], [], []
    intent_graphs_data = []
    for k2, (t, wi, y) in enumerate(zip(show_t, show_in, show_y)):
        wb = assemble_batch([wi], [y], cand, rev)
        s = model.infer_scores(wb)
        scores_list.append(s)
        g = build_intent_graph(s, cand, q=0.90)
        densities.append(len(g["strength"]) / (N * (N - 1)))
        hs = extract_hotspots(g, min_size=2)
        hotspot_counts.append(len(hs))
        hh = (t // 24, t % 24)
        window_labels.append(f"win{k2}  t={t} (d{hh[0]} h{hh[1]})")

        intent_graphs_data.append({
            "edges": g["edges"],
            "strength": g["strength"],
            "delta": g["delta"]
        })

    # ---------------- figures (all timestamped, png+svg+pdf) ----------------
    print("[viz] writing figures...")
    produced = {}
    produced["candidate"] = viz.plot_candidate_graph(
        data.coords, cand, out_dir, timestamp) # , chains=data.true_chains)
    produced["intent_panel"] = viz.plot_intent_windows(
        data.coords, cand, scores_list, window_labels, out_dir, timestamp,
        q=0.90)
    produced["intent_single"] = []
    for k2 in range(n_show):
        produced["intent_single"].append(viz.plot_single_intent_window(
            data.coords, cand, scores_list[k2], window_labels[k2],
            out_dir, timestamp, q=0.90, idx=k2))
    produced["training"] = viz.plot_training_curve(history, out_dir, timestamp)
    produced["ablation"] = viz.plot_edge_mask_ablation(abl, out_dir, timestamp)
    produced["density"] = viz.plot_intent_density(
        densities, window_labels, out_dir, timestamp)

    # ---------------- data exports (timestamped) ----------------
    npz_path = os.path.join(out_dir, f"edge_intent_scores_{timestamp}.npz") # 保存每个窗口的原始边意图得分
    np.savez_compressed(
        npz_path,
        edges=cand["edges"], coords=data.coords,
        window_t=np.array(show_t),
        scores=np.array(scores_list),
        candidate_edge_feats=cand["edge_feats"])
    # 保存每个窗口的意图图结构、边权重和方向信息
    intent_g_npz_path = os.path.join(out_dir, f"intent_graphs_data_{timestamp}.npz")
    intent_edges_list = [d["edges"] for d in intent_graphs_data]
    intent_strength_list = [d["strength"] for d in intent_graphs_data]
    intent_delta_list = [d["delta"] for d in intent_graphs_data]
    np.savez_compressed(
        intent_g_npz_path,
        edges_list=np.array(intent_edges_list, dtype=object),
        strength_list=np.array(intent_strength_list, dtype=object),
        delta_list=np.array(intent_delta_list))
    print(f"[save] Intent graphs structure data saved to {intent_g_npz_path}")

    summary = {
        "timestamp": timestamp,
        "N": int(N), "T": int(T), "W": W, "H": H,
        "candidate_edges": int(cand["E"]),
        "train_windows": len(tr_t), "val_windows": len(va_t),
        "test_windows": len(te_t),
        "test_MAE": test_mae, "persistence_MAE": test_mae_persist,
        "edge_mask_ablation": abl,
        "intent_density_per_window": densities,
        "hotspots_per_window": hotspot_counts,
        "intent_quantile_q": 0.90,
    }
    sum_path = os.path.join(out_dir, f"run_summary_{timestamp}.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
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
