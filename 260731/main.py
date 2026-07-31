# -*- coding: utf-8 -*-
"""RA-TCGF 主入口：训练后对多个时间步执行推断与部署。

关键输出：
1. 每个部署时间步的 S_D、节点分数、部署节点/边（NPZ）；
2. 每个部署时间步的融合图和部署图；
3. 多时间步部署序列总览图；
4. deployment_metrics.csv、deployment_manifest.json、run_summary.json；
5. loss 曲线与选定时间步的六类派生视图。
"""
from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import os as _os
    import sys as _sys
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _sys.path.insert(0, _os.path.dirname(_here))
    __package__ = _os.path.basename(_here)

import argparse
import csv
import os
import random
from typing import Iterable, List

import numpy as np
import pandas as pd
import torch

from .config import Config
from .utils.io_utils import RunPaths
from .utils.data_loading import load_all, load_causal
from .dataset import SampleBuilder
from .model import RATCGF
from .train import train_model, save_model
from .modules.module4_deploy import greedy_deploy
from . import visualize as viz

try:
    from . import viz_causal_style as vcs
except ImportError:  # 可选：缺少该文件时仍可完成核心多时间步部署输出
    vcs = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(pref: str) -> torch.device:
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def elastic_utility(S_D: torch.Tensor, dep_cfg, seed: int = 0) -> float:
    """估计融合边分数在随机边故障后的保留比例。"""
    base = float(S_D.sum().item())
    if base < 1e-8:
        return 0.0
    ii, jj = torch.nonzero(S_D > 0, as_tuple=True)
    E = int(ii.shape[0])
    if E == 0:
        return 0.0

    rng = np.random.default_rng(seed)
    remains = []
    n_samples = max(1, int(dep_cfg.fail_samples))
    for _ in range(n_samples):
        k = max(1, int(round(0.1 * E)))
        k = min(k, E)
        sel = rng.choice(E, size=k, replace=False)
        dropped = float(S_D[ii[sel], jj[sel]].sum().item())
        remains.append(max(0.0, (base - dropped) / base))
    return float(np.mean(remains))


def _resolve_causal_dict(cfg, bundle):
    """获得绘图需要的完整因果字典；没有真实文件时包装 A_C。"""
    if os.path.exists(cfg.causal_path()):
        return load_causal(cfg.causal_path())
    A = bundle["A_C"]
    g = np.where(A > 0, "-->", "").astype(object)[:, :, None]
    v = A[:, :, None]
    return {"graph": g, "val_matrix": v,
            "meta": bundle.get("causal_meta", {})}


def _choose_source_centers(builder: SampleBuilder, result: dict, split: str) -> List[int]:
    """选择最终部署来自训练集、验证集还是全部有效时间步。"""
    split = split.lower()
    if split == "train":
        centers = result.get("train_centers", [])
    elif split == "val":
        centers = result.get("val_centers", [])
    elif split == "all":
        centers = builder.valid_centers
    else:
        raise ValueError(f"未知 deploy source_split: {split}")
    return [int(c) for c in centers]




def _filter_centers_by_time(
    builder: SampleBuilder,
    centers: Iterable[int],
    start_time: str | None = None,
    end_time: str | None = None,
) -> List[int]:
    """按部署/视图对应的时刻 t=pos 过滤中心步，时间范围为闭区间。"""
    start = pd.Timestamp(start_time) if start_time else None
    end = pd.Timestamp(end_time) if end_time else None
    if start is not None and end is not None and start > end:
        raise ValueError(f"起始时间晚于结束时间: {start_time} > {end_time}")

    selected = []
    for c in centers:
        c = int(c)
        pos = builder._pos(c)
        t = builder.timestamp_at_pos(pos)
        if start is not None and t < start:
            continue
        if end is not None and t > end:
            continue
        selected.append(c)
    return selected

def _select_centers(
    centers: Iterable[int],
    num_steps: int,
    stride: int = 1,
    selection: str = "latest",
) -> List[int]:
    """从候选中心步中选取部署时间步，避免无控制地导出全部图片。"""
    seq = list(dict.fromkeys(int(c) for c in centers))
    if not seq:
        return []
    stride = max(1, int(stride))
    seq = seq[::stride]
    if num_steps is None or int(num_steps) <= 0 or int(num_steps) >= len(seq):
        return seq

    n = int(num_steps)
    mode = selection.lower()
    if mode == "latest":
        return seq[-n:]
    if mode == "earliest":
        return seq[:n]
    if mode == "uniform":
        idx = np.linspace(0, len(seq) - 1, n, dtype=int)
        return [seq[i] for i in np.unique(idx)]
    raise ValueError(f"未知 deploy selection: {selection}")


def _write_metrics_csv(rows: List[dict], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "sequence_index", "intent_index", "pos", "deployment_time",
        "prediction_target_pos", "prediction_target_time",
        "window_start_pos", "window_start_time",
        "num_nodes", "num_edges", "cost", "budget", "budget_ratio",
        "u_track", "u_elastic", "total_utility",
        "jaccard_intent", "jaccard_persist",
        "consensus_hits", "deployed_score",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _safe_time_tag(time_str: str) -> str:
    return str(time_str).replace("-", "").replace(":", "").replace(" ", "T")


def _save_step_npz(paths, sample: dict, S_D: np.ndarray,
                    node_scores: np.ndarray, deploy: dict) -> str:
    target_c = int(sample["c"])
    pos = int(sample["pos"])
    time_tag = _safe_time_tag(sample["time"])
    arr_path = paths.array_file(
        f"pos{pos:06d}_{time_tag}_c{target_c:04d}_SD_and_deploy", "npz")
    edge_array = np.asarray(deploy["edges"], dtype=float).reshape(-1, 3)
    np.savez_compressed(
        arr_path,
        S_D=S_D,
        node_scores=node_scores,
        deploy_nodes=np.asarray(deploy["nodes"], dtype=np.int64),
        deploy_edge_index=edge_array[:, :2].astype(np.int64),
        deploy_edge_score=edge_array[:, 2].astype(float),
        A_dep=deploy["A_dep"],
        A_dep_weighted=deploy["A_dep_weighted"],
        intent_index=np.int64(target_c),
        deployment_pos=np.int64(pos),
        deployment_time=np.asarray(sample["time"]),
        prediction_target_pos=np.int64(sample["target_pos"]),
        prediction_target_time=np.asarray(sample["target_time"]),
        window_start_pos=np.int64(sample["window_start_pos"]),
        window_start_time=np.asarray(sample["window_start_time"]),
    )
    return arr_path

def run(cfg: Config, args) -> str:
    set_seed(cfg.train.seed)
    device = pick_device(cfg.train.device)
    paths = RunPaths(cfg.out_root)
    print(f"[run] 设备={device}  输出={paths.root}")

    # 1. 数据加载与时间窗口构造
    bundle = load_all(cfg)
    builder = SampleBuilder(bundle, cfg, device)
    cfg._in_dim = builder.in_dim()
    print(f"[data] N={bundle['N']}  n_show={len(bundle['A_I_list'])}  "
          f"有效窗口={len(builder.valid_centers)}  F={cfg._in_dim}")
    print(f"[data] 合成={bundle['used_synthetic']}")
    print(f"[time] pos=0 -> {builder.time_str_at_pos(0)}; "
          f"pos={builder.T - 1} -> {builder.time_str_at_pos(builder.T - 1)}")
    print(f"[time] intent alignment={bundle['intent_alignment']}")

    # 2. 训练并保存最佳模型
    model = RATCGF(cfg)
    model, result = train_model(model, builder, cfg, device)
    model_path = save_model(model, cfg, result, paths)
    print(f"[save] 模型 -> {model_path}")

    # 3. 独立选择“最终部署”的时间步
    source_centers = _choose_source_centers(
        builder, result, cfg.deploy.source_split)
    source_centers = _filter_centers_by_time(
        builder, source_centers,
        cfg.deploy.start_time, cfg.deploy.end_time)
    deploy_centers = _select_centers(
        source_centers,
        cfg.deploy.num_steps,
        cfg.deploy.step_stride,
        cfg.deploy.selection,
    )
    if not deploy_centers:
        raise RuntimeError("没有可部署时间步。请检查 L、r、数据长度和训练/验证划分。")
    print(f"[deploy] source={cfg.deploy.source_split}  "
          f"range=[{cfg.deploy.start_time or '-inf'}, {cfg.deploy.end_time or '+inf'}]  "
          f"候选={len(source_centers)}  选中={len(deploy_centers)}  "
          f"selection={cfg.deploy.selection}  stride={cfg.deploy.step_stride}")

    coords = builder.coords
    deployment_root = os.path.join(paths.figures, "deployment_steps")
    os.makedirs(deployment_root, exist_ok=True)

    model.eval()
    metric_rows: List[dict] = []
    sequence_records: List[dict] = []
    last_out = None
    last_sample = None

    for seq_idx, target_c in enumerate(deploy_centers):
        sample = builder.build(target_c)
        with torch.no_grad():
            out = model(sample)

        S_D = out["S_D"].detach().cpu().numpy()
        nscore = out["node_scores"].detach().cpu().numpy()
        A_I_cur = sample["A_I_cur"].detach().cpu().numpy()
        A_P = out["views"]["P"].detach().cpu().numpy()
        A_CI = out["views"]["CI"].detach().cpu().numpy()

        deploy = greedy_deploy(
            S_D, nscore, A_I_cur, A_P, cfg.deploy, A_CI=A_CI)
        deployed_weighted = torch.as_tensor(
            deploy["A_dep_weighted"], dtype=out["S_D"].dtype,
            device=out["S_D"].device)
        u_el = elastic_utility(deployed_weighted, cfg.deploy,
                               seed=cfg.train.seed + int(target_c))
        deploy["u_elastic"] = u_el
        deploy["total_utility"] = float(
            cfg.deploy.lambda_tr * deploy["u_track"]
            + cfg.deploy.lambda_el * u_el
            - cfg.deploy.lambda_c * deploy["budget_ratio"]
        )

        pos = int(sample["pos"])
        dt_str = sample["time"]
        target_dt_str = sample["target_time"]
        time_tag = _safe_time_tag(dt_str)

        arr_path = _save_step_npz(paths, sample, S_D, nscore, deploy)

        # 所有时间步平铺在 deployment_steps/；文件名以 pos+时间开头，
        # 因而按文件名排序即可按数据时间顺序浏览。
        file_prefix = (
            f"pos{pos:06d}_{time_tag}_c{target_c:04d}_seq{seq_idx:03d}")
        if cfg.deploy.export_fused_graph:
            viz.plot_fused_visualization(
                S_D, nscore, coords, paths,
                base=f"{file_prefix}_fused_SD",
                dt_str=dt_str,
                directory=deployment_root,
            )
        viz.plot_deployment_subgraph(
            deploy, coords, paths,
            base=f"{file_prefix}_deployment_subgraph",
            dt_str=dt_str,
            directory=deployment_root,
        )

        row = {
            "sequence_index": seq_idx,
            "intent_index": int(target_c),
            "pos": pos,
            "deployment_time": dt_str,
            "prediction_target_pos": int(sample["target_pos"]),
            "prediction_target_time": target_dt_str,
            "window_start_pos": int(sample["window_start_pos"]),
            "window_start_time": sample["window_start_time"],
            "num_nodes": deploy["num_nodes"],
            "num_edges": deploy["num_edges"],
            "cost": deploy["cost"],
            "budget": deploy["budget"],
            "budget_ratio": deploy["budget_ratio"],
            "u_track": deploy["u_track"],
            "u_elastic": u_el,
            "total_utility": deploy["total_utility"],
            "jaccard_intent": deploy["jaccard_intent"],
            "jaccard_persist": deploy["jaccard_persist"],
            "consensus_hits": deploy["consensus_hits"],
            "deployed_score": deploy["deployed_score"],
        }
        metric_rows.append(row)
        sequence_records.append({
            **row,
            "nodes": deploy["nodes"],
            "edges": deploy["edges"],
            "array_path": arr_path,
            "figure_dir": deployment_root,
            "figure_prefix": file_prefix,
        })
        print(
            f"[deploy {seq_idx + 1:03d}/{len(deploy_centers):03d}] "
            f"c={target_c} pos={pos} time={dt_str} "
            f"|V|={deploy['num_nodes']} |E|={deploy['num_edges']} "
            f"cost={deploy['cost']:.2f}/{deploy['budget']:.2f} "
            f"track={deploy['u_track']:.3f} elastic={u_el:.3f}"
        )
        last_out = out
        last_sample = sample

    # 4. 多时间步汇总输出
    metrics_csv = os.path.join(paths.root, "deployment_metrics.csv")
    _write_metrics_csv(metric_rows, metrics_csv)
    paths.dump_json(sequence_records, "deployment_manifest")
    viz.plot_deployment_metrics(metric_rows, paths)
    viz.plot_deployment_sequence(sequence_records, coords, paths)
    viz.plot_loss_curve(result["history"], paths)

    # 5. 六类派生视图批量导出：先按起止时间过滤，再控制步长和数量。
    view_candidates = _filter_centers_by_time(
        builder, builder.valid_centers,
        cfg.views.start_time, cfg.views.end_time)
    view_centers = _select_centers(
        view_candidates,
        cfg.views.num_steps,
        cfg.views.step_stride,
        cfg.views.selection,
    )
    if not view_centers:
        raise RuntimeError("views 导出范围内没有有效时间步。请检查起止时间、L 和 r。")
    print(f"[views] range=[{cfg.views.start_time or '-inf'}, "
          f"{cfg.views.end_time or '+inf'}]  candidates={len(view_candidates)}  "
          f"selected={len(view_centers)}  selection={cfg.views.selection}  "
          f"stride={cfg.views.step_stride}")
    view_manifest = viz.export_97_views(
        model, builder, view_centers, coords, paths, cfg, device)

    # 6. 可选：最后一个部署时间步的输入图与高质量参考视图
    if args.export_input_graphs and vcs is not None and last_out is not None:
        target_c = deploy_centers[-1]
        pos = int(last_sample["pos"])
        dt_str = last_sample["time"]
        causal = _resolve_causal_dict(cfg, bundle)
        vcs.plot_input_causal(causal, coords, paths, dt_str=dt_str)
        delta_list = bundle.get("delta_list")
        delta = (float(delta_list[target_c])
                 if delta_list is not None and len(delta_list) > target_c
                 else None)
        vcs.plot_input_intent(
            bundle["A_I_list"][target_c], coords, paths,
            step=int(target_c), delta=delta, dt_str=dt_str)
    elif args.export_input_graphs and vcs is None:
        print("[warn] 缺少 viz_causal_style.py，跳过输入图导出。")

    # 7. 汇总不再只记录最后一步，而是记录完整序列统计
    numeric = lambda key: np.asarray([r[key] for r in metric_rows], dtype=float)
    summary = {
        "output_dir": paths.root,
        "model_path": model_path,
        "N": bundle["N"],
        "n_show": len(bundle["A_I_list"]),
        "valid_centers": len(builder.valid_centers),
        "used_synthetic": bundle["used_synthetic"],
        "time_axis": {
            "pos0": builder.time_str_at_pos(0),
            "last_pos": builder.T - 1,
            "last_time": builder.time_str_at_pos(builder.T - 1),
            "step_minutes": cfg.data.step_minutes,
        },
        "intent_alignment": bundle["intent_alignment"],
        "best_val": result["best_val"],
        "deploy_source_split": cfg.deploy.source_split,
        "deploy_start_time": cfg.deploy.start_time,
        "deploy_end_time": cfg.deploy.end_time,
        "deploy_steps": len(metric_rows),
        "deploy_centers": deploy_centers,
        "deployment_metrics_csv": metrics_csv,
        "mean_num_nodes": float(numeric("num_nodes").mean()),
        "mean_num_edges": float(numeric("num_edges").mean()),
        "mean_cost": float(numeric("cost").mean()),
        "mean_u_track": float(numeric("u_track").mean()),
        "mean_u_elastic": float(numeric("u_elastic").mean()),
        "mean_total_utility": float(numeric("total_utility").mean()),
        "views_start_time": cfg.views.start_time,
        "views_end_time": cfg.views.end_time,
        "views_requested_steps": cfg.views.num_steps,
        "views_exported_steps": len(view_manifest),
        "view_folders": [f for _, f, _ in viz.VIEW_EXPORTS],
        "plot_style": {
            "policy": "strictly_preserve_previous_style",
            "network_and_derived_views": "visualize.py original quiver style",
            "input_and_reference_views": "viz_causal_style.py PCMCI MultiDiGraph style",
            "standalone_input_plots": "viz_inputs.py original style",
            "allowed_changes": "time labels, unique filenames, output directories only",
        },
    }
    paths.dump_json(summary, "run_summary")
    print(f"[done] 多时间步部署及汇总文件位于 {paths.root}")
    return paths.root


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RA-TCGF multi-step deployment pipeline")
    # 路径参数默认 None，真正默认值统一由 config.py 管理，避免三处配置不一致。
    p.add_argument("--data-dir", default=r"C:\Work-lcr\Tjnu-p\Mp\Intention-Recognition\pems_spatial_kmeans_topK")
    p.add_argument("--flow-csv", default="id_standardize_Env_A_timeseries.csv",
                   help="节点原始时间序列")
    p.add_argument("--coord-csv", default="id_node_positions.csv",
                   help="节点坐标")
    p.add_argument("--causal-pkl", default="subgraph_threshold_0.5.pkl",
                   help="因果图 .pkl")
    p.add_argument("--intent-npz", default="intent_graphs_data_20260703_192855.npz",
                   help="意图图 .npz")
    p.add_argument("--node-feat", default=None,
                   help="可选的初始特征文件（.npy/.csv）")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    p.add_argument("--gnn", default=None, choices=["gcn", "gat", "sage"])
    p.add_argument("--show-steps", "--views-steps", dest="views_steps",
                   type=int, default=0,
                   help="views_steps 的导出时间步数；<=0 表示范围内全部")
    p.add_argument("--views-start-time", default="2025-07-24 22:00:00",
                   help="派生视图起始时间（含），如 2025-01-07 00:00:00")
    p.add_argument("--views-end-time", default="2025-07-31 23:00:00",
                   help="派生视图结束时间（含）")
    p.add_argument("--views-stride", type=int, default=1,
                   help="派生视图候选时间的抽样间隔")
    p.add_argument("--views-selection", default="earliest",
                   choices=["latest", "earliest", "uniform"],
                   help="范围内选择最近、最早或均匀分布的若干步")
    p.add_argument("--deploy-steps", type=int, default=0,
                   help="最终部署图输出时间步数；<=0 表示时间范围内全部")
    p.add_argument("--deploy-start-time", default="2025-07-24 22:00:00",
                   help="部署起始时间（含），如 2025-01-07 00:00:00")
    p.add_argument("--deploy-end-time", default="2025-07-31 23:00:00",
                   help="部署结束时间（含）")
    p.add_argument("--deploy-stride", type=int, default=1,
                   help="部署候选时间步的抽样间隔")
    p.add_argument("--deploy-selection", default="earliest",
                   choices=["latest", "earliest", "uniform"],
                   help="从候选时间步中选择最近、最早或均匀分布的若干步")
    p.add_argument("--deploy-split", default="all",
                   choices=["train", "val", "all"],
                   help="最终部署使用训练集、验证集或全部有效中心步")
    p.add_argument("--start-datetime", default="2025-01-01 00:00:00",
                   help="数据集起点；本数据集必须为 2025-01-01 00:00:00")
    p.add_argument("--step-minutes", type=int, default=60,
                   help="采样间隔分钟数；当前数据为 60")
    p.add_argument("--intent-start-pos", type=int, default=None,
                   help="A_I_list[0] 对应的原始流量 pos；不填则显式采用末尾对齐")
    p.add_argument("--strict-data", action="store_true",
                   help="缺少真实输入文件时直接中止，不生成合成数据")
    p.add_argument("--export-input-graphs", action="store_true",
                   help="额外导出最后一个部署步的因果/意图输入图")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = Config()
    if args.data_dir is not None:
        cfg.data.data_dir = args.data_dir.strip()
    if args.flow_csv is not None:
        cfg.data.flow_csv = args.flow_csv.strip()
    if args.coord_csv is not None:
        cfg.data.coord_csv = args.coord_csv.strip()
    if args.causal_pkl is not None:
        cfg.data.causal_pkl = args.causal_pkl.strip()
    if args.intent_npz is not None:
        cfg.data.intent_npz = args.intent_npz.strip()
    if args.node_feat is not None:
        cfg.data.node_feat_file = args.node_feat.strip()
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.device is not None:
        cfg.train.device = args.device
    if args.views_steps is not None:
        cfg.views.num_steps = args.views_steps
    if args.views_start_time is not None:
        cfg.views.start_time = args.views_start_time.strip()
    if args.views_end_time is not None:
        cfg.views.end_time = args.views_end_time.strip()
    if args.views_stride is not None:
        cfg.views.step_stride = args.views_stride
    if args.views_selection is not None:
        cfg.views.selection = args.views_selection
    if args.gnn is not None:
        cfg.model.gnn_type = args.gnn
    if args.deploy_steps is not None:
        cfg.deploy.num_steps = args.deploy_steps
    if args.deploy_start_time is not None:
        cfg.deploy.start_time = args.deploy_start_time.strip()
    if args.deploy_end_time is not None:
        cfg.deploy.end_time = args.deploy_end_time.strip()
    if args.deploy_stride is not None:
        cfg.deploy.step_stride = args.deploy_stride
    if args.deploy_selection is not None:
        cfg.deploy.selection = args.deploy_selection
    if args.deploy_split is not None:
        cfg.deploy.source_split = args.deploy_split
    if args.start_datetime is not None:
        cfg.data.start_datetime = args.start_datetime.strip()
    if args.step_minutes is not None:
        cfg.data.step_minutes = args.step_minutes
    if args.intent_start_pos is not None:
        cfg.data.intent_start_pos = args.intent_start_pos
    if args.strict_data:
        cfg.data.allow_synthetic = False
    run(cfg, args)


if __name__ == "__main__":
    main()
