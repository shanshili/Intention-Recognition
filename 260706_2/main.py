# -*- coding: utf-8 -*-
r"""RA-TCGF 流程的入口点。

执行：数据加载 -> 训练 -> 保存模型 ->
推断 S_D^t -> 部署子图 G_D^t -> 所有图表 ->
导出所选时间步的 6 个视图。

图（输入 + 目标步的视图）采用 "PCMCI MultiDiGraph" 有向样式
(viz_causal_style)。97 步的大批量导出使用 visualize.export_97_views 中
带 `quiver` 的快速路径。

用法：
    python -m ratcgf.main
    python main.py --data-dir <路径> --strict-data      # 要求真实数据
"""

if __name__ == "__main__" and __package__ in (None, ""):
    import os as _os
    import sys as _sys
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _sys.path.insert(0, _os.path.dirname(_here))   # 加入父目录
    __package__ = _os.path.basename(_here)          # 通常为 "ratcgf"

import argparse
import os
import random

import numpy as np
import torch

from .config import Config
from .utils.io_utils import RunPaths
from .utils.data_loading import load_all, load_causal
from .dataset import SampleBuilder
from .model import RATCGF
from .train import Trainer, train_model, save_model
from .modules.module4_deploy import node_scores, greedy_deploy
from . import visualize as viz
from . import viz_causal_style as vcs


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(pref: str) -> torch.device:
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def elastic_utility(model, sample, S_D, dep_cfg, device):
    r"""U_elastic^t：S_D 在随机边故障下的鲁棒性。"""
    base = float(S_D.sum().item())
    if base < 1e-8:
        return 0.0
    rng = np.random.default_rng(0)
    ii, jj = torch.nonzero(S_D, as_tuple=True)
    E = ii.shape[0]
    if E == 0:
        return 0.0
    drops = []
    for _ in range(dep_cfg.fail_samples):
        k = max(1, int(0.1 * E))
        sel = rng.choice(E, size=k, replace=False)
        remain = base - float(S_D[ii[sel], jj[sel]].sum().item())
        drops.append(remain / base)
    return float(np.mean(drops))


def _resolve_causal_dict(cfg, bundle):
    r"""用于绘制 G_C 的完整因果字典（带符号/滞后）。

    - 存在真实 .pkl -> 重新加载（保留带符号的 val_matrix + 滞后）。
    - 合成           -> 将 A_C 包装在兼容的 1 滞后字典中。
    """
    if os.path.exists(cfg.causal_path()):
        return load_causal(cfg.causal_path())
    A = bundle["A_C"]
    g = np.where(A > 0, "-->", "").astype(object)[:, :, None]
    v = A[:, :, None]
    return {"graph": g, "val_matrix": v, "meta": bundle.get("causal_meta", {})}


def run(cfg: Config, args):
    import matplotlib as mpl
    # 全局字体：统一使用 Times New Roman，并整体放大字号（对所有图片生效）
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['axes.unicode_minus'] = False
    mpl.rcParams['font.size'] = 15
    mpl.rcParams['axes.titlesize'] = 17
    mpl.rcParams['axes.labelsize'] = 15
    mpl.rcParams['xtick.labelsize'] = 13
    mpl.rcParams['ytick.labelsize'] = 13
    mpl.rcParams['legend.fontsize'] = 13
    mpl.rcParams['figure.titlesize'] = 19

    set_seed(cfg.train.seed)
    device = pick_device(cfg.train.device)
    paths = RunPaths(cfg.out_root)
    print(f"[run] 设备={device}  输出={paths.root}")

    # --- 1. 数据加载 ---
    bundle = load_all(cfg)
    print(f"[data] N={bundle['N']}  n_show={len(bundle['A_I_list'])}")
    print(f"[data] 按文件维度={bundle['node_dims']}  "
          f"node_feat={'有' if bundle.get('node_feat') is not None else '无'}")
    print(f"[data] 合成={bundle['used_synthetic']}")
    builder = SampleBuilder(bundle, cfg, device)
    cfg._in_dim = builder.in_dim()
    print(f"[data] 有效窗口={len(builder.valid_centers)}  F={cfg._in_dim}")

    # --- 2. 模型 + 训练 ---
    model = RATCGF(cfg)
    model, result = train_model(model, builder, cfg, device)

    # --- 3. 保存训练模型（带时间戳） ---
    model_path = save_model(model, cfg, result, paths)
    print(f"[save] 模型 -> {model_path}")

    # # --- 4. 最近一步的推断 ---
    # centers = builder.valid_centers
    # target_c = centers[-1]
    # sample = builder.build(target_c)
    # model.eval()
    # with torch.no_grad():
    #     out = model(sample)
    # S_D = out["S_D"].detach().cpu().numpy()
    # nscore = out["node_scores"].detach().cpu().numpy()
    #
    # # --- 5. 部署子图 G_D^t ---
    # A_I_cur = sample["A_I_cur"].detach().cpu().numpy()
    # A_P = out["views"]["P"].detach().cpu().numpy()
    # deploy = greedy_deploy(S_D, nscore, A_I_cur, A_P, cfg.deploy)
    # u_el = elastic_utility(model, sample, out["S_D"], cfg.deploy, device)
    # deploy["u_elastic"] = u_el
    # print(f"[deploy] 节点={deploy['num_nodes']} 边={deploy['num_edges']} "
    #       f"成本={deploy['cost']:.2f} U_track={deploy['u_track']:.3f} "
    #       f"U_elastic={u_el:.3f}")
    #
    # # --- 6. 保存数组 ---
    # arr_path = paths.array_file("SD_and_deploy", "npz")
    # np.savez(arr_path, S_D=S_D, node_scores=nscore,
    #          deploy_nodes=np.array(deploy["nodes"]),
    #          deploy_edges=np.array(deploy["edges"], dtype=object),
    #          A_dep=deploy["A_dep"], target_center=target_c)
    # print(f"[save] 数组 -> {arr_path}")
    #
    # # --- 7. 主要图表 (loss / fused / deployment) ---
    # # 生成时间字符串用于图表标题：数据每小时一条，起点 2025-01-01 00:00，
    # # 因此中心索引 target_c 对应第 target_c 个小时（自动跨天/跨月）。
    # dt_str = viz.step_to_dt_str(target_c)
    # coords = builder.coords
    # viz.plot_loss_curve(result["history"], paths)
    # viz.plot_fused_visualization(S_D, nscore, coords, paths, dt_str=dt_str)
    # viz.plot_deployment_subgraph(deploy, coords, paths, dt_str=dt_str)
    # print("[fig] loss / fused / deployment 已保存")
    #
    # # --- 7b. 输入图（参考 PCMCI 有向样式） ---
    # #   如果存在 .pkl/.npz 则为真实数据；否则为合成数据。
    # causal = _resolve_causal_dict(cfg, bundle)
    # A_I_draw = bundle["A_I_list"][target_c]
    # delta_list = bundle.get("delta_list")
    # delta = (float(delta_list[target_c])
    #          if delta_list is not None and len(delta_list) > target_c else None)
    # vcs.plot_input_causal(causal, coords, paths, dt_str=dt_str)
    # vcs.plot_input_intent(A_I_draw, coords, paths, step=int(target_c), delta=delta, dt_str=dt_str)
    # src_tag = "真实" if not bundle["used_synthetic"]["causal"] else "合成"
    # print(f"[fig] 输入图（因果+意图，{src_tag}）已保存")
    #
    # #   如果存在 .pkl/.npz 则为真实数据；否则为合成数据。
    # views_np = {k: v.detach().cpu().numpy() for k, v in out["views"].items()}
    # views_np["E"] = out["decay"].detach().cpu().numpy()
    # ref_dir = os.path.join(paths.figures, "views_target_ref")
    # os.makedirs(ref_dir, exist_ok=True)
    # for key, folder, title in viz.VIEW_EXPORTS:      # CI, R, CmI, P, N, E
    #     # 因果派生视图（causal_inactive / R_C）转置以保持因果方向一致
    #     vcs.plot_view_reference(
    #         views_np[key], coords, paths,
    #         base=f"view_ref_{folder}_c{target_c}",
    #         title=f"{title}  (step t=c{target_c})",
    #         directory=ref_dir, dt_str=dt_str,
    #         transpose=(key == "CmI"))
    # print(f"[fig] 目标步的 6 个视图（有向样式）-> {ref_dir}")
    # --- 4. 推断所有时间步并绘制部署图 ---
    # --- 4. 推断所有时间步并绘制部署图 ---
    centers = builder.valid_centers
    model.eval()
    for target_c in centers:
        sample = builder.build(target_c)
        with torch.no_grad():
            out = model(sample)
        S_D = out["S_D"].detach().cpu().numpy()
        nscore = out["node_scores"].detach().cpu().numpy()

        # --- 5. 部署子图 G_D^t ---
        A_I_cur = sample["A_I_cur"].detach().cpu().numpy()
        A_P = out["views"]["P"].detach().cpu().numpy()
        deploy = greedy_deploy(S_D, nscore, A_I_cur, A_P, cfg.deploy)
        u_el = elastic_utility(model, sample, out["S_D"], cfg.deploy, device)
        deploy["u_elastic"] = u_el
        print(f"[deploy] step={target_c} 节点={deploy['num_nodes']} 边={deploy['num_edges']} "
              f"成本={deploy['cost']:.2f} U_track={deploy['u_track']:.3f} "
              f"U_elastic={u_el:.3f}")

        # --- 6. 保存数组 ---
        arr_path = paths.array_file(f"SD_and_deploy_c{target_c}", "npz")
        np.savez(arr_path, S_D=S_D, node_scores=nscore,
                 deploy_nodes=np.array(deploy["nodes"]),
                 deploy_edges=np.array(deploy["edges"], dtype=object),
                 A_dep=deploy["A_dep"], target_center=target_c)

        # --- 7. 主要图表 (fused / deployment) ---
        dt_str = viz.step_to_dt_str(target_c)
        coords = builder.coords
        # 为不同时间步的图片指定独立的子目录，避免覆盖
        step_paths = paths.sub(f"step_{target_c}")
        viz.plot_fused_visualization(S_D, nscore, coords, step_paths, dt_str=dt_str)
        viz.plot_deployment_subgraph(deploy, coords, step_paths, dt_str=dt_str)

        # --- 7b. 输入图（参考 PCMCI 有向样式） ---
        causal = _resolve_causal_dict(cfg, bundle)
        A_I_draw = bundle["A_I_list"][target_c]
        delta_list = bundle.get("delta_list")
        delta = (float(delta_list[target_c])
                 if delta_list is not None and len(delta_list) > target_c else None)
        vcs.plot_input_causal(causal, coords, step_paths, dt_str=dt_str)
        vcs.plot_input_intent(A_I_draw, coords, step_paths, step=int(target_c), delta=delta, dt_str=dt_str)

        views_np = {k: v.detach().cpu().numpy() for k, v in out["views"].items()}
        views_np["E"] = out["decay"].detach().cpu().numpy()
        ref_dir = os.path.join(step_paths.figures, "views_target_ref")
        os.makedirs(ref_dir, exist_ok=True)
        for key, folder, title in viz.VIEW_EXPORTS:
            vcs.plot_view_reference(
                views_np[key], coords, step_paths,
                base=f"view_ref_{folder}_c{target_c}",
                title=f"{title}  (step t=c{target_c})",
                directory=ref_dir, dt_str=dt_str,
                transpose=(key == "CmI"))

    print("[fig] 所有时间步的 fused / deployment / 输入图 / 视图已保存")
    # loss 曲线只需绘制一次
    viz.plot_loss_curve(result["history"], paths)
    print("[fig] loss 曲线已保存")


    # --- 8. 所有时间步的 6 个视图大批量导出 ---
    manifest = viz.export_97_views(model, builder, centers, coords,
                                   paths, cfg, device)

    # --- 9. 最终日志 ---
    paths.dump_json({
        "output_dir": paths.root,
        "N": bundle["N"], "n_show": len(bundle["A_I_list"]),
        "valid_centers": len(centers),
        "used_synthetic": bundle["used_synthetic"],
        "best_val": result["best_val"],
        "deploy": {k: deploy[k] for k in
                   ["num_nodes", "num_edges", "cost", "u_track", "u_elastic"]},
        "views_exported_steps": len(manifest),
        "view_folders": [f for _, f, _ in viz.VIEW_EXPORTS],
        "input_graph_step": int(target_c),
        "model_path": model_path,
    }, "run_summary")
    print(f"[done] 汇总与所有文件位于 {paths.root}")
    return paths.root


def build_argparser():
    p = argparse.ArgumentParser(description="RA-TCGF pipeline")
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
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    p.add_argument("--show-steps", type=int, default=192,
                   help="要导出的时间步数 t（默认 97）")
    p.add_argument("--gnn", default=None, choices=["gcn", "gat", "sage"])
    p.add_argument("--strict-data", action="store_true",
                   help="要求真实数据：若缺少任何文件则中止而不合成")
    return p


def main():
    args = build_argparser().parse_args()
    cfg = Config()
    if args.data_dir: cfg.data.data_dir = args.data_dir.strip()
    if args.flow_csv: cfg.data.flow_csv = args.flow_csv.strip()
    if args.coord_csv: cfg.data.coord_csv = args.coord_csv.strip()
    if args.causal_pkl: cfg.data.causal_pkl = args.causal_pkl.strip()
    if args.intent_npz: cfg.data.intent_npz = args.intent_npz.strip()
    if args.node_feat: cfg.data.node_feat_file = args.node_feat.strip()
    if args.epochs: cfg.train.epochs = args.epochs
    if args.device: cfg.train.device = args.device
    if args.show_steps: cfg.train.num_show_steps = args.show_steps
    if args.gnn: cfg.model.gnn_type = args.gnn
    if args.strict_data:
        cfg.data.allow_synthetic = False
    run(cfg, args)


if __name__ == "__main__":
    main()
