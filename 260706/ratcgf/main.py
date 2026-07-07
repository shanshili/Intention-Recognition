# -*- coding: utf-8 -*-
r"""Punto de entrada del pipeline RA-TCGF.

Ejecuta: carga de datos -> entrenamiento -> guardado del modelo ->
inferencia de S_D^t -> subgrafo de despliegue G_D^t -> todas las figuras ->
exportacion de las 6 vistas para los pasos t seleccionados.

Figuras de grafo (entrada + vistas del paso objetivo) en estilo dirigido
"PCMCI MultiDiGraph" (viz_causal_style). El volcado masivo de 95 pasos usa el
camino rapido con `quiver` de visualize.export_97_views.

Uso:
    python -m ratcgf.main
    python main.py --data-dir <ruta> --strict-data      # exige datos reales
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os as _os
    import sys as _sys
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _sys.path.insert(0, _os.path.dirname(_here))   # anade la carpeta padre
    __package__ = _os.path.basename(_here)          # normalmente "ratcgf"

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
    r"""U_elastic^t: robustez de S_D ante fallos aleatorios de aristas."""
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
    r"""Dict causal completo (con signo/lag) para dibujar G_C.

    - .pkl real presente  -> se recarga (conserva val_matrix con signo + lag).
    - sintetico           -> se envuelve A_C en un dict compatible de 1 lag.
    """
    if os.path.exists(cfg.causal_path()):
        return load_causal(cfg.causal_path())
    A = bundle["A_C"]
    g = np.where(A > 0, "-->", "").astype(object)[:, :, None]
    v = A[:, :, None]
    return {"graph": g, "val_matrix": v, "meta": bundle.get("causal_meta", {})}


def run(cfg: Config, args):
    set_seed(cfg.train.seed)
    device = pick_device(cfg.train.device)
    paths = RunPaths(cfg.out_root)
    print(f"[run] dispositivo={device}  salida={paths.root}")

    # --- 1. carga de datos ---
    bundle = load_all(cfg)
    print(f"[data] N={bundle['N']}  n_show={len(bundle['A_I_list'])}")
    print(f"[data] dims por fichero={bundle['node_dims']}  "
          f"node_feat={'si' if bundle.get('node_feat') is not None else 'no'}")
    print(f"[data] sintetico={bundle['used_synthetic']}")
    builder = SampleBuilder(bundle, cfg, device)
    cfg._in_dim = builder.in_dim()
    print(f"[data] ventanas validas={len(builder.valid_centers)}  F={cfg._in_dim}")

    # --- 2. modelo + entrenamiento ---
    model = RATCGF(cfg)
    model, result = train_model(model, builder, cfg, device)

    # --- 3. guardar modelo entrenado (con timestamp) ---
    model_path = save_model(model, cfg, result, paths)
    print(f"[save] modelo -> {model_path}")

    # --- 4. inferencia en el paso mas reciente ---
    centers = builder.valid_centers
    target_c = centers[-1]
    sample = builder.build(target_c)
    model.eval()
    with torch.no_grad():
        out = model(sample)
    S_D = out["S_D"].detach().cpu().numpy()
    nscore = out["node_scores"].detach().cpu().numpy()

    # --- 5. subgrafo de despliegue G_D^t ---
    A_I_cur = sample["A_I_cur"].detach().cpu().numpy()
    A_P = out["views"]["P"].detach().cpu().numpy()
    deploy = greedy_deploy(S_D, nscore, A_I_cur, A_P, cfg.deploy)
    u_el = elastic_utility(model, sample, out["S_D"], cfg.deploy, device)
    deploy["u_elastic"] = u_el
    print(f"[deploy] nodos={deploy['num_nodes']} aristas={deploy['num_edges']} "
          f"coste={deploy['cost']:.2f} U_track={deploy['u_track']:.3f} "
          f"U_elastic={u_el:.3f}")

    # --- 6. guardar arrays ---
    arr_path = paths.array_file("SD_and_deploy", "npz")
    np.savez(arr_path, S_D=S_D, node_scores=nscore,
             deploy_nodes=np.array(deploy["nodes"]),
             deploy_edges=np.array(deploy["edges"], dtype=object),
             A_dep=deploy["A_dep"], target_center=target_c)
    print(f"[save] arrays -> {arr_path}")

    # --- 7. figuras principales (loss / fused / deployment) ---
    coords = builder.coords
    viz.plot_loss_curve(result["history"], paths)
    viz.plot_fused_visualization(S_D, nscore, coords, paths)
    viz.plot_deployment_subgraph(deploy, coords, paths)
    print("[fig] loss / fused / deployment guardadas")

    # --- 7b. grafos de ENTRADA (estilo dirigido PCMCI de referencia) ---
    #   Reales si los .pkl/.npz existen; sinteticos en caso contrario.
    causal = _resolve_causal_dict(cfg, bundle)
    A_I_draw = bundle["A_I_list"][target_c]
    delta_list = bundle.get("delta_list")
    delta = (float(delta_list[target_c])
             if delta_list is not None and len(delta_list) > target_c else None)
    vcs.plot_input_causal(causal, coords, paths)
    vcs.plot_input_intent(A_I_draw, coords, paths, step=int(target_c), delta=delta)
    src_tag = "reales" if not bundle["used_synthetic"]["causal"] else "sinteticos"
    print(f"[fig] grafos de entrada (causal+intencion, {src_tag}) guardados")

    # --- 7c. las 6 vistas derivadas del paso objetivo, en estilo referencia ---
    views_np = {k: v.detach().cpu().numpy() for k, v in out["views"].items()}
    views_np["E"] = out["decay"].detach().cpu().numpy()
    ref_dir = os.path.join(paths.figures, "views_target_ref")
    os.makedirs(ref_dir, exist_ok=True)
    for key, folder, title in viz.VIEW_EXPORTS:      # CI, R, CmI, P, N, E
        vcs.plot_view_reference(
            views_np[key], coords, paths,
            base=f"view_ref_{folder}_c{target_c}",
            title=f"{title}  (step t=c{target_c})",
            directory=ref_dir)
    print(f"[fig] 6 vistas del paso objetivo (estilo dirigido) -> {ref_dir}")

    # --- 8. volcado masivo de las 6 vistas para todos los pasos (quiver) ---
    manifest = viz.export_97_views(model, builder, centers, coords,
                                   paths, cfg, device)

    # --- 9. log final ---
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
    print(f"[done] resumen y todos los ficheros en {paths.root}")
    return paths.root


def build_argparser():
    p = argparse.ArgumentParser(description="RA-TCGF pipeline")
    p.add_argument("--data-dir", default="C:\Work-lcr\Tjnu-p\Mp\Intention-Recognition\pems_spatial_kmeans_topK")
    p.add_argument("--flow-csv", default="id_standardize_Env_A_timeseries.csv",
                   help="serie temporal cruda de nodo")
    p.add_argument("--coord-csv", default="id_node_positions.csv",
                   help="coordenadas de nodo")
    p.add_argument("--causal-pkl", default="subgraph_threshold_0.5.pkl",
                   help="grafo causal .pkl")
    p.add_argument("--intent-npz", default="intent_graphs_data_20260703_192855.npz",
                   help="grafo intencion .npz")
    p.add_argument("--node-feat", default=None,
                   help="fichero opcional de features iniciales (.npy/.csv)")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    p.add_argument("--show-steps", type=int, default=97,
                   help="numero de pasos t a exportar (por defecto 97)")
    p.add_argument("--gnn", default=None, choices=["gcn", "gat", "sage"])
    p.add_argument("--strict-data", action="store_true",
                   help="exige datos reales: si falta algun fichero, aborta "
                        "en vez de sintetizar")
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
