"""Draw stored adjacent gate windows without training or model inference.

Examples (execute from this file's directory):
  python plot_intent_sequences.py
  python plot_intent_sequences.py --viz-start-test 20 --viz-count 97
  python plot_intent_sequences.py --mode saved_threshold --formats png
  python plot_intent_sequences.py --models dynamic_gate --seeds 0 --q .90
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from experiment.sequence_visualization import (
    PlotSettings, discover_runs, load_sequence, render_sequence, timestamp_now,
)


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", type=Path, default=Path(__file__).resolve().parent / "outputs" / "suite_20261013_221703_583502")
    ap.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["test"])
    ap.add_argument("--models", nargs="+", help="Saved model folder names; default all models")
    ap.add_argument("--seeds", nargs="+", help="0 1 2 or seed_0 seed_1 seed_2; default all saved seeds")
    ap.add_argument("--start", "--viz-start-test", dest="start", type=int, default=0,
                    help="Start index within each selected split (legacy alias retained)")
    ap.add_argument("--viz-count", type=int, default=10, help="Number of adjacent saved windows; use 97 for the longer legacy sequence")
    ap.add_argument("--panel-count", type=int, default=10, help="First adjacent windows in the overview; 0 disables panel")
    ap.add_argument("--mode", choices=("topk", "saved_threshold", "window_quantile"), default=None)
    ap.add_argument("--fraction", type=float, default=.10, help="Top-K fraction of all candidate edges")
    ap.add_argument("--q", type=float, default=None, help="Legacy per-window quantile; implies window_quantile if mode omitted")
    ap.add_argument("--formats", nargs="+", choices=("png", "svg", "pdf"), default=["png", "svg", "pdf"])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--norm", choices=("log", "linear"), default="log")
    ap.add_argument("--vmin", type=float, help="Fixed color limit; supply together with vmax")
    ap.add_argument("--vmax", type=float)
    ap.add_argument("--node-labels", choices=("index", "id", "none"), default="index")
    ap.add_argument("--node-size", type=float, default=70)
    ap.add_argument("--arrow-size", type=float, default=8)
    ap.add_argument("--min-linewidth", type=float, default=.6)
    ap.add_argument("--max-linewidth", type=float, default=3.2)
    ap.add_argument("--figure-width", type=float, default=8)
    ap.add_argument("--figure-height", type=float, default=7)
    ap.add_argument("--xlim", type=float, nargs=2)
    ap.add_argument("--ylim", type=float, nargs=2)
    ap.add_argument("--allow-gaps", action="store_true", help="Explicitly permit and record gaps in selected saved windows")
    ap.add_argument("--out", type=Path, help="Optional separate output root; default each split/figures/consecutive")
    ap.add_argument("--list-only", action="store_true", help="Inventory models/seeds/splits without drawing")
    return ap


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    if args.q is not None and args.mode not in (None, "window_quantile"):
        ap.error("--q is only applicable to --mode window_quantile")
    settings = PlotSettings(start=args.start, count=args.viz_count, panel_count=args.panel_count,
                            mode=args.mode or ("window_quantile" if args.q is not None else "topk"),
                            fraction=args.fraction, quantile=.90 if args.q is None else args.q,
                            formats=tuple(dict.fromkeys(args.formats)), dpi=args.dpi, norm_scale=args.norm,
                            vmin=args.vmin, vmax=args.vmax, node_labels=args.node_labels,
                            node_size=args.node_size, arrow_size=args.arrow_size,
                            min_linewidth=args.min_linewidth, max_linewidth=args.max_linewidth,
                            figure_width=args.figure_width, figure_height=args.figure_height,
                            xlim=args.xlim, ylim=args.ylim, allow_gaps=args.allow_gaps)
    records = discover_runs(args.suite, args.splits, args.models, args.seeds)
    if args.list_only:
        print(json.dumps(records, ensure_ascii=False, indent=2))
        return 0
    timestamp = timestamp_now()
    completed, skipped, failed = [], [], []
    for record in records:
        tag = f"{record['model']}/{record['seed']}/{record['split']}"
        if record["status"] == "not_applicable":
            print(f"[skip] {tag}: {record['reason']}", flush=True)
            skipped.append(record)
            continue
        if record["status"] != "ready":
            failed.append({**record, "status": "failed"})
            print(f"[failed] {tag}: {record['reason']}", flush=True)
            continue
        try:
            data = load_sequence(record["score_path"])
            base = (args.out.resolve() / record["model"] / record["seed"] / record["split"] / "figures" / "consecutive"
                    if args.out is not None else Path(record["split_dir"]) / "figures" / "consecutive")
            print(f"[plot] {tag}: mode={settings.mode}, start={settings.start}, count={settings.count}", flush=True)
            manifest = render_sequence(data, base, settings, timestamp)
            completed.append({**record, "status": "completed", "manifest_path": manifest["manifest_path"],
                              "rendered_windows": manifest["rendered_windows"], "effective_mode": manifest["effective_mode"]})
            print(f"[saved] {tag}: {manifest['rendered_windows']} frames; {manifest['output_dir']}", flush=True)
        except (ValueError, OSError, KeyError) as error:
            failed.append({**record, "status": "failed", "reason": str(error)})
            print(f"[failed] {tag}: {error}", flush=True)
    output = (args.out.resolve() if args.out is not None else args.suite.resolve()) / "figures" / "consecutive" / timestamp
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"suite_plot_manifest_{timestamp}.json"
    manifest = {"suite": str(args.suite.resolve()), "timestamp": timestamp, "settings": asdict(settings),
                "status": "completed" if not failed else "completed_with_errors",
                "completed": completed, "skipped": skipped, "failed": failed,
                "graph_runs_rendered": len(completed), "figure_frames_rendered": sum(item["rendered_windows"] for item in completed),
                "source_arrays_modified": False, "model_inference_performed": False}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"[manifest] {manifest_path}", flush=True)
    print(f"[done] {len(completed)} rendered runs; {len(skipped)} skipped; {len(failed)} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
