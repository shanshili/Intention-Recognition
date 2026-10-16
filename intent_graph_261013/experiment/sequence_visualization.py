"""Offline consecutive graph rendering; only NumPy artifacts are read.

The public array API also permits training-time integration after artifacts are
saved. No model, dataset loader, PyTorch, or evaluator is imported here.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import FancyArrowPatch
import numpy as np


@dataclass
class SequenceData:
    scores: np.ndarray
    coords: np.ndarray
    edges: np.ndarray
    window_ids: np.ndarray
    target_times: np.ndarray
    protocol: dict
    node_ids: np.ndarray | None = None
    model: str = "unknown"
    seed: str = "unknown"
    split: str = "test"
    score_kind: str = "unknown"
    threshold: dict = field(default_factory=dict)
    source: str | None = None
    protocol_id: str | None = None


@dataclass
class PlotSettings:
    start: int = 0
    count: int = 10
    panel_count: int = 10
    mode: str = "topk"
    fraction: float = 0.10
    quantile: float = 0.90
    formats: tuple[str, ...] = ("png", "svg", "pdf")
    dpi: int = 300
    norm_scale: str = "log"
    vmin: float | None = None
    vmax: float | None = None
    node_labels: str = "index"
    node_size: float = 70.0
    arrow_size: float = 8.0
    min_linewidth: float = 0.6
    max_linewidth: float = 3.2
    figure_width: float = 8.0
    figure_height: float = 7.0
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    allow_gaps: bool = False


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                          encoding="utf-8")


def timestamp_now():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def discover_runs(suite, splits=("test",), models=None, seeds=None):
    """Inventory saved runs, explicitly recording absent/non-applicable scores.

    This intentionally does not consult graph_metrics.available: fixed graphs
    have saved gates even though they have no learned ranking.
    """
    suite = Path(suite).resolve()
    if not suite.is_dir():
        raise FileNotFoundError(f"Suite directory does not exist: {suite}")
    models = None if models is None else set(models)
    seeds = None if seeds is None else {str(seed).removeprefix("seed_") for seed in seeds}
    records = []
    for model_path in sorted(suite.iterdir()):
        if not model_path.is_dir() or (models is not None and model_path.name not in models):
            continue
        for seed_path in sorted(model_path.glob("seed_*")):
            if not seed_path.is_dir() or (seeds is not None and seed_path.name[5:] not in seeds):
                continue
            for split in splits:
                split_path = seed_path / "splits" / split
                # Legacy single-split runs keep their test artifacts at root.
                if split == "test" and not split_path.is_dir() and (seed_path / "run_summary.json").exists():
                    split_path = seed_path
                summary = _read_json(split_path / "run_summary.json")
                if not summary and not split_path.exists():
                    records.append({"model": model_path.name, "seed": seed_path.name, "split": split,
                                    "status": "missing_split", "reason": "No saved split artifacts"})
                    continue
                score_path = split_path / "edge_scores.npz"
                graph = _read_json(split_path / "graph_evaluation.json")
                record = {"model": model_path.name, "seed": seed_path.name, "split": split,
                          "split_dir": str(split_path), "score_path": str(score_path)}
                if score_path.exists():
                    record["status"] = "ready"
                elif graph.get("score_kind") == "none" or model_path.name in {"persistence", "ridge_ar", "local"}:
                    record.update(status="not_applicable", reason="This model exports no graph gates")
                else:
                    record.update(status="missing_scores", reason="Expected edge_scores.npz is absent")
                records.append(record)
    if not records:
        raise ValueError("No model/seed runs match the selected suite, models, seeds, and splits")
    return records


def _saved_validation_threshold(split_path):
    split_path = Path(split_path)
    run_path = split_path.parent.parent if split_path.parent.name == "splits" else split_path
    # Prefer the saved validation result; a copy in test is accepted only if
    # its provenance explicitly says validation or an explicit fixed value.
    paths = [run_path / "splits" / "val" / "graph_evaluation.json",
             split_path / "graph_evaluation.json"]
    for path in paths:
        spec = dict(_read_json(path).get("threshold", {}))
        if not spec.get("enabled") or spec.get("value") is None:
            continue
        rule = spec.get("rule", "")
        calibration = spec.get("calibration_split")
        if calibration == "val" or str(rule).startswith("validation") or rule in {"explicit", "explicit_absolute_threshold", "predeclared_absolute"}:
            value = float(spec["value"])
            if not np.isfinite(value):
                raise ValueError(f"Nonfinite saved threshold in {path}")
            spec["source_path"] = str(path.resolve())
            return spec
    return {}


def load_sequence(score_path):
    """Load self-contained artifacts, with saved protocol metadata alongside."""
    score_path = Path(score_path).resolve()
    split_path = score_path.parent
    run_path = split_path.parent.parent if split_path.parent.name == "splits" else split_path
    config = _read_json(split_path / "resolved_config.json") or _read_json(run_path / "resolved_config.json")
    protocol = config.get("protocol", {})
    if not all(key in protocol for key in ("W", "step_minutes")):
        raise ValueError(f"Saved resolved_config.json must specify W and step_minutes: {split_path}")
    with np.load(score_path, allow_pickle=False) as archive:
        required = {"edge_scores", "coords", "candidate_edges", "window_id", "target_times"}
        if not required.issubset(archive.files):
            raise ValueError(f"Missing artifact arrays: {sorted(required - set(archive.files))}")
        split = split_path.name if split_path.parent.name == "splits" else "test"
        return SequenceData(
            scores=archive["edge_scores"], coords=archive["coords"], edges=archive["candidate_edges"],
            window_ids=archive["window_id"], target_times=archive["target_times"], protocol=protocol,
            node_ids=archive["node_ids"] if "node_ids" in archive.files else None,
            model=str(config.get("model") or run_path.parent.name), seed=run_path.name, split=split,
            score_kind=str(archive["score_kind"].item()) if "score_kind" in archive.files else "unknown",
            threshold=_saved_validation_threshold(split_path), source=str(score_path),
            protocol_id=str(archive["protocol_id"].item()) if "protocol_id" in archive.files else config.get("protocol_id"))


def select_mask(scores, mode, fraction=0.10, quantile=0.90, threshold=None):
    """Return selection and positive-rendering masks; stable ties use edge order."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("A gate row must be one-dimensional")
    positive = np.isfinite(scores) & (scores > 0)
    if mode == "topk":
        if not 0 < fraction <= 1:
            raise ValueError("fraction must lie in (0, 1]")
        keep = np.zeros(len(scores), dtype=bool)
        k = min(len(scores), int(np.ceil(fraction * len(scores))))
        # Invalid scores rank last; a selected zero is never drawn.
        order = np.argsort(-np.where(np.isfinite(scores), scores, -np.inf), kind="stable")
        keep[order[:k]] = True
        value = None
    elif mode == "saved_threshold":
        if threshold is None or not np.isfinite(threshold):
            raise ValueError("saved_threshold requires a frozen saved validation/fixed threshold")
        value = float(threshold)
        keep = positive & (scores >= value)
    elif mode == "window_quantile":
        if not 0 <= quantile <= 1:
            raise ValueError("quantile must lie in [0, 1]")
        # Legacy exploration: threshold every window separately, retaining ties.
        # It is not a validation-calibrated threshold or an exact fixed budget.
        finite = scores[np.isfinite(scores)]
        value = float(np.quantile(finite, quantile)) if len(finite) else None
        keep = positive & (scores >= value) if value is not None else np.zeros(len(scores), dtype=bool)
    elif mode == "all_candidate":
        keep = np.ones(len(scores), dtype=bool)
        value = None
    else:
        raise ValueError(f"Unknown selection mode: {mode}")
    return keep, keep & positive, value


def validate_sequence(data, settings):
    scores, coords, edges = np.asarray(data.scores), np.asarray(data.coords), np.asarray(data.edges)
    ids, times = np.asarray(data.window_ids), np.asarray(data.target_times)
    if scores.ndim != 2 or coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("scores must be [windows,edges]; coords must be [nodes,2]")
    if len(coords) == 0 or not np.all(np.isfinite(coords)):
        raise ValueError("Finite node coordinates are required")
    if edges.shape != (scores.shape[1], 2) or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("candidate_edges must be integer [edges,2] matching the gate columns")
    if edges.size and (edges.min() < 0 or edges.max() >= len(coords)):
        raise ValueError("Candidate edge indices are outside the node array")
    if edges.size and (np.any(edges[:, 0] == edges[:, 1]) or len(np.unique(edges, axis=0)) != len(edges)):
        raise ValueError("Candidate edges must be unique directed pairs without self-loops")
    if data.node_ids is not None and np.asarray(data.node_ids).shape != (len(coords),):
        raise ValueError("node_ids must match the stored node coordinate order")
    if ids.shape != (scores.shape[0],) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("window_id must be integer [windows]")
    if times.ndim != 2 or len(times) != len(scores) or times.shape[1] < 1:
        raise ValueError("target_times must be [windows,horizon]")
    if settings.start < 0 or settings.start >= len(scores) or settings.count < 1:
        raise ValueError(f"start/count must select saved windows; available range 0..{len(scores)-1}")
    if settings.panel_count < 0 or settings.dpi < 1:
        raise ValueError("panel_count must be nonnegative and DPI must be positive")
    if settings.node_labels not in {"index", "id", "none"}:
        raise ValueError("node_labels must be index, id, or none")
    if not settings.formats or any(ext not in {"png", "svg", "pdf"} for ext in settings.formats):
        raise ValueError("formats must contain png, svg, or pdf")
    if min(settings.figure_width, settings.figure_height, settings.node_size, settings.arrow_size) <= 0:
        raise ValueError("Figure, node, and arrow sizes must be positive")
    if settings.min_linewidth < 0 or settings.max_linewidth < settings.min_linewidth:
        raise ValueError("linewidth bounds must satisfy 0 <= minimum <= maximum")
    for bounds in (settings.xlim, settings.ylim):
        if bounds is not None and (len(bounds) != 2 or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]):
            raise ValueError("Coordinate plot bounds must be finite increasing pairs")
    W, step = int(data.protocol.get("W", 0)), int(data.protocol.get("step_minutes", 0))
    stride = int(data.protocol.get("train_stride" if data.split == "train" else "eval_stride", 1))
    if min(W, step, stride) < 1:
        raise ValueError("Saved W, step_minutes, and split stride must be positive")
    if "H" in data.protocol and int(data.protocol["H"]) != times.shape[1]:
        raise ValueError("Saved horizon and target_times disagree")
    stop = min(len(scores), settings.start + settings.count)
    picks = np.arange(settings.start, stop)
    selected_scores = scores[picks]
    if not np.all(np.isfinite(selected_scores)) or np.any(selected_scores < 0) or np.any(selected_scores > 1):
        raise ValueError("Selected raw gate scores must all be finite and within [0, 1]")
    if data.score_kind == "fixed" and not np.all(selected_scores == 1):
        raise ValueError("Fixed score_kind requires all-one gates to label a fixed candidate graph")
    times = times[picks].astype("datetime64[ns]")
    if np.any(np.isnat(times)):
        raise ValueError("Selected target timestamps contain NaT")
    spacing = np.timedelta64(step, "m").astype("timedelta64[ns]")
    if times.shape[1] > 1 and not np.all(np.diff(times, axis=1) == spacing):
        raise ValueError("Target horizon timestamps disagree with step_minutes")
    adjacent = np.diff(ids[picks]) == stride
    adjacent_time = np.diff(times[:, 0]) == stride * spacing
    contiguous = bool(np.all(adjacent) and np.all(adjacent_time))
    if not contiguous and not settings.allow_gaps:
        raise ValueError("Selected windows are not consecutive at the saved split stride; "
                         "choose a contiguous range or explicitly use --allow-gaps")
    return picks, stride, contiguous


def shared_normalization(scores, settings):
    positive = np.asarray(scores)[np.isfinite(scores) & (np.asarray(scores) > 0)]
    if settings.norm_scale not in {"log", "linear"}:
        raise ValueError("norm_scale must be log or linear")
    if (settings.vmin is None) != (settings.vmax is None):
        raise ValueError("Specify both vmin and vmax together")
    if settings.vmin is not None:
        low, high = float(settings.vmin), float(settings.vmax)
        source = "explicit_fixed_range"
    elif len(positive):
        low, high = (float(v) for v in np.quantile(positive, [0.01, 0.99]))
        if high <= low:
            low, high = max(low * 0.5, np.finfo(float).tiny), high * 1.5
        source = "sequence_positive_quantiles_01_99"
    else:
        low, high, source = 1e-12, 1.0, "no_positive_scores"
    if not np.isfinite(low) or not np.isfinite(high) or high <= low or (settings.norm_scale == "log" and low <= 0):
        raise ValueError("Color limits must be finite increasing values; log vmin must be positive")
    cls = colors.LogNorm if settings.norm_scale == "log" else colors.Normalize
    return cls(vmin=low, vmax=high, clip=True), {
        "scale": settings.norm_scale, "vmin": low, "vmax": high, "range_source": source,
        "shared_across_all_sequence_figures": True, "has_positive_scores": bool(len(positive)),
        "clipping": "Values outside limits receive the endpoint color"}


def _time_text(value):
    return np.datetime_as_string(value, unit="m").replace("T", " ")


def _draw(ax, data, row, keep, norm, settings, panel=False):
    coords, edges, scores = np.asarray(data.coords), np.asarray(data.edges), np.asarray(data.scores[row])
    drawn = keep & np.isfinite(scores) & (scores > 0)
    cmap = plt.get_cmap("plasma")
    ax.scatter(coords[:, 0], coords[:, 1], s=settings.node_size * (0.5 if panel else 1),
               color="0.85", edgecolors="0.5", linewidths=0.5, zorder=3)
    for edge in np.flatnonzero(drawn):
        i, j = edges[edge]
        # float32 LogNorm may round a clipped endpoint slightly above one.
        # Bound the display value before deriving alpha/linewidth; raw gates
        # and numerical evaluation artifacts remain unchanged.
        strength = float(np.clip(float(norm(scores[edge])), 0.0, 1.0))
        ax.add_patch(FancyArrowPatch(coords[i], coords[j], arrowstyle="-|>",
                                    mutation_scale=settings.arrow_size,
                                    connectionstyle="arc3,rad=0.08", color=cmap(strength),
                                    linewidth=settings.min_linewidth + (settings.max_linewidth - settings.min_linewidth) * strength,
                                    alpha=float(np.clip(0.15 + 0.85 * strength, 0.0, 1.0)), zorder=2))
    if settings.node_labels != "none":
        labels = data.node_ids if settings.node_labels == "id" and data.node_ids is not None else np.arange(len(coords))
        for label, (x, y) in zip(labels, coords):
            ax.text(x, y, str(label), ha="center", va="center", fontsize=5 if panel else 6,
                    color="0.25", zorder=4)
    ax.set_aspect("equal", adjustable="box")
    # Fixed geographic extent avoids changing camera positions across frames.
    for setter, limits, column in ((ax.set_xlim, settings.xlim, 0), (ax.set_ylim, settings.ylim, 1)):
        lo, hi = limits if limits is not None else (float(coords[:, column].min()), float(coords[:, column].max()))
        pad = (hi - lo) * 0.04 if hi > lo else 0.5
        setter((lo, hi) if limits is not None else (lo - pad, hi + pad))
    ax.grid(alpha=0.2)
    if panel:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xlabel("x / stored coordinate 0")
        ax.set_ylabel("y / stored coordinate 1")
    if not np.any(drawn):
        ax.text(0.5, 0.03, "No positive selected gates", transform=ax.transAxes,
                ha="center", fontsize=8)


def _save(fig, destination, stem, timestamp, settings):
    paths = []
    try:
        for extension in settings.formats:
            path = destination / f"{stem}_{timestamp}.{extension}"
            fig.savefig(path, bbox_inches="tight", dpi=settings.dpi)
            paths.append(str(path.resolve()))
    finally:
        plt.close(fig)
    return paths


def _colorbar(fig, axes, norm, normalization):
    if normalization["has_positive_scores"]:
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="plasma"), ax=axes, shrink=0.75,
                     label=f"Raw gate score (shared {normalization['scale']} color scale)")


def render_sequence(data, output_dir, settings=None, timestamp=None):
    """Render consecutive saved windows into output_dir/<timestamp>/<mode>.

    Returns a JSON-compatible manifest. Fixed-gate models always show all
    candidate gates and cannot accidentally imply a learned Top-K ranking.
    This API accepts SequenceData (or the same fields as a dict).
    """
    if isinstance(data, dict):
        data = SequenceData(**data)
    if isinstance(settings, dict):
        settings = PlotSettings(**settings)
    settings = settings or PlotSettings()
    picks, stride, contiguous = validate_sequence(data, settings)
    mode = "all_candidate" if data.score_kind == "fixed" else settings.mode
    threshold = data.threshold.get("value") if data.threshold.get("enabled") else None
    if mode == "saved_threshold":
        rule = str(data.threshold.get("rule", ""))
        if data.threshold.get("calibration_split") != "val" and not rule.startswith("validation") and rule not in {"explicit", "explicit_absolute_threshold", "predeclared_absolute"}:
            raise ValueError("saved_threshold requires saved validation/fixed provenance; no test quantile is fitted")
    masks, records = [], []
    step = np.timedelta64(int(data.protocol["step_minutes"]), "m")
    denominator = len(data.coords) * (len(data.coords) - 1)
    E = data.scores.shape[1]
    for ordinal, row in enumerate(picks):
        keep, drawn, value = select_mask(data.scores[row], mode, settings.fraction, settings.quantile, threshold)
        masks.append(keep)
        target_start, target_end = np.asarray(data.target_times[row]).astype("datetime64[ns]")[[0, -1]]
        input_end = target_start - step
        input_start = input_end - (int(data.protocol["W"]) - 1) * step
        records.append({"sequence_index": ordinal, "split_window_index": int(row), "window_id": int(data.window_ids[row]),
                        "input_start_t": int(data.window_ids[row]) - int(data.protocol["W"]) + 1,
                        "input_end_t": int(data.window_ids[row]), "target_start_t": int(data.window_ids[row]) + 1,
                        "target_end_t": int(data.window_ids[row]) + data.target_times.shape[1],
                        "input_start_time": _time_text(input_start), "input_end_time": _time_text(input_end),
                        "target_start_time": _time_text(target_start), "target_end_time": _time_text(target_end),
                        "selected_edges": int(keep.sum()), "positive_drawn_edges": int(drawn.sum()),
                        "candidate_edges": E, "selection_threshold": value,
                        "selected_candidate_density": float(keep.sum() / E) if E else 0.0,
                        "selected_directed_density": float(keep.sum() / denominator) if denominator else 0.0,
                        "positive_candidate_density": float(drawn.sum() / E) if E else 0.0,
                        "positive_directed_density": float(drawn.sum() / denominator) if denominator else 0.0})
    norm, normalization = shared_normalization(data.scores[picks], settings)
    timestamp = timestamp or timestamp_now()
    if any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in timestamp):
        raise ValueError("timestamp may contain only letters, digits, underscores, and hyphens")
    destination = Path(output_dir).resolve() / timestamp / mode
    destination.mkdir(parents=True, exist_ok=False)
    paths, singles = [], []
    graph_name = "Fixed candidate graph (all-one gates; no learned ranking)" if mode == "all_candidate" else {
        "topk": f"Exact Top-{100 * settings.fraction:g}% raw gate ranking",
        "saved_threshold": f"Saved validation/fixed threshold = {threshold:g}" if threshold is not None else "Saved threshold",
        "window_quantile": f"Per-window quantile q={settings.quantile:g} (legacy exploratory view)"}[mode]

    def label(record, compact=False):
        first = f"window[{record['split_window_index']}], input end ID={record['window_id']}"
        line = f"target {record['target_start_time']} - {record['target_end_time']}"
        history = f"input {record['input_start_time']} - {record['input_end_time']}"
        counts = f"selected={record['selected_edges']}/{E}; positive drawn={record['positive_drawn_edges']}"
        return f"{first}\n{line}\n{history}\n{counts}"

    # Candidate drawing is directed and truthful about geometry, independent of
    # learned scores. It is a reference, not an extracted intention graph.
    fig, ax = plt.subplots(figsize=(settings.figure_width, settings.figure_height))
    candidate_data = SequenceData(**{**data.__dict__, "scores": np.ones((1, E))})
    _draw(ax, candidate_data, 0, np.ones(E, dtype=bool), colors.Normalize(vmin=0, vmax=1), settings)
    ax.set_title(f"Candidate directed graph G0: N={len(data.coords)}, E={E}")
    paths += _save(fig, destination, "candidate_directed_graph", timestamp, settings)
    for row, keep, record in zip(picks, masks, records):
        fig, ax = plt.subplots(figsize=(settings.figure_width, settings.figure_height))
        _draw(ax, data, int(row), keep, norm, settings)
        ax.set_title(f"{data.model} / {data.seed} / {data.split}\n{graph_name}\n{label(record)}", fontsize=9)
        _colorbar(fig, ax, norm, normalization)
        frame_paths = _save(fig, destination, f"intent_graph_window_{record['sequence_index']:04d}", timestamp, settings)
        record["figure_paths"] = frame_paths
        singles.extend(frame_paths)
    paths.extend(singles)
    n_panel = min(settings.panel_count, len(picks))
    if n_panel:
        ncols = min(5, n_panel)
        nrows = int(np.ceil(n_panel / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 4.4 * nrows), squeeze=False)
        fig.subplots_adjust(top=0.84, hspace=0.48, wspace=0.18)
        for ordinal, ax in enumerate(axes.ravel()):
            if ordinal >= n_panel:
                ax.axis("off")
                continue
            _draw(ax, data, int(picks[ordinal]), masks[ordinal], norm, settings, panel=True)
            ax.set_title(label(records[ordinal], compact=True), fontsize=7)
        fig.suptitle(f"{data.model} / {data.seed} / {data.split}: first {n_panel} adjacent windows\n{graph_name}", fontsize=12)
        _colorbar(fig, axes.ravel().tolist(), norm, normalization)
        paths += _save(fig, destination, f"intent_graphs_{n_panel}windows", timestamp, settings)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for key, name in (("selected_candidate_density", "selected / E0"),
                      ("selected_directed_density", "selected / N(N-1)"),
                      ("positive_candidate_density", "positive drawn / E0"),
                      ("positive_directed_density", "positive drawn / N(N-1)")):
        ax.plot(np.arange(len(records)), [record[key] for record in records], marker=".", label=name)
    tick_step = max(1, int(np.ceil(len(records) / 12)))
    ticks = np.arange(0, len(records), tick_step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([records[index]["target_start_time"] for index in ticks], rotation=40, ha="right", fontsize=7)
    ax.set_xlabel("Target start time (configured data clock)")
    ax.set_ylabel("Edge density")
    ax.set_title(f"{data.model} / {data.seed} / {data.split}: {len(records)} adjacent saved windows")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    paths += _save(fig, destination, "intent_graph_density", timestamp, settings)
    index_path = destination / f"window_index_{timestamp}.csv"
    csv_records = [{key: value for key, value in record.items() if key != "figure_paths"} for record in records]
    with index_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_records[0]))
        writer.writeheader()
        writer.writerows(csv_records)
    manifest = {"status": "completed", "timestamp": timestamp, "output_dir": str(destination),
                "source": data.source, "source_sha256": hashlib.sha256(Path(data.source).read_bytes()).hexdigest() if data.source else None,
                "model": data.model, "seed": data.seed, "split": data.split, "score_kind": data.score_kind,
                "protocol_id": data.protocol_id, "requested_settings": asdict(settings), "effective_mode": mode,
                "graph_label": graph_name, "threshold": data.threshold if mode == "saved_threshold" else None,
                "normalization": normalization, "consecutive_at_saved_stride": contiguous, "saved_stride": stride,
                "step_minutes": int(data.protocol["step_minutes"]), "W": int(data.protocol["W"]),
                "H": data.target_times.shape[1], "requested_windows": settings.count, "rendered_windows": len(picks),
                "timestamp_provenance": {
                    "target_times": "saved edge_scores.npz target_times array",
                    "input_times": "derived from saved first target time, W, and step_minutes",
                    "configured_start_time": data.protocol.get("start_time"),
                    "real_world_clock_independently_verified_by_plotter": False,
                    "window_id_semantics": "saved input-end row index; target indices start at window_id + 1"},
                "remaining_range_truncated": len(picks) < settings.count, "window_index_csv": str(index_path),
                "figures": paths, "windows": records,
                "interpretation": "Raw gates are not calibrated causal effects or contribution probabilities. "
                                  "Top-K density is fixed by budget; zero selected scores are suppressed only in rendering. "
                                  "Threshold density can vary. Timestamps inherit the saved configured data clock. "
                                  "Color normalization is shared within this sequence, not automatically across models/seeds."}
    manifest_path = destination / f"manifest_{timestamp}.json"
    manifest["manifest_path"] = str(manifest_path)
    _write_json(manifest_path, manifest)
    return manifest
