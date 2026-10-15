"""Single experiment orchestration; models cannot override shared data targets."""

from __future__ import annotations

from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from .artifacts import save_multisplit_run, write_json
from .data import prepare_data
from .dataset import WindowDataset
from .evaluation import evaluate_model, validate_evaluation_config
from .models import MODEL_NAMES, NEURAL_MODEL_NAMES, canonical_model_name, make_model
from .protocol import protocol_hash


ROOT = Path(__file__).resolve().parents[1]


def environment_metadata():
    versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "matplotlib", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python_executable": sys.executable, "python_version": platform.python_version(),
            "platform": platform.platform(), "packages": versions}


def code_metadata():
    digest = hashlib.sha256()
    for source in sorted(ROOT.rglob("*.py")):
        relative = source.relative_to(ROOT)
        if any(part in {".venv", "__pycache__", "outputs", "tests"} for part in relative.parts):
            continue
        digest.update(relative.as_posix().encode())
        digest.update(source.read_bytes())
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                  check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    return {"git_revision": revision, "experiment_source_sha256": digest.hexdigest()}


def run_suite(protocol_path, experiments_path, models=None, seeds=None, data_overrides=None,
              training_overrides=None, evaluation_overrides=None, model_overrides=None,
              out=None, make_plots=True, prepare_only=False):
    started = time.perf_counter()
    experiments = json.loads(Path(experiments_path).read_text(encoding="utf-8"))
    if set(experiments) != {"models", "training", "evaluation"}:
        raise ValueError("Experiment config must contain only models, training, evaluation")
    configured_models = {}
    for requested_name, model_config in experiments["models"].items():
        name = canonical_model_name(requested_name)
        if name in configured_models:
            raise ValueError(f"Duplicate aliases for {name}; B6 and M0 must share one run")
        configured_models[name] = model_config
    selected = [canonical_model_name(name) for name in models] if models is not None else list(configured_models)
    if not selected or len(set(selected)) != len(selected) or any(name not in MODEL_NAMES for name in selected):
        raise ValueError(f"Choose unique model names from {MODEL_NAMES}")
    if any(name not in configured_models for name in selected):
        raise ValueError("Every selected model must have a model configuration")
    training = dict(experiments["training"])
    supplied_training = dict(training_overrides or {})
    if "loss" in supplied_training:
        training["loss"] = {**training.get("loss", {}), **supplied_training.pop("loss")}
    training.update(supplied_training)
    active_seeds = list(seeds) if seeds is not None else list(training.get("seeds", [0, 1, 2]))
    if not active_seeds or len(set(active_seeds)) != len(active_seeds) or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in active_seeds):
        raise ValueError("Training seeds must be unique nonnegative integers")
    training["seeds"] = active_seeds
    evaluation = validate_evaluation_config({**experiments["evaluation"], **(evaluation_overrides or {})})
    data = prepare_data(protocol_path, data_overrides)
    print(f"[data] protocol={data.protocol_id}; T={data.metadata['T']}; N={data.metadata['N']}; "
          f"E={data.metadata['E']}; windows={data.metadata['split_sizes']}", flush=True)
    destination = Path(out).resolve() if out is not None else ROOT / "outputs"
    suite = destination / datetime.now().strftime("suite_%Y%m%d_%H%M%S_%f")
    suite.mkdir(parents=True, exist_ok=False)
    write_json(suite / "protocol.json", data.protocol)
    write_json(suite / "data_metadata.json", data.metadata)
    np.savez_compressed(suite / "suite_data.npz", mean=data.mean, std=data.std,
                        coords=data.coords, node_ids=np.asarray(data.node_ids),
                        candidate_edges=data.graph["edges"], candidate_edge_features=data.graph["edge_feats"],
                        degree=data.graph["degree"], timestamps=data.timestamps,
                        train_window_id=data.splits["train"], val_window_id=data.splits["val"],
                        test_window_id=data.splits["test"], observed=data.observed,
                        protocol_id=np.asarray(data.protocol_id))
    identity = protocol_hash(evaluation)
    if prepare_only:
        summary = {"status": "prepared", "protocol_id": data.protocol_id,
                   "evaluation_signature": identity, "output_dir": str(suite), "data": data.metadata}
        write_json(suite / "suite_summary.json", summary)
        return summary
    environment, code = environment_metadata(), code_metadata()
    records, run_paths = [], []
    for name in selected:
        config = dict(configured_models[name])
        if name in NEURAL_MODEL_NAMES:
            overrides = dict(model_overrides or {})
            if name.startswith("m"):
                # Construction version defines its temporal encoder; --encoder
                # remains an override for the four original neural controls.
                overrides.pop("encoder", None)
            config.update(overrides)
        feature_set = config.get("input_features")
        if feature_set is not None:
            from .model_inputs import select_model_inputs
            model_data = select_model_inputs(data, feature_set)
        else:
            model_data = data
        datasets = {split: WindowDataset(model_data, split) for split in data.splits}
        model_seeds = active_seeds if name in NEURAL_MODEL_NAMES else [0]
        for seed in model_seeds:
            model = make_model(name, model_data, config, seed)
            print(f"[fit] model={name}; seed={seed}; kind={model.kind}", flush=True)
            fit = model.fit(datasets["train"], datasets["val"], training)
            print(f"[fit] finished model={name}; best_epoch={fit.get('best_epoch')}; "
                  f"val_MAE={fit.get('val_mae')}; device={fit.get('device', 'cpu')}", flush=True)
            # Evaluate the restored/selected checkpoint on every split.  The
            # validation pass calibrates the optional gate threshold; train and
            # test reuse that frozen specification and never select parameters.
            val_result = evaluate_model(model, datasets["val"], evaluation,
                                        validation=datasets["val"])
            threshold_spec = val_result.get("graph_metrics", {}).get("threshold")
            train_result = evaluate_model(model, datasets["train"], evaluation,
                                          threshold_spec=threshold_spec)
            test_result = evaluate_model(model, datasets["test"], evaluation,
                                         threshold_spec=threshold_spec)
            evaluations = {"train": train_result, "val": val_result, "test": test_result}
            result = test_result
            run_path = suite / name / f"seed_{seed}"
            run_training = {key: value for key, value in training.items() if key != "seeds"}
            signature = protocol_hash({"model": name, "model_config": config, "training": run_training})
            summary = {"status": "completed", "model": name, "seed": seed,
                       "protocol_id": data.protocol_id, "evaluation_signature": identity,
                       "run_config_signature": signature, "model_config": config,
                       "training_config": run_training, "evaluation_config": evaluation,
                       "fit": fit, "metrics": result["metrics"], "graph_metrics": result["graph_metrics"],
                       "capabilities": model.capabilities,
                       "construction": getattr(model, "construction_metadata", None),
                       "data": {**model_data.metadata, "step_minutes": data.protocol["step_minutes"]},
                       "environment": environment, "code_version": code}
            extension = {"gradient": ".pt", "closed_form": ".npz", "none": ".json"}[model.kind]
            run_path.mkdir(parents=True, exist_ok=False)
            model.save(run_path / ("model" + extension))
            saved = save_multisplit_run(run_path, model_data, model, fit, evaluations, summary,
                                        make_plots=make_plots)
            records.append({"model": name, "seed": seed, "run": saved["output_dir"],
                            "mae": result["metrics"]["mae"], "skill": result["metrics"]["skill"],
                            "device": fit.get("device", "cpu"),
                            "split_metrics": {s: evaluations[s]["metrics"] for s in evaluations}})
            run_paths.append(run_path)
            from compare_runs import compare_runs
            from .reporting import write_reports
            write_reports(compare_runs(run_paths), suite)
            skill = result["metrics"]["skill"]
            skill_text = "N/A (zero persistence error)" if skill is None else f"{skill:.6g}"
            print(f"[test] {name} seed={seed}: MAE={result['metrics']['mae']:.6g}; "
                  f"Skill={skill_text}; output={run_path}", flush=True)
            # Preserve incremental suite records if later models fail.
            write_json(suite / "suite_summary.json", {"status": "running", "protocol_id": data.protocol_id,
                       "evaluation_signature": identity, "output_dir": str(suite), "runs": records})
    from compare_runs import compare_runs
    comparison = compare_runs(run_paths)
    from experiment.reporting import write_reports
    write_reports(comparison, suite)
    summary = {"status": "completed", "protocol_id": data.protocol_id,
               "evaluation_signature": identity, "output_dir": str(suite), "runs": records,
               "total_seconds": time.perf_counter() - started}
    write_json(suite / "suite_summary.json", summary)
    print(f"[done] {suite}", flush=True)
    return summary
