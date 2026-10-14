"""Immutable data/evaluation protocol and content-based comparison identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def load_protocol(path: str | Path, overrides: dict | None = None) -> dict:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    if overrides:
        protocol.update(overrides)
    defaults = {
        "missing_mask_path": None,
        "max_steps": None,
        "start_time": "2025-01-01 00:00:00",
        "step_minutes": 5,
        "W": 12,
        "H": 3,
        "train_stride": 3,
        "eval_stride": 1,
        "split_ratios": [0.6, 0.2, 0.2],
        "features": ["level"],
        "candidate_graph": {"k": 4, "tau_max": 4},
        "known_missing_intervals": [],
        "input_fill": "forward_train_mean",
        "target_missing_policy": "observed_only",
        "std_epsilon": 1e-6,
        "evaluation_version": "intent_graph_v1",
    }
    for key, value in defaults.items():
        protocol.setdefault(key, value)
    for name in ("flow_path", "coord_path", "missing_mask_path"):
        value = protocol.get(name)
        if name != "missing_mask_path" and not value:
            raise ValueError(f"Missing protocol field: {name}")
        if value:
            candidate = Path(value)
            protocol[name] = str((PROJECT_ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve())
    for key in ("W", "H", "train_stride", "eval_stride", "step_minutes"):
        value = protocol[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if protocol["max_steps"] is not None:
        value = protocol["max_steps"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("max_steps must be null or a positive integer")
    ratios = protocol["split_ratios"]
    if len(ratios) != 3 or any(v <= 0 for v in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("split_ratios must contain three positive ratios summing to one")
    selected = protocol["features"]
    if not selected or selected[0] != "level" or len(set(selected)) != len(selected):
        raise ValueError("features must be unique and start with level")
    if not set(selected) <= {"level", "delta", "time", "periodic_residual"}:
        raise ValueError("Unknown feature selection")
    if protocol["input_fill"] != "forward_train_mean" or protocol["target_missing_policy"] != "observed_only":
        raise ValueError("Only causal forward filling and original observed targets are supported")
    if protocol["std_epsilon"] <= 0:
        raise ValueError("std_epsilon must be positive")
    graph = protocol["candidate_graph"]
    for key in ("k", "tau_max"):
        if isinstance(graph.get(key), bool) or not isinstance(graph.get(key), int) or graph[key] < (0 if key == "k" else 1):
            raise ValueError(f"candidate_graph.{key} has an invalid value")
    protocol["protocol_source"] = str(source)
    return protocol


def assert_comparable(left: dict, right: dict) -> None:
    if left.get("protocol_id") != right.get("protocol_id"):
        raise ValueError("Results have different data/evaluation protocols; direct comparison is invalid")
