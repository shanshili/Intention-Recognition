"""Last-observation baseline with no fitting or spatial information."""

import json
from pathlib import Path

import numpy as np


class Persistence:
    kind = "none"
    name = "persistence"
    capabilities = {"supports_ranking": False, "supports_intervention": False,
                    "score_kind": "none", "intervention_scope": None}
    parameter_count = 0

    def __init__(self, data, config=None, seed=0):
        self.config = dict(config or {})
        if self.config:
            raise ValueError(f"Persistence accepts no model parameters: {sorted(self.config)}")
        self.seed = int(seed)
        self.horizon = int(data.protocol["H"])

    def fit(self, train_dataset, val_dataset, training_config=None):
        return {"kind": self.kind, "epochs": 0, "best_epoch": None,
                "history": [], "fit_seconds": 0.0, "parameter_count": 0}

    def predict(self, batch, edge_mask=None):
        if edge_mask is not None:
            raise ValueError("Persistence does not support edge interventions.")
        prediction = np.repeat(np.asarray(batch["last"])[..., None], self.horizon, axis=-1)
        return {"prediction": prediction, "edge_scores": None, "effective_weight": None}

    def save(self, path):
        Path(path).write_text(json.dumps({"model": self.name, "horizon": self.horizon,
                                         "config": self.config}), encoding="utf-8")
