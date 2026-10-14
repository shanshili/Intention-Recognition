"""Per-node direct multi-horizon Ridge autoregression, in residual form."""

import time

import numpy as np


class RidgeAR:
    kind = "closed_form"
    name = "ridge_ar"
    capabilities = {"supports_ranking": False, "supports_intervention": False,
                    "score_kind": "none", "intervention_scope": None}

    def __init__(self, data, config=None, seed=0):
        self.config = dict(config or {})
        unknown = set(self.config) - {"alpha", "alpha_candidates"}
        if unknown:
            raise ValueError(f"Unknown ridge_ar model parameters: {sorted(unknown)}")
        self.seed = int(seed)
        self.horizon = int(data.protocol["H"])
        self.window = int(data.protocol["W"])
        self.n_nodes = int(data.features.shape[1])
        self.n_features = int(data.features.shape[2])
        self.coef = None
        self.intercept = None
        self.alpha = None
        self.parameter_count = self.n_nodes * self.horizon * (self.window * self.n_features + 1)

    @staticmethod
    def _collect(dataset, batch_size):
        histories, targets, masks, lasts = [], [], [], []
        for batch in dataset.iter_batches(batch_size):
            histories.append(batch["history"])
            targets.append(batch["target"])
            masks.append(batch["target_mask"])
            lasts.append(batch["last"])
        if not histories:
            raise ValueError("Ridge fitting needs at least one training window.")
        return tuple(np.concatenate(items, axis=0) for items in (histories, targets, masks, lasts))

    def _fit_alpha(self, history, target, mask, last, alpha):
        from sklearn.linear_model import Ridge

        features = history.transpose(0, 2, 1, 3).reshape(len(history), self.n_nodes, -1)
        residual = target - last[..., None]
        coef = np.empty((self.n_nodes, self.horizon, features.shape[-1]), dtype=np.float64)
        intercept = np.empty((self.n_nodes, self.horizon), dtype=np.float64)
        for node in range(self.n_nodes):
            # Most datasets share the observation mask across horizons. Fit
            # the direct horizons together so one SVD serves all outputs.
            node_mask = np.asarray(mask[:, node], dtype=bool)
            if np.all(node_mask == node_mask[:, :1]) and node_mask[:, 0].any():
                observed = node_mask[:, 0]
                estimator = Ridge(alpha=alpha, solver="svd", fit_intercept=True)
                estimator.fit(features[observed, node], residual[observed, node])
                coef[node] = estimator.coef_
                intercept[node] = estimator.intercept_
                continue
            for horizon in range(self.horizon):
                observed = np.asarray(mask[:, node, horizon], dtype=bool)
                if not observed.any():
                    raise ValueError(f"No observed training targets for node {node}, horizon {horizon + 1}.")
                estimator = Ridge(alpha=alpha, solver="svd", fit_intercept=True)
                estimator.fit(features[observed, node], residual[observed, node, horizon])
                coef[node, horizon] = estimator.coef_
                intercept[node, horizon] = estimator.intercept_
        return coef, intercept

    def _validation_mae(self, dataset, batch_size):
        error_sum, count = 0.0, 0
        for batch in dataset.iter_batches(batch_size):
            predicted = self.predict(batch)["prediction"]
            observed = np.asarray(batch["target_mask"], dtype=bool)
            if not np.isfinite(predicted[observed]).all():
                raise FloatingPointError("Ridge validation prediction contains nonfinite observed outputs.")
            error_sum += float(np.abs(predicted - batch["target"])[observed].sum(dtype=np.float64))
            count += int(observed.sum())
        if count == 0:
            raise ValueError("Ridge alpha selection needs observed validation targets.")
        return error_sum / count

    def fit(self, train_dataset, val_dataset, training_config=None):
        started = time.perf_counter()
        training_config = dict(training_config or {})
        batch_size = int(training_config.get("batch_size", 128))
        candidate_values = self.config.get("alpha_candidates")
        candidates = [self.config.get("alpha", 1.0)] if candidate_values is None else list(candidate_values)
        if not candidates or any(not np.isfinite(float(value)) or float(value) <= 0 for value in candidates):
            raise ValueError("Ridge alpha candidates must be finite and strictly positive.")
        history, target, mask, last = self._collect(train_dataset, batch_size)
        validation, best = [], None
        for alpha in candidates:
            self.coef, self.intercept = self._fit_alpha(history, target, mask, last, float(alpha))
            score = self._validation_mae(val_dataset, batch_size)
            validation.append({"alpha": float(alpha), "val_mae": score})
            if best is None or score < best[0]:
                best = (score, float(alpha), self.coef.copy(), self.intercept.copy())
        _, self.alpha, self.coef, self.intercept = best
        return {"kind": self.kind, "epochs": 0, "best_epoch": None,
                "selected_alpha": self.alpha, "val_mae": best[0], "history": validation,
                "fit_seconds": time.perf_counter() - started,
                "parameter_count": self.parameter_count}

    def predict(self, batch, edge_mask=None):
        if edge_mask is not None:
            raise ValueError("RidgeAR does not support edge interventions.")
        if self.coef is None:
            raise RuntimeError("Fit RidgeAR before prediction.")
        history = np.asarray(batch["history"])
        features = history.transpose(0, 2, 1, 3).reshape(len(history), self.n_nodes, -1)
        residual = np.einsum("bnk,nhk->bnh", features, self.coef) + self.intercept[None, ...]
        prediction = np.asarray(batch["last"])[..., None] + residual
        return {"prediction": prediction, "edge_scores": None, "effective_weight": None}

    def save(self, path):
        if self.coef is None:
            raise RuntimeError("Fit RidgeAR before saving.")
        np.savez_compressed(path, coef=self.coef, intercept=self.intercept,
                            alpha=np.asarray(self.alpha), seed=np.asarray(self.seed))
