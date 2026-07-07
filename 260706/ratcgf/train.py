# -*- coding: utf-8 -*-
r"""Bucle de entrenamiento del modelo RA-TCGF.

Objetivo auto-supervisado: predecir el flujo del siguiente instante a partir de
la representacion de nodo fusionada h_{i,D}^t (utilidad de prediccion U_pred),
con una regularizacion L1 de dispersion sobre S_D^t.
"""
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F


class Trainer:
    def __init__(self, model, builder, cfg, device):
        self.model = model.to(device)
        self.builder = builder
        self.cfg = cfg
        self.device = device
        self.opt = torch.optim.Adam(
            model.parameters(), lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay)
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_pred": [], "train_sparse": [],
        }

    def _loss_for(self, sample) -> Dict:
        out = self.model(sample)
        pred_loss = F.mse_loss(out["y_hat"], sample["target"])
        sparse = out["S_D"].abs().mean()
        loss = pred_loss + self.cfg.train.sparsity_lambda * sparse
        return {"loss": loss, "pred": pred_loss, "sparse": sparse}

    def _run_epoch(self, centers: List[int], train: bool) -> Dict:
        self.model.train(train)
        bs = self.cfg.train.batch_size
        losses, preds, sparses = [], [], []
        if train:
            np.random.shuffle(centers)
        self.opt.zero_grad()
        for k, c in enumerate(centers):
            sample = self.builder.build(c)
            res = self._loss_for(sample)
            if train:
                (res["loss"] / bs).backward()
                if (k + 1) % bs == 0 or (k + 1) == len(centers):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.train.grad_clip)
                    self.opt.step()
                    self.opt.zero_grad()
            losses.append(float(res["loss"].detach()))
            preds.append(float(res["pred"].detach()))
            sparses.append(float(res["sparse"].detach()))
        return {"loss": float(np.mean(losses)),
                "pred": float(np.mean(preds)),
                "sparse": float(np.mean(sparses))}

    def fit(self):
        train_c, val_c = self.builder.split(self.cfg.data.train_ratio)
        best_val = float("inf")
        best_state = None
        for epoch in range(self.cfg.train.epochs):
            tr = self._run_epoch(list(train_c), train=True)
            with torch.no_grad():
                va = self._run_epoch(list(val_c), train=False)
            self.history["train_loss"].append(tr["loss"])
            self.history["val_loss"].append(va["loss"])
            self.history["train_pred"].append(tr["pred"])
            self.history["train_sparse"].append(tr["sparse"])
            if va["loss"] < best_val:
                best_val = va["loss"]
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()}
            print(f"[epoch {epoch+1:03d}/{self.cfg.train.epochs}] "
                  f"train={tr['loss']:.4f} (pred={tr['pred']:.4f}) "
                  f"val={va['loss']:.4f}")
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return {"history": self.history, "best_val": best_val,
                "train_centers": list(train_c), "val_centers": list(val_c)}


# ---------------------------------------------------------------------------
# Funciones envoltorio a nivel de modulo (API funcional)
# ---------------------------------------------------------------------------
def train_model(model, builder, cfg, device):
    """Entrena `model` y devuelve (model_entrenado, result).

    Envoltorio funcional de la clase Trainer para uso directo:
        model, result = train_model(model, builder, cfg, device)
    """
    trainer = Trainer(model, builder, cfg, device)
    result = trainer.fit()
    return trainer.model, result


def save_model(model, cfg, result, paths,
               base: str = "ratcgf_model", ext: str = "pt") -> str:
    """Guarda el modelo entrenado (pesos + config + historia) con timestamp.

    Devuelve la ruta del fichero .pt generado.
    """
    path = paths.model_file(base, ext)
    torch.save({
        "state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "view_names": getattr(model, "view_names", None),
        "history": (result or {}).get("history"),
        "best_val": (result or {}).get("best_val"),
    }, path)
    return path
