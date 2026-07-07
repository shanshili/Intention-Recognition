# -*- coding: utf-8 -*-
"""Utilidades de E/S: directorios con marca temporal y guardado de figuras.

Todos los nombres de fichero de salida incorporan un timestamp y se guardan
bajo un directorio de salida propio de esa ejecucion. Cada figura se persiste
simultaneamente en SVG, PNG, PDF y EPS.
"""
import os
import json
import datetime as _dt
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# El backend EPS/PostScript no soporta transparencia y emite un aviso por cada
# figura con alpha. Los EPS se generan igual (opacos); silenciamos ese log.
import logging as _logging
_logging.getLogger("matplotlib.backends.backend_ps").setLevel(_logging.ERROR)

# Formatos de imagen requeridos.
IMAGE_FORMATS: List[str] = ["svg", "png", "pdf", "eps"]


def make_timestamp() -> str:
    """Marca temporal legible y valida como nombre de fichero."""
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


class RunPaths:
    """Gestiona el directorio de salida de una ejecucion con timestamp.

    Estructura:
        outputs/run_<ts>/
            models/       -> modelo entrenado
            figures/      -> figuras (fusion, perdida, subgrafo despliegue)
            views_97steps/-> las 6 vistas para los pasos t seleccionados
            arrays/       -> tensores/matrices (.npz, .npy)
            logs/         -> config y metricas
    """

    def __init__(self, out_root: str = "outputs", timestamp: str = None):
        self.ts = timestamp or make_timestamp()
        self.root = os.path.join(out_root, f"run_{self.ts}")
        self.models = os.path.join(self.root, "models")
        self.figures = os.path.join(self.root, "figures")
        self.views = os.path.join(self.root, "views_97steps")
        self.arrays = os.path.join(self.root, "arrays")
        self.logs = os.path.join(self.root, "logs")
        for d in [self.root, self.models, self.figures,
                  self.views, self.arrays, self.logs]:
            os.makedirs(d, exist_ok=True)

    # --- helpers de nombres con timestamp ---------------------------------
    def stamped(self, base: str, ext: str) -> str:
        """Devuelve 'base_<ts>.<ext>'."""
        return f"{base}_{self.ts}.{ext}"

    def model_file(self, base: str = "ratcgf_model", ext: str = "pt") -> str:
        return os.path.join(self.models, self.stamped(base, ext))

    def array_file(self, base: str, ext: str = "npz") -> str:
        return os.path.join(self.arrays, self.stamped(base, ext))

    def log_file(self, base: str, ext: str = "json") -> str:
        return os.path.join(self.logs, self.stamped(base, ext))

    # --- guardado de figuras multi-formato --------------------------------
    def save_figure(self, fig, directory: str, base: str) -> Dict[str, str]:
        """Guarda `fig` en SVG/PNG/PDF/EPS con timestamp. Devuelve rutas."""
        os.makedirs(directory, exist_ok=True)
        paths = {}
        for fmt in IMAGE_FORMATS:
            path = os.path.join(directory, self.stamped(base, fmt))
            try:
                fig.savefig(path, format=fmt, bbox_inches="tight", dpi=200)
                paths[fmt] = path
            except Exception as exc:  # eps a veces falla con transparencias
                print(f"[warn] no se pudo guardar {fmt}: {exc}")
        plt.close(fig)
        return paths

    def dump_json(self, obj, base: str) -> str:
        path = self.log_file(base, "json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)
        return path
