from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_csv(path: str | Path, fieldnames: list[str], row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_parameter_bundle(
    base: str | Path,
    parameters: Iterable[np.ndarray],
    keys: list[str],
    metadata: dict | None = None,
) -> None:
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    arrays = list(parameters)
    np.savez(str(base) + ".npz", **{f"arr_{i}": p for i, p in enumerate(arrays)})
    with open(str(base) + "_keys.json", "w", encoding="utf-8") as f:
        json.dump(keys, f, ensure_ascii=False, indent=2)
    if metadata is not None:
        with open(str(base) + "_meta.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_parameter_bundle(base: str | Path) -> tuple[list[str], list[np.ndarray], dict]:
    base = Path(base)
    npz_path = Path(str(base) + ".npz")
    keys_path = Path(str(base) + "_keys.json")
    meta_path = Path(str(base) + "_meta.json")
    if not npz_path.is_file() or not keys_path.is_file():
        raise FileNotFoundError(f"Fichiers manquants pour {base}")
    with keys_path.open("r", encoding="utf-8") as f:
        keys = json.load(f)
    data = np.load(npz_path)
    params = [data[f"arr_{i}"] for i in range(len(keys))]
    metadata = {}
    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    return keys, params, metadata


def average_parameters(
    parameter_sets: list[list[np.ndarray]], weights: list[float] | None = None
) -> list[np.ndarray]:
    if not parameter_sets:
        raise ValueError("Aucun jeu de paramètres à agréger")
    n_params = len(parameter_sets[0])
    if any(len(p) != n_params for p in parameter_sets):
        raise ValueError("Les modèles n'ont pas le même nombre de tenseurs")
    for j in range(n_params):
        ref_shape = parameter_sets[0][j].shape
        if any(p[j].shape != ref_shape for p in parameter_sets):
            raise ValueError(f"Forme incompatible pour le tenseur {j}")

    if weights is None:
        weights = [1.0] * len(parameter_sets)
    if len(weights) != len(parameter_sets):
        raise ValueError("Nombre de poids incompatible")
    w = np.asarray(weights, dtype=np.float64)
    if np.any(w < 0) or float(w.sum()) <= 0:
        raise ValueError("Les poids doivent être positifs et de somme non nulle")
    w = w / w.sum()

    averaged: list[np.ndarray] = []
    for j in range(n_params):
        acc = np.zeros_like(parameter_sets[0][j], dtype=np.float64)
        for wi, params in zip(w, parameter_sets):
            acc += wi * params[j]
        averaged.append(acc.astype(parameter_sets[0][j].dtype, copy=False))
    return averaged


def blend_parameters(
    global_params: list[np.ndarray], client_params: list[np.ndarray], alpha: float
) -> list[np.ndarray]:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha doit appartenir à ]0, 1]")
    if len(global_params) != len(client_params):
        raise ValueError("Paramètres incompatibles")
    return [
        ((1.0 - alpha) * g + alpha * c).astype(g.dtype, copy=False)
        for g, c in zip(global_params, client_params)
    ]
