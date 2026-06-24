"""I/O для golden-фикстур (REBUILD_SPEC §6).

JSON-фикстура: человекочитаемый дамп входов, эталонных выходов, R-сниппета
происхождения и допусков сверки. numpy-массивы сериализуются как вложенные
списки; при загрузке числовые поля доступны как есть, а ``arr()`` приводит их к
``np.ndarray``.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

import numpy as np

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")

# Допуски сверки по величинам (REBUILD_SPEC §6, таблица).
TOLERANCES: Dict[str, float] = {
    "scheffe_coef": 1e-8,     # коэффициенты OLS — детерминированы
    "d_optimality": 1e-9,     # det/efficiency для фиксированной матрицы
    "i_optimality": 1e-9,
    "desirability": 1e-8,
    "gp": 1e-6,               # μ/σ при фиксированных гиперпараметрах
}


# ----------------------------------------------------------------------
def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def save_fixture(name: str, data: Dict[str, Any]) -> str:
    """Записать фикстуру ``<GOLDEN_DIR>/<name>.json``; вернуть путь."""
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    path = os.path.join(GOLDEN_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(data), fh, ensure_ascii=False, indent=2)
    return path


def load_fixture(name: str) -> Dict[str, Any]:
    """Загрузить фикстуру по имени (без расширения)."""
    path = os.path.join(GOLDEN_DIR, f"{name}.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def list_fixtures() -> list:
    if not os.path.isdir(GOLDEN_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(GOLDEN_DIR) if f.endswith(".json"))


def arr(x: Any) -> np.ndarray:
    """Привести вложенные списки фикстуры к float-массиву numpy."""
    return np.asarray(x, dtype=float)
