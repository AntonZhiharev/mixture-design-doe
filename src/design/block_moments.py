"""
design/block_moments.py — §13.5 моменты на ПРОИЗВЕДЕНИИ области + I-критерий.

Матрица моментов ``W = E_D[f(x) f(x)ᵀ]`` по объединённой области
``D = simplex × [0,1]^d`` (любой множитель может отсутствовать). Оценивается
Monte-Carlo — той же конвенцией, что существующий ``i_optimal.region_moment_matrix``
(MC по ДОПУСТИМОЙ области), но по СОСТАВНЫМ координатам блоков:

  * mixture-координаты сэмплируются по выполнимому симплексу (``SimplexRegion``);
  * process-координаты — равномерно по кубу [0,1]^d (в коде);
  * блоки независимы ⇒ кросс-моменты факторизуются автоматически (произведение
    интегралов) — отдельно их считать не нужно, MC по произведению уже это даёт.

Бит-в-бит инвариант (регресс §13.9): для mixture-only схемы при тех же ``n_mc`` и
``seed`` ``block_moment_matrix`` совпадает с ``i_optimal.region_moment_matrix``
(одна и та же выборка ``region.random_points`` и одна и та же модельная матрица,
т.к. ``model_matrix`` mixture-only == ``scheffe_matrix``).

I-критерий — ТА ЖЕ свёртка, что в M5 (``i_optimal``):
``phi_I = trace((XᵀX + ridge·I)⁻¹ W)``, меньше — точнее прогноз.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.block_geometry import random_points
from ..core.schema import ProjectSchema
from .block_model import ModelTerms, build_model_terms, model_matrix


def block_moment_matrix(schema: ProjectSchema, n_mc: int = 5000,
                        seed: Optional[int] = None,
                        terms: Optional[ModelTerms] = None) -> np.ndarray:
    """MC-оценка ``W = E_D[f fᵀ]`` по произведению области (симплекс × норм. куб).

    ``terms`` (опц.) — q_eff-редуцированный базис; по умолчанию полная модель схемы.
    """
    pts = random_points(schema, n_mc, seed=seed)        # составные координаты
    mt = terms if terms is not None else build_model_terms(schema)
    F = model_matrix(schema, pts, terms=mt)
    return (F.T @ F) / F.shape[0]


def i_value(X_model: np.ndarray, W: np.ndarray, ridge: float = 1e-8) -> float:
    """``phi_I = trace((XᵀX + ridge·I)⁻¹ W)`` — средняя дисперсия прогноза.

    ``X_model`` — модельная матрица (n × p); ``W`` — матрица моментов (p × p).
    Та же конвенция и тот же ridge, что в ``i_optimal`` (M5).
    """
    X = np.atleast_2d(np.asarray(X_model, dtype=float))
    p = X.shape[1]
    W = np.asarray(W, dtype=float)
    if W.shape != (p, p):
        raise ValueError(f"W должна быть {p}x{p}, дано {W.shape}.")
    XtX = X.T @ X + np.eye(p) * ridge
    try:
        inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return float("inf")
    return float(np.trace(inv @ W))
