"""
core/block_geometry.py — §13.4 поблочный геометрический слой.

Полиморфизм по ТИПУ блока: каждая операция применяет правило своего блока к его
координатам составного вектора ``[x_0..x_{q-1}, z_0..z_{d-1}]``.

  * MIXTURE → симплекс-правила (проекция/центроид/вершины через ``SimplexRegion``).
  * PROCESS → бокс [0,1] в КОДЕ (clip к интервалу).

Жёсткое правило (§13.4): симплекс-проекция трогает **только** x; интервальный clip —
**только** z. Σx=1 проверяется **только** на mixture-блоке. «Вырожденные» режимы
(нет mixture / нет process) — частные случаи: при отсутствии блока его множитель в
декартовом произведении просто исчезает, без отдельной ветки кода.

Бит-в-бит инвариант: для mixture-only схемы ``build_candidate_pool`` совпадает с
``design.d_optimal.build_candidate_pool`` (тот же набор вершины+центроид+random при
том же seed) — декартово произведение с единственным mixture-множителем = он сам.
"""
from __future__ import annotations

from itertools import product as _iproduct
from typing import List, Optional

import numpy as np

from .schema import (MIXTURE, PROCESS, ProjectSchema, VariableBlock,
                     ordered_blocks, split_composite)

_TOL = 1e-9


# ----------------------------------------------------------------------
# Проекция / валидация (поблочно)
# ----------------------------------------------------------------------
def project(schema: ProjectSchema, vec: np.ndarray) -> np.ndarray:
    """Поблочная проекция составного вектора в допустимую область.

    MIXTURE → ``SimplexRegion.clip`` (границы + ренормировка Σ=1);
    PROCESS → clip к [0,1] (в коде).
    """
    parts = split_composite(schema, vec)
    out: List[np.ndarray] = []
    for b in ordered_blocks(schema):
        coords = np.asarray(parts[b.kind], float)
        if b.is_mixture:
            out.append(b.as_simplex_region().clip(coords))
        else:
            out.append(np.clip(coords, 0.0, 1.0))
    return np.concatenate(out) if out else np.empty(0)


def validate(schema: ProjectSchema, vec: np.ndarray, tol: float = 1e-6) -> bool:
    """Поблочная валидация: Σx=1 только на MIXTURE; [0,1] только на PROCESS."""
    parts = split_composite(schema, vec)
    for b in ordered_blocks(schema):
        coords = np.asarray(parts[b.kind], float)
        if b.is_mixture:
            if abs(coords.sum() - 1.0) > tol:
                return False
            lo = np.asarray(b.lower, float)
            hi = np.asarray(b.upper, float)
            if np.any(coords < lo - tol) or np.any(coords > hi + tol):
                return False
        else:
            if np.any(coords < -tol) or np.any(coords > 1.0 + tol):
                return False
    return True


# ----------------------------------------------------------------------
# Генераторы кандидатов по блокам
# ----------------------------------------------------------------------
def _box_structured(d: int) -> np.ndarray:
    """Структурные точки куба [0,1]^d: факторные углы (2^d при d≤5) + центр."""
    if d == 0:
        return np.empty((1, 0))
    if d <= 5:
        corners = np.array(list(_iproduct([0.0, 1.0], repeat=d)), dtype=float)
    else:  # для больших d углов слишком много — берём только границы по осям
        corners = np.vstack([np.zeros(d), np.ones(d)])
    center = np.full((1, d), 0.5)
    return np.vstack([corners, center])


def _block_structured(b: VariableBlock, include_vertices: bool,
                      include_centroid: bool) -> np.ndarray:
    if b.is_mixture:
        region = b.as_simplex_region()
        parts: List[np.ndarray] = []
        if include_vertices:
            V = region.extreme_vertices()
            if len(V):
                parts.append(V)
        if include_centroid:
            parts.append(region.centroid().reshape(1, -1))
        return np.vstack(parts) if parts else np.empty((0, b.size))
    return _box_structured(b.size)


def _block_random(b: VariableBlock, n: int, rng: np.random.Generator,
                  seed: Optional[int]) -> np.ndarray:
    if b.is_mixture:
        # тот же генератор, что в существующем пуле (бит-в-бит для mixture-only)
        return b.as_simplex_region().random_points(n, seed=seed)
    return rng.random((n, b.size))


def _cartesian(blocks_pts: List[np.ndarray], total_dim: int) -> np.ndarray:
    """Декартово произведение построчных множеств блоков (конкатенация координат)."""
    result: Optional[np.ndarray] = None
    for arr in blocks_pts:
        if arr.shape[0] == 0:
            continue
        if result is None:
            result = arr
        else:
            a = np.repeat(result, arr.shape[0], axis=0)
            b = np.tile(arr, (result.shape[0], 1))
            result = np.hstack([a, b])
    return result if result is not None else np.empty((0, total_dim))


def random_points(schema: ProjectSchema, n: int,
                  seed: Optional[int] = None) -> np.ndarray:
    """n случайных допустимых составных точек (mixture-random × process-uniform)."""
    rng = np.random.default_rng(seed)
    parts = [_block_random(b, n, rng, seed) for b in ordered_blocks(schema)]
    return np.hstack(parts) if parts else np.empty((n, 0))


def build_candidate_pool(schema: ProjectSchema, n_random: int = 500,
                         seed: Optional[int] = None,
                         include_vertices: bool = True,
                         include_centroid: bool = True) -> np.ndarray:
    """Пул кандидатов = (произведение структурных множеств) ∪ (n_random случайных).

    Для mixture-only сводится к существующему пулу (вершины+центроид+random при том
    же seed) — декартово произведение с единственным mixture-множителем = он сам.
    """
    blocks = ordered_blocks(schema)
    total_dim = sum(b.size for b in blocks)
    rng = np.random.default_rng(seed)

    structured = [_block_structured(b, include_vertices, include_centroid)
                  for b in blocks]
    prod = _cartesian(structured, total_dim)

    rand_parts = [_block_random(b, n_random, rng, seed) for b in blocks]
    rnd = np.hstack(rand_parts) if rand_parts and n_random > 0 else np.empty((0, total_dim))

    if len(prod) and len(rnd):
        pool = np.vstack([prod, rnd])
    elif len(prod):
        pool = prod
    else:
        pool = rnd

    # де-дупликация (тот же atol, что в design.d_optimal.build_candidate_pool)
    uniq: List[np.ndarray] = []
    for p in pool:
        if not any(np.allclose(p, u, atol=1e-7) for u in uniq):
            uniq.append(p)
    return np.array(uniq) if uniq else np.empty((0, total_dim))
