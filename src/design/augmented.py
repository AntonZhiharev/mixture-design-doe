"""
design/augmented.py — §13.6/§13.7 блочный I-оптимальный ДОБОР (augmented design).

Оркестратор: собирает поблочную геометрию (pool, §13.4), моменты на произведении
области (W, §13.5) и ФИКСИРОВАННЫЕ строки уже измеренных точек, после чего вызывает
СУЩЕСТВУЮЩИЙ ``i_optimal.i_optimal_augment_sequential`` — переиспользуя его критерий
остановки §5.5 (sufficiency / rel_gain / budget / pool_exhausted). Второго критерия
остановки НЕ заводим (§13.6): единственное, что инъектируется в M5 — построитель
блочной модельной матрицы; геометрию M5 не знает (pool готов снаружи).

§13.7 (schema evolution): ``select_fixed_rows`` решает, какие существующие точки
переиспользуемы как фикс. Переиспользуема точка, содержащая координаты ВСЕХ блоков
целевой схемы. Новые ПЕРЕМЕННЫЕ (расширение пространства, напр. добавлен process-блок)
требуют миграционной политики — точки без их координат сюда не попадают (skipped).
Новые ЧЛЕНЫ из уже существующих переменных (напр. кросс x·z из старых x,z) считаются
из координат бесплатно — миграция не нужна (различение членов vs переменных — в
``schema.schema_diff_vars``).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..core import block_geometry as bg
from ..core.schema import DataPoint, ProjectSchema, composite_coords
from .block_model import ModelTerms, build_model_terms, model_matrix
from .block_moments import analytic_moment_matrix, block_moment_matrix
from .i_optimal import IAugmentResult, i_optimal_augment_sequential


def build_moments(schema: ProjectSchema, *, terms: Optional[ModelTerms] = None,
                  method: str = "analytic", n_mc: int = 5000,
                  seed: Optional[int] = None) -> np.ndarray:
    """Матрица моментов ``W = E_D[f fᵀ]`` на произведении области (§13.5).

    * ``method="analytic"`` (дефолт) — закрытая форма на СТАНДАРТНОМ симплексе ×
      ``[0,1]^d`` (``block_moments.analytic_moment_matrix``): детерминированно, без
      MC-смещения сэмплера (§13.11), границы ``L/U`` mixture НЕ учитываются (область
      интереса = компонентный симплекс, §13.5).
    * ``method="mc"`` — Monte-Carlo по ДОПУСТИМОЙ области (``block_moment_matrix``):
      для быстрых прикидок; сэмплер неравномерен (стянут к центроиду).
    """
    mt = terms if terms is not None else build_model_terms(schema)
    if method == "analytic":
        return analytic_moment_matrix(schema, terms=mt)
    if method == "mc":
        return block_moment_matrix(schema, n_mc=n_mc, seed=seed, terms=mt)
    raise ValueError(
        f"Unknown moment_method '{method}'. Use 'analytic' or 'mc'.")


def select_fixed_rows(points: Sequence[DataPoint], target_schema: ProjectSchema
                      ) -> Tuple[np.ndarray, List[DataPoint], List[DataPoint]]:
    """§13.7: отобрать переиспользуемые как ФИКС точки для добора в ``target_schema``.

    Возвращает ``(fixed_matrix (k×(q+d) составных координат), used, skipped)``.
    Точка попадает в ``used`` только если содержит координаты всех блоков целевой
    схемы (иначе — новая переменная без миграции → ``skipped``).
    """
    used: List[DataPoint] = []
    skipped: List[DataPoint] = []
    rows: List[np.ndarray] = []
    for pt in points:
        try:
            rows.append(composite_coords(target_schema, pt))
            used.append(pt)
        except (ValueError, KeyError):
            skipped.append(pt)
    dim = target_schema.n_mixture + target_schema.n_process
    fixed = np.vstack(rows) if rows else np.empty((0, dim))
    return fixed, used, skipped


def augmented_design(
        target_schema: ProjectSchema, existing_points: Sequence[DataPoint], *,
        terms: Optional[ModelTerms] = None, moment_method: str = "analytic",
        n_max: int = 50, margin: int = 12, min_total: Optional[int] = None,
        rel_tol: float = 0.03, n_random: int = 500, n_mc: int = 5000,
        ridge: float = 1e-8, seed: Optional[int] = None
        ) -> Tuple[IAugmentResult, List[DataPoint], List[DataPoint]]:
    """§13.6: блочный I-оптимальный добор к ``existing_points`` в ``target_schema``.

    pool и W строит блочный слой (§13.4/§13.5); M5 только копит ``EᵀE`` и отбирает
    по §5.5. Различие mixture-only / process-only / mixture-process живёт в наборе
    блоков схемы, а не в ветке кода. Возвращает ``(IAugmentResult, used, skipped)``.

    ``moment_method`` (§13.11): ``"analytic"`` (дефолт) — детерминированные моменты
    на СТАНДАРТНОМ симплексе × ``[0,1]^d`` (границы ``L/U`` mixture не учитываются,
    область интереса = компонентный симплекс, §13.5); ``"mc"`` — Monte-Carlo по
    допустимой области (``n_mc``/``seed``; для быстрых прикидок, сэмплер смещён).
    """
    mt = terms if terms is not None else build_model_terms(target_schema)
    fixed, used, skipped = select_fixed_rows(existing_points, target_schema)
    pool = bg.build_candidate_pool(target_schema, n_random=n_random, seed=seed)
    W = build_moments(target_schema, terms=mt, method=moment_method,
                      n_mc=n_mc, seed=seed)

    def mm(X: np.ndarray) -> np.ndarray:
        return model_matrix(target_schema, X, terms=mt)

    res = i_optimal_augment_sequential(
        existing=(fixed if len(fixed) else None), candidates=pool, moments=W,
        model_matrix_fn=mm, n_max=n_max, margin=margin, min_total=min_total,
        rel_tol=rel_tol, ridge=ridge, seed=seed)
    return res, used, skipped
