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

from math import factorial
from typing import List, Optional, Tuple

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


# ----------------------------------------------------------------------
# §13.5 АНАЛИТИЧЕСКИЕ моменты (закрытая форма) — гейт golden-группы A
# ----------------------------------------------------------------------
# Терм = монома по составным координатам [x_0..x_{q-1}, z_0..z_{d-1}]; индекс <q —
# степень x_i, ≥q — степень z_{idx-q}. Произведение f_r·f_s — снова монома, и её
# матожидание по D = (станд. симплекс) × [0,1]^d факторизуется (блоки независимы):
#   E[∏ x_i^{a_i}] = (q-1)!·∏ a_i! / (q-1+Σa)!     (Дирихле(1) на стандартном симплексе)
#   E[∏ z_k^{b_k}] = ∏ 1/(b_k+1)                   (равномерно на кубе)
# Это даёт ДЕТЕРМИНИРОВАННУЮ M (в отличие от MC ``block_moment_matrix``), пригодную
# для сверки с эталоном при atol 1e-8 (§6/§13.10 шаг 3).


def _term_exponents(term: Tuple[int, ...], q: int, d: int
                    ) -> Tuple[List[int], List[int]]:
    """Кортеж глобальных индексов → (степени по x [q], степени по z [d])."""
    a = [0] * q
    b = [0] * d
    for g in term:
        if g < q:
            a[g] += 1
        else:
            b[g - q] += 1
    return a, b


def _simplex_moment(a: List[int], q: int) -> float:
    """E[∏ x_i^{a_i}] на стандартном симплексе S^{q-1} (Дирихле(1,…,1)).

    q==0 (нет mixture-блока) → множитель 1 (mixture-интеграла нет).
    """
    if q == 0:
        return 1.0
    s = sum(a)
    num = factorial(q - 1)
    for ai in a:
        num *= factorial(ai)
    return num / factorial(q - 1 + s)


def _cube_moment(b: List[int]) -> float:
    """E[∏ z_k^{b_k}] на [0,1]^d (равномерно). d==0 → 1."""
    val = 1.0
    for bk in b:
        val /= (bk + 1)
    return val


def analytic_moment_matrix(schema: ProjectSchema,
                           terms: Optional[ModelTerms] = None) -> np.ndarray:
    """Аналитическая M = E_D[f fᵀ] (закрытая форма) на произведении области.

    Симплекс берётся СТАНДАРТНЫЙ (псевдокомпоненты, §13.5) — границы L/U mixture не
    учитываются (моменты считаются в компонентном симплексе). Process — [0,1]^d.
    Совпадает с MC ``block_moment_matrix`` в пределе n→∞ (кросс-проверка в тестах),
    но детерминирована ⇒ годится для golden-A (atol 1e-8).
    """
    mt = terms if terms is not None else build_model_terms(schema)
    q, d = mt.q, mt.d
    exps = [_term_exponents(t, q, d) for t in mt.terms]
    p = len(exps)
    M = np.empty((p, p))
    for r in range(p):
        ar, br = exps[r]
        for s in range(p):
            as_, bs = exps[s]
            a = [ar[i] + as_[i] for i in range(q)]
            b = [br[k] + bs[k] for k in range(d)]
            M[r, s] = _simplex_moment(a, q) * _cube_moment(b)
    return M


