"""
core/simplex.py — M1: constrained mixture geometry.

A ``SimplexRegion`` describes a constrained mixture design space:

    L_i <= x_i <= U_i ,   sum_i x_i = 1 .

Capabilities (REBUILD_SPEC module M1):
  * extreme_vertices()  — McLean-Anderson extreme-vertices enumeration.
  * to_pseudo / from_pseudo  — L-pseudocomponent transform.
  * to_ilr / from_ilr        — isometric log-ratio (q-1 coords, avoids Σx=1 degeneracy).
  * is_feasible / clip        — bound + simplex checks.
  * random_points / centroid  — candidate generation helpers.

R reference for vertices: ``mixexp::Xvert``.
"""
from __future__ import annotations

import warnings
from itertools import product
from typing import List, Sequence, Tuple

import numpy as np

_TOL = 1e-9


class SimplexRegion:
    """Constrained mixture region with per-component lower/upper bounds."""

    def __init__(self, lower: Sequence[float] | None = None,
                 upper: Sequence[float] | None = None,
                 q: int | None = None,
                 names: Sequence[str] | None = None):
        if lower is not None:
            self.lower = np.asarray(lower, dtype=float)
            self.q = len(self.lower)
        elif q is not None:
            self.q = int(q)
            self.lower = np.zeros(self.q)
        else:
            raise ValueError("Provide either `lower` bounds or `q`.")

        self.upper = (np.asarray(upper, dtype=float)
                      if upper is not None else np.ones(self.q))

        if len(self.upper) != self.q:
            raise ValueError("lower and upper must have the same length.")
        self.names = (list(names) if names is not None
                      else [chr(ord("A") + i) for i in range(self.q)])
        self._validate()

    # ------------------------------------------------------------------
    def _validate(self) -> None:
        if np.any(self.lower < -_TOL):
            raise ValueError("Lower bounds must be >= 0.")
        if np.any(self.upper > 1 + _TOL):
            raise ValueError("Upper bounds must be <= 1.")
        if np.any(self.lower > self.upper + _TOL):
            raise ValueError("Each lower bound must be <= its upper bound.")
        if self.lower.sum() > 1 + _TOL:
            raise ValueError(f"Sum of lower bounds = {self.lower.sum():.4f} > 1 (infeasible).")
        if self.upper.sum() < 1 - _TOL:
            raise ValueError(f"Sum of upper bounds = {self.upper.sum():.4f} < 1 (infeasible).")

    @property
    def L_sum(self) -> float:
        return float(self.lower.sum())

    # ------------------------------------------------------------------
    # Feasibility
    # ------------------------------------------------------------------
    def is_feasible(self, x: Sequence[float], tol: float = 1e-6) -> bool:
        x = np.asarray(x, dtype=float)
        if abs(x.sum() - 1.0) > tol:
            return False
        return bool(np.all(x >= self.lower - tol) and np.all(x <= self.upper + tol))

    def clip(self, x: Sequence[float]) -> np.ndarray:
        """Clip to bounds and renormalise to sum 1 (best-effort projection)."""
        x = np.clip(np.asarray(x, dtype=float), self.lower, self.upper)
        s = x.sum()
        if s <= 0:
            return self.lower + (1 - self.L_sum) / self.q
        return x / s

    # ------------------------------------------------------------------
    # Extreme vertices (McLean-Anderson)
    # ------------------------------------------------------------------
    def extreme_vertices(self, dedup_tol: float = 1e-6) -> np.ndarray:
        """Enumerate extreme vertices of the constrained simplex.

        Method: for each choice of the single "free" component, fix the other
        q-1 components at either their lower or upper bound, solve the free one
        from Σx=1, and keep the point if all bounds are satisfied.
        """
        verts: List[np.ndarray] = []
        idx_all = range(self.q)
        for free in idx_all:
            others = [i for i in idx_all if i != free]
            for combo in product([0, 1], repeat=self.q - 1):  # 0=lower, 1=upper
                x = np.empty(self.q)
                for i, bit in zip(others, combo):
                    x[i] = self.upper[i] if bit else self.lower[i]
                x[free] = 1.0 - sum(x[i] for i in others)
                if (self.lower[free] - _TOL) <= x[free] <= (self.upper[free] + _TOL):
                    if self.is_feasible(x, tol=1e-6):
                        verts.append(x)

        if not verts:
            return np.empty((0, self.q))
        V = np.array(verts)
        return self._dedup(V, dedup_tol)

    @staticmethod
    def _dedup(points: np.ndarray, tol: float) -> np.ndarray:
        unique: List[np.ndarray] = []
        for p in points:
            if not any(np.allclose(p, u, atol=tol) for u in unique):
                unique.append(p)
        return np.array(unique)

    # ------------------------------------------------------------------
    # Centroids
    # ------------------------------------------------------------------
    def centroid(self) -> np.ndarray:
        """Centroid of the extreme vertices (feasible interior point)."""
        V = self.extreme_vertices()
        if len(V) == 0:
            return self.clip(np.full(self.q, 1.0 / self.q))
        return V.mean(axis=0)

    # ------------------------------------------------------------------
    # L-pseudocomponents
    # ------------------------------------------------------------------
    def to_pseudo(self, x: Sequence[float]) -> np.ndarray:
        """x -> w :  w_i = (x_i - L_i) / (1 - ΣL)."""
        x = np.asarray(x, dtype=float)
        denom = 1.0 - self.L_sum
        if denom <= _TOL:
            return x.copy()
        return (x - self.lower) / denom

    def from_pseudo(self, w: Sequence[float]) -> np.ndarray:
        """w -> x :  x_i = L_i + w_i (1 - ΣL)."""
        w = np.asarray(w, dtype=float)
        return self.lower + w * (1.0 - self.L_sum)

    # ------------------------------------------------------------------
    # ILR (isometric log-ratio) — q-1 unconstrained coordinates
    # ------------------------------------------------------------------
    @staticmethod
    def _ilr_basis(D: int) -> np.ndarray:
        """Orthonormal ILR contrast basis V (D x (D-1)), Egozcue construction."""
        V = np.zeros((D, D - 1))
        for i in range(1, D):
            norm = np.sqrt(i / (i + 1.0))
            V[:i, i - 1] = norm / i
            V[i, i - 1] = -norm
        return V

    def to_ilr(self, x: Sequence[float], eps: float = 1e-12) -> np.ndarray:
        """Map composition x (q parts) -> ILR coordinates z (q-1)."""
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        X = np.atleast_2d(x)
        X = np.clip(X, eps, None)
        X = X / X.sum(axis=1, keepdims=True)
        logX = np.log(X)
        clr = logX - logX.mean(axis=1, keepdims=True)
        Z = clr @ self._ilr_basis(self.q)
        return Z[0] if single else Z

    def from_ilr(self, z: Sequence[float]) -> np.ndarray:
        """Map ILR coordinates z (q-1) -> composition x (q parts)."""
        z = np.asarray(z, dtype=float)
        single = z.ndim == 1
        Z = np.atleast_2d(z)
        clr = Z @ self._ilr_basis(self.q).T
        X = np.exp(clr)
        X = X / X.sum(axis=1, keepdims=True)
        return X[0] if single else X

    # ------------------------------------------------------------------
    # Random feasible points (rejection sampling in pseudocomponent space)
    # ------------------------------------------------------------------
    def random_points(self, n: int, seed: int | None = None,
                      max_tries: int = 10000,
                      groups: Sequence[Sequence[int]] | None = None
                      ) -> np.ndarray:
        """Generate ``n`` random feasible interior points.

        Основной путь — rejection sampling в пространстве псевдокомпонентов.
        Если из-за узких границ приёмка мала и точек не хватает, остаток
        добирается ВЫПУКЛЫМИ КОМБИНАЦИЯМИ крайних вершин (всегда допустимы,
        область — их выпуклая оболочка) с предупреждением. Прежний тихий
        fallback «забить центроидом» вырождал пул кандидатов (одинаковые
        точки) и маскировал проблему узкой области.

        ``groups`` (iter31) — НЕПЕРЕСЕКАЮЩИЕСЯ группы индексов компонентов
        («функциональные ниши», mixture-of-mixtures). Включает
        СТРАТИФИЦИРОВАННОЕ сэмплирование по сумме каждой группы: сумма Σ_g
        тянется РАВНОМЕРНО на допустимом интервале, затем раскладывается
        внутри группы conditional narrowing'ом (без rejection). Зачем: при
        равномерной мере на политопе сумма группы из k компонентов
        концентрируется (~Beta(k, q−k)) и края её диапазона недостижимы —
        главная ось «доза ниши» не покрывается планом.

        ОСОЗНАННЫЙ ВЫБОР МЕРЫ: со стратификацией распределение даёт
        равномерную МАРГИНАЛЬ по суммам групп, но НЕравномерную совокупную
        меру на области (узкие условные слои получают повышенный вес). Для
        DoE/GP покрытие физически значимых осей важнее равномерности объёма.
        НЕ «исправлять» обратно на равномерную меру — это регресс покрытия.
        ``groups=None``/пусто → прежнее поведение бит-в-бит.
        """
        if groups:
            return self._random_points_grouped(n, seed, groups)
        rng = np.random.default_rng(seed)
        out: List[np.ndarray] = []
        tries = 0
        while len(out) < n and tries < max_tries:
            tries += 1
            # Dirichlet on pseudo-simplex, then map back to x-space
            w = rng.dirichlet(np.ones(self.q))
            x = self.from_pseudo(w)
            if self.is_feasible(x):
                out.append(x)
        if len(out) < n:
            k = n - len(out)
            warnings.warn(
                f"SimplexRegion.random_points: rejection sampling дал "
                f"{len(out)}/{n} точек за {max_tries} попыток (узкие границы); "
                f"{k} точек добрано выпуклыми комбинациями крайних вершин.",
                UserWarning, stacklevel=2)
            V = self.extreme_vertices()
            if len(V) > 0:
                W = rng.dirichlet(np.full(len(V), 0.5), size=k)
                out.extend(W @ V)
            else:  # региона-полтопа нет (числовая экзотика) — последний резерв
                while len(out) < n:
                    out.append(self.centroid())
        return np.array(out[:n])

    # ------------------------------------------------------------------
    # Групповое (mixture-of-mixtures) сэмплирование — iter31
    # ------------------------------------------------------------------
    def _validate_groups(self, groups: Sequence[Sequence[int]]
                         ) -> List[List[int]]:
        """Нормализовать/проверить группы индексов: диапазон и непересечение."""
        seen: set = set()
        norm: List[List[int]] = []
        for g in groups:
            idx = [int(i) for i in g]
            if not idx:
                raise ValueError("Пустая группа в groups недопустима.")
            for i in idx:
                if not (0 <= i < self.q):
                    raise ValueError(
                        f"Индекс компонента {i} вне диапазона 0..{self.q - 1}.")
                if i in seen:
                    raise ValueError(
                        f"Компонент {i} входит более чем в одну группу.")
                seen.add(i)
            norm.append(idx)
        return norm

    def _random_points_grouped(self, n: int, seed: int | None,
                               groups: Sequence[Sequence[int]]) -> np.ndarray:
        """n точек со СТРАТИФИКАЦИЕЙ сумм групп (см. docstring random_points).

        Схема на точку: (1) суммы частей (группы + «остальные») тянутся
        последовательно, каждая равномерно на интервале, суженном по уже
        выбранным и по границам оставшихся (замыкание Σ=1 на последней части);
        (2) сумма части раскладывается по её компонентам conditional
        narrowing'ом (:func:`_narrowing_split`, без rejection). Допустимость
        гарантирована конструкцией: валидация региона (ΣL≤1≤ΣU) делает все
        сужённые интервалы непустыми.
        """
        idx_groups = self._validate_groups(groups)
        grouped = {i for g in idx_groups for i in g}
        rest = [i for i in range(self.q) if i not in grouped]
        parts: List[List[int]] = list(idx_groups) + ([rest] if rest else [])
        rng = np.random.default_rng(seed)
        lo, hi = self.lower, self.upper
        p_lo = [float(lo[p].sum()) for p in parts]
        p_hi = [float(hi[p].sum()) for p in parts]
        m = len(parts)
        X = np.empty((int(n), self.q), dtype=float)
        for t in range(int(n)):
            left = 1.0
            lo_after = float(sum(p_lo))
            hi_after = float(sum(p_hi))
            for j, p in enumerate(parts):
                lo_after -= p_lo[j]
                hi_after -= p_hi[j]
                if j == m - 1:
                    s = left                      # замыкание Σ=1
                else:
                    a = max(p_lo[j], left - hi_after)
                    b = min(p_hi[j], left - lo_after)
                    if b < a:                     # числовая страховка
                        a = b = 0.5 * (a + b)
                    s = float(rng.uniform(a, b)) if b > a else a
                X[t, p] = _narrowing_split(lo[p], hi[p], s, rng)
                left -= s
        return X

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"SimplexRegion(q={self.q}, "
                f"L={np.round(self.lower, 3).tolist()}, "
                f"U={np.round(self.upper, 3).tolist()})")


# ----------------------------------------------------------------------
# Conditional narrowing split (iter31) — раскладка суммы без rejection
# ----------------------------------------------------------------------
def _narrowing_split(lo: np.ndarray, hi: np.ndarray, total: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Разложить сумму ``total`` по компонентам с границами ``[lo, hi]``.

    Conditional narrowing: компоненты обходятся в СЛУЧАЙНОМ порядке (снимает
    систематическое смещение первых координат); для очередного компонента
    допустимый интервал сужается по остатку суммы и границам оставшихся:
    ``[max(L_i, s − ΣU_ост), min(U_i, s − ΣL_ост)]``, значение — равномерно
    в нём; последний компонент замыкает сумму. Без rejection: валидная точка
    с первой попытки. Требует ``ΣL ≤ total ≤ ΣU`` (иначе ValueError).
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    k = len(lo)
    s = float(total)
    lo_rest = float(lo.sum())
    hi_rest = float(hi.sum())
    if s < lo_rest - 1e-9 or s > hi_rest + 1e-9:
        raise ValueError(
            f"_narrowing_split: сумма {s:.6f} вне [{lo_rest:.6f}, {hi_rest:.6f}].")
    out = np.empty(k, dtype=float)
    order = rng.permutation(k)
    for pos, i in enumerate(order):
        lo_rest -= float(lo[i])
        hi_rest -= float(hi[i])
        if pos == k - 1:
            v = min(max(s, float(lo[i])), float(hi[i]))  # замыкание + страховка
        else:
            a = max(float(lo[i]), s - hi_rest)
            b = min(float(hi[i]), s - lo_rest)
            if b < a:                                    # числовая страховка
                a = b = 0.5 * (a + b)
            v = float(rng.uniform(a, b)) if b > a else a
        out[i] = v
        s -= v
    return out


# ----------------------------------------------------------------------
# Parts (mass-parts) → fraction-bounds conversion
# ----------------------------------------------------------------------
def parts_ranges_to_fraction_bounds(parts_min: Sequence[float],
                                    parts_max: Sequence[float]
                                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Convert per-component mass-part ranges to fraction bounds L_i ≤ x_i ≤ U_i.

    A user prefers to enter recipes as **mass parts** relative to a base
    component (e.g. base = 100 parts, others vary in some parts range). The
    fraction of component ``i`` is ``x_i = p_i / Σ_j p_j`` where the total
    Σ_j p_j floats with the recipe — so the base's *fraction* range floats too.

    For each component ``i`` with parts range ``[a_i, b_i]`` (a fixed base has
    ``a_i = b_i = 100``), the tightest axis-aligned (box) bounds on its fraction
    are obtained at the extreme totals::

        L_i = a_i / (a_i + Σ_{j≠i} b_j)      # own minimum, others maximal
        U_i = b_i / (b_i + Σ_{j≠i} a_j)      # own maximum, others minimal

    The returned ``(lower, upper)`` is the tightest box that contains the true
    (curved) parts-region; together with Σx=1 it is a valid feasible region for
    :class:`SimplexRegion`.

    Parameters
    ----------
    parts_min, parts_max : sequences of non-negative parts (same length q).
        For the fixed base component pass ``parts_min[i] == parts_max[i]``.

    Returns
    -------
    (lower, upper) : np.ndarray, np.ndarray  — fraction bounds (length q).
    """
    a = np.asarray(parts_min, dtype=float)
    b = np.asarray(parts_max, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("parts_min and parts_max must be 1-D of equal length.")
    if np.any(a < 0) or np.any(b < 0):
        raise ValueError("Parts must be non-negative.")
    if np.any(a > b + _TOL):
        raise ValueError("Each parts_min must be <= its parts_max.")
    if b.sum() <= _TOL:
        raise ValueError("Total parts must be positive.")

    sum_a = a.sum()
    sum_b = b.sum()
    lower = a / (a + (sum_b - b))
    upper = b / (b + (sum_a - a))
    # числовая страховка: в [0,1] и L ≤ U
    lower = np.clip(lower, 0.0, 1.0)
    upper = np.clip(upper, 0.0, 1.0)
    upper = np.maximum(upper, lower)
    return lower, upper


