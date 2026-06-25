"""
design/i_optimal.py — M5: I-optimal local design.

I-optimality targets accurate PREDICTION (minimum average prediction variance),
which is the right criterion for local refinement of a regime (REBUILD_SPEC M5,
rule #5: "D точит коэффициенты, I точит прогноз").

I-criterion:   phi_I = trace( (MᵀM)⁻¹ · W ),   smaller is better,
where W = E_region[ f(x) f(x)ᵀ ] is the region moment matrix, estimated by
Monte-Carlo over a large feasible sample.

Generator: BATCH coordinate (point) exchange over a candidate pool + multiple
random restarts (rules #2, #9).  The MODEL is an explicit input (rule #1).

R reference: ``AlgDesign::optFederov(criterion="I")``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import factorial
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..core.linalg import (scheffe_matrix, scheffe_matrix_terms,
                           scheffe_term_indices)
from ..core.simplex import SimplexRegion


def _model_matrix(X, model, terms):
    """Model matrix: explicit ``terms`` (q_eff-reduced) override ``model``."""
    if terms is not None:
        return scheffe_matrix_terms(X, terms)
    return scheffe_matrix(X, model)



@dataclass
class IOptimalResult:
    design: np.ndarray
    indices: np.ndarray
    i_score: float                      # average prediction variance (lower better)
    model: Union[str, int] = "quadratic"
    n_restarts: int = 0
    history: List[float] = field(default_factory=list)

    def to_state(self) -> dict:
        return {
            "design": np.asarray(self.design),
            "indices": np.asarray(self.indices),
            "i_score": float(self.i_score),
            "model": self.model,
            "n_restarts": self.n_restarts,
        }


def _simplex_term_moments(terms: Sequence[Tuple[int, ...]], q: int) -> np.ndarray:
    """Аналитические моменты ``E[f_r f_s]`` на СТАНДАРТНОМ симплексе ``S^{q-1}``.

    ``terms`` — Scheffé-термы (кортежи индексов компонентов; ``q_eff``-редукция
    допустима). Закрытая форма (Дирихле(1,…,1), §13.5)::

        E[∏ x_i^{a_i}] = (q-1)!·∏ a_i! / (q-1+Σa)! ,

    где ``a_i`` — суммарная степень компонента ``i`` в произведении термов ``r,s``.
    Та же формула, что в ``block_moments.analytic_moment_matrix`` ⇒ детерминированно
    (без MC-смещения ``SimplexRegion.random_points``, §13.11) и совпадает бит-в-бит
    с блочной аналитикой для mixture-only. ``q`` — ПОЛНОЕ число компонентов (симплекс
    интегрируется по всем ``q``, неактивные компоненты остаются свободными).
    """
    expo = []
    for t in terms:
        a = [0] * q
        for i in t:
            a[i] += 1
        expo.append(a)
    p = len(expo)
    M = np.empty((p, p))
    fq = factorial(q - 1)
    for r in range(p):
        for s in range(p):
            a = [expo[r][i] + expo[s][i] for i in range(q)]
            num = fq
            for ai in a:
                num *= factorial(ai)
            M[r, s] = num / factorial(q - 1 + sum(a))
    return M


def region_moment_matrix(region: SimplexRegion, model: Union[str, int],
                         n_mc: int = 5000, seed: Optional[int] = None,
                         terms: Optional[List] = None,
                         method: str = "mc") -> np.ndarray:
    """Региональная матрица моментов ``W = E[f(x) f(x)ᵀ]``.

    ``method``:

    * ``"mc"`` (по умолчанию) — Monte-Carlo по ДОПУСТИМОЙ (ограниченной) области
      через ``region.random_points`` (границы ``L_i, U_i`` учитываются, §5.5.2).
      ВНИМАНИЕ: сэмплер неравномерен (стянут к центроиду, §13.11) — годится для
      быстрых прикидок, но не равен равномерному интегралу.
    * ``"analytic"`` — закрытая форма на СТАНДАРТНОМ симплексе (§13.5): границы
      ``L/U`` НЕ учитываются (область интереса = компонентный симплекс),
      детерминированно и точно. Совпадает с ``block_moments.analytic_moment_matrix``
      для mixture-only.

    Если ``terms`` задан, ``W`` строится на ``q_eff``-редуцированном базисе
    (см. :func:`scheffe_active_terms`).
    """
    if method == "analytic":
        tlist = (list(terms) if terms is not None
                 else scheffe_term_indices(region.q, model))
        return _simplex_term_moments(tlist, region.q)
    if method == "mc":
        pts = region.random_points(n_mc, seed=seed)
        F = _model_matrix(pts, model, terms)
        return (F.T @ F) / F.shape[0]
    raise ValueError(f"Unknown method '{method}'. Use 'mc' or 'analytic'.")



def i_optimal_design(candidates: np.ndarray, n_runs: int,
                     moments: np.ndarray,
                     model: Union[str, int] = "quadratic",
                     n_restarts: int = 10, max_iter: int = 100,
                     ridge: float = 1e-8,
                     seed: Optional[int] = None) -> IOptimalResult:
    """Find an (approximately) I-optimal subset of ``n_runs`` candidate rows.

    Parameters
    ----------
    candidates : (n_cand, q) feasible mixture points.
    n_runs     : number of runs to select.
    moments    : region moment matrix W (p x p), same column order as the model.
    model      : explicit Scheffe model order.
    """
    candidates = np.atleast_2d(np.asarray(candidates, dtype=float))
    n_cand, q = candidates.shape
    C = scheffe_matrix(candidates, model)
    p = C.shape[1]
    W = np.asarray(moments, dtype=float)
    if W.shape != (p, p):
        raise ValueError(f"moments must be {p}x{p}, got {W.shape}.")
    if n_cand < n_runs:
        raise ValueError(f"Candidate pool ({n_cand}) smaller than n_runs ({n_runs}).")
    if n_runs < p:
        import warnings
        warnings.warn(f"n_runs={n_runs} < p={p}: information matrix singular.",
                      UserWarning, stacklevel=2)

    rng = np.random.default_rng(seed)
    eye_p = np.eye(p) * ridge

    def i_score(idx: np.ndarray) -> float:
        M = C[idx]
        XtX = M.T @ M + eye_p
        try:
            inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            return float("inf")
        return float(np.trace(inv @ W))

    best_idx: Optional[np.ndarray] = None
    best_score = float("inf")
    history: List[float] = []

    for _ in range(max(1, n_restarts)):
        idx = rng.choice(n_cand, size=n_runs, replace=False)
        cur = i_score(idx)
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for slot in range(n_runs):
                cur_point = idx[slot]
                used = set(idx.tolist())
                best_swap = cur
                best_cand = cur_point
                for cand in range(n_cand):
                    if cand in used and cand != cur_point:
                        continue
                    idx[slot] = cand
                    sc = i_score(idx)
                    if sc < best_swap - 1e-12:
                        best_swap = sc
                        best_cand = cand
                idx[slot] = best_cand
                if best_cand != cur_point:
                    cur = best_swap
                    improved = True
        history.append(cur)
        if cur < best_score:
            best_score = cur
            best_idx = idx.copy()

    return IOptimalResult(design=candidates[best_idx], indices=best_idx,
                          i_score=best_score, model=model,
                          n_restarts=n_restarts, history=history)


@dataclass
class IAugmentResult:
    """Результат I-оптимального добора с критерием остановки (FinalCheckList §5.5).

    ``new_points`` — только добранные точки; ``i_history`` — I-критерий
    ОБЪЕДИНЁННОГО плана после каждого добора (начиная с ``i_base`` — база одна).
    ``stop_reason`` ∈ {'sufficiency','rel_gain','budget','pool_exhausted'}.
    """
    new_points: np.ndarray
    indices: np.ndarray
    i_base: float                 # I только базы (existing)
    i_final: float                # I объединённого плана (база + добор)
    i_history: List[float] = field(default_factory=list)
    stop_reason: str = ""
    n_existing: int = 0
    n_added: int = 0
    p: int = 0                    # число параметров модели (p_quad на q_eff)
    min_total: int = 0            # порог достаточности n_existing+n_added
    rel_tol: float = 0.0
    n_max: int = 0
    cond_number: float = float("nan")   # cond(XᵀX) итогового плана


def i_optimal_augment_sequential(
        existing: np.ndarray, candidates: np.ndarray, moments: np.ndarray, *,
        model: Union[str, int] = "quadratic", terms: Optional[List] = None,
        model_matrix_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        n_max: int = 50, min_total: Optional[int] = None,
        margin: int = 12, rel_tol: float = 0.03, ridge: float = 1e-8,
        seed: Optional[int] = None) -> IAugmentResult:
    """I-оптимальный добор к ``existing`` с КРИТЕРИЕМ ОСТАНОВКИ (FinalCheckList §5.5.3).

    Жадно (forward selection) добавляет по одной точке, минимизирующей I-критерий
    ОБЪЕДИНЁННОГО плана. Останавливается по ЛЮБОМУ из условий:

    * **достаточность**: ``n_existing + n_added >= min_total``
      (по умолчанию ``min_total = p + margin``, где ``p`` — число параметров
      РЕДУЦИРОВАННОЙ модели на ``q_eff``);
    * **затухание выигрыша**: относительное улучшение
      ``(I_old - I_new)/I_old < rel_tol`` (по умолчанию 3 %);
    * **бюджет**: ``n_added >= n_max`` (жёсткий потолок стадии);
    * пул кандидатов исчерпан.

    Никакого абсолютного порога вида ``while I > thr`` — он «протекает», т.к.
    ``I ∝ 1/n`` и в ноль не упрётся (§5.5.3).

    ``model_matrix_fn`` (§13.6): опц. построитель модельной матрицы ``f(X)`` для
    mixture-process. ``None`` ⇒ mixture-only поведение бит-в-бит (Scheffé). Гейт
    достаточности использует ``p = C.shape[1]`` (число столбцов модели), поэтому
    блочная модель не вызывает преждевременной остановки. Геометрия в M5 НЕ
    передаётся — pool/moments готовит блочный слой снаружи (§13.4/§13.5).
    """
    existing = (np.atleast_2d(np.asarray(existing, dtype=float))
                if existing is not None and len(np.asarray(existing)) else None)
    candidates = np.atleast_2d(np.asarray(candidates, dtype=float))
    q = candidates.shape[1]
    # §13.6: инъекция построителя модельной матрицы. None → текущее mixture-only
    # поведение (Scheffé) бит-в-бит; блочный augment передаёт сюда composite-модель.
    mm = (model_matrix_fn if model_matrix_fn is not None
          else (lambda X: _model_matrix(X, model, terms)))
    C = mm(candidates)
    n_cand, p = C.shape
    W = np.asarray(moments, dtype=float)
    # Условие согласованности §13.6-A: модель и моменты — из одного block_model
    # (одни термы) ⇒ число столбцов C обязано совпасть с размером W.
    if W.shape != (p, p):
        raise ValueError(f"moments must be {p}x{p}, got {W.shape}.")
    if min_total is None:
        min_total = p + int(margin)
    n_existing = 0 if existing is None else int(existing.shape[0])

    E = (mm(existing) if existing is not None
         else np.empty((0, p)))
    eye_p = np.eye(p) * ridge
    M_acc = E.T @ E                     # информация уже зафиксированной части

    def i_of(M_info: np.ndarray) -> float:
        try:
            inv = np.linalg.inv(M_info + eye_p)
        except np.linalg.LinAlgError:
            return float("inf")
        return float(np.trace(inv @ W))

    i_base = i_of(M_acc)
    i_hist: List[float] = [i_base]
    chosen: List[int] = []
    i_prev = i_base
    stop_reason = "budget"

    while True:
        # (1) достаточность — самостоятельное условие остановки (любое из трёх)
        if n_existing + len(chosen) >= min_total:
            stop_reason = "sufficiency"
            break
        # (3) бюджет точек стадии
        if len(chosen) >= n_max:
            stop_reason = "budget"
            break
        # выбрать кандидата, максимально снижающего I объединённого плана
        best_c, best_i = -1, i_prev
        used = set(chosen)
        for c in range(n_cand):
            if c in used:
                continue
            row = C[c][:, None]
            sc = i_of(M_acc + row @ row.T)
            if sc < best_i - 1e-15:
                best_i, best_c = sc, c
        if best_c < 0:
            stop_reason = "pool_exhausted"
            break
        # (2) относительный выигрыш на точку
        rel = (i_prev - best_i) / i_prev if np.isfinite(i_prev) and i_prev > 0 else 1.0
        if rel < rel_tol and np.isfinite(i_prev):
            stop_reason = "rel_gain"
            break
        # принять точку
        row = C[best_c][:, None]
        M_acc = M_acc + row @ row.T
        chosen.append(best_c)
        i_prev = best_i
        i_hist.append(best_i)

    idx = np.asarray(chosen, dtype=int)
    new_pts = candidates[idx] if len(idx) else np.empty((0, q))
    try:
        cond = float(np.linalg.cond(M_acc + eye_p))
    except np.linalg.LinAlgError:
        cond = float("inf")
    return IAugmentResult(
        new_points=new_pts, indices=idx, i_base=i_base, i_final=i_prev,
        i_history=i_hist, stop_reason=stop_reason, n_existing=n_existing,
        n_added=int(len(idx)), p=int(p), min_total=int(min_total),
        rel_tol=float(rel_tol), n_max=int(n_max), cond_number=cond)



def i_optimal_for_region(region: SimplexRegion, n_runs: int,
                         model: Union[str, int] = "quadratic",
                         n_random: int = 400, n_mc: int = 5000,
                         n_restarts: int = 10,
                         seed: Optional[int] = None) -> IOptimalResult:
    """Convenience: build candidate pool + region moments, then run I-optimal."""
    from .d_optimal import build_candidate_pool
    pool = build_candidate_pool(region, n_random=n_random, seed=seed)
    W = region_moment_matrix(region, model, n_mc=n_mc, seed=seed)
    return i_optimal_design(pool, n_runs, W, model=model,
                            n_restarts=n_restarts, seed=seed)


def i_optimal_augment(existing: np.ndarray, candidates: np.ndarray,
                      n_add: int, moments: np.ndarray,
                      model: Union[str, int] = "quadratic",
                      n_restarts: int = 10, max_iter: int = 100,
                      ridge: float = 1e-8,
                      seed: Optional[int] = None) -> IOptimalResult:
    """Подобрать ``n_add`` НОВЫХ точек к уже имеющемуся плану ``existing``.

    Это I-оптимальный *добор* (augmentation): строки ``existing`` (например,
    уже измеренный D-оптимальный план M2) **фиксируются**, а среди ``candidates``
    выбираются ``n_add`` дополнительных точек так, чтобы I-критерий
    ОБЪЕДИНЁННОГО плана ``[existing; new]`` был минимален
    (``phi_I = trace((MᵀM)⁻¹ · W)``, меньше — точнее прогноз по области).

    Возвращает :class:`IOptimalResult`, где ``design`` / ``indices`` — это
    ТОЛЬКО новые точки (добор), а ``i_score`` — I-критерий объединённого плана.

    Чтобы новые точки гарантированно отличались от уже измеренных, исключайте
    точки ``existing`` из ``candidates`` до вызова (см. ``run_m5``).
    """
    candidates = np.atleast_2d(np.asarray(candidates, dtype=float))
    n_cand = candidates.shape[0]
    C = scheffe_matrix(candidates, model)
    p = C.shape[1]
    W = np.asarray(moments, dtype=float)
    if W.shape != (p, p):
        raise ValueError(f"moments must be {p}x{p}, got {W.shape}.")
    if n_add < 1:
        raise ValueError("n_add must be >= 1.")
    if n_cand < n_add:
        raise ValueError(
            f"Candidate pool ({n_cand}) smaller than n_add ({n_add}).")

    if existing is not None and len(np.atleast_2d(np.asarray(existing))):
        E = scheffe_matrix(np.atleast_2d(np.asarray(existing, dtype=float)),
                           model)
    else:
        E = np.empty((0, p))
    EtE = E.T @ E

    rng = np.random.default_rng(seed)
    eye_p = np.eye(p) * ridge

    def i_score(idx: np.ndarray) -> float:
        M = C[idx]
        XtX = EtE + M.T @ M + eye_p     # информация ОБЪЕДИНЁННОГО плана
        try:
            inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            return float("inf")
        return float(np.trace(inv @ W))

    best_idx: Optional[np.ndarray] = None
    best_score = float("inf")
    history: List[float] = []

    for _ in range(max(1, n_restarts)):
        idx = rng.choice(n_cand, size=n_add, replace=False)
        cur = i_score(idx)
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for slot in range(n_add):
                cur_point = idx[slot]
                used = set(idx.tolist())
                best_swap = cur
                best_cand = cur_point
                for cand in range(n_cand):
                    if cand in used and cand != cur_point:
                        continue
                    idx[slot] = cand
                    sc = i_score(idx)
                    if sc < best_swap - 1e-12:
                        best_swap = sc
                        best_cand = cand
                idx[slot] = best_cand
                if best_cand != cur_point:
                    cur = best_swap
                    improved = True
        history.append(cur)
        if cur < best_score:
            best_score = cur
            best_idx = idx.copy()

    return IOptimalResult(design=candidates[best_idx], indices=best_idx,
                          i_score=best_score, model=model,
                          n_restarts=n_restarts, history=history)


