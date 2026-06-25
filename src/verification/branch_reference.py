"""
verification/branch_reference.py — аналитический оптимум ВЕТКИ над составной
областью mixture×process (REBUILD_SPEC §8 «сравнение с аналитическим решением»,
§5/§12 ветки).

Ветка — это цель ``{property → DesirabilitySpec}``. Её «аналитическое решение» —
точка состава+режима, максимизирующая overall-desirability ПО ИСТИНЕ
(:class:`MultiMixtureProcessTruth`), а не по суррогату. Именно к этому эталону
обязано сойтись ядро (GP + acquisition по desirability).

Оптимум ищется поверх ПРОИЗВЕДЕНИЯ области (симплекс рецепта × куб [0,1]^d):
  1) плотный случайный скан составной области (грубый максимум);
  2) локальное уточнение SLSQP с ограничением Σx=1 и боксами (если есть scipy).

Это «истинный» (по физике) оптимум ветки над ПОЛНОЙ областью — финальная цель.
Пайплайн приближается к нему постадийно (см. поэтапное раскрытие переменных).
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from ..optimize.desirability import Desirability, DesirabilitySpec
from .mixture_process_truth import (MultiMixtureProcessTruth,
                                    composite_random_points)


def _desirability_at(truth: MultiMixtureProcessTruth,
                     goal: Mapping[str, DesirabilitySpec],
                     Xc: np.ndarray) -> np.ndarray:
    """Overall-desirability цели ветки по ИСТИНЕ в точках ``Xc`` (n×(q+d))."""
    Xc = np.atleast_2d(np.asarray(Xc, float))
    missing = set(goal) - set(truth.property_names)
    if missing:
        raise KeyError(f"Цель ветки ссылается на свойства вне истины: "
                       f"{sorted(missing)}.")
    means = {p: truth.truths[p].true(Xc) for p in goal}
    return Desirability(dict(goal)).overall(means)


def branch_optimum(truth: MultiMixtureProcessTruth,
                   goal: Mapping[str, DesirabilitySpec], *,
                   n_scan: int = 20000, seed: int = 0,
                   refine: bool = True) -> Dict[str, Any]:
    """Аналитический оптимум ветки над полной составной областью.

    Возвращает ``{"x": составной вектор оптимума, "d": overall-desirability в
    нём, "y": {property → значение истины}, "x_scan"/"d_scan": результат
    грубого скана до уточнения}``.
    """
    schema = truth.schema
    q = int(schema.n_mixture)
    d = int(schema.n_process)

    Xc = composite_random_points(schema, int(n_scan), seed=seed)
    dvals = np.asarray(_desirability_at(truth, goal, Xc), float).ravel()
    b = int(np.argmax(dvals))
    x_scan = Xc[b].copy()
    d_scan = float(dvals[b])
    best_x, best_d = x_scan.copy(), d_scan

    if refine:
        try:
            from scipy.optimize import minimize

            mb = schema.mixture_block()
            bounds = []
            if mb is not None:
                bounds += [(float(lo), float(hi))
                           for lo, hi in zip(mb.lower, mb.upper)]
            bounds += [(0.0, 1.0)] * d

            cons = []
            if q > 0:
                cons.append({"type": "eq",
                             "fun": lambda v: float(np.sum(v[:q]) - 1.0)})

            res = minimize(
                lambda v: -float(_desirability_at(truth, goal, v)[0]),
                x_scan, method="SLSQP", bounds=bounds, constraints=cons,
                options={"maxiter": 300, "ftol": 1e-9})

            cand = np.asarray(res.x, float)
            if q > 0:
                cand[:q] = np.clip(cand[:q], 0.0, None)
                s = cand[:q].sum()
                if s > 0:
                    cand[:q] = cand[:q] / s
            if d > 0:
                cand[q:] = np.clip(cand[q:], 0.0, 1.0)
            d_cand = float(_desirability_at(truth, goal, cand)[0])
            if d_cand >= best_d:
                best_x, best_d = cand, d_cand
        except Exception:  # noqa: BLE001 — без scipy остаётся скан-оптимум
            pass

    y_opt = {p: float(truth.truths[p].true(best_x.reshape(1, -1))[0])
             for p in truth.property_names}
    return {"x": best_x, "d": float(best_d), "y": y_opt,
            "x_scan": x_scan, "d_scan": d_scan}


# ----------------------------------------------------------------------
# Фазовый (ограниченный маской свободы) оптимум — «потолок» фазы
# ----------------------------------------------------------------------
def _mask_from(spec, names, size: int) -> np.ndarray:
    """Имена/индексы свободных переменных → булева маска длиной ``size``."""
    m = np.zeros(int(size), dtype=bool)
    names = list(names)
    for item in spec:
        idx = names.index(item) if isinstance(item, str) else int(item)
        if 0 <= idx < size:
            m[idx] = True
    return m


def masked_region_points(truth: MultiMixtureProcessTruth, n: int, seed: int,
                         mixture_free, process_free, baseline) -> np.ndarray:
    """Точки в ОГРАНИЧЕННОЙ маской области: свободные координаты варьируют,
    «закрытые» держатся на ``baseline`` (Σx=1 сохраняется). Та же проекция, что
    в :class:`MixtureProcessRunner` — но в верификационном слое, без ядра."""
    schema = truth.schema
    q = int(schema.n_mixture)
    d = int(schema.n_process)
    base = np.asarray(baseline, float).ravel()
    full = composite_random_points(schema, int(n), seed=seed)
    out = np.tile(base, (int(n), 1))

    if q > 0:
        mf = _mask_from(mixture_free, schema.mixture_names, q)
        held = ~mf
        held_sum = float(base[:q][held].sum()) if held.any() else 0.0
        c = np.tile(base[:q], (int(n), 1))
        if mf.any():
            samp = full[:, :q]
            fs = samp[:, mf].sum(axis=1, keepdims=True)
            fs = np.where(fs > 1e-12, fs, 1.0)
            c[:, mf] = samp[:, mf] * (1.0 - held_sum) / fs
        out[:, :q] = c

    if d > 0:
        pf = _mask_from(process_free, schema.process_names, d)
        z = np.tile(base[q:], (int(n), 1))
        if pf.any():
            z[:, pf] = full[:, q:][:, pf]
        out[:, q:] = z

    return out


def branch_optimum_masked(truth: MultiMixtureProcessTruth,
                          goal: Mapping[str, DesirabilitySpec], *,
                          baseline, mixture_free=(), process_free=(),
                          n_scan: int = 40000, seed: int = 0) -> Dict[str, Any]:
    """«Потолок» ветки под маской свободы фазы (плотный скан, без scipy).

    Лучшая достижимая desirability, когда варьируются лишь ``mixture_free`` и
    ``process_free`` (остальное — на ``baseline``). Это эталон «дотянул ли
    пайплайн до потенциала ФАЗЫ» (в отличие от глобального :func:`branch_optimum`).
    Скан в низкоразмерной области точен без локального уточнения.
    """
    pts = masked_region_points(truth, int(n_scan), seed,
                               mixture_free, process_free, baseline)
    dvals = np.asarray(_desirability_at(truth, goal, pts), float).ravel()
    b = int(np.argmax(dvals))
    xb = pts[b]
    y = {p: float(truth.truths[p].true(xb.reshape(1, -1))[0])
         for p in truth.property_names}
    return {"x": xb, "d": float(dvals[b]), "y": y}


