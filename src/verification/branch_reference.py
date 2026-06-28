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
                     Xc: np.ndarray, *,
                     cost_fn=None, cost_name: str = "cost",
                     cost_spec: "DesirabilitySpec" = None) -> np.ndarray:
    """Overall-desirability цели ветки по ИСТИНЕ в точках ``Xc`` (n×(q+d)).

    Если задан ``cost_fn`` (§15.6 §3): цена за изделие складывается как
    дополнительная ``min``-цель ``cost_name``. ``cost_fn`` считает цену ПО ИСТИНЕ
    напрямую (например ``price_состав(X)·truth.truths['rho'].true(X)``) — Шеффе-фит
    цены НЕ нужен (см. ``make_item_cost_fn``/``price_per_item``). Эталон и пайплайн
    используют ОДНУ форму цены; различие лишь в источнике ρ (истина vs суррогат).
    ``cost_spec`` ОБЯЗАТЕЛЕН при ``cost_fn`` (фиксированный диапазон цены — иначе
    скан и уточнение мерили бы по разным шкалам)."""
    Xc = np.atleast_2d(np.asarray(Xc, float))
    missing = set(goal) - set(truth.property_names)
    if missing:
        raise KeyError(f"Цель ветки ссылается на свойства вне истины: "
                       f"{sorted(missing)}.")
    specs = dict(goal)
    means = {p: truth.truths[p].true(Xc) for p in goal}
    if cost_fn is not None:
        if cost_spec is None:
            raise ValueError("cost_fn требует явный cost_spec (фиксированный "
                             "диапазон цены для согласованности скан/уточнение).")
        means[cost_name] = np.asarray(cost_fn(Xc), float).ravel()
        specs[cost_name] = cost_spec
    return Desirability(specs).overall(means)


def branch_optimum(truth: MultiMixtureProcessTruth,
                   goal: Mapping[str, DesirabilitySpec], *,
                   n_scan: int = 20000, seed: int = 0,
                   refine: bool = True,
                   cost_fn=None, cost_name: str = "cost",
                   cost_spec: "DesirabilitySpec" = None) -> Dict[str, Any]:
    """Аналитический оптимум ветки над полной составной областью.

    ``cost_fn``/``cost_name``/``cost_spec`` (§15.6 §3) — опциональная цена за
    изделие как ``min``-цель (см. :func:`_desirability_at`); ``cost_fn`` считает
    цену по ИСТИНЕ (``price_состав·rho_truth``), без Шеффе-фита.

    Возвращает ``{"x": составной вектор оптимума, "d": overall-desirability в
    нём, "y": {property → значение истины}, "x_scan"/"d_scan": результат
    грубого скана до уточнения}``.
    """
    schema = truth.schema
    q = int(schema.n_mixture)
    d = int(schema.n_process)
    ckw = dict(cost_fn=cost_fn, cost_name=cost_name, cost_spec=cost_spec)

    Xc = composite_random_points(schema, int(n_scan), seed=seed)
    dvals = np.asarray(_desirability_at(truth, goal, Xc, **ckw), float).ravel()
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
                lambda v: -float(_desirability_at(truth, goal, v, **ckw)[0]),
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
            d_cand = float(_desirability_at(truth, goal, cand, **ckw)[0])
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
    «закрытые» process-параметры держатся на ``baseline``; «закрытые» (не
    раскрытые append'ом) mixture-компоненты — на ГРАНИ симплекса 0 (§15.0.4).

    Зеркалит :meth:`MixtureProcessRunner._to_full`: неоткрытый mixture-компонент
    (например, C в фазе 1) физически ОТСУТСТВУЕТ ⇒ его доля 0, а не baseline 1/3.
    Иначе «потолок» фазы считался бы на ДРУГОЙ истине (C=1/3), чем меряет пайплайн
    (C=0), и сравнение d_best с потолком рассогласовано."""
    schema = truth.schema
    q = int(schema.n_mixture)
    d = int(schema.n_process)
    base = np.asarray(baseline, float).ravel()
    full = composite_random_points(schema, int(n), seed=seed)
    out = np.tile(base, (int(n), 1))

    if q > 0:
        mf = _mask_from(mixture_free, schema.mixture_names, q)
        # §15.0.4: закрытые mixture-компоненты на грани 0 (held_sum=0), свободные
        # заполняют всю долю 1 — как 2-компонентный симплекс {A,B} в фазе 1.
        c = np.zeros((int(n), q))
        if mf.any():
            samp = full[:, :q]
            fs = samp[:, mf].sum(axis=1, keepdims=True)
            fs = np.where(fs > 1e-12, fs, 1.0)
            c[:, mf] = samp[:, mf] / fs
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
                          n_scan: int = 40000, seed: int = 0,
                          refine: bool = True, n_starts: int = 8) -> Dict[str, Any]:
    """«Потолок» ветки под маской свободы фазы: плотный скан + локальное уточнение.

    Лучшая достижимая desirability, когда варьируются лишь ``mixture_free`` и
    ``process_free`` (закрытый mixture — на грани 0 (§15.0.4), закрытый process —
    на ``baseline``). Это эталон «дотянул ли пайплайн до потенциала ФАЗЫ»
    (в отличие от глобального :func:`branch_optimum`).

    После грубого скана делается МУЛЬТИСТАРТ-SLSQP по СВОБОДНЫМ осям (свободные
    mixture-доли с Σ=1, свободные process-коды ∈ [0,1]); закрытые координаты
    держатся фиксированными. Уточнение запускается из ``n_starts`` лучших точек
    скана и берётся максимум — иначе одиночный старт в высокой размерности (фаза
    «всё открыто», 5 свободных осей) садится в локальный бассейн на ~1e-3 ниже
    глобального, и потолок перестаёт быть строгой верхней границей для измеренного
    ``d_best`` пайплайна. Без уточнения чистый скан занижает максимум и ломает
    монотонность потолков по фазам — а раскрытие переменных НЕ сужает достижимое
    (область фазы k+1 ⊇ фазы k), поэтому потолок обязан расти. ``refine=False``
    оставляет чистый скан (если scipy недоступен — тихий фолбэк на скан).
    """
    schema = truth.schema
    q = int(schema.n_mixture)
    d = int(schema.n_process)
    pts = masked_region_points(truth, int(n_scan), seed,
                               mixture_free, process_free, baseline)
    dvals = np.asarray(_desirability_at(truth, goal, pts), float).ravel()
    b = int(np.argmax(dvals))
    xb = pts[b].copy()
    best_d = float(dvals[b])

    if refine:
        try:
            from scipy.optimize import minimize

            mf = (_mask_from(mixture_free, schema.mixture_names, q)
                  if q > 0 else np.zeros(0, dtype=bool))
            pf = (_mask_from(process_free, schema.process_names, d)
                  if d > 0 else np.zeros(0, dtype=bool))
            free_mix = np.where(mf)[0]
            free_proc = np.where(pf)[0]
            n_fm = int(free_mix.size)
            # шаблон закрытых координат (held mixture=0 / held process=baseline);
            # одинаков для всех masked-точек, поэтому фиксируем один раз
            held = pts[b].copy()

            def expand(v):
                x = held.copy()
                if n_fm:
                    x[free_mix] = v[:n_fm]
                for k, j in enumerate(free_proc):
                    x[q + j] = v[n_fm + k]
                return x

            n_free = n_fm + int(free_proc.size)
            if n_free:
                bounds = [(0.0, 1.0)] * n_free
                cons = []
                if n_fm > 0:
                    cons.append({"type": "eq",
                                 "fun": lambda v: float(np.sum(v[:n_fm]) - 1.0)})
                # МУЛЬТИСТАРТ: уточняем из top-K точек скана, берём лучшее
                topk = np.argsort(dvals)[::-1][:max(1, int(n_starts))]
                for si in topk:
                    s_pt = pts[si]
                    v0 = np.concatenate([s_pt[free_mix], s_pt[q + free_proc]])
                    res = minimize(
                        lambda v: -float(
                            _desirability_at(truth, goal, expand(v))[0]),
                        v0, method="SLSQP", bounds=bounds, constraints=cons,
                        options={"maxiter": 300, "ftol": 1e-9})
                    cand = expand(np.asarray(res.x, float))
                    if n_fm > 0:                # перенормировать свободные доли (Σ=1)
                        fm = np.clip(cand[free_mix], 0.0, None)
                        s = fm.sum()
                        if s > 0:
                            cand[free_mix] = fm / s
                    if free_proc.size:
                        cand[q + free_proc] = np.clip(cand[q + free_proc], 0.0, 1.0)
                    d_cand = float(_desirability_at(truth, goal, cand)[0])
                    if d_cand >= best_d:
                        xb, best_d = cand, d_cand
        except Exception:  # noqa: BLE001 — без scipy остаётся скан-оптимум
            pass

    y = {p: float(truth.truths[p].true(xb.reshape(1, -1))[0])
         for p in truth.property_names}
    return {"x": xb, "d": float(best_d), "y": y}


