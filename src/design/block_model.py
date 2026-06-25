"""
design/block_model.py — §13.3 единый генератор термов модели (вход дизайна).

Генератор ОДИН для всех режимов (mixture-only / process-only / mixture-process):
режим определяется НАЛИЧИЕМ блоков в схеме, а не флагом. Никаких ``if mode==…``.

Полная форма отклика (§13.3):

    η = Σ β_i x_i + Σ β_ij x_i x_j        (Scheffé, mixture; БЕЗ intercept)
      + Σ γ_k z_k + Σ γ_kl z_k z_l        (RSM, process; в коде [0,1])
      + Σ δ_ik x_i z_k                    (кросс mixture×process)

Терм представлен кортежем ГЛОБАЛЬНЫХ индексов составного вектора координат
``[x_0..x_{q-1}, z_0..z_{d-1}]`` (см. ``schema.composite_coords``); пустой кортеж
``()`` — intercept (столбец единиц). Модельная матрица строится перемножением
соответствующих столбцов — ровно как ``linalg.scheffe_matrix_terms``, но на
составных координатах.

Инвариант бит-в-бит (§13.9-регресс): для mixture-only схемы термы совпадают с
``scheffe_term_indices(q, order)`` в том же порядке ⇒ ``model_matrix`` поэлементно
равна ``scheffe_matrix`` (mixture-координаты занимают первые q столбцов).

Инвариант intercept (§13.3): mixture-блок несёт константу через Σx=1 ⇒ intercept
НЕ добавляется; process-only (mixture-блока нет) ⇒ intercept НУЖЕН.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.linalg import scheffe_term_indices
from ..core.schema import (MIXTURE, PROCESS, ProjectSchema, ModelSpec,
                           DataPoint, composite_matrix)

Term = Tuple[int, ...]


@dataclass
class ModelTerms:
    """Список термов модели (в глобальных индексах) + имена + разбивка по категориям."""

    terms: List[Term]
    names: List[str]
    categories: List[str]              # 'intercept'|'mixture'|'process'|'cross' на каждый терм
    q: int
    d: int

    @property
    def p(self) -> int:
        return len(self.terms)

    def breakdown(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for c in self.categories:
            out[c] = out.get(c, 0) + 1
        return out

    @property
    def has_intercept(self) -> bool:
        return any(c == "intercept" for c in self.categories)


# ----------------------------------------------------------------------
# Термы по блокам
# ----------------------------------------------------------------------
def _mixture_terms(q: int, order: str) -> List[Term]:
    # глобальные индексы == локальные (mixture-координаты идут первыми)
    return [tuple(idx) for idx in scheffe_term_indices(q, order)]


def _mixture_names(q: int, order: str, names: Tuple[str, ...]) -> List[str]:
    out = []
    for idx in scheffe_term_indices(q, order):
        out.append("*".join(names[i] for i in idx))
    return out


def _process_terms(q: int, d: int, order: str) -> Tuple[List[Term], List[str]]:
    """RSM-термы по process-координатам (глобальный сдвиг на q)."""
    terms: List[Term] = []
    cats: List[str] = []
    # линейные γ_k z_k
    for k in range(d):
        terms.append((q + k,))
    if order == "quadratic":
        # чистые квадраты γ_kk z_k^2
        for k in range(d):
            terms.append((q + k, q + k))
        # парные γ_kl z_k z_l
        for k, l in combinations(range(d), 2):
            terms.append((q + k, q + l))
    cats = ["process"] * len(terms)
    return terms, cats


def _process_names(d: int, order: str, names: Tuple[str, ...]) -> List[str]:
    out = [names[k] for k in range(d)]
    if order == "quadratic":
        out += [f"{names[k]}^2" for k in range(d)]
        out += [f"{names[k]}*{names[l]}" for k, l in combinations(range(d), 2)]
    return out


def _cross_terms(q: int, d: int, level: str, main: Tuple[int, ...]
                 ) -> List[Term]:
    """Кросс-члены δ_ik x_i z_k по уровню (§13.3)."""
    if level == "additive":
        return []
    if level == "cross-main":
        # только главные компоненты (после M3). До M3 main пуст → вырождается в additive.
        comps = [i for i in main if 0 <= i < q]
    elif level == "full-cross":
        comps = list(range(q))
    else:  # pragma: no cover - валидируется в ModelSpec
        raise ValueError(f"Неизвестный cross_level '{level}'.")
    return [(i, q + k) for i in comps for k in range(d)]


# ----------------------------------------------------------------------
# Главный генератор (§13.3 build_model_terms)
# ----------------------------------------------------------------------
def build_model_terms(schema: ProjectSchema) -> ModelTerms:
    """Единый генератор термов для любой схемы. Режим = по наличию блоков."""
    mix = schema.mixture_block()
    proc = schema.process_block()
    q = mix.size if mix else 0
    d = proc.size if proc else 0
    m: ModelSpec = schema.model

    terms: List[Term] = []
    names: List[str] = []
    cats: List[str] = []

    # process-only ⇒ нужен intercept (mixture не несёт константу)
    if mix is None:
        terms.append(())
        names.append("1")
        cats.append("intercept")

    if mix is not None:
        mt = _mixture_terms(q, m.mixture_order)
        terms += mt
        names += _mixture_names(q, m.mixture_order, mix.names)
        cats += ["mixture"] * len(mt)

    if proc is not None:
        pt, pc = _process_terms(q, d, m.process_order)
        terms += pt
        names += _process_names(d, m.process_order, proc.names)
        cats += pc

    if mix is not None and proc is not None:
        ct = _cross_terms(q, d, m.cross_level, m.main_components)
        terms += ct
        for (i, gk) in ct:
            k = gk - q
            names.append(f"{mix.names[i]}:{proc.names[k]}")
            cats.append("cross")

    # Инвариант intercept (§13.3): mixture ⇒ нет intercept; process-only ⇒ есть.
    has_int = any(c == "intercept" for c in cats)
    if mix is not None:
        assert not has_int, "Scheffé-модель не должна содержать intercept (Σx=1)."
    else:
        assert has_int, "process-only модель обязана содержать intercept."

    return ModelTerms(terms=terms, names=names, categories=cats, q=q, d=d)


# ----------------------------------------------------------------------
# Модельная матрица на составных координатах
# ----------------------------------------------------------------------
def model_matrix(schema: ProjectSchema, Xc: np.ndarray,
                 terms: Optional[ModelTerms] = None) -> np.ndarray:
    """Построить модельную матрицу из составных координат ``Xc`` (n × (q+d)).

    ``Xc`` = ``[x_0..x_{q-1}, z_0..z_{d-1}]`` (process — в коде [0,1]).
    """
    Xc = np.atleast_2d(np.asarray(Xc, dtype=float))
    n = Xc.shape[0]
    mt = terms if terms is not None else build_model_terms(schema)
    cols = []
    for idx in mt.terms:
        if len(idx) == 0:            # intercept
            cols.append(np.ones(n))
            continue
        col = np.ones(n)
        for j in idx:
            col = col * Xc[:, j]
        cols.append(col)
    return np.column_stack(cols) if cols else np.empty((n, 0))


def model_matrix_points(schema: ProjectSchema, points,
                        terms: Optional[ModelTerms] = None) -> np.ndarray:
    """Модельная матрица из списка :class:`DataPoint` (через составные координаты)."""
    Xc = composite_matrix(schema, list(points))
    return model_matrix(schema, Xc, terms=terms)


def count_params(schema: ProjectSchema) -> int:
    """Число параметров p выбранной модели (контроль n vs p, §13.3)."""
    return build_model_terms(schema).p


# ----------------------------------------------------------------------
# Бюджет-guard для cross-main (решение по открытому вопросу: дефолт + guard)
# ----------------------------------------------------------------------
def resolve_model_for_budget(schema: ProjectSchema, n_runs: int,
                             margin: int = 10) -> Tuple[ProjectSchema, Dict]:
    """Понизить уровень модели, если бюджет ``n_runs`` не покрывает p (§13.3).

    Логика (адаптивный дефолт по фазе + guard): если ``n_runs < p + margin`` —
    сначала убираем кросс-члены (``cross_level→additive``), затем, если всё ещё
    мало, понижаем ``process_order→linear``. Возвращает (новая_схема, лог).
    Это и есть бюджет-guard: дефолт ``cross-main`` никогда не требует
    недостижимого n — при нехватке точек он сам деградирует.
    """
    log: Dict = {"requested": schema.model.to_dict(), "n_runs": int(n_runs),
                 "margin": int(margin), "steps": []}
    cur = schema
    p = count_params(cur)
    log["p_initial"] = p
    if n_runs >= p + margin:
        log["final"] = cur.model.to_dict()
        log["p_final"] = p
        return cur, log

    # шаг 1: убрать кросс-члены
    if cur.model.cross_level != "additive":
        cur = cur.with_changes(model=replace(cur.model, cross_level="additive",
                                             main_components=()))
        p = count_params(cur)
        log["steps"].append({"action": "cross_level->additive", "p": p})
        if n_runs >= p + margin:
            log["final"] = cur.model.to_dict()
            log["p_final"] = p
            return cur, log

    # шаг 2: понизить process_order
    if cur.model.process_order == "quadratic" and cur.process_block() is not None:
        cur = cur.with_changes(model=replace(cur.model, process_order="linear"))
        p = count_params(cur)
        log["steps"].append({"action": "process_order->linear", "p": p})

    log["final"] = cur.model.to_dict()
    log["p_final"] = count_params(cur)
    return cur, log
