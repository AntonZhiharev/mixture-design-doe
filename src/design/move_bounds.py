"""
design/move_bounds.py — §15.0.3 (REVISED): примитив движения границ симплекса
как самостоятельная подсистема.

Смена оси проектирования (§15.0.3):

    БЫЛО (узко):   diagnose → если H1 → relax bounds(A) к вершине.
    СТАЛО (общо):  move_bounds(Δ) — корректный примитив с инвариантами, валидный
                   для ЛЮБОГО направления (расширение/сужение/сдвиг), для ЛЮБОГО
                   компонента, с гарантией согласованности симплекса.

Принцип: примитив **не знает**, ЗАЧЕМ его вызвали (economy, расширение области
интереса, сужение под физический запрет, дотягивание до эталона). Он гарантирует
КОРРЕКТНОСТЬ операции. Кто решает «надо подвинуть» — отдельный слой (диагностика —
лишь ОДИН из вызывающих, см. :func:`suggest_bounds_move`).

Четыре качественно разных операции (§15.0.3.1):

    relax     L↓ или U↑          область растёт      fixed-строки тривиально внутри
    restrict  L↑ или U↓          область сжимается   часть fixed может выпасть
    shift     L,U в одну сторону область едет        часть выпадает + extrapolation
    rebalance разные компоненты  форма меняется      связность (Σ=1) — высшая опасность
              в разные стороны

> ``rebalance`` пока НЕ реализован (по решению сессии §15.0.3): классифицируется,
> но :func:`move_bounds` для него поднимает ``NotImplementedError`` — отдельная
> функция будет добавлена, ЕСЛИ появится необходимость.

Универсальный инвариант симплекс-замкнутости (§15.0.3.2) проверяется при ЛЮБОМ
движении mixture-границ; process-переменные (нет Σ=1) из него ИСКЛЮЧЕНЫ.

Политика выпадающих fixed-строк (§15.0.3.3): ``exclude`` (исключить, но сохранить
в истории) / ``boundary`` (спроецировать — ЗАПРЕЩЕНО для точек с измеренным Y,
класс §15.0.4) / ``error`` (запретить движение, теряющее точки).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.schema import (MIXTURE, PROCESS, DataPoint, ProjectSchema,
                             VariableBlock, is_missing)
from src.core.schema_evolution import point_in_region

EPS = 1e-9

# Типы движения (§15.0.3.1)
MOVE_RELAX = "relax"
MOVE_RESTRICT = "restrict"
MOVE_SHIFT = "shift"
MOVE_REBALANCE = "rebalance"

# Политики выпадающих fixed-строк (§15.0.3.3)
POLICY_EXCLUDE = "exclude"
POLICY_BOUNDARY = "boundary"
POLICY_ERROR = "error"

# Интенты слоя-потребителя (§15.0.3.4)
INTENT_REACH_TARGET = "reach_target"
INTENT_PHYSICAL = "physical_constraint"
INTENT_ROI = "region_of_interest"
INTENT_SHRINK = "shrink_explored"


class RegionMoveError(Exception):
    """Движение границ невозможно/запрещено политикой (§15.0.3.3)."""


# ----------------------------------------------------------------------
# Результат движения границ
# ----------------------------------------------------------------------
@dataclass
class RegionMove:
    """Классифицированное и проверенное на инварианты движение границ.

    ``move_bounds`` НЕ применяет дельты вслепую: он возвращает ``RegionMove`` с
    типом операции и уже построенной ``region_after`` (прошедшей инвариант-чеки).
    Поведение fixed-строк зависит от ``move_type`` (см. :func:`handle_dropped_fixed`).
    """

    move_type: str
    deltas: Dict[str, Tuple[float, float]]
    region_after: ProjectSchema


@dataclass
class DropResult:
    """Решение по одной fixed-строке при движении границ (§15.0.3.3).

    ``kind``: ``KEEP`` (внутри новой области) / ``DROP`` (выпала, исключена, но
    остаётся в истории) / ``PROJECT`` (спроецирована на границу — только для
    точек БЕЗ измеренного Y).
    """

    kind: str
    point: DataPoint


@dataclass
class RegionState:
    """Контейнер «область + история точек» для обратимого движения границ.

    ``history`` — ВСЕ точки (никогда не удаляются): выпавшая при сужении точка
    остаётся в истории и снова становится активной при обратном расширении
    (§15.0.3.3 «история ≠ активный pool»). Активный набор вычисляется по запросу
    через :func:`active_fixed`.
    """

    region: ProjectSchema
    history: List[DataPoint] = field(default_factory=list)


# ----------------------------------------------------------------------
# Вспомогательное: границы переменной из схемы
# ----------------------------------------------------------------------
def _var_bounds(schema: ProjectSchema, var: str) -> Tuple[float, float]:
    for b in schema.blocks:
        if var in b.names:
            j = b.names.index(var)
            return float(b.lower[j]), float(b.upper[j])
    raise KeyError(f"Переменная '{var}' не найдена в схеме.")


def _is_mixture_var(schema: ProjectSchema, var: str) -> bool:
    mb = schema.mixture_block()
    return mb is not None and var in mb.names


# ----------------------------------------------------------------------
# §15.0.3.1 Классификация типа движения
# ----------------------------------------------------------------------
def _classify_var(old_lo: float, old_hi: float,
                  new_lo: float, new_hi: float, tol: float = EPS) -> Optional[str]:
    """Тип движения ОДНОЙ переменной (или ``None``, если границы не изменились)."""
    dlo = new_lo - old_lo
    dhi = new_hi - old_hi
    if abs(dlo) <= tol and abs(dhi) <= tol:
        return None
    expand = (dlo <= tol) and (dhi >= -tol)      # L↓/= и U↑/= → область растёт
    restrict = (dlo >= -tol) and (dhi <= tol)    # L↑/= и U↓/= → область сжимается
    if expand and not restrict:
        return MOVE_RELAX
    if restrict and not expand:
        return MOVE_RESTRICT
    # обе границы поехали в одну сторону → сдвиг
    return MOVE_SHIFT


def classify_move(schema: ProjectSchema,
                  deltas: Dict[str, Tuple[float, float]]) -> str:
    """Классифицировать движение: relax/restrict/shift/rebalance (§15.0.3.1).

    Агрегирует покомпонентные типы: единый тип → он же; РАЗНЫЕ типы по разным
    компонентам (например, A расширяем, B сужаем при Σ=1) → ``rebalance``.
    """
    types = set()
    for var, (nlo, nhi) in deltas.items():
        olo, ohi = _var_bounds(schema, var)
        t = _classify_var(olo, ohi, float(nlo), float(nhi))
        if t is not None:
            types.add(t)
    if not types:
        return MOVE_RELAX                 # тождественное движение (no-op) — безопасно
    if len(types) == 1:
        return next(iter(types))
    return MOVE_REBALANCE                 # смешанные направления по компонентам


# ----------------------------------------------------------------------
# §15.0.3.2 Инварианты
# ----------------------------------------------------------------------
def check_nonempty(schema: ProjectSchema,
                   deltas: Dict[str, Tuple[float, float]]) -> None:
    """Каждый интервал непуст: ``new_lo <= new_hi`` (§15.0.3.1)."""
    for var, (nlo, nhi) in deltas.items():
        assert float(nlo) <= float(nhi) + EPS, (
            f"{var}: new_lo={nlo} > new_hi={nhi} — пустой интервал, область пуста")


def check_simplex_closure(schema: ProjectSchema,
                          deltas: Dict[str, Tuple[float, float]]) -> None:
    """Универсальный инвариант симплекс-замкнутости (§15.0.3.2).

    Проверяется при ЛЮБОМ движении mixture-границ (расширение/сужение/сдвиг):

        Σ L_v ≤ 1 ≤ Σ U_v   и   ∀v: U_v ≤ 1 − Σ_{w≠v} L_w .

    process-переменные (T,P) ИСКЛЮЧЕНЫ — у них нет Σ=1, движение свободно в [0,1].
    """
    mb = schema.mixture_block()
    if mb is None:
        return
    names = list(mb.names)
    new_lo = list(map(float, mb.lower))
    new_hi = list(map(float, mb.upper))
    for var, (nlo, nhi) in deltas.items():
        if var in names:
            j = names.index(var)
            new_lo[j] = float(nlo)
            new_hi[j] = float(nhi)

    sum_lo = sum(new_lo)
    sum_hi = sum(new_hi)
    assert sum_lo <= 1.0 + EPS, (
        f"Σ нижних границ {sum_lo:.6f} > 1 — пустой симплекс")
    assert sum_hi >= 1.0 - EPS, (
        f"Σ верхних границ {sum_hi:.6f} < 1 — Σ=1 недостижима")
    # Покомпонентная достижимость проверяется ТОЛЬКО для компонентов, которые
    # ДВИГАЕТ этот вызов: U_v ≤ 1 − Σ_{w≠v} L_w. Номинальный U_i=1.0 у НЕ-двигаемых
    # компонентов при L_j>0 других — штатная и допустимая ситуация (SimplexRegion
    # валидирует область лишь по ΣL≤1≤ΣU, верхняя граница не обязана быть тугой).
    # Применять «∀v» ко всем сломало бы любую нормальную область с U_i=1.0; здесь
    # — ответственность ДВИЖЕНИЯ: новый U не должен превышать достижимого. Это и
    # ловит «A→1 при B_lo>0» (§15.0.3.2), не трогая существующие номинальные U.
    for var in deltas:
        if var not in names:
            continue
        j = names.index(var)
        reachable_hi = 1.0 - (sum_lo - new_lo[j])   # 1 − Σ_{w≠j} L_w
        assert new_hi[j] <= reachable_hi + EPS, (
            f"{var}.hi={new_hi[j]:.6f} > достижимого {reachable_hi:.6f} "
            f"при текущих L других компонентов")



# ----------------------------------------------------------------------
# Применение дельт → новая область (та же версия схемы: это REGION, не bump)
# ----------------------------------------------------------------------
def apply_deltas(schema: ProjectSchema,
                 deltas: Dict[str, Tuple[float, float]]) -> ProjectSchema:
    """Построить ``region_after`` (новые блоки), сохранив ``version`` схемы.

    ``move_bounds`` — примитив ОБЛАСТИ, не эволюции схемы: ``schema_version`` за
    движение границ НЕ растёт (§15.1.5 / Предусловие 2). Версионные bump'ы —
    дело ``schema_evolution.evolve_schema``, не этого примитива.
    """
    new_blocks: List[VariableBlock] = []
    for b in schema.blocks:
        names = list(b.names)
        lo = list(map(float, b.lower))
        hi = list(map(float, b.upper))
        changed = False
        for var, (nlo, nhi) in deltas.items():
            if var in names:
                j = names.index(var)
                lo[j] = float(nlo)
                hi[j] = float(nhi)
                changed = True
        if not changed:
            new_blocks.append(b)
        elif b.is_mixture:
            new_blocks.append(VariableBlock.mixture(names, lower=lo, upper=hi))
        else:
            new_blocks.append(VariableBlock.process(names, lower=lo, upper=hi))
    return schema.with_changes(blocks=tuple(new_blocks))


# ----------------------------------------------------------------------
# §15.0.3.1 Примитив move_bounds
# ----------------------------------------------------------------------
def move_bounds(schema: ProjectSchema,
                deltas: Dict[str, Tuple[float, float]]) -> RegionMove:
    """Корректное движение границ области в ЛЮБОМ направлении (§15.0.3.1).

    ``deltas``: ``{var: (new_lo, new_hi)}``. Примитив КЛАССИФИЦИРУЕТ операцию,
    проверяет инварианты (непустота интервалов + симплекс-замкнутость для
    mixture) и возвращает ``RegionMove`` с уже построенной ``region_after``. Он
    НЕ применяет дельты вслепую и НЕ знает, зачем его вызвали.

    ``rebalance`` (разные компоненты в разные стороны при Σ=1) пока НЕ реализован
    — поднимает ``NotImplementedError`` (отдельная функция при необходимости).
    """
    if not deltas:
        raise ValueError("deltas пусты — нечего двигать.")
    # неизвестные переменные → явная ошибка (а не молчаливое игнорирование)
    for var in deltas:
        _var_bounds(schema, var)

    check_nonempty(schema, deltas)
    move = classify_move(schema, deltas)
    if move == MOVE_REBALANCE:
        raise NotImplementedError(
            "rebalance (разные mixture-компоненты в разные стороны при Σ=1) "
            "пока не реализован (§15.0.3) — будет отдельной функцией при "
            "появлении необходимости.")
    check_simplex_closure(schema, deltas)
    region_after = apply_deltas(schema, deltas)
    return RegionMove(move_type=move, deltas=dict(deltas), region_after=region_after)


# ----------------------------------------------------------------------
# §15.0.3.3 Политика выпадающих fixed-строк
# ----------------------------------------------------------------------
def _has_measured_Y(point: DataPoint) -> bool:
    """True, если у точки есть хотя бы один ИЗМЕРЕННЫЙ отклик (не MISSING)."""
    return any(not is_missing(v) for v in point.Y.values())


def _project_to_region(point: DataPoint, region: ProjectSchema) -> DataPoint:
    """Спроецировать координаты точки в область (mixture-clip + process-clip [0,1])."""
    new_X: Dict[str, List[float]] = {}
    mb = region.mixture_block()
    if mb is not None and MIXTURE in point.X:
        x = mb.as_simplex_region().clip(point.X[MIXTURE])
        new_X[MIXTURE] = [float(v) for v in x]
    pb = region.process_block()
    if pb is not None and PROCESS in point.X:
        z = np.clip(np.asarray(point.X[PROCESS], float), 0.0, 1.0)
        new_X[PROCESS] = [float(v) for v in z]
    # координаты прочих блоков (если вдруг есть) — без изменений
    for k, v in point.X.items():
        new_X.setdefault(k, [float(c) for c in v])
    return DataPoint(schema_version=point.schema_version, X=new_X,
                     Y=dict(point.Y), origin_tag=dict(point.origin_tag),
                     fixed_in_augment=point.fixed_in_augment)


def handle_dropped_fixed(point: DataPoint, region_after: ProjectSchema,
                         policy: str = POLICY_EXCLUDE) -> DropResult:
    """Решить судьбу fixed-строки при движении границ (§15.0.3.3).

    Внутри новой области → ``KEEP`` (fixed как обычно). Иначе по политике:

      * ``exclude``  — исключить из активного pool (но ОСТАВИТЬ в истории);
      * ``boundary`` — спроецировать на границу; **ЗАПРЕЩЕНО** для точек с
        измеренным Y (проекция меняет X, а Y измерен в исходном X → тот же класс
        рассогласования Y↔X, что §15.0.4);
      * ``error``    — запретить движение, теряющее точку.
    """
    if point_in_region(point, region_after):
        return DropResult("KEEP", point)
    if policy == POLICY_EXCLUDE:
        return DropResult("DROP", point)
    if policy == POLICY_BOUNDARY:
        if _has_measured_Y(point):
            raise RegionMoveError(
                "точка с измеренным Y не проецируется на границу: проекция меняет "
                "X, а Y измерен в исходном X → рассогласование Y↔X (класс §15.0.4). "
                "boundary допустим только для кандидатов без Y.")
        return DropResult("PROJECT", _project_to_region(point, region_after))
    if policy == POLICY_ERROR:
        raise RegionMoveError(
            f"точка {point.X} вне новой области, policy='error' — движение, "
            f"теряющее данные, запрещено.")
    raise ValueError(f"неизвестная политика выпадающих fixed-строк: '{policy}'.")


# ----------------------------------------------------------------------
# Обратимое применение движения к области с историей точек
# ----------------------------------------------------------------------
def active_fixed(state: RegionState, policy: str = POLICY_EXCLUDE) -> List[DataPoint]:
    """Активные fixed-точки = точки истории, валидные для ТЕКУЩЕЙ области.

    Выпавшие по ``exclude`` не попадают сюда, но остаются в ``state.history`` —
    при обратном расширении области они снова станут активными (обратимость,
    §15.0.3.3).
    """
    out: List[DataPoint] = []
    for p in state.history:
        res = handle_dropped_fixed(p, state.region, policy=policy)
        if res.kind in ("KEEP", "PROJECT"):
            out.append(res.point)
    return out


def apply_move(state: RegionState, move: RegionMove,
               policy: str = POLICY_EXCLUDE) -> RegionState:
    """Применить движение к области, СОХРАНИВ всю историю точек (обратимо).

    История переносится целиком (точки никогда не удаляются). Для ``policy=error``
    предварительно проверяется, что ни одна точка истории не выпадает из новой
    области (иначе ``RegionMoveError`` — движение отклоняется без потерь).
    """
    if policy == POLICY_ERROR:
        for p in state.history:
            handle_dropped_fixed(p, move.region_after, policy=POLICY_ERROR)
    return RegionState(region=move.region_after, history=list(state.history))


# ----------------------------------------------------------------------
# §15.0.3.4 Диагностика как ПОТРЕБИТЕЛЬ примитива (через intent)
# ----------------------------------------------------------------------
def _coords_within_bounds(schema: ProjectSchema,
                          coords: Dict[str, float], tol: float = 1e-9) -> bool:
    for var, val in coords.items():
        lo, hi = _var_bounds(schema, var)
        if val < lo - tol or val > hi + tol:
            return False
    return True


def suggest_bounds_move(schema: ProjectSchema, *, intent: str,
                        optimum: Optional[Dict[str, float]] = None,
                        new_limit: Optional[Dict[str, Tuple[float, float]]] = None,
                        new_roi: Optional[Dict[str, Tuple[float, float]]] = None,
                        ) -> Optional[Dict[str, Tuple[float, float]]]:
    """Слой-ПОТРЕБИТЕЛЬ примитива: решить НУЖНО ли двигать и КУДА (§15.0.3.4).

    Возвращает ``deltas`` для :func:`move_bounds` или ``None``, если движение не
    показано. ``intent`` — ЗАЧЕМ двигаем; economy-диагностика (H1) — ЛИШЬ ОДИН из
    интентов (``reach_target``), не предусловие существования примитива.

      * ``reach_target``       — дотянуть область до эталона ``optimum`` (бывш. H1).
        Если оптимум уже внутри (H2/H3/H4) → ``None`` (движения границ нет,
        диагностика направит в семплинг/augmented/физ-предел).
      * ``physical_constraint``— внешний запрет сузил допустимое (``new_limit``).
      * ``region_of_interest`` — заказчик расширил/сдвинул область (``new_roi``).
      * ``shrink_explored``    — сузить под уточнённую физику (``new_roi``).
    """
    if intent == INTENT_REACH_TARGET:
        if not optimum:
            return None
        deltas: Dict[str, Tuple[float, float]] = {}
        for var, val in optimum.items():
            lo, hi = _var_bounds(schema, var)
            if val < lo - EPS:
                deltas[var] = (float(val), hi)        # расширить нижнюю границу
            elif val > hi + EPS:
                deltas[var] = (lo, float(val))        # расширить верхнюю границу
        return deltas or None                          # уже внутри → None (H2-like)
    if intent == INTENT_PHYSICAL:
        return dict(new_limit) if new_limit else None
    if intent in (INTENT_ROI, INTENT_SHRINK):
        return dict(new_roi) if new_roi else None
    return None
