# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 16 / §15.0.3 (REVISED) — примитив движения границ симплекса.

Покрывает §15.0.3.6:
  * 4 направления классификации (relax/restrict/shift/rebalance);
  * инвариант симплекс-замкнутости во ВСЕ стороны (Σ нижних / Σ верхних /
    покомпонентная достижимость);
  * process-переменные ИСКЛЮЧЕНЫ из инварианта связности;
  * запрет boundary-проекции для точек с измеренным Y (класс §15.0.4);
  * выпавшая точка исключена из активного pool, но СОХРАНЕНА в истории;
  * обратимость: сузили → расширили обратно → точка восстановлена;
  * диагностика — ПОТРЕБИТЕЛЬ примитива через intent (H1 — один из многих).

``rebalance`` по решению сессии §15.0.3 пока НЕ реализован: классифицируется, но
``move_bounds`` поднимает ``NotImplementedError`` (отдельная функция при нужде).
"""
import pytest

from src.core.schema import (MIXTURE, DataPoint, ProjectSchema, VariableBlock,
                             MISSING)
from src.design.move_bounds import (
    MOVE_RELAX, MOVE_RESTRICT, MOVE_SHIFT, MOVE_REBALANCE,
    RegionMoveError, RegionState,
    move_bounds, classify_move, check_simplex_closure,
    handle_dropped_fixed, apply_move, active_fixed, suggest_bounds_move,
)


# ----------------------------------------------------------------------
# Помощники: области и точки
# ----------------------------------------------------------------------
def _mix(lower, upper, names=("A", "B", "C")):
    block = VariableBlock.mixture(list(names), lower=list(lower), upper=list(upper))
    return ProjectSchema(version=1, blocks=(block,))


def _mix_process():
    mix = VariableBlock.mixture(["A", "B", "C"])
    proc = VariableBlock.process(["T", "P"], lower=[0.2, 0.2], upper=[0.8, 0.8])
    return ProjectSchema(version=1, blocks=(mix, proc))


def _point(mix_coords, y=None):
    return DataPoint(schema_version=1, X={MIXTURE: list(mix_coords)},
                     Y=(y or {}))


# ----------------------------------------------------------------------
# §15.0.3.1 Классификация четырёх направлений
# ----------------------------------------------------------------------
def test_move_bounds_classifies_four_directions():
    # A∈[0.1,0.5]; B,C свободны — позволяет различить relax/restrict/shift.
    r = _mix([0.1, 0.0, 0.0], [0.5, 1.0, 1.0])
    assert move_bounds(r, {"A": (0.0, 1.0)}).move_type == MOVE_RELAX     # расширение
    assert move_bounds(r, {"A": (0.2, 0.5)}).move_type == MOVE_RESTRICT  # сужение
    assert move_bounds(r, {"A": (0.3, 0.7)}).move_type == MOVE_SHIFT     # сдвиг


def test_rebalance_classified_but_not_implemented():
    """rebalance (A расширяем, B сужаем) КЛАССИФИЦИРУЕТСЯ, но move_bounds его не делает."""
    r = _mix([0.2, 0.0, 0.0], [0.5, 1.0, 1.0])          # A∈[0.2,0.5]
    deltas = {"A": (0.0, 0.9), "B": (0.1, 0.9)}          # A: relax, B: restrict
    assert classify_move(r, deltas) == MOVE_REBALANCE
    with pytest.raises(NotImplementedError, match="rebalance"):
        move_bounds(r, deltas)



def test_region_after_keeps_schema_version():
    """Движение границ — REGION, не эволюция: schema_version НЕ растёт (§15.1.5)."""
    r = _mix([0.1, 0.0, 0.0], [0.5, 1.0, 1.0])
    mv = move_bounds(r, {"A": (0.0, 1.0)})
    assert mv.region_after.version == r.version == 1
    a_lo, a_hi = mv.region_after.mixture_block().lower[0], mv.region_after.mixture_block().upper[0]
    assert (a_lo, a_hi) == (0.0, 1.0)


# ----------------------------------------------------------------------
# §15.0.3.2 Инвариант симплекс-замкнутости во ВСЕ стороны
# ----------------------------------------------------------------------
def test_closure_sum_lower_exceeds_one():
    r = _mix([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    with pytest.raises(AssertionError, match="пустой симплекс"):
        move_bounds(r, {"A": (0.6, 1.0), "B": (0.6, 1.0)})   # Σ нижних = 1.2 > 1


def test_closure_sum_upper_below_one():
    r = _mix([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    with pytest.raises(AssertionError, match="недостижима"):
        # все верхние опущены → Σ верхних = 0.6 < 1 → Σ=1 недостижима
        move_bounds(r, {"A": (0.0, 0.2), "B": (0.0, 0.2), "C": (0.0, 0.2)})


def test_closure_component_unreachable():
    # B,C заперты снизу на 0.3 → A.hi не может превысить 1−0.6 = 0.4
    r = _mix([0.0, 0.3, 0.3], [0.4, 1.0, 1.0])
    with pytest.raises(AssertionError, match="достижимого"):
        move_bounds(r, {"A": (0.0, 0.9)})


def test_closure_relax_to_vertex_ok_when_others_freed():
    """A→вершина (hi=1.0) валиден, ЕСЛИ нижние других = 0 (достижимость соблюдена)."""
    r = _mix([0.0, 0.0, 0.0], [0.4, 1.0, 1.0])
    mv = move_bounds(r, {"A": (0.0, 1.0)})
    assert mv.move_type == MOVE_RELAX


# ----------------------------------------------------------------------
# §15.0.3.2 process-переменные ИСКЛЮЧЕНЫ из инварианта связности
# ----------------------------------------------------------------------
def test_process_vars_exempt_from_closure():
    r = _mix_process()
    mv = move_bounds(r, {"T": (0.0, 1.0), "P": (0.0, 1.0)})   # нет Σ=1 → свободно
    assert mv.move_type == MOVE_RELAX
    pb = mv.region_after.process_block()
    assert tuple(pb.lower) == (0.0, 0.0) and tuple(pb.upper) == (1.0, 1.0)


def test_check_simplex_closure_ignores_process():
    """check_simplex_closure не падает на process-дельтах (даже «широких»)."""
    r = _mix_process()
    check_simplex_closure(r, {"T": (0.0, 1.0)})   # не должно бросать


# ----------------------------------------------------------------------
# §15.0.3.3 Политика выпадающих fixed-строк
# ----------------------------------------------------------------------
def test_dropped_fixed_with_measured_Y_cannot_project():
    narrow = _mix([0.0, 0.0, 0.0], [0.3, 1.0, 1.0])     # A ограничен 0.3
    pt = _point([0.5, 0.3, 0.2], y={"p": 1.0})          # A=0.5 вне области, Y измерен
    with pytest.raises(RegionMoveError, match="Y"):
        handle_dropped_fixed(pt, narrow, policy="boundary")


def test_dropped_candidate_without_Y_can_project():
    narrow = _mix([0.0, 0.0, 0.0], [0.3, 1.0, 1.0])
    cand = _point([0.5, 0.3, 0.2], y={"p": MISSING})    # без измеренного Y
    res = handle_dropped_fixed(cand, narrow, policy="boundary")
    assert res.kind == "PROJECT"
    x = res.point.X[MIXTURE]
    assert abs(sum(x) - 1.0) < 1e-9                     # проекция остаётся композицией Σ=1
    assert x[0] < 0.5                                   # A подтянут к границе (best-effort clip)



def test_dropped_fixed_error_policy_raises():
    narrow = _mix([0.0, 0.0, 0.0], [0.3, 1.0, 1.0])
    pt = _point([0.5, 0.3, 0.2], y={"p": 1.0})
    with pytest.raises(RegionMoveError, match="policy='error'"):
        handle_dropped_fixed(pt, narrow, policy="error")


def test_kept_point_inside_region():
    wide = _mix([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    pt = _point([0.5, 0.3, 0.2], y={"p": 1.0})
    res = handle_dropped_fixed(pt, wide, policy="exclude")
    assert res.kind == "KEEP" and res.point is pt


# ----------------------------------------------------------------------
# §15.0.3.3 Выпавшая точка исключена из pool, но сохранена в истории
# ----------------------------------------------------------------------
def test_dropped_fixed_excluded_not_deleted():
    wide = _mix([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    pt = _point([0.8, 0.1, 0.1], y={"p": 1.0})
    s0 = RegionState(region=wide, history=[pt])
    assert pt in active_fixed(s0)                       # внутри широкой области

    restrict = move_bounds(wide, {"A": (0.0, 0.3)})     # A→0.3 ⇒ pt (A=0.8) выпадает
    s1 = apply_move(s0, restrict, policy="exclude")
    assert pt in s1.history                             # НЕ удалена из истории
    assert pt not in active_fixed(s1)                   # но не в активном pool


def test_shrink_then_expand_recovers_point():
    """Обратимость: сузили (точка выпала) → расширили обратно → точка восстановлена."""
    wide = _mix([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    pt = _point([0.8, 0.1, 0.1], y={"p": 1.0})
    s0 = RegionState(region=wide, history=[pt])

    s1 = apply_move(s0, move_bounds(wide, {"A": (0.0, 0.3)}), policy="exclude")
    assert pt not in active_fixed(s1)                   # выпала

    s2 = apply_move(s1, move_bounds(s1.region, {"A": (0.0, 1.0)}), policy="exclude")
    assert pt in active_fixed(s2)                       # восстановлена из истории


def test_apply_move_error_policy_rejects_dropping_move():
    wide = _mix([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    pt = _point([0.8, 0.1, 0.1], y={"p": 1.0})
    s0 = RegionState(region=wide, history=[pt])
    restrict = move_bounds(wide, {"A": (0.0, 0.3)})
    with pytest.raises(RegionMoveError):
        apply_move(s0, restrict, policy="error")        # движение теряет точку → отказ


# ----------------------------------------------------------------------
# §15.0.3.4 Диагностика — ПОТРЕБИТЕЛЬ примитива через intent
# ----------------------------------------------------------------------
def test_intent_reach_target_moves_when_optimum_outside():
    s = _mix([0.1, 0.0, 0.0], [0.5, 1.0, 1.0])          # A∈[0.1,0.5]
    deltas = suggest_bounds_move(s, intent="reach_target", optimum={"A": 0.8})
    assert deltas is not None                           # H1-аналог: дотянуть к оптимуму
    assert deltas["A"] == (0.1, 0.8)
    # и результат скармливается примитиву без потери корректности:
    mv = move_bounds(s, deltas)
    assert mv.move_type == MOVE_RELAX


def test_intent_reach_target_none_when_optimum_inside():
    """H2/H3/H4: оптимум уже внутри → движения границ НЕТ (None)."""
    s = _mix([0.1, 0.0, 0.0], [0.5, 1.0, 1.0])
    assert suggest_bounds_move(s, intent="reach_target", optimum={"A": 0.3}) is None


def test_intent_physical_constraint_is_equal_consumer():
    """physical_constraint — равноправный с economy интент (не предусловие)."""
    s = _mix([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    deltas = suggest_bounds_move(s, intent="physical_constraint",
                                 new_limit={"A": (0.2, 0.4)})
    assert deltas == {"A": (0.2, 0.4)}
    assert move_bounds(s, deltas).move_type == MOVE_RESTRICT


def test_intent_region_of_interest_consumer():
    s = _mix([0.1, 0.0, 0.0], [0.5, 1.0, 1.0])
    deltas = suggest_bounds_move(s, intent="region_of_interest",
                                 new_roi={"A": (0.0, 0.9)})
    assert deltas == {"A": (0.0, 0.9)}


def test_unknown_var_raises():
    s = _mix([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    with pytest.raises(KeyError):
        move_bounds(s, {"Z": (0.0, 0.5)})


def test_empty_deltas_raises():
    s = _mix([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        move_bounds(s, {})
