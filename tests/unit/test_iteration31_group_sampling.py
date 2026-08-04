# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 31 / Функциональные группы: стратификация суммы ниши.

Контекст (перепроверка «осей мягкой фазы» после iter30): при равномерной мере
на политопе сумма группы из k компонентов концентрируется (~Beta(k, q−k)) —
края диапазона дозы ниши (r_soft ∈ [0, 0.40] при 4×U=0.10) кандидатами не
покрывались (факт: min=0.050, max=0.342, std=0.027 из 2000 точек).

Проверяемый канон iter31:
  * SimplexRegion.random_points(groups=...) — равномерная МАРГИНАЛЬ по сумме
    каждой группы (mixture-of-mixtures, conditional narrowing без rejection);
    все точки допустимы (Σx=1, индивидуальные L/U);
  * groups=None/[] — прежнее поведение бит-в-бит (обратная совместимость);
  * runner: проектные группы (set_mixture_sampling_groups) стратифицируют
    seed-пул; группы ветки (Branch.sampling_groups) — намерение ветки,
    None → наследование проектных; валидация имён/непересечения — явные ошибки;
  * сериализация: Branch.to_state/from_state и runner_to_state/from_state
    сохраняют группы (старые state без ключа → дефолты).
"""
import numpy as np
import pytest

from src.core.schema import ModelSpec, ProjectSchema
from src.core.simplex import SimplexRegion, _narrowing_split
from src.apps.mixture_process_runner import MixtureProcessRunner
from src.apps.campaign_state import runner_to_state, runner_from_state
from src.design.branches import Branch
from src.optimize.desirability import DesirabilitySpec

# Сценарий recheck3: BASE, FILLER + 4 конкурирующих компонента одной ниши
NAMES6 = ["BASE", "FILLER", "PBNK", "CPE", "DL", "SBM"]
LO6 = [0.20, 0.05, 0.00, 0.00, 0.00, 0.00]
UP6 = [0.80, 0.50, 0.10, 0.10, 0.10, 0.10]
SOFT_IDX = [2, 3, 4, 5]
SOFT_NAMES = ["PBNK", "CPE", "DL", "SBM"]


class _Oracle:
    property_names = ["modulus"]

    def evaluate(self, Xc):
        Xc = np.atleast_2d(np.asarray(Xc, float))
        r = Xc[:, SOFT_IDX].sum(axis=1)
        return (100.0 - 250.0 * r + 300.0 * r ** 2).reshape(-1, 1)


def _region():
    return SimplexRegion(lower=LO6, upper=UP6, names=NAMES6)


def _runner(seed=0):
    schema = ProjectSchema.mixture_only(
        NAMES6, lower=LO6, upper=UP6,
        model=ModelSpec(cross_level="additive", mixture_order="quadratic"))
    return MixtureProcessRunner(schema, _Oracle(), seed=seed, n_restarts=2)


# ----------------------------------------------------------------------
# Ядро: SimplexRegion.random_points(groups=...)
# ----------------------------------------------------------------------
def test_group_sampling_covers_dose_edges():
    """Сумма группы покрывает ВЕСЬ диапазон [0, 0.40] (раньше 0.05–0.34)."""
    X = _region().random_points(2000, seed=11, groups=[SOFT_IDX])
    rs = X[:, SOFT_IDX].sum(axis=1)
    assert rs.min() < 0.02, f"нижний край не покрыт: min={rs.min():.4f}"
    assert rs.max() > 0.38, f"верхний край не покрыт: max={rs.max():.4f}"
    assert rs.std() > 0.09, f"стратификация не работает: std={rs.std():.4f}"


def test_group_sampling_points_feasible():
    reg = _region()
    X = reg.random_points(1500, seed=3, groups=[SOFT_IDX])
    assert np.allclose(X.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(X >= np.asarray(LO6) - 1e-9)
    assert np.all(X <= np.asarray(UP6) + 1e-9)


def test_group_sampling_backward_compat_bitwise():
    """groups=None и отсутствие параметра — идентичные точки (тот же RNG-путь)."""
    reg = _region()
    a = reg.random_points(60, seed=7)
    b = reg.random_points(60, seed=7, groups=None)
    assert np.array_equal(a, b)


def test_group_sampling_deterministic_by_seed():
    reg = _region()
    a = reg.random_points(100, seed=5, groups=[SOFT_IDX])
    b = reg.random_points(100, seed=5, groups=[SOFT_IDX])
    c = reg.random_points(100, seed=6, groups=[SOFT_IDX])
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_group_sampling_multiple_groups_feasible():
    """Две непересекающиеся группы + «остальные» — все точки допустимы."""
    reg = _region()
    X = reg.random_points(500, seed=1, groups=[[2, 3], [4, 5]])
    assert np.allclose(X.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(X <= np.asarray(UP6) + 1e-9)
    r23 = X[:, [2, 3]].sum(axis=1)
    assert r23.max() > 0.18 and r23.min() < 0.02   # края Σ∈[0, 0.20]


def test_group_sampling_validation_errors():
    reg = _region()
    with pytest.raises(ValueError):                 # пересечение групп
        reg.random_points(10, seed=0, groups=[[2, 3], [3, 4]])
    with pytest.raises(ValueError):                 # индекс вне диапазона
        reg.random_points(10, seed=0, groups=[[2, 99]])
    with pytest.raises(ValueError):                 # пустая группа
        reg.random_points(10, seed=0, groups=[[]])


def test_narrowing_split_respects_bounds_and_sum():
    rng = np.random.default_rng(0)
    lo = np.array([0.0, 0.0, 0.0, 0.0])
    hi = np.array([0.10, 0.10, 0.10, 0.10])
    for total in (0.0, 0.05, 0.2, 0.4):
        x = _narrowing_split(lo, hi, total, rng)
        assert abs(x.sum() - total) < 1e-9
        assert np.all(x >= -1e-12) and np.all(x <= 0.10 + 1e-12)
    with pytest.raises(ValueError):                 # сумма недостижима
        _narrowing_split(lo, hi, 0.5, rng)


# ----------------------------------------------------------------------
# Runner: проектные группы (априорное знание) — seed-пул
# ----------------------------------------------------------------------
def test_runner_project_groups_seed_coverage():
    r = _runner()
    r.set_mixture_sampling_groups([SOFT_NAMES])
    X = r.propose_seed(400, seed=11)
    rs = X[:, SOFT_IDX].sum(axis=1)
    assert rs.min() < 0.05 and rs.max() > 0.35
    assert np.allclose(X.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(X <= np.asarray(UP6) + 1e-9)


def test_runner_without_groups_unchanged():
    """Проект без групп ведёт себя как раньше (и после сброса [])."""
    a = _runner().propose_seed(50, seed=9)
    r = _runner()
    r.set_mixture_sampling_groups([SOFT_NAMES])
    r.set_mixture_sampling_groups([])               # сброс
    b = r.propose_seed(50, seed=9)
    assert np.array_equal(a, b)


def test_runner_group_validation_errors():
    r = _runner()
    with pytest.raises(KeyError):                   # неизвестный компонент
        r.set_mixture_sampling_groups([["PBNK", "NOPE"]])
    with pytest.raises(ValueError):                 # пересечение
        r.set_mixture_sampling_groups([["PBNK", "CPE"], ["CPE"]])


# ----------------------------------------------------------------------
# Ветка: группы как намерение + наследование проект→ветка
# ----------------------------------------------------------------------
def test_branch_groups_intent_and_inheritance():
    r = _runner()
    r.set_mixture_sampling_groups([SOFT_NAMES])
    r.seed_initial(n=12, seed=0)
    goal = {"modulus": DesirabilitySpec(kind="max", low=40.0, high=100.0)}
    b_inherit = r.add_branch("наследует", goal, budget=6)
    b_own = r.add_branch("своя группа", goal, budget=6,
                         sampling_groups=[["PBNK", "CPE"]])
    assert b_inherit.sampling_groups is None        # None → проектные
    assert b_own.sampling_groups == [["PBNK", "CPE"]]

    # пул ветки-наследника стратифицирован ПРОЕКТНЫМИ группами
    cand = r._phase_candidates(500, 42, groups=b_inherit.sampling_groups)
    rs = cand[:, SOFT_IDX].sum(axis=1)
    assert rs.max() > 0.35 and rs.min() < 0.05

    # смена намерения ветки позже (после скрининга)
    r.set_branch_sampling_groups(b_own.id, None)    # вернуть наследование
    assert r.branches[b_own.id].sampling_groups is None
    r.set_branch_sampling_groups(b_own.id, [SOFT_NAMES])
    assert r.branches[b_own.id].sampling_groups == [SOFT_NAMES]
    with pytest.raises(KeyError):
        r.set_branch_sampling_groups("нет_такой", [SOFT_NAMES])


def test_branch_propose_points_with_groups_feasible():
    r = _runner()
    r.seed_initial(n=12, seed=0)
    goal = {"modulus": DesirabilitySpec(kind="max", low=40.0, high=100.0)}
    br = r.add_branch("b", goal, budget=6, sampling_groups=[SOFT_NAMES])
    X = r.propose_points(br.id, n_points=3, n_candidates=200)
    assert X.shape == (3, 6)
    assert np.allclose(X.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(X <= np.asarray(UP6) + 1e-6)


def test_locked_group_member_held_not_stratified():
    """Запертый компонент группы держится на значении, точки допустимы."""
    lo = list(LO6); hi = list(UP6)
    lo[5] = hi[5] = 0.05                            # SBM заперт на 0.05
    schema = ProjectSchema.mixture_only(
        NAMES6, lower=lo, upper=hi,
        model=ModelSpec(cross_level="additive", mixture_order="quadratic"))
    r = MixtureProcessRunner(schema, _Oracle(), seed=0, n_restarts=2)
    r.set_mixture_sampling_groups([SOFT_NAMES])     # SBM в группе, но заперт
    X = r.propose_seed(200, seed=4)
    assert np.allclose(X[:, 5], 0.05, atol=1e-9)
    assert np.allclose(X.sum(axis=1), 1.0, atol=1e-9)
    r_free = X[:, [2, 3, 4]].sum(axis=1)            # свободная часть ниши
    assert r_free.max() > 0.28 and r_free.min() < 0.03


# ----------------------------------------------------------------------
# Сериализация: Branch и state кампании
# ----------------------------------------------------------------------
def test_branch_state_roundtrip_groups():
    goal = {"modulus": DesirabilitySpec(kind="max", low=40.0, high=100.0)}
    b1 = Branch(id="b1", name="с группами", goal=goal,
                sampling_groups=[["PBNK", "CPE"]])
    b2 = Branch(id="b2", name="наследует", goal=goal)   # None
    r1 = Branch.from_state(b1.to_state())
    r2 = Branch.from_state(b2.to_state())
    assert r1.sampling_groups == [["PBNK", "CPE"]]
    assert r2.sampling_groups is None
    # старый state без ключа (обратная совместимость)
    old = b1.to_state()
    del old["sampling_groups"]
    assert Branch.from_state(old).sampling_groups is None


def test_campaign_state_roundtrip_groups():
    r = _runner()
    r.set_mixture_sampling_groups([SOFT_NAMES])
    r.seed_initial(n=12, seed=0)
    goal = {"modulus": DesirabilitySpec(kind="max", low=40.0, high=100.0)}
    r.add_branch("b", goal, budget=6, branch_id="b1",
                 sampling_groups=[["PBNK", "CPE"]])
    state = runner_to_state(r)
    r2 = runner_from_state(state, oracle=_Oracle())
    assert r2.sampling_groups == [SOFT_NAMES]
    assert r2.branches["b1"].sampling_groups == [["PBNK", "CPE"]]
    # старый state без ключа — дефолт пусто
    del state["runner"]["sampling_groups"]
    r3 = runner_from_state(state, oracle=_Oracle())
    assert r3.sampling_groups == []


# ----------------------------------------------------------------------
# UI: чистые хелперы формы (без Streamlit-рендера)
# ----------------------------------------------------------------------
def test_ui_parse_groups_roundtrip():
    from src.apps.campaign_ui import (parse_sampling_groups,
                                      sampling_groups_to_text)
    txt = "PBNK, CPE\nDL, SBM\n\n"
    groups = parse_sampling_groups(txt)
    assert groups == [["PBNK", "CPE"], ["DL", "SBM"]]
    assert parse_sampling_groups("") == []
    assert sampling_groups_to_text(groups) == "PBNK, CPE\nDL, SBM"
    assert parse_sampling_groups(sampling_groups_to_text(groups)) == groups


def test_ui_setup_prefill_contains_groups():
    from src.apps.campaign_ui import setup_prefill_from_runner
    r = _runner()
    r.set_mixture_sampling_groups([SOFT_NAMES])
    out = setup_prefill_from_runner(r)
    assert out["setup_groups"] == ", ".join(SOFT_NAMES)
