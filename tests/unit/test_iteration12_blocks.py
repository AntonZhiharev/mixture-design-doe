# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 12 / §13.3-§13.4: блочный генератор термов и поблочная геометрия.

Проверяемый канон (REBUILD_SPEC §13.3 / §13.4):
  * единый генератор термов, режим = по наличию блоков (нет ``if mode``);
  * Scheffé БЕЗ intercept; process-only С intercept;
  * кросс-члены x_i·z_k присутствуют при cross-main/full-cross;
  * mixture-only: модельная матрица == явная Scheffé-матрица (фундамент бит-в-бит);
  * проекция/валидатор поблочные: симплекс → только x, clip [0,1] → только z;
  * пул кандидатов = декартово произведение блоков; вырожденные режимы — частные случаи.
"""
import numpy as np
import pytest

from src.core.schema import (
    MIXTURE, PROCESS, VariableBlock, ModelSpec, ProjectSchema)
from src.design.block_model import (
    build_model_terms, model_matrix, count_params, resolve_model_for_budget)
from src.core import block_geometry as bg


def _mp_schema(model=None):
    return ProjectSchema.mixture_process(
        VariableBlock.mixture(["A", "B", "C"]),
        VariableBlock.process(["T", "t"], [100, 10], [200, 20]),
        model=model)


# ----------------------------------------------------------------------
# §13.3 генератор термов
# ----------------------------------------------------------------------
def test_mixture_only_matrix_equals_explicit_scheffe():
    """Фундамент бит-в-бит: model_matrix == явная Scheffé-матрица (q=3, quad)."""
    s = ProjectSchema.mixture_only(["A", "B", "C"])  # дефолт mixture_order=quadratic
    mt = build_model_terms(s)
    assert mt.terms == [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]
    assert not mt.has_intercept                      # Scheffé без intercept
    X = np.array([[0.2, 0.3, 0.5], [1.0, 0.0, 0.0], [0.5, 0.25, 0.25]])
    ref = np.column_stack([
        X[:, 0], X[:, 1], X[:, 2],
        X[:, 0] * X[:, 1], X[:, 0] * X[:, 2], X[:, 1] * X[:, 2]])
    assert np.allclose(model_matrix(s, X), ref)


def test_process_only_has_intercept_and_param_count():
    s = ProjectSchema.process_only(["T", "t"], [100, 10], [200, 20])  # quad
    mt = build_model_terms(s)
    assert mt.has_intercept
    assert mt.breakdown().get("mixture", 0) == 0
    # intercept(1) + linear(2) + squares(2) + interaction(1) = 6
    assert mt.p == 6 and count_params(s) == 6
    Z = np.array([[0.0, 0.0], [0.5, 1.0]])
    M = model_matrix(s, Z)
    assert np.allclose(M[:, 0], 1.0)                 # первый столбец — intercept


def test_no_intercept_when_mixture_present():
    assert not build_model_terms(_mp_schema()).has_intercept


def test_cross_terms_levels():
    # cross-main с пустыми main → кросс нет (адаптивно как additive)
    s0 = _mp_schema(ModelSpec(cross_level="cross-main", main_components=()))
    assert build_model_terms(s0).breakdown().get("cross", 0) == 0
    # cross-main с 2 главными компонентами, d=2 → 2*2 = 4 кросс-члена
    s1 = _mp_schema(ModelSpec(cross_level="cross-main", main_components=(0, 1)))
    assert build_model_terms(s1).breakdown()["cross"] == 4
    # full-cross: q=3 * d=2 = 6
    s2 = _mp_schema(ModelSpec(cross_level="full-cross"))
    assert build_model_terms(s2).breakdown()["cross"] == 6
    # имена кросс-членов вида "A:T"
    names = build_model_terms(s1).names
    assert "A:T" in names and "B:t" in names


def test_mixture_process_param_count_default():
    # mixture quad(6) + process quad(5) + cross-main(main пуст → 0) = 11
    assert count_params(_mp_schema()) == 11


def test_resolve_model_for_budget_downgrades():
    s = _mp_schema(ModelSpec(cross_level="full-cross"))   # p=17
    p0 = count_params(s)
    s2, log = resolve_model_for_budget(s, n_runs=12, margin=10)
    assert count_params(s2) < p0
    assert s2.model.cross_level == "additive"
    assert s2.model.process_order == "linear"
    assert len(log["steps"]) == 2 and log["p_final"] < log["p_initial"]


def test_resolve_model_keeps_when_budget_enough():
    s = _mp_schema(ModelSpec(cross_level="cross-main", main_components=(0,)))
    s2, log = resolve_model_for_budget(s, n_runs=100, margin=10)
    assert s2.model.cross_level == "cross-main"
    assert log["steps"] == []


# ----------------------------------------------------------------------
# §13.4 поблочная геометрия
# ----------------------------------------------------------------------
def test_project_is_per_block():
    s = _mp_schema()
    vec = np.array([0.5, 0.5, 0.5, 1.5, -0.2])        # mixture Σ≠1, process вне [0,1]
    pr = bg.project(s, vec)
    assert np.isclose(pr[:3].sum(), 1.0)              # симплекс-проекция → только x
    assert np.all(pr[3:] >= -1e-9) and np.all(pr[3:] <= 1.0 + 1e-9)  # clip → только z
    assert bg.validate(s, pr)


def test_validate_rejects_bad_blocks():
    s = _mp_schema()
    assert bg.validate(s, [0.2, 0.3, 0.5, 0.4, 0.6])
    assert not bg.validate(s, [0.2, 0.3, 0.4, 0.4, 0.6])   # Σx≠1
    assert not bg.validate(s, [0.2, 0.3, 0.5, 1.4, 0.6])   # z вне [0,1]


def test_candidate_pool_mixture_process_feasible():
    s = _mp_schema()
    pool = bg.build_candidate_pool(s, n_random=20, seed=0)
    assert pool.shape[1] == 5                          # q+d
    assert np.allclose(pool[:, :3].sum(axis=1), 1.0)   # mixture-часть на симплексе
    assert np.all(pool[:, 3:] >= -1e-9) and np.all(pool[:, 3:] <= 1.0 + 1e-9)


def test_candidate_pool_mixture_only_on_simplex():
    s = ProjectSchema.mixture_only(["A", "B", "C"])
    pool = bg.build_candidate_pool(s, n_random=20, seed=0)
    assert pool.shape[1] == 3
    assert np.allclose(pool.sum(axis=1), 1.0)
