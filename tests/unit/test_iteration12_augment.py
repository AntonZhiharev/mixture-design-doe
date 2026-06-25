# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 12 / §13.6-§13.7: блочный augmented design + schema evolution.

Проверяемый канон:
  * инъекция model_matrix_fn для mixture-only == legacy Scheffé-путь БИТ-В-БИТ
    (та же траектория, та же точка и причина остановки) — M5 не сломан (§13.9);
  * Условие §13.6-A: рассогласование модель/моменты ловится ассертом (негатив);
  * select_fixed_rows (§13.7): новая ПЕРЕМЕННАЯ (нет process-координат) → точка
    не переиспользуется; полная точка → используется;
  * augmented_design — общий механизм: mixture-process И вырожденный process-only
    работают БЕЗ спец-веток; добор остаётся в допустимой области поблочно.
"""
import numpy as np
import pytest

from src.core.schema import (MIXTURE, PROCESS, VariableBlock, ProjectSchema,
                             DataPoint)
from src.core import block_geometry as bg
from src.design.block_model import build_model_terms, model_matrix
from src.design.block_moments import block_moment_matrix
from src.design.i_optimal import i_optimal_augment_sequential
from src.design.augmented import select_fixed_rows, augmented_design


def _mp():
    return ProjectSchema.mixture_process(
        VariableBlock.mixture(["A", "B", "C"]),
        VariableBlock.process(["T", "t"], [100, 10], [200, 20]))


# ----------------------------------------------------------------------
# §13.9 регресс: инъекция == legacy для mixture-only (бит-в-бит)
# ----------------------------------------------------------------------
def test_injection_equals_legacy_mixture_only():
    s = ProjectSchema.mixture_only(["A", "B", "C"])
    pool = bg.build_candidate_pool(s, n_random=80, seed=0)
    mt = build_model_terms(s)
    W = block_moment_matrix(s, n_mc=3000, seed=0, terms=mt)
    existing = pool[:5]

    r_legacy = i_optimal_augment_sequential(
        existing, pool, W, model="quadratic", margin=4, n_max=8, seed=1)

    def mm(X):
        return model_matrix(s, X, terms=mt)

    r_block = i_optimal_augment_sequential(
        existing, pool, W, model_matrix_fn=mm, margin=4, n_max=8, seed=1)

    assert np.array_equal(r_legacy.indices, r_block.indices)
    assert np.isclose(r_legacy.i_final, r_block.i_final, rtol=0, atol=1e-8)
    assert r_legacy.stop_reason == r_block.stop_reason
    assert r_legacy.p == r_block.p
    assert r_legacy.i_history == pytest.approx(r_block.i_history, abs=1e-8)


# ----------------------------------------------------------------------
# §13.6-A: согласованность модель/моменты (негативный тест)
# ----------------------------------------------------------------------
def test_condition_A_rejects_mismatched_moments():
    s = _mp()
    pool = bg.build_candidate_pool(s, n_random=20, seed=0)
    mt = build_model_terms(s)
    W_wrong = np.eye(mt.p + 1)           # размер моментов ≠ числу столбцов модели

    def mm(X):
        return model_matrix(s, X, terms=mt)

    with pytest.raises(ValueError):
        i_optimal_augment_sequential(None, pool, W_wrong, model_matrix_fn=mm)


# ----------------------------------------------------------------------
# §13.7: select_fixed_rows — новая переменная vs полная точка
# ----------------------------------------------------------------------
def test_select_fixed_rows_schema_evolution():
    target = _mp()
    p_old = DataPoint(schema_version=1, X={MIXTURE: [0.3, 0.3, 0.4]})       # нет PROCESS
    p_new = DataPoint(schema_version=2,
                      X={MIXTURE: [0.2, 0.3, 0.5], PROCESS: [0.5, 0.5]})
    fixed, used, skipped = select_fixed_rows([p_old, p_new], target)
    assert used == [p_new]
    assert skipped == [p_old]                # новая переменная z → нужна миграция
    assert fixed.shape == (1, 5)
    assert np.allclose(fixed[0], [0.2, 0.3, 0.5, 0.5, 0.5])


# ----------------------------------------------------------------------
# §13.6: общий механизм — mixture-process и вырожденный process-only
# ----------------------------------------------------------------------
def test_augmented_design_mixture_process():
    target = _mp()
    pool0 = bg.build_candidate_pool(target, n_random=10, seed=2)
    existing = [DataPoint(schema_version=2,
                          X={MIXTURE: list(r[:3]), PROCESS: list(r[3:])})
                for r in pool0[:6]]
    res, used, skipped = augmented_design(
        target, existing, n_max=10, margin=6, n_random=100, n_mc=1500, seed=2)

    assert res.p == build_model_terms(target).p
    assert len(used) == 6 and skipped == []
    assert res.stop_reason in {"sufficiency", "rel_gain", "budget", "pool_exhausted"}
    if len(res.new_points):
        assert np.allclose(res.new_points[:, :3].sum(axis=1), 1.0)        # mixture
        assert np.all(res.new_points[:, 3:] >= -1e-9)                     # process [0,1]
        assert np.all(res.new_points[:, 3:] <= 1.0 + 1e-9)


def test_augmented_design_process_only_degenerate():
    target = ProjectSchema.process_only(["z1", "z2"], [0, 0], [10, 10])
    res, used, skipped = augmented_design(
        target, [], n_max=8, margin=4, n_random=80, n_mc=1500, seed=0)

    assert res.p == build_model_terms(target).p     # intercept+lin+sq+int = 6
    assert res.n_added >= 1
    if len(res.new_points):                          # валидатор НЕ проверяет Σx=1
        assert np.all(res.new_points >= -1e-9)
        assert np.all(res.new_points <= 1.0 + 1e-9)
