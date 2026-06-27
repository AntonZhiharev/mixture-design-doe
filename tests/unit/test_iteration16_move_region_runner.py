# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 16 / §15.0.3 — интеграция примитива move_bounds в раннер.

``MixtureProcessRunner.move_region`` — REGION-операция (движение границ
существующих компонентов БЕЗ bump версии, §15.2.4), построенная на примитиве
``design.move_bounds`` (итерация 16). Проверяем:

  * relax/restrict меняют область, ``schema_version`` НЕ растёт;
  * сужение исключает выпавшие точки из активного pool, но СОХРАНЯЕТ в истории;
  * обратимость: расширили обратно → выпавшие точки вернулись в активный pool;
  * policy=error отклоняет движение, теряющее измеренную точку (без потерь);
  * boundary-проекция запрещена для точек с измеренным Y (класс §15.0.4);
  * rebalance / нарушение симплекс-замкнутости пробрасываются примитивом;
  * журнал _region_moves фиксирует тип/intent/policy.
"""
import numpy as np
import pytest

from src.core.schema import ProjectSchema, VariableBlock, ModelSpec
from src.design.block_model import build_model_terms
from src.design.move_bounds import (MOVE_RELAX, MOVE_RESTRICT, RegionMoveError)
from src.optimize.desirability import DesirabilitySpec
from src.verification.mixture_process_truth import MultiMixtureProcessTruth
from src.apps.mixture_process_runner import MixtureProcessRunner


def _schema():
    mix = VariableBlock.mixture(["A", "B", "C"])
    proc = VariableBlock.process(["T", "P"], lower=[0.0, 0.0], upper=[1.0, 1.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    return ProjectSchema.mixture_process(mix, proc, model=model)


def _oracle(schema, seed=0):
    p = build_model_terms(schema).p
    rng = np.random.default_rng(seed)
    return MultiMixtureProcessTruth(
        schema, {"p0": rng.normal(size=p), "p1": rng.normal(size=p)},
        noise_sd=0.0)


def _runner(n_seed=24, seed=1):
    schema = _schema()
    r = MixtureProcessRunner(schema, _oracle(schema), seed=seed, n_restarts=3,
                             baseline=[1 / 3, 1 / 3, 1 / 3, 0.5, 0.5])
    # полная фаза: A,B,C + T,P открыты — кандидаты по всей составной области
    r.begin_phase(mixture_free=["A", "B", "C"], process_free=["T", "P"])
    r.seed_initial(n=n_seed, seed=seed)
    return r


def _active_A(r):
    """Доли A активных (мигрированных к текущей области) точек."""
    mig = r._migrated_points()
    return np.array([p.X["MIXTURE"][0] for p in mig])



# ----------------------------------------------------------------------
# Версия НЕ растёт; область реально меняется
# ----------------------------------------------------------------------
def test_move_region_relax_keeps_version_changes_bounds():
    r = _runner()
    v0 = r.current_schema_version
    mv = r.move_region({"A": (0.0, 0.5)}, intent="physical_constraint")
    assert mv.move_type == MOVE_RESTRICT
    assert r.current_schema_version == v0                       # §15.2.4: без bump
    mb = r.current_schema.mixture_block()
    assert (mb.lower[0], mb.upper[0]) == (0.0, 0.5)             # область изменилась
    assert len(r.schema_history.versions) == 1                  # история не разрослась


def test_move_region_relax_classified():
    r = _runner()
    r.move_region({"A": (0.0, 0.4)})                            # сначала сузим
    mv = r.move_region({"A": (0.0, 1.0)})                       # расширим
    assert mv.move_type == MOVE_RELAX


# ----------------------------------------------------------------------
# Сужение: выпавшие точки вне активного pool, но в истории; обратимость
# ----------------------------------------------------------------------
def test_restrict_excludes_points_but_keeps_in_history():
    r = _runner()
    n_hist = len(r.points)
    n_active0 = len(r._migrated_points())
    assert n_active0 == n_hist                                  # всё активно изначально

    r.move_region({"A": (0.0, 0.3)})                            # A≤0.3 ⇒ часть выпала
    active_A = _active_A(r)
    assert np.all(active_A <= 0.3 + 1e-9)                       # активны только A≤0.3
    assert len(r.points) == n_hist                              # история НЕ урезана
    assert len(r._migrated_points()) < n_active0                # активный pool сжался


def test_shrink_then_expand_recovers_points():
    r = _runner()
    n_active0 = len(r._migrated_points())
    r.move_region({"A": (0.0, 0.3)})
    assert len(r._migrated_points()) < n_active0                # выпали
    r.move_region({"A": (0.0, 1.0)})                            # расширили обратно
    assert len(r._migrated_points()) == n_active0               # восстановились


def test_fit_surrogates_works_after_restrict():
    """GP переобучается на сжатом активном pool без падения (skipped не фатальны)."""
    r = _runner()
    r.move_region({"A": (0.0, 0.3)})
    assert set(r.surrogates) == {"p0", "p1"}
    assert r.X.shape[0] == len(r._migrated_points())


# ----------------------------------------------------------------------
# Политики выпадающих точек
# ----------------------------------------------------------------------
def test_policy_error_rejects_dropping_measured_point():
    r = _runner()
    assert np.any(_active_A(r) > 0.3)
    with pytest.raises(RegionMoveError):
        r.move_region({"A": (0.0, 0.3)}, policy="error")
    assert r.current_schema.mixture_block().upper[0] == 1.0     # область не тронута


def test_policy_boundary_forbidden_for_measured_Y():
    r = _runner()
    assert np.any(_active_A(r) > 0.3)
    with pytest.raises(RegionMoveError):
        r.move_region({"A": (0.0, 0.3)}, policy="boundary")


# ----------------------------------------------------------------------
# Журнал движений + проброс инвариантов примитива
# ----------------------------------------------------------------------
def test_region_moves_journal_records_intent():
    r = _runner()
    r.move_region({"A": (0.0, 0.5)}, intent="region_of_interest")
    assert r._region_moves[-1]["move_type"] == MOVE_RESTRICT
    assert r._region_moves[-1]["intent"] == "region_of_interest"
    assert r._region_moves[-1]["policy"] == "exclude"


def test_rebalance_propagates_not_implemented():
    r = _runner()
    r.move_region({"A": (0.2, 0.5)})                            # сначала сузим A
    with pytest.raises(NotImplementedError, match="rebalance"):
        # A: lo 0.2→0.0 (relax) + B: hi 1.0→0.9 (restrict) = разнонаправленно
        r.move_region({"A": (0.0, 0.5), "B": (0.0, 0.9)})


def test_closure_violation_propagates():
    r = _runner()
    with pytest.raises(AssertionError):
        r.move_region({"A": (0.6, 1.0), "B": (0.6, 1.0)})       # Σ нижних = 1.2 > 1
