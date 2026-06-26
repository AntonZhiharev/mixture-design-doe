# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 15 / §15 — боевая верификация schema augmentation + M8-argmax.

Реализация ведётся по порядку §15.3 (REBUILD_SPEC_15_battle_augmentation.md).

Шаг 1 (§15.1.2, ловушка хранения): закрытые фазой координаты пишутся в
``DataPoint.X`` РЕАЛЬНЫМ значением (например, ``T=0.5``), а не подразумеваются
маской. Без этого ``select_fixed_rows`` не увидит T у точки фазы 1 и выбросит её.
Здесь фиксируем, что :class:`MixtureProcessRunner` ведёт версионированную базу
``points`` (DataPoint), а numpy-кэши X/Y/origin согласованы с ней.
"""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from src.core.schema import (MIXTURE, PROCESS, ProjectSchema, VariableBlock,
                             ModelSpec)
from src.design.block_model import build_model_terms
from src.optimize.desirability import DesirabilitySpec
from src.verification.mixture_process_truth import MultiMixtureProcessTruth
from src.apps.mixture_process_runner import MixtureProcessRunner

warnings.filterwarnings("ignore", category=ConvergenceWarning)


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


def _runner():
    schema = _schema()
    return MixtureProcessRunner(schema, _oracle(schema), seed=1, n_restarts=3,
                                baseline=[1 / 3, 1 / 3, 1 / 3, 0.5, 0.5])


# ----------------------------------------------------------------------
# §15.3 шаг 1 — baseline пишется в X точки реальным значением
# ----------------------------------------------------------------------
def test_phase1_baseline_stored_as_value_in_datapoint():
    r = _runner()
    # фаза 1: свободны только A,B; C и весь process закрыты на baseline
    r.set_free(mixture_free=["A", "B"], process_free=[])
    r.seed_initial(n=12, seed=2)

    T_idx, P_idx = 0, 1
    assert len(r.points) == 12
    for pt in r.points:
        # §15.1.2: T=0.5 и P=0.5 ЛЕЖАТ в X как значение (не маска)
        assert pt.X[PROCESS][T_idx] == 0.5
        assert pt.X[PROCESS][P_idx] == 0.5
        # закрытый mixture-компонент C тоже зафиксирован значением (baseline 1/3)
        assert np.isclose(pt.X[MIXTURE][2], 1 / 3)
        # свободные A,B варьируют, но Σ=1 сохраняется
        assert np.isclose(sum(pt.X[MIXTURE]), 1.0, atol=1e-9)
        # все точки фазы 1 ссылаются на версию схемы 1
        assert pt.schema_version == 1
        assert pt.origin_tag["origin"] == "seed"

    # свободные A,B действительно варьируют по базе
    A = np.array([pt.X[MIXTURE][0] for pt in r.points])
    B = np.array([pt.X[MIXTURE][1] for pt in r.points])
    assert A.std() > 0.05 and B.std() > 0.05


def test_derived_arrays_consistent_with_points():
    r = _runner()
    r.set_free(mixture_free=["A", "B"], process_free=[])
    r.seed_initial(n=10, seed=4)

    # numpy-кэши — производные от ведущей базы points
    assert r.X.shape == (10, 5)
    assert r.Y.shape == (10, 2)
    assert len(r.origin) == 10 and set(r.origin) == {"seed"}

    # построчная сверка: X[i] == [A,B,C,T,P] точки i, Y[i] == отклики точки i
    for i, pt in enumerate(r.points):
        row = list(pt.X[MIXTURE]) + list(pt.X[PROCESS])
        assert np.allclose(r.X[i], row)
        assert np.allclose(r.Y[i], [pt.Y["p0"], pt.Y["p1"]])


def test_branch_round_appends_versioned_points():
    r = _runner()
    r.set_free(mixture_free=["A", "B"], process_free=[])
    r.seed_initial(n=12, seed=5)
    n0 = len(r.points)

    br = r.add_branch("opt", {"p0": DesirabilitySpec("max", low=-5, high=5)},
                      budget=4, satisfy_at=2.0)
    r.run_branch_round(br.id, n_points=2, n_candidates=200)

    assert len(r.points) == n0 + 2
    assert r.X.shape == (n0 + 2, 5) and r.Y.shape == (n0 + 2, 2)
    # новые точки ветки: origin-тег и версия схемы (фаза 1 ⇒ v1)
    for pt in r.points[n0:]:
        assert pt.origin_tag["origin"] == f"branch:{br.id}"
        assert pt.schema_version == 1
        # process всё ещё на baseline (фаза не открывала T/P)
        assert pt.X[PROCESS] == [0.5, 0.5]
    assert r.origin_counts()[f"branch:{br.id}"] == 2
