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

from src.core.schema import (MIXTURE, PROCESS, DataPoint, ProjectSchema,
                             VariableBlock, ModelSpec, schema_diff_vars,
                             schema_diff_bounds)
from src.core.schema_evolution import (SchemaHistory, evolve_schema,
                                       select_fixed_rows, known_constant)
from src.core.simplex import SimplexRegion
from src.design.block_model import build_model_terms
from src.optimize.desirability import DesirabilitySpec, optimize_desirability

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


# ----------------------------------------------------------------------
# §15.3 шаг 2a — schema_diff_bounds ортогонален schema_diff_vars (Предусл.4)
# ----------------------------------------------------------------------
def _phase_schemas():
    """old: фаза 1 (C заперт [1/3,1/3], process только T);
    new: C раскрыт [0,1] (релаксация области) + добавлен process P (append)."""
    mix_old = VariableBlock.mixture(["A", "B", "C"],
                                    lower=[0.0, 0.0, 1 / 3], upper=[1.0, 1.0, 1 / 3])
    proc_old = VariableBlock.process(["T"], lower=[0.0], upper=[1.0])
    old = ProjectSchema.mixture_process(mix_old, proc_old, version=1)

    mix_new = VariableBlock.mixture(["A", "B", "C"],
                                    lower=[0.0, 0.0, 0.0], upper=[1.0, 1.0, 1.0])
    proc_new = VariableBlock.process(["T", "P"], lower=[0.0, 0.0], upper=[1.0, 1.0])
    new = ProjectSchema.mixture_process(mix_new, proc_new, version=2)
    return old, new


def test_diff_vars_and_bounds_are_orthogonal():
    old, new = _phase_schemas()
    # ось переменных: видит ТОЛЬКО добавленный process P, mixture-состав не менялся
    dv = schema_diff_vars(old, new)
    assert dv[PROCESS] == ["P"] and dv[MIXTURE] == []
    # ось границ: видит ТОЛЬКО переребаунденный C; T не менялся, P не общий → пусто
    db = schema_diff_bounds(old, new)
    assert set(db[MIXTURE]) == {"C"}
    assert db[PROCESS] == {}
    olo, ohi, nlo, nhi = db[MIXTURE]["C"]
    assert (olo, ohi) == (1 / 3, 1 / 3) and (nlo, nhi) == (0.0, 1.0)


def test_diff_bounds_symmetric_detects_narrowing():
    old, new = _phase_schemas()
    db = schema_diff_bounds(new, old)   # обратное направление: сужение C
    assert set(db[MIXTURE]) == {"C"}
    olo, ohi, nlo, nhi = db[MIXTURE]["C"]
    assert (olo, ohi) == (0.0, 1.0) and (nlo, nhi) == (1 / 3, 1 / 3)


def test_diff_bounds_empty_when_no_change():
    old, _ = _phase_schemas()
    assert schema_diff_bounds(old, old) == {MIXTURE: {}, PROCESS: {}}


# ----------------------------------------------------------------------
# §15.3 шаг 2b — evolve relax_bounds + change_log (Предусл.4 / §15.1.5)
# ----------------------------------------------------------------------
def test_evolve_relax_bounds_records_change_log_no_var_added():
    mix = VariableBlock.mixture(["A", "B", "C"],
                                lower=[0.0, 0.0, 1 / 3], upper=[1.0, 1.0, 1 / 3])
    proc = VariableBlock.process(["T"], lower=[0.0], upper=[1.0])
    s1 = ProjectSchema.mixture_process(mix, proc, version=1)

    s2 = evolve_schema(s1, relax_bounds={"C": (0.0, 1.0)})
    mb = s2.mixture_block()
    assert (mb.lower, mb.upper) == ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    # релаксация — это НЕ добавление переменной (состав симплекса тот же)
    assert schema_diff_vars(s1, s2)[MIXTURE] == [] and schema_diff_vars(s1, s2)[PROCESS] == []
    assert set(schema_diff_bounds(s1, s2)[MIXTURE]) == {"C"}
    # причина зафиксирована в change_log, append-причин нет
    assert {"relax_bounds": "C"} in s2.change_log
    assert all("append_param" not in e for e in s2.change_log)


def test_evolve_atomic_append_and_relax_one_bump_both_logged():
    """§15.1.5: один target несёт +P (append) И C-relax → ОДИН bump, обе причины."""
    mix = VariableBlock.mixture(["A", "B", "C"],
                                lower=[0.0, 0.0, 1 / 3], upper=[1.0, 1.0, 1 / 3])
    proc = VariableBlock.process(["T"], lower=[0.0], upper=[1.0])
    s_after_T = ProjectSchema.mixture_process(mix, proc, version=2)

    s3 = evolve_schema(s_after_T, add_process=[("P", 0.0, 1.0)],
                       migration={"P": known_constant(0.5)},
                       relax_bounds={"C": (0.0, 1.0)})
    assert s3.version == 3                       # ОДИН bump (формально за append P)
    assert s3.process_names == ("T", "P")
    assert {"append_param": "P"} in s3.change_log
    assert {"relax_bounds": "C"} in s3.change_log


# ----------------------------------------------------------------------
# §15.3 шаг 2c — select_fixed_rows резолвит ось within_new_bounds
# ----------------------------------------------------------------------
def test_select_fixed_keeps_baseline_C_point_after_relax():
    """Боевой (§15.4): baseline-C точка фазы 1 ∈ релаксованных bounds → остаётся."""
    mix = VariableBlock.mixture(["A", "B", "C"],
                                lower=[0.0, 0.0, 1 / 3], upper=[1.0, 1.0, 1 / 3])
    proc = VariableBlock.process(["T"], lower=[0.0], upper=[1.0])
    s1 = ProjectSchema.mixture_process(mix, proc, version=1)
    hist = SchemaHistory.start(s1)
    s2 = evolve_schema(s1, relax_bounds={"C": (0.0, 1.0)})
    hist.add(s2)

    pt = DataPoint(schema_version=1, X={MIXTURE: [1 / 3, 1 / 3, 1 / 3],
                                        PROCESS: [0.5]}, Y={})
    used, skipped = select_fixed_rows([pt], s2, hist)
    assert len(used) == 1 and skipped == []
    assert used[0].X[MIXTURE] == [1 / 3, 1 / 3, 1 / 3]   # C сохранён, ∈ релакс [0,1]


def test_select_fixed_drops_point_outside_narrowed_bounds():
    """Негативный (§15.4): точка вне СУЖЕННЫХ bounds исключается, не протекает."""
    mix = VariableBlock.mixture(["A", "B", "C"],
                                lower=[0.0, 0.0, 0.0], upper=[1.0, 1.0, 1.0])
    proc = VariableBlock.process(["T"], lower=[0.0], upper=[1.0])
    s1 = ProjectSchema.mixture_process(mix, proc, version=1)
    hist = SchemaHistory.start(s1)
    s2 = evolve_schema(s1, relax_bounds={"C": (0.0, 0.2)})   # СУЖЕНИЕ C
    hist.add(s2)

    pt = DataPoint(schema_version=1, X={MIXTURE: [1 / 3, 1 / 3, 1 / 3],
                                        PROCESS: [0.5]}, Y={})   # C=1/3 > 0.2
    used, skipped = select_fixed_rows([pt], s2, hist)
    assert used == [] and len(skipped) == 1


# ----------------------------------------------------------------------
# §15.3 шаг 3 — M8-argmax над составной областью (mixture×process), маска
# ----------------------------------------------------------------------
def test_optimize_desirability_process_box_pushes_to_edge():
    """§15.1.4: argmax давит СВОБОДНУЮ process-координату к краю куба (T→1).

    Детерминированно (без GP): предиктор p0 = T (composite-индекс q+0).
    """
    region = SimplexRegion(lower=[0.0, 0.0, 0.0], upper=[1.0, 1.0, 1.0],
                           names=["A", "B", "C"])
    q = 3
    predictors = {"p0": lambda X: np.asarray(X, float)[:, q + 0]}   # = T
    specs = {"p0": DesirabilitySpec("max", low=0.0, high=1.0)}
    res = optimize_desirability(region, predictors, specs,
                                n_candidates=500, refine_iters=200, n_starts=5,
                                seed=0, process_lower=[0.0, 0.0],
                                process_upper=[1.0, 1.0])
    assert res.x.shape == (q + 2,)                 # составной рецепт [A,B,C,T,P]
    assert res.x[q + 0] > 0.9                      # T прижат к верхней границе


def test_optimize_desirability_respects_fixed_process():
    """Закрытая (process_fixed) координата держится на значении, не варьируется."""
    region = SimplexRegion(lower=[0.0, 0.0, 0.0], upper=[1.0, 1.0, 1.0],
                           names=["A", "B", "C"])
    q = 3
    predictors = {"p0": lambda X: np.asarray(X, float)[:, q + 0]}   # = T
    specs = {"p0": DesirabilitySpec("max", low=0.0, high=1.0)}
    res = optimize_desirability(region, predictors, specs,
                                n_candidates=500, refine_iters=200, n_starts=5,
                                seed=0, process_lower=[0.0, 0.0],
                                process_upper=[1.0, 1.0],
                                process_fixed={1: 0.3})     # P закрыт на 0.3
    assert res.x[q + 1] == 0.3                     # P не двигался
    assert res.x[q + 0] > 0.9                      # T всё равно к краю


def test_runner_optimize_xbest_respects_phase_mask():
    """Раннер: M8-argmax в фазе 1 держит закрытые C и T,P на baseline."""
    r = _runner()
    r.set_free(mixture_free=["A", "B"], process_free=[])   # C,T,P закрыты
    r.seed_initial(n=14, seed=3)
    br = r.add_branch("opt", {"p0": DesirabilitySpec("max", low=-5, high=5)},
                      budget=10, satisfy_at=2.0)
    res = r.optimize_xbest(br.id, n_candidates=500, refine_iters=100, n_starts=4)

    # закрытые координаты заперты на baseline независимо от модели
    assert np.isclose(res.x[2], 1 / 3)             # C = baseline
    assert res.x[3] == 0.5 and res.x[4] == 0.5     # T,P = baseline (process_fixed)
    # свободный симплекс: Σ(A,B,C)=1
    assert np.isclose(res.x[:3].sum(), 1.0, atol=1e-6)




