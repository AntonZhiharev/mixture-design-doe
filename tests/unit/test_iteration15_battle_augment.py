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

Реализация по §15.3 (REBUILD_SPEC_15_battle_augmentation.md). После ре-архитектуры
раннера (§15.3.6) свобода фазы кодируется СХЕМОЙ, а не маской:

  * +process (T,P) = APPEND в схему (``augment_phase_schema``, version+1,
    миграция старых точек = ``known-constant(baseline)``);
  * +mixture (C)   = APPEND mixture-компонента (``augment_phase_mixture``,
    version+1, §15.0.4; миграция старых точек = ``known-constant(0.0)`` — грань
    симплекса C=0, Σ переопределяется A+B=1 → A+B+C=1);
  * атомарно       = ``augment_phase_atomic`` (один bump, обе причины в логе).

``baseline-as-value`` (§15.1.2) теперь проявляется при миграции: точка фазы k-1
получает РЕАЛЬНОЕ значение закрытого параметра (T=0.5) в составных координатах,
а не MISSING — поэтому переиспользуется как fixed (не выбрасывается).
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
# §15.3 шаг 1 / §15.3.6 — фаза 1 кодируется схемой (mixture-only, C заперт)
# ----------------------------------------------------------------------
def test_phase1_is_mixture_only_two_components():
    """§15.0.4: фаза 1 РЕАЛЬНО 2-компонентна {A,B} — C ОТСУТСТВУЕТ (не заперт 1/3)."""
    r = _runner()
    r.begin_phase(mixture_free=["A", "B"], process_free=[])
    r.seed_initial(n=12, seed=2)

    assert r.current_schema_version == 1
    assert r.dim == 2 and r.q == 2 and r.d == 0   # 2-компонентный симплекс, нет process
    assert "C" not in r.current_schema.mixture_names   # C нет в схеме v1
    assert len(r.points) == 12
    for pt in r.points:
        assert PROCESS not in pt.X           # process-блока в схеме v1 нет
        assert len(pt.X[MIXTURE]) == 2       # 2 координаты {A,B}, БЕЗ C
        assert np.isclose(sum(pt.X[MIXTURE]), 1.0, atol=1e-9)
        assert pt.schema_version == 1
        assert pt.origin_tag["origin"] == "seed"

    A = np.array([pt.X[MIXTURE][0] for pt in r.points])
    B = np.array([pt.X[MIXTURE][1] for pt in r.points])
    assert A.std() > 0.05 and B.std() > 0.05            # свободные A,B варьируют


def test_derived_arrays_consistent_with_points():
    r = _runner()
    r.begin_phase(mixture_free=["A", "B"], process_free=[])
    r.seed_initial(n=10, seed=4)

    assert r.X.shape == (10, 2)              # фаза 1 — 2 mixture-координаты {A,B}
    assert r.Y.shape == (10, 2)
    assert len(r.origin) == 10 and set(r.origin) == {"seed"}
    for i, pt in enumerate(r.points):
        assert np.allclose(r.X[i], pt.X[MIXTURE])
        assert np.allclose(r.Y[i], [pt.Y["p0"], pt.Y["p1"]])


def test_branch_round_appends_versioned_points():
    r = _runner()
    r.begin_phase(mixture_free=["A", "B"], process_free=[])
    r.seed_initial(n=12, seed=5)
    n0 = len(r.points)

    br = r.add_branch("opt", {"p0": DesirabilitySpec("max", low=-5, high=5)},
                      budget=4, satisfy_at=2.0)
    out = r.run_branch_round(br.id, n_points=2, n_candidates=200)

    assert out["added"] == 2                 # бюджет раунда соблюдён (acq+argmax)
    assert len(r.points) == n0 + 2
    for pt in r.points[n0:]:
        assert pt.origin_tag["origin"] == f"branch:{br.id}"
        assert pt.schema_version == 1
        assert PROCESS not in pt.X           # фаза 1 не открывала process
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
    dv = schema_diff_vars(old, new)
    assert dv[PROCESS] == ["P"] and dv[MIXTURE] == []
    db = schema_diff_bounds(old, new)
    assert set(db[MIXTURE]) == {"C"}
    assert db[PROCESS] == {}
    olo, ohi, nlo, nhi = db[MIXTURE]["C"]
    assert (olo, ohi) == (1 / 3, 1 / 3) and (nlo, nhi) == (0.0, 1.0)


def test_diff_bounds_symmetric_detects_narrowing():
    old, new = _phase_schemas()
    db = schema_diff_bounds(new, old)
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
    assert schema_diff_vars(s1, s2)[MIXTURE] == [] and schema_diff_vars(s1, s2)[PROCESS] == []
    assert set(schema_diff_bounds(s1, s2)[MIXTURE]) == {"C"}
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
    assert used[0].X[MIXTURE] == [1 / 3, 1 / 3, 1 / 3]


def test_select_fixed_drops_point_outside_narrowed_bounds():
    """Негативный (§15.4): точка вне СУЖЕННЫХ bounds исключается, не протекает."""
    mix = VariableBlock.mixture(["A", "B", "C"],
                                lower=[0.0, 0.0, 0.0], upper=[1.0, 1.0, 1.0])
    proc = VariableBlock.process(["T"], lower=[0.0], upper=[1.0])
    s1 = ProjectSchema.mixture_process(mix, proc, version=1)
    hist = SchemaHistory.start(s1)
    s2 = evolve_schema(s1, relax_bounds={"C": (0.0, 0.2)})
    hist.add(s2)

    pt = DataPoint(schema_version=1, X={MIXTURE: [1 / 3, 1 / 3, 1 / 3],
                                        PROCESS: [0.5]}, Y={})
    used, skipped = select_fixed_rows([pt], s2, hist)
    assert used == [] and len(skipped) == 1


# ----------------------------------------------------------------------
# §15.3 шаг 3 — M8-argmax над составной областью (mixture×process)
# ----------------------------------------------------------------------
def test_optimize_desirability_process_box_pushes_to_edge():
    """§15.1.4: argmax давит СВОБОДНУЮ process-координату к краю куба (T→1)."""
    region = SimplexRegion(lower=[0.0, 0.0, 0.0], upper=[1.0, 1.0, 1.0],
                           names=["A", "B", "C"])
    q = 3
    predictors = {"p0": lambda X: np.asarray(X, float)[:, q + 0]}   # = T
    specs = {"p0": DesirabilitySpec("max", low=0.0, high=1.0)}
    res = optimize_desirability(region, predictors, specs,
                                n_candidates=500, refine_iters=200, n_starts=5,
                                seed=0, process_lower=[0.0, 0.0],
                                process_upper=[1.0, 1.0])
    assert res.x.shape == (q + 2,)
    assert res.x[q + 0] > 0.9


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
                                process_fixed={1: 0.3})
    assert res.x[q + 1] == 0.3
    assert res.x[q + 0] > 0.9


def test_runner_optimize_xbest_phase1_region():
    """Раннер: M8-argmax в фазе 1 даёт 2-компонентный рецепт {A,B} (C отсутствует)."""
    r = _runner()
    r.begin_phase(mixture_free=["A", "B"], process_free=[])   # 2-компонентный симплекс
    r.seed_initial(n=14, seed=3)
    br = r.add_branch("opt", {"p0": DesirabilitySpec("max", low=-5, high=5)},
                      budget=10, satisfy_at=2.0)
    res = r.optimize_xbest(br.id, n_candidates=500, refine_iters=100, n_starts=4)

    assert res.x.shape == (2,)                     # фаза 1 — только A,B
    assert np.isclose(res.x[:2].sum(), 1.0, atol=1e-6)


# ----------------------------------------------------------------------
# §15.3 шаг 4 — augment_phase_schema: append +T, переиспользование fixed
# ----------------------------------------------------------------------
def test_augment_phase_schema_reuses_phase1_as_fixed():
    """§14/§15.2.1: +T в схему → version+1; точки фазы 1 мигрируют (НЕ выброшены),
    закрытый T записан РЕАЛЬНЫМ значением baseline (§15.1.2), не MISSING."""
    r = _runner()
    r.begin_phase(mixture_free=["A", "B"], process_free=[])
    r.seed_initial(n=14, seed=6)
    n_seed = len(r.points)

    r.augment_phase_schema(["T"])                  # version 1 → 2

    assert r.current_schema_version == 2
    assert r.current_schema.process_names == ("T",)
    assert {"append_param": "T"} in r.current_schema.change_log
    assert r.current_schema.migration["T"]["policy"] == "known-constant"

    # P6: точки версии 1 ЖИВЫ в базе (НЕ переизмерены)
    assert all(pt.schema_version == 1 for pt in r.points)
    assert len(r.points) == n_seed
    # фаза 1 точки мигрированы к v2: T дописан baseline'ом (0.5), а не MISSING.
    # Колонки: A,B (2-комп. симплекс §15.0.4) + T → индекс T = 2.
    assert r.X.shape == (n_seed, 3)                # 2 mixture {A,B} + 1 process T
    assert np.allclose(r.X[:, 2], 0.5)             # §15.1.2 baseline-as-value
    # P4 (§15.1.3): стартовая инфо-матрица fixed-строк непустая
    assert r.start_info_matrix_rank() > 0

    # P5: модель новой схемы содержит T-члены (T и кросс A:T)
    names = build_model_terms(r.current_schema).names
    assert "T" in names and "A:T" in names


def test_augment_phase_schema_new_round_adds_versioned_points():
    """После +T новые точки ветки несут версию 2 и process-координату."""
    r = _runner()
    r.begin_phase(mixture_free=["A", "B"], process_free=[])
    r.seed_initial(n=12, seed=7)
    br = r.add_branch("opt", {"p0": DesirabilitySpec("max", low=-5, high=5)},
                      budget=8, satisfy_at=2.0)
    r.augment_phase_schema(["T"])
    r.run_branch_round(br.id, n_points=2, n_candidates=200)

    new = [pt for pt in r.points if pt.schema_version == 2]
    assert len(new) == 2
    for pt in new:
        assert PROCESS in pt.X and len(pt.X[PROCESS]) == 1
    # P6: версии 1 и 2 сосуществуют в общей базе
    assert {1, 2} <= {pt.schema_version for pt in r.points}


# ----------------------------------------------------------------------
# §15.0.4 — +C = APPEND mixture-компонента: schema_version РАСТЁТ
# ----------------------------------------------------------------------
def test_augment_mixture_C_bumps_version_and_appends():
    """§15.0.4: +C = APPEND нового компонента симплекса → version+1; точки фазы 1
    мигрируют на грань ``known-constant(0.0)`` (а НЕ baseline 1/3)."""
    r = _runner()
    r.begin_phase(mixture_free=["A", "B"], process_free=[])
    r.seed_initial(n=12, seed=8)
    r.augment_phase_schema(["T"])               # version → 2 (есть process)
    v_before = r.current_schema_version
    assert "C" not in r.current_schema.mixture_names    # C ещё нет в схеме

    r.augment_phase_mixture(["C"])              # APPEND C → version+1

    assert r.current_schema_version == v_before + 1      # §15.0.4: version РАСТЁТ
    assert r.current_schema.mixture_names == ("A", "B", "C")
    mb = r.current_schema.mixture_block()
    assert (mb.lower[2], mb.upper[2]) == (0.0, 1.0)      # C — полный диапазон доли
    assert {"append_mixture": "C"} in r.current_schema.change_log
    assert r.current_schema.migration["C"]["policy"] == "known-constant"
    assert r.current_schema.migration["C"]["value"] == 0.0   # грань симплекса C=0
    # точки фазы 1 мигрируют как fixed на грань C=0 (не выброшены, не 1/3)
    assert r.X.shape[0] == len(r.points)
    assert np.allclose(r.X[:, 2], 0.0)          # C-координата мигрированных = 0


# ----------------------------------------------------------------------
# §15.1.5 + §15.0.4 — атомарность фазы 3 (+P append + C append вместе)
# ----------------------------------------------------------------------
def test_atomic_phase_one_bump_both_appends_logged():
    """§15.1.5+§15.0.4: один target несёт +P (append) И +C (append) → ОДИН bump."""
    r = _runner()
    r.begin_phase(mixture_free=["A", "B"], process_free=[])
    r.seed_initial(n=12, seed=9)
    r.augment_phase_schema(["T"])               # → v2
    v_after_T = r.current_schema_version

    r.augment_phase_atomic(["P"], ["C"])        # +P append + C append атомарно

    assert r.current_schema_version == v_after_T + 1      # ОДИН bump
    log = r.current_schema.change_log
    assert {"append_param": "P"} in log and {"append_mixture": "C"} in log
    assert r.current_schema.process_names == ("T", "P")
    assert r.current_schema.mixture_names == ("A", "B", "C")
    mb = r.current_schema.mixture_block()
    assert (mb.lower[2], mb.upper[2]) == (0.0, 1.0)      # C добавлен в той же фазе
    # дизайн остаётся идентифицируемым (I-критерий конечен)
    assert np.isfinite(r.design_i_value())
