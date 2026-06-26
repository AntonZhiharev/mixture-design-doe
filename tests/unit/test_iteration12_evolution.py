# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 12 / §13.7: эволюция схемы + миграция точек (прогрессия 3 стадий).

Прогрессия (как просил архитектурный обзор):
  Стадия 1 — mixture-only;
  Стадия 2 — + PROCESS-блок (mixture-process), старые точки мигрируют по политике;
  Стадия 3 — + новый PARAMETER (process-переменная) + новый ОТКЛИК (target),
             старые точки: unknown → отбрасываются, known-constant → переиспользуются,
             новый отклик → Y[y_new]=MISSING.

Проверяем канон §13.7: версионирование, явные политики (никаких молчаливых 0/средних),
асимметрия «новый ЧЛЕН из старых переменных (бесплатно) vs новая ПЕРЕМЕННАЯ (политика)».
"""
import numpy as np
import pytest

from src.core.schema import (MIXTURE, PROCESS, MISSING, is_missing,
                             VariableBlock, ModelSpec, ResponseSpec,
                             ProjectSchema, DataPoint)
from src.core.schema_evolution import (
    SchemaHistory, evolve_schema, migrate_point, select_fixed_rows,
    known_constant, unknown, recompute)
from src.design.block_model import build_model_terms
from src.design.augmented import augmented_design


# ----------------------------------------------------------------------
# Фикстуры прогрессии
# ----------------------------------------------------------------------
def _stage1():
    return ProjectSchema.mixture_only(
        ["A", "B", "C"], responses=[ResponseSpec("y1", "max")])


def _v1_points(s1, n=6):
    rng = np.random.default_rng(0)
    pts = []
    for _ in range(n):
        x = rng.dirichlet(np.ones(3))
        pts.append(DataPoint(schema_version=s1.version,
                             X={MIXTURE: list(x)}, Y={"y1": float(rng.random())}))
    return pts


# ----------------------------------------------------------------------
# evolve_schema — версии, рост блока/модели
# ----------------------------------------------------------------------
def test_evolve_bumps_version_and_grows_process():
    s1 = _stage1()
    s2 = evolve_schema(s1, add_process=[("T", 100, 200), ("t", 10, 20)],
                       migration={"T": known_constant(150), "t": known_constant(15)})
    assert s2.version == 2
    assert s2.process_names == ("T", "t")
    assert build_model_terms(s1).p == 6           # mixture quad
    assert build_model_terms(s2).p == 11          # + process quad(5), cross-main main=∅ → 0
    s3 = evolve_schema(s2, add_process=[("pH", 3, 9)],
                       add_responses=[ResponseSpec("y2", "max")],
                       migration={"pH": unknown()})
    assert s3.version == 3
    assert s3.process_names == ("T", "t", "pH")
    assert s3.response_names == ("y1", "y2")
    assert build_model_terms(s3).p == 15          # mixture 6 + process quad(d=3 → 9)


def test_schema_history_immutable():
    s1 = _stage1()
    h = SchemaHistory.start(s1)
    s2 = h.add(evolve_schema(s1, add_process=[("T", 100, 200), ("t", 10, 20)]))
    assert h.get(1) is s1 and h.get(2) is s2 and h.latest() is s2
    with pytest.raises(ValueError):
        h.add(s1)                                  # повторная версия запрещена


# ----------------------------------------------------------------------
# Стадия 1 → 2: миграция known-constant
# ----------------------------------------------------------------------
def test_stage2_migrate_known_constant():
    s1 = _stage1()
    s2 = evolve_schema(s1, add_process=[("T", 100, 200), ("t", 10, 20)],
                       migration={"T": known_constant(150), "t": known_constant(15)})
    hist = SchemaHistory.start(s1)
    hist.add(s2)
    pts = _v1_points(s1)
    used, skipped = select_fixed_rows(pts, s2, hist)
    assert len(used) == len(pts) and skipped == []
    mp = used[0]
    assert mp.schema_version == 2 and mp.fixed_in_augment
    assert np.allclose(mp.X[PROCESS], [0.5, 0.5])   # (150-100)/100, (15-10)/10
    assert mp.X[MIXTURE] == pts[0].X[MIXTURE]       # рецепт сохранён
    assert mp.Y["y1"] == pts[0].Y["y1"]             # измеренный отклик сохранён
    mp.validate(s2)                                  # Σx=1 на mixture, z∈[0,1]


# ----------------------------------------------------------------------
# Стадия 2 → 3: новый параметр (unknown→drop / known→reuse) + новый отклик→MISSING
# ----------------------------------------------------------------------
def _stage2_points(s1, s2, hist):
    used, _ = select_fixed_rows(_v1_points(s1), s2, hist)
    return used


def test_stage3_unknown_parameter_drops_points():
    s1 = _stage1()
    s2 = evolve_schema(s1, add_process=[("T", 100, 200), ("t", 10, 20)],
                       migration={"T": known_constant(150), "t": known_constant(15)})
    hist = SchemaHistory.start(s1); hist.add(s2)
    v2 = _stage2_points(s1, s2, hist)
    s3 = evolve_schema(s2, add_process=[("pH", 3, 9)],
                       add_responses=[ResponseSpec("y2", "max")],
                       migration={"pH": unknown()})
    hist.add(s3)
    used, skipped = select_fixed_rows(v2, s3, hist)
    assert used == [] and len(skipped) == len(v2)   # unknown → не в fixed


def test_stage3_known_parameter_reuses_and_new_response_missing():
    s1 = _stage1()
    s2 = evolve_schema(s1, add_process=[("T", 100, 200), ("t", 10, 20)],
                       migration={"T": known_constant(150), "t": known_constant(15)})
    hist = SchemaHistory.start(s1); hist.add(s2)
    v2 = _stage2_points(s1, s2, hist)
    s3 = evolve_schema(s2, add_process=[("pH", 3, 9)],
                       add_responses=[ResponseSpec("y2", "max")],
                       migration={"pH": known_constant(6.0)})
    hist.add(s3)
    used, skipped = select_fixed_rows(v2, s3, hist)
    assert len(used) == len(v2) and skipped == []
    mp = used[0]
    assert np.allclose(mp.X[PROCESS], [0.5, 0.5, 0.5])   # pH: (6-3)/(9-3)=0.5
    assert mp.Y["y1"] == v2[0].Y["y1"]                   # старый отклик сохранён
    assert is_missing(mp.Y["y2"])                        # новый отклик → MISSING (не 0!)
    mp.validate(s3)


def test_recompute_policy():
    s1 = _stage1()
    s2 = evolve_schema(s1, add_process=[("T", 0, 10)],
                       migration={"T": recompute("from_first")})
    hist = SchemaHistory.start(s1); hist.add(s2)
    pts = _v1_points(s1, n=2)
    # recompute: код берём как первая mixture-доля (демонстрация вычислимости из X)
    fns = {"from_first": lambda pt: pt.X[MIXTURE][0]}
    used, skipped = select_fixed_rows(pts, s2, hist, recompute_fns=fns)
    assert len(used) == 2 and skipped == []
    assert np.isclose(used[0].X[PROCESS][0], pts[0].X[MIXTURE][0])


# ----------------------------------------------------------------------
# Асимметрия §13.7: новый ЧЛЕН из старых переменных → миграция НЕ нужна
# ----------------------------------------------------------------------
def test_model_only_evolution_reuses_points_free():
    s1 = _stage1()
    s2 = evolve_schema(s1, add_process=[("T", 100, 200), ("t", 10, 20)],
                       migration={"T": known_constant(150), "t": known_constant(15)})
    hist = SchemaHistory.start(s1); hist.add(s2)
    v2 = _stage2_points(s1, s2, hist)
    # эволюция ТОЛЬКО модели (full-cross): новые ЧЛЕНЫ из тех же x,z — не переменные
    s2b = evolve_schema(s2, model=ModelSpec(cross_level="full-cross",
                                            main_components=()))
    hist.add(s2b)
    used, skipped = select_fixed_rows(v2, s2b, hist)
    assert len(used) == len(v2) and skipped == []         # переиспользуются бесплатно
    assert np.allclose(used[0].X[PROCESS], v2[0].X[PROCESS])
    assert build_model_terms(s2b).p > build_model_terms(s2).p   # модель выросла


# ----------------------------------------------------------------------
# §15.0.4: append mixture-компонента (C) — version+1, грань C=0, миграция
# ----------------------------------------------------------------------
def _stage1_2comp():
    """Фаза 1 (§15.0.4): РЕАЛЬНО 2-компонентный симплекс {A,B}, A+B=1, C нет."""
    return ProjectSchema.mixture_only(
        ["A", "B"], responses=[ResponseSpec("y1", "max")])


def _v1_points_2comp(s1, n=6):
    rng = np.random.default_rng(0)
    pts = []
    for _ in range(n):
        x = rng.dirichlet(np.ones(2))               # [A,B], A+B=1
        pts.append(DataPoint(schema_version=s1.version,
                             X={MIXTURE: list(x)}, Y={"y1": float(rng.random())}))
    return pts


def test_append_mixture_bumps_version_and_grows_simplex():
    """append_mixture(C): version+1, состав симплекса 2→3, C-термы в v2 (не в v1)."""
    s1 = _stage1_2comp()
    s2 = evolve_schema(s1, add_mixture=[("C", 0.0, 1.0)],
                       migration={"C": known_constant(0.0)})
    assert s2.version == 2
    assert s1.mixture_names == ("A", "B")
    assert s2.mixture_names == ("A", "B", "C")
    # причина зафиксирована в change_log новой версии
    assert {"append_mixture": "C"} in s2.change_log
    # модель: C-термы появляются только в v2 (фаза 1 — quadratic Scheffé на {A,B})
    n1 = build_model_terms(s1).names
    n2 = build_model_terms(s2).names
    assert "C" not in n1 and "A*C" not in n1
    assert "C" in n2 and "A*C" in n2 and "B*C" in n2


def test_append_mixture_fixed_rows_on_edge_C0():
    """Точки фазы 1 мигрируют на грань C=0: [A,B] → [A,B,0], Σ=1 сходится."""
    s1 = _stage1_2comp()
    s2 = evolve_schema(s1, add_mixture=[("C", 0.0, 1.0)],
                       migration={"C": known_constant(0.0)})
    hist = SchemaHistory.start(s1); hist.add(s2)
    pts = _v1_points_2comp(s1)
    used, skipped = select_fixed_rows(pts, s2, hist)
    assert len(used) == len(pts) and skipped == []
    mp = used[0]
    assert mp.schema_version == 2 and mp.fixed_in_augment
    assert mp.origin_tag.get("migrated_from") == 1
    # грань C=0 (РЕАЛЬНАЯ доля, НЕ 1/3 и НЕ code-трансформ); A,B сохранены
    assert mp.X[MIXTURE][2] == 0.0
    assert mp.X[MIXTURE][:2] == pts[0].X[MIXTURE]
    assert abs(sum(mp.X[MIXTURE]) - 1.0) < 1e-9          # A+B+0=1 ⟺ A+B=1
    assert mp.Y["y1"] == pts[0].Y["y1"]                  # измеренный отклик сохранён
    mp.validate(s2)                                       # Σx=1, доли ≥ 0


def test_append_mixture_unknown_policy_drops_points():
    """Без политики (unknown) для нового mixture-компонента точка не годна."""
    s1 = _stage1_2comp()
    s2 = evolve_schema(s1, add_mixture=[("C", 0.0, 1.0)],
                       migration={"C": unknown()})
    hist = SchemaHistory.start(s1); hist.add(s2)
    pts = _v1_points_2comp(s1)
    used, skipped = select_fixed_rows(pts, s2, hist)
    assert used == [] and len(skipped) == len(pts)


def test_append_mixture_known_constant_is_real_proportion_not_code():
    """ОТЛИЧИЕ от process: mixture known-constant хранит РЕАЛЬНУЮ долю, без
    code-трансформа куба (грань C=0.0 пишется как 0.0, а не (0-lo)/(hi-lo))."""
    s1 = _stage1_2comp()
    # даже при «сдвинутых» границах компонента доля кладётся как value, без кода
    s2 = evolve_schema(s1, add_mixture=[("C", 0.0, 0.5)],
                       migration={"C": known_constant(0.0)})
    hist = SchemaHistory.start(s1); hist.add(s2)
    mp = select_fixed_rows(_v1_points_2comp(s1, n=1), s2, hist)[0][0]
    assert mp.X[MIXTURE][2] == 0.0       # доля как есть; process бы дал (0-0)/0.5=0 тоже,
    # но семантика именно «доля», проверяем ненулевой value напрямую:
    s2b = evolve_schema(s1, add_mixture=[("C", 0.0, 1.0)],
                        migration={"C": known_constant(0.2)})
    histb = SchemaHistory.start(s1); histb.add(s2b)
    # точка с A=B=0.4 (Σ=0.8) + C=0.2 → Σ=1 валидна
    p = DataPoint(schema_version=1, X={MIXTURE: [0.4, 0.4]}, Y={"y1": 1.0})
    mpb = select_fixed_rows([p], s2b, histb)[0][0]
    assert mpb.X[MIXTURE][2] == 0.2      # РЕАЛЬНАЯ доля, не code
    assert abs(sum(mpb.X[MIXTURE]) - 1.0) < 1e-9


# ----------------------------------------------------------------------
# Связка с §13.6: добор поверх мигрированных точек (общий механизм)
# ----------------------------------------------------------------------

def test_augmented_design_on_migrated_points():
    s1 = _stage1()
    s2 = evolve_schema(s1, add_process=[("T", 100, 200), ("t", 10, 20)],
                       migration={"T": known_constant(150), "t": known_constant(15)})
    hist = SchemaHistory.start(s1); hist.add(s2)
    v2 = _stage2_points(s1, s2, hist)
    res, used, skipped = augmented_design(
        s2, v2, n_max=6, margin=4, n_random=80, n_mc=1500, seed=1)
    assert len(used) == len(v2) and skipped == []
    assert res.p == build_model_terms(s2).p
    if len(res.new_points):
        assert np.allclose(res.new_points[:, :3].sum(axis=1), 1.0)   # mixture
        assert np.all(res.new_points[:, 3:] >= -1e-9)
        assert np.all(res.new_points[:, 3:] <= 1.0 + 1e-9)
