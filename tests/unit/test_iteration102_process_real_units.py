# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 102 — PROCESS-координаты точки хранятся в ФИЗИЧЕСКИХ единицах.

Находка (проба 16.09.2026 на ручном раннере A,B × T∈[150,200], y = T/100):
после ``restrict_bounds("T", 170, 180)`` все 16 точек оставались в активном
пуле (12 из них сняты при T вне [170,180]), а код 0.9, показывавший T=195,
стал «показывать» T=179 при том же прогнозе GP 1.95 (истина 1.79). Причина:
``DataPoint.X["PROCESS"]`` хранил код [0,1], ``point_in_region`` проверял
лишь код ∈ [0,1], а границы переинтерпретировали код — точки «переезжали» с
границами, и измеренный Y приписывался другому режиму (класс §15.0.4).
REBUILD_SPEC §16.2.1.2 (OPEN) утверждал обратное — «для ручной кампании
работает как задумано»; проба это опровергла.

Что проверяем:
  1. ядро: ``validate``/``point_in_region`` — вхождение в ``[lo, hi]`` блока;
     ``composite_coords`` кодирует под схему-аргумент; ``migrate_point`` кладёт
     ``known-constant`` реальным значением; вырожденная ось → код 0 / ``lo``;
  2. раннер: точка хранит физику режима; ``move_region`` по process-оси
     ИСКЛЮЧАЕТ точки вне новых границ и ВОЗВРАЩАЕТ их при обратном relax;
     GP после сужения предсказывает физику, а не переехавший код; зажатая
     ось даёт кандидатов ровно на значении; ``_to_full``/``_from_full_process``
     согласованы через физику;
  3. персистентность: сейв нового формата несёт маркер ``process_units:
     real``; СТАРЫЙ сейв (без маркера, код) конвертируется по границам версии
     точки — и совпадает с тем, что UI показывал до iter102;
  4. черновик плана (незафиксированный seed): ``recode_seed_plan`` сохраняет
     физику при движении границ и считает строки вне новых границ; без
     сохранённых границ — план как есть.
"""
import json
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.apps import campaign_state as cst
from src.apps.campaign import CampaignController
from src.apps.campaign_ui import (build_setup_runner, process_bounds_of,
                                  process_code_to_real, recode_seed_plan)
from src.core.schema import (MIXTURE, PROCESS, DataPoint, ProjectSchema,
                             VariableBlock, composite_coords,
                             point_dict_process_units, process_in_bounds)
from src.core.schema_evolution import (evolve_schema, known_constant,
                                       migrate_point, point_from_legacy_code,
                                       point_in_region)
from src.optimize.desirability import DesirabilitySpec

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _schema(t_lo=100.0, t_hi=200.0):
    return ProjectSchema.mixture_process(
        VariableBlock.mixture(["A", "B", "C"]),
        VariableBlock.process(["T", "t"], [t_lo, 10.0], [t_hi, 20.0]))


def _manual_runner(seed=3):
    return build_setup_runner(
        mixture_names=["A", "B"], process_names=["T"],
        process_lower=[150.0], process_upper=[200.0],
        response_names=["y"], seed=seed)


def _seed_linear_in_T(r, n=16, seed=3):
    """Снять n точек ручным путём; y = T/100 — чистая функция реальной T."""
    X = r._phase_candidates(n, seed)
    T = process_code_to_real(r, X)[:, 2]
    r.commit_seed(X, (T / 100.0).reshape(-1, 1))
    return X, T


# ======================================================================
# 1. Ядро: физика в точке, код под схему
# ======================================================================
class TestCoreContract:
    def test_validate_checks_block_bounds_not_unit_interval(self):
        s = _schema()
        DataPoint(1, {MIXTURE: [0.2, 0.3, 0.5], PROCESS: [140.0, 16.0]}).validate(s)
        with pytest.raises(ValueError, match="вне границ"):
            DataPoint(1, {MIXTURE: [0.2, 0.3, 0.5], PROCESS: [0.4, 0.6]}).validate(s)
        with pytest.raises(ValueError, match="вне границ"):
            DataPoint(1, {MIXTURE: [0.2, 0.3, 0.5], PROCESS: [201.0, 16.0]}).validate(s)

    def test_process_in_bounds_tolerance_is_relative_to_span(self):
        pb = VariableBlock.process(["T"], [100.0], [200.0])
        assert process_in_bounds(pb, [200.0 + 1e-5], tol=1e-6)        # 1e-5 < 1e-4
        assert not process_in_bounds(pb, [200.0 + 1e-3], tol=1e-6)
        pinned = VariableBlock.process(["T"], [170.0], [170.0])
        assert process_in_bounds(pinned, [170.0 + 5e-7], tol=1e-6)     # абсолютный
        assert not process_in_bounds(pinned, [170.01], tol=1e-6)
        assert not process_in_bounds(pb, [150.0, 1.0])                 # длина

    def test_composite_coords_encodes_under_given_schema(self):
        pt = DataPoint(1, {MIXTURE: [0.2, 0.3, 0.5], PROCESS: [140.0, 16.0]})
        assert np.allclose(composite_coords(_schema(), pt)[3:], [0.4, 0.6])
        assert np.allclose(composite_coords(_schema(120.0, 160.0), pt)[3:],
                           [0.5, 0.6])
        # вне границ — код вне [0,1], без клипа (сигнал, не подмена)
        assert composite_coords(_schema(150.0, 200.0), pt)[3] < 0.0

    def test_point_in_region_uses_physical_bounds(self):
        pt = DataPoint(1, {MIXTURE: [0.2, 0.3, 0.5], PROCESS: [140.0, 16.0]})
        assert point_in_region(pt, _schema())
        assert not point_in_region(pt, _schema(150.0, 200.0))
        assert point_in_region(pt, _schema(140.0, 140.0))        # зажата ровно на 140

    def test_degenerate_axis_codes_to_zero_and_decodes_to_lower(self):
        pb = VariableBlock.process(["T", "t"], [170.0, 10.0], [170.0, 20.0])
        assert np.allclose(pb.to_code([170.0, 15.0]), [0.0, 0.5])
        # любой код на зажатой оси → lo (раньше: lo + код, «T = 170.9»)
        assert np.allclose(pb.from_code([0.9, 0.5]), [170.0, 15.0])
        assert np.allclose(pb.from_code(pb.to_code([170.0, 12.0])), [170.0, 12.0])

    def test_migrate_known_constant_stores_real_value(self):
        s1 = ProjectSchema.mixture_only(["A", "B", "C"])
        s2 = evolve_schema(s1, add_process=[("T", 100.0, 200.0)],
                           migration={"T": known_constant(150.0)})
        pt = DataPoint(1, {MIXTURE: [0.2, 0.3, 0.5]}, Y={"y": 1.0})
        mp = migrate_point(pt, s1, s2)
        assert mp.X[PROCESS] == [150.0]
        assert np.allclose(composite_coords(s2, mp)[3], 0.5)
        # границы ЦЕЛИ на хранимое значение не влияют: та же миграция под
        # другие границы даёт то же 150, другой код
        s2b = evolve_schema(s1, add_process=[("T", 140.0, 160.0)],
                            migration={"T": known_constant(150.0)})
        mp_b = migrate_point(pt, s1, s2b)
        assert mp_b.X[PROCESS] == [150.0]
        assert np.allclose(composite_coords(s2b, mp_b)[3], 0.5)

    def test_process_code_requires_schema(self):
        pt = DataPoint(1, {MIXTURE: [0.2, 0.3, 0.5], PROCESS: [140.0, 16.0]})
        assert np.allclose(pt.process_real(), [140.0, 16.0])
        assert np.allclose(pt.process_code(_schema()), [0.4, 0.6])
        with pytest.raises(ValueError, match="требует схему"):
            pt.process_code()


# ======================================================================
# 2. Раннер: движение process-границ больше не переезжает точки
# ======================================================================
class TestRunnerMoveRegionProcess:
    def test_points_store_physical_regime(self):
        r = _manual_runner()
        X, T = _seed_linear_in_T(r, n=6)
        stored = np.array([p.X[PROCESS][0] for p in r.points])
        assert np.allclose(stored, T)
        assert np.all((stored >= 150.0) & (stored <= 200.0))
        # матрица X раннера — код под ТЕКУЩУЮ схему, обратим к физике
        assert np.allclose(process_code_to_real(r, r.X)[:, 2], T)

    def test_restrict_excludes_points_outside_and_relax_restores(self):
        r = _manual_runner()
        _, T = _seed_linear_in_T(r, n=16)
        n_out = int(((T < 170.0) | (T > 180.0)).sum())
        assert n_out > 0
        ctrl = CampaignController(r)
        ctrl.restrict_bounds("T", 170.0, 180.0)
        active = r._migrated_points()
        assert len(active) == 16 - n_out                 # выпали ровно вне [170,180]
        assert all(170.0 - 1e-9 <= p.X[PROCESS][0] <= 180.0 + 1e-9 for p in active)
        assert len(r.points) == 16                       # И-1: история цела
        ctrl.relax_bounds("T", 150.0, 200.0)
        assert len(r._migrated_points()) == 16           # вернулись (§15.0.3.3)

    def test_surrogate_predicts_physics_after_restrict(self):
        """До iter102: код 0.9 после сужения означал T=179, а GP отдавал 1.95."""
        r = _manual_runner()
        _seed_linear_in_T(r, n=16)
        CampaignController(r).restrict_bounds("T", 170.0, 180.0)
        probe = np.array([[0.5, 0.5, 0.9]])                # код под [170,180] → T=179
        assert np.isclose(process_code_to_real(r, probe)[0, 2], 179.0)
        mu = float(r.surrogates["y"].predict(probe).mean[0])
        assert abs(mu - 1.79) < 0.05, f"GP {mu:.3f} ≠ истине 1.79 при T=179"

    def test_pinned_axis_candidates_sit_exactly_on_value(self):
        r = _manual_runner()
        CampaignController(r).deactivate_variable("T", value=170.0)
        Xn = r._phase_candidates(5, 7)
        assert np.allclose(process_code_to_real(r, Xn)[:, 2], 170.0)
        # и точка, снятая на зажатой оси, хранит ровно 170
        r.commit_seed(Xn[:2], np.array([[1.7], [1.7]]))
        assert all(p.X[PROCESS][0] == 170.0 for p in r.points)

    def test_to_full_and_back_go_through_physics(self):
        r = _manual_runner()
        _seed_linear_in_T(r, n=8)
        CampaignController(r).restrict_bounds("T", 170.0, 180.0)
        # код фазы 0.0 → T=170 → код ПОЛНОЙ схемы [150,200] = 0.4
        full0 = r._to_full(np.array([0.5, 0.5, 0.0]))
        assert np.isclose(full0[2], 0.4)
        full1 = r._to_full(np.array([0.5, 0.5, 1.0]))     # T=180 → 0.6
        assert np.isclose(full1[2], 0.6)
        assert np.isclose(r._from_full_process(full0[2:])[0], 0.0)
        assert np.isclose(r._from_full_process(full1[2:])[0], 1.0)

    def test_branch_reference_recipe_uses_physics(self):
        """x_best — полный вектор; в координаты фазы переводится через физику."""
        r = _manual_runner()
        _seed_linear_in_T(r, n=10)
        r.add_branch("b", {"y": DesirabilitySpec("max", low=1.0, high=2.0)},
                     budget=5, satisfy_at=1.1, branch_id="b")
        r.branches["b"].x_best = [0.5, 0.5, 0.4]           # полная схема: T=170
        CampaignController(r).restrict_bounds("T", 170.0, 180.0)
        ref = r._branch_reference_recipe("b")
        assert np.isclose(ref[2], 0.0)                     # T=170 = lo фазы


# ======================================================================
# 3. Персистентность: маркер формата + конверсия старого сейва
# ======================================================================
class TestPersistence:
    def test_new_save_carries_marker_and_roundtrips_physics(self):
        r = _manual_runner()
        _, T = _seed_linear_in_T(r, n=6)
        state = json.loads(json.dumps(cst.runner_to_state(r), ensure_ascii=False))
        assert all(point_dict_process_units(d) == "real"
                   for d in state["runner"]["points"])
        r2 = cst.runner_from_state(state)
        assert np.allclose([p.X[PROCESS][0] for p in r2.points], T)
        assert r2.surrogate_coverage() == r.surrogate_coverage()

    def test_legacy_save_code_is_converted_by_version_bounds(self):
        """Старый сейв: PROCESS в коде, маркера нет. Читается по границам версии."""
        r = _manual_runner()
        _, T = _seed_linear_in_T(r, n=6)
        state = json.loads(json.dumps(cst.runner_to_state(r), ensure_ascii=False))
        # эмулируем сейв до iter102: код [0,1] под границы v1, без маркера
        pb = r.current_schema.process_block()
        for d in state["runner"]["points"]:
            d["X"][PROCESS] = [float(v) for v in pb.to_code(d["X"][PROCESS])]
            d.pop("process_units", None)
        r2 = cst.runner_from_state(state)
        assert np.allclose([p.X[PROCESS][0] for p in r2.points], T)
        # повторный сейв — уже новый формат
        again = cst.runner_to_state(r2)
        assert all(point_dict_process_units(d) == "real"
                   for d in again["runner"]["points"])

    def test_legacy_conversion_is_idempotent_on_marker(self):
        s = _schema()
        pt = DataPoint(1, {MIXTURE: [0.2, 0.3, 0.5], PROCESS: [140.0, 16.0]})
        d_new = pt.to_dict()
        assert point_from_legacy_code(d_new, s).X[PROCESS] == [140.0, 16.0]
        d_old = {**d_new, "X": {MIXTURE: [0.2, 0.3, 0.5], PROCESS: [0.4, 0.6]}}
        d_old.pop("process_units")
        assert np.allclose(point_from_legacy_code(d_old, s).X[PROCESS],
                           [140.0, 16.0])
        # несогласованный старый сейв — отказ, не догадка
        bad = {**d_old, "X": {MIXTURE: [0.2, 0.3, 0.5], PROCESS: [0.4]}}
        with pytest.raises(ValueError, match="не согласован"):
            point_from_legacy_code(bad, s)

    def test_legacy_points_of_older_version_use_their_own_bounds(self):
        """Точка v1 в старом сейве кодировалась под границы v1, не текущие."""
        r = _manual_runner()
        _, T = _seed_linear_in_T(r, n=6)
        # v2: новая ось при известной константе — история версий растёт
        CampaignController(r).add_process_var("P", known_constant(3.0),
                                              lower=1.0, upper=5.0)
        state = json.loads(json.dumps(cst.runner_to_state(r), ensure_ascii=False))
        hist = {int(s["version"]): ProjectSchema.from_dict(s)
                for s in state["runner"]["schema_history"]}
        for d in state["runner"]["points"]:
            pb = hist[int(d["schema_version"])].process_block()
            d["X"][PROCESS] = [float(v) for v in pb.to_code(d["X"][PROCESS])]
            d.pop("process_units", None)
        r2 = cst.runner_from_state(state)
        assert np.allclose([p.X[PROCESS][0] for p in r2.points], T)
        mig = r2._migrated_points()
        assert np.allclose([p.X[PROCESS][1] for p in mig], 3.0)   # P = 3 (физика)


# ======================================================================
# 4. Черновик плана: перекодировка при движении границ
# ======================================================================
class TestPendingPlanRecode:
    def test_recode_preserves_physics_and_counts_outside(self):
        r = _manual_runner()
        X = r._phase_candidates(12, 3)
        saved = process_bounds_of(r)
        T_before = process_code_to_real(r, X)[:, 2]
        assert saved == {"T": [150.0, 200.0]}
        CampaignController(r).restrict_bounds("T", 170.0, 180.0)
        Xn, n_out = recode_seed_plan(X, saved, r)
        assert n_out == int(((T_before < 170.0) | (T_before > 180.0)).sum())
        # физика каждой строки сохранена (даже у строк вне новых границ)
        assert np.allclose(process_code_to_real(r, Xn)[:, 2], T_before)
        # mixture-часть не тронута
        assert np.allclose(Xn[:, :2], X[:, :2])

    def test_recode_identity_without_saved_bounds_or_same_bounds(self):
        r = _manual_runner()
        X = r._phase_candidates(5, 3)
        Xs, n = recode_seed_plan(X, None, r)
        assert n == 0 and np.allclose(Xs, X)
        Xs, n = recode_seed_plan(X, process_bounds_of(r), r)
        assert n == 0 and np.allclose(Xs, X)
        # состав осей сменился — план чужой схемы, не трогаем
        Xs, n = recode_seed_plan(X, {"rotor_Hz": [40.0, 60.0]}, r)
        assert n == 0 and np.allclose(Xs, X)

    def test_committed_plan_matches_printed_regime(self):
        """Наряд напечатан при границах v1; фиксация под теми же границами
        даёт в базе РОВНО те T, что были в наряде."""
        r = _manual_runner()
        X = r._phase_candidates(8, 3)
        printed_T = process_code_to_real(r, X)[:, 2]
        r.commit_seed(X, (printed_T / 100.0).reshape(-1, 1))
        assert np.allclose([p.X[PROCESS][0] for p in r.points], printed_T)
