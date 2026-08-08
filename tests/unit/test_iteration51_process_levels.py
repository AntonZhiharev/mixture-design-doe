# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 51 / UI_REVISION_SPEC P2.1 — ДИСКРЕТНЫЕ УРОВНИ process-осей (ядро).

До этого шага ядро считало КАЖДУЮ process-ось непрерывным боксом. Для
экструдера с двумя передачами (400/900 об/мин) это давало ровно ту тихую
подмену, против которой A0.6: план предлагал 673 об/мин, оператор ставил
900, а модель училась на 673 — и «оптимум» оказывался недостижимым режимом.

Слой уровней — ПРОЕКЦИЯ на сетку (не отдельная геометрия):

  * уровни задаются в ФИЗИЧЕСКИХ единицах (код зависит от границ оси —
    при их правке уровни «поехали бы» молча);
  * снап — ближайший уровень, при точной равноудалённости МЕНЬШИЙ
    (детерминизм: иначе исход решает ошибка округления);
  * проекция ПОСЛЕ розыгрыша, а не «выбор из списка»: у Соболя поток общий
    на все d координат — подмена сломала бы low-discrepancy покрытие
    ОСТАЛЬНЫХ (непрерывных) осей.

Покрытие: чистый модуль ``design.levels``; политика раннера
(``set_process_levels`` / ``snap_process_axes`` / пул кандидатов / argmax);
персистентность в ``campaign_state`` (включая старые сейвы без ключа).
"""
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.apps import campaign_state as cst
from src.apps.campaign_ui import build_setup_runner
from src.design.levels import (levels_caption, levels_to_code,
                               normalize_levels, snap_matrix_to_levels,
                               snap_to_levels)
from src.optimize.desirability import DesirabilitySpec

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Golden кампании ПВХ: ротор экструдера — две передачи, температура —
# непрерывная ось. Именно эта пара («дискретная + непрерывная») ловит
# ошибку «снапнули всё подряд».
RPM_LEVELS = [400.0, 900.0]


def _runner(*, levels=None, seed: int = 3):
    """Раннер с ДВУМЯ process-осями: T (непрерывная) и rpm (уровни)."""
    r = build_setup_runner(
        mixture_names=["A", "B", "C"],
        process_names=["T", "rpm"],
        process_lower=[150.0, 400.0], process_upper=[200.0, 900.0],
        response_names=["y"], seed=seed)
    if levels is not None:
        r.set_process_levels(levels)
    return r


# ======================================================================
# 1. Чистый модуль: нормализация уровней (A0.6 — явные ошибки)
# ======================================================================
class TestNormalizeLevels:

    def test_sorts_and_returns_floats(self):
        assert normalize_levels([900, 400]) == [400.0, 900.0]

    def test_single_level_allowed(self):
        """«В этой кампании только 900 об/мин» — законная постановка."""
        assert normalize_levels([900.0], lower=400.0, upper=900.0) == [900.0]

    def test_empty_is_error(self):
        # пустой список неотличим от «оси на сетке нет» — выключать надо
        # отсутствием ключа, а не пустотой
        with pytest.raises(ValueError, match="пуст"):
            normalize_levels([])

    def test_duplicates_are_error(self):
        with pytest.raises(ValueError, match="совпадают"):
            normalize_levels([900.0, 900.0 + 1e-12])

    def test_non_finite_is_error(self):
        with pytest.raises(ValueError, match="конечное"):
            normalize_levels([400.0, np.inf])

    def test_non_numeric_is_error(self):
        with pytest.raises(ValueError, match="не число"):
            normalize_levels([400.0, "быстро"])

    def test_out_of_bounds_is_error(self):
        with pytest.raises(ValueError, match="выше верхней"):
            normalize_levels([400.0, 1500.0], lower=400.0, upper=900.0)
        with pytest.raises(ValueError, match="ниже нижней"):
            normalize_levels([100.0, 900.0], lower=400.0, upper=900.0)


# ======================================================================
# 2. Чистый модуль: проекция на сетку
# ======================================================================
class TestSnapToLevels:

    def test_nearest_level(self):
        out = snap_to_levels([401.0, 899.0, 600.0], RPM_LEVELS)
        assert out.tolist() == [400.0, 900.0, 400.0]

    def test_tie_goes_to_smaller(self):
        """Ровно посередине (650) — берём МЕНЬШИЙ уровень (детерминизм)."""
        assert float(snap_to_levels([650.0], RPM_LEVELS)[0]) == 400.0

    def test_outside_grid_clamps_to_edge(self):
        out = snap_to_levels([-10.0, 1e6], RPM_LEVELS)
        assert out.tolist() == [400.0, 900.0]

    def test_idempotent(self):
        once = snap_to_levels([401.0, 899.0, 650.0], RPM_LEVELS)
        twice = snap_to_levels(once, RPM_LEVELS)
        assert np.array_equal(once, twice)

    def test_shape_preserved(self):
        out = snap_to_levels(np.zeros((3, 2)) + 700.0, RPM_LEVELS)
        assert out.shape == (3, 2)

    def test_unsorted_levels_rejected(self):
        with pytest.raises(ValueError, match="возрастанию"):
            snap_to_levels([500.0], [900.0, 400.0])


class TestLevelsToCode:

    def test_endpoints_map_to_unit_interval(self):
        code = levels_to_code(RPM_LEVELS, 400.0, 900.0)
        assert code.tolist() == [0.0, 1.0]

    def test_interior_level(self):
        code = levels_to_code([400.0, 650.0, 900.0], 400.0, 900.0)
        assert np.allclose(code, [0.0, 0.5, 1.0])

    def test_degenerate_span_uses_unit_divisor(self):
        # тот же контракт, что у VariableBlock.to_code (деление на 1)
        assert levels_to_code([5.0], 5.0, 5.0).tolist() == [0.0]


class TestSnapMatrix:

    def test_only_listed_columns_touched(self):
        Z = np.array([[0.3, 0.4], [0.7, 0.6]])
        out = snap_matrix_to_levels(Z, {1: [0.0, 1.0]})
        assert out[:, 0].tolist() == [0.3, 0.7]      # непрерывная ось цела
        assert out[:, 1].tolist() == [0.0, 1.0]

    def test_input_not_mutated(self):
        Z = np.array([[0.3, 0.4]])
        snap_matrix_to_levels(Z, {1: [0.0, 1.0]})
        assert Z[0, 1] == 0.4

    def test_column_out_of_range(self):
        with pytest.raises(IndexError):
            snap_matrix_to_levels(np.zeros((2, 2)), {5: [0.0, 1.0]})


def test_levels_caption_states_absence_explicitly():
    """Пусто ≠ «поле не заполнено»: подпись говорит про непрерывность прямо."""
    assert "непрерывн" in levels_caption({})
    cap = levels_caption({"rpm": RPM_LEVELS})
    assert "rpm" in cap and "400" in cap and "900" in cap


# ======================================================================
# 3. Политика раннера: валидация и снап координат
# ======================================================================
class TestRunnerPolicy:

    def test_set_and_clear(self):
        r = _runner()
        assert r.process_levels == {}
        r.set_process_levels({"rpm": [900.0, 400.0]})
        assert r.process_levels == {"rpm": [400.0, 900.0]}   # отсортировано
        r.set_process_levels(None)
        assert r.process_levels == {}

    def test_unknown_axis_rejected(self):
        r = _runner()
        with pytest.raises(KeyError):
            r.set_process_levels({"нет_такой": [1.0]})

    def test_level_outside_axis_bounds_rejected(self):
        """Недостижимый режим в сетке — ошибка конфигурации, не «подрежем»."""
        r = _runner()
        with pytest.raises(ValueError):
            r.set_process_levels({"rpm": [400.0, 1200.0]})

    def test_snap_process_axes_is_idempotent_and_keeps_mixture(self):
        r = _runner(levels={"rpm": RPM_LEVELS})
        X = r._phase_candidates(16, seed=11)
        once = r.snap_process_axes(X)
        twice = r.snap_process_axes(once)
        assert np.array_equal(once, twice)
        assert np.allclose(once[:, :r.q], X[:, :r.q])       # смесь не тронута

    def test_axis_outside_current_phase_is_ignored(self):
        """Ось, не раскрытая в текущей фазе, выпадает (как имена в группах)."""
        r = _runner()
        r.begin_phase(["A", "B", "C"], ["T"])                # rpm вне фазы
        r.set_process_levels({"rpm": RPM_LEVELS})
        assert r._levels_code_current() == {}
        X = r._phase_candidates(8, seed=5)
        assert X.shape[1] == r.q + 1


# ======================================================================
# 4. Пул кандидатов: дискретная ось на сетке, непрерывная — не тронута
# ======================================================================
class TestCandidatePool:

    def test_discrete_axis_takes_only_levels(self):
        r = _runner(levels={"rpm": RPM_LEVELS})
        X = r._phase_candidates(64, seed=7)
        rpm_code = X[:, r.q + 1]
        assert set(np.unique(np.round(rpm_code, 12)).tolist()) <= {0.0, 1.0}

    def test_continuous_axis_stays_continuous(self):
        r = _runner(levels={"rpm": RPM_LEVELS})
        X = r._phase_candidates(64, seed=7)
        assert len(np.unique(np.round(X[:, r.q], 9))) > 10

    def test_other_columns_bit_identical_to_no_levels_run(self):
        """Проекция одной оси НЕ меняет розыгрыш остальных координат.

        Именно поэтому снап делается ПОСЛЕ розыгрыша Соболя, а не подменой
        его выбором из списка: иначе поток координат уехал бы целиком.
        """
        base = _runner()._phase_candidates(32, seed=9)
        snapped = _runner(levels={"rpm": RPM_LEVELS})._phase_candidates(32, seed=9)
        q = 3
        assert np.array_equal(base[:, :q + 1], snapped[:, :q + 1])
        assert not np.array_equal(base[:, q + 1], snapped[:, q + 1])

    def test_no_levels_means_bit_identical_behaviour(self):
        a = _runner()._phase_candidates(24, seed=4)
        b = _runner(levels={})._phase_candidates(24, seed=4)
        assert np.array_equal(a, b)

    def test_propose_seed_returns_reachable_modes(self):
        r = _runner(levels={"rpm": RPM_LEVELS})
        Xs = r.propose_seed(10)
        assert set(np.unique(np.round(Xs[:, r.q + 1], 12)).tolist()) <= {0.0, 1.0}


# ======================================================================
# 5. argmax: оптимум выдаётся в ДОСТИЖИМОМ режиме
# ======================================================================
def _seeded_runner_with_branch():
    """Раннер с измеренной базой и веткой (для optimize_xbest)."""
    r = _runner(levels={"rpm": RPM_LEVELS}, seed=5)
    X = r.propose_seed(14)
    # истина: оптимум по rpm лежит В СЕРЕДИНЕ (0.5) — соблазн для оптимизатора
    # уехать с сетки максимален; при работающем слое он обязан вернуться на
    # ближайший достижимый уровень
    y = (2.0 * X[:, 0] + X[:, r.q] - 3.0 * (X[:, r.q + 1] - 0.5) ** 2)
    r.commit_seed(X, y.reshape(-1, 1))
    r.add_branch("b", {"y": DesirabilitySpec("max", low=float(y.min()),
                                             high=float(y.max()) + 1.0)},
                 branch_id="b1")
    return r


class TestArgmaxOnLevels:

    def test_xbest_process_axis_on_level(self):
        r = _seeded_runner_with_branch()
        res = r.optimize_xbest("b1", n_candidates=200, refine_iters=40,
                               n_starts=2)
        rpm_code = float(res.x[r.q + 1])
        assert rpm_code in (0.0, 1.0)

    def test_continuous_axis_not_snapped_in_xbest(self):
        r = _seeded_runner_with_branch()
        res = r.optimize_xbest("b1", n_candidates=200, refine_iters=40,
                               n_starts=2)
        t_code = float(res.x[r.q])
        assert 0.0 <= t_code <= 1.0


# ======================================================================
# 6. Персистентность (A0.6: после load уровни не теряются)
# ======================================================================
class TestPersistence:

    def test_state_roundtrip(self):
        r0 = _runner(levels={"rpm": RPM_LEVELS})
        r1 = cst.runner_from_state(cst.runner_to_state(r0))
        assert r1.process_levels == {"rpm": [400.0, 900.0]}

    def test_file_save_load(self, tmp_path):
        r0 = _runner(levels={"rpm": RPM_LEVELS})
        cst.save_campaign(r0, str(tmp_path), "lv")
        r1 = cst.load_campaign(str(tmp_path), "lv")
        assert r1.process_levels == {"rpm": [400.0, 900.0]}
        # и слой РАБОТАЕТ после загрузки, а не просто лежит в поле
        X = r1._phase_candidates(32, seed=7)
        assert set(np.unique(np.round(X[:, r1.q + 1], 12)).tolist()) <= {0.0, 1.0}

    def test_old_state_without_key_loads_as_continuous(self):
        state = cst.runner_to_state(_runner())
        state["runner"].pop("process_levels", None)      # сейв до P2.1
        r = cst.runner_from_state(state)
        assert r.process_levels == {}
