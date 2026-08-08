# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 57 / UI_REVISION_SPEC P3.3 — СВЯЗАННЫЕ process-оси.

Кампания ПВХ требует производной величины ``dT_head = T_адаптер − T_пласт``
с полосой реализуемости по железу: нагреватель адаптера держит перепад лишь
в паспортном диапазоне. Пока process-оси считались независимым боксом, план
мог предложить нереализуемый перепад — оператор ставил «что получится», а
модель училась на координатах из таблицы (тихая подмена против A0.6).

Слой связок — ПРОЕКЦИЯ на полосу ``lo ≤ A − B ≤ hi`` (канон слоя уровней
iter51): полоса в ФИЗИЧЕСКИХ единицах; проекция — минимальный L2-сдвиг пары
на грань полосы с учётом боксов; идемпотентна; применяется ПОСЛЕ розыгрыша
(low-discrepancy остальных осей не трогается) и в argmax
(``process_project`` в ``optimize_desirability``).

Покрытие: чистый модуль ``design.linked_axes``; политика раннера
(``set_process_links`` / ``snap_linked_axes`` / ``linked_axes_report`` /
пул кандидатов / argmax / конфликт с уровнями); персистентность в
``campaign_state`` (включая старые сейвы без ключа); чистые UI-хелперы
(парсер/round-trip/подпись/паспорт).
"""
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.apps import campaign_state as cst
from src.apps.campaign_ui import (build_setup_runner,
                                  campaign_passport_dataframe,
                                  parse_process_links,
                                  process_links_to_text,
                                  seed_links_caption,
                                  setup_prefill_from_runner)
from src.design.linked_axes import (ProcessLink, links_caption,
                                    normalize_links, snap_pair_to_band)
from src.optimize.desirability import DesirabilitySpec

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Golden кампании ПВХ: T_plast (150…200) и T_adapter (150…250); перепад
# dT_head = T_adapter − T_plast реализуем железом только в [10, 60].
AXES = ["T_plast", "T_adapter", "rpm"]
LOWER = [150.0, 150.0, 400.0]
UPPER = [200.0, 250.0, 900.0]
LINK = {"name": "dT_head", "minuend": "T_adapter", "subtrahend": "T_plast",
        "lo": 10.0, "hi": 60.0}


def _norm(links, *, axes=AXES, lower=LOWER, upper=UPPER):
    return normalize_links(links, names=axes, lower=lower, upper=upper)


def _runner(*, links=None, seed: int = 3):
    """Раннер: смесь {A,B,C} × process {T_plast, T_adapter, rpm}."""
    r = build_setup_runner(
        mixture_names=["A", "B", "C"],
        process_names=AXES, process_lower=LOWER, process_upper=UPPER,
        response_names=["y"], seed=seed)
    if links is not None:
        r.set_process_links(links)
    return r


def _dt_phys(r, X):
    """Разность T_adapter − T_plast (физические единицы) по составному X."""
    X = np.atleast_2d(np.asarray(X, float))
    tp = LOWER[0] + X[:, r.q + 0] * (UPPER[0] - LOWER[0])
    ta = LOWER[1] + X[:, r.q + 1] * (UPPER[1] - LOWER[1])
    return ta - tp


# ======================================================================
# 1. Чистый модуль: нормализация связок (A0.6 — явные ошибки)
# ======================================================================
class TestNormalizeLinks:

    def test_valid_link(self):
        out = _norm([LINK])
        assert len(out) == 1
        lk = out[0]
        assert isinstance(lk, ProcessLink)
        assert (lk.name, lk.minuend, lk.subtrahend) == (
            "dT_head", "T_adapter", "T_plast")
        assert (lk.lo, lk.hi) == (10.0, 60.0)

    def test_one_sided_bounds(self):
        lk = _norm([dict(LINK, lo=None)])[0]
        assert lk.lo == -np.inf and lk.hi == 60.0
        lk = _norm([dict(LINK, hi=None)])[0]
        assert lk.lo == 10.0 and lk.hi == np.inf

    def test_both_bounds_open_rejected(self):
        with pytest.raises(ValueError, match="обе границы"):
            _norm([dict(LINK, lo=None, hi=None)])

    def test_unknown_axis_rejected(self):
        with pytest.raises(KeyError):
            _norm([dict(LINK, minuend="нет_такой")])

    def test_same_axis_both_sides_rejected(self):
        with pytest.raises(ValueError, match="совпадают"):
            _norm([dict(LINK, subtrahend="T_adapter")])

    def test_name_collision_with_axis_rejected(self):
        with pytest.raises(ValueError, match="совпадает с именем"):
            _norm([dict(LINK, name="rpm")])

    def test_duplicate_link_name_rejected(self):
        with pytest.raises(ValueError, match="дважды"):
            _norm([LINK, dict(LINK, minuend="T_plast",
                              subtrahend="T_adapter", lo=-60, hi=-10)])

    def test_axis_in_two_links_rejected(self):
        """Ось не более чем в ОДНОЙ связке: совместная проекция двух полос
        с общей осью не гарантирована — честный отказ (A0.6)."""
        other = {"name": "dT2", "minuend": "T_adapter", "subtrahend": "rpm",
                 "lo": -1000.0, "hi": 1000.0}
        # сделаем полосу валидной пересечением: achievable = [-750, 100]
        other["lo"], other["hi"] = -700.0, 0.0
        with pytest.raises(ValueError, match="уже участвует"):
            _norm([LINK, other])

    def test_lo_ge_hi_rejected(self):
        with pytest.raises(ValueError, match="lo < hi"):
            _norm([dict(LINK, lo=60.0, hi=10.0)])

    def test_nan_bound_rejected(self):
        with pytest.raises(ValueError, match="NaN"):
            _norm([dict(LINK, lo=float("nan"))])

    def test_empty_band_vs_achievable_rejected(self):
        # достижимый диапазон T_adapter − T_plast = [−50, 100]; полоса
        # [150, 200] с ним не пересекается — область пуста.
        with pytest.raises(ValueError, match="не пересекает"):
            _norm([dict(LINK, lo=150.0, hi=200.0)])


# ======================================================================
# 2. Чистый модуль: проекция пары на полосу
# ======================================================================
class TestSnapPairToBand:
    A_B = ((150.0, 200.0), (150.0, 250.0))     # (b_bounds ↔ T_plast?) см. ниже

    def test_inside_band_untouched(self):
        a = np.array([180.0])                  # T_adapter
        b = np.array([160.0])                  # T_plast: d = 20 ∈ [10, 60]
        a2, b2 = snap_pair_to_band(a, b, 10.0, 60.0,
                                   (150.0, 250.0), (150.0, 200.0))
        assert float(a2[0]) == 180.0 and float(b2[0]) == 160.0

    def test_upper_violation_projected_symmetrically(self):
        # d = 240 − 160 = 80 > 60 → сдвиг ∓10 обеим осям: (230, 170), d = 60
        a2, b2 = snap_pair_to_band(np.array([240.0]), np.array([160.0]),
                                   10.0, 60.0, (150.0, 250.0), (150.0, 200.0))
        assert np.isclose(float(a2[0]), 230.0)
        assert np.isclose(float(b2[0]), 170.0)
        assert np.isclose(float(a2[0] - b2[0]), 60.0)

    def test_lower_violation_projected(self):
        # d = 155 − 195 = −40 < 10 → на грань d = 10 c учётом бокса
        a2, b2 = snap_pair_to_band(np.array([155.0]), np.array([195.0]),
                                   10.0, 60.0, (150.0, 250.0), (150.0, 200.0))
        assert np.isclose(float(a2[0] - b2[0]), 10.0)
        assert 150.0 <= float(a2[0]) <= 250.0
        assert 150.0 <= float(b2[0]) <= 200.0

    def test_projection_respects_boxes_at_corner(self):
        # b у нижней границы: симметричный сдвиг вывел бы b ниже 150 —
        # решение прижимается к отрезку грани внутри бокса
        a2, b2 = snap_pair_to_band(np.array([249.0]), np.array([151.0]),
                                   10.0, 60.0, (150.0, 250.0), (150.0, 200.0))
        assert np.isclose(float(a2[0] - b2[0]), 60.0)
        assert float(b2[0]) >= 150.0 - 1e-12

    def test_idempotent(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(150.0, 250.0, 64)
        b = rng.uniform(150.0, 200.0, 64)
        a1, b1 = snap_pair_to_band(a, b, 10.0, 60.0,
                                   (150.0, 250.0), (150.0, 200.0))
        a2, b2 = snap_pair_to_band(a1, b1, 10.0, 60.0,
                                   (150.0, 250.0), (150.0, 200.0))
        assert np.allclose(a1, a2) and np.allclose(b1, b2)
        assert np.all(a1 - b1 >= 10.0 - 1e-9)
        assert np.all(a1 - b1 <= 60.0 + 1e-9)

    def test_inputs_not_mutated(self):
        a = np.array([240.0])
        b = np.array([160.0])
        snap_pair_to_band(a, b, 10.0, 60.0, (150.0, 250.0), (150.0, 200.0))
        assert float(a[0]) == 240.0 and float(b[0]) == 160.0


def test_links_caption_states_absence_explicitly():
    """Пусто ≠ «поле не заполнено»: подпись говорит про независимость прямо."""
    assert "независимы" in links_caption([])
    cap = links_caption(_norm([LINK]))
    assert "dT_head" in cap and "T_adapter" in cap and "10" in cap


# ======================================================================
# 3. Политика раннера: валидация, снап координат, отчёт реализуемости
# ======================================================================
class TestRunnerPolicy:

    def test_set_and_clear(self):
        r = _runner()
        assert r.process_links == []
        r.set_process_links([LINK])
        assert len(r.process_links) == 1
        assert r.process_links[0].name == "dT_head"
        r.set_process_links(None)
        assert r.process_links == []

    def test_unknown_axis_rejected(self):
        with pytest.raises(KeyError):
            _runner().set_process_links([dict(LINK, minuend="нет")])

    def test_conflict_links_then_levels(self):
        """Ось связки не может нести дискретные уровни (обе стороны)."""
        r = _runner(links=[LINK])
        with pytest.raises(ValueError, match="связк"):
            r.set_process_levels({"T_plast": [150.0, 200.0]})

    def test_conflict_levels_then_links(self):
        r = _runner()
        r.set_process_levels({"T_adapter": [150.0, 250.0]})
        with pytest.raises(ValueError, match="уровн"):
            r.set_process_links([LINK])

    def test_levels_on_free_axis_coexist(self):
        """rpm вне связки — уровни на нём легальны рядом со связкой."""
        r = _runner(links=[LINK])
        r.set_process_levels({"rpm": [400.0, 900.0]})
        X = r._phase_candidates(32, seed=7)
        d = _dt_phys(r, X)
        assert np.all(d >= 10.0 - 1e-9) and np.all(d <= 60.0 + 1e-9)
        rpm_code = X[:, r.q + 2]
        assert set(np.unique(np.round(rpm_code, 12)).tolist()) <= {0.0, 1.0}

    def test_snap_linked_axes_idempotent_and_keeps_mixture(self):
        r = _runner(links=[LINK])
        rng = np.random.default_rng(5)
        X = np.hstack([np.full((16, 3), 1.0 / 3.0),
                       rng.uniform(0.0, 1.0, size=(16, 3))])
        once = r.snap_linked_axes(X)
        twice = r.snap_linked_axes(once)
        assert np.array_equal(once, twice)
        assert np.allclose(once[:, :r.q], X[:, :r.q])       # смесь не тронута
        d = _dt_phys(r, once)
        assert np.all(d >= 10.0 - 1e-9) and np.all(d <= 60.0 + 1e-9)

    def test_link_with_axis_outside_phase_drops(self):
        """Ось связки вне текущей фазы → связка выпадает (как уровни)."""
        r = _runner()
        r.begin_phase(["A", "B", "C"], ["T_plast"])          # adapter вне фазы
        r.set_process_links([LINK])
        assert r._links_current() == []
        X = r._phase_candidates(8, seed=5)
        assert X.shape[1] == r.q + 1

    def test_linked_axes_report(self):
        r = _runner(links=[LINK])
        # точка 1: d = 20 (ок); точка 2: d = 80 (вне полосы, до проекции)
        X = np.array([
            [1/3, 1/3, 1/3, (160-150)/50, (180-150)/100, 0.5],
            [1/3, 1/3, 1/3, (160-150)/50, (240-150)/100, 0.5]])
        rep = r.linked_axes_report(X)
        assert len(rep) == 1
        e = rep[0]
        assert e["name"] == "dT_head"
        assert np.allclose(e["values"], [20.0, 80.0])
        assert e["ok"] == [True, False]
        assert e["n_off"] == 1


# ======================================================================
# 4. Пул кандидатов: связка выполняется, чужие оси не тронуты
# ======================================================================
class TestCandidatePool:

    def test_all_candidates_feasible(self):
        r = _runner(links=[LINK])
        X = r._phase_candidates(128, seed=7)
        d = _dt_phys(r, X)
        assert np.all(d >= 10.0 - 1e-9) and np.all(d <= 60.0 + 1e-9)

    def test_other_columns_bit_identical_to_no_links_run(self):
        """Проекция пары НЕ меняет розыгрыш остальных координат (rpm и
        смесь бит-в-бит) — снап делается ПОСЛЕ розыгрыша Соболя."""
        base = _runner()._phase_candidates(32, seed=9)
        snapped = _runner(links=[LINK])._phase_candidates(32, seed=9)
        q = 3
        assert np.array_equal(base[:, :q], snapped[:, :q])          # смесь
        assert np.array_equal(base[:, q + 2], snapped[:, q + 2])    # rpm
        # а связанные оси — изменились (в исходном пуле есть нарушители)
        assert not np.array_equal(base[:, q:q + 2], snapped[:, q:q + 2])

    def test_no_links_means_bit_identical_behaviour(self):
        a = _runner()._phase_candidates(24, seed=4)
        b = _runner(links=[])._phase_candidates(24, seed=4)
        assert np.array_equal(a, b)

    def test_propose_seed_feasible(self):
        r = _runner(links=[LINK])
        Xs = r.propose_seed(10)
        d = _dt_phys(r, Xs)
        assert np.all(d >= 10.0 - 1e-9) and np.all(d <= 60.0 + 1e-9)


# ======================================================================
# 5. argmax: оптимум выдаётся только с реализуемой разностью
# ======================================================================
def _seeded_runner_with_branch():
    """Истина тянет перепад к 0 — соблазн нарушить полосу [10, 60]
    максимален; при работающем слое argmax обязан остаться в полосе."""
    r = _runner(links=[LINK], seed=5)
    X = r.propose_seed(14)
    d = _dt_phys(r, X)
    y = 2.0 * X[:, 0] - 0.01 * d ** 2
    r.commit_seed(X, y.reshape(-1, 1))
    r.add_branch("b", {"y": DesirabilitySpec("max", low=float(y.min()),
                                             high=float(y.max()) + 1.0)},
                 branch_id="b1")
    return r


class TestArgmaxOnBand:

    def test_xbest_within_band(self):
        r = _seeded_runner_with_branch()
        res = r.optimize_xbest("b1", n_candidates=200, refine_iters=60,
                               n_starts=2)
        d = float(_dt_phys(r, res.x.reshape(1, -1))[0])
        assert 10.0 - 1e-6 <= d <= 60.0 + 1e-6


# ======================================================================
# 6. Персистентность (A0.6: после load связки не теряются)
# ======================================================================
class TestPersistence:

    def test_state_roundtrip(self):
        r0 = _runner(links=[LINK])
        r1 = cst.runner_from_state(cst.runner_to_state(r0))
        assert len(r1.process_links) == 1
        lk = r1.process_links[0]
        assert (lk.name, lk.minuend, lk.subtrahend, lk.lo, lk.hi) == (
            "dT_head", "T_adapter", "T_plast", 10.0, 60.0)

    def test_one_sided_inf_json_safe(self):
        """±inf в полосе пишется null (валидный JSON) и восстанавливается."""
        import json
        r0 = _runner(links=[dict(LINK, hi=None)])
        state = cst.runner_to_state(r0)
        json.dumps(state)                       # не Infinity-литерал
        assert state["runner"]["process_links"][0]["hi"] is None
        r1 = cst.runner_from_state(state)
        assert r1.process_links[0].hi == np.inf

    def test_file_save_load_and_layer_works(self, tmp_path):
        r0 = _runner(links=[LINK])
        cst.save_campaign(r0, str(tmp_path), "lk")
        r1 = cst.load_campaign(str(tmp_path), "lk")
        X = r1._phase_candidates(32, seed=7)
        d = _dt_phys(r1, X)
        assert np.all(d >= 10.0 - 1e-9) and np.all(d <= 60.0 + 1e-9)

    def test_old_state_without_key_loads_as_independent(self):
        state = cst.runner_to_state(_runner())
        state["runner"].pop("process_links", None)      # сейв до P3.3
        r = cst.runner_from_state(state)
        assert r.process_links == []


# ======================================================================
# 7. UI-хелперы (чистые): парсер, round-trip, подпись, паспорт, префилл
# ======================================================================
class TestUiHelpers:

    def test_parse_basic(self):
        out = parse_process_links("dT_head: T_adapter - T_plast : 10, 60")
        assert out == [{"name": "dT_head", "minuend": "T_adapter",
                        "subtrahend": "T_plast", "lo": 10.0, "hi": 60.0}]

    def test_parse_open_side_star(self):
        out = parse_process_links("dT: T_adapter - T_plast : *, 60")
        assert out[0]["lo"] is None and out[0]["hi"] == 60.0

    def test_parse_blank_lines_ignored(self):
        assert parse_process_links("\n\n") == []

    def test_parse_errors_carry_line_number(self):
        with pytest.raises(ValueError, match="Строка 1"):
            parse_process_links("dT_head T_adapter - T_plast 10, 60")
        with pytest.raises(ValueError, match="ровно один"):
            parse_process_links("dT: a - b - c : 1, 2")
        with pytest.raises(ValueError, match="два значения"):
            parse_process_links("dT: a - b : 1")
        with pytest.raises(ValueError, match="не число"):
            parse_process_links("dT: a - b : x, 2")

    def test_round_trip_text(self):
        txt = ("dT_head: T_adapter - T_plast : 10, 60\n"
               "dP: p2 - p1 : *, 5")
        assert process_links_to_text(parse_process_links(txt)) == txt

    def test_to_text_accepts_process_link_objects(self):
        lk = ProcessLink("dT", "a", "b", lo=-np.inf, hi=60.0)
        assert process_links_to_text([lk]) == "dT: a - b : *, 60"

    def test_seed_links_caption_empty_without_links(self):
        r = _runner()
        assert seed_links_caption(r, r._phase_candidates(4, seed=1)) == ""

    def test_seed_links_caption_ok_plan(self):
        r = _runner(links=[LINK])
        cap = seed_links_caption(r, r._phase_candidates(8, seed=1))
        assert "реализуемы" in cap and "dT_head" in cap

    def test_seed_links_caption_warns_off_band(self):
        r = _runner(links=[LINK])
        # точка с d = 80 (связки «задали после построения плана»)
        X = np.array([[1/3, 1/3, 1/3, (160-150)/50, (240-150)/100, 0.5]])
        cap = seed_links_caption(r, X)
        assert "⚠️" in cap and "заново" in cap

    def test_passport_row(self):
        df = campaign_passport_dataframe(_runner(links=[LINK]))
        row = df[df["параметр"] == "связанные process-оси (разности)"]
        assert len(row) == 1
        assert "dT_head" in str(row["значение"].iloc[0])
        # без связок — явное «—»
        df0 = campaign_passport_dataframe(_runner())
        row0 = df0[df0["параметр"] == "связанные process-оси (разности)"]
        assert str(row0["значение"].iloc[0]) == "—"

    def test_setup_prefill_carries_links(self):
        out = setup_prefill_from_runner(_runner(links=[LINK]))
        assert out["setup_process_links"] == (
            "dT_head: T_adapter - T_plast : 10, 60")