# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 94 — состав проекта РАСТЁТ, база НЕ теряется (§16.2/§16.6 + И-1).

Закрываемый отказ (живая сессия 14.08.2026). Помощник получил дословно:

    «Проект в сессии уже СОБРАН: поля формы сетапа — черновик пересборки,
     точечная правка полей к нему не применяется»

— и правка встала. Разбор показал ДВЕ причины, обе в ядре, а не в ассистенте:

  1. **Состав вселенной был приговором конструктора.** ``full_schema``
     присваивалась один раз, а ``add_process_var`` / ``add_mixture_component``
     умели лишь РАСКРЫВАТЬ объявленное (``KeyError`` на новое имя). Компонент, о
     котором не подумали при рождении проекта, добавить было некуда — оставалась
     пересборка через форму, то есть НОВЫЙ раннер с пустой базой: смена состава
     стоила всех измеренных опытов (нарушение И-1).
  2. **Пустая база считалась сбоем.** ``augment_phase_*`` / ``move_region``
     звали ``fit_surrogates``, а тот на пустой базе бросает ``RuntimeError('Нет
     данных')``. В том живом проекте было ровно 0 точек — то есть даже после
     снятия запрета правка упала бы в ядре.

Здесь проверяется устройство после правки:

  * ``declare_variables`` дописывает имя в КОНЕЦ блока полной схемы — тем же
    правилом «старые координаты = ПРЕФИКС новых», которым живёт
    ``migrate_point``; поэтому миграция точек НЕ переписывалась;
  * ``baseline`` продлевается СОГЛАСОВАННО: mixture-значение вставляется ПЕРЕД
    process-частью (baseline — один вектор ``[mix, proc]``, дописывание в конец
    сдвинуло бы process-координаты и молча испортило бы измерение);
  * добавление компонента/оси к ИЗМЕРЕННОЙ базе её не урезает (И-1), версия
    схемы растёт, старые точки мигрируют на грань C=0 / baseline оси;
  * «удаление» компонента = ``deactivate_variable`` (зажим в [0,0]): точка
    выходит из активного pool, но ОСТАЁТСЯ в истории и восстановима;
  * оракул с ФИКСИРОВАННОЙ схемой (синтетическая истина) расширение отвергает —
    иначе он вернул бы отклики другой физики;
  * расширенная вселенная переживает save/load (формат файла не менялся).
"""
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.core.schema import ModelSpec, ProjectSchema, VariableBlock
from src.core.schema_evolution import known_constant, point_in_region
from src.apps.campaign import CampaignController
from src.apps.campaign_ui import build_setup_runner
from src.apps.mixture_process_runner import MixtureProcessRunner
from src.verification.mixture_process_truth import MultiMixtureProcessTruth

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ----------------------------------------------------------------------
# Кампания РУЧНОГО ввода — ровно та, что рождает кнопка «🏗 Построить проект»
# ----------------------------------------------------------------------
def _manual_runner(seed: int = 3) -> MixtureProcessRunner:
    return build_setup_runner(
        mixture_names=["A", "B"], process_names=["T"],
        process_lower=[150.0], process_upper=[200.0],
        response_names=["strength", "gloss"], seed=seed)


def _measure_manual(r: MixtureProcessRunner, n: int = 8) -> None:
    """Снять n точек ручным путём (Y вносит «человек»), как commit_seed."""
    X = r._phase_candidates(n, r.seed)
    Y = np.column_stack([np.linspace(1.0, 2.0, len(X)),
                         np.linspace(3.0, 4.0, len(X))])
    r.points = [r._make_point(X[i], Y[i], "seed", block=1)
                for i in range(len(X))]
    r.refit_if_possible()


# ======================================================================
# 1. declare_variables — вселенная расширяема, инварианты сохранены
# ======================================================================
class TestDeclareVariables:
    def test_new_component_appended_to_end_of_full_schema(self):
        r = _manual_runner()
        r.declare_variables(mixture=[("C", 0.0, 0.4)])
        # append именно в КОНЕЦ: правило миграции (префикс) не нарушено
        assert list(r.full_schema.mixture_names) == ["A", "B", "C"]
        assert r.q_full == 3
        assert r._full_mix.upper[-1] == pytest.approx(0.4)

    def test_new_axis_appended_with_real_units(self):
        r = _manual_runner()
        r.declare_variables(process=[("rotor_Hz", 40.0, 60.0)])
        assert list(r.full_schema.process_names) == ["T", "rotor_Hz"]
        assert r.d_full == 2
        assert (r._full_proc.lower[-1], r._full_proc.upper[-1]) == (40.0, 60.0)

    def test_baseline_stays_aligned_when_component_added(self):
        """Гвоздь: mixture-значение вставляется ПЕРЕД process-частью.

        baseline — единый вектор ``[mix(q_full), proc(d_full)]``. Если дописать
        новый компонент в КОНЕЦ вектора, process-координата съедет на позицию
        компонента, и ``_to_full`` начнёт мерить не тот режим.
        """
        r = _manual_runner()
        proc_before = list(np.asarray(r.baseline, float))[r.q_full:]
        r.declare_variables(mixture=[("C", 0.0, 1.0)])
        base = list(np.asarray(r.baseline, float))
        assert len(base) == r.q_full + r.d_full
        assert base[r.q_full - 1] == pytest.approx(0.0)   # новый компонент = грань
        assert base[r.q_full:] == pytest.approx(proc_before)

    def test_baseline_of_new_axis_is_mid_code(self):
        r = _manual_runner()
        r.declare_variables(process=[("rotor_Hz", 40.0, 60.0)])
        assert float(np.asarray(r.baseline, float)[-1]) == pytest.approx(0.5)

    def test_declaration_is_all_or_nothing(self):
        """Отказ по одной оси не оставляет вселенную полурасширенной."""
        r = _manual_runner()
        with pytest.raises(ValueError, match="выродились"):
            r.declare_variables(mixture=[("C", 0.0, 1.0)],
                                process=[("bad", 10.0, 10.0)])
        assert list(r.full_schema.mixture_names) == ["A", "B"]
        assert list(r.full_schema.process_names) == ["T"]
        assert np.asarray(r.baseline, float).size == 3

    def test_name_clashes_refused(self):
        r = _manual_runner()
        for decl in ({"mixture": [("A", 0.0, 1.0)]},
                     {"process": [("T", 1.0, 2.0)]},
                     {"mixture": [("strength", 0.0, 1.0)]}):
            with pytest.raises(ValueError, match="заняты"):
                r.declare_variables(**decl)

    def test_duplicate_inside_declaration_refused(self):
        r = _manual_runner()
        with pytest.raises(ValueError, match="Дубли"):
            r.declare_variables(mixture=[("C", 0.0, 1.0), ("C", 0.0, 1.0)])

    def test_empty_declaration_refused(self):
        with pytest.raises(ValueError, match="не задано"):
            _manual_runner().declare_variables()

    def test_declaration_alone_does_not_change_the_game(self):
        """Объявление ≠ ввод в игру: версия и текущая схема не двигаются."""
        r = _manual_runner()
        v0 = r.current_schema_version
        cur0 = list(r.current_schema.mixture_names)
        r.declare_variables(mixture=[("C", 0.0, 1.0)])
        assert r.current_schema_version == v0
        assert list(r.current_schema.mixture_names) == cur0

    def test_synthetic_oracle_refuses_extension(self):
        """Истина с фиксированной схемой не умеет мерить новую переменную."""
        mix = VariableBlock.mixture(["A", "B", "C"])
        proc = VariableBlock.process(["T"], lower=[0.0], upper=[1.0])
        schema = ProjectSchema.mixture_process(
            mix, proc, model=ModelSpec(mixture_order="quadratic"))
        from src.design.block_model import build_model_terms
        p = build_model_terms(schema).p
        oracle = MultiMixtureProcessTruth(schema, {"p0": [0.0] * p},
                                          noise_sd=0.0)
        r = MixtureProcessRunner(schema, oracle, seed=1, n_restarts=2)
        with pytest.raises(ValueError, match="схеме и новые"):
            r.declare_variables(process=[("P", 0.0, 1.0)])


# ======================================================================
# 2. И-1: правка состава ЖИВОГО проекта не стоит измеренных опытов
# ======================================================================
class TestMeasuredBaseSurvives:
    def test_new_component_keeps_all_measured_points(self):
        r = _manual_runner()
        _measure_manual(r, n=8)
        ctrl = CampaignController(r)
        v0, n0 = r.current_schema_version, len(r.points)

        ctrl.add_mixture_component("C", lower=0.0, upper=0.3)

        assert "C" in r.current_schema.mixture_names
        assert r.current_schema_version == v0 + 1        # append → bump
        assert len(r.points) == n0                       # И-1: база цела
        assert len(r._migrated_points()) == n0           # мигрировали на грань C=0
        assert all(point_in_region(p, r.current_schema)
                   for p in r._migrated_points())
        # доля нового компонента у старых точек — ровно 0 (Σ сходится)
        assert all(abs(float(p.X["MIXTURE"][-1])) < 1e-12
                   for p in r._migrated_points())

    def test_new_axis_keeps_all_measured_points(self):
        r = _manual_runner()
        _measure_manual(r, n=8)
        ctrl = CampaignController(r)
        v0, n0 = r.current_schema_version, len(r.points)

        ctrl.add_process_var("rotor_Hz", known_constant(50.0),
                             lower=40.0, upper=60.0)

        assert "rotor_Hz" in r.current_schema.process_names
        assert r.current_schema_version == v0 + 1
        assert len(r.points) == n0
        assert len(r._migrated_points()) == n0
        # 50 Hz в интервале 40…60 → код 0.5 у всех исторических точек
        assert all(abs(float(p.X["PROCESS"][-1]) - 0.5) < 1e-9
                   for p in r._migrated_points())

    def test_new_axis_without_bounds_refused_by_name(self):
        """Границы НОВОЙ оси не выдумываются: нет lower/upper — явный отказ.

        Тип отказа — ``KeyError`` (прежний контракт «имени в проекте нет»),
        чтобы вызывающая сторона не различала «нет оси» и «нет границ» по типу.
        """
        r = _manual_runner()
        ctrl = CampaignController(r)
        with pytest.raises(KeyError, match="границы"):
            ctrl.add_process_var("rotor_Hz", known_constant(50.0))

    def test_declared_axis_still_needs_explicit_migration(self):
        """A0.6 не ослаблен: политика миграции обязательна и для новой оси."""
        r = _manual_runner()
        ctrl = CampaignController(r)
        with pytest.raises(ValueError, match="A0.6"):
            ctrl.add_process_var("rotor_Hz", {"foo": 1},
                                 lower=40.0, upper=60.0)


# ======================================================================
# 3. Пустая база — законное состояние, а не сбой (вторая мина отказа)
# ======================================================================
class TestEmptyBaseIsNotAFailure:
    def test_component_added_to_unmeasured_project(self):
        """Тот самый живой кейс: проект собран, точек 0, состав правится."""
        r = _manual_runner()
        assert r.points == []
        ctrl = CampaignController(r)
        ctrl.add_mixture_component("C", lower=0.0, upper=0.3)
        assert "C" in r.current_schema.mixture_names
        assert r.surrogates == {}          # моделей нет — и это нормально

    def test_axis_added_to_unmeasured_project(self):
        r = _manual_runner()
        ctrl = CampaignController(r)
        ctrl.add_process_var("rotor_Hz", known_constant(50.0),
                             lower=40.0, upper=60.0)
        assert "rotor_Hz" in r.current_schema.process_names

    def test_bounds_move_on_unmeasured_project(self):
        r = _manual_runner()
        CampaignController(r).relax_bounds("T", 140.0, 210.0)
        pb = r.current_schema.process_block()
        assert (pb.lower[0], pb.upper[0]) == (140.0, 210.0)

    def test_refit_reports_no_data_without_raising(self):
        r = _manual_runner()
        assert r.refit_if_possible() is False
        _measure_manual(r, n=8)
        assert r.refit_if_possible() is True
        assert set(r.surrogates) == {"strength", "gloss"}


# ======================================================================
# 4. «Удаление» = ДЕАКТИВАЦИЯ: из поиска выходит, из истории — нет
# ======================================================================
class TestDeactivation:
    def test_deactivated_component_is_pinned_to_zero(self):
        r = _manual_runner()
        _measure_manual(r, n=8)
        ctrl = CampaignController(r)
        ctrl.add_mixture_component("C", lower=0.0, upper=0.4)
        n_all = len(r.points)

        ctrl.deactivate_variable("C")

        mb = r.current_schema.mixture_block()
        j = list(mb.names).index("C")
        assert (mb.lower[j], mb.upper[j]) == (0.0, 0.0)
        # версия НЕ растёт: это region-move, а не эволюция схемы
        assert len(r.points) == n_all                    # И-1: история цела
        # кандидаты плана больше не варьируют C
        cand = r._phase_candidates(24, r.seed)
        assert np.allclose(cand[:, j], 0.0, atol=1e-9)

    def test_deactivation_does_not_bump_version(self):
        r = _manual_runner()
        _measure_manual(r, n=8)
        v0 = r.current_schema_version
        CampaignController(r).deactivate_variable("A", value=0.0)
        assert r.current_schema_version == v0

    def test_deactivated_axis_is_pinned_and_reversible(self):
        r = _manual_runner()
        _measure_manual(r, n=8)
        ctrl = CampaignController(r)
        n_all = len(r.points)

        ctrl.deactivate_variable("T", value=150.0)
        pb = r.current_schema.process_block()
        assert (pb.lower[0], pb.upper[0]) == (150.0, 150.0)
        assert len(r.points) == n_all                    # история цела

        # обратимость (§15.0.3.3): расширяем назад — точки возвращаются в pool
        ctrl.relax_bounds("T", 150.0, 200.0)
        assert len(r._migrated_points()) == n_all


# ======================================================================
# 5. Расширенная вселенная переживает save/load (формат файла НЕ менялся)
# ======================================================================
def test_extended_universe_survives_save_load(tmp_path):
    from src.apps import campaign_state as cst

    r = _manual_runner()
    _measure_manual(r, n=8)
    ctrl = CampaignController(r)
    ctrl.add_mixture_component("C", lower=0.0, upper=0.3)
    ctrl.add_process_var("rotor_Hz", known_constant(50.0),
                         lower=40.0, upper=60.0)
    n_all = len(r.points)

    cst.save_campaign(r, tmp_path, "iter94")
    back = cst.load_campaign(tmp_path, "iter94")

    assert list(back.full_schema.mixture_names) == ["A", "B", "C"]
    assert list(back.full_schema.process_names) == ["T", "rotor_Hz"]
    assert list(back.current_schema.mixture_names) == ["A", "B", "C"]
    assert back.q_full == 3 and back.d_full == 2
    assert np.asarray(back.baseline, float).size == 5
    assert len(back.points) == n_all                      # И-1 через диск
    assert len(back._migrated_points()) == n_all


# ======================================================================
# 6. Ассистент: слепой гейт заменён МАРШРУТИЗАЦИЕЙ по стадии проекта
# ======================================================================
class TestAssistantRouting:
    """Тот самый отказ помощника: «Проект в сессии уже СОБРАН…».

    Гейт смотрел ТОЛЬКО на «есть ли движок» — и правка вставала даже у проекта
    с пустой базой. Теперь ответ различает стадию и путь применения поля.
    """

    @staticmethod
    def _ctx(runner=None, fields=None):
        from src.assistant.session import new_session
        from src.assistant.tools.registry import ToolContext
        return ToolContext(project="iter94", runner=runner,
                           session=new_session("iter94"),
                           extra={"setup_fields": dict(fields or {})})

    def test_stage_draft_when_no_runner(self):
        from src.assistant.tools.readonly import project_stage
        assert project_stage(self._ctx())["stage"] == "draft"

    def test_stage_empty_for_built_but_unmeasured(self):
        from src.assistant.tools.readonly import project_stage
        info = project_stage(self._ctx(runner=_manual_runner()))
        assert info["stage"] == "empty"
        assert info["n_points"] == 0 and info["n_branches"] == 0

    def test_stage_live_when_points_measured(self):
        from src.assistant.tools.readonly import project_stage
        r = _manual_runner()
        _measure_manual(r, n=8)
        info = project_stage(self._ctx(runner=r))
        assert info["stage"] == "live" and info["n_points"] == 8

    def test_routes_split_live_and_rebuild(self):
        from src.assistant.tools.readonly import setup_field_route
        assert setup_field_route("setup_process_levels", "live")["route"] == "live"
        assert setup_field_route("setup_mix", "live")["route"] == "live"
        assert setup_field_route("setup_seed", "live")["route"] == "rebuild"
        # до сборки маршрут один: черновик формы
        assert setup_field_route("setup_seed", "draft")["route"] == "form"

    def test_live_routes_name_existing_engine_operations(self):
        """Карта маршрутов не выдумана: перечисленные сеттеры существуют.

        Иначе подсказка «правится runner.set_x» уводила бы модель к
        несуществующему методу — правдоподобно и неверно.
        """
        from src.assistant.tools.readonly import SETUP_LIVE_ROUTES
        r = _manual_runner()
        ctrl = CampaignController(r)
        for how in SETUP_LIVE_ROUTES.values():
            head = how.split()[0]
            owner, attr = head.split(".", 1)
            target = r if owner == "runner" else ctrl
            assert hasattr(target, attr), f"нет операции {head}"

    def test_propose_stages_edit_for_built_project(self):
        from src.assistant.tools.registry import PROPOSE, dispatch
        ctx = self._ctx(runner=_manual_runner(), fields={"setup_seed": 1})
        out = dispatch(ctx, "propose_setup_fields",
                       {"fields": {"setup_seed": 7}, "rationale": "воспроизводимость"},
                       allowed_kinds=[PROPOSE])
        assert out["staged"] is True and out["stage"] == "empty"
        assert ctx.session.staged_setups()[0].id == out["setup_id"]

    def test_propose_warns_about_price_on_live_project(self):
        from src.assistant.tools.registry import PROPOSE, dispatch
        r = _manual_runner()
        _measure_manual(r, n=8)
        ctx = self._ctx(runner=r, fields={"setup_mix": "A, B"})
        out = dispatch(ctx, "propose_setup_fields",
                       {"fields": {"setup_mix": "A, B, C", "setup_seed": 7},
                        "rationale": "новый компонент"},
                       allowed_kinds=[PROPOSE])
        assert out["staged"] is True and out["stage"] == "live"
        assert "8 измеренных точек" in out["warning"]
        # поле с живым путём и поле без него различены
        assert out["routes"]["setup_mix"]["route"] == "live"
        assert out["routes"]["setup_seed"]["route"] == "rebuild"

    def test_get_setup_fields_reports_stage_and_routes(self):
        from src.assistant.tools.registry import READONLY, dispatch
        r = _manual_runner()
        _measure_manual(r, n=8)
        out = dispatch(self._ctx(runner=r,
                                 fields={"setup_process_levels": "T: 150, 200",
                                         "setup_seed": 1}),
                       "get_setup_fields", {}, allowed_kinds=[READONLY])
        assert out["stage"] == "live" and out["n_points"] == 8
        assert out["routes"]["setup_process_levels"]["route"] == "live"
        assert out["routes"]["setup_seed"]["route"] == "rebuild"

    def test_dock_panel_no_longer_disables_apply(self):
        """Кнопка «✅ Применить правку» больше не выключена при проекте.

        Прежде ``disabled=runner is not None`` — правка застревала в стейдже
        навсегда, в том числе у проекта с пустой базой. Проверяем сам источник
        панели: проще и надёжнее, чем гонять Streamlit ради одного флага.
        """
        import inspect
        from src.apps import assistant_dock
        src = inspect.getsource(assistant_dock._render_setup_edits)
        assert "disabled=runner is not None" not in src
        assert "Проект ЖИВОЙ" in src          # цена названа вслух

    def test_prompt_teaches_stage_routing_and_deactivation(self):
        """Правило выбора живёт в промпте: иначе модель снова полезет пакетом."""
        from src.assistant.prompts import architect_system_prompt
        for has_runner in (False, True):
            text = architect_system_prompt(project="iter94",
                                           has_runner=has_runner)
            assert "СТАДИЯ ПРОЕКТА" in text
            assert "route='live'" in text and "route='rebuild'" in text
            assert "ДЕАКТИВАЦИЯ" in text


# ======================================================================
# 7. UI живого проекта: новое имя вводится ЗДЕСЬ, а не пересборкой
# ======================================================================
def test_evolution_panel_accepts_new_names_and_deactivation():
    """Панель «🧬 Эволюция схемы» перестала быть только «раскрытием».

    До iter94 она рисовала ``selectbox`` по ``hidden_mix``/``hidden_proc``, а у
    проекта из формы эти списки ВСЕГДА пусты (``begin_phase`` из UI не
    вызывается) — то есть панель показывала «всё уже в схеме» и добавить
    компонент через неё было нельзя вообще.
    """
    import inspect
    from src.apps import campaign_ui as cui
    src = inspect.getsource(cui.render_schema_evolution)
    assert 'st.selectbox("переменная"' not in src        # был селектор
    assert 'st.selectbox("компонент"' not in src
    assert 'key="camp_ev_proc"' in src and "text_input" in src
    assert "camp_ev_off_btn" in src                      # деактивация есть
    assert "deactivate_variable" in src
