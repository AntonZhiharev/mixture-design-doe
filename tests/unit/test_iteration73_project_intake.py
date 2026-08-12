# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 73 / ASSISTANT_SPEC — ПАКЕТ ПРОЕКТА: рождение проекта пакетом.

Закрываемый отказ (живая сессия 11.08.2026). Ассистент собрал геометрию верно,
положил пакет спеки в стейдж, человек нажал «Применить спеку» — и на закладке
«Старт» НИЧЕГО не изменилось. В журнал решений при этом легла запись
``apply_spec`` от «человек (UI)». Разбор показал три независимых дефекта:

  1. **применять было НЕКУДА.** ``apply_spec`` пишет геометрию через
     ``runner.set_phr_spec``, а раннера в сессии не было; результат уходил в
     ``ctx.spec``, который живёт один прогон Streamlit. Логическая ошибка:
     phr-спека — это mixture-блок СХЕМЫ, значит применение спеки в пустой
     сессии есть РОЖДЕНИЕ проекта, а инструмент написан как правка
     существующего;
  2. **одной спеки для рождения проекта не хватает.** ``build_setup_runner``
     требует ещё ОТКЛИКИ и ПРОЦЕСС-ОСИ с границами, а их в спеке нет по схеме
     (``spec_schema.not_in_spec``) — и выдумывать их за технолога нельзя;
  3. **статус решения не сохранялся.** Кнопки человека меняли статус только в
     памяти: в ``session.json`` пакет оставался ``staged``, и после
     перезапуска предлагался к применению снова.

Что закрывают тесты:

  * :mod:`design.project_package` разбирает пакет ЯДРОМ и объясняет, чего не
    хватает, называя блоки; пример схемы сам проходит разбор;
  * МАНИФЕСТ отвечает на требование «из JSON должно быть понятно, что именно
    загружается»: состав, отклики, процесс-оси с границами и единицами;
  * проекция в поля формы сетапа совпадает с ключами и ЯРЛЫКАМИ виджетов
    ``campaign_ui`` и принимается его же парсерами (round-trip по spec_hash);
  * ``project_schema`` / ``validate_project_package`` (readonly),
    ``propose_project`` (propose), ``apply_project`` / ``reject_project``
    (write, модели не выданы, разовый токен);
  * принятие пакета НЕ собирает раннер: путь сборки один — штатная кнопка;
  * пакет проекта не заводит проект заново поверх собранного;
  * :func:`context.persist_session` сохраняет статус на диск;
  * маршрутизация: «собери проект» уходит в ``project_intake``.
"""
import json

import pytest

from src.assistant import context as actx
from src.assistant import store
from src.assistant.consent import ACTIONS, ConsentRegistry
from src.assistant.session import (PATCH_APPLIED, PATCH_REJECTED, PATCH_STAGED,
                                   new_session)
from src.assistant.tools import (AGENT_KINDS, PROPOSE, READONLY, WRITE,
                                 ToolContext, ToolError, dispatch, tool_names)
from src.assistant.tools.write import (issue_apply_project_token,
                                       issue_reject_project_token)
from src.assistant.views import (project_apply_caption,
                                 project_blocks_dataframe,
                                 project_status_caption,
                                 staged_projects_dataframe)
from src.design import project_package as pp

PROJECT = "pvc_edge_v1"

#: Пакет проекта референсной кампании: те же узлы, что в iter71, плюс блоки,
#: которых в спеке нет по схеме — отклики и процесс-оси.
SPEC_BLOCK = {
    "spec_version": 2,
    "group_order": ["SOFT"],
    "nodes": [
        {"name": "RESIN", "role": "FIXED", "value": 100.0},
        {"name": "DINP", "role": "ABSOLUTE", "range": [4.0, 14.0]},
        {"name": "SOFT", "role": "GROUP_TOTAL", "range": [5.0, 15.0],
         "members": ["PBNK", "CPE"]},
        {"name": "PBNK", "role": "SHARE_FREE", "group": "SOFT",
         "share_range": [0.0, 0.70], "max_phr": 8.0},
        {"name": "CPE", "role": "SHARE_CLOSURE", "group": "SOFT",
         "min_phr": 3.0},
    ],
}

PACKAGE = {
    "package_kind": "project",
    "label": "кромка ПВХ: первичный ввод проекта",
    "seed": 3,
    "spec": SPEC_BLOCK,
    "responses": [{"name": "gloss", "unit": "%"},
                  {"name": "dE", "unit": "ΔE", "note": "против эталона"}],
    "process": [
        {"name": "T_plast", "range": [165.0, 185.0], "unit": "°C"},
        {"name": "rotor_rpm", "range": [400.0, 900.0], "unit": "об/мин",
         "levels": [400.0, 900.0]},
    ],
    "covariates": ["SME"],
    "passport": {"campaign_label": "PVC-кромка-2026",
                 "weighing_step_g": 0.1, "grams_per_phr": 10.0},
}


class _Runner:
    """Движок в объёме, который читают инструменты (has_project/статус)."""

    def __init__(self, spec=None, *, props=("gloss",), proc=("T",),
                 mix=("RESIN",), points=()):
        self.phr_spec = spec
        self.property_names = list(props)
        self.points = list(points)
        self.current_schema = type("_S", (), {
            "process_names": list(proc), "mixture_names": list(mix)})()


def _ctx(tmp_path=None, *, runner=None, consent=None, session=None):
    return ToolContext(
        spec=getattr(runner, "phr_spec", None),
        runner=runner,
        session=session if session is not None else new_session(PROJECT),
        root=str(tmp_path) if tmp_path is not None else "",
        project=PROJECT if tmp_path is not None else "",
        extra={"consent": consent} if consent is not None else {})


def _stage(ctx, package=PACKAGE, **kw):
    """Положить пакет проекта в стейдж и вернуть его id."""
    out = dispatch(ctx, "propose_project",
                   {"package": package,
                    "rationale": kw.pop("rationale", "первичный ввод проекта"),
                    **kw},
                   allowed_kinds=[PROPOSE])
    assert out["staged"] is True, out
    return out["project_id"]


# ----------------------------------------------------------------------
# 1. Ядро: разбор пакета и объяснение, чего не хватает
# ----------------------------------------------------------------------
class TestPackageParsing:
    def test_example_from_schema_parses(self):
        """Пример схемы САМ проходит разбор: иначе он хуже отсутствия примера."""
        pkg = pp.parse_project_package(pp.PROJECT_EXAMPLE)
        assert pkg.spec_hash
        assert pkg.response_names and pkg.process_names

    def test_spec_only_package_names_missing_blocks(self):
        """Главный урок отказа: одной СПЕКИ для проекта недостаточно."""
        with pytest.raises(pp.PackageError) as exc:
            pp.parse_project_package({"spec": SPEC_BLOCK})
        msg = str(exc.value)
        assert "responses" in msg and "process" in msg
        # Сказано не только «нет блоков», но и ПОЧЕМУ их нет в спеке.
        assert "по схеме" in msg

    def test_unknown_top_key_is_error_not_ignored(self):
        """Опечатка в имени блока не должна молча терять половину проекта."""
        with pytest.raises(pp.PackageError, match="неизвестные ключи"):
            pp.parse_project_package({**PACKAGE, "responce": []})

    def test_nodes_beside_spec_explained(self):
        """Частый промах: узлы рядом с 'spec', а не внутри него."""
        with pytest.raises(pp.PackageError, match="ВНУТРЬ блока 'spec'"):
            pp.parse_project_package({**PACKAGE, "nodes": []})

    def test_levels_as_count_refused(self):
        """«levels: 3» — политика ПЛАНА; сетка железа это СПИСОК значений."""
        bad = {**PACKAGE, "process": [
            {"name": "T", "range": [1.0, 2.0], "levels": 3}]}
        with pytest.raises(pp.PackageError, match="СПИСОК достижимых значений"):
            pp.parse_project_package(bad)

    def test_levels_outside_range_refused(self):
        """Режим вне границ — план предложил бы недостижимое (A0.6)."""
        bad = {**PACKAGE, "process": [
            {"name": "rpm", "range": [400.0, 900.0], "levels": [400.0, 1200.0]}]}
        with pytest.raises(pp.PackageError, match="ВНЕ границ"):
            pp.parse_project_package(bad)

    def test_degenerate_axis_refused(self):
        """Постоянный параметр — не ось проекта, и это сказано словами."""
        bad = {**PACKAGE, "process": [{"name": "T", "range": [180.0, 180.0]}]}
        with pytest.raises(pp.PackageError, match="строго больше"):
            pp.parse_project_package(bad)

    def test_response_clashing_with_component_refused(self):
        """Отклик и компонент — разные столбцы одной базы, совпасть не могут."""
        bad = {**PACKAGE, "responses": ["DINP"]}
        with pytest.raises(pp.PackageError, match="совпадают с компонентами"):
            pp.parse_project_package(bad)

    def test_axis_clashing_with_response_refused(self):
        bad = {**PACKAGE,
               "process": [{"name": "gloss", "range": [1.0, 2.0]}]}
        with pytest.raises(pp.PackageError, match="совпадают с откликами"):
            pp.parse_project_package(bad)

    def test_min_max_form_of_range_accepted(self):
        """Модель часто пишет {'min','max'} — принимаем, приводя к канону."""
        pkg = pp.parse_project_package({
            **PACKAGE,
            "process": [{"name": "T", "range": {"min": 165, "max": 185}}]})
        assert pkg.process[0]["range"] == [165.0, 185.0]

    def test_json_string_package_accepted(self):
        """Пакет может приехать строкой JSON (вложение, буфер) — разбираем."""
        pkg = pp.parse_project_package(json.dumps(PACKAGE, ensure_ascii=False))
        assert pkg.spec_hash == pp.parse_project_package(PACKAGE).spec_hash

    def test_spec_package_kind_refused_with_pointer(self):
        """Пакет ТОЛЬКО геометрии направляется в свой инструмент."""
        with pytest.raises(pp.PackageError, match="apply_spec"):
            pp.parse_project_package({**PACKAGE, "package_kind": "spec"})


# ----------------------------------------------------------------------
# 2. Манифест: «из JSON понятно, ЧТО загружается» (требование пользователя)
# ----------------------------------------------------------------------
class TestManifest:
    def test_blocks_named_separately(self):
        """Блоки перечислены ПОРОЗНЬ: состав, отклики, оси, режимы, ковариаты."""
        m = pp.package_manifest(pp.parse_project_package(PACKAGE))
        names = [b["блок"] for b in m["blocks"]]
        assert any("состав" in n for n in names)
        assert any("отклик" in n for n in names)
        assert any("процесс-оси" in n for n in names)
        assert any("режимы" in n for n in names)
        assert any("ковариат" in n for n in names)
        assert any("паспорт" in n for n in names)

    def test_axis_details_carry_bounds_and_units(self):
        """У оси в манифесте видны ГРАНИЦЫ и ЕДИНИЦЫ — сверять построчно."""
        m = pp.package_manifest(pp.parse_project_package(PACKAGE))
        axes = next(b for b in m["blocks"] if "процесс-оси" in b["блок"])
        assert "165…185" in axes["детали"] and "°C" in axes["детали"]
        assert "об/мин" in axes["детали"]
        assert "РЕАЛЬНЫЕ" in axes["единицы"]

    def test_response_details_carry_units(self):
        m = pp.package_manifest(pp.parse_project_package(PACKAGE))
        resp = next(b for b in m["blocks"] if "отклик" in b["блок"])
        assert "gloss [%]" in resp["детали"] and "dE [ΔE]" in resp["детали"]

    def test_levels_block_says_when_axes_continuous(self):
        """«Нет сетки» сказано словами, а не пустой строкой (A0.6)."""
        no_levels = {**PACKAGE,
                     "process": [{"name": "T", "range": [1.0, 2.0]}]}
        m = pp.package_manifest(pp.parse_project_package(no_levels))
        blk = next(b for b in m["blocks"] if "режимы" in b["блок"])
        assert "непрерывн" in blk["что"]

    def test_caption_counts_every_block(self):
        cap = pp.manifest_caption(pp.parse_project_package(PACKAGE))
        assert "компонентов" in cap and "отклики: 2" in cap
        assert "процесс-оси: 2" in cap and "на сетке: 1" in cap

    def test_blocks_dataframe_columns(self):
        """Таблица UI строится из манифеста и имеет колонку «единицы»."""
        m = pp.package_manifest(pp.parse_project_package(PACKAGE))
        df = project_blocks_dataframe(m)
        assert list(df.columns) == ["блок", "что", "детали", "единицы"]
        assert len(df) == len(m["blocks"])


# ----------------------------------------------------------------------
# 3. Проекция в поля формы сетапа (раннер собирает штатная кнопка)
# ----------------------------------------------------------------------
class TestSetupPrefill:
    def test_labels_match_campaign_ui(self):
        """Ярлыки режима дублируются строкой — разъезд ловится ТЕСТОМ."""
        from src.apps import campaign_ui as ui
        assert pp.SETUP_MODE_PHR == ui._MODE_PHR
        assert pp.SETUP_SRC_JSON == ui._PHR_SRC_JSON

    def test_prefill_fills_all_three_blocks(self):
        pre = pp.package_to_setup_prefill(pp.parse_project_package(PACKAGE))
        assert pre["setup_resp"] == "gloss, dE"
        assert pre["setup_proc"] == "T_plast, rotor_rpm"
        assert pre["setup_comp_mode"] == pp.SETUP_MODE_PHR
        assert pre["setup_seed"] == 3
        # границы осей — по индексам виджетов формы (d = число осей)
        assert pre["setup_plo_2_0"] == 165.0 and pre["setup_phi_2_0"] == 185.0
        assert pre["setup_plo_2_1"] == 400.0 and pre["setup_phi_2_1"] == 900.0

    def test_prefill_texts_parse_back_by_campaign_ui(self):
        """Текстовые поля читаются ШТАТНЫМИ парсерами формы (единый канон)."""
        from src.apps import campaign_ui as ui
        pkg = pp.parse_project_package(PACKAGE)
        pre = pp.package_to_setup_prefill(pkg)
        assert ui.parse_process_levels(pre["setup_process_levels"]) == {
            "rotor_rpm": [400.0, 900.0]}
        assert ui.parse_covariate_names(pre["setup_covariates"]) == ["SME"]
        # spec round-trip: отпечаток геометрии не зависит от канала передачи
        assert ui.parse_phr_spec_json(
            pre["setup_phr_json"]).spec_hash() == pkg.spec_hash

    def test_passport_fields_projected(self):
        pre = pp.package_to_setup_prefill(pp.parse_project_package(PACKAGE))
        assert pre["setup_campaign_label"] == "PVC-кромка-2026"
        assert pre["setup_pass_weigh_step"] == 0.1
        # iter83 (ОСОЗНАННАЯ смена контракта поля): `setup_pass_weigh_gpp`
        # показывает ВЕС ЗАМЕСА в кг. Паспорт пакета хранит масштаб ядра
        # (grams_per_phr = 10), перевод идёт по верху Σphr спеки пакета
        # (107…137 phr по листьям): 10 · 137 / 1000 = 1.37 кг. Обратный
        # перевод точен — batch_grams_per_phr(spec, 1.37) снова даёт 10.
        assert pre["setup_pass_weigh_gpp"] == pytest.approx(1.37)

    def test_links_and_pairs_text_parse_back(self):
        """Паспортные связки/пары идут в форму в её же синтаксисе."""
        from src.apps import campaign_ui as ui
        # iter79: полоса связки теперь проверяется ЯДРОМ уже на разборе пакета
        # (`normalize_process_links` → `linked_axes.normalize_links`), поэтому
        # данные обязаны быть реализуемыми. Прежняя запись (T_plast − rotor_rpm
        # ≥ 10) физически невозможна: разность этих осей лежит в [-735, -215], и
        # на кнопке «🏗 Построить проект» ядро отвергало её той же ошибкой —
        # тест проходил лишь потому, что до iter79 связки не валидировались.
        # Смысл проверки прежний: связка идёт в форму её же синтаксисом, а
        # открытая сторона полосы пишется «*».
        pkg = pp.parse_project_package({**PACKAGE, "passport": {
            "preflight_pairs": [{"left": ["DINP"], "right": ["SOFT"]}],
            "process_links": [{"name": "dT", "minuend": "rotor_rpm",
                               "subtrahend": "T_plast", "lo": 250, "hi": None}],
            "material_lots": {"DINP": "лот-7"},
            "anchor_recipes": {"серийный": {"DINP": 8.0}},
        }})
        pre = pp.package_to_setup_prefill(pkg)
        assert ui.parse_preflight_pairs(pre["setup_preflight_pairs"]) == [
            (["DINP"], ["SOFT"])]
        link = ui.parse_process_links(pre["setup_process_links"])[0]
        assert link["minuend"] == "rotor_rpm" and link["lo"] == 250.0
        assert link["hi"] is None                  # открытая сторона — «*»
        assert ui.parse_material_lots(pre["setup_material_lots"]) == {
            "DINP": "лот-7"}
        assert ui.parse_anchor_recipes(pre["setup_anchor_recipes"]) == {
            "серийный": {"DINP": 8.0}}


# ----------------------------------------------------------------------
# 4. Классы доступа: применение — акт ЧЕЛОВЕКА
# ----------------------------------------------------------------------
class TestAccessClasses:
    def test_readonly_tools_registered(self):
        names = tool_names([READONLY])
        assert "project_schema" in names
        assert "validate_project_package" in names

    def test_propose_is_available_to_model(self):
        assert "propose_project" in tool_names([PROPOSE])
        assert "propose_project" in tool_names(list(AGENT_KINDS))

    def test_write_tools_never_offered_to_model(self):
        """Модель не должна даже видеть кнопочные инструменты (iter63-канон)."""
        agent = tool_names(list(AGENT_KINDS))
        assert "apply_project" not in agent
        assert "reject_project" not in agent
        assert {"apply_project", "reject_project"} <= set(tool_names([WRITE]))

    def test_apply_blocked_for_agent_kinds(self):
        ctx = _ctx(consent=ConsentRegistry())
        with pytest.raises(ToolError, match="write"):
            dispatch(ctx, "apply_project",
                     {"project_id": "proj_1", "human_token": "я-сам"},
                     allowed_kinds=list(AGENT_KINDS))

    def test_consent_actions_extended(self):
        assert "apply_project" in ACTIONS and "reject_project" in ACTIONS

    def test_token_for_other_action_does_not_fit(self):
        """Согласие на отклонение не годится для принятия (и наоборот)."""
        ctx = _ctx(consent=ConsentRegistry())
        pid = _stage(ctx)
        token = issue_reject_project_token(ctx, pid)
        with pytest.raises(ToolError, match="выдано на действие 'reject_project'"):
            dispatch(ctx, "apply_project",
                     {"project_id": pid, "human_token": token},
                     allowed_kinds=[WRITE])

    def test_apply_needs_token_at_all(self):
        ctx = _ctx(consent=ConsentRegistry())
        pid = _stage(ctx)
        with pytest.raises(ToolError):
            dispatch(ctx, "apply_project",
                     {"project_id": pid, "human_token": ""},
                     allowed_kinds=[WRITE])


# ----------------------------------------------------------------------
# 5. Инструменты: схема, dry-run, стейдж
# ----------------------------------------------------------------------
class TestTools:
    def test_schema_works_without_project(self):
        """Инструмент первичного ввода не может требовать проект."""
        out = dispatch(_ctx(), "project_schema", {})
        assert out["required"] == ["spec", "responses", "process"]
        assert out["current"]["project_present"] is False
        assert "propose_project" in out["current"]["note"]

    def test_schema_reports_existing_project(self):
        runner = _Runner(props=("gloss", "dE"), proc=("T_plast",))
        out = dispatch(_ctx(runner=runner), "project_schema", {})
        assert out["current"]["project_present"] is True
        assert out["current"]["responses"] == ["gloss", "dE"]
        assert "propose_spec" in out["current"]["note"]

    def test_schema_example_is_valid(self):
        """Пример из инструмента собирается ядром (иначе он вредит)."""
        out = dispatch(_ctx(), "project_schema", {"include_example": True})
        pp.parse_project_package(out["example"])

    def test_dry_run_returns_manifest_not_exception(self):
        out = dispatch(_ctx(), "validate_project_package",
                       {"package": PACKAGE})
        assert out["ok"] is True
        assert out["manifest"]["blocks"]
        assert out["q_components"] and out["dim_z"]

    def test_dry_run_refusal_is_result(self):
        """Ход модели не должен падать: отказ — это ok=False с подсказкой."""
        out = dispatch(_ctx(), "validate_project_package",
                       {"package": {"spec": SPEC_BLOCK}})
        assert out["ok"] is False
        assert "responses" in out["error"]
        assert "project_schema" in out["hint"]

    def test_invalid_package_not_staged(self):
        """Неприменимый пакет в стейдж не попадает: кнопка не должна врать."""
        ctx = _ctx()
        out = dispatch(ctx, "propose_project",
                       {"package": {"spec": SPEC_BLOCK}, "rationale": "x"},
                       allowed_kinds=[PROPOSE])
        assert out["staged"] is False and out["ok"] is False
        assert ctx.session.staged_projects() == []

    def test_propose_warns_when_project_exists(self):
        ctx = _ctx(runner=_Runner())
        out = dispatch(ctx, "propose_project",
                       {"package": PACKAGE, "rationale": "x"},
                       allowed_kinds=[PROPOSE])
        assert out["staged"] is True
        assert "УЖЕ есть собранный проект" in out["warning"]

    def test_staged_dataframe_shows_blocks(self):
        """Таблица стейджа отвечает «что приедет» числами по блокам."""
        ctx = _ctx()
        _stage(ctx)
        df = staged_projects_dataframe(ctx.session, only_staged=True)
        row = df.iloc[0]
        assert row["компонентов"] == 4        # RESIN, DINP, PBNK, CPE
        assert row["откликов"] == 2 and row["процесс-осей"] == 2
        assert "gloss [%]" in row["отклики"]
        assert "165…185" in row["оси"]


# ----------------------------------------------------------------------
# 6. ПРИНЯТИЕ человеком — то, что было сломано
# ----------------------------------------------------------------------
class TestHumanApply:
    def test_apply_returns_setup_prefill_and_changes_status(self, tmp_path):
        """Главный тест шага: кнопка ДАЁТ результат, а не «успех» без следа."""
        ctx = _ctx(tmp_path, consent=ConsentRegistry())
        pid = _stage(ctx)
        out = actx.human_apply_project(ctx, pid, author="человек (тест)")
        assert out["ok"] is True
        assert out["status"] == PATCH_APPLIED
        pre = out["setup_prefill"]
        assert pre["setup_resp"] == "gloss, dE"
        assert pre["setup_proc"] == "T_plast, rotor_rpm"
        assert ctx.session.project_by_id(pid).status == PATCH_APPLIED

    def test_apply_does_not_build_runner(self, tmp_path):
        """Путь сборки ОДИН: раннер рождает штатная кнопка, не инструмент."""
        ctx = _ctx(tmp_path, consent=ConsentRegistry())
        pid = _stage(ctx)
        out = actx.human_apply_project(ctx, pid)
        assert ctx.runner is None                 # движок не создан
        assert "Построить проект" in out["next_step"]

    def test_apply_is_single_use(self, tmp_path):
        ctx = _ctx(tmp_path, consent=ConsentRegistry())
        pid = _stage(ctx)
        actx.human_apply_project(ctx, pid)
        with pytest.raises(ToolError, match="повторное применение"):
            actx.human_apply_project(ctx, pid)

    def test_apply_refused_when_project_already_built(self, tmp_path):
        """Пакет проекта не заводит проект заново: точки и ветки не теряем."""
        ctx = _ctx(tmp_path, consent=ConsentRegistry())
        pid = _stage(ctx)
        ctx.runner = _Runner(points=[object()])
        with pytest.raises(ToolError, match="уже собран проект"):
            actx.human_apply_project(ctx, pid)
        assert ctx.session.project_by_id(pid).status == PATCH_STAGED

    def test_apply_writes_decision_log(self, tmp_path):
        """Решение фиксируется журналом: состав, отклики, оси — в записи."""
        ctx = _ctx(tmp_path, consent=ConsentRegistry())
        pid = _stage(ctx, label="кромка ПВХ: проект", level="L1",
                     source="таблица технолога")
        actx.human_apply_project(ctx, pid, author="Жихарев",
                                 note="сверено с паспортами")
        rec = store.read_log(tmp_path, PROJECT, "decisions")[-1]
        assert rec["kind"] == "apply_project"
        assert rec["author"] == "Жихарев"
        assert rec["responses"] == ["gloss", "dE"]
        assert rec["process"] == ["T_plast", "rotor_rpm"]
        assert rec["level"] == "L1"
        assert rec["spec_hash_after"]

    def test_reject_writes_decision_and_blocks_apply(self, tmp_path):
        ctx = _ctx(tmp_path, consent=ConsentRegistry())
        pid = _stage(ctx)
        actx.human_reject_project(ctx, pid, "ждём границы от технолога",
                                  author="Жихарев")
        assert ctx.session.project_by_id(pid).status == PATCH_REJECTED
        rec = store.read_log(tmp_path, PROJECT, "decisions")[-1]
        assert rec["kind"] == "reject_project"
        assert "ждём границы" in rec["rationale"]
        with pytest.raises(ToolError, match="уже в статусе"):
            actx.human_apply_project(ctx, pid)

    def test_apply_caption_says_what_is_left_to_do(self, tmp_path):
        """Подпись не должна создавать иллюзию «проект готов»."""
        ctx = _ctx(tmp_path, consent=ConsentRegistry())
        cap = project_apply_caption(
            actx.human_apply_project(ctx, _stage(ctx)))
        assert "откликов: 2" in cap and "процесс-осей: 2" in cap
        assert "Построить проект" in cap

    def test_token_bound_to_package_spec_hash(self, tmp_path):
        """Токен привязан к отпечатку спеки ПАКЕТА (проекта ещё нет)."""
        ctx = _ctx(tmp_path, consent=ConsentRegistry())
        pid = _stage(ctx)
        token = issue_apply_project_token(ctx, pid)
        pkg_hash = pp.parse_project_package(PACKAGE).spec_hash
        out = dispatch(ctx, "apply_project",
                       {"project_id": pid, "human_token": token},
                       allowed_kinds=[WRITE])
        assert out["spec_hash"] == pkg_hash


# ----------------------------------------------------------------------
# 7. Персистентность: статус решения переживает перезапуск
# ----------------------------------------------------------------------
class TestPersistence:
    def test_persist_session_saves_applied_status(self, tmp_path):
        """Наблюдалось: applied в журнале, staged в session.json (iter73-фикс)."""
        ctx = _ctx(tmp_path, consent=ConsentRegistry())
        pid = _stage(ctx)
        store.save_session(ctx.session, tmp_path, PROJECT)
        actx.human_apply_project(ctx, pid)
        assert actx.persist_session(ctx) is True
        again = store.load_session(tmp_path, PROJECT)
        assert again.project_by_id(pid).status == PATCH_APPLIED

    def test_package_survives_save_load(self, tmp_path):
        ctx = _ctx(tmp_path, consent=ConsentRegistry())
        pid = _stage(ctx, label="кромка ПВХ")
        store.save_session(ctx.session, tmp_path, PROJECT)
        again = store.load_session(tmp_path, PROJECT)
        p = again.project_by_id(pid)
        assert p is not None and p.label == "кромка ПВХ"
        # пакет восстановлен целиком: спека собирается, отклики на месте
        pkg = pp.parse_project_package(p.payload())
        assert pkg.response_names == ["gloss", "dE"]

    def test_sessions_without_projects_key_load(self, tmp_path):
        """Сессии, записанные до iter73, обязаны открываться как раньше."""
        s = new_session(PROJECT)
        s.add_message("user", "вопрос про УФ")
        store.save_session(s, tmp_path, PROJECT)
        path = tmp_path / PROJECT / "assistant" / "session.json"
        state = json.loads(path.read_text(encoding="utf-8"))
        state.pop("projects", None)
        path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        again = store.load_session(tmp_path, PROJECT)
        assert again.projects == []
        assert again.messages[0].content == "вопрос про УФ"

    def test_persist_without_project_is_false_not_crash(self):
        """Нет проекта на диске — честное False, а не исключение (A0.6)."""
        assert actx.persist_session(_ctx()) is False


# ----------------------------------------------------------------------
# 8. Видимость состояния и маршрутизация
# ----------------------------------------------------------------------
class TestStatusAndRouting:
    def test_status_caption_without_project_warns_about_name_field(self):
        """Ровно та путаница, на которую указал пользователь."""
        cap = project_status_caption(None, project="my_project")
        assert "НЕТ" in cap
        assert "ДЛЯ СОХРАНЕНИЯ" in cap

    def test_status_caption_with_project_lists_blocks(self):
        cap = project_status_caption(
            _Runner(props=("gloss",), proc=("T", "P"), mix=("A", "B")),
            project="pvc")
        assert "Проект собран" in cap and "«pvc»" in cap
        assert "2 компонентов × 2 процесс-осей" in cap
        assert "phr-спека НЕ задана" in cap        # спеки нет — сказано прямо

    def test_status_caption_shows_spec_hash(self):
        spec = pp.parse_project_package(PACKAGE).spec
        cap = project_status_caption(_Runner(spec))
        assert spec.spec_hash()[:12] in cap

    def test_router_sends_project_request_to_project_intake(self):
        from src.assistant.prompts import route
        for q in ("Проекта пока нет, собери проект по этому составу",
                  "Заведём кампанию: состав, отклики и процесс-оси",
                  "создай проект целиком"):
            assert route(q).scenario == "project_intake", q

    def test_router_keeps_spec_only_requests_in_spec_intake(self):
        """Правка геометрии по-прежнему идёт пакетом СПЕКИ, не проекта."""
        from src.assistant.prompts import route
        assert route("добавь новый компонент стеарат кальция"
                     ).scenario == "spec_intake"
        assert route("смени роль PBNK на SHARE_SIMPLEX"
                     ).scenario == "spec_intake"

    def test_scenario_ten_registered_and_routes(self):
        from src.assistant.prompts import (GOLDEN_SCENARIOS, HUMAN_ONLY,
                                           route_scenario, scenario)
        sc = scenario("project_intake")
        assert sc.id == 10
        assert route_scenario(sc).scenario == "project_intake"
        assert "apply_project" in HUMAN_ONLY and "reject_project" in HUMAN_ONLY
        assert len(GOLDEN_SCENARIOS) == 10

    def test_limits_block_explains_two_package_kinds(self):
        """Промпт обязан различать «нет проекта» и «нет спеки»."""
        from src.assistant.prompts import LIMITS_BLOCK
        assert "propose_project" in LIMITS_BLOCK
        assert "нет проекта" in LIMITS_BLOCK.lower()

    def test_prompt_without_runner_offers_project_package(self):
        from src.assistant.prompts import architect_system_prompt
        txt = architect_system_prompt(project=PROJECT, has_runner=False)
        assert "propose_project" in txt
        assert "НЕКУДА" in txt


# ----------------------------------------------------------------------
# 9. Панель дока: кнопки и блоки на экране
# ----------------------------------------------------------------------
class _Col:
    def __init__(self, seen):
        self._seen = seen

    def button(self, label, **kw):
        self._seen["buttons"].append((label, bool(kw.get("disabled"))))
        return False


class _Exp:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeSt:
    """Минимальный двойник Streamlit: считаем, что нарисовано (как в iter71)."""

    def __init__(self):
        self.seen = {"buttons": [], "dataframe": 0, "json": 0, "warning": 0}

    def markdown(self, *_a, **_k):
        pass

    def caption(self, *_a, **_k):
        pass

    def warning(self, *_a, **_k):
        self.seen["warning"] += 1

    def dataframe(self, *_a, **_k):
        self.seen["dataframe"] += 1

    def json(self, *_a, **_k):
        self.seen["json"] += 1

    def expander(self, *_a, **_k):
        return _Exp()

    def columns(self, n):
        return [_Col(self.seen) for _ in range(n)]

    def text_input(self, *_a, **_k):
        return ""


class TestDockPanel:
    def test_panel_shows_blocks_json_and_buttons(self, monkeypatch):
        from src.apps import assistant_dock as dock
        ctx = _ctx()
        _stage(ctx, label="кромка ПВХ: проект")
        fake = _FakeSt()
        monkeypatch.setattr(dock, "st", fake)
        dock._render_project_packages(ctx, ctx.session, None)
        # две таблицы: список пакетов + МАНИФЕСТ блоков (что грузится)
        assert fake.seen["dataframe"] == 2
        assert fake.seen["json"] == 1            # JSON доступен человеку целиком
        labels = [b for b, _ in fake.seen["buttons"]]
        assert any("Принять проект" in b for b in labels)
        assert any("Отклонить проект" in b for b in labels)

    def test_apply_button_disabled_when_project_exists(self, monkeypatch):
        """Кнопка без последствий не должна быть нажимаемой (исходный отказ)."""
        from src.apps import assistant_dock as dock
        ctx = _ctx(runner=_Runner())
        _stage(ctx)
        fake = _FakeSt()
        monkeypatch.setattr(dock, "st", fake)
        dock._render_project_packages(ctx, ctx.session, ctx.runner)
        disabled = {b: d for b, d in fake.seen["buttons"]}
        assert disabled["✅ Принять проект"] is True
        assert fake.seen["warning"] >= 1          # причина сказана словами

    def test_empty_panel_explains_when_package_appears(self, monkeypatch):
        from src.apps import assistant_dock as dock
        ctx = _ctx()
        fake = _FakeSt()
        monkeypatch.setattr(dock, "st", fake)
        dock._render_project_packages(ctx, ctx.session, None)
        assert fake.seen["buttons"] == [] and fake.seen["dataframe"] == 0


# ----------------------------------------------------------------------
# 10. Живое приложение: состояние проекта видно на «Старте»
# ----------------------------------------------------------------------
APP = "src/apps/streamlit_app.py"


def test_start_tab_states_project_presence():
    """Статус-строка стоит ВЫШЕ формы: «есть проект или нет» без догадок.

    До iter73 это читалось по косвенным признакам (поле «Имя проекта» со
    значением по умолчанию, раскрытость экспандера), а единственная явная
    надпись стояла НИЖЕ длинной формы сетапа — её не было видно без скролла.
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception
    warns = [str(w.value) for w in at.warning]
    assert any("Проекта в сессии НЕТ" in w for w in warns), warns
    # и сказано, что имя в поле — это имя для сохранения, а не открытый проект
    assert any("ДЛЯ СОХРАНЕНИЯ" in w for w in warns)


def test_start_tab_reports_built_project():
    """После сборки демо-проекта та же строка становится утвердительной."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=300).run()
    demo = [w for w in at.button if w.key == "camp_create"]
    assert demo, "кнопка демо-проекта не найдена"
    demo[0].click().run()
    assert not at.exception
    start = [w for w in at.button if w.key == "ws_tab_start"]
    if start:
        start[0].click().run()
    assert not at.exception
    ok = [str(s.value) for s in at.success]
    assert any("Проект собран" in s for s in ok), ok


def test_dock_shows_project_package_panel():
    """Панель пакетов проекта нарисована в доке даже когда стейдж пуст."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception
    texts = [str(m.value) for m in at.markdown]
    assert any("Предложенные проекты" in t for t in texts), \
        "панель пакетов проекта не найдена в доке"
