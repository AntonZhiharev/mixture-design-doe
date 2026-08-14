# Copyright 2026 DOE contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 96 — помощник ПРЕДЛАГАЕТ запись в журналы, человек правит и пишет.

Разрыв, оставшийся после iter80: инструменты записи существуют
(``record_decision`` / ``add_local_fact``, класс ``write``), кнопки ручного
ввода существуют («✍️ Записать решение в журнал»), а предложение помощника
доходило до журнала ТОЛЬКО текстом ответа. В живой сессии проекта «кромка ПВХ»
это видно дословно: формулировка для журнала лежит в ``## OPEN_QUESTIONS``
(«кнопка «✍️ Записать решение в журнал» — записываете вы»), то есть перенос в
поля оставался ручной работой человека — и регулярно не делался: договорённость
обсудили, журнал пуст.

Шаг закрывает это тем же контуром, что уже принят для патчей, пакетов спеки,
пакетов проекта и правок полей формы (``propose_*`` → стейдж → кнопка человека),
с одним существенным отличием: **поля записи РЕДАКТИРУЕМЫ до фиксации**.
Причина — авторство: запись в журнал решений и особенно L1-факт идут от имени
человека (ASSISTANT_SPEC §370, L1 отменяет литературу), поэтому он подписывает
свою формулировку, а не одобряет чужую.

Здесь фиксируется:

  * ``propose_decision`` / ``propose_fact`` — класс ``propose`` (модели
    доступны), кладут запись в СТЕЙДЖ сессии и НИЧЕГО не пишут на диск;
  * ``apply_note`` / ``reject_note`` — класс ``write``: модели недоступны,
    требуют разовый токен, применяют ПРАВКУ человека и пишут в журнал именно её;
  * инвариант iter63 не ослаблен, а запись без обязательных полей не проходит
    ни на входе (стейдж), ни на выходе (правка человека всё стёрла);
  * старые сессии (без ключа ``notes``) открываются как были;
  * панель живёт в правой инфо-панели, поля — в ``st.form`` (iter95), запись
    идёт человеческим путём ``context.human_*``;
  * MCP-сервер ``doe-campaign`` новые инструменты НЕ экспортирует: предложение
    должно попасть в панель, где у человека есть кнопки.
"""
from __future__ import annotations

import inspect
import json

import pytest

from src.assistant import context as actx
from src.assistant import store, views
from src.assistant.consent import ACTIONS, ConsentRegistry
from src.assistant.session import (NOTE_DECISION, NOTE_FACT, PATCH_APPLIED,
                                   PATCH_REJECTED, PATCH_STAGED,
                                   AssistantSession, StagedNote, new_session)
from src.assistant.tools import (AGENT_KINDS, PROPOSE, READONLY, WRITE,
                                 ToolContext, ToolError, dispatch, tool_names)
from src.design.phr_sampler import PhrSpec

PROJECT = "кромка ПВХ"

SPEC_NODES = [
    {"name": "RESIN", "role": "FIXED", "value": 100.0},
    {"name": "DINP", "role": "ABSOLUTE", "range": [4.0, 14.0]},
]


def _spec() -> PhrSpec:
    return PhrSpec.from_dicts(SPEC_NODES)


def _ctx(tmp_path=None, *, spec=True, consent=None) -> ToolContext:
    return ToolContext(spec=_spec() if spec else None,
                       session=new_session(PROJECT),
                       root=str(tmp_path) if tmp_path is not None else "",
                       project=PROJECT if tmp_path is not None else "",
                       extra={"consent": consent or ConsentRegistry()})


def _stage_decision(ctx, **kw):
    args = {"title": "мел до 100 phr для белых компаундов",
            "rationale": "L1-факт цеха: так льют серийно",
            "nodes": ["DINP"], "level": "L1"}
    args.update(kw)
    return dispatch(ctx, "propose_decision", args,
                    allowed_kinds=[READONLY, PROPOSE])


def _stage_fact(ctx, **kw):
    args = {"statement": "смеситель типа A — не выше 120 °C",
            "scope": "смеситель", "source": "SOP цеха"}
    args.update(kw)
    return dispatch(ctx, "propose_fact", args,
                    allowed_kinds=[READONLY, PROPOSE])


# ======================================================================
# 1. Граница классов: предлагать — модели, записывать — человеку
# ======================================================================
class TestAccessBoundary:

    def test_propose_tools_are_available_to_the_model(self):
        agent = set(tool_names(AGENT_KINDS))
        assert {"propose_decision", "propose_fact"} <= agent

    def test_write_tools_are_not(self):
        """iter63 не ослаблен: фиксирует запись человек кнопкой."""
        agent = set(tool_names(AGENT_KINDS))
        assert not ({"apply_note", "reject_note"} & agent)
        assert {"apply_note", "reject_note"} <= set(tool_names([WRITE]))

    def test_model_call_of_write_tool_is_refused_by_class(self):
        with pytest.raises(ToolError, match="класс"):
            dispatch(_ctx(), "apply_note",
                     {"note_id": "note_1", "human_token": "x"},
                     allowed_kinds=AGENT_KINDS)

    def test_consent_actions_registered(self):
        """Токен выдаётся только на действие из белого списка."""
        assert "apply_note" in ACTIONS and "reject_note" in ACTIONS

    def test_prompt_forbids_write_tools_in_every_scenario(self):
        from src.assistant.prompts import HUMAN_ONLY
        assert "apply_note" in HUMAN_ONLY and "reject_note" in HUMAN_ONLY


# ======================================================================
# 2. Предложение: стейдж, а не запись
# ======================================================================
class TestProposeGoesToStage:

    def test_decision_is_staged_and_nothing_is_written(self, tmp_path):
        ctx = _ctx(tmp_path)
        out = _stage_decision(ctx)
        assert out["staged"] is True and out["kind"] == NOTE_DECISION
        assert out["fields"]["title"].startswith("мел до 100")
        # Поля названы явно: человек должен знать, что именно он правит.
        assert out["editable"] == ["title", "rationale", "nodes", "author"]
        assert "Зафиксировать решение" in out["button"]
        # Журнал на диске пуст: предложение — не запись.
        assert store.read_log(str(tmp_path), PROJECT, "decisions") == []
        assert len(ctx.session.staged_notes()) == 1

    def test_fact_is_staged_with_l1_by_default(self, tmp_path):
        ctx = _ctx(tmp_path)
        out = _stage_fact(ctx)
        assert out["staged"] is True and out["kind"] == NOTE_FACT
        assert ctx.session.staged_notes()[0].level == "L1"
        assert store.read_log(str(tmp_path), PROJECT, "local_facts") == []

    def test_incomplete_decision_is_not_staged(self):
        """Решение без обоснования — не предложение, а видимость работы."""
        ctx = _ctx()
        out = _stage_decision(ctx, rationale="   ")
        assert out["staged"] is False and "rationale" in out["error"]
        assert ctx.session.staged_notes() == []

    def test_empty_fact_is_not_staged(self):
        ctx = _ctx()
        out = _stage_fact(ctx, statement="")
        assert out["staged"] is False and "statement" in out["error"]
        assert ctx.session.staged_notes() == []

    def test_nodes_accept_string_and_list(self):
        """Модель присылает узлы и строкой, и списком — обе формы годятся."""
        ctx = _ctx()
        _stage_decision(ctx, nodes="DINP, RESIN")
        assert ctx.session.staged_notes()[0].fields["nodes"] == ["DINP",
                                                                "RESIN"]

    def test_rationale_is_required_by_schema(self):
        with pytest.raises(ToolError, match="rationale"):
            dispatch(_ctx(), "propose_decision", {"title": "t"},
                     allowed_kinds=[READONLY, PROPOSE])

    def test_unknown_note_kind_is_refused_by_container(self):
        with pytest.raises(ValueError, match="вид записи"):
            new_session(PROJECT).stage_note(
                StagedNote(kind="branch_goal", fields={"title": "t"}))


# ======================================================================
# 3. Фиксация человеком: в журнал уходит ПРАВКА, а не исходный текст
# ======================================================================
class TestHumanApplyWithEdits:

    def test_edited_fields_land_in_the_journal(self, tmp_path):
        """Главный смысл шага: человек правит поля ДО записи."""
        ctx = _ctx(tmp_path)
        note_id = _stage_decision(ctx)["note_id"]
        out = actx.human_apply_note(
            ctx, note_id,
            fields={"title": "мел до 90 phr (проверено на линии 2)",
                    "rationale": "замер КИП 14.08, протокол №17",
                    "nodes": ["DINP"], "author": "Жихарев"},
            author="человек (UI)")
        assert out["ok"] and out["status"] == PATCH_APPLIED
        assert set(out["edited_fields"]) == {"title", "rationale", "author"}
        rec = store.read_log(str(tmp_path), PROJECT, "decisions")[-1]
        assert rec["title"] == "мел до 90 phr (проверено на линии 2)"
        assert rec["rationale"].startswith("замер КИП")
        assert rec["kind"] == "decision" and rec["nodes"] == ["DINP"]
        # Происхождение записи видно: предложил помощник, поправил человек.
        assert rec["proposed_by"] == "assistant"
        assert rec["edited_by_human"] is True
        assert rec["author"] == "человек (UI)"
        # spec_hash пишется на момент записи — иначе решение не сопоставить
        # с геометрией кампании.
        assert rec["spec_hash"] == _spec().spec_hash()

    def test_unedited_proposal_is_written_as_proposed(self, tmp_path):
        ctx = _ctx(tmp_path)
        note_id = _stage_fact(ctx)["note_id"]
        out = actx.human_apply_note(ctx, note_id, author="технолог")
        assert out["edited_fields"] == []
        rec = store.read_log(str(tmp_path), PROJECT, "local_facts")[-1]
        assert rec["statement"] == "смеситель типа A — не выше 120 °C"
        assert rec["level"] == "L1" and rec["edited_by_human"] is False
        assert "L1 отменяет" in out["note"]

    def test_model_can_read_what_human_wrote(self, tmp_path):
        """Запись бесполезна, если не видна помощнику в следующем ходе."""
        ctx = _ctx(tmp_path)
        actx.human_apply_note(ctx, _stage_fact(ctx)["note_id"])
        got = dispatch(ctx, "get_local_facts", {"scope": "смеситель"})
        assert got["n"] == 1
        actx.human_apply_note(ctx, _stage_decision(ctx)["note_id"])
        dec = dispatch(ctx, "get_decisions", {"limit": 5})
        assert dec["decisions"][-1]["title"].startswith("мел до 100")

    def test_apply_is_single_use(self, tmp_path):
        """Токен одноразовый, и статус записи терминальный."""
        ctx = _ctx(tmp_path)
        note_id = _stage_decision(ctx)["note_id"]
        actx.human_apply_note(ctx, note_id)
        with pytest.raises((ToolError, ValueError)):
            actx.human_apply_note(ctx, note_id)
        assert len(store.read_log(str(tmp_path), PROJECT, "decisions")) == 1
        assert ctx.session.note_by_id(note_id).status == PATCH_APPLIED
        assert ctx.session.staged_notes() == []

    def test_without_token_nothing_is_written(self, tmp_path):
        ctx = _ctx(tmp_path)
        note_id = _stage_decision(ctx)["note_id"]
        with pytest.raises(ToolError):
            dispatch(ctx, "apply_note",
                     {"note_id": note_id, "human_token": ""},
                     allowed_kinds=[WRITE])
        assert store.read_log(str(tmp_path), PROJECT, "decisions") == []

    def test_edit_that_empties_required_field_is_refused(self, tmp_path):
        """Человек стёр обоснование — пустую запись писать нельзя."""
        ctx = _ctx(tmp_path)
        note_id = _stage_decision(ctx)["note_id"]
        with pytest.raises(ToolError, match="rationale"):
            actx.human_apply_note(ctx, note_id, fields={"rationale": "  "})
        assert store.read_log(str(tmp_path), PROJECT, "decisions") == []
        # Запись осталась в стейдже: отказ не должен её терять.
        assert ctx.session.note_by_id(note_id).status == PATCH_STAGED

    def test_unknown_field_is_refused_not_swallowed(self, tmp_path):
        ctx = _ctx(tmp_path)
        note_id = _stage_fact(ctx)["note_id"]
        with pytest.raises(ToolError, match="title"):
            actx.human_apply_note(ctx, note_id, fields={"title": "не то поле"})
        assert ctx.session.note_by_id(note_id).status == PATCH_STAGED

    def test_decision_without_spec_is_written_anyway(self, tmp_path):
        """Решение бывает и до геометрии («берём кромку, а не профиль»)."""
        ctx = _ctx(tmp_path, spec=False)
        staged = _stage_decision(ctx, title="берём кромку, а не профиль")
        out = actx.human_apply_note(ctx, staged["note_id"])
        # `persisted` — поле ОТВЕТА (флаг «легло на диск»), в самой записи
        # журнала его нет: журнал хранит решение, а не отчёт о записи.
        assert out["decision"]["persisted"] is True
        rec = store.read_log(str(tmp_path), PROJECT, "decisions")[-1]
        assert rec["spec_hash"] == ""
        assert rec["title"] == "берём кромку, а не профиль"

    def test_no_project_says_so_instead_of_silent_loss(self):
        """Без проекта запись некуда положить — говорим прямо (A0.6)."""
        ctx = _ctx()
        out = actx.human_apply_note(ctx, _stage_decision(ctx)["note_id"])
        assert out["decision"]["persisted"] is False
        assert out["decision"]["note"]


# ======================================================================
# 4. Отказ человека тоже попадает в историю
# ======================================================================
class TestHumanReject:

    def test_reject_logs_reason_and_frees_the_panel(self, tmp_path):
        ctx = _ctx(tmp_path)
        note_id = _stage_fact(ctx)["note_id"]
        out = actx.human_reject_note(ctx, note_id, "это про другой смеситель",
                                     author="технолог")
        assert out["status"] == PATCH_REJECTED
        assert ctx.session.staged_notes() == []
        # Факт НЕ записан, а отказ виден в журнале решений.
        assert store.read_log(str(tmp_path), PROJECT, "local_facts") == []
        rec = store.read_log(str(tmp_path), PROJECT, "decisions")[-1]
        assert rec["kind"] == "reject_fact_note"
        assert "смеситель типа A" in rec["title"]
        assert rec["rationale"] == "это про другой смеситель"

    def test_reject_twice_is_an_error(self, tmp_path):
        ctx = _ctx(tmp_path)
        note_id = _stage_decision(ctx)["note_id"]
        actx.human_reject_note(ctx, note_id, "рано")
        with pytest.raises((ToolError, ValueError)):
            actx.human_reject_note(ctx, note_id, "рано")


# ======================================================================
# 5. Показ: таблица предложений и подписи журнала
# ======================================================================
class TestViews:

    def _session(self) -> AssistantSession:
        s = new_session(PROJECT)
        s.stage_note(StagedNote(
            kind=NOTE_DECISION, level="L1", label="мел",
            fields={"title": "мел до 100 phr", "rationale": "опыт цеха",
                    "nodes": ["DINP"], "author": "Жихарев"}))
        s.stage_note(StagedNote(
            kind=NOTE_FACT, level="L1",
            fields={"statement": "смеситель A — не выше 120 °C",
                    "scope": "смеситель", "source": "SOP", "author": ""}))
        return s

    def test_dataframe_shows_kind_subject_and_edit_flag(self):
        df = views.staged_notes_dataframe(self._session(), only_staged=True)
        assert list(df["вид"]) == ["решение (журнал решений)",
                                   "факт производства (L1)"]
        assert df.iloc[0]["суть"] == "мел до 100 phr"
        assert df.iloc[0]["область/узлы"] == "DINP"
        assert df.iloc[1]["область/узлы"] == "смеситель"
        # Кто автор формулировки, попавшей в журнал, — вопрос ревизора.
        assert list(df["правил человек"]) == ["нет", "нет"]

    def test_caption_counts_kinds_separately(self):
        txt = views.staged_notes_caption(self._session())
        assert "ждут решения: 2" in txt and "решений: 1" in txt
        assert "фактов производства: 1" in txt

    def test_empty_panel_explains_itself(self):
        txt = views.staged_notes_caption(new_session(PROJECT))
        assert "Пусто" in txt and "поправить" in txt

    def test_apply_caption_names_journal_and_edits(self):
        txt = views.note_apply_caption(
            {"kind": NOTE_DECISION, "edited_fields": ["title"],
             "decision": {"persisted": True}})
        assert "журнал решений" in txt and "title" in txt
        warn = views.note_apply_caption(
            {"kind": NOTE_FACT, "edited_fields": [],
             "fact": {"persisted": False, "note": "проект не указан"}})
        assert "факты производства (L1)" in warn and "не сохранена" in warn

    def test_button_labels_come_from_one_source(self):
        from src.assistant.session import NOTE_BUTTON
        assert views.note_button(NOTE_DECISION) == NOTE_BUTTON[NOTE_DECISION]
        assert views.note_button(NOTE_FACT) == NOTE_BUTTON[NOTE_FACT]
        assert views.note_button("что-то новое").startswith("✅")

    def test_reject_kinds_labelled_in_journal_table(self):
        recs = [{"ts": "2026-08-14T10:00:00+00:00",
                 "kind": "reject_decision_note", "title": "ОТКЛОНЕНО: …",
                 "author": "человек", "rationale": "рано"},
                {"ts": "2026-08-14T10:05:00+00:00", "kind": "reject_fact_note",
                 "title": "ОТКЛОНЕНО: …", "author": "человек",
                 "rationale": "другой участок"}]
        df = views.decisions_dataframe(recs)
        assert list(df["вид"]) == ["запись решения отклонена",
                                   "запись факта отклонена"]
        # Отказы считаются отдельно от применений (контракт iter80).
        assert "отклонено: 2" in views.decisions_caption(recs)


# ======================================================================
# 6. Сессия: сериализация и совместимость со старыми файлами
# ======================================================================
class TestSessionRoundTrip:

    def test_notes_survive_save_and_load(self, tmp_path):
        s = new_session(PROJECT)
        s.stage_note(StagedNote(kind=NOTE_FACT, level="L1",
                                fields={"statement": "плотность измеряется",
                                        "scope": "cost", "source": "",
                                        "author": "технолог"}))
        store.save_session(s, str(tmp_path), PROJECT)
        back = store.load_session(str(tmp_path), PROJECT)
        assert len(back.staged_notes()) == 1
        got = back.staged_notes()[0]
        assert got.kind == NOTE_FACT and got.level == "L1"
        assert got.fields["statement"] == "плотность измеряется"

    def test_old_session_without_notes_key_opens(self):
        """Шаг не должен стоить человеку уже накопленной переписки."""
        state = new_session(PROJECT).to_state()
        state.pop("notes")
        s = AssistantSession.from_state(json.loads(json.dumps(state)))
        assert s.notes == [] and s.is_empty()

    def test_status_transition_is_terminal(self):
        s = new_session(PROJECT)
        note = s.stage_note(StagedNote(kind=NOTE_DECISION,
                                       fields={"title": "t",
                                               "rationale": "r"}))
        s.set_note_status(note.id, PATCH_APPLIED)
        with pytest.raises(ValueError, match="повторный переход"):
            s.set_note_status(note.id, PATCH_REJECTED)
        with pytest.raises(KeyError):
            s.set_note_status("note_нет", PATCH_APPLIED)


# ======================================================================
# 7. Интерфейс: панель в правой зоне, поля в форме, путь человеческий
# ======================================================================
class TestPanelContract:
    """Контракт проверяется ПО ИСХОДНИКУ (как iter80 §5/iter95): рендер
    браузера юнит-тест проверить не может — честная граница шага."""

    def test_panel_lives_in_the_info_column(self):
        dock = pytest.importorskip("src.apps.assistant_dock")
        info = inspect.getsource(dock.render_assistant_info)
        chat = inspect.getsource(dock.render_assistant_dock)
        assert "_render_note_proposals(" in info
        assert "_render_note_proposals(" not in chat

    def test_fields_are_editable_widgets_inside_a_form(self):
        """Смысл шага в UI: поля правятся, и правка не перезапускает скрипт."""
        dock = pytest.importorskip("src.apps.assistant_dock")
        src = inspect.getsource(dock._render_note_proposals) + \
            inspect.getsource(dock._note_fields_form)
        assert 'st.form(f"dock_note_form_' in src
        assert "st.form_submit_button(" in src
        assert "st.button(" not in src
        assert "st.text_input(" in src and "st.text_area(" in src
        assert "clear_on_submit" not in src

    def test_write_goes_through_human_path_only(self):
        dock = pytest.importorskip("src.apps.assistant_dock")
        src = inspect.getsource(dock._render_note_proposals)
        assert "human_apply_note(" in src and "human_reject_note(" in src
        assert "dispatch(" not in src
        # Статус записи обязан пережить перезапуск приложения (iter73).
        assert "persist_session(" in src

    def test_edited_fields_are_passed_to_apply(self):
        """Кнопка пишет ТО, что человек видел: поля формы уходят в apply."""
        dock = pytest.importorskip("src.apps.assistant_dock")
        src = inspect.getsource(dock._render_note_proposals)
        assert "fields=fields" in src

    def test_reject_requires_reason(self):
        dock = pytest.importorskip("src.apps.assistant_dock")
        src = inspect.getsource(dock._render_note_proposals)
        assert "назовите причину" in src


# ======================================================================
# 8. Помощник знает про панель, а MCP-сервер её не подменяет
# ======================================================================
class TestPromptAndMcp:

    def test_prompt_names_panel_and_buttons_that_exist(self):
        import os

        from src.assistant.prompts import LIMITS_BLOCK, UI_BLOCK
        from src.assistant.session import NOTE_BUTTON
        repo = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        code = open(os.path.join(repo, "src", "apps", "assistant_dock.py"),
                    encoding="utf-8").read()
        assert "📝 Предложенные записи в журналы" in UI_BLOCK
        assert "📝 Предложенные записи в журналы" in code
        for label in NOTE_BUTTON.values():
            assert label in UI_BLOCK, f"промпт не знает «{label}»"
        # Правило выбора инструмента: не пересказывать формулировку в ответе.
        assert "propose_decision" in LIMITS_BLOCK
        assert "propose_fact" in LIMITS_BLOCK

    def test_ui_guide_mentions_the_panel(self):
        from src.apps import assistant as ai
        right = ai.campaign_ui_guide()["layout"]["правая зона"]
        assert "📝 Предложенные записи в журналы" in right

    def test_mcp_server_does_not_export_the_new_tools(self):
        """Предложение должно попасть в панель с кнопками, а не в чужую сессию."""
        from src.mcp import campaign_tools as ct
        hidden = set(ct.hidden_names())
        assert {"propose_decision", "propose_fact", "apply_note",
                "reject_note"} <= hidden
        assert not ({"propose_decision", "apply_note"}
                    & set(ct.exported_names()))
        refused = ct._refuse_hidden("apply_note")
        assert "write" in refused and "человек" in refused.lower()


# ======================================================================
# 9. ЖИВОЙ рендер панели: поля правятся, правка доходит до журнала
# ======================================================================
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

#: Мини-приложение: только панель предложенных записей. Полный `main()` здесь
#: не нужен и стоил бы минуты прогона — проверяется контракт ОДНОЙ панели.
_PROBE_APP = '''
import streamlit as st

from src.apps import assistant_dock as dock
from src.assistant import store
from src.assistant.consent import ConsentRegistry
from src.assistant.tools import ToolContext

ROOT = r"{root}"
PROJECT = "{project}"

# Сессия читается С ДИСКА на каждом прогоне — как в приложении: id записи
# стабилен, а статус после фиксации переживает перезапуск скрипта.
session = store.load_session(ROOT, PROJECT)
if "probe_consent" not in st.session_state:
    st.session_state["probe_consent"] = ConsentRegistry()
ctx = ToolContext(session=session, root=ROOT, project=PROJECT,
                  extra={{"consent": st.session_state["probe_consent"]}})
dock._render_note_proposals(ctx, session)
'''


def _probe(tmp_path, note: StagedNote) -> AppTest:
    session = new_session(PROJECT)
    session.stage_note(note)
    store.save_session(session, str(tmp_path), PROJECT)
    src = _PROBE_APP.format(root=str(tmp_path), project=PROJECT)
    return AppTest.from_string(src, default_timeout=60).run()


def test_live_panel_writes_the_edited_text(tmp_path):
    """Живой прогон: технолог поправил формулировку и нажал кнопку.

    Проверяется то, чего исходник не докажет: виджеты полей действительно
    отрисованы значениями предложения, их значение МЕНЯЕТСЯ, и в журнал
    проекта уходит именно правка человека.
    """
    at = _probe(tmp_path, StagedNote(
        kind=NOTE_FACT, level="L1", label="смеситель",
        fields={"statement": "смеситель типа A — не выше 120 °C",
                "scope": "смеситель", "source": "слова технолога",
                "author": ""}))
    assert not at.exception

    stmt = at.text_area[0]
    assert stmt.value == "смеситель типа A — не выше 120 °C"
    stmt.set_value("смеситель типа A — не выше 115 °C (по факту КИП)")
    [i for i in at.text_input if "Кто утверждает" in i.label][0] \
        .set_value("Жихарев")
    [b for b in at.button if "Зафиксировать факт" in b.label][0].click().run()
    assert not at.exception

    recs = store.read_log(str(tmp_path), PROJECT, "local_facts")
    assert len(recs) == 1
    assert recs[0]["statement"] == "смеситель типа A — не выше 115 °C (по факту КИП)"
    assert recs[0]["author"] == "человек (UI)"
    assert recs[0]["edited_by_human"] is True
    # Панель опустела, а статус записи сохранён на диск (iter73).
    assert store.load_session(str(tmp_path), PROJECT).staged_notes() == []


def test_live_panel_refuses_reject_without_reason(tmp_path):
    """Отказ без причины не проходит: он тоже идёт в журнал решений."""
    at = _probe(tmp_path, StagedNote(
        kind=NOTE_DECISION,
        fields={"title": "мел до 100 phr", "rationale": "опыт цеха",
                "nodes": [], "author": ""}))
    [b for b in at.button if "Отклонить запись" in b.label][0].click().run()
    assert not at.exception
    assert any("причину" in e.value for e in at.error)
    # Запись осталась в панели, журнал решений пуст.
    assert len(store.load_session(str(tmp_path), PROJECT).staged_notes()) == 1
    assert store.read_log(str(tmp_path), PROJECT, "decisions") == []
