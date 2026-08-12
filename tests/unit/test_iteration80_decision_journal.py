# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 80 — журнал решений и факты цеха ВИДНЫ и ЗАПОЛНЯЮТСЯ из интерфейса.

Замечание пользователя 12.08.2026: «не вижу, куда должны писаться принятые
решения». Разбор показал, что запись работала, а показа и ручного ввода не было:

  1. `decision_log.jsonl` заполнялся ТОЛЬКО побочно — кнопками утверждения
     пакетов и патчей (проверено на живом проекте: `apply_project` +
     `reject_project` от «человек (UI)»). Панели на скриншоте («Предложенные
     проекты/спеки/патчи») показывают СТЕЙДЖ, и после применения они пустеют
     штатно — журнала там нет вовсе;
  2. `views.decisions_dataframe` существовал и был покрыт тестом, но вызывался
     только из демо-скрипта: ни один модуль `src/apps/*` журнал не читал;
  3. `record_decision` / `add_local_fact` (класс `write`, ASSISTANT_SPEC §370 —
     «L1-факты и ADR пишет ЧЕЛОВЕК») не имели в интерфейсе ни кнопки, ни пути:
     `issue_decision_token` / `issue_fact_token` не вызывались нигде, кроме
     тестов. Технолог не мог записать ни решения, ни факта цеха.

Здесь фиксируется:

  * показ обоих журналов (таблицы + подписи), включая РАЗЛИЧЕНИЕ применения и
    отказа: «проект принят» и «правка границы отклонена» — разные события;
  * ручная запись через человеческий путь `context.human_*` (разовый токен
    внутри), запись ложится на диск и читается инструментами модели;
  * инвариант iter63 НЕ ослаблен: модели `record_decision` / `add_local_fact`
    по-прежнему недоступны, а без токена запись не проходит;
  * оба журнала живут в ПРАВОЙ инфо-панели (iter72), а не в колонке диалога.
"""
import inspect

import pytest

from src.apps import assistant_dock as dock
from src.assistant import context as actx
from src.assistant import store, views
from src.assistant.consent import ConsentRegistry
from src.assistant.session import new_session
from src.assistant.tools import (AGENT_KINDS, WRITE, ToolContext, ToolError,
                                 dispatch, tool_names)
from src.design.phr_sampler import PhrSpec

PROJECT = "кромка ПВХ"

SPEC_NODES = [
    {"name": "RESIN", "role": "FIXED", "value": 100.0},
    {"name": "DINP", "role": "ABSOLUTE", "range": [4.0, 14.0]},
]


def _spec() -> PhrSpec:
    return PhrSpec.from_dicts(SPEC_NODES)


def _ctx(tmp_path, *, spec=True) -> ToolContext:
    return ToolContext(spec=_spec() if spec else None,
                       session=new_session(PROJECT), root=str(tmp_path),
                       project=PROJECT, extra={"consent": ConsentRegistry()})


# ======================================================================
# 1. Показ журнала решений: применение и отказ различимы
# ======================================================================
class TestDecisionsView:

    RECORDS = [
        {"ts": "2026-08-12T14:43:09+03:00", "kind": "reject_project",
         "title": "ОТКЛОНЕНО: пакет проекта кромка ПВХ",
         "author": "человек (UI)", "spec_hash": "",
         "rationale": "ждём границы процесс-осей"},
        {"ts": "2026-08-12T14:46:51+03:00", "kind": "apply_project",
         "title": "кромка ПВХ pvc_edge_v1: первичный ввод проекта",
         "author": "человек (UI)", "nodes": ["FILLER", "DINP"],
         "spec_hash_after": "69b1afd27ee3248f6", "rationale": "по паспортам"},
        {"ts": "2026-08-12T15:02:00+03:00", "kind": "decision",
         "title": "мел до 100 phr", "author": "технолог",
         "nodes": ["FILLER.total"], "spec_hash": "c63b7e1696e1c449",
         "rationale": "L1-факт цеха: белые компаунды"},
    ]

    def test_kind_column_separates_apply_from_reject(self):
        """Принятие и отказ — РАЗНЫЕ события, а не однородные записи.

        Без этого столбца таблица журнала выглядела одинаково для «проект
        принят» и «проект отклонён»: различие пряталось в слове «ОТКЛОНЕНО»
        внутри заголовка.
        """
        df = views.decisions_dataframe(self.RECORDS)
        assert list(df["вид"]) == ["проект отклонён", "проект принят",
                                   "решение человека"]
        assert df.iloc[2]["узлы"] == "FILLER.total"
        assert df.iloc[2]["spec_hash"] == "c63b7e1696e1"

    def test_unknown_kind_is_shown_as_is(self):
        """Незнакомый вид записи не сглаживается общим словом.

        Новый вид появится в журнале раньше, чем в словаре подписей; человек
        должен увидеть расхождение, а не правдоподобное «решение».
        """
        assert views.decision_kind_label("apply_branch_goal") == \
            "apply_branch_goal"
        assert views.decision_kind_label("") == "решение"
        assert views.decision_kind_label("apply_patch") == "граница изменена"

    def test_caption_counts_apply_and_reject_separately(self):
        txt = views.decisions_caption(self.RECORDS)
        assert "всего: 3" in txt and "принято: 1" in txt
        assert "отклонено: 1" in txt and "записано вручную: 1" in txt
        assert "последняя: 2026-08-12" in txt

    def test_empty_journal_says_so(self):
        assert views.decisions_caption([]) == "записей нет"
        assert views.decisions_dataframe([]).empty
        # столбцы существуют и у пустой таблицы — UI рисует её без ветвления
        assert "вид" in views.decisions_dataframe([]).columns


# ======================================================================
# 2. Показ фактов цеха: уровень знания и источник
# ======================================================================
class TestFactsView:

    FACTS = [
        {"ts": "2026-08-12T15:10:00+03:00", "scope": "смеситель",
         "statement": "смеситель типа A — не выше 120 °C", "author": "технолог",
         "source": "SOP цеха", "level": "L1"},
        {"ts": "2026-08-12T15:12:00+03:00", "scope": "cost",
         "statement": "плотность компаунда ИЗМЕРЯЕТСЯ", "author": "технолог",
         "source": "", "level": "L1"},
    ]

    def test_level_and_source_are_columns(self):
        """L1 отменяет литературу — уровень виден, а не подразумевается."""
        df = views.local_facts_dataframe(self.FACTS)
        assert list(df["уровень"]) == ["L1", "L1"]
        assert df.iloc[0]["откуда"] == "SOP цеха"
        assert df.iloc[1]["откуда"] == "—"          # источник не назван
        assert df.iloc[0]["область"] == "смеситель"

    def test_caption_lists_scopes(self):
        txt = views.facts_caption(self.FACTS)
        assert "всего: 2" in txt and "смеситель" in txt and "cost" in txt

    def test_empty_facts_say_so(self):
        assert views.facts_caption([]) == "фактов нет"
        assert "факт" in views.local_facts_dataframe([]).columns


# ======================================================================
# 3. Ручная запись человеком: решение и факт доходят до диска
# ======================================================================
class TestHumanWrites:

    def test_decision_written_by_human_lands_on_disk(self, tmp_path):
        ctx = _ctx(tmp_path)
        out = actx.human_record_decision(
            ctx, "мел до 100 phr", "L1-факт цеха: белые компаунды",
            nodes=["DINP"], author="Жихарев")
        assert out["ok"] and out["decision"]["persisted"]
        rec = store.read_log(str(tmp_path), PROJECT, "decisions")[-1]
        assert rec["title"] == "мел до 100 phr"
        assert rec["author"] == "Жихарев" and rec["kind"] == "decision"
        # отпечаток геометрии в записи обязателен: иначе решение не сопоставить
        assert rec["spec_hash"] == _spec().spec_hash()

    def test_fact_written_by_human_is_l1_and_readable_by_model(self, tmp_path):
        ctx = _ctx(tmp_path)
        out = actx.human_add_local_fact(
            ctx, "смеситель типа A — не выше 120 °C", scope="смеситель",
            source="SOP цеха", author="Жихарев")
        assert out["ok"] and out["fact"]["persisted"]
        rec = store.read_log(str(tmp_path), PROJECT, "local_facts")[-1]
        assert rec["level"] == "L1" and rec["author"] == "Жихарев"
        # модель ЧИТАЕТ факт (readonly) — иначе запись бесполезна для диалога
        got = dispatch(ctx, "get_local_facts", {"scope": "смеситель"})
        assert got["n"] == 1

    def test_written_decision_is_visible_in_the_table(self, tmp_path):
        """Путь целиком: кнопка человека → журнал → таблица показа."""
        ctx = _ctx(tmp_path)
        actx.human_record_decision(ctx, "dT_head = −35 K подтверждён КИП",
                                   "замер на линии", author="технолог")
        recs = store.read_log(str(tmp_path), PROJECT, "decisions")
        df = views.decisions_dataframe(recs)
        assert df.iloc[-1]["решение"].startswith("dT_head")
        assert df.iloc[-1]["вид"] == "решение человека"

    def test_decision_before_spec_is_allowed_without_hash(self, tmp_path):
        """Решение может быть принято ДО сборки проекта — но без отпечатка.

        «Не проверено» не выдаётся за «пройдено» (ASSISTANT_SPEC §361): пустой
        `spec_hash` — честный признак, что геометрии на тот момент не было.
        """
        ctx = _ctx(tmp_path, spec=False)
        out = actx.human_record_decision(ctx, "берём кромку, а не профиль",
                                         "решение по продукту")
        assert out["ok"]
        rec = store.read_log(str(tmp_path), PROJECT, "decisions")[-1]
        assert rec["spec_hash"] == ""


# ======================================================================
# 4. Инвариант iter63 не ослаблен: пишет ЧЕЛОВЕК, и только с токеном
# ======================================================================
class TestWriteBoundaryHeld:

    def test_model_still_cannot_record_decisions_or_facts(self):
        agent = set(tool_names(AGENT_KINDS))
        assert not ({"record_decision", "add_local_fact"} & agent)

    def test_without_token_nothing_is_written(self, tmp_path):
        ctx = _ctx(tmp_path)
        with pytest.raises(ToolError):
            dispatch(ctx, "record_decision",
                     {"title": "t", "rationale": "r", "human_token": ""},
                     allowed_kinds=[WRITE])
        with pytest.raises(ToolError):
            dispatch(ctx, "add_local_fact",
                     {"statement": "s", "human_token": "нет-такого"},
                     allowed_kinds=[WRITE])
        assert store.read_log(str(tmp_path), PROJECT, "decisions") == []
        assert store.read_log(str(tmp_path), PROJECT, "local_facts") == []

    def test_token_is_single_use(self, tmp_path):
        """Одно нажатие — одна запись: повтор тем же токеном не проходит."""
        from src.assistant.tools.write import issue_decision_token
        ctx = _ctx(tmp_path)
        token = issue_decision_token(ctx, "мел до 100 phr")
        args = {"title": "мел до 100 phr", "rationale": "r",
                "human_token": token}
        assert dispatch(ctx, "record_decision", dict(args),
                        allowed_kinds=[WRITE])["ok"]
        with pytest.raises(ToolError):
            dispatch(ctx, "record_decision", dict(args), allowed_kinds=[WRITE])
        assert len(store.read_log(str(tmp_path), PROJECT, "decisions")) == 1

    def test_no_project_means_not_silently_lost(self):
        """Без проекта запись помечается несохранённой, а не исчезает."""
        ctx = ToolContext(spec=_spec(), session=new_session(PROJECT),
                          extra={"consent": ConsentRegistry()})
        out = actx.human_record_decision(ctx, "решение без проекта", "r")
        assert out["decision"]["persisted"] is False
        assert out["decision"]["note"]


# ======================================================================
# 5. Место в интерфейсе: журналы — в ПРАВОЙ инфо-панели (iter72)
# ======================================================================
class TestPanelPlacement:

    def test_journals_are_rendered_in_info_column(self):
        """Журналы нужны на любой закладке и не прячутся под диалогом."""
        info_src = inspect.getsource(dock.render_assistant_info)
        dock_src = inspect.getsource(dock.render_assistant_dock)
        for marker in ("_render_decisions(", "_render_local_facts("):
            assert marker in info_src, f"«{marker}» не попал в инфо-панель"
            assert marker not in dock_src, f"«{marker}» в колонке диалога"

    def test_panels_read_the_journal_from_disk(self):
        """Панели читают ЖУРНАЛ (файл проекта), а не стейдж сессии.

        Ровно это и было причиной вопроса: панели утверждения показывают
        стейдж и после применения пустеют, а журнал живёт в файле.
        """
        src = inspect.getsource(dock._render_decisions)
        assert "read_log(" in src and '"decisions"' in src
        facts = inspect.getsource(dock._render_local_facts)
        assert "read_log(" in facts and '"local_facts"' in facts

    def test_manual_write_goes_through_human_path(self):
        """Кнопки зовут context.human_* — единственный путь к классу write."""
        src = inspect.getsource(dock._render_decisions) + \
            inspect.getsource(dock._render_local_facts)
        assert "human_record_decision(" in src
        assert "human_add_local_fact(" in src
        # прямого вызова инструмента в обход человеческого пути быть не должно
        assert "dispatch(" not in src


# ======================================================================
# 6. Помощник ЗНАЕТ про новые панели (иначе отправит человека не туда)
# ======================================================================
class TestAssistantKnowsTheScreen:

    def test_prompt_names_both_journals_and_their_buttons(self):
        from src.assistant.prompts import UI_BLOCK
        assert "📚 Журнал решений" in UI_BLOCK
        assert "🏭 Факты производства (L1)" in UI_BLOCK
        assert "✍️ Записать решение в журнал" in UI_BLOCK
        assert "✍️ Записать факт в журнал" in UI_BLOCK

    def test_prompt_separates_staging_panels_from_the_journal(self):
        """Именно эта путаница и породила вопрос «куда пишутся решения».

        Панель предложений после применения пустеет штатно, журнал — история в
        файле. Помощник обязан различать их словами, а не отправлять человека
        искать принятое решение в панели утверждения.
        """
        from src.assistant.prompts import UI_BLOCK
        assert "ЖДЁТ утверждения" in UI_BLOCK and "ИСТОРИЯ" in UI_BLOCK

    def test_ui_guide_lists_journals_in_the_right_zone(self):
        from src.apps import assistant as ai
        right = ai.campaign_ui_guide()["layout"]["правая зона"]
        assert "📚 Журнал решений" in right
        assert "🏭 Факты производства (L1)" in right

    def test_buttons_named_in_prompt_exist_in_the_code(self):
        """Кнопка, названная промптом, должна существовать на экране.

        Контракт iter74: человек ищет подпись глазами, и выдуманная кнопка
        ломает доверие к помощнику сильнее, чем отказ ответить.
        """
        import os
        from src.assistant.prompts import UI_BLOCK
        repo = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        src = open(os.path.join(repo, "src", "apps", "assistant_dock.py"),
                   encoding="utf-8").read()
        for label in ("✍️ Записать решение в журнал",
                      "✍️ Записать факт в журнал",
                      "📚 Журнал решений", "🏭 Факты производства (L1)"):
            assert label in UI_BLOCK, f"промпт не знает «{label}»"
            assert label in src, f"«{label}» нет в интерфейсе"
