# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 78 — подсказки ПЕРВОГО ВХОДА (шаг не определён).

Замечание пользователя 12.08.2026: на только что открытом приложении под
подписью «Спросить про открытую закладку» стояли кнопки «Можно строить план?»
и «Что в базе?». Закладки в фокусе ещё нет, кампании тоже — оба вопроса дают
формально верный, но бесполезный ответ («не проверено», «база пуста»), а
подпись обещает контекст, которого не существует.

Здесь фиксируется поведение первого входа:

  1. у `UNKNOWN_SECTION` ровно две подсказки — «С чего начать» и
     «Краткое описание функционала»;
  2. обе МАРШРУТИЗИРУЮТСЯ (не в `clarify`) и ведут к состоянию проекта и
     полям формы сетапа: «кнопка есть, а инструмента для неё нет» запрещено
     контрактом подсказок (ASSISTANT_SPEC §461);
  3. подсказки ОПРЕДЕЛЁННЫХ шагов не тронуты — правка касается только места
     «шаг не определён»;
  4. язык подсказок остаётся операционным (без внутреннего сленга).
"""
import pytest

from src.assistant import context as actx
from src.assistant.context import UNKNOWN_SECTION, UiFocus, suggested_questions
from src.assistant.prompts import (KIND_CLARIFY, KIND_STATUS, route, scenario)


# ======================================================================
# 1. Карта места «шаг не определён»
# ======================================================================
def test_unknown_section_asks_are_onboarding_questions():
    """Первый вход спрашивает «с чего начать», а не про гейты плана."""
    labels = [label for label, _ in UNKNOWN_SECTION.asks]
    assert labels == ["С чего начать", "Краткое описание функционала"]


def test_unknown_section_dropped_campaign_state_questions():
    """Прежние подсказки ушли: на пустом проекте они бесполезны."""
    questions = " ".join(q for _, q in UNKNOWN_SECTION.asks).lower()
    assert "можно уже строить план" not in questions
    assert "сколько точек в базе" not in questions


def test_onboarding_asks_need_no_node_or_branch():
    """Ни узла, ни ветки на первом входе нет — подсказка не может их требовать."""
    for _, tmpl in UNKNOWN_SECTION.asks:
        assert "{node}" not in tmpl and "{branch}" not in tmpl


# ======================================================================
# 2. Маршрут: кнопка ведёт к инструментам, а не в «уточните»
# ======================================================================
@pytest.mark.parametrize("question", [q for _, q in UNKNOWN_SECTION.asks],
                         ids=[lbl for lbl, _ in UNKNOWN_SECTION.asks])
def test_onboarding_question_routes_somewhere(question):
    r = route(question)
    assert r.kind != KIND_CLARIFY, question
    assert r.kind == KIND_STATUS
    assert r.tools == ("campaign_overview", "get_setup_fields")


def test_onboarding_route_reads_setup_fields():
    """До сборки проекта всё введённое живёт в ПОЛЯХ формы — их надо прочитать."""
    r = route("С чего начать работу в этом проекте?")
    assert "get_setup_fields" in r.tools


@pytest.mark.parametrize("phrasing", [
    "с чего начать",
    "С чего мне начать?",
    "Что умеет эта программа?",
    "Краткое описание функционала",
    "Какой следующий шаг?",
    "Как здесь работать?",
])
def test_onboarding_phrasings_are_recognised(phrasing):
    """Человек спрашивает разными словами — все они про первый вход."""
    assert route(phrasing).kind == KIND_STATUS


def test_onboarding_rule_does_not_shadow_plan_status():
    """Вопрос о проверке плана остаётся при своих инструментах."""
    r = route("Можно уже строить план? Что показала проверка плана?")
    assert r.tools == ("preflight", "campaign_overview")


def test_onboarding_rule_does_not_shadow_explain():
    """«Почему диапазон не такой» — по-прежнему объяснение геометрии."""
    assert route("Почему диапазон DINP не такой, как я вводил?").tools \
        == ("explain_node",)


def test_onboarding_key_has_no_golden_scenario():
    """Новое правило роутера — не golden-сценарий §8: каталог §8 не менялся."""
    assert route("С чего начать?").scenario == ""
    with pytest.raises(KeyError):
        scenario("onboarding")


# ======================================================================
# 3. Подсказки дока на первом входе
# ======================================================================
def test_suggestions_on_empty_focus_are_enabled():
    """Кнопки первого входа активны и без собранного проекта, и без узла."""
    sugs = suggested_questions(UiFocus(), has_runner=False)
    assert [s.label for s in sugs] == ["С чего начать",
                                       "Краткое описание функционала"]
    assert all(s.enabled for s in sugs)
    assert all(s.tools for s in sugs)


def test_suggestions_on_empty_focus_never_clarify():
    for s in suggested_questions(UiFocus()):
        assert s.kind != KIND_CLARIFY, s.question


def test_known_sections_keep_their_own_asks():
    """Правка касается ТОЛЬКО места «шаг не определён»."""
    seed = actx.section("seed")
    assert ("Можно строить план?",
            "Можно уже строить план? Что показала проверка плана?") in seed.asks
    base = actx.section("base")
    assert any(label == "Что в базе?" for label, _ in base.asks)


def test_focus_sections_still_route_without_clarify():
    """Общий контракт iter65 не нарушен ни на одном шаге."""
    for sec in actx.FOCUS_SECTIONS:
        f = UiFocus(section_key=sec.key, node="DINP", branch="b1")
        sugs = suggested_questions(f)
        assert sugs, sec.key
        for s in sugs:
            assert s.kind != KIND_CLARIFY, (sec.key, s.question)


# ======================================================================
# 4. Язык подсказок (iter74: без внутреннего сленга)
# ======================================================================
def test_onboarding_wording_has_no_slang():
    blob = (UNKNOWN_SECTION.title + " " + UNKNOWN_SECTION.doing + " "
            + " ".join(f"{lbl} {q}" for lbl, q in UNKNOWN_SECTION.asks)).lower()
    for bad in ("seed", "preflight", "undo", "spawn", "стейдж", "оракул",
                "суррогат"):
        assert bad not in blob, f"«{bad}» в подсказках первого входа"
