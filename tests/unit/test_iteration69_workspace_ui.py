# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 69 — рабочая область на закладках + лента диалога как в Cline.

Багрепорт пользователя: экран проекта был ОДНОЙ простынёй (сетап → seed → база →
ветки → стол → скрининг → эволюция) плюс док ассистента справа, и оба делили
один скролл документа. Любой ответ ассистента уводил страницу целиком: чтобы
дописать сообщение, приходилось искать поле ввода, а чтобы вернуться к таблице —
скроллить назад.

Решение (эскиз пользователя): слева панель диалога (лента скроллится внутри
себя, ввод закреплён под лентой), справа рабочая область с ДВУМЯ рядами
закладок — закладки рабочей области и закладки веток проекта. Рабочая область
живёт в контейнере фиксированной высоты со своим скроллом и от чата не ползёт.

Две части (как у существующих ``*_ui`` тестов):

* ЧИСТАЯ логика раскладки (:mod:`src.apps.workspace`, без Streamlit): гейты
  доступности закладок, дефолт по состоянию проекта, устойчивость выбора,
  соответствие закладок карте мест ассистента (``ui_focus``), порядок ленты;
* headless AppTest: приложение рендерится новой раскладкой, старая простыня не
  возвращается, а полный ручной поток (сетап → seed → ветка → стол) проходится
  ЧЕРЕЗ закладки.
"""
import os
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import pytest
from sklearn.exceptions import ConvergenceWarning

from src.apps import workspace as ws
from src.assistant.context import SECTIONS_BY_KEY, focus_from_state

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ======================================================================
# 1. Закладки рабочей области: гейты, дефолт, устойчивость выбора
# ======================================================================
def test_tabs_are_visible_but_disabled_without_project():
    """Пустая сессия: закладки НЕ исчезают — они выключены с причиной (A0.6)."""
    states = ws.tab_states(has_project=False)
    assert [s.key for s in states] == [t.key for t in ws.WORKSPACE_TABS]
    off = {s.key: s.why for s in states if not s.enabled}
    assert "branches" in off
    assert all(w for w in off.values()), "у выключенной закладки нет причины"
    # iter72: «Старт» доступен всегда — там сетап и загрузка проекта (вход);
    # «Обзор» тоже не требует состояния проекта.
    assert ws.enabled_keys(has_project=False) == ["start", "overview"]


def test_points_gate_opens_base_branches_analysis_and_schema():
    """Измеренный seed открывает всё, что читает базу; до него — только «Старт»."""
    assert ws.enabled_keys(has_project=True, n_points=0) == ["start", "overview"]
    full = ws.enabled_keys(has_project=True, n_points=10)
    assert full == ["start", "base", "branches", "screening", "evolution",
                    "overview"]


def test_gate_reason_distinguishes_no_project_from_no_points():
    """Причина отказа конкретна: «нет проекта» и «seed не измерен» — разные."""
    no_proj = {s.key: s.why for s in ws.tab_states(has_project=False)}
    no_pts = {s.key: s.why for s in ws.tab_states(has_project=True,
                                                  n_points=0)}
    assert "проект не собран" in no_proj["base"]
    assert "seed" in no_pts["base"].lower()


def test_default_tab_follows_project_state():
    # iter72: пустая сессия открывается на «Старте» — там собирают/загружают
    # проект (раньше форма жила в сайдбаре и дефолтом был «Обзор»)
    assert ws.default_tab_key(has_project=False) == "start"
    assert ws.default_tab_key(has_project=True, n_points=0) == "start"
    # seed измерен → человека интересует работа с ветками, а не сделанный seed
    assert ws.default_tab_key(has_project=True, n_points=8) == "branches"
    assert ws.default_tab_key(has_project=True, n_points=8,
                              n_branches=2) == "branches"


def test_resolve_tab_keeps_human_choice_while_it_exists():
    key, why = ws.resolve_tab("screening", has_project=True, n_points=5)
    assert key == "screening" and why == ""


def test_resolve_tab_explains_forced_switch():
    """Закладка стала недоступна → переход на дефолт с ЯВНОЙ причиной."""
    key, why = ws.resolve_tab("base", has_project=True, n_points=0)
    assert key == "start"
    assert why and "недоступна" in why and "seed" in why.lower()


def test_resolve_tab_survives_unknown_key():
    """Незнакомый ключ (старое состояние сессии) не роняет раскладку."""
    key, why = ws.resolve_tab("legacy_tab", has_project=True, n_points=5)
    assert key == "branches" and "больше нет" in why
    # пустой запрос — просто дефолт, без шума
    assert ws.resolve_tab("", has_project=True, n_points=5) == ("branches", "")


# ======================================================================
# 2. Автопродвижение по фазам проекта (иначе человек застревает)
# ======================================================================
def test_phase_key_tracks_project_lifecycle():
    assert ws.phase_key(has_project=False) == ws.PHASE_EMPTY
    assert ws.phase_key(has_project=True, n_points=0) == ws.PHASE_SETUP
    assert ws.phase_key(has_project=True, n_points=9) == ws.PHASE_MEASURED
    assert ws.phase_key(has_project=True, n_points=9,
                        n_branches=1) == ws.PHASE_BRANCHED


def test_decide_tab_moves_forward_when_phase_changes():
    """Собрали проект, стоя на «Обзоре» → открывается «Старт», и это сказано."""
    d = ws.decide_tab("overview", prev_phase=ws.PHASE_EMPTY, has_project=True,
                      n_points=0)
    assert d.key == "start" and d.moved
    assert "проект собран" in d.notice and "Старт" in d.notice
    # зафиксировали seed → рабочая область сама открывает ветки
    d2 = ws.decide_tab("start", prev_phase=ws.PHASE_SETUP, has_project=True,
                       n_points=10)
    assert d2.key == "branches" and "измерен" in d2.notice


def test_decide_tab_keeps_choice_inside_one_phase():
    """Фаза не менялась → выбор человека не трогаем и молчим."""
    d = ws.decide_tab("screening", prev_phase=ws.PHASE_BRANCHED,
                      has_project=True, n_points=10, n_branches=2)
    assert d.key == "screening" and d.notice == "" and not d.moved


def test_decide_tab_first_run_is_silent():
    """Первый прогон (фазы ещё не было) — просто дефолт, без уведомления."""
    d = ws.decide_tab(None, prev_phase=None, has_project=False)
    # iter72: дефолт пустой сессии — «Старт» (вход в проект), не «Обзор»
    assert d.key == "start" and d.notice == ""


# ======================================================================
# 3. Связь с картой мест ассистента (ui_focus, iter65)
# ======================================================================
def test_every_tab_focus_key_is_a_known_assistant_section():
    """Раскладка и карта мест ассистента не должны разъезжаться."""
    for tab in ws.WORKSPACE_TABS:
        if tab.focus:
            assert tab.focus in SECTIONS_BY_KEY, tab.key


def test_focus_section_for_maps_tabs_and_ignores_unknown():
    assert ws.focus_section_for("start") == "seed"
    assert ws.focus_section_for("branches") == "branch"
    assert ws.focus_section_for("overview") == ""     # обзор — не шаг потока
    assert ws.focus_section_for("нет такой") == ""


def test_published_focus_of_active_tab_is_read_by_assistant():
    """Активная закладка + ветка = фокус хода (чистая проводка, без Streamlit)."""
    state = {"ui_focus": {"section": ws.focus_section_for("branches"),
                          "branch": "premium"}}
    f = focus_from_state(state)
    assert f.section_key == "branch" and f.branch == "premium"
    assert "Ветки" in f.title


# ======================================================================
# 4. Второй ряд: закладки веток
# ======================================================================
@dataclass
class _Br:
    name: str


def test_branch_labels_show_name_and_keep_id():
    labels = ws.branch_labels({"b1": _Br("premium"), "b2": _Br("rho_focus")})
    assert labels == {"b1": "premium (b1)", "b2": "rho_focus (b2)"}
    # имя совпало с id → не дублируем его в скобках
    assert ws.branch_labels({"premium": _Br("premium")}) == {
        "premium": "premium"}
    assert ws.branch_labels(None) == {}


def test_resolve_branch_falls_back_to_first_existing():
    assert ws.resolve_branch("b2", ["b1", "b2"]) == "b2"
    assert ws.resolve_branch("gone", ["b1", "b2"]) == "b1"   # ветку удалили
    assert ws.resolve_branch("b1", []) == ""                 # веток нет вовсе


# ======================================================================
# 5. Лента диалога: порядок «старые сверху», урезание с оговоркой
# ======================================================================
@dataclass
class _Msg:
    role: str
    content: str
    images: Tuple[str, ...] = ()


def _msgs(n: int) -> List[_Msg]:
    out: List[_Msg] = []
    for i in range(n):
        out.append(_Msg("user", f"вопрос {i}"))
        out.append(_Msg("assistant", f"ответ {i}"))
    return out


def test_feed_keeps_chronological_order_newest_last():
    """Как в Cline: история выше, свежая реплика — внизу ленты."""
    items = ws.feed_items(_msgs(3))
    assert [i.content for i in items][:2] == ["вопрос 0", "ответ 0"]
    assert items[-1].content == "ответ 2"


def test_feed_drops_service_roles_but_keeps_images():
    msgs = [_Msg("system", "[фокус] ..."), _Msg("tool", "explain_node"),
            _Msg("user", "смотри скриншот", ("sha1",)),
            _Msg("assistant", "вижу")]
    items = ws.feed_items(msgs)
    assert [i.role for i in items] == ["user", "assistant"]
    assert items[0].images == ("sha1",)


def test_feed_limit_takes_the_tail():
    items = ws.feed_items(_msgs(30), limit=4)
    assert len(items) == 4
    assert items[-1].content == "ответ 29"       # хвост, а не голова


def test_feed_hint_admits_truncation():
    assert "прокручивается вверх" in ws.feed_hint(6, 6)
    hint = ws.feed_hint(4, 60)
    assert "последние 4 из 60" in hint and "сохранена" in hint


def test_panel_heights_are_fixed_and_independent():
    """Фиксированные высоты — суть решения: без них общий скролл возвращается."""
    assert isinstance(ws.WORKSPACE_HEIGHT, int) and ws.WORKSPACE_HEIGHT > 400
    assert isinstance(ws.CHAT_FEED_HEIGHT, int) and ws.CHAT_FEED_HEIGHT > 300
    # лента ниже рабочей области: под ней ещё стоит поле ввода
    assert ws.CHAT_FEED_HEIGHT < ws.WORKSPACE_HEIGHT


def test_dialog_count_ignores_service_messages():
    """Подпись «показаны последние N из M» считает ТОЛЬКО реплики диалога."""
    msgs = _msgs(2) + [_Msg("system", "[фокус] ..."), _Msg("tool", "{}")]
    assert ws.dialog_count(msgs) == 4
    assert ws.dialog_count(None) == 0


# ======================================================================
# 6. headless AppTest — новая раскладка живого приложения
# ======================================================================
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(_REPO, "src", "apps", "streamlit_app.py")


def _click(at, key):
    b = [w for w in at.button if w.key == key]
    assert b, f"кнопка {key} не найдена"
    b[0].click().run()


def _focus(at):
    """Фокус ассистента из состояния приложения.

    В AppTest ``session_state`` — ``SafeSessionState`` без ``keys()``, поэтому
    приводить его к dict напрямую нельзя (в живом Streamlit это прокси с
    ``keys()``, и ``focus_from_state(st.session_state)`` работает как есть).
    Берём готовый снимок ``filtered_state``.
    """
    return focus_from_state(at.session_state.filtered_state)


def test_app_renders_tab_row_and_chat_input_together():
    """Пустая сессия: ряд закладок нарисован, поле ввода чата — тоже.

    Ключевая проверка раскладки: диалог и рабочая область существуют
    ОДНОВРЕМЕННО (не вкладками верхнего уровня, как было до iter69), поэтому
    ответ ассистента больше не уводит человека с рабочего экрана.
    """
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception
    keys = {w.key for w in at.button}
    # ряд закладок рабочей области (по кнопке на закладку)
    assert "ws_tab_overview" in keys
    # iter72: «Старт» доступен и БЕЗ проекта (там сетап/загрузка — вход);
    # закладки, требующие точек, не нарисованы, но ЯВНО перечислены как
    # недоступные (см. подпись «Пока недоступно: …»)
    assert "ws_tab_start" in keys
    assert "ws_tab_base" not in keys
    assert any("Пока недоступно" in str(c.value) for c in at.caption)
    # поле ввода диалога — на той же странице, что и рабочая область
    assert at.chat_input, "поле ввода диалога не нарисовано"


def test_dialog_panel_has_feed_hint_and_its_own_input():
    """Панель диалога: лента с подписью + СВОЁ поле ввода под ней (как в Cline).

    Подпись ленты — признак того, что переписка живёт в отдельном скроллируемом
    контейнере: она печатается сразу под лентой и над полем ввода.
    """
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception
    assert any(w.key == "dock_input" for w in at.chat_input), \
        "поле ввода дока ассистента не найдено"
    caps = [str(c.value) for c in at.caption]
    assert any("прокручивается вверх" in c or "Показаны последние" in c
               for c in caps), "подписи ленты диалога нет"


def test_workspace_tabs_follow_project_and_publish_focus():
    """Демо-проект → закладка веток открывается сама и публикует ui_focus."""
    at = AppTest.from_file(APP, default_timeout=300).run()
    _click(at, "camp_create")
    assert not at.exception
    # автопродвижение: фаза сменилась (появились точки и ветки) → «🌿 Ветки»
    assert at.session_state["ws_tab"] == "branches"
    assert at.session_state["ws_phase"] == ws.PHASE_BRANCHED
    # ui_focus (iter65) указывает на ветку — это и есть «место» пользователя
    focus = _focus(at)
    assert focus.section_key == "branch" and focus.branch
    # второй ряд закладок — ВЕТКИ демо-проекта
    keys = {w.key for w in at.button}
    assert "camp_branch_tab_premium" in keys
    assert "camp_branch_tab_rho_focus" in keys


def test_branch_tab_click_switches_lens():
    """Клик по закладке ветки меняет линзу контекста (camp_branch + ui_focus)."""
    at = AppTest.from_file(APP, default_timeout=300).run()
    _click(at, "camp_create")
    assert at.session_state["camp_branch"] == "premium"
    _click(at, "camp_branch_tab_rho_focus")
    assert not at.exception
    assert at.session_state["camp_branch"] == "rho_focus"
    assert _focus(at).branch == "rho_focus"


def test_workspace_tab_click_switches_panel():
    """Клик по закладке рабочей области переключает панель и не роняет app."""
    at = AppTest.from_file(APP, default_timeout=300).run()
    _click(at, "camp_create")
    _click(at, "ws_tab_base")
    assert not at.exception
    assert at.session_state["ws_tab"] == "base"
    # на закладке базы фокус ассистента — общая база опытов
    assert _focus(at).section_key == "base"
    # и здесь живёт редактор коррекции откликов (§17.2.1), а не на «Старте»
    assert [w for w in at.button if w.key == "camp_correct_save"]


def test_overview_tab_hosts_assistant_overview():
    """«🤖 Обзор» — закладка рабочей области, а не отдельная вкладка сверху."""
    at = AppTest.from_file(APP, default_timeout=300).run()
    _click(at, "camp_create")
    _click(at, "ws_tab_overview")
    assert not at.exception
    assert at.session_state["ws_tab"] == "overview"
    # чат-обзор кампании (ключ ввода прежний — ai_input) доступен здесь
    assert any(w.key == "ai_input" for w in at.chat_input)
