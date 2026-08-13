# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 88 — рабочая область БЕЗ ограничения по высоте (обычная страница).

Запрос технолога (13.08.2026): «не удобно, что рабочая область имеет ограничение
по высоте, приходится её постоянно прокручивать в ограниченном пространстве,
пусть она останется как обычная страница с прокруткой».

Что было. iter69 обёртывал содержимое активной закладки в
``st.container(height=WORKSPACE_HEIGHT)`` (760 px) ради того, чтобы ответ
ассистента в левой колонке не двигал таблицу в центре. Практическая цена
оказалась выше выгоды: с реальным ПВХ-проектом в центре стоят форма сетапа на 18
компонентов, план на 135 опытов и рабочий стол ветки — всё это читалось через
щель 760 px, причём ВНУТРЕННИЙ скролл соседствовал с внешним скроллом страницы.

Что стало. Ограничение снято: ``height`` контейнеру не передаётся, содержимое
растёт по своей длине, прокрутка одна — страницы. Решение вынесено в ЧИСТУЮ
:func:`src.apps.workspace.workspace_box_kwargs` с явным флагом
:data:`src.apps.workspace.WORKSPACE_SCROLL`, чтобы возврат к окну iter69 был
правкой одного значения.

Лента ДИАЛОГА при этом остаётся в контейнере фиксированной высоты: там смысл
обратный — история уходит вверх, а поле ввода не должно уезжать вниз (iter69 §4).
"""
import inspect
import os
import warnings

import pytest
from sklearn.exceptions import ConvergenceWarning

from src.apps import campaign_ui as ui
from src.apps import workspace as ws

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ======================================================================
# 1. Чистый слой: аргументы контейнера рабочей области
# ======================================================================
def test_workspace_scroll_is_off_by_default():
    """Флаг — часть контракта: рабочая область высоту НЕ ограничивает."""
    assert ws.WORKSPACE_SCROLL is False


def test_box_kwargs_have_no_height_when_scroll_is_off():
    """Обычная страница = никакого ``height`` в вызове контейнера."""
    kw = ws.workspace_box_kwargs()
    assert kw == {}
    assert "height" not in kw


def test_box_kwargs_ignore_streamlit_support_when_scroll_is_off():
    """Версия Streamlit на выбор не влияет, пока окно выключено."""
    assert ws.workspace_box_kwargs(supports_height=False) == {}
    assert ws.workspace_box_kwargs(supports_height=True) == {}


def test_box_kwargs_can_restore_iter69_window():
    """Возврат к окну — правка ОДНОГО значения, а не реконструкция кода."""
    saved = ws.WORKSPACE_SCROLL
    try:
        ws.WORKSPACE_SCROLL = True
        assert ws.workspace_box_kwargs() == {
            "height": ws.WORKSPACE_HEIGHT, "border": True}
        # старый Streamlit без ``height`` не должен падать даже с флагом
        assert ws.workspace_box_kwargs(supports_height=False) == {}
    finally:
        ws.WORKSPACE_SCROLL = saved


def test_chat_feed_keeps_its_fixed_height():
    """Лента диалога — по-прежнему окно: ввод под ней не должен уезжать."""
    assert isinstance(ws.CHAT_FEED_HEIGHT, int) and ws.CHAT_FEED_HEIGHT > 300


# ======================================================================
# 2. UI берёт высоту ТОЛЬКО из чистого слоя (не хардкодом на месте)
# ======================================================================
def test_render_workspace_uses_pure_box_kwargs():
    """Иначе решение размазывается по вызову виджета и проверяется глазами."""
    src = inspect.getsource(ui.render_workspace)
    assert "workspace_box_kwargs(" in src
    assert "height=wsx.WORKSPACE_HEIGHT" not in src


def test_chat_feed_still_passes_height_in_dock():
    """Регресс-барьер: снятие окна с ЦЕНТРА не должно распустить ленту."""
    from src.apps import assistant_dock as dock
    src = inspect.getsource(dock.render_assistant_dock)
    assert "height=wsx.CHAT_FEED_HEIGHT" in src


# ======================================================================
# 3. headless AppTest — приложение живёт без окна в центре
# ======================================================================
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(_REPO, "src", "apps", "streamlit_app.py")


def test_app_runs_and_keeps_tabs_and_dialog():
    """Пустая сессия: закладки и поле ввода диалога на месте, исключений нет."""
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception
    keys = {w.key for w in at.button}
    assert "ws_tab_start" in keys and "ws_tab_overview" in keys
    assert any(w.key == "dock_input" for w in at.chat_input)


def test_demo_project_workspace_renders_without_window():
    """С демо-проектом центр рисуется целиком (панели ветки доступны)."""
    at = AppTest.from_file(APP, default_timeout=300).run()
    demo = [w for w in at.button if w.key == "camp_create"]
    assert demo, "кнопка демо-проекта не найдена"
    demo[0].click().run()
    assert not at.exception
    assert at.session_state["ws_tab"] == "branches"
    base = [w for w in at.button if w.key == "ws_tab_base"]
    assert base, "закладка базы опытов не нарисована"
    base[0].click().run()
    assert not at.exception
    # редактор коррекции откликов живёт на «Базе» — значит содержимое отрисовано
    assert [w for w in at.button if w.key == "camp_correct_save"]
