# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 72 — экран из ТРЁХ зон (эскиз пользователя).

Концепция: ассистент — инструмент взаимодействия с программой, поэтому ЛЕВАЯ
зона целиком отдана ему (диалог + панели утверждения его предложений).
ЦЕНТРАЛЬНАЯ зона — рабочая область на закладках (появляются, когда есть данные
— гейты iter69 сохраняются). КРАЙНЯЯ ПРАВАЯ зона — постоянная дополнительная
информация, нужная на разных закладках и большей частью связанная с работой
ассистента: 📎 вложения, 🖼 выхлоп песочницы, 📌 состояние сессии.

Раньше (iter69) колонок было две, и инфо-панели жили в левой колонке ПОД
перепиской: до них приходилось скроллить сквозь весь диалог.

Проверяем:

* ЧИСТУЮ раскладку (:data:`src.apps.workspace.MAIN_COLUMNS`) — три зоны,
  центр самый широкий;
* разнесение панелей по исходникам дока: справка (вложения/артефакты/сессия)
  ушла в :func:`render_assistant_info`, а панели УТВЕРЖДЕНИЯ (пакеты спеки,
  патчи — применяет только человек) остались в :func:`render_assistant_dock`;
* headless AppTest: все три зоны рендерятся ОДНОВРЕМЕННО на одной странице.
"""
import inspect
import os
import warnings

import pytest
from sklearn.exceptions import ConvergenceWarning

from src.apps import assistant_dock as dock
from src.apps import workspace as ws

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ======================================================================
# 1. Чистая раскладка: три зоны, центр — рабочая область — самый широкий
# ======================================================================
def test_main_columns_define_three_zones():
    """Эскиз iter72: диалог | рабочая область | инфо-панель."""
    assert len(ws.MAIN_COLUMNS) == 3
    left, center, right = ws.MAIN_COLUMNS
    assert all(isinstance(w, int) and w > 0 for w in ws.MAIN_COLUMNS)
    # центр — таблицы и формы потока §17, ему нужна наибольшая ширина
    assert center > left and center > right


# ======================================================================
# 2. Разнесение панелей: справка — вправо, утверждение — у ассистента
# ======================================================================
def test_info_panels_moved_out_of_dialog_column():
    """Вложения/файлы расчётов/состояние переписки больше НЕ под перепиской."""
    dock_src = inspect.getsource(dock.render_assistant_dock)
    info_src = inspect.getsource(dock.render_assistant_info)
    # iter74: подписи панелей переведены на операционные формулировки
    # («выхлоп песочницы» → «файлы расчётов ассистента»), поэтому маркером
    # служит новая формулировка блока состояния.
    for marker in ("_render_attachments(", "_render_artifacts(",
                   "Состояние переписки"):
        assert marker not in dock_src, f"«{marker}» остался в колонке диалога"
        assert marker in info_src, f"«{marker}» не попал в инфо-панель"


def test_approval_panels_stay_with_assistant():
    """Пакеты спеки и патчи — часть работы с ассистентом (решение человека)."""
    dock_src = inspect.getsource(dock.render_assistant_dock)
    assert "_render_spec_packages(" in dock_src
    assert "_render_patches(" in dock_src
    info_src = inspect.getsource(dock.render_assistant_info)
    assert "_render_spec_packages(" not in info_src
    assert "_render_patches(" not in info_src


def test_both_columns_share_one_session():
    """Обе колонки читают сессию одним путём — двух копий состояния нет."""
    for fn in (dock.render_assistant_dock, dock.render_assistant_info):
        assert "dock_session(root, project)" in inspect.getsource(fn)


# ======================================================================
# 3. headless AppTest — три зоны живут на одной странице
# ======================================================================
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(_REPO, "src", "apps", "streamlit_app.py")


def test_app_renders_all_three_zones_together():
    """Диалог, закладки рабочей области и инфо-панель — одновременно."""
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception
    # левая зона: поле ввода диалога ассистента
    assert any(w.key == "dock_input" for w in at.chat_input), \
        "поле ввода дока ассистента не найдено"
    # центральная зона: ряд закладок рабочей области
    keys = {w.key for w in at.button}
    assert "ws_tab_overview" in keys
    # правая зона: заголовок инфо-панели и кнопка очистки из «Состояния сессии»
    subs = [str(s.value) for s in at.subheader]
    assert any("Инфо" in s for s in subs), "заголовок инфо-панели не найден"
    assert "dock_clear" in keys, "панель состояния переписки не нарисована"


def test_info_panel_survives_demo_project():
    """С собранным демо-проектом инфо-панель остаётся на месте."""
    at = AppTest.from_file(APP, default_timeout=300).run()
    demo = [w for w in at.button if w.key == "camp_create"]
    assert demo, "кнопка демо-проекта не найдена"
    demo[0].click().run()
    assert not at.exception
    keys = {w.key for w in at.button}
    assert "dock_clear" in keys
    assert any(w.key == "dock_input" for w in at.chat_input)


def test_project_entry_lives_on_start_tab_not_sidebar():
    """Сетап и персистентность проекта — на закладке «🌱 Старт», сайдбар пуст.

    Концепция iter72: самая левая область целиком у ассистента, у «Проекта»
    своя первая закладка рабочей области. Пустая сессия открывается сразу на
    ней — там форма «🆕 Новый проект» (`setup_build`), панель «📁 Проект»
    (`save_campaign`/`load_campaign`) и удаление (admin).
    """
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception
    # дефолтная закладка пустой сессии — «Старт» (вход в проект)
    assert at.session_state["ws_tab"] == "start"
    keys = {w.key for w in at.button}
    assert "setup_build" in keys, "форма сетапа не на «Старте»"
    assert "save_campaign" in keys and "load_campaign" in keys, \
        "панель «📁 Проект» не на «Старте»"
    # сайдбар упразднён: в нём не осталось ни одного виджета
    assert not at.sidebar.button, "в сайдбаре остались кнопки"
    assert not at.sidebar.text_input, "в сайдбаре остались поля ввода"
    # уйдя на «Обзор», человек теряет форму сетапа с экрана (она не дублируется)
    tab = [w for w in at.button if w.key == "ws_tab_overview"]
    tab[0].click().run()
    assert not at.exception
    assert "setup_build" not in {w.key for w in at.button}
    # имя проекта — состояние приложения: переживает уход с закладки «Старт»
    assert at.session_state["campaign_name"] == "my_project"
