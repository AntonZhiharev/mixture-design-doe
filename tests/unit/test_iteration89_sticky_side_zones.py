# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 89 — боковые зоны ЛИПКИЕ при прокрутке рабочей области.

Запрос технолога (13.08.2026, продолжение iter88): «сделай, чтобы правый и левый
блоки всегда были в поле видимости, пока рабочую область листаешь вниз».

Контекст. iter88 снял с центра окно 760 px — центр стал обычной страницей с
одним скроллом. Побочный эффект: теперь вниз уезжает ВСЯ страница, вместе с
диалогом ассистента слева и инфо-панелью справа. А это ровно те две зоны,
которые нужны постоянно: спросить «почему такой диапазон» и посмотреть файлы
расчётов человек хочет, НЕ возвращаясь к началу страницы.

Решение — ``position: sticky`` на боковых КОЛОНКАХ. Проверенные факты о разметке
Streamlit 1.58 (по бандлу ``static/js/index.dkY5s53S.js``), на которых оно
держится:

* скроллит ``section[data-testid="stMain"]`` — у него ``overflow: auto`` и
  ``height: 100dvh``, значит sticky внутри прилипает к этому контейнеру;
* колонка — ``div[data-testid="stColumn"]``, прямой flex-элемент строки; своего
  ``overflow`` у неё нет (иначе sticky бы гасился);
* ``st.columns`` НЕ принимает ``key``, поэтому адресовать колонку из CSS можно
  только «изнутри»: контейнер с ``key`` даёт класс ``st-key-<key>``
  (документировано в docstring ``st.container``), а до колонки поднимаемся
  через ``:has()``.

Тест проверяет КОНТРАКТ (состав правил и точки крепления), а не внешний вид:
отрисовку CSS браузером юнит-тестом не проверить, но исчезновение селектора,
ключа или самого вызова — можно.
"""
import inspect
import os
import re
import warnings

import pytest
from sklearn.exceptions import ConvergenceWarning

from src.apps import workspace as ws

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ======================================================================
# 1. Чистый слой: состав CSS-правил
# ======================================================================
def test_zone_keys_are_distinct_and_css_safe():
    """Ключ едет в имя класса — «грязные» символы Streamlit заменил бы на «-»."""
    assert ws.DOCK_ZONE_KEY != ws.INFO_ZONE_KEY
    for key in (ws.DOCK_ZONE_KEY, ws.INFO_ZONE_KEY):
        assert re.fullmatch(r"[A-Za-z0-9_-]+", key), key


def test_css_sticks_the_column_not_the_inner_container():
    """Липкой должна быть КОЛОНКА: внутренний контейнер уедет вместе с ней."""
    css = ws.sticky_zones_css()
    for key in (ws.DOCK_ZONE_KEY, ws.INFO_ZONE_KEY):
        assert f'[data-testid="stColumn"]:has(.st-key-{key})' in css
    assert "position: sticky" in css


def test_css_has_align_self_flex_start():
    """Растянутому (align-items: stretch) элементу прилипать некуда."""
    assert "align-self: flex-start" in ws.sticky_zones_css()


def test_css_is_wrapped_in_style_tag():
    """Инъекция идёт через st.markdown(unsafe_allow_html=True)."""
    css = ws.sticky_zones_css()
    assert css.startswith("<style>") and css.rstrip().endswith("</style>")


def test_css_is_disabled_on_narrow_screens():
    """Узкий экран: Streamlit переносит колонки вниз — липкость там мешает."""
    css = ws.sticky_zones_css()
    assert f"@media (min-width: {ws.STICKY_MIN_WIDTH_PX}px)" in css
    assert ws.STICKY_MIN_WIDTH_PX > 0


def test_side_zones_get_their_own_scroll_when_taller_than_screen():
    """Иначе низ прилипшей зоны недосягаем: она НЕ едет вместе со страницей."""
    css = ws.sticky_zones_css()
    assert "overflow-y: auto" in css
    assert "max-height: calc(100dvh -" in css


def test_side_scroll_can_be_switched_off():
    """Ограничение высоты — параметр, а не вшитое решение."""
    css = ws.sticky_zones_css(side_scroll=False)
    assert "overflow-y: auto" not in css
    assert "position: sticky" in css       # липкость при этом остаётся


def test_top_offset_is_configurable_and_matches_max_height():
    """Отступ сверху и предел высоты считаются от ОДНОГО числа."""
    css = ws.sticky_zones_css(top_rem=4.0)
    assert "top: 4rem" in css
    assert "max-height: calc(100dvh - 4.5rem)" in css


# ======================================================================
# 2. UI цепляется к чистому слою (а не к порядковым селекторам)
# ======================================================================
def test_app_injects_css_and_wraps_zones_with_keys():
    """Без обёрток с ключами селектор ни к чему не прицепится."""
    from src.apps import streamlit_app as app
    src = inspect.getsource(app.main)
    assert "sticky_zones_css()" in src
    assert "st.container(key=wsx.DOCK_ZONE_KEY)" in src
    assert "st.container(key=wsx.INFO_ZONE_KEY)" in src
    # порядковые селекторы поехали бы от любой правки раскладки
    assert "nth-child" not in src


def test_center_zone_is_not_sticky():
    """Липким становится только окружение: центр обязан листаться (iter88)."""
    css = ws.sticky_zones_css()
    assert "stColumn:nth" not in css
    # в правилах фигурируют РОВНО две зоны — левая и правая
    assert css.count(":has(") == 2


# ======================================================================
# 3. headless AppTest — приложение живёт с инъекцией и обёртками
# ======================================================================
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(_REPO, "src", "apps", "streamlit_app.py")


def test_app_runs_with_sticky_css_and_keeps_all_three_zones():
    """Пустая сессия: CSS отрисован, диалог/закладки/инфо-панель на месте."""
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception
    blobs = [str(m.value) for m in at.markdown]
    assert any("position: sticky" in b for b in blobs), "CSS не отрисован"
    keys = {w.key for w in at.button}
    assert "ws_tab_start" in keys                              # центр
    assert any(w.key == "dock_input" for w in at.chat_input)   # левая зона
    assert "dock_clear" in keys                                # правая зона


def test_zone_wrappers_do_not_break_demo_project():
    """Вложенный контейнер не должен ломать рендер панелей внутри зон."""
    at = AppTest.from_file(APP, default_timeout=300).run()
    demo = [w for w in at.button if w.key == "camp_create"]
    assert demo, "кнопка демо-проекта не найдена"
    demo[0].click().run()
    assert not at.exception
    keys = {w.key for w in at.button}
    assert "dock_clear" in keys
    assert any(w.key == "dock_input" for w in at.chat_input)
