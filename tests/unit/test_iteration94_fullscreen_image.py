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
"""Iteration 94 — картинка из ответа помощника «просвечивала» при раскрытии.

Отказ (наблюдение технолога 14.08.2026): нажатие «Fullscreen» на графике из
ответа помощника показывало картинку поверх страницы, но сквозь неё была видна
рабочая область и инфо-панель, а сам показ был обрезан по колонке.

Разбор — по фактам, а не по догадке.

1. **Файл не виноват.** Артефакты песочницы — png в режиме ``RGBA``, но с
   ``alpha = 255`` и белым фоном (проверено на 6 последних файлах кампании);
   ``savefig.transparent`` в matplotlib 3.11 = ``False``, ``figure.facecolor``
   = ``white``. Прозрачности в данных нет.
2. **Код показа не виноват.** ``assistant_dock._render_outputs`` рисует обычный
   ``st.image``; полноэкранный режим — штатный механизм Streamlit.
3. **Причина — сцепка нашего iter89 с обёрткой Streamlit.** По бандлу 1.58
   (``static/js/withFullScreenWrapper.Cayfkf4W.js``) раскрытый элемент получает
   ``position: fixed``, ``top/left/right/bottom: 0``,
   ``background: theme.colors.bgColor``, ``overflow: auto``,
   ``zIndex: theme.zIndices.fullscreenWrapper``; в теме
   (``index.dkY5s53S.js``) ``fullscreenWrapper = 1e6 + 50``. Базовый стиль
   колонки ``position`` НЕ задаёт — у стокового Streamlit z-index обёртки
   считается от корня, и раскрытие закрывает страницу.

   Наш ``sticky_zones_css`` (iter89) ставит колонке ``position: sticky``, а
   sticky, по справке MDN о ``position``, **всегда создаёт новый stacking
   context**. Значит z-index 1000050 действует только ВНУТРИ колонки, а её
   ``max-height`` + ``overflow-y: auto`` вдобавок обрезают обёртку.

4. **Признак «сейчас раскрыто», доступный из CSS** — кнопка выхода: в бандле
   ``t && !o && n && P(iW, {label: `Close fullscreen`…})``, где ``n`` =
   ``isFullScreen``, то есть в свёрнутом состоянии кнопки в DOM нет вообще.
   Кнопка — настоящий ``<button>`` (``wk = G('button')`` → ``Jk`` → ``$k``,
   ``aria-label`` = подпись) и лежит ВНУТРИ поддерева элемента (``ImageList``
   рендерит тулбар сам, ``createPortal`` в этом чанке нет), поэтому ``:has()``
   от колонки до неё достаёт.

Лечение — :func:`workspace.fullscreen_escape_css`: пока такая кнопка есть в
колонке, с колонки снимается липкость, ограничение высоты и обрезка.

Тесты проверяют КОНТРАКТ (состав правил, точки крепления, инъекция, отсутствие
регресса iter89) — как браузер применил правила, юнит-тест проверить не может;
это честная граница, та же, что в iter89/iter93.
"""
from __future__ import annotations

import inspect
import os
import re
import warnings

import pytest

from src.apps import workspace as ws

warnings.filterwarnings("ignore")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(_REPO, "src", "apps", "streamlit_app.py")


# ======================================================================
# 1. Чистый слой: состав правил снятия липкости
# ======================================================================
class TestFullscreenEscapeCss:
    def test_wrapped_in_style_tag(self):
        """Инъекция идёт через st.markdown(unsafe_allow_html=True)."""
        css = ws.fullscreen_escape_css()
        assert css.startswith("<style>") and css.rstrip().endswith("</style>")

    def test_targets_the_column_by_exit_button(self):
        """Крепление — кнопка выхода: её нет в DOM, пока показ свёрнут."""
        css = ws.fullscreen_escape_css()
        for label in ws.FULLSCREEN_EXIT_LABELS:
            assert (f'[data-testid="stColumn"]:has(button[aria-label="{label}"])'
                    in css)

    def test_cancels_every_property_that_traps_the_overlay(self):
        """Ловушек ТРИ: stacking context, предел высоты и обрезка.

        Снять только ``position`` мало: ``max-height`` с ``overflow`` обрежут
        «полноэкранный» блок по колонке даже без stacking context.
        """
        css = ws.fullscreen_escape_css()
        assert "position: static" in css
        assert "max-height: none" in css
        assert "overflow: visible" in css

    def test_exit_labels_are_both_known_spellings(self):
        """Разные чанки Streamlit пишут подпись по-разному — нужны обе."""
        assert "Close fullscreen" in ws.FULLSCREEN_EXIT_LABELS
        assert "Exit fullscreen" in ws.FULLSCREEN_EXIT_LABELS

    def test_labels_are_safe_for_attribute_selector(self):
        """Подпись едет в CSS-селектор: кавычка в ней сломала бы правило."""
        for label in ws.FULLSCREEN_EXIT_LABELS:
            assert re.fullmatch(r"[A-Za-z ]+", label), label

    def test_lives_in_the_same_media_query_as_stickiness(self):
        """Ниже порога колонки не липкие — отменять там нечего."""
        css = ws.fullscreen_escape_css()
        assert f"@media (min-width: {ws.STICKY_MIN_WIDTH_PX}px)" in css

    def test_min_width_is_a_parameter(self):
        """Порог — аргумент, а не вшитое число (как в sticky_zones_css)."""
        assert "@media (min-width: 1200px)" in ws.fullscreen_escape_css(
            min_width_px=1200)

    def test_no_ordinal_selectors(self):
        """Порядковый селектор поехал бы от любой правки раскладки, и МОЛЧА."""
        assert "nth-child" not in ws.fullscreen_escape_css()

    def test_addresses_only_the_expanded_column(self):
        """Правило адресует РОВНО колонку с раскрытым показом, не страницу."""
        css = ws.fullscreen_escape_css()
        assert css.count(":has(") == len(ws.FULLSCREEN_EXIT_LABELS)
        assert ".st-key-" not in css        # зоны по ключам здесь не при чём


# ======================================================================
# 2. Липкость iter89 этой правкой не сломана
# ======================================================================
class TestStickyContractKept:
    def test_side_zones_are_still_sticky(self):
        """iter89 остаётся: боковые зоны по-прежнему держатся на экране."""
        css = ws.sticky_zones_css()
        assert "position: sticky" in css
        for key in (ws.DOCK_ZONE_KEY, ws.INFO_ZONE_KEY):
            assert f'[data-testid="stColumn"]:has(.st-key-{key})' in css

    def test_two_decisions_are_two_functions(self):
        """Разные решения — разные функции: компромисс виден, а не спрятан."""
        assert ws.sticky_zones_css() != ws.fullscreen_escape_css()
        assert "Close fullscreen" not in ws.sticky_zones_css()


# ======================================================================
# 3. UI действительно впрыскивает правило
# ======================================================================
def _app():
    from src.apps import streamlit_app as app
    return app


class TestAppInjection:
    def test_app_injects_both_css_blocks(self):
        """Без вызова правило не доедет до страницы."""
        src = inspect.getsource(_app().main)
        assert "fullscreen_escape_css()" in src
        assert "sticky_zones_css()" in src          # оба, а не вместо

    def test_escape_is_injected_after_sticky(self):
        """У правил равная специфичность — снятие обязано идти ПОЗЖЕ."""
        src = inspect.getsource(_app().main)
        assert src.index("sticky_zones_css()") < src.index(
            "fullscreen_escape_css()")


# ======================================================================
# 4. Живой рендер: приложение поднимается с обеими инъекциями
# ======================================================================
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def test_app_runs_and_renders_both_css_blocks():
    """Пустая сессия: липкость и её снятие отрисованы, исключений нет."""
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception
    blobs = [str(m.value) for m in at.markdown]
    assert any("position: sticky" in b for b in blobs), "CSS липкости не отрисован"
    assert any("Close fullscreen" in b for b in blobs), "CSS снятия не отрисован"
