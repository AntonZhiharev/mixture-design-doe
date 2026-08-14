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
"""Iteration 93 — два отказа ввода, найденные технологом 14.08.2026.

**Отказ 1: Ctrl+C открывал диалог очистки кеша.** Копирование выделенного
текста из ответа помощника вызывало «Clear caches» — то есть обычная работа с
текстом дёргала разрушающее действие.

Разбор по бандлу Streamlit 1.58 (``static/js/index.dkY5s53S.js``), факты:

1. приложение регистрирует ГЛОБАЛЬНЫЕ горячие клавиши ``keyName: `r,c,esc```;
2. обработчик — ``handleKeyDown = (e,t) => { if(!((e===`c`||e===`r`) && Sz(t)))
   switch(e){ case `c`: G9(isOwner, toolbarMode) && this.openClearCacheDialog()``;
3. ``Sz(t)`` гасит клавишу ТОЛЬКО если фокус внутри ``INPUT``/``TEXTAREA``/
   ``contenteditable``. Выделенный текст ответа лежит в обычном ``div`` —
   значит защита не срабатывает;
4. сочетание регистрируется и на keydown, и на keyup
   (``lz(n, {keydown:!0, keyup:!0}, f)``). На keydown модификатор ``ctrl``
   отсекает вызов, но если отпустить **Ctrl раньше C**, keyup приходит уже без
   ``ctrlKey`` — проверка ``mods.length===0 && !BR[17]`` проходит, и диалог
   открывается;
5. гейт ``G9 = (isOwner, toolbarMode) => toolbarMode==DEVELOPER ? true :
   (VIEWER||MINIMAL) ? false : isOwner||localhost`` — на localhost истина.

Лечение — ``client.toolbarMode = "viewer"`` в проектном
``.streamlit/config.toml``: developer-действия выключаются вместе с клавишей.
Цена нулевая, и это ПРОВЕРЯЕТСЯ здесь: кеша Streamlit в коде нет вообще,
очищать нечего.

**Отказ 2: скриншот не вставлялся в поле ввода.** Docstring дока обещал
мультимодальный ввод, а Ctrl+V молча не работал. Причина не в нашем коде:
в ``ChatInput.*.js`` нет ни ``paste``, ни ``clipboard``, ни ``onPaste`` —
штатный виджет умеет только выбор файла и drag&drop.

Лечение — мост :func:`workspace.chat_paste_js`: картинка из ``clipboardData``
подставляется в СКРЫТЫЙ ``input[type=file]`` того же поля ввода
(``react-dropzone`` читает ``e.target.files``) и объявляется штатным событием
``change``. Новых зависимостей нет, свой протокол не заводится.

Тесты проверяют КОНТРАКТ (состав моста, точки крепления, настройка, отсутствие
кеша) — отрисовку в браузере юнит-тестом не проверить, но исчезновение
селектора, ключа, вызова или настройки поймать можно.
"""
from __future__ import annotations

import ast
import inspect
import os
import re
import tomllib
import warnings

import pytest

from src.apps import workspace as ws

warnings.filterwarnings("ignore")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG = os.path.join(_REPO, ".streamlit", "config.toml")
APP = os.path.join(_REPO, "src", "apps", "streamlit_app.py")


# ======================================================================
# 1. Отказ 1: настройка, гасящая разрушающую горячую клавишу
# ======================================================================
class TestToolbarModeConfig:
    def test_project_config_exists(self):
        """Без файла настройка не доедет: Streamlit читает $CWD/.streamlit."""
        assert os.path.isfile(CONFIG), CONFIG

    def test_toolbar_mode_disables_developer_actions(self):
        """САМ БАГ: при `auto` на localhost клавиша «c» открывает очистку кеша."""
        with open(CONFIG, "rb") as fh:
            cfg = tomllib.load(fh)
        assert cfg["client"]["toolbarMode"] in ("viewer", "minimal")

    def test_config_is_valid_for_streamlit(self):
        """Опечатка в ключе роняла бы приложение на старте."""
        from streamlit import config as stc

        assert "client.toolbarMode" in stc.get_config_options()

    def test_config_explains_why(self):
        """A0.6: настройка без причины через месяц выглядит случайной."""
        text = open(CONFIG, encoding="utf-8").read()
        assert "Ctrl+C" in text or "keyup" in text
        assert "toolbarMode" in text

    def test_app_does_not_use_streamlit_cache(self):
        """Обоснование цены: очищать нечего, кеша Streamlit в коде нет.

        Если кеш когда-нибудь появится, тест упадёт — и решение «выключить
        developer-действия» придётся пересмотреть осознанно, а не по инерции.
        """
        bad = []
        for root, _dirs, files in os.walk(os.path.join(_REPO, "src")):
            for name in files:
                if not name.endswith(".py"):
                    continue
                body = open(os.path.join(root, name), encoding="utf-8").read()
                for pat in (r"@st\.cache", r"st\.cache_data\(",
                            r"st\.cache_resource\("):
                    if re.search(pat, body):
                        bad.append(f"{name}: {pat}")
        assert not bad, ("появился кеш Streamlit — пересмотрите toolbarMode:\n"
                         + "\n".join(bad))


# ======================================================================
# 2. Отказ 2: чистый слой моста вставки
# ======================================================================
class TestPasteBridge:
    def test_bridge_targets_the_input_by_widget_key(self):
        """Точка крепления — класс `st-key-<ключ>`, а не порядковый селектор."""
        js = ws.chat_paste_js()
        assert f".st-key-{ws.DOCK_INPUT_KEY}" in js
        assert "nth-child" not in js

    def test_key_is_css_safe(self):
        """Ключ едет в имя класса: «грязные» символы Streamlit заменил бы на «-»."""
        assert re.fullmatch(r"[A-Za-z0-9_-]+", ws.DOCK_INPUT_KEY)

    def test_bridge_feeds_the_hidden_file_input(self):
        """ГВОЗДЬ ШАГА: файл идёт ШТАТНЫМ путём загрузки, без своего протокола."""
        js = ws.chat_paste_js()
        assert "input[type=file]" in js
        assert "DataTransfer" in js           # собрать FileList иначе нельзя
        assert "input.files" in js
        assert 'new Event("change"' in js     # react-dropzone слушает change

    def test_bridge_reads_the_clipboard(self):
        js = ws.chat_paste_js()
        assert 'addEventListener("paste"' in js
        assert "clipboardData" in js
        assert "image/" in js                 # берём только картинки

    def test_text_paste_is_not_hijacked(self):
        """Без картинки в буфере обработчик обязан уйти молча.

        Иначе Ctrl+V текста в поле ввода перестал бы работать — мы бы починили
        один ввод, сломав другой.
        """
        js = ws.chat_paste_js()
        body = js[js.index('addEventListener("paste"'):]
        assert body.index("if (!imgs.length) { return; }") \
            < body.index("preventDefault"), \
            "preventDefault не должен вызываться раньше проверки на картинку"

    def test_bridge_is_installed_once(self):
        """Скрипт вставляется на каждом прогоне: без флага слушатели копились бы."""
        js = ws.chat_paste_js()
        assert ws.PASTE_BRIDGE_FLAG in js
        assert "window[FLAG]" in js

    def test_pasted_file_gets_a_distinguishable_name(self):
        """Из буфера Chrome отдаёт «image.png» — в переписке они бы слиплись."""
        js = ws.chat_paste_js()
        assert "screenshot-" in js
        assert "Date.now()" in js

    def test_key_is_a_parameter_not_a_hardcoded_literal(self):
        """Другое поле ввода — другой ключ; мост не должен быть вшит в одно."""
        js = ws.chat_paste_js("other_input")
        assert ".st-key-other_input" in js
        assert f".st-key-{ws.DOCK_INPUT_KEY}" not in js

    def test_bridge_is_wrapped_in_script_tag(self):
        """Инъекция идёт через st.html(unsafe_allow_javascript=True)."""
        js = ws.chat_paste_js()
        assert js.startswith("<script>") and js.rstrip().endswith("</script>")

    def test_no_placeholder_is_left_unreplaced(self):
        """Незаменённый шаблон означал бы мост, который никуда не цепляется."""
        js = ws.chat_paste_js()
        assert "__KEY__" not in js and "__FLAG__" not in js


# ======================================================================
# 3. UI цепляется к чистому слою (и один ключ на оба конца)
# ======================================================================
class TestDockWiring:
    def test_dock_injects_the_bridge_and_allows_javascript(self):
        from src.apps import assistant_dock as dock

        src = inspect.getsource(dock._chat_submission)
        assert "wsx.chat_paste_js(" in src
        assert "unsafe_allow_javascript=True" in src

    def test_dock_uses_the_shared_key(self):
        """Расхождение ключа и селектора сломало бы вставку МОЛЧА."""
        from src.apps import assistant_dock as dock

        src = inspect.getsource(dock._chat_submission)
        assert "key=wsx.DOCK_INPUT_KEY" in src
        assert 'key="dock_input"' not in src

    def test_shared_key_keeps_the_historical_value(self):
        """На «dock_input» держатся тесты iter88/89 — менять его нельзя молча."""
        assert ws.DOCK_INPUT_KEY == "dock_input"

    def test_bridge_goes_before_the_widget(self):
        """Скрипт ищет поле в документе: сначала мост, потом само поле.

        Порядок проверяется по AST, а не по тексту — так тест не цепляется к
        форматированию.
        """
        from src.apps import assistant_dock as dock

        tree = ast.parse(inspect.getsource(dock._chat_submission))
        order = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ("html", "chat_input"):
                    order.append((node.func.attr, node.lineno))
        assert [k for k, _ in order] == ["html", "chat_input"], order

    def test_placeholder_mentions_paste(self):
        """Возможность, о которой не сказано, для человека не существует."""
        from src.apps import assistant_dock as dock

        src = inspect.getsource(dock._chat_submission)
        assert "вставить из буфера" in src

    def test_launcher_runs_from_the_repo_root(self):
        """Иначе проектный config.toml не найдётся и настройка потеряется."""
        import run_streamlit_app as launcher

        assert "cwd=HERE" in inspect.getsource(launcher.main)


# ======================================================================
# 4. headless AppTest — приложение живёт с мостом и настройкой
# ======================================================================
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _bridge_protos(at):
    """Элементы ``st.html`` с нашим мостом (AppTest типа для html не имеет).

    Обход дерева: у блоков ``children`` — словарь по индексу, поэтому обходим
    значения, а не сам объект.
    """
    out = []

    def walk(node):
        proto = getattr(node, "proto", None)
        body = getattr(proto, "body", None) if proto is not None else None
        if isinstance(body, str) and ws.PASTE_BRIDGE_FLAG in body:
            out.append(proto)
        kids = getattr(node, "children", None)
        for child in (kids.values() if isinstance(kids, dict) else (kids or [])):
            walk(child)

    walk(at.main)
    return out


def test_app_runs_and_keeps_the_input_with_the_bridge():
    """Пустая сессия: приложение поднялось, поле ввода на месте."""
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception
    assert any(w.key == ws.DOCK_INPUT_KEY for w in at.chat_input)


def test_bridge_is_rendered_once_with_javascript_enabled():
    """ГВОЗДЬ: без `unsafe_allow_javascript` скрипт был бы вырезан санитайзером.

    Streamlit прогоняет ``st.html`` через DOMPurify и добавляет ``script`` в
    разрешённые теги ТОЛЬКО при этом флаге. Дубль вставки тоже проверяем: два
    моста = два слушателя = двойная вставка скриншота.
    """
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception
    protos = _bridge_protos(at)
    assert len(protos) == 1, f"мостов должно быть ровно 1, найдено {len(protos)}"
    assert protos[0].unsafe_allow_javascript is True


def test_app_still_renders_sticky_css_after_the_change():
    """Регресс iter89: инъекция CSS не должна пострадать от новой вставки."""
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception
    blobs = [str(m.value) for m in at.markdown]
    assert any("position: sticky" in b for b in blobs)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
