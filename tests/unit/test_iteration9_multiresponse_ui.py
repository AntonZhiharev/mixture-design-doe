# Copyright 2026 AZH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Итерация 9, шаг 3b (UI) — мультиотклик в M3/M6 (REBUILD_SPEC §12).

Через headless AppTest: задаём несколько свойств, заполняем отклики
симулятором, прогоняем M3/M6 и проверяем, что появился селектор свойства
(`m3_prop`/`m6_prop`) и переключение между свойствами не роняет приложение.
"""
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(_REPO, "src", "apps", "streamlit_app.py")


def _slider(at, key, value):
    s = [w for w in at.slider if w.key == key]
    assert s, f"слайдер {key} не найден"
    s[0].set_value(value).run()


def _click(at, key):
    b = [w for w in at.button if w.key == key]
    assert b, f"кнопка {key} не найдена"
    b[0].click().run()


def test_m3_m6_per_property_selector_ui():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception

    # ускоряем: q=3, рестартов поменьше
    _slider(at, "q_0", 3)
    _slider(at, "restarts_0", 2)

    # задаём три свойства
    props = [w for w in at.text_input if w.key == "props_0"]
    assert props, "поле свойств не найдено"
    props[0].set_value("Прочность, Вязкость, Цвет").run()
    assert not at.exception

    # M1 → M2 → заполнить симулятором
    _click(at, "run_M1")
    _click(at, "run_M2")
    _click(at, "fill_sim")
    assert not at.exception

    # M3: скрининг на все свойства + селектор свойства
    _click(at, "run_M3")
    assert not at.exception
    m3sel = [w for w in at.selectbox if w.key == "m3_prop"]
    assert m3sel, "селектор свойства M3 не появился"
    assert set(m3sel[0].options) >= {"Прочность", "Вязкость", "Цвет"}

    runner = at.session_state["runner"]
    assert set(runner.results["M3_fit"]["per_property"]) == \
        {"Прочность", "Вязкость", "Цвет"}

    # переключим свойство — без падений
    m3sel[0].set_value("Вязкость").run()
    assert not at.exception

    # M6: суррогат на все свойства + словарь surrogates
    _click(at, "run_M6")
    assert not at.exception
    m6sel = [w for w in at.selectbox if w.key == "m6_prop"]
    assert m6sel, "селектор свойства M6 не появился"
    assert set(runner.surrogates) == {"Прочность", "Вязкость", "Цвет"}
