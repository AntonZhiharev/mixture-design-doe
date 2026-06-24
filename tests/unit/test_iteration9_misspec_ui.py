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
"""Итерация 9 (UI) — панель диагностики misspecification во вкладке «Ветки».

Через headless AppTest: M2 + симулятор, затем кнопка «Проверить модель проекта»
во вкладке «🌿 Ветки» не валит приложение и выдаёт сводку по свойствам.
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


def test_diagnostics_button_ui():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception

    _slider(at, "q_0", 3)
    _slider(at, "restarts_0", 2)

    _click(at, "run_M1")
    _click(at, "run_M2")
    _click(at, "fill_sim")
    assert not at.exception

    # кнопка диагностики доступна (раздел misspecification во вкладке «Ветки»)
    btn = [w for w in at.button if w.key == "diagnose_base"]
    assert btn, "кнопка диагностики модели не найдена"
    _click(at, "diagnose_base")
    assert not at.exception
