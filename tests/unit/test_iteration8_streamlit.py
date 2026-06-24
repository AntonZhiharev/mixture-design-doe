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
"""Headless smoke-тест Streamlit-приложения (src/apps/streamlit_app.py).

Через `streamlit.testing.v1.AppTest` выполняет скрипт без сервера и проверяет,
что начальный рендер UI (создание проекта, вкладки стадий, sidebar) проходит
без исключений. Пропускается, если streamlit не установлен.
"""
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "src", "apps", "streamlit_app.py")


def test_streamlit_app_renders():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    # заголовок и индикатор прогресса присутствуют
    titles = [t.value for t in at.title]
    assert any("Pipeline" in t for t in titles)


def test_streamlit_run_m1_button():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    # найти кнопку запуска M1 и кликнуть
    m1 = [b for b in at.button if b.key == "run_M1"]
    assert m1, "кнопка run_M1 не найдена"
    m1[0].click().run()
    assert not at.exception
