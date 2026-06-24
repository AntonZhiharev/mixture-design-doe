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
"""Итерация 9, шаг 1b — UI «Сохранить/Загрузить проект» (headless AppTest).

Проверяет, что кнопка «💾 Сохранить проект» создаёт проект на диске, а
селектор + «📂 Загрузить проект» восстанавливают runner с нужной
конфигурацией. Пишет во временный проект `apptest_io` под project_ui и убирает
его за собой.
"""
import os
import shutil

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(_REPO, "src", "apps", "streamlit_app.py")
PROJ_NAME = "apptest_io"
PROJ_DIR = os.path.join(_REPO, "project_ui", PROJ_NAME)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    if os.path.isdir(PROJ_DIR):
        shutil.rmtree(PROJ_DIR, ignore_errors=True)


def _set_name(at, value):
    # имя проекта — text_input с ключом name_in_<ver>, ver=0 на старте
    boxes = [w for w in at.text_input if w.key == "name_in_0"]
    assert boxes, "поле имени проекта не найдено"
    boxes[0].set_value(value).run()


def test_save_then_load_project_via_ui():
    # --- сохранить проект ---
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    _set_name(at, PROJ_NAME)

    m1 = [b for b in at.button if b.key == "run_M1"]
    assert m1
    m1[0].click().run()
    assert not at.exception

    save = [b for b in at.button if b.key == "save_project"]
    assert save, "кнопка сохранения проекта не найдена"
    save[0].click().run()
    assert not at.exception
    assert os.path.isfile(os.path.join(PROJ_DIR, "state.json"))

    # --- загрузить проект в свежей сессии ---
    at2 = AppTest.from_file(APP, default_timeout=60).run()
    assert not at2.exception
    sel = [w for w in at2.selectbox if w.key == "proj_select"]
    assert sel, "селектор проектов не найден"
    sel[0].set_value(PROJ_NAME).run()
    load = [b for b in at2.button if b.key == "load_project"]
    assert load
    load[0].click().run()
    assert not at2.exception

    runner = at2.session_state["runner"]
    assert runner.cfg.name == PROJ_NAME
    assert runner.q == 5  # дефолтный q сохранён и восстановлен
