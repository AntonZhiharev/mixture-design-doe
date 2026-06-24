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
"""Итерация 9, шаг 3d (UI) — менеджер веток (REBUILD_SPEC §5/§12).

Через headless AppTest: задаём свойства, прогоняем M2 + симулятор, во вкладке
«Ветки» заводим ветку и гоняем портфельный раунд (арбитраж бюджета). Проверяем,
что точки попали в ОБЩУЮ базу с origin-тегом ветки и без падений приложения.
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


def test_branch_manager_portfolio_round_ui():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception

    # ускоряем: q=3, рестартов поменьше
    _slider(at, "q_0", 3)
    _slider(at, "restarts_0", 2)

    # два свойства
    props = [w for w in at.text_input if w.key == "props_0"]
    assert props, "поле свойств не найдено"
    props[0].set_value("Прочность, Вязкость").run()
    assert not at.exception

    # M1 → M2 → заполнить симулятором
    _click(at, "run_M1")
    _click(at, "run_M2")
    _click(at, "fill_sim")
    assert not at.exception

    runner = at.session_state["runner"]
    n0 = len(runner.design)

    # вкладка «Ветки»: целевое свойство-селектор должен присутствовать
    bsel = [w for w in at.selectbox if w.key == "branch_prop"]
    assert bsel, "селектор целевого свойства ветки не найден"
    assert set(bsel[0].options) >= {"Прочность", "Вязкость"}

    # завести ветку (по умолчанию: max первого свойства, бюджет 6)
    _click(at, "add_branch")
    assert not at.exception
    runner = at.session_state["runner"]
    assert len(runner.branches) == 1
    bid = next(iter(runner.branches))

    # портфельный раунд (по умолчанию 4 слота) — арбитр отдаёт их единственной ветке
    _click(at, "run_portfolio")
    assert not at.exception

    runner = at.session_state["runner"]
    assert len(runner.design) == n0 + 4
    counts = runner.origin_counts()
    assert counts.get(f"branch:{bid}", 0) == 4
    assert runner.branches[bid].spent == 4
    # общая модель проекта: суррогаты на каждое свойство
    assert set(runner.surrogates) == {"Прочность", "Вязкость"}
