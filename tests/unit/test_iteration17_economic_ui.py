# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 17 / §15.6 ШАГ 7 (UI) — экономический стоп + flat-ось (A0.7).

Через headless AppTest: M2 + симулятор, заводим ветку, открываем панель
«💰 §15.6» и жмём «Оценить экономику» — проверяем, что предложения (триада/стоп/
flat-ось) рендерятся БЕЗ падений и состояние проекта НЕ меняется (A0.6 read-only).
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


def test_economic_panel_readonly_proposals_ui():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception

    _slider(at, "q_0", 3)
    _slider(at, "restarts_0", 2)
    props = [w for w in at.text_input if w.key == "props_0"]
    assert props, "поле свойств не найдено"
    props[0].set_value("Прочность").run()
    assert not at.exception

    _click(at, "run_M1")
    _click(at, "run_M2")
    _click(at, "fill_sim")
    assert not at.exception

    # завести ветку (max первого свойства)
    _click(at, "add_branch")
    assert not at.exception
    runner = at.session_state["runner"]
    assert len(runner.branches) == 1
    n_before = 0 if runner.design is None else len(runner.design)

    # панель §15.6 присутствует (селектор ветки/компонента)
    assert [w for w in at.selectbox if w.key == "econ_branch"]
    assert [w for w in at.selectbox if w.key == "econ_comp"]

    # нажать «Оценить экономику» — предложения рендерятся, проект НЕ меняется
    _click(at, "econ_eval")
    assert not at.exception
    runner = at.session_state["runner"]
    n_after = 0 if runner.design is None else len(runner.design)
    assert n_after == n_before                  # A0.6: read-only, точек не добавили
    assert runner.branches  # ветка на месте
