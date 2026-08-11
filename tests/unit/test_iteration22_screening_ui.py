# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 22 (UI) — секция «Анализ скрининга (M3)» во вкладке «Кампания».

Headless AppTest: создаём демо-кампанию (общий пул + 2 ветки), затем в секции
M3 (:func:`campaign_ui.render_screening_analysis`) нажимаем «матрицу влияний» и
«разобрать свойство». Проверяем, что кнопки есть, приложение не падает и результат
кешируется в session_state (чистая математика уже покрыта test_iteration22_screening_m3).
"""
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_APP = os.path.join(_REPO, "src", "apps", "streamlit_app.py")


def _click(at, key):
    b = [w for w in at.button if w.key == key]
    assert b, f"кнопка {key} не найдена"
    b[0].click().run()


def _open_tab(at, tab_key):
    """iter69: рабочая область на закладках — открыть нужную (ряд кнопок)."""
    _click(at, f"ws_tab_{tab_key}")


def test_screening_section_renders_and_computes():
    at = AppTest.from_file(_APP, default_timeout=300).run()
    assert not at.exception

    # создать демо-кампанию (seed измерен, 2 ветки) → секция M3 доступна
    _click(at, "camp_create")
    assert not at.exception

    # iter69: анализ скрининга живёт на закладке «📊 Анализ» рабочей области
    _open_tab(at, "screening")
    assert not at.exception

    # кнопки секции «Анализ скрининга (M3)» присутствуют
    assert [w for w in at.button if w.key == "camp_m3_overview_btn"], \
        "кнопка матрицы влияний M3 не найдена"
    assert [w for w in at.button if w.key == "camp_m3_fit_btn"], \
        "кнопка разбора свойства M3 не найдена"

    # матрица влияний (компонент × свойство) — считается и кешируется
    _click(at, "camp_m3_overview_btn")
    assert not at.exception
    assert "camp_m3_matrix" in at.session_state
    mat = at.session_state["camp_m3_matrix"]
    # 3 компонента × 3 свойства демо-оракула (A,B,C × strength,gloss,rho)
    assert list(mat.index) == ["A", "B", "C"]
    assert set(mat.columns) == {"strength", "gloss", "rho"}

    # детальный разбор свойства — Scheffé-fit + ARD, отчёт кешируется
    _click(at, "camp_m3_fit_btn")
    assert not at.exception
    assert "camp_m3_report" in at.session_state
    rep = at.session_state["camp_m3_report"]

    assert rep["components"] == ["A", "B", "C"]
    assert len(rep["coefficients"]) == 6      # quadratic Scheffé, q=3
    assert "d_overall" not in rep             # это скрининг, не рецепт ветки
    assert 0.0 <= rep["summary"]["r2"] <= 1.0
