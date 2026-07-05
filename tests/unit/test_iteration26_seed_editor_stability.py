# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 26 — багфикс: пропадающие значения в редакторе seed-откликов (§17.4).

Багрепорт: при ручном вводе откликов в таблицу стартового дизайна первая
введённая ячейка в каждой следующей колонке «исчезала», повторный ввод
принимался. Причина — петля «выход редактора → его вход»: df для
``st.data_editor`` пересобирался КАЖДЫЙ rerun с уже внесённым черновиком Y
(``setup_seed_Y``); для Streamlit изменившиеся данные = НОВЫЙ виджет, и свежая
(ещё не запечённая в df) правка сбрасывалась.

Фикс: вход редактора кэшируется в ``session_state['setup_seed_df']`` по
сигнатуре (дизайн Xs + размер пробы) и пересобирается ТОЛЬКО при её смене или
явном заполнении (демо-кнопка, новый дизайн, загрузка проекта) — черновик Y
вливается в кэш в этот момент, поэтому правки не теряются.

Тесты (headless AppTest поверх streamlit_app):
  * стабильность: изменение черновика ``setup_seed_Y`` между прогонами НЕ
    меняет вход редактора (df не пересобирается → виджет не сбрасывается);
  * демо-кнопка «Заполнить тестовыми» пересобирает df с новыми Y;
  * смена размера пробы (сигнатуры) пересобирает df, ВЛИВАЯ черновик Y
    (правки переживают легитимную пересборку).
"""
import os
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(_REPO, "src", "apps", "streamlit_app.py")

# lab-колонки дефолтного проекта формы сетапа (strength, gloss, rho)
LAB_COLS = ["strength (lab)", "gloss (lab)", "rho (lab)"]


def _app_with_seed_design() -> AppTest:
    """Свежая сессия: построить проект (дефолты формы) + предложить seed из 6 точек."""
    at = AppTest.from_file(APP, default_timeout=300).run()
    [w for w in at.button if w.key == "setup_build"][0].click().run()
    assert not at.exception
    at.session_state["setup_seed_n"] = 6
    [w for w in at.button if w.key == "setup_propose_seed"][0].click().run()
    assert not at.exception
    assert "setup_seed_df" in at.session_state, "кэш входа редактора не создан"
    return at


def test_editor_input_stable_when_draft_changes():
    """Регресс (пропадающие значения): черновик Y меняется → вход редактора НЕТ.

    Раньше df пересобирался из черновика на каждый rerun ⇒ data_editor получал
    «новые данные», сбрасывал незапечённую правку, и введённое значение
    пропадало. Теперь df в кэше стабилен, пока не сменилась сигнатура."""
    at = _app_with_seed_design()
    df0 = at.session_state["setup_seed_df"].copy()
    # lab-колонки пусты (места под ручной ввод)
    assert df0[LAB_COLS].isna().all().all()

    # имитируем черновик из редактора: пользователь ввёл gloss = 6.6247 в №1
    Y = np.full((6, 3), np.nan)
    Y[0, 1] = 6.6247
    at.session_state["setup_seed_Y"] = Y
    at.run()
    assert not at.exception

    # вход редактора НЕ пересобрался (иначе виджет сбросил бы свежие правки)
    pd.testing.assert_frame_equal(at.session_state["setup_seed_df"], df0)


def test_demo_fill_rebuilds_editor_input_with_values():
    """Демо-кнопка «Заполнить тестовыми» — ЯВНОЕ действие: df пересобирается."""
    at = _app_with_seed_design()
    [w for w in at.button if w.key == "setup_fill_demo"][0].click().run()
    assert not at.exception
    df = at.session_state["setup_seed_df"]
    Y = np.asarray(at.session_state["setup_seed_Y"], float)
    assert not df[LAB_COLS].isna().any().any()
    for j, col in enumerate(LAB_COLS):
        assert np.allclose(np.asarray(df[col], float), np.round(Y[:, j], 4))


def test_batch_change_rebuilds_input_and_keeps_draft():
    """Смена размера пробы (сигнатуры) пересобирает df, вливая черновик Y."""
    at = _app_with_seed_design()
    Y = np.full((6, 3), np.nan)
    Y[0, 1] = 6.6247
    at.session_state["setup_seed_Y"] = Y
    at.number_input(key="setup_seed_batch").set_value(2.0).run()
    assert not at.exception
    df = at.session_state["setup_seed_df"]
    # появились столбцы расхода сырья (сигнатура сменилась → пересборка)
    assert any("(кг)" in c for c in df.columns)
    # черновик пережил пересборку: gloss №1 = 6.6247
    assert float(df["gloss (lab)"].iloc[0]) == pytest.approx(6.6247)