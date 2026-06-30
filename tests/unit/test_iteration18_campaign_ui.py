# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 18 — ШАГ 4: Streamlit-вкладка «Кампания» (ТЗ v1.1, §16/§16.1).

Две части (как у существующих *_ui тестов):
  * ЧИСТЫЕ хелперы (build_demo_campaign_runner / *_dataframe) — без Streamlit;
  * headless AppTest: создать демо-кампанию во вкладке и сменить роль ρ без
    падений приложения (И-5 виден: premium ρ=PRICE_INPUT, rho_focus ρ=OPTIMIZED).
"""
import os
import warnings

import pytest
from sklearn.exceptions import ConvergenceWarning

from src.design.branches import ROLE_OPTIMIZED, ROLE_PRICE_INPUT
from src.apps import campaign as cv
from src.apps.campaign_ui import (build_demo_campaign_runner,
                                   role_table_dataframe, spawn_review_dataframe)

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ======================================================================
# Чистые хелперы (без Streamlit)
# ======================================================================
def test_build_demo_runner_contrasting_roles():
    r = build_demo_campaign_runner(n_seed=12)
    assert set(r.branches) == {"premium", "rho_focus"}
    # premium: ρ питает цену, не цель ⇒ PRICE_INPUT, канал живой
    assert r.response_role("premium", "rho") == ROLE_PRICE_INPUT
    assert r.price_channel_suppressed("premium") is False
    # rho_focus: ρ И цель, И цена ⇒ OPTIMIZED, канал занулён (И-5/Гр-1)
    assert r.response_role("rho_focus", "rho") == ROLE_OPTIMIZED
    assert r.price_channel_suppressed("rho_focus") is True
    # общий пул и суррогаты на каждое свойство
    assert set(r.surrogates) == {"strength", "gloss", "rho"}
    assert len(r.points) == 12


def test_role_table_dataframe_columns_and_money():
    r = build_demo_campaign_runner(n_seed=12)
    rep = cv.branch_role_report(r, "rho_focus")
    df = role_table_dataframe(rep)
    assert {"отклик", "роль", "ден. канал ρ", "покрытие"} <= set(df.columns)
    rho_row = df[df["отклик"] == "rho"].iloc[0]
    assert "ZEROED" in rho_row["ден. канал ρ"]      # OPTIMIZED ⇒ занулён


def test_spawn_review_dataframe_marks_objective_override():
    r = build_demo_campaign_runner(n_seed=12)
    ctrl = cv.CampaignController(r)
    from src.optimize.desirability import DesirabilitySpec
    rev = ctrl.preview_spawn(
        "premium", new_goals={"rho": DesirabilitySpec("min", low=0.5, high=1.5)})
    df = spawn_review_dataframe(rev)
    rho_row = df[df["отклик"] == "rho"].iloc[0]
    assert rho_row["изменение"] == "изменено объективом ветки"
    assert "ZEROED" in rho_row["ден. канал ρ (ребёнок)"]


# ======================================================================
# headless AppTest — вкладка «Кампания»
# ======================================================================
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(_REPO, "src", "apps", "streamlit_app.py")


def _click(at, key):
    b = [w for w in at.button if w.key == key]
    assert b, f"кнопка {key} не найдена"
    b[0].click().run()


def test_campaign_tab_demo_and_switch_role():
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception

    # создать демо-кампанию (кнопка во вкладке «Кампания»)
    _click(at, "camp_create")
    assert not at.exception
    ctrl = at.session_state["campaign_ctrl"]
    assert set(ctrl.runner.branches) == {"premium", "rho_focus"}

    # дефолтная линза — premium (ρ=PRICE_INPUT, канал живой)
    assert ctrl.runner.response_role("premium", "rho") == ROLE_PRICE_INPUT
    assert ctrl.runner.price_channel_suppressed("premium") is False

    # сменить роль ρ → OPTIMIZED (спеки по умолчанию min[0.5,1.5]); канал занулится
    _click(at, "camp_do_switch")
    assert not at.exception
    ctrl = at.session_state["campaign_ctrl"]
    assert ctrl.runner.response_role("premium", "rho") == ROLE_OPTIMIZED
    assert ctrl.runner.price_channel_suppressed("premium") is True
