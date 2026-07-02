# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 19 / §16.6 E+ — UI мультицели (§16.3) и рабочего стола (§16.4).

Канон «логика+тест, потом UI»: логика §16.2–§16.5 уже покрыта
(``test_iteration19_*``); здесь проверяем ТОНКИЙ UI кампании
(``campaign_ui.render_campaign`` поверх ``CampaignController``). Две части, как в
существующих ``*_ui`` тестах:

  * ЧИСТЫЕ хелперы (``goal_editor_dataframe`` / ``workbench_points_dataframe``) —
    без Streamlit;
  * headless AppTest: во вкладке «Кампания» задать цель (мультицель, §16.3) и
    прогнать раунд рабочего стола (§16.4) без падения приложения; проверить, что
    общая база выросла и у долитых точек origin-тег ветки (И-1).
"""
import os
import warnings

import pytest
from sklearn.exceptions import ConvergenceWarning

from src.design.branches import ROLE_OPTIMIZED
from src.apps import campaign as cv
from src.apps.campaign_ui import (build_demo_campaign_runner,
                                   goal_editor_dataframe,
                                   workbench_points_dataframe)

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ======================================================================
# Чистые хелперы (без Streamlit)
# ======================================================================
def test_goal_editor_dataframe_lists_multiple_goals():
    r = build_demo_campaign_runner(n_seed=12)
    df = goal_editor_dataframe(r, "premium")           # 2 цели: strength+gloss
    assert {"цель (отклик)", "вид", "low", "high", "target", "вес"} \
        <= set(df.columns)
    assert set(df["цель (отклик)"]) == {"strength", "gloss"}
    assert set(df["вид"]) == {"max"}


def test_workbench_points_dataframe_has_all_properties_and_origin():
    r = build_demo_campaign_runner(n_seed=12)
    ctrl = cv.CampaignController(r)
    res = ctrl.run_round("premium", n_points=3, n_candidates=150)
    df = workbench_points_dataframe(r, res)
    # по ВСЕМ P свойствам + origin-тег ветки (И-1, общий пул)
    assert list(r.property_names) == ["strength", "gloss", "rho"]
    assert {"origin", "strength", "gloss", "rho"} <= set(df.columns)
    assert len(df) == res["added"] == 3
    assert set(df["origin"]) == {"branch:premium"}


# ======================================================================
# headless AppTest — вкладка «Кампания»: мультицель (§16.3) + стол (§16.4)
# ======================================================================
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(_REPO, "src", "apps", "streamlit_app.py")


def _click(at, key):
    b = [w for w in at.button if w.key == key]
    assert b, f"кнопка {key} не найдена"
    b[0].click().run()


def test_campaign_tab_multigoal_editor_adds_goal():
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception
    _click(at, "camp_create")
    assert not at.exception

    # линза по умолчанию — premium; форма цели по умолчанию: rho max[0,10]
    # (selectbox отклика первый — strength; переставим на rho через session_state)
    at.session_state["camp_goal_resp"] = "rho"
    at.session_state["camp_goal_kind"] = "max"
    at.run()
    _click(at, "camp_goal_set")
    assert not at.exception

    ctrl = at.session_state["campaign_ctrl"]
    # rho стал целью premium ⇒ роль OPTIMIZED (§16.0); канал занулился (И-5/Гр-1)
    assert "rho" in ctrl.runner.branches["premium"].goal
    assert ctrl.runner.response_role("premium", "rho") == ROLE_OPTIMIZED
    assert ctrl.runner.price_channel_suppressed("premium") is True


def test_campaign_tab_workbench_round_grows_shared_base():
    # §17.5 (Ш5): рабочий стол — РУЧНОЙ цикл §17.2 (предложить → Y → долить),
    # а не авто-оракул. Демо-ветка «premium» уже создана build_demo_campaign_runner.
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception
    _click(at, "camp_create")
    ctrl = at.session_state["campaign_ctrl"]
    n_hist0 = len(ctrl.runner.points)

    at.session_state["camp_wb_n_premium"] = 3
    at.run()
    _click(at, "camp_wb_propose_premium")          # предложить (read-only)
    assert not at.exception
    assert len(at.session_state["campaign_ctrl"].runner.points) == n_hist0
    _click(at, "camp_wb_fill_premium")             # Y от демо-оракула (A0.6)
    _click(at, "camp_wb_commit_premium")           # долить в общую базу
    assert not at.exception

    ctrl = at.session_state["campaign_ctrl"]
    # база выросла на N; у долитых точек origin-тег ветки (§17.2/§16.4, И-1)
    assert len(ctrl.runner.points) == n_hist0 + 3
    assert ctrl.runner.origin_counts().get("branch:premium", 0) == 3
    # долив запечатал дно undo (измеренную правду не откатить, Тр-7.2/7.3)
    assert ctrl.can_undo() is False


