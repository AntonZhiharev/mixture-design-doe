# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 23 — C4/финал: единый поток кампании в streamlit_app (§17.6.1).

Снос M1…M8 UI + PipelineRunner из главного потока: приложение теперь запускает
ТОЛЬКО кампанию (сетап → seed → ветки → рабочий стол → эволюция), а salvage
подключён на C1–C3 (ассистент campaign-native, персистентность кампании, Excel).

Две части (как у существующих *_ui тестов):
  * ЧИСТАЯ логика (без Streamlit): campaign-native ответ ассистента строится из
    сводки кампании (`build_campaign_context`) с промптом `campaign_system_prompt`,
    без стадий M1…M8;
  * headless AppTest: приложение рендерится единым потоком, старого сайдбара
    M1…M8 нет; демо-кампания создаётся, сохраняется и загружается (C2).
"""
import os
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.apps import assistant as ai


warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ======================================================================
# Чистая логика: campaign-native ассистент (C1) — без PipelineRunner/M1…M8
# ======================================================================
def test_campaign_assistant_reply_uses_campaign_prompt(monkeypatch):
    captured = {}

    def fake_call(messages, *, model=None):
        captured["messages"] = messages
        return "ответ"

    monkeypatch.setattr(ai, "call_llm", fake_call)
    overview = {
        "property_names": ["strength", "rho"],
        "n_points": 6,
        "origin_counts": {"M2": 6},
        "branches": [{"id": "b1", "name": "premium",
                      "price_channel_suppressed": False}],
    }
    out = ai.campaign_assistant_reply(overview, [], "что делать дальше?")
    assert out == "ответ"

    # системный промпт — КАМПАНИЯ, а не M1…M8
    sys_msg = captured["messages"][0]["content"]
    assert "КАМПАНИЯ" in sys_msg
    assert "M1" not in sys_msg or "M1…M8" in sys_msg  # M1…M8 лишь в отрицании

    # контекст — campaign-native (mode=campaign, есть ветки/база)
    ctx_msg = captured["messages"][1]["content"]
    assert '"mode": "campaign"' in ctx_msg
    assert '"n_points": 6' in ctx_msg


def test_build_campaign_context_is_campaign_native():
    ctx = ai.build_campaign_context({"property_names": ["y"], "n_points": 3,
                                     "origin_counts": {}, "branches": []})
    assert ctx["mode"] == "campaign"
    # карта UI — единый поток кампании (§17), без стадий M1…M8
    assert "flow" in ctx["ui_guide"]
    assert "M2" not in ctx  # нет метрик стадий pipeline


# ======================================================================
# headless AppTest — единый поток кампании (main() streamlit_app)
# ======================================================================
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.apps import campaign_state as cs  # noqa: E402
from src.apps.streamlit_app import CAMPAIGN_ROOT  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(_REPO, "src", "apps", "streamlit_app.py")


def _click(at, key):
    b = [w for w in at.button if w.key == key]
    assert b, f"кнопка {key} не найдена"
    b[0].click().run()


def test_campaign_is_single_flow_no_pipeline_ui():
    """Приложение = единый поток кампании; сайдбара/кнопок M1…M8 больше нет."""
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception

    # старого pipeline-UI нет: ни кнопки создания проекта M1…M8, ни прогонов стадий
    keys = {w.key for w in at.button}
    assert "run_M1" not in keys
    assert "load_battle" not in keys      # «🧪 Заполнить тестовыми» (пресет M1…M8)
    # есть кампейновые кнопки: создать демо-кампанию (вкладка) + персистентность
    assert "camp_create" in keys
    assert "save_campaign" in keys and "load_campaign" in keys


def _serializable_ctrl():
    """Живая кампания с СЕРИАЛИЗУЕМОЙ ценовой ногой (linear_price_fn, C2).

    Демо-кампания использует ``demo_price_fn`` БЕЗ дескриптора ``price_spec`` —
    C2 (A0.6) осознанно отказывает такой сохранять. Для проверки проводки
    save/load собираем кампанию сетапом + линейной ценой (несёт price_spec).
    """
    from src.apps import campaign as cv
    from src.apps.campaign_ui import build_setup_runner
    from src.optimize.desirability import DesirabilitySpec

    runner = build_setup_runner(
        mixture_names=["A", "B", "C"], process_names=["T", "P"],
        process_lower=[0.0, 0.0], process_upper=[1.0, 1.0],
        response_names=["strength", "gloss", "rho"], seed=1)
    ctrl = cv.CampaignController(runner)
    Xseed = np.asarray(ctrl.propose_seed(12, seed=1), float)
    Yseed = np.vstack([runner._measure(np.asarray(x, float)) for x in Xseed])
    ctrl.commit_seed(Xseed, Yseed)
    ctrl.create_branch(
        "premium",
        {"strength": DesirabilitySpec("max", low=2.0, high=12.0)},
        branch_id="premium", budget=20,
        price_fn=cs.linear_price_fn([95.0, 200.0, 23.0]),
        cost_spec=DesirabilitySpec("min", low=0.0, high=300.0, weight=0.5),
        rho_property="rho")
    return ctrl


def test_campaign_save_and_load_roundtrip():
    """Кампанию можно сохранить (C2) через сайдбар и загрузить в свежей сессии."""
    name = "regr_c4_camp"
    # чистим возможный хвост прошлых прогонов
    try:
        cs.delete_campaign(CAMPAIGN_ROOT, name)
    except Exception:  # noqa: BLE001
        pass
    try:
        ctrl = _serializable_ctrl()
        n_pts = len(ctrl.runner.points)

        # инъектируем готовую кампанию в сессию приложения и сохраняем (сайдбар)
        at = AppTest.from_file(APP, default_timeout=300)
        at.session_state["campaign_ctrl"] = ctrl
        at.run()
        assert not at.exception
        at.text_input(key="campaign_name").set_value(name).run()
        _click(at, "save_campaign")
        assert not at.exception
        assert name in cs.list_campaigns(CAMPAIGN_ROOT)

        # загрузить кампанию в свежей сессии
        at2 = AppTest.from_file(APP, default_timeout=300).run()
        at2.selectbox(key="campaign_select").set_value(name).run()
        _click(at2, "load_campaign")
        assert not at2.exception
        loaded = at2.session_state["campaign_ctrl"]
        assert set(loaded.runner.branches) == {"premium"}
        assert len(loaded.runner.points) == n_pts
        # ценовая нога пережила save/load (роль ρ = PRICE_INPUT, канал живой)
        assert loaded.runner.price_channel_suppressed("premium") is False
    finally:
        try:
            cs.delete_campaign(CAMPAIGN_ROOT, name)
        except Exception:  # noqa: BLE001
            pass


