# Copyright 2026 DOE contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 82 — ПРАВКА ЦЕН СЫРЬЯ НА ЖИВОМ ПРОЕКТЕ (продолжение iter75).

Что было не так до iter82 (наблюдение сессии 12.08.2026):
  * ядро (``set_project_economics``) менять цены умеет в любой момент и базу не
    трогает, но в UI цены жили ТОЛЬКО в форме «🆕 Новый проект», а её кнопка
    зовёт ``build_setup_runner`` — новый раннер с ПУСТОЙ базой. То есть смена
    закупочной цены стоила пользователю всех измеренных опытов;
  * ветка наследует проектную цену ОДИН РАЗ, при рождении (``create_branch``
    кладёт замыкание в ``_branch_cost``) — это СНИМОК. После правки цен старые
    ветки продолжали считать по старым ценам, и увидеть это было негде.

Здесь закрываем оба хвоста ЛОГИКОЙ (UI — следующим слоем):
  * :func:`campaign.project_economics_report` — по каким ценам считает проект и
    каждая ветка, флаг ``stale`` для расхождения;
  * :meth:`CampaignController.set_project_prices` — правка цен на живом проекте
    (база не трогается) + ЯВНЫЙ перенос новых цен на существующие ветки.
"""
import numpy as np
import pytest

from src.apps.campaign import (CampaignController, branch_price_vector,
                               project_economics_report)
from src.apps.campaign_ui import build_setup_runner
from src.apps import campaign_state as cst
from src.design.branches import ROLE_OPTIMIZED, ROLE_PRICE_INPUT
from src.optimize.desirability import DesirabilitySpec

MIX = ["A", "B", "C"]
PROC = ["T", "P"]
RESP = ["strength", "gloss", "rho"]
PRICES = {"A": 95.0, "B": 200.0, "C": 23.0}
PRICES_NEW = {"A": 120.0, "B": 180.0, "C": 30.0}
COST_SPEC = DesirabilitySpec("min", low=0.0, high=300.0, weight=0.5)
GOAL = {"strength": DesirabilitySpec("max", low=2.0, high=12.0)}


def _runner(responses=RESP):
    return build_setup_runner(
        mixture_names=MIX, process_names=PROC,
        process_lower=[150.0, 1.0], process_upper=[200.0, 5.0],
        response_names=list(responses), seed=1)


def _measured(runner, n=8):
    """Снять стартовый план демо-генератором ManualOracle — база непуста."""
    X = runner.propose_seed(n)
    Y = np.asarray(runner.oracle.evaluate(X), float)
    runner.commit_seed(X, Y)
    return runner


def _configured(prices=PRICES, role=ROLE_PRICE_INPUT):
    r = _measured(_runner())
    r.set_project_economics(rho_property="rho", prices=prices,
                            rho_unit="кг/изд", currency_unit="₽",
                            mass_unit="кг", rho_default_role=role)
    return r


def _ctrl_with_branch(prices=PRICES, role=ROLE_PRICE_INPUT):
    r = _configured(prices, role)
    ctrl = CampaignController(r)
    bid = ctrl.create_branch("prod", dict(GOAL), budget=5,
                             cost_spec=COST_SPEC, branch_id="prod")["branch_id"]
    return ctrl, bid


# ======================================================================
# 1. Цены меняются НА ЖИВОМ проекте: база опытов не страдает
# ======================================================================
class TestEditOnLiveProject:

    def test_prices_change_without_touching_measured_base(self):
        """Гвоздь iter82: смена цены НЕ стоит ни одной измеренной точки."""
        ctrl, _ = _ctrl_with_branch()
        r = ctrl.runner
        n_points = len(r.points)
        X_before = np.array(r.X, float, copy=True)
        Y_before = np.array(r.Y, float, copy=True)
        assert n_points > 0                    # иначе тест ничего не доказывает

        out = ctrl.set_project_prices(PRICES_NEW)

        assert r.component_prices == PRICES_NEW
        assert out["prices_before"] == PRICES
        assert out["prices_after"] == PRICES_NEW
        assert len(r.points) == n_points
        assert np.allclose(r.X, X_before)
        assert np.allclose(r.Y, Y_before)

    def test_rho_units_and_role_survive_price_edit(self):
        """Правим ТОЛЬКО цены: ρ, единицы и роль ρ по умолчанию не сбрасываются."""
        ctrl, _ = _ctrl_with_branch(role=ROLE_OPTIMIZED)
        ctrl.set_project_prices(PRICES_NEW)
        r = ctrl.runner
        assert (r.rho_property, r.rho_unit) == ("rho", "кг/изд")
        assert (r.currency_unit, r.mass_unit) == ("₽", "кг")
        assert r.rho_default_role == ROLE_OPTIMIZED
        assert r.economics_enabled is True

    def test_price_vector_follows_new_prices(self):
        ctrl, _ = _ctrl_with_branch()
        ctrl.set_project_prices(PRICES_NEW)
        assert ctrl.runner.project_price_vector() == [120.0, 180.0, 30.0]

    def test_new_branch_born_after_edit_uses_new_prices(self):
        """Без переноса новые цены действуют на ветки, созданные ПОСЛЕ правки."""
        ctrl, old_bid = _ctrl_with_branch()
        ctrl.set_project_prices(PRICES_NEW)
        new_bid = ctrl.create_branch("prod2", dict(GOAL), budget=5,
                                     cost_spec=COST_SPEC,
                                     branch_id="prod2")["branch_id"]
        assert branch_price_vector(ctrl.runner, new_bid) == [120.0, 180.0, 30.0]
        # ...а старая ветка всё ещё на своём снимке — это и есть stale
        assert branch_price_vector(ctrl.runner, old_bid) == [95.0, 200.0, 23.0]

    def test_edit_survives_save_load(self, tmp_path):
        ctrl, _ = _ctrl_with_branch()
        ctrl.set_project_prices(PRICES_NEW)
        cst.save_campaign(ctrl.runner, str(tmp_path), "p82")
        r1 = cst.load_campaign(str(tmp_path), "p82")
        assert r1.component_prices == PRICES_NEW


# ======================================================================
# 2. Существующие ветки: расхождение ВИДНО, перенос — ЯВНЫЙ (A0.6)
# ======================================================================
class TestBranchesStaleAndTransfer:

    def test_branch_keeps_old_snapshot_until_asked(self):
        """По умолчанию ветка НЕ переписывается: её цена — часть её истории."""
        ctrl, bid = _ctrl_with_branch()
        out = ctrl.set_project_prices(PRICES_NEW)
        assert out["applied_to_branches"] is False
        assert out["branches_updated"] == []
        assert branch_price_vector(ctrl.runner, bid) == [95.0, 200.0, 23.0]
        assert out["n_stale_branches"] == 1

    def test_report_marks_stale_branch(self):
        ctrl, bid = _ctrl_with_branch()
        ctrl.set_project_prices(PRICES_NEW)
        rep = ctrl.economics_report()
        assert rep["price_vector"] == [120.0, 180.0, 30.0]
        row = next(b for b in rep["branches"] if b["branch_id"] == bid)
        assert row["stale"] is True
        assert row["prices"] == [95.0, 200.0, 23.0]

    def test_apply_to_branches_rewrites_leg_and_clears_stale(self):
        ctrl, bid = _ctrl_with_branch()
        out = ctrl.set_project_prices(PRICES_NEW, apply_to_branches=True)
        assert out["applied_to_branches"] is True
        assert [u["branch_id"] for u in out["branches_updated"]] == [bid]
        assert branch_price_vector(ctrl.runner, bid) == [120.0, 180.0, 30.0]
        assert out["n_stale_branches"] == 0
        assert ctrl.economics_report()["n_stale_branches"] == 0

    def test_transfer_rescores_d_best(self):
        """Перенос меняет ОЦЕНКУ (d_best), а не измеренную правду (И-1)."""
        ctrl, bid = _ctrl_with_branch()
        n_points = len(ctrl.runner.points)
        # цены задраны в 20 раз: себестоимость выходит за high=300 ⇒ d падает
        expensive = {k: v * 20.0 for k, v in PRICES.items()}
        out = ctrl.set_project_prices(expensive, apply_to_branches=True)
        upd = out["branches_updated"][0]
        assert upd["d_best_after"] < upd["d_best_before"]
        assert len(ctrl.runner.points) == n_points

    def test_transfer_can_target_subset(self):
        ctrl, bid1 = _ctrl_with_branch()
        bid2 = ctrl.create_branch("prod2", dict(GOAL), budget=5,
                                  cost_spec=COST_SPEC,
                                  branch_id="prod2")["branch_id"]
        ctrl.set_project_prices(PRICES_NEW, apply_to_branches=True,
                                branch_ids=[bid2])
        assert branch_price_vector(ctrl.runner, bid2) == [120.0, 180.0, 30.0]
        assert branch_price_vector(ctrl.runner, bid1) == [95.0, 200.0, 23.0]

    def test_transfer_refuses_unknown_branch(self):
        ctrl, _ = _ctrl_with_branch()
        with pytest.raises(KeyError, match="Нет веток"):
            ctrl.set_project_prices(PRICES_NEW, apply_to_branches=True,
                                    branch_ids=["нет-такой"])

    def test_technical_branch_is_skipped_not_broken(self):
        """Ветка без ценовой ноги переносом не затрагивается (не падаем)."""
        ctrl, bid = _ctrl_with_branch()
        tech = ctrl.create_branch("tech", dict(GOAL), budget=5,
                                  branch_id="tech")
        assert tech["has_price_leg"] is False
        out = ctrl.set_project_prices(PRICES_NEW, apply_to_branches=True)
        assert [u["branch_id"] for u in out["branches_updated"]] == [bid]
        assert branch_price_vector(ctrl.runner, "tech") is None


# ======================================================================
# 3. Отказы и предупреждения (A0.6): молчаливой дыры в цене нет
# ======================================================================
class TestRefusals:

    def test_refuses_when_economics_not_configured(self):
        """Экономика без ρ — цены прикладывать некуда, это явный отказ."""
        r = _measured(_runner())
        r.set_project_economics(enabled=False)
        with pytest.raises(ValueError, match="не настроена"):
            CampaignController(r).set_project_prices(PRICES_NEW)

    def test_refuses_partial_prices(self):
        ctrl, _ = _ctrl_with_branch()
        with pytest.raises(ValueError, match="Не заданы цены"):
            ctrl.set_project_prices({"A": 1.0, "B": 2.0})
        # отказ не оставил проект в половинчатом состоянии
        assert ctrl.runner.component_prices == PRICES

    def test_refuses_unknown_component(self):
        ctrl, _ = _ctrl_with_branch()
        with pytest.raises(KeyError):
            ctrl.set_project_prices({**PRICES, "Z": 1.0})

    def test_refuses_negative_price(self):
        ctrl, _ = _ctrl_with_branch()
        with pytest.raises(ValueError, match="неотрицательной"):
            ctrl.set_project_prices({**PRICES, "A": -1.0})

    def test_all_zero_prices_are_flagged_not_hidden(self):
        """Нули допустимы явно, но отчёт обязан о них сказать."""
        ctrl, _ = _ctrl_with_branch()
        out = ctrl.set_project_prices({"A": 0.0, "B": 0.0, "C": 0.0})
        assert out["all_zero"] is True
        assert ctrl.economics_report()["all_zero"] is True

    def test_undo_stack_is_sealed_after_price_edit(self):
        """Стек undo нёс СТАРУЮ ногу: откат вернул бы старую цену молча."""
        ctrl, bid = _ctrl_with_branch()
        ctrl.set_desirability(bid, "strength",
                              DesirabilitySpec("max", low=1.0, high=10.0))
        assert ctrl.can_undo() is True
        out = ctrl.set_project_prices(PRICES_NEW)
        assert out["undo_available"] is False
        assert ctrl.can_undo() is False


# ======================================================================
# 4. Read-model: состояние экономики читается и БЕЗ веток
# ======================================================================
class TestReport:

    def test_report_on_project_without_branches(self):
        rep = project_economics_report(_configured())
        assert rep["enabled"] is True and rep["configured"] is True
        assert rep["rho_property"] == "rho"
        assert rep["prices"] == PRICES
        assert rep["mixture_names"] == MIX
        assert rep["branches"] == [] and rep["n_stale_branches"] == 0
        assert rep["n_points"] > 0

    def test_report_on_disabled_economics(self):
        r = _measured(_runner())
        r.set_project_economics(enabled=False)
        rep = project_economics_report(r)
        assert rep["enabled"] is False and rep["configured"] is False
        assert rep["price_vector"] == [] and rep["all_zero"] is False

    def test_report_shows_money_channel_of_branch(self):
        """ρ=OPTIMIZED ⇒ денежный канал занулён (И-5) — видно в отчёте."""
        ctrl, bid = _ctrl_with_branch(role=ROLE_OPTIMIZED)
        row = next(b for b in ctrl.economics_report()["branches"]
                   if b["branch_id"] == bid)
        assert row["money_channel"] == "zeroed"

    def test_branch_price_vector_none_for_opaque_price_fn(self):
        """Нелинейная/непрозрачная цена ⇒ «не знаю», а не выдуманные нули."""
        ctrl, bid = _ctrl_with_branch()
        ctrl.runner.set_branch_cost(bid, lambda Xc: np.zeros(len(Xc)),
                                    COST_SPEC, rho_property="rho")
        assert branch_price_vector(ctrl.runner, bid) is None
        row = next(b for b in ctrl.economics_report()["branches"]
                   if b["branch_id"] == bid)
        assert row["prices"] is None and row["stale"] is False


# ======================================================================
# 5. UI-слой: таблицы и отчёт — чистые функции (без Streamlit)
# ======================================================================
class TestUiPureHelpers:

    def test_economics_dataframe_reads_engine(self):
        from src.apps.campaign_ui import project_economics_dataframe
        df = project_economics_dataframe(_configured())
        assert list(df["компонент"]) == MIX
        assert list(df["цена, ₽/кг"]) == [95.0, 200.0, 23.0]

    def test_branch_prices_dataframe_marks_stale(self):
        from src.apps.campaign_ui import branch_prices_dataframe
        ctrl, bid = _ctrl_with_branch()
        ctrl.set_project_prices(PRICES_NEW)
        df = branch_prices_dataframe(ctrl.runner)
        row = df[df["id"] == bid].iloc[0]
        assert row["цены"] == "95, 200, 23"
        assert row["актуальность"] == "устарели"

    def test_message_lists_changed_positions_and_stale_warning(self):
        from src.apps.campaign_ui import price_edit_message
        ctrl, _ = _ctrl_with_branch()
        msg = price_edit_message(ctrl.set_project_prices(PRICES_NEW))
        assert "A: 95 → 120" in msg
        assert "C: 23 → 30" in msg
        assert "B: 200 → 180" in msg
        assert "СТАРЫМ ценам" in msg          # ветка осталась на снимке
        assert "База опытов не изменилась" in msg

    def test_message_reports_branch_transfer(self):
        from src.apps.campaign_ui import price_edit_message
        ctrl, _ = _ctrl_with_branch()
        msg = price_edit_message(
            ctrl.set_project_prices(PRICES_NEW, apply_to_branches=True))
        assert "Ветки переведены на новые цены" in msg
        assert "d_best" in msg


# ======================================================================
# 6. Сквозной прогон UI: цены правятся БЕЗ пересборки проекта
# ======================================================================
pytest.importorskip("streamlit")
import os  # noqa: E402

from streamlit.testing.v1 import AppTest  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(_REPO, "src", "apps", "streamlit_app.py")


def _built_app():
    """Собрать проект формой сетапа с ценами и снять стартовый план.

    Закладку «🌱 Старт» выставляем ЯВНО: измеренный seed меняет фазу проекта, и
    рабочая область штатно открывает «🌿 Ветки» (``workspace.decide_tab``) — на
    ней панели настроек проекта нет. Панель экономики живёт на «Старте», которая
    доступна всегда (это вход в проект).
    """
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.session_state["setup_econ_prices"] = "95, 200, 23"
    at.run()
    btn = [w for w in at.button if w.key == "setup_build"]
    assert btn, "кнопка setup_build не найдена"
    btn[0].click().run()
    assert not at.exception
    runner = at.session_state["campaign_ctrl"].runner
    _measured(runner)                 # база непуста — иначе нечего терять
    at.session_state["ws_tab"] = "start"
    at.session_state["ws_phase"] = "measured"
    at.run()
    return at, runner


def test_price_form_is_present_on_live_project():
    at, _ = _built_app()
    assert not at.exception
    keys = {w.key for w in at.text_input} | {w.key for w in at.checkbox}
    assert "proj_prices_txt" in keys
    assert "proj_prices_to_branches" in keys


def test_price_edit_from_ui_keeps_measured_base():
    """Гвоздь: правка цен в UI не обнуляет базу (в отличие от пересборки)."""
    at, runner = _built_app()
    n_points = len(runner.points)
    assert n_points > 0

    at.text_input(key="proj_prices_txt").set_value("120, 180, 30").run()
    btn = [w for w in at.button if "Применить цены" in str(w.label)]
    assert btn, "кнопка «Применить цены» не найдена"
    btn[0].click().run()
    assert not at.exception

    r = at.session_state["campaign_ctrl"].runner
    assert r.component_prices == PRICES_NEW
    assert len(r.points) == n_points          # база жива
