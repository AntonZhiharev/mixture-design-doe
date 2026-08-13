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
"""Iteration 85 — ЦЕНЫ СЫРЬЯ ВНОСЯТСЯ ПОЗЖЕ СБОРКИ (хвост iter75/iter82).

Отказ, зафиксированный в живой сессии (13.08.2026). Кнопка «🏗 Построить
проект» отказывала сообщением «Нужно 18 цен компонентов [PVC_67 … SA_1860] …
Пустых значений быть не может», и проект не создавался. Разбор показал ДВА
независимых дефекта:

  1. **Рассинхрон источника имён компонентов (баг).** Поле цен в форме сетапа
     размечалось по ТЕКСТОВОМУ полю «Компоненты смеси» (``mix_live``), а гейт
     сборки требовал цены по ``setup_mixture_names``, которая в режиме
     phr-спеки текстовое поле ИГНОРИРУЕТ и берёт ``spec.component_names``. С
     18-компонентной спекой форма просила «3 цены» (дефолт «A, B, C»), сборка
     требовала 18 — ввести правильное число было нельзя в принципе.
  2. **Гейт цен на сборке противоречил iter82.** iter82 объявил правку цен
     операцией ЖИВОГО проекта, но задать цены впервые было негде: сборка без
     полного набора цен отказывала, а панель «💰 Экономика проекта» при пустых
     ценах отсылала назад в форму сетапа. Замкнутый круг; цена сырья приходит
     из снабжения позже, чем начинаются опыты.

Решение (вариант B, решение пользователя от 13.08.2026): состояние «ρ
объявлена, цены НЕ заданы» — ЛЕГАЛЬНОЕ и видимое. Пустой набор ≠ нули:
``project_price_vector() == []`` ⇒ ценовой ноги нет вовсе, ветки рождаются
техническими, себестоимость не считается. Нуль по-прежнему означает «сырьё
бесплатно» и в расчёт идёт. Частичный список — по-прежнему отказ (A0.6).
"""
import numpy as np
import pytest

from src.apps.campaign import CampaignController, project_economics_report
from src.apps.campaign_ui import (build_setup_runner, parse_component_prices,
                                  setup_mixture_names)
from src.apps import campaign_state as cst
from src.design.branches import ROLE_PRICE_INPUT
from src.optimize.desirability import DesirabilitySpec

MIX = ["A", "B", "C"]
PROC = ["T", "P"]
RESP = ["strength", "gloss", "rho"]
PRICES = {"A": 95.0, "B": 200.0, "C": 23.0}
COST_SPEC = DesirabilitySpec("min", low=0.0, high=300.0, weight=0.5)
GOAL = {"strength": DesirabilitySpec("max", low=2.0, high=12.0)}


def _runner(responses=RESP):
    return build_setup_runner(
        mixture_names=MIX, process_names=PROC,
        process_lower=[150.0, 1.0], process_upper=[200.0, 5.0],
        response_names=list(responses), seed=1)


def _measured(runner, n=8):
    X = runner.propose_seed(n)
    Y = np.asarray(runner.oracle.evaluate(X), float)
    runner.commit_seed(X, Y)
    return runner


def _priceless():
    """Проект с объявленной ρ, но БЕЗ цен — состояние «цены ещё не заданы»."""
    r = _measured(_runner())
    r.set_project_economics(rho_property="rho", prices=None,
                            rho_unit="кг/изд", currency_unit="₽",
                            mass_unit="кг", rho_default_role=ROLE_PRICE_INPUT)
    return r


# ======================================================================
# 1. Ядро: «ρ объявлена, цены не заданы» — легальное состояние
# ======================================================================
class TestCoreAcceptsDeferredPrices:

    def test_economics_without_prices_is_accepted(self):
        r = _priceless()
        assert r.economics_enabled is True
        assert r.rho_property == "rho"
        assert r.component_prices == {}
        # единицы и роль ρ сохранены: пропущены только ЦЕНЫ
        assert (r.rho_unit, r.currency_unit, r.mass_unit) == ("кг/изд", "₽", "кг")
        assert r.rho_default_role == ROLE_PRICE_INPUT

    def test_no_prices_means_no_price_leg_at_all(self):
        """Пустой набор ≠ нули: ценовой ноги НЕТ, а не «всё бесплатно»."""
        assert _priceless().project_price_vector() == []

    def test_empty_dict_equivalent_to_none(self):
        r = _measured(_runner())
        r.set_project_economics(rho_property="rho", prices={})
        assert r.component_prices == {}

    def test_zero_prices_are_still_prices(self):
        """Явный нуль — заявление «сырьё бесплатно», он в расчёт ИДЁТ."""
        r = _measured(_runner())
        r.set_project_economics(rho_property="rho",
                                prices={"A": 0.0, "B": 0.0, "C": 0.0})
        assert r.project_price_vector() == [0.0, 0.0, 0.0]
        assert project_economics_report(r)["all_zero"] is True

    def test_partial_prices_still_refused(self):
        """Недобор по-прежнему отказ: нули доопределились бы молча (A0.6)."""
        r = _measured(_runner())
        with pytest.raises(ValueError, match="Не заданы цены"):
            r.set_project_economics(rho_property="rho",
                                    prices={"A": 95.0, "B": 200.0})

    def test_unknown_component_still_refused(self):
        r = _measured(_runner())
        with pytest.raises(KeyError, match="которых нет среди компонентов"):
            r.set_project_economics(rho_property="rho",
                                    prices=dict(PRICES, Z=1.0))

    def test_rho_still_required(self):
        """Послабление касается ТОЛЬКО цен: ρ без имени — отказ."""
        r = _measured(_runner())
        with pytest.raises(ValueError, match="плотности"):
            r.set_project_economics(rho_property="", prices=None)

    def test_priceless_state_survives_roundtrip(self):
        r1 = cst.runner_from_state(cst.runner_to_state(_priceless()))
        assert r1.economics_enabled is True
        assert r1.rho_property == "rho"
        assert r1.component_prices == {}
        assert r1.project_price_vector() == []


# ======================================================================
# 2. Ветки до ввода цен — технические, после ввода — с ценовой ногой
# ======================================================================
class TestBranchesWithoutPrices:

    def test_branch_is_technical_while_prices_unknown(self):
        ctrl = CampaignController(_priceless())
        out = ctrl.create_branch("prod", dict(GOAL), budget=5,
                                 cost_spec=COST_SPEC, branch_id="prod")
        # ногу наследовать не из чего (вектор цен пуст) — и это не падение
        assert out["has_price_leg"] is False
        assert out["price_leg_inherited"] is False

    def test_prices_entered_later_reach_new_branches(self):
        """Цены доводятся штатным iter82-путём, база опытов не страдает."""
        ctrl = CampaignController(_priceless())
        n_points = len(ctrl.runner.points)
        out = ctrl.set_project_prices(PRICES)
        assert ctrl.runner.component_prices == PRICES
        assert out["prices_before"] == {}
        assert len(ctrl.runner.points) == n_points

        after = ctrl.create_branch("later", dict(GOAL), budget=5,
                                   cost_spec=COST_SPEC, branch_id="later")
        assert after["has_price_leg"] is True
        assert after["price_leg_inherited"] is True

    def test_set_project_prices_allowed_on_priceless_project(self):
        """iter82 отказывал без ρ; при живой ρ цены задаются ВПЕРВЫЕ здесь."""
        ctrl = CampaignController(_priceless())
        ctrl.set_project_prices(PRICES)          # не должно бросать
        assert ctrl.runner.project_price_vector() == [95.0, 200.0, 23.0]

    def test_still_refused_when_rho_absent(self):
        """Экономика выключена ⇒ прикладывать цены некуда (контракт iter82)."""
        r = _measured(_runner())
        r.set_project_economics(enabled=False)
        with pytest.raises(ValueError, match="не настроена"):
            CampaignController(r).set_project_prices(PRICES)


# ======================================================================
# 3. Read-model: «цен нет» отличимо от «цены нулевые»
# ======================================================================
class TestReport:

    def test_report_separates_unknown_from_zero(self):
        rep = project_economics_report(_priceless())
        assert rep["enabled"] is True
        assert rep["rho_property"] == "rho"
        assert rep["prices"] == {}
        assert rep["price_vector"] == []
        # ключевое: «не знаю» НЕ маскируется под «всё по нулю»
        assert rep["all_zero"] is False
        assert rep["configured"] is False

    def test_dataframe_lists_components_even_without_prices(self):
        from src.apps.campaign_ui import project_economics_dataframe
        df = project_economics_dataframe(_priceless())
        assert list(df["компонент"]) == MIX


# ======================================================================
# 4. Парсер формы: пусто = «не заданы», недобор = отказ
# ======================================================================
class TestParser:

    def test_empty_text_means_prices_not_set(self):
        assert parse_component_prices("", MIX) == {}
        assert parse_component_prices("   ", MIX) == {}

    def test_full_list_parsed_by_position(self):
        assert parse_component_prices("95, 200, 23", MIX) == PRICES

    def test_partial_list_refused_with_expected_count(self):
        with pytest.raises(ValueError, match="Нужно 3 цен"):
            parse_component_prices("95, 200", MIX)

    def test_refusal_text_offers_the_legal_way_out(self):
        """Сообщение обязано называть выход, а не только запрет (A0.6)."""
        with pytest.raises(ValueError, match="ПУСТЫМ"):
            parse_component_prices("95, 200", MIX)

    def test_non_numeric_still_refused(self):
        with pytest.raises(ValueError, match="Нужно 3 цен"):
            parse_component_prices("95, дорого, 23", MIX)

    def test_zeros_typed_explicitly_are_kept(self):
        """Явные нули — это ЦЕНЫ (сырьё бесплатно), а не «не заданы»."""
        assert parse_component_prices("0, 0, 0", MIX) == {
            "A": 0.0, "B": 0.0, "C": 0.0}


# ======================================================================
# 5. Правка №1: имена компонентов формы = имена, которые требует сборка
# ======================================================================
class TestNamesSourceMatchesBuild:

    def test_phr_spec_names_win_over_text_field(self):
        """Гвоздь бага: в phr-режиме цены спрашиваются по именам СПЕКИ."""
        class _Spec:
            component_names = ["PVC_67", "DINP", "Chalk_1T", "OPE"]

        spec = _Spec()
        names = setup_mixture_names(["A", "B", "C"], spec)
        assert names == spec.component_names
        # ⇒ цены, введённые по именам спеки, проходят гейт без «нужно 18»
        prices = parse_component_prices("70, 30, 12, 0.3", names)
        assert list(prices) == spec.component_names

    def test_without_spec_text_field_is_used(self):
        assert setup_mixture_names(MIX, None) == MIX


# ======================================================================
# 5b. Устаревшее значение поля цен из СОХРАНЁННОГО ЧЕРНОВИКА
# ======================================================================
class TestStalePricesField:
    """Живой отказ 13.08.2026: сообщение приходило не из формы, а с ДИСКА.

    ``setup_draft.json`` ПВХ-проекта нёс ``setup_econ_prices = "0, 0, 0"``
    (записано, когда в составе стояли дефолтные «A, B, C»), а компонентов в
    phr-спеке 18. Черновик возвращается в форму через ``setup_prefill_pending``,
    и Streamlit при существующем ключе игнорирует новый ``value=""`` — поэтому
    ни правка дефолта, ни перезапуск приложения отказ не снимали.
    """

    def test_detects_prices_left_from_another_composition(self):
        from src.apps.campaign_ui import stale_prices_count
        eighteen = [f"C{i}" for i in range(18)]
        assert stale_prices_count("0, 0, 0", eighteen) == 3

    def test_matching_count_is_not_stale(self):
        from src.apps.campaign_ui import stale_prices_count
        assert stale_prices_count("95, 200, 23", MIX) is None

    def test_empty_and_non_numeric_are_not_stale(self):
        from src.apps.campaign_ui import stale_prices_count
        assert stale_prices_count("", MIX) is None
        assert stale_prices_count("   ", MIX) is None
        # нечисловое — забота парсера (явный отказ), а не «устарело»
        assert stale_prices_count("95, дорого, 23", MIX) is None

    def test_zeros_matching_composition_are_kept(self):
        """«0, 0, 0» при ТРЁХ компонентах — законный ввод, не мусор."""
        from src.apps.campaign_ui import stale_prices_count
        assert stale_prices_count("0, 0, 0", MIX) is None

    def test_real_pvc_draft_is_recognised_as_stale(self):
        """Регресс на фактических данных проекта пользователя (18 vs 3)."""
        from src.apps.campaign_ui import stale_prices_count
        pvc = ["PVC_67", "PVC_71", "DINP", "Chalk_1T", "Chalk_95T", "CPE_135A",
               "PBNK_3355", "PMPlus_8", "DL_531", "DL_60", "AKLUB_K_435",
               "OPE", "PF711", "PF711LB", "SBM_55", "TiO2_BLR895", "UV_CSFCP",
               "SA_1860"]
        assert len(pvc) == 18
        assert stale_prices_count("0, 0, 0", pvc) == 3


# ======================================================================
# 6. Сквозной прогон UI: проект СОБИРАЕТСЯ с пустым полем цен
# ======================================================================
pytest.importorskip("streamlit")
import os  # noqa: E402

from streamlit.testing.v1 import AppTest  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(_REPO, "src", "apps", "streamlit_app.py")


def _click(at, key):
    btn = [w for w in at.button if w.key == key]
    assert btn, f"кнопка {key} не найдена"
    btn[0].click().run()
    return at


def test_setup_builds_project_with_empty_prices():
    """Гвоздь iter85: пустое поле цен НЕ блокирует сборку проекта."""
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception
    at.session_state["setup_econ_prices"] = ""
    at.run()
    _click(at, "setup_build")

    assert not at.exception
    assert "campaign_ctrl" in at.session_state          # проект СОБРАН
    r = at.session_state["campaign_ctrl"].runner
    assert r.economics_enabled is True
    assert r.rho_property == "rho"
    assert r.component_prices == {}                     # цены отложены
    assert r.project_price_vector() == []


def test_setup_field_is_empty_by_default_and_states_it_is_allowed():
    """Дефолт — ПУСТО (а не нули) + видимое сообщение, что так можно."""
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception
    fld = [w for w in at.text_input if w.key == "setup_econ_prices"]
    assert fld and str(fld[0].value) == ""
    assert any("не заданы" in str(m.value) for m in at.info)


def test_stale_prices_from_draft_do_not_block_build():
    """Живой отказ: «0, 0, 0» под 18 компонентов приходили из черновика.

    Воспроизводим ИМЕННО тот вход: поле цен несёт 3 значения, а состав — 18
    компонентов (как в ``setup_draft.json`` ПВХ-проекта). Ожидание: поле
    очищается с предупреждением, сборка проходит, цены остаются отложенными.
    """
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception
    eighteen = ", ".join(f"C{i}" for i in range(18))
    # ровно тот путь, которым приходит черновик с диска (streamlit_app:
    # _load_draft_project → setup_prefill_pending → render_setup_form)
    at.session_state["setup_prefill_pending"] = {
        "setup_mix": eighteen,
        "setup_econ_prices": "0, 0, 0",     # значения ПОД ПРЕЖНИЙ состав
    }
    at.run()

    fld = [w for w in at.text_input if w.key == "setup_econ_prices"]
    assert fld and str(fld[0].value) == ""          # устаревшее ОЧИЩЕНО
    assert any("очищено" in str(w.value) for w in at.warning)   # и не молча

    _click(at, "setup_build")
    assert not at.exception
    assert "campaign_ctrl" in at.session_state      # проект СОБРАН
    r = at.session_state["campaign_ctrl"].runner
    assert r.component_prices == {}
    assert len(r.current_schema.mixture_names) == 18


def test_setup_still_refuses_partial_prices():
    """Частичный список — по-прежнему отказ, проект не собирается (A0.6)."""
    at = AppTest.from_file(APP, default_timeout=240).run()
    at.session_state["setup_econ_prices"] = "95, 200"
    at.run()
    _click(at, "setup_build")
    assert not at.exception                             # сообщение, не падение
    assert "campaign_ctrl" not in at.session_state
    assert any("цен компонентов" in str(e.value) for e in at.error)
