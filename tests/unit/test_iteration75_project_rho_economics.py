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
"""Iteration 75 — ЭКОНОМИКА ПРОЕКТА: плотность ρ и цены сырья (§3/§15.6).

Требование технолога (решение сессии 12.08.2026): плотность ρ входит в экономику
проекта ПО УМОЛЧАНИЮ, поэтому объявляется на СТАРТОВОЙ странице как проектный
атрибут, а не выбирается заново в каждой ветке. Отключить её можно ЯВНО, и так же
явно — перевести в ЦЕЛЕВОЙ параметр (ядро умеет: ``switch_role``, роли §5).

Что было не так до iter75:
  * ρ и цены компонентов жили ТОЛЬКО в ``_branch_cost[branch_id]`` — цены сырья
    вводились в каждой ветке заново, и одно и то же сырьё могло стоить разное
    в разных ветках одной кампании (факт проекта, выданный за намерение ветки);
  * на стартовой странице ρ не было видно вовсе: она пряталась в дефолтной
    строке откликов, а её экономический смысл всплывал только в форме ветки.

Единицы (решение той же сессии): масса цены и масса ρ — ОДНА И ТА ЖЕ единица,
поэтому ``price_изд = price_состав·ρ`` даёт валюту за изделие без переводных
коэффициентов (см. ``price_per_item``, §3 physics-трактовка — УМНОЖЕНИЕ).
"""
import numpy as np
import pytest

from src.apps.campaign import CampaignController
from src.apps.campaign_ui import build_setup_runner
from src.apps import campaign_state as cst
from src.design.branches import ROLE_OPTIMIZED, ROLE_PRICE_INPUT
from src.optimize.desirability import DesirabilitySpec

MIX = ["A", "B", "C"]
PROC = ["T", "P"]
RESP = ["strength", "gloss", "rho"]
PRICES = {"A": 95.0, "B": 200.0, "C": 23.0}


def _runner(responses=RESP):
    """Проект формы сетапа: 3 компонента × 2 процесс-оси, пустая база."""
    return build_setup_runner(
        mixture_names=MIX, process_names=PROC,
        process_lower=[150.0, 1.0], process_upper=[200.0, 5.0],
        response_names=list(responses), seed=1)


def _measured(runner, n=8):
    """Снять стартовый план демо-оракулом (ManualOracle.evaluate) — база непуста."""
    X = runner.propose_seed(n)
    Y = np.asarray(runner.oracle.evaluate(X), float)
    runner.commit_seed(X, Y)
    return runner


# ======================================================================
# 1. Проектный уровень: ρ и цены живут на раннере, а не в ветке
# ======================================================================
class TestProjectLevelEconomics:

    def test_enabled_by_default_but_unconfigured(self):
        """Экономика включена по умолчанию — но ρ пока не объявлена."""
        r = _runner()
        assert r.economics_enabled is True
        assert r.rho_property == ""
        assert r.component_prices == {}
        # роль ρ по умолчанию — вход себестоимости (не цель)
        assert r.rho_default_role == ROLE_PRICE_INPUT

    def test_set_project_economics_stores_rho_prices_units(self):
        r = _runner()
        r.set_project_economics(rho_property="rho", prices=PRICES,
                                rho_unit="кг/изд", currency_unit="₽",
                                mass_unit="кг")
        assert r.economics_enabled is True
        assert r.rho_property == "rho"
        assert r.component_prices == PRICES
        assert (r.rho_unit, r.currency_unit, r.mass_unit) == ("кг/изд", "₽", "кг")

    def test_price_vector_follows_current_phase_axes_order(self):
        """Вектор цен идёт В ПОРЯДКЕ mixture-осей текущей фазы (нога линейна)."""
        r = _runner()
        r.set_project_economics(rho_property="rho", prices=PRICES)
        assert r.project_price_vector() == [95.0, 200.0, 23.0]

    def test_disable_is_explicit_and_clears_everything(self):
        """Выключение — ЯВНОЕ: ρ/цены/единицы обнуляются (A0.6, не молча)."""
        r = _runner()
        r.set_project_economics(rho_property="rho", prices=PRICES,
                                mass_unit="кг")
        r.set_project_economics(enabled=False)
        assert r.economics_enabled is False
        assert r.rho_property == ""
        assert r.component_prices == {}
        assert r.mass_unit == ""
        assert r.project_price_vector() == []


# ======================================================================
# 2. Отказы (A0.6): молчаливая дыра в экономике недопустима
# ======================================================================
class TestRefusals:

    def test_refuses_rho_not_among_responses(self):
        """ρ обязана быть ОТКЛИКОМ (§3: полноценное GP-свойство)."""
        r = _runner(responses=["strength", "gloss"])
        with pytest.raises(KeyError, match="не среди откликов"):
            r.set_project_economics(rho_property="rho", prices=PRICES)

    def test_refuses_enabled_without_rho(self):
        r = _runner()
        with pytest.raises(ValueError, match="плотности"):
            r.set_project_economics(rho_property="", prices=PRICES)

    def test_refuses_partial_prices(self):
        """Пропущенная цена НЕ доопределяется нулём — иначе цена занижена молча."""
        r = _runner()
        with pytest.raises(ValueError, match="Не заданы цены"):
            r.set_project_economics(rho_property="rho",
                                    prices={"A": 95.0, "B": 200.0})

    def test_refuses_unknown_component_in_prices(self):
        r = _runner()
        with pytest.raises(KeyError, match="которых нет среди компонентов"):
            r.set_project_economics(rho_property="rho",
                                    prices=dict(PRICES, Z=1.0))

    def test_refuses_negative_or_nonfinite_price(self):
        r = _runner()
        with pytest.raises(ValueError, match="неотрицательной"):
            r.set_project_economics(rho_property="rho",
                                    prices=dict(PRICES, A=-1.0))
        with pytest.raises(ValueError, match="неотрицательной"):
            r.set_project_economics(rho_property="rho",
                                    prices=dict(PRICES, A=float("inf")))

    def test_refuses_unknown_default_role(self):
        r = _runner()
        with pytest.raises(ValueError, match="Роль ρ"):
            r.set_project_economics(rho_property="rho", prices=PRICES,
                                    rho_default_role="reference")


# ======================================================================
# 3. Ветка НАСЛЕДУЕТ проектную ногу: цены не вводятся заново
# ======================================================================
class TestBranchInheritsProjectEconomics:

    def test_branch_inherits_price_leg_without_own_price_fn(self):
        r = _measured(_runner())
        r.set_project_economics(rho_property="rho", prices=PRICES,
                                mass_unit="кг", currency_unit="₽")
        ctrl = CampaignController(r)

        out = ctrl.create_branch(
            "premium", {"strength": DesirabilitySpec("max", low=2.0, high=12.0)},
            budget=10,
            cost_spec=DesirabilitySpec("min", low=0.0, high=300.0, weight=0.5))

        assert out["has_price_leg"] is True
        assert out["price_leg_inherited"] is True
        assert out["rho_property"] == "rho"
        # ρ НЕ в цели ⇒ роль PRICE_INPUT ⇒ денежный канал ЖИВОЙ (И-5)
        assert out["price_channel_suppressed"] is False
        assert out["rho_goal_auto"] is False

    def test_inherited_prices_are_identical_across_branches(self):
        """Главный смысл правки: сырьё стоит ОДИНАКОВО во всех ветках проекта."""
        r = _measured(_runner())
        r.set_project_economics(rho_property="rho", prices=PRICES)
        ctrl = CampaignController(r)
        spec = DesirabilitySpec("min", low=0.0, high=300.0, weight=0.5)
        bids = [ctrl.create_branch(
            nm, {"strength": DesirabilitySpec("max", low=2.0, high=12.0)},
            budget=5, cost_spec=spec)["branch_id"]
            for nm in ("premium", "economy")]

        vecs = [r._branch_cost[b]["price_fn"].price_spec["prices"] for b in bids]
        assert vecs[0] == vecs[1] == [95.0, 200.0, 23.0]

    def test_no_cost_spec_means_technical_branch(self):
        """Без порога себестоимости ветка остаётся технической (не отказ)."""
        r = _measured(_runner())
        r.set_project_economics(rho_property="rho", prices=PRICES)
        ctrl = CampaignController(r)
        out = ctrl.create_branch(
            "tech", {"gloss": DesirabilitySpec("max", low=1.0, high=13.0)},
            budget=5)
        assert out["has_price_leg"] is False
        assert out["price_leg_inherited"] is False

    def test_economics_disabled_no_inheritance(self):
        r = _measured(_runner())
        r.set_project_economics(enabled=False)
        ctrl = CampaignController(r)
        out = ctrl.create_branch(
            "tech", {"gloss": DesirabilitySpec("max", low=1.0, high=13.0)},
            budget=5,
            cost_spec=DesirabilitySpec("min", low=0.0, high=300.0))
        assert out["has_price_leg"] is False

    def test_explicit_branch_price_fn_wins_over_project(self):
        """Своя нога ветки не подменяется проектной (явное важнее дефолта)."""
        r = _measured(_runner())
        r.set_project_economics(rho_property="rho", prices=PRICES)
        ctrl = CampaignController(r)
        own = cst.linear_price_fn([1.0, 2.0, 3.0])
        out = ctrl.create_branch(
            "own", {"gloss": DesirabilitySpec("max", low=1.0, high=13.0)},
            budget=5, price_fn=own,
            cost_spec=DesirabilitySpec("min", low=0.0, high=300.0),
            rho_property="rho")
        assert out["price_leg_inherited"] is False
        cfg = r._branch_cost[out["branch_id"]]
        assert cfg["price_fn"].price_spec["prices"] == [1.0, 2.0, 3.0]


# ======================================================================
# 4. ρ как ЦЕЛЕВОЙ параметр: явное включение через проектную политику
# ======================================================================
class TestRhoAsTargetParameter:

    def test_default_role_optimized_adds_rho_goal_and_zeroes_channel(self):
        r = _measured(_runner())
        r.set_project_economics(rho_property="rho", prices=PRICES,
                                rho_default_role=ROLE_OPTIMIZED)
        ctrl = CampaignController(r)
        out = ctrl.create_branch(
            "rho_focus",
            {"strength": DesirabilitySpec("max", low=2.0, high=12.0)},
            budget=10,
            cost_spec=DesirabilitySpec("min", low=0.0, high=300.0, weight=0.5))
        bid = out["branch_id"]

        assert out["rho_goal_auto"] is True
        assert "rho" in r.branches[bid].goal
        # ρ в цели ⇒ роль OPTIMIZED ⇒ денежный канал ЗАНУЛЁН (И-5/Гр-1)
        assert r.response_role(bid, "rho") == ROLE_OPTIMIZED
        assert out["price_channel_suppressed"] is True

    def test_auto_goal_is_min_within_measured_range(self):
        """Дефолтная цель по ρ — min в диапазоне ИЗМЕРЕННЫХ значений."""
        r = _measured(_runner())
        r.set_project_economics(rho_property="rho", prices=PRICES,
                                rho_default_role=ROLE_OPTIMIZED)
        ctrl = CampaignController(r)
        out = ctrl.create_branch(
            "rf", {"strength": DesirabilitySpec("max", low=2.0, high=12.0)},
            budget=5, cost_spec=DesirabilitySpec("min", low=0.0, high=300.0))

        spec = r.branches[out["branch_id"]].goal["rho"]
        col = np.asarray(r.Y, float)[:, r.prop_index["rho"]]
        assert spec.kind == "min"
        assert spec.low == pytest.approx(float(col.min()))
        assert spec.high == pytest.approx(float(col.max()))

    def test_explicit_rho_goal_not_overwritten(self):
        """Заданную пользователем цель по ρ авто-дефолт НЕ перебивает."""
        r = _measured(_runner())
        r.set_project_economics(rho_property="rho", prices=PRICES,
                                rho_default_role=ROLE_OPTIMIZED)
        ctrl = CampaignController(r)
        mine = DesirabilitySpec("min", low=0.5, high=1.5, weight=2.0)
        out = ctrl.create_branch(
            "rf", {"strength": DesirabilitySpec("max", low=2.0, high=12.0),
                   "rho": mine},
            budget=5, cost_spec=DesirabilitySpec("min", low=0.0, high=300.0))
        assert out["rho_goal_auto"] is False
        assert r.branches[out["branch_id"]].goal["rho"] == mine

    def test_switch_role_still_works_on_inherited_leg(self):
        """Ядро §5: роль ρ переключается и на НАСЛЕДОВАННОЙ ноге."""
        r = _measured(_runner())
        r.set_project_economics(rho_property="rho", prices=PRICES)
        ctrl = CampaignController(r)
        bid = ctrl.create_branch(
            "b", {"strength": DesirabilitySpec("max", low=2.0, high=12.0)},
            budget=5,
            cost_spec=DesirabilitySpec("min", low=0.0,
                                       high=300.0))["branch_id"]
        assert r.response_role(bid, "rho") == ROLE_PRICE_INPUT

        res = ctrl.switch_role(bid, "rho", ROLE_OPTIMIZED,
                               spec=DesirabilitySpec("min", low=0.5, high=1.5))
        assert res["price_channel_suppressed"] is True
        assert r.response_role(bid, "rho") == ROLE_OPTIMIZED


# ======================================================================
# 5. Персистентность: экономика проекта переживает save/load
# ======================================================================
class TestPersistence:

    def test_roundtrip_preserves_project_economics(self):
        r0 = _measured(_runner())
        r0.set_project_economics(rho_property="rho", prices=PRICES,
                                 rho_unit="кг/изд", currency_unit="₽",
                                 mass_unit="кг",
                                 rho_default_role=ROLE_OPTIMIZED)
        r1 = cst.runner_from_state(cst.runner_to_state(r0))

        assert r1.economics_enabled is True
        assert r1.rho_property == "rho"
        assert r1.component_prices == PRICES
        assert (r1.rho_unit, r1.currency_unit, r1.mass_unit) == \
            ("кг/изд", "₽", "кг")
        assert r1.rho_default_role == ROLE_OPTIMIZED

    def test_roundtrip_preserves_disabled_state(self):
        r0 = _measured(_runner())
        r0.set_project_economics(enabled=False)
        r1 = cst.runner_from_state(cst.runner_to_state(r0))
        assert r1.economics_enabled is False
        assert r1.rho_property == ""

    def test_legacy_state_without_keys_loads_disabled(self):
        """Старый сейв без ключей: экономика ВЫКЛЮЧЕНА, не «включена без ρ»."""
        r0 = _measured(_runner())
        r0.set_project_economics(rho_property="rho", prices=PRICES)
        state = cst.runner_to_state(r0)
        for k in ("economics_enabled", "rho_property", "component_prices",
                  "rho_unit", "currency_unit", "mass_unit",
                  "rho_default_role"):
            state["runner"].pop(k, None)

        r1 = cst.runner_from_state(state)
        assert r1.economics_enabled is False
        assert r1.rho_property == ""
        assert r1.component_prices == {}

    def test_roundtrip_of_enabled_but_unconfigured_project(self):
        """Проект сохранён ДО заполнения блока экономики: «вкл, но не настроена».

        Это валидное состояние нового проекта (дефолт конструктора), и load не
        имеет права ни упасть, ни выдать его за настроенную ногу.
        """
        r0 = _measured(_runner())
        assert r0.economics_enabled is True and r0.rho_property == ""
        r1 = cst.runner_from_state(cst.runner_to_state(r0))
        assert r1.economics_enabled is True
        assert r1.rho_property == ""
        assert r1.component_prices == {}
        assert r1.project_price_vector() == []

    def test_inherited_branch_leg_survives_roundtrip(self):
        """Наследованная веткой цена сериализуема (price_spec), значит доживает."""
        r0 = _measured(_runner())
        r0.set_project_economics(rho_property="rho", prices=PRICES)
        ctrl = CampaignController(r0)
        bid = ctrl.create_branch(
            "premium", {"strength": DesirabilitySpec("max", low=2.0, high=12.0)},
            budget=5,
            cost_spec=DesirabilitySpec("min", low=0.0, high=300.0,
                                       weight=0.5))["branch_id"]

        r1 = cst.runner_from_state(cst.runner_to_state(r0))
        cfg = r1._branch_cost[bid]
        assert cfg["rho_property"] == "rho"
        assert cfg["price_fn"].price_spec["prices"] == [95.0, 200.0, 23.0]


# ======================================================================
# 6. Единицы: одна масса у цены и у ρ ⇒ произведение даёт валюту за изделие
# ======================================================================
def test_item_price_is_composition_price_times_rho():
    """price_изд = price_состав[вал/масса] · ρ[масса/изд] = вал/изд (§3)."""
    from src.optimize.desirability import make_item_cost_fn, price_per_item

    r = _measured(_runner())
    r.set_project_economics(rho_property="rho", prices=PRICES, mass_unit="кг")
    price_fn = cst.linear_price_fn(r.project_price_vector())

    X = np.atleast_2d(r.X[0])
    rho_pred = (lambda Xq: r.surrogates["rho"].predict(Xq).mean)
    cost_fn = make_item_cost_fn(price_fn, rho_pred)

    pc = float(np.asarray(price_fn(X), float).ravel()[0])
    rho = float(np.asarray(rho_pred(X), float).ravel()[0])
    assert float(np.asarray(cost_fn(X), float).ravel()[0]) == \
        pytest.approx(float(price_per_item(pc, rho)))
    # знак §3: МЕНЬШЕ ρ ⇒ дешевле изделие (вспенивание), не наоборот
    assert float(price_per_item(pc, rho * 0.5)) < float(price_per_item(pc, rho))


# ======================================================================
# 7. UI-хелперы стартовой страницы (чистые, без Streamlit)
# ======================================================================
class TestSetupUiHelpers:

    def test_ensure_rho_appends_when_missing(self):
        from src.apps.campaign_ui import ensure_rho_in_responses
        assert ensure_rho_in_responses(["gloss", "strength"], "rho") == \
            ["gloss", "strength", "rho"]

    def test_ensure_rho_is_idempotent_and_keeps_order(self):
        """ρ уже в списке — порядок столбцов «(lab)» не сдвигается."""
        from src.apps.campaign_ui import ensure_rho_in_responses
        src = ["rho", "gloss"]
        assert ensure_rho_in_responses(src, "rho") == src

    def test_parse_prices_maps_by_position(self):
        from src.apps.campaign_ui import parse_component_prices
        assert parse_component_prices("95, 200, 23", MIX) == PRICES

    def test_parse_prices_refuses_wrong_count(self):
        from src.apps.campaign_ui import parse_component_prices
        with pytest.raises(ValueError, match="Нужно 3 цен"):
            parse_component_prices("95, 200", MIX)

    def test_role_label_roundtrip(self):
        from src.apps.campaign_ui import (rho_role_from_label, rho_role_to_label,
                                          RHO_ROLE_LABELS)
        for label in RHO_ROLE_LABELS:
            assert rho_role_to_label(rho_role_from_label(label)) == label
        assert rho_role_from_label(RHO_ROLE_LABELS[0]) == ROLE_PRICE_INPUT
        assert rho_role_from_label(RHO_ROLE_LABELS[1]) == ROLE_OPTIMIZED

    def test_formula_caption_states_single_mass_unit(self):
        """Подпись обязана называть единицы явно — иначе ρ введут в г/см³."""
        from src.apps.campaign_ui import item_cost_formula_caption
        txt = item_cost_formula_caption("₽", "кг", "кг/изд")
        assert "[₽/кг]" in txt and "[кг/изд]" in txt and "[₽/изд]" in txt

    def test_prefill_roundtrips_project_economics(self):
        """Загрузка проекта возвращает блок экономики в поля формы (иначе стёрся)."""
        from src.apps.campaign_ui import (setup_prefill_from_runner,
                                          RHO_ROLE_TARGET_LABEL)
        r = _runner()
        r.set_project_economics(rho_property="rho", prices=PRICES,
                                rho_unit="кг/изд", currency_unit="₽",
                                mass_unit="кг",
                                rho_default_role=ROLE_OPTIMIZED)
        pre = setup_prefill_from_runner(r)
        assert pre["setup_econ_on"] is True
        assert pre["setup_econ_rho"] == "rho"
        assert pre["setup_econ_prices"] == "95, 200, 23"
        assert pre["setup_econ_rho_unit"] == "кг/изд"
        assert pre["setup_econ_rho_role"] == RHO_ROLE_TARGET_LABEL

    def test_prefill_marks_disabled_economics(self):
        from src.apps.campaign_ui import setup_prefill_from_runner
        r = _runner()
        r.set_project_economics(enabled=False)
        assert setup_prefill_from_runner(r)["setup_econ_on"] is False


# ======================================================================
# 8. Сквозной прогон формы: блок экономики ЖИВЁТ на стартовой странице
# ======================================================================
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402
import os  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(_REPO, "src", "apps", "streamlit_app.py")


def _click(at, key):
    b = [w for w in at.button if w.key == key]
    assert b, f"кнопка {key} не найдена"
    b[0].click().run()


def test_setup_page_has_economics_widgets_by_default():
    """Ключевое требование: ρ видна на СТАРТОВОЙ странице сразу, без веток."""
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception
    keys = {w.key for w in at.checkbox} | {w.key for w in at.text_input} \
        | {w.key for w in at.radio}
    assert "setup_econ_on" in keys        # галка «учитывать себестоимость»
    assert "setup_econ_rho" in keys       # имя отклика плотности
    assert "setup_econ_prices" in keys    # цены сырья
    assert "setup_econ_rho_role" in keys  # роль ρ (вход цены / целевой параметр)
    # включена ПО УМОЛЧАНИЮ
    assert [w for w in at.checkbox if w.key == "setup_econ_on"][0].value is True


def test_setup_build_configures_project_economics():
    at = AppTest.from_file(APP, default_timeout=240).run()
    at.session_state["setup_econ_prices"] = "95, 200, 23"
    at.session_state["setup_econ_rho_unit"] = "кг/изд"
    at.run()
    _click(at, "setup_build")
    assert not at.exception

    r = at.session_state["campaign_ctrl"].runner
    assert r.economics_enabled is True
    assert r.rho_property == "rho"
    assert r.component_prices == PRICES
    assert r.rho_unit == "кг/изд"
    assert r.rho_default_role == ROLE_PRICE_INPUT


def test_setup_appends_rho_to_responses_when_missing():
    """ρ дописывается в отклики ⇒ появляется столбец «(lab)» в стартовом плане."""
    at = AppTest.from_file(APP, default_timeout=240).run()
    at.session_state["setup_resp"] = "strength, gloss"   # ρ пользователь не ввёл
    at.session_state["setup_econ_prices"] = "95, 200, 23"
    at.run()
    _click(at, "setup_build")
    assert not at.exception

    r = at.session_state["campaign_ctrl"].runner
    assert list(r.property_names) == ["strength", "gloss", "rho"]
    assert r.rho_property == "rho"


def test_setup_can_disable_economics_explicitly():
    at = AppTest.from_file(APP, default_timeout=240).run()
    at.session_state["setup_econ_on"] = False
    at.run()
    _click(at, "setup_build")
    assert not at.exception

    r = at.session_state["campaign_ctrl"].runner
    assert r.economics_enabled is False
    assert r.rho_property == ""
    # ρ не навязана: отклики остались как ввёл пользователь (дефолт формы)
    assert "rho" in r.property_names   # дефолтная строка откликов её содержит


def test_setup_warns_when_all_prices_are_zero():
    """Явные нули: экономика «включена», но себестоимость тождественно ноль.

    iter85: нули больше НЕ дефолт поля (дефолт — пусто, «цены пока неизвестны»),
    поэтому вводим их явно. Смысл проверки прежний: заявление «всё сырьё
    бесплатно» обязано быть озвучено, а не проглочено (A0.6).
    """
    at = AppTest.from_file(APP, default_timeout=240).run()
    at.session_state["setup_econ_prices"] = "0, 0, 0"
    at.run()
    assert not at.exception
    assert any("нулевые" in str(w.value) for w in at.warning)


def test_setup_refuses_incomplete_prices_without_building():
    """Неполные цены — ЯВНЫЙ отказ формы, проект не собирается (A0.6)."""
    at = AppTest.from_file(APP, default_timeout=240).run()
    at.session_state["setup_econ_prices"] = "95, 200"    # трёх компонентов, две цены
    at.run()
    _click(at, "setup_build")
    assert not at.exception                              # не падение, а сообщение
    assert "campaign_ctrl" not in at.session_state
    assert any("цен компонентов" in str(e.value) for e in at.error)
