# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 20 / C2 (§17.6.1) — персистентность кампании (save/load/delete).

Канон «логика+тест, потом UI»: здесь ЛОГИКА персистентности
(:mod:`src.apps.campaign_state`) поверх ``MixtureProcessRunner`` — до UI-кнопок C4.
Проверяем round-trip живой кампании (сетап §17.4 + измеренный seed + ветка с
ценовой ногой + добор):

  * общая база точек (И-1), origin-теги и numpy-кэши X/Y воспроизводятся;
  * ветки целиком (цели/бюджет/статус/x*/d_best) и ценовая нога (ρ/cost_spec/
    сериализуемый price_spec) переживают save/load;
  * суррогаты НЕ сериализуются, а детерминированно ПЕРЕОБУЧАЮТСЯ из точек
    (одна модель физики §5/§12) — предсказания совпадают;
  * A0.6: price_fn без сериализуемого дескриптора — явный отказ, а не тихая
    потеря цены; save/load/delete защищены от traversal по имени.
"""
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.apps import campaign_state as cst
from src.apps.campaign import CampaignController
from src.apps.campaign_ui import build_setup_runner
from src.optimize.desirability import DesirabilitySpec

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ======================================================================
# Фикстура: живая кампания (сетап → измеренный seed → ветка+цена → добор)
# ======================================================================
def _live_campaign() -> CampaignController:
    runner = build_setup_runner(
        mixture_names=["A", "B", "C"], process_names=["T", "P"],
        process_lower=[0.0, 0.0], process_upper=[1.0, 1.0],
        response_names=["strength", "gloss", "rho"], seed=1)
    ctrl = CampaignController(runner)

    # стартовый дизайн: предложить (read-only) → измерить демо-оракулом → commit
    Xseed = np.asarray(ctrl.propose_seed(14, seed=1), float)
    Yseed = np.vstack([runner._measure(np.asarray(x, float)) for x in Xseed])
    ctrl.commit_seed(Xseed, Yseed)

    # ветка с мультицелью + ценовой ногой ρ (линейная цена состава — сериализуема)
    ctrl.create_branch(
        "premium",
        {"strength": DesirabilitySpec("max", low=2.0, high=12.0),
         "gloss": DesirabilitySpec("max", low=1.0, high=13.0)},
        branch_id="premium", budget=20, satisfy_at=1.1,
        price_fn=cst.linear_price_fn([95.0, 200.0, 23.0]),
        cost_spec=DesirabilitySpec("min", low=0.0, high=300.0, weight=0.5),
        rho_property="rho", volume=1.0e4, cost_exp=1.0e-3, horizon=100.0)

    # добор ветки: предложить → измерить → долить (растит базу, двигает d_best)
    Xc = np.asarray(ctrl.propose_points("premium", n_points=2, seed=3), float)
    Yc = np.vstack([runner._measure(np.asarray(x, float)) for x in Xc])
    ctrl.commit_measured("premium", Xc, Yc)
    return ctrl


# ======================================================================
# Round-trip состояния (in-memory) — база / ветки / цена / суррогаты
# ======================================================================
def test_state_roundtrip_preserves_base_and_arrays():
    r0 = _live_campaign().runner
    state = cst.runner_to_state(r0)
    r1 = cst.runner_from_state(state)

    assert list(r1.property_names) == list(r0.property_names)
    assert len(r1.points) == len(r0.points)
    assert np.allclose(r1.X, r0.X)
    assert np.allclose(r1.Y, r0.Y)
    assert r1.origin_counts() == r0.origin_counts()          # origin-теги (И-1)
    assert np.allclose(r1.baseline, r0.baseline)
    assert r1.current_schema_version == r0.current_schema_version


def test_state_roundtrip_preserves_branch_and_price_leg():
    r0 = _live_campaign().runner
    r1 = cst.runner_from_state(cst.runner_to_state(r0))

    assert set(r1.branches) == set(r0.branches)
    b0, b1 = r0.branches["premium"], r1.branches["premium"]
    assert b1.goal == b0.goal                                # мультицель целиком
    assert (b1.budget, b1.spent, b1.status) == (b0.budget, b0.spent, b0.status)
    assert b1.d_best == pytest.approx(b0.d_best)
    assert np.allclose(np.asarray(b1.x_best, float),
                       np.asarray(b0.x_best, float))
    assert (b1.volume, b1.cost_exp, b1.horizon) == \
           (b0.volume, b0.cost_exp, b0.horizon)              # экономика ветки

    # ценовая нога: ρ/имя/десирабилити + сериализуемый дескриптор price_fn
    c0, c1 = r0._branch_cost["premium"], r1._branch_cost["premium"]
    assert c1["rho_property"] == c0["rho_property"] == "rho"
    assert c1["cost_name"] == c0["cost_name"]
    assert c1["cost_spec"] == c0["cost_spec"]
    assert c1["price_fn"].price_spec == c0["price_fn"].price_spec
    probe = r0.X[:3]
    assert np.allclose(c1["price_fn"](probe), c0["price_fn"](probe))


def test_surrogates_refit_from_points_match():
    r0 = _live_campaign().runner
    r1 = cst.runner_from_state(cst.runner_to_state(r0))
    assert set(r1.surrogates) == set(r0.surrogates)          # одна модель физики
    for name in r0.property_names:
        m0 = np.asarray(r0.surrogates[name].predict(r0.X).mean, float)
        m1 = np.asarray(r1.surrogates[name].predict(r0.X).mean, float)
        assert np.allclose(m0, m1, atol=1e-6)                # переобучены 1-в-1


def test_overview_roundtrip_equal():
    r0 = _live_campaign().runner
    r1 = cst.runner_from_state(cst.runner_to_state(r0))
    o0, o1 = CampaignController(r0).overview(), CampaignController(r1).overview()
    assert o0["property_names"] == o1["property_names"]
    assert o0["n_points"] == o1["n_points"]
    assert o0["origin_counts"] == o1["origin_counts"]
    assert [b["id"] for b in o0["branches"]] == [b["id"] for b in o1["branches"]]
    assert o0["branches"][0]["d_best"] == pytest.approx(o1["branches"][0]["d_best"])


# ======================================================================
# Файловая персистентность: save / load / list / delete
# ======================================================================
def test_save_load_list_delete_cycle(tmp_path):
    r0 = _live_campaign().runner
    root = str(tmp_path)

    path = cst.save_campaign(r0, root, "camp1")
    assert path.endswith("campaign.json")
    assert cst.list_campaigns(root) == ["camp1"]

    r1 = cst.load_campaign(root, "camp1")
    assert len(r1.points) == len(r0.points)
    assert set(r1.branches) == set(r0.branches)
    assert np.allclose(r1.X, r0.X)

    # повторное сохранение под тем же именем — идемпотентно (перезапись)
    cst.save_campaign(r1, root, "camp1")
    assert cst.list_campaigns(root) == ["camp1"]

    assert cst.delete_campaign(root, "camp1") is True
    assert cst.list_campaigns(root) == []
    assert cst.delete_campaign(root, "camp1") is False       # уже нет


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        cst.load_campaign(str(tmp_path), "nope")


def test_name_and_traversal_guards(tmp_path):
    r0 = _live_campaign().runner
    for bad in ("", "..", "a/b", "a\\b"):
        with pytest.raises(ValueError):
            cst.save_campaign(r0, str(tmp_path), bad)
        with pytest.raises(ValueError):
            cst.delete_campaign(str(tmp_path), bad)


# ======================================================================
# A0.6: несериализуемая ценовая функция — честный отказ, не тихая потеря
# ======================================================================
def test_unserialisable_price_fn_refused():
    r = build_setup_runner(
        mixture_names=["A", "B", "C"], process_names=["T", "P"],
        process_lower=[0.0, 0.0], process_upper=[1.0, 1.0],
        response_names=["strength", "gloss", "rho"], seed=1)
    ctrl = CampaignController(r)
    Xseed = np.asarray(ctrl.propose_seed(12, seed=1), float)
    Yseed = np.vstack([r._measure(np.asarray(x, float)) for x in Xseed])
    ctrl.commit_seed(Xseed, Yseed)
    r.add_branch("b", {"strength": DesirabilitySpec("max", low=2.0, high=12.0)},
                 branch_id="b")
    # голое замыкание без дескриптора price_spec — сериализовать нельзя
    r.set_branch_cost("b", lambda Xc: np.zeros(len(np.atleast_2d(Xc))),
                      DesirabilitySpec("min", low=0.0, high=300.0),
                      rho_property="rho")
    with pytest.raises(ValueError):
        cst.runner_to_state(r)


def test_price_fn_spec_roundtrip():
    fn = cst.linear_price_fn([1.0, 2.0, 3.0])
    spec = cst.price_fn_to_spec(fn)
    assert spec == {"kind": "linear", "prices": [1.0, 2.0, 3.0]}
    fn2 = cst.price_fn_from_spec(spec)
    X = np.array([[0.2, 0.3, 0.5, 0.4, 0.6]])
    assert np.allclose(fn2(X), fn(X))
    with pytest.raises(ValueError):
        cst.price_fn_to_spec(lambda Xc: Xc)                  # нет price_spec
    with pytest.raises(ValueError):
        cst.price_fn_from_spec({"kind": "weird"})            # неизвестный вид
