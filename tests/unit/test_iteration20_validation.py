# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 20 / §17.3 (Ш2) — валидация «не хватает данных» перед пересчётом.

Гейт-тест единой проверки готовности ветки (REBUILD_SPEC_17 §17.3). A0.6 / чистота
проводника: система НЕ считает молча на дырах — перед любым пересчётом (re-score /
M8-argmax / §4-стоп) :meth:`CampaignController.validate_ready` называет, ЧЕГО именно
не хватает, и возвращает ``{ok, missing, text}``. Проверяем, что КАЖДАЯ из четырёх
веток отказа §17.3 воспроизводима, что дыры НАКАПЛИВАЮТСЯ (а не «первая же»), и что
happy-path даёт ``ok=True``. Проверка read-only: база и undo-стек не трогаются.
"""
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.core.schema import ProjectSchema, VariableBlock, ModelSpec
from src.design.block_model import build_model_terms
from src.optimize.desirability import DesirabilitySpec
from src.verification.mixture_process_truth import MultiMixtureProcessTruth
from src.apps.mixture_process_runner import MixtureProcessRunner
from src.apps.campaign import (CampaignController, READINESS_EMPTY_OBJECTIVE,
                               READINESS_UNMEASURED_GOAL,
                               READINESS_PRICE_LEG_INCOMPLETE,
                               READINESS_MIGRATION_PENDING)

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _schema():
    mix = VariableBlock.mixture(["A", "B", "C"])
    proc = VariableBlock.process(["T", "P"], lower=[0.0, 0.0], upper=[1.0, 1.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    return ProjectSchema.mixture_process(mix, proc, model=model)


def _price_fn(Xc):
    return np.ones(np.atleast_2d(Xc).shape[0], float)


def _make_runner(n_seed=16):
    """Раннер с оракулом; ``n_seed=0`` ⇒ БЕЗ стартового дизайна (пустая база)."""
    schema = _schema()
    p = build_model_terms(schema).p
    rng = np.random.default_rng(0)
    oracle = MultiMixtureProcessTruth(
        schema, {"p0": rng.normal(size=p), "p1": rng.normal(size=p),
                 "p2": rng.normal(size=p)}, noise_sd=0.0)
    r = MixtureProcessRunner(schema, oracle, seed=1, n_restarts=2,
                             baseline=[1 / 3, 1 / 3, 1 / 3, 0.5, 0.5])
    if n_seed:
        r.seed_initial(n=n_seed, seed=1)
    return r


def _codes(res):
    return {m["code"] for m in res["missing"]}


# ======================================================================
# happy-path: цель измерена, ценовая нога полна ⇒ ok=True
# ======================================================================
def test_ready_ok_when_goal_and_price_measured():
    r = _make_runner()
    r.add_branch("work", {"p0": DesirabilitySpec("max", low=-5, high=5)},
                 branch_id="b1", budget=12)
    r.set_branch_cost("b1", _price_fn,
                      DesirabilitySpec("min", low=0.0, high=10.0),
                      rho_property="p1")
    ctrl = CampaignController(r)

    n_base = len(r.points)
    res = ctrl.validate_ready("b1")

    assert res["ok"] is True
    assert res["missing"] == []
    assert res["branch_id"] == "b1"
    assert len(r.points) == n_base            # read-only: база не тронута
    assert not ctrl.can_undo()                # undo-стек не тронут


# ======================================================================
# (1) пустой объектив ⇒ empty_objective
# ======================================================================
def test_refuses_empty_objective():
    r = _make_runner()
    r.add_branch("empty", {}, branch_id="e1", budget=5)
    ctrl = CampaignController(r)

    res = ctrl.validate_ready("e1")

    assert res["ok"] is False
    assert READINESS_EMPTY_OBJECTIVE in _codes(res)


# ======================================================================
# (2) свойства целей без единого измерения ⇒ unmeasured_goal_properties
# ======================================================================
def test_refuses_unmeasured_goal_properties():
    r = _make_runner(n_seed=0)                # база пуста — ничего не измерено
    r.add_branch("work", {"p0": DesirabilitySpec("max", low=-5, high=5)},
                 branch_id="b1", budget=12)
    ctrl = CampaignController(r)

    res = ctrl.validate_ready("b1")

    assert res["ok"] is False
    assert READINESS_UNMEASURED_GOAL in _codes(res)
    unmeasured = next(m for m in res["missing"]
                      if m["code"] == READINESS_UNMEASURED_GOAL)
    assert "p0" in unmeasured["responses"]


# ======================================================================
# (2b) ρ ценовой ноги тоже обязана быть измерена (входит в required)
# ======================================================================
def test_unmeasured_includes_price_rho():
    r = _make_runner(n_seed=0)
    r.add_branch("work", {"p0": DesirabilitySpec("max", low=-5, high=5)},
                 branch_id="b1", budget=12)
    r.set_branch_cost("b1", _price_fn,
                      DesirabilitySpec("min", low=0.0, high=10.0),
                      rho_property="p1")
    ctrl = CampaignController(r)

    res = ctrl.validate_ready("b1")
    unmeasured = next(m for m in res["missing"]
                      if m["code"] == READINESS_UNMEASURED_GOAL)
    assert {"p0", "p1"} <= set(unmeasured["responses"])


# ======================================================================
# (3) ценовая нога объявлена, но неполна ⇒ price_leg_incomplete
# ======================================================================
def test_refuses_incomplete_price_leg():
    r = _make_runner()
    r.add_branch("work", {"p0": DesirabilitySpec("max", low=-5, high=5)},
                 branch_id="b1", budget=12)
    # намеренно битая ценовая конфигурация (в обход set_branch_cost): ρ вне
    # свойств оракула, нет price_fn, нет cost_spec — валидатор обязан назвать это.
    r._branch_cost["b1"] = {"rho_property": "ghost", "price_fn": None,
                            "cost_spec": None, "cost_name": "price"}
    ctrl = CampaignController(r)

    res = ctrl.validate_ready("b1")

    assert res["ok"] is False
    assert READINESS_PRICE_LEG_INCOMPLETE in _codes(res)


# ======================================================================
# (4) несмигрированные точки ⇒ migration_pending (ядро сигналит RuntimeError)
# ======================================================================
def test_refuses_pending_migration(monkeypatch):
    r = _make_runner()
    r.add_branch("work", {"p0": DesirabilitySpec("max", low=-5, high=5)},
                 branch_id="b1", budget=12)
    ctrl = CampaignController(r)

    def _raise():
        raise RuntimeError("2 точек не мигрировали к схеме v2 — проверь политику.")

    # изолируем ветку валидации от труднопроизводимого реального сбоя миграции:
    # реальная эволюция без политики падает уже на augment (fit_surrogates), сюда
    # мягкий отказ приходит именно из сигнала ядра ``_migrated_points`` → RuntimeError.
    monkeypatch.setattr(r, "_migrated_points", _raise)

    res = ctrl.validate_ready("b1")

    assert res["ok"] is False
    assert READINESS_MIGRATION_PENDING in _codes(res)


# ======================================================================
# накопление: несколько дыр одновременно (не «первая же»)
# ======================================================================
def test_accumulates_multiple_missing():
    r = _make_runner(n_seed=0)                # пустая база ⇒ недомер
    r.add_branch("work", {"p0": DesirabilitySpec("max", low=-5, high=5)},
                 branch_id="b1", budget=12)
    r._branch_cost["b1"] = {"rho_property": "ghost", "price_fn": None,
                            "cost_spec": None, "cost_name": "price"}
    ctrl = CampaignController(r)

    res = ctrl.validate_ready("b1")
    codes = _codes(res)

    assert res["ok"] is False
    assert READINESS_UNMEASURED_GOAL in codes
    assert READINESS_PRICE_LEG_INCOMPLETE in codes
    # текст сводки перечисляет все дыры (для UI/ассистента)
    assert res["text"].count("•") == len(res["missing"])


# ======================================================================
# несуществующая ветка ⇒ KeyError (контекст ветки обязателен)
# ======================================================================
def test_unknown_branch_raises():
    r = _make_runner()
    ctrl = CampaignController(r)
    with pytest.raises(KeyError):
        ctrl.validate_ready("nope")
