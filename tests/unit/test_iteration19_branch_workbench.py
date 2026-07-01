# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 19 / §16.4 — рабочий стол ветки: внутриветочный цикл добора точек.

Гейт-тест карты §16.6: «зайдя в ветку», пользователь имеет полный цикл раунда
(предложить → измерить → долить → переобучить → x*/d_best → §4 стоп), поверх
``run_branch_round`` (демо-режим: синтетический оракул). Проверяем §16.4:

  * база выросла на N; у долитых точек origin-тег ``branch:{id}`` (И-1, общий пул);
  * d_best НЕ убыл между раундами (монотонность измеренного лучшего);
  * суррогаты переобучены на расширенной базе (одна модель на проект);
  * §4-стоп возвращает ЛЕГАЛЬНЫЙ ``stop_reason``;
  * A0.6 — без явной команды (раунда) НИЧЕГО не мерится; раунд ЗАПЕЧАТЫВАЕТ дно
    undo (измеренную правду откатить нельзя, Тр-7.2/7.3).
"""
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.core.schema import ProjectSchema, VariableBlock, ModelSpec
from src.design.block_model import build_model_terms
from src.optimize.desirability import DesirabilitySpec
from src.optimize.economic_stop import (STOP_CEIL, STOP_STAGNATION,
                                         STOP_NOT_ECONOMICAL)
from src.verification.mixture_process_truth import MultiMixtureProcessTruth
from src.apps.mixture_process_runner import MixtureProcessRunner
from src.apps.campaign import CampaignController

warnings.filterwarnings("ignore", category=ConvergenceWarning)

_LEGAL_REASONS = {None, STOP_CEIL, STOP_STAGNATION, STOP_NOT_ECONOMICAL}


def _schema():
    mix = VariableBlock.mixture(["A", "B", "C"])
    proc = VariableBlock.process(["T", "P"], lower=[0.0, 0.0], upper=[1.0, 1.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    return ProjectSchema.mixture_process(mix, proc, model=model)


def _price_fn(Xc):
    return np.ones(np.atleast_2d(Xc).shape[0], float)


def _runner(n_seed=16):
    schema = _schema()
    p = build_model_terms(schema).p
    rng = np.random.default_rng(0)
    oracle = MultiMixtureProcessTruth(
        schema, {"p0": rng.normal(size=p), "p1": rng.normal(size=p),
                 "p2": rng.normal(size=p)}, noise_sd=0.0)
    r = MixtureProcessRunner(schema, oracle, seed=1, n_restarts=2,
                             baseline=[1 / 3, 1 / 3, 1 / 3, 0.5, 0.5])
    r.seed_initial(n=n_seed, seed=1)
    return r


def _branch(r, bid="b1"):
    br = r.add_branch("work", {"p0": DesirabilitySpec("max", low=-5, high=5)},
                      branch_id=bid, budget=12)
    r.set_branch_cost(bid, _price_fn,
                      DesirabilitySpec("min", low=0.0, high=10.0),
                      rho_property="p1")
    br.volume, br.cost_exp, br.horizon = 1000.0, 0.01, 50.0
    return br


# ======================================================================
# Раунд добора: база выросла на N, origin-тег ветки, суррогаты переобучены
# ======================================================================
def test_round_grows_base_with_branch_origin():
    r = _runner()
    _branch(r, "b1")
    ctrl = CampaignController(r)
    n_hist0 = len(r.points)
    n_base0 = 0 if r.X is None else len(r.X)

    out = ctrl.run_round("b1", n_points=3, explore_frac=0.3, n_candidates=200)

    assert out["added"] == 3
    assert len(r.points) == n_hist0 + 3                # общая база выросла на N
    assert (0 if r.X is None else len(r.X)) == n_base0 + 3
    # у долитых точек origin-тег ветки (И-1, общий пул с origin-тегами)
    assert r.points[-1].origin_tag["origin"] == "branch:b1"
    assert r.origin_counts().get("branch:b1", 0) == 3
    # суррогаты переобучены на расширенной базе (одна модель на проект)
    assert set(r.surrogates) == {"p0", "p1", "p2"}


# ======================================================================
# d_best НЕ убывает между раундами (монотонность измеренного лучшего)
# ======================================================================
def test_dbest_monotone_across_rounds():
    r = _runner()
    _branch(r, "b1")
    ctrl = CampaignController(r)

    o1 = ctrl.run_round("b1", n_points=3, n_candidates=200)
    d1 = float(o1["d_best"])
    o2 = ctrl.run_round("b1", n_points=3, n_candidates=200)
    d2 = float(o2["d_best"])

    assert d1 >= 0.0
    assert d2 >= d1 - 1e-12                             # монотонно не убывает


# ======================================================================
# §4-стоп возвращает ЛЕГАЛЬНЫЙ stop_reason
# ======================================================================
def test_stop_decision_returns_legal_reason():
    r = _runner()
    br = _branch(r, "b1")
    ctrl = CampaignController(r)
    ctrl.run_round("b1", n_points=3, n_candidates=200)

    dec = r.branch_stop_decision("b1", delta_d=0.05, ceil=0.9, n_round=1)
    assert dec.reason in _LEGAL_REASONS
    assert isinstance(dec.econ_red_flag, bool)


# ======================================================================
# A0.6 — без явной команды ничего не мерится; раунд запечатывает дно undo
# ======================================================================
def test_nothing_measured_without_round_and_round_seals_undo():
    r = _runner()
    _branch(r, "b1")
    ctrl = CampaignController(r)
    n_hist0 = len(r.points)

    # обратимая настройка (вес) — база НЕ трогается (A0.6: ничего не мерится молча)
    ctrl.set_weights("b1", {"p0": 2.0})
    assert len(r.points) == n_hist0
    assert ctrl.can_undo() is True

    # раунд = новые измерения ⇒ дно undo запечатано (Тр-7.2/7.3)
    ctrl.run_round("b1", n_points=2, n_candidates=200)
    assert len(r.points) == n_hist0 + 2
    assert ctrl.can_undo() is False
    with pytest.raises(IndexError):
        ctrl.undo()
