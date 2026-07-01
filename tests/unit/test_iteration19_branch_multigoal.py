# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 19 / §16.3 — мультицелевая ветка с диапазонами.

Гейт-тест карты §16.6: ТЗ снимает ограничение «одна ветка — одно целевое
значение». В одной ветке сосуществуют несколько целей РАЗНЫХ видов
(``min``/``max``/``target``), диапазонов и весов; редактор целей
(``set_desirability`` / ``set_weights`` / ``delete_goal`` в
``CampaignController``) меняет d_best и рекомендацию x* предсказуемо, а удаление
ПОСЛЕДНЕЙ цели — отказ (ветке нужен объектив). Каждая правка обратима (undo, §7)
и НЕ трогает измеренную правду (И-1).
"""
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.core.schema import ProjectSchema, VariableBlock, ModelSpec
from src.design.block_model import build_model_terms
from src.design.branches import ROLE_OPTIMIZED, ROLE_REFERENCE
from src.optimize.desirability import DesirabilitySpec
from src.verification.mixture_process_truth import MultiMixtureProcessTruth
from src.apps.mixture_process_runner import MixtureProcessRunner
from src.apps.campaign import CampaignController

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _schema():
    mix = VariableBlock.mixture(["A", "B", "C"])
    proc = VariableBlock.process(["T", "P"], lower=[0.0, 0.0], upper=[1.0, 1.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    return ProjectSchema.mixture_process(mix, proc, model=model)


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


def _triple_goal():
    """Три цели РАЗНЫХ видов/диапазонов/весов над одной веткой."""
    return {
        "p0": DesirabilitySpec("max", low=-4.0, high=4.0, weight=1.0),
        "p1": DesirabilitySpec("min", low=-3.0, high=6.0, weight=2.0),
        "p2": DesirabilitySpec("target", low=-5.0, high=5.0, target=0.0,
                               weight=1.5),
    }


def _roles(ctrl, bid):
    return {x["response"]: x for x in ctrl.role_report(bid)["responses"]}


# ======================================================================
# Ветка несёт ≥3 целей разных видов/диапазонов/весов
# ======================================================================
def test_branch_carries_three_distinct_goals():
    r = _runner()
    r.add_branch("multi", _triple_goal(), branch_id="b1")
    ctrl = CampaignController(r)
    rep = ctrl.role_report("b1")
    by = {x["response"]: x for x in rep["responses"]}

    assert rep["by_role"][ROLE_OPTIMIZED] == ["p0", "p1", "p2"]
    # разные ВИДЫ desirability
    assert {by["p0"]["desirability_kind"], by["p1"]["desirability_kind"],
            by["p2"]["desirability_kind"]} == {"max", "min", "target"}
    # разные ВЕСА
    assert by["p0"]["weight"] == 1.0
    assert by["p1"]["weight"] == 2.0
    assert by["p2"]["weight"] == 1.5


# ======================================================================
# add цели (set_desirability над новым откликом) — d_best/x* пересчитаны
# ======================================================================
def test_add_goal_recomputes_dbest_and_xopt():
    r = _runner()
    # старт с ДВУХ целей; p2 добавим как третью
    r.add_branch("multi", {"p0": DesirabilitySpec("max", low=-4, high=4),
                           "p1": DesirabilitySpec("min", low=-3, high=6)},
                 branch_id="b1")
    ctrl = CampaignController(r)
    assert _roles(ctrl, "b1")["p2"]["role"] == ROLE_REFERENCE

    res = ctrl.set_desirability(
        "b1", "p2", DesirabilitySpec("target", low=-5, high=5, target=0.0))

    # p2 стал целью (OPTIMIZED); рекомендация x* и d_best пересчитаны
    assert _roles(ctrl, "b1")["p2"]["role"] == ROLE_OPTIMIZED
    assert res["x_opt_after"] is not None
    assert np.isfinite(res["d_best_after"])
    assert res["recommendation_shift"] is not None
    assert res["undo_available"] is True


# ======================================================================
# edit веса — предсказуемо двигает оценку (re-score), обратимо (undo)
# ======================================================================
def test_edit_weight_rescore_and_undo():
    r = _runner()
    r.add_branch("multi", _triple_goal(), branch_id="b1")
    ctrl = CampaignController(r)
    # свежая ветка не пере-оценена (d_best=0.0); форсируем re-score под ИСХОДНЫМ
    # объективом (no-op вес) — это эталон d_best под весами по умолчанию.
    d_ref = ctrl.set_weights("b1", {"p1": 2.0})["d_best_after"]

    res = ctrl.set_weights("b1", {"p1": 5.0})
    assert res["op"] == "set_weights"
    assert np.isfinite(res["d_best_after"])
    assert _roles(ctrl, "b1")["p1"]["weight"] == 5.0

    ctrl.undo()                                        # откат веса 5 → 2
    assert _roles(ctrl, "b1")["p1"]["weight"] == 2.0
    # d_best вернулся к оценке под исходным объективом (re-score обратим)
    assert float(r.branches["b1"].d_best) == pytest.approx(d_ref)



# ======================================================================
# delete цели — ok, роль → REFERENCE; удаление ПОСЛЕДНЕЙ — отказ
# ======================================================================
def test_delete_goals_down_to_last_is_refused():
    r = _runner()
    r.add_branch("multi", _triple_goal(), branch_id="b1")
    ctrl = CampaignController(r)

    ctrl.delete_goal("b1", "p2")                       # 3 → 2
    assert _roles(ctrl, "b1")["p2"]["role"] == ROLE_REFERENCE
    assert "p2" not in r.branches["b1"].goal

    ctrl.delete_goal("b1", "p1")                       # 2 → 1
    assert set(r.branches["b1"].goal) == {"p0"}

    with pytest.raises(ValueError):                    # последняя — нельзя
        ctrl.delete_goal("b1", "p0")
    assert set(r.branches["b1"].goal) == {"p0"}        # объектив сохранён


# ======================================================================
# И-1 — правки целей не трогают измеренную историю
# ======================================================================
def test_goal_edits_preserve_history():
    r = _runner()
    r.add_branch("multi", _triple_goal(), branch_id="b1")
    ctrl = CampaignController(r)
    n_hist = len(r.points)

    ctrl.set_weights("b1", {"p0": 3.0})
    ctrl.set_desirability("b1", "p1", DesirabilitySpec("max", low=-6, high=6))
    ctrl.delete_goal("b1", "p2")

    assert len(r.points) == n_hist                     # история цела (И-1)
