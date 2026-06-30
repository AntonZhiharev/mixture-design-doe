# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 18 — ШАГ 2: CampaignController (мутации §4 + смена роли §5 + undo §7).

Проверяем обратимые мутации намерения ветки поверх MixtureProcessRunner:

  * §5 / И-5 / Тр-5.5 — смена роли ρ переключает денежный канал (ZEROED↔ALIVE);
    PRICE_INPUT→OPTIMIZED требует DesirabilitySpec;
  * Тр-5.4 / Гр-3 / П-5 — мутация атомарна в пределах ветки; ДРУГИЕ ветки целы;
  * §7 / П-10 — undo откатывает обратимую интерпретацию; прогон раунда обнуляет
    стек (дно = последний снятый раунд, Тр-7.2/7.3);
  * И-1 / П-11 — история не урезается ни одной мутацией;
  * §4 ярус-1 — веса / форма десирабилити / удаление цели (с защитой объектива).
"""
import json
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
import pytest

from src.core.schema import ProjectSchema, VariableBlock, ModelSpec
from src.design.block_model import build_model_terms
from src.design.branches import ROLE_OPTIMIZED, ROLE_PRICE_INPUT
from src.optimize.desirability import DesirabilitySpec
from src.verification.mixture_process_truth import MultiMixtureProcessTruth
from src.apps.mixture_process_runner import MixtureProcessRunner
from src.apps.campaign import CampaignController

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ----------------------------------------------------------------------
def _schema():
    mix = VariableBlock.mixture(["A", "B", "C"])
    proc = VariableBlock.process(["T", "P"], lower=[0.0, 0.0], upper=[1.0, 1.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    return ProjectSchema.mixture_process(mix, proc, model=model)


def _price_fn(Xc):
    return np.ones(np.atleast_2d(Xc).shape[0], float)


def _runner(n_seed=14):
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


def _cost_branch(r):
    """b2: ρ (p1) НЕ цель, питает цену ⇒ PRICE_INPUT."""
    r.add_branch("cost", {"p0": DesirabilitySpec("max", low=-5, high=5)},
                 branch_id="b2")
    r.set_branch_cost("b2", _price_fn,
                      DesirabilitySpec("min", low=0.0, high=10.0),
                      rho_property="p1")


def _dual_branch(r):
    """b3: ρ (p1) И цель, И питает цену ⇒ OPTIMIZED (канал занулён)."""
    r.add_branch("dual", {"p0": DesirabilitySpec("max", low=-5, high=5),
                          "p1": DesirabilitySpec("min", low=-5, high=5)},
                 branch_id="b3")
    r.set_branch_cost("b3", _price_fn,
                      DesirabilitySpec("min", low=0.0, high=10.0),
                      rho_property="p1")


# ======================================================================
# §5 / И-5 — смена роли переключает денежный канал ρ
# ======================================================================
def test_switch_price_input_to_optimized_zeroes_channel():
    r = _runner()
    _cost_branch(r)
    ctrl = CampaignController(r)
    res = ctrl.switch_role("b2", "p1", ROLE_OPTIMIZED,
                           spec=DesirabilitySpec("min", low=-5, high=5))
    assert res["role_before"] == ROLE_PRICE_INPUT
    assert res["role_after"] == ROLE_OPTIMIZED
    assert res["price_channel_suppressed"] is True   # канал занулён (Гр-1)
    assert res["undo_available"] is True
    # отражено и во view-model: money_channel p1 стал zeroed
    chan = {x["response"]: x["money_channel"]
            for x in ctrl.role_report("b2")["responses"]}
    assert chan["p1"] == "zeroed"


def test_switch_optimized_to_price_input_revives_channel():
    r = _runner()
    _dual_branch(r)
    ctrl = CampaignController(r)
    res = ctrl.switch_role("b3", "p1", ROLE_PRICE_INPUT)
    assert res["role_before"] == ROLE_OPTIMIZED
    assert res["role_after"] == ROLE_PRICE_INPUT
    assert res["price_channel_suppressed"] is False  # канал живой (ALIVE)
    chan = {x["response"]: x["money_channel"]
            for x in ctrl.role_report("b3")["responses"]}
    assert chan["p1"] == "alive"


def test_switch_to_optimized_requires_spec():
    r = _runner()
    _cost_branch(r)
    ctrl = CampaignController(r)
    with pytest.raises(ValueError):
        ctrl.switch_role("b2", "p1", ROLE_OPTIMIZED)   # нет spec


def test_switch_requires_price_leg():
    r = _runner()
    r.add_branch("plain", {"p0": DesirabilitySpec("max", low=-5, high=5)},
                 branch_id="b1")
    ctrl = CampaignController(r)
    with pytest.raises(ValueError):
        ctrl.switch_role("b1", "p1", ROLE_OPTIMIZED,
                         spec=DesirabilitySpec("min", low=-5, high=5))


# ======================================================================
# Тр-5.4 / Гр-3 / П-5 — другие ветки не затронуты
# ======================================================================
def test_switch_leaves_other_branches_untouched():
    r = _runner()
    _cost_branch(r)
    _dual_branch(r)
    ctrl = CampaignController(r)

    b3_goal_before = sorted(r.branches["b3"].goal.keys())
    b3_dbest_before = float(r.branches["b3"].d_best)
    ctrl.switch_role("b2", "p1", ROLE_OPTIMIZED,
                     spec=DesirabilitySpec("min", low=-5, high=5))
    # b3 (ρ=OPTIMIZED) не сдвинулся: ни объектив, ни оценка, ни роль
    assert sorted(r.branches["b3"].goal.keys()) == b3_goal_before
    assert float(r.branches["b3"].d_best) == b3_dbest_before
    assert r.response_role("b3", "p1") == ROLE_OPTIMIZED


# ======================================================================
# §7 / П-10 — undo обратимой мутации; раунд обнуляет стек
# ======================================================================
def test_undo_restores_role():
    r = _runner()
    _cost_branch(r)
    ctrl = CampaignController(r)
    ctrl.switch_role("b2", "p1", ROLE_OPTIMIZED,
                     spec=DesirabilitySpec("min", low=-5, high=5))
    assert r.response_role("b2", "p1") == ROLE_OPTIMIZED
    und = ctrl.undo()
    assert und["op"] == "undo" and und["undone"] == "switch_role"
    assert r.response_role("b2", "p1") == ROLE_PRICE_INPUT   # роль вернулась
    assert r.price_channel_suppressed("b2") is False
    assert ctrl.can_undo() is False


def test_round_seals_undo_floor():
    r = _runner()
    _cost_branch(r)
    ctrl = CampaignController(r)
    ctrl.set_weights("b2", {"p0": 2.0})
    assert ctrl.can_undo() is True
    # прогон раунда = новые измерения (И-1) ⇒ дно стека, undo обнулён (Тр-7.2/7.3)
    ctrl.run_round("b2", n_points=2, explore_frac=0.2, n_candidates=120)
    assert ctrl.can_undo() is False
    with pytest.raises(IndexError):
        ctrl.undo()


# ======================================================================
# §4 ярус-1 — веса / форма / удаление цели
# ======================================================================
def test_set_weights_and_undo():
    r = _runner()
    _cost_branch(r)
    ctrl = CampaignController(r)
    w0 = {x["response"]: x["weight"]
          for x in ctrl.role_report("b2")["responses"]}["p0"]
    ctrl.set_weights("b2", {"p0": w0 + 3.0})
    w1 = {x["response"]: x["weight"]
          for x in ctrl.role_report("b2")["responses"]}["p0"]
    assert w1 == w0 + 3.0
    ctrl.undo()
    w2 = {x["response"]: x["weight"]
          for x in ctrl.role_report("b2")["responses"]}["p0"]
    assert w2 == w0


def test_set_weights_rejects_non_goal():
    r = _runner()
    _cost_branch(r)
    ctrl = CampaignController(r)
    with pytest.raises(KeyError):
        ctrl.set_weights("b2", {"p2": 1.0})   # p2 — REFERENCE, не цель


def test_delete_goal_and_last_goal_guard():
    r = _runner()
    _dual_branch(r)
    ctrl = CampaignController(r)
    # удаляем одну из двух целей — ok (p1 становится PRICE_INPUT)
    ctrl.delete_goal("b3", "p1")
    assert "p1" not in r.branches["b3"].goal
    assert r.response_role("b3", "p1") == ROLE_PRICE_INPUT
    # удалить последнюю цель нельзя — объектив обязан существовать
    with pytest.raises(ValueError):
        ctrl.delete_goal("b3", "p0")


# ======================================================================
# И-1 / П-11 — история не урезается; результат JSON-сериализуем
# ======================================================================
def test_mutation_preserves_history_and_is_serializable():
    r = _runner()
    _dual_branch(r)
    ctrl = CampaignController(r)
    n_before = len(r.points)
    res = ctrl.switch_role("b3", "p1", ROLE_PRICE_INPUT)
    assert len(r.points) == n_before          # история цела (И-1)
    s = json.dumps(res, ensure_ascii=False)   # JSON-сериализуем (для UI/MCP)
    assert isinstance(s, str)
    assert res["op"] == "switch_role"
