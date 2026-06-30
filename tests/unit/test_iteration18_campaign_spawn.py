# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 18 — ШАГ 3: spawn ветки с наследованием ролей (§8, П-4).

  * Тр-8.1 — дефолт = наследование ролей родителя;
  * Тр-8.1а — review-сводка отдаётся при spawn И при preview (без создания);
  * Тр-8.1б — сводка различает «унаследовано как есть» / «изменено для ветки»;
  * Тр-8.1в / И-5 — новый объектив над ρ ⇒ ρ в ребёнке OPTIMIZED + канал ZEROED,
    помечено «changed_by_objective»;
  * Тр-5.4 / И-1 — родитель и пул не затронуты spawn-ом (история цела).
"""
import json
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
import pytest

from src.core.schema import ProjectSchema, VariableBlock, ModelSpec
from src.design.block_model import build_model_terms
from src.design.branches import ROLE_OPTIMIZED, ROLE_PRICE_INPUT, ROLE_REFERENCE
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


def _parent_cost(r):
    """Родитель: p0 OPTIMIZED, p1 (=ρ) PRICE_INPUT, p2 REFERENCE."""
    r.add_branch("parent", {"p0": DesirabilitySpec("max", low=-5, high=5)},
                 branch_id="P")
    r.set_branch_cost("P", _price_fn,
                      DesirabilitySpec("min", low=0.0, high=10.0),
                      rho_property="p1")


def _row(review, response):
    return next(x for x in review["responses"] if x["response"] == response)


# ======================================================================
# Тр-8.1 — наследование по умолчанию
# ======================================================================
def test_spawn_inherits_roles_by_default():
    r = _runner()
    _parent_cost(r)
    ctrl = CampaignController(r)
    res = ctrl.spawn_branch("P", "child", child_id="C")
    # ребёнок есть и роли совпали с родителем
    assert "C" in r.branches
    assert r.branch_roles("C") == r.branch_roles("P")
    # сводка: всё унаследовано как есть
    assert res["review"]["any_role_changed_by_objective"] is False
    assert all(row["change"] == "inherited"
               for row in res["review"]["responses"])
    # ρ-канал унаследован живым (PRICE_INPUT)
    assert res["price_channel_suppressed"] is False
    assert r.response_role("C", "p1") == ROLE_PRICE_INPUT


# ======================================================================
# Тр-8.1в / И-5 / П-4 — новый объектив перебивает роль ρ
# ======================================================================
def test_spawn_objective_overrides_rho_role_and_zeroes_channel():
    r = _runner()
    _parent_cost(r)
    ctrl = CampaignController(r)
    res = ctrl.spawn_branch(
        "P", "child_q", child_id="CQ",
        new_goals={"p1": DesirabilitySpec("min", low=-5, high=5)})
    # ρ (p1) в ребёнке стала целью ⇒ OPTIMIZED, денежный канал занулён (И-5)
    assert r.response_role("CQ", "p1") == ROLE_OPTIMIZED
    assert res["price_channel_suppressed"] is True
    row = _row(res["review"], "p1")
    assert row["role_parent"] == ROLE_PRICE_INPUT
    assert row["role_child"] == ROLE_OPTIMIZED
    assert row["role_changed"] is True
    assert row["change"] == "changed_by_objective"
    assert row["money_channel_child"] == "zeroed"
    assert res["review"]["any_role_changed_by_objective"] is True
    # нетронутые отклики помечены как унаследованные (Тр-8.1б)
    assert _row(res["review"], "p0")["change"] == "inherited"
    assert _row(res["review"], "p2")["change"] == "inherited"


# ======================================================================
# Тр-8.1а — preview не создаёт ветку
# ======================================================================
def test_preview_spawn_does_not_create_branch():
    r = _runner()
    _parent_cost(r)
    ctrl = CampaignController(r)
    n_branches = len(r.branches)
    review = ctrl.preview_spawn(
        "P", new_goals={"p1": DesirabilitySpec("min", low=-5, high=5)})
    assert len(r.branches) == n_branches      # ничего не создано (A0.6)
    assert _row(review, "p1")["change"] == "changed_by_objective"


# ======================================================================
# Тр-8.1б — «тронуто, но роль та же» отличается от «унаследовано как есть»
# ======================================================================
def test_spawn_summary_distinguishes_touched_same_role():
    r = _runner()
    _parent_cost(r)
    ctrl = CampaignController(r)
    # перезадаём p0 (уже OPTIMIZED) новой формой — роль та же, но отклик тронут
    res = ctrl.spawn_branch(
        "P", "child2", child_id="C2",
        new_goals={"p0": DesirabilitySpec("max", low=-3, high=7)})
    row0 = _row(res["review"], "p0")
    assert row0["overridden"] is True
    assert row0["role_changed"] is False
    assert row0["change"] == "overridden_same_role"
    assert _row(res["review"], "p1")["change"] == "inherited"


# ======================================================================
# Тр-5.4 / И-1 — родитель и общий пул не затронуты spawn-ом
# ======================================================================
def test_spawn_leaves_parent_and_history_untouched():
    r = _runner()
    _parent_cost(r)
    ctrl = CampaignController(r)
    parent_roles_before = r.branch_roles("P")
    n_hist = len(r.points)
    res = ctrl.spawn_branch(
        "P", "child3", child_id="C3",
        new_goals={"p1": DesirabilitySpec("min", low=-5, high=5)})
    assert r.branch_roles("P") == parent_roles_before   # родитель цел (Гр-3)
    assert r.price_channel_suppressed("P") is False
    assert len(r.points) == n_hist                       # история цела (И-1)
    json.dumps(res, ensure_ascii=False)                  # сериализуемо (UI/MCP)


def test_spawn_without_cost_inheritance_makes_rho_reference():
    r = _runner()
    _parent_cost(r)
    ctrl = CampaignController(r)
    # не наследуем ценовую ногу → у ребёнка ρ ни цель, ни цена ⇒ REFERENCE
    res = ctrl.spawn_branch("P", "child4", child_id="C4", inherit_cost=False)
    assert r.response_role("C4", "p1") == ROLE_REFERENCE
    assert _row(res["review"], "p1")["role_child"] == ROLE_REFERENCE
    assert _row(res["review"], "p1")["change"] == "changed_by_objective"
