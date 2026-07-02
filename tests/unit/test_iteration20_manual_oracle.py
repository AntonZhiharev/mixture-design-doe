# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 20 / §17.2 (Ш1) — ручной оракул: предложить → зафиксировать Y.

Гейт-тест расщеплённого измерения (REBUILD_SPEC_17 §17.2). Ручной цикл ветки —
это ДВЕ явные половины поверх ОДНОГО пула (канон §5/§12):

  * :meth:`CampaignController.propose_points` — предложить кандидатов БЕЗ
    измерения (A0.6): общая база и undo-стек НЕ трогаются (read-only);
  * :meth:`CampaignController.commit_measured` — дописать ВНЕСЁННЫЕ Y в общую
    базу с origin-тегом ``branch:{id}`` (И-1) и ЗАПЕЧАТАТЬ дно undo (измеренную
    правду откатить нельзя, Тр-7.2/7.3, как штатный раунд).

Синтетический оракул здесь играет роль «пользователя, внёсшего Y»: propose даёт
X, оракул меряет их (``runner._measure``), commit фиксирует. Проверяем контракт
форм X/Y (валидация полноты данных — Ш2) и инварианты честности.
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


def _branch(r, bid="b1", budget=12):
    r.add_branch("work", {"p0": DesirabilitySpec("max", low=-5, high=5)},
                 branch_id=bid, budget=budget)
    r.set_branch_cost(bid, _price_fn,
                      DesirabilitySpec("min", low=0.0, high=10.0),
                      rho_property="p1")
    return bid


def _measure_all(r, X):
    """Роль пользователя: измерить предложенные X оракулом → Y (n×P)."""
    return np.vstack([r._measure(np.asarray(x, float)) for x in X])


# ======================================================================
# propose_points — read-only: база и undo-стек НЕ трогаются (A0.6)
# ======================================================================
def test_propose_is_readonly_and_preserves_undo():
    r = _runner(); bid = _branch(r)
    ctrl = CampaignController(r)
    # мутация намерения → undo доступен
    ctrl.set_weights(bid, {"p0": 2.0})
    assert ctrl.can_undo()

    n_base_before = len(r.points)
    X = ctrl.propose_points(bid, n_points=3)

    assert X.shape == (3, r.dim)              # координаты текущей схемы
    assert len(r.points) == n_base_before      # база НЕ выросла (read-only)
    assert ctrl.can_undo()                     # предложение не сбрасывает undo


# ======================================================================
# commit_measured — долив в ОБЩУЮ базу, origin-тег, запечатывание undo
# ======================================================================
def test_commit_appends_to_pool_with_branch_origin_and_seals_undo():
    r = _runner(); bid = _branch(r)
    ctrl = CampaignController(r)
    ctrl.set_weights(bid, {"p0": 2.0})
    assert ctrl.can_undo()

    n_base_before = len(r.points)
    d_before = r.branches[bid].d_best
    X = ctrl.propose_points(bid, n_points=3)
    Y = _measure_all(r, X)

    out = ctrl.commit_measured(bid, X, Y)

    assert out["added"] == 3
    assert len(r.points) == n_base_before + 3        # И-1: общий пул вырос на N
    # origin-тег ветки у долитых точек
    tags = [p.origin_tag.get("origin") for p in r.points[-3:]]
    assert tags == [f"branch:{bid}"] * 3
    # измеренный d_best не убыл (монотонность лучшего, §3)
    assert r.branches[bid].d_best >= d_before
    # Тр-7.2/7.3: раунд запечатал дно undo (измеренную правду не откатить)
    assert not ctrl.can_undo()
    with pytest.raises(IndexError):
        ctrl.undo()


# ======================================================================
# Валидация форм X/Y (фундамент Ш2 «не хватает данных»)
# ======================================================================
def test_commit_rejects_wrong_property_count():
    r = _runner(); bid = _branch(r)
    ctrl = CampaignController(r)
    X = ctrl.propose_points(bid, n_points=2)
    Y_bad = np.zeros((2, len(r.property_names) - 1))   # недостаёт свойства
    with pytest.raises(ValueError):
        ctrl.commit_measured(bid, X, Y_bad)


def test_commit_rejects_row_mismatch_and_bad_dim():
    r = _runner(); bid = _branch(r)
    ctrl = CampaignController(r)
    X = ctrl.propose_points(bid, n_points=2)
    Y = _measure_all(r, X)
    # строк Y меньше числа точек
    with pytest.raises(ValueError):
        ctrl.commit_measured(bid, X, Y[:1])
    # неверная размерность X (координат на точку)
    with pytest.raises(ValueError):
        ctrl.commit_measured(bid, X[:, :r.dim - 1], Y)


# ======================================================================
# Бюджет: propose капается остатком, commit пустого — no-op
# ======================================================================
def test_propose_caps_at_budget_and_empty_commit_is_noop():
    r = _runner(); bid = _branch(r, budget=2)
    ctrl = CampaignController(r)
    X = ctrl.propose_points(bid, n_points=5)
    assert X.shape[0] == 2                    # ограничено остатком бюджета

    n_before = len(r.points)
    out = ctrl.commit_measured(bid, np.empty((0, r.dim)),
                               np.empty((0, len(r.property_names))))
    assert out["added"] == 0
    assert len(r.points) == n_before
