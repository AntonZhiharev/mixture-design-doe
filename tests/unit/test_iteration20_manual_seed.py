# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 20 / §17.4 (Ш3) — ручной СТАРТОВЫЙ оракул: предложить seed → Y.

Гейт-тест расщеплённого стартового дизайна (REBUILD_SPEC_17 §17.4). Как branch-цикл
§17.2, но для seed (ветки ещё нет): :meth:`propose_seed` предлагает дизайн БЕЗ
измерения (read-only), :meth:`commit_seed` дописывает ВНЕСЁННЫЕ Y в общую базу
(origin "seed", И-1) и обучает суррогаты. Синт.оракул играет «пользователя,
внёсшего Y». Проверяем read-only propose, рост базы/фит по commit, валидацию форм,
детерминизм по seed и то, что после ручного seed штатно поднимается branch-цикл.
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


def _fresh_runner():
    """Раннер БЕЗ стартового дизайна (пустая база) — как реальный сетап §17.4."""
    schema = _schema()
    p = build_model_terms(schema).p
    rng = np.random.default_rng(0)
    oracle = MultiMixtureProcessTruth(
        schema, {"p0": rng.normal(size=p), "p1": rng.normal(size=p),
                 "p2": rng.normal(size=p)}, noise_sd=0.0)
    return MixtureProcessRunner(schema, oracle, seed=1, n_restarts=2,
                                baseline=[1 / 3, 1 / 3, 1 / 3, 0.5, 0.5])


def _measure_all(r, X):
    """Роль пользователя: измерить предложенные X оракулом → Y (n×P)."""
    return np.vstack([r._measure(np.asarray(x, float)) for x in X])


# ======================================================================
# propose_seed — read-only: база пуста, суррогатов нет (A0.6)
# ======================================================================
def test_propose_seed_is_readonly():
    r = _fresh_runner()
    X = r.propose_seed(10, seed=1)
    assert X.shape == (10, r.dim)              # координаты текущей схемы (q+d)
    assert len(r.points) == 0                  # база НЕ тронута (read-only)
    assert not r.surrogates                    # суррогаты не обучены


def test_propose_seed_is_deterministic():
    r = _fresh_runner()
    X1 = r.propose_seed(8, seed=7)
    X2 = r.propose_seed(8, seed=7)
    assert np.allclose(X1, X2)                 # тот же seed ⇒ тот же дизайн


# ======================================================================
# commit_seed — рост общей базы, origin "seed", фит суррогатов
# ======================================================================
def test_commit_seed_fills_base_and_fits():
    r = _fresh_runner()
    X = r.propose_seed(12, seed=1)
    Y = _measure_all(r, X)

    out = r.commit_seed(X, Y)

    assert out["added"] == 12
    assert out["n_base"] == 12
    assert len(r.points) == 12
    # все стартовые точки помечены origin "seed" (И-1, общий пул)
    assert {p.origin_tag.get("origin") for p in r.points} == {"seed"}
    # суррогаты обучены и видят n точек по всем P свойствам
    assert set(r.surrogates) == set(r.property_names)
    assert r.X.shape[0] == 12 and r.Y.shape == (12, len(r.property_names))


# ======================================================================
# Валидация форм X/Y (та же чистота, что commit_measured §17.2)
# ======================================================================
def test_commit_seed_rejects_bad_shapes():
    r = _fresh_runner()
    X = r.propose_seed(4, seed=1)
    Y = _measure_all(r, X)
    with pytest.raises(ValueError):                       # свойств меньше P
        r.commit_seed(X, Y[:, :-1])
    with pytest.raises(ValueError):                       # строк Y ≠ числу точек
        r.commit_seed(X, Y[:2])
    with pytest.raises(ValueError):                       # координат на точку < dim
        r.commit_seed(X[:, :r.dim - 1], Y)


def test_commit_seed_empty_is_noop():
    r = _fresh_runner()
    out = r.commit_seed(np.empty((0, r.dim)),
                        np.empty((0, len(r.property_names))))
    assert out["added"] == 0
    assert len(r.points) == 0
    assert not r.surrogates


# ======================================================================
# Интеграция: после ручного seed штатно поднимается branch-цикл §17.2
# ======================================================================
def test_manual_seed_unblocks_branch_cycle():
    r = _fresh_runner()
    X = r.propose_seed(14, seed=1)
    r.commit_seed(X, _measure_all(r, X))
    n_seed = len(r.points)

    ctrl = CampaignController(r)
    r.add_branch("work", {"p0": DesirabilitySpec("max", low=-5, high=5)},
                 branch_id="b1", budget=8)
    # валидация Ш2 не должна ругаться — seed измерен по всем P
    assert ctrl.validate_ready("b1")["ok"] is True

    Xc = ctrl.propose_points("b1", n_points=3)
    Yc = _measure_all(r, Xc)
    out = ctrl.commit_measured("b1", Xc, Yc)

    assert out["added"] == 3
    assert len(r.points) == n_seed + 3
    tags = [p.origin_tag.get("origin") for p in r.points[-3:]]
    assert tags == ["branch:b1"] * 3


# ======================================================================
# Контроллер-passthrough propose_seed/commit_seed (для UI-объекта Ш3b)
# ======================================================================
def test_controller_seed_passthrough_and_seals_undo():
    r = _fresh_runner()
    ctrl = CampaignController(r)
    X = ctrl.propose_seed(10, seed=1)
    assert X.shape == (10, r.dim)
    assert len(r.points) == 0                  # propose — read-only

    out = ctrl.commit_seed(X, _measure_all(r, X))
    assert out["added"] == 10 and len(r.points) == 10
    assert ctrl.can_undo() is False            # стартовая правда запечатала undo
