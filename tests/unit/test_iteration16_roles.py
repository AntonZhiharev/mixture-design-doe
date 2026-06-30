# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 16 — §5/§12 шаг A: РОЛЬ отклика в ветке (ветка × отклик).

Роль — атрибут ПАРЫ (ветка × отклик), а НЕ глобальное свойство отклика, и
ВЫВОДИТСЯ из намерения ветки (goal + ценовая нога), без отдельного хранения.
XOR честности (§5) разрешается ПРИОРИТЕТОМ M2: OPTIMIZED > PRICE_INPUT >
REFERENCE — поэтому роль всегда однозначна. Ключевой случай Гр-1: ρ
одновременно ЦЕЛЬ и питает цену ⇒ роль = OPTIMIZED, а ценовой σ_ρ-канал помечен
к занулению (:meth:`price_channel_suppressed`) — крючок под атрибуцию шага B.
"""
import numpy as np
import pytest

from src.core.schema import ProjectSchema, VariableBlock, ModelSpec
from src.design.block_model import build_model_terms
from src.design.branches import (ROLE_OPTIMIZED, ROLE_PRICE_INPUT,
                                  ROLE_REFERENCE)
from src.optimize.desirability import DesirabilitySpec
from src.verification.mixture_process_truth import MultiMixtureProcessTruth
from src.apps.mixture_process_runner import MixtureProcessRunner


def _schema():
    mix = VariableBlock.mixture(["A", "B", "C"])
    proc = VariableBlock.process(["T", "P"], lower=[0.0, 0.0], upper=[1.0, 1.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    return ProjectSchema.mixture_process(mix, proc, model=model)


def _runner():
    schema = _schema()
    p = build_model_terms(schema).p
    rng = np.random.default_rng(0)
    # три полноценных свойства: p0 — цель качества, p1 — ρ (плотность), p2 — справка
    oracle = MultiMixtureProcessTruth(
        schema, {"p0": rng.normal(size=p), "p1": rng.normal(size=p),
                 "p2": rng.normal(size=p)}, noise_sd=0.0)
    return MixtureProcessRunner(schema, oracle, seed=1, n_restarts=3,
                                baseline=[1 / 3, 1 / 3, 1 / 3, 0.5, 0.5])


def _price_fn(Xc):
    return np.ones(np.atleast_2d(Xc).shape[0], float)   # ₽/кг состава = const


def test_roles_quality_only_branch():
    """Ветка без цены: цель ⇒ OPTIMIZED, остальные ⇒ REFERENCE; цены нет."""
    r = _runner()
    r.add_branch("opt", {"p0": DesirabilitySpec("max", low=-5, high=5)},
                 branch_id="b1")
    assert r.response_role("b1", "p0") == ROLE_OPTIMIZED
    assert r.response_role("b1", "p1") == ROLE_REFERENCE
    assert r.response_role("b1", "p2") == ROLE_REFERENCE
    assert r.branch_roles("b1") == {"p0": ROLE_OPTIMIZED, "p1": ROLE_REFERENCE,
                                    "p2": ROLE_REFERENCE}
    assert r.price_channel_suppressed("b1") is False


def test_rho_as_price_input_when_not_a_goal():
    """ρ (p1) не цель, но питает цену ⇒ PRICE_INPUT; канал НЕ занулён."""
    r = _runner()
    r.add_branch("cost", {"p0": DesirabilitySpec("max", low=-5, high=5)},
                 branch_id="b2")
    r.set_branch_cost("b2", _price_fn,
                      DesirabilitySpec("min", low=0.0, high=10.0),
                      rho_property="p1")
    assert r.response_role("b2", "p1") == ROLE_PRICE_INPUT
    assert r.response_role("b2", "p0") == ROLE_OPTIMIZED
    assert r.response_role("b2", "p2") == ROLE_REFERENCE
    assert r.price_channel_suppressed("b2") is False


def test_rho_goal_and_price_priority_optimized_suppresses_channel():
    """Гр-1: ρ (p1) одновременно ЦЕЛЬ и питает цену ⇒ роль OPTIMIZED (приоритет
    M2), а ценовой σ_ρ-канал помечен к занулению."""
    r = _runner()
    r.add_branch("dual", {"p0": DesirabilitySpec("max", low=-5, high=5),
                          "p1": DesirabilitySpec("min", low=-5, high=5)},
                 branch_id="b3")
    r.set_branch_cost("b3", _price_fn,
                      DesirabilitySpec("min", low=0.0, high=10.0),
                      rho_property="p1")
    assert r.response_role("b3", "p1") == ROLE_OPTIMIZED   # НЕ PRICE_INPUT
    assert r.price_channel_suppressed("b3") is True


def test_role_is_branch_local_attribute():
    """Один отклик (p1) носит РАЗНЫЕ роли в разных ветках (атрибут пары)."""
    r = _runner()
    r.add_branch("cost", {"p0": DesirabilitySpec("max", low=-5, high=5)},
                 branch_id="b2")
    r.set_branch_cost("b2", _price_fn,
                      DesirabilitySpec("min", low=0.0, high=10.0),
                      rho_property="p1")
    r.add_branch("dual", {"p0": DesirabilitySpec("max", low=-5, high=5),
                          "p1": DesirabilitySpec("min", low=-5, high=5)},
                 branch_id="b3")
    r.set_branch_cost("b3", _price_fn,
                      DesirabilitySpec("min", low=0.0, high=10.0),
                      rho_property="p1")
    assert r.response_role("b2", "p1") == ROLE_PRICE_INPUT
    assert r.response_role("b3", "p1") == ROLE_OPTIMIZED


def test_responses_by_role_inverse_index():
    """Обратный индекс {роль → [отклики]} покрывает ВСЕ свойства оракула."""
    r = _runner()
    r.add_branch("cost", {"p0": DesirabilitySpec("max", low=-5, high=5)},
                 branch_id="b2")
    r.set_branch_cost("b2", _price_fn,
                      DesirabilitySpec("min", low=0.0, high=10.0),
                      rho_property="p1")
    by_role = r.responses_by_role("b2")
    assert by_role[ROLE_OPTIMIZED] == ["p0"]
    assert by_role[ROLE_PRICE_INPUT] == ["p1"]
    assert by_role[ROLE_REFERENCE] == ["p2"]
    # объединение ролей = все свойства, без потерь/дублей
    flat = sum(by_role.values(), [])
    assert sorted(flat) == sorted(r.property_names)


def test_unknown_response_and_branch_raise():
    """Неизвестный отклик или ветка ⇒ KeyError (явная ошибка, не молча)."""
    r = _runner()
    r.add_branch("opt", {"p0": DesirabilitySpec("max", low=-5, high=5)},
                 branch_id="b1")
    with pytest.raises(KeyError):
        r.response_role("b1", "nope")
    with pytest.raises(KeyError):
        r.response_role("nobranch", "p0")
