# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 17 / §15.6 ШАГ 4 — двойной стоп-критерий + stop_reason.

Покрывает §4:
  * Branch.volume/cost_exp/horizon (§2) + resolve_horizon (override на раунд, §2.1);
  * expected_price_improvement / economic_value (денежный путь §6, через ρ);
  * decide_stop — чистая логика §4 (AND), stop_reason ∈
    {ceil_reached, stagnation, not_economical} с приоритетом причин.
"""
import numpy as np

from src.design.branches import Branch
from src.optimize.desirability import DesirabilitySpec
from src.optimize.economic_stop import (
    STOP_CEIL, STOP_STAGNATION, STOP_NOT_ECONOMICAL,
    expected_price_improvement, economic_value, decide_stop,
)


def _branch(**kw):
    goal = {"strength": DesirabilitySpec("max", low=0.0, high=10.0)}
    base = dict(id="b1", name="b1", goal=goal, budget=10)
    base.update(kw)
    return Branch(**base)


# ----------------------------------------------------------------------
# §2 / §2.1 — экономические атрибуты ветки и override горизонта.
# ----------------------------------------------------------------------
def test_branch_economic_attrs_defaults_neutral():
    b = _branch()
    # дефолты нейтральны → экономика не задана (обратная совместимость)
    assert b.volume == 0.0 and b.cost_exp == 0.0 and b.horizon == 0.0


def test_resolve_horizon_default_and_round_override():
    b = _branch(horizon=12.0)
    assert b.resolve_horizon() == 12.0            # default ветки
    assert b.resolve_horizon(override=3.0) == 3.0  # разовый override на раунд
    assert b.horizon == 12.0                       # override НЕ меняет default


def test_branch_economics_roundtrip_serialisation():
    b = _branch(volume=1000.0, cost_exp=250.0, horizon=8.0,
                d_best=0.4, spent=2)
    b2 = Branch.from_state(b.to_state())
    assert b2.volume == 1000.0 and b2.cost_exp == 250.0 and b2.horizon == 8.0
    # старое состояние без экономики читается с нейтральными дефолтами
    legacy = {"id": "x", "name": "x", "goal": {}, "budget": 5}
    b3 = Branch.from_state(legacy)
    assert b3.volume == 0.0 and b3.cost_exp == 0.0 and b3.horizon == 0.0


# ----------------------------------------------------------------------
# §6 — денежная оценка: EI на удешевление изделия через σ_ρ.
# ----------------------------------------------------------------------
def test_expected_price_improvement_zero_when_no_room_and_grows_with_sigma():
    pc = 2.0
    # price_best ниже типичной цены ⇒ почти нет улучшения (EI≈0)
    ei0 = expected_price_improvement(pc, rho_mean=1.0, rho_std=0.01,
                                     price_best=0.5, n_mc=4000, seed=1)
    assert ei0[0] < 0.05
    # та же μ, но больше σ_ρ ⇒ хвост вниз создаёт реальное ожидаемое удешевление
    ei_lowsig = expected_price_improvement(pc, rho_mean=1.0, rho_std=0.1,
                                           price_best=2.0, n_mc=8000, seed=2)
    ei_hisig = expected_price_improvement(pc, rho_mean=1.0, rho_std=0.5,
                                          price_best=2.0, n_mc=8000, seed=2)
    assert ei_hisig[0] > ei_lowsig[0] > 0.0, (
        "EI на удешевление должен расти с σ_ρ (разведка по цене, §5/§6).")


def test_economic_value_scales_with_volume_and_horizon():
    ei = np.array([0.0, 0.2, 0.1])              # ₽/изделие по кандидатам
    # берётся МАКСИМУМ EI, умножается на V и H
    val = economic_value(ei, volume=1000.0, horizon=5.0)
    assert val == 0.2 * 1000.0 * 5.0
    assert economic_value(ei, volume=0.0, horizon=5.0) == 0.0  # нет объёма — нет выгоды



# ----------------------------------------------------------------------
# §4 — чистая логика двойного стопа: AND-семантика + приоритет причин.
# ----------------------------------------------------------------------
def test_decide_stop_continue_when_all_conditions_hold():
    # есть прогресс, потолок не достигнут, экономически выгодно → продолжаем
    r = decide_stop(delta_d=0.02, d_best=0.6, ceil=0.9,
                    economic_value=500.0, cost_exp=100.0, eps=5e-3)
    assert r is None


def test_decide_stop_ceil_reached_has_top_priority():
    # d_best >= ceil ⇒ ceil_reached, даже если и прогресс встал, и невыгодно
    r = decide_stop(delta_d=0.0, d_best=0.95, ceil=0.9,
                    economic_value=0.0, cost_exp=100.0)
    assert r == STOP_CEIL


def test_decide_stop_not_economical_when_room_but_unprofitable():
    # потолок не достигнут, прогресс есть, но economic_value <= c_exp
    r = decide_stop(delta_d=0.02, d_best=0.6, ceil=0.9,
                    economic_value=50.0, cost_exp=100.0)
    assert r == STOP_NOT_ECONOMICAL


def test_decide_stop_stagnation_when_progress_died_but_money_remained():
    # потолок не достигнут, деньги ещё были бы, но Δd < eps → stagnation
    r = decide_stop(delta_d=0.001, d_best=0.6, ceil=0.9,
                    economic_value=500.0, cost_exp=100.0, eps=5e-3)
    assert r == STOP_STAGNATION


def test_decide_stop_not_economical_beats_stagnation():
    # нарушены И прогресс (Δd<eps), И экономика — приоритет not_economical
    r = decide_stop(delta_d=0.0, d_best=0.6, ceil=0.9,
                    economic_value=10.0, cost_exp=100.0)
    assert r == STOP_NOT_ECONOMICAL


def test_decide_stop_neutral_economics_falls_back_to_technical():
    # V=H=0 (экономика не задана) ⇒ economic_value=0 ⇒ экономический гейт всегда
    # нарушен; но при достигнутом потолке причина — ceil_reached (приоритет).
    val = economic_value(np.array([0.3]), volume=0.0, horizon=0.0)
    assert val == 0.0
    r = decide_stop(delta_d=0.02, d_best=0.95, ceil=0.9,
                    economic_value=val, cost_exp=0.0)
    assert r == STOP_CEIL
