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
    ECON_BINDING, ECON_ADVISORY, StopDecision,
    expected_price_improvement, economic_value, decide_stop, evaluate_stop,
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


# ----------------------------------------------------------------------
# §4 ПОЛИТИКА ЭКОНОМИКИ — evaluate_stop(StopDecision) + ECON_BINDING/ADVISORY.
#
# Экономическая нога §4 атрибутирована цене (§5) и ПРИНЦИПИАЛЬНО уходит в 0 там,
# где ценовой рычаг исчерпан, а качественный headroom огромен (дешёвый-плохой
# угол: attr=0 при d≪ceil). Под ECON_BINDING это ветирует ещё-улучшаемую ветку
# (мина: низкий c_exp морозит ветку в худшей точке на t=0). ECON_ADVISORY делает
# техрычаг ведущим: пока Δd≥ε ∧ d<ceil — тянем, экономику несём как ред-флаг.
# ----------------------------------------------------------------------
def test_evaluate_stop_returns_stop_decision_with_red_flag():
    # тип результата — StopDecision(reason, econ_red_flag); невыгодно ⇒ флаг True
    dec = evaluate_stop(delta_d=0.02, d_best=0.6, ceil=0.9,
                        economic_value=50.0, cost_exp=100.0)
    assert isinstance(dec, StopDecision)
    assert dec.reason == STOP_NOT_ECONOMICAL and dec.econ_red_flag is True
    # выгодно ⇒ флаг False
    dec2 = evaluate_stop(delta_d=0.02, d_best=0.6, ceil=0.9,
                         economic_value=500.0, cost_exp=100.0)
    assert dec2.reason is None and dec2.econ_red_flag is False


def test_decide_stop_is_thin_reason_wrapper_over_evaluate_stop():
    # decide_stop == evaluate_stop(...).reason на наборе кейсов, обе политики
    cases = [
        dict(delta_d=0.02, d_best=0.6, ceil=0.9, economic_value=500.0, cost_exp=100.0),
        dict(delta_d=0.02, d_best=0.6, ceil=0.9, economic_value=50.0, cost_exp=100.0),
        dict(delta_d=0.0,  d_best=0.95, ceil=0.9, economic_value=0.0, cost_exp=100.0),
        dict(delta_d=0.001, d_best=0.6, ceil=0.9, economic_value=500.0, cost_exp=100.0),
    ]
    for c in cases:
        for pol in (ECON_BINDING, ECON_ADVISORY):
            assert decide_stop(econ_policy=pol, **c) == \
                   evaluate_stop(econ_policy=pol, **c).reason


def test_binding_is_default_and_preserves_canon():
    # дефолт = ECON_BINDING ⇒ нынешняя §4-каноника не меняется
    assert evaluate_stop(delta_d=0.02, d_best=0.6, ceil=0.9,
                         economic_value=50.0, cost_exp=100.0).reason == \
        evaluate_stop(delta_d=0.02, d_best=0.6, ceil=0.9, economic_value=50.0,
                      cost_exp=100.0, econ_policy=ECON_BINDING).reason


def test_advisory_does_not_freeze_branch_at_cheap_bad_start():
    """МИНА §5.3: дешёвый-плохой старт (attr=0) при d≪ceil и ЖИВОМ техпрогрессе.

    BINDING ветирует (not_economical) — ветка замёрзла бы в худшей точке.
    ADVISORY: техрычаг ведёт — reason=None (тянем), но ред-флаг ПОДНЯТ (показать).
    """
    mine = dict(delta_d=0.124, d_best=0.47, ceil=0.99,
                economic_value=0.0, cost_exp=700.0)
    # binding — мина срабатывает
    assert decide_stop(**mine) == STOP_NOT_ECONOMICAL
    # advisory — мина обезврежена, но экономика не исчезает молча
    dec = evaluate_stop(econ_policy=ECON_ADVISORY, **mine)
    assert dec.reason is None
    assert dec.econ_red_flag is True


def test_advisory_stops_on_stagnation_not_economics():
    # техпрогресс встал (Δd<ε), есть headroom, невыгодно: ADVISORY ⇒ stagnation
    # (НЕ not_economical), ред-флаг несёт экономику
    dec = evaluate_stop(delta_d=0.001, d_best=0.6, ceil=0.99,
                        economic_value=0.0, cost_exp=700.0,
                        econ_policy=ECON_ADVISORY)
    assert dec.reason == STOP_STAGNATION and dec.econ_red_flag is True


def test_advisory_ceiling_remains_absolute():
    # потолок — абсолютный приоритет и под ADVISORY (нечего улучшать)
    dec = evaluate_stop(delta_d=0.124, d_best=0.995, ceil=0.99,
                        economic_value=5000.0, cost_exp=700.0,
                        econ_policy=ECON_ADVISORY)
    assert dec.reason == STOP_CEIL and dec.econ_red_flag is False


def test_invalid_econ_policy_raises():
    import pytest
    with pytest.raises(ValueError):
        evaluate_stop(delta_d=0.02, d_best=0.6, ceil=0.9, economic_value=50.0,
                      cost_exp=100.0, econ_policy="whatever")


# ----------------------------------------------------------------------
# ПОЛОСА c_exp≈700: «гони» (attr>c_exp) vs «стоп» (хвост) на траектории
# economy (числа из _econ_curve_diag): пик ≈1084, attr=0 на дешёвом-плохом старте.
# При 1500·N=7500 полосы НЕТ (пик 1084 ≪ 7500) — экономика инертна. При c_exp≈700
# полоса появляется, и тут видно различие политик:
#   * BINDING тормозит на ПЕРВОМ же раунде (attr 22.7 ≤ 700) — мина на t=0;
#   * ADVISORY проходит ВСЮ полосу «гони» и останавливается ТЕХнически (стагнация),
#     неся ред-флаг там, где экономика просела.
# ----------------------------------------------------------------------
# (d_best как доля d_opt, attr EI·V·H) — t=0..1, economy-кривая (см. диагностику)
_TRAJ_D = [0.469, 0.593, 0.692, 0.772, 0.840, 0.898, 0.948, 0.965, 0.970,
           0.971, 0.9715]
_TRAJ_ATTR = [0.0, 22.7, 15.0, 615.3, 535.9, 1061.8, 1075.2, 1084.3, 692.6,
              159.6, 0.0]
_C_EXP_MID = 700.0          # ВНУТРИ кривой: 0 < 700 < пик 1084
_CEIL = 0.99               # не достигается (max 0.9715) → стоп будет техническим
_EPS = 5e-3


def _walk(policy):
    """Прогнать траекторию через ядро: вернуть (stop_index, decisions).
    stop_index — номер раунда (1-based), где reason != None впервые."""
    decisions, stop_index = [], None
    prev = _TRAJ_D[0]
    for i in range(1, len(_TRAJ_D)):
        delta = _TRAJ_D[i] - prev
        prev = _TRAJ_D[i]
        dec = evaluate_stop(delta_d=delta, d_best=_TRAJ_D[i], ceil=_CEIL,
                            economic_value=_TRAJ_ATTR[i], cost_exp=_C_EXP_MID,
                            eps=_EPS, econ_policy=policy)
        decisions.append(dec)
        if dec.reason is not None and stop_index is None:
            stop_index = i
    return stop_index, decisions


def test_cexp_band_exists_inside_curve():
    # c_exp≈700 реально внутри кривой: есть полоса «гони» (attr>c_exp) И хвост
    go = [a for a in _TRAJ_ATTR if a > _C_EXP_MID]
    stop = [a for a in _TRAJ_ATTR if a <= _C_EXP_MID]
    assert go and stop, "c_exp должен делить кривую на «гони» и «стоп»"
    assert max(_TRAJ_ATTR) > _C_EXP_MID > min(_TRAJ_ATTR)
    # при 1500·N=7500 полосы «гони» НЕТ (пик ≪ порога) — экономика инертна
    assert max(_TRAJ_ATTR) < 7500.0


def test_binding_freezes_at_cheap_bad_start_low_cexp():
    # BINDING + низкий c_exp: стоп на ПЕРВОМ раунде (attr 22.7 ≤ 700) → мина t=0
    stop_b, dec_b = _walk(ECON_BINDING)
    assert stop_b == 1
    assert dec_b[0].reason == STOP_NOT_ECONOMICAL


def test_advisory_tugs_through_go_band_then_technical_stop():
    # ADVISORY проходит всю полосу «гони» и стопает ТЕХнически (стагнация),
    # сильно позже мины BINDING; в точке стопа ред-флаг поднят (экономика просела)
    stop_a, dec_a = _walk(ECON_ADVISORY)
    stop_b, _ = _walk(ECON_BINDING)
    assert stop_a is not None and stop_a > stop_b      # тянул дольше мины
    assert dec_a[stop_a - 1].reason == STOP_STAGNATION  # стоп технический
    # до стопа экономика НЕ ветировала ни разу (price-only нога молчала)
    assert all(d.reason is None for d in dec_a[:stop_a - 1])
    # полоса «гони» реально пройдена: был раунд с attr>c_exp ДО стопа
    passed_go = any(_TRAJ_ATTR[i + 1] > _C_EXP_MID for i in range(stop_a - 1))
    assert passed_go
    # на техническом стопе экономика просела → ред-флаг показан пользователю
    assert dec_a[stop_a - 1].econ_red_flag is True

