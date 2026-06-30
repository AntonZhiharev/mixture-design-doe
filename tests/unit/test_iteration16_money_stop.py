# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 16 — шаг D: ФАКТИЧЕСКАЯ денежная нога стопа ветки, ЧИТАЮЩАЯ роль ρ.

§16.0 решил роль (ветка × отклик) и атрибуцию-крючок в ЧИСТЫХ функциях
``economic_stop`` (``rho_optimized`` зануляет ценовой σ_ρ-канал, Гр-1). Но раннер
денежную ногу не вызывал вовсе. Здесь проверяем, что ``MixtureProcessRunner``
доводит зануление канала до РЕАЛЬНОГО денежного VoI-гейта:

  * :meth:`branch_economic_value` — ₽ за горизонт через ``price_attributed_value``
    с ``rho_optimized=price_channel_suppressed(bid)`` (роль ρ читается end-to-end);
  * :meth:`branch_stop_decision` — двойной стоп §4 с этой денежной ногой.

Полная матрица покрытия (роль × сценарий), различающие (не декоративные)
проверки, + регрессия: при занулённом канале денежная нога НЕ фантомит
(``not_economical``/red-flag поднимается честно).

Замечание о трактовке «несколько ценовых ног». Архитектура (``set_branch_cost``)
держит ОДНУ ценовую ногу на ветку (один ρ + price_fn). «Несколько ног» здесь
читается как ветка с НЕСКОЛЬКИМИ качественными ногами d_i ПЛЮС ценовая нога:
именно знаменатель атрибуции (Σw всех ног) и роль ρ — то, что §16 проверяет на
многоного́й ветке.
"""
import numpy as np
import pytest

from src.core.schema import ProjectSchema, VariableBlock, ModelSpec
from src.design.block_model import build_model_terms
from src.optimize.desirability import DesirabilitySpec
from src.optimize.economic_stop import (STOP_NOT_ECONOMICAL,
                                         ECON_BINDING, ECON_ADVISORY)
from src.verification.mixture_process_truth import MultiMixtureProcessTruth
from src.apps.mixture_process_runner import MixtureProcessRunner


def _schema():
    mix = VariableBlock.mixture(["A", "B", "C"])
    proc = VariableBlock.process(["T", "P"], lower=[0.0, 0.0], upper=[1.0, 1.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    return ProjectSchema.mixture_process(mix, proc, model=model)


def _runner():
    """Раннер с тремя полноценными свойствами оракула + засеянный (GP готов)."""
    schema = _schema()
    p = build_model_terms(schema).p
    rng = np.random.default_rng(0)
    oracle = MultiMixtureProcessTruth(
        schema, {"p0": rng.normal(size=p), "p1": rng.normal(size=p),
                 "p2": rng.normal(size=p)}, noise_sd=0.0)
    r = MixtureProcessRunner(schema, oracle, seed=1, n_restarts=3,
                             baseline=[1 / 3, 1 / 3, 1 / 3, 0.5, 0.5])
    r.seed_initial(14)
    return r


def _price_fn(Xc):
    return np.ones(np.atleast_2d(Xc).shape[0], float)   # ₽/кг состава = const


def _set_economics(br, *, volume, cost_exp, horizon):
    br.volume = float(volume)
    br.cost_exp = float(cost_exp)
    br.horizon = float(horizon)


# ----------------------------------------------------------------------
# Матрица покрытия: роль ρ × сценарий ветки → денежная нога
# ----------------------------------------------------------------------
def test_branch_without_rho_has_no_money_leg():
    """Ветка вообще без ρ (нет ценовой ноги) ⇒ денежная нога = 0.0 (нет гейта)."""
    r = _runner()
    br = r.add_branch("tech", {"p0": DesirabilitySpec("max", low=-5, high=5)},
                      branch_id="bt")
    _set_economics(br, volume=100.0, cost_exp=1.0, horizon=10.0)
    assert r.branch_economic_value("bt") == 0.0


def test_rho_goal_without_price_has_no_money_leg():
    """ρ=цель-БЕЗ-цены: ρ (p1) в goal, но ценовая нога не задана ⇒ 0.0.

    Роль p1 = OPTIMIZED, однако без ``set_branch_cost`` денежного VoI-канала нет
    в принципе — атрибутировать нечего."""
    r = _runner()
    br = r.add_branch("dual_noprice",
                      {"p0": DesirabilitySpec("max", low=-5, high=5),
                       "p1": DesirabilitySpec("min", low=-5, high=5)},
                      branch_id="bd")
    _set_economics(br, volume=100.0, cost_exp=1.0, horizon=10.0)
    assert r.price_channel_suppressed("bd") is False   # нет цены ⇒ не занулён
    assert r.branch_economic_value("bd") == 0.0


def test_rho_price_input_channel_alive_gives_money():
    """ρ=цена-БЕЗ-цели (PRICE_INPUT): канал ЖИВ ⇒ денежная нога > 0."""
    r = _runner()
    br = r.add_branch("cost", {"p0": DesirabilitySpec("max", low=-5, high=5)},
                      branch_id="b2")
    r.set_branch_cost("b2", _price_fn,
                      DesirabilitySpec("min", low=0.0, high=10.0),
                      rho_property="p1")
    _set_economics(br, volume=100.0, cost_exp=1.0, horizon=10.0)
    assert r.price_channel_suppressed("b2") is False
    assert r.branch_economic_value("b2") > 0.0


def test_rho_optimized_suppresses_money_leg():
    """ρ=цель+цена (OPTIMIZED, Гр-1): канал занулён ⇒ денежная нога = 0.0 РОВНО.

    Двойной счёт одной δρ убран: σ_ρ уже оправдана качественной ногой d_ρ."""
    r = _runner()
    br = r.add_branch("dual", {"p0": DesirabilitySpec("max", low=-5, high=5),
                               "p1": DesirabilitySpec("min", low=-5, high=5)},
                      branch_id="b3")
    r.set_branch_cost("b3", _price_fn,
                      DesirabilitySpec("min", low=0.0, high=10.0),
                      rho_property="p1")
    _set_economics(br, volume=100.0, cost_exp=1.0, horizon=10.0)
    assert r.price_channel_suppressed("b3") is True
    assert r.branch_economic_value("b3") == 0.0


def test_branch_local_discrimination_same_oracle():
    """Branch-local (Гр-3): один оракул, одни числа — НО b2(ρ=цена)⇒₽>0,
    b3(ρ=цель+цена)⇒0. Разный РЕЗУЛЬТАТ доказывает, что роль реально читается."""
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
    for bid in ("b2", "b3"):
        _set_economics(r.branches[bid], volume=100.0, cost_exp=1.0, horizon=10.0)
    v_b2 = r.branch_economic_value("b2")
    v_b3 = r.branch_economic_value("b3")
    assert v_b2 > 0.0
    assert v_b3 == 0.0


def test_multi_goal_branch_role_still_read():
    """Многоного́я ветка (несколько качественных ног + цена): роль ρ всё равно
    читается. PRICE_INPUT-вариант ⇒ канал жив (>0); OPTIMIZED-вариант ⇒ 0."""
    r = _runner()
    # PRICE_INPUT: p0,p2 — качество, p1 — только цена (не цель)
    r.add_branch("multi_price",
                 {"p0": DesirabilitySpec("max", low=-5, high=5),
                  "p2": DesirabilitySpec("max", low=-5, high=5)},
                 branch_id="bm2")
    r.set_branch_cost("bm2", _price_fn,
                      DesirabilitySpec("min", low=0.0, high=10.0),
                      rho_property="p1")
    # OPTIMIZED: те же качества + p1 ещё и цель ⇒ канал занулён
    r.add_branch("multi_opt",
                 {"p0": DesirabilitySpec("max", low=-5, high=5),
                  "p2": DesirabilitySpec("max", low=-5, high=5),
                  "p1": DesirabilitySpec("min", low=-5, high=5)},
                 branch_id="bm3")
    r.set_branch_cost("bm3", _price_fn,
                      DesirabilitySpec("min", low=0.0, high=10.0),
                      rho_property="p1")
    for bid in ("bm2", "bm3"):
        _set_economics(r.branches[bid], volume=100.0, cost_exp=1.0, horizon=10.0)
    assert r.price_channel_suppressed("bm2") is False
    assert r.price_channel_suppressed("bm3") is True
    assert r.branch_economic_value("bm2") > 0.0
    assert r.branch_economic_value("bm3") == 0.0


# ----------------------------------------------------------------------
# Регрессия evaluate_stop/decide_stop: занулённый канал не «фантомит»
# ----------------------------------------------------------------------
def test_stop_decision_suppressed_channel_does_not_phantom_money():
    """OPTIMIZED-ρ (канал занулён): денежная нога=0 ⇒ под ECON_BINDING причина
    not_economical, econ_red_flag честно True (мнимой выгоды за σ_ρ нет)."""
    r = _runner()
    br = r.add_branch("dual", {"p0": DesirabilitySpec("max", low=-5, high=5),
                               "p1": DesirabilitySpec("min", low=-5, high=5)},
                      branch_id="b3")
    r.set_branch_cost("b3", _price_fn,
                      DesirabilitySpec("min", low=0.0, high=10.0),
                      rho_property="p1")
    _set_economics(br, volume=100.0, cost_exp=1.0, horizon=10.0)
    br.d_best = 0.5    # есть куда расти (d_best < ceil) — стоп НЕ из-за потолка
    dec = r.branch_stop_decision("b3", delta_d=0.1, ceil=0.9)
    assert dec.reason == STOP_NOT_ECONOMICAL   # денег нет ⇒ экономика связывает
    assert dec.econ_red_flag is True           # «невыгодно» поднято честно


def test_stop_decision_live_channel_money_is_real():
    """PRICE_INPUT-ρ (канал жив): при дешёвом раунде денежная нога > цены раунда
    ⇒ red-flag снят, технический прогресс не ветируется (reason=None)."""
    r = _runner()
    br = r.add_branch("cost", {"p0": DesirabilitySpec("max", low=-5, high=5)},
                      branch_id="b2")
    r.set_branch_cost("b2", _price_fn,
                      DesirabilitySpec("min", low=0.0, high=10.0),
                      rho_property="p1")
    # большой объём/горизонт и крошечная цена опыта ⇒ ev заведомо перекрывает N·c_exp
    _set_economics(br, volume=10000.0, cost_exp=1e-3, horizon=100.0)
    br.d_best = 0.5
    dec = r.branch_stop_decision("b2", delta_d=0.1, ceil=0.9)
    assert dec.econ_red_flag is False          # деньги реальны ⇒ не «невыгодно»
    assert dec.reason is None                   # тех. прогресс жив, экономика ок


def test_stop_decision_advisory_carries_red_flag_without_vetoing():
    """ECON_ADVISORY: занулённый канал (ev=0) НЕ ветирует живой техпрогресс
    (reason=None при Δd≥ε), но red-flag несётся для показа (A0.6 — не молча)."""
    r = _runner()
    br = r.add_branch("dual", {"p0": DesirabilitySpec("max", low=-5, high=5),
                               "p1": DesirabilitySpec("min", low=-5, high=5)},
                      branch_id="b3")
    r.set_branch_cost("b3", _price_fn,
                      DesirabilitySpec("min", low=0.0, high=10.0),
                      rho_property="p1")
    _set_economics(br, volume=100.0, cost_exp=1.0, horizon=10.0)
    br.d_best = 0.5
    dec = r.branch_stop_decision("b3", delta_d=0.1, ceil=0.9,
                                 econ_policy=ECON_ADVISORY)
    assert dec.reason is None                   # тех. прогресс не ветируется
    assert dec.econ_red_flag is True            # но «невыгодно» видно


def test_unknown_branch_raises():
    """Неизвестная ветка ⇒ KeyError (явная ошибка, не молча)."""
    r = _runner()
    with pytest.raises(KeyError):
        r.branch_economic_value("nope")
    with pytest.raises(KeyError):
        r.branch_stop_decision("nope", delta_d=0.1, ceil=0.9)
