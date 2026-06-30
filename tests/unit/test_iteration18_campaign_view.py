# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 18 — ШАГ 1: честный view-model кампании (ТЗ v1.1, §16/§16.1).

Проверяем READ-ONLY слой :mod:`src.apps.campaign` на реальном
:class:`MixtureProcessRunner` (тот же путь сборки, что в iteration16):

  * П-3 / Тр-3.3 — контекст ветки однозначен: один отклик носит РАЗНЫЕ роли в
    разных ветках, репорт всегда несёт branch_id/branch_name;
  * П-1 / П-9 — XOR внутри ветки: каждый отклик ровно с одной ролью, обратный
    индекс покрывает ВСЕ свойства без потерь/дублей;
  * П-6 / И-5 / Гр-1 — денежный канал ρ: zeroed при OPTIMIZED, alive при
    PRICE_INPUT;
  * §6 / Тр-6.3/6.6 — покрытие N/M честное; история не урезается (П-11);
  * §16.1 — объяснение «почему за ρ нет денег» (reason_code + текст);
  * сводка кампании JSON-сериализуема (для MCP/assistant).
"""
import json
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from src.core.schema import ProjectSchema, VariableBlock, ModelSpec
from src.design.block_model import build_model_terms
from src.design.branches import (ROLE_OPTIMIZED, ROLE_PRICE_INPUT,
                                  ROLE_REFERENCE)
from src.optimize.desirability import DesirabilitySpec
from src.verification.mixture_process_truth import MultiMixtureProcessTruth
from src.apps.mixture_process_runner import MixtureProcessRunner
from src.apps import campaign as cv

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ----------------------------------------------------------------------
def _schema():
    mix = VariableBlock.mixture(["A", "B", "C"])
    proc = VariableBlock.process(["T", "P"], lower=[0.0, 0.0], upper=[1.0, 1.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    return ProjectSchema.mixture_process(mix, proc, model=model)


def _seeded_runner(n_seed=14):
    """Runner с тремя свойствами (p0 цель, p1=ρ, p2 справка) и общей базой."""
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


def _price_fn(Xc):
    return np.ones(np.atleast_2d(Xc).shape[0], float)


def _add_cost_branch(r):
    """Ветка `cost`: ρ (p1) НЕ цель, но питает цену ⇒ PRICE_INPUT."""
    r.add_branch("cost", {"p0": DesirabilitySpec("max", low=-5, high=5)},
                 branch_id="b2")
    r.set_branch_cost("b2", _price_fn,
                      DesirabilitySpec("min", low=0.0, high=10.0),
                      rho_property="p1")


def _add_dual_branch(r):
    """Ветка `dual`: ρ (p1) И цель, И питает цену ⇒ OPTIMIZED (канал занулён)."""
    r.add_branch("dual", {"p0": DesirabilitySpec("max", low=-5, high=5),
                          "p1": DesirabilitySpec("min", low=-5, high=5)},
                 branch_id="b3")
    r.set_branch_cost("b3", _price_fn,
                      DesirabilitySpec("min", low=0.0, high=10.0),
                      rho_property="p1")


# ======================================================================
# П-1 / П-9 — XOR внутри ветки + полнота обратного индекса
# ======================================================================
def test_role_report_xor_and_completeness():
    r = _seeded_runner()
    _add_cost_branch(r)
    rep = cv.branch_role_report(r, "b2")

    # репорт несёт явный контекст ветки (Тр-3.3)
    assert rep["branch_id"] == "b2"
    assert rep["branch_name"] == "cost"
    assert rep["context_explicit"] is True

    # каждый отклик ровно с одной ролью; покрыты ВСЕ свойства оракула
    names = [resp["response"] for resp in rep["responses"]]
    assert sorted(names) == sorted(r.property_names)
    roles = {resp["response"]: resp["role"] for resp in rep["responses"]}
    assert roles == {"p0": ROLE_OPTIMIZED, "p1": ROLE_PRICE_INPUT,
                     "p2": ROLE_REFERENCE}

    # обратный индекс by_role — разбиение без потерь/дублей (П-9)
    flat = sum(rep["by_role"].values(), [])
    assert sorted(flat) == sorted(r.property_names)
    assert len(flat) == len(set(flat))


# ======================================================================
# П-3 / Тр-3.3 — один отклик, РАЗНЫЕ роли в разных ветках (линза ветки)
# ======================================================================
def test_role_is_branch_local_lens():
    r = _seeded_runner()
    _add_cost_branch(r)
    _add_dual_branch(r)

    role_b2 = {x["response"]: x["role"]
               for x in cv.branch_role_report(r, "b2")["responses"]}
    role_b3 = {x["response"]: x["role"]
               for x in cv.branch_role_report(r, "b3")["responses"]}
    # тот же p1 — PRICE_INPUT в b2 и OPTIMIZED в b3 (смена ветки меняет тег)
    assert role_b2["p1"] == ROLE_PRICE_INPUT
    assert role_b3["p1"] == ROLE_OPTIMIZED


# ======================================================================
# П-6 / И-5 / Гр-1 — денежный канал ρ: alive vs zeroed
# ======================================================================
def test_money_channel_alive_vs_zeroed():
    r = _seeded_runner()
    _add_cost_branch(r)
    _add_dual_branch(r)

    rep2 = cv.branch_role_report(r, "b2")
    rep3 = cv.branch_role_report(r, "b3")
    chan2 = {x["response"]: x["money_channel"] for x in rep2["responses"]}
    chan3 = {x["response"]: x["money_channel"] for x in rep3["responses"]}

    # b2: ρ=PRICE_INPUT ⇒ канал живой; b3: ρ=OPTIMIZED ⇒ канал занулён
    assert chan2["p1"] == cv.MONEY_ALIVE
    assert rep2["price_channel_suppressed"] is False
    assert chan3["p1"] == cv.MONEY_ZEROED
    assert rep3["price_channel_suppressed"] is True
    # у не-ρ откликов денежного канала нет
    assert chan2["p0"] is None and chan2["p2"] is None


# ======================================================================
# §6 / Тр-6.3 / П-11 — покрытие N/M честное, история не урезается
# ======================================================================
def test_coverage_full_when_all_measured():
    r = _seeded_runner(n_seed=14)
    cov = cv.response_coverage(r)
    base_n = len(r.points)
    assert base_n == 14
    for name in r.property_names:
        # оракул меряет ВСЕ свойства ⇒ measured == total == размер базы
        assert cov[name]["measured"] == base_n
        assert cov[name]["total"] == base_n
        assert cov[name]["fraction"] == 1.0

    _add_cost_branch(r)
    rep = cv.branch_role_report(r, "b2")
    # при полном покрытии флаг низкого покрытия не поднимается
    assert all(not resp["low_coverage"] for resp in rep["responses"])


def test_coverage_low_flag_with_high_threshold():
    """Доля-порог переносим (Тр-6.4): порог 1.5 (заведомо недостижим) ⇒ все
    отклики помечены низким покрытием — флаг работает по доле, не по абс. N."""
    r = _seeded_runner()
    _add_cost_branch(r)
    rep = cv.branch_role_report(r, "b2", coverage_fraction=1.5)
    assert all(resp["low_coverage"] for resp in rep["responses"])
    assert rep["coverage_fraction_threshold"] == 1.5


# ======================================================================
# §16.1 — объяснение «почему за ρ нет денег»
# ======================================================================
def test_money_explanation_no_price_leg():
    r = _seeded_runner()
    r.add_branch("plain", {"p0": DesirabilitySpec("max", low=-5, high=5)},
                 branch_id="b1")
    ex = cv.branch_money_explanation(r, "b1")
    assert ex["has_price_leg"] is False
    assert ex["reason_code"] == "no_price_leg"
    assert ex["economic_value"] == 0.0


def test_money_explanation_rho_optimized_zeroed():
    r = _seeded_runner()
    _add_dual_branch(r)
    ex = cv.branch_money_explanation(r, "b3", n_candidates=120, n_mc=64, seed=0)
    assert ex["has_price_leg"] is True
    assert ex["price_channel_suppressed"] is True
    assert ex["reason_code"] == "rho_optimized_zeroed"
    # денежная нога занулена (Гр-1): economic_value == 0
    assert ex["economic_value"] == 0.0
    assert "занул" in ex["text"].lower()


def test_money_explanation_price_input_alive():
    r = _seeded_runner()
    _add_cost_branch(r)
    ex = cv.branch_money_explanation(r, "b2", n_candidates=120, n_mc=64, seed=0)
    assert ex["has_price_leg"] is True
    assert ex["price_channel_suppressed"] is False
    assert ex["reason_code"] == "price_input_alive"
    assert "ALIVE" in ex["text"]


# ======================================================================
# Сводка кампании — JSON-сериализуема (для MCP/assistant)
# ======================================================================
def test_campaign_overview_is_json_serializable():
    r = _seeded_runner()
    _add_cost_branch(r)
    _add_dual_branch(r)
    ov = cv.campaign_overview(r, with_money=True, n_candidates=120, n_mc=64,
                              seed=0)
    # сериализуется без ошибок (значит, нет numpy/несериализуемых типов)
    s = json.dumps(ov, ensure_ascii=False)
    assert isinstance(s, str)
    assert ov["n_points"] == len(r.points)
    assert {b["id"] for b in ov["branches"]} == {"b2", "b3"}
    # занулённый канал виден в сводке именно у dual (b3)
    by_id = {b["id"]: b for b in ov["branches"]}
    assert by_id["b3"]["price_channel_suppressed"] is True
    assert by_id["b2"]["price_channel_suppressed"] is False
