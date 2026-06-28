# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 13 — §5 per-property: денежная ценность, атрибутированная цене.

Чистые (без GP/runner) юнит-тесты :func:`price_attributed_value`: деньги за
раунд засчитываются ТОЛЬКО за прирост d_overall ветки, идущий ЧЕРЕЗ цену.
Это чинит objective-agnostic «фантом дешёвого угла» :func:`economic_value`
(диагностика §5.3): дешёвый рецепт, который ветка не преследует (там её
d_overall не растёт), больше НЕ даёт денежной ценности.
"""
import numpy as np

from src.optimize.economic_stop import (price_attributed_value, economic_value)
from src.optimize.desirability import DesirabilitySpec, desirability_value


# 2-осевая цель: цена (вес 2) + прочность (вес 1) — Σw=3, w_price=2/3.
W_PRICE, W_STR = 2.0, 1.0
W_TOT = W_PRICE + W_STR


def _overall(dp, ds):
    """d_overall = взвешенное гео-среднее (dp^2 · ds^1)^(1/3) (как в M8)."""
    return float((dp ** W_PRICE * ds ** W_STR) ** (1.0 / W_TOT))


def _value(*, dp_cur, dp_cand, ds_cur, ds_cand, price_savings=50.0,
           volume=1.0, horizon=12.0):
    do_cur = _overall(dp_cur, ds_cur)
    do_cand = _overall(dp_cand, ds_cand)
    return price_attributed_value(
        np.atleast_1d(price_savings),
        d_overall_cur=do_cur, d_overall_cand=np.atleast_1d(do_cand),
        d_price_cur=dp_cur, d_price_cand=np.atleast_1d(dp_cand),
        price_weight=W_PRICE, total_weight=W_TOT,
        volume=volume, horizon=horizon)


def test_phantom_cheap_corner_yields_zero():
    """Дешёвый угол: цена падает (dp 0.4->0.6), но прочность рушится (0.6->0.2)
    так, что d_overall ПАДАЕТ -> α=0 -> денег 0 (фантом убран)."""
    v = _value(dp_cur=0.4, dp_cand=0.6, ds_cur=0.6, ds_cand=0.2)
    assert v == 0.0


def test_price_only_improvement_full_credit():
    """Улучшение чисто через цену (прочность не двигается) -> α≈1 ->
    economic_value ≈ price_savings·V·H (вся экономия засчитана)."""
    v = _value(dp_cur=0.4, dp_cand=0.6, ds_cur=0.5, ds_cand=0.5,
               price_savings=50.0, volume=2.0, horizon=12.0)
    assert abs(v - 50.0 * 2.0 * 12.0) < 1e-6


def test_other_axis_only_improvement_yields_zero():
    """Улучшение чисто через ДРУГУЮ ось (цена не двигается): вклад цены 0 ->
    α=0 -> денег 0 (хотя d_overall вырос)."""
    v = _value(dp_cur=0.5, dp_cand=0.5, ds_cur=0.4, ds_cand=0.6)
    assert v == 0.0


def test_mixed_improvement_partial_credit():
    """Растут обе оси: α∈(0,1) -> 0 < value < полная цена."""
    ps, V, H = 50.0, 1.0, 12.0
    v = _value(dp_cur=0.4, dp_cand=0.6, ds_cur=0.5, ds_cand=0.55,
               price_savings=ps, volume=V, horizon=H)
    assert 0.0 < v < ps * V * H


def test_veto_candidate_yields_zero():
    """Кандидат с вето (d_overall_cand=0): прироста нет -> денег 0."""
    v = price_attributed_value(
        np.array([50.0]), d_overall_cur=0.5, d_overall_cand=np.array([0.0]),
        d_price_cur=0.5, d_price_cand=np.array([0.9]),
        price_weight=W_PRICE, total_weight=W_TOT, volume=1.0, horizon=12.0)
    assert v == 0.0


def test_max_over_candidates_picks_genuine_over_phantom():
    """По набору кандидатов берётся максимум: фантомный дешёвый угол даёт 0,
    честный ценовой кандидат — полную цену; max выбирает честный."""
    ps = np.array([80.0, 50.0])           # phantom дешевле, но не преследуется
    dp_cur, ds_cur = 0.4, 0.5
    # cand0: phantom (цена 0.9, но прочность рушится 0.5->0.05 -> overall падает)
    # cand1: genuine (цена 0.6, прочность та же -> overall растёт через цену)
    dp_cand = np.array([0.9, 0.6])
    ds_cand = np.array([0.05, 0.5])
    do_cur = _overall(dp_cur, ds_cur)
    do_cand = np.array([_overall(0.9, 0.05), _overall(0.6, 0.5)])
    assert do_cand[0] < do_cur          # phantom реально снижает d_overall

    v = price_attributed_value(
        ps, d_overall_cur=do_cur, d_overall_cand=do_cand,
        d_price_cur=dp_cur, d_price_cand=dp_cand,
        price_weight=W_PRICE, total_weight=W_TOT, volume=1.0, horizon=10.0)
    # фантом (cand0) отсеян, выигрывает честный cand1 -> 50·1·10
    assert abs(v - 50.0 * 1.0 * 10.0) < 1e-6


def test_attributed_never_exceeds_agnostic():
    """Атрибуция — это ГЕЙТ: денежная ценность не больше objective-agnostic
    economic_value (α∈[0,1] только урезает)."""
    ps = np.array([50.0, 30.0])
    do_cur = _overall(0.4, 0.5)
    do_cand = np.array([_overall(0.6, 0.5), _overall(0.55, 0.5)])
    V, H = 3.0, 12.0
    attr = price_attributed_value(
        ps, d_overall_cur=do_cur, d_overall_cand=do_cand,
        d_price_cur=0.4, d_price_cand=np.array([0.6, 0.55]),
        price_weight=W_PRICE, total_weight=W_TOT, volume=V, horizon=H)
    agno = economic_value(ps, V, H)
    assert attr <= agno + 1e-9


def test_uses_desirability_value_consistently():
    """Санити: d_price из desirability_value(min-spec) подставляется штатно и
    функция не падает (интеграция со spec'ом ветки)."""
    spec = DesirabilitySpec("min", low=50.0, high=200.0, weight=W_PRICE)
    price_cur, price_cand = 150.0, 90.0          # дешевле -> d_price растёт
    dp_cur = float(desirability_value(price_cur, spec))
    dp_cand = float(desirability_value(price_cand, spec))
    assert dp_cand > dp_cur
    v = _value(dp_cur=dp_cur, dp_cand=dp_cand, ds_cur=0.5, ds_cand=0.5,
               price_savings=60.0, volume=1.0, horizon=12.0)
    assert v > 0.0
