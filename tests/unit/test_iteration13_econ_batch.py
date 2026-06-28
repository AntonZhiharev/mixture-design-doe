# Copyright 2025 The mixture-design-doe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""§4-BATCH: q-EI «улучшение лучшей из N» (max-тип) + N*.

Проверяем КАНОН связки стопа: левая часть — БАТЧ best-of-N (НЕ max_x одной точки),
правая — N·c_exp. Ключевые свойства, без которых «×N точек» снова перекосит стоп:

  * best-of-1 == max-single (одна точка = старый маржинальный путь);
  * монотонность по N (больше точек — не хуже);
  * ВОГНУТОСТЬ: маржинальные приросты НЕ возрастают (best-of-N сублинейна) —
    это и даёт интерьерный N* против линейной N·c_exp;
  * N* = max{k: Δvalue(k) > c_exp} согласован с кривой (граничные c_exp → 0 / N_max);
  * атрибуция §5: α=0 ⇒ денег 0 (фантом дешёвого угла не воскресает в батче).
"""
import numpy as np

from src.optimize.economic_stop import (
    best_of_n_value, best_of_n_curve_value, optimal_round_size,
    expected_price_improvement, economic_value, price_attribution_alpha)


def _candidates():
    """Разнородные кандидаты: цена состава растёт, ρ~N(μ,σ) с заметным σ —
    у best-of-N есть из чего выбирать (разные хвосты удешевления)."""
    n = 12
    comp_price = np.linspace(20.0, 120.0, n)        # ₽/кг, все разные
    rho_mean = np.full(n, 1.0)
    rho_std = np.full(n, 0.25)                       # honest σ_ρ
    price_best = 150.0                               # порог EI (₽/изд)
    return comp_price, rho_mean, rho_std, price_best


def test_best_of_one_equals_max_single():
    """best-of-1 = max_x EI·V·H (тот же seed/n_mc ⇒ те же MC-draws): батч при N=1
    в точности воспроизводит старый маржинальный путь."""
    cp, mu, sd, pb = _candidates()
    V, H, S, seed = 5.0, 12.0, 4000, 7
    curve = best_of_n_curve_value(cp, mu, sd, pb, n_max=1, volume=V, horizon=H,
                                  n_mc=S, seed=seed)
    ei = expected_price_improvement(cp, mu, sd, price_best=pb, n_mc=S, seed=seed)
    assert curve.shape == (1,)
    assert np.isclose(curve[0], economic_value(ei, V, H), rtol=1e-12, atol=1e-9)


def test_monotone_and_concave():
    """Кривая best-of-N неубывающая и ВОГНУТАЯ (маржиналь не возрастает).

    На ФИКСИРОВАННОМ MC-матриксе E[max] — монотонная submodular функция множества,
    greedy даёт маржинали, не возрастающие ТОЧНО (без MC-допуска)."""
    cp, mu, sd, pb = _candidates()
    curve = best_of_n_curve_value(cp, mu, sd, pb, n_max=12, volume=3.0,
                                  horizon=10.0, n_mc=6000, seed=11)
    marg = np.diff(np.concatenate(([0.0], curve)))
    assert np.all(np.diff(curve) >= -1e-9)           # монотонна
    assert np.all(np.diff(marg) <= 1e-9)             # вогнута (Δ маржинали ≤ 0)
    assert curve[-1] >= curve[0]                     # best-of-N ≥ max-single


def test_optimal_round_size_interior_and_edges():
    """N* = max{k: Δvalue(k) > c_exp}: интерьерный при умеренном c_exp, и
    краевые случаи c_exp→0 / c_exp→∞ дают N_max / 0."""
    cp, mu, sd, pb = _candidates()
    curve = best_of_n_curve_value(cp, mu, sd, pb, n_max=12, volume=3.0,
                                  horizon=10.0, n_mc=6000, seed=11)
    marg = np.diff(np.concatenate(([0.0], curve)))

    # интерьерный c_exp между маржиналью 1-й и последней точки
    c_mid = 0.5 * (marg[0] + marg[-1])
    nstar = optimal_round_size(curve, c_mid)
    assert 1 <= nstar < len(curve)
    # согласованность с определением: маржиналь N*-й > c_exp, (N*+1)-й ≤ c_exp
    assert marg[nstar - 1] > c_mid
    assert marg[nstar] <= c_mid

    assert optimal_round_size(curve, 0.0) == len(curve)        # всё окупается
    assert optimal_round_size(curve, marg[0] + 1.0) == 0       # даже 1-я нет


def test_attribution_zero_kills_money():
    """§5 в батче: α=0 ⇒ денег 0 (фантом не воскресает); смешанная α — деньги
    только от кандидатов с α>0, и батч ≤ безатрибутивного."""
    cp, mu, sd, pb = _candidates()
    V, H = 5.0, 12.0
    n = cp.shape[0]
    # α=0 везде → 0
    z = np.zeros(n)
    assert best_of_n_value(cp, mu, sd, pb, n_batch=5, volume=V, horizon=H,
                           alpha=z, n_mc=4000, seed=3) == 0.0
    # смешанная α: половина нулей
    a = np.where(np.arange(n) % 2 == 0, 1.0, 0.0)
    v_attr = best_of_n_value(cp, mu, sd, pb, n_batch=5, volume=V, horizon=H,
                             alpha=a, n_mc=4000, seed=3)
    v_full = best_of_n_value(cp, mu, sd, pb, n_batch=5, volume=V, horizon=H,
                             alpha=None, n_mc=4000, seed=3)
    assert 0.0 < v_attr <= v_full + 1e-9


def test_alpha_helper_matches_canon():
    """price_attribution_alpha воспроизводит каноническую α (вынос без копий):
    рост чисто через цену ⇒ α≈1; нет роста цели ⇒ α=0."""
    # кандидат A: и d_overall, и d_price растут (рост через цену) → α≈1
    # кандидат B: d_overall падает (цель не растёт) → α=0
    alpha = price_attribution_alpha(
        d_overall_cur=0.40, d_overall_cand=np.array([0.60, 0.30]),
        d_price_cur=0.40, d_price_cand=np.array([0.60, 0.90]),
        price_weight=1.0, total_weight=1.0)            # цена — единственная ось
    assert np.isclose(alpha[0], 1.0, atol=1e-9)        # вся выгода через цену
    assert alpha[1] == 0.0                             # d_overall упал → денег нет
