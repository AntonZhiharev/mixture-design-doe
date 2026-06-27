# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 17 / §15.6 ШАГ 2 — цена за ИЗДЕЛИЕ через плотность ρ (§3).

Источник истины (physics-трактовка, решение сессии §15.6):

    price_изд = price_состав(A,B,C) · ρ(A,B,C,T,P)

Меньше ρ (вспенивание ПВХ) → легче изделие → больше изделий из того же сырья →
НИЖЕ цена за штуку. Проверяем:

  1. формула — умножение, и её монотонность (foam → дешевле);
  2. ρ как ОТДЕЛЬНЫЙ канал: T/P попадают в цену изделия ТОЛЬКО через ρ
     (числитель price_состав их не видит) — новый ценовой рычаг §3;
  3. `make_item_cost_fn` собирает composite-aware cost_fn над [x..., z...];
  4. интеграция в `optimize_desirability(cost_fn=...)` как обычное min-свойство.
"""
import numpy as np

from src.core.simplex import SimplexRegion
from src.optimize.desirability import (
    DesirabilitySpec, price_per_item, make_item_cost_fn, optimize_desirability,
)


# ----------------------------------------------------------------------
# 1. Формула price_изд = price_состав · ρ и её монотонность.
# ----------------------------------------------------------------------
def test_price_per_item_is_multiplication():
    # скаляр
    assert price_per_item(2.0, 3.0) == 6.0
    # вектор, поэлементно
    pc = np.array([2.0, 2.0, 4.0])
    rho = np.array([1.0, 0.5, 2.0])
    np.testing.assert_allclose(price_per_item(pc, rho), [2.0, 1.0, 8.0])


def test_foaming_lowers_item_price():
    """При фиксированной цене состава МЕНЬШЕ ρ ⇒ ДЕШЕВЛЕ изделие (§3, ПВХ)."""
    pc = 3.0
    dense = price_per_item(pc, rho=1.0)     # плотное изделие
    foamed = price_per_item(pc, rho=0.6)    # вспененное (легче)
    assert foamed < dense, (
        "вспенивание (меньше ρ) должно УДЕШЕВЛЯТЬ изделие — physics-трактовка §3; "
        f"получено foamed={foamed} >= dense={dense} (знак формулы вывернут?)")


# ----------------------------------------------------------------------
# 2/3. make_item_cost_fn: composite-aware, T/P входят в цену ТОЛЬКО через ρ.
# ----------------------------------------------------------------------
def _composition_price(X):
    """price_состав(A,B,C) = 2A+3B+4C — зависит ТОЛЬКО от состава (mixture)."""
    X = np.atleast_2d(np.asarray(X, float))
    A, B, C = X[:, 0], X[:, 1], X[:, 2]
    return 2.0 * A + 3.0 * B + 4.0 * C


def _rho_pred(X):
    """ρ(A,B,C,T,P): зависит от состава И режима. Выше P ⇒ ниже ρ (вспенивание)."""
    X = np.atleast_2d(np.asarray(X, float))
    A = X[:, 0]
    P = X[:, 4]
    return 1.2 - 0.5 * P + 0.2 * A         # P гонит ρ вниз → изделие легче


def test_item_cost_fn_composite_and_process_channel():
    cost_fn = make_item_cost_fn(_composition_price, _rho_pred)

    # один состав (A,B,C), два режима по P: только ρ-канал меняет цену изделия
    base = np.array([0.5, 0.3, 0.2])
    x_lowP = np.concatenate([base, [0.5, 0.0]]).reshape(1, -1)
    x_highP = np.concatenate([base, [0.5, 1.0]]).reshape(1, -1)

    pc_low = _composition_price(x_lowP)
    pc_high = _composition_price(x_highP)
    # числитель (цена состава) НЕ зависит от P — состав тот же:
    np.testing.assert_allclose(pc_low, pc_high)

    c_low = cost_fn(x_lowP)
    c_high = cost_fn(x_highP)
    # а цена ИЗДЕЛИЯ зависит — через ρ: выше P → ниже ρ → дешевле изделие:
    assert c_high[0] < c_low[0], (
        "process-переменная P должна двигать цену изделия ЧЕРЕЗ ρ "
        f"(c_highP={c_high[0]:.4f} !< c_lowP={c_low[0]:.4f}) — рычаг §3 не работает.")
    # и численно совпадает с прямой формулой price_состав · ρ:
    np.testing.assert_allclose(
        c_high, price_per_item(pc_high, _rho_pred(x_highP)))


# ----------------------------------------------------------------------
# 4. Интеграция в optimize_desirability как min-свойство (mixture×process).
# ----------------------------------------------------------------------
def test_item_cost_folds_into_optimizer_as_min_property():
    region = SimplexRegion(lower=[0.0, 0.0, 0.0], upper=[1.0, 1.0, 1.0])
    # одно качество-свойство (max strength) + цена изделия как min
    predictors = {
        "strength": lambda X: (6.0 * np.atleast_2d(X)[:, 0]
                               + 5.0 * np.atleast_2d(X)[:, 1]
                               + 4.0 * np.atleast_2d(X)[:, 2]),
    }
    specs = {"strength": DesirabilitySpec("max", low=2.0, high=12.0, weight=1.0)}
    cost_fn = make_item_cost_fn(_composition_price, _rho_pred)
    cost_spec = DesirabilitySpec("min", low=0.0, high=10.0, weight=2.0)

    res = optimize_desirability(
        region, predictors, specs,
        cost_fn=cost_fn, cost_spec=cost_spec, cost_name="price_item",
        n_candidates=1500, refine_iters=200, seed=7,
        process_lower=[0.0, 0.0], process_upper=[1.0, 1.0])

    # составной рецепт [A,B,C,T,P]; цена изделия попала в d_individual как min
    assert res.x.shape[0] == 5
    assert region.is_feasible(res.x[:3])
    assert "price_item" in res.d_individual
    assert 0.0 <= res.d_overall <= 1.0
    # цена с большим весом на min должна толкнуть P к 1 (минимум ρ → дешевле):
    assert res.x[4] > 0.5, (
        f"оптимизатор не использовал ρ-рычаг по P (P*={res.x[4]:.3f}); "
        "цена изделия с весом 2 должна тянуть P к минимуму ρ.")
