# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 13 — ЦЕНА ИЗДЕЛИЯ как cost-цель ветки (REBUILD_SPEC §15.6 §3).

Канон §3: ρ (плотность) — ПОЛНОЦЕННЫЙ моделируемый отклик (GP), а цена за изделие
``price_изд = price_состав(x) · ρ̂(x)`` СОБИРАЕТСЯ на лету через
``make_item_cost_fn`` и входит в desirability ветки как ``min``-цель — БЕЗ
отдельного свойства «price» и без Шеффе-фита произведения. ``price_состав``
детерминирована (известные цены компонентов), единственный неизвестный множитель —
ρ.

Проверяем проводку cost в ядро runner:
  1) ``set_branch_cost`` требует, чтобы ρ было среди свойств оракула;
  2) с ценовой целью M8-argmax (``optimize_xbest``) выбирает рецепт ДЕШЕВЛЕ по
     цене изделия, чем без неё (цена реально давит рецепт);
  3) измеренный ``d_best`` ветки складывает цену по ИЗМЕРЕННОЙ ρ (раунд проходит);
  4) обратная совместимость: без ценовой цели поведение прежнее;
  5) эталон ``branch_optimum(cost_fn=...)`` с ценой по ИСТИНЕ тоже смещает
     оптимум в сторону удешевления.
"""
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.core.schema import ProjectSchema, VariableBlock, ModelSpec
from src.design.block_model import build_model_terms
from src.optimize.desirability import DesirabilitySpec
from src.verification.mixture_process_truth import MultiMixtureProcessTruth
from src.verification.branch_reference import branch_optimum
from src.apps.mixture_process_runner import MixtureProcessRunner

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# известные цены компонентов состава [усл.ед./кг] — C самый дешёвый
PRICE = {"A": 10.0, "B": 30.0, "C": 50.0}


def _coef(schema, contributions):
    terms = build_model_terms(schema)
    v = np.zeros(terms.p)
    for i, name in enumerate(terms.names):
        v[i] = float(contributions.get(name, 0.0))
    return v


def _build_truth():
    """3-компонентный мир {A,B,C} (без процесса) — мини-полигон для cost-проводки.

    q (max) тянет к дорогому C; ρ≈1 (линейный блендинг A=B=C=1 ⇒ на симплексе ρ=1
    точно), поэтому price_изд = price_состав и цена тянет к дешёвому A — честный
    спор «качество vs цена».
    """
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic")
    s = ProjectSchema.mixture_only(["A", "B", "C"], model=model)
    coef_by = {

        "q":   _coef(s, {"A": 1.0, "B": 5.0, "C": 9.0}),   # max → C
        "rho": _coef(s, {"A": 1.0, "B": 1.0, "C": 1.0}),   # ρ≡1 на симплексе
    }
    return MultiMixtureProcessTruth(s, coef_by, noise_sd=0.0)


def _model_schema():
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic")
    return ProjectSchema.mixture_only(["A", "B", "C"], model=model)



def _price_sostav(Xc):
    """Цена состава [усл.ед./кг]: детерминированная функция состава (не свойство)."""
    Xc = np.atleast_2d(np.asarray(Xc, float))
    w = np.array([PRICE["A"], PRICE["B"], PRICE["C"]], float)
    return Xc[:, :3] @ w


def _price_izd_truth(truth, x):
    """Цена изделия ПО ИСТИНЕ: price_состав·ρ_truth в одной точке."""
    xc = np.asarray(x, float).reshape(1, -1)
    return float(_price_sostav(xc)[0] * truth.truths["rho"].true(xc)[0])


# ----------------------------------------------------------------------
def test_set_branch_cost_requires_rho_property():
    truth = _build_truth()
    runner = MixtureProcessRunner(_model_schema(), truth,
                                  baseline=[1/3, 1/3, 1/3], seed=1, n_restarts=2)
    runner.begin_phase(mixture_free=["A", "B", "C"])
    runner.seed_initial(n=12, seed=1)
    runner.add_branch("q", {"q": DesirabilitySpec("max", low=0.0, high=10.0)},
                      budget=10, satisfy_at=1.1, branch_id="q")
    spec = DesirabilitySpec("min", low=10.0, high=50.0, weight=2.0)
    with pytest.raises(KeyError):
        runner.set_branch_cost("q", _price_sostav, spec, rho_property="nope")


def test_item_cost_pushes_argmax_cheaper():
    """§3: ценовая цель смещает M8-argmax к дешёвому рецепту (цена реально давит)."""
    truth = _build_truth()
    goal = {"q": DesirabilitySpec("max", low=0.0, high=10.0, weight=1.0)}
    price_spec = DesirabilitySpec("min", low=10.0, high=50.0, weight=2.0)

    # --- без ценовой цели: argmax тянет к дорогому C ---
    r0 = MixtureProcessRunner(_model_schema(), truth, baseline=[1/3, 1/3, 1/3],
                              seed=3, n_restarts=2)
    r0.begin_phase(mixture_free=["A", "B", "C"])
    r0.seed_initial(n=14, seed=3)
    r0.add_branch("q", goal, budget=10, satisfy_at=1.1, branch_id="q")
    x_free = np.asarray(r0.optimize_xbest("q").x, float)
    price_free = _price_izd_truth(truth, x_free)

    # --- с ценовой целью: argmax учитывает цену изделия (price_состав·ρ̂) ---
    r1 = MixtureProcessRunner(_model_schema(), truth, baseline=[1/3, 1/3, 1/3],
                              seed=3, n_restarts=2)
    r1.begin_phase(mixture_free=["A", "B", "C"])
    r1.seed_initial(n=14, seed=3)
    r1.add_branch("q", goal, budget=10, satisfy_at=1.1, branch_id="q")
    r1.set_branch_cost("q", _price_sostav, price_spec, rho_property="rho")
    x_cost = np.asarray(r1.optimize_xbest("q").x, float)
    price_cost = _price_izd_truth(truth, x_cost)

    # цена изделия с cost-целью строго дешевле, и рецепт сдвинут от C к A
    assert price_cost < price_free - 1e-3, (
        f"ценовая цель не удешевила рецепт: {price_cost:.2f} !< {price_free:.2f}")
    assert x_cost[0] > x_free[0], "cost-цель должна повысить долю дешёвого A"
    assert x_cost[2] < x_free[2], "cost-цель должна снизить долю дорогого C"


def test_item_cost_round_folds_measured_price():
    """Раунд ветки с ценой проходит и измеренный d_best включает цену по ρ."""
    truth = _build_truth()
    goal = {"q": DesirabilitySpec("max", low=0.0, high=10.0, weight=1.0)}
    price_spec = DesirabilitySpec("min", low=10.0, high=50.0, weight=2.0)
    runner = MixtureProcessRunner(_model_schema(), truth, baseline=[1/3, 1/3, 1/3],
                                  seed=5, n_restarts=2)
    runner.begin_phase(mixture_free=["A", "B", "C"])
    runner.seed_initial(n=14, seed=5)
    runner.add_branch("q", goal, budget=12, satisfy_at=1.1, branch_id="q")
    runner.set_branch_cost("q", _price_sostav, price_spec, rho_property="rho")
    for _ in range(2):
        runner.run_branch_round("q", n_points=4, explore_frac=0.2,
                                 n_candidates=300)
    br = runner.branches["q"]
    assert br.x_best is not None
    assert 0.0 <= br.d_best <= 1.0
    # рецепт сдвинут к дешёвому A (цена в цели): price_изд ниже, чем у дорогого C-края
    price_best = _price_izd_truth(truth, br.x_best)
    assert price_best < 50.0  # дешевле чистого C


def test_branch_optimum_costfn_shifts_reference():
    """Эталон с cost_fn (цена по ИСТИНЕ) смещает оптимум в сторону удешевления."""
    truth = _build_truth()
    goal = {"q": DesirabilitySpec("max", low=0.0, high=10.0, weight=1.0)}
    price_spec = DesirabilitySpec("min", low=10.0, high=50.0, weight=2.0)

    def cost_fn(Xc):
        Xc = np.atleast_2d(np.asarray(Xc, float))
        return _price_sostav(Xc) * np.asarray(truth.truths["rho"].true(Xc), float).ravel()

    opt_free = branch_optimum(truth, goal, n_scan=20000, seed=11)
    opt_cost = branch_optimum(truth, goal, n_scan=20000, seed=11,
                              cost_fn=cost_fn, cost_name="price",
                              cost_spec=price_spec)
    p_free = _price_izd_truth(truth, opt_free["x"])
    p_cost = _price_izd_truth(truth, opt_cost["x"])
    assert p_cost < p_free - 1e-3, (
        f"cost_fn не удешевил эталонный оптимум: {p_cost:.2f} !< {p_free:.2f}")
    # без cost_fn эталон требует cost_spec при заданном cost_fn (контракт)
    with pytest.raises(ValueError):
        branch_optimum(truth, goal, n_scan=2000, seed=1, cost_fn=cost_fn)
