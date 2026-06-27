# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 17 / §15.6 ШАГ 5 — hard/soft границы (A0.5) + денежная триада (§6).

Покрывает:
  * boundary_hits — детектор упора оптимума в границу (геометрия);
  * money_triad / boundary_signal — экономия/цена добычи/окупаемость; soft →
    триада, hard → HardBoundaryError (A0.5, движение запрещено по происхождению);
  * Runner: set_border_origin/border_origin (дефолт soft) + move_region отказ
    двигать hard-границу (A0.6 — не двигаем молча).
"""
import numpy as np
import pytest

from src.core.schema import ProjectSchema, VariableBlock
from src.design.move_bounds import (boundary_hits, BORDER_HARD, BORDER_SOFT,
                                     RegionMoveError)
from src.optimize.economic_stop import (money_triad, boundary_signal,
                                         MoneyTriad, HardBoundaryError)


def _schema(lower=(0.0, 0.0, 0.0), upper=(1.0, 1.0, 0.2)):
    mix = VariableBlock.mixture(["A", "B", "C"], lower=list(lower), upper=list(upper))
    return ProjectSchema(version=1, blocks=(mix,), responses=(), model=None)


# ----------------------------------------------------------------------
# boundary_hits — где оптимум упёрся (геометрия, без hard/soft).
# ----------------------------------------------------------------------
def test_boundary_hits_detects_upper_and_lower_and_interior():
    s = _schema(upper=(1.0, 1.0, 0.2))
    # C упёрся в верхнюю границу 0.2; A внутри; ничего по нижней
    hits = boundary_hits(s, {"A": 0.5, "B": 0.3, "C": 0.2})
    assert hits == {"C": "upper"}
    # строго внутри → пусто
    assert boundary_hits(s, {"A": 0.4, "B": 0.4, "C": 0.1}) == {}
    # нижняя граница (упор только по A; C внутри широкого кэпа)
    s2 = _schema(lower=(0.1, 0.0, 0.0), upper=(1.0, 1.0, 1.0))
    assert boundary_hits(s2, {"A": 0.1, "B": 0.6, "C": 0.3}) == {"A": "lower"}


# ----------------------------------------------------------------------
# money_triad — три цифры §6 + окупаемость vs горизонт.
# ----------------------------------------------------------------------
def test_money_triad_numbers_and_worth_it():
    # экономия 0.5 ₽/изд · 1000 изд = 500 ₽/период; добыча 10·20 = 200 ₽
    # окупаемость 200/500 = 0.4 периода ≤ H=2 ⇒ стоит
    t = money_triad("C", "upper", delta_price_item=0.5, volume=1000.0,
                    n_experiments=10, cost_exp=20.0, horizon=2.0)
    assert isinstance(t, MoneyTriad)
    assert t.saving_per_period == 500.0
    assert t.acquisition_cost == 200.0
    assert t.payback_periods == pytest.approx(0.4)
    assert t.worth_it is True


def test_money_triad_not_worth_when_payback_exceeds_horizon():
    # экономия мала → окупаемость 2000/10 = 200 периодов > H=5 ⇒ не стоит
    t = money_triad("C", "upper", delta_price_item=0.01, volume=1000.0,
                    n_experiments=100, cost_exp=20.0, horizon=5.0)
    assert t.payback_periods == pytest.approx(200.0)
    assert t.worth_it is False


def test_money_triad_infinite_payback_when_no_saving():
    t = money_triad("C", "upper", delta_price_item=0.0, volume=1000.0,
                    n_experiments=10, cost_exp=20.0, horizon=5.0)
    assert t.saving_per_period == 0.0
    assert t.payback_periods == float("inf")
    assert t.worth_it is False


# ----------------------------------------------------------------------
# boundary_signal — soft → триада; hard → отказ (A0.5).
# ----------------------------------------------------------------------
def test_boundary_signal_soft_returns_triad():
    t = boundary_signal("C", "upper", BORDER_SOFT, delta_price_item=0.5,
                        volume=1000.0, n_experiments=10, cost_exp=20.0,
                        horizon=2.0)
    assert isinstance(t, MoneyTriad) and t.worth_it is True


def test_boundary_signal_hard_refuses():
    with pytest.raises(HardBoundaryError):
        boundary_signal("C", "upper", BORDER_HARD, delta_price_item=0.5,
                        volume=1000.0, n_experiments=10, cost_exp=20.0,
                        horizon=2.0)


# ----------------------------------------------------------------------
# Runner: origin-реестр (дефолт soft) + move_region отказ двигать hard.
# ----------------------------------------------------------------------
def _runner():
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    from src.core.schema import ModelSpec
    from src.verification.mixture_process_truth import MultiMixtureProcessTruth
    from src.apps.mixture_process_runner import MixtureProcessRunner
    from src.design.block_model import build_model_terms

    mix = VariableBlock.mixture(["A", "B", "C"], lower=[0.0, 0.0, 0.0],
                                upper=[1.0, 1.0, 0.2])
    model = ModelSpec(mixture_order="quadratic")
    s = ProjectSchema(version=1, blocks=(mix,), responses=(), model=model)
    terms = build_model_terms(s)
    # одно свойство (strength) — достаточно для проверки origin/guard
    coef = {"strength": np.zeros(terms.p)}
    coef["strength"][0] = 6.0
    truth = MultiMixtureProcessTruth(s, coef, noise_sd=0.0)
    r = MixtureProcessRunner(s, truth, seed=0, n_restarts=2)
    r.seed_initial(n=12, seed=0)            # данные нужны для refit после move
    return r


def test_runner_border_origin_default_soft_and_settable():
    r = _runner()
    assert r.border_origin("C") == BORDER_SOFT           # дефолт — soft
    r.set_border_origin("C", BORDER_HARD)
    assert r.border_origin("C") == BORDER_HARD
    with pytest.raises(ValueError):
        r.set_border_origin("C", "rubber")               # невалидное origin


def test_move_region_refuses_hard_border():
    r = _runner()
    r.set_border_origin("C", BORDER_HARD)
    # движение hard-границы запрещено по происхождению (A0.5/A0.6)
    with pytest.raises(RegionMoveError):
        r.move_region({"C": (0.0, 1.0)}, intent="region_of_interest")
    # снятие hard-метки осознанно → движение снова разрешено
    r.set_border_origin("C", BORDER_SOFT)
    mv = r.move_region({"C": (0.0, 1.0)}, intent="region_of_interest")
    assert mv.region_after.mixture_block().upper[2] == 1.0

