"""Диагностика §5.3: ТРАЕКТОРИЯ EI*V*H по раундам (старт -> сходимость).

Проверяем, что objective-attributed денежная нога НЕ переобнулена: вдали от
оптимума (ранние раунды), где ценовой выигрыш реален, attributed EI*V*H должна
быть > 0 (в идеале > c_exp) и МОНОТОННО спадать к 0 по мере сходимости. Если
стартует у 0 — фикс переусерден.

Сравниваем построчно:
  * EI$         = max_x EI_price(x)            (₽/изд, сырое удешевление по ρ)
  * agno EI*V*H = economic_value (objective-agnostic, старый путь)
  * attr EI*V*H = price_attributed_value       (§5 per-property, новый путь)
Гоняем ветки economy (вес цены 2 — главный ценовой драйвер) и white (D дорогой).
"""
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from tests.unit.test_iteration13_battle import (
    _build_econ_truth, _econ_goals, _econ_model_schema, _comp_price,
    _item_price_truth_fn, _attributed_econ_value, _COMPS4, _BRANCH_ECON)
from src.apps.mixture_process_runner import MixtureProcessRunner
from src.verification.branch_reference import branch_optimum
from src.optimize.economic_stop import (expected_price_improvement,
                                         economic_value)

H_HORIZON, C_EXP = 12.0, 4500.0
BRANCHES = ["economy", "white", "premium", "fast"]
BUDGET = 60
SEED_N = 8          # МАЛЫЙ стартовый дизайн -> ветки стартуют ВДАЛИ от оптимума


truth = _build_econ_truth()
goals, price_spec, (plo, phi), (wlo, whi) = _econ_goals(truth)
item_price = _item_price_truth_fn(truth)

opt = {}
for i, bid in enumerate(BRANCHES):
    opt[bid] = branch_optimum(truth, goals[bid], n_scan=40000, seed=600 + i,
                              cost_fn=item_price, cost_name="price",
                              cost_spec=price_spec[bid])

runner = MixtureProcessRunner(_econ_model_schema(), truth,
                              baseline=[0.25, 0.25, 0.25, 0.25, 0.5, 0.5],
                              seed=13, n_restarts=2)
runner.begin_phase(mixture_free=_COMPS4, process_free=["T", "P"])
runner.seed_initial(n=SEED_N, seed=13)

rho_i = runner.prop_index["rho"]
for bid in BRANCHES:
    br = runner.add_branch(bid, goals[bid], budget=BUDGET, satisfy_at=1.1,
                           branch_id=bid)
    runner.set_branch_cost(bid, _comp_price, price_spec[bid],
                           rho_property="rho", cost_name="price")
    br.volume = float(_BRANCH_ECON[bid][0])
    br.cost_exp = C_EXP
    br.horizon = H_HORIZON

print(f"c_exp={C_EXP:.0f}  H={H_HORIZON:.0f}  ceil=99%*d_opt")
for bid in BRANCHES:
    br = runner.branches[bid]
    V = br.volume
    ceil = 0.99 * opt[bid]["d"]
    price_best = float("inf")
    print(f"\n=== branch {bid}  V={V:.0f}  d_opt={opt[bid]['d']:.3f}  "
          f"ceil={ceil:.3f}  price_w={price_spec[bid].weight:g} ===")
    print(f"{'round':>5}|{'spent':>5}|{'d_best':>7}|{'%opt':>5}|"
          f"{'price_best':>10}|{'EI$':>7}|{'agno EI*V*H':>12}|{'attr EI*V*H':>12}")
    rnd, at_ceil = 0, 0
    while br.remaining() > 0:
        rnd += 1

        res = runner.run_branch_round(bid, n_points=5, explore_frac=0.2,
                                      n_candidates=500)
        Ynew = np.atleast_2d(res["y_new"])
        Xnew = np.atleast_2d(res["x_new"])
        price_best = min(price_best,
                         float(np.min(_comp_price(Xnew) * Ynew[:, rho_i])))
        cands = runner._phase_candidates(500, runner.seed + br.spent)
        pc = _comp_price(cands)
        pred = runner.surrogates["rho"].predict(cands)
        ei = expected_price_improvement(pc, pred.mean, pred.std,
                                        price_best=price_best,
                                        seed=runner.seed + br.spent)
        ev_agno = economic_value(ei, V, H_HORIZON)
        ev_attr = _attributed_econ_value(runner, bid, cands, ei,
                                         goal_specs=goals[bid],
                                         price_spec=price_spec[bid],
                                         comp_price_fn=_comp_price,
                                         rho_name="rho")
        pct = 100.0 * br.d_best / opt[bid]["d"] if opt[bid]["d"] > 0 else 0.0
        print(f"{rnd:>5}|{br.spent:>5}|{br.d_best:>7.3f}|{pct:>4.0f}%|"
              f"{price_best:>10.1f}|{float(np.max(ei)):>7.2f}|"
              f"{ev_agno:>12.1f}|{ev_attr:>12.1f}")
        # стоп: сошлись (>=ceil) ДВА раунда подряд — траектория уже видна
        at_ceil = at_ceil + 1 if br.d_best >= ceil else 0
        if at_ceil >= 2 and rnd >= 4:
            break

print("\nReading: healthy fix => attr EI*V*H starts > 0 (ideally > c_exp) far "
      "from the optimum and decays to 0 at convergence; it must NOT start at 0.")


