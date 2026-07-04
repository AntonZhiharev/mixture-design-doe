"""Диагностика §5.3 (контролируемая кривая): attr EI*V*H vs БЛИЗОСТЬ к оптимуму.

В battle GP сходится за ~1 раунд, поэтому «далёкого от оптимума» режима почти нет.
Здесь убираем артефакт скорости сходимости: берём ИСТИНУ как предиктор (без GP) и
ведём «текущий лучший рецепт» ветки по прямой от ПЛОХОГО старта (дешёвый, но цель
проседает) к АНАЛИТИЧЕСКОМУ оптимуму. На каждом шаге считаем obj-agnostic и
obj-attributed денежную ногу. Здоровый фикс: attr стартует > 0 (в идеале > c_exp)
ВДАЛИ от оптимума и МОНОТОННО спадает к 0 у оптимума; agnostic так и держит «фантом».
"""
import numpy as np

from tests.unit.test_iteration13_battle import (
    _build_econ_truth, _econ_goals, _comp_price, _item_price_truth_fn, _COMPS4)
from src.verification.mixture_process_truth import composite_random_points
from src.verification.branch_reference import branch_optimum
from src.optimize.desirability import Desirability, desirability_value
from src.optimize.economic_stop import (expected_price_improvement,
                                         economic_value, price_attributed_value)

import sys
BID = sys.argv[1] if len(sys.argv) > 1 else "economy"

V, H, C_EXP = 5.0, 12.0, 4500.0     # V=5: реалистичный объём (как economy)
RHO_REL_SD = 0.05                    # синтетический honest σ_ρ = 5% от μ_ρ

truth = _build_econ_truth()
goals, price_spec, _, _ = _econ_goals(truth)
goal = goals[BID]
pspec = price_spec[BID]
item_price = _item_price_truth_fn(truth)
rho_t = lambda X: np.asarray(truth.truths["rho"].true(np.atleast_2d(X)), float).ravel()

# полная цель ветки = свойства цели + ось цены (как в M8)
full = dict(goal); full["price"] = pspec
desir = Desirability(full)
w_price = pspec.weight
w_tot = sum(s.weight for s in full.values())

# аналитический оптимум ветки с ценой и «плохой» старт (cheap-but-bad угол)
opt = branch_optimum(truth, goal, n_scan=40000, seed=600 + 3,
                     cost_fn=item_price, cost_name="price", cost_spec=pspec)
x_opt = np.asarray(opt["x"], float)
d_opt = opt["d"]

# кандидаты для max_x EI / max_x attr (как в пайплайне)
cands = composite_random_points(truth.schema, 4000, seed=7)
pc_c = _comp_price(cands)
rho_c = rho_t(cands)
ei_std_c = RHO_REL_SD * np.abs(rho_c)

def _props(X):
    X = np.atleast_2d(X)
    p = {nm: np.asarray(truth.truths[nm].true(X), float).ravel() for nm in goal}
    p["price"] = item_price(X)
    return p

# плохой старт: самый ДЕШЁВЫЙ кандидат (минимальная цена изделия) — туда тянет
# «фантом», но whiteStrength там высокий (плохо), т.е. d_overall низкий.
price_c = item_price(cands)
x_bad = cands[int(np.argmin(price_c))]

print(f"branch={BID}  V={V:.0f}  H={H:.0f}  c_exp={C_EXP:.0f}  d_opt={d_opt:.3f}")
print(f"path: cheap-bad start -> analytic optimum (truth as predictor, no GP)")
print(f"{'t':>4}|{'d_cur/d_opt':>11}|{'price_best':>10}|{'maxEI$':>7}|"
      f"{'agno EI*V*H':>12}|{'attr EI*V*H':>12}")
for t in np.linspace(0.0, 1.0, 11):
    x_cur = (1.0 - t) * x_bad + t * x_opt
    x_cur[:4] = np.clip(x_cur[:4], 0, None)
    x_cur[:4] /= x_cur[:4].sum()             # держим валидный состав
    p_cur = _props(x_cur)
    do_cur = float(desir.overall(p_cur)[0])
    dp_cur = float(desirability_value(p_cur["price"][0], pspec))
    price_best = float(item_price(x_cur)[0])

    ei = expected_price_improvement(pc_c, rho_c, ei_std_c,
                                    price_best=price_best, seed=1)
    p_cand = _props(cands)
    do_cand = desir.overall(p_cand)
    dp_cand = desirability_value(p_cand["price"], pspec)

    agno = economic_value(ei, V, H)
    attr = price_attributed_value(ei, d_overall_cur=do_cur, d_overall_cand=do_cand,
                                  d_price_cur=dp_cur, d_price_cand=dp_cand,
                                  price_weight=w_price, total_weight=w_tot,
                                  volume=V, horizon=H)
    print(f"{t:>4.1f}|{do_cur/d_opt:>11.3f}|{price_best:>10.1f}|"
          f"{float(np.max(ei)):>7.1f}|{agno:>12.1f}|{attr:>12.1f}")

print(f"\nc_exp line = {C_EXP:.0f}. Healthy: attr starts > c_exp far from the "
      f"optimum (low d_cur) and decays to ~0 at the optimum; agnostic keeps the "
      f"phantom high throughout.")
