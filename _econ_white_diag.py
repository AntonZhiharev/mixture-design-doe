"""Диагностика §5.3: фантомный EI у тяжёлой 4-комп ветки `white`.

Гоняем white в ПРОД-режиме — БЕЗ технического плеча (ceil=inf, eps~0), то есть
ceil_reached/stagnation отключены, единственный реальный стоп — экономический
(not_economical) или бюджет. Снимаем ТРАЕКТОРИЮ EI_price·V·H по раундам и
смотрим: спадает ли она ниже c_exp (σ_ρ калибрована, здоровая разведка) или
плато высоко (хронический фантом σ_ρ).

Дополнительно в argmax-EI кандидате печатаем σ ВСЕХ свойств — показать, что EI
сидит на ρ/цене (разведка), а не на лимитирующем whiteStrength.

Запуск:  .venv\\Scripts\\python.exe _econ_white_diag.py
"""
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from tests.unit.test_iteration13_battle import (
    _build_econ_truth, _econ_goals, _econ_model_schema, _comp_price,
    _COMPS4, _BRANCH_ECON, _item_price_truth_fn)
from src.apps.mixture_process_runner import MixtureProcessRunner
from src.verification.branch_reference import branch_optimum
from src.optimize.economic_stop import (expected_price_improvement,
                                         economic_value)

H_HORIZON, C_EXP = 12.0, 4500.0
BID = "white"
BUDGET = 45            # > step6(30), чтобы увидеть длинную траекторию
N_POINTS = 5


def main():
    truth = _build_econ_truth()
    goals, price_spec, (plo, phi), (wlo, whi) = _econ_goals(truth)
    item_price = _item_price_truth_fn(truth)

    # аналитический оптимум white С ценой — опорная точка %opt
    opt = branch_optimum(truth, goals[BID], n_scan=40000, seed=600 + 3,
                         cost_fn=item_price, cost_name="price",
                         cost_spec=price_spec[BID])
    d_opt = opt["d"]

    runner = MixtureProcessRunner(_econ_model_schema(), truth,
                                  baseline=[0.25, 0.25, 0.25, 0.25, 0.5, 0.5],
                                  seed=13, n_restarts=2)
    runner.begin_phase(mixture_free=_COMPS4, process_free=["T", "P"])
    runner.seed_initial(n=28, seed=13)

    br = runner.add_branch(BID, goals[BID], budget=BUDGET, satisfy_at=1.1,
                           branch_id=BID)
    runner.set_branch_cost(BID, _comp_price, price_spec[BID],
                           rho_property="rho", cost_name="price")
    br.volume = float(_BRANCH_ECON[BID][0])
    br.cost_exp = C_EXP
    br.horizon = H_HORIZON

    rho_i = runner.prop_index["rho"]
    props = list(runner.property_names)

    print(f"\n=== WHITE PROD-MODE DIAG (no ceil leg): c_exp={C_EXP:.0f} "
          f"V={br.volume:.0f} H={br.horizon:.0f} d_opt={d_opt:.3f} ===")
    print(f"price_izd range=[{plo:.0f},{phi:.0f}]  (EI gate threshold c_exp={C_EXP:.0f})")
    hdr = (f"{'round':>5}|{'spent':>5}|{'d_best':>7}|{'%opt':>5}|"
           f"{'price_best':>10}|{'maxEI$':>7}|{'EI*V*H':>8}|{'>c_exp?':>7}|"
           f"{'sd_rho@*':>9}|" + "|".join(f"sd_{p[:5]:>6}" for p in props))
    print(hdr)

    price_best = float("inf")
    rnd = 0
    while br.remaining() > 0:
        rnd += 1
        res = runner.run_branch_round(BID, n_points=N_POINTS, explore_frac=0.2,
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
        ev = economic_value(ei, br.volume, br.horizon)
        k = int(np.argmax(ei))                      # argmax-EI кандидат
        # σ ВСЕХ свойств в этом кандидате (разложение источника неопределённости)
        xk = cands[k:k+1]
        sd_all = {p: float(runner.surrogates[p].predict(xk).std[0]) for p in props}
        sd_rho_star = sd_all["rho"]
        pct = 100.0 * br.d_best / d_opt if d_opt > 0 else 0.0
        gate = "yes" if ev > C_EXP else "NO"
        row = (f"{rnd:>5}|{br.spent:>5}|{br.d_best:>7.3f}|{pct:>4.0f}%|"
               f"{price_best:>10.1f}|{float(ei[k]):>7.2f}|{ev:>8.1f}|{gate:>7}|"
               f"{sd_rho_star:>9.3f}|"
               + "|".join(f"{sd_all[p]:>8.3f}" for p in props))
        print(row)

    print("\nЧтение: если EI*V*H монотонно спадает и уходит в '>c_exp?=NO' — σ_ρ "
          "калибрована (здоровая разведка, экономстоп СРАБОТАЛ бы). Если плато "
          "высоко при '%opt'~100 — хронический фантом σ_ρ (sd_rho@* не спадает).")
    print("sd_*@* — постериорный σ каждого свойства в argmax-EI кандидате: "
          "EI сидит на ρ (цена), лимитирующий whiteStrength в EI НЕ входит.")


if __name__ == "__main__":
    main()
