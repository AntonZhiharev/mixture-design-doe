"""TEMP diagnostic (not committed): 4-component world {A,B,C,D} + new property
whiteStrength (min). REAL component prices A=95, B=200, C=23, D=315 rub/kg.

Question: at each branch's analytic optimum, do components stay 'in play'
(non-degenerate)? Tune truth coefficients so:
  * B does NOT degenerate in economy;
  * fast moves a bit off the 0-boundaries;
  * the new 'white' branch (whiteStrength min, D-driven) has an INTERIOR
    tradeoff (A,B,C,D all > 0) via pairwise goal conflicts.

price_sostav = 95A + 200B + 23C + 315D  [rub/kg]
price_izd    = price_sostav * rho        (per-branch density)
"""
import numpy as np
from src.core.schema import ProjectSchema, VariableBlock, ModelSpec
from src.design.block_model import build_model_terms, model_matrix

from src.optimize.desirability import DesirabilitySpec
from src.verification.mixture_process_truth import (MultiMixtureProcessTruth,
                                                    composite_random_points)
from src.verification.branch_reference import branch_optimum

COMPS = ["A", "B", "C", "D"]
PRICE = {"A": 95.0, "B": 200.0, "C": 23.0, "D": 315.0}
# branch -> (volume t/mo, density rho)
BRANCH_ECON = {
    "premium": (1.0, 1.0),
    "economy": (5.0, 1.1),
    "fast":    (10.0, 1.4),
    "white":   (1.0, 1.0),
}


def truth_schema():
    mix = VariableBlock.mixture(COMPS)
    proc = VariableBlock.process(["T", "P"], lower=[0.0, 0.0], upper=[1.0, 1.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="cubic",
                      process_order="quadratic")
    return ProjectSchema.mixture_process(mix, proc, model=model)


def coef(schema, contributions):
    terms = build_model_terms(schema)
    v = np.zeros(terms.p)
    for i, name in enumerate(terms.names):
        v[i] = float(contributions.get(name, 0.0))
    return v


def build_truth(noise_sd=0.0):
    s = truth_schema()
    coef_by = {
        # strength (max): B is ESSENTIAL via strong B-synergies; pure C is WEAK
        # (C linear small), so dropping B costs a lot of strength -> keeps B in
        # economy despite B being expensive. D slightly helps via B*D.
        "strength":  coef(s, {"A": 6, "B": 10, "C": 2, "D": 2,
                              "A:T": 5, "C:T": 3, "T^2": -3,
                              "A*B": 9, "A*C": 5, "B*C": 16, "A*B*C": 12,
                              "B*D": 4}),



        # gloss (max/target): B-led ridge; LOW base from A/C/P so reaching the
        # target=8 genuinely needs some B -> fast keeps B in play.
        "gloss":     coef(s, {"A": 3, "B": 7, "C": 3, "D": 2, "P": 6, "P^2": -4,
                              "A*B": 7, "B*C": 14, "A*C": 6, "A*B*C": 15}),
        # dry_time (min): A strongly fast, C slow -> fast must keep A; T-gated
        "dry_time":  coef(s, {"A": -4, "B": 2, "C": 4, "D": 1,
                              "T": -4, "C:T": -5, "T^2": 2}),
        # whiteStrength (MIN, better smaller): D is the whitener; ALL of A,B,C
        # whiten in PAIR with D (A*D/B*D/C*D < 0), so white wants a real BLEND
        # around D, not a single partner. A,B,C linear comparable (no single
        # 'cleanest' vertex). Interior tradeoff A,B,C,D all > 0.
        "whiteStrength": coef(s, {"A": 5, "B": 6, "C": 3, "D": 1,
                                  "C*D": -8, "A*D": -5, "B*D": -4}),

        # rho = DENSITY response (output of each experiment), linear blend
        # rho = 0.7A + B + 1.7C + 0.6D  -- a FACTOR of item price, not a goal here
        "rho":       coef(s, {"A": 0.7, "B": 1.0, "C": 1.7, "D": 0.6}),
        # price_sostav = 95A+200B+23C+315D  [rub/kg], linear (composition only)
        "price_sostav": coef(s, {k: PRICE[k] for k in COMPS}),
    }
    # price_izd = price_sostav * rho  (per-ITEM price, QUADRATIC in composition).
    # It lives in the Scheffe quadratic subspace -> fit its coef exactly by lstsq
    # so branch_optimum can read truth.truths["price"] like any other property.
    Xc = composite_random_points(s, 8000, seed=7)
    M = model_matrix(s, Xc, terms=build_model_terms(s))
    y_izd = (M @ coef_by["price_sostav"]) * (M @ coef_by["rho"])
    price_coef, *_ = np.linalg.lstsq(M, y_izd, rcond=None)
    coef_by["price"] = price_coef
    return MultiMixtureProcessTruth(s, coef_by, noise_sd=noise_sd)



def price_range(truth):
    # truth.truths["price"] is ALREADY price_izd = price_sostav*rho (per item)
    Xc = composite_random_points(truth.schema, 60000, seed=1)
    pi = truth.truths["price"].true(Xc)
    return float(pi.min()), float(pi.max())



def prop_range(truth, name):
    Xc = composite_random_points(truth.schema, 60000, seed=2)
    y = truth.truths[name].true(Xc)
    return float(y.min()), float(y.max())


def branch_goals(truth):
    goals = {}
    plo, phi = price_range(truth)
    for bid, (V, rho) in BRANCH_ECON.items():
        w_price = {"premium": 0.3, "economy": 1.0, "fast": 0.5, "white": 0.5}[bid]


        price_spec = DesirabilitySpec("min", low=plo, high=phi, weight=w_price)
        if bid == "premium":
            goals[bid] = {
                "strength": DesirabilitySpec("max", low=2.0, high=12.0, weight=1.0),
                "gloss":    DesirabilitySpec("max", low=1.0, high=13.0, weight=1.0),
                "price":    price_spec}
        elif bid == "economy":
            goals[bid] = {
                "price":    price_spec,
                "strength": DesirabilitySpec("max", low=2.0, high=12.0, weight=1.0),
                "dry_time": DesirabilitySpec("min", low=-3.0, high=3.0, weight=1.0)}
        elif bid == "fast":
            goals[bid] = {
                "dry_time": DesirabilitySpec("min", low=-3.0, high=3.0, weight=1.5),
                "gloss":    DesirabilitySpec("target", low=1.0, high=13.0,
                                             target=8.0, weight=1.0),
                "price":    price_spec}
        else:  # white: whiteStrength MIN (lead) + strength max (conflict on B/D)
               # + price min (conflict on D=315). Interior tradeoff.
            wlo, whi = prop_range(truth, "whiteStrength")
            goals[bid] = {
                "whiteStrength": DesirabilitySpec("min", low=wlo, high=whi, weight=2.0),
                "strength":      DesirabilitySpec("max", low=2.0, high=12.0, weight=1.0),
                "price":         price_spec}
    return goals


def main():
    truth = build_truth(noise_sd=0.0)
    goals = branch_goals(truth)
    print("4-comp {A,B,C,D}; prices A=95 B=200 C=23 D=315 rub/kg")
    print("new property whiteStrength (MIN, smaller=better)\n")
    THR = 0.05
    for i, (bid, goal) in enumerate(goals.items()):
        V, rho = BRANCH_ECON[bid]
        opt = branch_optimum(truth, goal, n_scan=60000, seed=100 + i)
        x = opt["x"]
        A, B, C, D = x[0], x[1], x[2], x[3]
        T, P = x[4], x[5]
        izd = float(truth.truths["price"].true(x.reshape(1, -1))[0])
        rho = float(truth.truths["rho"].true(x.reshape(1, -1))[0])
        psost = float(truth.truths["price_sostav"].true(x.reshape(1, -1))[0])
        drop = [nm for nm, val in zip(COMPS, (A, B, C, D)) if val < THR]
        print(f"[{bid}] V={V} t/mo  goals={list(goal)}")
        print(f"   A={A:.3f} B={B:.3f} C={C:.3f} D={D:.3f} | T={T:.2f} P={P:.2f}")
        print(f"   rho={rho:.2f}  price_sostav={psost:.0f}rub/kg  price_izd={izd:.0f}  d={opt['d']:.3f}")

        print(f"   DEGENERATE(<{THR}): {drop or 'none'}")
        for p in ("strength", "gloss", "dry_time", "whiteStrength"):
            yp = float(truth.truths[p].true(x.reshape(1, -1))[0])
            print(f"     {p}={yp:.2f}", end="")
        print("\n")


if __name__ == "__main__":
    main()
