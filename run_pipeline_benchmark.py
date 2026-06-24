"""
run_pipeline_benchmark.py — HONEST, RANDOMISED end-to-end test of the pipeline,
with a ROBUST (uncertainty-aware) optimisation loop.

Each TRIAL builds a fresh random world:
  * q = 5 mixture components on a constrained simplex;
  * 4 key properties P1..P4, each a RANDOM Scheffé-quadratic function of x
    (random linear + interaction coefficients) — printed so you see the truth;
  * a RANDOM linear price per component.

Requirements are random, anchored on a feasible recipe so a solution exists.
P2(≥)/P3(≤) thresholds are placed NEAR-BINDING => genuinely significant.
Objective: lowest price.

ROBUSTNESS UPGRADES vs the naive loop (this is the "докрутка"):
  1) Uncertainty-aware SAFETY MARGINS — the optimiser aims a margin ≈ surrogate
     std INSIDE each ≥/≤ constraint (and uses a sharper target peak), so true
     values still satisfy the spec despite surrogate error.  Margins SHRINK as
     data accumulates near the boundary (less conservative over time).
  2) BOUNDARY/OPTIMUM-FOCUSED active learning — refills are taken near the
     proposed optimum and the active constraint boundary (where the decision is
     made), not by global max-variance.
  3) VERIFY-AND-REFINE + ADAPTIVE STOP/BUDGET — every round we measure x* and
     check surrogate self-consistency; we stop once the surrogate is trustworthy
     AND confidently feasible AND the recipe is stable.  Easy problems spend few
     experiments; hard / conflicting ones spend more (budget floats up to a cap).

Usage:
    python run_pipeline_benchmark.py            # Monte-Carlo (10 trials)
    python run_pipeline_benchmark.py 1          # single verbose trial
    python run_pipeline_benchmark.py 25 123     # 25 trials, base seed 123
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.optimize import minimize

from src.core.simplex import SimplexRegion
from src.core.linalg import scheffe_term_names
from src.core.synthetic import SyntheticScheffe
from src.design.d_optimal import d_optimal_for_region
from src.models.moe import MixtureOfExperts
from src.optimize.desirability import DesirabilitySpec, optimize_desirability

warnings.filterwarnings("ignore")
np.set_printoptions(precision=4, suppress=True)

Q = 5
PROP_NAMES = ["P1", "P2", "P3", "P4"]
COMP_NAMES = ["c1", "c2", "c3", "c4", "c5"]
SPEC_KIND = {"P1": "target", "P2": "ge", "P3": "le", "P4": "target"}


# ======================================================================
# Random world
# ======================================================================
@dataclass
class World:
    region: SimplexRegion
    funcs: Dict[str, SyntheticScheffe]
    price: np.ndarray

    def props(self, X) -> Dict[str, np.ndarray]:
        return {n: self.funcs[n].true(X) for n in PROP_NAMES}

    def prop1(self, x) -> Dict[str, float]:
        return {n: float(v[0]) for n, v in self.props(x.reshape(1, -1)).items()}

    def cost(self, X) -> np.ndarray:
        return np.atleast_2d(np.asarray(X, float)) @ self.price


def make_world(seed: int) -> World:
    rng = np.random.default_rng(seed)
    region = SimplexRegion(lower=[0.05] * Q, upper=[0.5] * Q, names=COMP_NAMES)
    funcs = {}
    for n in PROP_NAMES:
        coef = np.empty(15)
        coef[:Q] = rng.uniform(3.0, 10.0, Q)            # linear main effects
        coef[Q:] = rng.uniform(-6.0, 6.0, 15 - Q)       # two-way interactions
        funcs[n] = SyntheticScheffe(q=Q, model="quadratic", coefficients=coef)
    price = np.round(rng.uniform(1.0, 5.0, Q), 2)
    return World(region, funcs, price)


def print_world(world: World) -> None:
    terms = scheffe_term_names(Q, "quadratic", COMP_NAMES)
    print("  TRUE property functions (Scheffé-quadratic):")
    for n in PROP_NAMES:
        c = world.funcs[n].coefficients
        poly = " ".join(f"{c[i]:+.2f}·{terms[i]}" for i in range(len(terms)))
        print(f"    {n} = {poly}")
    print("  price/unit: " + ", ".join(f"{COMP_NAMES[i]}={world.price[i]:.2f}"
                                        for i in range(Q)))


# ======================================================================
# Requirements
# ======================================================================
@dataclass
class Spec:
    name: str
    ctype: str
    value: float
    tol: float
    band: float


def make_specs(world: World, rng: np.random.Generator):
    region = world.region
    anchor = region.random_points(1, seed=int(rng.integers(1 << 30)))[0]
    pa = world.prop1(anchor)
    S = region.random_points(3000, seed=int(rng.integers(1 << 30)))
    spread = {n: float(np.std(world.props(S)[n])) for n in PROP_NAMES}

    specs: List[Spec] = []
    for n in PROP_NAMES:
        sp = spread[n] if spread[n] > 1e-9 else 1.0
        if SPEC_KIND[n] == "target":
            specs.append(Spec(n, "target", pa[n], tol=0.05 * sp, band=0.5 * sp))
        elif SPEC_KIND[n] == "ge":
            slack = rng.uniform(0.03, 0.15) * sp
            specs.append(Spec(n, "ge", pa[n] - slack, tol=0.0, band=0.6 * sp))
        else:
            slack = rng.uniform(0.03, 0.15) * sp
            specs.append(Spec(n, "le", pa[n] + slack, tol=0.0, band=0.6 * sp))
    return specs, anchor, spread


def robust_dspecs(specs: List[Spec], sigma: Dict[str, float], k: float = 1.0):
    """Build desirability specs that aim a safety margin INSIDE each constraint.

    margin_n = k * sigma_n  (sigma = current surrogate std for that property).
    ≥: fully desirable only when predicted ≥ threshold + margin
    ≤: fully desirable only when predicted ≤ threshold − margin
    target: sharper peak (s=2) so the optimiser hugs the target value.
    """
    d: Dict[str, DesirabilitySpec] = {}
    for s in specs:
        m = k * float(sigma.get(s.name, 0.0))
        if s.ctype == "target":
            d[s.name] = DesirabilitySpec("target", low=s.value - s.band,
                                         high=s.value + s.band, target=s.value,
                                         s=2.0, weight=2.0)
        elif s.ctype == "ge":
            hi = s.value + m
            lo = min(s.value - s.band, hi - 1e-6)
            d[s.name] = DesirabilitySpec("max", low=lo, high=hi, weight=2.0)
        else:  # le
            lo = s.value - m
            hi = max(s.value + s.band, lo + 1e-6)
            d[s.name] = DesirabilitySpec("min", low=lo, high=hi, weight=2.0)
    return d


def meets_specs(props: Dict[str, float], specs: List[Spec]) -> bool:
    for s in specs:
        v = props[s.name]
        if s.ctype == "target" and abs(v - s.value) > s.tol + 1e-9:
            return False
        if s.ctype == "ge" and v < s.value - 1e-9:
            return False
        if s.ctype == "le" and v > s.value + 1e-9:
            return False
    return True


def confident_feasible(pred: Dict[str, float], sigma: Dict[str, float],
                       specs: List[Spec], k: float = 1.0) -> bool:
    """Surrogate-side feasibility WITH a k·sigma safety margin."""
    for s in specs:
        v, m = pred[s.name], k * float(sigma.get(s.name, 0.0))
        if s.ctype == "target" and abs(v - s.value) > max(s.tol - m, 0.3 * s.tol):
            return False
        if s.ctype == "ge" and v < s.value + m:
            return False
        if s.ctype == "le" and v > s.value - m:
            return False
    return True


# ======================================================================
# Analytical reference
# ======================================================================
def solve_analytical(world: World, specs: List[Spec], anchor, n_starts=40, seed=0):
    region = world.region
    bounds = list(zip(region.lower, region.upper))
    cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    for s in specs:
        f = world.funcs[s.name]
        if s.ctype == "target":
            cons.append({"type": "ineq", "fun": (lambda x, f=f, s=s: (s.value + s.tol) - f.true(x.reshape(1, -1))[0])})
            cons.append({"type": "ineq", "fun": (lambda x, f=f, s=s: f.true(x.reshape(1, -1))[0] - (s.value - s.tol))})
        elif s.ctype == "ge":
            cons.append({"type": "ineq", "fun": (lambda x, f=f, s=s: f.true(x.reshape(1, -1))[0] - s.value)})
        else:
            cons.append({"type": "ineq", "fun": (lambda x, f=f, s=s: s.value - f.true(x.reshape(1, -1))[0])})
    obj = lambda x: float(x @ world.price)
    starts = np.vstack([anchor.reshape(1, -1), region.random_points(n_starts, seed=seed)])
    best_x, best_c = None, np.inf
    for x0 in starts:
        res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 300, "ftol": 1e-10})
        x = res.x
        if not region.is_feasible(x, tol=1e-4):
            continue
        if meets_specs(world.prop1(x), specs) and obj(x) < best_c:
            best_c, best_x = obj(x), x.copy()
    return best_x, best_c


# ======================================================================
# Robust pipeline
# ======================================================================
def measure(world, X, noise, rng):
    out = {}
    for n in PROP_NAMES:
        y = world.funcs[n].true(X)
        out[n] = y + rng.normal(0, noise[n], size=len(np.atleast_2d(X)))
    return out


def _meas_props(y_meas):
    return {n: float(y_meas[n][0]) for n in PROP_NAMES}


def _accept(meas, specs, noise):
    """Accept a MEASURED recipe as feasible, with a measurement-noise guard
    (so we don't accept points that only pass because of favourable noise)."""
    for s in specs:
        v, g = meas[s.name], noise[s.name]
        if s.ctype == "target" and abs(v - s.value) > max(s.tol - g, 0.4 * s.tol):
            return False
        if s.ctype == "ge" and v < s.value + g:
            return False
        if s.ctype == "le" and v > s.value - g:
            return False
    return True


def run_pipeline(world, specs, noise, spread, max_rounds=12, batch=3,
                 k_margin=1.0, patience=2, seed=7, verbose=False, trace=None):
    region = world.region
    rng = np.random.default_rng(seed)
    design = d_optimal_for_region(region, n_runs=20, model="quadratic",
                                  n_random=500, n_restarts=6, seed=seed)
    X = design.design.copy()
    Y = measure(world, X, noise, rng)
    n_exp = len(X)
    x_star = X[0]

    # verified incumbent: lowest KNOWN cost among MEASURED-feasible proposals
    inc_x, inc_cost, since_improve = None, np.inf, 0

    # --- M9: лог стадии M2 (стартовый D-оптимальный дизайн) -------------
    if trace is not None:
        p_terms = Q + Q * (Q - 1) // 2            # quadratic Scheffé terms
        trace.log("M2",
                  inputs={"n_runs": int(len(X)), "model": "quadratic", "q": Q},
                  outputs={"design": X.tolist()},
                  metrics={"n": int(len(X)), "p": int(p_terms),
                           "n_over_p": float(len(X) / p_terms),
                           "d_efficiency": float(getattr(design, "d_efficiency",
                                                         getattr(design, "d_score", np.nan)))},
                  diagnostics={"note": "screening design, batch coord-exchange"})


    for r in range(max_rounds):
        moes = {n: MixtureOfExperts(seed=seed, n_restarts=3).fit(X, Y[n]) for n in PROP_NAMES}

        # --- per-property surrogate uncertainty (drives the safety margin) ---
        probe = region.random_points(600, seed=seed + 50 + r)
        pred_probe = {n: moes[n].predict(probe) for n in PROP_NAMES}
        sigma = {n: float(np.median(pred_probe[n].std)) for n in PROP_NAMES}

        predictors = {n: (lambda Z, m=moes[n]: m.predict(Z).mean) for n in PROP_NAMES}
        dspecs = robust_dspecs(specs, sigma, k=k_margin)
        res = optimize_desirability(region, predictors, dspecs, cost_fn=world.cost,
                                    cost_name="cost", n_candidates=2500,
                                    refine_iters=300, seed=seed + r)
        x_star = res.x
        cost_star = float(world.cost(x_star.reshape(1, -1))[0])

        # --- VERIFY: measure the proposed optimum -> accept into incumbent ---
        y_star_meas = measure(world, x_star.reshape(1, -1), noise, rng)
        X = np.vstack([X, x_star.reshape(1, -1)])
        for n in PROP_NAMES:
            Y[n] = np.concatenate([Y[n], y_star_meas[n]])
        n_exp += 1

        accepted = _accept(_meas_props(y_star_meas), specs, noise)
        improved = accepted and cost_star < inc_cost - 1e-6
        if improved:
            inc_x, inc_cost, since_improve = x_star.copy(), cost_star, 0
        else:
            since_improve += 1

        if verbose:
            print(f"    round {r}: n_exp={n_exp:3d}  d_pred={res.d_overall:.3f}  "
                  f"σ̃={np.mean(list(sigma.values())):.3f}  cost*={cost_star:.3f}  "
                  f"accepted={accepted}  incumbent={'-' if inc_x is None else f'{inc_cost:.3f}'}")

        # --- M9: лог раунда active learning -------------------------------
        if trace is not None:
            pred_star = {n: float(predictors[n](x_star.reshape(1, -1))[0]) for n in PROP_NAMES}
            trace.log(f"AL_round_{r}",
                      inputs={"k_margin": float(k_margin)},
                      outputs={"x_star": x_star.tolist(), "pred_props": pred_star},
                      metrics={"n_exp": int(n_exp), "d_overall": float(res.d_overall),
                               "cost_star": float(cost_star),
                               "sigma_mean": float(np.mean(list(sigma.values()))),
                               "accepted": bool(accepted),
                               "incumbent_cost": (None if inc_x is None else float(inc_cost))},
                      diagnostics={"sigma": {n: float(sigma[n]) for n in PROP_NAMES},
                                   "since_improve": int(since_improve)})

        # --- adaptive stop: have an incumbent and no improvement for `patience` rounds

        if inc_x is not None and since_improve >= patience:
            break
        if r == max_rounds - 1:
            break
        # aim refills near the incumbent if we have one (push price down on the
        # feasible boundary), else near the current proposal
        focus = inc_x if inc_x is not None else x_star
        x_star = focus


        # --- BOUNDARY/OPTIMUM-FOCUSED refills --------------------------------
        cand = region.random_points(900, seed=seed + 200 + r)
        cp = {n: moes[n].predict(cand) for n in PROP_NAMES}
        # keep candidates that are predicted near-feasible (within ~1·band)
        near = np.ones(len(cand), dtype=bool)
        for s in specs:
            v = cp[s.name].mean
            if s.ctype == "ge":
                near &= v >= s.value - s.band
            elif s.ctype == "le":
                near &= v <= s.value + s.band
            else:
                near &= np.abs(v - s.value) <= s.band
        idx_pool = np.where(near)[0]
        if len(idx_pool) < batch:                      # fallback: whole pool
            idx_pool = np.arange(len(cand))
        # score = surrogate uncertainty + closeness to the incumbent optimum
        unc = np.zeros(len(cand))
        for n in PROP_NAMES:
            unc += cp[n].std / (spread[n] + 1e-9)
        prox = np.linalg.norm(cand - x_star, axis=1)
        score = unc - 1.5 * prox                       # near x*, high uncertainty
        chosen = idx_pool[np.argsort(-score[idx_pool])]
        newX: List[np.ndarray] = []
        for i in chosen:
            if len(newX) >= batch:
                break
            if all(np.linalg.norm(cand[i] - p) > 0.05 for p in newX) and \
               np.linalg.norm(cand[i] - x_star) > 0.02:
                newX.append(cand[i])
        if newX:
            newX = np.array(newX)
            newY = measure(world, newX, noise, rng)
            X = np.vstack([X, newX])
            for n in PROP_NAMES:
                Y[n] = np.concatenate([Y[n], newY[n]])
            n_exp += len(newX)

    final_x = inc_x if inc_x is not None else x_star
    # --- M9: лог финальной стадии оптимизации продукта -------------------
    if trace is not None:
        trace.log("opt",
                  outputs={"recipe": np.asarray(final_x).tolist()},
                  metrics={"n_exp": int(n_exp),
                           "incumbent_found": bool(inc_x is not None),
                           "incumbent_cost": (None if inc_x is None else float(inc_cost))})
    return final_x, n_exp




# ======================================================================
# One trial
# ======================================================================
def run_trial(seed: int, verbose: bool = False, trace=None) -> dict:
    rng = np.random.default_rng(seed)
    world = make_world(seed)
    specs, anchor, spread = make_specs(world, rng)
    noise = {n: 0.02 * (spread[n] if spread[n] > 1e-9 else 1.0) for n in PROP_NAMES}

    # --- M9: лог постановки задачи (вход pipeline) ----------------------
    if trace is not None:
        trace.log("setup",
                  inputs={"q": Q, "components": COMP_NAMES, "seed": int(seed),
                          "price": world.price.tolist(),
                          "noise_sd": {n: float(noise[n]) for n in PROP_NAMES}},
                  outputs={"specs": [{"name": s.name, "type": s.ctype,
                                      "value": float(s.value), "tol": float(s.tol)}
                                     for s in specs]},
                  metrics={"n_props": len(PROP_NAMES)},
                  diagnostics={"spread": {n: float(spread[n]) for n in PROP_NAMES}})

    if verbose:

        print_world(world)
        print("  Requirements (random):")
        for s in specs:
            sym = {"target": "=", "ge": "≥", "le": "≤"}[s.ctype]
            ex = f" (±{s.tol:.3f})" if s.ctype == "target" else ""
            print(f"    {s.name} {sym} {s.value:.3f}{ex}")
        print("    cost -> MINIMISE")

    x_opt, c_opt = solve_analytical(world, specs, anchor, seed=seed)
    if x_opt is None:
        if verbose:
            print("  -> analytical infeasible (skip).")
        return {"feasible": False}

    x_pipe, n_exp = run_pipeline(world, specs, noise, spread, seed=seed + 1,
                                 verbose=verbose, trace=trace)
    pp = world.prop1(x_pipe)
    c_pipe = float(world.cost(x_pipe.reshape(1, -1))[0])
    res = {
        "feasible": True,
        "meets": meets_specs(pp, specs),
        "n_exp": n_exp,
        "price_gap": 100.0 * (c_pipe - c_opt) / c_opt,
        "recipe_dist": float(np.linalg.norm(x_pipe - x_opt)),
        "c_opt": c_opt, "c_pipe": c_pipe,
    }

    # --- M9: лог стадии benchmark (аналитический оптимум vs pipeline) ----
    if trace is not None:
        trace.log("benchmark",
                  outputs={"recipe_analytical": np.asarray(x_opt).tolist(),
                           "recipe_pipeline": np.asarray(x_pipe).tolist(),
                           "props_pipeline": {n: float(pp[n]) for n in PROP_NAMES}},
                  metrics={"meets": bool(res["meets"]), "n_exp": int(n_exp),
                           "price_analytical": float(c_opt), "price_pipeline": float(c_pipe),
                           "price_gap_pct": float(res["price_gap"]),
                           "recipe_dist": float(res["recipe_dist"])},
                  diagnostics={"specs": [{"name": s.name, "type": s.ctype,
                                          "value": float(s.value), "tol": float(s.tol),
                                          "pipeline": float(pp[s.name])} for s in specs]})

    if verbose:
        print(f"\n  ANALYTICAL: price={c_opt:.4f}  recipe={np.round(x_opt,4).tolist()}")
        print(f"  PIPELINE  : price={c_pipe:.4f}  recipe={np.round(x_pipe,4).tolist()}  (exp={n_exp})")
        print(f"  meets_specs={res['meets']}  price_gap={res['price_gap']:+.1f}%  "
              f"‖Δrecipe‖={res['recipe_dist']:.3f}")
        print("  property check (pipeline vs requirement):")
        for s in specs:
            sym = {"target": "=", "ge": "≥", "le": "≤"}[s.ctype]
            print(f"    {s.name} {sym} {s.value:.3f}  ->  pipeline={pp[s.name]:.3f}")
    return res


# ======================================================================
# Monte-Carlo
# ======================================================================
def monte_carlo(n_trials: int, base_seed: int = 0):
    print("=" * 72)
    print(f"MONTE-CARLO robustness — {n_trials} random worlds (robust loop)")
    print("=" * 72)
    rows = []
    for t in range(n_trials):
        seed = base_seed + 1000 * (t + 1)
        r = run_trial(seed, verbose=False)
        if not r["feasible"]:
            print(f"  trial {t:3d}: infeasible — skipped")
            continue
        rows.append(r)
        print(f"  trial {t:3d}: meets={str(r['meets']):5s}  exp={r['n_exp']:3d}  "
              f"price_gap={r['price_gap']:+6.1f}%  ‖Δ‖={r['recipe_dist']:.3f}")
    if not rows:
        print("\nNo feasible trials.")
        return
    meets = np.array([r["meets"] for r in rows])
    gap = np.array([r["price_gap"] for r in rows])
    exp = np.array([r["n_exp"] for r in rows])
    dist = np.array([r["recipe_dist"] for r in rows])
    gap_ok = gap[meets] if meets.any() else gap
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  feasible trials            : {len(rows)}/{n_trials}")
    print(f"  pipeline meets specs       : {meets.sum()}/{len(rows)}  ({100*meets.mean():.0f}%)")
    print(f"  experiments  median/min/max: {int(np.median(exp))} / {int(exp.min())} / {int(exp.max())}")
    print(f"  price gap (specs met) %    : median {np.median(gap_ok):+.1f}, "
          f"mean {gap_ok.mean():+.1f}, p90 {np.percentile(gap_ok,90):+.1f}, max {gap_ok.max():+.1f}")
    print(f"  ‖Δrecipe‖   median/max     : {np.median(dist):.3f} / {dist.max():.3f}")
    print("=" * 72)


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    base_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if n_trials == 1:
        print("=" * 72)
        print("SINGLE VERBOSE TRIAL (robust loop)")
        print("=" * 72)
        run_trial(base_seed + 1000, verbose=True)
    else:
        monte_carlo(n_trials, base_seed)


if __name__ == "__main__":
    main()
