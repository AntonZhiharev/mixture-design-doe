"""
run_iteration6_demo.py — REBUILD Iteration 6 demo (point 6 of the rebuild).

M8 product optimisation: multi-criteria Derringer-Suich desirability + cost,
maximised over the constrained mixture simplex.

Story: a PVC-like formulation with q=3 components and two measured properties
  * "flex"     — flexibility, larger-is-better;
  * "thermal"  — thermal stability, target value is best;
plus a recipe "cost" (smaller-is-better).  We fit a MoE surrogate (M6) for each
property from a synthetic design, then let M8 search the simplex for the recipe
that best balances all objectives and cost.

Usage:
    python run_iteration6_demo.py
"""
from pathlib import Path

import numpy as np

from src.core.simplex import SimplexRegion
from src.core.state import ProjectState
from src.models.moe import MixtureOfExperts
from src.optimize.desirability import DesirabilitySpec, optimize_desirability


# ----------------------------------------------------------------------
# Synthetic "true" properties (the lab we are emulating)
# ----------------------------------------------------------------------
def true_flex(X):
    """Flexibility: higher is better, favoured by component 1 (plasticiser)."""
    x = np.atleast_2d(X)
    return 8.0 * x[:, 1] + 3.0 * x[:, 0] * x[:, 2] + 2.0 * x[:, 0]


def true_thermal(X):
    """Thermal stability: peaks for a balanced recipe (target-is-best)."""
    x = np.atleast_2d(X)
    return 10.0 - 12.0 * np.sum((x - np.array([0.4, 0.3, 0.3])) ** 2, axis=1)


def recipe_cost(X):
    """Cost per unit: component 0 (resin) is the expensive one."""
    x = np.atleast_2d(X)
    return 5.0 * x[:, 0] + 1.5 * x[:, 1] + 1.0 * x[:, 2]


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    seed, q = 7, 3
    project_dir = Path("project_demo")
    region = SimplexRegion(q=q, names=["resin", "plasticiser", "filler"])

    print("=" * 70)
    print("REBUILD — Iteration 6 demo (M8 desirability optimisation on simplex)")
    print("=" * 70)

    # ---- training design + responses (with mild noise) ----------------
    rng = np.random.default_rng(seed)
    X = region.random_points(40, seed=seed)
    y_flex = true_flex(X) + rng.normal(0, 0.15, len(X))
    y_thermal = true_thermal(X) + rng.normal(0, 0.15, len(X))

    moe_flex = MixtureOfExperts(seed=seed, n_restarts=6).fit(X, y_flex)
    moe_thermal = MixtureOfExperts(seed=seed, n_restarts=6).fit(X, y_thermal)
    print(f"\n[M6] fitted MoE surrogates on n={len(X)} runs "
          f"(K_flex={moe_flex.n_regimes}, K_thermal={moe_thermal.n_regimes})")

    # predictors wrap the MoE mean prediction
    predictors = {
        "flex": lambda Z: moe_flex.predict(Z).mean,
        "thermal": lambda Z: moe_thermal.predict(Z).mean,
    }

    # ---- desirability specs (the product targets) ---------------------
    specs = {
        "flex": DesirabilitySpec("max", low=2.0, high=7.0, weight=1.0),
        "thermal": DesirabilitySpec("target", low=6.0, high=10.0,
                                    target=9.0, weight=1.5),
    }
    cost_spec = DesirabilitySpec("min", low=1.5, high=5.0, weight=1.0)

    # ---- M8 optimisation: properties only, then + cost ----------------
    res_props = optimize_desirability(region, predictors, specs,
                                      n_candidates=4000, refine_iters=600,
                                      seed=seed)
    print("\n[M8] best recipe by PROPERTIES only:")
    print("   " + res_props.summary().replace("\n", "\n   "))

    res_full = optimize_desirability(region, predictors, specs,
                                     cost_fn=recipe_cost, cost_spec=cost_spec,
                                     cost_name="cost", n_candidates=4000,
                                     refine_iters=600, seed=seed)
    print("\n[M8] best recipe with PROPERTIES + COST:")
    print("   " + res_full.summary().replace("\n", "\n   "))
    print(f"\n   cost pushed resin down: "
          f"{res_props.x[0]:.3f} -> {res_full.x[0]:.3f} "
          f"(cost {recipe_cost(res_props.x.reshape(1,-1))[0]:.2f} -> "
          f"{recipe_cost(res_full.x.reshape(1,-1))[0]:.2f})")

    # ---- checkpoint ---------------------------------------------------
    state = ProjectState.load(project_dir) if (project_dir / "state.json").exists() \
        else ProjectState(name="iteration6_demo", config={"q": q})
    state.set_stage("M8_optimization")
    state.put("m8_best_recipe", res_full.x)
    state.put("m8_best_d_overall", res_full.d_overall)
    state.put("m8_best_properties", res_full.properties)
    state.checkpoint(project_dir, label="after_M8")
    print(f"\n[ckpt] saved -> {project_dir/'checkpoints'/'after_M8.json'}")
    print(f"[ckpt] available: {state.list_checkpoints(project_dir)}")

    print("\n" + "=" * 70)
    print("Iteration 6 OK — M8 desirability balanced flexibility, thermal target "
          "and cost into one feasible optimal recipe on the simplex.")
    print("=" * 70)


if __name__ == "__main__":
    main()
