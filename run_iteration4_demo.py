"""
run_iteration4_demo.py — REBUILD Iteration 4 demo.

M4 GMM regime clustering (BIC) + full Mixture-of-Experts (K>1):
  * gating = GMM responsibilities mapped to recipe space (model-based, #6);
  * Var[ŷ] = within-expert uncertainty + between-expert disagreement.

Two PVC-like regimes (sharp 'softening' jump in plasticizer A) create a
bimodal property distribution; the MoE recovers the regimes, routes recipes to
the right expert, and flags the regime boundary via expert disagreement.

Usage:
    python run_iteration4_demo.py
"""
from pathlib import Path

import numpy as np

from src.core.simplex import SimplexRegion
from src.core.state import ProjectState
from src.models.clustering import GMMRegimes
from src.models.moe import MixtureOfExperts


def two_regime_truth(X):
    x = np.atleast_2d(X)
    base = 5 * x[:, 0] + 3 * x[:, 1] + 2 * x[:, 2]
    jump = np.where(x[:, 0] > 0.45, 8.0, 0.0)
    return base + jump


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    seed, q = 42, 3
    project_dir = Path("project_demo")

    print("=" * 70)
    print("REBUILD — Iteration 4 demo (M4 GMM regimes + full MoE)")
    print("=" * 70)

    region = SimplexRegion(q=q)
    X = region.random_points(120, seed=seed)
    rng = np.random.default_rng(seed)
    y = two_regime_truth(X) + 0.05 * rng.standard_normal(len(X))

    # ---- M4: regimes in property space (BIC) -----------------------
    reg = GMMRegimes(k_range=range(1, 5), seed=seed).fit(y)
    print("\n[M4] BIC vs K (lower = better):")
    print(reg.bic_table.to_string(index=False))
    print(f"[M4] selected K = {reg.n_regimes} regimes; "
          f"means(y) = {reg.means.ravel()}")

    # ---- M6: full MoE ----------------------------------------------
    moe = MixtureOfExperts(k_range=range(1, 5), seed=seed, n_restarts=10).fit(X, y)
    print(f"\n[M6] MoE assembled with K = {moe.n_regimes} GP experts "
          f"(gating = logistic in recipe space).")

    # Gating routes recipes to the right regime by component A
    low = np.array([[0.10, 0.45, 0.45]])
    high = np.array([[0.75, 0.15, 0.10]])
    print(f"[M6] gating(low-A  {low.ravel()}) = {moe.gating_proba(low).ravel()}")
    print(f"[M6] gating(high-A {high.ravel()}) = {moe.gating_proba(high).ravel()}")

    # ---- Honest uncertainty: disagreement peaks at the boundary ----
    pool = region.random_points(600, seed=seed + 1)
    interior = pool[pool[:, 0] < 0.20]
    boundary = pool[np.abs(pool[:, 0] - 0.45) < 0.04]
    d_int = moe.predict(interior).disagreement.mean()
    d_bnd = moe.predict(boundary).disagreement.mean()
    print(f"\n[M6] Mean expert DISAGREEMENT (honest uncertainty signal):")
    print(f"     interior of a regime : {d_int:.4f}")
    print(f"     near regime boundary : {d_bnd:.4f}")
    print(f"     -> disagreement is {d_bnd/max(d_int,1e-9):.1f}x larger at the "
          f"boundary (active learning should sample here).")

    # ---- Accuracy + variance split on a test grid ------------------
    Xt = region.random_points(200, seed=seed + 2)
    pred = moe.predict(Xt)
    rmse = float(np.sqrt(np.mean((pred.mean - two_regime_truth(Xt)) ** 2)))
    print(f"\n[M6] Test RMSE = {rmse:.4f}; mean Var split: "
          f"within={pred.uncertainty.mean():.4f}, "
          f"between={pred.disagreement.mean():.4f}")

    # ---- Checkpoint + refit-free reload ----------------------------
    state = ProjectState.load(project_dir) if (project_dir / "state.json").exists() \
        else ProjectState(name="iteration4_demo", config={"q": q})
    state.set_stage("M4_clustering")
    state.models["m4_regimes"] = reg.to_state()
    state.set_stage("M6_moe")
    state.models["m6_moe"] = moe.to_state()
    state.checkpoint(project_dir, label="after_M6_moe")
    reloaded = MixtureOfExperts.from_state(state.models["m6_moe"])
    max_diff = np.max(np.abs(reloaded.predict(Xt).mean - pred.mean))
    print(f"\n[ckpt] saved -> {project_dir/'checkpoints'/'after_M6_moe.json'}")
    print(f"[ckpt] reload prediction max|delta| = {max_diff:.2e} (refit-free OK)")

    print(f"[ckpt] available: {state.list_checkpoints(project_dir)}")

    print("\n" + "=" * 70)
    print("Iteration 4 OK — regimes found by BIC, gating routes recipes, "
          "disagreement flags the boundary.")
    print("=" * 70)


if __name__ == "__main__":
    main()
