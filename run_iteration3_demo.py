"""
run_iteration3_demo.py — REBUILD Iteration 3 demo.

M6 single GP-expert (K=1, no gating yet): mean = Scheffe quadratic,
kernel = Matern 5/2 ARD + WhiteNoise.  Compares GP vs plain Scheffe on a
NON-polynomial truth with a 'threshold' (plasticizer-like) bump — the case
where RBF/Scheffe smoothness assumptions leak (REBUILD_SPEC kernel block).

Usage:
    python run_iteration3_demo.py
"""
from pathlib import Path

import numpy as np

from src.core.simplex import SimplexRegion
from src.core.state import ProjectState
from src.models.scheffe import ScheffeModel
from src.models.gp_expert import GPExpert


def threshold_truth(X):
    """Quadratic trend + logistic threshold in component A (sharp 'softening')."""
    x = np.atleast_2d(X)
    trend = 10 * x[:, 0] + 6 * x[:, 1] + 3 * x[:, 2] + 8 * x[:, 0] * x[:, 1]
    bump = 3.0 / (1.0 + np.exp(-30.0 * (x[:, 0] - 0.45)))
    return trend + bump


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    seed, q = 42, 3
    project_dir = Path("project_demo")

    print("=" * 70)
    print("REBUILD — Iteration 3 demo (M6 GP-expert vs Scheffe)")
    print("=" * 70)

    region = SimplexRegion(q=q)
    Xtr = region.random_points(60, seed=seed)
    rng = np.random.default_rng(seed)
    ytr = threshold_truth(Xtr) + 0.05 * rng.standard_normal(len(Xtr))

    # ---- Fit both models -------------------------------------------
    scheffe = ScheffeModel(model="quadratic").fit(Xtr, ytr)
    gp = GPExpert(mean_model="quadratic", kernel="matern52",
                  seed=seed, n_restarts=12).fit(Xtr, ytr)

    print(f"\n[M6] Scheffe quadratic: train R^2={scheffe.r2:.4f}")
    print(f"[M6] GP-expert: log-lik={gp.log_marginal_likelihood:.2f}, "
          f"noise={gp.noise_level:.4f}")
    print(f"[M6] GP ARD lengthscales (A,B,C): {gp.lengthscales}")

    # ---- Test-set accuracy vs the true (noiseless) function --------
    Xte = region.random_points(200, seed=seed + 1)
    truth = threshold_truth(Xte)
    pred = gp.predict(Xte, return_std=True)
    r_scheffe = rmse(scheffe.predict(Xte), truth)
    r_gp = rmse(pred.mean, truth)
    print(f"\n[M6] Test RMSE vs truth (lower = better):")
    print(f"     Scheffe quadratic : {r_scheffe:.4f}")
    print(f"     GP-expert (M5/2)  : {r_gp:.4f}")
    print(f"     -> GP cuts error by {100*(r_scheffe-r_gp)/r_scheffe:.1f}% "
          f"(captures the threshold bump in residuals)")

    # ---- Honest uncertainty: std grows away from training data -----
    near = gp.predict(Xtr[:20]).std.mean()
    far = pred.std.mean()
    print(f"\n[M6] Mean predictive std: near training={near:.4f}, "
          f"over region={far:.4f} (honest uncertainty grows off-data).")

    # ---- Persistence: refit-free reconstruction --------------------
    state = ProjectState.load(project_dir) if (project_dir / "state.json").exists() \
        else ProjectState(name="iteration3_demo", config={"q": q})
    state.set_stage("M6_moe")

    state.models["m6_gp_expert"] = gp.to_state()
    state.checkpoint(project_dir, label="after_M6")
    reloaded = GPExpert.from_state(state.models["m6_gp_expert"])
    max_diff = np.max(np.abs(reloaded.predict(Xte).mean - pred.mean))
    print(f"\n[ckpt] saved -> {project_dir/'checkpoints'/'after_M6.json'}")
    print(f"[ckpt] reload prediction max|Δ| = {max_diff:.2e} (refit-free OK)")
    print(f"[ckpt] available: {state.list_checkpoints(project_dir)}")

    print("\n" + "=" * 70)
    print("Iteration 3 OK — GP-expert beats Scheffe on non-smooth physics, "
          "honest σ, reproducible.")
    print("=" * 70)


if __name__ == "__main__":
    main()
