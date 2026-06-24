"""
run_iteration5_demo.py — REBUILD Iteration 5 demo.

M7 active learning: the M6<->M7 cycle in two phases.
  Phase A (refinement): acquisition = max predictive std -> shrink uncertainty.
  Phase B (recipe search): acquisition = Expected Improvement -> find the
  recipe that minimises the property, argmax taken over the constrained
  simplex via a candidate set (grab #10).

Oracle = a smooth PVC-like property with a single optimum; we start from a tiny
design and let active learning drive both uncertainty and the incumbent down.

Usage:
    python run_iteration5_demo.py
"""
from pathlib import Path

import numpy as np

from src.core.simplex import SimplexRegion
from src.core.state import ProjectState
from src.models.moe import MixtureOfExperts
from src.design.active_learning import active_learning_loop


def oracle(X):
    """Smooth property, minimised near recipe (0.5, 0.3, 0.2)."""
    x = np.atleast_2d(X)
    c = np.array([0.5, 0.3, 0.2])
    return np.sum((x - c) ** 2, axis=1) + 0.15 * x[:, 0] * x[:, 1]


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    seed, q = 42, 3
    project_dir = Path("project_demo")
    region = SimplexRegion(q=q)

    print("=" * 70)
    print("REBUILD — Iteration 5 demo (M7 active learning, M6<->M7 cycle)")
    print("=" * 70)

    # ---- tiny starting design -------------------------------------
    X0 = region.random_points(12, seed=seed)
    y0 = oracle(X0)
    grid = region.random_points(500, seed=seed + 99)
    m0 = MixtureOfExperts(seed=seed, n_restarts=6).fit(X0, y0)
    std0 = m0.predict(grid).std.max()
    print(f"\n[start] n={len(y0)}, best y={y0.min():.4f}, "
          f"max model std over region={std0:.4f}")

    # ---- Phase A: refinement (max predictive std) -----------------
    resA = active_learning_loop(region, oracle, X0, y0, n_iter=8,
                                acquisition="max_std", batch=2,
                                n_candidates=400, seed=seed,
                                model_kwargs={"n_restarts": 6})
    stdA = resA.model.predict(grid).std.max()
    print(f"\n[M7-A refine | max_std] added {len(resA.y)-len(y0)} pts -> "
          f"n={len(resA.y)}")
    print(f"        max model std over region: {std0:.4f} -> {stdA:.4f} "
          f"({100*(std0-stdA)/std0:.0f}% lower)")
    for h in resA.history:
        print(f"        iter {h['iter']}: n={h['n']:3d}  K={h['K']}  "
              f"max_acq(std)={h['max_acq']:.4f}")

    # ---- Phase B: recipe search (Expected Improvement) ------------
    resB = active_learning_loop(region, oracle, resA.X, resA.y, n_iter=12,
                                acquisition="ei", batch=1, n_candidates=600,
                                maximize=False, acq_tol=1e-4, seed=seed + 1,
                                model_kwargs={"n_restarts": 6})
    x_best, y_best = resB.best(maximize=False)
    print(f"\n[M7-B search | EI] added {len(resB.y)-len(resA.y)} pts -> "
          f"n={len(resB.y)}"
          f"{'  (stopped early: EI < tol)' if resB.stopped_early else ''}")
    print(f"        best y: {y0.min():.4f} -> {y_best:.4f}")
    print(f"        best recipe: {x_best}  (true optimum ~ [0.5 0.3 0.2])")
    for h in resB.history:
        print(f"        iter {h['iter']}: n={h['n']:3d}  best_y={h['best_y']:.4f}  "
              f"max_acq(EI)={h['max_acq']:.4e}")

    # ---- Checkpoint -------------------------------------------------
    state = ProjectState.load(project_dir) if (project_dir / "state.json").exists() \
        else ProjectState(name="iteration5_demo", config={"q": q})
    state.set_stage("M7_active_learning")
    state.models["m7_final_moe"] = resB.model.to_state()
    state.data["X"] = resB.X
    state.data["y"] = resB.y
    state.checkpoint(project_dir, label="after_M7")
    print(f"\n[ckpt] saved -> {project_dir/'checkpoints'/'after_M7.json'}")
    print(f"[ckpt] available: {state.list_checkpoints(project_dir)}")

    print("\n" + "=" * 70)
    print("Iteration 5 OK — active learning cut uncertainty, then EI homed in "
          "on the optimal recipe (feasible on the simplex).")
    print("=" * 70)


if __name__ == "__main__":
    main()
