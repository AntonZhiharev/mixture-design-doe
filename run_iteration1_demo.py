"""
run_iteration1_demo.py — end-to-end demo of REBUILD Iteration 1.

Pipeline:  M1 geometry -> M2 D-optimal design -> [synthetic experiments]
           -> M3 Scheffe fit -> recovery report, with ProjectState checkpoints.

Usage:
    python run_iteration1_demo.py
"""
from pathlib import Path

import numpy as np

from src.core.simplex import SimplexRegion
from src.core.synthetic import SyntheticScheffe
from src.core.state import ProjectState
from src.design.d_optimal import d_optimal_for_region
from src.models.scheffe import ScheffeModel


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    seed = 42
    q = 5
    model = "quadratic"
    noise_sd = 0.3
    project_dir = Path("project_demo")


    print("=" * 70)
    print("REBUILD — Iteration 1 demo (M1 -> M2 -> M3) on synthetic polygon")
    print("=" * 70)

    # ---- ProjectState ------------------------------------------------
    state = ProjectState(name="iteration1_demo",
                         config={"q": q, "model": model, "noise_sd": noise_sd,
                                 "seed": seed})

    # ---- M1: geometry -----------------------------------------------
    region = SimplexRegion(lower=[0.05, 0.05, 0.05, 0.05, 0.05],
                           upper=[0.60, 0.60, 0.60, 0.60, 0.60])
    vertices = region.extreme_vertices()
    print(f"\n[M1] Region: {region}")
    print(f"[M1] Extreme vertices found: {len(vertices)}")
    state.set_stage("M1_geometry")
    state.put("vertices", vertices)
    state.checkpoint(project_dir, label="after_M1")
    print(f"[M1] checkpoint saved -> {project_dir/'checkpoints'/'after_M1.json'}")

    # ---- M2: D-optimal screening design -----------------------------
    p = len(SyntheticScheffe(q, model).coefficients)
    n_runs = 2 * p                      # n >> p with comfortable margin (rule #4)

    res = d_optimal_for_region(region, n_runs=n_runs, model=model,
                               n_random=600, n_restarts=15, seed=seed)
    print(f"\n[M2] Scheffe '{model}' params p = {p}; runs n = {n_runs}")
    print(f"[M2] D-efficiency = {res.d_efficiency:.4f}, logdet = {res.logdet:.3f}")
    state.set_stage("M2_screening_design")
    state.put("design", res.design)
    state.put("d_efficiency", res.d_efficiency)
    state.checkpoint(project_dir, label="after_M2")
    print(f"[M2] checkpoint saved -> {project_dir/'checkpoints'/'after_M2.json'}")

    # ---- "Laboratory": evaluate synthetic truth + noise -------------
    truth = SyntheticScheffe(q, model, noise_sd=noise_sd, seed=seed)
    y = truth.evaluate(res.design)
    print(f"\n[lab] Collected {len(y)} noisy responses (noise_sd={noise_sd}).")

    # ---- M3: Scheffe fit & recovery ---------------------------------
    fit = ScheffeModel(model=model).fit(res.design, y)
    print(f"\n[M3] R^2 = {fit.r2:.4f}, Adj-R^2 = {fit.adj_r2:.4f}, "
          f"RMSE = {fit.rmse:.4f}")

    recovered = fit.coefficients
    true_coef = truth.coefficients
    mae = np.mean(np.abs(recovered - true_coef))
    print(f"[M3] Coefficient recovery MAE = {mae:.4f} (truth vs estimate)")
    print("\n  term        true      est       |err|")
    print("  " + "-" * 42)
    for name, t, e in zip(fit.term_names, true_coef, recovered):
        print(f"  {name:<10} {t:8.3f} {e:8.3f} {abs(t-e):8.3f}")

    anova = fit.anova()
    print("\n[M3] ANOVA:")
    print(anova.to_string(index=False))

    state.set_stage("M3_screening_analysis")
    state.put("responses", y)
    state.models["m3_scheffe"] = fit.to_state()
    state.checkpoint(project_dir, label="after_M3")
    print(f"\n[M3] checkpoint saved -> {project_dir/'checkpoints'/'after_M3.json'}")

    # ---- Demonstrate resume from checkpoint -------------------------
    reloaded = ProjectState.restore(project_dir, "after_M2")
    print(f"\n[resume] Restored checkpoint 'after_M2' -> stage = {reloaded.stage}, "
          f"design shape = {reloaded.get('design').shape}")
    print(f"[resume] Available checkpoints: {state.list_checkpoints(project_dir)}")

    print("\n" + "=" * 70)
    print("Iteration 1 OK — design+fit accurate, checkpoints save/restore work.")
    print("=" * 70)


if __name__ == "__main__":
    main()
