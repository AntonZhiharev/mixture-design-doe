"""
design/i_optimal.py — M5: I-optimal local design.

I-optimality targets accurate PREDICTION (minimum average prediction variance),
which is the right criterion for local refinement of a regime (REBUILD_SPEC M5,
rule #5: "D точит коэффициенты, I точит прогноз").

I-criterion:   phi_I = trace( (MᵀM)⁻¹ · W ),   smaller is better,
where W = E_region[ f(x) f(x)ᵀ ] is the region moment matrix, estimated by
Monte-Carlo over a large feasible sample.

Generator: BATCH coordinate (point) exchange over a candidate pool + multiple
random restarts (rules #2, #9).  The MODEL is an explicit input (rule #1).

R reference: ``AlgDesign::optFederov(criterion="I")``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np

from ..core.linalg import scheffe_matrix
from ..core.simplex import SimplexRegion


@dataclass
class IOptimalResult:
    design: np.ndarray
    indices: np.ndarray
    i_score: float                      # average prediction variance (lower better)
    model: Union[str, int] = "quadratic"
    n_restarts: int = 0
    history: List[float] = field(default_factory=list)

    def to_state(self) -> dict:
        return {
            "design": np.asarray(self.design),
            "indices": np.asarray(self.indices),
            "i_score": float(self.i_score),
            "model": self.model,
            "n_restarts": self.n_restarts,
        }


def region_moment_matrix(region: SimplexRegion, model: Union[str, int],
                         n_mc: int = 5000, seed: Optional[int] = None) -> np.ndarray:
    """Monte-Carlo estimate of W = E[f(x) f(x)ᵀ] over the feasible region."""
    pts = region.random_points(n_mc, seed=seed)
    F = scheffe_matrix(pts, model)
    return (F.T @ F) / F.shape[0]


def i_optimal_design(candidates: np.ndarray, n_runs: int,
                     moments: np.ndarray,
                     model: Union[str, int] = "quadratic",
                     n_restarts: int = 10, max_iter: int = 100,
                     ridge: float = 1e-8,
                     seed: Optional[int] = None) -> IOptimalResult:
    """Find an (approximately) I-optimal subset of ``n_runs`` candidate rows.

    Parameters
    ----------
    candidates : (n_cand, q) feasible mixture points.
    n_runs     : number of runs to select.
    moments    : region moment matrix W (p x p), same column order as the model.
    model      : explicit Scheffe model order.
    """
    candidates = np.atleast_2d(np.asarray(candidates, dtype=float))
    n_cand, q = candidates.shape
    C = scheffe_matrix(candidates, model)
    p = C.shape[1]
    W = np.asarray(moments, dtype=float)
    if W.shape != (p, p):
        raise ValueError(f"moments must be {p}x{p}, got {W.shape}.")
    if n_cand < n_runs:
        raise ValueError(f"Candidate pool ({n_cand}) smaller than n_runs ({n_runs}).")
    if n_runs < p:
        import warnings
        warnings.warn(f"n_runs={n_runs} < p={p}: information matrix singular.",
                      UserWarning, stacklevel=2)

    rng = np.random.default_rng(seed)
    eye_p = np.eye(p) * ridge

    def i_score(idx: np.ndarray) -> float:
        M = C[idx]
        XtX = M.T @ M + eye_p
        try:
            inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            return float("inf")
        return float(np.trace(inv @ W))

    best_idx: Optional[np.ndarray] = None
    best_score = float("inf")
    history: List[float] = []

    for _ in range(max(1, n_restarts)):
        idx = rng.choice(n_cand, size=n_runs, replace=False)
        cur = i_score(idx)
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for slot in range(n_runs):
                cur_point = idx[slot]
                used = set(idx.tolist())
                best_swap = cur
                best_cand = cur_point
                for cand in range(n_cand):
                    if cand in used and cand != cur_point:
                        continue
                    idx[slot] = cand
                    sc = i_score(idx)
                    if sc < best_swap - 1e-12:
                        best_swap = sc
                        best_cand = cand
                idx[slot] = best_cand
                if best_cand != cur_point:
                    cur = best_swap
                    improved = True
        history.append(cur)
        if cur < best_score:
            best_score = cur
            best_idx = idx.copy()

    return IOptimalResult(design=candidates[best_idx], indices=best_idx,
                          i_score=best_score, model=model,
                          n_restarts=n_restarts, history=history)


def i_optimal_for_region(region: SimplexRegion, n_runs: int,
                         model: Union[str, int] = "quadratic",
                         n_random: int = 400, n_mc: int = 5000,
                         n_restarts: int = 10,
                         seed: Optional[int] = None) -> IOptimalResult:
    """Convenience: build candidate pool + region moments, then run I-optimal."""
    from .d_optimal import build_candidate_pool
    pool = build_candidate_pool(region, n_random=n_random, seed=seed)
    W = region_moment_matrix(region, model, n_mc=n_mc, seed=seed)
    return i_optimal_design(pool, n_runs, W, model=model,
                            n_restarts=n_restarts, seed=seed)
