"""
design/d_optimal.py — M2: D-optimal screening design.

BATCH coordinate-exchange (Fedorov-style point exchange) over a candidate pool,
with MULTIPLE RANDOM RESTARTS (REBUILD_SPEC rules #2, #9).

The MODEL is an EXPLICIT INPUT (rule #1) — the same generator produces designs
for 'linear' / 'quadratic' / 'cubic' Scheffe models.

R reference: ``AlgDesign::optFederov(criterion="D")`` — compared by D-efficiency
within ~1-2 % (the algorithm is stochastic; not point-for-point).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np

from ..core.linalg import scheffe_matrix, slogdet, d_efficiency
from ..core.simplex import SimplexRegion


@dataclass
class DesignResult:
    """Result of a D-optimal design search."""
    design: np.ndarray                 # n_runs x q  (component space)
    indices: np.ndarray                # rows chosen from the candidate pool
    logdet: float                      # log det(MᵀM) of the chosen design
    d_efficiency: float                # normalised D-efficiency
    model: Union[str, int] = "quadratic"
    n_restarts: int = 0
    history: List[float] = field(default_factory=list)  # best logdet per restart

    def to_state(self) -> dict:
        return {
            "design": np.asarray(self.design),
            "indices": np.asarray(self.indices),
            "logdet": self.logdet,
            "d_efficiency": self.d_efficiency,
            "model": self.model,
            "n_restarts": self.n_restarts,
        }


# ---------------------------------------------------------------------------
# Candidate-pool construction
# ---------------------------------------------------------------------------

def build_candidate_pool(region: SimplexRegion, n_random: int = 500,
                         seed: Optional[int] = None,
                         include_vertices: bool = True,
                         include_centroid: bool = True) -> np.ndarray:
    """Construct a candidate pool: extreme vertices + centroid + random interior."""
    rng_seed = seed
    parts: List[np.ndarray] = []
    if include_vertices:
        V = region.extreme_vertices()
        if len(V):
            parts.append(V)
    if include_centroid:
        parts.append(region.centroid().reshape(1, -1))
    if n_random > 0:
        parts.append(region.random_points(n_random, seed=rng_seed))
    pool = np.vstack(parts)
    # de-duplicate
    uniq: List[np.ndarray] = []
    for p in pool:
        if not any(np.allclose(p, u, atol=1e-7) for u in uniq):
            uniq.append(p)
    return np.array(uniq)


# ---------------------------------------------------------------------------
# Core: D-optimal via coordinate (point) exchange + restarts
# ---------------------------------------------------------------------------

def d_optimal_design(candidates: np.ndarray, n_runs: int,
                     model: Union[str, int] = "quadratic",
                     n_restarts: int = 10, max_iter: int = 100,
                     ridge: float = 1e-10,
                     seed: Optional[int] = None) -> DesignResult:
    """Find an (approximately) D-optimal subset of ``n_runs`` candidate rows.

    Parameters
    ----------
    candidates : (n_cand, q) array of feasible mixture points.
    n_runs     : number of runs to select.
    model      : EXPLICIT Scheffe model order ('linear'|'quadratic'|'cubic'|int).
    n_restarts : random restarts (multi-start against multimodal objective).
    max_iter   : max coordinate-exchange sweeps per restart.
    ridge      : tiny jitter added to MᵀM for numerical stability of logdet.
    """
    candidates = np.atleast_2d(np.asarray(candidates, dtype=float))
    n_cand, q = candidates.shape
    C = scheffe_matrix(candidates, model)        # (n_cand, p) model matrix
    p = C.shape[1]
    if n_runs < p:
        import warnings
        warnings.warn(
            f"n_runs={n_runs} < p={p}: design cannot estimate the model "
            "(singular). Increase n_runs.", UserWarning, stacklevel=2)
    elif n_runs < p + 5:
        import warnings
        warnings.warn(
            f"n_runs={n_runs} is close to p={p} (only {n_runs - p} residual "
            "d.o.f.): R² may look misleadingly high. Recommend n ≳ p+5..10 "
            "plus replicates.", UserWarning, stacklevel=2)
    if n_cand < n_runs:

        raise ValueError(f"Candidate pool ({n_cand}) smaller than n_runs ({n_runs}).")

    rng = np.random.default_rng(seed)
    eye_p = np.eye(p) * ridge

    def info_logdet(idx: np.ndarray) -> float:
        M = C[idx]
        return slogdet(M.T @ M + eye_p)

    best_idx: Optional[np.ndarray] = None
    best_ld = float("-inf")
    history: List[float] = []

    for _ in range(max(1, n_restarts)):
        # Random start
        idx = rng.choice(n_cand, size=n_runs, replace=False)
        cur_ld = info_logdet(idx)

        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for slot in range(n_runs):
                cur_point = idx[slot]
                # Try swapping this slot with every candidate not already used
                used = set(idx.tolist())
                best_swap_ld = cur_ld
                best_swap_cand = cur_point
                for cand in range(n_cand):
                    if cand in used and cand != cur_point:
                        continue
                    idx[slot] = cand
                    ld = info_logdet(idx)
                    if ld > best_swap_ld + 1e-12:
                        best_swap_ld = ld
                        best_swap_cand = cand
                idx[slot] = best_swap_cand
                if best_swap_cand != cur_point:
                    cur_ld = best_swap_ld
                    improved = True

        history.append(cur_ld)
        if cur_ld > best_ld:
            best_ld = cur_ld
            best_idx = idx.copy()

    design = candidates[best_idx]
    deff = d_efficiency(C[best_idx])
    return DesignResult(design=design, indices=best_idx, logdet=best_ld,
                        d_efficiency=deff, model=model,
                        n_restarts=n_restarts, history=history)


def d_optimal_for_region(region: SimplexRegion, n_runs: int,
                         model: Union[str, int] = "quadratic",
                         n_random: int = 500, n_restarts: int = 10,
                         seed: Optional[int] = None) -> DesignResult:
    """Convenience: build a candidate pool from ``region`` then run D-optimal."""
    pool = build_candidate_pool(region, n_random=n_random, seed=seed)
    return d_optimal_design(pool, n_runs, model=model,
                            n_restarts=n_restarts, seed=seed)
