"""
design/active_learning.py — M7: active learning loop (the M6<->M7 cycle).

Acquisition by phase (REBUILD_SPEC M7):
  * refinement      -> max predictive std  OR  max expert-disagreement
                       (sample where the MoE is least sure / experts argue);
  * recipe search   -> Expected Improvement (EI) or Lower Confidence Bound (LCB).

argmax of the acquisition is taken over a CANDIDATE SET drawn from the
constrained simplex (grab #10: no free-R^q gradient; stay feasible).  Batch
proposals use a greedy min-distance filter for diversity (grab #2: batch, not
greedy-sequential-without-diversity).

Exit criterion (REBUILD_SPEC §2): stop when the max acquisition falls below a
tolerance (uncertainty already low / EI no longer proposes better points).

R reference: ``DiceOptim::EI`` (expected improvement).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np
from scipy.stats import norm

from ..core.simplex import SimplexRegion
from ..models.moe import MixtureOfExperts


# ----------------------------------------------------------------------
# Acquisition functions (vectorised; higher score = more desirable point)
# ----------------------------------------------------------------------
def expected_improvement(mu, sigma, y_best, maximize=False, xi=0.0):
    """EI for improving over the incumbent y_best (default: minimisation)."""
    mu = np.asarray(mu, float); sigma = np.maximum(np.asarray(sigma, float), 1e-12)
    imp = (mu - y_best - xi) if maximize else (y_best - mu - xi)
    z = imp / sigma
    return np.maximum(imp * norm.cdf(z) + sigma * norm.pdf(z), 0.0)


def lower_confidence_bound(mu, sigma, kappa=2.0):
    """LCB = mu - kappa*sigma (for minimisation; pick the smallest)."""
    return np.asarray(mu, float) - kappa * np.asarray(sigma, float)


# ----------------------------------------------------------------------
def acquisition_scores(model, candidates, acquisition="max_std",
                       y_best=None, maximize=False, kappa=2.0):
    """Return a 'higher = better' score per candidate for the chosen rule."""
    pred = model.predict(candidates)
    mean, std = pred.mean, pred.std
    if acquisition == "max_std":
        return std
    if acquisition == "max_disagreement":
        dis = getattr(pred, "disagreement", None)
        if dis is None:
            return std
        return np.sqrt(np.maximum(dis, 0.0))
    if acquisition == "ei":
        if y_best is None:
            raise ValueError("EI requires y_best.")
        return expected_improvement(mean, std, y_best, maximize=maximize)
    if acquisition == "lcb":
        # minimisation: best = smallest LCB -> score = -LCB
        return -lower_confidence_bound(mean, std, kappa=kappa)
    raise ValueError(f"Unknown acquisition '{acquisition}'.")


# ----------------------------------------------------------------------
def propose_batch(model, candidates, acquisition="max_std", batch=1,
                  y_best=None, maximize=False, kappa=2.0, min_dist=0.05):
    """Pick up to `batch` diverse, feasible candidates by acquisition score."""
    candidates = np.atleast_2d(np.asarray(candidates, float))
    scores = acquisition_scores(model, candidates, acquisition,
                                y_best=y_best, maximize=maximize, kappa=kappa)
    order = np.argsort(-scores)
    chosen: List[int] = []
    for idx in order:
        if len(chosen) >= batch:
            break
        if all(np.sqrt(((candidates[idx] - candidates[c]) ** 2).sum()) > min_dist
               for c in chosen):
            chosen.append(int(idx))
    if not chosen:                          # degenerate: take the very best
        chosen = [int(order[0])]
    return candidates[chosen], scores[chosen]


# ----------------------------------------------------------------------
@dataclass
class ActiveLearningResult:
    X: np.ndarray
    y: np.ndarray
    model: MixtureOfExperts
    history: list = field(default_factory=list)   # per-iteration diagnostics
    stopped_early: bool = False

    def best(self, maximize=False):
        i = int(np.argmax(self.y) if maximize else np.argmin(self.y))
        return self.X[i], float(self.y[i])


def active_learning_loop(region: SimplexRegion,
                         oracle: Callable[[np.ndarray], np.ndarray],
                         X0, y0, n_iter: int = 10,
                         acquisition: str = "max_std", batch: int = 1,
                         n_candidates: int = 500, maximize: bool = False,
                         acq_tol: float = 1e-3, sigma_tol: Optional[float] = None,
                         kappa: float = 2.0,
                         model_kwargs: Optional[dict] = None,
                         seed: Optional[int] = None) -> ActiveLearningResult:
    """Run the M6<->M7 cycle: fit MoE -> propose -> query oracle -> repeat."""
    model_kwargs = dict(model_kwargs or {})
    model_kwargs.setdefault("seed", seed)
    X = np.atleast_2d(np.asarray(X0, float)).copy()
    y = np.asarray(y0, float).ravel().copy()

    history, model, stopped = [], None, False
    for it in range(n_iter):
        model = MixtureOfExperts(**model_kwargs).fit(X, y)
        cands = region.random_points(n_candidates,
                                     seed=None if seed is None else seed + it + 1)
        y_best = (float(y.max()) if maximize else float(y.min()))
        newX, sc = propose_batch(model, cands, acquisition=acquisition,
                                  batch=batch, y_best=y_best, maximize=maximize,
                                  kappa=kappa)
        max_acq = float(np.max(sc))
        max_sigma = float(np.max(model.predict(cands).std))
        history.append({"iter": it, "n": len(y), "best_y": y_best,
                        "max_acq": max_acq, "max_sigma": max_sigma,
                        "K": model.n_regimes})

        # Stop on low acquisition; when `sigma_tol` is given, ALSO require the
        # surrogate to be confident (two-condition stop, REBUILD_SPEC §2/§12):
        # don't quit while any candidate still has large predictive sigma.
        acq_low = (acquisition in ("max_std", "max_disagreement", "ei")
                   and max_acq < acq_tol)
        should_stop = acq_low if sigma_tol is None else (acq_low and max_sigma < sigma_tol)
        if should_stop:
            stopped = True
            break

        newY = np.asarray(oracle(newX), float).ravel()
        X = np.vstack([X, newX])
        y = np.concatenate([y, newY])

    if model is None:
        model = MixtureOfExperts(**model_kwargs).fit(X, y)
    return ActiveLearningResult(X=X, y=y, model=model, history=history,
                                stopped_early=stopped)
