"""
Iteration-9 backlog (low-risk, from FinalCheckList audit):

  1. M2 D-optimal — warn when n_runs ≈ p (R²=1 deceptive), not only n_runs < p.
  2. M8 optimise   — explicit MULTI-START local refinement (`n_starts`).
  3. M7 active learning — two-condition stop: low acquisition AND small sigma.

These touch neither the golden references nor the (not-yet-built) branch layer.

Run:  python -m pytest tests/unit/test_iteration9_backlog.py -v
"""
import warnings

import numpy as np
import pytest

from src.core.simplex import SimplexRegion
from src.design.d_optimal import d_optimal_for_region
from src.design.active_learning import active_learning_loop
from src.optimize.desirability import DesirabilitySpec, optimize_desirability


# ----------------------------------------------------------------------
# 1. n ≈ p warning (Block 2)
# ----------------------------------------------------------------------
def test_d_optimal_warns_when_n_close_to_p():
    # q=3 quadratic -> p = 3 + C(3,2) = 6 ; ask for just p+1 runs
    region = SimplexRegion(q=3)
    with pytest.warns(UserWarning, match="close to p"):
        d_optimal_for_region(region, n_runs=7, model="quadratic",
                             n_random=200, n_restarts=2, seed=0)


def test_d_optimal_no_close_warning_when_ample():
    region = SimplexRegion(q=3)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        d_optimal_for_region(region, n_runs=20, model="quadratic",
                             n_random=300, n_restarts=2, seed=0)
    assert not any("close to p" in str(w.message) for w in rec)


# ----------------------------------------------------------------------
# 2. M8 multi-start refinement (Block 10)
# ----------------------------------------------------------------------
def _quad_to_target(X, center):
    x = np.atleast_2d(X)
    return np.sum((x - np.asarray(center)) ** 2, axis=1)


def test_optimize_accepts_n_starts_and_not_worse():
    region = SimplexRegion(q=3)
    center = np.array([0.5, 0.3, 0.2])
    predictors = {"p": lambda X: _quad_to_target(X, center)}
    specs = {"p": DesirabilitySpec("min", low=0.0, high=1.0)}

    res1 = optimize_desirability(region, predictors, specs, n_starts=1,
                                 n_candidates=1500, refine_iters=300, seed=7)
    res5 = optimize_desirability(region, predictors, specs, n_starts=5,
                                 n_candidates=1500, refine_iters=300, seed=7)
    # multi-start must never be worse than a single start
    assert res5.d_overall >= res1.d_overall - 1e-9
    assert region.is_feasible(res5.x)


def test_optimize_history_records_multiple_starts():
    region = SimplexRegion(q=3)
    center = np.array([0.4, 0.4, 0.2])
    predictors = {"p": lambda X: _quad_to_target(X, center)}
    specs = {"p": DesirabilitySpec("min", low=0.0, high=1.0)}

    res = optimize_desirability(region, predictors, specs, n_starts=3,
                                n_candidates=1000, refine_iters=100, seed=1)
    n_start_events = sum(1 for h in res.history if h.get("stage") == "start")
    assert n_start_events == 3
    assert res.n_starts == 3


# ----------------------------------------------------------------------
# 3. Active-learning two-condition stop (Block 6)
# ----------------------------------------------------------------------
def _make_al_inputs(seed=0, n0=10):
    region = SimplexRegion(q=3)
    X0 = region.random_points(n0, seed=seed)
    oracle = lambda X: np.atleast_2d(X)[:, 0]      # smooth, deterministic
    y0 = oracle(X0)
    return region, oracle, X0, y0


def test_al_stops_on_low_acq_when_sigma_gate_disabled():
    region, oracle, X0, y0 = _make_al_inputs()
    # huge acq_tol -> acquisition always "below" tolerance; no sigma gate
    res = active_learning_loop(region, oracle, X0, y0, n_iter=4,
                               acquisition="max_std", acq_tol=1e9,
                               n_candidates=60, seed=0)
    assert res.stopped_early is True
    assert len(res.history) == 1


def test_al_sigma_gate_blocks_early_stop():
    region, oracle, X0, y0 = _make_al_inputs()
    # acq below tol, but sigma_tol=0 can never be met -> must NOT stop early
    res = active_learning_loop(region, oracle, X0, y0, n_iter=4,
                               acquisition="max_std", acq_tol=1e9,
                               sigma_tol=0.0, n_candidates=60, seed=0)
    assert res.stopped_early is False
    assert len(res.history) == 4
    # diagnostics expose the sigma actually seen
    assert "max_sigma" in res.history[0]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
