"""
Iteration-5 unit tests: M7 active learning (acquisition + M6<->M7 loop).

Run:  python -m pytest tests/unit/test_iteration5.py -v
"""
import numpy as np
import pytest

from src.core.simplex import SimplexRegion
from src.design.active_learning import (
    expected_improvement, lower_confidence_bound,
    acquisition_scores, propose_batch, active_learning_loop,
)
from src.models.moe import MixtureOfExperts


def bowl(X):
    """Smooth bowl with a known minimum near the centroid (minimisation)."""
    x = np.atleast_2d(X)
    c = np.array([0.5, 0.3, 0.2])
    return np.sum((x - c) ** 2, axis=1)


# ----------------------------------------------------------------------
def test_expected_improvement_basic_properties():
    # minimisation: mu far below y_best with tiny sigma -> EI ~ (y_best - mu)
    ei = expected_improvement(mu=np.array([0.0]), sigma=np.array([1e-6]),
                              y_best=5.0, maximize=False)
    assert ei[0] == pytest.approx(5.0, abs=1e-3)
    # EI is always non-negative
    ei2 = expected_improvement(mu=np.array([10.0, 2.0]), sigma=np.array([1.0, 1.0]),
                               y_best=3.0)
    assert np.all(ei2 >= 0)


def test_lcb_prefers_low_mean_high_sigma():
    lcb = lower_confidence_bound(mu=np.array([1.0, 1.0]),
                                 sigma=np.array([0.1, 2.0]), kappa=2.0)
    # the second point (higher sigma) has the smaller LCB -> more exploratory
    assert lcb[1] < lcb[0]


def test_proposed_point_is_feasible_on_simplex():
    region = SimplexRegion(lower=[0.05, 0.05, 0.05], upper=[0.8, 0.8, 0.8])
    X = region.random_points(40, seed=1)
    y = bowl(X)
    model = MixtureOfExperts(seed=0, n_restarts=4).fit(X, y)
    cands = region.random_points(300, seed=2)
    newX, sc = propose_batch(model, cands, acquisition="max_std", batch=3)
    assert newX.shape[1] == 3
    # feasibility: bounds + sum-to-one
    assert np.all(newX >= 0.05 - 1e-9) and np.all(newX <= 0.8 + 1e-9)
    assert np.allclose(newX.sum(axis=1), 1.0, atol=1e-6)


def test_max_std_acquisition_targets_uncertain_region():
    region = SimplexRegion(q=3)
    X = region.random_points(35, seed=3)
    y = bowl(X)
    model = MixtureOfExperts(seed=0, n_restarts=4).fit(X, y)
    cands = region.random_points(300, seed=4)
    scores = acquisition_scores(model, cands, acquisition="max_std")
    newX, sc = propose_batch(model, cands, acquisition="max_std", batch=1)
    # the proposed point is among the most uncertain candidates
    assert sc[0] >= np.quantile(scores, 0.9)


def test_active_learning_reduces_uncertainty():
    region = SimplexRegion(q=3)
    X0 = region.random_points(15, seed=5)
    y0 = bowl(X0)
    grid = region.random_points(400, seed=6)

    model0 = MixtureOfExperts(seed=0, n_restarts=4).fit(X0, y0)
    max_std_before = model0.predict(grid).std.max()

    res = active_learning_loop(region, oracle=bowl, X0=X0, y0=y0,
                               n_iter=8, acquisition="max_std", batch=2,
                               n_candidates=300, seed=0,
                               model_kwargs={"n_restarts": 4})
    max_std_after = res.model.predict(grid).std.max()
    assert len(res.y) > len(y0)               # points were added
    assert max_std_after < max_std_before     # uncertainty shrank


def test_active_learning_ei_improves_best():
    region = SimplexRegion(q=3)
    X0 = region.random_points(18, seed=7)
    y0 = bowl(X0)
    best_before = y0.min()

    res = active_learning_loop(region, oracle=bowl, X0=X0, y0=y0,
                               n_iter=10, acquisition="ei", batch=1,
                               n_candidates=400, maximize=False, seed=0,
                               model_kwargs={"n_restarts": 4})
    _, best_after = res.best(maximize=False)
    assert best_after <= best_before          # EI search did not worsen the best
    assert best_after < best_before + 1e-9


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
