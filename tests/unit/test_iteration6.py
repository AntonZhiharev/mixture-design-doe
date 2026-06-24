"""
Iteration-6 unit tests: M8 desirability optimisation on the simplex.

Run:  python -m pytest tests/unit/test_iteration6.py -v
"""
import numpy as np
import pytest

from src.core.simplex import SimplexRegion
from src.optimize.desirability import (
    DesirabilitySpec, Desirability, DesirabilityResult,
    desirability_value, overall_desirability, optimize_desirability,
)


# ----------------------------------------------------------------------
# Per-property desirability transforms
# ----------------------------------------------------------------------
def test_desirability_max_monotone_and_clamped():
    spec = DesirabilitySpec("max", low=0.0, high=10.0)
    assert desirability_value(-1.0, spec) == pytest.approx(0.0)
    assert desirability_value(0.0, spec) == pytest.approx(0.0)
    assert desirability_value(5.0, spec) == pytest.approx(0.5)
    assert desirability_value(10.0, spec) == pytest.approx(1.0)
    assert desirability_value(99.0, spec) == pytest.approx(1.0)


def test_desirability_min_is_mirror_of_max():
    spec = DesirabilitySpec("min", low=0.0, high=10.0)
    assert desirability_value(-1.0, spec) == pytest.approx(1.0)
    assert desirability_value(2.5, spec) == pytest.approx(0.75)
    assert desirability_value(10.0, spec) == pytest.approx(0.0)
    assert desirability_value(50.0, spec) == pytest.approx(0.0)


def test_desirability_target_peaks_at_target():
    spec = DesirabilitySpec("target", low=0.0, high=10.0, target=4.0)
    assert desirability_value(4.0, spec) == pytest.approx(1.0)
    assert desirability_value(2.0, spec) == pytest.approx(0.5)     # lower ramp
    assert desirability_value(7.0, spec) == pytest.approx(0.5)     # upper ramp
    assert desirability_value(0.0, spec) == pytest.approx(0.0)
    assert desirability_value(10.0, spec) == pytest.approx(0.0)
    assert desirability_value(-1.0, spec) == pytest.approx(0.0)


def test_shape_exponent_makes_spec_stricter():
    lenient = DesirabilitySpec("max", low=0.0, high=1.0, s=0.5)
    strict = DesirabilitySpec("max", low=0.0, high=1.0, s=2.0)
    # at the midpoint, the strict spec gives a much smaller desirability
    assert desirability_value(0.5, strict) < 0.5 < desirability_value(0.5, lenient)


def test_invalid_specs_raise():
    with pytest.raises(ValueError):
        DesirabilitySpec("max", low=1.0, high=0.0)            # high<=low
    with pytest.raises(ValueError):
        DesirabilitySpec("target", low=0.0, high=1.0)         # no target
    with pytest.raises(ValueError):
        DesirabilitySpec("target", low=0.0, high=1.0, target=2.0)  # target outside
    with pytest.raises(ValueError):
        DesirabilitySpec("bogus", low=0.0, high=1.0)          # unknown kind


# ----------------------------------------------------------------------
# Overall aggregation (weighted geometric mean)
# ----------------------------------------------------------------------
def test_overall_is_geometric_mean():
    d = {"a": np.array([0.25]), "b": np.array([0.64])}
    got = overall_desirability(d)
    assert got[0] == pytest.approx(np.sqrt(0.25 * 0.64))


def test_overall_zero_vetoes():
    d = {"a": np.array([0.0, 0.5]), "b": np.array([0.9, 0.8])}
    got = overall_desirability(d)
    assert got[0] == pytest.approx(0.0)          # a zero kills the product
    assert got[1] == pytest.approx(np.sqrt(0.5 * 0.8))


def test_weighting_shifts_overall_toward_heavy_property():
    d = {"a": np.array([0.2]), "b": np.array([0.8])}
    eq = overall_desirability(d, weights={"a": 1.0, "b": 1.0})[0]
    heavy_b = overall_desirability(d, weights={"a": 1.0, "b": 3.0})[0]
    assert heavy_b > eq                          # leaning on the good property


def test_desirability_bundle_individual_and_overall():
    desir = Desirability({
        "strength": DesirabilitySpec("max", low=0.0, high=1.0),
        "cost": DesirabilitySpec("min", low=0.0, high=1.0),
    })
    props = {"strength": np.array([0.8]), "cost": np.array([0.2])}
    ind = desir.individual(props)
    assert ind["strength"][0] == pytest.approx(0.8)
    assert ind["cost"][0] == pytest.approx(0.8)
    assert desir.overall(props)[0] == pytest.approx(0.8)


# ----------------------------------------------------------------------
# Optimisation on the simplex
# ----------------------------------------------------------------------
def _quad_to_target(X, center, scale=1.0):
    """A smooth property that peaks (low value) near `center`."""
    x = np.atleast_2d(X)
    return scale * np.sum((x - np.asarray(center)) ** 2, axis=1)


def test_optimize_returns_feasible_recipe():
    region = SimplexRegion(lower=[0.1, 0.1, 0.1], upper=[0.7, 0.7, 0.7])
    center = np.array([0.5, 0.3, 0.2])
    predictors = {"prop": lambda X: _quad_to_target(X, center)}
    # prop is "smaller-is-better"; range chosen around plausible values
    specs = {"prop": DesirabilitySpec("min", low=0.0, high=0.5)}

    res = optimize_desirability(region, predictors, specs,
                                n_candidates=2000, refine_iters=300, seed=0)
    assert isinstance(res, DesirabilityResult)
    assert region.is_feasible(res.x)
    assert 0.0 <= res.d_overall <= 1.0


def test_optimize_finds_known_optimum():
    region = SimplexRegion(q=3)
    center = np.array([0.5, 0.3, 0.2])          # the true best recipe
    predictors = {"prop": lambda X: _quad_to_target(X, center)}
    specs = {"prop": DesirabilitySpec("min", low=0.0, high=1.0)}

    res = optimize_desirability(region, predictors, specs,
                                n_candidates=3000, refine_iters=600, seed=1)
    # recovered recipe should be close to the true optimum
    assert np.allclose(res.x, center, atol=0.05)
    assert res.d_overall > 0.95


def test_optimize_balances_two_objectives_and_cost():
    region = SimplexRegion(q=3)
    # property A wants component 0 high; cost grows with component 0
    predictors = {
        "A": lambda X: np.atleast_2d(X)[:, 0],            # maximise -> push x0 up
    }
    specs = {"A": DesirabilitySpec("max", low=0.0, high=1.0)}
    cost_fn = lambda X: np.atleast_2d(X)[:, 0]            # cost grows with x0

    # without cost: x0 should be pushed high
    res_free = optimize_desirability(region, predictors, specs,
                                     n_candidates=3000, refine_iters=400, seed=2)
    # with cost (min): the optimiser must compromise -> lower x0
    res_cost = optimize_desirability(region, predictors, specs,
                                     cost_fn=cost_fn, n_candidates=3000,
                                     refine_iters=400, seed=2)
    assert res_free.x[0] > res_cost.x[0]
    assert "cost" in res_cost.d_individual
    assert region.is_feasible(res_cost.x)


def test_refinement_does_not_worsen_global_best():
    region = SimplexRegion(q=3)
    center = np.array([0.4, 0.4, 0.2])
    predictors = {"p": lambda X: _quad_to_target(X, center)}
    specs = {"p": DesirabilitySpec("min", low=0.0, high=1.0)}

    res_global = optimize_desirability(region, predictors, specs,
                                       n_candidates=1500, refine_iters=0, seed=3)
    res_refined = optimize_desirability(region, predictors, specs,
                                        n_candidates=1500, refine_iters=500, seed=3)
    assert res_refined.d_overall >= res_global.d_overall - 1e-9


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
