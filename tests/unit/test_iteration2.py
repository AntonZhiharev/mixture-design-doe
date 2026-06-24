"""
Iteration-2 unit tests: ARD-GP component screening (-> q_eff) and
I-optimal local design (M5).

Run:  python -m pytest tests/unit/test_iteration2.py -v
"""
import numpy as np
import pytest

from src.core.simplex import SimplexRegion
from src.core.synthetic import SyntheticScheffe
from src.core.linalg import scheffe_term_indices
from src.models.screening import ARDScreening, screen_components, significant_terms
from src.models.scheffe import ScheffeModel
from src.design.d_optimal import build_candidate_pool, d_optimal_design
from src.design.i_optimal import (
    region_moment_matrix, i_optimal_design, i_optimal_for_region,
)


def _dominant_A_coefficients(q=5):
    """Scheffe-quadratic coefficients dominated by component A."""
    terms = scheffe_term_indices(q, "quadratic")
    coef = np.zeros(len(terms))
    for t, idx in enumerate(terms):
        if idx == (0,):
            coef[t] = 30.0           # A linear: dominant
        elif len(idx) == 1:
            coef[t] = 1.0            # other linear: small
        # interactions stay ~0
    return coef


# ----------------------------------------------------------------------
# ARD-GP screening
# ----------------------------------------------------------------------
def test_screening_structure_is_valid():
    poly = SyntheticScheffe(q=5, model="quadratic", noise_sd=0.2, seed=3)
    region = SimplexRegion(q=5)
    X = region.random_points(60, seed=5)
    y = poly.evaluate(X)
    res = screen_components(X, y, seed=0, n_restarts=6)
    assert res.lengthscales.shape == (5,)
    assert np.all(res.lengthscales > 0)
    assert np.all((res.importance >= 0) & (res.importance <= 1.0 + 1e-9))
    assert 1 <= res.q_eff <= 5
    assert res.active.sum() == res.q_eff
    assert len(res.table) == 5


def test_screening_flags_dominant_component():
    q = 5
    coef = _dominant_A_coefficients(q)
    poly = SyntheticScheffe(q=q, model="quadratic", coefficients=coef,
                            noise_sd=0.05, seed=1)
    region = SimplexRegion(q=q)
    X = region.random_points(80, seed=2)
    y = poly.evaluate(X)
    res = ARDScreening(seed=0, n_restarts=10).fit(X, y)
    # A (index 0) should be the most important component
    assert int(np.argmax(res.importance)) == 0
    assert res.active[0]


def test_significant_terms_from_ols():
    poly = SyntheticScheffe(q=4, model="quadratic", noise_sd=0.1, seed=7)
    region = SimplexRegion(q=4)
    X = region.random_points(50, seed=8)
    y = poly.evaluate(X)
    model = ScheffeModel(model="quadratic").fit(X, y)
    sig = significant_terms(model, alpha=0.05)
    assert "term" in sig.columns and "p_value" in sig.columns
    assert (sig["p_value"] < 0.05).all()


# ----------------------------------------------------------------------
# I-optimal design
# ----------------------------------------------------------------------
def test_region_moment_matrix_shape_psd():
    region = SimplexRegion(q=4)
    W = region_moment_matrix(region, "quadratic", n_mc=2000, seed=0)
    p = len(scheffe_term_indices(4, "quadratic"))
    assert W.shape == (p, p)
    # symmetric & positive semidefinite
    assert np.allclose(W, W.T, atol=1e-10)
    assert np.all(np.linalg.eigvalsh(W) > -1e-9)


def test_i_optimal_returns_feasible_design():
    region = SimplexRegion(lower=[0.1, 0.1, 0.1], upper=[0.8, 0.8, 0.8])
    res = i_optimal_for_region(region, n_runs=8, model="quadratic",
                               n_random=200, n_mc=2000, n_restarts=4, seed=0)
    assert res.design.shape == (8, 3)
    assert np.isfinite(res.i_score)
    assert all(region.is_feasible(p) for p in res.design)


def test_i_optimal_beats_random_subset_on_i_score():
    region = SimplexRegion(q=4)
    pool = build_candidate_pool(region, n_random=250, seed=0)
    W = region_moment_matrix(region, "quadratic", n_mc=3000, seed=0)
    res = i_optimal_design(pool, n_runs=16, moments=W, model="quadratic",
                           n_restarts=8, seed=0)
    # random baseline I-score
    rng = np.random.default_rng(7)
    from src.core.linalg import scheffe_matrix
    idx = rng.choice(len(pool), size=16, replace=False)
    M = scheffe_matrix(pool[idx], "quadratic")
    rand_i = float(np.trace(np.linalg.inv(M.T @ M + 1e-8 * np.eye(M.shape[1])) @ W))
    assert res.i_score <= rand_i + 1e-9


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
