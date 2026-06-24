"""
Iteration-1 unit tests: core linalg, M1 simplex, synthetic polygon,
M2 D-optimal design, M3 Scheffe fit, and ProjectState persistence.

Run:  python -m pytest tests/unit/test_iteration1.py -v
"""
import numpy as np
import pytest

from src.core.linalg import (
    scheffe_term_indices, scheffe_matrix, n_scheffe_params, d_efficiency,
)
from src.core.simplex import SimplexRegion
from src.core.synthetic import SyntheticScheffe, default_polygon
from src.core.state import ProjectState
from src.models.scheffe import ScheffeModel
from src.design.d_optimal import (
    build_candidate_pool, d_optimal_design, d_optimal_for_region,
)


# ----------------------------------------------------------------------
# core/linalg
# ----------------------------------------------------------------------
def test_scheffe_param_counts():
    # q=5 quadratic -> 5 + C(5,2)=10 => 15
    assert n_scheffe_params(5, "quadratic") == 15
    assert n_scheffe_params(5, "linear") == 5
    assert n_scheffe_params(5, "cubic") == 15 + 10  # +C(5,3)=10 => 25


def test_scheffe_matrix_shape_and_values():
    X = np.array([[0.5, 0.3, 0.2]])
    M = scheffe_matrix(X, "quadratic")
    # q=3 quadratic -> 3 linear + 3 interactions = 6 columns
    assert M.shape == (1, 6)
    # linear cols equal x; first interaction = x0*x1
    assert np.isclose(M[0, 0], 0.5)
    assert np.isclose(M[0, 3], 0.5 * 0.3)


# ----------------------------------------------------------------------
# M1 simplex
# ----------------------------------------------------------------------
def test_unconstrained_vertices_are_pure_components():
    region = SimplexRegion(q=3)
    V = region.extreme_vertices()
    # Pure-component vertices: identity rows (in some order)
    assert V.shape == (3, 3)
    assert np.allclose(np.sort(V.sum(axis=1)), 1.0)
    assert np.allclose(np.sort(V.flatten())[-3:], 1.0)


def test_constrained_region_vertices_feasible():
    region = SimplexRegion(lower=[0.1, 0.1, 0.1], upper=[0.7, 0.7, 0.7])
    V = region.extreme_vertices()
    assert len(V) > 0
    for v in V:
        assert region.is_feasible(v)


def test_pseudocomponent_roundtrip():
    region = SimplexRegion(lower=[0.1, 0.2, 0.0], upper=[0.8, 0.9, 0.7])
    x = np.array([0.3, 0.4, 0.3])
    w = region.to_pseudo(x)
    x_back = region.from_pseudo(w)
    assert np.allclose(x, x_back, atol=1e-12)


def test_ilr_roundtrip():
    region = SimplexRegion(q=4)
    x = np.array([0.4, 0.3, 0.2, 0.1])
    z = region.to_ilr(x)
    assert z.shape == (3,)
    x_back = region.from_ilr(z)
    assert np.allclose(x, x_back, atol=1e-10)


def test_random_points_feasible():
    region = SimplexRegion(lower=[0.1, 0.1, 0.05, 0.0, 0.0],
                           upper=[0.6, 0.6, 0.5, 0.4, 0.4])
    pts = region.random_points(50, seed=1)
    assert pts.shape == (50, 5)
    assert all(region.is_feasible(p) for p in pts)


# ----------------------------------------------------------------------
# Synthetic polygon
# ----------------------------------------------------------------------
def test_synthetic_true_vs_noisy():
    poly = SyntheticScheffe(q=5, model="quadratic", noise_sd=0.0, seed=0)
    X = SimplexRegion(q=5).random_points(20, seed=3)
    y_true = poly.true(X)
    y_eval = poly.evaluate(X)
    assert np.allclose(y_true, y_eval)  # no noise


def test_default_polygon_shapes():
    poly = default_polygon(noise_sd=0.5, seed=42)
    assert poly.q == 5
    assert len(poly.coefficients) == 15


# ----------------------------------------------------------------------
# M3 Scheffe — exact recovery on noiseless data
# ----------------------------------------------------------------------
def test_scheffe_recovers_known_coefficients_noiseless():
    poly = SyntheticScheffe(q=4, model="quadratic", noise_sd=0.0, seed=7)
    region = SimplexRegion(q=4)
    X = region.random_points(60, seed=11)
    y = poly.true(X)
    model = ScheffeModel(model="quadratic").fit(X, y)
    assert np.allclose(model.coefficients, poly.coefficients, atol=1e-6)
    assert model.r2 > 0.999999


def test_scheffe_predict_matches_truth():
    poly = SyntheticScheffe(q=3, model="quadratic", noise_sd=0.0, seed=1)
    region = SimplexRegion(q=3)
    X = region.random_points(40, seed=2)
    y = poly.true(X)
    model = ScheffeModel(model="quadratic").fit(X, y)
    Xt = region.random_points(10, seed=99)
    assert np.allclose(model.predict(Xt), poly.true(Xt), atol=1e-6)


# ----------------------------------------------------------------------
# M2 D-optimal
# ----------------------------------------------------------------------
def test_d_optimal_returns_requested_runs():
    region = SimplexRegion(q=4)
    pool = build_candidate_pool(region, n_random=200, seed=0)
    res = d_optimal_design(pool, n_runs=14, model="quadratic",
                           n_restarts=5, seed=0)
    assert res.design.shape == (14, 4)
    assert np.isfinite(res.logdet)
    assert res.d_efficiency > 0


def test_d_optimal_beats_random_subset():
    region = SimplexRegion(q=4)
    pool = build_candidate_pool(region, n_random=300, seed=0)
    res = d_optimal_design(pool, n_runs=16, model="quadratic",
                           n_restarts=8, seed=0)
    # Random subset baseline
    rng = np.random.default_rng(123)
    from src.core.linalg import scheffe_matrix as smat, slogdet
    rand_idx = rng.choice(len(pool), size=16, replace=False)
    rand_ld = slogdet(smat(pool[rand_idx], "quadratic").T @
                      smat(pool[rand_idx], "quadratic"))
    assert res.logdet >= rand_ld - 1e-6  # D-optimal no worse than random


def test_d_optimal_for_region_feasible():
    region = SimplexRegion(lower=[0.1, 0.1, 0.1], upper=[0.8, 0.8, 0.8])
    res = d_optimal_for_region(region, n_runs=8, model="quadratic",
                               n_random=200, n_restarts=4, seed=0)
    assert all(region.is_feasible(p) for p in res.design)


# ----------------------------------------------------------------------
# ProjectState persistence
# ----------------------------------------------------------------------
def test_project_state_save_load(tmp_path):
    st = ProjectState(name="demo", config={"q": 5, "seed": 42})
    st.set_stage("M2_screening_design")
    st.put("design", np.arange(12).reshape(4, 3).astype(float))
    folder = tmp_path / "proj"
    st.save(folder)
    loaded = ProjectState.load(folder)
    assert loaded.name == "demo"
    assert loaded.stage == "M2_screening_design"
    assert np.allclose(loaded.get("design"), np.arange(12).reshape(4, 3))


def test_project_state_checkpoint_restore(tmp_path):
    st = ProjectState(name="demo")
    st.set_stage("M1_geometry")
    st.put("vertices", np.eye(3))
    folder = tmp_path / "proj"
    st.checkpoint(folder, label="after_M1")
    # advance and overwrite
    st.set_stage("M3_screening_analysis")
    st.put("vertices", np.zeros((3, 3)))
    st.save(folder)
    restored = ProjectState.restore(folder, "after_M1")
    assert restored.stage == "M1_geometry"
    assert np.allclose(restored.get("vertices"), np.eye(3))
    assert "after_M1" in st.list_checkpoints(folder)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
