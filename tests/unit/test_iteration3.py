"""
Iteration-3 unit tests: M6 single GP-expert (mean = Scheffe, kernel =
Matern 5/2 ARD).  Validates exact recovery on polynomials, flexibility on a
non-polynomial truth (GP beats plain Scheffe), honest std, and persistence.

Run:  python -m pytest tests/unit/test_iteration3.py -v
"""
import numpy as np
import pytest

from src.core.simplex import SimplexRegion
from src.core.synthetic import SyntheticScheffe
from src.models.scheffe import ScheffeModel
from src.models.gp_expert import GPExpert


def _nonpolynomial_truth(X):
    """Quadratic Scheffe trend + a smooth non-polynomial 'threshold' bump."""
    x = np.atleast_2d(X)
    trend = 10 * x[:, 0] + 6 * x[:, 1] + 3 * x[:, 2] + 8 * x[:, 0] * x[:, 1]
    bump = 2.5 / (1.0 + np.exp(-25.0 * (x[:, 0] - 0.4)))   # logistic threshold in A
    return trend + bump


# ----------------------------------------------------------------------
def test_gp_expert_fits_and_predicts_shapes():
    region = SimplexRegion(q=4)
    X = region.random_points(40, seed=1)
    poly = SyntheticScheffe(q=4, model="quadratic", noise_sd=0.1, seed=2)
    y = poly.evaluate(X)
    gp = GPExpert(mean_model="quadratic", seed=0, n_restarts=4).fit(X, y)
    Xt = region.random_points(7, seed=3)
    pred = gp.predict(Xt, return_std=True)
    assert pred.mean.shape == (7,)
    assert pred.std.shape == (7,)
    assert np.all(pred.std > 0)
    assert gp.lengthscales.shape == (4,)
    assert gp.noise_level >= gp.noise_floor


def test_gp_expert_recovers_polynomial_truth():
    region = SimplexRegion(q=3)
    X = region.random_points(40, seed=5)
    poly = SyntheticScheffe(q=3, model="quadratic", noise_sd=0.0, seed=6)
    y = poly.true(X)
    gp = GPExpert(mean_model="quadratic", seed=0, n_restarts=5).fit(X, y)
    Xt = region.random_points(15, seed=7)
    pred = gp.predict(Xt)
    # mean = Scheffe already exact; GP residual ~ 0
    assert np.allclose(pred.mean, poly.true(Xt), atol=1e-3)


def test_gp_beats_scheffe_on_nonpolynomial_truth():
    region = SimplexRegion(q=3)
    Xtr = region.random_points(60, seed=10)
    rng = np.random.default_rng(11)
    ytr = _nonpolynomial_truth(Xtr) + 0.05 * rng.standard_normal(len(Xtr))

    scheffe = ScheffeModel(model="quadratic").fit(Xtr, ytr)
    gp = GPExpert(mean_model="quadratic", seed=0, n_restarts=8).fit(Xtr, ytr)

    Xte = region.random_points(50, seed=12)
    truth = _nonpolynomial_truth(Xte)
    rmse_scheffe = np.sqrt(np.mean((scheffe.predict(Xte) - truth) ** 2))
    rmse_gp = np.sqrt(np.mean((gp.predict(Xte).mean - truth) ** 2))
    # GP captures the non-polynomial bump in its residual kernel
    assert rmse_gp < rmse_scheffe


def test_gp_expert_persistence_roundtrip():
    region = SimplexRegion(q=3)
    X = region.random_points(35, seed=20)
    poly = SyntheticScheffe(q=3, model="quadratic", noise_sd=0.1, seed=21)
    y = poly.evaluate(X)
    gp = GPExpert(mean_model="quadratic", seed=0, n_restarts=5).fit(X, y)

    state = gp.to_state()
    gp2 = GPExpert.from_state(state)

    Xt = region.random_points(10, seed=22)
    p1 = gp.predict(Xt)
    p2 = gp2.predict(Xt)
    assert np.allclose(p1.mean, p2.mean, atol=1e-6)
    assert np.allclose(p1.std, p2.std, atol=1e-6)


def test_rbf_option_runs():
    region = SimplexRegion(q=3)
    X = region.random_points(30, seed=30)
    poly = SyntheticScheffe(q=3, model="quadratic", noise_sd=0.1, seed=31)
    y = poly.evaluate(X)
    gp = GPExpert(kernel="rbf", seed=0, n_restarts=3).fit(X, y)
    pred = gp.predict(region.random_points(5, seed=32))
    assert pred.mean.shape == (5,)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
