"""
Iteration-4 unit tests: M4 GMM regimes (BIC) + full Mixture-of-Experts.

Run:  python -m pytest tests/unit/test_iteration4.py -v
"""
import numpy as np
import pytest

from src.core.simplex import SimplexRegion
from src.models.clustering import GMMRegimes
from src.models.gp_expert import GPExpert
from src.models.moe import MixtureOfExperts


def two_regime_truth(X):
    """Two regimes separated by component A with a jump (bimodal in y)."""
    x = np.atleast_2d(X)
    base = 5 * x[:, 0] + 3 * x[:, 1] + 2 * x[:, 2]
    jump = np.where(x[:, 0] > 0.45, 8.0, 0.0)        # discontinuity
    return base + jump


# ----------------------------------------------------------------------
def test_gmm_finds_two_regimes_on_bimodal():
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(0.0, 0.3, 60),
                        rng.normal(10.0, 0.3, 60)])
    res = GMMRegimes(k_range=range(1, 5), seed=0).fit(y)
    assert res.n_regimes == 2
    assert res.responsibilities.shape == (120, 2)
    assert set(res.bic_table["K"]) >= {1, 2, 3, 4}


def test_gmm_single_regime_on_unimodal():
    rng = np.random.default_rng(1)
    y = rng.normal(5.0, 1.0, 100)
    res = GMMRegimes(k_range=range(1, 5), seed=0).fit(y)
    assert res.n_regimes == 1


def test_moe_prediction_structure():
    region = SimplexRegion(q=3)
    X = region.random_points(80, seed=2)
    rng = np.random.default_rng(3)
    y = two_regime_truth(X) + 0.05 * rng.standard_normal(len(X))
    moe = MixtureOfExperts(seed=0, n_restarts=5).fit(X, y)

    Xt = region.random_points(20, seed=4)
    pred = moe.predict(Xt)
    assert pred.mean.shape == (20,)
    # gating rows sum to 1
    assert np.allclose(pred.gating.sum(axis=1), 1.0, atol=1e-8)
    # variance decomposition: total = within + between, all non-negative
    assert np.all(pred.uncertainty >= 0)
    assert np.all(pred.disagreement >= 0)
    assert np.allclose(pred.std ** 2,
                       pred.uncertainty + pred.disagreement, atol=1e-8)


def test_moe_gating_discriminates_regimes_in_recipe_space():
    """Gating (in recipe space) must route low-A and high-A points to
    different dominant experts — the model-based pre-image of M4 regimes."""
    region = SimplexRegion(q=3)
    X = region.random_points(120, seed=5)
    rng = np.random.default_rng(6)
    y = two_regime_truth(X) + 0.05 * rng.standard_normal(len(X))
    moe = MixtureOfExperts(seed=0, n_restarts=6).fit(X, y)
    assert moe.n_regimes >= 2

    # Build clearly-separated probe points by component A.
    low = region.random_points(200, seed=70)
    low = low[low[:, 0] < 0.25][:15]
    high = region.random_points(200, seed=71)
    high = high[high[:, 0] > 0.65][:15]

    g_low = moe.gating_proba(low).mean(axis=0).argmax()
    g_high = moe.gating_proba(high).mean(axis=0).argmax()
    assert g_low != g_high            # different regimes get different experts


def test_moe_disagreement_grows_at_regime_boundary():
    """Between-expert disagreement (honest uncertainty) is larger near the
    regime boundary than deep inside a single regime."""
    region = SimplexRegion(q=3)
    X = region.random_points(120, seed=8)
    rng = np.random.default_rng(9)
    y = two_regime_truth(X) + 0.05 * rng.standard_normal(len(X))
    moe = MixtureOfExperts(seed=0, n_restarts=6).fit(X, y)
    assert moe.n_regimes >= 2

    pool = region.random_points(400, seed=80)
    interior = pool[pool[:, 0] < 0.20][:25]
    boundary = pool[np.abs(pool[:, 0] - 0.45) < 0.05][:25]
    dis_interior = moe.predict(interior).disagreement.mean()
    dis_boundary = moe.predict(boundary).disagreement.mean()
    assert dis_boundary > dis_interior



def test_moe_persistence_roundtrip():
    region = SimplexRegion(q=3)
    X = region.random_points(80, seed=8)
    rng = np.random.default_rng(9)
    y = two_regime_truth(X) + 0.05 * rng.standard_normal(len(X))
    moe = MixtureOfExperts(seed=0, n_restarts=5).fit(X, y)

    state = moe.to_state()
    moe2 = MixtureOfExperts.from_state(state)

    Xt = region.random_points(15, seed=10)
    p1, p2 = moe.predict(Xt), moe2.predict(Xt)
    assert np.allclose(p1.mean, p2.mean, atol=1e-6)
    assert np.allclose(p1.std, p2.std, atol=1e-6)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
