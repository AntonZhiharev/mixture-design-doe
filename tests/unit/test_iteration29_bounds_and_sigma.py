# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 29 / Багфиксы честности кампании: границы кандидатов + σ суррогата.

Проверяемый канон:
  * ``MixtureProcessRunner._phase_candidates`` УВАЖАЕТ индивидуальные
    L/U-границы СВОБОДНЫХ mixture-компонентов (раньше — чистый Дирихле ×
    остаток, до 99% кандидатов нарушали U_i);
  * ``SimplexRegion.random_points`` при исчерпании rejection sampling добирает
    точки выпуклыми комбинациями крайних вершин (с предупреждением), а не молча
    забивает пул копиями центроида;
  * ``GPExpert`` защищён от n < p: тренд Шеффе автоматически даунгрейдится
    (quadratic → linear → constant), пока n < p + max(mean_min_dof, p)
    (т.е. n < 2p, ужесточено в iter30) — иначе OLS интерполирует,
    остатки ≈ 0 и GP выдаёт σ=0 («уверенно врёт»).
"""
import warnings

import numpy as np
import pytest

from src.core.schema import ModelSpec, ProjectSchema, VariableBlock
from src.core.simplex import SimplexRegion
from src.apps.mixture_process_runner import MixtureProcessRunner
from src.models.gp_expert import GPExpert


class _Oracle6:
    """6-компонентный оракул с нелинейностью (линейный тренд не интерполирует)."""

    property_names = ["y1"]

    def evaluate(self, Xc) -> np.ndarray:
        Xc = np.atleast_2d(np.asarray(Xc, float))
        y = (3.0 * Xc[:, 0] + 2.0 * Xc[:, 1]
             + 0.8 * np.sin(9.0 * Xc[:, 0] * Xc[:, 1]))
        return y.reshape(-1, 1)


def _runner6(lower, upper, seed=3):
    names = ["A", "B", "C", "D", "E", "F"]
    mix = VariableBlock.mixture(names, lower=lower, upper=upper)
    proc = VariableBlock.process(["T"], lower=[0.0], upper=[1.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    schema = ProjectSchema.mixture_process(mix, proc, model=model)
    return MixtureProcessRunner(schema, _Oracle6(), seed=seed, n_restarts=2)


# ----------------------------------------------------------------------
# _phase_candidates: границы свободных компонентов
# ----------------------------------------------------------------------
def test_phase_candidates_respect_free_upper_bounds():
    lower = [0.0] * 6
    upper = [1.0, 0.15, 0.15, 0.15, 0.15, 0.15]
    r = _runner6(lower, upper)
    X = r.propose_seed(40, seed=11)
    mix = X[:, :6]
    assert np.allclose(mix.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(mix >= np.asarray(lower) - 1e-9)
    assert np.all(mix <= np.asarray(upper) + 1e-9), (
        f"нарушений U_i: {(mix > np.asarray(upper) + 1e-9).sum()}")


def test_phase_candidates_locked_plus_bounded_free():
    # C заперт на 0.2 (lower==upper); свободные A ≤ 0.5, B ≤ 0.7
    names = ["A", "B", "C"]
    mix = VariableBlock.mixture(names, lower=[0.0, 0.0, 0.2],
                                upper=[0.5, 0.7, 0.2])
    proc = VariableBlock.process(["T"], lower=[0.0], upper=[1.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    schema = ProjectSchema.mixture_process(mix, proc, model=model)
    r = MixtureProcessRunner(schema, _Oracle6(), seed=1, n_restarts=2)
    X = r.propose_seed(30, seed=7)
    m = X[:, :3]
    assert np.allclose(m[:, 2], 0.2, atol=1e-12)          # locked держится
    assert np.allclose(m.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(m[:, 0] <= 0.5 + 1e-9)
    assert np.all(m[:, 1] <= 0.7 + 1e-9)


def test_phase_candidates_deterministic_by_seed():
    r = _runner6([0.0] * 6, [1.0, 0.15, 0.15, 0.15, 0.15, 0.15])
    X1 = r.propose_seed(12, seed=42)
    X2 = r.propose_seed(12, seed=42)
    assert np.allclose(X1, X2)


# ----------------------------------------------------------------------
# SimplexRegion.random_points: fallback без центроидной свалки
# ----------------------------------------------------------------------
def test_random_points_fallback_no_centroid_pileup():
    reg = SimplexRegion(lower=[0.3, 0.3, 0.3], upper=[0.35, 0.35, 0.4])
    with pytest.warns(UserWarning, match="rejection sampling"):
        X = reg.random_points(20, seed=0, max_tries=3)
    assert X.shape == (20, 3)
    for x in X:
        assert reg.is_feasible(x, tol=1e-6)
    # старый fallback давал одинаковые копии центроида — теперь точки различны
    uniq = np.unique(np.round(X, 8), axis=0)
    assert len(uniq) > 5


# ----------------------------------------------------------------------
# GPExpert: защита от n < p (даунгрейд тренда, честная σ)
# ----------------------------------------------------------------------
def _sim_data(n, q=6, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.dirichlet(np.ones(q), size=n)
    y = 3.0 * X[:, 0] + 2.0 * X[:, 1] + rng.normal(0.0, noise, size=n)
    return X, y


def test_gpexpert_downgrades_quadratic_to_linear_at_n12_q6():
    X, y = _sim_data(12)
    with pytest.warns(UserWarning, match="не идентифицируем"):
        gp = GPExpert(mean_model="quadratic", seed=0, n_restarts=2).fit(X, y)
    assert gp.mean_model_effective_ == "linear"
    Xt = np.random.default_rng(1).dirichlet(np.ones(6), size=25)
    pred = gp.predict(Xt)
    assert np.all(pred.std > 0.0)
    assert float(pred.std.mean()) > 0.05        # σ не схлопнулась в 0


def test_gpexpert_keeps_quadratic_with_enough_data():
    X, y = _sim_data(45)                        # p_quad=21, 45 ≥ 2·21 (iter30)
    gp = GPExpert(mean_model="quadratic", seed=0, n_restarts=2).fit(X, y)
    assert gp.mean_model_effective_ == "quadratic"


def test_gpexpert_constant_fallback_and_state_roundtrip():
    X, y = _sim_data(4)                         # даже linear (p=6) не лезет
    with pytest.warns(UserWarning, match="не идентифицируем"):
        gp = GPExpert(mean_model="quadratic", seed=0, n_restarts=2).fit(X, y)
    assert gp.mean_model_effective_ == "constant"
    Xt = np.random.default_rng(2).dirichlet(np.ones(6), size=10)
    p1 = gp.predict(Xt)
    assert np.all(p1.std > 0.0)
    # персистентность: constant-тренд переживает to_state/from_state
    gp2 = GPExpert.from_state(gp.to_state())
    p2 = gp2.predict(Xt)
    assert np.allclose(p1.mean, p2.mean, atol=1e-8)
    assert np.allclose(p1.std, p2.std, atol=1e-8)


def test_runner_surrogate_sigma_positive_after_small_seed():
    """Интеграция: 12-точечный seed при q=6 → суррогат не выдаёт σ=0."""
    r = _runner6([0.0] * 6, [1.0] * 6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r.seed_initial(n=12, seed=5)
    gp = r.surrogates["y1"]
    assert gp.mean_model_effective_ != "quadratic"   # даунгрейд сработал
    cand = r.propose_seed(20, seed=99)
    pred = gp.predict(cand)
    assert np.all(pred.std > 0.0)