# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 12 / §13.11 (пункт 4): перевод M5/augmented на АНАЛИТИЧЕСКИЕ моменты.

Проверяемый канон:
  * ``augmented.build_moments`` по умолчанию (``method="analytic"``) == продакшн
    ``block_moments.analytic_moment_matrix`` и ДЕТЕРМИНИРОВАН (не зависит от
    seed/n_mc) — устраняет MC-смещение ``SimplexRegion.random_points`` (§13.11);
  * ``method="mc"`` остаётся доступным (бит-в-бит == ``block_moment_matrix``);
  * блочный добор ``augmented_design`` по умолчанию аналитический ⇒ результат
    инвариантен к ``n_mc`` при фиксированном seed;
  * легаси-путь M5 ``i_optimal.region_moment_matrix(method="analytic")`` для
    mixture-only СОВПАДАЕТ с блочной аналитикой (две независимые реализации одной
    закрытой формы, atol 1e-12) и детерминирован; дефолт остаётся ``"mc"``;
  * неверный ``method`` ⇒ ValueError.
"""
import numpy as np
import pytest

from src.core.schema import VariableBlock, ModelSpec, ProjectSchema, DataPoint, MIXTURE, PROCESS
from src.core import block_geometry as bg
from src.design.block_model import build_model_terms
from src.design.block_moments import analytic_moment_matrix, block_moment_matrix
from src.design.augmented import build_moments, augmented_design
from src.design.i_optimal import region_moment_matrix


def _mixture():
    return ProjectSchema.mixture_only(["A", "B", "C"])


def _mp():
    return ProjectSchema.mixture_process(
        VariableBlock.mixture(["A", "B", "C"]),
        VariableBlock.process(["T", "t"], [100, 10], [200, 20]))


# ----------------------------------------------------------------------
# build_moments: аналитика == продакшн analytic_moment_matrix, детерминизм
# ----------------------------------------------------------------------
@pytest.mark.parametrize("schema_fn", [_mixture, _mp])
def test_build_moments_analytic_matches_block_analytic(schema_fn):
    s = schema_fn()
    mt = build_model_terms(s)
    W = build_moments(s, terms=mt, method="analytic")
    W_ref = analytic_moment_matrix(s, terms=mt)
    assert W.shape == (mt.p, mt.p)
    assert np.allclose(W, W_ref, rtol=0, atol=1e-12)


@pytest.mark.parametrize("schema_fn", [_mixture, _mp])
def test_build_moments_analytic_is_deterministic(schema_fn):
    """Аналитика не зависит от seed/n_mc — это закрытая форма."""
    s = schema_fn()
    mt = build_model_terms(s)
    W_a = build_moments(s, terms=mt, method="analytic", n_mc=1000, seed=0)
    W_b = build_moments(s, terms=mt, method="analytic", n_mc=9999, seed=123)
    assert np.array_equal(W_a, W_b)


@pytest.mark.parametrize("schema_fn", [_mixture, _mp])
def test_build_moments_mc_matches_block_moment_matrix(schema_fn):
    """method='mc' — это ровно прежний MC-путь (бит-в-бит при тех же n_mc/seed)."""
    s = schema_fn()
    mt = build_model_terms(s)
    W = build_moments(s, terms=mt, method="mc", n_mc=2000, seed=7)
    W_ref = block_moment_matrix(s, n_mc=2000, seed=7, terms=mt)
    assert np.array_equal(W, W_ref)


def test_build_moments_invalid_method_raises():
    s = _mixture()
    with pytest.raises(ValueError):
        build_moments(s, method="bogus")


# ----------------------------------------------------------------------
# augmented_design: дефолт аналитический ⇒ инвариант к n_mc
# ----------------------------------------------------------------------
def test_augmented_design_analytic_invariant_to_n_mc():
    """Дефолтный (аналитический) добор не зависит от n_mc при том же seed.

    Пул и жадный отбор детерминированы (seed фикс), W — закрытая форма (n_mc
    игнорируется) ⇒ траектория, индексы и I совпадают для разных n_mc.
    """
    target = _mp()
    pool0 = bg.build_candidate_pool(target, n_random=10, seed=2)
    existing = [DataPoint(schema_version=2,
                          X={MIXTURE: list(r[:3]), PROCESS: list(r[3:])})
                for r in pool0[:6]]
    res_a, _, _ = augmented_design(
        target, existing, n_max=10, margin=6, n_random=100, n_mc=1000, seed=2)
    res_b, _, _ = augmented_design(
        target, existing, n_max=10, margin=6, n_random=100, n_mc=8000, seed=2)
    assert np.array_equal(res_a.indices, res_b.indices)
    assert res_a.i_final == res_b.i_final
    assert res_a.i_history == pytest.approx(res_b.i_history, abs=0.0)


def test_augmented_design_invalid_moment_method_raises():
    target = _mp()
    with pytest.raises(ValueError):
        augmented_design(target, [], moment_method="bogus",
                         n_random=20, n_max=4, margin=2, seed=0)


# ----------------------------------------------------------------------
# легаси-путь M5: region_moment_matrix(method="analytic")
# ----------------------------------------------------------------------
def test_region_analytic_matches_block_analytic_mixture_only():
    """Две независимые реализации одной закрытой формы совпадают (atol 1e-12)."""
    s = _mixture()                                    # quadratic, q=3
    region = s.mixture_block().as_simplex_region()
    W_region = region_moment_matrix(region, "quadratic", method="analytic")
    mt = build_model_terms(s)
    W_block = analytic_moment_matrix(s, terms=mt)
    assert np.allclose(W_region, W_block, rtol=0, atol=1e-12)


def test_region_analytic_is_deterministic():
    s = _mixture()
    region = s.mixture_block().as_simplex_region()
    W_a = region_moment_matrix(region, "quadratic", method="analytic", seed=0)
    W_b = region_moment_matrix(region, "quadratic", method="analytic", seed=999)
    assert np.array_equal(W_a, W_b)


def test_region_default_method_is_mc():
    """Дефолт остаётся MC (регресс не сломан): default == method='mc'."""
    s = _mixture()
    region = s.mixture_block().as_simplex_region()
    W_default = region_moment_matrix(region, "quadratic", n_mc=1500, seed=3)
    W_mc = region_moment_matrix(region, "quadratic", n_mc=1500, seed=3,
                                method="mc")
    assert np.array_equal(W_default, W_mc)


def test_region_invalid_method_raises():
    s = _mixture()
    region = s.mixture_block().as_simplex_region()
    with pytest.raises(ValueError):
        region_moment_matrix(region, "quadratic", method="bogus")


def test_region_analytic_psd_and_symmetric():
    s = _mixture()
    region = s.mixture_block().as_simplex_region()
    W = region_moment_matrix(region, "quadratic", method="analytic")
    assert np.allclose(W, W.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(W) > 0)
