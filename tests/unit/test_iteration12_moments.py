# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 12 / §13.5: моменты на произведении области + I-критерий.

Проверяемый канон (REBUILD_SPEC §13.5 / §13.9-регресс):
  * W = E_D[f fᵀ] по D = симплекс × [0,1]^d; process-моменты сверяются с
    аналитическими (известны на кубе [0,1]);
  * mixture-only: W совпадает с существующим ``i_optimal.region_moment_matrix``
    (та же выборка + та же модельная матрица) — фундамент M5 не сломан;
  * i_value = trace((XᵀX+ridge·I)⁻¹ W) — та же свёртка, что в M5.
"""
import numpy as np
import pytest

from src.core.schema import VariableBlock, ModelSpec, ProjectSchema
from src.core.block_geometry import random_points
from src.design.block_model import build_model_terms, model_matrix
from src.design.block_moments import block_moment_matrix, i_value
from src.design.i_optimal import region_moment_matrix


def _mp_schema():
    return ProjectSchema.mixture_process(
        VariableBlock.mixture(["A", "B", "C"]),
        VariableBlock.process(["T", "t"], [100, 10], [200, 20]))


def test_process_only_moments_match_analytic():
    """process-only linear: W сверяется с аналитическими моментами на [0,1]^2."""
    s = ProjectSchema.process_only(["z1", "z2"], [0, 0], [10, 10],
                                   model=ModelSpec(process_order="linear"))
    # термы: [1, z1, z2] → W = E[f fᵀ] по равномерному [0,1]^2
    W = block_moment_matrix(s, n_mc=200000, seed=0)
    expected = np.array([
        [1.0, 0.5, 0.5],          # E[1], E[z1], E[z2]
        [0.5, 1 / 3, 0.25],       # E[z1], E[z1^2], E[z1 z2]
        [0.5, 0.25, 1 / 3],       # E[z2], E[z1 z2], E[z2^2]
    ])
    assert W.shape == (3, 3)
    assert np.allclose(W, expected, atol=0.01)


def test_mixture_only_moments_equal_existing_region_matrix():
    """Регресс: mixture-only W == region_moment_matrix при тех же n_mc/seed/model."""
    s = ProjectSchema.mixture_only(["A", "B", "C"])     # quadratic
    region = s.mixture_block().as_simplex_region()
    W_existing = region_moment_matrix(region, "quadratic", n_mc=2000, seed=1)
    W_block = block_moment_matrix(s, n_mc=2000, seed=1)
    assert np.allclose(W_block, W_existing)


def test_i_value_matches_manual_trace():
    s = _mp_schema()
    X = model_matrix(s, random_points(s, 40, seed=3))
    W = block_moment_matrix(s, n_mc=4000, seed=3)
    iv = i_value(X, W, ridge=1e-8)
    p = X.shape[1]
    manual = float(np.trace(np.linalg.inv(X.T @ X + 1e-8 * np.eye(p)) @ W))
    assert np.isclose(iv, manual)
    assert iv > 0.0


def test_i_value_shape_mismatch_raises():
    s = _mp_schema()
    X = model_matrix(s, random_points(s, 20, seed=0))
    with pytest.raises(ValueError):
        i_value(X, np.eye(X.shape[1] + 1))
