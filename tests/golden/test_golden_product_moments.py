# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Golden-группа A (REBUILD_SPEC §13.5 / §6): моменты на произведении области.

Сверяем ПРОДАКШН ``block_moments.analytic_moment_matrix`` с НЕЗАВИСИМЫМ эталоном
``verification.reference.ref_product_moment_matrix`` (другой путь: Γ-функция против
factorial) при atol 1e-8 — на трёх областях прогрессии:
  1) mixture-only (Scheffé, симплекс),
  2) process-only (RSM, куб [0,1]^d, с intercept),
  3) mixture-process (Scheffé × RSM + кросс x·z).

Плюс: мост ``analytic ≈ равномерный MC`` (явный numpy dirichlet+uniform — НЕ
продакшн-сэмплер, который неравномерен) и сверка I-критерия ``i_value`` против
``ref_i_criterion`` (та же tr[(XᵀX)⁻¹M]) при 1e-8.
"""
import numpy as np
import pytest

from src.core.schema import VariableBlock, ModelSpec, ProjectSchema
from src.core import block_geometry as bg
from src.design.block_model import build_model_terms, model_matrix
from src.design.block_moments import analytic_moment_matrix, i_value
from src.verification.reference import (
    ref_product_moment_matrix, ref_i_criterion)


def _mixture():
    return ProjectSchema.mixture_only(["A", "B", "C"])                  # p=6


def _process():
    return ProjectSchema.process_only(["z1", "z2"], [0, 0], [1, 1])    # p=6


def _mixture_process():
    return ProjectSchema.mixture_process(
        VariableBlock.mixture(["A", "B", "C"]),
        VariableBlock.process(["T", "t"], [100, 10], [200, 20]),
        model=ModelSpec(cross_level="cross-main", main_components=(0, 1)))  # p=15


def _term_exponents_independent(mt):
    """Независимая (в тесте) распаковка термов в (a,b)-степени для эталона."""
    exps = []
    for t in mt.terms:
        a = [0] * mt.q
        b = [0] * mt.d
        for g in t:
            if g < mt.q:
                a[g] += 1
            else:
                b[g - mt.q] += 1
        exps.append((a, b))
    return exps


@pytest.mark.parametrize("schema_fn,label", [
    (_mixture, "mixture-only"),
    (_process, "process-only"),
    (_mixture_process, "mixture-process"),
])
def test_analytic_moments_match_independent_reference(schema_fn, label):
    s = schema_fn()
    mt = build_model_terms(s)
    M_prod = analytic_moment_matrix(s, terms=mt)
    M_ref = ref_product_moment_matrix(_term_exponents_independent(mt), mt.q, mt.d)
    assert M_prod.shape == (mt.p, mt.p)
    # две независимые реализации одной закрытой формы → совпадение до 1e-8
    assert np.allclose(M_prod, M_ref, rtol=0, atol=1e-8), label
    # M симметрична и PD (моменты невырождены)
    assert np.allclose(M_prod, M_prod.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(M_prod) > 0), label


def _uniform_moment_matrix_mc(s, mt, n=100000, seed=0):
    """Независимый РАВНОМЕРНЫЙ MC по произведению области (numpy dirichlet+uniform).

    ВАЖНО: продакшн ``block_moment_matrix`` использует ``SimplexRegion.random_points``,
    который НЕ равномерен на симплексе (стянут к центроиду) ⇒ его моменты не равны
    интегралу §13.5. Поэтому мост к аналитике строим через ЯВНО равномерный сэмплер.
    """
    rng = np.random.default_rng(seed)
    parts = []
    if mt.q > 0:
        parts.append(rng.dirichlet(np.ones(mt.q), size=n))   # равномерно на симплексе
    if mt.d > 0:
        parts.append(rng.random((n, mt.d)))                  # равномерно на кубе
    P = np.hstack(parts)
    F = model_matrix(s, P, terms=mt)
    return (F.T @ F) / n


@pytest.mark.parametrize("schema_fn", [_mixture, _process, _mixture_process])
def test_analytic_moments_bridge_to_uniform_monte_carlo(schema_fn):
    s = schema_fn()
    mt = build_model_terms(s)
    M_an = analytic_moment_matrix(s, terms=mt)
    M_mc = _uniform_moment_matrix_mc(s, mt, n=100000, seed=0)
    assert np.allclose(M_an, M_mc, atol=5e-3)     # равномерный MC сходится к аналитике


@pytest.mark.parametrize("schema_fn", [_mixture, _process, _mixture_process])
def test_i_value_matches_reference_on_analytic_moments(schema_fn):
    s = schema_fn()
    mt = build_model_terms(s)
    M = analytic_moment_matrix(s, terms=mt)
    pool = bg.build_candidate_pool(s, n_random=200, seed=1)
    X = model_matrix(s, pool, terms=mt)
    iv = i_value(X, M, ridge=0.0)                  # tr[(XᵀX)⁻¹ M]
    iv_ref = ref_i_criterion(X, M)                 # независимый расчёт того же
    assert np.isclose(iv, iv_ref, rtol=1e-8, atol=1e-8)
    assert iv > 0.0
