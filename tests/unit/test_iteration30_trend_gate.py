# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 30 / Ужесточение порога тренда GPExpert: n ≥ 2p вместо n ≥ p+3.

Контекст (перепроверка после iter29): фиксированный запас mean_min_dof=3
пускал quadratic-тренд при q=6 уже с n=24 (p=21 → 3 dof остатков). OLS почти
интерполировал, σ ядра схлопывалась, эмпирическое покрытие 95%-интервалов
падало до 0.24 и восстанавливалось только к n≈45. Порог поднят до
n ≥ p + max(mean_min_dof, p), т.е. фактически n ≥ 2p.

Проверяемый канон:
  * границы гейта: quadratic при q=6 (p=21) допускается с n=42, не раньше;
    при q=3 (p=6) — с n=12, не раньше;
  * в бывшей «дыре» q=6/n=24 тренд остаётся linear, σ не схлопывается,
    эмпирическое покрытие 95%-интервалов ≥ 0.8 (было 0.24);
  * mean_min_dof остаётся нижним пределом запаса для малых p
    (linear при q=2: p=2, но требуется n ≥ 2+3=5, а не 4).
"""
import warnings

import numpy as np
import pytest

from src.models.gp_expert import GPExpert


def _truth(X: np.ndarray) -> np.ndarray:
    """Нелинейная «правда» — линейный тренд её не интерполирует."""
    return (3.0 * X[:, 0] + 2.0 * X[:, 1]
            + 0.8 * np.sin(9.0 * X[:, 0] * X[:, 1]))


def _sim(n, q=6, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.dirichlet(np.ones(q), size=n)
    y = _truth(X) + rng.normal(0.0, noise, size=n)
    return X, y


def _fit_quiet(X, y, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return GPExpert(mean_model="quadratic", seed=0,
                        n_restarts=2, **kw).fit(X, y)


# ----------------------------------------------------------------------
# Границы гейта n ≥ 2p
# ----------------------------------------------------------------------
def test_gate_boundary_q6():
    # p_quad(6)=21 → quadratic только с n ≥ 42
    X, y = _sim(41, q=6)
    assert _fit_quiet(X, y).mean_model_effective_ == "linear"
    X, y = _sim(42, q=6)
    assert _fit_quiet(X, y).mean_model_effective_ == "quadratic"


def test_gate_boundary_q3():
    # p_quad(3)=6 → quadratic только с n ≥ 12
    X, y = _sim(11, q=3)
    assert _fit_quiet(X, y).mean_model_effective_ == "linear"
    X, y = _sim(12, q=3)
    assert _fit_quiet(X, y).mean_model_effective_ == "quadratic"


def test_mean_min_dof_still_floor_for_small_p():
    # q=2: p_linear=2, запас max(3, 2)=3 → linear требует n ≥ 5
    X, y = _sim(4, q=2)
    assert _fit_quiet(X, y).mean_model_effective_ == "constant"
    X, y = _sim(5, q=2)
    assert _fit_quiet(X, y).mean_model_effective_ == "linear"


# ----------------------------------------------------------------------
# Бывшая «дыра» q=6 / n=24: честная σ и покрытие
# ----------------------------------------------------------------------
def test_no_sigma_collapse_at_q6_n24():
    noise = 0.3
    X, y = _sim(24, q=6, noise=noise, seed=1)
    with pytest.warns(UserWarning, match="не идентифицируем"):
        gp = GPExpert(mean_model="quadratic", seed=0, n_restarts=4).fit(X, y)
    assert gp.mean_model_effective_ == "linear"      # quadratic не пролез
    Xt = np.random.default_rng(2).dirichlet(np.ones(6), size=200)
    pred = gp.predict(Xt)
    # σ не схлопнулась: хотя бы на уровне заметной доли шума измерения
    assert float(np.median(pred.std)) > 0.3 * noise


def test_coverage95_at_q6_n24():
    """Эмпирическое покрытие 95%-интервалов ≥ 0.8 (до правки было ≈0.24)."""
    noise = 0.3
    covs = []
    for seed in (1, 2, 3):
        X, y = _sim(24, q=6, noise=noise, seed=seed)
        gp = _fit_quiet(X, y, kernel="matern52")
        Xt = np.random.default_rng(100 + seed).dirichlet(np.ones(6), size=300)
        pred = gp.predict(Xt)
        hit = np.abs(_truth(Xt) - pred.mean) <= 1.96 * pred.std
        covs.append(float(hit.mean()))
    assert float(np.median(covs)) >= 0.8, f"cov95 по seed'ам: {covs}"