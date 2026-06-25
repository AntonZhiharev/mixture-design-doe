# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Golden GP-постериор μ/σ на СОСТАВНЫХ координатах (REBUILD_SPEC §13.11 п.3 / §6).

Закрывает остаток §13.11: ранее регресс GP был только mixture-only (через
бит-в-бит инъекцию и моменты), но НЕ через GP-постериор на составных координатах
[x_0..x_{q-1}, z_0..z_{d-1}] (mixture-доли + process-код [0,1]).

Здесь продакшн ``models.gp_expert.GPExpert`` (sklearn: Const·Matérn5/2-ARD + White,
mean = Scheffé OLS) сверяется при ФИКСИРОВАННЫХ гиперпараметрах с НЕЗАВИСИМЫМ
эталоном ``verification.reference.gp_posterior`` (другой путь: ручной
Cholesky-постериор, numpy-only) на трёх режимах прогрессии: mixture-only,
process-only, mixture-process. atol/rtol 1e-6.

Ключевое: ARD-ядро и σ работают на (q+d)-мерных составных координатах со
смешанной геометрией (симплекс × куб). Гиперпараметры θ берутся из обученного
ядра и подаются в эталон — сверяем постериор ПРИ фикс. θ (как DiceKriging::km).

Тренд (Scheffé-OLS) — ОБЩИЙ носитель с обеих сторон: при mixture-process составной
Scheffé-базис РАНГ-ДЕФИЦИТЕН (Σ_i x_i·z_k = z_k из-за Σx=1), его OLS-проекция
устойчива только через lstsq/SVD (продакшн ``mean_``), поэтому НЕЗАВИСИМО
сверяется именно GP-ПОСТЕРИОР остатков (μ_resid, σ) — ровно цель §13.11 п.3.
"""

import numpy as np
import pytest

from src.models.gp_expert import GPExpert
from src.verification.reference import gp_posterior



# ----------------------------------------------------------------------
# Генерация составных координат по режиму
# ----------------------------------------------------------------------
def _composite_coords(q: int, d: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """[x (Σ=1, симплекс) | z (равномерно [0,1], код)] — составные координаты."""
    parts = []
    if q > 0:
        parts.append(rng.dirichlet(np.ones(q), size=n))   # mixture: Σx=1, x∈[0,1]
    if d > 0:
        parts.append(rng.random((n, d)))                  # process-код ∈ [0,1]
    return np.hstack(parts)


def _surface(Xc: np.ndarray, q: int, d: int) -> np.ndarray:
    """Гладкая поверхность, ЗАВИСЯЩАЯ от состава x и кода z (с кросс x·z).

    Содержит члены ВНЕ квадратичного Scheffé-базиса (z_k^2, тройной кросс) ⇒
    остатки OLS-тренда ненулевые ⇒ GP реально работает (σ варьируется).
    """
    n = Xc.shape[0]
    y = np.zeros(n)
    if q > 0:
        x = Xc[:, :q]
        y += x @ (1.0 + np.arange(q))                     # линейный тренд по x
        if q >= 2:
            y += 2.5 * x[:, 0] * x[:, 1]                   # mixture-кросс
    if d > 0:
        z = Xc[:, q:q + d]
        y += z @ (0.7 + 0.3 * np.arange(d))               # линейный по z
        y += 1.3 * z[:, 0] ** 2                            # ВНЕ Scheffé-квадратики
    if q > 0 and d > 0:
        y += 3.0 * Xc[:, 0] * Xc[:, q]                     # кросс x_0·z_0
        if d >= 2:
            y += 1.1 * Xc[:, 0] * Xc[:, q] * Xc[:, q + 1]  # тройной кросс x_0·z_0·z_1
    return y


# ----------------------------------------------------------------------
# Извлечение фикс. гиперпараметров из обученного продакшн-ядра
# ----------------------------------------------------------------------
def _kernel_hyperparameters(gp: GPExpert):
    """(const, length_scale[ARD], noise) из Const·base + White."""
    kern = gp.gp_.kernel_
    const = float(kern.k1.k1.constant_value)              # ConstantKernel
    length_scale = np.atleast_1d(kern.k1.k2.length_scale).astype(float)  # base ARD
    noise = float(kern.k2.noise_level)                    # WhiteKernel
    return const, length_scale, noise


# ----------------------------------------------------------------------
# Параметризация режимов прогрессии (как в golden-группе A)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("q,d,label", [
    (3, 0, "mixture-only"),
    (0, 2, "process-only"),
    (3, 2, "mixture-process"),
])
def test_gp_posterior_on_composite_matches_independent_reference(q, d, label):
    rng = np.random.default_rng(20260625)
    n_train, n_test = 30, 8

    Xtr = _composite_coords(q, d, n_train, rng)
    Xte = _composite_coords(q, d, n_test, rng)
    y = _surface(Xtr, q, d) + 0.01 * rng.standard_normal(n_train)

    # --- продакшн: обучаем, фиксируем θ через to_state/from_state, предсказываем ---
    gp = GPExpert(mean_model="quadratic", kernel="matern52", seed=0,
                  n_restarts=8).fit(Xtr, y)
    gp_fixed = GPExpert.from_state(gp.to_state())          # optimizer=None ⇒ фикс. θ
    prod = gp_fixed.predict(Xte, return_std=True)

    # --- независимый эталон при ТЕХ ЖЕ θ ---
    const, length_scale, noise = _kernel_hyperparameters(gp_fixed)
    assert length_scale.shape == (q + d,), f"{label}: ARD ℓ на составных координатах"

    # Тренд — общий носитель (продакшн mean_, устойчив к ранг-дефициту базиса);
    # независимо сверяем GP-постериор остатков на ТОЧНЫХ остатках и фикс. θ.
    trend_te = gp_fixed.mean_.predict(Xte)
    resid_train = gp_fixed._resid                          # остатки, на которых учился GP
    post = gp_posterior(gp_fixed._X, resid_train, Xte,
                        const, length_scale, noise, kernel="matern52")

    ref_mean = trend_te + post["mean"]
    ref_std = post["std"]

    np.testing.assert_allclose(prod.mean, ref_mean, atol=1e-6, rtol=1e-6,
                               err_msg=f"{label}: μ на составных координатах")
    np.testing.assert_allclose(prod.std, ref_std, atol=1e-6, rtol=1e-6,
                               err_msg=f"{label}: σ на составных координатах")

    # GP реально работает: остатки ненулевые, σ положительна
    assert np.linalg.norm(resid_train) > 1e-6, f"{label}: тренд не должен быть точным"
    assert np.all(prod.std > 0.0), f"{label}: σ должна быть положительной"


def test_mixture_process_cross_dependence_is_modelled():
    """Поверхность с кросс x·z воспроизводится точнее, чем чисто аддитивный тренд.

    Санити-проверка, что составные координаты реально несут кросс-зависимость
    (а не разваливаются в две независимые задачи): GP на [x|z] восстанавливает
    поверхность с x_0·z_0 на отложенной выборке с малой ошибкой.
    """
    rng = np.random.default_rng(7)
    q, d = 3, 2
    Xtr = _composite_coords(q, d, 60, rng)
    Xte = _composite_coords(q, d, 20, rng)
    ytr = _surface(Xtr, q, d)
    yte = _surface(Xte, q, d)

    gp = GPExpert(mean_model="quadratic", kernel="matern52", seed=0,
                  n_restarts=10).fit(Xtr, ytr)
    pred = gp.predict(Xte, return_std=False)
    rmse = float(np.sqrt(np.mean((pred - yte) ** 2)))
    span = float(yte.max() - yte.min())
    assert rmse < 0.15 * span, f"RMSE={rmse:.4f} велик относительно размаха {span:.4f}"
