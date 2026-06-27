# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 17 / §15.6 ШАГ 1 (ФУНДАМЕНТ) — калибровка σ движка GP+MoE.

§15.6 §5 «Подводный камень 3»: весь VoI держится на ОДНОМ скрытом допущении —
GP честен о своей уверенности (σ). Если σ врёт, VoI врёт следом. Поэтому ДО того,
как на σ строить экономику (price_изд, EI_₽, денежную триаду), проверяем фундамент
на РЕАЛЬНОМ движке (`src/models/gp_expert.py`, чьё ядро включает WhiteKernel и
отдаёт honest predictive std).

Эталон — «premium-угол»: модель учат на ПЛОТНОМ интерьере симплекса×процесса и
оставляют РАЗРЕЖЕННЫМ premium-угол (высокие strength·gloss). На отложенных точках
проверяем три свойства честного σ (A0.1/A0.2 — это про движок, не про степень
Scheffé):

  1. КАЛИБРОВКА: стандартизованные остатки z=(y−μ)/σ имеют RMS ≈ 1 (σ не дутый и
     не заниженный «в среднем»).
  2. ПОКРЫТИЕ: эмпирическая доля |z|<=1.96 близка к номинальным 95% (интервал не
     слишком узкий/широкий).
  3. ЧЕСТНОСТЬ В РАЗРЕЖЕННОМ УГЛУ (premium): предсказанный σ в недосэмплированном
     углу СТРОГО больше, чем в плотном интерьере. Это и есть «причина исследовать»
     (§5): VoI пойдёт в угол осознанно, а не от штрафа.

Без этого теста экономика §15.6 строилась бы на непроверенном фундаменте.
"""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from src.models.gp_expert import GPExpert

# Чистый/малошумный отклик загоняет noise→0 и часть length-scale к границам ⇒
# sklearn сыплет ConvergenceWarning. Это не ошибка фита; глушим только warning.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ----------------------------------------------------------------------
# Синтетическая «истина» над mixture {A,B,C} × process {T,P} (код [0,1]).
# Богаче quadratic-тренда (есть A*B*C и P^2) — остаток ловит GP (§15.6 A0.1).
# ----------------------------------------------------------------------
def _truth(Xc: np.ndarray) -> np.ndarray:
    A, B, C = Xc[:, 0], Xc[:, 1], Xc[:, 2]
    T, P = Xc[:, 3], Xc[:, 4]
    return (6.0 * A + 6.0 * B + 5.0 * C
            + 8.0 * A * B + 9.0 * A * C + 9.0 * B * C
            + 8.0 * A * B * C                      # degree-3: только GP-остаток
            + 5.0 * A * T - 3.0 * T * T            # process gating + чистый квадрат
            + 7.0 * P - 5.0 * P * P)               # вогнутость по P


def _simplex_process(n: int, rng: np.random.Generator) -> np.ndarray:
    """n составных точек [A,B,C,T,P]: Дирихле-симплекс × куб [0,1]^2."""
    mix = rng.dirichlet(np.ones(3), size=n)
    proc = rng.uniform(0.0, 1.0, size=(n, 2))
    return np.hstack([mix, proc])


# Граница «premium-угла» по СУММЕ долей A+B (strength·gloss-ridge): чем выше
# A+B, тем дальше от плотного интерьера и тем глубже разрежённая дыра.
_CORNER_CUT = 0.55          # обучение: всё с A+B > cut вырезается (дыра)
_CORNER_DEEP = 0.65         # тест-точки УГЛА берём глубже cut (точно в дыре)


def _is_premium_corner(Xc: np.ndarray, cut: float = _CORNER_CUT) -> np.ndarray:
    """Premium-угол: суммарно высокая доля A+B (strength·gloss-ridge).

    Этот угол НАМЕРЕННО оставлен разреженным в обучении — тут σ (эпистемическая
    часть) обязан быть выше, чем в плотном интерьере (фундамент VoI/§5).
    """
    return (Xc[:, 0] + Xc[:, 1]) > cut


def _fit_with_sparse_premium(seed: int = 7):
    """Учим GP на ПЛОТНОМ интерьере (premium-угол вырезан) + малый шум.

    Шум держим небольшим (0.05): иначе σ упирается в noise-floor и эпистемический
    подъём в разрежённом углу тонет в шуме измерения (см. §15.6 §5 п.3 — σ должна
    реагировать на разрежение, а не быть слепо однородной).
    """
    rng = np.random.default_rng(seed)
    # широкий пул, затем выкидываем premium-угол → угол разрежен
    pool = _simplex_process(1500, rng)
    train = pool[~_is_premium_corner(pool)]
    train = train[:160]                              # компактная обучающая база
    noise_sd = 0.05
    y = _truth(train) + rng.normal(0.0, noise_sd, size=len(train))
    gp = GPExpert(mean_model="quadratic", kernel="matern52",
                  seed=seed, n_restarts=4).fit(train, y)
    return gp, train, noise_sd, rng


# ----------------------------------------------------------------------
# 1. КАЛИБРОВКА: RMS стандартизованных остатков ≈ 1 на отложенных точках.
# ----------------------------------------------------------------------
def test_sigma_calibration_rms_near_one():
    gp, _train, noise_sd, rng = _fit_with_sparse_premium(seed=11)

    # held-out из ТОЙ ЖЕ области обучения (плотный интерьер, без premium-угла):
    holdout = _simplex_process(600, rng)
    holdout = holdout[~_is_premium_corner(holdout)][:200]
    y_true = _truth(holdout) + rng.normal(0.0, noise_sd, size=len(holdout))

    pred = gp.predict(holdout)
    mu = np.asarray(pred.mean, float).ravel()
    sd = np.asarray(pred.std, float).ravel()
    assert np.all(sd > 0), "σ должен быть строго положителен (honest predictive)."

    z = (y_true - mu) / sd
    rms = float(np.sqrt(np.mean(z ** 2)))
    # Честный σ ⇒ RMS(z) ≈ 1. Допуск широкий: проверяем порядок (нет дутости в
    # разы и нет занижения в разы), а не точную единицу.
    assert 0.5 <= rms <= 2.0, f"σ не калиброван: RMS(z)={rms:.3f} вне [0.5, 2.0]."


# ----------------------------------------------------------------------
# 2. ПОКРЫТИЕ: доля |z|<=1.96 близка к номинальным 95%.
# ----------------------------------------------------------------------
def test_sigma_coverage_close_to_nominal():
    gp, _train, noise_sd, rng = _fit_with_sparse_premium(seed=23)

    holdout = _simplex_process(700, rng)
    holdout = holdout[~_is_premium_corner(holdout)][:250]
    y_true = _truth(holdout) + rng.normal(0.0, noise_sd, size=len(holdout))

    pred = gp.predict(holdout)
    mu = np.asarray(pred.mean, float).ravel()
    sd = np.asarray(pred.std, float).ravel()
    z = np.abs((y_true - mu) / sd)
    coverage = float(np.mean(z <= 1.96))
    # Номинал 0.95; интервал не должен быть существенно узким/широким.
    assert coverage >= 0.80, (
        f"σ-интервал слишком УЗКИЙ (overconfident): покрытие {coverage:.2%} < 80%.")
    assert coverage <= 0.999, (
        f"σ-интервал слишком ШИРОКИЙ (underconfident): покрытие {coverage:.2%}.")


# ----------------------------------------------------------------------
# 3. ЧЕСТНОСТЬ В РАЗРЕЖЕННОМ УГЛУ: σ(premium-угол) > σ(плотный интерьер).
#    Это свойство, на котором держится VoI (§5): неуверенность как причина
#    исследовать. Если бы σ был слепо однородным — VoI не нашёл бы куда смотреть.
# ----------------------------------------------------------------------
def test_sigma_higher_in_sparse_premium_corner():
    gp, _train, _noise_sd, rng = _fit_with_sparse_premium(seed=31)

    # плотный интерьер (обучали тут) vs разреженный premium-угол (не обучали):
    dense = _simplex_process(2000, rng)
    dense = dense[~_is_premium_corner(dense)][:120]

    # точки ГЛУБОКО в углу (A+B > _CORNER_DEEP, заведомо в дыре):
    corner = []
    while len(corner) < 60:
        cand = _simplex_process(800, rng)
        cand = cand[_is_premium_corner(cand, cut=_CORNER_DEEP)]
        corner.extend(cand.tolist())
    corner = np.asarray(corner[:60], float)

    sd_dense = np.asarray(gp.predict(dense).std, float).ravel()
    sd_corner = np.asarray(gp.predict(corner).std, float).ravel()

    assert sd_corner.mean() > sd_dense.mean(), (
        "σ в разреженном premium-углу должен быть ВЫШЕ, чем в плотном интерьере "
        f"(угол {sd_corner.mean():.4f} <= интерьер {sd_dense.mean():.4f}); "
        "без этого VoI не отличит, куда вести разведку (§15.6 §5).")
    # запас ощутимый: эпистемическая часть σ доминирует над noise-floor в дыре
    assert sd_corner.mean() > 1.5 * sd_dense.mean(), (
        f"запас σ в углу мал: {sd_corner.mean():.4f} vs {sd_dense.mean():.4f} "
        "(<1.5x) — σ почти не реагирует на разрежение, фундамент VoI шаткий.")

