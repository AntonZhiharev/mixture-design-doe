# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 17 / §15.6 ШАГ 3 — VoI = per-property MC-EI с разложением вклада.

Проверяем форму §5 (НЕ EI по агрегату, а MC по свойствам через РЕАЛЬНУЮ
d_overall) на ИЗОЛИРОВАННЫХ фейковых GP с управляемыми μ/σ — так логика VoI
тестируется без шума фита настоящего GP (фундамент σ проверен в ШАГЕ 1):

  1. EI неотрицателен; растёт на кандидате с большим σ перспективной оси.
  2. Разложение: Σ contributions == ei; limiting_axis = ось, чья разведка
     реально двигает агрегат.
  3. ЗАЩИТА §5: дутый σ УЖЕ-добранного свойства (d_i≈1) НЕ даёт justified EI;
     добор veto-лимитирующего свойства — даёт (justified=True, и это его ось).
"""
import numpy as np

from src.optimize.desirability import DesirabilitySpec
from src.optimize.voi import mc_ei_decomposed, VoIResult


# ----------------------------------------------------------------------
# Фейковый GP: фиксированные μ/σ по строкам кандидатов (предсказуемая логика).
# ----------------------------------------------------------------------
class _FakeGP:
    def __init__(self, mean, std):
        self._mean = np.asarray(mean, float)
        self._std = np.asarray(std, float)

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        n = X.shape[0]

        class _P:
            pass
        p = _P()
        # если длины совпали — отдаём построчно; иначе broadcast скаляром
        p.mean = (self._mean if self._mean.shape[0] == n
                  else np.full(n, self._mean.ravel()[0]))
        p.std = (self._std if self._std.shape[0] == n
                 else np.full(n, self._std.ravel()[0]))
        return p


# ----------------------------------------------------------------------
# 1. EI неотрицателен и реагирует на σ перспективной оси.
# ----------------------------------------------------------------------
def test_ei_nonnegative_and_grows_with_uncertainty():
    cand = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]])
    # обе точки одинаковы по μ (чуть ниже планки), но точка 1 более неуверенна:
    goal = {"strength": DesirabilitySpec("max", low=0.0, high=10.0, weight=1.0)}
    sur = {"strength": _FakeGP(mean=[6.0, 6.0], std=[0.2, 2.5])}

    res = mc_ei_decomposed(sur, goal, cand, d_best=0.6, n_mc=2000, seed=1)
    assert isinstance(res, VoIResult)
    assert np.all(res.ei_all >= 0.0), "EI обязан быть неотрицателен."
    # более неуверенная точка (σ=2.5) даёт больше EI → argmax должен выбрать её
    assert res.ei_all[1] > res.ei_all[0], (
        f"EI должен расти с σ перспективной оси: {res.ei_all}")
    assert np.allclose(res.x_next, cand[1]), "x_next = argmax EI (неуверенная точка)."


# ----------------------------------------------------------------------
# 2. Разложение вклада: Σ == ei, limiting_axis = двигающая агрегат ось.
# ----------------------------------------------------------------------
def test_contribution_decomposition_sums_and_points_to_limiting_axis():
    cand = np.array([[0.4, 0.3, 0.3]])
    # две оси: strength уже добрана (μ высоко, d≈1), gloss — лимитирует (μ низко).
    goal = {
        "strength": DesirabilitySpec("max", low=0.0, high=10.0, weight=1.0),
        "gloss":    DesirabilitySpec("max", low=0.0, high=10.0, weight=1.0),
    }
    sur = {
        "strength": _FakeGP(mean=[9.5], std=[0.5]),   # почти добрана
        "gloss":    _FakeGP(mean=[4.0], std=[2.0]),   # лимитирует, неуверенна
    }
    res = mc_ei_decomposed(sur, goal, cand, d_best=0.55, n_mc=4000, seed=2)

    # вклады складываются в ei (с нормировкой) — допуск на MC-шум:
    assert abs(sum(res.contributions.values()) - res.ei) < 1e-9
    # лимитирующая ось — gloss (она держит geo-mean внизу):
    assert res.limiting_axis == "gloss", res.contributions
    assert res.contributions["gloss"] > res.contributions["strength"]


# ----------------------------------------------------------------------
# 3. ЗАЩИТА §5: дутый σ уже-добранного свойства НЕ обосновывает EI.
# ----------------------------------------------------------------------
def test_protection_inflated_sigma_of_satisfied_property_not_justified():
    cand = np.array([[0.4, 0.3, 0.3]])
    goal = {
        "strength": DesirabilitySpec("max", low=0.0, high=10.0, weight=1.0),
        "gloss":    DesirabilitySpec("max", low=0.0, high=10.0, weight=1.0),
    }
    # strength добрана (d≈1), но σ ДУТЫЙ; gloss лимитирует с умеренным σ.
    sur = {
        "strength": _FakeGP(mean=[9.8], std=[5.0]),   # дутый σ уже-добранной оси
        "gloss":    _FakeGP(mean=[3.5], std=[1.5]),   # реальный лимитёр
    }
    res = mc_ei_decomposed(sur, goal, cand, d_best=0.5, n_mc=4000, seed=3)

    # дутый σ strength НЕ должен перетянуть вклад: лимитёр и top-вклад = gloss
    assert res.limiting_axis == "gloss", res.contributions
    assert res.justified, (
        "EI должен быть ОБОСНОВАН разведкой реально лимитирующей оси (gloss), "
        f"а не дутым σ уже-добранной strength: {res.contributions}")
    assert res.contributions["gloss"] > res.contributions["strength"], (
        f"geo-mean/veto не должен пускать дутый σ добранной оси в агрегат: "
        f"{res.contributions}")


def test_justified_false_when_limiting_axis_is_unexplorable():
    """Лимитирующая ось НЕИССЛЕДУЕМА (σ≈0), а крохотный EI идёт от НЕ-лимитирующей
    оси → justified=False: сигнал диагностике, что EI подозрителен (§5 ЗАЩИТА).

    Подбор: gloss держит geo-mean у самого порога d_best (d_gloss мала, σ_gloss≈0
    — добрать нечем), поэтому весь остаточный EI создаёт ТОЛЬКО strength (σ>0). Это
    ровно случай «high EI не обоснован разведкой лимитирующего свойства»."""
    cand = np.array([[0.4, 0.3, 0.3]])
    goal = {
        "strength": DesirabilitySpec("max", low=0.0, high=10.0, weight=1.0),
        "gloss":    DesirabilitySpec("max", low=0.0, high=10.0, weight=1.0),
    }
    sur = {
        "strength": _FakeGP(mean=[9.5], std=[3.0]),    # двигаема, но НЕ лимитёр
        "gloss":    _FakeGP(mean=[1.0], std=[0.01]),   # лимитёр, но неисследуем
    }
    res = mc_ei_decomposed(sur, goal, cand, d_best=0.3, n_mc=8000, seed=5)
    assert res.ei > 0.0, "для осмысленной проверки нужен ненулевой EI."
    # лимитёр (худшая d_i) — gloss, но двигает EI только strength:
    assert res.limiting_axis == "strength"
    assert not res.justified, (
        "EI идёт не от лимитирующей оси (та неисследуема) → justified=False "
        f"({res.contributions})")


# ----------------------------------------------------------------------
# 4. ИНТЕГРАЦИЯ: VoI поверх НАСТОЯЩИХ GPExpert (honest σ из ШАГА 1).
#    x_next должен уходить в разреженный угол лимитирующего свойства.
# ----------------------------------------------------------------------
def test_voi_with_real_gp_steers_to_sparse_limiting_region():
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    from src.models.gp_expert import GPExpert

    rng = np.random.default_rng(17)

    def gloss_truth(X):
        A, B = X[:, 0], X[:, 1]
        return 2.0 + 9.0 * B + 6.0 * A * B        # высокий gloss требует большого B

    def strength_truth(X):
        A, B, C = X[:, 0], X[:, 1], X[:, 2]
        return 6.0 * A + 5.0 * B + 5.0 * C        # легко добирается почти везде

    # обучаем на пуле, ВЫРЕЗАЯ зону высокого B (gloss-лимитёр разрежён):
    pool = rng.dirichlet(np.ones(3), size=1200)
    train = pool[pool[:, 1] < 0.45][:150]
    gp_g = GPExpert(seed=17, n_restarts=4).fit(
        train, gloss_truth(train) + rng.normal(0, 0.05, len(train)))
    gp_s = GPExpert(seed=18, n_restarts=4).fit(
        train, strength_truth(train) + rng.normal(0, 0.05, len(train)))
    sur = {"gloss": gp_g, "strength": gp_s}

    goal = {
        "gloss":    DesirabilitySpec("max", low=2.0, high=11.0, weight=1.5),
        "strength": DesirabilitySpec("max", low=2.0, high=12.0, weight=1.0),
    }
    cand = rng.dirichlet(np.ones(3), size=600)
    res = mc_ei_decomposed(sur, goal, cand, d_best=0.4, n_mc=512, seed=19)

    # структурная валидность сквозного прогона на НАСТОЯЩИХ GP:
    assert res.ei >= 0.0 and np.isfinite(res.ei)
    assert np.isclose(res.x_next.sum(), 1.0, atol=1e-6)        # на симплексе
    assert abs(sum(res.contributions.values()) - res.ei) < 1e-9
    assert set(res.contributions) == {"gloss", "strength"}

    # VoI предпочитает НЕУВЕРЕННУЮ зону: суммарный σ в x_next выше медианы по пулу
    # (разведка §5 — растёт с σ; honest σ из ШАГА 1 реально варьируется по пулу).
    sig_x = float(gp_g.predict(res.x_next.reshape(1, -1)).std[0]
                  + gp_s.predict(res.x_next.reshape(1, -1)).std[0])
    sig_pool = (np.asarray(gp_g.predict(cand).std, float)
                + np.asarray(gp_s.predict(cand).std, float))
    assert sig_x >= np.median(sig_pool), (
        f"VoI должен предпочесть неуверенную зону: σ(x_next)={sig_x:.3f} "
        f"< медиана пула {np.median(sig_pool):.3f}")

