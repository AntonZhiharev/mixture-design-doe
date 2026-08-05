# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 34 / Перепроверка находок внешней архитектурной сессии (04.08.2026).

Эмпирика замерена скриптом _iter34_diag*.py ДО написания порогов
(данные: seed=20260804, n=512/4096; см. числа в docstring'ах классов).

Находка 1 (Σ групп связаны Σx=1): подтверждена. Реализация iter31 —
  последовательное сужение: суммы ВСЕХ групп строго в допустимых
  интервалах [max(ΣL_g, 1−ΣU_rest), min(ΣU_g, 1−ΣL_rest)]; равномерная
  маргиналь ТОЧНА для первой перечисленной группы (KS≈0.02), для
  последующих — условная (KS≈0.38, coverage≈0.80). Это документированное
  свойство меры, а не баг; требовать KS<0.05 от всех групп — ложный
  инвариант (три равномерные маргинали при ΣA+ΣB+ΣC=1 несовместимы).

Находка 2 (vertex-fallback стягивает к центроиду): подтверждена
  (q95(r/r_max)=0.28 при пороге стянутости 0.75) и ЗАКРЫТА в iter35 —
  fallback заменён последовательным сужением с singleton-группами
  (всегда валидная точка за O(q), без rejection и без перечисления
  вершин). Плюс факты: фикстура «узкой области» из сессии (ΣU=0.99<1)
  НЕВАЛИДНА — конструктор честно отвергает её; старый vertex-fallback
  при q≥15 был вычислительно патологичен (extreme_vertices = q·2^(q−1)
  комбинаций — на q=17 процесс убит >5 мин).

Находка 4 (phr-рецепт через внешний wf-бокс нежизнеспособен):
  подтверждена и усилена — acceptance 0/200000 (самая режущая ось
  UV_CSFCP: 4.5% поосных попаданий). Production-ответ уже в репо:
  PhrSpec (iter33), сэмплинг без rejection.

Находка 3 (groups не проброшен в block_geometry, PROCESS iid uniform) —
  факт кода; канонический путь — runner (set_mixture_sampling_groups /
  set_phr_spec), покрыт tests/unit/test_iteration31_group_sampling.py и
  test_iteration33_phr_sampler.py. Здесь не дублируется.
"""
import warnings

import numpy as np
import pytest

from src.core.simplex import SimplexRegion

# ----------------------------------------------------------------------
# Хелперы (локальные: в src подобного модуля нет, и для тестов не нужен)
# ----------------------------------------------------------------------
def ks_uniform_scaled(x, lo, hi):
    """Статистика Колмогорова-Смирнова относительно U[lo, hi]."""
    u = np.sort((np.asarray(x, dtype=float) - lo) / (hi - lo))
    n = u.size
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - u), np.max(u - (i - 1) / n)))


def admissible_sum_interval(lower, upper, idx):
    """[max(ΣL_g, 1−ΣU_rest), min(ΣU_g, 1−ΣL_rest)] — спецификация iter31."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    mask = np.zeros(lower.size, dtype=bool)
    mask[list(idx)] = True
    lo = max(lower[mask].sum(), 1.0 - upper[~mask].sum())
    hi = min(upper[mask].sum(), 1.0 - lower[~mask].sum())
    return float(lo), float(hi)


def coverage_q(s, lo, hi, q=(0.02, 0.98)):
    """Доля допустимого интервала между квантилями q2–q98 выборки сумм."""
    return float((np.quantile(s, q[1]) - np.quantile(s, q[0])) / (hi - lo))


# ----------------------------------------------------------------------
# Фикстура G3: 6 компонентов, 3 группы покрывают всё; у групп A и B нижний
# конец допустимого интервала ЗАЖАТ условием 1−ΣU_rest (не собственным ΣL).
# ----------------------------------------------------------------------
G3_NAMES = ("A1", "A2", "B1", "B2", "C1", "C2")
G3_LOWER = (0.20, 0.20, 0.05, 0.05, 0.05, 0.05)   # Σ = 0.60
G3_UPPER = (0.30, 0.30, 0.15, 0.15, 0.12, 0.12)   # Σ = 1.14
G3_GROUPS = {"A": (0, 1), "B": (2, 3), "C": (4, 5)}
GOLDEN = {"A": (0.46, 0.60), "B": (0.16, 0.30), "C": (0.10, 0.24)}
SEEDS = [1, 42, 20260804]


def _g3():
    return SimplexRegion(lower=G3_LOWER, upper=G3_UPPER, names=G3_NAMES)


def _groups_list():
    return [list(v) for v in G3_GROUPS.values()]


# ----------------------------------------------------------------------
# A. Golden-значения допустимых интервалов сумм групп
# ----------------------------------------------------------------------
class TestAdmissibleInterval:
    """ΣL=0.60, ΣU=1.14; рукой:
    A: max(.40, 1−.54)=.46 .. min(.60, 1−.20)=.60
    B: max(.10, 1−.84)=.16 .. min(.30, 1−.50)=.30
    C: max(.10, 1−.90)=.10 .. min(.24, 1−.50)=.24
    """

    @pytest.mark.parametrize("g", ["A", "B", "C"])
    def test_matches_hand_computed(self, g):
        lo, hi = admissible_sum_interval(G3_LOWER, G3_UPPER, G3_GROUPS[g])
        assert lo == pytest.approx(GOLDEN[g][0], abs=1e-12)
        assert hi == pytest.approx(GOLDEN[g][1], abs=1e-12)

    def test_clamping_is_active_for_A_and_B(self):
        """При упрощении формулы до [ΣL_g, ΣU_g] нижние концы были бы 0.40/0.10."""
        assert admissible_sum_interval(G3_LOWER, G3_UPPER, (0, 1))[0] > 0.40 + 1e-9
        assert admissible_sum_interval(G3_LOWER, G3_UPPER, (2, 3))[0] > 0.10 + 1e-9

    def test_full_coverage_group_collapses_to_point(self):
        lo, hi = admissible_sum_interval(G3_LOWER, G3_UPPER, range(6))
        assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)


# ----------------------------------------------------------------------
# B. Групповой сэмплер: допустимость + суммы строго в golden-интервалах
# ----------------------------------------------------------------------
class TestGroupSumsInsideGolden:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_rows_feasible_and_sums_inside(self, seed):
        X = _g3().random_points(512, seed=seed, groups=_groups_list())
        assert np.allclose(X.sum(axis=1), 1.0, atol=1e-9)
        assert np.all(X >= np.asarray(G3_LOWER) - 1e-9)
        assert np.all(X <= np.asarray(G3_UPPER) + 1e-9)
        for g, idx in G3_GROUPS.items():
            lo, hi = GOLDEN[g]
            s = X[:, list(idx)].sum(axis=1)
            assert s.min() >= lo - 1e-9, f"{g}: min={s.min():.6f} < {lo}"
            assert s.max() <= hi + 1e-9, f"{g}: max={s.max():.6f} > {hi}"

    def test_first_group_uniform_and_covered(self):
        """Первая перечисленная группа: равномерность точная (замер KS=0.019)."""
        X = _g3().random_points(4096, seed=20260804, groups=_groups_list())
        lo, hi = GOLDEN["A"]
        s = X[:, [0, 1]].sum(axis=1)
        assert ks_uniform_scaled(s, lo, hi) < 0.05
        assert coverage_q(s, lo, hi) > 0.90            # замер 0.957

    @pytest.mark.parametrize("g", ["B", "C"])
    def test_later_groups_conditionally_covered(self, g):
        """Поздние группы: покрытие условное, но края интервала достижимы
        (замер: cov≈0.80, min/max вплотную к концам)."""
        X = _g3().random_points(4096, seed=20260804, groups=_groups_list())
        lo, hi = GOLDEN[g]
        s = X[:, list(G3_GROUPS[g])].sum(axis=1)
        w = hi - lo
        assert coverage_q(s, lo, hi) > 0.70            # замер 0.80-0.81
        assert s.min() < lo + 0.10 * w, f"{g}: нижний конец не достигнут"
        assert s.max() > hi - 0.02 * w, f"{g}: верхний конец не достигнут"

    def test_order_dependence_is_documented_behavior(self):
        """Находка 1: три равномерные маргинали при Σ=1 несовместимы.
        Последовательное сужение делает равномерной ПЕРВУЮ группу; при
        перестановке порядка «равномерная» группа меняется. Замер:
        KS(A|ABC)=0.019, KS(A|CBA)=0.380, KS(C|CBA)=0.019."""
        lo_a, hi_a = GOLDEN["A"]
        lo_c, hi_c = GOLDEN["C"]
        X_abc = _g3().random_points(4096, seed=20260804, groups=_groups_list())
        X_cba = _g3().random_points(4096, seed=20260804,
                                    groups=[[4, 5], [2, 3], [0, 1]])
        ks_a_abc = ks_uniform_scaled(X_abc[:, [0, 1]].sum(axis=1), lo_a, hi_a)
        ks_a_cba = ks_uniform_scaled(X_cba[:, [0, 1]].sum(axis=1), lo_a, hi_a)
        ks_c_cba = ks_uniform_scaled(X_cba[:, [4, 5]].sum(axis=1), lo_c, hi_c)
        assert ks_a_abc < 0.05 and ks_c_cba < 0.05
        assert ks_a_cba > 0.20      # A перестала быть равномерной — ожидаемо

    def test_ungrouped_is_measurably_worse(self):
        """Регрессия на исходный баг «модель не видит разницы»:
        без groups покрытие сумм проседает (замер: 0.27-0.36 против 0.80+)."""
        Xg = _g3().random_points(512, seed=20260804, groups=_groups_list())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            Xp = _g3().random_points(512, seed=20260804, groups=None)
        gains = []
        for g, idx in G3_GROUPS.items():
            lo, hi = GOLDEN[g]
            cg = coverage_q(Xg[:, list(idx)].sum(axis=1), lo, hi)
            cu = coverage_q(Xp[:, list(idx)].sum(axis=1), lo, hi)
            gains.append(cg - cu)
        assert max(gains) > 0.15, f"groups= не даёт выигрыша: {gains}"

    def test_single_group_marginal_is_uniform(self):
        """Одна группа — равномерность суммы достижима точно (замер KS=0.0098)."""
        X = _g3().random_points(4096, seed=11, groups=[[0, 1]])
        lo, hi = GOLDEN["A"]
        assert ks_uniform_scaled(X[:, [0, 1]].sum(axis=1), lo, hi) < 0.05

    def test_single_component_group_is_legal(self):
        X = _g3().random_points(256, seed=1, groups=[[2]])
        lo, hi = admissible_sum_interval(G3_LOWER, G3_UPPER, (2,))
        assert (lo, hi) == (pytest.approx(0.05), pytest.approx(0.15))
        assert X[:, 2].min() >= lo - 1e-9 and X[:, 2].max() <= hi + 1e-9
        assert coverage_q(X[:, 2], lo, hi) > 0.85


# ----------------------------------------------------------------------
# C. Fallback узкой области (находка 2): iter35 — narrowing вместо вершин
# ----------------------------------------------------------------------
class TestVertexFallback:
    # Валидная, но узкая область: ΣL=0.89 ≤ 1 ≤ ΣU=1.03 (замер acceptance 0.12%)
    NARROW_LO = (0.240, 0.240, 0.120, 0.120, 0.085, 0.085)
    NARROW_HI = (0.270, 0.270, 0.140, 0.140, 0.105, 0.105)

    def test_infeasible_narrow_fixture_rejected_by_constructor(self):
        """Фикстура «узкой области» из внешней сессии (ΣU=0.99<1) невалидна —
        конструктор обязан её отвергнуть, а не сэмплить."""
        with pytest.raises(ValueError, match="upper"):
            SimplexRegion(lower=self.NARROW_LO,
                          upper=(0.260, 0.260, 0.135, 0.135, 0.100, 0.100))

    def test_narrow_bounds_warn(self):
        reg = SimplexRegion(lower=self.NARROW_LO, upper=self.NARROW_HI)
        with pytest.warns(UserWarning,
                          match="перегенерирован последовательным сужением"):
            reg.random_points(64, seed=1, max_tries=50)

    def test_fallback_points_not_centroid_collapsed(self):
        """iter35/36: narrowing-fallback НЕ стягивает точки к центроиду
        (старый vertex-fallback давал q95(r/r_max)=0.28); все точки
        допустимы, различны, края области достижимы. iter36 —
        всё-или-ничего: при недоборе rejection ВЕСЬ план перегенерируется
        narrowing'ом (однородная мера, без смеси распределений)."""
        reg = SimplexRegion(lower=self.NARROW_LO, upper=self.NARROW_HI)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            X = reg.random_points(512, seed=1, max_tries=50)
        assert X.shape == (512, 6)
        for x in X:
            assert reg.is_feasible(x, tol=1e-6)
        assert len(np.unique(np.round(X, 8), axis=0)) > 500
        c = np.asarray(reg.centroid(), dtype=float)
        V = np.asarray(reg.extreme_vertices(), dtype=float)
        r_max = np.max(np.linalg.norm(V - c, axis=1))
        r = np.linalg.norm(X - c, axis=1) / r_max
        assert np.quantile(r, 0.95) > 0.5, \
            f"narrowing-fallback стянут к центроиду: q95={np.quantile(r, 0.95):.3f}"
        assert reg.last_sampling_info["n_fallback"] == 512
        assert reg.last_sampling_info["n_rejection"] == 0
        assert reg.last_sampling_info["method"] == "narrowing"

    def test_method_is_always_homogeneous(self):
        """iter36 (внешняя сессия 05.08.2026): смешение мер в одном плане
        запрещено — narrowing-точки концентрировались в труднодостижимых
        углах, где информация дороже всего. ``method`` обязан быть одним из
        однородных значений; ``rejection+narrowing`` не существует."""
        reg = SimplexRegion(lower=self.NARROW_LO, upper=self.NARROW_HI)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            reg.random_points(64, seed=1, max_tries=50)
        assert reg.last_sampling_info["method"] in (
            "rejection", "narrowing", "grouped")
        assert reg.last_sampling_info["n_rejection_discarded"] >= 0
        _g3().random_points(16, seed=1)

    def test_no_fallback_on_grouped_path(self):
        """Групповой путь (sequential narrowing) без rejection → без fallback."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _g3().random_points(512, seed=1, groups=_groups_list())
        assert not [x for x in w if issubclass(x.category, UserWarning)]

    def test_max_tries_is_total_budget_not_per_point(self):
        """Документирует семантику: max_tries — СУММАРНЫЙ бюджет попыток на
        вызов. Следствие: большой n на области с умеренной приёмкой (~0.4%
        у G3) тихо уходит в fallback, малый n — нет."""
        with warnings.catch_warnings(record=True) as w_small:
            warnings.simplefilter("always")
            _g3().random_points(16, seed=20260804)
        with warnings.catch_warnings(record=True) as w_big:
            warnings.simplefilter("always")
            _g3().random_points(4096, seed=20260804)
        assert not [x for x in w_small if issubclass(x.category, UserWarning)]
        assert [x for x in w_big if issubclass(x.category, UserWarning)]


# ----------------------------------------------------------------------
# D. Находка 4 (характеризация): реальный phr-рецепт через внешний wf-бокс
# ----------------------------------------------------------------------
RECIPE_PHR = {
    "PVC_67": (30.0, 100.0), "PVC_71": (0.0, 70.0), "DINP": (4.0, 14.0),
    "Chalk_1T": (0.0, 17.5), "Chalk_95T": (1.5, 25.0), "CPE_135A": (3.0, 15.0),
    "PBNK_3355": (0.0, 8.0), "PMPlus_8": (0.08, 0.90), "DL_531": (0.05, 0.72),
    "PF711": (2.1, 5.0), "PF711LB": (0.0, 2.0), "DL_60": (0.12, 0.84),
    "AKLUB_K_435": (0.04, 0.72), "OPE": (0.04, 0.72), "TiO2_BLR895": (0.3, 8.0),
    "SBM_55": (0.07, 0.45), "UV_CSFCP": (0.05, 0.30),
}
TOTAL_MIN, TOTAL_MAX = 121.12, 172.75


class TestRecipeViaSimplexIsInfeasible:
    """Эмпирическое обоснование PhrSpec (iter33): внешняя box-аппроксимация
    phr→wf раздувает область ×(T_max/T_min)≈1.43 по каждой оси, узкие полосы
    (UV: [2.9e-4, 2.5e-3]) обваливают acceptance до нуля.

    Историческая справка: до iter35 вызывать reg.random_points здесь было
    нельзя — vertex-fallback тянул extreme_vertices с q·2^(q−1) ≈ 1.1 млн
    комбинаций при q=17 (зависание на минуты). Теперь fallback — narrowing
    (см. test_random_points_survives_q17). Acceptance по-прежнему меряем
    векторно (характеризация меры, не механизма).
    """

    def test_rejection_acceptance_is_zero(self):
        names = tuple(RECIPE_PHR)
        lo = np.array([v[0] / TOTAL_MAX for v in RECIPE_PHR.values()])
        hi = np.array([min(v[1] / TOTAL_MIN, 1.0) for v in RECIPE_PHR.values()])
        reg = SimplexRegion(lower=lo, upper=hi, names=names)
        rng = np.random.default_rng(1)
        W = rng.dirichlet(np.ones(reg.q), size=200_000)
        X = reg.from_pseudo(W)
        ok = (np.all(X >= lo - 1e-6, axis=1) & np.all(X <= hi + 1e-6, axis=1))
        # замер 04.08.2026: 0/200000; порог с запасом — «практически ноль»
        assert ok.mean() < 1e-4, (
            f"acceptance неожиданно не обвалился ({ok.mean():.2e}) — "
            f"перепроверить оценку TOTAL_MIN/TOTAL_MAX")

    def test_random_points_survives_q17(self):
        """iter35: на реальном q=17 wf-боксе random_points больше НЕ виснет —
        narrowing-fallback даёт валидные точки без перечисления вершин.
        (Сама box-аппроксимация остаётся плохим входом — см. PhrSpec.)"""
        names = tuple(RECIPE_PHR)
        lo = np.array([v[0] / TOTAL_MAX for v in RECIPE_PHR.values()])
        hi = np.array([min(v[1] / TOTAL_MIN, 1.0) for v in RECIPE_PHR.values()])
        reg = SimplexRegion(lower=lo, upper=hi, names=names)
        with pytest.warns(UserWarning, match="последовательным сужением"):
            X = reg.random_points(32, seed=7, max_tries=200)
        assert X.shape == (32, 17)
        assert np.allclose(X.sum(axis=1), 1.0, atol=1e-9)
        assert np.all(X >= lo - 1e-9) and np.all(X <= hi + 1e-9)
        assert reg.last_sampling_info["n_fallback"] == 32
