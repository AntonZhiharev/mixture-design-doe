# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 35 / Защита кампании PVC: PhrSpec на реальной рецептуре.

Развёртка находок iter34 с геометрии (SimplexRegion) на РАБОЧИЙ путь
кампании (PhrSpec, iter33) — см. docs/CAMPAIGN_SPEC_PVC.md. Канон:

  * реальная 17-компонентная рецептура выражается спекой БЕЗ rejection:
    смола PVC_67+PVC_71 = 100 phr (share-группа), UV — absolute с
    ДИНАМИЧЕСКИМ ПОТОЛКОМ (cap_to=DINP, cap_ratio=0.03): ТРАПЕЦИЯ
    p_UV ∈ [0.05, min(0.30, 0.03·p_DINP)], а не клин ratio_to
    (правка внешней сессии 05.08.2026: растворимость ограничивает УФ
    только сверху; клин вшивал положительную корреляцию с доминирующей
    осью DINP и монотонный prior, которого физика не требует);
  * golden-числа narrowing: немонотонность hi по потреблённой сумме
    (0.40 → 0.70 полка → 0.5333) — свойство формулы
    [max(L, s−ΣU_ост), min(U, s−ΣL_ост)], не снимка кода;
  * round-trip anchors: encode(p)→z→decode(z)≈p, рецепт вне спеки —
    явный ValueError;
  * spec_hash: порядок узлов — часть спеки (влияет на меру сэмплера),
    отпечаток обязан меняться при перестановке;
  * премикс: чистая арифметика навески (SBM_55/UV_CSFCP уходят в
    премикс при δ=0.02 phr);
  * регрессия архитектуры: рецептурная схема идёт через PhrSpec и
    НИКОГДА не попадает в rejection-путь SimplexRegion (иначе кто-то
    однажды вернёт 0/200000); preflight работает в том же пространстве
    (reference строится тем же phr-путём через _phase_candidates).
"""
import warnings

import numpy as np
import pytest

from src.core.schema import ModelSpec, ProjectSchema
from src.design.phr_sampler import PhrSpec, premix_required
from src.apps.mixture_process_runner import MixtureProcessRunner

# ----------------------------------------------------------------------
# Реальная рецептура (phr): смола = 100 (share-группа PVC_67/PVC_71),
# UV_CSFCP — absolute [0.05, 0.30] с потолком 0.03·DINP (трапеция).
# Golden потолка: DINP=4 → hi=0.12; DINP=10 и 14 → hi=0.30 (полка).
# ----------------------------------------------------------------------
UV_LO, UV_HI, UV_CAP = 0.05, 0.30, 0.03

RECIPE_DICTS = [
    {"name": "resin", "mode": "fixed", "value": 100.0},
    {"name": "PVC_67", "mode": "share_of", "of": "resin",
     "lo": 0.30, "hi": 1.00},
    {"name": "PVC_71", "mode": "share_of", "of": "resin",
     "lo": 0.00, "hi": 0.70},
    {"name": "DINP", "mode": "absolute", "lo": 4.0, "hi": 14.0},
    {"name": "Chalk_1T", "mode": "absolute", "lo": 0.0, "hi": 17.5},
    {"name": "Chalk_95T", "mode": "absolute", "lo": 1.5, "hi": 25.0},
    {"name": "CPE_135A", "mode": "absolute", "lo": 3.0, "hi": 15.0},
    {"name": "PBNK_3355", "mode": "absolute", "lo": 0.0, "hi": 8.0},
    {"name": "PMPlus_8", "mode": "absolute", "lo": 0.08, "hi": 0.90},
    {"name": "DL_531", "mode": "absolute", "lo": 0.05, "hi": 0.72},
    {"name": "PF711", "mode": "absolute", "lo": 2.1, "hi": 5.0},
    {"name": "PF711LB", "mode": "absolute", "lo": 0.0, "hi": 2.0},
    {"name": "DL_60", "mode": "absolute", "lo": 0.12, "hi": 0.84},
    {"name": "AKLUB_K_435", "mode": "absolute", "lo": 0.04, "hi": 0.72},
    {"name": "OPE", "mode": "absolute", "lo": 0.04, "hi": 0.72},
    {"name": "TiO2_BLR895", "mode": "absolute", "lo": 0.3, "hi": 8.0},
    {"name": "SBM_55", "mode": "absolute", "lo": 0.07, "hi": 0.45},
    {"name": "UV_CSFCP", "mode": "absolute", "lo": UV_LO, "hi": UV_HI,
     "cap_to": "DINP", "cap_ratio": UV_CAP},
]

COMPONENTS = ["PVC_67", "PVC_71", "DINP", "Chalk_1T", "Chalk_95T",
              "CPE_135A", "PBNK_3355", "PMPlus_8", "DL_531", "PF711",
              "PF711LB", "DL_60", "AKLUB_K_435", "OPE", "TiO2_BLR895",
              "SBM_55", "UV_CSFCP"]

# Anchor: производственный рецепт в phr (внутри спеки; UV=0.2 ≤ 0.03·DINP=0.3)
ANCHOR_PHR = {
    "PVC_67": 70.0, "PVC_71": 30.0, "DINP": 10.0, "Chalk_1T": 10.0,
    "Chalk_95T": 10.0, "CPE_135A": 8.0, "PBNK_3355": 4.0, "PMPlus_8": 0.5,
    "DL_531": 0.3, "PF711": 3.0, "PF711LB": 1.0, "DL_60": 0.5,
    "AKLUB_K_435": 0.3, "OPE": 0.3, "TiO2_BLR895": 3.0,
    "SBM_55": 0.2, "UV_CSFCP": 0.2,
}


def _spec() -> PhrSpec:
    return PhrSpec.from_dicts(RECIPE_DICTS)


def _anchor_vec(spec: PhrSpec) -> np.ndarray:
    return np.array([ANCHOR_PHR[nm] for nm in spec.component_names])


# ----------------------------------------------------------------------
# A. Спека реальной рецептуры: структура и интервалы
# ----------------------------------------------------------------------
class TestRecipeSpec:

    def test_structure_and_intervals(self):
        spec = _spec()
        assert spec.component_names == COMPONENTS
        assert spec.q == 17
        iv = spec.phr_intervals()
        # share-группа смолы восстанавливает исходные phr-диапазоны точно
        assert iv["PVC_67"] == pytest.approx((30.0, 100.0))
        assert iv["PVC_71"] == pytest.approx((0.0, 70.0))
        # absolute+cap: [0.05, min(0.30, 0.03·14=0.42)] = [0.05, 0.30]
        assert iv["UV_CSFCP"] == pytest.approx((0.05, 0.30))

    def test_candidates_sum1_bounds_and_resin_100(self):
        spec = _spec()
        Z = spec.sample_z(2048, seed=1)
        P = spec.decode(Z)
        # смола: PVC_67 + PVC_71 = 100 phr ТОЧНО (Σ долей = 1 конструкцией)
        np.testing.assert_allclose(P[:, 0] + P[:, 1], 100.0, atol=1e-9)
        iv = spec.phr_intervals()
        for j, nm in enumerate(spec.component_names):
            lo, hi = iv[nm]
            assert np.all(P[:, j] >= lo - 1e-9), nm
            assert np.all(P[:, j] <= hi + 1e-9), nm
        X = spec.to_fractions(P)
        np.testing.assert_allclose(X.sum(axis=1), 1.0, atol=1e-9)
        lo_f, hi_f = spec.fraction_bounds()
        assert np.all(X >= lo_f - 1e-9) and np.all(X <= hi_f + 1e-9)

    def test_uv_trapezoid_by_construction(self):
        """Ось, обвалившая wf-бокс (iter34: acceptance 0/200000), в PhrSpec
        держится конструкцией — теперь ТРАПЕЦИЕЙ: UV ≤ 0.03·DINP (сверху,
        растворимость) при свободном низе UV ≥ 0.05 phr; инвариант
        переживает нормировку p → x (доли пропорциональны phr)."""
        spec = _spec()
        X = spec.sample_candidates(2048, seed=2)
        uv = spec.component_names.index("UV_CSFCP")
        dinp = spec.component_names.index("DINP")
        ratio = X[:, uv] / X[:, dinp]
        assert np.all(ratio <= UV_CAP + 1e-9)          # потолок конструкцией
        P = spec.decode(spec.sample_z(2048, seed=2))
        p_uv = P[:, uv]
        assert np.all(p_uv >= UV_LO - 1e-9)
        assert np.all(p_uv <= UV_HI + 1e-9)


# ----------------------------------------------------------------------
# A2. Геометрия UV: трапеция, а не клин (внешняя сессия 05.08.2026)
# ----------------------------------------------------------------------
class TestUvTrapezoidGeometry:
    """Клин ratio_to имел три следствия: (1) нельзя поставить низкий УФ при
    высоком ДИНФ; (2) corr(p_UV, p_DINP) ≈ 0.9 по построению — эффект УФ
    загрязнялся самой сильной осью; (3) вшит монотонный prior «больше
    пластификатора → больше абсорбера». Трапеция снимает все три."""

    def _phr(self, n=4096, seed=20260805):
        spec = _spec()
        P = spec.decode(spec.sample_z(n, seed=seed))
        col = {nm: j for j, nm in enumerate(spec.component_names)}
        return P, col

    def test_golden_cap_dinp4_012(self):
        """DINP=4 → hi_eff = min(0.30, 0.03·4) = 0.12: рецепт UV=0.12
        принимается, UV=0.13 отвергается по потолку."""
        spec = _spec()
        p = _anchor_vec(spec).copy()
        col = {nm: j for j, nm in enumerate(spec.component_names)}
        p[col["DINP"]] = 4.0
        p[col["UV_CSFCP"]] = 0.12
        spec.encode(p)                                 # ровно на потолке — ок
        p[col["UV_CSFCP"]] = 0.13
        with pytest.raises(ValueError, match="UV_CSFCP.*потолок"):
            spec.encode(p)

    @pytest.mark.parametrize("dinp", [10.0, 14.0])
    def test_golden_cap_plateau_030(self, dinp):
        """DINP=10 и 14 → hi_eff = 0.30 (полка собственного hi): UV=0.30
        принимается, UV=0.31 отвергается уже СОБСТВЕННОЙ границей."""
        spec = _spec()
        p = _anchor_vec(spec).copy()
        col = {nm: j for j, nm in enumerate(spec.component_names)}
        p[col["DINP"]] = dinp
        p[col["UV_CSFCP"]] = 0.30
        spec.encode(p)                                 # полка достижима
        p[col["UV_CSFCP"]] = 0.31
        with pytest.raises(ValueError, match="UV_CSFCP"):
            spec.encode(p)

    def test_low_uv_at_high_dinp_reachable(self):
        """Следствие 1 клина снято: низкий УФ при высоком ДИНФ достижим
        (в клине min UV при DINP=14 был 0.0125·14 = 0.175)."""
        spec = _spec()
        p = _anchor_vec(spec).copy()
        col = {nm: j for j, nm in enumerate(spec.component_names)}
        p[col["DINP"]] = 14.0
        p[col["UV_CSFCP"]] = 0.05
        spec.encode(p)                                 # не бросает
        P, c = self._phr()
        high_d = P[:, c["DINP"]] > 12.0
        assert P[high_d, c["UV_CSFCP"]].min() < 0.08   # план реально там бывает

    def test_uv_dinp_correlation_dropped(self):
        """Следствие 2 клина снято: corr(p_UV, p_DINP) упала с ≈0.9 (клин,
        по построению) до геометрического пола трапеции.

        ЧЕСТНАЯ ОГОВОРКА к порогу «< 0.4» внешней сессии: при равномерной
        маргинали DINP (канон кампании — доминирующая ось) и условно-
        равномерном UV в [0.05, min(0.30, 0.03·D)] corr = 0.4235
        аналитически (замер 05.08.2026: 0.422 при n=4096) — это свойство
        ФОРМЫ трапеции, а не сэмплера. Опустить ниже 0.4 можно только
        исказив маргиналь DINP (сэмплинг UV-первым даёт ≈0.38) — осознанно
        НЕ делаем. Порог 0.45 = геометрический пол + запас на шум."""
        P, c = self._phr()
        r = np.corrcoef(P[:, c["UV_CSFCP"]], P[:, c["DINP"]])[0, 1]
        assert abs(r) < 0.45, f"corr(p_UV, p_DINP) = {r:.3f}"

    def test_uv_free_at_low_tio2(self):
        """Гипотеза кампании «мало титана + средний УФ» требует свободы по
        УФ независимо от пластификатора: при TiO2 < 1 phr план обязан
        варьировать УФ минимум в 3 раза (max/min > 3)."""
        P, c = self._phr()
        mask = P[:, c["TiO2_BLR895"]] < 1.0
        assert mask.sum() >= 50, "слишком мало точек с TiO2 < 1 phr"
        uv = P[mask, c["UV_CSFCP"]]
        assert uv.max() / uv.min() > 3.0, \
            f"max/min UV при TiO2<1 = {uv.max() / uv.min():.2f}"


# ----------------------------------------------------------------------
# B. Golden-числа narrowing: немонотонность hi (0.40 → 0.70 → 0.5333)
# ----------------------------------------------------------------------
NARROW_DICTS = [
    {"name": "base", "mode": "fixed", "value": 100.0},
    {"name": "tot", "mode": "absolute", "lo": 2.0, "hi": 5.0},
    {"name": "c1", "mode": "share_of", "of": "tot", "lo": 0.2, "hi": 0.4},
    {"name": "c2", "mode": "share_of", "of": "tot", "lo": 0.1, "hi": 0.7},
    {"name": "c3", "mode": "share_of", "of": "tot", "lo": 0.1, "hi": 0.5},
]


def narrowed_interval(lo_i, hi_i, s_left, lo_rest, hi_rest):
    """[max(L_i, s − ΣU_ост), min(U_i, s − ΣL_ост)] — формула narrowing
    (iter31 _narrowing_split / iter34 admissible_sum_interval)."""
    return (max(lo_i, s_left - hi_rest), min(hi_i, s_left - lo_rest))


class TestNarrowingGolden:
    """Числа не зависят от снимка кода — только от формулы narrowing."""

    def test_hi_non_monotone_040_070_05333(self):
        # шаг 1: ось c1 (остаток = c2+c3): hi режет СОБСТВЕННЫЙ U = 0.40
        a1, b1 = narrowed_interval(0.2, 0.4, 1.0, 0.1 + 0.1, 0.7 + 0.5)
        assert (a1, b1) == pytest.approx((0.2, 0.40))
        # шаг 2 при c1=0.2: hi = 0.70 — ПОЛКА собственного U
        a2, b2 = narrowed_interval(0.1, 0.7, 1.0 - 0.2, 0.1, 0.5)
        assert (a2, b2) == pytest.approx((0.3, 0.70))
        # шаг 2 при c1=11/30: hi = 8/15 ≈ 0.5333 — режет остаток массы
        a3, b3 = narrowed_interval(0.1, 0.7, 1.0 - 11.0 / 30.0, 0.1, 0.5)
        assert b3 == pytest.approx(8.0 / 15.0)         # 0.5333…
        # немонотонность по потреблённой сумме: 0.40 → 0.70 → 0.5333
        assert b1 < b2 and b3 < b2

    def test_sampler_respects_narrowing_envelope(self):
        """Все выборки PhrSpec лежат в marginal-огибающей формулы и
        достигают её краёв; совместное ограничение c2 ≤ 0.9 − c1 (c3 ≥ 0.1)
        выполняется точно."""
        spec = PhrSpec.from_dicts(NARROW_DICTS)
        Z = spec.sample_z(4096, seed=3)
        col = {nm: j for j, nm in enumerate(spec.z_names)}
        c1 = Z[:, col["c1"]]
        c2 = Z[:, col["c2"]]
        c3 = Z[:, col["c3"]]
        np.testing.assert_allclose(c1 + c2 + c3, 1.0, atol=1e-9)
        # marginal-огибающая каждой оси (формула narrowing при s=1)
        for v, (lo_i, hi_i), rest in [
                (c1, (0.2, 0.4), (0.1 + 0.1, 0.7 + 0.5)),
                (c2, (0.1, 0.7), (0.2 + 0.1, 0.4 + 0.5)),
                (c3, (0.1, 0.5), (0.2 + 0.1, 0.4 + 0.7))]:
            a, b = narrowed_interval(lo_i, hi_i, 1.0, rest[0], rest[1])
            assert v.min() >= a - 1e-9 and v.max() <= b + 1e-9
        # совместное ограничение точное (не только в огибающей)
        assert np.all(c2 <= 0.9 - c1 + 1e-9)
        # края достижимы (полка 0.70 у c2 и потолок 0.40 у c1)
        assert c1.max() > 0.38 and c2.max() > 0.65 and c2.min() < 0.20


# ----------------------------------------------------------------------
# C. Round-trip anchors (исторические рецепты в phr)
# ----------------------------------------------------------------------
class TestAnchors:

    def test_anchor_roundtrip_with_golden_z(self):
        spec = _spec()
        p = _anchor_vec(spec)
        z = spec.encode(p)
        col = {nm: j for j, nm in enumerate(spec.z_names)}
        assert z[col["PVC_67"]] == pytest.approx(0.70)     # доля смолы
        assert z[col["PVC_71"]] == pytest.approx(0.30)
        assert z[col["UV_CSFCP"]] == pytest.approx(0.2)    # absolute = phr
        assert z[col["DINP"]] == pytest.approx(10.0)       # absolute = phr
        np.testing.assert_allclose(spec.decode(z), p, atol=1e-9)

    def test_anchor_outside_spec_rejected(self):
        spec = _spec()
        bad = _anchor_vec(spec)
        bad[spec.component_names.index("UV_CSFCP")] = 0.35   # > hi=0.30
        with pytest.raises(ValueError, match="UV_CSFCP"):
            spec.encode(bad)

    def test_anchor_above_solubility_cap_rejected(self):
        spec = _spec()
        bad = _anchor_vec(spec)
        bad[spec.component_names.index("DINP")] = 4.0        # потолок = 0.12
        assert bad[spec.component_names.index("UV_CSFCP")] == 0.2
        with pytest.raises(ValueError, match="UV_CSFCP.*потолок"):
            spec.encode(bad)

    def test_anchor_resin_split_off_100_rejected(self):
        spec = _spec()
        bad = _anchor_vec(spec)
        bad[spec.component_names.index("PVC_71")] = 25.0     # 70+25 ≠ 100
        with pytest.raises(ValueError, match="resin"):
            spec.encode(bad)


# ----------------------------------------------------------------------
# D. spec_hash: порядок узлов — часть спеки
# ----------------------------------------------------------------------
class TestSpecHash:

    def test_stable_and_roundtrip(self):
        h1 = _spec().spec_hash()
        h2 = _spec().spec_hash()
        assert h1 == h2 and len(h1) == 64
        # from_dicts(to_dicts) — та же спека, тот же отпечаток
        clone = PhrSpec.from_dicts(_spec().to_dicts())
        assert clone.spec_hash() == h1
        assert clone.component_names == COMPONENTS

    def test_node_order_changes_hash(self):
        """Порядок узлов влияет на меру сэмплера (iter34, находка 1) —
        перестановка ОБЯЗАНА давать другой отпечаток."""
        d = [dict(x) for x in RECIPE_DICTS]
        i = next(j for j, x in enumerate(d) if x["name"] == "Chalk_1T")
        k = next(j for j, x in enumerate(d) if x["name"] == "Chalk_95T")
        d[i], d[k] = d[k], d[i]
        assert PhrSpec.from_dicts(d).spec_hash() != _spec().spec_hash()

    def test_bound_change_changes_hash(self):
        d = [dict(x) for x in RECIPE_DICTS]
        next(x for x in d if x["name"] == "DINP")["hi"] = 15.0
        assert PhrSpec.from_dicts(d).spec_hash() != _spec().spec_hash()

    def test_cap_change_changes_hash(self):
        """Потолок растворимости — часть геометрии: другой cap_ratio —
        другая спека (и другой отпечаток)."""
        d = [dict(x) for x in RECIPE_DICTS]
        next(x for x in d if x["name"] == "UV_CSFCP")["cap_ratio"] = 0.04
        assert PhrSpec.from_dicts(d).spec_hash() != _spec().spec_hash()


# ----------------------------------------------------------------------
# E. Правило премикса (арифметика навески; SBM_55 / UV_CSFCP)
# ----------------------------------------------------------------------
class TestPremixRule:
    DELTA = 0.02      # весы 0.1 г при 5 г на 1 phr

    def test_sbm_and_uv_need_premix(self):
        assert premix_required(self.DELTA, 0.07, 0.45)   # 0.0526 > 0.05
        assert premix_required(self.DELTA, 0.05, 0.30)   # 0.08 > 0.05

    def test_wide_axes_direct_weighing(self):
        assert not premix_required(self.DELTA, 4.0, 14.0)     # DINP
        assert not premix_required(self.DELTA, 1.5, 25.0)     # Chalk_95T

    def test_fine_scale_removes_premix(self):
        assert not premix_required(0.001, 0.05, 0.30)

    def test_degenerate_range_rejected(self):
        with pytest.raises(ValueError, match="вырожденный"):
            premix_required(self.DELTA, 0.2, 0.2)


# ----------------------------------------------------------------------
# F. Регрессия архитектуры: кампания идёт через PhrSpec, не через
#    rejection-путь SimplexRegion; preflight — в том же пространстве
# ----------------------------------------------------------------------
class _Oracle:
    property_names = ["gloss"]

    def evaluate(self, Xc):
        Xc = np.atleast_2d(np.asarray(Xc, float))
        return (10.0 * Xc[:, 0]).reshape(-1, 1)


def _runner():
    spec = _spec()
    lo, hi = spec.fraction_bounds()
    schema = ProjectSchema.mixture_only(
        spec.component_names, lower=lo.tolist(), upper=hi.tolist(),
        model=ModelSpec(cross_level="additive", mixture_order="quadratic"))
    runner = MixtureProcessRunner(schema, _Oracle(), seed=0)
    runner.set_phr_spec(spec)
    return runner, spec


class TestCampaignPath:

    def test_candidates_never_hit_rejection_path(self):
        """До PhrSpec этот регион давал acceptance 0/200000 и уходил в
        fallback; phr-путь обязан работать БЕЗ единого warning'а."""
        runner, spec = _runner()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            X = runner._phase_candidates(256, seed=7)
        assert not [x for x in w if issubclass(x.category, UserWarning)], \
            [str(x.message) for x in w]
        assert X.shape == (256, 17)
        np.testing.assert_allclose(X.sum(axis=1), 1.0, atol=1e-9)
        uv = spec.component_names.index("UV_CSFCP")
        dinp = spec.component_names.index("DINP")
        ratio = X[:, uv] / X[:, dinp]
        assert np.all(ratio <= UV_CAP + 1e-9)          # трапеция: потолок

    def test_preflight_reference_in_phr_space(self):
        """iter32-гейты считаются относительно reference, построенного ТЕМ ЖЕ
        phr-путём (_phase_candidates уважает set_phr_spec) — сравнение
        like-with-like, а не wf-бокс против phr-кандидатов."""
        runner, spec = _runner()
        X = runner._phase_candidates(48, seed=5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rep = runner.preflight(X, n_ref=128)
        # reference-пул не провалился в rejection-fallback SimplexRegion
        assert not [x for x in w if issubclass(x.category, UserWarning)
                    and "rejection sampling" in str(x.message)]
        assert isinstance(rep.passed, bool)