# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 49 — шаг B7 ревизии контракта phr-спеки: контракт-ответ
ядра на точку (:meth:`PhrSpec.point_report`).

Три слоя ответа (UI_REVISION_SPEC, план B7):

  * ``effective_bounds`` — по КАЖДОМУ узлу спеки (включая внутренние
    тоталы и производные closure) эффективные границы его собственной
    координаты В ТОЧКЕ + метки ``active_lo``/``active_hi`` — какое
    ограничение задало границу (fixed / range / derived / window /
    cap / min_phr / max_phr / partners);
  * ``premix`` — правило премикса по СТАТИЧЕСКОМУ интервалу phr листа
    (``None`` = правило неприменимо: δ не задан или интервал вырожден);
  * ``phr_nominal`` vs ``phr_actual`` — РАЗДЕЛЬНО (actual — снап к
    δ-сетке весов, только при заданном ``delta_phr``).

A0.6: номинал вне геометрии и пустые границы — строки ``violations``
(``ok=False``), НЕ исключения; исключения — только ошибки данных
(длина ``p``, нулевой тотал/референс, ``delta_phr ≤ 0``).

Golden для active-меток — немонотонная функция ``hi(T)`` iter45:
``hi(T) = min(0.70, 8/T, 1 − 3/T)`` → 0.40 @T=5 (partners),
полка 0.70 на T∈[10; 11.4286] (range — тай-брейк в пользу простого
объяснения), 0.5333 @T=15 (max_phr).
"""
import numpy as np
import pytest

from src.design.phr_sampler import PhrSpec

# ----------------------------------------------------------------------
# PVC-подобная v2-спека: все режимы координат (fixed / absolute /
# absolute+cap+log / group total / share free+closure с phr-лимитами /
# ratio_to). Числа групп SOFT — golden iter45.
# ----------------------------------------------------------------------
NODES = [
    {"name": "RESIN", "role": "FIXED", "value": 100.0},
    {"name": "DINP", "role": "ABSOLUTE", "range": [4.0, 14.0]},
    {"name": "ESO", "role": "FIXED", "value": 2.5},
    {"name": "SOFT", "role": "GROUP_TOTAL", "range": [5.0, 15.0],
     "members": ["PBNK", "CPE"]},
    {"name": "PBNK", "role": "SHARE_FREE", "group": "SOFT",
     "share_range": [0.0, 0.70], "max_phr": 8.0},
    {"name": "CPE", "role": "SHARE_CLOSURE", "group": "SOFT",
     "min_phr": 3.0},
    {"name": "STAB", "role": "GROUP_TOTAL", "range": [3.5, 5.0],
     "members": ["PF_LB", "PF"]},
    {"name": "PF_LB", "role": "SHARE_FREE", "group": "STAB",
     "share_range": [0.0, 0.40]},
    {"name": "PF", "role": "SHARE_CLOSURE", "group": "STAB"},
    {"name": "SBM", "role": "RATIO_TO", "reference": "STAB",
     "range": [0.02, 0.09]},
    {"name": "TiO2", "role": "ABSOLUTE", "range": [0.3, 8.0],
     "scale": "log"},
    {"name": "UV", "role": "ABSOLUTE_CAPPED", "range": [0.05, 0.30],
     "scale": "log", "cap_to": ["DINP", "ESO"], "cap_ratio": 0.03},
]


def _spec() -> PhrSpec:
    return PhrSpec.from_dicts(NODES)


def _point(dinp=6.0, t_soft=5.0, phi_pbnk=0.2, t_stab=4.0,
           phi_pflb=0.25, r_sbm=0.05, tio2=1.0, uv=0.10):
    """Рецепт в phr, порядок = component_names:
    RESIN, DINP, ESO, PBNK, CPE, PF_LB, PF, SBM, TiO2, UV."""
    return [100.0, dinp, 2.5,
            phi_pbnk * t_soft, (1.0 - phi_pbnk) * t_soft,
            phi_pflb * t_stab, (1.0 - phi_pflb) * t_stab,
            r_sbm * t_stab, tio2, uv]


# ======================================================================
# 1. Структура ответа
# ======================================================================
class TestStructure:

    def test_covers_all_nodes_in_spec_order(self):
        spec = _spec()
        rep = spec.point_report(_point())
        assert list(rep.effective_bounds) == [d["name"] for d in NODES]
        for nm, b in rep.effective_bounds.items():
            assert b.name == nm
            assert np.isfinite(b.coord) and np.isfinite(b.phr)

    def test_without_delta_actual_none_premix_none(self):
        spec = _spec()
        rep = spec.point_report(_point())
        assert rep.ok
        assert rep.phr_actual is None
        assert rep.delta_phr is None
        assert all(v is None for v in rep.premix.values())
        np.testing.assert_allclose(rep.phr_nominal, _point())

    def test_internal_totals_reconstructed_from_children(self):
        spec = _spec()
        rep = spec.point_report(_point(t_soft=7.0, t_stab=4.5))
        assert rep.effective_bounds["SOFT"].phr == pytest.approx(7.0)
        assert rep.effective_bounds["STAB"].phr == pytest.approx(4.5)

    def test_decoded_points_are_ok(self):
        """Точки самого сэмплера обязаны проходить собственный контракт."""
        spec = _spec()
        P = spec.decode(spec.sample_z(40, seed=7))
        for row in P:
            rep = spec.point_report(row)
            assert rep.ok, rep.violations


# ======================================================================
# 2. Метки active: какое ограничение сработало
# ======================================================================
class TestActiveLabels:

    def test_fixed_and_plain_range(self):
        rep = _spec().point_report(_point())
        b = rep.effective_bounds["RESIN"]
        assert (b.lo, b.hi) == (100.0, 100.0)
        assert (b.active_lo, b.active_hi) == ("fixed", "fixed")
        d = rep.effective_bounds["DINP"]
        assert (d.lo, d.hi) == (4.0, 14.0)
        assert (d.active_lo, d.active_hi) == ("range", "range")
        s = rep.effective_bounds["SBM"]
        assert (s.lo, s.hi) == (0.02, 0.09)
        assert s.coord == pytest.approx(0.05)

    def test_cap_active_depends_on_point(self):
        spec = _spec()
        # фаза мала: потолок 0.03·(6+2.5)=0.255 < 0.30 → cap активен
        b = spec.point_report(_point(dinp=6.0)).effective_bounds["UV"]
        assert b.hi == pytest.approx(0.255)
        assert b.active_hi == "cap"
        assert b.active_lo == "range"
        # фаза велика: потолок 0.03·(12+2.5)=0.435 > 0.30 → range
        b2 = spec.point_report(_point(dinp=12.0)).effective_bounds["UV"]
        assert b2.hi == pytest.approx(0.30)
        assert b2.active_hi == "range"

    def test_log_axis_bounds_reported_in_phr(self):
        """Контракт отвечает в физических единицах: у log-оси границы
        в phr, а не в ln phr (шкала — деталь сэмплера)."""
        b = _spec().point_report(_point()).effective_bounds["TiO2"]
        assert (b.lo, b.hi) == (0.3, 8.0)
        assert b.coord == pytest.approx(1.0)

    def test_share_hi_golden_iter45(self):
        """hi(T) = min(0.70, 8/T, 1−3/T) немонотонна: активное
        ограничение меняется с тоталом."""
        spec = _spec()
        # T=5: партнёрский min_phr давит сильнее всех → 0.40, partners
        b5 = spec.point_report(
            _point(t_soft=5.0, phi_pbnk=0.2)).effective_bounds["PBNK"]
        assert b5.hi == pytest.approx(0.40)
        assert b5.active_hi == "partners"
        # T=15: собственный max_phr → 8/15, max_phr
        b15 = spec.point_report(
            _point(t_soft=15.0, phi_pbnk=0.2)).effective_bounds["PBNK"]
        assert b15.hi == pytest.approx(8.0 / 15.0)
        assert b15.active_hi == "max_phr"
        # T=10.5 (полка): всё упирается в заявленный share_range —
        # при равенстве кандидатов тай-брейк в пользу простого объяснения
        b105 = spec.point_report(
            _point(t_soft=10.5, phi_pbnk=0.2)).effective_bounds["PBNK"]
        assert b105.hi == pytest.approx(0.70)
        assert b105.active_hi == "range"

    def test_closure_derived_and_min_phr(self):
        spec = _spec()
        b = spec.point_report(
            _point(t_soft=5.0, phi_pbnk=0.2)).effective_bounds["CPE"]
        # lo: техминимум 3 phr при T=5 → 0.6, сильнее производного 0.3
        assert b.lo == pytest.approx(0.6)
        assert b.active_lo == "min_phr"
        # hi: производный диапазон closure (1 − φᴸ_free = 1.0)
        assert b.hi == pytest.approx(1.0)
        assert b.active_hi == "derived"

    def test_matches_share_bounds_at_total(self):
        """Границы в отчёте согласованы с share_bounds_at_total."""
        spec = _spec()
        for t in (5.0, 8.0, 10.5, 15.0):
            rep = spec.point_report(_point(t_soft=t, phi_pbnk=0.1))
            lo, hi = spec.share_bounds_at_total("SOFT", t)
            for i, nm in enumerate(["PBNK", "CPE"]):
                b = rep.effective_bounds[nm]
                assert b.lo == pytest.approx(lo[i])
                assert b.hi == pytest.approx(hi[i])

    def test_window_label_on_narrowed_total(self):
        """min_phr=6 у closure сужает окно тотала SOFT до [6, 15] —
        нижняя граница тотала получает метку window."""
        nodes = [dict(d) for d in NODES]
        for d in nodes:
            if d["name"] == "CPE":
                d["min_phr"] = 6.0
        spec = PhrSpec.from_dicts(nodes)
        rep = spec.point_report(_point(t_soft=7.0, phi_pbnk=0.1))
        b = rep.effective_bounds["SOFT"]
        assert b.lo == pytest.approx(6.0)
        assert b.active_lo == "window"
        assert b.hi == pytest.approx(15.0)
        assert b.active_hi == "range"
        assert rep.ok, rep.violations


# ======================================================================
# 3. Премикс и nominal vs actual
# ======================================================================
class TestPremixAndActual:

    def test_premix_golden(self):
        """Golden CAMPAIGN_SPEC_PVC §5 (δ=0.02 phr): SBM [0.07, 0.45] и
        UV [0.05, 0.30] → премикс; DINP [4, 14] → прямая навеска;
        fixed-оси → None («правило неприменимо»)."""
        rep = _spec().point_report(_point(), delta_phr=0.02)
        assert rep.premix["SBM"] is True
        assert rep.premix["UV"] is True
        assert rep.premix["DINP"] is False
        assert rep.premix["TiO2"] is False
        assert rep.premix["RESIN"] is None
        assert rep.premix["ESO"] is None

    def test_actual_matches_quantize_and_ok_on_grid(self):
        spec = _spec()
        p = _point()                        # все листья кратны 0.02
        rep = spec.point_report(p, delta_phr=0.02)
        assert rep.ok, rep.violations
        qr = spec.quantize_recipe(p, 0.02)
        np.testing.assert_allclose(rep.phr_actual, qr.p_actual)
        np.testing.assert_allclose(rep.phr_actual, p, atol=1e-9)
        np.testing.assert_allclose(rep.phr_nominal, p)
        assert rep.delta_phr == pytest.approx(0.02)

    def test_actual_snaps_off_grid_nominal(self):
        """nominal и actual — РАЗДЕЛЬНО: actual снапится к δ-сетке,
        nominal остаётся как предложило ядро."""
        spec = _spec()
        p = _point(dinp=6.007)
        rep = spec.point_report(p, delta_phr=0.02)
        j = spec.component_names.index("DINP")
        assert rep.phr_nominal[j] == pytest.approx(6.007)
        assert rep.phr_actual[j] == pytest.approx(6.0)

    def test_quantize_violations_propagate(self):
        """δ шире интервала оси → в интервале нет узла сетки; нарушение
        квантования попадает в общий список отчёта."""
        rep = _spec().point_report(_point(), delta_phr=0.5)
        assert not rep.ok
        assert any("нет узла сетки" in v for v in rep.violations)


# ======================================================================
# 4. Диагностика (A0.6) и ошибки данных
# ======================================================================
class TestViolationsAndErrors:

    def test_nominal_out_of_bounds_flagged_not_raised(self):
        rep = _spec().point_report(_point(dinp=3.0))
        assert not rep.ok
        assert any(v.startswith("DINP:") and "вне эффективных границ" in v
                   for v in rep.violations)

    def test_uv_above_point_cap_flagged(self):
        # 0.28 < статического hi=0.30, но > потолка точки 0.255
        rep = _spec().point_report(_point(dinp=6.0, uv=0.28))
        assert not rep.ok
        assert any(v.startswith("UV:") and "cap" in v
                   for v in rep.violations)

    def test_min_phr_violation_flagged(self):
        # T=5, φ_CPE=0.5 → 2.5 phr < техминимума 3.0 (lo_eff=0.6)
        rep = _spec().point_report(_point(t_soft=5.0, phi_pbnk=0.5))
        assert not rep.ok
        assert any(v.startswith("CPE:") and "min_phr" in v
                   for v in rep.violations)

    def test_empty_effective_bounds_flagged(self):
        """Референс ниже своего lo → потолок точки падает ниже lo
        cap-оси: границы пусты — диагностика, не исключение."""
        spec = PhrSpec.from_dicts([
            {"name": "base", "role": "FIXED", "value": 10.0},
            {"name": "ref", "role": "ABSOLUTE", "range": [1.0, 10.0]},
            {"name": "capped", "role": "ABSOLUTE_CAPPED",
             "range": [0.5, 1.0], "cap_to": ["ref"], "cap_ratio": 0.5},
        ])
        rep = spec.point_report([10.0, 0.6, 0.8])
        assert not rep.ok
        assert any(v.startswith("ref:") for v in rep.violations)
        assert any(v.startswith("capped:")
                   and "границы пусты" in v for v in rep.violations)

    def test_wrong_size_raises(self):
        with pytest.raises(ValueError, match="компонентов"):
            _spec().point_report([1.0, 2.0])

    def test_zero_group_total_raises(self):
        p = _point()
        p[3] = p[4] = 0.0                    # PBNK = CPE = 0 → T_SOFT = 0
        with pytest.raises(ValueError, match="тотал"):
            _spec().point_report(p)

    def test_nonpositive_delta_raises(self):
        with pytest.raises(ValueError, match="delta_phr"):
            _spec().point_report(_point(), delta_phr=0.0)