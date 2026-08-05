# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 47 — шаг B5 ревизии контракта phr-спеки: лог-сэмплинг.

Оси ``scale='log'`` (TiO2_BLR895, UV_CSFCP в PVC-спеке): z-координата —
``ln phr``, сэмплинг равномерен в ``[ln lo, ln hi]``, границы
ЛОГАРИФМИРУЮТСЯ; cap-потолок применяется ПОСЛЕ экспоненцирования
(потолок в phr, не в логах: логарифмируется уже суженная граница
``min(hi, cap_ratio·Σref)``, а не сами референсы).

Мотивация (UI_REVISION_SPEC, B5): доля точек TiO2 < 1 phr — 9,1 %
uniform против 36,7 % log; UV < 0,12 phr — 28 % против 49 %; отклик
по УФ экстремально сатурирующий (при 0,12 phr A₃₄₁=3,3) — вся
информация в нижней декаде. Референсные доли — это АНАЛИТИЧЕСКИЕ
значения лог-равномерной маргинали, они и проверяются:

    P(p < c) = ln(c/lo) / ln(hi/lo).
"""
import math

import numpy as np
import pytest

from src.design.phr_sampler import PhrSpec

# ----------------------------------------------------------------------
# Референсные оси B5 (диапазоны PVC-спеки «pvc_edge_v1»)
# ----------------------------------------------------------------------
SPEC_TIO2 = [
    {"name": "base", "role": "FIXED", "value": 100.0},
    {"name": "TiO2", "role": "ABSOLUTE", "range": [0.3, 8.0],
     "scale": "log"},
]

SPEC_UV_FREE = [                      # УФ без cap: чистая маргиналь
    {"name": "base", "role": "FIXED", "value": 100.0},
    {"name": "UV", "role": "ABSOLUTE", "range": [0.05, 0.30],
     "scale": "log"},
]

SPEC_UV_CAP = [                       # УФ с потолком по фазе (как в PVC)
    {"name": "base", "role": "FIXED", "value": 100.0},
    {"name": "DINP", "role": "ABSOLUTE", "range": [4.0, 14.0]},
    {"name": "ESO", "role": "FIXED", "value": 2.50},
    {"name": "UV", "role": "ABSOLUTE_CAPPED", "range": [0.05, 0.30],
     "scale": "log", "cap_to": ["DINP", "ESO"], "cap_ratio": 0.03},
]


def _col(spec, name):
    return spec.component_names.index(name)


def _zcol(spec, name):
    return spec.z_names.index(name)


# ======================================================================
# 1. Маргинали: лог-равномерность даёт плотность в нижней декаде
# ======================================================================
class TestLogMarginal:

    def test_tio2_lower_decade_share(self):
        # P(p < 1) = ln(1/0.3)/ln(8/0.3) ≈ 0.3667 (uniform дал бы 0.091)
        spec = PhrSpec.from_dicts(SPEC_TIO2)
        P = spec.decode(spec.sample_z(20000, seed=0))
        share = float(np.mean(P[:, _col(spec, "TiO2")] < 1.0))
        ref = math.log(1.0 / 0.3) / math.log(8.0 / 0.3)
        assert share == pytest.approx(ref, abs=0.02)
        assert share > 0.30                       # заведомо НЕ uniform (9.1%)

    def test_uv_lower_decade_share(self):
        # P(p < 0.12) = ln(0.12/0.05)/ln(0.30/0.05) ≈ 0.489 (uniform: 0.28)
        spec = PhrSpec.from_dicts(SPEC_UV_FREE)
        P = spec.decode(spec.sample_z(20000, seed=1))
        share = float(np.mean(P[:, _col(spec, "UV")] < 0.12))
        ref = math.log(0.12 / 0.05) / math.log(0.30 / 0.05)
        assert share == pytest.approx(ref, abs=0.02)
        assert share > 0.40                       # заведомо НЕ uniform (28%)

    def test_ln_z_uniform_and_in_bounds(self):
        spec = PhrSpec.from_dicts(SPEC_TIO2)
        Z = spec.sample_z(20000, seed=2)
        z = Z[:, _zcol(spec, "TiO2")]
        ln_lo, ln_hi = math.log(0.3), math.log(8.0)
        assert np.all(z >= ln_lo) and np.all(z <= ln_hi)
        # равномерность ln z: среднее и дисперсия аналитические
        assert float(z.mean()) == pytest.approx(0.5 * (ln_lo + ln_hi),
                                                abs=0.03)
        assert float(z.var()) == pytest.approx((ln_hi - ln_lo) ** 2 / 12.0,
                                               rel=0.05)

    def test_decoded_phr_in_declared_range(self):
        spec = PhrSpec.from_dicts(SPEC_TIO2)
        p = spec.decode(spec.sample_z(2000, seed=3))[:, _col(spec, "TiO2")]
        assert np.all(p >= 0.3 - 1e-12) and np.all(p <= 8.0 + 1e-12)


# ======================================================================
# 2. Геометрия z: границы логарифмируются, roundtrip, clip
# ======================================================================
class TestZGeometry:

    def test_z_bounds_logarithmized(self):
        spec = PhrSpec.from_dicts(SPEC_TIO2)
        lo, hi = spec.z_bounds()
        j = _zcol(spec, "TiO2")
        assert lo[j] == pytest.approx(math.log(0.3))
        assert hi[j] == pytest.approx(math.log(8.0))

    def test_decode_encode_roundtrip(self):
        spec = PhrSpec.from_dicts(SPEC_UV_CAP)
        Z = spec.sample_z(200, seed=4)
        assert np.allclose(spec.encode(spec.decode(Z)), Z, atol=1e-9)

    def test_encode_checks_bounds_in_phr(self):
        spec = PhrSpec.from_dicts(SPEC_TIO2)
        z = spec.encode([100.0, 2.0])             # p=2 phr → z=ln 2
        assert z[_zcol(spec, "TiO2")] == pytest.approx(math.log(2.0))
        with pytest.raises(ValueError, match="вне границ"):
            spec.encode([100.0, 10.0])            # 10 > hi=8 (проверка в phr)

    def test_clip_z_projects_in_log_scale(self):
        spec = PhrSpec.from_dicts(SPEC_TIO2)
        j = _zcol(spec, "TiO2")
        ln_lo, ln_hi = math.log(0.3), math.log(8.0)
        # валидные точки не двигаются
        Z = spec.sample_z(100, seed=5)
        assert np.allclose(spec.clip_z(Z), Z)
        # выход за границы клипится к ln-границам; проекция идемпотентна
        zbad = np.array([[5.0], [-9.0]])          # e⁵≈148 phr / e⁻⁹≈0 phr
        zc = spec.clip_z(zbad)
        assert zc[0, j] == pytest.approx(ln_hi)
        assert zc[1, j] == pytest.approx(ln_lo)
        assert np.allclose(spec.clip_z(zc), zc)


# ======================================================================
# 3. Cap применяется ПОСЛЕ экспоненцирования (потолок в phr)
# ======================================================================
class TestCapAfterExp:

    def test_sampled_uv_respects_phr_cap(self):
        spec = PhrSpec.from_dicts(SPEC_UV_CAP)
        P = spec.decode(spec.sample_z(4000, seed=6))
        uv = P[:, _col(spec, "UV")]
        cap = 0.03 * (P[:, _col(spec, "DINP")] + 2.50)   # потолок В PHR
        assert np.all(uv <= np.minimum(0.30, cap) + 1e-9)
        assert np.all(uv >= 0.05 - 1e-12)
        # трапеция живая: у точек с активным cap (< 0.30) верх достижим —
        # условная ЛОГ-равномерность в [lo, cap], а не глухая зона под 0.30
        capped = cap < 0.30
        assert np.any(capped)
        assert float(np.max(uv[capped] / cap[capped])) > 0.95

    def test_clip_z_cap_upstream_wins(self):
        # DINP=4 → cap = 0.03·(4+2.5) = 0.195 < 0.30: UV опускается к
        # ln(0.195), референс (доминирующая ось) не трогается
        spec = PhrSpec.from_dicts(SPEC_UV_CAP)
        jd, ju = _zcol(spec, "DINP"), _zcol(spec, "UV")
        z = np.zeros(spec.dim_z)
        z[jd] = 4.0
        z[ju] = math.log(0.30)
        zc = spec.clip_z(z)
        assert zc[jd] == pytest.approx(4.0)
        assert zc[ju] == pytest.approx(math.log(0.195))
        p = spec.decode(zc)
        assert p[_col(spec, "UV")] == pytest.approx(0.195)

    def test_encode_rejects_above_cap(self):
        spec = PhrSpec.from_dicts(SPEC_UV_CAP)
        # base, DINP, ESO, UV — порядок component_names = порядок спеки
        with pytest.raises(ValueError, match="превышает потолок"):
            spec.encode([100.0, 4.0, 2.50, 0.25])  # cap=0.195 < 0.25 ≤ hi

    def test_ln_uniform_conditional_on_cap(self):
        # при ФИКСИРОВАННОМ потолке (DINP сужен до вырожденного диапазона
        # нельзя — absolute требует lo<hi у premix, но не у спеки; берём
        # узкий) маргиналь UV близка к лог-равномерной на [0.05, cap]
        nodes = [dict(d) for d in SPEC_UV_CAP]
        nodes[1] = dict(nodes[1], range=[4.0, 4.0001])   # cap ≈ 0.195
        spec = PhrSpec.from_dicts(nodes)
        P = spec.decode(spec.sample_z(20000, seed=7))
        uv = P[:, _col(spec, "UV")]
        cap = 0.03 * (4.0 + 2.50)
        # P(p < 0.1 | потолок cap) = ln(0.1/0.05)/ln(cap/0.05)
        ref = math.log(0.1 / 0.05) / math.log(cap / 0.05)
        assert float(np.mean(uv < 0.1)) == pytest.approx(ref, abs=0.02)


# ======================================================================
# 4. Интеграция: log-оси в спеке с группами/ratio; hash не зависит от B5
# ======================================================================
class TestIntegration:

    MIXED = [
        {"name": "base", "role": "FIXED", "value": 100.0},
        {"name": "G", "role": "GROUP_TOTAL", "range": [10.0, 20.0],
         "members": ["X", "Y"]},
        {"name": "X", "role": "SHARE_FREE", "group": "G",
         "share_range": [0.2, 0.8]},
        {"name": "Y", "role": "SHARE_CLOSURE", "group": "G"},
        {"name": "R", "role": "RATIO_TO", "reference": "G",
         "range": [0.02, 0.09]},
        {"name": "TiO2", "role": "ABSOLUTE", "range": [0.3, 8.0],
         "scale": "log"},
    ]

    def test_mixed_spec_geometry(self):
        spec = PhrSpec.from_dicts(self.MIXED)
        Z = spec.sample_z(300, seed=8)
        assert Z.shape == (300, spec.dim_z)
        assert np.linalg.matrix_rank(Z - Z.mean(axis=0)) == spec.dim_z
        X = spec.sample_candidates(64, seed=9)
        assert np.allclose(X.sum(axis=1), 1.0)
        assert np.allclose(spec.encode(spec.decode(Z)), Z, atol=1e-9)

    def test_phr_intervals_stay_in_phr(self):
        # интервальная валидация/premix-слой работают В PHR, лог их не гнёт
        spec = PhrSpec.from_dicts(SPEC_TIO2)
        assert spec.phr_intervals()["TiO2"] == pytest.approx((0.3, 8.0))

    def test_hash_stable_and_scale_sensitive(self):
        # B5 меняет только геометрические ОПЕРАЦИИ; сериализация/хеш —
        # прежние (iter46/B6): round-trip стабилен, scale входит в хеш
        spec = PhrSpec.from_dicts(SPEC_TIO2)
        assert PhrSpec.from_dicts(spec.to_dicts()).spec_hash() \
            == spec.spec_hash()
        linear = PhrSpec.from_dicts([
            SPEC_TIO2[0],
            {"name": "TiO2", "role": "ABSOLUTE", "range": [0.3, 8.0]},
        ])
        assert linear.spec_hash() != spec.spec_hash()