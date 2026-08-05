# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 46 — шаги B2+B6+B8 ревизии контракта phr-спеки («pvc_edge_v1»).

Причина B2: legacy ``share_of`` даёт каждому члену группы свою z-ось —
пары (φ, 1−φ) точно коллинеарны: rank(Z)=16 при dim_z=22, cond→∞, VIF→∞,
ARD-длины пар не идентифицируются, preflight-гейт считает не то
пространство. Новый контракт:

  * роли ``SHARE_FREE``/``SHARE_CLOSURE`` (k=2) и ``SHARE_SIMPLEX`` (k≥3);
  * closure и ПОСЛЕДНИЙ член simplex-группы — БЕЗ z-оси (производные
    координаты ``1 − Σ партнёров``): PVC-спека 22 → 16 осей, rank=dim;
  * схема сериализации v2 (B6): ``role``/``range``/``share_range``/
    ``group``/``members``/``reference``/``scale``/``spec_version``;
    legacy-схема v1 (``mode``) работает как раньше (старые хеши валидны);
  * валидация «closure/fixed без range» (B8) — ошибка, не тихое
    игнорирование; ``members`` обязаны ТОЧНО совпадать с детьми группы;
  * C5: share-бокс группы ``φᵢᵁ ≤ 1 − Σ_{j≠i} φⱼᴸ`` НЕСТРОГО (LUB впритык
    0.60 = 1 − 0.40 — не ошибка);
  * ``scale='log'`` принимается схемой/хешем; лог-сэмплинг реализован
    в iter47/B5 (см. test_iteration47_log_sampling.py).

Golden-хеш v2-спеки в тесты НЕ закладывается (решение C4) — проверяются
только round-trip-инварианты.
"""
import json
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.apps import campaign_state as cst
from src.apps import campaign_ui as ui
from src.apps.campaign_ui import build_setup_runner
from src.design.phr_sampler import PhrNode, PhrSpec

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ----------------------------------------------------------------------
# Референсная PVC-спека контракта «pvc_edge_v1» в схеме v2 (16 z-осей).
# ----------------------------------------------------------------------
PVC_V2_NODES = [
    {"name": "RESIN.total", "role": "GROUP_TOTAL_FIXED", "value": 100.0,
     "members": ["PVC_67", "PVC_71"]},
    {"name": "PVC_67", "role": "SHARE_FREE", "group": "RESIN.total",
     "share_range": [0.30, 1.00]},
    {"name": "PVC_71", "role": "SHARE_CLOSURE", "group": "RESIN.total"},
    {"name": "DINP", "role": "ABSOLUTE", "range": [4.0, 14.0]},
    {"name": "ESO", "role": "FIXED", "value": 2.50},
    {"name": "FILLER.total", "role": "GROUP_TOTAL", "range": [5.0, 25.0],
     "members": ["Chalk_95T", "Chalk_1T"]},
    {"name": "Chalk_95T", "role": "SHARE_FREE", "group": "FILLER.total",
     "share_range": [0.30, 1.00]},
    {"name": "Chalk_1T", "role": "SHARE_CLOSURE", "group": "FILLER.total"},
    {"name": "SOFT.total", "role": "GROUP_TOTAL", "range": [5.0, 15.0],
     "members": ["PBNK_3355", "CPE_135A"]},
    {"name": "PBNK_3355", "role": "SHARE_FREE", "group": "SOFT.total",
     "share_range": [0.00, 0.70]},
    {"name": "CPE_135A", "role": "SHARE_CLOSURE", "group": "SOFT.total"},
    {"name": "ACR.total", "role": "GROUP_TOTAL", "range": [0.20, 1.20],
     "members": ["DL_531", "PMPlus_8"]},
    {"name": "DL_531", "role": "SHARE_FREE", "group": "ACR.total",
     "share_range": [0.25, 0.60]},
    {"name": "PMPlus_8", "role": "SHARE_CLOSURE", "group": "ACR.total"},
    # LUB (k=3): C5 впритык — AKLUB/OPE hi=0.60 = 1 − (0.30 + 0.10)
    {"name": "LUB.total", "role": "GROUP_TOTAL", "range": [0.40, 1.20],
     "members": ["DL_60", "AKLUB_K_435", "OPE"]},
    {"name": "DL_60", "role": "SHARE_SIMPLEX", "group": "LUB.total",
     "share_range": [0.30, 0.70]},
    {"name": "AKLUB_K_435", "role": "SHARE_SIMPLEX", "group": "LUB.total",
     "share_range": [0.10, 0.60]},
    {"name": "OPE", "role": "SHARE_SIMPLEX", "group": "LUB.total",
     "share_range": [0.10, 0.60]},
    {"name": "STAB.total", "role": "GROUP_TOTAL", "range": [3.5, 5.0],
     "members": ["PF711LB", "PF711"]},
    {"name": "PF711LB", "role": "SHARE_FREE", "group": "STAB.total",
     "share_range": [0.00, 0.40]},
    {"name": "PF711", "role": "SHARE_CLOSURE", "group": "STAB.total"},
    {"name": "SBM_55", "role": "RATIO_TO", "reference": "STAB.total",
     "range": [0.02, 0.09]},
    {"name": "TiO2_BLR895", "role": "ABSOLUTE", "range": [0.3, 8.0]},
    {"name": "UV_CSFCP", "role": "ABSOLUTE_CAPPED", "range": [0.05, 0.30],
     "cap_to": ["DINP", "ESO"], "cap_ratio": 0.03},
    {"name": "SA_1860", "role": "FIXED", "value": 0.10},
]
PVC_V2 = {"spec_version": 2, "nodes": PVC_V2_NODES}

Z_AXES_16 = ["PVC_67", "DINP", "FILLER.total", "Chalk_95T", "SOFT.total",
             "PBNK_3355", "ACR.total", "DL_531", "LUB.total", "DL_60",
             "AKLUB_K_435", "STAB.total", "PF711LB", "SBM_55",
             "TiO2_BLR895", "UV_CSFCP"]

# Та же геометрия в LEGACY-схеме v1 (плоская, до B2): 22 z-оси — контроль
# коллинеарности пар (φ, 1−φ), ради устранения которой сделан B2.
LEGACY_FLAT = [
    {"name": "RESIN.total", "mode": "fixed", "value": 100.0},
    {"name": "PVC_67", "mode": "share_of", "of": "RESIN.total",
     "lo": 0.30, "hi": 1.00},
    {"name": "PVC_71", "mode": "share_of", "of": "RESIN.total",
     "lo": 0.00, "hi": 0.70},
    {"name": "DINP", "mode": "absolute", "lo": 4.0, "hi": 14.0},
    {"name": "FILLER.total", "mode": "absolute", "lo": 5.0, "hi": 25.0},
    {"name": "Chalk_95T", "mode": "share_of", "of": "FILLER.total",
     "lo": 0.30, "hi": 1.00},
    {"name": "Chalk_1T", "mode": "share_of", "of": "FILLER.total",
     "lo": 0.00, "hi": 0.70},
]

# Малая v2-спека для быстрых проверок: base=100, группа G = X (free) + Y
# (closure), G ∈ [10, 20], φ_X ∈ [0.2, 0.8] ⇒ производный φ_Y ∈ [0.2, 0.8].
SMALL_V2 = [
    {"name": "base", "role": "FIXED", "value": 100.0},
    {"name": "G", "role": "GROUP_TOTAL", "range": [10.0, 20.0],
     "members": ["X", "Y"]},
    {"name": "X", "role": "SHARE_FREE", "group": "G",
     "share_range": [0.2, 0.8]},
    {"name": "Y", "role": "SHARE_CLOSURE", "group": "G"},
]


def _pvc():
    return PhrSpec.from_dicts(PVC_V2)


def _small(nodes=None, **node_patches):
    """Малая спека c патчами узлов по имени: _small(Y={"min_phr": 5.0})."""
    out = []
    for d in (nodes or SMALL_V2):
        d2 = dict(d)
        if d2["name"] in node_patches:
            d2.update(node_patches[d2["name"]])
        out.append(d2)
    return PhrSpec.from_dicts(out)


# ======================================================================
# 1. Схема v2 (B6): разбор, обёртка, размерности, строгость ключей
# ======================================================================
class TestSchemaV2Parsing:

    def test_dims_and_axis_names(self):
        spec = _pvc()
        assert spec.schema_version == 2
        assert spec.q == 19
        assert spec.dim_z == 16                      # 22 → 16 (B2)
        assert spec.z_names == Z_AXES_16
        # производные члены не имеют z-оси, но остаются компонентами
        for derived in ("PVC_71", "Chalk_1T", "CPE_135A", "PMPlus_8",
                        "PF711", "OPE"):
            assert derived not in spec.z_names
            assert derived in spec.component_names

    def test_wrapper_and_bare_list_same_spec(self):
        assert (PhrSpec.from_dicts(PVC_V2).spec_hash()
                == PhrSpec.from_dicts(PVC_V2_NODES).spec_hash())

    def test_roundtrip_to_dicts(self):
        spec = _pvc()
        d = spec.to_dicts()
        again = PhrSpec.from_dicts(d)
        assert again.to_dicts() == d
        assert again.spec_hash() == spec.spec_hash()
        assert again.schema_version == 2
        # v2-сериализация — role-ключи, mode отсутствует
        assert all("role" in nd and "mode" not in nd for nd in d)

    def test_wrapper_validation(self):
        with pytest.raises(ValueError, match="не поддерживается"):
            PhrSpec.from_dicts({"spec_version": 1, "nodes": SMALL_V2})
        with pytest.raises(ValueError, match="Неизвестные ключи обёртки"):
            PhrSpec.from_dicts({"spec_version": 2, "nodes": SMALL_V2,
                                "extra": 1})
        with pytest.raises(ValueError, match="списком узлов"):
            PhrSpec.from_dicts({"spec_version": 2, "nodes": "oops"})

    def test_mixed_mode_and_role_rejected(self):
        mixed = [dict(SMALL_V2[0]),
                 {"name": "f", "mode": "absolute", "lo": 1.0, "hi": 2.0}]
        with pytest.raises(ValueError, match="Смешаны схемы"):
            PhrSpec.from_dicts(mixed)

    def test_unknown_role_and_missing_keys(self):
        with pytest.raises(ValueError, match="неизвестная роль"):
            PhrSpec.from_dicts([{"name": "x", "role": "SHARE"}])
        with pytest.raises(ValueError, match="нет обязательных ключей"):
            PhrSpec.from_dicts(
                [{"name": "x", "role": "SHARE_FREE", "group": "G"}])

    def test_legacy_keys_in_v2_node_rejected(self):
        bad = [dict(d) for d in SMALL_V2]
        bad[2] = dict(bad[2], lo=0.2)                # legacy-ключ 'lo'
        with pytest.raises(ValueError, match="не входят в схему v2"):
            PhrSpec.from_dicts(bad)

    def test_cap_to_must_be_list(self):
        with pytest.raises(ValueError, match="непустой СПИСОК"):
            PhrSpec.from_dicts([
                {"name": "a", "role": "ABSOLUTE", "range": [1.0, 2.0]},
                {"name": "u", "role": "ABSOLUTE_CAPPED",
                 "range": [0.1, 0.3], "cap_to": "a", "cap_ratio": 0.1},
            ])

    def test_members_exact_match_required(self):
        swapped = [dict(d) for d in SMALL_V2]
        swapped[1] = dict(swapped[1], members=["Y", "X"])   # порядок другой
        with pytest.raises(ValueError, match="не совпадают"):
            PhrSpec.from_dicts(swapped)
        extra = [dict(d) for d in SMALL_V2]
        extra[1] = dict(extra[1], members=["X", "Y", "Z"])  # лишний член
        with pytest.raises(ValueError, match="не совпадают"):
            PhrSpec.from_dicts(extra)

    def test_group_parent_must_declare_members(self):
        bad = [dict(d) for d in SMALL_V2]
        bad[1] = {"name": "G", "role": "ABSOLUTE", "range": [10.0, 20.0]}
        with pytest.raises(ValueError, match="нет 'members'"):
            PhrSpec.from_dicts(bad)


# ======================================================================
# 2. B8: closure/fixed задаются БЕЗ range — ошибка, не тихое игнорирование
# ======================================================================
class TestB8ClosureWithoutRange:

    def test_closure_with_share_range_rejected(self):
        bad = [dict(d) for d in SMALL_V2]
        bad[3] = dict(bad[3], share_range=[0.2, 0.8])
        with pytest.raises(ValueError, match="ПРОИЗВОДНЫЙ"):
            PhrSpec.from_dicts(bad)

    def test_closure_with_range_rejected(self):
        bad = [dict(d) for d in SMALL_V2]
        bad[3] = dict(bad[3], range=[2.0, 16.0])
        with pytest.raises(ValueError, match="ПРОИЗВОДНЫЙ"):
            PhrSpec.from_dicts(bad)

    def test_fixed_with_range_rejected(self):
        bad = [dict(d) for d in SMALL_V2]
        bad[0] = dict(bad[0], range=[90.0, 110.0])
        with pytest.raises(ValueError, match="задаётся ключом 'value'"):
            PhrSpec.from_dicts(bad)

    def test_group_total_fixed_with_range_rejected(self):
        nodes = [
            {"name": "R", "role": "GROUP_TOTAL_FIXED", "value": 100.0,
             "members": ["a", "b"], "range": [90.0, 110.0]},
            {"name": "a", "role": "SHARE_FREE", "group": "R",
             "share_range": [0.3, 1.0]},
            {"name": "b", "role": "SHARE_CLOSURE", "group": "R"},
        ]
        with pytest.raises(ValueError, match="задаётся ключом 'value'"):
            PhrSpec.from_dicts(nodes)

    def test_direct_node_closure_with_bounds_rejected(self):
        nodes = [
            PhrNode("base", "fixed", value=100.0),
            PhrNode("G", "absolute", lo=10.0, hi=20.0),
            PhrNode("X", "share_free", lo=0.2, hi=0.8, ref="G"),
            PhrNode("Y", "share_closure", lo=0.1, hi=0.9, ref="G"),
        ]
        with pytest.raises(ValueError, match="диапазон доли не задаётся"):
            PhrSpec(nodes)


# ======================================================================
# 3. B2: инварианты состава групп (C1/C2/C5)
# ======================================================================
class TestGroupComposition:

    def _spec_with_roles(self, role_x: str, role_y: str):
        nodes = [dict(d) for d in SMALL_V2]
        nodes[2]["role"] = role_x
        nodes[3]["role"] = role_y
        if role_y != "SHARE_CLOSURE":
            nodes[3]["share_range"] = [0.2, 0.8]
        if role_x == "SHARE_CLOSURE":
            nodes[2].pop("share_range", None)
        return nodes

    def test_k2_requires_exactly_one_closure_and_one_free(self):
        with pytest.raises(ValueError, match="РОВНО один"):
            PhrSpec.from_dicts(self._spec_with_roles("SHARE_FREE",
                                                     "SHARE_FREE"))
        with pytest.raises(ValueError, match="РОВНО один"):
            PhrSpec.from_dicts(self._spec_with_roles("SHARE_CLOSURE",
                                                     "SHARE_CLOSURE"))

    def test_k2_simplex_rejected(self):
        with pytest.raises(ValueError, match="SHARE_SIMPLEX допустим только"):
            PhrSpec.from_dicts(self._spec_with_roles("SHARE_SIMPLEX",
                                                     "SHARE_SIMPLEX"))

    def test_k3_closure_rejected(self):
        nodes = [
            {"name": "L", "role": "GROUP_TOTAL", "range": [1.0, 2.0],
             "members": ["a", "b", "c"]},
            {"name": "a", "role": "SHARE_SIMPLEX", "group": "L",
             "share_range": [0.3, 0.7]},
            {"name": "b", "role": "SHARE_SIMPLEX", "group": "L",
             "share_range": [0.1, 0.6]},
            {"name": "c", "role": "SHARE_CLOSURE", "group": "L"},
        ]
        with pytest.raises(ValueError, match="все члены должны"):
            PhrSpec.from_dicts(nodes)

    def test_single_member_group_rejected(self):
        nodes = [
            {"name": "base", "role": "FIXED", "value": 100.0},
            {"name": "G", "role": "GROUP_TOTAL", "range": [10.0, 20.0],
             "members": ["X"]},
            {"name": "X", "role": "SHARE_FREE", "group": "G",
             "share_range": [0.2, 1.0]},
        ]
        with pytest.raises(ValueError, match="из одного"):
            PhrSpec.from_dicts(nodes)

    def test_mixing_legacy_and_new_in_one_group_rejected(self):
        nodes = [
            PhrNode("base", "fixed", value=100.0),
            PhrNode("G", "absolute", lo=10.0, hi=20.0),
            PhrNode("X", "share_free", lo=0.2, hi=0.8, ref="G"),
            PhrNode("Y", "share_of", lo=0.2, hi=0.8, ref="G"),
        ]
        with pytest.raises(ValueError, match="смешаны legacy"):
            PhrSpec(nodes)

    def test_v2_spec_rejects_legacy_share_of_group(self):
        # смесь групп РАЗНЫХ схем в одной спеке: новая группа тянет схему v2,
        # а v2 не поддерживает legacy share_of — явная ошибка
        nodes = [
            PhrNode("base", "fixed", value=100.0),
            PhrNode("G", "absolute", lo=10.0, hi=20.0),
            PhrNode("X", "share_free", lo=0.2, hi=0.8, ref="G"),
            PhrNode("Y", "share_closure", ref="G"),
            PhrNode("H", "absolute", lo=1.0, hi=2.0),
            PhrNode("u", "share_of", lo=0.2, hi=0.8, ref="H"),
            PhrNode("v", "share_of", lo=0.2, hi=0.8, ref="H"),
        ]
        with pytest.raises(ValueError, match="share_of"):
            PhrSpec(nodes)

    def test_c5_share_box_nonstrict(self):
        # впритык (0.60 = 1 − 0.40) — НЕ ошибка (референсная LUB-группа)
        assert _pvc().dim_z == 16
        # а вот φᵁ=0.65 > 1 − 0.40 = 0.60 — ошибка C5
        nodes = [
            {"name": "L", "role": "GROUP_TOTAL", "range": [1.0, 2.0],
             "members": ["a", "b", "c"]},
            {"name": "a", "role": "SHARE_SIMPLEX", "group": "L",
             "share_range": [0.3, 0.7]},
            {"name": "b", "role": "SHARE_SIMPLEX", "group": "L",
             "share_range": [0.1, 0.6]},
            {"name": "c", "role": "SHARE_SIMPLEX", "group": "L",
             "share_range": [0.1, 0.65]},
        ]
        with pytest.raises(ValueError, match="share-бокс несовместен"):
            PhrSpec.from_dicts(nodes)

    def test_group_total_with_cap_rejected(self):
        nodes = [
            PhrNode("base", "fixed", value=100.0),
            PhrNode("G", "absolute", lo=1.0, hi=20.0,
                    cap_refs=("base",), cap_ratio=0.2),
            PhrNode("X", "share_free", lo=0.2, hi=0.8, ref="G"),
            PhrNode("Y", "share_closure", ref="G"),
        ]
        with pytest.raises(ValueError, match="не может иметь cap_to"):
            PhrSpec(nodes)


# ======================================================================
# 4. Геометрия: closure вне z — rank(Z) = dim_z (сам смысл B2)
# ======================================================================
class TestGeometry:

    def test_rank_equals_dim(self):
        spec = _pvc()
        Z = spec.sample_z(400, seed=0)
        assert Z.shape == (400, 16)
        assert np.linalg.matrix_rank(Z - Z.mean(axis=0)) == 16

    def test_legacy_flat_control_is_rank_deficient(self):
        """Контроль-мотивация B2: у legacy-пары (φ, 1−φ) rank < dim."""
        legacy = PhrSpec.from_dicts(LEGACY_FLAT)
        assert legacy.schema_version == 1
        # 6 осей: DINP + FILLER.total + 4 share-оси (по 2 в каждой группе)
        assert legacy.dim_z == 6
        Z = legacy.sample_z(300, seed=1)
        assert (np.linalg.matrix_rank(Z - Z.mean(axis=0), tol=1e-8)
                == legacy.dim_z - 2)        # по одной зависимости на группу

    def test_decode_group_sums_and_bounds(self):
        spec = _pvc()
        Z = spec.sample_z(300, seed=2)
        P = spec.decode(Z)
        col = {nm: j for j, nm in enumerate(spec.component_names)}
        zc = {nm: j for j, nm in enumerate(spec.z_names)}
        # k=2: закрытие точное, производная доля в производных границах
        assert np.allclose(P[:, col["PVC_67"]] + P[:, col["PVC_71"]], 100.0)
        share_71 = P[:, col["PVC_71"]] / 100.0
        assert np.all(share_71 >= -1e-12) and np.all(share_71 <= 0.70 + 1e-12)
        filler = Z[:, zc["FILLER.total"]]
        assert np.allclose(P[:, col["Chalk_95T"]] + P[:, col["Chalk_1T"]],
                           filler)
        # k=3 (simplex): Σ долей = 1, производный OPE в заявленных границах
        lub = Z[:, zc["LUB.total"]]
        s = np.column_stack([P[:, col[m]] / lub
                             for m in ("DL_60", "AKLUB_K_435", "OPE")])
        assert np.allclose(s.sum(axis=1), 1.0)
        assert np.all(s[:, 2] >= 0.10 - 1e-9)
        assert np.all(s[:, 2] <= 0.60 + 1e-9)

    def test_encode_roundtrip_and_rejects_out_of_closure(self):
        spec = _pvc()
        Z = spec.sample_z(50, seed=3)
        P = spec.decode(Z)
        assert np.allclose(spec.encode(P), Z, atol=1e-9)
        # рецепт с φ_PVC_67 = 0.2 < 0.30 — вне границ свободной доли
        p_bad = P[0].copy()
        col = {nm: j for j, nm in enumerate(spec.component_names)}
        p_bad[col["PVC_67"]] = 20.0
        p_bad[col["PVC_71"]] = 80.0
        with pytest.raises(ValueError, match="вне границ"):
            spec.encode(p_bad)

    def test_clip_z_idempotent_and_projects(self):
        spec = _pvc()
        Z = spec.sample_z(100, seed=4)
        assert np.allclose(spec.clip_z(Z), Z)          # валидные не двигаются
        rng = np.random.default_rng(5)
        lo, hi = spec.z_bounds()
        Zbad = rng.uniform(lo - 1.0, hi + 1.0, size=(100, spec.dim_z))
        Zc = spec.clip_z(Zbad)
        assert np.allclose(spec.clip_z(Zc), Zc)        # проекция идемпотентна
        P = spec.decode(Zc)                            # и допустима: Σ группы
        col = {nm: j for j, nm in enumerate(spec.component_names)}
        assert np.allclose(P[:, col["PVC_67"]] + P[:, col["PVC_71"]], 100.0)
        share_free = P[:, col["PVC_67"]] / 100.0
        assert np.all(share_free >= 0.30 - 1e-9)
        assert np.all(share_free <= 1.00 + 1e-9)

    def test_z_bounds_shapes_and_values(self):
        spec = _pvc()
        lo, hi = spec.z_bounds()
        assert lo.shape == (16,) and hi.shape == (16,)
        j = spec.z_names.index("PVC_67")
        assert (lo[j], hi[j]) == (0.30, 1.00)

    def test_sample_candidates_simplex(self):
        X = _pvc().sample_candidates(64, seed=6)
        assert X.shape == (64, 19)
        assert np.allclose(X.sum(axis=1), 1.0)

    def test_phr_intervals_include_derived(self):
        iv = _pvc().phr_intervals()
        assert iv["PVC_71"] == pytest.approx((0.0, 70.0))
        lo, hi = iv["PMPlus_8"]
        assert lo == pytest.approx(0.40 * 0.20)
        assert hi == pytest.approx(0.75 * 1.20)

    def test_share_base_bounds_accessor(self):
        spec = _pvc()
        assert spec.share_base_bounds("PVC_71") == pytest.approx((0.0, 0.70))
        assert spec.share_base_bounds("DL_60") == (0.30, 0.70)
        with pytest.raises(ValueError, match="не является share-узлом"):
            spec.share_base_bounds("DINP")


# ======================================================================
# 5. min_phr/max_phr (iter45/B1) работают и на новых ролях
# ======================================================================
class TestPhrLimitsOnNewRoles:

    def test_closure_min_phr_enforced_in_sampling(self):
        spec = _small(Y={"min_phr": 5.0})
        Z = spec.sample_z(300, seed=7)
        P = spec.decode(Z)
        j = spec.component_names.index("Y")
        assert np.all(P[:, j] >= 5.0 - 1e-9)

    def test_encode_rejects_below_min_phr(self):
        spec = _small(Y={"min_phr": 5.0})
        # T=20, φ_Y=0.2 → p_Y=4 < 5 — ниже техминимума
        with pytest.raises(ValueError, match="ниже технологического"):
            spec.encode([100.0, 16.0, 4.0])

    def test_share_bounds_at_total_on_new_group(self):
        spec = _small(Y={"min_phr": 5.0})
        lo, hi = spec.share_bounds_at_total("G", 10.0)
        # члены в порядке спеки: X, Y; φ_Y ≥ 5/10 = 0.5 ⇒ φ_X ≤ 0.5
        assert lo[1] == pytest.approx(0.5)
        assert hi[0] == pytest.approx(0.5)


# ======================================================================
# 6. scale (B6): схема/хеш принимают; лог-сэмплинг — iter47/B5
# ======================================================================
class TestScale:

    def _log_spec(self):
        return PhrSpec.from_dicts([
            {"name": "base", "role": "FIXED", "value": 100.0},
            {"name": "TiO2", "role": "ABSOLUTE", "range": [0.3, 8.0],
             "scale": "log"},
        ])

    def test_scale_parses_and_roundtrips_in_hash(self):
        spec = self._log_spec()
        d = spec.to_dicts()
        assert next(x for x in d if x["name"] == "TiO2")["scale"] == "log"
        assert PhrSpec.from_dicts(d).spec_hash() == spec.spec_hash()
        linear = PhrSpec.from_dicts([
            {"name": "base", "role": "FIXED", "value": 100.0},
            {"name": "TiO2", "role": "ABSOLUTE", "range": [0.3, 8.0]},
        ])
        assert linear.spec_hash() != spec.spec_hash()  # scale — часть геометрии

    def test_geometry_ops_accept_log_since_iter47(self):
        # iter47/B5: гейт снят — геометрия log-осей работает
        # (подробные инварианты — test_iteration47_log_sampling.py)
        spec = self._log_spec()
        Z = spec.sample_z(10, seed=0)
        assert Z.shape == (10, spec.dim_z)
        lo, hi = spec.z_bounds()
        assert np.all(lo <= hi)
        Zc = spec.clip_z(Z)
        assert np.allclose(Zc, Z)              # валидные точки не двигаются

    def test_scale_validation(self):
        with pytest.raises(ValueError, match="неизвестная шкала"):
            PhrSpec.from_dicts([
                {"name": "base", "role": "FIXED", "value": 100.0},
                {"name": "a", "role": "ABSOLUTE", "range": [1.0, 2.0],
                 "scale": "log10"}])
        with pytest.raises(ValueError, match="lo > 0"):
            PhrSpec.from_dicts([
                {"name": "base", "role": "FIXED", "value": 100.0},
                {"name": "a", "role": "ABSOLUTE", "range": [0.0, 2.0],
                 "scale": "log"}])
        # scale вне ABSOLUTE* — лишний ключ схемы (RATIO_TO его не имеет)
        with pytest.raises(ValueError, match="не входят в схему v2"):
            PhrSpec.from_dicts([
                {"name": "base", "role": "FIXED", "value": 100.0},
                {"name": "r", "role": "RATIO_TO", "reference": "base",
                 "range": [0.1, 0.2], "scale": "log"}])


# ======================================================================
# 7. Персистентность (campaign_state) и legacy-совместимость
# ======================================================================
class TestPersistenceAndLegacy:

    def _v2_runner(self):
        spec = _small()
        lo, hi = spec.fraction_bounds()
        runner = build_setup_runner(
            mixture_names=list(spec.component_names), process_names=["T"],
            process_lower=[0.0], process_upper=[1.0],
            response_names=["strength"],
            mixture_lower=lo.tolist(), mixture_upper=hi.tolist(), seed=3)
        runner.set_phr_spec(spec)
        return runner

    def test_v2_spec_survives_state_roundtrip(self):
        r0 = self._v2_runner()
        h0 = r0.phr_spec.spec_hash()
        r1 = cst.runner_from_state(cst.runner_to_state(r0))
        assert r1.phr_spec is not None
        assert r1.phr_spec.schema_version == 2
        assert r1.phr_spec.spec_hash() == h0

    def test_legacy_spec_serialization_unchanged(self):
        legacy = PhrSpec.from_dicts(LEGACY_FLAT)
        assert legacy.schema_version == 1
        d = legacy.to_dicts()
        assert all("mode" in nd and "role" not in nd for nd in d)
        assert PhrSpec.from_dicts(d).spec_hash() == legacy.spec_hash()


# ======================================================================
# 8. UI-хелперы: JSON-обёртка, summary с производными границами, tree-guard
# ======================================================================
class TestUIHelpers:

    def test_parse_phr_spec_json_accepts_wrapper(self):
        spec = ui.parse_phr_spec_json(json.dumps(PVC_V2, ensure_ascii=False))
        assert spec.dim_z == 16
        assert spec.spec_hash() == _pvc().spec_hash()

    def test_summary_shows_derived_closure_bounds(self):
        df = ui.phr_spec_summary_dataframe(_pvc())
        row = df[df["узел"] == "PVC_71"].iloc[0]
        assert row["режим"] == "share_closure"
        assert row["lo"] == pytest.approx(0.0)
        assert row["hi"] == pytest.approx(0.70)

    def test_phr_tree_from_spec_rejects_v2(self):
        with pytest.raises(ValueError, match="legacy-схему v1"):
            ui.phr_tree_from_spec(_pvc())

    def test_setup_prefill_v2_uses_json_channel_only(self):
        runner = TestPersistenceAndLegacy()._v2_runner()
        out = ui.setup_prefill_from_runner(runner)
        assert "setup_phr_tree" not in out          # дерево — только v1
        spec2 = ui.parse_phr_spec_json(out["setup_phr_json"])
        assert spec2.spec_hash() == runner.phr_spec.spec_hash()