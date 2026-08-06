# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 48 — шаг B4 ревизии контракта phr-спеки: ``group_order``.

``group_order`` — приоритет GROUP_TOTAL-групп кампании (C2,
CAMPAIGN_SPEC_PVC §1: FILLER → SOFT → ACR → LUB → STAB):

  * ТОЧНАЯ перестановка множества групп с ролью ``GROUP_TOTAL``:
    пропуски / лишние имена / дубли / не-тоталы — ошибка валидации,
    не тихое игнорирование;
  * ``GROUP_TOTAL_FIXED`` (RESIN) исключается: тотал детерминирован,
    стратифицировать нечего;
  * входит в ``to_dicts()``/``spec_hash()``: перестановка групп меняет
    отпечаток — без этого хеш не воспроизводит контракт кампании;
  * спеки БЕЗ ``group_order`` сериализуются байт-в-байт как до iter48
    (плоский список, хеши прежних спек не «уехали»);
  * только схема v2 (в legacy v1 роли GROUP_TOTAL нет).

Честная граница (зафиксирована в докстринге модуля): в phr-пути тоталы
групп — НЕЗАВИСИМЫЕ absolute-оси, каждая получает точную равномерную
маргиналь независимо от порядка — меру phr-сэмплера ``group_order``
не меняет (порядко-зависимость KS≈0.019/0.38 — свойство fraction-space
группового сэмплера iter31). Поэтому здесь проверяется и инвариантность
геометрии к порядку (z_bounds/phr_intervals не зависят от group_order).

Golden-хеш v2-спеки в тесты НЕ закладывается (решение C4) — проверяются
round-trip-инварианты и чувствительность хеша к порядку.
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
# Малая v2-спека: fixed-группа RESIN (исключается из group_order) +
# две GROUP_TOTAL-группы FILLER и SOFT (подлежат перечислению).
# ----------------------------------------------------------------------
NODES = [
    {"name": "RESIN", "role": "GROUP_TOTAL_FIXED", "value": 100.0,
     "members": ["R1", "R2"]},
    {"name": "R1", "role": "SHARE_FREE", "group": "RESIN",
     "share_range": [0.30, 1.00]},
    {"name": "R2", "role": "SHARE_CLOSURE", "group": "RESIN"},
    {"name": "FILLER", "role": "GROUP_TOTAL", "range": [5.0, 25.0],
     "members": ["F1", "F2"]},
    {"name": "F1", "role": "SHARE_FREE", "group": "FILLER",
     "share_range": [0.30, 1.00]},
    {"name": "F2", "role": "SHARE_CLOSURE", "group": "FILLER"},
    {"name": "SOFT", "role": "GROUP_TOTAL", "range": [5.0, 15.0],
     "members": ["S1", "S2"]},
    {"name": "S1", "role": "SHARE_FREE", "group": "SOFT",
     "share_range": [0.00, 0.70]},
    {"name": "S2", "role": "SHARE_CLOSURE", "group": "SOFT"},
]

ORDER = ["FILLER", "SOFT"]


def _wrap(order):
    return {"spec_version": 2, "group_order": list(order), "nodes": NODES}


def _spec(order=ORDER):
    return PhrSpec.from_dicts(_wrap(order))


# ======================================================================
# 1. Модель: разбор, доступ, инвариантность геометрии к порядку
# ======================================================================
class TestGroupOrderModel:

    def test_parsed_and_exposed(self):
        spec = _spec()
        assert spec.group_order == ORDER
        assert spec.schema_version == 2

    def test_unset_is_empty(self):
        spec = PhrSpec.from_dicts(NODES)
        assert spec.group_order == []
        # обёртка без ключа — тоже «не задан»
        spec2 = PhrSpec.from_dicts({"spec_version": 2, "nodes": NODES})
        assert spec2.group_order == []

    def test_geometry_independent_of_order(self):
        """group_order — контракт кампании, НЕ переключатель геометрии:
        в phr-пути тоталы независимы, границы/интервалы не зависят от
        порядка (перестановка меняет ТОЛЬКО отпечаток)."""
        a = _spec(["FILLER", "SOFT"])
        b = _spec(["SOFT", "FILLER"])
        c = PhrSpec.from_dicts(NODES)                # без group_order
        for s in (b, c):
            assert s.z_names == a.z_names
            assert s.component_names == a.component_names
            np.testing.assert_allclose(np.c_[s.z_bounds()],
                                       np.c_[a.z_bounds()])
            assert s.phr_intervals() == a.phr_intervals()

    def test_sampling_works_with_group_order(self):
        spec = _spec()
        X = spec.sample_candidates(64, seed=0)
        assert X.shape == (64, spec.q)
        np.testing.assert_allclose(X.sum(axis=1), 1.0, atol=1e-12)


# ======================================================================
# 2. Валидация: ТОЧНАЯ перестановка множества GROUP_TOTAL (C2)
# ======================================================================
class TestGroupOrderValidation:

    def test_missing_group_rejected(self):
        with pytest.raises(ValueError, match="не перечислены группы"):
            _spec(["FILLER"])

    def test_unknown_name_rejected(self):
        with pytest.raises(ValueError, match="не найден среди узлов"):
            _spec(["FILLER", "SOFT", "XXX"])

    def test_non_total_node_rejected(self):
        with pytest.raises(ValueError, match="не является тоталом"):
            _spec(["F1", "SOFT"])

    def test_fixed_total_rejected(self):
        with pytest.raises(ValueError, match="GROUP_TOTAL_FIXED"):
            _spec(["RESIN", "FILLER", "SOFT"])

    def test_duplicates_rejected(self):
        with pytest.raises(ValueError, match="дубли"):
            _spec(["FILLER", "FILLER", "SOFT"])

    def test_legacy_v1_rejected(self):
        legacy = [
            PhrNode(name="total", mode="absolute", lo=5.0, hi=25.0),
            PhrNode(name="a", mode="share_of", ref="total", lo=0.3, hi=1.0),
            PhrNode(name="b", mode="share_of", ref="total", lo=0.0, hi=0.7),
            PhrNode(name="base", mode="fixed", value=100.0),
        ]
        with pytest.raises(ValueError, match="схемой v2"):
            PhrSpec(legacy, group_order=["total"])

    def test_wrapper_wrong_type_rejected(self):
        with pytest.raises(ValueError, match="СПИСКОМ"):
            PhrSpec.from_dicts({"spec_version": 2, "nodes": NODES,
                                "group_order": "FILLER"})

    def test_wrapper_unknown_key_still_rejected(self):
        with pytest.raises(ValueError, match="Неизвестные ключи обёртки"):
            PhrSpec.from_dicts({"spec_version": 2, "nodes": NODES,
                                "group_ordr": ORDER})


# ======================================================================
# 3. Сериализация и хеш: round-trip, чувствительность к порядку,
#    байт-в-байт совместимость спек без group_order
# ======================================================================
class TestGroupOrderHash:

    def test_roundtrip_preserves_order_and_hash(self):
        spec = _spec()
        d = spec.to_dicts()
        assert isinstance(d, dict)                   # обёртка при заданном
        assert d["spec_version"] == 2
        assert d["group_order"] == ORDER
        again = PhrSpec.from_dicts(d)
        assert again.group_order == ORDER
        assert again.to_dicts() == d
        assert again.spec_hash() == spec.spec_hash()
        # и через JSON-текст (как в сейве/UI)
        again2 = PhrSpec.from_dicts(json.loads(json.dumps(d)))
        assert again2.spec_hash() == spec.spec_hash()

    def test_hash_differs_from_spec_without_order(self):
        assert _spec().spec_hash() != PhrSpec.from_dicts(NODES).spec_hash()

    def test_hash_sensitive_to_permutation(self):
        assert (_spec(["FILLER", "SOFT"]).spec_hash()
                != _spec(["SOFT", "FILLER"]).spec_hash())

    def test_unset_serialization_unchanged(self):
        """Спека без group_order сериализуется плоским СПИСКОМ —
        байт-в-байт как до iter48, хеши прежних спек не «уехали»."""
        spec = PhrSpec.from_dicts(NODES)
        d = spec.to_dicts()
        assert isinstance(d, list)
        assert all("role" in nd for nd in d)


# ======================================================================
# 4. Персистентность (campaign_state) и UI-канал (JSON)
# ======================================================================
class TestPersistenceAndUi:

    def _runner(self):
        spec = _spec()
        lo, hi = spec.fraction_bounds()
        runner = build_setup_runner(
            mixture_names=list(spec.component_names), process_names=["T"],
            process_lower=[0.0], process_upper=[1.0],
            response_names=["strength"],
            mixture_lower=lo.tolist(), mixture_upper=hi.tolist(), seed=3)
        runner.set_phr_spec(spec)
        return runner

    def test_state_roundtrip_preserves_group_order(self):
        r0 = self._runner()
        h0 = r0.phr_spec.spec_hash()
        r1 = cst.runner_from_state(cst.runner_to_state(r0))
        assert r1.phr_spec is not None
        assert r1.phr_spec.group_order == ORDER
        assert r1.phr_spec.spec_hash() == h0

    def test_parse_phr_spec_json_accepts_group_order(self):
        text = json.dumps(_wrap(ORDER), ensure_ascii=False)
        spec = ui.parse_phr_spec_json(text)
        assert spec.group_order == ORDER
        assert spec.spec_hash() == _spec().spec_hash()

    def test_setup_prefill_json_roundtrip(self):
        runner = self._runner()
        out = ui.setup_prefill_from_runner(runner)
        spec2 = ui.parse_phr_spec_json(out["setup_phr_json"])
        assert spec2.group_order == ORDER
        assert spec2.spec_hash() == runner.phr_spec.spec_hash()


# ======================================================================
# 5. Референсная PVC-спека кампании (_phr_spec_pvc.json — локальный файл
#    сетапа, в git не хранится: без него тест честно скипается)
# ======================================================================
class TestPvcCampaignSpec:

    def test_pvc_json_carries_campaign_order(self):
        try:
            f = open("_phr_spec_pvc.json", encoding="utf-8")
        except FileNotFoundError:
            pytest.skip("_phr_spec_pvc.json отсутствует (локальный файл "
                        "сетапа, вне git)")
        with f:
            spec = PhrSpec.from_dicts(json.load(f))
        # порядок кампании — ЗАФИКСИРОВАН (CAMPAIGN_SPEC_PVC §1)
        assert spec.group_order == ["FILLER.total", "SOFT.total",
                                    "ACR.total", "LUB.total", "STAB.total"]
        assert spec.dim_z == 16
        # RESIN.total (GROUP_TOTAL_FIXED) в group_order отсутствует
        assert "RESIN.total" not in spec.group_order
        # round-trip сохраняет отпечаток
        again = PhrSpec.from_dicts(spec.to_dicts())
        assert again.spec_hash() == spec.spec_hash()