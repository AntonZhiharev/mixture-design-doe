# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 40 / UI_REVISION_SPEC iter40 — персистентность политики кампании.

До iter40 save/load кампании МОЛЧА терял (против A0.6 и CAMPAIGN_SPEC_PVC §3
«записать ДО первого замера, задним числом не восстанавливается»):

  * ``phr_spec`` — decode-геометрию кампании (iter33/38): после загрузки
    сэмплер кандидатов и optimize_xbest откатывались на бокс долей;
  * ``campaign_label`` — метку кампании (iter37 п.2): новые точки после
    загрузки переставали получать ``campaign``/``spec_hash`` в origin_tag;
  * ``preflight_pairs`` — обязательные 2D-пары (iter37 п.4): pair-coverage
    гейт preflight после загрузки всегда был пуст.

Проверяем round-trip всех трёх политик через runner_to_state/runner_from_state
и файловый save/load (JSON-native), обратную совместимость старых сейвов и
сквозной сценарий «save → load → добор»: точка, добавленная ПОСЛЕ загрузки,
несёт ``spec_hash``/``campaign`` в origin_tag.
"""
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.apps import campaign_state as cst
from src.apps.campaign import CampaignController
from src.apps.campaign_ui import build_setup_runner
from src.design.phr_sampler import PhrSpec

# Единый источник референсной 19-компонентной спеки PVC (iter35):
# дублировать нельзя — golden-хеш обязан жить в одном месте.
from tests.unit.test_iteration35_phr_spec_campaign import (COMPONENTS,
                                                           RECIPE_DICTS)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Golden-хеш референсной спеки (CAMPAIGN_SPEC_PVC §3, состояние 05.08.2026).
REFERENCE_HASH = ("049d0e35b43070f8322739600fadac7a"
                  "7a820f70c24e71546126979e34360450")

# Малая спека для сквозного сценария (дорогая 19-компонентная не нужна):
# resin=100 (share-группа A/B) + absolute filler.
SMALL_DICTS = [
    {"name": "resin", "mode": "fixed", "value": 100.0},
    {"name": "A", "mode": "share_of", "of": "resin", "lo": 0.30, "hi": 1.00},
    {"name": "B", "mode": "share_of", "of": "resin", "lo": 0.00, "hi": 0.70},
    {"name": "filler", "mode": "absolute", "lo": 5.0, "hi": 20.0},
]


def _pvc_runner():
    """Раннер PVC-кампании: 19 компонентов из спеки, полная политика, без точек."""
    spec = PhrSpec.from_dicts(RECIPE_DICTS)
    lo, hi = spec.fraction_bounds()
    runner = build_setup_runner(
        mixture_names=list(COMPONENTS), process_names=["T"],
        process_lower=[150.0], process_upper=[200.0],
        response_names=["gloss"],
        mixture_lower=lo.tolist(), mixture_upper=hi.tolist(), seed=1)
    runner.set_phr_spec(spec)
    runner.set_campaign_label("PVC-профиль-2026")
    runner.set_preflight_pairs([
        ("UV_CSFCP", "TiO2_BLR895"),
        ("DINP", "TiO2_BLR895"),
        ("T", ["PMPlus_8", "DL_531"]),          # ось-сумма Σ_ACR
    ])
    return runner


def _small_runner():
    """Раннер малой спеки (сквозной сценарий с добором точек)."""
    spec = PhrSpec.from_dicts(SMALL_DICTS)
    lo, hi = spec.fraction_bounds()
    runner = build_setup_runner(
        mixture_names=list(spec.component_names), process_names=["T"],
        process_lower=[0.0], process_upper=[1.0],
        response_names=["strength"],
        mixture_lower=lo.tolist(), mixture_upper=hi.tolist(), seed=2)
    runner.set_phr_spec(spec)
    runner.set_campaign_label("small-campaign")
    runner.set_preflight_pairs([("A", "filler")])
    return runner


# ======================================================================
# 1. Round-trip phr-спеки: геометрия (hash) выживает бит-в-бит
# ======================================================================
class TestPhrSpecRoundtrip:

    def test_spec_hash_survives_state_roundtrip(self):
        r0 = _pvc_runner()
        assert r0.phr_spec.spec_hash() == REFERENCE_HASH
        r1 = cst.runner_from_state(cst.runner_to_state(r0))
        assert r1.phr_spec is not None
        assert r1.phr_spec.spec_hash() == REFERENCE_HASH

    def test_spec_survives_file_save_load(self, tmp_path):
        """Файловый цикл: to_dicts обязан быть JSON-native (json.dumps)."""
        r0 = _pvc_runner()
        cst.save_campaign(r0, str(tmp_path), "pvc")
        r1 = cst.load_campaign(str(tmp_path), "pvc")
        assert r1.phr_spec.spec_hash() == REFERENCE_HASH
        assert list(r1.phr_spec.component_names) == list(COMPONENTS)

    def test_no_spec_roundtrips_as_none(self):
        runner = build_setup_runner(
            mixture_names=["A", "B", "C"], process_names=["T"],
            process_lower=[0.0], process_upper=[1.0],
            response_names=["strength"], seed=1)
        r1 = cst.runner_from_state(cst.runner_to_state(runner))
        assert r1.phr_spec is None


# ======================================================================
# 2. Round-trip метки кампании и обязательных 2D-пар
# ======================================================================
class TestLabelAndPairsRoundtrip:

    def test_campaign_label_roundtrip(self):
        r1 = cst.runner_from_state(cst.runner_to_state(_pvc_runner()))
        assert r1.campaign_label == "PVC-профиль-2026"

    def test_preflight_pairs_roundtrip_normalized(self):
        """Пары восстанавливаются в НОРМАЛИЗОВАННОМ виде (списки имён),
        включая ось-сумму ["PMPlus_8", "DL_531"] и process-ось T."""
        r0 = _pvc_runner()
        r1 = cst.runner_from_state(cst.runner_to_state(r0))
        assert [tuple(map(tuple, p)) for p in r1.preflight_pairs] == \
               [tuple(map(tuple, p)) for p in r0.preflight_pairs]
        assert (["T"], ["PMPlus_8", "DL_531"]) in [
            (list(a), list(b)) for a, b in r1.preflight_pairs]

    def test_empty_label_and_pairs_stay_empty(self):
        runner = build_setup_runner(
            mixture_names=["A", "B", "C"], process_names=["T"],
            process_lower=[0.0], process_upper=[1.0],
            response_names=["strength"], seed=1)
        r1 = cst.runner_from_state(cst.runner_to_state(runner))
        assert r1.campaign_label == ""
        assert r1.preflight_pairs == []


# ======================================================================
# 3. Обратная совместимость: старый сейв без новых ключей
# ======================================================================
class TestBackwardCompatibility:

    def test_old_save_without_policy_keys_loads(self):
        r0 = _pvc_runner()
        state = cst.runner_to_state(r0)
        for key in ("phr_spec", "campaign_label", "preflight_pairs"):
            state["runner"].pop(key)                  # сейв «до iter40»
        r1 = cst.runner_from_state(state)
        assert r1.phr_spec is None
        assert r1.campaign_label == ""
        assert r1.preflight_pairs == []


# ======================================================================
# 4. Сквозной сценарий: save → load → добор (метаданные точки живы)
# ======================================================================
class TestEndToEndAfterLoad:

    def test_point_added_after_load_carries_metadata(self, tmp_path):
        r0 = _small_runner()
        expected_hash = r0.phr_spec.spec_hash()
        cst.save_campaign(r0, str(tmp_path), "small")

        r1 = cst.load_campaign(str(tmp_path), "small")
        ctrl = CampaignController(r1)
        X = np.asarray(ctrl.propose_seed(10, seed=3), float)
        Y = np.vstack([r1._measure(np.asarray(x, float)) for x in X])
        ctrl.commit_seed(X, Y)

        assert len(r1.points) == 10
        for p in r1.points:
            assert p.origin_tag.get("campaign") == "small-campaign"
            assert p.origin_tag.get("spec_hash") == expected_hash

    def test_preflight_after_load_sees_pairs(self, tmp_path):
        r0 = _small_runner()
        cst.save_campaign(r0, str(tmp_path), "small")
        r1 = cst.load_campaign(str(tmp_path), "small")

        X = np.asarray(r1.propose_seed(10, seed=3), float)
        report = r1.preflight(X)
        assert len(report.pair_coverage) == 1     # гейт пары (A, filler) активен