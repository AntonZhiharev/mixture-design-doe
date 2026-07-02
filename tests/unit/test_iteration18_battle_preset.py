# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Тесты пресета STEP 6 (apps/battle_preset) и детерминированной истины.

Проверяем: (1) снимок конфигурации содержит известную истину под все свойства;
(2) векторы коэф. имеют правильную длину под cubic-модель; (3) лаборатория
детерминирована и совпадает с заданными линейными коэф. на чистых компонентах;
(4) истина переживает round-trip config_snapshot → from_snapshot.
"""
import numpy as np

import json

from src.core.linalg import scheffe_term_indices
from src.apps.battle_preset import (battle_step6_snapshot,
                                     battle_step6_coef_by_property,
                                     battle_step6_description,
                                     battle_step6_truth_readable)
from src.apps.pipeline_runner import PipelineConfig, PipelineRunner



def test_snapshot_has_known_truth_for_all_properties():
    snap = battle_step6_snapshot()
    props = snap["property_names"]
    assert props == ["strength", "gloss", "dry_time", "whiteStrength", "rho"]
    coef = snap["truth_coef_by_property"]
    assert set(coef) == set(props)
    # длина вектора = число термов cubic-модели для q=4
    p = len(scheffe_term_indices(4, "cubic"))
    for name in props:
        assert len(coef[name]) == p
    # цены и единица перенесены
    assert snap["cost_coeffs"] == [95.0, 200.0, 23.0, 315.0]
    assert snap["cost_unit"] == "усл.ед/кг"


def test_lab_is_deterministic_on_pure_components(tmp_path):
    snap = battle_step6_snapshot(noise_sd=0.0)
    cfg = PipelineConfig.from_snapshot(snap["name"], snap)
    runner = PipelineRunner(cfg, tmp_path / "battle")
    vertices = np.eye(4)  # чистые компоненты A,B,C,D

    # rho — чисто линейное свойство: на вершине A/B/C/D = линейный коэф.
    rho_idx = runner.property_names.index("rho")
    rho = runner.truth_multi.truths[rho_idx].true(vertices)
    np.testing.assert_allclose(rho, [0.7, 1.0, 1.7, 0.6], atol=1e-9)

    # strength линейные коэф. A,B,C,D = 6,10,2,2 (взаимодействия = 0 в вершине)
    s_idx = runner.property_names.index("strength")
    strength = runner.truth_multi.truths[s_idx].true(vertices)
    np.testing.assert_allclose(strength, [6.0, 10.0, 2.0, 2.0], atol=1e-9)


def test_truth_survives_config_roundtrip(tmp_path):
    snap = battle_step6_snapshot(noise_sd=0.0)
    cfg = PipelineConfig.from_snapshot(snap["name"], snap)
    runner = PipelineRunner(cfg, tmp_path / "battle")
    snap2 = runner.config_snapshot()
    assert snap2["truth_coef_by_property"] is not None
    assert (snap2["truth_coef_by_property"]["rho"]
            == battle_step6_coef_by_property()["rho"])

    # пересборка из второго снимка даёт ту же истину на вершинах
    cfg2 = PipelineConfig.from_snapshot("battle2", snap2)
    runner2 = PipelineRunner(cfg2, tmp_path / "battle2")
    v = np.eye(4)
    i = runner2.property_names.index("rho")
    np.testing.assert_allclose(runner2.truth_multi.truths[i].true(v),
                               [0.7, 1.0, 1.7, 0.6], atol=1e-9)


def test_truth_readable_matches_sparse_truth():
    """Читаемая раскладка истины отражает разреженные термы STEP 6."""
    readable = battle_step6_truth_readable()
    # rho — чисто линейное: 4 линейных терма, без взаимодействий
    rho = readable["rho"]
    assert {t["term"] for t in rho} == {"A", "B", "C", "D"}
    assert all(t["order"] == 1 for t in rho)
    assert {t["term"]: t["coef"] for t in rho} == {
        "A": 0.7, "B": 1.0, "C": 1.7, "D": 0.6}
    # strength несёт тройной терм A·B·C (order=3, coef=12)
    strength = {t["term"]: t for t in readable["strength"]}
    assert strength["A·B·C"]["order"] == 3
    assert strength["A·B·C"]["coef"] == 12.0
    assert strength["B·C"]["coef"] == 16.0


def test_description_is_json_serializable_and_complete():
    """Карточка для ассистента сериализуема и несёт ключевые поля набора."""
    desc = battle_step6_description()
    json.dumps(desc, ensure_ascii=False)  # сериализуемость без падений

    assert desc["button_label"] == "🧪 Заполнить тестовыми данными"
    # компоненты с ценами совпадают со снимком
    comps = {c["name"]: c["price"] for c in desc["components"]}
    assert comps == {"A": 95.0, "B": 200.0, "C": 23.0, "D": 315.0}
    # свойства перечислены с пояснениями
    prop_names = [p["name"] for p in desc["properties"]]
    assert prop_names == ["strength", "gloss", "dry_time", "whiteStrength", "rho"]
    # параметры лаборатории совпадают с дефолтами снимка
    snap = battle_step6_snapshot()
    assert desc["seed"] == snap["seed"]
    assert desc["noise_sd"] == snap["noise_sd"]
    assert desc["truth_model"] == snap["truth_model"]
    # известная истина включена в читаемом виде по каждому свойству
    terms = desc["known_truth"]["terms_by_property"]
    assert set(terms) == set(prop_names)
    assert terms["rho"] == battle_step6_truth_readable()["rho"]
    # порядок работы — непустой список шагов
    assert isinstance(desc["how_to_use"], list) and desc["how_to_use"]


