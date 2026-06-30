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

from src.core.linalg import scheffe_term_indices
from src.apps.battle_preset import (battle_step6_snapshot,
                                     battle_step6_coef_by_property)
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
