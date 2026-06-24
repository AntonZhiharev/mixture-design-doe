# Copyright 2026 AZH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Итерация 9, шаг 3a — мультиотклик в ядре (REBUILD_SPEC §12).

Проверяет `MultiSyntheticScheffe` и интеграцию `Y (n×P)` / `property_names` в
`PipelineRunner` (M2 собирает все свойства; save/load сохраняет Y и имена).
"""
import numpy as np
import pytest

from src.core.synthetic import SyntheticScheffe, MultiSyntheticScheffe
from src.apps.pipeline_runner import (PipelineConfig, PipelineRunner,
                                       list_projects)


# ---------------- MultiSyntheticScheffe ----------------
def test_multi_truth_shapes_and_property0_matches_single():
    q, props = 4, ["A", "B", "C"]
    lab = MultiSyntheticScheffe(q, props, model="quadratic",
                                noise_sd=0.0, seed=42)
    X = np.full((5, q), 1.0 / q)
    Y = lab.true(X)
    assert Y.shape == (5, 3)
    # свойство 0 совпадает с одно-откликовым SyntheticScheffe(seed=base)
    single = SyntheticScheffe(q, model="quadratic", noise_sd=0.0, seed=42)
    assert np.allclose(Y[:, 0], single.true(X))
    # свойства различаются (разные истины)
    assert not np.allclose(Y[:, 0], Y[:, 1])


def test_multi_truth_validates_empty():
    with pytest.raises(ValueError):
        MultiSyntheticScheffe(3, [], model="quadratic")


# ---------------- PipelineRunner multi-response ----------------
def _cfg(tmp, name="mr", props=("Прочность", "Вязкость", "Цвет")):
    return PipelineConfig(name=name, q=4, model="quadratic", noise_sd=0.1,
                          seed=3, n_random=120, n_restarts=2,
                          property_names=list(props)), tmp / name


def test_runner_m2_collects_all_properties(tmp_path):
    cfg, pdir = _cfg(tmp_path)
    r = PipelineRunner(cfg, pdir)
    assert r.property_names == ["Прочность", "Вязкость", "Цвет"]
    r.run_m1()
    out = r.run_m2(simulate=True)
    n = out["n"]
    assert r.Y.shape == (n, 3)
    assert out["Y"].shape == (n, 3)
    assert out["property_names"] == ["Прочность", "Вязкость", "Цвет"]
    # первичное свойство = столбец 0
    assert np.allclose(r.y, r.Y[:, 0])


def test_runner_m2_no_simulate_is_nan(tmp_path):
    cfg, pdir = _cfg(tmp_path)
    r = PipelineRunner(cfg, pdir)
    r.run_m1()
    r.run_m2(simulate=False)
    assert r.Y.shape[1] == 3
    assert np.all(np.isnan(r.Y))
    # simulate_responses заполняет все свойства
    Y = r.simulate_responses()
    assert Y.shape == r.Y.shape
    assert not np.any(np.isnan(r.Y))


def test_runner_saveload_preserves_properties_and_Y(tmp_path):
    root = tmp_path / "project_ui"
    cfg, pdir = _cfg(root, name="mr_io")
    r = PipelineRunner(cfg, pdir)
    r.run_m1(); r.run_m2(simulate=True)
    r.save_project()
    assert "mr_io" in list_projects(root)

    loaded = PipelineRunner.from_project(root, "mr_io")
    assert loaded.property_names == ["Прочность", "Вязкость", "Цвет"]
    assert loaded.Y is not None and loaded.Y.shape == r.Y.shape
    assert np.allclose(loaded.Y, r.Y)
    assert np.allclose(loaded.y, r.Y[:, 0])
