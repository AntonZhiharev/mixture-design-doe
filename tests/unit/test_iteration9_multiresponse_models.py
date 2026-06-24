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
"""Итерация 9, шаг 3b — мультиотклик в моделях (REBUILD_SPEC §12).

M3 screening (`run_m3_fit`/`run_m3_ard`) запускается на КАЖДОЕ свойство, а M6
строит словарь общих на проект суррогатов ``{property → MoE}``. Верхний уровень
результата сохраняет совместимость (первичное свойство = столбец 0).
"""
import numpy as np
import pytest

from src.apps.pipeline_runner import (PipelineConfig, PipelineRunner,
                                       list_projects)
from src.models.moe import MixtureOfExperts


PROPS = ("Прочность", "Вязкость", "Цвет")


def _runner(pdir, props=PROPS):
    cfg = PipelineConfig(name="mr3b", q=4, model="quadratic", noise_sd=0.1,
                         seed=3, n_random=120, n_restarts=2,
                         property_names=list(props))
    r = PipelineRunner(cfg, pdir)
    r.run_m1()
    r.run_m2(simulate=True)
    return r


# ---------------- M3a: Scheffe fit per property ----------------
def test_m3_fit_runs_for_all_properties(tmp_path):
    r = _runner(tmp_path / "p")
    out = r.run_m3_fit()
    assert set(out["per_property"]) == set(PROPS)
    assert out["property_names"] == list(PROPS)
    # каждая модель имеет r2 / rmse
    for name in PROPS:
        info = out["per_property"][name]
        assert info["property"] == name
        assert np.isfinite(info["r2"]) and np.isfinite(info["rmse"])
    # верхний уровень = первичное свойство (столбец 0) — совместимость
    assert out["r2"] == out["per_property"][PROPS[0]]["r2"]
    # отдельная модель в состоянии на каждое свойство + алиас первичного
    for i in range(len(PROPS)):
        assert f"m3_scheffe__{i}" in r.state.models
    assert r.state.models["m3_scheffe"] == r.state.models["m3_scheffe__0"]


# ---------------- M3b: ARD screening per property ----------------
def test_m3_ard_runs_for_all_properties(tmp_path):
    r = _runner(tmp_path / "p")
    out = r.run_m3_ard()
    assert set(out["per_property"]) == set(PROPS)
    for name in PROPS:
        info = out["per_property"][name]
        assert info["property"] == name
        assert info["q_eff"] >= 1
        assert isinstance(info["active"], list)
    # совместимость: верхний уровень = первичное свойство
    assert out["q_eff"] == out["per_property"][PROPS[0]]["q_eff"]
    for i in range(len(PROPS)):
        assert f"m3_ard_screening__{i}" in r.state.models
    assert (r.state.models["m3_ard_screening"]
            == r.state.models["m3_ard_screening__0"])


# ---------------- M6: surrogate dict property -> MoE ----------------
def test_m6_builds_surrogate_per_property(tmp_path):
    r = _runner(tmp_path / "p")
    out = r.run_m6()
    # словарь суррогатов — по одному MoE на свойство
    assert set(r.surrogates) == set(PROPS)
    for name in PROPS:
        assert isinstance(r.surrogates[name], MixtureOfExperts)
    # первичный суррогат = алиас self.moe (для M7/M8/benchmark)
    assert r.moe is r.surrogates[PROPS[0]]
    # диагностика per_property
    assert set(out["per_property"]) == set(PROPS)
    assert out["n_regimes"] == out["per_property"][PROPS[0]]["n_regimes"]
    for i in range(len(PROPS)):
        assert f"m6_moe__{i}" in r.state.models


def test_m6_surrogates_differ_between_properties(tmp_path):
    r = _runner(tmp_path / "p")
    r.run_m6()
    X = r.region.random_points(50, seed=123)
    p0 = r.surrogates[PROPS[0]].predict(X).mean
    p1 = r.surrogates[PROPS[1]].predict(X).mean
    # разные истины → разные предсказания
    assert not np.allclose(p0, p1)


# ---------------- save/load preserves surrogates ----------------
def test_saveload_restores_surrogates(tmp_path):
    root = tmp_path / "project_ui"
    r = _runner(root / "mr3b")
    r.run_m6()
    X = r.region.random_points(40, seed=7)
    before = {n: r.surrogates[n].predict(X).mean for n in PROPS}
    r.save_project()
    assert "mr3b" in list_projects(root)

    loaded = PipelineRunner.from_project(root, "mr3b")
    assert set(loaded.surrogates) == set(PROPS)
    for name in PROPS:
        assert isinstance(loaded.surrogates[name], MixtureOfExperts)
        after = loaded.surrogates[name].predict(X).mean
        assert np.allclose(after, before[name])
    # первичный суррогат восстановлен как self.moe
    assert loaded.moe is not None


# ---------------- single-property backward compat ----------------
def test_single_property_still_works(tmp_path):
    r = _runner(tmp_path / "p", props=("y",))
    fit = r.run_m3_fit()
    assert list(fit["per_property"]) == ["y"]
    moe = r.run_m6()
    assert list(r.surrogates) == ["y"]
    assert "n_regimes" in moe
