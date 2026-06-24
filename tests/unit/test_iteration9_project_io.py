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
"""Итерация 9, шаг 1 — сохранение/загрузка ПРОЕКТА целиком.

Проверяем, что `PipelineRunner.save_project` + `PipelineRunner.from_project`
полностью восстанавливают конфигурацию (q, имена, границы долей, стоимость) и
рабочие данные (дизайн, отклики), а `list_projects` находит сохранённый проект.
Streamlit не требуется.
"""
import numpy as np
import pytest

from src.apps.pipeline_runner import (PipelineConfig, PipelineRunner,
                                       list_projects)


def _make_runner(root, name="proj_io"):
    cfg = PipelineConfig(name=name, q=4, model="quadratic", noise_sd=0.15,
                         seed=7, n_random=150, n_restarts=2, n_blocks=2,
                         lower=[0.05, 0.0, 0.1, 0.0],
                         upper=[0.6, 0.5, 0.7, 0.4],
                         names=["Stab", "Filler", "Resin", "Plast"],
                         cost_coeffs=[2.0, 1.0, 3.5, 0.5])
    return PipelineRunner(cfg, root / name)


def test_save_and_reload_project(tmp_path):
    root = tmp_path / "project_ui"
    runner = _make_runner(root)
    runner.run_m1()
    runner.run_m2()

    # вписали "реальные" отклики и сохранили проект
    runner.y = np.round(runner.y + 0.01, 3)
    path = runner.save_project()
    assert path.endswith("state.json")

    # проект виден в каталоге
    assert "proj_io" in list_projects(root)

    # загрузка в новый runner
    loaded = PipelineRunner.from_project(root, "proj_io")

    # конфигурация восстановлена точно
    assert loaded.q == 4
    assert list(loaded.names) == ["Stab", "Filler", "Resin", "Plast"]
    assert np.allclose(loaded.region.lower, [0.05, 0.0, 0.1, 0.0])
    assert np.allclose(loaded.region.upper, [0.6, 0.5, 0.7, 0.4])
    assert np.allclose(loaded.cost_coeffs, [2.0, 1.0, 3.5, 0.5])
    assert loaded.cfg.model == "quadratic"
    assert loaded.cfg.seed == 7
    assert loaded.cfg.n_blocks == 2

    # рабочие данные восстановлены
    assert loaded.design is not None and loaded.y is not None
    assert np.allclose(loaded.design, runner.design)
    assert np.allclose(loaded.y, runner.y)
    assert loaded.state.stage == "M2_screening_design"


def test_list_projects_empty(tmp_path):
    assert list_projects(tmp_path / "nope") == []


def test_from_snapshot_defaults_for_old_state(tmp_path):
    # старый/частичный снимок без новых полей — грузится на дефолтах
    cfg = PipelineConfig.from_snapshot("legacy", {"q": 3})
    assert cfg.q == 3
    assert cfg.model == "quadratic"
    assert cfg.n_runs_factor == 2.0
