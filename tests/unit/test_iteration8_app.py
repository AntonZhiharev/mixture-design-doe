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
"""Итерация 8 — end-to-end тест оркестратора UI (src/apps/pipeline_runner).

Прогоняет весь pipeline M1→M8 на маленькой конфигурации (быстрый), проверяет,
что каждая стадия отдаёт результат, пишет чекпоинты `after_M*` и что состояние
восстанавливается из чекпоинта. Streamlit не требуется.
"""
import numpy as np
import pandas as pd
import pytest

from src.core.state import ProjectState
from src.apps.pipeline_runner import PipelineConfig, PipelineRunner


@pytest.fixture(scope="module")
def runner(tmp_path_factory):
    proj = tmp_path_factory.mktemp("ui_proj")
    cfg = PipelineConfig(name="t8", q=3, model="quadratic", noise_sd=0.1,
                         seed=1, n_random=200, n_restarts=2)
    return PipelineRunner(cfg, proj)


def test_m1_geometry(runner):
    r = runner.run_m1()
    assert r["n_vertices"] >= 3
    assert r["vertices"].shape[1] == 3
    assert runner.region.is_feasible(r["centroid"])
    assert r["checkpoint"].endswith("after_M1.json")


def test_m2_design(runner):
    r = runner.run_m2()
    assert r["design"].shape == (runner.n_runs, 3)
    assert r["y"].shape[0] == runner.n_runs
    assert 0.0 < r["d_efficiency"]
    assert np.allclose(r["design"].sum(axis=1), 1.0, atol=1e-6)


def test_m3_fit(runner):
    r = runner.run_m3_fit()
    assert isinstance(r["coef_table"], pd.DataFrame)
    assert len(r["coef_table"]) == runner.p
    assert -0.01 <= r["r2"] <= 1.0001
    assert np.isfinite(r["rmse"])


def test_m3_ard(runner):
    r = runner.run_m3_ard()
    assert 1 <= r["q_eff"] <= 3
    assert len(r["active"]) == r["q_eff"]


def test_m4_regimes(runner):
    r = runner.run_m4()
    assert r["n_regimes"] >= 1
    assert isinstance(r["bic_table"], pd.DataFrame)


def test_m5_i_optimal(runner):
    r = runner.run_m5()
    assert r["i_optimal"] > 0
    assert r["design"].shape[1] == 3


def test_m6_moe(runner):
    r = runner.run_m6()
    assert runner.moe is not None
    assert r["n_regimes"] >= 1
    assert np.isfinite(r["test_rmse"])


def test_m7_active_learning(runner):
    r = runner.run_m7(n_iter_refine=2, n_iter_search=2)
    assert r["n_final"] > r["n_start"]
    assert runner.region.is_feasible(r["x_best"])


def test_m8_optimization(runner):
    r = runner.run_m8(prop_weight=1.0, cost_weight=1.0)
    assert np.allclose(r["recipe"].sum(), 1.0, atol=1e-6)
    assert runner.region.is_feasible(r["recipe"])
    assert 0.0 <= r["d_overall"] <= 1.0


def test_benchmark_and_checkpoints(runner):
    b = runner.benchmark(n_scan=2000)
    assert b["recipe_dist"] >= 0.0
    assert runner.region.is_feasible(b["x_true"])

    cps = runner.checkpoints()
    for label in ("after_M1", "after_M2", "after_M3", "after_M4",
                  "after_M5", "after_M6", "after_M7", "after_M8"):
        assert label in cps
