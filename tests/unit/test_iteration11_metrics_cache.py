# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 11 / лёгкая персистентность метрик стадий + инвалидация.

Проверяемый канон (по запросу пользователя):
  * после ``save_project`` → ``from_project`` метрики пройденных стадий доступны
    БЕЗ повторного дорогого пересчёта (``cached_metrics`` восстановлены с диска),
    хотя in-memory ``results`` после загрузки пуст;
  * ассистент после загрузки считает стадии пройденными
    (``build_context.stages_done``) и видит их числа (``metrics``), не помечая
    их как «без метрик»;
  * изменение откликов (симуляция/ручной ввод) ИНВАЛИДИРУЕТ метрики стадий,
    зависящих от ``y`` (M3–M8), сохраняя независимые от откликов (M1/M2/M5);
  * ручной пересчёт стадии возвращает её свежие метрики (приоритет над кэшем).
"""
import warnings

import numpy as np
import pytest

from src.apps.pipeline_runner import PipelineConfig, PipelineRunner
from src.apps import assistant as ai

warnings.filterwarnings("ignore")


def _runner(tmp_path, name="m11"):
    cfg = PipelineConfig(name=name, q=3, model="linear", property_names=["A", "B"],
                         seed=1, n_restarts=2, noise_sd=0.1)
    r = PipelineRunner(cfg, tmp_path / name)
    r.run_m1()
    r.run_m2(simulate=True)
    r.run_m3_fit()
    r.run_m3_ard()
    r.run_m4()
    r.run_m6()
    return r


# ----------------------------------------------------------------------
def test_metrics_survive_save_load_without_results(tmp_path):
    """Кэш метрик переживает перезагрузку, хотя results после load пуст."""
    r = _runner(tmp_path)
    fresh = r.stage_metrics()
    assert {"M1", "M2", "M3_fit", "M3_ard", "M4", "M6"} <= set(fresh)
    r.save_project()

    root = str(tmp_path)
    r2 = PipelineRunner.from_project(root, "m11")
    # in-memory results пуст — стадии в этой сессии не выполнялись
    assert r2.results == {}
    # но метрики восстановлены из кэша на диске
    cached = r2.stage_metrics()
    assert {"M1", "M2", "M3_fit", "M3_ard", "M4", "M6"} <= set(cached)
    assert "r2" in cached["M3_fit"]["A"]
    # числовые значения совпадают с исходными (M3_fit R² свойства A)
    assert cached["M3_fit"]["A"]["r2"] == pytest.approx(
        fresh["M3_fit"]["A"]["r2"], rel=1e-9)


def test_loaded_context_marks_stages_done_with_metrics(tmp_path):
    """Ассистент после загрузки: стадии пройдены И с метриками (не «пустые»)."""
    r = _runner(tmp_path)
    r.save_project()
    r2 = PipelineRunner.from_project(str(tmp_path), "m11")

    ctx = ai.build_context(r2)
    for s in ("M1", "M2", "M3_fit", "M3_ard", "M4", "M6"):
        assert s in ctx["stages_done"]
        assert s in ctx["metrics"]
    # раз метрики восстановлены — список «без метрик» по этим стадиям пуст
    assert not ({"M3_fit", "M3_ard", "M4", "M6"}
                & set(ctx["stages_without_metrics"]))


def test_invalidate_metrics_on_response_change(tmp_path):
    """Смена откликов сбрасывает зависящие от y стадии, оставляя M1/M2/M5."""
    r = _runner(tmp_path)
    r.run_m5()                       # план, не зависит от откликов
    before = set(r.stage_metrics())
    assert {"M3_fit", "M4", "M6", "M5"} <= before

    r.simulate_responses()           # отклики изменились → инвалидация
    after = set(r.stage_metrics())
    # зависящие от откликов стадии сброшены
    assert {"M3_fit", "M3_ard", "M4", "M6"}.isdisjoint(after)
    # независимые от откликов (геометрия/план) сохранены
    assert {"M1", "M2", "M5"} <= after


def test_plan_coords_and_blocks_in_metrics(tmp_path):
    """M2/M5 метрики содержат координаты плана и разбиение по блокам.

    Закрывает замечание ассистента: «состав/координаты точек нового плана,
    число добавляемых runs и разбиение по блокам в контексте отсутствуют».
    """
    cfg = PipelineConfig(name="m11b", q=3, model="linear",
                         property_names=["A"], seed=2, n_restarts=2,
                         noise_sd=0.1, n_blocks=3)
    r = PipelineRunner(cfg, tmp_path / "m11b")
    r.run_m1()
    r.run_m2(simulate=True)
    r.run_m5()
    m = r.stage_metrics()

    # M2: координаты плана + имена компонентов + разбиение по блокам
    n2 = m["M2"]["n"]
    assert len(m["M2"]["design"]) == n2
    assert len(m["M2"]["design"][0]) == 3            # q=3 доли на точку
    assert m["M2"]["component_names"] == list(r.names)
    assert sum(m["M2"]["block_sizes"].values()) == n2
    assert set(m["M2"]["block_sizes"]) == {1, 2, 3}  # n_blocks=3

    # M5: координаты предлагаемого плана + n_runs + разбиение по блокам
    n5 = m["M5"]["n_runs"]
    assert len(m["M5"]["design"]) == n5
    assert m["M5"]["applied"] is False
    assert sum(m["M5"]["block_sizes"].values()) == n5

    # сериализуемо для ассистента/MCP (numpy → стандартные типы)
    import json
    json.dumps(m)


def test_recompute_overrides_cached_metric(tmp_path):

    """Пересчёт стадии в сессии перекрывает кэш с диска (свежие числа)."""
    r = _runner(tmp_path)
    r.save_project()
    r2 = PipelineRunner.from_project(str(tmp_path), "m11")

    cached_r2 = r2.stage_metrics()["M3_fit"]["A"]["r2"]
    # перезапускаем M3_fit в новой сессии — результат идентичный (детерминирован),
    # но теперь он берётся из свежих results, а не из кэша
    r2.run_m3_fit()
    assert "M3_fit" in r2.results
    fresh_r2 = r2.stage_metrics()["M3_fit"]["A"]["r2"]
    assert fresh_r2 == pytest.approx(cached_r2, rel=1e-6)
