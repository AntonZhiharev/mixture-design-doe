# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 28 / Blocking в campaign-flow (MixtureProcessRunner + UI).

Проверяемый канон (продолжение iteration 27):
  * СТАРТОВЫЙ blocking seed-дизайна: ``n_blocks_start`` партий назначаются
    ОПТИМАЛЬНО (interchange по блочному D-критерию на mixture-долях),
    детерминированно по seed раннера (показ = фиксация);
  * blocking ДОБОРА: каждый commit-раунд ветки — НОВАЯ партия → ОДИН новый
    номер блока (sequential);
  * блок точки живёт в ``origin_tag["block"]`` → переживает миграцию схемы и
    save/load кампании; ``n_blocks_start`` персистится в campaign.json;
  * UI-таблицы: «Блок» в базе кампании, в seed-дизайне (при nb>1) и в
    таблице долитых точек раунда.
"""
import warnings

import numpy as np
import pytest

from src.core.schema import ModelSpec, ProjectSchema, VariableBlock
from src.apps.mixture_process_runner import MixtureProcessRunner
from src.apps import campaign_state as cs
from src.optimize.desirability import DesirabilitySpec

warnings.filterwarnings("ignore")


class _Oracle:
    property_names = ["y1", "y2"]

    def evaluate(self, Xc) -> np.ndarray:
        Xc = np.atleast_2d(np.asarray(Xc, float))
        y1 = 3.0 * Xc[:, 0] + 2.0 * Xc[:, 1] + 1.0 * Xc[:, 2]
        y2 = 1.0 * Xc[:, 0] + 2.0 * Xc[:, 1] + 3.0 * Xc[:, 2]
        return np.column_stack([y1, y2])


def _runner(n_blocks=2, seed=5):
    mix = VariableBlock.mixture(["A", "B", "C"])
    proc = VariableBlock.process(["T"], lower=[0.0], upper=[1.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    schema = ProjectSchema.mixture_process(mix, proc, model=model)
    return MixtureProcessRunner(schema, _Oracle(), seed=seed, n_restarts=2,
                                n_blocks_start=n_blocks)


# ----------------------------------------------------------------------
# Стартовый blocking seed-дизайна
# ----------------------------------------------------------------------
def test_seed_initial_assigns_optimal_blocks():
    r = _runner(n_blocks=2)
    r.seed_initial(n=10)
    blocks = [p.origin_tag.get("block") for p in r.points]
    assert all(b is not None for b in blocks)
    assert set(blocks) == {1, 2}
    _, cnt = np.unique(blocks, return_counts=True)
    assert sorted(cnt.tolist()) == [5, 5]              # сбалансировано
    assert r.point_blocks() == [int(b) for b in blocks]


def test_seed_block_labels_deterministic_propose_equals_commit():
    """Показ (propose) и фиксация (commit_seed) дают ОДНИ И ТЕ ЖЕ метки."""
    r = _runner(n_blocks=3, seed=9)
    X = r.propose_seed(12)
    lab1 = r.seed_block_labels(X)
    lab2 = r.seed_block_labels(X)
    assert np.array_equal(lab1, lab2)                  # детерминизм по seed
    Y = r._measure(X)
    r.commit_seed(X, Y)
    assert r.point_blocks() == [int(v) for v in lab1]


def test_seed_single_block_default():
    r = _runner(n_blocks=1)
    r.seed_initial(n=8)
    assert set(r.point_blocks()) == {1}


def test_blocking_summary_keys():
    r = _runner(n_blocks=2)
    r.seed_initial(n=10)
    d = r.blocking_summary()
    assert d["n_blocks"] == 2
    assert sum(d["block_sizes"].values()) == 10
    assert np.isfinite(d["d_eff_blocked"])
    assert np.isfinite(d["d_loss_pct"])


# ----------------------------------------------------------------------
# Blocking добора: commit-раунд = новый блок
# ----------------------------------------------------------------------
def _runner_with_branch(n_blocks=2):
    r = _runner(n_blocks=n_blocks)
    r.seed_initial(n=10)
    r.add_branch("b", {"y1": DesirabilitySpec("max", low=0.0, high=4.0)},
                 budget=10, branch_id="b1")
    return r


def test_commit_measured_new_block_per_round():
    r = _runner_with_branch(n_blocks=2)
    max_b0 = max(r.point_blocks())                     # = 2 (стартовые партии)

    out1 = r.run_branch_round("b1", n_points=2, n_candidates=100)
    assert out1["added"] == 2
    bl = r.point_blocks()
    assert bl[-2:] == [max_b0 + 1] * 2                 # добор №1 → блок 3
    assert max(bl[:-2]) == max_b0                      # база не перенумерована

    out2 = r.run_branch_round("b1", n_points=1, n_candidates=100)
    assert out2["added"] == 1
    assert r.point_blocks()[-1] == max_b0 + 2          # добор №2 → блок 4


def test_legacy_points_without_block_default_to_one():
    """Старые точки без метки блока считаются блоком 1; добор → блок 2."""
    r = _runner(n_blocks=1)
    r.seed_initial(n=8)
    for p in r.points:                                 # эмуляция старого проекта
        p.origin_tag.pop("block", None)
    assert r.point_blocks() == [1] * 8
    assert r._next_block() == 2


# ----------------------------------------------------------------------
# Персистентность (campaign.json)
# ----------------------------------------------------------------------
def test_blocks_survive_campaign_save_load(tmp_path):
    r = _runner_with_branch(n_blocks=2)
    r.run_branch_round("b1", n_points=2, n_candidates=100)
    blocks_before = list(r.point_blocks())
    cs.save_campaign(r, tmp_path, "blk")

    r2 = cs.load_campaign(tmp_path, "blk", oracle=_Oracle())
    assert int(getattr(r2, "n_blocks_start", 1)) == 2
    assert list(r2.point_blocks()) == blocks_before
    # добор после загрузки продолжает нумерацию блоков
    assert r2._next_block() == max(blocks_before) + 1


# ----------------------------------------------------------------------
# UI-таблицы (чистые хелперы campaign_ui, без Streamlit-рендера)
# ----------------------------------------------------------------------
def test_campaign_base_dataframe_has_block_column():
    from src.apps import campaign_ui as ui
    r = _runner_with_branch(n_blocks=2)
    r.run_branch_round("b1", n_points=2, n_candidates=100)
    df = ui.campaign_base_dataframe(r)
    assert "Блок" in df.columns
    assert list(df["Блок"]) == list(r.point_blocks())
    # добор в последней партии (новый блок)
    assert int(df["Блок"].iloc[-1]) == max(r.point_blocks())


def test_seed_design_dataframe_block_column_only_when_blocked():
    from src.apps import campaign_ui as ui
    r = _runner(n_blocks=2)
    X = r.propose_seed(10)
    df = ui.seed_design_dataframe(r, X)
    assert "Блок" in df.columns
    assert list(df["Блок"]) == [int(v) for v in r.seed_block_labels(X)]

    r1 = _runner(n_blocks=1)
    X1 = r1.propose_seed(8)
    df1 = ui.seed_design_dataframe(r1, X1)
    assert "Блок" not in df1.columns                   # без блокировки — нет


def test_workbench_points_dataframe_has_block():
    from src.apps import campaign_ui as ui
    r = _runner_with_branch(n_blocks=1)
    res = r.run_branch_round("b1", n_points=2, n_candidates=100)
    wdf = ui.workbench_points_dataframe(r, res)
    assert "Блок" in wdf.columns
    assert set(wdf["Блок"].astype(int)) == {max(r.point_blocks())}


# ----------------------------------------------------------------------
# Видимость blocking в UI (подписи; фикс «не вижу блокинг» при nb=1)
# ----------------------------------------------------------------------
def test_seed_blocking_caption_off_explains_how_to_enable():
    """nb=1 (дефолт): подпись явно говорит, что blocking выключен и как включить."""
    from src.apps import campaign_ui as ui
    r = _runner(n_blocks=1)
    X = r.propose_seed(8)
    txt = ui.seed_blocking_caption(r, X)
    assert "выключена" in txt
    assert "Партий (блоков)" in txt


def test_seed_blocking_caption_on_shows_sizes_and_loss():
    """nb>1: подпись несёт размеры партий и цену блокировки (d_loss %)."""
    from src.apps import campaign_ui as ui
    r = _runner(n_blocks=2)
    X = r.propose_seed(10)
    txt = ui.seed_blocking_caption(r, X)
    assert "включена" in txt
    assert "блок 1" in txt and "блок 2" in txt
    assert "%" in txt


def test_base_blocking_caption_after_commit_and_round():
    """База с nb>1 и добором: подпись со сводкой партий; nb=1 → пустая строка."""
    from src.apps import campaign_ui as ui
    r = _runner_with_branch(n_blocks=2)
    r.run_branch_round("b1", n_points=2, n_candidates=100)
    txt = ui.base_blocking_caption(r)
    assert "блок 1" in txt and "блок 3" in txt      # старт 1..2 + добор → 3
    assert "НОВЫЙ блок" in txt

    r1 = _runner(n_blocks=1)
    r1.seed_initial(n=8)
    assert ui.base_blocking_caption(r1) == ""       # 1 партия — не показываем


# ----------------------------------------------------------------------
# Названия блоков (block_factor / block_names): таблицы, подписи, Excel, save/load
# ----------------------------------------------------------------------
def _xlsx_to_df(data: bytes):
    import io
    import pandas as pd
    return pd.read_excel(io.BytesIO(data))


def test_block_display_and_named_tables():
    from src.apps import campaign_ui as ui
    r = _runner(n_blocks=2)
    r.block_factor = "рабочая смена"
    r.block_names = {1: "смена А", 2: "смена Б"}
    assert ui.block_display(r, 1) == "смена А"
    assert ui.block_display(r, 3) == "3"            # без имени — номер

    X = r.propose_seed(10)
    df = ui.seed_design_dataframe(r, X)
    assert "Партия" in df.columns                   # имена → столбец «Партия»
    lab = [int(v) for v in r.seed_block_labels(X)]
    assert list(df["Партия"]) == [{1: "смена А", 2: "смена Б"}[b] for b in lab]

    cap = ui.seed_blocking_caption(r, X)
    assert "смена А" in cap and "рабочая смена" in cap

    # без имён столбца «Партия» нет (старое поведение не тронуто)
    r0 = _runner(n_blocks=2)
    assert "Партия" not in ui.seed_design_dataframe(r0, r0.propose_seed(10)).columns


def test_block_names_in_base_and_excel():
    """Имена блоков доезжают до общей базы и ОБЕИХ Excel-выгрузок."""
    from src.apps import campaign_ui as ui
    r = _runner_with_branch(n_blocks=2)
    r.block_factor = "оператор"
    r.block_names = {1: "Иванов", 2: "Петров"}
    r.run_branch_round("b1", n_points=2, n_candidates=100)   # добор → блок 3

    df = ui.campaign_base_dataframe(r)
    assert "Партия" in df.columns
    assert set(df["Партия"]) == {"Иванов", "Петров", "3"}    # добор без имени

    xdf = _xlsx_to_df(ui.campaign_base_excel_bytes(r))
    assert "Блок" in xdf.columns and "Партия" in xdf.columns
    assert list(xdf["Блок"]) == list(r.point_blocks())
    assert set(xdf["Партия"].astype(str)) == {"Иванов", "Петров", "3"}

    X = r.propose_seed(6)
    sdf = _xlsx_to_df(ui.seed_design_excel_bytes(r, X))
    assert "Блок" in sdf.columns and "Партия" in sdf.columns

    cap = ui.base_blocking_caption(r)
    assert "Иванов" in cap and "оператор" in cap


def test_block_names_survive_save_load(tmp_path):
    r = _runner(n_blocks=2)
    r.seed_initial(n=8)
    r.block_factor = "партия сырья"
    r.block_names = {1: "сырьё №101", 2: "сырьё №102"}
    cs.save_campaign(r, tmp_path, "blk_names")

    r2 = cs.load_campaign(tmp_path, "blk_names", oracle=_Oracle())
    assert r2.block_factor == "партия сырья"
    assert r2.block_names == {1: "сырьё №101", 2: "сырьё №102"}  # int-ключи


def test_setup_prefill_carries_blocking():
    """Загрузка проекта префиллит и число партий, и фактор, и имена блоков."""
    from src.apps import campaign_ui as ui
    r = _runner(n_blocks=3)
    r.block_factor = "смена"
    r.block_names = {2: "ночная"}
    out = ui.setup_prefill_from_runner(r)
    assert out["setup_seed_blocks"] == 3
    assert out["setup_block_factor"] == "смена"
    assert out["setup_block_name_2"] == "ночная"
