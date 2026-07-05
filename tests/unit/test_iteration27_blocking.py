# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 27 / Blocking: стартовый (оптимальный) + добор (sequential).

Проверяемый канон:
  * СТАРТОВЫЙ blocking (M2): план разбивается на блоки ОПТИМАЛЬНО —
    interchange по log det блочной модели (Шеффе + dummy блоков), а не
    round-robin; результат не хуже round-robin по блочному D-критерию;
  * blocking ДОБОРА: каждый добор точек (M5 / M7 / раунд ветки) — это НОВАЯ
    партия и получает НОВЫЙ блок (следующий свободный номер);
  * диагностика: цена блокировки (потеря информации модельных термов после
    учёта блоков, через дополнение Шура) считается и конечна;
  * runner: run_m2 назначает оптимальные блоки; run_branch_round дописывает
    новый блок; blocks переживают save/load.
"""
import warnings

import numpy as np
import pytest

from src.core.simplex import SimplexRegion
from src.design.d_optimal import d_optimal_for_region
from src.design.blocking import (
    BlockingResult, assign_blocks_start, augment_block_labels,
    balanced_block_sizes, block_dummies, blocked_logdet,
    blocked_scheffe_matrix, blocking_diagnostics,
)
from src.apps.pipeline_runner import PipelineConfig, PipelineRunner
from src.optimize.desirability import DesirabilitySpec

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
# Утилиты
# ----------------------------------------------------------------------
def test_balanced_block_sizes():
    assert balanced_block_sizes(10, 2) == [5, 5]
    assert balanced_block_sizes(11, 3) == [4, 4, 3]
    assert balanced_block_sizes(5, 5) == [1, 1, 1, 1, 1]
    with pytest.raises(ValueError):
        balanced_block_sizes(10, 0)


def test_block_dummies_reference_block():
    """B блоков → B-1 dummy-столбцов; блок 1 — референс (все нули)."""
    lab = [1, 1, 2, 3, 2]
    Z = block_dummies(lab)
    assert Z.shape == (5, 2)
    # строки блока 1 — нулевые (референс)
    assert np.all(Z[0] == 0) and np.all(Z[1] == 0)
    # блок 2 → первый столбец, блок 3 → второй
    assert Z[2, 0] == 1 and Z[2, 1] == 0
    assert Z[3, 0] == 0 and Z[3, 1] == 1
    # один блок → пустая dummy-матрица
    assert block_dummies([1, 1, 1]).shape == (3, 0)


def test_blocked_matrix_shape_and_rank():
    """[F | Z] имеет p+B-1 столбцов и полный ранг при достатке точек."""
    region = SimplexRegion(q=3)
    res = d_optimal_for_region(region, n_runs=12, model="quadratic",
                               n_random=150, n_restarts=2, seed=0)
    blk = assign_blocks_start(res.design, 2, model="quadratic",
                              n_restarts=3, seed=0)
    M = blocked_scheffe_matrix(res.design, blk.labels, "quadratic")
    p = 6  # quadratic, q=3
    assert M.shape == (12, p + 1)
    assert np.linalg.matrix_rank(M) == p + 1


# ----------------------------------------------------------------------
# 1) Стартовый blocking: оптимальное разбиение
# ----------------------------------------------------------------------
def _start_design(q=3, n_runs=12, seed=0):
    region = SimplexRegion(q=q)
    return d_optimal_for_region(region, n_runs=n_runs, model="quadratic",
                                n_random=150, n_restarts=2, seed=seed).design


def test_assign_blocks_start_balanced_and_labeled():
    X = _start_design()
    blk = assign_blocks_start(X, 3, model="quadratic", n_restarts=3, seed=1)
    assert isinstance(blk, BlockingResult)
    assert len(blk.labels) == len(X)
    assert set(blk.labels.tolist()) == {1, 2, 3}
    _, cnt = np.unique(blk.labels, return_counts=True)
    assert sorted(cnt.tolist()) == [4, 4, 4]          # сбалансировано
    assert np.isfinite(blk.logdet)
    assert 0.0 < blk.d_efficiency <= 1.5


def test_assign_blocks_start_beats_round_robin():
    """Оптимальное разбиение НЕ ХУЖЕ round-robin по блочному D-критерию."""
    X = _start_design(n_runs=14)
    blk = assign_blocks_start(X, 2, model="quadratic", n_restarts=4, seed=2)
    rr = (np.arange(len(X)) % 2) + 1                  # старый round-robin
    ld_rr = blocked_logdet(X, rr, "quadratic")
    assert blk.logdet >= ld_rr - 1e-9


def test_assign_blocks_start_deterministic_with_seed():
    X = _start_design()
    a = assign_blocks_start(X, 2, model="quadratic", n_restarts=3, seed=7)
    b = assign_blocks_start(X, 2, model="quadratic", n_restarts=3, seed=7)
    assert np.array_equal(a.labels, b.labels)
    assert a.logdet == pytest.approx(b.logdet)


def test_assign_blocks_start_single_block_trivial():
    X = _start_design()
    blk = assign_blocks_start(X, 1, model="quadratic")
    assert blk.n_blocks == 1
    assert np.all(blk.labels == 1)


def test_assign_blocks_start_explicit_sizes_and_validation():
    X = _start_design(n_runs=12)
    blk = assign_blocks_start(X, 2, model="quadratic",
                              block_sizes=[8, 4], n_restarts=2, seed=0)
    _, cnt = np.unique(blk.labels, return_counts=True)
    assert sorted(cnt.tolist()) == [4, 8]
    with pytest.raises(ValueError):                    # сумма != n
        assign_blocks_start(X, 2, block_sizes=[5, 4])


def test_assign_blocks_start_warns_when_underdetermined():
    """n < p + B - 1 → блочная модель сингулярна → предупреждение."""
    X = _start_design(q=3, n_runs=7)                  # p=6, B=3 → нужно ≥ 8
    with pytest.warns(UserWarning):
        assign_blocks_start(X, 3, model="quadratic", n_restarts=1, seed=0)


# ----------------------------------------------------------------------
# 2) Blocking добора: новый блок на партию
# ----------------------------------------------------------------------
def test_augment_block_labels_new_block():
    full = augment_block_labels([1, 1, 2, 2], 3)
    assert full.tolist() == [1, 1, 2, 2, 3, 3, 3]     # добор = блок 3


def test_augment_block_labels_empty_base():
    assert augment_block_labels(None, 2).tolist() == [1, 1]
    assert augment_block_labels([], 2).tolist() == [1, 1]


def test_augment_block_labels_multi_new_blocks():
    full = augment_block_labels([1, 1], 5, n_new_blocks=2)
    # добор разбит на 2 новых блока (3 и 4) сбалансированно
    assert full[:2].tolist() == [1, 1]
    new = full[2:]
    assert set(new.tolist()) == {2, 3}
    _, cnt = np.unique(new, return_counts=True)
    assert sorted(cnt.tolist()) == [2, 3]


def test_augment_block_labels_zero_and_negative():
    assert augment_block_labels([1, 2], 0).tolist() == [1, 2]
    with pytest.raises(ValueError):
        augment_block_labels([1], -1)


def test_sequential_augment_labels_grow_monotonically():
    """Последовательные доборы получают возрастающие номера блоков."""
    lab = augment_block_labels(np.ones(6, dtype=int), 3)   # добор №1 → блок 2
    lab = augment_block_labels(lab, 2)                      # добор №2 → блок 3
    lab = augment_block_labels(lab, 1)                      # добор №3 → блок 4
    assert lab.tolist() == [1] * 6 + [2] * 3 + [3] * 2 + [4]


# ----------------------------------------------------------------------
# Диагностика
# ----------------------------------------------------------------------
def test_blocking_diagnostics_keys_and_sanity():
    X = _start_design(n_runs=14)
    blk = assign_blocks_start(X, 2, model="quadratic", n_restarts=3, seed=3)
    d = blocking_diagnostics(X, blk.labels, "quadratic")
    assert d["n_blocks"] == 2
    assert sum(d["block_sizes"].values()) == len(X)
    assert np.isfinite(d["d_eff_unblocked"]) and d["d_eff_unblocked"] > 0
    assert np.isfinite(d["d_eff_blocked"]) and d["d_eff_blocked"] > 0
    assert np.isfinite(d["d_eff_model_adj"])
    # учёт блоков не добавляет информации модельным термам
    assert d["d_eff_model_adj"] <= d["d_eff_unblocked"] + 1e-9
    assert d["d_loss_pct"] >= -1e-6


def test_blocking_diagnostics_single_block_no_loss():
    X = _start_design()
    d = blocking_diagnostics(X, np.ones(len(X), dtype=int), "quadratic")
    assert d["n_blocks"] == 1
    assert d["d_loss_pct"] == pytest.approx(0.0, abs=1e-6)


# ----------------------------------------------------------------------
# Интеграция с runner
# ----------------------------------------------------------------------
def _runner(tmp_path, name="m27", n_blocks=2, q=3):
    cfg = PipelineConfig(name=name, q=q, model="linear",
                         property_names=["A"], seed=3, n_restarts=2,
                         noise_sd=0.1, n_blocks=n_blocks, n_random=150)
    r = PipelineRunner(cfg, tmp_path / name)
    r.run_m1()
    r.run_m2(simulate=True)
    return r


def test_run_m2_optimal_blocking_and_diagnostics(tmp_path):
    r = _runner(tmp_path, n_blocks=2)
    m2 = r.results["M2"]
    bl = np.asarray(m2["blocks"]).astype(int)
    assert len(bl) == m2["n"]
    assert set(bl.tolist()) == {1, 2}
    _, cnt = np.unique(bl, return_counts=True)
    assert abs(int(cnt[0]) - int(cnt[1])) <= 1        # сбалансировано
    # диагностика стартового blocking присутствует и конечна
    diag = m2["blocking"]
    assert diag is not None
    assert np.isfinite(diag["d_eff_blocked"])
    assert np.isfinite(diag["d_loss_pct"])
    # разбиение оптимально: не хуже round-robin по блочному D-критерию
    rr = (np.arange(len(bl)) % 2) + 1
    assert (blocked_logdet(r.design, bl, r.cfg.model)
            >= blocked_logdet(r.design, rr, r.cfg.model) - 1e-9)


def test_run_m2_single_block_default(tmp_path):
    r = _runner(tmp_path, name="m27s", n_blocks=1)
    m2 = r.results["M2"]
    assert np.all(np.asarray(m2["blocks"]) == 1)
    assert m2["blocking"] is None


def test_branch_round_appends_new_block(tmp_path):
    """Добор точек веткой = НОВЫЙ блок в общей базе."""
    r = _runner(tmp_path, name="m27b", n_blocks=2)
    n0 = len(r.design)
    max_b0 = int(np.max(r.blocks))
    r.add_branch("goal", {"A": DesirabilitySpec("max", low=0.0, high=1.0)},
                 budget=4)
    out = r.run_branch_round("b1", n_points=2, refit=False)
    assert out["added"] == 2
    assert len(r.blocks) == len(r.design) == n0 + 2
    # новые точки — в новом блоке (следующий свободный номер)
    assert r.blocks[n0:].tolist() == [max_b0 + 1] * 2
    # база не перенумерована
    assert int(np.max(r.blocks[:n0])) == max_b0

    # второй добор → ещё один новый блок
    out2 = r.run_branch_round("b1", n_points=2, refit=False)
    assert out2["added"] == 2
    assert r.blocks[n0 + 2:].tolist() == [max_b0 + 2] * 2


def test_blocks_survive_save_load_after_augment(tmp_path):
    r = _runner(tmp_path, name="m27p", n_blocks=2)
    r.add_branch("goal", {"A": DesirabilitySpec("max", low=0.0, high=1.0)},
                 budget=3)
    r.run_branch_round("b1", n_points=1, refit=False)
    blocks_before = np.asarray(r.blocks).astype(int).copy()
    r.save_project()

    r2 = PipelineRunner.from_project(str(tmp_path), "m27p")
    assert r2.blocks is not None
    assert np.array_equal(np.asarray(r2.blocks).astype(int), blocks_before)
    assert len(r2.blocks) == len(r2.design)


def test_m5_metrics_preview_uses_new_block(tmp_path):
    """Предпросмотр блоков плана M5 — НОВЫЙ блок, а не round-robin по базе."""
    r = _runner(tmp_path, name="m27m5", n_blocks=2)
    r.run_m5()
    m = r.stage_metrics()
    n5 = m["M5"]["n_runs"]
    if n5 > 0:
        assert sum(m["M5"]["block_sizes"].values()) == n5
        max_base = int(np.max(np.asarray(r.blocks).astype(int)))
        # все предлагаемые точки — в блоках СТРОГО выше блоков базы
        assert min(m["M5"]["blocks"]) > max_base - 1
        assert set(m["M5"]["blocks"]) == {max_base + 1}
    # сериализуемо (ассистент/MCP)
    import json
    json.dumps(m)