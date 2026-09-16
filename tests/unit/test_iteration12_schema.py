# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 12 / §13.1 фундамент Mixture-Process + Schema Evolution.

Проверяемый канон (REBUILD_SPEC §13.1 / §13.9-«Фундамент»):
  * Точка = составной версионированный объект ``{X:{block:[...]}, Y:{resp:val|MISSING}}``
    + ``schema_version`` + ``origin_tag`` (НЕ плоский массив);
  * ``MISSING`` — явный сентинел, допустим ТОЛЬКО в Y, не в X;
  * ``X["PROCESS"]`` — в коде [0,1]; ``code↔real`` обратима;
  * Σx=1 проверяется ТОЛЬКО на MIXTURE-блоке; на PROCESS — интервал [0,1];
  * ≤1 MIXTURE, ≤1 PROCESS, оба пустых запрещены;
  * mixture-only / process-only / mixture-process — частные случаи (набор блоков);
  * ``schema_diff_vars`` различает новые ПЕРЕМЕННЫЕ (расширение пространства).
"""
import numpy as np
import pytest

from src.core.schema import (
    MIXTURE, PROCESS, MISSING, is_missing,
    VariableBlock, ModelSpec, ResponseSpec, ProjectSchema, DataPoint,
    ordered_blocks, composite_coords, composite_matrix, split_composite,
    schema_diff_vars,
)


# ----------------------------------------------------------------------
# VariableBlock: code↔real + валидация границ
# ----------------------------------------------------------------------
def test_process_code_real_roundtrip():
    b = VariableBlock.process(["T", "t"], lower=[100.0, 10.0], upper=[200.0, 20.0])
    code = b.to_code([150.0, 15.0])
    assert np.allclose(code, [0.5, 0.5])
    real = b.from_code(code)
    assert np.allclose(real, [150.0, 15.0])
    # двумерный вход (n×d) кодируется построчно
    codes = b.to_code(np.array([[100.0, 10.0], [200.0, 20.0]]))
    assert np.allclose(codes, [[0.0, 0.0], [1.0, 1.0]])


def test_mixture_block_to_simplex_region():
    b = VariableBlock.mixture(["A", "B", "C"])
    region = b.as_simplex_region()
    assert region.q == 3
    # process-блок не даёт симплекс-региона
    p = VariableBlock.process(["z"], [0.0], [10.0])
    with pytest.raises(ValueError):
        p.as_simplex_region()


def test_infeasible_mixture_bounds_raise():
    # сумма нижних границ > 1 → симплекс невыполним
    with pytest.raises(ValueError):
        VariableBlock.mixture(["A", "B"], lower=[0.6, 0.6])


# ----------------------------------------------------------------------
# ProjectSchema: число блоков, частные случаи, запреты
# ----------------------------------------------------------------------
def test_schema_modes_block_counts():
    mo = ProjectSchema.mixture_only(["A", "B", "C"])
    assert mo.n_mixture == 3 and mo.n_process == 0

    po = ProjectSchema.process_only(["T", "t"], [100, 10], [200, 20])
    assert po.n_mixture == 0 and po.n_process == 2

    mp = ProjectSchema.mixture_process(
        VariableBlock.mixture(["A", "B"]),
        VariableBlock.process(["T"], [100], [200]))
    assert mp.n_mixture == 2 and mp.n_process == 1
    assert mp.mixture_names == ("A", "B") and mp.process_names == ("T",)


def test_schema_forbids_empty_and_duplicates():
    with pytest.raises(ValueError):                 # оба блока пустые
        ProjectSchema(version=1, blocks=())
    with pytest.raises(ValueError):                 # 2 mixture
        ProjectSchema(version=1, blocks=(
            VariableBlock.mixture(["A", "B"]),
            VariableBlock.mixture(["C", "D"])))
    with pytest.raises(ValueError):                 # 2 process
        ProjectSchema(version=1, blocks=(
            VariableBlock.process(["x"], [0], [1]),
            VariableBlock.process(["y"], [0], [1])))


def test_modelspec_defaults_and_validation():
    m = ModelSpec()
    assert m.cross_level == "cross-main"            # дефолт (§13.3)
    assert m.process_order == "quadratic"
    with pytest.raises(ValueError):
        ModelSpec(cross_level="quad-cross")
    with pytest.raises(ValueError):
        ModelSpec(process_order="cubic")


# ----------------------------------------------------------------------
# DataPoint: поблочная валидация инвариантов
# ----------------------------------------------------------------------
def _mp_schema():
    return ProjectSchema.mixture_process(
        VariableBlock.mixture(["A", "B", "C"]),
        VariableBlock.process(["T", "t"], [100, 10], [200, 20]),
        responses=(ResponseSpec("visc", "min"),))


def test_datapoint_valid_passes():
    # iter102: PROCESS в точке — ФИЗИЧЕСКИЕ единицы (T=140 ∈ [100,200],
    # t=16 ∈ [10,20]); код [0,1] строится под схему в composite_coords.
    s = _mp_schema()
    pt = DataPoint(schema_version=1,
                   X={MIXTURE: [0.2, 0.3, 0.5], PROCESS: [140.0, 16.0]},
                   Y={"visc": 12.0}, origin_tag={"stage": "M2"})
    assert pt.validate(s) is pt


def test_datapoint_mixture_sum_must_be_one():
    s = _mp_schema()
    pt = DataPoint(1, {MIXTURE: [0.2, 0.3, 0.4], PROCESS: [0.5, 0.5]})
    with pytest.raises(ValueError):
        pt.validate(s)


def test_datapoint_process_must_be_within_block_bounds():
    # iter102: физическое значение вне границ блока — отказ (t=0.5 ∉ [10,20]).
    # До iter102 здесь проверялся код ∈ [0,1], и 150 «не было кодом».
    s = _mp_schema()
    pt = DataPoint(1, {MIXTURE: [0.2, 0.3, 0.5], PROCESS: [150.0, 0.5]})
    with pytest.raises(ValueError, match="вне границ"):
        pt.validate(s)
    # у зажатой оси (lo == hi) допуск абсолютный, а не доля нулевого размаха
    s_pin = ProjectSchema.mixture_process(
        VariableBlock.mixture(["A", "B", "C"]),
        VariableBlock.process(["T", "t"], [170, 10], [170, 20]))
    DataPoint(1, {MIXTURE: [0.2, 0.3, 0.5], PROCESS: [170.0, 15.0]}).validate(s_pin)
    with pytest.raises(ValueError, match="вне границ"):
        DataPoint(1, {MIXTURE: [0.2, 0.3, 0.5], PROCESS: [171.0, 15.0]}).validate(s_pin)


def test_missing_forbidden_in_X_allowed_in_Y():
    s = _mp_schema()
    bad = DataPoint(1, {MIXTURE: [0.2, MISSING, 0.5], PROCESS: [140.0, 16.0]})
    with pytest.raises(ValueError):
        bad.validate(s)
    ok = DataPoint(1, {MIXTURE: [0.2, 0.3, 0.5], PROCESS: [140.0, 16.0]},
                   Y={"visc": MISSING})
    assert ok.validate(s) is ok


# ----------------------------------------------------------------------
# Составные координаты: порядок MIXTURE → PROCESS, код под схему (iter102)
# ----------------------------------------------------------------------
def test_composite_coords_order_and_split():
    s = _mp_schema()
    pt = DataPoint(1, {MIXTURE: [0.2, 0.3, 0.5], PROCESS: [140.0, 16.0]})
    vec = composite_coords(s, pt)
    # mixture как есть, process — КОД под границы схемы: (140−100)/100, (16−10)/10
    assert np.allclose(vec, [0.2, 0.3, 0.5, 0.4, 0.6])
    parts = split_composite(s, vec)
    assert np.allclose(parts[MIXTURE], [0.2, 0.3, 0.5])
    assert np.allclose(parts[PROCESS], [0.4, 0.6])
    M = composite_matrix(s, [pt, pt])
    assert M.shape == (2, 5)
    # та же точка под ДРУГИМИ границами T — другой код, физика та же
    s2 = ProjectSchema.mixture_process(
        VariableBlock.mixture(["A", "B", "C"]),
        VariableBlock.process(["T", "t"], [120, 10], [160, 20]))
    assert np.allclose(composite_coords(s2, pt)[3], 0.5)
    assert np.allclose(pt.process_code(s2), [0.5, 0.6])
    with pytest.raises(ValueError, match="требует схему"):
        pt.process_code()


def test_ordered_blocks_mixture_first():
    s = _mp_schema()
    kinds = [b.kind for b in ordered_blocks(s)]
    assert kinds == [MIXTURE, PROCESS]


# ----------------------------------------------------------------------
# Сериализация: round-trip схемы и точки (включая MISSING ↔ null)
# ----------------------------------------------------------------------
def test_schema_serialization_roundtrip():
    s = _mp_schema()
    s2 = ProjectSchema.from_dict(s.to_dict())
    assert s2.to_dict() == s.to_dict()
    assert s2.n_mixture == 3 and s2.n_process == 2


def test_datapoint_serialization_roundtrip_with_missing():
    pt = DataPoint(2, {MIXTURE: [0.2, 0.3, 0.5], PROCESS: [140.0, 16.0]},
                   Y={"visc": 12.0, "cost": MISSING},
                   origin_tag={"stage": "M5", "schema_version": 2})
    d = pt.to_dict()
    assert d["Y"]["cost"] is None                  # MISSING → null на диске
    assert d["process_units"] == "real"            # iter102: маркер формата
    pt2 = DataPoint.from_dict(d)
    assert is_missing(pt2.Y["cost"]) and pt2.Y["visc"] == 12.0
    assert pt2.schema_version == 2
    assert np.allclose(pt2.X[PROCESS], [140.0, 16.0])   # физика дословно
    # точка без process-блока маркера не несёт (нечего маркировать)
    assert "process_units" not in DataPoint(1, {MIXTURE: [0.5, 0.5]}).to_dict()


# ----------------------------------------------------------------------
# schema_diff_vars: новые ПЕРЕМЕННЫЕ при расширении схемы (§13.7)
# ----------------------------------------------------------------------
def test_schema_diff_vars_detects_added_process():
    old = ProjectSchema.mixture_only(["A", "B", "C"], version=1)
    new = ProjectSchema.mixture_process(
        VariableBlock.mixture(["A", "B", "C"]),
        VariableBlock.process(["T"], [100], [200]), version=2)
    diff = schema_diff_vars(old, new)
    assert diff[PROCESS] == ["T"]                  # появилась новая переменная
    assert diff[MIXTURE] == []                     # mixture не расширялся
