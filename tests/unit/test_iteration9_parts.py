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
"""Итерация 9, шаг 2a — пересчёт массовых частей (база=100) в диапазоны долей.

Проверяем формулу `parts_ranges_to_fraction_bounds` на ручном примере, её
совместимость с `SimplexRegion` и то, что доля базы действительно «плавает».
"""
import numpy as np
import pytest

from src.core.simplex import SimplexRegion, parts_ranges_to_fraction_bounds


def test_manual_example_3comp_base100():
    # база (comp0) фиксирована = 100; comp1 ∈ [10,20]; comp2 ∈ [5,15]
    a = [100.0, 10.0, 5.0]
    b = [100.0, 20.0, 15.0]
    lower, upper = parts_ranges_to_fraction_bounds(a, b)

    assert lower == pytest.approx([100/135, 10/125, 5/125], rel=1e-6)
    assert upper == pytest.approx([100/115, 20/125, 15/125], rel=1e-6)

    # диапазон доли базы НЕ постоянен (плавает)
    assert lower[0] < upper[0]


def test_region_accepts_and_recipe_feasible():
    a = [100.0, 10.0, 5.0]
    b = [100.0, 20.0, 15.0]
    lower, upper = parts_ranges_to_fraction_bounds(a, b)
    region = SimplexRegion(lower=lower, upper=upper)

    # конкретный рецепт в частях → доли, должен лежать в области
    parts = np.array([100.0, 15.0, 10.0])
    x = parts / parts.sum()
    assert region.is_feasible(x)

    # граничные рецепты тоже допустимы (сумма bounds охватывает 1)
    assert region.lower.sum() <= 1.0 + 1e-9
    assert region.upper.sum() >= 1.0 - 1e-9


def test_fixed_base_only_one_varies():
    # 2 компонента: база=100, второй ∈ [0,100]
    lower, upper = parts_ranges_to_fraction_bounds([100.0, 0.0], [100.0, 100.0])
    # второй: доля от 0 до 100/200=0.5
    assert lower == pytest.approx([100/200, 0.0], rel=1e-6)
    assert upper == pytest.approx([1.0, 100/200], rel=1e-6)


def test_validation_errors():
    with pytest.raises(ValueError):
        parts_ranges_to_fraction_bounds([1.0, 2.0], [1.0])      # длины
    with pytest.raises(ValueError):
        parts_ranges_to_fraction_bounds([5.0], [3.0])           # min>max
    with pytest.raises(ValueError):
        parts_ranges_to_fraction_bounds([-1.0], [2.0])          # отрицательное
