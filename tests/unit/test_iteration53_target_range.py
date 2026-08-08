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
"""Iteration 53 — P2.2 (UI_REVISION_SPEC): плато-таргет в DesirabilitySpec.

Двусторонний таргет-ДИАПАЗОН (постановка «желатинизация 60–70 %»):
d = 1 на [target_low, target_high], линейные (или степенные) рампы к 0 на
краях low/high, вне [low, high] — 0. Точечный ``target`` здесь не годится:
пик предпочёл бы середину допуска и тянул бы оптимизатор от дешёвого края.

Проверяется:
  * ядро: форма кривой (плато/рампы/нули/показатели s, s2), валидации;
  * сериализация: asdict → DesirabilitySpec(**d) round-trip; Branch
    to_state/from_state; старые записи (без новых ключей) грузятся;
  * argmax: optimize_desirability находит точку ВНУТРИ плато;
  * UI-хелперы: build_goal_spec / draft_add_goal / draft_goal_specs /
    draft_goal_text / goal_editor_dataframe (чистые, без Streamlit).
"""
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from src.core.simplex import SimplexRegion
from src.design.branches import Branch
from src.optimize.desirability import (DesirabilitySpec, desirability_value,
                                       optimize_desirability,
                                       overall_desirability)
from src.apps import campaign_ui as ui


# ----------------------------------------------------------------------
# Ядро: форма кривой
# ----------------------------------------------------------------------
def _plateau(low=50.0, tl=60.0, th=70.0, high=80.0, **kw):
    return DesirabilitySpec("target_range", low=low, high=high,
                            target_low=tl, target_high=th, **kw)


def test_plateau_is_one_inside_range_inclusive():
    spec = _plateau()
    y = np.array([60.0, 62.5, 65.0, 70.0])
    assert np.allclose(desirability_value(y, spec), 1.0)


def test_ramps_are_linear_by_default():
    spec = _plateau()
    d = desirability_value(np.array([55.0, 75.0]), spec)
    # (55-50)/(60-50) = 0.5; (80-75)/(80-70) = 0.5
    assert d == pytest.approx([0.5, 0.5])


def test_zero_outside_low_high():
    spec = _plateau()
    d = desirability_value(np.array([49.9, 50.0, 80.0, 90.0]), spec)
    assert d[0] == 0.0 and d[3] == 0.0
    # на самих краях рампы дают 0 (числитель нулевой)
    assert d[1] == 0.0 and d[2] == 0.0


def test_shape_exponents_s_and_s2():
    spec = _plateau(s=2.0, s2=0.5)
    d = desirability_value(np.array([55.0, 75.0]), spec)
    assert d[0] == pytest.approx(0.5 ** 2.0)
    assert d[1] == pytest.approx(0.5 ** 0.5)


def test_s2_defaults_to_s():
    spec = _plateau(s=3.0)
    assert spec.s2 == pytest.approx(3.0)


def test_scalar_input_ok():
    spec = _plateau()
    assert float(desirability_value(65.0, spec)) == pytest.approx(1.0)


# ----------------------------------------------------------------------
# Ядро: валидации (A0.6 — ничего молча)
# ----------------------------------------------------------------------
def test_requires_both_plateau_bounds():
    with pytest.raises(ValueError, match="target_low"):
        DesirabilitySpec("target_range", low=50, high=80, target_low=60)
    with pytest.raises(ValueError, match="target_low"):
        DesirabilitySpec("target_range", low=50, high=80, target_high=70)


def test_requires_strict_ordering():
    for tl, th in [(60.0, 60.0), (70.0, 60.0), (50.0, 70.0), (60.0, 80.0)]:
        with pytest.raises(ValueError, match="target_low < target_high"):
            DesirabilitySpec("target_range", low=50, high=80,
                             target_low=tl, target_high=th)


def test_point_target_forbidden_on_range():
    with pytest.raises(ValueError, match="plateau"):
        DesirabilitySpec("target_range", low=50, high=80,
                         target_low=60, target_high=70, target=65.0)


def test_plateau_fields_forbidden_on_other_kinds():
    with pytest.raises(ValueError, match="only valid"):
        DesirabilitySpec("max", low=0, high=1, target_low=0.5)
    with pytest.raises(ValueError, match="only valid"):
        DesirabilitySpec("target", low=0, high=1, target=0.5, target_high=0.7)


def test_old_kinds_untouched():
    # прежние виды собираются как раньше (обратная совместимость)
    DesirabilitySpec("max", low=0, high=1)
    DesirabilitySpec("min", low=0, high=1)
    DesirabilitySpec("target", low=0, high=1, target=0.5)


# ----------------------------------------------------------------------
# Сериализация: dataclass round-trip + Branch state + старые записи
# ----------------------------------------------------------------------
def test_asdict_roundtrip():
    spec = _plateau(weight=2.0, s=1.5)
    d = asdict(spec)
    back = DesirabilitySpec(**d)
    assert back.kind == "target_range"
    assert back.target_low == pytest.approx(60.0)
    assert back.target_high == pytest.approx(70.0)
    assert back.weight == pytest.approx(2.0)
    y = np.linspace(45, 85, 41)
    assert np.allclose(desirability_value(y, back),
                       desirability_value(y, spec))


def test_branch_state_roundtrip_with_plateau_goal():
    br = Branch(id="b1", name="gel", goal={"gel_pct": _plateau(weight=1.5)})
    back = Branch.from_state(br.to_state())
    g = back.goal["gel_pct"]
    assert g.kind == "target_range"
    assert (g.target_low, g.target_high) == (60.0, 70.0)
    assert g.weight == pytest.approx(1.5)


def test_legacy_goal_dict_without_new_keys_loads():
    # старый сейв: записи целей не знают target_low/target_high
    legacy = {"kind": "max", "low": 2.0, "high": 12.0, "target": None,
              "s": 1.0, "s2": 1.0, "weight": 1.0}
    spec = DesirabilitySpec(**legacy)
    assert spec.target_low is None and spec.target_high is None


# ----------------------------------------------------------------------
# Агрегация и argmax
# ----------------------------------------------------------------------
def test_overall_desirability_with_plateau():
    d = overall_desirability({
        "gel": desirability_value(np.array([65.0, 40.0]), _plateau()),
        "str": np.array([0.5, 0.5]),
    })
    assert d[0] == pytest.approx(np.sqrt(0.5))
    assert d[1] == 0.0                      # вне [low, high] → veto


def test_optimizer_lands_inside_plateau():
    # y = 100·x0 на 2-компонентном симплексе; плато [60, 70] ⇒ x0* ∈ [0.6, 0.7]
    region = SimplexRegion(q=2)
    res = optimize_desirability(
        region,
        predictors={"gel": lambda X: 100.0 * np.atleast_2d(X)[:, 0]},
        specs={"gel": _plateau()},
        n_candidates=400, refine_iters=50, n_starts=2, seed=0)
    assert res.d_overall == pytest.approx(1.0)
    assert 0.6 - 1e-6 <= res.x[0] <= 0.7 + 1e-6


# ----------------------------------------------------------------------
# UI-хелперы (чистые, без Streamlit)
# ----------------------------------------------------------------------
def test_build_goal_spec_range():
    spec = ui.build_goal_spec(ui.GOAL_KIND_RANGE, low=50, high=80,
                              target_low=60, target_high=70, weight=2.0)
    assert spec.kind == "target_range"
    assert (spec.target_low, spec.target_high) == (60.0, 70.0)
    assert spec.weight == pytest.approx(2.0)


def test_build_goal_spec_range_requires_fields():
    with pytest.raises(ValueError, match="low и high"):
        ui.build_goal_spec(ui.GOAL_KIND_RANGE, target_low=60, target_high=70)
    with pytest.raises(ValueError, match="плато"):
        ui.build_goal_spec(ui.GOAL_KIND_RANGE, low=50, high=80)


def test_goal_kind_range_registered():
    assert ui.GOAL_KIND_RANGE in ui.GOAL_KINDS


def test_draft_add_goal_stores_plateau_and_builds_spec():
    d = ui.draft_add_goal([], resp="gel", kind=ui.GOAL_KIND_RANGE,
                          low=50.0, high=80.0,
                          target_low=60.0, target_high=70.0, weight=1.5)
    assert d[0]["target_low"] == 60.0 and d[0]["target_high"] == 70.0
    specs = ui.draft_goal_specs(d)
    assert specs["gel"].kind == "target_range"
    assert specs["gel"].target_high == pytest.approx(70.0)


def test_draft_add_goal_invalid_plateau_raises_immediately():
    with pytest.raises(ValueError):
        ui.draft_add_goal([], resp="gel", kind=ui.GOAL_KIND_RANGE,
                          low=50.0, high=80.0,
                          target_low=70.0, target_high=60.0)


def test_draft_goal_text_mentions_plateau():
    d = ui.draft_add_goal([], resp="gel", kind=ui.GOAL_KIND_RANGE,
                          low=50.0, high=80.0,
                          target_low=60.0, target_high=70.0)
    txt = ui.draft_goal_text(d[0])
    assert "плато" in txt and "60.0" in txt and "70.0" in txt


def test_goal_editor_dataframe_shows_plateau_range():
    br = Branch(id="b1", name="gel", goal={"gel_pct": _plateau()})
    runner = SimpleNamespace(branches={"b1": br})
    df = ui.goal_editor_dataframe(runner, "b1")
    row = df.iloc[0]
    assert row["вид"] == "target_range"
    assert "плато" in str(row["target"])
    assert "60.0" in str(row["target"]) and "70.0" in str(row["target"])


def test_goal_editor_dataframe_point_target_unchanged():
    br = Branch(id="b2", name="t", goal={
        "y": DesirabilitySpec("target", low=0.0, high=10.0, target=5.0)})
    runner = SimpleNamespace(branches={"b2": br})
    df = ui.goal_editor_dataframe(runner, "b2")
    assert df.iloc[0]["target"] == pytest.approx(5.0)