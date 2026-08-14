# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 42 — слой НАВЕСКИ (UI_REVISION_SPEC §42, шаг P1.1).

Три части шага:

  * **42.1 логика** — :meth:`PhrSpec.fractions_to_phr`: обратное к
    ``to_fractions``. Доли масштаба не несут, поэтому Σphr восстанавливается
    по ЯКОРЮ — ``fixed``-листу спеки; остальные fixed-листья служат проверкой
    согласованности (расхождение = ошибка ДАННЫХ, явный ValueError);
  * **42.2 δ навески** — :func:`campaign_ui.weighing_delta_phr`:
    ``δ = шаг весов (г) / (г на 1 phr)`` (golden CAMPAIGN_SPEC_PVC §5:
    0.1 г при 5 г/phr → δ = 0.02 phr);
  * **42.3 карта навески** — :func:`campaign_ui.recipe_weighing_dataframe`
    поверх контракта ядра ``point_report`` (iter49): nominal / actual /
    граммы / премикс / нарушение. Требование §5: дозируется и фиксируется
    ACTUAL, модель должна видеть actual, а не nominal.

A0.6: нарушения геометрии — ДИАГНОСТИКА в колонке «нарушение», не исключение.
"""
import numpy as np
import pandas as pd
import pytest

from src.apps.campaign_ui import (build_setup_runner, recipe_weighing_dataframe,
                                  seed_design_excel_bytes,
                                  seed_weighing_dataframe, snap_design_to_grid,
                                  weighing_caption, weighing_delta_phr)
from src.design.phr_sampler import PhrSpec

# ----------------------------------------------------------------------
# PVC-подобная v2-спека с ДВУМЯ fixed-листьями (RESIN — якорь, ESO —
# проверка согласованности) + группа с техлимитами + log-ось с cap.
# ----------------------------------------------------------------------
NODES = [
    {"name": "RESIN", "role": "FIXED", "value": 100.0},
    {"name": "DINP", "role": "ABSOLUTE", "range": [4.0, 14.0]},
    {"name": "ESO", "role": "FIXED", "value": 2.5},
    {"name": "SOFT", "role": "GROUP_TOTAL", "range": [5.0, 15.0],
     "members": ["PBNK", "CPE"]},
    {"name": "PBNK", "role": "SHARE_FREE", "group": "SOFT",
     "share_range": [0.0, 0.70], "max_phr": 8.0},
    {"name": "CPE", "role": "SHARE_CLOSURE", "group": "SOFT",
     "min_phr": 3.0},
    {"name": "UV", "role": "ABSOLUTE_CAPPED", "range": [0.05, 0.30],
     "scale": "log", "cap_to": ["DINP", "ESO"], "cap_ratio": 0.03},
]


def _spec() -> PhrSpec:
    return PhrSpec.from_dicts(NODES)


def _point(spec: PhrSpec, seed: int = 0) -> np.ndarray:
    """Валидный номинальный рецепт (phr) из собственного сэмплера спеки."""
    return spec.decode(spec.sample_z(1, seed=seed)[0])


# ----------------------------------------------------------------------
# 42.1 — fractions_to_phr: round-trip и якорь
# ----------------------------------------------------------------------
def test_round_trip_single_recipe():
    spec = _spec()
    p = _point(spec, seed=3)
    back = spec.fractions_to_phr(spec.to_fractions(p))
    assert np.allclose(back, p, rtol=1e-10, atol=1e-9)


def test_round_trip_batch_matrix():
    spec = _spec()
    P = spec.decode(spec.sample_z(7, seed=11))
    back = spec.fractions_to_phr(spec.to_fractions(P))
    assert back.shape == P.shape
    assert np.allclose(back, P, rtol=1e-10, atol=1e-9)


def test_anchor_value_reproduced_exactly():
    """Якорный компонент получает РОВНО свою константу (нормировка на Σx)."""
    spec = _spec()
    p = _point(spec, seed=5)
    back = spec.fractions_to_phr(spec.to_fractions(p))
    j_resin = spec.component_names.index("RESIN")
    j_eso = spec.component_names.index("ESO")
    assert back[j_resin] == pytest.approx(100.0, abs=1e-9)
    assert back[j_eso] == pytest.approx(2.5, abs=1e-6)


def test_no_fixed_leaf_is_explicit_error():
    """Без fixed-листа масштаб НЕОПРЕДЕЛИМ — явная ошибка, не «принято 100»."""
    spec = PhrSpec.from_dicts([
        {"name": "A", "role": "ABSOLUTE", "range": [1.0, 10.0]},
        {"name": "B", "role": "ABSOLUTE", "range": [1.0, 10.0]},
    ])
    with pytest.raises(ValueError, match="НЕОПРЕДЕЛИМ"):
        spec.fractions_to_phr([0.5, 0.5])


def test_inconsistent_fractions_rejected():
    """Второй fixed-лист — проверка: доли не из этой спеки → ошибка данных."""
    spec = _spec()
    x = spec.to_fractions(_point(spec, seed=1))
    j_eso = spec.component_names.index("ESO")
    bad = x.copy()
    bad[j_eso] *= 3.0                     # ESO больше не 2.5 phr при якоре 100
    bad = bad / bad.sum()
    with pytest.raises(ValueError, match="не согласованы со спекой"):
        spec.fractions_to_phr(bad)


def test_wrong_length_and_zero_sum_are_errors():
    spec = _spec()
    with pytest.raises(ValueError, match="ожидалось"):
        spec.fractions_to_phr([0.5, 0.5])
    with pytest.raises(ValueError, match="сумма долей"):
        spec.fractions_to_phr(np.zeros(spec.q))


# ----------------------------------------------------------------------
# 42.2 — δ навески из параметров лаборатории
# ----------------------------------------------------------------------
def test_delta_phr_golden_from_campaign_spec():
    # CAMPAIGN_SPEC_PVC §5: весы 0.1 г, загрузка 5 г на 1 phr → δ = 0.02 phr
    assert weighing_delta_phr(0.1, 5.0) == pytest.approx(0.02)


@pytest.mark.parametrize("step, gpp", [(0.0, 5.0), (-0.1, 5.0),
                                       (0.1, 0.0), (0.1, -5.0)])
def test_delta_phr_rejects_nonpositive(step, gpp):
    with pytest.raises(ValueError):
        weighing_delta_phr(step, gpp)


# ----------------------------------------------------------------------
# 42.3 — карта навески (чистый хелпер UI поверх point_report)
# ----------------------------------------------------------------------
def test_weighing_dataframe_structure_and_actual_on_grid():
    spec = _spec()
    x = spec.to_fractions(_point(spec, seed=7))
    delta = 0.02
    df = recipe_weighing_dataframe(spec, x, delta, grams_per_phr=5.0)
    assert list(df["компонент"]) == list(spec.component_names)
    for col in ("phr nominal", "phr actual", "граммы actual", "премикс",
                "нарушение"):
        assert col in df.columns
    # actual — узлы δ-сетки (снап квантованием), с допуском на округление показа
    act = np.asarray(df["phr actual"], float)
    assert np.allclose(act / delta, np.round(act / delta), atol=1e-3)
    # граммы = actual · (г на 1 phr)
    assert np.allclose(np.asarray(df["граммы actual"], float), act * 5.0,
                       atol=1e-3)


def test_premix_flags_golden_uv_yes_dinp_no():
    """golden §5 при δ=0.02: UV [0.05, 0.30] → премикс, DINP [4, 14] → прямая."""
    spec = _spec()
    x = spec.to_fractions(_point(spec, seed=2))
    df = recipe_weighing_dataframe(spec, x, 0.02).set_index("компонент")
    assert df.loc["UV", "премикс"] == "да"
    assert df.loc["DINP", "премикс"] == "нет"
    # fixed-оси: правило премикса неприменимо (вырожденный интервал) — «—»
    assert df.loc["RESIN", "премикс"] == "—"
    assert df.loc["ESO", "премикс"] == "—"


def test_valid_recipe_has_no_violations():
    spec = _spec()
    x = spec.to_fractions(_point(spec, seed=4))
    df = recipe_weighing_dataframe(spec, x, 0.02)
    assert all(str(v) == "" for v in df["нарушение"])


def test_violations_are_reported_not_raised():
    """A0.6: номинал вне границ — строка в колонке «нарушение», не исключение."""
    spec = _spec()
    p = _point(spec, seed=6)
    p[spec.component_names.index("DINP")] = 20.0      # > hi = 14
    df = recipe_weighing_dataframe(spec, spec.to_fractions(p), 0.02)
    df = df.set_index("компонент")
    assert str(df.loc["DINP", "нарушение"]).strip() != ""
    assert str(df.loc["UV", "нарушение"]).strip() == ""


def test_caption_mentions_delta_and_actual():
    spec = _spec()
    cap = weighing_caption(spec, 0.02)
    assert "0.02" in cap
    assert "actual" in cap
    assert "UV" in cap                     # ось требует премикса при δ=0.02


# ----------------------------------------------------------------------
# 42.4 — снап плана к δ-сетке (фиксируется actual) и карта навески плана
# ----------------------------------------------------------------------
def _runner(spec: PhrSpec):
    """Раннер РЕАЛЬНОГО сетапа с именами компонентов ИЗ СПЕКИ (iter41.1)."""
    return build_setup_runner(
        mixture_names=list(spec.component_names), process_names=["T"],
        process_lower=[150.0], process_upper=[200.0], response_names=["y"],
        seed=0)


def _plan(spec: PhrSpec, n: int = 4, seed: int = 0) -> np.ndarray:
    """Составной план (n × (q+1)): доли из спеки + процесс в коде [0,1]."""
    X = spec.sample_candidates(n, seed=seed)
    return np.column_stack([X, np.linspace(0.0, 1.0, n)])


def test_snap_puts_phr_on_grid_and_keeps_process():
    spec = _spec()
    X = _plan(spec, n=5, seed=1)
    delta = 0.02
    Xs = snap_design_to_grid(spec, X, delta)
    assert Xs.shape == X.shape
    # процесс-часть НЕ тронута (снапится только рецептура)
    assert np.allclose(Xs[:, spec.q:], X[:, spec.q:])
    # каждая строка в phr — на δ-сетке
    for i in range(len(Xs)):
        p = spec.fractions_to_phr(Xs[i, :spec.q])
        assert np.allclose(p / delta, np.round(p / delta), atol=1e-6)


def test_snap_is_idempotent():
    spec = _spec()
    X = _plan(spec, n=4, seed=2)
    once = snap_design_to_grid(spec, X, 0.02)
    twice = snap_design_to_grid(spec, once, 0.02)
    assert np.allclose(once, twice, atol=1e-12)


def test_snap_rejects_too_narrow_plan():
    spec = _spec()
    with pytest.raises(ValueError, match="координат"):
        snap_design_to_grid(spec, np.zeros((2, spec.q - 1)), 0.02)


def test_seed_weighing_dataframe_is_long_format():
    spec = _spec()
    runner = _runner(spec)
    X = _plan(spec, n=3, seed=3)
    df = seed_weighing_dataframe(runner, spec, X, 0.02, grams_per_phr=5.0)
    assert len(df) == 3 * spec.q          # строка = «опыт × компонент»
    # база пуста ⇒ будущие номера опытов 1…3, каждый по q строк
    assert sorted(set(df["№ опыта"])) == [1, 2, 3]
    assert list(df[df["№ опыта"] == 1]["компонент"]) == list(spec.component_names)


def test_excel_gets_weighing_sheet_only_with_spec_and_delta():
    import io
    spec = _spec()
    runner = _runner(spec)
    X = _plan(spec, n=2, seed=4)
    plain = pd.ExcelFile(io.BytesIO(
        seed_design_excel_bytes(runner, X)))
    # iter97: лист «Отклики» появляется всегда; предмет теста — «Навеска»,
    # которая по-прежнему требует ОБА условия (спека + δ).
    assert "Навеска" not in plain.sheet_names
    assert plain.sheet_names == ["Стартовый дизайн", "Отклики"]
    withw = pd.ExcelFile(io.BytesIO(seed_design_excel_bytes(
        runner, X, spec=spec, delta_phr=0.02, grams_per_phr=5.0)))
    assert "Навеска" in withw.sheet_names
    wdf = withw.parse("Навеска")
    assert len(wdf) == 2 * spec.q
    assert "phr actual" in wdf.columns
