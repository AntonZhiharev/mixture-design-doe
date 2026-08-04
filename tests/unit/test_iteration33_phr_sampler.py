# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 33 / phr/DAG z-сэмплер-плагин (DECODE_LAYER_PROPOSAL, этап A).

Двухслойная параметризация z → p (phr) → x (доли, Σ=1) как САМПЛЕР-ПЛАГИН,
без изменения схемы/модели. Проверяемый канон iter33:

  * спека → DAG: 4 режима (absolute/share_of/ratio_to/fixed), топосорт,
    циклы и неизвестные ссылки — явные ошибки КОНСТРУКТОРА;
  * статическая валидация интервальной арифметикой ДО сэмплинга: пустое
    пересечение долей группы (Σlo>1 / Σhi<1), нулевой референс — ошибки
    конфига, не sample-time;
  * сэмплинг без rejection: оси z в границах, доли share-группы Σ=1
    (conditional narrowing iter31), суммы групп покрывают края диапазона;
  * decode/encode — roundtrip (anchors задаются в phr); рецепт вне спеки —
    явный ValueError;
  * доли: Σx=1 конструкцией; ratio_to-инвариант переживает нормировку;
    fraction_bounds — консервативный бокс, содержащий всех кандидатов;
  * мост в раннере: phr_spec=None → прежний путь бит-в-бит; активная спека
    даёт валидные кандидаты; несовпадение состава — warning + fallback.
"""
import numpy as np
import pytest

from src.core.schema import ModelSpec, ProjectSchema
from src.design.phr_sampler import PhrSpec
from src.apps.mixture_process_runner import MixtureProcessRunner

# PVC-подобная рецептура: смола = 100 phr (fixed), группа стабилизатора
# с ТОТАЛОМ 2..5 phr и долями внутри, SBM — ratio_to к тоталу группы
# (reference НЕ родитель SBM — ровно кейс DAG из proposal).
PVC_DICTS = [
    {"name": "resin", "mode": "fixed", "value": 100.0},
    {"name": "plasticizer", "mode": "absolute", "lo": 40.0, "hi": 60.0},
    {"name": "stab_total", "mode": "absolute", "lo": 2.0, "hi": 5.0},
    {"name": "Ca_st", "mode": "share_of", "of": "stab_total",
     "lo": 0.2, "hi": 0.7},
    {"name": "Zn_st", "mode": "share_of", "of": "stab_total",
     "lo": 0.1, "hi": 0.5},
    {"name": "ester", "mode": "share_of", "of": "stab_total",
     "lo": 0.1, "hi": 0.6},
    {"name": "SBM", "mode": "ratio_to", "to": "stab_total",
     "lo": 0.02, "hi": 0.09},
    {"name": "filler", "mode": "absolute", "lo": 0.0, "hi": 30.0},
]
COMPONENTS = ["resin", "plasticizer", "Ca_st", "Zn_st", "ester", "SBM",
              "filler"]
STAB_COLS = [2, 3, 4]           # Ca_st, Zn_st, ester в component_names
SBM_COL = 5


def _spec() -> PhrSpec:
    return PhrSpec.from_dicts(PVC_DICTS)


class _Oracle:
    property_names = ["modulus"]

    def evaluate(self, Xc):
        Xc = np.atleast_2d(np.asarray(Xc, float))
        return (100.0 * Xc[:, 1]).reshape(-1, 1)


def _model():
    return ModelSpec(cross_level="additive", mixture_order="quadratic")


# ----------------------------------------------------------------------
# Спека: парсинг, DAG, статическая валидация
# ----------------------------------------------------------------------
def test_parse_topo_components():
    spec = _spec()
    # stab_total — ВНУТРЕННИЙ узел (родитель share-группы): не компонент
    assert spec.component_names == COMPONENTS
    assert spec.q == 7
    # z-оси: все не-fixed узлы (включая тотал группы), fixed — без оси
    assert spec.z_names == ["plasticizer", "stab_total", "Ca_st", "Zn_st",
                            "ester", "SBM", "filler"]
    assert spec.dim_z == 7
    # интервальная арифметика прошла и дала phr-интервалы всех узлов
    iv = spec.phr_intervals()
    assert iv["resin"] == (100.0, 100.0)
    assert iv["SBM"] == pytest.approx((0.02 * 2.0, 0.09 * 5.0))


def test_cycle_detected():
    with pytest.raises(ValueError, match="Цикл"):
        PhrSpec.from_dicts([
            {"name": "base", "mode": "fixed", "value": 100.0},
            {"name": "A", "mode": "ratio_to", "to": "B",
             "lo": 0.1, "hi": 0.2},
            {"name": "B", "mode": "ratio_to", "to": "A",
             "lo": 0.1, "hi": 0.2},
        ])


def test_unknown_reference():
    with pytest.raises(ValueError, match="не найден"):
        PhrSpec.from_dicts([
            {"name": "base", "mode": "fixed", "value": 100.0},
            {"name": "A", "mode": "ratio_to", "to": "ghost",
             "lo": 0.1, "hi": 0.2},
        ])


def test_empty_share_intersection_is_config_error():
    # Σhi долей группы < 1 — раскладка Σ=1 невозможна: ошибка ДО сэмплинга
    with pytest.raises(ValueError, match="пустое пересечение"):
        PhrSpec.from_dicts([
            {"name": "base", "mode": "fixed", "value": 100.0},
            {"name": "tot", "mode": "absolute", "lo": 2.0, "hi": 5.0},
            {"name": "a", "mode": "share_of", "of": "tot",
             "lo": 0.0, "hi": 0.3},
            {"name": "b", "mode": "share_of", "of": "tot",
             "lo": 0.0, "hi": 0.4},
        ])


def test_reversed_bounds_rejected():
    with pytest.raises(ValueError, match="некорректные границы"):
        PhrSpec.from_dicts([
            {"name": "base", "mode": "fixed", "value": 100.0},
            {"name": "A", "mode": "absolute", "lo": 5.0, "hi": 2.0},
        ])


def test_zero_reference_rejected():
    # референс с нижней границей 0: encode-деление и Σ=1 не определены
    with pytest.raises(ValueError, match="строго положительным"):
        PhrSpec.from_dicts([
            {"name": "base", "mode": "fixed", "value": 100.0},
            {"name": "tot", "mode": "absolute", "lo": 0.0, "hi": 5.0},
            {"name": "a", "mode": "share_of", "of": "tot",
             "lo": 0.2, "hi": 0.8},
            {"name": "b", "mode": "share_of", "of": "tot",
             "lo": 0.2, "hi": 0.8},
        ])


# ----------------------------------------------------------------------
# Сэмплинг z и decode
# ----------------------------------------------------------------------
def test_sample_z_bounds_and_share_sum():
    spec = _spec()
    Z = spec.sample_z(300, seed=0)
    assert Z.shape == (300, 7)
    col = {nm: j for j, nm in enumerate(spec.z_names)}
    assert np.all(Z[:, col["plasticizer"]] >= 40.0 - 1e-12)
    assert np.all(Z[:, col["plasticizer"]] <= 60.0 + 1e-12)
    assert np.all(Z[:, col["SBM"]] >= 0.02 - 1e-12)
    assert np.all(Z[:, col["SBM"]] <= 0.09 + 1e-12)
    shares = Z[:, [col["Ca_st"], col["Zn_st"], col["ester"]]]
    np.testing.assert_allclose(shares.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(shares >= np.array([0.2, 0.1, 0.1]) - 1e-9)
    assert np.all(shares <= np.array([0.7, 0.5, 0.6]) + 1e-9)


def test_decode_ranges_and_fixed_base():
    spec = _spec()
    P = spec.decode(spec.sample_z(300, seed=1))
    assert P.shape == (300, 7)
    np.testing.assert_allclose(P[:, 0], 100.0)     # resin = 100 phr всегда
    iv = spec.phr_intervals()
    for j, nm in enumerate(spec.component_names):
        lo, hi = iv[nm]
        assert np.all(P[:, j] >= lo - 1e-9), nm
        assert np.all(P[:, j] <= hi + 1e-9), nm


def test_group_sum_coverage():
    # стратификация тотала группы: края диапазона [2, 5] достижимы планом
    spec = _spec()
    P = spec.decode(spec.sample_z(400, seed=2))
    stab = P[:, STAB_COLS].sum(axis=1)
    assert np.all(stab >= 2.0 - 1e-9) and np.all(stab <= 5.0 + 1e-9)
    assert stab.min() < 2.15 and stab.max() > 4.85


# ----------------------------------------------------------------------
# encode: roundtrip и отказ на рецепте вне спеки
# ----------------------------------------------------------------------
def test_roundtrip_z_p_z():
    spec = _spec()
    Z = spec.sample_z(50, seed=3)
    P = spec.decode(Z)
    Z2 = spec.encode(P)
    np.testing.assert_allclose(Z2, Z, atol=1e-9)
    # и p → z → p (исторический рецепт в phr)
    P2 = spec.decode(spec.encode(P))
    np.testing.assert_allclose(P2, P, atol=1e-9)


def test_roundtrip_single_point_1d():
    spec = _spec()
    z = spec.sample_z(1, seed=4)[0]
    p = spec.decode(z)
    assert p.ndim == 1 and p.shape == (7,)
    np.testing.assert_allclose(spec.encode(p), z, atol=1e-9)


def test_encode_rejects_recipe_outside_spec():
    spec = _spec()
    p = spec.decode(spec.sample_z(1, seed=5)[0])
    bad = p.copy()
    bad[1] = 100.0                                 # plasticizer вне [40, 60]
    with pytest.raises(ValueError, match="plasticizer"):
        spec.encode(bad)
    bad2 = p.copy()
    bad2[0] = 90.0                                 # resin ≠ fixed 100
    with pytest.raises(ValueError, match="resin"):
        spec.encode(bad2)


# ----------------------------------------------------------------------
# Доли: Σ=1, ratio-инвариант, консервативный fraction-бокс
# ----------------------------------------------------------------------
def test_fractions_sum_ratio_and_bounds():
    spec = _spec()
    X = spec.sample_candidates(300, seed=6)
    assert X.shape == (300, 7)
    np.testing.assert_allclose(X.sum(axis=1), 1.0, atol=1e-9)
    # ratio_to-инвариант переживает нормировку p → x (доли пропорциональны phr)
    ratio = X[:, SBM_COL] / X[:, STAB_COLS].sum(axis=1)
    assert np.all(ratio >= 0.02 - 1e-9) and np.all(ratio <= 0.09 + 1e-9)
    lo, hi = spec.fraction_bounds()
    assert np.all(X >= lo - 1e-9) and np.all(X <= hi + 1e-9)


# ----------------------------------------------------------------------
# Мост в раннере (этап A: сэмплер-плагин, схема/модель нетронуты)
# ----------------------------------------------------------------------
def _runner_with_spec():
    spec = _spec()
    lo, hi = spec.fraction_bounds()
    schema = ProjectSchema.mixture_only(
        spec.component_names, lower=lo.tolist(), upper=hi.tolist(),
        model=_model())
    runner = MixtureProcessRunner(schema, _Oracle(), seed=0)
    return runner, spec


def test_runner_phr_candidates_valid_and_deterministic():
    runner, spec = _runner_with_spec()
    runner.set_phr_spec(spec)
    X1 = runner._phase_candidates(60, seed=7)
    X2 = runner._phase_candidates(60, seed=7)
    np.testing.assert_allclose(X1, X2)             # детерминизм по seed
    assert X1.shape == (60, 7)
    np.testing.assert_allclose(X1.sum(axis=1), 1.0, atol=1e-9)
    lo, hi = spec.fraction_bounds()
    assert np.all(X1 >= lo - 1e-9) and np.all(X1 <= hi + 1e-9)
    ratio = X1[:, SBM_COL] / X1[:, STAB_COLS].sum(axis=1)
    assert np.all(ratio >= 0.02 - 1e-9) and np.all(ratio <= 0.09 + 1e-9)


def test_runner_none_path_bit_identical():
    # phr_spec=None (дефолт) → прежний сэмплер; выключение спеки возвращает
    # тот же поток кандидатов, что у раннера, никогда её не видевшего
    runner_a, spec = _runner_with_spec()
    runner_b, _ = _runner_with_spec()
    runner_b.set_phr_spec(spec)
    runner_b.set_phr_spec(None)
    Xa = runner_a._phase_candidates(40, seed=11)
    Xb = runner_b._phase_candidates(40, seed=11)
    np.testing.assert_allclose(Xa, Xb)


def test_runner_mismatch_warns_and_falls_back():
    # схема шире спеки (лишний компонент): честный warning + прежний путь
    spec = _spec()
    names = spec.component_names + ["EXTRA"]
    schema = ProjectSchema.mixture_only(
        names, lower=[0.0] * 8, upper=[1.0] * 8, model=_model())
    runner = MixtureProcessRunner(schema, _Oracle(), seed=0)
    runner.set_phr_spec(spec)                      # компоненты известны схеме
    with pytest.warns(UserWarning, match="не совпадает"):
        X = runner._phase_candidates(20, seed=13)
    assert X.shape == (20, 8)
    np.testing.assert_allclose(X.sum(axis=1), 1.0, atol=1e-9)


def test_set_phr_spec_unknown_component_rejected():
    spec = _spec()
    schema = ProjectSchema.mixture_only(
        ["A", "B", "C"], lower=[0.0] * 3, upper=[1.0] * 3, model=_model())
    runner = MixtureProcessRunner(schema, _Oracle(), seed=0)
    with pytest.raises(KeyError, match="не найдены"):
        runner.set_phr_spec(spec)