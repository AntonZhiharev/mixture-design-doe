# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 38 / B1 decode-слоя (ревизия этапа B, DECODE_LAYER_PROPOSAL):
phr-геометрия в оптимизаторе desirability.

Дыра (аудит 05.08.2026): `optimize_desirability` был слеп к phr-геометрии на
ОБЕИХ стадиях — глобальный пул из бокса долей, refine с clip в бокс. Оптимум
уезжает в углы бокса, где cap-потолки (трапеция UV) и нарушаются: невалидный
x* — типичный исход, не редкий. Проверяемый канон iter38:

  * `PhrSpec.z_bounds` — статические границы z-осей (масштаб возмущений);
  * `PhrSpec.clip_z` — проекция z в область спеки ПО ПОСТРОЕНИЮ (не rejection):
    условный интервал cap-узлов в топопорядке, share-группы — clip +
    детерминированное перераспределение невязки Σ=1; идемпотентна на валидных;
  * `optimize_desirability(phr_spec=...)` — глобальный пул `sample_z → decode`,
    refine возмущением в z + clip_z + decode: результат допустим по построению
    (encode проходит), refine реально двигается у границы (у cap-потолка);
  * дефолтный путь демонстрирует баг (документируем): x* нарушает cap;
  * `optimize_xbest` раннера прокидывает активную phr-спеку (та же политика
    совпадения состава, что `_phase_candidates`: несовпадение — warning + бокс).
"""
import numpy as np
import pytest

from src.core.schema import ModelSpec, ProjectSchema
from src.core.simplex import SimplexRegion
from src.design.phr_sampler import PhrSpec
from src.optimize.desirability import DesirabilitySpec, optimize_desirability
from src.apps.mixture_process_runner import MixtureProcessRunner

# Спека iter33 (shares + ratio_to) — структурные тесты clip_z
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
SHARE_NAMES = ("Ca_st", "Zn_st", "ester")

# UV-трапеция (cap по референсу): бокс долей разрешает UV=0.30 при DINP=4,
# а потолок спеки — 0.03·DINP=0.12. Углы бокса phr-нелегальны — ровно кейс B1.
CAP_DICTS = [
    {"name": "resin", "mode": "fixed", "value": 100.0},
    {"name": "DINP", "mode": "absolute", "lo": 4.0, "hi": 14.0},
    {"name": "UV", "mode": "absolute", "lo": 0.05, "hi": 0.30,
     "cap_to": "DINP", "cap_ratio": 0.03},
    {"name": "filler", "mode": "absolute", "lo": 0.0, "hi": 30.0},
]
DINP_COL, UV_COL = 1, 2                     # в component_names / долях


def _model():
    return ModelSpec(cross_level="additive", mixture_order="quadratic")


def _fractions_to_phr(spec: PhrSpec, x: np.ndarray) -> np.ndarray:
    """Доли → phr для спек с fixed-листом resin=100 первым компонентом."""
    x = np.asarray(x, float).ravel()
    return x * (100.0 / x[0])


def _ratio_predictor(X):
    """Максимизируемый отклик: UV/DINP — тянет оптимум в запрещённый угол
    бокса (максимум ratio в боксе 0.30/4 = 0.075 ≫ потолка 0.03)."""
    X = np.atleast_2d(np.asarray(X, float))
    return X[:, UV_COL] / (X[:, DINP_COL] + 1e-12)


RATIO_GOAL = {"y": DesirabilitySpec("max", low=0.0, high=0.08)}


# ----------------------------------------------------------------------
# z_bounds
# ----------------------------------------------------------------------
def test_z_bounds_order_and_values():
    spec = PhrSpec.from_dicts(PVC_DICTS)
    lo, hi = spec.z_bounds()
    assert lo.shape == hi.shape == (spec.dim_z,)
    j = spec.z_names.index("plasticizer")
    assert (lo[j], hi[j]) == (40.0, 60.0)
    j = spec.z_names.index("SBM")
    assert (lo[j], hi[j]) == (0.02, 0.09)
    # у cap-узла z_bounds отдаёт СТАТИЧЕСКИЙ hi (динамику делает clip_z)
    spec_cap = PhrSpec.from_dicts(CAP_DICTS)
    lo, hi = spec_cap.z_bounds()
    j = spec_cap.z_names.index("UV")
    assert (lo[j], hi[j]) == (0.05, 0.30)


# ----------------------------------------------------------------------
# clip_z: допустимость по построению, идемпотентность, share-группы
# ----------------------------------------------------------------------
@pytest.mark.parametrize("dicts", [PVC_DICTS, CAP_DICTS])
def test_clip_z_valid_by_construction(dicts):
    # шумные z далеко за областью → после clip_z encode(decode(z)) проходит
    # (encode валидирует ВСЮ геометрию: границы осей, cap, fixed) и даёт
    # roundtrip тот же z
    spec = PhrSpec.from_dicts(dicts)
    rng = np.random.default_rng(0)
    Z = spec.sample_z(200, seed=1)
    lo, hi = spec.z_bounds()
    noise = rng.normal(0.0, 1.0, size=Z.shape) * (hi - lo)
    Zc = spec.clip_z(Z + noise)
    P = spec.decode(Zc)
    Zback = spec.encode(P)                 # не бросает ⇒ внутри геометрии
    np.testing.assert_allclose(Zback, Zc, atol=1e-9)


@pytest.mark.parametrize("dicts", [PVC_DICTS, CAP_DICTS])
def test_clip_z_idempotent_on_valid(dicts):
    spec = PhrSpec.from_dicts(dicts)
    Z = spec.sample_z(100, seed=2)
    np.testing.assert_allclose(spec.clip_z(Z), Z, atol=1e-9)


def test_clip_z_share_group_redistribution():
    spec = PhrSpec.from_dicts(PVC_DICTS)
    cols = [spec.z_names.index(nm) for nm in SHARE_NAMES]
    lo = np.array([0.2, 0.1, 0.1])
    hi = np.array([0.7, 0.5, 0.6])
    for blowup in (5.0, -5.0):             # избыток и дефицит суммы долей
        z = spec.sample_z(1, seed=3)[0].copy()
        for j in cols:
            z[j] = blowup
        zc = spec.clip_z(z)
        s = np.array([zc[j] for j in cols])
        assert abs(float(s.sum()) - 1.0) < 1e-9
        assert np.all(s >= lo - 1e-9) and np.all(s <= hi + 1e-9)


def test_clip_z_cap_is_conditional():
    # UV clip'ается к ПЕР-ТОЧЕЧНОМУ потолку 0.03·DINP, а не к статическому 0.30
    spec = PhrSpec.from_dicts(CAP_DICTS)
    z = spec.sample_z(1, seed=4)[0].copy()
    z[spec.z_names.index("DINP")] = 4.0     # cap = 0.12
    z[spec.z_names.index("UV")] = 0.30      # в статических границах, но > cap
    zc = spec.clip_z(z)
    assert zc[spec.z_names.index("UV")] == pytest.approx(0.12)


def test_clip_z_shapes_and_dim_error():
    spec = PhrSpec.from_dicts(CAP_DICTS)
    z1 = spec.clip_z(spec.sample_z(1, seed=5)[0])
    assert z1.ndim == 1 and z1.shape == (spec.dim_z,)
    Z2 = spec.clip_z(spec.sample_z(3, seed=5))
    assert Z2.shape == (3, spec.dim_z)
    with pytest.raises(ValueError, match="clip_z"):
        spec.clip_z(np.zeros(spec.dim_z + 1))


# ----------------------------------------------------------------------
# Оптимизатор: дефолтный путь ДОКУМЕНТИРУЕТ баг, phr-путь его закрывает
# ----------------------------------------------------------------------
def _region(spec: PhrSpec) -> SimplexRegion:
    lo, hi = spec.fraction_bounds()
    return SimplexRegion(lower=lo, upper=hi)


def test_optimizer_default_path_violates_cap():
    # регресс-документация B1: без phr_spec оптимум уезжает в угол бокса,
    # где UV > 0.03·DINP — рецепт вне спеки (encode бросает)
    spec = PhrSpec.from_dicts(CAP_DICTS)
    res = optimize_desirability(_region(spec), {"y": _ratio_predictor},
                                RATIO_GOAL, n_candidates=500,
                                refine_iters=200, n_starts=3, seed=0)
    p = _fractions_to_phr(spec, res.x)
    assert p[UV_COL] > 0.03 * p[DINP_COL] + 1e-6   # потолок нарушен
    # encode бросает: рецепт вне геометрии спеки (в углу бокса нарушается и
    # cap, и согласованность долей с тоталом — DINP выпадает из [4, 14])
    with pytest.raises(ValueError, match="encode"):
        spec.encode(p)


def test_optimizer_phr_path_respects_cap_and_refines_at_border():
    spec = PhrSpec.from_dicts(CAP_DICTS)
    res = optimize_desirability(_region(spec), {"y": _ratio_predictor},
                                RATIO_GOAL, n_candidates=500,
                                refine_iters=200, n_starts=3, seed=0,
                                phr_spec=spec)
    p = _fractions_to_phr(spec, res.x)
    spec.encode(p)                                  # допустим по построению
    ratio = p[UV_COL] / p[DINP_COL]
    assert ratio <= 0.03 + 1e-9
    # refine РАБОТАЕТ у границы (не рухнул, как rejection): оптимум прижат
    # к потолку трапеции (максимум UV/DINP при cap — это ratio = 0.03)
    assert ratio >= 0.027


def test_optimizer_phr_path_deterministic():
    spec = PhrSpec.from_dicts(CAP_DICTS)
    kw = dict(n_candidates=200, refine_iters=50, n_starts=2, seed=7,
              phr_spec=spec)
    r1 = optimize_desirability(_region(spec), {"y": _ratio_predictor},
                               RATIO_GOAL, **kw)
    r2 = optimize_desirability(_region(spec), {"y": _ratio_predictor},
                               RATIO_GOAL, **kw)
    np.testing.assert_allclose(r1.x, r2.x)
    assert r1.d_overall == pytest.approx(r2.d_overall)


def test_optimizer_phr_spec_q_mismatch_raises():
    spec = PhrSpec.from_dicts(CAP_DICTS)                  # q=4
    region3 = SimplexRegion(lower=[0.0] * 3, upper=[1.0] * 3)
    with pytest.raises(ValueError, match="phr_spec"):
        optimize_desirability(region3, {"y": _ratio_predictor}, RATIO_GOAL,
                              n_candidates=50, refine_iters=0, seed=0,
                              phr_spec=spec)


# ----------------------------------------------------------------------
# Раннер: optimize_xbest уважает активную phr-спеку
# ----------------------------------------------------------------------
class _OracleRatio:
    """Отклик = UV/DINP по ПОЛНЫМ долям (столбцы 1, 2 полной схемы)."""
    property_names = ["ratio"]

    def evaluate(self, Xc):
        Xc = np.atleast_2d(np.asarray(Xc, float))
        return (Xc[:, DINP_COL + 1] / (Xc[:, DINP_COL] + 1e-12)).reshape(-1, 1)


def test_runner_optimize_xbest_respects_phr_spec():
    spec = PhrSpec.from_dicts(CAP_DICTS)
    lo, hi = spec.fraction_bounds()
    schema = ProjectSchema.mixture_only(
        spec.component_names, lower=lo.tolist(), upper=hi.tolist(),
        model=_model())
    runner = MixtureProcessRunner(schema, _OracleRatio(), seed=0)
    runner.set_phr_spec(spec)
    runner.seed_initial(12)
    br = runner.add_branch("uv", {"ratio": DesirabilitySpec(
        "max", low=0.0, high=0.08)}, budget=4)
    res = runner.optimize_xbest(br.id, n_candidates=300, refine_iters=60,
                                n_starts=2)
    p = _fractions_to_phr(spec, res.x[:spec.q])
    spec.encode(p)                                  # phr-валидный argmax
    assert p[UV_COL] <= 0.03 * p[DINP_COL] + 1e-6


def test_runner_optimize_xbest_mismatch_warns_and_uses_box():
    # схема шире спеки (лишний компонент): warning + прежний путь (бокс)
    spec = PhrSpec.from_dicts(CAP_DICTS)
    names = spec.component_names + ["EXTRA"]
    schema = ProjectSchema.mixture_only(
        names, lower=[0.0] * 5, upper=[1.0] * 5, model=_model())
    runner = MixtureProcessRunner(schema, _OracleRatio(), seed=0)
    runner.set_phr_spec(spec)                       # компоненты известны схеме
    with pytest.warns(UserWarning, match="не совпадает"):
        runner.seed_initial(10)
    br = runner.add_branch("uv", {"ratio": DesirabilitySpec(
        "max", low=0.0, high=0.08)}, budget=4)
    with pytest.warns(UserWarning, match="optimize_xbest использует"):
        res = runner.optimize_xbest(br.id, n_candidates=100, refine_iters=10,
                                    n_starts=1)
    assert res.x.shape == (5,)                      # бокс-путь, размер схемы