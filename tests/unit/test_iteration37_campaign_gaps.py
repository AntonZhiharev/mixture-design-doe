# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 37 / Закрытие 5 пунктов скрин-аудита кампании (05.08.2026).

Проверяемый канон:

  1. **Аугментация переиспользует точки фазы 1** — ``propose_augment``
     (greedy maximin от existing) + ``propose_seed`` при непустой базе
     пристёгивает план к существующим точкам, а не генерирует с нуля;
     на пустой базе поведение прежнее (бит-в-бит с ``_phase_candidates``).
  2. **Индикатор кампании/фазы в метаданных точки** — ``origin_tag``
     несёт ``campaign`` (метка), ``spec_hash`` (геометрия активной
     phr-спеки), ``schema_version`` (фаза) и ``block`` (партия).
  3. **Квантование nominal → actual** — ``PhrSpec.quantize_recipe``:
     снап к δ-сетке весов внутри границ, «после округления точка всё ещё
     в границах» проверяется явно; диапазон уже шага δ — violation.
  4. **Покрытие обязательных 2D-пар** — pair-coverage гейт в preflight
     (``set_preflight_pairs`` / ``preflight(pairs=…)``), включая ось-сумму.
  5. **PROCESS сэмплится Sobol, не iid uniform** — низкая дискрепанция
     process-куба пулов кандидатов, детерминизм по seed сохранён.
"""
import numpy as np
import pytest

from src.core.schema import ModelSpec, ProjectSchema, VariableBlock
from src.design.phr_sampler import PhrSpec
from src.apps.mixture_process_runner import MixtureProcessRunner


class _Oracle:
    property_names = ["y1"]

    def evaluate(self, Xc):
        Xc = np.atleast_2d(np.asarray(Xc, float))
        y = (3.0 * Xc[:, 0] + 2.0 * Xc[:, 1]
             + 0.8 * np.sin(9.0 * Xc[:, 0] * Xc[:, 1]))
        return y.reshape(-1, 1)


def _model():
    return ModelSpec(cross_level="additive", mixture_order="quadratic")


def _runner_mix3(seed=0):
    schema = ProjectSchema.mixture_only(
        ["A", "B", "C"], lower=[0.0] * 3, upper=[1.0] * 3, model=_model())
    return MixtureProcessRunner(schema, _Oracle(), seed=seed, n_restarts=2)


def _runner_mix_proc(seed=0):
    mix = VariableBlock.mixture(["A", "B", "C"], lower=[0.0] * 3,
                                upper=[1.0] * 3)
    proc = VariableBlock.process(["T", "Pp"], lower=[0.0, 0.0],
                                 upper=[1.0, 1.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    schema = ProjectSchema.mixture_process(mix, proc, model=model)
    return MixtureProcessRunner(schema, _Oracle(), seed=seed, n_restarts=2)


def _min_dist_to(base: np.ndarray, pts: np.ndarray) -> float:
    d2 = ((pts[:, None, :] - base[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(d2.min(axis=1)).min())


# ======================================================================
# П.1 — аугментация переиспользует существующие точки
# ======================================================================
def test_propose_seed_empty_base_unchanged():
    """Пустая база: propose_seed = прежний путь бит-в-бит."""
    r = _runner_mix3()
    np.testing.assert_allclose(r.propose_seed(8, seed=7),
                               r._phase_candidates(8, 7))


def test_propose_augment_fills_holes_vs_from_scratch():
    r = _runner_mix3(seed=1)
    r.seed_initial(10, seed=2)
    base = np.asarray(r.X, float)
    aug = r.propose_augment(6, seed=5)
    assert aug.shape == (6, 3)
    np.testing.assert_allclose(aug.sum(axis=1), 1.0, atol=1e-9)
    # план «с нуля» тем же seed — baseline для сравнения
    raw = r._phase_candidates(6, 5)
    assert _min_dist_to(base, aug) >= _min_dist_to(base, raw)
    # новые точки не дублируют существующие
    assert _min_dist_to(base, aug) > 1e-3


def test_propose_seed_reuses_after_phase_expansion():
    """Ядро staged-сценария скрина: +компонент C → план фазы 2 пристёгивается
    к точкам фазы 1 (лежащим на грани C=0), а не генерируется с нуля."""
    r = _runner_mix3(seed=3)
    r.begin_phase(["A", "B"])
    r.seed_initial(8, seed=4)
    r.augment_phase_mixture(["C"])                 # фаза 2: симплекс A+B+C=1
    assert r.q == 3
    base = np.asarray(r.X, float)                  # мигрированные: C=0
    np.testing.assert_allclose(base[:, 2], 0.0, atol=1e-12)

    X2 = r.propose_seed(6, seed=9)                 # база непуста → augment-путь
    assert X2.shape == (6, 3)
    np.testing.assert_allclose(X2, r.propose_augment(6, seed=9))
    raw = r._phase_candidates(6, 9)
    assert _min_dist_to(base, X2) >= _min_dist_to(base, raw)
    # maximin от грани C=0 уводит новые точки в новое измерение
    assert X2[:, 2].max() > 0.2

    # reuse_existing=False — прежнее поведение принудительно
    np.testing.assert_allclose(r.propose_seed(6, seed=9,
                                              reuse_existing=False), raw)


# ======================================================================
# П.2 — индикатор кампании/фазы/геометрии в метаданных точки
# ======================================================================
SPEC_DICTS = [
    {"name": "resin", "mode": "fixed", "value": 100.0},
    {"name": "DINP", "mode": "absolute", "lo": 4.0, "hi": 14.0},
    {"name": "UV", "mode": "absolute", "lo": 0.05, "hi": 0.30,
     "cap_to": "DINP", "cap_ratio": 0.03},
    {"name": "SBM", "mode": "absolute", "lo": 0.07, "hi": 0.45},
]


def _runner_with_spec(seed=0):
    spec = PhrSpec.from_dicts(SPEC_DICTS)
    lo, hi = spec.fraction_bounds()
    schema = ProjectSchema.mixture_only(
        spec.component_names, lower=lo.tolist(), upper=hi.tolist(),
        model=_model())
    r = MixtureProcessRunner(schema, _Oracle(), seed=seed, n_restarts=2)
    r.set_phr_spec(spec)
    return r, spec


def test_point_metadata_has_campaign_phase_spec_hash_block():
    r, spec = _runner_with_spec()
    r.set_campaign_label("PVC-2026-K1")
    r.seed_initial(6, seed=1)
    for p in r.points:
        tag = p.origin_tag
        assert tag["campaign"] == "PVC-2026-K1"
        assert tag["spec_hash"] == spec.spec_hash()
        assert tag["schema_version"] == r.current_schema_version   # фаза
        assert "block" in tag                                       # партия


def test_metadata_without_label_and_spec_backcompat():
    r = _runner_mix3()
    r.seed_initial(5, seed=1)
    for p in r.points:
        assert "campaign" not in p.origin_tag
        assert "spec_hash" not in p.origin_tag


# ======================================================================
# П.3 — квантование: nominal vs actual, границы после округления
# ======================================================================
def test_quantize_snaps_to_grid_ok():
    spec = PhrSpec.from_dicts(SPEC_DICTS)
    # порядок листьев: resin, DINP, UV, SBM
    rep = spec.quantize_recipe([100.0, 7.503, 0.214, 0.20], 0.02)
    assert rep.ok, rep.violations
    np.testing.assert_allclose(rep.p_actual, [100.0, 7.50, 0.22, 0.20],
                               atol=1e-9)
    assert rep.moved_max == pytest.approx(0.006, abs=1e-9)
    np.testing.assert_allclose(rep.p_nominal, [100.0, 7.503, 0.214, 0.20])


def test_quantize_snaps_inward_at_lower_bound():
    """UV на нижней границе 0.05: ближайший узел сетки 0.04 вне границ —
    снап ВНУТРЬ интервала (0.06), без violation."""
    spec = PhrSpec.from_dicts(SPEC_DICTS)
    rep = spec.quantize_recipe([100.0, 8.0, 0.05, 0.20], 0.02)
    assert rep.ok, rep.violations
    assert rep.p_actual[2] == pytest.approx(0.06)


def test_quantize_flags_axis_narrower_than_grid():
    """Интервал без единого узла δ-сетки — ось нечитаема прямой навеской."""
    spec = PhrSpec.from_dicts([
        {"name": "resin", "mode": "fixed", "value": 100.0},
        {"name": "narrow", "mode": "absolute", "lo": 0.05, "hi": 0.07},
    ])
    rep = spec.quantize_recipe([100.0, 0.06], 0.04)   # узлы 0.04/0.08 вне
    assert not rep.ok
    assert any("narrow" in v and "премикс" in v for v in rep.violations)


def test_quantize_flags_fixed_off_grid():
    """fixed-значение вне δ-сетки: навесить точно невозможно — violation."""
    spec = PhrSpec.from_dicts([
        {"name": "resin", "mode": "fixed", "value": 100.0},
        {"name": "SA", "mode": "fixed", "value": 0.05},
        {"name": "DINP", "mode": "absolute", "lo": 4.0, "hi": 14.0},
    ])
    rep = spec.quantize_recipe([100.0, 0.05, 8.0], 0.02)
    assert not rep.ok
    assert any("SA" in v for v in rep.violations)


def test_quantize_validates_input():
    spec = PhrSpec.from_dicts(SPEC_DICTS)
    with pytest.raises(ValueError, match="delta_phr"):
        spec.quantize_recipe([100.0, 8.0, 0.1, 0.2], 0.0)
    with pytest.raises(ValueError, match="компонентов"):
        spec.quantize_recipe([100.0, 8.0], 0.02)


# ======================================================================
# П.4 — покрытие обязательных 2D-пар (pair-coverage гейт preflight)
# ======================================================================
def test_pair_coverage_good_plan_passes():
    r = _runner_mix_proc(seed=2)
    r.set_preflight_pairs([("T", "Pp"), (["A", "B"], "T")])
    X = r.propose_seed(24, seed=11)
    rep = r.preflight(X)
    assert len(rep.pair_coverage) == 2
    assert rep.pair_ok, rep.failures
    # строки пар присутствуют в таблице показа
    assert any("пара" in row["Проверка"] for row in rep.rows())


def test_pair_coverage_degenerate_plan_fails():
    r = _runner_mix_proc(seed=2)
    X = r.propose_seed(24, seed=11)
    Xbad = X.copy()
    Xbad[:, 3] = 0.5                       # T заморожена → 2D-пара не покрыта
    rep = r.preflight(Xbad, pairs=[("T", "Pp")])
    assert not rep.pair_ok
    assert any("покрытие пары" in f for f in rep.failures)
    assert rep.pair_coverage[0].coverage < 0.6


def test_set_preflight_pairs_validates_names():
    r = _runner_mix_proc()
    with pytest.raises(KeyError, match="не найдена"):
        r.set_preflight_pairs([("T", "GHOST")])
    with pytest.raises(ValueError, match="2 элемента"):
        r.set_preflight_pairs([("T",)])


def test_pair_with_axis_outside_phase_is_dropped():
    """Пара с осью вне текущей фазы выпадает из проверки (как группы)."""
    r = _runner_mix3(seed=3)
    r.begin_phase(["A", "B"])              # C вне фазы 1
    r.set_preflight_pairs([("A", "C")])
    X = r.propose_seed(12, seed=4)
    rep = r.preflight(X)
    assert rep.pair_coverage == []         # пара не проверялась
    assert rep.pair_ok


# ======================================================================
# П.5 — PROCESS: Sobol вместо iid uniform
# ======================================================================
def test_process_cube_low_discrepancy_and_bounds():
    from scipy.stats import qmc
    r = _runner_mix_proc(seed=5)
    X = r._phase_candidates(64, seed=3)
    proc = X[:, 3:]
    assert np.all(proc >= 0.0) and np.all(proc <= 1.0)
    iid = np.random.default_rng(3).uniform(size=(64, 2))
    assert qmc.discrepancy(proc) < qmc.discrepancy(iid)


def test_process_cube_deterministic_by_seed():
    r = _runner_mix_proc(seed=5)
    np.testing.assert_allclose(r._phase_candidates(20, seed=8),
                               r._phase_candidates(20, seed=8))


def test_process_cube_used_on_phr_path_too():
    """phr-путь _phase_candidates тоже получает Sobol-process."""
    from scipy.stats import qmc
    spec = PhrSpec.from_dicts(SPEC_DICTS)
    lo, hi = spec.fraction_bounds()
    mix = VariableBlock.mixture(spec.component_names, lower=lo.tolist(),
                                upper=hi.tolist())
    proc = VariableBlock.process(["T", "Pp"], lower=[0.0, 0.0],
                                 upper=[1.0, 1.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    schema = ProjectSchema.mixture_process(mix, proc, model=model)
    r = MixtureProcessRunner(schema, _Oracle(), seed=0, n_restarts=2)
    r.set_phr_spec(spec)
    X = r._phase_candidates(64, seed=7)
    assert X.shape == (64, spec.q + 2)
    pz = X[:, spec.q:]
    iid = np.random.default_rng(7).uniform(size=(64, 2))
    assert qmc.discrepancy(pz) < qmc.discrepancy(iid)