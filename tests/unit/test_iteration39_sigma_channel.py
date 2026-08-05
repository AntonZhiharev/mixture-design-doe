# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 39 — блокер 2 DECODE_LAYER_PROPOSAL («до сетапа»): σ-канал до
оптимизатора + замечания внешней ревизии 05.08.2026.

Проверяемый канон:

  * :class:`ChanceConstraint` — ``Pr(y_min ≤ y ≤ y_max) ≥ 1−α``; d-фактор
    ``clip(p/(1−α),0,1)`` гладкий (плоского нуля нет — направление возврата
    сохраняется), σ→0 вырождается в индикатор среднего;
  * ``sigma_predictors`` — σ-канал в ``optimize_desirability``: ограничение
    реально сдвигает оптимум с безусловного argmax к вероятностно допустимой
    области; отсутствие mean/σ-предиктора — явный KeyError;
  * :func:`hard_threshold_spec` — порог на среднее с ramp ШИРИНОЙ шума
    измерения (замечание 1 ревизии: узкий ramp = плоский нуль без градиента);
  * ``DesirabilityResult.binding_report`` — какое ограничение бинднулось и на
    скольких точках глобального пула («оптимум не найден» ≠ «оптимум
    запрещён»);
  * MoE: ``MoEPrediction.std`` — ПОЛНАЯ предиктивная σ (замечание 2 ревизии:
    Var = Σπσ² внутри + Σπ(μ−μ̄)² межэкспертное, неопределённость гейта);
  * clip_z (замечание 3 ревизии): идемпотентность как проекции
    ``clip(clip(z)) == clip(z)`` на ПРОИЗВОЛЬНОМ z; точка внутри области не
    двигается; асимметрия «upstream побеждает» (UV опускается, DINP не
    поднимается);
  * runner: ``optimize_xbest(chance_constraints=…)`` строит σ-канал из общих
    суррогатов автоматически (``predict(X).std``).
"""
import numpy as np
import pytest

from scipy.stats import norm

from src.core.schema import ModelSpec, ProjectSchema
from src.core.simplex import SimplexRegion
from src.design.phr_sampler import PhrSpec
from src.models.moe import MixtureOfExperts
from src.optimize.desirability import (ChanceConstraint, DesirabilitySpec,
                                       desirability_value,
                                       hard_threshold_spec,
                                       optimize_desirability)
from src.apps.mixture_process_runner import MixtureProcessRunner

# ----------------------------------------------------------------------
# ChanceConstraint: математика
# ----------------------------------------------------------------------
def test_chance_prob_two_sided_matches_normal_cdf():
    con = ChanceConstraint(y_min=0.0, y_max=1.0, alpha=0.05)
    p = con.prob(0.5, 0.5)
    expected = norm.cdf(1.0) - norm.cdf(-1.0)          # Φ(1) − Φ(−1)
    assert p[0] == pytest.approx(expected, abs=1e-12)


def test_chance_prob_one_sided():
    con = ChanceConstraint(y_max=1.0, alpha=0.05)      # y_min = −inf
    assert con.prob(1.0, 0.3)[0] == pytest.approx(0.5, abs=1e-12)
    assert con.prob(-5.0, 0.3)[0] == pytest.approx(1.0, abs=1e-9)


def test_chance_prob_sigma_zero_degenerates_to_indicator():
    con = ChanceConstraint(y_min=0.0, y_max=1.0, alpha=0.05)
    assert con.prob(0.5, 0.0)[0] == pytest.approx(1.0)
    assert con.prob(2.0, 0.0)[0] == pytest.approx(0.0)


def test_chance_dfactor_one_when_satisfied_and_smooth_when_not():
    con = ChanceConstraint(y_max=1.0, alpha=0.05)
    # глубоко в допустимой области — фактор ровно 1 (ограничение молчит)
    assert con.dfactor(0.0, 0.1)[0] == pytest.approx(1.0)
    # в недопустимой области фактор >0 и СТРОГО убывает по μ — есть
    # направление возврата (нет плоского нуля, в отличие от veto по среднему)
    mus = np.linspace(1.0, 2.0, 21)
    f = con.dfactor(mus, np.full_like(mus, 0.2))
    assert np.all(f > 0.0)
    assert np.all(np.diff(f) < 0.0)


def test_chance_constraint_validation():
    with pytest.raises(ValueError, match="alpha"):
        ChanceConstraint(y_max=1.0, alpha=0.0)
    with pytest.raises(ValueError, match="y_min < y_max"):
        ChanceConstraint(y_min=2.0, y_max=1.0)
    with pytest.raises(ValueError, match="finite"):
        ChanceConstraint()                              # оба конца бесконечны


# ----------------------------------------------------------------------
# hard_threshold_spec: ramp шириной шума измерения (замечание 1)
# ----------------------------------------------------------------------
def test_hard_threshold_spec_ge():
    spec = hard_threshold_spec(5.0, 0.2, "ge")          # ramp [4.8, 5.0]
    assert desirability_value(5.1, spec) == pytest.approx(1.0)
    assert desirability_value(4.79, spec) == pytest.approx(0.0)
    assert desirability_value(4.9, spec) == pytest.approx(0.5)


def test_hard_threshold_spec_le():
    spec = hard_threshold_spec(5.0, 0.2, "le")          # ramp [5.0, 5.2]
    assert desirability_value(4.9, spec) == pytest.approx(1.0)
    assert desirability_value(5.21, spec) == pytest.approx(0.0)
    assert desirability_value(5.1, spec) == pytest.approx(0.5)


def test_hard_threshold_spec_zero_ramp_rejected():
    # нулевой ramp = плоский нуль без направления возврата — явный отказ
    with pytest.raises(ValueError, match="ramp"):
        hard_threshold_spec(5.0, 0.0, "ge")
    with pytest.raises(ValueError, match="direction"):
        hard_threshold_spec(5.0, 0.2, "between")


# ----------------------------------------------------------------------
# optimize_desirability: σ-канал сдвигает оптимум, binding-отчёт
# ----------------------------------------------------------------------
def _region2() -> SimplexRegion:
    return SimplexRegion(lower=[0.0, 0.0], upper=[1.0, 1.0])


def _pred_y(X):
    return np.atleast_2d(np.asarray(X, float))[:, 0]     # максимизируем x0


def _pred_dE(X):
    return 2.0 * np.atleast_2d(np.asarray(X, float))[:, 0]


def _sig_const(s):
    return lambda X: np.full(len(np.atleast_2d(X)), float(s))


GOAL = {"y": DesirabilitySpec("max", low=0.0, high=1.0)}


def test_optimizer_without_constraint_goes_to_corner():
    res = optimize_desirability(_region2(), {"y": _pred_y}, GOAL,
                                n_candidates=800, refine_iters=200,
                                n_starts=3, seed=0)
    assert res.x[0] > 0.95
    # binding_report есть и без chance-constraints (спеки всегда репортятся)
    assert res.binding_report["n_pool"] >= 800
    assert "y" in res.binding_report["specs"]
    assert res.binding_report["chance"] == {}


def test_optimizer_chance_constraint_moves_optimum():
    # Pr(dE ≤ 1) ≥ 0.95 при dE = 2·x0, σ = 0.1 ⇒ жёсткая граница по среднему
    # x0 ≤ (1 − z_{0.95}·σ)/2 ≈ 0.418; мягкий штраф допускает лёгкий заступ.
    cc = {"dE": ChanceConstraint(y_max=1.0, alpha=0.05)}
    res = optimize_desirability(
        _region2(), {"y": _pred_y, "dE": _pred_dE}, GOAL,
        n_candidates=800, refine_iters=300, n_starts=3, seed=0,
        chance_constraints=cc, sigma_predictors={"dE": _sig_const(0.1)})
    assert 0.38 <= res.x[0] <= 0.47                     # не угол x0→1
    rep = res.binding_report["chance"]["dE"]
    assert rep["alpha"] == pytest.approx(0.05)
    assert rep["n_below"] > 0                           # часть пула запрещена
    assert rep["prob_at_optimum"] >= 0.93               # у границы 1−α


def test_optimizer_chance_constraint_sigma_zero_is_mean_threshold():
    # σ→0: вероятность = индикатор среднего ⇒ оптимум прижат к 2·x0 = 1
    cc = {"dE": ChanceConstraint(y_max=1.0, alpha=0.05)}
    res = optimize_desirability(
        _region2(), {"y": _pred_y, "dE": _pred_dE}, GOAL,
        n_candidates=800, refine_iters=300, n_starts=3, seed=0,
        chance_constraints=cc, sigma_predictors={"dE": _sig_const(0.0)})
    assert res.x[0] == pytest.approx(0.5, abs=0.02)
    assert res.binding_report["chance"]["dE"]["satisfied_at_optimum"]


def test_optimizer_chance_constraint_requires_predictors():
    cc = {"dE": ChanceConstraint(y_max=1.0, alpha=0.05)}
    with pytest.raises(KeyError, match="mean"):
        optimize_desirability(_region2(), {"y": _pred_y}, GOAL,
                              n_candidates=50, refine_iters=0, seed=0,
                              chance_constraints=cc,
                              sigma_predictors={"dE": _sig_const(0.1)})
    with pytest.raises(KeyError, match="sigma"):
        optimize_desirability(_region2(), {"y": _pred_y, "dE": _pred_dE},
                              GOAL, n_candidates=50, refine_iters=0, seed=0,
                              chance_constraints=cc)


def test_optimizer_none_constraints_bitwise_backward_compatible():
    kw = dict(n_candidates=300, refine_iters=100, n_starts=2, seed=7)
    r_old = optimize_desirability(_region2(), {"y": _pred_y}, GOAL, **kw)
    r_new = optimize_desirability(_region2(), {"y": _pred_y}, GOAL,
                                  chance_constraints=None,
                                  sigma_predictors=None, **kw)
    np.testing.assert_allclose(r_old.x, r_new.x)
    assert r_old.d_overall == pytest.approx(r_new.d_overall)


def test_binding_report_spec_veto_counts():
    # порог на среднее hard_threshold_spec: y ≥ 0.5 (ramp 0.1) — часть пула
    # (x0 < 0.4) под veto d=0; в оптимуме порог не активен (d=1)
    specs = {"y": DesirabilitySpec("max", low=0.0, high=1.0),
             "thr": hard_threshold_spec(0.5, 0.1, "ge")}
    preds = {"y": _pred_y, "thr": _pred_y}
    res = optimize_desirability(_region2(), preds, specs,
                                n_candidates=800, refine_iters=100,
                                n_starts=2, seed=0)
    rep = res.binding_report["specs"]["thr"]
    assert rep["n_veto"] > 0
    assert 0.0 < rep["frac_veto"] < 1.0
    assert rep["d_at_optimum"] == pytest.approx(1.0)


# ----------------------------------------------------------------------
# MoE: std — ПОЛНАЯ предиктивная σ (внутри + межэкспертное, замечание 2)
# ----------------------------------------------------------------------
def test_moe_std_includes_gate_disagreement():
    rng = np.random.default_rng(0)
    n = 30
    W = rng.dirichlet(np.ones(3), size=n)
    y = np.where(W[:, 0] > 1.0 / 3.0, 5.0, 0.0) + rng.normal(0, 0.05, n)
    moe = MixtureOfExperts(mean_model="linear", k_range=[2],
                           n_restarts=1, seed=0).fit(W, y)
    Xt = rng.dirichlet(np.ones(3), size=50)
    pred = moe.predict(Xt)
    # Var[y] = Σπσ² (внутри) + Σπ(μ_k−μ̄)² (гейт) — оба слагаемых в std
    np.testing.assert_allclose(pred.std ** 2,
                               pred.uncertainty + pred.disagreement,
                               atol=1e-10)
    # на границе зон ответственности межэкспертная компонента строго > 0:
    # только «внутри» переоценило бы Pr(ΔE≤max) именно там
    assert pred.disagreement.max() > 1e-6


# ----------------------------------------------------------------------
# clip_z (замечание 3): проекция, неподвижность внутри, upstream побеждает
# ----------------------------------------------------------------------
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
CAP_DICTS = [
    {"name": "resin", "mode": "fixed", "value": 100.0},
    {"name": "DINP", "mode": "absolute", "lo": 4.0, "hi": 14.0},
    {"name": "UV", "mode": "absolute", "lo": 0.05, "hi": 0.30,
     "cap_to": "DINP", "cap_ratio": 0.03},
    {"name": "filler", "mode": "absolute", "lo": 0.0, "hi": 30.0},
]


@pytest.mark.parametrize("dicts", [PVC_DICTS, CAP_DICTS])
def test_clip_z_projection_idempotent_on_arbitrary_z(dicts):
    # clip(clip(z)) == clip(z) на ПРОИЗВОЛЬНОМ (дико невалидном) z: renorm
    # share-осей в топопорядке не имеет права заново нарушить upstream-cap
    spec = PhrSpec.from_dicts(dicts)
    rng = np.random.default_rng(1)
    Z = spec.sample_z(200, seed=2)
    lo, hi = spec.z_bounds()
    wild = Z + rng.normal(0.0, 3.0, size=Z.shape) * (hi - lo)
    Zc = spec.clip_z(wild)
    np.testing.assert_allclose(spec.clip_z(Zc), Zc, atol=1e-9)


@pytest.mark.parametrize("dicts", [PVC_DICTS, CAP_DICTS])
def test_clip_z_interior_point_not_moved(dicts):
    # допустимая точка НЕ двигается — защита от тихого смещения на границу
    spec = PhrSpec.from_dicts(dicts)
    Z = spec.sample_z(200, seed=3)
    np.testing.assert_allclose(spec.clip_z(Z), Z, atol=1e-12)


def test_clip_z_upstream_wins_over_downstream():
    # конфликт «УФ выше потолка 0.03·DINP»: clip ОПУСКАЕТ УФ (downstream),
    # референс DINP (upstream) сохраняется — асимметрия по построению
    spec = PhrSpec.from_dicts(CAP_DICTS)
    z = spec.sample_z(1, seed=4)[0].copy()
    z[spec.z_names.index("DINP")] = 4.0                 # потолок = 0.12
    z[spec.z_names.index("UV")] = 0.30
    zc = spec.clip_z(z)
    assert zc[spec.z_names.index("UV")] == pytest.approx(0.12)
    assert zc[spec.z_names.index("DINP")] == pytest.approx(4.0)


# ----------------------------------------------------------------------
# Runner: σ-канал из общих суррогатов (optimize_xbest)
# ----------------------------------------------------------------------
class _Oracle2:
    """ratio = UV/DINP (цель), dE = 10·UV (ограничение)."""
    property_names = ["ratio", "dE"]

    def evaluate(self, Xc):
        Xc = np.atleast_2d(np.asarray(Xc, float))
        ratio = Xc[:, 2] / (Xc[:, 1] + 1e-12)
        dE = 10.0 * Xc[:, 2]
        return np.column_stack([ratio, dE])


def _runner():
    spec = PhrSpec.from_dicts(CAP_DICTS)
    lo, hi = spec.fraction_bounds()
    schema = ProjectSchema.mixture_only(
        spec.component_names, lower=lo.tolist(), upper=hi.tolist(),
        model=ModelSpec(cross_level="additive", mixture_order="quadratic"))
    runner = MixtureProcessRunner(schema, _Oracle2(), seed=0, n_restarts=1)
    runner.seed_initial(12)
    return runner


def test_runner_optimize_xbest_builds_sigma_channel():
    runner = _runner()
    br = runner.add_branch("uv", {"ratio": DesirabilitySpec(
        "max", low=0.0, high=0.08)}, budget=4)
    cc = {"dE": ChanceConstraint(y_max=5.0, alpha=0.2)}   # dE НЕ в goal ветки
    res = runner.optimize_xbest(br.id, n_candidates=200, refine_iters=20,
                                n_starts=1, chance_constraints=cc)
    rep = res.binding_report["chance"]["dE"]
    assert rep["alpha"] == pytest.approx(0.2)
    assert 0.0 <= rep["prob_at_optimum"] <= 1.0
    assert "dE" in res.properties                        # mean достроен


def test_runner_optimize_xbest_unknown_constraint_raises():
    runner = _runner()
    br = runner.add_branch("uv", {"ratio": DesirabilitySpec(
        "max", low=0.0, high=0.08)}, budget=4)
    with pytest.raises(KeyError, match="chance-constraint"):
        runner.optimize_xbest(br.id, n_candidates=50, refine_iters=0,
                              n_starts=1, chance_constraints={
                                  "nope": ChanceConstraint(y_max=1.0)})