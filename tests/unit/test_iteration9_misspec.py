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
"""Итерация 9 — детекторы misspecification (FinalCheckList Блок 7, §12).

Покрывает три канонных детектора:
  1. «вне всех режимов» (малые gₖ) + novelty (экстраполяция);
  2. триггер переразбиения K+1 (BIC-оптимальное число режимов ≠ текущему);
  3. стагнация ветки (d_best не растёт `patience` раундов).
Плюс интеграция в PipelineRunner (`diagnose_base`, флаг `stagnating`).
"""
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.models.diagnostics import (  # noqa: E402
    min_distance, nn_scale, novelty_ratio, gating_confidence,
    out_of_regime_mask, recommend_n_regimes, needs_recluster, diagnose)
from src.design.branches import Branch  # noqa: E402
from src.optimize.desirability import DesirabilitySpec  # noqa: E402
from src.apps.pipeline_runner import PipelineConfig, PipelineRunner  # noqa: E402


# ---- duck-typed суррогат для детерминированных юнит-тестов детекторов -----
class _Pred:
    def __init__(self, n):
        self.disagreement = np.zeros(n)
        self.std = np.ones(n)
        self.mean = np.zeros(n)


class StubMoE:
    def __init__(self, gating):
        self.G = np.atleast_2d(np.asarray(gating, float))
        self.K_ = self.G.shape[1]

    def gating_proba(self, X):
        n = np.atleast_2d(np.asarray(X, float)).shape[0]
        return self.G[:n]

    def predict(self, X):
        return _Pred(np.atleast_2d(np.asarray(X, float)).shape[0])


# ======================================================================
# Геометрия / novelty
# ======================================================================
def test_min_distance_and_nn_scale():
    ref = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    q = np.array([[0.0, 0.0], [5.0, 5.0]])
    d = min_distance(q, ref)
    assert d[0] == pytest.approx(0.0)
    assert d[1] == pytest.approx(np.sqrt(5.0 ** 2 + 4.0 ** 2))  # до (1,0)
    assert nn_scale(ref) == pytest.approx(1.0)


def test_novelty_ratio_flags_far_point():
    rng = np.random.default_rng(0)
    ref = 0.5 + 0.02 * rng.standard_normal((30, 3))   # плотное облако
    near = ref[:1]
    far = np.array([[10.0, -10.0, 5.0]])
    nov_near = novelty_ratio(near, ref)[0]
    nov_far = novelty_ratio(far, ref)[0]
    assert nov_far > 10.0 * max(nov_near, 1e-6)
    assert nov_far > 3.0          # экстраполяция по умолчанию


# ======================================================================
# Детектор 1: вне всех режимов (малые gₖ)
# ======================================================================
def test_out_of_regime_uses_gating_confidence():
    moe = StubMoE([[0.95, 0.05], [0.50, 0.50], [0.55, 0.45]])
    X = np.zeros((3, 2))
    conf = gating_confidence(moe, X)
    assert np.allclose(conf, [0.95, 0.50, 0.55])
    mask = out_of_regime_mask(moe, X, tau=0.6)
    assert mask.tolist() == [False, True, True]


def test_single_regime_never_out_of_regime():
    moe = StubMoE([[1.0], [1.0], [1.0]])      # K=1
    X = np.zeros((3, 1))
    assert np.allclose(gating_confidence(moe, X), 1.0)
    # даже при жёстком пороге одно-режимная модель не «вне режимов»
    assert not out_of_regime_mask(moe, X, tau=0.99).any()


def test_diagnose_report_summary_fields():
    moe = StubMoE([[0.95, 0.05], [0.4, 0.6], [0.45, 0.55]])
    ref = np.array([[0.5, 0.5], [0.4, 0.6], [0.6, 0.4]])
    X = np.array([[0.5, 0.5], [0.45, 0.55], [9.0, -9.0]])
    rep = diagnose(moe, X, ref=ref, tau=0.6, novelty_factor=3.0)
    s = rep.summary
    assert s["n"] == 3
    assert 0.0 <= s["frac_out_of_regime"] <= 1.0
    assert rep.extrapolation[-1]            # дальняя точка — экстраполяция
    assert rep.flagged.any()


# ======================================================================
# Детектор 2: триггер переразбиения K+1
# ======================================================================
def test_recommend_n_regimes_unimodal_vs_bimodal():
    rng = np.random.default_rng(1)
    uni = 1.0 + 0.05 * rng.standard_normal(60)
    bi = np.concatenate([0.0 + 0.03 * rng.standard_normal(40),
                         5.0 + 0.03 * rng.standard_normal(40)])
    assert recommend_n_regimes(uni, seed=0) == 1
    assert recommend_n_regimes(bi, seed=0) >= 2


def test_needs_recluster_triggers_when_K_outdated():
    rng = np.random.default_rng(2)
    bi = np.concatenate([0.0 + 0.03 * rng.standard_normal(40),
                         5.0 + 0.03 * rng.standard_normal(40)])
    one_regime = StubMoE([[1.0]])           # обучена как K=1
    need, rec = needs_recluster(one_regime, bi, seed=0)
    assert rec >= 2 and need is True


# ======================================================================
# Детектор 3: стагнация ветки
# ======================================================================
def _branch_with_dbest(seq):
    br = Branch(id="b1", name="t", goal={}, budget=10, satisfy_at=0.9)
    for i, d in enumerate(seq):
        br.d_best = float(d)
        br.history.append({"round": i + 1, "d_best": float(d)})
    br.refresh_status()
    return br


def test_branch_stagnation_detector():
    improving = _branch_with_dbest([0.1, 0.3, 0.5])
    assert not improving.is_stagnating(patience=2, min_delta=1e-3)

    plateau = _branch_with_dbest([0.5, 0.5, 0.5])
    assert plateau.is_stagnating(patience=2, min_delta=1e-3)

    # недостаточно истории — не стагнация
    short = _branch_with_dbest([0.5])
    assert not short.is_stagnating(patience=2)

    # достигшая цели ветка стагнацией не считается
    done = _branch_with_dbest([0.95, 0.95, 0.95])
    assert done.status == "satisfied"
    assert not done.is_stagnating(patience=2)


# ======================================================================
# Интеграция в PipelineRunner
# ======================================================================
def _runner(tmp_path):
    cfg = PipelineConfig(name="misspec", q=3, model="quadratic", noise_sd=0.2,
                         seed=7, n_restarts=3,
                         property_names=["Прочность", "Вязкость"])
    return PipelineRunner(cfg, tmp_path / "misspec")


def test_runner_diagnose_base(tmp_path):
    r = _runner(tmp_path)
    r.run_m2(simulate=True)
    r.run_m6()
    rep = r.diagnose_base()
    assert set(rep["per_property"]) == {"Прочность", "Вязкость"}
    for name, info in rep["per_property"].items():
        for key in ("frac_out_of_regime", "frac_extrapolation",
                    "frac_flagged", "current_K", "recommended_K",
                    "needs_recluster"):
            assert key in info
        assert info["current_K"] >= 1
    assert rep["n_query"] == len(r.design)


def test_branch_round_reports_stagnating_flag(tmp_path):
    r = _runner(tmp_path)
    r.run_m2(simulate=True)
    r.run_m6()
    col = np.asarray(r.Y, float)[:, 0]
    spec = DesirabilitySpec("max", low=float(col.min()), high=float(col.max()))
    bid = r.add_branch("ветка", {"Прочность": spec}, budget=3).id
    out = r.run_branch_round(bid, n_points=1)
    assert "stagnating" in out and isinstance(out["stagnating"], bool)
