# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 32 / Preflight-диагностика плана ДО прогона (DECODE_LAYER_PROPOSAL, шаг 1).

Вырожденный план (тесный phr-бокс, слипшиеся компоненты, дыры покрытия суммы
группы, n<p) должен ловиться ДЁШЕВО и ДО измерений. Проверяемый канон iter32:

  * гейты ОТНОСИТЕЛЬНЫЕ к reference-пулу той же области (классические
    абсолютные cond<30/VIF<5/|corr|<0.30 в долях Шеффе неприменимы —
    эмпирика: хороший план n=24 при q=6 имеет cond≈2000, VIF≈1e5, corr≈0.9);
  * хороший план проходит все гейты; тесный бокс валит cond/VIF/blind/coverage;
    слипшаяся пара (DL≡SBM) валит rank и corr (пара именуется); n<p валит rank;
  * runner.preflight — read-only (база/суррогаты не меняются), детерминирован
    по seed, валидирует размерность;
  * план БЕЗ стратификации при заданных проектных группах честно проваливает
    coverage (сумма ниши не покрыта — ровно наблюдение iter31);
  * UI-хелперы (caption/details) — чистые, без Streamlit-рендера.
"""
import numpy as np
import pytest

from src.core.schema import ModelSpec, ProjectSchema
from src.core.simplex import SimplexRegion
from src.design.block_model import build_model_terms
from src.design.preflight import (PreflightThresholds, preflight_design)
from src.apps.mixture_process_runner import MixtureProcessRunner

# Сценарий recheck3 (как в iter31): BASE, FILLER + 4 конкурента одной ниши
NAMES6 = ["BASE", "FILLER", "PBNK", "CPE", "DL", "SBM"]
LO6 = [0.20, 0.05, 0.00, 0.00, 0.00, 0.00]
UP6 = [0.80, 0.50, 0.10, 0.10, 0.10, 0.10]
SOFT_IDX = [2, 3, 4, 5]
SOFT_NAMES = ["PBNK", "CPE", "DL", "SBM"]


class _Oracle:
    property_names = ["modulus"]

    def evaluate(self, Xc):
        Xc = np.atleast_2d(np.asarray(Xc, float))
        r = Xc[:, SOFT_IDX].sum(axis=1)
        return (100.0 - 250.0 * r + 300.0 * r ** 2).reshape(-1, 1)


def _schema():
    return ProjectSchema.mixture_only(
        NAMES6, lower=LO6, upper=UP6,
        model=ModelSpec(cross_level="additive", mixture_order="quadratic"))


def _region():
    return SimplexRegion(lower=LO6, upper=UP6, names=NAMES6)


def _ref(n=512, seed=99):
    return _region().random_points(n, seed=seed, groups=[SOFT_IDX])


def _runner(seed=0):
    return MixtureProcessRunner(_schema(), _Oracle(), seed=seed, n_restarts=2)


# ----------------------------------------------------------------------
# Ядро: preflight_design (чистая функция)
# ----------------------------------------------------------------------
def test_good_design_passes_all_gates():
    X = _region().random_points(24, seed=1, groups=[SOFT_IDX])
    rep = preflight_design(_schema(), X, _ref(), groups=[SOFT_IDX])
    assert rep.passed, f"хороший план не прошёл: {rep.failures}"
    assert rep.rank == rep.rank_ref == build_model_terms(_schema()).p
    assert rep.failures == []


def test_tiny_box_fails_cond_vif_blind_coverage():
    """Тесный бокс (аналог phr-проекции) — главный сценарий вырождения."""
    tiny = SimplexRegion(lower=[0.44, 0.28, 0.05, 0.05, 0.05, 0.05],
                         upper=[0.48, 0.32, 0.07, 0.07, 0.07, 0.07])
    X = tiny.random_points(24, seed=2)
    rep = preflight_design(_schema(), X, _ref(), groups=[SOFT_IDX])
    assert not rep.passed
    assert not rep.cond_ok
    assert not rep.vif_ok
    assert not rep.blind_ok
    assert not rep.coverage_ok
    assert len(rep.failures) >= 4


def test_collapsed_pair_fails_rank_and_corr_named():
    """DL≡SBM (слипшиеся оси) — точная коллинеарность: rank + именованная пара."""
    X = _region().random_points(24, seed=3)
    X[:, 4] = X[:, 5] = 0.5 * (X[:, 4] + X[:, 5])
    rep = preflight_design(_schema(), X, _ref())
    assert not rep.rank_ok
    assert not rep.corr_ok
    assert rep.corr_max_abs > 0.99
    assert set(rep.corr_pair) == {"DL", "SBM"}


def test_too_few_points_fails_rank():
    X = _region().random_points(12, seed=4)          # n=12 < p=21
    rep = preflight_design(_schema(), X, _ref())
    assert not rep.rank_ok
    assert rep.rank < rep.rank_ref


def test_unstratified_design_fails_group_coverage():
    """Равномерный план БЕЗ групп не покрывает сумму ниши (наблюдение iter31)."""
    X = _region().random_points(24, seed=5)          # без стратификации
    rep = preflight_design(_schema(), X, _ref(), groups=[SOFT_IDX])
    g = rep.group_coverage[0]
    assert g.names == SOFT_NAMES
    assert not g.ok, f"coverage={g.coverage:.2f} (ожидали провал < 0.8)"
    assert not rep.passed


def test_dimension_mismatch_errors():
    X = _region().random_points(10, seed=0)
    with pytest.raises(ValueError):                  # X_ref другой размерности
        preflight_design(_schema(), X, X[:, :5])
    with pytest.raises(ValueError):                  # схема ≠ размерность X
        preflight_design(_schema(), X[:, :5], X[:, :5])


def test_report_rows_and_summary_structure():
    X = _region().random_points(24, seed=1, groups=[SOFT_IDX])
    rep = preflight_design(_schema(), X, _ref(), groups=[SOFT_IDX])
    rows = rep.rows()
    assert len(rows) == 5 + 1                        # 5 базовых + 1 группа
    assert all({"Проверка", "План", "Допуск", "ОК"} <= set(r) for r in rows)
    s = rep.summary()
    assert s["passed"] is True and s["n"] == 24 and s["p"] == 21
    assert s["failures"] == []


def test_custom_thresholds_respected():
    """Ужесточение порогов делает гейты строже (пороги — параметры, не магия)."""
    X = _region().random_points(24, seed=1, groups=[SOFT_IDX])
    strict = PreflightThresholds(cond_factor=1.0, vif_factor=1.0)
    rep = preflight_design(_schema(), X, _ref(), thresholds=strict)
    assert not (rep.cond_ok and rep.vif_ok)          # n=24 хуже ref n=512


# ----------------------------------------------------------------------
# Runner: preflight (read-only, детерминизм, валидация, locked-фаза)
# ----------------------------------------------------------------------
def test_runner_preflight_good_seed_passes():
    r = _runner()
    r.set_mixture_sampling_groups([SOFT_NAMES])
    X = r.propose_seed(24, seed=11)
    rep = r.preflight(X)
    assert rep.passed, f"seed-план не прошёл: {rep.failures}"
    assert rep.group_coverage and rep.group_coverage[0].ok


def test_runner_preflight_read_only_and_deterministic():
    r = _runner()
    r.set_mixture_sampling_groups([SOFT_NAMES])
    X = r.propose_seed(24, seed=11)
    n_before = len(r.points)
    a = r.preflight(X, seed=5)
    b = r.preflight(X, seed=5)
    c = r.preflight(X, seed=6)
    assert len(r.points) == n_before == 0            # read-only
    assert a.summary() == b.summary()                # детерминизм по seed
    assert a.cond_ref != c.cond_ref or a.eig_min_ref != c.eig_min_ref


def test_runner_preflight_dim_validation():
    r = _runner()
    with pytest.raises(ValueError):
        r.preflight(np.ones((4, 3)))


def test_runner_preflight_locked_component_phase():
    """Запертый компонент (SBM=0.05): структурное вырождение НЕ считается
    провалом — reference той же фазы вырожден так же (rank == rank_ref)."""
    lo = list(LO6); hi = list(UP6)
    lo[5] = hi[5] = 0.05
    schema = ProjectSchema.mixture_only(
        NAMES6, lower=lo, upper=hi,
        model=ModelSpec(cross_level="additive", mixture_order="quadratic"))
    r = MixtureProcessRunner(schema, _Oracle(), seed=0, n_restarts=2)
    r.set_mixture_sampling_groups([SOFT_NAMES])
    X = r.propose_seed(30, seed=4)
    rep = r.preflight(X)
    assert rep.rank == rep.rank_ref < build_model_terms(schema).p
    assert rep.passed, f"locked-фаза: {rep.failures}"


def test_runner_preflight_flags_bad_plan():
    """Тот же runner честно валит план, сжатый в тесный под-бокс области."""
    r = _runner()
    r.set_mixture_sampling_groups([SOFT_NAMES])
    tiny = SimplexRegion(lower=[0.44, 0.28, 0.05, 0.05, 0.05, 0.05],
                         upper=[0.48, 0.32, 0.07, 0.07, 0.07, 0.07])
    rep = r.preflight(tiny.random_points(24, seed=2))
    assert not rep.passed
    assert rep.failures


# ----------------------------------------------------------------------
# UI: чистые хелперы (без Streamlit-рендера)
# ----------------------------------------------------------------------
def test_ui_preflight_caption_and_details():
    from src.apps.campaign_ui import (preflight_details_dataframe,
                                      seed_preflight_caption)
    r = _runner()
    r.set_mixture_sampling_groups([SOFT_NAMES])
    good = r.preflight(r.propose_seed(24, seed=11))
    cap = seed_preflight_caption(good)
    assert cap.startswith("🔎") and "информативен" in cap

    tiny = SimplexRegion(lower=[0.44, 0.28, 0.05, 0.05, 0.05, 0.05],
                         upper=[0.48, 0.32, 0.07, 0.07, 0.07, 0.07])
    bad = r.preflight(tiny.random_points(24, seed=2))
    cap_bad = seed_preflight_caption(bad)
    assert cap_bad.startswith("⚠️")
    for fail in bad.failures:
        assert fail in cap_bad

    df = preflight_details_dataframe(bad)
    assert list(df.columns) == ["Проверка", "План", "Допуск", "ОК"]
    assert len(df) == 6 and not df["ОК"].all()