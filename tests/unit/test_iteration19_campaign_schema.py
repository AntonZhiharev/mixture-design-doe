# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 19 / §16.2 — динамическая схема как ШТАТНАЯ операция кампании.

Гейт-тест карты §16.6: фасад ``CampaignController.add_process_var`` /
``add_mixture_component`` / ``add_response`` / ``relax_bounds`` /
``restrict_bounds`` поверх ядра (``augment_phase_*`` / ``move_region`` /
``evolve_schema``). Проверяем инварианты §16.2:

  * A0.6 — миграция НЕ молча: добавление ПЕРЕМЕННОЙ требует ЯВНОЙ политики;
  * И-1 — общая база не урезается ни одной операцией схемы;
  * версионирование — append переменной/отклика bump'ает версию; смена ТОЛЬКО
    границ (relax/restrict) — region-move БЕЗ bump (§15.2.4);
  * сквозной сценарий {A,B} → +T → +C: база цела, версия выросла, мигрированные
    точки в области, а выпавшие по сужению — честно исключены из активного pool,
    но ОСТАЮТСЯ в истории.
"""
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.core.schema import ProjectSchema, VariableBlock, ModelSpec, ResponseSpec
from src.core.schema_evolution import (known_constant, unknown,
                                        point_in_region)
from src.design.block_model import build_model_terms
from src.design.move_bounds import MOVE_RESTRICT
from src.verification.mixture_process_truth import MultiMixtureProcessTruth
from src.apps.mixture_process_runner import MixtureProcessRunner
from src.apps.campaign import CampaignController

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ----------------------------------------------------------------------
# Полная схема проекта {A,B,C} × {T,P}; фаза стартует урезанной (§16.2)
# ----------------------------------------------------------------------
def _schema():
    mix = VariableBlock.mixture(["A", "B", "C"])
    proc = VariableBlock.process(["T", "P"], lower=[0.0, 0.0], upper=[1.0, 1.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    return ProjectSchema.mixture_process(mix, proc, model=model)


def _runner(mixture_free=("A", "B"), process_free=(), n_seed=14, seed=1):
    schema = _schema()
    p = build_model_terms(schema).p
    rng = np.random.default_rng(0)
    oracle = MultiMixtureProcessTruth(
        schema, {"p0": rng.normal(size=p), "p1": rng.normal(size=p)},
        noise_sd=0.0)
    r = MixtureProcessRunner(schema, oracle, seed=seed, n_restarts=2,
                             baseline=[1 / 3, 1 / 3, 1 / 3, 0.5, 0.5])
    r.begin_phase(mixture_free=list(mixture_free),
                  process_free=list(process_free))
    r.seed_initial(n=n_seed, seed=seed)
    return r


def _all_in_region(r):
    return all(point_in_region(p, r.current_schema)
               for p in r._migrated_points())


# ======================================================================
# add_process_var — append process (bump версии) + A0.6 + И-1
# ======================================================================
def test_add_process_var_bumps_version_keeps_history():
    r = _runner()
    ctrl = CampaignController(r)
    v0 = r.current_schema_version
    n_hist = len(r.points)

    ctrl.add_process_var("T", known_constant(0.5))

    assert r.current_schema_version == v0 + 1          # append → bump
    assert "T" in r.current_schema.process_names
    assert len(r.points) == n_hist                     # И-1: база цела
    assert len(r._migrated_points()) == n_hist         # все мигрировали
    assert _all_in_region(r)


def test_add_process_var_requires_explicit_migration():
    r = _runner()
    ctrl = CampaignController(r)
    # A0.6: политика без ключа "policy" / не-dict — отвергается
    with pytest.raises(ValueError):
        ctrl.add_process_var("T", {"foo": 1})
    with pytest.raises(ValueError):
        ctrl.add_process_var("T", "known-constant")


def test_add_process_var_unknown_axis_rejected():
    r = _runner()
    ctrl = CampaignController(r)
    with pytest.raises(KeyError):
        ctrl.add_process_var("Z", known_constant(0.5))


# ======================================================================

# add_mixture_component — append компонента (Σ переопределяется) + миграция
# ======================================================================
def test_add_mixture_component_default_migration_and_history():
    r = _runner()
    ctrl = CampaignController(r)
    v0 = r.current_schema_version
    n_hist = len(r.points)

    ctrl.add_mixture_component("C")                    # дефолт known_constant(0.0)

    assert r.current_schema_version == v0 + 1
    assert "C" in r.current_schema.mixture_names
    assert len(r.points) == n_hist                     # И-1
    assert len(r._migrated_points()) == n_hist         # мигрировали на грань C=0
    assert _all_in_region(r)


def test_add_mixture_component_rejects_non_zero_migration():
    r = _runner()
    ctrl = CampaignController(r)
    with pytest.raises(ValueError):
        ctrl.add_mixture_component("C", known_constant(0.5))
    with pytest.raises(ValueError):
        ctrl.add_mixture_component("C", unknown())


def test_add_mixture_component_unknown_rejected():
    r = _runner()
    ctrl = CampaignController(r)
    with pytest.raises(KeyError):
        ctrl.add_mixture_component("Z")


# ======================================================================
# Сквозной сценарий §16.2: {A,B} → +T → +C (гейт)
# ======================================================================
def test_campaign_schema_flow_AB_then_T_then_C():
    r = _runner(mixture_free=("A", "B"), process_free=(), n_seed=16)
    ctrl = CampaignController(r)
    n_hist = len(r.points)
    assert r.current_schema_version == 1
    assert list(r.current_schema.mixture_names) == ["A", "B"]

    ctrl.add_process_var("T", known_constant(0.5))     # v1 → v2
    ctrl.add_mixture_component("C")                    # v2 → v3

    assert r.current_schema_version == 3               # версия выросла
    assert list(r.current_schema.mixture_names) == ["A", "B", "C"]
    assert "T" in r.current_schema.process_names
    assert len(r.points) == n_hist                     # база НЕ урезана (И-1)
    assert len(r._migrated_points()) == n_hist         # все мигрированы к v3
    assert _all_in_region(r)                           # и в области
    # суррогаты переобучены на расширенной составной координате
    assert set(r.surrogates) == {"p0", "p1"}


# ======================================================================
# relax/restrict — region-move БЕЗ bump; выпавшие в истории (И-1)
# ======================================================================
def test_restrict_bounds_no_version_bump_excludes_but_keeps_history():
    r = _runner(mixture_free=("A", "B"), n_seed=24)
    ctrl = CampaignController(r)
    v0 = r.current_schema_version
    n_hist = len(r.points)
    n_active0 = len(r._migrated_points())
    assert np.any([p.X["MIXTURE"][0] > 0.3 for p in r._migrated_points()])

    mv = ctrl.restrict_bounds("A", 0.0, 0.3)

    assert mv.move_type == MOVE_RESTRICT
    assert r.current_schema_version == v0              # region-move: без bump
    assert len(r.points) == n_hist                     # история цела (И-1)
    assert len(r._migrated_points()) < n_active0       # активный pool сжался
    assert all(p.X["MIXTURE"][0] <= 0.3 + 1e-9
               for p in r._migrated_points())


def test_relax_then_restrict_reversible():
    r = _runner(mixture_free=("A", "B"), n_seed=24)
    ctrl = CampaignController(r)
    n_active0 = len(r._migrated_points())
    ctrl.restrict_bounds("A", 0.0, 0.3)
    assert len(r._migrated_points()) < n_active0        # выпали
    ctrl.relax_bounds("A", 0.0, 1.0)                    # расширили обратно
    assert len(r._migrated_points()) == n_active0       # восстановились


# ======================================================================
# add_response — новый отклик bump'ает версию схемы (Y[new]=MISSING у старых)
# ======================================================================
def test_add_response_bumps_version():
    r = _runner()
    ctrl = CampaignController(r)
    v0 = r.current_schema_version
    n_hist = len(r.points)

    new = ctrl.add_response(ResponseSpec("newresp", kind="min"))

    assert new.version == v0 + 1
    assert r.current_schema_version == v0 + 1
    assert "newresp" in r.current_schema.response_names
    assert len(r.points) == n_hist                     # И-1
