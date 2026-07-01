# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 19 / §16.5 — замкнутый цикл кампании (сквозной, интеграционный).

Гейт-тест карты §16.6: сшивка §16.2–§16.4 в единый ПОВТОРЯЕМЫЙ контур
«старт минимальным набором → раунды добора → упор/новая цель/новый компонент →
эволюция схемы с миграцией → правка целей → снова раунды». Ключевые свойства
(ТЗ §16.5):

  * цикл ПОВТОРЯЕМ: расширение параметров/целей возвращает в раунды добора на ТОЙ
    ЖЕ общей базе (миграция, не пересборка проекта) — база растёт монотонно;
  * ОДНА модель физики на проект (канон §5/§12): словарь суррогатов один и тот же
    (по свойствам оракула), у веток своей модели нет;
  * spawn сиблинга (§8) работает на ОБЩЕМ пуле — новая продуктовая линия не плодит
    вторую модель.
"""
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.core.schema import ProjectSchema, VariableBlock, ModelSpec
from src.core.schema_evolution import known_constant, point_in_region
from src.design.block_model import build_model_terms
from src.optimize.desirability import DesirabilitySpec
from src.optimize.economic_stop import (STOP_CEIL, STOP_STAGNATION,
                                         STOP_NOT_ECONOMICAL)
from src.verification.mixture_process_truth import MultiMixtureProcessTruth
from src.apps.mixture_process_runner import MixtureProcessRunner
from src.apps.campaign import CampaignController

warnings.filterwarnings("ignore", category=ConvergenceWarning)

_LEGAL_REASONS = {None, STOP_CEIL, STOP_STAGNATION, STOP_NOT_ECONOMICAL}


def _schema():
    mix = VariableBlock.mixture(["A", "B", "C"])
    proc = VariableBlock.process(["T", "P"], lower=[0.0, 0.0], upper=[1.0, 1.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    return ProjectSchema.mixture_process(mix, proc, model=model)


def _price_fn(Xc):
    return np.ones(np.atleast_2d(Xc).shape[0], float)


def _runner_AB(n_seed=16):
    """Старт МИНИМАЛЬНЫМ набором: фаза {A,B}, T/P и C ещё закрыты (§16.5 старт)."""
    schema = _schema()
    p = build_model_terms(schema).p
    rng = np.random.default_rng(0)
    oracle = MultiMixtureProcessTruth(
        schema, {"p0": rng.normal(size=p), "p1": rng.normal(size=p),
                 "p2": rng.normal(size=p)}, noise_sd=0.0)
    r = MixtureProcessRunner(schema, oracle, seed=1, n_restarts=2,
                             baseline=[1 / 3, 1 / 3, 1 / 3, 0.5, 0.5])
    r.begin_phase(mixture_free=["A", "B"], process_free=[])
    r.seed_initial(n=n_seed, seed=1)
    return r


def _branch(r, bid="b1"):
    br = r.add_branch("main", {"p0": DesirabilitySpec("max", low=-5, high=5)},
                      branch_id=bid, budget=40)
    r.set_branch_cost(bid, _price_fn,
                      DesirabilitySpec("min", low=0.0, high=10.0),
                      rho_property="p1")
    br.volume, br.cost_exp, br.horizon = 1000.0, 0.01, 50.0
    return br


# ======================================================================
# Сквозной замкнутый контур: infill → evolve(+T,+C) → правка цели → infill
# ======================================================================
def test_closed_loop_infill_evolve_infill_repeatable():
    r = _runner_AB(n_seed=16)
    _branch(r, "b1")
    ctrl = CampaignController(r)

    sizes = [len(r.points)]

    # --- фаза 1: раунды добора на {A,B} ---
    ctrl.run_round("b1", n_points=3, n_candidates=200)
    ctrl.run_round("b1", n_points=3, n_candidates=200)
    sizes.append(len(r.points))
    assert r.current_schema_version == 1

    # --- упор → §16.2 эволюция схемы с ЯВНОЙ миграцией (та же база) ---
    ctrl.add_process_var("T", known_constant(0.5))     # v1 → v2
    ctrl.add_mixture_component("C")                    # v2 → v3
    assert r.current_schema_version == 3
    assert list(r.current_schema.mixture_names) == ["A", "B", "C"]
    sizes.append(len(r.points))                        # эволюция НЕ урезала базу

    # все точки мигрированы к v3 и лежат в области (миграция, не пересборка)
    assert len(r._migrated_points()) == len(r.points)
    assert all(point_in_region(p, r.current_schema)
               for p in r._migrated_points())

    # --- §16.3: новая цель клиента (правка целей на той же ветке) ---
    ctrl.set_desirability("b1", "p2",
                          DesirabilitySpec("target", low=-5, high=5, target=0.0))

    # --- фаза 2: снова раунды добора — уже на РАСШИРЕННОЙ схеме (повторяемость) ---
    ctrl.run_round("b1", n_points=3, n_candidates=200)
    ctrl.run_round("b1", n_points=3, n_candidates=200)
    sizes.append(len(r.points))

    # база росла МОНОТОННО через весь контур (общий пул, миграция не пересборка)
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]
    # ОДНА модель физики на проект: словарь суррогатов — по свойствам оракула
    assert set(r.surrogates) == {"p0", "p1", "p2"}
    # §4-стоп на итоговой схеме возвращает легальную причину
    dec = r.branch_stop_decision("b1", delta_d=0.05, ceil=0.9)
    assert dec.reason in _LEGAL_REASONS


# ======================================================================
# spawn сиблинга (§8) работает на ОБЩЕМ пуле — без второй модели физики
# ======================================================================
def test_spawn_sibling_shares_single_pool_and_model():
    r = _runner_AB(n_seed=16)
    _branch(r, "b1")
    ctrl = CampaignController(r)
    ctrl.run_round("b1", n_points=3, n_candidates=200)

    surro_ids_before = set(r.surrogates)
    n_before = len(r.points)

    # новая продуктовая линия = сиблинг с иной целью, тот же общий пул/модель
    spawn = ctrl.spawn_branch("b1", "sibling", child_id="b2",
                              new_goals={"p2": DesirabilitySpec("max", low=-5,
                                                                high=5)})
    assert spawn["child_id"] == "b2"
    ctrl.run_round("b2", n_points=3, n_candidates=200)

    assert len(r.points) == n_before + 3               # сиблинг долил в ОБЩИЙ пул
    assert r.origin_counts().get("branch:b2", 0) == 3
    # модель физики НЕ раздвоилась: тот же словарь свойств (одна модель на проект)
    assert set(r.surrogates) == surro_ids_before == {"p0", "p1", "p2"}
