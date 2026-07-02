# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 20 / §17.6 (Ш6) — эволюция схемы В ЛЮБОЙ МОМЕНТ из UI кампании.

Логика фасада (``add_process_var``/``add_mixture_component``/``add_response``/
``relax_bounds``/``restrict_bounds``) уже покрыта ``test_iteration19_campaign_schema``.
Здесь — ТОНКИЙ UI ``campaign_ui.render_schema_evolution``: headless AppTest на
вкладке «Кампания» раскрывает объявленную процесс-ось из полной схемы кнопкой —
версия схемы растёт, ось появляется в текущей схеме, общая база не урезается (И-1).
"""
import os
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.core.schema import ProjectSchema, VariableBlock, ModelSpec
from src.design.block_model import build_model_terms
from src.verification.mixture_process_truth import MultiMixtureProcessTruth
from src.apps.mixture_process_runner import MixtureProcessRunner
from src.apps.campaign import CampaignController

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _phased_ctrl(n_seed=14):
    """Раннер полной схемы {A,B,C}×{T,P}, стартующий урезанной фазой {A,B} (§16.2)."""
    mix = VariableBlock.mixture(["A", "B", "C"])
    proc = VariableBlock.process(["T", "P"], lower=[0.0, 0.0], upper=[1.0, 1.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    schema = ProjectSchema.mixture_process(mix, proc, model=model)
    p = build_model_terms(schema).p
    rng = np.random.default_rng(0)
    oracle = MultiMixtureProcessTruth(
        schema, {"p0": rng.normal(size=p), "p1": rng.normal(size=p)}, noise_sd=0.0)
    r = MixtureProcessRunner(schema, oracle, seed=1, n_restarts=2,
                             baseline=[1 / 3, 1 / 3, 1 / 3, 0.5, 0.5])
    r.begin_phase(mixture_free=["A", "B"], process_free=[])
    r.seed_initial(n=n_seed, seed=1)
    return CampaignController(r)


pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(_REPO, "src", "apps", "streamlit_app.py")


def _click(at, key):
    b = [w for w in at.button if w.key == key]
    assert b, f"кнопка {key} не найдена"
    b[0].click().run()


def test_schema_evolution_reveals_process_axis_from_ui():
    ctrl = _phased_ctrl()
    assert "T" not in ctrl.runner.current_schema.process_names
    v0 = ctrl.runner.current_schema_version
    n_base = len(ctrl.runner.points)

    at = AppTest.from_file(APP, default_timeout=360)
    at.session_state["campaign_ctrl"] = ctrl
    at.run()
    assert not at.exception

    # Ш6: раскрыть процесс-ось T (дефолт первый скрытый; миграция known_constant=0)
    _click(at, "camp_ev_proc_btn")
    assert not at.exception

    r = at.session_state["campaign_ctrl"].runner
    assert "T" in r.current_schema.process_names        # ось раскрыта
    assert r.current_schema_version > v0                # версия схемы поднята
    assert len(r.points) == n_base                      # И-1: база не урезана
