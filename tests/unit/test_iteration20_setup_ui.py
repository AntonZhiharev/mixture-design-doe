# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 20 / §17.4 (Ш3b) — UI РЕАЛЬНОГО сетапа кампании + ручной seed.

Канон «логика+тест, потом UI»: логика ручного seed (§17.4 propose_seed/commit_seed)
уже покрыта (``test_iteration20_manual_seed``); здесь проверяем ТОНКИЙ UI —
``campaign_ui.build_setup_runner`` (чистая сборка раннера из формы) и форму
``render_setup_form`` + ручной seed-цикл ``render_seed_entry`` поверх реального
``MixtureProcessRunner``. Две части, как в существующих ``*_ui`` тестах:

  * ЧИСТЫЕ хелперы (``ManualOracle`` / ``build_setup_runner``) — без Streamlit;
  * headless AppTest: во вкладке «Кампания» собрать проект (mixture+процесс+
    отклики), предложить seed (read-only), заполнить Y демо-оракулом и
    зафиксировать — общая база растёт, суррогаты обучены, origin-тег «seed» (И-1).
"""
import os
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.apps.campaign import CampaignController
from src.apps.campaign_ui import ManualOracle, build_setup_runner

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ======================================================================
# Чистые хелперы (без Streamlit)
# ======================================================================
def _setup_runner(**kw):
    base = dict(mixture_names=["A", "B", "C"], process_names=["T", "P"],
                process_lower=[0.0, 0.0], process_upper=[1.0, 1.0],
                response_names=["strength", "gloss", "rho"], seed=1)
    base.update(kw)
    return build_setup_runner(**base)


def test_manual_oracle_is_deterministic_and_finite():
    o = ManualOracle(["a", "b"])
    X = np.random.default_rng(0).uniform(size=(4, 5))
    Y1, Y2 = o.evaluate(X), o.evaluate(X)
    assert Y1.shape == (4, 2)                 # столбец на свойство
    assert np.allclose(Y1, Y2)                # тот же вход ⇒ тот же выход
    assert np.all(np.isfinite(Y1))


def test_build_setup_runner_empty_base_manual_oracle():
    r = _setup_runner()
    assert r.q == 3 and r.dim == 5            # симплекс {A,B,C} × куб {T,P}
    assert list(r.property_names) == ["strength", "gloss", "rho"]
    assert len(r.points) == 0                 # база пуста — Y ещё не внесены
    assert not r.surrogates                   # суррогатов нет
    assert isinstance(r.oracle, ManualOracle)


def test_build_setup_runner_validates_inputs():
    with pytest.raises(ValueError):           # нет процесс-параметров (§17.4)
        _setup_runner(process_names=[], process_lower=[], process_upper=[])
    with pytest.raises(ValueError):           # границы процесса не по размеру
        _setup_runner(process_lower=[0.0], process_upper=[1.0, 1.0])
    with pytest.raises(ValueError):           # нет откликов
        _setup_runner(response_names=[])


def test_build_setup_runner_seed_cycle_grows_base():
    r = _setup_runner()
    ctrl = CampaignController(r)
    X = ctrl.propose_seed(8, seed=1)
    assert X.shape == (8, r.dim)
    assert len(r.points) == 0                 # propose — read-only (A0.6)

    Y = np.vstack([r._measure(np.asarray(x, float)) for x in X])
    out = ctrl.commit_seed(X, Y)

    assert out["added"] == 8 and len(r.points) == 8
    assert set(r.surrogates) == set(r.property_names)
    assert {p.origin_tag.get("origin") for p in r.points} == {"seed"}
    assert ctrl.can_undo() is False           # стартовая правда запечатала undo


# ======================================================================
# headless AppTest — вкладка «Кампания»: сетап §17.4 + ручной seed-цикл
# ======================================================================
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(_REPO, "src", "apps", "streamlit_app.py")


def _click(at, key):
    b = [w for w in at.button if w.key == key]
    assert b, f"кнопка {key} не найдена"
    b[0].click().run()


def test_campaign_setup_form_builds_manual_project():
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception
    _click(at, "setup_build")                 # дефолты: A,B,C × T,P + 3 отклика
    assert not at.exception

    ctrl = at.session_state["campaign_ctrl"]
    r = ctrl.runner
    assert r.q == 3 and r.dim == 5
    assert list(r.property_names) == ["strength", "gloss", "rho"]
    assert len(r.points) == 0                 # база пуста — ждём ручной seed


def test_campaign_setup_seed_manual_entry_grows_base():
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception
    _click(at, "setup_build")
    assert not at.exception

    at.session_state["setup_seed_n"] = 8
    at.run()
    _click(at, "setup_propose_seed")          # предложить seed (read-only)
    assert not at.exception
    assert "setup_seed_X" in at.session_state
    assert len(at.session_state["campaign_ctrl"].runner.points) == 0

    _click(at, "setup_fill_demo")             # Y от демо-оракула (явно, A0.6)
    assert not at.exception
    assert "setup_seed_Y" in at.session_state


    _click(at, "setup_commit_seed")           # зафиксировать → база растёт
    assert not at.exception

    r = at.session_state["campaign_ctrl"].runner
    assert len(r.points) == 8
    assert set(r.surrogates) == set(r.property_names)
    assert {p.origin_tag.get("origin") for p in r.points} == {"seed"}
