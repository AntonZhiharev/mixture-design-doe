# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 104 — UI гейта измеримости (UI 2.4 / UI 2.1) и области обучения.

Канон «сначала логика + тест, потом UI»: ядро — в
``test_iteration104_training_scope.py``. Здесь:

  * фасад ``CampaignController``: ``set_project_gate`` / ``set_training_scope``
    / ``training_scope_diagnostics`` (проектный уровень, undo не трогают),
    ``set_branch_gate`` — обратимая мутация намерения ветки (снимок undo несёт
    гейт), ``create_branch(gate=)`` — гейт при рождении, неверный гейт не
    оставляет ветку;
  * чистые подписи/хелперы UI: ``project_gate_caption``,
    ``branch_gate_caption``, ``surrogate_coverage_caption`` (область
    «история»), ``training_scope_caption``, ``gate_direction_code``,
    ``gate_default_threshold``;
  * headless AppTest: после seed на «Старте» доступен добор области без ветки
    с блоком проектного гейта (UI 2.4) — объявить гейт → предложить добор →
    зафиксировать; в форме ветки — поле гейта (UI 2.1), ветка рождается с
    собственным гейтом; на рабочем столе гейт виден и снимается.
"""
import os
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.apps import campaign_ui as ui
from src.apps.campaign import CampaignController
from src.apps.campaign_ui import build_setup_runner
from src.apps.mixture_process_runner import SCOPE_ACTIVE, SCOPE_HISTORY
from src.optimize.desirability import DesirabilitySpec

warnings.filterwarnings("ignore", category=ConvergenceWarning)

PROPS = ["strength", "gloss", "rho"]


def _seeded_ctrl(n_seed=12, *, gaps=()):
    """Реальный раннер (§17.4) + seed демо-оракулом; ``gaps`` — индексы точек,
    у которых gloss «не измерен» (с причиной)."""
    r = build_setup_runner(
        mixture_names=["A", "B", "C"], process_names=["T", "P"],
        process_lower=[0.0, 0.0], process_upper=[1.0, 1.0],
        response_names=PROPS, seed=1)
    ctrl = CampaignController(r)
    X = np.asarray(ctrl.propose_seed(n_seed, seed=1), float)
    Y = np.vstack([r._measure(np.asarray(x, float)) for x in X])
    reasons = [None] * len(X)
    for i in gaps:
        Y[i, 1] = np.nan
        reasons[i] = {"gloss": "образец не получен"}
    ctrl.commit_seed(X, Y, missing_reasons=reasons if gaps else None)
    return ctrl


# ======================================================================
# 1. Фасад контроллера
# ======================================================================
def test_controller_project_gate_and_scope():
    ctrl = _seeded_ctrl()
    r = ctrl.runner
    out = ctrl.set_project_gate("strength", 3.0, "ge")
    assert out["gate"] == {"response": "strength", "threshold": 3.0,
                           "direction": "ge"}
    assert out["scope_before"] == SCOPE_ACTIVE
    assert out["scope_after"] == SCOPE_HISTORY
    assert out["surrogate_fitted"] and out["coverage"]["scope"] == SCOPE_HISTORY
    assert ctrl.can_undo() is False                    # проектный уровень
    out2 = ctrl.set_training_scope("gloss", SCOPE_HISTORY)
    assert (out2["scope_before"], out2["scope_after"]) == (SCOPE_ACTIVE,
                                                           SCOPE_HISTORY)
    assert r.training_scope("gloss") == SCOPE_HISTORY
    d = ctrl.training_scope_diagnostics("gloss")
    assert d["response"] == "gloss" and d["history_helps"] is None
    out3 = ctrl.set_project_gate(None)
    assert out3["gate"] is None and r.project_gate() is None


def test_controller_branch_gate_is_undoable_intent():
    ctrl = _seeded_ctrl()
    r = ctrl.runner
    ctrl.create_branch("b", {"strength": DesirabilitySpec("max", low=0, high=10)},
                       branch_id="b1")
    d0 = r.branches["b1"].d_best
    out = ctrl.set_branch_gate("b1", "gloss", 2.0, "ge")
    assert out["op"] == "set_branch_gate" and out["undo_available"]
    assert out["gate"]["source"] == "branch" and out["gate"]["response"] == "gloss"
    # гейт — не цель: goal и измеренный d_best не тронуты
    assert set(r.branches["b1"].goal) == {"strength"}
    assert r.branches["b1"].d_best == pytest.approx(d0)
    assert r.training_scope("gloss") == SCOPE_HISTORY
    ctrl.undo()
    assert r.branch_gate("b1") is None
    # область обучения — проектный факт, откату не подлежит
    assert r.training_scope("gloss") == SCOPE_HISTORY
    # снимок другой мутации тоже несёт гейт: undo правки весов вернёт гейт
    ctrl.set_branch_gate("b1", "gloss", 2.0, "ge")
    ctrl.set_weights("b1", {"strength": 2.0})
    ctrl.undo()
    assert r.branch_gate("b1")["response"] == "gloss"
    with pytest.raises(KeyError, match="Нет ветки"):
        ctrl.set_branch_gate("nope", "gloss", 1.0)


def test_create_branch_with_gate_and_inheritance():
    ctrl = _seeded_ctrl()
    r = ctrl.runner
    goals = {"strength": DesirabilitySpec("max", low=0, high=10)}
    out = ctrl.create_branch("own", goals, branch_id="o",
                             gate={"response": "gloss", "threshold": 1.5,
                                   "direction": "le"})
    assert out["gate"]["source"] == "branch"
    assert r.branch_gate("o") == {"response": "gloss", "threshold": 1.5,
                                  "direction": "le"}
    # без gate — наследует проектный
    ctrl.set_project_gate("rho", 0.5, "ge")
    out2 = ctrl.create_branch("inh", goals, branch_id="i")
    assert out2["gate"]["source"] == "project" and r.branch_gate("i") is None
    assert r.effective_branch_gate("i")["response"] == "rho"
    # неверный гейт — ветка НЕ создаётся
    n = len(r.branches)
    with pytest.raises(KeyError, match="не среди свойств"):
        ctrl.create_branch("bad", goals, branch_id="bad",
                           gate={"response": "нет", "threshold": 1.0})
    with pytest.raises(ValueError, match="direction"):
        ctrl.create_branch("bad2", goals, branch_id="bad2",
                           gate={"response": "gloss", "threshold": 1.0,
                                 "direction": "between"})
    assert len(r.branches) == n and "bad" not in r.branches


# ======================================================================
# 2. Чистые подписи и хелперы
# ======================================================================
def test_gate_helpers_pure():
    assert ui.gate_direction_code("≥ (не ниже порога)") == "ge"
    assert ui.gate_direction_code("le") == "le"
    with pytest.raises(ValueError, match="Направление"):
        ui.gate_direction_code("between")
    ctrl = _seeded_ctrl()
    thr = ui.gate_default_threshold(ctrl.runner, "strength")
    vals = [p.Y["strength"] for p in ctrl.runner.points]
    assert thr == pytest.approx(float(np.median(vals)))
    assert ui.gate_default_threshold(ctrl.runner, "нет") == 0.0


def test_captions_three_states():
    ctrl = _seeded_ctrl(gaps=(0, 3))
    r = ctrl.runner
    # гейт не объявлен, пропуски есть → подсказка объявить
    cap0 = ui.project_gate_caption(r)
    assert "не объявлен" in cap0 and "2 непроведённых" in cap0
    ctrl.create_branch("b", {"strength": DesirabilitySpec("max", low=0, high=10)},
                       branch_id="b1")
    assert "нет" in ui.branch_gate_caption(r, "b1") and "2" in \
        ui.branch_gate_caption(r, "b1")
    # проектный гейт объявлен и обучен
    ctrl.set_project_gate("strength", 3.0, "ge")
    cap1 = ui.project_gate_caption(r)
    assert "«strength» ≥ 3" in cap1 and "всей истории" in cap1
    bc = ui.branch_gate_caption(r, "b1")
    assert "проектный" in bc and "В целях ветки" not in bc   # strength — цель
    # собственный гейт по отклику НЕ из целей → предупреждение про argmax
    ctrl.set_branch_gate("b1", "gloss", 2.0, "ge")
    bc2 = ui.branch_gate_caption(r, "b1")
    assert "собственный" in bc2 and "В целях ветки «gloss» нет" in bc2
    # покрытие: область «история» видна в подписи модели
    cov_cap = ui.surrogate_coverage_caption(r)
    assert "вся история" in cov_cap and "strength" in cov_cap
    # без пропусков и без гейта — подпись пустая (не шумим)
    assert ui.project_gate_caption(_seeded_ctrl().runner) == ""
    # LOO-подпись: три состояния
    d = ctrl.training_scope_diagnostics("gloss")
    assert "gloss" in ui.training_scope_caption(d)
    fake = {"response": "y", "scope_now": "active", "n_active": 9,
            "n_outside": 4, "history_helps": True,
            "active": {"loo_logp": -1.0, "loo_rmse": 0.5},
            "history": {"loo_logp": -0.5, "loo_rmse": 0.4}, "note": "ЛУЧШЕ"}
    txt = ui.training_scope_caption(fake)
    assert "RMSE 0.5" in txt and "RMSE 0.4" in txt and "ЛУЧШЕ" in txt



# ======================================================================
# 3. headless AppTest — UI 2.4 (добор без ветки) и UI 2.1 (гейт ветки)
# ======================================================================
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(_REPO, "src", "apps", "streamlit_app.py")


def _click(at, key):
    b = [w for w in at.button if w.key == key]
    assert b, f"кнопка {key} не найдена"
    b[0].click().run()


def _keys(at, kind):
    return {w.key for w in getattr(at, kind)}


def test_ui_project_gate_augment_and_branch_gate_flow():
    at = AppTest.from_file(APP, default_timeout=420).run()
    assert not at.exception
    _click(at, "setup_build")
    at.session_state["setup_seed_n"] = 10
    at.run()
    # пустая база: блока гейта нет (множитель применить нечем)
    assert "setup_gate_set" not in _keys(at, "button")
    _click(at, "setup_propose_seed")
    _click(at, "setup_fill_demo")
    _click(at, "setup_commit_seed")
    assert not at.exception
    ctrl = at.session_state["campaign_ctrl"]
    assert len(ctrl.runner.points) == 10

    # UI 2.4: добор без ветки на «Старте» — по переключателю
    at.session_state["ws_tab"] = "start"
    at.run()
    tog = [w for w in at.checkbox if w.key == "ui_show_augment"]
    assert tog, "переключатель добора области не найден"
    tog[0].check().run()
    assert not at.exception
    assert "setup_gate_set" in _keys(at, "button")
    assert "setup_propose_seed" in _keys(at, "button")
    # объявить проектный гейт по strength
    [w for w in at.selectbox if w.key == "setup_gate_resp"][0]\
        .select("strength").run()
    _click(at, "setup_gate_set")
    assert not at.exception
    r = at.session_state["campaign_ctrl"].runner
    assert r.project_gate()["response"] == "strength"
    assert r.training_scope("strength") == SCOPE_HISTORY
    # предложить добор с множителем → зафиксировать демо-значениями
    at.session_state["setup_seed_n"] = 4
    at.run()
    _click(at, "setup_propose_seed")
    assert not at.exception
    assert at.session_state["setup_seed_X"].shape == (4, 5)
    _click(at, "setup_fill_demo")
    _click(at, "setup_commit_seed")
    assert not at.exception
    r = at.session_state["campaign_ctrl"].runner
    assert len(r.points) == 14
    assert r.project_gate()["response"] == "strength"   # пережил rerun

    # UI 2.1: форма ветки — собственный гейт
    at.session_state["ws_tab"] = "branches"
    at.run()
    assert "camp_nb_use_gate" in _keys(at, "checkbox")
    _click(at, "camp_nb_add_goal")
    [w for w in at.checkbox if w.key == "camp_nb_use_gate"][0].check().run()
    assert "camp_nb_gate_resp" in _keys(at, "selectbox")
    [w for w in at.selectbox if w.key == "camp_nb_gate_resp"][0]\
        .select("gloss").run()
    _click(at, "camp_nb_create")
    assert not at.exception
    r = at.session_state["campaign_ctrl"].runner
    bid = next(iter(r.branches))
    g = r.branch_gate(bid)
    assert g is not None and g["response"] == "gloss"
    assert r.effective_branch_gate(bid)["source"] == "branch"
    # рабочий стол: редактор гейта ветки на месте, «снять» возвращает проектный
    assert f"camp_wb_gate_set_{bid}" in _keys(at, "button")
    assert f"camp_wb_gate_clear_{bid}" in _keys(at, "button")
    _click(at, f"camp_wb_gate_clear_{bid}")
    assert not at.exception
    r = at.session_state["campaign_ctrl"].runner
    assert r.branch_gate(bid) is None
    assert r.effective_branch_gate(bid)["source"] == "project"

