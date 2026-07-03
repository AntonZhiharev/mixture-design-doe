# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 20 / §17.5 (Ш4–Ш5) — ручные мультицелевые ветки + рабочий стол.

Канон «логика+тест, потом UI»:

  * ЧИСТАЯ логика — :meth:`CampaignController.create_branch` (мультицель + роли +
    ценовая нога) и хелпер ``make_linear_price_fn`` — без Streamlit;
  * headless AppTest — полный РУЧНОЙ поток вкладки «Кампания» (§17.2): собрать
    проект → seed → создать ветку с 2 целями + ценовой ногой → предложить точки
    (read-only) → заполнить Y демо-оракулом → долить в общую базу (commit_measured,
    origin=branch:{id}), суррогаты переобучены, undo запечатан (Тр-7.2/7.3).
"""
import os
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.optimize.desirability import DesirabilitySpec
from src.apps.campaign import CampaignController
from src.apps import campaign as cv
from src.apps.campaign_ui import (build_setup_runner, make_linear_price_fn,
                                  draft_add_goal, draft_remove_goal,
                                  build_demo_campaign_runner,
                                  role_table_dataframe, experiment_index)



warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ======================================================================
# Чистые хелперы (без Streamlit)
# ======================================================================
def _seeded_ctrl(n_seed=12):
    """Собрать реальный runner (§17.4) и снять стартовый seed демо-оракулом."""
    r = build_setup_runner(
        mixture_names=["A", "B", "C"], process_names=["T", "P"],
        process_lower=[0.0, 0.0], process_upper=[1.0, 1.0],
        response_names=["strength", "gloss", "rho"], seed=1)
    ctrl = CampaignController(r)
    X = ctrl.propose_seed(n_seed, seed=1)
    Y = np.vstack([r._measure(np.asarray(x, float)) for x in X])
    ctrl.commit_seed(X, Y)
    return ctrl


def test_make_linear_price_fn_is_composition_weighted():
    fn = make_linear_price_fn([100.0, 200.0, 50.0])
    # чистый компонент A ⇒ цена состава = 100; процесс-оси не влияют
    price = fn(np.array([[1.0, 0.0, 0.0, 0.7, 0.3]]))
    assert np.isclose(price[0], 100.0)
    price2 = fn(np.array([[0.0, 0.5, 0.5, 0.0, 0.0]]))
    assert np.isclose(price2[0], 0.5 * 200.0 + 0.5 * 50.0)


def test_create_branch_multigoal_with_price_leg():
    ctrl = _seeded_ctrl()
    goals = {"strength": DesirabilitySpec("max", low=0.0, high=10.0),
             "gloss": DesirabilitySpec("max", low=0.0, high=10.0)}
    out = ctrl.create_branch(
        "work", goals, branch_id="w1", budget=15, satisfy_at=1.0,
        price_fn=make_linear_price_fn([100.0, 200.0, 50.0]),
        cost_spec=DesirabilitySpec("min", low=0.0, high=300.0),
        rho_property="rho")
    assert out["branch_id"] == "w1"
    assert out["n_goals"] == 2
    assert out["has_price_leg"] and out["rho_property"] == "rho"
    # ρ НЕ в цели ⇒ роль PRICE_INPUT ⇒ денежный канал ЖИВОЙ (не занулён, И-5)
    assert out["price_channel_suppressed"] is False
    assert "w1" in ctrl.runner.branches
    assert ctrl.can_undo() is False          # создание запечатало undo


def test_create_branch_rejects_empty_objective_and_incomplete_price():
    ctrl = _seeded_ctrl()
    with pytest.raises(ValueError):          # пустой объектив (§17.3)
        ctrl.create_branch("bad", {}, branch_id="b0")
    with pytest.raises(ValueError):          # price_fn без cost_spec/rho
        ctrl.create_branch(
            "bad", {"strength": DesirabilitySpec("max", low=0, high=1)},
            branch_id="b1", price_fn=make_linear_price_fn([1, 1, 1]))


def test_create_branch_rho_in_goal_zeroes_channel():
    ctrl = _seeded_ctrl()
    # ρ в цели И питает цену ⇒ роль OPTIMIZED ⇒ канал ЗАНУЛЁН (Гр-1)
    out = ctrl.create_branch(
        "rf", {"rho": DesirabilitySpec("min", low=0.0, high=5.0)},
        branch_id="rf", price_fn=make_linear_price_fn([100.0, 200.0, 50.0]),
        cost_spec=DesirabilitySpec("min", low=0.0, high=300.0),
        rho_property="rho")
    assert out["price_channel_suppressed"] is True


# ======================================================================
# headless AppTest — полный ручной поток вкладки «Кампания» (§17.2/§17.5)
# ======================================================================
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(_REPO, "src", "apps", "streamlit_app.py")


def _click(at, key):
    b = [w for w in at.button if w.key == key]
    assert b, f"кнопка {key} не найдена"
    b[0].click().run()


def test_campaign_manual_branch_and_workbench_flow():
    at = AppTest.from_file(APP, default_timeout=360).run()
    assert not at.exception

    # 1) собрать проект и снять seed демо-оракулом
    _click(at, "setup_build")
    at.session_state["setup_seed_n"] = 10
    at.run()
    _click(at, "setup_propose_seed")
    _click(at, "setup_fill_demo")
    _click(at, "setup_commit_seed")                # commit → st.rerun → база измерена
    assert not at.exception
    ctrl = at.session_state["campaign_ctrl"]
    assert len(ctrl.runner.points) == 10
    assert not ctrl.runner.branches                # веток ещё нет
    # §17.4→§17.5: после ОДНОГО коммита сразу открыто создание веток (без 2-го клика)
    assert any(w.key == "camp_nb_add_goal" for w in at.button)



    # 2) Ш4: собрать цель и создать ветку вручную (дефолт отклик=strength)
    _click(at, "camp_nb_add_goal")
    assert at.session_state["camp_new_goals"]      # черновик цели набран
    _click(at, "camp_nb_create")
    assert not at.exception
    ctrl = at.session_state["campaign_ctrl"]
    assert len(ctrl.runner.branches) == 1          # ветка создана
    bid = next(iter(ctrl.runner.branches))

    # 3) Ш5: ручной цикл рабочего стола — предложить (read-only)
    n_base = len(ctrl.runner.points)
    _click(at, f"camp_wb_propose_{bid}")
    assert not at.exception
    assert f"camp_wb_X_{bid}" in at.session_state
    assert len(at.session_state["campaign_ctrl"].runner.points) == n_base  # read-only

    # 4) заполнить Y демо-оракулом и долить в общую базу (commit_measured)
    _click(at, f"camp_wb_fill_{bid}")
    _click(at, f"camp_wb_commit_{bid}")
    assert not at.exception
    r = at.session_state["campaign_ctrl"].runner
    assert len(r.points) > n_base                  # общий пул вырос (И-1)
    tags = {p.origin_tag.get("origin") for p in r.points}
    assert f"branch:{bid}" in tags                 # origin-тег ветки


# ======================================================================
# §17.5 — черновик целей: добавление/замена/поштучное удаление (чистая логика)
# ======================================================================
def test_draft_add_goal_appends_distinct_and_replaces_same_resp():
    d = draft_add_goal([], resp="strength", kind="max", low=2, high=12, weight=1)
    assert len(d) == 1
    d = draft_add_goal(d, resp="gloss", kind="max", low=1, high=13, weight=2)
    assert len(d) == 2                              # разный отклик ⇒ 2 цели
    # тот же отклик ⇒ ЗАМЕНА (без дубля, без тихого схлопывания при создании)
    d = draft_add_goal(d, resp="strength", kind="min", low=0, high=5, weight=3)
    assert len(d) == 2
    s = [g for g in d if g["resp"] == "strength"][0]
    assert s["kind"] == "min" and s["weight"] == 3.0
    # target хранится только для вида target
    dt = draft_add_goal([], resp="rho", kind="target", low=0, high=2,
                        weight=1, target=1.0)
    assert dt[0]["target"] == 1.0


def test_draft_remove_goal_by_index_is_pure_and_idempotent():
    d = draft_add_goal([], resp="strength", kind="max", low=0, high=1, weight=1)
    d = draft_add_goal(d, resp="gloss", kind="max", low=0, high=1, weight=1)
    d2 = draft_remove_goal(d, 0)
    assert len(d) == 2 and len(d2) == 1             # вход не мутирован
    assert d2[0]["resp"] == "gloss"
    assert draft_remove_goal(d2, 5) == d2           # вне диапазона — без изменений


def test_campaign_branch_draft_two_goals_delete_and_relabel():
    """headless: 2 цели (смена отклика), поштучное 🗑, ярлык «Цель (отклик)»."""
    at = AppTest.from_file(APP, default_timeout=360).run()
    assert not at.exception
    _click(at, "setup_build")
    at.session_state["setup_seed_n"] = 10
    at.run()
    _click(at, "setup_propose_seed")
    _click(at, "setup_fill_demo")
    _click(at, "setup_commit_seed")
    at.run()

    # ярлык переименован единообразно в «Цель (отклик)» (не «отклик»)
    labels = {w.label for w in at.selectbox}
    assert "Цель (отклик)" in labels
    assert "отклик" not in labels

    # 1-я цель (strength), затем смена отклика на gloss и 2-я цель
    _click(at, "camp_nb_add_goal")
    assert len(at.session_state["camp_new_goals"]) == 1
    sb = [w for w in at.selectbox if w.key == "camp_nb_resp"][0]
    sb.select("gloss").run()
    _click(at, "camp_nb_add_goal")
    assert len(at.session_state["camp_new_goals"]) == 2

    # поштучное удаление первой цели → в черновике остаётся 1 (gloss)
    _click(at, "camp_nb_del_goal_0")
    draft = at.session_state["camp_new_goals"]
    assert len(draft) == 1 and draft[0]["resp"] == "gloss"
    assert not at.exception


# ======================================================================
# Регресс: колонка «вес» role-таблицы Arrow-совместима (однотипно строковая)
# ======================================================================
def test_role_table_weight_column_is_arrow_safe_uniform_str():
    """У целей — число, у нецелевых — «—»; смешение float/str ломало Arrow.

    Ранее колонка «вес» несла float у целей и строку «—» у нецелевых откликов —
    st.dataframe писал «Serialization … unsuccessful» и авто-приводил тип. Теперь
    колонка строго строковая ⇒ Arrow сериализует без ошибки/варнинга.
    """
    pa = pytest.importorskip("pyarrow")
    r = build_demo_campaign_runner(n_seed=12)
    rep = cv.branch_role_report(r, "premium")   # strength/gloss — цели, rho — нет
    df = role_table_dataframe(rep)
    col = list(df["вес"])
    assert all(isinstance(v, str) for v in col)          # без смешения типов
    assert any(v == "—" for v in col) and any(v != "—" for v in col)
    pa.Table.from_pandas(df)                             # без ArrowInvalid


def test_commit_seed_without_Y_shows_friendly_error_not_nan_crash():
    """commit_seed с пустыми Y → внятное сообщение (не сырое «Input y contains NaN»)."""
    at = AppTest.from_file(APP, default_timeout=360).run()
    _click(at, "setup_build")
    _click(at, "setup_propose_seed")                 # предложили, НО не заполнили Y
    _click(at, "setup_commit_seed")
    assert not at.exception                          # UI не упал
    msgs = " ".join(e.value for e in at.error)
    assert "Заполните измеренные отклики" in msgs     # понятная подсказка
    assert "NaN" not in msgs                          # без сырого sklearn-текста
    assert len(at.session_state["campaign_ctrl"].runner.points) == 0  # база пуста


# ======================================================================
# Сквозная нумерация опытов проекта (№ опыта = позиция в общей базе)
# ======================================================================
def test_experiment_index_is_sequential_after_base():
    """Предложенные точки нумеруются сквозным № опыта проекта (1-based).

    Если в общей базе уже ``base_count`` опытов, N новых получают их БУДУЩИЕ
    номера ``base_count+1 … base_count+N`` (порядок добавления в ``runner.points``).
    Индекс именован «№ опыта» — так он и показывается слева в таблицах UI.
    """
    idx = experiment_index(12, 3)
    assert list(idx) == [13, 14, 15]
    assert idx.name == "№ опыта"
    # пустая база ⇒ нумерация с 1 (стартовый seed-дизайн)
    assert list(experiment_index(0, 5)) == [1, 2, 3, 4, 5]
    # вырожденный случай: ноль новых точек ⇒ пустой индекс (без падения)
    assert list(experiment_index(7, 0)) == []
    # номера непрерывны и продолжают базу от ветки к ветке (общий пул, И-1)
    prev = experiment_index(10, 3)          # seed(10) → ветка добрала 3 (№ 11..13)
    nxt = experiment_index(13, 2)           # следующее предложение (№ 14..15)
    assert prev[-1] + 1 == nxt[0]






