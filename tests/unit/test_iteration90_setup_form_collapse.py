# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 90 — форма «🆕 Новый проект» схлопывалась при правке границы оси.

Живой отказ 13.08.2026: технолог поменял ВЕРХНИЙ предел одной из процесс-осей
в форме §17.4 — форма закрылась, лента диалога прокрутилась заново, внешне
«страница перезагрузилась». Диагностика на AppTest (Streamlit 1.58) показала
СЦЕПКУ двух дефектов:

1. **Экспандер формы был stateless** (``expanded=ctrl is None`` без ``key``):
   раскрытость жила только в браузере, привязанная к ПОЗИЦИИ узла в дереве
   элементов, а дефолт пересчитывался на каждом прогоне. Как только проект
   собран, дефолт = «свёрнуто», и любой сдвиг дерева выше формы
   перемонтировал экспандер с этим дефолтом.

2. **Сдвиг дерева поставляло уведомление смены фазы** (``decide_tab``):
   фаза фиксируется в ``session_state`` ПОСЛЕ отрисовки области, поэтому
   «проект собран → открыта закладка …» вставало в дерево НЕ на прогоне
   кнопки, а на СЛЕДУЮЩЕМ действии пользователя — им и оказалась правка
   границы оси.

Правки iter90:

* раскрытость — состояние ВИДЖЕТА (``st.expander(key=…, on_change="rerun")``)
  с пином в app-state (Streamlit чистит widget-state неактивных прогонов);
  ``expanded=`` передаётся КОНСТАНТОЙ — он входит в element_id, и меняющееся
  значение пересоздавало бы виджет со сбросом состояния (ровно старый дефект);
* загрузчики проекта управляют формой ЯВНО через отложенный ключ
  ``SETUP_FORM_OPEN_PENDING`` (собранный проект → свернуть, черновик →
  раскрыть);
* ``decide_tab`` не рапортует о «переходе» на закладку, где человек УЖЕ
  стоит, — уведомление-призрак больше не вставляется в дерево;
* фолбэк для Streamlit без stateful-экспандера (< 1.58) — прежнее поведение.
"""
import os
import warnings

import pytest

from src.apps import campaign_state as cst
from src.apps import workspace as ws
from src.apps.campaign_ui import (SETUP_FORM_OPEN_KEY,
                                  SETUP_FORM_OPEN_PENDING,
                                  setup_expander_kwargs)

warnings.filterwarnings("ignore")


# ======================================================================
# 1. Чистая логика: аргументы экспандера формы
# ======================================================================
class TestSetupExpanderKwargs:
    def test_stateful_pins_key_and_constant_expanded(self):
        kw = setup_expander_kwargs(stateful=True, fallback_expanded=False)
        assert kw["key"] == SETUP_FORM_OPEN_KEY
        assert kw["on_change"] == "rerun"
        # expanded — КОНСТАНТА: он входит в element_id, и зависимость от
        # «есть ли проект» пересоздавала бы виджет со сбросом состояния.
        assert kw["expanded"] is True

    def test_stateful_ignores_fallback(self):
        assert setup_expander_kwargs(stateful=True, fallback_expanded=True) \
            == setup_expander_kwargs(stateful=True, fallback_expanded=False)

    def test_fallback_keeps_iter72_behaviour(self):
        assert setup_expander_kwargs(
            stateful=False, fallback_expanded=True) == {"expanded": True}
        assert setup_expander_kwargs(
            stateful=False, fallback_expanded=False) == {"expanded": False}

    def test_ui_state_keys_are_not_setup_keys(self):
        """Раскрытость — состояние ИНТЕРФЕЙСА: в черновик проекта не пишется."""
        for key in (SETUP_FORM_OPEN_KEY, SETUP_FORM_OPEN_PENDING):
            assert not key.startswith("setup_")
            assert cst.setup_draft_fields({key: True}) == {}


# ======================================================================
# 2. Чистая логика: decide_tab не рапортует о переходе «в ту же закладку»
# ======================================================================
class TestDecideTabNoGhostNotice:
    def test_phase_change_on_same_tab_is_silent(self):
        """Собрали проект, СТОЯ на «Старте», — переход никуда не случился."""
        d = ws.decide_tab("start", prev_phase=ws.PHASE_EMPTY,
                          has_project=True, n_points=0)
        assert d.key == "start"
        assert d.phase == ws.PHASE_SETUP
        assert d.notice == "" and not d.moved

    def test_phase_change_from_other_tab_still_reports(self):
        """Регресс iter69: реальный автопереход по-прежнему объясняется."""
        d = ws.decide_tab("overview", prev_phase=ws.PHASE_EMPTY,
                          has_project=True, n_points=0)
        assert d.key == "start" and d.moved
        assert "проект собран" in d.notice

    def test_first_run_still_silent(self):
        d = ws.decide_tab(None, prev_phase=None, has_project=False)
        assert d.key == "start" and d.notice == ""

    def test_seed_commit_still_moves_to_branches(self):
        d = ws.decide_tab("start", prev_phase=ws.PHASE_SETUP,
                          has_project=True, n_points=10)
        assert d.key == "branches" and d.moved and "измерен" in d.notice


# ======================================================================
# 3. headless AppTest — сценарий живого отказа
# ======================================================================
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = os.path.join(_REPO, "src", "apps", "streamlit_app.py")


def _form_expanded(at) -> bool:
    """Фактическая раскрытость формы сетапа (proto, который уйдёт на фронт)."""
    for b in at.main.expander:
        if "\U0001F195" in b.label:                       # 🆕
            return bool(b.proto.expanded)
    raise AssertionError("экспандер формы сетапа не найден")


def _edit_number(at, key, value):
    w = [x for x in at.number_input if x.key == key]
    assert w, f"number_input {key} не найден"
    w[0].set_value(value).run()


def test_form_stays_open_after_editing_process_bound():
    """Гвоздь iter90: правка ВЕРХНЕЙ границы оси не схлопывает форму."""
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert _form_expanded(at) is True                     # пустая сессия
    [b for b in at.button if b.key == "setup_build"][0].click().run()
    assert "campaign_ctrl" in at.session_state
    assert _form_expanded(at) is True                     # сборка не закрыла

    _edit_number(at, "setup_phi_T", 230.0)                # живой сценарий
    assert not at.exception
    assert at.session_state["setup_phi_T"] == 230.0
    assert _form_expanded(at) is True, \
        "правка границы оси схлопнула форму (живой отказ 13.08.2026)"

    _edit_number(at, "setup_plo_T", 155.0)                # и ещё раз
    assert _form_expanded(at) is True


def test_no_ghost_phase_notice_after_build_on_start_tab():
    """Уведомление «проект собран → …» не всплывает на следующем действии."""
    at = AppTest.from_file(APP, default_timeout=300).run()
    [b for b in at.button if b.key == "setup_build"][0].click().run()
    _edit_number(at, "setup_phi_T", 230.0)
    assert not any("проект собран" in str(m.value) for m in at.info), \
        "уведомление смены фазы вставилось в дерево с опозданием на прогон"


def test_user_collapse_survives_reruns():
    """Свернул руками — форма НЕ раскрывается сама на следующем действии."""
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.session_state[SETUP_FORM_OPEN_KEY] = False         # человек свернул
    at.run()
    assert _form_expanded(at) is False
    _edit_number(at, "setup_seed", 2)                     # любое действие
    assert _form_expanded(at) is False


def test_loading_built_project_collapses_form():
    """Отложенный ключ: собранный проект → форма сворачивается (как раньше)."""
    at = AppTest.from_file(APP, default_timeout=300).run()
    [b for b in at.button if b.key == "setup_build"][0].click().run()
    # путь загрузчика собранного проекта (streamlit_app: load_campaign)
    at.session_state[SETUP_FORM_OPEN_PENDING] = False
    at.run()
    assert _form_expanded(at) is False
    # ...а загрузка ЧЕРНОВИКА раскрывает — черновик грузят, чтобы править
    at.session_state[SETUP_FORM_OPEN_PENDING] = True
    at.run()
    assert _form_expanded(at) is True
