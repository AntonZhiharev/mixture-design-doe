# Copyright 2026 DOE contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 95 — правка полей журналов НЕ перезапускает страницу.

Живой отказ (наблюдение технолога 14.08.2026): «когда редактируешь поля,
особенно поле принятых решений, форма становится белой (полупрозрачной) и
потом как бы обновляется — такая реакция должна быть на кнопку „сохранить“,
а не на начало редактирования».

Разбор — по модели исполнения Streamlit, а не по догадке:

1. **Это не баг наших обработчиков.** Streamlit фиксирует значение
   ``st.text_input``/``st.text_area`` при ПОТЕРЕ ФОКУСА поля (дописал
   «Решение одной строкой», кликнул в «Почему так решили») и каждая
   фиксация перезапускает ВЕСЬ скрипт. «Белая полупрозрачная страница» —
   штатное затемнение устаревших элементов на время прогона.
2. **Больнее всего в журналах** («📚 Журнал решений», «🏭 Факты
   производства»): там 4 поля подряд БЕЗ ``st.form`` — четыре полных
   прогона большого приложения на одну запись, до нажатия кнопки.
3. **Лечение — штатный ``st.form``**: внутри формы виджеты не отправляют
   значения до нажатия ``st.form_submit_button``, перезапуск остаётся ровно
   один — на «✍️ Записать…», как человек и ожидает. Паттерн уже принят в
   кодовой базе (``proj_prices_form`` цен проекта в ``campaign_ui``).
4. **``clear_on_submit`` НЕ используется намеренно**: валидация («решение
   без обоснования не пишется») происходит ПОСЛЕ submit, и очистка формы
   стирала бы введённый текст ровно в момент отказа.

Отправка сообщения помощнику под это лечение не попадает: её ``st.rerun()``
после постановки хода — осознанный контракт iter91 (вопрос должен встать в
ленту, ход идёт в фоне), там перезапуск один и по делу.

Тесты проверяют КОНТРАКТ по исходнику (как iter80 §5): рендер браузера
юнит-тест проверить не может — это та же честная граница, что в iter89/93/94.
"""
from __future__ import annotations

import inspect

import pytest

pytest.importorskip("streamlit")

from src.apps import assistant_dock as dock  # noqa: E402


# ======================================================================
# 1. Оба журнала собраны в st.form
# ======================================================================
class TestJournalsUseForms:

    def test_decisions_panel_wraps_fields_in_form(self):
        """Поля решения не перезапускают скрипт до кнопки записи."""
        src = inspect.getsource(dock._render_decisions)
        assert 'st.form("dock_dec_form"' in src
        assert "st.form_submit_button(" in src

    def test_facts_panel_wraps_fields_in_form(self):
        src = inspect.getsource(dock._render_local_facts)
        assert 'st.form("dock_fact_form"' in src
        assert "st.form_submit_button(" in src

    def test_form_keys_are_distinct(self):
        """Две формы на одной странице обязаны иметь разные ключи."""
        both = inspect.getsource(dock._render_decisions) + \
            inspect.getsource(dock._render_local_facts)
        assert both.count('st.form("') == 2
        assert "dock_dec_form" in both and "dock_fact_form" in both

    def test_save_is_submit_button_not_plain_button(self):
        """Обычный st.button внутри st.form запрещён самим Streamlit;
        а вне формы он возвращал бы прежнее поведение (rerun на поля)."""
        for fn in (dock._render_decisions, dock._render_local_facts):
            src = inspect.getsource(fn)
            assert "st.button(" not in src, (
                f"{fn.__name__}: запись должна идти через form_submit_button")

    def test_no_clear_on_submit(self):
        """Валидация идёт ПОСЛЕ submit: очистка формы стирала бы текст
        человека ровно в момент отказа «решение без обоснования»."""
        both = inspect.getsource(dock._render_decisions) + \
            inspect.getsource(dock._render_local_facts)
        assert "clear_on_submit" not in both


# ======================================================================
# 2. Контракты iter80 не ослаблены
# ======================================================================
class TestIter80ContractsSurvive:

    def test_button_labels_preserved_verbatim(self):
        """Подписи кнопок знает промпт помощника (UI_BLOCK, iter74/80):
        переименование сломало бы навигацию «ищите кнопку …» молча."""
        from src.assistant.prompts import UI_BLOCK
        src = inspect.getsource(dock._render_decisions) + \
            inspect.getsource(dock._render_local_facts)
        for label in ("✍️ Записать решение в журнал",
                      "✍️ Записать факт в журнал"):
            assert label in src
            assert label in UI_BLOCK

    def test_manual_write_still_goes_through_human_path(self):
        """Форма — про отрисовку; путь записи остался человеческим."""
        src = inspect.getsource(dock._render_decisions) + \
            inspect.getsource(dock._render_local_facts)
        assert "human_record_decision(" in src
        assert "human_add_local_fact(" in src
        assert "dispatch(" not in src

    def test_validation_still_refuses_empty_input(self):
        """Пустое решение/факт по-прежнему не пишутся (A0.6)."""
        src = inspect.getsource(dock._render_decisions)
        assert "Решение без обоснования" in src
        facts = inspect.getsource(dock._render_local_facts)
        assert "Пустой факт записать нельзя" in facts


# ======================================================================
# 3. Rerun отправки сообщения помощнику — осознанный, не лечится формой
# ======================================================================
class TestChatRerunIsIntentional:

    def test_chat_input_is_not_inside_a_form(self):
        """st.chat_input в форму не кладётся (запрещено Streamlit), и его
        rerun — контракт iter91: вопрос встаёт в ленту, ход идёт в фоне."""
        src = inspect.getsource(dock._chat_submission)
        assert "st.form" not in src

    def test_turn_still_runs_in_background(self):
        """Гвоздь iter91 цел: в теле отрисовки хода нет."""
        body = inspect.getsource(dock.render_assistant_dock)
        assert "actx.run_turn" not in body
        assert "start_background_turn" in body
