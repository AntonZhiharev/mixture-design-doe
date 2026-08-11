# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 67 — повтор упавшего хода ассистента КНОПКОЙ, а не молча.

Наблюдение с прогона UI: транспорт (`assistant/llm.py::_http_transport`) делает
ОДНУ попытку, поэтому разовый обрыв TLS (``WinError 10054``) заканчивал ход
ошибкой — при том что тот же запрос секундой позже проходит. Человек оставался
в тупике: ответа нет, а повторить вопрос можно только перенабрав его руками.

Решение сознательно НЕ авторетрай:

* скрытые повторы тратят деньги и время (модель — платная, ход шёл 82 с),
  а на неверном ключе или пустом счёте крутились бы вхолостую;
* поэтому отказ показывается ЯВНО (всплывающее предупреждение + подсвеченная
  кнопка), а повтор остаётся решением человека (A0.6 — не блокируем молча
  и не действуем молча).

Проверяется чистая функция :func:`views.retry_prompt`, которую рисует док
(`src/apps/assistant_dock.py::_render_retry`) — по канону `.clinerules`
логика тестируется без запуска Streamlit.
"""
import pytest

from src.assistant import views
from src.assistant.context import run_turn
from src.assistant.llm import LLMError
from src.assistant.session import new_session
from src.assistant.views import RetryPrompt, retry_prompt

PROJECT = "pvc_edge_v1"

#: Та же референсная геометрия, что в iter65 (golden iter45/49/50) — усечённая
#: до узлов, которых достаточно для сборки контекста хода.
NODES = [
    {"name": "RESIN", "role": "FIXED", "value": 100.0},
    {"name": "DINP", "role": "ABSOLUTE", "range": [4.0, 14.0]},
    {"name": "TiO2", "role": "ABSOLUTE", "range": [0.3, 8.0], "scale": "log"},
]

#: Реальный текст отказа с прогона 11.08.2026 (сессия my_project): именно он
#: должен уводить пользователя в «повторите», а не в «проверьте ключ».
WINERROR_10054 = (
    "LLMError: Сетевая ошибка обращения к OpenRouter: <urlopen error "
    "[WinError 10054] Удаленный хост принудительно разорвал существующее "
    "подключение>")


def _ctx(tmp_path, session=None):
    from src.assistant.tools import ToolContext
    from src.design.phr_sampler import PhrSpec
    return ToolContext(spec=PhrSpec.from_dicts(NODES),
                       session=session or new_session(PROJECT),
                       root=str(tmp_path), project=PROJECT)


def _boom(message):
    def transport(payload, *, key="", timeout=0):
        raise LLMError(message)
    return transport


def _answer(text):
    return {"choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"total_tokens": 7}}


def _ok_transport(payload, *, key="", timeout=0):
    return _answer("## ОТВЕТ\nна связи")


# ======================================================================
# 1. Успех и «хода не было» кнопку не рисуют
# ======================================================================
def test_successful_turn_shows_no_retry(tmp_path):
    """После нормального ответа кнопка повтора висеть не должна."""
    s = new_session(PROJECT)
    res = run_turn(s, _ctx(tmp_path, s), "Привет, ты на связи?",
                   transport=_ok_transport)
    assert res.ok
    prompt = retry_prompt(res)
    assert not prompt.show and not prompt
    assert prompt.toast == ""


def test_no_turn_at_all_shows_no_retry():
    assert not retry_prompt(None)
    assert retry_prompt(None) == RetryPrompt()


# ======================================================================
# 2. Отказ: тост с причиной + вопрос для повтора
# ======================================================================
def test_failed_turn_offers_retry_with_the_same_question(tmp_path):
    """Кнопка переотправляет СЛОВА ЧЕЛОВЕКА, а не пересказ модели."""
    s = new_session(PROJECT)
    question = "Привет, ты на связи?"
    res = run_turn(s, _ctx(tmp_path, s), question,
                   transport=_boom(WINERROR_10054))
    assert not res.ok

    prompt = retry_prompt(res)
    assert prompt and prompt.show
    assert prompt.question == question
    assert prompt.retryable is True
    assert prompt.icon == "⚠️"
    assert "Повторить отправку" in prompt.button_label
    # Пользователь должен видеть ПРИЧИНУ, а не только факт отказа.
    assert "10054" in prompt.toast or "Сетевая ошибка" in prompt.toast
    assert prompt.hint


def test_toast_is_short_but_full_error_stays_in_session(tmp_path):
    """Тост живёт секунды: он усечён, а полный текст остаётся в ленте."""
    s = new_session(PROJECT)
    long_error = WINERROR_10054 + " " + "подробности " * 60
    res = run_turn(s, _ctx(tmp_path, s), "Что в базе?",
                   transport=_boom(long_error))
    prompt = retry_prompt(res)
    assert len(prompt.toast) <= views.TOAST_CHARS + 40
    assert "…" in prompt.toast
    # Полная причина не теряется: она в ответе ассистента и в ходе.
    assert "10054" in res.error and "10054" in s.messages[-1].content


# ======================================================================
# 3. Классификация причин: что лечится повтором, а что нет
# ======================================================================
@pytest.mark.parametrize("error,retryable", [
    (WINERROR_10054, True),
    ("LLMError: Сетевая ошибка обращения к OpenRouter: <urlopen error timed out>",
     True),
    ("LLMError: OpenRouter HTTP 429. Слишком часто: подождите и повторите.",
     True),
    ("LLMError: OpenRouter HTTP 502. bad gateway", True),
    ("LLMError: OpenRouter HTTP 401. Проверьте OPENROUTER_API_KEY.", False),
    ("LLMError: OpenRouter HTTP 402. На счёте OpenRouter закончились средства.",
     False),
    ("LLMError: OpenRouter HTTP 404. Такой модели нет", False),
    ("LLMError: Не задан OPENROUTER_API_KEY: укажите ключ", False),
])
def test_retryable_is_decided_by_the_reason(error, retryable):
    """Повтор предлагается там, где он может помочь.

    Обрыв связи, таймаут, 429 и 5xx — да. Неверный ключ, пустой счёт и
    несуществующая модель повтором не лечатся, и звать человека «попробовать
    снова» значит гонять его по кругу.
    """
    prompt = retry_prompt({"ok": False, "error": error, "question": "q"})
    assert prompt.retryable is retryable
    assert prompt.show
    if not retryable:
        assert prompt.icon == "⛔"
        # Кнопку всё равно не запрещаем (решает человек), но предупреждаем.
        assert "ещё раз" in prompt.button_label


def test_unknown_reason_still_offers_retry():
    """Незнакомую причину не выдаём за диагноз, но повторить даём."""
    prompt = retry_prompt({"ok": False, "error": "LLMError: ???",
                           "question": "q"})
    assert prompt.retryable is True and prompt.hint


def test_missing_error_text_does_not_crash_the_prompt():
    prompt = retry_prompt({"ok": False, "question": "q"})
    assert prompt.show and "причина неизвестна" in prompt.toast


# ======================================================================
# 4. Док: тонкий слой поверх чистой логики
# ======================================================================
def test_dock_exposes_retry_widgets():
    """UI-док лишь рисует: сама логика проверена выше."""
    pytest.importorskip("streamlit")
    from src.apps import assistant_dock

    assert callable(assistant_dock._render_retry)
    # Упавший вопрос хранится ОТДЕЛЬНО от K_PENDING: тот отправляется сам,
    # а этот ждёт нажатия кнопки.
    assert assistant_dock.K_FAILED != assistant_dock.K_PENDING


def test_retry_after_failure_reaches_the_model(tmp_path):
    """Сценарий целиком: отказ → кнопка → повтор тем же вопросом → ответ."""
    s = new_session(PROJECT)
    ctx = _ctx(tmp_path, s)
    question = "Привет, ты на связи?"

    failed = run_turn(s, ctx, question, transport=_boom(WINERROR_10054))
    prompt = retry_prompt(failed)
    assert prompt.show and prompt.question == question

    # Кнопка кладёт prompt.question в очередь — док переотправляет его как есть.
    again = run_turn(s, ctx, prompt.question, transport=_ok_transport)
    assert again.ok and "на связи" in again.text
    assert not retry_prompt(again)
    # История честная: оба вопроса человека и оба ответа на месте.
    assert [m.role for m in s.messages] == ["user", "assistant",
                                            "user", "assistant"]
