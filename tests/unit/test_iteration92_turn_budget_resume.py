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
"""Iteration 92 — обрыв хода по бюджету был ЛОВУШКОЙ, а не паузой.

Живой отказ 14.08.2026 (ПВХ-кампания, 23 узла). Человек принял решения по
двум температурным осям и попросил пересобрать проект. Ход упёрся в бюджет,
помощник ответил «⏱ бюджет исчерпан… повторите вопрос — контекст вызовов
сохранён в сессии». Человек повторил — тот же текст. Замер по журналам
проекта: ход 234 с (4 вызова, инструменты 0,2 с, 265 676 prompt-токенов) и
повтор 210 с (10 вызовов, инструменты 2,9 с, 213 818 prompt-токенов).
Инструменты съели 0,1 % времени — всё остальное ушло на генерацию аргументов
пакета проекта (14–16 тыс. completion-токенов за ход).

Разбор дал ТРИ независимых дефекта.

**1. Обещание в тексте было ложным.** ``ASSISTANT_SPEC`` §219 требует
дописывать ``new_messages`` в сессию целиком (включая роль ``tool``), «иначе
следующий вопрос потеряет контекст уже сделанных вызовов». Демо и тесты
iter60 так и делали, а боевой ``context.run_turn`` — нет: он брал из
результата только ``text``/``calls``/``usage``. В переписке живой сессии
оказалось 26 ``assistant`` + 30 ``user`` и НИ ОДНОГО ``tool``. Поэтому повтор
начинался с чистого листа и упирался в тот же предел.

**2. Ход платил за обречённый запрос.** Бюджет проверялся ТОЛЬКО после пачки
инструментов, поэтому цикл успевал заказать новую генерацию, зная, что
времени не осталось: деньги списаны, ответа нет.

**3. Предел нельзя было поднять.** ``TIME_BUDGET_S`` — константа модуля, а
``assistant_dock`` не передавал ``time_budget_s`` в цикл. Для кампании, где
честный ход длится 234 с, «180 с и не иначе» — тупик.

Отдельно проверяется правка промпта: у модели БЫЛ точечный инструмент
(``propose_setup_fields``, в том же ходе применённый к двум другим полям), но
не было явного правила «границы и уровни процесс-осей — это поля формы».
Поэтому на две оси она собирала пакет проекта целиком.

Инварианты, которые фиксируют тесты:

* прерванный ход ОСТАВЛЯЕТ в сессии вызовы и их результаты (§219);
* завершённый ход этого НЕ делает — иначе дубль аргументов жёг бы бюджет;
* результат инструмента в переписке усечён С ПОМЕТКОЙ (A0.6);
* служебные роли не лезут в ленту разговора;
* новый запрос к модели не начинается без запаса времени;
* текст обрыва называет сделанное и НЕ обещает несуществующего;
* бюджет и предел шагов читаются из окружения на каждый ход.
"""
from __future__ import annotations

import json
import time

import pytest

from src.apps import workspace as wsx
from src.assistant import config as cfg
from src.assistant import context as actx
from src.assistant import llm
from src.assistant.session import new_session

PROJECT = "iter92"


# ======================================================================
# Заготовки: транспорт и контекст без сети
# ======================================================================
class FakeTransport:
    """Отдаёт заранее подготовленные ответы модели; сети нет."""

    def __init__(self, bodies, *, delay_s: float = 0.0):
        self.bodies = list(bodies)
        self.delay_s = float(delay_s)
        self.payloads = []

    def __call__(self, payload, *, key="", timeout=0, url=""):
        self.payloads.append(payload)
        if self.delay_s:
            time.sleep(self.delay_s)
        if not self.bodies:
            raise AssertionError("модель вызвана больше раз, чем ожидалось")
        return self.bodies.pop(0)


def _msg(content="", tool_calls=None):
    out = {"role": "assistant", "content": content}
    if tool_calls:
        out["tool_calls"] = tool_calls
    return {"choices": [{"message": out}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                      "total_tokens": 15}}


def _call(name, args, cid="call_1"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


class Ctx:
    """Минимальный контекст инструментов: ни ядра, ни файлов не нужно."""

    def __init__(self, session):
        self.session = session
        self.runner = None
        self.root = ""
        self.project = PROJECT
        self.spec = None
        self.extra = {}

    def require_session(self):
        return self.session


# ======================================================================
# 1. Контекст вызовов переживает обрыв (ASSISTANT_SPEC §219)
# ======================================================================
class TestResumeContext:

    def test_interrupted_turn_keeps_calls_and_results(self):
        """Ровно то, чего не хватало: после обрыва вызовы ЕСТЬ в переписке."""
        s = new_session(PROJECT)
        new_messages = [
            {"role": "assistant", "content": "",
             "tool_calls": [_call("get_spec", {})]},
            {"role": "tool", "tool_call_id": "call_1", "name": "get_spec",
             "content": '{"spec_hash": "deadbeef"}'},
        ]
        assert actx._persist_tool_context(s, new_messages) == 2
        assert [m.role for m in s.messages] == ["assistant", "tool"]
        # то, ради чего всё: результат вызова уйдёт в контекст следующего хода
        tool_msgs = [m for m in s.context_messages() if m.get("role") == "tool"]
        assert tool_msgs and "deadbeef" in tool_msgs[0]["content"]
        assert tool_msgs[0]["tool_call_id"] == "call_1"

    def test_empty_assistant_stub_is_not_written(self):
        """Реплика-пустышка читалась бы как «помощник промолчал»."""
        s = new_session(PROJECT)
        assert actx._persist_tool_context(
            s, [{"role": "assistant", "content": "   "}]) == 0
        assert s.messages == []

    def test_huge_tool_result_clipped_with_note(self):
        """Молча обрезанный результат — вывод по половине таблицы (A0.6)."""
        s = new_session(PROJECT)
        big = "x" * (actx.RESUME_TOOL_CHARS + 5000)
        actx._persist_tool_context(
            s, [{"role": "tool", "name": "get_spec", "content": big}])
        body = s.messages[0].content
        assert len(body) < actx.RESUME_TOOL_CHARS + 200
        assert "вызови инструмент снова" in body

    def test_service_roles_never_reach_the_feed(self):
        """Аудит инструментов — отдельная панель, а не реплики разговора."""
        s = new_session(PROJECT)
        s.add_message("user", "вопрос")
        actx._persist_tool_context(
            s, [{"role": "assistant", "content": "",
                 "tool_calls": [_call("get_spec", {})]},
                {"role": "tool", "name": "get_spec", "content": "{}"}])
        assert [f.role for f in wsx.feed_items(s.messages)] == ["user"]
        assert wsx.dialog_count(s.messages) == 1

    def test_finished_turn_does_not_duplicate_tool_context(self):
        """У завершённого хода выводы уже в ответе: дубль жёг бы бюджет."""
        s = new_session(PROJECT)
        tr = FakeTransport([_msg("", [_call("get_spec", {})]),
                            _msg("## ОТВЕТ\nГраница 4…14 phr.")])
        res = actx.run_turn(s, Ctx(s), "почему такая граница?", transport=tr,
                            kinds=["readonly"], persist=False)
        assert res.stopped_reason == "final"
        assert [m.role for m in s.messages] == ["user", "assistant"]

    def test_budget_stop_writes_context_through_run_turn(self):
        """Сквозной инвариант: обрыв в БОЕВОМ ходе оставляет контекст.

        Бюджет 0,01 с гарантирует обрыв сразу после первой пачки вызовов.
        Инструмент здесь отказывает (спеки в контексте нет) — и это ровно тот
        случай, который обязан сохраниться: отказ уходит модели как результат
        (A0.6), и повтор не должен запрашивать его заново.
        """
        s = new_session(PROJECT)
        tr = FakeTransport([_msg("", [_call("get_spec", {})]),
                            _msg("не понадобится")])
        res = actx.run_turn(s, Ctx(s), "разбери спеку", transport=tr,
                            kinds=["readonly"], persist=False,
                            time_budget_s=0.01)
        assert res.stopped_reason == "time_budget"
        assert "tool" in [m.role for m in s.messages]


# ======================================================================
# 2. Предохранитель: не заказывать обречённый запрос
# ======================================================================
class TestRequestHeadroom:

    def test_no_new_request_without_headroom(self):
        """Второй запрос НЕ отправлен: денег стоит, ответа не даст.

        Бюджет чуть МЕНЬШЕ порога запаса: время ещё не вышло (прежняя проверка
        после инструментов промолчала бы и заказала генерацию), но на новый
        запрос его заведомо не хватает.
        """
        tr = FakeTransport([_msg("", [_call("get_spec", {})]),
                            _msg("этот ответ не должен быть запрошен")])
        res = llm.run_tool_loop(
            [{"role": "user", "content": "?"}],
            dispatch=lambda n, a: {"ok": 1},
            time_budget_s=cfg.REQUEST_HEADROOM_S - 1.0, transport=tr)
        assert res.stopped_reason == "time_budget"
        assert len(tr.payloads) == 1, "лишний запрос к модели всё-таки ушёл"
        assert "НЕ отправлено" in res.text

    def test_first_request_is_always_made(self):
        """Ход без единого запроса — это не ход, а молчание."""
        tr = FakeTransport([_msg("## ОТВЕТ\nготово")])
        res = llm.run_tool_loop([{"role": "user", "content": "?"}],
                                dispatch=lambda n, a: {"ok": 1},
                                time_budget_s=cfg.MIN_TIME_BUDGET_S,
                                transport=tr)
        assert res.stopped_reason == "final" and len(tr.payloads) == 1


# ======================================================================
# 3. Текст обрыва честен
# ======================================================================
class TestBudgetMessage:

    def test_text_lists_done_tools_and_promises_resume(self):
        tr = FakeTransport([_msg("", [_call("get_spec", {})]),
                            _msg("не понадобится")])

        def slow(name, args):
            time.sleep(0.05)
            return {"ok": 1}

        res = llm.run_tool_loop([{"role": "user", "content": "?"}],
                                dispatch=slow, time_budget_s=0.01,
                                transport=tr)
        assert res.stopped_reason == "time_budget"
        assert "get_spec" in res.text
        assert "продолжит работу с этого места" in res.text

    def test_text_no_longer_claims_context_saved_by_itself(self):
        """Прежняя формулировка обещала то, чего код не делал."""
        txt = llm._budget_text(180.0, llm.LLMResult())
        assert "контекст вызовов сохранён в сессии" not in txt
        assert "бюджет времени" in txt.lower()

    def test_text_points_at_the_real_cause(self):
        """Причина — объём генерации, а не «медленные инструменты»."""
        res = llm.LLMResult()
        res.calls = [{"tool": "validate_project_package", "duration_s": 0.01}]
        txt = llm._budget_text(180.0, res)
        assert "сузьте вопрос" in txt.lower()
        assert "бюджет времени в панели" in txt


# ======================================================================
# 4. Лимиты настраиваются оператором
# ======================================================================
class TestConfigurableLimits:

    def test_budget_read_from_env(self, monkeypatch):
        monkeypatch.setenv("DOE_ASSISTANT_TIME_BUDGET_S", "600")
        assert cfg.time_budget_s() == pytest.approx(600.0)

    def test_garbage_env_falls_back_to_default(self, monkeypatch):
        """Опечатка в настройке не должна ронять помощника."""
        monkeypatch.setenv("DOE_ASSISTANT_TIME_BUDGET_S", "полчаса")
        assert cfg.time_budget_s() == pytest.approx(cfg.DEFAULT_TIME_BUDGET_S)

    def test_too_small_budget_is_lifted_to_floor(self, monkeypatch):
        """«Бюджет 5 с» = ход, который не может завершиться никогда."""
        monkeypatch.setenv("DOE_ASSISTANT_TIME_BUDGET_S", "5")
        assert cfg.time_budget_s() == pytest.approx(cfg.MIN_TIME_BUDGET_S)

    def test_loop_takes_env_limit_without_arguments(self, monkeypatch):
        """Смена настройки действует со следующего хода, без перезапуска."""
        monkeypatch.setenv("DOE_ASSISTANT_MAX_ITERATIONS", "1")
        tr = FakeTransport([_msg("", [_call("get_spec", {})])])
        res = llm.run_tool_loop([{"role": "user", "content": "?"}],
                                dispatch=lambda n, a: {"ok": 1}, transport=tr)
        assert res.stopped_reason == "max_iterations" and res.iterations == 1

    def test_save_limits_writes_env_file(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DOE_ASSISTANT_TIME_BUDGET_S", raising=False)
        path = tmp_path / ".env"
        cfg.save_limits(budget_s=420.0, iterations=12, path=str(path))
        text = path.read_text(encoding="utf-8")
        assert "DOE_ASSISTANT_TIME_BUDGET_S=420" in text
        assert "DOE_ASSISTANT_MAX_ITERATIONS=12" in text
        assert cfg.time_budget_s() == pytest.approx(420.0)

    def test_save_limits_keeps_existing_key(self, tmp_path):
        """Менять бюджет нельзя ценой затирания ключа."""
        path = tmp_path / ".env"
        path.write_text("OPENROUTER_API_KEY=sk-secret\n", encoding="utf-8")
        cfg.save_limits(budget_s=300.0, path=str(path))
        assert "OPENROUTER_API_KEY=sk-secret" in path.read_text(encoding="utf-8")

    def test_save_limits_without_values_refuses(self):
        with pytest.raises(ValueError, match="Нечего сохранять"):
            cfg.save_limits()


# ======================================================================
# 5. Промпт: соразмерность правки (две оси ≠ пакет проекта)
# ======================================================================
class TestProportionalEditPrompt:

    def test_prompt_ranks_tools_from_cheap_to_expensive(self):
        from src.assistant.prompts import architect_system_prompt

        text = architect_system_prompt(project=PROJECT, has_runner=False)
        assert "СОРАЗМЕРНОСТЬ ПРАВКИ" in text
        assert "propose_setup_fields" in text
        assert "ТОЛЬКО если меняется" in text

    def test_prompt_says_process_axes_live_in_form_fields(self):
        """Ровно то знание, которого не хватило 14.08.2026."""
        from src.assistant.prompts import architect_system_prompt

        text = architect_system_prompt(project=PROJECT, has_runner=False)
        assert "setup_process_levels" in text
        assert "ПЕРЕСБОРКА ≠ ПАКЕТ С НУЛЯ" in text

    def test_built_project_does_not_get_setup_field_advice(self):
        """Собранному проекту поля формы не адресуют: там патч и пакет спеки."""
        from src.assistant.prompts import architect_system_prompt

        text = architect_system_prompt(project=PROJECT, has_runner=True)
        assert "ПЕРЕСБОРКА ≠ ПАКЕТ С НУЛЯ" not in text


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
