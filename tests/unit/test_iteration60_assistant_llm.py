# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 60 / ASSISTANT_SPEC — ЦИКЛ вызова инструментов + интернет.

Ассистент-архитектор обязан отвечать ЧИСЛАМИ ИЗ ЯДРА: модель просит
инструмент, мы исполняем его локально и возвращаем результат в диалог. Это
переводит помощника из режима «помню, что там было» в режим «проверил
вызовом», ради которого и затевался слой (запрет §2 системного промпта:
каждое число — либо из инструмента, либо с цитатой).

Сеть в тестах не нужна: транспорт подменяется (`transport=`), поэтому
проверяется именно ПОВЕДЕНИЕ цикла — цепочка вызовов, возврат ошибки
инструмента МОДЕЛИ (а не пользователю стектрейсом), лимиты итераций и
времени, накопление usage, суффикс `:online` и события прогресса для UI.
"""
import json

import pytest

from src.assistant import llm
from src.assistant.session import new_session
from src.assistant.store import save_session


# ----------------------------------------------------------------------
# Фейковый транспорт: сценарий ответов задаётся списком
# ----------------------------------------------------------------------
def _msg(content="", tool_calls=None, usage=None):
    m = {"role": "assistant", "content": content}
    if tool_calls:
        m["tool_calls"] = tool_calls
    body = {"choices": [{"message": m}]}
    if usage:
        body["usage"] = usage
    return body


def _call(name, args, cid="call_1"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


class FakeTransport:
    """Отдаёт заранее заданные ответы и запоминает отправленные payload'ы."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads = []

    def __call__(self, payload, *, key="", timeout=0):
        self.payloads.append(payload)
        if not self.responses:
            raise AssertionError("Фейковый транспорт исчерпан: лишний запрос")
        return self.responses.pop(0)


# ----------------------------------------------------------------------
# 1. Модель и тумблер интернета
# ----------------------------------------------------------------------
def test_online_suffix_added_once_and_removable():
    """`:online` — тумблер, а не другая модель: приписывается ровно один раз."""
    assert llm.online_model("anthropic/claude-sonnet-4.5", web=True) \
        == "anthropic/claude-sonnet-4.5:online"
    assert llm.online_model("anthropic/claude-sonnet-4.5:online", web=True) \
        == "anthropic/claude-sonnet-4.5:online"
    assert llm.online_model("anthropic/claude-sonnet-4.5:online", web=False) \
        == "anthropic/claude-sonnet-4.5"
    assert llm.is_online(llm.online_model("x/y", web=True)) is True


def test_web_flag_reaches_payload():
    tr = FakeTransport([_msg("ответ с вебом")])
    res = llm.run_tool_loop([{"role": "user", "content": "TDS?"}],
                            model="a/b", web=True, transport=tr)
    assert tr.payloads[0]["model"] == "a/b:online"
    assert res.web is True and res.model.endswith(":online")


def test_no_key_no_transport_explains_itself(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_KEY", raising=False)
    with pytest.raises(llm.LLMError, match="OPENROUTER_API_KEY"):
        llm.chat_once([{"role": "user", "content": "x"}])


# ----------------------------------------------------------------------
# 2. Простой ответ без инструментов
# ----------------------------------------------------------------------
def test_plain_answer_returns_text_and_message():
    tr = FakeTransport([_msg("Диапазон охватывает оптимум.",
                             usage={"total_tokens": 120})])
    res = llm.run_tool_loop([{"role": "user", "content": "верх 0,5?"}],
                            transport=tr)
    assert res.text.startswith("Диапазон")
    assert res.stopped_reason == "final" and res.iterations == 1
    assert res.calls == [] and res.usage["total_tokens"] == 120
    assert [m["role"] for m in res.new_messages] == ["assistant"]


# ----------------------------------------------------------------------
# 3. Один вызов инструмента и цепочка из двух
# ----------------------------------------------------------------------
def test_single_tool_call_result_returned_to_model():
    tr = FakeTransport([
        _msg("", [_call("get_spec", {})], usage={"total_tokens": 300}),
        _msg("q=19, dim_z=16 — спека v2.", usage={"total_tokens": 90}),
    ])
    seen = []

    def dispatch(name, args):
        seen.append((name, args))
        return {"q": 19, "dim_z": 16, "spec_hash": "c63b7e16"}

    res = llm.run_tool_loop([{"role": "user", "content": "что в спеке?"}],
                            dispatch=dispatch, tools=[{"type": "function"}],
                            transport=tr)

    assert seen == [("get_spec", {})]
    assert res.text.startswith("q=19")
    assert res.iterations == 2 and res.n_tool_calls == 1
    assert res.usage["total_tokens"] == 390
    roles = [m["role"] for m in res.new_messages]
    assert roles == ["assistant", "tool", "assistant"]
    # результат инструмента ушёл модели вторым запросом
    second = tr.payloads[1]["messages"]
    assert second[-1]["role"] == "tool" and "dim_z" in second[-1]["content"]


def test_two_tool_calls_in_one_message():
    """Модель может попросить два инструмента сразу — исполняем оба по порядку."""
    tr = FakeTransport([
        _msg("", [_call("get_spec", {}, "c1"),
                  _call("explain_node", {"name": "PBNK_3355"}, "c2")]),
        _msg("готово"),
    ])
    calls = []

    res = llm.run_tool_loop([{"role": "user", "content": "?"}],
                            dispatch=lambda n, a: calls.append(n) or {"ok": 1},
                            transport=tr)
    assert calls == ["get_spec", "explain_node"]
    assert [c["tool"] for c in res.calls] == ["get_spec", "explain_node"]
    assert sum(1 for m in res.new_messages if m["role"] == "tool") == 2


def test_chained_tool_calls_across_iterations():
    tr = FakeTransport([
        _msg("", [_call("get_spec", {}, "c1")]),
        _msg("", [_call("simulate_bounds", {"patch": {}}, "c2")]),
        _msg("corr(UV, DINP) = 0.12 — трапеция, не клин."),
    ])
    res = llm.run_tool_loop([{"role": "user", "content": "?"}],
                            dispatch=lambda n, a: {"tool": n}, transport=tr)
    assert res.iterations == 3 and res.n_tool_calls == 2
    assert "0.12" in res.text and res.stopped_reason == "final"


# ----------------------------------------------------------------------
# 4. Ошибки инструментов — МОДЕЛИ, а не пользователю стектрейсом (A0.6)
# ----------------------------------------------------------------------
def test_tool_error_is_returned_to_model_not_raised():
    tr = FakeTransport([
        _msg("", [_call("validate_spec", {"patch": {"role": "SHARE_CLOSURE"}})]),
        _msg("Отказано: при k≥3 closure запрещён."),
    ])

    def dispatch(name, args):
        raise ValueError("k=3 ⇒ SHARE_CLOSURE запрещён")

    res = llm.run_tool_loop([{"role": "user", "content": "?"}],
                            dispatch=dispatch, transport=tr)

    assert res.calls[0]["ok"] is False
    assert "SHARE_CLOSURE" in res.calls[0]["error"]
    tool_msg = [m for m in res.new_messages if m["role"] == "tool"][0]
    assert "ОШИБКА ИНСТРУМЕНТА" in tool_msg["content"]
    assert res.text.startswith("Отказано")     # модель учла отказ


def test_broken_tool_arguments_reported_as_tool_error():
    bad = {"id": "c1", "function": {"name": "get_spec", "arguments": "{не json"}}
    tr = FakeTransport([_msg("", [bad]), _msg("понял")])
    res = llm.run_tool_loop([{"role": "user", "content": "?"}],
                            dispatch=lambda n, a: {"ok": 1}, transport=tr)
    assert res.calls[0]["ok"] is False and "JSON" in res.calls[0]["error"]


def test_huge_tool_result_truncated_with_note():
    """Усечение объявляется: молча обрезанная таблица = «данных нет»."""
    tr = FakeTransport([_msg("", [_call("get_runs", {})]), _msg("ок")])
    res = llm.run_tool_loop(
        [{"role": "user", "content": "?"}],
        dispatch=lambda n, a: "x" * (llm.MAX_TOOL_RESULT_CHARS + 5000),
        transport=tr)
    tool_msg = [m for m in res.new_messages if m["role"] == "tool"][0]
    assert "усечён" in tool_msg["content"]
    assert len(tool_msg["content"]) < llm.MAX_TOOL_RESULT_CHARS + 500


def test_tool_call_without_dispatch_is_explicit():
    """Инструменты недоступны — говорим прямо, а не выдаём пустой ответ."""
    tr = FakeTransport([_msg("", [_call("get_spec", {})])])
    res = llm.run_tool_loop([{"role": "user", "content": "?"}], transport=tr)
    assert res.stopped_reason == "no_dispatch"


# ----------------------------------------------------------------------
# 5. Лимиты цикла
# ----------------------------------------------------------------------
def test_max_iterations_stops_with_honest_message():
    tr = FakeTransport([_msg("", [_call("get_spec", {})]) for _ in range(5)])
    res = llm.run_tool_loop([{"role": "user", "content": "?"}],
                            dispatch=lambda n, a: {"ok": 1},
                            max_iterations=3, transport=tr)
    assert res.stopped_reason == "max_iterations" and res.iterations == 3
    assert "предел" in res.text and "3" in res.text


def test_time_budget_stops_loop():
    tr = FakeTransport([_msg("", [_call("slow_tool", {})]),
                        _msg("не понадобится")])

    def slow(name, args):
        import time as _t
        _t.sleep(0.05)
        return {"ok": 1}

    res = llm.run_tool_loop([{"role": "user", "content": "?"}], dispatch=slow,
                            time_budget_s=0.01, transport=tr)
    assert res.stopped_reason == "time_budget" and "Бюджет времени" in res.text


def test_max_iterations_must_be_positive():
    with pytest.raises(ValueError, match="max_iterations"):
        llm.run_tool_loop([], max_iterations=0, transport=FakeTransport([]))


# ----------------------------------------------------------------------
# 6. Прогресс для UI (пользователь не должен думать, что «зависло»)
# ----------------------------------------------------------------------
def test_progress_events_cover_tool_run():
    tr = FakeTransport([_msg("", [_call("run_pytest", {"subset": "iter45"})]),
                        _msg("тесты зелёные")])
    events = []
    llm.run_tool_loop([{"role": "user", "content": "?"}],
                      dispatch=lambda n, a: {"passed": 13},
                      transport=tr, on_event=events.append)

    kinds = [e["kind"] for e in events]
    assert kinds == ["llm_request", "tool_start", "tool_end", "llm_request",
                     "done"]
    assert all("elapsed_s" in e for e in events)

    caps = [llm.progress_caption(e) for e in events]
    assert "run_pytest" in caps[1] and caps[1].startswith("🔧")
    assert caps[-1].startswith("🏁") and "готово" in caps[-1]


def test_failing_on_event_does_not_break_the_turn():
    """Сбой отрисовки прогресса не должен ронять ответ ассистента."""
    tr = FakeTransport([_msg("ответ")])

    def boom(_e):
        raise RuntimeError("UI упал")

    res = llm.run_tool_loop([{"role": "user", "content": "?"}], transport=tr,
                            on_event=boom)
    assert res.text == "ответ"


def test_progress_caption_marks_failed_tool():
    txt = llm.progress_caption({"kind": "tool_end", "tool": "preflight",
                                "ok": False, "duration_s": 1.2,
                                "error": "gate failed"})
    assert txt.startswith("⛔") and "gate failed" in txt


# ----------------------------------------------------------------------
# 7. Транспорт: HTTP-ошибки объясняются человеку
# ----------------------------------------------------------------------
def test_http_error_message_has_hint(monkeypatch):
    import io
    import urllib.error

    def raise_402(*a, **kw):
        raise urllib.error.HTTPError(
            llm.OPENROUTER_URL, 402, "Payment Required", {},
            io.BytesIO(b'{"error":"insufficient credits"}'))


    monkeypatch.setattr(llm.urllib.request, "urlopen", raise_402)
    with pytest.raises(llm.LLMError, match="402"):
        llm.chat_once([{"role": "user", "content": "x"}], key="sk-test")


def test_unexpected_body_rejected():
    with pytest.raises(llm.LLMError, match="Неожиданный ответ"):
        llm.chat_once([{"role": "user", "content": "x"}], key="k",
                      transport=lambda payload, key="", timeout=0: {"oops": 1})


# ----------------------------------------------------------------------
# 8. Стыковка с сессией: ход целиком ложится в память проекта
# ----------------------------------------------------------------------
def test_turn_messages_persist_into_session(tmp_path):
    tr = FakeTransport([_msg("", [_call("get_spec", {})]),
                        _msg("q=19", usage={"total_tokens": 500})])
    s = new_session("pvc_edge_v1", model="anthropic/claude-sonnet-4.5",
                    web_enabled=True)
    s.add_message("user", "что в спеке?")

    res = llm.run_tool_loop(s.context_messages(), dispatch=lambda n, a: {"q": 19},
                            web=s.web_enabled, transport=tr)
    for m in res.new_messages:
        s.add_message(m["role"], m.get("content", ""),
                      tool_calls=m.get("tool_calls", []),
                      tool_call_id=m.get("tool_call_id", ""),
                      name=m.get("name", ""),
                      model=res.model if m["role"] == "assistant" else "",
                      web=res.web if m["role"] == "assistant" else False)
    s.add_usage(res.usage)
    save_session(s, tmp_path)

    from src.assistant.store import load_session
    loaded = load_session(tmp_path, "pvc_edge_v1")
    assert [m.role for m in loaded.messages] == ["user", "assistant", "tool",
                                                 "assistant"]
    assert loaded.messages[1].tool_calls and loaded.messages[2].name == "get_spec"
    assert loaded.messages[-1].web is True
    assert loaded.usage["total_tokens"] == 500
