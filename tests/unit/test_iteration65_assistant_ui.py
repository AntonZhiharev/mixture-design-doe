# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 65 / ASSISTANT_SPEC — док справа и контекст ПО МЕСТУ (ui_focus).

Чат ассистента переезжает в правую колонку и виден на КАЖДОМ шаге потока.
Отсюда новая обязанность: ассистент должен знать, где сейчас человек, и
отвечать про то, на что тот смотрит («Объясни эту ось», «Что изменится,
если…»). Проверяется здесь ровно это — и ровно теми чистыми функциями,
которые пойдут в интерфейс:

  * **фокус читается из обычного словаря** (:func:`context.focus_from_state`),
    поэтому «контекст по месту» тестируется без запуска Streamlit;
  * **подстановка предмета** («эту ось» → «ось DINP») делается только тем, что
    ЕСТЬ в фокусе, и НЕ даёт права отвечать по памяти: блок фокуса прямо
    говорит, что числа по-прежнему считают инструменты;
  * **подсказки шага генерируются** и каждая маршрутизируется роутером iter64
    в осмысленный вид: «кнопка есть, а инструмента под неё нет» невозможно;
  * **ход — одна функция** :func:`context.run_turn`: вопрос человека остаётся
    в истории как сказан, модели уходит разрешённый, вызовы попадают в сессию
    и в ``tool_calls.jsonl``, отказ модели не роняет док;
  * **применяет патч человек**: :func:`context.human_apply` выдаёт разовый
    токен и гасит его, а попытка модели сделать то же самое отбивается
    диспетчером (iter63) — ход при этом не ломается.
"""
import json

import pytest

from src.assistant import context, store, views
from src.assistant.context import (FILES_MARK, FOCUS_KEY, FOCUS_MARK,
                                    FOCUS_SECTIONS, UNKNOWN_SECTION, UiFocus,
                                    build_turn_messages, focus_block,
                                    focus_caption, focus_from_state,
                                    human_apply, human_reject, normalize_focus,
                                    resolve_question, run_turn, section,
                                    suggested_questions)
from src.assistant.files import attach_file
from src.assistant.llm import LLMError
from src.assistant.prompts import KIND_CLARIFY, PROMPT_MARK, route
from src.assistant.session import PATCH_APPLIED, PATCH_REJECTED, new_session
from src.assistant.tools import ToolError
from src.design.phr_sampler import PhrSpec

PROJECT = "pvc_edge_v1"

#: Та же референсная геометрия, что в iter61/63/64 (golden iter45/49/50).
NODES = [
    {"name": "RESIN", "role": "FIXED", "value": 100.0},
    {"name": "DINP", "role": "ABSOLUTE", "range": [4.0, 14.0]},
    {"name": "ESO", "role": "FIXED", "value": 2.5},
    {"name": "SOFT", "role": "GROUP_TOTAL", "range": [5.0, 15.0],
     "members": ["PBNK", "CPE"]},
    {"name": "PBNK", "role": "SHARE_FREE", "group": "SOFT",
     "share_range": [0.0, 0.70], "max_phr": 8.0},
    {"name": "CPE", "role": "SHARE_CLOSURE", "group": "SOFT", "min_phr": 3.0},
    {"name": "TiO2", "role": "ABSOLUTE", "range": [0.3, 8.0], "scale": "log"},
]


def _spec() -> PhrSpec:
    return PhrSpec.from_dicts(NODES)


def _ctx(tmp_path, session=None):
    from src.assistant.tools import ToolContext
    return ToolContext(spec=_spec(), session=session or new_session(PROJECT),
                       root=str(tmp_path), project=PROJECT)


# ----------------------------------------------------------------------
# Транспорт «модель» без сети (как в iter60/63/64)
# ----------------------------------------------------------------------
def _fn(name, args=None):
    return {"id": f"call_{name}", "type": "function",
            "function": {"name": name,
                         "arguments": json.dumps(args or {}, ensure_ascii=False)}}


def _answer(text, calls=None):
    msg = {"role": "assistant", "content": text}
    if calls:
        msg["tool_calls"] = list(calls)
    return {"choices": [{"message": msg}], "usage": {"total_tokens": 11}}


def _scripted(script):
    seq = list(script)

    def transport(payload, *, key="", timeout=0):
        return seq.pop(0) if seq else _answer("готово")

    return transport


def _boom(payload, *, key="", timeout=0):
    raise LLMError("OpenRouter HTTP 401. Проверьте OPENROUTER_API_KEY.")


# ======================================================================
# 1. Фокус: карта шагов и чтение из session_state
# ======================================================================
def test_sections_have_unique_keys_and_asks():
    keys = [s.key for s in FOCUS_SECTIONS]
    assert len(keys) == len(set(keys))
    assert all(s.title and s.doing and s.asks for s in FOCUS_SECTIONS)


def test_unknown_section_is_not_an_error():
    """Интерфейс развивается быстрее спеки: незнакомый шаг не роняет док."""
    assert section("не-такой-шаг") is UNKNOWN_SECTION
    assert section("") is UNKNOWN_SECTION
    assert "не строй догадок" in UNKNOWN_SECTION.doing


def test_normalize_focus_from_plain_dict():
    """Секции UI публикуют ОБЫЧНЫЙ словарь — без импорта слоя ассистента."""
    f = normalize_focus({"section": "spec", "node": "DINP",
                         "recipe_phr": [100, 8.0, "2.5"]})
    assert (f.section_key, f.node) == ("spec", "DINP")
    assert f.recipe_phr == [100.0, 8.0, 2.5]
    assert normalize_focus(None).is_empty()
    assert normalize_focus(f) is f


def test_normalize_focus_survives_broken_recipe():
    f = normalize_focus({"section": "weighing", "recipe_phr": ["мусор"]})
    assert f.recipe_phr is None and f.section_key == "weighing"


def test_focus_from_state_prefers_section_over_manual_selector():
    """Место, где стоит человек, — факт; ручной селектор лишь уточняет."""
    st = {FOCUS_KEY: {"section": "evolution", "node": "TiO2"},
          "assistant_focus_section": "spec", "assistant_focus_node": "DINP"}
    f = focus_from_state(st)
    assert (f.section_key, f.node) == ("evolution", "TiO2")


def test_focus_from_state_falls_back_to_existing_ui_keys():
    f = focus_from_state({"camp_branch": "b_root",
                          "assistant_focus_section": "branch"})
    assert f.section_key == "branch" and f.branch == "b_root"
    assert focus_from_state({}).is_empty()
    assert focus_from_state(None).is_empty()


def test_focus_caption_names_node_and_branch():
    cap = focus_caption(UiFocus(section_key="spec", node="DINP", branch="b1"))
    assert "DINP" in cap and "b1" in cap and "phr-спека" in cap


def test_focus_block_forbids_answering_from_memory():
    """Фокус объясняет ЧТО спрашивают, но не заменяет вызов инструмента."""
    txt = focus_block({"section": "spec", "node": "DINP"})
    assert "DINP" in txt and "эта ось" in txt
    assert "не заменяет инструменты" in txt


def test_focus_block_shows_recipe_and_note():
    txt = focus_block(UiFocus(section_key="weighing", recipe_phr=[100.0, 8.0],
                              note="δ = 0.05 phr"))
    assert "100, 8" in txt and "0.05 phr" in txt


# ======================================================================
# 2. Вопрос по месту
# ======================================================================
def test_resolve_question_substitutes_focus_node():
    f = UiFocus(section_key="spec", node="DINP")
    assert resolve_question("Объясни эту ось", f) == "Объясни ось DINP"
    assert "узел DINP" in resolve_question("Почему этот узел зажат?", f)


def test_resolve_question_without_focus_changes_nothing():
    """Лучше честное «какую ось?», чем подстановка наугад."""
    assert resolve_question("Объясни эту ось", None) == "Объясни эту ось"
    assert resolve_question("Объясни эту ось", UiFocus()) == "Объясни эту ось"


def test_resolve_question_substitutes_branch():
    f = UiFocus(section_key="branch", branch="b_soft")
    assert "ветку b_soft" in resolve_question("Что делать с этой веткой?", f)


@pytest.mark.parametrize("sec", [s.key for s in FOCUS_SECTIONS])
def test_every_suggestion_routes_somewhere(sec):
    """Подсказка обязана вести к инструментам, а не в «уточните»."""
    f = UiFocus(section_key=sec, node="DINP", branch="b1")
    sugs = suggested_questions(f)
    assert sugs, f"у шага {sec} нет подсказок"
    for s in sugs:
        assert s.kind != KIND_CLARIFY, (sec, s.question)
        assert s.enabled and s.question


def test_suggestion_without_node_stays_visible_but_disabled():
    """Исчезнувшая кнопка читалась бы как «здесь так спрашивать нельзя»."""
    sugs = suggested_questions(UiFocus(section_key="spec"))
    off = [s for s in sugs if not s.enabled]
    assert off and all(s.why for s in off)
    assert any("узел не выбран" in s.why for s in off)
    assert len(sugs) == len(section("spec").asks)   # ничего не пропало


def test_suggestion_marks_unbuilt_project():
    """«Не проверено» ≠ «пройдено»: без проекта статус-вопрос честно помечен."""
    sugs = suggested_questions(UiFocus(section_key="seed", node="DINP"),
                               has_runner=False)
    status = [s for s in sugs if s.kind == "status"]
    assert status and all("не собран" in s.why for s in status)


def test_suggestion_tools_match_router():
    s = suggested_questions(UiFocus(section_key="spec", node="DINP"))[0]
    assert tuple(route(s.question).tools) == s.tools


# ======================================================================
# 3. Сборка сообщений хода
# ======================================================================
def test_turn_messages_hold_single_prompt_focus_and_question():
    s = new_session(PROJECT)
    s.add_message("user", "старый вопрос")
    msgs = build_turn_messages(s, question="Объясни эту ось",
                               focus={"section": "spec", "node": "DINP"},
                               spec_hash="abc123")
    heads = [m for m in msgs if str(m["content"]).startswith(PROMPT_MARK)]
    focus = [m for m in msgs if str(m["content"]).startswith(FOCUS_MARK)]
    assert len(heads) == 1 and len(focus) == 1
    assert msgs[0]["content"].startswith(PROMPT_MARK)      # инструкция первой
    assert msgs[-1]["content"] == "Объясни ось DINP"       # вопрос разрешён
    assert "abc123" in msgs[0]["content"]


def test_turn_messages_do_not_accumulate_focus_between_turns():
    """Фокус живёт только в сборке: вчерашнее место не всплывёт завтра."""
    s = new_session(PROJECT)
    build_turn_messages(s, question="раз", focus={"section": "spec"})
    msgs = build_turn_messages(s, question="два", focus={"section": "seed"})
    focus = [m for m in msgs if str(m["content"]).startswith(FOCUS_MARK)]
    # iter74: шаг называется «Стартовый план опытов» (без внутреннего «seed»).
    assert len(focus) == 1 and "Стартовый план опытов" in focus[0]["content"]


def test_turn_messages_include_attachment_digest(tmp_path):
    s = new_session(PROJECT)
    attach_file(s, tmp_path, "TDS_TiO2.txt",
                "ПАСПОРТ TiO2\nd50 = 0.28 мкм\n".encode("utf-8"),
                project=PROJECT)
    msgs = build_turn_messages(s, question="что взять в спеку?", focus=None)
    files = [m for m in msgs if str(m["content"]).startswith(FILES_MARK)]
    assert len(files) == 1
    assert "TDS_TiO2.txt" in files[0]["content"] and "d50" in files[0]["content"]
    assert "read_attachment" in files[0]["content"]


def test_turn_messages_say_project_is_not_built():
    s = new_session(PROJECT)
    msgs = build_turn_messages(s, question="preflight?", has_runner=False)
    assert "НЕ собран" in msgs[0]["content"]


def test_empty_focus_adds_no_block():
    s = new_session(PROJECT)
    msgs = build_turn_messages(s, question="привет", focus=None)
    assert not [m for m in msgs if str(m["content"]).startswith(FOCUS_MARK)]


# ======================================================================
# 4. Ход по месту (run_turn)
# ======================================================================
def test_run_turn_keeps_human_words_and_calls_core(tmp_path):
    """История хранит сказанное человеком, модель получает разрешённый вопрос."""
    s = new_session(PROJECT)
    ctx = _ctx(tmp_path, s)
    res = run_turn(
        s, ctx, "Объясни эту ось", focus={"section": "spec", "node": "DINP"},
        spec_hash=_spec().spec_hash(),
        transport=_scripted([
            _answer("", [_fn("explain_node", {"name": "DINP"})]),
            _answer("## ОТВЕТ\nВерх DINP — договорённость.\n\n"
                    "## ЧИСЛА\nexplain_node(DINP): [4, 14] phr.")]))
    assert res.ok and res.kind == "explain"
    assert res.question == "Объясни эту ось"           # как сказал человек
    assert res.resolved == "Объясни ось DINP"          # как ушло модели
    assert [c["tool"] for c in res.calls] == ["explain_node"]
    assert s.messages[0].content == "Объясни эту ось"
    assert s.messages[-1].role == "assistant"
    assert set(res.sections) == {"ОТВЕТ", "ЧИСЛА"}


def test_run_turn_writes_audit_to_session_and_journal(tmp_path):
    """Разбор обязан воспроизводиться через неделю (§3.7)."""
    s = new_session(PROJECT)
    ctx = _ctx(tmp_path, s)
    run_turn(s, ctx, "Почему диапазон DINP такой?",
             transport=_scripted([
                 _answer("", [_fn("explain_node", {"name": "DINP"})]),
                 _answer("## ОТВЕТ\nтехлимит")]))
    assert [c.tool for c in s.tool_calls] == ["explain_node"]
    log = store.read_log(tmp_path, PROJECT, "tool_calls")
    assert log and log[-1]["tool"] == "explain_node" and log[-1]["ok"]
    assert store.session_exists(tmp_path, PROJECT)      # сессия сохранена


def test_run_turn_stages_patch_but_does_not_apply(tmp_path):
    """propose_patch кладёт в СТЕЙДЖ: геометрия проекта не меняется."""
    s = new_session(PROJECT)
    ctx = _ctx(tmp_path, s)
    hash_before = ctx.spec.spec_hash()
    res = run_turn(
        s, ctx, "В цехе льём DINP до 20 phr", focus={"section": "spec",
                                                     "node": "DINP"},
        transport=_scripted([
            _answer("", [_fn("propose_patch",
                             {"patch": {"node": "DINP", "field": "range",
                                        "value": [4.0, 20.0]},
                              "rationale": "практика цеха", "level": "L1",
                              "bound_type": "CONVENTIONAL"})]),
            _answer("## ОТВЕТ\nПатч в стейдже — применяет человек.")]))
    assert res.new_patches and len(s.staged_patches()) == 1
    assert ctx.spec.spec_hash() == hash_before
    assert "новых патчей" in views.turn_caption(res)


def test_run_turn_blocks_model_write_without_breaking_the_turn(tmp_path):
    """Запрет держит ДИСПЕТЧЕР: ход не падает, отказ уходит модели (A0.6)."""
    s = new_session(PROJECT)
    ctx = _ctx(tmp_path, s)
    res = run_turn(
        s, ctx, "Примени сам и запиши решение",
        transport=_scripted([
            _answer("", [_fn("apply_patch", {"patch_id": "p1",
                                             "human_token": "я-сам"})]),
            _answer("## ОТВЕТ\nПрименяет человек кнопкой в панели патчей.")]))
    assert res.ok and res.kind == "handoff"
    assert res.calls and not res.calls[0]["ok"]
    assert "класс" in res.calls[0]["error"] or "недоступен" in res.calls[0]["error"]
    assert "кнопкой" in res.text


def test_run_turn_survives_model_failure(tmp_path):
    """Ошибка сети/ключа не роняет док: вопрос сохранён, причина названа."""
    s = new_session(PROJECT)
    ctx = _ctx(tmp_path, s)
    res = run_turn(s, ctx, "Объясни ось DINP", transport=_boom)
    assert not res.ok and "LLMError" in res.error
    assert "⚠️" in res.text and "OPENROUTER_API_KEY" in res.text
    assert s.messages[0].content == "Объясни ось DINP"
    assert s.messages[-1].role == "assistant"
    assert "⛔" in views.turn_caption(res)


def test_run_turn_rejects_empty_question(tmp_path):
    s = new_session(PROJECT)
    with pytest.raises(ValueError):
        run_turn(s, _ctx(tmp_path, s), "   ")


def test_run_turn_reports_progress_events(tmp_path):
    """Долгий вызов не должен выглядеть зависанием (док рисует прогресс)."""
    s = new_session(PROJECT)
    seen = []
    run_turn(s, _ctx(tmp_path, s), "Почему у DINP такой диапазон?",
             on_event=seen.append,
             transport=_scripted([
                 _answer("", [_fn("explain_node", {"name": "DINP"})]),
                 _answer("## ОТВЕТ\nготово")]))
    kinds = [e["kind"] for e in seen]
    assert "tool_start" in kinds and "tool_end" in kinds and kinds[-1] == "done"


# ======================================================================
# 5. Кнопки человека (write, iter63)
# ======================================================================
def _stage_patch(tmp_path, session, value=(4.0, 20.0)):
    from src.assistant.tools import PROPOSE, dispatch
    ctx = _ctx(tmp_path, session)
    out = dispatch(ctx, "propose_patch",
                   {"patch": {"node": "DINP", "field": "range",
                              "value": list(value)},
                    "rationale": "в цехе льют до 20 phr", "level": "L1",
                    "bound_type": "CONVENTIONAL"},
                   allowed_kinds=[PROPOSE])
    return ctx, out["patch_ids"][0]


def test_human_apply_applies_patch_and_logs_decision(tmp_path):
    s = new_session(PROJECT)
    ctx, pid = _stage_patch(tmp_path, s)
    out = human_apply(ctx, pid, note="решили на планёрке", author="технолог")
    assert out["ok"] and s.patch_by_id(pid).status == PATCH_APPLIED
    assert out["spec_hash_before"] != out["spec_hash_after"]
    decisions = store.read_log(tmp_path, PROJECT, "decisions")
    assert decisions and decisions[-1]["kind"] == "apply_patch"
    assert "сохраните кампанию" in out["persist_hint"].lower()


def test_human_apply_is_single_use(tmp_path):
    """Кнопка выдаёт РАЗОВЫЙ токен: применить дважды нельзя."""
    s = new_session(PROJECT)
    ctx, pid = _stage_patch(tmp_path, s)
    human_apply(ctx, pid)
    with pytest.raises(ToolError):
        human_apply(ctx, pid)


def test_human_reject_is_logged_like_apply(tmp_path):
    """Отказ фиксируется наравне с применением — спор разрешает журнал."""
    s = new_session(PROJECT)
    ctx, pid = _stage_patch(tmp_path, s)
    out = human_reject(ctx, pid, "верх 14 — предел линии", author="технолог")
    assert out["ok"] and s.patch_by_id(pid).status == PATCH_REJECTED
    kinds = [r.get("kind") for r in store.read_log(tmp_path, PROJECT, "decisions")]
    assert "reject_patch" in kinds


# ======================================================================
# 6. Показ (те же чистые хелперы, что и в доке)
# ======================================================================
def test_suggestions_dataframe_keeps_disabled_rows():
    df = views.suggestions_dataframe(suggested_questions(
        UiFocus(section_key="spec")))
    assert list(df.columns) == ["кнопка", "вопрос", "маршрут", "инструменты",
                                "доступна", "почему"]
    assert (df["доступна"] == "нет").any()
    assert df.loc[df["доступна"] == "нет", "почему"].iloc[0] != "—"


def test_turn_caption_reports_route_and_calls(tmp_path):
    s = new_session(PROJECT)
    res = run_turn(s, _ctx(tmp_path, s), "Объясни ось DINP",
                   transport=_scripted([
                       _answer("", [_fn("explain_node", {"name": "DINP"})]),
                       _answer("## ОТВЕТ\nготово")]))
    cap = views.turn_caption(res)
    assert "маршрут" in cap and "вызовов инструментов: 1" in cap
    assert "разделы: ОТВЕТ" in cap
    assert views.turn_caption(None) == "хода не было"


def test_dock_module_exposes_pure_helpers():
    """UI-док — тонкий слой: вся логика уже проверена выше."""
    pytest.importorskip("streamlit")
    from src.apps import assistant_dock

    assert callable(assistant_dock.render_assistant_dock)
    assert callable(assistant_dock.dock_focus)
