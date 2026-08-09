# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 64 / ASSISTANT_SPEC — промпт архитектора и маршрутизация (§8).

Промпт — рабочая инструкция, а не «характер» помощника, поэтому его можно и
нужно проверять тестом:

  * **каталог инструментов собирается ИЗ РЕЕСТРА** — новый инструмент виден
    модели сразу, а класс ``write`` не попадает в него ВООБЩЕ (применение —
    акт человека, iter63, и промпт не должен намекать на обратное);
  * **иерархия знания L1 > L2 > L3** и запрет усреднять конфликт названы
    словами: расхождение цеха и литературы уходит в ``OPEN_QUESTIONS``;
  * **8 golden-сценариев §8 маршрутизируются верно** — типовой вопрос
    технолога приводит к тем инструментам, без которых ответ был бы работой
    по памяти. Маршрут проверяется ЧИСТЫМ роутером (:func:`prompts.route`):
    «ассистент отвечает правильно» нельзя закрывать тестом, который ходит в
    сеть, а вот «правильно ли выбран путь» — можно.

Плюс контракт хода: попытка модели применить патч самой блокируется
диспетчером, ход не падает (отказ уходит модели), и сверка
:func:`prompts.check_routing` показывает нарушение маршрута явно.
"""
import pytest

from src.assistant import prompts
from src.assistant.prompts import (GOLDEN_SCENARIOS, HUMAN_ONLY, KIND_CLARIFY,
                                    KIND_HANDOFF, KIND_PROPOSE, KIND_WHATIF,
                                    architect_system_prompt, check_routing,
                                    missing_sections, parse_sections, route,
                                    route_caption, route_scenario, scenario,
                                    tool_catalog, with_system)
from src.assistant.session import new_session
from src.assistant.tools import (AGENT_KINDS, READONLY, TOOLS, WRITE,
                                  ToolContext, tool_names, tool_specs)
from src.assistant.tools.registry import ToolDef, dispatcher
from src.assistant.views import routing_dataframe, scenarios_dataframe
from src.assistant import llm
from src.design.phr_sampler import PhrSpec

PROJECT = "pvc_edge_v1"

#: Референсная геометрия iter61/63 (golden iter45/49/50).
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
    {"name": "UV", "role": "ABSOLUTE_CAPPED", "range": [0.05, 0.30],
     "scale": "log", "cap_to": ["DINP", "ESO"], "cap_ratio": 0.03},
]


def _spec() -> PhrSpec:
    return PhrSpec.from_dicts(NODES)


def _ctx(tmp_path) -> ToolContext:
    return ToolContext(spec=_spec(), session=new_session(PROJECT),
                       root=str(tmp_path), project=PROJECT)


# ----------------------------------------------------------------------
# Транспорт «модель» без сети (как в iter60/63)
# ----------------------------------------------------------------------
def _fn(name, args=None):
    import json
    return {"id": f"call_{name}", "type": "function",
            "function": {"name": name,
                         "arguments": json.dumps(args or {}, ensure_ascii=False)}}


def _answer(text, calls=None):
    msg = {"role": "assistant", "content": text}
    if calls:
        msg["tool_calls"] = list(calls)
    return {"choices": [{"message": msg}], "usage": {"total_tokens": 10}}


def _scripted(script):
    seq = list(script)

    def transport(payload, *, key="", timeout=0):
        return seq.pop(0) if seq else _answer("готово")

    return transport


# ======================================================================
# 1. Системный промпт
# ======================================================================
class TestSystemPrompt:
    def test_role_is_architect_not_chat(self):
        text = architect_system_prompt(project=PROJECT)
        assert "АРХИТЕКТОР" in text
        # Главный тезис слоя: числа приходят из ядра, а не из памяти модели.
        assert "ЧИСЛАМИ ИЗ ЯДРА" in text

    def test_knowledge_hierarchy_declared(self):
        text = architect_system_prompt()
        assert "L1 > L2 > L3" in text
        for level in ("L1", "L2", "L3"):
            assert level in text

    def test_conflict_is_signal_not_average(self):
        # Усреднение конфликта цеха и литературы выглядит как согласие,
        # которого не было — промпт обязан это запрещать явно.
        text = architect_system_prompt()
        assert "OPEN_QUESTIONS" in text
        assert "усреднить" in text

    def test_answer_format_sections(self):
        text = architect_system_prompt()
        for section in ("## ОТВЕТ", "## PATCH", "## OPEN_QUESTIONS"):
            assert section in text
        for field in ("bound_type", "level", "confidence", "affects_hash"):
            assert field in text

    def test_hard_limits_named(self):
        text = architect_system_prompt()
        assert "tests/" in text          # golden-числа — контракт
        assert "spec_hash" in text       # сдвиг отпечатка называется вслух
        assert "Не проверено" in text or "«Не проверено»" in text

    def test_catalog_comes_from_registry(self):
        catalog = tool_catalog()
        for name in tool_names(AGENT_KINDS):
            assert f"`{name}`" in catalog

    def test_write_tools_never_in_catalog(self):
        # Класс write недостижим по построению (iter63); промпт не должен
        # даже намекать модели, что она может применить патч сама.
        catalog = tool_catalog()
        for name in tool_names([WRITE]):
            assert f"`{name}`" not in catalog

    def test_new_tool_appears_without_editing_prompt(self):
        # Список, переписанный руками, разъезжается с кодом: проверяем, что
        # каталог именно ГЕНЕРИРУЕТСЯ.
        TOOLS["dummy_probe_tool"] = ToolDef(
            name="dummy_probe_tool", description="проба пера",
            parameters={"type": "object", "properties": {}},
            fn=lambda ctx: None, kind=READONLY)
        try:
            assert "`dummy_probe_tool`" in tool_catalog()
            assert "проба пера" in architect_system_prompt()
        finally:
            TOOLS.pop("dummy_probe_tool", None)

    def test_long_running_marked(self):
        assert "⏳" in tool_catalog()

    def test_project_not_built_is_said_aloud(self):
        text = architect_system_prompt(has_runner=False)
        assert "НЕ собран" in text
        built = architect_system_prompt(has_runner=True)
        assert "НЕ собран" not in built

    def test_web_toggle_marks_l2(self):
        on = architect_system_prompt(web=True)
        off = architect_system_prompt(web=False)
        assert "ВКЛЮЧЁН" in on and "L2" in on
        assert "выключен" in off

    def test_spec_hash_and_attachments_in_context(self):
        text = architect_system_prompt(project=PROJECT, spec_hash="deadbeef",
                                       n_attachments=3)
        assert "deadbeef" in text
        assert "вложений в сессии: 3" in text

    def test_extra_block_appended(self):
        text = architect_system_prompt(extra="ФОКУС UI: секция seed-цикла")
        assert text.strip().endswith("ФОКУС UI: секция seed-цикла")

    def test_routing_table_lists_all_scenarios(self):
        text = architect_system_prompt()
        for sc in GOLDEN_SCENARIOS:
            assert sc.title in text
            for tool in sc.tools:
                assert f"`{tool}`" in text


# ======================================================================
# 2. Golden-сценарии как таблица-контракт
# ======================================================================
class TestScenarios:
    def test_exactly_eight(self):
        assert len(GOLDEN_SCENARIOS) == 8

    def test_ids_and_keys_unique(self):
        assert [s.id for s in GOLDEN_SCENARIOS] == list(range(1, 9))
        assert len({s.key for s in GOLDEN_SCENARIOS}) == 8

    def test_every_tool_exists_in_registry(self):
        agent = set(tool_names(AGENT_KINDS))
        for sc in GOLDEN_SCENARIOS:
            for tool in sc.tools:
                assert tool in agent, f"{sc.key}: инструмента {tool} нет"

    def test_human_only_matches_write_class(self):
        # Список запрещённого не выдуман: это ровно класс write реестра.
        assert sorted(HUMAN_ONLY) == sorted(tool_names([WRITE]))

    def test_every_scenario_forbids_human_only(self):
        for sc in GOLDEN_SCENARIOS:
            assert set(HUMAN_ONLY) <= set(sc.forbidden)

    def test_each_scenario_has_reason(self):
        for sc in GOLDEN_SCENARIOS:
            assert len(sc.rule) > 20, f"{sc.key}: правило без причины"

    def test_lookup_by_id_and_key(self):
        assert scenario(3).key == "l1_widen"
        assert scenario("l1_widen").id == 3
        assert scenario(scenario(3)) is scenario(3)

    def test_unknown_scenario_explains_itself(self):
        with pytest.raises(KeyError) as exc:
            scenario("нет-такого")
        assert "l1_widen" in str(exc.value)
        with pytest.raises(KeyError):
            scenario(99)


# ======================================================================
# 3. Маршрутизация (DoD iter64)
# ======================================================================
@pytest.mark.parametrize("sc", GOLDEN_SCENARIOS, ids=[s.key for s in GOLDEN_SCENARIOS])
def test_golden_scenario_routes_correctly(sc):
    """Каждый из 8 сценариев §8 попадает в свой маршрут и свои инструменты."""
    r = route_scenario(sc)
    assert r.scenario == sc.key
    assert r.kind == sc.kind
    assert tuple(r.tools) == tuple(sc.tools)


class TestRouter:
    def test_unknown_question_asks_to_clarify(self):
        # Угаданный маршрут дороже честного «уточните».
        assert route("расскажи анекдот про полимеры").kind == KIND_CLARIFY
        assert route("").kind == KIND_CLARIFY

    def test_action_request_beats_bound_change(self):
        # «Расширь и примени сам» — это просьба ДЕЙСТВОВАТЬ: инструментов у
        # модели для неё нет, нужна кнопка человека.
        r = route("Расширь DINP до 20 phr и примени сам.")
        assert r.kind == KIND_HANDOFF
        assert r.tools == ()

    def test_question_beats_order(self):
        # «Что изменится, если сузить…» — вопрос, а не поручение: dry-run,
        # стейдж не трогаем.
        r = route("Что изменится, если сузить DINP до 8 phr?")
        assert r.kind == KIND_WHATIF
        assert "propose_patch" not in r.tools
        assert route("Сузь DINP до 8 phr").kind == KIND_PROPOSE

    def test_case_and_spacing_insensitive(self):
        sc = scenario("wedge_refusal")
        loud = sc.user.upper().replace(" ", "   ")
        assert route(loud).scenario == sc.key

    def test_forbidden_always_carried(self):
        for text in ("почему такой диапазон", "прогони тесты", "ерунда"):
            assert set(route(text).forbidden) == set(HUMAN_ONLY)

    def test_caption_names_tools_and_reason(self):
        cap = route_caption(route_scenario("explain_bounds"))
        assert "explain_node" in cap and "маршрут" in cap
        # У «примени сам» инструментов нет — подпись объясняет ПОЧЕМУ,
        # а не оставляет пустое место.
        handoff = route_caption(route("Ну примени уже."))
        assert "кнопка человека" in handoff and "→" not in handoff



# ======================================================================
# 4. Сверка фактического хода с маршрутом
# ======================================================================
class TestCheckRouting:
    def test_ok_when_required_tools_called(self):
        rep = check_routing("explain_bounds", ["explain_node", "get_spec"])
        assert rep["ok"] and rep["missing"] == []
        assert rep["extra"] == ["get_spec"]

    def test_answer_from_memory_is_a_failure(self):
        rep = check_routing("explain_bounds", [])
        assert not rep["ok"]
        assert "explain_node" in rep["missing"]
        assert "по памяти" in " ".join(rep["problems"])

    def test_self_apply_is_a_failure(self):
        rep = check_routing("l1_widen",
                            ["explain_node", "validate_spec", "propose_patch",
                             "apply_patch"])
        assert not rep["ok"]
        assert rep["forbidden_used"] == ["apply_patch"]
        assert "акт человека" in " ".join(rep["problems"])

    def test_whatif_must_not_stage_patch(self):
        rep = check_routing("whatif",
                            ["validate_spec", "simulate_bounds", "propose_patch"])
        assert not rep["ok"]
        assert "propose_patch" in rep["forbidden_used"]

    def test_accepts_dicts_and_objects(self):
        calls = [{"tool": "explain_node", "ok": True}]
        assert check_routing("explain_bounds", calls)["ok"]

        class _Call:
            tool = "explain_node"

        assert check_routing("explain_bounds", [_Call()])["ok"]


# ======================================================================
# 5. Сборка сообщений хода
# ======================================================================
class TestWithSystem:
    def test_prompt_goes_first(self):
        msgs = with_system("ПРОМПТ", [{"role": "user", "content": "привет"}])
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"].startswith(prompts.PROMPT_MARK)
        assert msgs[1]["content"] == "привет"

    def test_not_duplicated_on_second_turn(self):
        first = with_system("ПРОМПТ", [{"role": "user", "content": "раз"}])
        second = with_system("ПРОМПТ-2", first + [{"role": "user",
                                                   "content": "два"}])
        heads = [m for m in second
                 if str(m["content"]).startswith(prompts.PROMPT_MARK)]
        assert len(heads) == 1
        assert "ПРОМПТ-2" in heads[0]["content"]

    def test_session_notes_survive(self):
        # Пометка усечения истории (iter58) несёт ФАКТ, а не инструкцию —
        # выкидывать её нельзя.
        note = {"role": "system", "content": "[сессия] Ранние сообщения опущены"}
        msgs = with_system("ПРОМПТ", [note, {"role": "user", "content": "?"}])
        assert any(str(m["content"]).startswith("[сессия]") for m in msgs)


# ======================================================================
# 6. Ход целиком: маршрут исполняется инструментами (без сети)
# ======================================================================
class TestTurnRouting:
    def test_explain_scenario_answers_with_core_numbers(self, tmp_path):
        ctx = _ctx(tmp_path)
        calls = []
        call = dispatcher(ctx, allowed_kinds=AGENT_KINDS,
                          on_call=lambda rec: calls.append(rec))
        sc = scenario("explain_bounds")
        script = [_answer("", [_fn("explain_node", {"name": "PBNK",
                                                    "totals": [15.0]})]),
                  _answer("## ОТВЕТ\nПотолок доли зажат техлимитом 8 phr.")]
        res = llm.run_tool_loop(
            with_system(architect_system_prompt(project=PROJECT),
                        [{"role": "user", "content": sc.user}]),
            dispatch=call, tools=tool_specs(AGENT_KINDS),
            transport=_scripted(script))
        assert check_routing(sc, res.calls)["ok"]
        assert calls and calls[0]["ok"]

    def test_model_cannot_apply_patch_itself(self, tmp_path):
        ctx = _ctx(tmp_path)
        call = dispatcher(ctx, allowed_kinds=AGENT_KINDS)
        sc = scenario("handoff")
        script = [_answer("", [_fn("apply_patch", {"patch_id": "p1",
                                                   "human_token": "я-сам"})]),
                  _answer("## ОТВЕТ\nПрименяет человек кнопкой «Применить».")]
        res = llm.run_tool_loop(
            with_system(architect_system_prompt(project=PROJECT),
                        [{"role": "user", "content": sc.user}]),
            dispatch=call, tools=tool_specs(AGENT_KINDS),
            transport=_scripted(script))
        # Ход НЕ падает: отказ уходит модели как результат инструмента (A0.6).
        assert res.stopped_reason == "final"
        assert res.calls[0]["ok"] is False
        tool_msg = [m for m in res.new_messages if m.get("role") == "tool"][0]
        assert "ОШИБКА ИНСТРУМЕНТА" in tool_msg["content"]
        rep = check_routing(sc, res.calls)
        assert not rep["ok"] and rep["forbidden_used"] == ["apply_patch"]

    def test_write_tools_not_offered_to_model(self):
        names = {s["function"]["name"] for s in tool_specs(AGENT_KINDS)}
        assert not (names & set(HUMAN_ONLY))
        assert {"explain_node", "propose_patch", "run_pytest"} <= names


# ======================================================================
# 7. Разбор ответа архитектора
# ======================================================================
class TestAnswerSections:
    ANSWER = ("## ОТВЕТ\nВерх DINP — договорённость.\n\n"
              "## ЧИСЛА\nexplain_node: DINP 4–14 phr.\n\n"
              "## PATCH\nnode: DINP, from: [4,14], to: [4,20]\n\n"
              "## OPEN_QUESTIONS\n")

    def test_sections_parsed(self):
        out = parse_sections(self.ANSWER)
        assert set(out) == {"ОТВЕТ", "ЧИСЛА", "PATCH"}
        assert "договорённость" in out["ОТВЕТ"]
        assert out["PATCH"].startswith("node: DINP")

    def test_empty_section_dropped(self):
        # Пустой OPEN_QUESTIONS не должен превращаться в «вопросов нет»
        # в панели: его просто нет.
        assert "OPEN_QUESTIONS" not in parse_sections(self.ANSWER)

    def test_missing_required_section_reported(self):
        assert missing_sections(self.ANSWER) == []
        assert missing_sections("просто текст") == ["ОТВЕТ"]
        assert missing_sections(self.ANSWER,
                                required=("ОТВЕТ", "OPEN_QUESTIONS")) == \
            ["OPEN_QUESTIONS"]


# ======================================================================
# 8. Показ (те же чистые хелперы, что пойдут в док iter65)
# ======================================================================
class TestViews:
    def test_scenarios_table(self):
        df = scenarios_dataframe()
        assert len(df) == 8
        assert list(df.columns) == ["№", "сценарий", "маршрут", "инструменты",
                                    "нельзя", "правило"]
        handoff = df[df["№"] == 7].iloc[0]
        assert "кнопка человека" in handoff["инструменты"]

    def test_routing_table_marks_violation(self):
        good = check_routing("explain_bounds", ["explain_node"])
        bad = check_routing("explain_bounds", ["apply_patch"])
        df = routing_dataframe([good, bad])
        assert df.iloc[0]["итог"].startswith("✅")
        assert df.iloc[1]["итог"].startswith("⛔")
        assert df.iloc[1]["запрещённое"] == "apply_patch"

    def test_empty_routing_table_has_columns(self):
        df = routing_dataframe([])
        assert df.empty and "сценарий" in df.columns
