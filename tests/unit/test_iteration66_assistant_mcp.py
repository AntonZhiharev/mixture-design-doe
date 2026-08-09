# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 66 / ASSISTANT_SPEC — MCP-сервер `doe-campaign`.

Шаг закрывает разрыв каналов: до iter66 числа из ядра кампании были доступны
ТОЛЬКО внутри Streamlit-дока, а Cline в редакторе рассуждал о спеке по
исходникам — то есть по памяти. Сервер отдаёт внешнему агенту РОВНО те же
read-only инструменты (`src/assistant/tools`), поэтому проверяем не «вернулся
словарь», а границы контракта:

  * экспортируется ровно класс ``readonly``; ``write`` / ``propose`` /
    ``sandbox`` не выдаются — и запрет держится ДИСПЕТЧЕРОМ, а не списком имён
    (тест снимает фильтр сервера и всё равно получает отказ);
  * золотые числа доходят до внешнего агента без искажения: `explain_node`
    через MCP воспроизводит немонотонную ``hi_φ(T)`` (0.5333 @T=15, iter45/B1);
  * «не собран» ≠ «всё хорошо»: проект без ``campaign.json`` отвечает отказом
    с причиной, а не молчаливым «ok»;
  * обёртки MCP ГЕНЕРИРУЮТСЯ из JSON-схем реестра — новый read-only
    инструмент появляется в сервере сам, write-инструмент не появляется;
  * каждый вызов пишется в аудит кампании с пометкой ``via="mcp"``, а сессия
    ассистента при этом НЕ переписывается (сервер только читает).

Сеть и пакет ``mcp`` не нужны: сервер — тонкая обёртка над чистой логикой
:mod:`src.mcp.campaign_tools`.
"""
import inspect
import json
import os
import time
import warnings

import pytest
from sklearn.exceptions import ConvergenceWarning

from src.assistant import store
from src.assistant.session import new_session
from src.assistant.tools import (PROPOSE, READONLY, SANDBOX, TOOLS, WRITE,
                                 ToolError, tool_names)
from src.assistant.tools.registry import register
from src.design.phr_sampler import PhrSpec
from src.mcp import campaign_tools as ct

warnings.filterwarnings("ignore", category=ConvergenceWarning)

PROJECT = "pvc_edge_v1"
NOTES = "notes_only"

# Референсная v2-спека (та же геометрия, что golden iter45/61): группа SOFT с
# техлимитами даёт немонотонную верхнюю границу доли.
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


# ======================================================================
# Фикстуры: каталог кампаний с двумя проектами
# ======================================================================
@pytest.fixture(scope="module")
def campaigns(tmp_path_factory):
    """Каталог с сохранённой кампанией и проектом «одна переписка»."""
    from src.apps.campaign_state import save_campaign
    from src.apps.campaign_ui import build_setup_runner

    root = str(tmp_path_factory.mktemp("campaigns"))
    spec = _spec()
    lo, hi = spec.fraction_bounds()
    runner = build_setup_runner(
        mixture_names=list(spec.component_names), process_names=["T"],
        process_lower=[150.0], process_upper=[200.0],
        response_names=["gloss"],
        mixture_lower=lo.tolist(), mixture_upper=hi.tolist(), seed=1)
    runner.set_phr_spec(spec)
    runner.set_campaign_label("PVC-кромка-2026")
    save_campaign(runner, root, PROJECT)

    s = new_session(PROJECT)
    s.add_message("user", "почему верх PBNK не 0.70 при T=15?")
    store.save_session(s, root, PROJECT)
    store.append_log(root, PROJECT, "local_facts",
                     {"scope": "PBNK", "statement": "склад держит не более 8 phr",
                      "author": "технолог"})

    # Проект БЕЗ движка: только переписка ассистента.
    store.save_session(new_session(NOTES), root, NOTES)
    return root


@pytest.fixture(autouse=True)
def _env(campaigns, monkeypatch):
    """Сервер смотрит в тестовый каталог; кэш контекста сбрасывается."""
    monkeypatch.setenv(ct.ROOT_ENV, campaigns)
    ct.clear_cache()
    yield
    ct.clear_cache()


# ======================================================================
# 1. Контракт экспорта: наружу выходит РОВНО класс readonly
# ======================================================================
class TestExportContract:

    def test_exported_is_exactly_readonly_registry(self):
        assert ct.exported_names() == sorted(tool_names([READONLY]))
        assert ct.EXPORTED_KINDS == (READONLY,)

    def test_core_readonly_tools_are_exported(self):
        names = set(ct.exported_names())
        assert {"get_spec", "explain_node", "validate_spec", "simulate_bounds",
                "preflight", "point_report", "encode_recipe", "get_runs",
                "campaign_overview", "get_local_facts",
                "get_decisions"} <= names

    def test_write_propose_sandbox_never_exported(self):
        hidden = set(tool_names([WRITE, PROPOSE, SANDBOX]))
        assert hidden, "классы write/propose/sandbox пусты — тест бессмыслен"
        assert hidden.isdisjoint(set(ct.exported_names()))
        assert set(ct.hidden_names()) == hidden
        for name in hidden:
            assert not ct.is_exported(name)

    def test_write_tool_refused_with_reason(self):
        """Отказ объясняет, ГДЕ кнопка, а не просто «нет такого инструмента»."""
        out = ct.call_tool(PROJECT, "apply_patch", {"patch_id": "p_1"})
        assert out["ok"] is False
        assert "write" in out["error"]
        assert "ЧЕЛОВЕК" in out["error"]
        assert "get_spec" in " ".join(out["available"])

    def test_propose_tool_refused_with_reason(self):
        out = ct.call_tool(PROJECT, "propose_patch", {"patch": {}})
        assert out["ok"] is False
        assert "стейдж" in out["error"].lower() or "СТЕЙДЖ" in out["error"]

    def test_sandbox_tool_refused(self):
        out = ct.call_tool(PROJECT, "run_pytest", {})
        assert out["ok"] is False
        assert "не экспортируется" in out["error"]

    def test_unknown_tool_lists_available(self):
        out = ct.call_tool(PROJECT, "make_coffee")
        assert out["ok"] is False
        assert "не зарегистрирован" in out["error"]

    def test_ban_is_held_by_dispatcher_not_by_name_list(self, monkeypatch):
        """Снимаем фильтр сервера — отбивает реестр (запрет живёт в коде)."""
        monkeypatch.setattr(ct, "is_exported", lambda name: True)
        out = ct.call_tool(PROJECT, "apply_patch",
                           {"patch_id": "p_1", "human_token": "я-сам"})
        assert out["ok"] is False
        assert "класс" in out["error"]


# ======================================================================
# 2. Проекты каталога: угадывать нельзя
# ======================================================================
class TestProjects:

    def test_list_projects_reports_what_exists(self):
        rows = {d["project"]: d for d in ct.list_projects()}
        assert set(rows) == {PROJECT, NOTES}
        assert rows[PROJECT]["has_campaign"] and rows[PROJECT]["has_session"]
        assert rows[NOTES]["has_campaign"] is False
        assert rows[NOTES]["has_session"] is True

    def test_missing_root_is_empty_not_crash(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ct.ROOT_ENV, str(tmp_path / "нет-такого"))
        assert ct.list_projects() == []

    def test_unknown_project_is_explained(self):
        with pytest.raises(ToolError, match="нет в каталоге"):
            ct.resolve_project("чужой_проект")

    def test_empty_project_not_guessed_when_many(self):
        with pytest.raises(ToolError, match="Назовите проект явно"):
            ct.resolve_project("")

    def test_empty_project_resolved_when_single(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ct.ROOT_ENV, str(tmp_path))
        store.save_session(new_session("solo"), str(tmp_path), "solo")
        assert ct.resolve_project("") == "solo"

    def test_empty_root_says_where_to_save(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ct.ROOT_ENV, str(tmp_path / "пусто"))
        with pytest.raises(ToolError, match="нет ни одного проекта"):
            ct.resolve_project("")


# ======================================================================
# 3. Статус проекта: состояние берётся из файла, а не из догадок
# ======================================================================
class TestProjectStatus:

    def test_status_of_saved_campaign(self):
        st = ct.project_status(PROJECT)
        assert st["has_campaign"] is True
        assert st["campaign_label"] == "PVC-кромка-2026"
        assert st["has_phr_spec"] is True
        assert st["spec_hash"] == _spec().spec_hash()
        assert st["n_points"] == 0
        assert st["property_names"] == ["gloss"]
        assert st["session"]["messages"] >= 1

    def test_status_without_engine_says_not_checked(self):
        st = ct.project_status(NOTES)
        assert st["has_campaign"] is False
        assert "не проверено" in st["note"]
        assert st["session"]["messages"] == 0

    def test_status_does_not_build_runner(self, monkeypatch):
        """Статус не должен стоить переобучения суррогатов."""
        import src.apps.campaign_state as cs

        def _boom(*a, **kw):        # pragma: no cover — не должен вызываться
            raise AssertionError("project_status собрал движок")

        monkeypatch.setattr(cs, "load_campaign", _boom)
        assert ct.project_status(PROJECT)["spec_hash"] == _spec().spec_hash()


# ======================================================================
# 4. Числа доходят до внешнего агента без искажения (golden iter45/B1)
# ======================================================================
class TestGoldenNumbers:

    def test_get_spec_returns_project_hash(self):
        out = ct.call_tool(PROJECT, "get_spec", {"include_nodes": False})
        assert out["ok"] is True
        assert out["result"]["spec_hash"] == _spec().spec_hash()
        assert out["result"]["campaign"]["label"] == "PVC-кромка-2026"

    def test_explain_node_reproduces_nonmonotonic_hi(self):
        out = ct.call_tool(PROJECT, "explain_node",
                           {"name": "PBNK", "totals": [5.0, 10.5, 15.0]})
        assert out["ok"] is True
        rows = {round(r["total"], 2): r for r in
                out["result"]["effective_shares"]}
        assert rows[5.0]["share_hi"] == pytest.approx(0.40, abs=1e-6)
        assert rows[10.5]["share_hi"] == pytest.approx(0.70, abs=1e-6)
        assert rows[15.0]["share_hi"] == pytest.approx(0.5333, abs=1e-4)

    def test_local_facts_reach_external_agent(self):
        out = ct.call_tool(PROJECT, "explain_node", {"name": "PBNK"})
        facts = out["result"].get("local_facts", [])
        assert any("8 phr" in f.get("statement", "") for f in facts)

    def test_point_report_on_sampled_recipe(self):
        spec = _spec()
        recipe = [float(v) for v in spec.decode(spec.sample_z(1, seed=0))[0]]
        out = ct.call_tool(PROJECT, "point_report", {"recipe_phr": recipe})
        assert out["ok"] is True
        assert out["result"]["ok"] is True
        assert set(out["result"]["effective_bounds"]) >= {"PBNK", "CPE"}

    def test_validate_spec_is_dry_run_only(self):
        """Dry-run через MCP не двигает геометрию проекта."""
        before = ct.call_tool(PROJECT, "get_spec",
                              {"include_nodes": False})["result"]["spec_hash"]
        out = ct.call_tool(PROJECT, "validate_spec",
                           {"patch": {"node": "DINP", "field": "range",
                                      "value": [4.0, 20.0]}})
        assert out["ok"] is True and out["result"]["ok"] is True
        assert out["result"]["affects_hash"] is True
        after = ct.call_tool(PROJECT, "get_spec",
                             {"include_nodes": False})["result"]["spec_hash"]
        assert before == after == _spec().spec_hash()

    def test_bad_arguments_are_explained(self):
        out = ct.call_tool(PROJECT, "explain_node", {"name": "НЕТ_ТАКОГО"})
        assert out["ok"] is False
        assert "нет в спеке" in out["error"]


# ======================================================================
# 5. «Не собран» ≠ «всё хорошо»
# ======================================================================
class TestHonestAboutMissingEngine:

    def test_preflight_without_campaign_refuses_with_reason(self):
        out = ct.call_tool(NOTES, "preflight", {})
        assert out["ok"] is False
        assert "не собран" in out["error"]

    def test_context_note_explains_absence(self):
        pc = ct.load_context(NOTES)
        assert pc.has_runner is False
        assert "campaign.json" in pc.note
        assert pc.spec_hash == ""

    def test_spec_questions_without_spec_refuse(self):
        out = ct.call_tool(NOTES, "get_spec", {})
        assert out["ok"] is False
        assert "phr-спека" in out["error"]
        assert "campaign.json" in out.get("note", "")


# ======================================================================
# 6. Аудит и чтение без записи
# ======================================================================
class TestAuditAndReadOnly:

    def test_call_is_written_to_campaign_audit(self, campaigns):
        ct.call_tool(PROJECT, "get_spec", {"include_nodes": False})
        recs = store.read_log(campaigns, PROJECT, "tool_calls")
        assert recs, "вызов через MCP не попал в аудит кампании"
        last = recs[-1]
        assert last["tool"] == "get_spec" and last["ok"] is True
        assert last["via"] == "mcp"

    def test_failed_call_is_audited_too(self, campaigns):
        ct.call_tool(PROJECT, "explain_node", {"name": "НЕТ_ТАКОГО"})
        last = store.read_log(campaigns, PROJECT, "tool_calls")[-1]
        assert last["ok"] is False and last["via"] == "mcp"
        assert "нет в спеке" in last["error"]

    def test_session_file_is_not_rewritten(self, campaigns):
        path = store.session_path(campaigns, PROJECT)
        before = path.stat().st_mtime_ns
        time.sleep(0.01)
        ct.call_tool(PROJECT, "explain_node", {"name": "PBNK"})
        assert path.stat().st_mtime_ns == before

    def test_no_patches_appear_from_mcp(self):
        ct.call_tool(PROJECT, "get_spec", {})
        assert ct.load_context(PROJECT).ctx.session.staged_patches() == []


# ======================================================================
# 7. Кэш контекста: правка в интерфейсе видна серверу
# ======================================================================
class TestContextCache:

    def test_context_is_cached_between_calls(self):
        a = ct.load_context(PROJECT)
        b = ct.load_context(PROJECT)
        assert a is b

    def test_changed_campaign_invalidates_cache(self, campaigns):
        a = ct.load_context(PROJECT)
        path = os.path.join(campaigns, PROJECT, ct.STATE_FILE)
        stat = os.stat(path)
        os.utime(path, (stat.st_atime + 5, stat.st_mtime + 5))
        assert ct.load_context(PROJECT) is not a

    def test_use_cache_false_rebuilds(self):
        a = ct.load_context(PROJECT)
        assert ct.load_context(PROJECT, use_cache=False) is not a


# ======================================================================
# 8. Обёртки MCP генерируются из реестра
# ======================================================================
class TestGeneratedWrappers:

    def test_signature_matches_registry_schema(self):
        fn = ct.build_wrappers(lambda project, tool, args: (project, tool, args))[
            "explain_node"]
        params = list(inspect.signature(fn).parameters)
        assert params == ["name", "totals", "project"]
        sig = inspect.signature(fn)
        assert sig.parameters["name"].default is inspect.Parameter.empty
        assert sig.parameters["totals"].default is None
        assert sig.parameters["project"].default == ""

    def test_wrapper_passes_only_given_arguments(self):
        seen = {}

        def _call(project, tool, args):
            seen.update({"project": project, "tool": tool, "args": args})
            return {"ok": True}

        w = ct.build_wrappers(_call)
        w["explain_node"]("PBNK", project=PROJECT)
        assert seen == {"project": PROJECT, "tool": "explain_node",
                        "args": {"name": "PBNK"}}
        w["explain_node"]("PBNK", [15.0], PROJECT)
        assert seen["args"] == {"name": "PBNK", "totals": [15.0]}

    def test_wrapper_docstring_comes_from_registry(self):
        w = ct.build_wrappers(lambda project, tool, args: None)
        assert TOOLS["explain_node"].description[:40] in w["explain_node"].__doc__
        assert "project" in w["explain_node"].__doc__

    def test_all_exported_tools_have_wrappers(self):
        w = ct.build_wrappers(lambda project, tool, args: None)
        assert sorted(w) == ct.exported_names()
        assert ct.conflicting_tools() == []

    def test_new_readonly_tool_appears_without_editing_server(self):
        @register("dummy_probe_tool", description="тест iter66",
                  parameters={"type": "object", "properties": {
                      "n": {"type": "integer"}}}, kind=READONLY)
        def _probe(ctx, n=1):       # pragma: no cover — только схема
            return n

        try:
            assert "dummy_probe_tool" in ct.exported_names()
            w = ct.build_wrappers(lambda project, tool, args: (tool, args))
            assert list(inspect.signature(w["dummy_probe_tool"]).parameters) \
                == ["n", "project"]
        finally:
            TOOLS.pop("dummy_probe_tool", None)

    def test_new_write_tool_never_appears(self):
        @register("dummy_write_tool66", description="тест iter66", kind=WRITE)
        def _w(ctx):                # pragma: no cover — только реестр
            return "applied"

        try:
            assert "dummy_write_tool66" not in ct.exported_names()
            assert "dummy_write_tool66" not in ct.build_wrappers(
                lambda project, tool, args: None)
            assert "dummy_write_tool66" in ct.hidden_names()
        finally:
            TOOLS.pop("dummy_write_tool66", None)

    def test_project_argument_conflict_is_refused_not_silent(self):
        @register("dummy_project_tool", description="тест iter66",
                  parameters={"type": "object", "properties": {
                      "project": {"type": "string"}}}, kind=READONLY)
        def _p(ctx, project=""):    # pragma: no cover — только схема
            return project

        try:
            assert "dummy_project_tool" in ct.conflicting_tools()
            assert "dummy_project_tool" not in ct.build_wrappers(
                lambda project, tool, args: None)
            with pytest.raises(ValueError, match="занят сервером"):
                ct.wrapper_signature(TOOLS["dummy_project_tool"])
        finally:
            TOOLS.pop("dummy_project_tool", None)

    def test_catalog_lists_arguments_and_long_running(self):
        cat = {d["tool"]: d for d in ct.tool_catalog()}
        assert cat["explain_node"]["required"] == ["name"]
        assert cat["preflight"]["long_running"] is True
        assert set(cat) == set(ct.exported_names())


# ======================================================================
# 9. Сервер: сборка и самопроверка без пакета mcp
# ======================================================================
class TestServerModule:

    def test_selftest_runs_without_mcp_package(self, capsys):
        from src.mcp import campaign_server as srv

        assert srv._selftest(PROJECT) == 0
        out = capsys.readouterr().out
        assert "doe-campaign" in out
        assert PROJECT in out
        assert "apply_patch" in out          # отказ показан словами

    def test_selftest_on_empty_root(self, tmp_path, monkeypatch, capsys):
        from src.mcp import campaign_server as srv

        monkeypatch.setenv(ct.ROOT_ENV, str(tmp_path / "пусто"))
        assert srv._selftest() == 0
        assert "нет сохранённых кампаний" in capsys.readouterr().out

    def test_status_resource_is_json(self):
        payload = json.dumps(ct.project_status(PROJECT), ensure_ascii=False)
        assert json.loads(payload)["project"] == PROJECT
