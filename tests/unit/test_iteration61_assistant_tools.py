# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 61 / ASSISTANT_SPEC — READ-ONLY инструменты ассистента.

Инструменты закрывают худший класс отказа помощника — работу ПО ПАМЯТИ.
Ответ про геометрию обязан приходить из ядра: `PhrSpec` (роли, эффективные
границы, `spec_hash`), `point_report`, розыгрыш кандидатов, `preflight`, общая
база точек. Поэтому тесты проверяют не «функция вернула словарь», а GOLDEN
ЧИСЛА, ради которых слой существует:

  * `explain_node` на SOFT-группе воспроизводит НЕМОНОТОННУЮ
    ``hi_φ(T) = min(0.70, 8/T, 1 − 3/T)``: 0.40 @T=5 · полка 0.70 @T=10.5 ·
    0.5333 @T=15 (golden iter45/B1) — по двум точкам вывод сделать нельзя;
  * `simulate_bounds` численно различает ТРАПЕЦИЮ по фазе (corr ≈ 0.14) и
    КЛИН `RATIO_TO` (corr ≈ 0.89) — тот самый аргумент, которым ассистент
    отказывает на «привяжи УФ к пигменту»;
  * `validate_spec` — dry-run: инвариант «closure без range» ловится, спека
    проекта НЕ меняется, изменение `spec_hash` объявляется.

Плюс контракт реестра: классы доступа (write недоступен в readonly-режиме),
проверка аргументов, отказы с объяснением (A0.6).
"""
import numpy as np
import pytest

from src.assistant import store
from src.assistant.files import attach_file
from src.assistant.session import new_session
from src.assistant.tools import (TOOLS, ToolContext, ToolError, dispatch,
                                  tool_names, tool_specs)
from src.assistant.tools.readonly import (apply_patch_to_dicts,
                                           build_patched_spec, normalize_patch)
from src.assistant.tools.registry import READONLY, WRITE, dispatcher
from src.design.phr_sampler import PhrSpec

PROJECT = "pvc_edge_v1"

# Референсная v2-спека: роли, лог-оси, cap-трапеция, техлимиты SOFT
# (та же геометрия, что golden iter45/49/50).
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

#: Патч «привяжи УФ к пластификатору жёстко» — КЛИН вместо трапеции.
WEDGE_PATCH = {"node": "UV",
               "set": {"role": "RATIO_TO", "reference": "DINP",
                       "range": [0.0125, 0.0214]},
               "unset": ["cap_to", "cap_ratio", "scale"]}


def _spec() -> PhrSpec:
    return PhrSpec.from_dicts(NODES)


def _ctx(**kw) -> ToolContext:
    return ToolContext(spec=_spec(), **kw)


# ======================================================================
# 1. Реестр: схемы, классы доступа, валидация аргументов
# ======================================================================
class TestRegistry:

    def test_specs_are_valid_function_schemas(self):
        specs = tool_specs([READONLY])
        assert specs, "read-only инструменты не зарегистрированы"
        for s in specs:
            assert s["type"] == "function"
            fn = s["function"]
            assert fn["name"] and fn["description"]
            assert fn["parameters"]["type"] == "object"
            assert isinstance(fn["parameters"]["properties"], dict)
            for req in fn["parameters"].get("required", []):
                assert req in fn["parameters"]["properties"], \
                    f"{fn['name']}: required-аргумент вне properties"

    def test_core_tools_present(self):
        names = set(tool_names([READONLY]))
        assert {"get_spec", "explain_node", "validate_spec", "simulate_bounds",
                "preflight", "get_runs", "point_report", "encode_recipe",
                "get_local_facts", "get_decisions", "list_attachments",
                "read_attachment", "campaign_overview"} <= names

    def test_unknown_tool_lists_available(self):
        with pytest.raises(ToolError, match="не зарегистрирован"):
            dispatch(_ctx(), "make_coffee", {})

    def test_missing_required_argument_explained(self):
        with pytest.raises(ToolError, match="обязательные аргументы"):
            dispatch(_ctx(), "explain_node", {})

    def test_unknown_argument_rejected(self):
        with pytest.raises(ToolError, match="неизвестные аргументы"):
            dispatch(_ctx(), "get_spec", {"whatever": 1})

    def test_write_tools_not_reachable_in_readonly_mode(self):
        """Даже зная имя write-инструмента, модель не выполнит его сама."""
        from src.assistant.tools.registry import register

        @register("dummy_write_tool", description="тест", kind=WRITE)
        def _w(ctx):  # pragma: no cover — вызывается только через dispatch
            return "applied"

        try:
            with pytest.raises(ToolError, match="класс"):
                dispatch(_ctx(), "dummy_write_tool", {},
                         allowed_kinds=(READONLY,))
            assert dispatch(_ctx(), "dummy_write_tool", {},
                            allowed_kinds=(READONLY, WRITE)) == "applied"
        finally:
            TOOLS.pop("dummy_write_tool", None)

    def test_missing_spec_is_explained_not_crash(self):
        with pytest.raises(ToolError, match="phr-спека"):
            dispatch(ToolContext(), "get_spec", {})

    def test_missing_runner_is_explained(self):
        with pytest.raises(ToolError, match="не собран"):
            dispatch(_ctx(), "get_runs", {})

    def test_dispatcher_audits_success_and_failure(self):
        seen = []
        call = dispatcher(_ctx(), on_call=seen.append)
        call("get_spec", {"include_nodes": False})
        with pytest.raises(ToolError):
            call("explain_node", {"name": "НЕТ_ТАКОГО"})
        assert [r["tool"] for r in seen] == ["get_spec", "explain_node"]
        assert seen[0]["ok"] is True and seen[1]["ok"] is False
        assert "нет в спеке" in seen[1]["error"]


# ======================================================================
# 2. get_spec — снимок геометрии
# ======================================================================
class TestGetSpec:

    def test_reports_hash_dims_and_log_axes(self):
        out = dispatch(_ctx(), "get_spec", {})
        assert out["spec_hash"] == _spec().spec_hash()
        assert out["q_components"] == 7 and out["dim_z"] == 5
        assert out["log_axes"] == ["TiO2", "UV"]
        assert out["component_names"][0] == "RESIN"
        assert out["phr_intervals"]["PBNK"] == [0.0, 8.0]
        assert out["phr_intervals"]["CPE"] == [3.0, 15.0]

    def test_nodes_carry_roles_from_core(self):
        spec = _spec()
        out = dispatch(ToolContext(spec=spec), "get_spec", {})
        for d in out["nodes"]:
            assert d["role"] == spec.role_of(d["name"])

    def test_json_serializable(self):
        import json
        json.dumps(dispatch(_ctx(), "get_spec", {}), ensure_ascii=False)


# ======================================================================
# 3. explain_node — golden немонотонности (iter45/B1)
# ======================================================================
class TestExplainNode:

    def test_share_node_reports_nonmonotone_effective_bounds(self):
        """hi_φ(T) = min(0.70, 8/T, 1−3/T): 0.40 @5 · полка 0.70 · 0.5333 @15."""
        out = dispatch(_ctx(), "explain_node",
                       {"name": "PBNK", "totals": [5.0, 10.5, 15.0]})
        hi = {r["total"]: round(r["share_hi"], 4) for r in out["effective_shares"]}
        assert hi[5.0] == 0.4
        assert hi[10.5] == 0.7
        assert hi[15.0] == 0.5333
        assert hi[10.5] > hi[15.0], "немонотонность hi(T) должна быть видна"
        assert "НЕМОНОТОННОЙ" in out["note"]

    def test_partner_limit_raises_own_floor(self):
        """Потолок партнёра поднимает чужой пол: lo_CPE(15) = 1 − 8/15."""
        out = dispatch(_ctx(), "explain_node",
                       {"name": "CPE", "totals": [15.0]})
        row = out["effective_shares"][0]
        assert row["share_lo"] == pytest.approx(1.0 - 8.0 / 15.0, abs=1e-6)

    def test_capped_node_explains_trapezoid_not_wedge(self):
        out = dispatch(_ctx(), "explain_node", {"name": "UV"})
        assert out["role"] == "ABSOLUTE_CAPPED"
        assert out["cap"]["cap_to"] == ["DINP", "ESO"]
        assert out["cap"]["cap_ratio"] == pytest.approx(0.03)
        assert "трапеция" in out["cap"]["note"]
        assert out["scale"] == "log"

    def test_group_members_and_totals_reported(self):
        out = dispatch(_ctx(), "explain_node", {"name": "PBNK"})
        assert out["group"] == "SOFT"
        assert out["group_members"] == ["PBNK", "CPE"]
        assert out["group_total_interval"] == [5.0, 15.0]

    def test_unknown_node_lists_known(self):
        with pytest.raises(ToolError, match="нет в спеке"):
            dispatch(_ctx(), "explain_node", {"name": "Chalk_1T"})

    def test_local_facts_attached_to_node(self, tmp_path):
        store.append_log(tmp_path, PROJECT, "local_facts",
                         {"scope": "PBNK", "statement": "склад держит ≤ 8 phr"})
        out = dispatch(ToolContext(spec=_spec(), root=str(tmp_path),
                                   project=PROJECT),
                       "explain_node", {"name": "PBNK"})
        assert out["local_facts"][0]["statement"].startswith("склад")


# ======================================================================
# 4. Патчи и validate_spec (dry-run)
# ======================================================================
class TestPatchAndValidate:

    def test_normalize_patch_accepts_all_forms(self):
        a = normalize_patch({"node": "DINP", "field": "range", "value": [4, 20]})
        b = normalize_patch({"DINP": {"range": [4, 20]}})
        assert a == b == [{"node": "DINP", "set": {"range": [4, 20]},
                           "unset": []}]
        assert len(normalize_patch([a[0], b[0]])) == 2

    def test_patch_without_value_or_fields_rejected(self):
        with pytest.raises(ToolError, match="нечего записывать"):
            normalize_patch({"node": "DINP", "field": "range"})
        with pytest.raises(ToolError, match="пуст"):
            normalize_patch({"node": "DINP"})

    def test_patch_does_not_touch_project_spec(self):
        spec = _spec()
        before = spec.spec_hash()
        build_patched_spec(spec, {"node": "DINP", "field": "range",
                                  "value": [4.0, 20.0]})
        assert spec.spec_hash() == before
        assert spec.phr_intervals()["DINP"] == (4.0, 14.0)

    def test_unknown_node_in_patch_explained(self):
        with pytest.raises(ToolError, match="Добавление/удаление узлов"):
            apply_patch_to_dicts(_spec(), {"node": "Chalk", "field": "range",
                                           "value": [0, 1]})

    def test_validate_reports_diff_and_hash_change(self):
        out = dispatch(_ctx(), "validate_spec",
                       {"patch": {"node": "DINP", "field": "range",
                                  "value": [4.0, 20.0]}})
        assert out["ok"] is True and out["affects_hash"] is True
        changed = {d["node"]: d for d in out["changed_intervals"]}
        assert changed["DINP"]["before"] == [4.0, 14.0]
        assert changed["DINP"]["after"] == [4.0, 20.0]
        assert "ДРУГАЯ геометрия" in out["warning"]

    def test_validate_rejects_closure_with_range(self):
        """Инвариант схемы: у SHARE_CLOSURE диапазон ПРОИЗВОДНЫЙ (iter46/B8)."""
        out = dispatch(_ctx(), "validate_spec",
                       {"patch": {"node": "CPE", "field": "share_range",
                                  "value": [0.1, 0.9]}})
        assert out["ok"] is False
        assert out["spec_hash_before"] == _spec().spec_hash()
        assert "не сбой" in out["hint"]

    def test_validate_rejects_closure_role_for_third_member(self):
        """k≥3 ⇒ closure запрещён: golden-сценарий «три смазки»."""
        nodes = NODES + [
            {"name": "LUB", "role": "GROUP_TOTAL", "range": [0.4, 1.2],
             "members": ["DL60", "AKLUB", "OPE"]},
            {"name": "DL60", "role": "SHARE_SIMPLEX", "group": "LUB",
             "share_range": [0.30, 0.70]},
            {"name": "AKLUB", "role": "SHARE_SIMPLEX", "group": "LUB",
             "share_range": [0.10, 0.60]},
            {"name": "OPE", "role": "SHARE_SIMPLEX", "group": "LUB",
             "share_range": [0.10, 0.60]},
        ]
        ctx = ToolContext(spec=PhrSpec.from_dicts(nodes))
        out = dispatch(ctx, "validate_spec",
                       {"patch": {"node": "OPE",
                                  "set": {"role": "SHARE_CLOSURE"},
                                  "unset": ["share_range"]}})
        assert out["ok"] is False

    def test_validate_no_change_keeps_hash(self):
        out = dispatch(_ctx(), "validate_spec",
                       {"patch": {"node": "DINP", "field": "range",
                                  "value": [4.0, 14.0]}})
        assert out["ok"] is True and out["affects_hash"] is False
        assert out["n_changed"] == 0 and out["warning"] == ""


# ======================================================================
# 5. simulate_bounds — КЛИН против ТРАПЕЦИИ (числа, а не рассуждение)
# ======================================================================
class TestSimulateBounds:

    def test_current_trapezoid_correlation_is_weak(self):
        out = dispatch(_ctx(), "simulate_bounds",
                       {"n": 400, "seed": 0, "pair": ["UV", "DINP"]})
        corr = out["current"]["pair_corr"]
        assert 0.05 <= corr <= 0.30, f"трапеция по фазе даёт слабую связь, {corr}"
        lo, hi = out["current"]["sigma_phr"]
        assert 100.0 < lo < hi < 200.0

    def test_wedge_patch_makes_node_follow_reference(self):
        """RATIO_TO вшивает монотонный prior: corr взлетает 0.14 → ~0.9."""
        out = dispatch(_ctx(), "simulate_bounds",
                       {"patch": WEDGE_PATCH, "n": 400, "seed": 0,
                        "pair": ["UV", "DINP"]})
        assert out["ok"] is True
        assert out["proposed"]["pair_corr"] > 0.80
        assert out["pair_corr_shift"] > 0.6
        assert out["current"]["pair_corr"] < 0.30

    def test_widening_range_moves_sigma_phr(self):
        out = dispatch(_ctx(), "simulate_bounds",
                       {"patch": {"node": "DINP", "field": "range",
                                  "value": [4.0, 40.0]},
                        "n": 200, "seed": 1})
        assert out["proposed"]["sigma_phr"][1] > out["current"]["sigma_phr"][1]
        assert out["sigma_phr_shift"][1] > 10.0

    def test_invalid_patch_reported_not_raised(self):
        out = dispatch(_ctx(), "simulate_bounds",
                       {"patch": {"node": "CPE", "field": "share_range",
                                  "value": [0.1, 0.9]}, "n": 50})
        assert out["ok"] is False and "валидацию" in out["error"]

    def test_unknown_pair_member_explained(self):
        with pytest.raises(ToolError, match="такого компонента нет"):
            dispatch(_ctx(), "simulate_bounds",
                     {"n": 50, "pair": ["UV", "Chalk"]})

    def test_deterministic_by_seed(self):
        a = dispatch(_ctx(), "simulate_bounds", {"n": 100, "seed": 7})
        b = dispatch(_ctx(), "simulate_bounds", {"n": 100, "seed": 7})
        assert a["current"]["sigma_phr"] == b["current"]["sigma_phr"]

    def test_marked_long_running_for_ui_progress(self):
        from src.assistant.tools.registry import is_long_running
        assert is_long_running("simulate_bounds") is True
        assert is_long_running("preflight") is True
        assert is_long_running("get_spec") is False


# ======================================================================
# 6. Рецепт: point_report и encode_recipe
# ======================================================================
class TestRecipeTools:

    def _recipe(self, t_soft=10.0, phi=0.5, dinp=6.0, tio2=1.0, uv=0.10):
        # порядок component_names: RESIN, DINP, ESO, PBNK, CPE, TiO2, UV
        return [100.0, dinp, 2.5, phi * t_soft, (1 - phi) * t_soft, tio2, uv]

    def test_point_report_marks_active_constraint(self):
        """При T=15 верх PBNK держит складской лимит 8 phr (= доля 8/15).

        Границы share-узла живут в ДОЛЯХ, а значение узла — и в доле
        (``coord``), и в phr: инструмент отдаёт обе величины, иначе сравнение
        «0.533 против 8» выглядело бы нарушением там, где его нет.
        """
        out = dispatch(_ctx(), "point_report",
                       {"recipe_phr": self._recipe(t_soft=15.0, phi=8.0 / 15.0)})
        assert out["ok"] is True
        pbnk = out["effective_bounds"]["PBNK"]
        assert pbnk["active_hi"] == "max_phr", "при T=15 давит складской лимит 8"
        assert pbnk["hi"] == pytest.approx(8.0 / 15.0, abs=1e-6)
        assert pbnk["coord"] == pytest.approx(8.0 / 15.0, abs=1e-6)
        assert pbnk["phr"] == pytest.approx(8.0, abs=1e-6)


    def test_point_report_flags_out_of_bounds_without_exception(self):
        """A0.6: точка вне геометрии — нарушение в отчёте, а не исключение."""
        out = dispatch(_ctx(), "point_report",
                       {"recipe_phr": self._recipe(t_soft=15.0, phi=0.9)})
        assert out["ok"] is False and out["violations"]

    def test_point_report_premix_needs_delta(self):
        without = dispatch(_ctx(), "point_report",
                           {"recipe_phr": self._recipe()})
        assert set(without["premix"].values()) == {None}
        withd = dispatch(_ctx(), "point_report",
                         {"recipe_phr": self._recipe(), "delta_phr": 0.02})
        assert withd["premix"]["UV"] is True, "узкий диапазон УФ ⇒ премикс"
        assert withd["phr_actual"] is not None

    def test_wrong_recipe_length_explained(self):
        with pytest.raises(ToolError, match="Порядок обязателен"):
            dispatch(_ctx(), "point_report", {"recipe_phr": [1.0, 2.0]})

    def test_encode_recipe_roundtrip(self):
        p = self._recipe()
        out = dispatch(_ctx(), "encode_recipe", {"recipe_phr": p})
        assert out["representable"] is True
        assert out["sigma_phr"] == pytest.approx(sum(p))
        assert len(out["z"]) == _spec().dim_z

    def test_encode_rejects_unrepresentable_anchor_with_reason(self):
        """Серийный рецепт вне спеки — честный отказ, а не подгонка."""
        out = dispatch(_ctx(), "encode_recipe",
                       {"recipe_phr": self._recipe(dinp=25.0)})
        assert out["representable"] is False
        assert out["error"] and "непредставимым" in out["hint"]


# ======================================================================
# 7. Знание проекта: факты, решения, документы
# ======================================================================
class TestKnowledgeTools:

    def test_local_facts_and_decisions(self, tmp_path):
        store.append_log(tmp_path, PROJECT, "local_facts",
                         {"scope": "cost", "statement": "плотность измеряется"})
        store.append_log(tmp_path, PROJECT, "decisions",
                         {"title": "мел до 100 phr", "nodes": ["FILLER.total"]})
        ctx = ToolContext(spec=_spec(), root=str(tmp_path), project=PROJECT)

        facts = dispatch(ctx, "get_local_facts", {"scope": "cost"})
        assert facts["n"] == 1 and "ТОЛЬКО человек" in facts["note"]
        assert dispatch(ctx, "get_local_facts", {"scope": "клей"})["n"] == 0

        dec = dispatch(ctx, "get_decisions", {"limit": 5})
        assert dec["decisions"][0]["title"] == "мел до 100 phr"

    def test_attachments_listing_and_reading(self, tmp_path):
        session = new_session(PROJECT)
        attach_file(session, tmp_path, "TDS.txt",
                    "d50: НЕ УКАЗАН\nБелизна ISO: 87".encode("utf-8"))
        ctx = ToolContext(spec=_spec(), session=session, root=str(tmp_path),
                          project=PROJECT)

        lst = dispatch(ctx, "list_attachments", {})
        assert lst["n"] == 1 and lst["files"][0]["name"] == "TDS.txt"

        got = dispatch(ctx, "read_attachment", {"name": "TDS.txt"})
        assert "Белизна" in got["text"] and got["has_more"] is False

        with pytest.raises(ToolError, match="нет в сессии"):
            dispatch(ctx, "read_attachment", {"name": "нет.pdf"})

    def test_attachment_tools_need_session(self):
        with pytest.raises(ToolError, match="Сессия"):
            dispatch(_ctx(), "list_attachments", {})


# ======================================================================
# 8. Стыковка с циклом модели (iter60): инструменты реально исполняются
# ======================================================================
def test_tool_loop_uses_real_tools(tmp_path):
    """Модель просит get_spec → диспетчер отдаёт НАСТОЯЩИЙ hash из ядра."""
    import json

    from src.assistant import llm

    spec = _spec()
    ctx = ToolContext(spec=spec, root=str(tmp_path), project=PROJECT)
    audit = []
    call = dispatcher(ctx, on_call=audit.append)

    responses = [
        {"choices": [{"message": {"role": "assistant", "content": "",
                                  "tool_calls": [
                                      {"id": "c1", "function": {
                                          "name": "get_spec",
                                          "arguments": json.dumps(
                                              {"include_nodes": False})}}]}}]},
        {"choices": [{"message": {"role": "assistant",
                                  "content": "спека прочитана"}}]},
    ]

    def transport(payload, *, key="", timeout=0):
        return responses.pop(0)

    res = llm.run_tool_loop([{"role": "user", "content": "какой hash?"}],
                            dispatch=call, tools=tool_specs([READONLY]),
                            transport=transport)

    tool_msg = [m for m in res.new_messages if m["role"] == "tool"][0]
    assert spec.spec_hash() in tool_msg["content"]
    assert audit and audit[0]["tool"] == "get_spec" and audit[0]["ok"] is True
