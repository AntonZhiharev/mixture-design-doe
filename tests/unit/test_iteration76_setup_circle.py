# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 76 — замкнутый круг НЕсобранного проекта (живая сессия 12.08.2026).

Три связанных отказа одного экрана:

  1. **«Ось пары 'FILLER' не найдена…»** — кнопка «🏗 Построить проект» падала
     на паре preflight по имени УЗЛА-ГРУППЫ спеки: ядро валидировало имена
     только против листьев схемы, хотя тотал группы — валидная ось-сумма.
     Хуже того, dry-run пакета (`validate_project_package`) пары паспорта
     вообще не проверял — отказ всплывал уже ПОСЛЕ принятия пакета человеком.
  2. **«Сохранить проект» до сборки было нельзя** — а сборка падала (п.1):
     заполненная форма терялась при закрытии вкладки. Теперь до сборки
     сохраняется ЧЕРНОВИК настроек (`setup_draft.json`), он виден в списке
     проектов и возвращается в поля формы при загрузке.
  3. **Помощник не видел полей формы и не мог их точечно править** — данные
     до сборки живут только в ``st.session_state``. Теперь снимок полей
     уходит в ``ToolContext.extra['setup_fields']`` (`get_setup_fields`),
     точечная правка — ``propose_setup_fields`` → стейдж → кнопка человека
     (`apply_setup_fields`, write, модели не выдан).

  Плюс защита переписки: смена имени проекта переключает сессию ассистента
  ЯВНО (``K_SWITCH_MSG``), а не молча (раньше выглядело как потеря диалога).
"""
import json

import pytest

from src.apps import campaign_state as cst
from src.apps.campaign_ui import build_setup_runner, parse_preflight_pairs
from src.assistant import context as actx
from src.assistant.consent import ACTIONS, ConsentRegistry
from src.assistant.session import (PATCH_APPLIED, PATCH_REJECTED, PATCH_STAGED,
                                   AssistantSession, StagedSetup, new_session)
from src.assistant.tools import (AGENT_KINDS, PROPOSE, READONLY, WRITE,
                                 ToolContext, ToolError, dispatch, tool_names)
from src.design import project_package as pp
from src.design.phr_sampler import PhrSpec

PROJECT = "iter76_circle"

#: Спека с группой SOFT (тотал = сумма членов) — минимальный аналог
#: FILLER/SOFT из живой сессии.
SPEC_BLOCK = {
    "spec_version": 2,
    "group_order": ["SOFT"],
    "nodes": [
        {"name": "RESIN", "role": "FIXED", "value": 100.0},
        {"name": "DINP", "role": "ABSOLUTE", "range": [4.0, 14.0]},
        {"name": "SOFT", "role": "GROUP_TOTAL", "range": [5.0, 15.0],
         "members": ["PBNK", "CPE"]},
        {"name": "PBNK", "role": "SHARE_FREE", "group": "SOFT",
         "share_range": [0.0, 0.70], "max_phr": 8.0},
        {"name": "CPE", "role": "SHARE_CLOSURE", "group": "SOFT",
         "min_phr": 3.0},
    ],
}

PACKAGE = {
    "package_kind": "project",
    "spec": SPEC_BLOCK,
    "responses": ["gloss"],
    "process": [{"name": "T_plast", "range": [165.0, 185.0], "unit": "°C"}],
}


def _spec() -> PhrSpec:
    return PhrSpec.from_dicts(SPEC_BLOCK)


def _runner(with_spec: bool = True):
    spec = _spec()
    lo, hi = spec.fraction_bounds()
    r = build_setup_runner(
        mixture_names=list(spec.component_names), process_names=["T_plast"],
        process_lower=[165.0], process_upper=[185.0],
        response_names=["gloss"],
        mixture_lower=lo.tolist(), mixture_upper=hi.tolist(), seed=1)
    if with_spec:
        r.set_phr_spec(spec)
    return r


def _ctx(tmp_path=None, *, runner=None, consent=None, session=None,
         setup_fields=None):
    extra = {}
    if consent is not None:
        extra["consent"] = consent
    if setup_fields is not None:
        extra["setup_fields"] = setup_fields
    return ToolContext(
        spec=getattr(runner, "phr_spec", None), runner=runner,
        session=session if session is not None else new_session(PROJECT),
        root=str(tmp_path) if tmp_path is not None else "",
        project=PROJECT if tmp_path is not None else "",
        extra=extra)


# ======================================================================
# 1. ЯДРО: set_preflight_pairs понимает имя группы спеки
# ======================================================================
class TestGroupPairExpansion:
    def test_group_name_expands_to_members(self):
        """Пара по группе (скрин 2 живой сессии) больше не роняет сборку."""
        r = _runner()
        r.set_preflight_pairs([("SOFT", "DINP")])
        assert r.preflight_pairs == [(["PBNK", "CPE"], ["DINP"])]

    def test_group_inside_sum_axis_expands(self):
        """Группа внутри оси-суммы тоже разворачивается в листья."""
        r = _runner()
        r.set_preflight_pairs([(["SOFT", "DINP"], "T_plast")])
        assert r.preflight_pairs == [(["PBNK", "CPE", "DINP"], ["T_plast"])]

    def test_spec_group_members_public(self):
        assert _spec().group_members() == {"SOFT": ["PBNK", "CPE"]}

    def test_unknown_name_error_lists_groups(self):
        """Отказ называет и координаты, и группы — из него виден следующий шаг."""
        r = _runner()
        with pytest.raises(KeyError, match="групп phr-спеки"):
            r.set_preflight_pairs([("GHOST", "DINP")])

    def test_without_spec_group_name_still_refused(self):
        """Без спеки группы нет — прежний явный отказ сохранён (A0.6)."""
        r = _runner(with_spec=False)
        with pytest.raises(KeyError, match="не найдена"):
            r.set_preflight_pairs([("SOFT", "DINP")])

    def test_pairs_text_from_form_with_group_builds(self):
        """Путь формы: текст «SOFT | DINP» проходит через штатный парсер."""
        r = _runner()
        r.set_preflight_pairs(parse_preflight_pairs("SOFT | DINP"))
        assert r.preflight_pairs == [(["PBNK", "CPE"], ["DINP"])]


# ======================================================================
# 2. ПАКЕТ ПРОЕКТА: пары паспорта валидируются на dry-run, не на кнопке
# ======================================================================
class TestPackagePairValidation:
    def test_group_pair_accepted_and_canonized(self):
        pkg = pp.parse_project_package({**PACKAGE, "passport": {
            "preflight_pairs": [["SOFT", "DINP"]]}})
        assert pkg.passport["preflight_pairs"] == [[["SOFT"], ["DINP"]]]

    def test_process_axis_in_pair_accepted(self):
        pkg = pp.parse_project_package({**PACKAGE, "passport": {
            "preflight_pairs": [{"left": ["T_plast"], "right": "DINP"}]}})
        assert pkg.passport["preflight_pairs"] == [[["T_plast"], ["DINP"]]]

    def test_unknown_pair_name_refused_at_parse(self):
        """Главный фикс: ошибка ловится dry-run'ом, а не кнопкой сборки."""
        with pytest.raises(pp.PackageError, match="FILLER"):
            pp.parse_project_package({**PACKAGE, "passport": {
                "preflight_pairs": [["FILLER", "DINP"]]}})

    def test_validate_tool_returns_refusal_as_result(self):
        """validate_project_package: отказ — РЕЗУЛЬТАТ (ok=False), не взрыв."""
        out = dispatch(_ctx(), "validate_project_package",
                       {"package": {**PACKAGE, "passport": {
                           "preflight_pairs": [["FILLER", "DINP"]]}}},
                       allowed_kinds=[READONLY])
        assert out["ok"] is False and "FILLER" in out["error"]

    def test_prefill_pair_text_builds_runner(self):
        """Сквозной круг: пакет с парой по группе → префилл → сборка ядром."""
        pkg = pp.parse_project_package({**PACKAGE, "passport": {
            "preflight_pairs": [["SOFT", "DINP"]]}})
        pre = pp.package_to_setup_prefill(pkg)
        r = _runner()
        r.set_preflight_pairs(
            parse_preflight_pairs(pre["setup_preflight_pairs"]))
        assert r.preflight_pairs == [(["PBNK", "CPE"], ["DINP"])]


# ======================================================================
# 3. ЧЕРНОВИК НАСТРОЕК: сохранить/загрузить проект ДО сборки
# ======================================================================
class TestSetupDraft:
    FIELDS = {"setup_mix": "A, B, C", "setup_resp": "gloss",
              "setup_seed": 7, "setup_econ_on": True,
              "setup_pass_weigh_step": 0.1}

    def test_fields_snapshot_filters_scalars_and_prefix(self):
        state = {**self.FIELDS,
                 "setup_seed_X": [[1.0]],            # объект прогона — мимо
                 "setup_phr_spec_obj": object(),     # не сериализуем — мимо
                 "campaign_name": "x",               # не setup_* — мимо
                 "setup_list": [1, 2]}               # не скаляр — мимо
        assert cst.setup_draft_fields(state) == self.FIELDS

    def test_save_load_roundtrip(self, tmp_path):
        cst.save_setup_draft(tmp_path, "p1", self.FIELDS)
        assert cst.load_setup_draft(tmp_path, "p1") == self.FIELDS

    def test_empty_draft_refused(self, tmp_path):
        with pytest.raises(ValueError, match="Черновик пуст"):
            cst.save_setup_draft(tmp_path, "p1", {})

    def test_draft_only_project_listed_and_not_built(self, tmp_path):
        cst.save_setup_draft(tmp_path, "draft_p", self.FIELDS)
        assert cst.list_campaigns(tmp_path) == ["draft_p"]
        assert cst.campaign_exists(tmp_path, "draft_p") is False

    def test_missing_draft_is_none(self, tmp_path):
        assert cst.load_setup_draft(tmp_path, "nope") is None

    def test_delete_removes_draft_only_project(self, tmp_path):
        cst.save_setup_draft(tmp_path, "draft_p", self.FIELDS)
        assert cst.delete_campaign(tmp_path, "draft_p") is True
        assert cst.list_campaigns(tmp_path) == []

    def test_saving_built_project_drops_stale_draft(self, tmp_path):
        """После сборки источник истины — campaign.json: черновик устарел."""
        cst.save_setup_draft(tmp_path, "p1", self.FIELDS)
        cst.save_campaign(_runner(), tmp_path, "p1")
        assert cst.load_setup_draft(tmp_path, "p1") is None
        assert cst.campaign_exists(tmp_path, "p1") is True


# ======================================================================
# 4. СЕССИЯ: StagedSetup — стейдж, статусы, совместимость
# ======================================================================
class TestStagedSetupSession:
    def test_round_trip_state(self):
        s = new_session(PROJECT)
        s.stage_setup(StagedSetup(fields={"setup_resp": "gloss, rho"},
                                  label="добавить rho", rationale="§3"))
        s2 = AssistantSession.from_state(s.to_state())
        assert len(s2.setups) == 1
        assert s2.setups[0].fields == {"setup_resp": "gloss, rho"}
        assert s2.setups[0].status == PATCH_STAGED

    def test_empty_fields_refused(self):
        with pytest.raises(ValueError, match="пуста"):
            new_session(PROJECT).stage_setup(StagedSetup(fields={}))

    def test_terminal_status_transition_once(self):
        s = new_session(PROJECT)
        st = s.stage_setup(StagedSetup(fields={"setup_seed": 2}))
        s.set_setup_status(st.id, PATCH_APPLIED)
        with pytest.raises(ValueError, match="повторный переход"):
            s.set_setup_status(st.id, PATCH_REJECTED)

    def test_old_session_without_setups_loads(self):
        state = new_session(PROJECT).to_state()
        state.pop("setups", None)
        assert AssistantSession.from_state(state).setups == []


# ======================================================================
# 5. ИНСТРУМЕНТЫ: снимок формы + точечная правка полей
# ======================================================================
class TestSetupFieldTools:
    def test_access_classes(self):
        assert "get_setup_fields" in tool_names([READONLY])
        assert "propose_setup_fields" in tool_names([PROPOSE])
        wr = tool_names([WRITE])
        assert "apply_setup_fields" in wr and "reject_setup_fields" in wr
        # write модели не выдан
        assert "apply_setup_fields" not in tool_names(AGENT_KINDS)
        assert "apply_setup" in ACTIONS and "reject_setup" in ACTIONS

    def test_get_setup_fields_reads_snapshot(self):
        out = dispatch(_ctx(setup_fields={"setup_resp": "gloss"}),
                       "get_setup_fields", {}, allowed_kinds=[READONLY])
        assert out["fields"] == {"setup_resp": "gloss"} and out["n"] == 1
        assert "propose_setup_fields" in out["note"]

    def test_get_setup_fields_empty_without_ui(self):
        out = dispatch(_ctx(), "get_setup_fields", {},
                       allowed_kinds=[READONLY])
        assert out["n"] == 0 and "первичный ввод" in out["note"]

    def test_propose_allowed_when_project_built(self):
        """iter94: у собранного проекта правка полей больше НЕ отвергается.

        Прежде здесь стоял слепой отказ «проект уже СОБРАН», и он бил даже по
        проекту с пустой базой, где терять было нечего. Теперь правка ложится в
        стейдж, а ответ несёт стадию и маршрут применения по каждому полю.
        """
        out = dispatch(_ctx(runner=_runner()), "propose_setup_fields",
                       {"fields": {"setup_resp": "x"}, "rationale": "r"},
                       allowed_kinds=[PROPOSE])
        assert out["staged"] is True
        assert out["stage"] == "empty"          # движок есть, база пуста
        assert out["routes"]["setup_resp"]["route"] == "live"

    def test_propose_refuses_non_setup_keys_and_objects(self):
        ctx = _ctx()
        out = dispatch(ctx, "propose_setup_fields",
                       {"fields": {"campaign_name": "x"}, "rationale": "r"},
                       allowed_kinds=[PROPOSE])
        assert out["staged"] is False and "setup_" in out["error"]
        out = dispatch(ctx, "propose_setup_fields",
                       {"fields": {"setup_resp": ["gloss"]}, "rationale": "r"},
                       allowed_kinds=[PROPOSE])
        assert out["staged"] is False and "скаляр" in out["error"].lower()

    def test_propose_returns_current_values_for_diff(self):
        ctx = _ctx(setup_fields={"setup_resp": "gloss"})
        out = dispatch(ctx, "propose_setup_fields",
                       {"fields": {"setup_resp": "gloss, rho"},
                        "rationale": "ρ — отклик (§3)"},
                       allowed_kinds=[PROPOSE])
        assert out["staged"] is True
        assert out["current_values"] == {"setup_resp": "gloss"}
        assert ctx.session.staged_setups()[0].id == out["setup_id"]

    def test_apply_flow_returns_prefill_and_logs(self, tmp_path):
        """Кнопка человека: правка → setup_prefill + журнал + статус applied."""
        ctx = _ctx(tmp_path, consent=ConsentRegistry())
        out = dispatch(ctx, "propose_setup_fields",
                       {"fields": {"setup_resp": "gloss, rho"},
                        "rationale": "ρ"}, allowed_kinds=[PROPOSE])
        res = actx.human_apply_setup(ctx, out["setup_id"],
                                     author="человек (UI)")
        assert res["ok"] is True and res["status"] == PATCH_APPLIED
        assert res["setup_prefill"] == {"setup_resp": "gloss, rho"}
        assert "Построить проект" in res["next_step"]
        assert res["decision"]["kind"] == "apply_setup"

    def test_apply_is_single_use(self, tmp_path):
        ctx = _ctx(tmp_path, consent=ConsentRegistry())
        out = dispatch(ctx, "propose_setup_fields",
                       {"fields": {"setup_seed": 5}, "rationale": "r"},
                       allowed_kinds=[PROPOSE])
        actx.human_apply_setup(ctx, out["setup_id"])
        with pytest.raises(ToolError, match="уже в статусе"):
            actx.human_apply_setup(ctx, out["setup_id"])

    def test_apply_allowed_when_project_built(self, tmp_path):
        """iter94: правка применяется и после сборки — поля это черновик.

        Раньше здесь был ``ToolError``: правка, предложенная до сборки,
        застревала в стейдже навсегда. Теперь она применяется (движок при этом
        НЕ трогается — меняются только поля формы), а стадия отдаётся в ответе.
        """
        ctx = _ctx(tmp_path, consent=ConsentRegistry())
        out = dispatch(ctx, "propose_setup_fields",
                       {"fields": {"setup_seed": 5}, "rationale": "r"},
                       allowed_kinds=[PROPOSE])
        ctx.runner = _runner()
        res = actx.human_apply_setup(ctx, out["setup_id"])
        assert res["ok"] is True and res["stage"] == "empty"
        assert res["setup_prefill"] == {"setup_seed": 5}

    def test_reject_flow_logs_reason(self, tmp_path):
        ctx = _ctx(tmp_path, consent=ConsentRegistry())
        out = dispatch(ctx, "propose_setup_fields",
                       {"fields": {"setup_seed": 5}, "rationale": "r"},
                       allowed_kinds=[PROPOSE])
        res = actx.human_reject_setup(ctx, out["setup_id"], "не согласен",
                                      author="человек (UI)")
        assert res["status"] == PATCH_REJECTED
        assert ctx.session.setup_by_id(out["setup_id"]).status == \
            PATCH_REJECTED

    def test_model_cannot_call_write_directly(self):
        with pytest.raises(ToolError, match="классу 'write'"):
            dispatch(_ctx(), "apply_setup_fields",
                     {"setup_id": "x", "human_token": "y"},
                     allowed_kinds=list(AGENT_KINDS))


# ======================================================================
# 6. Снимок формы уходит модели как JSON
# ======================================================================
def test_setup_fields_snapshot_json_serializable():
    """Снимок формы уходит модели как JSON — несериализуемое отфильтровано."""
    fields = cst.setup_draft_fields({
        "setup_mix": "A, B", "setup_seed": 1, "setup_econ_on": False,
        "setup_phr_spec_obj": object()})
    json.dumps(fields)                                # не должно упасть
    assert "setup_phr_spec_obj" not in fields
