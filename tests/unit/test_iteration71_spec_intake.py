# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 71 / ASSISTANT_SPEC — ПАКЕТ спеки: первичный ввод и эволюция.

Закрываемый отказ (наблюдался в живой сессии): ассистент собрал геометрию
кампании верно ЧИСЛЕННО, но не смог отдать её в проект. Причины обе не в
модели:

  1. **инструмента не было вовсе** — ``propose_patch`` правит ПОЛЕ
     существующего узла и требует уже существующую спеку, поэтому первичный
     ввод и добавление/удаление узла им невыразимы в принципе;
  2. **схема v2 модели не сообщалась** — формат приходилось восстанавливать по
     markdown-таблицам переписки, и JSON выходил с ключами
     ``components``/``groups``/``process``/``levels``, которые валидатор
     отвергает целиком.

Что закрывают тесты:

  * ``spec_schema`` отдаёт схему ИЗ ЯДРА (``_ROLE_TABLE``) и работает БЕЗ
    активной спеки — иначе первичный ввод снова упирается в её отсутствие;
  * ``validate_spec_package`` объясняет ровно те два промаха, на которых
    сорвалась живая сессия, и не роняет ход;
  * ``propose_spec`` (класс ``propose``) кладёт пакет в СТЕЙДЖ; невалидный
    пакет в стейдж не попадает;
  * ``apply_spec``/``reject_spec`` — класс ``write``: модели не выданы, требуют
    разовый токен человека, привязанный к отпечатку спеки на момент нажатия;
  * гейты применения ЧЕСТНЫ: при первичном вводе и при смене состава
    компонентов «не проверено» не выдаётся за «пройдено»;
  * применение и отказ пишутся в ``decision_log.jsonl``;
  * пакет переживает сохранение/загрузку сессии, а сессии БЕЗ пакетов
    (записанные до iter71) читаются как раньше.
"""
import json

import numpy as np
import pytest

from src.assistant import context as actx
from src.assistant import store
from src.assistant.consent import ACTIONS, ConsentError, ConsentRegistry
from src.assistant.session import (PATCH_APPLIED, PATCH_REJECTED, PATCH_STAGED,
                                    AssistantSession, StagedSpec, new_session)
from src.assistant.tools import (AGENT_KINDS, PROPOSE, READONLY, WRITE,
                                  ToolContext, ToolError, dispatch, tool_names)
from src.assistant.tools.write import (issue_apply_spec_token,
                                        issue_reject_spec_token)
from src.assistant.views import (session_caption, spec_apply_caption,
                                  staged_specs_dataframe)
from src.design.phr_sampler import PhrSpec

PROJECT = "pvc_edge_v1"

#: Референсная геометрия (та же, что в iter61/63): группа k=2 с техлимитами,
#: лог-ось и cap-фаза — по одному представителю каждого класса узлов.
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

PACKAGE = {"spec_version": 2, "group_order": ["SOFT"], "nodes": NODES}

#: Ровно то, что прислала модель в живой сессии: своя обёртка вместо 'nodes'.
BAD_WRAPPER = {
    "version": "pvc_edge_v1",
    "group_order": ["SOFT"],
    "components": [{"name": "DINP", "role": "ABSOLUTE", "range": [4, 14]}],
    "groups": [],
    "process": [{"name": "T_plast", "range": [165, 185], "levels": 3}],
}

#: Второй промах живой сессии: 'levels' внутри узла.
BAD_KEYS = {"spec_version": 2, "nodes": [
    dict(d, **({"levels": 3} if d["name"] == "DINP" else {})) for d in NODES]}


def _spec() -> PhrSpec:
    return PhrSpec.from_dicts(PACKAGE)


class _Report:
    """Минимальный duck-typed отчёт preflight (нужны passed/failures)."""

    def __init__(self, passed: bool, failures=()):
        self.passed = bool(passed)
        self.failures = list(failures)


class _Runner:
    """Движок в объёме, который нужен write-инструментам (как в iter63)."""

    def __init__(self, spec=None, *, X=None, preflights=None, names=None):
        self.phr_spec = spec
        self.X = (np.atleast_2d(np.asarray(X, float))
                  if X is not None else np.empty((0, 0)))
        self._preflights = list(preflights or [])
        self._names = names
        self.applied = []

    def set_phr_spec(self, spec):
        # Имитация ШТАТНОЙ проверки runner.set_phr_spec: компоненты спеки
        # обязаны существовать среди mixture-компонентов схемы.
        if self._names is not None:
            unknown = [nm for nm in spec.component_names
                       if nm not in set(self._names)]
            if unknown:
                raise KeyError(f"Компоненты phr-спеки {unknown} не найдены "
                               f"среди mixture-компонентов схемы.")
        self.phr_spec = spec
        self.applied.append(spec.spec_hash())

    def preflight(self, X):
        if not self._preflights:
            return _Report(True)
        return self._preflights.pop(0)


def _ctx(tmp_path=None, *, spec=None, runner=None, consent=None, session=None):
    return ToolContext(
        spec=spec, runner=runner,
        session=session if session is not None else new_session(PROJECT),
        root=str(tmp_path) if tmp_path is not None else "",
        project=PROJECT if tmp_path is not None else "",
        extra={"consent": consent} if consent is not None else {})


def _stage(ctx, package=PACKAGE, **kw):
    """Положить пакет в стейдж и вернуть его id."""
    out = dispatch(ctx, "propose_spec",
                   {"package": package,
                    "rationale": kw.pop("rationale", "геометрия кампании"),
                    **kw},
                   allowed_kinds=[READONLY, PROPOSE])
    assert out["staged"] is True
    return out["spec_id"]


# ======================================================================
# 1. spec_schema — схема ИЗ ЯДРА и БЕЗ активной спеки
# ======================================================================
class TestSpecSchema:

    def test_works_without_active_spec(self):
        """Главный смысл инструмента: он нужен, когда спеки ЕЩЁ НЕТ."""
        out = dispatch(_ctx(), "spec_schema", {})
        assert out["current"]["present"] is False
        assert "ПЕРВИЧНЫЙ" in out["current"]["note"]

    def test_roles_come_from_core_table(self):
        """Схема не пересказана в промпте, а прочитана из ``_ROLE_TABLE``."""
        from src.design.phr_sampler import _ROLE_TABLE

        out = dispatch(_ctx(), "spec_schema", {})
        assert set(out["roles"]) == set(_ROLE_TABLE)
        for role, (mode, required, allowed) in _ROLE_TABLE.items():
            assert out["roles"][role]["mode"] == mode
            assert out["roles"][role]["required"] == sorted(required)
            assert out["roles"][role]["allowed"] == sorted(allowed)

    def test_closure_has_no_range_in_schema(self):
        """Инвариант B8 виден МОДЕЛИ, а не только валидатору."""
        closure = dispatch(_ctx(), "spec_schema", {})["roles"]["SHARE_CLOSURE"]
        assert "range" not in closure["allowed"]
        assert "share_range" not in closure["allowed"]

    def test_example_is_actually_valid(self):
        """Пример из схемы обязан собираться ядром.

        Пример, который не собирается, хуже отсутствия примера: модель повторит
        его буквально и получит отказ.
        """
        out = dispatch(_ctx(), "spec_schema", {})
        spec = PhrSpec.from_dicts(out["example"])
        assert spec.spec_hash() and spec.q >= 1

    def test_not_in_spec_names_the_traps(self):
        """Ключи, на которых сорвалась живая сессия, названы явно."""
        out = dispatch(_ctx(), "spec_schema", {})
        for key in ("levels", "premix", "process", "components/groups"):
            assert key in out["not_in_spec"]
        assert "set_process_levels" in out["not_in_spec"]["levels"]

    def test_shows_hash_when_spec_present(self):
        out = dispatch(_ctx(spec=_spec()), "spec_schema",
                       {"include_example": False})
        assert out["current"]["present"] is True
        assert out["current"]["spec_hash"] == _spec().spec_hash()
        assert "example" not in out

    def test_exported_to_mcp_as_readonly(self):
        """Схема — знание, а не действие: обязана быть в read-only MCP."""
        import src.mcp.campaign_tools as ct

        assert "spec_schema" in ct.exported_names()
        assert "validate_spec_package" in ct.exported_names()


# ======================================================================
# 2. validate_spec_package — объясняет ровно те промахи, что были в жизни
# ======================================================================
class TestValidatePackage:

    def test_foreign_wrapper_explained(self):
        """'components'/'groups'/'process' → указание на плоский 'nodes'."""
        with pytest.raises(ToolError, match="nodes"):
            dispatch(_ctx(), "validate_spec_package", {"package": BAD_WRAPPER})

    def test_extra_wrapper_keys_rejected(self):
        with pytest.raises(ToolError, match="Лишние ключи обёртки"):
            dispatch(_ctx(), "validate_spec_package",
                     {"package": {"spec_version": 2, "nodes": NODES,
                                  "version": "v1"}})

    def test_levels_inside_node_is_result_not_crash(self):
        """Отказ ядра — РЕЗУЛЬТАТ вызова: модель должна получить причину."""
        out = dispatch(_ctx(), "validate_spec_package", {"package": BAD_KEYS})
        assert out["ok"] is False
        assert "levels" in out["error"]
        assert "spec_schema" in out["hint"]

    def test_empty_package_refused(self):
        with pytest.raises(ToolError, match="без узлов"):
            dispatch(_ctx(), "validate_spec_package",
                     {"package": {"spec_version": 2, "nodes": []}})

    def test_first_spec_diff_and_warning(self):
        out = dispatch(_ctx(), "validate_spec_package", {"package": PACKAGE})
        assert out["ok"] is True
        assert out["diff"]["first_spec"] is True
        assert out["diff"]["q_before"] == 0
        assert out["q_components"] == _spec().q
        assert out["spec_hash"] == _spec().spec_hash()
        assert "ПЕРВИЧНЫЙ" in out["warning"]

    def test_diff_reports_added_removed_and_hash(self):
        """Дифф отвечает на вопрос «что именно я утверждаю»."""
        pkg = json.loads(json.dumps(PACKAGE))
        pkg["nodes"] = [d for d in pkg["nodes"] if d["name"] != "TiO2"]
        pkg["nodes"].append({"name": "CaSt", "role": "ABSOLUTE",
                             "range": [0.1, 1.0]})
        out = dispatch(_ctx(spec=_spec()), "validate_spec_package",
                       {"package": pkg})
        assert out["ok"] is True
        assert out["diff"]["added"] == ["CaSt"]
        assert out["diff"]["removed"] == ["TiO2"]
        assert out["diff"]["affects_hash"] is True
        assert "УДАЛЯЕТ" in out["warning"]

    def test_role_change_is_reported(self):
        """Смена роли — то, чего патч не умеет; дифф обязан её показать.

        Группа k=2 растёт до k=3: по инварианту ядра closure при k≥3 запрещён,
        поэтому ВСЕ члены становятся SHARE_SIMPLEX — это одновременно и смена
        роли, и добавление узла, то есть чистая эволюция схемы.
        """
        pkg = json.loads(json.dumps(PACKAGE))
        for d in pkg["nodes"]:
            if d["name"] in ("PBNK", "CPE"):
                d["role"] = "SHARE_SIMPLEX"
                d["share_range"] = [0.1, 0.6]
                d.pop("max_phr", None)
                d.pop("min_phr", None)
            if d["name"] == "SOFT":
                d["members"] = ["PBNK", "CPE", "MBS"]
        pkg["nodes"].append({"name": "MBS", "role": "SHARE_SIMPLEX",
                             "group": "SOFT", "share_range": [0.1, 0.6]})
        out = dispatch(_ctx(spec=_spec()), "validate_spec_package",
                       {"package": pkg})
        assert out["ok"] is True, out.get("error")
        changed = {r["node"]: (r["before"], r["after"])
                   for r in out["diff"]["role_changed"]}
        assert changed["CPE"] == ("SHARE_CLOSURE", "SHARE_SIMPLEX")
        assert out["diff"]["added"] == ["MBS"]

    def test_does_not_touch_project_spec(self):
        """dry-run: активная геометрия остаётся прежней бит-в-бит."""
        spec = _spec()
        before = spec.spec_hash()
        ctx = _ctx(spec=spec)
        pkg = json.loads(json.dumps(PACKAGE))
        pkg["nodes"][1]["range"] = [4.0, 20.0]
        dispatch(ctx, "validate_spec_package", {"package": pkg})
        assert ctx.spec.spec_hash() == before

    def test_flat_list_accepted(self):
        """Пакет без group_order — плоский список, как в ``from_dicts``."""
        out = dispatch(_ctx(), "validate_spec_package", {"package": NODES})
        assert out["ok"] is True
        assert out["diff"]["group_order_after"] == []


# ======================================================================
# 3. propose_spec — СТЕЙДЖ, доступный модели; применение ей недоступно
# ======================================================================
class TestProposeSpec:

    def test_agent_may_propose_but_not_apply(self):
        """Граница классов: предложение — модели, применение — человеку."""
        agent = set(tool_names(AGENT_KINDS))
        assert {"spec_schema", "validate_spec_package", "propose_spec"} <= agent
        assert "apply_spec" not in agent
        assert "reject_spec" not in agent
        assert {"apply_spec", "reject_spec"} <= set(tool_names([WRITE]))

    def test_stages_package_without_touching_project(self):
        ctx = _ctx()
        sid = _stage(ctx, label="кромка ПВХ: первичный ввод", level="L1")
        assert [s.id for s in ctx.session.staged_specs()] == [sid]
        assert ctx.session.spec_by_id(sid).status == PATCH_STAGED
        assert ctx.spec is None          # пакет только предложен

    def test_invalid_package_not_staged(self):
        """Заведомо неприменимый пакет не должен попадать в панель кнопок."""
        ctx = _ctx()
        out = dispatch(ctx, "propose_spec",
                       {"package": BAD_KEYS, "rationale": "проба"},
                       allowed_kinds=[READONLY, PROPOSE])
        assert out["staged"] is False and out["ok"] is False
        assert "levels" in out["error"]
        assert ctx.session.staged_specs() == []

    def test_first_spec_warning_names_future_hash(self):
        out = dispatch(_ctx(), "propose_spec",
                       {"package": PACKAGE, "rationale": "первичный ввод"},
                       allowed_kinds=[READONLY, PROPOSE])
        assert out["diff"]["first_spec"] is True
        assert _spec().spec_hash()[:12] in out["warning"]

    def test_removal_warning_is_loud(self):
        pkg = json.loads(json.dumps(PACKAGE))
        pkg["nodes"] = [d for d in pkg["nodes"] if d["name"] != "TiO2"]
        out = dispatch(_ctx(spec=_spec()), "propose_spec",
                       {"package": pkg, "rationale": "TiO2 выведен"},
                       allowed_kinds=[READONLY, PROPOSE])
        assert out["diff"]["removed"] == ["TiO2"]
        assert "УДАЛЯЕТ" in out["warning"]

    def test_rationale_is_required(self):
        with pytest.raises(ToolError, match="rationale"):
            dispatch(_ctx(), "propose_spec", {"package": PACKAGE},
                     allowed_kinds=[READONLY, PROPOSE])

    def test_empty_nodes_refused_by_session(self):
        """Страховка контейнера: пустой пакет в стейдж не ложится."""
        with pytest.raises(ValueError, match="без узлов"):
            new_session(PROJECT).stage_spec(StagedSpec(nodes=[]))


# ======================================================================
# 4. apply_spec — только человек, только разовым токеном
# ======================================================================
class TestApplyConsent:

    def test_actions_registered(self):
        assert "apply_spec" in ACTIONS and "reject_spec" in ACTIONS

    def test_without_token_refused(self):
        ctx = _ctx(consent=ConsentRegistry())
        sid = _stage(ctx)
        with pytest.raises(ToolError, match="подтвержд"):
            dispatch(ctx, "apply_spec", {"spec_id": sid, "human_token": ""},
                     allowed_kinds=[WRITE])
        assert ctx.session.spec_by_id(sid).status == PATCH_STAGED

    def test_token_is_single_use(self):
        ctx = _ctx(consent=ConsentRegistry())
        sid = _stage(ctx)
        token = issue_apply_spec_token(ctx, sid)
        dispatch(ctx, "apply_spec", {"spec_id": sid, "human_token": token},
                 allowed_kinds=[WRITE])
        sid2 = _stage(ctx, rationale="ещё раз")
        with pytest.raises(ToolError, match="уже использован"):
            dispatch(ctx, "apply_spec", {"spec_id": sid2, "human_token": token},
                     allowed_kinds=[WRITE])

    def test_token_bound_to_this_package(self):
        """Подтверждение одной геометрии не применяет другую."""
        ctx = _ctx(consent=ConsentRegistry())
        sid_a = _stage(ctx, rationale="вариант A")
        sid_b = _stage(ctx, package=NODES, rationale="вариант B")
        token_a = issue_apply_spec_token(ctx, sid_a)
        with pytest.raises(ToolError, match="ДРУГОЙ объект"):
            dispatch(ctx, "apply_spec", {"spec_id": sid_b,
                                         "human_token": token_a},
                     allowed_kinds=[WRITE])

    def test_patch_token_does_not_apply_spec(self):
        """Согласие на сдвиг границы не годится для замены всей спеки."""
        from src.assistant.tools.write import issue_apply_token

        ctx = _ctx(spec=_spec(), consent=ConsentRegistry())
        sid = _stage(ctx, package=NODES, rationale="иная геометрия")
        token = issue_apply_token(ctx, sid)           # действие apply_patch
        with pytest.raises(ToolError, match="действие 'apply_patch'"):
            dispatch(ctx, "apply_spec", {"spec_id": sid, "human_token": token},
                     allowed_kinds=[WRITE])

    def test_geometry_changed_after_confirmation(self):
        """Гонка «спеку подменили между кнопкой и применением» закрыта."""
        ctx = _ctx(spec=_spec(), consent=ConsentRegistry())
        sid = _stage(ctx, package=NODES, rationale="плоский вариант")
        token = issue_apply_spec_token(ctx, sid)
        other = json.loads(json.dumps(PACKAGE))
        other["nodes"][1]["range"] = [4.0, 20.0]
        ctx.spec = PhrSpec.from_dicts(other)          # отпечаток сдвинулся
        with pytest.raises(ToolError, match="Геометрия изменилась"):
            dispatch(ctx, "apply_spec", {"spec_id": sid, "human_token": token},
                     allowed_kinds=[WRITE])

    def test_unknown_id_explained(self):
        ctx = _ctx(consent=ConsentRegistry())
        with pytest.raises(ToolError, match="нет в сессии"):
            dispatch(ctx, "apply_spec", {"spec_id": "spec_zzz",
                                         "human_token": "t"},
                     allowed_kinds=[WRITE])

    def test_double_apply_refused(self):
        ctx = _ctx(consent=ConsentRegistry())
        sid = _stage(ctx)
        actx.human_apply_spec(ctx, sid)
        with pytest.raises(ToolError, match="повторное применение"):
            actx.human_apply_spec(ctx, sid)


# ======================================================================
# 5. Гейты применения ЧЕСТНЫ + спека доезжает до проекта
# ======================================================================
class TestGatesAndEffect:

    def test_first_spec_reaches_runner_and_context(self):
        """Первичный ввод: геометрия появляется в проекте, гейты неприменимы."""
        runner = _Runner(spec=None,
                         names=[d["name"] for d in NODES])       # схема шире
        ctx = _ctx(runner=runner, consent=ConsentRegistry())
        sid = _stage(ctx, label="первичный ввод")
        out = actx.human_apply_spec(ctx, sid, author="человек (тест)")
        assert out["ok"] is True
        assert out["spec_hash_before"] == ""
        assert out["spec_hash_after"] == _spec().spec_hash()
        assert runner.phr_spec.spec_hash() == _spec().spec_hash()
        assert ctx.spec.spec_hash() == _spec().spec_hash()
        # «не проверено» НЕ выдаётся за «пройдено»
        assert out["gates"]["checked"] is False
        assert "сравнивать не с чем" in out["gates"]["reason"]
        assert out["gates"]["points"]["checked"] is False
        assert "зафиксирована" in out["warning"]

    def test_narrowing_bounds_blocked_by_existing_points(self):
        """Состав тот же ⇒ работает обычный гейт точек (сужение задним числом).

        Точка с DINP = 13 phr уже измерена; пакет с верхом 8 выбрасывает её из
        геометрии — применение блокируется, пакет остаётся в стейдже.
        """
        spec = _spec()
        # DINP = 13 phr ставим явно: гейт должен ловить именно эту точку
        phr = spec.decode(spec.sample_z(1, seed=0))[0]
        phr[list(spec.component_names).index("DINP")] = 13.0
        measured = np.atleast_2d(spec.to_fractions(phr))
        runner = _Runner(spec=spec, X=measured)
        ctx = _ctx(spec=spec, runner=runner, consent=ConsentRegistry())
        pkg = json.loads(json.dumps(PACKAGE))
        pkg["nodes"][1]["range"] = [4.0, 8.0]
        sid = _stage(ctx, package=pkg, rationale="сузить верх DINP")
        with pytest.raises(ToolError, match="ГЕЙТ ПРИМЕНЕНИЯ"):
            actx.human_apply_spec(ctx, sid)
        assert ctx.session.spec_by_id(sid).status == PATCH_STAGED
        assert ctx.spec.spec_hash() == spec.spec_hash()   # геометрия прежняя

    def test_component_change_marks_history_break_not_silent(self):
        """Смена состава: гейт точек неприменим, но разрыв истории назван."""
        spec = _spec()
        runner = _Runner(spec=spec,
                         names=[d["name"] for d in NODES] + ["CaSt"])
        ctx = _ctx(spec=spec, runner=runner, consent=ConsentRegistry())
        pkg = json.loads(json.dumps(PACKAGE))
        pkg["nodes"].append({"name": "CaSt", "role": "ABSOLUTE",
                             "range": [0.1, 1.0]})
        sid = _stage(ctx, package=pkg, rationale="вводим стеарат кальция")
        out = actx.human_apply_spec(ctx, sid)
        assert out["ok"] is True
        assert out["gates"]["history_break"] is True
        assert out["gates"]["checked"] is False
        assert out["diff"]["components_added"] == ["CaSt"]
        assert "ДРУГОМУ пространству" in out["warning"]

    def test_component_unknown_to_schema_refused(self):
        """Расширение состава сверх схемы — эволюция СХЕМЫ, а не пакета."""
        runner = _Runner(spec=None, names=["DINP"])      # схема узкая
        ctx = _ctx(runner=runner, consent=ConsentRegistry())
        sid = _stage(ctx)
        with pytest.raises(ToolError, match="не принята проектом"):
            actx.human_apply_spec(ctx, sid)
        assert ctx.session.spec_by_id(sid).status == PATCH_STAGED

    def test_role_change_applies_when_components_same(self):
        """Обмен closure↔free при том же составе: гейты обычные, спека доезжает.

        Это ровно то, чего патч не умеет (роль — не поле границы), но состав
        компонентов не меняется, поэтому гейт точек ПРИМЕНИМ и должен работать.
        """
        spec = _spec()
        pkg = json.loads(json.dumps(PACKAGE))
        for d in pkg["nodes"]:
            if d["name"] == "PBNK":                 # был free → стал closure
                d["role"] = "SHARE_CLOSURE"
                d.pop("share_range", None)
            elif d["name"] == "CPE":                # был closure → стал free
                d["role"] = "SHARE_FREE"
                d["share_range"] = [0.3, 1.0]
        runner = _Runner(spec=spec, names=[d["name"] for d in NODES])
        ctx = _ctx(spec=spec, runner=runner, consent=ConsentRegistry())
        sid = _stage(ctx, package=pkg, rationale="closure переносим на PBNK")
        out = actx.human_apply_spec(ctx, sid)
        assert out["ok"] is True
        assert out["gates"]["history_break"] is False
        assert out["gates"]["checked"] is True      # состав тот же ⇒ гейты идут
        assert out["affects_hash"] is True
        roles = {r["node"]: r["after"] for r in out["diff"]["role_changed"]}
        assert roles["CPE"] == "SHARE_FREE"
        assert roles["PBNK"] == "SHARE_CLOSURE"
        assert runner.phr_spec.role_of("PBNK") == "SHARE_CLOSURE"


# ======================================================================
# 6. Журнал решений, отказ, персистентность и показ
# ======================================================================
class TestJournalAndPersistence:

    def test_apply_writes_decision(self, tmp_path):
        ctx = _ctx(tmp_path, consent=ConsentRegistry())
        sid = _stage(ctx, label="кромка ПВХ: первичный ввод", level="L1",
                     source="таблица технолога", rationale="геометрия принята")
        actx.human_apply_spec(ctx, sid, author="Жихарев",
                              note="сверено с паспортами")
        log = store.read_log(tmp_path, PROJECT, "decisions")
        assert len(log) == 1
        rec = log[0]
        assert rec["kind"] == "apply_spec"
        assert rec["spec_id"] == sid
        assert rec["first_spec"] is True
        assert rec["author"] == "Жихарев"
        assert rec["level"] == "L1"
        assert rec["spec_hash_after"] == _spec().spec_hash()

    def test_reject_also_writes_decision(self, tmp_path):
        """Отказ фиксируется наравне с применением (спор решает журнал)."""
        ctx = _ctx(tmp_path, consent=ConsentRegistry())
        sid = _stage(ctx, label="вариант с мелом 2–100")
        actx.human_reject_spec(ctx, sid, "Σphr едет ×2.17, ждём preflight",
                               author="Жихарев")
        assert ctx.session.spec_by_id(sid).status == PATCH_REJECTED
        log = store.read_log(tmp_path, PROJECT, "decisions")
        assert log[0]["kind"] == "reject_spec"
        assert "preflight" in log[0]["rationale"]

    def test_rejected_cannot_be_applied(self, tmp_path):
        ctx = _ctx(tmp_path, consent=ConsentRegistry())
        sid = _stage(ctx)
        actx.human_reject_spec(ctx, sid, "не сейчас")
        with pytest.raises(ToolError, match="уже в статусе"):
            actx.human_apply_spec(ctx, sid)

    def test_package_survives_save_load(self, tmp_path):
        ctx = _ctx(tmp_path, consent=ConsentRegistry())
        sid = _stage(ctx, label="кромка ПВХ")
        store.save_session(ctx.session, tmp_path, PROJECT)
        again = store.load_session(tmp_path, PROJECT)
        s = again.spec_by_id(sid)
        assert s is not None and s.status == PATCH_STAGED
        assert s.label == "кромка ПВХ"
        # пакет round-trip'ится в ТУ ЖЕ геометрию (отпечаток бит-в-бит)
        assert PhrSpec.from_dicts(s.payload()).spec_hash() == _spec().spec_hash()

    def test_old_session_without_specs_loads(self):
        """Сессии, записанные до iter71, читаются как раньше (ключа нет)."""
        state = new_session(PROJECT).to_state()
        state.pop("specs")
        s = AssistantSession.from_state(state)
        assert s.specs == []
        assert s.staged_specs() == []

    def test_payload_shape_matches_to_dicts(self):
        """Пакет с group_order — обёртка; без него — плоский список."""
        wrapped = StagedSpec(nodes=NODES, group_order=["SOFT"])
        flat = StagedSpec(nodes=NODES)
        assert isinstance(wrapped.payload(), dict)
        assert wrapped.payload()["spec_version"] == 2
        assert isinstance(flat.payload(), list)

    def test_dataframe_distinguishes_intake_from_evolution(self):
        ctx = _ctx()
        _stage(ctx, label="первичный")
        ctx.spec = _spec()
        pkg = json.loads(json.dumps(PACKAGE))
        pkg["nodes"] = [d for d in pkg["nodes"] if d["name"] != "TiO2"]
        _stage(ctx, package=pkg, rationale="TiO2 выведен")
        df = staged_specs_dataframe(ctx.session, only_staged=True)
        assert list(df["вид"]) == ["первичный ввод", "эволюция геометрии"]
        assert df.iloc[1]["−узлы"] == "TiO2"
        assert "⚠️" in df.iloc[1]["хеш"]

    def test_caption_names_first_spec_and_break(self):
        ctx = _ctx(runner=_Runner(names=[d["name"] for d in NODES]),
                   consent=ConsentRegistry())
        sid = _stage(ctx)
        cap = spec_apply_caption(actx.human_apply_spec(ctx, sid))
        assert "впервые" in cap
        # iter74: подпись говорит «проверки неприменимы» (без слова «гейты»).
        assert "проверки неприменимы" in cap

    def test_session_caption_counts_packages(self):
        ctx = _ctx()
        _stage(ctx)
        # iter74: подпись сессии говорит «ждёт применения», а не «в стейдже».
        assert "пакетов спеки ждёт применения: 1" in session_caption(ctx.session)

    def test_dock_panel_renders_and_offers_both_buttons(self, monkeypatch):
        """Панель дока действительно рисуется и предлагает ОБЕ кнопки.

        Заглушка ``st`` (как в iter70) нужна, чтобы проверить факт вызова
        виджетов настоящим кодом дока без запуска Streamlit: панель, которую
        никто не звал, — это мёртвый код, а не интерфейс.
        """
        import src.apps.assistant_dock as dock

        seen = {"buttons": [], "json": 0, "dataframe": 0}

        class _Col:
            def button(self, label, **kw):
                seen["buttons"].append(label)
                return False

        class _Exp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        class _FakeSt:
            def markdown(self, *_a, **_k):
                pass

            def caption(self, *_a, **_k):
                pass

            def warning(self, *_a, **_k):
                pass

            def dataframe(self, *_a, **_k):
                seen["dataframe"] += 1

            def json(self, *_a, **_k):
                seen["json"] += 1

            def expander(self, *_a, **_k):
                return _Exp()

            def columns(self, n):
                return [_Col() for _ in range(n)]

            def text_input(self, *_a, **_k):
                return ""

        ctx = _ctx()
        _stage(ctx, label="кромка ПВХ")
        monkeypatch.setattr(dock, "st", _FakeSt())
        dock._render_spec_packages(ctx, ctx.session)
        assert seen["dataframe"] == 1
        assert seen["json"] == 1                 # JSON пакета доступен человеку
        assert any("Применить" in b for b in seen["buttons"])
        assert any("Отклонить" in b for b in seen["buttons"])

    def test_prompt_forbids_applying_spec_by_model(self):
        """Промпт и роутер обязаны держать apply_spec за человеком."""
        from src.assistant.prompts import (HUMAN_ONLY, architect_system_prompt,
                                            check_routing, route)

        assert "apply_spec" in HUMAN_ONLY and "reject_spec" in HUMAN_ONLY
        text = architect_system_prompt(kinds=AGENT_KINDS)
        assert "propose_spec" in text and "spec_schema" in text
        assert "apply_spec" not in text.split("ИНСТРУМЕНТЫ")[-1]
        r = route("Давай заполним проект: вот состав кромки ПВХ")
        assert r.scenario == "spec_intake"
        assert "propose_spec" in r.tools
        bad = check_routing("spec_intake", ["apply_spec"])
        assert bad["ok"] is False
