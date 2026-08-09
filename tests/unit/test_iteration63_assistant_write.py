# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 63 / ASSISTANT_SPEC — WRITE-инструменты и журнал решений.

Здесь проверяется главный инвариант слоя: **ни один write-инструмент не
меняет состояние сам**. Модель может ПРЕДЛОЖИТЬ патч (класс ``propose`` —
стейдж сессии), а применяет его человек разовым токеном подтверждения
(:mod:`assistant.consent`). Инструменты класса ``write`` в ход модели не
выдаются вообще.

Что именно закрывают тесты:

  * подтверждение ОДНОРАЗОВО, привязано к действию, цели и к отпечатку спеки
    (``spec_hash`` на момент нажатия кнопки) — «подтвердил один патч, применил
    другой» и гонка «геометрию успели поменять» невозможны;
  * гейты применения: невалидный патч, выпадение УЖЕ ИЗМЕРЕННЫХ точек из
    геометрии, ухудшение preflight — каждый блокирует применение с объяснением,
    а патч остаётся в стейдже;
  * применение и ОТКАЗ одинаково попадают в ``decision_log.jsonl`` вместе со
    ``spec_hash``: спор «почему тогда так решили» разрешает журнал;
  * L1-факты цеха добавляет ЧЕЛОВЕК — инструмент требует токен и пишет уровень
    L1 (он отменяет литературу).
"""
import numpy as np
import pytest

from src.assistant import store
from src.assistant.consent import (ACTIONS, ConsentError, ConsentRegistry,
                                    issue_token)
from src.assistant.session import (PATCH_APPLIED, PATCH_REJECTED, PATCH_STAGED,
                                    new_session)
from src.assistant.tools import (AGENT_KINDS, PROPOSE, READONLY, WRITE,
                                  ToolContext, ToolError, dispatch, tool_names)
from src.assistant.tools.registry import dispatcher
from src.assistant.tools.write import (issue_apply_token, issue_fact_token,
                                        issue_reject_token, patch_gates)
from src.assistant.views import (apply_result_caption, consents_dataframe,
                                  decisions_dataframe,
                                  staged_patches_dataframe)
from src.design.phr_sampler import PhrSpec

PROJECT = "pvc_edge_v1"

#: Та же референсная геометрия, что в iter61 (golden iter45/49/50).
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

#: Расширение верха DINP — типовая правка «в цехе льём больше пластификатора».
WIDEN = {"node": "DINP", "field": "range", "value": [4.0, 20.0]}

#: Сужение верха DINP — правка, задним числом выбрасывающая точку с DINP=13.
NARROW = {"node": "DINP", "field": "range", "value": [4.0, 8.0]}

#: Патч, который валидатор спеки обязан отклонить (closure не имеет своей доли).
INVALID = {"node": "CPE", "field": "share_range", "value": [0.1, 0.9]}


def _spec() -> PhrSpec:
    return PhrSpec.from_dicts(NODES)


class _Report:
    """Минимальный duck-typed отчёт preflight (нужны ``passed``/``failures``)."""

    def __init__(self, passed: bool, failures=()):
        self.passed = bool(passed)
        self.failures = list(failures)


class _Runner:
    """Движок кампании в объёме, который нужен write-инструментам.

    Настоящий :class:`MixtureProcessRunner` тянет GP/суррогаты и к этому шагу
    отношения не имеет: проверяются ГЕЙТЫ, а не движок.
    """

    def __init__(self, spec, *, X=None, preflights=None):
        self.phr_spec = spec
        self.X = (np.atleast_2d(np.asarray(X, float))
                  if X is not None else np.empty((0, 0)))
        self._preflights = list(preflights or [])
        self.applied = []

    def set_phr_spec(self, spec):
        self.phr_spec = spec
        self.applied.append(spec.spec_hash())

    def preflight(self, X):
        if not self._preflights:
            return _Report(True)
        return self._preflights.pop(0)


def _fractions(spec, phr):
    return spec.to_fractions(np.asarray(phr, float))


def _ctx(tmp_path, *, spec=None, runner=None, consent=None, session=None):
    spec = spec if spec is not None else _spec()
    return ToolContext(spec=spec, runner=runner,
                       session=session if session is not None
                       else new_session(PROJECT),
                       root=str(tmp_path), project=PROJECT,
                       extra={"consent": consent or ConsentRegistry()})


def _propose(ctx, patch=None, **kw):
    args = {"patch": patch if patch is not None else WIDEN,
            "rationale": "L1: в цехе льют до 20 phr DINP", "level": "L1",
            "bound_type": "CONVENTIONAL", "source": "технолог",
            "confidence": "high"}
    args.update(kw)
    return dispatch(ctx, "propose_patch", args, allowed_kinds=[PROPOSE])


# ======================================================================
# 1. Классы доступа: что вообще может модель
# ======================================================================
class TestAccessClasses:

    def test_write_tools_registered(self):
        names = set(tool_names([WRITE]))
        assert {"apply_patch", "reject_patch", "record_decision",
                "add_local_fact"} <= names

    def test_propose_patch_is_not_write(self):
        assert "propose_patch" in tool_names([PROPOSE])
        assert "propose_patch" not in tool_names([WRITE])

    def test_agent_kinds_exclude_write(self):
        """Модель ходит по readonly+propose+sandbox; write ей не выдаётся."""
        assert WRITE not in AGENT_KINDS
        agent = set(tool_names(AGENT_KINDS))
        assert "propose_patch" in agent and "get_spec" in agent
        assert not ({"apply_patch", "reject_patch", "record_decision",
                     "add_local_fact"} & agent)

    def test_apply_patch_blocked_for_agent(self, tmp_path):
        ctx = _ctx(tmp_path)
        with pytest.raises(ToolError, match="классу 'write'"):
            dispatch(ctx, "apply_patch",
                     {"patch_id": "x", "human_token": "y"},
                     allowed_kinds=AGENT_KINDS)

    def test_readonly_mode_forbids_proposing_too(self, tmp_path):
        """Режим «только чтение» (MCP, iter66) не даёт даже стейджить."""
        ctx = _ctx(tmp_path)
        with pytest.raises(ToolError, match="недоступен"):
            dispatch(ctx, "propose_patch", {"patch": WIDEN, "rationale": "x"},
                     allowed_kinds=[READONLY])


# ======================================================================
# 2. propose_patch — предложение, а не изменение
# ======================================================================
class TestProposePatch:

    def test_stages_patch_without_touching_spec(self, tmp_path):
        ctx = _ctx(tmp_path)
        before = ctx.spec.spec_hash()
        out = _propose(ctx)
        assert out["staged"] and len(out["patch_ids"]) == 1
        assert ctx.spec.spec_hash() == before, "спека проекта изменилась!"
        assert ctx.session.staged_patches(), "патч не попал в стейдж"

    def test_records_before_and_after_values(self, tmp_path):
        ctx = _ctx(tmp_path)
        _propose(ctx)
        p = ctx.session.patches[0]
        assert p.node == "DINP" and p.field_name == "range"
        assert list(p.from_value) == [4.0, 14.0]
        assert list(p.to_value) == [4.0, 20.0]
        assert p.status == PATCH_STAGED

    def test_carries_knowledge_level_and_source(self, tmp_path):
        ctx = _ctx(tmp_path)
        _propose(ctx)
        p = ctx.session.patches[0]
        assert (p.level, p.bound_type, p.source, p.confidence) == \
               ("L1", "CONVENTIONAL", "технолог", "high")
        assert p.rationale

    def test_announces_hash_shift(self, tmp_path):
        ctx = _ctx(tmp_path)
        out = _propose(ctx)
        assert out["affects_hash"] is True
        assert out["spec_hash_before"] != out["spec_hash_after"]
        assert any(d["node"] == "DINP" for d in out["changed_intervals"])

    def test_invalid_patch_is_not_staged(self, tmp_path):
        """Заведомо неприменимый патч не копится в UI — модель правит его сама."""
        ctx = _ctx(tmp_path)
        out = _propose(ctx, patch=INVALID)
        assert out["staged"] is False and out["ok"] is False
        assert out["error"] and out["hint"]
        assert ctx.session.patches == []

    def test_empty_patch_explained(self, tmp_path):
        ctx = _ctx(tmp_path)
        with pytest.raises(ToolError):
            _propose(ctx, patch={"node": "DINP"})

    def test_multi_node_patch_gives_one_entry_per_node(self, tmp_path):
        ctx = _ctx(tmp_path)
        out = _propose(ctx, patch={"DINP": {"range": [4.0, 20.0]},
                                   "TiO2": {"range": [0.3, 9.0]}})
        assert len(out["patch_ids"]) == 2
        assert {p.node for p in ctx.session.patches} == {"DINP", "TiO2"}

    def test_requires_session(self, tmp_path):
        ctx = _ctx(tmp_path)
        ctx.session = None
        with pytest.raises(ToolError, match="Сессия"):
            _propose(ctx)

    def test_table_shows_staged_patch(self, tmp_path):
        ctx = _ctx(tmp_path)
        _propose(ctx)
        df = staged_patches_dataframe(ctx.session, only_staged=True)
        assert len(df) == 1 and df.iloc[0]["узел"] == "DINP"
        assert "меняется" in df.iloc[0]["хеш"]


# ======================================================================
# 3. Подтверждение человека (consent)
# ======================================================================
class TestConsent:

    def test_actions_whitelisted(self):
        reg = ConsentRegistry()
        with pytest.raises(ConsentError, match="Неизвестное действие"):
            reg.issue("rm_rf", "/")
        assert set(ACTIONS) >= {"apply_patch", "reject_patch"}

    def test_token_is_single_use(self):
        reg = ConsentRegistry()
        t = reg.issue("reject_patch", "p1").token
        reg.consume(t, action="reject_patch", target="p1")
        with pytest.raises(ConsentError, match="ОДНОРАЗОВОЕ"):
            reg.consume(t, action="reject_patch", target="p1")

    def test_token_bound_to_target(self):
        reg = ConsentRegistry()
        t = reg.issue("reject_patch", "p1").token
        with pytest.raises(ConsentError, match="ДРУГОЙ объект"):
            reg.consume(t, action="reject_patch", target="p2")

    def test_token_bound_to_action(self):
        reg = ConsentRegistry()
        t = reg.issue("reject_patch", "p1").token
        with pytest.raises(ConsentError, match="на действие"):
            reg.consume(t, action="apply_patch", target="p1")

    def test_token_expires(self):
        clock = {"t": 1000.0}
        reg = ConsentRegistry(ttl_s=60.0, clock=lambda: clock["t"])
        t = reg.issue("reject_patch", "p1").token
        clock["t"] += 61.0
        with pytest.raises(ConsentError, match="истёк"):
            reg.consume(t, action="reject_patch", target="p1")

    def test_unknown_token_rejected(self):
        reg = ConsentRegistry()
        with pytest.raises(ConsentError, match="не найден"):
            reg.consume("сочинённый-токен", action="apply_patch", target="p1")

    def test_empty_token_explains_the_button(self):
        reg = ConsentRegistry()
        with pytest.raises(ConsentError, match="кнопку"):
            reg.consume("", action="apply_patch", target="p1")

    def test_pending_and_revoke(self):
        reg = ConsentRegistry()
        c = reg.issue("apply_patch", "p1")
        assert [x.token for x in reg.pending()] == [c.token]
        assert reg.revoke(c.token) is True
        assert reg.pending() == []

    def test_module_level_issue_token(self):
        assert issue_token("record_decision", "заголовок")

    def test_view_shows_consents(self):
        reg = ConsentRegistry()
        reg.issue("apply_patch", "p1", context_hash="abcdef1234567890")
        df = consents_dataframe(reg.pending())
        assert len(df) == 1 and df.iloc[0]["действие"] == "apply_patch"


# ======================================================================
# 4. apply_patch — применяет ЧЕЛОВЕК
# ======================================================================
class TestApplyPatch:

    def _staged(self, tmp_path, **kw):
        ctx = _ctx(tmp_path, **kw)
        out = _propose(ctx)
        return ctx, out["patch_ids"][0]

    def _apply(self, ctx, pid, token, **kw):
        args = {"patch_id": pid, "human_token": token}
        args.update(kw)
        return dispatch(ctx, "apply_patch", args, allowed_kinds=[WRITE])

    def test_without_token_nothing_changes(self, tmp_path):
        ctx, pid = self._staged(tmp_path)
        before = ctx.spec.spec_hash()
        with pytest.raises(ToolError, match="кнопку"):
            self._apply(ctx, pid, "")
        assert ctx.spec.spec_hash() == before
        assert ctx.session.patch_by_id(pid).status == PATCH_STAGED

    def test_applies_with_token(self, tmp_path):
        ctx, pid = self._staged(tmp_path)
        before = ctx.spec.spec_hash()
        out = self._apply(ctx, pid, issue_apply_token(ctx, pid),
                          note="согласовано с технологом", author="Жихарев")
        assert out["ok"] and out["status"] == PATCH_APPLIED
        assert out["spec_hash_before"] == before
        assert ctx.spec.spec_hash() == out["spec_hash_after"] != before
        assert ctx.spec.phr_intervals()["DINP"] == (4.0, 20.0)
        assert out["warning"] and out["persist_hint"]

    def test_runner_receives_new_spec(self, tmp_path):
        runner = _Runner(_spec())
        ctx, pid = self._staged(tmp_path, runner=runner, spec=runner.phr_spec)
        out = self._apply(ctx, pid, issue_apply_token(ctx, pid))
        assert runner.applied == [out["spec_hash_after"]]
        assert runner.phr_spec.phr_intervals()["DINP"] == (4.0, 20.0)

    def test_second_apply_of_same_patch_refused(self, tmp_path):
        ctx, pid = self._staged(tmp_path)
        self._apply(ctx, pid, issue_apply_token(ctx, pid))
        with pytest.raises(ToolError, match="уже в статусе"):
            self._apply(ctx, pid, issue_apply_token(ctx, pid))

    def test_token_of_another_patch_refused(self, tmp_path):
        ctx, pid = self._staged(tmp_path)
        other = _propose(ctx, patch={"node": "TiO2", "field": "range",
                                     "value": [0.3, 9.0]})["patch_ids"][0]
        with pytest.raises(ToolError, match="ДРУГОЙ объект"):
            self._apply(ctx, pid, issue_apply_token(ctx, other))
        assert ctx.session.patch_by_id(pid).status == PATCH_STAGED

    def test_token_invalid_after_geometry_moved(self, tmp_path):
        """Кнопку нажали при одной спеке, а применяют при другой — отказ."""
        ctx, pid = self._staged(tmp_path)
        token = issue_apply_token(ctx, pid)
        ctx.spec = PhrSpec.from_dicts(
            [dict(d, range=[0.3, 9.0]) if d["name"] == "TiO2" else d
             for d in NODES])
        with pytest.raises(ToolError, match="Геометрия изменилась"):
            self._apply(ctx, pid, token)

    def test_unknown_patch_lists_staged(self, tmp_path):
        ctx, pid = self._staged(tmp_path)
        with pytest.raises(ToolError, match="нет в сессии"):
            self._apply(ctx, "patch_нет", issue_apply_token(ctx, "patch_нет"))

    def test_decision_written_to_log(self, tmp_path):
        ctx, pid = self._staged(tmp_path)
        out = self._apply(ctx, pid, issue_apply_token(ctx, pid),
                          author="технолог")
        recs = store.read_log(str(tmp_path), PROJECT, "decisions")
        assert len(recs) == 1
        rec = recs[0]
        assert rec["kind"] == "apply_patch" and rec["nodes"] == ["DINP"]
        assert rec["spec_hash"] == out["spec_hash_before"]
        assert rec["spec_hash_after"] == out["spec_hash_after"]
        assert rec["author"] == "технолог" and rec["level"] == "L1"
        assert not decisions_dataframe(recs).empty

    def test_caption_names_hash_shift(self, tmp_path):
        ctx, pid = self._staged(tmp_path)
        out = self._apply(ctx, pid, issue_apply_token(ctx, pid))
        cap = apply_result_caption(out)
        assert "отпечаток" in cap and "→" in cap


# ======================================================================
# 5. Гейты применения
# ======================================================================
class TestGates:

    def test_existing_point_falling_out_blocks_apply(self, tmp_path):
        """Сужение границы задним числом выбрасывает измеренную точку."""
        spec = _spec()
        phr = [100.0, 13.0, 2.5, 7.0, 8.0, 1.0, 0.10]   # DINP = 13
        runner = _Runner(spec, X=[_fractions(spec, phr)])
        ctx = _ctx(tmp_path, spec=spec, runner=runner)
        pid = _propose(ctx, patch=NARROW,
                       rationale="сузим верх")["patch_ids"][0]
        with pytest.raises(ToolError, match="ГЕЙТ ПРИМЕНЕНИЯ"):
            dispatch(ctx, "apply_patch",
                     {"patch_id": pid, "human_token": issue_apply_token(ctx, pid)},
                     allowed_kinds=[WRITE])
        assert ctx.session.patch_by_id(pid).status == PATCH_STAGED, \
            "патч должен остаться в стейдже"
        assert runner.phr_spec.phr_intervals()["DINP"] == (4.0, 14.0)

    def test_point_inside_new_geometry_does_not_block(self, tmp_path):
        spec = _spec()
        phr = [100.0, 6.0, 2.5, 7.0, 8.0, 1.0, 0.10]    # DINP = 6 — влезает
        runner = _Runner(spec, X=[_fractions(spec, phr)])
        ctx = _ctx(tmp_path, spec=spec, runner=runner)
        pid = _propose(ctx, patch=NARROW, rationale="сузим верх")["patch_ids"][0]
        out = dispatch(ctx, "apply_patch",
                       {"patch_id": pid,
                        "human_token": issue_apply_token(ctx, pid)},
                       allowed_kinds=[WRITE])
        assert out["ok"] and out["gates"]["points"]["n_lost"] == 0

    def test_preflight_degradation_blocks_apply(self, tmp_path):
        spec = _spec()
        runner = _Runner(spec, preflights=[_Report(True),
                                           _Report(False, ["max|corr| 0.99"])])
        ctx = _ctx(tmp_path, spec=spec, runner=runner)
        pid = _propose(ctx)["patch_ids"][0]
        with pytest.raises(ToolError, match="preflight"):
            dispatch(ctx, "apply_patch",
                     {"patch_id": pid, "human_token": issue_apply_token(ctx, pid)},
                     allowed_kinds=[WRITE])

    def test_preflight_red_before_does_not_block(self, tmp_path):
        """Если гейты были красными и до патча — правки не запрещаем."""
        spec = _spec()
        runner = _Runner(spec, preflights=[_Report(False, ["rank"]),
                                           _Report(False, ["rank"])])
        ctx = _ctx(tmp_path, spec=spec, runner=runner)
        pid = _propose(ctx)["patch_ids"][0]
        out = dispatch(ctx, "apply_patch",
                       {"patch_id": pid,
                        "human_token": issue_apply_token(ctx, pid)},
                       allowed_kinds=[WRITE])
        assert out["ok"] and out["gates"]["preflight"]["ok"]

    def test_gates_report_says_what_was_not_checked(self, tmp_path):
        """Без проекта гейты не «пройдены» — они НЕ ПРОВЕРЕНЫ, и это видно."""
        ctx = _ctx(tmp_path)
        gates = patch_gates(ctx, ctx.spec, PhrSpec.from_dicts(NODES))
        assert gates["ok"] is True
        assert gates["points"]["checked"] is False
        assert gates["preflight"]["checked"] is False
        assert gates["points"]["reason"] and gates["preflight"]["reason"]


# ======================================================================
# 6. Отклонение, решения, L1-факты
# ======================================================================
class TestRejectAndJournals:

    def test_reject_requires_token_and_logs(self, tmp_path):
        ctx = _ctx(tmp_path)
        pid = _propose(ctx)["patch_ids"][0]
        with pytest.raises(ToolError):
            dispatch(ctx, "reject_patch",
                     {"patch_id": pid, "human_token": "", "reason": "нет"},
                     allowed_kinds=[WRITE])
        out = dispatch(ctx, "reject_patch",
                       {"patch_id": pid,
                        "human_token": issue_reject_token(ctx, pid),
                        "reason": "паспорт не подтверждает 20 phr"},
                       allowed_kinds=[WRITE])
        assert out["status"] == PATCH_REJECTED
        assert ctx.session.patch_by_id(pid).status == PATCH_REJECTED
        recs = store.read_log(str(tmp_path), PROJECT, "decisions")
        assert recs[-1]["kind"] == "reject_patch"
        assert "ОТКЛОНЕНО" in recs[-1]["title"]

    def test_rejected_patch_cannot_be_applied(self, tmp_path):
        ctx = _ctx(tmp_path)
        pid = _propose(ctx)["patch_ids"][0]
        dispatch(ctx, "reject_patch",
                 {"patch_id": pid, "human_token": issue_reject_token(ctx, pid),
                  "reason": "нет"}, allowed_kinds=[WRITE])
        with pytest.raises(ToolError, match="уже в статусе"):
            dispatch(ctx, "apply_patch",
                     {"patch_id": pid,
                      "human_token": issue_apply_token(ctx, pid)},
                     allowed_kinds=[WRITE])

    def test_record_decision_writes_adr_with_spec_hash(self, tmp_path):
        ctx = _ctx(tmp_path)
        title = "LUB (k=3): все члены SHARE_SIMPLEX"
        reg = ctx.extra["consent"]
        out = dispatch(ctx, "record_decision",
                       {"title": title, "rationale": "closure при k≥3 запрещён",
                        "nodes": ["OPE"], "author": "технолог",
                        "human_token": reg.issue("record_decision",
                                                 title).token},
                       allowed_kinds=[WRITE])
        assert out["ok"] and out["decision"]["persisted"]
        rec = store.read_log(str(tmp_path), PROJECT, "decisions")[-1]
        assert rec["title"] == title
        assert rec["spec_hash"] == ctx.spec.spec_hash()

    def test_local_fact_requires_human_and_is_l1(self, tmp_path):
        ctx = _ctx(tmp_path)
        st = "Плотность компаунда ИЗМЕРЯЕТСЯ, не считается из компонентов"
        with pytest.raises(ToolError):
            dispatch(ctx, "add_local_fact",
                     {"statement": st, "human_token": ""},
                     allowed_kinds=[WRITE])
        out = dispatch(ctx, "add_local_fact",
                       {"statement": st, "scope": "cost",
                        "human_token": issue_fact_token(ctx, st)},
                       allowed_kinds=[WRITE])
        assert out["ok"]
        rec = store.read_log(str(tmp_path), PROJECT, "local_facts")[-1]
        assert rec["level"] == "L1" and rec["statement"] == st
        got = dispatch(ctx, "get_local_facts", {"scope": "cost"})
        assert got["n"] == 1

    def test_log_without_project_is_not_silent(self):
        """Нет проекта — запись не потеряна молча, а помечена как несохранённая."""
        ctx = ToolContext(spec=_spec(), session=new_session(PROJECT),
                          extra={"consent": ConsentRegistry()})
        title = "решение без проекта"
        out = dispatch(ctx, "record_decision",
                       {"title": title, "rationale": "r",
                        "human_token": ctx.extra["consent"].issue(
                            "record_decision", title).token},
                       allowed_kinds=[WRITE])
        assert out["decision"]["persisted"] is False
        assert out["decision"]["note"]


# ======================================================================
# 7. Аудит и стыковка с ходом модели
# ======================================================================
class TestAuditAndLoop:

    def test_audit_records_propose_and_refusal(self, tmp_path):
        ctx = _ctx(tmp_path)
        audit = []
        call = dispatcher(ctx, allowed_kinds=AGENT_KINDS, on_call=audit.append)
        call("propose_patch", {"patch": WIDEN, "rationale": "r"})
        with pytest.raises(ToolError):
            call("apply_patch", {"patch_id": "x", "human_token": "y"})
        assert [a["tool"] for a in audit] == ["propose_patch", "apply_patch"]
        assert audit[0]["ok"] is True and audit[1]["ok"] is False
        assert "write" in audit[1]["error"]

    def test_full_cycle_survives_session_round_trip(self, tmp_path):
        """Предложили → сохранили проект → открыли → применили."""
        ctx = _ctx(tmp_path)
        pid = _propose(ctx)["patch_ids"][0]
        store.save_session(ctx.session, str(tmp_path))
        ctx.session = store.load_session(str(tmp_path), PROJECT)
        assert [p.id for p in ctx.session.staged_patches()] == [pid]
        out = dispatch(ctx, "apply_patch",
                       {"patch_id": pid,
                        "human_token": issue_apply_token(ctx, pid)},
                       allowed_kinds=[WRITE])
        assert out["ok"]
        store.save_session(ctx.session, str(tmp_path))
        again = store.load_session(str(tmp_path), PROJECT)
        assert again.patch_by_id(pid).status == PATCH_APPLIED
        assert again.staged_patches() == []
