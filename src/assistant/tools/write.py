"""assistant/tools/write.py — предложение и ПРИМЕНЕНИЕ изменений (iter63).

Read-only инструменты (iter61) отвечают «что сейчас», песочница (iter62) —
«проверено ли это». Здесь закрывается последний участок работы архитектора:
ЗАФИКСИРОВАТЬ решение — поправить геометрию спеки и записать, почему компания
так решила.

Разделение ответственности жёсткое (ASSISTANT_SPEC §2):

* :func:`propose_patch` (класс ``propose``) — модель кладёт патч в СТЕЙДЖ
  сессии. Спека проекта не меняется; патч, не прошедший dry-run валидации, в
  стейдж НЕ попадает (иначе в UI копились бы заведомо неприменимые пункты);
* :func:`apply_patch` / :func:`reject_patch` / :func:`record_decision` /
  :func:`add_local_fact` (класс ``write``) — исполняются ТОЛЬКО с разовым
  токеном человека (:mod:`assistant.consent`) и в ход модели не выдаются
  вовсе (``AGENT_KINDS``).

Гейты применения (патч, проваливший любой из них, не применяется):

1. **валидация конструктора** — патч должен давать корректную спеку;
2. **уже собранные точки** — если после патча существующие опыты выпадают из
   геометрии, «сужение границы» молча обесценило бы часть базы;
3. **preflight** — если на розыгрыше по новой спеке относительные гейты
   ломаются там, где на старой проходили, патч вводит вырожденность
   (продублированная ось, доля, линейно зависимая от тотала).

Каждое применение и каждый отказ пишутся в ``decision_log.jsonl``: спор «почему
мы тогда так решили» должен разрешаться журналом, а не памятью участников.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..consent import DEFAULT_REGISTRY, ConsentError, ConsentRegistry
from ..session import PATCH_APPLIED, PATCH_REJECTED, StagedPatch
from ..store import append_log
from .readonly import (_f, build_patched_spec, normalize_patch, spec_payload,
                       validate_spec)
from .registry import PROPOSE, WRITE, ToolContext, ToolError, register

#: Сколько кандидатов разыгрывать в preflight-гейте применения.
GATE_SIM_N = 96


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def registry_for(ctx: ToolContext) -> ConsentRegistry:
    """Реестр подтверждений из контекста (UI кладёт свой) или общий."""
    reg = (ctx.extra or {}).get("consent")
    return reg if isinstance(reg, ConsentRegistry) else DEFAULT_REGISTRY


def _consume(ctx: ToolContext, human_token: str, *, action: str,
             target: str = "", context_hash: str = ""):
    """Погасить токен, переведя отказ согласия в отказ ИНСТРУМЕНТА.

    Модель должна получить причину как результат вызова (A0.6), а не увидеть
    исключение чужого слоя.
    """
    token = str(human_token or "") or str(ctx.human_token or "")
    try:
        return registry_for(ctx).consume(token, action=action, target=target,
                                         context_hash=context_hash)
    except ConsentError as exc:
        raise ToolError(str(exc)) from exc


def _log(ctx: ToolContext, kind: str, record: Dict[str, Any]) -> Dict[str, Any]:
    """Дописать запись в журнал проекта (если проект известен)."""
    if not (ctx.root and ctx.project):
        return {**record, "persisted": False,
                "note": "проект не указан — запись не сохранена на диск"}
    append_log(ctx.root, ctx.project, kind, record)
    return {**record, "persisted": True}


# ----------------------------------------------------------------------
# Патч: разбор и представление
# ----------------------------------------------------------------------
def _field_names(op: Dict[str, Any]) -> str:
    """Человекочитаемое имя поля(ей) одной операции патча."""
    names = list(op.get("set", {}) or {}) + [f"-{u}" for u in op.get("unset", [])]
    return ", ".join(names)


def _current_values(spec, node: str, op: Dict[str, Any]) -> Any:
    """Что стоит в узле СЕЙЧАС по затрагиваемым полям (колонка «было»).

    Без этого таблица патчей в UI показывала бы только «стало», и решение
    принималось бы вслепую.
    """
    nodes, meta = spec_payload(spec)
    src = meta if node in ("__spec__", "spec", "*") else next(
        (d for d in nodes if str(d.get("name")) == node), {})
    keys = list(op.get("set", {}) or {}) + list(op.get("unset", []) or [])
    if len(keys) == 1:
        return src.get(keys[0])
    return {k: src.get(k) for k in keys}


def _to_values(op: Dict[str, Any]) -> Any:
    set_ = dict(op.get("set", {}) or {})
    unset = list(op.get("unset", []) or [])
    if len(set_) == 1 and not unset:
        return list(set_.values())[0]
    out: Dict[str, Any] = dict(set_)
    for u in unset:
        out[str(u)] = None
    return out


@register(
    "propose_patch",
    description=(
        "ПРЕДЛОЖИТЬ правку phr-спеки: патч проходит сухую валидацию и кладётся "
        "в стейдж сессии — применит его ЧЕЛОВЕК кнопкой в интерфейсе. Спека "
        "проекта этим вызовом НЕ меняется. Обязательно объясни изменение: "
        "тип границы (PHYSICAL — закон природы/паспорт, CONVENTIONAL — "
        "договорённость цеха), уровень знания (L1 факт технолога > L2 "
        "литература > L3 проверяемое расчётом) и источник. Патч, не прошедший "
        "валидацию, в стейдж не попадает — исправь и предложи снова."),
    parameters={"type": "object", "properties": {
        "patch": {"type": "object",
                  "description": "патч как в validate_spec: {'node','field',"
                                 "'value'} либо {'node','set','unset'}"},
        "rationale": {"type": "string",
                      "description": "почему так: физика/паспорт/практика цеха"},
        "bound_type": {"type": "string",
                       "description": "PHYSICAL | CONVENTIONAL"},
        "level": {"type": "string", "description": "L1 | L2 | L3"},
        "source": {"type": "string",
                   "description": "источник: паспорт, ГОСТ, слова технолога, "
                                  "расчёт инструмента"},
        "confidence": {"type": "string", "description": "high | med | low"}},
        "required": ["patch", "rationale"]},
    kind=PROPOSE)
def propose_patch(ctx: ToolContext, patch: Any, rationale: str,
                  bound_type: str = "", level: str = "", source: str = "",
                  confidence: str = "") -> Dict[str, Any]:
    spec = ctx.require_spec()
    session = ctx.require_session()
    ops = normalize_patch(patch)                    # ToolError на кривой форме
    if not ops:
        raise ToolError("Патч пуст: нечего предлагать.")

    check = validate_spec(ctx, patch)               # dry-run на КОПИИ спеки
    if not check.get("ok"):
        return {"staged": False, "ok": False, "error": check.get("error", ""),
                "hint": ("Патч отклонён валидацией и в стейдж НЕ положен: "
                         "предлагать заведомо неприменимое изменение — значит "
                         "перекладывать разбор на человека. Исправь патч.")}

    ids: List[str] = []
    for op in ops:
        p = StagedPatch(
            node=op["node"], field_name=_field_names(op),
            from_value=_f(_current_values(spec, op["node"], op)),
            to_value=_f(_to_values(op)),
            bound_type=str(bound_type or ""), level=str(level or ""),
            source=str(source or ""), rationale=str(rationale or ""),
            confidence=str(confidence or ""),
            affects_hash=bool(check.get("affects_hash")),
            raw={"ops": [op], "spec_hash_before": check["spec_hash_before"],
                 "spec_hash_after": check["spec_hash_after"],
                 "changed_intervals": check["changed_intervals"]})
        session.stage_patch(p)
        ids.append(p.id)

    return {"staged": True, "ok": True, "patch_ids": ids,
            "affects_hash": bool(check.get("affects_hash")),
            "spec_hash_before": check["spec_hash_before"],
            "spec_hash_after": check["spec_hash_after"],
            "changed_intervals": check["changed_intervals"],
            "note": ("Патч НЕ применён: он в стейдже. Применяет человек кнопкой "
                     "(разовый токен подтверждения). Скажи пользователю, что "
                     "именно поедет и что изменится в отпечатке спеки."),
            "warning": check.get("warning", "")}


# ----------------------------------------------------------------------
# Гейты применения
# ----------------------------------------------------------------------
def _existing_points_gate(ctx: ToolContext, spec, patched) -> Dict[str, Any]:
    """Останутся ли УЖЕ ИЗМЕРЕННЫЕ точки внутри новой геометрии.

    Сужение границы задним числом обесценивает часть базы: точки останутся в
    таблице, но перестанут принадлежать области, по которой строится модель.
    Такое решение может быть осознанным — но не молчаливым.
    """
    runner = ctx.runner
    if runner is None:
        return {"checked": False, "reason": "проект не собран — точек нет"}
    X = np.atleast_2d(np.asarray(getattr(runner, "X", np.empty((0, 0))), float))
    if X.size == 0 or X.shape[0] == 0:
        return {"checked": False, "reason": "база точек пуста"}
    q = int(spec.q)
    if X.shape[1] < q:
        return {"checked": False,
                "reason": "координаты точек не совпадают с составом спеки"}

    lost: List[Dict[str, Any]] = []
    checked = 0
    for i, row in enumerate(X[:, :q]):
        try:
            phr = spec.fractions_to_phr(row)
        except Exception:                            # noqa: BLE001
            continue                                 # точка иной фазы/схемы
        checked += 1
        rep = patched.point_report(phr)
        if not rep.ok:
            lost.append({"index": i + 1, "violations": list(rep.violations)[:3]})
    return {"checked": bool(checked), "n_checked": checked,
            "n_lost": len(lost), "lost": lost[:5],
            "ok": not lost,
            "reason": "" if checked else "ни одна точка не сопоставлена спеке"}


def _preflight_gate(ctx: ToolContext, spec, patched, *, n: int = GATE_SIM_N
                    ) -> Dict[str, Any]:
    """Не вводит ли патч вырожденность геометрии (относительные гейты).

    Сравниваем розыгрыш ДО и ПОСЛЕ: блокируем, только если гейты проходили, а
    после патча перестали. Абсолютная краснота бывает и до патча (мало точек,
    узкая область) — блокировать по ней значило бы запретить любые правки.
    """
    runner = ctx.runner
    if runner is None or not hasattr(runner, "preflight"):
        return {"checked": False, "reason": "проект не собран — preflight "
                                            "не с чем сравнивать"}
    try:
        before = runner.preflight(_candidates(spec, n))
        after = runner.preflight(_candidates(patched, n))
    except Exception as exc:                         # noqa: BLE001
        return {"checked": False,
                "reason": f"preflight не выполнился: {type(exc).__name__}: {exc}"}
    b_ok = bool(getattr(before, "passed", True))
    a_ok = bool(getattr(after, "passed", True))
    return {"checked": True, "passed_before": b_ok, "passed_after": a_ok,
            "ok": not (b_ok and not a_ok),
            "failures_after": list(getattr(after, "failures", []) or [])}


def _candidates(spec, n: int) -> np.ndarray:
    """Матрица долей по спеке — вход preflight (mixture-часть)."""
    P = np.atleast_2d(np.asarray(spec.decode(spec.sample_z(int(n), seed=0)),
                                 float))
    return np.atleast_2d(np.asarray([spec.to_fractions(row) for row in P],
                                    float))


def patch_gates(ctx: ToolContext, spec, patched) -> Dict[str, Any]:
    """Все гейты применения разом + итоговый вердикт с причинами."""
    points = _existing_points_gate(ctx, spec, patched)
    pre = _preflight_gate(ctx, spec, patched)
    blocked: List[str] = []
    if points.get("checked") and not points.get("ok"):
        blocked.append(
            f"после патча {points['n_lost']} из {points['n_checked']} уже "
            f"измеренных точек выходят за геометрию "
            f"(например #{points['lost'][0]['index']}: "
            f"{points['lost'][0]['violations']})")
    if pre.get("checked") and not pre.get("ok"):
        blocked.append("preflight на розыгрыше по новой спеке проваливает "
                       f"гейты, которые проходили до патча: "
                       f"{pre.get('failures_after')}")
    return {"ok": not blocked, "blocked": blocked,
            "points": points, "preflight": pre}


# ----------------------------------------------------------------------
# apply_patch
# ----------------------------------------------------------------------
@register(
    "apply_patch",
    description=(
        "ПРИМЕНИТЬ патч из стейджа к спеке проекта. Требует разового токена "
        "подтверждения человека (выдаётся кнопкой в интерфейсе). Блокируется, "
        "если патч не проходит валидацию, выбрасывает уже измеренные точки из "
        "геометрии или ломает preflight-гейты, которые до патча проходили. "
        "Применение записывается в журнал решений компании."),
    parameters={"type": "object", "properties": {
        "patch_id": {"type": "string", "description": "id патча из стейджа"},
        "human_token": {"type": "string",
                        "description": "разовый токен подтверждения человека"},
        "note": {"type": "string", "description": "комментарий к применению"},
        "author": {"type": "string", "description": "кто принял решение"}},
        "required": ["patch_id", "human_token"]},
    kind=WRITE, long_running=True)
def apply_patch(ctx: ToolContext, patch_id: str, human_token: str,
                note: str = "", author: str = "") -> Dict[str, Any]:
    session = ctx.require_session()
    spec = ctx.require_spec()
    patch = session.patch_by_id(str(patch_id))
    if patch is None:
        raise ToolError(
            f"Патча '{patch_id}' нет в сессии. В стейдже: "
            f"{[p.id for p in session.staged_patches()] or 'пусто'}.")
    if patch.status != "staged":
        raise ToolError(
            f"Патч '{patch_id}' уже в статусе '{patch.status}': повторное "
            f"применение запрещено — предложите новый патч.")

    hash_before = spec.spec_hash()
    _consume(ctx, human_token, action="apply_patch", target=str(patch_id),
             context_hash=hash_before)

    ops = list((patch.raw or {}).get("ops") or [])
    if not ops:
        raise ToolError(f"Патч '{patch_id}' не содержит операций (raw.ops): "
                        f"применять нечего.")
    try:
        patched = build_patched_spec(spec, ops)
    except ToolError:
        raise
    except Exception as exc:                          # noqa: BLE001
        raise ToolError(
            f"ГЕЙТ ВАЛИДАЦИИ: патч не даёт корректной спеки "
            f"({type(exc).__name__}: {exc}). Изменение НЕ применено, геометрия "
            f"прежняя.") from exc

    gates = patch_gates(ctx, spec, patched)
    if not gates["ok"]:
        raise ToolError(
            "ГЕЙТ ПРИМЕНЕНИЯ: изменение НЕ применено. " +
            "; ".join(gates["blocked"]) +
            ". Патч остаётся в стейдже: смягчите правку или обсудите "
            "последствия с технологом.")

    runner = ctx.runner
    if runner is not None and hasattr(runner, "set_phr_spec"):
        try:
            runner.set_phr_spec(patched)
        except Exception as exc:                      # noqa: BLE001
            raise ToolError(
                f"Спека не принята проектом ({type(exc).__name__}: {exc}). "
                f"Скорее всего патч меняет СОСТАВ компонентов — это эволюция "
                f"схемы кампании, а не правка границы.") from exc
    if ctx.spec is not None:
        ctx.spec = patched

    hash_after = patched.spec_hash()
    session.set_patch_status(patch.id, PATCH_APPLIED, reason=str(note or ""))

    decision = _log(ctx, "decisions", {
        "ts": _now(),
        "title": f"{patch.node}.{patch.field_name}: {patch.from_value} → "
                 f"{patch.to_value}",
        "nodes": [patch.node], "author": str(author or "человек"),
        "spec_hash": hash_before, "spec_hash_after": hash_after,
        "rationale": patch.rationale, "bound_type": patch.bound_type,
        "level": patch.level, "source": patch.source,
        "confidence": patch.confidence, "patch_id": patch.id,
        "affects_hash": hash_before != hash_after,
        "note": str(note or ""), "kind": "apply_patch",
        "gates": {"points": gates["points"].get("ok", None),
                  "preflight": gates["preflight"].get("ok", None)}})

    return _f({
        "ok": True, "patch_id": patch.id, "status": PATCH_APPLIED,
        "spec_hash_before": hash_before, "spec_hash_after": hash_after,
        "affects_hash": hash_before != hash_after,
        "changed_intervals": (patch.raw or {}).get("changed_intervals", []),
        "gates": gates, "decision": decision,
        "warning": ("spec_hash изменился ⇒ ранее собранные точки относятся к "
                    "ПРЕЖНЕЙ геометрии: план дальше строится в новой области."
                    if hash_before != hash_after else ""),
        "persist_hint": ("Спека изменена в памяти проекта — сохраните кампанию, "
                         "чтобы правка пережила перезапуск."),
    })


@register(
    "reject_patch",
    description=(
        "Отклонить патч из стейджа (решение человека, требует токен "
        "подтверждения). Отказ ТОЖЕ записывается в журнал решений: через "
        "полгода спор «почему не расширили границу» разрешает журнал, а не "
        "память участников."),
    parameters={"type": "object", "properties": {
        "patch_id": {"type": "string", "description": "id патча из стейджа"},
        "human_token": {"type": "string", "description": "разовый токен"},
        "reason": {"type": "string", "description": "почему отклонён"},
        "author": {"type": "string", "description": "кто решил"}},
        "required": ["patch_id", "human_token", "reason"]},
    kind=WRITE)
def reject_patch(ctx: ToolContext, patch_id: str, human_token: str,
                 reason: str, author: str = "") -> Dict[str, Any]:
    session = ctx.require_session()
    patch = session.patch_by_id(str(patch_id))
    if patch is None:
        raise ToolError(f"Патча '{patch_id}' нет в сессии.")
    _consume(ctx, human_token, action="reject_patch", target=str(patch_id))
    session.set_patch_status(patch.id, PATCH_REJECTED, reason=str(reason))
    decision = _log(ctx, "decisions", {
        "ts": _now(),
        "title": f"ОТКЛОНЕНО: {patch.node}.{patch.field_name} → {patch.to_value}",
        "nodes": [patch.node], "author": str(author or "человек"),
        "spec_hash": (patch.raw or {}).get("spec_hash_before", ""),
        "rationale": str(reason), "patch_id": patch.id, "kind": "reject_patch"})
    return {"ok": True, "patch_id": patch.id, "status": PATCH_REJECTED,
            "decision": decision}


@register(
    "record_decision",
    description=(
        "Записать РЕШЕНИЕ КОМПАНИИ (ADR) в журнал проекта: что решили, почему, "
        "какие узлы затронуты. Требует токен подтверждения человека — журнал "
        "фиксирует решения людей, а не выводы модели."),
    parameters={"type": "object", "properties": {
        "title": {"type": "string", "description": "суть решения одной строкой"},
        "rationale": {"type": "string", "description": "обоснование"},
        "nodes": {"type": "array", "items": {"type": "string"},
                  "description": "затронутые узлы спеки"},
        "author": {"type": "string", "description": "кто решил"},
        "human_token": {"type": "string", "description": "разовый токен"}},
        "required": ["title", "rationale", "human_token"]},
    kind=WRITE)
def record_decision(ctx: ToolContext, title: str, rationale: str,
                    human_token: str, nodes: Optional[Sequence[str]] = None,
                    author: str = "") -> Dict[str, Any]:
    _consume(ctx, human_token, action="record_decision", target=str(title))
    spec_hash = ""
    try:
        spec_hash = ctx.require_spec().spec_hash()
    except ToolError:
        pass                                  # решение может быть и до спеки
    rec = _log(ctx, "decisions", {
        "ts": _now(), "title": str(title), "rationale": str(rationale),
        "nodes": [str(n) for n in (nodes or [])],
        "author": str(author or "человек"), "spec_hash": spec_hash,
        "kind": "decision"})
    return {"ok": True, "decision": rec}


@register(
    "add_local_fact",
    description=(
        "Записать L1-факт цеха (высший приоритет знания, отменяет литературу). "
        "Требует токен подтверждения: факты добавляет ЧЕЛОВЕК — ты можешь "
        "только предложить формулировку и попросить подтвердить."),
    parameters={"type": "object", "properties": {
        "statement": {"type": "string", "description": "сам факт одной фразой"},
        "scope": {"type": "string",
                  "description": "к чему относится: узел, свойство, процесс"},
        "source": {"type": "string", "description": "откуда: кто/протокол/опыт"},
        "author": {"type": "string", "description": "кто утверждает"},
        "human_token": {"type": "string", "description": "разовый токен"}},
        "required": ["statement", "human_token"]},
    kind=WRITE)
def add_local_fact(ctx: ToolContext, statement: str, human_token: str,
                   scope: str = "", source: str = "",
                   author: str = "") -> Dict[str, Any]:
    _consume(ctx, human_token, action="add_local_fact", target=str(statement))
    rec = _log(ctx, "local_facts", {
        "ts": _now(), "statement": str(statement), "scope": str(scope or ""),
        "source": str(source or ""), "author": str(author or "технолог"),
        "level": "L1"})
    return {"ok": True, "fact": rec,
            "note": "L1 отменяет L2/L3: при конфликте с литературой выноси "
                    "расхождение в OPEN_QUESTIONS, не усредняй."}


# ----------------------------------------------------------------------
# Хелперы для UI (кнопки подтверждения)
# ----------------------------------------------------------------------
def issue_apply_token(ctx: ToolContext, patch_id: str, *,
                      ttl_s: Optional[float] = None, note: str = "") -> str:
    """Токен на применение патча (кнопка «Применить» в доке).

    Привязан к отпечатку спеки: если до нажатия «Применить» геометрия успела
    измениться, токен не сработает — человек подтверждал другое состояние.
    """
    spec_hash = ""
    try:
        spec_hash = ctx.require_spec().spec_hash()
    except ToolError:
        pass
    return registry_for(ctx).issue("apply_patch", str(patch_id),
                                   context_hash=spec_hash, ttl_s=ttl_s,
                                   note=note).token


def issue_reject_token(ctx: ToolContext, patch_id: str, *,
                       ttl_s: Optional[float] = None) -> str:
    """Токен на отклонение патча (кнопка «Отклонить»)."""
    return registry_for(ctx).issue("reject_patch", str(patch_id),
                                   ttl_s=ttl_s).token


def issue_decision_token(ctx: ToolContext, title: str, *,
                         ttl_s: Optional[float] = None) -> str:
    """Токен на запись решения компании (кнопка «Зафиксировать решение»)."""
    return registry_for(ctx).issue("record_decision", str(title),
                                   ttl_s=ttl_s).token


def issue_fact_token(ctx: ToolContext, statement: str, *,
                     ttl_s: Optional[float] = None) -> str:
    """Токен на запись L1-факта (кнопка «Добавить факт цеха»)."""
    return registry_for(ctx).issue("add_local_fact", str(statement),
                                   ttl_s=ttl_s).token
