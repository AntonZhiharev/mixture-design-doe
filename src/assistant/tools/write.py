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
from ..session import (PATCH_APPLIED, PATCH_REJECTED, StagedPatch,
                       StagedProject, StagedSetup, StagedSpec)
from ..store import append_log
from .readonly import (_f, active_spec, build_patched_spec,
                       build_spec_from_package, has_project, normalize_patch,
                       normalize_spec_package, spec_package_diff, spec_payload,
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
# propose_spec — ПАКЕТ спеки целиком (iter71)
# ----------------------------------------------------------------------
@register(
    "propose_spec",
    description=(
        "ПРЕДЛОЖИТЬ phr-спеку ПАКЕТОМ целиком: первичный ввод геометрии (когда "
        "спеки в проекте ещё нет) и её ЭВОЛЮЦИЯ — добавить узел, удалить узел, "
        "сменить роль, перестроить группы. Патчем (propose_patch) такое "
        "невозможно: он правит поля СУЩЕСТВУЮЩИХ узлов. Пакет валидируется "
        "ядром и кладётся в стейдж сессии; спека проекта этим вызовом НЕ "
        "меняется — применяет человек кнопкой в интерфейсе. Формат пакета "
        "возьми из spec_schema, не из памяти. Обязательно объясни, что "
        "меняется и почему (уровень знания L1|L2|L3 и источник)."),
    parameters={"type": "object", "properties": {
        "package": {"type": "object",
                    "description": "спека ЦЕЛИКОМ: {'spec_version': 2, "
                                   "'nodes': [...], 'group_order': [...]}"},
        "rationale": {"type": "string",
                      "description": "почему такая геометрия: физика, паспорт, "
                                     "практика цеха, расчёт"},
        "label": {"type": "string",
                  "description": "короткая метка пакета (например "
                                 "«кромка ПВХ: первичный ввод»)"},
        "level": {"type": "string", "description": "L1 | L2 | L3"},
        "source": {"type": "string", "description": "источник сведений"},
        "confidence": {"type": "string", "description": "high | med | low"}},
        "required": ["package", "rationale"]},
    kind=PROPOSE)
def propose_spec(ctx: ToolContext, package: Any, rationale: str,
                 label: str = "", level: str = "", source: str = "",
                 confidence: str = "") -> Dict[str, Any]:
    """Пакет спеки → стейдж сессии (после валидации ЯДРОМ).

    Отказ валидации возвращается РЕЗУЛЬТАТОМ, а в стейдж не попадает: пункт,
    заведомо неприменимый, перекладывал бы разбор на человека — тот нажал бы
    «Применить» и получил ошибку вместо решения.
    """
    session = ctx.require_session()
    spec = active_spec(ctx)                  # может отсутствовать: первичный ввод
    nodes, order, version = normalize_spec_package(package)

    try:
        candidate = build_spec_from_package(package)
    except ToolError:
        raise
    except Exception as exc:                          # noqa: BLE001
        return {"staged": False, "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "hint": ("Пакет отклонён валидатором ядра и в стейдж НЕ "
                         "положен: геометрия проекта не тронута. Сверь ключи и "
                         "инварианты со spec_schema и предложи заново.")}

    diff = spec_package_diff(spec, candidate)
    staged = StagedSpec(
        nodes=nodes, group_order=order, spec_version=version,
        label=str(label or ""), rationale=str(rationale or ""),
        level=str(level or ""), source=str(source or ""),
        confidence=str(confidence or ""),
        summary={**diff, "nodes_total": len(nodes)})
    session.stage_spec(staged)

    out: Dict[str, Any] = {
        "staged": True, "ok": True, "spec_id": staged.id,
        "diff": diff, "nodes_total": len(nodes),
        "note": ("Пакет НЕ применён: он в стейдже. Применяет человек кнопкой "
                 "(разовый токен подтверждения). Скажи пользователю, что "
                 "именно поедет: состав, роли, отпечаток спеки."),
    }
    if diff.get("first_spec"):
        out["warning"] = ("Первичный ввод геометрии: после применения "
                          f"spec_hash = {diff['spec_hash_after'][:12]}… — "
                          f"именно к нему будут отнесены все дальнейшие точки.")
    elif diff.get("removed") or diff.get("components_removed"):
        out["warning"] = (
            f"Пакет УДАЛЯЕТ узлы {diff.get('removed')} (компоненты: "
            f"{diff.get('components_removed')}). Уже собранные точки "
            f"относятся к прежней геометрии — назови это вслух.")
    elif diff.get("affects_hash"):
        out["warning"] = ("spec_hash меняется ⇒ другая геометрия кампании: "
                          "ранее собранные точки относятся к прежнему "
                          "отпечатку.")
    return _f(out)


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


# ----------------------------------------------------------------------
# apply_spec / reject_spec — решение человека по ПАКЕТУ спеки (iter71)
# ----------------------------------------------------------------------
def spec_package_gates(ctx: ToolContext, spec, candidate) -> Dict[str, Any]:
    """Гейты применения ПАКЕТА спеки.

    Отличие от :func:`patch_gates` — в честности сравнения:

    * **спеки не было** (первичный ввод) — сравнивать не с чем, гейты
      неприменимы; это называется словами, а не выдаётся за «проверено»;
    * **состав компонентов изменился** (узел добавлен/удалён) — прежние точки
      живут в ДРУГОМ пространстве координат, и прогон их через новую спеку дал
      бы бессмысленный вердикт. Такой пакет не блокируется гейтом точек, но
      помечается как разрыв истории: решение осознанное, а не молчаливое (A0.6);
    * состав тот же (сменились роли/границы) — работают обычные гейты патча.
    """
    if spec is None:
        return {"ok": True, "checked": False, "blocked": [],
                "history_break": False,
                "reason": "первичный ввод геометрии: сравнивать не с чем",
                "points": {"checked": False,
                           "reason": "спеки до применения не было"},
                "preflight": {"checked": False,
                              "reason": "спеки до применения не было"}}
    if list(spec.component_names) != list(candidate.component_names):
        return {
            "ok": True, "checked": False, "blocked": [], "history_break": True,
            "reason": ("состав компонентов изменился — прежние точки лежат в "
                       "другом пространстве координат, гейт точек неприменим"),
            "components_before": list(spec.component_names),
            "components_after": list(candidate.component_names),
            "points": {"checked": False, "reason": "иной состав компонентов"},
            "preflight": {"checked": False,
                          "reason": "иной состав компонентов"}}
    gates = patch_gates(ctx, spec, candidate)
    return {**gates, "checked": True, "history_break": False}


def _spec_decision_record(staged: StagedSpec, diff: Dict[str, Any],
                          gates: Dict[str, Any], hash_before: str,
                          hash_after: str, author: str, note: str
                          ) -> Dict[str, Any]:
    """Запись в журнал решений о применении пакета спеки.

    Заголовок собирается ИЗ ДИФФА, а не из вольной формулировки: через полгода
    по журналу должно быть видно, был это первичный ввод или эволюция и какие
    узлы затронуты.
    """
    if staged.label:
        title = staged.label
    elif diff.get("first_spec"):
        title = (f"phr-спека: первичный ввод "
                 f"({diff.get('q_after', 0)} компонентов)")
    else:
        title = (f"phr-спека: эволюция геометрии "
                 f"(+{len(diff.get('added', []))} / "
                 f"−{len(diff.get('removed', []))} узлов, "
                 f"ролей изменено: {len(diff.get('role_changed', []))})")
    nodes = sorted(set(list(diff.get("added", []))
                       + list(diff.get("removed", []))
                       + [str(r.get("node")) for r
                          in diff.get("role_changed", [])]))
    return {
        "ts": _now(), "title": title, "nodes": nodes,
        "author": str(author or "человек"),
        "spec_hash": hash_before, "spec_hash_after": hash_after,
        "rationale": staged.rationale, "level": staged.level,
        "source": staged.source, "confidence": staged.confidence,
        "spec_id": staged.id, "affects_hash": hash_before != hash_after,
        "note": str(note or ""), "kind": "apply_spec",
        "first_spec": bool(diff.get("first_spec")),
        "history_break": bool(gates.get("history_break")),
        "gates": {"points": (gates.get("points", {}) or {}).get("ok"),
                  "preflight": (gates.get("preflight", {}) or {}).get("ok")},
    }


@register(
    "apply_spec",
    description=(
        "ПРИМЕНИТЬ пакет спеки из стейджа к проекту (первичный ввод геометрии "
        "или её эволюция). Требует разового токена подтверждения человека — "
        "кнопка в интерфейсе. Блокируется, если пакет не собирается ядром или "
        "(при НЕИЗМЕННОМ составе компонентов) выбрасывает уже измеренные точки "
        "и ломает preflight-гейты, которые до этого проходили. Применение "
        "записывается в журнал решений компании."),
    parameters={"type": "object", "properties": {
        "spec_id": {"type": "string",
                    "description": "id пакета спеки из стейджа"},
        "human_token": {"type": "string",
                        "description": "разовый токен подтверждения человека"},
        "note": {"type": "string", "description": "комментарий к применению"},
        "author": {"type": "string", "description": "кто принял решение"}},
        "required": ["spec_id", "human_token"]},
    kind=WRITE, long_running=True)
def apply_spec(ctx: ToolContext, spec_id: str, human_token: str,
               note: str = "", author: str = "") -> Dict[str, Any]:
    session = ctx.require_session()
    staged = session.spec_by_id(str(spec_id))
    if staged is None:
        raise ToolError(
            f"Пакета спеки '{spec_id}' нет в сессии. В стейдже: "
            f"{[s.id for s in session.staged_specs()] or 'пусто'}.")
    if staged.status != "staged":
        raise ToolError(
            f"Пакет '{spec_id}' уже в статусе '{staged.status}': повторное "
            f"применение запрещено — предложите новый пакет.")

    spec = active_spec(ctx)
    # Токен привязан к отпечатку спеки НА МОМЕНТ нажатия кнопки. При первичном
    # вводе отпечатка нет — привязка к пустой строке, и это честно: человек
    # подтверждал именно состояние «спеки нет».
    hash_before = spec.spec_hash() if spec is not None else ""
    _consume(ctx, human_token, action="apply_spec", target=str(spec_id),
             context_hash=hash_before)

    try:
        candidate = build_spec_from_package(staged.payload())
    except ToolError:
        raise
    except Exception as exc:                          # noqa: BLE001
        raise ToolError(
            f"ГЕЙТ ВАЛИДАЦИИ: пакет не даёт корректной спеки "
            f"({type(exc).__name__}: {exc}). Изменение НЕ применено, геометрия "
            f"прежняя.") from exc

    gates = spec_package_gates(ctx, spec, candidate)
    if not gates["ok"]:
        raise ToolError(
            "ГЕЙТ ПРИМЕНЕНИЯ: пакет НЕ применён. "
            + "; ".join(gates.get("blocked", []))
            + ". Пакет остаётся в стейдже: смягчите правку или обсудите "
              "последствия с технологом.")

    runner = ctx.runner
    if runner is not None and hasattr(runner, "set_phr_spec"):
        try:
            runner.set_phr_spec(candidate)
        except Exception as exc:                      # noqa: BLE001
            raise ToolError(
                f"Спека не принята проектом ({type(exc).__name__}: {exc}). "
                f"Компоненты пакета должны существовать среди "
                f"mixture-компонентов схемы: если состав кампании расширяется, "
                f"это эволюция СХЕМЫ проекта — её делает человек в сетапе, а "
                f"не применение пакета.") from exc
    ctx.spec = candidate

    hash_after = candidate.spec_hash()
    diff = spec_package_diff(spec, candidate)
    session.set_spec_status(staged.id, PATCH_APPLIED, reason=str(note or ""))
    decision = _log(ctx, "decisions", _spec_decision_record(
        staged, diff, gates, hash_before, hash_after, author, note))

    warning = ""
    if diff.get("first_spec"):
        warning = (f"Геометрия кампании зафиксирована: spec_hash "
                   f"{hash_after[:12]}…. Дальнейшие точки относятся к нему.")
    elif gates.get("history_break"):
        warning = ("Состав компонентов изменился ⇒ ранее собранные точки "
                   "принадлежат ДРУГОМУ пространству координат: они остаются в "
                   "базе, но к новой геометрии не относятся.")
    elif hash_before != hash_after:
        warning = ("spec_hash изменился ⇒ ранее собранные точки относятся к "
                   "ПРЕЖНЕЙ геометрии: план дальше строится в новой области.")

    return _f({
        "ok": True, "spec_id": staged.id, "status": PATCH_APPLIED,
        "spec_hash_before": hash_before, "spec_hash_after": hash_after,
        "affects_hash": hash_before != hash_after,
        "diff": diff, "gates": gates, "decision": decision,
        "warning": warning,
        "persist_hint": ("Спека изменена в памяти проекта — сохраните "
                         "кампанию, чтобы правка пережила перезапуск."),
    })


@register(
    "reject_spec",
    description=(
        "Отклонить пакет спеки из стейджа (решение человека, требует токен "
        "подтверждения). Отказ ТОЖЕ идёт в журнал решений: «почему не приняли "
        "эту геометрию» должно разрешаться журналом, а не памятью участников."),
    parameters={"type": "object", "properties": {
        "spec_id": {"type": "string", "description": "id пакета из стейджа"},
        "human_token": {"type": "string", "description": "разовый токен"},
        "reason": {"type": "string", "description": "почему отклонён"},
        "author": {"type": "string", "description": "кто решил"}},
        "required": ["spec_id", "human_token", "reason"]},
    kind=WRITE)
def reject_spec(ctx: ToolContext, spec_id: str, human_token: str,
                reason: str, author: str = "") -> Dict[str, Any]:
    session = ctx.require_session()
    staged = session.spec_by_id(str(spec_id))
    if staged is None:
        raise ToolError(f"Пакета спеки '{spec_id}' нет в сессии.")
    _consume(ctx, human_token, action="reject_spec", target=str(spec_id))
    session.set_spec_status(staged.id, PATCH_REJECTED, reason=str(reason))
    summary = staged.summary or {}
    decision = _log(ctx, "decisions", {
        "ts": _now(),
        "title": f"ОТКЛОНЕНО: пакет спеки {staged.label or staged.id}",
        "nodes": sorted(summary.get("added", []) or []),
        "author": str(author or "человек"),
        "spec_hash": str(summary.get("spec_hash_before", "") or ""),
        "rationale": str(reason), "spec_id": staged.id,
        "kind": "reject_spec"})
    return {"ok": True, "spec_id": staged.id, "status": PATCH_REJECTED,
            "decision": decision}


# ----------------------------------------------------------------------
# Пакет ПРОЕКТА (iter73): предложить → принять человеком
# ----------------------------------------------------------------------
@register(
    "propose_project",
    description=(
        "ПРЕДЛОЖИТЬ ПРОЕКТ ЦЕЛИКОМ, когда его в сессии ещё нет: состав "
        "('spec' — phr-спека), отклики ('responses') и процесс-оси с границами "
        "('process'), плюс необязательные 'covariates', 'passport', 'seed'. "
        "Именно этим инструментом закрывается первичный ввод: одной phr-спеки "
        "не хватает, потому что откликов и осей в ней нет по схеме, а без них "
        "движок не собирается. Пакет проверяется ядром и кладётся в СТЕЙДЖ; "
        "проект этим вызовом НЕ создаётся — принимает человек кнопкой. Формат "
        "бери из project_schema. Отклики и границы осей НЕ выдумывай: не "
        "назвали — спроси."),
    parameters={"type": "object", "properties": {
        "package": {"type": "object",
                    "description": "пакет проекта ЦЕЛИКОМ (см. project_schema)"},
        "rationale": {"type": "string",
                      "description": "почему такой состав, отклики и оси: "
                                     "физика, паспорт, практика цеха"},
        "label": {"type": "string",
                  "description": "короткая метка пакета (например «кромка ПВХ: "
                                 "первичный ввод проекта»)"},
        "level": {"type": "string", "description": "L1 | L2 | L3"},
        "source": {"type": "string", "description": "источник сведений"},
        "confidence": {"type": "string", "description": "high | med | low"}},
        "required": ["package", "rationale"]},
    kind=PROPOSE)
def propose_project(ctx: ToolContext, package: Any, rationale: str,
                    label: str = "", level: str = "", source: str = "",
                    confidence: str = "") -> Dict[str, Any]:
    """Пакет проекта → стейдж сессии (после разбора ЯДРОМ).

    Отказ разбора возвращается РЕЗУЛЬТАТОМ: пункт, заведомо неприменимый, в
    стейдже был бы ловушкой — человек нажал бы «Принять» и получил ошибку
    вместо проекта (ровно то, что случилось с пакетом спеки в живой сессии).
    """
    from ...design.project_package import (PackageError, manifest_caption,
                                           package_manifest,
                                           parse_project_package)
    session = ctx.require_session()
    try:
        pkg = parse_project_package(package)
    except PackageError as exc:
        return {"staged": False, "ok": False, "error": str(exc),
                "hint": ("Пакет отклонён ядром и в стейдж НЕ положен: проект не "
                         "тронут. Сверь блоки и единицы со project_schema и "
                         "предложи заново.")}

    manifest = package_manifest(pkg)
    staged = StagedProject(
        package=dict(pkg.raw), label=str(label or pkg.label or ""),
        rationale=str(rationale or ""), level=str(level or ""),
        source=str(source or ""), confidence=str(confidence or ""),
        summary=manifest)
    session.stage_project(staged)

    out: Dict[str, Any] = {
        "staged": True, "ok": True, "project_id": staged.id,
        "manifest": manifest, "caption": manifest_caption(pkg),
        "note": ("Пакет НЕ применён: он в стейдже. Принимает человек кнопкой "
                 "(разовый токен). Перечисли ему по блокам, что приедет: "
                 "компоненты, отклики с единицами, оси с границами."),
    }
    if has_project(ctx):
        out["warning"] = (
            "В сессии УЖЕ есть собранный проект: пакет проекта его не заменяет "
            "и применён не будет. Для правки геометрии нужен пакет спеки "
            "(propose_spec), отклики и оси меняет человек в сетапе.")
    return _f(out)


@register(
    "apply_project",
    description=(
        "ПРИНЯТЬ пакет проекта из стейджа: блоки пакета переносятся в поля "
        "формы сетапа («🆕 Новый проект»), после чего проект собирает штатная "
        "кнопка «🏗 Построить проект». Требует разового токена подтверждения "
        "человека — кнопка в интерфейсе. Отклоняется, если проект в сессии уже "
        "собран: это рождение проекта, а не его правка."),
    parameters={"type": "object", "properties": {
        "project_id": {"type": "string",
                       "description": "id пакета проекта из стейджа"},
        "human_token": {"type": "string",
                        "description": "разовый токен подтверждения человека"},
        "note": {"type": "string", "description": "комментарий к принятию"},
        "author": {"type": "string", "description": "кто принял решение"}},
        "required": ["project_id", "human_token"]},
    kind=WRITE)
def apply_project(ctx: ToolContext, project_id: str, human_token: str,
                  note: str = "", author: str = "") -> Dict[str, Any]:
    """Принять пакет проекта: вернуть ПРЕФИЛЛ формы сетапа и записать решение.

    Раннер здесь НЕ собирается сознательно: сборка проекта в приложении
    остаётся одна — штатная кнопка формы. Инструмент отвечает за другое:
    проверить пакет, погасить токен, отдать значения полей и зафиксировать
    решение в журнале компании.
    """
    from ...design.project_package import (PackageError, manifest_caption,
                                           package_manifest,
                                           package_to_setup_prefill,
                                           parse_project_package)
    session = ctx.require_session()
    staged = session.project_by_id(str(project_id))
    if staged is None:
        raise ToolError(
            f"Пакета проекта '{project_id}' нет в сессии. В стейдже: "
            f"{[p.id for p in session.staged_projects()] or 'пусто'}.")
    if staged.status != "staged":
        raise ToolError(
            f"Пакет '{project_id}' уже в статусе '{staged.status}': повторное "
            f"применение запрещено — предложите новый пакет.")
    if has_project(ctx):
        raise ToolError(
            "В сессии уже собран проект: пакетом проекта он не заводится "
            "заново — иначе молча пропали бы измеренные точки и ветки. Пакет "
            "остаётся в стейдже. Правку геометрии применяйте пакетом спеки, а "
            "отклики и оси меняйте в сетапе, собрав проект заново осознанно.")

    try:
        pkg = parse_project_package(staged.payload())
    except PackageError as exc:
        raise ToolError(
            f"ГЕЙТ ВАЛИДАЦИИ: пакет проекта не разбирается ядром ({exc}). "
            f"Проект НЕ создан, состояние прежнее.") from exc
    # iter79: ПРЕФИЛЛ считается ДО гашения токена. Раньше порядок был обратный,
    # и сбой проекции пакета в поля формы (например, связка осей строкой в
    # паспорте) сжигал разовый токен: пакет оставался в стейдже, а повторное
    # «Применить» требовало нового подтверждения. Ошибка проекции — это отказ
    # ГЕЙТА, а не внутренняя поломка: она должна дойти до человека как ToolError
    # (UI покажет её через st.error), а не уронить страницу целиком.
    try:
        prefill = package_to_setup_prefill(pkg)
    except PackageError as exc:
        raise ToolError(
            f"ГЕЙТ ВАЛИДАЦИИ: пакет проекта не переносится в поля формы "
            f"({exc}). Проект НЕ создан, состояние прежнее, подтверждение не "
            f"израсходовано — исправьте пакет и предложите заново."
        ) from exc

    # Токен привязан к отпечатку СПЕКИ пакета: подменить пакет между нажатием
    # кнопки и вызовом нельзя.
    _consume(ctx, human_token, action="apply_project", target=str(project_id),
             context_hash=pkg.spec_hash)

    manifest = package_manifest(pkg)
    session.set_project_status(staged.id, PATCH_APPLIED, reason=str(note or ""))
    decision = _log(ctx, "decisions", _project_decision_record(
        staged, pkg, author=author, note=note))
    return _f({
        "ok": True, "project_id": staged.id, "status": PATCH_APPLIED,
        "spec_hash": pkg.spec_hash, "manifest": manifest,
        "caption": manifest_caption(pkg),
        "setup_prefill": prefill, "decision": decision,
        "next_step": ("Поля формы «🆕 Новый проект» на закладке «🌱 Старт» "
                      "заполнены из пакета. Проверьте их и нажмите "
                      "«🏗 Построить проект» — сборка проекта в приложении "
                      "одна, и она остаётся за вами."),
        "persist_hint": ("После сборки сохраните проект (панель «📁 Проект»), "
                         "иначе он не переживёт перезапуск."),
    })


def _project_decision_record(staged: StagedProject, pkg: Any, *,
                             author: str, note: str) -> Dict[str, Any]:
    """Запись в журнал о принятии пакета проекта.

    Заголовок собирается ИЗ ПАКЕТА, а не из вольной формулировки: через полгода
    по журналу должно быть видно, из чего проект родился — сколько компонентов,
    какие отклики и оси.
    """
    return {
        "ts": _now(),
        "title": (staged.label
                  or f"проект: первичный ввод ({len(pkg.component_names)} "
                     f"компонентов, {len(pkg.responses)} откликов, "
                     f"{len(pkg.process)} процесс-осей)"),
        "nodes": sorted(pkg.spec.phr_intervals()),
        "author": str(author or "человек"),
        "spec_hash": "", "spec_hash_after": pkg.spec_hash,
        "rationale": staged.rationale, "level": staged.level,
        "source": staged.source, "confidence": staged.confidence,
        "project_id": staged.id, "affects_hash": True,
        "note": str(note or ""), "kind": "apply_project",
        "responses": list(pkg.response_names),
        "process": list(pkg.process_names),
        "covariates": list(pkg.covariates),
    }


@register(
    "reject_project",
    description=(
        "Отклонить пакет проекта из стейджа (решение человека, требует токен). "
        "Отказ ТОЖЕ идёт в журнал решений: «почему не завели проект в таком "
        "виде» должно разрешаться журналом, а не памятью участников."),
    parameters={"type": "object", "properties": {
        "project_id": {"type": "string", "description": "id пакета из стейджа"},
        "human_token": {"type": "string", "description": "разовый токен"},
        "reason": {"type": "string", "description": "почему отклонён"},
        "author": {"type": "string", "description": "кто решил"}},
        "required": ["project_id", "human_token", "reason"]},
    kind=WRITE)
def reject_project(ctx: ToolContext, project_id: str, human_token: str,
                   reason: str, author: str = "") -> Dict[str, Any]:
    session = ctx.require_session()
    staged = session.project_by_id(str(project_id))
    if staged is None:
        raise ToolError(f"Пакета проекта '{project_id}' нет в сессии.")
    _consume(ctx, human_token, action="reject_project", target=str(project_id))
    session.set_project_status(staged.id, PATCH_REJECTED, reason=str(reason))
    summary = staged.summary or {}
    decision = _log(ctx, "decisions", {
        "ts": _now(),
        "title": f"ОТКЛОНЕНО: пакет проекта {staged.label or staged.id}",
        "nodes": sorted(summary.get("components", []) or []),
        "author": str(author or "человек"),
        "spec_hash": str(summary.get("spec_hash", "") or ""),
        "rationale": str(reason), "project_id": staged.id,
        "level": staged.level, "source": staged.source,
        "kind": "reject_project",
    })
    return {"ok": True, "project_id": staged.id, "status": PATCH_REJECTED,
            "decision": decision}


# ----------------------------------------------------------------------
# Правка ПОЛЕЙ ФОРМЫ сетапа (iter76): предложить → принять человеком
# ----------------------------------------------------------------------
#: Скалярные типы значений полей формы — то, что честно переносится в виджеты.
_SETUP_SCALARS = (str, int, float, bool)


@register(
    "propose_setup_fields",
    description=(
        "ПРЕДЛОЖИТЬ ТОЧЕЧНУЮ ПРАВКУ ПОЛЕЙ формы «🆕 Новый проект» ДО сборки "
        "проекта: {ключ_поля: новое_значение} — только изменяемые "
        "поля, остальные не трогаются. Ключи и текущие значения бери из "
        "get_setup_fields (например 'setup_resp' — отклики через запятую, "
        "'setup_preflight_pairs' — пары построчно «A | B», 'setup_phr_json' — "
        "phr-спека JSON). Правка кладётся в СТЕЙДЖ; применяет человек "
        "кнопкой, значения попадут в поля формы, а проект соберёт кнопка "
        "«🏗 Построить проект». Для СОБРАННОГО проекта не годится — там "
        "propose_patch/propose_spec."),
    parameters={"type": "object", "properties": {
        "fields": {"type": "object",
                   "description": "{ключ_поля_формы: новое значение} — "
                                  "только скаляры (строка/число/булево)"},
        "rationale": {"type": "string",
                      "description": "почему такая правка: физика, паспорт, "
                                     "слова технолога"},
        "label": {"type": "string",
                  "description": "короткая метка правки (например «верх "
                                 "mixer_freq 50→60 Hz»)"},
        "level": {"type": "string", "description": "L1 | L2 | L3"},
        "source": {"type": "string", "description": "источник сведений"},
        "confidence": {"type": "string", "description": "high | med | low"}},
        "required": ["fields", "rationale"]},
    kind=PROPOSE)
def propose_setup_fields(ctx: ToolContext, fields: Any, rationale: str,
                         label: str = "", level: str = "", source: str = "",
                         confidence: str = "") -> Dict[str, Any]:
    """Правка полей формы → стейдж сессии (применяет человек кнопкой).

    Гейтов ядра здесь нет намеренно: поля формы — черновик, их валидирует
    штатная кнопка «🏗 Построить проект» (те же парсеры и сеттеры, что при
    ручном вводе). Зато есть проверка ФОРМЫ правки: ключи ``setup_*`` и
    скалярные значения — иначе применённая правка молча не легла бы в виджеты.
    """
    session = ctx.require_session()
    if ctx.runner is not None:
        return {"staged": False, "ok": False,
                "error": "Проект в сессии уже СОБРАН: поля формы сетапа — "
                         "черновик пересборки, точечная правка полей к нему "
                         "не применяется. Геометрию правь пакетом спеки "
                         "(propose_spec) или патчем (propose_patch).",
                "hint": "Правка не положена в стейдж."}
    if not isinstance(fields, dict) or not fields:
        return {"staged": False, "ok": False,
                "error": "Ожидается непустой объект {ключ_поля: значение}.",
                "hint": "Ключи полей смотри в get_setup_fields."}
    bad_keys = [k for k in fields if not str(k).startswith("setup_")]
    if bad_keys:
        return {"staged": False, "ok": False,
                "error": f"Ключи {bad_keys} не похожи на поля формы сетапа: "
                         f"все ключи начинаются с 'setup_' "
                         f"(см. get_setup_fields).",
                "hint": "Правка не положена в стейдж."}
    bad_vals = {str(k): type(v).__name__ for k, v in fields.items()
                if v is not None and not isinstance(v, _SETUP_SCALARS)}
    if bad_vals:
        return {"staged": False, "ok": False,
                "error": f"Значения полей должны быть скалярами (строка/"
                         f"число/булево), получено: {bad_vals}. Списки "
                         f"кодируются строкой поля (например, имена через "
                         f"запятую, пары построчно).",
                "hint": "Правка не положена в стейдж."}

    current = dict((ctx.extra or {}).get("setup_fields") or {})
    staged = session.stage_setup(StagedSetup(
        fields={str(k): v for k, v in fields.items()},
        label=str(label or ""), rationale=str(rationale or ""),
        level=str(level or ""), source=str(source or ""),
        confidence=str(confidence or "")))
    return _f({
        "staged": True, "ok": True, "setup_id": staged.id,
        "fields": dict(staged.fields),
        "current_values": {k: current.get(k) for k in staged.fields},
        "note": ("Правка НЕ применена: она в стейдже (панель «📝 Предложенные "
                 "правки полей»). Применяет человек кнопкой; значения лягут в "
                 "поля формы «🆕 Новый проект», проект соберёт кнопка "
                 "«🏗 Построить проект». Скажи пользователю, какие поля и "
                 "почему меняются."),
    })


@register(
    "apply_setup_fields",
    description=(
        "ПРИМЕНИТЬ правку полей формы сетапа из стейджа: значения переносятся "
        "в поля формы «🆕 Новый проект». Требует разовый токен подтверждения "
        "человека — кнопка в интерфейсе. Отклоняется, если проект уже собран."),
    parameters={"type": "object", "properties": {
        "setup_id": {"type": "string", "description": "id правки из стейджа"},
        "human_token": {"type": "string", "description": "разовый токен"},
        "note": {"type": "string", "description": "комментарий человека"},
        "author": {"type": "string", "description": "кто решил"}},
        "required": ["setup_id", "human_token"]},
    kind=WRITE)
def apply_setup_fields(ctx: ToolContext, setup_id: str, human_token: str,
                       note: str = "", author: str = "") -> Dict[str, Any]:
    """Принятие правки полей человеком → ``setup_prefill`` для формы.

    Раннер здесь НЕ трогается (его нет по определению шага); результат —
    словарь значений полей, который UI кладёт в ``setup_prefill_pending``,
    плюс запись в журнал решений (решение человека должно быть видно потом).
    """
    session = ctx.require_session()
    staged = session.setup_by_id(str(setup_id))
    if staged is None:
        raise ToolError(
            f"Правки сетапа '{setup_id}' нет в сессии. В стейдже: "
            f"{[s.id for s in session.staged_setups()] or 'пусто'}.")
    if staged.status != "staged":
        raise ToolError(
            f"Правка '{setup_id}' уже в статусе '{staged.status}': повторное "
            f"применение запрещено — предложите новую правку.")
    if ctx.runner is not None:
        raise ToolError(
            "Проект в сессии уже собран: правка полей формы к нему не "
            "применяется. Правка остаётся в стейдже.")
    _consume(ctx, human_token, action="apply_setup", target=str(setup_id))

    session.set_setup_status(staged.id, PATCH_APPLIED, reason=str(note or ""))
    decision = _log(ctx, "decisions", {
        "ts": _now(),
        "title": (staged.label
                  or f"правка полей сетапа: {sorted(staged.fields)}"),
        "nodes": [], "author": str(author or "человек"),
        "spec_hash": "", "rationale": staged.rationale,
        "level": staged.level, "source": staged.source,
        "confidence": staged.confidence, "setup_id": staged.id,
        "fields": sorted(staged.fields), "note": str(note or ""),
        "kind": "apply_setup"})
    return _f({
        "ok": True, "setup_id": staged.id, "status": PATCH_APPLIED,
        "setup_prefill": dict(staged.fields), "decision": decision,
        "next_step": (f"Поля {sorted(staged.fields)} формы «🆕 Новый проект» "
                      f"обновлены из правки. Проверьте их и нажмите "
                      f"«🏗 Построить проект» — сборка проекта остаётся "
                      f"за вами."),
        "persist_hint": ("Черновик формы можно сохранить кнопкой "
                         "«💾 Сохранить проект» (панель «📁 Проект») — "
                         "он переживёт перезапуск и до сборки."),
    })


@register(
    "reject_setup_fields",
    description=(
        "Отклонить правку полей сетапа из стейджа (решение человека, требует "
        "токен). Отказ идёт в журнал решений."),
    parameters={"type": "object", "properties": {
        "setup_id": {"type": "string", "description": "id правки из стейджа"},
        "human_token": {"type": "string", "description": "разовый токен"},
        "reason": {"type": "string", "description": "почему отклонена"},
        "author": {"type": "string", "description": "кто решил"}},
        "required": ["setup_id", "human_token", "reason"]},
    kind=WRITE)
def reject_setup_fields(ctx: ToolContext, setup_id: str, human_token: str,
                        reason: str, author: str = "") -> Dict[str, Any]:
    session = ctx.require_session()
    staged = session.setup_by_id(str(setup_id))
    if staged is None:
        raise ToolError(f"Правки сетапа '{setup_id}' нет в сессии.")
    _consume(ctx, human_token, action="reject_setup", target=str(setup_id))
    session.set_setup_status(staged.id, PATCH_REJECTED, reason=str(reason))
    decision = _log(ctx, "decisions", {
        "ts": _now(),
        "title": f"ОТКЛОНЕНО: правка сетапа {staged.label or staged.id}",
        "nodes": [], "author": str(author or "человек"),
        "spec_hash": "", "rationale": str(reason),
        "setup_id": staged.id, "fields": sorted(staged.fields),
        "kind": "reject_setup"})
    return {"ok": True, "setup_id": staged.id, "status": PATCH_REJECTED,
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


def issue_apply_spec_token(ctx: ToolContext, spec_id: str, *,
                           ttl_s: Optional[float] = None,
                           note: str = "") -> str:
    """Токен на применение ПАКЕТА спеки (кнопка «Применить спеку», iter71).

    Привязан к отпечатку спеки на момент нажатия. При первичном вводе спеки
    нет — ``context_hash`` пуст, и это единственный корректный вариант:
    подтверждается состояние «геометрии ещё не было».
    """
    spec = active_spec(ctx)
    return registry_for(ctx).issue(
        "apply_spec", str(spec_id),
        context_hash=spec.spec_hash() if spec is not None else "",
        ttl_s=ttl_s, note=note).token


def issue_reject_spec_token(ctx: ToolContext, spec_id: str, *,
                            ttl_s: Optional[float] = None) -> str:
    """Токен на отклонение пакета спеки (кнопка «Отклонить спеку»)."""
    return registry_for(ctx).issue("reject_spec", str(spec_id),
                                   ttl_s=ttl_s).token


def issue_apply_project_token(ctx: ToolContext, project_id: str, *,
                              ttl_s: Optional[float] = None,
                              note: str = "") -> str:
    """Токен на принятие ПАКЕТА ПРОЕКТА (кнопка «Принять проект», iter73).

    Привязан к отпечатку СПЕКИ ИЗ ПАКЕТА, а не к активной спеке проекта: проекта
    в этот момент нет по определению, а подменить пакет между нажатием кнопки и
    вызовом инструмента нельзя.
    """
    from ...design.project_package import PackageError, parse_project_package

    session = ctx.require_session()
    staged = session.project_by_id(str(project_id))
    if staged is None:
        raise ToolError(f"Пакета проекта '{project_id}' нет в сессии.")
    try:
        spec_hash = parse_project_package(staged.payload()).spec_hash
    except PackageError as exc:
        raise ToolError(f"Пакет проекта '{project_id}' не разбирается ядром "
                        f"({exc}) — подтверждать нечего.") from exc
    return registry_for(ctx).issue("apply_project", str(project_id),
                                   context_hash=spec_hash, ttl_s=ttl_s,
                                   note=note).token


def issue_reject_project_token(ctx: ToolContext, project_id: str, *,
                               ttl_s: Optional[float] = None) -> str:
    """Токен на отклонение пакета проекта (кнопка «Отклонить проект»)."""
    return registry_for(ctx).issue("reject_project", str(project_id),
                                   ttl_s=ttl_s).token


def issue_apply_setup_token(ctx: ToolContext, setup_id: str, *,
                            ttl_s: Optional[float] = None,
                            note: str = "") -> str:
    """Токен на применение ПРАВКИ ПОЛЕЙ сетапа (кнопка «Применить», iter76).

    ``context_hash`` пуст намеренно: правка относится к ЧЕРНОВИКУ формы, у
    которого отпечатка геометрии нет по определению шага (проект не собран).
    """
    return registry_for(ctx).issue("apply_setup", str(setup_id),
                                   ttl_s=ttl_s, note=note).token


def issue_reject_setup_token(ctx: ToolContext, setup_id: str, *,
                             ttl_s: Optional[float] = None) -> str:
    """Токен на отклонение правки полей сетапа (кнопка «Отклонить»)."""
    return registry_for(ctx).issue("reject_setup", str(setup_id),
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
