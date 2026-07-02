"""apps/campaign.py — честный view-model кампании поверх MixtureProcessRunner.

Блок «Per-branch роли откликов и эволюция кампании» (ТЗ v1.1, REBUILD_SPEC
§16/§16.1). Это ШАГ 1: ЧИСТАЯ ЛОГИКА (без Streamlit) — read-model, который
превращает состояние ядра в честные, контекстно-однозначные данные для UI,
MCP `doe-introspect` и ассистента. Канон репозитория: сначала логика + тест,
потом UI.

Что обязан гарантировать этот слой (инварианты честности ТЗ):

  * **Тр-3.3 / П-3 — контекст ветки однозначен.** Role-tag — УСЛОВНАЯ правда,
    валидная только в контексте КОНКРЕТНОЙ ветки. Поэтому каждый репорт всегда
    несёт ``branch_id``/``branch_name`` и роли считаются строго для запрошенной
    ветки (``runner.response_role`` branch-local, Гр-3). Безконтекстных ролей
    модуль не отдаёт.
  * **П-1 / П-9 — XOR внутри ветки.** В ветке отклик несёт РОВНО одну роль;
    ``responses_by_role`` разбивает ВСЕ отклики оракула без потерь и дублей.
  * **И-5 / Гр-1 / П-6 — денежный канал role-aware.** Для ρ-отклика ценовой ноги
    канал помечается ``zeroed`` (роль OPTIMIZED ⇒ σ_ρ занулена) либо ``alive``
    (роль PRICE_INPUT ⇒ σ_ρ кормит деньги). Это читается из РЕАЛЬНОЙ атрибуции
    ядра (``price_channel_suppressed``), а не выдумывается тегом.
  * **§6 / Тр-6.3/6.6 — покрытие N/M.** Покрытие отклика берётся из ОБЩЕЙ базы
    (``runner.points``), которая НИКОГДА не урезается (И-1/П-11). Полное покрытие
    по недомеренному отклику не подразумевается: показываем честные N/M и флаг
    низкого покрытия (доля-порог переносим между откликами, Тр-6.4).
  * **§16.1 — объяснение «почему за ρ нет денег».** :func:`branch_money_explanation`
    отдаёт причину занулённого/живого денежного канала — для ассистента и MCP.

A0.6: модуль НИЧЕГО не меняет — он только ЧИТАЕТ состояние (read-only). Мутации
(смена роли, веса, spawn, undo) — отдельный слой следующих шагов.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.schema_evolution import (KNOWN_CONSTANT, evolve_schema,
                                      known_constant)
from ..design.branches import (ROLE_OPTIMIZED, ROLE_PRICE_INPUT, ROLE_REFERENCE,
                                ROLE_PRIORITY)
from ..optimize.desirability import Desirability, DesirabilitySpec


# ----------------------------------------------------------------------
# Ярлыки ролей и денежного канала (для UI/MCP; единый источник — без копий)
# ----------------------------------------------------------------------
ROLE_LABELS_RU: Dict[str, str] = {
    ROLE_OPTIMIZED: "цель",            # нога качества d_i (оптимизируем)
    ROLE_PRICE_INPUT: "цена-вход",     # питает цену изделия (ρ ценовой ноги)
    ROLE_REFERENCE: "справочная",      # меряется, но ни цель, ни цена
}
ROLE_CODE_LABELS: Dict[str, str] = {
    ROLE_OPTIMIZED: "OPTIMIZED",
    ROLE_PRICE_INPUT: "PRICE_INPUT",
    ROLE_REFERENCE: "REFERENCE",
}

# Денежный канал ρ-отклика (И-5): занулён (ушёл в качество) / живой (кормит цену).
MONEY_ZEROED = "zeroed"
MONEY_ALIVE = "alive"

# Доля-порог достаточного покрытия (Тр-6.4): переносима между откликами; abs. N
# виден в N/M. Это ADVISORY-параметр view-model, ЯДРО им не трогается (Тр-6.5).
DEFAULT_COVERAGE_FRACTION = 0.5


# ----------------------------------------------------------------------
# Покрытие истории (§6): N измерено из M в ОБЩЕЙ базе (история не урезается)
# ----------------------------------------------------------------------
def response_coverage(runner) -> Dict[str, Dict[str, float]]:
    """``{отклик → {measured, total, fraction}}`` по ОБЩЕЙ базе (§6 / Тр-6.3).

    ``measured`` — число точек базы с КОНЕЧНЫМ измеренным значением отклика;
    ``total`` — размер общей базы (``len(runner.points)``); ``fraction`` =
    measured/total (0.0 при пустой базе). Поскольку история никогда не урезается
    (И-1/П-11), ``total`` равен полному числу снятых точек. Для отклика,
    введённого задним числом (+отклик, §6), у старых точек значения нет ⇒
    measured < total — частичное покрытие отображается честно, без фантома
    «полного покрытия» по недомеренному свойству.
    """
    points = list(getattr(runner, "points", []) or [])
    total = len(points)
    out: Dict[str, Dict[str, float]] = {}
    for name in runner.property_names:
        measured = 0
        for p in points:
            y = getattr(p, "Y", {}) or {}
            val = y.get(name, None)
            if val is None:
                continue
            try:
                if np.isfinite(float(val)):
                    measured += 1
            except (TypeError, ValueError):
                continue
        frac = (measured / total) if total > 0 else 0.0
        out[name] = {"measured": int(measured), "total": int(total),
                     "fraction": float(frac)}
    return out


# ----------------------------------------------------------------------
# Ценовая конфигурация ветки (read-only): какой отклик — ρ ценовой ноги
# ----------------------------------------------------------------------
def branch_price_config(runner, branch_id: str) -> Optional[Dict[str, str]]:
    """``{rho_property, cost_name}`` ценовой ноги ветки или ``None`` (нет цены).

    Читает конфигурацию ``set_branch_cost`` раннера (ветка без неё — чисто
    техническая, денежного канала нет). Возвращаемое — только имена (для UI/MCP),
    без callable.
    """
    cfg = (getattr(runner, "_branch_cost", {}) or {}).get(branch_id)
    if not cfg:
        return None
    return {"rho_property": str(cfg.get("rho_property")),
            "cost_name": str(cfg.get("cost_name", "price"))}


def _money_channel(response: str, rho_property: Optional[str],
                   suppressed: bool) -> Optional[str]:
    """Статус денежного канала ρ-отклика (И-5): ``zeroed``/``alive``/``None``.

    Имеет смысл ТОЛЬКО для отклика, который является ρ ценовой ноги ветки.
    ``suppressed`` (=``price_channel_suppressed`` ветки) ⇒ роль ρ = OPTIMIZED,
    σ_ρ-канал занулён (Гр-1) ⇒ ``zeroed``; иначе ρ = PRICE_INPUT, канал живой ⇒
    ``alive``. Для не-ρ откликов денежного канала нет ⇒ ``None``.
    """
    if rho_property is None or response != rho_property:
        return None
    return MONEY_ZEROED if suppressed else MONEY_ALIVE


# ----------------------------------------------------------------------
# Per-branch роль-репорт (контекст ветки однозначен — Тр-3.3)
# ----------------------------------------------------------------------
def branch_role_report(runner, branch_id: str, *,
                       coverage_fraction: float = DEFAULT_COVERAGE_FRACTION
                       ) -> Dict[str, Any]:
    """Честный role-репорт ветки (Тр-2/Тр-3/§6) — JSON-сериализуемый.

    Каждый отклик оракула получает свою роль В КОНТЕКСТЕ ИМЕННО ЭТОЙ ветки
    (Тр-3.3), ярлык, статус денежного канала (И-5) и покрытие N/M (§6). XOR
    честности (П-1/П-9) гарантируется приоритетом M2 в ядре — здесь он лишь
    отражается. ``coverage_fraction`` — доля-порог низкого покрытия (Тр-6.4/6.6).
    """
    if branch_id not in getattr(runner, "branches", {}):
        raise KeyError(f"Нет ветки '{branch_id}'.")
    br = runner.branches[branch_id]
    roles = runner.branch_roles(branch_id)
    by_role = runner.responses_by_role(branch_id)
    suppressed = bool(runner.price_channel_suppressed(branch_id))
    pcfg = branch_price_config(runner, branch_id)
    rho = pcfg["rho_property"] if pcfg else None
    cov = response_coverage(runner)

    responses: List[Dict[str, Any]] = []
    for name in runner.property_names:
        role = roles[name]
        spec = (br.goal or {}).get(name)
        c = cov[name]
        responses.append({
            "response": name,
            "role": role,
            "role_label": ROLE_LABELS_RU[role],
            "role_code": ROLE_CODE_LABELS[role],
            "in_goal": spec is not None,
            "desirability_kind": getattr(spec, "kind", None),
            "weight": (float(getattr(spec, "weight", 0.0))
                       if spec is not None else None),
            "feeds_price": (rho is not None and rho == name),
            "money_channel": _money_channel(name, rho, suppressed),
            "coverage_measured": c["measured"],
            "coverage_total": c["total"],
            "coverage_fraction": c["fraction"],
            "low_coverage": bool(c["fraction"] < float(coverage_fraction)),
        })

    return {
        "branch_id": branch_id,
        "branch_name": getattr(br, "name", branch_id),
        # Тр-3.3: линза ветки активна и однозначна — тег читается без догадок.
        "context_explicit": True,
        "has_price_leg": pcfg is not None,
        "rho_property": rho,
        "price_channel_suppressed": suppressed,
        "by_role": {r: list(by_role.get(r, [])) for r in ROLE_PRIORITY},
        "responses": responses,
        "coverage_fraction_threshold": float(coverage_fraction),
    }


# ----------------------------------------------------------------------
# §16.1 — объяснение «почему за ρ нет денег» (для ассистента/MCP)
# ----------------------------------------------------------------------
def branch_money_explanation(runner, branch_id: str, *,
                             compute_value: bool = True,
                             **econ_kwargs: Any) -> Dict[str, Any]:
    """Честное объяснение денежного канала ветки (И-5/Гр-1, §16.1) — read-only.

    Возвращает причину (``reason_code``) и человекочитаемый ``text``:

      * ``no_price_leg`` — у ветки нет ценовой ноги (``set_branch_cost`` не задан):
        чисто техническая ветка, денежного VoI-канала нет;
      * ``rho_optimized_zeroed`` — ρ носит роль OPTIMIZED (цель И питает цену):
        ценовой σ_ρ-разведочный канал ЗАНУЛЁН (двойной счёт одной δρ убран),
        денежная нога раунда = 0;
      * ``price_input_alive`` — ρ носит роль PRICE_INPUT: канал ЖИВОЙ, разведка ρ
        засчитывается деньгами.

    ``compute_value`` ⇒ при наличии суррогатов считает ``economic_value`` ветки
    (``runner.branch_economic_value``, read-only; ``**econ_kwargs`` пробрасываются
    туда). Метод НИЧЕГО не измеряет (A0.6).
    """
    if branch_id not in getattr(runner, "branches", {}):
        raise KeyError(f"Нет ветки '{branch_id}'.")
    pcfg = branch_price_config(runner, branch_id)
    suppressed = bool(runner.price_channel_suppressed(branch_id))
    name = getattr(runner.branches[branch_id], "name", branch_id)

    if pcfg is None:
        return {
            "branch_id": branch_id, "branch_name": name,
            "has_price_leg": False, "price_channel_suppressed": False,
            "rho_property": None, "reason_code": "no_price_leg",
            "economic_value": 0.0,
            "text": ("Ветка чисто техническая: ценовой ноги нет "
                     "(set_branch_cost не задан) — денежного VoI-канала нет, "
                     "стоп идёт на технических ногах (И-2)."),
        }

    rho = pcfg["rho_property"]
    economic_value: Optional[float] = None
    value_error: Optional[str] = None
    if compute_value and getattr(runner, "surrogates", None):
        try:
            economic_value = float(
                runner.branch_economic_value(branch_id, **econ_kwargs))
        except Exception as exc:  # noqa: BLE001 — объяснение не должно падать
            value_error = str(exc)

    if suppressed:
        reason = "rho_optimized_zeroed"
        text = (
            f"ρ-отклик «{rho}» носит роль OPTIMIZED (цель И питает цену). "
            "По И-5/Гр-1 ценовой σ_ρ-разведочный канал ЗАНУЛЁН (α≡0): вся "
            "неопределённость ρ уже оправдана качественной ногой d_ρ, засчитывать "
            "её ещё и деньгами — двойной счёт одной δρ. Денежная нога раунда = 0. "
            "Ценовая выгода остаётся лишь от детерминированного ВЫБОРА состава "
            "(price_состав·μ_ρ) — это технический рычаг desirability, не денежный "
            "VoI-гейт.")
    else:
        reason = "price_input_alive"
        tail = (f" Денежная ценность раунда ≈ {economic_value:,.0f} ₽ за горизонт."
                if economic_value is not None else "")
        text = (
            f"ρ-отклик «{rho}» носит роль PRICE_INPUT (питает цену, не цель). "
            "Ценовой σ_ρ-канал ЖИВОЙ (ALIVE): разведка ρ удешевляет изделие "
            "(price_изд = price_состав·ρ) и засчитывается деньгами." + tail)

    out: Dict[str, Any] = {
        "branch_id": branch_id, "branch_name": name,
        "has_price_leg": True, "price_channel_suppressed": suppressed,
        "rho_property": rho, "reason_code": reason,
        "economic_value": economic_value, "text": text,
    }
    if value_error is not None:
        out["economic_value_error"] = value_error
    return out


# ----------------------------------------------------------------------
# Сводка кампании (для MCP/assistant-контекста и обзорного UI)
# ----------------------------------------------------------------------
def campaign_overview(runner, *, with_money: bool = False,
                      coverage_fraction: float = DEFAULT_COVERAGE_FRACTION,
                      **money_kwargs: Any) -> Dict[str, Any]:
    """JSON-сериализуемая сводка кампании: общая база + ветки с ролями (read-only).

    Для каждой ветки — статус/бюджет/d_best, обратный индекс ролей, флаг
    занулённого денежного канала (И-5) и список откликов с низким покрытием (§6).
    ``with_money`` добавляет объяснение денежного канала (см.
    :func:`branch_money_explanation`; ``**money_kwargs`` пробрасываются туда).
    Питает и UI-обзор, и контекст ассистента/MCP (§16.1).
    """
    points = list(getattr(runner, "points", []) or [])
    branches: List[Dict[str, Any]] = []
    for bid in getattr(runner, "branches", {}):
        br = runner.branches[bid]
        rep = branch_role_report(runner, bid, coverage_fraction=coverage_fraction)
        item: Dict[str, Any] = {
            "id": bid,
            "name": rep["branch_name"],
            "status": getattr(br, "status", None),
            "d_best": float(getattr(br, "d_best", 0.0)),
            "budget": int(getattr(br, "budget", 0)),
            "spent": int(getattr(br, "spent", 0)),
            "remaining": int(br.remaining()) if hasattr(br, "remaining") else None,
            "by_role": rep["by_role"],
            "has_price_leg": rep["has_price_leg"],
            "rho_property": rep["rho_property"],
            "price_channel_suppressed": rep["price_channel_suppressed"],
            "low_coverage_responses": [r["response"] for r in rep["responses"]
                                       if r["low_coverage"]],
        }
        if with_money:
            item["money"] = branch_money_explanation(runner, bid, **money_kwargs)
        branches.append(item)

    return {
        "property_names": list(runner.property_names),
        "n_points": len(points),
        "origin_counts": (runner.origin_counts()
                          if hasattr(runner, "origin_counts") else {}),
        "coverage_fraction_threshold": float(coverage_fraction),
        "branches": branches,
    }


# ----------------------------------------------------------------------
# §17.3 (Ш2) — Валидация «не хватает данных» ПЕРЕД пересчётом/argmax/стопом
# ----------------------------------------------------------------------
# A0.6 / чистота проводника: система НЕ считает молча на дырах. Перед любым
# пересчётом (re-score / M8-argmax / §4-стоп) единая проверка называет, ЧЕГО
# именно не хватает, и возвращает ``{ok, missing, text}``; UI показывает отказ и
# НЕ запускает пересчёт. Каждая ветка отказа именована (``code``) и воспроизводима
# (гейт-тест §17.7). Read-only: ничего не измеряет и не меняет (И-1).
READINESS_EMPTY_OBJECTIVE = "empty_objective"
READINESS_UNMEASURED_GOAL = "unmeasured_goal_properties"
READINESS_PRICE_LEG_INCOMPLETE = "price_leg_incomplete"
READINESS_MIGRATION_PENDING = "migration_pending"


def validate_branch_ready(runner, branch_id: str) -> Dict[str, Any]:
    """Готова ли ветка к пересчёту (§17.3) — единая проверка «не хватает данных».

    Возвращает ``{ok, branch_id, missing, text}``. ``missing`` — список
    ``{code, responses, text}`` по каждой обнаруженной дыре; ``ok = not missing``.
    Проверяются ЧЕТЫРЕ ветки отказа §17.3 (все накапливаются, а не первая-же):

      1. **пустой объектив** — у ветки нет ни одной цели ⇒ desirability/argmax и
         §4-стоп не определены;
      2. **недомеренные свойства** — свойства целей (и ρ ценовой ноги) без единого
         конечного измерения в ОБЩЕЙ базе (``runner.points`` не урезается, И-1);
      3. **неполная ценовая нога** — ``set_branch_cost`` объявлен, но ρ / функция
         цены состава / десирабилити цены отсутствуют;
      4. **несмигрированные точки** — в базе есть точки без валидной миграции к
         текущей схеме (§16.2: добавили ось/компонент без применимой политики).

    Read-only (A0.6): не измеряет и не меняет состояние. Пункт 4 читает сигнал
    ядра (``_migrated_points`` → ``RuntimeError``) и переводит его в мягкий отказ
    UI вместо сырого исключения.
    """
    if branch_id not in getattr(runner, "branches", {}):
        raise KeyError(f"Нет ветки '{branch_id}'.")
    br = runner.branches[branch_id]
    goal = dict(getattr(br, "goal", None) or {})
    prop_names = list(runner.property_names)
    cfg = (getattr(runner, "_branch_cost", {}) or {}).get(branch_id)
    missing: List[Dict[str, Any]] = []

    # (1) пустой объектив — без целей argmax/стоп/re-score не определены
    if not goal:
        missing.append({
            "code": READINESS_EMPTY_OBJECTIVE,
            "responses": [],
            "text": ("У ветки нет ни одной цели: desirability/argmax и §4-стоп "
                     "не определены — задайте хотя бы одну цель."),
        })

    # (3) ценовая нога объявлена, но неполна
    rho = None
    if cfg is not None:
        rho = cfg.get("rho_property")
        problems: List[str] = []
        if not rho or rho not in prop_names:
            problems.append("ρ-свойство не задано или не среди свойств оракула")
        if not callable(cfg.get("price_fn")):
            problems.append("функция цены состава (price_fn) не задана")
        if cfg.get("cost_spec") is None:
            problems.append("десирабилити цены (cost_spec) не задана")
        if problems:
            missing.append({
                "code": READINESS_PRICE_LEG_INCOMPLETE,
                "responses": ([rho] if rho else []),
                "text": ("Ценовая нога объявлена, но неполна: "
                         + "; ".join(problems) + "."),
            })

    # (2) нет измеренных откликов по свойствам целей (+ ρ ценовой ноги)
    required = set(goal)
    if rho and rho in prop_names:
        required.add(rho)
    if required:
        cov = response_coverage(runner)
        unmeasured = sorted(r for r in required
                            if int(cov.get(r, {}).get("measured", 0)) == 0)
        if unmeasured:
            missing.append({
                "code": READINESS_UNMEASURED_GOAL,
                "responses": unmeasured,
                "text": ("Нет ни одного измерения в общей базе по свойствам: "
                         + ", ".join(unmeasured)
                         + " — внесите Y (§17.2) до пересчёта."),
            })

    # (4) несмигрированные точки (§16.2): ядро сигналит сбой миграции RuntimeError
    try:
        runner._migrated_points()
    except RuntimeError as exc:
        missing.append({
            "code": READINESS_MIGRATION_PENDING,
            "responses": [],
            "text": ("В базе есть точки без валидной миграции к текущей схеме "
                     f"(§16.2): {exc}"),
        })

    ok = not missing
    if ok:
        text = "Ветка готова к пересчёту: данных достаточно."
    else:
        text = ("Не хватает данных для пересчёта:\n"
                + "\n".join(f"• {m['text']}" for m in missing))
    return {"ok": ok, "branch_id": branch_id, "missing": missing, "text": text}


# ======================================================================
# ШАГ 2 — Контроллер кампании: обратимые мутации (§4) + смена роли (§5) +
# undo-стек (§7). Поверх того же MixtureProcessRunner. Канон:

#
#   * Мутации меняют ОЦЕНКУ/АКТИВНЫЙ ПУЛ/ОБЪЕКТИВ, НЕ измеренную правду (И-1):
#     история (``runner.points``) не урезается ни одной операцией (П-11).
#   * Роль ВЫВОДИТСЯ из намерения (goal + ценовая нога), не хранится отдельно
#     (нет дубля состояния). Поэтому «смена роли» — это операция над намерением:
#     PRICE_INPUT→OPTIMIZED добавляет отклик в goal (нужен DesirabilitySpec),
#     OPTIMIZED→PRICE_INPUT убирает из goal (отклик остаётся ρ ценовой ноги).
#     Переключение OPTIMIZED↔PRICE_INPUT переключает денежный канал ρ
#     (И-5/Гр-1: ZEROED↔ALIVE) ТОЛЬКО в этой ветке (Тр-5.5/Гр-3).
#   * Каждая мутация: snapshot намерения → применить → per-branch RE-SCORE →
#     показать смещение M8-рекомендации x* (X→Y, Тр-4.2/Тр-5.3) → запись в undo.
#   * Атомарность В ПРЕДЕЛАХ ВЕТКИ (Тр-5.4): тег/объектив/оценка вместе; ДРУГИЕ
#     ветки не затрагиваются (общий пул разделяет измеренную правду, не атрибуцию).
#   * Undo (§7) откатывает ТОЛЬКО обратимую ИНТЕРПРЕТАЦИЮ (роли/веса/форма/цель).
#     Прогон раунда (новые измерения) в стек НЕ входит и ОБНУЛЯЕТ его (Тр-7.2/7.3):
#     дно стека = последний снятый раунд (измеренную правду откатить нельзя, И-1).
# ======================================================================
# Роли, между которыми разрешён осознанный switch в текущей итерации (REFERENCE —
# backlog §10): денежный канал ρ переключается именно между этими двумя.
SWITCHABLE_ROLES = (ROLE_OPTIMIZED, ROLE_PRICE_INPUT)

# Лёгкие параметры M8-argmax для показа смещения рекомендации x* (вызывается на
# каждую мутацию — глубокий мультистарт не нужен, точка не измеряется).
_XOPT_KW = dict(n_candidates=300, refine_iters=60, n_starts=2)


class CampaignController:
    """Обратимые мутации намерения ветки + undo поверх ``MixtureProcessRunner``.

    НЕ владеет данными — оборачивает существующий runner (одна модель физики на
    проект, общий пул, §5/§12). Все read-методы делегируют ШАГ-1 view-model.
    """

    def __init__(self, runner):
        self.runner = runner
        # стек снимков намерения для undo (§7); дно сбрасывается прогоном раунда.
        self._undo: List[Dict[str, Any]] = []

    # -- read-model (ШАГ 1) passthrough -------------------------------
    def role_report(self, branch_id: str, **kw) -> Dict[str, Any]:
        return branch_role_report(self.runner, branch_id, **kw)

    def money_explanation(self, branch_id: str, **kw) -> Dict[str, Any]:
        return branch_money_explanation(self.runner, branch_id, **kw)

    def overview(self, **kw) -> Dict[str, Any]:
        return campaign_overview(self.runner, **kw)

    # -- §17.3 (Ш2) валидация готовности ветки к пересчёту (read-only) --
    def validate_ready(self, branch_id: str) -> Dict[str, Any]:
        """§17.3: единая проверка «не хватает данных» ПЕРЕД пересчётом/argmax/стопом.

        Тонкий проброс в :func:`validate_branch_ready` (read-only, A0.6). UI обязан
        вызывать её перед :meth:`run_round`/:meth:`commit_measured`/argmax и НЕ
        запускать пересчёт при ``ok == False`` — вместо молчаливого счёта на дырах
        показать перечень недостающего (``missing``/``text``)."""
        return validate_branch_ready(self.runner, branch_id)

    # -- snapshot / restore намерения ветки (для undo) ----------------

    def _snapshot(self, branch_id: str) -> Dict[str, Any]:
        br = self.runner.branches[branch_id]
        cost = (self.runner._branch_cost.get(branch_id)
                if hasattr(self.runner, "_branch_cost") else None)
        return {
            "goal": {k: replace(v) for k, v in (br.goal or {}).items()},
            "d_best": float(br.d_best),
            "x_best": (list(br.x_best) if br.x_best is not None else None),
            "cost": (dict(cost) if cost is not None else None),
        }

    def _restore(self, branch_id: str, snap: Dict[str, Any]) -> None:
        br = self.runner.branches[branch_id]
        br.goal = {k: replace(v) for k, v in snap["goal"].items()}
        br.d_best = float(snap["d_best"])
        br.x_best = (list(snap["x_best"]) if snap["x_best"] is not None else None)
        if hasattr(self.runner, "_branch_cost"):
            if snap["cost"] is not None:
                self.runner._branch_cost[branch_id] = dict(snap["cost"])
            else:
                self.runner._branch_cost.pop(branch_id, None)

    # -- per-branch re-score (оценка под текущий объектив, не правда) --
    def _rescore(self, branch_id: str) -> None:
        """Пересчитать ``d_best``/``x_best`` ветки по ОБЩЕЙ базе под ТЕКУЩИЙ
        объектив (goal + цена). Меняется ОЦЕНКА, не измеренные Y (И-1)."""
        runner = self.runner
        br = runner.branches[branch_id]
        if not br.goal or runner.X is None or runner.Y is None \
                or len(runner.X) == 0:
            return
        specs = dict(br.goal)
        meas = {n: np.asarray(runner.Y[:, runner.prop_index[n]], float)
                for n in br.goal}
        cfg = (runner._branch_cost.get(branch_id)
               if hasattr(runner, "_branch_cost") else None)
        if cfg is not None:
            pc = np.asarray(cfg["price_fn"](runner.X), float).ravel()
            rho = np.asarray(runner.Y[:, runner.prop_index[cfg["rho_property"]]],
                             float)
            meas[cfg["cost_name"]] = pc * rho
            specs[cfg["cost_name"]] = cfg["cost_spec"]
        d = np.asarray(Desirability(specs).overall(meas), float).ravel()
        if d.size == 0:
            return
        bi = int(np.argmax(d))
        br.d_best = float(d[bi])
        br.x_best = runner._to_full(runner.X[bi]).tolist()
        br.refresh_status()

    def _x_opt(self, branch_id: str) -> Optional[List[float]]:
        """M8-argmax рецепт ветки (рекомендация x*) или ``None``, если не считается
        (пустой goal / нет суррогатов). Read-only: точка НЕ измеряется."""
        br = self.runner.branches[branch_id]
        if not br.goal or not getattr(self.runner, "surrogates", None):
            return None
        try:
            res = self.runner.optimize_xbest(branch_id, **_XOPT_KW)
            return [round(float(v), 4) for v in np.asarray(res.x, float).ravel()]
        except Exception:  # noqa: BLE001 — рекомендация необязательна
            return None

    # -- общий каркас обратимой мутации -------------------------------
    def _apply(self, op: str, branch_id: str, response: Optional[str],
               mutate_fn) -> Dict[str, Any]:
        if branch_id not in self.runner.branches:
            raise KeyError(f"Нет ветки '{branch_id}'.")
        n_hist_before = len(getattr(self.runner, "points", []) or [])
        role_before = (self.runner.response_role(branch_id, response)
                       if response else None)
        x_before = self._x_opt(branch_id)
        d_before = float(self.runner.branches[branch_id].d_best)

        snap = self._snapshot(branch_id)
        mutate_fn()                               # применяем намерение
        self._rescore(branch_id)                  # переоценка под новый объектив
        self._undo.append({"op": op, "branch_id": branch_id, "snap": snap})

        x_after = self._x_opt(branch_id)
        role_after = (self.runner.response_role(branch_id, response)
                      if response else None)
        # И-1/П-11: ни одна мутация не урезает историю.
        assert len(getattr(self.runner, "points", []) or []) == n_hist_before
        return self._result(op, branch_id, response, role_before, role_after,
                            d_before, x_before, x_after)

    def _result(self, op, branch_id, response, role_before, role_after,
                d_before, x_before, x_after) -> Dict[str, Any]:
        shift = None
        if (x_before is not None and x_after is not None
                and len(x_before) == len(x_after)):
            shift = float(np.linalg.norm(np.asarray(x_after)
                                         - np.asarray(x_before)))
        return {
            "op": op,
            "branch_id": branch_id,
            "response": response,
            "role_before": role_before,
            "role_after": role_after,
            "price_channel_suppressed":
                bool(self.runner.price_channel_suppressed(branch_id)),
            "d_best_before": d_before,
            "d_best_after": float(self.runner.branches[branch_id].d_best),
            # Тр-4.2/Тр-5.3: «рекомендация сместилась X→Y» (M8-argmax x*).
            "x_opt_before": x_before,
            "x_opt_after": x_after,
            "recommendation_shift": shift,
            "undo_available": bool(self._undo),
        }

    # -- §4 ярус-1: веса целей ----------------------------------------
    def set_weights(self, branch_id: str,
                    weights: Dict[str, float]) -> Dict[str, Any]:
        """Изменить веса целей ветки (ярус-1, re-score десирабилити)."""
        br = self.runner.branches[branch_id]
        for resp in weights:
            if resp not in (br.goal or {}):
                raise KeyError(f"'{resp}' не цель ветки '{branch_id}' "
                               f"(вес можно менять только у цели).")

        def _mut():
            for resp, w in weights.items():
                br.goal[resp] = replace(br.goal[resp], weight=float(w))

        return self._apply("set_weights", branch_id, None, _mut)

    # -- §4 ярус-1: форма десирабилити / TARGET / +цель над откликом ---
    def set_desirability(self, branch_id: str, response: str,
                         spec: DesirabilitySpec) -> Dict[str, Any]:
        """Задать/заменить десирабилити-спеку отклика (форма/TARGET; +цель).

        Если отклика ещё не было в goal — это +цель над ИЗМЕРЯЕМЫМ откликом
        (роль становится OPTIMIZED). Отклик обязан быть свойством оракула.
        """
        if response not in self.runner.property_names:
            raise KeyError(f"Отклик '{response}' не среди свойств оракула "
                           f"{list(self.runner.property_names)}.")
        br = self.runner.branches[branch_id]

        def _mut():
            br.goal[response] = spec

        return self._apply("set_desirability", branch_id, response, _mut)

    # -- §4 ярус-1: удалить цель --------------------------------------
    def delete_goal(self, branch_id: str, response: str) -> Dict[str, Any]:
        """Удалить цель из ветки (нога выпадает; история цела). Запрещено удалять
        ПОСЛЕДНЮЮ цель — ветке нужен объектив (иначе re-score/argmax не определены).
        """
        br = self.runner.branches[branch_id]
        if response not in (br.goal or {}):
            raise KeyError(f"'{response}' не цель ветки '{branch_id}'.")
        if len(br.goal) <= 1:
            raise ValueError("Нельзя удалить последнюю цель ветки — объектив "
                             "должен существовать (смените цель вместо удаления).")

        def _mut():
            del br.goal[response]

        return self._apply("delete_goal", branch_id, response, _mut)

    # -- §5: смена роли (ветка × отклик), per-branch -------------------
    def switch_role(self, branch_id: str, response: str, to_role: str, *,
                    spec: Optional[DesirabilitySpec] = None) -> Dict[str, Any]:
        """Сменить роль ``response`` в ветке (§5) — переключает денежный канал ρ.

        Допустимы переходы между ``OPTIMIZED`` и ``PRICE_INPUT`` (REFERENCE —
        backlog). Предусловие: отклик — ρ ценовой ноги ветки (``feeds_price``),
        иначе денежного канала нет и переключать нечего (сначала задайте цену
        через ``set_branch_cost``). Переход:

          * ``PRICE_INPUT → OPTIMIZED`` — добавить отклик в goal (нужен ``spec``);
            канал ρ ZEROED (И-5/Гр-1);
          * ``OPTIMIZED → PRICE_INPUT`` — убрать отклик из goal (остаётся ρ
            ценовой ноги); канал ρ ALIVE.

        Атомарно в пределах ветки (Тр-5.4); другие ветки не затронуты (Гр-3).
        """
        if to_role not in SWITCHABLE_ROLES:
            raise ValueError(f"Сменить роль можно только между "
                             f"{SWITCHABLE_ROLES} (REFERENCE — backlog §10).")
        if branch_id not in self.runner.branches:
            raise KeyError(f"Нет ветки '{branch_id}'.")
        pcfg = branch_price_config(self.runner, branch_id)
        if pcfg is None or pcfg["rho_property"] != response:
            raise ValueError(
                f"'{response}' не ρ ценовой ноги ветки '{branch_id}': денежного "
                f"канала нет, переключать роль нечем. Сначала задайте цену "
                f"(set_branch_cost с rho_property='{response}').")
        role_now = self.runner.response_role(branch_id, response)
        if role_now == to_role:
            raise ValueError(f"Отклик '{response}' уже в роли '{to_role}'.")
        br = self.runner.branches[branch_id]

        if to_role == ROLE_OPTIMIZED:
            if spec is None:
                raise ValueError(
                    "Переход PRICE_INPUT→OPTIMIZED требует DesirabilitySpec "
                    "(вид/диапазон/вес цели для этого отклика).")

            def _mut():
                br.goal[response] = spec
        else:  # ROLE_PRICE_INPUT — убрать из goal, ρ остаётся ценовой ногой
            def _mut():
                br.goal.pop(response, None)

        return self._apply("switch_role", branch_id, response, _mut)

    # -- undo (§7) ----------------------------------------------------
    def can_undo(self) -> bool:
        return bool(self._undo)

    def undo(self) -> Dict[str, Any]:
        """Откатить последнюю обратимую мутацию (§7). Откатывает ИНТЕРПРЕТАЦИЮ
        (роль/веса/форму/цель), НЕ измеренную правду. Если стек пуст (дно =
        последний снятый раунд) — :class:`IndexError`."""
        if not self._undo:
            raise IndexError("Стек undo пуст: дно — последний снятый раунд "
                             "(измеренную правду откатить нельзя, И-1).")
        entry = self._undo.pop()
        bid = entry["branch_id"]
        self._restore(bid, entry["snap"])
        self._rescore(bid)
        return {
            "op": "undo", "undone": entry["op"], "branch_id": bid,
            "price_channel_suppressed":
                bool(self.runner.price_channel_suppressed(bid)),
            "d_best": float(self.runner.branches[bid].d_best),
            "undo_available": bool(self._undo),
        }

    # -- прогон раунда: ОБНУЛЯЕТ undo-стек (Тр-7.2/7.3) ----------------
    def run_round(self, branch_id: str, **kw) -> Dict[str, Any]:
        """Прогнать раунд ветки и ЗАПЕЧАТАТЬ дно undo: измеренная правда
        неоткатываема (И-1), стек обратимых настроек обнуляется (Тр-7.2/7.3)."""
        out = self.runner.run_branch_round(branch_id, **kw)
        self._undo.clear()
        return out

    def run_portfolio_round(self, total_slots: int, **kw) -> Dict[str, Any]:
        """Портфельный раунд (арбитр бюджета) + запечатывание дна undo."""
        out = self.runner.run_portfolio_round(total_slots, **kw)
        self._undo.clear()
        return out

    # -- §17.2 (Ш1) ручной оракул: предложить (read-only) → зафиксировать Y --
    def propose_points(self, branch_id: str, n_points: int = 2,
                       **kw) -> Any:
        """§17.2: предложить точки ветки БЕЗ измерения (read-only, база не меняется).

        Тонкий проброс в :meth:`MixtureProcessRunner.propose_points`. Ни общая база
        (``runner.points``), ни undo-стек НЕ трогаются: это лишь предложение
        кандидатов, которое пользователь затем измеряет и фиксирует через
        :meth:`commit_measured` (первая половина ручного цикла, A0.6).
        """
        return self.runner.propose_points(branch_id, n_points=n_points, **kw)

    def commit_measured(self, branch_id: str, X: Any, Y: Any) -> Dict[str, Any]:
        """§17.2: зафиксировать ВНЕСЁННЫЕ Y предложенных точек ветки.

        Доливает измеренные точки в ОБЩУЮ базу (origin=branch:{id}, И-1) через
        :meth:`MixtureProcessRunner.commit_measured` и ЗАПЕЧАТЫВАЕТ дно undo:
        измеренная правда откату не подлежит (как :meth:`run_round`, Тр-7.2/7.3).
        Вторая половина ручного цикла «предложить → зафиксировать Y».
        """
        out = self.runner.commit_measured(branch_id, X, Y)
        self._undo.clear()
        return out

    # ------------------------------------------------------------------
    # §16.2 — Фасад эволюции схемы кампании (штатная операция живого проекта)
    #
    # Тонкая обёртка над runner.augment_phase_*/move_region/evolve_schema:
    # выводит эволюцию схемы (добавление компонента смеси / процесс-переменной /
    # отклика, движение границ) в ЯВНЫЙ контракт кампании (ТЗ §16.2) и жёстко
    # требует политику миграции старых точек (A0.6 — миграция НЕ молча). Одна
    # модель физики (канон §5/§12), общая база НЕ урезается (И-1): всё делает
    # ядро; контроллер валидирует и делегирует. Эволюция схемы — структурная
    # веха: как прогон раунда, она ЗАПЕЧАТЫВАЕТ дно undo (обратимая
    # интерпретация до эволюции не откатывается сквозь смену версии, Тр-7.2/7.3).
    # Переменная/компонент обязаны быть объявлены в ПОЛНОЙ схеме проекта — модель
    # «прогрессивного раскрытия» (append совсем новой оси вне ядра, §16.6).
    # ------------------------------------------------------------------
    @staticmethod
    def _require_migration_policy(name: str, migration: Any) -> Dict[str, Any]:
        """A0.6: политика миграции старых точек ОБЯЗАНА быть задана явно.

        Принимает dict одной из политик ``schema_evolution`` (``known_constant``/
        ``unknown``/``recompute``). Молчаливого дефолта нет — добавление
        переменной без явной политики отвергается.
        """
        if not isinstance(migration, dict) or "policy" not in migration:
            raise ValueError(
                f"A0.6: добавление '{name}' требует ЯВНОЙ политики миграции "
                f"старых точек (known_constant(v) / unknown() / recompute(fn)) — "
                f"молчаливой миграции нет.")
        return dict(migration)

    def add_process_var(self, name: str, migration: Dict[str, Any], *,
                        lower: Optional[float] = None,
                        upper: Optional[float] = None) -> Any:
        """§16.2: APPEND process-переменной как ШТАТНАЯ операция кампании (v+1).

        Делегирует ``runner.augment_phase_schema``. ``migration`` обязателен
        (A0.6): политика для старых точек, мерившихся БЕЗ этой оси (обычно
        ``known_constant(baseline)``). ``lower``/``upper`` переопределяют границы
        (иначе — из полной схемы). Общая база не урезается (И-1); версия растёт.
        """
        proc = (list(self.runner._full_proc.names)
                if getattr(self.runner, "_full_proc", None) else [])
        if name not in proc:
            raise KeyError(
                f"process-переменная '{name}' не объявлена в полной схеме {proc} "
                f"— фасад раскрывает объявленные оси (append новой вне ядра, §16.6).")
        mig = self._require_migration_policy(name, migration)
        bounds = ({name: (float(lower), float(upper))}
                  if lower is not None and upper is not None else None)
        out = self.runner.augment_phase_schema([name], migration={name: mig},
                                               bounds=bounds)
        self._undo.clear()
        return out

    def add_mixture_component(self, name: str,
                              migration: Optional[Dict[str, Any]] = None, *,
                              lower: Optional[float] = None,
                              upper: Optional[float] = None) -> Any:
        """§16.2: APPEND mixture-компонента как ШТАТНАЯ операция кампании (v+1).

        Делегирует ``runner.augment_phase_mixture``. Σ переопределяется
        (``A+B=1 → A+B+C=1``). Единственная Σ-совместимая политика миграции
        старых точек — ``known_constant(0.0)`` (грань симплекса C=0), поэтому при
        ``migration=None`` берётся именно она; иное отвергается (дефолт — не
        «молчаливый средний», а физически вынужденная грань, §15.0.4).
        """
        mix = (list(self.runner._full_mix.names)
               if getattr(self.runner, "_full_mix", None) else [])
        if name not in mix:
            raise KeyError(
                f"mixture-компонент '{name}' не объявлен в полной схеме {mix} — "
                f"фасад раскрывает объявленные компоненты (append нового вне ядра, "
                f"§16.6).")
        if migration is None:
            mig = known_constant(0.0)
        else:
            mig = self._require_migration_policy(name, migration)
            if not (mig.get("policy") == KNOWN_CONSTANT
                    and float(mig.get("value", 1.0)) == 0.0):
                raise ValueError(
                    f"mixture-append '{name}': единственная Σ-совместимая "
                    f"политика — known_constant(0.0) (грань симплекса C=0, "
                    f"§15.0.4); дано {mig}.")
        bounds = ({name: (float(lower), float(upper))}
                  if lower is not None and upper is not None else None)
        out = self.runner.augment_phase_mixture([name], migration={name: mig},
                                                bounds=bounds)
        self._undo.clear()
        return out

    def add_response(self, spec) -> Any:
        """§16.2: ввести новый ОТКЛИК в схему (v+1); у старых точек Y[new]=MISSING.

        Обёртка над ``evolve_schema(add_responses=…)``: эволюционирует СХЕМУ
        (bump версии, change_log). Физические измерения даёт оракул
        (``property_names``), поэтому суррогаты здесь не переобучаются — новый
        отклик подхватится, когда оракул начнёт его отдавать (у исторических
        точек значение честно MISSING, суррогат учится только на измеренных,
        §13.7).
        """
        r = self.runner
        new = evolve_schema(r.current_schema, add_responses=[spec])
        r.schema_history.add(new)
        r.current_schema = new
        r.current_schema_version = int(new.version)
        self._undo.clear()
        return new

    def relax_bounds(self, var: str, lower: float, upper: float,
                     **kw) -> Any:
        """§16.2: РАСШИРИТЬ область интереса по ``var`` (region-move, БЕЗ bump).

        Делегирует ``runner.move_region`` (примитив move_bounds классифицирует
        relax/restrict и проверяет симплекс-замкнутость). Это НЕ append-переменной:
        состав схемы тот же, версия не растёт (§15.2.4). hard-граница (A0.5) —
        :class:`RegionMoveError`. История цела; выпавшие точки восстановимы.
        """
        kw.setdefault("intent", "relax")
        out = self.runner.move_region({var: (float(lower), float(upper))}, **kw)
        self._undo.clear()
        return out

    def restrict_bounds(self, var: str, lower: float, upper: float,
                        **kw) -> Any:
        """§16.2: СУЗИТЬ область интереса по ``var`` (region-move, БЕЗ bump).

        Обёртка-близнец :meth:`relax_bounds` (тот же примитив; направление
        движения классифицирует move_bounds). Точки вне новой области легально
        исключаются из активного pool по ``policy`` (дефолт ``exclude``), но
        ОСТАЮТСЯ в истории (И-1). ``policy='error'`` запрещает потерю измеренной точки.
        """
        kw.setdefault("intent", "restrict")
        out = self.runner.move_region({var: (float(lower), float(upper))}, **kw)
        self._undo.clear()
        return out

    # -- §8: spawn ветки с НАСЛЕДОВАНИЕМ ролей -------------------------

    # Дефолт = наследование намерения родителя (валидный XOR копируется ⇒ ребёнок
    # честен by default, Тр-8.1). Наследование НЕ молчаливое: spawn отдаёт
    # обзорную сводку (Тр-8.1а) с пометкой «унаследовано как есть» vs «изменено
    # для ветки» (Тр-8.1б). Новый объектив ПЕРЕБИВАЕТ унаследованную роль
    # (Тр-8.1в): цель над ρ ⇒ ρ в ребёнке OPTIMIZED (канал цены ZEROED, И-5).
    def _merged_child_goal(self, parent_id: str,
                           new_goals: Optional[Dict[str, DesirabilitySpec]]
                           ) -> Dict[str, DesirabilitySpec]:
        """Цель ребёнка = копия цели родителя, перекрытая ``new_goals``."""
        parent = self.runner.branches[parent_id]
        goal = {k: replace(v) for k, v in (parent.goal or {}).items()}
        for resp, spec in (new_goals or {}).items():
            if resp not in self.runner.property_names:
                raise KeyError(f"Отклик '{resp}' не среди свойств оракула "
                               f"{list(self.runner.property_names)}.")
            goal[resp] = spec
        return goal

    def _spawn_summary(self, parent_id: str,
                       merged_goal: Dict[str, DesirabilitySpec],
                       new_goals: Optional[Dict[str, DesirabilitySpec]],
                       inherit_cost: bool) -> Dict[str, Any]:
        """Обзорная сводка ролей будущего ребёнка vs родитель (Тр-8.1а/б/в)."""
        parent_roles = self.runner.branch_roles(parent_id)
        pcfg = branch_price_config(self.runner, parent_id)
        rho = pcfg["rho_property"] if (pcfg and inherit_cost) else None
        child_suppressed = (rho is not None and rho in merged_goal)
        ng = set(new_goals or {})

        rows: List[Dict[str, Any]] = []
        any_changed = False
        for p in self.runner.property_names:
            rp = parent_roles[p]
            if p in merged_goal:
                rc = ROLE_OPTIMIZED
            elif rho is not None and p == rho:
                rc = ROLE_PRICE_INPUT
            else:
                rc = ROLE_REFERENCE
            overridden = p in ng
            role_changed = rp != rc
            if role_changed:
                change = "changed_by_objective"   # Тр-8.1в: объектив перебил роль
                any_changed = True
            elif overridden:
                change = "overridden_same_role"    # тронуто, но роль та же
            else:
                change = "inherited"               # унаследовано как есть
            rows.append({
                "response": p, "role_parent": rp, "role_child": rc,
                "overridden": overridden, "role_changed": role_changed,
                "change": change,
                "money_channel_child": _money_channel(p, rho, child_suppressed),
            })
        return {
            "parent_id": parent_id, "rho_property": rho,
            "inherit_cost": bool(inherit_cost),
            "any_role_changed_by_objective": any_changed,
            "responses": rows,
        }

    def preview_spawn(self, parent_id: str, *,
                      new_goals: Optional[Dict[str, DesirabilitySpec]] = None,
                      inherit_cost: bool = True) -> Dict[str, Any]:
        """Обзорная сводка наследования ролей БЕЗ создания ветки (Тр-8.1а).

        Кладёт перед пользователем, что унаследовано как есть и что перебито
        объективом ветки — для подтверждения «в один клик или правка». Ничего не
        меняет (A0.6)."""
        if parent_id not in self.runner.branches:
            raise KeyError(f"Нет родительской ветки '{parent_id}'.")
        merged = self._merged_child_goal(parent_id, new_goals)
        return self._spawn_summary(parent_id, merged, new_goals, inherit_cost)

    def spawn_branch(self, parent_id: str, child_name: str, *,
                     child_id: Optional[str] = None,
                     new_goals: Optional[Dict[str, DesirabilitySpec]] = None,
                     budget: Optional[int] = None,
                     satisfy_at: Optional[float] = None,
                     inherit_cost: bool = True) -> Dict[str, Any]:
        """Создать ветку-сиблинг на ОБЩЕМ пуле с наследованием ролей (§8).

        Цель = цель родителя, перекрытая ``new_goals`` (приоритет объектива,
        Тр-8.1в); ценовая нога наследуется (``inherit_cost``) — тот же ρ/цена, что
        делает ρ в ребёнке либо PRICE_INPUT (унаследовано), либо OPTIMIZED+ZEROED
        (если новый объектив ввёл цель над ρ). Возвращает ``review``-сводку
        (Тр-8.1а/б). Модель физики общая (канон §5/§12) — отдельной модели у
        ребёнка нет.
        """
        if parent_id not in self.runner.branches:
            raise KeyError(f"Нет родительской ветки '{parent_id}'.")
        parent = self.runner.branches[parent_id]
        merged = self._merged_child_goal(parent_id, new_goals)
        review = self._spawn_summary(parent_id, merged, new_goals, inherit_cost)

        child = self.runner.add_branch(
            child_name, merged,
            budget=int(budget if budget is not None else parent.budget),
            satisfy_at=float(satisfy_at if satisfy_at is not None
                             else parent.satisfy_at),
            branch_id=child_id)
        pcfg = (self.runner._branch_cost.get(parent_id)
                if hasattr(self.runner, "_branch_cost") else None)
        if inherit_cost and pcfg is not None:
            self.runner.set_branch_cost(
                child.id, pcfg["price_fn"], pcfg["cost_spec"],
                rho_property=pcfg["rho_property"], cost_name=pcfg["cost_name"])
        return {
            "op": "spawn", "parent_id": parent_id,
            "child_id": child.id, "child_name": child.name,
            "price_channel_suppressed":
                bool(self.runner.price_channel_suppressed(child.id)),
            "review": review,
        }
