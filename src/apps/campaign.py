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

from typing import Any, Dict, List, Optional

import numpy as np

from ..design.branches import (ROLE_OPTIMIZED, ROLE_PRICE_INPUT, ROLE_REFERENCE,
                                ROLE_PRIORITY)

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
