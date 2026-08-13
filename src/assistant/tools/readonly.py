"""assistant/tools/readonly.py — read-only инструменты ассистента (iter61).

Закрывают главный класс отказа помощника: «работа по памяти». Каждый ответ про
геометрию кампании должен приходить ИЗ ЯДРА — из активной `PhrSpec`, из
`point_report`, из `preflight`, из общей базы точек, — а не из пересказа
предыдущих сообщений.

Инструменты (ASSISTANT_SPEC §3.1):

* ``get_spec`` — спека, ``spec_hash``, ``group_order``, dim; закрывает работу
  по устаревшему снимку;
* ``explain_node`` — роль, эффективные границы, ЧТО именно ограничивает узел
  сейчас (в т.ч. немонотонная ``hi_φ(T)``); закрывает «почему диапазон не
  такой, как я ввёл»;
* ``validate_spec`` — dry-run патча: ошибки конструктора, дифф границ, поедет
  ли отпечаток;
* ``simulate_bounds`` — что поедет по ЧИСЛАМ (Σphr, корреляции, диапазоны) без
  генерации плана; закрывает «клин vs трапеция» и неверный референс;
* ``preflight`` — cond/VIF/corr/покрытие/провалившиеся гейты вместо статистики
  «на глаз»;
* ``get_runs`` / ``campaign_overview`` — фактические прогоны, отклики,
  ковариаты, ветки;
* ``point_report`` / ``encode_recipe`` — разбор конкретного рецепта (импорт
  серийной рецептуры как anchor);
* ``get_local_facts`` / ``get_decisions`` / ``list_attachments`` /
  ``read_attachment`` — L1-знание, история решений компании и документы.

Все они НИЧЕГО не меняют: спека, база и файлы только читаются.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..files import attachment_text, find_attachment
from ..store import read_log
from .registry import READONLY, ToolContext, ToolError, register

#: Сколько кандидатов разыгрывать в ``simulate_bounds`` по умолчанию.
DEFAULT_SIM_N = 400

_SHARE_ROLES = ("SHARE_FREE", "SHARE_CLOSURE", "SHARE_SIMPLEX", "SHARE_OF")


# ----------------------------------------------------------------------
# Работа со спекой как с данными
# ----------------------------------------------------------------------
def spec_payload(spec) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """``to_dicts()`` → (узлы, мета-обёртка).

    Схема v2 может приходить обёрткой (``spec_version``/``group_order``/
    ``nodes``), v1 — плоским списком; инструментам нужен один вид.
    """
    dicts = spec.to_dicts()
    if isinstance(dicts, dict):
        meta = {k: v for k, v in dicts.items() if k != "nodes"}
        return list(dicts.get("nodes", [])), meta
    return list(dicts), {}


def _nodes_by_name(nodes: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(d.get("name")): dict(d) for d in nodes}


def _f(x: Any) -> Any:
    """numpy → python (ответ инструмента обязан быть JSON-сериализуемым)."""
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, np.ndarray):
        return [_f(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {str(k): _f(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_f(v) for v in x]
    return x


def normalize_patch(patch: Any) -> List[Dict[str, Any]]:
    """Патч в любой из принятых форм → список операций ``{node, set, unset}``.

    Принимаем то, что естественно порождает модель:

    * ``{"node": "UV", "field": "range", "value": [0.05, 0.2]}``;
    * ``{"node": "UV", "set": {...}, "unset": ["cap_to"]}``;
    * ``{"UV": {"range": [...]}, "DINP": {"range": [...]}}``;
    * список любого из перечисленного.

    Спека целиком правится через узел ``__spec__`` (например
    ``{"node": "__spec__", "set": {"group_order": [...]}}``).
    """
    if patch is None:
        return []
    if isinstance(patch, (list, tuple)):
        out: List[Dict[str, Any]] = []
        for p in patch:
            out.extend(normalize_patch(p))
        return out
    if not isinstance(patch, dict):
        raise ToolError("Патч должен быть объектом или списком объектов, "
                        f"получено: {type(patch).__name__}.")

    if "node" in patch:
        node = str(patch["node"])
        set_ = dict(patch.get("set") or {})
        unset = [str(u) for u in (patch.get("unset") or [])]
        if "field" in patch:
            if "value" not in patch:
                raise ToolError("В патче есть 'field', но нет 'value': "
                                "нечего записывать в поле узла.")
            set_[str(patch["field"])] = patch["value"]
        if not set_ and not unset:
            raise ToolError(f"Патч узла '{node}' пуст: нужен 'set' "
                            f"(или пара 'field'/'value') либо 'unset'.")
        return [{"node": node, "set": set_, "unset": unset}]

    # форма {"узел": {поля}}
    ops: List[Dict[str, Any]] = []
    for name, fields in patch.items():
        if not isinstance(fields, dict):
            raise ToolError(
                f"Патч узла '{name}' должен быть объектом полей, получено "
                f"{type(fields).__name__}. Используйте "
                f"{{'node': '{name}', 'field': ..., 'value': ...}}.")
        ops.append({"node": str(name), "set": dict(fields), "unset": []})
    return ops


def apply_patch_to_dicts(spec, patch: Any) -> Any:
    """Применить патч к СЕРИАЛИЗАЦИИ спеки (сама спека не трогается).

    Возвращает объект того же вида, что ``to_dicts()``, — его можно отдать
    ``PhrSpec.from_dicts`` для dry-run. Патч НЕ применяется к проекту: это
    задача write-инструмента с подтверждением человека (ASSISTANT_SPEC §2).
    """
    nodes, meta = spec_payload(spec)
    nodes = copy.deepcopy(nodes)
    meta = dict(meta)
    index = {str(d.get("name")): i for i, d in enumerate(nodes)}

    for op in normalize_patch(patch):
        name = op["node"]
        if name in ("__spec__", "spec", "*"):
            for k, v in op["set"].items():
                meta[str(k)] = v
            for k in op["unset"]:
                meta.pop(str(k), None)
            meta.setdefault("spec_version", 2)
            continue
        if name not in index:
            raise ToolError(
                f"Узла '{name}' нет в спеке. Есть: {sorted(index)}. "
                f"Добавление/удаление узлов этим патчем не делается — это "
                f"смена схемы проекта (эволюция), а не правка границы.")
        node = nodes[index[name]]
        for k, v in op["set"].items():
            node[str(k)] = v
        for k in op["unset"]:
            node.pop(str(k), None)

    if meta:
        meta.setdefault("spec_version", 2)
        return {**meta, "nodes": nodes}
    return nodes


def build_patched_spec(spec, patch: Any):
    """Копия спеки с применённым патчем (конструктор валидирует)."""
    from ...design.phr_sampler import PhrSpec  # локально: тяжёлый импорт
    return PhrSpec.from_dicts(apply_patch_to_dicts(spec, patch))


# ----------------------------------------------------------------------
# spec_schema (iter71)
# ----------------------------------------------------------------------
#: Пример-минимум ВАЛИДНОГО пакета: по одному представителю каждого класса
#: узлов (тотал + пара долей при k=2, техлимиты, лог-ось, cap-фаза, fixed).
#: Модель, которой формат пересказан словами, восстанавливает его по памяти о
#: markdown-таблицах и промахивается ключами; пример ИЗ ЯДРА — единственный
#: способ передать формат без искажения.
SPEC_EXAMPLE: Dict[str, Any] = {
    "spec_version": 2,
    "group_order": ["SOFT"],
    "nodes": [
        {"name": "RESIN", "role": "FIXED", "value": 100.0},
        {"name": "DINP", "role": "ABSOLUTE", "range": [4.0, 14.0]},
        {"name": "ESO", "role": "FIXED", "value": 2.5},
        {"name": "SOFT", "role": "GROUP_TOTAL", "range": [3.0, 15.0],
         "members": ["CPE", "PBNK"]},
        {"name": "CPE", "role": "SHARE_CLOSURE", "group": "SOFT",
         "min_phr": 3.0},
        {"name": "PBNK", "role": "SHARE_FREE", "group": "SOFT",
         "share_range": [0.0, 0.70], "max_phr": 8.0},
        {"name": "TiO2", "role": "ABSOLUTE", "range": [0.3, 8.0],
         "scale": "log"},
        {"name": "UV", "role": "ABSOLUTE_CAPPED", "range": [0.05, 0.30],
         "scale": "log", "cap_to": ["DINP", "ESO"], "cap_ratio": 0.03},
    ],
}

#: Чего в спеке НЕТ, хотя туда постоянно пытаются положить. Каждый пункт —
#: не запрет ради запрета, а указание, где эта сущность живёт на самом деле.
SPEC_NOT_HERE: Dict[str, str] = {
    "levels": "число уровней оси — политика ПЛАНА, не геометрии: для "
              "process-осей задаётся в сетапе («rotor_rpm: 400, 900» → "
              "runner.set_process_levels), для состава уровней нет вовсе.",
    "premix": "премикс не флаг узла, а ВЫВОД из разрешения весов: "
              "premix_required(delta_phr, lo, hi) — см. point_report "
              "(аргумент delta_phr) и разрешение весов в паспорте кампании.",
    "process": "process-оси (температуры, обороты) — отдельный блок схемы; в "
               "phr-спеке их нет, она описывает только состав.",
    "components/groups": "узлы НЕ делятся на «компоненты» и «группы»: это ОДИН "
                         "плоский список 'nodes', где тотал группы — такой же "
                         "узел с role GROUP_TOTAL(_FIXED) и 'members'.",
    "version": "метка версии кампании живёт в паспорте (campaign_label); в "
               "обёртке спеки допустим только 'spec_version': 2.",
    "lo/hi/of/to": "legacy-ключи схемы v1: в v2 это 'range'/'share_range' "
                   "(пара [lo, hi]) и 'group'/'reference'.",
}


@register(
    "spec_schema",
    description=(
        "СХЕМА phr-спеки из ядра: все роли узлов с обязательными и допустимыми "
        "ключами, формат обёртки (spec_version/group_order/nodes), инварианты "
        "валидатора (k=2 → ровно один SHARE_CLOSURE; k≥3 → только "
        "SHARE_SIMPLEX; closure и FIXED без диапазона), готовый ВАЛИДНЫЙ "
        "пример и список ключей, которых в спеке НЕТ (levels, premix, "
        "process…). Работает БЕЗ активной спеки — вызывай ПЕРЕД тем, как "
        "предлагать пакет (propose_spec): формат по памяти не восстанавливай."),
    parameters={"type": "object", "properties": {
        "include_example": {"type": "boolean",
                            "description": "включить пример пакета "
                                           "(по умолчанию да)"}}})
def spec_schema(ctx: ToolContext, include_example: bool = True
                ) -> Dict[str, Any]:
    """Схема v2 как ДАННЫЕ — прямо из ``_ROLE_TABLE`` ядра.

    Спека здесь не требуется НАМЕРЕННО: именно этот инструмент нужен, когда
    геометрии ещё нет и её предстоит ввести впервые. Источник — таблица ролей
    конструктора, поэтому схема не может разъехаться с валидатором: такой
    разъезд и есть тот отказ, который инструмент закрывает.
    """
    from ...design.phr_sampler import _ROLE_TABLE, _SCALES   # тяжёлый импорт

    roles: Dict[str, Any] = {}
    for role, (mode, required, allowed) in sorted(_ROLE_TABLE.items()):
        roles[role] = {
            "mode": mode,
            "required": sorted(required),
            "allowed": sorted(allowed),
            "optional": sorted(set(allowed) - set(required) - {"name", "role"}),
        }
    out: Dict[str, Any] = {
        "spec_version": 2,
        "wrapper": {
            "keys": ["spec_version", "nodes", "group_order"],
            "note": "Пакет — либо СПИСОК узлов, либо обёртка "
                    "{'spec_version': 2, 'nodes': [...], 'group_order': "
                    "[...]}. Иных ключей верхнего уровня нет.",
        },
        "roles": roles,
        "scales": list(_SCALES),
        "invariants": [
            "Группа из k=2 членов: РОВНО один SHARE_CLOSURE и один SHARE_FREE "
            "(доля closure производная 1−φ, поэтому share_range у него нет).",
            "Группа из k≥3 членов: ВСЕ члены SHARE_SIMPLEX, SHARE_CLOSURE "
            "запрещён; симплекс должен быть совместен: Σlo ≤ 1 ≤ Σhi.",
            "SHARE_CLOSURE / FIXED / GROUP_TOTAL_FIXED — без 'range' и "
            "'share_range': наличие это ошибка, а не тихое игнорирование.",
            "'members' тотала обязаны ТОЧНО совпадать (состав и порядок) с "
            "узлами, у которых 'group' равен имени этого тотала.",
            "group_order — ТОЧНАЯ перестановка множества GROUP_TOTAL-групп "
            "(GROUP_TOTAL_FIXED исключается); входит в spec_hash.",
            "Лишние ключи узла — ошибка валидации (в т.ч. legacy lo/hi/of/to).",
        ],
        "not_in_spec": dict(SPEC_NOT_HERE),
        "hint": "Собрал пакет — зови propose_spec: он валидирует его ЯДРОМ и "
                "кладёт в стейдж, откуда человек применяет кнопкой.",
    }
    if include_example:
        out["example"] = copy.deepcopy(SPEC_EXAMPLE)

    # «Схему знаю» и «текущая геометрия такая» — разные утверждения, поэтому
    # отпечаток активной спеки показывается отдельным полем (или его отсутствие
    # называется прямо: это первичный ввод).
    spec = ctx.spec
    if spec is None and ctx.runner is not None:
        spec = getattr(ctx.runner, "phr_spec", None)
    out["current"] = ({"present": True, "spec_hash": spec.spec_hash(),
                       "q_components": int(spec.q), "dim_z": int(spec.dim_z)}
                      if spec is not None else
                      {"present": False,
                       "note": "Активной спеки нет: геометрия ещё не введена. "
                               "Это ПЕРВИЧНЫЙ ввод — предлагай полный пакет "
                               "(propose_spec); патчем узлы не добавляются."})
    return out


# ----------------------------------------------------------------------
# get_spec
# ----------------------------------------------------------------------
@register(
    "get_spec",
    description=(
        "Актуальная phr-спека кампании: роли узлов, границы, шкалы, "
        "техлимиты, group_order, spec_hash, размерности. ВЫЗЫВАЙ ПЕРВЫМ "
        "перед любым рассуждением о границах: спека могла измениться."),
    parameters={"type": "object", "properties": {
        "include_nodes": {"type": "boolean",
                          "description": "включить полный список узлов "
                                         "(по умолчанию да)"}}})
def get_spec(ctx: ToolContext, include_nodes: bool = True) -> Dict[str, Any]:
    spec = ctx.require_spec()
    nodes, meta = spec_payload(spec)
    intervals = {k: [float(v[0]), float(v[1])]
                 for k, v in spec.phr_intervals().items()}
    out: Dict[str, Any] = {
        "spec_hash": spec.spec_hash(),
        "schema_version": int(getattr(spec, "schema_version", 1) or 1),
        "q_components": int(spec.q),
        "dim_z": int(spec.dim_z),
        "group_order": list(getattr(spec, "group_order", []) or []),
        "component_names": list(spec.component_names),
        "phr_intervals": intervals,
        # iter84: ИСПРАВЛЕН баг двойного счёта (найден iter83, 12.08.2026).
        # Раньше здесь суммировались ВСЕ узлы `phr_intervals()`, включая
        # узлы-ТОТАЛЫ групп; тотал группы = сумма своих детей ⇒ группы
        # считались ДВАЖДЫ и верх Σphr был завышен: на референсной спеке
        # `pvc_edge_v1` выходило 114.85…162.80 вместо верного 109.85…147.80
        # (расхождение ровно на интервал тотала SOFT). Именно оттуда шли Σphr,
        # которые ассистент цитировал в ПВХ-сессии.
        #
        # Считает ЯДРО (`PhrSpec.sigma_phr_bounds` — сумма по ЛИСТЬЯМ): своей
        # копии арифметики здесь больше нет, поэтому окно технолога и ответ
        # помощника не могут разойтись. Тесты — test_iteration84_sigma_phr_fix.py
        # и test_iteration83_batch_weighing.py::TestSigmaPhr.
        "sigma_phr_static": [float(v) for v in spec.sigma_phr_bounds()],
        "log_axes": [d["name"] for d in nodes if d.get("scale") == "log"],
        "meta": _f(meta),
    }
    if include_nodes:
        out["nodes"] = [{**_f(d), "role": spec.role_of(str(d.get("name")))}
                        for d in nodes]
    runner = ctx.runner
    if runner is not None:
        out["campaign"] = {
            "label": getattr(runner, "campaign_label", ""),
            "property_names": list(getattr(runner, "property_names", []) or []),
            "covariate_names": list(getattr(runner, "covariate_names", []) or []),
            "process_levels": _f(getattr(runner, "process_levels", {}) or {}),
            "n_points": len(getattr(runner, "points", []) or []),
        }
    return out


# ----------------------------------------------------------------------
# explain_node
# ----------------------------------------------------------------------
@register(
    "explain_node",
    description=(
        "Объяснить узел спеки: роль, координата, статический интервал phr, "
        "какое ограничение активно (собственный диапазон, техлимит min/max "
        "phr, партнёры по группе, потолок cap, окно тотала), таблица "
        "эффективных долей при разных суммах группы. Отвечает на вопрос "
        "«почему диапазон не такой, как я ввёл»."),
    parameters={"type": "object", "properties": {
        "name": {"type": "string", "description": "имя узла"},
        "totals": {"type": "array", "items": {"type": "number"},
                   "description": "суммы группы, для которых показать "
                                  "эффективные доли (по умолчанию сетка)"}},
        "required": ["name"]})
def explain_node(ctx: ToolContext, name: str,
                 totals: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    spec = ctx.require_spec()
    nodes, _ = spec_payload(spec)
    by_name = _nodes_by_name(nodes)
    name = str(name)
    if name not in by_name:
        raise ToolError(f"Узла '{name}' нет в спеке. Есть: {sorted(by_name)}.")

    node = by_name[name]
    role = spec.role_of(name)
    intervals = spec.phr_intervals()
    lo, hi = intervals.get(name, (float("nan"), float("nan")))
    out: Dict[str, Any] = {
        "name": name, "role": role, "node": _f(node),
        "phr_interval": [float(lo), float(hi)],
        "is_component": name in list(spec.component_names),
        "scale": node.get("scale", "linear"),
    }

    if role in _SHARE_ROLES:
        parent = str(node.get("group") or node.get("of") or "")
        out["group"] = parent
        members = [d["name"] for d in nodes
                   if str(d.get("group") or d.get("of") or "") == parent]
        out["group_members"] = members
        p_lo, p_hi = intervals.get(parent, (0.0, 0.0))
        out["group_total_interval"] = [float(p_lo), float(p_hi)]
        grid = list(totals) if totals else _default_totals(float(p_lo),
                                                           float(p_hi))
        rows = []
        idx = members.index(name) if name in members else None
        for t in grid:
            try:
                blo, bhi = spec.share_bounds_at_total(parent, float(t))
            except (ValueError, KeyError) as exc:
                rows.append({"total": float(t), "error": str(exc)})
                continue
            if idx is None:
                continue
            rows.append({"total": float(t),
                         "share_lo": float(blo[idx]), "share_hi": float(bhi[idx]),
                         "phr_lo": float(blo[idx] * t),
                         "phr_hi": float(bhi[idx] * t)})
        out["effective_shares"] = rows
        out["note"] = (
            "Эффективные доли считаются ПОСЛЕ розыгрыша суммы группы "
            "(conditional narrowing): собственный share_range, техлимиты "
            "min_phr/max_phr и лимиты ПАРТНЁРОВ (Σφ=1) сужают интервал, "
            "поэтому hi(T) бывает НЕМОНОТОННОЙ — не делай вывод по двум "
            "точкам.")
    if node.get("cap_to"):
        out["cap"] = {"cap_to": list(node.get("cap_to") or []),
                      "cap_ratio": float(node.get("cap_ratio", 0.0)),
                      "note": "потолок = cap_ratio · Σ(cap_to) В ТОЧКЕ "
                              "(трапеция по фазе, не клин): нижняя граница "
                              "от референса не зависит."}
    if node.get("reference"):
        out["reference"] = node["reference"]
        out["note_ratio"] = ("RATIO_TO — КЛИН: масштабируются оба конца, узел "
                             "почти линейно следует за референсом.")

    facts = _facts(ctx, scope=name)
    if facts:
        out["local_facts"] = facts
    return out


def _default_totals(lo: float, hi: float, k: int = 5) -> List[float]:
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return [float(lo)]
    return [float(v) for v in np.linspace(lo, hi, k)]


# ----------------------------------------------------------------------
# validate_spec (dry-run)
# ----------------------------------------------------------------------
@register(
    "validate_spec",
    description=(
        "Сухой прогон патча спеки: применить к КОПИИ, прогнать валидацию "
        "конструктора, показать ошибки, дифф эффективных границ phr и "
        "изменится ли spec_hash. Ничего не применяет."),
    parameters={"type": "object", "properties": {
        "patch": {"type": "object",
                  "description": "патч: {'node': имя, 'field': поле, "
                                 "'value': значение} либо {'node':…, 'set':{…}, "
                                 "'unset':[…]}; спека целиком — узел '__spec__'"}},
        "required": ["patch"]})
def validate_spec(ctx: ToolContext, patch: Any) -> Dict[str, Any]:
    spec = ctx.require_spec()
    before = spec.phr_intervals()
    hash_before = spec.spec_hash()
    try:
        patched = build_patched_spec(spec, patch)
    except ToolError:
        raise
    except Exception as exc:  # noqa: BLE001 — ошибка валидации это РЕЗУЛЬТАТ
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "spec_hash_before": hash_before,
                "hint": "Патч отклонён валидацией спеки — геометрия осталась "
                        "прежней. Это ответ по существу, а не сбой."}

    after = patched.phr_intervals()
    diff = []
    for nm in sorted(set(before) | set(after)):
        b = before.get(nm)
        a = after.get(nm)
        if b is None or a is None or abs(a[0] - b[0]) > 1e-12 \
                or abs(a[1] - b[1]) > 1e-12:
            diff.append({"node": nm,
                         "before": None if b is None else [float(b[0]), float(b[1])],
                         "after": None if a is None else [float(a[0]), float(a[1])]})
    hash_after = patched.spec_hash()
    return {
        "ok": True,
        "spec_hash_before": hash_before,
        "spec_hash_after": hash_after,
        "affects_hash": hash_before != hash_after,
        "dim_z_before": int(spec.dim_z), "dim_z_after": int(patched.dim_z),
        "changed_intervals": diff,
        "n_changed": len(diff),
        "warning": ("spec_hash меняется ⇒ это ДРУГАЯ геометрия кампании: "
                    "план и уже собранные точки относятся к прежнему "
                    "отпечатку." if hash_before != hash_after else ""),
    }


# ----------------------------------------------------------------------
# validate_spec_package (iter71)
# ----------------------------------------------------------------------
def normalize_spec_package(package: Any) -> Tuple[List[Dict[str, Any]],
                                                  List[str], int]:
    """Пакет спеки из модели → ``(узлы, group_order, spec_version)``.

    Принимаем и обёртку, и плоский список — то же, что ``PhrSpec.from_dicts``.
    Отказ формулируется УКАЗАНИЕМ на верный вид: именно здесь модель, знающая
    геометрию словами, но не знающая схему, теряет ход (её JSON приходит с
    ключами вроде ``components``/``groups``/``process``).
    """
    if package is None:
        raise ToolError("Пакет спеки пуст: нечего валидировать. Формат — "
                        "вызови spec_schema.")
    if isinstance(package, (list, tuple)):
        nodes = list(package)
        order: List[str] = []
        version = 2
    elif isinstance(package, dict):
        if "nodes" not in package:
            raise ToolError(
                f"В пакете нет ключа 'nodes' (пришли: {sorted(package)}). "
                f"Узлы НЕ делятся на 'components'/'groups'/'process': это ОДИН "
                f"плоский список 'nodes', где тотал группы — узел с role "
                f"GROUP_TOTAL(_FIXED) и 'members'. Вызови spec_schema.")
        extra = set(package) - {"spec_version", "nodes", "group_order"}
        if extra:
            raise ToolError(
                f"Лишние ключи обёртки пакета: {sorted(extra)}. Допустимы "
                f"только 'spec_version', 'nodes', 'group_order' (см. "
                f"spec_schema): метка версии кампании живёт в паспорте, "
                f"process-оси и уровни в спеке не задаются.")
        raw = package.get("nodes")
        if not isinstance(raw, (list, tuple)):
            raise ToolError("'nodes' должен быть СПИСКОМ узлов-объектов.")
        nodes = list(raw)
        go = package.get("group_order") or []
        if isinstance(go, str) or not isinstance(go, (list, tuple)):
            raise ToolError("'group_order' должен быть СПИСКОМ имён "
                            "GROUP_TOTAL-узлов.")
        order = [str(x) for x in go]
        version = int(package.get("spec_version", 2) or 2)
    else:
        raise ToolError(f"Пакет спеки должен быть объектом-обёрткой или "
                        f"списком узлов, получено: {type(package).__name__}.")
    if not nodes:
        raise ToolError("Пакет спеки без узлов: предлагать пустую геометрию "
                        "нельзя.")
    bad = [i for i, d in enumerate(nodes, 1) if not isinstance(d, dict)]
    if bad:
        raise ToolError(f"Узлы пакета должны быть объектами "
                        f"{{'name': …, 'role': …}}; не объекты в позициях {bad}.")
    return [dict(d) for d in nodes], order, version


def build_spec_from_package(package: Any):
    """Собрать :class:`PhrSpec` из пакета (конструктор валидирует)."""
    from ...design.phr_sampler import PhrSpec           # тяжёлый импорт

    nodes, order, version = normalize_spec_package(package)
    payload: Any = ({"spec_version": version, "group_order": order,
                     "nodes": nodes} if order else nodes)
    return PhrSpec.from_dicts(payload)


def spec_package_diff(spec, candidate) -> Dict[str, Any]:
    """Чем НОВАЯ спека отличается от текущей: состав, роли, границы, отпечаток.

    Именно этот разбор человек видит рядом с кнопкой подтверждения. Показываем
    не «спека изменилась», а ЧТО именно: добавленные и удалённые узлы, смена
    ролей, съехавшие интервалы — иначе решение принимается вслепую (A0.6).
    """
    new_iv = candidate.phr_intervals()
    new_roles = {nm: candidate.role_of(nm) for nm in new_iv}
    out: Dict[str, Any] = {
        "spec_hash_after": candidate.spec_hash(),
        "q_after": int(candidate.q), "dim_z_after": int(candidate.dim_z),
        "component_names_after": list(candidate.component_names),
        "group_order_after": list(getattr(candidate, "group_order", []) or []),
        "first_spec": spec is None,
    }
    if spec is None:
        # Первичный ввод: сравнивать не с чем, и делать вид, будто «ничего не
        # изменилось», нельзя — геометрия появляется там, где её не было.
        out.update({"spec_hash_before": "", "added": sorted(new_iv),
                    "removed": [], "role_changed": [], "changed_intervals": [],
                    "components_added": list(candidate.component_names),
                    "components_removed": [], "affects_hash": True,
                    "q_before": 0, "dim_z_before": 0})
        return _f(out)

    old_iv = spec.phr_intervals()
    old_roles = {nm: spec.role_of(nm) for nm in old_iv}
    common = sorted(set(old_iv) & set(new_iv))
    changed = []
    for nm in common:
        b, a = old_iv[nm], new_iv[nm]
        if abs(a[0] - b[0]) > 1e-12 or abs(a[1] - b[1]) > 1e-12:
            changed.append({"node": nm,
                            "before": [float(b[0]), float(b[1])],
                            "after": [float(a[0]), float(a[1])]})
    hash_before = spec.spec_hash()
    out.update({
        "spec_hash_before": hash_before,
        "affects_hash": hash_before != out["spec_hash_after"],
        "q_before": int(spec.q), "dim_z_before": int(spec.dim_z),
        "added": sorted(set(new_iv) - set(old_iv)),
        "removed": sorted(set(old_iv) - set(new_iv)),
        "role_changed": [{"node": nm, "before": old_roles[nm],
                          "after": new_roles[nm]}
                         for nm in common if old_roles[nm] != new_roles[nm]],
        "changed_intervals": changed,
        "components_added": sorted(set(candidate.component_names)
                                   - set(spec.component_names)),
        "components_removed": sorted(set(spec.component_names)
                                     - set(candidate.component_names)),
    })
    return _f(out)


def active_spec(ctx: ToolContext):
    """Активная спека или ``None`` — БЕЗ отказа.

    Отличается от :meth:`ToolContext.require_spec` намеренно: инструментам
    первичного ввода отсутствие спеки не мешает, оно и есть их случай.
    """
    if ctx.spec is not None:
        return ctx.spec
    return getattr(ctx.runner, "phr_spec", None) if ctx.runner else None


@register(
    "validate_spec_package",
    description=(
        "Сухой прогон ПАКЕТА спеки целиком — первичный ввод геометрии или её "
        "эволюция (добавить узел, удалить узел, сменить роль, расширить "
        "границы): собрать ядром, показать ошибки валидации, эффективные phr, "
        "dim_z, spec_hash и ДИФФ против текущей спеки (что добавлено, удалено, "
        "где сменилась роль). Ничего не применяет и в стейдж не кладёт. "
        "Работает и когда активной спеки ещё нет. Формат — spec_schema."),
    parameters={"type": "object", "properties": {
        "package": {"type": "object",
                    "description": "спека ЦЕЛИКОМ: {'spec_version': 2, "
                                   "'nodes': [...], 'group_order': [...]} "
                                   "или просто список узлов"}},
        "required": ["package"]})
def validate_spec_package(ctx: ToolContext, package: Any) -> Dict[str, Any]:
    spec = active_spec(ctx)
    try:
        candidate = build_spec_from_package(package)
    except ToolError:
        raise
    except Exception as exc:      # noqa: BLE001 — отказ валидации это РЕЗУЛЬТАТ
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "hint": "Пакет отклонён валидатором ЯДРА — геометрия проекта "
                        "не тронута. Это ответ по существу, а не сбой: сверь "
                        "ключи и инварианты со spec_schema и собери заново."}

    nodes, _ = spec_payload(candidate)
    diff = spec_package_diff(spec, candidate)
    out: Dict[str, Any] = {
        "ok": True,
        "diff": diff,
        "phr_intervals": {k: [float(v[0]), float(v[1])]
                          for k, v in candidate.phr_intervals().items()},
        "log_axes": [str(d["name"]) for d in nodes if d.get("scale") == "log"],
        "nodes_total": len(nodes),
        "q_components": int(candidate.q),
        "dim_z": int(candidate.dim_z),
        "spec_hash": candidate.spec_hash(),
    }
    if diff.get("first_spec"):
        out["warning"] = ("Это ПЕРВИЧНЫЙ ввод геометрии: у проекта спеки нет, "
                          "сравнивать не с чем.")
    elif diff.get("removed") or diff.get("components_removed"):
        out["warning"] = (
            f"Пакет УДАЛЯЕТ узлы {diff.get('removed')} (компоненты: "
            f"{diff.get('components_removed')}): точки, собранные в прежней "
            f"геометрии, к новой не относятся.")
    elif diff.get("affects_hash"):
        out["warning"] = ("spec_hash меняется ⇒ это ДРУГАЯ геометрия кампании: "
                          "ранее собранные точки относятся к прежнему "
                          "отпечатку.")
    return _f(out)


# ----------------------------------------------------------------------
# Пакет ПРОЕКТА (iter73): схема и dry-run
# ----------------------------------------------------------------------
def has_project(ctx: ToolContext) -> bool:
    """Собран ли проект в этой сессии (движок есть) — БЕЗ отказа.

    Отличать «проекта нет» от «спеки нет» обязательно: первое означает, что
    проект надо РОЖДАТЬ (пакетом проекта), второе — что в существующем проекте
    не задана геометрия (пакетом спеки). Смешение этих случаев и было причиной
    «нажал Применить, ничего не изменилось».
    """
    return ctx.runner is not None


@register(
    "get_setup_fields",
    description=(
        "ТЕКУЩИЕ ЗНАЧЕНИЯ ПОЛЕЙ формы «🆕 Новый проект» (сетап) — снимок "
        "того, что человек уже ввёл, ДО сборки проекта. Зови ПЕРЕД тем, как "
        "предлагать первичный ввод или правку: если поля уже заполнены, "
        "предлагай ТОЧЕЧНУЮ правку (propose_setup_fields), а не пакет "
        "проекта с нуля. Ключи полей — те же, что принимает "
        "propose_setup_fields."),
    parameters={"type": "object", "properties": {}})
def get_setup_fields(ctx: ToolContext) -> Dict[str, Any]:
    """Снимок полей формы сетапа из контекста (кладёт UI-слой, iter76).

    Инструменты чистые и Streamlit не видят, поэтому снимок ``setup_*``-полей
    передаёт док через ``ctx.extra['setup_fields']``. Отсутствие снимка — не
    ошибка: MCP-сервер и демо работают без формы.
    """
    fields = dict((ctx.extra or {}).get("setup_fields") or {})
    out: Dict[str, Any] = {
        "n": len(fields), "fields": _f(fields),
        "project_present": has_project(ctx),
    }
    if has_project(ctx):
        out["note"] = ("Проект уже СОБРАН: поля формы — лишь черновик "
                       "пересборки, состояние движка смотри в "
                       "campaign_overview/get_spec.")
    elif not fields:
        out["note"] = ("Форма пуста или снимок недоступен (вызов вне UI). "
                       "Если проекта нет и полей нет — это первичный ввод: "
                       "собирай пакет проекта (propose_project).")
    else:
        out["note"] = ("Проект НЕ собран, но поля формы уже заполнены. "
                       "Точечные изменения предлагай через "
                       "propose_setup_fields — пакет проекта целиком "
                       "затёр бы ручной ввод человека.")
    return out


@register(
    "project_schema",
    description=(
        "СХЕМА ПАКЕТА ПРОЕКТА из ядра: какие блоки нужны, чтобы РОДИТЬ проект "
        "(состав 'spec' = phr-спека, 'responses' = отклики, 'process' = "
        "процесс-оси с границами в реальных единицах), какие необязательны "
        "('covariates', 'passport', 'seed'), в каких ЕДИНИЦАХ что задаётся, "
        "инварианты и готовый ВАЛИДНЫЙ пример. Работает без проекта и без "
        "спеки — это инструмент ПЕРВИЧНОГО ввода. Зови ПЕРЕД propose_project: "
        "формат по памяти не восстанавливай. Одной phr-спеки для рождения "
        "проекта НЕ хватает: откликов и осей в ней нет по схеме."),
    parameters={"type": "object", "properties": {
        "include_example": {"type": "boolean",
                            "description": "включить пример пакета "
                                           "(по умолчанию да)"}}})
def project_schema(ctx: ToolContext, include_example: bool = True
                   ) -> Dict[str, Any]:
    """Схема пакета проекта как ДАННЫЕ (источник — :mod:`design.project_package`).

    Состояние проекта отдаётся отдельным полем ``current``: «схему знаю» и «в
    проекте уже есть движок» — разные утверждения, и от второго зависит, какой
    инструмент уместен (``propose_project`` против ``propose_spec``).
    """
    from ...design.project_package import project_package_schema

    out = project_package_schema(include_example=bool(include_example))
    spec = active_spec(ctx)
    if has_project(ctx):
        runner = ctx.runner
        out["current"] = {
            "project_present": True,
            "responses": [str(p) for p in
                          (getattr(runner, "property_names", []) or [])],
            "process": [str(p) for p in
                        (getattr(getattr(runner, "current_schema", None),
                                 "process_names", []) or [])],
            "points": int(len(getattr(runner, "points", []) or [])),
            "spec_hash": spec.spec_hash() if spec is not None else "",
            "note": "Проект уже собран: пакетом ПРОЕКТА он не заводится "
                    "заново. Геометрию правь пакетом спеки (propose_spec), "
                    "отклики и оси меняет человек в сетапе.",
        }
    else:
        out["current"] = {
            "project_present": False,
            "note": "Проекта в сессии нет — это ПЕРВИЧНЫЙ ввод: собери пакет "
                    "проекта целиком (propose_project). Отклики и границы "
                    "процесс-осей НЕ выдумывай: их называет технолог; если их "
                    "не назвали, спроси в OPEN_QUESTIONS.",
        }
    return out


@register(
    "validate_project_package",
    description=(
        "Сухой прогон ПАКЕТА ПРОЕКТА: разобрать блоки ядром (спека — тем же "
        "конструктором PhrSpec), показать МАНИФЕСТ «что именно загружается» по "
        "блокам (компоненты, отклики с единицами, процесс-оси с границами и "
        "режимами, ковариаты, паспорт), список недостающего и ошибки. Ничего "
        "не применяет и в стейдж не кладёт. Формат — project_schema."),
    parameters={"type": "object", "properties": {
        "package": {"type": "object",
                    "description": "пакет проекта ЦЕЛИКОМ: {'package_kind': "
                                   "'project', 'spec': {...}, 'responses': "
                                   "[...], 'process': [...]}"}},
        "required": ["package"]})
def validate_project_package(ctx: ToolContext, package: Any) -> Dict[str, Any]:
    """Dry-run пакета проекта: отказ — это РЕЗУЛЬТАТ, а не исключение.

    Ход модели не должен падать из-за неверного пакета: она обязана прочитать
    причину и собрать пакет заново. Поэтому :class:`PackageError` (пакет собран
    неверно) превращается в ``ok=False`` с подсказкой, а не в ``ToolError``.
    """
    from ...design.project_package import (PackageError, manifest_caption,
                                           package_manifest,
                                           parse_project_package)
    try:
        pkg = parse_project_package(package)
    except PackageError as exc:
        return {"ok": False, "error": str(exc),
                "hint": "Пакет отклонён ядром — проект не тронут. Это ответ по "
                        "существу, а не сбой: сверь блоки и единицы со "
                        "project_schema и собери заново."}
    out: Dict[str, Any] = {
        "ok": True,
        "manifest": package_manifest(pkg),
        "caption": manifest_caption(pkg),
        "spec_hash": pkg.spec_hash,
        "q_components": int(pkg.spec.q),
        "dim_z": int(pkg.spec.dim_z),
        "phr_intervals": {k: [float(v[0]), float(v[1])]
                          for k, v in pkg.spec.phr_intervals().items()},
        "project_present": has_project(ctx),
    }
    if has_project(ctx):
        out["warning"] = (
            "Проект в сессии УЖЕ собран: пакет проекта его не заменяет и "
            "применён не будет. Правку геометрии предлагай пакетом спеки "
            "(propose_spec); отклики и оси меняет человек в сетапе.")
    return _f(out)


# ----------------------------------------------------------------------
# simulate_bounds
# ----------------------------------------------------------------------
@register(
    "simulate_bounds",
    description=(
        "Что поедет ЧИСЛЕННО, если применить патч: Σphr, диапазоны узлов, "
        "корреляции между осями (в phr) — до и после. Плана не генерирует, "
        "состояние не меняет. Используй, чтобы отличить клин (corr≈0.9) от "
        "трапеции по фазе (corr≈0.1–0.2)."),
    parameters={"type": "object", "properties": {
        "patch": {"type": "object", "description": "патч (как в validate_spec); "
                                                   "без него — только текущая спека"},
        "n": {"type": "integer", "description": f"кандидатов (по умолчанию "
                                                f"{DEFAULT_SIM_N})"},
        "seed": {"type": "integer", "description": "seed розыгрыша"},
        "pair": {"type": "array", "items": {"type": "string"},
                 "description": "пара узлов, для которой явно показать corr"}}},
    long_running=True)
def simulate_bounds(ctx: ToolContext, patch: Any = None,
                    n: int = DEFAULT_SIM_N, seed: int = 0,
                    pair: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    spec = ctx.require_spec()
    n = int(max(20, min(int(n or DEFAULT_SIM_N), 5000)))

    base = _sample_stats(spec, n=n, seed=int(seed), pair=pair)
    out: Dict[str, Any] = {"n": n, "seed": int(seed), "current": base}

    if patch:
        try:
            patched = build_patched_spec(spec, patch)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001
            return {**out, "ok": False,
                    "error": f"патч не прошёл валидацию: "
                             f"{type(exc).__name__}: {exc}"}
        out["proposed"] = _sample_stats(patched, n=n, seed=int(seed), pair=pair)
        out["ok"] = True
        out["sigma_phr_shift"] = [
            round(out["proposed"]["sigma_phr"][i] - base["sigma_phr"][i], 4)
            for i in range(2)]
        if pair and len(pair) == 2:
            c0 = base.get("pair_corr")
            c1 = out["proposed"].get("pair_corr")
            if c0 is not None and c1 is not None:
                out["pair_corr_shift"] = round(float(c1) - float(c0), 4)
    return _f(out)


def _sample_stats(spec, *, n: int, seed: int,
                  pair: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Статистики розыгрыша в ФИЗИЧЕСКИХ единицах (phr)."""
    P = np.atleast_2d(np.asarray(spec.decode(spec.sample_z(n, seed=seed)), float))
    names = list(spec.component_names)
    sums = P.sum(axis=1)
    ranges = {nm: [float(P[:, i].min()), float(P[:, i].max())]
              for i, nm in enumerate(names)}

    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(P, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    top: List[Dict[str, Any]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            top.append({"a": names[i], "b": names[j],
                        "corr": round(float(C[i, j]), 3)})
    top.sort(key=lambda d: -abs(d["corr"]))

    out: Dict[str, Any] = {
        "spec_hash": spec.spec_hash(),
        "sigma_phr": [float(sums.min()), float(sums.max())],
        "sigma_phr_mean": float(sums.mean()),
        "phr_ranges": ranges,
        "max_abs_corr": top[0] if top else None,
        "top_corr": top[:8],
    }
    if pair and len(pair) == 2:
        try:
            i, j = names.index(str(pair[0])), names.index(str(pair[1]))
        except ValueError as exc:
            raise ToolError(
                f"Пара {list(pair)}: такого компонента нет среди "
                f"{names}.") from exc
        out["pair"] = [str(pair[0]), str(pair[1])]
        out["pair_corr"] = round(float(C[i, j]), 3)
    return out


# ----------------------------------------------------------------------
# preflight
# ----------------------------------------------------------------------
@register(
    "preflight",
    description=(
        "Диагностика плана ДО прогона: rank/cond/VIF/max|corr|, слепое "
        "направление, покрытие сумм групп и обязательных 2D-пар, список "
        "проваленных гейтов. Считает по текущей базе точек проекта (или по "
        "предложенному seed-плану, если база пуста). Долгая операция."),
    parameters={"type": "object", "properties": {
        "n": {"type": "integer",
              "description": "размер seed-плана, если база пуста"}}},
    long_running=True)
def preflight(ctx: ToolContext, n: Optional[int] = None) -> Dict[str, Any]:
    runner = ctx.require_runner()
    # БАГФИКС (аудит 13.08.2026): у живого раннера пустая база — это
    # ``X = None`` (см. MixtureProcessRunner._rebuild_arrays), а не массив
    # (0, 0); np.asarray(None) даёт форму (1, 1), guard «база пуста» не
    # срабатывал, и ядро падало «ожидалось 22 координат на точку, дано 1».
    Xattr = getattr(runner, "X", None)
    X = (np.atleast_2d(np.asarray(Xattr, float)) if Xattr is not None
         else np.empty((0, 0)))
    source = "база точек проекта"
    if X.size == 0 or X.shape[0] == 0:
        # Черновик стартового плана (draft.seed_X в campaign.json) — план,
        # который РЕАЛЬНО видит человек в UI до фиксации. Если он сохранён,
        # проверять надо именно его, а не свежесгенерированный аналог.
        draft_X = _draft_seed_X(ctx)
        if draft_X is not None:
            X = draft_X
            source = (f"черновик стартового плана сохранённого проекта "
                      f"(n={X.shape[0]}), база пуста")
            if n is not None and int(n) != X.shape[0]:
                source += (f"; запрошенный n={int(n)} проигнорирован — "
                           f"проверяется план, предложенный в интерфейсе")
        else:
            k = int(n or 16)
            X = np.atleast_2d(np.asarray(runner.propose_seed(k), float))
            source = f"предложенный seed-план (n={k}), база пуста"
    try:
        report = runner.preflight(X)
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"preflight не выполнился: {type(exc).__name__}: "
                        f"{exc}") from exc

    data = report if isinstance(report, dict) else _dataclass_to_dict(report)
    data = _f(data)
    data["source"] = source
    data["n_points"] = int(X.shape[0])
    data["note"] = ("Гейты ОТНОСИТЕЛЬНЫЕ (сверка с равномерным пулом той же "
                    "области): абсолютные пороги регрессии (cond<30, VIF<5) в "
                    "долях Шеффе неприменимы. Провал объясняй физикой: какие "
                    "два узла дублируют функцию или где доли линейно зависимы "
                    "от тотала.")
    return data


def _dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    from dataclasses import asdict, is_dataclass
    if is_dataclass(obj):
        return asdict(obj)
    return {k: v for k, v in vars(obj).items() if not k.startswith("_")}


def _draft_seed_X(ctx: ToolContext) -> Optional[np.ndarray]:
    """``seed_X`` из черновика сохранённого проекта (``draft`` в campaign.json).

    Предложенный в UI стартовый план до фиксации живёт ТОЛЬКО черновиком
    (``session_state`` → ``draft`` при «💾 Сохранить проект»): без этого чтения
    инструменты проверяли не тот план, что видит человек (аудит 13.08.2026).
    Нет файла / нет черновика / битые данные → ``None`` (не ошибка: черновик
    необязателен).
    """
    if not (ctx.root and ctx.project):
        return None
    try:
        from src.apps.campaign_state import load_campaign_draft
        draft = load_campaign_draft(ctx.root, ctx.project)
    except Exception:  # noqa: BLE001 — нет campaign.json = черновика нет
        return None
    if not draft or draft.get("seed_X") is None:
        return None
    try:
        X = np.atleast_2d(np.asarray(draft["seed_X"], float))
    except (TypeError, ValueError):
        return None
    return X if X.size else None


# ----------------------------------------------------------------------
# Точка и рецепт
# ----------------------------------------------------------------------
@register(
    "point_report",
    description=(
        "Контракт ядра на КОНКРЕТНЫЙ рецепт (phr по компонентам в порядке "
        "component_names): эффективные границы каждого узла с меткой, какое "
        "ограничение сработало (range/derived/window/cap/min_phr/max_phr/"
        "partners), требование премикса, nominal vs actual при заданном шаге "
        "весов, список нарушений."),
    parameters={"type": "object", "properties": {
        "recipe_phr": {"type": "array", "items": {"type": "number"},
                       "description": "рецепт в phr, порядок component_names"},
        "delta_phr": {"type": "number",
                      "description": "разрешение весов δ (phr); без него "
                                     "премикс не считается"}},
        "required": ["recipe_phr"]})
def point_report(ctx: ToolContext, recipe_phr: Sequence[float],
                 delta_phr: Optional[float] = None) -> Dict[str, Any]:
    spec = ctx.require_spec()
    p = np.asarray(list(recipe_phr), float)
    if p.shape[0] != int(spec.q):
        raise ToolError(
            f"Рецепт содержит {p.shape[0]} значений, а компонентов в спеке "
            f"{int(spec.q)}. Порядок обязателен: {list(spec.component_names)}.")
    try:
        rep = spec.point_report(p, delta_phr=delta_phr)
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"point_report: {type(exc).__name__}: {exc}") from exc

    # Координата узла и его phr — РАЗНЫЕ величины (доля vs phr у share-узлов,
    # коэффициент vs phr у ratio_to): отдаём обе, иначе модель сравнит долю с
    # границей в phr и объяснит нарушение неверно.
    bounds = {nm: {"mode": eb.mode, "coord": float(eb.coord),
                   "phr": float(eb.phr), "lo": float(eb.lo),
                   "hi": float(eb.hi), "active_lo": eb.active_lo,
                   "active_hi": eb.active_hi}
              for nm, eb in rep.effective_bounds.items()}

    return _f({
        "ok": bool(rep.ok),
        "violations": list(rep.violations),
        "effective_bounds": bounds,
        "premix": {k: v for k, v in rep.premix.items()},
        "phr_nominal": rep.phr_nominal,
        "phr_actual": rep.phr_actual,
        "delta_phr": rep.delta_phr,
    })


@register(
    "encode_recipe",
    description=(
        "Импорт рецепта: phr → внутренние координаты (z) и доли. Отвечает, "
        "ПРЕДСТАВИМ ли рецепт в текущей спеке (годится ли как anchor) и что "
        "именно не влезло."),
    parameters={"type": "object", "properties": {
        "recipe_phr": {"type": "array", "items": {"type": "number"},
                       "description": "рецепт в phr, порядок component_names"}},
        "required": ["recipe_phr"]})
def encode_recipe(ctx: ToolContext, recipe_phr: Sequence[float]) -> Dict[str, Any]:
    spec = ctx.require_spec()
    p = np.asarray(list(recipe_phr), float)
    if p.shape[0] != int(spec.q):
        raise ToolError(
            f"Рецепт содержит {p.shape[0]} значений, ожидалось {int(spec.q)} "
            f"({list(spec.component_names)}).")
    try:
        z = spec.encode(p)
    except Exception as exc:  # noqa: BLE001 — непредставимость это ОТВЕТ
        rep = spec.point_report(p)
        return _f({"representable": False,
                   "error": f"{type(exc).__name__}: {exc}",
                   "violations": list(rep.violations),
                   "hint": "Рецепт вне геометрии спеки: либо расширить "
                           "границы патчем (с обоснованием), либо признать "
                           "anchor непредставимым — молча подгонять нельзя."})
    return _f({"representable": True, "z": z,
               "fractions": spec.to_fractions(p),
               "sigma_phr": float(p.sum())})


# ----------------------------------------------------------------------
# Прогоны и кампания
# ----------------------------------------------------------------------
@register(
    "get_runs",
    description=(
        "Фактические опыты проекта: координаты, измеренные отклики, "
        "ковариаты (телеметрия), origin-теги (seed/ветка/кампания/spec_hash)."),
    parameters={"type": "object", "properties": {
        "limit": {"type": "integer", "description": "сколько последних точек"}}})
def get_runs(ctx: ToolContext, limit: int = 50) -> Dict[str, Any]:
    runner = ctx.require_runner()
    points = list(getattr(runner, "points", []) or [])
    limit = int(max(1, min(int(limit or 50), 500)))
    tail = points[-limit:]
    rows = []
    for i, pt in enumerate(tail, start=len(points) - len(tail) + 1):
        d = pt.to_dict() if hasattr(pt, "to_dict") else dict(pt)
        d["index"] = i
        rows.append(d)
    return _f({
        "n_total": len(points), "n_returned": len(rows),
        "property_names": list(getattr(runner, "property_names", []) or []),
        "covariate_names": list(getattr(runner, "covariate_names", []) or []),
        "origin_counts": dict(runner.origin_counts()
                              if hasattr(runner, "origin_counts") else {}),
        "runs": rows,
    })


@register(
    "campaign_overview",
    description=(
        "Состояние кампании: свойства, размер общей базы, источники точек, "
        "ветки (цели, бюджет, статус, d_best, роли откликов, денежный канал), "
        "паспорт (метка, лоты сырья, anchor-рецепты, разрешение весов)."),
    parameters={"type": "object", "properties": {}})
def campaign_overview(ctx: ToolContext) -> Dict[str, Any]:
    runner = ctx.require_runner()
    out: Dict[str, Any] = {
        "property_names": list(getattr(runner, "property_names", []) or []),
        "n_points": len(getattr(runner, "points", []) or []),
        "origin_counts": dict(runner.origin_counts()
                              if hasattr(runner, "origin_counts") else {}),
        "campaign_label": getattr(runner, "campaign_label", ""),
        "material_lots": dict(getattr(runner, "material_lots", {}) or {}),
        "anchor_recipes": {k: dict(v) for k, v in
                           (getattr(runner, "anchor_recipes", {}) or {}).items()},
        "weighing": {"step_g": float(getattr(runner, "weighing_step_g", 0.0) or 0),
                     "grams_per_phr": float(
                         getattr(runner, "grams_per_phr", 0.0) or 0)},
        "process_levels": _f(getattr(runner, "process_levels", {}) or {}),
        "covariate_names": list(getattr(runner, "covariate_names", []) or []),
    }
    branches = []
    for bid, b in (getattr(runner, "branches", {}) or {}).items():
        branches.append({
            "id": bid, "name": getattr(b, "name", bid),
            "status": getattr(b, "status", None),
            "budget": getattr(b, "budget", None),
            "spent": getattr(b, "spent", None),
            "d_best": getattr(b, "d_best", None),
            "goals": {p: getattr(s, "kind", str(s))
                      for p, s in (getattr(b, "goal", {}) or {}).items()},
        })
    out["branches"] = branches
    return _f(out)


# ----------------------------------------------------------------------
# Знание: факты, решения, документы
# ----------------------------------------------------------------------
def _facts(ctx: ToolContext, scope: str = "") -> List[Dict[str, Any]]:
    if not (ctx.root and ctx.project):
        return []
    recs = read_log(ctx.root, ctx.project, "local_facts")
    if scope:
        s = str(scope).lower()
        recs = [r for r in recs
                if s in str(r.get("scope", "")).lower()
                or s in str(r.get("statement", "")).lower()]
    return recs


@register(
    "get_local_facts",
    description=(
        "L1-знание технолога (локальные факты цеха): ВЫСШИЙ приоритет, "
        "отменяет литературу и справочники. Конфликт L1 и веб-источника — "
        "сигнал, а не ошибка: не усредняй, спрашивай."),
    parameters={"type": "object", "properties": {
        "scope": {"type": "string",
                  "description": "фильтр по области/тексту факта"}}})
def get_local_facts(ctx: ToolContext, scope: str = "") -> Dict[str, Any]:
    facts = _facts(ctx, scope)
    return {"n": len(facts), "scope": scope, "facts": facts,
            "note": "Факты добавляет ТОЛЬКО человек; ты можешь лишь "
                    "предложить формулировку."}


@register(
    "get_decisions",
    description=("Журнал принятых решений компании (ADR): что решили, почему, "
                 "какие узлы затронуты, при каком spec_hash."),
    parameters={"type": "object", "properties": {
        "limit": {"type": "integer", "description": "сколько последних записей"}}})
def get_decisions(ctx: ToolContext, limit: int = 20) -> Dict[str, Any]:
    if not (ctx.root and ctx.project):
        return {"n": 0, "decisions": []}
    recs = read_log(ctx.root, ctx.project, "decisions", limit=int(limit or 20))
    return {"n": len(recs), "decisions": recs}


@register(
    "list_attachments",
    description="Документы, приложенные к сессии (паспорта, выгрузки, протоколы).",
    parameters={"type": "object", "properties": {}})
def list_attachments(ctx: ToolContext) -> Dict[str, Any]:
    session = ctx.require_session()
    return {"n": len(session.attachments),
            "files": [{"name": a.name, "mime": a.mime, "size": a.size,
                       "n_chars": a.n_chars, "truncated": a.truncated,
                       "note": a.note} for a in session.attachments]}


@register(
    "read_attachment",
    description=("Прочитать фрагмент приложенного документа. Числа из паспорта "
                 "цитируй ТОЛЬКО отсюда; если характеристики нет — так и "
                 "скажи и вынеси в OPEN_QUESTIONS."),
    parameters={"type": "object", "properties": {
        "name": {"type": "string", "description": "имя файла или префикс sha256"},
        "start": {"type": "integer", "description": "смещение в символах"},
        "length": {"type": "integer", "description": "длина фрагмента"}},
        "required": ["name"]})
def read_attachment(ctx: ToolContext, name: str, start: int = 0,
                    length: int = 8000) -> Dict[str, Any]:
    session = ctx.require_session()
    if find_attachment(session, name) is None:
        known = [a.name for a in session.attachments]
        raise ToolError(f"Документа '{name}' нет в сессии. Приложены: {known}.")
    return attachment_text(session, ctx.root, name, project=ctx.project,
                           start=int(start), length=int(length))
