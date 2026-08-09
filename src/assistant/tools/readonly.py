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
        "sigma_phr_static": [float(sum(v[0] for v in intervals.values())),
                             float(sum(v[1] for v in intervals.values()))],
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
    X = np.atleast_2d(np.asarray(getattr(runner, "X", np.empty((0, 0))), float))
    source = "база точек проекта"
    if X.size == 0 or X.shape[0] == 0:
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
