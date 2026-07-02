"""apps/campaign_state.py — C2 (§17.6.1): персистентность кампании.

Сохранение / загрузка / удаление проекта-КАМПАНИИ поверх
:class:`MixtureProcessRunner` (движок §17), в отличие от старой персистентности
`PipelineRunner` (M1–M8, `src/core/state.py`), которая уходит вместе с M1–M8-UI.

Канон (`.clinerules`, REBUILD_SPEC §5/§12):
  * ОДНА модель физики на проект — суррогаты НЕ сериализуются: физика живёт в
    ИЗМЕРЕННЫХ точках общей базы (И-1), а суррогаты детерминированно
    ПЕРЕОБУЧАЮТСЯ из точек при загрузке (:meth:`MixtureProcessRunner.fit_surrogates`,
    те же seed/kernel/mean_model ⇒ воспроизводимо).
  * Ветка (:class:`Branch`) — контейнер намерения; сериализуется целиком
    (`Branch.to_state`): цели, бюджет, статус, история, экономика, x*/d_best.
  * A0.6 / чистота проводника: НЕ сериализуем молча то, что не восстановимо
    честно. Ценовая нога ветки держит callable ``price_fn`` — сериализуем ЛИШЬ
    те функции, что несут явный сериализуемый дескриптор ``price_spec``
    (см. :func:`linear_price_fn`); иначе — явный отказ, а не тихая потеря цены.

Формат на диске: ``root/<name>/campaign.json`` (JSON-native; MISSING кодируется
как null через ``DataPoint.to_dict``). Оракул кампании — :class:`ManualOracle`
(истину вносит пользователь; ``evaluate`` — лишь демо-заполнение), поэтому
восстанавливается из ``property_names`` без хранения кода.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..core.schema import DataPoint, ProjectSchema
from ..core.schema_evolution import SchemaHistory
from ..design.branches import Branch
from ..optimize.desirability import DesirabilitySpec
from .mixture_process_runner import MixtureProcessRunner

FORMAT_VERSION = "campaign-v1"
_STATE_FILE = "campaign.json"


# ----------------------------------------------------------------------
# Ценовая нога ветки: сериализуемый дескриптор price_fn (A0.6 — честно или отказ)
# ----------------------------------------------------------------------
def linear_price_fn(prices: Sequence[float]) -> Callable[[Any], np.ndarray]:
    """Линейная цена состава ₽/кг = Σ(доля_i·цена_i) по mixture-долям.

    Возвращает callable ``Xc → цена состава`` (первые ``len(prices)`` координат
    составного вектора; процесс-оси на цену состава не влияют). В отличие от
    произвольного замыкания, помечает результат СЕРИАЛИЗУЕМЫМ дескриптором
    ``price_spec`` (``{"kind": "linear", "prices": [...]}``), чтобы ценовую ногу
    ветки можно было сохранить и честно восстановить (C2). Единый источник линейной
    ценовой ноги кампании — :func:`campaign_ui.make_linear_price_fn` делегирует сюда.
    """
    w = np.asarray(list(prices), float)

    def _fn(Xc):
        Xc = np.atleast_2d(np.asarray(Xc, float))
        q = min(w.shape[0], Xc.shape[1])
        return Xc[:, :q] @ w[:q]

    _fn.price_spec = {"kind": "linear", "prices": [float(v) for v in w]}
    return _fn


def price_fn_to_spec(price_fn: Any) -> Dict[str, Any]:
    """Сериализуемый дескриптор ценовой функции или явный отказ (A0.6).

    Читает атрибут ``price_spec`` функции (его вешает :func:`linear_price_fn`).
    Функцию без дескриптора сохранить нельзя — вместо тихой потери ценовой ноги
    поднимаем :class:`ValueError` с подсказкой (используйте ``linear_price_fn``
    или навесьте ``price_spec`` на свою функцию цены).
    """
    spec = getattr(price_fn, "price_spec", None)
    if not isinstance(spec, dict) or "kind" not in spec:
        raise ValueError(
            "Ценовую ногу ветки нельзя сериализовать: функция цены состава не "
            "несёт дескриптора price_spec. Соберите её через "
            "campaign_state.linear_price_fn(prices) (или навесьте атрибут "
            "price_spec={'kind': ..., ...}) — молчаливой потери цены нет (A0.6).")
    return dict(spec)


def price_fn_from_spec(spec: Dict[str, Any], *,
                       registry: Optional[Dict[str, Callable]] = None
                       ) -> Callable[[Any], np.ndarray]:
    """Восстановить ценовую функцию из дескриптора (обратное к :func:`price_fn_to_spec`).

    ``registry`` — необязательный словарь ``kind → builder(spec)`` для нестандартных
    ценовых ног; встроенный вид — ``linear``.
    """
    kind = spec.get("kind")
    if registry and kind in registry:
        return registry[kind](spec)
    if kind == "linear":
        return linear_price_fn(spec["prices"])
    raise ValueError(f"Неизвестный вид ценовой ноги '{kind}': передайте builder "
                     f"через registry={{'{kind}': ...}}.")


# ----------------------------------------------------------------------
# Runner ⇄ state (JSON-native)
# ----------------------------------------------------------------------
def _spec_to_dict(spec: DesirabilitySpec) -> Dict[str, Any]:
    return asdict(spec)


def _spec_from_dict(d: Dict[str, Any]) -> DesirabilitySpec:
    return DesirabilitySpec(**dict(d))


def _region_move_to_dict(mv: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-safe копия записи журнала движений области (deltas-кортежи → списки)."""
    out = dict(mv)
    deltas = out.get("deltas")
    if isinstance(deltas, dict):
        out["deltas"] = {k: [float(x) for x in v] for k, v in deltas.items()}
    return out


def runner_to_state(runner: MixtureProcessRunner) -> Dict[str, Any]:
    """Сериализовать состояние кампании в JSON-native словарь (без суррогатов).

    Сохраняет: полную схему проекта, историю версий + текущую версию/схему,
    baseline и GP-параметры, ОБЩУЮ базу точек (И-1), ветки целиком и ценовую
    конфигурацию (через сериализуемый дескриптор ``price_spec``), происхождение
    границ и журнал движений области. Суррогаты и numpy-кэши (X/Y/origin) —
    производные, восстанавливаются переобучением из точек при загрузке.
    """
    branch_cost: Dict[str, Any] = {}
    for bid, cfg in (getattr(runner, "_branch_cost", {}) or {}).items():
        branch_cost[bid] = {
            "price_spec": price_fn_to_spec(cfg["price_fn"]),
            "cost_spec": _spec_to_dict(cfg["cost_spec"]),
            "cost_name": str(cfg.get("cost_name", "price")),
            "rho_property": str(cfg["rho_property"]),
        }

    return {
        "format": FORMAT_VERSION,
        "oracle": {"kind": "manual",
                   "property_names": list(runner.property_names)},
        "runner": {
            "full_schema": runner.full_schema.to_dict(),
            "current_schema": runner.current_schema.to_dict(),
            "current_schema_version": int(runner.current_schema_version),
            "schema_history": [s.to_dict() for s in runner.schema_history.versions],
            "baseline": [float(v) for v in np.asarray(runner.baseline, float)],
            "seed": int(runner.seed),
            "n_restarts": int(runner.n_restarts),
            "gp_mean_model": str(runner.gp_mean_model),
            "gp_kernel": str(runner.gp_kernel),
            "points": [p.to_dict() for p in runner.points],
            "branches": {bid: br.to_state()
                         for bid, br in runner.branches.items()},
            "branch_cost": branch_cost,
            "border_origin": dict(getattr(runner, "_border_origin", {}) or {}),
            "region_moves": [_region_move_to_dict(m)
                             for m in getattr(runner, "_region_moves", []) or []],
            "drop_policy": str(getattr(runner, "_drop_policy", "exclude")),
        },
    }


def _default_oracle(property_names: Sequence[str]):
    """Оракул кампании по умолчанию — :class:`ManualOracle` (истину вносит user).

    Ленивый импорт: держит модуль независимым от Streamlit на уровне загрузки
    (``campaign_ui`` тянет ``streamlit``). ``ManualOracle`` сам по себе — чистый
    numpy-класс (только ``property_names`` + демо-``evaluate``)."""
    from .campaign_ui import ManualOracle
    return ManualOracle(list(property_names))


def runner_from_state(state: Dict[str, Any], *, oracle: Any = None,
                      price_fn_registry: Optional[Dict[str, Callable]] = None
                      ) -> MixtureProcessRunner:
    """Восстановить :class:`MixtureProcessRunner` из словаря :func:`runner_to_state`.

    ``oracle`` — если не задан, реконструируется :class:`ManualOracle` из
    сохранённых ``property_names`` (кампания меряет вручную; ``evaluate`` — лишь
    демо-заполнение). ``price_fn_registry`` — билдеры нестандартных ценовых ног.
    Суррогаты ПЕРЕОБУЧАЮТСЯ из точек (если база непуста) — одна модель физики
    (§5/§12), воспроизводимо по seed/kernel/mean_model.
    """
    if state.get("format") != FORMAT_VERSION:
        raise ValueError(f"Неподдерживаемый формат кампании: {state.get('format')!r} "
                         f"(ожидался {FORMAT_VERSION!r}).")
    r = state["runner"]
    full_schema = ProjectSchema.from_dict(r["full_schema"])
    if oracle is None:
        oracle = _default_oracle(state["oracle"]["property_names"])

    runner = MixtureProcessRunner(
        full_schema, oracle,
        baseline=list(r["baseline"]),
        seed=int(r.get("seed", 0)),
        n_restarts=int(r.get("n_restarts", 4)),
        gp_mean_model=str(r.get("gp_mean_model", "quadratic")),
        gp_kernel=str(r.get("gp_kernel", "matern52")),
    )

    # История версий + текущая схема/версия — восстанавливаем ВЕРНО (move_region
    # мог сдвинуть границы без bump; сериализуем current_schema отдельно).
    history = SchemaHistory()
    for s in r["schema_history"]:
        history.add(ProjectSchema.from_dict(s))
    runner.schema_history = history
    runner.current_schema = ProjectSchema.from_dict(r["current_schema"])
    runner.current_schema_version = int(r["current_schema_version"])

    runner.points = [DataPoint.from_dict(d) for d in r.get("points", [])]
    runner.branches = {bid: Branch.from_state(d)
                       for bid, d in r.get("branches", {}).items()}

    # Ценовая нога ветки — восстанавливаем price_fn из дескриптора и валидируем
    # через штатный set_branch_cost (проверит ветку и ρ-свойство).
    for bid, cfg in (r.get("branch_cost", {}) or {}).items():
        price_fn = price_fn_from_spec(cfg["price_spec"], registry=price_fn_registry)
        runner.set_branch_cost(
            bid, price_fn, _spec_from_dict(cfg["cost_spec"]),
            rho_property=str(cfg["rho_property"]),
            cost_name=str(cfg.get("cost_name", "price")))

    runner._border_origin = dict(r.get("border_origin", {}) or {})
    runner._region_moves = [dict(m) for m in r.get("region_moves", []) or []]
    runner._drop_policy = str(r.get("drop_policy", "exclude"))

    # Суррогаты — производные: переобучаем из точек (одна модель физики §5/§12).
    if runner.points:
        runner.fit_surrogates()
    return runner


# ----------------------------------------------------------------------
# Файловая персистентность: save / load / list / delete
# ----------------------------------------------------------------------
def _validate_name(name: str) -> str:
    name = (name or "").strip()
    if not name or name in (".", "..") or any(s in name for s in ("/", "\\")):
        raise ValueError(f"Недопустимое имя кампании: {name!r}")
    return name


def save_campaign(runner: MixtureProcessRunner, root: str | Path,
                  name: str) -> str:
    """Сохранить кампанию в ``root/<name>/campaign.json``; вернуть путь к файлу.

    Каталог создаётся при необходимости; существующий файл перезаписывается
    (сохранение — идемпотентно по имени)."""
    name = _validate_name(name)
    target = Path(root) / name
    target.mkdir(parents=True, exist_ok=True)
    path = target / _STATE_FILE
    state = runner_to_state(runner)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return str(path)


def load_campaign(root: str | Path, name: str, *, oracle: Any = None,
                  price_fn_registry: Optional[Dict[str, Callable]] = None
                  ) -> MixtureProcessRunner:
    """Загрузить кампанию по имени из ``root`` (обратное к :func:`save_campaign`)."""
    name = _validate_name(name)
    path = Path(root) / name / _STATE_FILE
    if not path.exists():
        raise FileNotFoundError(f"Кампания '{name}' не найдена в {root}.")
    state = json.loads(path.read_text(encoding="utf-8"))
    return runner_from_state(state, oracle=oracle,
                             price_fn_registry=price_fn_registry)


def list_campaigns(root: str | Path) -> List[str]:
    """Имена сохранённых кампаний в ``root`` (каталоги с ``campaign.json``)."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and (p / _STATE_FILE).exists())


def delete_campaign(root: str | Path, name: str) -> bool:
    """Удалить сохранённую кампанию (каталог ``root/<name>``) целиком.

    Защита от ошибок (как ``pipeline_runner.delete_project``): анти-traversal по
    имени, целевой каталог обязан быть прямым потомком ``root`` и валидной
    кампанией (наличие ``campaign.json``), иначе :class:`ValueError`. Возвращает
    ``True`` при удалении, ``False`` — если кампании нет."""
    name = _validate_name(name)
    root = Path(root).resolve()
    target = (root / name).resolve()
    if target.parent != root:
        raise ValueError(f"Кампания вне каталога кампаний: {target}")
    if not target.exists():
        return False
    if not (target / _STATE_FILE).exists():
        raise ValueError(f"'{name}' не похож на кампанию (нет {_STATE_FILE}) — "
                         f"удаление отклонено.")
    shutil.rmtree(target)
    return True
