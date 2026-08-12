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
from ..design.branches import Branch, ROLE_PRICE_INPUT
from ..design.linked_axes import ProcessLink
from ..design.phr_sampler import PhrSpec
from ..optimize.desirability import ChanceConstraint, DesirabilitySpec
from .mixture_process_runner import MixtureProcessRunner


FORMAT_VERSION = "campaign-v1"
_STATE_FILE = "campaign.json"

#: iter76: ЧЕРНОВИК НАСТРОЕК несобранного проекта — значения полей формы
#: «🆕 Новый проект». До этого файла сохранить можно было только СОБРАННЫЙ
#: проект (campaign.json), и заполненная форма терялась при закрытии вкладки —
#: замкнутый круг «собрать не могу (ошибка), сохранить не могу (не собран)».
_SETUP_DRAFT_FILE = "setup_draft.json"

#: Ключи формы сетапа, которые НЕ идут в черновик: объекты конкретного прогона
#: (PhrSpec-объект, numpy-план seed, состояние data_editor) и КНОПКИ (их bool
#: живёт в session_state, но Streamlit запрещает класть его обратно —
#: StreamlitValueAssignmentNotAllowedError при префилле) — они либо не
#: JSON-сериализуемы, либо восстанавливаются из других полей.
_SETUP_DRAFT_SKIP = frozenset({
    "setup_seed_X", "setup_seed_Y", "setup_seed_df", "setup_seed_df_sig",
    "setup_seed_editor", "setup_phr_spec_obj", "setup_phr_tree",
    # кнопки формы сетапа (st.button / st.download_button)
    "setup_build", "setup_propose_seed", "setup_commit_seed",
    "setup_fill_demo", "setup_seed_dl",
})


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


def _chance_to_dict(con: ChanceConstraint) -> Dict[str, Any]:
    """iter43: ``ChanceConstraint`` → JSON-safe словарь.

    ``±inf`` (односторонние ограничения — штатный случай ΔE ≤ max) пишется как
    ``null``: `json.dump` умеет писать нестандартный литерал ``Infinity``, но
    такой файл перестаёт быть валидным JSON для внешних читателей. ``None``
    восстанавливается обратно в ``∓inf`` (см. :func:`_chance_from_dict`).
    """
    return {
        "y_min": (float(con.y_min) if np.isfinite(con.y_min) else None),
        "y_max": (float(con.y_max) if np.isfinite(con.y_max) else None),
        "alpha": float(con.alpha),
    }


def _chance_from_dict(d: Dict[str, Any]) -> ChanceConstraint:
    """iter43: словарь → ``ChanceConstraint`` (``null`` → ``∓inf``)."""
    y_min = d.get("y_min", None)
    y_max = d.get("y_max", None)
    return ChanceConstraint(
        y_min=(-np.inf if y_min is None else float(y_min)),
        y_max=(np.inf if y_max is None else float(y_max)),
        alpha=float(d.get("alpha", 0.05)))


def _link_to_dict(lk: ProcessLink) -> Dict[str, Any]:
    """P3.3: ``ProcessLink`` → JSON-safe словарь (``±inf`` → ``null``).

    Тот же приём, что у :func:`_chance_to_dict`: литерал ``Infinity`` сделал
    бы файл невалидным JSON для внешних читателей.
    """
    return {
        "name": str(lk.name), "minuend": str(lk.minuend),
        "subtrahend": str(lk.subtrahend),
        "lo": (float(lk.lo) if np.isfinite(lk.lo) else None),
        "hi": (float(lk.hi) if np.isfinite(lk.hi) else None),
    }


def _region_move_to_dict(mv: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-safe копия записи журнала движений области (deltas-кортежи → списки)."""

    out = dict(mv)
    deltas = out.get("deltas")
    if isinstance(deltas, dict):
        out["deltas"] = {k: [float(x) for x in v] for k, v in deltas.items()}
    return out


def runner_to_state(runner: MixtureProcessRunner, *,
                    draft: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Сериализовать состояние кампании в JSON-native словарь (без суррогатов).

    Сохраняет: полную схему проекта, историю версий + текущую версию/схему,
    baseline и GP-параметры, ОБЩУЮ базу точек (И-1), ветки целиком и ценовую
    конфигурацию (через сериализуемый дескриптор ``price_spec``), происхождение
    границ и журнал движений области. Суррогаты и numpy-кэши (X/Y/origin) —
    производные, восстанавливаются переобучением из точек при загрузке.

    ``draft`` — необязательный JSON-native черновик UI (например, предложенный,
    но ещё НЕ зафиксированный стартовый дизайн: ``{"seed_X": [[...]],
    "seed_Y": [[...|null]]}``). Без него проект, сохранённый до ``commit_seed``,
    терял стартовый план — пользователь видел «пустую» загрузку (A0.6).
    """
    branch_cost: Dict[str, Any] = {}
    for bid, cfg in (getattr(runner, "_branch_cost", {}) or {}).items():
        branch_cost[bid] = {
            "price_spec": price_fn_to_spec(cfg["price_fn"]),
            "cost_spec": _spec_to_dict(cfg["cost_spec"]),
            "cost_name": str(cfg.get("cost_name", "price")),
            "rho_property": str(cfg["rho_property"]),
        }
    # iter43 (§43.1): вероятностные ограничения ветки — dataclass без callable,
    # сериализуется целиком (в отличие от ценовой ноги отказ невозможен).
    branch_chance: Dict[str, Any] = {
        bid: {prop: _chance_to_dict(con) for prop, con in cons.items()}
        for bid, cons in (getattr(runner, "_branch_chance", {}) or {}).items()
        if cons}

    state: Dict[str, Any] = {

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
            "n_blocks_start": int(getattr(runner, "n_blocks_start", 1)),
            # blocking-метаданные (показ/Excel): фактор блокировки + имена блоков
            # (JSON-ключи — строки; при загрузке возвращаем int-номера).
            "block_factor": str(getattr(runner, "block_factor", "") or ""),
            "block_names": {str(int(k)): str(v) for k, v in
                            (getattr(runner, "block_names", {}) or {}).items()},
            "points": [p.to_dict() for p in runner.points],
            "branches": {bid: br.to_state()
                         for bid, br in runner.branches.items()},
            "branch_cost": branch_cost,
            # iter43 (§43.1): вероятностные ограничения ветки
            "branch_chance": branch_chance,
            "border_origin": dict(getattr(runner, "_border_origin", {}) or {}),

            # iter31: проектные функциональные группы (политика сэмплирования)
            "sampling_groups": [list(g) for g in
                                (getattr(runner, "sampling_groups", []) or [])],
            # iter40 (UI_REVISION_SPEC): политика кампании — phr-спека
            # (decode-слой iter33/38), метка кампании и обязательные 2D-пары
            # preflight (iter37 п.2/п.4). Без сериализации save/load молча
            # откатывал сэмплер/оптимизатор на бокс и терял метаданные
            # origin_tag новых точек (A0.6 — тихая потеря недопустима).
            "phr_spec": (runner.phr_spec.to_dicts()
                         if getattr(runner, "phr_spec", None) is not None
                         else None),
            "campaign_label": str(getattr(runner, "campaign_label", "") or ""),
            # P2.1: дискретные уровни process-осей (ФИЗИЧЕСКИЕ единицы).
            # Без сохранения после load план снова стал бы непрерывным, и
            # кампания молча начала бы предлагать недостижимые режимы (A0.6).
            "process_levels": {str(k): [float(v) for v in vals] for k, vals in
                               (getattr(runner, "process_levels", {}) or {}).items()},
            # P3.3: связанные process-оси (dT_head = A − B ∈ [lo, hi]).
            # Без сериализации save/load молча вернул бы независимые оси, и
            # кампания предлагала бы нереализуемые перепады (A0.6).
            "process_links": [_link_to_dict(lk) for lk in
                              (getattr(runner, "process_links", []) or [])],
            "preflight_pairs": [[list(a), list(b)] for a, b in
                                (getattr(runner, "preflight_pairs", []) or [])],
            # P2.3: паспорт кампании (CAMPAIGN_SPEC_PVC §3) — лоты сырья,
            # anchor-рецепты (phr), разрешение весов. Записывается ДО первого
            # замера; без сериализации save/load молча терял бы паспорт (A0.6).
            "material_lots": {str(k): str(v) for k, v in
                              (getattr(runner, "material_lots", {}) or {}).items()},
            "anchor_recipes": {str(rn): {str(c): float(v)
                                         for c, v in (rec or {}).items()}
                               for rn, rec in
                               (getattr(runner, "anchor_recipes", {}) or {}).items()},
            "weighing_step_g": float(getattr(runner, "weighing_step_g", 0.0) or 0.0),
            "grams_per_phr": float(getattr(runner, "grams_per_phr", 0.0) or 0.0),
            # P3.1: объявленные КОВАРИАТЫ базы (телеметрия прогона). Значения
            # per-point едут внутри origin_tag точек (points → to_dict) и
            # отдельного канала не требуют; без сериализации ОБЪЯВЛЕНИЯ
            # столбцы после load молча пропадали бы из таблиц (A0.6).
            "covariate_names": [str(n) for n in
                                (getattr(runner, "covariate_names", []) or [])],
            # iter75: ЭКОНОМИКА ПРОЕКТА (ρ-отклик + цены сырья + единицы +
            # роль ρ по умолчанию). Проектный уровень, не ветка: без
            # сериализации save/load молча вернул бы кампанию без
            # себестоимости, а ветки — к ручному вводу цен (A0.6).
            "economics_enabled": bool(getattr(runner, "economics_enabled",
                                              False)),
            "rho_property": str(getattr(runner, "rho_property", "") or ""),
            "rho_unit": str(getattr(runner, "rho_unit", "") or ""),
            "currency_unit": str(getattr(runner, "currency_unit", "") or ""),
            "mass_unit": str(getattr(runner, "mass_unit", "") or ""),
            "component_prices": {str(k): float(v) for k, v in
                                 (getattr(runner, "component_prices", {})
                                  or {}).items()},
            "rho_default_role": str(getattr(runner, "rho_default_role", "")
                                    or ""),
            "region_moves": [_region_move_to_dict(m)
                             for m in getattr(runner, "_region_moves", []) or []],
            "drop_policy": str(getattr(runner, "_drop_policy", "exclude")),
        },
    }
    if draft:
        state["draft"] = dict(draft)
    return state


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
        n_blocks_start=int(r.get("n_blocks_start", 1)),
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

    # iter43 (§43.1): вероятностные ограничения ветки — ШТАТНЫМ сеттером
    # (валидация имён откликов). Старые сейвы без ключа → ограничений нет.
    for bid, cons in (r.get("branch_chance", {}) or {}).items():
        runner.set_branch_chance(
            bid, {prop: _chance_from_dict(d) for prop, d in (cons or {}).items()})

    runner.block_factor = str(r.get("block_factor", "") or "")

    runner.block_names = {int(k): str(v) for k, v in
                          (r.get("block_names", {}) or {}).items()}
    runner._border_origin = dict(r.get("border_origin", {}) or {})
    # iter31: проектные группы (старые сейвы без ключа → пусто)
    runner.sampling_groups = [list(g) for g in
                              (r.get("sampling_groups", []) or [])]
    # iter40: политика кампании — восстанавливаем ШТАТНЫМИ сеттерами
    # (валидация против полной схемы); старые сейвы без ключей → выключено.
    spec_dicts = r.get("phr_spec")
    if spec_dicts:
        runner.set_phr_spec(PhrSpec.from_dicts(spec_dicts))
    label = str(r.get("campaign_label", "") or "")
    if label:
        runner.set_campaign_label(label)
    pairs = r.get("preflight_pairs", []) or []
    if pairs:
        runner.set_preflight_pairs(pairs)
    # P2.1: уровни восстанавливаются ШТАТНЫМ сеттером (валидация имён и
    # границ); старый сейв без ключа → оси непрерывны, как и были.
    levels = r.get("process_levels", {}) or {}
    if levels:
        runner.set_process_levels(levels)
    # P3.3: связки — ШТАТНЫМ сеттером (валидация имён/полосы/конфликтов);
    # старый сейв без ключа → оси независимы, как и были.
    links = r.get("process_links", []) or []
    if links:
        runner.set_process_links(links)
    # P2.3: паспорт кампании — ШТАТНЫМИ сеттерами (валидация имён/значений);
    # старый сейв без ключей → паспорт пуст (лоты/anchor'ы/весы не заданы).
    lots = r.get("material_lots", {}) or {}
    if lots:
        runner.set_material_lots(lots)
    anchors = r.get("anchor_recipes", {}) or {}
    if anchors:
        runner.set_anchor_recipes(anchors)
    step_g = float(r.get("weighing_step_g", 0.0) or 0.0)
    gpp = float(r.get("grams_per_phr", 0.0) or 0.0)
    if step_g > 0 or gpp > 0:
        runner.set_weighing_resolution(step_g, gpp)
    # P3.1: ковариаты — ШТАТНЫМ сеттером (валидация имён); старый сейв без
    # ключа → столбцы не объявлены (значения в origin_tag точек целы).
    cov_names = r.get("covariate_names", []) or []
    if cov_names:
        runner.set_covariate_names(cov_names)
    # iter75: экономика проекта — ШТАТНЫМ сеттером (валидация ρ против откликов
    # и цен против компонентов). Старый сейв без ключа: ключа нет ⇒ экономика
    # ВЫКЛЮЧЕНА, а не «включена без ρ» — чужой проект не мутируем догадкой
    # (A0.6). Дефолт конструктора True рассчитан на НОВЫЙ проект, где форма
    # сетапа сразу задаёт ρ.
    if "economics_enabled" not in r:
        runner.set_project_economics(enabled=False)
    elif not bool(r.get("economics_enabled")):
        runner.set_project_economics(enabled=False)
    elif str(r.get("rho_property", "") or ""):
        runner.set_project_economics(
            enabled=True,
            rho_property=str(r.get("rho_property")),
            prices={str(k): float(v) for k, v in
                    (r.get("component_prices", {}) or {}).items()},
            rho_unit=str(r.get("rho_unit", "") or ""),
            currency_unit=str(r.get("currency_unit", "") or ""),
            mass_unit=str(r.get("mass_unit", "") or ""),
            rho_default_role=str(r.get("rho_default_role")
                                 or ROLE_PRICE_INPUT))
    else:
        # «Включена, но НЕ НАСТРОЕНА»: пользователь ещё не назвал ρ (проект
        # сохранён до заполнения блока экономики). Через сеттер это не провести
        # — он справедливо требует ρ; поэтому восстанавливаем ровно то же
        # состояние, что даёт конструктор, не выдавая его за настроенную ногу.
        runner.economics_enabled = True
        runner.rho_property = ""
        runner.component_prices = {}
        runner.rho_unit = str(r.get("rho_unit", "") or "")
        runner.currency_unit = str(r.get("currency_unit", "") or "")
        runner.mass_unit = str(r.get("mass_unit", "") or "")
        runner.rho_default_role = str(r.get("rho_default_role")
                                      or ROLE_PRICE_INPUT)
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
        raise ValueError(f"Недопустимое имя проекта: {name!r}")
    return name


def save_campaign(runner: MixtureProcessRunner, root: str | Path,
                  name: str, *, draft: Optional[Dict[str, Any]] = None) -> str:
    """Сохранить проект (кампанию) в ``root/<name>/campaign.json``; вернуть путь.

    Каталог создаётся при необходимости; существующий файл перезаписывается
    (сохранение — идемпотентно по имени). ``draft`` — необязательный черновик
    UI (см. :func:`runner_to_state`): например, предложенный, но ещё не
    зафиксированный стартовый дизайн."""
    name = _validate_name(name)
    target = Path(root) / name
    target.mkdir(parents=True, exist_ok=True)
    path = target / _STATE_FILE
    state = runner_to_state(runner, draft=draft)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    # iter76: черновик настроек относился к НЕсобранному проекту; после
    # сохранения собранного он устарел (источник истины — campaign.json,
    # форма префиллится из раннера). Оставить его — значит однажды загрузить
    # старые поля поверх новых.
    stale = target / _SETUP_DRAFT_FILE
    if stale.exists():
        stale.unlink()
    return str(path)


def load_campaign(root: str | Path, name: str, *, oracle: Any = None,
                  price_fn_registry: Optional[Dict[str, Callable]] = None
                  ) -> MixtureProcessRunner:
    """Загрузить проект по имени из ``root`` (обратное к :func:`save_campaign`)."""
    name = _validate_name(name)
    path = Path(root) / name / _STATE_FILE
    if not path.exists():
        raise FileNotFoundError(f"Проект '{name}' не найден в {root}.")
    state = json.loads(path.read_text(encoding="utf-8"))
    return runner_from_state(state, oracle=oracle,
                             price_fn_registry=price_fn_registry)


def load_campaign_draft(root: str | Path, name: str) -> Optional[Dict[str, Any]]:
    """Черновик UI сохранённого проекта (ключ ``draft``) или ``None``.

    Обратное к ``save_campaign(..., draft=...)``: возвращает словарь черновика
    (например, стартовый дизайн до фиксации) без реконструкции раннера."""
    name = _validate_name(name)
    path = Path(root) / name / _STATE_FILE
    if not path.exists():
        raise FileNotFoundError(f"Проект '{name}' не найден в {root}.")
    state = json.loads(path.read_text(encoding="utf-8"))
    draft = state.get("draft")
    return dict(draft) if isinstance(draft, dict) else None


def list_campaigns(root: str | Path) -> List[str]:
    """Имена сохранённых проектов в ``root``.

    Проект — каталог с ``campaign.json`` (собранный) ИЛИ с ``setup_draft.json``
    (iter76: черновик настроек, проект ещё не собран). Черновик должен быть
    виден в списке загрузки — иначе сохранённая до сборки форма недостижима.
    """
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and ((p / _STATE_FILE).exists()
                                     or (p / _SETUP_DRAFT_FILE).exists()))


def campaign_exists(root: str | Path, name: str) -> bool:
    """Есть ли у проекта СОБРАННОЕ состояние (``campaign.json``)."""
    return (Path(root) / _validate_name(name) / _STATE_FILE).exists()


# ----------------------------------------------------------------------
# iter76: черновик настроек НЕсобранного проекта (поля формы сетапа)
# ----------------------------------------------------------------------
def setup_draft_fields(state: Any) -> Dict[str, Any]:
    """Снимок полей формы «🆕 Новый проект» из ``session_state`` (чистая).

    Берутся все ключи ``setup_*`` со СКАЛЯРНЫМИ значениями (str/int/float/
    bool/None) — ровно то, что человек ввёл в форму и что можно честно
    вернуть через ``setup_prefill_pending``. Объекты прогона (numpy-план,
    PhrSpec, состояние редактора) исключены явно (:data:`_SETUP_DRAFT_SKIP`).
    """
    out: Dict[str, Any] = {}
    for k, v in dict(state or {}).items():
        key = str(k)
        if not key.startswith("setup_") or key in _SETUP_DRAFT_SKIP:
            continue
        if v is None or isinstance(v, (str, int, float, bool)):
            out[key] = v
    return out


def save_setup_draft(root: str | Path, name: str,
                     fields: Dict[str, Any]) -> str:
    """Сохранить черновик настроек в ``root/<name>/setup_draft.json``.

    Черновик — это «ссылка» несобранного проекта: каталог появляется с
    момента ввода имени, а не с момента сборки, поэтому рядом уже могут жить
    переписка ассистента (``assistant/``) и, позже, ``campaign.json``.
    """
    name = _validate_name(name)
    if not fields:
        raise ValueError("Черновик пуст: в форме «🆕 Новый проект» нет "
                         "заполненных полей — сохранять нечего.")
    target = Path(root) / name
    target.mkdir(parents=True, exist_ok=True)
    path = target / _SETUP_DRAFT_FILE
    path.write_text(json.dumps(dict(fields), ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return str(path)


def load_setup_draft(root: str | Path, name: str) -> Optional[Dict[str, Any]]:
    """Черновик настроек проекта или ``None`` (черновика нет — не ошибка)."""
    path = Path(root) / _validate_name(name) / _SETUP_DRAFT_FILE
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data) if isinstance(data, dict) else None


def delete_campaign(root: str | Path, name: str) -> bool:
    """Удалить сохранённый проект (каталог ``root/<name>``) целиком.

    Защита от ошибок (как ``pipeline_runner.delete_project``): анти-traversal по
    имени, целевой каталог обязан быть прямым потомком ``root`` и валидным
    проектом (наличие ``campaign.json``), иначе :class:`ValueError`. Возвращает
    ``True`` при удалении, ``False`` — если проекта нет."""
    name = _validate_name(name)
    root = Path(root).resolve()
    target = (root / name).resolve()
    if target.parent != root:
        raise ValueError(f"Проект вне каталога проектов: {target}")
    if not target.exists():
        return False
    # iter76: проектом считается и черновик настроек (setup_draft.json) —
    # несобранный, но сохранённый проект тоже должен удаляться штатно.
    if not ((target / _STATE_FILE).exists()
            or (target / _SETUP_DRAFT_FILE).exists()):
        raise ValueError(f"'{name}' не похож на проект (нет {_STATE_FILE} и "
                         f"{_SETUP_DRAFT_FILE}) — удаление отклонено.")
    shutil.rmtree(target)
    return True
