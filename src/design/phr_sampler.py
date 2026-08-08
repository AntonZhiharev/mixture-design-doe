"""
design/phr_sampler.py — этап A decode-слоя (DECODE_LAYER_PROPOSAL, iter33):
z-сэмплер-плагин parts/phr-спеки.

Двухслойная параметризация phr-рецептур (PVC-компаунды и т.п.):

    z (design coords) --decode--> p (phr) --w_i = p_i / Σp--> x (доли, Σ=1)

Спека — список узлов, каждый в одном из 4 режимов координат:

  * ``absolute``  — phr равномерно в ``[lo, hi]`` (в т.ч. ТОТАЛ группы);
    опционально с ДИНАМИЧЕСКИМ ПОТОЛКОМ ``cap_to``/``cap_ratio``:
    ``cap_to`` — имя узла ИЛИ СПИСОК имён (ФАЗА, значения складываются):
    ``p ∈ [lo, min(hi, cap_ratio · Σ value(cap_to))]`` — ТРАПЕЦИЯ, не клин.
    Кейс (сессии 05.08.2026, UV_CSFCP): растворимость ограничивает УФ
    только СВЕРХУ и определяется ПЛАСТИФИКАТОРНОЙ ФАЗОЙ целиком
    (`UV ≤ 0.03·(DINP + ESO)`), нижняя граница 0.05 phr — требование
    по защите и от пластификатора НЕ зависит. ``ratio_to`` здесь неверен:
    он масштабирует ОБА конца (клин), вшивая положительную корреляцию с
    доминирующей осью и монотонный prior, которого физика не требует;
    опционально ``scale='log'`` (iter47/B5): z-координата оси — ``ln phr``,
    сэмплинг равномерен в ``[ln lo, ln hi]`` (лог-равномерная маргиналь по
    phr: плотность в нижней декаде, где живёт сатурирующий отклик);
    cap-потолок задан В PHR и применяется ПОСЛЕ экспоненцирования —
    пер-точечно логарифмируется уже суженная граница
    ``min(hi, cap_ratio·Σref)``, а не сами референсы;
  * ``share_of``  — доля родительского узла ``of`` в ``[lo, hi] ⊆ [0,1]``;
    доли одной группы связаны Σ=1 (раскладка без rejection —
    :func:`core.simplex._narrowing_split`, канон iter31: uniform-MARGINAL
    по физически значимым осям); опционально ``min_phr``/``max_phr`` —
    ТЕХНОЛОГИЧЕСКИЕ лимиты в phr (складской лимит, техминимум), которые
    НЕ являются боксом по доле: они превращаются в CONDITIONAL NARROWING
    после розыгрыша тотала группы ``T`` (iter45/B1):
    ``φ ∈ [max(φᴸ, min_phr/T), min(φᵁ, max_phr/T)]`` — эффективный
    диапазон доли зависит от точки и НЕМОНОТОНЕН по ``T``
    (см. :meth:`PhrSpec.share_bounds_at_total`);
  * ``ratio_to``  — коэффициент к ПРОИЗВОЛЬНОМУ узлу ``to`` (не обязательно
    родителю — поэтому DAG, а не дерево): ``p = r · value(to)``,
    ``r ∈ [lo, hi]`` (пример: «SBM = 0.02…0.09 × Σ стабилизатора»);
  * ``fixed``     — константа phr (базовый компонент: смола = 100).

Новый контракт долей (iter46/B2, ревизия «pvc_edge_v1»): legacy ``share_of``
даёт КАЖДОМУ члену группы свою z-ось, поэтому пары (φ, 1−φ) точно
коллинеарны — rank(Z) < dim_z, cond → ∞, ARD-длины пар не идентифицируются.
Вместо этого — роли с явным замыканием:

  * ``share_free``    — свободная доля k=2-группы (z-ось, ``share_range``);
  * ``share_closure`` — замыкание k=2-группы: z-оси НЕТ, диапазон
    ПРОИЗВОДНЫЙ ``[1−φᵁ_free, 1−φᴸ_free]``; задавать range запрещено —
    ошибка валидации, не тихое игнорирование (B8);
  * ``share_simplex`` — совместные доли k≥3-группы (все члены): z-осей
    k−1, ПОСЛЕДНИЙ член группы — зависимая координата ``1 − Σ остальных``
    (замыкание — внутреннее свойство сэмплера, не роль узла, C1).

Схема сериализации v2 (iter46/B6): роли ``FIXED`` / ``ABSOLUTE`` /
``ABSOLUTE_CAPPED`` / ``GROUP_TOTAL`` / ``GROUP_TOTAL_FIXED`` /
``SHARE_FREE`` / ``SHARE_CLOSURE`` / ``SHARE_SIMPLEX`` / ``RATIO_TO``;
ключи ``role`` / ``range`` / ``share_range`` / ``group`` / ``members`` /
``reference`` / ``scale`` / ``spec_version`` / ``group_order`` —
см. :meth:`PhrSpec.from_dicts`.
Legacy-схема v1 (``mode``) продолжает работать без изменений: старые сейвы
и хеши валидны, обе схемы различимы по ключам.

``group_order`` (iter48/B4, C2) — приоритет GROUP_TOTAL-групп кампании:
ТОЧНАЯ перестановка множества групп с ролью ``GROUP_TOTAL``
(``GROUP_TOTAL_FIXED`` исключается: тотал детерминирован, стратифицировать
нечего). Входит в ``to_dicts()``/``spec_hash()``. ВАЖНО (честная граница):
в phr-пути тоталы групп — НЕЗАВИСИМЫЕ absolute-оси, каждая получает ТОЧНУЮ
равномерную маргиналь независимо от порядка — меру ЭТОГО сэмплера
``group_order`` не меняет (в отличие от fraction-space группового сэмплера
iter31, где точную маргиналь получает только первая группа: KS≈0.019
против ≈0.38 у последующих, замер iter34). Порядок фиксируется как
контракт кампании (приоритет осей, CAMPAIGN_SPEC_PVC §1) и обязан входить
в отпечаток: при любом использовании fraction-space groups или будущей
стратификации без него план не воспроизводится.

Узел, на который ссылаются share_of-дети, — ВНУТРЕННИЙ (его phr раздаётся
детям, компонентом смеси он не является). Компоненты = листья DAG.

Статическая валидация интервальной арифметикой выполняется В КОНСТРУКТОРЕ
(пустые пересечения — ошибка КОНФИГА, а не sample-time): циклы, неизвестные
ссылки, невыполнимые доли группы (Σlo>1 или Σhi<1), нулевые референсы.

Этап A НЕ меняет схему/модель: :meth:`PhrSpec.sample_candidates` выдаёт
готовые кандидаты-доли (Σ=1 конструкцией) для пула кандидатов раннера;
:meth:`PhrSpec.fraction_bounds` даёт консервативный fraction-бокс для
построения mixture-блока схемы (та же математика экстремальных тоталей,
что :func:`core.simplex.parts_ranges_to_fraction_bounds`). Слои m (навеска,
quantize, премикс) и модель в z — этап B.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.simplex import _narrowing_split, parts_ranges_to_fraction_bounds

MODE_ABSOLUTE = "absolute"
MODE_SHARE_OF = "share_of"
MODE_RATIO_TO = "ratio_to"
MODE_FIXED = "fixed"
# Новый контракт долей (iter46/B2): closure/зависимый член — без z-оси.
MODE_SHARE_FREE = "share_free"
MODE_SHARE_CLOSURE = "share_closure"
MODE_SHARE_SIMPLEX = "share_simplex"
_NEW_SHARE_MODES = (MODE_SHARE_FREE, MODE_SHARE_CLOSURE, MODE_SHARE_SIMPLEX)
_SHARE_MODES = (MODE_SHARE_OF,) + _NEW_SHARE_MODES
_MODES = (MODE_ABSOLUTE, MODE_SHARE_OF, MODE_RATIO_TO,
          MODE_FIXED) + _NEW_SHARE_MODES
_SCALES = ("linear", "log")

_TOL = 1e-9

# ----------------------------------------------------------------------
# Схема сериализации v2 (iter46/B6): role → (mode, обязательные ключи,
# допустимые ключи). Всё вне allowed — ошибка (в т.ч. legacy-ключи
# lo/hi/of/to/mode и range у closure/fixed — B8: не тихое игнорирование).
# ----------------------------------------------------------------------
_ROLE_TABLE: Dict[str, Tuple[str, frozenset, frozenset]] = {
    "FIXED": ("fixed",
              frozenset({"value"}),
              frozenset({"name", "role", "value"})),
    "ABSOLUTE": ("absolute",
                 frozenset({"range"}),
                 frozenset({"name", "role", "range", "scale"})),
    "ABSOLUTE_CAPPED": ("absolute",
                        frozenset({"range", "cap_to", "cap_ratio"}),
                        frozenset({"name", "role", "range", "scale",
                                   "cap_to", "cap_ratio"})),
    "GROUP_TOTAL": ("absolute",
                    frozenset({"range", "members"}),
                    frozenset({"name", "role", "range", "members"})),
    "GROUP_TOTAL_FIXED": ("fixed",
                          frozenset({"value", "members"}),
                          frozenset({"name", "role", "value", "members"})),
    "SHARE_FREE": ("share_free",
                   frozenset({"group", "share_range"}),
                   frozenset({"name", "role", "group", "share_range",
                              "min_phr", "max_phr"})),
    "SHARE_CLOSURE": ("share_closure",
                      frozenset({"group"}),
                      frozenset({"name", "role", "group",
                                 "min_phr", "max_phr"})),
    "SHARE_SIMPLEX": ("share_simplex",
                      frozenset({"group", "share_range"}),
                      frozenset({"name", "role", "group", "share_range",
                                 "min_phr", "max_phr"})),
    "RATIO_TO": ("ratio_to",
                 frozenset({"reference", "range"}),
                 frozenset({"name", "role", "reference", "range"})),
}


@dataclass
class PhrNode:
    """Узел phr-спеки. ``ref`` — имя референса (``of`` у share_of, ``to``
    у ratio_to); у absolute/fixed референса нет. ``cap_refs``/``cap_ratio``
    (только absolute) — динамический потолок по СУММЕ референсов (фазе):
    ``hi_eff = min(hi, cap_ratio · Σ value(cap_refs))``.
    ``min_phr``/``max_phr`` (share-режимы, iter45/B1) — технологические
    лимиты узла В PHR: conditional narrowing доли после розыгрыша тотала,
    а НЕ бокс по доле (``None`` — лимита нет). ``scale`` (только absolute,
    iter46/B6) — шкала сэмплинга оси: ``linear`` | ``log``; при ``log``
    (iter47/B5) z-координата оси — ``ln phr``: сэмплинг/границы/clip идут
    в лог-шкале, cap-потолок — в phr (применяется после экспоненцирования).
    У ``share_closure`` границы НЕ задаются (``lo=hi=0`` — сентинель):
    диапазон производный от свободного партнёра (iter46/B8)."""
    name: str
    mode: str
    lo: float = 0.0
    hi: float = 0.0
    value: float = 0.0
    ref: str = ""
    cap_refs: Tuple[str, ...] = ()
    cap_ratio: float = 0.0
    min_phr: Optional[float] = None
    max_phr: Optional[float] = None
    scale: str = "linear"


class PhrSpec:
    """DAG phr-узлов: валидация, сэмплинг z, decode/encode, доли.

    Использование::

        spec = PhrSpec.from_dicts([
            {"name": "resin", "mode": "fixed", "value": 100.0},
            {"name": "stab_total", "mode": "absolute", "lo": 2, "hi": 5},
            {"name": "Ca_st", "mode": "share_of", "of": "stab_total",
             "lo": 0.2, "hi": 0.7},
            ...
        ])
        X = spec.sample_candidates(200, seed=0)   # (n, q) доли, Σ=1
    """

    def __init__(self, nodes: Sequence[PhrNode],
                 schema_version: Optional[int] = None,
                 group_order: Optional[Sequence[str]] = None):
        self.nodes: List[PhrNode] = list(nodes)
        if not self.nodes:
            raise ValueError("Пустая phr-спека недопустима.")
        self._by_name: Dict[str, PhrNode] = {}
        for nd in self.nodes:
            if not nd.name:
                raise ValueError("Узел без имени недопустим.")
            if nd.name in self._by_name:
                raise ValueError(f"Дублирующееся имя узла '{nd.name}'.")
            if nd.mode not in _MODES:
                raise ValueError(
                    f"Узел '{nd.name}': неизвестный режим '{nd.mode}' "
                    f"(допустимо: {list(_MODES)}).")
            self._by_name[nd.name] = nd
        self._validate_refs()
        # share-группы: родитель → упорядоченный список имён детей
        self._share_groups: Dict[str, List[str]] = {}
        for nd in self.nodes:
            if nd.mode in _SHARE_MODES:
                self._share_groups.setdefault(nd.ref, []).append(nd.name)
        # iter46/B2: статические границы ДОЛИ каждого share-узла (у closure
        # — производные от свободного партнёра) и производные члены групп
        # (closure / последний член simplex-группы) — БЕЗ собственной z-оси
        self._share_base: Dict[str, Tuple[float, float]] = {}
        self._derived_of: Dict[str, str] = {}
        self._validate_share_groups()
        # версия схемы сериализации: 1 — legacy (mode), 2 — роли (iter46/B6)
        has_new = any(nd.mode in _NEW_SHARE_MODES or nd.scale != "linear"
                      for nd in self.nodes)
        if schema_version is None:
            schema_version = 2 if has_new else 1
        if schema_version not in (1, 2):
            raise ValueError(
                f"Неизвестная версия схемы phr-спеки: {schema_version!r} "
                f"(поддерживаются 1 и 2).")
        if schema_version == 1 and has_new:
            raise ValueError(
                "Схема v1 (legacy) не поддерживает роли share_free/"
                "share_closure/share_simplex и scale — используйте "
                "схему v2 (role-ключи).")
        if schema_version == 2 and any(nd.mode == MODE_SHARE_OF
                                       for nd in self.nodes):
            raise ValueError(
                "Схема v2 не поддерживает legacy-режим 'share_of' — "
                "используйте SHARE_FREE/SHARE_CLOSURE (k=2) или "
                "SHARE_SIMPLEX (k≥3).")
        self.schema_version: int = int(schema_version)
        # iter48/B4 (C2): приоритет GROUP_TOTAL-групп кампании. Если задан —
        # это ТОЧНАЯ перестановка множества групп с ролью GROUP_TOTAL
        # (absolute-тотал share-группы); GROUP_TOTAL_FIXED исключается
        # (тотал детерминирован, стратифицировать нечего). Пропуски, лишние
        # имена, дубли и fixed-тоталы — ошибка, не тихое игнорирование.
        # Пустой список эквивалентен «не задан» (ключ опциональный).
        self.group_order: List[str] = [str(g) for g in (group_order or [])]
        if self.group_order:
            if self.schema_version != 2:
                raise ValueError(
                    "group_order поддерживается только схемой v2 (роли): "
                    "в legacy-схеме v1 роли GROUP_TOTAL нет (iter48/B4).")
            dups = sorted({g for g in self.group_order
                           if self.group_order.count(g) > 1})
            if dups:
                raise ValueError(
                    f"group_order: дубли {dups} недопустимы — требуется "
                    f"ТОЧНАЯ перестановка множества GROUP_TOTAL (C2).")
            strat = [nd.name for nd in self.nodes
                     if nd.name in self._share_groups
                     and nd.mode == MODE_ABSOLUTE]
            for g in self.group_order:
                if g not in self._by_name:
                    raise ValueError(
                        f"group_order: узел '{g}' не найден среди узлов "
                        f"спеки (GROUP_TOTAL-группы: {strat}).")
                if g not in self._share_groups:
                    raise ValueError(
                        f"group_order: узел '{g}' не является тоталом "
                        f"share-группы (GROUP_TOTAL-группы: {strat}).")
                if self._by_name[g].mode == MODE_FIXED:
                    raise ValueError(
                        f"group_order: '{g}' — GROUP_TOTAL_FIXED, его тотал "
                        f"детерминирован, стратифицировать нечего (C2) — "
                        f"исключите его из списка.")
            missing = [g for g in strat if g not in self.group_order]
            if missing:
                raise ValueError(
                    f"group_order: не перечислены группы {missing} — "
                    f"требуется ТОЧНАЯ перестановка множества GROUP_TOTAL "
                    f"(C2), частичный список недопустим.")
        self._topo: List[str] = self._toposort()
        # компоненты смеси = листья (НЕ родители share-групп), порядок спеки
        self.component_names: List[str] = [
            nd.name for nd in self.nodes if nd.name not in self._share_groups]
        if len(self.component_names) < 2:
            raise ValueError(
                "phr-спека должна давать ≥2 компонента смеси (листа DAG).")
        # z-оси: НЕ-fixed узлы в порядке спеки МИНУС производные члены
        # групп нового контракта (iter46/B2: closure и последний член
        # simplex-группы восстанавливаются как 1 − Σ партнёров — точной
        # линейной зависимости в design-пространстве больше нет)
        derived = set(self._derived_of.values())
        self.z_names: List[str] = [nd.name for nd in self.nodes
                                   if nd.mode != MODE_FIXED
                                   and nd.name not in derived]
        self._z_col: Dict[str, int] = {nm: j for j, nm
                                       in enumerate(self.z_names)}
        # оси с scale='log' (iter47/B5): их z-координата — ln phr
        self._log_axes: List[str] = [nd.name for nd in self.nodes
                                     if nd.scale == "log"]
        # статическая интервальная валидация (ДО любого сэмплинга)
        self._interval: Dict[str, Tuple[float, float]] = {}
        # окно тотала группы, суженное phr-лимитами её членов (iter45/B1):
        # родитель share-группы → эффективные границы его absolute-оси
        self._total_window: Dict[str, Tuple[float, float]] = {}
        self._validate_intervals()

    # ------------------------------------------------------------------
    # Конструкторы
    # ------------------------------------------------------------------
    @classmethod
    def from_dicts(cls, dicts) -> "PhrSpec":
        """Спека из JSON-представления. Понимает ДВЕ схемы:

        * **v1 (legacy, ``mode``)** — список узлов с ключами ``name``,
          ``mode``, ``lo``/``hi``, ``value`` (fixed), ``of`` (share_of) /
          ``to`` (ratio_to), ``cap_to``/``cap_ratio`` (потолок absolute-оси;
          ``cap_to`` — имя или СПИСОК имён, значения складываются),
          ``min_phr``/``max_phr``. Доли share_of без явных границ —
          ``[0, 1]``. Поведение и хеши прежние (старые сейвы валидны);
        * **v2 (роли, iter46/B6)** — список узлов с ключом ``role``
          (см. ``_ROLE_TABLE``) и ключами ``range``/``share_range``
          (пары ``[lo, hi]``), ``group``/``members``/``reference``/``scale``;
          допускается обёртка ``{"spec_version": 2, "nodes": [...]}``
          с опциональным ключом ``group_order`` (iter48/B4) — ТОЧНАЯ
          перестановка множества GROUP_TOTAL-групп (приоритет осей
          кампании, входит в ``spec_hash``).
          Ключи вне схемы роли (в т.ч. legacy lo/hi/of/to и range у
          closure/fixed) — ошибка валидации, не тихое игнорирование (B8).

        Смешивать ``mode`` и ``role`` в одном списке нельзя.
        """
        if isinstance(dicts, Mapping):               # v2-обёртка
            extra = set(dicts) - {"spec_version", "nodes", "group_order"}
            if extra:
                raise ValueError(
                    f"Неизвестные ключи обёртки спеки: {sorted(extra)} "
                    f"(допустимы 'spec_version', 'nodes' и 'group_order').")
            ver = dicts.get("spec_version")
            if ver not in (2, "2"):
                raise ValueError(
                    f"spec_version={ver!r} не поддерживается (ожидается 2; "
                    f"legacy-схема v1 — плоский список узлов с 'mode').")
            raw = dicts.get("nodes")
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                raise ValueError(
                    "Обёртка спеки: 'nodes' должен быть списком узлов.")
            go = dicts.get("group_order")
            if go is not None:
                if isinstance(go, str) or not isinstance(go, Sequence):
                    raise ValueError(
                        "Обёртка спеки: 'group_order' должен быть СПИСКОМ "
                        "имён GROUP_TOTAL-узлов (iter48/B4).")
                go = [str(x) for x in go]
            return cls(cls._nodes_from_roles(list(raw)), schema_version=2,
                       group_order=go)
        items = list(dicts)
        has_role = any("role" in d for d in items)
        has_mode = any("mode" in d for d in items)
        if has_role and has_mode:
            raise ValueError(
                "Смешаны схемы: часть узлов с 'mode' (v1), часть с 'role' "
                "(v2) — спека должна быть в одной схеме.")
        if has_role:
            return cls(cls._nodes_from_roles(items), schema_version=2)
        nodes: List[PhrNode] = []
        for d in items:
            mode = str(d.get("mode", ""))
            ref = str(d.get("of", d.get("to", "")) or "")
            lo = float(d.get("lo", 0.0))
            hi = float(d.get("hi", 1.0 if mode == MODE_SHARE_OF else 0.0))
            raw_cap = d.get("cap_to", "") or ""
            if isinstance(raw_cap, str):
                cap_refs: Tuple[str, ...] = (raw_cap,) if raw_cap else ()
            else:
                cap_refs = tuple(str(x) for x in raw_cap)
            raw_min = d.get("min_phr", None)
            raw_max = d.get("max_phr", None)
            nodes.append(PhrNode(name=str(d.get("name", "")), mode=mode,
                                 lo=lo, hi=hi,
                                 value=float(d.get("value", 0.0)), ref=ref,
                                 cap_refs=cap_refs,
                                 cap_ratio=float(d.get("cap_ratio", 0.0)),
                                 min_phr=(None if raw_min is None
                                          else float(raw_min)),
                                 max_phr=(None if raw_max is None
                                          else float(raw_max))))
        return cls(nodes, schema_version=1)

    @classmethod
    def _nodes_from_roles(cls, items: Sequence[Mapping[str, Any]]
                          ) -> List[PhrNode]:
        """iter46/B6: разбор узлов схемы v2 (роли) в :class:`PhrNode`.

        Строгая валидация ключей per-role (``_ROLE_TABLE``): недостающие
        обязательные и лишние ключи — ошибка (ловит и legacy lo/hi/of/to,
        и range у closure/fixed — B8). Кросс-проверка ``members`` групп:
        список обязан ТОЧНО совпадать (состав И порядок) с узлами,
        объявившими ``group`` на этот тотал (C2-строгость: несовпадение —
        ошибка, а не «лишние игнорируются»).
        """
        def _pair(d: Mapping[str, Any], key: str, name: str,
                  role: str) -> Tuple[float, float]:
            v = d[key]
            if not isinstance(v, (list, tuple)) or len(v) != 2:
                raise ValueError(
                    f"Узел '{name}' ({role}): '{key}' должен быть парой "
                    f"[lo, hi], получено {v!r}.")
            return float(v[0]), float(v[1])

        nodes: List[PhrNode] = []
        declared: Dict[str, List[str]] = {}      # тотал → members из схемы
        children: Dict[str, List[str]] = {}      # тотал → узлы с group=…
        names: List[str] = []
        for d in items:
            if not isinstance(d, Mapping):
                raise ValueError(f"Узел спеки должен быть объектом, "
                                 f"получено {type(d).__name__}.")
            name = str(d.get("name", ""))
            role = str(d.get("role", ""))
            if role not in _ROLE_TABLE:
                raise ValueError(
                    f"Узел '{name}': неизвестная роль '{role}' "
                    f"(допустимо: {sorted(_ROLE_TABLE)}).")
            mode, required, allowed = _ROLE_TABLE[role]
            if role in ("SHARE_CLOSURE", "FIXED", "GROUP_TOTAL_FIXED"):
                bad = [k for k in ("range", "share_range") if k in d]
                if bad:
                    reason = ("диапазон closure ПРОИЗВОДНЫЙ: "
                              "[1−φᵁ_free, 1−φᴸ_free]"
                              if role == "SHARE_CLOSURE"
                              else "fixed-узел задаётся ключом 'value'")
                    raise ValueError(
                        f"Узел '{name}' ({role}): ключи {bad} недопустимы — "
                        f"{reason} (B8: ошибка, не тихое игнорирование).")
            missing = sorted(required - set(d))
            if missing:
                raise ValueError(
                    f"Узел '{name}' ({role}): нет обязательных ключей "
                    f"{missing}.")
            extra = sorted(set(d) - allowed)
            if extra:
                raise ValueError(
                    f"Узел '{name}' ({role}): ключи {extra} не входят в "
                    f"схему v2 (допустимо: {sorted(allowed)}).")
            lo = hi = 0.0
            if "range" in d:
                lo, hi = _pair(d, "range", name, role)
            elif "share_range" in d:
                lo, hi = _pair(d, "share_range", name, role)
            cap_refs: Tuple[str, ...] = ()
            if role == "ABSOLUTE_CAPPED":
                raw_cap = d["cap_to"]
                if (isinstance(raw_cap, str)
                        or not isinstance(raw_cap, (list, tuple))
                        or not raw_cap):
                    raise ValueError(
                        f"Узел '{name}' (ABSOLUTE_CAPPED): 'cap_to' — "
                        f"непустой СПИСОК имён узлов (фаза), получено "
                        f"{raw_cap!r}.")
                cap_refs = tuple(str(x) for x in raw_cap)
            ref = str(d.get("group", d.get("reference", "")) or "")
            raw_min = d.get("min_phr", None)
            raw_max = d.get("max_phr", None)
            nodes.append(PhrNode(
                name=name, mode=mode, lo=lo, hi=hi,
                value=float(d.get("value", 0.0)), ref=ref,
                cap_refs=cap_refs,
                cap_ratio=float(d.get("cap_ratio", 0.0)),
                min_phr=(None if raw_min is None else float(raw_min)),
                max_phr=(None if raw_max is None else float(raw_max)),
                scale=str(d.get("scale", "linear"))))
            names.append(name)
            if "members" in d:
                mv = d["members"]
                if (isinstance(mv, str)
                        or not isinstance(mv, (list, tuple)) or not mv):
                    raise ValueError(
                        f"Узел '{name}' ({role}): 'members' — непустой "
                        f"СПИСОК имён узлов группы.")
                declared[name] = [str(x) for x in mv]
            if mode in _NEW_SHARE_MODES and ref:
                children.setdefault(ref, []).append(name)
        for parent, members in declared.items():
            actual = children.get(parent, [])
            if actual != members:
                raise ValueError(
                    f"Группа '{parent}': 'members' {members} не совпадают "
                    f"со списком узлов group='{parent}' в порядке спеки "
                    f"{actual} — состав и порядок обязаны совпадать точно.")
        for parent in children:
            if parent in set(names) and parent not in declared:
                raise ValueError(
                    f"Узел '{parent}': на него ссылаются share-узлы, но "
                    f"роль не GROUP_TOTAL/GROUP_TOTAL_FIXED (нет 'members').")
        return nodes

    # ------------------------------------------------------------------
    # Валидация структуры
    # ------------------------------------------------------------------
    def _validate_refs(self) -> None:
        for nd in self.nodes:
            needs_ref = nd.mode in _SHARE_MODES or nd.mode == MODE_RATIO_TO
            if needs_ref:
                if not nd.ref:
                    raise ValueError(
                        f"Узел '{nd.name}' ({nd.mode}): не указан референс "
                        f"('of'/'to'/'group').")
                if nd.ref not in self._by_name:
                    raise ValueError(
                        f"Узел '{nd.name}': референс '{nd.ref}' не найден "
                        f"среди узлов спеки.")
                if nd.ref == nd.name:
                    raise ValueError(
                        f"Узел '{nd.name}': ссылка на самого себя.")
            elif nd.ref:
                raise ValueError(
                    f"Узел '{nd.name}' ({nd.mode}): референс недопустим.")
            if nd.cap_refs:
                if nd.mode != MODE_ABSOLUTE:
                    raise ValueError(
                        f"Узел '{nd.name}' ({nd.mode}): cap_to допустим "
                        f"только для absolute.")
                if len(set(nd.cap_refs)) != len(nd.cap_refs):
                    raise ValueError(
                        f"Узел '{nd.name}': дубли в cap_to "
                        f"{list(nd.cap_refs)} недопустимы.")
                for cr in nd.cap_refs:
                    if cr not in self._by_name:
                        raise ValueError(
                            f"Узел '{nd.name}': cap-референс '{cr}' "
                            f"не найден среди узлов спеки.")
                    if cr == nd.name:
                        raise ValueError(
                            f"Узел '{nd.name}': cap-ссылка на самого себя.")
                if nd.cap_ratio <= 0:
                    raise ValueError(
                        f"Узел '{nd.name}': cap_ratio должен быть > 0 "
                        f"(получено {nd.cap_ratio}).")
            elif nd.cap_ratio:
                raise ValueError(
                    f"Узел '{nd.name}': cap_ratio без cap_to недопустим.")
            if nd.min_phr is not None or nd.max_phr is not None:
                if nd.mode not in _SHARE_MODES:
                    raise ValueError(
                        f"Узел '{nd.name}' ({nd.mode}): min_phr/max_phr "
                        f"допустимы только для share-режимов — у absolute/"
                        f"ratio_to границы уже заданы в своих координатах.")
                if nd.min_phr is not None and nd.min_phr < 0:
                    raise ValueError(
                        f"Узел '{nd.name}': min_phr={nd.min_phr} < 0.")
                if (nd.min_phr is not None and nd.max_phr is not None
                        and nd.min_phr > nd.max_phr + _TOL):
                    raise ValueError(
                        f"Узел '{nd.name}': min_phr={nd.min_phr} > "
                        f"max_phr={nd.max_phr}.")
                if nd.max_phr is not None and nd.max_phr <= _TOL:
                    raise ValueError(
                        f"Узел '{nd.name}': max_phr={nd.max_phr} ≤ 0.")
            if nd.mode == MODE_SHARE_CLOSURE and (nd.lo or nd.hi):
                raise ValueError(
                    f"Узел '{nd.name}' (share_closure): диапазон доли не "
                    f"задаётся — он ПРОИЗВОДНЫЙ от свободного партнёра "
                    f"[1−φᵁ_free, 1−φᴸ_free] (iter46/B8).")
            if nd.scale not in _SCALES:
                raise ValueError(
                    f"Узел '{nd.name}': неизвестная шкала scale="
                    f"'{nd.scale}' (допустимо: {list(_SCALES)}).")
            if nd.scale == "log":
                if nd.mode != MODE_ABSOLUTE:
                    raise ValueError(
                        f"Узел '{nd.name}' ({nd.mode}): scale='log' допустим "
                        f"только для absolute-осей.")
                if nd.lo <= 0:
                    raise ValueError(
                        f"Узел '{nd.name}': scale='log' требует lo > 0 "
                        f"(получено {nd.lo}).")

    def _validate_share_groups(self) -> None:
        """iter46/B2: состав share-групп нового контракта и производные
        границы closure.

        Инварианты (ревизия контракта «pvc_edge_v1», C1/C5):

          * смешивать legacy ``share_of`` и новые роли в одной группе нельзя;
          * k=2 → РОВНО один ``share_closure`` + один ``share_free``
            (``share_simplex`` при k=2 запрещён);
          * k≥3 → ВСЕ члены ``share_simplex`` (closure запрещён: замыкание —
            внутреннее свойство сэмплера, не роль узла, C1); зависимая
            координата — ПОСЛЕДНИЙ член группы (порядок узлов — часть спеки);
          * share-бокс группы (C5): ``φᵢᵁ ≤ 1 − Σ_{j≠i} φⱼᴸ`` НЕСТРОГО
            (LUB впритык: 0.60 = 1 − 0.40 — не ошибка; строгое ``<`` дало
            бы ложный отказ);
          * родитель группы нового контракта — absolute без cap
            (GROUP_TOTAL) или fixed (GROUP_TOTAL_FIXED), scale='linear'.

        Заполняет ``_share_base`` (статические границы долей; у closure —
        производные ``[1−φᵁ_free, 1−φᴸ_free]``) и ``_derived_of``
        (родитель → член группы без z-оси).
        """
        for parent, members in self._share_groups.items():
            new_m = [m for m in members
                     if self._by_name[m].mode in _NEW_SHARE_MODES]
            if not new_m:                          # legacy-группа share_of
                for m in members:
                    nd = self._by_name[m]
                    self._share_base[m] = (nd.lo, nd.hi)
                continue
            if len(new_m) != len(members):
                raise ValueError(
                    f"Группа '{parent}': смешаны legacy 'share_of' и роли "
                    f"нового контракта — недопустимо.")
            for m in members:                      # доли ⊆ [0,1] до производных
                nd = self._by_name[m]
                if nd.mode == MODE_SHARE_CLOSURE:
                    continue
                if nd.lo < -_TOL or nd.lo > nd.hi + _TOL or nd.hi > 1 + _TOL:
                    raise ValueError(
                        f"Узел '{m}': некорректный share_range "
                        f"[{nd.lo}, {nd.hi}] (нужно 0 ≤ lo ≤ hi ≤ 1).")
            k = len(members)
            closures = [m for m in members
                        if self._by_name[m].mode == MODE_SHARE_CLOSURE]
            frees = [m for m in members
                     if self._by_name[m].mode == MODE_SHARE_FREE]
            simplex = [m for m in members
                       if self._by_name[m].mode == MODE_SHARE_SIMPLEX]
            if k < 2:
                raise ValueError(
                    f"Группа '{parent}': группа нового контракта из одного "
                    f"узла не имеет смысла (нужно k ≥ 2).")
            if k == 2:
                if simplex:
                    raise ValueError(
                        f"Группа '{parent}': SHARE_SIMPLEX допустим только "
                        f"при k ≥ 3; для k=2 — SHARE_FREE + SHARE_CLOSURE.")
                if len(closures) != 1 or len(frees) != 1:
                    raise ValueError(
                        f"Группа '{parent}' (k=2): требуется РОВНО один "
                        f"SHARE_CLOSURE и один SHARE_FREE (получено "
                        f"closure={len(closures)}, free={len(frees)}).")
                fr = self._by_name[frees[0]]
                self._share_base[frees[0]] = (fr.lo, fr.hi)
                self._share_base[closures[0]] = (1.0 - fr.hi, 1.0 - fr.lo)
                self._derived_of[parent] = closures[0]
            else:
                if len(simplex) != k:
                    raise ValueError(
                        f"Группа '{parent}' (k={k} ≥ 3): все члены должны "
                        f"быть SHARE_SIMPLEX (SHARE_CLOSURE/SHARE_FREE при "
                        f"k≥3 запрещены — замыкание делает сэмплер, C1).")
                lo_sum = sum(self._by_name[m].lo for m in members)
                for m in members:
                    nd = self._by_name[m]
                    self._share_base[m] = (nd.lo, nd.hi)
                    lub = 1.0 - (lo_sum - nd.lo)
                    if nd.hi > lub + _TOL:         # C5, сравнение НЕСТРОГОЕ
                        raise ValueError(
                            f"Группа '{parent}': share-бокс несовместен — "
                            f"φᵁ('{m}')={nd.hi:g} > 1 − Σφᴸ(партнёров)="
                            f"{lub:g} (C5).")
                self._derived_of[parent] = members[-1]
            p = self._by_name[parent]
            if p.mode == MODE_ABSOLUTE and p.cap_refs:
                raise ValueError(
                    f"Группа '{parent}': тотал группы нового контракта не "
                    f"может иметь cap_to (роль GROUP_TOTAL — absolute "
                    f"без cap).")
            if p.mode not in (MODE_ABSOLUTE, MODE_FIXED):
                raise ValueError(
                    f"Группа '{parent}': тотал группы нового контракта "
                    f"должен быть GROUP_TOTAL (absolute) или "
                    f"GROUP_TOTAL_FIXED (fixed), а не '{p.mode}'.")
            if p.scale != "linear":
                raise ValueError(
                    f"Группа '{parent}': тотал группы задаётся в линейной "
                    f"шкале (scale='{p.scale}' недопустим).")

    def _toposort(self) -> List[str]:
        """Топосорт Кана по рёбрам ref→node и cap_refs→node;
        цикл — явный ValueError."""
        indeg = {nd.name: 0 for nd in self.nodes}
        out_edges: Dict[str, List[str]] = {nd.name: [] for nd in self.nodes}
        for nd in self.nodes:
            if nd.ref:
                out_edges[nd.ref].append(nd.name)
                indeg[nd.name] += 1
            for cr in nd.cap_refs:
                out_edges[cr].append(nd.name)
                indeg[nd.name] += 1
        queue = [nd.name for nd in self.nodes if indeg[nd.name] == 0]
        order: List[str] = []
        while queue:
            nm = queue.pop(0)
            order.append(nm)
            for child in out_edges[nm]:
                indeg[child] -= 1
                if indeg[child] == 0:
                    queue.append(child)
        if len(order) != len(self.nodes):
            cyc = sorted(nm for nm, dg in indeg.items() if dg > 0)
            raise ValueError(
                f"Цикл в phr-спеке (ratio_to/share_of): узлы {cyc}.")
        return order

    def _validate_intervals(self) -> None:
        """Интервальная арифметика по DAG: [lo,hi] phr каждого узла.

        Ловит невыполнимые конфиги ДО сэмплинга: отрицательные/перевёрнутые
        границы, пустое пересечение долей группы (Σlo>1 или Σhi<1), нулевой
        нижний интервал референса (encode-деление и Σ=1 требуют value>0).
        """
        for nd in self.nodes:
            if nd.mode == MODE_FIXED:
                if nd.value < 0:
                    raise ValueError(f"Узел '{nd.name}': fixed value < 0.")
                continue
            b_lo, b_hi = self._share_base.get(nd.name, (nd.lo, nd.hi))
            if b_lo < -_TOL or b_lo > b_hi + _TOL:
                raise ValueError(
                    f"Узел '{nd.name}': некорректные границы "
                    f"[{b_lo}, {b_hi}] (нужно 0 ≤ lo ≤ hi).")
            if nd.mode in _SHARE_MODES and b_hi > 1 + _TOL:
                raise ValueError(
                    f"Узел '{nd.name}': доля share-узла должна быть ≤ 1 "
                    f"(hi={b_hi}).")
        for parent, members in self._share_groups.items():
            s_lo = sum(self._share_base[m][0] for m in members)
            s_hi = sum(self._share_base[m][1] for m in members)
            if s_lo > 1 + _TOL or s_hi < 1 - _TOL:
                raise ValueError(
                    f"Группа share_of узла '{parent}': пустое пересечение — "
                    f"Σlo={s_lo:.4f}, Σhi={s_hi:.4f}, требуется Σlo ≤ 1 ≤ Σhi.")
        for nm in self._topo:
            nd = self._by_name[nm]
            if nd.mode == MODE_FIXED:
                iv = (nd.value, nd.value)
            elif nd.mode == MODE_ABSOLUTE:
                if nd.cap_refs:
                    r_lo = sum(self._interval[cr][0] for cr in nd.cap_refs)
                    r_hi = sum(self._interval[cr][1] for cr in nd.cap_refs)
                    cap_min = nd.cap_ratio * r_lo
                    if cap_min < nd.lo - _TOL:
                        raise ValueError(
                            f"Узел '{nd.name}': при минимальной фазе "
                            f"Σ{list(nd.cap_refs)}={r_lo:.6g} потолок "
                            f"cap_ratio·Σref={cap_min:.6g} < lo={nd.lo} — "
                            f"пустой диапазон (трапеция вырождается).")
                    iv = (nd.lo, min(nd.hi, nd.cap_ratio * r_hi))
                else:
                    iv = (nd.lo, nd.hi)
            else:  # ratio_to / share-режимы: произведение интервалов ≥ 0
                r_lo, r_hi = self._interval[nd.ref]
                if r_lo <= _TOL:
                    raise ValueError(
                        f"Узел '{nd.name}' ({nd.mode}) ссылается на "
                        f"'{nd.ref}' с нижней границей {r_lo:.6g} ≤ 0 — "
                        f"референс должен быть строго положительным.")
                s_lo, s_hi = self._share_base.get(nm, (nd.lo, nd.hi))
                iv = (s_lo * r_lo, s_hi * r_hi)
                if nd.mode in _SHARE_MODES and (nd.min_phr is not None
                                                or nd.max_phr is not None):
                    lo_p, hi_p = iv
                    if nd.min_phr is not None:
                        lo_p = max(lo_p, nd.min_phr)
                    if nd.max_phr is not None:
                        hi_p = min(hi_p, nd.max_phr)
                    if lo_p > hi_p + _TOL:
                        raise ValueError(
                            f"Узел '{nd.name}': phr-лимиты "
                            f"[{nd.min_phr}, {nd.max_phr}] не пересекаются "
                            f"с достижимым диапазоном доли "
                            f"[{iv[0]:.6g}, {iv[1]:.6g}] "
                            f"(φ∈[{s_lo:g}, {s_hi:g}] при тотале "
                            f"'{nd.ref}'∈[{r_lo:.6g}, {r_hi:.6g}]).")
                    iv = (lo_p, hi_p)
            self._interval[nm] = iv
            if nm in self._share_groups:
                self._interval[nm] = self._narrow_total_window(nm, iv)
        if sum(self._interval[nm][1] for nm in self.component_names) <= _TOL:
            raise ValueError("Суммарный phr компонентов не может быть > 0 — "
                             "пустая рецептура.")

    # ------------------------------------------------------------------
    # phr-лимиты share-узлов (iter45/B1): окно тотала и per-point narrowing
    # ------------------------------------------------------------------
    def _narrow_total_window(self, parent: str,
                             iv: Tuple[float, float]) -> Tuple[float, float]:
        """Сузить интервал ТОТАЛА группы до значений ``T``, при которых
        доли её членов с ``min_phr``/``max_phr`` вообще выполнимы.

        Три источника ограничений на ``T`` (все — следствие того, что лимит
        задан в phr, а координата узла — доля):

        1. пер-узловая непустота ``max(φᴸ, min/T) ≤ min(φᵁ, max/T)``:
           ``T ≥ min_i/φᵁ_i`` и ``T ≤ max_i/φᴸ_i``;
        2. ``Σ max(φᴸ_i, min_i/T) ≤ 1`` — иначе минимумы не помещаются
           в тотал (функция невозрастает по ``T`` ⇒ даёт нижнюю границу);
        3. ``Σ min(φᵁ_i, max_i/T) ≥ 1`` — иначе тотал нечем набрать
           (тоже невозрастает ⇒ даёт верхнюю границу).

        Итог — интервал ``[a, b] ⊆ iv``. Пустой — ошибка КОНФИГА (спека
        не задаёт ни одного реализуемого рецепта). Если окно у́же
        заявленного интервала, тотал обязан быть сужаемой осью
        (``absolute`` без cap) или ``fixed``, уже попадающим в окно; иначе
        сужение пришлось бы делать пер-точечно вверх по DAG — это НЕ
        реализовано и отвергается явной ошибкой, а не тихим приближением.

        Диагностика идёт от частного к общему (сначала точное указание на
        узел, потом групповое утверждение) — иначе пользователь получает
        сообщение про «окно тотала» там, где виноват один лимит. Для
        ``fixed``-тотала (``t_lo == t_hi``) пер-узловая проверка (0) и окно
        (1) ЭКВИВАЛЕНТНЫ, поэтому недостижимый лимит всегда диагностируется
        как «узел X: лимиты не пересекаются с достижимым диапазоном»
        (отдельной fixed-ветки нет — она была бы мёртвым кодом).
        """
        members = self._share_groups[parent]
        limited = [m for m in members
                   if self._by_name[m].min_phr is not None
                   or self._by_name[m].max_phr is not None]
        if not limited:
            return iv
        t_lo, t_hi = iv
        if t_lo <= _TOL:
            raise ValueError(
                f"Группа '{parent}': phr-лимиты членов требуют строго "
                f"положительного тотала, а его нижняя граница {t_lo:.6g}.")
        for m in limited:            # (0) достижимость лимита самим узлом
            nd = self._by_name[m]
            s_lo, s_hi = self._share_base[m]
            reach_lo, reach_hi = s_lo * t_lo, s_hi * t_hi
            lim_lo = nd.min_phr if nd.min_phr is not None else 0.0
            lim_hi = nd.max_phr if nd.max_phr is not None else float("inf")
            if max(reach_lo, lim_lo) > min(reach_hi, lim_hi) + _TOL:
                raise ValueError(
                    f"Узел '{m}': phr-лимиты [{lim_lo:g}, {lim_hi:g}] "
                    f"не пересекаются с достижимым диапазоном "
                    f"[{reach_lo:.6g}, {reach_hi:.6g}] "
                    f"(φ∈[{s_lo:g}, {s_hi:g}] при тотале "
                    f"'{parent}'∈[{t_lo:.6g}, {t_hi:.6g}]).")
        a0, b0 = 0.0, float("inf")   # (1) окно из пер-узловой непустоты
        for m in limited:
            nd = self._by_name[m]
            s_lo, s_hi = self._share_base[m]
            if nd.min_phr is not None and s_hi > _TOL:
                a0 = max(a0, nd.min_phr / s_hi)
            if nd.max_phr is not None and s_lo > _TOL:
                b0 = min(b0, nd.max_phr / s_lo)
        if a0 > b0 + _TOL:
            raise ValueError(
                f"Группа '{parent}': phr-лимиты членов {limited} несовместимы "
                f"между собой — тотал должен быть одновременно ≥ {a0:.6g} "
                f"и ≤ {b0:.6g}.")
        a, b = max(float(t_lo), a0), min(float(t_hi), b0)
        if a > b + _TOL:
            raise ValueError(
                f"Группа '{parent}': окно тотала по phr-лимитам "
                f"[{a0:.6g}, {b0:.6g}] не пересекается с интервалом тотала "
                f"[{t_lo:.6g}, {t_hi:.6g}].")

        def f_min(T: float) -> float:           # Σ нижних долей при тотале T
            return sum(self._share_lo_at(m, T) for m in members)

        def f_max(T: float) -> float:           # Σ верхних долей при тотале T
            return sum(self._share_hi_at(m, T) for m in members)

        if f_min(a) > 1.0 + _TOL:               # (2) минимумы не помещаются
            if f_min(b) > 1.0 + _TOL:
                raise ValueError(
                    f"Группа '{parent}': при любом тотале из "
                    f"[{t_lo:.6g}, {t_hi:.6g}] сумма минимальных долей "
                    f"(с учётом min_phr) > 1 — рецепт нереализуем.")
            a = _bisect_decreasing(f_min, a, b, 1.0)
        if f_max(b) < 1.0 - _TOL:               # (3) тотал нечем набрать
            if f_max(a) < 1.0 - _TOL:
                raise ValueError(
                    f"Группа '{parent}': при любом тотале из "
                    f"[{t_lo:.6g}, {t_hi:.6g}] сумма максимальных долей "
                    f"(с учётом max_phr) < 1 — рецепт нереализуем.")
            b = _bisect_decreasing(f_max, a, b, 1.0)
        if a > b + _TOL:
            raise ValueError(
                f"Группа '{parent}': окно тотала пусто после учёта phr-"
                f"лимитов ([{a:.6g}, {b:.6g}]).")
        if a <= t_lo + _TOL and b >= t_hi - _TOL:
            return iv                            # ничего не сузилось
        nd_p = self._by_name[parent]
        if nd_p.mode != MODE_ABSOLUTE or nd_p.cap_refs:
            raise ValueError(
                f"Группа '{parent}': phr-лимиты членов сужают тотал до "
                f"[{a:.6g}, {b:.6g}], но тотал задан режимом "
                f"'{nd_p.mode}'{' с cap' if nd_p.cap_refs else ''} — "
                f"сужение такой оси не поддерживается; задайте тотал "
                f"absolute-узлом или ослабьте лимиты.")
        self._total_window[parent] = (a, b)
        return (a, b)

    def _share_lo_at(self, name: str, total: float) -> float:
        nd = self._by_name[name]
        lo = self._share_base[name][0]
        if nd.min_phr is not None:
            lo = max(lo, nd.min_phr / total)
        return min(lo, self._share_hi_at(name, total))

    def _share_hi_at(self, name: str, total: float) -> float:
        nd = self._by_name[name]
        hi = self._share_base[name][1]
        if nd.max_phr is not None:
            hi = min(hi, nd.max_phr / total)
        return hi

    def _axis_bounds(self, name: str) -> Tuple[float, float]:
        """Эффективные границы z-оси узла: у absolute-родителя группы с
        phr-лимитами это ОКНО ТОТАЛА, иначе заявленные ``[lo, hi]``."""
        nd = self._by_name[name]
        return self._total_window.get(name, (nd.lo, nd.hi))

    def share_base_bounds(self, name: str) -> Tuple[float, float]:
        """Статические границы ДОЛИ share-узла: у ``share_closure`` —
        ПРОИЗВОДНЫЙ диапазон ``[1−φᵁ_free, 1−φᴸ_free]`` (iter46/B2),
        у остальных — заявленный ``share_range`` / ``[lo, hi]``."""
        if name not in self._share_base:
            raise ValueError(
                f"share_base_bounds: '{name}' не является share-узлом "
                f"(share-узлы: {sorted(self._share_base)}).")
        return self._share_base[name]

    def _share_box_at_total(self, members: Sequence[str],
                            total: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """Пер-точечный бокс долей группы при тоталах ``total`` (n,):
        ``lo₀ᵢ = max(φᴸ, min_phr/T)``, ``hi₀ᵢ = min(φᵁ, max_phr/T)``.
        Возвращает две матрицы (n × k). Связь Σ=1 здесь НЕ применяется —
        её делает :func:`_narrowing_split` (сэмплинг) / renorm (clip_z);
        учёт партнёров по группе — :meth:`share_bounds_at_total`."""
        T = np.asarray(total, dtype=float).ravel()
        k = len(members)
        LO = np.empty((T.size, k), dtype=float)
        HI = np.empty((T.size, k), dtype=float)
        safe_T = np.where(T > _TOL, T, _TOL)
        for i, m in enumerate(members):
            nd = self._by_name[m]
            b_lo, b_hi = self._share_base[m]
            lo_i = np.full(T.size, float(b_lo))
            hi_i = np.full(T.size, float(b_hi))
            if nd.min_phr is not None:
                lo_i = np.maximum(lo_i, nd.min_phr / safe_T)
            if nd.max_phr is not None:
                hi_i = np.minimum(hi_i, nd.max_phr / safe_T)
            lo_i = np.minimum(lo_i, hi_i)       # числовая страховка
            LO[:, i] = lo_i
            HI[:, i] = hi_i
        return LO, HI

    def share_bounds_at_total(self, parent: str, total: float
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """ЭФФЕКТИВНЫЕ границы долей членов группы ``parent`` при тотале
        ``total`` — с учётом phr-лимитов узла И партнёров по группе (Σφ=1):

            ``loᵢ = max(lo₀ᵢ, 1 − Σ_{j≠i} hi₀ⱼ)``,
            ``hiᵢ = min(hi₀ᵢ, 1 − Σ_{j≠i} lo₀ⱼ)``.

        ⚠️ Функция ``hi(T)`` НЕМОНОТОННА: рост тотала ослабляет собственный
        потолок ``max_phr/T``, но одновременно ужесточает вклад партнёров
        через их ``min_phr/T``. Пример (PVC, iter45): φᵁ=0.70, ``max_phr``
        узла 8.0, у партнёра ``min_phr``=3.0 →
        ``hi(T) = min(0.70, 8/T, 1 − 3/T)``: 0.40 при T=5, полка 0.70 на
        T∈[10, 11.4286], 0.5333 при T=15. Тест «hi растёт с T» дал бы
        ложный отказ.
        """
        if parent not in self._share_groups:
            raise ValueError(
                f"share_bounds_at_total: '{parent}' не является родителем "
                f"share-группы (группы: {sorted(self._share_groups)}).")
        T = float(total)
        if T <= _TOL:
            raise ValueError(
                f"share_bounds_at_total: тотал должен быть > 0 (получено {T}).")
        members = self._share_groups[parent]
        LO, HI = self._share_box_at_total(members, np.array([T]))
        lo0, hi0 = LO[0], HI[0]
        lo = np.maximum(lo0, 1.0 - (hi0.sum() - hi0))
        hi = np.minimum(hi0, 1.0 - (lo0.sum() - lo0))
        return lo, hi

    # ------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------
    @property
    def q(self) -> int:
        """Число компонентов смеси (листьев DAG)."""
        return len(self.component_names)

    @property
    def dim_z(self) -> int:
        """Число design-координат z."""
        return len(self.z_names)

    def phr_intervals(self) -> Dict[str, Tuple[float, float]]:
        """Интервалы phr каждого узла (результат статической валидации)."""
        return dict(self._interval)

    # ------------------------------------------------------------------
    # Сэмплинг z (без rejection)
    # ------------------------------------------------------------------
    def sample_z(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """``n`` точек z-куба: absolute/ratio_to — равномерно по оси;
        absolute с cap — равномерно в ПЕР-ТОЧЕЧНОМ интервале
        ``[lo, min(hi, cap_ratio · Σ value(cap_refs))]`` (условная
        равномерность на трапеции, референс — фаза);
        доли share-группы — :func:`_narrowing_split`
        (Σ=1 без rejection). Обход — топологический (референсы cap вычислены
        раньше зависимых осей).

        Мера: uniform-MARGINAL по физически значимым осям (тот же осознанный
        выбор, что iter31 ``SimplexRegion.random_points(groups=…)``).

        iter46/B2: доли группы разыгрываются ЦЕЛИКОМ (все k членов), но в z
        пишутся только члены с собственной координатой — производный член
        (closure / последний simplex) в z не входит.

        iter47/B5: оси ``scale='log'`` — z-координата ``ln phr``, равномерно
        в ``[ln lo, ln hi]``; у cap-осей потолок вычисляется В PHR
        (``min(hi, cap_ratio·Σ value(cap_refs))``), логарифмируется уже
        суженная граница — условная ЛОГ-равномерность на трапеции.
        """
        rng = np.random.default_rng(seed)
        n = int(n)
        Z = np.empty((n, self.dim_z), dtype=float)
        vals: Dict[str, np.ndarray] = {}
        done_groups: set = set()
        for nm in self._topo:
            nd = self._by_name[nm]
            if nd.mode == MODE_FIXED:
                vals[nm] = np.full(n, nd.value)
            elif nd.mode == MODE_ABSOLUTE:
                a_lo, a_hi = self._axis_bounds(nm)
                if nd.scale == "log":                    # iter47/B5
                    if nd.cap_refs:
                        base = np.sum([vals[cr] for cr in nd.cap_refs],
                                      axis=0)
                        hi_eff = np.minimum(a_hi, nd.cap_ratio * base)
                        hi_eff = np.maximum(hi_eff, a_lo)  # числ. страховка
                        ln_hi = np.log(hi_eff)           # потолок В PHR
                    else:
                        ln_hi = math.log(a_hi)
                    ln_lo = math.log(a_lo)
                    z = ln_lo + (ln_hi - ln_lo) * rng.random(n)
                    Z[:, self._z_col[nm]] = z
                    vals[nm] = np.exp(z)
                else:
                    if nd.cap_refs:
                        base = np.sum([vals[cr] for cr in nd.cap_refs],
                                      axis=0)
                        hi_eff = np.minimum(a_hi, nd.cap_ratio * base)
                        hi_eff = np.maximum(hi_eff, a_lo)  # числ. страховка
                        z = a_lo + (hi_eff - a_lo) * rng.random(n)
                    else:
                        z = rng.uniform(a_lo, a_hi, size=n)
                    Z[:, self._z_col[nm]] = z
                    vals[nm] = z
            elif nd.mode == MODE_RATIO_TO:
                z = rng.uniform(nd.lo, nd.hi, size=n)
                Z[:, self._z_col[nm]] = z
                vals[nm] = z * vals[nd.ref]
            else:                              # share-режимы: группа целиком
                if nd.ref in done_groups:
                    continue
                members = self._share_groups[nd.ref]
                LO, HI = self._share_box_at_total(members, vals[nd.ref])
                S = np.empty((n, len(members)), dtype=float)
                for t in range(n):
                    S[t] = _narrowing_split(LO[t], HI[t], 1.0, rng)
                for i, m in enumerate(members):
                    if m in self._z_col:       # производный член — без z-оси
                        Z[:, self._z_col[m]] = S[:, i]
                    vals[m] = S[:, i] * vals[nd.ref]
                done_groups.add(nd.ref)
        return Z

    # ------------------------------------------------------------------
    # Границы и проекция z (iter38, B1: refine оптимизатора в z)
    # ------------------------------------------------------------------
    def z_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """СТАТИЧЕСКИЕ границы z-осей ``(lo, hi)`` в порядке ``z_names``.

        У cap-узлов ``hi`` — статический потолок узла; динамический
        (``cap_ratio · Σ value(cap_refs)``) учитывается :meth:`clip_z`
        пер-точечно. У absolute-тотала группы с phr-лимитами возвращается
        ОКНО ТОТАЛА (iter45/B1) — вне окна доли членов нереализуемы.
        У log-осей (iter47/B5) границы ЛОГАРИФМИРУЮТСЯ (z — ``ln phr``):
        ширина ``ln hi − ln lo`` даёт МУЛЬТИПЛИКАТИВНЫЙ масштаб возмущений.
        Ширины ``hi − lo`` — естественный масштаб возмущений
        по осям z (у осей разные единицы: phr / ln phr / доли / коэфф.).
        """
        bounds = []
        for nm in self.z_names:
            nd = self._by_name[nm]
            if nd.mode == MODE_ABSOLUTE:
                a_lo, a_hi = self._axis_bounds(nm)
                if nd.scale == "log":                    # iter47/B5: ln phr
                    bounds.append((math.log(a_lo), math.log(a_hi)))
                else:
                    bounds.append((a_lo, a_hi))
            elif nd.mode in _SHARE_MODES:
                bounds.append(self._share_base[nm])
            else:
                bounds.append((nd.lo, nd.hi))
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        return lo, hi

    def clip_z(self, z: Sequence[float] | np.ndarray) -> np.ndarray:
        """Проекция произвольного z в допустимую область спеки —
        допустимость ПО ПОСТРОЕНИЮ, без rejection (iter38, B1).

        Бокс в z НЕ статический, поэтому обход топологический:

          * absolute — clip в ``[lo, hi]``; с cap — в УСЛОВНЫЙ интервал
            ``[lo, max(lo, min(hi, cap_ratio · Σ value(cap_refs)))]``
            (референсы к этому моменту уже спроецированы);
          * ratio_to — clip коэффициента в ``[lo, hi]``;
          * share-группа — clip долей в ``[lo, hi]`` + детерминированное
            перераспределение невязки Σ=1 пропорционально запасу до границы
            (дефицит — по headroom ``hi − s``, избыток — по slack
            ``s − lo``); выполнимость гарантирована статической валидацией
            (Σlo ≤ 1 ≤ Σhi).

        Идемпотентна на валидных z (``clip_z(sample_z(...)) == sample_z``),
        и как проекция вообще: ``clip_z(clip_z(z)) == clip_z(z)`` для любого
        z (после прохода все узлы лежат в своих условных интервалах, а
        renorm share-групп в топопорядке не может заново нарушить уже
        спроецированный upstream — референсы вычислены раньше зависимых
        осей). Точка ВНУТРИ области не двигается (``clip_z(z) == z``) —
        защита от тихого смещения на границу. Оба свойства — тесты iter39.

        ⚠️ АСИММЕТРИЯ ПО ПОСТРОЕНИЮ (осознанный выбор, зафиксирован iter39):
        при проходе в топопорядке UPSTREAM-узлы ПОБЕЖДАЮТ, downstream
        подрезается под уже спроецированные референсы. Конфликт «УФ выше
        потолка ``cap_ratio·Σфаза``» разрешается ОПУСКАНИЕМ УФ, а не
        поднятием ДИНФ: референс (доминирующая ось рецептуры) сохраняется.
        Это разумный дефолт, но не единственный вариант — альтернатива
        «двигать референс навстречу» сознательно НЕ реализована.

        Используется refine-циклом оптимизатора: возмущение в z → clip_z →
        decode — каждая проба допустима, в отличие от rejection, который
        у границы (где и лежит оптимум) обваливается (урок iter34).

        iter47/B5: log-оси проецируются В ЛОГ-ШКАЛЕ — z клипится в
        ``[ln lo, ln hi_eff]``, где потолок ``hi_eff`` вычислен В PHR
        (cap применяется после экспоненцирования).
        """
        z = np.asarray(z, dtype=float)
        single = z.ndim == 1
        Z = np.atleast_2d(z).astype(float).copy()
        if Z.shape[1] != self.dim_z:
            raise ValueError(
                f"clip_z: ожидалось {self.dim_z} z-координат "
                f"({self.z_names}), получено {Z.shape[1]}.")
        vals: Dict[str, np.ndarray] = {}
        done_groups: set = set()
        for nm in self._topo:
            nd = self._by_name[nm]
            if nd.mode == MODE_FIXED:
                vals[nm] = np.full(len(Z), nd.value)
            elif nd.mode == MODE_ABSOLUTE:
                j = self._z_col[nm]
                a_lo, a_hi = self._axis_bounds(nm)
                if nd.cap_refs:
                    base = np.sum([vals[cr] for cr in nd.cap_refs], axis=0)
                    hi_eff = np.maximum(
                        np.minimum(a_hi, nd.cap_ratio * base), a_lo)
                else:
                    hi_eff = a_hi
                if nd.scale == "log":                    # iter47/B5: ln phr
                    Z[:, j] = np.clip(Z[:, j],
                                      math.log(a_lo), np.log(hi_eff))
                    vals[nm] = np.exp(Z[:, j])
                else:
                    Z[:, j] = np.clip(Z[:, j], a_lo, hi_eff)
                    vals[nm] = Z[:, j]
            elif nd.mode == MODE_RATIO_TO:
                j = self._z_col[nm]
                Z[:, j] = np.clip(Z[:, j], nd.lo, nd.hi)
                vals[nm] = Z[:, j] * vals[nd.ref]
            else:                              # share-режимы: группа целиком
                if nd.ref in done_groups:
                    continue
                members = self._share_groups[nd.ref]
                LO, HI = self._share_box_at_total(members, vals[nd.ref])
                S = np.empty((len(Z), len(members)), dtype=float)
                der_i = -1
                for i, m in enumerate(members):
                    if m in self._z_col:
                        S[:, i] = Z[:, self._z_col[m]]
                    else:                      # производный член: остаток
                        der_i = i
                if der_i >= 0:
                    others = [i for i in range(len(members)) if i != der_i]
                    S[:, der_i] = 1.0 - S[:, others].sum(axis=1)
                S = np.clip(S, LO, HI)
                resid = 1.0 - S.sum(axis=1)
                idx = np.where(resid > _TOL)[0]
                if idx.size:                   # дефицит → добрать по headroom
                    head = HI[idx] - S[idx]
                    S[idx] += head * (resid[idx] / head.sum(axis=1))[:, None]
                idx = np.where(resid < -_TOL)[0]
                if idx.size:                   # избыток → снять по slack
                    slack = S[idx] - LO[idx]
                    S[idx] += slack * (resid[idx] / slack.sum(axis=1))[:, None]
                for i, m in enumerate(members):
                    if m in self._z_col:
                        Z[:, self._z_col[m]] = S[:, i]
                    vals[m] = S[:, i] * vals[nd.ref]
                done_groups.add(nd.ref)
        return Z[0] if single else Z

    # ------------------------------------------------------------------
    # decode / encode
    # ------------------------------------------------------------------
    def decode(self, z: Sequence[float] | np.ndarray) -> np.ndarray:
        """z → p: phr компонентов (столбцы = ``component_names``).
        Log-оси (iter47/B5) экспоненцируются: ``p = exp(z)``."""
        z = np.asarray(z, dtype=float)
        single = z.ndim == 1
        Z = np.atleast_2d(z)
        if Z.shape[1] != self.dim_z:
            raise ValueError(
                f"decode: ожидалось {self.dim_z} z-координат "
                f"({self.z_names}), получено {Z.shape[1]}.")
        vals: Dict[str, np.ndarray] = {}
        for nm in self._topo:
            nd = self._by_name[nm]
            if nd.mode == MODE_FIXED:
                vals[nm] = np.full(len(Z), nd.value)
            elif nd.mode == MODE_ABSOLUTE:
                col = Z[:, self._z_col[nm]]
                vals[nm] = np.exp(col) if nd.scale == "log" else col.copy()
            else:  # ratio_to / share-режимы
                if nm in self._z_col:
                    vals[nm] = Z[:, self._z_col[nm]] * vals[nd.ref]
                else:      # производный член группы: 1 − Σ долей партнёров
                    others = [m for m in self._share_groups[nd.ref]
                              if m != nm]
                    share = 1.0 - np.sum(
                        [Z[:, self._z_col[m]] for m in others], axis=0)
                    vals[nm] = share * vals[nd.ref]
        P = np.column_stack([vals[nm] for nm in self.component_names])
        return P[0] if single else P

    def encode(self, p: Sequence[float] | np.ndarray,
               tol: float = 1e-6) -> np.ndarray:
        """p → z (обратное к :meth:`decode`): для anchors/исторических
        рецептов, заданных в phr. Внутренние узлы восстанавливаются суммой
        детей; несоответствие fixed-значению или выход за границы осей —
        явный ValueError (anchor вне области — ошибка данных). Проверки
        границ/cap у absolute-осей идут В PHR; z-координата log-оси
        (iter47/B5) — ``ln phr``."""
        p = np.asarray(p, dtype=float)
        single = p.ndim == 1
        P = np.atleast_2d(p)
        if P.shape[1] != self.q:
            raise ValueError(
                f"encode: ожидалось {self.q} компонентов "
                f"({self.component_names}), получено {P.shape[1]}.")
        leaf_col = {nm: j for j, nm in enumerate(self.component_names)}
        vals: Dict[str, np.ndarray] = {}
        for nm in reversed(self._topo):        # дети раньше родителей
            if nm in self._share_groups:
                vals[nm] = np.sum(
                    [vals[m] for m in self._share_groups[nm]], axis=0)
            else:
                vals[nm] = P[:, leaf_col[nm]]
        Z = np.empty((len(P), self.dim_z), dtype=float)
        for nm in self._topo:
            nd = self._by_name[nm]
            if nd.mode == MODE_FIXED:
                if not np.allclose(vals[nm], nd.value, atol=tol,
                                   rtol=1e-9):
                    raise ValueError(
                        f"encode: узел '{nm}' fixed={nd.value}, а в рецепте "
                        f"{vals[nm]} — рецепт вне спеки.")
                continue
            if nd.mode == MODE_ABSOLUTE:
                zj = vals[nm]
            else:                              # ratio_to / share-режимы
                denom = vals[nd.ref]
                if np.any(denom <= _TOL):
                    raise ValueError(
                        f"encode: референс '{nd.ref}' узла '{nm}' равен 0 — "
                        f"коэффициент не определён.")
                zj = vals[nm] / denom
            if nd.mode == MODE_ABSOLUTE:
                b_lo, b_hi = self._axis_bounds(nm)
            else:
                b_lo, b_hi = self._share_base.get(nm, (nd.lo, nd.hi))
            if np.any(zj < b_lo - tol) or np.any(zj > b_hi + tol):
                raise ValueError(
                    f"encode: узел '{nm}' ({nd.mode}) вне границ "
                    f"[{b_lo}, {b_hi}]: значения {zj}.")
            if nd.mode in _SHARE_MODES and (nd.min_phr is not None
                                            or nd.max_phr is not None):
                v = vals[nm]                   # phr узла (доля × тотал)
                if nd.min_phr is not None and np.any(v < nd.min_phr - tol):
                    raise ValueError(
                        f"encode: узел '{nm}' ниже технологического минимума "
                        f"min_phr={nd.min_phr:g}: значения {v} phr.")
                if nd.max_phr is not None and np.any(v > nd.max_phr + tol):
                    raise ValueError(
                        f"encode: узел '{nm}' выше лимита "
                        f"max_phr={nd.max_phr:g}: значения {v} phr.")
            if nd.mode == MODE_ABSOLUTE and nd.cap_refs:
                cap = nd.cap_ratio * np.sum(
                    [vals[cr] for cr in nd.cap_refs], axis=0)
                if np.any(zj > cap + tol):
                    raise ValueError(
                        f"encode: узел '{nm}' превышает потолок "
                        f"{nd.cap_ratio:g}·Σ{list(nd.cap_refs)} (= {cap}): "
                        f"значения {zj} — рецепт вне спеки.")
            if nd.mode == MODE_ABSOLUTE and nd.scale == "log":
                # iter47/B5: z log-оси — ln phr; все проверки выше выполнены
                # в phr (lo > 0 из валидации гарантирует ln определён)
                zj = np.log(np.maximum(zj, np.finfo(float).tiny))
            if nm in self._z_col:              # производные члены — без z-оси
                Z[:, self._z_col[nm]] = zj
        return Z[0] if single else Z

    # ------------------------------------------------------------------
    # Доли и кандидаты
    # ------------------------------------------------------------------
    def to_fractions(self, p: Sequence[float] | np.ndarray) -> np.ndarray:
        """p → x: ``x_i = p_i / Σp`` (Σx=1 конструкцией)."""
        p = np.asarray(p, dtype=float)
        single = p.ndim == 1
        P = np.atleast_2d(p)
        s = P.sum(axis=1)
        if np.any(s <= _TOL):
            raise ValueError("to_fractions: суммарный phr рецепта ≤ 0.")
        X = P / s[:, None]
        return X[0] if single else X

    def fractions_to_phr(self, x: Sequence[float] | np.ndarray,
                         tol: float = 1e-6) -> np.ndarray:
        """x → p: ОБРАТНОЕ к :meth:`to_fractions` (iter42/42.1, слой навески).

        Доли масштаба не несут (``Σx = 1`` для любой загрузки), поэтому
        суммарный phr восстанавливается по ЯКОРЮ — ``fixed``-листу спеки
        (смола = 100, ESO = 2.5 phr и т.п.)::

            Σp = value_fixed / x_fixed,      p = x · Σp

        Якорь берётся ПЕРВЫМ (в порядке спеки) fixed-листом с ``value > 0``;
        остальные fixed-листья служат ПРОВЕРКОЙ согласованности: если
        восстановленный phr расходится с их константой больше ``tol``
        (относительно величины константы), входные доли не принадлежат этой
        спеке — ошибка ДАННЫХ, явный ``ValueError`` (как в :meth:`encode`),
        а не тихое приближение.

        Без fixed-листа масштаб неопределим В ПРИНЦИПЕ (все узлы задают лишь
        пропорции) — тоже ``ValueError`` с объяснением, а не «принято 100».

        Принимает один рецепт (q,) или матрицу (n × q) — для построчной
        навески seed-плана. Вход нормируется на ``Σx`` (защита от накопленной
        погрешности долей после clip/quantize), поэтому значение якорного
        компонента в результате равно его константе точно.
        """
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        X = np.atleast_2d(x)
        if X.shape[1] != self.q:
            raise ValueError(
                f"fractions_to_phr: ожидалось {self.q} компонентов "
                f"({self.component_names}), получено {X.shape[1]}.")
        anchors = [(j, nm, float(self._by_name[nm].value))
                   for j, nm in enumerate(self.component_names)
                   if self._by_name[nm].mode == MODE_FIXED
                   and self._by_name[nm].value > _TOL]
        if not anchors:
            raise ValueError(
                "fractions_to_phr: в спеке нет fixed-листа с положительным "
                "value — суммарный phr по долям НЕОПРЕДЕЛИМ (доли задают "
                "только пропорции). Добавьте якорь (например, смола = 100 "
                "phr) или ведите навеску сразу в phr.")
        s = X.sum(axis=1)
        if np.any(s <= _TOL):
            raise ValueError("fractions_to_phr: сумма долей рецепта ≤ 0.")
        Xn = X / s[:, None]
        j0, nm0, v0 = anchors[0]
        xf = Xn[:, j0]
        if np.any(xf <= _TOL):
            raise ValueError(
                f"fractions_to_phr: доля якорного компонента '{nm0}' равна 0 "
                f"— масштаб (Σp = {v0:g}/x) не определён.")
        total = v0 / xf
        P = Xn * total[:, None]
        for j, nm, v in anchors[1:]:
            scale = max(1.0, abs(v))
            bad = np.abs(P[:, j] - v) > float(tol) * scale
            if np.any(bad):
                i = int(np.argmax(bad))
                raise ValueError(
                    f"fractions_to_phr: доли не согласованы со спекой — по "
                    f"якорю '{nm0}' (={v0:g} phr) восстановленный phr узла "
                    f"'{nm}' равен {P[i, j]:.6g}, а спека фиксирует {v:g} "
                    f"(строка {i}). Проверьте, что доли получены из ЭТОЙ "
                    f"спеки.")
        return P[0] if single else P

    def sample_candidates(self, n: int,
                          seed: Optional[int] = None) -> np.ndarray:
        """``n`` кандидатов-долей (n × q, Σ=1): sample_z → decode →
        to_fractions. Готовый вход для пула кандидатов раннера (этап A —
        сэмплер-плагин, схема/модель не затрагиваются)."""
        return self.to_fractions(self.decode(self.sample_z(n, seed=seed)))

    # ------------------------------------------------------------------
    # Роли узлов (iter50/P1.3): ЕДИНЫЙ источник ролей для сериализации и UI
    # ------------------------------------------------------------------
    def role_of(self, name: str) -> str:
        """Роль узла контракта v2 (см. таблицу ролей UI_REVISION_SPEC).

        Роль — не отдельное поле узла, а СЛЕДСТВИЕ структуры: fixed/absolute
        родитель share-группы → ``GROUP_TOTAL_FIXED``/``GROUP_TOTAL``,
        absolute с динамическим потолком → ``ABSOLUTE_CAPPED`` и т.д.
        Метод — единственный источник этого вывода: им пользуются и
        сериализация :meth:`_to_role_dicts`, и показ в UI (иначе роль в
        таблице интерфейса и роль в отпечатке спеки могли бы разойтись).

        Спека схемы v1 (legacy) ролей не имеет: её ``share_of``-узлы
        отдаются как ``SHARE_OF`` — честная пометка «legacy-режим», а не
        подмена одной из новых ролей (выбор «кто closure» изменил бы меру
        сэмплера молча, A0.6; см. iter46, п.1)."""
        nd = self._by_name.get(str(name))
        if nd is None:
            raise ValueError(
                f"role_of: узел '{name}' не найден в спеке "
                f"(узлы: {[n.name for n in self.nodes]}).")
        is_group = nd.name in self._share_groups
        if nd.mode == MODE_FIXED:
            return "GROUP_TOTAL_FIXED" if is_group else "FIXED"
        if nd.mode == MODE_ABSOLUTE:
            if is_group:
                return "GROUP_TOTAL"
            return "ABSOLUTE_CAPPED" if nd.cap_refs else "ABSOLUTE"
        if nd.mode == MODE_RATIO_TO:
            return "RATIO_TO"
        return {MODE_SHARE_FREE: "SHARE_FREE",
                MODE_SHARE_CLOSURE: "SHARE_CLOSURE",
                MODE_SHARE_SIMPLEX: "SHARE_SIMPLEX",
                MODE_SHARE_OF: "SHARE_OF"}[nd.mode]

    # ------------------------------------------------------------------
    # Версионирование спеки (iter35): порядок узлов — часть спеки
    # ------------------------------------------------------------------
    def to_dicts(self) -> List[Dict[str, Any]] | Dict[str, Any]:

        """Каноническая сериализация спеки (round-trip c :meth:`from_dicts`).

        ПОРЯДОК УЗЛОВ СОХРАНЯЕТСЯ: он определяет и порядок компонентов, и
        порядок z-осей, и — через :func:`core.simplex._narrowing_split` /
        последовательное сужение — какие оси получают точную равномерную
        маргиналь (iter34, находка 1). Перестановка узлов — ДРУГАЯ спека.

        Формат определяется ``schema_version``: v1 — legacy-ключи
        ``mode``/``lo``/``hi``/``of``/``to`` (байт-в-байт как до iter46,
        хеши старых спек не меняются); v2 — role-ключи (iter46/B6,
        :meth:`_to_role_dicts`).

        iter48/B4: при заданном ``group_order`` v2-спека сериализуется
        ОБЁРТКОЙ ``{"spec_version": 2, "group_order": [...], "nodes":
        [...]}`` — порядок групп входит в ``spec_hash``. Спеки БЕЗ
        ``group_order`` сериализуются плоским списком байт-в-байт как до
        iter48 (хеши прежних спек не меняются).
        """
        if self.schema_version == 2:
            nodes = self._to_role_dicts()
            if self.group_order:
                return {"spec_version": 2,
                        "group_order": list(self.group_order),
                        "nodes": nodes}
            return nodes
        out: List[Dict[str, Any]] = []
        for nd in self.nodes:
            d: Dict[str, Any] = {"name": nd.name, "mode": nd.mode}
            if nd.mode == MODE_FIXED:
                d["value"] = float(nd.value)
            else:
                d["lo"] = float(nd.lo)
                d["hi"] = float(nd.hi)
            if nd.mode == MODE_SHARE_OF:
                d["of"] = nd.ref
                # phr-лимиты — часть геометрии (сужают область), поэтому
                # входят в отпечаток; отсутствие ключей = спека без лимитов,
                # хеш таких спек не меняется (совместимость с iter35)
                if nd.min_phr is not None:
                    d["min_phr"] = float(nd.min_phr)
                if nd.max_phr is not None:
                    d["max_phr"] = float(nd.max_phr)
            elif nd.mode == MODE_RATIO_TO:
                d["to"] = nd.ref
            if nd.cap_refs:                    # динамический потолок — часть
                # геометрии, входит в отпечаток; каноническая форма:
                # один референс — строка (обратная совместимость хеша),
                # фаза — список имён
                d["cap_to"] = (nd.cap_refs[0] if len(nd.cap_refs) == 1
                               else list(nd.cap_refs))
                d["cap_ratio"] = float(nd.cap_ratio)
            out.append(d)
        return out

    def _to_role_dicts(self) -> List[Dict[str, Any]]:
        """iter46/B6: сериализация в схему v2 (роли) — обратное к
        :meth:`_nodes_from_roles`; round-trip сохраняет ``spec_hash``.
        Роли восстанавливаются из структуры ЕДИНЫМ источником
        (:meth:`role_of`): fixed/absolute-родитель share-группы →
        GROUP_TOTAL_FIXED/GROUP_TOTAL, absolute с cap → ABSOLUTE_CAPPED;
        ``members`` — из фактических детей группы."""
        out: List[Dict[str, Any]] = []
        for nd in self.nodes:
            is_group = nd.name in self._share_groups
            role = self.role_of(nd.name)
            d: Dict[str, Any] = {"name": nd.name, "role": role}
            if nd.mode == MODE_FIXED:
                d["value"] = float(nd.value)
                if is_group:
                    d["members"] = list(self._share_groups[nd.name])
            elif nd.mode == MODE_ABSOLUTE:
                if is_group:
                    d["range"] = [float(nd.lo), float(nd.hi)]
                    d["members"] = list(self._share_groups[nd.name])
                else:
                    d["range"] = [float(nd.lo), float(nd.hi)]
                    if nd.scale != "linear":
                        d["scale"] = nd.scale
                    if nd.cap_refs:
                        d["cap_to"] = list(nd.cap_refs)
                        d["cap_ratio"] = float(nd.cap_ratio)
            elif nd.mode == MODE_RATIO_TO:
                d["reference"] = nd.ref
                d["range"] = [float(nd.lo), float(nd.hi)]
            else:                              # share-роли нового контракта
                d["group"] = nd.ref

                if nd.mode != MODE_SHARE_CLOSURE:
                    d["share_range"] = [float(nd.lo), float(nd.hi)]
                if nd.min_phr is not None:
                    d["min_phr"] = float(nd.min_phr)
                if nd.max_phr is not None:
                    d["max_phr"] = float(nd.max_phr)
            out.append(d)
        return out

    def spec_hash(self) -> str:
        """SHA-256 отпечаток спеки (hex, 64 символа) для воспроизводимости.

        Хеш чувствителен к ПОРЯДКУ узлов (см. :meth:`to_dicts`): порядок
        влияет на меру сэмплера, поэтому обязан входить в отпечаток.
        iter48/B4: ``group_order`` (приоритет GROUP_TOTAL-групп кампании)
        тоже входит в отпечаток — его перестановка меняет хеш.
        Записывайте хеш в документацию кампании — через полгода по нему
        можно проверить, что геометрия не «уехала».
        """
        payload = json.dumps(self.to_dicts(), ensure_ascii=False,
                             separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Квантование навески (iter37, скрин-аудит п.3): nominal vs actual
    # ------------------------------------------------------------------
    def quantize_recipe(self, p: Sequence[float] | np.ndarray,
                        delta_phr: float) -> "QuantizeReport":
        """Номинальный рецепт → ФАКТИЧЕСКАЯ навеска на сетке весов + проверка
        «после округления до разрешения весов точка всё ещё в границах».

        Реальный лабораторный риск (скрин-аудит 05.08.2026, п.3): весы имеют
        шаг ``delta_phr`` (в phr текущей загрузки), лаборант навешивает НЕ
        номинал, а ближайшее кратное шага — и округлённая точка может выйти
        из геометрии спеки (границы узла, cap-потолок, fixed-значение).

        Алгоритм:

        1. Каждый ЛИСТ снапится к ближайшему узлу δ-сетки ВНУТРИ своего
           статического интервала phr; если ближайший узел вылетает за
           ``[lo, hi]`` — берётся ближайший узел внутри интервала; если в
           интервале НЕТ ни одного узла сетки (диапазон уже шага весов) —
           violation: ось нечитаема прямой навеской (см.
           :func:`premix_required`).
        2. По фактическим (actual) значениям листьев пересчитываются
           внутренние узлы (сумма детей) и проверяются ДИНАМИЧЕСКИЕ
           ограничения спеки: fixed-значения, границы share_of/ratio_to,
           cap-потолки. Допуск проверки каждого узла — накопленная ошибка
           округления его листьев (δ × число листьев под узлом): честный
           сдвиг от квантования не флагается, структурные выходы — флагаются.

        Ничего не подгоняется молча (A0.6): каждый выход за геометрию —
        строка в ``violations`` (``ok=False``), решение (премикс, пересчёт
        рецепта, другая навеска) — за пользователем.
        """
        delta = float(delta_phr)
        if delta <= 0:
            raise ValueError("quantize_recipe: delta_phr должен быть > 0.")
        p = np.asarray(p, dtype=float).ravel()
        if p.size != self.q:
            raise ValueError(
                f"quantize_recipe: ожидалось {self.q} компонентов "
                f"({self.component_names}), получено {p.size}.")
        leaf_col = {nm: j for j, nm in enumerate(self.component_names)}
        violations: List[str] = []

        # --- 1: снап листьев к δ-сетке внутри статического интервала -----
        actual = np.empty_like(p)
        for nm, j in leaf_col.items():
            lo, hi = self._interval[nm]
            x = round(p[j] / delta) * delta
            if x < lo - _TOL:
                x = math.ceil((lo - _TOL) / delta) * delta
            elif x > hi + _TOL:
                x = math.floor((hi + _TOL) / delta) * delta
            if x < lo - _TOL or x > hi + _TOL:
                violations.append(
                    f"{nm}: в интервале phr [{lo:g}, {hi:g}] нет узла сетки "
                    f"δ={delta:g} — прямой навеской ось нечитаема (премикс).")
                x = min(max(round(p[j] / delta) * delta, lo), hi)
            actual[j] = x

        # --- 2: фактические значения всех узлов (дети → родители) --------
        vals: Dict[str, float] = {}
        n_leaves: Dict[str, int] = {}
        for nm in reversed(self._topo):
            if nm in self._share_groups:
                members = self._share_groups[nm]
                vals[nm] = float(sum(vals[m] for m in members))
                n_leaves[nm] = int(sum(n_leaves[m] for m in members))
            else:
                vals[nm] = float(actual[leaf_col[nm]])
                n_leaves[nm] = 1

        # --- 3: динамические ограничения с допуском на округление --------
        for nm in self._topo:
            nd = self._by_name[nm]
            v = vals[nm]
            tol_v = delta * n_leaves[nm] + _TOL
            if nd.mode == MODE_FIXED:
                if abs(v - nd.value) > tol_v:
                    violations.append(
                        f"{nm}: fixed={nd.value:g}, факт {v:g} — отклонение "
                        f"больше допуска квантования {tol_v:g}.")
            elif nd.mode == MODE_ABSOLUTE:
                if v < nd.lo - tol_v or v > nd.hi + tol_v:
                    violations.append(
                        f"{nm}: факт {v:g} вне границ [{nd.lo:g}, {nd.hi:g}] "
                        f"с допуском {tol_v:g}.")
                if nd.cap_refs:
                    cap_sum = float(sum(vals[cr] for cr in nd.cap_refs))
                    tol_cap = (tol_v + nd.cap_ratio
                               * sum(delta * n_leaves[cr] + _TOL
                                     for cr in nd.cap_refs))
                    if v > nd.cap_ratio * cap_sum + tol_cap:
                        violations.append(
                            f"{nm}: факт {v:g} превышает потолок "
                            f"{nd.cap_ratio:g}·Σ{list(nd.cap_refs)} "
                            f"(= {nd.cap_ratio * cap_sum:g}) с допуском "
                            f"{tol_cap:g} — точка вне трапеции.")
            else:                                   # share-режимы / ratio_to
                ref_v = vals[nd.ref]
                if ref_v <= _TOL:
                    violations.append(
                        f"{nm}: референс '{nd.ref}' после округления равен 0 "
                        f"— коэффициент не определён.")
                    continue
                r = v / ref_v
                s_lo, s_hi = self._share_base.get(nm, (nd.lo, nd.hi))
                tol_r = (tol_v + s_hi * (delta * n_leaves[nd.ref] + _TOL)
                         ) / ref_v
                if r < s_lo - tol_r or r > s_hi + tol_r:
                    violations.append(
                        f"{nm}: коэффициент {r:g} вне границ "
                        f"[{s_lo:g}, {s_hi:g}] с допуском {tol_r:g}.")
                if nd.min_phr is not None and v < nd.min_phr - tol_v:
                    violations.append(
                        f"{nm}: факт {v:g} phr ниже технологического минимума "
                        f"{nd.min_phr:g} с допуском {tol_v:g}.")
                if nd.max_phr is not None and v > nd.max_phr + tol_v:
                    violations.append(
                        f"{nm}: факт {v:g} phr выше лимита {nd.max_phr:g} "
                        f"с допуском {tol_v:g}.")

        moved = np.abs(actual - p)
        return QuantizeReport(
            p_nominal=p.copy(), p_actual=actual, delta_phr=delta,
            moved_max=float(moved.max()) if moved.size else 0.0,
            violations=violations)

    # ------------------------------------------------------------------
    # Контракт-ответ ядра на точку (iter49/B7): effective_bounds + active,
    # premix_required, phr nominal vs actual — раздельно
    # ------------------------------------------------------------------
    def point_report(self, p: Sequence[float] | np.ndarray,
                     delta_phr: Optional[float] = None,
                     tol: float = 1e-6) -> "PointReport":
        """Структурированный ответ ядра на ОДНУ точку-рецепт (iter49/B7).

        Вход — НОМИНАЛЬНЫЙ рецепт ``p`` в phr (листья DAG, порядок =
        ``component_names``); внутренние узлы восстанавливаются суммой
        детей. Выход — :class:`PointReport` с тремя слоями:

        1. ``effective_bounds`` — для КАЖДОГО узла спеки (включая
           внутренние тоталы и производные closure) эффективные границы
           его СОБСТВЕННОЙ координаты В ЭТОЙ ТОЧКЕ и метки
           ``active_lo``/``active_hi`` — какое ограничение задало границу:

           * ``fixed``    — фиксированное значение узла;
           * ``range``    — заявленный интервал (``range``/``share_range``);
           * ``derived``  — производный диапазон closure
             ``[1−φᵁ_free, 1−φᴸ_free]`` (iter46/B2);
           * ``window``   — окно тотала группы по phr-лимитам членов
             (iter45/B1);
           * ``cap``      — динамический потолок ``cap_ratio·Σ(cap_to)``,
             вычисленный ПО ЗНАЧЕНИЯМ ЭТОЙ ТОЧКИ;
           * ``min_phr`` / ``max_phr`` — техлимиты узла при тотале точки;
           * ``partners`` — партнёрское сужение Σφ=1
             (``1 − Σ границ партнёров`` с учётом ИХ phr-лимитов).

           При равенстве кандидатов (плато ``hi(T)``, LUB впритык)
           приоритет у более ПРОСТОГО объяснения: range/derived →
           min/max_phr → partners; метка «партнёры» появляется, только
           когда партнёрское сужение СТРОГО активно. Границы
           absolute-осей — ВСЕГДА в phr (у log-осей iter47/B5 тоже:
           шкала — деталь сэмплера, контракт отвечает в физических
           единицах).

        2. ``premix`` — правило премикса :func:`premix_required` по
           СТАТИЧЕСКОМУ интервалу phr листа (``phr_intervals``); без
           ``delta_phr`` и на вырожденных интервалах (fixed-оси) —
           ``None`` («правило неприменимо»), а не ``False``.

        3. ``phr_nominal`` vs ``phr_actual`` — РАЗДЕЛЬНО: actual — снап к
           δ-сетке весов (:meth:`quantize_recipe`), только при заданном
           ``delta_phr``; его violations добавляются в общий список.

        Ничего не блокируется молча (A0.6): номинал вне эффективных
        границ и пустые границы — строки в ``violations`` (``ok=False``),
        НЕ исключения. Исключения — только ошибки ДАННЫХ, при которых
        координаты не определены: неверная длина ``p``, нулевой тотал
        share-группы / референс ratio-узла, ``delta_phr ≤ 0``.
        """
        if delta_phr is not None and float(delta_phr) <= 0:
            raise ValueError("point_report: delta_phr должен быть > 0.")
        p = np.asarray(p, dtype=float).ravel()
        if p.size != self.q:
            raise ValueError(
                f"point_report: ожидалось {self.q} компонентов "
                f"({self.component_names}), получено {p.size}.")
        leaf_col = {nm: j for j, nm in enumerate(self.component_names)}
        vals: Dict[str, float] = {}
        for nm in reversed(self._topo):        # дети раньше родителей
            if nm in self._share_groups:
                vals[nm] = float(sum(vals[m]
                                     for m in self._share_groups[nm]))
            else:
                vals[nm] = float(p[leaf_col[nm]])
        violations: List[str] = []

        def _pick_max(cands):                  # lo = max кандидатов;
            v0, lab0 = cands[0]                # тай-брейк — первый (простой)
            for v, lab in cands[1:]:
                if v > v0 + _TOL:
                    v0, lab0 = v, lab
            return v0, lab0

        def _pick_min(cands):                  # hi = min; тот же приоритет
            v0, lab0 = cands[0]
            for v, lab in cands[1:]:
                if v < v0 - _TOL:
                    v0, lab0 = v, lab
            return v0, lab0

        bounds: Dict[str, EffectiveBound] = {}
        for nd in self.nodes:                  # порядок спеки
            nm = nd.name
            v = vals[nm]
            if nd.mode == MODE_FIXED:
                coord = v
                lo = hi = nd.value
                al = ah = "fixed"
            elif nd.mode == MODE_ABSOLUTE:
                coord = v                      # phr (лог-ось — тоже в phr)
                a_lo, a_hi = self._axis_bounds(nm)
                narrowed = nm in self._total_window
                lo = a_lo
                al = ("window" if narrowed and a_lo > nd.lo + _TOL
                      else "range")
                hi_c = [(a_hi, "window" if narrowed and a_hi < nd.hi - _TOL
                         else "range")]
                if nd.cap_refs:
                    cap = nd.cap_ratio * float(
                        sum(vals[cr] for cr in nd.cap_refs))
                    hi_c.append((cap, "cap"))
                hi, ah = _pick_min(hi_c)
            elif nd.mode == MODE_RATIO_TO:
                ref_v = vals[nd.ref]
                if ref_v <= _TOL:
                    raise ValueError(
                        f"point_report: референс '{nd.ref}' узла '{nm}' "
                        f"равен {ref_v:g} — коэффициент не определён.")
                coord = v / ref_v
                lo, hi = nd.lo, nd.hi
                al = ah = "range"
            else:                              # share-режимы
                T = vals[nd.ref]
                if T <= _TOL:
                    raise ValueError(
                        f"point_report: тотал '{nd.ref}' группы узла "
                        f"'{nm}' равен {T:g} — доли не определены.")
                coord = v / T
                base_lo, base_hi = self._share_base[nm]
                base_lab = ("derived" if nd.mode == MODE_SHARE_CLOSURE
                            else "range")
                lo_c = [(base_lo, base_lab)]
                hi_c = [(base_hi, base_lab)]
                if nd.min_phr is not None:
                    lo_c.append((nd.min_phr / T, "min_phr"))
                if nd.max_phr is not None:
                    hi_c.append((nd.max_phr / T, "max_phr"))
                members = self._share_groups[nd.ref]
                LO0, HI0 = self._share_box_at_total(members, np.array([T]))
                i = members.index(nm)
                lo_c.append((1.0 - (float(HI0[0].sum()) - float(HI0[0, i])),
                             "partners"))
                hi_c.append((1.0 - (float(LO0[0].sum()) - float(LO0[0, i])),
                             "partners"))
                lo, al = _pick_max(lo_c)
                hi, ah = _pick_min(hi_c)
            bounds[nm] = EffectiveBound(
                name=nm, mode=nd.mode, coord=float(coord), phr=float(v),
                lo=float(lo), hi=float(hi), active_lo=al, active_hi=ah)
            if lo > hi + tol:
                violations.append(
                    f"{nm}: эффективные границы пусты в этой точке "
                    f"([{lo:g}, {hi:g}], lo: {al}, hi: {ah}).")
            elif coord < lo - tol or coord > hi + tol:
                violations.append(
                    f"{nm}: nominal {coord:g} вне эффективных границ "
                    f"[{lo:g}, {hi:g}] (lo: {al}, hi: {ah}).")

        premix: Dict[str, Optional[bool]] = {}
        for nm in self.component_names:
            lo_i, hi_i = self._interval[nm]
            if delta_phr is None or hi_i <= lo_i + _TOL:
                premix[nm] = None              # правило неприменимо
            else:
                premix[nm] = premix_required(delta_phr, lo_i, hi_i)

        phr_actual: Optional[np.ndarray] = None
        if delta_phr is not None:
            qr = self.quantize_recipe(p, delta_phr)
            phr_actual = qr.p_actual
            violations.extend(qr.violations)

        return PointReport(
            phr_nominal=p.copy(), phr_actual=phr_actual,
            delta_phr=None if delta_phr is None else float(delta_phr),
            effective_bounds=bounds, premix=premix, violations=violations)

    def fraction_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Консервативный fraction-бокс ``L_i ≤ x_i ≤ U_i`` по интервалам
        phr листьев (математика экстремальных тоталей
        :func:`parts_ranges_to_fraction_bounds`). Бокс СОДЕРЖИТ образ
        decode (корреляции ratio_to/share_of его только сужают) — годится
        для построения mixture-блока схемы, содержащего всех кандидатов."""
        a = np.array([self._interval[nm][0] for nm in self.component_names])
        b = np.array([self._interval[nm][1] for nm in self.component_names])
        return parts_ranges_to_fraction_bounds(a, b)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"PhrSpec(q={self.q}, dim_z={self.dim_z}, "
                f"components={self.component_names})")


@dataclass
class QuantizeReport:
    """Итог квантования рецепта к разрешению весов (iter37, слой m, п.3).

    ``p_nominal``/``p_actual`` — phr листьев ДО/ПОСЛЕ снапа к δ-сетке
    (порядок = ``PhrSpec.component_names``); ``moved_max`` — максимальный
    сдвиг листа; ``violations`` — человекочитаемые нарушения геометрии
    после округления (пусто ⇒ ``ok``). Дозируйте ФАКТИЧЕСКИЕ значения и
    храните их рядом с номиналом: модель должна видеть actual, а не nominal.
    """
    p_nominal: np.ndarray
    p_actual: np.ndarray
    delta_phr: float
    moved_max: float
    violations: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations


@dataclass
class EffectiveBound:
    """Эффективные границы ОДНОГО узла в точке (iter49/B7, элемент
    :class:`PointReport`). ``coord`` — значение узла в его СОБСТВЕННОЙ
    координате (phr / доля / коэффициент), ``phr`` — значение в phr
    (у листьев совпадает с рецептом, у тоталов — сумма детей);
    ``lo``/``hi`` — эффективные границы координаты В ЭТОЙ ТОЧКЕ;
    ``active_lo``/``active_hi`` — какое ограничение задало границу
    (``fixed`` / ``range`` / ``derived`` / ``window`` / ``cap`` /
    ``min_phr`` / ``max_phr`` / ``partners`` —
    см. :meth:`PhrSpec.point_report`)."""
    name: str
    mode: str
    coord: float
    phr: float
    lo: float
    hi: float
    active_lo: str
    active_hi: str


@dataclass
class PointReport:
    """Контракт-ответ ядра на точку (iter49/B7).

    ``phr_nominal``/``phr_actual`` — РАЗДЕЛЬНО: номинал, как предложило
    ядро, и факт после снапа к δ-сетке весов (``None`` без ``delta_phr``).
    Дозируйте и фиксируйте actual — модель должна видеть actual, а не
    nominal (CAMPAIGN_SPEC_PVC §5). ``effective_bounds`` — по КАЖДОМУ
    узлу спеки: эффективные границы в точке + метки active;
    ``premix`` — лист → нужен ли премикс (``None`` — правило
    неприменимо: δ не задан или интервал вырожден); ``violations`` —
    номинал вне геометрии + нарушения квантования (пусто ⇒ ``ok``).
    """
    phr_nominal: np.ndarray
    phr_actual: Optional[np.ndarray]
    delta_phr: Optional[float]
    effective_bounds: Dict[str, EffectiveBound]
    premix: Dict[str, Optional[bool]]
    violations: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations


# ----------------------------------------------------------------------
# Бисекция по НЕВОЗРАСТАЮЩЕЙ функции (iter45/B1): окно тотала группы
# ----------------------------------------------------------------------
def _bisect_decreasing(f, lo: float, hi: float, target: float,
                       iters: int = 200) -> float:
    """Наименьший ``T ∈ [lo, hi]`` с ``f(T) ≤ target`` для НЕВОЗРАСТАЮЩЕЙ
    ``f``. Предполагается ``f(hi) ≤ target`` (проверено вызывающим кодом);
    возвращается правый конец вилки — точка, где условие уже выполнено
    (консервативно: окно тотала не расширяется за счёт ошибки бисекции).
    """
    a, b = float(lo), float(hi)
    for _ in range(int(iters)):
        mid = 0.5 * (a + b)
        if f(mid) <= target:
            b = mid
        else:
            a = mid
        if b - a <= 1e-12 * max(1.0, abs(b)):
            break
    return b


# ----------------------------------------------------------------------
# Правило премикса (iter35) — арифметика навески, слой m (этап B, вынесено
# вперёд как чистая функция: от геометрии не зависит)
# ----------------------------------------------------------------------
def premix_required(delta_phr: float, lo: float, hi: float,
                    threshold: float = 0.05) -> bool:
    """Нужен ли премикс для компонента с диапазоном phr ``[lo, hi]``.

    ``delta_phr`` — разрешение навески в phr (шаг весов, переведённый в phr
    текущей загрузки: ``delta_phr = шаг_весов_г / (г на 1 phr)``). Правило
    (DECODE_LAYER_PROPOSAL, слой m): если ошибка дозирования съедает больше
    ``threshold`` (5%) РАБОЧЕГО ДИАПАЗОНА оси — компонент дозируется через
    премикс (разбавленный концентрат), иначе план по этой оси нечитаем::

        delta_phr / (hi - lo) > threshold  =>  премикс

    Пример (замер 04.08.2026): весы 0.1 г при 5 г/phr → δ=0.02 phr;
    SBM_55 [0.07, 0.45]: 0.02/0.38 ≈ 0.053 > 0.05 → премикс;
    UV_CSFCP [0.05, 0.30]: 0.02/0.25 = 0.08 > 0.05 → премикс;
    DINP [4, 14]: 0.02/10 = 0.002 → прямая навеска.
    """
    delta_phr = float(delta_phr)
    lo = float(lo)
    hi = float(hi)
    if delta_phr < 0:
        raise ValueError("premix_required: delta_phr должен быть ≥ 0.")
    if hi <= lo + _TOL:
        raise ValueError(
            f"premix_required: вырожденный диапазон [{lo}, {hi}] — "
            f"фиксированная дозировка, правило премикса неприменимо.")
    return delta_phr / (hi - lo) > float(threshold)
