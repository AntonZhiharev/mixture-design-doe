"""
design/phr_sampler.py — этап A decode-слоя (DECODE_LAYER_PROPOSAL, iter33):
z-сэмплер-плагин parts/phr-спеки.

Двухслойная параметризация phr-рецептур (PVC-компаунды и т.п.):

    z (design coords) --decode--> p (phr) --w_i = p_i / Σp--> x (доли, Σ=1)

Спека — список узлов, каждый в одном из 4 режимов координат:

  * ``absolute``  — phr равномерно в ``[lo, hi]`` (в т.ч. ТОТАЛ группы);
    опционально с ДИНАМИЧЕСКИМ ПОТОЛКОМ ``cap_to``/``cap_ratio``:
    ``p ∈ [lo, min(hi, cap_ratio · value(cap_to))]`` — ТРАПЕЦИЯ, а не клин.
    Кейс (сессия 05.08.2026, UV_CSFCP): растворимость ограничивает УФ
    только СВЕРХУ (`UV ≤ 0.03·DINP`), нижняя граница 0.05 phr — требование
    по защите и от пластификатора НЕ зависит. ``ratio_to`` здесь неверен:
    он масштабирует ОБА конца (клин), вшивая положительную корреляцию с
    доминирующей осью и монотонный prior, которого физика не требует;
  * ``share_of``  — доля родительского узла ``of`` в ``[lo, hi] ⊆ [0,1]``;
    доли одной группы связаны Σ=1 (раскладка без rejection —
    :func:`core.simplex._narrowing_split`, канон iter31: uniform-MARGINAL
    по физически значимым осям);
  * ``ratio_to``  — коэффициент к ПРОИЗВОЛЬНОМУ узлу ``to`` (не обязательно
    родителю — поэтому DAG, а не дерево): ``p = r · value(to)``,
    ``r ∈ [lo, hi]`` (пример: «SBM = 0.02…0.09 × Σ стабилизатора»);
  * ``fixed``     — константа phr (базовый компонент: смола = 100).

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
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.simplex import _narrowing_split, parts_ranges_to_fraction_bounds

MODE_ABSOLUTE = "absolute"
MODE_SHARE_OF = "share_of"
MODE_RATIO_TO = "ratio_to"
MODE_FIXED = "fixed"
_MODES = (MODE_ABSOLUTE, MODE_SHARE_OF, MODE_RATIO_TO, MODE_FIXED)

_TOL = 1e-9


@dataclass
class PhrNode:
    """Узел phr-спеки. ``ref`` — имя референса (``of`` у share_of, ``to``
    у ratio_to); у absolute/fixed референса нет. ``cap_ref``/``cap_ratio``
    (только absolute) — динамический потолок
    ``hi_eff = min(hi, cap_ratio · value(cap_ref))``."""
    name: str
    mode: str
    lo: float = 0.0
    hi: float = 0.0
    value: float = 0.0
    ref: str = ""
    cap_ref: str = ""
    cap_ratio: float = 0.0


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

    def __init__(self, nodes: Sequence[PhrNode]):
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
            if nd.mode == MODE_SHARE_OF:
                self._share_groups.setdefault(nd.ref, []).append(nd.name)
        self._topo: List[str] = self._toposort()
        # компоненты смеси = листья (НЕ родители share-групп), порядок спеки
        self.component_names: List[str] = [
            nd.name for nd in self.nodes if nd.name not in self._share_groups]
        if len(self.component_names) < 2:
            raise ValueError(
                "phr-спека должна давать ≥2 компонента смеси (листа DAG).")
        # z-оси: все НЕ-fixed узлы в порядке спеки (share-узел = его доля)
        self.z_names: List[str] = [nd.name for nd in self.nodes
                                   if nd.mode != MODE_FIXED]
        self._z_col: Dict[str, int] = {nm: j for j, nm
                                       in enumerate(self.z_names)}
        # статическая интервальная валидация (ДО любого сэмплинга)
        self._interval: Dict[str, Tuple[float, float]] = {}
        self._validate_intervals()

    # ------------------------------------------------------------------
    # Конструкторы
    # ------------------------------------------------------------------
    @classmethod
    def from_dicts(cls, dicts: Sequence[Mapping[str, Any]]) -> "PhrSpec":
        """Спека из списка словарей: ключи ``name``, ``mode``, ``lo``, ``hi``,
        ``value`` (fixed), ``of`` (share_of) / ``to`` (ratio_to),
        ``cap_to``/``cap_ratio`` (динамический потолок absolute-оси).
        Доли share_of без явных границ — ``[0, 1]``."""
        nodes: List[PhrNode] = []
        for d in dicts:
            mode = str(d.get("mode", ""))
            ref = str(d.get("of", d.get("to", "")) or "")
            lo = float(d.get("lo", 0.0))
            hi = float(d.get("hi", 1.0 if mode == MODE_SHARE_OF else 0.0))
            nodes.append(PhrNode(name=str(d.get("name", "")), mode=mode,
                                 lo=lo, hi=hi,
                                 value=float(d.get("value", 0.0)), ref=ref,
                                 cap_ref=str(d.get("cap_to", "") or ""),
                                 cap_ratio=float(d.get("cap_ratio", 0.0))))
        return cls(nodes)

    # ------------------------------------------------------------------
    # Валидация структуры
    # ------------------------------------------------------------------
    def _validate_refs(self) -> None:
        for nd in self.nodes:
            needs_ref = nd.mode in (MODE_SHARE_OF, MODE_RATIO_TO)
            if needs_ref:
                if not nd.ref:
                    raise ValueError(
                        f"Узел '{nd.name}' ({nd.mode}): не указан референс "
                        f"('of'/'to').")
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
            if nd.cap_ref:
                if nd.mode != MODE_ABSOLUTE:
                    raise ValueError(
                        f"Узел '{nd.name}' ({nd.mode}): cap_to допустим "
                        f"только для absolute.")
                if nd.cap_ref not in self._by_name:
                    raise ValueError(
                        f"Узел '{nd.name}': cap-референс '{nd.cap_ref}' "
                        f"не найден среди узлов спеки.")
                if nd.cap_ref == nd.name:
                    raise ValueError(
                        f"Узел '{nd.name}': cap-ссылка на самого себя.")
                if nd.cap_ratio <= 0:
                    raise ValueError(
                        f"Узел '{nd.name}': cap_ratio должен быть > 0 "
                        f"(получено {nd.cap_ratio}).")
            elif nd.cap_ratio:
                raise ValueError(
                    f"Узел '{nd.name}': cap_ratio без cap_to недопустим.")

    def _toposort(self) -> List[str]:
        """Топосорт Кана по рёбрам ref→node и cap_ref→node;
        цикл — явный ValueError."""
        indeg = {nd.name: 0 for nd in self.nodes}
        out_edges: Dict[str, List[str]] = {nd.name: [] for nd in self.nodes}
        for nd in self.nodes:
            if nd.ref:
                out_edges[nd.ref].append(nd.name)
                indeg[nd.name] += 1
            if nd.cap_ref:
                out_edges[nd.cap_ref].append(nd.name)
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
            if nd.lo < -_TOL or nd.lo > nd.hi + _TOL:
                raise ValueError(
                    f"Узел '{nd.name}': некорректные границы "
                    f"[{nd.lo}, {nd.hi}] (нужно 0 ≤ lo ≤ hi).")
            if nd.mode == MODE_SHARE_OF and nd.hi > 1 + _TOL:
                raise ValueError(
                    f"Узел '{nd.name}': доля share_of должна быть ≤ 1 "
                    f"(hi={nd.hi}).")
        for parent, members in self._share_groups.items():
            s_lo = sum(self._by_name[m].lo for m in members)
            s_hi = sum(self._by_name[m].hi for m in members)
            if s_lo > 1 + _TOL or s_hi < 1 - _TOL:
                raise ValueError(
                    f"Группа share_of узла '{parent}': пустое пересечение — "
                    f"Σlo={s_lo:.4f}, Σhi={s_hi:.4f}, требуется Σlo ≤ 1 ≤ Σhi.")
        for nm in self._topo:
            nd = self._by_name[nm]
            if nd.mode == MODE_FIXED:
                iv = (nd.value, nd.value)
            elif nd.mode == MODE_ABSOLUTE:
                if nd.cap_ref:
                    r_lo, r_hi = self._interval[nd.cap_ref]
                    cap_min = nd.cap_ratio * r_lo
                    if cap_min < nd.lo - _TOL:
                        raise ValueError(
                            f"Узел '{nd.name}': при минимальном "
                            f"'{nd.cap_ref}'={r_lo:.6g} потолок "
                            f"cap_ratio·ref={cap_min:.6g} < lo={nd.lo} — "
                            f"пустой диапазон (трапеция вырождается).")
                    iv = (nd.lo, min(nd.hi, nd.cap_ratio * r_hi))
                else:
                    iv = (nd.lo, nd.hi)
            else:  # ratio_to / share_of: произведение неотрицательных интервалов
                r_lo, r_hi = self._interval[nd.ref]
                if r_lo <= _TOL:
                    raise ValueError(
                        f"Узел '{nd.name}' ({nd.mode}) ссылается на "
                        f"'{nd.ref}' с нижней границей {r_lo:.6g} ≤ 0 — "
                        f"референс должен быть строго положительным.")
                iv = (nd.lo * r_lo, nd.hi * r_hi)
            self._interval[nm] = iv
        if sum(self._interval[nm][1] for nm in self.component_names) <= _TOL:
            raise ValueError("Суммарный phr компонентов не может быть > 0 — "
                             "пустая рецептура.")

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
        ``[lo, min(hi, cap_ratio · value(cap_ref))]`` (условная равномерность
        на трапеции); доли share-группы — :func:`_narrowing_split`
        (Σ=1 без rejection). Обход — топологический (референсы cap вычислены
        раньше зависимых осей).

        Мера: uniform-MARGINAL по физически значимым осям (тот же осознанный
        выбор, что iter31 ``SimplexRegion.random_points(groups=…)``).
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
                if nd.cap_ref:
                    hi_eff = np.minimum(nd.hi, nd.cap_ratio * vals[nd.cap_ref])
                    hi_eff = np.maximum(hi_eff, nd.lo)   # числовая страховка
                    z = nd.lo + (hi_eff - nd.lo) * rng.random(n)
                else:
                    z = rng.uniform(nd.lo, nd.hi, size=n)
                Z[:, self._z_col[nm]] = z
                vals[nm] = z
            elif nd.mode == MODE_RATIO_TO:
                z = rng.uniform(nd.lo, nd.hi, size=n)
                Z[:, self._z_col[nm]] = z
                vals[nm] = z * vals[nd.ref]
            else:                              # share_of: группа целиком
                if nd.ref in done_groups:
                    continue
                members = self._share_groups[nd.ref]
                cols = [self._z_col[m] for m in members]
                lo = np.array([self._by_name[m].lo for m in members])
                hi = np.array([self._by_name[m].hi for m in members])
                for t in range(n):
                    Z[t, cols] = _narrowing_split(lo, hi, 1.0, rng)
                for m in members:
                    vals[m] = Z[:, self._z_col[m]] * vals[nd.ref]
                done_groups.add(nd.ref)
        return Z

    # ------------------------------------------------------------------
    # decode / encode
    # ------------------------------------------------------------------
    def decode(self, z: Sequence[float] | np.ndarray) -> np.ndarray:
        """z → p: phr компонентов (столбцы = ``component_names``)."""
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
                vals[nm] = Z[:, self._z_col[nm]].copy()
            else:  # ratio_to / share_of
                vals[nm] = Z[:, self._z_col[nm]] * vals[nd.ref]
        P = np.column_stack([vals[nm] for nm in self.component_names])
        return P[0] if single else P

    def encode(self, p: Sequence[float] | np.ndarray,
               tol: float = 1e-6) -> np.ndarray:
        """p → z (обратное к :meth:`decode`): для anchors/исторических
        рецептов, заданных в phr. Внутренние узлы восстанавливаются суммой
        детей; несоответствие fixed-значению или выход за границы осей —
        явный ValueError (anchor вне области — ошибка данных)."""
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
            else:                              # ratio_to / share_of
                denom = vals[nd.ref]
                if np.any(denom <= _TOL):
                    raise ValueError(
                        f"encode: референс '{nd.ref}' узла '{nm}' равен 0 — "
                        f"коэффициент не определён.")
                zj = vals[nm] / denom
            if np.any(zj < nd.lo - tol) or np.any(zj > nd.hi + tol):
                raise ValueError(
                    f"encode: узел '{nm}' ({nd.mode}) вне границ "
                    f"[{nd.lo}, {nd.hi}]: значения {zj}.")
            if nd.mode == MODE_ABSOLUTE and nd.cap_ref:
                cap = nd.cap_ratio * vals[nd.cap_ref]
                if np.any(zj > cap + tol):
                    raise ValueError(
                        f"encode: узел '{nm}' превышает потолок "
                        f"{nd.cap_ratio:g}·'{nd.cap_ref}' (= {cap}): "
                        f"значения {zj} — рецепт вне спеки.")
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

    def sample_candidates(self, n: int,
                          seed: Optional[int] = None) -> np.ndarray:
        """``n`` кандидатов-долей (n × q, Σ=1): sample_z → decode →
        to_fractions. Готовый вход для пула кандидатов раннера (этап A —
        сэмплер-плагин, схема/модель не затрагиваются)."""
        return self.to_fractions(self.decode(self.sample_z(n, seed=seed)))

    # ------------------------------------------------------------------
    # Версионирование спеки (iter35): порядок узлов — часть спеки
    # ------------------------------------------------------------------
    def to_dicts(self) -> List[Dict[str, Any]]:
        """Каноническая сериализация спеки (round-trip c :meth:`from_dicts`).

        ПОРЯДОК УЗЛОВ СОХРАНЯЕТСЯ: он определяет и порядок компонентов, и
        порядок z-осей, и — через :func:`core.simplex._narrowing_split` /
        последовательное сужение — какие оси получают точную равномерную
        маргиналь (iter34, находка 1). Перестановка узлов — ДРУГАЯ спека.
        """
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
            elif nd.mode == MODE_RATIO_TO:
                d["to"] = nd.ref
            if nd.cap_ref:                     # динамический потолок — часть
                d["cap_to"] = nd.cap_ref       # геометрии, входит в отпечаток
                d["cap_ratio"] = float(nd.cap_ratio)
            out.append(d)
        return out

    def spec_hash(self) -> str:
        """SHA-256 отпечаток спеки (hex, 64 символа) для воспроизводимости.

        Хеш чувствителен к ПОРЯДКУ узлов (см. :meth:`to_dicts`): порядок
        влияет на меру сэмплера, поэтому обязан входить в отпечаток.
        Записывайте хеш в документацию кампании — через полгода по нему
        можно проверить, что геометрия не «уехала».
        """
        payload = json.dumps(self.to_dicts(), ensure_ascii=False,
                             separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

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
