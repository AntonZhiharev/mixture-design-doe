"""design/linked_axes.py — СВЯЗАННЫЕ process-оси (P3.3, UI_REVISION_SPEC).

Мотивация (CAMPAIGN_SPEC_PVC): часть физически значимых величин кампании —
РАЗНОСТИ двух process-осей. Канонический пример ПВХ: ``dT_head =
T_адаптер − T_пласт`` — движущая сила сплавления в голове экструдера.
Железо ограничивает саму РАЗНОСТЬ (нагреватель адаптера не может держать
перепад больше паспортного, а отрицательный перепад гонит расплав назад),
хотя каждая ось по отдельности в своих границах. Пока ядро считало
process-оси независимым боксом, план мог предложить пару температур с
нереализуемым перепадом — оператор ставил «что получится», модель училась
на координатах из таблицы (ровно та тихая подмена, против которой A0.6).

Слой связок — ПРОЕКЦИЯ на полосу, а не отдельная геометрия (канон
слоя уровней iter51):

  * связка объявляется в ФИЗИЧЕСКИХ единицах: ``lo ≤ A − B ≤ hi``
    (код [0,1] зависит от границ осей — при их правке полоса «поехала бы»
    молча);
  * проекция — минимальный сдвиг ПАРЫ (a, b) по L2 в физических единицах
    на ближайшую грань полосы (симметричный сдвиг ``∓Δ/2`` обеим осям),
    затем — точное решение на грани с учётом боксов осей; операция
    ИДЕМПОТЕНТНА;
  * статическая валидация требует НЕПУСТОГО пересечения полосы с
    достижимым диапазоном разности ``[aL−bU, aU−bL]`` — иначе проекция
    была бы не определена, и это ошибка КОНФИГУРАЦИИ, а не данных;
  * ось может состоять НЕ БОЛЕЕ чем в ОДНОЙ связке: последовательная
    проекция двух полос с общей осью не гарантирует совместного решения,
    и «почти удовлетворённая» вторая связка была бы тихой ложью (A0.6).

Все функции ЧИСТЫЕ (numpy, без Streamlit и без раннера) — тестируются
напрямую (канон UI_REVISION_SPEC §1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = ["ProcessLink", "normalize_links", "snap_pair_to_band",
           "links_caption"]

_TOL = 1e-9


@dataclass(frozen=True)
class ProcessLink:
    """Связка ``name = minuend − subtrahend ∈ [lo, hi]`` (физические единицы).

    ``lo``/``hi`` — границы реализуемости РАЗНОСТИ по железу; односторонняя
    связка задаётся ``±inf`` (хотя бы одна сторона обязана быть конечной).
    Сама по себе dataclass валидацию не делает — единый источник правил
    :func:`normalize_links` (там есть контекст: имена и границы осей).
    """

    name: str
    minuend: str
    subtrahend: str
    lo: float = -np.inf
    hi: float = np.inf


def _as_bound(value: Any, *, default: float, where: str) -> float:
    """None → default (±inf); число → float; NaN/мусор → явная ошибка."""
    if value is None:
        return float(default)
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: граница {value!r} не число.")
    if np.isnan(f):
        raise ValueError(f"{where}: граница NaN недопустима.")
    return f


def normalize_links(links: Sequence[Any], *, names: Sequence[str],
                    lower: Sequence[float], upper: Sequence[float],
                    tol: float = _TOL) -> List[ProcessLink]:
    """Проверить и нормализовать список связок против осей ``names``.

    ``links`` — последовательность :class:`ProcessLink` или словарей с
    ключами ``name`` / ``minuend`` / ``subtrahend`` / ``lo`` / ``hi``
    (``lo``/``hi`` могут быть ``None`` — односторонняя связка).
    ``names``/``lower``/``upper`` — process-оси и их ФИЗИЧЕСКИЕ границы.

    Ошибки (явные, не тихая правка — A0.6):

      * пустое имя связки; имя, совпадающее с именем оси (одно имя с двумя
        смыслами — тихая путаница); повтор имени связки;
      * ``minuend``/``subtrahend`` не среди осей или совпадают между собой;
      * ось участвует более чем в ОДНОЙ связке (см. шапку модуля);
      * обе границы бесконечны («связка без ограничения» — не связка);
        ``lo ≥ hi``; NaN;
      * полоса НЕ пересекает достижимый диапазон разности
        ``[aL−bU, aU−bL]`` — конфигурация пуста, проекция не определена.
    """
    known = [str(n) for n in names]
    lo_by = {n: float(l) for n, l in zip(known, lower)}
    hi_by = {n: float(u) for n, u in zip(known, upper)}
    out: List[ProcessLink] = []
    seen_names: set = set()
    busy_axes: set = set()
    for i, raw in enumerate(links):
        where = f"связка №{i + 1}"
        if isinstance(raw, ProcessLink):
            nm, a, b = raw.name, raw.minuend, raw.subtrahend
            lo_v, hi_v = raw.lo, raw.hi
        elif isinstance(raw, Mapping):
            nm = str(raw.get("name", "") or "")
            a = str(raw.get("minuend", "") or "")
            b = str(raw.get("subtrahend", "") or "")
            lo_v, hi_v = raw.get("lo", None), raw.get("hi", None)
        else:
            raise ValueError(
                f"{where}: ожидается ProcessLink или словарь "
                f"{{name, minuend, subtrahend, lo, hi}}, дано {type(raw).__name__}.")
        nm = str(nm).strip()
        a, b = str(a).strip(), str(b).strip()
        if not nm:
            raise ValueError(f"{where}: пустое имя производной величины.")
        if nm in known:
            raise ValueError(
                f"{where}: имя '{nm}' совпадает с именем process-оси — одно "
                f"имя с двумя смыслами это тихая путаница данных (A0.6).")
        if nm in seen_names:
            raise ValueError(f"{where}: имя '{nm}' задано дважды.")
        for ax in (a, b):
            if ax not in known:
                raise KeyError(
                    f"{where} '{nm}': ось '{ax}' не найдена среди "
                    f"process-осей {known}.")
        if a == b:
            raise ValueError(
                f"{where} '{nm}': уменьшаемое и вычитаемое совпадают "
                f"('{a}') — разность тождественно 0.")
        for ax in (a, b):
            if ax in busy_axes:
                raise ValueError(
                    f"{where} '{nm}': ось '{ax}' уже участвует в другой "
                    f"связке. Ось может состоять не более чем в ОДНОЙ "
                    f"связке: совместная проекция двух полос с общей осью "
                    f"не гарантирована (A0.6 — честный отказ вместо "
                    f"«почти удовлетворено»).")
        lo_f = _as_bound(lo_v, default=-np.inf, where=f"{where} '{nm}' (lo)")
        hi_f = _as_bound(hi_v, default=np.inf, where=f"{where} '{nm}' (hi)")
        if not (np.isfinite(lo_f) or np.isfinite(hi_f)):
            raise ValueError(
                f"{where} '{nm}': обе границы бесконечны — «связка без "
                f"ограничения» не связка. Задайте lo и/или hi.")
        if not (lo_f < hi_f):
            raise ValueError(
                f"{where} '{nm}': требуется lo < hi (дано lo={lo_f:g}, "
                f"hi={hi_f:g}).")
        # достижимый диапазон разности при независимых боксах осей
        ach_lo = lo_by[a] - hi_by[b]
        ach_hi = hi_by[a] - lo_by[b]
        if max(lo_f, ach_lo) > min(hi_f, ach_hi) + tol:
            raise ValueError(
                f"{where} '{nm}': полоса [{lo_f:g}; {hi_f:g}] не пересекает "
                f"достижимый диапазон разности {a} − {b} ∈ "
                f"[{ach_lo:g}; {ach_hi:g}] (границы осей). Область пуста — "
                f"исправьте полосу или границы осей.")
        seen_names.add(nm)
        busy_axes.update((a, b))
        out.append(ProcessLink(name=nm, minuend=a, subtrahend=b,
                               lo=lo_f, hi=hi_f))
    return out


def _project_to_plane(a: np.ndarray, b: np.ndarray, c: float,
                      a_bounds: Tuple[float, float],
                      b_bounds: Tuple[float, float]
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Ближайшая (L2) точка грани ``a − b = c`` внутри боксов осей.

    Параметризация грани: ``a = t``, ``b = t − c``,
    ``t ∈ [max(aL, bL + c), min(aU, bU + c)]``; безусловный минимум
    ``t* = (a + b + c) / 2`` (симметричный сдвиг ``∓Δ/2``), затем clip к
    отрезку. Непустота отрезка гарантирована статической валидацией
    :func:`normalize_links` (грань достижима ⇔ ``c`` в достижимом
    диапазоне разности).
    """
    aL, aU = float(a_bounds[0]), float(a_bounds[1])
    bL, bU = float(b_bounds[0]), float(b_bounds[1])
    t = (a + b + c) / 2.0
    t = np.clip(t, max(aL, bL + c), min(aU, bU + c))
    return t, t - c


def snap_pair_to_band(a, b, lo: float, hi: float,
                      a_bounds: Tuple[float, float],
                      b_bounds: Tuple[float, float],
                      tol: float = 1e-12
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Спроецировать пары ``(a, b)`` на полосу ``lo ≤ a − b ≤ hi`` (копии).

    Всё — в ФИЗИЧЕСКИХ единицах (метрика L2 в градусах/оборотах осмысленна;
    в коде [0,1] оси с разным размахом весились бы по-разному). Пары внутри
    полосы НЕ трогаются (⇒ идемпотентность); нарушители сдвигаются на
    ближайшую грань (``a − b = hi`` либо ``= lo``) с учётом боксов осей.
    Результат всегда в боксах и в полосе — при условии, что полоса прошла
    статическую валидацию :func:`normalize_links`.
    """
    a = np.asarray(a, float).copy()
    b = np.asarray(b, float).copy()
    d = a - b
    if np.isfinite(hi):
        m = d > hi + tol
        if np.any(m):
            a2, b2 = _project_to_plane(a[m], b[m], float(hi),
                                       a_bounds, b_bounds)
            a[m], b[m] = a2, b2
    if np.isfinite(lo):
        m = d < lo - tol
        if np.any(m):
            a2, b2 = _project_to_plane(a[m], b[m], float(lo),
                                       a_bounds, b_bounds)
            a[m], b[m] = a2, b2
    return a, b


def links_caption(links: Sequence[Any]) -> str:
    """Подпись «какие оси связаны» для паспорта кампании (чистая).

    Пусто → явная строка «связанных осей нет»: молчание читалось бы как
    «поле не заполнено», а это разные вещи (тот же канон, что
    :func:`design.levels.levels_caption`).
    """
    items = list(links or [])
    if not items:
        return "Связанных process-осей нет: оси независимы."
    parts = []
    for lk in items:
        if isinstance(lk, Mapping):
            nm, a, b = lk.get("name"), lk.get("minuend"), lk.get("subtrahend")
            lo = lk.get("lo", None)
            hi = lk.get("hi", None)
            lo = -np.inf if lo is None else float(lo)
            hi = np.inf if hi is None else float(hi)
        else:
            nm, a, b, lo, hi = lk.name, lk.minuend, lk.subtrahend, lk.lo, lk.hi
        lo_s = f"{float(lo):g}" if np.isfinite(lo) else "−∞"
        hi_s = f"{float(hi):g}" if np.isfinite(hi) else "+∞"
        parts.append(f"{nm} = {a} − {b} ∈ [{lo_s}; {hi_s}]")
    return "Связанные process-оси — " + "; ".join(parts) + "."