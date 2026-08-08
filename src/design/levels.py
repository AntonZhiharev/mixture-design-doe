"""design/levels.py — ДИСКРЕТНЫЕ УРОВНИ process-осей (P2.1, UI_REVISION_SPEC).

Мотивация (CAMPAIGN_SPEC_PVC): часть process-осей физически НЕ непрерывна.
Экструдер даёт 400 или 900 об/мин (две передачи), профиль температур
набирается ступенями зоны, «время выдержки» — по таймеру с шагом. Пока
ядро считало такие оси непрерывным боксом, происходило ровно то, чего
требует избегать A0.6, — МОЛЧАЛИВОЕ расхождение плана и лаборатории:
план предлагал 673 об/мин, оператор ставил 900, а модель училась на 673.

Слой уровней — ПРОЕКЦИЯ на сетку, а не отдельная геометрия:

  * уровни задаются в ФИЗИЧЕСКИХ единицах (400/900 об/мин), а не в коде
    [0,1] — код зависит от границ оси, и при их правке уровни «поехали бы»
    молча;
  * снап — на БЛИЖАЙШИЙ уровень; при точной равноудалённости берётся
    МЕНЬШИЙ (детерминизм: иначе исход решает ошибка округления, и один и
    тот же план воспроизводился бы по-разному);
  * частота уровня в случайном пуле пропорциональна ширине его ячейки
    Вороного (для симметричных сеток — равномерно). Это осознанно: тот же
    принцип проекции применяется и к low-discrepancy пулу (Sobol), где
    подмена розыгрыша «выбором из списка» разрушила бы равномерность
    покрытия 2D-пар осей.

Все функции ЧИСТЫЕ (numpy, без Streamlit и без раннера) — тестируются
напрямую (канон UI_REVISION_SPEC §1).
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np

__all__ = ["normalize_levels", "snap_to_levels", "levels_to_code",
           "snap_matrix_to_levels", "levels_caption"]

_TOL = 1e-9


def normalize_levels(values: Sequence[float], *,
                     lower: Optional[float] = None,
                     upper: Optional[float] = None,
                     name: str = "ось",
                     tol: float = 1e-9) -> list:
    """Проверить и упорядочить уровни ОДНОЙ оси (физические единицы).

    Возвращает отсортированный по возрастанию список ``float``.

    Ошибки (явные, а не тихая правка — A0.6):

      * пустой список — «уровни заданы, но их нет» неотличимо от «оси на
        сетке нет»; выключать ось надо отсутствием ключа;
      * нечисловое значение / NaN / ±inf;
      * повторы (различие ≤ ``tol`` считается повтором: две «одинаковые»
        передачи 900 и 900.0000001 — ошибка ввода, а не сетка из двух);
      * уровень вне ``[lower, upper]`` — сетка обязана лежать в области
        оси, иначе план предлагал бы недостижимый режим.

    ОДИН уровень допустим: «в этой кампании только 900 об/мин» —
    законная постановка (ось фактически зафиксирована), и она честнее,
    чем схлопнутые границы, потому что видна в паспорте кампании.
    """
    vals = list(values)
    if not vals:
        raise ValueError(
            f"{name}: список уровней пуст. Чтобы ось была непрерывной, "
            "не задавайте для неё уровни вовсе.")
    out: list = []
    for v in vals:
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{name}: уровень {v!r} не число.")
        if not np.isfinite(f):
            raise ValueError(f"{name}: уровень {v!r} не конечное число.")
        out.append(f)
    out.sort()
    for a, b in zip(out[:-1], out[1:]):
        if abs(b - a) <= tol:
            raise ValueError(
                f"{name}: уровни {a:g} и {b:g} совпадают (различие ≤ {tol:g}).")
    if lower is not None and out[0] < float(lower) - tol:
        raise ValueError(
            f"{name}: уровень {out[0]:g} ниже нижней границы оси "
            f"{float(lower):g}.")
    if upper is not None and out[-1] > float(upper) + tol:
        raise ValueError(
            f"{name}: уровень {out[-1]:g} выше верхней границы оси "
            f"{float(upper):g}.")
    return out


def snap_to_levels(values, levels: Sequence[float]) -> np.ndarray:
    """Спроецировать значения на БЛИЖАЙШИЙ уровень (ties → меньший).

    ``levels`` обязаны быть отсортированы по возрастанию
    (:func:`normalize_levels`). Форма результата совпадает с формой входа;
    скаляр на входе → массив нулевой размерности не возвращается — вход
    приводится к ``np.asarray`` и сохраняет форму.

    Проекция ИДЕМПОТЕНТНА: снап уже снапнутого вектора ничего не меняет —
    без этого повторный показ плана «дрейфовал» бы между раундами.
    """
    lv = np.asarray(levels, float)
    if lv.ndim != 1 or lv.size == 0:
        raise ValueError("levels: непустой одномерный список уровней.")
    if np.any(np.diff(lv) <= 0):
        raise ValueError("levels: уровни должны идти строго по возрастанию.")
    x = np.asarray(values, float)
    idx = np.searchsorted(lv, x)
    left = np.clip(idx - 1, 0, lv.size - 1)
    right = np.clip(idx, 0, lv.size - 1)
    dl = np.abs(x - lv[left])
    dr = np.abs(lv[right] - x)
    take_left = dl <= dr                      # tie → МЕНЬШИЙ уровень
    return np.where(take_left, lv[left], lv[right])


def levels_to_code(levels: Sequence[float], lower: float,
                   upper: float) -> np.ndarray:
    """Уровни физической оси → координаты кода ``[0,1]`` (как VariableBlock).

    Вырожденная ось (``upper == lower``) кодируется нулями — тот же
    контракт, что у :meth:`core.schema.VariableBlock.to_code` (деление на
    единичный span), чтобы код уровней и код точек считались одинаково.
    """
    lv = np.asarray(levels, float)
    lo, hi = float(lower), float(upper)
    span = (hi - lo) if (hi - lo) > _TOL else 1.0
    return (lv - lo) / span


def snap_matrix_to_levels(Z, levels_by_col: Mapping[int, Sequence[float]]
                          ) -> np.ndarray:
    """Снап СТОЛБЦОВ матрицы к сеткам уровней (ключ — индекс столбца).

    ``Z`` — ``n×d`` (обычно process-часть в коде [0,1]), ``levels_by_col`` —
    ``{индекс столбца: уровни В ТЕХ ЖЕ единицах, что столбец}``. Столбцы
    без сетки не трогаются вовсе (непрерывные оси), поэтому включение
    уровней на одной оси не меняет остальные. Возвращается КОПИЯ.
    """
    Z = np.array(np.atleast_2d(np.asarray(Z, float)), copy=True)
    for j, lv in (levels_by_col or {}).items():
        j = int(j)
        if j < 0 or j >= Z.shape[1]:
            raise IndexError(
                f"levels_by_col: столбец {j} вне матрицы ширины {Z.shape[1]}.")
        Z[:, j] = snap_to_levels(Z[:, j], lv)
    return Z


def levels_caption(levels_by_name: Mapping[str, Sequence[float]]) -> str:
    """Подпись «какие оси дискретны» для паспорта кампании (чистая).

    Пусто → явная строка «все process-оси непрерывны»: молчание читалось
    бы как «поле не заполнено», а это разные вещи.
    """
    items = {str(k): list(v) for k, v in (levels_by_name or {}).items()}
    if not items:
        return "Дискретных уровней нет: все process-оси непрерывны."
    parts = []
    for name in items:
        lv = items[name]
        vals = ", ".join(f"{float(v):g}" for v in lv)
        parts.append(f"{name}: {len(lv)} ур. ({vals})")
    return "Дискретные оси — " + "; ".join(parts) + "."
