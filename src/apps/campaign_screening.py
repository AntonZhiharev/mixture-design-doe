"""apps/campaign_screening.py — M3-минималка: интерпретируемый анализ скрининга.

После измеренного стартового дизайна (seed) ядро кампании обучает суррогаты
GP+MoE (канон §5/§12), но пользователю нужна БЫСТРАЯ интерпретируемая сводка
«что показали опыты»: какой компонент на какое свойство влияет и насколько модель
объясняет данные. Этот слой — перенос M3 из спецификации (REBUILD_SPEC §17, «3b»:
«M3 screening — Scheffé-fit + ARD на КАЖДОЕ свойство; сводка главных эффектов»).

Объём минималки (согласовано): анализ ведётся ТОЛЬКО по СОСТАВУ (mixture-доли,
Scheffé-quadratic) — процесс-оси здесь игнорируются; их роль остаётся за общими
суррогатами GP+MoE. Слой ЧИСТЫЙ (без Streamlit): все функции тестируются напрямую
и возвращают либо ``pandas.DataFrame`` (для показа/Excel), либо JSON-сериализуемый
``dict`` (для UI и MCP-ассистента).

Переиспользуем проверенные M3-инструменты (decision #1 «перенос рабочих M1–M3»):
  * :class:`src.models.scheffe.ScheffeModel` — OLS Шеффе + ANOVA + p-values;
  * :func:`src.models.screening.significant_terms` — значимые термы (p < α);
  * :func:`src.models.screening.screen_components` — ARD-GP важность компонентов.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..models.scheffe import ScheffeModel
from ..models.screening import significant_terms, screen_components


# ----------------------------------------------------------------------
# Извлечение (состав, отклик) из ОБЩЕЙ базы кампании (mixture-only)
# ----------------------------------------------------------------------
def mixture_response(runner, prop: str) -> tuple[np.ndarray, np.ndarray]:
    """(Xmix, y) для свойства ``prop`` из общей базы: только mixture-доли + отклик.

    Берёт первые ``q`` координат составного ``runner.X`` (это mixture-доли Σ=1;
    процесс-оси идут после и здесь НЕ используются, объём минималки) и столбец
    ``runner.Y`` соответствующего свойства. Строки с не-измеренным (NaN) откликом
    выбрасываются (§13.7: новый отклик у старых точек = MISSING). ``prop`` вне
    схемы или пустая база — явный отказ (A0.6)."""
    if prop not in list(runner.property_names):
        raise KeyError(f"Свойство «{prop}» не объявлено в схеме "
                       f"({list(runner.property_names)}).")
    X = getattr(runner, "X", None)
    Y = getattr(runner, "Y", None)
    if X is None or Y is None or len(X) == 0:
        raise ValueError("Общая база пуста — сначала измерьте стартовый дизайн "
                         "(seed).")
    X = np.atleast_2d(np.asarray(X, float))
    Y = np.atleast_2d(np.asarray(Y, float))
    q = int(runner.current_schema.n_mixture)
    j = runner.prop_index[prop]
    Xmix = X[:, :q]
    y = Y[:, j]
    keep = ~np.isnan(y)
    return Xmix[keep], y[keep]


# ----------------------------------------------------------------------
# Фиты (переиспользуют M3-инструменты)
# ----------------------------------------------------------------------
def fit_scheffe(runner, prop: str, *, model: str = "quadratic") -> ScheffeModel:
    """Обучить :class:`ScheffeModel` (mixture-only) на свойстве ``prop`` (M3)."""
    Xmix, y = mixture_response(runner, prop)
    names = list(runner.current_schema.mixture_names)
    return ScheffeModel(model=model, names=names).fit(Xmix, y)


def component_importance(runner, prop: str, *, rel_threshold: float = 0.15,
                         n_restarts: int = 6, seed: Optional[int] = 0):
    """ARD-GP ранжирование компонентов по влиянию на ``prop`` (M3-скрининг)."""
    Xmix, y = mixture_response(runner, prop)
    names = list(runner.current_schema.mixture_names)
    return screen_components(Xmix, y, names=names, rel_threshold=rel_threshold,
                             n_restarts=n_restarts, seed=seed)


# ----------------------------------------------------------------------
# Сводный отчёт по свойству (JSON-сериализуемый — для UI и MCP)
# ----------------------------------------------------------------------
def _records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """DataFrame → список записей с НАТИВНЫМИ python-типами (JSON-safe; NaN→None)."""
    import json
    return json.loads(df.to_json(orient="records"))


def screening_report(runner, prop: str, *, alpha: float = 0.05,
                     model: str = "quadratic", rel_threshold: float = 0.15,
                     n_restarts: int = 6, seed: Optional[int] = 0
                     ) -> Dict[str, Any]:
    """M3-сводка по свойству ``prop`` (JSON-сериализуемый ``dict``).

    Содержит: имена компонентов; статистику фита (n, p, R², adj-R², RMSE, флаг
    недоопределённости n<p); таблицу коэффициентов Шеффе (оценка/SE/t/p);
    ANOVA (F-тест значимости регрессии); список значимых термов (p < ``alpha``);
    ARD-ранжирование компонентов (lengthscale/важность/active) и ``q_eff``.
    Всё анализируется ТОЛЬКО по составу (mixture-only, объём минималки)."""
    sm = fit_scheffe(runner, prop, model=model)
    sc = component_importance(runner, prop, rel_threshold=rel_threshold,
                              n_restarts=n_restarts, seed=seed)
    summ = sm.summary()
    summ["underdetermined"] = bool(summ["n"] < summ["p"])
    sig = significant_terms(sm, alpha=alpha)
    return {
        "property": prop,
        "model": model,
        "components": list(runner.current_schema.mixture_names),
        "summary": {k: (float(v) if isinstance(v, (int, float, np.floating))
                        and not isinstance(v, bool) else v)
                    for k, v in summ.items()},
        "coefficients": _records(sm.coefficient_table()),
        "anova": _records(sm.anova()),
        "alpha": float(alpha),
        "significant_terms": _records(sig),
        "n_significant": int(len(sig)),
        "component_ranking": _records(sc.table),
        "q_eff": int(sc.q_eff),
        "gp_loglik": float(sc.gp_loglik),
        "noise_level": float(sc.noise_level),
    }


# ----------------------------------------------------------------------
# Матрица влияний «компонент × свойство» — главный итог скрининга
# ----------------------------------------------------------------------
def influence_matrix(runner, properties: Optional[Sequence[str]] = None, *,
                     rel_threshold: float = 0.15, n_restarts: int = 6,
                     seed: Optional[int] = 0) -> pd.DataFrame:
    """ARD-важность компонентов на каждое свойство → матрица (компонент × свойство).

    Строки — компоненты смеси, столбцы — свойства; значение ∈ [0,1] (важность =
    нормированная обратная ARD-lengthscale, максимум по свойству = 1). Это главный
    итог скрининга «какой компонент на какое свойство влияет» одной таблицей.
    Свойства без измерений/с недостаточными данными пропускаются (столбец NaN)."""
    comps = list(runner.current_schema.mixture_names)
    props = list(properties) if properties is not None else list(
        runner.property_names)
    cols: Dict[str, np.ndarray] = {}
    for p in props:
        try:
            sc = component_importance(runner, p, rel_threshold=rel_threshold,
                                      n_restarts=n_restarts, seed=seed)
            imp = {c: float(v) for c, v in zip(sc.component_names, sc.importance)}
            cols[p] = np.array([imp.get(c, np.nan) for c in comps], float)
        except (ValueError, KeyError):
            cols[p] = np.full(len(comps), np.nan)
    return pd.DataFrame(cols, index=comps).round(4)


def screening_overview(runner, properties: Optional[Sequence[str]] = None, *,
                       rel_threshold: float = 0.15, n_restarts: int = 6,
                       seed: Optional[int] = 0) -> Dict[str, Any]:
    """Матрица влияний + R² по свойствам → JSON-сериализуемая сводка (UI/MCP)."""
    mat = influence_matrix(runner, properties, rel_threshold=rel_threshold,
                           n_restarts=n_restarts, seed=seed)
    r2: Dict[str, Any] = {}
    for p in list(mat.columns):
        try:
            r2[p] = float(fit_scheffe(runner, p).r2)
        except (ValueError, KeyError):
            r2[p] = None
    return {
        "components": list(mat.index),
        "properties": list(mat.columns),
        "importance": [[None if pd.isna(v) else float(v) for v in row]
                       for row in mat.to_numpy()],
        "r2": r2,
    }
