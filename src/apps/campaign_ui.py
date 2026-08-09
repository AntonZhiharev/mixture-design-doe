"""apps/campaign_ui.py — Streamlit-вкладка «Кампания» (ТЗ v1.1, §16/§16.1).

ШАГ 4: тонкий UI поверх :mod:`src.apps.campaign` (ШАГ 1–3). Канон «логика+тест,
потом UI» соблюдён: вся честность живёт в campaign-слое; здесь — только показ и
кнопки. Разделение:

  * ЧИСТЫЕ хелперы (``build_demo_campaign_runner``, ``role_table_dataframe``,
    ``spawn_review_dataframe``, ``goal_editor_dataframe``,
    ``workbench_points_dataframe``) НЕ зовут Streamlit — тестируются напрямую;
  * :func:`render_campaign` рисует вкладку через ``st`` (тест — headless AppTest).

UI работает с :class:`CampaignController` поверх ОТДЕЛЬНОГО
:class:`MixtureProcessRunner` (составная область mixture×process с ρ), который не
смешивается с pipeline-runner-ом M1–M8: это другая модель проекта (канон §5/§12,
одна модель физики на проект). Демо-оракул синтетический и детерминированный —
чтобы вкладку можно было запустить без реальной лаборатории.

A0.6: всё, что меняет состояние (смена роли, spawn, раунд, правка целей), делает
ТОЛЬКО явная кнопка пользователя; роли и денежный канал показываются read-only.
"""
from __future__ import annotations

import json
import math
import zlib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st


from ..core.schema import ModelSpec, ProjectSchema, VariableBlock
from ..core.simplex import parts_ranges_to_fraction_bounds
from ..optimize.desirability import (ChanceConstraint, DesirabilitySpec,
                                     hard_threshold_spec)
from ..apps.mixture_process_runner import MixtureProcessRunner

from ..apps import campaign as cv
from ..apps import campaign_screening as csx
from ..apps import campaign_state as cs

from ..design.branches import ROLE_OPTIMIZED, ROLE_PRICE_INPUT
from ..design.blocking import blocking_diagnostics
from ..design.levels import levels_caption
from ..design.linked_axes import links_caption
from ..design.phr_sampler import PhrSpec, premix_required




# ----------------------------------------------------------------------
# Синтетический детерминированный оракул демо-кампании (без лаборатории)
# ----------------------------------------------------------------------
_DEMO_PRICE = {"A": 95.0, "B": 200.0, "C": 23.0}   # ₽/кг состава (известны)


class _DemoOracle:
    """Оракул демо: 3 свойства от составных координат ``[A,B,C,T,P]``.

    ``rho`` (плотность) — полноценный отклик (множитель цены изделия, §3); цена
    состава детерминирована (:func:`demo_price_fn`). Линейно-разнообразные функции
    достаточно богаты, чтобы роли/каналы было видно, и дёшевы для GP-фита.
    """

    property_names = ["strength", "gloss", "rho"]

    def evaluate(self, Xc) -> np.ndarray:
        Xc = np.atleast_2d(np.asarray(Xc, float))
        A, B, C = Xc[:, 0], Xc[:, 1], Xc[:, 2]
        T = Xc[:, 3] if Xc.shape[1] > 3 else np.zeros(len(Xc))
        P = Xc[:, 4] if Xc.shape[1] > 4 else np.zeros(len(Xc))
        strength = 6.0 * A + 5.0 * B + 4.0 * C + 3.0 * T
        gloss = 3.0 * A + 6.0 * B + 5.0 * C + 4.0 * P
        rho = 0.8 * A + 1.0 * B + 1.4 * C
        return np.column_stack([strength, gloss, rho])


def demo_price_fn(Xc) -> np.ndarray:
    """Цена состава ₽/кг = доли·цены компонентов (детерминирована, без процесса)."""
    Xc = np.atleast_2d(np.asarray(Xc, float))
    w = np.array([_DEMO_PRICE["A"], _DEMO_PRICE["B"], _DEMO_PRICE["C"]], float)
    return Xc[:, :3] @ w


def build_demo_campaign_runner(*, seed: int = 7, n_seed: int = 14
                               ) -> MixtureProcessRunner:
    """Собрать демо-кампанию: runner + общий пул + две КОНТРАСТНЫЕ ветки.

    Контраст под И-5/Гр-1 (виден в UI сразу):
      * ``premium`` — ρ НЕ цель, питает цену ⇒ роль PRICE_INPUT (канал ALIVE);
      * ``rho_focus`` — ρ в цели (min) И питает цену ⇒ роль OPTIMIZED (канал
        ZEROED): денежная нога занулена, двойной счёт δρ убран.
    Обе ветки имеют экономику (V/c_exp/H), чтобы объяснение §16.1 показывало ₽.
    """
    mix = VariableBlock.mixture(["A", "B", "C"])
    proc = VariableBlock.process(["T", "P"], lower=[0.0, 0.0], upper=[1.0, 1.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    schema = ProjectSchema.mixture_process(mix, proc, model=model)
    runner = MixtureProcessRunner(schema, _DemoOracle(),
                                  baseline=[1 / 3, 1 / 3, 1 / 3, 0.5, 0.5],
                                  seed=int(seed), n_restarts=2)
    runner.seed_initial(n=int(n_seed), seed=int(seed))

    price_spec = DesirabilitySpec("min", low=0.0, high=300.0, weight=0.5)
    runner.add_branch(
        "premium", {"strength": DesirabilitySpec("max", low=2.0, high=12.0),
                    "gloss": DesirabilitySpec("max", low=1.0, high=13.0)},
        budget=20, satisfy_at=1.1, branch_id="premium")
    runner.set_branch_cost("premium", demo_price_fn, price_spec,
                           rho_property="rho")
    runner.add_branch(
        "rho_focus", {"strength": DesirabilitySpec("max", low=2.0, high=12.0),
                      "rho": DesirabilitySpec("min", low=0.5, high=1.5)},
        budget=20, satisfy_at=1.1, branch_id="rho_focus")
    runner.set_branch_cost("rho_focus", demo_price_fn, price_spec,
                           rho_property="rho")
    for bid in ("premium", "rho_focus"):
        b = runner.branches[bid]
        b.volume, b.cost_exp, b.horizon = 1.0e4, 1.0e-3, 100.0
    return runner


# ----------------------------------------------------------------------
# §17.4 (Ш3b) — РЕАЛЬНЫЙ сетап: ручной оракул + сборка раннера из формы
# ----------------------------------------------------------------------
class ManualOracle:
    """Оракул РУЧНОГО сетапа (§17.4): несёт имена свойств; истинные Y — от пользователя.

    Реальная лаборатория меряет ВНЕ системы, поэтому :meth:`evaluate` НЕ выдаёт
    себя за настоящую истину — это лишь ДЕТЕРМИНИРОВАННЫЙ демо-генератор для кнопки
    «Заполнить тестовыми» (прогоны без лаборатории). Коэффициенты стабильно
    выводятся из имени свойства (``crc32`` — воспроизводимо между процессами), но
    это синтетика: реальные отклики всегда вносит пользователь (``commit_seed`` /
    ``commit_measured``, A0.6). ``evaluate`` принимает ПОЛНЫЙ составной вектор
    ``Xc`` (n×dim) — как того требует контракт раннера.
    """

    def __init__(self, property_names: Sequence[str]):
        self.property_names: List[str] = list(property_names)

    def evaluate(self, Xc) -> np.ndarray:
        Xc = np.atleast_2d(np.asarray(Xc, float))
        n, dim = Xc.shape
        cols: List[np.ndarray] = []
        for name in self.property_names:
            rng = np.random.default_rng(zlib.crc32(str(name).encode("utf-8")))
            w = rng.uniform(0.5, 2.0, size=dim)
            b = float(rng.uniform(0.0, 1.0))
            cols.append(Xc @ w + b)
        return np.column_stack(cols) if cols else np.empty((n, 0), float)


def is_manual_campaign(runner) -> bool:
    """True, если истину кампании вносит пользователь (:class:`ManualOracle`).

    P0: для таких кампаний авто-оракульные действия (прогон раунда демо-оракулом)
    скрываются — ``ManualOracle.evaluate`` лишь синтетический демо-генератор, и
    авто-раунд молча записал бы в РЕАЛЬНУЮ базу выдуманные Y (A0.6). Загруженные
    с диска кампании тоже manual (``campaign_state`` реконструирует ManualOracle).
    Чистая (без Streamlit)."""
    return isinstance(getattr(runner, "oracle", None), ManualOracle)


def build_setup_runner(*, mixture_names: Sequence[str],
                       process_names: Sequence[str],
                       process_lower: Sequence[float],
                       process_upper: Sequence[float],
                       response_names: Sequence[str],
                       mixture_lower: Optional[Sequence[float]] = None,
                       mixture_upper: Optional[Sequence[float]] = None,
                       baseline: Optional[Sequence[float]] = None,
                       seed: int = 0, n_restarts: int = 2
                       ) -> MixtureProcessRunner:
    """§17.4: собрать ``MixtureProcessRunner`` РЕАЛЬНОГО сетапа (ручной оракул).

    Составная область — симплекс {mixture} × куб {process} СРАЗУ (процесс-параметры
    с самого старта, §17.4); отклики — имена пользователя. Оракул —
    :class:`ManualOracle` (Y вносится вручную через seed/branch-циклы §17.2/§17.4;
    ``evaluate`` — лишь демо-заполнение). База ПУСТА, суррогатов нет: стартовый
    дизайн предлагается (``propose_seed``) и меряется пользователем
    (``commit_seed``). ``baseline`` по умолчанию — равномерная смесь (1/q) +
    середина каждого процесс-интервала.
    """
    mixture_names = [str(s) for s in mixture_names]
    process_names = [str(s) for s in process_names]
    response_names = [str(s) for s in response_names]
    if not mixture_names:
        raise ValueError("Нужен хотя бы один компонент смеси.")
    if not process_names:
        raise ValueError("Нужен хотя бы один процесс-параметр "
                         "(§17.4: процесс-параметры задаются сразу).")
    if not response_names:
        raise ValueError("Нужен хотя бы один отклик (свойство).")
    pl = [float(v) for v in process_lower]
    pu = [float(v) for v in process_upper]
    if len(pl) != len(process_names) or len(pu) != len(process_names):
        raise ValueError("Число границ процесса не совпадает с числом параметров "
                         f"({len(process_names)}).")

    mix = VariableBlock.mixture(mixture_names, lower=mixture_lower,
                                upper=mixture_upper)
    proc = VariableBlock.process(process_names, lower=pl, upper=pu)
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    schema = ProjectSchema.mixture_process(mix, proc, model=model)
    oracle = ManualOracle(response_names)
    if baseline is None:
        q = len(mixture_names)
        baseline = [1.0 / q] * q + [(lo + hi) / 2.0 for lo, hi in zip(pl, pu)]
    return MixtureProcessRunner(schema, oracle, baseline=list(baseline),
                                seed=int(seed), n_restarts=int(n_restarts))


def setup_coord_names(runner) -> List[str]:
    """Имена составных координат ТЕКУЩЕЙ схемы: mixture-компоненты + process-оси."""
    sch = runner.current_schema
    return list(sch.mixture_names) + list(sch.process_names)


def process_code_to_real(runner, X):
    """Составной ``X`` (процесс в коде [0,1]) → копия с процесс-осями в РЕАЛЬНЫХ
    единицах (замечание 2). Mixture-доли остаются как есть (они уже физические).

    Раннер хранит точки в внутреннем коде [0,1] по каждой процесс-оси; для показа
    и Excel пользователю нужны абсолютные величины (T=150…200 °C, а не 0…1).
    Денормализация — покомпонентная :meth:`VariableBlock.from_code` процесс-блока
    ТЕКУЩЕЙ схемы (обратимо к нормировке движка). Чистая (без Streamlit)."""
    X = np.atleast_2d(np.asarray(X, float)).copy()
    pb = runner.current_schema.process_block()
    q = len(runner.current_schema.mixture_names)
    if pb is not None:
        d = len(pb.names)
        if X.shape[1] >= q + d and d > 0:
            for i in range(len(X)):
                X[i, q:q + d] = pb.from_code(X[i, q:q + d])
    return X



def make_linear_price_fn(prices: Sequence[float]):
    """§17.5: ЦЕНА СОСТАВА ₽/кг из цен компонентов — линейная нога ρ (Ш4).

    Возвращает callable ``Xc → цена состава`` = Σ(доля_i·цена_i) по mixture-долям
    (первые ``len(prices)`` координат составного вектора; процесс-оси на цену
    состава не влияют, как в демо). Детерминирована и чиста (без Streamlit) —
    тестируется напрямую. Используется как ``composition_price_fn`` в
    :meth:`CampaignController.create_branch` (item-цена = состав·ρ, §3/§15.6).
    Единый источник — :func:`campaign_state.linear_price_fn`: результат несёт
    сериализуемый дескриптор ``price_spec``, поэтому ценовая нога переживает
    save/load кампании (C2, §17.6.1).
    """
    return cs.linear_price_fn(prices)


# ----------------------------------------------------------------------
# §17.4 — научная терминология UI (замена жаргона; единый источник ярлыков)
# ----------------------------------------------------------------------
# «Ценовая нога» — внутренний жаргон; в UI показываем операционный термин.
COST_MODEL_LABEL = "Модель себестоимости изделия (плотность ρ → ₽/изд)"
# Единицы по умолчанию (замечание 7): валюта и масса выносятся в подписи/Excel.
CURRENCY_UNIT = "₽"
MASS_UNIT = "кг"


def recommended_seed_size(q: int, d: int) -> int:
    """§17.4 (замечание 4): рекомендуемый размер стартового (скрининг) дизайна.

    Универсальная формула от числа компонентов смеси ``q`` и числа
    процесс-параметров ``d``. Первый этап — СКРИНИНГ: оцениваем главные эффекты и
    смесь-процессные взаимодействия кросс-модели «смесь-линейно × процесс-линейно»,
    число членов которой ``P = q·(1 + d)``. Добавляем ~50 % запаса на остаточную
    дисперсию/lack-of-fit: ``N = P + ⌈P/2⌉``, но не меньше ``q + d + 1`` (иначе
    даже главные эффекты не разрешимы). Пример: q=3, d=2 → P=9 → N=14.
    """
    q = int(q)
    d = int(d)
    if q < 1:
        raise ValueError("Нужен хотя бы один компонент смеси (q ≥ 1).")
    if d < 0:
        raise ValueError("Число процесс-параметров не может быть отрицательным.")
    p_terms = q * (1 + d)
    n = p_terms + math.ceil(p_terms / 2)
    return int(max(n, q + d + 1))


def mixture_amounts_to_fractions(amounts: Sequence[float]) -> np.ndarray:
    """Части (любые ≥0) → доли состава (Σ=1) нормировкой по сумме (замечание 1).

    Программа умеет хранить состав и в ЧАСТЯХ, и в ДОЛЯХ и переводить одно в
    другое: доли = части / Σ(части). Пустой/отрицательный/нулевой ввод — явный
    отказ (A0.6: не нормируем молча мусор)."""
    a = np.asarray(amounts, float).ravel()
    if a.size == 0:
        raise ValueError("Ожидается непустой вектор количеств компонентов.")
    if np.any(a < 0):
        raise ValueError("Количества компонентов не могут быть отрицательными.")
    s = float(a.sum())
    if s <= 0:
        raise ValueError("Сумма количеств должна быть положительной.")
    return a / s


def resolve_mixture_bounds(n: int, lower_txt: str, upper_txt: str, *,
                           mode: str = "fractions"):
    """Границы состава в ДОЛЯХ [0,1] из ввода в долях ИЛИ частях (замечание 1).

    ``mode='fractions'`` — границы уже доли; ``mode='parts'`` — части, переводятся
    в доли через :func:`mixture_utils.convert_parts_to_proportions` (нормировка по
    сумме верхних границ). Если границы не заданы (пусто) — возвращает
    ``(None, None)`` (полный симплекс). Число границ обязано совпасть с ``n``.
    """
    lo = _parse_floats(lower_txt)
    hi = _parse_floats(upper_txt)
    if not lo and not hi:
        return None, None
    if lo is None or hi is None:
        raise ValueError("Границы состава — числа через запятую.")
    if len(lo) != int(n) or len(hi) != int(n):
        raise ValueError(f"Границ состава должно быть по {int(n)} "
                         "(по числу компонентов смеси).")
    if mode == "parts":
        from ..utils.mixture_utils import convert_parts_to_proportions
        pairs = convert_parts_to_proportions(list(zip(lo, hi)))
        lo = [float(p[0]) for p in pairs]
        hi = [float(p[1]) for p in pairs]
    return lo, hi


# ----------------------------------------------------------------------
# C3 (§17.6.1) — выгрузка ОБЩЕЙ базы кампании + расход сырья (доли↔части)
# ----------------------------------------------------------------------
def campaign_base_dataframe(runner, *, batch_kg: Optional[float] = None
                            ) -> pd.DataFrame:
    """§17.6.1 (C3): ОБЩАЯ база кампании → таблица для показа/Excel (read-only).

    По строке на опыт общей базы (``runner.points``, И-1): сквозной «№ опыта»,
    человекочитаемый origin, составные координаты (mixture-доли + процесс в
    РЕАЛЬНЫХ единицах) и измеренные отклики по всем P. При заданном ``batch_kg``
    добавляются столбцы расхода сырья ``{компонент} ({кг})`` = доля·batch —
    сколько взвесить на партию (замечание 7: единицы нужны лаборатории). Пустая
    база → пустой DataFrame.
    """
    coord_names = setup_coord_names(runner)
    props = list(runner.property_names)
    X = getattr(runner, "X", None)
    if X is None or len(X) == 0:
        return pd.DataFrame()
    X = np.atleast_2d(np.asarray(X, float))
    Y = np.atleast_2d(np.asarray(getattr(runner, "Y", np.empty((len(X), 0))),
                                 float))
    origins = list(getattr(runner, "origin", []) or [])
    # blocking: метки партий (блоков) активных точек — порядок совпадает с X
    try:
        blocks = list(runner.point_blocks())
    except Exception:  # noqa: BLE001 — блоки не критичны для показа базы
        blocks = []
    has_block_names = bool(getattr(runner, "block_names", {}) or {})
    mix_names = list(runner.current_schema.mixture_names)
    n_mix = len(mix_names)
    # Показ процесса в АБСОЛЮТНЫХ единицах (замечание 2): mixture-доли не трогаем,
    # процесс-оси денормализуем из внутреннего кода [0,1]. Расход сырья — по долям.
    Xreal = process_code_to_real(runner, X)

    # P3.1: ковариаты (телеметрия прогона) — столбцы базы, НЕ отклики модели;
    # выравнены с X (порядок активных точек, как point_blocks).
    cov_names = list(getattr(runner, "covariate_names", []) or [])
    covs: List[Dict[str, float]] = []
    if cov_names:
        try:
            covs = list(runner.active_point_covariates())
        except Exception:  # noqa: BLE001 — ковариаты не критичны для показа
            covs = []

    rows: List[Dict[str, Any]] = []
    for i in range(len(X)):
        row: Dict[str, Any] = {"№ опыта": i + 1}
        og = origins[i] if i < len(origins) else ""
        row["источник"] = origin_label(runner, og)
        row["Блок"] = int(blocks[i]) if i < len(blocks) else 1
        if has_block_names:
            row["Партия"] = block_display(runner, row["Блок"])
        for j, cn in enumerate(coord_names[:X.shape[1]]):
            row[cn] = round(float(Xreal[i, j]), 4)

        if batch_kg is not None and float(batch_kg) > 0:
            for j, cn in enumerate(mix_names):
                row[f"{cn} ({MASS_UNIT})"] = round(
                    float(X[i, j]) * float(batch_kg), 4)
        for k, pn in enumerate(props):
            row[f"{pn} (изм.)"] = (round(float(Y[i, k]), 4)
                                   if k < Y.shape[1] else np.nan)
        for cn in cov_names:
            v = (covs[i].get(cn) if i < len(covs) else None)
            row[f"{cn} (ковариата)"] = (round(float(v), 4)
                                        if v is not None else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def measured_responses_editor_df(runner) -> pd.DataFrame:
    """§17.2.1: общая база → таблица-редактор ИЗМЕРЕННЫХ откликов (правка ошибок ввода).

    Строка на точку ``runner.points`` В ПОРЯДКЕ базы (индекс строки = «№ опыта» − 1
    = ``point_index`` для :meth:`MixtureProcessRunner.correct_measured`): сквозной
    номер, человекочитаемый источник и ТЕКУЩЕЕ измеренное значение каждого свойства
    (по ``property_names``). Редактируются только столбцы-отклики; координаты и
    происхождение неизменны (И-1). Чистый хелпер (без Streamlit) — тестируется
    напрямую и служит эталоном сравнения «старое↔новое» для коррекции."""
    props = list(runner.property_names)
    rows: List[Dict[str, Any]] = []
    for i, p in enumerate(getattr(runner, "points", []) or []):
        y = getattr(p, "Y", {}) or {}
        og = (p.origin_tag.get("origin", "seed")
              if getattr(p, "origin_tag", None) else "seed")
        row: Dict[str, Any] = {"№ опыта": i + 1,
                               "источник": origin_label(runner, og)}
        for pn in props:
            val = y.get(pn, None)
            try:
                row[pn] = float(val) if val is not None else None
            except (TypeError, ValueError):
                row[pn] = None
        rows.append(row)
    return pd.DataFrame(rows)


def covariates_editor_df(runner) -> pd.DataFrame:
    """P3.1: общая база → таблица-редактор КОВАРИАТ (телеметрии прогона).

    Строка на точку ``runner.points`` В ПОРЯДКЕ базы (индекс строки =
    «№ опыта» − 1 = ``point_index`` для
    :meth:`MixtureProcessRunner.set_point_covariates`): сквозной номер,
    человекочитаемый источник и текущее значение каждой ОБЪЯВЛЕННОЙ ковариаты
    (``None`` = телеметрия не снята — честная пустая ячейка, не 0.0).
    Двойник :func:`measured_responses_editor_df` для ковариат: редактируются
    только столбцы-ковариаты; координаты и Y не трогаются (И-1). Чистая
    (без Streamlit) — тестируется напрямую."""
    cov_names = list(getattr(runner, "covariate_names", []) or [])
    covs = (runner.point_covariates()
            if hasattr(runner, "point_covariates") else [])
    rows: List[Dict[str, Any]] = []
    for i, p in enumerate(getattr(runner, "points", []) or []):
        og = (p.origin_tag.get("origin", "seed")
              if getattr(p, "origin_tag", None) else "seed")
        row: Dict[str, Any] = {"№ опыта": i + 1,
                               "источник": origin_label(runner, og)}
        vals = covs[i] if i < len(covs) else {}
        for cn in cov_names:
            v = vals.get(cn)
            row[cn] = float(v) if v is not None else None
        rows.append(row)
    return pd.DataFrame(rows)


def covariate_rows_from_editor(edited, names: Sequence[str],
                               *, suffix: str = " (ковариата)"
                               ) -> List[Dict[str, float]]:
    """P3.1: собрать per-point строки ковариат из таблицы-редактора (чистая).

    ``edited`` — DataFrame редактора (seed/workbench) со столбцами
    ``{имя}{suffix}``; NaN/пустые ячейки ПРОПУСКАЮТСЯ (телеметрия не снята —
    допустимо, в отличие от откликов «(lab)», где пустых быть не может).
    Возвращает список словарей длиной в число строк — вход ``covariates=``
    для ``commit_seed``/``commit_measured``. Валидацию имён/чисел делает
    ШТАТНЫЙ раннер (канон iter52: правила в UI не дублируются)."""
    out: List[Dict[str, float]] = []
    for _, row in edited.iterrows():
        vals: Dict[str, float] = {}
        for nm in names:
            col = f"{nm}{suffix}"
            if col not in edited.columns:
                continue
            v = row[col]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv):
                vals[nm] = fv
        out.append(vals)
    return out


def campaign_base_excel_bytes(runner, *, batch_kg: Optional[float] = None
                              ) -> bytes:
    """§17.6.1 (C3): ОБЩАЯ база кампании → xlsx-байты (для кнопки скачивания).

    Лист «База опытов» = :func:`campaign_base_dataframe` (с расходом сырья, если
    задан ``batch_kg``). Чистый хелпер (без Streamlit) — тестируется напрямую;
    возвращает готовые к отдаче байты .xlsx (openpyxl)."""
    import io
    df = campaign_base_dataframe(runner, batch_kg=batch_kg)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        (df if not df.empty else pd.DataFrame({"инфо": ["база пуста"]})).to_excel(
            xw, sheet_name="База опытов", index=False)
    buf.seek(0)
    return buf.getvalue()


def seed_design_dataframe(runner, Xs, Ys=None, *, batch_kg: Optional[float] = None
                          ) -> pd.DataFrame:
    """§17.4 (C3): предложенный СТАРТОВЫЙ дизайн → таблица для показа/Excel.

    По строке на предложенную seed-точку: будущий «№ опыта» (база пока пуста ⇒
    1…N), заблокированные составные координаты (mixture-доли + процесс в РЕАЛЬНЫХ
    единицах) и столбцы откликов ``{свойство} (lab)`` (пустые — места под ручной
    ввод в лаборатории, либо уже внесённые ``Ys``). При заданном ``batch_kg``
    добавляются столбцы расхода сырья ``{компонент} ({кг})`` = доля·batch —
    сколько взвесить на опыт (замечание 7). Чистый хелпер — тестируется напрямую.
    """
    coord_names = setup_coord_names(runner)
    props = list(runner.property_names)
    cov_names_seed = list(getattr(runner, "covariate_names", []) or [])
    Xs = np.atleast_2d(np.asarray(Xs, float))
    ncoord = Xs.shape[1]
    mix_names = list(runner.current_schema.mixture_names)
    Ya = np.atleast_2d(np.asarray(Ys, float)) if Ys is not None else None
    # Показ процесса в АБСОЛЮТНЫХ единицах (замечание 2): mixture-доли не трогаем,
    # процесс-оси денормализуем из внутреннего кода [0,1]. Расход сырья считаем по
    # ДОЛЯМ (mixture), поэтому берём их из исходного Xs (real == code для mixture).
    Xreal = process_code_to_real(runner, Xs)

    # blocking: оптимальные метки партий стартового дизайна (детерминированы по
    # seed раннера ⇒ при commit_seed точки получат ЭТИ ЖЕ метки)
    seed_blocks = None
    if int(getattr(runner, "n_blocks_start", 1)) > 1 and len(Xs):
        try:
            seed_blocks = np.asarray(runner.seed_block_labels(Xs), int)
        except Exception:  # noqa: BLE001 — блоки не критичны для показа
            seed_blocks = None

    has_block_names = bool(getattr(runner, "block_names", {}) or {})
    nums = list(experiment_index(len(runner.points), len(Xs)))
    rows: List[Dict[str, Any]] = []
    for i in range(len(Xs)):
        row: Dict[str, Any] = {"№ опыта": nums[i]}
        if seed_blocks is not None:
            row["Блок"] = int(seed_blocks[i])
            if has_block_names:
                row["Партия"] = block_display(runner, int(seed_blocks[i]))
        for j, cn in enumerate(coord_names[:ncoord]):
            row[cn] = round(float(Xreal[i, j]), 4)
        if batch_kg is not None and float(batch_kg) > 0:
            for j, cn in enumerate(mix_names):
                row[f"{cn} ({MASS_UNIT})"] = round(
                    float(Xs[i, j]) * float(batch_kg), 4)
        for k, pn in enumerate(props):
            row[f"{pn} (lab)"] = (round(float(Ya[i, k]), 4)
                                  if Ya is not None and k < Ya.shape[1]
                                  else np.nan)
        # P3.1: места под ТЕЛЕМЕТРИЮ прогона (объявленные ковариаты) —
        # заполняются при измерении; пустые ячейки допустимы (не отклик).
        for cn in cov_names_seed:
            row[f"{cn} (ковариата)"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)



def block_display(runner, b: int) -> str:
    """Отображаемое имя блока: пользовательское имя или номер (чистая).

    Имена блоков (:attr:`MixtureProcessRunner.block_names`) — метаданные показа
    (оператор / партия сырья / смена): помогают понять ПРИНЦИП блокировки, на
    математику не влияют. Нет имени → строка с номером."""
    names = getattr(runner, "block_names", {}) or {}
    nm = str(names.get(int(b), "")).strip()
    return nm if nm else str(int(b))


def seed_blocking_caption(runner, Xs) -> str:
    """Iteration 28+: ВИДИМАЯ сводка блокировки стартового дизайна (подпись).

    Blocking был «невидим»: при ``n_blocks_start == 1`` (дефолт) таблица
    намеренно не несёт столбца «Блок», и в UI не оставалось НИ ОДНОГО
    упоминания блокировки — пользователь не знал, что она есть и как её
    включить. Возвращает строку-подпись под seed-таблицей:

      * 1 партия  — подсказка, что blocking выключен и как включить;
      * nb > 1    — размеры партий и цена блокировки (потеря D-информации
        модельных термов, :func:`design.blocking.blocking_diagnostics`).

    Чистая (без Streamlit) — тестируется напрямую.
    """
    nb = max(1, int(getattr(runner, "n_blocks_start", 1)))
    Xs = np.atleast_2d(np.asarray(Xs, float))
    if nb <= 1 or len(Xs) == 0:
        return ("🧱 Блокировка выключена: весь стартовый план — ОДНА партия "
                "(столбца «Блок» нет). Если опыты нельзя поставить одной "
                "партией / за один день — увеличьте «Партий (блоков)» выше: "
                "план оптимально разобьётся на партии и появится столбец "
                "«Блок».")
    factor = str(getattr(runner, "block_factor", "") or "").strip()
    factor_txt = f" Фактор блокировки: {factor}." if factor else ""
    try:
        lab = np.asarray(runner.seed_block_labels(Xs), int)
        q = len(runner.current_schema.mixture_names)
        diag = blocking_diagnostics(
            Xs[:, :q], lab, model=getattr(runner, "gp_mean_model", "quadratic"))
        sizes = ", ".join(
            f"блок {b} «{block_display(runner, b)}»: {c} оп."
            if block_display(runner, b) != str(b) else f"блок {b}: {c} оп."
            for b, c in sorted(diag["block_sizes"].items()))
        return (f"🧱 Блокировка включена: {nb} партий ({sizes}); потеря "
                f"D-информации модельных термов ≈ {diag['d_loss_pct']:.1f}% "
                "(метки — в столбце «Блок»; показ = фиксация: при "
                f"commit_seed точки получат ЭТИ ЖЕ блоки).{factor_txt}")
    except Exception:  # noqa: BLE001 — подпись не должна ломать seed-цикл
        return (f"🧱 Блокировка включена: {nb} партий — метки в столбце "
                "«Блок».")


def base_blocking_caption(runner) -> str:
    """Iteration 28+: сводка блокировки ОБЩЕЙ базы кампании (подпись).

    Показывается под таблицей общей базы: сколько партий (блоков), их размеры
    и цена блокировки по :meth:`MixtureProcessRunner.blocking_summary`.
    Одна партия / пустая база → пустая строка (подпись не показывается).
    Чистая (без Streamlit) — тестируется напрямую."""
    try:
        bs = runner.blocking_summary()
        if int(bs.get("n_blocks", 0)) <= 1:
            return ""
        sizes = ", ".join(
            f"блок {b} «{block_display(runner, b)}»: {c} оп."
            if block_display(runner, b) != str(b) else f"блок {b}: {c} оп."
            for b, c in sorted(bs["block_sizes"].items()))
        loss = bs.get("d_loss_pct")
        loss_txt = (f"; потеря D-информации ≈ {float(loss):.1f}%"
                    if loss is not None and np.isfinite(float(loss)) else "")
        factor = str(getattr(runner, "block_factor", "") or "").strip()
        factor_txt = f" Фактор блокировки: {factor}." if factor else ""
        return (f"🧱 Партии (блоки) базы: {bs['n_blocks']} ({sizes}){loss_txt}. "
                f"Каждый добор ветки автоматически получает НОВЫЙ блок.{factor_txt}")
    except Exception:  # noqa: BLE001 — подпись не должна ломать показ базы
        return ""


def seed_preflight_caption(report) -> str:
    """iter32: однострочная сводка preflight-диагностики seed-плана (подпись).

    ``report`` — :class:`src.design.preflight.PreflightReport` (гейты
    ОТНОСИТЕЛЬНЫЕ к reference-пулу той же области, см. модуль). Зелёная строка —
    план информативен; жёлтая — перечисляет причины провала. Read-only и БЕЗ
    блокировки (A0.6): решение фиксировать план остаётся за пользователем.
    Чистая (без Streamlit) — тестируется напрямую."""
    if report.passed:
        return (f"🔎 Preflight: план информативен (rank {report.rank}/"
                f"{report.rank_ref}, cond ×{report.cond / max(report.cond_ref, 1e-12):.1f} "
                f"от reference-пула области).")
    return "⚠️ Preflight: " + "; ".join(report.failures)


def preflight_details_dataframe(report) -> pd.DataFrame:
    """iter32: таблица проверок preflight для показа (чистая, без Streamlit).

    Строки — :meth:`PreflightReport.rows` (проверка / значение плана / допуск /
    ОК); допуски относительные к reference-пулу области (см. design/preflight).
    """
    return pd.DataFrame(report.rows())


def seed_design_excel_bytes(runner, Xs, Ys=None, *,
                            batch_kg: Optional[float] = None,
                            spec: Optional[PhrSpec] = None,
                            delta_phr: Optional[float] = None,
                            grams_per_phr: Optional[float] = None) -> bytes:
    """§17.4 (C3): предложенный стартовый дизайн → xlsx-байты (кнопка скачивания).

    Лист «Стартовый дизайн» = :func:`seed_design_dataframe` (с расходом сырья, если
    задан ``batch_kg``; пустые «(lab)» — места под ручной ввод откликов). Чистый
    хелпер (без Streamlit) — тестируется напрямую; отдаёт готовые байты .xlsx.

    iter42.4: при активной phr-спеке и заданном ``delta_phr`` добавляется ЛИСТ
    «Навеска» (:func:`seed_weighing_dataframe`) — phr nominal/actual, граммы,
    премикс и нарушения по каждому опыту. Отдельный лист, а не 3·q колонок
    в основном: при q≈19 широкая таблица нечитаема."""
    import io
    df = seed_design_dataframe(runner, Xs, Ys, batch_kg=batch_kg)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        (df if not df.empty else pd.DataFrame({"инфо": ["дизайн пуст"]})).to_excel(
            xw, sheet_name="Стартовый дизайн", index=False)
        if spec is not None and delta_phr is not None:
            wdf = seed_weighing_dataframe(runner, spec, Xs, float(delta_phr),
                                          grams_per_phr=grams_per_phr)
            if not wdf.empty:
                wdf.to_excel(xw, sheet_name="Навеска", index=False)
    buf.seek(0)
    return buf.getvalue()


def branch_recipe_dataframe(runner, branch_id, *, batch_kg: Optional[float] = None,
                            n_candidates: int = 2000, refine_iters: int = 200,
                            n_starts: int = 5) -> pd.DataFrame:
    """§17.6.1 (C3): рекомендованный РЕЦЕПТ ветки x* → одна строка (показ/Excel).

    Запускает M8-argmax (:meth:`MixtureProcessRunner.optimize_xbest`) по ОБЩИМ
    суррогатам (GP, :class:`GPExpert`): максимум desirability целей ветки над
    составной областью.
    Строка несёт: имя ветки, состав-доли + процесс в РЕАЛЬНЫХ единицах (замечание
    2), предсказанные свойства целей ``{свойство} (прогноз)``, итог ``d_overall`` и
    по-целевые ``d[{свойство}]``; при ``batch_kg`` — расход сырья на пробу. Чистый
    хелпер (без Streamlit) — тестируется напрямую."""
    res = runner.optimize_xbest(branch_id, n_candidates=int(n_candidates),
                                refine_iters=int(refine_iters),
                                n_starts=int(n_starts))
    return _recipe_row_dataframe(runner, branch_id, res, batch_kg=batch_kg)


def _recipe_row_dataframe(runner, branch_id, res, *,
                          batch_kg: Optional[float] = None) -> pd.DataFrame:
    """iter43: строка рецепта из ГОТОВОГО ``DesirabilityResult`` (общая часть).

    Выделено из :func:`branch_recipe_dataframe`, чтобы
    :func:`branch_recipe_with_binding` собирал ту же таблицу БЕЗ повторного
    прогона M8-argmax (иначе binding-отчёт стоил бы второй оптимизации)."""
    coord_names = setup_coord_names(runner)
    mix_names = list(runner.current_schema.mixture_names)
    x = np.atleast_2d(np.asarray(res.x, float))
    xr = process_code_to_real(runner, x)[0]  # процесс → реальные единицы
    row: Dict[str, Any] = {"ветка": runner.branches[branch_id].name}
    for j, cn in enumerate(coord_names[:x.shape[1]]):
        row[cn] = round(float(xr[j]), 4)
    if batch_kg is not None and float(batch_kg) > 0:
        for j, cn in enumerate(mix_names):
            row[f"{cn} ({MASS_UNIT})"] = round(float(x[0, j]) * float(batch_kg), 4)
    for pn, val in (res.properties or {}).items():
        row[f"{pn} (прогноз)"] = round(float(val), 4)
    row["d_overall"] = round(float(res.d_overall), 4)
    for pn, dv in (res.d_individual or {}).items():
        row[f"d[{pn}]"] = round(float(dv), 4)
    return pd.DataFrame([row])



def branch_recipe_with_binding(runner, branch_id, *,
                               batch_kg: Optional[float] = None,
                               n_candidates: int = 2000,
                               refine_iters: int = 200, n_starts: int = 5):
    """iter43.3 (§43.3): рецепт ветки x* + ``binding_report`` ОДНИМ прогоном.

    :func:`branch_recipe_dataframe` возвращает только таблицу (её сигнатура и
    Excel-выгрузка не меняются), а binding-отчёт нужен рядом — иначе пришлось бы
    гонять M8-argmax дважды. Возвращает ``(df, binding_report)``; отчёт —
    ``DesirabilityResult.binding_report`` (veto-статистика целей + chance).
    Чистая (без Streamlit)."""
    res = runner.optimize_xbest(branch_id, n_candidates=int(n_candidates),
                               refine_iters=int(refine_iters),
                               n_starts=int(n_starts))
    df = _recipe_row_dataframe(runner, branch_id, res, batch_kg=batch_kg)
    return df, dict(res.binding_report or {})


def binding_report_dataframe(report: Mapping[str, Any]) -> pd.DataFrame:
    """iter43.3 (§43.3): ``binding_report`` → таблица «что связывает оптимум».

    Строки — ограничения ДВУХ типов (колонка «тип»):

    * ``veto (цель)`` — desirability-цель: ``% пула`` = доля точек глобального
      пула с ``d_i = 0`` (порог допустимости), «в x*» = ``d_i`` в оптимуме;
    * ``вероятностное Pr`` — chance-ограничение: ``% пула`` = доля точек с
      ``Pr < 1−α``, «в x*» = вероятность в оптимуме, «порог» = ``1−α``.

    «выполнено в x*»: для цели — ``d_i > 0``; для chance — ``Pr ≥ 1−α`` (флаг
    ядра ``satisfied_at_optimum``). Чистая (без Streamlit)."""
    rows: List[Dict[str, Any]] = []
    for name, st_ in (report.get("specs") or {}).items():
        d_at = float(st_.get("d_at_optimum", 0.0))
        rows.append({
            "ограничение": name,
            "тип": "veto (цель)",
            "% пула под биндингом": round(100.0 * float(
                st_.get("frac_veto", 0.0)), 1),
            "в x*": round(d_at, 4),
            "порог": "d > 0",
            "выполнено в x*": "да" if d_at > 0.0 else "нет",
        })
    for name, ch in (report.get("chance") or {}).items():
        alpha = float(ch.get("alpha", 0.05))
        rows.append({
            "ограничение": name,
            "тип": "вероятностное Pr",
            "% пула под биндингом": round(100.0 * float(
                ch.get("frac_below", 0.0)), 1),
            "в x*": round(float(ch.get("prob_at_optimum", 0.0)), 4),
            "порог": round(1.0 - alpha, 4),
            "выполнено в x*": ("да" if ch.get("satisfied_at_optimum")
                               else "нет"),
        })
    return pd.DataFrame(rows)


def binding_report_caption(report: Mapping[str, Any]) -> str:
    """iter43.3 (§43.3): подпись «оптимум не найден» vs «оптимум ЗАПРЕЩЁН».

    CAMPAIGN_SPEC_PVC §7 требует различать две принципиально разные ситуации:

    * **оптимум ЗАПРЕЩЁН** — ограничение нарушено на ВСЁМ пуле (100 % точек под
      биндингом): допустимой области в этой геометрии нет — ослаблять
      ограничение или расширять область, добор точек не поможет;
    * **оптимум НЕ НАЙДЕН** — допустимые точки в пуле есть, но x* их не достиг:
      это вопрос поиска (n_candidates / refine / мультистарт), не постановки.

    Если все ограничения выполнены в x* — зелёная строка. Чистая (без
    Streamlit)."""
    n_pool = int(report.get("n_pool", 0) or 0)
    unmet: List[str] = []
    forbidden: List[str] = []
    for name, st_ in (report.get("specs") or {}).items():
        if float(st_.get("d_at_optimum", 0.0)) <= 0.0:
            unmet.append(name)
        if float(st_.get("frac_veto", 0.0)) >= 1.0 - 1e-12:
            forbidden.append(name)
    for name, ch in (report.get("chance") or {}).items():
        if not ch.get("satisfied_at_optimum", True):
            unmet.append(name)
        if float(ch.get("frac_below", 0.0)) >= 1.0 - 1e-12:
            forbidden.append(name)
    if forbidden:
        return (f"⛔ Оптимум ЗАПРЕЩЁН: {', '.join(sorted(set(forbidden)))} "
                f"нарушено во ВСЁМ пуле ({n_pool} точек) — допустимой области "
                "в этой геометрии нет. Ослабьте ограничение или расширьте "
                "область; добор точек не поможет (CAMPAIGN_SPEC_PVC §7).")
    if unmet:
        return (f"⚠️ Оптимум НЕ НАЙДЕН: {', '.join(sorted(set(unmet)))} не "
                f"выполнено в x*, но допустимые точки в пуле есть ({n_pool}) — "
                "это вопрос поиска: увеличьте пул кандидатов / итерации "
                "уточнения / число стартов (CAMPAIGN_SPEC_PVC §7).")
    return (f"✅ Все ограничения выполнены в x* (пул {n_pool} точек): оптимум "
            "лежит в допустимой области.")


def branch_recipe_excel_bytes(runner, branch_id, *,
                              batch_kg: Optional[float] = None,

                              n_candidates: int = 2000, refine_iters: int = 200,
                              n_starts: int = 5) -> bytes:
    """§17.6.1 (C3): рекомендованный рецепт ветки → xlsx-байты (кнопка скачивания).

    Лист «Рецепт» = :func:`branch_recipe_dataframe`. Чистый хелпер (без Streamlit)
    — тестируется напрямую; отдаёт готовые байты .xlsx (openpyxl)."""
    import io
    df = branch_recipe_dataframe(runner, branch_id, batch_kg=batch_kg,
                                 n_candidates=n_candidates,
                                 refine_iters=refine_iters, n_starts=n_starts)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="Рецепт", index=False)
    buf.seek(0)
    return buf.getvalue()



# ----------------------------------------------------------------------
# Чистые таблицы для показа (без Streamlit) — тестируемы напрямую
# ----------------------------------------------------------------------



_MONEY_RU = {cv.MONEY_ZEROED: "занулён (ZEROED)", cv.MONEY_ALIVE: "живой (ALIVE)",
             None: "—"}
_CHANGE_RU = {"inherited": "унаследовано как есть",
              "overridden_same_role": "тронуто, роль та же",
              "changed_by_objective": "изменено объективом ветки"}

# Легенда §4-стопа: причина остановки раунда (двойной критерий §4/§6).
_STOP_RU: Dict[Optional[str], str] = {
    None: "▶ продолжать (есть куда и выгодно)",
    "ceil_reached": "🎯 потолок достигнут (ceil_reached)",
    "stagnation": "🛑 прогресс встал (stagnation)",
    "not_economical": "💸 невыгодно (not_economical)",
}

# Подсказка к полю «Значимость цели» (ядровый вес взвеш. геом-среднего d_i).
# Только текст UI — сам параметр в ядре остаётся ``DesirabilitySpec.weight``.
_WEIGHT_HELP = (
    "Значимость цели — её относительный приоритет в компромиссе между откликами. "
    "Итог ветки = взвешенное геометрическое среднее желательностей d_i: "
    "d_overall = (Π d_i^w)^(1/Σw). Больше значимость → оптимум сильнее тянется "
    "выполнить именно эту цель, жертвуя другими. Важны только ОТНОШЕНИЯ "
    "значимостей (равные = равная важность; умножить все на число — без эффекта). "
    "Не спасает от veto: если хоть одна d_i=0, итог=0 при любой значимости."
)



def role_table_dataframe(report: Dict[str, Any]) -> pd.DataFrame:
    """Role-репорт ветки → таблица для показа (контекст ветки уже зашит в report)."""
    rows = []
    for r in report["responses"]:
        rows.append({
            "отклик": r["response"],
            "роль": r["role_label"],
            "код": r["role_code"],
            "в цели": "да" if r["in_goal"] else "—",
            # Колонка строго строковая: у целей — число, у нецелевых — «—».
            # Смешение float/str ломает Arrow-сериализацию st.dataframe (варнинг
            # «Serialization … unsuccessful» и авто-приведение типа колонки «вес»).
            "вес": (f'{float(r["weight"]):g}' if r["weight"] is not None else "—"),
            "ден. канал ρ": _MONEY_RU.get(r["money_channel"], "—"),

            "покрытие": f'{r["coverage_measured"]}/{r["coverage_total"]}',
            "низк. покрытие": "⚠️" if r["low_coverage"] else "",
        })
    return pd.DataFrame(rows)


def spawn_review_dataframe(review: Dict[str, Any]) -> pd.DataFrame:
    """Review-сводка наследования ролей при spawn → таблица (Тр-8.1а/б/в)."""
    rows = []
    for r in review["responses"]:
        rows.append({
            "отклик": r["response"],
            "роль родителя": r["role_parent"],
            "роль ребёнка": r["role_child"],
            "изменение": _CHANGE_RU.get(r["change"], r["change"]),
            "ден. канал ρ (ребёнок)": _MONEY_RU.get(r["money_channel_child"], "—"),
        })
    return pd.DataFrame(rows)


def goal_editor_dataframe(runner, branch_id: str) -> pd.DataFrame:
    """Текущие цели ветки → таблица (§16.3): отклик, вид, диапазон, target, вес.

    Ветка — это НАБОР целей (мультицель): каждая цель несёт свой вид
    (``min``/``max``/``target``/``target_range``), диапазон ``[low, high]``
    (и ``target`` для target-типа; для плато P2.2 в колонке target показан
    диапазон ``[target_low, target_high]``) и вес геом-среднего. Читает
    ``branch.goal`` (read-only)."""
    br = runner.branches[branch_id]
    rows = []
    for resp, spec in (br.goal or {}).items():
        if spec.kind == "target_range":
            # P2.2: плато-таргет — в target-колонке диапазон, не точка
            tgt_show = (f"[{round(float(spec.target_low), 4)}, "
                        f"{round(float(spec.target_high), 4)}] (плато)")
        elif spec.target is not None:
            tgt_show = round(float(spec.target), 4)
        else:
            tgt_show = "—"
        rows.append({
            "цель (отклик)": resp,
            "вид": spec.kind,
            "low": round(float(spec.low), 4),
            "high": round(float(spec.high), 4),
            "target": tgt_show,
            "вес": round(float(spec.weight), 4),
        })
    return pd.DataFrame(rows)


def workbench_points_dataframe(runner, result: Dict[str, Any]) -> pd.DataFrame:
    """Долитые за раунд точки → таблица измеренных откликов (§16.4).

    ``result`` — выхлоп ``run_branch_round`` (через ``CampaignController.run_round``):
    берём ``y_new`` (n×P, порядок ``property_names``) и помечаем origin-тегом ветки
    (И-1: точки уже в общей базе, здесь только показ)."""
    y = np.atleast_2d(np.asarray(result.get("y_new"), float))
    if y.size == 0:
        return pd.DataFrame()
    cols = list(runner.property_names)
    df = pd.DataFrame(y[:, [runner.prop_index[c] for c in cols]], columns=cols)
    df.insert(0, "origin", f"branch:{result.get('branch')}")
    # blocking добора: только что залитые точки — последние в базе, их партия =
    # последний (максимальный) блок общей базы
    try:
        bl = list(runner.point_blocks())
        if len(bl) >= len(df):
            df.insert(1, "Блок", [int(b) for b in bl[-len(df):]])
            if getattr(runner, "block_names", {}) or {}:
                df.insert(2, "Партия", [block_display(runner, int(b))
                                        for b in bl[-len(df):]])
    except Exception:  # noqa: BLE001 — блок не критичен для показа
        pass
    return df


def experiment_index(base_count: int, n: int) -> pd.Index:
    """Сквозные 1-based номера опытов проекта для ``n`` новых точек.

    «Номер опыта» = позиция точки в ЕДИНОЙ общей базе (``runner.points``),
    куда опыты добавляются по порядку (seed, затем commit'ы веток). Для ещё
    не залитых предложенных точек это их БУДУЩИЕ номера: если в базе уже
    ``base_count`` опытов, новые получат ``base_count+1 … base_count+n``.
    Возвращает именованный (``«№ опыта»``) индекс для показа в таблицах.
    """
    start = int(base_count) + 1
    return pd.Index(range(start, start + int(n)), name="№ опыта")


def origin_label(runner, origin: str) -> str:
    """Человекочитаемый ярлык origin-тега для ПОКАЗА (сам тег не меняется).

    Origin-тег точки в общей базе каноничен и завязан на id ветки
    (``"seed"`` / ``"M2"`` / ``"branch:{id}"``, И-1): id стабилен, а имя ветки
    можно переименовать, поэтому в ДАННЫХ хранится id. Для таблиц UI подменяем
    ``"branch:{id}"`` на ``"{имя} ({id})"``; прочие теги — как есть.
    """
    if isinstance(origin, str) and origin.startswith("branch:"):
        bid = origin.split(":", 1)[1]
        br = runner.branches.get(bid)
        if br is not None:
            return f"{br.name} ({bid})"
    return str(origin)



# ----------------------------------------------------------------------
# Streamlit-рендер вкладки (тест — headless AppTest)
# ----------------------------------------------------------------------

def get_campaign_controller() -> Optional["cv.CampaignController"]:
    """Контроллер демо-кампании из session_state (или ``None``, если не создан)."""
    return st.session_state.get("campaign_ctrl")


def publish_ui_focus(section: str, **fields) -> None:
    """iter65 (ASSISTANT_SPEC): сообщить доку ассистента, ГДЕ сейчас пользователь.

    Док стоит в правой колонке и виден на каждом шаге; чтобы вопрос «объясни
    эту ось» имел смысл, ему нужен ФАКТ о месте — шаг потока, узел спеки,
    выбранная ветка. Пишем обычный словарь в ``st.session_state['ui_focus']``:
    поток не должен импортировать слой ассистента ради одной записи, а разбор
    словаря — чистая функция ``assistant.context.focus_from_state`` (её и
    проверяет тест).

    Страница показывает несколько секций сразу, поэтому фокусом объявляется
    САМАЯ КОНКРЕТНАЯ из активных: пока база пуста — стартовый дизайн, дальше —
    выбранная ветка. Ручной селектор узла в доке лишь уточняет этот факт.
    """
    focus = {"section": str(section)}
    focus.update({k: v for k, v in fields.items()
                  if v not in (None, "", [], ())})
    st.session_state["ui_focus"] = focus


def _flash(msg: str, kind: str = "success") -> None:
    """P0: отложенное уведомление — переживает ``st.rerun`` мутации.

    Мутации состояния завершаются ``st.rerun()`` (иначе таблицы ВЫШЕ кнопки
    показывали устаревшее состояние до следующего клика), а rerun стирает вывод
    текущего прогона. Сообщение складывается в session_state и показывается в
    начале следующего прогона (:func:`_show_flashes`)."""
    st.session_state.setdefault("camp_flash", []).append((str(kind), str(msg)))


def _show_flashes() -> None:
    """Показать и очистить отложенные уведомления (:func:`_flash`)."""
    for kind, msg in st.session_state.pop("camp_flash", []):
        getattr(st, kind, st.info)(msg)


def _invalidate_branch_caches() -> None:
    """P0: сбросить кеши дорогих расчётов после мутаций состояния.

    Кеши результатов (₽-объяснение канала ρ, рецепт x*) живут под ключами
    ``cache_*`` в session_state; после смены целей/весов/ролей/точек они
    устаревают и должны пересчитываться по кнопке заново."""
    for k in [k for k in st.session_state if str(k).startswith("cache_")]:
        st.session_state.pop(k, None)


def campaign_assistant_overview(
        ctrl: Optional["cv.CampaignController"] = None) -> Optional[Dict[str, Any]]:
    """Сводка кампании для ассистента/MCP (§16.1) или ``None``, если кампании нет.

    Берёт контроллер из ``session_state`` (либо переданный) и отдаёт
    ``campaign_overview`` с объяснением денежного канала ρ (``with_money=True``),
    но БЕЗ дорогого econ-MC (``compute_value=False``) — нужны роли, занулённый/
    живой канал и причина «почему за ρ есть/нет денег» (``reason_code``/``text``),
    а не точная ₽-оценка. Read-only (A0.6); ошибки гасятся в ``None``, чтобы мост
    к ассистенту/Cline никогда не ронял основной UI.
    """
    ctrl = ctrl if ctrl is not None else get_campaign_controller()
    if ctrl is None:
        return None
    try:
        return ctrl.overview(with_money=True, compute_value=False)
    except Exception:  # noqa: BLE001 — мост не должен ломать UI
        return None


def _rho_of(runner, branch_id: str) -> Optional[str]:
    pcfg = cv.branch_price_config(runner, branch_id)
    return pcfg["rho_property"] if pcfg else None


def _parse_names(text: str) -> List[str]:
    """Разобрать имена через запятую/точку-с-запятой в список (без пустых)."""
    return [t.strip() for t in str(text).replace(";", ",").split(",") if t.strip()]


def _parse_floats(text: str) -> Optional[List[float]]:
    """Разобрать числа через запятую/точку-с-запятой; ``None`` при нечисловом вводе."""
    try:
        return [float(v) for v in str(text).replace(";", ",").split(",")
                if str(v).strip()]
    except ValueError:
        return None


def parse_sampling_groups(text: str) -> List[List[str]]:
    """iter31: разобрать функциональные группы компонентов из текста.

    Формат: одна СТРОКА = одна группа, имена компонентов через запятую
    (пустые строки игнорируются). Валидацию имён/непересечения делает движок
    (``set_mixture_sampling_groups`` / ``add_branch``) — явные ошибки (A0.6).
    Чистая (без Streamlit) — тестируется напрямую.
    """
    groups: List[List[str]] = []
    for line in str(text).splitlines():
        names = _parse_names(line)
        if names:
            groups.append(names)
    return groups


def sampling_groups_to_text(groups) -> str:
    """iter31: группы → текст формы (обратное к :func:`parse_sampling_groups`)."""
    return "\n".join(", ".join(g) for g in (groups or []))


# ----------------------------------------------------------------------
# iter52 / P2.1-UI (UI_REVISION_SPEC) — ДИСКРЕТНЫЕ УРОВНИ process-осей в UI.
# Ядро (iter51) умеет проецировать план и argmax на сетку уровней, но задать
# сетку было НЕЧЕМ: из интерфейса «rotor_rpm: 400, 900» не вводилось, в
# паспорте не показывалось — политика кампании существовала только в коде.
# Хелперы ниже ЧИСТЫЕ (без Streamlit); нормализацию (сортировка, дубли,
# границы оси) делает ШТАТНЫЙ ``runner.set_process_levels`` (A0.6).
# ----------------------------------------------------------------------
def parse_process_levels(text: str) -> Dict[str, List[float]]:
    """iter52/P2.1-UI: разобрать дискретные уровни process-осей из текста формы.

    Формат: одна СТРОКА = одна ось, ``имя: уровень, уровень…`` в РЕАЛЬНЫХ
    единицах (``rotor_rpm: 400, 900``). Пустые строки игнорируются; ось без
    строки остаётся НЕПРЕРЫВНОЙ (выключать сетку надо отсутствием строки, а
    не пустым списком — иначе «уровней нет» неотличимо от «оси нет в сетке»).

    Здесь ловится только СИНТАКСИС (с номером строки). Смысловая валидация —
    имя оси против схемы, попадание уровня в границы, дубли и сортировка —
    остаётся за штатным :meth:`MixtureProcessRunner.set_process_levels`
    (единый источник правил, A0.6). Чистая (без Streamlit) — round-trip с
    :func:`process_levels_to_text`.
    """
    out: Dict[str, List[float]] = {}
    for ln, line in enumerate(str(text or "").splitlines(), start=1):
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(
                f"Строка {ln}: ожидается «ось: уровень, уровень…» "
                f"(нет разделителя «:»): {line.strip()!r}.")
        name, rhs = line.split(":", 1)
        nm = name.strip()
        if not nm:
            raise ValueError(f"Строка {ln}: пустое имя оси: {line.strip()!r}.")
        vals = _parse_floats(rhs)
        if vals is None:
            raise ValueError(
                f"Строка {ln}: уровни оси «{nm}» — числа через запятую "
                f"(получено {rhs.strip()!r}).")
        if not vals:
            raise ValueError(
                f"Строка {ln}: у оси «{nm}» не задано ни одного уровня. "
                "Чтобы ось осталась непрерывной, не упоминайте её вовсе.")
        if nm in out:
            raise ValueError(
                f"Строка {ln}: ось «{nm}» указана повторно — перечислите все "
                "её уровни в ОДНОЙ строке.")
        out[nm] = vals
    return out


def process_levels_to_text(levels) -> str:
    """iter52: уровни → текст формы (обратное к :func:`parse_process_levels`)."""
    return "\n".join(
        f"{nm}: " + ", ".join(f"{float(v):g}" for v in vals)
        for nm, vals in (levels or {}).items())


def seed_levels_caption(runner, Xs) -> str:
    """iter52/P2.1-UI: подпись «план стоит на достижимых режимах» (чистая).

    Пусто (нет дискретных осей) → ПУСТАЯ строка: подпись «все оси непрерывны»
    у каждой таблицы плана была бы шумом (в паспорте кампании это состояние
    показывается явно). Если сетки заданы — перечисляем оси и проверяем сам
    план: точки вне сетки означают, что уровни задали ПОСЛЕ построения плана
    (например, загрузили проект и добавили политику) — это сигнал пересчитать
    план, а не повод блокировать фиксацию (A0.6).
    """
    levels = dict(getattr(runner, "process_levels", {}) or {})
    if not levels:
        return ""
    head = "⚙️ " + levels_caption(levels)
    X = np.atleast_2d(np.asarray(Xs, float))
    if X.size == 0:
        return head
    try:
        off = int(np.count_nonzero(
            np.abs(runner.snap_process_axes(X) - X).max(axis=1) > 1e-9))
    except (ValueError, IndexError):
        return head
    if off == 0:
        return (f"{head} Все {len(X)} точек плана стоят на уровнях — "
                "лаборатория поставит ровно то, что в таблице.")
    return (f"{head} ⚠️ {off} из {len(X)} точек ВНЕ сетки: уровни заданы "
            "после построения плана. Предложите план заново, иначе оператор "
            "поставит ближайший достижимый режим, а модель будет учиться на "
            "координатах из таблицы (A0.6 — не блокируем, но предупреждаем).")


# ----------------------------------------------------------------------
# P3.3 (UI_REVISION_SPEC) — связанные process-оси: чистые парсеры формы.
# Здесь ловится только СИНТАКСИС строки; смысловая валидация (имена осей,
# lo<hi, пересечение с достижимым диапазоном, ось не в двух связках,
# конфликт с уровнями) — за штатным ``runner.set_process_links`` (A0.6).
# ----------------------------------------------------------------------
def parse_process_links(text: str) -> List[Dict[str, Any]]:
    """P3.3: разобрать СВЯЗКИ process-осей из текста формы.

    Формат: одна СТРОКА = одна связка, ``имя: осьA - осьB : lo, hi`` в
    РЕАЛЬНЫХ единицах (``dT_head: T_adapter - T_plast : 10, 60``). Границу
    можно открыть звёздочкой (``*, 60`` — только верхний предел). Пустые
    строки игнорируются. Разность парсится по ОДНОМУ дефису — имена осей с
    дефисом этим каналом не задать (явная ошибка с подсказкой).

    Чистая (без Streamlit); round-trip с :func:`process_links_to_text`.
    Возвращает список словарей для
    :meth:`MixtureProcessRunner.set_process_links`.
    """
    out: List[Dict[str, Any]] = []
    for ln, line in enumerate(str(text or "").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Строка {ln}: ожидается «имя: осьA - осьB : lo, hi» "
                f"(два разделителя «:»): {line.strip()!r}.")
        name = parts[0].strip()
        if not name:
            raise ValueError(f"Строка {ln}: пустое имя производной величины: "
                             f"{line.strip()!r}.")
        expr = parts[1]
        sides = expr.split("-")
        if len(sides) != 2 or not sides[0].strip() or not sides[1].strip():
            raise ValueError(
                f"Строка {ln}: разность «{expr.strip()}» должна иметь вид "
                f"«осьA - осьB» (ровно один «-»; имена осей с дефисом этим "
                f"каналом задать нельзя).")
        a, b = sides[0].strip(), sides[1].strip()
        toks = [t.strip() for t in parts[2].split(",")]
        if len(toks) != 2:
            raise ValueError(
                f"Строка {ln}: полоса «{parts[2].strip()}» — два значения "
                f"через запятую («lo, hi»; открытая сторона — «*»).")
        bounds: List[Optional[float]] = []
        for t in toks:
            if t in ("*", ""):
                bounds.append(None)
                continue
            try:
                bounds.append(float(t))
            except ValueError:
                raise ValueError(
                    f"Строка {ln}: граница {t!r} не число (и не «*»).")
        out.append({"name": name, "minuend": a, "subtrahend": b,
                    "lo": bounds[0], "hi": bounds[1]})
    return out


def process_links_to_text(links) -> str:
    """P3.3: связки → текст формы (обратное к :func:`parse_process_links`).

    Принимает и ``ProcessLink``, и словари сериализации; ``±inf``/``None``
    пишется звёздочкой (открытая сторона полосы).
    """
    def _b(v) -> str:
        if v is None:
            return "*"
        v = float(v)
        return f"{v:g}" if np.isfinite(v) else "*"

    lines = []
    for lk in (links or []):
        if isinstance(lk, dict):
            nm, a, b = lk.get("name"), lk.get("minuend"), lk.get("subtrahend")
            lo, hi = lk.get("lo"), lk.get("hi")
        else:
            nm, a, b, lo, hi = (lk.name, lk.minuend, lk.subtrahend,
                                lk.lo, lk.hi)
        lines.append(f"{nm}: {a} - {b} : {_b(lo)}, {_b(hi)}")
    return "\n".join(lines)


def seed_links_caption(runner, Xs) -> str:
    """P3.3: подпись «план реализуем по связкам осей» (чистая).

    Пусто (связок нет) → ПУСТАЯ строка (как :func:`seed_levels_caption` —
    подпись у каждой таблицы была бы шумом; в паспорте состояние показано
    явно). Иначе — перечень связок + проверка реализуемости самого плана
    (:meth:`linked_axes_report`): точки вне полосы означают, что связки
    задали ПОСЛЕ построения плана — сигнал пересчитать, а не блокировка
    (A0.6).
    """
    links = list(getattr(runner, "process_links", []) or [])
    if not links:
        return ""
    head = "🔗 " + links_caption(links)
    X = np.atleast_2d(np.asarray(Xs, float))
    if X.size == 0:
        return head
    try:
        reports = runner.linked_axes_report(X)
    except (ValueError, IndexError):
        return head
    if not reports:
        return head          # обе оси связки вне текущей фазы
    off = int(sum(r["n_off"] for r in reports))
    if off == 0:
        return (f"{head} Все {len(X)} точек плана реализуемы — разности "
                "осей в полосе железа.")
    return (f"{head} ⚠️ {off} наруш. полосы в плане: связки заданы после "
            "построения плана. Предложите план заново, иначе оператор "
            "поставит достижимый перепад, а модель будет учиться на "
            "координатах из таблицы (A0.6 — не блокируем, но предупреждаем).")



# ----------------------------------------------------------------------
# iter41 (UI_REVISION_SPEC §41) — чистые хелперы phr-спеки и паспорта
# кампании (без Streamlit, тестируются напрямую)
# ----------------------------------------------------------------------
_MODE_PHR = "phr-спека (JSON)"


def parse_phr_spec_json(text: str) -> PhrSpec:
    """iter41.1: разобрать phr-спеку из JSON-текста формы/файла.

    Формат — список узлов :meth:`PhrSpec.from_dicts` (тот же, что
    :meth:`PhrSpec.to_dicts` → входит в ``spec_hash``). Ошибки формата
    JSON заворачиваются в человекочитаемый ``ValueError``; ошибки
    КОНСТРУКТОРА спеки (циклы, ссылки, пустые пересечения) — наружу как
    есть: они уже человекочитаемы и указывают на узел (A0.6).
    """
    s = str(text or "").strip()
    if not s:
        raise ValueError("Пустой ввод: вставьте JSON-список узлов phr-спеки "
                         "(формат PhrSpec.from_dicts).")
    try:
        data = json.loads(s)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Некорректный JSON (строка {exc.lineno}, позиция {exc.colno}): "
            f"{exc.msg}.") from exc
    if isinstance(data, dict):
        # iter46/B6: обёртка схемы v2 {"spec_version": 2, "nodes": [...]};
        # объект БЕЗ 'nodes' — это не обёртка, а ошибка формата (например,
        # один узел объектом вместо списка) — прежнее понятное сообщение.
        if "nodes" not in data:
            raise ValueError(
                "Ожидался JSON-СПИСОК узлов ([{...}, ...]) или обёртка "
                "{\"spec_version\": 2, \"nodes\": [...]}, получен объект "
                f"с ключами {sorted(data)}.")
        return PhrSpec.from_dicts(data)
    if not isinstance(data, list):
        raise ValueError("Ожидался JSON-СПИСОК узлов ([{...}, ...]) или "
                         "обёртка {\"spec_version\": 2, \"nodes\": [...]}, "
                         f"получен {type(data).__name__}.")
    if not all(isinstance(d, dict) for d in data):
        raise ValueError("Каждый узел спеки должен быть JSON-объектом "
                         "{\"name\": ..., \"mode\"/\"role\": ...}.")
    return PhrSpec.from_dicts(data)


def phr_spec_summary_dataframe(spec: PhrSpec) -> pd.DataFrame:
    """iter41.1 (+iter50/P1.3): сводка узлов спеки — узел / РОЛЬ / режим /
    lo / hi / ref / cap_to / cap_ratio / min_phr / max_phr / scale /
    «компонент смеси?» (лист DAG).

    Для ``fixed``-узлов lo=hi=value (честный вырожденный интервал);
    у ``share_closure`` диапазон ПРОИЗВОДНЫЙ (iter46/B2).

    iter50/P1.3: роль (:meth:`PhrSpec.role_of` — тот же источник, что и
    сериализация), технологические лимиты ``min_phr``/``max_phr``
    (iter45/B1) и шкала оси ``scale`` (iter47/B5) были невидимы из UI:
    пользователь не мог сверить с таблицей кампании ни «CPE ≥ 3 phr», ни
    «TiO₂ по логу» — а это ЧАСТЬ ГЕОМЕТРИИ и часть ``spec_hash``.
    ``min_phr``/``max_phr`` отсутствуют как ``NaN`` (числовая колонка), а
    не «—»: пустая ячейка не должна выглядеть как значение.

    Чистая (без Streamlit) — тестируется напрямую."""
    leaves = set(spec.component_names)
    rows: List[Dict[str, Any]] = []
    for nd in spec.nodes:
        if nd.mode == "fixed":
            lo, hi = float(nd.value), float(nd.value)
        elif nd.mode == "share_closure":
            # iter46/B2: у closure диапазон ПРОИЗВОДНЫЙ [1−φᵁ, 1−φᴸ] —
            # показываем его, а не сентинель (0, 0)
            lo, hi = map(float, spec.share_base_bounds(nd.name))
        else:
            lo, hi = float(nd.lo), float(nd.hi)
        rows.append({
            "узел": nd.name, "роль": spec.role_of(nd.name),
            "режим": nd.mode, "lo": lo, "hi": hi,
            "ref": nd.ref or "",
            "cap_to": ", ".join(nd.cap_refs),
            "cap_ratio": float(nd.cap_ratio) if nd.cap_refs else np.nan,
            "min_phr": (np.nan if nd.min_phr is None else float(nd.min_phr)),
            "max_phr": (np.nan if nd.max_phr is None else float(nd.max_phr)),
            "scale": str(nd.scale),
            "компонент смеси": nd.name in leaves,
        })
    return pd.DataFrame(rows)


def phr_spec_policy_caption(spec: PhrSpec) -> str:
    """iter50/P1.3: одна строка о ГЕОМЕТРИЧЕСКОЙ политике спеки —
    схема, q/dim_z, ``group_order``, лог-оси, узлы с техлимитами, хеш.

    Всё перечисленное входит в ``spec_hash`` (iter46–48), но до iter50 в
    интерфейсе не показывалось: две спеки с разным приоритетом групп или
    разной шкалой оси выглядели одинаково, а планы давали разные. Чистая
    (без Streamlit)."""
    log_axes = [nd.name for nd in spec.nodes if nd.scale == "log"]
    limited = [nd.name for nd in spec.nodes
               if nd.min_phr is not None or nd.max_phr is not None]
    order = list(getattr(spec, "group_order", []) or [])
    parts = [f"схема v{int(getattr(spec, 'schema_version', 1))}",
             f"компонентов q={spec.q}", f"z-осей {spec.dim_z}"]
    parts.append("приоритет групп (group_order): " + (" → ".join(order)
                 if order else "не задан"))
    parts.append("лог-оси (сэмплинг по ln phr): "
                 + (", ".join(log_axes) if log_axes else "нет"))
    parts.append("техлимиты min/max_phr: "
                 + (", ".join(limited) if limited else "нет"))
    return (" · ".join(parts)
            + f". spec_hash {spec.spec_hash()[:12]}… — всё перечисленное "
              "входит в отпечаток: смена шкалы, лимита или порядка групп "
              "меняет хеш и геометрию плана.")


# --- iter50/P1.3: блок «эффективные границы точки» (контракт iter49/B7) ---
# Метки active контракта — «что именно ограничивает точку». Словарь нужен
# для подписи: сырые метки стабильны (часть контракта ядра, на них опираются
# тесты), а лаборанту нужен русский смысл.
ACTIVE_LABEL_RU: Dict[str, str] = {
    "fixed": "фиксированное значение узла",
    "range": "заявленный интервал спеки",
    "derived": "производный диапазон замыкания (closure)",
    "window": "окно тотала группы (phr-лимиты членов)",
    "cap": "динамический потолок cap_ratio·Σ(cap_to) в ЭТОЙ точке",
    "min_phr": "технологический минимум узла (phr)",
    "max_phr": "технологический лимит узла (phr)",
    "partners": "партнёры по группе (Σφ = 1)",
}

_COORD_KIND = {
    "fixed": "phr", "absolute": "phr", "ratio_to": "коэффициент",
    "share_of": "доля", "share_free": "доля", "share_closure": "доля",
    "share_simplex": "доля",
}


def point_bounds_dataframe(spec: PhrSpec, x_fractions: Sequence[float], *,
                           delta_phr: Optional[float] = None
                           ) -> pd.DataFrame:
    """iter50/P1.3: ЭФФЕКТИВНЫЕ границы каждого узла В ЭТОЙ ТОЧКЕ.

    Вход — состав в ДОЛЯХ (как его отдаёт движок: строка seed-плана, x*
    ветки); phr восстанавливается :meth:`PhrSpec.fractions_to_phr`, всё
    остальное берётся из контракта ядра :meth:`PhrSpec.point_report`
    (iter49/B7) — геометрия в UI НЕ дублируется.

    | колонка | смысл |
    |---|---|
    | узел / роль | структура спеки (:meth:`PhrSpec.role_of`) |
    | координата | в чём измеряется узел: phr / доля / коэффициент |
    | значение | значение координаты в точке |
    | phr | значение узла в phr (у тоталов — сумма детей) |
    | lo / hi | эффективные границы координаты В ЭТОЙ точке |
    | активна lo / активна hi | КАКОЕ ограничение задало границу |
    | в границах | ✓ / ✗ (номинал внутри эффективных границ) |

    Зачем: условные границы §4 спеки (немонотонная ``hi_φ(T)``) из
    интерфейса не сверялись вообще — «почему план не даёт такую точку»
    оставалось без ответа. Нарушения НЕ блокируют (A0.6): это диагностика.

    Чистая (без Streamlit) — тестируется напрямую."""
    p_nom = spec.fractions_to_phr(np.asarray(x_fractions, dtype=float).ravel())
    rep = spec.point_report(p_nom, delta_phr=delta_phr)
    rows: List[Dict[str, Any]] = []
    for nm, b in rep.effective_bounds.items():
        inside = (b.lo - 1e-6) <= b.coord <= (b.hi + 1e-6) and b.lo <= b.hi
        rows.append({
            "узел": nm,
            "роль": spec.role_of(nm),
            "координата": _COORD_KIND.get(b.mode, b.mode),
            "значение": round(float(b.coord), 6),
            "phr": round(float(b.phr), 4),
            "lo": round(float(b.lo), 6),
            "hi": round(float(b.hi), 6),
            "активна lo": b.active_lo,
            "активна hi": b.active_hi,
            "в границах": "✓" if inside else "✗",
        })
    return pd.DataFrame(rows)


def point_bounds_caption(df: pd.DataFrame) -> str:
    """iter50/P1.3: подпись под таблицей эффективных границ — расшифровка
    ТОЛЬКО тех меток ``active``, которые в этой точке реально встретились
    (глоссарий на восемь строк никто не читает), и счётчик узлов вне
    границ. Чистая (без Streamlit)."""
    if df is None or df.empty:
        return "Точка не разобрана: таблица границ пуста."
    labels: List[str] = []
    for col in ("активна lo", "активна hi"):
        for v in df[col]:
            if v not in labels:
                labels.append(str(v))
    gloss = "; ".join(f"{lab} — {ACTIVE_LABEL_RU.get(lab, lab)}"
                      for lab in labels)
    bad = int((df["в границах"] == "✗").sum())
    tail = ("все узлы внутри эффективных границ"
            if bad == 0 else
            f"⚠️ вне границ: {bad} узл. — точка не принадлежит геометрии "
            f"спеки (диагностика, фиксация не блокируется, A0.6)")
    return (f"Границы посчитаны ДЛЯ ЭТОЙ точки (контракт ядра, iter49/B7): "
            f"cap/окно тотала/партнёры зависят от самой точки. "
            f"Метки: {gloss}. {tail}.")



def phr_spec_fraction_dataframe(spec: PhrSpec) -> pd.DataFrame:
    """iter41.1: интервалы phr компонентов (:meth:`PhrSpec.phr_intervals`)
    и рассчитанные ДОЛИ для mixture-блока (:meth:`PhrSpec.fraction_bounds`)
    — аналог таблицы режима «Массовые части». Индекс = компоненты смеси."""
    iv = spec.phr_intervals()
    lo, hi = spec.fraction_bounds()
    return pd.DataFrame(
        {"phr lo": [iv[nm][0] for nm in spec.component_names],
         "phr hi": [iv[nm][1] for nm in spec.component_names],
         "доля L": np.round(lo, 6), "доля U": np.round(hi, 6)},
        index=list(spec.component_names))


def weighing_delta_phr(step_g: float, grams_per_phr: float) -> float:
    """iter42.2: разрешение навески δ в phr из ПАРАМЕТРОВ ЛАБОРАТОРИИ.

    ``δ = шаг весов (г) / (г на 1 phr)``. Пример CAMPAIGN_SPEC_PVC §5:
    весы 0.1 г при загрузке 5 г на 1 phr → δ = 0.02 phr. Оба аргумента
    строго положительны: нулевой шаг весов означал бы бесконечное
    разрешение (правило премикса выродилось бы), нулевая загрузка —
    неопределённый перевод г → phr. Чистая (без Streamlit)."""
    step = float(step_g)
    gpp = float(grams_per_phr)
    if step <= 0:
        raise ValueError("Шаг весов должен быть > 0 г (иначе разрешение "
                         "навески не определено).")
    if gpp <= 0:
        raise ValueError("Загрузка «г на 1 phr» должна быть > 0 (иначе "
                         "перевод граммов в phr не определён).")
    return step / gpp


def recipe_weighing_dataframe(spec: PhrSpec, x_fractions: Sequence[float],
                              delta_phr: float, *,
                              grams_per_phr: Optional[float] = None
                              ) -> pd.DataFrame:
    """iter42.3: карта НАВЕСКИ одного рецепта — nominal / actual / премикс.

    Вход — состав в ДОЛЯХ (как его отдаёт движок: seed-план, x* ветки);
    phr восстанавливается :meth:`PhrSpec.fractions_to_phr` по fixed-якорю
    спеки. Всё остальное берётся из контракта ядра
    :meth:`PhrSpec.point_report` (iter49) — логика навески не дублируется:

    | колонка | источник |
    |---|---|
    | компонент | ``spec.component_names`` |
    | phr nominal | ``fractions_to_phr(x)`` |
    | phr actual | ``quantize_recipe(p, δ).p_actual`` (внутри point_report) |
    | граммы actual | ``p_actual · г/phr`` (если задана загрузка) |
    | премикс | ``premix_required`` по ``phr_intervals``; «—» = неприменимо |
    | нарушение | строки ``violations``, относящиеся к этому узлу |

    ⚠️ Дозируйте и фиксируйте ФАКТИЧЕСКИЕ значения (actual): модель должна
    видеть actual, а не nominal (CAMPAIGN_SPEC_PVC §5). Нарушения НЕ
    блокируют — это диагностика (A0.6), решение за пользователем.

    Чистая (без Streamlit) — тестируется напрямую."""
    p_nom = spec.fractions_to_phr(np.asarray(x_fractions, dtype=float).ravel())
    rep = spec.point_report(p_nom, delta_phr=float(delta_phr))
    actual = rep.phr_actual
    rows: List[Dict[str, Any]] = []
    for j, nm in enumerate(spec.component_names):
        pm = rep.premix.get(nm)
        row: Dict[str, Any] = {
            "компонент": nm,
            "phr nominal": round(float(p_nom[j]), 4),
            "phr actual": round(float(actual[j]), 4),
        }
        if grams_per_phr is not None and float(grams_per_phr) > 0:
            row["граммы actual"] = round(
                float(actual[j]) * float(grams_per_phr), 4)
        row["премикс"] = "—" if pm is None else ("да" if pm else "нет")
        row["нарушение"] = "; ".join(
            v.split(": ", 1)[1] if ": " in v else v
            for v in rep.violations if v.startswith(f"{nm}: "))
        rows.append(row)
    return pd.DataFrame(rows)


def weighing_caption(spec: PhrSpec, delta_phr: float) -> str:
    """iter42.3: подпись под картой навески — δ, сколько осей требует
    премикса и обязательное требование фиксировать actual."""
    iv = spec.phr_intervals()
    need = [nm for nm in spec.component_names
            if iv[nm][1] > iv[nm][0] + 1e-9
            and premix_required(float(delta_phr), *iv[nm])]
    part = (f"премикс нужен для {len(need)} осей ({', '.join(need)})"
            if need else "все оси читаются прямой навеской")
    return (f"δ = {float(delta_phr):g} phr · {part}. Дозируйте и фиксируйте "
            f"ФАКТИЧЕСКИЕ значения (actual): модель должна видеть actual, "
            f"а не nominal (CAMPAIGN_SPEC_PVC §5).")


def snap_design_to_grid(spec: PhrSpec, X, delta_phr: float) -> np.ndarray:
    """iter42.4: снап ПЛАНА к δ-сетке весов — фиксируется ACTUAL, не nominal.

    Вход — составной план ``X`` (n × (q+d)): первые ``spec.q`` столбцов —
    mixture-ДОЛИ, остальные (процесс) НЕ трогаются. Каждая строка идёт
    ``доли → phr (fractions_to_phr) → quantize_recipe(δ).p_actual → доли``.

    Зачем ДО фиксации: лаборант физически навесит кратное шагу весов, и если
    зафиксировать номинал, модель будет учиться на координатах, которых в
    стакане не было (CAMPAIGN_SPEC_PVC §5). Операция ИДЕМПОТЕНТНА: точка,
    уже стоящая на δ-сетке, не двигается — поэтому безопасно применять на
    каждом прогоне UI. Чистая (без Streamlit)."""
    X = np.atleast_2d(np.asarray(X, dtype=float)).copy()
    q = int(spec.q)
    if X.shape[1] < q:
        raise ValueError(
            f"snap_design_to_grid: в плане {X.shape[1]} координат, а спека "
            f"описывает {q} компонентов смеси.")
    for i in range(len(X)):
        p_nom = spec.fractions_to_phr(X[i, :q])
        p_act = spec.quantize_recipe(p_nom, float(delta_phr)).p_actual
        X[i, :q] = spec.to_fractions(p_act)
    return X


def seed_weighing_dataframe(runner, spec: PhrSpec, X, delta_phr: float, *,
                            grams_per_phr: Optional[float] = None
                            ) -> pd.DataFrame:
    """iter42.4: карта навески ВСЕГО плана в ДЛИННОМ формате (показ/Excel).

    Строка = «опыт × компонент» (а не 3·q колонок на опыт: при q≈19 широкая
    таблица нечитаема и в Excel, и на экране). Колонки: «№ опыта» +
    результат :func:`recipe_weighing_dataframe` по этой строке плана.
    Номера опытов — БУДУЩИЕ номера в общей базе (:func:`experiment_index`).
    Чистая (без Streamlit)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    nums = list(experiment_index(len(getattr(runner, "points", []) or []),
                                 len(X)))
    frames: List[pd.DataFrame] = []
    for i in range(len(X)):
        df = recipe_weighing_dataframe(spec, X[i, :spec.q], delta_phr,
                                       grams_per_phr=grams_per_phr)
        df.insert(0, "№ опыта", nums[i])
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------
# iter41.4 (расширение §41.1) — ИЕРАРХИЧЕСКИЙ ручной ввод спеки.
#
# Рецептура в лаборатории мыслится группами («стабилизатор 3.5…5 phr, внутри
# PF711 60…100 %»), а не плоским списком узлов DAG. Дерево ввода — ПЛОСКИЙ
# список блоков верхнего уровня, каждый из которых либо ГРУППА (узел-тотал +
# доли-дети share_of), либо ОДИНОЧНЫЙ узел (absolute / fixed / ratio_to).
# Порядок блоков и порядок детей задаёт пользователь: он входит в spec_hash
# (перестановка узлов — другая мера сэмплера, iter35), поэтому кнопки ▲/▼ —
# не косметика, а часть спецификации.
#
# Дерево JSON-native (списки/словари/числа/строки) — переживает session_state
# и сериализацию. Все функции ниже ЧИСТЫЕ (без Streamlit).
# ----------------------------------------------------------------------
PHR_TOTAL_MODES = ("absolute", "fixed")
PHR_SINGLE_MODES = ("absolute", "fixed", "ratio_to")


def phr_group_block(name: str, *, total_mode: str = "absolute",
                    lo: float = 0.0, hi: float = 0.0, value: float = 0.0,
                    children: Optional[Sequence[Mapping[str, Any]]] = None
                    ) -> Dict[str, Any]:
    """iter41.4: блок-ГРУППА дерева ввода (узел-тотал + доли-дети).

    ``total_mode='absolute'`` — тотал группы phr ∈ [lo, hi] (наполнитель
    5…25 phr); ``'fixed'`` — тотал константа (смола = 100). Дети —
    словари ``{"name", "lo", "hi"}`` с ДОЛЯМИ группы (0…1)."""
    return {"kind": "group", "name": str(name), "total_mode": str(total_mode),
            "lo": float(lo), "hi": float(hi), "value": float(value),
            "children": [dict(c) for c in (children or [])]}


def phr_single_block(name: str, *, mode: str = "absolute", lo: float = 0.0,
                     hi: float = 0.0, value: float = 0.0, ref: str = "",
                     cap_to: Any = "", cap_ratio: float = 0.0,
                     scale: str = "linear") -> Dict[str, Any]:
    """iter41.4: блок-ОДИНОЧНЫЙ узел дерева ввода (компонент вне групп).

    ``absolute`` — phr ∈ [lo, hi] (опц. динамический потолок ``cap_to`` /
    ``cap_ratio``); ``fixed`` — константа; ``ratio_to`` — коэффициент
    [lo, hi] к узлу ``ref`` (SBM = 0.02…0.09 × Σ стабилизатора).

    iter56/P3.2: ``scale`` — шкала сэмплинга absolute-оси (``linear`` |
    ``log``, iter47/B5); доступна только в схеме v2."""
    return {"kind": "single", "name": str(name), "mode": str(mode),
            "lo": float(lo), "hi": float(hi), "value": float(value),
            "ref": str(ref or ""), "cap_to": cap_to,
            "cap_ratio": float(cap_ratio),
            "scale": str(scale or "linear")}


def _phr_cap_refs(raw: Any) -> List[str]:
    """iter41.4: ``cap_to`` из формы → список имён.

    Принимает список/кортеж ИЛИ строку через запятую (``"DINP, ESO"``) —
    оба канала дают ОДИН И ТОТ ЖЕ набор ссылок, иначе ввод строкой давал
    бы другой spec_hash, чем ввод списком."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(s).strip() for s in raw if str(s).strip()]
    return _parse_names(str(raw))


def validate_phr_tree(tree: Sequence[Mapping[str, Any]], *,
                      schema_version: int = 1) -> None:
    """iter41.4: проверить дерево ввода ДО конструктора :class:`PhrSpec`.

    Ловит то, что конструктор поймать не может или сообщит невнятно:

    * пустое дерево, неизвестный ``kind``/режим, пустые имена;
    * дубли имён (узел-тотал и компонент — одно пространство имён);
    * **группа без детей** — конструктор молча сделает её ЛИСТОМ, то есть
      компонентом смеси: тихая порча состава (A0.6 — не молчим);
    * ``ratio_to`` без ссылки, ``cap_ratio`` без ``cap_to`` и наоборот,
      ``lo > hi``.

    iter56/P3.2: ``schema_version`` — целевая схема сериализации.
    v1 ОТВЕРГАЕТ v2-фичи (closure-пометка, техлимиты min/max phr,
    ``scale='log'``) явной ошибкой: молча выброшенное поле — потерянное
    намерение (A0.6). v2 проверяет то, что конструктор УВИДЕТЬ не может:
    доли L/U у closure-компонента (в схему они не попадают — диапазон
    производный) и min > max техлимитов.

    Ошибки конструктора (циклы, Σдолей, неизвестные ссылки, состав ролей
    k=2/k≥3) остаются за :meth:`PhrSpec.from_dicts` — здесь их не
    дублируем (канон iter52: второй набор правил разошёлся бы с ядром)."""
    if not tree:
        raise ValueError("Спека пуста: добавьте хотя бы один узел "
                         "(группу или компонент).")
    seen: List[str] = []

    def _claim(name: str, where: str) -> None:
        nm = str(name or "").strip()
        if not nm:
            raise ValueError(f"{where}: пустое имя узла.")
        if nm in seen:
            raise ValueError(f"{where}: имя '{nm}' уже занято — имена узлов "
                             "(групп и компонентов) должны быть уникальны.")
        seen.append(nm)

    for i, blk in enumerate(tree, start=1):
        kind = str(blk.get("kind", ""))
        name = str(blk.get("name", "") or "").strip()
        where = f"Блок {i} ('{name}')" if name else f"Блок {i}"
        if kind == "group":
            _claim(name, where)
            tm = str(blk.get("total_mode", ""))
            if tm not in PHR_TOTAL_MODES:
                raise ValueError(f"{where}: тотал группы — 'absolute' или "
                                 f"'fixed', получено '{tm}'.")
            if tm == "absolute" and float(blk.get("lo", 0.0)) > float(
                    blk.get("hi", 0.0)):
                raise ValueError(f"{where}: нижняя граница тотала больше "
                                 "верхней.")
            children = list(blk.get("children") or [])
            if not children:
                raise ValueError(
                    f"{where}: группа без компонентов. Такой узел станет "
                    "обычным компонентом смеси (лист DAG) — добавьте "
                    "компоненты или сделайте его одиночным узлом.")
            for j, ch in enumerate(children, start=1):
                cw = f"{where}, компонент {j}"
                _claim(ch.get("name", ""), cw)
                if float(ch.get("lo", 0.0)) > float(ch.get("hi", 0.0)):
                    raise ValueError(f"{cw}: доля L больше доли U.")
                closure = bool(ch.get("closure", False))
                mn = ch.get("min_phr", None)
                mx = ch.get("max_phr", None)
                if int(schema_version) == 1:
                    if closure:
                        raise ValueError(
                            f"{cw}: пометка closure доступна только в "
                            "схеме v2 (роли) — переключите схему спеки.")
                    if mn is not None or mx is not None:
                        raise ValueError(
                            f"{cw}: техлимиты min/max phr доступны только "
                            "в схеме v2 (роли) — переключите схему спеки.")
                    continue
                if closure and (float(ch.get("lo", 0.0)) != 0.0
                                or float(ch.get("hi", 0.0)) != 0.0):
                    raise ValueError(
                        f"{cw}: у closure-компонента доли L/U не задаются "
                        "— диапазон ПРОИЗВОДНЫЙ [1−φᵁ, 1−φᴸ] от партнёра "
                        "(B8); очистите значения (0).")
                if (mn is not None and mx is not None
                        and float(mn) > float(mx)):
                    raise ValueError(f"{cw}: min phr больше max phr.")
        elif kind == "single":
            _claim(name, where)
            md = str(blk.get("mode", ""))
            if md not in PHR_SINGLE_MODES:
                raise ValueError(f"{where}: режим — один из "
                                 f"{list(PHR_SINGLE_MODES)}, получено '{md}'.")
            if md in ("absolute", "ratio_to") and float(
                    blk.get("lo", 0.0)) > float(blk.get("hi", 0.0)):
                raise ValueError(f"{where}: нижняя граница больше верхней.")
            if md == "ratio_to" and not str(blk.get("ref", "") or "").strip():
                raise ValueError(f"{where}: режим ratio_to требует ссылку на "
                                 "узел (поле «к узлу»).")
            caps = _phr_cap_refs(blk.get("cap_to", ""))
            ratio = float(blk.get("cap_ratio", 0.0) or 0.0)
            if caps and md != "absolute":
                raise ValueError(f"{where}: динамический потолок (cap_to) "
                                 "допустим только для absolute.")
            if caps and ratio <= 0:
                raise ValueError(f"{where}: задан cap_to, но cap_ratio ≤ 0.")
            if ratio > 0 and not caps:
                raise ValueError(f"{where}: задан cap_ratio, но не указан "
                                 "cap_to (узел или список узлов).")
            scale = str(blk.get("scale", "linear") or "linear")
            if scale not in ("linear", "log"):
                raise ValueError(f"{where}: неизвестная шкала '{scale}' "
                                 "(допустимо: linear, log).")
            if scale == "log":
                if int(schema_version) == 1:
                    raise ValueError(
                        f"{where}: scale='log' доступен только в схеме v2 "
                        "(роли) — переключите схему спеки.")
                if md != "absolute":
                    raise ValueError(
                        f"{where}: scale='log' допустим только для "
                        "absolute (лог-шкала — свойство собственной "
                        "phr-оси).")
        else:
            raise ValueError(f"{where}: неизвестный тип блока '{kind}' "
                             "(ожидается 'group' или 'single').")


def phr_tree_to_dicts(tree: Sequence[Mapping[str, Any]], *,
                      schema_version: int = 1,
                      group_order: Optional[Sequence[str]] = None
                      ) -> List[Dict[str, Any]] | Dict[str, Any]:
    """iter41.4: дерево ввода → представление формата
    :meth:`PhrSpec.from_dicts`.

    Порядок результата = порядок блоков; внутри группы сначала идёт
    узел-тотал, затем его дети в порядке таблицы. Именно этот порядок
    попадёт в ``spec_hash``. Перед разворотом вызывается
    :func:`validate_phr_tree`.

    iter56/P3.2: ``schema_version=2`` — сериализация в РОЛИ нового
    контракта (iter46/B6): дети группы при k=2 — SHARE_FREE +
    SHARE_CLOSURE (кто closure — пометка ``closure`` ребёнка), при k≠2 —
    SHARE_SIMPLEX (closure-пометка эмитится как есть и отвергается
    конструктором с внятным объяснением — состав ролей НЕ дублируем,
    канон iter52); техлимиты ``min_phr``/``max_phr`` и ``scale``
    прокидываются. Непустой ``group_order`` заворачивает результат в
    обёртку ``{"spec_version": 2, "group_order": [...], "nodes": [...]}``
    (iter48/B4 — порядок групп входит в отпечаток); с v1 ``group_order``
    несовместим (ролей GROUP_TOTAL в legacy нет) — явная ошибка."""
    validate_phr_tree(tree, schema_version=schema_version)
    go = [str(g) for g in (group_order or [])]
    if int(schema_version) == 1:
        if go:
            raise ValueError(
                "group_order поддерживается только схемой v2 (роли): "
                "в legacy-схеме v1 роли GROUP_TOTAL нет (iter48/B4).")
        out: List[Dict[str, Any]] = []
        for blk in tree:
            name = str(blk["name"]).strip()
            if str(blk.get("kind")) == "group":
                if str(blk.get("total_mode")) == "fixed":
                    out.append({"name": name, "mode": "fixed",
                                "value": float(blk.get("value", 0.0))})
                else:
                    out.append({"name": name, "mode": "absolute",
                                "lo": float(blk.get("lo", 0.0)),
                                "hi": float(blk.get("hi", 0.0))})
                for ch in blk.get("children") or []:
                    out.append({"name": str(ch["name"]).strip(),
                                "mode": "share_of", "of": name,
                                "lo": float(ch.get("lo", 0.0)),
                                "hi": float(ch.get("hi", 0.0))})
                continue
            md = str(blk.get("mode"))
            if md == "fixed":
                out.append({"name": name, "mode": "fixed",
                            "value": float(blk.get("value", 0.0))})
            elif md == "ratio_to":
                out.append({"name": name, "mode": "ratio_to",
                            "to": str(blk.get("ref", "")).strip(),
                            "lo": float(blk.get("lo", 0.0)),
                            "hi": float(blk.get("hi", 0.0))})
            else:
                node: Dict[str, Any] = {"name": name, "mode": "absolute",
                                        "lo": float(blk.get("lo", 0.0)),
                                        "hi": float(blk.get("hi", 0.0))}
                caps = _phr_cap_refs(blk.get("cap_to", ""))
                if caps:
                    node["cap_to"] = caps[0] if len(caps) == 1 else caps
                    node["cap_ratio"] = float(blk.get("cap_ratio", 0.0))
                out.append(node)
        return out
    # --- схема v2 (роли, iter56/P3.2) ---
    nodes: List[Dict[str, Any]] = []
    for blk in tree:
        name = str(blk["name"]).strip()
        if str(blk.get("kind")) == "group":
            children = list(blk.get("children") or [])
            members = [str(c["name"]).strip() for c in children]
            if str(blk.get("total_mode")) == "fixed":
                nodes.append({"name": name, "role": "GROUP_TOTAL_FIXED",
                              "value": float(blk.get("value", 0.0)),
                              "members": members})
            else:
                nodes.append({"name": name, "role": "GROUP_TOTAL",
                              "range": [float(blk.get("lo", 0.0)),
                                        float(blk.get("hi", 0.0))],
                              "members": members})
            k = len(children)
            for ch in children:
                cn = str(ch["name"]).strip()
                if bool(ch.get("closure", False)):
                    d: Dict[str, Any] = {"name": cn,
                                         "role": "SHARE_CLOSURE",
                                         "group": name}
                else:
                    d = {"name": cn,
                         "role": "SHARE_FREE" if k == 2 else "SHARE_SIMPLEX",
                         "group": name,
                         "share_range": [float(ch.get("lo", 0.0)),
                                         float(ch.get("hi", 0.0))]}
                if ch.get("min_phr", None) is not None:
                    d["min_phr"] = float(ch["min_phr"])
                if ch.get("max_phr", None) is not None:
                    d["max_phr"] = float(ch["max_phr"])
                nodes.append(d)
            continue
        md = str(blk.get("mode"))
        if md == "fixed":
            nodes.append({"name": name, "role": "FIXED",
                          "value": float(blk.get("value", 0.0))})
        elif md == "ratio_to":
            nodes.append({"name": name, "role": "RATIO_TO",
                          "reference": str(blk.get("ref", "")).strip(),
                          "range": [float(blk.get("lo", 0.0)),
                                    float(blk.get("hi", 0.0))]})
        else:
            caps = _phr_cap_refs(blk.get("cap_to", ""))
            d = {"name": name,
                 "role": "ABSOLUTE_CAPPED" if caps else "ABSOLUTE",
                 "range": [float(blk.get("lo", 0.0)),
                           float(blk.get("hi", 0.0))]}
            if str(blk.get("scale", "linear") or "linear") == "log":
                d["scale"] = "log"
            if caps:
                d["cap_to"] = caps          # v2: cap_to — всегда СПИСОК
                d["cap_ratio"] = float(blk.get("cap_ratio", 0.0))
            nodes.append(d)
    if go:
        return {"spec_version": 2, "group_order": go, "nodes": nodes}
    return nodes


def phr_tree_from_spec(spec: PhrSpec) -> List[Dict[str, Any]]:
    """iter41.4: спека → дерево ввода (обратное к
    :func:`phr_tree_to_dicts`).

    Нужно для префилла формы после загрузки проекта: узел, на который
    ссылаются share-дети, становится ГРУППОЙ; сами дети уходят внутрь
    неё; остальные узлы — одиночные. Порядок сохраняется, поэтому
    round-trip даёт ТОТ ЖЕ ``spec_hash``.

    iter56/P3.2: поддержана и схема v2 (роли) — закрыт хвост iter46,
    когда дерево было legacy-only и единственным каналом ввода спеки
    кампании оставался JSON. У v2 проекция требует, чтобы члены группы
    шли в спеке СРАЗУ ЗА своим тоталом: дерево форсирует именно такой
    порядок узлов при обратной сборке, и иная раскладка молча изменила
    бы ``spec_hash`` на round-trip (A0.6 — явная ошибка с советом
    использовать канал «JSON / файл», а не тихая смена отпечатка).
    ``group_order`` — свойство спеки целиком, в дерево не входит
    (его отражает поле формы; см. ``spec.group_order``)."""
    if getattr(spec, "schema_version", 1) == 2:
        kids_v2: Dict[str, List[Any]] = {}
        for nd in spec.nodes:
            if nd.mode in ("share_free", "share_closure", "share_simplex"):
                kids_v2.setdefault(nd.ref, []).append(nd)
        tree: List[Dict[str, Any]] = []
        nodes = list(spec.nodes)
        i = 0
        while i < len(nodes):
            nd = nodes[i]
            if nd.name in kids_v2:
                members = kids_v2[nd.name]
                expect = [m.name for m in members]
                actual = [n.name for n in nodes[i + 1:i + 1 + len(members)]]
                if actual != expect:
                    raise ValueError(
                        f"Группа '{nd.name}': члены {expect} идут в спеке "
                        f"не сразу за тоталом (далее: {actual}) — такой "
                        "порядок узлов не проецируется в дерево без "
                        "изменения spec_hash; правьте спеку каналом "
                        "«JSON / файл».")
                children = [{"name": m.name, "lo": float(m.lo),
                             "hi": float(m.hi), "min_phr": m.min_phr,
                             "max_phr": m.max_phr,
                             "closure": m.mode == "share_closure"}
                            for m in members]
                tree.append(phr_group_block(
                    nd.name,
                    total_mode="fixed" if nd.mode == "fixed" else "absolute",
                    lo=float(nd.lo), hi=float(nd.hi), value=float(nd.value),
                    children=children))
                i += 1 + len(members)
                continue
            if nd.mode in ("share_free", "share_closure", "share_simplex"):
                raise ValueError(
                    f"Узел '{nd.name}': член группы '{nd.ref}' идёт в "
                    "спеке ДО своего тотала — такой порядок узлов не "
                    "проецируется в дерево без изменения spec_hash; "
                    "правьте спеку каналом «JSON / файл».")
            tree.append(phr_single_block(
                nd.name, mode=nd.mode, lo=float(nd.lo), hi=float(nd.hi),
                value=float(nd.value), ref=nd.ref or "",
                cap_to=list(nd.cap_refs), cap_ratio=float(nd.cap_ratio),
                scale=str(nd.scale or "linear")))
            i += 1
        return tree
    kids: Dict[str, List[Dict[str, Any]]] = {}
    for nd in spec.nodes:
        if nd.mode == "share_of":
            kids.setdefault(nd.ref, []).append(
                {"name": nd.name, "lo": float(nd.lo), "hi": float(nd.hi)})
    tree = []
    for nd in spec.nodes:
        if nd.mode == "share_of":
            continue
        if nd.name in kids:
            tree.append(phr_group_block(
                nd.name,
                total_mode="fixed" if nd.mode == "fixed" else "absolute",
                lo=float(nd.lo), hi=float(nd.hi), value=float(nd.value),
                children=kids[nd.name]))
            continue
        tree.append(phr_single_block(
            nd.name, mode=nd.mode, lo=float(nd.lo), hi=float(nd.hi),
            value=float(nd.value), ref=nd.ref or "",
            cap_to=list(nd.cap_refs), cap_ratio=float(nd.cap_ratio)))
    return tree


def phr_tree_move(tree: Sequence[Mapping[str, Any]], index: int, delta: int
                  ) -> List[Dict[str, Any]]:
    """iter41.4: переместить блок верхнего уровня (кнопки ▲/▼).

    Возвращает НОВЫЙ список (исходный не мутируется). Выход за границы —
    список без изменений: край списка не ошибка пользователя."""
    items = [dict(b) for b in tree]
    j = int(index) + int(delta)
    if not (0 <= int(index) < len(items)) or not (0 <= j < len(items)):
        return items
    items[int(index)], items[j] = items[j], items[int(index)]
    return items


def phr_children_dataframe(block: Mapping[str, Any], *,
                           schema_version: int = 1) -> pd.DataFrame:
    """iter41.4: дети группы → таблица редактора (компонент / доля L / U).

    iter56/P3.2: в схеме v2 добавляются колонки ``min phr`` / ``max phr``
    (техлимиты узла в phr; NaN = не задан — нуль выглядел бы значением,
    канон iter50) и ``closure`` (зависимый член k=2-группы без z-оси)."""
    v2 = int(schema_version) == 2
    rows = []
    for c in (block.get("children") or []):
        r: Dict[str, Any] = {"компонент": str(c.get("name", "")),
                             "доля L": float(c.get("lo", 0.0)),
                             "доля U": float(c.get("hi", 1.0))}
        if v2:
            mn = c.get("min_phr", None)
            mx = c.get("max_phr", None)
            r["min phr"] = float(mn) if mn is not None else np.nan
            r["max phr"] = float(mx) if mx is not None else np.nan
            r["closure"] = bool(c.get("closure", False))
        rows.append(r)
    if not rows:
        r = {"компонент": "", "доля L": 0.0, "доля U": 1.0}
        if v2:
            r.update({"min phr": np.nan, "max phr": np.nan,
                      "closure": False})
        rows = [r]
    return pd.DataFrame(rows)


def phr_children_from_dataframe(df, *, schema_version: int = 1
                                ) -> List[Dict[str, Any]]:
    """iter41.4: таблица редактора → дети группы (порядок строк сохраняется).

    Строки с пустым именем отбрасываются — это «пустой хвост» динамического
    редактора, а не ошибка ввода. iter56/P3.2: в схеме v2 читаются
    колонки ``min phr`` / ``max phr`` (NaN/пусто → None — «лимита нет»)
    и ``closure``."""
    v2 = int(schema_version) == 2
    out: List[Dict[str, Any]] = []
    for _, r in pd.DataFrame(df).iterrows():
        nm = str(r.get("компонент", "") or "").strip()
        if not nm:
            continue
        try:
            lo = float(r.get("доля L", 0.0) or 0.0)
            hi = float(r.get("доля U", 0.0) or 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Компонент '{nm}': доли должны быть числами "
                             "(0…1).") from exc
        ch: Dict[str, Any] = {"name": nm, "lo": lo, "hi": hi}
        if v2:
            mn = r.get("min phr", None)
            mx = r.get("max phr", None)
            try:
                ch["min_phr"] = (None if mn is None or pd.isna(mn)
                                 else float(mn))
                ch["max_phr"] = (None if mx is None or pd.isna(mx)
                                 else float(mx))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Компонент '{nm}': техлимиты min/max phr должны быть "
                    "числами (phr) или пустыми.") from exc
            ch["closure"] = bool(r.get("closure", False))
        out.append(ch)
    return out


def parse_preflight_pairs(text: str) -> List[Tuple[List[str], List[str]]]:
    """iter41.2: разобрать обязательные 2D-пары из текста формы.

    Формат: одна СТРОКА = одна пара, стороны через ``|``, ось-сумма —
    имена через запятую (``T | PMPlus_8, DL_531``). Пустые строки
    игнорируются. Ошибки формата — явный ``ValueError`` с номером строки;
    валидацию ИМЁН против схемы делает :meth:`set_preflight_pairs` (A0.6).
    Чистая (без Streamlit) — round-trip c :func:`preflight_pairs_to_text`.
    """
    pairs: List[Tuple[List[str], List[str]]] = []
    for ln, line in enumerate(str(text or "").splitlines(), start=1):
        if not line.strip():
            continue
        sides = line.split("|")
        if len(sides) != 2:
            raise ValueError(
                f"Строка {ln}: ожидается «осьA | осьB» (ровно один "
                f"разделитель «|»): {line.strip()!r}.")
        a, b = _parse_names(sides[0]), _parse_names(sides[1])
        if not a or not b:
            raise ValueError(
                f"Строка {ln}: пустая сторона пары: {line.strip()!r}.")
        pairs.append((a, b))
    return pairs


def preflight_pairs_to_text(pairs) -> str:
    """iter41.2: пары → текст формы (обратное к :func:`parse_preflight_pairs`)."""
    return "\n".join(f"{', '.join(a)} | {', '.join(b)}"
                     for a, b in (pairs or []))


def parse_material_lots(text: str) -> Dict[str, str]:
    """P2.3: разобрать ЛОТЫ СЫРЬЯ из текста формы (паспорт кампании).

    Формат: одна СТРОКА = один компонент, «компонент: лот» (первое «:» —
    разделитель, лот может содержать пробелы/дефисы). Пустые строки
    игнорируются. Здесь ловится только СИНТАКСИС (с номером строки);
    валидацию имён против схемы делает штатный ``set_material_lots``
    (канон iter52: второй набор правил в UI разошёлся бы с ядром, A0.6).
    Чистая (без Streamlit) — round-trip с :func:`material_lots_to_text`.
    """
    out: Dict[str, str] = {}
    for ln, line in enumerate(str(text or "").splitlines(), start=1):
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(
                f"Строка {ln}: ожидается «компонент: лот» (разделитель "
                f"«:»): {line.strip()!r}.")
        nm, lot = line.split(":", 1)
        nm, lot = nm.strip(), lot.strip()
        if not nm or not lot:
            raise ValueError(
                f"Строка {ln}: пустое имя компонента или лота: "
                f"{line.strip()!r}.")
        if nm in out:
            raise ValueError(
                f"Строка {ln}: компонент '{nm}' указан дважды — лот "
                f"компонента должен быть один.")
        out[nm] = lot
    return out


def material_lots_to_text(lots) -> str:
    """P2.3: лоты → текст формы (обратное к :func:`parse_material_lots`)."""
    return "\n".join(f"{nm}: {lot}" for nm, lot in (lots or {}).items())


def parse_anchor_recipes(text: str) -> Dict[str, Dict[str, float]]:
    """P2.3: разобрать ANCHOR-РЕЦЕПТЫ (phr) из текста формы.

    Формат: одна СТРОКА = один рецепт — «имя: комп=phr, комп=phr, …»
    (как референсный anchor CAMPAIGN_SPEC_PVC §3: ``anchor_main:
    PVC_67=70, PVC_71=30, DINP=10``). Первое «:» отделяет имя; пары —
    через запятую, значение — числом после «=». Здесь только СИНТАКСИС
    (номер строки в ошибке); имена компонентов и диапазоны значений
    проверяет штатный ``set_anchor_recipes``. Чистая (без Streamlit) —
    round-trip с :func:`anchor_recipes_to_text`.
    """
    out: Dict[str, Dict[str, float]] = {}
    for ln, line in enumerate(str(text or "").splitlines(), start=1):
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(
                f"Строка {ln}: ожидается «имя: комп=phr, комп=phr…» "
                f"(разделитель «:»): {line.strip()!r}.")
        rname, rest = line.split(":", 1)
        rname = rname.strip()
        if not rname:
            raise ValueError(f"Строка {ln}: пустое имя anchor-рецепта.")
        if rname in out:
            raise ValueError(
                f"Строка {ln}: anchor-рецепт '{rname}' указан дважды.")
        rec: Dict[str, float] = {}
        for tok in rest.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "=" not in tok:
                raise ValueError(
                    f"Строка {ln}: ожидается «комп=phr», дано {tok!r}.")
            comp, val = tok.split("=", 1)
            comp = comp.strip()
            if not comp:
                raise ValueError(
                    f"Строка {ln}: пустое имя компонента в {tok!r}.")
            if comp in rec:
                raise ValueError(
                    f"Строка {ln}: компонент '{comp}' в рецепте "
                    f"'{rname}' указан дважды.")
            try:
                rec[comp] = float(val)
            except ValueError:
                raise ValueError(
                    f"Строка {ln}: доза '{comp}' не число: {val.strip()!r}.")
        if not rec:
            raise ValueError(
                f"Строка {ln}: anchor-рецепт '{rname}' пуст — укажите "
                f"хотя бы один «комп=phr».")
        out[rname] = rec
    return out


def anchor_recipes_to_text(recipes) -> str:
    """P2.3: anchor-рецепты → текст (обратное к :func:`parse_anchor_recipes`)."""
    return "\n".join(
        f"{rn}: " + ", ".join(f"{c}={float(v):g}" for c, v in rec.items())
        for rn, rec in (recipes or {}).items())


def parse_covariate_names(text: str) -> List[str]:
    """P3.1: разобрать ОБЪЯВЛЕНИЕ ковариат базы из текста формы.

    Формат: имена через запятую / точку-с-запятой / перенос строки
    (``SME, Die_Pressure, торк``). Здесь только СИНТАКСИС (пустые
    отбрасываются); дубли, коллизии с откликами/координатами схемы ловит
    ШТАТНЫЙ :meth:`MixtureProcessRunner.set_covariate_names` (канон iter52:
    второй набор правил в UI разошёлся бы с ядром, A0.6). Чистая (без
    Streamlit) — round-trip с :func:`covariate_names_to_text`."""
    out: List[str] = []
    for line in str(text or "").splitlines():
        out.extend(_parse_names(line))
    return out


def covariate_names_to_text(names) -> str:
    """P3.1: ковариаты → текст формы (обратное к :func:`parse_covariate_names`)."""
    return ", ".join(str(n) for n in (names or []))


def setup_mixture_names(field_names: Sequence[str],
                        spec: Optional[PhrSpec]) -> List[str]:
    """iter41.1: имена компонентов при сборке проекта.

    При АКТИВНОЙ phr-спеке имена берутся из ``spec.component_names``
    (поле «Компоненты смеси» игнорируется) — иначе рассинхрон имён, и
    раннер молча (warning) откатится на бокс-сэмплер. Без спеки — имена
    из поля формы. Чистая (без Streamlit) — тестируется напрямую."""
    if spec is not None:
        return list(spec.component_names)
    return [str(s) for s in field_names]


def campaign_passport_dataframe(runner) -> pd.DataFrame:
    """iter41.3 (+iter52/P2.1-UI, +P2.3): паспорт кампании — phr-спека (q,
    dim_z, hash-префикс 12 симв.), метка, обязательные 2D-пары, ДИСКРЕТНЫЕ
    уровни process-осей, ПОРЯДОК ГРУПП спеки, лоты сырья, anchor-рецепты и
    разрешение весов. «—» = политика не задана (видимость по A0.6, ничего
    не скрываем). Чистая (без Streamlit)."""
    spec = getattr(runner, "phr_spec", None)
    if spec is not None:
        spec_val = (f"q={spec.q}, dim_z={spec.dim_z}, "
                    f"hash {spec.spec_hash()[:12]}…")
    else:
        spec_val = "—"
    label = str(getattr(runner, "campaign_label", "") or "")
    pairs = getattr(runner, "preflight_pairs", []) or []
    pairs_val = preflight_pairs_to_text(pairs).replace("\n", " ; ")
    # iter52: сетка режимов — такая же политика кампании, как метка и пары
    # (что умеет железо), и точно так же обязана быть видна после загрузки.
    levels_val = process_levels_to_text(
        getattr(runner, "process_levels", {}) or {}).replace("\n", " ; ")
    # P3.3: связанные оси — политика «что умеет железо», как уровни.
    links_val = process_links_to_text(
        getattr(runner, "process_links", []) or []).replace("\n", " ; ")
    # P2.3: group_order — READ-ONLY из активной спеки (единый источник —
    # PhrSpec, iter48/B4: порядок входит в spec_hash; отдельного поля в
    # раннере нет — дубль состояния разошёлся бы с отпечатком).
    order = list(getattr(spec, "group_order", []) or []) if spec else []
    # P2.3: лоты сырья / anchor-рецепты / разрешение весов (CAMPAIGN_SPEC_PVC
    # §3: «записать ДО первого замера»).
    lots_val = material_lots_to_text(
        getattr(runner, "material_lots", {}) or {}).replace("\n", " ; ")
    anchors = getattr(runner, "anchor_recipes", {}) or {}
    anchors_val = " ; ".join(f"{rn} ({len(rec)} комп.)"
                             for rn, rec in anchors.items())
    step_g = float(getattr(runner, "weighing_step_g", 0.0) or 0.0)
    gpp = float(getattr(runner, "grams_per_phr", 0.0) or 0.0)
    weigh_val = (f"шаг {step_g:g} г · {gpp:g} г/phr · "
                 f"δ = {step_g / gpp:g} phr" if step_g > 0 and gpp > 0 else "")
    # P3.1: объявленные ковариаты базы (телеметрия прогона) — политика
    # кампании, обязана быть видна после загрузки (как метка/пары/уровни).
    cov_val = covariate_names_to_text(
        getattr(runner, "covariate_names", []) or [])
    return pd.DataFrame([
        {"параметр": "phr-спека (decode-слой)", "значение": spec_val},
        {"параметр": "метка кампании", "значение": label or "—"},
        {"параметр": "обязательные 2D-пары", "значение": pairs_val or "—"},
        {"параметр": "дискретные уровни process-осей",
         "значение": levels_val or "—"},
        {"параметр": "связанные process-оси (разности)",
         "значение": links_val or "—"},
        {"параметр": "порядок групп (group_order)",
         "значение": (" → ".join(order) if order else "—")},
        {"параметр": "лоты сырья", "значение": lots_val or "—"},
        {"параметр": "anchor-рецепты (phr)", "значение": anchors_val or "—"},
        {"параметр": "разрешение весов (δ)", "значение": weigh_val or "—"},
        {"параметр": "ковариаты базы (телеметрия)",
         "значение": cov_val or "—"},
    ])



# ----------------------------------------------------------------------
# iter41.4 — Streamlit-часть иерархического ввода спеки
# ----------------------------------------------------------------------
_PHR_SRC_TREE = "Иерархия (ручной ввод)"
_PHR_SRC_JSON = "JSON / файл"
# iter56/P3.2: схема сериализации дерева — v1 (legacy share_of) или v2
# (роли нового контракта). Схема входит в spec_hash: v1 и v2 одной и той
# же геометрии дают РАЗНЫЕ отпечатки (и разную меру: у v1 каждая доля со
# своей z-осью, у v2 closure/последний simplex-член — производные).
_PHR_SCHEMA_V1 = "v1 (legacy: share_of)"
_PHR_SCHEMA_V2 = "v2 (роли: closure/simplex)"


def _phr_uid(block: Dict[str, Any], fallback: int) -> str:
    """Стабильный идентификатор блока для ключей виджетов.

    Ключи виджетов НЕЛЬЗЯ привязывать к индексу: после ▲/▼ Streamlit
    вернул бы в переставленные блоки старые значения (state живёт по
    ключу). Поэтому каждому блоку выдаётся ``_uid``, переживающий
    перестановку. Служебное поле игнорируется :func:`phr_tree_to_dicts`."""
    uid = str(block.get("_uid", "") or "")
    if not uid:
        uid = f"{fallback}_{abs(zlib.crc32(str(block.get('name', '')).encode()))}"
        block["_uid"] = uid
    return uid


def _render_phr_json_input(key_prefix: str) -> Optional[PhrSpec]:
    """iter41.1: канал «JSON / файл» — uploader (приоритет) + textarea."""
    st.caption(
        "JSON-список узлов в формате PhrSpec.from_dicts "
        "(absolute / share_of / ratio_to / fixed) — тот же формат, что "
        "to_dicts (входит в spec_hash). Поле «Компоненты смеси» выше "
        "ИГНОРИРУЕТСЯ: имена берутся из спеки (листья DAG).")
    up = st.file_uploader("JSON-файл спеки (опц.)", type=["json"],
                          key=f"{key_prefix}_phr_file")
    txt = st.text_area("phr-спека (JSON)", value="", height=220,
                       key=f"{key_prefix}_phr_json")
    if up is not None:
        st.caption("Используется загруженный файл "
                   "(текстовое поле игнорируется).")
        try:
            src = up.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            st.error("Файл спеки не в UTF-8 — сохраните JSON в UTF-8.")
            return None
    else:
        src = txt
    try:
        return parse_phr_spec_json(src)
    except ValueError as exc:
        st.error(str(exc))
        return None


def _render_phr_group_block(blk: Dict[str, Any], uid: str,
                            key_prefix: str,
                            schema_version: int = 1) -> None:
    """iter41.4: поля ГРУППЫ — имя, тотал (absolute/fixed) и таблица долей.

    iter56/P3.2: в схеме v2 таблица детей получает колонки ``min phr`` /
    ``max phr`` (техлимиты, пусто = нет) и ``closure`` (зависимый член
    k=2-группы; у k≥3 все члены — SHARE_SIMPLEX, closure не помечается)."""
    gc = st.columns([3, 2, 2, 2])
    blk["name"] = gc[0].text_input(
        "Имя группы", value=str(blk.get("name", "")),
        key=f"{key_prefix}_phr_gname_{uid}",
        placeholder="например: FILLER.total")
    blk["total_mode"] = gc[1].selectbox(
        "Тотал группы", list(PHR_TOTAL_MODES),
        index=list(PHR_TOTAL_MODES).index(
            str(blk.get("total_mode", "absolute"))),
        key=f"{key_prefix}_phr_gmode_{uid}",
        help="absolute — суммарный phr группы в диапазоне [lo, hi]; "
             "fixed — константа (смола = 100 phr).")
    if str(blk["total_mode"]) == "fixed":
        blk["value"] = gc[2].number_input(
            "phr (константа)", value=float(blk.get("value", 0.0)), step=1.0,
            format="%.4f", key=f"{key_prefix}_phr_gval_{uid}")
    else:
        blk["lo"] = gc[2].number_input(
            "phr lo", value=float(blk.get("lo", 0.0)), step=0.5,
            format="%.4f", key=f"{key_prefix}_phr_glo_{uid}")
        blk["hi"] = gc[3].number_input(
            "phr hi", value=float(blk.get("hi", 0.0)), step=0.5,
            format="%.4f", key=f"{key_prefix}_phr_ghi_{uid}")

    st.caption("Компоненты группы и их ДОЛИ группы (0…1). Сумма нижних ≤ 1 ≤ "
               "сумма верхних — иначе конструктор спеки откажет. Порядок "
               "строк = порядок узлов (входит в spec_hash)."
               + (" v2: closure — зависимый член k=2-группы (доли L/U "
                  "оставьте 0 — диапазон производный); min/max phr — "
                  "техлимиты узла в phr (пусто = нет)."
                  if int(schema_version) == 2 else ""))
    # dkey несёт версию схемы: при переключении v1↔v2 кэш таблицы со старым
    # составом колонок должен пересобраться, а не показывать урезанный вид.
    dkey = f"{key_prefix}_phr_kids_df_{uid}_v{int(schema_version)}"
    if dkey not in st.session_state:
        st.session_state[dkey] = phr_children_dataframe(
            blk, schema_version=schema_version)
    edited = st.data_editor(st.session_state[dkey], num_rows="dynamic",
                            use_container_width=True, hide_index=True,
                            key=f"{key_prefix}_phr_kids_{uid}_v"
                                f"{int(schema_version)}")
    try:
        blk["children"] = phr_children_from_dataframe(
            edited, schema_version=schema_version)
    except ValueError as exc:
        st.error(str(exc))


def _render_phr_single_block(blk: Dict[str, Any], uid: str,
                             key_prefix: str,
                             schema_version: int = 1) -> None:
    """iter41.4: поля ОДИНОЧНОГО узла (absolute / fixed / ratio_to).

    iter56/P3.2: в схеме v2 у absolute-узла доступна шкала сэмплинга
    ``scale`` (linear | log, iter47/B5)."""
    sc = st.columns([3, 2, 2, 2])
    blk["name"] = sc[0].text_input(
        "Имя компонента", value=str(blk.get("name", "")),
        key=f"{key_prefix}_phr_sname_{uid}", placeholder="например: DINP")
    blk["mode"] = sc[1].selectbox(
        "Режим", list(PHR_SINGLE_MODES),
        index=list(PHR_SINGLE_MODES).index(str(blk.get("mode", "absolute"))),
        key=f"{key_prefix}_phr_smode_{uid}",
        help="absolute — phr в [lo, hi]; fixed — константа; ratio_to — "
             "коэффициент [lo, hi] к другому узлу (SBM = 0.02…0.09 × "
             "Σ стабилизатора).")
    md = str(blk["mode"])
    if md == "fixed":
        blk["value"] = sc[2].number_input(
            "phr (константа)", value=float(blk.get("value", 0.0)), step=0.1,
            format="%.4f", key=f"{key_prefix}_phr_sval_{uid}")
        return
    lo_label = "коэф. lo" if md == "ratio_to" else "phr lo"
    hi_label = "коэф. hi" if md == "ratio_to" else "phr hi"
    blk["lo"] = sc[2].number_input(
        lo_label, value=float(blk.get("lo", 0.0)), step=0.05, format="%.4f",
        key=f"{key_prefix}_phr_slo_{uid}")
    blk["hi"] = sc[3].number_input(
        hi_label, value=float(blk.get("hi", 0.0)), step=0.05, format="%.4f",
        key=f"{key_prefix}_phr_shi_{uid}")
    if md == "ratio_to":
        blk["ref"] = st.text_input(
            "К узлу (ratio_to)", value=str(blk.get("ref", "")),
            key=f"{key_prefix}_phr_sref_{uid}",
            placeholder="например: STAB.total")
        return
    cc = st.columns([3, 2])
    caps_now = _phr_cap_refs(blk.get("cap_to", ""))
    blk["cap_to"] = cc[0].text_input(
        "Динамический потолок: узлы (через запятую, опц.)",
        value=", ".join(caps_now), key=f"{key_prefix}_phr_scap_{uid}",
        help="Потолок по СУММЕ указанных узлов (фазе): "
             "hi_eff = min(hi, cap_ratio · Σ value). Пример: DINP, ESO.")
    blk["cap_ratio"] = cc[1].number_input(
        "cap_ratio", value=float(blk.get("cap_ratio", 0.0)), step=0.01,
        format="%.4f", key=f"{key_prefix}_phr_scapr_{uid}")
    if int(schema_version) == 2:
        blk["scale"] = st.selectbox(
            "Шкала оси (v2)", ["linear", "log"],
            index=(1 if str(blk.get("scale", "linear")) == "log" else 0),
            key=f"{key_prefix}_phr_sscale_{uid}",
            help="log — сэмплинг равномерен по ln phr (iter47/B5: вся "
                 "информация сатурирующих осей TiO2/УФ — в нижней декаде); "
                 "требует lo > 0. Шкала входит в spec_hash.")


def _render_phr_tree_input(key_prefix: str) -> Optional[PhrSpec]:
    """iter41.4: канал «Иерархия» — группы с компонентами + одиночные узлы.

    Дерево живёт в ``session_state[f"{key_prefix}_phr_tree"]`` и правится
    на месте; порядок блоков меняют кнопки ▲/▼ (он входит в ``spec_hash``).
    Возвращает спеку или ``None``, если ввод пока невалиден (ошибка уже
    показана пользователю)."""
    tkey = f"{key_prefix}_phr_tree"
    tree: List[Dict[str, Any]] = st.session_state.setdefault(tkey, [])
    st.caption(
        "Рецептура вводится как в лаборатории: ГРУППА (суммарный phr) → "
        "компоненты группы в ДОЛЯХ, плюс одиночные компоненты вне групп. "
        "Поле «Компоненты смеси» выше ИГНОРИРУЕТСЯ — имена берутся отсюда. "
        "Порядок узлов входит в spec_hash: переставляйте кнопками ▲/▼.")
    # iter56/P3.2: выбор схемы сериализации — контракт кампании, а не
    # косметика: v1 и v2 дают разные z-оси (у v2 closure без оси) и разные
    # отпечатки. Дерево одной геометрии в v1 и v2 — РАЗНЫЕ спеки.
    sv_label = st.radio(
        "Схема спеки", [_PHR_SCHEMA_V1, _PHR_SCHEMA_V2],
        key=f"{key_prefix}_phr_schema", horizontal=True,
        help="v1 — legacy share_of: каждая доля со своей z-осью (пары "
             "(φ, 1−φ) коллинеарны). v2 — роли нового контракта: k=2 → "
             "free+closure, k≥3 → simplex (замыкание без z-оси), техлимиты "
             "min/max phr, scale='log', group_order. Схема входит в "
             "spec_hash — v1 и v2 дают РАЗНЫЕ отпечатки.")
    sv = 2 if sv_label == _PHR_SCHEMA_V2 else 1
    group_order: List[str] = []
    if sv == 2:
        go_text = st.text_input(
            "group_order — приоритет GROUP_TOTAL-групп (через запятую, опц.)",
            key=f"{key_prefix}_phr_group_order",
            placeholder="например: FILLER.total, SOFT.total",
            help="ТОЧНАЯ перестановка множества GROUP_TOTAL-групп "
                 "(CAMPAIGN_SPEC_PVC §1, iter48/B4) — приоритет осей "
                 "кампании; входит в spec_hash. Пусто — не задан. "
                 "Валидацию делает конструктор спеки.")
        group_order = _parse_names(go_text)

    ac = st.columns([2, 2, 1])
    if ac[0].button("➕ Группа", key=f"{key_prefix}_phr_add_group",
                    use_container_width=True):
        tree.append(phr_group_block(f"группа{len(tree) + 1}"))
        st.session_state[tkey] = tree
        st.rerun()
    if ac[1].button("➕ Компонент вне групп",
                    key=f"{key_prefix}_phr_add_single",
                    use_container_width=True):
        tree.append(phr_single_block(f"компонент{len(tree) + 1}"))
        st.session_state[tkey] = tree
        st.rerun()
    if ac[2].button("🧹", key=f"{key_prefix}_phr_clear",
                    help="Очистить спеку целиком"):
        st.session_state[tkey] = []
        for k in [k for k in st.session_state
                  if str(k).startswith(f"{key_prefix}_phr_kids_df_")]:
            st.session_state.pop(k, None)
        st.rerun()

    if not tree:
        st.info("Спека пуста: добавьте первую группу (например, "
                "«RESIN.total» = 100 phr) или одиночный компонент.")
        return None

    for i, blk in enumerate(tree):
        uid = _phr_uid(blk, i)
        is_group = str(blk.get("kind")) == "group"
        hc = st.columns([6, 1, 1, 1])
        hc[0].markdown(f"**{i + 1}. {'📦 группа' if is_group else '🔹 узел'} "
                       f"— {blk.get('name', '')}**")
        if hc[1].button("▲", key=f"{key_prefix}_phr_up_{uid}",
                        disabled=(i == 0)):
            st.session_state[tkey] = phr_tree_move(tree, i, -1)
            st.rerun()
        if hc[2].button("▼", key=f"{key_prefix}_phr_dn_{uid}",
                        disabled=(i == len(tree) - 1)):
            st.session_state[tkey] = phr_tree_move(tree, i, +1)
            st.rerun()
        if hc[3].button("🗑", key=f"{key_prefix}_phr_del_{uid}"):
            st.session_state[tkey] = [b for j, b in enumerate(tree) if j != i]
            for k in [k for k in st.session_state if str(k).startswith(
                    f"{key_prefix}_phr_kids_df_{uid}")]:
                st.session_state.pop(k, None)
            st.rerun()
        if is_group:
            _render_phr_group_block(blk, uid, key_prefix, sv)
        else:
            _render_phr_single_block(blk, uid, key_prefix, sv)
        st.divider()

    st.session_state[tkey] = tree
    try:
        dicts = phr_tree_to_dicts(tree, schema_version=sv,
                                  group_order=group_order)
    except ValueError as exc:      # ошибки дерева (пустая группа, дубли…)
        st.error(str(exc))
        return None
    try:
        return PhrSpec.from_dicts(dicts)
    except ValueError as exc:      # ошибки конструктора (Σдолей, ссылки…)
        st.error(str(exc))
        return None


def render_composition_bounds(names: Sequence[str], *, key_prefix: str = "setup"):
    """§17.4 (замечание 1): ограничения состава — «Доли (0…1)» ИЛИ «Массовые части
    (база = 100)». Возвращает ``(lower, upper)`` в ДОЛЯХ или ``(None, None)``.

    Форма-близнец сайдбара pipeline (`streamlit_app._composition_bounds`):
    экспандер с радио «Способ ввода» и попарными полями ``L·X`` / ``U·X`` по
    каждому компоненту (а не «через запятую»). В режиме «Массовые части» базовый
    компонент = 100 частей, остальные задаются диапазоном частей → доли считает
    каноничная :func:`core.simplex.parts_ranges_to_fraction_bounds` (tightest box,
    та же математика, что в pipeline). Пустой список компонентов — ``(None, None)``.
    """
    q = len(names)
    if q == 0:
        return None, None
    # NB: без st.expander — форма зовётся ВНУТРИ экспандера сетапа, а Streamlit
    # запрещает вложенные экспандеры; используем контейнер с заголовком.
    st.markdown("**📐 Ограничения состава (опц.)**")
    with st.container():
        mode = st.radio(
            "Способ ввода",
            ["Доли (0…1)", "Массовые части (база = 100)", _MODE_PHR],
            key=f"{key_prefix}_comp_mode",
            help="«Массовые части»: базовый компонент = 100 частей, остальные "
                 "задаются диапазоном частей, а доли (и плавающий диапазон доли "
                 "базы) считаются автоматически. «Доли»: границы доли каждого "
                 "компонента 0…1. «phr-спека (JSON)»: DAG-спека parts/phr "
                 "(decode-слой iter33) — имена компонентов и границы долей "
                 "возьмутся из спеки.")
        spec_key = f"{key_prefix}_phr_spec_obj"
        if mode == _MODE_PHR:
            # iter41.1/41.4: два РАВНОПРАВНЫХ канала одной и той же спеки —
            # иерархический ручной ввод (по умолчанию) и JSON/файл. Оба
            # сходятся в PhrSpec, дальше показ и возврат общие.
            src_mode = st.radio(
                "Ввод спеки", [_PHR_SRC_TREE, _PHR_SRC_JSON],
                key=f"{key_prefix}_phr_src", horizontal=True,
                help="«Иерархия» — задать группы и компоненты руками (порядок "
                     "узлов меняется кнопками ▲/▼). «JSON / файл» — вставить "
                     "или загрузить готовую спеку в формате "
                     "PhrSpec.from_dicts.")
            spec = (_render_phr_tree_input(key_prefix)
                    if src_mode == _PHR_SRC_TREE
                    else _render_phr_json_input(key_prefix))
            if spec is None:
                st.session_state[spec_key] = None
                return None, None
            st.session_state[spec_key] = spec
            st.caption(f"Компоненты смеси из спеки ({spec.q}): "
                       f"{', '.join(spec.component_names)} · z-осей: "
                       f"{spec.dim_z}.")
            st.dataframe(phr_spec_summary_dataframe(spec),
                         use_container_width=True, hide_index=True)
            # iter50/P1.3: политика геометрии (роли уже в таблице выше) —
            # порядок групп, лог-оси и техлимиты входят в spec_hash, но были
            # не видны: две «одинаковые» спеки давали разные планы.
            st.caption(phr_spec_policy_caption(spec))
            st.caption("Интервалы phr компонентов и рассчитанные ДОЛИ для "
                       "mixture-блока схемы (fraction_bounds):")

            st.dataframe(phr_spec_fraction_dataframe(spec),
                         use_container_width=True)
            st.code(spec.spec_hash(), language=None)
            st.caption("spec_hash активной спеки: зафиксируйте хеш и лоты "
                       "сырья ДО первого замера (CAMPAIGN_SPEC_PVC §3) — "
                       "задним числом не восстанавливается.")
            lo_arr, hi_arr = spec.fraction_bounds()
            return lo_arr.tolist(), hi_arr.tolist()
        st.session_state[spec_key] = None
        if mode.startswith("Доли"):
            st.caption("Доли каждого компонента (0…1). Сумма нижних ≤ 1 ≤ сумма "
                       "верхних. Оставьте 0…1, если ограничений нет.")
            lower: List[float] = []
            upper: List[float] = []
            for i in range(q):
                cc = st.columns(2)
                lo_i = cc[0].number_input(
                    f"L · {names[i]}", min_value=0.0, max_value=1.0, value=0.0,
                    step=0.01, format="%.4f", key=f"{key_prefix}_lo_{q}_{i}")
                hi_i = cc[1].number_input(
                    f"U · {names[i]}", min_value=0.0, max_value=1.0, value=1.0,
                    step=0.01, format="%.4f", key=f"{key_prefix}_hi_{q}_{i}")
                lower.append(float(lo_i))
                upper.append(float(hi_i))
            nontrivial = any(l > 0 for l in lower) or any(u < 1 for u in upper)
            return (lower, upper) if nontrivial else (None, None)

        # --- режим массовых частей (база = 100) ---
        base_i = st.selectbox(
            "Базовый компонент (= 100 частей)", list(range(q)),
            format_func=lambda i: names[i], key=f"{key_prefix}_base_{q}")
        st.caption("Диапазон массовых частей для остальных компонентов "
                   "(база фиксирована = 100 частей):")
        pmin = [0.0] * q
        pmax = [0.0] * q
        for i in range(q):
            if i == base_i:
                pmin[i] = pmax[i] = 100.0
                st.markdown(f"**{names[i]}** — база: 100 частей (фиксировано)")
                continue
            cc = st.columns(2)
            pmin[i] = cc[0].number_input(
                f"min частей · {names[i]}", min_value=0.0, value=0.0, step=1.0,
                key=f"{key_prefix}_pmin_{q}_{i}")
            pmax[i] = cc[1].number_input(
                f"max частей · {names[i]}", min_value=0.0, value=100.0, step=1.0,
                key=f"{key_prefix}_pmax_{q}_{i}")
        try:
            lo_arr, hi_arr = parts_ranges_to_fraction_bounds(pmin, pmax)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Не удалось пересчитать части в доли: {exc}")
            return None, None
        tbl = pd.DataFrame(
            {"min частей": pmin, "max частей": pmax,
             "доля L": np.round(lo_arr, 4), "доля U": np.round(hi_arr, 4)},
            index=list(names)[:q])
        st.caption("Рассчитанные диапазоны долей для алгоритма:")
        st.dataframe(tbl, use_container_width=True)
        return lo_arr.tolist(), hi_arr.tolist()


def render_process_bounds(names: Sequence[str], *, key_prefix: str = "setup"):
    """§17.4 (замечание 2): границы процесс-параметров — попарные поля L/U на КАЖДЫЙ
    параметр в РЕАЛЬНЫХ единицах (интерфейс-близнец ограничений состава).

    Заменяет ввод «через запятую»: для каждого параметра — своя строка с нижней и
    верхней РЕАЛЬНОЙ границей (T=150…200 °C, P=1…5 бар); понятнее и меньше ошибок.
    Нормировку в код [0,1] делает движок сам. Возвращает ``(lower, upper)``
    списками реальных величин или ``(None, None)``, если параметров нет. Первые
    две оси получают осмысленные дефолты (T=150…200, P=1…5), остальные — 0…1.
    """
    d = len(names)
    if d == 0:
        return None, None
    st.markdown("**⚙️ Границы процесс-параметров (реальные единицы)**")
    st.caption("Для каждого параметра — нижняя и верхняя РЕАЛЬНАЯ граница "
               "(например, T = 150…200 °C, P = 1…5 бар). Нормировку в код [0,1] "
               "программа делает сама (замечание 2).")
    _defaults = {0: (150.0, 200.0), 1: (1.0, 5.0)}
    lower: List[float] = []
    upper: List[float] = []
    for i in range(d):
        dlo, dhi = _defaults.get(i, (0.0, 1.0))
        cc = st.columns([1, 2, 2])
        cc[0].markdown(f"**{names[i]}**")
        lo_i = cc[1].number_input(
            f"нижняя · {names[i]}", value=float(dlo), step=1.0, format="%.4f",
            key=f"{key_prefix}_plo_{d}_{i}")
        hi_i = cc[2].number_input(
            f"верхняя · {names[i]}", value=float(dhi), step=1.0, format="%.4f",
            key=f"{key_prefix}_phi_{d}_{i}")
        lower.append(float(lo_i))
        upper.append(float(hi_i))
    return lower, upper


def setup_prefill_from_runner(runner) -> Dict[str, Any]:
    """C2 (§17.6.1): значения формы сетапа из ЗАГРУЖЕННОГО проекта.

    Обратная проекция «раннер → ключи виджетов формы "🆕 Новый проект"»: имена
    компонентов/процесс-параметров/откликов, границы долей (режим «Доли (0…1)» —
    массовые части не сериализуются, честно показываем рассчитанные доли),
    реальные границы процесс-осей и seed раннера. Без неё форма после загрузки
    проекта показывала дефолты («A, B, C», 0…1, 150…200), и пользователь видел
    «настройки не подтянулись» (A0.6). Чистая (без Streamlit) — тестируется
    напрямую; применяется отложенно через ключ ``setup_prefill_pending``
    (Streamlit запрещает менять ключ виджета, созданного в том же прогоне).
    """
    sch = runner.current_schema
    mix = list(sch.mixture_names)
    proc = list(sch.process_names)
    q, d = len(mix), len(proc)
    out: Dict[str, Any] = {
        "setup_mix": ", ".join(mix),
        "setup_resp": ", ".join(str(p) for p in runner.property_names),
        "setup_proc": ", ".join(proc),
        "setup_seed": int(getattr(runner, "seed", 0)),
        "setup_comp_mode": "Доли (0…1)",
    }
    mb = sch.mixture_block()
    if mb is not None:
        for i in range(q):
            out[f"setup_lo_{q}_{i}"] = float(mb.lower[i])
            out[f"setup_hi_{q}_{i}"] = float(mb.upper[i])
    pb = sch.process_block()
    if pb is not None:
        for i in range(d):
            out[f"setup_plo_{d}_{i}"] = float(pb.lower[i])
            out[f"setup_phi_{d}_{i}"] = float(pb.upper[i])
    # iter31: функциональные группы компонентов (проектная политика)
    out["setup_groups"] = sampling_groups_to_text(
        getattr(runner, "sampling_groups", []) or [])
    # blocking: число партий + фактор/имена блоков (виджеты seed-секции)
    out["setup_seed_blocks"] = int(getattr(runner, "n_blocks_start", 1))
    out["setup_block_factor"] = str(getattr(runner, "block_factor", "") or "")
    for b, nm in (getattr(runner, "block_names", {}) or {}).items():
        out[f"setup_block_name_{int(b)}"] = str(nm)
    # iter41.3: паспорт кампании (метка + обязательные 2D-пары) — политика
    # записывается ДО первого замера и обязана отражаться после загрузки.
    out["setup_campaign_label"] = str(
        getattr(runner, "campaign_label", "") or "")
    out["setup_preflight_pairs"] = preflight_pairs_to_text(
        getattr(runner, "preflight_pairs", []) or [])
    # P2.3: паспорт кампании — лоты/anchor'ы/весы обязаны отражаться в форме
    # после загрузки (иначе повторная сборка проекта молча стёрла бы паспорт).
    out["setup_material_lots"] = material_lots_to_text(
        getattr(runner, "material_lots", {}) or {})
    out["setup_anchor_recipes"] = anchor_recipes_to_text(
        getattr(runner, "anchor_recipes", {}) or {})
    out["setup_pass_weigh_step"] = float(
        getattr(runner, "weighing_step_g", 0.0) or 0.0)
    out["setup_pass_weigh_gpp"] = float(
        getattr(runner, "grams_per_phr", 0.0) or 0.0)
    # iter52/P2.1-UI: дискретные уровни process-осей — политика кампании,
    # которую после загрузки обязана показывать и ФОРМА (иначе повторная
    # сборка проекта молча вернула бы непрерывные оси).
    out["setup_process_levels"] = process_levels_to_text(
        getattr(runner, "process_levels", {}) or {})
    # P3.3: связки осей — политика кампании; без префилла повторная сборка
    # проекта молча вернула бы независимые оси.
    out["setup_process_links"] = process_links_to_text(
        getattr(runner, "process_links", []) or [])
    # P3.1: объявленные ковариаты базы — тоже политика кампании: без
    # префилла повторная сборка проекта молча стёрла бы объявление.
    out["setup_covariates"] = covariate_names_to_text(
        getattr(runner, "covariate_names", []) or [])

    # iter41.3: активная phr-спека → режим «phr-спека (JSON)» с каноническим
    # JSON to_dicts (round-trip: parse_phr_spec_json(json).spec_hash() == hash).
    spec = getattr(runner, "phr_spec", None)
    if spec is not None:
        out["setup_comp_mode"] = _MODE_PHR
        out["setup_phr_json"] = json.dumps(spec.to_dicts(),
                                           ensure_ascii=False, indent=2)
        # iter41.4: то же самое деревом — чтобы иерархический канал показывал
        # загруженную спеку, а не пустую форму (round-trip сохраняет hash).
        # iter56/P3.2: дерево проецирует ОБЕ схемы (закрыт хвост iter46);
        # схема и group_order префиллятся своими полями формы.
        ver = int(getattr(spec, "schema_version", 1))
        out["setup_phr_schema"] = (_PHR_SCHEMA_V2 if ver == 2
                                   else _PHR_SCHEMA_V1)
        if ver == 2:
            out["setup_phr_group_order"] = ", ".join(
                getattr(spec, "group_order", []) or [])
        try:
            out["setup_phr_tree"] = phr_tree_from_spec(spec)
        except ValueError:
            # Непроецируемый порядок узлов v2 (члены группы не сразу за
            # тоталом): дерево не префиллим — канал JSON выше остаётся
            # полным источником истины, spec_hash не трогаем (A0.6).
            pass
    return out


def project_settings_dataframe(runner) -> pd.DataFrame:
    """Сводка настроек ТЕКУЩЕГО проекта: переменная / тип / границы L…U /
    уровни.

    Показывается после сборки/загрузки проекта, чтобы пользователь видел, какие
    доли компонентов и реальные границы процесс-параметров действуют в движке
    (единый источник истины — ``runner.current_schema``, а не поля формы).

    iter52/P2.1-UI: колонка «уровни» помечает ДИСКРЕТНЫЕ process-оси
    (``rotor_rpm: 400, 900``) — иначе строка «400…900» читалась бы как
    непрерывный интервал, хотя план и оптимум выдаются только на сетке.
    У компонентов смеси ячейка пустая (правило неприменимо), у непрерывных
    process-осей — явное «непрерывная».

    Чистая (без Streamlit) — тестируется напрямую."""
    rows: List[Dict[str, Any]] = []
    sch = runner.current_schema
    levels = dict(getattr(runner, "process_levels", {}) or {})
    mb = sch.mixture_block()
    if mb is not None:
        for nm, lo, hi in zip(mb.names, mb.lower, mb.upper):
            rows.append({"переменная": str(nm), "тип": "компонент смеси (доля)",
                         "нижняя": float(lo), "верхняя": float(hi),
                         "уровни": ""})
    pb = sch.process_block()
    if pb is not None:
        for nm, lo, hi in zip(pb.names, pb.lower, pb.upper):
            lv = levels.get(str(nm))
            rows.append({"переменная": str(nm),
                         "тип": "процесс-параметр (реальные единицы)",
                         "нижняя": float(lo), "верхняя": float(hi),
                         "уровни": (", ".join(f"{float(v):g}" for v in lv)
                                    if lv else "непрерывная")})
    return pd.DataFrame(rows)



def render_project_settings(runner) -> None:
    """📋 Панель «настройки проекта» (читает движок, не форму).

    C2: после загрузки проекта именно здесь видно, что доли компонентов,
    границы процесс-параметров и отклики ПОДТЯНУЛИСЬ (двойник формы сетапа,
    но по фактическому состоянию раннера)."""
    with st.expander("📋 Настройки проекта"):
        st.caption(
            f"Отклики: {', '.join(runner.property_names)} · схема v"
            f"{int(runner.current_schema_version)} · seed раннера = "
            f"{int(getattr(runner, 'seed', 0))} · общая база: "
            f"{len(runner.points)} точек. Границы ниже — те, по которым "
            "реально работает движок (сохраняются и загружаются с проектом).")
        groups = getattr(runner, "sampling_groups", []) or []
        if groups:
            st.caption("Функциональные группы (стратификация суммы ниши, "
                       "iter31): "
                       + " · ".join("{" + ", ".join(g) + "}" for g in groups))
        # iter52/P2.1-UI: дискретные оси — политика кампании «что умеет
        # железо»; тут состояние показывается ЯВНО (в т.ч. «все непрерывны»),
        # чтобы после загрузки было видно, действует сетка или нет.
        st.caption(levels_caption(getattr(runner, "process_levels", {}) or {}))
        # P3.3: связанные оси — то же правило видимости, что у уровней.
        st.caption(links_caption(getattr(runner, "process_links", []) or []))
        st.dataframe(project_settings_dataframe(runner),
                     use_container_width=True, hide_index=True)
        # iter41.3: паспорт кампании — phr-спека / метка / пары (read-only).

        st.caption("🪪 Паспорт кампании (CAMPAIGN_SPEC_PVC §3):")
        st.dataframe(campaign_passport_dataframe(runner),
                     use_container_width=True, hide_index=True)
        spec = getattr(runner, "phr_spec", None)
        if spec is not None:
            st.code(spec.spec_hash(), language=None)
            st.caption("spec_hash активной phr-спеки (полный hex) — сверяйте "
                       "с зафиксированным в документации кампании.")
            # iter50/P1.3: геометрия спеки видна и ПОСЛЕ загрузки проекта —
            # роли/техлимиты/шкалы/порядок групп, а не только хеш.
            st.caption(phr_spec_policy_caption(spec))
            st.dataframe(phr_spec_summary_dataframe(spec),
                         use_container_width=True, hide_index=True)



def render_setup_form() -> None:

    """§17.4 (Ш3b): форма РЕАЛЬНОГО сетапа — mixture + процесс + отклики.


    По кнопке строит :class:`MixtureProcessRunner` (:class:`ManualOracle`, пустая
    база) и кладёт :class:`CampaignController` в ``session_state`` под тем же
    ключом, что и демо-кампания (главный поток §17 — один движок). Реального
    оракула нет: стартовые отклики вносит пользователь в ручном seed-цикле ниже.

    UX: форма живёт в ЛЕВОЙ панели (сайдбар) рядом с «📁 Проект» — основная
    область не загромождается; кнопка «🏗 Построить проект» — ВНУТРИ формы
    (это её submit: собирает раннер из введённых полей), поэтому не висит
    отдельной сиротой на странице.
    """
    # C2: отложенный префилл формы из ЗАГРУЖЕННОГО проекта — применяем ДО
    # инстанцирования виджетов формы (ключ кладёт загрузчик в сайдбаре;
    # менять session_state виджета можно только до его создания в прогоне).
    _pending = st.session_state.pop("setup_prefill_pending", None)
    if _pending:
        for _k, _v in dict(_pending).items():
            st.session_state[_k] = _v
    with st.sidebar.expander("🆕 Новый проект — реальный сетап (§17.4)",
                             expanded=get_campaign_controller() is None):
        st.caption(
            "Составная область СРАЗУ: симплекс компонентов смеси (Σ=1) × куб "
            "процесс-параметров (реальные единицы). Отклики (свойства) меряются "
            "вручную — оракула-симулятора нет (кнопка «Заполнить тестовыми» в "
            "seed-цикле оставлена для прогонов без лаборатории, A0.6).")
        mix_txt = st.text_input("Компоненты смеси (через запятую)",
                                value="A, B, C", key="setup_mix")
        resp_txt = st.text_input("Отклики / свойства (через запятую)",
                                 value="strength, gloss, rho", key="setup_resp")

        # Замечание 1: ограничения состава — форма «Доли / Массовые части» с
        # попарными полями L·X / U·X по каждому компоненту (форма-близнец сайдбара
        # pipeline). Части → доли каноничной parts_ranges_to_fraction_bounds.
        mix_live = _parse_names(mix_txt)
        mlo, mhi = render_composition_bounds(mix_live, key_prefix="setup")
        # iter41.1: активная phr-спека (разобрана формой выше) и признак режима.
        phr_mode = str(st.session_state.get("setup_comp_mode", "")) == _MODE_PHR
        phr_spec_live: Optional[PhrSpec] = (
            st.session_state.get("setup_phr_spec_obj") if phr_mode else None)

        # iter31: функциональные группы — априорное химическое знание «эти
        # компоненты — одна ниша». Включает стратифицированное сэмплирование по
        # СУММЕ группы: без него равномерная выборка не достаёт края диапазона
        # суммарной дозы ниши (Beta-концентрация), план не покрывает главную ось.
        # iter41.1: при активной phr-спеке группы к сэмплингу НЕ применяются
        # (_phase_candidates идёт phr-путём) — поле скрыто с подписью.
        if phr_mode:
            groups_txt = ""
            st.caption("Функциональные группы (iter31) в режиме phr-спеки не "
                       "применяются: сэмплинг кандидатов идёт decode-путём "
                       "спеки (iter33), стратификация групп — только для "
                       "box-режимов.")
        else:
            groups_txt = st.text_area(
                "Функциональные группы компонентов (опц.)", value="",
                key="setup_groups",
                help="Одна строка = одна группа: имена компонентов через "
                     "запятую (например, конкурирующие пластификаторы одной "
                     "ниши). Стартовый дизайн будет равномерно покрывать "
                     "СУММАРНУЮ дозу каждой группы, а не только середину "
                     "диапазона. Группы не должны пересекаться. Пусто — без "
                     "группировки.")

        proc_txt = st.text_input("Процесс-параметры (через запятую)",
                                 value="T, P", key="setup_proc")
        # Замечание 2: границы процесса — попарные поля L/U на каждый параметр в
        # РЕАЛЬНЫХ единицах (форма-близнец ограничений состава), а не «через
        # запятую» — понятнее и меньше ошибок. Нормировку в код [0,1] движок
        # делает сам.
        proc_live = _parse_names(proc_txt)
        plo, phi = render_process_bounds(proc_live, key_prefix="setup")

        # iter52/P2.1-UI: ДИСКРЕТНЫЕ уровни process-осей (что умеет железо).
        # Ядро (iter51) снапит и план, и argmax на сетку — но задать её было
        # нечем: план предлагал 673 об/мин, оператор ставил 900 (A0.6).
        levels_txt = st.text_area(
            "Дискретные уровни process-осей (опц.)", value="",
            key="setup_process_levels",
            help="Одна строка = одна ось: «имя: уровень, уровень…» в РЕАЛЬНЫХ "
                 "единицах (например, «rotor_rpm: 400, 900» — две передачи "
                 "экструдера). Ось без строки остаётся НЕПРЕРЫВНОЙ. И план, и "
                 "рекомендованный оптимум будут выдаваться ТОЛЬКО в этих "
                 "режимах — иначе лаборатория ставит ближайший достижимый, а "
                 "модель учится на другом значении.")

        # P3.3: связанные оси — производная величина = разность двух осей с
        # полосой реализуемости по железу (dT_head = T_адаптер − T_пласт).
        links_txt = st.text_area(
            "Связанные process-оси (опц.)", value="",
            key="setup_process_links",
            help="Одна строка = одна связка: «имя: осьA - осьB : lo, hi» в "
                 "РЕАЛЬНЫХ единицах (например, «dT_head: T_adapter - "
                 "T_plast : 10, 60» — перепад в голове экструдера, который "
                 "держит нагреватель). Открытая сторона — «*». И план, и "
                 "оптимум будут выдаваться только с реализуемой разностью; "
                 "ось не может одновременно быть в связке и на дискретных "
                 "уровнях.")

        # iter41.2: паспорт кампании — записать ДО первого замера

        # (CAMPAIGN_SPEC_PVC §3: задним числом не восстанавливается).
        st.markdown("**🪪 Паспорт кампании (опц., CAMPAIGN_SPEC_PVC §3)**")
        st.caption("Заполните ДО первого замера: метаданные точек задним "
                   "числом не восстанавливаются.")
        label_txt = st.text_input(
            "Метка кампании", value="", key="setup_campaign_label",
            help="Пишется в origin_tag каждой новой точки (вместе со "
                 "spec_hash активной phr-спеки и номером партии) — без неё "
                 "блочный дрейф при staged-расширении не отделить.")
        pairs_txt = st.text_area(
            "Обязательные 2D-пары", value="", key="setup_preflight_pairs",
            help="Гейт pair-coverage в preflight. Одна строка = пара, стороны "
                 "через «|», ось-сумма — имена через запятую. Например:\n"
                 "UV_CSFCP | TiO2_BLR895\n"
                 "T | PMPlus_8, DL_531")
        # P2.3: лоты сырья, anchor-рецепты и разрешение весов — часть
        # паспорта (CAMPAIGN_SPEC_PVC §3: записать ДО первого замера).
        lots_txt = st.text_area(
            "Лоты сырья (опц.)", value="", key="setup_material_lots",
            help="Партия (лот) сырья каждого компонента: одна строка = "
                 "«компонент: лот» (например, «TiO2_BLR895: L-2408-17»). "
                 "Постфактум не восстановить, какая партия стояла за точкой "
                 "— лотовый дрейф припишется составу.")
        anchors_txt = st.text_area(
            "Anchor-рецепты, phr (опц.)", value="",
            key="setup_anchor_recipes",
            help="Реперные производственные рецепты в phr: одна строка = "
                 "«имя: комп=phr, комп=phr, …» (например, «anchor_main: "
                 "PVC_67=70, PVC_71=30, DINP=10»). Против anchor'а сверяется "
                 "round-trip спеки и дрейф между фазами.")
        pwc = st.columns(2)
        pass_step = pwc[0].number_input(
            "Шаг весов, г (паспорт)", min_value=0.0, value=0.0, step=0.01,
            format="%.4f", key="setup_pass_weigh_step",
            help="Дискретность лабораторных весов — дефолт слоя навески "
                 "(iter42). 0 у ОБОИХ полей = не задано; заполнено ОДНО из "
                 "двух — явная ошибка (δ не определим).")
        pass_gpp = pwc[1].number_input(
            "г на 1 phr (паспорт)", min_value=0.0, value=0.0, step=0.5,
            format="%.4f", key="setup_pass_weigh_gpp",
            help="Загрузка смесителя: сколько граммов приходится на 1 phr. "
                 "Вместе с шагом весов даёт δ_phr = шаг / (г на 1 phr).")
        # P3.1: ковариаты базы — телеметрия прогона (M(t)/SME, Die_Pressure,
        # торк, вытяжка, наработка вала): столбцы базы, НЕ отклики модели.
        cov_txt = st.text_input(
            "Ковариаты базы — телеметрия прогона (опц.)", value="",
            key="setup_covariates",
            help="Имена через запятую (например: SME, Die_Pressure, торк, "
                 "вытяжка, наработка_вала). Это СТОЛБЦЫ общей базы, а не "
                 "отклики: в модель/суррогаты не входят, желательности не "
                 "несут, но записываются при каждом замере — постфактум "
                 "телеметрию прогона не восстановить. Значения вносятся в "
                 "таблицах seed/добора (пустые ячейки допустимы).")

        seed_v = st.number_input(

            "Seed раннера (зерно ГСЧ проекта)", value=1, step=1, key="setup_seed",
            help="Зерно генератора случайных чисел движка проекта: фиксирует "
                 "воспроизводимость стартового дизайна и внутренних рестартов "
                 "оптимизатора. Одно и то же значение → тот же результат; на "
                 "состав и границы НЕ влияет.")
        if st.button("🏗 Построить проект", key="setup_build"):
            try:
                mix = _parse_names(mix_txt)
                proc = _parse_names(proc_txt)
                resp = _parse_names(resp_txt)
                # plo/phi уже собраны формой render_process_bounds выше (реальные
                # единицы, попарно L/U). Пустой список процесс-параметров → форма
                # вернула (None, None); build_setup_runner отвергнет пустой proc.
                if plo is None or phi is None:
                    raise ValueError("Добавьте хотя бы один процесс-параметр "
                                     "и задайте его границы (§17.4).")
                # mlo/mhi уже посчитаны формой render_composition_bounds выше
                # (доли; None — полный симплекс; в режиме phr-спеки —
                # fraction_bounds спеки).
                if phr_mode and phr_spec_live is None:
                    raise ValueError(
                        "Режим «phr-спека (JSON)»: спека не разобрана — "
                        "исправьте JSON (ошибка показана в форме выше).")
                # iter41.1: имена компонентов — из спеки (поле игнорируется),
                # иначе рассинхрон имён и молчаливый откат на бокс-сэмплер.
                mix = setup_mixture_names(mix, phr_spec_live)

                runner = build_setup_runner(
                    mixture_names=mix, process_names=proc,
                    process_lower=plo, process_upper=phi,
                    response_names=resp, mixture_lower=mlo, mixture_upper=mhi,
                    seed=int(seed_v))
                if phr_spec_live is not None:
                    # iter41.1: decode-слой активен — кандидаты пойдут
                    # phr-путём; группы iter31 в этом режиме не применяются.
                    runner.set_phr_spec(phr_spec_live)
                else:
                    # iter31: проектные функциональные группы (валидация —
                    # движком)
                    runner.set_mixture_sampling_groups(
                        parse_sampling_groups(groups_txt))
                # iter41.2: паспорт кампании — ДО первого замера (валидация
                # имён пар — штатным set_preflight_pairs, A0.6).
                if str(label_txt).strip():
                    runner.set_campaign_label(str(label_txt).strip())
                runner.set_preflight_pairs(parse_preflight_pairs(pairs_txt))
                # P2.3: паспорт — лоты/anchor'ы/весы. Синтаксис ловят чистые
                # парсеры (номер строки), имена валидируют ШТАТНЫЕ сеттеры.
                runner.set_material_lots(parse_material_lots(lots_txt))
                runner.set_anchor_recipes(parse_anchor_recipes(anchors_txt))
                runner.set_weighing_resolution(float(pass_step),
                                               float(pass_gpp))
                # iter52/P2.1-UI: дискретные уровни — ПОСЛЕ сборки схемы
                # (валидация имён/границ — штатным set_process_levels, A0.6).
                levels_now = parse_process_levels(levels_txt)
                runner.set_process_levels(levels_now)
                # P3.3: связки — ПОСЛЕ уровней (сеттеры проверяют конфликт
                # осей «уровни × связка» в обе стороны, A0.6).
                links_now = parse_process_links(links_txt)
                runner.set_process_links(links_now)
                # P3.1: ковариаты базы — валидация имён (дубли, коллизии с
                # откликами/осями) ШТАТНЫМ set_covariate_names (A0.6).
                runner.set_covariate_names(parse_covariate_names(cov_txt))
                st.session_state["campaign_ctrl"] = cv.CampaignController(runner)

                for k in ("setup_seed_X", "setup_seed_Y",
                          "setup_seed_df", "setup_seed_df_sig"):
                    st.session_state.pop(k, None)
                st.success(
                    f"Проект собран: смесь {mix} × процесс {proc}, отклики {resp}."
                    + (f" phr-спека активна (hash "
                       f"{phr_spec_live.spec_hash()[:12]}…)."
                       if phr_spec_live is not None else "")
                    + (f" {levels_caption(runner.process_levels)}"
                       if levels_now else "")
                    + (f" {links_caption(runner.process_links)}"
                       if links_now else "")
                    + " База пуста — предложите и измерьте стартовый дизайн "
                      "ниже.")

            except (ValueError, KeyError) as exc:
                st.error(str(exc))


def render_seed_entry(ctrl: "cv.CampaignController") -> None:
    """§17.4: ручной СТАРТОВЫЙ цикл «предложить seed → внести Y → зафиксировать».

    Пока стартовый дизайн не измерен (база пуста), это единственная активная
    секция вкладки: ``propose_seed`` (read-only) → таблица ввода Y (по всем P) →
    ``commit_seed`` (доливает в общую базу origin=seed, обучает суррогаты).
    «Заполнить тестовыми» берёт Y из демо-оракула (``_measure``) — ЯВНОЕ действие
    (A0.6). Составные координаты заблокированы; правятся только столбцы «(lab)».
    """
    runner = ctrl.runner
    props = list(runner.property_names)
    coord_names = setup_coord_names(runner)
    # iter65: пока база пуста, это ЕДИНСТВЕННАЯ активная секция — значит,
    # пользователь именно здесь, и док ассистента должен спрашивать про seed.
    publish_ui_focus("seed")
    st.markdown("### 🌱 Стартовый дизайн (seed) — ручной ввод откликов (§17.4)")
    st.caption(
        f"Отклики проекта: {', '.join(props)}. Предложите N точек по составной "
        "области, внесите измеренные Y по каждому свойству и зафиксируйте — точки "
        "лягут в ОБЩУЮ базу (origin=seed), суррогаты обучатся (И-1).")
    # Замечание 4: рекомендуемый N скрининга считаем от q компонентов и d
    # процесс-параметров — предлагаем сразу как значение по умолчанию.
    q = len(runner.current_schema.mixture_names)
    d = len(runner.current_schema.process_names)
    rec_n = recommended_seed_size(q, d)
    sc = st.columns([1, 1, 1, 1])
    seed_n = sc[0].number_input(
        "N seed-точек", min_value=2, max_value=200, value=int(rec_n), step=1,
        key="setup_seed_n",
        help=f"Рекомендация для скрининга: N = q·(1+d) + ⌈q·(1+d)/2⌉ = {rec_n} "
             f"(q={q} компонентов, d={d} процесс-параметров) — число членов "
             "кросс-модели «смесь-линейно × процесс-линейно» плюс ~50% запаса на "
             "остаточную дисперсию. Значение можно изменить вручную.")

    seed_design = sc[1].number_input(
        "зерно ГСЧ (воспроизводимость)", value=1, step=1,
        key="setup_seed_design",
        help="Зерно генератора случайных чисел для построения стартового "
             "дизайна. Одно и то же значение → тот же набор из N точек "
             "(воспроизводимо); другое значение → другой случайный вариант. "
             "К числу «seed-точек» отношения не имеет — это разные «seed».")

    # P0-багфикс «+/− срабатывает со второго клика»: НЕ передаём value= вместе
    # с key — меняющийся default конфликтует со state виджета, и Streamlit
    # проглатывает первое нажатие. Ключ инициализируется ОДИН раз из раннера
    # (до создания виджета), дальше состоянием владеет сам ключ.
    if "setup_seed_blocks" not in st.session_state:
        st.session_state["setup_seed_blocks"] = int(
            getattr(runner, "n_blocks_start", 1))
    nb_blocks = sc[2].number_input(
        "Партий (блоков)", min_value=1, max_value=20, step=1,
        key="setup_seed_blocks",
        help="Blocking стартового дизайна: если опыты нельзя поставить одной "
             "партией / за один день, план ОПТИМАЛЬНО разбивается на блоки "
             "(interchange по блочному D-критерию) — эффект партии ловится "
             "блочной моделью и не искажает оценки состава. 1 — без "
             "блокировки. Доборы веток автоматически получают НОВЫЙ блок "
             "(каждая партия добора — отдельный блок).")
    runner.n_blocks_start = max(1, int(nb_blocks))

    if sc[3].button("📐 Предложить seed-дизайн", key="setup_propose_seed"):
        X = np.asarray(ctrl.propose_seed(int(seed_n), seed=int(seed_design)), float)
        st.session_state["setup_seed_X"] = X
        st.session_state.pop("setup_seed_Y", None)
        # Новый дизайн — сброс состояния редактора Y (иначе data_editor наложит
        # СТАРЫЕ правки ячеек на новые строки); кнопка выше редактора, поэтому
        # виджет ещё не создан в этом прогоне и ключ можно чистить.
        st.session_state.pop("setup_seed_editor", None)
        st.session_state.pop("setup_seed_df", None)
        st.session_state.pop("setup_seed_df_sig", None)

    # Названия блоков (опц.): фактор блокировки + имя каждой партии — чтобы был
    # понятен ПРИНЦИП блокировки (оператор / партия сырья / рабочая смена).
    # Метаданные показа/Excel; на оптимальное разбиение не влияют; персистятся
    # в campaign.json (block_factor / block_names).
    nb_now = int(getattr(runner, "n_blocks_start", 1))
    if nb_now > 1:
        with st.expander("🧱 Названия партий (блоков) — что их различает (опц.)"):
            st.caption(
                "Блок — группа опытов, поставленных в одинаковых условиях "
                "(одна партия сырья / один оператор / одна смена). Назовите "
                "фактор и сами партии — имена попадут в таблицы и Excel "
                "(столбец «Партия»); на расчёт блокировки они не влияют.")
            bf = st.text_input(
                "Что различает партии (фактор блокировки)",
                value=str(getattr(runner, "block_factor", "") or ""),
                key="setup_block_factor",
                placeholder="например: оператор / партия сырья / рабочая смена")
            runner.block_factor = bf.strip()
            names_now = dict(getattr(runner, "block_names", {}) or {})
            ncols = st.columns(min(nb_now, 4))
            new_names: Dict[int, str] = {}
            for b in range(1, nb_now + 1):
                nm = ncols[(b - 1) % len(ncols)].text_input(
                    f"Имя блока {b}", value=str(names_now.get(b, "")),
                    key=f"setup_block_name_{b}",
                    placeholder=f"например: смена {b}")
                if nm.strip():
                    new_names[b] = nm.strip()
            runner.block_names = new_names

    Xs = st.session_state.get("setup_seed_X")
    if Xs is None:
        return
    Xs = np.atleast_2d(np.asarray(Xs, float))

    # iter42.2/42.4: слой НАВЕСКИ — только при активной phr-спеке, имена которой
    # совпадают с mixture-компонентами текущей фазы (иначе доли не той спеки).
    spec_w: Optional[PhrSpec] = getattr(runner, "phr_spec", None)
    if spec_w is not None and (list(spec_w.component_names)
                               != list(runner.current_schema.mixture_names)):
        spec_w = None
    delta_phr: Optional[float] = None
    gpp_val: Optional[float] = None
    if spec_w is not None:
        with st.expander("⚖️ Навеска (phr): разрешение весов, премикс, actual",
                         expanded=True):
            st.caption(
                "Слой навески (CAMPAIGN_SPEC_PVC §5): доли плана переводятся в "
                "phr по fixed-якорю спеки, снапятся к сетке весов и "
                "фиксируются как ACTUAL — модель должна видеть actual, а не "
                "nominal. Задайте параметры лаборатории:")
            wc = st.columns(2)
            # P2.3: дефолты слоя навески — из ПАСПОРТА кампании (если задан):
            # разрешение весов записывается при сетапе и переживает load, а
            # прежние 0.1/5.0 остаются фолбэком для кампаний без паспорта.
            step_g = wc[0].number_input(
                "Шаг весов, г", min_value=0.0,
                value=float(getattr(runner, "weighing_step_g", 0.0) or 0.1),
                step=0.01, format="%.4f", key="setup_weigh_step",
                help="Дискретность лабораторных весов. 0 — слой навески "
                     "выключен (план фиксируется как nominal). Дефолт — из "
                     "паспорта кампании (P2.3), если он задан.")
            gpp = wc[1].number_input(
                "г на 1 phr (загрузка)", min_value=0.0,
                value=float(getattr(runner, "grams_per_phr", 0.0) or 5.0),
                step=0.5, format="%.4f", key="setup_weigh_gpp",
                help="Сколько граммов приходится на 1 phr при этой загрузке "
                     "смесителя: δ_phr = шаг весов / (г на 1 phr). Дефолт — "
                     "из паспорта кампании (P2.3), если он задан.")
            if float(step_g) > 0 and float(gpp) > 0:
                try:
                    delta_phr = weighing_delta_phr(step_g, gpp)
                    gpp_val = float(gpp)
                except ValueError as exc:
                    st.error(str(exc))
                    delta_phr = None
            else:
                st.caption("Слой навески выключен (шаг весов или загрузка = 0): "
                           "план фиксируется как NOMINAL.")
            if delta_phr is not None:
                Xs_snap = snap_design_to_grid(spec_w, Xs, delta_phr)
                moved = float(np.abs(Xs_snap - Xs).max()) if len(Xs) else 0.0
                if moved > 0:
                    st.session_state["setup_seed_X"] = Xs_snap
                    Xs = Xs_snap
                st.caption(
                    weighing_caption(spec_w, delta_phr)
                    + (f" План СНАПНУТ к δ-сетке (макс. сдвиг доли "
                       f"{moved:.2e}) — фиксируется actual."
                       if moved > 0 else
                       " План уже стоит на δ-сетке — фиксируется actual."))
                nums_w = list(experiment_index(len(runner.points), len(Xs)))
                sel_w = st.selectbox("Карта навески для опыта №", nums_w,
                                     key="setup_weigh_row")
                i_sel = nums_w.index(sel_w)
                wdf_one = recipe_weighing_dataframe(
                    spec_w, Xs[i_sel, :spec_w.q], delta_phr,
                    grams_per_phr=gpp_val)
                st.dataframe(wdf_one, use_container_width=True,
                             hide_index=True)
                bad = [v for v in wdf_one["нарушение"] if str(v).strip()]
                if bad:
                    st.warning(
                        "Навеска этого опыта выходит за геометрию спеки "
                        f"({len(bad)} узл.) — см. колонку «нарушение». "
                        "Фиксация НЕ блокируется (A0.6): решение за вами "
                        "(премикс, другая загрузка, пересчёт плана).")

        # iter50/P1.3: «эффективные границы точки» — ОТДЕЛЬНЫЙ блок, доступный
        # и БЕЗ δ (навеска — про весы, границы — про геометрию: cap, окно
        # тотала, партнёры зависят от самой точки, §4 спеки).
        with st.expander("🔎 Эффективные границы точки (контракт ядра, "
                         "iter49/B7)"):
            st.caption(
                "«Почему план не даёт такую точку» — ответ ядра ПО ЭТОЙ точке: "
                "какие границы действуют на каждый узел и КАКОЕ ограничение их "
                "задало (заявленный интервал / потолок cap / окно тотала / "
                "техлимит phr / партнёры по группе). Read-only, ничего не "
                "блокирует (A0.6).")
            nums_b = list(experiment_index(len(runner.points), len(Xs)))
            sel_b = st.selectbox("Границы для опыта №", nums_b,
                                 key="setup_bounds_row")
            try:
                bdf_pt = point_bounds_dataframe(
                    spec_w, Xs[nums_b.index(sel_b), :spec_w.q],
                    delta_phr=delta_phr)
                st.dataframe(bdf_pt, use_container_width=True,
                             hide_index=True)
                st.caption(point_bounds_caption(bdf_pt))
            except ValueError as exc:
                st.error(f"Границы точки не рассчитаны: {exc}")

    if st.button("🧪 Заполнить тестовыми (демо-оракул)", key="setup_fill_demo"):

        st.session_state["setup_seed_Y"] = np.vstack(
            [runner._measure(np.asarray(x, float)) for x in Xs])
        st.session_state.pop("setup_seed_editor", None)
        st.session_state.pop("setup_seed_df", None)
        st.session_state.pop("setup_seed_df_sig", None)

    # Замечание 7: размер ПРОБЫ (партии) — добавляет столбцы расхода сырья
    # {компонент} (кг) = доля·batch, чтобы понимать, сколько взвесить на опыт.
    mix_names = list(runner.current_schema.mixture_names)
    batch = st.number_input(
        f"Размер пробы, {MASS_UNIT}/опыт (для расхода сырья и Excel)",
        min_value=0.0, value=0.0, step=0.1, key="setup_seed_batch",
        help="0 — только состав в долях; >0 — добавит столбцы расхода сырья "
             f"({MASS_UNIT}) = доля компонента × размер пробы (замечание 7). "
             "На сам дизайн (координаты) не влияет — только показ и выгрузка.")
    batch_kg = float(batch) if batch > 0 else None
    mass_cols = ([f"{c} ({MASS_UNIT})" for c in mix_names]
                 if batch_kg is not None else [])

    Ys = st.session_state.get("setup_seed_Y")
    lab_cols = [f"{p} (lab)" for p in props]
    # Единый источник таблицы (показ = редактор = Excel): чистый хелпер строит
    # «№ опыта» + заблокированные координаты (+ расход сырья) + столбцы «(lab)».
    #
    # БАГФИКС (пропадающие значения): вход data_editor обязан быть СТАБИЛЬНЫМ
    # между прогонами. Раньше df пересобирался КАЖДЫЙ rerun с уже внесённым
    # черновиком Y — для Streamlit изменившиеся данные = НОВЫЙ виджет, и свежая
    # (ещё не запечённая в df) правка сбрасывалась: первая введённая ячейка в
    # каждой следующей колонке «исчезала», повторный ввод принимался. Теперь df
    # кэшируется по сигнатуре (дизайн Xs + размер пробы) и пересобирается только
    # при её смене / явном заполнении (демо-кнопка, новый дизайн, загрузка
    # проекта) — черновик Y вливается в кэш в этот момент, правки не теряются.
    sig = (Xs.tobytes(), Xs.shape, batch_kg,
           int(getattr(runner, "n_blocks_start", 1)),
           tuple(sorted((getattr(runner, "block_names", {}) or {}).items())),
           # P3.1: смена объявленных ковариат меняет состав столбцов таблицы
           tuple(getattr(runner, "covariate_names", []) or []))
    if (st.session_state.get("setup_seed_df_sig") != sig
            or "setup_seed_df" not in st.session_state):
        st.session_state["setup_seed_df"] = seed_design_dataframe(
            runner, Xs, Ys, batch_kg=batch_kg)
        st.session_state["setup_seed_df_sig"] = sig
        # Пересборка входа → чистим состояние виджета (мы выше редактора по
        # коду, он ещё не создан в этом прогоне — ключ чистить безопасно).
        st.session_state.pop("setup_seed_editor", None)
    df = st.session_state["setup_seed_df"]
    cov_names_ui = list(getattr(runner, "covariate_names", []) or [])
    st.caption("Составные координаты заблокированы; заполняются только столбцы "
               "«свойство (lab)» (вручную или кнопкой «Заполнить тестовыми»)"
               + (" и «… (ковариата)» — телеметрия прогона, пустые ячейки "
                  "допустимы (P3.1)" if cov_names_ui else "")
               + ":")
    blk_cols = [c for c in ("Блок", "Партия") if c in df.columns]
    edited = st.data_editor(df, use_container_width=True, height=320,
                            hide_index=True,
                            disabled=["№ опыта", *blk_cols,
                                      *coord_names[:Xs.shape[1]],
                                      *mass_cols],
                            key="setup_seed_editor")
    # Iteration 28+: blocking должен быть ВИДЕН всегда — и когда включён
    # (размеры партий + цена блокировки), и когда выключен (как включить).
    st.caption(seed_blocking_caption(runner, Xs))
    # iter52/P2.1-UI: пометка дискретных осей у ПЛАНА — иначе колонка
    # «rotor_rpm» читается как непрерывная, а лаборатория ставит уровень.
    _lv_txt = seed_levels_caption(runner, Xs)
    if _lv_txt:
        st.caption(_lv_txt)
    # P3.3: реализуемость связок осей — та же логика, что у уровней.
    _lk_txt = seed_links_caption(runner, Xs)
    if _lk_txt:
        st.caption(_lk_txt)

    # iter32: preflight-диагностика предложенного плана (read-only, A0.6 —
    # НЕ блокирует commit). Кэш по сигнатуре дизайна: reference-пул и SVD
    # пересчитываются только при смене предложенных точек.
    pf_sig = (Xs.tobytes(), Xs.shape)
    if st.session_state.get("setup_seed_pf_sig") != pf_sig:
        try:
            st.session_state["setup_seed_pf"] = runner.preflight(Xs)
        except Exception as exc:  # noqa: BLE001 — диагностика не ломает seed-цикл
            st.session_state["setup_seed_pf"] = None
            st.session_state["setup_seed_pf_err"] = str(exc)
        st.session_state["setup_seed_pf_sig"] = pf_sig
    pf = st.session_state.get("setup_seed_pf")
    if pf is not None:
        st.caption(seed_preflight_caption(pf))
        with st.expander("🔎 Preflight: детали проверок плана"):
            st.caption(
                "Гейты ОТНОСИТЕЛЬНЫЕ: план сравнивается с равномерным "
                "reference-пулом ЭТОЙ ЖЕ области (классические абсолютные "
                "пороги cond<30 / VIF<5 в долях Шеффе неприменимы). Провал — "
                "сигнал пересмотреть план ДО измерений; фиксация не блокируется.")
            st.dataframe(preflight_details_dataframe(pf),
                         use_container_width=True, hide_index=True)
    elif st.session_state.get("setup_seed_pf_err"):
        st.caption("🔎 Preflight недоступен: "
                   + str(st.session_state["setup_seed_pf_err"]))
    # C2: черновик Y из редактора живёт в session_state (setup_seed_Y), чтобы
    # частично внесённые отклики можно было сохранить в проект ДО фиксации
    # (commit_seed) и восстановить при загрузке. NaN допустимы (пустые ячейки).
    try:
        st.session_state["setup_seed_Y"] = np.column_stack(
            [np.asarray(edited[c], float) for c in lab_cols])
    except (KeyError, TypeError, ValueError):
        pass
    # Для Excel ниже берём СВЕЖИЙ черновик (включая правки текущего прогона).
    Ys = st.session_state.get("setup_seed_Y", Ys)

    # C3: сохранить ПЛАН стартового эксперимента в Excel (ещё до фиксации) —
    # экспортируем внесённые в редакторе значения (пустые «(lab)» — под ручной
    # ввод в лаборатории), с расходом сырья, если задан размер пробы.
    st.download_button(
        "⬇️ Сохранить план в Excel (.xlsx)",
        data=seed_design_excel_bytes(runner, Xs, Ys, batch_kg=batch_kg,
                                     spec=spec_w, delta_phr=delta_phr,
                                     grams_per_phr=gpp_val),
        file_name="seed_design.xlsx", key="setup_seed_dl",
        mime="application/vnd.openxmlformats-officedocument."
             "spreadsheetml.sheet")

    if st.button("💾 Зафиксировать seed (commit_seed)", key="setup_commit_seed"):

        try:
            Y = np.column_stack([np.asarray(edited[c], float) for c in lab_cols])
            if np.isnan(Y).any():
                raise ValueError(
                    "Заполните измеренные отклики (столбцы «… (lab)») для ВСЕХ "
                    "точек — вручную в таблице или кнопкой «🧪 Заполнить "
                    "тестовыми». Пустые ячейки (None) фиксировать нельзя.")
            # P3.1: телеметрия прогона из столбцов «(ковариата)» — NaN
            # пропускаются (не снята); валидация имён/чисел — раннером.
            covs_seed = (covariate_rows_from_editor(edited, cov_names_ui)
                         if cov_names_ui else None)
            out = ctrl.commit_seed(Xs, Y, covariates=covs_seed)

            for k in ("setup_seed_X", "setup_seed_Y",
                      "setup_seed_df", "setup_seed_df_sig"):
                st.session_state.pop(k, None)
            # P0: уведомление через _flash — st.success перед st.rerun не
            # доживал до глаз пользователя (rerun стирает вывод прогона).
            _flash(
                f"Seed зафиксирован: +{out['added']} точек (origin=seed), общая "
                f"база = {out['n_base']}, суррогаты обучены. Дальше — создание "
                "веток (Ш4, §17.5).")
            # База стала непустой → сразу перерисовать вкладку, чтобы открылось
            # создание веток (§17.5) без второго клика (иначе ранний return в
            # render_campaign держит seed-секцию до следующего взаимодействия).
            st.rerun()
        except (ValueError, KeyError) as exc:
            st.error(str(exc))



# ----------------------------------------------------------------------
# Чистые хелперы черновика целей ветки (без Streamlit — тестируются напрямую)
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# iter43 (UI_REVISION_SPEC §43.2): виды целей в UI. Первые три — «сырые»
# DesirabilitySpec (min/max/target), последние два — ПОРОГИ на предсказанное
# СРЕДНЕЕ, собираемые через :func:`hard_threshold_spec` (ramp = ШУМ ИЗМЕРЕНИЯ
# отклика, iter39 замечание 1: узкий ramp = плоский нуль без градиента возврата).
# ----------------------------------------------------------------------
GOAL_KIND_GE = "порог ≥"
GOAL_KIND_LE = "порог ≤"
# P2.2 (UI_REVISION_SPEC): плато-таргет — d=1 на [target_low, target_high],
# рампы к 0 на low/high (двусторонний таргет-ДИАПАЗОН, а не пик в точке).
GOAL_KIND_RANGE = "target-диапазон"
GOAL_KINDS: List[str] = ["max", "min", "target", GOAL_KIND_RANGE,
                         GOAL_KIND_GE, GOAL_KIND_LE]

_THRESHOLD_KINDS = {GOAL_KIND_GE: "ge", GOAL_KIND_LE: "le"}

# Подсказка к «СКО шума измерения» (ширина ramp порога, iter39/§43.2).
_NOISE_SD_HELP = (
    "СКО шума измерения этого отклика (повторяемость метода). Порог реализуется "
    "не «обрывом», а НАКЛОНОМ шириной в один шум: d=1 в допустимой области, d=0 "
    "глубже порога на величину шума. Так у оптимизатора остаётся направление "
    "возврата в допустимую область (нулевая ширина даёт плоский нуль и «слепой» "
    "refine), а veto остаётся практически жёстким — различить «на пороге» и «чуть "
    "ниже» точнее шума измерения всё равно нельзя."
)

# Подсказка к плато-таргету (P2.2 UI_REVISION_SPEC).
_RANGE_HELP = (
    "Плато-таргет: внутри [плато от, плато до] желательность d=1 — все значения "
    "допуска РАВНОХОРОШИ (пример: желатинизация 60–70 %). Вне плато d линейно "
    "спадает к 0 на краях low/high. Точечный target здесь не годится: пик "
    "предпочёл бы середину допуска и тянул бы оптимум от его дешёвого края."
)

# Подсказка к вероятностному ограничению (chance-constraint, §43.2).
_CHANCE_HELP = (
    "Вероятностное ограничение: требуем, чтобы отклик попадал в допуск НЕ в "
    "среднем, а с заданной вероятностью — Pr(y в допуске) ≥ 1−α. Учитывает "
    "неопределённость модели (σ прогноза), поэтому «в среднем проходит, но "
    "разброс велик» уже не считается выполненным. Это НЕ цель: множитель к "
    "итоговому d_overall (роль отклика не меняется)."
)


def build_goal_spec(kind: str, *, low: Optional[float] = None,
                    high: Optional[float] = None,
                    target: Optional[float] = None,
                    weight: float = 1.0,
                    threshold: Optional[float] = None,
                    noise_sd: Optional[float] = None,
                    target_low: Optional[float] = None,
                    target_high: Optional[float] = None) -> DesirabilitySpec:
    """iter43 (§43.2) + P2.2: UI-вид цели → :class:`DesirabilitySpec` (чистый билдер).

    ``max``/``min``/``target`` — прямая сборка спеки по ``low``/``high``
    (+``target``). ``«порог ≥»``/``«порог ≤»`` — :func:`hard_threshold_spec`
    (порог на предсказанное СРЕДНЕЕ, ramp = ``noise_sd``): результат — обычный
    ``DesirabilitySpec``, поэтому он сериализуется и переживает save/load штатно,
    без нового вида в ядре. ``«target-диапазон»`` (P2.2) — плато-таргет
    ``kind='target_range'`` ядра: d=1 на ``[target_low, target_high]``, рампы
    к 0 на ``low``/``high``. Нехватка обязательных полей — явный ``ValueError``
    (A0.6: неполную цель молча не собираем).
    """
    if kind in _THRESHOLD_KINDS:
        if threshold is None:
            raise ValueError(f"Вид «{kind}» требует значение порога.")
        if noise_sd is None or float(noise_sd) <= 0.0:
            raise ValueError(
                f"Вид «{kind}» требует СКО шума измерения > 0 (ширина наклона): "
                f"нулевая ширина даёт плоский нуль без направления возврата.")
        return hard_threshold_spec(float(threshold), float(noise_sd),
                                   _THRESHOLD_KINDS[kind], weight=float(weight))
    if kind == GOAL_KIND_RANGE:
        if low is None or high is None:
            raise ValueError(f"Вид «{kind}» требует low и high (края рамп).")
        if target_low is None or target_high is None:
            raise ValueError(
                f"Вид «{kind}» требует границы плато «плато от»/«плато до» "
                f"(low < плато от < плато до < high).")
        return DesirabilitySpec("target_range", low=float(low),
                                high=float(high),
                                target_low=float(target_low),
                                target_high=float(target_high),
                                weight=float(weight))
    if kind not in ("max", "min", "target"):
        raise ValueError(f"Неизвестный вид цели '{kind}' (есть: {GOAL_KINDS}).")
    if low is None or high is None:
        raise ValueError(f"Вид «{kind}» требует low и high.")
    return DesirabilitySpec(kind, low=float(low), high=float(high),
                            target=(float(target) if kind == "target"
                                    and target is not None else None),
                            weight=float(weight))


def draft_add_goal(draft: Sequence[Dict[str, Any]], *, resp: str, kind: str,
                   low: Optional[float] = None, high: Optional[float] = None,
                   weight: float = 1.0,
                   target: Optional[float] = None,
                   threshold: Optional[float] = None,
                   noise_sd: Optional[float] = None,
                   target_low: Optional[float] = None,
                   target_high: Optional[float] = None) -> List[Dict[str, Any]]:
    """Добавить цель в черновик ветки (§17.5). Возвращает НОВЫЙ список.

    Цель по одному и тому же отклику НЕ дублируется: повторное добавление того же
    ``resp`` ЗАМЕНЯЕТ прежнюю запись (иначе при создании ветки дубли молча
    схлопнулись бы в ``goals[resp]`` — тихая потеря, A0.6). ``target`` хранится
    только для вида ``target``.

    iter43 (§43.2): для порогов (``«порог ≥»``/``«порог ≤»``) хранятся СЫРЫЕ
    входы ``threshold``/``noise_sd``, а не готовая спека — черновик остаётся
    редактируемым и объяснимым; спека собирается на фиксации
    (:func:`draft_goal_specs` → :func:`build_goal_spec`). Валидность вида
    проверяется сразу (сборкой), чтобы ошибка всплыла при добавлении, а не при
    создании ветки.
    """
    entry = {"resp": resp, "kind": kind,
             "low": (float(low) if low is not None else None),
             "high": (float(high) if high is not None else None),
             "weight": float(weight),
             "target": (float(target) if kind == "target" and target is not None
                        else None),
             "threshold": (float(threshold) if kind in _THRESHOLD_KINDS
                           and threshold is not None else None),
             "noise_sd": (float(noise_sd) if kind in _THRESHOLD_KINDS
                          and noise_sd is not None else None),
             "target_low": (float(target_low) if kind == GOAL_KIND_RANGE
                            and target_low is not None else None),
             "target_high": (float(target_high) if kind == GOAL_KIND_RANGE
                             and target_high is not None else None)}
    build_goal_spec(**{k: v for k, v in entry.items() if k != "resp"})  # валидация
    out = [dict(g) for g in draft]
    for i, g in enumerate(out):
        if g["resp"] == resp:
            out[i] = entry
            return out
    out.append(entry)
    return out


def draft_goal_specs(draft: Sequence[Dict[str, Any]]
                     ) -> Dict[str, DesirabilitySpec]:
    """iter43: черновик целей → ``{отклик: DesirabilitySpec}`` для создания ветки.

    Единственная точка сборки спек из черновика (UI больше не собирает их
    инлайном): порог превращается в ``hard_threshold_spec`` здесь.
    """
    out: Dict[str, DesirabilitySpec] = {}
    for g in draft:
        out[g["resp"]] = build_goal_spec(
            g["kind"], low=g.get("low"), high=g.get("high"),
            target=g.get("target"), weight=float(g.get("weight", 1.0)),
            threshold=g.get("threshold"), noise_sd=g.get("noise_sd"),
            target_low=g.get("target_low"), target_high=g.get("target_high"))
    return out


def draft_goal_text(entry: Mapping[str, Any]) -> str:
    """iter43: человекочитаемая строка одной цели черновика (для списка в UI)."""
    kind = entry["kind"]
    if kind in _THRESHOLD_KINDS:
        body = (f"{kind} {entry.get('threshold')} "
                f"(наклон = шум {entry.get('noise_sd')})")
    elif kind == GOAL_KIND_RANGE:
        body = (f"плато [{entry.get('target_low')}, {entry.get('target_high')}] "
                f"(рампы от {entry.get('low')} до {entry.get('high')})")
    elif kind == "target":
        body = (f"target [{entry.get('low')}, {entry.get('high')}], "
                f"пик {entry.get('target')}")
    else:
        body = f"{kind} [{entry.get('low')}, {entry.get('high')}]"
    return (f"**{entry['resp']}** — {body}, "
            f"значимость {entry.get('weight', 1.0)}")


def draft_add_chance(draft: Sequence[Dict[str, Any]], *, resp: str,
                     y_min: Optional[float] = None,
                     y_max: Optional[float] = None,
                     alpha: float = 0.05) -> List[Dict[str, Any]]:
    """iter43 (§43.2): добавить/заменить вероятностное ограничение в черновике.

    ``None`` у границы = «не ограничено» (``∓inf``): штатный односторонний случай
    ``Pr(y ≤ y_max) ≥ 1−α``. Валидность проверяется сразу конструктором
    :class:`ChanceConstraint` (α∈(0,1), y_min<y_max, хотя бы одна граница
    конечна) — ошибка всплывает при добавлении, а не при оптимизации.
    """
    entry = {"resp": resp,
             "y_min": (float(y_min) if y_min is not None else None),
             "y_max": (float(y_max) if y_max is not None else None),
             "alpha": float(alpha)}
    _chance_from_entry(entry)                       # валидация (A0.6)
    out = [dict(c) for c in draft]
    for i, c in enumerate(out):
        if c["resp"] == resp:
            out[i] = entry
            return out
    out.append(entry)
    return out


def draft_remove_chance(draft: Sequence[Dict[str, Any]],
                        index: int) -> List[Dict[str, Any]]:
    """iter43: убрать ограничение по индексу (идемпотентно, как у целей)."""
    out = [dict(c) for c in draft]
    if 0 <= index < len(out):
        del out[index]
    return out


def _chance_from_entry(entry: Mapping[str, Any]) -> ChanceConstraint:
    """Запись черновика → :class:`ChanceConstraint` (``None`` → ``∓inf``)."""
    return ChanceConstraint(
        y_min=(-np.inf if entry.get("y_min") is None else float(entry["y_min"])),
        y_max=(np.inf if entry.get("y_max") is None else float(entry["y_max"])),
        alpha=float(entry.get("alpha", 0.05)))


def draft_chance_constraints(draft: Sequence[Dict[str, Any]]
                             ) -> Dict[str, ChanceConstraint]:
    """iter43: черновик ограничений → ``{отклик: ChanceConstraint}`` для раннера."""
    return {c["resp"]: _chance_from_entry(c) for c in draft}


def chance_editor_dataframe(runner, branch_id: str) -> pd.DataFrame:
    """iter43 (§43.2): вероятностные ограничения ветки → таблица ОТДЕЛЬНЫМ блоком.

    Показывается НЕ вместе с целями: chance-ограничение — множитель к d_overall
    (Pr(y∈допуск) ≥ 1−α), а не нога качества; смешивать его с целями в одной
    таблице значит врать про роль отклика (§5). Пустой набор → пустая таблица.
    """
    rows = []
    for prop, con in (runner.branch_chance(branch_id) or {}).items():
        rows.append({
            "ограничение (отклик)": prop,
            "y_min": ("—" if not np.isfinite(con.y_min)
                      else round(float(con.y_min), 4)),
            "y_max": ("—" if not np.isfinite(con.y_max)
                      else round(float(con.y_max), 4)),
            "α": round(float(con.alpha), 4),
            "требование": f"Pr(y в допуске) ≥ {1.0 - float(con.alpha):.3f}",
        })
    return pd.DataFrame(rows)



def draft_remove_goal(draft: Sequence[Dict[str, Any]],
                      index: int) -> List[Dict[str, Any]]:
    """Убрать цель по индексу из черновика ветки. Возвращает НОВЫЙ список.

    Индекс вне диапазона — список возвращается без изменений (идемпотентно).
    """
    out = [dict(g) for g in draft]
    if 0 <= index < len(out):
        del out[index]
    return out


def render_branch_creation(ctrl: "cv.CampaignController") -> None:
    """§17.5 (Ш4): ВРУЧНУЮ создать ветку — мультицель + роли + ценовая нога.


    Замена авто-M7 (§17.0): пользователь объявляет намерение ветки сам. Цели
    набираются по одной в session_state (мультицель §16.3, каждая — вид/диапазон/
    вес); опционально включается ценовая нога (ρ-отклик + цены компонентов →
    ``make_linear_price_fn`` + desirability цены) и экономика (V/c_exp/H). Роли
    выводятся ядром из намерения (цель ⇒ OPTIMIZED, ρ без цели ⇒ PRICE_INPUT).
    Всё мутирующее — по кнопке (A0.6); логика/валидация — в
    :meth:`CampaignController.create_branch` (§17.3).
    """
    runner = ctrl.runner
    props = list(runner.property_names)
    mix_names = list(runner.current_schema.mixture_names)
    draft: List[Dict[str, Any]] = st.session_state.setdefault(
        "camp_new_goals", [])

    with st.expander("➕ Создать ветку вручную (§17.5 — цели/роли/ценовая нога)",
                     expanded=not list(runner.branches)):
        st.caption(
            "Ветка — контейнер НАМЕРЕНИЯ (канон §5/§12): несколько целей "
            "(мультицель §16.3) + опц. ценовая нога ρ. Модели у ветки нет — "
            "физика общая. Роли выводятся из намерения: цель ⇒ OPTIMIZED, ρ "
            "ценовой ноги без цели ⇒ PRICE_INPUT (И-5).")
        c = st.columns([2, 2, 1])
        name = c[0].text_input("Имя ветки",
                               value=f"branch{len(runner.branches) + 1}",
                               key="camp_nb_name")
        budget = c[1].number_input(
            "Бюджет ветки (число опытов)", min_value=1, max_value=200,
            value=20, step=1, key="camp_nb_budget",
            help="Максимум лабораторных опытов (точек добора), которые ветка "
                 "может потратить на достижение цели: 1 слот = 1 измеренная "
                 "точка. По исчерпании бюджета ветка останавливается.")
        satisfy = c[2].number_input(
            "Порог достаточности ceil (d_overall)", min_value=0.1,
            value=1.0, step=0.1, key="camp_nb_ceil",
            help="Уровень общей желательности d_overall (0…1+), по достижении "
                 "которого цель считается выполненной и раунды можно "
                 "останавливать (технический критерий §4).")


        # --- набор целей (мультицель) ---
        st.markdown("**🎯 Цели ветки (добавляйте по одной — мультицель §16.3)**")
        st.caption(
            "Ниже — конструктор ОДНОЙ цели. Настройте параметры и нажмите "
            "«➕ Добавить цель в ветку»: цель войдёт в ветку ТОЛЬКО после нажатия "
            "(до этого черновик пуст). Повторное добавление того же отклика "
            "ЗАМЕНЯЕТ прежнюю цель; удалить цель поштучно — кнопкой 🗑 ниже.")
        gc = st.columns([2, 2, 2, 2, 2])
        ng_resp = gc[0].selectbox("Цель (отклик)", props, key="camp_nb_resp")
        # iter43-хвост + P2.2: ПОЛНЫЙ набор видов и в форме создания ветки
        # (пороги/плато раньше задавались только в редакторе после создания).
        ng_kind = gc[1].selectbox("вид", GOAL_KINDS, key="camp_nb_kind")
        ng_lo = gc[2].number_input("low", value=0.0, step=0.5, key="camp_nb_lo")
        ng_hi = gc[3].number_input("high", value=10.0, step=0.5, key="camp_nb_hi")
        ng_w = gc[4].number_input("Значимость цели", min_value=0.01, value=1.0,
                                  step=0.5, key="camp_nb_w", help=_WEIGHT_HELP)

        ng_tgt = ng_tl = ng_th = ng_thr = ng_noise = None
        if ng_kind == "target":
            ng_tgt = st.number_input(
                "target (для вида target; low<target<high)",
                value=5.0, step=0.5, key="camp_nb_tgt")
        elif ng_kind == GOAL_KIND_RANGE:
            prc = st.columns(2)
            ng_tl = prc[0].number_input(
                "плато от (target_low)", value=60.0, step=0.5,
                key="camp_nb_tlo", help=_RANGE_HELP)
            ng_th = prc[1].number_input(
                "плато до (target_high)", value=70.0, step=0.5,
                key="camp_nb_thi", help=_RANGE_HELP)
        elif ng_kind in _THRESHOLD_KINDS:
            prc = st.columns(2)
            ng_thr = prc[0].number_input(
                "порог", value=10.0, step=0.5, key="camp_nb_thr")
            ng_noise = prc[1].number_input(
                "СКО шума измерения", min_value=1e-9, value=0.5, step=0.1,
                key="camp_nb_noise", help=_NOISE_SD_HELP)
        ac = st.columns([2, 2])
        if ac[0].button("➕ Добавить цель в ветку", key="camp_nb_add_goal"):
            try:
                st.session_state["camp_new_goals"] = draft_add_goal(
                    draft, resp=ng_resp, kind=ng_kind, low=float(ng_lo),
                    high=float(ng_hi), weight=float(ng_w),
                    target=(float(ng_tgt) if ng_tgt is not None else None),
                    target_low=(float(ng_tl) if ng_tl is not None else None),
                    target_high=(float(ng_th) if ng_th is not None else None),
                    threshold=(float(ng_thr) if ng_thr is not None else None),
                    noise_sd=(float(ng_noise) if ng_noise is not None
                              else None))
                draft = st.session_state["camp_new_goals"]
            except ValueError as exc:
                st.error(str(exc))
        if ac[1].button("🧹 Очистить цели", key="camp_nb_clear_goals"):
            st.session_state["camp_new_goals"] = []
            draft = []
        if draft:
            st.caption("Цели ветки (черновик) — 🗑 удаляет цель поштучно:")
            for i, g in enumerate(draft):
                rc = st.columns([9, 1])
                # P2.2: единый формат строки для всех видов (плато/пороги)
                rc[0].markdown(f"{i + 1}. {draft_goal_text(g)}")

                if rc[1].button("🗑", key=f"camp_nb_del_goal_{i}"):
                    st.session_state["camp_new_goals"] = draft_remove_goal(draft, i)
                    st.rerun()
        else:
            st.caption("Пока ни одной цели — ветке нужен минимум один объектив "
                       "(§17.3).")


        # --- модель себестоимости изделия (опц.) — замечания 6, 7 ---
        st.markdown(f"**💰 {COST_MODEL_LABEL} (опц., §3/§15.6)**")
        st.caption(
            f"Себестоимость изделия = цена состава ({CURRENCY_UNIT}/{MASS_UNIT}) × "
            "плотность ρ. Плотность ρ — отдельный отклик (роль «вход "
            "себестоимости»): чем точнее ρ, тем точнее себестоимость. Задайте цены "
            "компонентов и верхний приемлемый порог себестоимости.")
        use_price = st.checkbox(
            "Учитывать себестоимость изделия (ρ-отклик + цены компонентов)",
            key="camp_nb_use_price")
        rho_prop = None
        prices_txt = ""
        cost_hi = 300.0
        cur_unit = CURRENCY_UNIT
        if use_price:
            uc = st.columns([1, 2])
            cur_unit = uc[0].text_input(
                "Валюта", value=CURRENCY_UNIT, key="camp_nb_cur",
                help="Обозначение денежной единицы (₽, $, €…). Используется в "
                     "подписях цен, экономики и в Excel-выгрузке.")
            rho_prop = uc[1].selectbox(
                "Плотность ρ (отклик — вход себестоимости)", props,
                key="camp_nb_rho")
            pc = st.columns([3, 2])
            prices_txt = pc[0].text_input(
                f"Цены компонентов {mix_names}, {cur_unit}/{MASS_UNIT} "
                "(через запятую)",
                value=", ".join(["100"] * len(mix_names)), key="camp_nb_prices")
            cost_hi = pc[1].number_input(
                f"Верхний порог себестоимости, {cur_unit}/изд (выше → d=0)",
                min_value=0.0, value=300.0, step=10.0, key="camp_nb_cost_hi")
        st.markdown("**📈 Экономика ветки (опц., для двойного стопа §4/§6)**")
        st.caption(
            "Нужна, чтобы взвесить пользу уточнения против стоимости опытов "
            f"(денежный критерий остановки). Все суммы — в валюте «{cur_unit}».")
        ec = st.columns(3)
        vol = ec[0].number_input(
            "Объём выпуска V, изд/период", min_value=0.0, value=0.0,
            step=100.0, key="camp_nb_vol",
            help="Плановый выпуск изделий за один период. Масштабирует денежный "
                 "эффект снижения себестоимости (эффект = Δсебестоимости × V).")
        cexp = ec[1].number_input(
            f"Стоимость одного опыта, {cur_unit}/опыт", min_value=0.0, value=0.0,
            step=10.0, key="camp_nb_cexp",
            help="Во сколько обходится один лабораторный опыт (реагенты, время). "
                 "Сравнивается с денежной пользой уточнения: если опыт дороже "
                 "ожидаемой пользы — срабатывает экономический стоп.")
        hor = ec[2].number_input(
            "Горизонт планирования H, периодов", min_value=0.0, value=0.0,
            step=1.0, key="camp_nb_hor",
            help="Сколько периодов выпуска учитываем при оценке эффекта: денежная "
                 "польза накапливается за H периодов.")

        # iter31: функциональные группы ветки — ЭМПИРИЧЕСКОЕ знание из
        # скрининга («эти компоненты — одна ниша для цели ветки»). По умолчанию
        # наследуются проектные группы; override — намерение конкретной ветки.
        st.markdown("**🧩 Функциональные группы ветки (опц., iter31)**")
        proj_groups = getattr(runner, "sampling_groups", []) or []
        st.caption(
            "Пулы кандидатов ветки будут равномерно покрывать СУММАРНУЮ дозу "
            "каждой группы (стратификация суммы ниши). По умолчанию — "
            + ("наследуются проектные группы: "
               + " · ".join("{" + ", ".join(g) + "}" for g in proj_groups)
               if proj_groups else "проектных групп нет, стратификация выключена")
            + ". Задайте свои, если скрининг показал нишу для целей этой ветки.")
        use_groups = st.checkbox(
            "Задать группы для этой ветки (переопределить проектные)",
            key="camp_nb_use_groups")
        br_groups_txt = ""
        if use_groups:
            br_groups_txt = st.text_area(
                f"Группы (одна строка = одна группа; компоненты: {mix_names})",
                value="", key="camp_nb_groups",
                help="Имена компонентов через запятую, одна группа на строку. "
                     "Пусто при включённой галочке — явно БЕЗ стратификации "
                     "(отключает и проектные группы для этой ветки).")

        if st.button("🌿 Создать ветку", key="camp_nb_create"):
            try:
                if not draft:
                    raise ValueError("Добавьте хотя бы одну цель (§17.3).")
                # iter43/P2.2: ЕДИНАЯ точка сборки спек из черновика —
                # инлайн-сборка не знала порогов и плато (упала бы на них).
                goals = draft_goal_specs(draft)
                price_fn = cost_spec = None
                if use_price:
                    prices = _parse_floats(prices_txt)
                    if prices is None or len(prices) != len(mix_names):
                        raise ValueError(
                            f"Нужно {len(mix_names)} цен компонентов "
                            f"{mix_names} (через запятую).")
                    price_fn = make_linear_price_fn(prices)
                    cost_spec = DesirabilitySpec("min", low=0.0,
                                                 high=float(cost_hi), weight=0.5)
                out = ctrl.create_branch(
                    str(name), goals, budget=int(budget),
                    satisfy_at=float(satisfy), price_fn=price_fn,
                    cost_spec=cost_spec, rho_property=rho_prop,
                    volume=(float(vol) if vol > 0 else None),
                    cost_exp=(float(cexp) if cexp > 0 else None),
                    horizon=(float(hor) if hor > 0 else None),
                    sampling_groups=(parse_sampling_groups(br_groups_txt)
                                     if use_groups else None))
                st.session_state["camp_new_goals"] = []
                _invalidate_branch_caches()
                _flash(
                    f"Ветка «{out['branch_name']}» (`{out['branch_id']}`) создана: "
                    f"{out['n_goals']} цел., ценовая нога = {out['has_price_leg']}"
                    + (f" (ρ={out['rho_property']}, канал занулён="
                       f"{out['price_channel_suppressed']})"
                       if out['has_price_leg'] else "")
                    + f"; d_best={out['d_best']:.3f}.")
                st.rerun()
            except (ValueError, KeyError) as exc:
                st.error(str(exc))


def render_workbench(ctrl: "cv.CampaignController", bsel: str) -> None:
    """§17.5 (Ш5): рабочий стол ветки на РУЧНОМ цикле §17.2 (предложить → Y → долить).

    Замена авто-оракула (`run_branch_round`) реальным лабораторным циклом:
    ``validate_ready`` (§17.3 гейт) → ``propose_points`` (read-only) → таблица
    ввода измеренных Y по всем P → ``commit_measured`` (долив в общую базу
    origin=branch:{id}, переобучение суррогатов, §4-стоп). «Заполнить тестовыми»
    (демо-оракул ``_measure``) остаётся ЯВНОЙ кнопкой (A0.6). Составные координаты
    заблокированы; правятся только столбцы «свойство (lab)». Ключи session_state
    привязаны к ветке — смена ветки не тащит чужих кандидатов.
    """
    runner = ctrl.runner
    props = list(runner.property_names)
    coord_names = setup_coord_names(runner)
    kx, ky = f"camp_wb_X_{bsel}", f"camp_wb_Y_{bsel}"
    with st.expander("🛠 Рабочий стол ветки (§17.2/§16.4 — ручной добор)",
                     expanded=True):
        st.caption(
            "Реальный лабораторный цикл (§17.2): предложить N точек (read-only) → "
            "внести измеренные Y по всем свойствам → долить в ОБЩУЮ базу "
            "(origin=branch:{id}, И-1) → переобучить суррогаты → x*/d_best → "
            "§4-стоп. Мерим только по кнопке (A0.6); долив запечатывает undo.")
        br_now = runner.branches[bsel]
        st.caption(f"Ветка «{br_now.name}»: бюджет {br_now.budget}, потрачено "
                   f"{br_now.spent}, осталось {br_now.remaining()}, "
                   f"d_best={br_now.d_best:.3f}, статус {br_now.status}.")

        # §17.6.1 (C3): рекомендованный РЕЦЕПТ ветки x* + скачивание в Excel.
        # M8-argmax по ОБЩИМ GP-суррогатам дорогой → считаем ТОЛЬКО по
        # кнопке (A0.6), результат кешируем в session_state (df + готовые
        # xlsx-байты), чтобы rerun не пересчитывал оптимизацию.
        st.markdown("**📋 Рекомендованный рецепт ветки (x*) — по общим "
                    "GP-суррогатам**")
        st.caption(
            "Итог ветки одной строкой: рецепт-состав + процесс в РЕАЛЬНЫХ единицах, "
            "предсказанные свойства целей, общий d_overall и по-целевые d[·]. "
            "Считается по кнопке (M8-argmax дорогой); при размере пробы > 0 "
            "добавляется расход сырья на пробу.")
        rec_batch = st.number_input(
            f"Размер пробы, {MASS_UNIT}/опыт (для расхода сырья и Excel)",
            min_value=0.0, value=0.0, step=0.1, key=f"camp_wb_recipe_batch_{bsel}")
        rkey = f"cache_recipe_{bsel}"
        if st.button("📋 Рассчитать рецепт ветки (x*)",
                     key=f"camp_wb_recipe_btn_{bsel}"):
            import io
            try:
                bk = float(rec_batch) if float(rec_batch) > 0 else None
                # iter43.3: рецепт + binding_report ОДНИМ прогоном argmax
                df_rec, brep = branch_recipe_with_binding(
                    runner, bsel, batch_kg=bk)
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as xw:
                    df_rec.to_excel(xw, sheet_name="Рецепт", index=False)
                st.session_state[rkey] = (df_rec, buf.getvalue(), brep)
            except (ValueError, KeyError, RuntimeError) as exc:
                st.session_state.pop(rkey, None)
                st.error(f"Не удалось рассчитать рецепт ветки: {exc}")
        cached_rec = st.session_state.get(rkey)
        if cached_rec is not None:
            df_rec, xls_bytes, brep = cached_rec
            st.dataframe(df_rec, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Скачать рецепт ветки в Excel (.xlsx)", data=xls_bytes,
                file_name=f"branch_{bsel}_recipe.xlsx",
                key=f"camp_wb_recipe_dl_{bsel}",
                mime="application/vnd.openxmlformats-officedocument."
                     "spreadsheetml.sheet")
            # iter43.3 (§43.3): binding_report ОБЯЗАТЕЛЕН к просмотру — без
            # него «оптимум не найден» неотличим от «оптимум запрещён».
            st.caption(binding_report_caption(brep))
            bdf = binding_report_dataframe(brep)
            if not bdf.empty:
                st.caption("Что связывает оптимум (veto целей + вероятностные "
                           "ограничения; % — доля точек пула под биндингом):")
                st.dataframe(bdf, use_container_width=True, hide_index=True)


        # §17.3 (Ш2) гейт: перед предложением/пересчётом — проверка полноты данных
        ready = ctrl.validate_ready(bsel)

        if not ready["ok"]:
            st.error("Не хватает данных для добора (§17.3):\n" + ready["text"])

        wc = st.columns([1, 1, 1])
        wb_n = wc[0].number_input("N точек", min_value=1, max_value=20, value=3,
                                  step=1, key=f"camp_wb_n_{bsel}")
        wb_expl = wc[1].slider("explore", 0.0, 1.0, 0.3, 0.05,
                               key=f"camp_wb_expl_{bsel}")
        if wc[2].button("📐 Предложить точки (read-only)",
                        key=f"camp_wb_propose_{bsel}", disabled=not ready["ok"]):
            X = np.asarray(ctrl.propose_points(bsel, n_points=int(wb_n),
                                               explore_frac=float(wb_expl),
                                               n_candidates=200), float)
            st.session_state[kx] = X
            st.session_state.pop(ky, None)

        # P0: сводка ПОСЛЕДНЕГО долива живёт в session_state — переживает
        # st.rerun и остаётся видимой до следующего раунда этой ветки.
        last = st.session_state.get(f"camp_wb_last_{bsel}")
        if last is not None:
            # NB: без вложенного st.expander — рабочий стол сам живёт в
            # экспандере, а Streamlit запрещает вложенные экспандеры.
            st.success(last["msg"])
            st.caption("Измеренные отклики долитых точек (по всем P):")
            st.dataframe(last["points"], use_container_width=True,
                         hide_index=True)
            st.caption("Сводка источников общей базы:")
            st.dataframe(last["origins"], use_container_width=True,
                         hide_index=True)
            st.caption(last["stop"])

        Xs = st.session_state.get(kx)
        if Xs is None:
            return
        Xs = np.atleast_2d(np.asarray(Xs, float))
        if Xs.shape[0] == 0:
            st.info("Бюджет ветки исчерпан — предложить нечего.")
            return

        if st.button("🧪 Заполнить тестовыми (демо-оракул)",
                     key=f"camp_wb_fill_{bsel}"):
            st.session_state[ky] = np.vstack(
                [runner._measure(np.asarray(x, float)) for x in Xs])

        Ys = st.session_state.get(ky)
        df = pd.DataFrame(np.round(Xs, 4), columns=coord_names[:Xs.shape[1]])
        lab_cols = [f"{p} (lab)" for p in props]
        for j, col in enumerate(lab_cols):
            df[col] = (np.round(np.asarray(Ys, float)[:, j], 4)
                       if Ys is not None else np.nan)
        # P3.1: столбцы под телеметрию прогона (объявленные ковариаты) —
        # заполняются при измерении; пустые ячейки допустимы (не отклик).
        wb_cov_names = list(getattr(runner, "covariate_names", []) or [])
        for cn in wb_cov_names:
            df[f"{cn} (ковариата)"] = np.nan
        # Сквозная нумерация: предложенные точки ещё не залиты — показываем их
        # будущие номера в общей базе (len(points)+1 … +N). Явный read-only
        # столбец (st.data_editor игнорирует кастомный индекс — см. seed выше).
        df.insert(0, "№ опыта",
                  list(experiment_index(len(runner.points), len(df))))
        st.caption("Предложенные точки: координаты заблокированы, заполняются "
                   "только столбцы «свойство (lab)» (вручную или демо-кнопкой)"
                   + (" и «… (ковариата)» — телеметрия прогона (P3.1)"
                      if wb_cov_names else "") + ":")
        edited = st.data_editor(df, use_container_width=True, height=280,
                                hide_index=True,
                                disabled=["№ опыта", *coord_names[:Xs.shape[1]]],
                                key=f"camp_wb_editor_{bsel}")

        if st.button("💾 Долить измеренные (commit_measured)",
                     key=f"camp_wb_commit_{bsel}"):
            try:
                d_before = float(br_now.d_best)
                Y = np.column_stack([np.asarray(edited[c], float)
                                     for c in lab_cols])
                if np.isnan(Y).any():
                    raise ValueError(
                        "Заполните измеренные отклики (столбцы «… (lab)») для "
                        "ВСЕХ предложенных точек — вручную или кнопкой "
                        "«🧪 Заполнить тестовыми». Пустые ячейки (None) "
                        "доливать нельзя.")
                # P3.1: телеметрия прогона из столбцов «(ковариата)»
                covs_wb = (covariate_rows_from_editor(edited, wb_cov_names)
                           if wb_cov_names else None)
                res = ctrl.commit_measured(bsel, Xs, Y, covariates=covs_wb)

                st.session_state.pop(kx, None)
                st.session_state.pop(ky, None)
                wdf = workbench_points_dataframe(runner, res)
                if not wdf.empty:
                    # Те же сквозные номера, что точки получили в общей базе:
                    # последние len(wdf) опытов проекта — явным столбцом (единый
                    # вид с редакторами выше, без служебного индекса).
                    wdf.insert(0, "№ опыта", list(experiment_index(
                        len(runner.points) - len(wdf), len(wdf))))
                    # origin-тег канонично хранится как branch:{id}; для показа —
                    # человекочитаемое «Имя (id)» (см. origin_label).
                    wdf["origin"] = [origin_label(runner, o) for o in wdf["origin"]]
                    wdf = wdf.rename(columns={"origin": "ветка"})
                oc = pd.DataFrame(
                    [{"источник": origin_label(runner, k), "точек": v}
                     for k, v in runner.origin_counts().items()])
                # §4-стоп (двойной): технический И экономический, читает роль ρ
                delta_d = float(res["d_best"]) - d_before
                dec = runner.branch_stop_decision(
                    bsel, delta_d=delta_d, ceil=br_now.satisfy_at,
                    n_round=int(res["added"]) or 1, n_candidates=200,
                    n_mc=128, seed=0)
                # P0: сводка раунда — в session_state (переживает st.rerun и
                # остаётся видимой), а rerun сразу обновляет капшены/таблицы
                # ВЫШЕ по странице (бюджет, d_best, роли, база).
                st.session_state[f"camp_wb_last_{bsel}"] = {
                    "msg": (f"Долито {res['added']} точек "
                            f"(origin=branch:{bsel}); d_best {d_before:.3f} → "
                            f"{res['d_best']:.3f} (монотонно не убывает); "
                            f"общая база = {res['n_base']} точек."),
                    "points": wdf,
                    "origins": oc,
                    "stop": (
                        f"§4-стоп: **{_STOP_RU.get(dec.reason, dec.reason)}** "
                        f"(Δd={delta_d:+.4f}, d_best={res['d_best']:.3f}, "
                        f"ceil={br_now.satisfy_at:.3f}, "
                        f"econ_red_flag={dec.econ_red_flag})."),
                }
                _invalidate_branch_caches()
                st.rerun()
            except (ValueError, KeyError, RuntimeError) as exc:
                st.error(str(exc))


def render_schema_evolution(ctrl: "cv.CampaignController") -> None:
    """§17.6 / §16.2 (Ш6): эволюция схемы В ЛЮБОЙ МОМЕНТ + пересчёт (UI).

    Штатная операция живого проекта: добавить процесс-переменную / компонент
    смеси / отклик, подвинуть границы области. Миграция старых точек — по ЯВНОЙ
    политике (A0.6: молчаливой миграции нет). Всё делегируется фасаду
    :class:`CampaignController` (логика/валидация там); общая база не урезается
    (И-1), версия схемы растёт (кроме region-move). Read/rare-write — по кнопке.
    """
    from ..core.schema_evolution import known_constant

    runner = ctrl.runner
    full_proc = (list(runner._full_proc.names)
                 if getattr(runner, "_full_proc", None) else [])
    full_mix = (list(runner._full_mix.names)
                if getattr(runner, "_full_mix", None) else [])
    cur_proc = list(runner.current_schema.process_names)
    cur_mix = list(runner.current_schema.mixture_names)
    hidden_proc = [p for p in full_proc if p not in cur_proc]
    hidden_mix = [m for m in full_mix if m not in cur_mix]

    with st.expander("🧬 Эволюция схемы (§16.2 — раскрыть ось/отклик, подвинуть "
                     "границы)"):
        st.caption(
            "Живой проект меняется: раскрыть объявленную процесс-ось/компонент, "
            "ввести новый отклик, подвинуть границы области. Миграция старых точек "
            "— по ЯВНОЙ политике (A0.6). База не урезается (И-1); версия растёт.")

        # --- раскрыть процесс-переменную (known_constant baseline) ---
        st.markdown("**➕ Раскрыть процесс-переменную** (из полной схемы)")
        if hidden_proc:
            pc = st.columns([2, 2, 2])
            pv = pc[0].selectbox("переменная", hidden_proc, key="camp_ev_proc")
            pconst = pc[1].number_input("константа у старых точек (миграция)",
                                        value=0.0, step=0.1, key="camp_ev_proc_c")
            if pc[2].button("➕ Добавить процесс-ось", key="camp_ev_proc_btn"):
                try:
                    ctrl.add_process_var(pv, known_constant(float(pconst)))
                    st.success(f"Процесс-ось «{pv}» раскрыта (миграция "
                               f"known_constant={pconst}); версия схемы поднята.")
                except (ValueError, KeyError, RuntimeError) as exc:
                    st.error(str(exc))
        else:
            st.caption("Все объявленные процесс-оси уже в схеме.")

        # --- раскрыть компонент смеси (Σ-совместимо: known_constant(0)) ---
        st.markdown("**➕ Раскрыть компонент смеси** (грань симплекса C=0)")
        if hidden_mix:
            mc = st.columns([3, 2])
            mv = mc[0].selectbox("компонент", hidden_mix, key="camp_ev_mix")
            if mc[1].button("➕ Добавить компонент", key="camp_ev_mix_btn"):
                try:
                    ctrl.add_mixture_component(mv)
                    st.success(f"Компонент «{mv}» раскрыт (миграция "
                               "known_constant(0.0), грань симплекса); версия "
                               "поднята.")
                except (ValueError, KeyError, RuntimeError) as exc:
                    st.error(str(exc))
        else:
            st.caption("Все объявленные компоненты смеси уже в схеме.")

        # --- ввести новый отклик ---
        st.markdown("**➕ Ввести новый отклик** (у старых точек Y=MISSING)")
        rc = st.columns([3, 2])
        new_resp = rc[0].text_input("имя отклика", value="", key="camp_ev_resp")
        if rc[1].button("➕ Добавить отклик", key="camp_ev_resp_btn"):
            try:
                from ..core.schema import ResponseSpec
                if not new_resp.strip():
                    raise ValueError("Задайте имя отклика.")
                ctrl.add_response(ResponseSpec(name=new_resp.strip()))
                st.success(f"Отклик «{new_resp.strip()}» введён в схему (версия "
                           "поднята; у старых точек значение MISSING, §13.7).")
            except (ValueError, KeyError, TypeError) as exc:
                st.error(str(exc))

        # --- подвинуть границы (region-move, без bump) ---
        st.markdown("**↔ Подвинуть границы области** (relax/restrict, без bump)")
        bc = st.columns([2, 2, 2, 2])
        axes = cur_mix + cur_proc
        mv_ax = bc[0].selectbox("ось", axes, key="camp_ev_bound_ax")
        mv_lo = bc[1].number_input("нижняя", value=0.0, step=0.05,
                                   key="camp_ev_bound_lo")
        mv_hi = bc[2].number_input("верхняя", value=1.0, step=0.05,
                                   key="camp_ev_bound_hi")
        mv_intent = bc[3].selectbox("намерение", ["relax", "restrict"],
                                    key="camp_ev_bound_intent")
        if st.button("↔ Применить движение границ", key="camp_ev_bound_btn"):
            try:
                if mv_intent == "relax":
                    ctrl.relax_bounds(mv_ax, float(mv_lo), float(mv_hi))
                else:
                    ctrl.restrict_bounds(mv_ax, float(mv_lo), float(mv_hi))
                st.success(f"Границы «{mv_ax}» → [{mv_lo}, {mv_hi}] ({mv_intent}); "
                           "область обновлена (история цела, И-1).")
            except (ValueError, KeyError, RuntimeError) as exc:
                st.error(str(exc))


def render_screening_analysis(ctrl: "cv.CampaignController") -> None:
    """M3-минималка (UI): интерпретируемый анализ скрининга после измеренного seed.

    Показывает «что дали опыты» ДО построения веток: (1) сводную матрицу влияний
    «компонент × свойство» (ARD-важность 0…1) и (2) детальный разбор выбранного
    свойства — Scheffé-quadratic по составу: R²/adj-R²/RMSE/q_eff, коэффициенты,
    ANOVA, значимые термы и bar-chart важности компонентов. Анализ — ТОЛЬКО по
    составу (mixture-only, объём минималки); физика в целом остаётся за общими
    GP-суррогатами (канон §5/§12). Дорогие ARD-фиты считаются ТОЛЬКО по кнопке
    (A0.6), результат кешируется в session_state; вся математика — в чистом слое
    :mod:`campaign_screening` (тестируется без Streamlit).
    """
    runner = ctrl.runner
    props = list(runner.property_names)
    with st.expander("📊 Анализ скрининга (M3 — Scheffé + ARD по составу)"):
        st.caption(
            "Интерпретируемая сводка сразу после измеренного стартового дизайна: "
            "какой компонент на какое свойство влияет (ARD-важность) и насколько "
            "квадратичная модель Шеффе по составу объясняет данные (R²). Процесс "
            "здесь не участвует — его роль за общими GP-суррогатами. Считается "
            "по кнопке (ARD-фиты дорогие).")

        # --- сводная матрица влияний (главный итог скрининга) ---
        if st.button("📊 Рассчитать матрицу влияний (компонент × свойство)",
                     key="camp_m3_overview_btn"):
            with st.spinner("ARD-скрининг по каждому свойству…"):
                try:
                    st.session_state["camp_m3_matrix"] = csx.influence_matrix(
                        runner, n_restarts=4, seed=0)
                except (ValueError, KeyError, RuntimeError) as exc:
                    st.session_state.pop("camp_m3_matrix", None)
                    st.error(f"Не удалось построить матрицу влияний: {exc}")
        mat = st.session_state.get("camp_m3_matrix")
        if mat is not None:
            st.caption("ARD-важность компонентов (0…1; максимум по свойству = 1) "
                       "— чем ближе к 1, тем сильнее компонент влияет на свойство:")
            st.dataframe(mat, use_container_width=True)

        # --- детальный разбор одного свойства ---
        st.markdown("**🔬 Детальный разбор свойства (Scheffé-quadratic по составу)**")
        prop = st.selectbox("Свойство", props, key="camp_m3_prop")
        if st.button("🔬 Разобрать свойство (Scheffé-fit + ANOVA + ARD)",
                     key="camp_m3_fit_btn"):
            with st.spinner(f"Фит Шеффе + ARD для «{prop}»…"):
                try:
                    st.session_state["camp_m3_report"] = csx.screening_report(
                        runner, prop, n_restarts=4, seed=0)
                except (ValueError, KeyError, RuntimeError) as exc:
                    st.session_state.pop("camp_m3_report", None)
                    st.error(f"Не удалось разобрать свойство: {exc}")
        rep = st.session_state.get("camp_m3_report")
        if rep is not None and rep.get("property") in props:
            s = rep["summary"]
            mc = st.columns(4)
            mc[0].metric("R²", f"{s['r2']:.3f}")
            adj = s.get("adj_r2")
            mc[1].metric("adj-R²",
                         f"{adj:.3f}" if isinstance(adj, (int, float)) else "—")
            mc[2].metric("RMSE", f"{s['rmse']:.3g}")
            mc[3].metric("q_eff", rep["q_eff"])
            if s.get("underdetermined"):
                st.warning("n < p: модель недоопределена — R² вводит в "
                           "заблуждение, добавьте опыты (§17.4).")
            st.caption(f"Свойство «{rep['property']}» — коэффициенты Шеффе "
                       f"(mixture-only, {rep['model']}):")
            st.dataframe(pd.DataFrame(rep["coefficients"]),
                         use_container_width=True, hide_index=True)
            st.caption("ANOVA (значимость регрессии в целом, F-тест):")
            st.dataframe(pd.DataFrame(rep["anova"]),
                         use_container_width=True, hide_index=True)
            rank = pd.DataFrame(rep["component_ranking"])
            if not rank.empty and "importance" in rank.columns:
                st.caption("Важность компонентов (ARD, 0…1):")
                st.bar_chart(rank.set_index("component")["importance"])
            st.caption(f"Значимых термов (p < {rep['alpha']}): "
                       f"{rep['n_significant']}.")
            if rep["n_significant"]:
                st.dataframe(pd.DataFrame(rep["significant_terms"]),
                             use_container_width=True, hide_index=True)


def render_campaign() -> None:
    """Вкладка «🧬 Кампания»: реальный сетап §17.4 + роли + мультицель §16.3 +
    рабочий стол §16.4 + смена роли §5 + spawn §8 + undo §7 (мутации — по кнопке)."""


    # UX: демо-кнопки — компактно в ЗАГОЛОВКЕ (popover справа от подзаголовка;
    # для старых Streamlit без st.popover — фолбэк-экспандер), а не отдельным
    # широким блоком посреди страницы.
    hdr = st.columns([5, 1])
    hdr[0].subheader("🧬 Проект: per-branch роли откликов и эволюция (ТЗ v1.1)")
    _ctrl_now = get_campaign_controller()
    with hdr[1]:
        if hasattr(st, "popover"):
            _demo_box = st.popover("🧪 Демо")
        else:  # pragma: no cover — старые Streamlit без st.popover
            _demo_box = st.expander("🧪 Демо", expanded=_ctrl_now is None)
    st.caption(
        "Роль отклика — атрибут пары (ветка × отклик): один и тот же ρ может быть "
        "ЦЕЛЬЮ в одной ветке и ЦЕНОЙ-ВХОДОМ в другой. Денежный канал ρ читается из "
        "РЕАЛЬНОЙ атрибуции ядра (И-5/Гр-1): OPTIMIZED ⇒ занулён, PRICE_INPUT ⇒ "
        "живой. Всё, что меняет состояние, делает только ваша кнопка (A0.6).")

    # P0: уведомления мутаций (переживают st.rerun) — показ в начале прогона.
    _show_flashes()

    # §17.4 (Ш3b): форма реального сетапа проекта — в ЛЕВОЙ панели (сайдбар).
    render_setup_form()

    # P0: демо-проект — с явным подтверждением, если кампания уже есть
    # (раньше кнопка молча ЗАТИРАЛА текущую кампанию одним кликом).
    with _demo_box:
        st.caption("Демо-проект (синтетический оракул {A,B,C}×{T,P}) — готовый "
                   "проект для знакомства с интерфейсом: общий пул + "
                   "две контрастные ветки (premium: ρ=PRICE_INPUT, канал живой; "
                   "rho_focus: ρ=OPTIMIZED, канал занулён).")
        if _ctrl_now is not None:
            st.warning("Проект уже есть в сессии: создание демо ЗАМЕНИТ его. "
                       "Несохранённые изменения пропадут — сначала сохраните "
                       "проект в сайдбаре («📁 Проект»).")
            demo_ok = st.checkbox(
                "Понимаю: заменить текущий проект демо-проектом",
                key="camp_create_confirm")
        else:
            demo_ok = True
        if st.button("🧬 Создать / сбросить демо-проект", key="camp_create",
                     disabled=not demo_ok):
            with st.spinner("Сборка демо-проекта (общий пул + 2 ветки)…"):
                runner = build_demo_campaign_runner()
                st.session_state["campaign_ctrl"] = cv.CampaignController(runner)
            _flash("Демо-проект создан: ветки **premium** (ρ=PRICE_INPUT, "
                   "канал живой) и **rho_focus** (ρ=OPTIMIZED, канал занулён).")
            st.rerun()

    ctrl = get_campaign_controller()
    if ctrl is None:
        st.info("Соберите проект в форме «🆕 Новый проект» (левая панель) или "
                "создайте демо-проект — кнопка «🧪 Демо» в заголовке "
                "(синтетический оракул {A,B,C}×{T,P}).")
        return
    runner = ctrl.runner

    # C2: read-only сводка настроек ДЕЙСТВУЮЩЕГО проекта (из движка) — видно,
    # что доли компонентов/границы процесса/отклики подтянулись после загрузки.
    render_project_settings(runner)

    # §17.4 (Ш3b): пока стартовый дизайн НЕ измерен (база пуста) — единственная
    # активная секция это ручной seed-цикл; ветко-UI ниже требует измеренных данных.
    if len(runner.points) == 0:
        render_seed_entry(ctrl)
        return

    # §17.5 (Ш4): ручное создание веток — доступно после измеренного seed.
    render_branch_creation(ctrl)
    # §17.6/§16.2 (Ш6): эволюция схемы в любой момент (штатная операция проекта).
    render_schema_evolution(ctrl)

    # M3-минималка: интерпретируемый анализ скрининга сразу после измеренного seed
    # (Scheffé + ARD по составу) — «что дали опыты» до построения веток.
    render_screening_analysis(ctrl)

    # C3 (§17.6.1): выгрузка ОБЩЕЙ базы опытов кампании в Excel (+ расход сырья).

    with st.expander("⬇️ Выгрузить общую базу опытов в Excel (C3)"):
        st.caption(
            "Общая база всех измеренных опытов (И-1): № опыта, источник, состав "
            "(доли) + процесс (реальные единицы) и измеренные отклики. Укажите "
            f"размер партии ({MASS_UNIT}) — добавится расход каждого компонента на "
            "опыт, чтобы понимать, сколько сырья взвесить (замечание 7).")
        bc = st.columns([1, 2])
        batch = bc[0].number_input(
            f"Размер партии, {MASS_UNIT}/опыт", min_value=0.0, value=0.0,
            step=0.1, key="camp_base_batch",
            help="0 — только состав в долях; >0 — добавит столбцы расхода сырья "
                 f"({MASS_UNIT}) = доля компонента × размер партии.")
        batch_kg = float(batch) if batch > 0 else None
        base_df = campaign_base_dataframe(runner, batch_kg=batch_kg)
        st.dataframe(base_df, use_container_width=True, hide_index=True)
        _blk_txt = base_blocking_caption(runner)
        if _blk_txt:
            st.caption(_blk_txt)
        st.download_button(
            "⬇️ Скачать .xlsx",
            data=campaign_base_excel_bytes(runner, batch_kg=batch_kg),
            file_name="campaign_base.xlsx", key="camp_base_dl",
            mime="application/vnd.openxmlformats-officedocument."
                 "spreadsheetml.sheet")

    # §17.2.1: КОРРЕКЦИЯ ошибки ввода измеренных откликов (правка опечатки Y).
    with st.expander("✏️ Исправить измеренные отклики (коррекция ошибок ввода, "
                     "§17.2.1)"):
        st.caption(
            "Опечатку/ошибку ВВОДА отклика можно исправить: правятся только "
            "значения откликов, состав и «№ опыта» сохраняются (И-1 — история не "
            "урезается и не переупорядочивается). После сохранения суррогаты "
            "переобучаются, ветки переоцениваются. Координаты не редактируются.")
        props_corr = list(runner.property_names)
        edit_df = measured_responses_editor_df(runner)
        edited_corr = st.data_editor(
            edit_df, use_container_width=True, hide_index=True,
            disabled=["№ опыта", "источник"], key="camp_correct_editor")
        if st.button("💾 Сохранить исправления откликов", key="camp_correct_save"):
            try:
                n_fixed = 0
                for ridx in range(len(edited_corr)):
                    changes: Dict[str, float] = {}
                    for pn in props_corr:
                        ov = edit_df.iloc[ridx][pn]
                        nv = edited_corr.iloc[ridx][pn]
                        if pd.isna(nv):
                            continue
                        if pd.isna(ov) or not np.isclose(float(ov), float(nv)):
                            changes[pn] = float(nv)
                    if changes:
                        ctrl.correct_measured_point(ridx, changes)
                        n_fixed += 1
                if n_fixed:
                    _invalidate_branch_caches()
                    _flash(f"Исправлено опытов: {n_fixed}. Суррогаты "
                           "переобучены, ветки переоценены (И-1).")
                    st.rerun()
                else:
                    st.info("Изменений не обнаружено — редактировать нечего.")
            except (ValueError, KeyError, IndexError, RuntimeError) as exc:
                st.error(str(exc))

    # P3.1: КОВАРИАТЫ базы (телеметрия прогона) — отдельный редактор:
    # значения можно внести/исправить и ПОСЛЕ фиксации точки (телеметрия
    # снимается на прогоне и часто вносится позже откликов лаборатории).
    if list(getattr(runner, "covariate_names", []) or []):
        with st.expander("📈 Ковариаты базы — телеметрия прогона (P3.1)"):
            st.caption(
                "Столбцы базы, НЕ отклики модели: телеметрия (SME, "
                "Die_Pressure, торк…) в суррогаты не входит и желательности "
                "не несёт, но объясняет условия прогона за точкой. Пустая "
                "ячейка = «не снята» (допустимо); суррогаты при сохранении "
                "НЕ переобучаются — координаты и Y не меняются (И-1).")
            cov_names_base = list(runner.covariate_names)
            cov_df = covariates_editor_df(runner)
            edited_cov = st.data_editor(
                cov_df, use_container_width=True, hide_index=True,
                disabled=["№ опыта", "источник"], key="camp_cov_editor")
            if st.button("💾 Сохранить ковариаты", key="camp_cov_save"):
                try:
                    n_upd = 0
                    for ridx in range(len(edited_cov)):
                        changes: Dict[str, Any] = {}
                        for cn in cov_names_base:
                            ov = cov_df.iloc[ridx][cn]
                            nv = edited_cov.iloc[ridx][cn]
                            if pd.isna(nv) and pd.isna(ov):
                                continue
                            if pd.isna(nv):
                                changes[cn] = None      # стереть значение
                            elif pd.isna(ov) or not np.isclose(float(ov),
                                                               float(nv)):
                                changes[cn] = float(nv)
                        if changes:
                            ctrl.set_point_covariates(ridx, changes)
                            n_upd += 1
                    if n_upd:
                        _flash(f"Ковариаты обновлены у {n_upd} опытов "
                               "(координаты/Y/суррогаты не тронуты, P3.1).")
                        st.rerun()
                    else:
                        st.info("Изменений не обнаружено.")
                except (ValueError, KeyError, IndexError) as exc:
                    st.error(str(exc))

    bids = list(runner.branches)

    if not bids:
        st.info("Стартовый дизайн измерен, суррогаты обучены (общая база = "
                f"{len(runner.points)} точек). Создайте ветку в форме "
                "«➕ Создать ветку вручную» выше (Ш4, §17.5).")
        return


    # --- линза ветки (Тр-3.3): роли В КОНТЕКСТЕ выбранной ветки ----------

    def _branch_label(bid: str) -> str:
        """Показать имя ветки (+id) в селекторе; значение опции остаётся id."""
        br = runner.branches.get(bid)
        return f"{br.name} ({bid})" if br is not None else str(bid)

    bsel = st.selectbox("Ветка (линза контекста — Тр-3.3)", bids,
                        key="camp_branch", format_func=_branch_label)
    # iter65: после измеренного seed самый конкретный контекст — ВЫБРАННАЯ
    # ветка (её линза определяет и роли, и рабочий стол ниже).
    publish_ui_focus("branch", branch=bsel)
    rep = ctrl.role_report(bsel)
    st.caption(f"Линза ветки: **{rep['branch_name']}** (`{bsel}`). Role-tag "
               "валиден ТОЛЬКО в этом контексте; смена ветки меняет теги.")
    st.dataframe(role_table_dataframe(rep), use_container_width=True)

    with st.expander("💰 Почему за ρ есть/нет денег (§16.1)"):
        # P0: MC-оценка (n_mc=128) раньше считалась на КАЖДЫЙ rerun страницы
        # (любой клик по любому виджету) — теперь только по кнопке + кеш
        # (сбрасывается _invalidate_branch_caches после мутаций ветки).
        st.caption("Объяснение денежного канала ρ (MC-оценка) считается по "
                   "кнопке и кешируется до следующего изменения ветки.")
        mkey = f"cache_money_{bsel}"
        if st.button("💰 Объяснить денежный канал ρ",
                     key=f"camp_money_btn_{bsel}"):
            with st.spinner("MC-оценка денежного канала ρ…"):
                try:
                    st.session_state[mkey] = ctrl.money_explanation(
                        bsel, n_candidates=200, n_mc=128, seed=0)["text"]
                except (ValueError, KeyError, RuntimeError) as exc:
                    st.session_state.pop(mkey, None)
                    st.error(f"Не удалось оценить денежный канал: {exc}")
        if st.session_state.get(mkey):
            st.markdown(st.session_state[mkey])

    # --- §16.3: мультицелевой редактор ветки (несколько целей/диапазонов/весов)
    with st.expander("🎯 Редактор целей ветки (§16.3 — мультицель)"):
        st.caption(
            "Ветка — это НАБОР целей: несколько откликов, каждый со своим видом "
            "(min/max/target), диапазоном и весом (снято ограничение «одна ветка — "
            "одна цель»). Цель над откликом делает его роль OPTIMIZED (§16.0); "
            "удаление ПОСЛЕДНЕЙ цели запрещено — ветке нужен объектив. Правки "
            "обратимы (undo, §7) и НЕ трогают измеренную правду (И-1).")
        st.dataframe(goal_editor_dataframe(runner, bsel), use_container_width=True)

        st.markdown("**➕/✏️ Задать или заменить цель над откликом**")
        gc = st.columns([2, 2, 2, 2, 2])
        g_resp = gc[0].selectbox("Цель (отклик)", list(runner.property_names),
                                 key="camp_goal_resp")

        g_kind = gc[1].selectbox("вид", GOAL_KINDS, key="camp_goal_kind")
        g_lo = gc[2].number_input("low", value=0.0, step=0.5, key="camp_goal_lo")
        g_hi = gc[3].number_input("high", value=10.0, step=0.5, key="camp_goal_hi")
        g_w = gc[4].number_input("Значимость цели", min_value=0.01, value=1.0,
                                 step=0.5, key="camp_goal_w", help=_WEIGHT_HELP)

        g_tgt = g_tl = g_th = g_thr = g_noise = None
        if g_kind == "target":
            g_tgt = st.number_input(
                "target (только для вида target; low<target<high)",
                value=5.0, step=0.5, key="camp_goal_tgt")
        elif g_kind == GOAL_KIND_RANGE:
            prc = st.columns(2)
            g_tl = prc[0].number_input(
                "плато от (target_low)", value=60.0, step=0.5,
                key="camp_goal_tlo", help=_RANGE_HELP)
            g_th = prc[1].number_input(
                "плато до (target_high)", value=70.0, step=0.5,
                key="camp_goal_thi", help=_RANGE_HELP)
        elif g_kind in _THRESHOLD_KINDS:
            prc = st.columns(2)
            g_thr = prc[0].number_input(
                "порог", value=10.0, step=0.5, key="camp_goal_thr")
            g_noise = prc[1].number_input(
                "СКО шума измерения", min_value=1e-9, value=0.5, step=0.1,
                key="camp_goal_noise", help=_NOISE_SD_HELP)
        if st.button("💾 Задать / заменить цель", key="camp_goal_set"):
            try:
                # iter43/P2.2: сборка ЕДИНЫМ билдером (пороги/плато включ.)
                spec = build_goal_spec(
                    g_kind, low=float(g_lo), high=float(g_hi),
                    target=(float(g_tgt) if g_tgt is not None else None),
                    weight=float(g_w),
                    threshold=(float(g_thr) if g_thr is not None else None),
                    noise_sd=(float(g_noise) if g_noise is not None else None),
                    target_low=(float(g_tl) if g_tl is not None else None),
                    target_high=(float(g_th) if g_th is not None else None))
                res = ctrl.set_desirability(bsel, g_resp, spec)
                shift = res["recommendation_shift"]
                _invalidate_branch_caches()
                _flash(
                    f"Цель «{g_resp}» ({g_kind}) задана; d_best "
                    f"{res['d_best_before']:.3f} → {res['d_best_after']:.3f}"
                    + (f"; рекомендация x* сместилась на ≈{shift:.3f}."
                       if shift is not None else "; x* пересчитана."))
                st.rerun()
            except (ValueError, KeyError) as exc:
                st.error(str(exc))

        goals_now = list(runner.branches[bsel].goal or {})
        if goals_now:
            st.markdown("**⚖️ Значимость целей (относительный приоритет в "
                        "геом-среднем d_i)**")
            wcols = st.columns(len(goals_now))
            new_w: Dict[str, float] = {}
            for i, resp in enumerate(goals_now):
                cur_w = float(runner.branches[bsel].goal[resp].weight)
                new_w[resp] = wcols[i].number_input(
                    f"Значимость «{resp}»", min_value=0.01, value=cur_w,
                    step=0.5, key=f"camp_goal_w_{resp}", help=_WEIGHT_HELP)

            if st.button("⚖️ Применить веса", key="camp_goal_weights"):
                try:
                    res = ctrl.set_weights(
                        bsel, {r: float(v) for r, v in new_w.items()})
                    _invalidate_branch_caches()
                    _flash(f"Веса обновлены; d_best → "
                           f"{res['d_best_after']:.3f} (re-score, И-1).")
                    st.rerun()
                except (ValueError, KeyError) as exc:
                    st.error(str(exc))

            st.markdown("**🗑 Удалить цель** (последняя — отказ)")
            dc = st.columns([3, 2])
            del_resp = dc[0].selectbox("цель на удаление", goals_now,
                                       key="camp_goal_del_sel")
            if dc[1].button("🗑 Удалить цель", key="camp_goal_del"):
                try:
                    ctrl.delete_goal(bsel, del_resp)
                    _invalidate_branch_caches()
                    _flash(f"Цель «{del_resp}» удалена (роль → REFERENCE).")
                    st.rerun()
                except (ValueError, KeyError) as exc:
                    st.error(str(exc))

    # --- §17.5 (Ш5): рабочий стол ветки на РУЧНОМ цикле §17.2 (не авто-оракул)
    render_workbench(ctrl, bsel)

    # --- смена роли ρ (§5) ----------------------------------------------

    st.markdown("**🔁 Сменить роль ρ (§5) — переключает денежный канал**")
    rho = _rho_of(runner, bsel)
    if rho is None:
        st.caption("У ветки нет ценовой ноги — переключать роль ρ нечем.")
    else:
        cur = runner.response_role(bsel, rho)
        target = ROLE_PRICE_INPUT if cur == ROLE_OPTIMIZED else ROLE_OPTIMIZED
        st.caption(f"ρ = «{rho}»: текущая роль **{cur}** → станет **{target}** "
                   f"(канал {'ALIVE' if target == ROLE_PRICE_INPUT else 'ZEROED'}).")
        if target == ROLE_OPTIMIZED:
            cc = st.columns(3)
            cc[0].selectbox("вид цели", ["min", "max", "target"],
                            key="camp_sw_kind")
            cc[1].number_input("low", value=0.5, key="camp_sw_lo")
            cc[2].number_input("high", value=1.5, key="camp_sw_hi")
        if st.button(f"Сменить роль ρ → {target}", key="camp_do_switch"):
            try:
                spec = None
                if target == ROLE_OPTIMIZED:
                    kind = st.session_state.get("camp_sw_kind", "min")
                    lo = float(st.session_state.get("camp_sw_lo", 0.5))
                    hi = float(st.session_state.get("camp_sw_hi", 1.5))
                    tgt = (lo + hi) / 2.0 if kind == "target" else None
                    spec = DesirabilitySpec(kind, low=lo, high=hi, target=tgt)
                res = ctrl.switch_role(bsel, rho, target, spec=spec)
                shift = res["recommendation_shift"]
                _invalidate_branch_caches()
                _flash(
                    f"Роль ρ: {res['role_before']} → {res['role_after']}; "
                    f"канал занулён = {res['price_channel_suppressed']}; "
                    + (f"рекомендация x* сместилась на ≈{shift:.3f}."
                       if shift is not None else "рекомендация x* пересчитана."))
                st.rerun()
            except (ValueError, KeyError) as exc:
                st.error(str(exc))

    # --- undo (§7) + прогон раунда (запечатывает дно) -------------------
    cu = st.columns(2)
    if ctrl.can_undo():
        if cu[0].button("↩️ Undo последней настройки (§7)", key="camp_undo"):
            u = ctrl.undo()
            _invalidate_branch_caches()
            _flash(f"Откат «{u['undone']}» ветки {u['branch_id']} "
                   f"(undo_available={u['undo_available']}).", kind="info")
            st.rerun()
    else:
        cu[0].caption("Undo пуст: дно — последний снятый раунд (И-1).")
    # P0: авто-раунд зовёт оракул раннера; в РУЧНОЙ кампании (ManualOracle) это
    # записало бы в реальную базу СИНТЕТИЧЕСКИЕ Y — кнопка скрыта (A0.6).
    if is_manual_campaign(runner):
        cu[1].caption("Авто-прогон раунда недоступен: отклики этого проекта "
                      "вносятся ВРУЧНУЮ — используйте «🛠 Рабочий стол ветки» "
                      "выше (предложить → внести Y → долить).")
    elif cu[1].button("▶ Прогнать раунд ветки (демо-оракул; запечатает undo, "
                      "Тр-7.2/7.3)", key="camp_run_round"):
        try:
            ctrl.run_round(bsel, n_points=2, explore_frac=0.2, n_candidates=150)
            _invalidate_branch_caches()
            _flash("Раунд снят: новые измерения в общей базе, дно undo "
                   "обновлено.")
            st.rerun()
        except (ValueError, KeyError, RuntimeError) as exc:
            st.error(str(exc))

    # --- spawn ветки (§8) с наследованием ролей -------------------------
    st.markdown("**🌱 Spawn ветки (§8) — наследование ролей + review-сводка**")
    # NB: локальная переменная НЕ «cs» — иначе затеняется модуль campaign_state.
    spc = st.columns([2, 2, 2])
    parent = spc[0].selectbox("Родитель", bids, key="camp_spawn_parent",
                              format_func=_branch_label)
    cname = spc[1].text_input("Имя ребёнка", value="child", key="camp_spawn_name")
    over = spc[2].checkbox("новая цель над ρ (перебьёт роль, Тр-8.1в)",
                           key="camp_spawn_over")
    prho = _rho_of(runner, parent)
    new_goals = ({prho: DesirabilitySpec("min", low=0.5, high=1.5)}
                 if over and prho else None)

    if st.button("👁 Предпросмотр наследования (без создания)",
                 key="camp_spawn_preview"):
        rev = ctrl.preview_spawn(parent, new_goals=new_goals)
        st.dataframe(spawn_review_dataframe(rev), use_container_width=True)
        if rev["any_role_changed_by_objective"]:
            st.warning("Объектив ветки перебил унаследованную роль ρ — канал "
                       "цены будет занулён (И-5).")
    if st.button("🌱 Создать ветку (spawn)", key="camp_do_spawn"):
        try:
            cid = f"{parent}_child{len(bids)}"
            res = ctrl.spawn_branch(parent, cname, child_id=cid,
                                    new_goals=new_goals)
            _invalidate_branch_caches()
            # P0: review-сводка — в session_state (переживает st.rerun); rerun
            # нужен, чтобы селектор веток/таблицы сразу увидели ребёнка.
            st.session_state["camp_spawn_last"] = res["review"]
            _flash(
                f"Ветка «{res['child_name']}» создана (`{res['child_id']}`); "
                f"канал ρ занулён = {res['price_channel_suppressed']}.")
            st.rerun()
        except (ValueError, KeyError) as exc:
            st.error(str(exc))
    if st.session_state.get("camp_spawn_last") is not None:
        st.caption("Review-сводка наследования ролей последнего spawn:")
        st.dataframe(spawn_review_dataframe(st.session_state["camp_spawn_last"]),
                     use_container_width=True)
